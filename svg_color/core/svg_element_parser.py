# --- SVG Path Parsing ---
import re
import time
import numpy as np
import xml.etree.ElementTree as ET
import math
import json 
import logging
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any, Union, Iterator

# Use relative import assuming 'utils' is in the parent directory or sibling 'core'
# Adjust if your structure is different (e.g., from ..utils import ...)
# from svg_color.utils.svg_utils import color_distance, rgb_to_hex

from svg_color.utils import svg_utils
from svg_color.config import *
import cv2


logger = logging.getLogger(__name__)

class SvgElementParser():
    def _parse_svg_path(self, path_d: str) -> List[Tuple[float, float]]:
        """
        Parses an SVG path 'd' attribute string into a list of vertex coordinates.

        Handles Move (M, m), Line (L, l, H, h, V, v), Curve (C, c, S, s, Q, q, T, t),
        Arc (A, a), and ClosePath (Z, z) commands. Curves and arcs are simplified
        by only adding their final endpoint to the list.

        Args:
            path_d: The path data string from the 'd' attribute.

        Returns:
            A list of (x, y) tuples representing the path vertices. Returns an
            empty list if the path string is empty or a critical parsing error occurs.
        """
        if not path_d or not isinstance(path_d, str):
            return []

        points: List[Tuple[float, float]] = []
        current_pos = (0.0, 0.0)
        start_of_subpath = (0.0, 0.0)

        try:
            # Iterate through commands found in the path string
            for cmd_idx, match in enumerate(_PATH_COMMAND_REGEX.finditer(path_d)):
                command = match.group(1)
                params_str = match.group(2).strip()
                cmd_upper = command.upper()

                # Get expected parameters per segment for this command type
                segment_param_count = _CMD_PARAM_COUNTS.get(cmd_upper, 0)

                # Process parameters in chunks using the helper generator
                param_iterator = svg_utils.iter_params(params_str, segment_param_count)

                is_first_segment = True # Track first segment for Move command logic
                for params_segment in param_iterator:

                    # Handle Move command (sets current position and subpath start)
                    if cmd_upper == 'M':
                        target_pos = svg_utils.update_point(command, params_segment, current_pos)
                        # First M/m segment defines the new current position and subpath start
                        if is_first_segment:
                            current_pos = target_pos
                            start_of_subpath = current_pos
                            points.append(current_pos)
                            if cmd_idx == 0: # Very first M/m in the whole path
                                pass # start_of_subpath already set
                        else:
                            # Subsequent pairs in M/m are implicit Lineto/lineto
                            effective_command = 'l' if command == 'm' else 'L'
                            current_pos = svg_utils.update_point(effective_command, params_segment, current_pos)
                            points.append(current_pos)

                    # Handle ClosePath command (connects back to subpath start)
                    elif cmd_upper == 'Z':
                        if points and current_pos != start_of_subpath:
                            points.append(start_of_subpath) # Add point only if not already there
                        current_pos = start_of_subpath # Next command starts from here

                    # Handle all other commands (Line, Curve, Arc)
                    else:
                        # Update current position based on the command and add the new point
                        current_pos = svg_utils.update_point(command, params_segment, current_pos)
                        points.append(current_pos)

                    is_first_segment = False # No longer the first segment after one iteration

        except Exception as e:
                # Catch unexpected errors during the main loop or helpers
                logger.error(f"Unexpected error during SVG path parsing for '{path_d[:50]}...': {e}", exc_info=True)
                return []

        logger.debug(f"Parsed path '{path_d[:50]}...' into {len(points)} points.")
        return points


    def parse(self, elem: ET.Element) -> Optional[List[Tuple[float, float]]]:
        """
        Extracts vertex coordinates for supported SVG elements (path, polygon, rect, etc.).

        Args:
            elem: The ElementTree Element object.

        Returns:
            A list of (x, y) coordinate tuples, or None if the element type is
            unsupported or attributes are invalid/missing.
        """
        # tag_name = ET.QName(elem.tag).localname
        tag_name = elem.tag.split('}')[-1]
        coords: Optional[List[Tuple[float, float]]] = None

        try:
            # --- Handle different shape types ---
            if tag_name == "polygon":
                points_str = elem.get("points")
                if points_str:
                    # Robust parsing of points string
                    raw_points = re.split(r'[ ,\t\n]+', points_str.strip())
                    nums = [float(p) for p in raw_points if p] # Filter empty, convert
                    if len(nums) >= 2 and len(nums) % 2 == 0:
                        coords = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]

            elif tag_name == "path":
                d_attr = elem.get("d")
                if d_attr:
                    coords = self._parse_svg_path(d_attr) # Use the refactored parser

            elif tag_name == "rect":
                x = float(elem.get('x', 0))
                y = float(elem.get('y', 0))
                width = float(elem.get('width', 0))
                height = float(elem.get('height', 0))
                if width > 0 and height > 0:
                    # Define corners in clockwise order, closing the loop
                    coords = [(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)]

            elif tag_name == "circle":
                cx = float(elem.get('cx', 0))
                cy = float(elem.get('cy', 0))
                r = float(elem.get('r', 0))
                if r > 0:
                    # Approximate circle with a polygon
                    coords = []
                    for i in range(_CIRCLE_APPROX_SEGMENTS):
                        angle = 2 * math.pi * i / _CIRCLE_APPROX_SEGMENTS
                        coords.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
                    coords.append(coords[0]) # Close the loop

            elif tag_name == "ellipse":
                # Basic ellipse handling (ignores rotation via 'transform')
                cx = float(elem.get('cx', 0))
                cy = float(elem.get('cy', 0))
                rx = float(elem.get('rx', 0))
                ry = float(elem.get('ry', 0))
                if rx > 0 and ry > 0:
                    coords = []
                    for i in range(_CIRCLE_APPROX_SEGMENTS):
                            angle = 2 * math.pi * i / _CIRCLE_APPROX_SEGMENTS
                            coords.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
                    coords.append(coords[0]) # Close the loop

            # Add other potential shapes like 'line', 'polyline' here if needed

        except (ValueError, TypeError) as e:
                logger.warning(f"Invalid numeric attribute for element {tag_name}: {elem.attrib}. Error: {e}")
                return None # Failed to extract valid coordinates

        # --- Validation and Cleanup ---
        if coords and len(coords) >= 3:
            # Remove duplicate consecutive points and ensure at least 3 unique points remain
            # Using dict.fromkeys preserves order while removing duplicates efficiently
            unique_coords = list(dict.fromkeys(p for p in coords if p is not None))
            if len(unique_coords) >= 3:
                    return unique_coords # Return list of unique coordinates

        # Log if insufficient points were generated
        logger.debug(f"Insufficient unique coordinates ({len(coords) if coords else 0}) for element {tag_name}. Skipping.")
        return None