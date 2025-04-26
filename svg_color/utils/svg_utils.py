# src/svg_color_tool/svg_utils.py

import math
import logging
from typing import List, Tuple, Optional, Dict, Iterator

import xml.etree.ElementTree as ET

from svg_color.config import _SVG_NS, _CMD_PARAM_COUNTS, _PATH_PARAM_REGEX
from svg_color.utils.svg_utils import *


logger = logging.getLogger(__name__)

# --- SVG Utilities ---

def parse_svg_content(svg_content):
    
    try:
        # with open(svg_path, "r") as f:
        #     svg_content = f.read()

        parser = ET.XMLParser(encoding='utf-8') # Tolerate minor errors
        svg_bytes = svg_content.encode('utf-8') if isinstance(svg_content, str) else svg_content
        if not isinstance(svg_bytes, bytes):
                raise TypeError("svg_content must be convertible to bytes.")

        root = ET.fromstring(svg_bytes, parser=parser)
        if root is None: # Check if parser failed despite recover=True
                logger.error("Failed to parse SVG content: Parser returned None.")
                return None
        tree = ET.ElementTree(root)
        logger.info("SVG content parsed successfully.")
        return root, tree
    except ET.ParseError as parse_err:
        logger.error(f"XML Parsing Error in SVG: {parse_err}")
        return None # Critical failure
    except Exception as e:
            logger.error(f"Unexpected error during SVG parsing: {e}", exc_info=True)
            return None # Critical failure

    

# ET element ultilities
def update_or_create_style_tag(root: ET.Element, class_styles: Dict[str, str]):
    """
    Finds an existing CSS <style> tag or creates a new one in the SVG root,
    then populates it with the provided CSS class definitions.

    Args:
        root: The root Element of the SVG tree.
        class_styles: A dictionary mapping CSS class names (e.g., "st0")
                        to their fill hex color values (e.g., "#ff0000").
    """
    logger.debug(f"Updating/Creating CSS <style> tag with {len(class_styles)} rules.")

    # Generate the CSS rule strings, sorted by class name for consistency
    css_rules = [
        f"    .{cls} {{ fill: {hex_val}; }}"
        for cls, hex_val in sorted(class_styles.items())
    ]
    
    # with open("output.json", "w") as f:
    #     json_string = json.dumps(class_styles, indent=4, ensure_ascii=False)
    #     f.write(json_string)
    

    # Join rules with newlines and add surrounding newlines for readability
    css_content = "\n" + "\n".join(css_rules) + "\n"

    # --- Find or Create the <style> Element ---
    # Try to find an existing style tag, preferably one with type="text/css"
    style_tag = root.find(f".//{_SVG_NS}style[@type='text/css']")
    
    if style_tag is None:
        # If not found, look for any <style> tag as a fallback
        style_tag = root.find(f".//{_SVG_NS}style")

    if style_tag is None:
        # If no <style> tag exists at all, create a new one
        logger.info("No existing <style> tag found. Creating a new one.")
        style_tag = ET.Element(f"{_SVG_NS}style", {"type": "text/css"})
        style_tag.text = css_content
        
        # Insert the new tag strategically: after <defs> if it exists, otherwise at the beginning
        defs_tag = root.find(f".//{_SVG_NS}defs")
        insert_index = 0
        if defs_tag is not None:
            try:
                # Find the index of <defs> and insert after it
                insert_index = list(root).index(defs_tag) + 1
            except ValueError:
                logger.warning("<defs> tag found but could not determine its index. Inserting <style> at beginning.")
        root.insert(insert_index, style_tag)
    else:
        # If an existing <style> tag was found, update its content and type
        logger.info("Updating existing <style> tag content.")
        style_tag.set("type", "text/css") # Ensure the type attribute is correct
        style_tag.text = css_content     # Replace the existing text content
    

def iter_params(params_str: str, chunk_size: int) -> Iterator[List[float]]:
    """
    Yields chunks of numerical parameters extracted from a string.

    Parses a string containing space/comma-separated numbers and yields
    them in lists of the specified `chunk_size`. Handles potential
    parsing errors and logs warnings for invalid parameter counts.

    Args:
        params_str: The string containing parameters (e.g., "10 20 30 40").
        chunk_size: The expected number of parameters per chunk (e.g., 2 for coordinates).
                    If 0, expects no parameters.

    Yields:
        List[float]: A list containing `chunk_size` parameters.

    Returns:
        An empty iterator if parsing fails or parameter counts are invalid.
    """
    try:
        # Use precompiled regex for potentially better performance
        all_params = [float(p) for p in _PATH_PARAM_REGEX.findall(params_str)]
        total_params = len(all_params)

        # Handle commands expecting zero parameters (like 'Z')
        if chunk_size == 0:
            if total_params == 0:
                yield [] # Yield an empty list for Z command consistency
            else:
                logger.warning(f"Parameters found ('{params_str[:30]}...') for command expecting none. Ignoring.")
            return # Stop iteration for Z

        # Validate total parameter count for commands expecting parameters
        if total_params % chunk_size != 0:
            logger.warning(
                f"Parameter count ({total_params}) in '{params_str[:30]}...' "
                f"is not a multiple of expected chunk size {chunk_size}. Skipping command."
            )
            return # Stop iteration for this command

        # Yield parameters in chunks
        for i in range(0, total_params, chunk_size):
            yield all_params[i : i + chunk_size]

    except ValueError:
        logger.error(f"Error parsing numeric parameters in '{params_str[:50]}...'")
        return # Stop iteration on error

def update_point(command: str,params: List[float],current_point: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculates the new absolute point after applying a single SVG path command segment.

    Takes a command (e.g., 'L', 'm', 'h'), its parameters for *one* segment,
    and the current point, returning the new absolute coordinates. Handles
    both absolute and relative commands.

    Args:
        command: The SVG path command character.
        params: Numerical parameters for *this segment* of the command.
        current_point: The current (x, y) coordinates.

    Returns:
        The new absolute (x, y) coordinates. Returns `current_point` if
        the command or parameters are invalid for updating the point.
    """
    cmd_upper = command.upper()
    is_relative = command.islower()
    cx, cy = current_point

    try:
        # Commands defining a target point (x, y)
        if cmd_upper in ('M', 'L', 'T'):
            if len(params) != 2: return current_point # Invalid params for this command
            px, py = params
            return (cx + px, cy + py) if is_relative else (px, py)

        # Horizontal line command
        elif cmd_upper == 'H':
            if len(params) != 1: return current_point
            px = params[0]
            return (cx + px, cy) if is_relative else (px, cy) # Y remains the same

        # Vertical line command
        elif cmd_upper == 'V':
            if len(params) != 1: return current_point
            py = params[0]
            return (cx, cy + py) if is_relative else (cx, py) # X remains the same

        # Curve and Arc commands - we only track the final endpoint
        elif cmd_upper in ('C', 'S', 'Q', 'A'):
            expected_count = _CMD_PARAM_COUNTS.get(cmd_upper)
            # Check if params match expected count *for this segment*
            if expected_count is None or len(params) != expected_count:
                return current_point
            # Endpoint coordinates are always the last two parameters
            px_idx, py_idx = expected_count - 2, expected_count - 1
            px, py = params[px_idx], params[py_idx]
            return (cx + px, cy + py) if is_relative else (px, py)

        # ClosePath command - does not change the *final* point itself
        elif cmd_upper == 'Z':
            return current_point # Position determined by start_of_subpath in the caller

        else: # Should not happen if regex is correct, but handles unknown commands
            logger.warning(f"Unsupported command '{command}' during point update.")
            return current_point

    except (IndexError, TypeError) as e:
            logger.error(f"Error processing parameters {params} for command '{command}': {e}")
            return current_point # Return current point on error
    