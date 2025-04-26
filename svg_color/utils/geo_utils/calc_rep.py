import logging
import math
import heapq
from functools import cached_property
from typing import List, Tuple, Optional, Sequence # Sequence is more general than List here

# --- Geometry Dependencies ---
# Chỉ import những gì cần thiết trực tiếp

from shapely.geometry import Point, Polygon
from shapely.errors import ShapelyError # Import specific error
from .bbox import get_bounding_box



logger = logging.getLogger(__name__)



def shapely_calculate_representative_point(coords: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Calculates a point guaranteed to be inside the polygon defined by coordinates.

    Uses Shapely's `representative_point()` if available for robustness,
    otherwise falls back to the center of the bounding box.

    Args:
        coords: A list of (x, y) tuples defining the polygon vertices.

    Returns:
        An (x, y) tuple for the representative point, or None if calculation fails.
    """
    if not coords or len(coords) < 3:
        logger.debug("Cannot calculate representative point: requires at least 3 coordinates.")
        return None
    
    # --- Attempt 1: Use Shapely (preferred) ---
    # if Polygon and Point and ShapelyError: # Check if Shapely components were imported
    try:
        polygon = Polygon(coords)
        # Try to fix simple invalid geometries (e.g., self-touching)
        if not polygon.is_valid:
            polygon_fixed = polygon.buffer(0)
            # Check if fixing worked and resulted in a valid polygon
            if polygon_fixed.is_valid and not polygon_fixed.is_empty:
                polygon = polygon_fixed
            else:
                    logger.debug(f"Shapely polygon invalid and buffer(0) failed for coords: {coords[:3]}...")
                    # Fall through to bounding box fallback

        # If polygon is valid (or was fixed), use representative_point
        if polygon.is_valid and not polygon.is_empty:
                rep_point = polygon.representative_point()
                logger.debug("Calculated representative point using Shapely.")
                return rep_point.x, rep_point.y

    except (ShapelyError, ValueError, TypeError, Exception) as shape_err:
        # Catch potential errors during Shapely processing
        logger.warning(f"Shapely error calculating representative point ({shape_err}), falling back to bbox.")
        # Fall through to bounding box fallback


def bbox_calculate_representative_point(coords: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not coords or len(coords) < 3:
        logger.debug("Cannot calculate representative point: requires at least 3 coordinates.")
        return None
    
    logger.debug("Attempting to calculate bounding box center.")
    bbox = get_bounding_box(coords)
    # Ensure bbox is valid and has non-zero width and height
    if bbox and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        logger.debug("Using bounding box center as representative point.")
        return center_x, center_y
    else:
        # Log failure if both methods fail
        logger.warning(f"Cannot calculate representative point using Shapely or BBox center for coords: {coords[:3]}...")
        return None
    




def mixed_calculate_representative_point(coords: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Calculates a point guaranteed to be inside the polygon defined by coordinates.

    Uses Shapely's `representative_point()` if available for robustness,
    otherwise falls back to the center of the bounding box.

    Args:
        coords: A list of (x, y) tuples defining the polygon vertices.

    Returns:
        An (x, y) tuple for the representative point, or None if calculation fails.
    """
    if not coords or len(coords) < 3:
        logger.debug("Cannot calculate representative point: requires at least 3 coordinates.")
        return None

    # --- Attempt 1: Use Shapely (preferred) ---
    # if Polygon and Point and ShapelyError: # Check if Shapely components were imported
    result = shapely_calculate_representative_point(coords)

    # --- Attempt 2: Fallback to Bounding Box Center ---
    if result is None:
        return bbox_calculate_representative_point(coords)