# Helper class for cells in the priority queue
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def get_bounding_box(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    """Calculates the bounding box (min_x, min_y, max_x, max_y) from a list of points."""
    if not points:
        logger.debug("Cannot get bounding box for empty points list.")
        return None
    try:
        # Avoid modifying the original list if it's mutable
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        # Check if lists are empty after extraction (unlikely if points is not empty, but safe)
        if not x_vals or not y_vals:
             logger.warning("Extracted empty x_vals or y_vals from non-empty points list.")
             return None
        return (min(x_vals), min(y_vals), max(x_vals), max(y_vals))
    except (ValueError, TypeError) as e:
        # Catch potential errors if points contain non-numeric data
        logger.warning(f"Could not calculate bounding box due to invalid data in points: {points[:5]}... Error: {e}")
        return None
