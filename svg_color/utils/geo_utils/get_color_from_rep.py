import logging
import math
import heapq
from functools import cached_property
from typing import List, Tuple, Optional, Sequence

from shapely.geometry import Point, Polygon
from shapely.errors import ShapelyError

from svg_color.config import *
import numpy as np

import cv2



def sample_color_at_point(img_data: np.ndarray, point: Tuple[float, float]) -> Optional[Tuple[int, int, int]]:
        # (Implementation from previous refactoring)
        if img_data is None or img_data.size == 0 or point is None or cv2 is None: return None
        # ... (sampling logic) ...
        img_h, img_w = img_data.shape[:2]
        x_f, y_f = point
        x_coord = max(0, min(img_w - 1, int(round(x_f))))
        y_coord = max(0, min(img_h - 1, int(round(y_f))))
        half_patch = _COLOR_SAMPLING_PATCH_SIZE // 2
        y_start, y_end = max(0, y_coord - half_patch), min(img_h, y_coord + half_patch + 1)
        x_start, x_end = max(0, x_coord - half_patch), min(img_w, x_coord + half_patch + 1)
        patch = img_data[y_start:y_end, x_start:x_end]
        if patch.size == 0: return None # Or handle single pixel fallback
        try:
             num_channels = img_data.shape[2] if img_data.ndim == 3 else 1
             if num_channels >= 3: mean_color_float = np.mean(patch[:,:,:3], axis=(0, 1))
             elif num_channels == 1: mean_gray = np.mean(patch); mean_color_float = (mean_gray, mean_gray, mean_gray)
             else: return None
             return tuple(map(int, mean_color_float))
        except Exception: return None