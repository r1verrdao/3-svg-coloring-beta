import math
import logging
import numpy as np
import cv2
from typing import Tuple, Optional
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from svg_color.config import *

logger = logging.getLogger(__name__)

def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """Calculates the Euclidean distance between two RGB colors."""
    if not (isinstance(c1, tuple) and len(c1) == 3 and isinstance(c2, tuple) and len(c2) == 3):
        logger.error(f"Invalid color format for distance calculation: c1={c1}, c2={c2}")
        return float('inf') # Return infinity for invalid input
    # Use float for calculations to avoid potential overflow with large differences
    dr = float(c1[0]) - c2[0]
    dg = float(c1[1]) - c2[1]
    db = float(c1[2]) - c2[2]
    return math.sqrt(dr*dr + dg*dg + db*db)

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Converts an RGB tuple to a HEX color string."""
    if not (isinstance(rgb, tuple) and len(rgb) == 3):
        logger.error(f"Invalid RGB tuple for hex conversion: {rgb}")
        return '#000000' # Return black as default for invalid input
    # Clamp values to 0-255 and convert to int
    r, g, b = [max(0, min(255, int(c))) for c in rgb]
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def sample_color_at_point(img_data: np.ndarray, point: Tuple[float, float]) -> Optional[Tuple[int, int, int]]:
    """
    Samples the average RGB color from a small patch in the image around a given point.

    Args:
        img_data: The image data (expects RGB format as NumPy array).
        point: The (x, y) coordinates (can be float) where sampling should occur.

    Returns:
        An (R, G, B) integer tuple representing the average color, or None if
        sampling fails (e.g., point outside image, invalid image data, cv2 missing).
    """
    if img_data is None or img_data.size == 0 or point is None:
        logger.warning("Cannot sample color: Invalid image data or point.")
        return None
    if cv2 is None:
            logger.error("Cannot sample color: OpenCV (cv2) is not available.")
            return None

    img_h, img_w = img_data.shape[:2]
    x_f, y_f = point # Float coordinates from representative point

    # Convert float coordinates to integer pixel indices, clamping to image bounds
    x_coord = max(0, min(img_w - 1, int(round(x_f))))
    y_coord = max(0, min(img_h - 1, int(round(y_f))))

    # Define the 3x3 patch boundaries around the center pixel
    half_patch = _COLOR_SAMPLING_PATCH_SIZE // 2
    y_start = max(0, y_coord - half_patch)
    y_end = min(img_h, y_coord + half_patch + 1)
    x_start = max(0, x_coord - half_patch)
    x_end = min(img_w, x_coord + half_patch + 1)

    # Extract the patch from the image data
    patch = img_data[y_start:y_end, x_start:x_end]

    # Handle cases where the patch might be empty (e.g., point exactly on edge)
    if patch.size == 0:
            logger.warning(f"Sampling patch was empty at integer coords ({x_coord},{y_coord}). "
                        f"Check representative point calculation or image dimensions.")
            # Fallback to the single pixel at the calculated integer coordinates
            try:
                pixel_color = img_data[y_coord, x_coord]
                # Ensure 3 channels (assuming RGB input)
                if pixel_color.shape[0] >= 3:
                    return tuple(map(int, pixel_color[:3]))
                else: # Grayscale or other format - cannot return RGB
                    logger.warning(f"Cannot fallback to single pixel: Unexpected shape {pixel_color.shape}")
                    return None
            except IndexError:
                logger.error(f"Fallback pixel coordinates ({x_coord},{y_coord}) are out of bounds.")
                return None

    # Calculate the mean color of the patch (expecting RGB)
    try:
        # Ensure we have 3 dimensions (for channels) before slicing
        if patch.ndim != 3 or patch.shape[2] < 3:
                logger.warning(f"Patch has unexpected shape {patch.shape} at ({x_coord},{y_coord}). Cannot calculate mean RGB.")
                # Try single pixel fallback again
                pixel_color = img_data[y_coord, x_coord]
                if pixel_color.shape[0] >= 3: return tuple(map(int, pixel_color[:3]))
                else: return None

        # Calculate mean across the patch (height, width dimensions), take first 3 channels
        mean_color_float = np.mean(patch[:, :, :3], axis=(0, 1))
        # Convert mean float values to integer RGB tuple
        sampled_color = tuple(map(int, mean_color_float))
        return sampled_color
    except (IndexError, ValueError, Exception) as sample_err:
        # Catch potential errors during NumPy mean calculation
        logger.error(f"Error calculating mean color for patch at ({x_coord},{y_coord}): {sample_err}")
        return None

def get_dominant_color(img, points, method='kmeans', k=1):
    """
    Trích xuất màu chủ đạo từ ảnh PNG cho một vùng được xác định bởi points.
    - img: ảnh PNG (numpy array)
    - points: danh sách tọa độ của path trong SVG
    - method: phương pháp trích xuất ('mean', 'kmeans', 'most_common')
    """
    if not points:
        return (0, 0, 0)

    min_x = max(0, int(min(p[0] for p in points)))
    min_y = max(0, int(min(p[1] for p in points)))
    max_x = min(img.shape[1] - 1, int(max(p[0] for p in points)))
    max_y = min(img.shape[0] - 1, int(max(p[1] for p in points)))

    if min_x >= max_x or min_y >= max_y:
        return (0, 0, 0)

    roi = img[min_y:max_y, min_x:max_x]
    pixels = roi.reshape(-1, 3)  # Thêm dòng này để định nghĩa pixels

    # TODO: Chỉ giữ lại method Kmeans
    if method == 'mean':
        mean_color = np.mean(roi, axis=(0, 1))
        return tuple(map(int, mean_color))
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=1, random_state=0, n_init=10, algorithm="elkan")
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[0]
        return tuple(map(int, dominant_color))
    elif method == 'dbscan':
        db = DBSCAN(eps=0.5, min_samples=1).fit(pixels)
        labels = db.labels_  # Nhãn của từng pixel (-1 là nhiễu)
        unique_labels = set(labels)
        if -1 in unique_labels and len(unique_labels) == 1:
            # Chỉ có nhiễu, trả về màu trung bình
            mean_color = np.mean(pixels, axis=0)
            return tuple(map(int, mean_color))
        else:
            # Tìm cụm lớn nhất
            largest_cluster_label = max(unique_labels, key=lambda x: np.sum(labels == x) if x != -1 else 0)
            if largest_cluster_label == -1:
                # Không có cụm, trả về màu trung bình
                mean_color = np.mean(pixels, axis=0)
                return tuple(map(int, mean_color))
            else:
                # Tính màu trung bình của cụm lớn nhất
                cluster_pixels = pixels[labels == largest_cluster_label]
                mean_color = np.mean(cluster_pixels, axis=0)
                return tuple(map(int, mean_color))
    elif method == 'meanshift':
        # Ước lượng bandwidth
        bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=500)
        # Áp dụng MeanShift
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(pixels)
        labels = ms.labels_  # Nhãn của từng pixel
        cluster_centers = ms.cluster_centers_  # Tâm của các cụm
        # Tìm cụm lớn nhất
        labels_unique = np.unique(labels)
        largest_cluster_label = labels_unique[np.argmax([np.sum(labels == label) for label in labels_unique])]
        dominant_color = cluster_centers[largest_cluster_label]
        return tuple(map(int, dominant_color))
    else:  # most_common
        pixels_list = [tuple(map(int, pixel)) for pixel in pixels]
        counter = Counter(pixels_list)
        dominant_color = counter.most_common(1)[0][0]
        return dominant_color