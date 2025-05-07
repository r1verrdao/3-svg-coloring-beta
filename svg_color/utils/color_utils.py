import math
import logging
import numpy as np
import cv2
from typing import Tuple, Optional
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth, MiniBatchKMeans
from svg_color.config import *
import faiss
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import time
import concurrent.futures

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

def get_dominant_color(img, points, method='kmeans', k=9):
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
    print(f'Fragment shape: {pixels.shape}')


    if method == 'kmeans':
        # using sklearn
        # kmeans = KMeans(n_clusters=10, random_state=42, n_init=10, algorithm="elkan")
        # kmeans.fit(pixels)
        # dominant_color = kmeans.cluster_centers_[0]

        # kmeans = MiniBatchKMeans(n_clusters=10, random_state=42, n_init=10, batch_size=1000)
        # kmeans.fit(pixels)

        # using faiss
        optimal_k = find_optimal_k(pixels)
        # if pixels.shape[0] < k:
        #     k = pixels.shape[0]
        kmeans = faiss.Kmeans(k=optimal_k, d=pixels.shape[1], min_points_per_centroid=1)
        kmeans.train(pixels)

        # chọn màu chủ đạo
        # # Cách 1: lấy ngẫu nhiên 1 màu, có thể sẽ dẫn tới sai màu
        # dominant_color = kmeans.centroids[0] 

        # # Cách 2: another approach: weighted cluster size
        # # code for using sklearn
        # # Get cluster assignments
        # assignments = kmeans.predict(pixels)
        # # Count occurrences of each cluster
        # _, counts = np.unique(assignments, return_counts=True)
        # # Calculate weights (proportion of points in each cluster)
        # weights = counts / counts.sum()
        # # Get weighted average color
        # dominant_color = np.sum(kmeans.cluster_centers_ * weights[:, np.newaxis], axis=0)


        # code for using faiss

        # Get cluster assignments and counts
        _, assignments = kmeans.index.search(pixels, 1)
        assignments = assignments.flatten()
        n_clusters = kmeans.centroids.shape[0]
        all_cluster_ids = np.arange(n_clusters)
        # Get the clusters that actually have points
        unique_clusters, counts = np.unique(assignments, return_counts=True)

        # Initialize weights array with zeros for all possible clusters
        all_counts = np.zeros(n_clusters, dtype=int)
        # Fill in the counts for clusters that have points
        all_counts[unique_clusters] = counts
        # Calculate weights based on all clusters
        weights = all_counts / all_counts.sum()

        # Now weights shape will be (3,) and after np.newaxis it will be (3, 1)
        weights = weights[:, np.newaxis]

        # Calculate weighted average color
        dominant_color = np.sum(kmeans.centroids * weights, axis=0)

        # _, counts = np.unique(assignments, return_counts=True)
        # # Calculate weights (proportion of points in each cluster)
        # weights = counts / counts.sum()
        # print(kmeans.centroids.shape, weights[:, np.newaxis].shape)
        # # Get weighted average color (useful for creating a palette representative of the image)
        # dominant_color = np.sum(kmeans.centroids * weights[:, np.newaxis], axis=0)

        return tuple(map(int, dominant_color))
    
    

def find_optimal_k(data, max_k=9, method='distortion') -> int:
    """
    Automatically determine the optimal number of clusters using 
    faster metrics with FAISS.
    
    Args:
        data: The image data for clustering (reshaped to 2D array of pixels)
        max_k: Maximum number of clusters to consider
        method: Metric to use for elbow detection
               'distortion' - Average distance to centroid (faster than inertia)
               'bic' - Bayesian Information Criterion (fast for large datasets)
               'variance_ratio' - Ratio of between-cluster to within-cluster variance
    
    Returns:
        optimal_k: The optimal number of clusters
    """

    # Convert data to float32 (required by FAISS)
    data = np.ascontiguousarray(data.astype('float32'))
    n_samples = data.shape[0]
    d = data.shape[1]  # Dimensionality (e.g., 3 for RGB)
    
    # Prepare storage for metrics
    metrics = []
    
    if data.shape[0] == 1:
        return 1

    else:
        if data.shape[0] <= max_k:
            max_k = data.shape[0]

        k_values = range(2, max_k + 1)
        
        for k in k_values:

            # Initialize and train FAISS K-means
            kmeans = faiss.Kmeans(d, k, niter=100, verbose=False)
            kmeans.train(data)
            
            # Get cluster assignments and distances
            cluster_index = faiss.IndexFlatL2(d)
            cluster_index.add(kmeans.centroids)
            distances, labels = cluster_index.search(data, 1)
            distances = distances.flatten()
            labels = labels.flatten()
            
            # Calculate metrics based on chosen method
            if method == 'distortion':
                # Average distance to centroid (faster than sum of squares)
                metric_value = np.mean(distances)
                
            elif method == 'bic':
                # Bayesian Information Criterion (penalizes complexity)
                # Calculate variance for each cluster
                variance = 0
                for i in range(k):
                    cluster_points = data[labels == i]
                    if len(cluster_points) > 0:
                        centroid = kmeans.centroids[i]
                        # Use L2 distance directly instead of calculating variance
                        variance += np.sum(np.square(cluster_points - centroid))
                
                variance = variance / (n_samples - k)
                # BIC formula: n*log(variance) + k*log(n)
                metric_value = n_samples * np.log(variance) + k * np.log(n_samples)
                
            elif method == 'variance_ratio':
                # Calculate within-cluster variance
                within_var = np.sum(distances) 
                
                # Calculate between-cluster variance (total variance - within variance)
                # This is much faster than calculating actual between-cluster variance
                global_centroid = np.mean(data, axis=0)
                total_var = np.sum(np.square(data - global_centroid))
                between_var = total_var - within_var
                
                # Use ratio of between to within variance
                # Higher is better, so we'll negate for the elbow method
                if within_var > 0:
                    metric_value = -between_var/within_var
                else:
                    metric_value = -float('inf')
            
            metrics.append(metric_value)
            # print(f"K={k}, {method}={metric_value:.2f}, time={time.time()-start_time:.3f}s")
        
        # Determine direction for KneeLocator
        if method in ['distortion', 'bic']:
            direction = "decreasing"
        else:  # variance_ratio is already negated
            direction = "decreasing"
        
        # Automatically detect the elbow point using the KneeLocator
        kneedle = KneeLocator(
            k_values, 
            metrics, 
            S=1.0, 
            curve="convex", 
            direction=direction
        )
        
        optimal_k = kneedle.elbow
        print(optimal_k)
        return optimal_k

def calculate_metric_for_k(data: np.ndarray, k: int, method: str = 'distortion') -> Tuple[int, float]:
    """
    Calculate clustering metric for a specific k value using FAISS.
    
    Args:
        data: Numpy array of data points
        k: Number of clusters to use
        method: Metric to use ('distortion', 'bic', or 'variance_ratio')
        
    Returns:
        Tuple of (k, metric_value)
    """
    # Convert data to float32 (required by FAISS)
    data = np.ascontiguousarray(data.astype('float32'))
    n_samples = data.shape[0]
    d = data.shape[1]  # Dimensionality
    
    # Skip if k is greater than or equal to n_samples
    if k >= n_samples:
        logger.warning(f"Skipping k={k} because it's >= number of samples ({n_samples})")
        return k, None
    
    start_time = time.time()
    
    # Initialize and train FAISS K-means
    kmeans = faiss.Kmeans(d, k, niter=100, verbose=False)
    kmeans.train(data)
    
    # Get cluster assignments and distances
    cluster_index = faiss.IndexFlatL2(d)
    cluster_index.add(kmeans.centroids)
    distances, labels = cluster_index.search(data, 1)
    distances = distances.flatten()
    labels = labels.flatten()
    
    # Calculate metrics based on chosen method
    if method == 'distortion':
        # Average distance to centroid
        metric_value = np.mean(distances)
        
    elif method == 'bic':
        # Bayesian Information Criterion
        variance = 0
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroid = kmeans.centroids[i]
                variance += np.sum(np.square(cluster_points - centroid))
        
        variance = variance / (n_samples - k)
        metric_value = n_samples * np.log(variance) + k * np.log(n_samples)
        
    elif method == 'variance_ratio':
        # Within-cluster variance
        within_var = np.sum(distances) 
        
        # Between-cluster variance approximation
        global_centroid = np.mean(data, axis=0)
        total_var = np.sum(np.square(data - global_centroid))
        between_var = total_var - within_var
        
        # Higher is better, so negate for elbow method
        if within_var > 0:
            metric_value = -between_var/within_var
        else:
            metric_value = -float('inf')
    
    execution_time = time.time() - start_time
    logger.info(f"K={k}, {method}={metric_value:.2f}, time={execution_time:.3f}s")
    
    return k, metric_value

def find_optimal_k_multithreaded(data: np.ndarray, max_k: int = 9, method: str = 'distortion', 
                               max_workers: int = None) -> int:
    """
    Automatically determine the optimal number of clusters using parallel processing.
    
    Args:
        data: The image data for clustering (reshaped to 2D array of pixels)
        max_k: Maximum number of clusters to consider
        method: Metric to use for elbow detection
               'distortion' - Average distance to centroid
               'bic' - Bayesian Information Criterion
               'variance_ratio' - Ratio of between-cluster to within-cluster variance
        max_workers: Maximum number of worker threads (None = use default based on CPU count)
    
    Returns:
        optimal_k: The optimal number of clusters
    """
    # Handle edge cases
    n_samples = data.shape[0]
    
    if n_samples <= 1:
        return 1
    
    if n_samples <= max_k:
        max_k = n_samples - 1
    
    # Prepare range of k values (start from 2)
    k_values = list(range(2, max_k + 1))
    
    # No need for threading if only checking a few k values
    if len(k_values) <= 1:
        if len(k_values) == 0:
            return 1
        return k_values[0]
    
    start_time = time.time()
    logger.info(f"Starting multithreaded elbow method with {len(k_values)} values of k")
    
    # Create a list to store (k, metric) tuples in order
    results = []
    
    # Use ThreadPoolExecutor for parallel computation
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit jobs
        future_to_k = {
            executor.submit(calculate_metric_for_k, data, k, method): k 
            for k in k_values
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_k):
            k = future_to_k[future]
            try:
                result = future.result()
                if result[1] is not None:  # Only add if metric was calculated
                    results.append(result)
            except Exception as exc:
                logger.error(f"K={k} generated an exception: {exc}")
    
    # Sort results by k value
    results.sort(key=lambda x: x[0])
    
    # Extract k values and metrics
    valid_k_values = [r[0] for r in results]
    metrics = [r[1] for r in results]
    
    if not metrics:
        logger.warning("No valid metrics calculated, defaulting to k=1")
        return 1
    
    # Determine direction for KneeLocator
    if method in ['distortion', 'bic']:
        direction = "decreasing"
    else:  # variance_ratio is already negated
        direction = "decreasing"
    
    # Find the elbow point
    try:
        kneedle = KneeLocator(
            valid_k_values, 
            metrics, 
            S=1.0, 
            curve="convex", 
            direction=direction
        )
        
        optimal_k = kneedle.elbow
        if optimal_k is None:
            logger.warning("No clear elbow found, using k with best metric")
            best_idx = np.argmin(metrics) if direction == "decreasing" else np.argmax(metrics)
            optimal_k = valid_k_values[best_idx]
    except Exception as e:
        logger.error(f"Error in KneeLocator: {e}")
        # Fallback: Use k with best metric
        best_idx = np.argmin(metrics) if direction == "decreasing" else np.argmax(metrics)
        optimal_k = valid_k_values[best_idx]
    
    total_time = time.time() - start_time
    logger.info(f"Multithreaded elbow method completed in {total_time:.3f}s, optimal k={optimal_k}")
    
    return optimal_k