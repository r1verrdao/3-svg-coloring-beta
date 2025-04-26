import re
import sys
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import math
from shapely.geometry import Point, Polygon
import heapq
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth


def parse_path(path_d):
    """
    Phân tích chuỗi d trong thẻ path để lấy danh sách các điểm tạo thành path
    """
    # Tách các lệnh và các tọa độ
    commands = re.findall(r'[MLHVCSQTAZmlhvcsqtaz][^MLHVCSQTAZmlhvcsqtaz]*', path_d)

    # Danh sách các điểm
    points = []
    current_point = (0, 0)

    for cmd in commands:
        command = cmd[0]
        params = re.findall(r'[-+]?[0-9]*\.?[0-9]+', cmd[1:])
        params = [float(p) for p in params]

        if command == 'M':  # Move to absolute
            for i in range(0, len(params), 2):
                current_point = (params[i], params[i + 1])
                points.append(current_point)
        elif command == 'm':  # Move to relative
            for i in range(0, len(params), 2):
                current_point = (current_point[0] + params[i], current_point[1] + params[i + 1])
                points.append(current_point)
        elif command == 'L':  # Line to absolute
            for i in range(0, len(params), 2):
                current_point = (params[i], params[i + 1])
                points.append(current_point)
        elif command == 'l':  # Line to relative
            for i in range(0, len(params), 2):
                current_point = (current_point[0] + params[i], current_point[1] + params[i + 1])
                points.append(current_point)
        elif command == 'H':  # Horizontal line absolute
            for param in params:
                current_point = (param, current_point[1])
                points.append(current_point)
        elif command == 'h':  # Horizontal line relative
            for param in params:
                current_point = (current_point[0] + param, current_point[1])
                points.append(current_point)
        elif command == 'V':  # Vertical line absolute
            for param in params:
                current_point = (current_point[0], param)
                points.append(current_point)
        elif command == 'v':  # Vertical line relative
            for param in params:
                current_point = (current_point[0], current_point[1] + param)
                points.append(current_point)
        # Các lệnh phức tạp hơn như C, S, Q, T, A chúng ta sẽ chỉ lấy điểm cuối
        elif command.upper() in 'CSQTA':
            # Lấy điểm cuối cùng
            if command.upper() == 'A':
                if len(params) >= 7:
                    i = 0
                    while i + 6 < len(params):
                        if command.islower():
                            current_point = (current_point[0] + params[i + 5], current_point[1] + params[i + 6])
                        else:
                            current_point = (params[i + 5], params[i + 6])
                        points.append(current_point)
                        i += 7
            else:
                cmd_param_count = {'C': 6, 'S': 4, 'Q': 4, 'T': 2}
                param_count = cmd_param_count.get(command.upper(), 0)

                if param_count > 0:
                    i = 0
                    while i + param_count - 1 < len(params):
                        if command.islower():
                            current_point = (current_point[0] + params[i + param_count - 2],
                                             current_point[1] + params[i + param_count - 1])
                        else:
                            current_point = (params[i + param_count - 2], params[i + param_count - 1])
                        points.append(current_point)
                        i += param_count
    return points


def get_dominant_color(img, points, method='mean', k=1):
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


def color_distance(c1, c2):
    """
    Calculate Euclidean color distance between two RGB colors

    Args:
        c1 (tuple): First color as RGB tuple
        c2 (tuple): Second color as RGB tuple

    Returns:
        float: Color distance
    """
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return math.sqrt(dr * dr + dg * dg + db * db)


def merge_similar_colors(colors, max_colors=80, threshold=30):
    """
    Merge similar colors based on color distance with improved logic

    Args:
        colors (list): List of RGB color tuples
        max_colors (int): Maximum number of colors to return
        threshold (float): Color distance threshold for merging

    Returns:
        list: Merged colors list
    """
    # Nếu số lượng màu nhỏ hơn hoặc bằng max_colors, trả về nguyên danh sách
    if len(colors) <= max_colors:
        return colors

    # Sắp xếp các màu theo độ phổ biến (số lần xuất hiện)
    color_counts = Counter(colors)
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

    # Bắt đầu với các màu phổ biến nhất
    merged_colors = [color for color, _ in sorted_colors[:max_colors]]

    # Nếu vẫn chưa đủ max_colors, thử thêm các màu khác
    if len(merged_colors) < max_colors:
        remaining_colors = [color for color, _ in sorted_colors[max_colors:]]

        for color in remaining_colors:
            # Kiểm tra xem màu mới có gần với bất kỳ màu nào trong merged_colors không
            is_similar = any(color_distance(color, existing_color) <= threshold
                             for existing_color in merged_colors)

            if not is_similar and len(merged_colors) < max_colors:
                merged_colors.append(color)

    return merged_colors

def find_largest_area_color(img, elem, method='kmeans', k=1):
    """
    Tìm vị trí bounding box lớn nhất trong phần tử SVG và lấy màu tại trung tâm

    Args:
        img (numpy.ndarray): Ảnh nguồn để lấy màu
        elem (xml.etree.ElementTree.Element): Phần tử SVG (path hoặc polygon)
        method (str): Phương pháp chọn màu ('kmeans' hoặc 'most_common')
        k (int): Số lượng cụm màu cho KMeans

    Returns:
        tuple: Màu RGB tại khu vực lớn nhất
    """

    def point_to_polygon_distance(x, y, polygon):
        pt = Point(x, y)
        if polygon.contains(pt):
            return pt.distance(polygon.exterior)
        else:
            return -pt.distance(polygon.exterior)

    class Cell:
        def __init__(self, x, y, h, polygon):
            self.x = x
            self.y = y
            self.h = h
            self.d = point_to_polygon_distance(x, y, polygon)
            self.max = self.d + self.h * math.sqrt(2)

        def __lt__(self, other):
            return self.max > other.max

    def polylabel(polygon, precision=1.0):
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        cell_size = min(width, height)
        h = cell_size / 2.0

        best_cell = Cell((minx + maxx) / 2.0, (miny + maxy) / 2.0, 0, polygon)
        if best_cell.d < 0:
            for x in [minx, maxx]:
                for y in [miny, maxy]:
                    cell = Cell(x, y, 0, polygon)
                    if cell.d > best_cell.d:
                        best_cell = cell

        cell_queue = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                cell = Cell(x + h, y + h, h, polygon)
                heapq.heappush(cell_queue, (-cell.max, cell))
                y += cell_size
            x += cell_size

        while cell_queue:
            _, cell = heapq.heappop(cell_queue)
            if cell.d > best_cell.d:
                best_cell = cell
            if cell.max - best_cell.d <= precision:
                continue
            h_new = cell.h / 2.0
            for dx in [-h_new, h_new]:
                for dy in [-h_new, h_new]:
                    new_cell = Cell(cell.x + dx, cell.y + dy, h_new, polygon)
                    heapq.heappush(cell_queue, (-new_cell.max, new_cell))

        return best_cell.x, best_cell.y, best_cell.d

    def get_bounding_box(points):
        """
        Tính toán hình chữ nhật bao quanh các điểm
        """
        if not points:
            return (0, 0, 0, 0)

        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)

        return (min_x, min_y, max_x, max_y)

    def rgb_to_hex(rgb):
        """
        Chuyển đổi màu RGB sang mã hex
        """
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    # Trích xuất tọa độ từ phần tử SVG
    poly_coords = None
    if hasattr(elem, "get"):
        points_str = elem.get("points")
        if points_str:
            coords = []
            for pair in points_str.strip().split():
                x_str, y_str = pair.split(',')
                coords.append((float(x_str), float(y_str)))
            poly_coords = coords
        elif elem.tag.endswith("path"):
            d_attr = elem.get("d")
            if d_attr:
                try:
                    poly_coords = parse_path(d_attr)
                except Exception as e:
                    print(f"Lỗi khi phân tích path: {e}")

    if poly_coords is None:
        return None

    polygon = Polygon(poly_coords)

    # Tìm điểm trung tâm của bounding box lớn nhất
    opt_x, opt_y, _ = polylabel(polygon)

    # Lấy màu tại khu vực lớn nhất
    color = get_dominant_color(img, poly_coords, method=method, k=k)

    return color


def process_svg_with_png(svg_path, png_path, output_path, method='kmeans', k=1, max_colors=80, color_threshold=10):
    """
    Xử lý file SVG với ảnh PNG để lấy màu sắc cho từng path, giới hạn số lượng màu
    """
    # Kiểm tra giá trị max_colors
    if max_colors <= 0:
        print("Lỗi: Số lượng màu tối đa phải lớn hơn 0.")
        sys.exit(1)

    # Đọc ảnh PNG
    img = cv2.imread(png_path)
    if img is None:
        print(f"Không thể đọc ảnh: {png_path}")
        return

    # Đổi sang BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Parse SVG
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Xử lý các path và tạo danh sách style classes
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    polygons = root.findall(".//{http://www.w3.org/2000/svg}polygon")
    elements = paths + polygons

    # Trích xuất màu từ tất cả các phần tử
    all_colors = []
    for elem in elements:
        color = find_largest_area_color(img, elem, method=method, k=k)
        if color:
            all_colors.append(color)

    # Gộp màu giống nhau, với số lượng màu được giới hạn bởi max_colors
    merged_colors = merge_similar_colors(all_colors, max_colors=max_colors, threshold=color_threshold)

    # Ánh xạ màu gốc với màu đã gộp
    color_mapping = {}
    style_classes = {}

    for i, elem in enumerate(elements):
        color = find_largest_area_color(img, elem, method=method, k=k)
        if color:
            # Tìm màu gần nhất trong danh sách merged_colors
            closest_color = min(merged_colors, key=lambda x: color_distance(color, x))

            # Chuyển màu sang hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(closest_color[0], closest_color[1], closest_color[2])

            # Đặt thuộc tính fill trực tiếp vào path/polygon
            elem.set("fill", hex_color)

            # Xóa các thuộc tính style cũ có thể ảnh hưởng đến việc hiển thị
            if elem.get("style"):
                style_parts = elem.get("style").split(";")
                new_style_parts = []
                for part in style_parts:
                    if not part.strip().startswith("fill:"):
                        new_style_parts.append(part)
                if new_style_parts:
                    elem.set("style", ";".join(new_style_parts))
                else:
                    elem.attrib.pop("style", None)

            # Tạo class cho mỗi màu
            class_name = f"st{merged_colors.index(closest_color)}"
            style_classes[class_name] = hex_color

            # Kiểm tra và loại bỏ các class không mong muốn
            existing_classes = elem.get("class", "").split()
            filtered_classes = [cls for cls in existing_classes if not cls.startswith('st')]
            filtered_classes.append(class_name)
            elem.set("class", " ".join(filtered_classes))

    # Tìm hoặc tạo phần style
    style_element = None
    for child in root:
        if child.tag.endswith('style'):
            style_element = child
            break

    if style_element is None:
        style_element = ET.SubElement(root, "{http://www.w3.org/2000/svg}style")
        style_element.set("type", "text/css")
        style_content = "\n"
    else:
        style_content = style_element.text if style_element.text else "\n"
        style_lines = style_content.split("\n")
        new_style_lines = []
        for line in style_lines:
            is_duplicate = False
            for class_name in style_classes.keys():
                if f".{class_name}" in line and "fill:" in line:
                    is_duplicate = True
                    break
            if not is_duplicate:
                new_style_lines.append(line)
        style_content = "\n".join(new_style_lines)

    # Thêm các class style mới
    style_additions = "    " + "\n    ".join(
        [f".{class_name}{{fill:{color};}}" for class_name, color in style_classes.items()])

    # Kết hợp style cũ và mới
    if style_content.strip():
        style_content = style_content.rstrip() + "\n    " + style_additions + "\n"
    else:
        style_content = "\n" + style_additions + "\n"

    style_element.text = style_content

    # Lưu file SVG mới
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Đã lưu SVG với {len(merged_colors)} màu tại: {output_path}")
    print(f"Các màu được sử dụng: {merged_colors}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract colors from PNG and apply to SVG')
    parser.add_argument('--sketch_svg', help='Path to the SVG file', required=True)
    parser.add_argument('--full_color_png', help='Path to the PNG file', required=True)
    parser.add_argument('--output_svg', help='Path to save the output SVG file', required=True)
    parser.add_argument('--method', choices=['kmeans', 'dbscan', 'most_common', 'meanshift'], default='kmeans',
                        help='Method to find dominant color (default: kmeans)')
    parser.add_argument('--k', type=int, default=1, help='Number of clusters for KMeans (default: 1)')
    parser.add_argument('--max_colors', type=int, default=80, help='Maximum number of colors (default: 80)')
    parser.add_argument('--color_threshold', type=float, default=30,
                        help='Color distance threshold for merging similar colors (default: 30)')

    args = parser.parse_args()

    process_svg_with_png(args.sketch_svg, args.full_color_png, args.output_svg,
                         method=args.method, k=args.k,
                         max_colors=args.max_colors,
                         color_threshold=args.color_threshold)

if __name__ == "__main__":
    main()