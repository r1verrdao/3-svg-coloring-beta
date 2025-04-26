import logging
import math
import heapq
from functools import cached_property
from typing import List, Tuple, Optional, Sequence

from shapely.geometry import Point, Polygon
from shapely.errors import ShapelyError

from shapely.validation import explain_validity

from svg_color.config import POLYLABEL_PRECISION, POLYLABEL_MAX_ITERATIONS

logger = logging.getLogger(__name__)

class Cell:
    """
    Đại diện cho một ô vuông (cell) trong lưới của thuật toán Polylabel.

    Attributes:
        center_point (Point): Đối tượng Shapely Point cho tâm ô (x, y).
        h (float): Nửa chiều dài cạnh của ô.
        polygon (Polygon): Đa giác đang được phân tích.
        d (float): [Thuộc tính Cache] Khoảng cách có dấu từ center_point đến
                   cạnh đa giác gần nhất. Dương nếu ở trong, âm nếu ở ngoài.
        max_dist (float): [Thuộc tính] Khoảng cách tối đa có thể có từ bất kỳ
                          điểm nào bên trong ô này đến cạnh đa giác. Dùng cho
                          hàng đợi ưu tiên.
    """
    def __init__(self, x: float, y: float, h: float, polygon: Polygon):
        """Khởi tạo một Cell."""
        if not Point: # Kiểm tra lại nếu Shapely không load được
             raise RuntimeError("Shapely library is required but not loaded.")
        self.h = h
        self.center_point = Point(x, y)
        self.polygon = polygon
        # Không tính max_dist ở đây vì nó phụ thuộc vào d (là cached_property)

    @cached_property
    def d(self) -> float:
        """
        [Thuộc tính Cache] Tính khoảng cách có dấu từ tâm ô đến
        biên đa giác.

        Khoảng cách dương nghĩa là tâm nằm bên trong đa giác.
        Khoảng cách âm nghĩa là tâm nằm bên ngoài.

        Returns:
            Khoảng cách có dấu, hoặc -infinity nếu tính toán thất bại.
        """
        try:
            distance_to_boundary = self.polygon.boundary.distance(self.center_point)
            # Xử lý trường hợp điểm nằm rất gần hoặc trên biên (dung sai nhỏ)
            if distance_to_boundary < 1e-9:
                 return 0.0
            # Kiểm tra xem điểm có thực sự nằm trong không (tránh lỗi làm tròn)
            elif self.polygon.contains(self.center_point):
                return distance_to_boundary # Bên trong
            else:
                return -distance_to_boundary # Bên ngoài
        except ShapelyError as dist_err:
            logger.warning(f"ShapelyError calculating distance for point ({self.center_point.x},{self.center_point.y}) to polygon: {dist_err}")
            return -float('inf')
        except Exception as general_err:
            logger.warning(f"Unexpected error calculating distance for point ({self.center_point.x},{self.center_point.y}): {general_err}")
            return -float('inf')

    @property
    def max_dist(self) -> float:
        """
        [Thuộc tính] Tính khoảng cách tối đa có thể có từ bất kỳ điểm nào
        bên trong ô này đến cạnh đa giác. Phụ thuộc vào self.d.
        """
        # self.d sẽ được tính và cache khi được truy cập lần đầu ở đây
        return self.d + self.h * math.sqrt(2)

    def __lt__(self, other: 'Cell') -> bool:
        """So sánh cho max-heap (sử dụng heapq - min-heap). Ưu tiên max_dist cao hơn."""
        # self.max_dist và other.max_dist sẽ gọi property để lấy giá trị
        return self.max_dist < other.max_dist

    def __repr__(self) -> str:
        """Biểu diễn chuỗi để debug."""
        # Truy cập self.d và self.max_dist sẽ kích hoạt các property tương ứng
        return f"Cell(x={self.center_point.x:.2f}, y={self.center_point.y:.2f}, h={self.h:.2f}, d={self.d:.2f}, max={self.max_dist:.2f})"


# --- Lớp PolylabelCalculator ---
class PolylabelCalculator:
    """
    Đóng gói trạng thái và logic cho thuật toán Polylabel.

    Tìm "pole of inacsessabillity" - một điểm bên trong đa giác cách xa nhất
    so với biên của nó, thường dùng làm tâm trực quan tốt.
    """
    def __init__(self, polygon: Polygon, precision: float = POLYLABEL_PRECISION, max_iterations: int = POLYLABEL_MAX_ITERATIONS):
        """
        Khởi tạo PolylabelCalculator.

        Args:
            polygon: Đối tượng Shapely Polygon để phân tích.
            precision: Độ chính xác mong muốn cho kết quả (tiêu chí dừng).
            max_iterations: Giới hạn an toàn cho số lần lặp.
        """
        if not Polygon or not isinstance(polygon, Polygon) or polygon.is_empty:
             # Validation này thường được xử lý ở hàm facade, nhưng có thể thêm ở đây nếu cần
             raise ValueError("Invalid or empty Shapely Polygon provided to PolylabelCalculator.")

        self.precision = precision
        self.max_iterations = max_iterations
        self.polygon = polygon
        self.minx, self.miny, self.maxx, self.maxy = self.get_polygon_bbox(polygon)

        if self.minx == self.maxx or self.miny == self.maxy:
            raise ValueError("Polygon bounding box has zero area. Cannot calculate polylabel.")

        self._cell_queue: List[Tuple[float, Cell]] = [] # Min-heap (lưu trữ -max_dist)
        self.best_cell: Optional[Cell] = None

    @staticmethod
    def get_polygon_bbox(polygon: Polygon) -> Tuple[float, float, float, float]:
        """Lấy hộp giới hạn (bounding box) của đa giác."""
        return polygon.bounds

    def _create_cell(self, x: float, y: float, h: float) -> Cell:
        """Phương thức trợ giúp để tạo một Cell."""
        return Cell(x, y, h, self.polygon)

    def _find_initial_best_cell(self) -> Cell:
        """
        Xác định một 'best cell' ứng viên ban đầu.
        Thử trọng tâm đa giác trước. Nếu nằm ngoài, kiểm tra các góc.
        """
        # Bắt đầu với trọng tâm
        centroid = self.polygon.centroid
        initial_best = self._create_cell(centroid.x, centroid.y, 0) # h=0 cho điểm ban đầu

        # Nếu trọng tâm nằm ngoài, thử các góc
        if initial_best.d < 0:
            logger.debug("Centroid is outside the polygon. Checking corners for initial best cell.")
            best_corner_dist = -float('inf')
            best_corner_cell = initial_best # Giữ lại trọng tâm làm dự phòng

            try:
                 coords = list(self.polygon.exterior.coords)
                 checked_coords = set()
                 for px, py in coords:
                     if (px, py) not in checked_coords and len(checked_coords) < 10: # Giới hạn kiểm tra
                         corner_cell = self._create_cell(px, py, 0)
                         if corner_cell.d > best_corner_dist:
                             best_corner_dist = corner_cell.d
                             best_corner_cell = corner_cell
                         checked_coords.add((px, py))
                 # Nếu tìm thấy góc tốt hơn
                 if best_corner_cell.d > initial_best.d:
                      initial_best = best_corner_cell
            except AttributeError:
                 logger.warning("Could not access polygon exterior coordinates to check corners.")
            except Exception as e:
                 logger.warning(f"Error checking polygon corners: {e}")

        # Sử dụng center_point để lấy tọa độ cho log
        logger.debug(f"Initial best cell guess: d={initial_best.d:.3f} at ({initial_best.center_point.x:.2f}, {initial_best.center_point.y:.2f})")
        return initial_best


    def _initialize_queue(self):
        """Khởi tạo hàng đợi ưu tiên với các ô bao phủ hộp giới hạn."""
        width = self.maxx - self.minx
        height = self.maxy - self.miny
        cell_size = min(width, height)
        if cell_size <= 0:
             raise ValueError("Polygon has zero width or height after bounding box check.")

        h = cell_size / 2.0 # Nửa cạnh ô ban đầu

        if cell_size < self.precision * 2:
             h = cell_size / 4.0
             logger.debug(f"Polygon size ({cell_size:.2f}) is small relative to precision. Adjusting initial h to {h:.2f}")

        self.best_cell = self._find_initial_best_cell()
        # Không cần raise RuntimeError nếu best_cell là None ở đây, vì kiểm tra polygon hợp lệ ở init

        # Bao phủ hộp giới hạn bằng các ô ban đầu
        x = self.minx
        while x < self.maxx:
            y = self.miny
            while y < self.maxy:
                cell = self._create_cell(x + h, y + h, h)
                # Thêm (-max_dist, cell) vào min-heap
                heapq.heappush(self._cell_queue, (-cell.max_dist, cell))
                y += cell_size
            x += cell_size

        # Xử lý trường hợp hàng đợi trống (đa giác rất nhỏ/mỏng)
        if not self._cell_queue and self.best_cell:
             logger.warning("Initial cell queue is empty. Polygon might be too small or thin.")
             if self.best_cell.h == 0: # Chỉ thêm nếu best_cell là điểm ban đầu (h=0)
                 # Truy cập tọa độ qua center_point
                 fallback_cell = self._create_cell(self.best_cell.center_point.x,
                                                 self.best_cell.center_point.y,
                                                 h if h > 0 else self.precision) # Cần h > 0
                 if fallback_cell.h > 0: # Đảm bảo h thực sự > 0
                    heapq.heappush(self._cell_queue, (-fallback_cell.max_dist, fallback_cell))
                 else:
                      logger.warning("Could not create fallback cell with positive h.")


    def _process_queue(self) -> Optional[Tuple[float, float]]:
        """
        Thực hiện vòng lặp chính của thuật toán Polylabel để tìm tâm đa giác.
        """
        num_iterations = 0

        while self._cell_queue and num_iterations < self.max_iterations:
            num_iterations += 1

            try:
                neg_max_dist, current_cell = heapq.heappop(self._cell_queue)
            except IndexError:
                 logger.warning("Attempted to pop from an empty queue during processing.")
                 break

            # Cập nhật best_cell nếu tâm ô hiện tại tốt hơn
            if current_cell.d > self.best_cell.d:
                self.best_cell = current_cell

            # Cắt tỉa (Pruning)
            if current_cell.max_dist - self.best_cell.d <= self.precision:
                continue

            # Chia nhỏ (Subdivision)
            h_new = current_cell.h / 2.0
            if h_new < 1e-9:
                continue

            # Lấy tọa độ tâm từ center_point của ô hiện tại
            cx, cy = current_cell.center_point.x, current_cell.center_point.y
            sub_cells_coords = [
                (cx - h_new, cy - h_new), (cx + h_new, cy - h_new),
                (cx - h_new, cy + h_new), (cx + h_new, cy + h_new),
            ]

            # Thêm các ô con vào hàng đợi
            for sx, sy in sub_cells_coords:
                new_cell = self._create_cell(sx, sy, h_new)
                heapq.heappush(self._cell_queue, (-new_cell.max_dist, new_cell))

        # --- Kết thúc vòng lặp ---
        if num_iterations >= self.max_iterations:
            logger.warning(f"Polylabel reached max iterations ({self.max_iterations}). Result might be suboptimal. Best d={self.best_cell.d:.4f}")

        # Trả về kết quả
        if self.best_cell:
             if self.best_cell.d < 0:
                 logger.warning(f"Polylabel result has negative distance ({self.best_cell.d:.4f}), indicating the calculated center might be slightly outside the polygon.")
             # Trả về tọa độ từ center_point
             return self.best_cell.center_point.x, self.best_cell.center_point.y
        else:
            logger.error("Polylabel finished without finding a best cell.")
            return None


    def calculate(self) -> Optional[Tuple[float, float]]:
        """
        Thực thi thuật toán Polylabel.

        Returns:
            Tuple (x, y) cho tâm polylabel, hoặc None nếu tính toán thất bại.
        """
        try:
            self._initialize_queue()
            # Kiểm tra sau khi khởi tạo
            if not self._cell_queue and not self.best_cell:
                 logger.error("Failed to initialize Polylabel queue and best cell.")
                 return None
            # Nếu hàng đợi trống nhưng có best_cell ban đầu (vd: đa giác rất nhỏ)
            if not self._cell_queue and self.best_cell:
                 logger.warning("Initial queue was empty, returning initial best cell guess.")
                 # Trả về tọa độ từ center_point
                 return self.best_cell.center_point.x, self.best_cell.center_point.y

            # Nếu hàng đợi không trống, xử lý nó
            return self._process_queue()

        except ValueError as ve: # Bắt lỗi từ init hoặc các bước khác
             logger.error(f"Error during Polylabel initialization or processing: {ve}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error during Polylabel calculation: {e}", exc_info=True)
            return None


# --- Hàm Facade (Giao diện đơn giản) ---
def calculate_visual_center(coords: Sequence[Tuple[float, float]], precision: float = POLYLABEL_PRECISION) -> Optional[Tuple[float, float]]:
    """
    Tính toán cực bất khả xâm phạm (tâm trực quan) của một đa giác
    được định nghĩa bởi tọa độ sử dụng thuật toán Polylabel.

    Args:
        coords: Một chuỗi các tuple (x, y) định nghĩa biên ngoài
                của đa giác. Các vòng trong (lỗ) không được xử lý trực tiếp
                bởi cài đặt này nhưng có thể hoạt động nếu Shapely Polygon xử lý chúng.
        precision: Độ chính xác mong muốn cho kết quả. Kiểm soát mức độ
                   chính xác của tâm được định vị so với biên xa nhất.

    Returns:
        Một tuple (x, y) cho tâm polylabel, hoặc None nếu tính toán thất bại
        (ví dụ: tọa độ không hợp lệ, lỗi Shapely, lỗi thuật toán).
    """
    if not Point or not Polygon: # Kiểm tra Shapely có sẵn không
         logger.error("Shapely library is required for polylabel calculation but not found/loaded.")
         return None

    if not coords or len(coords) < 3:
        logger.warning("Insufficient coordinates provided to form a polygon (minimum 3 required).")
        return None

    try:
        # Tạo đối tượng đa giác, xử lý lỗi tiềm ẩn
        polygon = Polygon(coords)

        # Kiểm tra và cố gắng sửa đa giác không hợp lệ
        if not polygon.is_valid:
            # logger.warning(f"Input coordinates result in an invalid Shapely polygon: {explain_validity(polygon)}. Attempting buffer(0) fix.")
            polygon = polygon.buffer(0)
            if not polygon.is_valid or polygon.is_empty:
                 logger.error("Polygon remains invalid or became empty after buffer(0) fix.")
                 return None

        if polygon.is_empty:
             logger.warning("Input coordinates result in an empty Shapely polygon.")
             return None

        # Khởi tạo và chạy trình tính toán
        calculator = PolylabelCalculator(polygon, precision)
        center = calculator.calculate()
        return center

    except ShapelyError as se:
        logger.error(f"ShapelyError creating polygon from coordinates: {se}", exc_info=True)
        return None
    except ValueError as ve: # Bắt lỗi từ PolylabelCalculator
         logger.error(f"Error calculating polylabel: {ve}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error in calculate_visual_center: {e}", exc_info=True)
        return None

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Example 1: Simple Square
    square_coords = [(0, 0), (10, 0), (10, 10), (0, 10), (0,0)] # Đảm bảo đóng kín
    center_sq = calculate_visual_center(square_coords, precision=0.1)
    if center_sq:
        logger.info(f"Square Center: ({center_sq[0]:.2f}, {center_sq[1]:.2f}) (Expected: close to (5, 5))")
    else:
        logger.error("Failed to calculate center for square.")


    # Example 2: More complex shape (e.g., a 'C' shape)
    c_shape_coords = [
        (0, 0), (10, 0), (10, 2), (2, 2), (2, 8), (10, 8), (10, 10), (0, 10), (0,0) # Closed
    ]
    center_c = calculate_visual_center(c_shape_coords, precision=0.1)
    if center_c:
         logger.info(f"C-Shape Center: ({center_c[0]:.2f}, {center_c[1]:.2f})")
    else:
         logger.error("Failed to calculate center for C-shape.")

    # Example 3: Polygon with centroid outside (requires corner check)
    crescent_coords = [
        (0,0), (10, 5), (0, 10), (5, 5), (0,0) # Simple crescent-like shape
    ]
    center_cr = calculate_visual_center(crescent_coords, precision=0.1)
    if center_cr:
         logger.info(f"Crescent Center: ({center_cr[0]:.2f}, {center_cr[1]:.2f})")
    else:
         logger.error("Failed to calculate center for crescent.")

    # Example 4: Invalid input
    invalid_coords = [(0,0), (1,1)]
    center_inv = calculate_visual_center(invalid_coords)
    if not center_inv:
        logger.info("Correctly handled invalid coordinates (returned None).")

    # Example 5: Polygon that might become empty after buffer(0)
    # (Example: a line or self-intersecting in a way that cancels out)
    line_coords = [(0,0), (10,10), (0,0)]
    center_line = calculate_visual_center(line_coords)
    if not center_line:
         logger.info("Correctly handled line coordinates (returned None).")