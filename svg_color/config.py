import os
import logging # Import logging để định nghĩa các hằng số mức độ log
import re


__all__ = [
    # Project paths
    "CURRENT_FILE_PATH", "SRC_DIR", "PROJECT_ROOT",
    # Data & logging dirs
    "BASE_DATA_DIR", "LOG_DIR",
    # Subdirectory & file suffixes
    "INPUT_DIR_NAME", "OUTPUT_DIR_NAME",
    "INPUT_SVG_SUFFIX", "INPUT_PNG_SUFFIX", "OUTPUT_SVG_SUFFIX",
    # Logging config
    "LOG_FILE_NAME", "DEFAULT_LOG_LEVEL_STR", "LOG_LEVEL_STR",
    "LOG_LEVEL_MAP", "EFFECTIVE_LOG_LEVEL",
    "LOG_FORMAT", "LOG_DATE_FORMAT",
    # SVG Processing defaults
    "DEFAULT_MAX_COLORS", "DEFAULT_COLOR_THRESHOLD",
    # UI config
    "APP_TITLE",  # bạn có thể thêm MAX_UPLOAD_SIZE_* nếu bật
    # Other constants
    "DEFAULT_ENCODING",
    # SVG processing params
    "_COLOR_SAMPLING_PATCH_SIZE",
    "_CIRCLE_APPROX_SEGMENTS",
    "_PATH_COMMAND_REGEX", "_PATH_PARAM_REGEX", "_CMD_PARAM_COUNTS",
    # Namespace & elements
    "_SVG_NS", "_XLINK_NS", "_ELEMENTS_TO_PROCESS",
    "POLYLABEL_PRECISION",
    "POLYLABEL_MAX_ITERATIONS",
    "ALBUM_ID_PATTERN"
]

# --- Project Root Directory ---
# Xác định đường dẫn tuyệt đối đến thư mục gốc của dự án.
# Giả định file config.py nằm trong src/svg_color_tool/
# Điều chỉnh số lần gọi os.path.dirname nếu cấu trúc thư mục của bạn khác.
try:
    # __file__ là đường dẫn đến file config.py hiện tại
    CURRENT_FILE_PATH = os.path.abspath(__file__)
    
    # Đi lên 1 cấp thư mục từ config.py để đến svg_color 
    SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)
    # print(SRC_DIR)
     # Đi lên 1 cấp nữa từ src/ để đến thư mục gốc dự án
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
except NameError:
     # Fallback nếu __file__ không được định nghĩa (ví dụ: khi chạy trong môi trường tương tác nhất định)
    PROJECT_ROOT = os.path.abspath(".") # Sử dụng thư mục làm việc hiện tại làm gốc
    print(f"Warning: __file__ not defined, using current working directory as PROJECT_ROOT: {PROJECT_ROOT}")


# --- Data and Logging Directories ---
# Định nghĩa đường dẫn tương đối so với thư mục gốc của dự án
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# --- Standard Subdirectory and File Names/Suffixes ---
# Định nghĩa tên thư mục con và đuôi file chuẩn để đảm bảo tính nhất quán
INPUT_DIR_NAME = "input"
OUTPUT_DIR_NAME = "output"
INPUT_SVG_SUFFIX = ".svg"
INPUT_PNG_SUFFIX = ".png"
OUTPUT_SVG_SUFFIX = ".svg" # Thường giống đuôi file SVG đầu vào

# --- Logging Configuration ---
LOG_FILE_NAME = "app.log"
# Mức độ log mặc định. Có thể được ghi đè bởi biến môi trường.
# Các mức độ phổ biến: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
DEFAULT_LOG_LEVEL_STR = "INFO"
LOG_LEVEL_STR = os.environ.get("SVG_TOOL_LOG_LEVEL", DEFAULT_LOG_LEVEL_STR).upper()

# Ánh xạ chuỗi mức độ log sang hằng số của thư viện logging
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
# Lấy hằng số logging tương ứng, hoặc dùng INFO nếu chuỗi không hợp lệ
EFFECTIVE_LOG_LEVEL = LOG_LEVEL_MAP.get(LOG_LEVEL_STR, logging.INFO)

# Định dạng chuỗi log
LOG_FORMAT = '%(asctime)s - %(name)s - [%(levelname)s] - [%(funcName)s:%(lineno)d] - %(message)s'
# Định dạng thời gian cho log
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- SVG Processing Defaults (Optional) ---
# Giá trị mặc định cho các tham số của SvgProcessor
# Có thể được sử dụng khi khởi tạo SvgProcessor nếu không có giá trị cụ thể nào được cung cấp
DEFAULT_MAX_COLORS = 80
DEFAULT_COLOR_THRESHOLD = 30

# --- UI Configuration (Optional) ---
# Các hằng số liên quan đến giao diện người dùng Streamlit
APP_TITLE = "SVG Coloring Tool"
# Ví dụ: Giới hạn kích thước file upload (tính bằng bytes)
# MAX_UPLOAD_SIZE_MB = 50
# MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# --- Other Constants ---
# Mã hóa ký tự mặc định được sử dụng trong ứng dụng
DEFAULT_ENCODING = "utf-8"


# SVG Processing Params
_COLOR_SAMPLING_PATCH_SIZE = 3 # Size of the pixel neighborhood for color sampling
_CIRCLE_APPROX_SEGMENTS = 16 # Number of segments to approximate a circle as a polygon


_PATH_COMMAND_REGEX = re.compile(r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)')
# Regex to extract floating point numbers from the parameters string
_PATH_PARAM_REGEX = re.compile(r'[-+]?(?:\d*\.?\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?')
# Expected number of parameters for each command type (uppercase)
_CMD_PARAM_COUNTS = {'M': 2, 'L': 2, 'H': 1, 'V': 1, 'C': 6, 'S': 4, 'Q': 4, 'T': 2, 'A': 7, 'Z': 0}



# --- Constants for Clarity and Maintainability ---
_SVG_NS = "{http://www.w3.org/2000/svg}"
_XLINK_NS = "{http://www.w3.org/1999/xlink}"
_ELEMENTS_TO_PROCESS = ["path", "polygon", "rect", "circle", "ellipse"] # SVG tags to consider for coloring


POLYLABEL_PRECISION = 1.0 # Precision for polylabel calculation
POLYLABEL_MAX_ITERATIONS = 20000 # Thêm config cho max iterations


# Regex cho Album ID 
ALBUM_ID_PATTERN = r'^\d{6}|\d{10}$'