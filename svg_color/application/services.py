# application/services.py
from typing import IO, Tuple, Optional

import logging
import time
import numpy as np

from PIL import Image
from ..domain.models import ProcessingInput, ProcessingResult, ColorConfig
from ..core.svg_processor import SvgProcessor # Logic xử lý cốt lõi
from ..infrastructure.file_gateway import FileGateway # Gateway để tương tác file
# from ..core.exceptions import ProcessingError # Nếu có exception tùy chỉnh

logger = logging.getLogger(__name__)

class ProcessSvgService:
    """Orchestrates the SVG processing use case."""

    def __init__(self, file_gateway: FileGateway, processor: SvgProcessor):
        # Dependency Injection: Nhận FileGateway từ bên ngoài
        self.file_gateway = file_gateway
        # Khởi tạo SvgProcessor ở đây hoặc nhận từ ngoài nếu cần cấu hình phức tạp
        self.processor = processor
        # self.svg_processor = SvgProcessor() # Cấu hình mặc định

    def execute(self, album_id: str, svg_upload_stream: IO[bytes], png_upload_stream: IO[bytes], color_config: ColorConfig) -> ProcessingResult:
        """
        Executes the SVG processing pipeline.

        Args:
            album_id: The album ID.
            svg_upload_stream: File-like object for the uploaded SVG.
            png_upload_stream: File-like object for the uploaded PNG.
            config: Color processing configuration.

        Returns:
            A ProcessingResult object indicating success or failure.
        """
        start_time = time.time()
        logger.info(f"Starting ProcessSvgService for Album ID: {album_id}")

        # --- Chuẩn bị đường dẫn và thư mục ---
        try:
            self.file_gateway.ensure_directories(album_id)
            svg_path = self.file_gateway.get_input_svg_path(album_id)
            png_path = self.file_gateway.get_input_png_path(album_id)
            output_svg_path = self.file_gateway.get_output_svg_path(album_id)
        except OSError as e:
            return ProcessingResult(success=False, message=f"Error setting up directories: {e}")

        # --- Lưu file tải lên ---
        try:
            # Quan trọng: seek(0) trước khi lưu nếu stream đã được đọc
            svg_upload_stream.seek(0)
            png_upload_stream.seek(0)
            self.file_gateway.save_file(svg_upload_stream, svg_path)
            self.file_gateway.save_file(png_upload_stream, png_path)
        except (IOError, OSError, TypeError, AttributeError) as e:
             return ProcessingResult(success=False, message=f"Error saving uploaded files: {e}")

        # --- Thực hiện xử lý cốt lõi ---
        try:
            # Khởi tạo processor với config
            # self.processor = SvgProcessor(max_colors=config.max_colors, color_threshold=config.threshold)
            self.processor.set_color_config(color_config)
            # Gọi hàm process của processor (phiên bản này cần path)
            # Nếu SvgProcessor được sửa để nhận bytes/content, sẽ truyền vào đây
            logging.exception(svg_path, png_path)
            
            img = Image.open(png_path).convert("RGB")
            image_data = np.array(img)
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            # print(svg_content)
            # print()
            processed_tree = self.processor.process(svg_content, image_data) # , output_svg_path)

            if not processed_tree:
                # Lỗi đã được log bên trong processor, chỉ cần trả về thất bại
                return ProcessingResult(success=False, message="SVG processing core logic failed. Check logs.")

        # except ProcessingError as core_err: # Bắt lỗi nghiệp vụ cụ thể nếu có
        #     logger.error(f"Core processing error for {album_id}: {core_err}", exc_info=True)
        #     return ProcessingResult(success=False, message=f"Processing error: {core_err}")
        except Exception as e:
            logger.exception(f"Unexpected error during core SVG processing for {album_id}: {e}") # Log exception đầy đủ
            return ProcessingResult(success=False, message=f"Unexpected processing error: {e}")


        # --- Thành công ---
        self.file_gateway.save_svg_file(processed_tree, output_svg_path)
        
        duration = time.time() - start_time
        
        
        return ProcessingResult(
            success=True,
            output_svg_path=output_svg_path,
            message=f"Processing successful in {duration:.2f} seconds.",
            duration=duration
        )

# --- Có thể thêm class/function cho Use Case khác, ví dụ: FindSvgService ---
class FindSvgService:
    def __init__(self, file_gateway: FileGateway):
        self.file_gateway = file_gateway

    def execute(self, album_id: str) -> Tuple[Optional[str], Optional[str]]:
         """Finds the output SVG and input PNG paths."""
         logger.info(f"Executing FindSvgService for Album ID: {album_id}")
         output_svg_path, found = self.file_gateway.find_output_svg(album_id)
         input_png_path = None
         if found:
             # Kiểm tra xem PNG gốc có tồn tại không
             expected_png_path = self.file_gateway.get_input_png_path(album_id)
             if self.file_gateway.check_file_exists(expected_png_path):
                 input_png_path = expected_png_path
                 logger.info(f"Found original PNG: {input_png_path}")
             else:
                  logger.warning(f"Output SVG found, but original PNG missing at {expected_png_path}")
         return output_svg_path, input_png_path # Trả về cả path PNG nếu tìm thấy