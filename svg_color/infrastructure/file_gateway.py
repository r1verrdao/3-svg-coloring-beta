# infrastructure/file_gateway.py
import os, shutil
import logging
from typing import IO, Any, Union
from ..domain.models import ProcessingInput # Import từ domain nếu cần path chuẩn

logger = logging.getLogger(__name__)


class FileGateway: # Có thể kế thừa IFileGateway
    """Concrete implementation for interacting with the local file system."""

    def __init__(self, base_data_dir: str):
        if not base_data_dir:
            raise ValueError("Base data directory cannot be empty.")
        self.base_data_dir = base_data_dir
        try:
            os.makedirs(self.base_data_dir, exist_ok=True)
            logger.info(f"FileGateway initialized. Base data directory: '{self.base_data_dir}'")
        except OSError as e:
            logger.critical(f"Cannot create or access base data directory '{self.base_data_dir}': {e}", exc_info=True)
            raise

    def _get_path(self, *args) -> str:
        """Helper to join path components safely."""
        return os.path.join(*args)

    def get_album_base_path(self, album_id: str) -> str:
        return self._get_path(self.base_data_dir, album_id)

    def get_input_path(self, album_id: str) -> str:
        return self._get_path(self.get_album_base_path(album_id), "input")

    def get_output_path(self, album_id: str) -> str:
        return self._get_path(self.get_album_base_path(album_id), "output")

    def ensure_directories(self, album_id: str) -> None:
        """Ensures input and output directories exist for the album."""
        input_dir = self.get_input_path(album_id)
        output_dir = self.get_output_path(album_id)
        try:
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Directories ensured for {album_id}: {input_dir}, {output_dir}")
        except OSError as e:
            logger.error(f"Failed ensuring directories for {album_id}: {e}", exc_info=True)
            raise

    def save_file(self, data: Union[bytes, IO[bytes]], target_path: str) -> None:
        """Saves bytes or a file stream to the target path."""
        logger.debug(f"Attempting to save file to '{target_path}'")
        try:
            # Đảm bảo thư mục cha tồn tại
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "wb") as f:
                if isinstance(data, bytes):
                    f.write(data)
                elif hasattr(data, 'read'): # Check if it's a file-like object
                    shutil.copyfileobj(data, f)
                else:
                     raise TypeError("Data must be bytes or a file-like object with read().")
            logger.info(f"File saved successfully to '{target_path}'")
        except (IOError, OSError, TypeError) as e:
            logger.error(f"Failed to save file to '{target_path}': {e}", exc_info=True)
            raise


    def save_svg_file(self, processed_tree, target_path):
        # processed_root = processed_tree.getroot()
        try:
            processed_tree.write(target_path, encoding='utf-8', xml_declaration=True)
            logger.info(f"Successfully saved processed SVG to: {target_path}")
            print(f"Processed SVG saved to: {target_path}")

        except IOError as e:
            logger.error(f"Error writing processed SVG to file '{target_path}': {e}")
            print(f"Error saving file: {e}")



    def read_file_bytes(self, file_path: str) -> bytes:
        """Reads the entire content of a file as bytes."""
        logger.debug(f"Attempting to read bytes from '{file_path}'")
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            logger.info(f"Read {len(content)} bytes from '{file_path}'")
            return content
        except FileNotFoundError:
             logger.error(f"File not found: '{file_path}'")
             raise
        except (IOError, OSError) as e:
            logger.error(f"Failed to read bytes from '{file_path}': {e}", exc_info=True)
            raise

    def read_file_text(self, file_path: str, encoding='utf-8') -> str:
        """Reads the entire content of a file as text."""
        logger.debug(f"Attempting to read text from '{file_path}'")
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            logger.info(f"Read {len(content)} characters from '{file_path}'")
            return content
        except FileNotFoundError:
             logger.error(f"File not found: '{file_path}'")
             raise
        except (IOError, OSError, UnicodeDecodeError) as e:
            logger.error(f"Failed to read text from '{file_path}': {e}", exc_info=True)
            raise

    def check_file_exists(self, file_path: str) -> bool:
        """Checks if a file exists."""
        exists = os.path.exists(file_path)
        logger.debug(f"Existence check for '{file_path}': {exists}")
        return exists

    # --- Các hàm lấy đường dẫn chuẩn ---
    def get_input_svg_path(self, album_id: str) -> str:
        return self._get_path(self.get_input_path(album_id), f"{album_id}.svg")

    def get_input_png_path(self, album_id: str) -> str:
        return self._get_path(self.get_input_path(album_id), f"{album_id}.png")

    def get_output_svg_path(self, album_id: str) -> str:
        return self._get_path(self.get_output_path(album_id), f"{album_id}.svg")