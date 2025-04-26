# src/svg_color_tool/presentation/validators.py
import re
from typing import Tuple
import logging
from svg_color.config import ALBUM_ID_PATTERN

logger = logging.getLogger(__name__)

class InputValidator:
    """
    Provides validation methods for user inputs in the presentation layer.
    """
  
    @staticmethod
    def validate_album_id(album_id: str) -> Tuple[bool, str]:
        """
        Validates the Album ID format provided by the user.

        Args:
            album_id: The Album ID string input by the user.

        Returns:
            A tuple:
                - True if valid, False otherwise.
                - Error message or empty string.
        """
        if not album_id:
            logger.debug("Validation failed: Album ID is empty.")
            return False, "Album ID cannot be empty."

        if not re.fullmatch(ALBUM_ID_PATTERN, album_id):
            error_msg = "Album ID must contain exactly 6 or 10 digits."
            logger.debug(f"Validation failed: Album ID '{album_id}' does not match pattern. Message: {error_msg}")
            return False, error_msg

        logger.debug(f"Validation successful for Album ID: '{album_id}'")
        return True, ""