# svg_color/utils/ui_utils.py

import streamlit as st
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def display_svg_preview(svg_content: str, max_lines: int = 20, caption: Optional[str] = "SVG Preview"):
    """
    Displays a truncated preview of SVG code in the Streamlit UI.

    Args:
        svg_content: The SVG content as a string.
        max_lines: The maximum number of lines to display.
        caption: Optional caption to show below the code block.
    """
    if not isinstance(svg_content, str):
         st.warning("Invalid content type for SVG preview (expected string).")
         logger.warning(f"display_svg_preview received non-string type: {type(svg_content)}")
         return

    try:
        lines = svg_content.splitlines()
        preview_content = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            preview_content += "\n..." # Indicate truncation

        st.code(preview_content, language="xml")
        if caption:
             st.caption(f"{caption} (first {min(len(lines), max_lines)} lines)")

    except Exception as e:
        st.warning(f"Could not display SVG preview: {e}")
        logger.warning(f"Error displaying SVG preview string: {e}", exc_info=True)

# Add other general UI helper functions here, e.g., display_error_message, etc.