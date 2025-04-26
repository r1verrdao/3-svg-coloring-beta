# src/svg_color_tool/presentation/sidebar_renderer.py
import streamlit as st
import os
from typing import Tuple
from svg_color.config import (BASE_DATA_DIR, LOG_DIR, LOG_FILE_NAME, INPUT_DIR_NAME,
                    OUTPUT_DIR_NAME, INPUT_PNG_SUFFIX, INPUT_SVG_SUFFIX, OUTPUT_SVG_SUFFIX)

class SidebarRenderer:
    """Handles rendering the sidebar content."""
    def render(self, default_max_colors: int = 80, default_threshold: int = 30) -> Tuple[int, float]:
        with st.sidebar:
            # ... (instructions markdown) ...
            st.header("📖 Instructions")
            st.markdown("""
            **Process New SVG:**
            1. Enter **Album ID** (6 or 10 digits).
            2. Upload **PNG** color reference.
            3. Upload **SVG** sketch file.
            4. Adjust **Parameters** below.
            5. Click **'Process SVG'**.

            **Find Existing SVG:**
            1. Go to **'🔍 Find Existing SVG'** tab.
            2. Enter **Album ID**.
            3. Click **'Find SVG'**.
            """)
            st.divider()

            st.header("⚙️ Parameters")
            max_colors = st.slider("Max Colors", 1, 256, default_max_colors, 1, key="sidebar_max_colors")
            color_threshold = st.slider("Color Threshold", 0, 100, default_threshold, 1, key="sidebar_threshold")
            st.divider()

            st.header("🗂️ File Structure")
            # ... (file structure code block) ...
            st.code(f"""
            ./{os.path.basename(BASE_DATA_DIR)}/
            └── AlbumID/
                ├── {INPUT_DIR_NAME}/ ...
                └── {OUTPUT_DIR_NAME}/ ...
            ./{os.path.basename(LOG_DIR)}/
            └── {LOG_FILE_NAME}
            """, language="text")

        return int(max_colors), float(color_threshold)