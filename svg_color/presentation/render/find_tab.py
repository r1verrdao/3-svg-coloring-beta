# src/svg_color_tool/presentation/process_tab_renderer.py

import streamlit as st
import logging
import uuid
import time
import traceback
from typing import Optional, Any
from streamlit.delta_generator import DeltaGenerator

from svg_color.domain.models import ColorConfig, ProcessingResult
from svg_color.infrastructure import FileGateway
from svg_color.application import ProcessSvgService, FindSvgService
from ..app_state import AppState # Import AppState
from ..validators import InputValidator

from svg_color.utils.ui_utils import display_svg_preview # Import UI helper
from svg_color.config import DEFAULT_ENCODING

logger = logging.getLogger(__name__)

class FindTabRenderer:
    """Renders and handles logic for the 'Find Existing SVG' tab using AppState."""

    def __init__(self, app_state: AppState, find_service: FindSvgService, file_gateway: FileGateway):
        self.app_state = app_state
        self.find_service = find_service
        self.file_gateway = file_gateway # Needed for displaying results

    def render(self):
        """Renders the tab content and handles the find button click."""
        st.header("Find Existing Processed SVG")
        # ... (markdown description) ...

        search_id = st.text_input(
            "Album ID to Search",
            key="find_tab_album_id_input",
            placeholder="Enter 6 or 10 digits"
        )

        if st.button("Find SVG", key="find_tab_button"):
            logger.info(f"Find button clicked for Search ID: {search_id}")
            # Clear previous results
            self.app_state.clear_find_results()
            # Placeholder for status updates
            status_placeholder = st.empty()
            # Call internal handler
            self._handle_find_request(search_id, status_placeholder)
            # Result display happens after rerun based on state

        # Display results based on AppState after potential rerun
        svg_path, png_path = self.app_state.get_find_results()
        last_search_id = self.app_state.get_last_find_search_id()

        if svg_path:
            self._display_find_results(svg_path, png_path, last_search_id)
        elif self.app_state.was_last_find_unsuccessful():
             st.warning(f"❌ No processed SVG found for Album ID: {last_search_id}")
             st.caption("Ensure the ID is correct and was processed.")


    def _handle_find_request(self, search_id: str, status_placeholder: DeltaGenerator):
        """Handles button click: validates, calls service, updates AppState."""
        # 1. Validation
        is_valid_id, id_error_msg = InputValidator.validate_album_id(search_id)
        if not is_valid_id:
            status_placeholder.error(f"Invalid Album ID: {id_error_msg}")
            # Clear state if validation fails
            self.app_state.clear_find_results()
            self.app_state.set_last_find_album_id(search_id) # Still store invalid ID searched? Maybe not.
            return

        # 2. Execute Use Case
        try:
            status_placeholder.info(f"🔍 Searching for Album ID: {search_id}...")
            svg_path, png_path = self.find_service.execute(search_id)
            status_placeholder.empty() # Clear searching message

            # Update AppState with results (or None if not found)
            self.app_state.set_find_results(search_id, svg_path, png_path)

        except Exception as find_err:
            error_id = uuid.uuid4().hex[:8]
            detailed_error = traceback.format_exc()
            status_placeholder.error(f"Unexpected error during search! Error ID: {error_id}")
            logger.critical(f"Unhandled exception in Find SVG handler (Error ID: {error_id}):\n{detailed_error}")
            # Clear results state on error
            self.app_state.clear_find_results()
            # Optionally store the failed search ID
            self.app_state.set_last_find_album_id(search_id)


    def _display_find_results(self, svg_path: str, png_path: Optional[str], search_id: str):
        """Displays the found SVG and PNG files based on AppState."""
        st.success(f"✅ Found processed SVG for '{search_id}'!")
        st.subheader("📄 Found SVG Details")
        st.info(f"File Path: `{svg_path}`")

        # Display SVG
        try:
            svg_content = self.file_gateway.read_file_text(svg_path, encoding=DEFAULT_ENCODING)
            st.download_button("📥 Download Found SVG", svg_content, f"{search_id}_colored.svg", "image/svg+xml", key="download_find_res")
            st.subheader("📊 SVG Preview")
            display_svg_preview(svg_content, caption="Found SVG")
        except FileNotFoundError:
            st.error(f"SVG path found, but file missing at: `{svg_path}`")
        except Exception as read_err:
            st.error(f"Found SVG path, but failed to read/display: {read_err}")
            logger.error(f"Error reading/displaying found SVG '{svg_path}': {read_err}", exc_info=True)

        # Display PNG
        st.subheader("🖼️ Original PNG Image")
        if png_path and self.file_gateway.check_file_exists(png_path):
            try:
                st.image(png_path, caption=f"Original PNG for {search_id}", use_column_width='auto')
            except Exception as img_disp_e:
                st.warning(f"Could not display original PNG '{png_path}': {img_disp_e}")
                logger.error(f"Error displaying found PNG '{png_path}': {img_disp_e}", exc_info=True)
        else:
            st.caption("Original PNG file not found.")