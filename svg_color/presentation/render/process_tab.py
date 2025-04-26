# src/svg_color_tool/presentation/process_tab_renderer.py

import streamlit as st
import logging
import uuid
import time
import traceback
from typing import Optional, Any

from svg_color.domain.models import ColorConfig, ProcessingResult
from svg_color.infrastructure.file_gateway import FileGateway
from svg_color.application.services import ProcessSvgService
from svg_color.presentation.app_state import AppState
from svg_color.presentation.validators import InputValidator

from svg_color.utils.ui_utils import display_svg_preview # Import UI helper
from svg_color.config import DEFAULT_ENCODING

logger = logging.getLogger(__name__)

class ProcessTabRenderer:
    """Renders and handles logic for the 'Process New SVG' tab using AppState."""

    def __init__(self,
                 app_state: AppState,
                 process_service: ProcessSvgService,
                 file_gateway: FileGateway):
        self.app_state = app_state
        self.process_service = process_service
        self.file_gateway = file_gateway # Needed for displaying results

    def render(self, config: ColorConfig):
        """Renders the tab content and handles the process button click."""
        st.header("Process New SVG Sketch")
        # svg_color.. (markdown description) svg_color..

        # --- Input Widgets ---
        # Input widgets inherently maintain their state via Streamlit's key mechanism
        album_id = st.text_input("1. Album ID", key="proc_tab_album_id", placeholder="6 or 10 digits")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2. Upload Color PNG")
            png_file = st.file_uploader("Choose PNG", type=['png'], key="proc_tab_png_uploader")
            if png_file: st.image(png_file, caption='Color PNG Preview')
        with col2:
            st.subheader("3. Upload Sketch SVG")
            svg_file = st.file_uploader("Choose SVG", type=['svg'], key="proc_tab_svg_uploader")
            # SVG Preview Logic
            svg_preview_valid = False
            if svg_file:
                try:
                    svg_bytes = svg_file.getvalue()
                    display_svg_preview(svg_bytes.decode(DEFAULT_ENCODING)) # Use helper
                    svg_file.seek(0)
                    svg_preview_valid = True
                except Exception as e:
                    st.error(f"Cannot preview SVG: {e}")
                    svg_file = None

        st.subheader("4. Processing Parameters")
        st.caption(f"Using: Max Colors={config.max_colors}, Threshold={config.color_threshold} (Set in sidebar)")

        st.subheader("5. Start Processing")
        # --- Process Button and Handler ---
        if st.button("Process SVG", key="proc_tab_button", type="primary"):
            self._handle_process_request(album_id, png_file, svg_file, config)
            # NOTE: The actual result display happens *after* the rerun, triggered below

        # --- Result Display Area ---
        # This part reads the state *after* the button handler (if clicked) has updated it
        # and Streamlit has potentially rerun the script.
        result = self.app_state.get_processing_result()
        if result: # Only display if a result exists in state
            self._display_results(result, album_id)
                                  #  if result.success else self.app_state.get_last_process_album_id()) # Pass ID used for context


    def _handle_process_request(
        self, album_id: str, png_file: Optional[Any], svg_file: Optional[Any], config: ColorConfig
    ):
        """Handles button click: validates, calls service, updates AppState."""
        # 1. Validation
        is_valid_id, id_error_msg = InputValidator.validate_album_id(album_id)
        error_messages = []
        if not is_valid_id: error_messages.append(f"Invalid Album ID: {id_error_msg}")
        if not png_file: error_messages.append("PNG file required.")
        if not svg_file: error_messages.append("Valid SVG file required.")

        if error_messages:
            for msg in error_messages: st.error(msg)
            logger.warning(f"Processing aborted for '{album_id}': {'; '.join(error_messages)}")
            # Clear any previous result state if validation fails
            self.app_state.clear_processing_result()
            return

        # 2. Execute Use Case (with status updates)
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        result: Optional[ProcessingResult] = None # Initialize result
        try:
            status_placeholder.info("🚀 Processing startedsvg_color..")
            progress_bar.progress(5, text="Calling servicesvg_color..")
            result = self.process_service.execute(album_id, svg_file, png_file, config)
            progress_bar.progress(100, text="Complete.")
            status_placeholder.empty() # Clear "Processingsvg_color.." message on completion

        except Exception as e:
            progress_bar.progress(100)
            error_id = uuid.uuid4().hex[:8]
            detailed_error = traceback.format_exc()
            status_placeholder.error(f"Unexpected application error! Error ID: {error_id}")
            logger.critical(f"Unhandled exception (Error ID: {error_id}):\n{detailed_error}")
            # Create an error result object
            result = ProcessingResult(success=False, message=f"Unexpected error (ID: {error_id}).")

        finally:
             # Ensure progress bar is removed or finalized
             # progress_bar.empty() # Not a function, progress(100) handles it
             pass

        # 3. Update AppState with the result (success or failure)
        # Store the ID used for this attempt, needed if displaying error later
        # self.app_state.set_last_process_album_id(album_id) # Need to add this to AppState
        self.app_state.set_processing_result(result)
        # Streamlit will likely rerun now, and the render method will display the result


    def _display_results(self, result: ProcessingResult, album_id_context: str):
        """Displays the processing results based on the AppState."""
        st.subheader("🎉 Result")
        if result.success:
            st.success(f"✅ Processing successful! ({result.message})")
            if result.output_svg_path:
                # Check existence using the gateway
                if self.file_gateway.check_file_exists(result.output_svg_path):
                    try:
                        output_svg_content = self.file_gateway.read_file_text(result.output_svg_path, encoding=DEFAULT_ENCODING)
                        st.download_button(
                            label="📥 Download Processed SVG", data=output_svg_content,
                            file_name=f"{album_id_context}_colored.svg", mime="image/svg+xml",
                            key="download_proc_res"
                        )
                        st.info(f"Output location: `{result.output_svg_path}`")
                        st.subheader("📊 Output SVG Preview")
                        display_svg_preview(output_svg_content, caption="Processed SVG")
                    except Exception as read_err:
                        err_msg = f"Processed OK, but failed to read/display result: {read_err}"
                        st.error(err_msg)
                        logger.error(f"Error reading output SVG '{result.output_svg_path}': {read_err}", exc_info=True)
                else:
                    st.error(f"Processing successful, but output file is missing at: `{result.output_svg_path}`")
                    logger.error(f"Output file reported but not found: {result.output_svg_path}")
            else:
                st.warning("Processing successful, but no output path was provided.")
        else:
            st.error(f"❌ Processing Failed: {result.message}")