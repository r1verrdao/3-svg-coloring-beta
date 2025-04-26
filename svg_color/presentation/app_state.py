# src/svg_color_tool/presentation/app_state.py

import streamlit as st
from typing import Optional, Any, Tuple
import logging

from svg_color.domain.models import ProcessingResult

logger = logging.getLogger(__name__)

class AppState:
    """
    Manages the application's state using Streamlit's session state.

    Provides type-safe accessors and mutators, centralizing session state keys.
    """
    # --- Constants for Session State Keys ---
    _KEY_PROCESS_ALBUM_ID = "appstate_process_album_id"
    _KEY_FIND_ALBUM_ID = "appstate_find_album_id" # Stores the ID used in the last Find attempt
    _KEY_PROCESS_RESULT = "appstate_process_result"
    _KEY_FIND_RESULT_SVG_PATH = "appstate_find_svg_path"
    _KEY_FIND_RESULT_PNG_PATH = "appstate_find_png_path"
    # Flag to know if the "Find" button was clicked but resulted in no file found
    _KEY_FIND_ATTEMPTED_NOT_FOUND = "appstate_find_not_found_flag"

    def __init__(self):
        """Initializes default values in session state if they don't exist."""
        st.session_state.setdefault(self._KEY_PROCESS_ALBUM_ID, "")
        st.session_state.setdefault(self._KEY_FIND_ALBUM_ID, "")
        st.session_state.setdefault(self._KEY_PROCESS_RESULT, None)
        st.session_state.setdefault(self._KEY_FIND_RESULT_SVG_PATH, None)
        st.session_state.setdefault(self._KEY_FIND_RESULT_PNG_PATH, None)
        st.session_state.setdefault(self._KEY_FIND_ATTEMPTED_NOT_FOUND, False)
        logger.debug("AppState initialized.")

    # --- Getters ---
    def get_last_find_search_id(self) -> str:
        """Gets the Album ID used in the last 'Find' operation."""
        return st.session_state.get(self._KEY_FIND_ALBUM_ID, "")

    def get_processing_result(self) -> Optional[ProcessingResult]:
        """Gets the result object from the last 'Process' operation."""
        result = st.session_state.get(self._KEY_PROCESS_RESULT)
        # Optional: Add type check for robustness
        if result is not None and not isinstance(result, ProcessingResult):
             logger.warning(f"Session state for process result is not of type ProcessingResult: {type(result)}")
             return None
        return result

    def get_find_results(self) -> Tuple[Optional[str], Optional[str]]:
        """Gets the SVG and PNG paths from the last 'Find' operation."""
        svg_path = st.session_state.get(self._KEY_FIND_RESULT_SVG_PATH)
        png_path = st.session_state.get(self._KEY_FIND_RESULT_PNG_PATH)
        return svg_path, png_path

    def was_last_find_unsuccessful(self) -> bool:
        """Checks if the last 'Find' operation was attempted but found nothing."""
        return st.session_state.get(self._KEY_FIND_ATTEMPTED_NOT_FOUND, False)

    # --- Setters / Mutators ---
    def set_processing_result(self, result: Optional[ProcessingResult]):
        """Sets the result of a 'Process' operation."""
        # Optional: Validate type before setting
        if result is not None and not isinstance(result, ProcessingResult):
             logger.error(f"Attempted to set invalid type for processing result: {type(result)}")
             st.session_state[self._KEY_PROCESS_RESULT] = ProcessingResult(success=False, message="Internal state error.")
        else:
             st.session_state[self._KEY_PROCESS_RESULT] = result
        # Clear find results when a new process result is set
        self.clear_find_results()
        logger.debug(f"Processing result updated in state: Success={result.success if result else 'None'}")

    def set_find_results(self, search_id: str, svg_path: Optional[str], png_path: Optional[str]):
        """Sets the results of a 'Find' operation."""
        st.session_state[self._KEY_FIND_ALBUM_ID] = search_id
        st.session_state[self._KEY_FIND_RESULT_SVG_PATH] = svg_path
        st.session_state[self._KEY_FIND_RESULT_PNG_PATH] = png_path
        st.session_state[self._KEY_FIND_ATTEMPTED_NOT_FOUND] = (svg_path is None) # Set flag based on result
        # Clear process result when a new find is performed
        self.clear_processing_result()
        logger.debug(f"Find results updated in state for ID '{search_id}': SVG found={svg_path is not None}")

    def clear_processing_result(self):
        """Clears the stored processing result."""
        if self._KEY_PROCESS_RESULT in st.session_state:
            st.session_state[self._KEY_PROCESS_RESULT] = None
            logger.debug("Cleared processing result from state.")

    def clear_find_results(self):
        """Clears the stored find results and the associated search ID."""
        if self._KEY_FIND_RESULT_SVG_PATH in st.session_state:
            st.session_state[self._KEY_FIND_RESULT_SVG_PATH] = None
        if self._KEY_FIND_RESULT_PNG_PATH in st.session_state:
            st.session_state[self._KEY_FIND_RESULT_PNG_PATH] = None
        if self._KEY_FIND_ALBUM_ID in st.session_state:
            st.session_state[self._KEY_FIND_ALBUM_ID] = "" # Reset last searched ID
        if self._KEY_FIND_ATTEMPTED_NOT_FOUND in st.session_state:
             st.session_state[self._KEY_FIND_ATTEMPTED_NOT_FOUND] = False
        logger.debug("Cleared find results from state.")