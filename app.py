# src/svg_color_tool/presentation/streamlit_app.py

import streamlit as st
import logging, os

# Import components
from svg_color.config import *
from svg_color.domain import ColorConfig
from svg_color.core import SvgProcessor
from svg_color.infrastructure import FileGateway, LoggingConfigurator
from svg_color.application import ProcessSvgService, FindSvgService
from svg_color.presentation import AppState, SidebarRenderer, ProcessTabRenderer, FindTabRenderer

# Import ultilities for UI
from svg_color.utils.ui_utils import display_svg_preview

class AppOrchestrator:
    """Main orchestrator for the Streamlit application."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Setup logging first thing!
        self._setup_logging()
        self._setup_dependencies()
        self._setup_renderers()

    def _setup_logging(self):
        """Initializes and applies logging configuration using the configurator class."""
        try:
            configurator = LoggingConfigurator(
                log_level=EFFECTIVE_LOG_LEVEL,
                log_format=LOG_FORMAT,
                date_format=LOG_DATE_FORMAT,
                log_dir=LOG_DIR,
                log_file_name=LOG_FILE_NAME,
                log_to_console=True, # Explicitly enable console logging
                log_to_file=True,    # Explicitly enable file logging
                file_encoding=DEFAULT_ENCODING,
                force_config=True,   # Important for Streamlit
                noisy_loggers_to_silence=["PIL", "shapely"] # Define noisy loggers
            )
            configurator.configure()
            self.logger.info("Logging setup complete via LoggingConfigurator.")
        except Exception as log_err:
            # Handle critical logging setup failure
            st.error(f"Fatal Error: Failed to initialize application logging: {log_err}")
            # Print to stderr as a last resort
            print(f"CRITICAL: Logging setup failed - {log_err}", file=os.sys.stderr)
            st.stop() # Stop the app if logging can't be configured

    def _setup_dependencies(self):
        """Initializes core application services and gateways."""
        # ... (Initialization of AppState, FileGateway, Services remains the same) ...
        try:
            self.app_state = AppState()
            self.file_gateway = FileGateway(base_data_dir=BASE_DATA_DIR)
            self.svg_processor = SvgProcessor(color_config=ColorConfig())
            self.process_service = ProcessSvgService(file_gateway=self.file_gateway, processor = self.svg_processor)
            self.find_service = FindSvgService(file_gateway=self.file_gateway)
            self.logger.info("Core dependencies and AppState initialized.")
        except Exception as init_err:
            st.error(f"Fatal Error initializing core components: {init_err}")
            self.logger.critical(f"Core component initialization failed: {init_err}", exc_info=True)
            st.exception(init_err)
            st.stop()


    def _setup_renderers(self):
        """Initializes the UI rendering components."""
        # ... (Initialization of renderers remains the same) ...
        self.sidebar_renderer = SidebarRenderer()
        self.process_tab_renderer = ProcessTabRenderer(self.app_state, self.process_service, self.file_gateway)
        self.find_tab_renderer = FindTabRenderer(self.app_state, self.find_service, self.file_gateway)
        self.logger.info("UI renderers initialized.")


    def run(self):
        """Sets up the page config and runs the main UI rendering flow."""
        # Page config should still be the first st command
        st.set_page_config(page_title=APP_TITLE, page_icon="🎨", layout="wide")

        # Logging is already configured in __init__

        self.logger.info("--- Streamlit AppOrchestrator Run ---")

        # --- Render UI ---
        st.title(APP_TITLE)
        max_colors, color_threshold = self.sidebar_renderer.render()
        current_config = ColorConfig(max_colors=max_colors, color_threshold=color_threshold)
        tab_process, tab_find = st.tabs(["🚀 Process New SVG", "🔍 Find Existing SVG"])

        with tab_process:
            self.process_tab_renderer.render(current_config)
        with tab_find:
            self.find_tab_renderer.render()

        self.logger.info("--- Streamlit AppOrchestrator Run Finished ---")

# --- Application Entry Point ---
if __name__ == "__main__":
    orchestrator = AppOrchestrator()
    orchestrator.run()