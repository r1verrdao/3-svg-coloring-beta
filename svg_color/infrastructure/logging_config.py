# src/svg_color_tool/infrastructure/logging_config.py

import logging
import os
from typing import List, Optional

# Không import trực tiếp từ config ở đây để giữ lớp độc lập hơn
# Thay vào đó, nhận giá trị cấu hình qua __init__

class LoggingConfigurator:
    """
    Configures the Python standard logging system for the application.

    Takes configuration parameters during initialization and applies them
    via logging.basicConfig and other logging settings.
    """

    def __init__(self,
                 log_level: int,
                 log_format: str,
                 date_format: str,
                 log_dir: str,
                 log_file_name: str,
                 log_to_console: bool = True,
                 log_to_file: bool = True,
                 file_encoding: str = 'utf-8',
                 force_config: bool = True,
                 noisy_loggers_to_silence: Optional[List[str]] = None):
        """
        Initializes the LoggingConfigurator with specific settings.

        Args:
            log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
            log_format: The format string for log messages.
            date_format: The format string for the timestamp in logs.
            log_dir: The directory where log files should be stored.
            log_file_name: The name of the log file.
            log_to_console: Whether to output logs to the console (StreamHandler).
            log_to_file: Whether to output logs to a file (FileHandler).
            file_encoding: Encoding to use for the log file.
            force_config: Whether to force override existing logging config (useful for Streamlit).
            noisy_loggers_to_silence: A list of logger names to set to WARNING level.
        """
        self.log_level = log_level
        self.log_format = log_format
        self.date_format = date_format
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.file_encoding = file_encoding
        self.force_config = force_config
        self.noisy_loggers = noisy_loggers_to_silence or []
        self.log_file_path = os.path.join(self.log_dir, self.log_file_name) if self.log_to_file else None

        # Use a basic logger temporarily for setup messages if needed
        self._setup_logger = logging.getLogger(self.__class__.__name__ + ".Setup")

    def _create_handlers(self) -> List[logging.Handler]:
        """Creates and configures logging handlers based on settings."""
        handlers = []

        # File Handler
        if self.log_to_file and self.log_file_path:
            try:
                # Ensure the log directory exists
                os.makedirs(self.log_dir, exist_ok=True)
                file_handler = logging.FileHandler(self.log_file_path, encoding=self.file_encoding)
                # Optional: Set formatter per handler if not using basicConfig defaults
                # formatter = logging.Formatter(self.log_format, self.date_format)
                # file_handler.setFormatter(formatter)
                handlers.append(file_handler)
                self._setup_logger.debug(f"File handler configured for: {self.log_file_path}")
            except OSError as e:
                # Log error but don't necessarily stop the app, maybe console logging still works
                self._setup_logger.error(f"Failed to create log directory or file handler for '{self.log_file_path}': {e}", exc_info=True)
                # Use print as ultimate fallback if logging itself fails
                print(f"[ERROR] LoggingConfigurator: Failed to create file handler for '{self.log_file_path}': {e}", file=os.sys.stderr)


        # Console Handler
        if self.log_to_console:
            stream_handler = logging.StreamHandler() # Defaults to sys.stderr
            # Optional: Set formatter per handler
            # formatter = logging.Formatter(self.log_format, self.date_format)
            # stream_handler.setFormatter(formatter)
            handlers.append(stream_handler)
            self._setup_logger.debug("Stream handler configured.")

        return handlers

    def silence_noisy_loggers(self):
        """Sets the logging level for specified noisy loggers to WARNING."""
        if not self.noisy_loggers:
            return
        self._setup_logger.debug(f"Silencing noisy loggers: {self.noisy_loggers}")
        for logger_name in self.noisy_loggers:
            try:
                logging.getLogger(logger_name).setLevel(logging.WARNING)
            except Exception as e:
                 # Log warning if silencing fails for a specific logger
                self._setup_logger.warning(f"Could not set level for logger '{logger_name}': {e}")


    def configure(self):
        """Applies the logging configuration using logging.basicConfig."""
        handlers = self._create_handlers()
        if not handlers:
            self._setup_logger.warning("No logging handlers were configured.")
            # Use print as ultimate fallback
            print("[WARNING] LoggingConfigurator: No handlers created, logging inactive.", file=os.sys.stderr)
            return # Nothing more to do

        try:
            logging.basicConfig(
                level=self.log_level,
                format=self.log_format,
                datefmt=self.date_format,
                handlers=handlers,
                force=self.force_config # Override existing Streamlit/other handlers
            )

            # Use the configured root logger for confirmation message
            root_logger = logging.getLogger()
            root_logger.info(f"Logging configured: Level={logging.getLevelName(self.log_level)}. "
                             f"Handlers={len(handlers)} ({'File' if self.log_to_file else ''}{' & ' if self.log_to_file and self.log_to_console else ''}{'Console' if self.log_to_console else ''})")
            if self.log_to_file and self.log_file_path:
                 root_logger.info(f"Log file: {self.log_file_path}")

            # Apply silencing after basicConfig
            self.silence_noisy_loggers()

        except Exception as e:
            # Fallback print if basicConfig fails
            print(f"[CRITICAL] LoggingConfigurator: Failed to configure logging via basicConfig: {e}", file=os.sys.stderr)
            # Optionally re-raise a custom exception
            raise RuntimeError(f"Logging configuration failed: {e}") from e