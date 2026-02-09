"""
Logging configuration for the SC2 Replay Extraction Pipeline.

Provides centralized logging setup for both the main process and
worker processes spawned by ProcessPoolExecutor.

On Windows, spawned worker processes do NOT inherit the parent's
logging configuration, so each worker must configure its own handlers.
Both the main process and workers write to the same timestamped log file
in append mode.
"""

import logging
from datetime import datetime
from pathlib import Path


# Log format for the main process
MAIN_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# Log format for worker processes (includes process name for traceability)
WORKER_LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(processName)s] - %(name)s - %(message)s"


def setup_logging(log_dir: str = "logs/data_pipeline") -> str:
    """
    Configure logging for the main pipeline process.

    Creates a timestamped log file and configures the root logger with:
    - A FileHandler at DEBUG level (captures everything to the log file)
    - A StreamHandler at INFO level (prints INFO+ to console)

    Args:
        log_dir: Directory to store log files. Created if it doesn't exist.

    Returns:
        The absolute path to the created log file (needed by worker processes).
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Build timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"pipeline_{timestamp}.log"
    log_file_path = str(log_file.resolve())

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if setup_logging is called more than once
    if root_logger.handlers:
        return log_file_path

    # File handler - captures DEBUG and above to the log file
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(MAIN_LOG_FORMAT))

    # Console handler - prints INFO and above to stderr
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(MAIN_LOG_FORMAT))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger(__name__).info(f"Logging initialized. Log file: {log_file_path}")

    return log_file_path


def setup_worker_logging(log_file_path: str) -> None:
    """
    Configure logging inside a worker process.

    On Windows, ProcessPoolExecutor uses the 'spawn' start method, so child
    processes have no logging handlers configured. This function adds a
    FileHandler pointing to the same log file as the main process (append mode)
    so that all log output is centralized in one file.

    Args:
        log_file_path: Absolute path to the shared log file (returned by setup_logging).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if this worker is reused by the pool
    if root_logger.handlers:
        return

    # File handler - same log file as the main process, append mode
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(WORKER_LOG_FORMAT))

    root_logger.addHandler(file_handler)
