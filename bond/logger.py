# bond/logger.py
import logging
import sys
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not available
    pass

def setup_logger():
    """Sets up a standardized logger for the BOND project."""
    logger = logging.getLogger("bond")
    
    # Set log level from environment variable
    log_level = os.getenv("BOND_LOG_LEVEL", "WARNING").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logger()
