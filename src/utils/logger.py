"""
Structured logging configuration using loguru.
All modules import from here for consistent log formatting.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str = "outputs/logs", level: str = "INFO") -> None:
    """Configure loguru with file + console handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler (colored)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    # File handler (JSON-structured for analysis)
    logger.add(
        f"{log_dir}/run_{{time:YYYYMMDD_HHmmss}}.log",
        format="{time} | {level} | {name}:{function} | {message}",
        level="DEBUG",
        rotation="100 MB",
        compression="zip",
    )
    
    logger.info("Logger initialized.")


# Export the configured logger
__all__ = ["logger", "setup_logger"]