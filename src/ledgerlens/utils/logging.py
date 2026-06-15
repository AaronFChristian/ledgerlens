import sys
from pathlib import Path
from loguru import logger

def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{line}</cyan> — <level>{message}</level>",
        colorize=True,
    )

__all__ = ["setup_logging", "logger"]
