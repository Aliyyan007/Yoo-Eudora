"""Logger setup using loguru with file rotation and console output."""
import sys
from loguru import logger
from pathlib import Path


def setup_logger(level: str = "INFO", log_file: str = "logs/bot.log",
                 max_size_mb: int = 10, backup_count: int = 3):
    """Configure loguru logger with console + rotating file output."""
    logger.remove()

    # Console output with colorized format
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        colorize=True,
    )

    # File output with rotation
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=f"{max_size_mb} MB",
        retention=backup_count,
        encoding="utf-8",
    )

    return logger
