"""Shim — re-export the host bot's loguru logger.

The donor's utils/logger.py reconfigured loguru AND hijacked stdlib logging
(logging.basicConfig(force=True)) plus created a logs/ dir at import time.
Inside the host bot all logging is already set up by src/utils/logger.py,
so the action engine just shares it.
"""
from src.utils.logger import logger

__all__ = ["logger"]
