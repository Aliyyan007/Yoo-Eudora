"""Vendored logger shim — re-exports loguru's ``logger``.

The source repo's ``utils/logger.py`` also configured loguru sinks and
bridged stdlib logging; here the host application owns logging setup, so
this module is just the shared import target for the vendored code.
"""
from __future__ import annotations

from loguru import logger

__all__ = ["logger"]
