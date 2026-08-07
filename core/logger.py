"""Logging bootstrap.

Provides a single :func:`configure_logging` entry point (called once from
``main.py``) plus a :func:`get_logger` helper used by every module.  An
in-memory ring buffer handler is attached so the FastAPI panel can stream the
most recent log lines without touching the filesystem.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Final

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-34s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_RING_BUFFER_SIZE: Final[int] = 800

_CONFIGURED: bool = False


class RingBufferHandler(logging.Handler):
    """Keep the last ``capacity`` formatted records in memory for the web panel."""

    def __init__(self, capacity: int = _RING_BUFFER_SIZE) -> None:
        super().__init__()
        self.records: Deque[dict[str, Any]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(
                {
                    "timestamp": datetime.fromtimestamp(
                        record.created, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
            )
        except Exception:  # pragma: no cover - a logging handler must never raise
            self.handleError(record)

    def snapshot(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return the newest ``limit`` records, most recent first."""
        items: list[dict[str, Any]] = list(self.records)
        return list(reversed(items[-limit:]))


LOG_BUFFER: Final[RingBufferHandler] = RingBufferHandler()


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure the root logger exactly once.

    Args:
        level: Root log level name (``DEBUG`` ... ``CRITICAL``).
        log_dir: Directory for the rotating file handler.  When ``None`` only the
            stream and ring-buffer handlers are installed.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    root: logging.Logger = logging.getLogger()
    root.setLevel(level.upper())

    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    LOG_BUFFER.setFormatter(formatter)
    root.addHandler(LOG_BUFFER)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "quant_system.log",
            maxBytes=25 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Third-party libraries are extremely chatty at DEBUG level.
    for noisy in ("ccxt", "urllib3", "asyncio", "apscheduler.executors.default", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
