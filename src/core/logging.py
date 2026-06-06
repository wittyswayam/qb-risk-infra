"""Structured logging configuration using Python's logging module."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON, compatible with log aggregators."""

    RESERVED = {"message", "asctime", "levelname", "name", "exc_info", "exc_text"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        # Attach extra fields injected via LoggerAdapter or extra={...}
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in self.RESERVED:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%dT%H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_file: str | None = None,
) -> None:
    """Initialise root logger. Call once at process startup.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        fmt: 'json' for structured logs or 'text' for human-readable output.
        log_file: Optional path to write logs alongside stdout.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any handlers that may have been added implicitly.
    root.handlers.clear()

    formatter: logging.Formatter = (
        JSONFormatter() if fmt == "json" else TextFormatter()
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Suppress noisy third-party libraries at WARNING level.
    for noisy in ("urllib3", "httpx", "sqlalchemy.engine", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — preferred over direct logging.getLogger calls."""
    return logging.getLogger(name)
