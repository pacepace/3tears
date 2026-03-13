"""Structured logging with optional context injection via contextvars."""

from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from typing import Any

# ---------------------------------------------------------------------------
# Context variables – importable by other modules (e.g. tracing)
# ---------------------------------------------------------------------------

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_conversation_id: ContextVar[str | None] = ContextVar("conversation_id", default=None)


def set_context(**kwargs: str | None) -> None:
    """Set one or more context variables.

    Supported keys: ``correlation_id``, ``session_id``, ``conversation_id``.
    Values may be strings or objects with ``__str__`` (e.g. ``uuid.UUID``).
    """
    _var_map: dict[str, ContextVar[str | None]] = {
        "correlation_id": _correlation_id,
        "session_id": _session_id,
        "conversation_id": _conversation_id,
    }
    for key, value in kwargs.items():
        var = _var_map.get(key)
        if var is None:
            raise ValueError(f"Unknown context key: {key!r}. Valid keys: {sorted(_var_map)}")
        var.set(str(value) if value is not None else None)


def clear_context() -> None:
    """Reset all context variables to their defaults."""
    _correlation_id.set(None)
    _session_id.set(None)
    _conversation_id.set(None)


# ---------------------------------------------------------------------------
# Color support
# ---------------------------------------------------------------------------

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


def _color_enabled() -> bool:
    env = os.environ.get("THREETEARS_LOG_COLOR", "true").lower()
    if env in ("0", "false", "no"):
        return False
    return sys.stderr.isatty()


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class _ContextFormatter(logging.Formatter):
    """Formats log records with context IDs and call-site info."""

    def __init__(self, use_color: bool = False) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with context IDs and call-site info."""
        cid = _correlation_id.get() or "-"
        sid = _session_id.get() or "-"
        conv = _conversation_id.get() or "-"

        # Call-site info – set by _ThreeTearsLogger._log
        path = getattr(record, "call_path", record.pathname)
        func = getattr(record, "call_func", record.funcName)
        lineno = getattr(record, "call_lineno", record.lineno)

        level = record.levelname
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            level = f"{color}{level}{_RESET}"

        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        parts = [
            f"{level} {ts}",
            f"[cid:{cid}]",
            f"[sid:{sid}]",
            f"[conv:{conv}]",
            f"{path}/{func}.{lineno}:",
            record.getMessage(),
        ]

        extra_data: dict[str, Any] | None = getattr(record, "extra_data", None)
        if extra_data:
            parts.append(str(extra_data))

        msg = " ".join(parts)

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            msg = f"{msg}\n{record.exc_text}"
        return msg


# ---------------------------------------------------------------------------
# Custom logger class
# ---------------------------------------------------------------------------


class _ThreeTearsLogger(logging.Logger):
    """Logger that captures the *caller's* call site, not the logging internals."""

    def _log(  # type: ignore[override]
        self,
        level: int,
        msg: object,
        args: Any,
        exc_info: Any = None,
        extra: dict[str, Any] | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        # Find the caller frame – we need to skip *our* frame plus the
        # public method frame (info/debug/etc.).  The stdlib already adds
        # stacklevel, so we just pass it through and additionally record
        # our own call-site attributes for the formatter.
        import inspect

        frame = inspect.currentframe()
        # Walk up stacklevel + 1 frames (our _log + caller's info/debug/etc.)
        for _ in range(stacklevel + 1):
            if frame is not None:
                frame = frame.f_back

        if extra is None:
            extra = {}

        if frame is not None:
            extra["call_path"] = frame.f_code.co_filename
            extra["call_func"] = frame.f_code.co_name
            extra["call_lineno"] = frame.f_lineno

        super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info, stacklevel=stacklevel)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Install our custom logger class *before* any loggers are created.
logging.setLoggerClass(_ThreeTearsLogger)

_configured_handlers: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*.

    Each logger gets a stderr handler with the structured context formatter.
    The log level is controlled by the ``THREETEARS_LOG_LEVEL`` env var
    (default ``INFO``).
    """
    logger = logging.getLogger(name)

    if name not in _configured_handlers:
        _configured_handlers.add(name)

        level_name = os.environ.get("THREETEARS_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_ContextFormatter(use_color=_color_enabled()))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
