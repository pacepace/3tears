"""Structured logging for 3tears.

Library-friendly: ``get_logger`` returns a standard ``logging.Logger`` with a
``NullHandler`` so log output is silent unless the host application configures
handlers.  Host apps call ``configure_logging()`` (or attach their own
handlers to the ``threetears`` logger hierarchy) to see output.

Context variables (``set_context`` / ``clear_context``) are available for
correlation IDs — the built-in ``ContextFormatter`` reads them, and host
formatters can too via the public ContextVar instances.
"""

from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from typing import Any

# ---------------------------------------------------------------------------
# Context variables – importable by host applications and formatters
# ---------------------------------------------------------------------------

correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
conversation_id: ContextVar[str | None] = ContextVar("conversation_id", default=None)


def set_context(**kwargs: str | None) -> None:
    """Set one or more context variables.

    Supported keys: ``correlation_id``, ``session_id``, ``conversation_id``.
    """
    _var_map: dict[str, ContextVar[str | None]] = {
        "correlation_id": correlation_id,
        "session_id": session_id,
        "conversation_id": conversation_id,
    }
    for key, value in kwargs.items():
        var = _var_map.get(key)
        if var is None:
            raise ValueError(f"Unknown context key: {key!r}. Valid keys: {sorted(_var_map)}")
        var.set(str(value) if value is not None else None)


def clear_context() -> None:
    """Reset all context variables to their defaults."""
    correlation_id.set(None)
    session_id.set(None)
    conversation_id.set(None)


# ---------------------------------------------------------------------------
# Formatter (available for standalone use or host app adoption)
# ---------------------------------------------------------------------------

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",      # cyan
    logging.INFO: "\033[32m",       # green
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[31m",      # red
    logging.CRITICAL: "\033[1;31m", # bold red
}
_RESET = "\033[0m"


def _color_enabled() -> bool:
    """Check whether ANSI color output is enabled."""
    env = os.environ.get("THREETEARS_LOG_COLOR", "true").lower()
    if env in ("0", "false", "no"):
        return False
    return sys.stderr.isatty()


class ContextFormatter(logging.Formatter):
    """Formats log records with context IDs and call-site info.

    Reads ``correlation_id``, ``session_id``, and ``conversation_id`` from
    the module-level ContextVars.  Host apps that populate those vars (or
    map their own vars via ``set_context``) get structured context in every
    3tears log line.
    """

    def __init__(self, use_color: bool = False) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with context IDs and call-site info."""
        cid = correlation_id.get() or "-"
        sid = session_id.get() or "-"
        conv = conversation_id.get() or "-"

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
            f"{record.pathname}/{record.funcName}.{record.lineno}:",
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
# Public API
# ---------------------------------------------------------------------------

_null_handlers_added: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name*.

    Adds a ``NullHandler`` so that 3tears never produces output unless the
    host application configures handlers on the ``threetears`` hierarchy.
    This follows the Python library logging best practice.
    """
    logger = logging.getLogger(name)
    if name not in _null_handlers_added:
        _null_handlers_added.add(name)
        logger.addHandler(logging.NullHandler())
    return logger


def configure_logging(level: str = "INFO") -> None:
    """Configure the ``threetears`` logger hierarchy with structured output.

    Call this from standalone applications or test scripts that want 3tears
    log output on stderr.  Host apps like MetaLLM should NOT call this —
    they configure the ``threetears`` hierarchy with their own handlers.
    """
    root = logging.getLogger("threetears")
    if root.handlers:
        return  # already configured

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ContextFormatter(use_color=_color_enabled()))
    root.addHandler(handler)

    level_val = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(level_val)
