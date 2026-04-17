"""Structured logging with context correlation and automatic call-site capture.

Library-friendly: ``get_logger`` returns a ``ThreeTearsLogger`` with a
``NullHandler`` so log output is silent unless the host application calls
``configure_logging()`` or attaches its own handlers to the ``threetears``
logger hierarchy.

Context is generic -- applications call ``set_context()`` with whatever
key-value pairs make sense for their domain.  The built-in
``ContextFormatter`` renders all active context as ``[key:value]`` pairs
in every log line.  No hardcoded field names.

The ``ThreeTearsLogger`` subclass automatically captures call-site information
(file path, class name, function name, line number) via stack inspection,
giving structured logs accurate source locations even through wrapper layers.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import UTC, datetime
from types import FrameType
from typing import Any

# ---------------------------------------------------------------------------
# Generic context -- apps set whatever keys they need
# ---------------------------------------------------------------------------

_log_context: ContextVar[dict[str, str]] = ContextVar("threetears_log_context", default={})


def set_context(**kwargs: str | None) -> None:
    """Set context values that appear in every log line.

    Any keyword argument becomes a ``[key:value]`` tag in formatted output.
    Setting a key to ``None`` removes it from context.  Non-string values
    are coerced to ``str``.

    Example::

        set_context(cid=str(correlation_id), sid=str(session_id))
        # log output: INFO ... [cid:abc-123] [sid:def-456] ...

        set_context(tenant="acme", request="req-789")
        # log output: INFO ... [tenant:acme] [request:req-789] ...
    """
    ctx = _log_context.get().copy()
    for key, value in kwargs.items():
        if value is None:
            ctx.pop(key, None)
        else:
            ctx[key] = str(value)
    _log_context.set(ctx)


def clear_context() -> None:
    """Remove all context values."""
    _log_context.set({})


def get_context() -> dict[str, str]:
    """Return a copy of the current context dict."""
    return _log_context.get().copy()


# ---------------------------------------------------------------------------
# Call-site cache -- shared by ThreeTearsLogger and ContextFormatter
# ---------------------------------------------------------------------------

_call_site_cache: dict[tuple[str, int], tuple[str, str | None, str]] = {}

# Configurable path prefixes to strip from filenames for shorter log output.
# Host apps can append to this list (e.g. ``path_strip_prefixes.append("myapp/src/")``).
path_strip_prefixes: list[str] = []


def _shorten_path(filepath: str) -> str:
    """Shorten a file path by stripping known prefixes."""
    for prefix in path_strip_prefixes:
        if prefix in filepath:
            return filepath.split(prefix)[-1]
    return os.path.basename(filepath)


# ---------------------------------------------------------------------------
# Formatter
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
    """Check whether ANSI color output is enabled."""
    env = os.environ.get("THREETEARS_LOG_COLOR", "true").lower()
    if env in ("0", "false", "no"):
        return False
    return sys.stderr.isatty()


class ContextFormatter(logging.Formatter):
    """Formats log records with context tags, call-site info, and optional color.

    Reads the generic context dict and renders each entry as ``[key:value]``.
    When used with ``ThreeTearsLogger``, also includes enriched call-site
    attributes (shortened file path, class name).

    Format::

        LEVEL TIMESTAMP [key1:val1] [key2:val2] path/Class.func.line: message {extra}
    """

    def __init__(self, use_color: bool = False) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with context tags and call-site info."""
        level = record.levelname
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            level = f"{color}{level}{_RESET}"

        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Use enriched call-site attrs if available (from ThreeTearsLogger)
        call_file = getattr(record, "call_site_file", record.pathname)
        call_class = getattr(record, "call_site_class", None)
        call_func = getattr(record, "call_site_func", record.funcName)
        call_line = getattr(record, "call_site_line", record.lineno)

        if call_class:
            location = f"{call_file}/{call_class}.{call_func}.{call_line}"
        else:
            location = f"{call_file}/{call_func}.{call_line}"

        parts = [f"{level} {ts}"]

        # Render all context tags in insertion order
        ctx = _log_context.get()
        if ctx:
            for key, value in ctx.items():
                parts.append(f"[{key}:{value}]")

        parts.append(f"{location}:")
        parts.append(record.getMessage())

        extra_data: dict[str, Any] | None = getattr(record, "extra_data", None)
        if extra_data:
            try:
                parts.append(json.dumps(extra_data, default=str))
            except TypeError, ValueError:
                parts.append(repr(extra_data))

        msg = " ".join(parts)

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            msg = f"{msg}\n{record.exc_text}"
        return msg


# ---------------------------------------------------------------------------
# ThreeTearsLogger -- custom Logger with call-site capture
# ---------------------------------------------------------------------------


class ThreeTearsLogger(logging.Logger):
    """Custom logger with automatic call-site capture.

    Overrides ``makeRecord`` to walk the stack and detect the actual call site,
    including class name detection via ``self``/``cls`` in local variables.
    Results are cached by (filename, line_number) for performance.
    """

    def makeRecord(  # noqa: N802
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: object,
        args: tuple[object, ...] | Mapping[str, object],
        exc_info: Any,
        func: str | None = None,
        extra: Mapping[str, object] | None = None,
        sinfo: str | None = None,
    ) -> logging.LogRecord:
        """Create log record with enriched call-site information."""
        call_site_line = lno
        call_site_func = func or "unknown"

        cache_key = (fn, lno)

        call_site_class: str | None = None
        call_site_file: str | None = None
        if cache_key in _call_site_cache:
            call_site_file, call_site_class, _ = _call_site_cache[cache_key]
        else:
            frame: FrameType | None = sys._getframe()
            while frame is not None:
                frame_info = frame.f_code
                if frame_info.co_filename == fn and frame.f_lineno == lno:
                    f_locals = frame.f_locals
                    if "self" in f_locals:
                        call_site_class = type(f_locals["self"]).__name__
                    elif "cls" in f_locals:
                        cls_obj = f_locals["cls"]
                        if isinstance(cls_obj, type):
                            call_site_class = cls_obj.__name__
                    break
                frame = frame.f_back

            call_site_file = _shorten_path(fn)
            _call_site_cache[cache_key] = (call_site_file, call_site_class, "")

        record = super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)

        setattr(record, "call_site_file", call_site_file)
        setattr(record, "call_site_class", call_site_class)
        setattr(record, "call_site_func", call_site_func)
        setattr(record, "call_site_line", call_site_line)

        if extra and "extra_data" in extra:
            setattr(record, "extra_data", extra["extra_data"])

        return record


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_configured_loggers: set[str] = set()


def get_logger(name: str) -> ThreeTearsLogger:
    """Return a ``ThreeTearsLogger`` for *name*.

    Adds a ``NullHandler`` so that 3tears never produces output unless the
    host application configures handlers (via ``configure_logging()`` or
    manually).  This follows the Python library logging best practice.

    The first call for a given *name* installs the ``ThreeTearsLogger`` class
    and attaches a ``NullHandler``.  Subsequent calls return the existing logger.
    """
    logging.setLoggerClass(ThreeTearsLogger)
    logger = logging.getLogger(name)
    if name not in _configured_loggers:
        _configured_loggers.add(name)
        logger.addHandler(logging.NullHandler())
    return logger  # type: ignore[return-value]


def configure_logging(
    level: str = "INFO",
    *,
    color: bool | None = None,
    strip_prefixes: list[str] | None = None,
) -> None:
    """Configure the ``threetears`` logger hierarchy with structured output.

    Call this from standalone applications or entry points that want 3tears
    log output on stderr.  Library code should NOT call this -- it is the
    host application's responsibility.

    :param level: log level name (e.g. ``"INFO"``, ``"DEBUG"``).
        Defaults to the ``THREETEARS_LOG_LEVEL`` env var, then ``"INFO"``.
    :param color: force color on/off.  ``None`` (default) auto-detects from
        the ``THREETEARS_LOG_COLOR`` env var and terminal capability.
    :param strip_prefixes: path prefixes to strip from log file paths
        (e.g. ``["myapp/src/"]``).  Appended to the global
        ``path_strip_prefixes`` list.
    """
    if strip_prefixes:
        for p in strip_prefixes:
            if p not in path_strip_prefixes:
                path_strip_prefixes.append(p)

    level_str = os.environ.get("THREETEARS_LOG_LEVEL", level).upper()
    level_val = getattr(logging, level_str, logging.INFO)

    use_color = color if color is not None else _color_enabled()

    logging.setLoggerClass(ThreeTearsLogger)

    # Configure root logger so ALL loggers get output (not just threetears.*)
    py_root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_threetears", False) for h in py_root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(ContextFormatter(use_color=use_color))
        handler._threetears = True  # type: ignore[attr-defined]
        py_root.addHandler(handler)
    py_root.setLevel(level_val)

    # Also set the threetears hierarchy level
    tt_root = logging.getLogger("threetears")
    tt_root.setLevel(level_val)


def configure_third_party_logging(name: str, level: str | None = None) -> None:
    """Attach 3tears formatter and handler to a third-party logger.

    This lets external libraries emit structured logs through the same
    handler/format as 3tears' own loggers.

    :param name: logger name (e.g. ``"langchain_openrouter"``)
    :param level: log level override; defaults to ``THREETEARS_LOG_LEVEL`` env var.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        level_str = os.environ.get("THREETEARS_LOG_LEVEL", level or "INFO").upper()
        level_val = getattr(logging, level_str, logging.INFO)

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(ContextFormatter(use_color=_color_enabled()))
        logger.addHandler(handler)
        logger.setLevel(level_val)
        logger.propagate = False


def add_filter(name: str, log_filter: logging.Filter) -> None:
    """Add a filter to an existing stdlib logger by name.

    Centralises ``logging.getLogger()`` calls so consuming code never needs
    to import :mod:`logging` directly.
    """
    logging.getLogger(name).addFilter(log_filter)
