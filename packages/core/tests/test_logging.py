"""Tests for threetears.core.logging."""

from __future__ import annotations

import logging
from io import StringIO

import pytest

from threetears.core.logging import (
    ContextFormatter,
    clear_context,
    configure_logging,
    get_logger,
    set_context,
)


@pytest.fixture(autouse=True)
def _clean_context():
    """Ensure context is clean before and after each test."""
    clear_context()
    yield
    clear_context()


def _capture_handler() -> tuple[logging.Handler, StringIO]:
    """Create a handler that writes to a StringIO buffer."""
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(ContextFormatter(use_color=False))
    return handler, buf


def test_get_logger_returns_logger():
    logger = get_logger("test.basic")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.basic"


def test_get_logger_has_null_handler():
    logger = get_logger("test.null_handler")
    has_null = any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    assert has_null, "get_logger should add a NullHandler"


def test_get_logger_no_stream_handler():
    logger = get_logger("test.no_stream")
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler) for h in logger.handlers
    )
    assert not has_stream, "get_logger should NOT add a StreamHandler"


def test_log_format_without_context():
    logger = get_logger("test.no_ctx")
    handler, buf = _capture_handler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("hello world")
    finally:
        logger.removeHandler(handler)

    output = buf.getvalue()
    assert "[cid:-]" in output
    assert "[sid:-]" in output
    assert "[conv:-]" in output
    assert "hello world" in output


def test_log_format_with_context():
    set_context(conversation_id="abc")
    logger = get_logger("test.with_ctx")
    handler, buf = _capture_handler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("context test")
    finally:
        logger.removeHandler(handler)

    output = buf.getvalue()
    assert "[conv:abc]" in output
    assert "[cid:-]" in output
    assert "[sid:-]" in output


def test_clear_context():
    set_context(correlation_id="req-123", session_id="sess-456", conversation_id="conv-789")
    clear_context()

    logger = get_logger("test.clear")
    handler, buf = _capture_handler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("after clear")
    finally:
        logger.removeHandler(handler)

    output = buf.getvalue()
    assert "[cid:-]" in output
    assert "[sid:-]" in output
    assert "[conv:-]" in output


def test_extra_data_in_log():
    logger = get_logger("test.extra")
    handler, buf = _capture_handler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("with extras", extra={"extra_data": {"key": "value", "count": 42}})
    finally:
        logger.removeHandler(handler)

    output = buf.getvalue()
    assert "key" in output
    assert "value" in output
    assert "42" in output


def test_configure_logging_adds_handler():
    # Use a unique logger name so we don't conflict
    root = logging.getLogger("threetears")
    original_handlers = list(root.handlers)
    try:
        # Clear any existing handlers
        root.handlers.clear()
        configure_logging("DEBUG")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
    finally:
        root.handlers = original_handlers


def test_configure_logging_idempotent():
    root = logging.getLogger("threetears")
    original_handlers = list(root.handlers)
    try:
        root.handlers.clear()
        configure_logging("INFO")
        configure_logging("INFO")  # second call should be no-op
        assert len(root.handlers) == 1
    finally:
        root.handlers = original_handlers
