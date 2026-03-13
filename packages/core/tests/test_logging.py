"""Tests for threetears.core.logging."""

from __future__ import annotations

import logging
from io import StringIO

import pytest

from threetears.core.logging import clear_context, get_logger, set_context


@pytest.fixture(autouse=True)
def _clean_context():
    """Ensure context is clean before and after each test."""
    clear_context()
    yield
    clear_context()


def _capture_handler() -> tuple[logging.Handler, StringIO]:
    """Create a handler that writes to a StringIO buffer."""
    from threetears.core.logging import _ContextFormatter

    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_ContextFormatter(use_color=False))
    return handler, buf


def test_get_logger_returns_logger():
    logger = get_logger("test.basic")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.basic"


def test_log_format_without_context():
    logger = get_logger("test.no_ctx")
    handler, buf = _capture_handler()
    logger.addHandler(handler)
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
    try:
        logger.info("with extras", extra={"extra_data": {"key": "value", "count": 42}})
    finally:
        logger.removeHandler(handler)

    output = buf.getvalue()
    assert "key" in output
    assert "value" in output
    assert "42" in output
