"""Tests for threetears.observe.logging (structured logging)."""

from __future__ import annotations

import logging
from io import StringIO

import pytest

from threetears.observe.logging import (
    ContextFormatter,
    clear_context,
    configure_logging,
    get_context,
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
    # No context tags when context is empty
    assert "hello world" in output
    assert "[" not in output.split("Z ")[1].split(":")[0] if "Z " in output else True


def test_log_format_with_context():
    set_context(cid="abc", sid="def", conv="ghi")
    logger = get_logger("test.with_ctx")
    handler, buf = _capture_handler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("context test")
    finally:
        logger.removeHandler(handler)

    output = buf.getvalue()
    assert "[cid:abc]" in output
    assert "[sid:def]" in output
    assert "[conv:ghi]" in output


def test_clear_context():
    set_context(cid="req-123", sid="sess-456", conv="conv-789")
    clear_context()

    assert get_context() == {}

    logger = get_logger("test.clear")
    handler, buf = _capture_handler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("after clear")
    finally:
        logger.removeHandler(handler)

    output = buf.getvalue()
    # No context tags after clear
    assert "after clear" in output


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
    py_root = logging.getLogger()
    original_handlers = list(py_root.handlers)
    try:
        py_root.handlers = [h for h in py_root.handlers if not getattr(h, "threetears_owned", False)]
        configure_logging("DEBUG")
        tt_handlers = [h for h in py_root.handlers if getattr(h, "threetears_owned", False)]
        assert len(tt_handlers) == 1
        assert isinstance(tt_handlers[0], logging.StreamHandler)
    finally:
        py_root.handlers = original_handlers


def test_configure_logging_idempotent():
    py_root = logging.getLogger()
    original_handlers = list(py_root.handlers)
    try:
        py_root.handlers = [h for h in py_root.handlers if not getattr(h, "threetears_owned", False)]
        configure_logging("INFO")
        configure_logging("INFO")  # second call should be no-op
        tt_handlers = [h for h in py_root.handlers if getattr(h, "threetears_owned", False)]
        assert len(tt_handlers) == 1
    finally:
        py_root.handlers = original_handlers


def test_context_shared_across_calls():
    """Context set via set_context is visible in get_context."""
    set_context(cid="shared-123")
    assert get_context()["cid"] == "shared-123"
