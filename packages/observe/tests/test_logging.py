"""Tests for threetears.observe.logging."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from threetears.observe.logging import (
    ContextFormatter,
    ThreeTearsLogger,
    _call_site_cache,
    _shorten_path,
    clear_context,
    configure_logging,
    configure_third_party_logging,
    get_context,
    get_logger,
    path_strip_prefixes,
    set_context,
)


@pytest.fixture(autouse=True)
def _clean_context():
    """Reset context and call-site cache between tests."""
    clear_context()
    _call_site_cache.clear()
    yield
    clear_context()
    _call_site_cache.clear()


class TestContext:
    """Generic context get/set/clear."""

    def test_set_and_read(self):
        set_context(cid="req-123", sid="sess-456")
        ctx = get_context()
        assert ctx["cid"] == "req-123"
        assert ctx["sid"] == "sess-456"

    def test_arbitrary_keys(self):
        set_context(tenant="acme", request="req-789", custom_field="value")
        ctx = get_context()
        assert ctx == {"tenant": "acme", "request": "req-789", "custom_field": "value"}

    def test_clear_context(self):
        set_context(cid="a", sid="b", conv="c")
        clear_context()
        assert get_context() == {}

    def test_set_none_removes_key(self):
        set_context(cid="a", sid="b")
        set_context(cid=None)
        ctx = get_context()
        assert "cid" not in ctx
        assert ctx["sid"] == "b"

    def test_non_string_coerced(self):
        set_context(count=42)  # type: ignore[arg-type]
        assert get_context()["count"] == "42"

    def test_get_context_returns_copy(self):
        set_context(cid="a")
        ctx = get_context()
        ctx["cid"] = "mutated"
        assert get_context()["cid"] == "a"

    def test_set_context_merges(self):
        set_context(cid="a")
        set_context(sid="b")
        ctx = get_context()
        assert ctx == {"cid": "a", "sid": "b"}


class TestContextFormatter:
    """ContextFormatter output format."""

    def test_format_with_context(self):
        set_context(cid="cid-1", sid="sid-2", conv="conv-3")
        formatter = ContextFormatter(use_color=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "[cid:cid-1]" in output
        assert "[sid:sid-2]" in output
        assert "[conv:conv-3]" in output
        assert "hello" in output

    def test_format_without_context(self):
        formatter = ContextFormatter(use_color=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        # No context tags when context is empty
        assert "[" not in output.split("Z]")[0] if "Z]" in output else True
        assert "hello" in output

    def test_format_with_color(self):
        formatter = ContextFormatter(use_color=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "\033[32m" in output  # green for INFO

    def test_format_with_extra_data(self):
        formatter = ContextFormatter(use_color=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.extra_data = {"key": "value"}  # type: ignore[attr-defined]
        output = formatter.format(record)
        assert '"key"' in output
        assert '"value"' in output

    def test_format_with_call_site_class(self):
        formatter = ContextFormatter(use_color=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.call_site_file = "handler.py"  # type: ignore[attr-defined]
        record.call_site_class = "MyHandler"  # type: ignore[attr-defined]
        record.call_site_func = "handle"  # type: ignore[attr-defined]
        record.call_site_line = 42  # type: ignore[attr-defined]
        output = formatter.format(record)
        assert "handler.py/MyHandler.handle.42" in output

    def test_format_with_exception(self):
        formatter = ContextFormatter(use_color=False)
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="error",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        assert "ValueError: boom" in output

    def test_format_renders_custom_keys(self):
        set_context(tenant="acme", trace_id="t-123")
        formatter = ContextFormatter(use_color=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "[tenant:acme]" in output
        assert "[trace_id:t-123]" in output


class TestThreeTearsLogger:
    """ThreeTearsLogger call-site capture."""

    def test_logger_is_correct_class(self):
        logger = get_logger("test.threetears_logger")
        assert isinstance(logger, ThreeTearsLogger)

    def test_call_site_capture(self):
        logger = get_logger("test.call_site")
        handler = logging.Handler()
        records: list[logging.LogRecord] = []
        handler.emit = lambda r: records.append(r)  # type: ignore[assignment]
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("test message")

        assert len(records) == 1
        record = records[0]
        assert hasattr(record, "call_site_file")
        assert hasattr(record, "call_site_func")
        assert hasattr(record, "call_site_line")
        assert record.call_site_func == "test_call_site_capture"  # type: ignore[attr-defined]

        logger.removeHandler(handler)

    def test_extra_data_passthrough(self):
        logger = get_logger("test.extra_data")
        handler = logging.Handler()
        records: list[logging.LogRecord] = []
        handler.emit = lambda r: records.append(r)  # type: ignore[assignment]
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("test", extra={"extra_data": {"foo": "bar"}})

        assert len(records) == 1
        assert records[0].extra_data == {"foo": "bar"}  # type: ignore[attr-defined]

        logger.removeHandler(handler)

    def test_class_detection(self):
        logger = get_logger("test.class_detect")
        handler = logging.Handler()
        records: list[logging.LogRecord] = []
        handler.emit = lambda r: records.append(r)  # type: ignore[assignment]
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        class MyClass:
            def do_thing(self):
                logger.info("from method")

        MyClass().do_thing()

        assert len(records) == 1
        assert records[0].call_site_class == "MyClass"  # type: ignore[attr-defined]

        logger.removeHandler(handler)


class TestPathShortening:
    """File path shortening logic."""

    def test_strips_configured_prefix(self):
        path_strip_prefixes.append("myapp/src/")
        try:
            assert _shorten_path("/home/user/myapp/src/handlers/ws.py") == "handlers/ws.py"
        finally:
            path_strip_prefixes.remove("myapp/src/")

    def test_falls_back_to_basename(self):
        assert _shorten_path("/some/deep/path/module.py") == "module.py"


class TestConfigureLogging:
    """configure_logging() and configure_third_party_logging()."""

    def test_configure_logging_creates_handler(self):
        py_root = logging.getLogger()
        tt_root = logging.getLogger("threetears")
        original_py_handlers = py_root.handlers[:]
        original_tt_level = tt_root.level
        try:
            # Remove any existing threetears handlers
            py_root.handlers = [h for h in py_root.handlers if not getattr(h, "_threetears", False)]
            configure_logging(level="DEBUG", color=False)
            tt_handlers = [h for h in py_root.handlers if getattr(h, "_threetears", False)]
            assert len(tt_handlers) == 1
            assert isinstance(tt_handlers[0].formatter, ContextFormatter)
            assert tt_root.level == logging.DEBUG
        finally:
            py_root.handlers = original_py_handlers
            tt_root.level = original_tt_level

    def test_configure_logging_idempotent(self):
        py_root = logging.getLogger()
        tt_root = logging.getLogger("threetears")
        original_py_handlers = py_root.handlers[:]
        original_tt_level = tt_root.level
        try:
            py_root.handlers = [h for h in py_root.handlers if not getattr(h, "_threetears", False)]
            configure_logging(level="INFO", color=False)
            tt_handler_count = len([h for h in py_root.handlers if getattr(h, "_threetears", False)])
            configure_logging(level="DEBUG", color=False)
            new_tt_handler_count = len([h for h in py_root.handlers if getattr(h, "_threetears", False)])
            assert new_tt_handler_count == tt_handler_count  # no new handler
        finally:
            py_root.handlers = original_py_handlers
            tt_root.level = original_tt_level

    def test_configure_logging_with_strip_prefixes(self):
        original = path_strip_prefixes[:]
        try:
            configure_logging(strip_prefixes=["custom/prefix/"])
            assert "custom/prefix/" in path_strip_prefixes
        finally:
            path_strip_prefixes[:] = original

    def test_configure_third_party_logging(self):
        logger_name = "test.third_party_unique"
        logger = logging.getLogger(logger_name)
        original_handlers = logger.handlers[:]
        try:
            logger.handlers.clear()
            configure_third_party_logging(logger_name, level="WARNING")
            assert len(logger.handlers) == 1
            assert logger.level == logging.WARNING
            assert logger.propagate is False
        finally:
            logger.handlers = original_handlers

    @patch.dict("os.environ", {"THREETEARS_LOG_LEVEL": "WARNING"})
    def test_env_var_override(self):
        root = logging.getLogger("threetears")
        original_handlers = root.handlers[:]
        try:
            root.handlers.clear()
            configure_logging(color=False)
            assert root.level == logging.WARNING
        finally:
            root.handlers = original_handlers
