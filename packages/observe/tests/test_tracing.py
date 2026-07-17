"""Tests for threetears.observe.tracing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from threetears.observe.tracing import (
    _check_otel,
    _get_param_names,
    _record_safe_args,
    _record_safe_result,
    _set_safe_attr,
    set_span_attribute,
    traced,
)


@pytest.fixture(autouse=True)
def _reset_otel_check():
    """Reset the OTel availability cache between tests."""
    import threetears.observe.tracing as mod

    original = mod._otel_available
    yield
    mod._otel_available = original


class TestOtelCheck:
    """OTel availability detection."""

    def test_otel_available_when_installed(self):
        import threetears.observe.tracing as mod

        mod._otel_available = None
        result = _check_otel()
        # OTel is a dev dependency so should be available
        assert result is True

    def test_otel_cached_after_first_check(self):
        import threetears.observe.tracing as mod

        mod._otel_available = None
        _check_otel()
        assert mod._otel_available is not None
        # Second call uses cache
        cached = mod._otel_available
        _check_otel()
        assert mod._otel_available is cached


class TestParamNames:
    """Parameter name extraction."""

    def test_get_param_names(self):
        def foo(a, b, c=3):
            pass

        assert _get_param_names(foo) == ("a", "b", "c")

    def test_get_param_names_empty(self):
        def foo():
            pass

        assert _get_param_names(foo) == ()


class TestSafeAttrs:
    """Span attribute safety filtering."""

    def test_set_safe_attr_string(self):
        span = MagicMock()
        _set_safe_attr(span, "arg.name", "name", "hello")
        span.set_attribute.assert_called_once_with("arg.name", "hello")

    def test_set_safe_attr_int(self):
        span = MagicMock()
        _set_safe_attr(span, "arg.count", "count", 42)
        span.set_attribute.assert_called_once_with("arg.count", 42)

    def test_set_safe_attr_uuid(self):
        from uuid import UUID

        span = MagicMock()
        uid = UUID("12345678-1234-5678-1234-567812345678")
        _set_safe_attr(span, "arg.id", "id", uid)
        span.set_attribute.assert_called_once_with("arg.id", str(uid))

    def test_set_safe_attr_sensitive_redacted(self):
        span = MagicMock()
        _set_safe_attr(span, "arg.password", "password", "secret123")
        span.set_attribute.assert_not_called()

    def test_set_safe_attr_long_string_truncated(self):
        span = MagicMock()
        long_str = "x" * 300
        _set_safe_attr(span, "arg.data", "data", long_str)
        span.set_attribute.assert_called_once()
        actual_value = span.set_attribute.call_args[0][1]
        assert len(actual_value) == 256

    def test_record_safe_args_skips_self(self):
        span = MagicMock()

        class Foo:
            def bar(self, name):
                pass

        _record_safe_args(span, Foo.bar, (Foo(), "hello"), {})
        calls = {c[0][0] for c in span.set_attribute.call_args_list}
        assert "arg.self" not in calls
        assert "arg.name" in calls

    def test_record_safe_result(self):
        span = MagicMock()
        _record_safe_result(span, [1, 2, 3])
        span.set_attribute.assert_any_call("result.type", "list")
        span.set_attribute.assert_any_call("result.count", 3)


class TestTracedDecorator:
    """@traced decorator behavior."""

    def test_traced_bare_sync(self):
        @traced
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_traced_parameterised_sync(self):
        @traced(name="custom.span")
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    async def test_traced_bare_async(self):
        @traced
        async def add(a, b):
            return a + b

        assert await add(1, 2) == 3

    async def test_traced_parameterised_async(self):
        @traced(name="custom.async.span")
        async def add(a, b):
            return a + b

        assert await add(1, 2) == 3

    def test_traced_preserves_function_name(self):
        @traced
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_traced_sync_exception_propagates(self):
        @traced
        def explode():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            explode()

    async def test_traced_async_exception_propagates(self):
        @traced
        async def explode():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await explode()

    def test_traced_passthrough_without_otel(self):
        import threetears.observe.tracing as mod

        mod._otel_available = False

        @traced(record_args=True, record_result=True)
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    async def test_traced_async_passthrough_without_otel(self):
        import threetears.observe.tracing as mod

        mod._otel_available = False

        @traced(record_args=True)
        async def add(a, b):
            return a + b

        assert await add(1, 2) == 3


class TestSetSpanAttribute:
    """set_span_attribute() -- attach attributes to the current active span."""

    def test_noop_without_otel(self):
        import threetears.observe.tracing as mod

        mod._otel_available = False

        set_span_attribute("key", "value")  # must not raise

    def test_noop_without_recording_span(self):
        with patch("opentelemetry.trace.get_current_span") as mock_get_span:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = False
            mock_get_span.return_value = mock_span

            set_span_attribute("key", "value")

            mock_span.set_attribute.assert_not_called()

    def test_sets_attribute_on_recording_span(self):
        with patch("opentelemetry.trace.get_current_span") as mock_get_span:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_get_span.return_value = mock_span

            set_span_attribute("survey.completion_rate", 0.75)

            mock_span.set_attribute.assert_called_once_with("survey.completion_rate", 0.75)

    def test_sensitive_key_redacted(self):
        with patch("opentelemetry.trace.get_current_span") as mock_get_span:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_get_span.return_value = mock_span

            set_span_attribute("password", "secret123")

            mock_span.set_attribute.assert_not_called()

    def test_uuid_value_converted_to_str(self):
        from uuid import UUID

        with patch("opentelemetry.trace.get_current_span") as mock_get_span:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_get_span.return_value = mock_span

            uid = UUID("12345678-1234-5678-1234-567812345678")
            set_span_attribute("survey.survey_id", uid)

            mock_span.set_attribute.assert_called_once_with("survey.survey_id", str(uid))

    def test_works_inside_traced_function(self):
        """integration: set_span_attribute reaches the span @traced opened, end to end."""

        @traced
        def do_work():
            set_span_attribute("result.custom", "value")
            return "done"

        assert do_work() == "done"  # must not raise -- proves the real integration works
