"""Tests for threetears.observe.tracing."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from threetears.observe.tracing import set_span_attribute, traced


@contextmanager
def _tracing_without_otel(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """``threetears.observe.tracing`` imported afresh into a process where OpenTelemetry is absent.

    The availability check runs once per process and remembers its answer, so the module the
    rest of this suite shares has already seen OpenTelemetry installed. A fresh import with the
    distribution unimportable is the state an install without it is in. ``patch.dict`` restores
    ``sys.modules`` on exit and ``monkeypatch`` the package attribute the import rebinds, so the
    rest of the process never sees this copy; the copy itself stays usable after the block.

    :param monkeypatch: pytest's patcher
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the freshly imported module
    :rtype: Iterator[ModuleType]
    """
    import threetears.observe as observe_pkg

    monkeypatch.setattr(observe_pkg, "tracing", importlib.import_module("threetears.observe.tracing"))
    with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
        sys.modules.pop("threetears.observe.tracing", None)
        yield importlib.import_module("threetears.observe.tracing")


class TestOtelCheck:
    """OTel availability detection, observed through whether a traced call makes a span."""

    def test_otel_available_when_installed(self):
        # OTel is a dev dependency, so a traced call goes to a tracer
        with patch("opentelemetry.trace.get_tracer") as get_tracer:

            @traced
            def add(a, b):
                return a + b

            assert add(1, 2) == 3
        get_tracer.assert_called_once()

    def test_otel_cached_after_first_check(self, monkeypatch: pytest.MonkeyPatch):
        """the first check decides for the process: OTel arriving later does not change it."""
        with _tracing_without_otel(monkeypatch) as tracing:

            @tracing.traced
            def add(a, b):
                return a + b

            assert add(1, 2) == 3  # the check runs here, with OTel absent

        with patch("opentelemetry.trace.get_tracer") as get_tracer:
            assert add(1, 2) == 3  # OTel importable again
        get_tracer.assert_not_called()


def _span_attributes(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """call *fn* under ``@traced(record_args=True, record_result=True)`` and return what its span recorded.

    :param fn: the function to trace
    :ptype fn: Any
    :param args: positional arguments for the call
    :ptype args: Any
    :param kwargs: keyword arguments for the call
    :ptype kwargs: Any
    :return: every attribute set on the span, by key
    :rtype: dict[str, Any]
    """
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    with patch("opentelemetry.trace.get_tracer", return_value=tracer):
        traced(record_args=True, record_result=True)(fn)(*args, **kwargs)
    return {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}


def _arg_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """the ``arg.*`` attributes alone.

    :param attributes: a span's attributes
    :ptype attributes: dict[str, Any]
    :return: the recorded arguments
    :rtype: dict[str, Any]
    """
    return {key: value for key, value in attributes.items() if key.startswith("arg.")}


class TestParamNames:
    """Parameter names, as the keys recorded arguments are filed under."""

    def test_get_param_names(self):
        def foo(a, b, c=3):
            pass

        assert _arg_attributes(_span_attributes(foo, 1, 2, 3)) == {"arg.a": 1, "arg.b": 2, "arg.c": 3}

    def test_get_param_names_empty(self):
        def foo():
            pass

        assert _arg_attributes(_span_attributes(foo)) == {}


class TestSafeAttrs:
    """Span attribute safety filtering, observed on the span a traced call records into."""

    def test_set_safe_attr_string(self):
        def f(name):
            pass

        assert _arg_attributes(_span_attributes(f, "hello")) == {"arg.name": "hello"}

    def test_set_safe_attr_int(self):
        def f(count):
            pass

        assert _arg_attributes(_span_attributes(f, 42)) == {"arg.count": 42}

    def test_set_safe_attr_uuid(self):
        from uuid import UUID

        def f(id):
            pass

        uid = UUID("12345678-1234-5678-1234-567812345678")
        assert _arg_attributes(_span_attributes(f, uid)) == {"arg.id": str(uid)}

    def test_set_safe_attr_sensitive_redacted(self):
        def f(password):
            pass

        assert _arg_attributes(_span_attributes(f, "secret123")) == {}

    def test_set_safe_attr_long_string_truncated(self):
        def f(data):
            pass

        recorded = _arg_attributes(_span_attributes(f, "x" * 300))
        assert len(recorded["arg.data"]) == 256

    def test_record_safe_args_skips_self(self):
        class Foo:
            def bar(self, name):
                pass

        recorded = _arg_attributes(_span_attributes(Foo.bar, Foo(), "hello"))
        assert "arg.self" not in recorded
        assert recorded == {"arg.name": "hello"}

    def test_record_safe_result(self):
        def f():
            return [1, 2, 3]

        attributes = _span_attributes(f)
        assert attributes["result.type"] == "list"
        assert attributes["result.count"] == 3


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

    def test_traced_passthrough_without_otel(self, monkeypatch: pytest.MonkeyPatch):
        with _tracing_without_otel(monkeypatch) as tracing:

            @tracing.traced(record_args=True, record_result=True)
            def add(a, b):
                return a + b

            assert add(1, 2) == 3

    async def test_traced_async_passthrough_without_otel(self, monkeypatch: pytest.MonkeyPatch):
        with _tracing_without_otel(monkeypatch) as tracing:

            @tracing.traced(record_args=True)
            async def add(a, b):
                return a + b

            assert await add(1, 2) == 3


class TestSetSpanAttribute:
    """set_span_attribute() -- attach attributes to the current active span."""

    def test_noop_without_otel(self, monkeypatch: pytest.MonkeyPatch):
        with _tracing_without_otel(monkeypatch) as tracing:
            tracing.set_span_attribute("key", "value")  # must not raise

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
