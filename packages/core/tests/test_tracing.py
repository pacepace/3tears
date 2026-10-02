"""Tests for threetears.observe.tracing."""

from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from threetears.observe.tracing import traced


def test_traced_without_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    """When opentelemetry is not available, decorator is a pure passthrough.

    The availability check runs once per process and remembers its answer, and OpenTelemetry is
    installed here, so the module is imported afresh with the distribution unimportable -- the
    state an install without it is in. ``patch.dict`` restores ``sys.modules`` on exit and
    ``monkeypatch`` the package attribute the import rebinds, so nothing else sees this copy.
    """
    import threetears.observe as observe_pkg

    monkeypatch.setattr(observe_pkg, "tracing", importlib.import_module("threetears.observe.tracing"))
    with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.trace": None}):
        sys.modules.pop("threetears.observe.tracing", None)
        tracing = importlib.import_module("threetears.observe.tracing")

        @tracing.traced
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)

    assert result == 5


def _make_mock_span() -> MagicMock:
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    return span


def test_traced_sync_function():
    """With a mock tracer, verify span is created for sync function."""
    mock_span = _make_mock_span()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    @traced
    def multiply(a: int, b: int) -> int:
        return a * b

    with (
        patch("opentelemetry.trace.get_tracer", return_value=mock_tracer),
    ):
        result = multiply(3, 4)

    assert result == 12
    mock_tracer.start_as_current_span.assert_called_once()


def test_traced_async_function():
    """With a mock tracer, verify span is created for async function."""
    mock_span = _make_mock_span()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    @traced
    async def async_add(a: int, b: int) -> int:
        return a + b

    with (
        patch("opentelemetry.trace.get_tracer", return_value=mock_tracer),
    ):
        result = asyncio.run(async_add(5, 6))

    assert result == 11
    mock_tracer.start_as_current_span.assert_called_once()


def test_sensitive_params_filtered():
    """Sensitive parameter names (password, token, etc.) are redacted."""
    mock_span = _make_mock_span()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    @traced(record_args=True)
    def login(username: str, password: str, token: str = "") -> str:
        return "ok"

    with (
        patch("opentelemetry.trace.get_tracer", return_value=mock_tracer),
    ):
        login("alice", "s3cret", token="tok-123")

    # Collect all set_attribute calls into a dict
    attrs: dict[str, Any] = {}
    for call in mock_span.set_attribute.call_args_list:
        attrs[call[0][0]] = call[0][1]

    assert attrs.get("arg.username") == "alice"
    # Sensitive params are skipped entirely, not redacted
    assert "arg.password" not in attrs
    assert "arg.token" not in attrs


def test_traced_records_exception():
    """Exceptions are recorded on the span with ERROR status."""
    mock_span = _make_mock_span()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    @traced
    def failing() -> None:
        raise ValueError("boom")

    with (
        patch("opentelemetry.trace.get_tracer", return_value=mock_tracer),
    ):
        with pytest.raises(ValueError, match="boom"):
            failing()

    mock_span.set_status.assert_called_once()
    mock_span.record_exception.assert_called_once()
    exc_arg = mock_span.record_exception.call_args[0][0]
    assert isinstance(exc_arg, ValueError)
    assert str(exc_arg) == "boom"
