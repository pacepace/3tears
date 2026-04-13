"""integration tests for error translation across all provider types."""

from __future__ import annotations

from threetears.models.errors import friendly_api_error, identify_provider


class TestProviderIdentification:
    """integration tests for provider identification chain."""

    def test_generic_exception_returns_default(self) -> None:
        """generic exception without provider markers returns default label."""
        exc = RuntimeError("something broke")
        result = identify_provider(exc)
        assert result == "The LLM provider"

    def test_value_error_returns_default(self) -> None:
        """ValueError without provider keywords returns default label."""
        exc = ValueError("bad input")
        result = identify_provider(exc)
        assert result == "The LLM provider"


class TestErrorTranslation:
    """integration tests for error translation to friendly messages."""

    def test_generic_runtime_error(self) -> None:
        """generic RuntimeError produces non-empty friendly message."""
        exc = RuntimeError("unexpected failure")
        msg = friendly_api_error(exc)
        assert len(msg) > 0
        assert "RuntimeError" in msg
        assert "retry" in msg.lower() or "contact" in msg.lower()

    def test_generic_value_error(self) -> None:
        """generic ValueError produces non-empty friendly message."""
        exc = ValueError("bad model name")
        msg = friendly_api_error(exc)
        assert len(msg) > 0
        assert "Traceback" not in msg

    def test_openrouter_value_error(self) -> None:
        """ValueError mentioning OpenRouter API produces provider-specific message."""
        exc = ValueError("OpenRouter API returned 503")
        msg = friendly_api_error(exc)
        assert "OpenRouter" in msg
        assert "retry" in msg.lower()

    def test_connection_error(self) -> None:
        """ConnectionError produces friendly message without traceback."""
        exc = ConnectionError("refused")
        msg = friendly_api_error(exc)
        assert len(msg) > 0
        assert "Traceback" not in msg

    def test_no_message_contains_traceback_markers(self) -> None:
        """no friendly message contains raw traceback markers."""
        exceptions = [
            RuntimeError("test"),
            ValueError("test"),
            ConnectionError("test"),
            TimeoutError("test"),
            OSError("network down"),
        ]

        traceback_markers = ["Traceback", "File \"", "line ", "  at "]

        for exc in exceptions:
            msg = friendly_api_error(exc)
            for marker in traceback_markers:
                assert marker not in msg, (
                    f"friendly message for {type(exc).__name__} "
                    f"contains traceback marker: {marker!r}"
                )


class TestAnthropicErrorTranslation:
    """integration tests for Anthropic-specific error translation.

    tests require anthropic SDK to be installed. each test verifies
    that specific HTTP status codes produce appropriate friendly messages.
    """

    def test_anthropic_overloaded_error(self) -> None:
        """Anthropic 529 overloaded error produces retry message."""
        try:
            from anthropic import APIStatusError
            from httpx import Request, Response
        except ImportError:
            return

        request = Request(method="POST", url="https://api.anthropic.com/v1/messages")
        response = Response(
            status_code=529,
            request=request,
            json={"error": {"type": "overloaded_error", "message": "overloaded"}},
        )
        exc = APIStatusError(
            message="overloaded",
            response=response,
            body={"error": {"type": "overloaded_error", "message": "overloaded"}},
        )
        msg = friendly_api_error(exc)
        provider = identify_provider(exc)
        assert provider == "Anthropic"
        assert "overloaded" in msg.lower()
        assert "retry" in msg.lower()

    def test_anthropic_rate_limited_error(self) -> None:
        """Anthropic 429 rate limit error produces retry message."""
        try:
            from anthropic import APIStatusError
            from httpx import Request, Response
        except ImportError:
            return

        request = Request(method="POST", url="https://api.anthropic.com/v1/messages")
        response = Response(
            status_code=429,
            request=request,
            json={"error": {"type": "rate_limit_error", "message": "rate limited"}},
        )
        exc = APIStatusError(
            message="rate limited",
            response=response,
            body={"error": {"type": "rate_limit_error", "message": "rate limited"}},
        )
        msg = friendly_api_error(exc)
        assert "rate" in msg.lower()
        assert "retry" in msg.lower()

    def test_anthropic_server_error(self) -> None:
        """Anthropic 500 server error produces retry message."""
        try:
            from anthropic import APIStatusError
            from httpx import Request, Response
        except ImportError:
            return

        request = Request(method="POST", url="https://api.anthropic.com/v1/messages")
        response = Response(
            status_code=500,
            request=request,
            json={"error": {"type": "server_error", "message": "internal error"}},
        )
        exc = APIStatusError(
            message="internal error",
            response=response,
            body={"error": {"type": "server_error", "message": "internal error"}},
        )
        msg = friendly_api_error(exc)
        assert "outage" in msg.lower() or "server" in msg.lower()
        assert "retry" in msg.lower()

    def test_anthropic_auth_error(self) -> None:
        """Anthropic 401 auth error suggests contacting administrator."""
        try:
            from anthropic import APIStatusError
            from httpx import Request, Response
        except ImportError:
            return

        request = Request(method="POST", url="https://api.anthropic.com/v1/messages")
        response = Response(
            status_code=401,
            request=request,
            json={"error": {"type": "authentication_error", "message": "invalid key"}},
        )
        exc = APIStatusError(
            message="invalid key",
            response=response,
            body={"error": {"type": "authentication_error", "message": "invalid key"}},
        )
        msg = friendly_api_error(exc)
        assert "key" in msg.lower() or "administrator" in msg.lower()

    def test_anthropic_timeout_error(self) -> None:
        """Anthropic timeout error produces retry message."""
        try:
            from anthropic import APITimeoutError
            from httpx import Request
        except ImportError:
            return

        request = Request(method="POST", url="https://api.anthropic.com/v1/messages")
        exc = APITimeoutError(request=request)
        msg = friendly_api_error(exc)
        provider = identify_provider(exc)
        assert provider == "Anthropic"
        assert "timed out" in msg.lower() or "too long" in msg.lower()
        assert "retry" in msg.lower()

    def test_anthropic_connection_error(self) -> None:
        """Anthropic connection error produces connectivity message."""
        try:
            from anthropic import APIConnectionError
            from httpx import Request
        except ImportError:
            return

        request = Request(method="POST", url="https://api.anthropic.com/v1/messages")
        exc = APIConnectionError(request=request)
        msg = friendly_api_error(exc)
        assert "connect" in msg.lower()
