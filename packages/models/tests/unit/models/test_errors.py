"""tests for error translation layer."""

from __future__ import annotations

import httpx
from anthropic import APIConnectionError, APIStatusError, APITimeoutError

from threetears.models.errors import friendly_api_error, identify_provider


def _make_request() -> httpx.Request:
    """creates mock httpx request for exception constructors."""
    return httpx.Request("POST", "https://api.anthropic.com")


def _make_status_error(
    status_code: int,
    message: str = "error",
    body: object | None = None,
) -> APIStatusError:
    """creates APIStatusError with given status code and body."""
    response = httpx.Response(status_code, request=_make_request())
    return APIStatusError(message, response=response, body=body)


class _FakeOpenAIError(Exception):
    """fake exception pretending to come from openai module."""

    pass


_FakeOpenAIError.__module__ = "openai.errors"


class TestIdentifyProvider:
    """tests for identify_provider function."""

    def test_anthropic_exception_returns_anthropic(self) -> None:
        """anthropic SDK exception identified as Anthropic."""
        exc = _make_status_error(500)
        assert identify_provider(exc) == "Anthropic"

    def test_openai_module_exception_returns_openai(self) -> None:
        """exception from openai module identified as OpenAI."""
        exc = _FakeOpenAIError("something broke")
        assert identify_provider(exc) == "OpenAI"

    def test_openrouter_in_string_returns_openrouter(self) -> None:
        """exception with OpenRouter in string identified as OpenRouter."""
        exc = ValueError("OpenRouter returned status 500")
        assert identify_provider(exc) == "OpenRouter"

    def test_openai_in_string_case_insensitive_returns_openai(self) -> None:
        """exception with openai in string (case-insensitive) identified as OpenAI."""
        exc = RuntimeError("openai server error")
        assert identify_provider(exc) == "OpenAI"

    def test_generic_exception_returns_fallback(self) -> None:
        """generic exception returns fallback provider name."""
        exc = RuntimeError("something went wrong")
        assert identify_provider(exc) == "The LLM provider"

    def test_module_check_takes_priority_over_string_check(self) -> None:
        """module-based check takes priority over string content."""
        exc = _make_status_error(500, message="openrouter issue")
        assert identify_provider(exc) == "Anthropic"


class TestFriendlyApiError:
    """tests for friendly_api_error function."""

    def test_status_529_returns_overloaded_message(self) -> None:
        """APIStatusError 529 returns overloaded message."""
        exc = _make_status_error(529)
        result = friendly_api_error(exc)
        assert "overloaded" in result
        assert "1-2 minutes" in result

    def test_overloaded_error_body_returns_overloaded_message(self) -> None:
        """APIStatusError with overloaded_error body returns overloaded message."""
        body = {"error": {"type": "overloaded_error", "message": "overloaded"}}
        exc = _make_status_error(500, body=body)
        result = friendly_api_error(exc)
        assert "overloaded" in result
        assert "1-2 minutes" in result

    def test_status_429_returns_rate_limited_message(self) -> None:
        """APIStatusError 429 returns rate limited message."""
        exc = _make_status_error(429)
        result = friendly_api_error(exc)
        assert "rate-limited" in result
        assert "30 seconds" in result

    def test_status_500_returns_server_outage_message(self) -> None:
        """APIStatusError 500 returns server outage message."""
        exc = _make_status_error(500)
        result = friendly_api_error(exc)
        assert "server-side outage" in result
        assert "2-3 minutes" in result

    def test_status_502_returns_server_outage_message(self) -> None:
        """APIStatusError 502 returns server outage message."""
        exc = _make_status_error(502)
        result = friendly_api_error(exc)
        assert "server-side outage" in result

    def test_status_503_returns_server_outage_message(self) -> None:
        """APIStatusError 503 returns server outage message."""
        exc = _make_status_error(503)
        result = friendly_api_error(exc)
        assert "server-side outage" in result

    def test_status_401_returns_auth_failed_message(self) -> None:
        """APIStatusError 401 returns auth failed message."""
        exc = _make_status_error(401)
        result = friendly_api_error(exc)
        assert "rejected our API key" in result
        assert "administrator" in result

    def test_status_403_returns_unexpected_error_with_code(self) -> None:
        """APIStatusError 403 returns unexpected error with HTTP code."""
        exc = _make_status_error(403)
        result = friendly_api_error(exc)
        assert "unexpected error" in result
        assert "HTTP 403" in result

    def test_overloaded_error_body_with_malformed_error_field(self) -> None:
        """APIStatusError with non-dict error field in body does not crash."""
        body = {"error": "just a string, not a dict"}
        exc = _make_status_error(500, body=body)
        result = friendly_api_error(exc)
        assert "server-side outage" in result

    def test_overloaded_error_body_with_none_body(self) -> None:
        """APIStatusError with None body does not crash."""
        exc = _make_status_error(500, body=None)
        result = friendly_api_error(exc)
        assert "server-side outage" in result

    def test_status_400_returns_unexpected_error_with_code(self) -> None:
        """APIStatusError 400 returns unexpected error with HTTP code."""
        exc = _make_status_error(400)
        result = friendly_api_error(exc)
        assert "unexpected error" in result
        assert "HTTP 400" in result

    def test_timeout_error_returns_timeout_message(self) -> None:
        """APITimeoutError returns timeout message."""
        exc = APITimeoutError(request=_make_request())
        result = friendly_api_error(exc)
        assert "timed out" in result
        assert "retry" in result.lower()

    def test_connection_error_returns_connection_message(self) -> None:
        """APIConnectionError returns connection error message."""
        exc = APIConnectionError(message="Connection refused", request=_make_request())
        result = friendly_api_error(exc)
        assert "Could not connect" in result
        assert "network issue" in result

    def test_valueerror_with_openrouter_api_returns_openrouter_message(self) -> None:
        """ValueError with OpenRouter API in string returns OpenRouter message."""
        exc = ValueError("OpenRouter API returned 500")
        result = friendly_api_error(exc)
        assert "OpenRouter" in result
        assert "1-2 minutes" in result

    def test_valueerror_without_openrouter_returns_generic_fallback(self) -> None:
        """ValueError without OpenRouter API returns generic fallback."""
        exc = ValueError("something else went wrong")
        result = friendly_api_error(exc)
        assert "unexpected went wrong" in result
        assert "ValueError" in result

    def test_runtime_error_returns_generic_fallback(self) -> None:
        """RuntimeError returns generic fallback with type name."""
        exc = RuntimeError("unknown failure")
        result = friendly_api_error(exc)
        assert "RuntimeError" in result
        assert "unexpected went wrong" in result

    def test_generic_exception_returns_fallback_with_type_name(self) -> None:
        """generic Exception returns fallback with type name."""
        exc = Exception("mystery error")
        result = friendly_api_error(exc)
        assert "Exception" in result
        assert "administrator" in result

    def test_friendly_message_never_contains_traceback_markers(self) -> None:
        """friendly messages never contain traceback markers."""
        exceptions = [
            _make_status_error(500),
            _make_status_error(429),
            APITimeoutError(request=_make_request()),
            APIConnectionError(message="failed", request=_make_request()),
            ValueError("OpenRouter API error"),
            RuntimeError("Traceback (most recent call last):\n  File \"test.py\", line 1"),
        ]
        traceback_markers = ["Traceback", 'File "', "line "]
        for exc in exceptions:
            result = friendly_api_error(exc)
            for marker in traceback_markers:
                assert marker not in result, (
                    f"Message for {type(exc).__name__} contains traceback marker: {marker!r}"
                )


class TestFriendlyApiErrorProviderNames:
    """tests for provider name inclusion in friendly messages."""

    def test_anthropic_429_includes_anthropic_in_message(self) -> None:
        """anthropic 429 error includes Anthropic in message."""
        exc = _make_status_error(429)
        result = friendly_api_error(exc)
        assert "Anthropic" in result

    def test_generic_429_like_error_includes_fallback_provider(self) -> None:
        """generic error (not from known provider) includes fallback provider name."""
        exc = RuntimeError("rate limit exceeded")
        result = friendly_api_error(exc)
        assert "The LLM provider" not in result or "unexpected went wrong" in result
