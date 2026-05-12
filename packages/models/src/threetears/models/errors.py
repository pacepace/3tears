"""error translation layer for AI model provider exceptions.

maps raw LLM API exceptions into user-friendly message strings.
functions are synchronous, pure inspection — they do not raise,
log, or include raw tracebacks in output.
"""

from __future__ import annotations

__all__ = [
    "friendly_api_error",
    "identify_provider",
]

try:
    from anthropic import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
    )

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


def identify_provider(exc: Exception) -> str:
    """inspects exception to determine originating provider name.

    checks exception module path first, then falls back to
    inspecting string representation for provider keywords.

    :param exc: exception instance from LLM API call
    :ptype exc: Exception
    :return: human-readable provider name string
    :rtype: str
    """
    module = type(exc).__module__ or ""

    result = "The LLM provider"

    if "anthropic" in module:
        result = "Anthropic"
    elif "openai" in module:
        result = "OpenAI"
    elif "openrouter" in str(exc).lower():
        result = "OpenRouter"
    elif "openai" in str(exc).lower():
        result = "OpenAI"

    return result


def friendly_api_error(exc: Exception) -> str:
    """maps exception to user-facing error message string.

    translates raw LLM provider exceptions into friendly messages
    suitable for display to end users. never includes tracebacks
    or raw error details.

    :param exc: exception instance from LLM API call
    :ptype exc: Exception
    :return: user-friendly error message string
    :rtype: str
    """
    provider = identify_provider(exc)

    message = f"Something unexpected went wrong ({type(exc).__name__}). Please retry or contact an administrator."

    if _HAS_ANTHROPIC and isinstance(exc, APIStatusError):
        body = exc.body
        error_obj = body.get("error", {}) if isinstance(body, dict) else {}
        is_overloaded = isinstance(error_obj, dict) and error_obj.get("type") == "overloaded_error"

        if exc.status_code == 529 or is_overloaded:
            message = f"{provider} is overloaded right now. Please retry in 1-2 minutes."
        elif exc.status_code == 429:
            message = f"{provider} rate-limited our request. Please retry in about 30 seconds."
        elif 500 <= exc.status_code < 600:
            message = f"{provider} is having a server-side outage. Please retry in 2-3 minutes."
        elif exc.status_code == 401:
            message = f"{provider} rejected our API key. Please contact an administrator."
        else:
            message = f"{provider} returned an unexpected error (HTTP {exc.status_code}). Please retry in a minute."
    elif _HAS_ANTHROPIC and isinstance(exc, APITimeoutError):
        message = f"{provider} took too long to respond (request timed out). Please retry."
    elif _HAS_ANTHROPIC and isinstance(exc, APIConnectionError):
        message = f"Could not connect to {provider} (network issue). Check connectivity and retry."
    elif isinstance(exc, ValueError) and "OpenRouter API" in str(exc):
        message = "OpenRouter returned an error. Please retry in 1-2 minutes."

    return message
