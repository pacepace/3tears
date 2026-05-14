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


def _extract_provider_body_message(body: Any) -> str | None:
    """extract the provider's own user-facing error message from an API response body.

    Anthropic and OpenAI both return a structured ``{"error": {"message": "..."}}``
    shape on most non-transient client errors (400/401/402/403). That string is the
    most actionable thing we can show the user -- it names the actual problem in
    the provider's own words (``"Your credit balance is too low to access the
    Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."``)
    rather than a generic "HTTP 4xx" fallback that throws away the diagnostic.

    Returns ``None`` if the body doesn't have the expected shape or the message
    field is missing/empty.
    """
    if not isinstance(body, dict):
        return None
    error_obj = body.get("error")
    if not isinstance(error_obj, dict):
        return None
    msg = error_obj.get("message")
    if not isinstance(msg, str) or not msg.strip():
        return None
    return msg.strip()


def friendly_api_error(exc: Exception) -> str:
    """maps exception to user-facing error message string.

    translates raw LLM provider exceptions into friendly messages
    suitable for display to end users. never includes tracebacks
    or raw error details.

    For known transient classes (overloaded, rate-limited, 5xx) we
    substitute our own "please retry" guidance because the provider's
    own message is rarely better than the categorized advice. For
    non-recoverable client errors (400, 402, 403) we PREFER the
    provider's own ``body.error.message`` when present -- those
    messages contain the actual problem ("credit balance too low",
    "context window exceeded", "content policy violation") which is
    the only thing the user can act on. 401 is the exception: it
    indicates a server-side configuration problem (bad API key),
    not something the end user can fix, so we keep our own message.

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
        body_message = _extract_provider_body_message(body)

        if exc.status_code == 529 or is_overloaded:
            message = f"{provider} is overloaded right now. Please retry in 1-2 minutes."
        elif exc.status_code == 429:
            message = f"{provider} rate-limited our request. Please retry in about 30 seconds."
        elif 500 <= exc.status_code < 600:
            message = f"{provider} is having a server-side outage. Please retry in 2-3 minutes."
        elif exc.status_code == 401:
            message = f"{provider} rejected our API key. Please contact an administrator."
        elif exc.status_code in (400, 402, 403) and body_message is not None:
            # Surface the provider's own actionable message verbatim
            # (e.g. "Your credit balance is too low to access the
            # Anthropic API. Please go to Plans & Billing to upgrade
            # or purchase credits."). The pre-2026-05-13 behavior
            # was to substitute a generic "HTTP 4xx" line and throw
            # away the diagnostic, leaving the user staring at a
            # nondescript "unexpected error" while the actual answer
            # ("top up your credits") was sitting in the response body.
            message = f"{provider}: {body_message}"
        else:
            message = f"{provider} returned an unexpected error (HTTP {exc.status_code}). Please retry in a minute."
    elif _HAS_ANTHROPIC and isinstance(exc, APITimeoutError):
        message = f"{provider} took too long to respond (request timed out). Please retry."
    elif _HAS_ANTHROPIC and isinstance(exc, APIConnectionError):
        message = f"Could not connect to {provider} (network issue). Check connectivity and retry."
    elif isinstance(exc, ValueError) and "OpenRouter API" in str(exc):
        message = "OpenRouter returned an error. Please retry in 1-2 minutes."

    return message
