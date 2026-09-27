"""error translation layer for AI model provider exceptions.

maps raw LLM API exceptions into user-friendly message strings.
functions are synchronous, pure inspection — they do not raise,
log, or include raw tracebacks in output.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ModelCallTimeout",
    "ModelProviderError",
    "ModelRateLimitError",
    "friendly_api_error",
    "identify_provider",
    "is_provider_error",
]


class ModelCallTimeout(TimeoutError):
    """A model call ran past its whole-call deadline.

    Its own class, so a caller can tell the provider running long from any other
    ``TimeoutError`` -- its own deadlines included.
    """


class ModelProviderError(RuntimeError):
    """A model call's provider reported that the call failed.

    For a provider that answers a failure as data rather than raising an SDK
    exception of its own -- the Claude CLI behind a subscription sends its
    errors as messages. Raised in place of an answer, so the provider's error
    text never reaches a caller as the model's words, and a circuit breaker
    counts the call as the failure it was.

    :param detail: what the provider said, in its own words
    :ptype detail: str
    :param provider: human-readable name of the provider
    :ptype provider: str
    :param reason: the provider's own code for the failure (``"server_error"``,
        ``"authentication_failed"``, ...), when it gave one
    :ptype reason: str | None
    :param status: HTTP status of the provider's failing API call, when known
    :ptype status: int | None
    :param rejected_output: for a call asked for a structured answer, the last answer the
        provider rejected against the schema, exactly as the model gave it
    :ptype rejected_output: dict[str, Any] | None
    :param rejection: why the provider rejected ``rejected_output``, in its own words
    :ptype rejection: str | None
    """

    def __init__(
        self,
        detail: str,
        *,
        provider: str,
        reason: str | None = None,
        status: int | None = None,
        rejected_output: dict[str, Any] | None = None,
        rejection: str | None = None,
    ) -> None:
        self.detail = detail
        self.provider = provider
        self.reason = reason
        self.status = status
        self.rejected_output = rejected_output
        self.rejection = rejection
        said = f"{detail} (last rejected answer: {rejection})" if rejection else detail
        super().__init__(f"{provider} call failed ({reason or status or 'error'}): {said}")


class ModelRateLimitError(ModelProviderError):
    """The provider refused a call for a rate or usage limit.

    Its own class so a caller can wait for the limit rather than treat the
    provider as down. A subscription's session limit is one.

    :param detail: what the provider said, in its own words
    :ptype detail: str
    :param provider: human-readable name of the provider
    :ptype provider: str
    :param reason: the provider's own code for the failure, when it gave one
    :ptype reason: str | None
    :param status: HTTP status of the provider's failing API call, when known
    :ptype status: int | None
    :param resets: when the limit resets, in the provider's words, when it said
    :ptype resets: str | None
    :param rejected_output: the last structured answer the provider rejected, as for
        :class:`ModelProviderError`
    :ptype rejected_output: dict[str, Any] | None
    :param rejection: why the provider rejected ``rejected_output``, in its own words
    :ptype rejection: str | None
    """

    def __init__(
        self,
        detail: str,
        *,
        provider: str,
        reason: str | None = None,
        status: int | None = None,
        resets: str | None = None,
        rejected_output: dict[str, Any] | None = None,
        rejection: str | None = None,
    ) -> None:
        self.resets = resets
        super().__init__(
            detail,
            provider=provider,
            reason=reason,
            status=status,
            rejected_output=rejected_output,
            rejection=rejection,
        )


#: The packages a model call's own errors come from.
_PROVIDER_PACKAGES = frozenset({"anthropic", "openai", "openrouter", "httpx", "httpcore"})

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

    if isinstance(exc, ModelProviderError):
        result = exc.provider
    elif "anthropic" in module:
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


def _friendly_reported_failure(exc: ModelProviderError) -> str:
    """the user-facing message for a failure a provider reported as data.

    worded as :func:`friendly_api_error` words the API route's status of the
    same kind, so a caller shows one message whichever route failed. a limit
    that says when it resets says so; the provider's own words are shown where
    the API route shows them -- a billing or request problem the caller can act
    on.

    :param exc: the reported failure
    :ptype exc: ModelProviderError
    :return: user-friendly error message string
    :rtype: str
    """
    provider = exc.provider
    status = exc.status or 0
    message = f"{provider} returned an unexpected error. Please retry in a minute."
    if isinstance(exc, ModelRateLimitError):
        message = (
            f"{provider} has reached its usage limit. It resets {exc.resets}."
            if exc.resets
            else f"{provider} rate-limited our request. Please retry in about 30 seconds."
        )
    elif exc.reason == "authentication_failed" or status == 401:
        message = f"{provider} rejected our credentials. Please contact an administrator."
    elif status == 529:
        message = f"{provider} is overloaded right now. Please retry in 1-2 minutes."
    elif exc.reason == "server_error" or 500 <= status < 600:
        message = f"{provider} is having a server-side outage. Please retry in 2-3 minutes."
    elif exc.reason in ("billing_error", "invalid_request") and exc.detail.strip():
        message = f"{provider}: {exc.detail.strip()}"
    return message


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

    if isinstance(exc, ModelProviderError):
        message = _friendly_reported_failure(exc)
    elif _HAS_ANTHROPIC and isinstance(exc, APIStatusError):
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


def is_provider_error(exc: BaseException) -> bool:
    """Whether a model call's provider failed, rather than the caller's code.

    A provider SDK's or its HTTP client's exception, a :class:`ModelCallTimeout`,
    a failure the provider reported as data (:class:`ModelProviderError`, a rate
    limit included), the circuit breaker refusing a provider that keeps failing
    (:class:`~threetears.models.circuit_breaker.CircuitOpenError`), or the
    OpenRouter error the chat model raises as a ``ValueError``.

    :param exc: what the call raised
    :ptype exc: BaseException
    :return: ``True`` for a provider failure
    :rtype: bool
    """
    from threetears.models.circuit_breaker import CircuitOpenError

    package = (type(exc).__module__ or "").split(".", 1)[0]
    return (
        package in _PROVIDER_PACKAGES
        or isinstance(exc, (ModelCallTimeout, ModelProviderError, CircuitOpenError))
        or (isinstance(exc, ValueError) and "OpenRouter API" in str(exc))
    )
