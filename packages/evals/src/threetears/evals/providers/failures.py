"""What the engine may know about a call through the adapter that raised: :class:`ModelsFailureDescriber`.

The failure half of the port (:class:`~threetears.evals.schema.ProviderFailure`) for a client
:func:`~threetears.evals.providers.models_client` built. Neither LangChain nor 3tears-models wraps a provider's
exception (apart from LangChain's two context-overflow subclasses and models' own CLI and timeout types), so this
reads the SDKs' own exception types directly, by their structured fields (``status_code``, ``body``, the typed
``message``), and never by ``str(exc)``: on Anthropic and OpenAI that text is ``"Error code: N - <the parsed body>"``,
which is exactly what a description must not repeat.

models' ``friendly_api_error`` is not a substitute: it recognises only Anthropic's status error, sets no account
flag, and its fallback text is the exception's class.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Final

import anthropic
import openai
import pydantic
from langchain_core.exceptions import ContextOverflowError
from openrouter import errors as openrouter_errors

from threetears.evals.schema import ProviderFailure
from threetears.models import CircuitOpenError, ModelCallTimeout, ModelProviderError, ModelRateLimitError
from threetears.models.providers.structured_output import StructuredOutputError

from threetears.evals.providers.adapter import ModelsClientRefused

__all__ = ["ModelsFailureDescriber"]

#: The bare ``ValueError`` LangChain's OpenRouter model raises for an error the stream carried in a chunk, whose
#: text is the provider's message and its code and nothing else.
_OPENROUTER_STREAM_ERROR: Final = re.compile(
    r"OpenRouter API returned an error during streaming: (?P<message>.*) \(code: (?P<code>[^)]*)\)", re.DOTALL
)

#: Anthropic's notice when an account's prepaid credit is spent, which arrives as a 400 rather than a 402.
_ANTHROPIC_CREDIT_NOTICE: Final = "credit balance"

#: The CLI's own failure codes that refuse the calling account rather than the request.
_ACCOUNT_REASONS: Final = frozenset({"authentication_failed", "billing_error"})

#: How much of a provider's own message a description keeps, in characters.
_MESSAGE_LIMIT: Final = 500


def _body_message(body: object) -> str | None:
    """The provider's own message inside a response body: ``error.message``, or ``message``, or none.

    Args:
        body: The body as the SDK kept it: a decoded mapping, or the raw text.

    Returns:
        The message, or ``None`` when the body carries none.
    """
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return None
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    if isinstance(error, Mapping):
        body = error
    message = body.get("message")
    return message if isinstance(message, str) and message else None


def _error_field(body: object, name: str) -> object:
    """``body[name]`` or ``body['error'][name]``, whichever the SDK's body shape carries."""
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    if isinstance(error, Mapping) and name in error:
        return error.get(name)
    return body.get(name)


def _clip(text: str) -> str:
    return text if len(text) <= _MESSAGE_LIMIT else f"{text[:_MESSAGE_LIMIT]}…"


def _status_line(provider: str, status: int, message: str | None, *, overflow: bool) -> str:
    said = f": {_clip(message)}" if message else " (the response carried no message)"
    window = " (the prompt is over the model's context window)" if overflow else ""
    return f"{provider} {status}{window}{said}"


class ModelsFailureDescriber:
    """A :class:`~threetears.evals.schema.ProviderFailureDescriber` for the adapter's providers, total over ``BaseException``.

    Hand one to :class:`~threetears.evals.kernel.host.EvalHost` as ``failure_describer`` beside
    :func:`~threetears.evals.providers.models_clients`. It holds no state, so one instance serves every host.

    **What each exception reads as.** ``description`` is the provider, the HTTP status where there is one, and the
    provider's own message (from the body's ``error.message``, or OpenRouter's typed ``message``); the exception's
    text is never repeated where it carries the body. ``payload_withheld`` is set exactly where the exception's
    text, or one it chains, carries something the description leaves out:

    * an Anthropic or OpenAI status error (``"Error code: N - <body>"``), or an OpenAI or Anthropic reply the SDK
      could not validate: the body is withheld;
    * an OpenRouter status error whose typed message is the body itself (the SDK's fallback), its default error
      (whose text appends ``Body: …``) and its response-validation error (whose text carries the pydantic errors,
      with the input): withheld;
    * a pydantic ``ValidationError`` (a reply chunk the SDK's schema refused), whose text carries the input:
      withheld;
    * a bare ``ValueError`` (OpenAI's 200 that carries an error, which LangChain raises with the error dict as its
      text): withheld, except OpenRouter's mid-stream error, whose text is the message and the code;
    * an exception of no type named here: withheld, since its text is unknown, as
      :func:`~threetears.evals.kernel.withhold_failure_detail` does.

    It is not set for a typed OpenRouter error (its text is the message alone), a connection error or timeout, a
    breaker's refusal, a refusal of the adapter's own, a caller's schema error, or a CLI failure with no rejected
    answer: their text is the description.

    ``account_refused`` is set where the provider refused the calling ACCOUNT, so every model behind the key would
    get the same answer: a 401 or 403 from Anthropic or OpenAI; Anthropic's 402 and its 400 credit-balance notice;
    OpenAI's 429 ``insufficient_quota``, which looks like a rate limit and is not; OpenRouter's 401 and 402 (not
    its 403, which it also uses for moderation); a CLI ``authentication_failed`` or ``billing_error``, or a rate
    limit that names when a subscription resets; and a circuit breaker scoped to one credential. Anything it cannot
    tell is ``False``, and the failure is charged as any other.
    """

    def __call__(self, exc: BaseException) -> ProviderFailure:
        """Describe one raised exception without repeating a provider payload.

        Args:
            exc: Whatever the client raised, or anything else the engine caught around the call.

        Returns:
            The failure as the engine may record it.
        """
        overflow = isinstance(exc, ContextOverflowError)
        if isinstance(exc, anthropic.APIStatusError):
            return self._anthropic_status(exc, overflow=overflow)
        if isinstance(exc, openai.APIStatusError):
            return self._openai_status(exc, overflow=overflow)
        if isinstance(exc, openrouter_errors.OpenRouterError):
            return self._openrouter(exc)
        if isinstance(exc, anthropic.APIConnectionError | openai.APIConnectionError):
            provider = "anthropic" if isinstance(exc, anthropic.APIConnectionError) else "openai"
            timed_out = isinstance(exc, anthropic.APITimeoutError | openai.APITimeoutError)
            what = "timed out" if timed_out else "could not connect"
            return ProviderFailure(f"{provider}: the request {what}", payload_withheld=False, account_refused=False)
        if isinstance(exc, anthropic.APIError | openai.APIError):
            # A reply the SDK could not validate, or a context overflow outside a status error: its body is not
            # repeated, and its class and status are what is known.
            provider = "anthropic" if isinstance(exc, anthropic.APIError) else "openai"
            status = getattr(exc, "status_code", None)
            window = " (the prompt is over the model's context window)" if overflow else ""
            at = f" {status}" if isinstance(status, int) else ""
            return ProviderFailure(
                f"{provider}{at}{window}: {type(exc).__name__} (the response body is withheld)",
                payload_withheld=True,
                account_refused=False,
            )
        if isinstance(exc, openrouter_errors.NoResponseError):
            return ProviderFailure(
                "openrouter: no response was received", payload_withheld=False, account_refused=False
            )
        if isinstance(exc, CircuitOpenError):
            return ProviderFailure(str(exc), payload_withheld=False, account_refused=exc.credential_scoped)
        if isinstance(exc, ModelProviderError):
            return self._model_provider(exc)
        if isinstance(exc, ModelCallTimeout):
            return ProviderFailure(
                f"the model call ran past its deadline ({exc})", payload_withheld=False, account_refused=False
            )
        if isinstance(exc, TimeoutError):
            return ProviderFailure(
                f"the call timed out ({type(exc).__name__})", payload_withheld=False, account_refused=False
            )
        if isinstance(exc, ModelsClientRefused | StructuredOutputError):
            # The adapter's own refusal, or models' refusal of the caller's schema: the caller's request, in words
            # that carry no provider payload.
            return ProviderFailure(str(exc), payload_withheld=False, account_refused=False)
        if isinstance(exc, pydantic.ValidationError):
            return ProviderFailure(
                f"the provider's reply did not match its SDK's schema ({exc.error_count()} errors from "
                f"{exc.title}; the reply is withheld)",
                payload_withheld=True,
                account_refused=False,
            )
        if isinstance(exc, ValueError):
            return self._bare_value_error(exc)
        return ProviderFailure(
            f"{type(exc).__name__} (its text is withheld: the adapter does not know what it carries)",
            payload_withheld=True,
            account_refused=False,
        )

    def _anthropic_status(self, exc: anthropic.APIStatusError, *, overflow: bool) -> ProviderFailure:
        message = _body_message(exc.body)
        status = exc.status_code
        credit = status == 400 and message is not None and _ANTHROPIC_CREDIT_NOTICE in message.lower()
        return ProviderFailure(
            _status_line("anthropic", status, message, overflow=overflow),
            payload_withheld=True,
            account_refused=status in {401, 402, 403} or credit,
        )

    def _openai_status(self, exc: openai.APIStatusError, *, overflow: bool) -> ProviderFailure:
        message = _body_message(exc.body)
        status = exc.status_code
        code = exc.code if isinstance(exc.code, str) else _error_field(exc.body, "code")
        return ProviderFailure(
            _status_line("openai", status, message, overflow=overflow),
            payload_withheld=True,
            account_refused=status in {401, 403} or (status == 429 and code == "insufficient_quota"),
        )

    def _openrouter(self, exc: openrouter_errors.OpenRouterError) -> ProviderFailure:
        status = exc.status_code
        account = status in {401, 402}
        if isinstance(exc, openrouter_errors.ResponseValidationError):
            return ProviderFailure(
                f"openrouter {status}: the reply did not match the SDK's schema (the reply is withheld)",
                payload_withheld=True,
                account_refused=False,
            )
        if isinstance(exc, openrouter_errors.OpenRouterDefaultError):
            return ProviderFailure(
                _status_line("openrouter", status, _body_message(exc.body), overflow=False),
                payload_withheld=True,
                account_refused=account,
            )
        message = exc.message if exc.message and exc.message != exc.body else None
        return ProviderFailure(
            _status_line("openrouter", status, message or _body_message(exc.body), overflow=False),
            payload_withheld=message is None,
            account_refused=account,
        )

    def _model_provider(self, exc: ModelProviderError) -> ProviderFailure:
        code = exc.reason or (str(exc.status) if exc.status is not None else "error")
        resets = exc.resets if isinstance(exc, ModelRateLimitError) else None
        until = f", resets {resets}" if resets else ""
        rejected = exc.rejected_output is not None or exc.rejection is not None
        return ProviderFailure(
            f"{exc.provider} ({code}{until}): {_clip(exc.detail)}",
            payload_withheld=rejected,
            account_refused=exc.reason in _ACCOUNT_REASONS or bool(resets),
        )

    def _bare_value_error(self, exc: ValueError) -> ProviderFailure:
        streamed = _OPENROUTER_STREAM_ERROR.fullmatch(str(exc))
        if streamed is not None:
            code = streamed.group("code")
            account = code in {"401", "402"}
            return ProviderFailure(
                f"openrouter {code} (mid-stream): {_clip(streamed.group('message'))}",
                payload_withheld=False,
                account_refused=account,
            )
        return ProviderFailure(
            "ValueError (its text is withheld: an SDK raises a bare ValueError carrying the response's error body)",
            payload_withheld=True,
            account_refused=False,
        )
