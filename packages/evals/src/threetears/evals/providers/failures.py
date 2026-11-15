"""A failed request, as the client raises it (:class:`ProviderError`) and as the engine may know it.

The client speaks HTTP itself, so every failure a provider answers with is one type, :class:`ProviderError`: the
provider, the status, and the provider's own message, never the envelope around it (on OpenRouter that envelope can
carry the account's id). The body is kept on the error for one reader only, :meth:`ProviderError.refusal`, which
hands it to the shared negotiation readers, so a refusal of a setting is answered and anything else is raised as it
came.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final, Literal

import httpx
import pydantic

from threetears.evals.negotiation import (
    SettingRefusal,
    read_anthropic_refusal,
    read_openai_refusal,
    read_router_refusal,
)
from threetears.evals.schema import FailureOwner, ProviderFailure

__all__ = ["Provider", "ProviderError", "describe_failure", "error_from_response"]

#: The providers the client speaks to: OpenRouter, Anthropic's Messages API, and any OpenAI-compatible chat endpoint.
Provider = Literal["openrouter", "anthropic", "openai"]

#: How much of a provider's own message an error keeps, in characters.
_MESSAGE_LIMIT: Final = 500

#: Anthropic's notice when an account's prepaid credit is spent, which arrives as a 400 rather than a 402.
_CREDIT_NOTICE: Final = "credit balance"

#: The statuses by which the provider declines to serve a call at that moment: a rate limit (429), service
#: unavailable (503), and Anthropic's overloaded (529). Each says nothing about the model and would meet any request.
#: A 500 or a 502 is an error rather than a refusal, so it stays the candidate's.
_DECLINED: Final = frozenset({429, 503, 529})

#: What OpenRouter's 404 for a model it has retired says ("inception/mercury-3 is no longer available on OpenRouter").
#: Its other 404s say no endpoint takes the request: none supports tool use, none matches the account's data policy,
#: none takes the parameters under ``require_parameters``, no allowed provider is left. Those are the request's or the
#: account's setup, which a resume would meet again with the model still served. The body carries no code that tells
#: the two apart (its ``code`` is the status), so this one model-not-served answer is read off the message, narrowly,
#: the way Anthropic's credit notice is.
_ROUTER_RETIRED_MODEL: Final = "is no longer available"

#: How OpenRouter's 400 for a model id it does not know ends, read off the message for the same reason.
_ROUTER_INVALID_MODEL: Final = "is not a valid model id"

#: The error code OpenAI, and the OpenAI-compatible servers that copy its envelope, answer for a model it does not
#: serve (``error.code``), whatever the status.
_OPENAI_MODEL_NOT_FOUND: Final = "model_not_found"

#: Anthropic's error type for a resource it does not have (``error.type``); on the Messages API that is the model.
_ANTHROPIC_NOT_FOUND: Final = "not_found_error"


def _parsed(body: object) -> object:
    """``body`` as JSON when it is text, else as given; ``None`` when it is text that is not JSON."""
    if isinstance(body, bytes | str):
        try:
            return json.loads(body)
        except ValueError:
            # NOSILENT: a body that is not JSON carries no message to quote and no refusal to read; the error still
            # names the provider and the status, and its message says the body was not JSON.
            return None
    return body


def _error_object(body: object) -> Mapping[str, object]:
    """The error object of a provider's body: its ``error`` member, or the body itself."""
    outer = body if isinstance(body, Mapping) else {}
    inner = outer.get("error")
    return inner if isinstance(inner, Mapping) else outer


def _model_unserved(provider: Provider, status: int, error: Mapping[str, object], message: str | None) -> bool:
    """Whether the provider's answer says it does not serve the model the request named.

    Read off each provider's structured signal: Anthropic's 404 ``not_found_error`` and OpenAI's ``model_not_found``
    code. OpenRouter's answers carry no code of their own, so its two are read off the message: the 404 for a model it
    has retired, and the 400 for a model id it does not know. Its other 404s (no endpoint takes the request: tool use,
    data policy, parameters, allowed providers) are not this: the model is still served, so they stay the candidate's.

    :param provider: The provider that answered.
    :ptype provider: Provider
    :param status: The HTTP status, or an in-body error's code.
    :ptype status: int
    :param error: The body's error object.
    :ptype error: Mapping[str, object]
    :param message: The provider's own message, or ``None``.
    :ptype message: str | None
    :return: True when every call naming that model would meet the same answer.
    :rtype: bool
    """
    if provider == "anthropic":
        return status == 404 and error.get("type") == _ANTHROPIC_NOT_FOUND
    if provider == "openai":
        return error.get("code") == _OPENAI_MODEL_NOT_FOUND
    # The router, the one provider left.
    said = (message or "").lower()
    if status == 404:
        return _ROUTER_RETIRED_MODEL in said
    return status == 400 and said.rstrip(" .").endswith(_ROUTER_INVALID_MODEL)


class ProviderError(Exception):
    """A request the provider refused or failed: who, the HTTP status, and the provider's own message.

    ``str(error)`` is the provider, the status and the message (at most 500 characters), never the response body,
    so it is safe to log and to store. ``owner`` says whose failure it was, read off the status, and off the body
    where one status means two things: the calling ``account`` when every model behind the key would get the same
    answer (a key it refuses, credit spent, a quota exhausted), so a run stops instead of charging the candidate; the
    ``provider`` when it declined the call (any other 429 rate limit, or a 529 overloaded or 503 unavailable), which
    says nothing about the model, so the cell is excluded and the run goes on; ``unserved`` when it does not serve the
    model the request named, so every call naming it meets the same answer and the cell is lost to the test setup;
    otherwise the ``candidate``, as any failure is charged. An in-body code OpenRouter answers with a 200 is read the
    same way.

    :ivar provider: The provider that answered.
    :ivar status: The HTTP status, or the code an in-body error named (OpenRouter answers some failures with a 200).
    :ivar message: The provider's own message, or ``None`` when its answer carried none.
    :ivar attempts: How many times the call was sent before this answer stood: more than one when the provider
        declined it and the client sent it again (:class:`~threetears.evals.providers.DeclinedRetry`).
    :ivar owner: Whose failure it was. ``account``: a 401 or 403 (OpenRouter's 403 is moderation, so not there), a
        402, Anthropic's 400 credit-balance notice, or OpenAI's 429 ``insufficient_quota``. ``provider``: any other
        429, a 529 or a 503. ``unserved``: OpenRouter's 404 for a model it has retired ("is no longer available"; its
        other 404s, no endpoint taking the request, stay the candidate's) and its 400 for a model id it does not know, Anthropic's 404 ``not_found_error``, or OpenAI's
        ``model_not_found``. ``candidate``: everything else, a 500 and a 502 among it.
    :ivar credit_spent: Whether the answer says the key's prepaid credit is spent: Anthropic's 400 credit-balance
        notice, or a 402 (OpenRouter's insufficient credits). An ``account`` failure that another key may not meet,
        so it is the one answer a model fails over to a fallback key on (``<VARIABLE>_FALLBACKS``).
    """

    def __init__(
        self, provider: Provider, status: int, message: str | None, *, body: object = None, attempts: int = 1
    ) -> None:
        """Name the failure; ``body`` is kept for :meth:`refusal` alone and never shown.

        :param provider: The provider that answered.
        :ptype provider: Provider
        :param status: The HTTP status, or an in-body error's code.
        :ptype status: int
        :param message: The provider's own message, or ``None``.
        :ptype message: str | None
        :param body: The response body, parsed or as text, which the refusal readers read.
        :ptype body: object
        :param attempts: How many times the call was sent, this answer's attempt included.
        :ptype attempts: int
        """
        self.provider: Provider = provider
        self.status = status
        self.message = message[:_MESSAGE_LIMIT] if message else None
        self.attempts = attempts
        self._body = body
        error = _error_object(_parsed(body))
        code = error.get("code")
        credit = status == 400 and provider == "anthropic" and _CREDIT_NOTICE in (message or "").lower()
        account = (
            status == 401
            or status == 402
            or (status == 403 and provider != "openrouter")
            or credit
            or (status == 429 and code == "insufficient_quota")
        )
        self.owner: FailureOwner = (
            "account"
            if account
            else "provider"
            if status in _DECLINED
            else "unserved"
            if _model_unserved(provider, status, error, message)
            else "candidate"
        )
        self.credit_spent = credit or status == 402
        said = f": {self.message}" if self.message else " (the response carried no message)"
        tried = f" (after {attempts} attempts)" if attempts > 1 else ""
        super().__init__(f"{provider} {status}{said}{tried}")

    def refusal(self) -> SettingRefusal | None:
        """The setting this answer refused, read off the status and body by the provider's shared reader, or ``None``.

        What the negotiation answers with a resend (:func:`~threetears.evals.negotiation.negotiate`): a refusal of a
        setting the request carried. Anything else (``None``) is raised as it came.

        :return: The refusal, or ``None``.
        :rtype: SettingRefusal | None
        """
        if self.provider == "openrouter":
            return read_router_refusal(self.status, self._body)
        if self.provider == "anthropic":
            return read_anthropic_refusal(self.status, self._body)
        return read_openai_refusal(self.status, self._body)


def error_from_response(provider: Provider, status: int, body: object, *, attempts: int = 1) -> ProviderError:
    """The error for a response the provider answered with a failure, its message read off the body.

    :param provider: The provider that answered.
    :ptype provider: Provider
    :param status: The HTTP status (or an in-body error's code).
    :ptype status: int
    :param body: The body, parsed or as text.
    :ptype body: object
    :param attempts: How many times the call was sent, this answer's attempt included.
    :ptype attempts: int
    :return: The error to raise.
    :rtype: ProviderError
    """
    parsed = _parsed(body)
    message = _error_object(parsed).get("message")
    if parsed is None and body:
        message = "the response body is not JSON"
    text = message if isinstance(message, str) and message else None
    return ProviderError(provider, status, text, body=body, attempts=attempts)


def describe_failure(exc: BaseException) -> ProviderFailure:
    """What the engine may know of a call through the client that raised: the host's ``failure_describer``.

    Total over ``BaseException``. A :class:`ProviderError` is its own text (the provider, the status, the
    message; never the body) with its ``owner``. An ``httpx`` timeout or transport error names the host it
    was talking to and the error's type. A reply :meth:`~threetears.evals.providers.Model.json` could not validate
    names the schema and the first error, leaving out the reply itself. Anything else is withheld, since its text is
    unknown, as :func:`~threetears.evals.kernel.withhold_failure_detail` does. Every failure but a
    :class:`ProviderError` is the ``candidate``'s: nothing in it says the account or the provider declined the call.

    :param exc: Whatever the client raised, or anything else the engine caught around the call.
    :ptype exc: BaseException
    :return: The failure as the engine may record it.
    :rtype: ProviderFailure
    """
    if isinstance(exc, ProviderError):
        return ProviderFailure(str(exc), payload_withheld=False, owner=exc.owner)
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        host = exc.request.url.host if _has_request(exc) else "the provider"
        what = "timed out" if isinstance(exc, httpx.TimeoutException) else "could not be sent"
        return ProviderFailure(
            f"the request to {host} {what} ({type(exc).__name__})", payload_withheld=False, owner="candidate"
        )
    if isinstance(exc, TimeoutError):
        return ProviderFailure(f"the call timed out ({type(exc).__name__})", payload_withheld=False, owner="candidate")
    if isinstance(exc, pydantic.ValidationError):
        first = exc.errors()[0] if exc.error_count() else None
        where = ".".join(str(part) for part in first["loc"]) if first else ""
        said = f"; first: {where or '/'}: {first['msg']}" if first else ""
        return ProviderFailure(
            f"the reply did not validate as {exc.title} ({exc.error_count()} errors{said}; the reply is withheld)",
            payload_withheld=True,
            owner="candidate",
        )
    return ProviderFailure(
        f"{type(exc).__name__} (its text is withheld: the client does not know what it carries)",
        payload_withheld=True,
        owner="candidate",
    )


def _has_request(exc: httpx.TransportError | httpx.TimeoutException) -> bool:
    """Whether ``exc`` carries the request it failed on: ``httpx`` raises ``RuntimeError`` reading one it lacks."""
    try:
        _ = exc.request  # the property raises when the error was built without a request
    except RuntimeError:
        # NOSILENT: an error built without its request names no host; the description says "the provider" instead.
        return False
    return True
