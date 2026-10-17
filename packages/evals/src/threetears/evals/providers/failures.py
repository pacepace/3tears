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
from threetears.evals.schema import ProviderFailure

__all__ = ["Provider", "ProviderError", "describe_failure", "error_from_response"]

#: The providers the client speaks to: OpenRouter, Anthropic's Messages API, and any OpenAI-compatible chat endpoint.
Provider = Literal["openrouter", "anthropic", "openai"]

#: How much of a provider's own message an error keeps, in characters.
_MESSAGE_LIMIT: Final = 500

#: Anthropic's notice when an account's prepaid credit is spent, which arrives as a 400 rather than a 402.
_CREDIT_NOTICE: Final = "credit balance"


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


class ProviderError(Exception):
    """A request the provider refused or failed: who, the HTTP status, and the provider's own message.

    ``str(error)`` is the provider, the status and the message (at most 500 characters), never the response body,
    so it is safe to log and to store. ``account`` says the provider refused the calling account rather than the
    request (a key it refuses, credit spent, a quota exhausted): every model behind the key would get the same
    answer, so a run stops instead of charging the candidate.

    :ivar provider: The provider that answered.
    :ivar status: The HTTP status, or the code an in-body error named (OpenRouter answers some failures with a 200).
    :ivar message: The provider's own message, or ``None`` when its answer carried none.
    :ivar account: Whether the account was refused: a 401 or 403 (OpenRouter's 403 is moderation, so not there), a
        402, Anthropic's 400 credit-balance notice, or OpenAI's 429 ``insufficient_quota``.
    """

    def __init__(self, provider: Provider, status: int, message: str | None, *, body: object = None) -> None:
        """Name the failure; ``body`` is kept for :meth:`refusal` alone and never shown.

        :param provider: The provider that answered.
        :ptype provider: Provider
        :param status: The HTTP status, or an in-body error's code.
        :ptype status: int
        :param message: The provider's own message, or ``None``.
        :ptype message: str | None
        :param body: The response body, parsed or as text, which the refusal readers read.
        :ptype body: object
        """
        self.provider: Provider = provider
        self.status = status
        self.message = message[:_MESSAGE_LIMIT] if message else None
        self._body = body
        code = _error_object(_parsed(body)).get("code")
        credit = status == 400 and provider == "anthropic" and _CREDIT_NOTICE in (message or "").lower()
        self.account = (
            status == 401
            or status == 402
            or (status == 403 and provider != "openrouter")
            or credit
            or (status == 429 and code == "insufficient_quota")
        )
        said = f": {self.message}" if self.message else " (the response carried no message)"
        super().__init__(f"{provider} {status}{said}")

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


def error_from_response(provider: Provider, status: int, body: object) -> ProviderError:
    """The error for a response the provider answered with a failure, its message read off the body.

    :param provider: The provider that answered.
    :ptype provider: Provider
    :param status: The HTTP status (or an in-body error's code).
    :ptype status: int
    :param body: The body, parsed or as text.
    :ptype body: object
    :return: The error to raise.
    :rtype: ProviderError
    """
    parsed = _parsed(body)
    message = _error_object(parsed).get("message")
    if parsed is None and body:
        message = "the response body is not JSON"
    return ProviderError(provider, status, message if isinstance(message, str) and message else None, body=body)


def describe_failure(exc: BaseException) -> ProviderFailure:
    """What the engine may know of a call through the client that raised: the host's ``failure_describer``.

    Total over ``BaseException``. A :class:`ProviderError` is its own text (the provider, the status, the
    message; never the body) with its ``account`` flag. An ``httpx`` timeout or transport error names the host it
    was talking to and the error's type. A reply :meth:`~threetears.evals.providers.Model.json` could not validate
    names the schema and the first error, leaving out the reply itself. Anything else is withheld, since its text is
    unknown, as :func:`~threetears.evals.kernel.withhold_failure_detail` does.

    :param exc: Whatever the client raised, or anything else the engine caught around the call.
    :ptype exc: BaseException
    :return: The failure as the engine may record it.
    :rtype: ProviderFailure
    """
    if isinstance(exc, ProviderError):
        return ProviderFailure(str(exc), payload_withheld=False, account_refused=exc.account)
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        host = exc.request.url.host if _has_request(exc) else "the provider"
        what = "timed out" if isinstance(exc, httpx.TimeoutException) else "could not be sent"
        return ProviderFailure(
            f"the request to {host} {what} ({type(exc).__name__})", payload_withheld=False, account_refused=False
        )
    if isinstance(exc, TimeoutError):
        return ProviderFailure(
            f"the call timed out ({type(exc).__name__})", payload_withheld=False, account_refused=False
        )
    if isinstance(exc, pydantic.ValidationError):
        first = exc.errors()[0] if exc.error_count() else None
        where = ".".join(str(part) for part in first["loc"]) if first else ""
        said = f"; first: {where or '/'}: {first['msg']}" if first else ""
        return ProviderFailure(
            f"the reply did not validate as {exc.title} ({exc.error_count()} errors{said}; the reply is withheld)",
            payload_withheld=True,
            account_refused=False,
        )
    return ProviderFailure(
        f"{type(exc).__name__} (its text is withheld: the client does not know what it carries)",
        payload_withheld=True,
        account_refused=False,
    )


def _has_request(exc: httpx.TransportError | httpx.TimeoutException) -> bool:
    """Whether ``exc`` carries the request it failed on: ``httpx`` raises ``RuntimeError`` reading one it lacks."""
    try:
        _ = exc.request  # the property raises when the error was built without a request
    except RuntimeError:
        # NOSILENT: an error built without its request names no host; the description says "the provider" instead.
        return False
    return True
