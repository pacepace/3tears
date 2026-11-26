"""A failed request, as the client raises it: :class:`ProviderError`, and the refusal of a setting read off it.

The client speaks HTTP itself, so every failure a provider answers with is one type, :class:`ProviderError`: the
provider, the status, and the provider's own message, never the envelope around it (on OpenRouter that envelope can
carry the account's id). The body is kept on the error for one reader only, :meth:`ProviderError.refusal`, which
hands it to the shared negotiation readers, so a refusal of a setting is answered and anything else is raised as it
came.

It imports nothing of the engine at load, so a host that imports :class:`ProviderError` only to catch it pays for
this module alone; what the engine may know of a failure is :mod:`threetears.evals.providers.describe`'s.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from threetears.evals.negotiation import SettingRefusal
    from threetears.evals.schema import FailureOwner

__all__ = ["Provider", "ProviderError", "error_from_response"]

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

#: The ``error.metadata`` keys OpenRouter's errors page documents for a 403 that blocked the request's content: a
#: guardrail's matched ``patterns`` ("Request blocked: ..."), and moderation's ``reasons`` and ``flagged_input``. A
#: 403 carrying one is the candidate's; any other OpenRouter 403 (a key lacking a permission, a spent budget, a body
#: that says nothing) is the account's, as every other provider's 403 is.
_ROUTER_CONTENT_BLOCK: Final = frozenset({"patterns", "reasons", "flagged_input"})

#: The typed ``error_type`` values OpenRouter's errors page gives a 403 that the content or the model earned: a content
#: filter's flag, and a provider's refusal reported as an error. It sits in ``error.metadata`` on the chat completions
#: wire and beside the message on the Messages one.
_ROUTER_CONTENT_TYPES: Final = frozenset({"content_policy_violation", "refusal"})

#: The ``error.metadata`` key naming which budget an OpenRouter 403 found spent (a guardrail's, a workspace's or the
#: key's own limit), rejected before any provider saw the call.
_ROUTER_LIMIT_SOURCE: Final = "limit_source"

#: The ``error.metadata`` key holding when that budget's window rolls over (ISO 8601 UTC), null for a lifetime budget.
_ROUTER_RESETS_AT: Final = "resets_at"

#: The ``limit_source`` of a budget that is the key's own limit, which another key does not share. A guardrail's or a
#: workspace's budget (or a source the errors page does not name) is met by every key under the account.
_ROUTER_KEY_LIMIT: Final = "openrouter_key_limit"


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


def _metadata(error: Mapping[str, object]) -> Mapping[str, object]:
    """The error object's ``metadata`` member, or an empty one when it has none."""
    metadata = error.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _router_403(provider: Provider, status: int) -> bool:
    """Whether the answer is a 403 from the router, the one provider whose 403 its body must tell apart."""
    return provider == "openrouter" and status == 403


def _router_content_block(provider: Provider, status: int, error: Mapping[str, object]) -> bool:
    """Whether the answer is OpenRouter's 403 for the request's content: a guardrail's or moderation's block.

    Read off the metadata the errors page documents for one (:data:`_ROUTER_CONTENT_BLOCK`, or a typed
    :data:`_ROUTER_CONTENT_TYPES` ``error_type``), never off the message, which a permission refusal and a budget share
    the status with. A 403 naming a spent budget (``limit_source``) is the account's whatever else it carries: the
    call was rejected before any provider or filter saw it.

    :param provider: The provider that answered.
    :param status: The HTTP status, or an in-body error's code.
    :param error: The body's error object.
    :return: True for a 403 OpenRouter answered with such metadata; False for any other answer.
    """
    if not _router_403(provider, status):
        return False
    metadata = _metadata(error)
    if metadata.get(_ROUTER_LIMIT_SOURCE) is not None:
        return False
    error_type = metadata.get("error_type", error.get("error_type"))
    return error_type in _ROUTER_CONTENT_TYPES or not _ROUTER_CONTENT_BLOCK.isdisjoint(metadata)


def _router_budget_resets(provider: Provider, status: int, error: Mapping[str, object]) -> datetime | None:
    """When the budget OpenRouter's 403 found spent rolls over, or ``None``.

    ``None`` for any answer but OpenRouter's budget 403 (``error.metadata.limit_source``), and for one whose
    ``resets_at`` is null (a lifetime budget) or not an ISO 8601 instant.

    :param provider: The provider that answered.
    :param status: The HTTP status, or an in-body error's code.
    :param error: The body's error object.
    :return: The instant the budget resets, in UTC.
    """
    if not _router_403(provider, status):
        return None
    metadata = _metadata(error)
    resets = metadata.get(_ROUTER_RESETS_AT)
    if metadata.get(_ROUTER_LIMIT_SOURCE) is None or not isinstance(resets, str):
        return None
    try:
        when = datetime.fromisoformat(resets)
    except ValueError:
        # NOSILENT: a reset that is not an instant names no time to wait for; the failure is still the account's,
        # and the provider's own message, which the error keeps, says what it said.
        return None
    return when.astimezone(UTC) if when.tzinfo is not None else when.replace(tzinfo=UTC)


def _model_unserved(provider: Provider, status: int, error: Mapping[str, object], message: str | None) -> bool:
    """Whether the provider's answer says it does not serve the model the request named.

    Read off each provider's structured signal: Anthropic's 404 ``not_found_error`` and OpenAI's ``model_not_found``
    code. OpenRouter's answers carry no code of their own, so its two are read off the message: the 404 for a model it
    has retired, and the 400 for a model id it does not know. Its other 404s (no endpoint takes the request: tool use,
    data policy, parameters, allowed providers) are not this: the model is still served, so they stay the candidate's.

    :param provider: The provider that answered.
    :param status: The HTTP status, or an in-body error's code.
    :param error: The body's error object.
    :param message: The provider's own message, or ``None``.
    :return: True when every call naming that model would meet the same answer.
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
    :ivar owner: Whose failure it was. ``account``: a 401 or 403 (on OpenRouter, every 403 but a content block: a key
        lacking a permission, a spent guardrail, workspace or key budget, a body that says neither), a 402,
        Anthropic's 400 credit-balance notice, or OpenAI's 429 ``insufficient_quota``. ``provider``: any other
        429, a 529 or a 503. ``unserved``: OpenRouter's 404 for a model it has retired ("is no longer available"; its
        other 404s, no endpoint taking the request, stay the candidate's) and its 400 for a model id it does not know, Anthropic's 404 ``not_found_error``, or OpenAI's
        ``model_not_found``. ``candidate``: OpenRouter's 403 that blocked the request's content, which its body says
        with a guardrail's ``patterns``, moderation's ``reasons`` or ``flagged_input``, or an ``error_type`` of
        ``content_policy_violation`` or ``refusal``; and everything else, a 500 and a 502 among it.
    :ivar credit_spent: Whether the answer says the key's prepaid credit or its own limit is spent: Anthropic's 400
        credit-balance notice, a 402 (OpenRouter's insufficient credits), or OpenRouter's 403 naming the key's own
        spent limit (``limit_source`` :data:`_ROUTER_KEY_LIMIT`). An ``account`` failure that another key may not meet,
        so it is the one answer a model fails over to a fallback key on (``<VARIABLE>_FALLBACKS``). A guardrail's or a
        workspace's spent budget is not: every key under the account meets it.
    :ivar resets_at: When the spent budget rolls over, in UTC, for OpenRouter's 403 naming a spent guardrail,
        workspace or key budget (``error.metadata.limit_source``, its ``resets_at``): an ``account`` failure that
        lasts until then. ``None`` for every other answer, and for a budget with no reset (a lifetime one).
    """

    def __init__(
        self, provider: Provider, status: int, message: str | None, *, body: object = None, attempts: int = 1
    ) -> None:
        """Name the failure; ``body`` is kept for :meth:`refusal` alone and never shown.

        :param provider: The provider that answered.
        :param status: The HTTP status, or an in-body error's code.
        :param message: The provider's own message, or ``None``.
        :param body: The response body, parsed or as text, which the refusal readers read.
        :param attempts: How many times the call was sent, this answer's attempt included.
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
            or (status == 403 and not _router_content_block(provider, status, error))
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
        key_limit = _router_403(provider, status) and _metadata(error).get(_ROUTER_LIMIT_SOURCE) == _ROUTER_KEY_LIMIT
        self.credit_spent = credit or status == 402 or key_limit
        self.resets_at = _router_budget_resets(provider, status, error)
        said = f": {self.message}" if self.message else " (the response carried no message)"
        tried = f" (after {attempts} attempts)" if attempts > 1 else ""
        super().__init__(f"{provider} {status}{said}{tried}")

    def refusal(self) -> SettingRefusal | None:
        """The setting this answer refused, read off the status and body by the provider's shared reader, or ``None``.

        What the negotiation answers with a resend (:func:`~threetears.evals.negotiation.negotiate`): a refusal of a
        setting the request carried. Anything else (``None``) is raised as it came.

        :return: The refusal, or ``None``.
        """
        # Imported here, not at the top, so importing ProviderError loads no engine. The one caller is Model's
        # negotiation loop (model.py, read_refusal), and model.py imports the negotiation root at its top, so by then
        # this binds three names and loads nothing.
        from threetears.evals.negotiation import read_anthropic_refusal, read_openai_refusal, read_router_refusal

        if self.provider == "openrouter":
            return read_router_refusal(self.status, self._body)
        if self.provider == "anthropic":
            return read_anthropic_refusal(self.status, self._body)
        return read_openai_refusal(self.status, self._body)


def error_from_response(provider: Provider, status: int, body: object, *, attempts: int = 1) -> ProviderError:
    """The error for a response the provider answered with a failure, its message read off the body.

    :param provider: The provider that answered.
    :param status: The HTTP status (or an in-body error's code).
    :param body: The body, parsed or as text.
    :param attempts: How many times the call was sent, this answer's attempt included.
    :return: The error to raise.
    """
    parsed = _parsed(body)
    message = _error_object(parsed).get("message")
    if parsed is None and body:
        message = "the response body is not JSON"
    text = message if isinstance(message, str) and message else None
    return ProviderError(provider, status, text, body=body, attempts=attempts)
