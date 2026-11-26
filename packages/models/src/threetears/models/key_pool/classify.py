"""What a failed model call says about the key it ran on, one classifier per provider.

The pool core holds no status codes, error types or header names (KP-C1). It hands the exception an
attempt raised to its provider's :class:`KeyErrorClassifier` and acts on the :class:`Verdict` it gets
back: move to another key, hold this one until a time, or let the failure propagate unchanged.

An exception a classifier does not recognise is :attr:`VerdictKind.NOT_KEY_RELATED`. An unknown error
never burns a key; the worst it can do is fail the call the way it would have without a pool.

What each classifier reads, and where each rule's shape comes from:

- **anthropic** -- the SDK's ``APIStatusError`` (LangChain re-raises it as a subclass, so ``response``,
  ``body`` and ``type`` survive). The body is ``{"type": "error", "error": {"type": ..., "message":
  ..., "details": {"error_code": ...}}}`` (``anthropic.types.shared.ErrorResponse``; the
  ``details.error_code`` field is from the Claude API errors page). An error event inside a stream
  arrives as the same class carrying the stream's 200 status, so the error type decides, not the
  status. ``retry-after`` (and the SDK's own ``retry-after-ms``) first; then the
  ``anthropic-ratelimit-*-reset`` headers (RFC 3339) of a limit whose ``-remaining`` is 0.
- **openai** -- the SDK's ``APIStatusError``, whose ``body`` is the inner error object and whose
  ``code`` and ``type`` the SDK lifts from it; and its bare ``APIError`` for an error inside a stream,
  which carries no status. ``retry-after`` / ``retry-after-ms`` first; then the
  ``x-ratelimit-reset-*`` headers (a duration such as ``6m0s``) of a limit whose ``remaining`` is 0.
- **openrouter** -- the ``openrouter`` SDK's ``OpenRouterError`` (``status_code``, ``headers``, the
  raw ``body``), raw ``httpx`` transport errors (the SDK re-raises them unwrapped), and the
  ``ValueError`` ``langchain-openrouter`` raises for an error the API sends inside a 200, ending
  ``(code: N)``. Only ``retry-after``: the ``X-RateLimit-Reset`` OpenRouter sends on some 429s has no
  documented format.

``may_have_billed`` (KP-E4) is ``False`` only for a refusal before any answer: a 4xx status, or a
connection that never reached the provider. Everything else -- a 5xx, a read timeout, an error after a
stream began, an exception nothing here recognises -- may have been billed, so the pool settles it in
full rather than release it. Whether a stream began is the pool's to say (``stream_started``), since a
connection dropped mid-stream raises the same transport error as one that never connected; an error
that is itself a mid-stream shape counts as started whatever the pool says.
"""

from __future__ import annotations

import enum
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import anthropic
import httpx
import openai
import openrouter.errors

from threetears.models.providers.anthropic import ANTHROPIC_PROVIDER_NAME
from threetears.models.providers.openai import OPENAI_PROVIDER_NAME
from threetears.models.providers.openrouter import OPENROUTER_PROVIDER_NAME

__all__ = [
    "KeyErrorClassifier",
    "Verdict",
    "VerdictKind",
    "VerdictScope",
    "classifier_for",
]


class VerdictKind(enum.StrEnum):
    """What a failure says about the key it ran on."""

    #: the key's (or its account's) credit or spend limit is used up
    EXHAUSTED = "exhausted"
    #: the provider no longer accepts the key
    REVOKED = "revoked"
    #: the key may not use this model or resource
    FORBIDDEN = "forbidden"
    #: the key is rate-limited until :attr:`Verdict.until` (or the pool's floor, when that is ``None``)
    RATE_LIMITED = "rate_limited"
    #: the failure is not the key's; it propagates unchanged
    NOT_KEY_RELATED = "not_key_related"


class VerdictScope(enum.StrEnum):
    """Which keys a verdict applies to."""

    #: this key only
    KEY = "key"
    #: this key, for this model only
    KEY_AND_MODEL = "key_and_model"
    #: every key sharing this key's ``account`` (KP-C5)
    ACCOUNT = "account"


@dataclass(frozen=True, slots=True)
class Verdict:
    """A classifier's reading of one failed attempt.

    :ivar kind: what the failure says about the key
    :ivar scope: which keys it applies to; :attr:`VerdictScope.KEY` for a failure that is not the key's,
        where it means nothing
    :ivar may_have_billed: whether the provider may have charged for the attempt, so its reservation is
        settled in full rather than released (KP-E4)
    :ivar until: for :attr:`VerdictKind.RATE_LIMITED`, when the provider said the key may serve again, in
        UTC; ``None`` when it did not say, and always ``None`` for every other kind
    """

    kind: VerdictKind
    scope: VerdictScope
    may_have_billed: bool
    until: datetime | None = None

    def __post_init__(self) -> None:
        """refuse an ``until`` on a kind that has none, or one without a timezone.

        :raises ValueError: when ``until`` is set on a kind other than rate-limited, or is naive
        """
        if self.until is None:
            return
        if self.kind is not VerdictKind.RATE_LIMITED:
            raise ValueError(f"a {self.kind} verdict carries no until, got {self.until!r}")
        if self.until.tzinfo is None:
            raise ValueError(f"until must be timezone-aware, got {self.until!r}")

    @property
    def moves_key(self) -> bool:
        """Whether the failure is the key's, so the pool records it and tries another key.

        :return: ``True`` for every kind but :attr:`VerdictKind.NOT_KEY_RELATED`
        :rtype: bool
        """
        return self.kind is not VerdictKind.NOT_KEY_RELATED


class KeyErrorClassifier(Protocol):
    """Reads what one provider's failed attempt says about the key it ran on."""

    def classify(self, exc: BaseException, *, now: datetime, stream_started: bool = False) -> Verdict:
        """the verdict on the attempt that raised ``exc``.

        :param exc: what the attempt raised
        :ptype exc: BaseException
        :param now: the current time, timezone-aware, that a ``retry-after`` in seconds counts from
        :ptype now: datetime
        :param stream_started: whether the attempt had begun receiving the provider's answer before it
            failed; an attempt that had may have been billed whatever it raised
        :ptype stream_started: bool
        :return: the verdict
        :rtype: Verdict
        :raises ValueError: when ``now`` is naive
        """
        ...


# -- shared reading ------------------------------------------------------------------------------

#: The HTTP statuses a provider refuses a request with before doing any work for it: a 4xx.
_CLIENT_ERROR = range(400, 500)

#: A provider's own wait, in milliseconds. Not standard, but both Stainless SDKs honour it ahead of
#: ``retry-after``, so a provider that sends it means it.
_RETRY_AFTER_MS = "retry-after-ms"
_RETRY_AFTER = "retry-after"


def _utc(now: datetime) -> datetime:
    """``now`` in UTC, refusing a naive time.

    :param now: the caller's clock reading
    :ptype now: datetime
    :return: the same instant in UTC
    :rtype: datetime
    :raises ValueError: when ``now`` has no timezone
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be timezone-aware, got {now!r}")
    return now.astimezone(UTC)


def _after(now: datetime, seconds: float) -> datetime | None:
    """``now`` plus ``seconds``, a negative wait counting as none; ``None`` when it is not a usable number.

    :param now: the time the wait counts from, in UTC
    :ptype now: datetime
    :param seconds: the wait
    :ptype seconds: float
    :return: when the wait ends, or ``None`` for a non-finite or unrepresentable wait
    :rtype: datetime | None
    """
    if not math.isfinite(seconds):
        return None
    try:
        return now + timedelta(seconds=max(seconds, 0.0))
    except OverflowError:  # NOSILENT: a wait past the calendar's end is no time; the pool's floor holds the key
        return None


def _as_utc(moment: datetime) -> datetime:
    """``moment`` in UTC; a naive one is taken to be UTC already, as an HTTP date's ``-0000`` is."""
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _retry_after(headers: Mapping[str, str], now: datetime) -> datetime | None:
    """when ``retry-after`` says to come back: ``retry-after-ms``, then seconds, then an HTTP date.

    :param headers: the response's headers (case-insensitive, as ``httpx`` keeps them)
    :ptype headers: Mapping[str, str]
    :param now: the time a wait in seconds counts from, in UTC
    :ptype now: datetime
    :return: the time, or ``None`` when neither header is present and readable
    :rtype: datetime | None
    """
    millis = headers.get(_RETRY_AFTER_MS)
    if millis is not None:
        try:
            return _after(now, float(millis) / 1000)
        except ValueError:  # NOSILENT: an unreadable retry-after-ms falls through to retry-after
            pass
    value = headers.get(_RETRY_AFTER)
    if value is None:
        return None
    try:
        return _after(now, float(value))
    except ValueError:  # NOSILENT: not seconds, so it is read as an HTTP date next
        pass
    try:
        return max(_as_utc(parsedate_to_datetime(value)), now)
    except TypeError, ValueError:  # NOSILENT: an unreadable retry-after is an absent one
        return None


def _latest_exhausted_reset(
    headers: Mapping[str, str], limits: tuple[tuple[str, str], ...], read: Callable[[str], datetime | None]
) -> datetime | None:
    """the latest reset among the limits a response reports as used up.

    A 429 names no limit; the one it hit is the one with nothing remaining. When several are, the key
    can serve again only once all of them reset, so the latest wins.

    :param headers: the response's headers
    :ptype headers: Mapping[str, str]
    :param limits: each limit's ``(remaining header, reset header)``
    :ptype limits: tuple[tuple[str, str], ...]
    :param read: turns a reset header's value into a time, or ``None`` when it cannot
    :ptype read: Callable[[str], datetime | None]
    :return: the latest such reset, or ``None`` when no limit reports zero remaining with a readable reset
    :rtype: datetime | None
    """
    resets: list[datetime] = []
    for remaining, reset in limits:
        if headers.get(remaining, "").strip() != "0":
            continue
        value = headers.get(reset)
        moment = read(value) if value is not None else None
        if moment is not None:
            resets.append(moment)
    return max(resets) if resets else None


def _billed(*, status: int | None, stream_started: bool) -> bool:
    """whether an answered attempt may have been billed: anything but a 4xx refusal before a stream.

    :param status: the HTTP status of the failure, ``None`` when it carried none
    :ptype status: int | None
    :param stream_started: whether the answer had begun
    :ptype stream_started: bool
    :return: ``False`` only for a 4xx before any answer began
    :rtype: bool
    """
    return stream_started or status not in _CLIENT_ERROR


def _transport_verdict(exc: BaseException, *, stream_started: bool) -> Verdict | None:
    """the verdict on an ``httpx`` transport failure, or ``None`` when ``exc`` is not one.

    A connection that was never made, or a request that never left the pool of connections, cannot
    have been billed. A read that failed or timed out may have followed the provider's work.

    :param exc: what the attempt raised
    :ptype exc: BaseException
    :param stream_started: whether the answer had begun
    :ptype stream_started: bool
    :return: a not-key-related verdict, or ``None``
    :rtype: Verdict | None
    """
    if not isinstance(exc, httpx.TransportError):
        return None
    never_sent = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
    return Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=stream_started or not never_sent)


#: The verdict on an exception no rule recognises: not the key's, and possibly billed, since nothing
#: says it failed before the provider did any work.
_UNRECOGNISED = Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=True)


# -- anthropic -----------------------------------------------------------------------------------

#: ``error.details.error_code`` on the Messages API's 429 when the organisation's monthly spend cap is
#: reached (Claude API errors page, "Error codes"; rate-limits page, "Reaching your spend cap").
_ANTHROPIC_SPEND_CAP_CODE = "enforced_spend_limit_reached"

#: The starts of a 400 ``invalid_request_error`` message that says a spend limit the organisation or
#: workspace set was reached (Claude API rate-limits page, "Setting your own spend limit"). The API
#: gives this case no type or code of its own, so the message is the only signal.
#: Fixture: ``ANTHROPIC_SPEND_LIMIT_400`` / ``ANTHROPIC_WORKSPACE_SPEND_LIMIT_400``.
_ANTHROPIC_SPEND_LIMIT_PREFIXES = (
    "you have reached your specified api usage limits",
    "you have reached your specified workspace api usage limits",
)

#: Part of the 400 ``invalid_request_error`` message the API sends when prepaid credit runs out
#: (``errors.py``'s ``_extract_provider_body_message`` quotes it from a production incident). No type
#: or code marks it either. Fixture: ``ANTHROPIC_CREDIT_TOO_LOW_400``.
_ANTHROPIC_CREDIT_TOO_LOW = "credit balance is too low"

#: The ``anthropic-ratelimit-*`` limits a 429 may have used up, each ``(remaining, reset)``; the reset
#: is RFC 3339 (Claude API rate-limits page, "Response headers").
_ANTHROPIC_LIMITS = tuple(
    (f"{prefix}-{limit}-remaining", f"{prefix}-{limit}-reset")
    for prefix, limit in (
        ("anthropic-ratelimit", "requests"),
        ("anthropic-ratelimit", "tokens"),
        ("anthropic-ratelimit", "input-tokens"),
        ("anthropic-ratelimit", "output-tokens"),
        ("anthropic-priority", "input-tokens"),
        ("anthropic-priority", "output-tokens"),
    )
)


def _rfc3339(value: str) -> datetime | None:
    """an RFC 3339 time in UTC, or ``None`` when ``value`` is not one."""
    try:
        return _as_utc(datetime.fromisoformat(value.strip()))
    except ValueError:  # NOSILENT: an unreadable reset header is an absent one; the pool's floor holds the key
        return None


def _anthropic_error(body: object) -> dict[str, Any]:
    """the ``error`` object of an Anthropic error body, or an empty one when the body has none."""
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


class _AnthropicClassifier:
    """KP-C2's rules for the Anthropic API. Credit and rate limits belong to the organisation (or a
    workspace in it), so both are account-scoped; a key is revoked alone, and forbidden per model."""

    def classify(self, exc: BaseException, *, now: datetime, stream_started: bool = False) -> Verdict:
        """the verdict on the attempt that raised ``exc``; see :class:`KeyErrorClassifier`."""
        now = _utc(now)
        if isinstance(exc, anthropic.APIConnectionError):
            # a timeout may have followed the provider's work; a refused connection cannot have
            billed = stream_started or isinstance(exc, anthropic.APITimeoutError)
            return Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=billed)
        if not isinstance(exc, anthropic.APIStatusError):
            return _transport_verdict(exc, stream_started=stream_started) or _UNRECOGNISED

        status = exc.status_code
        # an error event inside a stream carries the stream's own 200: it began, whatever the pool saw
        started = stream_started or status < 400
        billed = _billed(status=status, stream_started=started)
        error = _anthropic_error(exc.body)
        error_type = error.get("type")
        details = error.get("details")
        code = details.get("error_code") if isinstance(details, dict) else None
        message = str(error.get("message") or "").strip().lower()

        verdict = Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=billed)
        if code == _ANTHROPIC_SPEND_CAP_CODE or error_type == "billing_error" or status == 402:
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, may_have_billed=billed)
        elif (error_type == "invalid_request_error" or status == 400) and (
            message.startswith(_ANTHROPIC_SPEND_LIMIT_PREFIXES) or _ANTHROPIC_CREDIT_TOO_LOW in message
        ):
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, may_have_billed=billed)
        elif error_type == "authentication_error" or status == 401:
            verdict = Verdict(VerdictKind.REVOKED, VerdictScope.KEY, may_have_billed=billed)
        elif error_type == "permission_error" or status == 403:
            verdict = Verdict(VerdictKind.FORBIDDEN, VerdictScope.KEY_AND_MODEL, may_have_billed=billed)
        elif error_type == "rate_limit_error" or status == 429:
            headers = exc.response.headers
            until = _retry_after(headers, now) or _latest_exhausted_reset(headers, _ANTHROPIC_LIMITS, _rfc3339)
            verdict = Verdict(VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, may_have_billed=billed, until=until)
        return verdict


# -- openai --------------------------------------------------------------------------------------

#: The ``code`` (or ``type``) of a 429 that is a used-up quota or spend limit rather than a rate limit.
#: ``insufficient_quota`` is KP-C2's; the other three are OpenAI's error-codes page's 429 rows
#: ("Credit balance exhausted", "Organization usage limit reached", "Project spend limit reached"),
#: whose ``type`` the page says can still be ``insufficient_quota``.
_OPENAI_EXHAUSTED_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "organization_usage_limit_exceeded",
        "project_spend_limit_exceeded",
    }
)

#: The ``code`` of a rate-limit error, the only signal a mid-stream ``APIError`` (which has no status)
#: gives that it was one.
_OPENAI_RATE_LIMIT_CODE = "rate_limit_exceeded"

#: The ``x-ratelimit-*`` limits a 429 may have used up, each ``(remaining, reset)``; the reset is a
#: duration such as ``1s`` or ``6m0s`` (OpenAI rate-limits guide, "Rate limits in headers").
_OPENAI_LIMITS = tuple(
    (f"x-ratelimit-remaining-{limit}", f"x-ratelimit-reset-{limit}")
    for limit in ("requests", "tokens", "project-tokens")
)

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _duration_reader(now: datetime) -> Callable[[str], datetime | None]:
    """a reader of OpenAI's reset durations (``1s``, ``6m0s``, ``20ms``), counted from ``now``."""

    def read(value: str) -> datetime | None:
        text = value.strip()
        parts = _DURATION_PART.findall(text)
        if not parts or "".join(number + unit for number, unit in parts) != text:
            return None
        return _after(now, sum(float(number) * _DURATION_UNIT_SECONDS[unit] for number, unit in parts))

    return read


class _OpenAIClassifier:
    """KP-C2's rules for the OpenAI API. Quota and rate limits belong to the organisation or project,
    so both are account-scoped; a key is revoked alone, and forbidden per model.

    A 402 is no OpenAI status, but this factory also serves OpenAI-compatible endpoints, some of which
    answer a used-up balance with one. Which of their keys share that balance is unknown, so it is
    held against the key alone: a sibling key that shares it costs one more refused call to learn.
    """

    def classify(self, exc: BaseException, *, now: datetime, stream_started: bool = False) -> Verdict:
        """the verdict on the attempt that raised ``exc``; see :class:`KeyErrorClassifier`."""
        now = _utc(now)
        if isinstance(exc, openai.APIConnectionError):
            billed = stream_started or isinstance(exc, openai.APITimeoutError)
            return Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=billed)
        if isinstance(exc, openai.APIStatusError):
            return self._status(exc, now=now, stream_started=stream_started)
        if isinstance(exc, openai.APIError):
            # an error event inside a stream: no status, and the stream had begun
            return self._mid_stream(exc)
        return _transport_verdict(exc, stream_started=stream_started) or _UNRECOGNISED

    def _status(self, exc: openai.APIStatusError, *, now: datetime, stream_started: bool) -> Verdict:
        status = exc.status_code
        billed = _billed(status=status, stream_started=stream_started)
        verdict = Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=billed)
        if status == 429 and (exc.code in _OPENAI_EXHAUSTED_CODES or exc.type in _OPENAI_EXHAUSTED_CODES):
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, may_have_billed=billed)
        elif status == 429:
            headers = exc.response.headers
            until = _retry_after(headers, now) or _latest_exhausted_reset(
                headers, _OPENAI_LIMITS, _duration_reader(now)
            )
            verdict = Verdict(VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, may_have_billed=billed, until=until)
        elif status == 402:
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.KEY, may_have_billed=billed)
        elif status == 401 or isinstance(exc, openai.AuthenticationError):
            verdict = Verdict(VerdictKind.REVOKED, VerdictScope.KEY, may_have_billed=billed)
        elif status == 403 or isinstance(exc, openai.PermissionDeniedError):
            verdict = Verdict(VerdictKind.FORBIDDEN, VerdictScope.KEY_AND_MODEL, may_have_billed=billed)
        return verdict

    def _mid_stream(self, exc: openai.APIError) -> Verdict:
        verdict = Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=True)
        if exc.code in _OPENAI_EXHAUSTED_CODES or exc.type in _OPENAI_EXHAUSTED_CODES:
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, may_have_billed=True)
        elif exc.code == _OPENAI_RATE_LIMIT_CODE:
            verdict = Verdict(VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, may_have_billed=True)
        return verdict


# -- openrouter ----------------------------------------------------------------------------------

#: The start of the ``ValueError`` ``langchain-openrouter`` raises for an error OpenRouter sends inside
#: a 200: ``OpenRouter API returned an error during streaming: <message> (code: <code>)`` mid-stream, or
#: ``OpenRouter API returned an error: <message> (code: <code>)`` in a non-streamed body. Its trailing
#: code is the only structure LangChain keeps.
_OPENROUTER_IN_BAND = "OpenRouter API returned an error"
_OPENROUTER_IN_BAND_CODE = re.compile(r"\(code: (\d+)\)\s*$")

#: ``error.metadata.limit_source`` on a 402 (OpenRouter limits page): the key's own credit limit, the
#: account's balance, or the account's balance net of requests still in flight.
_OPENROUTER_KEY_LIMIT = "openrouter_key_limit"
_OPENROUTER_CREDITS = "openrouter_credits"
_OPENROUTER_IN_FLIGHT = "openrouter_in_flight_budget"

#: ``error.metadata`` keys of a moderation refusal on a 403 (OpenRouter errors page,
#: ``ModerationErrorMetadata``). A flagged input is the request's fault, not the key's.
_OPENROUTER_MODERATION_KEYS = ("reasons", "flagged_input")


def _openrouter_metadata(body: str) -> dict[str, Any]:
    """``error.metadata`` of an OpenRouter error body, or an empty one when it has none."""
    try:
        parsed = json.loads(body)
    except ValueError:
        return {}
    error = parsed.get("error") if isinstance(parsed, dict) else None
    metadata = error.get("metadata") if isinstance(error, dict) else None
    return metadata if isinstance(metadata, dict) else {}


class _OpenRouterClassifier:
    """KP-C2's rules for OpenRouter. It limits rates per model and upstream route, so a 429 holds
    this key for this model only (``errors.is_rate_limit``). A 402 is scoped by the limit it names:
    a key's own credit limit is that key's, the account's balance is every key's.

    A 402 that names no limit is held against the key alone: OpenRouter's keys carry credit limits of
    their own, and holding a whole account on a key's limit would idle every sibling until the period
    rolls over, where the reverse mistake costs each sibling one refused call.
    """

    def classify(self, exc: BaseException, *, now: datetime, stream_started: bool = False) -> Verdict:
        """the verdict on the attempt that raised ``exc``; see :class:`KeyErrorClassifier`."""
        now = _utc(now)
        if isinstance(exc, openrouter.errors.OpenRouterError):
            return self._status(exc, now=now, stream_started=stream_started)
        if isinstance(exc, ValueError) and str(exc).startswith(_OPENROUTER_IN_BAND):
            return self._in_band(exc)
        return _transport_verdict(exc, stream_started=stream_started) or _UNRECOGNISED

    def _status(self, exc: openrouter.errors.OpenRouterError, *, now: datetime, stream_started: bool) -> Verdict:
        status = exc.status_code
        billed = _billed(status=status, stream_started=stream_started)
        metadata = _openrouter_metadata(exc.body)
        verdict = Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=billed)
        if status == 402:
            verdict = self._payment_required(metadata, _retry_after(exc.headers, now), billed=billed)
        elif status == 401:
            verdict = Verdict(VerdictKind.REVOKED, VerdictScope.KEY, may_have_billed=billed)
        elif status == 403 and not any(key in metadata for key in _OPENROUTER_MODERATION_KEYS):
            verdict = Verdict(VerdictKind.FORBIDDEN, VerdictScope.KEY_AND_MODEL, may_have_billed=billed)
        elif status == 429:
            until = _retry_after(exc.headers, now)
            verdict = Verdict(VerdictKind.RATE_LIMITED, VerdictScope.KEY_AND_MODEL, may_have_billed=billed, until=until)
        return verdict

    @staticmethod
    def _payment_required(metadata: Mapping[str, Any], until: datetime | None, *, billed: bool) -> Verdict:
        source = metadata.get("limit_source")
        if source == _OPENROUTER_IN_FLIGHT and until is not None:
            # the account's balance is held by requests still running: wait for them, as told
            return Verdict(VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, may_have_billed=billed, until=until)
        if source in (_OPENROUTER_CREDITS, _OPENROUTER_IN_FLIGHT):
            return Verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, may_have_billed=billed)
        return Verdict(VerdictKind.EXHAUSTED, VerdictScope.KEY, may_have_billed=billed)

    @staticmethod
    def _in_band(exc: ValueError) -> Verdict:
        """an error OpenRouter sent inside a 200, after it accepted the request.

        LangChain keeps only its message and code, so a 403 here cannot be told from a moderation
        refusal of the model's output; it is not held against the key, since a key held forbidden stays
        so until an operator resets it.
        """
        match = _OPENROUTER_IN_BAND_CODE.search(str(exc))
        code = int(match.group(1)) if match else None
        verdict = Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=True)
        if code == 429:
            verdict = Verdict(VerdictKind.RATE_LIMITED, VerdictScope.KEY_AND_MODEL, may_have_billed=True)
        elif code == 402:
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.KEY, may_have_billed=True)
        elif code == 401:
            verdict = Verdict(VerdictKind.REVOKED, VerdictScope.KEY, may_have_billed=True)
        return verdict


# -- lookup --------------------------------------------------------------------------------------

_CLASSIFIERS: dict[str, KeyErrorClassifier] = {
    ANTHROPIC_PROVIDER_NAME: _AnthropicClassifier(),
    OPENAI_PROVIDER_NAME: _OpenAIClassifier(),
    OPENROUTER_PROVIDER_NAME: _OpenRouterClassifier(),
}


def classifier_for(provider_name: str) -> KeyErrorClassifier:
    """The key-error classifier for a provider.

    :param provider_name: the provider's name, as a model's capabilities carry it (``"anthropic"``,
        ``"openai"``, ``"openrouter"``)
    :ptype provider_name: str
    :return: its classifier
    :rtype: KeyErrorClassifier
    :raises ValueError: when the provider has no classifier, so no pool can be built for it
    """
    try:
        return _CLASSIFIERS[provider_name]
    except KeyError:
        known = ", ".join(sorted(_CLASSIFIERS))
        raise ValueError(f"no key-error classifier for provider {provider_name!r}; known: {known}") from None
