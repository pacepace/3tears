"""What a failed model call says about the key it ran on, one classifier per provider.

The pool core holds no status codes, error types or header names (KP-C1). It hands the exception an
attempt raised to its provider's :class:`KeyErrorClassifier` and acts on the :class:`Verdict` it gets
back: move to another key, skip this key for the call, retry on this key, or let the failure
propagate unchanged.

**A rate limit is not a quota** (owner, 2026-11-26). Every provider answers both with a 429 (or a
402 or 403) and tells them apart in the body. A rate limit is retried on the same key -- by the
provider's SDK where it retries one, by the pool where it doesn't -- and never moves the key. Used-up
credit or spend, a revoked key and a forbidden one move to the next key. A rate limit whose wait runs
past the pool's threshold counts as exhaustion until the wait ends: a provider's daily cap is a
quota whatever its status says. Each classifier holds both halves of its provider's rule, so the
SDK hook that stops a retry (:meth:`KeyErrorClassifier.refuses_retry`) and the verdict on the error
that reaches the pool (:meth:`KeyErrorClassifier.classify`) never disagree.

An exception a classifier does not recognise is :attr:`VerdictKind.NOT_KEY_RELATED`. An unknown error
never burns a key; the worst it can do is fail the call the way it would have without a pool.

What each classifier reads, and where each rule's shape comes from:

- **anthropic** -- the SDK's ``APIStatusError`` (LangChain re-raises it as a subclass, so ``response``,
  ``body`` and ``type`` survive). The body is ``{"type": "error", "error": {"type": ..., "message":
  ..., "details": {"error_code": ...}}}`` (``anthropic.types.shared.ErrorResponse``; the
  ``details.error_code`` field is from the Claude API errors page). A 429 is exhaustion only when its
  ``error_code`` names a spend limit; any other 429 ``rate_limit_error`` (``ramp_limit_exceeded``
  included) is a rate limit. An error event inside a stream arrives as the same class carrying the
  stream's 200 status, so the error type decides, not the status. ``retry-after`` (and the SDK's own
  ``retry-after-ms``) first; then the ``anthropic-ratelimit-*-reset`` headers (RFC 3339) of a limit
  whose ``-remaining`` is 0.
- **openai** -- the SDK's ``APIStatusError``, whose ``body`` is the inner error object and whose
  ``code`` and ``type`` the SDK lifts from it; and its bare ``APIError`` for an error inside a stream,
  which carries no status. Quota and rate limit are both 429s, told apart by ``code`` (or ``type``).
  ``retry-after`` / ``retry-after-ms`` first; then the ``x-ratelimit-reset-*`` headers (a duration such
  as ``6m0s``) of a limit whose ``remaining`` is 0.
- **openrouter** -- the ``openrouter`` SDK's ``OpenRouterError`` (``status_code``, ``headers``, the
  raw ``body``), raw ``httpx`` transport errors (the SDK re-raises them unwrapped), and the
  ``ValueError`` ``langchain-openrouter`` raises for an error the API sends inside a 200, ending
  ``(code: N)``. The body's ``error.metadata`` names the limit (``limit_source``, ``reason``,
  ``scope``, ``resets_at``) and the error's canonical ``error_type`` (OpenRouter errors and limits
  pages). ``retry-after`` first; then ``X-RateLimit-Reset``, milliseconds since the Unix epoch.
  Its SDK never retries a 429 (``chat.send`` retries only ``5XX``), so the pool retries its rate
  limits itself (:attr:`KeyErrorClassifier.sdk_retries_rate_limits`).

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

    #: the key's (or its account's) credit, spend limit or quota is used up, until :attr:`Verdict.until`
    #: when the provider said, else the billing period's end; the pool moves to the next key
    EXHAUSTED = "exhausted"
    #: the provider no longer accepts the key; the pool moves to the next key
    REVOKED = "revoked"
    #: the key may not use this model or resource; the pool moves to the next key
    FORBIDDEN = "forbidden"
    #: the provider rate-limited the call. It is retried on the same key, as the provider said
    #: (:attr:`Verdict.until`), and is never a state of the key
    RATE_LIMITED = "rate_limited"
    #: this one request asks for more than the key may spend at all; the call skips the key, and
    #: nothing about the key changes
    OVERSIZED = "oversized"
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


#: The kinds that carry an ``until``: when a rate limit lifts, or when an exhaustion resets.
_TIMED = frozenset({VerdictKind.RATE_LIMITED, VerdictKind.EXHAUSTED})

#: The kinds the pool records on the key, moving the active key off it (KP-H1).
_MOVING = frozenset({VerdictKind.EXHAUSTED, VerdictKind.REVOKED, VerdictKind.FORBIDDEN})


@dataclass(frozen=True, slots=True)
class Verdict:
    """A classifier's reading of one failed attempt.

    :ivar kind: what the failure says about the key
    :ivar scope: which keys it applies to; :attr:`VerdictScope.KEY` for a failure that is not the key's,
        where it means nothing
    :ivar may_have_billed: whether the provider may have charged for the attempt, so its reservation is
        settled in full rather than released (KP-E4)
    :ivar until: for :attr:`VerdictKind.RATE_LIMITED`, when the provider said to retry; for
        :attr:`VerdictKind.EXHAUSTED`, when the provider said the limit resets. In UTC; ``None`` when
        the provider did not say, and always ``None`` for every other kind
    """

    kind: VerdictKind
    scope: VerdictScope
    may_have_billed: bool
    until: datetime | None = None

    def __post_init__(self) -> None:
        """refuse an ``until`` on a kind that has none, or one without a timezone.

        :raises ValueError: when ``until`` is set on a kind other than rate-limited or exhausted, or is
            naive
        """
        if self.until is None:
            return
        if self.kind not in _TIMED:
            raise ValueError(f"a {self.kind} verdict carries no until, got {self.until!r}")
        if self.until.tzinfo is None:
            raise ValueError(f"until must be timezone-aware, got {self.until!r}")

    @property
    def moves_key(self) -> bool:
        """Whether the pool records the failure on the key and moves the active key off it.

        :return: ``True`` for exhausted, revoked and forbidden
        :rtype: bool
        """
        return self.kind in _MOVING

    @property
    def tries_another_key(self) -> bool:
        """Whether the call runs again on another key: one that moves the key, or an oversized request.

        :return: ``True`` when :attr:`moves_key` is, and for :attr:`VerdictKind.OVERSIZED`
        :rtype: bool
        """
        return self.moves_key or self.kind is VerdictKind.OVERSIZED


class KeyErrorClassifier(Protocol):
    """Reads what one provider's failed attempt says about the key it ran on."""

    @property
    def sdk_retries_rate_limits(self) -> bool:
        """Whether the provider's SDK retries a rate limit itself, honouring ``retry-after``.

        Where it doesn't, the pool retries a rate-limited call on the same key.

        :return: ``True`` for an SDK that retries a 429
        :rtype: bool
        """
        ...

    def classify(
        self,
        exc: BaseException,
        *,
        now: datetime,
        stream_started: bool = False,
        exhaustion_wait: timedelta | None = None,
    ) -> Verdict:
        """the verdict on the attempt that raised ``exc``.

        :param exc: what the attempt raised
        :ptype exc: BaseException
        :param now: the current time, timezone-aware, that a ``retry-after`` in seconds counts from
        :ptype now: datetime
        :param stream_started: whether the attempt had begun receiving the provider's answer before it
            failed; an attempt that had may have been billed whatever it raised
        :ptype stream_started: bool
        :param exhaustion_wait: a rate limit whose wait ends more than this after ``now`` is exhausted
            until the wait ends; ``None`` reads every rate limit as one
        :ptype exhaustion_wait: timedelta | None
        :return: the verdict
        :rtype: Verdict
        :raises ValueError: when ``now`` is naive
        """
        ...

    def refuses_retry(
        self,
        status: int,
        headers: Mapping[str, str],
        body: bytes | None,
        *,
        now: datetime,
        exhaustion_wait: timedelta,
    ) -> bool:
        """Whether the SDK must raise a response at once instead of retrying it on the same key.

        True for a refusal that moves the key -- revoked, out of credit, forbidden, a quota a 429
        reports -- and for a rate limit whose wait runs past ``exhaustion_wait``. A rate limit the
        provider asks a short wait for, and any other status, is left to the SDK's retries.

        :param status: the response's HTTP status
        :ptype status: int
        :param headers: the response's headers (case-insensitive, as ``httpx`` keeps them)
        :ptype headers: Mapping[str, str]
        :param body: the response's body; read only for a 429, ``None`` for any other status. A body
            that is not the provider's JSON leaves the 429 to the SDK
        :ptype body: bytes | None
        :param now: the current time, timezone-aware, that a wait counts from
        :ptype now: datetime
        :param exhaustion_wait: the longest wait that is still a rate limit
        :ptype exhaustion_wait: timedelta
        :return: ``True`` to raise the response at once
        :rtype: bool
        :raises ValueError: when ``now`` is naive
        """
        ...


# -- shared reading ------------------------------------------------------------------------------

#: The HTTP statuses a provider refuses a request with before doing any work for it: a 4xx.
_CLIENT_ERROR = range(400, 500)

#: The statuses that are the key's whatever their body: revoked, out of credit, forbidden.
_KEY_REFUSALS = frozenset({401, 402, 403})

#: The status every provider sends both a rate limit and some quotas with.
_TOO_MANY_REQUESTS = 429

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
    except OverflowError:  # NOSILENT: a wait past the calendar's end is no time; the rate limit has no until
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


def _rfc3339(value: str) -> datetime | None:
    """an RFC 3339 time in UTC, or ``None`` when ``value`` is not one."""
    try:
        return _as_utc(datetime.fromisoformat(value.strip()))
    except ValueError:  # NOSILENT: an unreadable reset is an absent one; the limit has no until
        return None


def _waits_past(until: datetime | None, now: datetime, wait: timedelta) -> bool:
    """whether a wait that ends at ``until`` runs more than ``wait`` past ``now``."""
    return until is not None and until - now > wait


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


def _parsed(body: bytes | None) -> object:
    """a response body's JSON, or ``None`` when there is none or it is not JSON."""
    if body is None:
        return None
    try:
        return json.loads(body)
    except ValueError:  # NOSILENT: a body that is not JSON names no quota, so the 429 stays a rate limit
        return None


def _object(value: object, key: str) -> dict[str, Any]:
    """``value[key]`` when both are JSON objects, else an empty one."""
    inner = value.get(key) if isinstance(value, dict) else None
    return inner if isinstance(inner, dict) else {}


#: The verdict on an exception no rule recognises: not the key's, and possibly billed, since nothing
#: says it failed before the provider did any work.
_UNRECOGNISED = Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=True)


def _long_wait_exhausts(verdict: Verdict, now: datetime, exhaustion_wait: timedelta | None) -> Verdict:
    """``verdict``, with a rate limit whose wait runs past ``exhaustion_wait`` read as exhaustion until it ends.

    A wait this long is a cap, not a rate: the key is out until it ends. It is held on the whole key,
    since a key's state is not kept per model.

    :param verdict: a provider's verdict
    :ptype verdict: Verdict
    :param now: the time the wait counts from, in UTC
    :ptype now: datetime
    :param exhaustion_wait: the longest wait that is still a rate limit; ``None`` reads every one as one
    :ptype exhaustion_wait: timedelta | None
    :return: the verdict to act on
    :rtype: Verdict
    """
    if (
        exhaustion_wait is None
        or verdict.kind is not VerdictKind.RATE_LIMITED
        or not _waits_past(verdict.until, now, exhaustion_wait)
    ):
        return verdict
    scope = VerdictScope.KEY if verdict.scope is VerdictScope.KEY_AND_MODEL else verdict.scope
    return Verdict(VerdictKind.EXHAUSTED, scope, may_have_billed=verdict.may_have_billed, until=verdict.until)


def _refuses_retry(
    status: int,
    headers: Mapping[str, str],
    body: bytes | None,
    *,
    now: datetime,
    exhaustion_wait: timedelta,
    reports_quota: Callable[[object], bool],
    come_back: Callable[[Mapping[str, str], datetime], datetime | None],
) -> bool:
    """whether the SDK must raise a response at once (:meth:`KeyErrorClassifier.refuses_retry`), by one
    provider's reading of a 429.

    :param status: the response's HTTP status
    :ptype status: int
    :param headers: the response's headers
    :ptype headers: Mapping[str, str]
    :param body: the response's body for a 429, ``None`` otherwise
    :ptype body: bytes | None
    :param now: the current time, timezone-aware
    :ptype now: datetime
    :param exhaustion_wait: the longest wait that is still a rate limit
    :ptype exhaustion_wait: timedelta
    :param reports_quota: whether a 429's parsed JSON body reports a used-up quota, not a rate limit
    :ptype reports_quota: Callable[[object], bool]
    :param come_back: when a rate-limited response says the key may serve again, ``None`` if it doesn't say
    :ptype come_back: Callable[[Mapping[str, str], datetime], datetime | None]
    :return: ``True`` to raise the response at once
    :rtype: bool
    """
    now = _utc(now)
    if status in _KEY_REFUSALS:
        return True
    if status != _TOO_MANY_REQUESTS:
        return False
    if reports_quota(_parsed(body)):
        return True
    return _waits_past(come_back(headers, now), now, exhaustion_wait)


# -- anthropic -----------------------------------------------------------------------------------

#: ``error.details.error_code`` values on a Messages API 429 that report a spend limit used up, not
#: a rate (Claude API errors page, "Error codes"): the organisation's monthly spend cap, and a
#: contract's. Every other 429 code (``ramp_limit_exceeded`` among them) is a rate limit.
_ANTHROPIC_EXHAUSTED_CODES = frozenset({"enforced_spend_limit_reached", "contract_spend_limit_reached"})

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


def _anthropic_error(body: object) -> dict[str, Any]:
    """the ``error`` object of an Anthropic error body, or an empty one when the body has none."""
    return _object(body, "error")


def _anthropic_spend_code(error: Mapping[str, Any]) -> bool:
    """whether an Anthropic ``error`` object's ``details.error_code`` reports a spend limit used up."""
    return _object(error, "details").get("error_code") in _ANTHROPIC_EXHAUSTED_CODES


def _anthropic_reports_quota(body: object) -> bool:
    """whether an Anthropic 429's body reports a spend limit used up, not a rate limit."""
    return _anthropic_spend_code(_anthropic_error(body))


def _anthropic_come_back(headers: Mapping[str, str], now: datetime) -> datetime | None:
    """when an Anthropic rate limit says to come back: ``retry-after``, else its used-up limits' reset."""
    return _retry_after(headers, now) or _latest_exhausted_reset(headers, _ANTHROPIC_LIMITS, _rfc3339)


class _AnthropicClassifier:
    """KP-C2's rules for the Anthropic API. Credit, spend and rate limits belong to the organisation (or
    a workspace in it), so all are account-scoped; a key is revoked alone, and forbidden per model."""

    #: the SDK retries a 429, honouring ``retry-after``
    sdk_retries_rate_limits = True

    def classify(
        self,
        exc: BaseException,
        *,
        now: datetime,
        stream_started: bool = False,
        exhaustion_wait: timedelta | None = None,
    ) -> Verdict:
        """the verdict on the attempt that raised ``exc``; see :class:`KeyErrorClassifier`."""
        now = _utc(now)
        return _long_wait_exhausts(self._verdict(exc, now=now, stream_started=stream_started), now, exhaustion_wait)

    def refuses_retry(
        self, status: int, headers: Mapping[str, str], body: bytes | None, *, now: datetime, exhaustion_wait: timedelta
    ) -> bool:
        """whether the SDK must raise a response at once; see :class:`KeyErrorClassifier`."""
        return _refuses_retry(
            status,
            headers,
            body,
            now=now,
            exhaustion_wait=exhaustion_wait,
            reports_quota=_anthropic_reports_quota,
            come_back=_anthropic_come_back,
        )

    @staticmethod
    def _verdict(exc: BaseException, *, now: datetime, stream_started: bool) -> Verdict:
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
        message = str(error.get("message") or "").strip().lower()

        verdict = Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=billed)
        if _anthropic_spend_code(error) or error_type == "billing_error" or status == 402:
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, may_have_billed=billed)
        elif (error_type == "invalid_request_error" or status == 400) and (
            message.startswith(_ANTHROPIC_SPEND_LIMIT_PREFIXES) or _ANTHROPIC_CREDIT_TOO_LOW in message
        ):
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, may_have_billed=billed)
        elif error_type == "authentication_error" or status == 401:
            verdict = Verdict(VerdictKind.REVOKED, VerdictScope.KEY, may_have_billed=billed)
        elif error_type == "permission_error" or status == 403:
            verdict = Verdict(VerdictKind.FORBIDDEN, VerdictScope.KEY_AND_MODEL, may_have_billed=billed)
        elif error_type == "rate_limit_error" or status == _TOO_MANY_REQUESTS:
            until = _anthropic_come_back(exc.response.headers, now)
            verdict = Verdict(VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, may_have_billed=billed, until=until)
        return verdict


# -- openai --------------------------------------------------------------------------------------

#: The ``code`` (or ``type``) of a 429 that is a used-up quota, credit or spend limit rather than a rate
#: limit (OpenAI error-codes page, its 429 rows; the page says ``type`` can still be
#: ``insufficient_quota`` where ``code`` names the case).
_OPENAI_EXHAUSTED_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
        "billing_hard_limit_reached",
    }
)

#: The ``code`` of a rate-limit error, and its ``type``: the only signal a mid-stream ``APIError``
#: (which has no status) gives that it was one.
_OPENAI_RATE_LIMIT_CODES = frozenset({"rate_limit_exceeded", "slow_down"})
_OPENAI_RATE_LIMIT_TYPE = "rate_limit_error"

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


def _openai_quota(code: object, error_type: object) -> bool:
    """whether an OpenAI error's ``code`` or ``type`` reports a used-up quota."""
    return code in _OPENAI_EXHAUSTED_CODES or error_type in _OPENAI_EXHAUSTED_CODES


def _openai_reports_quota(body: object) -> bool:
    """whether an OpenAI 429's wire body -- which wraps the error object the SDK lifts ``code`` and
    ``type`` from -- reports a used-up quota."""
    error = _object(body, "error")
    return _openai_quota(error.get("code"), error.get("type"))


def _openai_come_back(headers: Mapping[str, str], now: datetime) -> datetime | None:
    """when an OpenAI rate limit says to come back: ``retry-after``, else its used-up limits' reset."""
    return _retry_after(headers, now) or _latest_exhausted_reset(headers, _OPENAI_LIMITS, _duration_reader(now))


class _OpenAIClassifier:
    """KP-C2's rules for the OpenAI API. Quota and rate limits belong to the organisation or project,
    so both are account-scoped; a key is revoked alone, and forbidden per model.

    A 402 is no OpenAI status, but this factory also serves OpenAI-compatible endpoints, some of which
    answer a used-up balance with one. Which of their keys share that balance is unknown, so it is
    held against the key alone: a sibling key that shares it costs one more refused call to learn.
    """

    #: the SDK retries a 429, honouring ``retry-after``
    sdk_retries_rate_limits = True

    def classify(
        self,
        exc: BaseException,
        *,
        now: datetime,
        stream_started: bool = False,
        exhaustion_wait: timedelta | None = None,
    ) -> Verdict:
        """the verdict on the attempt that raised ``exc``; see :class:`KeyErrorClassifier`."""
        now = _utc(now)
        return _long_wait_exhausts(self._verdict(exc, now=now, stream_started=stream_started), now, exhaustion_wait)

    def refuses_retry(
        self, status: int, headers: Mapping[str, str], body: bytes | None, *, now: datetime, exhaustion_wait: timedelta
    ) -> bool:
        """whether the SDK must raise a response at once; see :class:`KeyErrorClassifier`."""
        return _refuses_retry(
            status,
            headers,
            body,
            now=now,
            exhaustion_wait=exhaustion_wait,
            reports_quota=_openai_reports_quota,
            come_back=_openai_come_back,
        )

    def _verdict(self, exc: BaseException, *, now: datetime, stream_started: bool) -> Verdict:
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
        if status == _TOO_MANY_REQUESTS and _openai_quota(exc.code, exc.type):
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, may_have_billed=billed)
        elif status == _TOO_MANY_REQUESTS:
            until = _openai_come_back(exc.response.headers, now)
            verdict = Verdict(VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, may_have_billed=billed, until=until)
        elif status == 402:
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.KEY, may_have_billed=billed)
        elif status == 401 or isinstance(exc, openai.AuthenticationError):
            verdict = Verdict(VerdictKind.REVOKED, VerdictScope.KEY, may_have_billed=billed)
        elif status == 403 or isinstance(exc, openai.PermissionDeniedError):
            verdict = Verdict(VerdictKind.FORBIDDEN, VerdictScope.KEY_AND_MODEL, may_have_billed=billed)
        return verdict

    @staticmethod
    def _mid_stream(exc: openai.APIError) -> Verdict:
        verdict = Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=True)
        if _openai_quota(exc.code, exc.type):
            verdict = Verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, may_have_billed=True)
        elif exc.code in _OPENAI_RATE_LIMIT_CODES or exc.type == _OPENAI_RATE_LIMIT_TYPE:
            verdict = Verdict(VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, may_have_billed=True)
        return verdict


# -- openrouter ----------------------------------------------------------------------------------

#: The start of the ``ValueError`` ``langchain-openrouter`` raises for an error OpenRouter sends inside
#: a 200: ``OpenRouter API returned an error during streaming: <message> (code: <code>)`` mid-stream, or
#: ``OpenRouter API returned an error: <message> (code: <code>)`` in a non-streamed body. Its trailing
#: code is the only structure LangChain keeps.
_OPENROUTER_IN_BAND = "OpenRouter API returned an error"
_OPENROUTER_IN_BAND_CODE = re.compile(r"\(code: (\d+)\)\s*$")

#: ``error.metadata.limit_source`` values (OpenRouter limits page): a key's own credit limit, the
#: account's balance, its balance net of requests still in flight, and a guardrail's or workspace's
#: budget window.
_OPENROUTER_KEY_LIMIT = "openrouter_key_limit"
_OPENROUTER_CREDITS = "openrouter_credits"
_OPENROUTER_IN_FLIGHT = "openrouter_in_flight_budget"
_OPENROUTER_BUDGETS = frozenset({"openrouter_guardrail_budget", "openrouter_workspace_budget", _OPENROUTER_KEY_LIMIT})

#: ``error.metadata.reason`` on a 402 ``openrouter_credits`` whose one request costs more than the
#: whole budget (limits page): no retry helps, and the key can still serve smaller calls.
_OPENROUTER_WEIGHT_EXCEEDS_BUDGET = "weight_exceeds_budget"

#: ``error.metadata.scope`` values of a 403 budget that is the key's own (limits page); every other
#: scope (``member``, ``workspace``, a workspace default) is the account's.
_OPENROUTER_KEY_SCOPES = frozenset({"key_limit", "api_key"})

#: Canonical ``error_type`` values of a 403 that is the request's or the model's doing, not the key's
#: (errors page): a content filter's flag, and a provider's refusal reported as an error.
_OPENROUTER_NOT_THE_KEYS = frozenset({"content_policy_violation", "refusal"})

#: ``error.metadata`` keys of a moderation refusal on a 403 (OpenRouter errors page,
#: ``ModerationErrorMetadata``). A flagged input is the request's fault, not the key's.
_OPENROUTER_MODERATION_KEYS = ("reasons", "flagged_input")

#: The reset of the limit an OpenRouter 429 hit, in milliseconds since the Unix epoch (limits page).
_OPENROUTER_RATE_LIMIT_RESET = "x-ratelimit-reset"


def _epoch_millis(value: str) -> datetime | None:
    """a time given in milliseconds since the Unix epoch, in UTC; ``None`` when ``value`` is not one."""
    try:
        millis = float(value.strip())
    except ValueError:  # NOSILENT: an unreadable reset is an absent one; the rate limit has no until
        return None
    if not math.isfinite(millis):
        return None
    try:
        return datetime.fromtimestamp(millis / 1000, tz=UTC)
    except OverflowError, OSError, ValueError:  # NOSILENT: a reset off the calendar is no time
        return None


def _openrouter_error(body: str) -> dict[str, Any]:
    """``error`` of an OpenRouter error body, or an empty one when it has none."""
    return _object(_parsed(body.encode()), "error")


def _openrouter_reports_quota(body: object) -> bool:
    """never: OpenRouter reports a used-up quota with a 402 or a 403, not a 429."""
    del body
    return False


def _openrouter_come_back(headers: Mapping[str, str], now: datetime) -> datetime | None:
    """when an OpenRouter rate limit says to come back: ``retry-after``, else ``X-RateLimit-Reset``."""
    reset = headers.get(_OPENROUTER_RATE_LIMIT_RESET)
    return _retry_after(headers, now) or (_epoch_millis(reset) if reset is not None else None)


class _OpenRouterClassifier:
    """KP-C2's rules for OpenRouter. It limits rates per model and upstream route, so a 429 is this
    key's for this model only (``errors.is_rate_limit``). A 402 or a 403 budget is scoped by the limit
    it names: a key's own limit is that key's, the account's balance or a workspace budget is every
    key's.

    A 402 that names no limit is held against the key alone: OpenRouter's keys carry credit limits of
    their own, and holding a whole account on a key's limit would idle every sibling until the period
    rolls over, where the reverse mistake costs each sibling one refused call.
    """

    #: ``chat.send`` retries only ``5XX`` statuses (``openrouter/chat.py``), so the pool retries a 429
    sdk_retries_rate_limits = False

    def classify(
        self,
        exc: BaseException,
        *,
        now: datetime,
        stream_started: bool = False,
        exhaustion_wait: timedelta | None = None,
    ) -> Verdict:
        """the verdict on the attempt that raised ``exc``; see :class:`KeyErrorClassifier`."""
        now = _utc(now)
        return _long_wait_exhausts(self._verdict(exc, now=now, stream_started=stream_started), now, exhaustion_wait)

    def refuses_retry(
        self, status: int, headers: Mapping[str, str], body: bytes | None, *, now: datetime, exhaustion_wait: timedelta
    ) -> bool:
        """whether the SDK must raise a response at once; see :class:`KeyErrorClassifier`."""
        return _refuses_retry(
            status,
            headers,
            body,
            now=now,
            exhaustion_wait=exhaustion_wait,
            reports_quota=_openrouter_reports_quota,
            come_back=_openrouter_come_back,
        )

    def _verdict(self, exc: BaseException, *, now: datetime, stream_started: bool) -> Verdict:
        if isinstance(exc, openrouter.errors.OpenRouterError):
            return self._status(exc, now=now, stream_started=stream_started)
        if isinstance(exc, ValueError) and str(exc).startswith(_OPENROUTER_IN_BAND):
            return self._in_band(exc)
        return _transport_verdict(exc, stream_started=stream_started) or _UNRECOGNISED

    def _status(self, exc: openrouter.errors.OpenRouterError, *, now: datetime, stream_started: bool) -> Verdict:
        status = exc.status_code
        billed = _billed(status=status, stream_started=stream_started)
        error = _openrouter_error(exc.body)
        metadata = _object(error, "metadata")
        verdict = Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=billed)
        if status == 402:
            verdict = self._payment_required(metadata, _retry_after(exc.headers, now), billed=billed)
        elif status == 401:
            verdict = Verdict(VerdictKind.REVOKED, VerdictScope.KEY, may_have_billed=billed)
        elif status == 403:
            verdict = self._forbidden(error, metadata, billed=billed)
        elif status == _TOO_MANY_REQUESTS:
            until = _openrouter_come_back(exc.headers, now)
            verdict = Verdict(VerdictKind.RATE_LIMITED, VerdictScope.KEY_AND_MODEL, may_have_billed=billed, until=until)
        return verdict

    @staticmethod
    def _payment_required(metadata: Mapping[str, Any], until: datetime | None, *, billed: bool) -> Verdict:
        source = metadata.get("limit_source")
        if source == _OPENROUTER_IN_FLIGHT and until is not None:
            # the account's balance is held by requests still running: a rate limit, retried as told
            return Verdict(VerdictKind.RATE_LIMITED, VerdictScope.ACCOUNT, may_have_billed=billed, until=until)
        if source == _OPENROUTER_CREDITS and metadata.get("reason") == _OPENROUTER_WEIGHT_EXCEEDS_BUDGET:
            return Verdict(VerdictKind.OVERSIZED, VerdictScope.KEY, may_have_billed=billed)
        if source in (_OPENROUTER_CREDITS, _OPENROUTER_IN_FLIGHT):
            return Verdict(VerdictKind.EXHAUSTED, VerdictScope.ACCOUNT, may_have_billed=billed)
        return Verdict(VerdictKind.EXHAUSTED, VerdictScope.KEY, may_have_billed=billed)

    @staticmethod
    def _forbidden(error: Mapping[str, Any], metadata: Mapping[str, Any], *, billed: bool) -> Verdict:
        # ``error_type`` sits in ``metadata`` on Chat Completions and beside ``type`` on the Messages skin
        error_type = metadata.get("error_type", error.get("error_type"))
        if error_type in _OPENROUTER_NOT_THE_KEYS or any(key in metadata for key in _OPENROUTER_MODERATION_KEYS):
            return Verdict(VerdictKind.NOT_KEY_RELATED, VerdictScope.KEY, may_have_billed=billed)
        if metadata.get("limit_source") in _OPENROUTER_BUDGETS:
            scope = VerdictScope.KEY if metadata.get("scope") in _OPENROUTER_KEY_SCOPES else VerdictScope.ACCOUNT
            resets = metadata.get("resets_at")
            until = _rfc3339(resets) if isinstance(resets, str) else None
            return Verdict(VerdictKind.EXHAUSTED, scope, may_have_billed=billed, until=until)
        return Verdict(VerdictKind.FORBIDDEN, VerdictScope.KEY_AND_MODEL, may_have_billed=billed)

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
        if code == _TOO_MANY_REQUESTS:
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
