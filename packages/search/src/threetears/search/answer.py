"""The answer call layer -- one question, one provider, one grounded answer.

The layer between a caller's question and an answers adapter, mirroring
:func:`threetears.search.call.search` (``answers-requirements-design.md``
§10.3) and sharing its machinery rather than restating it: the budget refusal,
the pacing wait and the debit are :mod:`threetears.search.call`'s own, so a
rule about any of them lands for both kinds of call or neither. What it owns:

- **budget consultation and pacing, in Call's order** (AR-D1, D4, D5, D8):
  ``check`` then ``acquire`` then the provider call then ``record``, below the
  transport's retry boundary. The estimate ``check`` is handed is the
  provider's own (AR-C3) -- an answers provider holds its deployment's rates,
  which no capability declaration carries -- and ``record`` receives exactly
  the spend the caller is told, on success and on every failure (SR-E2,
  SR-E3);
- **dispositions completed from the declaration** (SR-B2): every criterion
  the caller sent has one answer at this boundary, however thorough the
  adapter was;
- **failure mapping** (AD9, SR-J1): an untyped escape from an adapter
  becomes :class:`~threetears.search.contracts.errors.TransportFailed`, as in
  Call;
- **one span per answer** (AR-J1, SR-I1), carrying the spend, the instance
  and egress, the source count, whether anchors arrived, and the token
  counts as attributes -- zero-cost without OpenTelemetry.

What it does not own, deliberately: a result cap (an answer has no result
count to bound), accumulation, and any retry of its own. A stream is never
retried once it has begun, and that rule is the transport's (AR-E1).

**The default bound is longer than a search's.** Brave states a 30-second
floor for one answer (design §4.2), so a caller naming no bound gets
:data:`DEFAULT_ANSWER_TIMEOUT_SECONDS`, not Call's twenty.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

from threetears.observe import get_logger, set_span_attribute, traced
from threetears.search.call import _completed_dispositions, _pace, _record, _refusal, _stamp
from threetears.search.contracts import (
    EGRESS_DIRECT,
    AnswerProvider,
    AnswerRequest,
    BudgetPort,
    GroundedAnswer,
    RateLimiterPort,
    SearchFailure,
    Spend,
    TransportFailed,
)

__all__ = ["answer"]

_logger = get_logger(__name__)

#: bound applied when the caller supplies no deadline: Brave's stated floor
#: for one answer with room for a slow one. A default, never a constant --
#: a caller under its own deadline passes what remains of it (SR-G1, SR-G2).
DEFAULT_ANSWER_TIMEOUT_SECONDS: Final[float] = 45.0


def _span_attributes(result: GroundedAnswer, *, provider: AnswerProvider, egress: str) -> None:
    """Attach one answer's facts to the current span (AR-J1).

    Token counts are attributes, never metric labels: they vary per call, and
    a label with unbounded values is a cardinality bill.

    :param result: the completed answer
    :ptype result: GroundedAnswer
    :param provider: the provider that answered
    :ptype provider: AnswerProvider
    :param egress: the exit the call left by
    :ptype egress: str
    """
    _span_spend(result.spend, provider=provider, egress=egress)
    set_span_attribute("answer.sources", len(result.sources))
    set_span_attribute("answer.anchored", bool(result.anchors))
    if result.prompt_tokens is not None:
        set_span_attribute("answer.prompt_tokens", result.prompt_tokens)
    if result.completion_tokens is not None:
        set_span_attribute("answer.completion_tokens", result.completion_tokens)


def _span_spend(spend: Spend, *, provider: AnswerProvider, egress: str) -> None:
    """Attach what one answer call consumed, and where it went, to the current span (AR-J1).

    Set on every outcome: a failed answer -- a refusal, a broken or timed-out
    stream -- cost something too, and a span without it hides the bill.

    :param spend: what the call consumed
    :ptype spend: Spend
    :param provider: the provider asked
    :ptype provider: AnswerProvider
    :param egress: the exit the call left by
    :ptype egress: str
    """
    set_span_attribute("answer.provider_instance", provider.provider_instance)
    set_span_attribute("answer.egress", egress)
    set_span_attribute("answer.spend.money", str(spend.money))
    set_span_attribute("answer.spend.calls", spend.calls)
    set_span_attribute("answer.spend.provider_units", str(spend.provider_units))
    set_span_attribute("answer.spend.wall_clock_seconds", spend.wall_clock_seconds)


@traced(name="threetears.search.answer")
async def answer(
    request: AnswerRequest,
    *,
    provider: AnswerProvider,
    timeout_seconds: float | None = None,
    budget: BudgetPort | None = None,
    limiter: RateLimiterPort | None = None,
    egress: str = EGRESS_DIRECT,
    clock: Callable[[], float] = time.monotonic,
) -> GroundedAnswer:
    """Turn one question into one grounded answer through one provider.

    :param request: what the caller asked
    :ptype request: AnswerRequest
    :param provider: the answers adapter to ask
    :ptype provider: AnswerProvider
    :param timeout_seconds: bound for the whole call, pacing included; None
        applies :data:`DEFAULT_ANSWER_TIMEOUT_SECONDS` (SR-G1, SR-G2)
    :ptype timeout_seconds: float | None
    :param budget: the local refusal authority, ``check`` before the call and
        ``record`` after (D4, D5). None makes no consultation at all
    :ptype budget: BudgetPort | None
    :param limiter: the pacing seam for ``(provider instance, egress)`` (D8,
        D20, AR-D1). Brave's two-per-second default is this limiter's
        configuration, never the adapter's constant. None leaves the call
        unpaced by this layer
    :ptype limiter: RateLimiterPort | None
    :param egress: which exit the provider's transport leaves by (D20)
    :ptype egress: str
    :param clock: the monotonic clock every wall-clock figure and the pacing
        wait are measured on
    :ptype clock: Callable[[], float]
    :return: the answer, one disposition per criterion, and the spend the
        call consumed. Zero sources is a success that says so (AR-A3)
    :rtype: GroundedAnswer
    :raises threetears.search.contracts.errors.SearchFailure: one of the
        typed classes, always carrying spend (SR-E3). Callers that must not
        see an exception go through :func:`threetears.search.bind_answer.bind_answer`
        (D10)
    """
    started = clock()
    scope_tags = request.budget_scope_tags
    remaining_seconds = DEFAULT_ANSWER_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds

    try:
        if budget is not None:
            # Before the provider call, and before pacing: a refused call needs
            # no pacing slot (Call's order, D4).
            decision = await budget.check(provider.estimate(request), scope_tags=scope_tags)
            if not decision.allowed:
                raise _refusal(decision, provider=provider, egress=egress)

        if limiter is not None:
            remaining_seconds = await _pace(
                limiter,
                provider=provider,
                egress=egress,
                budget_seconds=remaining_seconds,
                started=started,
                clock=clock,
            )

        try:
            result = await provider.answer(request, timeout_seconds=remaining_seconds)
        except SearchFailure as failure:
            _stamp(failure, elapsed=clock() - started)
            # the failure's spend is as real as a success's: a stream that broke
            # after the provider began answering was billed (AR-E1, SR-E3).
            await _record(budget, failure.spend, scope_tags=scope_tags)
            raise
        except Exception as exc:
            # An adapter letting an untyped exception out is the adapter's defect;
            # logged as one and re-raised typed, with only the wall-clock this
            # layer measured itself as its spend (Call's rule, SR-J1).
            _logger.exception(
                "answers provider %s raised an untyped exception; the adapter should map failures onto the taxonomy",
                provider.provider_instance,
            )
            failed_spend = Spend(wall_clock_seconds=clock() - started)
            await _record(budget, failed_spend, scope_tags=scope_tags)
            raise TransportFailed(
                f"answers provider {provider.provider_instance} failed with an unmapped {type(exc).__name__}: {exc}",
                spend=failed_spend,
                provider_instance=provider.provider_instance,
            ) from exc

        completed = result.model_copy(
            update={
                "dispositions": _completed_dispositions(request, result, provider),
                "spend": result.spend.model_copy(update={"wall_clock_seconds": clock() - started}),
            }
        )
        await _record(budget, completed.spend, scope_tags=scope_tags)
        _span_attributes(completed, provider=provider, egress=egress)
        return completed
    except SearchFailure as failure:
        # every way out carries its spend onto the span, a refusal or a broken
        # stream as much as an answer (AR-J1)
        _span_spend(failure.spend, provider=provider, egress=egress)
        set_span_attribute("answer.failure_class", failure.failure_class)
        raise
