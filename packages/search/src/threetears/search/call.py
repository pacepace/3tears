"""Call -- one query, one provider, one candidate set.

The layer between a caller's request and a provider adapter. It never
accumulates across providers (that is Aggregate's named type, D2), never
ranks (Select's, and only where a ranker was supplied), and never renders
(Bind's). What it does own:

- **safe default bounds when the caller tunes nothing** (SR-L6). A request
  with no result cap gets :data:`DEFAULT_MAX_RESULTS`, and one asking past
  :data:`MAX_RESULTS_CEILING` is refused rather than quietly clamped -- a
  local cap bounds a run's *shape*, and an overrun is a defect in the run
  rather than a billing event (D5). A call with no deadline gets
  :data:`DEFAULT_TIMEOUT_SECONDS`, as a parameter with a default rather
  than a constant in a call site (SR-G1).
- **criteria negotiation against declared capabilities** (SR-B4, SR-B2).
  The provider's declaration is consulted before the request goes out, and
  the returned dispositions are completed from it afterwards, so every
  criterion the caller sent has exactly one answer at this boundary no
  matter how thorough the adapter was.
- **failure mapping and spend attachment** (SR-J1, SR-E1, SR-E3). Nothing
  reaches a caller as an untyped exception, and every outcome -- including
  every failure -- carries what it consumed.

**Wall-clock is set here, not added here.** Call's measurement strictly
contains the transport's, so summing the two would report a call as having
taken twice as long as it did. Every other spend dimension accumulates; this
one is replaced with the authoritative number (SR-E2's discipline applied to
time).
"""

from __future__ import annotations

import time
from typing import Final

from threetears.observe import get_logger
from threetears.search.contracts import (
    CRITERION_MAX_RESULTS,
    CandidateSet,
    Criterion,
    CriterionDisposition,
    LocalCapExceeded,
    SearchFailure,
    SearchProvider,
    SearchRequest,
    Spend,
    TransportFailed,
)

__all__ = [
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_RESULTS_CEILING",
    "search",
]

_logger = get_logger(__name__)

#: candidates returned when the caller states no cap. Chosen to be safe
#: unturned on the smallest target (SR-L6): ten snippet-grade candidates is
#: a few tens of kilobytes, which fits beside another plane under a
#: ``MemoryMax`` cap without anyone having tuned anything.
DEFAULT_MAX_RESULTS: Final[int] = 10

#: the most this layer will let one call ask for. A caller past it is
#: refused with :class:`LocalCapExceeded` rather than clamped, because a
#: clamp is a silent degradation and this cap exists to make an overrun
#: visible as the defect it is (D5).
MAX_RESULTS_CEILING: Final[int] = 50

#: bound applied when the caller supplies no deadline. A default, not a
#: constant: every caller can override it, and one running under its own
#: deadline should pass what remains of it (SR-G1, SR-G2).
DEFAULT_TIMEOUT_SECONDS: Final[float] = 20.0


def _bounded_criteria(request: SearchRequest, *, max_results_ceiling: int) -> tuple[Criterion, ...]:
    """Return the request's criteria with this layer's default cap applied.

    :param request: the caller's request
    :ptype request: SearchRequest
    :param max_results_ceiling: the most a single call may ask for
    :ptype max_results_ceiling: int
    :return: the criteria to send, carrying a result cap either way
    :rtype: tuple[Criterion, ...]
    :raises LocalCapExceeded: when the caller asked past the ceiling
    """
    stated = [criterion for criterion in request.criteria if criterion.key == CRITERION_MAX_RESULTS]
    if not stated:
        return (*request.criteria, Criterion.max_results(DEFAULT_MAX_RESULTS))
    wanted = stated[-1].value
    if isinstance(wanted, int) and wanted > max_results_ceiling:
        raise LocalCapExceeded(
            f"request asks for {wanted} candidates; this call's local cap is {max_results_ceiling}",
            spend=Spend(),
            remediation=(
                f"lower max-results to {max_results_ceiling} or below, or raise max_results_ceiling on "
                f"the call if the run can afford the memory -- the cap bounds the run's shape, not its bill"
            ),
            scope=CRITERION_MAX_RESULTS,
        )
    return request.criteria


def _completed_dispositions(
    request: SearchRequest, result: CandidateSet, provider: SearchProvider
) -> tuple[CriterionDisposition, ...]:
    """Guarantee one disposition per criterion the caller sent (SR-B2).

    The adapter's own answers win: it knows what it actually did with each
    criterion, including precedence rules a declaration cannot express. Any
    criterion it answered for is left exactly as it reported. A criterion it
    said nothing about is answered from the provider's own capability
    declaration -- which is still the provider speaking, not a guess -- so a
    criterion can never reach a caller unanswered.

    :param request: the caller's request, holding the criteria to answer for
    :ptype request: SearchRequest
    :param result: what the provider returned
    :ptype result: CandidateSet
    :param provider: the provider, for its capability declaration
    :ptype provider: SearchProvider
    :return: the adapter's dispositions plus one per unanswered criterion
    :rtype: tuple[CriterionDisposition, ...]
    """
    answered = {disposition.criterion_key for disposition in result.dispositions}
    filled: list[CriterionDisposition] = list(result.dispositions)
    for criterion in request.criteria:
        if criterion.key in answered:
            continue
        answered.add(criterion.key)
        declared = provider.capabilities.disposition_for(criterion.key)
        filled.append(
            CriterionDisposition(
                criterion_key=criterion.key,
                disposition=declared,
                detail=(
                    f"answered from {provider.provider}'s capability declaration: the adapter reported "
                    f"nothing for this criterion"
                ),
            )
        )
        _logger.warning(
            "provider %s reported no disposition for criterion %s; answered from its declaration as %s",
            provider.provider_instance,
            criterion.key,
            declared,
        )
    return tuple(filled)


async def search(
    request: SearchRequest,
    *,
    provider: SearchProvider,
    timeout_seconds: float | None = None,
    max_results_ceiling: int = MAX_RESULTS_CEILING,
) -> CandidateSet:
    """Turn one request into one candidate set through one provider.

    :param request: what the caller asked for
    :ptype request: SearchRequest
    :param provider: the provider adapter to ask
    :ptype provider: SearchProvider
    :param timeout_seconds: bound for this call; None applies
        :data:`DEFAULT_TIMEOUT_SECONDS`. A caller under its own deadline
        should pass what remains of it (SR-G2)
    :ptype timeout_seconds: float | None
    :param max_results_ceiling: the most this call may ask for before it is
        refused (SR-L6, D5)
    :ptype max_results_ceiling: int
    :return: the candidates, one disposition per criterion, and the spend
        the call consumed. Zero candidates is a success (SR-J2)
    :rtype: CandidateSet
    :raises threetears.search.contracts.errors.SearchFailure: one of the
        typed classes, always carrying spend (SR-E3). Callers that must not
        see an exception at all go through
        :func:`threetears.search.bind.bind_search` (D10)
    """
    started = time.monotonic()
    bounded = request.model_copy(
        update={"criteria": _bounded_criteria(request, max_results_ceiling=max_results_ceiling)}
    )

    # --- Phase 1 PR 2 seams, deliberately left as seams -------------------
    # A ``BudgetPort.check(estimate)`` consultation belongs HERE, and a
    # ``BudgetPort.record(spend)`` immediately after the provider returns --
    # both on this side of the transport's retry boundary, so an attempt
    # that was retried and never billed never debits a budget (D4). A
    # ``RateLimiterPort`` acquisition keyed ``(provider instance, egress)``
    # belongs between the two (D8, D20). Neither port type exists yet
    # (search-spec.md §7, Phase 1 item 2) and neither is invented here: a
    # placeholder protocol would be a second vocabulary for that slice to
    # migrate off, which costs more than the wait.

    try:
        result = await provider.search(bounded, timeout_seconds=timeout_seconds or DEFAULT_TIMEOUT_SECONDS)
    except SearchFailure as failure:
        raise _timed(failure, elapsed=time.monotonic() - started) from failure
    except Exception as exc:
        # An adapter that lets an untyped exception out is a defect in the
        # adapter, not a reason for the caller to meet one: the taxonomy is
        # what every layer above is written against (SR-J1). Logged as the
        # defect it is, and re-raised typed rather than swallowed.
        _logger.exception(
            "provider %s raised an untyped exception; the adapter should map failures onto the taxonomy",
            provider.provider_instance,
        )
        raise TransportFailed(
            f"provider {provider.provider_instance} failed with an unmapped {type(exc).__name__}: {exc}",
            spend=Spend(wall_clock_seconds=time.monotonic() - started),
            provider_instance=provider.provider_instance,
        ) from exc

    return result.model_copy(
        update={
            "dispositions": _completed_dispositions(bounded, result, provider),
            "spend": result.spend.model_copy(update={"wall_clock_seconds": time.monotonic() - started}),
        }
    )


def _timed(failure: SearchFailure, *, elapsed: float) -> SearchFailure:
    """Restate ``failure`` with this layer's authoritative wall-clock.

    :param failure: the typed failure a provider raised
    :ptype failure: SearchFailure
    :param elapsed: seconds measured across the whole call
    :ptype elapsed: float
    :return: the same failure class, carrying the call's own wall-clock
    :rtype: SearchFailure
    """
    record = failure.to_record()
    timed_spend = record.spend.model_copy(update={"wall_clock_seconds": elapsed})
    return record.model_copy(update={"spend": timed_spend}).to_failure()
