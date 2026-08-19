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
- **budget consultation and pacing, in that order, around the provider
  call** (D4, D5, D8, D20). ``check`` then ``acquire`` then the call then
  ``record``, all of it below the transport's retry boundary so an attempt
  that was retried and never billed never debits a budget. Both ports are
  injected parameters and neither is imported here (P9, SR-L4): a consumer
  with no budgets passes nothing and pays nothing.

**What the budget is told is exactly what the caller is told.** ``record``
receives the same :class:`~threetears.search.contracts.spend.Spend` the
outcome carries -- the returned set's, or the typed failure's (SR-E3) --
never a locally re-derived tally (SR-E2). And it is called once per
*attempted* provider call and never otherwise: a call a budget refused and a
call a limiter never released have no bill for a budget to follow (D4).

**Where the on-by-default limiter is constructed, and why not here.**
§3.9's "MUST be on by default with safe rates (SR-L6)" is discharged by
:class:`~threetears.search.limiter.InProcessRateLimiter` itself: a
no-argument one is already paced at one call per second with a burst of
three, so a host gets safe rates by constructing one rather than by
choosing numbers. What Call does *not* do is mint one when the caller
passes none, and the reason is that it could not mint a useful one:

- a limiter constructed per call paces nothing, because two limiters do not
  share buckets -- the object has to outlive the call to be a limiter at
  all;
- the only object this module could share across calls is module-level
  mutable state, and the one such entry this package has argued under SR-O2
  is the limiter's own bucket map inside a limiter a *host* constructed
  (search-spec.md §3.9). A second, hidden, process-wide instance that no
  caller can configure, observe, or turn off is a different thing, and it
  would be reached for by exactly the ports discipline that says a limiter
  is injected at construction (P9, SR-L4);
- and it would have to choose a waiting policy for everyone. Fail-fast
  turns the fourth quick call in a process into a ``RateLimited`` the
  provider never issued, which is reacting rather than pacing (D8). Waiting
  has to come out of the caller's bound, which would make every call's
  provider timeout depend on hidden global state.

So the limiter is a parameter here, and the host constructs one per process
and passes it -- which is what :mod:`threetears.search.limiter` already
tells it to do ("Construct one per process and share it"). The default
belongs at the construction site that owns a process: the Phase-2 search
pod's per-spec wiring, where the base URL, the credentials and the egress
are already chosen.

**Wall-clock is set here, not added here.** Call's measurement strictly
contains the transport's, so summing the two would report a call as having
taken twice as long as it did. Every other spend dimension accumulates; this
one is replaced with the authoritative number (SR-E2's discipline applied to
time).
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Final

from threetears.observe import get_logger
from threetears.search.contracts import (
    CRITERION_MAX_RESULTS,
    EGRESS_DIRECT,
    PRICING_PER_WEIGHTED_UNIT,
    BudgetDecision,
    BudgetPort,
    CandidateSet,
    Criterion,
    CriterionDisposition,
    LocalCapExceeded,
    RateLimited,
    RateLimiterPort,
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
    "PACING_BURST_SCOPE",
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

#: the cap identity a never-grantable pacing ask reports on its refusal, so a
#: reader can tell "the host's burst is smaller than one call" from a budget
#: refusal without parsing prose. ``LocalCapExceeded.scope`` doubles as cap
#: identity here, following ``response-bytes`` and ``max-results``
#: (search-spec.md §7).
PACING_BURST_SCOPE: Final[str] = "pacing-burst"


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


def _estimate(provider: SearchProvider) -> Spend:
    """State what one prospective call to ``provider`` is expected to cost.

    Denominated in :class:`~threetears.search.contracts.spend.Spend` because
    the estimate and the eventual record must be the same five dimensions or
    the cap and the bill drift apart (SR-E2).

    ``calls=1`` is the floor every provider shares and the dimension SR-D1
    names first ("budgets in calls, not only money") -- it is also the whole
    of the honest estimate for a free self-hosted instance, which bills
    nothing and fails by ban (D6, SR-D6). The one addition is read off the
    provider's own *declaration* rather than its name: a provider that says
    it charges in weighted units charges at least one of them per call, so a
    budget denominated in credits gets a number it can refuse on instead of
    a zero it can never refuse on. It is a floor and not a quote -- what
    *this* request weighs (Tavily's ``advanced`` = 2 credits, SR-E4) depends
    on how the adapter planned it, and re-deriving that here would be Call
    keeping a second tally of a number the adapter already owns (SR-E2).
    Money is left at zero for the same reason: a per-request provider's
    price is a deployment fact no capability declaration carries, and an
    invented one would be a synthetic bill (D6).

    Wall-clock is left at zero deliberately. The only wall-clock figure Call
    could offer before the call is the deadline, which is a ceiling rather
    than an expectation, and reporting a ceiling as an estimate would refuse
    a 200ms call whenever less than the whole timeout remained.

    :param provider: the provider about to be called, for its declaration
    :ptype provider: SearchProvider
    :return: the estimate to hand :meth:`BudgetPort.check`
    :rtype: Spend
    """
    weighted = provider.capabilities.pricing_model == PRICING_PER_WEIGHTED_UNIT
    return Spend(
        calls=1,
        provider_units=Decimal(1) if weighted else Decimal(0),
        provider_unit=provider.capabilities.qualified_unit if weighted else None,
    )


def _refusal(decision: BudgetDecision, *, provider: SearchProvider, egress: str) -> LocalCapExceeded:
    """Turn a budget's refusal into this layer's typed failure (SR-D3, D5).

    The port returns a decision rather than raising, so the taxonomy mapping
    happens here, where the rest of it already lives. It maps onto
    :class:`~threetears.search.contracts.errors.LocalCapExceeded` and never
    onto :class:`~threetears.search.contracts.errors.QuotaExhausted`: this
    is the *local* authority bounding a run's shape, and the provider's own
    refusal about money can only ever come back from a provider call.

    :param decision: the refusal, carrying which scope said no and what it
        had consumed
    :ptype decision: BudgetDecision
    :param provider: the provider the refused call was bound for
    :ptype provider: SearchProvider
    :param egress: the exit the refused call would have left by (D20)
    :ptype egress: str
    :return: the failure to raise
    :rtype: LocalCapExceeded
    """
    scope = decision.scope or "an unnamed scope"
    return LocalCapExceeded(
        decision.reason or f"{scope} refused this call to {provider.provider_instance}",
        # The refusing scope's own consumed total, as the port reported it:
        # a run stopped by a cap still says what it cost (SR-E3), and the
        # number is the budget's, never one recomputed here (SR-E2).
        spend=decision.consumed,
        provider_instance=provider.provider_instance,
        remediation=decision.remediation,
        egress=egress,
        scope=decision.scope,
    )


async def _pace(
    limiter: RateLimiterPort,
    *,
    provider: SearchProvider,
    egress: str,
    budget_seconds: float,
    started: float,
) -> float:
    """Wait for this call's turn at ``(provider instance, egress)`` (D8, D20).

    The whole of the caller's remaining bound is offered as the wait, which
    is what makes this pacing rather than reacting: a call that arrives a
    little early sleeps toward its token and then proceeds, and only a call
    whose token never comes inside the bound it was given is refused. What
    the wait consumes is subtracted from what the provider is then given, so
    the bound the caller stated is the bound the whole call honours (SR-G2)
    -- Call may spend it pacing or spend it waiting on the provider, but not
    both over again.

    One call asks for one token. A provider's *weight* is a billing fact
    (SR-E4) that the budget already carries; how often we knock on an
    instance is a different quantity, and a host that wants a weighted key
    paced differently configures that key's rate rather than inflating
    every acquisition here.

    :param limiter: the injected pacing seam
    :ptype limiter: RateLimiterPort
    :param provider: the provider about to be called, for its instance name
    :ptype provider: SearchProvider
    :param egress: the exit the call will leave by (D20)
    :ptype egress: str
    :param budget_seconds: what is left of the caller's bound
    :ptype budget_seconds: float
    :param started: the call's own monotonic start, for authoritative
        wall-clock on the refusal
    :ptype started: float
    :return: what remains of the bound for the provider call itself
    :rtype: float
    :raises threetears.search.contracts.errors.RateLimited: when the key did
        not release inside the bound the caller gave
    :raises threetears.search.contracts.errors.LocalCapExceeded: when the
        key's configured burst is smaller than the one token a call costs,
        so no wait could ever release it
    """
    waiting_from = time.monotonic()
    try:
        decision = await limiter.acquire(
            provider_instance=provider.provider_instance,
            egress=egress,
            max_wait_seconds=max(0.0, budget_seconds),
        )
    except ValueError as exc:
        # A limiter whose key is configured with a burst below one token can
        # never release a single call, and the port's answer to an ask no
        # amount of waiting could satisfy is a ValueError rather than a
        # denial (threetears.search.limiter, and any adapter written to the
        # same shape). Left uncaught it would reach the caller untyped,
        # violating SR-J1 and -- through Bind's unmapped-exception path --
        # blaming the provider for what is purely local configuration.
        #
        # Mapped here rather than fixed in the limiter, for three reasons:
        #
        # - the taxonomy mapping belongs in the consuming layer, which is
        #   the same ruling that keeps a budget's refusal a returned
        #   decision rather than a raise (search-spec.md §7). Call is also
        #   the only layer that meets *every* limiter, ours or a host's
        #   adapter over core's TokenBucket, so mapping here covers the
        #   injected one too;
        # - a denial would be the wrong value for the limiter to return: a
        #   denial carries a retry_after, and this pacing loop sleeps toward
        #   it -- toward a token that is never coming;
        # - and the limiter must keep raising for a genuinely negative ask,
        #   which is a programming error rather than configuration.
        #
        # LocalCapExceeded rather than RateLimited because of who said no
        # (D5): RateLimited means "the pace is too fast, come back later",
        # and there is no later here. This is a locally-configured cap
        # refusing the run's shape, and ``scope`` names which cap.
        raise LocalCapExceeded(
            f"pacing for {provider.provider_instance} via {egress} can never release a call: {exc}",
            # The call never happened, so the only real dimension is the
            # wall-clock the caller waited (SR-E3), exactly as on the denial
            # below.
            spend=Spend(wall_clock_seconds=time.monotonic() - started),
            provider_instance=provider.provider_instance,
            remediation=(
                "raise burst_tokens for this key on the limiter the host constructed -- one call costs one "
                "token, so a burst below one paces every call to a standstill; this is our own pacing "
                "configuration, not the provider's refusal"
            ),
            egress=egress,
            scope=PACING_BURST_SCOPE,
        ) from exc
    waited = time.monotonic() - waiting_from
    if not decision.acquired:
        raise RateLimited(
            f"pacing for {provider.provider_instance} via {egress} did not release within "
            f"{budget_seconds:.3f}s; the call was not made",
            # The call never happened, so there is no bill: no calls, no
            # money, no bytes. Wall-clock is the one dimension that is real
            # -- the caller waited it -- and SR-E3's "every failure carries
            # what it consumed" is satisfied by saying exactly that, rather
            # than by charging for a request nobody sent.
            spend=Spend(wall_clock_seconds=time.monotonic() - started),
            provider_instance=provider.provider_instance,
            remediation=(
                "raise this call's timeout so pacing can wait it out, raise the key's configured rate if "
                "the instance can take it, or spread the calls -- this refusal is our own pacing, not the "
                "provider's"
            ),
            egress=egress,
            retry_after_seconds=decision.retry_after_seconds,
        )
    return max(0.0, budget_seconds - waited)


async def search(
    request: SearchRequest,
    *,
    provider: SearchProvider,
    timeout_seconds: float | None = None,
    max_results_ceiling: int = MAX_RESULTS_CEILING,
    budget: BudgetPort | None = None,
    limiter: RateLimiterPort | None = None,
    egress: str = EGRESS_DIRECT,
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
    :param budget: the local refusal authority to consult, ``check`` before
        the call and ``record`` after (D4, D5). None makes no consultation
        at all: a consumer without budgets is not given an implicit one,
        because a budget nobody configured has no cap to enforce
    :ptype budget: BudgetPort | None
    :param limiter: the pacing seam for this call's ``(provider instance,
        egress)`` key (D8, D20). None leaves the call unpaced by *this*
        layer -- see this module's note on where the on-by-default limiter
        of SR-L6 is constructed
    :ptype limiter: RateLimiterPort | None
    :param egress: which exit this provider's transport leaves by (D20),
        half of the pacing key and stamped onto refusals so a consumer can
        rebuild it. ``direct`` is a named value, not an absence, so it is
        the default rather than None
    :ptype egress: str
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
    scope_tags = request.budget_scope_tags
    remaining_seconds = DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds

    if budget is not None:
        # Before the provider call and below the retry boundary (D4). A
        # refusal short-circuits everything downstream of it, the limiter
        # included: a call that will not be made needs no pacing slot, and
        # taking one would pace the calls that *are* being made out of a
        # bucket a refusal already emptied.
        decision = await budget.check(_estimate(provider), scope_tags=scope_tags)
        if not decision.allowed:
            raise _refusal(decision, provider=provider, egress=egress)

    if limiter is not None:
        remaining_seconds = await _pace(
            limiter, provider=provider, egress=egress, budget_seconds=remaining_seconds, started=started
        )

    try:
        result = await provider.search(bounded, timeout_seconds=remaining_seconds)
    except SearchFailure as failure:
        _stamp(failure, elapsed=time.monotonic() - started)
        # SR-E3: a typed failure carries the spend it incurred, and that
        # spend is as real as a success's -- an attempt that reached the
        # provider and failed after billing has to debit. Recorded before
        # the failure propagates, and with the same number the caller is
        # about to receive. Nothing between the catch and this line can fail
        # on the failure's *class*, which is what SR-E2/SR-E3 need: a
        # third-party adapter's own SearchFailure subclass is as billable as
        # one of the seven, and a debit skipped because a taxonomy reader
        # did not recognise the class would under-count a call that was
        # attempted and possibly billed.
        await _record(budget, failure.spend, scope_tags=scope_tags)
        raise
    except Exception as exc:
        # An adapter that lets an untyped exception out is a defect in the
        # adapter, not a reason for the caller to meet one: the taxonomy is
        # what every layer above is written against (SR-J1). Logged as the
        # defect it is, and re-raised typed rather than swallowed.
        _logger.exception(
            "provider %s raised an untyped exception; the adapter should map failures onto the taxonomy",
            provider.provider_instance,
        )
        # Recorded too, and with the only honest number available: a
        # defective adapter reported no spend, so Call reports the one
        # dimension it measured itself and claims no bill it cannot see.
        # Silence here would be worse -- the attempt happened, and a budget
        # told nothing about it would under-count every defective call --
        # and inventing ``calls=1`` would be Call billing on a guess. Either
        # way the budget hears exactly what the caller hears.
        failed_spend = Spend(wall_clock_seconds=time.monotonic() - started)
        await _record(budget, failed_spend, scope_tags=scope_tags)
        raise TransportFailed(
            f"provider {provider.provider_instance} failed with an unmapped {type(exc).__name__}: {exc}",
            spend=failed_spend,
            provider_instance=provider.provider_instance,
        ) from exc

    completed = result.model_copy(
        update={
            "dispositions": _completed_dispositions(bounded, result, provider),
            "spend": result.spend.model_copy(update={"wall_clock_seconds": time.monotonic() - started}),
        }
    )
    # After the provider answered, with what it actually reported spending
    # (SR-E2) -- and still below the retry boundary, so the attempts the
    # transport made and was never billed for contribute nothing (D4).
    await _record(budget, completed.spend, scope_tags=scope_tags)
    return completed


async def _record(budget: BudgetPort | None, spend: Spend, *, scope_tags: tuple[str, ...]) -> None:
    """Debit ``spend`` where a budget was injected, and nothing where none was.

    One call site's worth of ``if`` lifted out of three, so that the rule --
    every attempted provider call is recorded exactly once, whatever its
    outcome -- is visible as one thing rather than repeated three times.

    A port that *raises* is deliberately not caught. The untyped-escape
    mapping above says "provider X failed with an unmapped ...", and putting
    a consumer's own bug behind that sentence would blame the provider for
    it; a defect in an injected port surfaces as the defect it is.

    :param budget: the injected authority, or None when the consumer has none
    :ptype budget: BudgetPort | None
    :param spend: what the attempt consumed, as the layer that made it
        reported (SR-E2)
    :ptype spend: Spend
    :param scope_tags: the scopes to debit -- the same tags the matching
        :meth:`BudgetPort.check` was given (SR-D2)
    :ptype scope_tags: tuple[str, ...]
    :return: nothing
    :rtype: None
    """
    if budget is not None:
        await budget.record(spend, scope_tags=scope_tags)


def _stamp(failure: SearchFailure, *, elapsed: float) -> None:
    """Give ``failure`` this layer's authoritative wall-clock, in place.

    **Why the failure is stamped rather than rebuilt.** The obvious spelling
    -- project to :class:`~threetears.search.contracts.errors.FailureRecord`,
    copy the spend, and rebuild through
    :meth:`~threetears.search.contracts.errors.FailureRecord.to_failure` --
    round-trips through the seven-entry class registry, which refuses
    anything it does not recognise (correctly: a reader meeting an unknown
    wire name must not guess, D26). But nothing here is reading a wire
    record. The typed failure is already in hand, raised in this process by
    an adapter that may legitimately be a third party's: base
    :class:`~threetears.search.contracts.errors.SearchFailure` and any
    subclass of it satisfy the seam, and D13 makes the taxonomy additive, so
    an unknown class is a supported shape rather than a defect. Sending one
    through the registry would replace the caller's typed failure with a
    ``ValueError`` *inside the except handler* -- the exact untyped escape
    SR-J1 forbids -- and would strand the debit SR-E3 owes.

    Rebuilding through ``type(failure)(...)`` was the other candidate and is
    rejected for a smaller reason: the taxonomy declares no constructor
    contract subclasses must keep, so a subclass with one required keyword
    of its own would fail exactly where the registry does. Stamping needs no
    such contract, preserves every field a subclass carries (``scope``,
    ``retry_after_seconds``, whatever a third party added), and keeps the
    original traceback because the caller re-raises rather than raising
    anew. The mutation is safe because the object is a failure in flight:
    it was constructed for this raise and is on its way to one caller.

    Wall-clock is *replaced* rather than added: Call's measurement strictly
    contains the transport's (see this module's docstring).

    :param failure: the typed failure a provider raised, mutated in place
    :ptype failure: SearchFailure
    :param elapsed: seconds measured across the whole call
    :ptype elapsed: float
    :return: nothing
    :rtype: None
    """
    failure.spend = failure.spend.model_copy(update={"wall_clock_seconds": elapsed})
