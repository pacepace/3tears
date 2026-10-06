"""Per-role token + cost accumulation for eval results.

Every role an eval cell spends on — candidate, judge, simulator, inner agent, and the
metered external APIs — is captured at its own call site from the object that call already
returns, and folded into a :class:`RoleUsageLedger`. The ledger emits the persisted :class:`~threetears.evals.contracts.models.RoleUsage`
rows on ``EvalResult.usage``.

Two rules the whole module exists to hold:

**Missing is not zero.** Token counts and dollars are ``None`` until something is actually
observed. A role that reported no reasoning split carries ``None`` reasoning; a role that
reported ``0`` carries ``0``. Coercing either way fabricates an observation, and per-role
cost attribution depends on being able to say "not measured".

**Rows are keyed by (role, model, provider, unit, price source), not role alone.** A judge run can score
different dimensions with different models (each :class:`~threetears.evals.contracts.models.JudgeConfig`
may pin its own), so blending them into one row would have to drop ``model`` — destroying
exactly the attribution that ``model`` + ``price_source`` exist to preserve, since dollars
are only re-derivable at current rates if you know which model spent them. The provider and
unit halves carry the same property for external spend, and one consequence beyond
attribution: two providers' weighted units are not one quantity, so keying on the pair means
a cross-provider sum has nowhere to happen. The price source is in the key for the same
reason one step further: dollars priced two ways are re-derivable only while they stay apart.
Cost views sum by role membership and are indifferent to how many rows a role occupies.
Token-metered roles report no provider, so their key degenerates to the model and its price
source.

**The engine names no price source of its own.** Where dollars came from is the caller's to say
— a completion client reports it (``CompletionResult.price_source``), a piece of background work
carries it on its :class:`~threetears.evals.contracts.models.AsyncDelivery`, a rate table names the
rate it applied — and the ledger stores what it is told. A source nobody named is ``None``, never
a provider the engine assumed.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.host.spend import ExternalSpend
from threetears.evals.contracts.models import CellTermination, RoleUsage, SimulatorPurpose, UsageRole
from threetears.evals.contracts.provider import sum_optional_tokens
from threetears.evals.contracts.spend import ExternalRateTable, reported_price_source
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.models import AsyncDelivery, EvalResult

log = get_logger(__name__)

#: Roles whose spend a production deployment would also incur. The judge and simulator are
#: measurement apparatus — they cost the eval program money but would never run in
#: production, so conflating the two views overstates what the candidate costs to operate.
#: Derived from membership; deliberately never a stored field on ``RoleUsage``.
PRODUCTION_REPLICATING_ROLES: frozenset[UsageRole] = frozenset({"candidate", "inner_agent", "external"})

#: Roles whose spend the runner folds into the blended ``EvalResult.cost_usd``, in the
#: order it is stored. Stamped onto every result it writes so the number can be placed:
#: this set has changed before and will change again, and a total whose composition is
#: not recorded cannot be compared with one from either side of a move.
#:
#: ``external`` is absent here because most external providers return no per-call dollar
#: figure: it joins the composition only for a run that resolved a rate table able to price
#: one of them, which is what :func:`blended_cost_roles` decides and what the stamped
#: ``cost_roles`` records. A total that named it unconditionally would claim an observation
#: nobody made.
#:
#: A tuple rather than a frozenset: this one is persisted, so a stable order keeps two
#: recordings of the same convention byte-identical. Its sibling above is derived at
#: read and rightly unordered.
BLENDED_COST_ROLES: tuple[UsageRole, ...] = ("candidate", "inner_agent", "judge", "simulator")


def blended_cost_roles(rate_table: ExternalRateTable | None) -> tuple[UsageRole, ...]:
    """The roles a run's blended ``EvalResult.cost_usd`` sums, in stored order.

    ``external`` is in exactly when the run resolved a rate table that can put dollars on
    the external calls it is able to make — the convention that run committed to, not an
    observation of whether any external call happened (a cell that searched nothing still
    names it, for the same reason a cell whose judge never ran still names ``judge``).
    Stamped onto every result so a stored total can be placed against the composition it
    was summed under.

    **Action-seam metered calls do not widen this**, and the reason is that this marker is a
    claim about ``cost_usd``. A run can put ``external`` ROWS on a result without any
    background tool — a candidate whose only metered tool is a synchronous one reports the
    calls it made. Those rows carry units and deliberately no dollars
    (:meth:`RoleUsageLedger.add_external_unpriced`), so they contribute nothing to the blended
    total, and naming the role for them would say the total sums a spend it does not. A reader
    wanting the volume reads ``usage``; ``cost_roles`` answers only "what is inside this
    number". When action-seam dollars become expressible — which needs the cassette withholding
    widened to that seam first — the resolution widens with them.

    Args:
        rate_table: The run's rate table, resolved once at launch, or None when it declared
            no usable rates at all.

    Returns:
        The role tuple to stamp on each of the run's results.
    """
    if rate_table is not None and rate_table.rates:
        return (*BLENDED_COST_ROLES, "external")
    return BLENDED_COST_ROLES


@dataclass
class CallUsage:
    """One LLM call's observed usage, for roles whose client result doesn't survive to the runner.

    Field names deliberately mirror :class:`~threetears.evals.contracts.provider.CompletionResult` so
    both feed :meth:`RoleUsageLedger.add_llm_result` through the same duck type — the
    ledger's subset of that protocol, which also names what eval reads off a completion
    elsewhere (``content``, and the ``stop_reason`` a truncation is read from). ``cost_usd``
    and ``reasoning_tokens`` are ``None`` when unobserved — the judge path in particular must
    carry an honest "no cost reported" rather than the ``0.0`` its blended-total field uses.
    The token counts follow the same rule: ``None`` is a provider that reported no count, never
    a measured zero.
    """

    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    #: Where ``cost_usd`` came from, as the completion that reported it named it
    #: (``CompletionResult.price_source``); ``None`` when it named none.
    price_source: str | None = None
    #: Provider calls folded into this usage — more than one when a caller retried and
    #: aggregated the attempts, all of which spent real tokens.
    calls: int = 1


@dataclass
class _ModelTotals:
    """Running totals for one (role, model) pair."""

    #: Every token total is ``int | None`` and accumulates through ``sum_optional_tokens``,
    #: so a role that reported no count keeps ``None`` rather than a 0 that would claim a
    #: measurement. This matters beyond reasoning: a background delivery whose trace was evicted
    #: reports a cost with no tokens, and the external role has no token concept at all.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float = 0.0
    #: Dollars need a separate flag because 0.0 is a plausible real cost and the
    #: accumulator is a float; tokens carry ``None`` in the field itself. Same
    #: missing-vs-zero rule, two representations forced by the types.
    cost_observed: bool = False
    #: A contribution arrived with no price. Unlike tokens, dollars do NOT keep "the numbers it
    #: does have": a row's ``cost_usd`` is read as what that row's calls cost, so a partial sum
    #: of a priced call and an unpriced one would be read as the whole. One unpriced
    #: contribution makes the row's dollars unknown.
    cost_unpriced: bool = False
    #: ``None`` until something actually counts calls. Same missing-vs-zero rule as the
    #: token fields: a role that reports spend without reporting how many calls produced it
    #: must not imply a count, because cost-per-call read off a fabricated denominator is
    #: worse than no denominator at all.
    call_count: int | None = None
    #: The provider's weighted units, accumulated from what each contribution REPORTED
    #: rather than multiplied out of a rate card at read. Deriving a total from one
    #: units-per-call figure is what billed a cheap call at a dearer call's rate — the code
    #: doing the arithmetic was not the code that knew the call. ``None`` until something reports units, on the same
    #: missing-is-not-zero rule as the token fields: a paid provider that publishes no
    #: per-call unit contributes calls and no units.
    provider_units: int | None = None


#: One ledger row's key: ``(model, provider, provider_unit, price_source, actor_id, purpose)``.
_RowKey = tuple[str | None, str | None, str | None, str | None, str | None, SimulatorPurpose | None]


@dataclass
class RoleUsageLedger:
    """Accumulates one role's LLM spend across every call it makes in a cell.

    Fed once per LLM call. ``rows()`` returns one :class:`RoleUsage` per distinct model, in
    first-seen order, or an empty list when the role never ran — the caller decides whether
    an absent role means "didn't run" or "not captured".
    """

    role: UsageRole
    #: The run's rate table, one entry per ``(provider, unit)``, resolved once at launch.
    #: ``None`` on every token-metered role and on an external role whose run declared no
    #: usable rates — then the calls are counted and nothing else is claimed about them.
    rate_table: ExternalRateTable | None = None
    #: Keyed by ``(model, provider, provider_unit, price_source, actor_id, purpose)``. There is no
    #: ledger-wide price source: each contribution names its own, so a row's provenance is what its
    #: contributions reported rather than what the ledger was built expecting. The last two are the
    #: simulator's attribution and ``None`` on every other role.
    _totals: dict[_RowKey, _ModelTotals] = field(default_factory=dict, init=False)

    @classmethod
    def for_external(cls, rate_table: ExternalRateTable | None) -> RoleUsageLedger:
        """Build the external role's ledger against the run's rate table.

        The one constructor for this role, so no caller can invent its own idea of what an
        unpriced external row looks like. A table prices each contribution at its own
        provider's rate, so the provenance is per row and is named where the pricing happens.

        Args:
            rate_table: The run's rate table, or None when it declared no usable rates.

        Returns:
            A ledger that prices each reported spend at its own ``(provider, unit)`` rate.
        """
        return cls(role="external", rate_table=rate_table)

    def add(
        self,
        *,
        model: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        reasoning_tokens: int | None,
        cost_usd: float | None,
        calls: int | None = 1,
        provider_units: int | None = None,
        provider: str | None = None,
        provider_unit: str | None = None,
        price_source: str | None = None,
        actor_id: str | None = None,
        purpose: SimulatorPurpose | None = None,
    ) -> None:
        """Fold one call's usage into this role's totals.

        Every token argument is optional and ``None`` means UNREPORTED, never zero — a
        background delivery whose trace was evicted carries a real cost with no token counts,
        and coercing those to 0 would state a measurement nobody made.

        ``calls`` is how many provider calls this contribution represents — 1 for a single
        LLM response, the batch size when a role reports its work in aggregate (background
        rounds, metered calls), or ``None`` when the caller genuinely cannot say. A row must
        count the SAME calls whose tokens it reports; where that is unknowable, ``None`` is
        the honest answer.

        ``provider_units`` is the provider's own metered unit for this contribution,
        reported by whoever made the call. ``None`` — every LLM role, and any provider
        publishing no countable unit — leaves the row untouched rather than claiming zero.

        ``price_source`` is where ``cost_usd`` came from, as the caller names it; ``None`` when
        it names none. The ledger never supplies one.

        **Totals key on ``(model, provider, provider_unit, price_source)``, not model alone.**
        That is what makes "units from different providers are never summed" structural rather
        than a rule every downstream reader has to remember: two providers land in two rows and
        there is no place for the addition to happen. The price source rides in the key for the
        same reason: dollars priced two ways in one row could not be re-derived from it.
        Token-metered roles pass no provider, so their key degenerates to the model and the
        source its client named.

        ``actor_id`` and ``purpose`` are the simulator's attribution — the simulated actor a call spoke
        for or chose, and whether it was an ``utterance`` or a ``schedule`` pick — and key the row too,
        so what each actor and the scheduler spent is a stored row of its own. Every other role passes
        neither.

        Raises:
            ValueError: ``actor_id`` or ``purpose`` was given to a role other than ``simulator``.
        """
        if (actor_id is not None or purpose is not None) and self.role != "simulator":
            raise ValueError(f"actor_id and purpose attribute simulator calls, not the {self.role!r} role's")
        key = (model or None, provider, provider_unit, price_source, actor_id, purpose)
        totals = self._totals.setdefault(key, _ModelTotals())
        totals.call_count = sum_optional_tokens(totals.call_count, calls)
        totals.provider_units = sum_optional_tokens(totals.provider_units, provider_units)
        # Sums what was reported rather than discarding real measurements when one call was
        # silent — a partially-reporting role keeps the numbers it does have.
        totals.prompt_tokens = sum_optional_tokens(totals.prompt_tokens, prompt_tokens)
        totals.completion_tokens = sum_optional_tokens(totals.completion_tokens, completion_tokens)
        totals.reasoning_tokens = sum_optional_tokens(totals.reasoning_tokens, reasoning_tokens)
        if cost_usd is not None:
            totals.cost_usd += cost_usd
            totals.cost_observed = True
        else:
            totals.cost_unpriced = True

    def add_llm_result(
        self, result: Any, *, actor_id: str | None = None, purpose: SimulatorPurpose | None = None
    ) -> None:
        """Fold an ``LLMResult``-shaped response (judge / simulator / generator clients).

        Read defensively via ``getattr``: the judge and simulator accept any object
        satisfying their narrow client protocols, and test doubles legitimately supply only
        the fields they exercise. An absent ``reasoning_tokens`` attribute is the same
        statement as an unreported one — unknown, not zero.

        Args:
            result: The response, or a :class:`CallUsage` standing for one.
            actor_id: The simulated actor the call spoke for or chose; simulator role only.
            purpose: Whether the simulator call was an ``utterance`` or a ``schedule`` pick; simulator
                role only.
        """
        self.add(
            model=getattr(result, "model", None) or None,
            prompt_tokens=getattr(result, "input_tokens", None),
            completion_tokens=getattr(result, "output_tokens", None),
            reasoning_tokens=getattr(result, "reasoning_tokens", None),
            cost_usd=getattr(result, "cost_usd", None),
            # A CallUsage may aggregate retries into one contribution; a bare LLMResult is
            # always exactly one call.
            calls=getattr(result, "calls", 1),
            price_source=getattr(result, "price_source", None),
            actor_id=actor_id,
            purpose=purpose,
        )

    def add_external(self, spend: ExternalSpend) -> None:
        """Record a reported external spend, priced at its OWN provider's declared rate.

        The delivery seam's entry point — a background tool's metered calls, whose spend the
        tool reports back and whose dollars the run may legitimately claim.

        **The caller supplies the volume; this only prices it.** Nothing here derives units
        from a provider parameter, which is what let a cheap call be billed at a dearer
        call's rate: the code doing the arithmetic was not the code that knew the call. Pricing goes through the run's table at this spend's exact
        ``(provider, unit)``, never at another provider's rate and never at a default
        standing in for a rate nobody set — an undeclared rate leaves the calls counted and
        honestly unpriced. The conversion lives here rather than at the call site so it has
        exactly one home: a per-caller conversion is how a second caller would come to
        disagree with the first about what a unit costs.

        Tokens stay ``None`` rather than 0: these are not LLM calls, so zero would be a
        claim about a quantity that does not apply.

        Args:
            spend: What the caller reports having consumed at one provider.
        """
        money = self.rate_table.money_for(spend) if self.rate_table is not None else spend.money
        self.add(
            model=None,
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            cost_usd=money,
            calls=spend.calls,
            provider_units=spend.provider_units,
            provider=spend.provider,
            provider_unit=spend.provider_unit,
            price_source=(
                reported_price_source(spend.provider, from_rate_table=spend.money is None)
                if money is not None and spend.provider is not None
                else None
            ),
        )

    def add_external_unpriced(self, spend: ExternalSpend) -> None:
        """Record an action-seam spend: counted, attributed, never priced.

        The candidate's own synchronous tool calls, as the host's tool dispatcher counts
        them. One thing separates them from :meth:`add_external`, and it is not the rate
        table. The dispatcher counts only what it actually dispatches: a metered tool the
        host mutes under eval reaches no provider and lands nothing here.

        **Dollars are structurally absent.** Not "absent because no rate was declared" —
        absent because this method never consults a rate at all. Counting happens at the
        dispatcher, through the ``CassetteProxy`` instance, so a REPLAYED call lands the
        same calls and units as a live one; that is the intended reading and not an
        oversight, because these are production-replicating VOLUME — what the candidate's
        behaviour would cost a production run — rather than an observation of what this
        apparatus just spent. Reporting zero on the replay arm would publish the
        cheaper-than-reality figure the delivery seam deliberately withholds rather than
        understates, and would let two eval configurations describe identical candidate
        behaviour differently, which is the one thing the cassette layer exists to prevent.
        Attaching money would make that replay report spend it never incurred, and the
        withholding that guards it (:func:`production_replicating_cost`) is scoped to the
        delivery seam. Widening it to a second seam changes what every run records, which
        is a measurement decision rather than an accounting fix. The signature holds that
        line: this method takes no rate and has no path to one.

        Args:
            spend: What the dispatcher counted at one provider.
        """
        self.add(
            model=None,
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            cost_usd=None,
            calls=spend.calls,
            provider_units=spend.provider_units,
            provider=spend.provider,
            provider_unit=spend.provider_unit,
        )

    def rows(self) -> list[RoleUsage]:
        """Build this role's persisted rows — one per ledger key (:meth:`add`), first-seen order.

        ``provider_units`` is the SUM of what each contribution REPORTED, never the call
        count multiplied by one rate: deriving a total from one units-per-call figure is
        what billed a ``basic`` search at an ``advanced`` rate. A run whose operator
        declared no price still consumed a countable amount of the provider's unit, so
        units are reported whether or not dollars are, and ``None`` when nothing reported
        any — the same missing-is-not-zero rule the token fields carry in their own types.

        **Two providers produce two rows, and that is the whole guarantee.** Units are
        comparable only within one ``(provider, unit)``, so keying the rows on it means
        there is nowhere for a cross-provider sum to happen — a rule downstream code cannot
        forget because it never gets the chance.

        Returns:
            One :class:`~threetears.evals.contracts.models.RoleUsage` per distinct key, or an empty list
            when the role never ran.
        """
        rows: list[RoleUsage] = []
        for (model, provider, provider_unit, price_source, actor_id, purpose), totals in self._totals.items():
            rows.append(
                RoleUsage(
                    role=self.role,
                    model=model,
                    prompt_tokens=totals.prompt_tokens,
                    completion_tokens=totals.completion_tokens,
                    reasoning_tokens=totals.reasoning_tokens,
                    cost_usd=round(totals.cost_usd, 6) if totals.cost_observed and not totals.cost_unpriced else None,
                    price_source=price_source if totals.cost_observed and not totals.cost_unpriced else None,
                    call_count=totals.call_count,
                    provider=provider,
                    provider_unit=provider_unit,
                    provider_units=totals.provider_units,
                    actor_id=actor_id,
                    purpose=purpose,
                )
            )
        return rows


def async_delivery_usage(
    deliveries: Sequence[AsyncDelivery] | None, *, rate_table: ExternalRateTable | None
) -> list[RoleUsage]:
    """The ``inner_agent`` and ``external`` rows the background work in ``deliveries`` reports spending (R3).

    Background work runs on a detached task, so none of its spend is in the candidate's own turns;
    each :class:`~threetears.evals.contracts.models.AsyncDelivery` carries what its work spent, and
    this folds every entry into the two roles. Work still in flight when the cell ended is folded
    like work that delivered — it spent either way, and dropping it would publish a
    production-replicating cost short by exactly the work that ran longest.

    A substituted entry reports no spend (``AsyncDelivery`` refuses one that does), so a replayed or
    seeded payload can contribute no dollars here. An entry reporting dollars but no token counts
    (an evicted trace) yields a cost-only row: a true statement about what the record evidences,
    where a zero would be an invented measurement. Paid non-LLM calls carry the charge their provider
    reported where it reported one, which wins over any rate; otherwise they are priced at the run's own
    rate for each ``(provider, unit)`` (:meth:`RoleUsageLedger.add_external`), and left counted and
    unpriced — unknown, never zero — where neither exists.

    Args:
        deliveries: The cell's background work, as its kind reported it; ``None`` when it watched none.
        rate_table: The run's rate table, or ``None`` when it priced nothing.

    Returns:
        The inner-agent rows, then the external rows — empty when nothing reported spend.
    """
    inner = RoleUsageLedger(role="inner_agent")
    external = RoleUsageLedger.for_external(rate_table)
    for entry in deliveries or ():
        if any(
            value is not None
            for value in (
                entry.input_tokens,
                entry.output_tokens,
                entry.reasoning_tokens,
                entry.llm_calls,
                entry.cost_usd,
            )
        ):
            inner.add(
                model=entry.model,
                prompt_tokens=entry.input_tokens,
                completion_tokens=entry.output_tokens,
                reasoning_tokens=entry.reasoning_tokens,
                cost_usd=entry.cost_usd,
                # Uncoerced: a row counts the SAME calls whose tokens it reports, and an entry that
                # did not say how many calls it made does not know.
                calls=entry.llm_calls,
                price_source=entry.price_source,
            )
        for spend in entry.external_spend:
            external.add_external(spend.as_external_spend())
    return [*inner.rows(), *external.rows()]


def blended_cost(usage: list[RoleUsage], cost_roles: Collection[UsageRole]) -> float | None:
    """The blended spend of ``usage`` over ``cost_roles`` — or ``None`` when any of it went unpriced.

    The one derivation of :attr:`~threetears.evals.contracts.models.EvalResult.cost_usd` and of a
    judge phase's or a re-judge's cost, from the rows that observed it. **Unpriced is a state,
    never zero**: a model call whose client reported no price — a local model, a client that
    prices nothing — spent something nobody can put a number on, so a total over it is unknown,
    and a sum of the priced rows alone would be read as the whole. That is why a cost cap cannot
    count it and a mean must not average it in (see
    :class:`~threetears.evals.run.budget.EvalRunCostCap` and
    :func:`~threetears.evals.contracts.scoring.compute_cost_summary`).

    **An ``external`` row without dollars contributes nothing here, rather than making the total
    unknown**, because by the time :func:`cell_cost` calls this the only such rows left are volume the
    composition deliberately leaves out: an action-seam metered call is counted unpriced by
    construction (:meth:`RoleUsageLedger.add_external_unpriced`), since pricing it would make a replay
    of it understate production. External spend the composition DOES claim — background work's paid
    calls in a run with declared rates — is held to the unpriced rule by :func:`cell_cost` before the
    rows are summed, because once folded into a row the two cannot be told apart.

    Args:
        usage: The rows to sum.
        cost_roles: The roles the total sums (``EvalResult.cost_roles``).

    Returns:
        The total, rounded to six places, or ``None`` when a model call in those roles went unpriced.
    """
    rows = [row for row in usage if row.role in cost_roles]
    if any(row.cost_usd is None and row.role != "external" for row in rows):
        return None
    return round(sum(row.cost_usd for row in rows if row.cost_usd is not None), 6)


def cell_cost(
    usage: list[RoleUsage], *, async_deliveries: Sequence[AsyncDelivery] | None, rate_table: ExternalRateTable | None
) -> float | None:
    """One cell's ``EvalResult.cost_usd``: its rows' blended spend over the run's cost roles, or ``None``.

    The single derivation the runner makes at every exit of a cell. ``None`` — unknown, never zero —
    when :func:`blended_cost` finds a model call that went unpriced, and also when the cell's
    background work reported paid non-LLM calls (:class:`~threetears.evals.contracts.models.AsyncExternalSpend`)
    that a run with declared rates could not price: such a run's ``cost_roles`` name ``external``,
    so its total claims those calls, and a call whose provider reported no charge and whose
    ``(provider, unit)`` the run holds no rate for is spend the total would otherwise silently leave
    out. A provider-reported charge prices its calls whatever the table holds, and wins over a rate
    it does hold. A run that declared no rates claims no external dollars, so its external calls are
    volume only in this total — a reported charge still lands on its ``external`` row, but the
    composition stamped on the result (``cost_roles``) does not sum that role. A substituted delivery reports no spend at all
    (:class:`~threetears.evals.contracts.models.AsyncDelivery` refuses one that does), so a replayed or
    seeded payload can make the cost neither larger nor unknown.

    Args:
        usage: The cell's rows — the kind's, its background work's (:func:`async_delivery_usage`)
            and the judge's.
        async_deliveries: The cell's background work, as its kind reported it.
        rate_table: The run's rate table, or ``None`` when it declared none.

    Returns:
        The blended spend, or ``None`` when part of it could not be priced.
    """
    cost_roles = blended_cost_roles(rate_table)
    if "external" in cost_roles and rate_table is not None:
        for entry in async_deliveries or ():
            for spend in entry.external_spend:
                if rate_table.money_for(spend.as_external_spend()) is None:
                    return None
    return blended_cost(usage, cost_roles)


def _sum_costs(rows: list[RoleUsage]) -> float | None:
    """Total the observed costs, or ``None`` when no row observed one."""
    observed = [row.cost_usd for row in rows if row.cost_usd is not None]
    return round(sum(observed), 6) if observed else None


def production_replicating_cost(usage: list[RoleUsage], *, substituted_deliveries: int) -> float | None:
    """Observed cost of the roles a production deployment would also pay for.

    Candidate + inner agent + external, excluding the judge and simulator, which exist
    only to measure them.

    **This is the observed subset, not a counterfactual.** It answers "of the dollars this
    cell really spent, which ones belong to production roles" — it does not answer "what
    would running this candidate for real cost", and the two coincide only when the run
    substituted nothing. A run that swept a model, overrode a tool config or a prompt, or
    stripped the candidate's learned memory has moved a knob production does not move, and
    the difference has **no reliable sign**: a cheaper swept model understates, while a
    memory-stripped candidate does background work where production would have answered from
    what it already held, and overstates. That is why the honest first-class counterfactual is separate
    work and this docstring is a qualified claim rather than a correction factor.

    ``None`` — unknown rather than free — in two cases:

    - no production role observed a cost at all;
    - **the result carries a substituted async DELIVERY.** A seeded or replayed delivery means
      the background model's dollars and the metered units were never spent, so the observed
      sum would understate production while claiming to be its subset. Failing toward
      "cheaper than reality" is the dangerous direction for a capacity or pricing decision,
      so the figure is withheld instead.

    **The second case is scoped to the delivery seam, and that scope is correct rather than
    a gap one seam over.** The cassette layer replays at two seams (see the module
    docstring of :mod:`threetears.evals.run.cassette_proxy`): the *delivery* seam (the async
    tools a kind declares), which the case above covers, and the *action* seam
    (the synchronous tools a kind's :class:`~threetears.evals.contracts.cassettes.ActionSeam` declares),
    where :class:`~threetears.evals.run.cassette_proxy.CassetteProxy` re-serves a synchronous
    ``act()`` result. An action-seam replay leaves no ``async_deliveries`` entry, so it
    reports a figure rather than withholding one — and the figure is not understated,
    because **no action-seam call puts DOLLARS in these rows**:

    - The only role a tool call could land dollars in is ``external``. Its delivery-seam
      feeder is :func:`async_delivery_usage`, reading the ``external_spend`` an
      :class:`~threetears.evals.contracts.models.AsyncDelivery` reports, which only background
      work carries.
    - Its action-seam feeder is the host's tool dispatcher, which counts every metered call
      through :meth:`RoleUsageLedger.add_external_unpriced` — calls and provider units, and by
      construction no dollars. That method consults no rate table, so there is no
      configuration in which an action-seam call contributes to this sum.

    So a live capture arm and a replay arm of such a cell report the SAME dollars — the
    candidate's own LLM spend, genuinely paid in both. What the replay withheld from the
    third party is provider UNITS, which this figure never counted in either mode. Those
    units ARE recorded — a synchronous metered tool reports its volume and its metered units
    on ``EvalResult.usage`` — so the count is covered. The dollars are not, deliberately:
    pricing an action-seam call is what would make a replay of it understate production, and
    taking that step means widening the withholding below to a second seam, which changes
    what every run records. The dollar-neutrality is what makes the scoping honest; if an
    action-seam call ever acquires a path to a dollar figure in these rows, the fix is to
    widen the withholding to that seam.

    ``substituted_deliveries`` is **required and deliberately has no default**. The rows
    alone cannot evidence the second case — a substituted delivery leaves no row, which is
    exactly why it is dangerous — so a caller holding only the rows would have no way to
    know they are incomplete. A default of "nothing was substituted" would let every
    existing call site keep compiling and keep returning the understated number silently,
    which is the failure being fixed, re-introduced as a convenience.
    :func:`count_substituted_deliveries` derives it from a result;
    :attr:`ResolvedUsage.substituted_deliveries` carries it on the resolved view.

    Args:
        usage: The per-role rows to sum.
        substituted_deliveries: How many of this result's async deliveries were supplied
            by a harness rather than produced by a background run. Any non-zero value
            withholds the figure.

    Returns:
        The observed production-role dollars, or ``None`` when the figure is unknown
        rather than free.
    """
    if substituted_deliveries:
        return None
    return _sum_costs([row for row in usage if row.role in PRODUCTION_REPLICATING_ROLES])


def program_cost(usage: list[RoleUsage]) -> float | None:
    """Cost of every role, i.e. what the eval program spent to produce this result.

    ``None`` when nothing observed a cost. This is a FLOOR, not a reconciliation: a
    requeued background delivery drops its cost, and the external role reports dollars only
    on a run whose operator declared a credit rate for its metered calls, so
    ``EvalResult.cost_usd`` remains the authoritative blended figure.
    """
    return _sum_costs(list(usage))


class ResolvedUsage(EvalBaseModel):
    """The per-role usage a read surface should show for one result, and how it got there.

    Returned as a value and never assigned back onto the result, for the same reason
    :class:`~threetears.evals.contracts.identity.DerivedContextIdentity` is: this is a *reading* of the
    record — it interprets an empty list against the recorded ``termination`` — and writing
    an interpretation into the field reserved for observations would make the two
    indistinguishable, turning any later save of that document into a backfill. There is
    deliberately no code path that persists these rows.

    ``source`` is the field a reader must not skip:

    - ``captured`` — the roles were observed at their own call sites while the cell ran.
    - ``lost`` — the cell was cut off before it returned: on its deadline, by its run's cancel,
      or by an apparatus fault (:data:`CUT_SHORT_TERMINATIONS`). ``usage`` holds the rows it had
      reported up to then (the run loop owns the cell's sink, so they survive the cut), and
      ``partial`` is ``True`` because the call in flight when it was cut never reported its
      usage: roles spent more than these rows say. This arm exists because
      the stored rows cannot carry the distinction — a cell cut off during its only call
      stores ``[]``, which means "capture ran and attributed nothing" on a candidate-factory
      failure — so the two are separated by the runner's recorded ``termination`` rather
      than by the shape of what is left.

    There is no third arm: every result carries ``usage`` (the runner writes a list at every
    exit and the field is required), so a result never lacks a per-role observation, and
    nothing reconstructs spend from a result's persisted trace.
    """

    usage: list[RoleUsage]
    source: Literal["captured", "lost"]
    #: A projection of ``source``, not a second signal: it is exactly
    #: ``source != "captured"`` on both arms. Kept because a reader-facing "do not
    #: read this as whole" flag is worth stating outright, but never branch on it where
    #: you mean ``source`` — the two cannot disagree, and code that treats them as
    #: independent invents a fourth state the model has no arm for.
    partial: bool
    #: Async deliveries whose payload a harness supplied instead of a background run
    #: producing it — a seeded case finding or a replayed capture. Non-zero means the rows
    #: on this view did not spend what production would have, so
    #: :func:`production_replicating_cost` declines to report a figure at all. Derived from
    #: the deliveries themselves on every arm rather than stored beside them, so a new
    #: substitution source cannot forget to declare itself.
    substituted_deliveries: int = 0


def count_substituted_deliveries(result: EvalResult) -> int:
    """Count the async deliveries in ``result`` whose payload a harness supplied.

    Read from the result's own ``async_deliveries`` record rather than from the per-turn trace,
    because that is the representation that **survives an analytic read**: the guard's two
    aggregate consumers — :func:`~threetears.evals.analysis.reporting._frontier_point` and
    :func:`~threetears.evals.contracts.scoring.compute_cost_summary`, which is what ``run_summary``
    reports — read results that carry no trace at all, since it lives in a sibling
    :class:`~threetears.evals.contracts.models.EvalTrace` document. A count read off the trace there
    would be 0 on a result that *had* substituted, indistinguishable from "nothing was
    substituted", and the guard would publish a candidate-only partial sum as a
    production-replicating cost — the understated number :func:`production_replicating_cost`
    exists to withhold.

    **Every background tool reports into the one record.** :class:`~threetears.evals.contracts.models.AsyncDelivery`
    is engine vocabulary and ``substituted`` is required on it, so a second tool that substitutes
    deliveries is counted here without a change to this function, and a kind cannot leave the arm
    unstated.

    **Not every cost surface passes through this guard, and the ones that do not are still
    wrong for the same underlying reason.** Export, pivot and history project
    ``PROJECTED_METRICS`` off raw ``result.cost_usd``, and :func:`estimate_cost` pools that
    same blended field — none of them reach production-replicating cost at all, so a result
    that replayed a DELIVERY understates them and this function is not what would fix it.
    That half is open; do not read this docstring as saying it is closed.

    **The cassette layer's ACTION seam is outside this count, deliberately.** A replayed
    synchronous tool call re-serves an ``act()`` result and produces no delivery at all, so it
    is not a substituted *delivery* and this function rightly reports 0 for it — see
    :func:`production_replicating_cost` for why that leaves the figure honest.

    Args:
        result: The result to inspect. Never mutated.

    Returns:
        How many deliveries a harness supplied. Any non-zero count is enough to withhold the
        production-replicating cost; the number is kept rather than a bool because it is the
        more informative disclosure at no extra cost.
    """
    return count_substituted(result.async_deliveries)


def count_substituted(deliveries: Sequence[AsyncDelivery] | None) -> int:
    """Count the deliveries a harness supplied, over the delivery record itself.

    The one predicate :func:`count_substituted_deliveries` reads a finished result through, for the
    caller that holds the record before any result exists: a kind computing its own measures from the
    deliveries its cell collected, which has no :class:`~threetears.evals.contracts.models.EvalResult`
    to hand over (and would otherwise build a partial one just to ask).

    Args:
        deliveries: The cell's async-delivery record — ``EvalResult.async_deliveries``, or a candidate
            output's ``async_deliveries`` — or ``None`` when nothing watched for background work.

    Returns:
        How many entries a harness supplied; 0 when nothing was watched.
    """
    if deliveries is None:
        # Nothing watched for background work — the candidate kind starts none — so there was no
        # delivery the harness could have substituted.
        return 0
    return sum(1 for entry in deliveries if entry.substituted)


#: The terminations of a cell cut off before it returned, whose usage therefore stops short of the
#: work in flight when it was cut — read by :func:`resolve_result_usage` to mark them partial.
CUT_SHORT_TERMINATIONS: frozenset[CellTermination] = frozenset({"cell_timeout", "cancelled", "apparatus_failed"})


def resolve_result_usage(result: EvalResult) -> ResolvedUsage:
    """Return the per-role usage a read surface should show for ``result``.

    The single place a read surface decides what a result's per-role spend *is*, so the
    surfaces answer identically rather than each inventing a rule. Writes nothing on any
    path.

    Every surface that shows one result's per-role spend calls it, so that no two surfaces
    can disagree about whether a figure is whole; a surface that discloses provenance must
    call this rather than inventing its own rule.

    A cell cut off before it returned is the one case the persisted value cannot answer on its
    own — by its deadline, by its run's cancel, or by an apparatus fault unwinding the kind
    (:data:`CUT_SHORT_TERMINATIONS`). Its ``usage`` holds what it had reported up to then —
    possibly ``[]``, byte-identical to a candidate-factory failure's honest "capture ran and
    attributed no roles" — and never the call in flight when it was cut, which reported nothing.
    Reading ``usage`` alone would publish ``source="captured", partial=False`` for such a cell,
    telling an operator that something which burned past its deadline consumed nothing anywhere.
    It reads the runner's recorded ``termination``, which is the branch that cut the capture
    short rather than a consequence of it.

    Args:
        result: The result to resolve.

    Returns:
        The captured rows — including the empty list, which means capture ran and attributed
        no roles — or the ``lost`` view (the rows kept, marked partial) for a cut-short cell.
    """
    # Substitution is a property of the deliveries, not of how the rows were obtained, so
    # it is read once here and attached to whichever arm answers — including `captured`,
    # which is the live-run arm and the one a seeded run takes.
    substituted = count_substituted_deliveries(result)
    if result.termination in CUT_SHORT_TERMINATIONS:
        return ResolvedUsage(usage=list(result.usage), source="lost", partial=True, substituted_deliveries=substituted)
    return ResolvedUsage(usage=list(result.usage), source="captured", partial=False, substituted_deliveries=substituted)


__all__ = [
    "BLENDED_COST_ROLES",
    "CUT_SHORT_TERMINATIONS",
    "PRODUCTION_REPLICATING_ROLES",
    "CallUsage",
    "ExternalRateTable",
    "ExternalSpend",
    "ResolvedUsage",
    "RoleUsageLedger",
    "async_delivery_usage",
    "blended_cost",
    "blended_cost_roles",
    "cell_cost",
    "count_substituted",
    "count_substituted_deliveries",
    "production_replicating_cost",
    "program_cost",
    "resolve_result_usage",
]
