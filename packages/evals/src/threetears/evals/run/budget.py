"""Per-run eval cost cap.

Eval bake-offs and campaigns spend real LLM budget. Cost was tracked per result
(``compute_cost_summary``) but nothing enforced a ceiling — a single run, or a
wedged candidate looping tool calls, could spend without limit. This module caps
ONE run:

- :class:`EvalRunCostCap` accumulates THIS run's spend in memory from each
  delivered result's ``cost_usd`` and trips once the accumulated total exceeds
  the configured (or per-run overridden) ceiling — or, in a run it enforces, on the
  first result whose spend could not be priced (see :class:`EvalRunCostCap`).
- Between cells, :func:`~threetears.evals.run.runner.execute_run` checks the cap and
  stops GRACEFULLY when it trips — already-delivered results stay persisted and
  the run is marked ``budget_stopped`` (a distinct, honest terminal status, not
  an infra ``failed``) by raising :class:`BudgetStoppedError`, which the job
  manager translates.
- Inside a cell, a kind that makes many paid calls of its own asks the same cap through its sink
  before each further one, counting what the cell has spent so far
  (:meth:`~threetears.evals.contracts.candidate_kind.CellSink.cost_cap_reached`): a conversation's
  simulator does (:func:`~threetears.evals.run.conversation.drive_conversation`). A cell stopped
  that way is excluded, and the run stops ``budget_stopped`` once it is saved.
- :class:`AccountExhaustedError` is the account-side twin: the provider account behind the
  run refused a candidate call (out of credit, or the key refused), so the run stops after
  that cell and is marked ``exhausted``. Nothing configures it; a cell observes it.

The cap is pure per-run arithmetic: no shared budget pool, no periods, no
BudgetService, no database I/O. One run can never exhaust another, raising the
ceiling takes effect on the very next run, and the cap check has no fail-open
surface (there is nothing to fail). Central-ledger recording of eval spend is
deliberately NOT wired here — it rides the campaign primitive, which
owns the async, off-loop path into the budget ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from threetears.evals.run import ceilings
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.models import CostCapOrigin

log = get_logger(__name__)


@dataclass(frozen=True)
class CapBreach:
    """The numbers that justify a graceful budget stop.

    Carried out of :meth:`EvalRunCostCap.check` so the stop can be reported as a
    measurement — this much was spent against this ceiling — rather than as the
    bare assertion "the cap was exceeded", which names neither the limit an
    operator set nor the spend that crossed it.

    Attributes:
        max_cost_usd: The ceiling that bound the run.
        accumulated_usd: This run's PRICED spend at the moment the cap tripped — a running cell's
            pending spend included, when a cell asked.
        unpriced_results: Results whose spend could not be priced, counting a running cell's unpriced
            pending spend as one. Non-zero is a breach on its own: the cap cannot say the run is inside
            a ceiling it cannot count against.
    """

    max_cost_usd: float
    accumulated_usd: float
    unpriced_results: int


class BudgetStoppedError(Exception):
    """Raised by :func:`~threetears.evals.run.runner.execute_run` when the cap trips mid-run.

    Signals that this run's accumulated cost exceeded its cap between cells and
    the run stopped gracefully. Already-delivered results are persisted *before*
    this is raised (each cell saves its result as it completes), so nothing
    produced is lost. The job manager catches it and marks the run
    ``budget_stopped`` — a distinct, honest terminal status, never an infra
    ``failed``.
    """

    def __init__(self, completed: int, total: int, breach: CapBreach):
        """Record how far the run got before the cap stopped it, and on what numbers.

        The breach is required rather than optional: this error exists only
        because a cap tripped, so an instance that cannot say which ceiling and
        which spend is one an operator has to go and reconstruct — which is the
        state the message used to leave them in.

        Args:
            completed: Number of results delivered (and persisted) before the stop.
            total: Total results the run's matrix would have produced.
            breach: The ceiling and the accumulated spend that crossed it.
        """
        self.completed = completed
        self.total = total
        self.breach = breach
        if breach.unpriced_results:
            reason = (
                f"{breach.unpriced_results} result(s) spent what could not be priced, beside "
                f"${breach.accumulated_usd:.4f} of priced spend, so a ${breach.max_cost_usd:.4f} cap cannot be enforced"
            )
        else:
            reason = (
                f"eval run cost cap exceeded — ${breach.accumulated_usd:.4f} spent against a "
                f"${breach.max_cost_usd:.4f} cap"
            )
        super().__init__(f"{reason} after {completed}/{total} results — run stopped gracefully")


class AccountExhaustedError(Exception):
    """Raised by :func:`~threetears.evals.run.runner.execute_run` when the run's paying account refuses a call.

    The account-side twin of :class:`BudgetStoppedError`. That one is this run's own ceiling
    doing its configured job; this is the provider account behind the run refusing a candidate,
    simulator or judge call — out of credit (402), or the key refused (401/403) — which a cell
    classifies as an apparatus fault and excludes. Every model behind the
    same key gets the same answer, so every later cell would be refused alike: the run stops
    after the cell that met the wall instead of launching the rest to be excluded one by one.

    Already-delivered results are persisted before this is raised, the refused cell's included.
    The job manager marks the run ``exhausted`` — a distinct terminal status, so an exhausted
    account never reads as a broken harness (``failed``) — and files this message under
    ``error_details``, unlike the two designed stops: nobody configured the account to run dry,
    and an operator has to act on it before the next launch can measure anything.

    Attributes:
        completed: Results delivered (and persisted) before the stop, the refused cell included.
        total: Results the run's matrix would have produced.
        accumulated_usd: What those delivered results cost, summed from what the loop recorded
            — carried out so the stop reports the spend the run had already incurred. The priced
            results' spend only.
        unpriced_results: Delivered results whose spend could not be priced, and so is not in
            ``accumulated_usd``.
        detail: The refused cell's own apparatus message, naming the account fault's kind.
    """

    def __init__(self, completed: int, total: int, *, accumulated_usd: float, unpriced_results: int, detail: str):
        """Record how far the run got, what it had spent, and the refusal that stopped it.

        Args:
            completed: Results delivered (and persisted) before the stop.
            total: Results the run's matrix would have produced.
            accumulated_usd: The delivered priced results' summed cost.
            unpriced_results: Delivered results whose spend could not be priced.
            detail: The refused cell's apparatus message.
        """
        self.completed = completed
        self.total = total
        self.accumulated_usd = accumulated_usd
        self.unpriced_results = unpriced_results
        self.detail = detail
        unpriced = f", and {unpriced_results} result(s) whose spend could not be priced" if unpriced_results else ""
        super().__init__(
            f"eval run stopped: the provider account refused a candidate call after {completed}/{total} results "
            f"(${accumulated_usd:.4f} spent{unpriced}) — {detail}"
        )


class EvalRunCostCap:
    """In-memory per-run cost cap for a single eval run.

    Accumulates THIS run's spend from each delivered result's ``cost_usd`` and
    trips (:attr:`exceeded`) once the running total exceeds :attr:`max_cost_usd`.
    Pure arithmetic — no BudgetService, no shared pool, no database I/O — so it
    bounds one run and one run only: a runaway sweep stops itself without
    starving (or being starved by) any other run, and the cap check can never
    fail open because there is nothing external to fail.

    Disabled (``enabled=False``, the enforcement toggle off) makes
    :attr:`exceeded` always ``False`` — the run is uncapped. Spend is still
    accumulated (harmless) so :attr:`accumulated_usd` stays meaningful for
    observability.

    **Unpriced spend stops an enforcing run at the first result that carries it.** A result
    whose ``cost_usd`` is ``None`` spent something nobody priced — a local model, a client that
    reports no price — and counting it as zero would let a capped run spend without bound while
    the cap reads it as inside the ceiling. The cap cannot refuse such a run before it starts:
    the completion-client port declares no pricing capability, and a client may price one call
    and not the next, so the first unpriced result is the earliest the fact is observable. A run
    that is capped stops there (``budget_stopped``, its reason naming the unpriced results); an
    uncapped one carries on and counts them.
    """

    def __init__(self, run_id: str, max_cost_usd: float, *, enabled: bool):
        """Bind the cap to a run id and its ceiling.

        Args:
            run_id: The eval run this cap bounds — surfaced in the missing-cost
                WARNING so a data gap is attributable to its run.
            max_cost_usd: The per-run ceiling; the run stops once accumulated
                spend exceeds it.
            enabled: Whether the cap enforces. ``False`` makes :attr:`exceeded`
                always ``False`` (uncapped). Required rather than defaulted:
                only the caller knows whether its host has eval enforcement
                switched on, and a cap that silently stops enforcing is worse
                than one that fails to construct.
        """
        self._run_id = run_id
        self._max_cost_usd = max_cost_usd
        self._enabled = enabled
        self._accumulated_usd = 0.0
        self._unpriced_results = 0

    @classmethod
    def for_run(
        cls, run_id: str, max_cost_usd: float | None, *, configured_max_cost_usd: float, enforcement_enabled: bool
    ) -> EvalRunCostCap:
        """Build the cap that bounds one run, from values the caller resolved.

        The one construction path in production, and the reason it exists rather than leaving the
        host to assemble it: the two ceilings this package bounds a run with take
        their constructor arguments in **mirror-opposite** forms — this class takes
        the ceiling regardless of enforcement plus a separate ``enabled``, while
        :class:`~threetears.evals.run.metering.MeteredCallLedger` takes the *effective*
        ceiling and reads ``None`` as unbounded. A host guessing by symmetry gets a
        cap with no ceiling at all, or a ledger that enforces in a run whose operator
        switched enforcement off. Which resolver each wants is the engine's knowledge,
        so the engine keeps it.

        Args:
            run_id: The eval run this cap bounds.
            max_cost_usd: Optional per-run override (validated ``> 0`` at the service
                boundary). ``None`` inherits the default below.
            configured_max_cost_usd: The default ceiling, as a value.
            enforcement_enabled: Whether the caller enforces eval ceilings at all.

        Returns:
            A cap bound to this run's ceiling, enforcing exactly when the caller said to.
        """
        return cls(
            run_id,
            ceilings.resolve_ceiling(max_cost_usd, configured=configured_max_cost_usd),
            enabled=enforcement_enabled,
        )

    @staticmethod
    def resolve_effective_ceiling(
        max_cost_usd: float | None, *, configured_max_cost_usd: float, enforcement_enabled: bool
    ) -> float | None:
        """Return the ceiling a run is actually bounded by, or ``None`` if uncapped.

        This class's half of :func:`threetears.evals.run.ceilings.resolve_effective_ceiling`, which
        holds the cascade and the reasoning. What is local to the cap is which of the two
        answers goes where: **this** is what the run RECORDS, while the cap itself is built
        through :meth:`for_run` from the unfiltered ceiling plus a separate ``enabled`` — so
        an unenforced run stores no ceiling while still carrying a cap object.

        Args:
            max_cost_usd: Optional per-run override, as passed to
                :meth:`for_run`.
            configured_max_cost_usd: The default ceiling, as a value.
            enforcement_enabled: Whether the caller enforces eval ceilings at
                all.

        Returns:
            The effective ceiling, or ``None`` when cost enforcement is
            disabled and the run is therefore uncapped.
        """
        return ceilings.resolve_effective_ceiling(
            max_cost_usd, configured=configured_max_cost_usd, enforcement_enabled=enforcement_enabled
        )

    @staticmethod
    def resolve_ceiling_origin(max_cost_usd: float | None, *, enforcement_enabled: bool) -> CostCapOrigin:
        """Return which tier of the cascade supplied the ceiling a run is bounded by.

        The companion of :meth:`resolve_effective_ceiling`, answering from the same launch and
        recorded beside the number it explains. Delegates to
        :func:`threetears.evals.run.ceilings.resolve_ceiling_origin`, which is the only place a
        :data:`~threetears.evals.contracts.models.CostCapOrigin` is produced — so a change to that
        vocabulary cannot reach the cost cap and miss the metered-call ledger.

        Args:
            max_cost_usd: Optional per-run override, as passed to :meth:`for_run`.
            enforcement_enabled: Whether the caller enforces eval ceilings at all.

        Returns:
            A :data:`~threetears.evals.contracts.models.CostCapOrigin` value: ``"uncapped"`` when
            enforcement is off (no ceiling bound the run, whatever the cascade would
            have resolved), else ``"chosen"`` for a launch-supplied ceiling and
            ``"inherited"`` for the configured default.
        """
        return ceilings.resolve_ceiling_origin(max_cost_usd, enforcement_enabled=enforcement_enabled)

    @property
    def accumulated_usd(self) -> float:
        """Total priced spend recorded for this run so far."""
        return self._accumulated_usd

    @property
    def unpriced_results(self) -> int:
        """Results recorded for this run whose spend could not be priced."""
        return self._unpriced_results

    @property
    def max_cost_usd(self) -> float:
        """The ceiling this run is bounded by."""
        return self._max_cost_usd

    @property
    def exceeded(self) -> bool:
        """True once enforcement is on and priced spend exceeds the cap, or any spend went unpriced."""
        return self._enabled and (self._accumulated_usd > self._max_cost_usd or self._unpriced_results > 0)

    def check(self, pending_usd: float | None = 0.0) -> CapBreach | None:
        """Report whether the run may proceed, and on what numbers if it may not.

        The gate :func:`~threetears.evals.run.runner.execute_run` calls before each cell, with nothing
        pending, and the one a running cell asks through its sink
        (:meth:`~threetears.evals.contracts.candidate_kind.CellSink.cost_cap_reached`) with the spend it
        has made that its result has not yet reported — a conversation's simulator calls
        (:func:`~threetears.evals.run.conversation.drive_conversation`). One rule for both, so the
        cap a cell is stopped by inside is the cap the run is stopped by between cells. Nothing is
        recorded here: a cell's whole spend is recorded once, by :meth:`record`, when its result lands.
        It returns the observation rather than a bare boolean because the run loop is where the stop is
        raised and the cap is the only thing that knows the ceiling and the spend — a ``False`` there
        reaches the operator as an unattributed assertion.

        Args:
            pending_usd: Spend a running cell has made that is not yet recorded; ``0.0`` between
                cells. ``None`` is pending spend that could not be priced, which breaches an enforcing
                cap as an unpriced result does.

        Returns:
            ``None`` while the run is within its cap counting that spend (including when enforcement
            is disabled, which is uncapped), or the :class:`CapBreach` that stops it.
        """
        if not self._enabled:
            return None
        accumulated = self._accumulated_usd + (pending_usd or 0.0)
        unpriced = self._unpriced_results + (1 if pending_usd is None else 0)
        if accumulated <= self._max_cost_usd and not unpriced:
            return None
        return CapBreach(max_cost_usd=self._max_cost_usd, accumulated_usd=accumulated, unpriced_results=unpriced)

    def record(self, cost_usd: float | None) -> None:
        """Add one delivered result's cost to this run's running total.

        An unpriced cost (``None``) is counted as unpriced, never as 0 — in an enforcing run it
        trips the cap on the next check — and logged at WARNING with the run id. Pure in-memory
        arithmetic; no budget-layer or database call, so there is no fail-open surface.

        Args:
            cost_usd: The result's ``cost_usd``, ``None`` when its spend could not be priced.
        """
        if cost_usd is None:
            self._unpriced_results += 1
            log.warning(
                "Eval run %s: a delivered result's spend could not be priced — %s",
                self._run_id,
                f"a ${self._max_cost_usd:.2f} cap cannot be enforced on it, so the run stops before its next cell"
                if self._enabled
                else "the run is uncapped, so it carries on and counts it",
            )
            return
        self._accumulated_usd += cost_usd


__all__ = [
    "AccountExhaustedError",
    "BudgetStoppedError",
    "CapBreach",
    "EvalRunCostCap",
]
