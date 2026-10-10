"""Per-run ceiling on the third-party calls an eval candidate may make.

``max_cost_usd`` (:mod:`threetears.evals.run.budget`) bounds LLM dollars and nothing else.
A tool action that spends a *provider* quota — search-API credits, a billed image
generation, a rationed third-party API key — costs the run nothing the cost cap
can see, so before this module nothing bounded it at all: a search tool spent
credits and touched no ledger, and an image tool billed a paid provider and touched
no ledger.

Three pieces, and they are deliberately separate:

- **The declaration** lives on the host's tool, beside its actuation declaration
  and for the same reason: a hand-maintained list of tool names in the eval layer
  drifts, and an action nobody classified fails closed to metered.
- **The count** happens at the host's action dispatcher — the one
  seam every tool call passes through, which a subclass ``act`` override cannot
  route around.
- **The bound** is this module: one :class:`MeteredCallLedger` per run,
  in-memory, pure arithmetic, no shared pool and no database I/O.

**It refuses calls; it does not stop the run.** A hard stop would discard a
partially-measured matrix, where a refusal keeps every cell measurable and records
the refusal count on ``metered_calls_refused``. The refusal reaches the candidate as an ordinary failed action
result — the action seam's own feedback channel, the same way the ``tools_allowed``
bound reaches it as a system perception on the attach path — and the run and its
affected results carry a disclosure, so a truncated measurement can never be
mistaken for a complete one.

**What it bounds, exactly:** action-seam calls. An async tool whose quota is spent
on a detached background run declares its
metering as per-delivery; its spend is counted where
it is reported (``AsyncDelivery.external_spend``, folded by
``usage_capture.async_delivery_usage``) and it is bounded by its own
tool config (``max_rounds``, ``max_search_calls``). Counting its dispatch here would
double-count what the delivery already reported and state a volume nobody measured.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from threetears.evals.contracts.host.spend import ExternalSpend
from threetears.evals.contracts.provider import sum_optional_tokens
from threetears.evals.run import ceilings
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.models import MeteredCallOrigin

log = get_logger(__name__)


@dataclass(frozen=True)
class MeteredCallTally:
    """What a run (or one cell's slice of it) metered, kept per provider.

    ``calls`` and ``refused`` are scalars because calls ARE summable across providers — a
    call is a call whoever served it, and the ceiling counts calls. **Weighted units are
    not**, so they live per ``(provider, unit)`` in :attr:`spends` and are never added into
    one number: one provider's credits and another's quota units are not one quantity, and
    a figure that adds them is not a smaller truth but a fabricated one.

    Units follow the missing-is-not-zero rule the per-role usage rows do: a provider that
    publishes no countable unit contributes calls and no units, which is a different fact
    from a provider reporting zero. So a tally can legitimately read "3 calls, 2 credits".

    Attributes:
        calls: Metered calls admitted and dispatched, across every provider.
        spends: One entry per ``(provider, unit)`` this tally observed, in first-seen order.
        refused: Calls the ceiling turned away, across every provider.
    """

    calls: int
    spends: tuple[ExternalSpend, ...]
    refused: int

    def delta_from(self, baseline: MeteredCallTally) -> MeteredCallTally:
        """Return what accumulated since ``baseline`` was taken.

        A before/after pair of run-level tallies is exactly one cell's contribution only while nothing
        else of the run calls between the two — a run executing its cells serially. A run executing them
        concurrently attributes each call to its cell as it is admitted instead
        (:meth:`MeteredCallLedger.cell`), which :func:`~threetears.evals.run.runner.metered_cell_tally`
        prefers. Subtracting rather than draining keeps the run total intact for the run-level disclosure
        — a drain would leave the ledger unable to say what the whole run did.

        Args:
            baseline: A tally taken from the same ledger at an earlier moment.

        Returns:
            The difference. A spend's ``provider_units`` is ``None`` whenever THIS cell
            observed no new units at that provider, preserving unknown-vs-zero.

            The unchanged-total case is the one that matters and it is not the
            never-observed case: a run whose first cell searched (credits counted)
            and whose second called only providers that publish no countable unit
            — ``image_gen``, ``weather`` — leaves the running total standing, and
            subtracting would hand that second cell ``units=0``. Zero is a claim:
            it says those calls consumed no provider units, and an operator
            comparing per-cell external rows would read them as free. Nothing was
            observed, so nothing is what this reports.

            Differencing is **per ``(provider, unit)``**, so a cell that spent at one
            provider does not appear to have zeroed another's. A provider absent from this
            tally contributes nothing rather than a negative, and a provider whose entry is
            unchanged in both calls and units drops out entirely — the cell did nothing
            there, and an entry saying so would read as an observation.
        """
        prior_by_key = {(sp.provider, sp.provider_unit): sp for sp in baseline.spends}
        deltas: list[ExternalSpend] = []
        for spend in self.spends:
            prior = prior_by_key.get((spend.provider, spend.provider_unit))
            prior_units = (prior.provider_units or 0) if prior is not None else 0
            prior_calls = prior.calls if prior is not None else 0
            if spend.provider_units is None or spend.provider_units == prior_units:
                units: int | None = None
            else:
                units = spend.provider_units - prior_units
            delta_calls = spend.calls - prior_calls
            if delta_calls == 0 and units is None:
                continue
            deltas.append(
                ExternalSpend(
                    provider=spend.provider,
                    calls=delta_calls,
                    provider_units=units,
                    provider_unit=spend.provider_unit if units is not None else None,
                )
            )
        return MeteredCallTally(
            calls=self.calls - baseline.calls,
            spends=tuple(deltas),
            refused=self.refused - baseline.refused,
        )


def _folded(spends: dict[tuple[str, str | None], ExternalSpend], spend: ExternalSpend | None) -> None:
    """Fold one call's reported consumption into per-``(provider, unit)`` totals, in place.

    Accumulates per ``(provider, unit)`` rather than into one running number, because two providers'
    weighted units are not one quantity. A report with no provider contributes nothing here — the call
    still happened and is counted by the caller, but nothing can be attributed.

    Args:
        spends: The totals to fold into.
        spend: The calling tool's consumption report for one call, or ``None``.
    """
    if spend is None or spend.provider is None:
        return
    key = (spend.provider, spend.provider_unit)
    prior = spends.get(key)
    spends[key] = ExternalSpend(
        provider=spend.provider,
        calls=(prior.calls if prior else 0) + spend.calls,
        provider_units=sum_optional_tokens(prior.provider_units if prior else None, spend.provider_units),
        provider_unit=spend.provider_unit,
    )


class CellMeter:
    """One cell's own slice of its run's metered calls, counted as each call is admitted or refused.

    Opened by :meth:`MeteredCallLedger.cell` around one cell. Every call the ledger decides while the
    meter is open IN THAT CELL'S CONTEXT lands here as well as in the run's totals — the context is the
    cell's task and every task it starts, so a cell running beside others is credited only its own calls.
    Units follow the run tally's rule: a provider that publishes none contributes calls and no units.
    """

    def __init__(self) -> None:
        """Start the cell at nothing metered."""
        self._calls = 0
        self._refused = 0
        self._spends: dict[tuple[str, str | None], ExternalSpend] = {}

    def tally(self) -> MeteredCallTally:
        """What this cell has metered so far."""
        return MeteredCallTally(calls=self._calls, spends=tuple(self._spends.values()), refused=self._refused)

    def admitted(self, spend: ExternalSpend | None) -> None:
        """Count one call the run's ceiling let through, with what it consumed."""
        self._calls += 1
        _folded(self._spends, spend)

    def refused(self) -> None:
        """Count one call the run's ceiling turned away."""
        self._refused += 1


class MeteredCallLedger:
    """In-memory per-run ceiling on metered third-party calls.

    One instance per eval run, shared by every cell — serial or concurrent: :meth:`admit` never yields to
    the event loop, so the check and the count are one step and no two cells can both take the last
    call under the ceiling. Each cell's own slice is counted as calls are decided (:meth:`cell`). Read at
    the host's action dispatcher through the subject it was handed to. Pure arithmetic: no provider
    call, no database I/O, nothing that can fail open.

    ``ceiling=None`` counts without bounding — the shape a run gets when eval
    enforcement is switched off, and the shape every non-eval caller gets by
    never having a ledger at all.

    ``none_declared=True`` is a host that declared it has no metered tools: the ceiling is ``0``,
    so every metered call is refused and counted, and each one is logged as the contradiction of
    the host's declaration it is rather than as a run reaching its limit.
    """

    def __init__(self, run_id: str, ceiling: int | None, *, none_declared: bool = False):
        """Bind the ledger to a run and its ceiling.

        Args:
            run_id: The eval run this bounds — named in the log line when the
                ceiling first binds, so a refusal is attributable to its run.
            ceiling: The maximum number of metered calls this run may make, or
                ``None`` for counted-but-unbounded.
            none_declared: The host declared it has no metered tools; requires a ``ceiling`` of 0.

        Raises:
            ValueError: ``none_declared`` with a ceiling other than 0.
        """
        if none_declared and ceiling != 0:
            raise ValueError(f"a host declaring no metered tools allows no metered call; got ceiling={ceiling!r}")
        self._none_declared = none_declared
        self._run_id = run_id
        self._ceiling = ceiling
        self._calls = 0
        self._spends: dict[tuple[str, str | None], ExternalSpend] = {}
        self._refused = 0
        self._announced = False

    @classmethod
    def for_run(
        cls,
        run_id: str,
        max_metered_calls: int | None,
        *,
        configured_max_metered_calls: int | None,
        enforcement_enabled: bool,
    ) -> MeteredCallLedger:
        """Build the ledger that bounds one run, from values the caller resolved.

        The one construction path in production, and the twin of
        :meth:`~threetears.evals.run.budget.EvalRunCostCap.for_run` for the reason stated
        there: the two ceilings take their constructor arguments in mirror-opposite
        forms, and a host guessing by symmetry builds a ledger that enforces in a run
        whose operator switched enforcement off. Which resolver each wants is the
        engine's knowledge, so the engine keeps it.

        Args:
            run_id: The eval run this bounds.
            max_metered_calls: Optional per-run override (validated ``> 0`` at the
                service boundary). ``None`` inherits the default below.
            configured_max_metered_calls: The default ceiling, as a value, or ``None`` for a host that
                declares it has no metered tools.
            enforcement_enabled: Whether the caller enforces eval ceilings at all.

        Returns:
            A ledger bound to this run's effective ceiling, counting without bounding
            when the caller has enforcement off, and refusing every metered call for a host
            that declared none.

        Raises:
            ValueError: An override for a host that declares no metered tools.
        """
        return cls(
            run_id,
            cls.resolve_effective_ceiling(
                max_metered_calls,
                configured_max_metered_calls=configured_max_metered_calls,
                enforcement_enabled=enforcement_enabled,
            ),
            none_declared=configured_max_metered_calls is None,
        )

    @staticmethod
    def resolve_effective_ceiling(
        max_metered_calls: int | None, *, configured_max_metered_calls: int | None, enforcement_enabled: bool
    ) -> int | None:
        """Return the ceiling a run is actually bounded by, or ``None`` if unbounded.

        This class's half of :func:`threetears.evals.run.ceilings.resolve_effective_ceiling`, which
        holds the cascade and the reasoning. What is local to the ledger is that one answer
        serves both purposes: this is what the run RECORDS **and** what :meth:`for_run` builds
        the ledger with, which is the mirror of the cost cap, where the two differ.

        A host that declares no metered tools (``configured_max_metered_calls=None``) gets ``0``
        whatever its enforcement: the declaration is a fact about its tools, not a ceiling it may
        switch off, and a metered call on it contradicts the declaration.

        Args:
            max_metered_calls: Optional per-run override.
            configured_max_metered_calls: The default ceiling, as a value, or ``None`` for a host that
                declares no metered tools.
            enforcement_enabled: Whether the caller enforces eval ceilings at all.

        Returns:
            The effective ceiling, ``0`` for a host declaring no metered tools, or ``None`` when
            eval enforcement is disabled.

        Raises:
            ValueError: An override for a host that declares no metered tools — it would bound nothing.
        """
        if configured_max_metered_calls is None:
            if max_metered_calls is not None:
                raise ValueError(
                    f"max_metered_calls={max_metered_calls} names a ceiling for a host that declares no metered tools; "
                    "nothing would be bounded by it"
                )
            return 0
        return ceilings.resolve_effective_ceiling(
            max_metered_calls, configured=configured_max_metered_calls, enforcement_enabled=enforcement_enabled
        )

    @staticmethod
    def resolve_ceiling_origin(
        max_metered_calls: int | None, *, configured_max_metered_calls: int | None, enforcement_enabled: bool
    ) -> MeteredCallOrigin:
        """Return which tier of the cascade supplied the ceiling a run is bounded by.

        The companion of :meth:`resolve_effective_ceiling`, answering from the same launch and
        recorded beside the number it explains. Delegates to
        :func:`threetears.evals.run.ceilings.resolve_ceiling_origin`, which is the only place a
        :data:`~threetears.evals.contracts.models.CostCapOrigin` is produced — so a change to that
        vocabulary cannot reach the metered-call ledger and miss the cost cap.

        Args:
            max_metered_calls: Optional per-run override.
            configured_max_metered_calls: The default ceiling, or ``None`` for a host that declares no
                metered tools.
            enforcement_enabled: Whether the caller enforces eval ceilings at all.

        Returns:
            A :data:`~threetears.evals.contracts.models.MeteredCallOrigin` value: ``"none_declared"``
            for a host declaring no metered tools, else ``"uncapped"`` when enforcement is off, else
            ``"chosen"`` or ``"inherited"``.
        """
        if configured_max_metered_calls is None:
            return "none_declared"
        return ceilings.resolve_ceiling_origin(max_metered_calls, enforcement_enabled=enforcement_enabled)

    @property
    def ceiling(self) -> int | None:
        """The ceiling this run is bounded by, or ``None`` when unbounded."""
        return self._ceiling

    def tally(self) -> MeteredCallTally:
        """Snapshot what this run has metered so far."""
        return MeteredCallTally(calls=self._calls, spends=tuple(self._spends.values()), refused=self._refused)

    @contextmanager
    def cell(self) -> Iterator[CellMeter]:
        """Meter one cell's own calls for as long as the block runs, in this context.

        The runner opens one around every cell it runs. A call this ledger decides inside the block —
        from the cell's task or any task the cell starts — is credited to the cell as well as to the run,
        so a cell's slice is exact whether or not other cells of the run are calling at the same time.

        Yields:
            The cell's meter.
        """
        meter = CellMeter()
        token = _OPEN_CELL.set((self, meter))
        try:
            yield meter
        finally:
            _OPEN_CELL.reset(token)

    def open_cell(self) -> CellMeter | None:
        """The meter of the cell this context is running, for this ledger, or ``None`` outside one."""
        current = _OPEN_CELL.get()
        return current[1] if current is not None and current[0] is self else None

    def admit(self, *, tool: str, action: str, spend: ExternalSpend | None) -> bool:
        """Decide whether one metered call may proceed, recording it either way.

        Called at the dispatcher **before** dispatch, which is what makes this a
        bound rather than a tally: a provider charges on the attempt, so admitting
        first and counting afterwards would let the call that crosses the ceiling
        be the one nobody stopped.

        Args:
            tool: Catalog name of the tool making the call.
            action: The action name, unprefixed.
            spend: What the calling tool says this ONE call consumes. ``None`` when the
                tool declared no provider at all, which the metering-declaration canary
                refuses for a metered action — the call is still counted and bounded, and
                lands unattributed, because a call the ceiling let through has happened
                whatever the declaration says about it.

        Returns:
            True when the call may proceed (and has been counted), False when the
            ceiling refuses it (and the refusal has been counted).
        """
        meter = self.open_cell()
        if self._ceiling is not None and self._calls >= self._ceiling:
            self._refused += 1
            if meter is not None:
                meter.refused()
            if self._none_declared:
                log.error(
                    "Eval run %s made a metered call at %s.%s, and its host declares no metered tools — the call is "
                    "refused; the host's declaration (LaunchSettings.max_metered_calls=None) or its tool's metering "
                    "declaration is wrong",
                    self._run_id,
                    tool,
                    action,
                )
            elif not self._announced:
                self._announced = True
                log.warning(
                    "Eval run %s reached its metered-call ceiling of %d at %s.%s — further metered "
                    "third-party calls are refused; the run continues and is disclosed as bounded",
                    self._run_id,
                    self._ceiling,
                    tool,
                    action,
                )
            return False
        self._calls += 1
        self._record_spend(spend)
        if meter is not None:
            meter.admitted(spend)
        return True

    def _record_spend(self, spend: ExternalSpend | None) -> None:
        """Fold one call's reported consumption into this run's per-provider totals.

        Accumulates per ``(provider, unit)`` rather than into one running number, because
        two providers' weighted units are not one quantity. A report with no provider
        contributes to the call count and nothing else — the call still happened and the
        ceiling still bounds it, but nothing can be attributed, and the runner writes those
        calls into a row of their own rather than losing them.

        Reads the fields directly rather than through ``getattr`` defaults. The type is
        shared with the host's tools (:mod:`threetears.evals.contracts.host.spend`), so a shape
        mismatch is not a state to degrade through: a defaulted read would count the call,
        attribute it to nobody and log nothing, which is a fallback masking a contract
        violation in the one module whose whole stance is that a broken mechanism must
        never look like a quiet one. The documented ``None`` case is handled explicitly;
        anything else raises where it can be attributed.

        **Money is deliberately not carried here**, and ``ExternalSpend.money`` is dropped
        rather than folded. Dollars are structurally absent at the action seam, and not
        because no rate was declared: counting happens at the dispatcher, through the
        cassette proxy, so a REPLAYED call lands the same volume as a live one. Attaching
        money would make a replay report spend it never incurred. Widening the spend
        vocabulary does not discharge that line — it is a seam decision, not a pricing one.

        Args:
            spend: The calling tool's consumption report for one call, or ``None``.
        """
        _folded(self._spends, spend)

    def refusal_message(self, *, tool: str, action: str) -> str:
        """The refusal the candidate reads when the ceiling turns a call away.

        Deliberately shaped like the ``tools_allowed`` attach refusal: it names
        what was refused, says the run — not the tool — imposed the limit, and
        tells the candidate to carry on rather than retry. A refusal that reads
        like a provider outage invites exactly the retry loop the ceiling exists
        to stop.

        Args:
            tool: Catalog name of the tool the call was for.
            action: The action name, unprefixed.

        Returns:
            One sentence pair, safe to hand back as an ``ActionResult`` description.
        """
        if self._none_declared:
            return (
                f"Cannot run '{tool}.{action}': this evaluation's host declares no metered third-party tools, so "
                "no metered call is allowed. Continue with what you already have."
            )
        return (
            f"Cannot run '{tool}.{action}': this evaluation run has reached its limit of "
            f"{self._ceiling} metered third-party calls. Continue with what you already have — "
            "further calls to paid or rationed providers will be refused."
        )


#: The cell whose calls are being metered in this context, and the ledger it meters for — so a meter opened
#: for one run's ledger never counts a call another ledger decided.
_OPEN_CELL: ContextVar[tuple[MeteredCallLedger, CellMeter] | None] = ContextVar(
    "threetears_evals_metered_cell", default=None
)


__all__ = [
    "CellMeter",
    "MeteredCallLedger",
    "MeteredCallTally",
]
