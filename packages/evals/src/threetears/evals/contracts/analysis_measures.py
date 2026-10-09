"""Per-cell measurement facts — the shapes a bundle computes and an analysis freezes.

These live in a leaf module rather than in :mod:`threetears.evals.analysis.bundle` because a stored
:class:`~threetears.evals.contracts.campaign.EvalAnalysis` carries them (its decision surface), and the
analysis models must not import the bundle: the bundle imports them. The bundle re-exports every
name here, so a caller that reads them off the bundle keeps working.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field, model_validator

from threetears.evals.contracts.metrics import AttributionScope, MeasurePopulation, MeritAxis
from threetears.evals.contracts.base import EvalDocumentModel


class MeasureSummary(EvalDocumentModel):
    """One measure's distribution across a set of results.

    **Metadata lives once, in the bundle's ``measure_catalog``, keyed by ``name``** — not
    on every summary. A collection is built for each run *and* for the campaign rollup, so
    inlining the descriptor meant repeating multi-sentence prose hundreds of times inside a
    bundle that IS the paid one-shot prompt. Two fields stay inline anyway because the
    ranking rule reads them on every measure and a join to check them would be friction
    exactly where the reasoning happens: ``attribution_scope`` (whether this measure can
    attribute a change to the subject under test at all) and ``higher_is_better`` (which
    end is the bad one).

    A measure takes exactly one of four shapes, by its descriptor's ``data_type``:

    - **numeric** — the distribution fields populated;
    - **categorical** — ``categories`` populated with counts;
    - **boolean** — ``rate`` and ``n_true``, with ``ci_low``/``ci_high`` the interval on the rate
      over its cases (Wilson's, where every case was observed once). A condition is counted, never averaged into a percentile;
    - **text** — ``texts``, every observation listed as evidence in observation order. Never
      aggregated: no mean, no mode, no count of distinct values stands in for what was said.

    ``sem`` is the standard error of the mean — the dispersion requirement, so no point
    estimate arrives without its spread. It is ``None`` rather than 0.0 below two
    observations, where the spread is unestimable rather than zero.
    """

    name: str = Field(
        min_length=1, description="The measure's registry name — its key into the bundle's measure_catalog."
    )
    attribution_scope: AttributionScope = Field(
        description="Whether the measure isolates one subsystem or reflects the whole end-to-end run."
    )
    higher_is_better: bool | None = Field(
        default=None, description="Direction, or None for a categorical measure (which has no direction)."
    )
    population: MeasurePopulation = Field(
        description=(
            "Which observations this summary was computed over: `scored` left out the ones the harness faulted, "
            "`all_observed` kept them, and `delivered` kept only the turns the candidate took — leaving out the "
            "faulted ones and the failures that took no turn (a call the model refused or errored on). A cost or "
            "latency measure, and `cost_usd`, is read over `delivered` on every surface unless it declares "
            "`all_observed`; any other measure over its declared population, otherwise the population of the "
            "surface it sits on. Two summaries of one name over different populations are different figures."
        )
    )
    n: int = Field(ge=0, description="Observations contributing to this measure.")
    n_independent: int = Field(
        default=0,
        ge=0,
        description=(
            "Distinct TEST CASES behind those observations — the independent draws. When it is below "
            "`n`, the observations are CLUSTERED (k repeats of the same case), and `sem`, `ci_low` and "
            "`ci_high` are computed over the cases, not the observations, so they already carry it. Treat "
            "`n_independent` as the sample size any claim of separation rests on."
        ),
    )
    n_zero: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Observations that were exactly zero. On a higher-is-better measure it is the best available "
            "PROXY for declines — the deliveries that produced nothing — and a latency claim must be paired "
            "against it, because 'fast' can be bought by giving up. It is not the decline count itself: an "
            "arm that declined without recording an observation never enters it. A raw count, not a rate. None on a categorical "
            "measure, where the question does not apply — which is not the same fact as a count of zero."
        ),
    )
    mean: float | None = Field(default=None, description="Arithmetic mean (numeric only).")
    p05: float | None = Field(
        default=None,
        description=(
            "5th percentile — the bad tail when higher is better (numeric only). Median-unbiased (Hyndman-Fan "
            "type 8): as likely above the true 5th percentile as below. None below 13 observations, where no "
            "estimate of it is: the smallest observation sits above the true one most of the time there."
        ),
    )
    p50: float | None = Field(default=None, description="Median, interpolated between the middle two (numeric only).")
    p95: float | None = Field(
        default=None,
        description=(
            "95th percentile — the bad tail when lower is better (numeric only). Median-unbiased (Hyndman-Fan "
            "type 8): as likely above the true 95th percentile as below. None below 13 observations, where no "
            "estimate of it is — the largest of 5 falls below the true p95 77% of the time — so read `max` there, "
            "as the worst case seen and never as a percentile. A summary stored before this rule interpolated "
            "linearly, which understates the tail at every size a cell has."
        ),
    )
    max: float | None = Field(
        default=None,
        description="Largest observed value (numeric only): the worst case seen where lower is better.",
    )
    sem: float | None = Field(
        default=None,
        description=(
            "Standard error of the mean (the dispersion requirement), over the test cases: cluster-robust, so a "
            "case's repeats are not counted as independent draws. None below n=2, or when every observation "
            "repeats one case, where it is unestimable."
        ),
    )
    ci_low: float | None = Field(
        default=None,
        description=(
            "Low bound of the 95% interval on the MEAN (t on `n_independent - 1` degrees of freedom, so honest "
            "at small n). None below n=2, or over a single case."
        ),
    )
    ci_high: float | None = Field(
        default=None,
        description=(
            "High bound of the 95% interval on the MEAN (t on `n_independent - 1` degrees of freedom, so honest "
            "at small n). None below n=2, or over a single case."
        ),
    )
    categories: dict[str, int] = Field(
        default_factory=dict, description="Value counts for a categorical measure; empty for a numeric one."
    )
    rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "A boolean measure's share of observations that held (`n_true / n`), with `ci_low`/`ci_high` its "
            "interval over the cases — Wilson's where every case was observed once. None on every other shape."
        ),
    )
    n_true: int | None = Field(
        default=None, ge=0, description="A boolean measure's observations that held. None on every other shape."
    )
    texts: list[str] = Field(
        default_factory=list,
        description=(
            "A text measure's observations, each listed whole, in observation order — evidence to read, never a "
            "statistic. Empty on every other shape."
        ),
    )

    def bad_tail(self) -> float | None:
        """The percentile at this measure's *worse* end, whichever end that is.

        A tail is only meaningful once you know which direction is bad. Reporting p95 for
        every measure invites a reader to compare ``delivered_items``' p95 — its BEST
        outcome — against a latency p95, its worst, as though both moved the same way.

        Returns:
            p05 when higher is better, p95 when lower is better, None for a categorical
            measure, one with no declared direction, or one with too few observations to
            estimate that tail (below 13).
        """
        if self.higher_is_better is None:
            return None
        return self.p05 if self.higher_is_better else self.p95

    @model_validator(mode="after")
    def _exactly_one_shape(self) -> MeasureSummary:
        """Reject a summary that is not exactly one of the four shapes.

        The split is a real either-or, and leaving it to convention means every consumer re-derives
        the discriminant its own way — one checks ``categories``, another ``p95 is not None``, a third
        the catalog's ``data_type`` — and they disagree the first time a half-populated summary
        appears. Enforcing it here makes the shape unrepresentable rather than merely undocumented.

        Raises:
            ValueError: If more than one shape is populated, or none is, or a boolean's count does not
                match its rate.
        """
        numeric = any(value is not None for value in (self.mean, self.p05, self.p50, self.p95, self.max))
        boolean = self.rate is not None or self.n_true is not None
        shapes = {
            "numeric": numeric,
            "categorical": bool(self.categories),
            "boolean": boolean,
            "text": bool(self.texts),
        }
        populated = sorted(name for name, present in shapes.items() if present)
        if len(populated) != 1:
            raise ValueError(
                f"measure {self.name!r} must take exactly one shape (numeric, categorical, boolean or text); "
                f"got {populated or 'none'}"
            )
        if boolean and (self.rate is None or self.n_true is None or self.n_true > self.n):
            raise ValueError(
                f"boolean measure {self.name!r} needs both rate and n_true, with n_true at most n; "
                f"got rate={self.rate}, n_true={self.n_true}, n={self.n}"
            )
        return self


class MeasureCollection(EvalDocumentModel):
    """Every measure a set of results carries, with the scopes that carry none named.

    ``absent_scopes`` is the reason this is a model rather than a bare list. A section
    that simply vanishes when empty gives the generator no signal about the gap — it
    reads an absence as "nothing to say here" and ranks on whatever remains, which is
    the exact failure this surface exists to prevent. Naming the empty scope lets the
    generator state the limit instead. This deliberately departs from the general
    conditional-serialization preference: here the absence is the message.
    """

    measures: list[MeasureSummary] = Field(default_factory=list, description="Measures observed, sorted by name.")
    absent_scopes: list[AttributionScope] = Field(
        default_factory=list,
        description="Attribution scopes with no measure at all — an explicit gap, never an omitted section.",
    )
    unreported_observations: list[str] = Field(
        default_factory=list,
        description=(
            "Observations the walk reached but could not summarise — an undescribed numeric name, "
            "a value of any type whose Python type contradicts its descriptor, or a DERIVED measure "
            "every result withheld because an input it is computed from was unmeasured. Names a gap "
            "that absent_scopes cannot: a subsystem with undescribed telemetry still measures "
            "nothing, while one described phase timing keeps its scope off the absent list. An "
            "entry is a measure name, optionally followed by a parenthesised reason where one "
            "applies to every withholding result — read for a human, not parsed."
        ),
    )


#: What a bar's verdict on one cell came to — see :attr:`BarVerdict.decision`.
BarDecision = Literal["cleared", "missed", "undecided", "no_interval", "no_data"]


class BarVerdict(EvalDocumentModel):
    """Whether one cell cleared one bar — computed here, never by the reader.

    **One population for every bar, whatever it names**: the cell's observations the harness did not
    fault. A faulted observation measured the rig breaking rather than the candidate — a judge
    reading a broken transcript, a check evaluated over a world the harness never seeded, a spend or
    a wall-clock the fault itself produced — so a bar is never cleared or failed on one, and how many
    were left out is stated beside the value rather than folded into it.

    **A bar on a turn's time or spend reads fewer**: the turns the candidate took (population
    ``delivered``, the cell's own reading of the measure), so a call its model refused straight away is
    in neither its value nor its ``n``. Its ``n_infra_excluded`` still counts only the faults; the
    failures left out are the cell's ``n_no_turn``. A cell where no result took a turn carries no value,
    and its verdict is ``no data``, never a clearance on a refusal's round trip.

    **The verdict is decided by the interval against the measure's declared margin, never the mean**
    (:func:`~threetears.evals.analysis.stats.interval_clears`), and it has three outcomes: cleared (the
    whole interval on the good side of the threshold less the margin), missed (the whole interval on the
    bad side), and undecided (the interval straddles it). Undecided is neither a pass nor a failure. A
    cell with a value but no interval (fewer than two observations) is not read at all. ``decision``
    names which of these, or the two absences, a verdict is.
    """

    variant_key: str = Field(
        min_length=1, description="The arm's variant — its key into variant_index, and half of its cell."
    )
    apparatus_class_id: str = Field(
        min_length=1, description="The rig it was measured under — the other half of its cell."
    )
    run_ids: list[str] = Field(
        min_length=1,
        description=(
            "The member runs whose observations this cell pooled, sorted — what a finding citing this "
            "verdict puts in `observation_refs`. The variant key is a cell coordinate, never a run id, and "
            "a citation of one is refused."
        ),
    )
    value: float | None = Field(
        default=None,
        description=(
            "The cell's mean of the bar's measure — for a goal-state check, the share of its observations "
            "that passed. None when the cell carries no observation of it."
        ),
    )
    sem: float | None = Field(
        default=None,
        description=(
            "Standard error of that mean, over the test cases, so a margin inside the noise can be said to be one."
        ),
    )
    n: int = Field(ge=0, description="Observations behind the value — the cell's non-faulted results only.")
    n_independent: int = Field(ge=0, description="Distinct test cases behind them.")
    n_infra_excluded: int = Field(
        default=0,
        ge=0,
        description=(
            "The cell's results left out because the harness faulted them — excluded from the value whatever "
            "the bar names, since a faulted observation measured the rig and not the candidate."
        ),
    )
    n_cannot_tell: int = Field(
        default=0,
        ge=0,
        description=(
            "For a bar on a judged dimension, the cell's non-faulted results the judge could not score on "
            "it — left out of the value, and not a fault. Zero for every other bar."
        ),
    )
    ci_low: float | None = Field(
        default=None,
        description=(
            "Low bound of the interval on `value` the verdict was decided on — the cell's own interval on the "
            "measure, as its summary states it. None below two observations, and on a verdict stored before bars "
            "read intervals."
        ),
    )
    ci_high: float | None = Field(
        default=None, description="High bound of that interval. None exactly when `ci_low` is."
    )
    margin: float | None = Field(
        default=None,
        description=(
            "The measure's declared margin the bar was read with, in its units — its materiality threshold, the "
            "difference too small to act on. None when it declares none: the bar is then held at the threshold "
            "itself."
        ),
    )
    cleared: bool | None = Field(
        default=None,
        description=(
            "True — cleared: the whole interval lies on the good side of the threshold less the margin, so the "
            "cell is shown no worse than the bar by more than the margin. False — missed: the whole interval lies "
            "on the bad side, so it is shown to fall short by more than the margin. None — no decision: the "
            "interval straddles the line (undecided), the cell has a value but no interval (fewer than two "
            "observations), or no observation. `decision` says which; none of the three is a pass or a failure. "
            "A stored verdict with `cleared` set and no interval predates interval verdicts: it was the cell's "
            "mean against the threshold, with no margin — read it as that point comparison "
            "(`decided_on_the_mean`)."
        ),
    )

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "The verdict as one word: `cleared`, `missed`, `undecided` (the interval straddles the threshold "
            "less the margin — neither a pass nor a failure), `no_interval` (a value from fewer than two "
            "observations, which is not read) or `no_data` (no observation). Derived from `cleared`, `value` and "
            "the interval, so it cannot disagree with them."
        )
    )
    @property
    def decision(self) -> BarDecision:
        """Which of the five a verdict is — the word every render branches on, never ``cleared`` alone."""
        if self.cleared is not None:
            return "cleared" if self.cleared else "missed"
        if self.value is None:
            return "no_data"
        if self.ci_low is None or self.ci_high is None:
            return "no_interval"
        return "undecided"

    @property
    def decided_on_the_mean(self) -> bool:
        """Whether this is a stored verdict from before bars read intervals: decided, with no interval.

        Every verdict decided now is decided on an interval, so a decision with none can only have been
        the old point comparison — the cell's mean against the threshold. Read structurally rather than
        from a stored flag, so an analysis frozen before the change needs no migration to say so.
        """
        return self.cleared is not None and (self.ci_low is None or self.ci_high is None)


class BarAdjudication(EvalDocumentModel):
    """One bar the campaign is held to, adjudicated against every cell.

    **The verdict is arithmetic and it is done here**, because a bar read by the generator was a bar
    nobody read: the declared thresholds reached the bundle raw, nothing compared them with anything,
    and the frontier's own clearing count kept its default of zero because no bar was ever passed to
    it — which a memo then quoted as "no arm cleared the bar". A reader now finds the comparison made,
    per cell, or the reason it could not be.
    """

    measure_id: str = Field(min_length=1, description="What the bar is read on, as the bar names it.")
    threshold: float = Field(description="The value the measure must reach.")
    direction: Literal["higher_is_better", "lower_is_better"] = Field(description="Which way clearing runs.")
    source: Literal["declared", "registered"] = Field(
        description=(
            "`declared` — the campaign's own bar. `registered` — the host's incumbent for this behavior, which "
            "applies because the campaign declared no bar on the same measure."
        )
    )
    state: Literal["adjudicated", "names_no_stored_measure", "not_numeric"] = Field(
        description=(
            "`adjudicated` — at least one cell carries the measure and every cell has a verdict. "
            "`names_no_stored_measure` — no non-faulted member result carries a readable value under this "
            "name, so the bar was never read and no verdict exists; a bar nobody could clear or fail, never "
            "one every arm failed. `reason` says why: nothing carried it, or the name is one no result can "
            "carry with a direction. `not_numeric` — the measure is categorical or boolean, so a threshold "
            "has nothing to compare."
        )
    )
    merit_axis: MeritAxis | None = Field(
        default=None,
        description=(
            "The merit axis the bar's measure serves, read off the descriptor the bar name resolved to — the "
            "host's declaration, never inferred from the name. None when the name did not resolve (no verdict "
            "exists, so it ranks nowhere) or the measure serves no axis — two causes one value cannot tell apart, "
            "so a reader takes which from `state` (`unreadable_name` for the first), never from this field alone. "
            "What the campaign's `merit_priority` and each question's `merit_axes` are matched against "
            "(`verdict_order`)."
        ),
    )
    reason: str | None = Field(default=None, description="Why no verdict exists, for the two non-adjudicated states.")
    verdicts: list[BarVerdict] = Field(
        default_factory=list,
        description="One per cell, ordered by (variant_key, apparatus_class_id). Empty unless the state is adjudicated.",
    )


__all__ = [
    "BarAdjudication",
    "BarDecision",
    "BarVerdict",
    "MeasureCollection",
    "MeasureSummary",
]
