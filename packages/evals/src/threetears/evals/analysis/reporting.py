"""Query-time projection of eval runs + results into flat, comparable rows.

This module is the read tier's foundation: one projection,
:func:`project_score_records`, that every downstream reporting surface consumes
(pivots, export, and the estimation engine), plus :func:`compute_comparison_sets`, which
answers *which of these runs may honestly be compared with each other*.

**Nothing here is stored.** Records are recomputed per query, exactly as
:func:`~threetears.evals.contracts.scoring.compute_pass_hat_k` is. That keeps the row shape free to
evolve while the surfaces that consume it are still being learned — a persisted
projection would freeze it against every future consumer. Aggregation is
calibrated to a corpus of dozens-to-hundreds of cells (one operator, one
scope); :func:`project_score_records` is the seam to push down into SQL if
that ever stops holding.

**The subject is a first-class coordinate, keyed on a stable id.** Composite quality is
comparable *within* a subject and never across one: rubric dimensions are derived
from each subject's own self-description and tools, so a support agent's 0.8 and a
coding assistant's 0.8 are different measurements wearing the same number, and pooling
them produces a figure that nothing downstream can falsify. Carrying
``subject_id`` on the row is what makes such pooling *detectable* — a surface
that ignores the coordinate is doing so visibly, in code a reviewer can point at,
rather than by omission nobody can see. It does not make pooling impossible: no
data model can stop a caller from averaging two numbers. It keys on
``SubjectSnapshot.subject_id`` and never on a display label: a name is not a stable
identity, so keying on it would split one renamed subject into two and merge two
same-named subjects into one.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from datetime import datetime
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, get_args

from pydantic import Field, model_validator

from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    ChangeLabel,
    case_rate_interval,
    exact_decimal,
    holm_adjust,
    interval_clears,
    separation_p,
)
from threetears.evals.contracts.analysis_measures import BarDecision
from threetears.evals.contracts.base import EvalBaseModel, VerbatimText
from threetears.evals.contracts.hashing import canonical_digest, canonical_json
from threetears.evals.contracts.host.profile import HostProfile
from threetears.evals.contracts.host.values import PooledProductionFooting, ProductionFooting
from threetears.evals.contracts.host.sweepables import CANDIDATE_MODEL_LEVER
from threetears.evals.contracts.identity import IDENTITY_VERSION, resolve_context_identity
from threetears.evals.contracts.metrics import MetricDescriptor, describe_measure
from threetears.evals.contracts.models import (
    NON_TERMINAL_RUN_STATUSES,
    OUTCOME_DIM_ID,
    RESERVED_DIM_IDS,
    SCALES,
    TRANSCRIPT_DIM_ID,
    RubricScale,
    utc_now_iso,
)
from threetears.evals.contracts.surface import FrontierDominance
from threetears.evals.contracts.result_condition import (
    JUDGE_CANNOT_TELL_OUTCOME,
    ResultOutcome,
    classify_result,
    counted_goal_verdicts,
    counted_rubric_scores,
    counted_score,
    delivered_a_turn,
    trial_exclusion,
)
from threetears.evals.contracts.scoring import (
    CompositeBasis,
    PassHatPoint,
    composite_basis,
    pool_composite_bases,
    case_pass_hat_k,
    pass_hat_k_at,
    pass_hat_k_cell,
    pool_pass_hat_k,
    pool_pass_hat_k_attempts,
    result_composite,
)
from threetears.observe import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from threetears.evals.contracts.models import EvalResult, EvalRun, GoalStateOutcome, LatencyMetrics, RunCompleteness

log = get_logger(__name__)


# The measures this projection emits today.
#
# These are OBSERVATION-level names and are deliberately not the registry's
# aggregate names: `threetears.evals.contracts.metrics` describes `mean_composite`,
# `composite_a/_b/_delta` — all of which are aggregates OVER these rows, not the
# rows themselves. Do not "fix" the mismatch by renaming these to a registry key:
# `describe_measure` is a total function, so a wrong name resolves to the
# unclassified arm (`family=None`, `transferability_class="scenario_bound"`)
# silently rather than raising. The aggregation layer maps a row metric to its
# aggregate descriptor; the mapping is that layer's job, not a naming coincidence.
#
# Scoped to these two. `METRIC_SCORE` below IS a registry key — the registry describes
# the raw per-observation judge score under that exact name — and it still maps to
# `mean_score` in the aggregation layer, because the mapping is about what a CELL holds,
# never about whether the row name resolves.
#: The observation-level measure holding a result's composite quality score.
METRIC_COMPOSITE = "composite"
METRIC_COST_USD = "cost_usd"

# The per-dimension raw judge score, on the 1-5 rubric scale. Described in the measure
# registry under this same name, so the one an operator meets here, in `export_results`'
# `metric` column and in `list_metrics` is one measure rather than three spellings.
#
# ONE metric whose dimension is a coordinate (`ScoreRecord.rubric_dim`), never a
# family of metric names like `character` / `grounding`. Three reasons, and the
# first is the one that would cost most to discover later:
#
# 1. A dim name in the metric namespace can COLLIDE. `TRANSCRIPT_DIM_ID` already
#    carries a `__` prefix for exactly this reason; a template dim named
#    `composite` would merge dim rows and composite rows into one aggregate,
#    because `pivot` selects on `r.metric == metric`. Wrong number, no error.
# 2. `row_factor='rubric_dim', column_factor='model'` answers "compare these runs
#    by dimension" in ONE table. A metric per dim needs N pivots and shows the
#    dims nowhere side by side.
# 3. A dotted name already means *open coordinate* here — `_axis_value` routes
#    any dotted axis to `factors`. Dim names are moving to a dotted
#    `<context>.<dim>` form so a dimension carries the context it was scored
#    against, and putting those in the metric namespace would give one syntax two
#    meanings. Stated as the direction it is: no dotted dim name exists in the
#    tree today, and this reason holds the shape open for one rather than
#    describing one.
#: The observation-level measure holding one judged rubric dimension's score: one row per dimension.
METRIC_SCORE = "score"

# The per-result wall-clock, read straight off `EvalResult.latency` rather than
# projected into a `ScoreRecord` — so it is NOT in `PROJECTED_METRICS` and `pivot`
# refuses it. It lives up here with the other names because it is the same KIND of
# name (observation-level, one result's value) and because `_AGGREGATE_OF_OBSERVATION`
# below has to map it: `history` series it, and a history point is a mean.
METRIC_TOTAL_MS = "total_ms"

# The two dual-score axes, as METRIC names. Deliberately the SAME strings as
# the reserved rubric-dim ids, aliased rather than re-spelled so one axis has one name
# wherever it appears — on `EvalResult.transcript_score.dim`, in a `JudgeConfig`
# binding, in the measure registry, and in this `metric` column.
#
# This is the case reason 1 above anticipates: the `__` prefix exists so a reserved id
# cannot collide with a template dim in the metric namespace, and these two are the ids
# it was minted for. They are NOT rubric dims on the `rubric_dim` coordinate, and that is
# the other half of the same decision — `compute_dimension_summary` does not aggregate
# them, so putting them on that coordinate would make the projection and the run summary
# disagree about which dimensions exist. A metric of its own is also the shape the
# question wants: their DIVERGENCE is what the catalog says separates a worse agent from
# a changed world, and a divergence between two metrics is one pivot rather than two
# levels of a coordinate.
#: The observation-level measure holding the dual-score transcript axis.
METRIC_TRANSCRIPT = TRANSCRIPT_DIM_ID
#: The observation-level measure holding the dual-score outcome axis.
METRIC_OUTCOME = OUTCOME_DIM_ID
# One row per goal-state check a result evaluated, keyed by `ScoreRecord.goal_check`: 1.0 passed, 0.0 not.
# A projected metric because this projection is the one producer behind export, pivot and every pooled
# read, and a code-graded verdict a reader cannot key by check is one they cannot act on.
METRIC_GOAL_STATE = "goal_state"

#: Every measure `project_score_records` can emit. Unlike the factor set, this one
#: is genuinely CLOSED — the projection is the only producer of score records, so
#: a name outside it can only be a typo. That distinction is why an unknown metric
#: is refused while an unknown dotted axis is not: a mistyped axis names a key a
#: run might not have set, but a mistyped metric names nothing that exists, and
#: answering it with an empty grid is indistinguishable from an empty scope.
#: A metric added to the projection adds itself here, in the same edit.
PROJECTED_METRICS = frozenset(
    {METRIC_COMPOSITE, METRIC_COST_USD, METRIC_SCORE, METRIC_TRANSCRIPT, METRIC_OUTCOME, METRIC_GOAL_STATE}
)

# The projected metrics that emit several rows per result, each keyed by a coordinate naming what the row is
# about: metric -> (that coordinate, what pooling spans in words, the short noun for one row's member). A cell
# of one of these is the statistic the
# registry describes only while the coordinate is pinned by an axis or filter; unpinned, it pools every value
# of it. Read by the pivot's scope check and POOLED caveat, the pivot's unknown-metric hint, the history
# refusal, and — through SCOPED_METRICS_HELP — the REST and MCP pivot and export help, so a metric added here is qualified on each of them at once,
# and a per-row metric NOT added here pools silently, which is the defect this table exists to make
# impossible to write without deciding.
SCOPED_METRICS: dict[str, tuple[str, str, str]] = {
    METRIC_SCORE: ("rubric_dim", "judged rubric dimension", "dimension"),
    METRIC_GOAL_STATE: ("goal_check", "goal-state check", "check"),
}

#: How each scoped metric must be read, in one sentence per metric, for every surface's help text (REST and MCP
#: alike) — rendered from the table rather than written beside it, so no surface can describe a subset.
SCOPED_METRICS_HELP = " ".join(
    f"'{metric}' has one row per {noun}: put '{field}' on an axis, or each cell pools every {noun}."
    for metric, (field, noun, _row) in SCOPED_METRICS.items()
)


# How far the parts may overrun the whole before the split is refused rather than
# reported.
#
# The parts nest inside the whole structurally — ``llm.call`` and ``tool.execute``
# spans are children of the ``agent.invoke`` turn root, and both the rounds loop and
# the action loop are serial — so a correct measurement can only leave the remainder
# at or above zero. The tolerance is therefore NOT a modelling allowance for
# overlapping work; it absorbs one thing only: the three numbers are summed in
# different groupings, so binary floating-point addition can put the difference a few
# units in the last place below zero on an exact partition.
#
# **Sized to that noise and not to a round number**, because a threshold generous
# enough to swallow a real overlap would silently convert a capture fault into a
# plausible-looking split. Durations are exact integer nanoseconds divided by 1e6, so
# the only error is the summation itself: bounded by roughly n·ε relative, and a cell
# emitting ~100 bucketed spans across a ~17-minute total puts that near 1e-8 ms. One
# nanosecond sits about two orders of magnitude above the noise and six below the
# shortest overlap any real span pair could produce. An overrun larger than this means
# a part was measured outside its whole — a span reaching a bucket from outside the
# turn roots — and that is a capture fault to report, not a negative component to
# publish.
PARTITION_TOLERANCE_MS = 1e-6

# The distinguishing clause of each reason the partition is withheld. Split out for the
# same reason the divergence lens splits its own: a test pins the CLAUSE, so the
# surrounding sentence can be rewritten for a reader without breaking the pin.
WITHHELD_UNMEASURED_COMPONENT = "not measured"
WITHHELD_PARTS_EXCEED_WHOLE = "sum to more than"


# =============================================================================
# Run completeness — the sentence a short run must carry
# =============================================================================

# The distinguishing clause of the degraded disclosure, split out for the same
# reason the withheld-partition clauses above are: tests pin the CLAUSE, so the
# sentence around it stays free to be rewritten for a reader.
DEGRADED_RUN_CLAUSE = "produced a usable measurement"

# What the breakdown above cannot claim when the counts were reconstructed rather
# than tallied by the loop. Rendered as its own sentence — the causes it qualifies
# are printed as fact, and a reader who takes "N never ran" literally on a run
# whose process was killed is reading a lower bound as a measurement.
RECONSTRUCTED_COUNTS_CLAUSE = (
    "The process running it died, so these counts were reconstructed from what reached storage: "
    "a cell that ran and failed to write is indistinguishable from one that never ran, and is "
    "counted among the latter."
)


def completeness_disclosure(completeness: RunCompleteness | None) -> str | None:
    """The one sentence a surface must show about a run that came up short.

    Returns ``None`` for a run that delivered its whole matrix, and for one
    carrying no completeness record at all — a record's absence is not evidence
    of a shortfall. A run without one has not reached a terminal state, or had
    every attempt to write the record refused (rare, and logged where it
    happened); rendering asserts nothing about either.

    The shortfall is broken down by cause because the three are not the same
    problem: cells the loop never ran mean the run was launched against less than
    it recorded, lost writes mean the harness dropped data it had, and infra
    exclusions mean the cells ran and told us nothing about the candidate. At
    least one is positive whenever the run is degraded. They account for the
    whole of ``expected - measured`` on the ``produced <= expected`` side, which
    is every run any caller can produce today; on the other side — a loop handed
    more cases than the run recorded — the surplus is not a cause of a shortfall
    and is deliberately not netted against one, so the printed causes are the
    reasons a cell went missing rather than a balancing identity.

    Args:
        completeness: The run's stored record, or ``None``.

    Returns:
        The disclosure, rendered verbatim by every surface, or ``None`` when
        there is nothing a reader needs warning about.
    """
    if completeness is None or not completeness.degraded:
        return None

    causes: list[str] = []
    never_ran = completeness.expected_cells - completeness.produced_cells
    if never_ran > 0:
        causes.append(f"{never_ran} never ran")
    lost = completeness.produced_cells - completeness.persisted_cells
    if lost > 0:
        causes.append(f"{lost} ran but the write was lost, leaving no stored result")
    if completeness.infra_excluded_cells > 0:
        causes.append(
            f"{completeness.infra_excluded_cells} excluded as a harness failure rather than a candidate outcome"
        )

    breakdown = f" ({'; '.join(causes)})" if causes else ""
    return (
        f"DEGRADED: {completeness.measured_cells} of {completeness.expected_cells} cells "
        f"{DEGRADED_RUN_CLAUSE}{breakdown}. Every rate on this run is computed over that shorter "
        "denominator, so reading it beside a complete run compares two different populations."
        + (f" {RECONSTRUCTED_COUNTS_CLAUSE}" if completeness.counted_from == "stored_results" else "")
    )


def degraded_run_disclosures(runs: Iterable[EvalRun]) -> dict[str, str]:
    """The short-matrix runs among these, each with the sentence that qualifies it.

    The seam the *pooling* surfaces read. :func:`completeness_disclosure` answers
    about one run, and the run-scoped surfaces (``get_run``, ``run_summary``,
    ``runs_compare``) each call it for the run they are about. An aggregator has
    no such run: it pools dozens into a rate, and every one of them was silently
    admitted because nothing had asked the question over a *set*. This asks it
    once, so ``frontier``, ``results_pivot`` and ``history`` cannot come to
    disagree about which of their runs were short.

    **Degradation is a property of the data, not of the run's status.** A
    ``completed`` run that lost a cell to a harness exclusion is degraded and
    passes every status filter there is, so a surface that reasoned about
    truncation from ``status`` alone would still pool it — which is exactly how
    a completed-but-short run reached all three aggregators by default.

    Args:
        runs: The runs behind an answer. Order is irrelevant; keying is by id.

    Returns:
        ``run_id -> disclosure`` for each degraded run, omitting every run that
        delivered its whole matrix and every run carrying no completeness record
        (a record's absence is not evidence of a shortfall — see
        :func:`completeness_disclosure`). Empty means nothing needs disclosing,
        so a caller can use emptiness as the predicate rather than re-deriving it.
    """
    return {run.id: sentence for run in runs if (sentence := completeness_disclosure(run.completeness)) is not None}


# =============================================================================
# Significance — the three-way read, and the disclosure that rides with it
# =============================================================================

# The reader-facing words for each read. Spelled out, never abbreviated: "n.s."
# was once read as nanoseconds by an operator — a fair guess in a report whose
# metrics are mostly named ``*_ms``, and the tell that an abbreviation a reader
# has to decode is not communication.
SIGNIFICANT_LABEL = "significant"
NOT_SIGNIFICANT_LABEL = "not significant"
NOT_TESTED_LABEL = "not tested"


# How a paired and an unpaired effect size are each named to a reader. They are
# not the same statistic: the paired one standardises the mean of the per-case
# DIFFERENCES by their own SD (Cohen's d_z), the unpaired one standardises the
# difference of means by the pooled SD (Cohen's d). Printing the paired name over
# the unpaired number claims a within-case comparison that never happened, which
# is the more persuasive of the two possible errors.
PAIRED_EFFECT_LABEL = "d_z"
UNPAIRED_EFFECT_LABEL = "d"
# The same two, bias-corrected (Hedges' g): what the engine's own tests report since the change from
# Cohen's d — ``compare_two_runs``' ``hedges_g`` and a regression flag's. A different number from d at the
# sample sizes an eval runs (0.5 against d's 0.88 at three pairs), so it is never printed under d's name.
PAIRED_HEDGES_LABEL = "g_z"
UNPAIRED_HEDGES_LABEL = "g"


def significance_read(*, significant: bool | None, p: float | None = None, effect: float | None = None) -> str:
    """Which of three things a significance flag may honestly be rendered as.

    **"Not significant" and "not tested" are different facts.** A flag arriving
    with no statistic behind it says only that nobody computed one; rendering it
    as a negative publishes a measured null the campaign never measured, and
    rendering it as a positive is the identical defect in the more persuasive
    direction. So a verdict is reported only when the statistic it came from
    travels with it.

    The predicate for "a test ran" is ``p is not None or effect is not None``,
    and a browser renderer must use the identical predicate so the two surfaces cannot answer
    differently about the same row. **A sample size is not a test:** ``n`` says
    how much data there was, never whether a difference cleared a threshold, so
    it is not part of the predicate and is not a parameter here.

    Args:
        significant: The verdict, or ``None`` when no test was run.
        p: The p-value the verdict was thresholded against, when one exists.
        effect: The effect size, when one exists — paired or not, since which
            test produced it changes what it is CALLED but not whether one ran.

    Returns:
        :data:`SIGNIFICANT_LABEL`, :data:`NOT_SIGNIFICANT_LABEL`, or
        :data:`NOT_TESTED_LABEL`.
    """
    if p is None and effect is None:
        return NOT_TESTED_LABEL
    if significant is True:
        return SIGNIFICANT_LABEL
    if significant is False:
        return NOT_SIGNIFICANT_LABEL
    return NOT_TESTED_LABEL


def format_significance(
    *,
    significant: bool | None,
    paired: bool,
    p: float | None = None,
    effect: float | None = None,
    n: int | None = None,
    hedges: bool = False,
) -> str:
    """The read plus the statistics behind it, as one cell a surface prints verbatim.

    **The single renderer of this rule.** Every server-side surface that shows a
    significance verdict calls this one: a host's compare table, its history
    table's regression flags, and a delta-table chart's values table
    (:mod:`threetears.evals.analysis.viz.intents.delta_table`). The same branch written by hand
    diverged on both its not-tested predicate and its number formatting before it
    was collapsed here, and the copy that outlived the collapse — the history
    table's — printed a bare "significant" for the one verdict that arrives with
    no p and no effect size at all.

    A browser kit's copy is the one that is deliberate, because nothing in TypeScript
    can call this; ``tests/test_significance_rule.py`` asserts this side as the facts
    that copy is pinned against, and the pin itself lives with the kit.

    The statistics are appended so a reader can check the verdict rather than
    take it — which is the whole difference between a descriptive report and an
    assertion. ``n`` is shown when known even on an untested row: a reader who
    sees ``not tested (n=1)`` learns *why* nothing was tested, where a bare
    "not tested" looks like a fault in the harness.

    Args:
        significant: The verdict, or ``None`` when no test was run.
        paired: Whether the test that produced ``effect`` paired its samples.
            Decides only the effect size's NAME (:data:`PAIRED_EFFECT_LABEL` vs
            :data:`UNPAIRED_EFFECT_LABEL`) — a caller must pass what actually
            ran, not what the surface is called.
        p: The p-value the verdict was thresholded against.
        effect: The effect size.
        n: The sample size the test would have run over.
        hedges: Whether ``effect`` is Hedges' g (the engine's tests, ``hedges_g``) rather than Cohen's d (a
            stored delta-table row's historical ``d_z``). Decides the name with ``paired``.

    Returns:
        e.g. ``"significant (p=0.0123, d_z=1.42, n=8)"`` or ``"not tested (n=1)"``.
    """
    if hedges:
        effect_label = PAIRED_HEDGES_LABEL if paired else UNPAIRED_HEDGES_LABEL
    else:
        effect_label = PAIRED_EFFECT_LABEL if paired else UNPAIRED_EFFECT_LABEL
    # The one number rule, not a fixed spelling of their own. A p-value is not
    # read against a column of its peers the way pass^k is — it is checked
    # against one threshold (α=0.05 vs p=0.04998), which the rule's four
    # significant figures already allow, and its small end is the one that
    # matters: a p of 3e-7 must not round to a zero. The effect size is a
    # measured magnitude with no bound, so its large end needs the rule too.
    parts = [
        part
        for part in (
            f"p={format_number(p)}" if p is not None else "",
            f"{effect_label}={format_number(effect)}" if effect is not None else "",
            f"n={n}" if n is not None else "",
        )
        if part
    ]
    read = significance_read(significant=significant, p=p, effect=effect)
    return f"{read} ({', '.join(parts)})" if parts else read


def significance_disclosure(*, paired: bool) -> str:
    """The sentence naming the test and threshold a comparison's verdicts rest on.

    Rendered verbatim by every surface that shows a significance verdict, for
    the reason :func:`completeness_disclosure` is: a reader must be able to see
    what was measured without reconstructing it from the code that measured it.
    Which test ran is not a detail — a paired test over shared cases and an
    unpaired test over different ones answer different questions, and most of
    what a small eval arm's verdict rests on is that difference.

    It closes by saying the output is descriptive, because until the judge is
    calibrated against human labels a flag here is not a trustworthy quality
    signal, and nothing routes it anywhere.

    Args:
        paired: Whether the comparison paired its samples by test case.

    Returns:
        The disclosure sentence.
    """
    from threetears.evals.analysis.stats import PAIRED_TEST_NAME, UNPAIRED_TEST_NAME

    test_name = PAIRED_TEST_NAME if paired else UNPAIRED_TEST_NAME
    return f"Significance: {test_name}. Descriptive only — no alerting, and no verdict without the statistic behind it."


def cross_subject_disclosure(subject_a: str | None, subject_b: str | None) -> str | None:
    """Say when a two-run comparison's composites belong to different subjects.

    Composite quality is comparable *within* a subject and never across one:
    rubric dimensions are derived from each subject's own self-description and
    tools, so two subjects' 0.8s are different measurements wearing the same
    number, and their difference is not a quantity.

    :func:`~threetears.evals.analysis.reads.compare_two_runs` calls this and,
    when it returns a sentence, withholds the composite delta AND the
    significance test on it rather than emitting numbers nothing downstream could
    falsify — in the service, so REST and MCP inherit one answer instead of each
    deciding. A surface that re-derived the rule from subject ids of its own
    would be the second implementation this exists to prevent.

    There is no undecidable case. This used to carry one — two runs whose subject nobody recorded
    are not evidence of sameness, and comparing two blanks for equality would manufacture exactly
    the claim this refuses — but a subject key can no longer be blank, so the pair is not
    constructible and the branch would answer about nothing.

    Args:
        subject_a: Run A's subject key.
        subject_b: Run B's subject key.

    Returns:
        The disclosure, or ``None`` when both sides name the same subject and
        the composites are therefore comparable.
    """
    left = (subject_a or "").strip()
    right = (subject_b or "").strip()
    if left == right:
        return None
    return (
        f"Cross-subject comparison: run A scored subject `{left}` and run B scored subject `{right}`. "
        "Rubric dimensions are derived from each subject's own self-description, so these composites "
        "are different measurements wearing one name — the composite delta is withheld, and the "
        "per-run composites must be read separately."
    )


class LatencyPartition(EvalBaseModel):
    """One result's ``total_ms`` split into its named parts and a named remainder.

    ``total_ms`` is the wall-clock of the candidate's turn-root spans, and the two
    measured parts nest inside it. What is left over — whatever the candidate does
    between its model and tool calls: building its input, parsing its output, updating
    and persisting state — had no name, so a whole-run latency movement that lived there could not be placed. It
    could only be reported as a total that moved while both named parts stayed flat,
    which reads exactly like a measurement error. Naming the remainder is what turns
    that into an attribution.

    **Derived, never stored.** ``orchestration_ms`` is ``total_ms`` minus the two
    parts, recomputed per query exactly as :func:`project_score_records` and
    :func:`~threetears.evals.contracts.scoring.compute_pass_hat_k` are. A persisted copy would be a
    second answer to a question the three captured components already settle.

    **What is deliberately NOT in here.** The drain wait, the judge phase and every
    background phase timing are milliseconds measured on the same clock that fall
    *outside* the turn roots — background work on a detached trace root, and scoring
    that happens after the turns end. They are disjoint from ``total_ms``, not
    components of it, which the registry records by leaving their ``contained_by``
    unset. Folding any of them into the remainder is the arithmetic that once produced
    a ~95-second unattributed swing describing no stretch of wall-clock at all, and
    the reason this class takes a :class:`~threetears.evals.contracts.models.LatencyMetrics` and
    reads three fields of it rather than summing whatever it holds.

    Exactly one of the split and ``withheld`` is present, enforced below: a caller
    that gets no numbers gets a sentence saying why, and never a silent ``None`` it
    has to invent an explanation for.
    """

    #: The whole being partitioned — wall-clock across the result's turn-root spans.
    total_ms: float | None = None
    #: The share inside model calls.
    llm_ms: float | None = None
    #: The share inside tool executions.
    tool_ms: float | None = None
    #: The named remainder: turn wall-clock inside neither a model call nor a tool
    #: execution. Clamped to exactly 0.0 when the parts overrun the whole by less than
    #: the float-noise tolerance, since a negative share of a whole describes nothing.
    orchestration_ms: float | None = None
    #: Why no split is published, as a sentence for a reader — set exactly when the
    #: split is absent. A partition needs all three components measured, and needs the
    #: parts to fit inside the whole.
    withheld: str | None = None

    @model_validator(mode="after")
    def _split_or_reason(self) -> LatencyPartition:
        """Refuse a record that is neither a whole split nor a whole refusal.

        Checks all four numbers rather than the remainder alone: a record carrying a
        total beside a withheld sentence would be a partition that half-published, and a
        reader has no way to tell which half to believe.
        """
        numbers = (self.total_ms, self.llm_ms, self.tool_ms, self.orchestration_ms)
        present = [value is not None for value in numbers]
        if any(present) != all(present) or all(present) == (self.withheld is not None):
            raise ValueError(
                "exactly one of the complete split / withheld must be set — got "
                f"total_ms={self.total_ms!r}, llm_ms={self.llm_ms!r}, tool_ms={self.tool_ms!r}, "
                f"orchestration_ms={self.orchestration_ms!r}, withheld={self.withheld!r}"
            )
        return self


def decompose_total_ms(latency: LatencyMetrics | None) -> LatencyPartition:
    """Split one result's ``total_ms`` into its parts and the named remainder.

    Args:
        latency: The result's captured latency, or ``None`` for a cell that timed
            nothing whatever.

    Returns:
        A :class:`LatencyPartition` carrying either the four numbers or a sentence
        naming why they were withheld.
    """
    if latency is None:
        return LatencyPartition(withheld="this cell timed nothing, so there is no total to partition.")

    components = {"total_ms": latency.total_ms, "llm_ms": latency.llm_ms, "tool_ms": latency.tool_ms}
    unmeasured = [name for name, value in components.items() if value is None]
    if unmeasured:
        # Not zero-filled. A cell can measure an `llm.call` while producing no turn
        # root — the candidate failing outside the turn wrapper does exactly that —
        # and treating the missing one as zero would publish a remainder computed
        # from a number nobody observed, in the same units as ones somebody did.
        verb = "was" if len(unmeasured) == 1 else "were"
        return LatencyPartition(
            withheld=(
                f"{', '.join(unmeasured)} {verb} {WITHHELD_UNMEASURED_COMPONENT}, so the parts cannot account for the whole."
            )
        )

    # Read back through the same dict the absence check walked, so the two can't
    # diverge into checking one set of fields and partitioning another.
    measured = {name: value for name, value in components.items() if value is not None}
    total, llm, tool = (float(measured[name]) for name in ("total_ms", "llm_ms", "tool_ms"))
    remainder = total - llm - tool
    if remainder < -PARTITION_TOLERANCE_MS:
        return LatencyPartition(
            withheld=(
                f"llm_ms and tool_ms {WITHHELD_PARTS_EXCEED_WHOLE} total_ms by "
                # The one number rule, and both ends of the range are why. A fixed
                # `.1f` renders every overrun below 50 microseconds as "by 0.0ms" — a
                # refusal whose own sentence says nothing happened — and the threshold
                # is a nanosecond, so those are reachable. A capped `%g` breaks the
                # other end: it goes scientific once the exponent reaches the precision,
                # so `.3g` prints a 464ms overrun, the size actually measured here, as
                # "by 4.64e+02ms". `format_number` keeps a small value non-zero and a
                # large one whole. This sentence is read by a person.
                f"{format_number(-remainder)}ms, so something reached a component bucket from outside the turn roots."
            )
        )
    return LatencyPartition(
        total_ms=total,
        llm_ms=llm,
        tool_ms=tool,
        # Clamped rather than published negative: within the tolerance the overrun is
        # float-addition noise on an exact partition, and a share of a whole that reads
        # as less than none of it would be a worse answer than the zero it really is.
        orchestration_ms=max(remainder, 0.0),
    )


#: The token a served-model coordinate (:attr:`ScoreRecord.served_model`) carries for candidate calls whose
#: response named no model, or that were stored before served models were recorded. Never the requested id.
SERVED_MODEL_UNRECORDED = "unrecorded"

#: How a served-model coordinate joins the models one result's candidate calls named. A result answered by two
#: models is itself a mixture, and its coordinate says so rather than picking one.
_SERVED_MODEL_JOIN = " + "

#: How many models answered a pooled set of candidate calls, as the provider's responses named them.
ServedModelState = Literal["one", "pooled", "unrecorded"]


def served_model_state(served_models: Collection[str], n_unrecorded: int) -> ServedModelState:
    """The one rule every served-model reading's state follows (#684).

    Args:
        served_models: The distinct models the responses named.
        n_unrecorded: The results with at least one candidate call whose response named none.

    Returns:
        ``pooled`` where two or more models were named, ``one`` where exactly one was and every call named it,
        ``unrecorded`` otherwise: whether one model answered cannot be established, and unknown is never one.
    """
    if len(served_models) > 1:
        return "pooled"
    return "one" if served_models and not n_unrecorded else "unrecorded"


class ResultServedReading(NamedTuple):
    """What one result's candidate calls say about the model that answered them.

    Attributes:
        requested: The candidate model the result's run asked for — the id an arm is keyed by.
        served: Every model the provider's responses named for its candidate calls.
        unrecorded: Some candidate row names no served model — a response that named none, or a row
            stored before served models were recorded.
    """

    requested: str
    served: frozenset[str]
    unrecorded: bool


def served_reading(result: EvalResult) -> ResultServedReading | None:
    """Read which model answered one result's candidate calls, off its candidate usage rows.

    ``RoleUsage.served_model`` only — what the provider's response named — and never ``RoleUsage.model``
    or the run's ``candidate_model``, which are what the launch asked for and, for a floating alias, name
    the pointer rather than the model behind it.

    Args:
        result: The result.

    Returns:
        The reading, or ``None`` when the candidate left no usage row: nothing was called, so nothing
        answered, and no claim is made about it.
    """
    rows = [row for row in result.usage if row.role == "candidate"]
    if not rows:
        return None
    return ResultServedReading(
        requested=result.model,
        served=frozenset(row.served_model for row in rows if row.served_model),
        unrecorded=any(not row.served_model for row in rows),
    )


def served_model_coordinate(reading: ResultServedReading | None) -> str | None:
    """One result's served models as the coordinate a pivot groups on and an export carries.

    Args:
        reading: The result's reading, from :func:`served_reading`.

    Returns:
        The named models, sorted and joined with `` + ``, with :data:`SERVED_MODEL_UNRECORDED` among them where
        some call named none (alone where none did). ``None`` where the candidate made no call.
    """
    if reading is None:
        return None
    return _SERVED_MODEL_JOIN.join(sorted(reading.served) + ([SERVED_MODEL_UNRECORDED] if reading.unrecorded else []))


class ServedModelReading(EvalBaseModel):
    """Which models answered the candidate calls a contestant, cell or row pooled (#684).

    A contestant is keyed by the model id its launch ASKED for, and a floating alias resolves on the provider's
    side, so runs of one contestant can have been answered by different models and still pool as one. The mixture
    is disclosed, not split: the key is fixed at launch, before any response names a model. The bundle's per-arm
    reading (``AnalysisContextBundle.arm_served_models``) is this one, keyed by arm.
    """

    served_models: list[str] = Field(
        default_factory=list,
        description=(
            "Every distinct model the provider's responses named as having answered these candidate calls, sorted. "
            "Never the requested id: a call whose response named no model adds nothing here and is counted in "
            "n_unrecorded."
        ),
    )
    n_results: int = Field(ge=1, description="The pooled results whose candidate calls left a usage row.")
    n_unrecorded: int = Field(
        ge=0,
        description=(
            "Of those, the results with at least one candidate call whose response named no model — or stored "
            "before served models were recorded. Not recorded, never a match with the requested id."
        ),
    )
    state: ServedModelState = Field(
        description=(
            "one = every candidate call named one and the same model. pooled = two or more models answered, so the "
            "numbers are a mixture of them. unrecorded = at most one model was named and some call named none, so "
            "whether one model answered cannot be established."
        )
    )

    @model_validator(mode="after")
    def _state_follows_the_counts(self) -> ServedModelReading:
        """The state is the one the served models and the unrecorded count imply, never a second opinion.

        Returns:
            The validated reading.

        Raises:
            ValueError: ``state`` disagrees with ``served_models`` and ``n_unrecorded``.
        """
        expected = served_model_state(self.served_models, self.n_unrecorded)
        if self.state != expected or self.n_unrecorded > self.n_results:
            raise ValueError(
                f"a pool with these served models and unrecorded calls is {expected!r}, not {self.state!r}"
            )
        return self

    def disclosure(self) -> str | None:
        """The reading in words, or ``None`` where one named model answered every call.

        Returns:
            A sentence for a pooled or unrecorded reading.
        """
        if self.state == "pooled":
            return (
                f"pools results answered by {len(self.served_models)} models ({', '.join(self.served_models)}) under "
                "one requested model, so its numbers mix those models"
            )
        if self.state == "unrecorded":
            named = f"; the rest named {self.served_models[0]}" if self.served_models else ""
            return (
                f"{self.n_unrecorded} of {self.n_results} result(s) made candidate calls whose response named no "
                f"model{named}, so whether one model answered them cannot be established"
            )
        return None


def pool_served_readings(readings: Iterable[ResultServedReading | None]) -> ServedModelReading | None:
    """Pool per-result readings into one, the rule every surface that groups by contestant reads.

    Args:
        readings: The pooled results' readings; ``None`` (no candidate call) contributes nothing.

    Returns:
        The pooled reading, or ``None`` when no result's candidate left a usage row.
    """
    own = [reading for reading in readings if reading is not None]
    if not own:
        return None
    models = sorted(set().union(*(reading.served for reading in own)))
    n_unrecorded = sum(1 for reading in own if reading.unrecorded)
    return ServedModelReading(
        served_models=models,
        n_results=len(own),
        n_unrecorded=n_unrecorded,
        state=served_model_state(models, n_unrecorded),
    )


def pooled_served_models(results: Sequence[EvalResult] | Sequence[ScoreRecord]) -> ServedModelReading | None:
    """Say which models answered the candidate calls a pooled figure rests on (#684).

    Shared, as :func:`pooled_cost_compositions` is, so the frontier, history, pivot and two-run comparison read
    one predicate, and the bundle's ``arm_served_models`` the same one (:func:`pool_served_readings`).

    Args:
        results: The results the caller pooled, or the rows a pivot cell pooled — read through
            :attr:`ScoreRecord.served_model`, once per result however many rows it put in the cell.

    Returns:
        The pooled reading, or ``None`` when no pooled result's candidate made a call.
    """
    readings: dict[str, ResultServedReading | None] = {}
    for result in results:
        if isinstance(result, ScoreRecord):
            if result.result_id not in readings:
                readings[result.result_id] = _reading_of_coordinate(result.served_model, result.model)
        else:
            readings[result.id] = served_reading(result)
    return pool_served_readings(readings.values())


def _reading_of_coordinate(coordinate: str | None, requested: str) -> ResultServedReading | None:
    """Read a served-model coordinate back into the reading :func:`served_model_coordinate` wrote it from."""
    if coordinate is None:
        return None
    tokens = coordinate.split(_SERVED_MODEL_JOIN)
    return ResultServedReading(
        requested=requested,
        served=frozenset(token for token in tokens if token != SERVED_MODEL_UNRECORDED),
        unrecorded=SERVED_MODEL_UNRECORDED in tokens,
    )


class ScoreRecord(EvalBaseModel):
    """One measurement at one fully-specified cell.

    A record exists only for a result that ran. The absence of a cell is not
    representable here and must not be: "did not run" is a property of a query's
    expected grid, not of an observation, and collapsing the two is how a missing
    cell comes to render as a zero.

    Coordinates are stable ids, never rendered strings. ``subject_label`` is
    carried as a display label only and must not be grouped on.

    **The measurement is ``metric``/``value``, except for a kind that grades itself.**
    A code-graded candidate has no composite and no judged dimension, so its grade
    rides on ``host_measures`` — the result's own mechanical measures, on its
    composite row alone. A judge-graded row and a code-graded row are then told
    apart by which of the two is populated, rather than by one of them being
    quietly empty everywhere the other is not.

    **Each identity key travels with the predicate version that produced it.**
    ``variant_key`` and ``context_key`` are derived by separate predicates that
    today share one :data:`~threetears.evals.contracts.identity.IDENTITY_VERSION` counter — so a
    bump for either moves the number stamped beside *both*, and a row can carry a
    version whose change did not touch its own predicate. Each key still needs its
    own column: a single ``identity_version`` on the row could only qualify one of
    them and would silently mis-qualify the other. A consumer that
    groups on one key gates comparability on *that* key's version: keys from
    different predicates are queryably distinct rather than silently regrouped,
    which is the whole reason the versions are stored beside the keys instead of
    only hashed into them.

    **The sentence above is an obligation, not a description of every surface that
    reads one.** The lenses that gate are ``frontier`` and ``history``, which group
    through :func:`_contestant_key` and carry the version in the key — but neither of
    those reads a ``ScoreRecord`` at all. The surface that reads this row is ``pivot``,
    whose axis set is deliberately open: ``_axis_value`` reads any declared coordinate by
    name, so ``row_factor="variant_key"`` groups the raw digest. The pivot does not gate
    there, because gating means deciding that one particular axis implies a partition,
    which the open-axis design does not let this module assert; it **discloses** instead
    (#672). A cell grouped on an identity key that pools more than one version of that
    key's predicate names the versions (:attr:`PivotCell.identity_versions`), and the
    table says which cells do (:attr:`PivotTable.identity_pooling_disclosure`).
    ``variant_identity_version`` is itself a declared coordinate, so pivoting the key
    against it separates the versions.

    The sentence stood here unqualified for some time while nothing did it, which is how
    the wrong merge it forbids reached production on the two lenses that now gate.

    **``context_identity_version`` is deliberately still consumerless, and that is not the
    same gap.** The variant side had a live wrong merge because two lenses GROUPED on
    ``variant_key``. Nothing groups on ``context_key`` anywhere: ``comparison_sets``
    partitions on ``(subject_id, template_id)`` and reads the key only to raise
    ``BADGE_CONTEXT_DIFFERS``, which is a per-pair comparison that already fails closed when
    the keys differ for any reason, version included. So there is no pooling decision for a
    context-side version gate to correct — the field waits for a consumer that groups, and
    stamping one today would be a gate over nothing.
    """

    # --- The comparability partition ---
    subject_id: str
    subject_label: str = ""

    # --- Location ---
    #
    # `scope_id` is the storage scope the observation was read under, and the engine
    # never interprets it: nothing branches on its value, it is not an identity
    # component, and no pooling or comparability rule reads it. **A caller may still
    # pivot on it** — the axis set is deliberately open, so `_axis_value` reads any
    # declared coordinate by name and "cost by scope" is a legitimate question. Opaque
    # means the engine assigns it no meaning, not that it is unqueryable.
    scope_id: str
    run_id: str
    result_id: str

    # --- Cell coordinates ---
    #
    # The two identity keys resolve at different granularities and that is a
    # property of what they measure, not an inconsistency to normalize away: a
    # variant is the resolved contestant stack, which varies per result because a
    # run carries several candidate models, while the measurement context is
    # pinned once for the whole run.
    template_id: str | None = None
    test_case_id: str
    model: str
    k_iteration: int
    variant_key: str | None = None
    variant_identity_version: int | None = None
    context_key: str | None = None
    context_identity_version: int | None = None

    # The run-pinned judge and simulator roles, carried as first-class
    # coordinates rather than only hashed into `context_key`'s roles component.
    # A digest is enough to gate comparability (badge that the roles differ,
    # per `comparison_sets`) but cannot be grouped ON: "which runs were pinned
    # to the same judge" is a pivot the hash cannot answer. Sourced run-level —
    # the roles are pinned once per run, exactly as `context_key` composes them
    # — so a pivot on `judge_model` partitions on the same value the
    # comparability badge reads, and the two surfaces cannot disagree about
    # which runs used the same judge. `None` means the run was not judged (judge) or ran
    # no simulated user, or recorded none (simulator). Clean-snapshot provenance stays
    # in the digest, un-pivotable, until a query needs it; cassette mode is carried below,
    # because a cost pivot needed it.
    #
    # **`judge_model` is the run's PIN and is NOT the model that scored this
    # row's dimension.** Under the judge-model cascade (role default < run pin <
    # per-dim `JudgeConfig.model`) a dim whose config names a model is scored by
    # that instead, so on a `score` row the pin is a fallback the dimension may
    # never have used — reading it as the judge of record attributed a
    # gemini-scored dimension to the run's gpt pin, on the same export whose
    # `run_summary` correctly reported two judges. `rubric_dim_judge_model`
    # below is the per-row answer; this one stays run-level so the badge
    # agreement above survives, and the two coordinates are kept apart rather
    # than one of them being made to mean both things.
    judge_model: str | None = None
    simulator_model: str | None = None

    # The cassette mode the run RECORDED (`off`, `capture` or `replay`), run-level like the
    # judge pin above. Carried because a replayed result did not spend what a live one does:
    # a replayed background delivery spent no inner-agent dollars, so its `cost_usd` is
    # smaller for a reason in the apparatus, not the configuration (#658). A cost pivot reads
    # it to withhold a cell that pools replayed with live results, the export carries it as a
    # column, and a caller can pivot on it like any other coordinate. `None` only on a row
    # built without a run.
    cassette_mode: str | None = None
    # How many of this result's async deliveries a harness supplied rather than the candidate's
    # background work producing them — a replayed capture or a seeded finding
    # (`count_substituted_deliveries`). The per-result half of the same fact: a seeded finding
    # substitutes in a run that recorded `off`, so the run's mode alone cannot say which results
    # spent less. Non-zero is what withholds the result's production-replicating cost.
    substituted_deliveries: int = 0

    # The model the provider's responses NAMED as having answered this result's candidate calls (#684) —
    # `served_model_coordinate`, read off `RoleUsage.served_model`, never off `model`, which is the id the
    # launch asked for and, for a floating alias, the pointer rather than the model behind it. A declared
    # coordinate so a pivot can split by it and the export carries it as a column. Several models a result's
    # calls named are joined with ` + `; `unrecorded` stands for calls whose response named none (or rows stored
    # before served models were recorded) and is never read as the requested id. `None` where the candidate
    # made no call at all.
    served_model: str | None = None

    created_at: str = ""

    # Which rubric dimension this row measures — a coordinate like `model` or
    # `test_case_id`, so it can go on either axis or into a filter with no new
    # machinery, and `_require_coordinate_name` accepts it statically.
    #
    # Populated only on `score` rows, and `None` there too when the result
    # carried no dims at all. A composite or cost row leaves it `None` because it
    # measures the result as a whole: those metrics have no dimension, and
    # inventing one would make a `rubric_dim` pivot of composites look
    # per-dimension when it is not.
    rubric_dim: str | None = None

    # The scale `rubric_dim` was judged on, set exactly where `rubric_dim` is: `value` is 1-5 on
    # `ordinal` and 1/0 on `pass_fail`, so a mean over rows of one dimension is a level or a pass
    # rate, and a pooling across the two is neither.
    rubric_scale: RubricScale | None = None

    # The goal-state check a `goal_state` row reports, as its expression — set on those rows and on no other.
    # A coordinate of its own rather than `rubric_dim`, which is pinned to template rubric dimensions: a check
    # is code-graded and a dimension is judged, and a `rubric_dim` pivot that pooled the two would mean
    # neither. A pivot or export can key on it like any other coordinate.
    goal_check: str | None = None

    # Which dimensions the COMPOSITE on this result was meaned over, sorted.
    #
    # Populated on `composite` rows alone, because it qualifies that number and no
    # other. A composite is a mean across whatever dims the result happened to
    # carry, so two composites are comparable only when they were meaned over the
    # SAME dims — otherwise one averages three things and the other five, and
    # their difference is partly a difference in what was being averaged rather
    # than in what was measured. Nothing recorded this before, so every pooled
    # quality comparison made that assumption silently and none of them could be
    # checked; a consumer can now gate on it instead of trusting that a template
    # never changed mid-campaign.
    #
    # `None` on a row whose composite is itself None (infra-excluded, or no dims
    # at all) — there is no basis for a number that does not exist, and an empty
    # list would read as "meaned over nothing", which is a different claim.
    dimension_basis: list[str] | None = None

    @model_validator(mode="after")
    def _a_composite_states_what_it_was_meaned_over(self) -> ScoreRecord:
        """A quality number with a value must name its basis; one without must not claim a basis.

        The structural half of the disclosure, and it lives HERE rather than on a companion type
        because this is the row that ships. A second model carrying the rule would be the concept
        landing twice with the guarantee on the copy nothing is on the path of — which is exactly
        how a pooled number with no basis stays constructible while a docstring says it cannot be.

        Returns:
            The validated record.

        **Three states, and the empty list is one of them.** ``None`` is "nobody recorded a
        basis", which is the defect. ``[]`` is "meaned over nothing", which is a real and common
        answer: a candidate failure scores 0.0 by policy rather than by averaging, so it has a
        value and legitimately no dimensions behind it. A non-empty list is a real mean. Refusing
        ``[]`` alongside ``None`` would reject every candidate-failure row and force the producer
        to lie about one of the two.

        Raises:
            ValueError: A composite carries a value and no basis was recorded at all, so nothing
                downstream can tell whether it is comparable with the next one; or it records a
                basis beside no value, which claims a mean that was never taken.
        """
        if self.metric != METRIC_COMPOSITE:
            return self
        if self.value is not None and self.dimension_basis is None:
            raise ValueError("a composite with a value must state the dimensions it was meaned over")
        if self.value is None and self.dimension_basis is not None:
            raise ValueError("a composite with no value cannot have been meaned over anything")
        return self

    # WHICH MODEL SCORED `rubric_dim` — the per-row half of the judge question
    # `judge_model` above cannot answer. Declared as a field, so it is pivotable,
    # filterable and exported by exactly the same machinery `rubric_dim` is, with
    # no per-coordinate work anywhere.
    #
    # `None` wherever no SINGLE model scored the row. A composite or cost row
    # measures the result as a whole, spanning every dim the judge scored, so
    # naming one model would make a `rubric_dim_judge_model` pivot of composites
    # look per-dimension when it is not; a dimensionless `score` row (the judge
    # scored nothing) has nothing to attribute either.
    #
    # That rule is NOT "null exactly where `rubric_dim` is null", which it used to
    # say and which the dual-score axes falsify: a `__transcript__` / `__outcome__`
    # row carries no `rubric_dim` (the metric IS the axis) and was scored by exactly
    # one judge, which the run's own map can name. Withholding it there would drop a
    # real attribution to preserve a coincidence between two fields.
    #
    # `None` ALSO where the run cannot say — see `dim_judge_model`, which is the
    # one definition of this value and reads the same map the run summary's
    # "Judged by" block and `bisect_runs`' `judge_dim_divergence` read. A cell in
    # a pivot or a CSV carries no caveat channel, so a reconstruction is not
    # published here as though it were a record.
    rubric_dim_judge_model: str | None = None

    # --- Open coordinates: every lever the host's registry resolves for the run ---
    #
    # A kind's overlays (`gm.difficulty`), an open family's members, and anything a
    # host registers next. These live in a map rather than as declared fields
    # because the factor set is deliberately OPEN: a new lever has to become
    # pivotable with no change to this model, no migration, and no per-key
    # rendering work. Enumerating them as fields would mean every new lever is a
    # schema change plus an edit in each read surface, which is the coupling the
    # open set exists to prevent.
    #
    # The *identity rule* stays closed even though the set does not: anything
    # that can change an outcome is either a recorded coordinate or explicitly
    # declared noise. A key absent here is absent because the run carried no
    # level for it, not because reporting declined to carry it.
    factors: dict[str, str] = {}

    # --- The measurement ---
    metric: str
    value: float | None = None

    # The roles a `cost_usd` row's dollars were summed over (`EvalResult.cost_roles`), set on
    # cost rows alone (#625). What the sum covers differs per run — metered third-party spend
    # enters only for a run whose operator declared a rate — so two equal totals can cover
    # different things, and a cost pivot reads this to say which compositions each cell pooled
    # (`pooled_cost_compositions`, the reading every pooled cost surface shares). `None` on every
    # other row, which measures no dollars.
    cost_roles: list[str] | None = None

    @model_validator(mode="after")
    def _only_a_cost_row_names_its_cost_roles(self) -> ScoreRecord:
        """A cost composition qualifies a cost and nothing else.

        Returns:
            The validated record.

        Raises:
            ValueError: A non-cost row carries a cost composition.
        """
        if self.metric != METRIC_COST_USD and self.cost_roles is not None:
            raise ValueError(f"a {self.metric!r} row measures no dollars, so it has no cost composition")
        return self

    # --- The kind's own grade, for a kind whose scoring is code rather than a judge ---
    #
    # `EvalResult.host_measures` carried straight through: the mechanical grade the
    # candidate kind computed itself, by registered measure name. An extractor reports
    # `field_accuracy` here (a classifier reports `match` and `confusion_cell`, from which the
    # engine derives accuracy itself); a judge-graded kind reports nothing and the map is empty.
    #
    # It exists because this projection was judge-centric — composite, cost, per-dimension
    # score — and a code-graded run has none of those. Its composite is null (there are no
    # rubric dims to mean), its score row is the dimensionless null, and its actual grade
    # was nowhere on the export at all, so "which case did this arm miss" could not be
    # asked of the surface built to answer it.
    #
    # AN OPEN MAP, for the same reason `factors` is one and NOT for the same reason: a
    # host declares whatever it measures, so a kind that grades on two axes rather than one
    # adds a key here and a column to the CSV, with no edit to this model and no second
    # shape for the two-grade case. What stays closed is the *metric* namespace — these are
    # deliberately not rows of their own, because `PROJECTED_METRICS` is closed precisely so
    # an unknown metric can be refused as a typo, and an open host catalogue in that
    # namespace would end that. They are also not `value`s on the composite row: a 0/1
    # mechanical grade and a 0-1 judge composite are different quantities, and pooling them
    # under one metric is exactly the silent difference this field exists to end.
    #
    # CARRIED ON THE COMPOSITE ROW ALONE — the result's quality row, enforced by
    # `_only_the_quality_row_carries_the_hosts_grade` below. Repeating it on every row of a
    # result would make a spreadsheet's mean over the column count each observation as many
    # times as the result has rows, which is the same inflation the per-observation
    # composite comment above refuses one field over. One row per result carries the grade,
    # so averaging the column is the arm's accuracy and filtering it to 0 names the cases
    # that arm missed.
    host_measures: dict[str, bool | float | str] = {}

    @model_validator(mode="after")
    def _only_the_quality_row_carries_the_hosts_grade(self) -> ScoreRecord:
        """A result's mechanical grade rides on its quality row, and on no other.

        Structural rather than a convention held by the producer, for the reason the
        composite's own validator is: the property a reader relies on is "one row per result
        carries this", and a second row carrying it makes every mean over the column silently
        wrong by a factor of however many rows that result emitted. Nothing downstream could
        detect that, because each individual value would be right.

        Returns:
            The validated record.

        Raises:
            ValueError: A non-composite row carries a host grade.
        """
        if self.metric != METRIC_COMPOSITE and self.host_measures:
            raise ValueError(
                f"a {self.metric!r} row cannot carry the host's grade — it belongs to the result, and the "
                "composite row is the one row per result that carries it"
            )
        return self

    # --- How this observation participates in scoring ---
    outcome: str


class ProjectionExclusions(EvalBaseModel):
    """What the projection dropped on the way to producing rows, and why.

    Carried on the wire rather than only logged. A corpus whose results cannot be placed
    projects to zero rows, and a surface that reports only "no
    observations" renders it identically to a scope nobody has ever run —
    which is the single most likely misreading of this whole tier, because the
    honest answer ("the data exists but cannot be grouped") and the wrong one
    ("there is no data") differ by a fact the operator has no other way to get.

    The classes are kept apart because they mean opposite things to the reader.
    ``results_without_run`` is data that *cannot* be placed — an integrity gap
    worth investigating.
    ``results_outside_queried_runs`` and ``results_from_archived_runs`` are data
    deliberately left out because the corpus was narrowed, which is the answer
    working as asked. Pooling a deliberate class with an unplaceable one was the
    original defect: the caller's own ``status="completed"`` default filters out
    every in-flight run, so the routine case reported its results under a counter
    documented as unplaceable.

    **The two deliberate classes are kept apart for a further reason — they have
    different remedies.** A run leaves the reported cohort either because the
    operator archived it or because the caller's ``status`` filter did not match
    it, and the fix differs: ``run_archive(archived=false)`` versus a wider
    ``status``. Counting both as ``results_outside_queried_runs`` let every read
    surface tell an operator to pass ``status='all'`` for observations that
    ``status='all'`` provably cannot reach — worse than saying nothing, because
    the advice can be followed exactly and return the identical answer.

    **Archival takes precedence when both apply**, because un-archiving is the
    step that is *necessary* in that case while a status widening alone would
    still not reach the run. The surfaces word the archival disclosure to stay
    true there: it says un-archiving returns the run to the reported cohort,
    where the ``status`` filter then applies as usual, rather than promising the
    observations reappear.
    """

    results_without_run: int = 0
    results_outside_queried_runs: int = 0
    results_from_archived_runs: int = 0

    @property
    def total(self) -> int:
        """Total observations dropped before any aggregation.

        Returns:
            The sum of every exclusion class. Deliberately includes the
            deliberately-filtered classes: the count exists to answer "is this
            scope empty, or did I just not ask for what is in it", and every
            cause makes an empty table non-empty underneath.
        """
        return self.results_without_run + self.results_outside_queried_runs + self.results_from_archived_runs


class ScoreProjection(EvalBaseModel):
    """The projected rows plus an account of everything that did not become one.

    The two travel together because they are only meaningful together: ``n=0``
    means something different depending on whether anything was excluded, and
    separating them is what lets a caller forget to ask.

    ``completeness_disclosures`` is the third of the same kind, and it is about
    rows that ARE here rather than rows that are not: a run that measured less
    than the matrix it promised still projects rows, and pooling them with a full
    run's computes a rate over two different populations. It rides on the
    projection so an aggregator cannot render the rows without the sentence.
    """

    records: list[ScoreRecord] = []
    exclusions: ProjectionExclusions = ProjectionExclusions()
    #: ``run_id -> DEGRADED sentence`` for every placed run that came up short of
    #: its matrix, from :func:`degraded_run_disclosures`. Restricted to runs that
    #: actually contributed a row: a degraded run whose every result was excluded
    #: qualifies nothing here, and disclosing it would warn about a population the
    #: answer does not contain.
    completeness_disclosures: dict[str, str] = {}


def _subject_id_of(run: EvalRun, result: EvalResult | None = None) -> str:
    """Resolve the real subject a measurement is attributed to.

    Two fields carry the same fact, populated by different generations of the
    runner: ``EvalResult.subject_id`` is per-result attribution and is ``None``
    on stub slots and pre-isolation rows, while ``SubjectSnapshot.subject_id`` is
    per-run and always present. The result-level field is preferred as the more
    specific one; the run's snapshot is the fallback. The fallback cannot
    misattribute, because it is read from the run that produced this very result.

    Callers differ in what they can offer, and the asymmetry is structural rather
    than an oversight: :func:`project_score_records` works per result and passes
    both, while :func:`compute_comparison_sets` groups *runs* and has no results to pass,
    so it always resolves through the snapshot. The two agree because both fields
    are written from the same capture — but if they ever disagreed, the row and
    the grouping would partition on different keys, which is why the preference
    order is defined in one place instead of at each call site.

    Args:
        run: The run supplying the snapshot fallback.
        result: The result whose attribution is preferred, when available.

    Returns:
        The subject key, never blank. ``SubjectSnapshot`` refuses a blank one, so the fallback
        below always answers — which is what retired the whole family of "excluded: no captured
        identity" counters this function used to feed. A blank subject was never a group of its
        own; it is now not a state.
    """
    attributed = (result.subject_id or "").strip() if result is not None else ""
    if attributed:
        return attributed
    return run.subject_snapshot.subject_id


class PlacedResult(NamedTuple):
    """One result matched to its run and its resolved subject, ready to group.

    The unit of :func:`place_results`. Carrying the ``run`` beside the result
    spares each caller a second lookup for the coordinates it reads off the run
    (subject name, template, ``created_at``, the frozen case set), and pins the
    ``subject_id`` the placement resolved so no caller re-derives it — resolving
    it twice is how two surfaces would drift on what "same subject" means.
    """

    run: EvalRun
    result: EvalResult
    subject_id: str


def place_results(
    runs: list[EvalRun],
    results: list[EvalResult],
    known_run_ids: set[str] | None,
    *,
    source: str,
    archived_run_ids: set[str] | None,
) -> tuple[list[PlacedResult], ProjectionExclusions]:
    """Match each result to its run and resolved subject, accounting for every drop.

    The prologue shared by :func:`project_score_records`, :func:`compute_frontier`, and
    :func:`compute_history`: a result cannot be placed without its run's coordinates, and
    a run with no captured subject identity cannot be grouped, because every
    consumer partitions on subject and a blank identity would pool unrelated
    subjects into one bucket. Both skips are **returned**, not merely logged —
    they are the two exclusions that turn a real corpus into an empty result set,
    so a surface that renders only the placed rows reports "nothing here" for data
    that exists and simply cannot be grouped.

    Extracted so the three surfaces cannot diverge on which observations they
    drop or how they classify the drop — an on-request narrowing must never be
    counted as an integrity gap, and vice versa. Callers apply their own further
    filtering (a subject restriction, a metric selection) over the placed rows.

    Args:
        runs: The runs supplying cell coordinates. Order is irrelevant.
        results: The observations to place. Output preserves this order.
        known_run_ids: Every run id that exists in the corpus, including ones a
            caller's own filter kept out of ``runs``. A result whose run is in
            this set but not in ``runs`` was excluded *on request* and is counted
            ``results_outside_queried_runs``; one whose run is in neither is
            genuinely unplaceable and counted ``results_without_run``. ``None``
            asserts ``runs`` is the whole corpus.
        source: The calling function's name, used only in the skip log line so a
            reader can tell which surface dropped rows.
        archived_run_ids: The corpus run ids the operator has archived. A result
            whose run is in this set is counted ``results_from_archived_runs``
            instead of ``results_outside_queried_runs`` — the two narrowings have
            different remedies, and only one of them is a wider ``status``.
            ``None`` asserts no archival narrowing was applied, which is correct
            only for a caller whose cohort already includes archived runs; it is
            never a licence to leave archival exclusions attributed to a filter
            that did not make them. Required with no default, here and on the
            three surfaces that forward it, so a caller that never thought about
            archival fails with ``TypeError`` instead of blaming ``status``.

    Returns:
        The placed results in input order — each carrying a non-blank
        ``subject_id`` — plus the exclusion accounting.
    """
    runs_by_id = {run.id: run for run in runs}

    placed: list[PlacedResult] = []
    skipped_no_run = 0
    skipped_outside_query = 0
    skipped_archived = 0

    for result in results:
        run = runs_by_id.get(result.eval_run_id)
        if run is None:
            if known_run_ids is not None and result.eval_run_id in known_run_ids:
                # Archival is checked FIRST, so a run that is both archived and
                # outside the caller's status narrowing is reported as archived:
                # un-archiving is the necessary step there, and a wider `status`
                # on its own would still not reach it.
                if archived_run_ids is not None and result.eval_run_id in archived_run_ids:
                    skipped_archived += 1
                else:
                    skipped_outside_query += 1
            else:
                skipped_no_run += 1
            continue

        placed.append(PlacedResult(run=run, result=result, subject_id=_subject_id_of(run, result)))

    if skipped_no_run:
        # Only the unplaceable class is logged. A result filtered out on
        # request is not an anomaly, and logging it would make the routine
        # `status="completed"` query emit a warning-shaped line every time.
        log.info("%s skipped rows: %d without a supplied run", source, skipped_no_run)
    return placed, ProjectionExclusions(
        results_without_run=skipped_no_run,
        results_outside_queried_runs=skipped_outside_query,
        results_from_archived_runs=skipped_archived,
    )


#: The types :func:`lever_level` renders through ``str()`` rather than as canonical JSON.
SCALAR_LEVEL_TYPES: tuple[type, ...] = (str, int, float, bool)

#: The level a lever sits at when the launch NAMED it with the value ``null`` — the operator set it
#: to nothing, which is a level, and is not the inherited ``'—'`` of a run that never named it.
#: JSON's own spelling, so it is the token the structured branch of :func:`lever_level` already
#: produced for ``None`` and the pivot and the export (which render every resolved value through
#: :func:`lever_level`) agree with the coverage map about it. Like every scalar it can collide with a
#: string level spelled the same (``"1024"`` and ``1024`` share one already); that is the rendering's
#: accepted limit, not a reason to spell a null as something JSON does not.
NULL_LEVEL = "null"


def lever_level(value: Any) -> str:
    """Render one lever's resolved value as the level a cohort is keyed on.

    A scalar renders as itself — ``1024``, not ``"1024"`` — matching the overlay spelling every
    surface downstream already carries. Anything structured renders as canonical JSON, which is a
    stable comparable key rather than a Python repr whose ordering could split two runs that
    resolved alike.

    **Content, never a display.** A host's own display for a level is more readable (``"20 chars ·
    3f9a1c"``) and is the wrong thing to bin on: a display is an abbreviation, two different levels
    can share one, and cohorts keyed on it would MERGE two variants — the direction nothing
    downstream can undo. :attr:`~threetears.evals.contracts.campaign.VariantIndexEntry.levers` carries
    the display beside the content hash for a reader who needs to recognise the level.

    **``None`` is** :data:`NULL_LEVEL`, **deliberately.** A caller hands one in only for a lever the
    launch named as ``null`` — a reader returning nothing is "the lever does not apply", and the
    caller skips it before reaching here — so the null is a level the operator set.

    Args:
        value: The resolved value, JSON-safe by the host's contract.

    Returns:
        The level.
    """
    if value is None:
        return NULL_LEVEL
    return str(value) if isinstance(value, SCALAR_LEVEL_TYPES) else canonical_json(value)


def _run_factors(
    placed: Sequence[tuple[EvalRun, EvalResult, str]], *, profile: HostProfile
) -> dict[str, dict[str, str]]:
    """Each placed run's open coordinates: every lever its host's registry resolves for it.

    Read through the registry and nowhere else — the fixed levers, the members of every open family
    the run carried, and a kind's overlays among them (each a lever its kind contract derives) — so a
    lever reaches a pivot and the export under the name the host gave it, with no per-lever work
    here. The candidate model is not among them: it is a declared coordinate of every record.

    Absence is preserved as absence: a lever whose reader returns nothing for this run (a lever of
    another kind, or one the run never recorded) has no entry, so a pivot groups those runs under the
    visible ``"—"`` level rather than inventing one.

    Resolved once per run, not per result: the host's reader pass is one unit of work, and asking it
    again per row would pay for it k × cases times for one answer.

    Args:
        placed: The ``(run, result, subject_id)`` triples being projected.
        profile: The host whose vocabulary this reads.

    Returns:
        Run id -> its coordinate map.
    """
    results_by_run: dict[str, list[EvalResult]] = {}
    runs: dict[str, EvalRun] = {}
    for run, result, _ in placed:
        runs.setdefault(run.id, run)
        results_by_run.setdefault(run.id, []).append(result)
    sweepables = profile.sweepables
    return {
        run_id: {
            lever: lever_level(value)
            for lever, value in sorted(sweepables.resolve_levers(run, results_by_run[run_id]).values.items())
            if lever != CANDIDATE_MODEL_LEVER and value is not None
        }
        for run_id, run in runs.items()
    }


def dim_judge_model(run: EvalRun, rubric_dim: str | None) -> str | None:
    """Name the model that scored ONE rubric dimension of one run.

    A lookup on the run's own attribution map, deliberately not a second
    evaluation of the judge-model cascade. The cascade is applied once, at launch,
    by :func:`~threetears.evals.contracts.models.resolve_effective_judges`, and its answer is
    stored on :attr:`~threetears.evals.contracts.models.EvalRun.effective_judges` — the same
    map the run summary's "Judged by" block renders and
    the ``judge_dim_divergence`` sweepable bisects on. Re-deriving it here
    would put a fourth reader on a rule that has already had three, and the whole
    point of storing the resolved attribution was that no reader has to re-run the
    cascade against today's configs.

    **Reads the HASHABLE map, so a reconstruction never answers.** A ``derived``
    attribution infers each dim's config from the records stored as of the run's
    start; it is displayable with a caveat, which is why
    a host's judge listing may show it, saying so. A pivot axis level and a CSV cell have no caveat channel — an inferred
    judge published there is indistinguishable from a recorded one, and a reader
    would group observations under a model that may never have scored them. The
    same reasoning keeps a reconstruction out of the context key and out of the
    comparison badge, and this function reads the same property they do rather
    than restating the eligibility rule.

    Args:
        run: The run that produced the observation.
        rubric_dim: The dimension being attributed, or ``None`` for a
            whole-result measure that spans every dim and has none.

    Returns:
        The model that scored ``rubric_dim``, or ``None`` when there is no single
        dimension to attribute, when the run recorded no attribution (or only a
        reconstruction), or when its recorded attribution has no entry for this
        dim — a dim the run never committed to a judge for is one nobody can
        name, and falling back to the run pin here is the exact
        misattribution this coordinate exists to end.
    """
    if rubric_dim is None:
        return None
    judges = run.hashable_effective_judges
    if judges is None:
        return None
    return judges.get(rubric_dim)


def project_score_records(
    runs: list[EvalRun],
    results: list[EvalResult],
    *,
    known_run_ids: set[str] | None = None,
    archived_run_ids: set[str] | None,
    profile: HostProfile,
) -> ScoreProjection:
    """Flatten runs + results into one row per (cell, measure).

    Results whose run is absent from ``runs`` are skipped — a row cannot be
    placed without its run's coordinates, and inventing them would fabricate a
    cell. Results whose run has no captured subject identity are also skipped,
    because every consumer of this projection partitions on subject and a blank
    identity would pool unrelated subjects into one bucket.

    Both skips are **returned**, not merely logged. A log line is invisible to
    the operator reading the answer, and these two exclusions are exactly the
    ones that turn a corpus into an empty result set — so a caller that renders
    the rows without the counts reports "nothing here" for data that exists and
    simply cannot be grouped.

    Args:
        runs: The runs supplying cell coordinates. Order is irrelevant.
        results: The observations to project.
        known_run_ids: Every run id that exists in the corpus, including ones a
            caller's own filter kept out of ``runs``. A result whose run is in
            this set but not in ``runs`` was excluded *on request* and is
            counted as ``results_outside_queried_runs``; one whose run is in
            neither is genuinely unplaceable and counted as
            ``results_without_run``. ``None`` asserts that ``runs`` is the whole
            corpus — correct for an unfiltered caller, and the only case where
            every absent run really is missing.
        archived_run_ids: The corpus run ids the operator has archived, so an
            archival exclusion is counted as ``results_from_archived_runs``
            rather than under the caller's ``status`` filter. See
            :func:`place_results`.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`ScoreProjection` — the rows, plus the exclusion counts and the
        completeness disclosures of every run that contributed a row while having
        measured less than its matrix. Every
        projected result contributes a row for every measure, carrying
        ``value=None`` when that measure has no value for it. A row therefore
        records *that the cell was attempted* independently of whether it was
        measured, which is what lets a consumer tell "tried and unmeasured" from
        "never tried". The metrics in :data:`SCOPED_METRICS` emit several rows for one result, each
        keyed by its scope coordinate. :data:`METRIC_SCORE` emits one per judged rubric dimension, keyed by
        :attr:`ScoreRecord.rubric_dim` and attributed to the model that scored
        *that* dimension on :attr:`ScoreRecord.rubric_dim_judge_model`, which is
        the run's judge pin only when the dimension carried no judge config of
        its own — and emits a single dimensionless null row when the judge scored
        none.
    """
    from threetears.evals.contracts.usage_capture import count_substituted_deliveries

    placed, exclusions = place_results(
        runs, results, known_run_ids, source="project_score_records", archived_run_ids=archived_run_ids
    )

    records: list[ScoreRecord] = []
    factors_by_run = _run_factors(placed, profile=profile)

    for run, result, subject_id in placed:
        coordinates: dict[str, Any] = {
            "subject_id": subject_id,
            "subject_label": run.subject_snapshot.subject_label,
            "scope_id": result.scope_id,
            "run_id": result.eval_run_id,
            "result_id": result.id,
            "template_id": run.template_id,
            "test_case_id": result.test_case_id,
            "model": result.model,
            "k_iteration": result.k_iteration,
            "variant_key": result.variant_key,
            "variant_identity_version": result.identity_version,
            "context_key": run.context_key,
            "context_identity_version": run.identity_version,
            "judge_model": run.judge_model,
            "simulator_model": run.simulator_model,
            "cassette_mode": run.cassette_mode,
            "substituted_deliveries": count_substituted_deliveries(result),
            "served_model": served_model_coordinate(served_reading(result)),
            "created_at": run.created_at,
            "factors": factors_by_run[run.id],
            "outcome": classify_result(result).value,
        }

        # Per-OBSERVATION composite, not the per-case mean over k. A row is one
        # measurement at one cell, and its coordinates carry `k_iteration` and
        # `result_id` — so writing a k-averaged value into each of them would
        # emit N identical rows claiming to be N distinct measurements, which
        # inflates every downstream `n` and drives within-case dispersion to
        # zero. Dispersion across k is precisely what the aggregation layer must
        # be able to compute from these rows.
        #
        # `result_composite` returns None twice over — for an infra-excluded
        # result, and for an ok result carrying no rubric dims. Both are
        # *unmeasured*, not zero, and the row is emitted anyway with a null
        # value. Aggregators drop nulls from the mean, so a harness failure
        # still cannot be absorbed into a quality figure; what the row adds is
        # the record that the cell was attempted at all.
        #
        # Emitting nothing was the earlier shape, and it made "tried and
        # unmeasured" indistinguishable from "never tried": a cell whose every
        # observation failed in the harness had no composite record, fell to the
        # empty-cell branch, and reported that the combination was never
        # attempted. It also silently zeroed the per-cell unmeasured count,
        # since a row that does not exist cannot be counted as dropped.
        # The basis travels with the number it qualifies, derived from the same
        # `rubric_scores` the composite is meaned over rather than from the
        # template's declared dims — a dim the template declares and this result
        # never scored is not in the mean, so naming it here would describe a
        # basis the arithmetic did not use.
        #
        # The host's own grade rides on THIS row, and the two are the same question asked of
        # two kinds of candidate: what this cell was worth. A judge-graded result answers it
        # in `value`; a code-graded one has no rubric dims to mean and answers it in
        # `host_measures`, which the export flattens to a column per measure. Carried
        # verbatim — an unregistered measure name is the registry's defect to report, not
        # this projection's to filter — and `{}` for every kind that grades with a judge, so
        # a judge-graded export gains no column at all.
        composite = result_composite(result)
        # A composite the judge could not complete is unmeasured for a reason of its own, and the
        # row says which, so a counter can tell it from a fault rather than from nothing.
        composite_coordinates = (
            {**coordinates, "outcome": JUDGE_CANNOT_TELL_OUTCOME}
            if trial_exclusion(result) == "judge_cannot_tell"
            else coordinates
        )
        records.append(
            ScoreRecord(
                metric=METRIC_COMPOSITE,
                value=composite,
                dimension_basis=composite_basis(result),
                host_measures=result.host_measures,
                **composite_coordinates,
            )
        )
        records.append(
            ScoreRecord(
                metric=METRIC_COST_USD, value=result.cost_usd, cost_roles=list(result.cost_roles), **coordinates
            )
        )

        # One row per judged dimension, carrying the RAW score on the dimension's own scale
        # (1-5, or 1/0 for pass/fail, named by `rubric_scale`) — not the 0-1 composite scale. Read straight off
        # `rubric_scores`, which holds the template dims alone: the reserved
        # `transcript_score` / `outcome_score` axes live in their own fields and
        # are emitted just below under METRIC NAMES of their own, so no reserved id
        # ever reaches `rubric_dim`.
        #
        # Excluding them is what lets these rows reproduce
        # `compute_dimension_summary`'s per-dimension means — under three
        # conditions, all of which are properties of the aggregation rather than
        # of these rows, and none of which this comment may assert away:
        # `compute_dimension_summary` divides by OBSERVATIONS, so the pivot
        # matches it under `sample_weighted` and not under the
        # `equal_per_scenario` default that gives each test case one vote; this
        # projection drops results it cannot place (see `exclusions`), so the two
        # agree only over a corpus where every result is placeable.
        #
        # **The value is what the dim counts as, from `counted_rubric_scores`** — the rule
        # `compute_dimension_summary` reads too. A harness-faulted result's rows are null (the
        # judge read a transcript the rig broke); a candidate failure's count at the
        # scale's floor (a turn the user got nothing from is not a
        # pass on any dimension, whatever the judge made of the silence). The judge's raw
        # reading stays on the result, where `get_result` shows it.
        #
        # Each dim row is also attributed to the model that scored THAT dim,
        # which the run-level `judge_model` in `coordinates` cannot express: the
        # pin is what a dim falls back to, and a dim whose `JudgeConfig` names a
        # model was scored by that instead. It is set per row rather than in
        # `coordinates` for the same reason `rubric_dim` is — the composite and
        # cost rows above span every dim and have no one judge to name.
        #
        # A result with no dims still emits one row, null-valued and with no
        # dimension. Same reason the composite row is emitted for an unmeasured
        # result: the row records that the cell was ATTEMPTED, and dropping it
        # would make "the judge scored no dimensions here" indistinguishable
        # from "nobody ran this combination".
        #
        # A dim the judge answered it could not tell on gets its own null row, outcome
        # ``judge_cannot_tell``: the trial is out of that dim's mean, and the row is how a
        # counter learns it is out and why, instead of the dim simply having fewer rows.
        cannot_tell_dims = [dim for dim in result.judge_cannot_tell if dim not in RESERVED_DIM_IDS]
        counted_by_dim = {score.dim: value for score, value in counted_rubric_scores(result) or ()}
        for rubric_score in result.rubric_scores:
            counted_value = counted_by_dim.get(rubric_score.dim)
            records.append(
                ScoreRecord(
                    metric=METRIC_SCORE,
                    rubric_dim=rubric_score.dim,
                    rubric_scale=rubric_score.scale,
                    rubric_dim_judge_model=dim_judge_model(run, rubric_score.dim),
                    value=None if counted_value is None else float(counted_value),
                    **coordinates,
                )
            )
        for dim in cannot_tell_dims:
            records.append(
                ScoreRecord(
                    metric=METRIC_SCORE,
                    rubric_dim=dim,
                    rubric_scale=run.rubric_scale(dim),
                    rubric_dim_judge_model=dim_judge_model(run, dim),
                    value=None,
                    **{**coordinates, "outcome": JUDGE_CANNOT_TELL_OUTCOME},
                )
            )
        if not result.rubric_scores and not cannot_tell_dims:
            records.append(ScoreRecord(metric=METRIC_SCORE, value=None, **coordinates))

        # One row per goal-state check the result carries, 1.0 passed and 0.0 not, keyed by the check —
        # including the template's checks a candidate-charged deadline stamps unevaluated, which count as
        # failed. No null row for a result that carries none: the rows above already say the cell was
        # attempted, and a check the result does not carry has no verdict to be null about. The value is the verdict
        # `counted_goal_verdicts` says every rate counts, which the bundle reads too: null for a
        # harness-faulted result (the check read the harness, so it stays out of every pass rate while the
        # row still records that it was attempted), and 0.0 for every check on a candidate failure.
        counted = counted_goal_verdicts(result)
        verdicts: list[tuple[GoalStateOutcome, float | None]]
        if counted is None:
            verdicts = [(goal, None) for goal in result.goal_state_outcomes]
        else:
            verdicts = [(goal, 1.0 if passed else 0.0) for goal, passed in counted]
        for goal, value in verdicts:
            records.append(
                ScoreRecord(
                    metric=METRIC_GOAL_STATE,
                    goal_check=goal.expression,
                    value=value,
                    **coordinates,
                )
            )

        # The dual-score axes. They live in their own fields rather than in
        # `rubric_scores`, so every export was structurally missing them — while
        # `list_metrics` publishes the pair as the one whose DIVERGENCE "separates a worse
        # agent from a changed world". An operator doing local analysis on the CSV had the
        # rubric dims and the composite and not that axis at all.
        #
        # EACH IS ITS OWN METRIC, and the reserved id never reaches `rubric_dim`. Two
        # recorded decisions meet here and this is the shape that honours both. The
        # `rubric_dim` coordinate is pinned to template dims (a test says so, and its
        # reason is that `compute_dimension_summary` does not aggregate these two, so a
        # reserved dim on that coordinate would make the two surfaces disagree about which
        # dims exist); and the `__` prefix on these ids exists, as `METRIC_SCORE`'s own
        # note says, precisely so they can live in the metric namespace without colliding
        # with a template dim. So a pooled `score` cell still averages template dims alone,
        # and the axes are addressable directly rather than as two of N levels on a
        # coordinate — which is what asking about their divergence actually needs.
        #
        # NO NULL ROW when an axis was not scored, unlike the dims above. That null row
        # exists to say the cell was ATTEMPTED and the judge scored no dimension; two more
        # nulls would say the same thing twice over, on axes the first row already covers.
        #
        # `dim` is read off the score rather than paired with a constant here: the judge
        # service stamps the reserved id it scored under, so a row can only be keyed by the
        # axis that produced it. It carries `rubric_dim_judge_model` even though
        # `rubric_dim` is None — see that field, where the rule is "no attribution where no
        # SINGLE model scored the row", which a per-axis judge answers.
        for reserved in (result.transcript_score, result.outcome_score):
            if reserved is None:
                continue
            # `RubricScore.dim` is a stored free-text field, and this is the one row whose
            # `metric` comes off stored data rather than off a module constant. Guarded so
            # the closed-set claim above `PROJECTED_METRICS` stays true by construction:
            # a mis-stamped or hand-written row would otherwise export a metric column
            # value `pivot` refuses as unknown and no descriptor describes.
            #
            # LOUD, because the absence it leaves already means something else. An unjudged
            # result gains no rows here either, so a silent skip would make a mis-stamped
            # axis read exactly like an unjudged one — the
            # make-an-absence-ambiguous failure this module refuses everywhere else, and
            # worse than the unguarded state, where the odd metric value was at least
            # visible and `pivot` refused it by name.
            if reserved.dim not in RESERVED_DIM_IDS:
                log.warning(
                    "eval.reporting dropping a dual-score row for result=%s run=%s: dim %r is not a reserved axis "
                    "(%s). The axis was scored and is NOT in this projection — its absence here is a defect in "
                    "what stamped it, not a result that went unjudged.",
                    result.id,
                    run.id,
                    reserved.dim,
                    ", ".join(sorted(RESERVED_DIM_IDS)),
                )
                continue
            # What the axis counts as, by the rule every judged score follows (`counted_score`):
            # null for a harness fault, the scale's floor for a candidate failure.
            counted_axis = counted_score(result, reserved)
            records.append(
                ScoreRecord(
                    metric=reserved.dim,
                    rubric_dim_judge_model=dim_judge_model(run, reserved.dim),
                    value=None if counted_axis is None else float(counted_axis),
                    **coordinates,
                )
            )
        # The one null axis row: an axis the judge answered it could not tell on. Unlike an
        # unscored axis this is a fact about the axis alone, and its count is owed wherever the
        # axis's exclusions are counted.
        for axis in sorted(RESERVED_DIM_IDS & set(result.judge_cannot_tell)):
            records.append(
                ScoreRecord(
                    metric=axis,
                    rubric_dim_judge_model=dim_judge_model(run, axis),
                    value=None,
                    **{**coordinates, "outcome": JUDGE_CANNOT_TELL_OUTCOME},
                )
            )

    # Over the runs that PLACED a row, not over every run supplied: a run whose
    # results were all excluded contributes nothing this table averages, and a
    # disclosure about it would qualify a population the answer does not contain.
    placed_run_ids = {placement.run.id for placement in placed}
    return ScoreProjection(
        records=records,
        exclusions=exclusions,
        completeness_disclosures=degraded_run_disclosures(run for run in runs if run.id in placed_run_ids),
    )


class CaseSetIdentity(EvalBaseModel):
    """One distinct case set inside a group, and which runs executed it.

    A ``template_id`` does not establish case-set identity while templates are
    mutable in place: a template can gain or lose cases between runs, so two runs
    of "the same suite" may have scored different denominators. The identity is
    already computed — it is ``ContextComponents.case_basis``, the digest of
    ``(template_id, sorted test_case_ids)`` stamped on every run at launch — but
    nothing rendered it, so the fact was recorded and unreadable.

    ``n_cases`` is the size of the set this fingerprint stands for, which is the
    number the group's ``shared_test_case_ids`` intersection cannot give you: an
    intersection of 3 is equally consistent with runs of 3 cases and runs of 31.
    """

    #: Full ``case_basis`` digest — the run's own stamped value where present,
    #: re-derived from its stored fields otherwise (a run its host assembled without
    #: the launch), so every run is identified on the same predicate. ``None`` when the run's basis could
    #: not be resolved at all: the entry then stands for that ONE run (``run_ids`` has
    #: one member), because an unresolved basis says the set is unknown and merging
    #: unknowns would assert a sameness nothing established. It is a state, not a
    #: digest, so a renderer branches on ``None`` rather than displaying anything.
    fingerprint: str | None
    n_cases: int
    run_ids: list[str]


# =============================================================================
# Measurement windows — WHEN a run was measured, and whether two runs overlap
# =============================================================================


class MeasurementWindow(EvalBaseModel):
    """The wall-clock span a run's cells were actually measured over — DERIVED.

    **Deliberately not called a "window" bare**, because that word is already
    taken one level up: ``CampaignWindow`` is a campaign's span, derived from its
    member runs' ``created_at``. This is a different quantity on a different
    basis, and the two disagree by hours on the same data, so they carry different
    names and different derivations rather than one name meaning two things.

    **The basis is every result's ``scored_at``, never the run's
    ``created_at``/``completed_at``.** A run document is saved — stamping
    ``created_at`` — *before* its task is created, and the task then waits on a
    semaphore that admits two jobs at a time, so ``created_at`` is when the run
    was enqueued and not when anything was measured. The gap is not academic: a
    ten-arm sweep whose arms were enqueued within half a minute of each other
    executed across four hours, so on a ``created_at`` basis every arm's span
    contains every other arm's and no pair can ever read as disjoint. A window
    derived that way would be silent on precisely the runs it exists to describe.
    ``scored_at`` is stamped when a cell's result object is built, at the end of
    that cell, which makes min/max over a run's results the span its cells really
    occupied.

    Both bounds are the ISO-8601 UTC strings the models stamp, compared as strings
    throughout: every eval timestamp comes from one stamper that always emits a
    ``+00:00`` offset, so lexicographic order is chronological order. Nothing here
    parses a datetime, which is what keeps a malformed stamp from raising on a
    read surface.
    """

    run_id: str
    #: Earliest ``scored_at`` among the run's results.
    start: str
    #: Latest ``scored_at`` among the run's results.
    end: str


def measurement_window(run_id: str, results: Sequence[EvalResult]) -> MeasurementWindow | None:
    """Derive one run's measurement window from the results it produced.

    Args:
        run_id: The run the window describes.
        results: That run's stored results. Callers filter to one run; nothing
            here checks ``eval_run_id``, so passing another run's results
            produces a window describing a population that never existed.

    Returns:
        The window, or ``None`` when the run produced no results. ``None`` is an
        honest absence and never a zero-length span at some arbitrary instant: a
        run that measured nothing did not measure it "at" a time, and a synthetic
        point would compare against other windows as though it had.
    """
    stamps = [r.scored_at for r in results if r.scored_at]
    if not stamps:
        return None
    return MeasurementWindow(run_id=run_id, start=min(stamps), end=max(stamps))


class WindowGap(NamedTuple):
    """One pair of runs whose measurement spans do not overlap, and how far apart they were.

    The pair is oriented — ``earlier`` finished before ``later`` began — so the gap
    is a forward duration rather than a signed difference a reader has to interpret.
    """

    earlier: MeasurementWindow
    later: MeasurementWindow
    #: Seconds between ``earlier.end`` and ``later.start``, or ``None`` when the
    #: recorded stamps cannot be read as instants. See :func:`_gap_seconds`.
    seconds: float | None


def _gap_seconds(earlier_end: str, later_start: str) -> float | None:
    """How long a run's span sat idle before the next one began, if that is knowable.

    **The lexicographic discipline is unchanged and this does not weaken it.** Ordering
    and the disjointness predicate still compare the ISO-8601 strings directly, so a
    malformed stamp cannot make a comparison raise. What is added is a *guarded*
    parse used for magnitude only: a stamp that will not parse, or a pair that mixes an
    offset-aware stamp with a naive one, yields no duration and the disclosure says so
    rather than failing.

    Args:
        earlier_end: The last ``scored_at`` of the run that finished first.
        later_start: The first ``scored_at`` of the run that began after it.

    Returns:
        The non-negative gap in seconds, or ``None`` when it cannot be computed
        honestly — an unparseable stamp, a naive/aware mix, or a parsed order that
        contradicts the lexicographic one (which means the two stamps are not on a
        shared clock and no duration between them is meaningful).
    """
    try:
        end = datetime.fromisoformat(earlier_end)
        start = datetime.fromisoformat(later_start)
    except TypeError, ValueError:
        # NOSILENT: None IS the report -- format_window_gap renders it as the uncomputable-gap clause
        return None
    try:
        delta = (start - end).total_seconds()
    except TypeError:
        # NOSILENT: One stamp carried a UTC offset and the other did not. Subtracting those is
        # not a duration anybody measured, so it is reported as uncomputable (None renders as
        # the uncomputable-gap clause).
        return None
    return delta if delta >= 0 else None


class OverlappingWindows(NamedTuple):
    """One pair of runs whose measurement spans DID overlap.

    Unoriented, unlike :class:`WindowGap`: there is no earlier and later to name when
    two spans share wall-clock time, and no duration between them to report. It carries
    the two windows and nothing else, because the only claim it supports is that these
    two runs were not measured apart.
    """

    first: MeasurementWindow
    second: MeasurementWindow


class WindowPairs(NamedTuple):
    """Every pair of a run set, split by whether the two spans overlapped.

    Both halves come off one enumeration. The disclosure counts the disjoint pairs
    against the total, and the total includes the overlapping half. Complementing one half
    somewhere else would put the overlap predicate in two places, which is the drift this
    type exists to prevent.
    """

    disjoint: list[WindowGap]
    overlapping: list[OverlappingWindows]

    @property
    def total(self) -> int:
        """How many pairs the run set has.

        Returns:
            The P a partial disclosure counts its disjoint pairs against.
        """
        return len(self.disjoint) + len(self.overlapping)


def classify_window_pairs(windows: Sequence[MeasurementWindow]) -> WindowPairs:
    """Split every pair of these runs by whether their measurement spans overlapped.

    The single derivation behind both the predicate and the prose: the badge, the
    quantifier the sentence states and the magnitudes it names are all read off this one
    enumeration, so they cannot come apart about which pairs are disjoint.

    Only windows that RESOLVED are passed in, and callers must keep it that way. A run
    with no results has no window, and letting its absence contribute would report a
    difference on the strength of what one run could not say — the same rule the
    case-set and attribution comparisons follow.

    **Any pair, not all pairs**, and the prose a caller renders from this must state
    that quantifier and not a stronger one. Three arms where two ran together and the
    third ran the next morning are still a set nothing held fixed across, so requiring
    every pair to be disjoint would report nothing for the common sweep shape of a few
    arms at a time — but printing the existential as "these runs were measured over
    non-overlapping spans" is the falsehood the generator escalated into a false
    headline. Fewer than two windows yields both halves
    empty: one run cannot be measured apart from itself, and zero runs assert nothing.

    Bounds are treated as closed and touching counts as overlap: two runs where one's
    last cell and the other's first share an instant were running against the same
    conditions, and the whole point of the question is whether they were.

    Args:
        windows: The resolved windows of the runs being compared.

    Returns:
        The pairs, split. Both halves are in the order the pairs are enumerated from
        ``windows``, and both are empty for fewer than two windows — one run cannot be
        measured apart from itself, and it cannot overlap itself either.
    """
    # Checked exhaustively over every pair. A sorted single pass is the shape that
    # answers "do ALL of them overlap", which is a different question and the one
    # a comparison does not want asked; over the handful of runs a comparison set
    # holds, the pairwise loop costs nothing and says what it means.
    disjoint: list[WindowGap] = []
    overlapping: list[OverlappingWindows] = []
    for i, a in enumerate(windows):
        for b in windows[i + 1 :]:
            if a.end < b.start:
                earlier, later = a, b
            elif b.end < a.start:
                earlier, later = b, a
            else:
                overlapping.append(OverlappingWindows(first=a, second=b))
                continue
            disjoint.append(WindowGap(earlier=earlier, later=later, seconds=_gap_seconds(earlier.end, later.start)))
    return WindowPairs(disjoint=disjoint, overlapping=overlapping)


def disjoint_window_pairs(windows: Sequence[MeasurementWindow]) -> list[WindowGap]:
    """Every pair of these runs whose measurement spans do not overlap, with its gap.

    The disjoint half of :func:`classify_window_pairs`, which holds the predicate and
    the reasoning. Kept as its own name because the disclosure and the badge ask only
    this question, and reading ``.disjoint`` at every call site would say less.

    Args:
        windows: The resolved windows of the runs being compared.

    Returns:
        One :class:`WindowGap` per non-overlapping pair, in the order the pairs are
        enumerated from ``windows``. Empty when every pair overlaps, and for fewer than
        two windows — one run cannot be measured apart from itself.
    """
    return classify_window_pairs(windows).disjoint


# The distinguishing clause of the disjoint-window disclosure, split out for the
# same reason :data:`DEGRADED_RUN_CLAUSE` is: a test pins the CLAUSE, so the
# sentence around it stays free to be rewritten for a reader.
DISJOINT_WINDOWS_CLAUSE = "measured over non-overlapping spans"


#: Above this many windows the disclosure summarises instead of listing every span.
#:
#: The listing form is the better disclosure and stays the default for the sizes an
#: operator actually reads. It stops being a disclosure at scale: a 27-run group
#: renders every span inline as one unbroken paragraph, and the sentence directing the
#: reader to "read it rather than the badge" becomes advice nobody can take. Four keeps
#: the pairwise and small-sweep cases — the ones where every span is the point —
#: verbatim.
MAX_INLINE_MEASUREMENT_WINDOWS = 4

#: How many non-overlapping pairs the disclosure names before it counts the rest.
#:
#: Six is exactly the pair count of :data:`MAX_INLINE_MEASUREMENT_WINDOWS` runs, so the
#: sizes an operator actually reads never truncate. The cap exists for the other end: a
#: ``full=true`` read of a 22-run campaign has 231 pairs, and a paragraph of them is the
#: same non-disclosure the uncapped span list was.
MAX_RENDERED_WINDOW_GAPS = 6

#: What a gap renders as when the recorded stamps cannot be read as instants. Said
#: rather than omitted: a pair silently missing its magnitude reads as a pair with no
#: gap, which is the opposite of what it means.
UNCOMPUTABLE_GAP_CLAUSE = "gap not computable from the recorded stamps"


def format_window_gap(seconds: float | None) -> str:
    """Render a gap between two measurement spans as a magnitude, in its own units.

    Two significant units and no unit words: ``50m31s``, ``1h04m``, ``5d23h``. The
    compact form is deliberate — this is a magnitude the reader weighs, not a sentence,
    and it sits inside a list of pairs where spelled-out units would bury the numbers.

    **It states the size and never what the size means.** Each disclosure computes a magnitude
    in the measure's own units; deciding whether that magnitude is material is the descriptor's
    threshold to declare and is not this function's.

    Args:
        seconds: The gap, or ``None`` when it could not be computed.

    Returns:
        The magnitude, or :data:`UNCOMPUTABLE_GAP_CLAUSE` when there is none.
    """
    if seconds is None:
        return UNCOMPUTABLE_GAP_CLAUSE
    total = round(seconds)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _gap_phrase(gap: WindowGap) -> str:
    """Name one non-overlapping pair and how far apart it was."""
    if gap.seconds is None:
        return f"{gap.earlier.run_id} and {gap.later.run_id}, {UNCOMPUTABLE_GAP_CLAUSE}"
    return f"{gap.earlier.run_id} and {gap.later.run_id}, {format_window_gap(gap.seconds)} apart"


def _widest_gaps_first(pairs: Sequence[WindowGap]) -> list[WindowGap]:
    """Order non-overlapping pairs so the biggest gap is read first.

    Widest first because that is the pair a reader most needs to see, and because
    truncating the tail then drops the least, never the most. Pairs whose gap could not
    be computed sort last — they are an admission rather than a measurement, and putting
    an admission where the widest gap belongs would read as a ranking. Ties break on run
    id so one set of windows always renders one way.
    """
    return sorted(
        pairs,
        key=lambda p: (0 if p.seconds is not None else 1, -(p.seconds or 0.0), p.earlier.run_id, p.later.run_id),
    )


def measurement_window_disclosure(windows: Sequence[MeasurementWindow], *, full: bool = False) -> str | None:
    """The sentence a comparison must carry when its runs did not share a clock.

    Descriptive, never a verdict: it states when each run was measured, which pairs did
    not overlap and by how much, and stops there. No threshold decides how far apart is
    too far, no severity is assigned, and nothing is corrected — how much a gap matters
    depends on what else moved in it, which this surface cannot see and the operator can.

    **It states the quantifier it can defend.** The condition is existential — see
    :func:`disjoint_window_pairs` — so the universal form ("these runs were measured over
    non-overlapping spans") is printed only when every pair really is disjoint. Otherwise
    the sentence counts: *D of the P pairs among these N runs*, with the remainder named
    as overlapping and the closing attribution scoped to the pairs that earned it.
    Rendering the existential as a universal is the defect this exists for — on the
    campaign that motivated the fix, three of six pairs overlapped while the sentence
    denied it, and the generator escalated the denial into a false headline.

    **Each non-overlapping pair carries its gap magnitude**, widest first, because
    a disclosure computes a magnitude in the measure's own
    units. An earlier form deliberately reported bounds and no durations, on the grounds
    that no read surface may parse a datetime; that guarantee is kept — ordering and the
    overlap predicate are still purely lexicographic — and only the magnitude is parsed,
    under a guard that yields :data:`UNCOMPUTABLE_GAP_CLAUSE` rather than raising.

    Above :data:`MAX_INLINE_MEASUREMENT_WINDOWS` the span list is replaced by the group's
    outer bounds and its two extreme windows, and the gap list collapses to the widest
    pair; both say how much they did not name and where to get it.

    Args:
        windows: The resolved windows of the runs being compared.
        full: List every span regardless of count. For surfaces whose consumer
            can collapse the text itself, and for an operator who asked. The pair
            list still caps at :data:`MAX_RENDERED_WINDOW_GAPS`, because pairs grow
            quadratically where spans grow linearly.

    Returns:
        The disclosure, rendered verbatim by every surface, or ``None`` when every
        pair of runs overlaps or there are too few windows to ask.
    """
    pairs = disjoint_window_pairs(windows)
    if not pairs:
        return None
    ordered = sorted(windows, key=lambda w: (w.start, w.end))
    n_runs = len(ordered)
    n_pairs = n_runs * (n_runs - 1) // 2
    inline = full or n_runs <= MAX_INLINE_MEASUREMENT_WINDOWS
    if inline:
        detail = "; ".join(f"{w.run_id} {w.start} to {w.end}" for w in ordered)
    else:
        first, last = ordered[0], ordered[-1]
        # The lower bound is `first.start` by construction — the sort is by start.
        # The upper bound is NOT `last.end` for the same reason: sorting by start
        # says nothing about which member ends last, so a run that began early and
        # overran holds the latest end while sitting first. Taking `last.end` would
        # understate the group's span in exactly that case.
        detail = (
            f"{n_runs} runs, measured between {first.start} "
            f"and {max(w.end for w in ordered)}; earliest {first.run_id} {first.start} to {first.end}; "
            f"latest {last.run_id} {last.start} to {last.end}; "
            f"{n_runs - 2} further span(s) not shown — pass full=true for every span"
        )

    widest_first = _widest_gaps_first(pairs)
    if inline:
        shown = widest_first[:MAX_RENDERED_WINDOW_GAPS]
        gap_detail = "; ".join(_gap_phrase(gap) for gap in shown)
        withheld = len(pairs) - len(shown)
        if withheld:
            gap_detail += f"; +{withheld} further non-overlapping pair(s), all narrower than these"
        gaps = f"Non-overlapping pairs, widest first: {gap_detail}."
    else:
        gaps = f"Widest non-overlapping pair: {_gap_phrase(widest_first[0])}"
        if len(pairs) > 1:
            gaps += f"; {len(pairs) - 1} further non-overlapping pair(s) not named"
        gaps += "."

    if len(pairs) == n_pairs:
        lead = f"These runs were {DISJOINT_WINDOWS_CLAUSE} of wall-clock time — every pair of them ({detail})."
        subject = "a difference between them"
    else:
        lead = (
            f"{len(pairs)} of the {n_pairs} pairs among the {n_runs} runs named here were "
            f"{DISJOINT_WINDOWS_CLAUSE} of wall-clock time; the other {n_pairs - len(pairs)} overlap ({detail})."
        )
        subject = "a difference between the two runs of a non-overlapping pair"
    return (
        f"{lead} {gaps} "
        "Anything that changed on the machine, the providers or the account between those spans "
        f"varies with the runs, so {subject} is not attributable to the runs alone."
    )


# The distinguishing clause of the cassette-span disclosure, split out for the same
# reason :data:`DISJOINT_WINDOWS_CLAUSE` is: a test pins the CLAUSE, so the sentence
# around it stays free to be rewritten for a reader.
CASSETTE_SPAN_CLAUSE = "did not all record the same cassette mode"

#: The one cassette mode that SUBSTITUTES third-party output. ``off`` and ``capture``
#: both run the third party live — capture additionally records what came back — so a
#: span across those two is a difference in recording, not in measurement. Naming the
#: substituting mode once keeps the disclosure's two branches from drifting apart.
SUBSTITUTING_CASSETTE_MODE = "replay"


def cassette_mode_disclosure(modes_by_run: Mapping[str, str]) -> str | None:
    """The sentence a comparison must carry when its runs recorded different cassette modes.

    **One vocabulary, two layers.** ``cassette_mode`` is already a declared apparatus
    confound in the host's sweepable declarations — "tool output was
    live for some of these runs and replayed for others, which changes both latency and
    content" — which is what ``bisect_runs`` and the analysis bundle's confound scan read,
    and the generated analysis already names it correctly. The badge derived from this
    sentence is therefore ``cassette_mode_differs``: the declared dimension name plus the
    suffix its sibling ``tool_config_differs`` already uses, so the reporting layer and the
    analysis layer name one thing once. What follows is the same claim said in the detail a
    comparison surface needs and a confound catalogue does not carry.

    Reports what the runs RECORDED and says so in those words, because nothing on a run
    corroborates the mode it recorded. Three candidate corroborators were checked and each
    is the same claim one level down or weaker than it: :attr:`~threetears.evals.contracts.models.EvalRun.cassette_corpus_id` is
    set exactly when the run claims replay, and
    :func:`~threetears.evals.contracts.usage_capture.count_substituted_deliveries` counts seeded case
    findings alongside replayed cassettes, so a non-zero count does not evidence replay
    and a zero one cannot separate "did not replay" from "replayed a template with no
    async delivery". So this asserts only the record and names the ambiguity, the choice
    the run-detail badge already makes.

    **Two spans, one badge.** Only ``replay`` substitutes: an arm that replayed re-served
    a recording rather than measuring the third party, which confounds its quality — a
    replayed arm is served only the asks its capture made (any other ask stops the cell), so
    it is measured on the questions the capturing candidate chose. **Its cost is confounded at
    one of the two seams, not both**, which is why the sentence names them separately: a
    replayed background DELIVERY spent no inner-agent dollars and its
    production-replicating cost is withheld, while a replayed tool ACTION (a search, say) saved
    provider credits that never entered the per-role rows on a live arm either, so its cost
    is reported and is the same figure the live arm would report. Saying "a replayed arm's
    cost is withheld" flatly was wrong for the second seam and told a reader to expect a
    blank where a number correctly appears — see
    :func:`~threetears.evals.contracts.usage_capture.production_replicating_cost`. A span that stays
    within ``off``/``capture`` is a difference in what was *recorded* while both arms ran
    live. Both are disclosed, because both are a difference in a condition that should have
    been holding still; the branch decides only what the sentence says the difference costs.

    Descriptive, never a verdict, on the rule :func:`measurement_window_disclosure`
    follows: no comparison is refused, no number is adjusted and no severity is assigned.
    Capture-versus-replay is a legitimate thing to run.

    Args:
        modes_by_run: One recorded ``cassette_mode`` per run being compared, keyed by run id.

    Returns:
        The disclosure, rendered verbatim by every surface, or ``None`` when fewer than
        two runs were supplied or every run recorded the same mode. Two arms that both
        replayed are uniform and disclose nothing: the substitution is on both sides, so
        the delta between them is honest.
    """
    if len(modes_by_run) < 2:
        return None
    recorded = set(modes_by_run.values())
    if len(recorded) < 2:
        return None
    detail = "; ".join(f"{run_id} recorded {mode}" for run_id, mode in sorted(modes_by_run.items()))
    if SUBSTITUTING_CASSETTE_MODE in recorded:
        cost = (
            "A replayed arm did not measure the third party — it re-served a recording. Where it "
            "replayed a background DELIVERY its production-replicating cost is withheld as unknown "
            "rather than reported smaller; where it only replayed a tool ACTION the cost is "
            "still reported, and is not understated, because that tool's spend never reaches the "
            "per-role rows on a live arm either — what replay saved there is search credits, not "
            "dollars. Its QUALITY is confounded either way: a replayed arm is served only the asks its "
            "capture made, so it is measured on the questions the capturing candidate chose to ask."
        )
    else:
        cost = (
            "Both 'off' and 'capture' run the third party live — capture additionally records what "
            "came back — so this is a difference in what was recorded rather than in what was measured."
        )
    return f"These runs {CASSETTE_SPAN_CLAUSE} ({detail}). {cost}"


#: The one origin value that means *this launch said so*. The other tiers — a value the
#: subject carried, or one inherited from a system default — supplied it from below the launch,
#: so a difference across them is one nobody asked this comparison for.
#:
#: The vocabulary of origins is a host's; what the engine owns is the RULE beneath it, and the
#: surface that renders a particular lever's sentence lives with that lever's vocabulary, in the host.
DECLARED_INPUT_ORIGIN = "chosen"


def difference_was_declared_at_launch(origins: Iterable[str | None]) -> bool:
    """Whether every one of these runs got its value from its own launch declaration.

    **The generic half of a lever-origin disclosure, and the part that stayed.** The rule is
    not about any one lever: a difference between arms is a *swept axis* when each arm's launch
    declared the value, and a *confound* when any arm inherited it from a tier the comparison
    never named. Only the vocabulary (``chosen`` / a subject-carried tier / an inherited
    default) is a host's; the rule is the same one a campaign's declared design will state
    formally once it declares one, at which point this reads the declaration instead of
    the origins. A host calls it to write, for example, the sentence about which model its
    arms delegated background work to.

    Conservative by construction: one arm that inherited its value is enough to make the
    whole difference undeclared, because the comparison cannot claim to be sweeping an
    axis one of its arms was never pointed at.

    Args:
        origins: One origin per compared run, as the run recorded it. ``None`` means the
            run recorded no origin and therefore declared nothing that can be read.

    Returns:
        ``True`` when every origin is :data:`DECLARED_INPUT_ORIGIN`. ``False`` for an
        empty set — nothing declared anything — so a caller can never read silence as a
        declaration.
    """
    listed = list(origins)
    return bool(listed) and all(origin == DECLARED_INPUT_ORIGIN for origin in listed)


class ComparisonSet(EvalBaseModel):
    """A group of runs that may be compared with each other, and on what basis.

    ``badges`` names every respect in which the group is *less* than fully
    comparable. An empty list means the group shares template, subject, context
    key, and test-case set — the only case where run-vs-run subtraction needs no
    caveat.

    ``shared_test_case_ids`` is the INTERSECTION across the group, and on its own
    it misleads in exactly the direction that matters: a sweep whose runs
    report 2 shared cases can have runs that each executed 6, and the intersection
    was small *because* the sets differed. ``case_sets`` names each distinct set
    and who ran it, so the reader sees how the group is split rather than only
    what survives the overlap.
    """

    subject_id: str
    subject_label: str = ""
    template_id: str | None = None
    run_ids: list[str]
    shared_test_case_ids: list[str]
    #: Distinct case sets within the group, most-used first. More than one entry is
    #: the ``case_set_differs`` badge's substance — the badge says *that* they
    #: differ, these say *how*.
    case_sets: list[CaseSetIdentity] = []
    badges: list[str]
    #: The ``measurement_windows_disjoint`` badge's substance: the sentence naming
    #: when each run was measured. The badge says the group did not share a clock;
    #: only this says whether that means four minutes or four hours, which is the
    #: whole of the judgement being handed to the reader. ``None`` whenever the
    #: badge is absent — the two are derived from one predicate and cannot disagree.
    measurement_window_disclosure: str | None = None
    #: The ``cassette_mode_differs`` badge's substance: the sentence naming what each run
    #: recorded and what the span costs. The badge says the group's arms did not all
    #: record one mode; only this says whether that means a recording difference between
    #: two live arms or a replayed arm that never measured the third party at all.
    #: ``None`` whenever the badge is absent — one predicate, so the two cannot disagree.
    cassette_mode_disclosure: str | None = None
    #: The runs that share this group's (subject, template) and that the caller's scope
    #: left out. Empty when unscoped, which is every group's ordinary state.
    #:
    #: A scope makes the badges honest about the set under analysis, and in doing so it
    #: hides how much else the group holds — so a reader meeting a badge-clean scoped
    #: group cannot tell whether the group really is three replicates or three of seven runs
    #: spanning several days. Naming the excluded runs is what keeps the narrowing visible:
    #: the argument is that a set quietly reduced reads identically to a set that was always
    #: that size.
    out_of_scope_run_ids: list[str] = []


# Badge vocabulary. ``context_differs`` extends the existing ``comparison_basis``
# vocabulary used by run comparison; ``case_set_differs`` records the drift that
# makes a longitudinal series dishonest when a suite silently gained or lost
# cases between runs.
BADGE_CONTEXT_DIFFERS = "context_differs"
BADGE_CASE_SET_DIFFERS = "case_set_differs"
# The pinned roles: models held fixed so the candidate is the only thing varying.
# When they differ between runs, a quality delta is confounded by a judge or
# simulator change and the comparison measures two things at once.
BADGE_ROLES_DIFFER = "roles_differ"
# At least one run's measurement context could not be fully reconstructed — a role pin
# it never recorded. Distinct from every other badge here, which reports a difference
# that WAS observed: this one reports that the comparison cannot be decided. It has to
# exist separately because an unrecorded pin makes the other badges come out CLEAN —
# two runs that both stored a blank compare equal on the blank — so the group would
# otherwise render as the one case needing no caveat at all.
BADGE_CONTEXT_INCOMPLETE = "context_incomplete"
# At least one run's case basis could not be resolved. The sibling of
# ``context_incomplete``, one axis over, and separate from ``case_set_differs`` for
# the reason that badge cannot carry: an unresolved basis means the sets are
# UNKNOWN, not known-different. Asserting a difference we did not observe is the
# same dishonesty as asserting a match. See ``CaseSetIdentity.fingerprint`` for
# why such runs never merge into one entry.
BADGE_CASE_SET_UNRESOLVED = "case_set_unresolved"
# The candidate's own resolved tool config (search depth, token budgets, call caps,
# the inner-agent model). Unlike the three above it is not a caveat on the
# comparison — sweeping it is usually the POINT — but it has to be said out loud,
# because it is invisible in the one place an operator looks for it: tool config
# composes the VARIANT key, so runs that differ across it still share a context key
# and, badged only on the conditions, render as one configuration — runs differing
# in three tool caps at once can read as one.
BADGE_TOOL_CONFIG_DIFFERS = "tool_config_differs"
# At least one pair of the group's runs was measured over spans that do not
# overlap, so anything that moved on the box, the providers or the account between
# those spans varies with the runs. It is the one badge here whose evidence is not
# in the run documents at all — it is derived from the results' ``scored_at``, so a
# group whose results were not supplied cannot be asked and is silent rather than
# clean. Descriptive only: how much a gap costs depends on what else changed in it,
# which this surface cannot see and the operator can.
BADGE_MEASUREMENT_WINDOWS_DISJOINT = "measurement_windows_disjoint"
# The group's runs did not all record the same cassette mode. It sits with
# ``roles_differ`` and ``tool_config_differs`` as a per-component badge over a condition
# the context key already hashes but cannot name: ``cassette`` is a context component
# (``identity.py``), so a capture-versus-replay pair fires ``context_differs`` — "something
# about the conditions moved" — and the operator is left to bisect which. That is the exact
# opacity these badges exist to remove, and it went unnamed on the one condition where the
# arms may not have measured the same thing at all. Like the disjoint-window badge it is
# gated on its DISCLOSURE rather than on a predicate of its own, so the flag can never
# appear with no sentence naming the modes behind it: the flag cannot tell an
# ``off``/``capture`` recording difference from a live-versus-replayed substitution, and
# that distinction is the whole of what a reader needs.
BADGE_CASSETTE_MODE_DIFFERS = "cassette_mode_differs"


class ComparisonSetsResult(EvalBaseModel):
    """Comparability groups plus the runs the caller's own scope left out.

    There is no blank-subject exclusion count, and its absence is the point: every run carries a
    non-blank subject key, so "a scope whose runs cannot be grouped" is not a state. A counter
    that could only ever read zero is worse than none — a reader takes zero as evidence.

    ``out_of_scope_run_ids`` names every supplied run the scope left out, so a reader meeting a
    small clean group can tell whether it is the whole story. Empty when unscoped.
    """

    comparison_sets: list[ComparisonSet] = []
    out_of_scope_run_ids: list[str] = []


def compute_comparison_sets(
    runs: list[EvalRun],
    *,
    results: Sequence[EvalResult] = (),
    full_windows: bool = False,
    scope_run_ids: Iterable[str] | None = None,
    profile: HostProfile,
) -> ComparisonSetsResult:
    """Group runs into sets that may honestly be compared, badging every caveat.

    Runs are grouped by (subject, template): those are the two coordinates whose
    difference makes a comparison meaningless rather than merely noisy. Within a
    group, differences that weaken — but do not invalidate — the comparison are
    reported as badges, never by silently dropping runs.

    **A badge speaks for the runs it was computed over, and a reader is usually
    asking about a smaller set than the group holds.** A campaign can hold
    three runs on one template while the group holds seven, spanning several days,
    and badge ``cassette_mode_differs`` because some non-members recorded with a
    cassette while every member recorded ``off`` — a caveat earned by runs
    the analysed set does not contain. ``scope_run_ids`` is how a caller
    asks about the set actually under analysis: the grouping key is unchanged and
    the badge logic is unchanged, and only the population they run over narrows.

    Two badges are not "a difference was observed". ``tool_config_differs``
    reports one that usually IS the point of the runs — it is here because it
    lives in the *variant* key rather than the context key, so a swept set is
    otherwise badge-silent and reads as one configuration. ``context_incomplete``
    reports the opposite: a comparison that cannot be decided, because a run
    never recorded a role pin. That one cannot be inferred from the others —
    an unrecorded pin makes every value-equality here come out clean.

    ``measurement_windows_disjoint`` is the one badge whose evidence is not in the
    run documents. When two runs occupied non-overlapping spans of wall-clock
    time, whatever moved on the box or at the providers between them varies with
    the runs — a real confound that every field on an :class:`EvalRun` compares
    equal on. The spans come from the results' ``scored_at``, which is why
    ``results`` exists as a parameter at all.

    Runs with no captured subject identity are excluded, for the reason given in
    :func:`project_score_records`.

    Args:
        runs: Runs to group. Order is irrelevant; output is sorted for stability.
        results: The results those runs produced, in any order and from any run —
            they are indexed by ``eval_run_id`` here. **A caller that omits them
            asks a narrower question**: the measurement-window badge cannot be
            decided without them, and an undecidable badge is silent, so a group
            reads as sharing a clock when nothing checked. Every production caller
            supplies the scope's results; the default exists for callers whose
            question really is only about the run documents.
        full_windows: List every measurement span rather than summarising above
            :data:`MAX_INLINE_MEASUREMENT_WINDOWS`. Off by default because this
            surface groups a whole scope — the group that most needs the
            disclosure is the one with the most runs, and so the one the listing
            form renders unreadable.
        scope_run_ids: Narrow every group to these runs before badging — the run
            ids of the set actually under analysis, typically a campaign's members
            resolved by the caller. ``None`` groups and badges every supplied run,
            which is the unscoped behaviour and is unchanged. **Run ids rather than
            a campaign id deliberately**: this module knows nothing about
            campaigns and must not learn — new eval machinery lands host-agnostic,
            and resolving a grouping concept
            to its members is the caller's one lookup. A scope naming runs that are
            not here is logged rather than refused: unlike an unknown run *status*,
            which names nothing that can exist, a member absent from the scope
            is a real and reachable state.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`ComparisonSetsResult` — one :class:`ComparisonSet` per
        (subject, template) group, sorted by subject then template, plus the
        count of runs excluded for having no captured subject identity, plus
        ``out_of_scope_run_ids`` naming what a scope left out — at the result
        level every supplied run the scope excluded, and per group the comparable
        runs it excluded from THAT group. Both are empty when unscoped. Under a
        scope, a group whose every member is out of scope is not emitted at all,
        because a group with no in-scope run has nothing to say about the set the
        caller asked about — its members are still named at the result level, so
        the narrowing stays visible rather than becoming a group that vanished.
    """
    results_by_run: dict[str, list[EvalResult]] = {}
    for result in results:
        results_by_run.setdefault(result.eval_run_id, []).append(result)

    # Narrowed BEFORE grouping, so every derived quantity below — the shared case
    # intersection, the case-set fingerprints, every badge and both disclosures —
    # is computed over one population. Filtering afterwards would leave a badge
    # earned by a run the returned group no longer lists, which is the defect one
    # level down from the one this parameter exists to fix.
    scoped: frozenset[str] | None = frozenset(scope_run_ids) if scope_run_ids is not None else None
    # What the scope left out, captured BEFORE the narrowing and keyed the same way the
    # groups are, so a scoped group can say what a reader would otherwise have to re-read
    # the scope to discover. Grouped here rather than derived by the caller because
    # only this function knows the grouping rule, and a second derivation of it is how a
    # group and its own exclusion list come to disagree about which runs are comparable.
    out_of_scope_run_ids: list[str] = []
    out_of_scope_by_key: dict[tuple[str, str | None], list[str]] = {}
    if scoped is not None:
        present = {run.id for run in runs}
        out_of_scope = [run for run in runs if run.id not in scoped]
        out_of_scope_run_ids = sorted(run.id for run in out_of_scope)
        for run in out_of_scope:
            out_of_scope_by_key.setdefault((_subject_id_of(run), run.template_id), []).append(run.id)
        if missing := sorted(scoped - present):
            # Logged rather than raised, and never silent: a scope that resolves to
            # nothing returns no groups, which reads exactly like "these runs are
            # not comparable with anything".
            log.info(
                "comparison_sets scope named %d run(s) not among the supplied runs: %s",
                len(missing),
                ", ".join(missing),
            )
        runs = [run for run in runs if run.id in scoped]

    # No blank-subject exclusion, and nothing to count: every run carries a non-blank subject key,
    # because ``SubjectSnapshot`` refuses a blank one. A counter that could only ever read zero is
    # a disclosure that tells a reader the opposite of the truth.
    groups: dict[tuple[str, str | None], list[EvalRun]] = {}
    for run in runs:
        groups.setdefault((_subject_id_of(run), run.template_id), []).append(run)

    sets: list[ComparisonSet] = []
    for (subject_id, template_id), grouped in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        member_case_sets = [set(run.test_case_ids) for run in grouped]
        shared = set.intersection(*member_case_sets) if member_case_sets else set()

        # Through the resolver, not off the raw field. `resolve_context_identity` is
        # the single place the stamped-or-derived decision is made, and reading
        # `run.context_key` directly opted this surface out of it: a run stamped under
        # an older predicate kept comparing on a key that predicate no longer produces,
        # and a run whose roles were never recorded reported a key as if it were whole.
        identities = {run.id: resolve_context_identity(run, profile) for run in grouped}

        # The group's distinct case sets, keyed by the run's own `case_basis` digest.
        # Computed BEFORE the badges because the case-set badge is derived from it:
        # one notion of "the sets differ", so the badge and the fingerprints cannot
        # come apart. Deciding the badge separately — over `set(test_case_ids)` — was
        # a second derivation that really did disagree, since a set comparison cannot
        # see a duplicate id that the sorted-list digest does.
        by_fingerprint: dict[str, list[EvalRun]] = {}
        unresolved: list[EvalRun] = []
        for run in grouped:
            basis = identities[run.id].context_components.case_basis
            # An unresolved basis is held apart, one entry PER RUN. Keying every such
            # run on a shared ``""`` merged them into one bucket, which broke two things
            # at once: the case-set badge could not fire between them (they compared as
            # a single set), and ``n_cases`` below — read from the first member as the
            # set's representative, sound only while a bucket really IS one set —
            # reported that one member's denominator as the whole bucket's. Keeping them
            # out of the digest-keyed buckets makes the merge structurally impossible
            # rather than merely unlikely, which matters because the state is latent
            # today: nothing stamps an identity without ``case_basis`` at the current
            # IDENTITY_VERSION, so the reachable version would arrive unnoticed on the
            # next bump.
            if basis:
                by_fingerprint.setdefault(basis, []).append(run)
            else:
                unresolved.append(run)

        badges: list[str] = []
        # Only RESOLVED fingerprints can evidence a difference. Counting the
        # per-run unresolved entries here would report "the sets differ" on the
        # strength of not knowing what they were.
        if len(by_fingerprint) > 1:
            badges.append(BADGE_CASE_SET_DIFFERS)
        if unresolved:
            badges.append(BADGE_CASE_SET_UNRESOLVED)
        if len({identities[run.id].context_key for run in grouped}) > 1:
            badges.append(BADGE_CONTEXT_DIFFERS)
        # Equality on the pins is only meaningful once both are recorded. Two runs that
        # each inherited a judge before the pin was resolved both carry None, and every
        # comparison here — the tuple below, the key above — comes out EQUAL, so the group
        # ships with no badge at all, which this surface defines as needing no caveat.
        # The absence has to be its own badge; nothing about the values can express it.
        if any(identities[run.id].partial for run in grouped):
            badges.append(BADGE_CONTEXT_INCOMPLETE)
        # Per-dim attribution joins the pins here for the same reason it joined the roles
        # component: two runs can agree on every pin and still have had their rubric scored
        # by different models, which is a difference in the apparatus wearing the appearance
        # of a replicate. Hashed rather than compared as a mapping so the tuple stays
        # hashable, and sorted so dim resolution order cannot fake a difference.
        #
        # RECORDED attribution only, via the same accessor ``derive_context_identity``
        # gates on, and compared SEPARATELY from the pins rather than folded into one
        # tuple per run. Both halves matter. Reading the map whenever it was merely
        # present let a DERIVED reconstruction speak with the authority of a record —
        # asserting or suppressing a comparability claim that ``identity.py`` refuses to
        # hash in the same breath. Folding it into the per-run tuple then reproduced the
        # absence-as-difference bug one level down: a run with no recorded attribution
        # contributes a different tuple element from one that has it, so merely MIXING
        # them fired the badge on the strength of what one run could not say. This is the
        # shape the case-basis arm above already uses — only resolved values evidence a
        # difference, and the undecidable case is BADGE_CONTEXT_INCOMPLETE's to carry.
        # The pinned CONFIG set is the third arm, on the argument above carried one step
        # further: two runs can agree on every pin AND on every effective judge model and
        # still have been scored by different judge PROMPTS, which is what a judge A/B is
        # made of. Without this arm such a pair fires only ``context_differs`` — "something
        # about the conditions moved" — which is precisely the opaque key mismatch the
        # per-component badges exist to save the operator from bisecting by hand.
        #
        # THE PINS COME FROM THE REGISTRY, through the declared readers, so this badge and the two
        # surfaces that print the delta cannot disagree about one pair of runs. It was a hand-built
        # ``{(run.judge_model, run.simulator_model)}``, which was wrong twice over: it spoke only
        # for the ENGINE's two roles, so a host that grades with code and declares its own role got
        # no arm at all however registry-driven everything upstream had become; and a tuple of raw
        # fields reproduced the absence-as-difference bug the attribution arm below had already
        # been fixed for — one arm recording a pin and another recording nothing is not an observed
        # difference, and ``BADGE_CONTEXT_INCOMPLETE`` is what carries it. ``comparability`` applies
        # that rule per input rather than per tuple.
        #
        # ``omits_apparatus`` takes every arm's VALUE rather than the declaration alone: a host that
        # left a seat unfilled and whose runs recorded one is contradicting itself, and the
        # runs win.
        #
        # **The two digest arms below stay, and are not redundant with it.** Both read a RUN-LEVEL
        # declaration that no ``Sweepable`` reads: ``judge_config_ids`` is declared here at launch,
        # while the core declaration of the same name reads the set the results were actually scored
        # WITH — the reader's own docstring says those are different questions. A judge A/B moves
        # the launch pin, and a run whose results were never scored would stop badging if this arm
        # were folded into the observed one. ``effective_judges`` is the same shape: the declaration
        # isolates DIVERGENCE from the pin, which is deliberately narrower than the whole map.
        sweepables = profile.sweepables
        pins_by_run = {run.id: sweepables.read_role_pins(run, results_by_run.get(run.id, ())) for run in grouped}
        pin_values = {name: [pins_by_run[run.id].get(name) for run in grouped] for name in sweepables.role_pins}
        pins_differ = any(
            sweepables.comparability(
                name,
                [profile.apparatus_level(run, name, value) for run, value in zip(grouped, values, strict=True)],
            )
            == "differs"
            for name, values in pin_values.items()
            if not profile.omits_apparatus(name, zip(grouped, values, strict=True))
        )
        recorded_attributions = {
            canonical_digest(dict(sorted(judges.items())))
            for run in grouped
            if (judges := run.hashable_effective_judges)
        }
        # ``is not None``, never truthiness: an empty set is the recording "no scored dim
        # carried a config", and two runs that both recorded it agree. Reading it as absent
        # would drop them out of the comparison and let a genuinely-differing third run in
        # the group pass unbadged.
        recorded_config_sets = {
            canonical_digest(dict(sorted(run.judge_config_ids.items())))
            for run in grouped
            if run.judge_config_ids is not None
        }
        if pins_differ or len(recorded_attributions) > 1 or len(recorded_config_sets) > 1:
            badges.append(BADGE_ROLES_DIFFER)
        # Digest the RESOLVED configs, not the overrides: an override restating the
        # subject's own value changes nothing the candidate faced, and comparing
        # overrides would badge that as a difference. This mirrors what the variant
        # key hashes, so the badge and the key can never disagree about what moved.
        if (
            len(
                {
                    canonical_digest(run.subject_snapshot.component_hashes().get("resolved_tool_configs"))
                    for run in grouped
                }
            )
            > 1
        ):
            badges.append(BADGE_TOOL_CONFIG_DIFFERS)
        # Only RESOLVED windows are collected, on the rule the case-basis and
        # attribution arms already follow: a run that cannot say when it was
        # measured is not thereby evidence that it was measured somewhere else.
        # The badge is gated on the DISCLOSURE rather than on the predicate
        # directly, so a group can never carry the flag with no sentence naming
        # the spans behind it — the flag alone cannot tell four minutes from four
        # hours, and that judgement is the reader's.
        windows = [
            w for run in grouped if (w := measurement_window(run.id, results_by_run.get(run.id, ()))) is not None
        ]
        window_disclosure = measurement_window_disclosure(windows, full=full_windows)
        if window_disclosure:
            badges.append(BADGE_MEASUREMENT_WINDOWS_DISJOINT)
        # Gated on the disclosure, never on a predicate of its own — the discipline the
        # window arm above follows, and for a sharper reason here: the flag alone cannot
        # tell an `off`/`capture` recording difference from a live-versus-replayed
        # substitution, and only the second makes the group's QUALITY numbers a mixture.
        # A badge with no sentence behind it would be the less useful half of the pair.
        cassette_disclosure = cassette_mode_disclosure({run.id: run.cassette_mode for run in grouped})
        if cassette_disclosure:
            badges.append(BADGE_CASSETTE_MODE_DIFFERS)

        case_sets = [
            CaseSetIdentity(
                fingerprint=fingerprint,
                n_cases=len(set(members[0].test_case_ids)),
                run_ids=sorted(run.id for run in members),
            )
            # Most-used set first, so the group's dominant denominator leads and the
            # odd run out is visibly the exception rather than one row among equals.
            for fingerprint, members in sorted(by_fingerprint.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ]
        # The unresolved runs last, one entry each: a set of one is the least-used there
        # is, and nothing established that any two of them ran the same cases.
        case_sets.extend(
            CaseSetIdentity(fingerprint=None, n_cases=len(set(run.test_case_ids)), run_ids=[run.id])
            for run in sorted(unresolved, key=lambda run: run.id)
        )

        sets.append(
            ComparisonSet(
                subject_id=subject_id,
                subject_label=grouped[0].subject_snapshot.subject_label,
                template_id=template_id,
                run_ids=sorted(run.id for run in grouped),
                shared_test_case_ids=sorted(shared),
                case_sets=case_sets,
                badges=badges,
                measurement_window_disclosure=window_disclosure,
                cassette_mode_disclosure=cassette_disclosure,
                out_of_scope_run_ids=sorted(out_of_scope_by_key.get((subject_id, template_id), [])),
            )
        )
    return ComparisonSetsResult(
        comparison_sets=sets,
        out_of_scope_run_ids=out_of_scope_run_ids,
    )


# =============================================================================
# Pivot — any two factors as axes, with the disclosures that make it honest.
# =============================================================================

# Weighting modes for rolling a cell's observations into one number.
#
# The default is equal-per-scenario: case counts in this corpus reflect budget
# history rather than importance, and k is unbalanced (k=1 vs k=3) across the
# same suite, so sample-weighting lets an artifact of how much we could afford
# to run drive the ranking. Sample-weighted stays available as an explicit,
# labeled toggle — it is the right answer when the question really is "what did
# the whole population do", and the cost of the default (a thin scenario gets an
# equal vote) is exactly what the per-cell `n` is displayed to expose.
#: Weighting mode: every scenario (case) gets an equal vote in a cell's number, however many observations it has.
WEIGHTING_EQUAL_PER_SCENARIO = "equal_per_scenario"
#: Weighting mode: every observation gets an equal vote, so a case with more repeats counts for more.
WEIGHTING_SAMPLE_WEIGHTED = "sample_weighted"
WEIGHTINGS = (WEIGHTING_EQUAL_PER_SCENARIO, WEIGHTING_SAMPLE_WEIGHTED)
#: The weighting a pivot uses when none is named: equal per scenario.
DEFAULT_WEIGHTING = WEIGHTING_EQUAL_PER_SCENARIO

# A cell's three states, which a renderer must keep visually distinct.
#
# `not_run` and a value of zero are different facts about the world, and
# collapsing them is the failure this whole projection is shaped to prevent: an
# unrun combination that renders as 0.0 reads as a catastrophic result rather
# than as an absent one.
#
# `unmeasured` is the third case — observations exist at this cell but none
# carried a value for this measure, which is neither "we never tried" nor "we
# measured zero". It is reached from real data by the projection emitting a
# null-valued row for an observation with no value: an infra-excluded result, or
# an ok result carrying no rubric dims. A projection that emitted nothing
# instead produced the exact collapse the first paragraph forbids, one state
# over — the cell fell to the empty-cell branch and claimed nobody tried a
# combination that was tried and failed in the harness.
#: A pivot cell state: observations landed here and carried a value for the measure.
CELL_MEASURED = "measured"
#: A pivot cell state: no observation landed here — the combination was never run, which is not a zero.
CELL_NOT_RUN = "not_run"
#: A pivot cell state: observations landed here, but none carried a value for the measure.
CELL_UNMEASURED = "unmeasured"
#: A fourth state, and not a kind of the other three: the cell HAS measured observations, and its
#: mean is withheld because it would pool two quantities that are not one distribution — today a
#: cost cell pooling replayed results with live ones (#658). `PivotCell.withheld` says why, and
#: `n` / `n_cases` / `outcomes` still say what the cell held, so withheld never reads as empty.
CELL_WITHHELD = "withheld"

# Observation-level metric name -> the registry name of the AGGREGATE that an
# aggregating surface (`pivot` cell, `history` series point) actually reports.
#
# It has a second job: inverted, it is the operator-facing ALIAS table
# (`_OBSERVATION_OF_AGGREGATE`), so every name `list_metrics` publishes for one of
# these quantities is a name the surfaces accept. Both jobs need the same pairs,
# which is why they are one table and not two that can drift.
#
# A row metric is never its own cell's descriptor: a row carries one observation,
# while `mean_composite` / `mean_cost_usd` / `mean_score` describe the mean OVER
# rows, which is what a cell holds. That is true whether or not the row metric is
# itself a registry key — `score` is one (the registry describes the raw 1-5 judge
# score) and `composite` / `cost_usd` are not, and all three map here regardless,
# because what a cell needs described is the aggregate.
#
# The mapping has to be explicit because `describe_measure` is total — an unmapped
# name resolves silently to whatever the registry happens to hold for it, which for
# an unseeded name is the unclassified arm (family=None, held at the strictest
# transferability class) and for `score` would be the per-observation 1-5 descriptor
# on a cell holding a mean. Adding a row metric means adding its aggregate here, or
# accepting the fallthrough knowingly.
_AGGREGATE_OF_OBSERVATION = {
    METRIC_COMPOSITE: "mean_composite",
    METRIC_COST_USD: "mean_cost_usd",
    # The registry declares `mean_score` `judge_mediated` on a (1.0, 5.0) range.
    # The cross-subject pooling refusal would fire without this entry — the unclassified arm holds
    # at `scenario_bound`, which is stricter still — but it would fire naming a
    # class nothing vetted, and every honest cell would report `family=None` with
    # no scale beside a number whose whole point is that it is 1-5 and not 0-1.
    METRIC_SCORE: "mean_score",
    # `history` only — `pivot` refuses `total_ms` (it is not projected). Without
    # this entry a latency series described itself with the `total_ms` descriptor,
    # which describes ONE result's wall-clock and its llm/tool/orchestration
    # partition — a partition a mean over cases does not have.
    METRIC_TOTAL_MS: "mean_total_ms",
    # The dual-score axes. Their per-observation descriptors are the registry entries
    # under these same names (1-5, family `dual_axis`); a cell holds the MEAN of those,
    # which is a different quantity and needs its own entry — the fallthrough would
    # otherwise describe a mean with the descriptor of one observation, the exact defect
    # the `score` entry above exists to avoid. Not mapped to `mean_score`, though the
    # arithmetic is identical: this table is inverted into the operator-facing alias map,
    # so three metrics sharing one aggregate name would make the inverse answer with
    # whichever came last.
    METRIC_TRANSCRIPT: "mean_transcript_score",
    METRIC_OUTCOME: "mean_outcome_score",
    # A cell of 1/0 check rows means a pass rate — of ONE check when the cell is keyed on goal_check.
    METRIC_GOAL_STATE: "goal_state_pass_rate",
}

# The inverse: registry aggregate name -> the observation-level name the surfaces
# key rows on. This is what closes the seam an operator hits when they read
# `list_metrics` and use what it says.
#
# `list_metrics` publishes the AGGREGATE names (`mean_composite`, `mean_cost_usd`,
# `mean_total_ms`, `mean_score`) because those describe what an aggregating surface
# returns. The `metric` parameter of `pivot` / `history` names the ROW being
# aggregated, so it took the short observation names — and `mean_composite`, the one
# name the catalog gives that quantity, was refused as unknown. Accepting both is the
# fix, in this direction and not the other:
#
#   - Renaming the row constants to the registry keys is forbidden at their definition
#     site (see the comment above `METRIC_COMPOSITE`) and would be wrong anyway: an
#     export row IS one observation, so `metric=composite` on a row is its honest name.
#   - Seeding a bare `composite` measure into the registry would publish a second
#     composite-quality entry describing a row nobody aggregates, and would leave
#     `mean_cost_usd` / `mean_total_ms` still refused.
#
# So the catalog name resolves to the row name at the surface boundary, and the row
# name keeps working. Derived rather than written out, so a new pair cannot be added
# to one direction only; `_metric_vocabulary` renders both names in every refusal, so
# an operator who guesses wrong is told the other one exists.
_OBSERVATION_OF_AGGREGATE = {aggregate: observation for observation, aggregate in _AGGREGATE_OF_OBSERVATION.items()}


def resolve_measure_name(metric: str) -> str:
    """Resolve a catalog (aggregate) measure name to the observation name rows carry.

    Args:
        metric: A measure name as an operator supplied it — either the
            observation-level row name (``composite``) or the registry name of its
            aggregate as ``list_metrics`` publishes it (``mean_composite``).

    Returns:
        The observation-level name, unchanged for anything this table does not
        cover — an unknown name must reach its surface's own refusal so the caller
        is told what IS accepted, rather than being rewritten into something else.
    """
    return _OBSERVATION_OF_AGGREGATE.get(metric, metric)


def _metric_vocabulary(accepted: frozenset[str]) -> str:
    """Render an accepted-measure set naming each row name beside its catalog name.

    Args:
        accepted: The observation-level names a surface accepts.

    Returns:
        A sorted, comma-joined list in which every measure that has a published
        aggregate name shows it too — a refusal that named only the row names is
        what sent an operator reading ``list_metrics`` in a circle.
    """
    return ", ".join(
        f"{name} (or {_AGGREGATE_OF_OBSERVATION[name]})" if name in _AGGREGATE_OF_OBSERVATION else name
        for name in sorted(accepted)
    )


# What each surface-accepted metric is, in a clause, for every help text that lists them. Keyed on
# exactly the metrics some surface accepts, and checked at import, so a metric added to one of the
# accepted sets without a gloss fails loudly instead of reaching a help text that never mentions it.
_METRIC_GLOSS: dict[str, str] = {
    METRIC_COMPOSITE: "0-1 quality",
    METRIC_COST_USD: "measuring spend in USD: the candidate's and the judge's and simulator's",
    METRIC_SCORE: "the raw judge score: 1-5, or 1/0 on a pass/fail dimension",
    METRIC_TOTAL_MS: "one result's wall-clock",
    METRIC_TRANSCRIPT: "dual-score axis, 1-5 and NOT a dimension",
    METRIC_OUTCOME: (
        "dual-score axis, 1-5 and NOT a dimension; its divergence from the transcript axis is what separates a "
        "worse agent from a changed world"
    ),
    METRIC_GOAL_STATE: "1 passed / 0 not, code-graded",
}


def metric_help(accepted: frozenset[str]) -> str:
    """Render the metrics a surface accepts, each glossed and beside its catalog name, for its help text.

    The one producer of that list for the pivot and history help on REST and MCP alike, so a help
    text cannot name a metric its surface refuses or leave out one it accepts.

    Args:
        accepted: The observation-level names the surface accepts.

    Returns:
        A semicolon-joined list, in name order.
    """
    return "; ".join(
        f"'{name}' ({_METRIC_GLOSS[name]}"
        + (f"; or its catalog name '{_AGGREGATE_OF_OBSERVATION[name]}'" if name in _AGGREGATE_OF_OBSERVATION else "")
        + ")"
        for name in sorted(accepted)
    )


# Transferability class that may be pooled across subjects.
#
# Composite quality is comparable within a subject and never across one,
# because rubric dimensions derive from each subject's own description and
# tools. That binds judge-mediated and scenario-bound measures. It does NOT bind
# mechanical ones — a dollar is a dollar whoever spent it — and refusing those
# too would put the cross-subject budget view out of reach for no gain.
_POOLABLE_ACROSS_SUBJECTS = "mechanical"


class PivotError(ValueError):
    """A pivot was requested that cannot be answered honestly.

    Raised rather than returned as an empty or caveated table: every case that
    reaches it is a question whose honest answer is "not like that" — an axis the
    rows do not carry, an unknown weighting mode, or a cross-subject pooling.
    Returning an empty table instead would be indistinguishable from "no data",
    which is how a refusal becomes a silent zero.
    """


#: The ``method_id`` of the cost estimator's predictions: a planned cell priced from the corpus's
#: own per-observation usage history (:func:`compute_estimate_cost`).
COST_PREDICTION_METHOD = "usage-history"


class PredictedValue(EvalBaseModel):
    """A modelled estimate, kept beside the observed value it predicts and never fused with it.

    The two are different claims: an observed value is what happened, a predicted one is what a
    model expected beforehand, and a decision made on the second believing it was the first is the
    failure the separation prevents. So a prediction never occupies an observed slot — it has its
    own field wherever it appears, and a surface shows the two side by side.

    Two writers: :func:`compute_estimate_cost` (``method_id`` :data:`COST_PREDICTION_METHOD`), one per
    planned cell, predicting that cell's total cost from the corpus's usage history; and
    :meth:`~threetears.evals.ops.LaunchEstimate.planned_costs`, one per priced arm of a launch, from the host's
    launch pricer — whose ``method_id`` is the pricer's own (``usage-history`` for the engine's
    :func:`~threetears.evals.ops.history_launch_pricer`) or ``launch-pricer`` for a pricer that names none.
    :func:`compute_pivot` sets either beside each observed cost its plan describes.

    ``value`` is the point estimate, ``interval_low`` / ``interval_high`` its uncertainty band
    (absent for a point-only method, or where the basis is too thin for one), ``method_id``
    identifies the estimator so two predictions from different methods are never silently compared,
    ``trust`` is the method's own support/confidence score (None for a method that states none),
    and ``computed_at`` is the freshness stamp — a stale prediction over moved data is worse than
    none.
    """

    value: float
    interval_low: float | None = None
    interval_high: float | None = None
    method_id: str
    trust: float | None = None
    computed_at: str


class PivotCell(EvalBaseModel):
    """One (row, column) cell: the number, and everything needed to trust it."""

    row: str
    column: str
    status: str
    value: float | None = None
    # What the cell was predicted to cost per observation when it was planned — set only on a cost
    # pivot handed the estimate made before launch (`compute_pivot(predicted_cost=...)`). Kept beside
    # `value` rather than folded into it so a prediction can never be mistaken for an observation.
    predicted: PredictedValue | None = None
    # Beside a prediction, how many of the cell's observations came from runs the plan did not make — the earlier
    # history its prediction may have been drawn from among them. None when there is no prediction, or when the
    # plan names no launched runs (it had not launched), so every observation here is one it did not make.
    n_unplanned: int | None = None
    # Observations that carried a value, and the distinct test cases they span.
    # Under equal-per-scenario weighting `n_cases` is the denominator the value
    # was divided by, so showing both is what lets a reader see that a cell's
    # mean rests on three cases or on thirty.
    n: int = 0
    n_cases: int = 0
    # Standard error of whatever the value averages — case means under
    # equal-per-scenario, raw observations under sample-weighted — so the spread
    # always describes the estimate actually reported.
    sem: float | None = None
    # Observations present at this cell that carried no value for this measure.
    # Counted rather than dropped: an aggregate over 3 of 12 observations is a
    # different claim from an aggregate over 12.
    n_unmeasured: int = 0
    # Observation counts per scoring outcome, so a cell whose mean rests largely
    # on candidate failures cannot look like one that rests on clean passes.
    outcomes: dict[str, int] = {}
    #: Why the cell's value is withheld, set exactly when ``status`` is ``withheld``: on a cost pivot,
    #: the cell pools results from runs that replayed their third party with results from runs that
    #: ran it live, and a replayed result did not spend what a live one does, so their mean is neither
    #: one's spend (#658). The counts above still say what the cell held.
    withheld: str | None = None
    #: The cassette modes the runs behind the cell's valued observations recorded, sorted. One entry is
    #: a uniform cell; ``replay`` beside another mode is the mix a cost cell withholds.
    cassette_modes: list[str] = []
    #: On a cost pivot, the role sets the cell's dollars were summed over
    #: (:func:`pooled_cost_compositions`), from the observations that carried a value (#625). More than
    #: one entry means the cell's own mean pools totals that covered different things; two cells whose
    #: entries differ are not comparable on cost, which :attr:`PivotTable.cost_compositions_differ`
    #: flags at the table. Empty on any other metric.
    cost_compositions: list[list[str]] = []
    #: On a composite pivot, what the cell's composites were meaned over (:func:`pooled_composite_basis`):
    #: the union of the observations' bases, and ``ragged`` when they were meaned over different dimension
    #: sets, so the cell's mean averages different questions (#638). ``None`` on any other metric and on a
    #: cell with no valued observation.
    composite_basis: CompositeBasis | None = None
    #: Of the ``n`` valued observations, how many carried a background delivery a harness supplied — seeded
    #: or replayed (:func:`~threetears.evals.contracts.usage_capture.count_substituted_deliveries`). Counted on
    #: every metric, because a substituted delivery is what the candidate read as well as what it did not pay
    #: for. ``0`` on a cell whose observations ran every delivery live.
    n_substituted: int = 0
    #: On a cost pivot, the sentence a cell carries when ``n_substituted`` is above zero: a substituted delivery
    #: spent none of its dollars, so the cell's spend leaves them out, and a cell built ONLY from such
    #: observations is no live run's spend at all. ``None`` on any other metric and on a cell with none.
    #: Stated on the cell rather than left to the export's ``substituted_deliveries`` column, because the cell
    #: is what is read.
    substitution_disclosure: str | None = None
    #: Identity key -> the predicate versions its observations here were stamped at, for each identity
    #: key (``variant_key``, ``context_key``) the table groups or filters on, and only where the cell
    #: pools more than one (#672). Two keys stamped at different versions cannot be shown FROM THE STAMP
    #: ALONE to describe one contestant — the frontier ranks them apart for that reason — so a cell
    #: pooling them may be averaging two conditions as repeats of one. Disclosed rather than split,
    #: because the pivot's axes are open and no one axis may imply a partition.
    identity_versions: dict[str, list[int]] = {}
    #: On a table grouped by contestant (``variant_key`` or ``model`` on an axis), which models the provider's
    #: responses named as having answered the candidate calls of the cell's observations (#684): ``one``,
    #: ``pooled`` — one requested model answered by several, so the cell's number mixes them — or
    #: ``unrecorded``. Disclosed rather than split, since the contestant is keyed at launch; ``served_model`` is a
    #: coordinate, so pivoting on it separates them. ``None`` on a table grouped on neither, and on a cell none of
    #: whose observations' candidate made a call.
    served_models: ServedModelReading | None = None


class SimpsonsFlag(EvalBaseModel):
    """A pooled column ranking that the per-row rankings mostly contradict.

    The pooled table says ``leader`` beats the other column overall, while more
    rows than not rank them the other way — the pooled number is being driven by
    which rows each column was measured on rather than by the columns
    themselves. Descriptive: the flag says the comparison is unsafe to read as a
    ranking, not which answer is correct.
    """

    column_a: str
    column_b: str
    pooled_leader: str
    rows_agreeing: int
    rows_disagreeing: int
    disagreeing_rows: list[str]


class PivotTable(EvalBaseModel):
    """A two-factor pivot over one measure, with its disclosures attached.

    ``formula`` states what was computed under the *requested weighting* rather
    than repeating the registry's single static formula, which is itself
    weighting-specific and would otherwise describe a different number than the
    one displayed. Everything not weighting-dependent — family, unit, direction —
    still comes from ``measure``, the registry descriptor.
    """

    metric: str
    measure: MetricDescriptor
    formula: str
    weighting: str
    row_factor: str
    column_factor: str
    rows: list[str]
    columns: list[str]
    cells: list[PivotCell]
    simpsons_flags: list[SimpsonsFlag] = []
    # Corpus-level accounting, so a table that answers over a fraction of the
    # supplied rows says so rather than looking complete. `n_filtered_out` counts
    # rows this query's own filters removed; `exclusions` counts observations
    # that never became rows at all, which is the difference between "you asked
    # for a subset" and "this data cannot be placed".
    n_observations: int = 0
    n_filtered_out: int = 0
    exclusions: ProjectionExclusions = ProjectionExclusions()
    #: ``run_id -> DEGRADED sentence`` for the runs behind these cells that measured
    #: less than the matrix they promised. The counterpart of ``exclusions`` on the
    #: other side of the seam: that field accounts for observations the table does
    #: NOT contain, this one qualifies observations it does. Every cell is affected,
    #: not a nameable subset — a pivot aggregates over every coordinate that is not
    #: an axis, and ``run_id`` is usually one of them — so the disclosure is stated
    #: at the table rather than marked per cell.
    completeness_disclosures: dict[str, str] = {}
    #: How many of ``n_observations`` came from those runs. The weight the caveat
    #: carries: two of two hundred is a footnote and two of four is the answer.
    n_degraded_observations: int = 0
    #: On a cost pivot, whether the table's valued observations were summed over more than one role set
    #: (#625) — within one cell or between cells. True means some cost here covered roles another did not,
    #: so a cheaper cell may only have priced fewer things; each cell's ``cost_compositions`` says which.
    cost_compositions_differ: bool = False
    #: On a composite pivot, whether the table's valued composites were meaned over more than one dimension
    #: set (#638) — within one cell or between cells. True means a difference between two cells may be a
    #: difference in what was averaged rather than in what was measured; each cell's ``composite_basis`` says
    #: which sets it pooled.
    composite_bases_differ: bool = False
    #: The sentence a comparison carries when its runs recorded different cassette modes
    #: (:func:`cassette_mode_disclosure`, the words ``runs_compare`` and ``comparison_sets`` use), over the
    #: runs behind this table's observations, or ``None`` when they all recorded one (#658). It qualifies
    #: every metric, not only cost: a replayed arm was also measured on the questions its capture asked.
    cassette_mode_disclosure: str | None = None
    #: One sentence naming every cell that pools more than one identity version of the key it is grouped
    #: or filtered on (:attr:`PivotCell.identity_versions`), or ``None`` when none does (#672).
    identity_pooling_disclosure: str | None = None
    #: One sentence naming every cell whose observations were answered by more than one served model, and
    #: counting those that cannot say (:attr:`PivotCell.served_models`), or ``None`` when every cell names one
    #: model or the table is not grouped by contestant (#684).
    served_model_disclosure: str | None = None
    #: Plans the cost estimate made that no cell describes — a model no level of the model axis carries, or a
    #: template no cell at that model holds alone — each as ``model`` or ``model (template)``. Named rather than
    #: dropped, since a prediction with nowhere to sit is still a fact about the plan: an arm that was priced and
    #: did not run, or ran where this table does not separate it. Empty when no estimate was given.
    unplaced_predicted_models: list[str] = []


def _require_coordinate_name(factor: str) -> None:
    """Reject a coordinate name that no score record could ever carry.

    Used for both axes and filter keys — they are the same namespace, and a
    filter on a coordinate that does not exist empties the table as convincingly
    as a real result does.

    Checked before any row is touched, because :func:`_axis_value` is only
    reached once there are rows to group — so on an empty selection (a scope
    with no observations of this measure, or a filter matching nothing) a typo'd
    axis would return an empty grid instead of a refusal, which is exactly the
    "indistinguishable from an empty scope" answer the refusals exist to
    prevent. Only the *static* half is decidable here; a dotted name stays legal
    by design, since the open set is not knowable without data and a run may
    simply not have set that key.

    Args:
        factor: A declared coordinate name, or a dotted open-coordinate name.

    Raises:
        PivotError: An undotted name the model does not declare, or the open-map
            container itself.
    """
    if "." in factor:
        return
    if factor == "factors":
        # The map is the container for open coordinates, never one itself —
        # stringifying it would collapse every run into one dict-shaped bucket.
        raise PivotError("'factors' is the open-coordinate map, not a coordinate — pivot on one of its dotted keys")
    if factor == "host_measures":
        # Same defect, and one step further from being an axis: these are the host's own
        # MEASUREMENTS of a cell, so grouping by one would partition the corpus by its own
        # answer. Refused here rather than left to `_axis_value`, which would stringify the
        # whole map and bucket every result under one dict.
        raise PivotError(
            "'host_measures' is the host's own grade of a cell, not a coordinate — read it from the export, "
            "where each measure is its own column"
        )
    # Checked against the declared fields, not `hasattr`: every model also
    # exposes methods, so `hasattr` would accept `model_dump` as an axis and
    # bucket the whole corpus under one stringified bound method.
    if factor not in ScoreRecord.model_fields:
        raise PivotError(
            f"unknown factor {factor!r} — score records carry no such coordinate "
            "(a host's levers are dotted, e.g. a kind's overlay '<kind>.<field>')"
        )


def _axis_value(record: ScoreRecord, factor: str) -> str:
    """Read one factor off a row, as the string an axis is keyed on.

    Two coordinate spaces, resolved in one place. A **declared** field
    (``model``, ``template_id``, …) is read by name; a **dotted** name falls
    through to the open ``factors`` map, which carries every lever the host's
    registry resolves for the run — a kind's overlays (``gm.difficulty``) among them —
    at their resolved levels. Neither is checked against a list of permitted axes: a
    new lever must become pivotable with no edit here, which is the whole point of
    keeping the factor set open.

    An absent value becomes a visible ``"—"`` level rather than being dropped,
    so a corpus where half the runs carry no override for a key shows that as a
    row of its own instead of quietly shrinking. That makes "ran without this
    override" a comparable cohort rather than missing data — which is exactly
    what a bake-off is asking about.

    Args:
        record: The row to read.
        factor: A declared coordinate name, or a dotted open-coordinate name.

    Returns:
        The coordinate's value as a string.

    Names are validated by :func:`_require_coordinate_name` before any row is read, so
    this is pure resolution. A *dotted* name never fails: it names a key some run
    may simply not have set, and refusing it would make "nobody overrode this"
    indistinguishable from a typo — the honest answer is a table of "—".
    """
    if "." in factor:
        return record.factors.get(factor) or "—"
    value = getattr(record, factor)
    return "—" if value is None or value == "" else str(value)


def _describe_aggregate(metric: str, *, profile: HostProfile) -> MetricDescriptor:
    """Describe what a pivot CELL holds, which is an aggregate over rows.

    Args:
        metric: The observation-level metric name carried on the rows.
        profile: The host whose vocabulary this reads.

    Returns:
        The aggregate's descriptor when one is registered, else the honest
        unclassified answer for the observation name itself.
    """
    return describe_measure(_AGGREGATE_OF_OBSERVATION.get(metric, metric), profile.measures)


def _effective_formula(metric: str, weighting: str, *, scoped: bool = True) -> str:
    """State exactly what the reported number is, under this weighting and scope.

    Two things can make the registry's static formula describe a different number
    than the cell shows, and both are stated here rather than left to the reader.
    The weighting is the first. The second is *scope*: the registry describes
    ``mean_score`` as the mean "for one rubric dimension", which is what a cell
    holds only while ``rubric_dim`` is pinned. With it pinned by neither axis nor
    filter, a cell averages every dimension the judge scored — a real question,
    and roughly the composite on the 1-5 scale, but not the one the descriptor
    names. Saying so is the same obligation the weighting clause discharges: an
    unqualified formula beside a qualified number is how a reader checks the
    wrong arithmetic and concludes the table is right.

    Args:
        metric: The observation-level metric being aggregated.
        weighting: The weighting mode the value was computed under.
        scoped: Whether the metric's scope coordinate (:data:`SCOPED_METRICS`) is pinned by an axis or
            filter. Only meaningful for a scoped metric; every other measure is whole-result and has
            nothing to pool. Both pinnings count here because both scope the cell, but only the axis is
            reachable from the REST and MCP surfaces — see the emitted caveat below.

    Returns:
        A one-line formula an operator can read at point of use.
    """
    if weighting == WEIGHTING_EQUAL_PER_SCENARIO:
        formula = f"mean over test cases of the per-case mean of {metric} (each case weighted equally)"
    else:
        formula = f"mean of {metric} over every contributing observation (cases weighted by their observation count)"
    if metric in SCOPED_METRICS and not scoped:
        scope_field, scope_noun, row_noun = SCOPED_METRICS[metric]
        # Names the AXIS only, though `scoped` also honours a filter.
        # This string ships to operators, and no operator can reach the filter:
        # `reads.pivot` composes `filters` itself from `subject_id`, and
        # neither surface accepts a `rubric_dim` filter — MCP would swallow it in
        # its kwarg union and REST declares no such query parameter, so the
        # advice would be followed, silently ignored, and answered with the same
        # pooled table still carrying this caveat. A remedy a reader cannot
        # perform is the false-capability claim this measure was added to end.
        formula += f", POOLED ACROSS EVERY {scope_noun.upper()} (put '{scope_field}' on an axis for one)"
        # The pooling also changes what the cell's EVIDENCE fields count, and
        # qualifying the value while leaving `n` unqualified is the same defect
        # one field over. Pooled, a row is a (result, dimension) pair: a judged
        # result contributes one row per dimension and an unjudged result exactly
        # one unmeasured row. So four judged 3-dim results beside four unjudged
        # ones report n=12 with 4 unmeasured — which reads as 75% measured when
        # half the results were never scored. With `rubric_dim` on an axis each
        # cell holds one dimension and `n` is a result count again.
        #
        # The DISPERSION is named per weighting rather than lumped in, because
        # `_aggregate` branches: under `equal_per_scenario` (the default) the SEM
        # is over test-case means — pooling changes those means, and how many
        # observations each averages, but not how MANY MEANS the SEM is taken
        # over — so claiming the dispersion
        # rides on rows would send an operator reconstructing an interval to `n`
        # instead of the case count and hand them one too narrow by ~sqrt(n/n_cases).
        # `sample_weighted` averages the flat rows, but its SEM is clustered by test case
        # (`_aggregate`), so its dispersion does not ride on rows either.
        formula += f"; n and the outcome counts are over (result x {row_noun}) rows here, not results"
        if weighting == WEIGHTING_SAMPLE_WEIGHTED:
            formula += "; the dispersion is clustered by test case, so the rows of one case are not independent draws"
        else:
            formula += "; the dispersion is over test-case means — a BASIS pooling does not change, though the means themselves do"
    return formula


def _aggregate(values_by_case: dict[str, list[float]], weighting: str) -> tuple[float, float | None]:
    """Roll one cell's observations into a value plus its dispersion.

    Args:
        values_by_case: Observed values grouped by test case.
        weighting: Which of :data:`WEIGHTINGS` to apply.

    Returns:
        ``(value, sem)``. The SEM is over whatever the value averages, so the
        two always describe the same estimate, and it counts cases, not observations:
        under ``sample_weighted`` the flat mean takes the cluster-robust SEM over the
        cases (:func:`~threetears.evals.analysis.stats.clustered_standard_error`), since
        a case's repeats are not independent draws.
    """
    from threetears.evals.analysis.stats import clustered_standard_error, standard_error_of_mean

    if weighting == WEIGHTING_EQUAL_PER_SCENARIO:
        case_means = [sum(vals) / len(vals) for vals in values_by_case.values()]
        return sum(case_means) / len(case_means), standard_error_of_mean(case_means)
    flat = [v for vals in values_by_case.values() for v in vals]
    cases = [case for case, vals in values_by_case.items() for _ in vals]
    return sum(flat) / len(flat), clustered_standard_error(flat, cases)


def _simpsons_flags(
    cells: dict[tuple[str, str], PivotCell],
    pooled: dict[str, float],
    rows: list[str],
    columns: list[str],
) -> list[SimpsonsFlag]:
    """Flag every column pair whose pooled order most rows contradict.

    ``pooled`` must be recomputed from the underlying observations with the row
    axis collapsed — the figure an operator sees when they *don't* break the
    comparison down — and emphatically not averaged from the cell values above.
    An unweighted mean of per-row cells cannot reverse the per-row order at all,
    so a guard built that way is arithmetically incapable of firing. The reversal
    lives precisely in the row weighting: a column measured mostly on easy rows
    outranks one measured mostly on hard rows even while losing every row.

    Pooling collapses rows, so the caller puts the thing being compared on the
    columns and the suspected confounder on the rows. Per-row leaders are read
    only from rows where both columns were measured — a row carrying just one of
    them has no order to contribute.

    Args:
        cells: Measured cells keyed by ``(row, column)``.
        pooled: Column level -> its value with the row axis collapsed.
        rows: Row levels, in display order.
        columns: Column levels, in display order.

    Returns:
        One flag per reversed pair, in column order. Empty when nothing reverses.
    """
    flags: list[SimpsonsFlag] = []
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            if col_a not in pooled or col_b not in pooled or pooled[col_a] == pooled[col_b]:
                continue
            pooled_leader = col_a if pooled[col_a] > pooled[col_b] else col_b

            per_row_leaders = []
            for row in rows:
                cell_a, cell_b = cells.get((row, col_a)), cells.get((row, col_b))
                if cell_a is None or cell_b is None or cell_a.value is None or cell_b.value is None:
                    continue
                if cell_a.value == cell_b.value:
                    continue
                per_row_leaders.append((row, col_a if cell_a.value > cell_b.value else col_b))

            if len(per_row_leaders) < 2:
                # One row cannot be a "majority of per-cell rankings"; flagging on
                # it would report every ordinary sampling difference as a paradox.
                continue

            disagreeing = [row for row, leader in per_row_leaders if leader != pooled_leader]
            agreeing = len(per_row_leaders) - len(disagreeing)
            if len(disagreeing) > agreeing:
                flags.append(
                    SimpsonsFlag(
                        column_a=col_a,
                        column_b=col_b,
                        pooled_leader=pooled_leader,
                        rows_agreeing=agreeing,
                        rows_disagreeing=len(disagreeing),
                        disagreeing_rows=disagreeing,
                    )
                )
    return flags


#: Each identity key a pivot can group on, and the coordinate carrying the predicate version that minted it.
_IDENTITY_KEY_VERSION_FIELDS: dict[str, str] = {
    "variant_key": "variant_identity_version",
    "context_key": "context_identity_version",
}


def _pooled_identity_versions(records: Sequence[ScoreRecord], keys: Iterable[str]) -> dict[str, list[int]]:
    """The identity versions a cell pools, for each identity key it is grouped or filtered on (#672).

    Args:
        records: The cell's observations.
        keys: The identity keys among the table's axes and filters.

    Returns:
        Key -> its versions, ascending, only for a key whose observations here carry more than one.
    """
    pooled: dict[str, list[int]] = {}
    for key in keys:
        field = _IDENTITY_KEY_VERSION_FIELDS[key]
        versions = sorted({version for record in records if (version := getattr(record, field)) is not None})
        if len(versions) > 1:
            pooled[key] = versions
    return pooled


def _identity_pooling_disclosure(cells: Sequence[PivotCell]) -> str | None:
    """Name every cell that pools more than one identity version of the key it is grouped on.

    The frontier's reason in the pivot's terms: two keys stamped at different versions cannot be shown from
    the stamp alone to describe one contestant, so the frontier ranks them apart (:func:`_contestant_key`).
    The pivot's axes are open, so it does not split; it says which cells pool and how to read them apart.

    Args:
        cells: The table's cells.

    Returns:
        The sentence, or ``None`` when no cell pools versions.
    """
    pooled = [cell for cell in cells if cell.identity_versions]
    if not pooled:
        return None
    named = "; ".join(
        f"({cell.row}, {cell.column}): "
        + ", ".join(
            f"{key} stamped at {', '.join(f'v{version}' for version in versions)}"
            for key, versions in cell.identity_versions.items()
        )
        for cell in pooled
    )
    return (
        f"{len(pooled)} cell(s) pool observations stamped at more than one identity version of the key they are "
        f"grouped on — {named}. Two keys stamped at different versions cannot be shown FROM THE STAMP ALONE to "
        "describe the same contestant (the frontier ranks them apart), so such a cell may average two conditions "
        "as repeats of one. Pivot the key against its version coordinate (variant_identity_version or "
        "context_identity_version) to read each version alone."
    )


#: The axes that group a pivot by contestant, on which each cell names the models that answered it (#684).
_CONTESTANT_FACTORS = frozenset({"variant_key", "model"})


def _served_model_disclosure(cells: Sequence[PivotCell]) -> str | None:
    """Name every cell answered by more than one served model, and count those that cannot say (#684).

    Args:
        cells: The table's cells.

    Returns:
        The sentence, or ``None`` when every cell that carries a reading names one model.
    """
    pooled = [cell for cell in cells if cell.served_models is not None and cell.served_models.state == "pooled"]
    unrecorded = [cell for cell in cells if cell.served_models is not None and cell.served_models.state == "unrecorded"]
    parts: list[str] = []
    if pooled:
        named = "; ".join(
            f"({cell.row}, {cell.column}): {', '.join(cell.served_models.served_models)}"
            for cell in pooled
            if cell.served_models is not None
        )
        parts.append(
            f"{len(pooled)} cell(s) pool observations one requested model was answered by several models — {named} — "
            "so each such number mixes them; pivot on served_model to read each model alone"
        )
    if unrecorded:
        parts.append(
            f"{len(unrecorded)} cell(s) rest on candidate calls whose response named no model, so whether one model "
            "answered them cannot be established"
        )
    return ". ".join(parts) + "." if parts else None


def _substitution_disclosure(n_substituted: int, n_valued: int) -> str | None:
    """The sentence a cost cell carries when some of its observations had a delivery a harness supplied.

    Disclosed rather than withheld, for the reason :func:`_cost_withheld` gives: two arms over a seeded
    template carry the same substitutions, so the comparison between their cells is honest. But a reader of
    one cell's dollars needs to know they leave the substituted deliveries' spend out — and when every
    observation substituted, that the figure describes no live run.

    Args:
        n_substituted: Valued observations carrying at least one substituted delivery.
        n_valued: Valued observations in the cell.

    Returns:
        The sentence, or ``None`` when nothing was substituted.
    """
    if n_substituted == 0:
        return None
    share = "every one" if n_substituted == n_valued else f"{n_substituted}"
    return (
        f"{share} of the {n_valued} observation(s) behind this spend carried a background delivery a harness "
        "supplied (seeded or replayed), which spent none of its dollars, so they are not in this figure"
        + (": it is no live run's spend." if n_substituted == n_valued else ".")
    )


def _cost_withheld(records: Sequence[ScoreRecord]) -> str | None:
    """Why a cost mean over these valued observations is withheld, or ``None`` when it is reported (#658).

    **A cost mean that pools replayed with live results is withheld rather than disclosed**, because a
    caveat beside a number does not stop it being read, and this number is neither population's spend. The
    observations come from runs that recorded ``replay`` and runs that ran the third party live: a replayed
    background delivery spent none of its dollars, and a replay serves only the asks its capture made (any
    other ask stops the cell), so even where nothing was substituted the replayed conversations are a
    selected population whose spend is not the live one's. A mix of ``off`` and ``capture`` is not withheld:
    both run the third party live.

    A uniform cell — every observation replayed, or every one live — is reported: its mean is one
    population's, and :attr:`PivotTable.cassette_mode_disclosure` says when the table's cells differ.

    **A substituted delivery within one mode is not a reason to withhold.** A seeded finding substitutes in
    a run that recorded ``off``, but it does so on the template's own cases, so two arms over those cases
    carry the same substitutions and the comparison between their cells is honest; the dollars it did not
    spend were never measuring spend either. The cell says so (:attr:`PivotCell.substitution_disclosure`), and
    each row's ``substituted_deliveries`` is an export column. Withholding on it would blank the cost of every
    arm over a seeded template.

    Args:
        records: The observations a cost mean would be taken over.

    Returns:
        The reason, or ``None``.
    """
    modes = sorted({record.cassette_mode for record in records if record.cassette_mode is not None})
    if SUBSTITUTING_CASSETTE_MODE in modes and len(modes) > 1:
        live = ", ".join(mode for mode in modes if mode != SUBSTITUTING_CASSETTE_MODE)
        return (
            f"pools results from runs that recorded cassette mode {SUBSTITUTING_CASSETTE_MODE} with results from "
            f"runs that recorded {live}. A replayed result re-served a recording rather than running its third "
            "party: where it replayed a background delivery it spent none of that delivery's dollars, and a "
            "replay serves only the asks its capture made, so the mean of the two is neither one's spend. Put "
            "'cassette_mode' on an axis to read each alone."
        )
    return None


def compute_pivot(
    records: list[ScoreRecord],
    *,
    row_factor: str,
    column_factor: str,
    metric: str,
    weighting: str = DEFAULT_WEIGHTING,
    filters: dict[str, str] | None = None,
    exclusions: ProjectionExclusions | None = None,
    completeness_disclosures: Mapping[str, str] | None = None,
    predicted_cost: CostEstimate | Sequence[PlannedCost] | None = None,
    profile: HostProfile,
) -> PivotTable:
    """Aggregate score records over any two factors, disclosing every caveat.

    The grid is the full cross-product of the axis levels actually observed, so
    a combination that was never run appears as a ``not_run`` cell rather than
    being absent from the table — an absent cell is indistinguishable from a
    zero once a renderer lays the grid out.

    **Pooling across subjects is refused for anything but a mechanical measure.**
    A pivot aggregates over every factor that is not an axis, so a
    ``model × template`` pivot over a multi-subject corpus would average one
    subject's composites with another's — different measurements wearing the same
    number, because rubric dimensions derive from each subject's own description
    and tools. Cost and latency are mechanical and are pooled freely. To pivot a
    judge-mediated measure over several subjects, put ``subject_id`` on an axis
    or filter to one.

    **A run that measured less than its matrix is aggregated and disclosed, never
    refused**, on the position :func:`compute_frontier` states: its rows are real
    measurements, and dropping them silently is the failure this tier exists to
    avoid. Every cell is affected rather than a nameable subset — a pivot pools over
    every coordinate that is not an axis, and ``run_id`` usually is not one — so the
    caveat is carried at the table in ``completeness_disclosures`` rather than marked
    per cell. The predicate is the completeness record, never the run's status.

    **What a cell pools that is not one quantity is said, or its number withheld.** On a cost pivot each
    cell names the role sets its dollars covered (``cost_compositions``) and the table flags when they
    differ (#625); a cost cell pooling replayed results with live ones is ``withheld`` with the reason,
    since a caveat does not stop a mean being read (#658); the table carries the comparison surfaces' cassette-mode sentence when its runs recorded
    different modes; and a cell grouped or filtered on an identity key that pools more than one version of
    its predicate names them (#672).

    Args:
        records: Rows from :func:`project_score_records`.
        row_factor: Coordinate to use as the row axis.
        column_factor: Coordinate to use as the column axis. This is the axis the
            Simpson's guard pools over rows, so put the thing being compared here.
        metric: Which measure to aggregate; rows carrying others are ignored. A
            measure in :data:`PROJECTED_METRICS`, or the registry name of its
            aggregate as ``list_metrics`` publishes it — see
            :func:`resolve_measure_name`, which holds the pairing. Stated
            relationally rather than as a list: the set has grown once already,
            and an enumeration here is one more place that would not have moved
            with it.
        weighting: One of :data:`WEIGHTINGS`.
        filters: Optional exact-match coordinate filters applied before
            aggregating, e.g. ``{"subject_id": "subj-1"}``.
        exclusions: What the projection dropped before these records existed,
            carried onto the table so the answer discloses what it could not
            see. Omitting it renders an all-excluded corpus as an empty one.
        completeness_disclosures: ``run_id -> DEGRADED sentence`` from
            :func:`degraded_run_disclosures`, for the runs behind these records.
            Narrowed here to the runs that survived ``filters``, so the table
            never carries a caveat about a run it did not aggregate. **Omitting
            it pools a short run's rows into every rate with nothing saying so**
            — the defect this parameter exists to end, and the reason it is a
            parameter rather than something re-derived per surface.
        predicted_cost: The estimate made BEFORE these observations, when the cells were planned
            (:func:`compute_estimate_cost`), or the planned costs of a launch priced by its host's pricer
            (:class:`PlannedCost`, one per priced arm). Each cell at a planned model carries that model's
            prediction in ``predicted``, beside the cost it observed and never in place of it. Passed
            in rather than computed here because a prediction drawn from a history that already
            holds the observations it predicts would be a restatement of them. Only a cost pivot
            with the candidate model on an axis has a cell a planned model's cost describes.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`PivotTable` whose every cell carries ``n``, dispersion, and its
        measured/unmeasured/not-run/withheld status, plus the completeness disclosures of
        the short runs its numbers were pooled from and the pooling disclosures above.

    Raises:
        PivotError: Unknown weighting or metric, an undotted axis or filter the
            rows do not declare, a cross-subject pooling, or a predicted cost handed
            to a pivot of another metric or one with no model axis. Every one
            is refused here rather than at an adapter, so both surfaces refuse
            identically instead of one of them answering an empty grid.
    """
    if weighting not in WEIGHTINGS:
        raise PivotError(f"unknown weighting {weighting!r} — expected one of {', '.join(WEIGHTINGS)}")
    # A cell holds `mean_composite`, so `mean_composite` is a name an operator can
    # legitimately arrive with — it is what `list_metrics` publishes for the
    # quantity. Resolved to the row name before the closed-set check, never after:
    # the check is what makes an unknown name a refusal instead of an empty grid.
    # Keep what the caller actually sent. Resolution maps a catalog name onto the row
    # name, and `mean_total_ms` is IN the alias table (it has to be, for `history`)
    # while `total_ms` is not a measure `pivot` accepts — so refusing from the rebound
    # name told an operator "unknown metric 'total_ms'" about a string they never
    # typed, and about the one name the docs say this surface does not take. The
    # refusal names their input; the accepted set names both vocabularies.
    requested = metric
    metric = resolve_measure_name(metric)
    if metric not in PROJECTED_METRICS:
        refusal = f"unknown metric {requested!r} — expected one of {_metric_vocabulary(PROJECTED_METRICS)}"
        # Naming what is available is not enough when the caller asked for a
        # DIMENSION: they wanted a real capability, reached for it under the
        # wrong noun, and a bare list of measure names reads as "per-dimension
        # means are unavailable" — the false-absence answer that sends a reader
        # away from a route that exists. Checked against the rows rather than
        # guessed from the name, so the sentence is only added when it is true.
        #
        # Names the AXIS alone. A `rubric_dim` filter scopes the same cell one
        # layer down, but `reads.pivot` composes `filters` itself and
        # neither surface accepts one — so of the two remedies, only the axis is
        # reachable by whoever reads this. Advice a reader cannot act on is the
        # false capability claim this refusal exists to prevent, and it would
        # land here of all places: this string fires precisely when someone has
        # already reached for the capability under the wrong noun.
        for scoped_metric, (scope_field, scope_noun, _row_noun) in SCOPED_METRICS.items():
            if any(getattr(record, scope_field) == metric for record in records):
                refusal += f"; {metric!r} is a {scope_noun}, not a measure — aggregate {scoped_metric!r} with '{scope_field}' on an axis"
                break
        raise PivotError(refusal)
    for name in (row_factor, column_factor, *(filters or {})):
        _require_coordinate_name(name)
    predictions = _planned_cost_per_observation(predicted_cost, metric, (row_factor, column_factor))

    measure = _describe_aggregate(metric, profile=profile)

    of_metric = [r for r in records if r.metric == metric]
    selected = of_metric
    for factor, wanted in (filters or {}).items():
        selected = [r for r in selected if _axis_value(r, factor) == wanted]
    # A per-dimension score's catalogue range spans every scale, since the catalogue cannot know
    # which dimensions a table holds. The table can: when every row was judged on one scale, its
    # measure carries that scale's range, so a 1-5 table does not claim a floor of 0.
    if len(table_scales := {r.rubric_scale for r in selected if r.rubric_scale is not None}) == 1:
        measure = measure.model_copy(update={"value_range": SCALES[table_scales.pop()].value_range})

    # The no-pooling rule, enforced before any number is computed rather than as a badge after:
    # a table that has already averaged across subjects cannot be un-averaged by
    # a caveat, and the reader most likely to miss the caveat is the one reading
    # a single headline figure.
    if (
        measure.transferability_class != _POOLABLE_ACROSS_SUBJECTS
        and row_factor != "subject_id"
        and column_factor != "subject_id"
    ):
        subjects = {r.subject_id for r in selected}
        if len(subjects) > 1:
            raise PivotError(
                f"{measure.name!r} is {measure.transferability_class} and spans {len(subjects)} subjects — "
                "put subject_id on an axis or filter to one; pooling it across subjects compares "
                "measurements derived from different rubrics"
            )

    grouped: dict[tuple[str, str], list[ScoreRecord]] = {}
    for record in selected:
        grouped.setdefault((_axis_value(record, row_factor), _axis_value(record, column_factor)), []).append(record)

    # A mean of 1-5 levels and 1/0 pass/fail answers is neither a level nor a pass rate, so a cell
    # pooling both is refused before it is computed, like the subject pooling above.
    for (row, column), members in grouped.items():
        if len(scales := {r.rubric_scale for r in members if r.rubric_scale is not None}) > 1:
            raise PivotError(
                f"cell ({row}, {column}) pools rubric dimensions judged on different scales ({', '.join(sorted(scales))})"
                " — put 'rubric_dim' on an axis so each cell holds one dimension"
            )

    rows = sorted({row for row, _ in grouped})
    columns = sorted({column for _, column in grouped})
    # The identity keys this table groups or filters on, each of which must not silently pool versions.
    identity_keys = sorted(
        {name for name in (row_factor, column_factor, *(filters or {})) if name in _IDENTITY_KEY_VERSION_FIELDS}
    )
    # A table grouped by contestant names, per cell, the models that answered it (#684).
    by_contestant = bool({row_factor, column_factor} & _CONTESTANT_FACTORS)

    cells: list[PivotCell] = []
    measured: dict[tuple[str, str], PivotCell] = {}
    placed: set[str] = set()
    for row in rows:
        for column in columns:
            at_cell = grouped.get((row, column), [])
            plan = _plan_for(predictions, row, column, at_cell, (row_factor, column_factor))
            planned = plan.predicted if plan is not None else None
            unplanned = (
                sum(1 for record in at_cell if record.run_id not in plan.run_ids)
                if plan is not None and plan.run_ids
                else None
            )
            if plan is not None:
                placed.add(plan.label)
            if not at_cell:
                cells.append(
                    PivotCell(row=row, column=column, status=CELL_NOT_RUN, predicted=planned, n_unplanned=unplanned)
                )
                continue

            outcomes: dict[str, int] = {}
            for record in at_cell:
                outcomes[record.outcome] = outcomes.get(record.outcome, 0) + 1

            values_by_case: dict[str, list[float]] = {}
            for record in at_cell:
                if record.value is not None:
                    values_by_case.setdefault(record.test_case_id, []).append(record.value)

            n_valued = sum(len(v) for v in values_by_case.values())
            identity_versions = _pooled_identity_versions(at_cell, identity_keys)
            served = pooled_served_models(at_cell) if by_contestant else None
            if not values_by_case:
                cells.append(
                    PivotCell(
                        predicted=planned,
                        n_unplanned=unplanned,
                        row=row,
                        column=column,
                        status=CELL_UNMEASURED,
                        n_unmeasured=len(at_cell),
                        outcomes=outcomes,
                        identity_versions=identity_versions,
                        served_models=served,
                    )
                )
                continue

            # What the value is drawn over is the valued observations, so the qualifiers below read those.
            valued = [record for record in at_cell if record.value is not None]
            n_substituted = sum(1 for record in valued if record.substituted_deliveries > 0)
            qualifiers: dict[str, Any] = {
                "cassette_modes": sorted({r.cassette_mode for r in valued if r.cassette_mode is not None}),
                "cost_compositions": pooled_cost_compositions(valued) if metric == METRIC_COST_USD else [],
                "composite_basis": pooled_composite_basis(valued) if metric == METRIC_COMPOSITE else None,
                "identity_versions": identity_versions,
                "served_models": served,
                "n_substituted": n_substituted,
                "substitution_disclosure": (
                    _substitution_disclosure(n_substituted, len(valued)) if metric == METRIC_COST_USD else None
                ),
            }
            withheld = _cost_withheld(valued) if metric == METRIC_COST_USD else None
            if withheld is not None:
                cells.append(
                    PivotCell(
                        predicted=planned,
                        n_unplanned=unplanned,
                        row=row,
                        column=column,
                        status=CELL_WITHHELD,
                        withheld=withheld,
                        n=n_valued,
                        n_cases=len(values_by_case),
                        n_unmeasured=len(at_cell) - n_valued,
                        outcomes=outcomes,
                        **qualifiers,
                    )
                )
                continue

            value, sem = _aggregate(values_by_case, weighting)
            cell = PivotCell(
                predicted=planned,
                n_unplanned=unplanned,
                row=row,
                column=column,
                status=CELL_MEASURED,
                value=value,
                n=n_valued,
                n_cases=len(values_by_case),
                sem=sem,
                n_unmeasured=len(at_cell) - n_valued,
                outcomes=outcomes,
                **qualifiers,
            )
            cells.append(cell)
            measured[(row, column)] = cell

    # The column figures with the row axis collapsed — what the operator reads
    # when they stop breaking the comparison down. Computed from the observations
    # rather than from the cells above, because the row weighting is the entire
    # mechanism the Simpson's guard exists to catch.
    # A cost column whose observations the cells' own rule would withhold has no pooled figure either.
    pooled: dict[str, float] = {}
    for column in columns:
        in_column = [r for r in selected if _axis_value(r, column_factor) == column and r.value is not None]
        if metric == METRIC_COST_USD and _cost_withheld(in_column) is not None:
            continue
        by_case: dict[str, list[float]] = {}
        for record in in_column:
            assert record.value is not None
            by_case.setdefault(record.test_case_id, []).append(record.value)
        if by_case:
            pooled[column], _ = _aggregate(by_case, weighting)

    return PivotTable(
        metric=metric,
        measure=measure,
        formula=_effective_formula(
            metric,
            weighting,
            scoped=metric not in SCOPED_METRICS
            or SCOPED_METRICS[metric][0] in (row_factor, column_factor, *(filters or {})),
        ),
        weighting=weighting,
        row_factor=row_factor,
        column_factor=column_factor,
        rows=rows,
        columns=columns,
        cells=cells,
        simpsons_flags=_simpsons_flags(measured, pooled, rows, columns),
        n_observations=len(selected),
        n_filtered_out=len(of_metric) - len(selected),
        exclusions=exclusions or ProjectionExclusions(),
        # Narrowed to the runs that reached a cell. A subject filter or a metric
        # selection can remove a short run entirely, and a caveat about a run the
        # table never averaged is one the reader cannot act on or check.
        completeness_disclosures={
            run_id: sentence
            for run_id, sentence in (completeness_disclosures or {}).items()
            if any(record.run_id == run_id for record in selected)
        },
        n_degraded_observations=sum(1 for record in selected if record.run_id in (completeness_disclosures or {})),
        cost_compositions_differ=metric == METRIC_COST_USD
        and len(pooled_cost_compositions([r for r in selected if r.value is not None])) > 1,
        composite_bases_differ=metric == METRIC_COMPOSITE
        and (table_basis := pooled_composite_basis([r for r in selected if r.value is not None])) is not None
        and table_basis.ragged,
        cassette_mode_disclosure=cassette_mode_disclosure(
            {record.run_id: record.cassette_mode for record in selected if record.cassette_mode is not None}
        ),
        identity_pooling_disclosure=_identity_pooling_disclosure(cells),
        served_model_disclosure=_served_model_disclosure(cells),
        unplaced_predicted_models=sorted({plan.label for plan in predictions} - placed),
    )


#: The coordinate a pivot axis names the template by.
_TEMPLATE_FACTOR = "template_id"


class _Plan(NamedTuple):
    """One plan as a pivot places it: whose cells it describes, and its prediction per observation."""

    model: str
    template_id: str | None
    run_ids: frozenset[str]
    predicted: PredictedValue

    @property
    def label(self) -> str:
        """The plan as an unplaced list names it."""
        return self.model if self.template_id is None else f"{self.model} ({self.template_id})"


def _planned_cost_per_observation(
    estimate: CostEstimate | Sequence[PlannedCost] | None, metric: str, axes: tuple[str, str]
) -> list[_Plan]:
    """Each plan's predicted cost per observation, read off the estimate made before the run.

    A planned cell's prediction is its sweep's TOTAL (``n_observations`` draws), and a pivot cell's
    value is a mean per observation, so the prediction is divided by the planned observation count —
    and so is its band. That is not a rescaling of convenience: the mean of the ``n`` planned
    observations is their total over ``n``, so the band on the total, divided by ``n``, is exactly the
    band on that mean, at the same level and on the same assumptions.
    One prediction, read two ways.

    Args:
        estimate: The estimate, or the planned costs, or None.
        metric: The pivot's resolved metric.
        axes: The pivot's row and column factors.

    Returns:
        One plan per priced model and template; empty when no estimate was given.

    Raises:
        PivotError: An estimate was given to a pivot of another metric, or to one with no model axis — a
            planned model's cost describes neither, so its prediction would sit beside a number it does not
            predict — or it plans one model on one template twice, which leaves no answer to which prediction
            a cell carries.
    """
    if estimate is None:
        return []
    if metric != METRIC_COST_USD:
        raise PivotError(
            f"a predicted cost sits beside an observed cost, and this pivot aggregates {metric!r} — "
            f"pivot {METRIC_COST_USD!r} to set the estimate beside what was spent"
        )
    if CANDIDATE_MODEL_LEVER not in axes:
        raise PivotError(
            f"the estimate predicts cost per planned model, and neither axis is {CANDIDATE_MODEL_LEVER!r} — "
            "put the model on an axis so each prediction has the cells it planned"
        )
    planned = estimate.planned_costs() if isinstance(estimate, CostEstimate) else list(estimate)
    keys = [(cell.model, cell.template_id) for cell in planned]
    if repeated := sorted({key for key in keys if keys.count(key) > 1}, key=str):
        named = ", ".join(model if template is None else f"{model} on {template}" for model, template in repeated)
        raise PivotError(
            f"the estimate plans {named} more than once, so no cell can say which prediction it carries — hand "
            "the pivot one plan per model and template"
        )
    plans: list[_Plan] = []
    for cell in planned:
        if cell.predicted is None:
            continue
        n = cell.n_observations
        plans.append(
            _Plan(
                model=cell.model,
                template_id=cell.template_id,
                run_ids=frozenset(cell.run_ids),
                predicted=cell.predicted.model_copy(
                    update={
                        "value": cell.predicted.value / n,
                        "interval_low": None
                        if cell.predicted.interval_low is None
                        else cell.predicted.interval_low / n,
                        "interval_high": None
                        if cell.predicted.interval_high is None
                        else cell.predicted.interval_high / n,
                    }
                ),
            )
        )
    return plans


def _plan_for(
    plans: list[_Plan], row: str, column: str, records: list[ScoreRecord], axes: tuple[str, str]
) -> _Plan | None:
    """The plan a pivot cell's observations are, or None — matched on the model and the template the cell holds.

    The cell's template is the template axis's level when one axis is the template, else the one template every
    observation in it shares; a cell holding several, or an empty cell with no template axis, holds none a
    template's plan can claim. A plan that names no template matches on the model alone, and one naming the
    cell's template is preferred to it.
    """
    row_factor, column_factor = axes
    model = row if row_factor == CANDIDATE_MODEL_LEVER else column
    templates: set[str | None]
    if _TEMPLATE_FACTOR in axes:
        templates = {row if row_factor == _TEMPLATE_FACTOR else column}
    else:
        templates = {record.template_id for record in records}
    template = next(iter(templates)) if len(templates) == 1 else None
    at_model = [plan for plan in plans if plan.model == model]
    return next(
        (plan for plan in at_model if plan.template_id is not None and plan.template_id == template), None
    ) or next((plan for plan in at_model if plan.template_id is None), None)


# =============================================================================
# Frontier — the verdict surface: cheapest variant clearing the bar, per subject.
# =============================================================================

# A frontier point is one (subject, variant, identity version): the resolved contestant stack that
# would ship if it won, keyed on `variant_key`. Grouping by model alone would
# pool two tool-config or prompt-override contestants of the same model into one
# point — the same silent-pooling error refused across subjects — so the
# variant is the unit. `model` rides along as the label.
#
# The corollary, which is not optional: because the key is the shipping stack, the
# CONDITIONS a variant was measured under do not partition its points. Every scenario
# template and every cassette mode it ran under pools into the one point. Both are
# disclosed on the point and carried onto the verdict rather than split out — keying a
# condition would restamp and re-cohort every stored result to close a rendering
# gap. `comparison_sets` is where a reader goes to group by template.
#
# Domination is over three axes: pass^k (higher better), production-replicating
# cost (lower better), and mean total latency (lower better). mean-composite is shown
# beside pass^k as the secondary quality read but is NOT a domination
# axis — ranking on two correlated quality measures would double-weight quality.
#
# Domination is a claim that one contestant is worse, so it is DECIDED, by test, never read off
# point estimates: two contestants drawn from one distribution differ on every continuous axis,
# and comparing the numbers flagged one of them dominated about a third of the time. See
# `_dominance_p` for the rule. Latency is ranked on the MEAN, not a tail, for the same reason:
# a ranking axis must carry a test, and at the sizes a frontier sees a 95th percentile has
# none — a 95% interval on a p95 has no upper end below 72 observations — while a mean has a paired test over
# cases. The tail is read beside it (the run summary's `p95_total_ms` / `max_total_ms`).
#
# Cost is restricted to the production-replicating roles (candidate + inner_agent
# + external). A frontier computed on judge/simulator-inclusive cost ranks a
# variant by money the operator would never spend in production, which is a defect.


class FrontierError(ValueError):
    """A frontier was requested that cannot be answered honestly.

    Raised rather than returned, the way :class:`PivotError` is, so a bad request
    refuses identically on both surfaces. A bar outside the quality range, or one
    that is not a number, is the case that reaches it: silently clamping or
    dropping it would produce a verdict against a threshold the caller never chose.
    """


def normalize_bar(value: float | str | None) -> float | None:
    """Parse a caller's quality bar to a float, identically on every surface.

    Lives here, in one place as
    :func:`~threetears.evals.contracts.status_filter.normalize_status_filter` does,
    so REST (which hands a validated float) and MCP (which hands the raw string a
    tool argument arrives as) reach :func:`compute_frontier` through one coercion, rather
    than each parsing it and disagreeing about what a blank or a non-number means
    — the divergence class the parity gate exists to catch.

    Args:
        value: Raw bar from the caller. ``None`` or a blank string means the bar
            was not supplied.

    Returns:
        The bar as a float, or ``None`` when it was not supplied.

    Raises:
        FrontierError: The value is present but not a number.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError as e:
            raise FrontierError(f"bar {value!r} is not a number in [0, 1]") from e
    return float(value)


class TwoPillarDisclosure(EvalBaseModel):
    """Why the verdict rests on one quality pillar, stated on every answer.

    The verdict definition requires clearing the bar on BOTH a capability axis
    and a boundary/robustness axis, so a cheap model that is brittle
    off-distribution is disqualified rather than crowned cheapest. A scored rubric
    dim now carries its axis (``RubricScore.axis``), and the frontier's pass^k and
    composite read capability dims only, so a boundary dim no longer moves them
    either way. The boundary pillar is decided as guardrails — each arm against a
    control, in the analysis bundle — and the frontier, which ranks contestants
    against an absolute bar with no control, does not yet disqualify on it. So it is
    still descoped WITH disclosure: an operator must never mistake "nothing was
    disqualified" for "nothing was checked".
    """

    boundary_pillar_available: bool = False
    verdict_rests_on: str = "capability pillar (pass^k over capability criteria) alone"
    reason: str = (
        "boundary dimensions are left out of pass^k and the composite and decided as guardrails against a "
        "control in the analysis bundle; the frontier does not disqualify a contestant on one"
    )


def _template_span_entries(templates_by_run: Mapping[str, str | None]) -> list[str]:
    """The suites behind one frontier point, one ``"<template> (n runs)"`` entry per suite.

    The half of the span that VARIES between two contestants: which suites, and how many
    runs each contributed. Split out from :func:`_template_span_disclosure`, whose other
    ninety words are the same on every marked point in a scope — a surface rendering
    several marked contestants owes the RULE once and the sets once each, the collapse
    a host's model-disparity lines make for a uniform judge set and for its reason: a
    caveat repeated per row trains the reader to skip the line, precisely where a verdict
    is being named.

    The disclosure is composed from this list rather than re-deriving it, so a surface
    printing the sentence and a surface printing the sets cannot disagree about whether a
    point spans, or about which suites it spanned when it does.

    Args:
        templates_by_run: One ``template_id`` per contributing run, keyed by run id. A
            ``None`` is a real observed state — an ad-hoc run assembled from explicit
            test-case ids, per :attr:`~threetears.evals.contracts.models.EvalRun.template_id` — so it
            counts as its own suite rather than being dropped or merged into a neighbour.

    Returns:
        One entry per distinct template, ad-hoc last, or an empty list when fewer than two
        runs contributed or every one of them ran the same template — "this point does not
        span", the same predicate the disclosure reports as ``None``.
    """
    if len(templates_by_run) < 2:
        return []
    by_template: dict[str | None, list[str]] = {}
    for run_id, template_id in templates_by_run.items():
        by_template.setdefault(template_id, []).append(run_id)
    if len(by_template) < 2:
        return []
    return [
        f"{template_id or 'ad-hoc (no template)'} ({len(run_ids)} run{'' if len(run_ids) == 1 else 's'})"
        for template_id, run_ids in sorted(by_template.items(), key=lambda kv: (kv[0] is None, kv[0] or ""))
    ]


def _template_span_disclosure(span_entries: Sequence[str]) -> str | None:
    """The sentence a frontier point carries when its observations span scenario suites.

    A point is one ``(subject, variant_key, identity version)``, and ``variant_key`` digests the resolved
    contestant stack — everything that would ship, and nothing about the conditions it was
    measured under. ``template_id`` is one of those conditions: it is hashed into the
    CONTEXT key, and :func:`compute_comparison_sets` groups on it. So every scenario suite a
    variant was ever entered into pools into this one point, and ``pass_hat_k``,
    ``mean_composite`` and ``mean_total_ms`` are each one number over cases drawn from all
    of them.

    Pool-and-disclose, not key-it — the position the cassette span already records for its
    axis (:func:`cassette_mode_disclosure`), taken here for the same two reasons. The
    variant key is what would SHIP and a scenario suite ships nothing; and re-keying it
    would restamp and re-cohort every stored result to close a rendering gap.

    It matters more on this surface than on any other, because this one names a VERDICT.
    Two suites need not be of equal difficulty, so adding an easy suite to a scope
    raises the pooled pass^k of every variant entered into it, without a single contestant
    changing — and two
    contestants measured over different mixes are not being ranked on the same cases at
    all.

    What it does NOT claim: that one template id is one suite. A template is mutable and
    versioned, so two runs sharing an id can still have executed different cases;
    :func:`compute_comparison_sets`' ``case_set_differs`` badge owns that axis, and silence here is
    not evidence against it.

    Descriptive, never a verdict, on the rule :func:`cassette_mode_disclosure` follows: no
    comparison is refused and no number is adjusted. Measuring one variant across several
    suites is a legitimate thing to do.

    Args:
        span_entries: This point's suites as :func:`_template_span_entries` renders them.
            Taken rather than re-derived, so the sentence and the per-point list a surface
            may print instead are one predicate over one grouping.

    Returns:
        The disclosure, rendered verbatim by a surface showing this point ALONE, or
        ``None`` when the point does not span. A surface rendering several marked points
        states this rule once and prints ``span_entries`` per point: the only thing that
        differs between two of these sentences is the parenthetical.
    """
    if not span_entries:
        return None
    detail = ", ".join(span_entries)
    return (
        f"This contestant's observations span {len(span_entries)} scenario templates ({detail}). pass^k, "
        "composite and latency here are each ONE number over cases pooled from all of them, and two suites "
        "need not be of equal difficulty — so this is a weighted average over whichever mix happened to run, "
        "not a score on any one suite, and a contestant measured over a different mix is not being ranked on "
        "the same cases. Group by template (`comparison_sets` does) before reading a gap between rows as a "
        "difference between the contestants."
    )


class FrontierDominator(EvalBaseModel):
    """One contestant shown to beat another point on every axis it measured, named as a ROW is named.

    The identity triple of :class:`FrontierPoint`, carried rather than projected down to
    ``model``. A point is one ``(subject, variant, identity version)`` and every surface renders it as
    ``<model> · <variant_key>``, so a dominator list of bare model names is strictly less
    identifying than the thing it points at: two variants of one model collapse into one
    entry, and the surviving string is the dominated row's OWN model, which reads as a row
    dominating itself: a point ``model-a · key1`` rendered as "dominated by model-a" while
    two distinct variants of ``model-a``, ``key2`` and ``key3``, beat it.

    **Identity, not a rendered label.** How much of the key to show is a surface's
    decision — MCP truncates it to match its own variant column — and a pre-formatted
    string here would either mismatch that column or deny a web surface the full key.
    ``frontier``'s whole job is *which variant should I move to*, and that answer is the
    key.
    """

    variant_key: str
    model: str
    #: The identity version stamped beside ``variant_key``, carried for the same reason the
    #: key is: since the lenses partition on it, two points can now share a model AND a key
    #: and differ only here — and under a context-side bump they DO, since the shared counter
    #: moves the variant stamp without moving the digest. Without this the self-domination defect above returns in a new form —
    #: `<model> · <key>` identifies BOTH rows, so the dominator label again names something
    #: indistinguishable from the row it dominates. Cross-version domination is correct and
    #: is not blocked: the gate says these are two contestants, and two contestants beating
    #: each other on measured axes is what a frontier is for. Only the LABEL needed fixing.
    variant_identity_version: int
    #: The Holm-adjusted p the domination was decided on, below :data:`~threetears.evals.analysis.stats.SIGNIFICANCE_ALPHA`
    #: (see :func:`_dominance_p`). Carried because a verdict a reader cannot check against its statistic is
    #: an assertion. ``None`` on a dominator stored before domination was tested: that one was read off
    #: point estimates.
    p_value: float | None = None


#: How a frontier verdict's pick stands on cost against the other contestants that cleared the bar with a
#: cost — see :attr:`FrontierVerdict.cost_decision`.
FrontierCostDecision = Literal["shown_cheapest", "not_separated", "untested", "only_cleared"]


class FrontierCostTie(EvalBaseModel):
    """A contestant that cleared the bar with a cost, which the verdict's pick was NOT shown cheaper than.

    Named as a row is named (the identity triple :class:`FrontierDominator` carries, for its reasons), with
    the cost it was read at and the p the comparison reached, so a reader can check why it stays in the set.
    """

    variant_key: str
    model: str
    variant_identity_version: int
    production_replicating_cost: float
    #: The Holm-adjusted p of the test that the pick costs less than this contestant, at or above
    #: :data:`~threetears.evals.analysis.stats.SIGNIFICANCE_ALPHA`. ``None`` when no test could decide — fewer
    #: than two cases carried a cost on a side, or every case differed by one amount over too few cases for the
    #: exact test to reach α — which is untested, not a tie the data showed.
    p_value: float | None = None


class FrontierPoint(EvalBaseModel):
    """One contestant's position on quality x cost x latency, within a subject.

    A point is one ``(subject, variant, predicate version)``. ``model`` is the human
    label; ``variant_key`` is the grouping identity, and it is qualified by the
    identity version stamped beside it — two keys carrying different stamps are ranked as
    separate contestants, since the stamp cannot say which predicate moved and merging on
    an unverifiable digest is the direction nothing downstream undoes.

    Every axis carries its own denominator (``n_pass_cases``, ``n_composite_cases``,
    ``n_cost``, ``n_latency``) because the axes are independently sourced: a
    point's cost mean can rest on fewer results than its quality when some observed no
    production-role cost, and a mean over 2 of 30 that reads like a mean over 30
    is exactly the misreading these counts exist to prevent.
    """

    variant_key: str
    model: str
    #: The predicate version that minted ``variant_key``. Carried onto the row because
    #: this surface GATES on it — two keys carrying different stamps are ranked separately,
    #: because the stamp cannot say which predicate moved and a wrong merge is the direction
    #: nothing downstream undoes — and a reader who meets that split needs the number that
    #: caused it.
    variant_identity_version: int
    #: Set when that predicate is not this build's.
    identity_version_disclosure: str | None = None

    # Quality — pass^k is the headline (and the domination axis), mean-composite
    # the secondary read shown alongside it. Both always travel together.
    # pass^k is the chance that `k` attempts at a case ALL pass (τ-bench; never pass@k, the
    # chance that at least one does), estimated without bias per case and averaged over the
    # cases measured at least `k` times — a case's attempts pooled across every run of one
    # cell (`scoring.pool_pass_hat_k`), so a repeat run adds depth rather than a second copy.
    # `k` is the subject's: every point is ranked at the same depth (see SubjectFrontier.k).
    # None when no case was scored that deep: nothing was measured,
    # which is not a pass^k of zero, and an unmeasured axis never wins a domination.
    pass_hat_k: float | None
    #: The depth ``pass_hat_k`` is read at — the subject's, so every point is ranked at one.
    k: int = 1
    #: The cases ``pass_hat_k`` averages over: case-within-cell units scored at least ``k`` times.
    n_pass_cases: int = 0
    #: pass^1..pass^K for this point, each with its own case count. pass^1 is the per-case pass
    #: rate; the deeper points show how fast reliability decays with repetition.
    pass_hat_k_curve: list[PassHatPoint] = []
    #: The interval on ``pass_hat_k`` at :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`, over its
    #: cases (:func:`~threetears.evals.analysis.stats.case_rate_interval`): the cases are the draws, and each
    #: case's own estimate is noisy too. ``None`` below two cases, where none is estimable, and on a point
    #: stored before pass^k carried one.
    pass_hat_k_ci_low: float | None = None
    #: The high end of that interval; ``None`` exactly when ``pass_hat_k_ci_low`` is.
    pass_hat_k_ci_high: float | None = None
    #: Attempts behind this point with nothing for pass^k to conjoin — no goal-state check and no judge — left
    #: out of pass^k and its curve rather than read as failures (#688).
    n_pass_no_criterion: int = 0
    #: Why ``pass_hat_k`` is None when the reason is that no attempt carried a pass criterion: the point has
    #: no pass^k, not one of 0, and so no cost per acceptable outcome either. None otherwise.
    pass_hat_k_unmeasured_reason: str | None = None
    #: How this point's pass^k reads against the bar, decided by its interval the way every campaign bar is
    #: (:func:`~threetears.evals.analysis.stats.interval_clears`): ``cleared``, ``missed``, ``undecided`` (the
    #: interval straddles the bar — neither a pass nor a failure), ``no_interval`` (fewer than two cases, not
    #: read) or ``no_data``. ``None`` when no bar was supplied.
    bar_decision: BarDecision | None = None
    mean_composite: float | None = None
    composite_sem: float | None = None
    n_composite_cases: int = 0
    #: What ``mean_composite`` was meaned over (:func:`pooled_composite_basis`), ``ragged`` when this point's
    #: results were scored on different dimension sets (#638). ``None`` when no composite was pooled, and on a
    #: point stored before the basis was carried — which says nothing about whether that pool was ragged.
    composite_basis: CompositeBasis | None = None

    # Cost — production-replicating only. ``None`` (never 0) when no
    # result at this point reported one; ``cost_is_partial`` when some did and some did
    # not — a result contributes nothing here when its production roles observed no cost,
    # when it carries no usage rows at all, or when a substituted delivery withheld the
    # figure (its background dollars were never spent, so the observed sum would understate
    # production), or when it is no turn the candidate took: a call its model refused, or a cell
    # the harness faulted, whose shortened spend would let the rig make the point look cheaper.
    production_replicating_cost: float | None = None
    n_cost: int = 0
    cost_is_partial: bool = False
    #: The role sets this point's cost mean was drawn over. More than one entry means the
    #: contestant's own results priced different things — and where two POINTS disagree,
    #: their costs are not comparable, so ranking one cheaper than the other is ranking a
    #: convention rather than a config. Empty when no result recorded a composition.
    cost_compositions: list[list[str]] = []
    # cost / pass-probability — the alternate denominator. The probability is pass^1, the
    # chance ONE attempt passes, because the cost is the mean of one attempt: dividing it by
    # pass^k would price a k-attempt streak at one attempt's cost. ``None`` when
    # cost is unknown or pass^1 is zero (dividing by a zero pass rate is undefined,
    # not "infinitely expensive").
    cost_per_acceptable_outcome: float | None = None
    #: What the runs behind ``production_replicating_cost`` set away from the subject's production configuration
    #: (#571), each run's own footing read off the host's sweepable declarations: the cost is what production would
    #: spend only where no run moved anything (``moved_nothing``). ``None`` when the frontier was computed without
    #: the host's declarations, and on a point stored before it was read — nobody checked, never "nothing moved".
    production_footing: PooledProductionFooting | None = None

    # Latency — total wall-clock ms, the performance axis. Mean over the turns the candidate took
    # (`delivered_a_turn`) that harvested a total; ``None`` when none did. The mean, not a tail: it is
    # the latency statistic a domination can be TESTED on at these sizes (see the section comment).
    mean_total_ms: float | None = None
    n_latency: int = 0
    #: Results whose candidate's model refused or errored with no turn taken — failures pass^k counts,
    #: left out of the cost and latency above. Equal to ``n_results`` (less any faulted) when every call
    #: failed that way: then the absent cost and latency mean every result failed, not that nothing was
    #: measured, and the point can dominate nothing on either axis.
    n_no_turn: int = 0

    # What this point rests on across all axes.
    n_results: int = 0
    n_cases: int = 0

    #: Set when this point's observations did NOT all record one cassette mode. The
    #: variant key is the stack that would SHIP, and cassette mode is apparatus rather
    #: than product, so it is deliberately absent from that key (it lives in the context
    #: key) — which means a capture arm and a replay arm of one contestant pool into this
    #: single point, correctly by that design and silently by this surface's rendering.
    #: The pooling is disclosed rather than split, because splitting would re-answer the
    #: variant/context identity split as a side effect of a rendering fix and would move
    #: every stored variant's identity with it. It qualifies EVERY axis here, not only
    #: cost: ``cost_is_partial`` already says a replayed arm withheld its cost, and
    #: ``pass_hat_k`` / ``mean_composite`` / ``cost_per_acceptable_outcome`` carry no such
    #: mark while resting on the same mixed pool — the ratio worst of all, since it
    #: divides a partial cost by a confounded pass rate and the two errors do not cancel.
    cassette_mode_disclosure: str | None = None

    #: Set when this point's observations did NOT all come from one scenario template. The
    #: same shape of pooling as ``cassette_mode_disclosure``, one axis over, and for the same
    #: reason: ``template_id`` is a measurement CONDITION (it is in the context key, and
    #: ``comparison_sets`` groups on it) while the variant key is the stack that would ship,
    #: so every suite a variant entered pools into this one point — correctly by that design
    #: and silently by this surface's rendering. It qualifies every axis here, not one:
    #: ``pass_hat_k``, ``mean_composite`` and ``mean_total_ms`` are each a weighted average
    #: over a mix of suites of unequal difficulty, and ``cost_per_acceptable_outcome``
    #: divides by that same pooled pass rate. Without it, adding an easy suite to a scope
    #: lifts the headline of every variant entered into it, with nothing on the row to show it.
    template_span_disclosure: str | None = None

    #: The suites behind that disclosure, one ``"<template> (n runs)"`` entry each — the only
    #: part of it that differs between two marked contestants. Carried beside the sentence
    #: rather than only inside it because a surface showing N marked points would otherwise
    #: print the same ninety-word rule N times, and a caveat printed N times is one nobody
    #: reads. Set exactly when ``template_span_disclosure`` is (both come from
    #: :func:`_template_span_entries`), so a surface can key on either without the two
    #: disagreeing about whether this point spans.
    template_span: list[str] = []

    #: ``run_id -> DEGRADED sentence`` for the runs behind this point that measured
    #: less than the matrix they promised. The third pooling disclosure on this model,
    #: and the one whose axis is not a measurement CONDITION but the measurement's own
    #: extent: a short run's cells are fewer, so every denominator on this row —
    #: ``n_pass_cases``, ``n_composite_cases``, ``n_cost``, ``n_latency`` — is partly
    #: made of a matrix that never finished. It moves the verdict rather than only
    #: qualifying it: admitting two truncated runs can take one contestant's
    #: pass^k from 1.00 over 3 cases to 1.00 over 4, lower its mean cost by a third,
    #: and flip its rival from *on frontier* to *dominated*.
    completeness_disclosures: dict[str, str] = {}

    #: Which models the provider's responses named as having answered this contestant's candidate calls (#684):
    #: ``one``, ``pooled`` — the contestant asked for one model id (a floating alias, say) and several models
    #: answered, so every axis here is a mixture of them — or ``unrecorded``, where some response named none and
    #: whether one model answered cannot be established. A contestant is keyed at launch, before any response
    #: names a model, so the mixture is disclosed rather than split. ``None`` when no result's candidate made a
    #: call, and on a point stored before it was read.
    served_models: ServedModelReading | None = None

    # Domination — a dominated point is grayed, never dropped: silently removing a
    # cheap-but-brittle variant looks identical to it never having run.
    #: True only when another point is SHOWN to beat this one (:attr:`dominance` ``dominated``).
    #: False is not a claim that nothing beats it. On a point stored before domination was tested
    #: (``dominance`` is ``None``), this was read off point estimates.
    dominated: bool = False
    #: ``dominated`` — some other point is shown better on every axis this one measured, by the test
    #: :func:`_dominance_p` states; ``not_separated`` — this point was tested against at least one other
    #: and no domination was shown, which says nothing about whether one exists; ``untested`` — no
    #: test against any other point could decide (fewer than two cases on an axis, no shared axis, or an axis
    #: on which every case differed by one amount over too few cases for the exact test to reach α — the
    #: reading :func:`~threetears.evals.analysis.stats.level_difference` also calls untested).
    #: ``None`` on a point stored before domination was tested.
    dominance: FrontierDominance | None = None
    #: Every point shown to beat this one on every axis, each carrying the identity a row
    #: carries and the p it was shown at. In the order the points are sorted, which is the order
    #: the table prints them, so a reader scanning for a named dominator meets them in that order.
    dominated_by: list[FrontierDominator] = []


class FrontierVerdict(EvalBaseModel):
    """The cheapest variant clearing the operator's bar for one subject — or, where the data cannot pick
    one, the set it is among.

    Present only when a bar was supplied AND at least one point both cleared it — its pass^k
    interval wholly at or above the bar, the rule every campaign bar is read by — and reported a
    production-replicating cost — a pick cannot be named cheapest
    on a cost nobody observed. "Cheapest" is decided by test, as domination is: the lowest point cost
    is named the cheapest only when it is shown cheaper than each rival (:attr:`cost_decision`), and
    otherwise the verdict names it beside every rival it was not shown cheaper than (:attr:`tied_with`).
    Two contestants drawn from one distribution always differ in point cost, so the lowest one alone
    would crown one of them every time. ``cost_is_partial`` rides along so a pick made on a
    partially-observed cost basis says so, and ``cassette_mode_disclosure`` for the
    stronger version of the same duty: a verdict is a RECOMMENDATION, so one drawn from
    a pool in which some observations replayed their third-party half has to carry that
    on the verdict itself, not only on a table row the reader may never scroll back to.
    ``template_span_disclosure`` rides along on exactly that rule — the pass^k this pick
    was named cheapest-above-the-bar on may be an average across scenario suites of
    different difficulty, which is a fact about the recommendation and not about a row.
    ``completeness_disclosures`` rides along for the sharpest version of it: the bar was
    applied to a pass^k, and the cheapest-by comparison to a cost, that partly rest on
    runs which never finished their matrix.
    """

    #: The contestant with the lowest point cost among those that cleared the bar with a cost. It is
    #: named THE cheapest only when :attr:`cost_decision` is ``shown_cheapest`` (or ``only_cleared``);
    #: otherwise it is one member, listed first, of the set :attr:`tied_with` completes.
    variant_key: str
    model: str
    pass_hat_k: float
    #: The depth ``pass_hat_k`` was read at — the subject's, the same as every point's.
    k: int = 1
    #: The interval the bar was decided on (:attr:`FrontierPoint.pass_hat_k_ci_low`). ``None`` only on a
    #: verdict stored before the bar read intervals: that one compared the point pass^k with the bar.
    pass_hat_k_ci_low: float | None = None
    pass_hat_k_ci_high: float | None = None
    production_replicating_cost: float | None = None
    cost_is_partial: bool = False
    #: The picked point's :attr:`FrontierPoint.production_footing`: the pick is made on a cost that is
    #: production's only where its runs moved nothing, so what they moved rides on the recommendation (#571).
    production_footing: PooledProductionFooting | None = None
    cassette_mode_disclosure: str | None = None
    template_span_disclosure: str | None = None
    completeness_disclosures: dict[str, str] = {}
    #: Rides along on the rule the four above ride on: a verdict is a RECOMMENDATION, so
    #: a pick whose key was minted by a superseded predicate says so where the
    #: recommendation is, not only on a table row the reader may never scroll back to.
    identity_version_disclosure: str | None = None
    #: The picked point's :attr:`FrontierPoint.served_models`, carried for the reason the disclosures above are:
    #: a recommendation of a contestant several models answered is a recommendation of their mixture.
    served_models: ServedModelReading | None = None
    #: The version stamped beside ``variant_key``, carried for the reason
    #: :attr:`FrontierDominator.variant_identity_version` is: since the lenses partition on
    #: it, ``(model, variant_key)`` no longer identifies a row, and a verdict a reader
    #: cannot match back to its row names an ambiguous pick. It is also what lets every
    #: reader of this surface decide "superseded?" from ONE signal — the version compared
    #: against the build's — rather than the row-renderers using the version and the verdict
    #: renderer using whether the sentence above is set, which are two predicates for one
    #: question and only agree while one producer keeps them in step.
    variant_identity_version: int
    #: Whether the pick is SHOWN cheaper than every other contestant that cleared the bar with a cost, by the
    #: separation test the frontier's dominance reads, one comparison per rival, Holm-adjusted together
    #: (:func:`_cost_ties`). ``shown_cheapest`` — every comparison separated in the pick's favour;
    #: ``not_separated`` — at least one rival could not be shown dearer, so the data says only that the
    #: cheapest is among the pick and :attr:`tied_with`, and the verdict names that set rather than a winner;
    #: ``untested`` — no test against any rival left in the set could decide (too few priced cases, or every
    #: case differing by one amount over too few cases for the exact test to reach α); ``only_cleared`` — no
    #: other contestant cleared the bar with a cost. ``None`` on a verdict stored before the pick was tested:
    #: that one was the lowest point cost, a winner the data may not have shown.
    cost_decision: FrontierCostDecision | None = None
    #: The rivals the pick was not shown cheaper than, ordered by point cost. Empty when ``cost_decision``
    #: is ``shown_cheapest`` or ``only_cleared``.
    tied_with: list[FrontierCostTie] = []


class SubjectFrontier(EvalBaseModel):
    """One subject's frontier: its points, and the verdict over them.

    Per-subject always: composite quality is derived from the subject's own rubric
    and is not comparable across subjects, so two subjects are two frontiers, never one.
    ``n_cleared_bar`` is carried so an absent verdict distinguishes "no variant cleared
    the bar" from "some cleared it but none has a known cost to be cheapest by", and
    ``n_undecided_bar`` so "none cleared" distinguishes "shown short" from "too few cases to say".
    """

    subject_id: str
    subject_label: str = ""
    #: The depth every point's ``pass_hat_k`` is read at, the bar applied to and domination
    #: decided on: the smallest ``k_runs`` among the subject's ranked runs, the depth every run
    #: here was commissioned to. One depth for the subject because pass^3 and pass^1 are
    #: different quantities, and ranking one contestant on each ranks the depths. A contestant
    #: measured deeper keeps its extra depth as precision (and on its curve), not as a harder bar.
    k: int = 1
    points: list[FrontierPoint] = []
    verdict: FrontierVerdict | None = None
    n_cleared_bar: int = 0
    #: Points whose pass^k interval straddles the bar: neither cleared nor missed.
    n_undecided_bar: int = 0


class FrontierResult(EvalBaseModel):
    """The verdict surface across every subject, plus its disclosures.

    ``bar`` is echoed back so the answer names the threshold its verdicts were
    made against (the bar is the caller's, never invented). ``two_pillar``
    states the descoped boundary pillar on every answer. ``exclusions`` and the
    ``n_*`` counts keep an all-excluded corpus from rendering as an empty one,
    exactly as :class:`PivotTable` does.
    """

    bar: float | None = None
    #: The 1–5 level a capability criterion had to reach for an attempt to pass, in every pass^k here
    #: (#642): the behavior's declared threshold
    #: (:meth:`~threetears.evals.contracts.host.BarRegistry.pass_threshold`) where the caller had one, else 3.
    #: A frontier stored before this was recorded defaults to 3, which is the threshold every pass^k was
    #: computed at then.
    rubric_threshold: int = 3
    subjects: list[SubjectFrontier] = []
    two_pillar: TwoPillarDisclosure = TwoPillarDisclosure()
    n_results: int = 0
    n_filtered_out: int = 0
    exclusions: ProjectionExclusions = ProjectionExclusions()
    #: ``run_id -> DEGRADED sentence`` for every run behind this frontier that came up
    #: short of its matrix, corpus-wide. The per-point copies say which contestants they
    #: land on; this one is the set, so a surface can state the rule once.
    completeness_disclosures: dict[str, str] = {}
    #: How many of ``n_results`` came from those runs — the weight of the caveat.
    n_degraded_observations: int = 0
    #: The distinct identity versions this answer's keyed observations were stamped at,
    #: ascending. Partitioning stops two stampings of one stack
    #: being RANKED together; it cannot stop them appearing as two rows, so the span and
    #: the sentence below are what keep a doubled row from reading as a mystery.
    identity_version_span: list[int] = []
    #: The one-sentence form of that span, or ``None`` when the answer rests on a single
    #: predicate. Computed from the placed rows rather than from the assembled
    #: points, so an answer whose points were all filtered out still says what
    #: it spanned.
    identity_span_disclosure: str | None = None


#: What both lenses group on. The predicate version is IN the key rather than a filter
#: applied beside it, because the join this gate protects *is* the grouping dict — a gate
#: keyed on a projection of the group key admits exactly the class it exists to refuse,
#: and the projected-away field is the gap.
ContestantKey = tuple[str, int]


def _contestant_key(result: EvalResult) -> ContestantKey:
    """Key a result to its contestant, and to the identity version stamped beside that key.

    One key, both lenses. ``frontier`` and ``history`` rank and trend the same
    contestants, so a grouping policy that lived in each of them separately would be
    two authorings of one rule — which is how the version gate came to exist on the
    bundle path and not here.

    The comparison this key performs is **equality, and there is no ``>=`` reading**.
    Keys stamped at different identity versions are incomparable in both directions: an
    older key is not "good enough" for a newer reader, and a newer key is not admissible
    to an older series. That is what makes this a partition rather than a compatibility
    check, and it is why a version mismatch is never resolved in one side's favour.

    **The stamp is one counter over two predicates, so it is a conservative
    discriminator rather than a precise one.** ``IDENTITY_VERSION`` backs the variant and
    context predicates together, so a bump on the CONTEXT side moves the number stamped
    beside a ``variant_key`` whose own predicate did not change — v9 and v10 are both that
    case. Such a pair carries a byte-identical digest under two stamps and is partitioned
    here anyway. That is deliberate and errs the safe way: the stamp cannot say WHICH
    predicate moved, and merging on a digest whose provenance is unverifiable is the wrong
    direction, since nothing downstream undoes a wrong merge. What it costs is a split that
    is sometimes narrower than necessary, which the disclosures state honestly rather than
    dressing up as a predicate change.

    Args:
        result: The observation to place.

    Returns:
        ``(variant_key, identity_version)`` — the key the runner stamped and the predicate
        version stamped beside it.
    """
    return (result.variant_key, result.identity_version)


def _identity_version_disclosure(version: int) -> str | None:
    """Say so when a contestant's key was minted by a predicate this build does not use.

    Derived from the same equality :func:`_contestant_key` groups on, never from a
    correlate: a sentence that agreed with the grouping on the observations we happen
    to have and disagreed on the one it exists for would be worse than none.

    Args:
        version: The predicate version stamped on the contestant's key.

    Returns:
        The sentence, or ``None`` when the key is this build's.
    """
    if version == IDENTITY_VERSION:
        return None
    return (
        f"stamped at identity version v{version}, not this build's v{IDENTITY_VERSION} — "
        "ranked separately from contestants stamped at a different version rather than pooled with them. "
        "The stamp does not say which predicate moved, so this may be a narrower split than the change warranted"
    )


def _identity_version_span(results: Iterable[EvalResult]) -> list[int]:
    """The distinct identity versions the KEYED observations in an answer were stamped at.

    Args:
        results: Every observation the answer is built over. Pass the placed rows
            rather than the assembled points: an answer whose subjects were all
            filtered out still spanned what it spanned, and a span derived downstream
            goes missing exactly when the reader most needs it.

    Returns:
        The versions, ascending.
    """
    return sorted({r.identity_version for r in results})


def _identity_span_disclosure(versions: Sequence[int]) -> str | None:
    """Explain a corpus that spans identity versions, so a doubled row is not a mystery.

    Partitioning stops two stampings of one stack being ranked as rival
    contestants; it does not stop them APPEARING as two rows. Without this sentence a
    reader meets one model twice, under two digests, with nothing saying why — which is
    the second half of the same defect and the one a per-row flag cannot answer.

    Args:
        versions: The span from :func:`_identity_version_span`.

    Returns:
        The sentence, or ``None`` when the answer rests on a single predicate — where
        there is no split to explain.
    """
    if len(versions) < 2:
        return None
    named = ", ".join(f"v{v}" for v in versions)
    return (
        f"spans identity versions {named}. Two keys stamped at different versions cannot be shown FROM THE "
        "STAMP ALONE to describe the same contestant, so they are listed separately — one model appearing "
        "more than once here is that split, not two rival configurations. The stamp covers both the variant "
        "and the context predicate, so some of these splits are narrower than the change that caused them."
    )


def _frontier_point(
    results: list[EvalResult],
    *,
    contestant: ContestantKey,
    k: int,
    cell_of_run: Mapping[str, Hashable],
    rubric_threshold: int,
    cassette_modes_by_run: Mapping[str, str],
    templates_by_run: Mapping[str, str | None],
    degraded_by_run: Mapping[str, str],
    footing_by_run: Mapping[str, ProductionFooting | None] | None,
) -> tuple[FrontierPoint, _ContestantCases]:
    """Aggregate one contestant's results into a single frontier point, and the per-case values behind it.

    Every axis aggregates over its own measured subset — pass^k reuses
    :func:`~threetears.evals.contracts.scoring.pool_pass_hat_k` (which drops infra-excluded
    iterations, and pools a case's attempts across the runs of one cell), composite drops the nulls :func:`~threetears.evals.contracts.scoring.result_composite`
    returns for infra-excluded and no-rubric results, cost sums only the
    production-replicating roles per result and means over the results that
    reported one, and latency means over the results that harvested a total. The
    subsets can legitimately differ in size, so each carries its own denominator.

    Args:
        results: One contestant's observations (all sharing a model, and a
            ``variant_key`` when captured). Never empty.
        contestant: The :func:`_contestant_key` these results were grouped under. Passed
            rather than re-derived off ``results[0]``: the key is what placed them
            together, so reading the identity back out of it makes "no row in this pool
            can disagree" structural instead of a claim a later change to the key could
            silently falsify — the same reason the gate lives IN the group key rather than
            beside it.
        k: The subject's depth, at which ``pass_hat_k`` is read (:attr:`SubjectFrontier.k`).
        cell_of_run: Run id → the cell its attempts are repeats of
            (:func:`~threetears.evals.contracts.scoring.pass_hat_k_cell`), so two runs of one
            configuration pool their attempts at a case and two configurations never do.
        rubric_threshold: Pass threshold forwarded to pass^k.
        cassette_modes_by_run: Every candidate run's recorded ``cassette_mode``, keyed by
            run id; narrowed here to the runs these results came from. **Required and
            deliberately without a default**, for the reason
            :func:`~threetears.evals.contracts.usage_capture.production_replicating_cost` refuses one:
            the map's absence and a genuinely uniform pool both yield no disclosure, so a
            default would let a caller that never supplied it render a live-versus-replayed
            pool as clean — the single failure this parameter exists to make impossible.
        templates_by_run: Every candidate run's ``template_id``, keyed by run id; narrowed
            here to the runs these results came from. **Required and deliberately without a
            default**, for the reason its neighbour above is: an absent map and a genuinely
            single-suite contestant both yield no disclosure, so a default would let a
            caller that never supplied it render a mixed-difficulty pool as clean.
        degraded_by_run: ``run_id -> DEGRADED sentence`` for the candidate runs that
            measured less than their matrix, from :func:`degraded_run_disclosures`;
            narrowed here to the runs these results came from. **Required and without a
            default**, on the rule its two neighbours state: an unsupplied map and a pool
            of genuinely complete runs both yield no disclosure, so a default would let a
            caller that forgot it rank a truncated contestant as if it were whole.
        footing_by_run: Each candidate run's production footing (``None`` for a run nobody could check), or
            ``None`` when the frontier was given no host declarations to read one with; narrowed here to the
            runs these results came from.

    Returns:
        A :class:`FrontierPoint` with quality (and its interval), cost, latency, and every
        denominator, and the per-case values each axis averages, which a domination test pairs
        on. Domination and the bar are filled by the caller, which needs the subject's other
        points and the bar to decide them.
    """
    from threetears.evals.contracts.usage_capture import count_substituted_deliveries, production_replicating_cost

    model = results[0].model
    # Unpacked from the group key, never re-derived off a row.
    variant_key, identity_version = contestant

    # pass^k — the canonical estimator, over this contestant's attempts pooled per case within
    # each cell: a repeat run of one configuration deepens its cases, while the same case under
    # another context (a second template, a replayed cassette) stays a case of its own beside it.
    pooled = pool_pass_hat_k(results, cell_of_run=cell_of_run, k=k, rubric_threshold=rubric_threshold)
    pass_hat_k = pooled["pass_hat_k"]
    pass_hat_1 = pass_hat_k_at(pooled["pass_hat_k_curve"], 1)["pass_hat_k"]
    # The same cases the headline averages, one estimate each: the interval runs over them, and a
    # domination test pairs two contestants on the test cases both measured.
    attempts, _ = pool_pass_hat_k_attempts(results, cell_of_run=cell_of_run, rubric_threshold=rubric_threshold)
    unit_estimates: list[float] = []
    unit_attempts = 0
    pass_by_case: dict[str, list[float]] = {}
    for (_, _, _, _, test_case_id), case_attempts in attempts.items():
        estimate = case_pass_hat_k(case_attempts, k)
        if estimate is None:
            continue
        unit_estimates.append(estimate)
        unit_attempts += len(case_attempts)
        pass_by_case.setdefault(test_case_id, []).append(estimate)
    pass_interval = case_rate_interval(unit_estimates, max_effective_n=unit_attempts / k) if unit_estimates else None

    # composite — case-weighted mean of the 0-1 scores, dropping the nulls that
    # mark infra-excluded and no-rubric results (a candidate failure scores 0.0
    # and stays in, depressing the mean rather than vanishing).
    values_by_case: dict[str, list[float]] = {}
    for result in results:
        composite = result_composite(result)
        if composite is not None:
            values_by_case.setdefault(result.test_case_id, []).append(composite)
    if values_by_case:
        mean_composite, composite_sem = _aggregate(values_by_case, WEIGHTING_EQUAL_PER_SCENARIO)
    else:
        mean_composite, composite_sem = None, None

    # cost — production-replicating roles only, meaned over the results that
    # reported one. None when none did; partial when some did and some did not.
    # Reads the RAW persisted `usage` where the run-summary aggregate reads
    # `resolve_result_usage`. The two once answered differently — a result whose
    # decomposition could be reconstructed at read counted toward that denominator and not
    # this one — which is why this site was deliberately left divergent: this is the surface
    # that RANKS contestants, and moving it was a ranking decision rather than a reporting
    # one. That reconstruction no longer exists, and with it gone the resolver returns rows
    # this call cannot distinguish from the raw field: identical on `captured`, `None` on
    # both sides when nothing was observed, and an empty list on a cancelled cell, which
    # sums to `None` exactly as the absent field does. So the two agree here in every case,
    # and reading the raw field is no longer a divergence to justify — it is the shorter way
    # to the same number. Aligning the call sites would be tidying, not a fix.
    #
    # The substitution disclosure is NOT part of that divergence and is read per result
    # regardless: a seeded or replayed delivery spent neither inner-agent dollars nor
    # tool credits, so letting it contribute its observed sum here would rank a
    # substituted contestant as cheaper than a live one on a difference in apparatus
    # rather than in configuration. Withholding is what keeps the ranking a comparison.
    #
    # The population is the turns the candidate took (`delivered_a_turn`), the one every comparison cost reads
    # (#619). A call the candidate's model refused or errored on took no turn, so its dollars are no turn's
    # spend, and averaged in they rank a refusing contestant cheap. A cell an apparatus fault cut short spent
    # less than a whole one, so averaged in it lets the rig make a contestant look cheaper. Both dollars stay
    # in program spend (`cost_usd`), which is accounting rather than a comparison.
    priced = [
        (r.test_case_id, c)
        for r, c in (
            (r, production_replicating_cost(r.usage, substituted_deliveries=count_substituted_deliveries(r)))
            for r in results
            if delivered_a_turn(r)
        )
        if c is not None
    ]
    observed_costs = [c for _, c in priced]
    n_cost = len(observed_costs)
    prod_cost = math.fsum(observed_costs) / n_cost if n_cost else None
    cost_is_partial = 0 < n_cost < len(results)
    # What that mean is made of. The frontier RANKS on cost, so a contestant whose run
    # priced its search calls against one whose run did not is a comparison of conventions
    # wearing the clothes of a comparison of configs — and nothing else on the point shows
    # it. Disclosed rather than corrected: which of the two is "right" depends on what the
    # operator is deciding, and the numbers are each honest about their own run.
    cost_compositions = pooled_cost_compositions(results)

    # latency — total wall-clock ms over the results that harvested a total, MINUS the cells a
    # harness failure produced. The same predicate `compute_pass_hat_k` and `compute_dimension_summary`
    # use, and for the same reason one step further on: this point is RANKED. Domination is
    # decided over pass^k x prod cost x total latency, and an infra-excluded cell carries a real
    # but truncated `LatencyMetrics` (unlike the timeout path, whose `_degraded_capture_fields`
    # leaves latency None), so pooling it here lets an apparatus fault push a contestant into or
    # out of the dominated set. Cost drops them too, above. A call the model refused or
    # errored on is out as well: it took no turn, and its round trip ranked an all-refusing contestant the fastest
    # on the subject, dominating the arms that answered. `delivered_a_turn` is that predicate — the
    # cells' own — and it keeps a turn the budget ended or the deadline struck, which really took
    # that long.
    timed = [
        (r.test_case_id, r.latency.total_ms)
        for r in results
        if r.latency is not None and r.latency.total_ms is not None and delivered_a_turn(r)
    ]
    totals = [total for _, total in timed]
    n_latency = len(totals)
    mean_total_ms = round(sum(totals) / n_latency, 3) if n_latency else None

    # cost / pass-probability — undefined when cost is unknown or nothing passed.
    # One attempt's mean cost over one attempt's pass probability, pass^1 — see the field.
    cost_per_acceptable_outcome = (prod_cost / pass_hat_1) if (prod_cost is not None and pass_hat_1) else None

    # Does this contestant's pool span cassette modes? Narrowed from the caller's whole
    # map to the runs these results actually came from, so a point can never be qualified
    # by a mode no observation in it recorded. Restricted to run ids the map knows, on the
    # rule every comparison here follows: a run whose mode was not supplied is not evidence
    # that it recorded a different one.
    contributing_runs = sorted({r.eval_run_id for r in results})
    contributing_modes = {
        run_id: cassette_modes_by_run[run_id] for run_id in contributing_runs if run_id in cassette_modes_by_run
    }

    # Does this contestant's pool span scenario suites? Narrowed the same way and on the
    # same rule: a run the map does not know is not evidence that it ran a different
    # template. A known run whose `template_id` is None IS evidence — that value is the
    # ad-hoc case, a real recorded state rather than an absence — so it stays in.
    contributing_templates = {
        run_id: templates_by_run[run_id] for run_id in contributing_runs if run_id in templates_by_run
    }
    # Grouped once and carried both ways: the sentence for a surface showing this point
    # alone, the entries for one showing it beside its siblings. Two derivations could
    # disagree; one cannot.
    template_span = _template_span_entries(contributing_templates)

    cases = _ContestantCases(
        pass_hat_k=_case_means((case, value) for case, values in pass_by_case.items() for value in values),
        production_replicating_cost=_case_means(priced),
        mean_total_ms=_case_means(timed),
    )
    point = FrontierPoint(
        cassette_mode_disclosure=cassette_mode_disclosure(contributing_modes),
        template_span=template_span,
        template_span_disclosure=_template_span_disclosure(template_span),
        # Narrowed to the contributing runs like the two above, and for the reason
        # they are: a caveat naming a run that put no observation on this row points
        # the reader at a denominator this point never used.
        completeness_disclosures={
            run_id: degraded_by_run[run_id] for run_id in contributing_runs if run_id in degraded_by_run
        },
        variant_key=variant_key,
        model=model,
        variant_identity_version=identity_version,
        identity_version_disclosure=_identity_version_disclosure(identity_version),
        pass_hat_k=pass_hat_k,
        k=k,
        n_pass_cases=pooled["n_cases_at_k"],
        pass_hat_k_curve=pooled["pass_hat_k_curve"],
        pass_hat_k_ci_low=None if pass_interval is None else pass_interval[0],
        pass_hat_k_ci_high=None if pass_interval is None else pass_interval[1],
        n_pass_no_criterion=pooled["n_no_criterion_excluded"],
        pass_hat_k_unmeasured_reason=pooled["pass_hat_k_unmeasured_reason"],
        mean_composite=mean_composite,
        composite_sem=composite_sem,
        composite_basis=pooled_composite_basis(results) if mean_composite is not None else None,
        n_composite_cases=len(values_by_case),
        production_replicating_cost=prod_cost,
        n_cost=n_cost,
        cost_is_partial=cost_is_partial,
        cost_compositions=cost_compositions,
        cost_per_acceptable_outcome=cost_per_acceptable_outcome,
        production_footing=(
            PooledProductionFooting(runs={run_id: footing_by_run.get(run_id) for run_id in contributing_runs})
            if footing_by_run is not None
            else None
        ),
        mean_total_ms=mean_total_ms,
        n_latency=n_latency,
        n_no_turn=sum(
            1 for r in results if classify_result(r) is ResultOutcome.CANDIDATE_FAIL and not delivered_a_turn(r)
        ),
        n_results=len(results),
        n_cases=len({r.test_case_id for r in results}),
        served_models=pooled_served_models(results),
    )
    return point, cases


class _ContestantCases(NamedTuple):
    """One contestant's per-test-case value on each domination axis: what a test between two pairs on.

    Keyed by ``test_case_id``, each the mean of the contestant's values at that case — its unit pass^k
    estimates (one per cell the case was measured under), or its observations' cost or total latency. An
    axis the contestant never measured is empty.
    """

    pass_hat_k: dict[str, Fraction]
    production_replicating_cost: dict[str, Fraction]
    mean_total_ms: dict[str, Fraction]


#: The domination axes, each a :class:`FrontierPoint` headline with its :class:`_ContestantCases` field of
#: the same name, and whether higher is better on it.
_DOMINATION_AXES: tuple[tuple[Literal["pass_hat_k", "production_replicating_cost", "mean_total_ms"], bool], ...] = (
    ("pass_hat_k", True),
    ("production_replicating_cost", False),
    ("mean_total_ms", False),
)


def _case_means(pairs: Iterable[tuple[str, float]]) -> dict[str, Fraction]:
    """The mean value at each case, from ``(case, value)`` pairs, exactly.

    Each value is read as the decimal it is written as (:func:`~threetears.evals.analysis.stats.exact_decimal`)
    and averaged over rationals — the arithmetic :func:`~threetears.evals.analysis.stats.level_difference`
    reads — so a constant per-case shift between two contestants stays a constant the separation test reads
    exactly, rather than acquiring a float residue its t-test would read as a tiny, perfectly consistent
    spread.
    """
    grouped: dict[str, list[Fraction]] = {}
    for case, value in pairs:
        grouped.setdefault(case, []).append(exact_decimal(value))
    return {case: sum(values, Fraction(0)) / len(values) for case, values in grouped.items()}


def _dominance_p(
    a: FrontierPoint, a_cases: _ContestantCases, b: FrontierPoint, b_cases: _ContestantCases
) -> float | None:
    """The p of the test that point ``a`` dominates point ``b``, or ``None`` where none can run.

    ``a`` dominates ``b`` when it is SHOWN better on every axis ``b`` measured — an intersection–union
    test (Berger 1982): each axis is tested on its own, and the claim stands only if every one rejects, so
    its p is the largest of theirs. Requiring every axis is what makes the conjunction a level-α test with
    no correction across axes; correcting across them would only make it stricter than α.

    Each axis is the engine's separation test (:func:`~threetears.evals.analysis.stats.separation_p`):
    paired over the test cases both contestants measured where they share at least two, Welch over each
    side's cases otherwise, two-sided, and counting only in ``a``'s favour — so on one axis a false call
    of "better" happens at most α/2 of the time.

    **"Better or equal" collapses to "better".** Showing a contestant no worse than another by at most a
    margin is an equivalence test, and these axes declare no margin; showing it no worse by zero is
    showing it better. So a tie on an axis — two contestants that each passed every case — blocks the
    claim rather than satisfying it: the data cannot say which is better there, and absence of evidence
    is not a claim.

    The missing-axis rule is the one point estimates were held to. If ``b`` measured an axis ``a`` did not,
    ``a`` cannot dominate: its unknown value there could be worse, and an unmeasured axis is never the
    winning "fastest" or "cheapest". The reverse does not block: an axis only ``a`` measured is skipped,
    which is what lets a working variant dominate one that failed everywhere (measured pass^k, no turn
    taken, so no cost or latency to rescue it).

    Args:
        a: The candidate dominator.
        a_cases: Its per-case values.
        b: The point tested for being dominated.
        b_cases: Its per-case values.

    Returns:
        The p: below α only when every axis ``b`` measured separates in ``a``'s favour, and 1.0 when an
        axis was tested and did not — a tie, a separation the other way, or no separation. ``None`` when
        no test could decide: ``b`` measured no axis, ``a`` lacks one ``b`` measured, or an axis has no test
        that can decide (:func:`_axis_p`).
    """
    measured = [(axis, higher) for axis, higher in _DOMINATION_AXES if getattr(b, axis) is not None]
    if not measured:
        return None
    largest = 0.0
    for axis, higher_is_better in measured:
        if getattr(a, axis) is None:
            return None
        p = _axis_p(getattr(a_cases, axis), getattr(b_cases, axis), higher_is_better=higher_is_better)
        if p is None:
            return None
        largest = max(largest, p)
    return largest


def _axis_p(
    a_values: Mapping[str, Fraction], b_values: Mapping[str, Fraction], *, higher_is_better: bool
) -> float | None:
    """The p of the test that ``a`` is better than ``b`` on one axis, counting only in ``a``'s favour.

    The engine's separation test (:func:`~threetears.evals.analysis.stats.separation_p`): paired over the
    test cases both sides measured where they share at least two, Welch over each side's cases otherwise,
    two-sided. Read in one direction: where the tested means do not favour ``a`` the p is 1.0, so a false
    call of "better" happens at most α/2 of the time. The one axis test both :func:`_dominance_p` and the
    verdict's cost comparison (:func:`_cost_ties`) read.

    Args:
        a_values: ``a``'s per-case values on the axis.
        b_values: ``b``'s.
        higher_is_better: Which way is better on the axis.

    Returns:
        The p, or ``None`` where no test can decide: fewer than two cases on a side, or no spread over too few
        cases for the exact test to reach α (:func:`~threetears.evals.analysis.stats.separation_p`).
    """
    if len(a_values) < 2 or len(b_values) < 2:
        return None
    shared = sorted(set(a_values) & set(b_values))
    paired = len(shared) >= 2
    a_side = [a_values[case] for case in shared] if paired else list(a_values.values())
    b_side = [b_values[case] for case in shared] if paired else list(b_values.values())
    p = separation_p(a_side, b_side, paired=paired)
    if p is None:
        return None
    gap = sum(a_side, Fraction(0)) / len(a_side) - sum(b_side, Fraction(0)) / len(b_side)
    a_better = gap > 0 if higher_is_better else gap < 0
    return p if a_better else 1.0


def _decide_dominance(points: list[FrontierPoint], cases: list[_ContestantCases]) -> None:
    """Fill every point's ``dominance``, ``dominated`` and ``dominated_by``, by test.

    Every pair of contestants in the subject is one comparison, with the smaller p of its two directions
    (:func:`_dominance_p`). Each direction's p is built from two-sided axis tests read in one direction
    only, so it errs at most α/2; the two directions are disjoint claims, so the smaller of the two is the
    pair's two-sided p, as the engine's every separation test is. The pairs are one family, Holm-adjusted together
    (:func:`~threetears.evals.analysis.stats.holm_adjust`), so the chance that ANY point in the subject is
    flagged dominated when it is not stays within α — the rule every family of between-arm claims in
    the engine is read by. A domination is shown when its pair's adjusted p is below α in that direction.

    Args:
        points: The subject's points, in display order. Mutated.
        cases: Each point's per-case values, aligned with ``points``.
    """
    family: list[tuple[int, int, float]] = []
    tested: set[int] = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            forward = _dominance_p(points[i], cases[i], points[j], cases[j])
            backward = _dominance_p(points[j], cases[j], points[i], cases[i])
            if forward is not None:
                tested.add(j)
            if backward is not None:
                tested.add(i)
            if forward is None and backward is None:
                continue
            if backward is None or (forward is not None and forward <= backward):
                family.append((i, j, forward if forward is not None else 1.0))
            else:
                family.append((j, i, backward))
    adjusted = holm_adjust([p for _, _, p in family])
    dominators: dict[int, list[tuple[int, float]]] = {}
    for (winner, loser, _), p_adjusted in zip(family, adjusted, strict=True):
        if p_adjusted < SIGNIFICANCE_ALPHA:
            dominators.setdefault(loser, []).append((winner, p_adjusted))
    for index, point in enumerate(points):
        # Built by walking the sorted points rather than by collecting a SET of names: a set over
        # `model` alone silently merges two variants of one model and leaves behind the dominated
        # row's own model name, which reads as a row dominating itself.
        found = sorted(dominators.get(index, []))
        point.dominated_by = [
            FrontierDominator(
                variant_key=points[winner].variant_key,
                model=points[winner].model,
                variant_identity_version=points[winner].variant_identity_version,
                p_value=p_adjusted,
            )
            for winner, p_adjusted in found
        ]
        point.dominated = bool(found)
        point.dominance = "dominated" if found else "not_separated" if index in tested else "untested"


def _cost_ties(
    pick: int, rivals: Sequence[int], points: Sequence[FrontierPoint], cases: Sequence[_ContestantCases]
) -> tuple[FrontierCostDecision, list[FrontierCostTie]]:
    """Whether the pick is shown cheaper than each rival, and the rivals it is not.

    One comparison per rival, each the frontier's own axis test on production-replicating cost
    (:func:`_axis_p`, counting only in the pick's favour), Holm-adjusted together
    (:func:`~threetears.evals.analysis.stats.holm_adjust`): dropping a rival from the set is a claim that it
    is dearer, and the claims are one family. The pick is the lowest point cost, so on identical
    contestants every comparison already leans its way; requiring each to separate is what keeps a
    "cheapest" named on noise near α.

    Args:
        pick: The index of the lowest point cost among the cleared, priced points.
        rivals: The other cleared, priced points' indices, ordered by point cost.
        points: The subject's points.
        cases: Each point's per-case values, aligned with ``points``.

    Returns:
        The decision and the rivals left in the set, each with its adjusted p (``None`` where untested).
    """
    if not rivals:
        return "only_cleared", []
    raw = [
        _axis_p(
            cases[pick].production_replicating_cost, cases[rival].production_replicating_cost, higher_is_better=False
        )
        for rival in rivals
    ]
    tested = [p for p in raw if p is not None]
    adjusted = iter(holm_adjust(tested))
    ties: list[FrontierCostTie] = []
    for rival, p in zip(rivals, raw, strict=True):
        p_adjusted = None if p is None else next(adjusted)
        if p_adjusted is not None and p_adjusted < SIGNIFICANCE_ALPHA:
            continue
        point = points[rival]
        assert point.production_replicating_cost is not None  # Only priced points are rivals.
        ties.append(
            FrontierCostTie(
                variant_key=point.variant_key,
                model=point.model,
                variant_identity_version=point.variant_identity_version,
                production_replicating_cost=point.production_replicating_cost,
                p_value=p_adjusted,
            )
        )
    if not ties:
        return "shown_cheapest", []
    if all(tie.p_value is None for tie in ties):
        return "untested", ties
    return "not_separated", ties


def _bar_decision(point: FrontierPoint, bar: float) -> BarDecision:
    """How a point's pass^k reads against the frontier's bar — the five words a campaign bar's verdict uses."""
    if point.pass_hat_k is None:
        return "no_data"
    if point.pass_hat_k_ci_low is None or point.pass_hat_k_ci_high is None:
        return "no_interval"
    cleared = interval_clears(
        (point.pass_hat_k_ci_low, point.pass_hat_k_ci_high), bar, margin=None, higher_is_better=True
    )
    return "undecided" if cleared is None else "cleared" if cleared else "missed"


def compute_frontier(
    runs: list[EvalRun],
    results: list[EvalResult],
    *,
    bar: float | None = None,
    subject_id: str | None = None,
    rubric_threshold: int = 3,
    known_run_ids: set[str] | None = None,
    archived_run_ids: set[str] | None,
    profile: HostProfile | None = None,
) -> FrontierResult:
    """Rank each subject's variants on quality x cost x latency and pick the cheapest above bar.

    One subject is one frontier: composite quality is derived from each
    subject's own rubric and is never comparable across subjects, so the subjects
    partition the answer at the top and are never pooled. Within a subject each
    contestant — one ``variant_key`` under one identity version, the resolved stack
    that would ship — becomes a point carrying pass^k (headline), mean-composite
    (secondary), production-replicating cost, total latency, and every denominator.
    The version is part of the contestant because one counter backs both key
    predicates and cannot say which moved, so pooling on the digest alone would merge
    on unverifiable provenance. The non-dominated set
    is computed and dominated points are flagged rather than dropped.

    The subjects partition; the CONDITIONS do not. ``variant_key`` names the shipping
    stack and nothing about how it was measured, so every scenario template and every
    cassette mode a variant was measured under pools into its single point. That is the
    recorded design (pool-and-disclose, not key-it), and both spans are disclosed on the
    point and carried onto the verdict: see ``FrontierPoint.template_span_disclosure`` and
    ``cassette_mode_disclosure``. A pooled pass^k over suites of unequal difficulty is
    still a real number about the variant — it is just not a score on any one suite, which
    is exactly what the disclosure says.

    **One depth per subject.** Every point's pass^k is read at :attr:`SubjectFrontier.k`, the
    shallowest ``k_runs`` among the subject's ranked runs, and estimated without bias from each
    case's attempts pooled across the runs of one cell
    (:func:`~threetears.evals.contracts.scoring.pool_pass_hat_k`): two runs of one variant under
    one ``context_key`` are more attempts at the same cases, while the same case measured under
    another context is a separate case beside it. So a repeat run sharpens a contestant's estimate
    rather than doubling its case count, and no contestant is ranked on a different depth than
    its rivals.

    When ``bar`` is supplied it gates pass^k, read by each point's pass^k interval the way every
    campaign bar is read (:attr:`FrontierPoint.bar_decision`): cleared only when the whole interval is at
    or above the bar, undecided when it straddles it. The cleared variant with the lowest known cost is
    the verdict's pick, and it is named the cheapest only when it is shown cheaper than every other cleared,
    priced variant by the dominance test's own cost comparison, Holm-adjusted over them; otherwise the
    verdict names the set the data cannot order (:attr:`FrontierVerdict.cost_decision`). An unsupplied bar
    yields the full frontier with no verdict, because inventing a quality threshold would editorialize.

    **Domination is decided by test, never read off point estimates** (:func:`_dominance_p`): a point is
    flagged dominated only when another is shown better on every axis it measured, the subject's pairs
    Holm-adjusted together, and otherwise reads ``not_separated`` or ``untested``. pass^k and the composite read
    capability dims only; boundary-pillar disqualification is descoped with disclosure — see
    :class:`TwoPillarDisclosure`.

    The two skips :func:`project_score_records` makes — a result whose run is
    absent, and one whose run captured no subject — are mirrored here and returned
    as :class:`ProjectionExclusions`, so an all-excluded corpus discloses that its
    data exists but cannot be placed rather than rendering as an empty scope.

    **A run that measured less than its matrix is admitted and disclosed, never
    refused** — the same pool-and-disclose position the cassette and template spans
    take, and for a stronger reason: its cells are real measurements of the same
    contestant, so dropping them would be the silent removal this surface already
    refuses for a dominated point. What it costs is the denominator, and that is
    what :attr:`FrontierPoint.completeness_disclosures` carries onto every affected
    row and onto the verdict. **The predicate is the completeness record, not the
    status:** a ``completed`` run with an infra-excluded cell is short too, and a
    surface that reasoned from ``status`` would pool it by default with nothing said.

    Args:
        runs: The runs supplying subject identity. Order is irrelevant.
        results: The observations to rank.
        bar: The pass^k threshold the verdict is made against, in ``[0, 1]``.
            ``None`` computes the frontier without a verdict.
        subject_id: When set, restrict to this subject; other subjects' results
            are counted as ``n_filtered_out`` rather than dropped silently.
        rubric_threshold: Forwarded to pass^k — a rubric score at or above it
            counts as a passing dimension — and recorded on the answer
            (:attr:`FrontierResult.rubric_threshold`). A campaign's bundle passes its
            behavior's declared threshold.
        known_run_ids: Every run id in the corpus, so a result excluded by the
            caller's own run filter (in ``known_run_ids`` but not ``runs``) is
            counted as filtered-on-request rather than unplaceable. See
            :func:`project_score_records`.
        archived_run_ids: The corpus run ids the operator has archived, so an
            archival exclusion is counted as ``results_from_archived_runs``
            rather than under the caller's ``status`` filter. See
            :func:`place_results`.
        profile: The host whose sweepable declarations each point's cost is read against: with it, every point
            and verdict names what each of its runs set away from the subject's production configuration
            (:attr:`FrontierPoint.production_footing`, #571). Read off the runs as given, so a run handed with its
            host payload elided reads as unchecked. ``None`` leaves that field ``None`` — nobody checked.

    Returns:
        A :class:`FrontierResult` — one :class:`SubjectFrontier` per subject,
        sorted by subject id, plus the corpus-level exclusion accounting and the
        completeness disclosures of every short run it ranked over.

    Raises:
        FrontierError: ``bar`` is outside ``[0, 1]`` — refused here so both
            surfaces refuse identically rather than one clamping it.
    """
    if bar is not None and not (0.0 <= bar <= 1.0):
        raise FrontierError(f"bar {bar!r} is outside the pass^k range [0, 1] — pass^k is a probability")

    placed, exclusions = place_results(
        runs, results, known_run_ids, source="frontier", archived_run_ids=archived_run_ids
    )

    by_subject: dict[str, dict[ContestantKey, list[EvalResult]]] = {}
    subject_labels: dict[str, str] = {}
    # Every placed run's recorded cassette mode, collected here because a point is
    # built from RESULTS and the mode lives on the RUN — a result carries no cassette
    # field of its own to stand in for it.
    cassette_modes_by_run: dict[str, str] = {}
    # Every placed run's template, collected for the same reason and on the same seam: a
    # point is built from RESULTS and the template lives on the RUN. `EvalResult` carries no
    # template of its own, and its `test_case_id` is not a substitute: resolving a case id to
    # the template that owns it needs the case documents, which this function is never given
    # and which an ad-hoc run (`template_id is None`) would not answer anyway.
    templates_by_run: dict[str, str | None] = {}
    # Every candidate run that came up short of its matrix, collected on the same seam
    # and for the same reason as the two maps above: a point is built from RESULTS and
    # completeness lives on the RUN. Keyed on the run rather than derived from its
    # STATUS, because a `completed` run can be degraded too — one cell infra-excluded
    # is a shorter denominator that no status filter can see.
    degraded_by_run: dict[str, str] = {}
    n_filtered_out = 0
    n_considered = 0
    n_degraded_observations = 0
    # The observations this answer actually rests on. Collected here rather than derived
    # from the assembled points for two reasons: an answer whose points were all dropped
    # downstream still spanned what it spanned, and narrowing to what SURVIVED the subject
    # filter keeps the span from naming a predicate no row in this answer was keyed under
    # — the same rule the cassette-mode and template maps below follow.
    considered: list[EvalResult] = []
    # Which configuration each run measured, so a contestant's repeat runs pool their attempts at
    # a case (more depth) and runs under different conditions never do. Collected on the same seam
    # as the maps above: a point is built from RESULTS and the measurement context lives on the RUN.
    cell_of_run: dict[str, Hashable] = {}
    # The depth each subject is ranked at: the shallowest k any of its ranked runs was
    # commissioned to (see SubjectFrontier.k).
    k_by_subject: dict[str, int] = {}
    # Each placed run and its results, for the production footing a point's cost names (#571).
    placed_runs: dict[str, EvalRun] = {}
    results_of_run: dict[str, list[EvalResult]] = {}

    for run, result, resolved in placed:
        if subject_id is not None and resolved != subject_id:
            n_filtered_out += 1
            continue

        n_considered += 1
        considered.append(result)
        by_subject.setdefault(resolved, {}).setdefault(_contestant_key(result), []).append(result)
        subject_labels[resolved] = run.subject_snapshot.subject_label
        cassette_modes_by_run[run.id] = run.cassette_mode
        templates_by_run[run.id] = run.template_id
        cell_of_run[run.id] = pass_hat_k_cell(run)
        placed_runs[run.id] = run
        results_of_run.setdefault(run.id, []).append(result)
        k_by_subject[resolved] = min(k_by_subject.get(resolved, run.k_runs), run.k_runs)
        if (short := completeness_disclosure(run.completeness)) is not None:
            degraded_by_run[run.id] = short
            n_degraded_observations += 1

    footing_by_run: dict[str, ProductionFooting | None] | None = (
        {
            run_id: None
            if run.elided_payload_paths
            else profile.sweepables.production_footing(run, results_of_run[run_id])
            for run_id, run in placed_runs.items()
        }
        if profile is not None
        else None
    )

    subjects: list[SubjectFrontier] = []
    for resolved in sorted(by_subject):
        groups = by_subject[resolved]
        subject_k = k_by_subject[resolved]
        built = [
            _frontier_point(
                groups[key],
                contestant=key,
                k=subject_k,
                cell_of_run=cell_of_run,
                rubric_threshold=rubric_threshold,
                cassette_modes_by_run=cassette_modes_by_run,
                templates_by_run=templates_by_run,
                degraded_by_run=degraded_by_run,
                footing_by_run=footing_by_run,
            )
            for key in sorted(groups, key=lambda k: (str(k), ""))
        ]
        # The version is in the sort key because the partition made a TIE on the first two
        # reachable: two points can share a model and a key and differ only in predicate.
        # Sort stability alone would have made that order deterministic but arbitrary —
        # and the two rows sit adjacent, which is where an unexplained order reads as noise.
        built.sort(key=lambda pair: (pair[0].model, pair[0].variant_key, pair[0].variant_identity_version))
        points = [point for point, _ in built]
        point_cases = [cases for _, cases in built]
        _decide_dominance(points, point_cases)

        verdict: FrontierVerdict | None = None
        n_cleared_bar = 0
        n_undecided_bar = 0
        if bar is not None:
            # The bar is read by the point's pass^k INTERVAL, the rule every campaign bar is read by
            # (`stats.interval_clears`): cleared only when the whole interval is at or above it, and a
            # straddle is undecided — neither a pass nor a failure, and never the cheapest pick.
            for point in points:
                point.bar_decision = _bar_decision(point, bar)
            cleared = [p for p in points if p.bar_decision == "cleared"]
            n_cleared_bar = len(cleared)
            n_undecided_bar = sum(1 for p in points if p.bar_decision == "undecided")
            # The cleared, priced points by point cost. The lowest is the pick, and it is named THE cheapest
            # only when it is shown cheaper than each of the rest (`_cost_ties`); otherwise the verdict names
            # it beside every rival it could not be shown cheaper than.
            costed = sorted(
                (
                    index
                    for index, p in enumerate(points)
                    if p.bar_decision == "cleared"
                    and p.production_replicating_cost is not None
                    and p.pass_hat_k is not None
                ),
                key=lambda i: (points[i].production_replicating_cost, -(points[i].pass_hat_k or 0.0), points[i].model),
            )
            if costed:
                pick = points[costed[0]]
                pick_cost, pick_pass_hat_k = pick.production_replicating_cost, pick.pass_hat_k
                assert pick_pass_hat_k is not None  # Only points with a pass^k are costed.
                cost_decision, tied_with = _cost_ties(costed[0], costed[1:], points, point_cases)
                verdict = FrontierVerdict(
                    variant_key=pick.variant_key,
                    model=pick.model,
                    pass_hat_k=pick_pass_hat_k,
                    k=subject_k,
                    pass_hat_k_ci_low=pick.pass_hat_k_ci_low,
                    pass_hat_k_ci_high=pick.pass_hat_k_ci_high,
                    production_replicating_cost=pick_cost,
                    cost_is_partial=pick.cost_is_partial,
                    production_footing=pick.production_footing,
                    # Carried from the picked point rather than re-derived: one predicate,
                    # so the verdict and the row it names cannot disagree about whether
                    # that contestant's observations spanned cassette modes.
                    cassette_mode_disclosure=pick.cassette_mode_disclosure,
                    # Carried from the picked point for the same one-predicate reason: the
                    # verdict and the row it names cannot disagree about which suites the
                    # pass^k it was picked on was averaged over.
                    template_span_disclosure=pick.template_span_disclosure,
                    # And on the same rule again: the bar was applied to this pass^k and the
                    # pick made on this cost, so a recommendation resting partly on runs that
                    # never finished their matrix has to say so where the recommendation is.
                    completeness_disclosures=pick.completeness_disclosures,
                    # Carried, not re-derived, for the one-predicate reason the four above
                    # are: the verdict and the row it names cannot disagree about which
                    # predicate minted the key this recommendation is addressed by.
                    identity_version_disclosure=pick.identity_version_disclosure,
                    served_models=pick.served_models,
                    variant_identity_version=pick.variant_identity_version,
                    cost_decision=cost_decision,
                    tied_with=tied_with,
                )

        subjects.append(
            SubjectFrontier(
                subject_id=resolved,
                subject_label=subject_labels.get(resolved, ""),
                k=subject_k,
                points=points,
                verdict=verdict,
                n_cleared_bar=n_cleared_bar,
                n_undecided_bar=n_undecided_bar,
            )
        )

    identity_versions = _identity_version_span(considered)

    return FrontierResult(
        bar=bar,
        rubric_threshold=rubric_threshold,
        subjects=subjects,
        n_results=n_considered,
        n_filtered_out=n_filtered_out,
        exclusions=exclusions,
        completeness_disclosures=degraded_by_run,
        n_degraded_observations=n_degraded_observations,
        identity_version_span=identity_versions,
        identity_span_disclosure=_identity_span_disclosure(identity_versions),
    )


# =============================================================================
# history — per-measure longitudinal series with honest regression flags
# =============================================================================

# The measures `history` can series. Composite, cost and the two dual-score axes are also
# `PROJECTED_METRICS` members; `total_ms` is not projected at all and is read straight off
# `EvalResult.latency`. Every member is numeric with a defined better direction, which
# is what the regression flag needs to call a move a decline or a gain — a directionless
# coordinate could not be "regressed". A metric outside this set is refused, not answered
# with an empty series, for the same reason `pivot` refuses an unknown measure.
#
# The dual-score axes are here because the read the catalog says the PAIR exists for is a
# read over time: `__outcome__` falling while `__transcript__` holds is a world that
# changed, and both falling together is a candidate that got worse. `pivot` answers that
# within one campaign and this is the surface that answers it across runs. The members of
# `PROJECTED_METRICS` that stay refused are the SCOPED ones (`SCOPED_METRICS`): each row is one
# dimension or one check, so there is no single per-run value to plot, while each axis above is
# one value per result.
#: The measures `history` can series; any other is refused rather than answered with an empty series.
HISTORY_METRICS = frozenset({METRIC_COMPOSITE, METRIC_COST_USD, METRIC_TOTAL_MS, METRIC_TRANSCRIPT, METRIC_OUTCOME})

if set(_METRIC_GLOSS) != PROJECTED_METRICS | HISTORY_METRICS:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        f"_METRIC_GLOSS must gloss exactly the accepted metrics: {sorted(set(_METRIC_GLOSS) ^ (PROJECTED_METRICS | HISTORY_METRICS))}"
    )


class HistoryError(ValueError):
    """A history was requested for a measure it cannot honestly series.

    Raised rather than returned as an empty series, for the reason
    :class:`PivotError` is: an empty series is indistinguishable from a measure
    nobody has recorded, so a refusal returned as data becomes a silent nothing.
    """


def _unknown_history_metric(metric: str) -> str:
    """Say why a measure cannot be seriesed, and where the caller should go instead.

    The single owner of that sentence — it is raised from two places, and two copies
    of one refusal drift.

    ``score`` earns its own arm rather than being enumerated past. It is not a typo:
    it is a registered measure that ``results_pivot`` accepts, ``export_results``
    emits and the catalog describes, so a caller reaching for it here asked a
    coherent question this surface cannot answer. A series carries ONE value per
    contestant per run and a raw judge score is per rubric DIMENSION, so there is no
    single per-run score to plot — and a message that only listed the three
    alternatives left the caller to infer that, which reads as "not implemented yet".

    Args:
        metric: The measure the caller asked to series.

    Returns:
        The refusal text, ready to raise as a :class:`HistoryError`.
    """
    expected = f"expected one of {_metric_vocabulary(HISTORY_METRICS)}"
    resolved = resolve_measure_name(metric)
    if resolved in SCOPED_METRICS and resolved != METRIC_SCORE:
        scope_field, scope_noun, _row_noun = SCOPED_METRICS[resolved]
        return (
            f"history cannot series {resolved!r} — each row is ONE {scope_noun}, so there is no single per-run value "
            f"to plot. Use the pivot surface with '{scope_field}' on an axis, which aggregates it as "
            f"{_AGGREGATE_OF_OBSERVATION.get(resolved, resolved)!r}. {expected}."
        )
    # Resolved for the BRANCH, while `metric` stays the caller's spelling for the
    # message: `mean_score` and `score` are the same refusal and must not take
    # different arms just because the operator read the catalog.
    if resolve_measure_name(metric) == METRIC_SCORE:
        return (
            f"history cannot series {METRIC_SCORE!r} — it is the raw judge score for one rubric "
            "DIMENSION, and a series carries one value per contestant per run, so there is no single "
            "per-run score to plot. For a per-dimension read use the pivot surface with 'rubric_dim' "
            "on an axis, which aggregates it as 'mean_score'; to track judged quality over time, "
            f"series 'composite', or {METRIC_TRANSCRIPT!r} / {METRIC_OUTCOME!r} to stay on the raw 1-5 "
            "scale — each of those is one value per result, which is what this surface can plot. "
            f"For history, {expected}."
        )
    return f"unknown history metric {metric!r} — {expected}"


def _attribution_withheld(descriptor: MetricDescriptor) -> bool:
    """Say whether a labelled move in this measure may be read as the contestant's.

    Read off the measure's ``transferability_class`` rather than its name, so the rule
    is a property of the catalog and not a list of metrics kept in step by hand. A
    ``scenario_bound`` measure is defined by the scenario, and a series cannot hold the
    externals that scenario touches still between two runs — the registry says so of the
    live case in its own words, that it is comparable across runs "only where the
    scenarios and the externals they touch held still". So the move is real and its
    cause is not decided by observing it.

    The looser classes are not asserted to be the contestant's either; they are simply
    outside what this predicate refuses. It says only that a scenario-bound label may
    not be attributed.

    Args:
        descriptor: The measure the series reports — the AGGREGATE descriptor, since
            that is what a point holds.

    Returns:
        Whether a verdict on this measure must decline to attribute the move.
    """
    return descriptor.transferability_class == "scenario_bound"


def _attribution_disclosure(descriptor: MetricDescriptor) -> str | None:
    """The one sentence saying why this measure's verdicts do not attribute.

    Carried on the answer rather than composed at each surface, so REST, MCP and any
    later renderer say the same thing — two copies of one caveat drift, which is why
    :func:`_unknown_history_metric` is shared for its refusal.

    Args:
        descriptor: The aggregate descriptor the series reports.

    Returns:
        The sentence, or ``None`` when the measure's verdicts carry no such limit.
    """
    if not _attribution_withheld(descriptor):
        return None
    return (
        f"{descriptor.name} is scenario-bound: its value is defined by the scenario, and a series "
        "cannot hold the externals that scenario touches still between two runs. Every regression "
        "flag here states the move and withholds attribution — the label is not a finding about "
        "the contestant."
    )


class RegressionFlag(EvalBaseModel):
    """A descriptive verdict on the change from the previous point in the series.

    Descriptive, never an alert: it discloses the ``test`` it ran and the
    thresholds it applied, so the label can never be read as a calibrated
    judgement — automated alerting stays gated behind judge calibration. ``label``
    is one of ``regressed`` / ``improved`` / ``equivalent`` / ``below_threshold`` /
    ``not_separated`` / ``untested`` (:class:`~threetears.evals.analysis.stats.ChangeVerdict`
    defines each). A move earns a directional label only when it is both
    statistically significant and over a magnitude threshold (a joint gate). A move
    that misses significance reads ``not_separated``, never "no change": the one label
    claiming no meaningful change is ``equivalent``, and it needs an equivalence test
    against the measure's declared margin (``equivalence_margin``) to pass. With no
    margin declared, no step can read ``equivalent``.

    ``crosses_epoch`` warns that the two runs span a suite-version boundary: the
    paired test then rests only on the cases the two still share, and the
    aggregate delta is confounded by the changed denominator, so a flag here says
    less than one within an epoch.

    ``crosses_cassette_mode`` is that warning's sibling on the apparatus axis, and the
    stronger of the two. The two runs did not record the same cassette mode, so where the
    step involves ``replay`` one of them re-served a recording instead of calling the
    third party — the measure did not move between two measurements of one thing, it
    moved between a measurement and a substitution. Cassette mode is deliberately outside
    the variant key (it is apparatus, not product), so both runs sit in ONE series and
    nothing else in the flag can show it. Descriptive, exactly as ``crosses_epoch`` is: it
    suppresses no verdict and adjusts no delta.

    ``attribution_withheld`` is the third of that family and differs from the other two in
    what it is about: they qualify this PAIR of runs, while it is a property of the measure
    and so holds for every step in the series. It fires where the measure is scenario-bound
    (:func:`_attribution_withheld`), and it neither suppresses the label nor adjusts the
    delta — a decline nobody can attribute is still worth seeing, and staying silent about
    it would be the reading this flag exists to prevent.
    """

    label: ChangeLabel
    delta: float | None = None
    relative_delta: float | None = None
    significant: bool | None = None
    exceeds_threshold: bool | None = None
    #: Hedges' g_z of the paired move — bias-corrected, so not comparable with a Cohen's d.
    hedges_g: float | None = None
    #: The p ``significant`` was thresholded against — the paired t's, or the exact
    #: sign-flip p where every case moved by one amount — and ``None`` on an
    #: ``untested`` step. Carried for the same reason ``hedges_g`` is: a verdict
    #: whose statistic is absent cannot be checked.
    p: float | None = None
    #: The TOST p an ``equivalent`` label was thresholded against — the larger of the
    #: two one-sided p's — or ``None`` wherever no equivalence test ran: no margin
    #: declared, too few pairs, or on a measure with no declared range a difference
    #: with no spread (:func:`~threetears.evals.analysis.stats.paired_equivalence`).
    equivalence_p: float | None = None
    #: The margin that test ran against, in the measure's units: the measure's declared
    #: materiality threshold, or ``None`` when it declares none.
    equivalence_margin: float | None = None
    n_pairs: int = 0
    crosses_epoch: bool = False
    crosses_cassette_mode: bool = False
    #: This measure's move cannot be attributed to the contestant. Set on the FLAG, not
    #: only on the answer, because the label is what a reader would otherwise take as the
    #: attribution — and a flag is routinely read as one row rather than beside the header.
    #: The sentence is :attr:`HistoryResult.attribution_disclosure`, stated once where the
    #: measure is described rather than repeated on every step.
    attribution_withheld: bool = False
    # Disclosure — the test and thresholds this verdict rests on, carried on every
    # flag so a reader never has to infer what "regressed" was measured against.
    test: str
    min_absolute_change: float
    min_relative_change: float


class SeriesPoint(EvalBaseModel):
    """One run's aggregate for the measure — the trend's unit, with its denominators.

    ``value`` is the equal-per-scenario mean over the run's cases; ``sem``, ``n``
    (observations that carried a value) and ``n_cases`` (distinct cases) travel
    with it because a point resting on 2 cases and one resting on 30 must not read
    alike. ``epoch`` is a 1-based ordinal that increments at each
    suite-version boundary, and ``epoch_boundary`` marks the point where the suite
    changed — so a step caused by the denominator changing does not read as a
    regression. ``delta_from_baseline`` is the descriptive red/green move from the
    series' first point; ``regression`` is the statistical verdict versus the
    immediately preceding point.
    """

    run_id: str
    created_at: str
    model: str
    value: float | None = None
    sem: float | None = None
    n: int = 0
    n_cases: int = 0
    epoch: int = 1
    epoch_boundary: bool = False
    is_baseline: bool = False
    delta_from_baseline: float | None = None
    regression: RegressionFlag | None = None
    #: On a COST series only: the role sets this point's dollars were summed over. It is
    #: here rather than on the series because a change lands between two points, and this
    #: is what says which two — the same job ``epoch_boundary`` does for the suite. A cost
    #: step caused by the composition moving is not a regression, and nothing else about
    #: the point can tell the two apart. Empty on every non-cost measure, which have no
    #: composition.
    cost_compositions: list[list[str]] = []
    #: On a COMPOSITE series only: what this point's composites were meaned over (#638), ``ragged`` when the
    #: run's results carried different dimension sets. Per point for the reason ``cost_compositions`` is: a
    #: step between two points meaned over different sets is a change in what was averaged, not a regression.
    #: ``None`` on every other measure and on a point with no composite.
    composite_basis: CompositeBasis | None = None
    #: This point's run's RECORDED cassette mode. Carried per point rather than once per
    #: series because it can move BETWEEN points — which is the whole defect: cassette
    #: mode is outside the variant key, so a capture run and a replay run of one
    #: contestant are two points of one series and the step between them reads as a
    #: quality change. A recorded mode is a claim, not an observation; see
    #: :func:`cassette_mode_disclosure`.
    cassette_mode: str | None = None
    #: This point's run's DEGRADED sentence, or ``None`` when it delivered the whole
    #: matrix it promised. A point IS one run, so unlike every other pooling surface
    #: this one can attribute the shortfall exactly — and it needs to most: the point
    #: sits on a time series beside complete runs and is handed a regression verdict
    #: against its neighbour, so a run that measured 2 of its 4 cells contributes a
    #: ``not_separated`` or ``regressed`` label computed over a denominator the comparison does
    #: not share. Derived from the run's completeness record rather than its status: a
    #: ``completed`` run with an infra-excluded cell is short too.
    completeness_disclosure: str | None = None
    #: Which models the provider's responses named as having answered this run's candidate calls (#684). Per point
    #: because a floating alias can resolve to a different model between two runs of one contestant, and the step
    #: between them then reads as a change in the contestant. ``None`` when no result's candidate made a call.
    served_models: ServedModelReading | None = None


class MeasureSeries(EvalBaseModel):
    """One contestant's measure over time, within a subject.

    A series is one ``(subject, variant, predicate version)``: the same resolved
    contestant tracked across its runs, so a regression is a real re-run decline rather
    than an artefact of pooling two configs. ``model`` is the human label;
    ``variant_key`` is the identity.

    An identity-version bump **splits** a series rather than marking an epoch inside one,
    which is the opposite of how a suite change is handled (:func:`_suite_epoch_key`) and is
    deliberate. A suite epoch keeps one contestant whole across a change to what it was
    scored against; an identity bump may have moved what the contestant *is*, and the stamp
    cannot say whether it did — so holding both versions in one series would assert a
    continuity nothing can verify, in the one place a reader reads continuity.
    """

    subject_id: str
    subject_label: str = ""
    variant_key: str
    #: The predicate version that minted ``variant_key``. Carried onto the row because
    #: this surface GATES on it — two keys carrying different stamps are ranked separately,
    #: because the stamp cannot say which predicate moved and a wrong merge is the direction
    #: nothing downstream undoes — and a reader who meets that split needs the number that
    #: caused it.
    variant_identity_version: int
    #: Set when that predicate is not this build's.
    identity_version_disclosure: str | None = None
    model: str
    points: list[SeriesPoint] = []
    #: Which models answered the candidate calls across the whole series (#684): ``pooled`` where one requested
    #: model was answered by several over its runs, so the series tracks a mixture rather than one model — each
    #: point says which answered it. ``None`` when no result's candidate made a call.
    served_models: ServedModelReading | None = None


class HistoryResult(EvalBaseModel):
    """Per-measure longitudinal series across contestants, with its disclosures.

    ``formula`` and ``weighting`` state what each point averages; ``higher_is_better``
    carries the measure's direction so a reader knows which way is a regression.
    ``min_absolute_change`` / ``min_relative_change`` echo the caller's regression
    gate (the thresholds are the caller's, disclosed, never invented), and the
    per-flag ``test`` names the statistic. ``equivalence_margin`` is the host's
    declared margin on the measure, the only thing that lets a step read
    ``equivalent``; ``None`` when it declares none. ``attribution_disclosure`` is the same
    obligation one rung up: on a measure whose verdicts cannot name a cause, it says so
    once for the answer. ``exclusions`` and the ``n_*`` counts keep
    an all-excluded corpus from rendering as an empty one, exactly as
    :class:`PivotTable` and :class:`FrontierResult` do.
    """

    metric: str
    measure: MetricDescriptor
    formula: str
    weighting: str
    higher_is_better: bool | None = None
    min_absolute_change: float
    min_relative_change: float
    equivalence_margin: float | None = None
    series: list[MeasureSeries] = []
    n_results: int = 0
    n_filtered_out: int = 0
    exclusions: ProjectionExclusions = ProjectionExclusions()
    #: ``run_id -> DEGRADED sentence`` for every run on these series that came up short
    #: of its matrix. The per-point copies mark which rows; this is the set, so a surface
    #: can state the rule once rather than per point.
    completeness_disclosures: dict[str, str] = {}
    #: How many of ``n_results`` came from those runs — the weight of the caveat.
    n_degraded_observations: int = 0
    #: The distinct identity versions this answer's keyed observations were stamped at,
    #: ascending. Partitioning stops two stampings of one stack
    #: being RANKED together; it cannot stop them appearing as two rows, so the span and
    #: the sentence below are what keep a doubled row from reading as a mystery.
    identity_version_span: list[int] = []
    #: The one-sentence form of that span, or ``None`` when the answer rests on a single
    #: predicate. Computed from the placed rows rather than from the assembled
    #: series, so an answer whose series were all filtered out still says what
    #: it spanned.
    identity_span_disclosure: str | None = None
    #: Why this measure's regression flags decline to attribute their move, or ``None``
    #: where they carry no such limit. A property of the MEASURE, so it is stated once
    #: here beside ``measure`` and ``formula`` rather than repeated on every flag; each
    #: flag carries the boolean (:attr:`RegressionFlag.attribution_withheld`) so a verdict
    #: read alone still declares its posture. Derived before any series is assembled, so
    #: an answer with no series at all still carries it.
    attribution_disclosure: str | None = None


def _history_value_of(metric: str) -> Callable[[EvalResult], float | None]:
    """Pick the per-result value extractor for a history measure.

    Args:
        metric: A measure in :data:`HISTORY_METRICS`.

    Returns:
        A callable reading the measure off one result, ``None`` when this result
        did not carry it — a null that the aggregation drops rather than counts as
        zero, so an unmeasured latency never reads as instant.

    Raises:
        HistoryError: The metric is not one this surface can series. Unreachable
            when :func:`compute_history` validated first; kept as the honest total answer.
    """
    if metric == METRIC_COMPOSITE:
        return result_composite
    if metric == METRIC_COST_USD:
        # Measuring spend, so every dollar the program spent: the population program spend keeps on every
        # surface that reads it — the cost pivot, a run summary's `mean_cost_usd`
        # (:func:`~threetears.evals.contracts.scoring.compute_cost_summary`) and the budget view. A call the
        # model refused before any turn was still billed, and a cell the harness faulted spent what it spent.
        # Leaving the refusal out while the pivot kept it gave one corpus two figures for one quantity. What an
        # arm COSTS reads only the turns taken, and is `production_replicating_cost`, which no series offers.
        return lambda result: result.cost_usd
    if metric == METRIC_TOTAL_MS:
        # Infra-excluded cells are withheld here for the reason they are on the frontier's
        # latency: an apparatus fault produces a REAL but truncated `LatencyMetrics`, and this
        # series issues regression verdicts. Composite already drops them (via `result_composite`
        # returning None), so leaving latency in made the two metrics on one surface answer
        # different questions — and a cassette miss could post a "faster" step that describes the
        # harness. A call the model refused or errored on took no turn, and is withheld for the frontier's
        # reason: `delivered_a_turn`, the one predicate every latency reading uses. Measuring spend keeps
        # both, because those dollars were spent (see METRIC_COST_USD above).
        return lambda result: (
            result.latency.total_ms
            if result.latency is not None and result.latency.total_ms is not None and delivered_a_turn(result)
            else None
        )
    if metric in (METRIC_TRANSCRIPT, METRIC_OUTCOME):

        def dual_axis_value(result: EvalResult) -> float | None:
            """Read one dual-score axis off a result, or ``None`` where it is not a measurement.

            Args:
                result: The observation to read.

            Returns:
                The raw 1-5 score, or ``None`` when the axis was not judged, was
                mis-stamped, or the cell was an apparatus failure.
            """
            # Named attributes rather than a computed one: this pair is read the same way
            # in `project_score_records`, and a `getattr` by string puts a rename of either
            # field out of the type checker's reach.
            axis = result.transcript_score if metric == METRIC_TRANSCRIPT else result.outcome_score
            if axis is None:
                return None
            # The guard the projection applies to these two fields, for its reason and one
            # tighter. `RubricScore.dim` is stored free text, so a mis-stamped row would
            # otherwise be meaned into a series and handed a regression verdict while `pivot`
            # and `export_results` drop it and warn — the two surfaces disagreeing about what
            # an axis is. Tighter because a series asks for ONE axis: the projection needs
            # only "is this a reserved id" to route the row to a metric, while here a
            # transcript score sitting in `outcome_score` would be a reserved id landing in
            # the wrong series. Loud there and loud here, because the absence it leaves
            # already means something else: an unjudged result carries no axis either, so a
            # silent skip makes a defect read as an unjudged run.
            if axis.dim != metric:
                log.warning(
                    "eval.reporting dropping a dual-score observation for result=%s from the %s series: dim %r is not "
                    "that axis. The axis was scored and is NOT in this series — its absence is a defect in what "
                    "stamped it, not a result that went unjudged.",
                    result.id,
                    metric,
                    axis.dim,
                )
                return None
            # The rule every judged score follows (`counted_score`), so one axis means one thing
            # on every surface: an infra-excluded cell is withheld — a judge reading a broken
            # transcript is not measuring the candidate, and this surface issues a regression
            # verdict on what it returns — and a candidate failure counts at the scale's floor.
            counted_axis = counted_score(result, axis)
            return None if counted_axis is None else float(counted_axis)

        return dual_axis_value
    raise HistoryError(_unknown_history_metric(metric))


def _per_case_means(
    results: list[EvalResult], value_of: Callable[[EvalResult], float | None]
) -> tuple[dict[str, float], int]:
    """Collapse one run's observations to one value per case (k averaged first).

    Averaging k iterations before the case enters the series is what makes each
    case weigh equally under equal-per-scenario, matching how pass^k and the
    composite summary count. Observations with no value for the measure are
    dropped, so a case counts only where it was actually measured.

    Args:
        results: One contestant's results within one run.
        value_of: The measure extractor from :func:`_history_value_of`.

    Returns:
        ``(per_case_mean, n_observations)`` — the per-case means keyed by
        ``test_case_id``, and how many observations carried a value.
    """
    by_case: dict[str, list[float]] = {}
    for result in results:
        value = value_of(result)
        if value is not None:
            by_case.setdefault(result.test_case_id, []).append(float(value))
    n_observations = sum(len(values) for values in by_case.values())
    per_case = {case_id: sum(values) / len(values) for case_id, values in by_case.items()}
    return per_case, n_observations


def _suite_epoch_key(run: EvalRun) -> tuple[str, tuple[str, ...]]:
    """The suite-version identity of a run: its template and frozen case set.

    Two runs are the same suite version iff they score the same template against
    the same test cases — exactly what ``context_components.case_basis`` digests,
    derived here straight from the run's own fields so it is defined for every
    run, stamped or not. A change in this key between two
    adjacent runs is a suite-version boundary: an aggregate step across it is the
    denominator changing, not necessarily the contestant regressing.

    Args:
        run: The run whose suite identity to key.

    Returns:
        ``(template_id, sorted test_case_ids)`` — the empty string stands in for an
        ad-hoc run's absent template so ad-hoc runs still key stably.
    """
    return (run.template_id or "", tuple(sorted(run.test_case_ids)))


def compute_history(
    runs: list[EvalRun],
    results: list[EvalResult],
    *,
    metric: str = METRIC_COMPOSITE,
    min_absolute_change: float = 0.0,
    min_relative_change: float = 0.0,
    subject_id: str | None = None,
    known_run_ids: set[str] | None = None,
    archived_run_ids: set[str] | None,
    profile: HostProfile,
) -> HistoryResult:
    """Series one measure over time per contestant, flagging real regressions.

    Each ``(subject, variant, identity version)`` contestant becomes a series; each of its runs
    becomes a time-ordered point carrying the measure's equal-per-scenario mean,
    its dispersion, and its denominators. The series is split into suite-version
    epochs (:func:`_suite_epoch_key`) so a step caused by the test set changing
    reads as an epoch boundary rather than a mysterious jump. Between adjacent
    points the change is classified by a paired test on the cases they share plus
    ``min_*_change`` magnitude thresholds (:func:`~threetears.evals.analysis.stats.paired_change`),
    and, where the measure declares a materiality threshold, an equivalence test
    against it; the tests and thresholds ride on every flag — descriptive, never an
    alert, until judge calibration lands.

    **On a scenario-bound measure the flag fires and withholds attribution.** Its value
    is defined by the scenario, whose externals no series can hold still between runs, so
    the move is real and its cause is not decided by observing it — see
    :func:`_attribution_withheld`. Staying silent instead would hide a decline; labelling
    it unqualified would name a culprit the measurement does not identify.

    Subjects are never pooled: composite quality is derived from each
    subject's own rubric, so two subjects are two sets of series. The two skips
    :func:`place_results` makes are returned as :class:`ProjectionExclusions`, so
    an all-excluded corpus discloses that its data exists but cannot be grouped
    rather than rendering as an empty one.

    **A run that measured less than its matrix is kept on the series and marked**,
    for the reason :func:`compute_frontier` gives for admitting one: its cells are real
    measurements of the same contestant, and dropping a point would put a gap in a
    time series where a run demonstrably happened. What it costs is disclosed on the
    point itself (:attr:`SeriesPoint.completeness_disclosure`), which is the closest
    any pooling surface can get to attributing it — a point IS one run. It matters
    here because the point is not merely averaged but *compared*: it receives a
    regression verdict against its neighbour, computed over a denominator the two do
    not share. The predicate is the completeness record, never the status.

    Args:
        runs: The runs supplying subject identity, suite set, and timestamps.
        results: The observations to series.
        metric: A measure in :data:`HISTORY_METRICS`, or the registry name of its
            aggregate as ``list_metrics`` publishes it (``mean_composite`` for
            ``composite``, and so on through
            :data:`_AGGREGATE_OF_OBSERVATION`) — see :func:`resolve_measure_name`.
            Defaults to composite quality.
        min_absolute_change: Smallest absolute move that counts as a regression,
            in the measure's own unit. ``0.0`` lets significance alone flag.
        min_relative_change: Smallest move relative to the baseline that counts, as
            a fraction. ``0.0`` lets significance alone flag.
        subject_id: When set, restrict to this subject; others are counted as
            ``n_filtered_out`` rather than dropped silently.
        known_run_ids: Every run id in the corpus, so a result excluded by the
            caller's own run filter is counted filtered-on-request rather than
            unplaceable. See :func:`project_score_records`.
        archived_run_ids: The corpus run ids the operator has archived, so an
            archival exclusion is counted as ``results_from_archived_runs``
            rather than under the caller's ``status`` filter. See
            :func:`place_results`.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`HistoryResult` — one :class:`MeasureSeries` per contestant,
        sorted by subject then model then variant then identity version, plus corpus-level accounting.

    Raises:
        HistoryError: ``metric`` is not one this surface can series — refused
            before any row is read, so a typo cannot return an empty series that
            reads like a measure nobody recorded.
    """
    from threetears.evals.analysis.stats import (
        EQUIVALENCE_TEST_NAME,
        PAIRED_TEST_NAME,
        paired_change,
        standard_error_of_mean,
    )

    # A series point is a mean over cases, so the catalog name for what this returns
    # is `mean_composite` / `mean_cost_usd` / `mean_total_ms` — accepted here beside
    # the row names, resolved before the closed-set check for the reason `pivot`
    # resolves before its own.
    # The caller's own spelling reaches the refusal — see `pivot` for why the resolved
    # name is the wrong thing to quote back.
    requested = metric
    metric = resolve_measure_name(metric)
    if metric not in HISTORY_METRICS:
        raise HistoryError(_unknown_history_metric(requested))

    value_of = _history_value_of(metric)
    descriptor = _describe_aggregate(metric, profile=profile)
    # All HISTORY_METRICS are directional; the guard keeps the type honest without
    # asserting an impossible None away.
    direction = descriptor.higher_is_better if descriptor.higher_is_better is not None else True
    # Decided from the descriptor once, before any row is read, so a corpus that yields no
    # series still says what its verdicts would have withheld — the empty-guard swallow the
    # sibling disclosures on this surface are already assembled ahead of.
    attribution_withheld = _attribution_withheld(descriptor)
    # The host's declared margin on the measure — the one margin a bar is read against too. Only it
    # licenses an `equivalent` step; the caller's gate never does.
    margin = descriptor.materiality_threshold
    flag_test = PAIRED_TEST_NAME if margin is None else f"{PAIRED_TEST_NAME}; {EQUIVALENCE_TEST_NAME}"

    placed, exclusions = place_results(
        runs, results, known_run_ids, source="history", archived_run_ids=archived_run_ids
    )

    series_groups: dict[tuple[str, ContestantKey], list[PlacedResult]] = {}
    subject_labels: dict[str, str] = {}
    # The short runs on these series. Keyed on the completeness record rather than on
    # status, because a `completed` run that lost a cell to a harness exclusion is short
    # too and reaches this series by default, where it is given a regression verdict.
    degraded_by_run: dict[str, str] = {}
    n_filtered_out = 0
    n_considered = 0
    n_degraded_observations = 0
    # The observations this answer rests on, narrowed to what survived the subject filter
    # — same population and same reason as `frontier`'s, so the two lenses cannot report
    # different spans over one corpus.
    considered: list[EvalResult] = []
    for row in placed:
        if subject_id is not None and row.subject_id != subject_id:
            n_filtered_out += 1
            continue
        n_considered += 1
        considered.append(row.result)
        series_groups.setdefault((row.subject_id, _contestant_key(row.result)), []).append(row)
        subject_labels[row.subject_id] = row.run.subject_snapshot.subject_label
        if (short := completeness_disclosure(row.run.completeness)) is not None:
            degraded_by_run[row.run.id] = short
            n_degraded_observations += 1

    series: list[MeasureSeries] = []
    for (resolved, contestant), rows in series_groups.items():
        runs_in_series: dict[str, EvalRun] = {}
        rows_by_run: dict[str, list[EvalResult]] = {}
        for row in rows:
            runs_in_series[row.run.id] = row.run
            rows_by_run.setdefault(row.run.id, []).append(row.result)
        ordered_run_ids = sorted(runs_in_series, key=lambda run_id: (runs_in_series[run_id].created_at, run_id))

        points: list[SeriesPoint] = []
        prev_epoch_key: tuple[str, tuple[str, ...]] | None = None
        prev_per_case: dict[str, float] = {}
        epoch_ordinal = 0
        baseline_value: float | None = None
        for index, run_id in enumerate(ordered_run_ids):
            run = runs_in_series[run_id]
            per_case, n_observations = _per_case_means(rows_by_run[run_id], value_of)
            case_means = list(per_case.values())
            value = math.fsum(case_means) / len(case_means) if case_means else None
            sem = standard_error_of_mean(case_means)

            epoch_key = _suite_epoch_key(run)
            boundary = index > 0 and epoch_key != prev_epoch_key
            if index == 0 or boundary:
                epoch_ordinal += 1

            regression: RegressionFlag | None = None
            if index > 0:
                # Compared against the immediately preceding run, because that is the pair
                # the flag is about. A series-level "these modes differ" would be true of
                # the whole series and would not say WHICH step crossed — the same reason
                # `crosses_epoch` is decided per adjacent pair rather than per series.
                crosses_cassette = run.cassette_mode != runs_in_series[ordered_run_ids[index - 1]].cassette_mode
                shared = sorted(set(per_case) & set(prev_per_case))
                verdict = paired_change(
                    [prev_per_case[case_id] for case_id in shared],
                    [per_case[case_id] for case_id in shared],
                    min_absolute_change=min_absolute_change,
                    min_relative_change=min_relative_change,
                    higher_is_better=direction,
                    equivalence_margin=margin,
                    value_range=descriptor.value_range,
                )
                regression = RegressionFlag(
                    label=verdict.label,
                    delta=verdict.delta,
                    relative_delta=verdict.relative_delta,
                    significant=verdict.significant,
                    exceeds_threshold=verdict.exceeds_threshold,
                    hedges_g=verdict.hedges_g,
                    p=verdict.p_value,
                    equivalence_p=verdict.equivalence_p,
                    equivalence_margin=verdict.equivalence_margin,
                    n_pairs=verdict.n_pairs,
                    crosses_epoch=boundary,
                    crosses_cassette_mode=crosses_cassette,
                    # Constant across the series — a property of the measure, not of this
                    # pair — and carried per flag for the reason `test` and the thresholds
                    # beside it are: a verdict read on its own must declare its own posture.
                    attribution_withheld=attribution_withheld,
                    test=flag_test,
                    min_absolute_change=min_absolute_change,
                    min_relative_change=min_relative_change,
                )

            if index == 0:
                baseline_value = value
            delta_from_baseline = value - baseline_value if (value is not None and baseline_value is not None) else None

            points.append(
                SeriesPoint(
                    run_id=run_id,
                    created_at=run.created_at,
                    model=rows_by_run[run_id][0].model,
                    value=value,
                    sem=sem,
                    n=n_observations,
                    n_cases=len(per_case),
                    epoch=epoch_ordinal,
                    epoch_boundary=boundary,
                    is_baseline=index == 0,
                    delta_from_baseline=delta_from_baseline,
                    regression=regression,
                    # Only the cost series has a composition; asking a latency or quality
                    # point what roles its dollars covered would answer a question it is
                    # not an answer to.
                    cost_compositions=(
                        pooled_cost_compositions(rows_by_run[run_id]) if metric == METRIC_COST_USD else []
                    ),
                    composite_basis=(
                        pooled_composite_basis(rows_by_run[run_id])
                        if metric == METRIC_COMPOSITE and value is not None
                        else None
                    ),
                    # On every measure, not only cost: a replayed run's LATENCY and QUALITY
                    # are as substituted as its dollars, and this is the series where a
                    # capture point and a replay point sit under one contestant heading.
                    cassette_mode=run.cassette_mode,
                    # Read from the map built over the placed rows rather than from the run
                    # again, so the point's mark and the result's corpus-level set are one
                    # predicate and cannot disagree about which runs were short.
                    completeness_disclosure=degraded_by_run.get(run_id),
                    served_models=pooled_served_models(rows_by_run[run_id]),
                )
            )
            prev_epoch_key = epoch_key
            prev_per_case = per_case

        first = rows[0].result
        # Unpacked from the group key rather than re-derived off a row, on
        # `_frontier_point`'s reasoning: the key is what placed these rows together, so
        # reading the identity back out of it is structural rather than asserted.
        variant_key, identity_version = contestant
        series.append(
            MeasureSeries(
                subject_id=resolved,
                subject_label=subject_labels.get(resolved, ""),
                variant_key=variant_key,
                variant_identity_version=identity_version,
                identity_version_disclosure=_identity_version_disclosure(identity_version),
                model=first.model,
                points=points,
                served_models=pooled_served_models([row.result for row in rows]),
            )
        )

    # Version in the key for the reason `frontier`'s point sort carries it: the partition
    # makes two series that share a subject, a model and a key reachable.
    series.sort(
        key=lambda measure_series: (
            measure_series.subject_id,
            measure_series.model,
            measure_series.variant_key,
            measure_series.variant_identity_version,
        )
    )

    identity_versions = _identity_version_span(considered)

    return HistoryResult(
        metric=metric,
        measure=descriptor,
        formula=_effective_formula(metric, WEIGHTING_EQUAL_PER_SCENARIO),
        weighting=WEIGHTING_EQUAL_PER_SCENARIO,
        higher_is_better=descriptor.higher_is_better,
        min_absolute_change=min_absolute_change,
        min_relative_change=min_relative_change,
        equivalence_margin=margin,
        series=series,
        n_results=n_considered,
        n_filtered_out=n_filtered_out,
        exclusions=exclusions,
        completeness_disclosures=degraded_by_run,
        n_degraded_observations=n_degraded_observations,
        identity_version_span=identity_versions,
        identity_span_disclosure=_identity_span_disclosure(identity_versions),
        attribution_disclosure=_attribution_disclosure(descriptor),
    )


# =============================================================================
# Program-budget view — spend, never excluding what quality excludes
# =============================================================================

# A run whose status a quality view drops BY DEFAULT. The read-tier quality
# surfaces (pivot / frontier / history) default to `status="completed"`, so any
# other status is spend that is invisible to them until a caller passes
# `status='all'` — at which point the run is admitted carrying its DEGRADED
# completeness disclosure. Named as a constant because the budget view's whole
# point is the *complement* of this set: the runs whose dollars are real even
# though their numbers carry no quality signal under the default cohort.
#
# "By default" is load-bearing and was once missing: this view stated that a
# stopped run "will never enter a quality view", which `status='all'` disproves
# in one call. Nor is the complement of this set the set of runs a quality
# surface can safely pool — a `completed` run that lost a cell to a harness
# exclusion is short too, and no status filter can see it. That is
# `RunCompleteness`'s question, not this constant's.
_QUALITY_INCLUDED_STATUS = "completed"


def pooled_composite_basis(results: Sequence[EvalResult] | Sequence[ScoreRecord]) -> CompositeBasis | None:
    """Say what a pooled composite was meaned over, and whether the pool is ragged (#638).

    Every surface that means composites across results pools numbers each meaned over whatever dimensions
    its result carried, so two pools that read alike can average different questions. Shared, as
    :func:`pooled_cost_compositions` is, so every surface reads one predicate
    (:func:`~threetears.evals.contracts.scoring.pool_composite_bases`).

    Args:
        results: The results whose composites the caller pooled, or the composite rows a pivot cell pooled
            (:attr:`ScoreRecord.dimension_basis`). A row that is not a composite row contributes nothing.

    Returns:
        The pooled basis, or None when nothing pooled carried a composite.
    """
    return pool_composite_bases(
        (result.dimension_basis if result.metric == METRIC_COMPOSITE else None)
        if isinstance(result, ScoreRecord)
        else composite_basis(result)
        for result in results
    )


def pooled_cost_compositions(results: Sequence[EvalResult] | Sequence[ScoreRecord]) -> list[list[str]]:
    """Say what a pooled cost total is made of.

    Every surface that adds ``cost_usd`` across results is adding numbers whose composition
    can differ per run, since metered calls become dollars only for a run whose operator
    declared a credit rate. The sums look identical either way, so the disclosure is the only thing
    that makes the difference readable. Shared rather than re-derived per surface for the
    usual reason: three copies of this predicate is three chances to disagree about what a
    total covers.

    Args:
        results: The results whose ``cost_usd`` the caller pooled, or the cost rows a pivot cell
            pooled (:attr:`ScoreRecord.cost_roles`, which carries the result's composition
            verbatim). A row carrying no composition — any row but a cost row — contributes none.

    Returns:
        The distinct compositions, sorted.
    """
    return sorted(list(c) for c in {tuple(result.cost_roles) for result in results if result.cost_roles is not None})


class BudgetRun(EvalBaseModel):
    """One run's real spend, counted whatever its status.

    Budget is the one lens that never excludes. A cancelled, failed, or
    budget-stopped run spent real dollars — an ``exhausted`` run carries its
    accumulated cost, never zero — and dropping it, as every quality view
    deliberately does by default, would under-report the bill. ``cumulative_cost_usd`` is
    the running total in ``created_at`` order, so the series answers "what had
    this scope cost by run N".

    ``subject_id`` / ``subject_label`` are carried as context, never as a
    partition: dollars are dollars across subjects (unlike composite quality,
    which is never pooled), so the budget totals sum over every subject. A
    blank ``subject_id`` is fine here — a run whose subject was never captured
    still spent money, and excluding it is exactly what a budget view must not
    do.
    """

    run_id: str
    status: str
    subject_id: str = ""
    subject_label: str = ""
    cost_usd: float
    n_results: int
    #: This run's results and re-judges whose spend could not be priced — real spend absent from
    #: ``cost_usd``, which sums the priced ones and is a floor whenever this is non-zero.
    n_unpriced: int
    created_at: str
    cumulative_cost_usd: float
    #: Which roles this run's dollars were summed over — the distinct
    #: ``EvalResult.cost_roles`` its results recorded. Normally one entry: a run resolves
    #: its cost convention once at launch. Empty when the run has no results.
    cost_compositions: list[list[str]] = []


class ProgramBudget(EvalBaseModel):
    """Program-lens spend over a scope, cancelled/failed/exhausted included.

    The program cost lens is *everything* — candidate, inner-agent,
    external, judge, simulator — which is what ``EvalResult.cost_usd`` already
    holds (the authoritative blended total). "Everything" is the lens, not
    necessarily the dollars: search providers report no per-call figure, so
    ``external`` contributes money only for a run whose operator declared the
    account's credit rate, and is counted-but-unpriced otherwise. That is why
    this view totals across a boundary it has to name: ``cost_compositions``
    lists the distinct role sets pooled into ``total_cost_usd``, and more than
    one entry means the number spans a change in what a dollar figure includes.
    Budget sums across **every** run
    regardless of status, which is the axis on which this view differs from the
    quality surfaces: they exclude non-``completed`` runs because an in-flight or
    failed run has no stable quality signal, and budget never does because the
    spend was real either way.

    ``n_incomplete_runs`` / ``incomplete_cost_usd`` make that asymmetry legible
    on the wire rather than only in the totals: they count the spend a quality
    view would have dropped. That bucket is then SPLIT, because its two halves
    mean opposite things to anyone deciding about a budget: an in-flight run's
    dollars have not bought a quality signal *yet* and normally will, while a
    cancelled or failed run's dollars never will. Reported as one number, the
    live half reads as waste. ``unattributed_cost_usd`` is real spend on results
    whose run is not in the listed set — normally zero within one scope, and
    surfaced rather than silently folded away so the never-exclude guarantee is
    structural, not a comment.
    """

    runs: list[BudgetRun] = []
    total_cost_usd: float = 0.0
    n_runs: int = 0
    n_incomplete_runs: int = 0
    incomplete_cost_usd: float = 0.0
    #: The half of the incomplete bucket that is still under way — ``pending`` or
    #: ``running``. Dropped by a quality view because a run that has not finished has
    #: no stable signal yet, not because the money went nowhere.
    n_in_flight_runs: int = 0
    in_flight_cost_usd: float = 0.0
    #: The statuses actually present in the in-flight half, sorted and distinct. Carried
    #: rather than left to a reader's enumeration of the status vocabulary: a surface that
    #: names the bucket from a hardcoded list describes a set it never looked at, and goes
    #: quietly wrong the day a status is added.
    in_flight_statuses: list[str] = []
    #: The half that stopped without completing — cancelled, failed, budget-stopped, and
    #: any terminal status added later. These dollars are final: the run is not coming
    #: back, so whatever it bought is all it will ever buy. That is a statement about the
    #: RUN and deliberately no longer one about the quality views: those default to
    #: ``status="completed"`` and so skip it, but ``status='all'`` admits it, where it
    #: arrives carrying its DEGRADED completeness disclosure. The earlier wording said no
    #: quality view would ever read it, which one call disproves.
    n_terminal_incomplete_runs: int = 0
    terminal_incomplete_cost_usd: float = 0.0
    #: The statuses actually present in the terminal half, sorted and distinct — same
    #: reason as :attr:`in_flight_statuses`.
    terminal_incomplete_statuses: list[str] = []
    unattributed_cost_usd: float = 0.0
    #: Distinct role sets the pooled ``total_cost_usd`` was summed over, sorted. One entry
    #: is a total whose parts mean the same thing; two or more is a total spanning a
    #: composition change, which is a real caveat on any comparison drawn across it.
    cost_compositions: list[list[str]] = []
    #: The part of the total spent re-judging results after their runs finished
    #: (``EvalResult.judge_rescores``). In the total and in each run's row because it is real
    #: program spend on that run's results; kept apart here because it is not in any result's
    #: ``cost_usd``, which measures the cell as it ran.
    rejudge_cost_usd: float = 0.0
    #: Results and re-judges across the scope whose spend could not be priced. Every dollar
    #: figure above sums the priced ones only, so a non-zero count makes each of them a floor that
    #: understates the bill by an amount nobody knows — reported beside them, never blended in.
    n_unpriced: int = 0


def compute_program_budget(runs: list[EvalRun], results: list[EvalResult]) -> ProgramBudget:
    """Sum program-lens spend per run over a scope, excluding nothing.

    Deliberately does **not** route through :func:`project_score_records`: that
    projection drops results whose run has no captured subject, which for spend
    would under-count the bill — the one thing a budget view must never do. It
    aggregates ``EvalResult.cost_usd`` (the authoritative blended total) directly
    instead, so a subject-less or failed run's dollars are still counted, plus each
    result's re-judge spend (``judge_rescores``), which that total leaves out.

    Args:
        runs: Every run in the scope, all statuses. Order is irrelevant; the
            rows are emitted in ``created_at`` order for the cumulative series.
        results: Every result in the scope. A result's ``cost_usd`` and its
            re-judge spend are attributed to its ``eval_run_id``.

    Returns:
        A :class:`ProgramBudget` — one :class:`BudgetRun` per listed run (zero
        cost for a run with no results), the cumulative series, and the
        incomplete-spend / unattributed-spend accounting that keeps the total
        honest about what it includes. The incomplete spend is reported both
        whole and split into its in-flight and terminally-incomplete halves,
        each with the statuses it was actually made of.
    """
    costs_by_run: dict[str, list[float]] = {}
    # Each run's results and re-judges whose spend went unpriced: in no dollar figure, so counted.
    unpriced_by_run: dict[str, int] = {}
    # Grouped alongside the dollars rather than looked up per row later, so a row's
    # disclosure is derived from the same results its total was summed from and the two
    # cannot come to describe different sets.
    results_by_run: dict[str, list[EvalResult]] = {}
    # A re-judge's spend rides with the result it re-scored: real program spend on that run,
    # which the result's own ``cost_usd`` deliberately leaves out.
    rejudge_cost = 0.0
    for result in results:
        spends = [result.cost_usd, *(rescore.cost_usd for rescore in result.judge_rescores)]
        result_rejudge = sum(rescore.cost_usd for rescore in result.judge_rescores if rescore.cost_usd is not None)
        rejudge_cost += result_rejudge
        costs_by_run.setdefault(result.eval_run_id, []).append(sum(spend for spend in spends if spend is not None))
        unpriced_by_run[result.eval_run_id] = unpriced_by_run.get(result.eval_run_id, 0) + spends.count(None)
        results_by_run.setdefault(result.eval_run_id, []).append(result)

    listed_ids = {run.id for run in runs}
    # Real spend on results whose run is not among those listed — never dropped,
    # merely un-rowed. Zero in the normal within-scope call; nonzero only on
    # an integrity gap, and then it belongs in the total, not the floor.
    unattributed = sum(cost for run_id, costs in costs_by_run.items() if run_id not in listed_ids for cost in costs)

    rows: list[BudgetRun] = []
    running = 0.0
    for run in sorted(runs, key=lambda r: (r.created_at, r.id)):
        costs = costs_by_run.get(run.id, [])
        cost = sum(costs)
        running += cost
        rows.append(
            BudgetRun(
                run_id=run.id,
                status=run.status,
                subject_id=_subject_id_of(run),
                subject_label=run.subject_snapshot.subject_label,
                cost_usd=cost,
                n_results=len(costs),
                n_unpriced=unpriced_by_run.get(run.id, 0),
                created_at=run.created_at,
                cumulative_cost_usd=running,
                cost_compositions=pooled_cost_compositions(results_by_run.get(run.id, [])),
            )
        )

    incomplete = [row for row in rows if row.status != _QUALITY_INCLUDED_STATUS]
    # Split by whether the run can still BECOME a quality signal. Classified against
    # NON_TERMINAL_RUN_STATUSES rather than against a list of failure statuses, for the
    # reason that constant documents: a status added to the vocabulary lands on the
    # terminal side by default, so the worst a new status can do here is be described as
    # final one release too early — never be described as failed while it is still running.
    in_flight = [row for row in incomplete if row.status in NON_TERMINAL_RUN_STATUSES]
    terminal_incomplete = [row for row in incomplete if row.status not in NON_TERMINAL_RUN_STATUSES]
    # Over EVERY result the total counted, listed and unattributed alike — the disclosure
    # has to cover the same population as the number it qualifies, or it would vouch for
    # dollars it never looked at.
    compositions = pooled_cost_compositions(results)
    return ProgramBudget(
        runs=rows,
        total_cost_usd=running + unattributed,
        n_runs=len(rows),
        n_incomplete_runs=len(incomplete),
        incomplete_cost_usd=sum(row.cost_usd for row in incomplete),
        n_in_flight_runs=len(in_flight),
        in_flight_cost_usd=sum(row.cost_usd for row in in_flight),
        in_flight_statuses=sorted({row.status for row in in_flight}),
        n_terminal_incomplete_runs=len(terminal_incomplete),
        terminal_incomplete_cost_usd=sum(row.cost_usd for row in terminal_incomplete),
        terminal_incomplete_statuses=sorted({row.status for row in terminal_incomplete}),
        unattributed_cost_usd=unattributed,
        cost_compositions=compositions,
        rejudge_cost_usd=rejudge_cost,
        n_unpriced=sum(unpriced_by_run.values()),
    )


# =============================================================================
# Orphaned runs — spend that no campaign claims, and therefore no analysis reads
# =============================================================================


class OrphanedRun(EvalBaseModel):
    """One run belonging to no campaign, with what it cost to produce.

    ``archived`` is carried rather than filtered on, and the distinction is the
    point of the row. An archived orphan was deliberately retired by an operator;
    an un-archived one is data nobody decided anything about, which is the finding
    this view exists to surface. Collapsing the two would report a curated
    exclusion as a leak.
    """

    run_id: str
    status: str
    archived: bool
    subject_id: str = ""
    subject_label: str = ""
    template_id: str | None = None
    #: The run's priced results' spend — a floor when :attr:`n_unpriced` is non-zero.
    cost_usd: float
    n_results: int
    #: The run's results whose spend could not be priced, absent from ``cost_usd``.
    n_unpriced: int
    created_at: str


class OrphanedRunsResult(EvalBaseModel):
    """Every run in a storage scope that no campaign holds, plus what that costs.

    Campaign membership is curated, never queried — a run enters a campaign only
    because an operator attached it — so a scope accumulates runs no analysis
    surface will ever read. Nothing reported the gap: a scope can hold more than twice
    the runs its only campaign holds, and the money the unheld runs spent is invisible to
    every quality and analysis view at once.

    ``n_campaigns_scanned`` is the honest denominator: every campaign in the
    scope, which is every campaign that can hold one of its runs, since a campaign
    holds only runs in its own scope. Zero campaigns makes every run an
    orphan, which is true and reads as alarming — the count is what lets a reader
    tell that state from a real leak.
    """

    orphaned_runs: list[OrphanedRun] = []
    n_orphaned: int = 0
    n_runs_in_scope: int = 0
    orphaned_cost_usd: float = 0.0
    #: Orphans an operator archived. Real spend, and deliberately retired — counted in
    #: the total, named separately so a curated exclusion is not read as overlooked data.
    n_archived_orphans: int = 0
    archived_orphan_cost_usd: float = 0.0
    n_campaigns_scanned: int = 0
    #: Orphans' results whose spend could not be priced — absent from every dollar figure here.
    n_unpriced: int = 0


def compute_orphaned_runs(
    runs: list[EvalRun],
    results: list[EvalResult],
    campaign_run_id_sets: list[list[str]],
) -> OrphanedRunsResult:
    """Report the runs no campaign claims, and the spend they represent.

    Args:
        runs: Every run in the scope, all statuses and including archived
            ones — an archived run's dollars were still spent, and this view
            reports on spend as much as on membership.
        results: Every result in the scope; a result's ``cost_usd`` is
            attributed to its ``eval_run_id``, matching :func:`compute_program_budget`.
        campaign_run_id_sets: One ``run_ids`` list per campaign in the
            scope. Taken as a list of lists rather than a pre-flattened set
            so the campaign COUNT is derived from the same argument as the claimed
            ids — a separately-passed count could disagree with the membership it
            claims to describe.

    Returns:
        An :class:`OrphanedRunsResult`, rows in ``created_at`` order.
    """
    claimed_run_ids = {run_id for run_ids in campaign_run_id_sets for run_id in run_ids}
    costs_by_run: dict[str, list[float | None]] = {}
    for result in results:
        costs_by_run.setdefault(result.eval_run_id, []).append(result.cost_usd)

    rows: list[OrphanedRun] = []
    for run in sorted(runs, key=lambda r: (r.created_at, r.id)):
        if run.id in claimed_run_ids:
            continue
        costs = costs_by_run.get(run.id, [])
        rows.append(
            OrphanedRun(
                run_id=run.id,
                status=run.status,
                archived=run.archived,
                subject_id=_subject_id_of(run),
                subject_label=run.subject_snapshot.subject_label,
                template_id=run.template_id,
                cost_usd=sum(cost for cost in costs if cost is not None),
                n_results=len(costs),
                n_unpriced=costs.count(None),
                created_at=run.created_at,
            )
        )

    archived_rows = [row for row in rows if row.archived]
    return OrphanedRunsResult(
        orphaned_runs=rows,
        n_orphaned=len(rows),
        n_runs_in_scope=len(runs),
        orphaned_cost_usd=sum(row.cost_usd for row in rows),
        n_archived_orphans=len(archived_rows),
        archived_orphan_cost_usd=sum(row.cost_usd for row in archived_rows),
        n_campaigns_scanned=len(campaign_run_id_sets),
        n_unpriced=sum(row.n_unpriced for row in rows),
    )


# =============================================================================
# Export — the projection's flat rows, as CSV or JSON
# =============================================================================

#: The two on-demand serializations. Parquet is deferred: pyarrow is not a current
#: dependency and the dependency-manifest rule governs — CSV covers DuckDB/pandas
#: ingestion, which is the stated need. A format outside this set is refused rather
#: than defaulted, so a typo'd `format=jsom` is a visible error, not a silent CSV.
ExportFormat = Literal["csv", "json"]

#: :data:`ExportFormat`'s values, in the order a refusal lists them.
EXPORT_FORMATS: tuple[ExportFormat, ...] = get_args(ExportFormat)

# A score record's open maps — neither is a column of its own. `factors` carries the
# host's levers; `host_measures` carries a code-graded kind's own grade. Each flattens to
# one column per key it holds, so a bake-off's swept knob and a classifier's accuracy are
# both pivotable pandas columns rather than nested blobs.
_EXPORT_OPEN_MAPS = ("factors", "host_measures")

# Prefix on every flattened host-measure column. Present so a grade cannot be mistaken for
# a lever in a spreadsheet, and so a host measure whose name collides with a declared
# coordinate cannot emit a duplicate header. A COLON rather than a dot, deliberately: a
# dotted column name is the open-COORDINATE syntax `_axis_value` routes to `factors`, so a
# dotted grade column would read as a pivotable axis and answer with a table of "—".
_HOST_MEASURE_COLUMN_PREFIX = "host_measure:"

# The declared ScoreRecord columns for a CSV export, derived from the model so a
# coordinate added later becomes a column with no edit here — the export layer's
# version of the open-factor rule. Derived by subtracting the open maps rather
# than by listing the scalars, for the same reason.
_EXPORT_SCALAR_COLUMNS = tuple(name for name in ScoreRecord.model_fields if name not in _EXPORT_OPEN_MAPS)

# CSV-formula-injection guard. A spreadsheet treats a cell whose text
# begins with one of these as a formula, so a user-authored value — a subject
# name, a prompt id, a run-scoped config override flattened into a factor column —
# like ``=cmd|'/C calc'!A1`` would execute on open. The characters, per OWASP.
_CSV_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _is_numeric(text: str) -> bool:
    """True when ``text`` is a plain number Python parses — safe to leave verbatim."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _csv_safe(value: str) -> str:
    """Force a spreadsheet-formula-triggering cell to text by prefixing a quote.

    A legitimate number (a negative delta, ``+1.5e3``) is left untouched: the
    export's contract is that numeric columns ingest to pandas as numbers, and
    ``'-0.5`` would corrupt that. A real formula payload never parses as a float,
    so number-gating neutralizes the attack surface without touching the data.
    """
    if value and value[0] in _CSV_FORMULA_TRIGGERS and not _is_numeric(value):
        return "'" + value
    return value


class ExportError(ValueError):
    """An export was requested in a format this surface cannot emit.

    Raised rather than defaulted to CSV: a caller who asked for ``parquet`` or
    mistyped ``jsom`` wants that request answered, not silently reinterpreted as
    a different format whose bytes they will then try to parse as the one they
    asked for.
    """


def export_records_csv(records: list[ScoreRecord] | list[dict[str, Any]]) -> str:
    """Serialize projection rows to CSV, one row per record, factors flattened to columns.

    The declared coordinates are fixed columns; every distinct open-factor key
    seen across the records becomes its own column (sorted), blank where a record
    did not carry it. That makes a swept lever a first-class pandas column — a
    kind's ``gm.difficulty`` sits beside ``model`` — rather than a nested blob a
    reader has to unpack, and it needs no per-key code, exactly as pivoting on
    such a key does.

    **A code-graded run's grade flattens the same way**, under
    :data:`_HOST_MEASURE_COLUMN_PREFIX` — ``host_measure:field_accuracy`` — so a kind
    that grades on two axes is two columns rather than a second export shape. The columns
    exist only when some record carries one, so an export of judge-graded runs alone is
    byte-identical to what it was before the grade had anywhere to go.

    Args:
        records: The projection rows, as :class:`ScoreRecord` instances or their
            JSON-safe dumps. Both are accepted because the service dumps the
            projection to JSON before the surfaces render it, while a direct
            caller has the models in hand.

    Returns:
        CSV text with a header row. A ``None`` value renders as an empty cell —
        never ``0`` or ``"None"`` — so an unmeasured observation is blank, not a
        fabricated zero, and ingests into pandas as ``NaN``. A cell whose text
        would trigger a spreadsheet formula (leading ``=``/``+``/``-``/``@``) is
        forced to text via :func:`_csv_safe`, except a legitimate number, which is
        left verbatim.
    """
    rows = [record.model_dump(mode="json") if isinstance(record, ScoreRecord) else record for record in records]
    factor_keys = sorted({key for row in rows for key in (row.get("factors") or {})})
    measure_keys = sorted({key for row in rows for key in (row.get("host_measures") or {})})
    header = list(_EXPORT_SCALAR_COLUMNS) + factor_keys + [_HOST_MEASURE_COLUMN_PREFIX + key for key in measure_keys]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for row in rows:
        factors = row.get("factors") or {}
        measures = row.get("host_measures") or {}
        cells = [
            "" if row.get(column) is None else _csv_safe(str(row.get(column))) for column in _EXPORT_SCALAR_COLUMNS
        ]
        cells += ["" if factors.get(key) is None else _csv_safe(str(factors[key])) for key in factor_keys]
        # Blank, never 0, for a measure this row did not take — the same rule `value` follows
        # one column over. A classifier that named no class reports no accuracy at all, and a
        # zero there would enter the mean as a wrong answer rather than as an absent one.
        cells += ["" if measures.get(key) is None else _csv_safe(str(measures[key])) for key in measure_keys]
        writer.writerow(cells)
    return buffer.getvalue()


def serialize_export(projection: ScoreProjection, *, fmt: str) -> str:
    """Serialize a projection to the requested export format, as a text body.

    The one serialization seam both surfaces share, so REST and MCP emit
    byte-identical exports. JSON carries the whole :class:`ScoreProjection` —
    ``records`` **and** ``exclusions`` — because an export that dropped the
    exclusion counts would let an all-excluded corpus read as an empty one, the
    same misreading the counts exist to prevent everywhere else. CSV is the flat
    rows alone; a reader who needs the exclusion accounting takes JSON.

    Args:
        projection: The rows to export, plus what the projection dropped.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The serialized body.

    Raises:
        ExportError: ``fmt`` is not one this surface can emit.
    """
    match export_format(fmt):
        case "csv":
            return export_records_csv(projection.records)
        case "json":
            return json.dumps(projection.model_dump(mode="json"))


def export_format(fmt: str) -> ExportFormat:
    """The export format a caller named, or the refusal naming the ones there are.

    Args:
        fmt: The format as the caller spelled it.

    Returns:
        It, as an :data:`ExportFormat`.

    Raises:
        ExportError: ``fmt`` is not one this surface can emit.
    """
    for known in EXPORT_FORMATS:
        if fmt == known:
            return known
    raise ExportError(f"unknown export format {fmt!r} — one of {', '.join(EXPORT_FORMATS)}")


class ScoreExport(EvalBaseModel):
    """A projection's rows serialized for analysis elsewhere, with the account a CSV body cannot carry.

    ``body`` is :func:`serialize_export`'s output byte for byte. The JSON form already holds the
    exclusions and the completeness disclosures; the CSV form is the flat rows alone, so the counts
    ride beside the body here — an export whose every result was excluded and one over an empty scope
    are both a header row, and only these fields tell them apart.
    """

    format: ExportFormat
    body: VerbatimText = Field(description="The export, exactly as serialized: CSV text or a JSON ScoreProjection.")
    n_records: int = Field(description="How many rows the export holds — one per projected observation.")
    exclusions: ProjectionExclusions
    #: ``run_id -> DEGRADED sentence`` for the exported runs that came up short of their matrix.
    completeness_disclosures: dict[str, str] = {}


def export_projection(projection: ScoreProjection, *, fmt: str) -> ScoreExport:
    """Serialize a projection in the named format, with the counts that qualify its rows.

    Args:
        projection: The rows to export, plus what the projection dropped.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The export.

    Raises:
        ExportError: ``fmt`` is not one this surface can emit.
    """
    known = export_format(fmt)
    return ScoreExport(
        format=known,
        body=serialize_export(projection, fmt=known),
        n_records=len(projection.records),
        exclusions=projection.exclusions,
        completeness_disclosures=projection.completeness_disclosures,
    )


# =============================================================================
# Cost estimation — a proposed sweep's cost from historical per-cell costs
# =============================================================================

# The band beside a cost estimate is a PREDICTION interval for what the proposed sweep
# will itself cost — not a confidence interval on the historical mean. The distinction is
# the whole reason this surface exists: the caller asks "what will this sweep cost", and
# the old `1.96 * n_obs * SEM` answered "how precisely does the history locate its own
# mean", which is a strictly narrower question and one that keeps NARROWING as history
# accumulates. Answering the narrow question in the wide question's place can publish a
# +/-2.4% band off a two-observation basis, and a sweep can then land well outside that
# band — in either direction, since nothing in that arithmetic prefers over-prediction.
#
# The band carries both terms: the uncertainty in where the history's distribution sits, and
# the variation the sweep's own observations will show. It is read on the LOG scale
# (`stats.lognormal_sum_prediction_band`), because costs are positive and right-skewed: a few
# long conversations cost several times the median. The normal-theory band it replaced,
# `t * s * sqrt(n_obs + n_obs^2 / n)`, covered a lognormal sweep's total (log-SD 1.0, five
# past observations) 82% of the time against its stated 95%. That band is kept only for a
# history holding a cost of zero or less, which the log scale cannot read.

# Smallest historical basis that gets a published band at all. Two observations yield
# exactly ONE difference: whatever spread they show is a single accident with nothing to
# check it against, and at df=1 Student's t is the Cauchy limit, where the multiplier
# contributes as much of the width as the data does. Publishing a number there dresses an
# accident as a measurement in whichever direction the pair happened to fall — narrow when
# they land close, which is the dangerous direction because narrow reads as confidence.
# Below this the cell reports its point estimate and no band, and the surfaces say in
# words why; three observations is the smallest sample whose spread rests on more than one
# difference.
#: The fewest past observations a cost estimate publishes a band from; below it, the point estimate alone.
COST_ESTIMATE_MIN_BASIS = 3

#: What every band assumes about its observations, stated on each band (:attr:`CostEstimateCell.band_basis`).
_COST_BAND_INDEPENDENCE = (
    "Each past observation and each planned one is treated as an independent draw; repeats of one case are not, "
    "so where cases differ in cost the band is narrower than it should be."
)


def _cost_band(history: Sequence[float], n_observations: int) -> tuple[float, float, str]:
    """The ~95% prediction band for the total of a sweep of ``n_observations``, and what it assumes.

    Log-scale (:func:`~threetears.evals.analysis.stats.lognormal_sum_prediction_band`) wherever every
    past cost is positive. A history with a zero or negative cost cannot be read on the log scale, and
    takes the normal-theory band ``t(0.95, n-1) * s * sqrt(m + m^2/n)`` around ``m`` times its mean, floored
    at zero — which assumes symmetric costs and says so.

    Args:
        history: The past per-observation costs, at least :data:`COST_ESTIMATE_MIN_BASIS` of them.
        n_observations: How many observations the proposed sweep would run.

    Returns:
        ``(low, high, basis)``: the band in dollars and the sentence stating its method and assumptions.
    """
    from threetears.evals.analysis.stats import (
        INTERVAL_LEVEL,
        lognormal_sum_prediction_band,
        standard_error_of_mean,
        t_critical_two_sided,
    )

    n = len(history)
    log_band = lognormal_sum_prediction_band(history, n_observations)
    if log_band is not None:
        return (
            log_band[0],
            log_band[1],
            f"{INTERVAL_LEVEL:.0%} prediction band for the sweep's total from a lognormal fit to {n} past "
            f"observations, the fit's own uncertainty included. {_COST_BAND_INDEPENDENCE}",
        )
    sem = standard_error_of_mean(list(history)) or 0.0
    centre = n_observations * math.fsum(history) / n
    half = (
        t_critical_two_sided(INTERVAL_LEVEL, n - 1)
        * sem
        * math.sqrt(n)
        * math.sqrt(n_observations + n_observations**2 / n)
    )
    return (
        max(0.0, centre - half),
        centre + half,
        f"{INTERVAL_LEVEL:.0%} normal-theory prediction band for the sweep's total from {n} past observations — "
        "some cost nothing, which the log scale cannot read, so this band assumes symmetric costs and is narrow "
        f"on the high side when a few cost far more than the rest. {_COST_BAND_INDEPENDENCE}",
    )


class CostEstimateError(ValueError):
    """A cost estimate was requested for a proposal that cannot be costed.

    Raised rather than returning a zero or empty estimate: an empty model list or
    a non-positive grid size is a malformed request, and answering it with ``$0``
    would read as "this sweep is free" rather than "you asked for nothing".
    """


class PlannedCost(EvalBaseModel):
    """One planned model's predicted cost over its planned observations, as a cost pivot's plan reads it.

    The plan a launch priced by its host's pricer hands a cost pivot: the same three facts a
    :class:`CostEstimateCell` carries for one — the model, how many observations its prediction totals, and
    the prediction — so the pivot divides it by the observations and sets it beside the cost observed, never in
    place of it.

    **Which cells it describes is part of the plan.** A prediction is for one model running one template, so it
    sits only in a pivot cell whose observations are all that template's at that model; and once the plan
    launched, ``run_ids`` names the runs it made, so a cell that pools other runs too — the earlier history the
    prediction was derived from among them — says how many of its observations the plan did not make
    (:attr:`PivotCell.n_unplanned`).

    Attributes:
        model: The planned candidate model.
        template_id: The template the plan runs; ``None`` for an estimate pooled over templates, which sits beside
            the model's cells whatever their template.
        run_ids: The runs the plan launched; empty before it launched, when the observed cost beside it is
            history the plan did not make.
        n_observations: The observations the prediction totals (cases × repeats).
        predicted: The predicted total cost, or ``None`` for a model nothing predicted.
    """

    model: str
    template_id: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    n_observations: int = Field(ge=1)
    predicted: PredictedValue | None = None


class CostEstimateCell(EvalBaseModel):
    """The predicted cost of running one proposed model, from its historical per-observation cost.

    One planned cell, and its prediction is a :class:`PredictedValue` (``method_id``
    :data:`COST_PREDICTION_METHOD`) rather than loose numbers, so the pivot can set it beside the
    cost the cell later observed without either one passing for the other.
    ``predicted.value`` is ``mean_cost_per_observation × n_observations``.
    ``predicted.interval_low`` / ``interval_high`` are the ~95% **prediction** band for what
    the proposed sweep will cost — not a confidence interval on the historical
    mean, which is a narrower claim than any caller of this surface is making — and
    ``band_basis`` says how it was drawn and what it assumes.

    The band is ``None`` when ``n_historical`` is below :data:`COST_ESTIMATE_MIN_BASIS`. At one observation the spread is
    *unknown*, not zero — the same rule the pivot applies to an n=1 cell. At two it
    is worse than unknown: two points give exactly one difference, so any band drawn
    from them is an accident dressed as a measurement, and it prints NARROW whenever
    the pair happens to land close, which is the direction that reads as confidence.
    A stated absence is the honest answer; the point estimate still stands.

    ``basis`` is ``no_history`` when no PRICED result in the corpus matches this model at
    the proposed cassette mode, and then every value is ``None``, ``predicted`` included: an estimate invented
    with no data is worse than an admitted gap. ``n_unpriced_historical`` counts the matching
    results left out because their spend went unpriced — the reason a cell can say
    ``no_history`` while the model has run.
    """

    model: str
    n_observations: int
    n_historical: int
    mean_cost_per_observation: float | None = None
    predicted: PredictedValue | None = None
    basis: Literal["historical", "no_history"]
    #: What the band assumes, in words: how ``predicted.interval_low`` / ``interval_high`` were drawn and what
    #: they leave out — among them that every observation is treated as independent. ``None`` with no band, or
    #: on an estimate stored before bands stated it: those were the normal-theory t band, which assumed
    #: symmetric costs and covered a skewed sweep's total well short of its 95%.
    band_basis: str | None = None
    #: Matching historical results whose spend could not be priced, and so are not in the basis:
    #: a mean drawn only from the priced ones prices a model that partly runs unpriced as if it
    #: never did. Counted rather than folded in as zeros, which would pull the estimate down.
    n_unpriced_historical: int = 0
    #: Which role sets the historical basis summed its dollars over. More than one entry
    #: means the mean was drawn across a change in what a cost figure includes — the same
    #: reason the basis is matched to the proposed cassette mode, surfaced rather than
    #: matched because the proposed run's own composition is not yet decided.
    basis_cost_compositions: list[list[str]] = []


class CostEstimate(EvalBaseModel):
    """A proposed run/campaign's predicted cost, per model and in total, banded where n allows.

    The estimate is deliberately an interval wherever the history can carry one:
    a single number would read as a quote when it is a projection from
    historical spread. The total bounds are the **sum** of the per-cell bounds — a
    conservative combination that assumes the cells' costs move together, so the
    band is a safe budgeting envelope rather than an optimistic one.

    ``total_interval_low`` / ``total_interval_high`` are ``None`` whenever any
    *priced* cell has no band of its own. The alternative — the previous
    behaviour — was to contribute such a cell's bare point to both bounds, which
    made the envelope tighter for the cells that knew least about themselves.
    A total that cannot be bracketed says so.

    ``n_uncovered_models`` counts proposed models whose basis was empty under the
    filters that were applied; their cost is not in the total, so a partially-covered
    estimate says how much of the proposal it could actually price rather than quietly
    costing only the covered part as if it were the whole.

    **The applied basis filters are echoed, not just obeyed.** ``subject_id`` and
    ``template_id`` are optional, and a scoped estimate can differ from a pooled one by
    more than a factor of two — so an estimate that did not carry which filters produced
    it would be indistinguishable from a pooled estimate in the artifact an operator
    reads, saves, or pastes into a budget conversation. ``None`` means the filter was not
    applied, i.e. the basis spans the whole scope on that axis.

    **``n_test_cases`` is echoed with its origin, for the same reason.** A derived 5
    and a supplied 5 are the same number and not the same claim. ``template_id`` used
    to filter the basis without touching the grid, so naming the template you were
    about to sweep still priced ONE case — a 5x understatement for a five-case template, scaling with
    ``k_runs`` and model count. Both fields are supplied by the caller: this function
    is pure over runs and results and cannot open a template, so the count and its
    provenance are resolved by the caller — the engine's launch pricer
    (:func:`~threetears.evals.ops.history_launch_pricer`) counts them off the arm's plan — and echoed here.
    """

    cassette_mode: str
    k_runs: int
    n_test_cases: int
    #: How many settings each model runs at — a campaign launch's ``variations``, each crossed
    #: with every model into one arm. ``1`` for a plain sweep. Echoed because a model's cell
    #: prices every arm that model runs, and a reader who cannot see the multiplier reads the
    #: cell as one run's cost.
    n_settings: int = 1
    #: Where ``n_test_cases`` came from: the caller stated it, it was counted off the named
    #: template's persisted cases, it is the count a launch asked to GENERATE (an upper bound:
    #: generation de-duplicates, so it can keep fewer), or there was nothing to derive from (1).
    n_test_cases_source: Literal["supplied", "derived", "generated", "default"] = "default"
    #: The named template's persisted case count in the scope; ``0`` for a template
    #: with none, ``None`` when no template was named. Reported even on the
    #: ``supplied`` path, so a caller can see the gap between their grid and the real one.
    template_case_count: int | None = None
    #: The named template's ``candidate_kind``, ``None`` when no template was named.
    #:
    #: Carried because ``template_case_count`` does not mean the same thing for every kind,
    #: and a reader with the count alone cannot tell which reading applies. A
    #: ``conversational-turn`` template with zero persisted cases has nothing to sweep; a
    #: ``classifier`` template with zero has its cases minted at launch from the subject's
    #: snapshot bank, so zero is its ordinary state. Resolved by the caller of
    #: :func:`compute_estimate_cost` and echoed here for the same reason the count is: that
    #: function is pure over runs and results and cannot open a template.
    template_candidate_kind: str | None = None
    #: The basis filters this estimate was computed under; ``None`` = not filtered.
    subject_id: str | None = None
    template_id: str | None = None
    cells: list[CostEstimateCell] = []
    total_estimated_cost: float | None = None
    total_interval_low: float | None = None
    total_interval_high: float | None = None
    n_uncovered_models: int = 0

    def planned_costs(self, run_ids: Sequence[str] = ()) -> list[PlannedCost]:
        """Each priced cell as a cost pivot's plan: its model, the template the estimate was scoped to, its prediction.

        Args:
            run_ids: The runs the plan launched, once it has; empty before.

        Returns:
            One :class:`PlannedCost` per cell with a prediction and at least one planned observation.
        """
        return [
            PlannedCost(
                model=cell.model,
                template_id=self.template_id,
                run_ids=list(run_ids),
                n_observations=cell.n_observations,
                predicted=cell.predicted,
            )
            for cell in self.cells
            if cell.predicted is not None and cell.n_observations >= 1
        ]


def compute_estimate_cost(
    runs: list[EvalRun],
    results: list[EvalResult],
    *,
    models: list[str],
    k_runs: int,
    n_test_cases: int,
    n_settings: int = 1,
    n_test_cases_source: Literal["supplied", "derived", "generated", "default"] = "default",
    template_case_count: int | None = None,
    template_candidate_kind: str | None = None,
    cassette_mode: str = "off",
    subject_id: str | None = None,
    template_id: str | None = None,
    computed_at: str | None = None,
    profile: HostProfile,
) -> CostEstimate:
    """Predict a proposed sweep's cost from historical per-observation costs.

    The proposed sweep is ``models × n_settings × n_test_cases × k_runs`` observations — a
    campaign launch runs every model at every one of its settings, one arm per pair, so each
    model's cell prices ``n_settings`` arms. For
    each model, the historical per-observation cost distribution — drawn from the
    corpus and **filtered to the proposed cassette mode** — gives a mean, scaled
    by the proposed observation count, and a ~95% **prediction** band around it.

    **The band predicts the sweep, not the history's mean**: the variation the sweep's own
    observations will show, plus the uncertainty in where the history's distribution sits. It is
    read on the log scale, because costs are positive and right-skewed
    (:func:`~threetears.evals.analysis.stats.lognormal_sum_prediction_band`); a history holding a zero
    cost takes the normal-theory ``t(0.95, n-1) * s * sqrt(n_obs + n_obs^2/n)`` instead and says so.
    Each band states its method and assumptions in the cell's ``band_basis`` — among them that every
    observation is treated as independent, which repeats of one case are not. A confidence interval on
    the mean — the construction before either — answers a narrower question and shrinks as history
    accumulates, which is why it could publish a +/-2.4% band from two observations and then miss the
    sweep it priced by 12%.

    **Below :data:`COST_ESTIMATE_MIN_BASIS` historical observations there is no band at all**, only the point
    estimate. Two observations give one difference; a width computed from it is an
    accident, and a narrow one reads as a measurement. See
    :data:`COST_ESTIMATE_MIN_BASIS`.

    **Cassette-aware.** A ``replay`` run's marginal cost differs from a live one
    (cached tool results), so history is matched to the proposed ``cassette_mode``
    rather than pooled across modes — a replay estimate rests on replay history or
    admits it has none, never borrows a live run's higher cost. Cost pools freely
    across subjects (dollars are dollars, unlike composite quality), so
    ``subject_id`` is an optional precision filter, not a required partition.

    **Template-aware, and for a different reason than the subject filter.** A
    template is not a relabelling of the same work — it fixes the test cases, and
    with them how much the subject researches and how many tokens each turn carries.
    Two templates in one scope can sit several-fold apart (say 3.4x) on the same
    candidate model, so a mean pooled across them misprices any *specific* proposal
    in whichever direction the scope's mix happens to lean. Overpricing merely
    over-reserves; underpricing lets a sweep meet ``max_cost_usd`` partway through,
    which is the failure this surface exists to prevent. Optional like
    ``subject_id`` — omitted, the basis is the whole scope, which is the right
    answer only when the question really is "what does a run here cost on average".
    Runs carrying no ``template_id`` are ad-hoc, built from explicit test cases, so
    they can match no proposed template and are excluded rather than pooled.

    Args:
        runs: The corpus runs, supplying each result's cassette mode and subject.
        results: The corpus results, supplying per-observation ``cost_usd``.
        models: The candidate models the proposed sweep would run.
        k_runs: Proposed iterations per test case.
        n_test_cases: Proposed test-case count.
        n_settings: How many settings each model runs at — a launch's ``variations``. The
            historical basis is per MODEL, not per setting, so every setting of one model is
            priced at that model's mean: a setting that changes how much the candidate works
            (``agent.max_rounds``, a longer directive) moves its real cost off that mean in
            whichever direction it moves the work, and nothing in the history can say which.
        n_test_cases_source: Where that count came from — echoed, not used. Resolving
            it needs a template read, which this pure function cannot do.
        template_case_count: The named template's persisted case count; echoed too.
        template_candidate_kind: The named template's candidate kind; echoed too, because
            what a zero case count means is the kind's answer rather than the count's.
        cassette_mode: The proposed run's cassette mode; history is matched to it.
        subject_id: Optional — restrict the historical basis to one subject.
        template_id: Optional — restrict the historical basis to one template, which
            is what the proposed sweep will actually run.
        computed_at: The instant stamped on each cell's prediction (ISO-8601); ``None`` stamps now.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`CostEstimate` — per-model cells, each predicted as a :class:`PredictedValue`, and a total, each banded only where its
        basis reaches the minimum; a thinner cell carries its point estimate alone, and one
        unbanded cell leaves the total unbanded too.

    Raises:
        CostEstimateError: The proposal is malformed — no models, or a
            non-positive grid.
    """
    if not models:
        raise CostEstimateError("no models proposed — nothing to estimate")
    if k_runs < 1 or n_test_cases < 1 or n_settings < 1:
        raise CostEstimateError("k_runs, n_test_cases and n_settings must all be >= 1 to estimate a sweep's cost")

    run_by_id = {run.id: run for run in runs}
    stamp = computed_at if computed_at is not None else utc_now_iso()

    def _eligible(result: EvalResult) -> bool:
        run = run_by_id.get(result.eval_run_id)
        if run is None or run.cassette_mode != cassette_mode:
            return False
        if template_id is not None and run.template_id != template_id:
            return False
        return subject_id is None or _subject_id_of(run) == subject_id

    costs_by_model: dict[str, list[float]] = {}
    unpriced_by_model: dict[str, int] = {}
    # The basis results themselves, kept beside their costs so each cell can disclose what
    # its mean was drawn across — the estimate pools history from either side of a
    # composition change and the pooled mean cannot show it.
    basis_by_model: dict[str, list[EvalResult]] = {}
    for result in results:
        if not _eligible(result):
            continue
        if result.cost_usd is None:
            unpriced_by_model[result.model] = unpriced_by_model.get(result.model, 0) + 1
            continue
        costs_by_model.setdefault(result.model, []).append(result.cost_usd)
        basis_by_model.setdefault(result.model, []).append(result)

    n_observations = n_settings * n_test_cases * k_runs
    cells: list[CostEstimateCell] = []
    total_estimate = 0.0
    total_low = 0.0
    total_high = 0.0
    n_uncovered = 0
    any_covered = False
    # One priced cell without a band makes the whole total unbracketable: adding its bare
    # point to both bounds would narrow the envelope on account of the cell that knows
    # least about itself, which is the wrong direction for a budgeting figure.
    total_bandable = True

    for model in models:
        history = costs_by_model.get(model, [])
        if not history:
            cells.append(
                CostEstimateCell(
                    model=model,
                    n_observations=n_observations,
                    n_historical=0,
                    basis="no_history",
                    n_unpriced_historical=unpriced_by_model.get(model, 0),
                )
            )
            n_uncovered += 1
            continue

        any_covered = True
        mean = sum(history) / len(history)
        estimate = n_observations * mean
        total_estimate += estimate

        low: float | None
        high: float | None
        band_basis: str | None
        if len(history) < COST_ESTIMATE_MIN_BASIS:
            # Too thin for a band. At n=1 the spread is unknown, not zero; at n=2 it is
            # one difference, which is an accident rather than a dispersion — and one
            # that prints narrow exactly when the pair lands close. The point estimate
            # stands and the absence is stated, never rendered as a false ± 0 and never
            # as a width the sample cannot support.
            low = high = band_basis = None
            total_bandable = False
        else:
            low, high, band_basis = _cost_band(history, n_observations)
            total_low += low
            total_high += high

        basis_compositions = pooled_cost_compositions(basis_by_model.get(model, []))
        cells.append(
            CostEstimateCell(
                model=model,
                n_observations=n_observations,
                n_historical=len(history),
                mean_cost_per_observation=mean,
                predicted=PredictedValue(
                    value=estimate,
                    interval_low=low,
                    interval_high=high,
                    method_id=COST_PREDICTION_METHOD,
                    computed_at=stamp,
                ),
                band_basis=band_basis,
                basis="historical",
                n_unpriced_historical=unpriced_by_model.get(model, 0),
                basis_cost_compositions=basis_compositions,
            )
        )

    if not any_covered:
        # Nothing in the corpus matched any proposed model at this cassette mode —
        # the honest answer is "cannot estimate", not a $0 total.
        return CostEstimate(
            cassette_mode=cassette_mode,
            k_runs=k_runs,
            n_test_cases=n_test_cases,
            n_settings=n_settings,
            n_test_cases_source=n_test_cases_source,
            template_case_count=template_case_count,
            template_candidate_kind=template_candidate_kind,
            subject_id=subject_id,
            template_id=template_id,
            cells=cells,
            n_uncovered_models=n_uncovered,
        )

    return CostEstimate(
        cassette_mode=cassette_mode,
        k_runs=k_runs,
        n_test_cases=n_test_cases,
        n_settings=n_settings,
        n_test_cases_source=n_test_cases_source,
        template_case_count=template_case_count,
        template_candidate_kind=template_candidate_kind,
        subject_id=subject_id,
        template_id=template_id,
        cells=cells,
        total_estimated_cost=total_estimate,
        total_interval_low=total_low if total_bandable else None,
        total_interval_high=total_high if total_bandable else None,
        n_uncovered_models=n_uncovered,
    )


__all__ = [
    "BADGE_CASE_SET_DIFFERS",
    "BADGE_CASE_SET_UNRESOLVED",
    "BADGE_CASSETTE_MODE_DIFFERS",
    "BADGE_CONTEXT_DIFFERS",
    "BADGE_CONTEXT_INCOMPLETE",
    "BADGE_MEASUREMENT_WINDOWS_DISJOINT",
    "BADGE_ROLES_DIFFER",
    "BADGE_TOOL_CONFIG_DIFFERS",
    "CASSETTE_SPAN_CLAUSE",
    "CELL_MEASURED",
    "CELL_NOT_RUN",
    "CELL_UNMEASURED",
    "CELL_WITHHELD",
    "COST_PREDICTION_METHOD",
    "DECLARED_INPUT_ORIGIN",
    "DEFAULT_WEIGHTING",
    "DEGRADED_RUN_CLAUSE",
    "DISJOINT_WINDOWS_CLAUSE",
    "EXPORT_FORMATS",
    "HISTORY_METRICS",
    "MAX_INLINE_MEASUREMENT_WINDOWS",
    "MAX_RENDERED_WINDOW_GAPS",
    "METRIC_COMPOSITE",
    "METRIC_COST_USD",
    "METRIC_GOAL_STATE",
    "METRIC_OUTCOME",
    "METRIC_SCORE",
    "METRIC_TOTAL_MS",
    "METRIC_TRANSCRIPT",
    "NOT_SIGNIFICANT_LABEL",
    "NOT_TESTED_LABEL",
    "NULL_LEVEL",
    "PAIRED_EFFECT_LABEL",
    "PAIRED_HEDGES_LABEL",
    "PARTITION_TOLERANCE_MS",
    "PROJECTED_METRICS",
    "RECONSTRUCTED_COUNTS_CLAUSE",
    "SCALAR_LEVEL_TYPES",
    "SCOPED_METRICS",
    "SCOPED_METRICS_HELP",
    "SIGNIFICANT_LABEL",
    "SUBSTITUTING_CASSETTE_MODE",
    "UNCOMPUTABLE_GAP_CLAUSE",
    "UNPAIRED_EFFECT_LABEL",
    "UNPAIRED_HEDGES_LABEL",
    "WEIGHTINGS",
    "WEIGHTING_EQUAL_PER_SCENARIO",
    "WEIGHTING_SAMPLE_WEIGHTED",
    "WITHHELD_PARTS_EXCEED_WHOLE",
    "WITHHELD_UNMEASURED_COMPONENT",
    "BudgetRun",
    "CaseSetIdentity",
    "ComparisonSet",
    "ComparisonSetsResult",
    "ContestantKey",
    "COST_ESTIMATE_MIN_BASIS",
    "CostEstimate",
    "CostEstimateCell",
    "CostEstimateError",
    "ExportError",
    "ExportFormat",
    "FrontierCostDecision",
    "FrontierCostTie",
    "FrontierDominance",
    "FrontierDominator",
    "FrontierError",
    "FrontierPoint",
    "FrontierResult",
    "FrontierVerdict",
    "HistoryError",
    "HistoryResult",
    "LatencyPartition",
    "MeasureSeries",
    "MeasurementWindow",
    "OrphanedRun",
    "OrphanedRunsResult",
    "OverlappingWindows",
    "PivotCell",
    "PivotError",
    "PivotTable",
    "PlacedResult",
    "PlannedCost",
    "PredictedValue",
    "ProgramBudget",
    "ProjectionExclusions",
    "RegressionFlag",
    "ScoreExport",
    "ScoreProjection",
    "ScoreRecord",
    "SeriesPoint",
    "SimpsonsFlag",
    "SubjectFrontier",
    "TwoPillarDisclosure",
    "WindowGap",
    "WindowPairs",
    "cassette_mode_disclosure",
    "classify_window_pairs",
    "completeness_disclosure",
    "compute_comparison_sets",
    "compute_estimate_cost",
    "compute_frontier",
    "compute_history",
    "compute_orphaned_runs",
    "compute_pivot",
    "compute_program_budget",
    "cross_subject_disclosure",
    "decompose_total_ms",
    "degraded_run_disclosures",
    "difference_was_declared_at_launch",
    "dim_judge_model",
    "disjoint_window_pairs",
    "export_format",
    "export_projection",
    "export_records_csv",
    "format_significance",
    "format_window_gap",
    "lever_level",
    "measurement_window",
    "measurement_window_disclosure",
    "metric_help",
    "normalize_bar",
    "place_results",
    "pooled_composite_basis",
    "pooled_cost_compositions",
    "project_score_records",
    "resolve_measure_name",
    "serialize_export",
    "significance_disclosure",
    "significance_read",
]
