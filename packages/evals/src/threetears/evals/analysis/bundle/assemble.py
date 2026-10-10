"""Closed context-bundle assembler for the eval analysis subsystem.

The generated Analysis lens is a *closed system*: the
generator reads a **pre-assembled context bundle** — the computed
reporting lenses' data plus a coverage map and prior insights — and writes the
:class:`~threetears.evals.kernel.campaign.EvalAnalysis` in one shot. Nothing fetches
during generation; the bundle *is* the context. This module builds that bundle.

Why closed matters: to A/B a generation prompt you regenerate over a **fixed**
bundle and compare. So the bundle is fingerprinted
(:meth:`AnalysisContextBundle.fingerprint`, a sha256 over its canonical JSON) and
that fingerprint is stamped onto every analysis' ``generation`` provenance — two
prompts run over the same fingerprint are comparable apples-to-apples.

**Reads reporting, never the runner.** The assembler composes existing
:mod:`threetears.evals.analysis.reporting` lenses (``compute_comparison_sets`` / ``compute_frontier`` /
``compute_program_budget`` / ``project_score_records``), the core ``stats`` /
``identity`` helpers, and storage. It introduces **no new scoring statistics** —
quality/cost/frontier math stays in ``reporting``; the only aggregation here is
descriptive telemetry (per-measure distributions and category counts, token sums)
that ``reporting`` does not provide and the golden analysis ranks
on, plus a structural per-lever coverage map. The allowed-dependency matrix
(``tests/test_package_matrix.py``) holds the ``threetears.evals.analysis`` package to
``schema``, ``kernel`` and itself — the run package's runner, simulator and judge included — and every module
of the engine is placed in a package, so no edge escapes it.

**Determinism.** Given the same runs + results + insights, the bundle — and thus
its fingerprint — is byte-identical: inputs are sorted before every lens call, every
time value derives from the runs and results rather than from the wall clock, and the
canonical encoding sorts keys. Those time values are ``window`` (from run
``created_at``) and ``measurement_windows`` / ``measurement_window_disclosure``, which
derive from result ``scored_at`` — a DIFFERENT field, and the reason a fixture fix once
took four passes: this line said "only ``window``/``created_at``", so pinning
``created_at`` alone looked complete while five leaves still moved (both measurement windows'
``start`` and ``end``, plus the disclosure rendered from them). The generation timestamp and token cost are
*not* in the bundle — they live on ``GenerationProvenance``, set at generate time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from datetime import UTC, datetime
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from itertools import chain, product
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol

from pydantic import BaseModel

from threetears.evals.analysis.agreement import (
    JudgeKey,
    judge_agreement,
    judge_evidence_tiers,
    judge_key,
    judge_self_agreement,
    inter_judge_agreement,
    tier_for_judges,
)
from threetears.evals.kernel.evidence_tiers import (
    JudgedEvidenceTier,
    JudgeEvidenceTier,
)
from threetears.evals.analysis.judge_drift import judge_drift
from threetears.evals.analysis.arms import arm_names, surface_order
from threetears.evals.analysis.contention import (
    contended_latency_sentence,
    withheld_latency,
    withhold_contended_latency,
)
from threetears.evals.analysis.cells import (
    ApparatusClass,
    Cell,
    Observation,
    apparatus_class_of,
    pool_observations,
    subject_key_instabilities,
)
from threetears.evals.analysis.confusion import label_statistics
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.reporting import (
    METRIC_COMPOSITE,
    METRIC_OUTCOME,
    METRIC_SCORE,
    METRIC_TRANSCRIPT,
    FrontierResult,
    ProgramBudget,
    ScoreRecord,
    completeness_disclosure,
    compute_comparison_sets,
    compute_frontier,
    compute_program_budget,
    decompose_total_ms,
    pooled_composite_basis,
    lever_level,
    measurement_window,
    measurement_window_disclosure,
    pool_served_readings,
    project_score_records,
    ResultServedReading,
    served_reading,
)
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    LevelDifference,
    bounded_difference_interval,
    clustered_standard_error,
    composite_significance,
    contrast_samples,
    difference_interval,
    equivalence_untested_reason,
    exact_decimal,
    GUARDRAIL_HELD_NEEDS_RANGE,
    guardrail_decision,
    holm_adjust,
    interval_permits_separation,
    interval_clears,
    level_difference,
    no_spread_p,
    observed_mean_interval,
    paired_equivalence,
    proportion_interval,
    separation_test,
    small_sample_case_means,
)
from threetears.evals.kernel.analysis_measures import BarAdjudication, BarVerdict, MeasureCollection, MeasureSummary
from threetears.evals.kernel.campaign import EvalInsight, ReadingKind, VariantIndexEntry, derive_window
from threetears.evals.kernel.scoring import median_unbiased_quantile
from threetears.evals.kernel.declaration import (
    JUDGED_MERIT_AXIS,
    BarName,
    CampaignDesign,
    SweptAxis,
    UnreadableBarName,
    axis_in_question_scope,
    exploratory_reading,
    resolve_bar_name,
)
from threetears.evals.schema.hashing import canonical_digest, canonical_json
from threetears.evals.kernel.host.profile import CANDIDATE_MODEL_LEVER, UNSEATED_LEVEL, HostProfile
from threetears.evals.schema.values import PooledProductionFooting, SweepableValue
from threetears.evals.kernel.identity import IDENTITY_VERSION, resolve_variant_identity
from threetears.evals.kernel.metrics import (
    ACCURACY_MEASURE,
    CONFUSION_CELL_MEASURE,
    FRONTIER_RANKING_MEASURE,
    MATCH_MEASURE,
    AttributionScope,
    ClassifierStatistic,
    MeasurePopulation,
    MeritAxis,
    MetricDescriptor,
    classifier_label_measure,
    classifier_label_of,
    confusion_of,
    declaration_of,
    describe_measure,
    describe_phase_timing,
    describe_reported_measure,
    describe_rubric_dim,
    goal_check_measure,
    is_code_graded,
    is_latency_measure,
    materiality,
    partition_components,
    remainder_withheld_reason,
    summary_population,
    undeclarable_host_measures,
)
from threetears.evals.kernel.covariates import undeclarable_covariates

# At runtime for its field set, which tells a result-level measure from a row-level one.
from threetears.evals.schema.models import (
    goal_check_proofs_as_read,
    stale_goal_check_proofs,
    CalibrationRating,
    EvalResult,
    GoalCheckProof,
    MeasureDeclaration,
    RubricScale,
)
from threetears.evals.kernel.provider import sum_optional_tokens
from threetears.evals.kernel.result_condition import (
    JUDGE_CANNOT_TELL_OUTCOME,
    ResultOutcome,
    classify_result,
    counted_goal_verdicts,
    delivered_a_turn,
    harness_faulted,
)
from threetears.evals.kernel.surface import (
    CellFacts,
    DecisionSurface,
    FrontierDominance,
    GuardrailCell,
    GuardrailCheck,
    GuardrailReadings,
    JudgedDimensionFacts,
    JudgedReading,
    MeasureFacts,
    StratumFacts,
    TimeAxis,
    TimeAxisBasis,
    TimePosition,
    all_failed_sentence,
)
from threetears.evals.kernel.usage_capture import (
    count_substituted_deliveries,
    production_replicating_cost,
    spend_observed,
)

from threetears.evals.analysis.bundle.caps import (
    _capped,
    _CELL_STATES,
    _MAX_DECLARED_CELLS,
    _MAX_DIVERGENCES,
    _MAX_FACTOR_PAIR_PIVOTS,
    _MAX_NEXT_EXPERIMENTS,
    _MAX_PRIOR_INSIGHTS,
    _MAX_REFUSED_MERGES,
)

from threetears.evals.analysis.bundle.schema import (
    _OBSERVED_MECHANISMS,
    _SERVED_MODEL_CONFOUNDS,
    AliasedFactors,
    AnalysisContextBundle,
    ArmMechanismReading,
    ArmServedModel,
    CANDIDATE_SERVED_MODEL_CONFOUND,
    CellCoordinate,
    ComparedCell,
    ComparisonFamily,
    ComparisonVerdict,
    Confound,
    DeclaredCellCoverage,
    DeclaredCrossing,
    DeclaredLevelCoverage,
    DesignArm,
    exploratory_disclosure,
    FactorPairCell,
    FactorPairPivot,
    FactorPairScan,
    FamilyComparison,
    GoalCheckProofReading,
    HeldFixedReading,
    host_declarations_digest,
    JudgeChange,
    JudgedArm,
    JudgedMeasure,
    JudgeDriftLink,
    JudgeIdentityLevel,
    LeverCoverageInput,
    MeasureMovement,
    MechanismCheck,
    MechanismUncheckedReason,
    MeritTier,
    MovementDirection,
    MultipleComparisons,
    observed_mechanism_key,
    OBSERVED_MECHANISM_PREFIX,
    QuestionScope,
    ReadingScope,
    RealizedDesign,
    RunSummary,
    ScopeDivergence,
    ShortCell,
    TelemetryRollup,
    TokenRollup,
    UNDECIDED_CONFOUND_PREFIX,
    UNVERIFIED_FOLD_PREFIX,
    VerdictOrder,
)

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.kernel.host.measures import MeasureRegistry
    from threetears.evals.kernel.campaign import EvalCampaign
    from threetears.evals.schema.models import EvalCaseStratum, EvalRun, SecondJudge


class CampaignReadStore(Protocol):
    """The six reads assembling a campaign's context bundle needs.

    Cut to what :func:`assemble_context_bundle` calls rather than to what a
    storage layer offers: the bundle reads member runs (in one batch, without the
    payload paths the host declares a listing may leave out), each run's results, the
    stratum each of their cases declares, the
    people's calibration ratings of those results, the
    subject's prior insights, and — for an insight that names one — whether the
    analysis that minted it is archived, and writes nothing at all. That flag is read
    because archiving an analysis RETRACTS what it minted (:func:`retracted_insights`),
    and it is a fact about the analysis, not the insight. Only the flag is asked for, so
    one stored analysis a newer validator rejects cannot abort assembly. A wider port would hand a
    second consumer a contract naming templates and campaigns to implement for a
    function that asks neither.

    Structural, so a host's own storage satisfies it by having the methods —
    :class:`~threetears.evals.kernel.storage.EvalStorage` does, with no
    inheritance and no registration.

    Every read is within the campaign's scope, which holds its runs, their results, and
    the insights and analyses minted over it. Positional parameters are positional-only,
    so an implementation's own parameter names never have to match the port's.
    ``scope_id`` is the engine's word for a partition it never interprets.
    """

    def load_eval_runs(
        self, run_ids: Sequence[str], scope_id: str, /, *, elide_payload: frozenset[str]
    ) -> list[EvalRun]:
        """Load the named runs within a scope in one read, leaving ``elide_payload`` out of each payload.

        A run that does not resolve there is absent from the answer.

        **Every returned run must record what the read left out**: an implementation calls
        :meth:`~threetears.evals.schema.models.EvalRun.note_elided_payload` with ``elide_payload`` on each run
        it returns. The store's projection cannot say so itself — a document with a path left out
        looks exactly like one stored without it — and an unmarked run reads as whole, so rebuilding
        the host's subject from it, or writing it back, proceeds with the value silently gone
        instead of refusing.
        """
        ...

    def query_eval_results_by_run(self, run_id: str, scope_id: str, /) -> list[EvalResult]:
        """Every result belonging to one run within a scope."""
        ...

    def load_case_strata(self, test_case_ids: Sequence[str], scope_id: str, /) -> list[EvalCaseStratum]:
        """The stratum each named test case declares, within a scope, in one read; an absent case is skipped.

        Read so each cell can be summarised again per kind of case (:attr:`CellFacts.strata`). Only the
        stratum is asked for, so a case's stimulus — the host's, and possibly large — is never shipped.
        """
        ...

    def query_calibration_ratings(self, scope_id: str, /, *, run_id: str) -> list[CalibrationRating]:
        """Every calibration rating of one run's results within a scope, oldest first — unpaged."""
        ...

    def query_insights(self, scope_id: str, /, *, subject_id: str) -> list[EvalInsight]:
        """Every insight recorded against a subject in a scope, newest observation first."""
        ...

    def analysis_archived(self, analysis_id: str, scope_id: str, /) -> bool | None:
        """Whether one stored analysis is archived, or ``None`` when it no longer resolves in the scope."""
        ...


# A cell needs at least this many repeats to read as measured rather than thin.
# k=1 is noise-dominated on a single template (a standing project learning:
# confirm finalists at k>=3+), so k<3 downgrades a swept lever to ``thin``.
_MEASURED_K_FLOOR = 3

# The registry's full attribution-scope vocabulary, so a scope that observed nothing can
# be named as absent rather than being missing. Ordered end-to-end first, matching how a
# reader narrows: the whole run, then the part of it under test.
_ATTRIBUTION_SCOPES: tuple[AttributionScope, ...] = ("end_to_end", "subsystem")

# The observation unit of a measure the result carries at most once — its own scalars, its
# open maps, and any single sub-model's leaves. Everything else is named for the list it
# rides on, so two units compare equal only when one observation of each describes the same
# thing. See ``_collect_measures``.
_PER_RESULT = "result"

# The result's blended spend, as a measure — the one lineage leaf that is read only where it was observed.
_COST_MEASURE = "cost_usd"

#: The spend belonging to the roles production runs — the candidate's cost, without the judge's. The one cost
#: a contrast between arms is tested on (:func:`_per_case_values`); ``cost_usd`` is what it cost to measure.
_CANDIDATE_SPEND = "production_replicating_cost"

# How a divergence is decided. It is a test of the DIFFERENCE between the two movements, never two
# movements graded apart and set side by side: "the whole moved" beside "the part did not" is the
# difference between a significant and a non-significant result, which is not itself significant
# (Gelman & Stern 2006), and with no divergence at all it published one 11% to 33% of the time. So
# each case's remainder (its whole minus its part, per-case means) is tested between the two levels
# by the engine's between-level test (`stats.level_difference`), and the lever's tests are corrected
# together by Holm's method. A movement graded on its own still reads `improved`, `regressed`,
# `not_separated` or `equivalent`, but it is context: no verdict on one movement decides a divergence.

# The distinguishing clause of each reason a cross-scope difference is withheld. The reason
# reaches the generator as a SENTENCE (see ``ScopeDivergence.unattributed_withheld``), and that
# prose gets tuned for its reader — so the clause that identifies WHICH condition fired is named
# here and shared with the tests. Rewording the sentence around it is then free, while changing
# what a test actually pins takes editing this line.
WITHHELD_OPPOSITE_DIRECTIONS = "have opposite better-directions"
WITHHELD_DIFFERENT_POPULATIONS = "are averaged over different populations"
WITHHELD_UNKNOWN_POPULATION = "has no observation unit"

# Why another swept lever varying inside a cohort clouds the comparison. Generic on
# purpose: which lever it is says nothing extra here, because a campaign sweeping it
# already believes it can move the numbers — that belief is what makes it a lever.
_SWEPT_LEVER_CONFOUNDS = (
    "another lever this campaign swept, which also took more than one value across these runs, "
    "so part of the movement may belong to it"
)

# Why a RESOLVED SURFACE still clouds a comparison after its family's swept members are taken
# back out of it. It reaches a confound list only in that state — where the residuals agree, the
# surface's movement IS the members' movement seen a second time and it is not named at all — so
# the sentence can say what the remaining difference means rather than hedging between the two.
_UNEXPLAINED_SURFACE_CONFOUNDS = (
    "the resolved surface that {family}'s members are written into, and it differed across these runs "
    "by more than the members swept here account for — something besides the swept knobs changed, so "
    "part of the movement may belong to whatever that was. What this dimension is: {prose}"
)

# Why a surface folded into a FIXED knob without a check still qualifies the comparison. The surface is
# in the variant key, so every run of an arm resolves one surface; with one arm per level of the knob the
# fold holds by construction, and a second change made exactly where the knob changed would fold with it.
# Named on every comparison that folds it that way, so a memo cannot present it as a checked non-confound.
_UNVERIFIED_FOLD_CONFOUNDS = (
    "folded, unverified: {surface} is the resolved surface {knob} is written into, and it moved with {knob} "
    "here — but every level of {knob} in these runs was run by one arm only, so nothing in them could have "
    "shown {surface} moving apart from {knob}. It is reported as the same change as {knob} without having been "
    "checked, and part of the movement may belong to anything else written into it; two arms at one level of "
    "{knob} would test it. What {surface} is: {prose}"
)

# The same, for a surface a FIXED lever is written into (a kind's overlay marked ``ResolvesInto``).
# It has no members to take out, so the sentence states the fold rule's own two ways of failing:
# runs that held the knob at one level carried different surfaces, which the knob cannot have done,
# or a run did not record the surface at all (the confound's ``undecided`` status says which).
_UNEXPLAINED_WRITTEN_SURFACE_CONFOUNDS = (
    "the resolved surface that {family} is written into, and across these runs it was not shown to have "
    "moved only where {family} did — it differed between runs that held {family} at one level, or some run "
    "did not record it — so part of the movement may belong to something besides that knob. What this "
    "dimension is: {prose}"
)


#: How a world dimension is named where it sits beside the sweepable apparatus dimensions.
#:
#: Prefixed rather than carried under its bare name, so the two registries' entries sit in disjoint
#: namespaces of one dimension map. ``HostProfile`` refuses a name registered in both, and the
#: prefix keeps the map's correctness from resting on that refusal: bare names that met would
#: collide and one fact would overwrite the other, which is the wrong-merge direction nothing
#: downstream can undo. The two kinds of entry say different things: a sweepable says which level
#: a run swept an input to, a world dimension says whether a run seeded it at all.
_WORLD_DIMENSION_PREFIX = "world:"


# What a world dimension placed differently across a cohort does to the comparison. The engine
# owns this clause because it is identical for every host and every dimension — one run seeded the
# state and another let the subject witness whatever was there, so the two were not measured
# against the same starting world. The host's own ``matters`` prose is appended to it rather than
# used as it, because the two answer different questions in different registers: ``matters`` says
# why a scenario would presume the dimension, and this slot says what a difference in it costs a
# reader. Handing one to the other produced a sentence that read as neither.
_WORLD_PLACEMENT_CONFOUND = (
    "the subject was placed in a different world across these runs — one seeded this dimension and "
    "another let the subject witness whatever was there — so they did not start from the same state"
)


def _world_confound_reason(matters: str) -> str:
    """Compose one world dimension's confound reason: the engine's clause, then the host's why.

    Args:
        matters: The dimension's required ``matters`` prose.

    Returns:
        The reason a confound catalog renders for it.
    """
    return f"{_WORLD_PLACEMENT_CONFOUND}. What this dimension is: {matters}"


def world_dimension_key(dimension: str) -> str:
    """Name ``dimension`` for the apparatus maps, where sweepables and world dimensions share a namespace.

    Args:
        dimension: A world dimension name, as its host declared it.

    Returns:
        The prefixed key. See :data:`_WORLD_DIMENSION_PREFIX`.
    """
    return f"{_WORLD_DIMENSION_PREFIX}{dimension}"


def _apparatus_confound_reasons(profile: HostProfile) -> dict[str, str]:
    """Apparatus dimension -> why a change in it clouds the measurement, for one host.

    Read straight off the host's own declarations, so a dimension a host registers reaches the
    confound scan with its reason attached and cannot arrive as a bare name.

    **World dimensions join the sweepable apparatus here**, because a world that moved across a
    campaign is the same defect one axis over: a dimension the subject perceives and one run
    seeded while another merely witnessed is a rival explanation for whatever moved, and one that
    silently widens the variance of every run in the campaign. Their reason is composed by
    :func:`_world_confound_reason` — the engine's generic clause about what a placement difference
    costs, plus the host's own ``matters`` prose about what the dimension is.

    **Built per call from the profile handed in rather than at import**, which is not a style
    choice: two hosts can assemble bundles in one process, and a module-level dict would freeze
    whichever host's declarations it was first built from and scan every other host against them.

    Args:
        profile: The host whose apparatus and world declarations the reasons come from.

    Returns:
        ``{dimension: reason}`` for every apparatus sweepable the host declares, plus every
        world dimension under :func:`world_dimension_key`.
    """
    reasons = {
        declared.name: declared.confounds
        for declared in profile.sweepables.declarations
        if declared.role == "apparatus" and declared.confounds is not None
    }
    if profile.world is not None:
        reasons.update(
            {
                world_dimension_key(declared.name): _world_confound_reason(declared.matters)
                for declared in profile.world.declarations
            }
        )
    return reasons


#: What the frontier lens's clearing count means when no bar was passed to it — nothing. The
#: sentence the MCP frontier render states for the same condition, carried on the bundle, and followed by
#: why no bar was passed (:func:`_frontier_bar`).
_FRONTIER_BAR_WITHHELD = (
    "withheld — no bar was supplied to the frontier lens, so it names no verdict and each subject's "
    "n_cleared_bar is a default of 0, not a count of arms that failed a bar. The bars this campaign is "
    "held to are adjudicated in bar_adjudications."
)


def _frontier_bar(
    behavior: str, design: CampaignDesign | None, *, profile: HostProfile
) -> tuple[float | None, str | None]:
    """The bar the frontier is given, or why it is given none.

    The frontier ranks on pass^k (:data:`~threetears.evals.kernel.metrics.FRONTIER_RANKING_MEASURE`) and
    reads a bar only on it, while a campaign's bars may name any measure. So it takes the campaign's
    effective bar on that measure — the declared one, else the host's registered one, exactly as
    :func:`_applicable_bars` resolves every bar — and none otherwise; a bar on another measure is adjudicated
    per cell in ``bar_adjudications`` and is never translated onto pass^k. The frontier reads the bar on each
    contestant's pass^k interval by the three-valued rule every bar is read by
    (:func:`~threetears.evals.analysis.stats.interval_clears`), with no margin, which is pass^k's own: its
    descriptor declares none.

    **More than one effective bar on pass^k is refused, not picked between**, as is one the frontier cannot
    read — lower-is-better, or a threshold outside pass^k's range [0, 1] (a host's registered bar is not
    range-checked where it is registered) — and the reason says so.

    Args:
        behavior: The campaign's behavior, the scope a registered bar is keyed under.
        design: The campaign's declaration.
        profile: The host whose registered bars apply.

    Returns:
        ``(bar, None)`` when the frontier takes a bar, else ``(None, reason)``: the withheld sentence followed
        by which bars exist, and on what measure, or why the one on pass^k was not passed.
    """
    bars = _applicable_bars(behavior, design, profile=profile)
    on_ranking = [bar for bar in bars if bar[0] == FRONTIER_RANKING_MEASURE]

    def named(bar: tuple[str, float, bool, str]) -> str:
        measure_id, threshold, higher_is_better, source = bar
        return f"{measure_id} {'≥' if higher_is_better else '≤'} {format_number(threshold)} ({source})"

    if len(on_ranking) == 1:
        _, threshold, higher_is_better, _ = on_ranking[0]
        if higher_is_better and 0.0 <= threshold <= 1.0:
            return threshold, None
        why = (
            f"The one bar on {FRONTIER_RANKING_MEASURE}, {named(on_ranking[0])}, is not one the frontier can read: "
            "pass^k is a probability whose higher end is better, so its bar is a threshold in [0, 1] to reach."
        )
    elif on_ranking:
        why = (
            f"{len(on_ranking)} bars name {FRONTIER_RANKING_MEASURE}, the measure the frontier ranks on — "
            f"{'; '.join(named(bar) for bar in on_ranking)} — and it takes one, so it was given none rather than "
            "a pick between them."
        )
    elif bars:
        why = (
            f"The frontier ranks on {FRONTIER_RANKING_MEASURE} and reads a bar only on it; this campaign's bars are "
            f"on other measures — {'; '.join(named(bar) for bar in bars)} — so none was passed to it."
        )
    else:
        why = "This campaign is held to no bar, declared or registered."
    return None, f"{_FRONTIER_BAR_WITHHELD} {why}"


# =============================================================================
# Assembly
# =============================================================================


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """A measure's ``q`` quantile, median-unbiased, or ``None`` where its sample cannot give one.

    :func:`~threetears.evals.kernel.scoring.median_unbiased_quantile` (Hyndman–Fan type 8), the rule the
    run summary's ``p95_total_ms`` reads too, so a tail figure means one thing on every surface. It replaced
    linear interpolation (numpy's default), which at the sizes a campaign's cells have sat below the true
    95th percentile 0.84 (n=5), 0.73 (n=15) and 0.68 (n=30) of the time — a tail figure that understates
    the tail. The median is unchanged by the switch: type 8 and linear interpolation place it alike.

    Args:
        sorted_values: Ascending-sorted values, at least one element.
        q: Quantile in (0, 1) — NOT 0-100.

    Returns:
        The estimate, or ``None`` for ``p05``/``p95`` below 13 observations, where no estimate is
        median-unbiased; ``max`` beside it is then the worst case seen, under its own name.
    """
    return median_unbiased_quantile(sorted_values, q)


#: The one name the candidate model answers to, everywhere. The coverage map, the divergence
#: lens and the effective configuration all reach for this axis, and two of them holding two
#: names for it would let the bundle report a lever twice — or, worse, report coverage for one
#: name while the config names the other, which is a bundle disagreeing with itself in one
#: payload.
#:
#: It is the one lever :func:`_lever_value` reads off the OBSERVATION rather than off the run,
#: because it is a declared coordinate of one (``ScoreRecord.model``): every record knows which
#: model produced it, including a record whose candidate role left no usage row for the run-level
#: resolution to recover a model from. **The name's shape has nothing to do with it** — that rule, dotted through
#: the effective config and plain off the record, is what bound every plain-named host lever to
#: ``'—'``, which is why it was retired.
#:
#: Declared in :mod:`threetears.evals.kernel.host.sweepables`, beside the core declaration that carries it,
#: so the literal has one owner; :mod:`threetears.evals.kernel.host.profile` re-exports it, and that is
#: where a host recovery rule claiming the name is refused. Read here.
_CANDIDATE_MODEL_LEVER = CANDIDATE_MODEL_LEVER


def _observed_model_levers(profile: HostProfile) -> dict[str, str]:
    """Levers whose *inherited* value is recoverable from what the run actually did.

    Maps a lever name to the :class:`RoleUsage` role whose ``model`` records it. A run that
    never exercised the role simply does not carry the lever — "the role never ran" and "the
    role ran at an unknown value" are different answers.

    The candidate entry is the engine's and is here for the same reason a host's is, arrived at
    from the other side. The candidate model was a lever only where a campaign held more than
    one, so a single-arm campaign's configuration named an INNER-AGENT model and stayed silent
    about the model that actually produced its numbers. A generator handed those two side by
    side — one in a slot labelled "the config", the other in a slot naming the candidate model — read the
    pair as one value contradicting itself and reported a config-provenance defect that did not
    exist. They never disagreed; they are different roles, and nothing in the bundle said so.

    Everything else comes from the host's profile, because the lever names are its tool
    vocabulary: hardcoding one here would attribute a second consumer's inner agent to the first
    consumer's tool, silently, in the module a paid generator reads.

    Built per call from the profile handed in rather than at import, because two hosts can
    assemble bundles in one process.

    Returns:
        The engine's rule merged with the host's. A host declaration colliding with the engine's
        reserved name cannot reach here — :class:`~threetears.evals.kernel.host.profile.HostProfile`
        refuses it at registration.
    """
    return {**profile.observed_model_levers, _CANDIDATE_MODEL_LEVER: "candidate"}


#: The level a run sits at for a lever it did not override and that nothing recovers — "ran
#: at the subject's own setting". A real cohort every un-overriding run shares, NOT a stand-in
#: for missing data: a lever whose value IS recoverable never lands here, and one that applied
#: but resolved ambiguously is left out of the lever entirely rather than pooled into it.
_INHERITED_DEFAULT_LEVEL = "—"


@dataclass(frozen=True)
class EffectiveLever:
    """One lever's value for one run, with how that value was established.

    Three states, never collapsed to two. ``overridden`` means the value was NAMED rather than
    recovered, and it has three sources: a launch that named it, an open family that RESTATES a
    fixed lever — the restatement IS the naming, which is the rule the one-vocabulary rule adds
    and the one most likely to be re-litigated — and a fixed declaration's own reader, which is
    why a campaign that swept nothing still shows every applicable lever stamped this way. A value recovered from what the run observably
    did is ``inherited``; a lever that applied but whose value no record pins is ``unknown`` with
    ``value=None``.

    The distinction is load-bearing for cohort assignment. Reading absence as a
    level of its own puts a run that INHERITED a value and a run that explicitly
    SET the same value into different cohorts, and reports a lever as moved when
    nothing moved — the control arm of any sweep inherits, so it is exactly the
    arm that gets mislabelled.
    """

    value: str | None
    provenance: Literal["overridden", "inherited", "unknown"]


def _observed_models(results: list[EvalResult], role: str) -> set[str]:
    """Every distinct model the given usage role spent tokens on across ``results``."""
    return {usage.model for result in results for usage in result.usage if usage.role == role and usage.model}


def _effective_config(run: EvalRun, results: list[EvalResult], *, profile: HostProfile) -> dict[str, EffectiveLever]:
    """Resolve a run's levers to the values it actually ran at, with provenance.

    **Read through the host's registry, never off one host's carriers.** The registry is the
    single lever vocabulary, so this asks
    :meth:`~threetears.evals.kernel.host.sweepables.SweepableRegistry.resolve_levers` what this run's
    levers are called and what it carried under them. While it read one host's overlay fields by
    hand, a host whose levers live anywhere else got ``coverage == []`` and an empty
    ``RunSummary.config``, and a real model reading that bundle correctly refused to attach
    outcomes to levels.

    Three sources, in this order, and the order is the provenance:

    1. **An open family's members** — the levers the launch NAMED. A kind overlay
       ``house_rules={'flanking': 'on'}`` resolves the member ``gm.house_rules.flanking``, which is
       the lever's declared name rather than a carrier path this function flattened for itself.
       A member named as ``null`` is the level :data:`~threetears.evals.analysis.reporting.NULL_LEVEL`,
       ``overridden`` — unless the lever has a recovery rule, which reads a null as "not stated"
       and resolves it in the second pass instead.
    2. **Recovery from observation** — :func:`_observed_model_levers` maps a lever to the usage
       role whose ``model`` is the value that ran. Two or more distinct models under one role
       make the value genuinely ambiguous, which is ``unknown`` rather than a guess at the first.
       A lever with a recovery rule is resolved by it alone: where the role never ran, the lever
       does not apply to this run.
    3. **The declaration's own reader** — a host lever carried on the run record with no overlay
       carrier and no inheritance tier, which is what the toy host's ``chunk_tokens`` is.

    **Recovered means ``inherited``, and that is a statement about how the value was
    established here, not about whether the run chose it.** The candidate model is the
    case that makes the distinction visible: a launch names its candidate model on
    ``EvalRun.candidate_model``, yet what this function reports is what the ``candidate`` role
    observably spent tokens on — the model that produced the numbers, which a provider's
    routing can make something other than the one declared. ``overridden`` means the value was
    NAMED rather than recovered, and it has three sources — a launch that named it, an open
    family that RESTATES a fixed lever, and a fixed declaration's own reader (source 3 above),
    which is why a campaign that swept nothing still shows every applicable lever stamped this
    way. This sentence said ``overridden`` was "reserved for the overlays above, the launch's
    departure from the subject's own configuration"; that was the third site of one correction
    already applied to :class:`EffectiveLever` and to ``RunIndexEntry.config_provenance``, and
    :func:`_resolve_config` twelve lines below has stated the rule correctly the whole time — a
    reader who believed this one would have concluded a lever stamped ``overridden`` was a
    departure when it may simply be what the declaration reads.

    Args:
        run: The run to read overlays from.
        results: That run's results — the observation side of the resolution.
        profile: The host whose vocabulary this reads.

    Returns:
        Each applicable lever mapped to its :class:`EffectiveLever`.
    """
    return _resolve_config(run, results, profile=profile)[0]


def _resolve_config(
    run: EvalRun, results: list[EvalResult], *, profile: HostProfile
) -> tuple[dict[str, EffectiveLever], frozenset[str]]:
    """:func:`_effective_config`, keeping the ENGAGED set its first two passes already know.

    The coverage map needs both: the resolved levers, and which of them the launch named or
    observation recovered — that pair is what :func:`_reportable_levers` decides a row on, and
    provenance cannot supply the second half, since ``overridden`` is written both by a family
    member and by a fixed declaration's own reader.

    Returned rather than re-derived at the call site. ``SweepableRegistry._resolve`` is one pass
    precisely so two surfaces cannot get different answers out of a reader that is not perfectly
    pure, and a caller re-asking reopens that window one layer up — on top of paying for every
    declaration's reader a second time, per run, for a lens that already had the answer.

    Args:
        run: The run to read.
        results: That run's results.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(each applicable lever's EffectiveLever, the levers this run ENGAGED)``.
    """
    resolution = profile.sweepables.resolve_levers(run, results)
    recovery = _observed_model_levers(profile=profile)
    flat: dict[str, EffectiveLever] = {}
    for lever in sorted(resolution.overlaid):
        value = resolution.values.get(lever)
        if value is None and lever in recovery:
            # A recovery rule gives ``null`` its own meaning — "not stated, read it off what ran" —
            # so the second pass resolves it, to ``inherited`` or ``unknown``, never to a level.
            continue
        # Anywhere else a NAMED null is a level the operator set (``NULL_LEVEL``). Skipping it, as
        # this once did, dropped the lever from the config, its provenance and the coverage map
        # together, so a sweep between a value and ``null`` read ``unswept`` (#574).
        flat[lever] = EffectiveLever(lever_level(value), "overridden")
    for lever, role in recovery.items():
        if lever in flat:
            continue
        observed = _observed_models(results, role)
        if not observed:
            # The role never ran, so the lever does not apply to this run at all.
            continue
        one = next(iter(observed)) if len(observed) == 1 else None
        flat[lever] = EffectiveLever(one, "inherited" if one else "unknown")
    for lever, value in resolution.values.items():
        # A lever the host declared a recovery rule for is resolved by that rule ALONE: reaching
        # its declaration's reader here would report the run-level projection (the candidate
        # model lever reads the run's whole model LIST) as one observation's level, and a lever
        # whose role never ran genuinely does not apply to the run rather than sitting at
        # whatever the record happens to hold. ``None`` here is a declaration's reader finding
        # nothing on this run — the lever does not apply — unlike a launch-named null, which the
        # first pass has already placed.
        if lever in flat or lever in recovery or value is None:
            continue
        flat[lever] = EffectiveLever(lever_level(value), "overridden")
    return flat, frozenset(resolution.overlaid | (set(recovery) & set(flat)))


def _effective_values(run: EvalRun, results: list[EvalResult], *, profile: HostProfile) -> dict[str, str]:
    """Each lever whose value IS established, as plain strings.

    Levers resolving to ``unknown`` are absent rather than present-and-empty: a
    caller comparing cohorts must be unable to accidentally treat "we could not
    establish this" as a level.
    """
    return {
        lever: eff.value
        for lever, eff in _effective_config(run, results, profile=profile).items()
        if eff.value is not None
    }


def _lever_value(record: ScoreRecord, lever: str, effective_by_run: dict[str, dict[str, EffectiveLever]]) -> str | None:
    """Read one lever's value for a score record, resolved the way the run lenses resolve it.

    Every lever but one is answered from the record's RUN effective configuration, never from
    ``record.factors``. ``factors`` is the launch-overlay flattening, so reading a lever off
    it puts a run that inherited a value in a different cohort from one that named the same
    value — the coverage surface reproducing, per record, the defect the run-derived lenses
    were converted to remove.

    The exception is :data:`_CANDIDATE_MODEL_LEVER`, and it is the ONLY one: the engine reserves
    that name for a declared coordinate of the observation itself, and
    :class:`~threetears.evals.kernel.host.profile.HostProfile` refuses a host recovery rule that claims it.
    Every record knows which model produced it, including one whose candidate role left no usage
    row for the run-level resolution to recover a model from, so reading it off the record is the
    true answer.

    **The name's SHAPE decides nothing.** This once switched on the dot — dotted through
    the effective config, plain off the record — which bound every host lever with a plain name
    (``chunk_tokens``) to ``'—'`` and reported a genuinely swept axis as ``unswept``, while
    ``RunSummary.config`` showed its two levels plainly. A lever's name is the host's, and a host
    that spells one without a dot is not thereby making a claim about the observation.

    This and :func:`_lever_levels` agree about every lever but that one, because a coverage entry
    and a divergence read over disagreeing cohorts cannot both be right about the same lever. On
    the candidate model they read different sources, and that function's docstring says why: it bins
    on the run's DECLARED model so a run whose candidate left no usage row keeps its level, where
    this reads the model one observation ran on. A run carries one model, so the two name the same
    level for every run assembly admits.

    Args:
        record: The score record to read.
        lever: A declared lever name.
        effective_by_run: Each run's resolved levers, keyed by run id.

    Returns:
        The level this record sits at, or ``None`` when the lever applied to its run but
        resolved ambiguously — no level can be claimed there, so the caller drops the record
        rather than pooling it into a cohort it cannot support.
    """
    if lever == _CANDIDATE_MODEL_LEVER:
        return _INHERITED_DEFAULT_LEVEL if not record.model else str(record.model)
    effective = effective_by_run.get(record.run_id, {}).get(lever)
    return _INHERITED_DEFAULT_LEVEL if effective is None else effective.value


def _is_reportable(descriptor: MetricDescriptor, measures: MeasureRegistry) -> bool:
    """Whether a measure belongs on the bundle's measure surfaces — what the generator reads and ranks from.

    Three filters, each excluding a class of measure that would otherwise mislead:

    - **Seeded only.** ``family is None`` marks a name nobody has described; the
      registry itself refuses to guess at one, and pooling it here would be that same
      guess made silently.
    - **Code-graded families only** (:func:`~threetears.evals.kernel.metrics.is_code_graded`, the
      predicate the bar-name resolver asks too, so the two cannot disagree). The generator ranks on
      mechanism, never on judged quality — and quality already has a home in the ``reporting`` lenses.
      This is a filter on registry *metadata*, not on a subject or scenario type. A host's own family
      is admitted exactly when the host declared it ``graded_by="code"``.

      **``classifier`` is admitted beside ``mechanical``, and the omission was a real
      defect**: the descriptive-telemetry rule's ranking half is about JUDGE scores — "no judge can score
      a run that produced nothing to grade" — and a classifier's grade has no judge
      anywhere in it. It is code compared against an expected label, which is the same
      kind of fact ``mechanical`` names. While the classifier was a second path its
      measures never reached a bundle and the proxy cost nothing; folding it in made
      every classifier campaign assemble a bundle carrying the parse-failure rates and
      **not accuracy, precision or recall** — the figures the campaign exists to produce.
      The remaining exclusions: ``rubric`` and ``dual_axis`` are the judge-mediated ones
      that rule actually names, and they stay out permanently. ``composite`` stays out because
      **no producer puts one on a bundle surface**, so admitting it would be widening on a
      case nothing exercises — which is how the ``mechanical``-only proxy came to be written.

      **``goal_state`` is admitted, on the classifier's argument**:
      a check's verdict is code compared against what the candidate did. Its exclusion had
      rested only on the no-producer claim, and that claim was about the check's own text —
      ``GoalStateOutcome`` carries ``expression`` / ``passed`` / ``detail``, so the generic
      carrier walk yields names and never a verdict. :func:`_goal_check_leaves` is the
      producer: one 0/1 per check per observation, named by
      :func:`~threetears.evals.kernel.metrics.goal_check_measure`, so each check's pass rate reaches every
      measure surface whether or not a bar names it.
    - **A numeric measure must have a direction, unless it is a declared diagnostic.**
      ``higher_is_better is None`` marks a raw count with no better end — per-role token
      counts, call counts. Nothing can be ranked on one, and pooling a per-role count
      across the candidate, judge and simulator rows produces a distribution of nothing in
      particular. A measure whose descriptor declares
      :attr:`~threetears.evals.kernel.metrics.MetricDescriptor.diagnostic` is the one exception,
      and a declared one — the engine's own (the candidate provider's output rate) and a host's
      (a signed error against what was asked) alike, through this one predicate: it has no better
      end either, but it explains a movement and a reader needs it beside the measures it
      explains. Declared rather than inferred, because nothing else on a descriptor tells a
      diagnostic from a count, and a guess that admitted counts would reopen the pooling defect.
      It is carried here so the run summaries, the catalog
      and the divergence lens see it; its missing direction is what keeps every direction-
      reading surface — a superlative, a bar — from treating it as a merit. Categorical
      measures are kept without a direction: they carry the *how did it conclude* signal
      (a forced-vs-voluntary split) that is ranked on as a rate, not a value. **Boolean and text
      measures are kept too**: a boolean is summarised as a rate with an interval and a text
      measure is listed as evidence, and neither is ever averaged into a distribution.
    """
    if not is_code_graded(descriptor, measures):
        return False
    if descriptor.data_type in ("categorical", "boolean", "text"):
        return True
    if descriptor.data_type != "numeric":
        return False
    return descriptor.higher_is_better is not None or descriptor.diagnostic


def _value_fits(descriptor: MetricDescriptor, value: float | str) -> bool:
    """Whether an observation's Python type agrees with what its descriptor claims.

    ``covariates`` is an open ``dict[str, str | float]``, so a value's type is not
    guaranteed to match the descriptor its key resolves to. Without this check a string
    arriving under a seeded-numeric name reaches ``float(value)`` and raises out of
    ``assemble_context_bundle``, destroying the whole analysis over one bad observation.
    Dropping the observation instead keeps the blast radius at one measure — and the drop
    is reported rather than silent (see ``MeasureCollection.unreported_observations``).
    """
    if descriptor.name == CONFUSION_CELL_MEASURE:
        # A confusion cell is categorical AND has a format: one that does not split into its two labels
        # could not be counted into the matrix the per-label statistics are derived from.
        return isinstance(value, str) and confusion_of(value) is not None
    if descriptor.data_type in ("categorical", "text"):
        return isinstance(value, str)
    if descriptor.data_type == "boolean":
        return isinstance(value, bool)
    return not isinstance(value, (str, bool))


def _scalar_leaves(model: BaseModel) -> Iterator[tuple[str, float | str]]:
    """Yield ``(field_name, value)`` for each present numeric/string scalar on a model.

    Booleans are excluded despite being ints in Python: a boolean measure is a
    condition, and averaging one into a percentile reads as a rate nobody computed.
    """
    for field_name in type(model).model_fields:
        value = getattr(model, field_name, None)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            yield field_name, float(value)
        elif isinstance(value, str):
            yield field_name, value


#: Result fields whose sub-models RECORD something done to the result rather than something
#: measured in its cell, so the measure walk must not read their scalars as observations. A
#: re-judge's record carries its own ``cost_usd`` — spend an operator paid after the run —
#: and walking it would pool that as a second cost observation of the cell, beside the prose
#: of its timestamp, prior error and model reported as measurements the registry lost. A repeat
#: of the judge's scores is a measurement of the JUDGE, read by ``judge_self_agreement``; walking
#: it would pool a repeat's score as a second observation of the candidate's cell. A second judge's scores are the
#: same: a measurement of the judges, read by ``inter_judge_agreement`` and ``judge_drift``.
RECORD_CARRIERS: frozenset[str] = frozenset({"judge_rescores", "judge_repeats", "judge_seconds"})


def _carrier_leaves(result: EvalResult, *, profile: HostProfile) -> Iterator[tuple[str, float | str, bool, str, str]]:
    """Yield ``(name, value, report_gaps, carrier, observation_unit)`` for every sub-model's leaves.

    This is the whole subject-agnostic mechanism: a result announces what it measured
    by *carrying* it, and the registry says what each name means. A subsystem that
    later lands its own lifecycle carrier is reported here with no change to this
    module — which is the property that makes the surface genuinely subject-agnostic
    rather than shaped for one tool with the name filed off.

    The carrier identity yielded last is the **field name** on the result, not the record
    class. Two subsystems can share a record type, subclass one, or land a generic
    lifecycle record; they cannot share a field, so the field is what distinguishes
    "these are two different quantities" from "these are more observations of one".

    ``report_gaps`` says whether an *undescribed* leaf of this carrier is worth reporting
    as a lost measurement. It is True only when the registry already describes at least one
    of the carrier's leaves, which separates the two ways a carrier can be undescribed:

    - **A described carrier that grew an undescribed field** — the likely drift, and a real
      loss. Reported.
    - **A carrier whose names are an OPEN name space** — judged rubric dimensions and
      goal-state expressions, which ``metrics.py`` deliberately does not enumerate and
      resolves through ``describe_rubric_dim`` / ``describe_goal_state`` instead. Nothing
      is lost, and reporting them would bury the real signal in permanent false positives.

    The blind spot is deliberate: a carrier with *no* described leaf at all — a wholly
    unregistered subsystem — reports nothing here, because it is indistinguishable at
    runtime from the open-name-space case.

    The last value yielded is the leaf's **observation unit** — what one observation of it
    describes. A single sub-model contributes at most one observation per result, so its
    measures are per-result (``'result'``) and may be differenced against each other; a LIST
    contributes one per element, so its measures are means per element of that list
    (``'usage[]'``, ``'async_deliveries[]'``) and differencing one against a per-result
    mean is wrong by however many elements a result carried. The two are indistinguishable
    once the values are pooled — both arrive in the measure's own honest unit — so the walk
    is the only place the difference can be observed, and it says so by name rather than
    leaving each comparison site to re-derive it.
    """
    for field_name in type(result).model_fields:
        if field_name in RECORD_CARRIERS:
            continue
        value = getattr(result, field_name, None)
        repeatable = isinstance(value, list)
        carriers: list[Any] = [value] if isinstance(value, BaseModel) else value if isinstance(value, list) else []
        observation_unit = f"{field_name}[]" if repeatable else _PER_RESULT
        for item in carriers:
            if not isinstance(item, BaseModel):
                continue
            leaves = list(_scalar_leaves(item))
            report_gaps = any(describe_measure(name, profile.measures).family is not None for name, _ in leaves)
            for name, leaf in leaves:
                yield name, leaf, report_gaps, field_name, observation_unit


def _derived_leaves(result: EvalResult) -> Iterator[tuple[str, float | str, bool, str, str]]:
    """Yield the measures a result implies but does not carry, in the carrier-leaf shape.

    Two. ``candidate_output_tokens_per_s`` is the candidate provider's output rate — see
    :func:`_candidate_output_throughput` — which is what separates a latency difference a
    lever caused from one the provider's load caused. The other is ``orchestration_ms``, the
    named remainder that closes the ``total_ms`` partition. Both are derived rather than
    captured on purpose — their inputs already settle them, so persisting either would be a
    second answer to one question — but a derived measure the bundle never walks is one the
    analysis cannot rank on, and this surface is the only way a subsystem reaches a report at all.

    Without it, a whole-run latency movement that lived in orchestration could only be
    reported as ``total_ms`` moving by more than ``llm_ms`` and ``tool_ms`` account for, which
    is indistinguishable in a report from a measurement fault. With it the movement has
    a component to be attributed to, and — because the registry declares it
    ``contained_by: total_ms`` — the divergence lens will difference it against the
    whole rather than refusing.

    Both are yielded under the ``latency`` carrier and at the per-result observation unit,
    matching the components they are computed from: a remainder pooled over a different
    unit from its own parts would be exactly the mismatch this decomposition exists to remove.
    """
    partition = decompose_total_ms(result.latency)
    if partition.orchestration_ms is not None:
        # `report_gaps=True`: the latency carrier's other leaves are all described, so
        # an undescribed one here would be real drift rather than an open name space.
        yield "orchestration_ms", partition.orchestration_ms, True, "latency", _PER_RESULT
    if (throughput := _candidate_output_throughput(result)) is not None:
        yield "candidate_output_tokens_per_s", throughput, True, "latency", _PER_RESULT


def goal_check_proofs_of(runs: Sequence[EvalRun], results: Iterable[EvalResult]) -> list[GoalCheckProofReading]:
    """Each goal check the runs' results graded, with the proof its runs froze at launch, in the order first met.

    Args:
        runs: The member runs.
        results: Their results.

    Returns:
        One reading per check.
    """
    graded: dict[str, list[str]] = {}
    for result in results:
        for outcome in result.goal_state_outcomes:
            runs_of = graded.setdefault(outcome.expression, [])
            if result.eval_run_id not in runs_of:
                runs_of.append(result.eval_run_id)
    by_id = {run.id: run for run in runs}
    refusals: dict[str, tuple[str, list[str]]] = {}
    for run in runs:
        for check, reason in (run.refused_goal_checks or {}).items():
            refusals.setdefault(check, (reason, []))[1].append(run.id)
    readings = []
    for check, run_ids in graded.items():
        members = [by_id[run_id] for run_id in run_ids if run_id in by_id]
        # As read under the current proof rules: a `proven` an older rule stamped is unproven (#665).
        recorded = [goal_check_proofs_as_read(run) for run in members]
        proofs = [None if record is None else record.get(check, "unproven") for record in recorded]
        refused = refusals.get(check)
        proof: GoalCheckProof = (
            "refuted"
            if "refuted" in proofs or refused is not None
            else "proven"
            if proofs and all(each == "proven" for each in proofs)
            else "unproven"
        )
        readings.append(
            GoalCheckProofReading(
                check=check,
                measure_id=goal_check_measure(check),
                proof=proof,
                runs=max(len(run_ids), 1),
                unrecorded=sum(1 for each in proofs if each is None),
                stale=sum(1 for run in members if check in stale_goal_check_proofs(run)),
                refused=None if refused is None else refused[0],
            )
        )
    # A check the grammar refused is graded on no cell, so no result names it; it is read from the runs.
    readings.extend(
        GoalCheckProofReading(
            check=check,
            measure_id=goal_check_measure(check),
            proof="refuted",
            runs=len(run_ids),
            unrecorded=0,
            refused=reason,
        )
        for check, (reason, run_ids) in refusals.items()
        if check not in graded
    )
    return readings


def _goal_check_leaves(result: EvalResult) -> Iterator[tuple[str, float | str, bool, str, str]]:
    """Yield each goal-state check's verdict on this observation, 1.0 passed and 0.0 not.

    The mechanical tier's verdicts, which the bundle carries unconditionally: a bar is a
    threshold someone chose to hold a check to, never the condition for the check's pass rate
    reaching a reader. The generic carrier walk cannot supply them — a ``GoalStateOutcome``'s
    scalars are its expression and detail as TEXT, and ``passed`` is a boolean the walk skips —
    so they are yielded here, one per check, under the name the registry mints for a check's
    measure. Per result, since an observation evaluates each of its checks once. Each verdict is
    the one :func:`~threetears.evals.kernel.result_condition.counted_goal_verdicts` says every rate counts, so
    this and the pivot cannot disagree: none from a harness-faulted result, whose checks read the
    harness (the cells already skip it through :func:`_non_faulted`, but run summaries, the
    telemetry rollup and the scope divergences collect over every result, and without this they
    would report a second rate for the same check), and a failure for every check on a candidate
    failure.
    """
    counted = counted_goal_verdicts(result)
    if counted is None:
        return
    for outcome, passed in counted:
        yield goal_check_measure(outcome.expression), 1.0 if passed else 0.0, False, "goal_state_outcomes", _PER_RESULT


def _failures_as_misses(results_by_run: dict[str, list[EvalResult]]) -> dict[str, list[EvalResult]]:
    """Each run's results, with a classifier's failure that landed no verdict read as a miss.

    A classifier kind lands ``match`` on every classification, and a call its model refused lands nothing
    unless the kind says otherwise — the quick callable kind does, a host's kind may not. Left absent, the
    failure was in no rate: not the match rate, not ``accuracy``, not a comparison's per-case values, so an
    arm that refused the cases it would have got wrong read MORE accurate than one that answered them, and
    one that refused everything had no accuracy to compare at all. So a candidate failure carrying no
    ``match`` is read with ``match`` False — the rule
    :func:`~threetears.evals.kernel.result_condition.counted_goal_verdicts` keeps for every goal check —
    when both of these hold:

    - **its case is a classification**: some result in the campaign landed ``match`` on that test case. A
      case nothing classified is not one a failure could have missed, whatever kind ran it.
    - **its run classifies**: no result the run delivered without failing lacks ``match`` on a case that is
      a classification. A run that answers a classified case without landing a verdict grades by something
      else (a scorer-only run of the same callable kind, beside a classifier run over the same cases), and
      giving its failures an accuracy would invent one. A run that delivered nothing has shown no such
      evidence, and its refusals are misses.

    Copied, never written back: what the kind stored is untouched.

    Args:
        results_by_run: Each run's stored results.

    Returns:
        The same results, each such failure replaced by a copy carrying ``match`` False.
    """
    classified_cases = {
        result.test_case_id
        for members in results_by_run.values()
        for result in members
        if isinstance(result.host_measures.get(MATCH_MEASURE), bool)
    }
    grading_otherwise = {
        run_id
        for run_id, members in results_by_run.items()
        if any(
            result.test_case_id in classified_cases
            and MATCH_MEASURE not in result.host_measures
            and classify_result(result) is ResultOutcome.OK
            for result in members
        )
    }

    def read(run_id: str, result: EvalResult) -> EvalResult:
        if (
            run_id not in grading_otherwise
            and result.test_case_id in classified_cases
            and MATCH_MEASURE not in result.host_measures
            and classify_result(result) is ResultOutcome.CANDIDATE_FAIL
        ):
            return result.model_copy(update={"host_measures": {**result.host_measures, MATCH_MEASURE: False}})
        return result

    return {run_id: [read(run_id, result) for result in members] for run_id, members in results_by_run.items()}


def _accuracy_leaves(result: EvalResult) -> Iterator[tuple[str, float | str, bool, str, str]]:
    """Yield the observation's classifier accuracy, 1.0 matched and 0.0 not, derived from its ``match``.

    ``match`` is the boolean a classifier kind lands, and a boolean summarises as a rate with no
    per-case mean, so it cannot carry the classifier's reading on the quality axis into a family of
    comparisons. ``accuracy`` is that reading, derived here from the one carried verdict — so a host
    lands one measure and every surface sees one comparison, rather than a host minting its own
    numeric copy beside ``match`` and every family testing the same verdict twice. A kind may not
    land ``accuracy`` itself (:func:`~threetears.evals.run.runner.refuse_engine_derived_host_measures`).
    Nothing is yielded for an observation that carries no ``match``, or one whose ``match`` is not a
    bool — that value is the walk's to drop and report, under ``match``'s own name.

    **A candidate failure is a miss**, whatever its ``match`` says — the rule
    :func:`~threetears.evals.kernel.result_condition.counted_goal_verdicts` keeps for every goal check,
    carried to a classifier's accuracy. A refused call answered nothing; read as anything but a miss, an arm
    that refused the cases it would have got wrong read MORE accurate than one that answered them. A
    classifier kind lands ``match`` on a failure for this reason — the quick callable kind lands it False
    with the expected label's confusion cell — so a failure is in the accuracy, the match rate and the
    per-label counts alike; and a host kind's failure that landed none is read with ``match`` False before
    the walk sees it (:func:`_failures_as_misses`). A failed observation still carrying no ``match`` is of a
    kind that classifies nothing, and yields nothing: an accuracy for it would be invented.

    Args:
        result: The observation.

    Yields:
        ``("accuracy", 1.0 | 0.0, True, "host_measures", "result")`` at most once.
    """
    matched = result.host_measures.get(MATCH_MEASURE)
    if isinstance(matched, bool):
        hit = matched and classify_result(result) is not ResultOutcome.CANDIDATE_FAIL
        yield ACCURACY_MEASURE, 1.0 if hit else 0.0, True, "host_measures", _PER_RESULT


def _candidate_output_throughput(result: EvalResult) -> float | None:
    """The candidate's output tokens per second of its own model-call time, or None.

    Both halves are the CANDIDATE's: ``llm_ms`` sums the candidate's model-call spans, and the
    tokens are summed over its own usage rows alone — a judge's or an inner agent's tokens were
    produced on a different clock and would inflate a rate over time they did not spend. Absent,
    never zero, when either half went unmeasured: a result with no candidate token count has no
    rate, and zero would read as a provider that produced nothing.

    Args:
        result: The observation.

    Returns:
        Tokens per second, or None when the candidate's output tokens or a non-zero ``llm_ms``
        is missing.
    """
    llm_ms = result.latency.llm_ms if result.latency is not None else None
    counted = [
        row.completion_tokens for row in result.usage if row.role == "candidate" and row.completion_tokens is not None
    ]
    if not llm_ms or not counted:
        return None
    return sum(counted) / (llm_ms / 1000.0)


def _withheld_derived(results: list[EvalResult], reported: Collection[str]) -> list[str]:
    """Name a derived measure no result could produce, with the reason it could not.

    A derived measure is the one kind that can vanish from the bundle without anything
    noticing. A CARRIED measure that is absent had no instrument; a derived one may have
    had every instrument and still be unreportable because an input was unmeasured — and
    reporting neither the value nor the reason hands the analysis exactly the unexplained
    gap the measure was added to close.

    Silence is still correct in two cases, and both are checked rather than assumed. When
    the measure IS reported, the mean carries its own ``n``, so partial coverage is
    already disclosed the way every other measure discloses it. When no result timed
    anything at all, nothing was withheld — the campaign simply has no latency, which
    ``absent_scopes`` is the right place for.

    Args:
        results: The results the collection was built from.
        reported: Measure names the walk successfully summarised.

    Returns:
        Entries of the form ``name (reason)``, empty when nothing was withheld. The
        reason is carried only when every withholding result gave the SAME one —
        otherwise the name alone, since a single reason chosen from several would be a
        claim about results it does not describe.
    """
    if "orchestration_ms" in reported:
        return []
    reasons = {
        p.withheld for result in results if (p := decompose_total_ms(result.latency)).withheld and result.latency
    }
    if not reasons:
        return []
    return [f"orchestration_ms ({reasons.pop()})" if len(reasons) == 1 else "orchestration_ms"]


def _lineage_leaves(result: EvalResult, *, profile: HostProfile) -> Iterator[tuple[str, float | str, MetricDescriptor]]:
    """Yield the result's own top-level scalars.

    Mostly lineage and provenance — ids, a schema version, the k index — with a couple of
    genuine measures mixed in (``cost_usd``). The registry sorts the two apart, so nothing
    here needs a hand-written exclusion list; but an *undescribed* name here is a version
    field rather than a lost measurement, which is why the caller does not report gaps
    from this source.

    **``cost_usd`` is an observation only where the result observed spend**
    (:func:`~threetears.evals.kernel.usage_capture.spend_observed`): a row in its cost roles carrying
    dollars. Without one the stored 0.0 is the sum of nothing — a candidate that reported no spend, not one
    that spent none — so it is left out here, and with it out of every cell, run, case and stratum this
    walk summarises. Decided per result, so every slicing of the same results agrees; a cell where no
    result observed spend carries no ``cost_usd`` reading at all, so nothing charts or tests it, and
    :attr:`AnalysisContextBundle.cost_unmeasured` says why.
    """
    observed = spend_observed(result.usage, result.cost_roles)
    for name, value in _scalar_leaves(result):
        if name == _COST_MEASURE and not observed:
            continue
        yield name, value, describe_measure(name, profile.measures)
    # The candidate's own spend, where the result measured one: what the arm costs, beside ``cost_usd``, what
    # it cost to measure (the judge's and simulator's spend included). Derived here rather than stored, so every
    # slicing the walk serves — a cell, a run, a case, a stratum — reads the one figure a contrast on cost tests.
    candidate_spend = production_replicating_cost(
        result.usage, substituted_deliveries=count_substituted_deliveries(result)
    )
    if candidate_spend is not None:
        yield _CANDIDATE_SPEND, candidate_spend, describe_measure(_CANDIDATE_SPEND, profile.measures)


def _open_map_leaves(
    result: EvalResult, *, profile: HostProfile
) -> Iterator[tuple[str, float | str, MetricDescriptor]]:
    """Yield the result's covariate, phase-timing and host-measure entries — all open key spaces.

    Phase timings resolve through :func:`describe_phase_timing` rather than
    :func:`describe_measure`, because a phase key is indistinguishable *by name* from a
    run-summary statistic and the bare resolver says so explicitly. The caller here
    knows which map it is reading, so it uses the resolver that knows too.

    Keys here are operator- and tool-authored rather than model fields, so a blank one is
    reachable, and the registry resolvers refuse it (correctly: an empty string is the
    absence of a name, not a name that failed to resolve). Skip rather than raise, so one
    malformed key cannot destroy the analysis it appears in.
    """
    # A core-named covariate no covariate writer lands — a result stored by another writer, or before the rule —
    # is dropped and named, never pooled, as an undeclarable host measure is (`_undeclarable_host_entries`).
    stray = set(undeclarable_covariates(result.covariates))
    for name, value in result.covariates.items():
        if name.strip() and name not in stray:
            yield name, value, describe_measure(name, profile.measures)
    for name, value in result.phase_timings.items():
        if name.strip():
            yield name, float(value), describe_phase_timing(name)
    # Host-declared measures resolve through `describe_measure`, which consults the
    # host's measure registry between the seeded core and the goal-state branch. So a name the host
    # declared arrives with the host's own descriptor — merit axis and direction included — and
    # one it did not stays undescribed and is reported as a lost measurement, exactly as an
    # unrecognised covariate is.
    #
    # The core wins a tie on the DESCRIPTOR, and only on the descriptor. A host reporting
    # `cost_usd` here would not redefine what the word means — but its numbers WOULD pool into
    # the engine's own spend distribution under that core descriptor, with `n` inflated,
    # because `record()` appends into one bucket per name at this level. So a host may not
    # DECLARE a measure named like a core one: `MeasureRegistry._defects` refuses it, and
    # `run_eval` refuses a scorer so named. A core name still arrives here legitimately — the
    # classifier track lands `match` and `confusion_cell` as host measures — and those pass. Any
    # other engine-owned key is one no host could have declared: the runner refuses a kind landing
    # one, and a result stored before that refusal has it dropped here and named as unreported
    # (`_undeclarable_host_entries`), never pooled into the engine's own observations of the name.
    smuggled = set(undeclarable_host_measures(result.host_measures))
    for name, value in result.host_measures.items():
        if name.strip() and name not in smuggled:
            yield name, value, describe_measure(name, profile.measures)


def _undeclarable_host_entries(results: Sequence[EvalResult]) -> list[str]:
    """The unreported-observation entries for host-measure keys the walk dropped as engine-owned.

    Each entry is the key with its reason in parentheses, the form
    :attr:`~threetears.evals.kernel.analysis_measures.MeasureCollection.unreported_observations` reads —
    the plain name could not carry it, because the engine's own measure of that name is usually pooled
    beside it and the bare name would read as a gap in the engine's reading rather than a drop of the host's.

    Args:
        results: The results the walk read.

    Returns:
        One entry per dropped key, sorted.
    """
    names = {name for result in results for name in undeclarable_host_measures(result.host_measures)}
    covariates = {name for result in results for name in undeclarable_covariates(result.covariates)}
    return [
        f"{name} (a host kind reported it on host_measures, where only the engine measures it; dropped, not pooled)"
        for name in sorted(names)
    ] + [
        f"{name} (a result carried it as a covariate, which no covariate writer lands; dropped, not pooled)"
        for name in sorted(covariates)
    ]


def in_population(population: MeasurePopulation, result: EvalResult) -> bool:
    """Whether a result is an observation of a measure read over ``population``.

    The one membership rule the measure walk applies, per measure and per result: a result the harness
    faulted is in ``all_observed`` only; ``delivered`` holds exactly the turns the candidate took
    (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`), so a failure that took no turn
    — a refusal, a model error — is in every population but that one, and a failure that took a turn (one
    its budget ended, its output cap cut, its deadline struck mid-call) is in all three.

    Args:
        population: The population the measure's summary is computed over
            (:func:`~threetears.evals.kernel.metrics.summary_population`).
        result: The result.

    Returns:
        True when the result's observations of the measure count.
    """
    if population == "delivered":
        return delivered_a_turn(result)
    if population == "scored":
        return not harness_faulted(result)
    return True


#: One measure's pooled observations: its descriptor, its values, and each value's test case.
_PooledMeasure = tuple[MetricDescriptor, list[float | str], list[str]]


def _measure_collection(
    results: list[EvalResult], *, profile: HostProfile, undeclared: MeasurePopulation
) -> MeasureCollection:
    """Build the scope-tagged measure surface over a set of results.

    See :func:`_collect_measures`, which this wraps for the callers that need only the
    collection and not the provenance of each measure.
    """
    return _collect_measures(results, profile=profile, undeclared=undeclared)[0]


def _collect_measures(
    results: list[EvalResult], *, profile: HostProfile, undeclared: MeasurePopulation
) -> tuple[MeasureCollection, dict[str, str], dict[str, _PooledMeasure]]:
    """Build the measure surface, and say what one observation of each measure describes.

    **Each measure is computed over its own population** (``MetricDescriptor.population``): a
    ``scored`` measure leaves out every result the harness faulted, an ``all_observed`` one keeps
    them, a ``delivered`` one holds only the turns the candidate took
    (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`), and every summary states which
    it was. ``results`` is therefore EVERY result in scope, faulted and failed ones included — the walk
    does the excluding, per measure, so a cell, a bar, a run summary and a divergence lens reporting
    one measure name report it over one population. A measure that declares none is computed over
    ``undeclared``, the population of the surface asking: ``scored`` for the decision surface's cells
    and bars, ``all_observed`` for a run's summary and the rollups, which is what each of those always
    computed — except a cost or latency measure, which every surface reads over ``delivered``
    (:func:`~threetears.evals.kernel.metrics.summary_population`): a refused call is a failure every
    rate counts against its arm, and not a 50 ms turn costing nothing on any of them.

    The second return value maps every pooled measure to its **observation unit**:
    ``'result'`` for the result's own scalars, its open maps, and the leaves of any single
    sub-model — at most one observation each — against one element of a repeatable LIST
    record (``'usage[]'``, ``'async_deliveries[]'``), where a result contributes as many
    observations as it carried elements. Two means over the same unit may be differenced; a
    per-result mean and a per-delivery mean may not, and nothing in the values themselves
    says which is which (both are honestly in the measure's own unit). Only the walk knows,
    so the walk is what names it — a count-based guess would misread the ordinary case of a
    measure simply missing from one result.

    A name observed at BOTH levels takes the outer unit, for the same reason the outer
    values win: a per-role ``cost_usd`` row decomposes the result-level ``cost_usd``, and
    what the collection publishes for that name is the result-level measure. That
    precedence is why the unit cannot be a property of the measure REGISTRY — one
    descriptor, keyed by name, would have to answer for both levels at once.

    Two levels are walked and merged with **the outermost winning any name collision**:
    a result's own observations, then the leaves of the carriers it holds. The
    precedence matters for exactly one real case — a per-role ``cost_usd`` row
    decomposes the result-level ``cost_usd`` rather than independently observing it, so
    pooling both would count the same spend twice at two different granularities.

    Every scope in the registry's vocabulary that ends up with no measure is named in
    ``absent_scopes`` instead of being silently missing, and an observation the walk reached
    but could not summarise is named in ``unreported_observations`` — see that field for why
    silence there was the dangerous case.

    The third return value is the pooled observations each summary was computed from, beside their
    cases, for a lens that tests between levels over per-case values rather than reading summaries.
    """
    # Each observation is pooled beside its test case. `n` alone cannot distinguish 15 independent
    # observations from 5 cases repeated 3 times, and the two license very different intervals —
    # pooling k repeats as independent draws narrows every interval by roughly sqrt(k). So the
    # summary computes its spread over the cases and counts them, from the very observations it
    # pooled: a name's two levels never contribute cases to each other.
    outer: dict[str, _PooledMeasure] = {}
    inner: dict[str, _PooledMeasure] = {}
    unreported: set[str] = set()
    carriers_by_name: dict[str, set[str]] = {}
    inner_units: dict[str, str] = {}

    def record(
        level: dict[str, _PooledMeasure],
        name: str,
        value: float | str,
        descriptor: MetricDescriptor,
        *,
        report_gaps: bool,
        case_id: str,
        result: EvalResult,
        carrier: str | None = None,
    ) -> None:
        # Outside its population before anything else: a faulted result is not an observation of a
        # `scored` or `delivered` measure at all, nor a failure that took no turn one of a `delivered`
        # measure, so it can neither contribute a value nor be reported as one lost.
        if not in_population(summary_population(descriptor, undeclared), result):
            return
        if not _is_reportable(descriptor, profile.measures):
            # An undescribed NUMBER from a telemetry source is the loss worth reporting: a
            # measurement the code emits that the registry cannot explain. Three things stay
            # quiet, each for its own reason — an undescribed string (almost always an
            # identifier: a trace id, a role, a price source), a measure a deliberate filter
            # excluded (a directionless count, a judged family), and anything from the
            # result's lineage fields, where "undescribed" means "not a measure" rather than
            # "a measure nobody described".
            if report_gaps and descriptor.family is None and not isinstance(value, str):
                unreported.add(name)
            return
        if not _value_fits(descriptor, value):
            unreported.add(name)
            return
        if carrier is not None:
            carriers_by_name.setdefault(name, set()).add(carrier)
        _, values, cases = level.setdefault(name, (descriptor, [], []))
        values.append(value)
        cases.append(case_id)

    for result in results:
        case_id = result.test_case_id
        for name, value, descriptor in _lineage_leaves(result, profile=profile):
            record(outer, name, value, descriptor, report_gaps=False, case_id=case_id, result=result)
        for name, value, descriptor in _open_map_leaves(result, profile=profile):
            record(outer, name, value, descriptor, report_gaps=True, case_id=case_id, result=result)
        for name, value, report_gaps, carrier, observation_unit in chain(
            _carrier_leaves(result, profile=profile),
            _derived_leaves(result),
            _goal_check_leaves(result),
            _accuracy_leaves(result),
        ):
            record(
                inner,
                name,
                value,
                describe_measure(name, profile.measures),
                report_gaps=report_gaps,
                case_id=case_id,
                result=result,
                carrier=carrier,
            )
            inner_units[name] = observation_unit

    # Two DIFFERENT carriers using the same leaf name are two different quantities that
    # happen to share a word — an inner agent's elapsed_ms and some future subsystem's are
    # not one distribution, and averaging them would report a number describing neither,
    # under whichever descriptor's prose won the name. Refuse to pool and say so, rather
    # than producing a plausible number nobody can trace.
    ambiguous = {name for name, carriers in carriers_by_name.items() if len(carriers) > 1 and name not in outer}
    for name in ambiguous:
        inner.pop(name, None)
        unreported.add(name)

    # Outer wins outright — both the values and the metadata — so a colliding inner
    # name can never contribute half of a measure. A name that is a field of the result itself
    # is the result-level measure even where no result here observed one — an unpriced result's
    # ``cost_usd`` is None — so the per-role rows that decompose it never stand in for it: a
    # sum of the priced rows under ``cost_usd`` is exactly the partial figure unpriced spend
    # must never become.
    pooled = {**{name: entry for name, entry in inner.items() if name not in EvalResult.model_fields}, **outer}

    measures = [
        _measure_summary(*pooled[name], population=summary_population(pooled[name][0], undeclared))
        for name in sorted(pooled)
    ]
    confusion = next((measure for measure in measures if measure.name == CONFUSION_CELL_MEASURE), None)
    if confusion is not None:
        _, cells, cell_cases = pooled[CONFUSION_CELL_MEASURE]
        observations = [(str(cell), case) for cell, case in zip(cells, cell_cases)]
        measures = sorted(
            [*measures, *_classifier_label_summaries(confusion, observations)], key=lambda measure: measure.name
        )
    present = {measure.attribution_scope for measure in measures}
    collection = MeasureCollection(
        measures=measures,
        absent_scopes=[scope for scope in _ATTRIBUTION_SCOPES if scope not in present],
        unreported_observations=sorted(
            (unreported - set(pooled))
            | set(_withheld_derived(results, pooled))
            | set(_undeclarable_host_entries(results))
        ),
    )
    # Outer names describe the result itself by construction; an inner name keeps whatever
    # the walk saw carrying it, and loses to the outer level on a collision — the same
    # precedence the values follow, so a measure's unit always describes the values pooled
    # under it. Names dropped along the way (unreportable, ambiguous) are excluded, so the
    # map is exactly the collection's own vocabulary.
    units = {**inner_units, **dict.fromkeys(outer, _PER_RESULT)}
    return collection, {name: units[name] for name in pooled}, pooled


def _classifier_label_summaries(
    confusion: MeasureSummary, observations: Sequence[tuple[str, str]]
) -> list[MeasureSummary]:
    """Each label's precision, recall and F1, derived from a cell's confusion matrix.

    The matrix is the ``confusion_cell`` measure's observations — one ``expected → predicted`` pair
    each, beside its test case — so the per-label statistics are counted from what the walk already
    pooled, over the same population, never re-read from the results, and counted by
    :func:`~threetears.evals.analysis.confusion.label_statistics`, the one count the run summary reads
    too. Precision and recall are
    proportions, so each is a boolean-shaped summary: its rate, the count behind it, and its interval
    over the cases (:func:`~threetears.evals.analysis.stats.proportion_interval`). F1 is not a proportion of anything, so it is a numeric summary with a mean and no
    spread — it has none by construction, at any n. It is the harmonic mean of precision and recall, so a
    label missing either has no F1 either, rather than an F1 of 0.0 stated over no evidence; its ``n`` is
    the label's support across both — the observations predicted or expected as it
    (``predicted + expected - hits``), which is what its value is computed over.

    Args:
        confusion: The ``confusion_cell`` summary.
        observations: The ``(confusion_cell, test_case_id)`` observations it summarises.

    Returns:
        The derived summaries, named by :func:`~threetears.evals.kernel.metrics.classifier_label_measure`.
        A label never predicted has no precision; one never expected has no recall; either has no F1.
    """
    derived: list[MeasureSummary] = []
    for statistics in label_statistics(observations):
        rates: tuple[tuple[ClassifierStatistic, int, int, float | None, tuple[float, float] | None], ...] = (
            (
                "precision",
                statistics.predicted,
                statistics.predicted_cases,
                statistics.precision,
                statistics.precision_interval,
            ),
            ("recall", statistics.expected, statistics.expected_cases, statistics.recall, statistics.recall_interval),
        )
        for statistic, n, cases, rate, interval in rates:
            if rate is not None:
                derived.append(
                    MeasureSummary(
                        name=classifier_label_measure(statistic, statistics.label),
                        attribution_scope=confusion.attribution_scope,
                        higher_is_better=True,
                        population=confusion.population,
                        n=n,
                        n_independent=cases,
                        rate=rate,
                        n_true=statistics.correct,
                        ci_low=None if interval is None else interval[0],
                        ci_high=None if interval is None else interval[1],
                    )
                )
        if statistics.f1 is not None:
            derived.append(
                MeasureSummary(
                    name=classifier_label_measure("f1", statistics.label),
                    attribution_scope=confusion.attribution_scope,
                    higher_is_better=True,
                    population=confusion.population,
                    n=statistics.predicted + statistics.expected - statistics.correct,
                    mean=statistics.f1,
                )
            )
    return derived


def _measure_summary(
    descriptor: MetricDescriptor,
    values: list[float | str],
    cases: list[str],
    *,
    population: MeasurePopulation,
) -> MeasureSummary:
    """Summarise one measure's observations in the shape its data type takes.

    A spread is computed over the test cases, never over the observations as if each were its own
    draw: the SEM is :func:`~threetears.evals.analysis.stats.clustered_standard_error`, and the interval
    is read on ``n_independent - 1`` degrees of freedom. Where every case was observed once, both are
    the unclustered forms exactly.

    Args:
        descriptor: The measure's registry descriptor, carried onto the summary.
        values: Its observations, at least one.
        cases: Each observation's test case, aligned with ``values``.
        population: The population those observations were drawn from, stated on the summary.

    Returns:
        A categorical summary (counts), a boolean one (rate + interval), a text one (every
        observation listed, nothing aggregated) or a numeric one (distribution + SEM + interval).
    """
    shape: dict[str, Any]
    if descriptor.data_type == "text":
        shape = {"texts": [str(value) for value in values]}
    elif descriptor.data_type == "boolean":
        n_true = sum(1 for value in values if value is True)
        interval = proportion_interval([value is True for value in values], cases)
        shape = {
            "rate": n_true / len(values),
            "n_true": n_true,
            "ci_low": None if interval is None else interval[0],
            "ci_high": None if interval is None else interval[1],
            "case_means": small_sample_case_means([1.0 if value is True else 0.0 for value in values], cases),
        }
    elif descriptor.data_type == "categorical":
        counts: dict[str, int] = {}
        for value in values:
            counts[str(value)] = counts.get(str(value), 0) + 1
        shape = {"categories": counts}
    else:
        observed = [float(value) for value in values]
        numeric = sorted(observed)
        mean = sum(numeric) / len(numeric)
        sem = clustered_standard_error(observed, cases)
        interval = observed_mean_interval(
            observed, cases=cases, value_range=descriptor.value_range, floor=descriptor.interval_floor
        )
        shape = {
            "mean": mean,
            "p05": _percentile(numeric, 0.05),
            "p50": _percentile(numeric, 0.50),
            "p95": _percentile(numeric, 0.95),
            "max": numeric[-1],
            "sem": sem,
            "n_zero": sum(1 for value in numeric if value == 0.0),
            # An interval on the MEAN at `stats.INTERVAL_LEVEL`, carried structurally so the generator never
            # has to derive one. The viz contract requires a `ci` on the distribution
            # and null_result payloads while forbidding the generator from inventing a
            # statistic the bundle does not give it — with only `sem` here, both of
            # those payloads were unfillable, so the findings that most need them (a
            # null result draws the arms' intervals, and a distribution its spread)
            # silently came back with no visualization at all. Interval of the mean,
            # not of the observations: it
            # answers "where does this arm's average sit", which is the question a
            # null result asks. Over the cases, not the observations, so k repeats of a case are not
            # k draws. None below n=2, or over a single case, where `sem` itself is unestimable. Inside
            # the measure's declared scale, and a 0/1 measure's is its proportion's interval — one rule,
            # `stats.observed_mean_interval`, so `accuracy` and the `match` it is derived from agree.
            "ci_low": None if interval is None else interval[0],
            "ci_high": None if interval is None else interval[1],
            # Below the band floor a chart draws the cases rather than the interval, so it needs them.
            "case_means": small_sample_case_means(observed, cases),
        }
    return MeasureSummary(
        name=descriptor.name,
        attribution_scope=descriptor.attribution_scope,
        higher_is_better=descriptor.higher_is_better,
        population=population,
        n=len(values),
        n_independent=len(set(cases)),
        **shape,
    )


#: What the runs showed about a resolved surface's movement, over one cohort.
#:
#: ``explained`` — the surface moved only because the knob written into it did, so it is the same
#: change seen twice: for an open family, every run's residual (the surface with the swept members
#: taken back out) agrees; for a fixed lever, the surface held one level within each of the lever's
#: levels, and some level was held by two or more arms, so the runs could have shown otherwise.
#: ``unverified`` — a fixed lever's fold that the runs could NOT have refuted: the surface held one
#: level within each of the lever's, but every level was held by one arm only, and every run of an
#: arm resolves the arm's one surface (the surface is in the variant key), so the dependency holds by
#: construction. The surface is still folded — the knob names the arm — and every lens that folds it
#: marks the comparison with an ``unverified_fold`` confound, because a check that could not run is
#: not a pass. ``unexplained`` — something besides the knob changed it: the residuals disagree, two
#: runs that held the lever at one level carried different surfaces, or a run the lever does not apply
#: to (another kind's) carried a surface no run of the lever's own kind in the cohort carries. ``undetermined`` — some run's
#: residual or surface could not be read, so neither can be shown; the surface is kept, because
#: folding it would be an inference.
SurfaceFold = Literal["explained", "unverified", "unexplained", "undetermined"]

#: The verdicts under which a surface is folded into its knob: one checked, one that could not be.
_FOLDED: frozenset[SurfaceFold] = frozenset({"explained", "unverified"})


class _SurfaceFolds:
    """The ONE answer to "did this resolved surface move on its own, across these runs".

    A host may register a knob AND the surface it is merged into as levers
    (:attr:`~threetears.evals.kernel.host.sweepables.Sweepable.resolves_into`): an open family's
    members and the tool configuration they are written into, or a kind's ``reasoning_effort``
    overlay and the resolved model parameters it is written into
    (:class:`~threetears.evals.kernel.host.kinds.ResolvesInto`). One turn of the knob then reaches
    every lens twice — as the knob and as the surface's content hash — and a lens that counted both
    reported a one-knob arm as ``multi_factor`` and each lever as confounded by the other. Every lens
    that decides what a comparison moved or what confounds it asks this object, over the cohort it is
    comparing, so two lenses over one cohort cannot come to different answers about one surface. Lenses
    over different cohorts can: a knob's coverage row pools every arm that moved the knob, while a design
    contrast reads only the arms its own departures cover, so each answer is about the runs it names.

    **Two rules, one per kind of knob, because only one of them has anything to take out.** An open
    family's members are names a host can remove from its surface, so the family's own residual
    reader answers. A fixed lever is one value with nothing to remove, and the surface without it is
    not something the engine could ask for — so the runs answer instead, by functional dependency:
    the surface folds where every level of the lever carries one level of the surface across the
    cohort, which is what "it moved only where the knob did" means when nothing else can be read.
    **What that cannot see:** the surface is in the variant key, so every run of one arm resolves
    one surface, and repeats of an arm can never disagree with it. A cohort with one ARM per level of
    the lever therefore satisfies the rule by construction, and a second change that rode in exactly
    where the knob changed folds with it. Only a level of the lever held by two or more arms can show
    anything else wrote into the surface — which is exactly the shape a sweep that changed something
    besides the knob produces, and the shape the rule keeps as a confound. A fold no such level
    tested is ``unverified``: still folded, so the knob names the arm, and marked on every comparison
    that folds it (:data:`UNVERIFIED_FOLD_PREFIX`), so it is never read as a checked non-confound.

    **A fixed lever's level is the level its variant coordinate carries**
    (:meth:`~threetears.evals.kernel.host.profile.HostProfile.engine_levels` and the host's
    variant-lever reader), so the fold and the variant key cannot disagree about whether two runs sat
    at one level — in particular, a run of another kind sits at that kind's "not this kind" level,
    never at a ``None`` a run of the lever's own kind can also hold. The lever does not apply to such
    a run and cannot have written its surface, so where such a run carries a surface no run of the
    lever's own kind in the cohort carries, the kind's change moved it and nothing is folded.

    **Per cohort, never campaign-wide**, because the answer depends on which runs are compared. A
    surface can be explained across the whole campaign — every member any run named taken out — and
    unexplained inside a contrast that swept only one of them, where a second key moved with nothing
    naming it. That second key is exactly what the rule exists to surface.

    Built once per assembly and memoised per ``(run, removed members)``, because a residual read
    is host code over the run's own payload and every lever × cohort asks.
    """

    def __init__(
        self, runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
    ) -> None:
        """Capture the campaign's runs, which members each family resolved on each, and each fixed knob's level.

        Args:
            runs: The campaign's resolved runs.
            results_by_run: Each run's results, keyed by run id.
            profile: The host whose vocabulary this reads.
        """
        self._registry = profile.sweepables
        self._surfaces = self._registry.resolution_surfaces
        self._runs = {run.id: run for run in runs}
        self._results = results_by_run
        resolutions = (
            {run.id: self._registry.resolve_levers(run, results_by_run.get(run.id, [])) for run in runs}
            if self._surfaces
            else {}
        )
        self._members: dict[str, dict[str, frozenset[str]]] = {
            run_id: dict(resolution.members_by_family) for run_id, resolution in resolutions.items()
        }
        self._member_values: dict[str, dict[str, Any]] = {
            run_id: {member: resolution.values.get(member) for member in resolution.overlaid}
            for run_id, resolution in resolutions.items()
        }
        # A fixed knob's level, as the variant key carries it, and its surface's raw value, both as
        # comparable keys. The knob is read off the variant coordinate so a run of another kind sits at
        # that kind's own level, apart from any value a run of the knob's kind holds. The surface is read
        # off the resolution, because its ``None`` is a run that did not record it, and that must stay
        # ``None`` so it can only ever read as "cannot say".
        claimants = {
            surface: claimant.name for surface, claimant in self._surfaces.items() if claimant.open_family is None
        }
        self._fixed_levels: dict[str, dict[str, str | None]] = {}
        self._inapplicable: dict[str, frozenset[str]] = {}
        for run in runs if claimants else ():
            coordinates = {
                **profile.engine_levels(run),
                **(profile.variant_levers(run) if profile.variant_levers is not None else {}),
            }
            values = resolutions[run.id].values
            levels: dict[str, str | None] = {}
            inapplicable: set[str] = set()
            for surface, knob in claimants.items():
                raw = values.get(surface)
                levels[surface] = None if raw is None else canonical_json(raw)
                coordinate = coordinates.get(knob)
                if coordinate is None:
                    levels[knob] = canonical_json(values.get(knob))
                else:
                    levels[knob] = coordinate.content_hash
                    if coordinate.not_of_kind is not None:
                        inapplicable.add(knob)
            self._fixed_levels[run.id] = levels
            self._inapplicable[run.id] = frozenset(inapplicable)
        # Which arm each run measured: a fixed knob's fold is testable only where one of its levels
        # was held by two or more arms. A run no observation keys stands for an arm of its own.
        self._arm_of: dict[str, str] = {
            run.id: variant_key_of_run(results_by_run.get(run.id, [])) or f"run:{run.id}" for run in runs
        }
        self._residuals: dict[tuple[str, str, frozenset[str]], str | None] = {}

    @property
    def surfaces(self) -> frozenset[str]:
        """Every lever name a knob resolves into — the only names this object ever folds."""
        return frozenset(self._surfaces)

    def fold(self, surface: str, cohort_run_ids: Collection[str]) -> SurfaceFold:
        """Decide whether ``surface`` moved on its own across a cohort.

        Args:
            surface: A name in :attr:`surfaces`.
            cohort_run_ids: The runs under comparison.

        Returns:
            The verdict — see :data:`SurfaceFold`.
        """
        claimant = self._surfaces[surface]
        cohort = [run_id for run_id in dict.fromkeys(cohort_run_ids) if run_id in self._runs]
        if claimant.open_family is None:
            return self._fold_by_level(surface, claimant.name, cohort)
        swept = frozenset().union(*(self._members.get(run_id, {}).get(claimant.name, frozenset()) for run_id in cohort))
        residuals = [self._residual(surface, run_id, swept) for run_id in cohort]
        if any(residual is None for residual in residuals):
            return "undetermined"
        return "explained" if len(set(residuals)) <= 1 else "unexplained"

    def folds_away(self, lever: str, cohort_run_ids: Collection[str]) -> bool:
        """True when ``lever`` is a resolved surface folded into its knob across the cohort.

        The question every call site actually asks, so none of them re-derives it from
        :meth:`fold` with a comparison that could drift. ``unverified`` folds as well as
        ``explained``; the comparison is then marked by :func:`_uncontrolled_dimensions`.

        Args:
            lever: Any lever name.
            cohort_run_ids: The runs under comparison.

        Returns:
            True only for a surface the cohort shows moved with its knob alone, checked or not; every
            other lever is False.
        """
        return lever in self._surfaces and self.fold(lever, cohort_run_ids) in _FOLDED

    def folds_away_in_contrast(
        self, lever: str, pair: Collection[str], contrast_cohort: Callable[[str], Collection[str]]
    ) -> bool:
        """:meth:`folds_away` for one contrast against the control, over the cohort that can decide it.

        An open family's residual is a check two runs can make, so a contrast asks it over its own
        pair. A fixed knob's fold is not: over two runs it reduces to "did the knob also move", which
        can never come out ``unexplained``, so the design would fold a surface every other lens calls a
        varying confound. It is decided over the contrast's cohort instead — the control and every
        contrast whose departures, the surface aside, fall within this one's — where a second arm at
        one of the knob's levels can show the surface moving on its own, and an arm that moved
        something else cannot make this contrast's knob answer for it.

        Args:
            lever: Any lever name.
            pair: The control's run and the contrast's.
            contrast_cohort: The contrast's cohort for a surface, asked only for a fixed knob's.

        Returns:
            Whether the contrast's movement of ``lever`` is its knob's, seen a second time.
        """
        if lever not in self._surfaces:
            return False
        fixed = self._surfaces[lever].open_family is None
        return self.folds_away(lever, contrast_cohort(lever) if fixed else pair)

    def swept_members(self, surface: str, run_ids: Collection[str]) -> dict[str, Any]:
        """The members of ``surface``'s family these runs overlaid, with the value each named.

        What an arm is named by once its surface folds away. The runs are one arm's: they share a
        variant key, so they resolved one surface, and a member one of them names while another
        inherits it resolved to the same value in both — which is why the union is the arm's, and
        why two of them cannot name one member at two values.

        A surface a FIXED lever is written into has no members: the lever names the arm under its
        own name, as the coordinate it carries in the arm's ``levers`` (registration refuses one
        that carries none), so nothing is added beside it.

        Args:
            surface: A name in :attr:`surfaces`.
            run_ids: The runs that carried one arm.

        Returns:
            Member name → the value named, sorted by name; empty when none of them overlaid a member,
            and for a surface a fixed lever is written into.
        """
        claimant = self._surfaces[surface]
        if claimant.open_family is None:
            return {}
        named: dict[str, Any] = {}
        for run_id in dict.fromkeys(run_ids):
            values = self._member_values.get(run_id, {})
            for member in self._members.get(run_id, {}).get(claimant.name, frozenset()):
                named[member] = values.get(member)
        return dict(sorted(named.items()))

    def _fold_by_level(self, surface: str, lever: str, cohort: list[str]) -> SurfaceFold:
        """Whether ``surface`` held one level within each of ``lever``'s levels across the cohort.

        Args:
            surface: The surface a fixed lever is written into.
            lever: That lever.
            cohort: The runs under comparison, deduplicated, each one this object holds.

        Returns:
            ``undetermined`` when some run did not record the surface; ``unexplained`` when two runs at
            one of the lever's levels carried different surfaces, or a run the lever does not apply to
            carried a surface no run of its own kind here carries; otherwise ``explained`` when some level of the
            lever was held by two or more arms, and ``unverified`` when none was.
        """
        levels = [self._fixed_levels.get(run_id, {}) for run_id in cohort]
        if any(level.get(surface) is None for level in levels):
            return "undetermined"
        own_kind = {
            level.get(surface)
            for run_id, level in zip(cohort, levels, strict=True)
            if lever not in self._inapplicable.get(run_id, frozenset())
        }
        if any(
            level.get(surface) not in own_kind
            for run_id, level in zip(cohort, levels, strict=True)
            if lever in self._inapplicable.get(run_id, frozenset())
        ):
            # A run of another kind carries the knob's "not this kind" level, so the knob did not write its
            # surface. Where that surface is one no run of the knob's own kind carries here, the kind's
            # change moved it, and the knob cannot answer for it.
            return "unexplained"
        surface_at: dict[str | None, str | None] = {}
        arms_at: dict[str | None, set[str]] = defaultdict(set)
        for run_id, level in zip(cohort, levels, strict=True):
            if surface_at.setdefault(level.get(lever), level.get(surface)) != level.get(surface):
                return "unexplained"
            arms_at[level.get(lever)].add(self._arm_of.get(run_id, f"run:{run_id}"))
        return "explained" if any(len(arms) > 1 for arms in arms_at.values()) else "unverified"

    def _residual(self, surface: str, run_id: str, removed: frozenset[str]) -> str | None:
        """One run's residual, as a comparable key, memoised.

        Args:
            surface: The surface to read.
            run_id: The run to read it off.
            removed: The family members to take out.

        Returns:
            The residual's canonical JSON, or None when the run does not carry the surface.
        """
        key = (surface, run_id, removed)
        if key not in self._residuals:
            content = self._registry.read_residual(surface, self._runs[run_id], self._results.get(run_id, []), removed)
            self._residuals[key] = None if content is None else canonical_json(content)
        return self._residuals[key]


class _CampaignArms(NamedTuple):
    """The campaign's runs grouped by the arm each measured — the ONE grouping every lens reads.

    A run carries exactly one arm (assembly refuses a member carrying several models), so a run's
    variant key is a property of the whole run and the lenses group whole runs by it.

    Attributes:
        keyed: Variant key → its runs, arms in order of their first run and runs in bundle order.
        unplaced: Runs none of whose observations resolved a variant key, in bundle order.
    """

    keyed: dict[str, list[EvalRun]]
    unplaced: list[EvalRun]

    def groups(self) -> dict[str, list[EvalRun]]:
        """Every arm, plus each unplaced run as a group of its own under ``run:<id>``.

        An unplaced run cannot pool with any other: treating two of them as one arm would invent a
        repeat nobody ran.

        Returns:
            Group key → its runs.
        """
        return {**self.keyed, **{f"run:{run.id}": [run] for run in self.unplaced}}


def _campaign_arms(
    runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
) -> _CampaignArms:
    """Group the campaign's runs by the arm each measured.

    Args:
        runs: The campaign's resolved runs, in bundle order.
        results_by_run: Each run's results, keyed by run id — where a variant key resolves.
        profile: The host whose vocabulary this reads.

    Returns:
        The grouping.
    """
    keyed: dict[str, list[EvalRun]] = {}
    unplaced: list[EvalRun] = []
    for run in runs:
        key = variant_key_of_run(results_by_run.get(run.id, []))
        if key is not None:
            keyed.setdefault(key, []).append(run)
        else:
            unplaced.append(run)
    return _CampaignArms(keyed=keyed, unplaced=unplaced)


def _arm_repeats(members: list[EvalRun]) -> int:
    """How many times an arm's runs repeated each of its cases, at the case repeated least.

    Per CASE, never a sum over runs: two runs of one arm over the same cases are that arm at twice
    the repeats, while two over different cases are each case once — summing their ``k_runs`` would
    claim replication that never happened. A run that recorded no case set is its own unnamed set,
    which no other run can be shown to share.

    Read off each run's planned ``k_runs`` and case set, as the coverage map's ``k`` always has been;
    a run that delivered fewer results than it planned is disclosed by ``short_runs``, not here.

    Args:
        members: The arm's runs.

    Returns:
        The minimum, over every case any member ran, of the ``k_runs`` summed across the members that ran it;
        0 for no members.
    """
    per_case: dict[str, int] = defaultdict(int)
    for run in members:
        for case in run.test_case_ids or [f"unrecorded:{run.id}"]:
            per_case[case] += run.k_runs
    return min(per_case.values(), default=0)


def _campaign_design(
    runs: list[EvalRun],
    control_variant: str | None,
    *,
    results_by_run: dict[str, list[EvalResult]],
    arms: _CampaignArms | None = None,
    archived_control_variants: Collection[str] = (),
    has_unresolved_members: bool = False,
    folds: _SurfaceFolds | None = None,
    profile: HostProfile,
) -> RealizedDesign:
    """Derive the campaign's design from its arms and the DECLARED control variant.

    An arm's moved levers are the keys where its EFFECTIVE configuration differs from the control
    arm's, plus ``model`` when the candidate differs — the same notion of "lever" the rest of this
    module uses, read against one reference arm instead of pooled across all of them. Effective,
    not merely declared: comparing launch overlays alone reports a lever as moved whenever one arm
    names a value the other inherits, which is the normal shape of a sweep against an un-overridden
    control.

    **Arms, not runs.** Every run of one arm carries the same configuration, so an arm is read off
    its first run and its other runs are its repeats. A re-run of the control therefore adds to the
    control's ``k`` rather than appearing as a contrast that moved nothing.

    Args:
        runs: The campaign's resolved runs.
        control_variant: The declared control variant key, or None.
        results_by_run: Each run's results, keyed by run id — the observation side of
            effective-config resolution, and where a variant key resolves.
        arms: :func:`_campaign_arms`'s answer, when the caller already has it; derived otherwise.
        archived_control_variants: The variant keys carried by member runs an operator archived.
            Supplied by the caller because archived runs are held out of every lens and their
            results are not otherwise read; it is what tells a deliberate exclusion apart from a
            variant nobody ever ran.
        has_unresolved_members: Whether any member run failed to load at all. Such a run's
            observations cannot be read, so whether it carried the control is unknowable — and
            reporting that as "never run" is the conflation the ``unresolved`` arm exists to
            prevent. A membership outliving its run is a recurring state in a real store, not a hypothetical.
        folds: The campaign's :class:`_SurfaceFolds`, shared with the other lenses when the caller
            has one; built from ``runs`` and ``results_by_run`` otherwise. A resolved surface whose
            movement between the control and a contrast is its swept members' is not a second moved
            lever — see :class:`_SurfaceFolds`.
        profile: The host whose vocabulary this reads.

    Returns:
        The design. ``shape`` is ``undesignated`` whenever no control resolved. With no contrasts
        it is vacuously ``one_factor_at_a_time`` — nothing is being compared, and the empty
        ``contrasts`` list is what a reader acts on, not the shape word.
    """
    grouping = arms if arms is not None else _campaign_arms(runs, results_by_run, profile=profile)
    arms_by_key = grouping.keyed
    unplaced = [run.id for run in grouping.unplaced]

    def arm(key: str, moved: dict[str, str] | None = None) -> DesignArm:
        members = arms_by_key[key]
        return DesignArm(
            variant_key=key, run_ids=[run.id for run in members], k=_arm_repeats(members), moved=moved or {}
        )

    if not control_variant or control_variant not in arms_by_key:
        excluded: Literal["archived", "unresolved", "unobserved"] | None = None
        if control_variant:
            if control_variant in set(archived_control_variants):
                # Certain, and the most actionable: a carrier exists and somebody removed it.
                excluded = "archived"
            elif has_unresolved_members:
                # NOT certain, and that is the answer. An unloadable member's observations
                # cannot be keyed, so "nothing carries it" is unproven — claiming `unobserved`
                # here sends an operator to re-run an experiment that may already have run.
                excluded = "unresolved"
            else:
                excluded = "unobserved"
        return RealizedDesign(control_excluded=excluded, unplaced_run_ids=unplaced)

    folds = folds if folds is not None else _SurfaceFolds(runs, results_by_run, profile=profile)
    control = arms_by_key[control_variant][0]
    control_overlays = _effective_values(control, results_by_run.get(control.id, []), profile=profile)
    control_model = control.candidate_model
    # Every contrast's departures from the control BEFORE any surface is folded: a fixed knob's fold is
    # decided over the contrasts whose departures fall within this one's, so they are needed for all of
    # them first. One run stands for each arm: its runs share a variant key, so they share a configuration.
    #
    # KNOWN LIMIT: this maps both "the lever did not apply" and "it applied but resolved
    # ambiguously" to the inherited-default level, so an ambiguous control reads as a
    # moved lever. That surfaces as a claim a reader can check against
    # `RunSummary.config_provenance`, rather than as a silently dropped comparison, which
    # is why it is accepted here. `_lever_levels` keeps the two apart because a cohort a
    # run cannot support is worse than a contrast it cannot make.
    #
    # The candidate model is read off the run's declared model rather than the effective
    # resolution, which reports a run whose candidate role left no usage row as carrying no
    # model at all — and that absence would read as the arm having moved its model to the
    # inherited default.
    departures: dict[str, dict[str, str]] = {}
    for key, members in arms_by_key.items():
        if key == control_variant:
            continue
        first = members[0]
        overlays = _effective_values(first, results_by_run.get(first.id, []), profile=profile)
        departed = {
            lever: overlays.get(lever) or _INHERITED_DEFAULT_LEVEL
            for lever in sorted((set(overlays) | set(control_overlays)) - {_CANDIDATE_MODEL_LEVER})
            if (overlays.get(lever) or _INHERITED_DEFAULT_LEVEL)
            != (control_overlays.get(lever) or _INHERITED_DEFAULT_LEVEL)
        }
        if first.candidate_model != control_model:
            departed[_CANDIDATE_MODEL_LEVER] = first.candidate_model
        departures[key] = departed
    control_runs = [run.id for run in arms_by_key[control_variant]]

    def contrast_cohort(surface: str, key: str) -> list[str]:
        """The runs a fixed knob's fold on ``surface`` is decided over for the contrast ``key``.

        The control and every contrast whose departures, ``surface`` aside, fall within this one's —
        this contrast itself, a second arm at one of its knob's levels, an arm that moved less. Those
        are the arms that can show the surface moving apart from the knob WITHIN this comparison. An
        arm that moved something this contrast did not (another lever, another kind) is a different
        comparison, and its surface moving would be blamed on this contrast's knob otherwise.
        """
        own = set(departures[key]) - {surface}
        return control_runs + [
            run.id
            for other, departed in departures.items()
            if set(departed) - {surface} <= own
            for run in arms_by_key[other]
        ]

    contrasts: list[DesignArm] = []
    for key, departed in departures.items():
        # A resolved surface is left out of `moved` only where it moved with its knob alone — the
        # pair's residuals agree, or, for a fixed knob, the contrasts that moved nothing beyond this
        # one show the surface held one level within each of the knob's (a pair alone cannot show
        # that: see `folds_away_in_contrast`). Then its new hash is the knob it was written from,
        # counted a second time, and keeping it would make every one-knob arm `multi_factor`. Where
        # it moved otherwise it stays, because the arm really did move something no swept knob names.
        pair = (control.id, arms_by_key[key][0].id)
        moved = {
            lever: level
            for lever, level in departed.items()
            if not folds.folds_away_in_contrast(lever, pair, partial(contrast_cohort, key=key))
        }
        contrasts.append(arm(key, moved))

    shape: Literal["one_factor_at_a_time", "multi_factor"] = (
        "one_factor_at_a_time" if all(len(c.moved) <= 1 for c in contrasts) else "multi_factor"
    )
    return RealizedDesign(control_arm=arm(control_variant), contrasts=contrasts, unplaced_run_ids=unplaced, shape=shape)


def _is_control_referenced(lever: str, design: RealizedDesign) -> bool:
    """True when the design makes this lever a contrast against the control.

    The single answer to "is this row a contrast or a marginal comparison", read by both the
    cohort builder and the label that describes what it built. Two derivations would be free
    to disagree, and one way to get there is tempting and wrong: inferring the answer from
    whether the cohort came out smaller than the campaign. Those coincide only by accident —
    a single-lever star, where every arm moved the one lever, has a cohort spanning every run
    AND is exactly the control-referenced contrast the campaign exists to draw.

    Args:
        lever: The lever under comparison.
        design: The derived design.

    Returns:
        True when a control resolved and at least one arm moved this lever off it.
    """
    return design.control_arm is not None and any(lever in c.moved for c in design.contrasts)


def _lever_cohort(lever: str, design: RealizedDesign, all_run_ids: list[str]) -> list[str]:
    """The runs a comparison on ``lever`` is actually drawn from.

    Without a control every run is in every lever's cohort, because nothing says which runs
    were meant to be read together — that is the marginal comparison the coverage map has
    always reported, and it is honest about being one.

    With a control, an arm that moved a DIFFERENT lever belongs to a different contrast and is
    not evidence about this one. Pooling it in is what invents a confound: the comparison's side
    at the control's level of this lever then also contains that arm's own moved lever at two
    values, and the scan correctly reports a difference the comparison never had.

    **A lever no arm moved is not narrowed at all**: every arm holds it at the control's level, so
    there is no contrast to narrow to, and the whole campaign is the honest cohort — which is what
    it was before any control was designated.

    Args:
        lever: The lever under comparison.
        design: The derived design.
        all_run_ids: Every resolved run id, in bundle order.

    Returns:
        The cohort's run ids in ``all_run_ids`` order: every run of the control arm, of each arm that
        moved this lever, and of any arm that moved no lever at all. A run that measured no keyable
        arm belongs to no contrast and is left out.
    """
    if not _is_control_referenced(lever, design) or design.control_arm is None:
        return all_run_ids
    keep = set(design.control_arm.run_ids)
    for contrast in design.contrasts:
        if lever in contrast.moved or not contrast.moved:
            keep |= set(contrast.run_ids)
    return [run_id for run_id in all_run_ids if run_id in keep]


def _lever_levels(
    runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
) -> dict[str, dict[str, list[str]]]:
    """Map each swept lever to its levels, and each level to the runs that sat at it.

    A lever is any axis a run's EFFECTIVE configuration names — which is whatever the
    host declares, under the host's own names — plus the candidate ``model`` when it actually
    varied. The same set the coverage map draws from, before that map narrows it to the
    levers the campaign engaged with (:func:`_reportable_levers`); this lens needs the wider set,
    because a confound is a dimension that moved whether or not anyone was comparing on it.

    **Effective, not declared — but absence still means something, and the two cases are
    different.** For a lever whose value is RECOVERABLE (see :func:`_observed_model_levers`),
    binning an un-named run at ``'—'`` reads absence as a level of its own and splits one
    cohort in two whenever a run inherits the very value another run names — the ordinary
    shape of a sweep against an un-overridden control. Those runs are pooled on the value
    that ran.

    For a lever that is NOT recoverable — a round budget, a token budget, anything no
    record pins — ``'—'`` is retained and is a real cohort: "ran at the subject's own
    setting", which every un-overriding run shares. Dropping those runs instead would delete
    the comparison, leaving a two-arm sweep with one level and no divergence to find.

    Only a lever that APPLIED and whose value is genuinely ambiguous (two models observed
    under one role) leaves a run out: there, no level can be claimed without inventing one.

    Levers observed at a single level are dropped: there is nothing to compare, so they can
    produce no divergence.

    **The candidate model is binned on the run's DECLARED model, not on its effective value**,
    which is the one place this lens deliberately parts from the resolution above: a run whose
    candidate role left no usage row resolves no model, and dropping it would delete a level the
    run plainly ran. A run carries exactly one model, so binning whole runs bins whole arms.

    Args:
        runs: The campaign's resolved runs.
        results_by_run: Each run's results, keyed by run id — the observation side of
            effective-config resolution.
        profile: The host whose vocabulary this reads.

    Returns:
        ``{lever: {level: [run_id, ...]}}`` for every lever with at least two levels.
    """
    levels: dict[str, dict[str, list[str]]] = {}
    configs = {run.id: _effective_config(run, results_by_run.get(run.id, []), profile=profile) for run in runs}
    lever_names = {key for flat in configs.values() for key in flat}
    # Observation is not the only way the model becomes a lever: a campaign whose results
    # carry no `candidate` usage row recovers nothing, yet the declared sets still differ.
    # `lever_names` is a set, so the two sources name one lever, never two.
    if len({run.candidate_model for run in runs}) > 1:
        lever_names.add(_CANDIDATE_MODEL_LEVER)
    for lever in sorted(lever_names):
        by_level: dict[str, list[str]] = {}
        for run in runs:
            if lever == _CANDIDATE_MODEL_LEVER:
                level: str = run.candidate_model
            else:
                effective = configs[run.id].get(lever)
                if effective is None:
                    # The lever does not apply to this run: it named no override and nothing
                    # recovers a value. That is the "ran at the subject's own setting" cohort,
                    # which every un-overriding run shares — dropping them would delete the
                    # contrast a sweep against an un-overridden control exists to make.
                    level = _INHERITED_DEFAULT_LEVEL
                elif effective.value is None:
                    # Applied, but ambiguous. No level can be claimed without inventing one.
                    continue
                else:
                    level = effective.value
            by_level.setdefault(level, []).append(run.id)
        if len(by_level) > 1:
            levels[lever] = by_level
    return levels


def _apparatus_levels(
    runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
) -> dict[str, dict[str, str | None]]:
    """Read each run's value for every apparatus dimension, as comparable level keys.

    The apparatus is everything a campaign is *not* tuning — the template, the judge, the
    simulated user, the cassette corpus, the subject behind it all. A swept lever moving is
    the experiment; the apparatus moving is the experiment quietly becoming a different one,
    which is why these are scanned separately from the overlays and reported distinguishably.

    Values are canonicalised to strings because the raw ones are not all hashable (two of
    these are sets) and because two structurally identical values must key the same.

    Args:
        runs: The campaign's resolved runs.
        results_by_run: Each run's results, for the dimensions observed per result.
        profile: The host whose vocabulary this reads.

    Returns:
        ``{dimension: {run_id: level_key}}`` over the host's apparatus sweepables and — under
        :func:`world_dimension_key` — its world dimensions, whose level is the placement this run
        derived for each. A ``None`` level key means the value
        cannot be compared at all — the dimension declares that a blank means "nobody
        recorded this" rather than "this is the recorded value". Which inputs those are is
        decided at the declaration, never here: a blank is a real level on some of them (an
        ad-hoc run genuinely has no template) and an absence on others, and only the
        declaration knows which. An input whose reader emits a sentinel for a recorded state
        — a run that pinned no judge config, say — never reaches this function as a blank at
        all, which is how a level that looks empty stops being read as an absence.

        A dimension these runs do not HAVE — pinned to a rig seat no kind among them fills
        (:attr:`~threetears.evals.kernel.host.kinds.KindContract.seats`) — is absent from the
        result entirely rather than present with a ``None`` level. That absence is the claim the
        confound scan already makes about a dimension that held still, and it is what keeps the
        dimension out of the apparatus class as well, since that is built from these keys.
    """
    levels: dict[str, dict[str, str | None]] = {}
    sweepables = profile.sweepables
    values_by_run = {run.id: sweepables.read_all(run, results_by_run.get(run.id, [])) for run in runs}
    apparatus = [declared for declared in sweepables.declarations if declared.role == "apparatus"]
    # ONE decision per dimension, taken over EVERY arm's value before any of them is stored.
    #
    # A dimension this host does not HAVE is omitted rather than read, so it reaches neither the
    # confound scan nor `apparatus_class_of` — whose dimension set is `set(apparatus_levels)`,
    # which is why the cell partition needs no separate rule. Omitting is the only honest answer
    # available for a blank: the declaration says a blank here is an absence, and reporting
    # "nobody recorded the simulated user" about a host that simulates nobody is a fact about the
    # engine's vocabulary rather than about the runs.
    #
    # **Across the cohort, not per run**, which is the whole of what `omits_apparatus` asks for:
    # "one arm recording a level is enough to refute the claim". Deciding inside the run loop
    # dropped a real apparatus DIFFERENCE — the arm that recorded a level was kept and the blank
    # arm was skipped, so the dimension held exactly one level, and a scan that sees one level and
    # no blank emits nothing. That is the failure the declaration-versus-data rule exists to
    # close, re-created one loop in. It also fired the contradiction warning once per run rather
    # than once per dimension.
    omitted = {
        declared.name
        for declared in apparatus
        if runs
        and profile.omits_apparatus(declared.name, [(run, values_by_run[run.id][declared.name]) for run in runs])
    }
    for run in runs:
        values = values_by_run[run.id]
        for declared in apparatus:
            if declared.name in omitted:
                continue
            # A run whose rig had no such seat reads at UNSEATED_LEVEL — a level, not an unknown — beside
            # a run that had it, so a cohort mixing judged and code-only runs of one kind does not read
            # the code-only runs' judge as undecided.
            value = profile.apparatus_level(run, declared.name, values[declared.name])
            undecided = sweepables.is_indeterminate(declared.name, value)
            levels.setdefault(declared.name, {})[run.id] = None if undecided else canonical_json(value)
        # The world this run placed the subject in, on the same axis for the same reason. A run
        # that recorded no placements reads UNDECIDED rather than as having placed nothing: the
        # record is what says what a run did with its world, and absence of a record is an
        # observation nobody made, which is the state that blocks a merge instead of faking one.
        if profile.world is not None:
            placements = run.world_placements
            for dimension in profile.world.declarations:
                key = world_dimension_key(dimension.name)
                levels.setdefault(key, {})[run.id] = None if placements is None else placements.get(dimension.name)
    return levels


def _uncontrolled_dimensions(
    lever: str,
    cohort_run_ids: list[str],
    lever_levels: dict[str, dict[str, list[str]]],
    apparatus_levels: dict[str, dict[str, str | None]],
    *,
    folds: _SurfaceFolds,
    profile: HostProfile,
) -> list[Confound]:
    """Name everything that was not held fixed across a cohort, and why each one matters.

    Grouping runs by one lever leaves every other swept lever free to vary inside the
    groups, so the comparison is marginal — averaged over whatever else moved — rather
    than controlled. That is the honest thing a campaign of this size can offer, and it is
    only misleading when it goes unsaid: a reader who assumes a clean A/B will attribute
    the whole movement to the one lever named.

    The apparatus dimensions are the half a lever scan structurally cannot see, because
    they are run attributes rather than overlays — and they are the more serious half.
    A swept lever varying is at least a knob somebody chose to turn; a template or a judge
    varying means the two arms were measured by different instruments, which no amount of
    replication fixes.

    An apparatus dimension that was never recorded on some run here is reported too, with
    ``status='undecided'``. Silence would read as "it held still", which is an observation
    nobody made — and the state is structural rather than worded into the reason, so a
    consumer never has to read prose to find out which of the three cases it is looking at.

    Args:
        lever: The lever being compared (excluded from its own confound list).
        cohort_run_ids: Every run in the cohort under comparison.
        lever_levels: The full lever → level → run-ids map.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.
        folds: The campaign's :class:`_SurfaceFolds`. A resolved surface that varied only because
            its swept members did is the lever under comparison (or its sibling members) seen a
            second time, so it is not named — and likewise a surface a fixed knob is written into,
            where it held one level within each of the knob's. One whose residual or surface could
            not be read is named as ``undecided``, because whether anything besides the swept knob
            moved is exactly what no run recorded.
        profile: The host whose vocabulary this reads.

    Returns:
        The swept levers that varied within the cohort (sorted), then the apparatus
        dimensions that varied or could not be decided (in declaration order).
    """
    cohort = set(cohort_run_ids)
    confounds: list[Confound] = []
    for other, by_level in sorted(lever_levels.items()):
        if other == lever or sum(1 for members in by_level.values() if cohort.intersection(members)) <= 1:
            continue
        if other in folds.surfaces:
            verdict = folds.fold(other, cohort_run_ids)
            if verdict == "explained":
                continue
            if verdict == "unverified":
                # Folded — the knob names the comparison — but marked, because these runs could not
                # have shown the surface moving apart from its knob, and an untested fold is no pass.
                confounds.append(Confound(dimension=f"{UNVERIFIED_FOLD_PREFIX}{other}", kind="unverified_fold"))
                continue
            if verdict == "undetermined":
                confounds.append(Confound(dimension=other, kind="swept_lever", status="undecided"))
                continue
        confounds.append(Confound(dimension=other, kind="swept_lever"))
    return confounds + _apparatus_confounds(cohort_run_ids, apparatus_levels, profile=profile)


def _apparatus_confounds(
    cohort_run_ids: Collection[str],
    apparatus_levels: dict[str, dict[str, str | None]],
    *,
    profile: HostProfile,
) -> list[Confound]:
    """Name every apparatus dimension that moved, or could not be shown to have held, in a cohort.

    The single producer of an apparatus confound. Three callers ask this question over three
    different cohorts — a lever's runs, a divergence's two arms, and the whole campaign — and
    they must answer it identically, because the same dimension reported as varied under one
    and silent under another reads to a generator as a fact about the comparison rather than
    about which run set it was asked over.

    Args:
        cohort_run_ids: The runs under comparison.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.
        profile: The host whose vocabulary this reads.

    Returns:
        One entry per dimension that varied or is undecided, **sorted by dimension**. A
        dimension that held still is absent — that absence is the claim, so it is only ever
        made about runs that recorded a value.

    Note:
        **Sorted rather than emitted in declaration order, because this list is fingerprinted.**
        ``apparatus_confounds`` and every ``coverage[].confounded_by`` are lists, ``to_dict()``
        preserves list order, and ``canonical_digest`` sorts dict keys only — so with declaration
        order the fingerprint moved whenever the registry was rearranged, over evidence that had
        not changed. A host's declaration order is a layout choice and must not be an input to an
        identity: the second host to register makes that unavoidable rather than merely untidy,
        since two hosts have no shared order to agree on.
    """
    cohort = set(cohort_run_ids)
    confounds: list[Confound] = []
    reasons = _apparatus_confound_reasons(profile=profile)
    for dimension in sorted(reasons):
        observed = {key for run_id, key in apparatus_levels.get(dimension, {}).items() if run_id in cohort}
        if None in observed:
            confounds.append(Confound(dimension=dimension, kind="apparatus", status="undecided"))
        elif len(observed) > 1:
            confounds.append(Confound(dimension=dimension, kind="apparatus"))
    return confounds


#: Measure name -> result id -> ``(case id, value)``: that result's one observation of it, and the case
#: it observed. Built once per bundle by :func:`_mechanism_observations` and read by every lens that
#: compares a mechanism across levels. The case is kept because the separation test's unit is the case.
_MechanismObservations = dict[str, dict[str, tuple[str, float]]]


def _mechanism_value(result: EvalResult, name: str, *, profile: HostProfile) -> float | None:
    """One result's observation of ``name``, read off the result itself, or None when it carries none.

    Walks the sources the measure surface walks — the result's own scalars, its covariate and host
    measure maps, its single sub-models and the measures it implies — and reads a value only where the
    result carries exactly one numeric observation of the name at the per-result unit. A row-level
    leaf (one per element of a list, such as a usage row per role) is not read: pooling one role's
    count with another's describes no mechanism. Two observations of one name is ambiguity, and an
    ambiguous result contributes nothing rather than one of its values chosen by order.

    Args:
        result: The result to read.
        name: A measure or covariate name.
        profile: The host whose measure registry resolves the name's population.

    Returns:
        The value, or None — a result outside the measure's population (a faulted one outside a ``scored``
        measure's; one that took no turn outside a ``delivered`` one's, which every cost or latency measure
        is read as unless it declares ``all_observed``), no observation, an ambiguous one, or one that is
        not a finite number. Any other measure that declares no population is read over every result here,
        as it always was.
    """
    # Over every result where the measure declares nothing, as a mechanism always read — except a turn's
    # time or spend, which every reader takes over the turns taken (`summary_population`).
    if not in_population(summary_population(describe_measure(name, profile.measures), "all_observed"), result):
        return None
    outer = [
        value
        for leaf, value, _descriptor in chain(
            _lineage_leaves(result, profile=profile), _open_map_leaves(result, profile=profile)
        )
        if leaf == name
    ]
    inner = [
        value
        for leaf, value, _gaps, _carrier, unit in chain(
            _carrier_leaves(result, profile=profile), _derived_leaves(result)
        )
        if leaf == name and unit == _PER_RESULT
    ]
    found = outer or inner
    if len(found) != 1:
        return None
    value = found[0]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _mechanism_observations(results: Sequence[EvalResult], *, profile: HostProfile) -> _MechanismObservations:
    """Every result's observation of every mechanism a lens here compares across levels.

    The names are each declared lever's ``acts_on`` and every covariate read as an observed mechanism
    (:data:`_OBSERVED_MECHANISMS`). Read once, so a coverage row, a divergence, a contrast and an arm
    reading of one measure read one set of values.

    Args:
        results: Every resolved result in the campaign.
        profile: The host whose declarations name the mechanisms.

    Returns:
        ``{name: {result id: (case id, value)}}``; a result that observed nothing of a name is absent under it.
    """
    names = {measure for _lever, measure in profile.sweepables.mechanisms}
    names |= set(_OBSERVED_MECHANISMS)
    observed: _MechanismObservations = {}
    for name in sorted(names):
        observed[name] = {
            result.id: (result.test_case_id, value)
            for result in results
            if (value := _mechanism_value(result, name, profile=profile)) is not None
        }
    return observed


def _per_case_means(values: Mapping[str, tuple[str, float]], result_ids: Collection[str]) -> dict[str, Fraction]:
    """One level's exact per-case means of one mechanism — the unit the separation test reads.

    Repeats of a case are averaged first, as :func:`_per_case_values` averages them for a family comparison,
    and averaged EXACTLY (:func:`~threetears.evals.analysis.stats.exact_decimal`): a float mean of three
    0.1s is 0.10000000000000002, so a constant measure read at three repeats a case on one level and one on
    another would differ by float noise with no spread, which the separation test counts as a gap.

    Args:
        values: Result id -> that result's ``(case id, value)``.
        result_ids: The level's result ids.

    Returns:
        Case id -> the exact mean of its results' observations; a case none of whose results observed it is
        absent.
    """
    by_case: dict[str, list[Fraction]] = defaultdict(list)
    for result_id in result_ids:
        if result_id in values:
            case_id, value = values[result_id]
            by_case[case_id].append(exact_decimal(value))
    return {
        case_id: sum(case_values, Fraction(0)) / len(case_values) for case_id, case_values in sorted(by_case.items())
    }


def _level_value(per_case: Mapping[str, Fraction]) -> Fraction:
    """A level's value of one mechanism: the exact mean of its per-case means.

    **The one derivation of "this level's value"**, read by the mechanism check, the observed-mechanism
    confound and the arm readings alike. Per case first, because the separation test runs over per-case
    means and a case repeated three times is still one case; a mean over results would weigh it thrice and
    show a different number for the same arm wherever the repeats are unequal.

    Args:
        per_case: The level's per-case means, from :func:`_per_case_means`; not empty.
    """
    return sum(per_case.values(), Fraction(0)) / len(per_case)


def _level_means(
    values: Mapping[str, tuple[str, float]], result_ids_by_level: Mapping[str, Collection[str]]
) -> dict[str, Fraction]:
    """Each level's value of one mechanism (:func:`_level_value`), for every level that observed it.

    Args:
        values: Result id -> that result's ``(case id, value)``.
        result_ids_by_level: The comparison's level -> the result ids at that level.

    Returns:
        The exact value per level that observed any; a level that observed none is absent.
    """
    return {
        level: _level_value(per_case)
        for level in sorted(result_ids_by_level)
        if (per_case := _per_case_means(values, result_ids_by_level[level]))
    }


class _LevelsSeparate(NamedTuple):
    """What :func:`_levels_separate` found over every pair of a lever's levels."""

    separated: bool
    #: Some pair had fewer than two cases on a side, or a spread that vanishes in floating point.
    untestable: bool
    #: Some pair shifted every case alike on a measure with no declared range, so no test of the mean ran on it.
    needs_range: bool


def _levels_separate(
    per_case: Mapping[str, Mapping[str, Fraction]], value_range: tuple[float, float] | None = None
) -> _LevelsSeparate:
    """Whether any pair of levels separates on per-case values, and whether any pair could not be tested.

    The engine's between-level test applied to every pair of levels
    (:func:`~threetears.evals.analysis.stats.level_difference`): paired over the cases both levels ran when
    they share at least two, else Welch's over each level's per-case values, the pairs Holm-corrected as one
    family and read against the same alpha. A gap with no spread at all (every case moved by the same nonzero
    amount, or two different constants) is read by the bounded test on the measure's declared range, and with
    none is not tested (the exact sign-flip test it once read asks about symmetry, not the mean — #597).

    Args:
        per_case: Level -> its exact per-case means, for every level that observed the measure.
        value_range: The measure's declared inclusive bounds, or None when it declares none.

    Returns:
        The :class:`_LevelsSeparate`.
    """
    raw: list[float] = []
    untestable = needs_range = False
    levels = sorted(per_case)
    for index, level_a in enumerate(levels):
        for level_b in levels[index + 1 :]:
            tested = level_difference(per_case[level_a], per_case[level_b], value_range=value_range)
            if tested.not_separated_reason is not None:
                needs_range = True
            elif tested.p_value is None:
                untestable = True
            else:
                raw.append(tested.p_value)
    separated = bool(raw) and min(holm_adjust(raw)) < SIGNIFICANCE_ALPHA
    return _LevelsSeparate(separated, untestable, needs_range)


def _mechanism_check(
    measure: str | None,
    result_ids_by_level: Mapping[str, Collection[str]],
    observations: _MechanismObservations,
    value_range: tuple[float, float] | None = None,
) -> MechanismCheck:
    """Decide whether a swept lever's declared mechanism measurably moved across its levels.

    Args:
        measure: The lever's ``acts_on``, or None when it declares none.
        result_ids_by_level: The lever's level -> the result ids at that level, over the row's cohort.
        observations: The campaign's mechanism observations.
        value_range: The mechanism measure's declared inclusive bounds, or None when it declares none.

    Returns:
        ``moved`` when some pair of observed levels separates; otherwise ``inert`` when every level observed
        the measure and every pair was tested; otherwise ``unchecked``, with the reason.
    """
    if measure is None:
        return MechanismCheck(state="unchecked", reason="not_declared")
    values = observations.get(measure, {})
    per_case = {
        level: cases
        for level in sorted(result_ids_by_level)
        if (cases := _per_case_means(values, result_ids_by_level[level]))
    }
    unobserved = [level for level in sorted(result_ids_by_level) if level not in per_case]
    state: Literal["moved", "inert", "unchecked"]
    reason: MechanismUncheckedReason | None = None
    found = _levels_separate(per_case, value_range)
    if len(result_ids_by_level) < 2:
        state, reason = "unchecked", "not_swept"
    elif found.separated:
        state = "moved"
    elif unobserved or len(per_case) < 2:
        state, reason = "unchecked", "levels_unobserved"
    elif found.untestable:
        state, reason = "unchecked", "too_few_observations"
    elif found.needs_range:
        state, reason = "unchecked", "uniform_move_needs_range"
    else:
        state = "inert"
    return MechanismCheck(
        state=state,
        measure=measure,
        level_means={level: float(_level_value(cases)) for level, cases in per_case.items()},
        level_n={level: len(cases) for level, cases in per_case.items()},
        unobserved_levels=unobserved,
        reason=reason,
    )


def _raises_observed_mechanism(lever: str, covariate: str, *, profile: HostProfile) -> bool:
    """Whether a comparison grouped by ``lever`` may name ``covariate`` as an observed-mechanism confound.

    **The one answer to that question**, asked by every lens that raises one — the coverage row, a
    divergence, and each pairwise contrast between two models — so no two of them can disagree about the
    same lever and covariate. Yes only for the candidate model, the comparison a pinned effort word fails to
    equalise; on any other lever the covariate moving is what the lever was swept to do. And never for the
    lever's own declared mechanism (``acts_on``): it moving is the lever's effect, not a rival to it.

    Args:
        lever: The lever the comparison varies.
        covariate: An observed-mechanism covariate.
        profile: The host whose declaration of the lever names its mechanism.
    """
    if lever != _CANDIDATE_MODEL_LEVER:
        return False
    return profile.sweepables.acts_on(lever) != covariate


def _observed_mechanism_confounds(
    lever: str,
    result_ids_by_level: Mapping[str, Collection[str]],
    observations: _MechanismObservations,
    *,
    profile: HostProfile,
) -> list[Confound]:
    """Name every observed mechanism whose levels' values diverged by at least its threshold.

    The third kind of confound, and the one no setting records: the levels compared ran under the same
    configuration as far as any lever says, and still did measurably different things. It qualifies the
    comparison and never suppresses it. A level that measured none of the covariate is left out of the
    comparison rather than read as zero, so a covariate nothing measured names no confound — the arm
    readings (``arm_mechanisms``) say it went unmeasured. Which covariates a lever may name at all is
    :func:`_raises_observed_mechanism`'s answer, asked here so every caller gets it.

    Args:
        lever: The lever the comparison varies.
        result_ids_by_level: The comparison's level -> the result ids at that level.
        observations: The campaign's mechanism observations.
        profile: The host whose declaration of the lever names its mechanism.

    Returns:
        One ``observed_mechanism`` confound per diverged covariate, sorted by dimension; empty for a lever
        the predicate excludes.
    """
    confounds: list[Confound] = []
    for covariate, mechanism in sorted(_OBSERVED_MECHANISMS.items()):
        if not _raises_observed_mechanism(lever, covariate, profile=profile):
            continue
        means = _level_means(observations.get(covariate, {}), result_ids_by_level)
        if len(means) < 2 or max(means.values()) - min(means.values()) < exact_decimal(mechanism.threshold):
            continue
        confounds.append(
            Confound(
                dimension=observed_mechanism_key(covariate),
                kind="observed_mechanism",
                level_values={level: float(mean) for level, mean in means.items()},
                threshold=mechanism.threshold,
            )
        )
    return confounds


def _model_contrast_confounds(
    side_a: Sequence[EvalResult],
    side_b: Sequence[EvalResult],
    observations: _MechanismObservations,
    *,
    profile: HostProfile,
) -> list[Confound]:
    """The observed-mechanism confounds of one pairwise contrast, when its two sides ran different models.

    Keyed by each side's model, so the confound names the two arms' values in the words a reader compares
    them in. A side whose results name no single model is no model contrast, and names nothing.

    Args:
        side_a: One side's results.
        side_b: The other side's results.
        observations: The campaign's mechanism observations.
        profile: The host whose declaration of the model lever names its mechanism.

    Returns:
        The confounds, or an empty list when the sides share a model or either names no single model.
    """
    models_a = {result.model for result in side_a}
    models_b = {result.model for result in side_b}
    if len(models_a) != 1 or len(models_b) != 1 or models_a == models_b:
        return []
    (model_a,), (model_b,) = models_a, models_b
    if model_a is None or model_b is None:
        return []
    return _observed_mechanism_confounds(
        _CANDIDATE_MODEL_LEVER,
        {model_a: {result.id for result in side_a}, model_b: {result.id for result in side_b}},
        observations,
        profile=profile,
    )


def _arm_production_footings(
    arms: _CampaignArms, results_by_run: Mapping[str, list[EvalResult]], *, profile: HostProfile
) -> dict[str, PooledProductionFooting]:
    """Each arm's production footing: what each of its runs set away from production (#571).

    An arm's production-replicating cost (its cells, contrasts and bars on that measure) pools its runs, and
    each run's footing is read off the host's sweepable declarations as :class:`RunSummary` reads it — so the
    arm carries every run's own, and says where they disagree, rather than one merged claim.

    Args:
        arms: The campaign's arms. A run that resolved no arm is in none.
        results_by_run: Each run's results.
        profile: The host whose declarations are read.

    Returns:
        Variant key -> the arm's pooled footing, for every arm.
    """
    return {
        variant_key: PooledProductionFooting(
            runs={
                run.id: None
                if run.elided_payload_paths
                else profile.sweepables.production_footing(run, results_by_run.get(run.id, []))
                for run in members
            }
        )
        for variant_key, members in sorted(arms.keyed.items())
    }


def _design_with_mechanism_confounds(
    design: RealizedDesign,
    results_by_run: Mapping[str, list[EvalResult]],
    observations: _MechanismObservations,
    *,
    served: _ServedModels,
    profile: HostProfile,
) -> RealizedDesign:
    """The design with each contrast arm's observed confounds against the control arm.

    Args:
        design: The derived design.
        results_by_run: Each run's results.
        observations: The campaign's mechanism observations.
        served: Which model answered each result's candidate calls, from :func:`_served_models`.
        profile: The host whose declaration of the model lever names its mechanism.

    Returns:
        The design, its contrasts qualified where they ran another model and a mechanism diverged, and where
        one requested model id was answered by more than one model across the two arms; unchanged when no
        control resolved.
    """
    if design.control_arm is None:
        return design
    control = [result for run_id in design.control_arm.run_ids for result in results_by_run.get(run_id, [])]
    contrasts = []
    for arm in design.contrasts:
        side = [result for run_id in arm.run_ids for result in results_by_run.get(run_id, [])]
        confounds = _model_contrast_confounds(control, side, observations, profile=profile)
        confounds += _served_model_confounds((result.id for result in (*control, *side)), served)
        contrasts.append(arm.model_copy(update={"mechanism_confounds": confounds}))
    return design.model_copy(update={"contrasts": contrasts})


def _arm_mechanisms(
    arms: _CampaignArms, results_by_run: Mapping[str, list[EvalResult]], observations: _MechanismObservations
) -> list[ArmMechanismReading]:
    """Each arm's mean of every observed-mechanism covariate, saying so where none was measured.

    Args:
        arms: The campaign's arms. A run that resolved no arm is in none, and so in no reading.
        results_by_run: Each run's results.
        observations: The campaign's mechanism observations.

    Returns:
        One reading per arm and covariate, sorted by arm then covariate.
    """
    readings: list[ArmMechanismReading] = []
    for variant_key, members in sorted(arms.keyed.items()):
        result_ids = {result.id for run in members for result in results_by_run.get(run.id, [])}
        for covariate in sorted(_OBSERVED_MECHANISMS):
            values = observations.get(covariate, {})
            means = _level_means(values, {variant_key: result_ids})
            readings.append(
                ArmMechanismReading(
                    variant_key=variant_key,
                    covariate=covariate,
                    mean=float(means[variant_key]) if variant_key in means else None,
                    n_measured=sum(1 for result_id in result_ids if result_id in values),
                    n_results=len(result_ids),
                )
            )
    return readings


#: Result id -> what its candidate calls say about the model that answered them. A result whose candidate
#: left no usage row is absent: nothing was called, so nothing answered, and no claim is made about it.
_ServedModels = dict[str, ResultServedReading]


def _served_models(results: Iterable[EvalResult]) -> _ServedModels:
    """Read which model answered each result's candidate calls, off its candidate usage rows.

    ``RoleUsage.served_model`` only — what the provider's response named — and never ``RoleUsage.model``
    or the run's ``candidate_model``, which are what the launch asked for and, for a floating alias, name
    the pointer rather than the model behind it.

    Args:
        results: The campaign's results.

    Returns:
        Each result's reading, for the results whose candidate left a usage row.
    """
    return {result.id: reading for result in results if (reading := served_reading(result)) is not None}


def _served_model_confounds(result_ids: Iterable[str], served: _ServedModels) -> list[Confound]:
    """Name the candidate's served model as a confound where one requested id was answered by more than one model.

    The served model is EXPECTED to move with the requested one — a comparison between two model ids is
    a comparison between the models that answered them — so a difference between arms that asked for
    different ids is the lever, not a confound. What is a confound is one requested id answered by two
    models across the runs compared: within one arm, its numbers are a mixture; between two arms that
    asked for the same id, the arms differ by a model nobody set. The rule is the fold the engine applies
    to a resolved surface (:class:`_SurfaceFolds`): the served model folds into the requested one where it
    is constant within each requested id, and only there.

    Args:
        result_ids: The results under comparison, every side together.
        served: The campaign's readings, from :func:`_served_models`.

    Returns:
        One ``served_model`` confound, ``varied`` where some requested id was answered by two or more named
        models, ``undecided`` where none was but some candidate call named no model — which model answered
        cannot be established there, and unknown is never read as one model. Empty otherwise, including where
        no result under comparison called its candidate.
    """
    by_requested: dict[str, set[str]] = {}
    unrecorded = False
    for result_id in result_ids:
        if (reading := served.get(result_id)) is not None:
            by_requested.setdefault(reading.requested, set()).update(reading.served)
            unrecorded = unrecorded or reading.unrecorded
    if any(len(models) > 1 for models in by_requested.values()):
        return [Confound(dimension=CANDIDATE_SERVED_MODEL_CONFOUND, kind="served_model")]
    if unrecorded:
        return [Confound(dimension=CANDIDATE_SERVED_MODEL_CONFOUND, kind="served_model", status="undecided")]
    return []


def _arm_served_models(
    arms: _CampaignArms, results_by_run: Mapping[str, list[EvalResult]], served: _ServedModels
) -> list[ArmServedModel]:
    """Each arm's served models, as the provider's responses named them.

    Args:
        arms: The campaign's arms. A run that resolved no arm is in none, and so in no reading.
        results_by_run: Each run's results.
        served: The campaign's readings, from :func:`_served_models`.

    Returns:
        One reading per arm whose candidate left a usage row, sorted by arm.
    """
    readings: list[ArmServedModel] = []
    for variant_key, members in sorted(arms.keyed.items()):
        pooled = pool_served_readings(
            served.get(result.id) for run in members for result in results_by_run.get(run.id, [])
        )
        if pooled is not None:
            readings.append(ArmServedModel(variant_key=variant_key, **pooled.model_dump()))
    return readings


def _confound_catalog(bundle: AnalysisContextBundle, *, profile: HostProfile) -> dict[str, str]:
    """Collect the reason for every confound dimension appearing anywhere in the bundle.

    The reason belongs in the payload — a name a reader cannot judge is a label — but it
    belongs there ONCE. The same dimension is re-stated by every lever that names it, by
    every divergence, and by the campaign-wide scan, so an inlined multi-sentence reason is
    the identical paragraph repeated emitters × dimensions times inside the bundle that IS
    the paid one-shot prompt. Same normalisation, same reason, as ``measure_catalog``.

    Built from the assembled bundle rather than alongside it, so the catalog cannot claim a
    dimension the lenses never emitted, and cannot miss one they did. **All three emitters
    are read**, not the two that came first: a dimension named only by the campaign-wide scan
    — the case a campaign that swept nothing produces — would otherwise reach a report as a
    bare name with no reason attached, which is the one thing this catalog exists to prevent.

    Args:
        bundle: The assembled bundle, with coverage and divergences already populated.
        profile: The host whose vocabulary this reads.

    Returns:
        ``{dimension: why}`` for every dimension named anywhere in the bundle's confound
        lists. A swept lever's reason is generic by design — a campaign sweeping a lever
        already believes it moves the numbers, which is what makes it a lever. An observed
        mechanism's reason is the engine's own, registered beside its threshold.
    """
    catalog: dict[str, str] = {}
    apparatus_reasons = _apparatus_confound_reasons(profile=profile)
    sweepables = profile.sweepables
    # A resolved surface reaches a confound list only once the knob written into it is accounted for
    # and something is still left over, so its reason says that rather than the generic sentence
    # about another knob: "another lever this campaign swept" would be false of a surface nobody
    # swept, and silent about the one thing a reader needs — that a change no knob names rode in.
    surface_reasons = {
        surface: (
            _UNEXPLAINED_SURFACE_CONFOUNDS
            if claimant.open_family is not None
            else _UNEXPLAINED_WRITTEN_SURFACE_CONFOUNDS
        ).format(family=claimant.name, prose=sweepables.reader_prose(surface))
        for surface, claimant in sweepables.resolution_surfaces.items()
    }
    unverified_reasons = {
        f"{UNVERIFIED_FOLD_PREFIX}{surface}": _UNVERIFIED_FOLD_CONFOUNDS.format(
            knob=claimant.name, surface=surface, prose=sweepables.reader_prose(surface)
        )
        for surface, claimant in sweepables.resolution_surfaces.items()
        if claimant.open_family is None
    }
    emitted = [confound for entry in bundle.coverage for confound in entry.confounded_by]
    emitted += [confound for divergence in bundle.scope_divergences for confound in divergence.confounded_by]
    emitted += bundle.apparatus_confounds
    emitted += [confound for arm in bundle.design.contrasts for confound in arm.mechanism_confounds]
    emitted += [
        confound
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
        for confound in comparison.mechanism_confounds
    ]
    for confound in emitted:
        # Branch on ``kind``, which the scan sets, rather than on whether the name happens to
        # be an apparatus key. A name lookup with a fallback makes two wrong answers
        # expressible — a swept lever colliding with an apparatus name silently takes the
        # apparatus reason, and an apparatus dimension missing from the map silently takes the
        # swept-lever sentence instead of failing. Neither is reachable today; both would be
        # invisible if they became so.
        if confound.kind == "apparatus":
            reason = apparatus_reasons[confound.dimension]
        elif confound.kind == "unverified_fold":
            reason = unverified_reasons[confound.dimension]
        elif confound.kind == "observed_mechanism":
            reason = _OBSERVED_MECHANISMS[confound.dimension.removeprefix(OBSERVED_MECHANISM_PREFIX)].reason
        elif confound.kind == "served_model":
            reason = _SERVED_MODEL_CONFOUNDS
        else:
            reason = surface_reasons.get(confound.dimension, _SWEPT_LEVER_CONFOUNDS)
        catalog[confound.dimension] = (
            f"{UNDECIDED_CONFOUND_PREFIX}; if it did, {reason}" if confound.status == "undecided" else reason
        )
    return catalog


#: Measure name -> case id -> the exact mean of that case's observations of it, at one level.
_PerCaseMeasures = dict[str, dict[str, Fraction]]


def _per_case_measures(pooled: Mapping[str, _PooledMeasure]) -> _PerCaseMeasures:
    """Each numeric measure's per-case means at one level, exact — the unit a between-level test reads.

    Repeats of a case are averaged first, so a case repeated three times is one case, and averaged
    exactly (:func:`~threetears.evals.analysis.stats.exact_decimal`) so a constant read at unequal repeats
    stays one constant rather than acquiring a float residue a test would read as spread.

    Args:
        pooled: The level's pooled observations, from :func:`_collect_measures`.

    Returns:
        ``{name: {case id: mean}}`` for every numeric measure; text, boolean and categorical measures are absent.
    """
    per_case: _PerCaseMeasures = {}
    for name, (descriptor, values, cases) in pooled.items():
        if descriptor.data_type in ("text", "boolean", "categorical"):
            continue
        by_case: dict[str, list[Fraction]] = defaultdict(list)
        for value, case in zip(values, cases):
            by_case[case].append(exact_decimal(float(value)))
        per_case[name] = {
            case: sum(case_values, Fraction(0)) / len(case_values) for case, case_values in sorted(by_case.items())
        }
    return per_case


def measure_movement(
    descriptor: MetricDescriptor, at_a: Mapping[str, Fraction], at_b: Mapping[str, Fraction]
) -> MeasureMovement:
    """Test one measure's movement between two levels against its own noise, and read it against what matters.

    Public as the one reading the scope-divergence lens grades a whole, a part and each component by, so
    :func:`component_carrier` can be handed movements read the way the lens reads them.

    Args:
        descriptor: The measure's descriptor — its name, scope, better direction, and the materiality
            threshold that is both the delta's label and the margin an equivalence test runs against.
        at_a: Its per-case means at the first level.
        at_b: Its per-case means at the second level.

    Returns:
        The movement, read by :func:`~threetears.evals.analysis.stats.level_difference`.
    """
    tested = level_difference(
        at_a, at_b, equivalence_margin=descriptor.materiality_threshold, value_range=descriptor.value_range
    )
    assert tested.mean_a is not None and tested.mean_b is not None and tested.delta is not None
    direction: MovementDirection
    if tested.separated is None:
        direction = "untested"
    elif tested.separated:
        direction = "improved" if (tested.delta > 0) == bool(descriptor.higher_is_better) else "regressed"
    elif tested.equivalent:
        direction = "equivalent"
    else:
        direction = "not_separated"
    return MeasureMovement(
        name=descriptor.name,
        scope=descriptor.attribution_scope,
        mean_a=tested.mean_a,
        mean_b=tested.mean_b,
        delta=tested.delta,
        se_of_delta=tested.se,
        test=tested.test,
        n_a=tested.n_a,
        n_b=tested.n_b,
        direction=direction,
        materiality=materiality(descriptor.materiality_threshold, tested.delta),
        not_separated_reason=tested.not_separated_reason,
    )


def _difference_range(
    minuend: MetricDescriptor | None, subtrahend: MetricDescriptor | None
) -> tuple[float, float] | None:
    """The range one measure minus another can take, from their declared ranges, or None unless both declare one.

    A remainder or a gap between two measures is what the scope lens tests, and a move with no spread is read on a
    declared range (:func:`~threetears.evals.analysis.stats.separation_test`): ``[low₁ − high₂, high₁ − low₂]``.
    """
    if minuend is None or subtrahend is None or minuend.value_range is None or subtrahend.value_range is None:
        return None
    (low_1, high_1), (low_2, high_2) = minuend.value_range, subtrahend.value_range
    return low_1 - high_2, high_1 - low_2


def _remainders(whole: Mapping[str, Fraction], part: Mapping[str, Fraction]) -> dict[str, Fraction]:
    """Each case's whole minus its part, at one level, over the cases carrying both — what a divergence tests."""
    return {case: value - part[case] for case, value in whole.items() if case in part}


def _comparable_pairs(
    level_a: MeasureCollection,
    level_b: MeasureCollection,
    catalog: dict[str, MetricDescriptor],
) -> Iterator[tuple[str, MeasureSummary, MeasureSummary, MeasureSummary, MeasureSummary]]:
    """Yield ``(unit, e_at_a, e_at_b, s_at_a, s_at_b)`` for each cross-scope pair sharing a unit.

    Only numeric, directional measures present at BOTH levels qualify: a measure observed at
    one level and not the other has no difference to grade, and one with no better direction
    cannot be said to have improved.

    Args:
        level_a: Measures at the first level.
        level_b: Measures at the second level.
        catalog: Descriptors by measure name — the source of each measure's unit.

    Yields:
        One tuple per comparable cross-scope pair.
    """
    at_a = {m.name: m for m in level_a.measures}
    at_b = {m.name: m for m in level_b.measures}
    shared = [
        name
        for name in sorted(at_a.keys() & at_b.keys())
        if at_a[name].mean is not None and at_b[name].mean is not None
    ]
    by_scope: dict[AttributionScope, list[str]] = {"end_to_end": [], "subsystem": []}
    for name in shared:
        if at_a[name].higher_is_better is not None and catalog.get(name) is not None and catalog[name].unit:
            by_scope[at_a[name].attribution_scope].append(name)
    for whole in by_scope["end_to_end"]:
        for part in by_scope["subsystem"]:
            unit = catalog[whole].unit
            if unit and unit == catalog[part].unit:
                yield unit, at_a[whole], at_b[whole], at_a[part], at_b[part]


def _unsound_subtraction(
    *,
    whole: MeasureSummary,
    part: MeasureSummary,
    catalog: dict[str, MetricDescriptor],
    observation_units: list[dict[str, str]],
    profile: HostProfile,
) -> str | None:
    """Say why differencing these two measures would describe nothing, or None if it would not.

    Four ways a subtraction goes wrong while both sides stay individually true, so the
    answer is a SENTENCE rather than a flag: the returned string is what a reader is owed
    in place of the number, and it is the bundle's job to state it rather than the
    generator's to guess it from a null.

    **Two clauses about containment, and they are different questions.** The first is
    whether the part is inside the whole at all. Sharing a unit makes two measures
    comparable, not nested: background-tool phases, the drain wait and the judge phase are all
    milliseconds that fall OUTSIDE the turn spans ``total_ms`` sums — background work by
    design, so a long background run cannot inflate turn latency, and judging because it
    scores after the turns have ended — so subtracting one from the other once produced a
    ~95-second "unattributed" swing that described no stretch of wall-clock at all. Containment is
    therefore declared on the measure (``MetricDescriptor.contained_by``) and defaults to
    absent, which fails closed: an undeclared part is reported beside its whole with both
    movements intact and no arithmetic between them.

    The second is whether the part EXHAUSTS the whole, and it is the one this function was
    missing. Containment was treated as sufficient, so a part that is one of three declared
    components earned the subtraction, and the leftover — the sum of the OTHER components'
    movements — was published as "unattributed". That word is a claim that nothing accounts
    for it, and here the catalog names exactly what does. Disjointness was handled and
    partition was not, which is why the sibling case (``async_wait_ms``, undeclared) was
    correctly withheld in the same analysis that got this one wrong.

    Args:
        whole: The end-to-end measure's summary at the first level.
        part: The subsystem measure's summary at the first level.
        catalog: Descriptors by measure name — the source of the containment declaration.
        observation_units: Each compared level's measure → observation-unit map.
        profile: The host whose vocabulary this reads.

    Returns:
        A sentence naming the reason, or None when the difference is sound.
    """
    if whole.higher_is_better != part.higher_is_better:
        return (
            f"{whole.name} and {part.name} {WITHHELD_OPPOSITE_DIRECTIONS}, so adding their movements "
            "would sum two changes that mean opposite things."
        )
    seen = {units.get(name) for units in observation_units for name in (whole.name, part.name)}
    if None in seen:
        # Defensive: the unit map is built from the same walk as the measures it accompanies,
        # so a measure present in the collection has one. Withholding beats assuming — and
        # beats a KeyError, which would take the whole analysis down over one measure.
        return f"{whole.name} or {part.name} {WITHHELD_UNKNOWN_POPULATION} on one of the two levels."
    if len(seen) > 1:
        return (
            f"{whole.name} and {part.name} {WITHHELD_DIFFERENT_POPULATIONS} "
            f"({' vs '.join(sorted(str(unit) for unit in seen))}), so their difference is off by however "
            "many observations each result contributed."
        )
    # Containment and exhaustion are the rule every remainder site shares, so they are asked of
    # the one predicate that states them — the attribution chart compiler asks the same question.
    return remainder_withheld_reason(part.name, whole.name, catalog.get(part.name), catalog, measures=profile.measures)


def _carried_by(
    whole: MeasureMovement,
    level_a: _PerCaseMeasures,
    level_b: _PerCaseMeasures,
    catalog: dict[str, MetricDescriptor],
    *,
    profile: HostProfile,
) -> tuple[list[MeasureMovement], str | None, float | None]:
    """Grade each declared component of a whole between two levels, and name the one carrying it.

    The components are read from the same declarations :func:`_unsound_subtraction` reads —
    :func:`~threetears.evals.kernel.metrics.partition_components` over the whole describable measure space —
    so the partition this names and the partition that withholds a remainder are one partition.
    What this adds is where the movement went, which a withheld remainder deliberately leaves
    unsaid: the lens's own rule is to report the other components rather than a remainder, and a
    reader handed only that rule and no components has nothing to report.

    Args:
        whole: The whole-run measure's movement.
        level_a: Per-case measures at the first level.
        level_b: Per-case measures at the second level.
        catalog: Descriptors by measure name.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(components, carried_by, carried_share)``, the carrier decided by :func:`component_carrier`.
    """
    components = [
        measure_movement(catalog[name], level_a[name], level_b[name])
        for name in partition_components(whole.name, catalog, measures=profile.measures)
        if level_a.get(name) and level_b.get(name)
    ]
    carrier = component_carrier(whole, components, level_a, level_b)
    if carrier is None:
        return components, None, None
    return components, carrier.name, carrier.delta / whole.delta


def component_carrier(
    whole: MeasureMovement,
    components: Sequence[MeasureMovement],
    level_a: Mapping[str, Mapping[str, Fraction]],
    level_b: Mapping[str, Mapping[str, Fraction]],
) -> MeasureMovement | None:
    """The component SHOWN to carry the whole's movement, or None where the data cannot name one.

    Public so the rule the scope-divergence lens names a ``carried_by`` component by can be read, and tested for its
    false-naming rate, on per-case values directly: the lens itself calls exactly this, over movements read by
    :func:`measure_movement`.

    The candidate is the component whose delta, in the whole's direction, is largest. It is named only
    when two things are shown, each by the engine's between-level test
    (:func:`~threetears.evals.analysis.stats.level_difference`): its own movement separates in the whole's
    direction, and it moved further that way than every other component — each case's difference between the
    candidate and that component, tested between the levels, Holm-adjusted over the other components. Named on
    the largest delta alone, two components moved alike would hand the carrier to whichever noise favoured.

    Args:
        whole: The whole-run measure's movement.
        components: Each component's movement.
        level_a: Per-case measures at the first level.
        level_b: Per-case measures at the second level.

    Returns:
        The carrier, or None when the whole's movement does not separate, no component moved its way, or the
        largest mover is not shown to move further than every other.
    """
    if whole.direction not in ("improved", "regressed") or whole.delta == 0.0:
        return None
    sign = 1.0 if whole.delta > 0 else -1.0
    moving = [component for component in components if component.delta * sign > 0]
    if not moving:
        return None
    top = max(moving, key=lambda component: (component.delta * sign, component.name))
    if top.direction != whole.direction:
        return None
    p_values: list[float] = []
    for other in components:
        if other.name == top.name:
            continue
        gap_a = {
            case: value - level_a[other.name][case]
            for case, value in level_a[top.name].items()
            if case in level_a[other.name]
        }
        gap_b = {
            case: value - level_b[other.name][case]
            for case, value in level_b[top.name].items()
            if case in level_b[other.name]
        }
        tested = level_difference(gap_a, gap_b)
        if tested.p_value is None or tested.delta is None or tested.delta * sign <= 0:
            return None
        p_values.append(tested.p_value)
    return top if all(p < SIGNIFICANCE_ALPHA for p in holm_adjust(p_values)) else None


class _DivergenceCount(NamedTuple):
    """How many whole-and-part pairs the divergence lens tested, and how many it could not."""

    tested: int
    untested: int


def _scope_divergences(
    runs: list[EvalRun],
    results_by_run: dict[str, list[EvalResult]],
    catalog: dict[str, MetricDescriptor],
    apparatus_levels: dict[str, dict[str, str | None]],
    design: RealizedDesign,
    *,
    folds: _SurfaceFolds,
    observations: _MechanismObservations,
    served: _ServedModels,
    profile: HostProfile,
) -> tuple[list[ScopeDivergence], int, _DivergenceCount]:
    """Find the lever changes where the whole run moved by a different amount than the part under test.

    Each level of each swept lever gets its own measure collection, built from that level's
    results by the same walk the rest of the bundle uses — so the observations compared here
    are the real ones, not summaries of summaries. Levels are then compared pairwise, and for
    each whole-and-part pair sharing a unit **the divergence itself is tested**: each case's
    whole minus its part (per-case means) between the two levels, by the engine's between-level
    test (:func:`~threetears.evals.analysis.stats.level_difference`). Every such test of one lever
    is one family, Holm-corrected at the engine's alpha, and a divergence is published only where
    its adjusted p is below it. Grading the whole and the part apart and publishing where their
    verdicts differ is NOT a test of the difference (Gelman & Stern 2006): a whole that
    separates beside a part that does not is ordinary noise, and that rule published a divergence
    that did not exist 11–33% of the time.

    Two honesty constraints ride along, because a comparison this cheap to produce is easy
    to over-read. The cohorts are grouped by ONE lever, so they also differ in whatever else
    moved — every such dimension is named in ``confounded_by``, and the reason a change in it
    matters is in the bundle's ``confound_catalog``.
    And the unattributed swing is stated only when the difference is sound at all — see
    :func:`_unsound_subtraction`, which names the reason when it is not, so a reader gets a
    sentence rather than a null.

    Args:
        runs: The campaign's resolved runs.
        results_by_run: Each run's results, keyed by run id.
        catalog: Descriptors by measure name, for the unit that makes a pair comparable.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.
        design: The derived design, narrowing each lever's cohort the same way the coverage
            map narrows it — a divergence read over pooled cells would disagree with the
            coverage entry for the same lever, and a reader has no way to tell which is right.
        folds: The campaign's :class:`_SurfaceFolds`. A resolved surface whose movement across
            its cohort is its swept members' is not compared as a lever of its own, for the reason
            the coverage map drops its row: its levels are the members' levels under another name,
            so every divergence it produced would restate one a member already reports.
        observations: The campaign's mechanism observations, compared across each divergence's two levels
            for the observed-mechanism confounds it names.
        served: Which model answered each result's candidate calls, for the served-model confound.
        profile: The host whose vocabulary this reads.

    Returns:
        The divergences to report (strongest first, capped), the count dropped by the cap, and how
        many pairs were tested and could not be.
    """
    found: list[tuple[float, ScopeDivergence]] = []
    n_tested = n_untested = 0
    all_run_ids = [run.id for run in runs]
    lever_levels = _lever_levels(runs, results_by_run, profile=profile)
    for lever, campaign_wide in lever_levels.items():
        cohort = set(_lever_cohort(lever, design, all_run_ids))
        by_level = {
            level: members for level, raw in campaign_wide.items() if (members := [r for r in raw if r in cohort])
        }
        if len(by_level) < 2 or folds.folds_away(lever, cohort):
            continue
        levels = sorted(by_level)
        collected = {
            level: _collect_measures(
                [result for run_id in by_level[level] for result in results_by_run.get(run_id, [])],
                profile=profile,
                undeclared="all_observed",
            )
            for level in levels
        }
        collections = {level: collection for level, (collection, _, _) in collected.items()}
        per_case = {level: _per_case_measures(pooled) for level, (_, _, pooled) in collected.items()}
        result_ids = {
            level: {result.id for run_id in by_level[level] for result in results_by_run.get(run_id, [])}
            for level in levels
        }
        units = {level: observation_units for level, (_, observation_units, _) in collected.items()}
        # Every test this lever's comparisons carried, with what a published divergence needs beside it.
        family: list[tuple[LevelDifference, dict[str, Any]]] = []
        for index, level_a in enumerate(levels):
            for level_b in levels[index + 1 :]:
                confounded = _uncontrolled_dimensions(
                    lever,
                    by_level[level_a] + by_level[level_b],
                    lever_levels,
                    apparatus_levels,
                    folds=folds,
                    profile=profile,
                ) + _observed_mechanism_confounds(
                    lever, {level: result_ids[level] for level in (level_a, level_b)}, observations, profile=profile
                )
                confounded += _served_model_confounds(result_ids[level_a] | result_ids[level_b], served)
                at_a, at_b = per_case[level_a], per_case[level_b]
                for unit, e_a, _e_b, s_a, _s_b in _comparable_pairs(
                    collections[level_a], collections[level_b], catalog
                ):
                    tested = level_difference(
                        _remainders(at_a[e_a.name], at_a[s_a.name]),
                        _remainders(at_b[e_a.name], at_b[s_a.name]),
                        value_range=_difference_range(catalog.get(e_a.name), catalog.get(s_a.name)),
                    )
                    if tested.p_value is None:
                        n_untested += 1
                        continue
                    whole = measure_movement(catalog[e_a.name], at_a[e_a.name], at_b[e_a.name])
                    part = measure_movement(catalog[s_a.name], at_a[s_a.name], at_b[s_a.name])
                    withheld = _unsound_subtraction(
                        whole=e_a,
                        part=s_a,
                        catalog=catalog,
                        observation_units=[units[level_a], units[level_b]],
                        profile=profile,
                    )
                    components, carried_by, carried_share = _carried_by(whole, at_a, at_b, catalog, profile=profile)
                    assert tested.test is not None
                    family.append(
                        (
                            tested,
                            {
                                "lever": lever,
                                "level_a": level_a,
                                "level_b": level_b,
                                "unit": unit,
                                "end_to_end": whole,
                                "subsystem": part,
                                "test": tested.test,
                                "n_cases_a": tested.n_a,
                                "n_cases_b": tested.n_b,
                                "p_raw": tested.p_value,
                                "unattributed_delta": None if withheld else whole.delta - part.delta,
                                "unattributed_withheld": withheld,
                                # Read from the same catalog `_unsound_subtraction` consulted, so the
                                # published fact and the decision made from it have one source.
                                "contained_by": (
                                    described.contained_by if (described := catalog.get(part.name)) else None
                                ),
                                "confounded_by": confounded,
                                "whole_components": components,
                                "carried_by": carried_by,
                                "carried_share": carried_share,
                            },
                        )
                    )
        n_tested += len(family)
        adjusted = holm_adjust([tested.p_value or 0.0 for tested, _ in family])
        for (tested, fields), p_adjusted in zip(family, adjusted):
            if p_adjusted >= SIGNIFICANCE_ALPHA:
                continue
            divergence = ScopeDivergence(**fields, p_adjusted=p_adjusted, family_size=len(family))
            # Rank by how far the whole's movement and the part's differ — the difference the test
            # read — as a fraction of the whole's own scale: the question a reader opens a divergence
            # to answer, and scale-free so milliseconds and dollars can be ordered against each other.
            # The scale takes both levels so a measure starting near zero cannot manufacture an
            # unbounded score.
            whole = divergence.end_to_end
            scale = max(abs(whole.mean_a), abs(whole.mean_b), 1e-9)
            strength = abs(tested.delta or 0.0) / scale
            found.append((strength, divergence))

    found.sort(
        key=lambda item: (
            -item[0],
            item[1].lever,
            item[1].level_a,
            item[1].level_b,
            item[1].end_to_end.name,
            item[1].subsystem.name,
        )
    )
    return (
        [divergence for _, divergence in found[:_MAX_DIVERGENCES]],
        max(0, len(found) - _MAX_DIVERGENCES),
        _DivergenceCount(n_tested, n_untested),
    )


def _token_rollup(results: list[EvalResult]) -> TokenRollup | None:
    """Sum token usage across results carrying a usage breakdown, or None.

    Sums what was reported (:func:`~threetears.evals.kernel.provider.sum_optional_tokens`, the
    package's one definition of that) and counts the results whose token-metered rows left a count
    unreported, rather than adding an unreported count as zero.
    """
    prompt: int | None = None
    completion: int | None = None
    reasoning: int | None = None
    n_with_usage = 0
    n_unreported = 0
    for result in results:
        if not result.usage:
            continue
        n_with_usage += 1
        for role in result.usage:
            prompt = sum_optional_tokens(prompt, role.prompt_tokens)
            completion = sum_optional_tokens(completion, role.completion_tokens)
            reasoning = sum_optional_tokens(reasoning, role.reasoning_tokens)
        if any(
            row.role != "external" and (row.prompt_tokens is None or row.completion_tokens is None)
            for row in result.usage
        ):
            n_unreported += 1
    if n_with_usage == 0:
        return None
    return TokenRollup(
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        n_results_with_usage=n_with_usage,
        n_results_tokens_unreported=n_unreported,
    )


def _result_has_error(result: EvalResult) -> bool:
    """True if a result carries any runner/judge/candidate/infra error.

    Deliberately NOT
    :func:`~threetears.evals.kernel.result_condition.resolve_result_condition`'s ``scoring`` axis,
    which every per-result read surface uses. The two answer different questions: this counts
    results that carry an error field at all, while the scoring axis decides how a result
    participates in aggregates — a candidate failure the turn budget or the output cap caused
    carries no error field and is still a hard fail there. Reads the categorized fields only,
    never the combined ``runner_error`` display string (the fragile-parse rule): the runner writes
    all of them from one ledger, so nothing is set there that is not set here.
    """
    return bool(result.judge_error or result.candidate_error or result.infra_error)


def _measured_prod_costs(run_results: list[EvalResult]) -> list[float]:
    """Every production-replicating cost a result actually measured — no placeholders.

    A result that evidences nothing to decompose, that observed no production-role spend,
    or that carries a substituted delivery yields ``None`` from
    :func:`production_replicating_cost`, and is dropped here rather than contributing a
    zero. That is the whole point: a zero nobody observed, averaged in, ranks the
    least-measured configuration the cheapest — failing toward "cheaper than reality",
    which is the dangerous direction for a config or capacity decision.

    ``substituted_deliveries`` is passed per result rather than assumed, because a
    substituted delivery leaves no usage row behind: the rows alone cannot evidence that
    they are incomplete.

    Read over the turns the candidate took
    (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`), as every cost reading is: a
    billed refusal's dollars are no turn's spend, and averaged in they made a refusing configuration cheap.

    Args:
        run_results: The run's results.

    Returns:
        One float per turn that measured a production-replicating cost, in input order.
    """
    measured: list[float] = []
    for result in run_results:
        if not delivered_a_turn(result):
            continue
        cost = production_replicating_cost(result.usage, substituted_deliveries=count_substituted_deliveries(result))
        if cost is not None:
            measured.append(cost)
    return measured


def _run_summary(
    run: EvalRun, run_results: list[EvalResult], reportable: set[str], *, profile: HostProfile
) -> RunSummary:
    """Digest one run + its results into a :class:`RunSummary`.

    Args:
        run: The run.
        run_results: That run's results.
        reportable: The levers the campaign's coverage map names — see
            :func:`_reportable_levers`. ``config`` is narrowed to these so the two surfaces
            cannot name different levers for one campaign, and so a contestant property nobody
            swept does not arrive as a config value the generator reads as a departure.
        profile: The host whose vocabulary this reads.

    Returns:
        The summary.
    """
    effective = {
        lever: eff for lever, eff in _effective_config(run, run_results, profile=profile).items() if lever in reportable
    }
    prod_costs = _measured_prod_costs(run_results)
    return RunSummary(
        run_id=run.id,
        status=str(run.status),
        created_at=run.created_at,
        candidate_model=run.candidate_model,
        k_runs=run.k_runs,
        config={lever: eff.value for lever, eff in effective.items() if eff.value is not None},
        config_provenance={lever: eff.provenance for lever, eff in effective.items()},
        n_results=len(run_results),
        n_errors=sum(1 for r in run_results if _result_has_error(r)),
        cost_usd=sum(r.cost_usd for r in run_results if r.cost_usd is not None),
        n_cost_unpriced=sum(1 for r in run_results if r.cost_usd is None),
        prod_cost_usd=sum(prod_costs) if prod_costs else None,
        mean_prod_cost_usd=(sum(prod_costs) / len(prod_costs)) if prod_costs else None,
        n_prod_cost_usd=len(prod_costs),
        # A run read without its host payload cannot be checked: an elided lever would read as the
        # subject's own setting. None says nobody checked rather than claiming nothing moved.
        production_footing=(
            None if run.elided_payload_paths else profile.sweepables.production_footing(run, run_results)
        ),
        measures=_measure_collection(run_results, profile=profile, undeclared="all_observed"),
    )


def _model_versions(runs: list[EvalRun], records: list[ScoreRecord]) -> dict[str, str]:
    """Collect distinct models by role — candidate/judge/simulator — as a flat map.

    A single value per role reads cleanly; a swept role (e.g. a judge bake-off)
    shows every value comma-joined rather than silently collapsing to one.
    """
    candidate = sorted({run.candidate_model for run in runs})
    judge = sorted(
        {run.judge_model for run in runs if run.judge_model} | {r.judge_model for r in records if r.judge_model}
    )
    simulator = sorted({r.simulator_model for r in records if r.simulator_model})
    versions: dict[str, str] = {}
    if candidate:
        versions["candidate"] = ", ".join(candidate)
    if judge:
        versions["judge"] = ", ".join(judge)
    if simulator:
        versions["simulator"] = ", ".join(simulator)
    return versions


def _within_level_dispersion(
    composite_records: list[ScoreRecord], lever: str, effective_by_run: dict[str, dict[str, EffectiveLever]]
) -> str:
    """A lever's within-level composite spread — the estimate's measurement noise.

    Groups the composite observations by the lever's own level, takes the standard
    error of the mean *within* each level (the core ``stats`` helper — no new
    statistic), and averages across levels. This is the noise the point estimate at
    a level carries — the "is this ±0.10 or ±0.01" signal the analysis wants — and it is
    lever-specific (a different grouping per lever), unlike a spread pooled across
    every observation. The standard error is over test cases
    (``clustered_standard_error``), since a level's repeats of one case are not independent
    draws, and it needs ≥2 cases, so a level with a lone case contributes nothing; if no
    level clears that bar the spread is unestimable and the field reads ``"unscored"``
    (honest, not a fabricated 0).

    Args:
        composite_records: Score records for the composite metric, ``value`` present.
        lever: The lever to group by.
        effective_by_run: Each run's resolved levers, keyed by run id.

    **A ragged pool says so in the text** (#638): where the composites behind the spread were meaned over
    different dimension sets (:func:`~threetears.evals.analysis.reporting.pooled_composite_basis`), the spread
    is partly the difference between those sets, and the text carries the sets beside the number.

    Returns:
        ``"±"`` and the mean within-level SEM in :func:`~threetears.evals.analysis.numbers.format_number`'s spelling,
        followed by the ragged-composite disclosure in parentheses where the pool is ragged, or ``"unscored"``.
    """
    by_level: dict[str, tuple[list[float], list[str]]] = {}
    pooled: list[ScoreRecord] = []
    for record in composite_records:
        if record.value is not None and (level := _lever_value(record, lever, effective_by_run)) is not None:
            values, cases = by_level.setdefault(level, ([], []))
            values.append(record.value)
            cases.append(record.test_case_id)
            pooled.append(record)
    sems = [sem for values, cases in by_level.values() if (sem := clustered_standard_error(values, cases)) is not None]
    if not sems:
        return "unscored"
    basis = pooled_composite_basis(pooled)
    ragged = f" ({basis.disclosure()})" if basis is not None and basis.ragged else ""
    return f"±{format_number(sum(sems) / len(sems))}{ragged}"


def _lever_k_floor(
    lever: str,
    records: list[ScoreRecord],
    k_by_arm: dict[str, int],
    group_of_run: dict[str, str],
    effective_by_run: dict[str, dict[str, EffectiveLever]],
) -> int:
    """Repeat depth available for comparing one lever's levels.

    Comparing levels needs replication at *each* level, so the weakest level binds the
    comparison; within a level, the best-replicated ARM is what a reader can lean on.
    Hence the minimum, across levels, of the maximum arm ``k`` observed at that level — an arm's
    repeats per case (:func:`_arm_repeats`), so two runs of one arm over one case set at k=3 are one
    arm at k=6.

    A campaign-wide ``min(k_runs)`` was the earlier answer, and it coupled every lever to
    the weakest run anywhere in the campaign: one k=1 exploratory run dropped *all* of them
    to ``thin``, including levers swept three-deep at every level, and no later replication
    could lift them because a minimum only falls. The per-level maximum localises the
    penalty to the lever that actually lacks repeats — a model swept once stays thin while
    a fanout swept three-deep at every level reads as measured.

    **This is not monotonic in campaign size, and should not be read as if it were.** A run
    that opens a NEW level still lowers the floor for its own lever, because that level
    genuinely has one repeat behind it — see the sibling test that pins exactly this. What
    the change removes is *spurious* coupling: a run can no longer degrade a lever it says
    nothing about.

    Args:
        lever: A lever name or a declared coordinate name.
        records: The projected score records (they carry the level and the run).
        k_by_arm: Repeats per case per arm group, keyed as :meth:`_CampaignArms.groups` keys them.
        group_of_run: Run id → the arm group it belongs to.
        effective_by_run: Each run's resolved levers, keyed by run id.

    Returns:
        The binding repeat depth, or 0 when no record carries a resolvable run.
    """
    best_at_level: dict[str, int] = {}
    for record in records:
        level = _lever_value(record, lever, effective_by_run)
        if level is None:
            continue
        k = k_by_arm.get(group_of_run.get(record.run_id, ""), 0)
        best_at_level[level] = max(best_at_level.get(level, 0), k)
    return min(best_at_level.values(), default=0)


def _reportable_levers(
    resolved: set[str],
    records: list[ScoreRecord],
    effective_by_run: dict[str, dict[str, EffectiveLever]],
    declared_design: CampaignDesign | None,
    engaged: set[str],
) -> set[str]:
    """Which levers earn a coverage row: the ones that MOVED, plus the ones the campaign declared.

    Reading every declared lever through the host's registry hands this
    lens the whole registry rather than one host's overlay carrier — which is the fix, and which
    also means an always-constant lever now reaches it. A host can have ten: a subject's
    components resolve a level on every run and move only when somebody edits the subject
    between arms. A row apiece would put ten constant lines in every memo a paid generator
    writes, and would invite a reader to sweep an axis whose levels are not contrastable.

    So a lever is reportable when the campaign **engaged** with it, in any of four ways:

    - it **moved** — more than one level across the campaign, read through the same
      :func:`_lever_value` the row's own ``levels`` are;
    - ``declared_design`` **names it as an axis**;
    - the **launch named it**, as an open family's member;
    - **observation recovered it** — the host declared a recovery rule and the role ran.

    The last two are what keep a HELD-FIXED lever reporting, and they are not decoration: a
    campaign that pinned one inner-agent model across every arm, and a single-arm campaign that
    never moved its candidate model, both get an ``unswept`` row rather than silence. "Held at
    one value" and "not a lever in this campaign" are different answers, and the silence is what
    once let a single-arm campaign's configuration name an inner-agent model and nothing else.
    The declared clause is load-bearing for a different reason: the completeness check
    (``generator._reject_incomplete_axis_coverage``) excuses silence about a declared axis only
    through a ``thin``/``unswept`` row, so a declared axis that resolved nothing must still get
    one — ``unswept``, at whatever the records bin to (every record at ``'—'`` where nothing
    resolved the lever, so ``cells=1``, not zero). "Held at one level, and that level is nothing
    anyone recorded" is the honest reading, and the refusal it would otherwise raise is not.

    **What this leaves out** is the one remaining case: a lever nothing but the run record speaks
    to, that never moved. Those are the contestant's own properties — a subject's backstory
    resolves a level on every run and moves only when somebody edits the subject between arms —
    and they are what the campaign held constant rather than measured. They are on NEITHER
    surface: ``_run_summary`` narrows ``config`` to this same set, deliberately, because ten
    content hashes per run each stamped ``overridden`` would tell the design lens every run
    departed from ten things it never touched. They reappear the moment one of them actually
    moves, which is the case worth seeing.

    Args:
        resolved: Every lever any run resolved, plus the candidate model where records disagree.
        records: The projected score records — the levels are read over these.
        effective_by_run: Each run's resolved levers, keyed by run id.
        declared_design: What the campaign SET OUT to sweep, or None when it declared nothing.
        engaged: Levers the launch named or observation recovered, across every run.

    Returns:
        The levers to build rows for.
    """
    declared = {axis.axis_id for axis in declared_design.axes} if declared_design else set()
    moved = {
        lever
        for lever in resolved
        if len({level for record in records if (level := _lever_value(record, lever, effective_by_run)) is not None})
        > 1
    }
    return moved | declared | (engaged & resolved)


def _run_axis_identities(
    axis_id: str, run: EvalRun, results: list[EvalResult], *, profile: HostProfile
) -> set[str] | None:
    """The identities a run's level on one declared axis can be joined to a declared level by, or None if unknown.

    Its variant coordinate where it has one (the engine's own and the host's), else the value the host's registry
    resolves for it, as its canonical digest and, for a string that may itself be a digest, as itself.

    Args:
        axis_id: The declared axis.
        run: The run.
        results: The run's results.
        profile: The host whose vocabulary this reads.

    Returns:
        The identities, or None when the run's level on the axis cannot be established.
    """
    coordinates = {
        **profile.engine_levels(run),
        **(profile.variant_levers(run) if profile.variant_levers is not None else {}),
    }
    if (coordinate := coordinates.get(axis_id)) is not None:
        return {coordinate.content_hash}
    value = profile.sweepables.resolve_levers(run, results).values.get(axis_id)
    if value is None:
        return None
    return {canonical_digest(value), value} if isinstance(value, str) else {canonical_digest(value)}


def _declared_crossing(
    design: CampaignDesign | None,
    runs: list[EvalRun],
    results_by_run: dict[str, list[EvalResult]],
    *,
    profile: HostProfile,
) -> DeclaredCrossing | None:
    """Mark every cell of a design that says which combinations it meant to run: ran, missing, or skipped by design.

    A cell is one declared level of every declared axis. A run sits at a cell when each axis's level joins to
    that cell's level (:func:`_run_axis_identities`). An unrun cell the design left out on purpose
    (:meth:`~threetears.evals.kernel.declaration.CampaignDesign.skipped_by_design`) is ``skipped_by_design``
    and is never a gap; an unrun cell it meant to run is ``not_run``, unless some run's level on an axis cannot
    be established, when it may be sitting there and the cell is ``undetermined``.

    Args:
        design: The campaign's declaration.
        runs: The campaign's resolved runs.
        results_by_run: Each run's results.
        profile: The host whose vocabulary this reads.

    Returns:
        The crossing, or None when there is no design or it says nothing about combinations — today's reading,
        where an unrun combination is neither skipped nor missing.
    """
    if design is None or not design.declares_cells():
        return None
    ran: set[tuple[str, ...]] = set()
    unestablished = False
    for run in runs:
        levels: list[str | None] = []
        for axis in design.axes:
            identities = _run_axis_identities(axis.axis_id, run, results_by_run.get(run.id, []), profile=profile)
            if identities is None:
                unestablished = True
                levels.append(None)
                continue
            matched = [level.content_hash for level in axis.values if level.content_hash in identities]
            levels.append(matched[0] if matched else None)
        if all(level is not None for level in levels):
            ran.add(tuple(level for level in levels if level is not None))
    cells = []
    for combination in product(*(axis.values for axis in design.axes)):
        identity = tuple(level.content_hash for level in combination)
        state: Literal["ran", "not_run", "skipped_by_design", "undetermined"]
        if identity in ran:
            state = "ran"
        elif design.skipped_by_design(identity):
            state = "skipped_by_design"
        else:
            state = "undetermined" if unestablished else "not_run"
        cells.append(
            DeclaredCellCoverage(
                levels={axis.axis_id: level.display for axis, level in zip(design.axes, combination, strict=True)},
                state=state,
            )
        )
    counts = {state: sum(1 for cell in cells if cell.state == state) for state in _CELL_STATES}
    kept, omitted = _capped(
        cells, _MAX_DECLARED_CELLS, weight=lambda cell: {"not_run": 2, "undetermined": 1}.get(cell.state, 0)
    )
    planned = len(cells) - counts["skipped_by_design"]
    sentence = (
        f"The design declares {len(cells)} cell(s) over its {len(design.axes)} axis(es) and meant to run {planned}: "
        f"{counts['ran']} ran"
        + (f", {counts['not_run']} never ran (a gap)" if counts["not_run"] else "")
        + (f", {counts['undetermined']} cannot be established" if counts["undetermined"] else "")
        + (
            f"; {counts['skipped_by_design']} were skipped by design, which is no gap"
            if counts["skipped_by_design"]
            else ""
        )
        + (f"; {omitted} cell(s) are left out of the list, gaps last to go" if omitted else "")
        + "."
    )
    return DeclaredCrossing(
        crossing=design.crossing,
        n_cells=len(cells),
        n_ran=counts["ran"],
        n_not_run=counts["not_run"],
        n_skipped_by_design=counts["skipped_by_design"],
        n_undetermined=counts["undetermined"],
        cells=kept,
        cells_omitted=omitted,
        sentence=sentence,
    )


def _declared_level_coverage(
    axis: SweptAxis,
    runs: list[EvalRun],
    results_by_run: dict[str, list[EvalResult]],
    *,
    profile: HostProfile,
) -> list[DeclaredLevelCoverage]:
    """Mark each level ``axis`` declares as ran, not run, or undetermined, joined on content identity.

    A run's level on the axis is its variant coordinate where it has one (the engine's own and the
    host's), else the value the host's registry resolves for it — joined to a declared level by
    ``content_hash``, the declaration's identity. A resolved value that is itself a digest is
    compared as one. A run whose level cannot be established (no coordinate, nothing resolved)
    blocks a ``not_run`` claim on every level no other run matched: it may be sitting at one.

    Args:
        axis: The declared axis.
        runs: The campaign's resolved runs.
        results_by_run: Each run's results.
        profile: The host whose vocabulary this reads.

    Returns:
        One entry per declared level, in the declaration's order.
    """
    observed: set[str] = set()
    unestablished = False
    for run in runs:
        identities = _run_axis_identities(axis.axis_id, run, results_by_run.get(run.id, []), profile=profile)
        if identities is None:
            unestablished = True
        else:
            observed |= identities
    return [
        DeclaredLevelCoverage(
            display=level.display,
            content_hash=level.content_hash,
            state="ran" if level.content_hash in observed else "undetermined" if unestablished else "not_run",
        )
        for level in axis.values
    ]


def _coverage_map(
    runs: list[EvalRun],
    records: list[ScoreRecord],
    apparatus_levels: dict[str, dict[str, str | None]],
    design: RealizedDesign,
    results_by_run: dict[str, list[EvalResult]],
    declared_design: CampaignDesign | None,
    *,
    folds: _SurfaceFolds,
    observations: _MechanismObservations,
    served: _ServedModels,
    arms: _CampaignArms | None = None,
    profile: HostProfile,
) -> list[LeverCoverageInput]:
    """Build the per-lever structural coverage map (the analysis's spine).

    A **lever** is whatever the host declares as one, resolved per run through
    :func:`_effective_config`. Which of them earns a row is :func:`_reportable_levers`: the ones
    that moved, plus the ones the campaign declared. A DECLARED axis that nothing resolved still
    gets a row reading ``unswept`` rather than no row at all — "held at one value" and "not a
    lever in this campaign" are different answers, and only the row can say which.
    For each, coverage reports how finely it was swept
    (``cells`` = distinct observed levels), a repeat floor (``k``, see
    :func:`_lever_k_floor` — per-lever and per arm, never a campaign-wide minimum),
    the samples informing it (``n`` = distinct results), the within-level composite
    spread (``dispersion``, see :func:`_within_level_dispersion`), and a coarse
    ``status``:

    - ``unswept`` — a single observed level (held fixed / not explored).
    - ``thin`` — swept, but ``k < 3`` or fewer than ~2 samples per cell (a k=1
      point estimate is noise-dominated).
    - ``measured`` — swept with an adequate repeat floor.

    Each lever also carries what did NOT hold still behind it (``confounded_by``). Most
    findings are formed per lever from this map rather than from the divergence lens, so a
    confound list that reached only the divergences left the common path unqualified: the
    generator was handed a lever's coverage with no way to know the campaign had also
    changed template and judge underneath it.

    **Each row states its own cohort in ``cohort_scope``, because the campaign-wide case is
    not simply "no control was designated".** Under a control, a lever some arm moved is read
    over the control plus those arms (:func:`_lever_cohort` says why pooling the rest
    manufactures a confound) — but a lever NO arm moved stays campaign-wide even then, since
    there is no contrast to narrow it to. The two rows are the same
    shape and differ only in what they may be read as, which is why the scope is a field
    rather than something a reader re-derives from the design.

    Deterministic: levers and their levels are sorted, so identical inputs give an
    identical map (and fingerprint).

    Args:
        runs: The campaign's resolved runs (for each run's ``k_runs`` and case set, counted per arm by :func:`_arm_repeats`).
        records: The projected score records (for levels, n, and composite spread).
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.
        design: The derived design — what each lever's comparison is drawn from. Without a
            control every lever is read over the whole campaign, which is the marginal
            comparison this map has always reported.
        results_by_run: Each run's results, keyed by run id — the observation side of
            effective-config resolution.
        declared_design: What the campaign SET OUT to sweep. A declared axis earns a row even
            when nothing resolved it — see :func:`_reportable_levers`.
        folds: The campaign's :class:`_SurfaceFolds`. A resolved surface earns no row of its own
            where its movement across its cohort is its swept members' movement — the member rows
            already report that change, and a second row levelled by content hash would count one
            override as two swept levers (and the same for a fixed knob's surface that moved only
            where the knob did). It keeps its row where the residuals disagree or cannot be read, or a
            fixed knob held at one level saw it differ, because then it carries something no knob's
            row does. A declared axis is never
            dropped this way: the completeness check needs its row whatever it resolved to.
        observations: The campaign's mechanism observations, which each row's ``mechanism`` check and its
            observed-mechanism confounds compare across the row's levels.
        served: Which model answered each result's candidate calls, for the served-model confound.
        arms: :func:`_campaign_arms`'s answer, when the caller already has it; derived otherwise.
        profile: The host whose vocabulary this reads.

    Returns:
        One :class:`LeverCoverageInput` per reportable lever, sorted by lever name.
    """
    if not records:
        return []

    # k is per ARM: the runs repeating one arm pool their repeats into it.
    groups = (arms if arms is not None else _campaign_arms(runs, results_by_run, profile=profile)).groups()
    group_of_run = {run.id: key for key, members in groups.items() for run in members}
    k_by_arm = {key: _arm_repeats(members) for key, members in groups.items()}
    resolved_by_run = {run.id: _resolve_config(run, results_by_run.get(run.id, []), profile=profile) for run in runs}
    effective_by_run = {run_id: flat for run_id, (flat, _engaged) in resolved_by_run.items()}
    # The launch-named and observation-recovered halves of "engaged", carried out of the same pass
    # that resolved the levels rather than re-derived: `overridden` is written both by a family
    # member and by a fixed declaration's own reader, so provenance cannot tell them apart.
    engaged = {name for _flat, names in resolved_by_run.values() for name in names}
    # The lever set comes from the registry, through `_effective_config`, and never from
    # `record.factors`. `factors` is the raw launch-overlay flattening in one host's carrier
    # spelling, so drawing the set from it re-created the second vocabulary that made a
    # declared axis unmatchable to its own coverage row.
    resolved = {lever for config in effective_by_run.values() for lever in config}
    # Observation is the primary source for the candidate model, and the records are its
    # floor: a result carrying no `candidate` usage row recovers nothing, yet every record
    # still knows which model produced it. `resolved` is a set, so the two sources name one
    # coverage row, never two.
    if len({record.model for record in records}) > 1:
        resolved.add(_CANDIDATE_MODEL_LEVER)
    lever_levels = _lever_levels(runs, results_by_run, profile=profile)
    all_run_ids = [run.id for run in runs]
    levers = _reportable_levers(resolved, records, effective_by_run, declared_design, engaged)

    coverage: list[LeverCoverageInput] = []
    declared_axes = {axis.axis_id: axis for axis in declared_design.axes} if declared_design else {}
    declared = set(declared_axes)
    for lever in sorted(levers):
        # Every number below is read over this lever's own cohort, which under a designated
        # control is usually narrower than the campaign — reporting a campaign-wide n or spread
        # beside a control-vs-cell contrast would describe a comparison that was never made.
        # Usually, not always: a lever no cell moved stays campaign-wide, and cohort_scope below
        # is what tells the reader which of the two this row is.
        cohort = set(_lever_cohort(lever, design, all_run_ids))
        if lever not in declared and folds.folds_away(lever, cohort):
            continue
        # A cohort with no records is emitted, not skipped: the lever exists in this campaign,
        # and dropping its row would tell the generator it does not. It comes out at cells=0
        # / n=0 / unswept, which is the honest reading — nothing here measured it.
        cohort_records = [record for record in records if record.run_id in cohort]
        composite_records = [r for r in cohort_records if r.metric == METRIC_COMPOSITE and r.value is not None]
        n_distinct_results = len({record.result_id for record in cohort_records})
        levels = sorted(
            {level for record in cohort_records if (level := _lever_value(record, lever, effective_by_run)) is not None}
        )
        cells = len(levels)
        # The same records the levels were read from, so a mechanism is compared over exactly the
        # levels and cohort every other number on the row describes.
        result_ids_by_level: dict[str, set[str]] = {}
        for record in cohort_records:
            if (level := _lever_value(record, lever, effective_by_run)) is not None:
                result_ids_by_level.setdefault(level, set()).add(record.result_id)
        k = _lever_k_floor(lever, cohort_records, k_by_arm, group_of_run, effective_by_run)
        # A declared axis is asked the authoring gate's own question. A campaign stored before that
        # gate, or past it, can still declare an apparatus or label input, and its row is then
        # `unswept` for a reason no run could change — said here, in the host's words, so the memo
        # can state the cause rather than report a sweep that never happened (#675).
        controllable = profile.controllable(lever) if lever in declared else None
        if cells <= 1:
            status: Literal["measured", "thin", "unswept"] = "unswept"
        elif k < _MEASURED_K_FLOOR or n_distinct_results < 2 * cells:
            status = "thin"
        else:
            status = "measured"
        coverage.append(
            LeverCoverageInput(
                name=lever,
                levels=levels,
                cells=cells,
                k=k,
                n=n_distinct_results,
                dispersion=_within_level_dispersion(composite_records, lever, effective_by_run),
                status=status,
                # Read from the same predicate the cohort was built with, never inferred from
                # its size: a single-lever star has every cell moving the one lever, so the
                # cohort spans every run while still being the contrast the campaign exists to
                # draw. Sizing it would label that row 'campaign' and tell the generator, two
                # paragraphs after "compare each cell to the control", that no contrast exists.
                cohort_scope="control_referenced" if _is_control_referenced(lever, design) else "campaign",
                declared_levels=(
                    _declared_level_coverage(declared_axes[lever], runs, results_by_run, profile=profile)
                    if lever in declared_axes
                    else []
                ),
                cannot_be_an_arm=(
                    controllable.reason if controllable is not None and controllable.state != "covered" else None
                ),
                confounded_by=_uncontrolled_dimensions(
                    lever, sorted(cohort), lever_levels, apparatus_levels, folds=folds, profile=profile
                )
                + _observed_mechanism_confounds(lever, result_ids_by_level, observations, profile=profile)
                + _served_model_confounds(chain.from_iterable(result_ids_by_level.values()), served),
                mechanism=_mechanism_check(
                    acts_on := profile.sweepables.acts_on(lever),
                    result_ids_by_level,
                    observations,
                    # The mechanism measure's declared range, which a shift of every case alike is read on.
                    describe_measure(acts_on, profile.measures).value_range if acts_on is not None else None,
                ),
            )
        )
    return coverage


def _factor_aliasing(
    run_ids: list[str],
    lever_levels: dict[str, dict[str, list[str]]],
    apparatus_levels: dict[str, dict[str, str | None]],
) -> tuple[list[AliasedFactors], FactorPairScan]:
    """Group the factors that moved in lockstep, and check every pair of factors for co-varying.

    A factor is anything that varied across the campaign's runs: a swept lever or the candidate model, as
    :func:`_lever_levels` reads them, and an apparatus dimension every run recorded at two or more levels (one some
    run never recorded is undecided, not a partition, and is named in ``apparatus_confounds``). Each factor
    splits the runs it was read on into groups, one per level. **Factors whose splits are identical are aliased**
    — the same runs, grouped the same way, whatever the levels are called — and are reported as one group.

    A pair **co-varies** when neither can be compared holding the other fixed: across the runs at any one level
    of one, the other takes a single level. Every pair is checked; a co-varying pair whose factors are not in one
    group gets a pivot crossing the two (a group stands in for each of its members, so its mates share one
    pivot), with every combination no run sat at as ``not_run``. Only interactions go unchecked
    (:data:`INTERACTION_ALIASING_UNCHECKED`).

    Args:
        run_ids: The campaign's resolved runs.
        lever_levels: Lever → level → run ids, from :func:`_lever_levels`.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.

    Returns:
        The lockstep groups (sorted by their first factor), and the pair scan.
    """
    by_factor: dict[str, dict[str, str]] = {}
    for lever, by_level in lever_levels.items():
        by_factor[lever] = {run_id: level for level, members in by_level.items() for run_id in members}
    for dimension, by_run in apparatus_levels.items():
        levels = [by_run.get(run_id) for run_id in run_ids]
        if dimension in by_factor or any(level is None for level in levels) or len(set(levels)) < 2:
            continue
        by_factor[dimension] = {run_id: str(by_run[run_id]) for run_id in run_ids}
    factors = sorted(name for name, by_run in by_factor.items() if len(set(by_run.values())) >= 2)

    def split(name: str) -> frozenset[frozenset[str]]:
        blocks: dict[str, set[str]] = {}
        for run_id, level in by_factor[name].items():
            blocks.setdefault(level, set()).add(run_id)
        return frozenset(frozenset(block) for block in blocks.values())

    by_split: dict[frozenset[frozenset[str]], list[str]] = {}
    for name in factors:
        by_split.setdefault(split(name), []).append(name)
    group_of = {name: members[0] for members in by_split.values() for name in members}
    groups = []
    for blocks, members in sorted(by_split.items(), key=lambda item: item[1][0]):
        if len(members) < 2:
            continue
        n_runs = sum(len(block) for block in blocks)
        across = f"all {n_runs} runs" if n_runs == len(run_ids) else f"the {n_runs} runs that recorded them"
        groups.append(
            AliasedFactors(
                factors=members,
                n_runs=n_runs,
                n_levels=len(blocks),
                sentence=(
                    f"{_listed_names(members)} move together across {across}, splitting them into the same "
                    f"{len(blocks)} groups; no comparison separates them, so a difference across them belongs to all "
                    f"{len(members)} at once."
                ),
            )
        )

    def varies_within(name: str, other: str) -> bool:
        shared = by_factor[name].keys() & by_factor[other].keys()
        seen: dict[str, set[str]] = {}
        for run_id in shared:
            seen.setdefault(by_factor[other][run_id], set()).add(by_factor[name][run_id])
        return any(len(levels) >= 2 for levels in seen.values())

    n_pairs = n_covarying = n_in_groups = 0
    pivots: dict[tuple[str, str], FactorPairPivot] = {}
    for index, row in enumerate(factors):
        for column in factors[index + 1 :]:
            n_pairs += 1
            if varies_within(row, column) and varies_within(column, row):
                continue
            n_covarying += 1
            if group_of[row] == group_of[column]:
                n_in_groups += 1
                continue
            first, second = sorted((group_of[row], group_of[column]))
            key = (first, second)
            if key not in pivots:
                pivots[key] = _factor_pair_pivot(key[0], key[1], by_factor, group_of)
    ordered = [pivots[key] for key in sorted(pivots)]
    kept, omitted = _capped(
        ordered,
        _MAX_FACTOR_PAIR_PIVOTS,
        weight=lambda pivot: sum(1 for cell in pivot.cells if cell.status == "not_run"),
    )
    outside = n_covarying - n_in_groups
    if len(factors) < 2:
        completeness = f"{len(factors)} factor varied, so no pair of factors could co-vary."
    else:
        completeness = (
            f"{n_pairs} factor pair(s) examined over the {len(factors)} factors that varied; {n_covarying} co-vary"
            + (f", {n_in_groups} of them inside a group that moves in lockstep" if n_in_groups else "")
            + (
                f"; the {outside} outside any group are shown in {len(ordered)} pivot(s), a group's members sharing one"
                if outside
                else ""
            )
            + (f", of which {omitted} with the fewest unrun combinations are left out" if omitted else "")
            + "."
        )
    return groups, FactorPairScan(
        factors=factors,
        n_pairs_examined=n_pairs,
        n_covarying=n_covarying,
        n_covarying_in_groups=n_in_groups,
        pivots=kept,
        pivots_omitted=omitted,
        completeness=completeness,
    )


def _factor_pair_pivot(
    row: str,
    column: str,
    by_factor: dict[str, dict[str, str]],
    group_of: dict[str, str],
) -> FactorPairPivot:
    """Cross two factors' observed levels over the runs both were read on, counting the runs at each combination."""
    shared = sorted(by_factor[row].keys() & by_factor[column].keys())
    counts: dict[tuple[str, str], int] = {}
    for run_id in shared:
        combination = (by_factor[row][run_id], by_factor[column][run_id])
        counts[combination] = counts.get(combination, 0) + 1
    rows = sorted({by_factor[row][run_id] for run_id in shared})
    columns = sorted({by_factor[column][run_id] for run_id in shared})
    mates = {name: [other for other in group_of if group_of[other] == name and other != name] for name in (row, column)}
    return FactorPairPivot(
        row_factor=row,
        column_factor=column,
        row_aliases=sorted(mates[row]),
        column_aliases=sorted(mates[column]),
        cells=[
            FactorPairCell(
                row_level=row_level,
                column_level=column_level,
                n_runs=counts.get((row_level, column_level), 0),
                status="ran" if (row_level, column_level) in counts else "not_run",
            )
            for row_level in rows
            for column_level in columns
        ],
    )


def _listed_names(names: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


#: Apparatus dimensions that joined the rig after cells were minted under ids that never digested them, each
#: mapped to the dimension whose seat it shares. Such a dimension stays out of a class's id at the levels that
#: say nothing about it the class does not already say: UNRECORDED (``None``) — every run stored before the
#: dimension existed — or the unseated level where its owner reads unseated too. At any recorded level it is
#: digested like every other dimension. So a stored run's cell keeps the id a stored analysis cites, and a run
#: that recorded the dimension gets a cell of its own, which never pools with the unrecorded one: the class
#: still lists the dimension (``unknown_dimensions``), so the merge rule refuses the pair, and the confound scan
#: reads it ``undecided``.
#:
#: **Why no two different classes can share an id.** Within one bundle every class is built over one dimension
#: set, so a class's unknown set is fixed by its recorded map, and two classes the id cannot tell apart differ
#: only in this dimension's level, which is neutral in both. Unrecorded beside unrecorded is the same class.
#: Unrecorded beside unseated cannot happen with the owner agreeing: unseated here needs the owner unseated
#: (the condition below), while unrecorded here means the run filled the seat, so its owner reads a recorded or
#: an unrecorded level, never unseated — the owner's own level tells the two classes apart. A dimension's
#: unseated level paired with a recorded owner (a run that filled no judge seat yet recorded a judge, which
#: :meth:`~threetears.evals.kernel.host.profile.HostProfile.omits_apparatus` reports as a contradiction) is
#: therefore digested, not neutral.
CELL_ID_NEUTRAL: Mapping[str, str] = {"judge_temperature": "judge_model"}


def _cell_id_neutral(run_id: str, apparatus_levels: dict[str, dict[str, str | None]]) -> frozenset[str]:
    """The :data:`CELL_ID_NEUTRAL` dimensions this run's class id leaves out, at the levels where it says nothing new.

    Args:
        run_id: The run.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.

    Returns:
        The dimensions to leave out of the run's class id; empty when every one is recorded, or absent from the
        bundle's apparatus altogether.
    """
    unseated = canonical_json(UNSEATED_LEVEL)
    neutral: set[str] = set()
    for dimension, owner in CELL_ID_NEUTRAL.items():
        if dimension not in apparatus_levels:
            continue
        level = apparatus_levels[dimension].get(run_id)
        if level is None or (level == unseated and apparatus_levels.get(owner, {}).get(run_id) == unseated):
            neutral.add(dimension)
    return frozenset(neutral)


def _apparatus_classes(
    runs: list[EvalRun],
    apparatus_levels: dict[str, dict[str, str | None]],
) -> dict[str, ApparatusClass]:
    """Classify each run's rig, so every observation it carries shares one class.

    A launching host declares its apparatus at launch, so a batch's observations were all measured
    under the same rig and reading it per run is exact rather than an approximation. A host
    whose apparatus genuinely moves within a batch supplies it per observation instead — the
    :class:`~threetears.evals.analysis.cells.Observation` carries the coordinate, and nothing
    downstream can tell which route produced it.

    Args:
        runs: The campaign's resolved runs.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`. A
            ``None`` level means the run never recorded that dimension.

    Returns:
        Run id → the class that run's observations belong to.
    """
    dimensions = set(apparatus_levels)
    return {
        run.id: apparatus_class_of(
            {dim: apparatus_levels.get(dim, {}).get(run.id) for dim in dimensions},
            dimensions=dimensions,
            id_neutral=_cell_id_neutral(run.id, apparatus_levels),
            # Read off the run, never assumed: the launch path stamps `commissioned`, and a host
            # capturing traffic it did not control writes `witnessed`. It enters the class id, so a
            # captured session beside a launched arm of the same variant is two cells everywhere.
            provenance=run.apparatus_provenance,
        )
        for run in runs
    }


def _observations(
    runs: list[EvalRun],
    results_by_run: dict[str, list[EvalResult]],
    classes: dict[str, ApparatusClass],
    *,
    scope_id: str,
    profile: HostProfile,
) -> tuple[list[Observation], list[VariantIndexEntry]]:
    """Turn this campaign's results into the observations the cell algebra pools.

    One observation per result, because that is where the variant resolves: a run carries
    several candidate models and the candidate is part of what the variant IS, so a run-level
    observation would pool contestants that never competed under one key.

    The key is the one the runner stamped on the result — every result carries one, since the
    engine resolves every run's candidate model and kind into its variant map.

    Args:
        runs: The campaign's resolved runs, in bundle order.
        results_by_run: Each run's results, keyed by run id.
        classes: Run id → apparatus class, from :func:`_apparatus_classes`.
        scope_id: The scope these were read under.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(observations, variant_index)`` — the observations in ``(run, result id)`` order, and
        one index entry per variant, carrying the lever map its key was
        digested from. The index is built HERE rather than beside the cells because this is the
        one place that holds both halves at once: a cell knows its variant key and nothing
        about the run and candidate model the levels came from.
    """
    observations = []
    index: dict[str, VariantIndexEntry] = {}
    for run in runs:
        for result in results_by_run.get(run.id, []):
            key, levers, unavailable = _variant_key_of(result, run, profile=profile)
            if key not in index:
                # An arm whose levels cannot be described is indexed anyway, saying why. It was
                # measured and it pools; what nothing today can supply is a DESCRIPTION of it.
                # Leaving it out of the index instead is what made such an arm vanish: every
                # consumer of the index — the arm table, the generator prompt, the coverage gate —
                # reads arms from here, so an unindexed arm is one no surface can report and no
                # reader can miss.
                index[key] = VariantIndexEntry(variant_key=key, levers=levers, levels_unavailable=unavailable)
            observations.append(
                Observation(
                    id=result.id,
                    scope_id=scope_id,
                    variant_key=key,
                    apparatus_class_id=classes[run.id].apparatus_class_id,
                    # The batch this was commissioned under. Provenance for a reader, never an
                    # input to the cell — two observations from different batches with identical
                    # apparatus are one cell, which is the whole point of pooling across them.
                    apparatus_ref=run.id,
                    case_ref=result.test_case_id,
                )
            )
    return observations, sorted(index.values(), key=lambda entry: entry.variant_key)


def _name_arms(
    index: list[VariantIndexEntry],
    observations: list[Observation],
    folds: _SurfaceFolds,
    run_ids: Collection[str],
) -> list[VariantIndexEntry]:
    """Name each arm by the knobs it swept wherever its resolved surface folds away.

    An arm's ``levers`` are its key's pre-image, and a host that registers an open family's
    resolved surface as a lever puts that surface there rather than the members — so every arm of
    a one-key tool-config sweep carried one opaque surface hash and no mention of the key it
    swept, and an arm table built from it rendered identical rows and could place none of the
    memo's swept-knob coordinates. The same fold every other lens asks decides it here: where the
    surface's residual agrees across the campaign, the members name the arm and the surface is
    recorded as ``folded``; where it does not, the surface moved on its own and stays the arm's
    name, exactly as it stays a moved lever and a confound elsewhere. A surface a fixed knob is
    written into folds the same way and adds nothing to ``swept``: the knob is already one of the
    arm's ``levers``, and names it once the surface is set aside.

    **Campaign-wide, because an arm's name is read against every other arm.** A surface explained
    only within some contrasts would name some arms by their members and others by an opaque hash,
    and the table would compare two vocabularies.

    Args:
        index: The variant index, one entry per keyed variant.
        observations: The observations the index was built from — each names its variant and the
            run it came from, which is what says which runs carried an arm.
        folds: The campaign's :class:`_SurfaceFolds`.
        run_ids: Every resolved member run — the cohort a name is read against.

    Returns:
        The index, each entry carrying ``swept`` and ``folded`` where a surface folds. Entries whose
        levels are unavailable, and every entry when nothing folds, are returned as they were.
    """
    explained = sorted(surface for surface in folds.surfaces if folds.folds_away(surface, run_ids))
    if not explained:
        return index
    runs_of: dict[str, dict[str, None]] = {}
    for observation in observations:
        # An inline-apparatus observation names no run, and so contributes no run's overlays.
        if observation.apparatus_ref is not None:
            runs_of.setdefault(observation.variant_key, {})[observation.apparatus_ref] = None
    named = []
    for entry in index:
        swept: dict[str, SweepableValue] = {}
        folded: list[str] = []
        for surface in explained:
            if surface not in entry.levers:
                continue
            members = folds.swept_members(surface, runs_of.get(entry.variant_key, {}))
            folded.append(surface)
            swept.update(
                {member: SweepableValue.of(value, display=lever_level(value)) for member, value in members.items()}
            )
        if not folded:
            named.append(entry)
            continue
        named.append(
            VariantIndexEntry(
                variant_key=entry.variant_key,
                levers=entry.levers,
                levels_unavailable=entry.levels_unavailable,
                swept=dict(sorted(swept.items())),
                folded=folded,
            )
        )
    return named


def _declared_level_names(index: list[VariantIndexEntry], declared: CampaignDesign | None) -> list[VariantIndexEntry]:
    """Name each arm's levels by what the campaign declared them as, where it declared them.

    A host displays a level by what it can see of it, and for a long text — a prompt in a prompt sweep — that is a
    fingerprint (``cognitive_style: 2304 chars · 539ef3``), so two arms that differ only in their text read as two
    digests. A campaign that declared the level (``SweptAxis.values``) said what to call it: "current text",
    "optimised text". That name replaces the host's display on every entry carrying the level, so every surface
    reading the index — the arm names, the report's tables and charts, and the writer's arm list — prints it.

    **Joined on ``(axis_id, content_hash)``, never on a display**: the hash is the level's identity, and a display
    is what is being replaced. The first declaration wins where one hash is declared twice on one axis, as an
    author reading the declaration top to bottom would expect. A level with no declared name keeps the host's
    display, and so does a lever's "not a run of this kind" level, which is the engine's and no declaration's.
    Only ``display`` changes: the variant key digests content hashes alone, so no key moves and no cell regroups.

    Args:
        index: The variant index, its arms already named by :func:`_name_arms`.
        declared: The campaign's declaration, or None when it declared nothing.

    Returns:
        The index, each declared level displayed by its declared name; entries no declaration names unchanged.
    """
    if declared is None:
        return index
    names: dict[tuple[str, str], str] = {}
    for axis in declared.axes:
        for value in axis.values:
            names.setdefault((axis.axis_id, value.content_hash), value.display)

    def named(levers: dict[str, SweepableValue]) -> dict[str, SweepableValue]:
        return {
            axis: level.model_copy(update={"display": name})
            if level.not_of_kind is None and (name := names.get((axis, level.content_hash))) is not None
            else level
            for axis, level in levers.items()
        }

    renamed: list[VariantIndexEntry] = []
    for entry in index:
        levers, swept = named(entry.levers), named(entry.swept)
        if levers == entry.levers and swept == entry.swept:
            renamed.append(entry)
            continue
        renamed.append(
            VariantIndexEntry(
                variant_key=entry.variant_key,
                levers=levers,
                levels_unavailable=entry.levels_unavailable,
                swept=swept,
                folded=entry.folded,
            )
        )
    return renamed


def _variant_key_of(
    result: EvalResult, run: EvalRun, *, profile: HostProfile
) -> tuple[str, dict[str, SweepableValue], str | None]:
    """This result's variant key, and the lever map that key was digested from.

    The key is the one the runner STAMPED on the result. The map behind it comes from the run's RECORDED
    pre-image when the run carries one, and only from a fresh derivation for a run its host
    assembled without the launch — see
    :func:`~threetears.evals.kernel.identity.resolve_variant_identity`, which owns that choice. The
    difference is the whole of this function's behaviour across an
    :data:`~threetears.evals.kernel.identity.IDENTITY_VERSION` bump: a recorded map needs no predicate to
    read it, so the arm stays described, while a derivation replays today's predicate and cannot.

    Args:
        result: The observation.
        run: The batch it belongs to.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(key, levers, unavailable)``. ``unavailable`` is the reason ``levers`` is empty, or ``None`` when it is empty because the
        host genuinely resolved no levers. **An empty map has two causes and they are not the same
        fact** — a host that registers no levers describes its arms exactly, with nothing; a key
        nothing here can place describes its arm not at all. Deciding that here, where
        ``reproduces`` is already known, is what stops each caller re-deriving it from the shape of
        the map and getting it wrong in its own way. :func:`_undescribable_arm_reason` says which
        of the two undescribable cases it is, in the numbers that tell an operator what to do.

    Raises:
        LeverCoordinateError: The host's variant map disagrees with its own registry.
            Reachable only on the DERIVED path — a run carrying its own pre-image is never
            checked against today's registry, because the registry says what a lever is now and
            the map records what one was. On that path it is loud: a campaign over a mis-registered
            host fails assembly rather than assembling with no coordinates, and the same registry
            would refuse every new observation anyway.

    Note:
        ``levers`` is the map the returned key was digested from, and is EMPTY with
        ``unavailable`` set for a stamped key nothing here can place — a result from a run that
        recorded no pre-image, read by a build whose predicate no longer reproduces the key. Its
        stored key stays authoritative for pooling (that is what it was measured under), and
        today's levels are not a description of it: indexing them would label the arm with a
        stack it never ran. The arm is still INDEXED, carrying that reason instead of levels — an
        arm that pooled and then appeared in no index was one the arm table, the generator prompt
        and the coverage gate alike could not see, which is a silence rather than a disclosure.
    """
    resolved = resolve_variant_identity(run=run, profile=profile)
    if resolved.variant_key == result.variant_key:
        return result.variant_key, resolved.levers, None
    return result.variant_key, {}, _undescribable_arm_reason(result, run)


def _undescribable_arm_reason(result: EvalResult, run: EvalRun) -> str:
    """Say why this arm's stamped key has no lever map behind it, in numbers an operator can act on.

    The two causes need different actions and the text is the only channel that carries the
    difference out to the arm table and to the pre-generation refusal that reads it. A run that
    recorded a map which does not digest to this key is a broken record; a run that recorded none
    (its host assembled it without the launch) is being read by a predicate other than the one it
    ran under, and re-running the campaign on this build fixes it.

    The version NUMBERS are what make the second actionable. "A predicate this build does not
    reproduce" is true of a run measured this morning across one bump and of a run from a year
    ago, and the remedy differs entirely.

    Args:
        result: The observation whose stamped key cannot be described.
        run: The batch it belongs to.

    Returns:
        The reason, for :attr:`~threetears.evals.kernel.campaign.VariantIndexEntry.levels_unavailable`.
    """
    if run.variant_levers is not None:
        return (
            "the run recorded a lever map for this candidate and it does not digest to the key stamped on this "
            "observation, so it is not this arm's pre-image and describing the arm with it would label it with a "
            "stack it never ran"
        )
    return (
        f"the key was stamped under identity predicate v{result.identity_version}, this build derives "
        f"v{IDENTITY_VERSION}, which does "
        f"not reproduce it, and the run recorded no lever map for this candidate — so nothing here can say what the "
        f"arm ran"
    )


def variant_key_of_run(results: Sequence[EvalResult]) -> str | None:
    """Which variant a run's observations carried — the authoring side of the control.

    A control is a variant key and nobody types a sha256, so the authoring surface points at a
    run and this addresses it. A run is one arm, so it names one variant.

    Args:
        results: One run's results, which carry the key the runner stamped on each.

    Returns:
        The variant key, or ``None`` for a run with no results yet — a run with no observation
        is in no arm.

    Note:
        **The key is taken from the first result.** Every result of a run is stamped from the one
        lever map the run carries, so two results differ in their stamped key only if they were
        written under different ``IDENTITY_VERSION``s, which a single runner pass cannot produce.
        Stated rather than defended against, because the check would cost every bundle assembly a
        comparison for a state no writer can reach — if one ever can (a backfill that re-stamps part
        of a run, say), this is where it would go unnoticed.
    """
    return results[0].variant_key if results else None


def _launch_disclosure(runs: Collection[EvalRun]) -> str | None:
    """Say how the campaign's runs were started, when they were not all started as one launch.

    Args:
        runs: The resolved member runs.

    Returns:
        The sentence, or None when every run shares one launch group or there are fewer than two.
    """
    if len(runs) < 2:
        return None
    groups = {run.launch_group_id for run in runs if run.launch_group_id is not None}
    alone = sum(1 for run in runs if run.launch_group_id is None)
    if alone == 0 and len(groups) == 1:
        return None
    parts = []
    if groups:
        parts.append(f"{len(groups)} separate launch{'es' if len(groups) != 1 else ''}")
    if alone:
        parts.append(f"{alone} run{'s' if alone != 1 else ''} started on {'their' if alone != 1 else 'its'} own")
    return (
        f"These {len(runs)} runs were not started as one launch ({' and '.join(parts)}), so arms from different "
        "starts were measured side by side only where their windows happen to overlap: a difference between "
        "them can come from when they ran as well as from their settings."
    )


def assemble_context_bundle(
    campaign: EvalCampaign,
    *,
    storage: CampaignReadStore,
    insights_as_of: str | None = None,
    profile: HostProfile,
) -> AnalysisContextBundle:
    """Assemble the closed context bundle for a campaign's runs.

    Loads the campaign's member runs + their results from the campaign's own
    ``scope_id`` — the only scope its members can live in — composes the ``reporting`` lenses
    over them, derives the coverage map and telemetry rollup, pulls the subject's
    prior insights — less any an archived analysis minted, which are named in
    ``retracted_insights`` instead — and returns a deterministic, fingerprintable bundle. Runs that
    don't resolve in the scope are reported in ``unresolved_run_ids``, and runs an
    operator archived in ``archived_run_ids``, rather than either being silently
    dropped. **Nothing fetches** beyond these reads.

    Archived runs are held out of every lens because this bundle is what a paid
    generation reads: a run archived precisely because its observation is junk —
    cancelled mid-flight, or produced by an apparatus since found broken — would
    otherwise enter each aggregate with nothing marking it, which is the failure the
    archive surface exists to end.

    Args:
        campaign: The campaign whose runs to summarise.
        storage: Eval storage — the read surface for runs, results, and insights.
        insights_as_of: Read the subject's prior insights as the ledger stood at this
            instant — only those observed strictly before it. ``None`` reads the whole
            ledger, which is what a generation launched now reads. A re-assembly FOR a stored
            analysis passes the instant that analysis's bundle was assembled: the generation
            fingerprints its bundle before it saves the insights it mints, and another analysis can
            mint during the provider call, so without the cutoff an analysis would re-assemble with
            insights it never read and could not reproduce itself. Compared as the ISO-8601 strings both stamps are written as.
            Retraction composes with it rather than being dated by it: of the insights read as of
            this instant, those whose analysis is archived NOW are retracted, because an archive
            says the insight was false when it was read too.
        profile: The host whose vocabulary this reads.

    Returns:
        A schema-valid, JSON-serializable :class:`AnalysisContextBundle`.
    """
    scope_id = campaign.scope_id
    # Resolve member runs, keeping an honest record of what did not resolve and what
    # was deliberately excluded — two different facts, kept in two different lists.
    #
    # One batch read, with the host's listing elisions: the bundle holds every member run at once
    # and reads none of what a host declares a listing may leave out — the elided runs carry the
    # mark that makes the one reader of it refuse.
    loaded = {
        run.id: run
        for run in storage.load_eval_runs(campaign.run_ids, scope_id, elide_payload=profile.listing_elisions)
    }
    resolved: list[EvalRun] = []
    unresolved: list[str] = []
    archived: list[str] = []
    for run_id in campaign.run_ids:
        run = loaded.get(run_id)
        if run is None:
            unresolved.append(run_id)
        elif run.archived:
            archived.append(run_id)
        else:
            resolved.append(run)

    # Deterministic input order → deterministic lens output → stable fingerprint.
    runs = sorted(resolved, key=lambda r: (r.created_at, r.id))
    # Before any lens reads a descriptor: each host measure is read as its runs were launched to read it, not as
    # whoever reads the campaign now declares it, so every lens below reads one set of declarations.
    profile, launch_declarations = _as_launched(runs, profile)
    run_ids = [run.id for run in runs]
    known_run_ids = set(run_ids)

    results_by_run: dict[str, list[EvalResult]] = {}
    for run in runs:
        run_results = storage.query_eval_results_by_run(run.id, scope_id)
        results_by_run[run.id] = sorted(run_results, key=lambda r: r.id)
    # Before anything reads them: a classifier's failure is a miss in every rate, not absent from them all.
    results_by_run = _failures_as_misses(results_by_run)
    # Before anything reads them either: latency read under concurrency is left out of every lens, never
    # compared as if it were clean (#701). Which results lost it is kept for the one line that says so.
    contended_ids = {
        result.id
        for result in withheld_latency(
            (result for run_results in results_by_run.values() for result in run_results), profile.measures
        )
    }
    results_by_run = {
        run_id: withhold_contended_latency(run_results, profile.measures)
        for run_id, run_results in results_by_run.items()
    }
    results = [result for run in runs for result in results_by_run[run.id]]

    # ``archived_run_ids=None`` is true here, not a default: archived members were removed
    # above, so ``runs`` is the whole corpus these surfaces read and none of it is archived.
    projection = project_score_records(
        runs, results, known_run_ids=known_run_ids, archived_run_ids=None, profile=profile
    )
    budget = compute_program_budget(runs, results)
    # What the campaign was not tuning, read once and shared by all three surfaces that
    # disclose it — the coverage map (where most findings are formed), the divergences, and
    # the campaign-wide scan that answers even when neither of those exists.
    apparatus_levels = _apparatus_levels(runs, results_by_run, profile=profile)
    # Derived before the lenses, because it decides which runs each of them compares.
    control_variant = campaign.declared_design.control if campaign.declared_design else None
    # Read only when a declared control might have been curated out, and only over the archived
    # set: "the runs carrying your control were archived" and "nothing ever ran it" are different
    # operator actions, and the archived runs' results are the only place the difference lives.
    # Every other path pays nothing for the distinction.
    archived_control_variants: set[str] = set()
    if control_variant and archived:
        for run_id in archived:
            archived_key = variant_key_of_run(storage.query_eval_results_by_run(run_id, scope_id))
            if archived_key is not None:
                archived_control_variants.add(archived_key)
    # One answer, shared by every lens, to "did a resolved surface move on its own" — the design's
    # contrasts, the coverage rows and their confounds, and the divergences all ask it over their
    # own cohorts, and two lenses answering it separately could disagree about one surface.
    folds = _SurfaceFolds(runs, results_by_run, profile=profile)
    arms = _campaign_arms(runs, results_by_run, profile=profile)
    # Every mechanism a lens compares across levels, read once so the coverage rows, the divergences
    # and the arm readings report one set of values.
    mechanisms = _mechanism_observations(results, profile=profile)
    # Which model answered each result's candidate calls, read once so every lens that names the served
    # model as a confound and the per-arm readings agree about it.
    served = _served_models(results)
    design = _campaign_design(
        runs,
        control_variant,
        results_by_run=results_by_run,
        arms=arms,
        archived_control_variants=archived_control_variants,
        has_unresolved_members=bool(unresolved),
        folds=folds,
        profile=profile,
    )
    design = _design_with_mechanism_confounds(design, results_by_run, mechanisms, served=served, profile=profile)
    # Built before the run summaries, because `RunSummary.config` names the levers this map
    # names. Both read `_effective_config`, so without that the two disagreed the moment the
    # registry became the vocabulary: a summary would carry every contestant property the host
    # declares — ten content hashes apiece, each stamped `overridden`, telling the design lens
    # that every run departed from ten things it never touched.
    coverage = _coverage_map(
        runs,
        projection.records,
        apparatus_levels,
        design,
        results_by_run,
        campaign.declared_design,
        folds=folds,
        observations=mechanisms,
        served=served,
        arms=arms,
        profile=profile,
    )
    reportable = {entry.name for entry in coverage}
    aliasing = _factor_aliasing(run_ids, _lever_levels(runs, results_by_run, profile=profile), apparatus_levels)

    # The cell algebra, over observations rather than runs. Independent of `design`, which is
    # the point: a cell is what pools, and pooling must not depend on whether anyone designated
    # a control.
    apparatus_classes = _apparatus_classes(runs, apparatus_levels)
    observations, variant_index = _observations(
        runs, results_by_run, apparatus_classes, scope_id=scope_id, profile=profile
    )
    variant_index = _name_arms(variant_index, observations, folds, run_ids)
    variant_index = _declared_level_names(variant_index, campaign.declared_design)
    cells, refused_merges, next_experiments = pool_observations(
        observations, {c.apparatus_class_id: c for c in apparatus_classes.values()}
    )
    # Capped like the divergences, keeping the entries that bear on the most evidence: a refusal by the
    # observations its two cells hold, a recording by the observations it would add.
    held = {(cell.variant_key, cell.apparatus_class_id): cell.n_observations for cell in cells}
    refused_merges, refused_merges_omitted = _capped(
        refused_merges,
        _MAX_REFUSED_MERGES,
        weight=lambda merge: sum(held.get((merge.variant_key, class_id), 0) for class_id in merge.apparatus_class_ids),
    )
    next_experiments, next_experiments_omitted = _capped(
        next_experiments,
        _MAX_NEXT_EXPERIMENTS,
        weight=lambda entry: entry.n_observations_if_recorded - entry.n_observations_now,
    )

    # Only the runs that RESOLVED a span contribute, exactly as runs_compare's do: a run
    # that produced nothing cannot say when it was measured, and letting that absence count
    # would report a difference on the strength of what one run could not say.
    # Sorted once, here, so the bundle's structured spans and the sentence rendered from them
    # are the same list read twice rather than two orderings of one fact.
    windows = sorted(
        (window for run in runs if (window := measurement_window(run.id, results_by_run[run.id])) is not None),
        key=lambda w: (w.start, w.end, w.run_id),
    )

    # The cutoff first, then retraction over what survives it: an insight the generation could not
    # have read is not one it was spared, so it is neither carried nor named as retracted.
    ledger = [
        insight
        for insight in storage.query_insights(scope_id, subject_id=campaign.subject_id)
        if insights_as_of is None or insight.observed_at < insights_as_of
    ]
    retracted = retracted_insights(ledger, lambda analysis_id: storage.analysis_archived(analysis_id, scope_id))
    prior_insights, prior_insights_omitted = _prior_insights(
        [insight for insight in ledger if insight.id not in retracted]
    )

    # The campaign's effective bar on the measure the frontier ranks on, or why it takes none.
    frontier_bar, frontier_bar_withheld = _frontier_bar(campaign.behavior, campaign.declared_design, profile=profile)
    bundle = AnalysisContextBundle(
        campaign_id=campaign.id,
        subject_id=campaign.subject_id,
        subject_kind=campaign.subject_kind,
        behavior=campaign.behavior,
        template_id=campaign.template_id,
        scope_id=scope_id,
        run_ids=run_ids,
        unresolved_run_ids=unresolved,
        archived_run_ids=archived,
        model_versions=_model_versions(runs, projection.records),
        window=derive_window([run.created_at for run in runs]),
        run_summaries=[_run_summary(run, results_by_run[run.id], reportable, profile=profile) for run in runs],
        # WITH the results, because the pinned-role badge reads result-level declarations
        # (``judge_config_ids`` is observed ACROSS a run's results, not declared on the run).
        # Called without them, this surface was blind to exactly the difference a judge A/B is
        # made of while the bundle beside it reported that difference from the same readers.
        comparison=compute_comparison_sets(runs, results=results, profile=profile),
        # pass^k at the behavior's declared threshold (#642), recorded on the frontier beside every figure,
        # and ranked against the campaign's bar on pass^k when it declares one (#679). Archived members were
        # removed before this point, so no archived set is passed (#670).
        frontier=compute_frontier(
            runs,
            results,
            bar=frontier_bar,
            known_run_ids=known_run_ids,
            archived_run_ids=None,
            rubric_threshold=profile.bars.pass_threshold(campaign.behavior),
            profile=profile,
            # The boundary pillar: each contestant's guardrail dimensions held against the campaign's control arm,
            # at the margins it declares, by the rule the bundle's guardrails are decided by (#613).
            control_variant_key=design.control_arm.variant_key if design.control_arm is not None else None,
            guardrail_margins=(
                {entry.dimension: entry.margin for entry in campaign.declared_design.guardrail_margins}
                if campaign.declared_design is not None
                else None
            ),
        ),
        frontier_bar_withheld=frontier_bar_withheld,
        telemetry=_telemetry_rollup(runs, results, budget, profile=profile),
        coverage=coverage,
        declared_design=campaign.declared_design,
        design=design,
        # A run that did not finish is still IN every aggregate above — archiving is what
        # removes a run, and a cancelled run that nobody archived stays pooled. Naming them
        # here is what lets the generator say a partial arm is partial instead of reading its
        # truncated case count as a real one.
        incomplete_runs={run.id: str(run.status) for run in runs if run.status != "completed"},
        # Status cannot answer how much of the matrix a run delivered: a run that reaches the
        # end of its loop stamps ``completed`` whatever its cells produced, so the shortfall is
        # read off the completeness record instead. Rendered through the shared helper so this
        # bundle and every run-level surface say the same sentence about the same run.
        short_runs={
            run.id: disclosure for run in runs if (disclosure := completeness_disclosure(run.completeness)) is not None
        },
        # Absence is not zero. The helper above returns None both for a run that delivered
        # everything and for one carrying no record at all, so the second case is named here
        # rather than folded into the first — reporting unknown as complete is the same error
        # one layer down.
        completeness_unknown_run_ids=[run.id for run in runs if run.completeness is None],
        short_cells=_short_cells(cells, campaign.declared_design),
        held_fixed_reading=_held_fixed_reading(runs, campaign.declared_design),
        # ``full`` because this bundle's reader is a model that cannot go and look:
        # above the inline cap the collapsed form names two spans and says where to
        # get the rest, which is an instruction only a human at a terminal can follow.
        # Campaigns routinely carry more runs than that cap.
        # The structured half, for surfaces that render windows from data. The sentence below is
        # what the model is handed to quote.
        measurement_windows=windows,
        measurement_window_disclosure=measurement_window_disclosure(windows, full=True),
        launch_disclosure=_launch_disclosure(runs),
        # Scanned over every resolved run, not over a lever's cohort. A campaign that swept
        # nothing has an empty coverage map and no divergences, and both confound-bearing
        # lenses iterate levers — so without this, a campaign whose template changed
        # underneath it reports that fact nowhere at all.
        apparatus_confounds=_apparatus_confounds(run_ids, apparatus_levels, profile=profile),
        aliased_factors=aliasing[0],
        factor_pairs=aliasing[1],
        declared_crossing=_declared_crossing(campaign.declared_design, runs, results_by_run, profile=profile),
        arm_mechanisms=_arm_mechanisms(arms, results_by_run, mechanisms),
        arm_served_models=_arm_served_models(arms, results_by_run, served),
        arm_production_footings=_arm_production_footings(arms, results_by_run, profile=profile),
        cells=cells,
        variant_index=variant_index,
        refused_merges=refused_merges,
        refused_merges_omitted=refused_merges_omitted,
        next_experiments=next_experiments,
        next_experiments_omitted=next_experiments_omitted,
        host_declarations_digest=host_declarations_digest(profile),
        # Read off the projection rather than the campaign, because the campaign states ONE
        # subject and this check exists to catch the case where the observations disagree with
        # that — a rename mid-campaign, or two subjects collided onto one key.
        subject_key_instabilities=subject_key_instabilities(
            (record.subject_id, record.subject_label) for record in projection.records
        ),
        prior_insights=prior_insights,
        prior_insights_omitted=prior_insights_omitted,
        retracted_insights=retracted,
        # Over the resolved members only, like every other lens: a rating of an archived run's result
        # calibrates a judge the bundle does not otherwise read. Ratings are read per run, in run order,
        # so the unpaired list is deterministic for the fingerprint.
        judge_agreement=judge_agreement(
            (rating for run in runs for rating in storage.query_calibration_ratings(scope_id, run_id=run.id)),
            results,
        ),
    )
    # The judge's reliability, before anything judged is summarised: every judged reading carries the
    # tier these two agreements decide for the judges that served it.
    bundle.judge_self_agreement = judge_self_agreement(results)
    bundle.judge_evidence_tiers = judge_evidence_tiers(
        bundle.judge_agreement, bundle.judge_self_agreement, _judged_keys(results)
    )
    bundle.inter_judge_agreement = inter_judge_agreement(results)
    bundle.judge_drift = judge_drift(results)
    bundle.judge_change = _judge_change(runs, results_by_run)
    bundle.goal_check_proofs = goal_check_proofs_of(runs, results)
    # Judged quality and the bars, per cell. Both read the cell algebra's own grouping, so every
    # per-arm number here describes observations the bundle already calls one arm, and neither
    # enters `measures` or the catalog: judged dimensions stay off the ranking surface, and are
    # carried here so that staying off it is not read as never having been measured.
    results_by_cell = _results_by_cell(cells, results)
    bundle.cost_unmeasured_cells, bundle.cost_unmeasured = _cost_unmeasured(results_by_cell)
    bundle.latency_contended_cells, bundle.latency_contended = _latency_contended(
        results_by_cell,
        contended_ids,
        declared=campaign.declared_design is not None and campaign.declared_design.measure_latency,
    )
    bundle.judged_measures = [
        measure.model_copy(
            update={
                "second_judges": [
                    row for row in bundle.inter_judge_agreement.dimensions if row.rubric_dim == measure.name
                ]
            }
        )
        for measure in _judged_measures(
            projection.records, results_by_cell, campaign.declared_design, tiers=bundle.judge_evidence_tiers
        )
    ]
    bundle.bar_adjudications = _bar_adjudications(
        campaign.behavior,
        campaign.declared_design,
        results_by_cell,
        projection.records,
        tiers=bundle.judge_evidence_tiers,
        frontier_bar=frontier_bar,
        profile=profile,
    )
    bundle.verdict_order = _verdict_order(bundle.bar_adjudications, campaign.declared_design)
    # The decision surface, over the same grouping and the same population the bars were read over,
    # and before the catalog: its collections are measure collections like any other here, so the
    # catalog has to describe their names too. Laid out in the surface's one row order (the control as
    # the reference, then the arms by name), which the writer reads and the frozen surface keeps.
    control = campaign.declared_design.control if campaign.declared_design else None
    names = arm_names(bundle.variant_index)
    bundle.cell_measures = _cell_measures(
        cells,
        results_by_cell,
        bundle.judged_measures,
        # Each cell again per kind of case, over the same grouping, population and tiers — read only for
        # the cases the results name, and only their strata.
        strata=_cell_strata(
            results_by_cell,
            {
                case.id: case.stratum
                for case in storage.load_case_strata(sorted({result.test_case_id for result in results}), scope_id)
            },
            projection.records,
            campaign.declared_design,
            tiers=bundle.judge_evidence_tiers,
            profile=profile,
        ),
        short_runs=bundle.short_runs,
        incomplete_runs=bundle.incomplete_runs,
        profile=profile,
        control=control,
        names=names,
    )
    bundle.all_failed_cells, bundle.all_failed = _all_failed(bundle.cell_measures)
    # The time axis, over the same algebra the decision surface was just read with, and before the catalog
    # for the same reason: its cells are measure collections the catalog has to describe.
    bundle.time_axis, bundle.time_axis_withheld = _time_axis(
        runs,
        results_by_run,
        observations,
        {c.apparatus_class_id: c for c in apparatus_classes.values()},
        projection.records,
        campaign.declared_design,
        tiers=bundle.judge_evidence_tiers,
        short_runs=bundle.short_runs,
        incomplete_runs=bundle.incomplete_runs,
        profile=profile,
        names=names,
    )
    bundle.measure_catalog = _measure_catalog(bundle, profile=profile)
    bundle.run_margins, bundle.run_margins_withheld = _run_margins(runs)
    bundle.launch_declarations = launch_declarations
    # After the catalog, which says each measure's better direction and axis: a family is the readings a
    # question's axes name, and a reading with no better end has no verdict to correct.
    bundle.multiple_comparisons, bundle.guardrails = _multiple_comparisons(
        campaign.declared_design,
        design,
        results_by_cell,
        projection.records,
        catalog=bundle.measure_catalog,
        judged_measures=bundle.judged_measures,
        observations=mechanisms,
        served=served,
        profile=profile,
        run_margins=bundle.run_margins,
        contended={
            key for key, members in results_by_cell.items() if any(result.id in contended_ids for result in members)
        },
    )
    bundle.reading_scope = _reading_scope(campaign.declared_design, bundle.measure_catalog, bundle.judged_measures)
    # Divergences pair measures by unit, which only the catalog knows, so they are derived
    # after it — and from the same descriptors the generator will read, never a second lookup
    # that could disagree with what the bundle says a measure is.
    bundle.scope_divergences, bundle.divergences_omitted, divergence_count = _scope_divergences(
        runs,
        results_by_run,
        bundle.measure_catalog,
        apparatus_levels,
        design,
        folds=folds,
        observations=mechanisms,
        served=served,
        profile=profile,
    )
    bundle.divergences_tested, bundle.divergences_untested = divergence_count
    # Last, because it reads what both confound-bearing lenses actually emitted rather than
    # what they might have — a catalog built from the declaration would name dimensions no
    # lens reported, and a reader would take that as a claim the campaign made.
    bundle.confound_catalog = _confound_catalog(bundle, profile=profile)
    return bundle


def _declared_words(declaration: MeasureDeclaration) -> str:
    """A measure's declaration in words, for the sentence naming a difference in one."""
    direction = {True: "higher is better", False: "lower is better", None: "no better end"}[
        declaration.higher_is_better
    ]
    parts = [direction, "a guardrail" if declaration.guardrail else f"merit axis {declaration.merit_axis or 'none'}"]
    margin = declaration.materiality_threshold
    parts.append("no margin" if margin is None else f"margin {format_number(margin)}")
    bounds = declaration.value_range
    parts.append("no range" if bounds is None else f"range {format_number(bounds[0])} to {format_number(bounds[1])}")
    return ", ".join(parts)


def _as_launched(runs: Sequence[EvalRun], profile: HostProfile) -> tuple[HostProfile, list[str]]:
    """The reading host's profile with each of its measures read as the member runs were launched to read it.

    A host's declarations — and on the quick path ``compare(margins=, ranges=, guardrails=)`` — can differ between
    the process that launched a campaign and one that reads it later. Each run freezes how its launching host
    declared every measure to be read (``EvalRun.declared_measures``), and a measure every run that recorded a
    declaration declared alike is read on that declaration, whatever the reading host says; a difference from the
    reader's is named. Runs launched under different declarations of one measure have no one declaration to read:
    the measure keeps the reader's, with no margin read on it (no comparison on it can read ``equivalent``), and
    that is named too. Runs that recorded none (stored before launches recorded them) are read on the reader's.

    Args:
        runs: The resolved member runs.
        profile: The reading host's profile.

    Returns:
        ``(profile, sentences)``: the profile every lens reads, and one sentence per measure read otherwise than
        the reader declares it.
    """
    from threetears.evals.kernel.host.measures import MeasureRegistry

    recorded = [run for run in runs if run.declared_measures]
    if not recorded:
        return profile, []
    sentences: list[str] = []
    read: dict[str, MetricDescriptor] = {}
    for name in sorted({name for run in recorded for name in run.declared_measures}):
        reader = profile.measures.get(name)
        declarations = [run.declared_measures.get(name) for run in recorded]
        if reader is None:
            sentences.append(
                f"{name} was declared by the host that launched these runs, and the reading host declares no such "
                "measure, so its readings are read as an undeclared measure's."
            )
            continue
        agreed = declarations[0]
        if agreed is None or any(declaration != agreed for declaration in declarations):
            read[name] = MetricDescriptor.model_validate({**reader.model_dump(), "materiality_threshold": None})
            sentences.append(
                f"The member runs were launched under different declarations of {name}, so it is read as the reading "
                "host declares it, with no margin: no comparison on it can read equivalent."
            )
            continue
        if declaration_of(reader) == agreed:
            continue
        read[name] = MetricDescriptor.model_validate({**reader.model_dump(), **agreed.model_dump()})
        sentences.append(
            f"{name} is read as its runs were launched to read it ({_declared_words(agreed)}), not as the reading "
            f"host now declares it ({_declared_words(declaration_of(reader))})."
        )
    if not read:
        return profile, sentences
    measures = MeasureRegistry(
        [read.get(name) or descriptor for name in profile.measures.names if (descriptor := profile.measures.get(name))],
        families=profile.measures.families,
    )
    return replace(profile, measures=measures), sentences


def _run_margins(runs: Sequence[EvalRun]) -> tuple[dict[str, float], str | None]:
    """The margins every member run declared alike on a core rate measure, and why any other was withheld.

    A margin decides what `equivalent` means for a pair of arms, so it is read only when every run of the
    campaign declared the same one on the measure — then whichever two arms a contrast sets side by side chose
    it alike. A margin some runs declared and others did not, or that runs declared differently, is read on
    none, and one sentence says so, naming the measures.

    Args:
        runs: The resolved member runs.

    Returns:
        ``(margins, withheld)``: the agreed margins by measure, in name order, and the sentence naming the
        measures whose declared margins disagree, or None.
    """
    declared = sorted({name for run in runs for name in run.declared_margins})
    agreed: dict[str, float] = {}
    disagreed: list[str] = []
    for name in declared:
        margins = {run.declared_margins.get(name) for run in runs}
        if len(margins) == 1 and None not in margins:
            agreed[name] = margins.pop()  # type: ignore[assignment]  # the one value, and it is not None
        else:
            disagreed.append(name)
    if not disagreed:
        return agreed, None
    return agreed, (
        f"The member runs do not all declare one margin on {', '.join(disagreed)}, so no margin is read on "
        f"{'it' if len(disagreed) == 1 else 'them'}: no comparison on {'it' if len(disagreed) == 1 else 'them'} "
        "can read equivalent. Relaunch every arm with the same margin to read one."
    )


def _measure_catalog(bundle: AnalysisContextBundle, *, profile: HostProfile) -> dict[str, MetricDescriptor]:
    """Collect the descriptor for every measure name appearing anywhere in the bundle.

    Built from the assembled bundle rather than alongside it, so the catalog cannot fall
    out of step with what the summaries actually reference: every name here came from a
    summary, and every summary's name is resolvable here.

    Args:
        bundle: The assembled bundle, before its catalog is attached.
        profile: The host whose vocabulary this reads.

    Returns:
        Descriptors keyed by measure name, in name order for a stable fingerprint.
    """
    collections = [
        bundle.telemetry.measures,
        *(summary.measures for summary in bundle.run_summaries),
        *(collection for cell in bundle.cell_measures for collection in _cell_collections(cell)),
        *(cell.measures for cell in _time_axis_cells(bundle)),
    ]
    names = {measure.name for collection in collections for measure in collection.measures}
    return {name: describe_reported_measure(name, profile.measures) for name in sorted(names)}


#: A cell's two coordinates, ``(variant_key, apparatus_class_id)`` — how the per-arm surfaces key it.
_CellKey = tuple[str, str]


def _cell_collections(cell: CellFacts) -> Iterator[MeasureCollection]:
    """A cell's measure collection, then each of its strata's.

    Walked wherever a reader needs every measure NAME a cell holds: a stratum is a smaller set of results,
    so it can carry a name the pooled cell does not — two carriers sharing a leaf name are refused pooling
    over the cell and may each be alone in a stratum.
    """
    yield cell.measures
    for stratum in cell.strata:
        yield stratum.measures


def _results_by_cell(cells: list[Cell], results: list[EvalResult]) -> dict[_CellKey, list[EvalResult]]:
    """Each cell's results, keyed by its two coordinates — the arms every per-arm surface here reads.

    Read off the cell algebra rather than re-grouped, so a per-arm number can only ever describe a
    set of observations the bundle already calls one cell.

    Args:
        cells: The pooled cells.
        results: Every resolved member result.

    Returns:
        ``{(variant_key, apparatus_class_id): [result, ...]}`` in coordinate order, each list in
        result-id order.
    """
    by_id = {result.id: result for result in results}
    return {
        (cell.variant_key, cell.apparatus_class_id): sorted(
            (by_id[observation_id] for observation_id in cell.observation_ids if observation_id in by_id),
            key=lambda result: result.id,
        )
        for cell in sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id))
    }


def _member_run_ids(members: list[EvalResult]) -> list[str]:
    """The runs a cell's results came from, sorted — what a per-arm surface names as citable.

    Read off the cell's own results rather than joined back through a run's configuration,
    because the cell IS its observations: whichever levers a host records, and whichever of them
    the variant index describes, the runs that produced a cell's observations are the runs it
    pooled. Every per-arm surface takes its ``run_ids`` from here, so a judged score and a bar
    verdict on one cell cannot name different runs.

    Args:
        members: One cell's results, from :func:`_results_by_cell`.

    Returns:
        The distinct ``eval_run_id`` values, sorted.
    """
    return sorted({result.eval_run_id for result in members})


def _judged_values(records: list[ScoreRecord]) -> Iterator[tuple[str, ScoreRecord]]:
    """Yield ``(dimension, record)`` for every projected row carrying a judged score.

    Two row shapes carry one: a ``score`` row, whose dimension is its ``rubric_dim`` coordinate, and
    a dual-score axis row, whose metric IS the reserved axis id. The dimensionless ``score`` row a
    result with no judged dimension emits is not one — it records that the cell was attempted.
    """
    for record in records:
        if record.value is None:
            continue
        if record.metric == METRIC_SCORE and record.rubric_dim:
            yield record.rubric_dim, record
        elif record.metric in (METRIC_TRANSCRIPT, METRIC_OUTCOME):
            yield record.metric, record


def _judged_rows(records: list[ScoreRecord]) -> Iterator[tuple[str, ScoreRecord]]:
    """:func:`_judged_values`, plus the null rows the judge answered it could not tell on or the harness faulted.

    What a judged measure COUNTS rather than means: those rows carry no value, so they never reach a
    mean, and are yielded here only so the arm can say how many left it and why — a faulted
    observation's score row is null since the projection reads ``counted_rubric_scores``.
    """
    yield from _judged_values(records)
    for record in records:
        if record.value is None and record.outcome in (JUDGE_CANNOT_TELL_OUTCOME, ResultOutcome.INFRA_EXCLUDE.value):
            if record.metric == METRIC_SCORE and record.rubric_dim:
                yield record.rubric_dim, record
            elif record.metric in (METRIC_TRANSCRIPT, METRIC_OUTCOME):
                yield record.metric, record


def _judged_keys(results: list[EvalResult]) -> set[JudgeKey]:
    """Every judge (:class:`~threetears.evals.analysis.agreement.JudgeKey`) a judged score among ``results`` was given by."""
    return {
        key
        for result in results
        for score in (*result.rubric_scores, result.transcript_score, result.outcome_score)
        if score is not None and (key := judge_key(result, score.dim)) is not None
    }


def _judged_measures(
    records: list[ScoreRecord],
    results_by_cell: dict[_CellKey, list[EvalResult]],
    design: CampaignDesign | None,
    *,
    tiers: list[JudgeEvidenceTier],
) -> list[JudgedMeasure]:
    """Summarise every judged dimension per cell, from the score projection assembly already holds.

    The projection is the source because it is where the dimension's NAME survives: the measure
    walk meets a judged score as a bare ``score`` leaf on a ``RubricScore`` and cannot tell one
    dimension from another. A score on an observation the harness faulted is counted beside the mean
    and left out of it, because a judge reading a broken transcript is not measuring the candidate.

    Args:
        records: The assembly's score projection rows.
        results_by_cell: Each cell's results, from :func:`_results_by_cell`.
        design: The campaign's declaration, for the bar naming a dimension.
        tiers: The judges' evidence tiers; each arm carries the weakest among the judges that served its
            counted scores (:func:`~threetears.evals.analysis.agreement.tier_for_judges`).

    Returns:
        One entry per dimension scored anywhere, sorted by name.
    """
    result_by_id = {result.id: result for members in results_by_cell.values() for result in members}
    cell_of_result = {result.id: key for key, members in results_by_cell.items() for result in members}
    boundary = _boundary_dimensions(result_by_id.values())
    scored: dict[str, dict[_CellKey, list[ScoreRecord]]] = {}
    for dimension, record in _judged_rows(records):
        key = cell_of_result.get(record.result_id)
        if key is not None:
            scored.setdefault(dimension, {}).setdefault(key, []).append(record)
    bars = {bar.measure_id: bar.threshold for bar in (design.bars if design else [])}
    measures = []
    for dimension in sorted(scored):
        scales = {
            row.rubric_scale for rows in scored[dimension].values() for row in rows if row.rubric_scale is not None
        }
        if len(scales) > 1:
            raise ValueError(f"rubric dimension {dimension!r} was judged on more than one scale: {sorted(scales)}")
        # The dual-score axes carry no `rubric_scale` (they are not template dims) and are 1-5.
        descriptor = describe_rubric_dim(dimension, scale=next(iter(scales), "ordinal"))
        arms = []
        for key in sorted(scored[dimension]):
            rows = scored[dimension][key]
            cannot_tell = [row for row in rows if row.outcome == JUDGE_CANNOT_TELL_OUTCOME]
            counted = [
                row for row in rows if row.outcome not in (ResultOutcome.INFRA_EXCLUDE.value, JUDGE_CANNOT_TELL_OUTCOME)
            ]
            valued = [row for row in counted if row.value is not None]
            values = [float(row.value) for row in valued if row.value is not None]
            value_cases = [row.test_case_id for row in valued]
            # The whole judge behind each counted score — dimension, scale, served model and config — so an arm
            # can only carry a tier measured for the very judges that scored it.
            served = [
                judge
                for row in counted
                if row.value is not None and (judge := judge_key(result_by_id[row.result_id], dimension)) is not None
            ]
            arms.append(
                JudgedArm(
                    variant_key=key[0],
                    apparatus_class_id=key[1],
                    run_ids=_member_run_ids(results_by_cell[key]),
                    n=len(values),
                    n_independent=len(set(value_cases)),
                    n_infra_excluded=len(rows) - len(counted) - len(cannot_tell),
                    n_cannot_tell=len(cannot_tell),
                    mean=sum(values) / len(values) if values else None,
                    sem=clustered_standard_error(values, value_cases) if values else None,
                    case_means=small_sample_case_means(values, value_cases),
                    evidence_tier=tier_for_judges(tiers, served),
                )
            )
        measures.append(
            JudgedMeasure(
                name=dimension,
                family=str(descriptor.family),
                value_range=descriptor.value_range,
                scale=descriptor.scale,
                higher_is_better=bool(descriptor.higher_is_better),
                axis="boundary" if dimension in boundary else "capability",
                bar_threshold=bars.get(dimension),
                arms=arms,
            )
        )
    return measures


def _boundary_dimensions(results: Iterable[EvalResult]) -> set[str]:
    """The judged dimensions any of ``results`` was scored on as a boundary dimension — the judged guardrails.

    Any one boundary score makes the dimension a guardrail: a dimension stamped both ways was declared a
    boundary somewhere, and reading it as a guardrail errs toward holding it rather than trading it.
    """
    return {score.dim for result in results for score in result.rubric_scores if score.axis == "boundary"}


def _unstamped_dimensions(results: Iterable[EvalResult]) -> list[str]:
    """The judged dimensions carrying a score judged before the rubric axis was stamped, sorted."""
    return sorted({score.dim for result in results for score in result.rubric_scores if score.axis is None})


#: A bar's per-cell reading: ``(mean, sem, n, n_independent, interval)``.
_BarReading = tuple[float | None, float | None, int, int, tuple[float, float] | None]

#: What adjudicating a bar came to — see :attr:`BarAdjudication.state`.
_BarState = Literal["adjudicated", "names_no_stored_measure", "not_numeric"]


def _applicable_bars(
    behavior: str, design: CampaignDesign | None, *, profile: HostProfile
) -> list[tuple[str, float, bool, Literal["declared", "registered"]]]:
    """The bars this campaign is held to: its own, then each registered incumbent it did not override.

    A campaign that declares no bar on a measure is held to the host's registered one — that is what
    an empty declaration MEANS — so adjudicating only the declared list would say nothing about the
    standard actually in force. A declared bar replaces the incumbent on its measure, and the ratchet
    has already refused one looser than it.

    Args:
        behavior: The campaign's behavior, the scope a registered bar is keyed under.
        design: The campaign's declaration.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(measure_id, threshold, higher_is_better, source)`` per bar, declared first, then registered,
        each in measure order.
    """
    declared = sorted(design.bars if design else [], key=lambda bar: bar.measure_id)
    overridden = {bar.measure_id for bar in declared}
    registered = sorted(
        (bar for bar in profile.bars.bars if bar.behavior == behavior and bar.measure not in overridden),
        key=lambda bar: bar.measure,
    )
    return [(bar.measure_id, bar.threshold, bar.direction == "higher_is_better", "declared") for bar in declared] + [
        (bar.measure, bar.threshold, bar.higher_is_better, "registered") for bar in registered
    ]


def _non_faulted(members: list[EvalResult]) -> list[EvalResult]:
    """A cell's results the harness did not fault — the one population every bar is adjudicated over.

    See :class:`BarVerdict` for why. Held as one function so the three kinds of bar name cannot
    each come to their own answer about which observations count.
    """
    return [result for result in members if not harness_faulted(result)]


def _candidate_failed(members: list[EvalResult]) -> int:
    """How many of a cell's results the candidate failed, for any cause — each counted against the arm.

    By :func:`~threetears.evals.kernel.result_condition.classify_result`, the rule every rate and bar
    counts a failure by.
    """
    return sum(1 for result in members if classify_result(result) is ResultOutcome.CANDIDATE_FAIL)


def _took_no_turn(members: list[EvalResult]) -> bool:
    """Whether a cell's results include a failure and no turn — the reading :attr:`CellFacts.all_failed` makes.

    Read off the results, for the lenses that hold them rather than the cell's facts, by the same predicate
    (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`), so the two cannot disagree.
    """
    return _no_turn(members) > 0 and not any(delivered_a_turn(result) for result in members)


def _no_turn(members: list[EvalResult]) -> int:
    """How many of a cell's failures took no turn — the ones its cost and latency readings left out.

    Exactly the candidate failures outside
    :func:`~threetears.evals.kernel.result_condition.delivered_a_turn`, the predicate the measure walk's
    ``delivered`` population is, so the count beside a cell's cost and latency is the number they left out.
    """
    return sum(
        1
        for result in members
        if classify_result(result) is ResultOutcome.CANDIDATE_FAIL and not delivered_a_turn(result)
    )


def _cannot_tell_on(bar: BarName, counted: list[EvalResult], judged_rows: dict[str, list[ScoreRecord]]) -> int:
    """How many of a cell's counted results the judge could not score on a judged bar's dimension."""
    if bar.kind != "judged":
        return 0
    return sum(
        1
        for result in counted
        for dimension, record in _judged_rows(judged_rows.get(result.id, []))
        if dimension == bar.name and record.outcome == JUDGE_CANNOT_TELL_OUTCOME
    )


def _bar_reading(
    bar: BarName, members: list[EvalResult], judged_rows: dict[str, list[ScoreRecord]], *, profile: HostProfile
) -> _BarReading | None:
    """Read one bar's value over one cell's results, from where its kind is carried.

    Args:
        bar: The resolved bar name — its kind says where the value lives.
        members: The cell's results, faulted ones included. A measure is read over its own population
            (:func:`_collect_measures` leaves a fault out of a ``scored`` one); a judged dimension over
            the non-faulted results, as every score is.
        judged_rows: Projected judged-score rows by result id, for a judged dimension.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(mean, sem, n, n_independent, interval)``, or None when no counted result carries the name.
        The interval is the one the cell's own reading states — a measure's summary, or a judged
        dimension's through the same rule (:func:`~threetears.evals.analysis.stats.observed_mean_interval`)
        — never one computed for the bar, so a bar and the cell it reads cannot state two widths.
    """
    if bar.kind in ("measure", "goal_state"):
        # A check's rate is the measure the walk publishes for it, read here rather than recomputed,
        # so a bar's value and the cell's measure are one number by construction.
        name = bar.name if bar.kind == "measure" else goal_check_measure(bar.name)
        collection = _measure_collection(members, profile=profile, undeclared="scored")
        summary = next((m for m in collection.measures if m.name == name), None)
        if summary is None or summary.mean is None:
            return None
        interval = None if summary.ci_low is None or summary.ci_high is None else (summary.ci_low, summary.ci_high)
        return summary.mean, summary.sem, summary.n, summary.n_independent, interval
    rows = [
        (float(record.value), record.test_case_id)
        for result in _non_faulted(members)
        for dimension, record in _judged_values(judged_rows.get(result.id, []))
        if dimension == bar.name and record.value is not None
    ]
    if not rows:
        return None
    values = [value for value, _ in rows]
    cases = [case for _, case in rows]
    return (
        sum(values) / len(values),
        clustered_standard_error(values, cases),
        len(values),
        len(set(cases)),
        observed_mean_interval(
            values, cases=cases, value_range=bar.descriptor.value_range, floor=bar.descriptor.interval_floor
        ),
    )


def _judged_bar_tier(
    bar: BarName,
    members: list[EvalResult],
    judged_rows: dict[str, list[ScoreRecord]],
    *,
    tiers: list[JudgeEvidenceTier],
) -> JudgedEvidenceTier | None:
    """The evidence tier a judged bar's verdict on one cell stands on, or None for a bar no judge scored.

    The weakest tier among the judges that served the very scores :func:`_bar_reading` read for the cell — its
    non-faulted results' values on the bar's dimension — by
    :func:`~threetears.evals.analysis.agreement.tier_for_judges`, the lookup every judged arm's tier is read by.
    """
    if bar.kind != "judged":
        return None
    served = [
        judge
        for result in _non_faulted(members)
        for dimension, _ in _judged_values(judged_rows.get(result.id, []))
        if dimension == bar.name and (judge := judge_key(result, dimension)) is not None
    ]
    return tier_for_judges(tiers, served)


def _bar_adjudications(
    behavior: str,
    design: CampaignDesign | None,
    results_by_cell: dict[_CellKey, list[EvalResult]],
    records: list[ScoreRecord],
    *,
    tiers: list[JudgeEvidenceTier],
    frontier_bar: float | None,
    profile: HostProfile,
) -> list[BarAdjudication]:
    """Adjudicate every applicable bar against every cell.

    **What a bar names is resolved by** :func:`~threetears.evals.kernel.declaration.resolve_bar_name`
    — the function the declaration gate asks — so a bar is read the way it was admitted, and a name
    the gate refuses is one this can never read. The gate supplies the template's names; this
    supplies the names the results actually carry (the judged dimensions they were scored on and
    the checks they evaluated), which is the same question asked of the evidence, and it lets a
    host's registered bar — which no gate saw — be resolved by the same rule.

    Every kind is read over the same population, each cell's non-faulted results; see
    :class:`BarVerdict`. A verdict on a judged dimension carries the evidence tier of the judges behind it
    (:func:`_judged_bar_tier`); every other verdict carries None.

    Args:
        behavior: The campaign's behavior.
        design: The campaign's declaration.
        results_by_cell: Each cell's results.
        records: The assembly's score projection rows, where a judged dimension's name survives.
        tiers: The judges' evidence tiers (``judge_evidence_tiers``).
        frontier_bar: The bar the frontier was given (:func:`_frontier_bar`), or None. A bar on pass^k carries
            no cell verdict, and its reason says whether the frontier read it.
        profile: The host whose vocabulary this reads.

    Returns:
        One adjudication per applicable bar, in :func:`_applicable_bars` order.
    """
    judged_rows: dict[str, list[ScoreRecord]] = {}
    for record in records:
        judged_rows.setdefault(record.result_id, []).append(record)
    all_results = [result for members in results_by_cell.values() for result in members]
    rubric_dimensions = {
        dimension: row.rubric_scale for dimension, row in _judged_rows(records) if row.rubric_scale is not None
    }
    goal_state_checks = {outcome.expression for result in all_results for outcome in result.goal_state_outcomes}
    counted_by_cell = {key: _non_faulted(members) for key, members in results_by_cell.items()}

    adjudications = []
    for measure_id, threshold, higher_is_better, source in _applicable_bars(behavior, design, profile=profile):
        resolved = resolve_bar_name(
            measure_id,
            rubric_dimensions=rubric_dimensions,
            goal_state_checks=goal_state_checks,
            measures=profile.measures,
        )
        state: _BarState
        reason: str | None = None
        verdicts: list[BarVerdict] = []
        if isinstance(resolved, UnreadableBarName):
            state = "not_numeric" if resolved.refusal == "not_numeric" else "names_no_stored_measure"
            reason = resolved.reason
            if measure_id == FRONTIER_RANKING_MEASURE:
                # Not "never read": no cell carries pass^k, and the frontier is where a bar on it is read.
                reason = (
                    f"{measure_id} is a rate over a contestant's cases that no single result carries, so no cell "
                    "verdict is given on it here. "
                    + (
                        "The frontier read this bar on each contestant's pass^k interval: see frontier.bar and each "
                        "point's bar_decision."
                        if frontier_bar is not None
                        else "The frontier was given no bar either; frontier_bar_withheld says why."
                    )
                )
        else:
            readings = {
                key: _bar_reading(resolved, members, judged_rows, profile=profile)
                for key, members in results_by_cell.items()
            }
            # A measure computed over every observation excluded none; reporting the cell's faults as
            # excluded from it would describe a population the value was not computed over.
            keeps_faults = resolved.kind == "measure" and resolved.descriptor.population == "all_observed"
            # The measure's declared margin — the difference too small to act on, the one margin the
            # history read's equivalence test is run against too. None holds the bar at its threshold.
            margin = resolved.descriptor.materiality_threshold
            if any(reading is not None for reading in readings.values()):
                state = "adjudicated"
                for key, members in results_by_cell.items():
                    mean, sem, n, n_independent, interval = readings[key] or (None, None, 0, 0, None)
                    verdicts.append(
                        BarVerdict(
                            variant_key=key[0],
                            apparatus_class_id=key[1],
                            run_ids=_member_run_ids(members),
                            value=mean,
                            sem=sem,
                            n=n,
                            n_independent=n_independent,
                            n_infra_excluded=0 if keeps_faults else len(members) - len(counted_by_cell[key]),
                            n_cannot_tell=_cannot_tell_on(resolved, counted_by_cell[key], judged_rows),
                            ci_low=None if interval is None else interval[0],
                            ci_high=None if interval is None else interval[1],
                            margin=margin,
                            cleared=None
                            if interval is None
                            else interval_clears(interval, threshold, margin=margin, higher_is_better=higher_is_better),
                            judge_evidence_tier=_judged_bar_tier(resolved, members, judged_rows, tiers=tiers),
                        )
                    )
            else:
                state = "names_no_stored_measure"
                reason = f"{measure_id} resolves as a {resolved.kind} name, but no non-faulted member result carries a value of it"
                # A turn's time or spend read nowhere because no result took a turn is not a name nothing
                # stores: every call failed, and the bar says so rather than send the reader to the registry.
                reads_a_turn = (
                    resolved.kind == "measure" and summary_population(resolved.descriptor, "scored") == "delivered"
                )
                members = [result for cell in results_by_cell.values() for result in cell]
                if reads_a_turn and members and not any(delivered_a_turn(result) for result in members):
                    reason = (
                        f"{measure_id} is a turn's time or spend, and no result took a turn: every result failed — the "
                        "candidate's model refused or errored on every call — or was excluded as a fault of the rig"
                    )
        adjudications.append(
            BarAdjudication(
                measure_id=measure_id,
                threshold=threshold,
                direction="higher_is_better" if higher_is_better else "lower_is_better",
                source=source,
                state=state,
                reason=reason,
                merit_axis=None if isinstance(resolved, UnreadableBarName) else resolved.descriptor.merit_axis,
                verdicts=verdicts,
            )
        )
    return adjudications


def _short_cells(cells: list[Cell], design: CampaignDesign | None) -> list[ShortCell]:
    """Name every cell holding fewer repetitions than the declaration intended.

    A cell is counted by its least-repeated case, since that case is the cell's weakest replication
    and pooling more of the others does not repair it.

    Args:
        cells: The pooled cells, in coordinate order.
        design: The campaign's declaration, or None.

    Returns:
        One entry per short cell, in coordinate order. Empty when nothing was declared — an unstated
        intention cannot be fallen short of — or when every cell met it. A cell with no case count
        has no repetitions to compare and is not listed: its own facts already say its cases were
        unrecorded, which is the honest answer (unknown, not short). The bundle's observations all
        name their case, since ``EvalResult.test_case_id`` is required.
    """
    intended = design.intended_repetitions if design is not None else None
    if intended is None:
        return []
    short = []
    for cell in sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id)):
        observed = cell.repeats_per_case_min
        if observed is None or observed >= intended:
            continue
        short.append(
            ShortCell(
                variant_key=cell.variant_key,
                apparatus_class_id=cell.apparatus_class_id,
                intended=intended,
                observed=observed,
                # Names no coordinate: the entry carries the cell's, and the writer's view renames those
                # to the cell's alias, which a digest spelled into prose would bypass.
                sentence=(
                    f"This cell ran its least-repeated case {observed} times against the {intended} repetitions "
                    "the campaign declared it intends per case in each cell, so its estimates rest on less "
                    "replication than the design set out to buy."
                ),
            )
        )
    return short


#: How a reader learns to report spend from the one candidate the engine cannot see into.
_HOW_TO_REPORT_SPEND = "A quick candidate reports its spend by returning an Answer."


def _cost_unmeasured(results_by_cell: dict[_CellKey, list[EvalResult]]) -> tuple[list[CellCoordinate], str | None]:
    """Name every cell where no result observed spend, with the one sentence that says what that means.

    A cell is listed when it holds a result storing a ``cost_usd`` and no result in it observed spend
    (:func:`~threetears.evals.kernel.usage_capture.spend_observed`): every number it stores is the sum of
    nothing, so the measure walk read none of them and the cell carries no ``cost_usd`` reading. A cell whose
    every result went unpriced is not listed — its cost is unknown for a reason ``cost_usd`` null already
    states — and neither is one where any result observed spend, whose own ``n`` discloses the rest. Read
    over the turns the candidate took (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`),
    the population a cell's ``cost_usd`` reading is read over (``delivered``), so a billed refusal cannot
    make a cell whose turns reported no spend look measured. A cell where no result took a turn is not
    listed: it has no cost because every call failed, which :func:`_all_failed` says, and "nobody reported
    spend" would be the wrong reason.

    Args:
        results_by_cell: Each cell's results, from :func:`_results_by_cell`.

    Returns:
        The unmeasured cells in coordinate order, and the sentence — None when there are none.
    """
    unmeasured: list[CellCoordinate] = []
    for (variant_key, apparatus_class_id), members in sorted(results_by_cell.items()):
        counted = [result for result in members if delivered_a_turn(result)]
        stores_a_cost = any(result.cost_usd is not None for result in counted)
        if stores_a_cost and not any(spend_observed(result.usage, result.cost_roles) for result in counted):
            unmeasured.append(CellCoordinate(variant_key=variant_key, apparatus_class_id=apparatus_class_id))
    if not unmeasured:
        return [], None
    if len(unmeasured) == len(results_by_cell):
        sentence = (
            "Cost was not measured: no result reported its spend, so the $0 each one stores is not a measurement "
            "and cost is neither charted nor tested."
        )
    else:
        sentence = (
            f"Cost was not measured in {len(unmeasured)} of {len(results_by_cell)} cells: no result there reported "
            "its spend, so the $0 those results store is not a measurement and cost is charted and tested only "
            "where it was measured."
        )
    return unmeasured, f"{sentence} {_HOW_TO_REPORT_SPEND}"


def _latency_contended(
    results_by_cell: dict[_CellKey, list[EvalResult]], contended_ids: Collection[str], *, declared: bool = False
) -> tuple[list[CellCoordinate], str | None]:
    """Name every cell whose latency read under concurrency was left out, with the one sentence that says so.

    Args:
        results_by_cell: Each cell's results, from :func:`_results_by_cell`.
        contended_ids: The results whose latency :func:`~threetears.evals.analysis.contention.withhold_contended_latency`
            removed before anything read them.
        declared: The campaign's design declares latency under test, so a run read under concurrency is one it
            cannot read its question from — said, with the remedy, rather than left to a reader to infer.

    Returns:
        The cells in coordinate order, and the sentence — None when nothing was left out.
    """
    cells: list[CellCoordinate] = []
    withheld = 0
    total = 0
    for (variant_key, apparatus_class_id), members in sorted(results_by_cell.items()):
        total += len(members)
        here = sum(1 for result in members if result.id in contended_ids)
        if here:
            withheld += here
            cells.append(CellCoordinate(variant_key=variant_key, apparatus_class_id=apparatus_class_id))
    where = "" if len(cells) == len(results_by_cell) else f" in {len(cells)} of {len(results_by_cell)} cells"
    return cells, contended_latency_sentence(withheld, total, where=where, declared=declared)


def _all_failed(cells: list[CellFacts]) -> tuple[list[CellCoordinate], str | None]:
    """Name every cell where no counted result took a turn, with the one sentence that says so.

    Read off the cells' own counts (:attr:`~threetears.evals.kernel.surface.CellFacts.all_failed`), so the
    list and the surface cannot disagree about which cell took no turn. Such a cell has no cost or latency
    reading, and without this the absence reads as "not measured" beside the arms that were.

    Args:
        cells: The decision surface's cells, from :func:`_cell_measures`.

    Returns:
        The cells in coordinate order, and the sentence
        (:func:`~threetears.evals.kernel.surface.all_failed_sentence`) — None when there are none.
    """
    failed = [
        CellCoordinate(variant_key=cell.variant_key, apparatus_class_id=cell.apparatus_class_id)
        for cell in sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id))
        if cell.all_failed
    ]
    if not failed:
        return [], None
    return failed, all_failed_sentence(len(failed), len(cells))


def _held_fixed_reading(runs: list[EvalRun], design: CampaignDesign | None) -> HeldFixedReading:
    """Compare what the campaign declared held fixed with the provenance every resolved run recorded.

    Args:
        runs: The resolved member runs.
        design: The campaign's declaration, or None.

    Returns:
        The reading. Its disclosure is composed from the branches actually taken: a declared apparatus
        the runs contradict, a mix of provenances nobody declared, and an uncontrolled stimulus each
        add their own sentence, and none of them adds one that did not happen.
    """
    provenance = {run.id: run.apparatus_provenance for run in sorted(runs, key=lambda r: r.id)}
    held_fixed = design.held_fixed if design is not None else None
    declared = held_fixed.apparatus if held_fixed is not None else None
    contradicting = sorted(run_id for run_id, found in provenance.items() if declared is not None and found != declared)
    sentences = []
    if contradicting:
        sentences.append(
            f"This campaign declares its apparatus {declared}, but {len(contradicting)} of its {len(provenance)} "
            f"resolved runs recorded otherwise ({', '.join(contradicting)}); their observations sit in cells of "
            "their own, and a finding drawn from them is drawn from a different kind of evidence than the "
            "declaration describes."
        )
    elif declared is None and len(set(provenance.values())) > 1:
        sentences.append(
            "This campaign declares nothing held fixed, and its runs mix commissioned and witnessed apparatus; the two "
            "never share a cell, so an arm measured both ways is reported as two cells."
        )
    if held_fixed is not None and held_fixed.stimulus == "uncontrolled":
        sentences.append(f"The stimulus was not held fixed: {held_fixed.stimulus_reason.strip()}")
    return HeldFixedReading(
        declared_stimulus=held_fixed.stimulus if held_fixed is not None else None,
        stimulus_reason=held_fixed.stimulus_reason if held_fixed is not None else "",
        declared_apparatus=declared,
        run_provenance=provenance,
        contradicting_run_ids=contradicting,
        disclosure=" ".join(sentences) or None,
    )


def _verdict_order(adjudications: list[BarAdjudication], design: CampaignDesign | None) -> VerdictOrder:
    """Order the adjudicated bars by the campaign's declared merit priority, and scope them per question.

    Only an ``adjudicated`` bar ranks: a bar with no verdict gives nothing to read in any order. A bar
    keeps its ``bar_adjudications`` position within its axis, so the priority reorders axes and never
    the bars on one.

    Args:
        adjudications: The bundle's bar adjudications, in their own order.
        design: The campaign's declaration, or None.

    Returns:
        The order. With no declaration or an empty priority, every adjudicated bar is unranked and no
        tier exists — the analysis must not invent a preference the campaign never stated.
    """
    priority = list(design.merit_priority) if design is not None else []
    by_axis: dict[MeritAxis | None, list[str]] = defaultdict(list)
    for bar in adjudications:
        if bar.state == "adjudicated":
            by_axis[bar.merit_axis].append(bar.measure_id)
    ranked = set(priority)
    unranked = [
        bar.measure_id
        for bar in adjudications
        if bar.state == "adjudicated" and (bar.merit_axis is None or bar.merit_axis not in ranked)
    ]
    questions = []
    for question in design.live_questions() if design is not None else []:
        if not question.merit_axes:
            continue
        # Each axis once: the declaration refuses a repeat, so the reader takes the list as it is.
        axes = sorted(question.merit_axes, key=lambda axis: priority.index(axis) if axis in ranked else len(priority))
        questions.append(
            QuestionScope(
                question_id=question.id,
                merit_axes=axes,
                bar_measure_ids=[measure for axis in axes for measure in by_axis.get(axis, [])],
                unbarred_axes=[axis for axis in axes if not by_axis.get(axis)],
            )
        )
    return VerdictOrder(
        merit_priority=priority,
        tiers=[MeritTier(axis=axis, bar_measure_ids=by_axis.get(axis, [])) for axis in priority],
        unranked_bar_measure_ids=unranked,
        questions=questions,
    )


#: Why a bundle carries no family of comparisons, one sentence per cause.
#: Opens the campaign-wide family's disclosure, where the campaign declares no question.
_NO_QUESTION_FAMILY = (
    "This campaign declares no live question, so every comparison it holds — each contrast against the control, "
    "on every reading on a merit axis — is corrected as one family."
)
_NO_CONTROL_TO_COMPARE_AGAINST = (
    "No control resolved, so there is no arm for a contrast to be tested against and no separation between arms is "
    "tested here; every comparison in this campaign is marginal."
)


def _per_case_values(
    members: list[EvalResult], judged_rows: dict[str, list[ScoreRecord]], *, profile: HostProfile
) -> dict[tuple[ReadingKind, str], dict[str, float]]:
    """Every numeric reading of one cell, as one mean per case — the unit a family comparison tests.

    A case is the independent draw (rule: count cases, not attempts), so repeats of one case are
    averaged before any test sees them. A measure's per-case value is read off the same walk the
    cell's own collection is (:func:`_measure_collection`, each measure over its own population), so
    the value tested and the value the cell reports are one computation; a judged dimension's, off the
    non-faulted results' scores, as :func:`_bar_reading` reads one.

    Args:
        members: One cell's results, faulted ones included.
        judged_rows: Projected judged-score rows by result id.
        profile: The host whose vocabulary this reads.

    Returns:
        ``{(reading, name): {test_case_id: per-case mean}}`` for every reading with a numeric value.
    """
    by_case: dict[str, list[EvalResult]] = defaultdict(list)
    for result in members:
        by_case[result.test_case_id].append(result)
    values: dict[tuple[ReadingKind, str], dict[str, float]] = defaultdict(dict)
    for case_id in sorted(by_case):
        # The candidate's own spend among them (``production_replicating_cost``, which the walk yields over the
        # turns the candidate took and only where a result measured it): a contrast on cost tests it, never
        # ``cost_usd``, which sums the judge's spend too.
        for summary in _measure_collection(by_case[case_id], profile=profile, undeclared="scored").measures:
            if summary.mean is not None:
                values[("measure", summary.name)][case_id] = summary.mean
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for result in _non_faulted(members):
        for dimension, record in _judged_values(judged_rows.get(result.id, [])):
            if record.value is not None:
                scores[(dimension, result.test_case_id)].append(float(record.value))
    for (dimension, case_id), case_scores in sorted(scores.items()):
        values[("judged", dimension)][case_id] = sum(case_scores) / len(case_scores)
    return values


class PlanningReading(NamedTuple):
    """One reading a comparison family could test, as earlier runs observed it: every repeat, by run and case."""

    reading: ReadingKind
    name: str
    higher_is_better: bool
    #: The reading's declared inclusive bounds, or None where it declares none.
    value_range: tuple[float, float] | None
    #: ``{run id: {test case id: [one value per repeat]}}``, in run and case order.
    repeats: dict[str, dict[str, list[float]]]


def planning_readings(
    runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
) -> list[PlanningReading]:
    """Every reading an unscoped comparison family would test, with each earlier run's per-repeat values.

    What a power pre-flight plans from: the readings :func:`_family_readings` admits for a question that names
    no axis (a measure with a better end on a merit axis, a capability judged dimension), each observation
    read by the one walk a family's per-case value is read by (:func:`_per_case_values`, over the one result),
    so a repeat's value here and a case's mean in a comparison are one computation. Guardrails are left out:
    they are held, not tested for a difference; so is latency read under concurrency, which no comparison reads.

    Args:
        runs: The earlier runs.
        results_by_run: Each run's results.
        profile: The host whose vocabulary this reads.

    Returns:
        One entry per reading with a value, sorted by reading kind and name.
    """
    results_by_run = _failures_as_misses({run.id: results_by_run.get(run.id, []) for run in runs})
    # Latency read under concurrency is in no comparison (#701), so it plans none either.
    results_by_run = {
        run_id: withhold_contended_latency(run_results, profile.measures)
        for run_id, run_results in results_by_run.items()
    }
    results = [result for run in runs for result in results_by_run[run.id]]
    projection = project_score_records(runs, results, known_run_ids=None, archived_run_ids=None, profile=profile)
    judged_rows: dict[str, list[ScoreRecord]] = {}
    scales: dict[str, RubricScale] = {}
    for record in projection.records:
        judged_rows.setdefault(record.result_id, []).append(record)
        if record.metric == METRIC_SCORE and record.rubric_dim and record.rubric_scale is not None:
            scales[record.rubric_dim] = record.rubric_scale
    boundary = _boundary_dimensions(results)
    repeats: dict[tuple[ReadingKind, str], dict[str, dict[str, list[float]]]] = {}
    for run in runs:
        for result in sorted(results_by_run[run.id], key=lambda one: (one.test_case_id, one.k_iteration, one.id)):
            for reading, by_case in _per_case_values([result], judged_rows, profile=profile).items():
                for case_id, value in by_case.items():
                    repeats.setdefault(reading, {}).setdefault(run.id, {}).setdefault(case_id, []).append(value)
    readings = []
    for (kind, name), by_run in sorted(repeats.items()):
        if kind == "measure":
            descriptor = describe_reported_measure(name, profile.measures)
            if (
                descriptor.higher_is_better is None
                or classifier_label_of(name) is not None
                or not axis_in_question_scope(descriptor.merit_axis, [])
            ):
                continue
            higher, value_range = descriptor.higher_is_better, descriptor.value_range
        else:
            if name in boundary:
                continue
            judged = describe_rubric_dim(name, scale=scales.get(name, "ordinal"))
            if judged.higher_is_better is None:
                continue
            higher, value_range = judged.higher_is_better, judged.value_range
        readings.append(PlanningReading(kind, name, higher, value_range, by_run))
    return readings


def _family_readings(
    axes: list[MeritAxis], catalog: dict[str, MetricDescriptor], judged_measures: list[JudgedMeasure]
) -> dict[tuple[ReadingKind, str], bool]:
    """The readings a question with these axes could draw a verdict from, each with its better direction.

    A measure qualifies when it has a better end (a reading with none cannot improve or regress) and
    sits on one of the axes; a per-label classifier statistic does not, because it is computed from a
    whole cell's confusion counts and has no per-case value to test. A capability judged dimension sits on
    the quality axis. A guardrail — a boundary judged dimension, or a measure declared one, which serves no
    axis — is in no family: it is held, never traded against a gain, and is decided on its own
    (:func:`_guardrails`). An unscoped question (no axes) asks about every axis — every axis, not every measure:
    a measure that serves no merit axis contributes to no verdict (:data:`MeritAxis`), scoped or not
    (:func:`~threetears.evals.kernel.declaration.axis_in_question_scope`).
    That is how the measuring apparatus's own readings stay out of a contrast between candidates: the
    judge phase's time (``judge_ms``), the drain wait, and ``cost_usd`` and ``program_cost``, which sum the
    judge's spend — what it cost to MEASURE an arm. The candidate's spend is ``production_replicating_cost``,
    on the cost axis.

    Args:
        axes: The question's merit axes; empty for an unscoped question.
        catalog: The bundle's measure catalog.
        judged_measures: The bundle's judged measures.

    Returns:
        ``{(reading, name): higher_is_better}``.
    """
    readings: dict[tuple[ReadingKind, str], bool] = {}
    for name, descriptor in catalog.items():
        if descriptor.higher_is_better is None or classifier_label_of(name) is not None:
            continue
        if axis_in_question_scope(descriptor.merit_axis, axes):
            readings[("measure", name)] = descriptor.higher_is_better
    if axis_in_question_scope(JUDGED_MERIT_AXIS, axes):
        for measure in judged_measures:
            if measure.axis == "capability":
                readings[("judged", measure.name)] = measure.higher_is_better
    return readings


def _test_samples(
    control_values: Mapping[str, float], contrast_values: Mapping[str, float]
) -> tuple[list[float], list[float], bool]:
    """The two samples a contrast against the control reads, and whether they are paired.

    Paired over the cases both cells ran when they share at least two — far more powerful, and the design
    a fixed case set exists for — else each side's per-case values, unpaired. One choice for every reading
    of a contrast, a comparison's and a guardrail's alike.

    Returns:
        ``(control sample, contrast sample, paired)``, the two aligned by case when paired
        (:func:`~threetears.evals.analysis.stats.contrast_samples`, the rule the frontier's boundary pillar reads too).
    """
    return contrast_samples(control_values, contrast_values)


class _Tested(NamedTuple):
    """One comparison before its family's correction: the comparison, the p's it carries, and its samples."""

    comparison: FamilyComparison
    p_raw: float | None
    equivalence_p_raw: float | None
    samples: tuple[list[float], list[float]]
    #: The reading's declared range, which bounds the interval on its delta (:func:`difference_interval`).
    value_range: tuple[float, float] | None = None


def _compare(
    reading: tuple[ReadingKind, str],
    higher_is_better: bool,
    control: tuple[_CellKey, dict[str, float]],
    contrast: tuple[_CellKey, dict[str, float]],
    *,
    threshold: float | None,
    margin_source: Literal["measure", "run"] | None = None,
    value_range: tuple[float, float] | None = None,
    no_turn: tuple[str, ...] = (),
    contended: tuple[str, ...] = (),
) -> _Tested:
    """Test one contrast against the control on one reading, before correction.

    Paired over the cases both cells ran when they share at least two — far more powerful, and the
    design a fixed case set exists for — else the unpaired test over each side's per-case values
    (:func:`~threetears.evals.analysis.stats.composite_significance`), and where the values have no spread
    the bounded test on the declared range every other surface reads that gap by
    (:func:`~threetears.evals.analysis.stats.separation_test`), so one concept has one answer. With no range
    such a gap is ``not_separated``, with ``not_separated_reason`` naming the remedy. The means, the counts and the delta
    are all over the cases the test read, and each side says how many of its own it left out, so the
    figures a reader sees are the figures the test saw. A paired comparison on a measure with a declared
    margin also runs the paired equivalence test (TOST) against it.

    Args:
        reading: The reading's kind and name.
        higher_is_better: Which way is better on it.
        control: The control cell and its per-case values.
        contrast: The contrast cell and its per-case values.
        threshold: The measure's declared materiality threshold, which labels the delta through the one
            predicate every surface uses (:func:`~threetears.evals.kernel.metrics.materiality`) and is the
            equivalence test's margin; None for a measure that declared none and for a judged dimension.
        margin_source: Where ``threshold`` came from (:func:`_margin_of`), which the comparison records.
        value_range: The reading's declared inclusive bounds, which the equivalence test reads so its error
            rate holds on coarse values at every n (:func:`~threetears.evals.analysis.stats.paired_equivalence`),
            and which bound the interval on the delta; None where it declares none.
        no_turn: Which sides (``"control"``, ``"arm"``) have no turn to read a turn's time or spend over —
            every result there failed with no turn taken — so an untested comparison says that, the reason,
            rather than that too few cases carried the reading.
        contended: Which sides of a reading of elapsed time held latency read under concurrency, which the
            bundle withheld (:mod:`~threetears.evals.analysis.contention`) — so an untested comparison says that,
            rather than that too few cases carried the reading.

    Returns:
        The comparison with its adjusted p's, interval and verdict still unset, its raw p's (None where no
        test produced one) for the family's correction, and the samples the test read, for the interval.
    """
    (control_key, control_values), (contrast_key, contrast_values) = control, contrast
    a, b, paired = _test_samples(control_values, contrast_values)
    # The separation test every surface reads a gap by (:func:`separation_test`): the t-test's where the values
    # have spread, and where they have none — every shared case moved by one amount, or each side constant — the
    # bounded test on the reading's declared range, decided on exact values. With no range a gap with no spread
    # is not separated, and says why: the exact sign-flip p tests symmetry, not the mean (#597).
    separation = separation_test(a, b, paired=paired, value_range=value_range)
    p_raw = separation.p
    hedges_g, _, _ = composite_significance(a, b, paired=paired)
    no_spread = (
        no_spread_p([exact_decimal(x) for x in a], [exact_decimal(y) for y in b], paired=paired)
        if len(a) >= 2 and len(b) >= 2
        else None
    )
    if no_spread is not None:
        # No spread, decided exactly: no finite effect size, whatever a float residue lets the t statistic say.
        hedges_g = None
    mean_a = sum(a) / len(a) if a else None
    mean_b = sum(b) / len(b) if b else None
    untested_reason = None
    if p_raw is None:
        # Named from the branch that refused, since the causes have different remedies.
        if no_turn:
            untested_reason = (
                f"every result of the {' and the '.join(no_turn)} failed with no turn taken, so there is no "
                "turn's time or spend to compare"
            )
        elif contended:
            untested_reason = (
                f"the {' and the '.join(contended)}'s latency was read while other cells or runs executed beside it, "
                "so it is withheld and not compared (`latency_contended`)"
            )
        elif len(a) < 2 or len(b) < 2:
            untested_reason = "fewer than two cases carry this reading on a side"
        elif separation.refusal is None:
            untested_reason = "the values differ by less than floating point resolves, so no t statistic exists"
    # The equivalence test only where the separation test produced a p, so each equivalence hypothesis has
    # its comparison's separation hypothesis beside it in the family (see holm_adjust's max_true).
    margin = threshold if paired and threshold and p_raw is not None else None
    # A margin with no declared range is not tested at all (#695): no test of a mean holds α without one, and
    # `equivalent` is the one claim that arms are alike. The comparison says why, naming the remedy.
    equivalence_refused = equivalence_untested_reason(margin, value_range)
    if equivalence_refused is not None:
        margin = None
    # The differences of exact values, so a float residue cannot pass for a spread nor a spread for a constant; on
    # the measure's declared range the bounded test decides either, at the error rate it states.
    _, equivalence_p_raw = paired_equivalence(
        [exact_decimal(y) - exact_decimal(x) for x, y in zip(a, b)], margin, value_range=value_range
    )
    delta = None if mean_a is None or mean_b is None else mean_b - mean_a
    comparison = FamilyComparison(
        reading=reading[0],
        name=reading[1],
        higher_is_better=higher_is_better,
        control=ComparedCell(
            variant_key=control_key[0],
            apparatus_class_id=control_key[1],
            n_cases=len(a),
            mean=mean_a,
            n_left_out=len(control_values) - len(a),
        ),
        contrast=ComparedCell(
            variant_key=contrast_key[0],
            apparatus_class_id=contrast_key[1],
            n_cases=len(b),
            mean=mean_b,
            n_left_out=len(contrast_values) - len(b),
        ),
        delta=delta,
        hedges_g=hedges_g if p_raw is not None else None,
        test=None if p_raw is None else ("paired" if paired else "unpaired"),
        basis=separation.basis,
        p_raw=p_raw,
        equivalence_margin=margin,
        margin_source=margin_source if threshold is not None else None,
        equivalence_p_raw=equivalence_p_raw,
        equivalence_untested_reason=equivalence_refused,
        # A gap with no spread on a reading with no declared range is not separated, with the refusal naming the
        # remedy; it joins no family, since no test ran to correct.
        verdict="untested" if untested_reason is not None else "not_separated",
        untested_reason=untested_reason,
        not_separated_reason=separation.refusal if untested_reason is None else None,
        materiality=None if delta is None else materiality(threshold, delta),
    )
    return _Tested(comparison, p_raw, equivalence_p_raw, (a, b), value_range)


def _margin_of(
    reading: tuple[ReadingKind, str], catalog: Mapping[str, MetricDescriptor], run_margins: Mapping[str, float]
) -> tuple[float | None, Literal["measure", "run"] | None]:
    """A reading's margin and where it came from: the descriptor's threshold, else its runs' declared margin.

    A core rate measure's descriptor declares none (:data:`~threetears.evals.kernel.metrics.RUN_MARGIN_MEASURES`),
    so the two never both exist for one measure. A judged dimension has neither.
    """
    if reading[0] != "measure":
        return None, None
    declared = catalog[reading[1]].materiality_threshold
    if declared is not None:
        return declared, "measure"
    if (margin := run_margins.get(reading[1])) is not None:
        return margin, "run"
    return None, None


def _family_disclosure(
    family_size: int, n_untested: int, alpha: float, *, n_equivalence: int = 0, campaign_wide: bool = False
) -> str:
    """The sentence a writer quotes about one family, composed from what the family holds."""
    asked = "the campaign holds" if campaign_wide else "this question asks about"
    if family_size == 0:
        sentence = f"No comparison {asked} carried a p, so it supports no separation between arms."
    else:
        equivalence = (
            f", with {n_equivalence} equivalence test{'s' if n_equivalence != 1 else ''} against a declared margin,"
            if n_equivalence
            else ""
        )
        sentence = (
            f"{family_size} comparison{'s' if family_size != 1 else ''} {asked} carried a p and "
            f"{'were' if family_size != 1 else 'was'}{equivalence} corrected together by Holm's method at "
            f"α={format_number(alpha)}; a separation stands only where the adjusted p is below it and the interval "
            f"excludes zero. Each interval "
            f"is at {format_number(100 * (1 - alpha / family_size))}%, so the family's intervals hold together at "
            f"{format_number(100 * (1 - alpha))}%."
        )
    if n_untested:
        sentence += f" {n_untested} more could not be tested; each says why."
    return f"{_NO_QUESTION_FAMILY} {sentence}" if campaign_wide else sentence


def _corrected_family(question_id: str | None, axes: list[MeritAxis], tested: list[_Tested]) -> ComparisonFamily:
    """Correct one family's tests together, and read each comparison's verdict and interval off the result.

    Every separation p and every equivalence p is Holm-adjusted as one family, capped at the separation
    count (:func:`~threetears.evals.analysis.stats.holm_adjust`'s ``max_true``): a comparison's two
    hypotheses — no difference, a difference of at least the margin — cannot both be true, so the family's
    error stays at α over every verdict it can reach. Each interval is at ``1 − α/m`` (Bonferroni over the
    ``m`` separations), which holds the family's intervals together at ``1 − α``.

    **A separation is read off its interval as well as its adjusted p** (#597): ``improved`` or ``regressed``
    needs the interval, where one exists, to exclude zero. Holm's later steps reject some comparisons whose
    Bonferroni interval still reaches zero, and a reader must never see a separation beside an interval that
    includes no change. An interval that excludes zero has ``m · p_raw < α``, which Holm always rejects, so the
    rule is Bonferroni's wherever an interval exists: a subset of Holm's rejections, so the family's error stays
    at α, at the cost of the separations Holm alone would add. Where every case moved by one amount no t
    interval exists, and the interval is the bounded test's on the declared range, the test whose p was
    corrected (#597); with no range that comparison was never tested and is not in the family.

    Args:
        question_id: The question the family serves, or None for the campaign-wide family.
        axes: The question's merit axes.
        tested: The family's comparisons, tested and uncorrected.

    Returns:
        The corrected family.
    """
    family_size = sum(1 for one in tested if one.p_raw is not None)
    n_equivalence = sum(1 for one in tested if one.equivalence_p_raw is not None)
    raw = [p for one in tested for p in (one.p_raw, one.equivalence_p_raw) if p is not None]
    adjusted = iter(holm_adjust(raw, max_true=family_size) if raw else [])
    interval_level = 1.0 - SIGNIFICANCE_ALPHA / family_size if family_size else None
    comparisons = []
    for one in tested:
        comparison = one.comparison
        if one.p_raw is not None and interval_level is not None:
            p_adjusted = next(adjusted)
            equivalence_p_adjusted = next(adjusted) if one.equivalence_p_raw is not None else None
            control_sample, contrast_sample = one.samples
            interval = (
                # No spread: no t interval exists, and on the declared range the bounded test gives one, so a
                # separation is never shown with no interval beside it (#597), nor identical values without bounds.
                bounded_difference_interval(
                    control_sample,
                    contrast_sample,
                    paired=comparison.test == "paired",
                    value_range=one.value_range,
                    confidence=interval_level,
                )
                if comparison.basis in ("bounded", "identical") and one.value_range is not None
                else difference_interval(
                    control_sample,
                    contrast_sample,
                    paired=comparison.test == "paired",
                    confidence=interval_level,
                    value_range=one.value_range,
                )
            )
            verdict: ComparisonVerdict = "not_separated"
            if p_adjusted < SIGNIFICANCE_ALPHA and comparison.delta and interval_permits_separation(interval):
                verdict = "improved" if (comparison.delta > 0) == comparison.higher_is_better else "regressed"
            elif equivalence_p_adjusted is not None and equivalence_p_adjusted < SIGNIFICANCE_ALPHA:
                verdict = "equivalent"
            comparison = comparison.model_copy(
                update={
                    "p_adjusted": p_adjusted,
                    "equivalence_p_adjusted": equivalence_p_adjusted,
                    "interval": interval,
                    "verdict": verdict,
                }
            )
        comparisons.append(comparison)
    n_untested = sum(1 for comparison in comparisons if comparison.verdict == "untested")
    return ComparisonFamily(
        question_id=question_id,
        merit_axes=axes,
        family_size=family_size,
        n_equivalence_tests=n_equivalence,
        interval_level=interval_level,
        n_untested=n_untested,
        comparisons=comparisons,
        disclosure=_family_disclosure(
            family_size, n_untested, SIGNIFICANCE_ALPHA, n_equivalence=n_equivalence, campaign_wide=question_id is None
        ),
    )


def _multiple_comparisons(
    declared: CampaignDesign | None,
    realized: RealizedDesign,
    results_by_cell: dict[_CellKey, list[EvalResult]],
    records: list[ScoreRecord],
    *,
    catalog: dict[str, MetricDescriptor],
    judged_measures: list[JudgedMeasure],
    observations: _MechanismObservations,
    served: _ServedModels,
    profile: HostProfile,
    run_margins: Mapping[str, float] | None = None,
    contended: Collection[_CellKey] = frozenset(),
) -> tuple[MultipleComparisons, GuardrailReadings]:
    """Test each contrast against the control, per live question — or campaign-wide — and decide every guardrail.

    A question's family is every comparison it could draw a verdict from: each contrast cell against
    the control cell under the same rig (a contrast across rigs differs by its instrument too, so it is
    not this test), on every reading on the question's axes (:func:`_family_readings`). The family is
    corrected by Holm's method over the comparisons that carried a p, and each verdict is read off its
    adjusted p, so a family of ten cannot hand the writer a chance "difference" as a finding.

    The guardrails are read over the same pairs and the same per-case values, and kept out of every
    family: each is decided on its own (:func:`_guardrails`), so no capability gain can offset one.

    Args:
        declared: The campaign's declaration, for its live questions.
        realized: The derived design, for the control arm.
        results_by_cell: Each cell's results.
        records: The assembly's score projection rows, where judged scores live.
        catalog: The bundle's measure catalog — direction and axis per measure.
        judged_measures: The bundle's judged measures.
        observations: The campaign's mechanism observations, read for each contrast between two models.
        served: Which model answered each result's candidate calls, read for every contrast.
        profile: The host whose vocabulary this reads.
        run_margins: The margins every member run declared alike on a core rate measure (:func:`_run_margins`),
            read as that measure's margin, since its descriptor declares none.
        contended: The cells holding latency read under concurrency, which the bundle withheld — named as the
            reason a comparison or a guardrail on elapsed time could not be read, where it could not.

    Returns:
        One family per live question, in declaration order; one campaign-wide family over every reading when
        the campaign declares no live question; or none, with the reason, when no control resolved. Beside
        them, every guardrail decided for each arm against the control.
    """
    guardrail_readings = _guardrail_readings(catalog, judged_measures, declared)
    unstamped = _unstamped_dimensions(result for members in results_by_cell.values() for result in members)
    if realized.control_arm is None:
        return MultipleComparisons(withheld=_NO_CONTROL_TO_COMPARE_AGAINST), _guardrails_of(
            guardrail_readings,
            [],
            unstamped=unstamped,
            withheld=_NO_CONTROL_TO_HOLD_AGAINST if guardrail_readings else None,
        )
    questions = declared.live_questions() if declared is not None else []
    # A campaign that asked nothing is not thereby licensed to report chance differences: its family is every
    # comparison it holds, on every reading, corrected as one.
    scopes: list[tuple[str | None, list[MeritAxis]]] = (
        [(question.id, list(question.merit_axes)) for question in questions] if questions else [(None, [])]
    )
    control_variant = realized.control_arm.variant_key

    judged_rows: dict[str, list[ScoreRecord]] = {}
    for record in records:
        judged_rows.setdefault(record.result_id, []).append(record)
    values = {key: _per_case_values(members, judged_rows, profile=profile) for key, members in results_by_cell.items()}
    pairs = [
        (control_key, contrast_key)
        for control_key in sorted(values)
        if control_key[0] == control_variant
        for contrast_key in sorted(values)
        if contrast_key[1] == control_key[1] and contrast_key[0] != control_variant
    ]

    # One answer per pair, stated on every reading the pair is compared on: which pair diverged is a fact
    # about the two arms, and a writer reading one comparison should not have to find it on another.
    pair_confounds = {
        pair: _model_contrast_confounds(
            results_by_cell[pair[0]], results_by_cell[pair[1]], observations, profile=profile
        )
        + _served_model_confounds((result.id for key in pair for result in results_by_cell[key]), served)
        for pair in pairs
    }
    # A judged dimension's scale bounds the interval on its delta; it declares no margin, so no equivalence reads it.
    judged_ranges = {judged.name: judged.value_range for judged in judged_measures if judged.value_range is not None}
    families = []
    for question_id, axes in scopes:
        readings = _family_readings(axes, catalog, judged_measures)
        tested: list[_Tested] = []
        for reading in sorted(readings):
            for control_key, contrast_key in sorted(pairs, key=lambda pair: (pair[0][1], pair[1][0])):
                control_values = values[control_key].get(reading, {})
                contrast_values = values[contrast_key].get(reading, {})
                if not control_values and not contrast_values:
                    continue
                threshold, margin_source = _margin_of(reading, catalog, run_margins or {})
                one = _compare(
                    reading,
                    readings[reading],
                    (control_key, control_values),
                    (contrast_key, contrast_values),
                    threshold=threshold,
                    margin_source=margin_source,
                    value_range=(
                        catalog[reading[1]].value_range if reading[0] == "measure" else judged_ranges.get(reading[1])
                    ),
                    no_turn=tuple(
                        side
                        for side, key in (("control", control_key), ("arm", contrast_key))
                        if reading[0] == "measure"
                        and summary_population(catalog[reading[1]], "scored") == "delivered"
                        and _took_no_turn(results_by_cell[key])
                    ),
                    contended=_contended_sides(reading, (control_key, contrast_key), catalog, contended),
                )
                confounds = pair_confounds[(control_key, contrast_key)]
                tested.append(
                    one._replace(comparison=one.comparison.model_copy(update={"mechanism_confounds": confounds}))
                )
        families.append(_corrected_family(question_id, axes, tested))
    checks = [
        _guardrail_check(
            reading,
            guardrail_readings[reading],
            (control_key, values[control_key].get(reading, {})),
            (contrast_key, values[contrast_key].get(reading, {})),
            contended=_contended_sides(reading, (control_key, contrast_key), catalog, contended),
        )
        for reading in sorted(guardrail_readings)
        for control_key, contrast_key in sorted(pairs, key=lambda pair: (pair[0][1], pair[1][0]))
        if values[control_key].get(reading) or values[contrast_key].get(reading)
    ]
    return MultipleComparisons(families=families), _guardrails_of(guardrail_readings, checks, unstamped=unstamped)


def _reading_scope(
    declared: CampaignDesign | None, catalog: dict[str, MetricDescriptor], judged_measures: list[JudgedMeasure]
) -> ReadingScope:
    """Label the readings no live question asks about exploratory — or, with no question, say so once.

    The same rule the families are scoped by (:func:`~threetears.evals.kernel.declaration.exploratory_reading`),
    so a reading is exploratory exactly when no question's family could test it.
    """
    questions = declared.live_questions() if declared is not None else []
    if not questions:
        return ReadingScope(questions_declared=False, disclosure=exploratory_disclosure(declared))
    return ReadingScope(
        questions_declared=True,
        exploratory_measures=sorted(
            name
            for name, descriptor in catalog.items()
            if not descriptor.guardrail and exploratory_reading(descriptor.merit_axis, questions)
        ),
        exploratory_dimensions=sorted(
            measure.name
            for measure in judged_measures
            if measure.axis == "capability" and exploratory_reading(JUDGED_MERIT_AXIS, questions)
        ),
    )


#: Why no guardrail was checked, where some reading is one.
_NO_CONTROL_TO_HOLD_AGAINST = (
    "No control resolved, so there is no arm to hold another against and no guardrail was checked: every guardrail "
    "here is unchecked, which is not the same as held."
)


class _Guardrail(NamedTuple):
    """What a guardrail check needs to know about its reading."""

    higher_is_better: bool
    #: The declared margin — a measure's ``materiality_threshold``, a judged dimension's from the campaign's
    #: ``guardrail_margins`` — or None when none is declared.
    margin: float | None
    value_range: tuple[float, float] | None


def _guardrail_readings(
    catalog: dict[str, MetricDescriptor], judged_measures: list[JudgedMeasure], declared: CampaignDesign | None
) -> dict[tuple[ReadingKind, str], _Guardrail]:
    """Every guardrail reading the bundle carries: the measures declared one and the boundary judged dimensions.

    A measure's margin is its declared ``materiality_threshold``, the one margin a measure has; a judged
    dimension's is the one the campaign declares for it (``CampaignDesign.guardrail_margins``), and with none
    it is held at zero change.
    """
    judged_margins = {entry.dimension: entry.margin for entry in declared.guardrail_margins} if declared else {}
    readings: dict[tuple[ReadingKind, str], _Guardrail] = {
        ("measure", name): _Guardrail(
            descriptor.higher_is_better, descriptor.materiality_threshold, descriptor.value_range
        )
        for name, descriptor in catalog.items()
        if descriptor.guardrail and descriptor.higher_is_better is not None
    }
    for measure in judged_measures:
        if measure.axis == "boundary":
            readings[("judged", measure.name)] = _Guardrail(
                measure.higher_is_better, judged_margins.get(measure.name), measure.value_range
            )
    return readings


def _contended_sides(
    reading: tuple[ReadingKind, str],
    pair: tuple[_CellKey, _CellKey],
    catalog: Mapping[str, MetricDescriptor],
    contended: Collection[_CellKey],
) -> tuple[str, ...]:
    """Which sides of a comparison on ``reading`` held latency read under concurrency: ``control``, ``arm``, both, none.

    Only a reading of elapsed time can have lost anything to the withholding, so any other reading names none.
    """
    if reading[0] != "measure" or reading[1] not in catalog or not is_latency_measure(catalog[reading[1]]):
        return ()
    return tuple(side for side, key in zip(("control", "arm"), pair, strict=True) if key in contended)


def _guardrail_check(
    reading: tuple[ReadingKind, str],
    guardrail: _Guardrail,
    control: tuple[_CellKey, dict[str, float]],
    contrast: tuple[_CellKey, dict[str, float]],
    *,
    contended: tuple[str, ...] = (),
) -> GuardrailCheck:
    """Decide one guardrail for one arm against the control under one rig (:func:`~threetears.evals.analysis.stats.guardrail_decision`).

    The samples are the ones a comparison on the same reading would read (:func:`_test_samples`), so a
    guardrail and a comparison never disagree about which cases were compared. An undecided check says
    why, in words that point at the remedy: more cases for a thin side, a declared range for a reading
    that declares none (it is never read ``held`` without one), and for a straddling interval the line it
    straddles.
    """
    (control_key, control_values), (contrast_key, contrast_values) = control, contrast
    a, b, paired = _test_samples(control_values, contrast_values)
    verdict = guardrail_decision(
        a,
        b,
        paired=paired,
        margin=guardrail.margin,
        higher_is_better=guardrail.higher_is_better,
        value_range=guardrail.value_range,
    )
    mean_a = sum(a) / len(a) if a else None
    mean_b = sum(b) / len(b) if b else None
    margin = guardrail.margin or 0.0
    reason = None
    if verdict.decision == "undecided":
        if contended and (len(a) < 2 or len(b) < 2):
            reason = (
                f"the {' and the '.join(contended)}'s latency was read while other cells or runs executed beside it, "
                "so it is withheld and the guardrail could not be checked on it (`latency_contended`)"
            )
        elif not b:
            reason = "the arm carries no value of it, so it was not checked against the control"
        elif not a:
            reason = "the control carries no value of it, so the arm could not be checked against one"
        elif len(a) < 2 or len(b) < 2:
            reason = "fewer than two cases carry it on a side, so no interval on the difference exists"
        elif verdict.refusal is not None:
            reason = verdict.refusal
        elif verdict.interval is None:
            reason = (
                "every shared case moved by the same amount, so a t interval has no width; "
                if paired
                else "each side's values are constant, so a t interval has no width; "
            ) + GUARDRAIL_HELD_NEEDS_RANGE
        else:
            line = -margin if guardrail.higher_is_better else margin
            reason = (
                f"the interval on the difference, [{format_number(verdict.interval[0])}, "
                f"{format_number(verdict.interval[1])}], reaches both sides of {format_number(line)}"
                + (" (the declared margin)" if guardrail.margin else " (no change: no margin is declared)")
                + ", so the arm is shown neither within it nor beyond it"
            )
    return GuardrailCheck(
        reading=reading[0],
        name=reading[1],
        higher_is_better=guardrail.higher_is_better,
        control=GuardrailCell(
            variant_key=control_key[0], apparatus_class_id=control_key[1], n_cases=len(a), mean=mean_a
        ),
        contrast=GuardrailCell(
            variant_key=contrast_key[0], apparatus_class_id=contrast_key[1], n_cases=len(b), mean=mean_b
        ),
        test=("paired" if paired else "unpaired") if verdict.interval is not None else None,
        delta=None if mean_a is None or mean_b is None else mean_b - mean_a,
        interval=verdict.interval,
        interval_basis=verdict.basis,
        margin=margin,
        margin_declared=guardrail.margin is not None,
        decision=verdict.decision,
        undecided_reason=reason,
    )


def _guardrails_of(
    readings: dict[tuple[ReadingKind, str], _Guardrail],
    checks: list[GuardrailCheck],
    *,
    unstamped: list[str],
    withheld: str | None = None,
) -> GuardrailReadings:
    """The bundle's guardrail section, from the readings that are guardrails and the checks run on them."""
    return GuardrailReadings(
        measures=sorted(name for kind, name in readings if kind == "measure"),
        dimensions=sorted(name for kind, name in readings if kind == "judged"),
        checks=checks,
        withheld=withheld,
        unstamped_dimensions=unstamped,
    )


#: What a judge change across the member runs means for a judged comparison, quoted when there is one.
_JUDGE_CHANGE_SENTENCE = (
    "The judged member runs were scored by {n} different judges (model, prompts or temperature), so a judged "
    "difference between arms judged differently may be the judge's rather than the subject's."
)
_JUDGE_CHANGE_DRIFT = (
    " A drift reading re-scored one side's evidence under the other side's judge ({links}); read its movement before "
    "attributing a judged difference to the subject."
)
_JUDGE_CHANGE_NO_DRIFT = (
    " Nothing measured how far the judge change alone moves the scores: re-score one side's stored evidence under "
    "the other side's judge (judge_drift_check) to read it."
)


def _judge_identity(run: EvalRun) -> tuple[str, tuple[tuple[str, str], ...], float | None] | None:
    """A judged run's judge as it recorded it at launch — pin, configs, temperature — or None for an unjudged run."""
    if run.judge_model is None:
        return None
    return run.judge_model, tuple(sorted((run.judge_config_ids or {}).items())), run.judge_temperature


def _names_level(
    judge: SecondJudge,
    level: tuple[str, tuple[tuple[str, str], ...], float | None],
    own: tuple[str, tuple[tuple[str, str], ...], float | None],
) -> bool:
    """Whether a second judge is the judge ``level`` names, asked of a run judged as ``own``.

    The model must be the level's pin; the prompts are the run's own when the second judge named none; and its
    temperature, when it named none, is what each prompt asks for — the run's own sampling.
    """
    configs = own[1] if judge.config_ids is None else tuple(sorted(judge.config_ids.items()))
    temperature = own[2] if judge.temperature is None else judge.temperature
    return judge.model == level[0] and configs == level[1] and temperature == level[2]


def _judge_change(runs: Sequence[EvalRun], results_by_run: Mapping[str, Sequence[EvalResult]]) -> JudgeChange:
    """Each judge the judged member runs recorded, and every drift reading among them that spans two of them."""
    by_level: dict[tuple[str, tuple[tuple[str, str], ...], float | None], list[EvalRun]] = {}
    for run in runs:
        if (identity := _judge_identity(run)) is not None:
            by_level.setdefault(identity, []).append(run)
    ordered = sorted(by_level, key=lambda level: (level[0], level[1], "" if level[2] is None else str(level[2])))
    levels = [
        JudgeIdentityLevel(
            judge_model=level[0],
            judge_config_ids=dict(level[1]),
            judge_temperature=level[2],
            run_ids=sorted(run.id for run in by_level[level]),
            variant_keys=sorted(
                {key for run in by_level[level] if (key := variant_key_of_run(results_by_run.get(run.id, [])))}
            ),
        )
        for level in ordered
    ]
    if len(levels) < 2:
        return JudgeChange(levels=levels)
    links = []
    for source_index, source in enumerate(ordered):
        for target_index, target in enumerate(ordered):
            if source_index == target_index:
                continue
            spanning = [
                (run, judging.pass_id)
                for run in by_level[source]
                for result in results_by_run.get(run.id, [])
                for judging in result.judge_seconds
                if _names_level(judging.judge, target, source)
            ]
            if not spanning:
                continue
            pass_ids = sorted({pass_id for _, pass_id in spanning})
            run_ids = sorted({run.id for run, _ in spanning})
            read = [
                result.model_copy(update={"judge_seconds": [j for j in result.judge_seconds if j.pass_id in pass_ids]})
                for run_id in run_ids
                for result in results_by_run.get(run_id, [])
            ]
            links.append(
                JudgeDriftLink(
                    from_level=source_index,
                    to_level=target_index,
                    run_ids=run_ids,
                    pass_ids=pass_ids,
                    drift=judge_drift(read),
                )
            )
    sentence = _JUDGE_CHANGE_SENTENCE.format(n=len(levels))
    if links:
        named = ", ".join(f"level {link.from_level} under level {link.to_level}'s judge" for link in links)
        sentence += _JUDGE_CHANGE_DRIFT.format(links=named)
    else:
        sentence += _JUDGE_CHANGE_NO_DRIFT
    return JudgeChange(levels=levels, drift_links=links, sentence=sentence)


def _judged_by_cell(judged_measures: list[JudgedMeasure]) -> dict[_CellKey, list[JudgedReading]]:
    """The judged measures transposed — each cell's readings, one per dimension it was scored on.

    Args:
        judged_measures: Judged measures, from :func:`_judged_measures`, sorted by name.

    Returns:
        Each cell's readings, sorted by dimension, keyed by the cell's coordinates.
    """
    judged_by_cell: dict[_CellKey, list[JudgedReading]] = {}
    # Judged measures are sorted by name, so each cell's readings arrive sorted by dimension.
    for measure in judged_measures:
        for arm in measure.arms:
            judged_by_cell.setdefault((arm.variant_key, arm.apparatus_class_id), []).append(
                JudgedReading(
                    dimension=measure.name,
                    mean=arm.mean,
                    sem=arm.sem,
                    n=arm.n,
                    n_independent=arm.n_independent,
                    case_means=arm.case_means,
                    n_infra_excluded=arm.n_infra_excluded,
                    n_cannot_tell=arm.n_cannot_tell,
                    evidence_tier=arm.evidence_tier,
                )
            )
    return judged_by_cell


#: How a stratum is keyed while the strata are built: a declared stratum, or None for the cases that declare none.
_StratumKey = str | None


def _cell_strata(
    results_by_cell: dict[_CellKey, list[EvalResult]],
    stratum_of_case: dict[str, str | None],
    records: list[ScoreRecord],
    design: CampaignDesign | None,
    *,
    tiers: list[JudgeEvidenceTier],
    profile: HostProfile,
) -> dict[_CellKey, list[StratumFacts]]:
    """Each cell read again per stratum of its cases — the same walk and transposition, over each stratum's results.

    A cell is broken down only when some case of it declares a stratum; one whose cases declare none has no
    entry here, and reads exactly as an unstratified cell does. Within a broken-down cell every result lands
    in one stratum — its case's, or the undeclared one — so the strata partition the cell.

    **The judged readings are scored over each stratum's own slice** by :func:`_judged_measures`, with the
    campaign's evidence tiers: a judge's reliability is measured over the whole campaign, not one kind of
    case of it.

    Args:
        results_by_cell: Each cell's results, from :func:`_results_by_cell`.
        stratum_of_case: The stratum each case declares, by case id; a case that did not resolve is absent
            and reads as declaring none.
        records: The score projection, for judged dimensions.
        design: The campaign's declaration, for the bar a judged dimension carries.
        tiers: The judges' evidence tiers.
        profile: The host whose vocabulary this reads.

    Returns:
        Each broken-down cell's strata — named strata in name order, then the undeclared one — keyed by the
        cell's coordinates.
    """
    by_stratum: dict[_StratumKey, dict[_CellKey, list[EvalResult]]] = {}
    for key, members in results_by_cell.items():
        if not any(stratum_of_case.get(result.test_case_id) is not None for result in members):
            continue
        for result in members:
            by_stratum.setdefault(stratum_of_case.get(result.test_case_id), {}).setdefault(key, []).append(result)
    strata: dict[_CellKey, list[StratumFacts]] = {}
    for stratum in sorted(by_stratum, key=lambda name: (name is None, name or "")):
        slice_by_cell = by_stratum[stratum]
        result_ids = {result.id for members in slice_by_cell.values() for result in members}
        judged = _judged_by_cell(
            _judged_measures(
                [record for record in records if record.result_id in result_ids], slice_by_cell, design, tiers=tiers
            )
        )
        for key, members in slice_by_cell.items():
            strata.setdefault(key, []).append(
                StratumFacts(
                    stratum=stratum,
                    n_observations=len(members),
                    n_cases=len({result.test_case_id for result in members}),
                    n_infra_excluded=len(members) - len(_non_faulted(members)),
                    n_candidate_failed=_candidate_failed(members),
                    n_no_turn=_no_turn(members),
                    measures=_measure_collection(members, profile=profile, undeclared="scored"),
                    judged=judged.get(key, []),
                )
            )
    return strata


def _cell_measures(
    cells: list[Cell],
    results_by_cell: dict[_CellKey, list[EvalResult]],
    judged_measures: list[JudgedMeasure],
    *,
    strata: dict[_CellKey, list[StratumFacts]],
    short_runs: dict[str, str],
    incomplete_runs: dict[str, str],
    profile: HostProfile,
    control: str | None,
    names: Mapping[str, str],
) -> list[CellFacts]:
    """Everything measured in each cell — the facts an analysis freezes as its decision surface.

    **One population, the bars' own**: each cell's results the harness did not fault
    (:func:`_non_faulted`), so a measure's value here and a :class:`BarVerdict` on the same cell and
    measure are one mean over one set of observations, never two readings a reader has to reconcile.
    A cost or latency measure is read over the ``delivered`` part of it — the turns the candidate took — by
    the same walk for the cell and the bar alike, and the failures that took none are counted beside it
    (``n_no_turn``, of ``n_candidate_failed``).

    **The judged readings are the judged measures, transposed** — read off what
    :func:`_judged_measures` produced rather than scored again, so a cell's reading of a dimension
    and that dimension's arm for the cell are the same numbers by construction, not by two
    population rules happening to agree.

    Args:
        cells: The pooled cells, for their replication.
        results_by_cell: Each cell's results, from :func:`_results_by_cell`.
        judged_measures: The bundle's judged measures, from :func:`_judged_measures`.
        strata: Each broken-down cell's strata, from :func:`_cell_strata`; a cell absent from it carries none.
        short_runs: The bundle's short-run sentences, by run id.
        incomplete_runs: The bundle's incomplete-run statuses, by run id.
        profile: The host whose vocabulary this reads.
        control: The declared control's variant key, or None.
        names: The bundle's arm names (:func:`~threetears.evals.analysis.arms.arm_names`), which order the arms.

    Returns:
        One entry per cell, in the decision surface's row order
        (:func:`~threetears.evals.analysis.arms.surface_order`): the control's cells first, as the reference,
        then every other arm alphabetically by name, each arm's rigs by id. Not a ranking.
    """
    judged_by_cell = _judged_by_cell(judged_measures)
    facts = []
    for cell in sorted(cells, key=lambda c: (c.variant_key, c.apparatus_class_id)):
        key = (cell.variant_key, cell.apparatus_class_id)
        members = results_by_cell[key]
        counted = _non_faulted(members)
        run_ids = _member_run_ids(members)
        facts.append(
            CellFacts(
                variant_key=cell.variant_key,
                apparatus_class_id=cell.apparatus_class_id,
                run_ids=run_ids,
                n_observations=cell.n_observations,
                n_cases=cell.n_cases,
                repeats_per_case_min=cell.repeats_per_case_min,
                repeats_per_case_max=cell.repeats_per_case_max,
                n_infra_excluded=len(members) - len(counted),
                n_candidate_failed=_candidate_failed(members),
                n_no_turn=_no_turn(members),
                # Every member, faulted ones included: the walk leaves a fault out of each measure whose
                # population is `scored` and keeps it in one whose population is `all_observed`.
                measures=_measure_collection(members, profile=profile, undeclared="scored"),
                judged=judged_by_cell.get(key, []),
                short_runs={run_id: short_runs[run_id] for run_id in run_ids if run_id in short_runs},
                incomplete_runs={run_id: incomplete_runs[run_id] for run_id in run_ids if run_id in incomplete_runs},
                strata=strata.get(key, []),
            )
        )
    return surface_order(facts, control=control, names=names)


def _time_axis(
    runs: list[EvalRun],
    results_by_run: dict[str, list[EvalResult]],
    observations: list[Observation],
    classes: dict[str, ApparatusClass],
    records: list[ScoreRecord],
    design: CampaignDesign | None,
    *,
    tiers: list[JudgeEvidenceTier],
    short_runs: dict[str, str],
    incomplete_runs: dict[str, str],
    profile: HostProfile,
    names: Mapping[str, str],
) -> tuple[TimeAxis | None, str | None]:
    """Place the campaign's runs in time, or say why they cannot be.

    **Each position's cells are the decision surface's own algebra over that position's runs**: the
    observations re-pooled (:func:`~threetears.evals.analysis.cells.pool_observations`), the judged
    dimensions summarised (:func:`_judged_measures`) and the cells measured (:func:`_cell_measures`) by the
    same functions the whole surface is, so a cell's figure at one build and its figure over the campaign
    differ only in which observations they read. A run that measured nothing has no place in time — it
    cannot say what was measured when — and is left out of every position.

    Args:
        runs: The resolved member runs, in creation order.
        results_by_run: Each run's results.
        observations: Every observation the cell algebra pooled.
        classes: Apparatus class id → the class, as the campaign's pooling read them.
        records: The score projection, for judged dimensions.
        design: The campaign's declaration, for the bar a judged dimension carries.
        tiers: The judges' evidence tiers, which every judged reading carries — the campaign's own, since a
            judge's reliability is measured over the whole campaign, not one position of it.
        short_runs: The bundle's short-run sentences, by run id.
        incomplete_runs: The bundle's incomplete-run statuses, by run id.
        profile: The host, whose ``release_label`` names its builds.
        names: The bundle's arm names, which order each position's cells as the whole surface's are.

    Returns:
        ``(axis, None)`` when the measuring runs span two or more positions, else ``(None, why)``.
    """
    measuring = [run for run in runs if results_by_run[run.id]]
    if not measuring:
        return None, "no run produced an observation, so nothing was measured at any time"
    basis, grouped, release_why = _time_positions(measuring, results_by_run, profile=profile)
    if len(grouped) < 2:
        return None, f"every run started on one day ({grouped[0][0]}) and {release_why}"
    positions = []
    for key, members in grouped:
        member_ids = {run.id for run in members}
        slice_cells, _, _ = pool_observations([obs for obs in observations if obs.apparatus_ref in member_ids], classes)
        slice_results = [result for run in members for result in results_by_run[run.id]]
        result_ids = {result.id for result in slice_results}
        by_cell = _results_by_cell(slice_cells, slice_results)
        judged = _judged_measures(
            [record for record in records if record.result_id in result_ids], by_cell, design, tiers=tiers
        )
        positions.append(
            TimePosition(
                key=key,
                first_run_at=members[0].created_at,
                last_run_at=members[-1].created_at,
                run_ids=sorted(member_ids),
                # A position's cells are not broken down by stratum: the breakdown is read over the whole
                # campaign, and per position it would multiply the bundle by every stratum at every build.
                cells=_cell_measures(
                    slice_cells,
                    by_cell,
                    judged,
                    strata={},
                    short_runs=short_runs,
                    incomplete_runs=incomplete_runs,
                    profile=profile,
                    control=design.control if design else None,
                    names=names,
                ),
            )
        )
    if basis == "release":
        return TimeAxis(basis=basis, release_label=profile.release_label, positions=positions), None
    return TimeAxis(basis=basis, basis_reason=release_why, positions=positions), None


def _time_positions(
    runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
) -> tuple[TimeAxisBasis, list[tuple[str, list[EvalRun]]], str]:
    """Group the measuring runs into time positions, earliest first.

    By the host's release label when it declares one, every run recorded it and the runs span two values of
    it; otherwise by the UTC day each run was created on. Either way the positions are ordered by when their
    earliest run was created, which is the one order every run records — a label is the host's string and
    nothing here can sort it.

    Args:
        runs: The runs that measured something, in creation order.
        results_by_run: Each run's results, which a label reader may read.
        profile: The host, whose ``release_label`` names its builds.

    Returns:
        ``(basis, positions, why_not_builds)``: each position's key and its runs in creation order, and — on a
        ``date`` basis — why the positions are not builds, in the words of the branch that found it, naming
        the runs that recorded no label where that is the reason. Empty on a ``release`` basis.

    Raises:
        RuntimeError: The release label names no registered input — registration refuses that, so the
            profile was built around its own check.
    """
    release_why = "the host labels no build"
    if profile.release_label is not None:
        declared = profile.sweepables.get(profile.release_label)
        if declared is None:
            raise RuntimeError(f"release_label {profile.release_label!r} names no registered input")
        # Normalised ONCE, and every check below reads the normalised label: a position's key is stripped
        # where it is stored (the base stance), so two labels that differ only in whitespace — a version read
        # from a file with its trailing newline — are one build, and grouping them as two would hand the axis
        # two positions it then refuses as repeated.
        values = {run.id: _release_label(declared.read(run, results_by_run[run.id])) for run in runs}
        unrecorded = [run.id for run in runs if values[run.id] is None]
        if unrecorded:
            named = ", ".join(sorted(unrecorded))
            release_why = f"{len(unrecorded)} of {len(runs)} runs recorded no {profile.release_label} ({named})"
        elif len(set(values.values())) > 1:
            return "release", _group_in_order(runs, lambda run: values[run.id] or ""), ""
        else:
            release_why = f"every run recorded one {profile.release_label} ({next(iter(values.values()))})"
    return "date", _group_in_order(runs, _utc_day), release_why


def _release_label(value: object) -> str | None:
    """A run's release label as a time position keys it: stripped, and ``None`` when it recorded none or only blanks."""
    if value is None:
        return None
    return str(value).strip() or None


def _group_in_order(runs: list[EvalRun], key: Callable[[EvalRun], str]) -> list[tuple[str, list[EvalRun]]]:
    """Group runs by ``key``, each group in creation order and the groups ordered by their earliest run."""
    groups: dict[str, list[EvalRun]] = {}
    for run in sorted(runs, key=lambda r: (r.created_at, r.id)):
        groups.setdefault(key(run), []).append(run)
    return list(groups.items())


def _utc_day(run: EvalRun) -> str:
    """The UTC calendar day a run was created on, as YYYY-MM-DD."""
    created = datetime.fromisoformat(run.created_at)
    if created.tzinfo is not None:
        created = created.astimezone(UTC)
    return created.date().isoformat()


def _time_axis_cells(bundle: AnalysisContextBundle) -> list[CellFacts]:
    """Every cell on the bundle's time axis, at every position — empty when there is no axis."""
    return [cell for position in bundle.time_axis.positions for cell in position.cells] if bundle.time_axis else []


def cell_measure_facts(bundle: AnalysisContextBundle) -> dict[str, MeasureFacts]:
    """What each measure named in the bundle's cells IS — the catalogue half of a decision surface.

    Read off ``measure_catalog``, which describes every measure collection in the bundle, the cells'
    included — so the unit, axis and direction frozen beside a cell's value are the ones the generator
    was handed for the same name, never a second lookup that could disagree with it.

    Args:
        bundle: An assembled bundle.

    Returns:
        One entry per measure name appearing in any cell — the time axis's included — in name order.
    """
    cells = [*bundle.cell_measures, *_time_axis_cells(bundle)]
    names = sorted(
        {measure.name for cell in cells for collection in _cell_collections(cell) for measure in collection.measures}
    )
    return {
        name: MeasureFacts(
            reader_name=bundle.measure_catalog[name].reader_name,
            unit=bundle.measure_catalog[name].unit,
            merit_axis=bundle.measure_catalog[name].merit_axis,
            higher_is_better=bundle.measure_catalog[name].higher_is_better,
            materiality_threshold=bundle.measure_catalog[name].materiality_threshold,
            population=bundle.measure_catalog[name].population,
            scale=bundle.measure_catalog[name].scale,
            guardrail=bundle.measure_catalog[name].guardrail,
        )
        for name in names
    }


def cell_dimension_facts(bundle: AnalysisContextBundle) -> dict[str, JudgedDimensionFacts]:
    """What each judged dimension scored in the bundle's cells IS — the judged half of a decision surface.

    Read off ``judged_measures``, the one place the bundle carries a dimension's polarity and scale,
    so the direction a merit claim on a judged score is read in, and the scale a chart draws it on,
    are the ones the generator was handed for the same name. Every dimension a cell carries has an
    entry there by construction: a cell's judged readings are ``judged_measures`` transposed
    (:func:`_cell_measures`).

    Args:
        bundle: An assembled bundle.

    Returns:
        One entry per dimension appearing in any cell, in name order.
    """
    described = {measure.name: measure for measure in bundle.judged_measures}
    names = sorted({reading.dimension for cell in bundle.cell_measures for reading in cell.judged})
    return {
        name: JudgedDimensionFacts(
            higher_is_better=described[name].higher_is_better,
            value_range=described[name].value_range,
            scale=described[name].scale,
            axis=described[name].axis,
        )
        for name in names
    }


def bundle_decision_surface(bundle: AnalysisContextBundle) -> DecisionSurface:
    """Freeze the bundle's per-cell facts into the decision surface an analysis — or a code-only report — reads.

    Copied, never recomputed: each per-cell number is the one assembly computed over the evidence. The
    control is the declaration's own variant key — the same value ``declared_design`` carries and the arm
    table marks as the control — so the two cannot name different arms.

    Args:
        bundle: The evidence set.

    Returns:
        The decision surface.
    """
    return DecisionSurface(
        control_variant_key=bundle.declared_design.control if bundle.declared_design else None,
        cells=bundle.cell_measures,
        bars=bundle.bar_adjudications,
        measures=cell_measure_facts(bundle),
        dimensions=cell_dimension_facts(bundle),
        time_axis=bundle.time_axis,
        frontier_dominance=_frontier_dominance(bundle.frontier),
        frontier_disqualified={
            point.variant_key: list(point.disqualified_by)
            for subject in bundle.frontier.subjects
            for point in subject.points
            if point.disqualified_by
        },
        rubric_threshold=bundle.frontier.rubric_threshold,
        guardrails=bundle.guardrails,
    )


def _frontier_dominance(frontier: FrontierResult) -> dict[str, FrontierDominance]:
    """Each variant's standing on the frontier lens — the verdict a frontier chart draws, never recomputed.

    The lens decides domination by test over per-case values (:func:`~threetears.evals.analysis.reporting.compute_frontier`),
    which a frozen surface does not carry, so a chart deciding it again from the surface's means would be a
    second rule for one question, and on means it called one of two identical arms dominated a third of the
    time. Keyed by variant because a cell is one; a variant the lens placed as more than one point (under two
    identity versions, or two subjects) has no one standing and is left out, so a chart reads it as untested.
    A point stored before domination was tested carries no standing either, and is left out the same way.

    Args:
        frontier: The bundle's frontier lens.

    Returns:
        ``{variant_key: dominance}``.
    """
    placed: dict[str, list[FrontierDominance | None]] = {}
    for subject in frontier.subjects:
        for point in subject.points:
            placed.setdefault(point.variant_key, []).append(point.dominance)
    return {key: standings[0] for key, standings in placed.items() if len(standings) == 1 and standings[0] is not None}


class InsightStanding(NamedTuple):
    """Where each listed insight's minting analysis stands — the two states a reader must be told.

    Attributes:
        retracted: Insight id → the ARCHIVED analysis that minted it. No reader may present it as
            live, and no generation reads it as prior context.
        orphaned: Insight id → the analysis it names that no longer resolves (a hard delete keeps the
            insights it minted). NOT retracted — nothing says it was shown false, so generations still
            read it — but its provenance can no longer be followed, and a listing must say so.
    """

    retracted: dict[str, str]
    orphaned: dict[str, str]


def insight_standing(
    insights: Iterable[EvalInsight],
    analysis_archived: Callable[[str], bool | None],
) -> InsightStanding:
    """Classify each insight by where its minting analysis stands, asking the store once per analysis.

    An analysis is archived when it was shown false, and the insights it minted are the part of
    it that keeps travelling: they are fed back to every later generation over the same subject
    as prior context, so an archive that left them in place kept the falsehood steering analyses
    while the report that asserted it was marked retired. Deleting each such insight is still the
    intended cleanup; this is what makes the archive itself sufficient to stop the steering, so a
    delete that is forgotten or deferred no longer leaves the generator reading a retracted claim.

    **Derived from the analysis, never copied onto the insight.** The archived flag is the one
    source of truth, so un-archiving an analysis restores its insights with nothing to un-mark, and
    no reader can disagree with another about whether an insight is retracted. Every reader that
    presents insights asks this function — the bundle's ``prior_insights``, and the ledger listing.

    An insight naming no analysis is neither: it never had a back-reference to lose. One naming an
    analysis that no longer resolves is ORPHANED rather than retracted, and the listings disclose it
    from ``orphaned``.

    Args:
        insights: The insights to classify.
        analysis_archived: Whether one analysis is archived, ``None`` when it does not resolve.
            Called at most once per distinct analysis, and only for insights that name one.

    Returns:
        Both maps, each in insight-id order and empty when nothing is in that state.
    """
    standing: dict[str, bool | None] = {}
    retracted: dict[str, str] = {}
    orphaned: dict[str, str] = {}
    for insight in insights:
        source = insight.source_analysis_id
        if not source:
            continue
        if source not in standing:
            standing[source] = analysis_archived(source)
        if standing[source] is True:
            retracted[insight.id] = source
        elif standing[source] is None:
            orphaned[insight.id] = source
    return InsightStanding(dict(sorted(retracted.items())), dict(sorted(orphaned.items())))


def retracted_insights(
    insights: Iterable[EvalInsight],
    analysis_archived: Callable[[str], bool | None],
) -> dict[str, str]:
    """Which of these insights an ARCHIVED analysis minted — :func:`insight_standing`'s ``retracted``.

    Args:
        insights: The insights to classify.
        analysis_archived: Whether one analysis is archived, ``None`` when it does not resolve.

    Returns:
        Insight id → the id of the archived analysis that minted it, in insight-id order.
    """
    return insight_standing(insights, analysis_archived).retracted


def insight_restatement_key(statement: str) -> str:
    """The claim an insight states, as two insights stating it compare — case, spacing and a final period aside.

    The one rule for "these two insights say the same thing", read by the bundle (which carries one insight per
    claim) and by the ledger write (which replaces a live insight a new one restates rather than adding a
    duplicate). Deliberately literal: two sentences that mean the same thing in different words are two keys,
    because deciding they are one claim is a judgement, and a wrong merge here would retire a claim nobody
    restated.

    Args:
        statement: An insight's statement.

    Returns:
        The comparison key.
    """
    return " ".join(statement.casefold().split()).rstrip(".").rstrip()


def superseding_insights(
    minted: Sequence[EvalInsight],
    ledger: Iterable[EvalInsight],
    analysis_archived: Callable[[str], bool | None],
) -> list[EvalInsight]:
    """The insights a generation writes: each one minted, taking the id of the live insight it restates.

    A generation mints one insight per finding that states one, and regenerating over the same evidence
    states the same claims again. Written as new rows, every regeneration grew the ledger by its whole
    output, and every one of those rows rode into the next paid prompt. So a minted insight whose claim
    (:func:`insight_restatement_key`) a LIVE ledger insight of the subject already states is written under
    that insight's id: the store's upsert replaces the old row with the restatement — its statement,
    confidence, evidence and minting analysis now the newer ones — and the ledger keeps its size. That is the
    insight's ``invalidation_trigger``, carried out.

    A RETRACTED insight (its analysis archived, :func:`retracted_insights`) is not live and is never replaced:
    a new analysis stating a claim an archive withdrew mints it afresh, and archiving that new analysis is
    what would withdraw it again. Where the ledger already holds several live insights stating one claim —
    written before this rule — the newest is the one replaced.

    Args:
        minted: The insights a generation returned, in its order.
        ledger: The subject's prior insights.
        analysis_archived: Whether one analysis is archived, ``None`` when it does not resolve.

    Returns:
        The insights to write, in ``minted`` order, each under its own id or the id of the insight it replaces.
        A claim the generation stated twice is written once.
    """
    prior = list(ledger)
    retracted = retracted_insights(prior, analysis_archived)
    live: dict[str, EvalInsight] = {}
    for insight in _sorted_insights([insight for insight in prior if insight.id not in retracted]):
        live.setdefault(insight_restatement_key(insight.statement), insight)
    written: list[EvalInsight] = []
    seen: set[str] = set()
    for insight in minted:
        key = insight_restatement_key(insight.statement)
        if key in seen:
            continue
        seen.add(key)
        replaced = live.get(key)
        written.append(insight if replaced is None else insight.model_copy(update={"id": replaced.id}))
    return written


def _prior_insights(live: list[EvalInsight]) -> tuple[list[EvalInsight], int]:
    """The live insights a bundle carries — the newest per claim, at most the cap — and how many it leaves out.

    Args:
        live: The subject's insights as of the cutoff, less the retracted ones.

    Returns:
        ``(carried, omitted)``: newest first, deterministic for the fingerprint.
    """
    newest: dict[str, EvalInsight] = {}
    for insight in _sorted_insights(live):
        newest.setdefault(insight_restatement_key(insight.statement), insight)
    carried, _beyond_cap = _capped(list(newest.values()), _MAX_PRIOR_INSIGHTS, weight=None)
    return carried, len(live) - len(carried)


def _sorted_insights(insights: list[EvalInsight]) -> list[EvalInsight]:
    """Order insights newest-first with an id tie-break — a stable fingerprint slice.

    ``query_insights`` already returns newest-first, but its equal-``observed_at``
    tie-break is storage's, not ours. The bundle fingerprint (and the prompt-A/B
    invariant it guards) must not depend on that, so we re-sort deterministically
    here: ``(observed_at, id)`` descending keeps newest-first and makes ties total.

    Args:
        insights: The subject's prior insights, in storage order.

    Returns:
        The same insights, deterministically ordered.
    """
    return sorted(insights, key=lambda insight: (insight.observed_at, insight.id), reverse=True)


def _telemetry_rollup(
    runs: list[EvalRun], results: list[EvalResult], budget: ProgramBudget, *, profile: HostProfile
) -> TelemetryRollup:
    """Compose the campaign-wide telemetry rollup from results + the budget lens."""
    return TelemetryRollup(
        n_runs=len(runs),
        n_results=len(results),
        n_errors=sum(1 for r in results if _result_has_error(r)),
        total_cost_usd=budget.total_cost_usd,
        incomplete_cost_usd=budget.incomplete_cost_usd,
        unattributed_cost_usd=budget.unattributed_cost_usd,
        measures=_measure_collection(results, profile=profile, undeclared="all_observed"),
        tokens=_token_rollup(results),
    )


__all__ = [
    "assemble_context_bundle",
    "bundle_decision_surface",
    "component_carrier",
    "goal_check_proofs_of",
    "insight_restatement_key",
    "measure_movement",
    "MeasureCollection",
    "MeasureSummary",
    "planning_readings",
    "PlanningReading",
    "superseding_insights",
]
