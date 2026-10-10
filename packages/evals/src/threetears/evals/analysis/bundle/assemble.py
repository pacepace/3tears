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

from collections import defaultdict
from datetime import UTC, datetime
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol


from threetears.evals.analysis.agreement import (
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
from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.contention import (
    contended_latency_sentence,
    withheld_latency,
    withhold_contended_latency,
)
from threetears.evals.analysis.cells import (
    ApparatusClass,
    Cell,
    Observation,
    pool_observations,
    subject_key_instabilities,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.reporting import (
    METRIC_SCORE,
    FrontierResult,
    ScoreRecord,
    completeness_disclosure,
    compute_comparison_sets,
    compute_frontier,
    compute_program_budget,
    measurement_window,
    measurement_window_disclosure,
    project_score_records,
)
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
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
    no_spread_p,
    observed_mean_interval,
    paired_equivalence,
    separation_test,
)
from threetears.evals.kernel.analysis_measures import BarAdjudication, BarVerdict, MeasureCollection, MeasureSummary
from threetears.evals.kernel.campaign import EvalInsight, ReadingKind, derive_window
from threetears.evals.kernel.declaration import (
    JUDGED_MERIT_AXIS,
    BarName,
    CampaignDesign,
    UnreadableBarName,
    axis_in_question_scope,
    exploratory_reading,
    resolve_bar_name,
)
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import (
    FRONTIER_RANKING_MEASURE,
    MeritAxis,
    MetricDescriptor,
    classifier_label_of,
    declaration_of,
    describe_reported_measure,
    describe_rubric_dim,
    goal_check_measure,
    is_latency_measure,
    materiality,
    summary_population,
)

# At runtime for its field set, which tells a result-level measure from a row-level one.
from threetears.evals.schema.models import (
    CalibrationRating,
    EvalResult,
    MeasureDeclaration,
    RubricScale,
)
from threetears.evals.kernel.result_condition import delivered_a_turn
from threetears.evals.kernel.surface import (
    CellFacts,
    DecisionSurface,
    FrontierDominance,
    GuardrailCell,
    GuardrailCheck,
    GuardrailReadings,
    JudgedDimensionFacts,
    MeasureFacts,
    TimeAxis,
    TimeAxisBasis,
    TimePosition,
    all_failed_sentence,
)
from threetears.evals.kernel.usage_capture import spend_observed

from threetears.evals.analysis.bundle.caps import (
    _capped,
    _MAX_NEXT_EXPERIMENTS,
    _MAX_REFUSED_MERGES,
)

from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    CellCoordinate,
    ComparedCell,
    ComparisonFamily,
    ComparisonVerdict,
    exploratory_disclosure,
    FamilyComparison,
    HeldFixedReading,
    host_declarations_digest,
    JudgeChange,
    JudgedMeasure,
    JudgeDriftLink,
    JudgeIdentityLevel,
    MeritTier,
    MultipleComparisons,
    QuestionScope,
    ReadingScope,
    RealizedDesign,
    ShortCell,
    VerdictOrder,
)

from threetears.evals.analysis.bundle.insights import (
    _prior_insights,
    retracted_insights,
)


from threetears.evals.analysis.bundle.observations import _apparatus_classes, _observations, variant_key_of_run

from threetears.evals.analysis.bundle.measures import (
    _failures_as_misses,
    _measure_collection,
    goal_check_proofs_of,
)

from threetears.evals.analysis.bundle.design import (
    _apparatus_levels,
    _campaign_arms,
    _campaign_design,
    _declared_level_names,
    _lever_levels,
    _name_arms,
    _SurfaceFolds,
)

from threetears.evals.analysis.bundle.confounds import _apparatus_confounds, _confound_catalog

from threetears.evals.analysis.bundle.mechanisms import (
    _arm_mechanisms,
    _arm_production_footings,
    _arm_served_models,
    _design_with_mechanism_confounds,
    _mechanism_observations,
    _MechanismObservations,
    _model_contrast_confounds,
    _served_model_confounds,
    _served_models,
    _ServedModels,
)

from threetears.evals.analysis.bundle.divergence import _scope_divergences

from threetears.evals.analysis.bundle.telemetry import _model_versions, _run_summary, _telemetry_rollup

from threetears.evals.analysis.bundle.coverage import _coverage_map, _declared_crossing, _factor_aliasing

from threetears.evals.analysis.bundle.cell_reads import (
    _boundary_dimensions,
    _cannot_tell_on,
    _cell_collections,
    _cell_measures,
    _cell_strata,
    _CellKey,
    _judged_keys,
    _judged_measures,
    _judged_rows,
    _judged_values,
    _member_run_ids,
    _non_faulted,
    _results_by_cell,
    _took_no_turn,
    _unstamped_dimensions,
)

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
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


__all__ = [
    "assemble_context_bundle",
    "bundle_decision_surface",
    "MeasureCollection",
    "MeasureSummary",
    "planning_readings",
    "PlanningReading",
]
