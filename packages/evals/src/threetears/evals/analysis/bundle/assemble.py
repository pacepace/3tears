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

from collections.abc import Collection, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from threetears.evals.analysis.agreement import (
    judge_agreement,
    judge_evidence_tiers,
    judge_self_agreement,
    inter_judge_agreement,
)
from threetears.evals.analysis.judge_drift import judge_drift
from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.contention import (
    withheld_latency,
    withhold_contended_latency,
)
from threetears.evals.analysis.cells import (
    pool_observations,
    subject_key_instabilities,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.reporting import measurement_window, measurement_window_disclosure, project_score_records
from threetears.evals.analysis.lenses.comparison_sets import compute_comparison_sets
from threetears.evals.analysis.lenses.frontier import compute_frontier
from threetears.evals.analysis.lenses.program_budget import compute_program_budget
from threetears.evals.analysis.completeness import completeness_disclosure
from threetears.evals.kernel.analysis_measures import MeasureCollection, MeasureSummary
from threetears.evals.kernel.campaign import EvalInsight, derive_window
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import (
    MetricDescriptor,
    declaration_of,
)
from threetears.evals.schema.models import (
    CalibrationRating,
    EvalResult,
    MeasureDeclaration,
)
from threetears.evals.analysis.bundle.caps import (
    _capped,
    _MAX_NEXT_EXPERIMENTS,
    _MAX_REFUSED_MERGES,
)
from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    host_declarations_digest,
)
from threetears.evals.analysis.bundle.insights import (
    _prior_insights,
    retracted_insights,
)
from threetears.evals.analysis.bundle.observations import _apparatus_classes, _observations, variant_key_of_run
from threetears.evals.analysis.bundle.measures import (
    _failures_as_misses,
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
    _served_models,
)
from threetears.evals.analysis.bundle.divergence import _scope_divergences
from threetears.evals.analysis.bundle.telemetry import _model_versions, _run_summary, _telemetry_rollup
from threetears.evals.analysis.bundle.coverage import _coverage_map, _declared_crossing, _factor_aliasing
from threetears.evals.analysis.bundle.cell_reads import (
    _cell_measures,
    _cell_strata,
    _judged_keys,
    _judged_measures,
    _results_by_cell,
)
from threetears.evals.analysis.bundle.bars import _bar_adjudications, _frontier_bar, _verdict_order
from threetears.evals.analysis.bundle.cell_notes import (
    _all_failed,
    _cost_unmeasured,
    _held_fixed_reading,
    _latency_contended,
    _short_cells,
)
from threetears.evals.analysis.bundle.comparisons import (
    _multiple_comparisons,
    _reading_scope,
)
from threetears.evals.analysis.bundle.judges import _judge_change
from threetears.evals.analysis.bundle.time_axis import _time_axis
from threetears.evals.analysis.bundle.surface import _measure_catalog

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.kernel.campaign import EvalCampaign
    from threetears.evals.schema.models import EvalCaseStratum, EvalRun


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


__all__ = [
    "assemble_context_bundle",
    "MeasureCollection",
    "MeasureSummary",
]
