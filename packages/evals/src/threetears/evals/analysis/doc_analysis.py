"""A campaign's analysis written as a report document: the stored analysis that links it, and the insights it mints.

:func:`doc_analysis` is what ``analysis_generate``'s job builds once the report writer has stored a document
(:func:`~threetears.evals.ops.analysis_generate`). Code builds what is stored beside the document, in every part the
writer does not write:

- **the analysis** (:class:`~threetears.evals.kernel.campaign.EvalAnalysis`): the campaign's frozen bundle (each
  aspect's, and the frame across them, read back from the snapshot the document was written from), its declaration,
  the models in play and the provenance. It holds no memo and no plan, and names its document (``report_doc_id``), so
  its ``basis`` is ``doc``. Its ``recommended_variant_key`` is the variant key of the arm the document recommends
  when ``recommends`` names exactly one; ``None`` when it names none, or two or more (a tie the document leaves to the
  reader);
- **the insights**: one per claim the document marks ``durable``, its statement the claim with every figure key written
  out, citing the runs behind the arms and aspects its keys name (the claim's text and its blocks' captions), at the
  tier the document's confidence form gives it, held to ``medium`` when the claim carries an open caveat of its own.

**Provenance** is the analyst skill's digest, never a prompt's registry key: the writer reads the skill the package
ships, so the digest is what says which words wrote the report.

**Closed dependency set.** It reads the bundle, the snapshot's report input and the stored document, and stores
nothing: storing is the service's, which holds the attempt.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple

from threetears.evals.analysis.bundle.insights import insight_restatement_key
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, LeverCoverageInput, RunSummary
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report.figure_keys import KeySubjects, key_subjects, keys_in, render
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.authored import Confidence
from threetears.evals.kernel.campaign import (
    AspectAnalysis,
    CoverageLens,
    EvalAnalysis,
    EvalInsight,
    GenerationProvenance,
    LeverCoverage,
    RunIndexEntry,
)
from threetears.evals.kernel.evidence_tiers import JUDGED_TIER_RULE
from threetears.evals.kernel.report_doc import Claim, ConfidenceForm, EvalReportDoc, ReportDoc
from threetears.evals.kernel.surface import DecisionSurface

#: ``DocAnalysis`` is the return type alone; nothing outside binds it by name.
__all__ = ["doc_analysis"]

#: The insight tier each of the document's confidence forms mints its durable claims at: a call the analyst states
#: ``high`` is ``high``, one sound only if its open caveats prove immaterial ``medium``, and a low one ``low``. A claim
#: that carries an open caveat of its own is held to ``medium`` at most (:data:`_OPEN_CAVEAT_CEILING`), whatever the
#: document's form: a claim's confidence reads its own methodology and variance, not only the call's.
_INSIGHT_CONFIDENCE: Final[dict[ConfidenceForm, Confidence]] = {
    "high": "high",
    "high_if_immaterial": "medium",
    "low": "low",
}

#: The highest tier a durable claim stated beside an open caveat of its own is minted at.
_OPEN_CAVEAT_CEILING: Final[Confidence] = "medium"

#: The tiers, weakest first, for holding a claim's tier to a ceiling.
_TIER_ORDER: Final[tuple[Confidence, ...]] = ("low", "medium", "high", "very_high")


class DocAnalysis(NamedTuple):
    """What a stored report document mints: the analysis that links it, and its insights.

    :ivar analysis: The analysis, ``basis`` ``doc``.
    :ivar insights: One per durable claim, in the document's order.
    """

    analysis: EvalAnalysis
    insights: list[EvalInsight]


def doc_analysis(
    bundle: CampaignBundle,
    report_input: ReportInput,
    stored: EvalReportDoc,
    *,
    input_digest: str,
    generator_model: str,
    skill_digest: str,
    bundle_assembled_at: str,
    token_cost: float | None,
    bounces: Mapping[str, int],
) -> DocAnalysis:
    """Build the analysis a stored report document is linked by, and its insights (this module's account).

    :param bundle: The campaign bundle the document's snapshot froze (``EvalReportSnapshot.bundle``).
    :ptype bundle: CampaignBundle
    :param report_input: The snapshot's report input, which the claims' keys resolve against.
    :ptype report_input: ReportInput
    :param stored: The stored document.
    :ptype stored: EvalReportDoc
    :param input_digest: The digest of the snapshot's report input (``EvalReportSnapshot.report_input_digest``): what
        the writer read, recorded as the analysis's ``user_message_digest``.
    :ptype input_digest: str
    :param generator_model: The model the report was written with, as the client names it.
    :ptype generator_model: str
    :param skill_digest: The analyst skill's digest (``load_skill().digest``), the provenance.
    :ptype skill_digest: str
    :param bundle_assembled_at: When the bundle was assembled (ISO-8601), the cutoff a re-assembly reads the insight
        ledger as of.
    :ptype bundle_assembled_at: str
    :param token_cost: What every turn cost together; ``None`` when any turn was unpriced, never ``0``.
    :ptype token_cost: float | None
    :param bounces: The analyst's submissions sent back, counted by what they were sent back for (``invariant``,
        ``shape``, ...), as the tally sums them; the editor's ask-again is no bounce and is not counted.
    :ptype bounces: Mapping[str, int]
    :return: The analysis and its insights.
    :rtype: DocAnalysis
    :raises UnresolvableReference: A durable claim's key does not resolve (a stored document's always do).
    :raises ValueError: An analysis or insight built from it does not validate, which is a bug, not the writer's.
    """
    doc = ReportDoc.model_validate(stored.doc)
    surfaces = bundle.surfaces()
    model_versions: dict[str, str] = {}
    for aspect in bundle.aspects:
        model_versions.update(aspect.model_versions)
    first = bundle.aspects[0]
    provenance = GenerationProvenance(
        prompt_id=skill_digest,
        prompt_version=skill_digest,
        generator_model=generator_model,
        bundle_fingerprint=bundle.fingerprint(),
        generated_at=stored.created_at,
        bundle_assembled_at=bundle_assembled_at,
        token_cost=token_cost,
        repair_attempts=sum(bounces.values()),
        repaired_refusal=_bounce_summary(bounces),
        user_message_digest=input_digest,
        cell_model_version=first.cell_model_version,
        bundle_schema_version=first.schema_version,
        host_declarations_digest=first.host_declarations_digest,
    )
    analysis = EvalAnalysis(
        id=str(uuid.uuid7()),
        recommended_variant_key=_recommended_variant_key(doc, report_input),
        campaign_id=bundle.campaign_id,
        subject_id=bundle.subject_id,
        subject_kind=bundle.subject_kind,
        behavior=bundle.behavior,
        observation_refs=bundle.run_ids,
        model_versions=model_versions,
        scope_id=bundle.scope_id,
        generation=provenance,
        design_snapshot=bundle.declaration,
        document=None,
        report_doc_id=stored.id,
        judged_tier_rule=JUDGED_TIER_RULE,
        aspects=[aspect_analysis(aspect, surfaces[aspect.aspect]) for aspect in bundle.aspects],
        campaign_bundle=bundle.frame(),
    )
    return DocAnalysis(analysis, _durable_insights(analysis, doc, bundle, report_input))


def _recommended_variant_key(doc: ReportDoc, report_input: ReportInput) -> str | None:
    """The variant key of the one arm the document recommends; ``None`` when it recommends none, or two or more."""
    if len(doc.recommends) != 1:
        return None
    (arm_id,) = doc.recommends
    arm = next((arm for arm in report_input.arms if arm.id == arm_id), None)
    return arm.variant_key if arm is not None else None


def _durable_insights(
    analysis: EvalAnalysis, doc: ReportDoc, bundle: CampaignBundle, report_input: ReportInput
) -> list[EvalInsight]:
    """One insight per claim the document marks durable, citing the runs behind its figure keys.

    The runs a claim rests on are read off the keys in its text and its blocks' captions: every run of each arm a key
    names, on each aspect it names (every aspect when it names none), and the control's too where the figure is a
    difference from it (:func:`~threetears.evals.analysis.report.figure_keys.key_subjects`). A key naming no arm rests
    on every arm of the aspects it names, and a claim with no key cites no run. Two claims stating one thing
    (:func:`~threetears.evals.analysis.bundle.insights.insight_restatement_key`) mint it once, from the first. Each
    insight's ``invalidation_trigger`` names what retires it, which the engine carries out.
    """
    trigger = (
        f"Retracted if analysis {analysis.id} is archived; replaced in place when a later analysis of this "
        "subject states the same claim."
    )
    form = _INSIGHT_CONFIDENCE[doc.confidence.form]
    checked = {caveat.id for caveat in doc.confidence.caveats if caveat.checked is not None}
    insights: list[EvalInsight] = []
    stated: set[str] = set()
    for claim in doc.claims:
        if not claim.durable:
            continue
        statement = render(claim.text, report_input, {})
        key = insight_restatement_key(statement)
        if not statement.strip() or key in stated:
            continue
        stated.add(key)
        insights.append(
            EvalInsight(
                scope_id=analysis.scope_id,
                subject_id=bundle.subject_id,
                subject_kind=bundle.subject_kind,
                statement=statement,
                confidence=_claim_confidence(form, claim, checked),
                evidence_run_ids=sorted(_claim_runs(claim, bundle, report_input)),
                model_versions=analysis.model_versions,
                invalidation_trigger=trigger,
                source_campaign_id=bundle.campaign_id,
                source_analysis_id=analysis.id,
            )
        )
    return insights


def _bounce_summary(bounces: Mapping[str, int]) -> str | None:
    """The analyst's bounces as the analysis's ``repaired_refusal`` keeps them: each kind and its count, the kinds in
    name order; ``None`` for a report none of whose submissions was sent back. Each bounce's own faults are in the
    stored analyst session."""
    counted = [f"{kind} x{count}" for kind, count in sorted(bounces.items()) if count]
    return f"submissions sent back, by kind: {', '.join(counted)}" if counted else None


def _claim_confidence(form: Confidence, claim: Claim, checked: set[str]) -> Confidence:
    """A durable claim's tier: the document's form's, held to :data:`_OPEN_CAVEAT_CEILING` when the claim names a
    caveat of its own that is not checked immaterial."""
    if not any(caveat not in checked for caveat in claim.caveats):
        return form
    return min(form, _OPEN_CAVEAT_CEILING, key=_TIER_ORDER.index)


def _claim_runs(claim: Claim, bundle: CampaignBundle, report_input: ReportInput) -> set[str]:
    """The runs one claim rests on (:func:`_durable_insights` says which)."""
    variant_of = {arm.id: arm.variant_key for arm in report_input.arms if arm.variant_key is not None}
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    every_aspect = [aspect.aspect for aspect in bundle.aspects]
    runs_by = {
        (aspect.aspect, cell.variant_key): set(cell.run_ids)
        for aspect in bundle.aspects
        for cell in aspect.cell_measures
    }
    paths = [key.path for text in (claim.text, *(block.caption for block in claim.evidence)) for key in keys_in(text)]
    runs: set[str] = set()
    for path in paths:
        subjects: KeySubjects = key_subjects(path, report_input)
        arms: Sequence[str] = subjects.arms or list(variant_of)
        if subjects.against_control and control is not None:
            arms = [*arms, control]
        for aspect in subjects.aspects or every_aspect:
            for arm in arms:
                runs |= runs_by.get((aspect, variant_of.get(arm, "")), set())
    return runs


# =============================================================================
# What a stored analysis keeps of each aspect
# =============================================================================

# How many measure columns the run-index table may carry. The measure name space is open
# by construction — phase timings are `<tool>_<phase>_ms`, one per tool per phase — so
# without a ceiling the table widens with the system forever. Twelve is a scan-width
# judgement, not a computed limit; what makes it safe is that every measure it drops is
# named in `measures_omitted`, and the full distribution for all of them stays on the
# bundle, which is what the report writer reads.
_MAX_RUN_INDEX_MEASURES = 12


def aspect_analysis(bundle: AnalysisContextBundle, surface: DecisionSurface) -> AspectAnalysis:
    """What a stored analysis keeps of one aspect: its frozen bundle, and what code read off it.

    :param bundle: The aspect's bundle.
    :ptype bundle: AnalysisContextBundle
    :param surface: The surface every reading on it was resolved against — the same object, so the stored numbers and
        the stored facts cannot describe two different bundles.
    :ptype surface: DecisionSurface
    :return: The aspect's part of the analysis.
    :rtype: AspectAnalysis
    """
    reported = _comparable_measures(bundle.run_summaries)
    all_names = {m.name for summary in bundle.run_summaries for m in summary.measures.measures}
    return AspectAnalysis(
        label=bundle.aspect,
        bundle=bundle.to_dict(),
        bundle_fingerprint=bundle.fingerprint(),
        decision_surface=surface,
        # Frozen from the bundle, never resolved at read: an analysis must keep saying what its
        # arms WERE, whatever anyone archives or curates afterwards.
        variant_index=bundle.variant_index,
        run_index=[
            _run_index_entry(summary, reported=reported, omitted=sorted(all_names - reported))
            for summary in bundle.run_summaries
        ],
        coverage=CoverageLens(levers=[_lever_from_bundle(lever) for lever in bundle.coverage]),
    )


def _lever_from_bundle(lever: LeverCoverageInput) -> LeverCoverage:
    """Promote a bundle coverage lever to an analysis lever — facts from the bundle.

    Every field comes verbatim from the bundle (never the writer), so a point estimate always carries its
    sample size and spread. Nothing is added: a lever's coverage is how finely it was swept, and a
    tier looked up from ``status`` would restate ``status`` while reading as a confidence.
    """
    return LeverCoverage(
        name=lever.name,
        cells=lever.cells,
        k=lever.k,
        n=lever.n,
        dispersion=lever.dispersion,
        status=lever.status,
    )


def _comparable_measures(summaries: list[RunSummary]) -> set[str]:
    """Measure names worth a run-index column — those whose value actually differs by run.

    The run index is a COMPARISON table: one row per run, one column per key, and the
    reader scans down a column to see what the config change did. A measure identical in
    every run costs a column and answers nothing there, while the key space itself is
    open — phase timings are ``<tool>_<phase>_ms``, one per tool per phase, growing with
    the system — so without a rule the table widens indefinitely.

    Selecting on variation matches the surface's job, but on its own it does NOT bound the
    table: a phase timing is a continuous float that differs between runs essentially
    always, so variation alone removes none of the growth. A hard cap does the bounding,
    and the two compose — variation decides *which* measures are worth a column, the cap
    decides *how many* a reader can actually scan. Both are disclosed by the caller.

    Ordering under the cap is by **relative spread** (range over typical value), so the
    measures kept are the ones that moved most across the campaign — the ones a reader
    opened the table to find. Ties break on name, so the choice is deterministic and the
    bundle fingerprint stays stable.

    It degenerates correctly: a single-run campaign has nothing to compare, so nothing
    varies and every measure becomes a candidate — ranked and capped by the same rule, here
    rather than in the caller, so the ceiling cannot be routed around by the fallback.

    :param summaries: Every run summary in the campaign.
    :ptype summaries: list[RunSummary]
    :return: The names worth a column, at most ``_MAX_RUN_INDEX_MEASURES`` of them.
    :rtype: set[str]
    """
    seen: dict[str, set[str]] = {}
    numeric: dict[str, list[float]] = {}
    for summary in summaries:
        for measure in summary.measures.measures:
            # Compare on the rendered value so a float and its repr agree, and so a
            # category-count dict compares by content rather than identity.
            rendered = repr(measure.categories) if measure.categories else repr(measure.bad_tail())
            seen.setdefault(measure.name, set()).add(rendered)
            tail = measure.bad_tail()
            if tail is not None:
                numeric.setdefault(measure.name, []).append(tail)

    varying = {name for name, values in seen.items() if len(values) > 1}
    # Nothing varying means nothing to compare — a single-run campaign, or a campaign whose
    # measures all render identically. Reporting every measure is the right answer there, but
    # it must still go through the cap: the fallback set is the WIDEST one, and phase timings
    # are an open `<tool>_<phase>_ms` name space, so exempting it would let the persisted
    # key_metrics grow without bound on exactly the shape that selects nothing.
    candidates = varying or set(seen)
    if len(candidates) <= _MAX_RUN_INDEX_MEASURES:
        return candidates

    def spread(name: str) -> float:
        values = numeric.get(name) or []
        if not values:
            return 0.0
        typical = max(abs(sum(values) / len(values)), 1e-9)
        return (max(values) - min(values)) / typical

    ranked = sorted(candidates, key=lambda name: (-spread(name), name))
    return set(ranked[:_MAX_RUN_INDEX_MEASURES])


def _run_index_entry(summary: RunSummary, *, reported: set[str], omitted: list[str]) -> RunIndexEntry:
    """Build one run-index row from a bundle run summary (factual, not the writer's).

    A numeric measure contributes the tail at its *worse* end — p95 where lower is better,
    p05 where higher is — under a key naming which (the maximum, as ``<name>_max``, where lower is
    better and too few observations give a p95), so a reader is never invited to compare
    a measure's best outcome against another's worst as though both moved the same way. A
    categorical measure contributes its counts.

    Measures identical across every run are left out and **named in ``measures_omitted``**
    rather than silently dropped: a truncated table that does not say what it truncated
    reads as though it covered everything. The full distribution for every measure, omitted
    or not, stays on the bundle, which is what the report writer reads.

    One degenerate case inverts the first sentence: when *nothing* varies there is no
    comparison to make, so identical measures are precisely what gets reported — trimmed to
    the ceiling by relative spread, with whatever the ceiling dropped named in
    ``measures_omitted`` exactly as on the ordinary path.
    """
    # Both cost sides, never one. ``program_cost_usd_floor`` is the blended program total —
    # it includes the judge and simulator scaffolding that exists only to measure the
    # candidate, and it is a floor besides — while the production-replicating figures below
    # describe what the configuration itself spent. Publishing only the first invites a
    # comparison no deployment would pay; publishing only the second hides what the campaign
    # cost to run.
    key_metrics: dict[str, Any] = {
        "n_results": summary.n_results,
        "n_errors": summary.n_errors,
        "program_cost_usd_floor": summary.cost_usd,
    }
    if summary.n_cost_unpriced:
        # The floor leaves these out, so it travels with how many: a run whose candidate ran on an
        # unpriced model is not cheap, its cost is partly unknown.
        key_metrics["n_cost_unpriced"] = summary.n_cost_unpriced
    if summary.prod_cost_usd is not None:
        # The MEAN leads, because it is the figure two runs can be compared on — a total
        # scales with how many results a run produced, so ranking on it ranks run size.
        key_metrics["mean_prod_cost_usd"] = summary.mean_prod_cost_usd
        key_metrics["total_prod_cost_usd"] = summary.prod_cost_usd
        # The denominator travels with the figure: a reader comparing two runs has to be
        # able to see that one of them measured far fewer results than it produced.
        key_metrics["n_prod_cost_usd"] = summary.n_prod_cost_usd
        key_metrics["n_prod_cost_unmeasured"] = summary.n_results - summary.n_prod_cost_usd
        # What the run set away from the subject's production configuration travels with the figure
        # (#571): it is production's cost only for a run that moved nothing, and a caveat printed on
        # every run alike is one a reader stops reading.
        footing = summary.production_footing
        key_metrics["prod_cost_footing"] = (
            "nobody checked which inputs this run moved off the subject's production configuration"
            if footing is None
            else footing.sentence()
        )
        if footing is not None and footing.moved:
            key_metrics["prod_cost_moved_off_production"] = dict(sorted(footing.moved.items()))
        if footing is not None and footing.unchecked:
            key_metrics["prod_cost_unchecked_against_production"] = dict(sorted(footing.unchecked.items()))
    for measure in summary.measures.measures:
        if measure.name not in reported:
            continue
        if measure.categories:
            key_metrics[measure.name] = measure.categories
            continue
        tail = measure.bad_tail()
        if tail is not None:
            key_metrics[f"{measure.name}_p05" if measure.higher_is_better else f"{measure.name}_p95"] = tail
        elif measure.higher_is_better is False and measure.max is not None:
            # Too few observations for a 95th percentile: the slowest seen, named as what it is.
            key_metrics[f"{measure.name}_max"] = measure.max
    if omitted:
        key_metrics["measures_omitted"] = ", ".join(omitted)
    return RunIndexEntry(
        run_id=summary.run_id,
        config=summary.config,
        config_provenance=summary.config_provenance,
        key_metrics=key_metrics,
    )
