"""A campaign's analysis written as a report document: the stored analysis that links it, and the insights it mints.

:func:`doc_analysis` is what ``analysis_generate``'s job builds once the report writer has stored a document
(:func:`~threetears.evals.ops.analysis_generate`). Code builds what is stored beside the document, as the planner's
analysis is built (:mod:`~threetears.evals.analysis.planned_analysis`) in every part the writer does not write:

- **the analysis** (:class:`~threetears.evals.kernel.campaign.EvalAnalysis`): the campaign's frozen bundle (each
  aspect's, and the frame across them, read back from the snapshot the document was written from), its declaration,
  the models in play and the provenance. It holds no memo and no plan, and names its document (``report_doc_id``), so
  its ``basis`` is ``doc``. Its ``recommended_variant_key`` is the variant key of the arm the document recommends
  when ``recommends`` names exactly one; ``None`` when it names none, or two or more (a tie the document leaves to the
  reader);
- **the insights**: one per claim the document marks ``durable``, its statement the claim with every figure key written
  out, citing the runs behind the arms and aspects its keys name (the claim's text and its blocks' captions).

**Provenance** is the analyst skill's digest, never a prompt's registry key: the writer reads the skill the package
ships, so the digest is what says which words wrote the report.

**Closed dependency set.** It reads the bundle, the snapshot's report input and the stored document, and stores
nothing: storing is the service's, which holds the attempt.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Final, NamedTuple

from threetears.evals.analysis.bundle.insights import insight_restatement_key
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.generator import aspect_analysis
from threetears.evals.analysis.report.figure_keys import KeySubjects, key_subjects, keys_in, render
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.authored import Confidence
from threetears.evals.kernel.campaign import EvalAnalysis, EvalInsight, GenerationProvenance
from threetears.evals.kernel.evidence_tiers import JUDGED_TIER_RULE
from threetears.evals.kernel.report_doc import Claim, ConfidenceForm, EvalReportDoc, ReportDoc

#: ``DocAnalysis`` is the return type alone; nothing outside binds it by name.
__all__ = ["doc_analysis"]

#: The insight tier each of the document's confidence forms mints its durable claims at: a call code finds sound
#: is ``high``, one sound only if its open caveats prove immaterial ``medium``, and a low one ``low``.
_INSIGHT_CONFIDENCE: Final[dict[ConfidenceForm, Confidence]] = {
    "high": "high",
    "high_if_immaterial": "medium",
    "low": "low",
}


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
    bounces: Sequence[str],
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
    :param bounces: What each submission sent back was sent back for, in order (``invariant``, ``shape``, ...).
    :ptype bounces: Sequence[str]
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
        repair_attempts=len(bounces),
        repaired_refusal=f"submissions sent back for: {', '.join(bounces)}" if bounces else None,
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
    confidence = _INSIGHT_CONFIDENCE[doc.confidence.form]
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
                confidence=confidence,
                evidence_run_ids=sorted(_claim_runs(claim, bundle, report_input)),
                model_versions=analysis.model_versions,
                invalidation_trigger=trigger,
                source_campaign_id=bundle.campaign_id,
                source_analysis_id=analysis.id,
            )
        )
    return insights


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
