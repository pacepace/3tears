"""A campaign's analysis written by the report planner: the stored analysis, the plan beside it, and its insights.

:func:`generate_planned_analysis` is what a generation runs (:func:`~threetears.evals.analysis.service.run_analysis_generation`):
the planner writes the campaign's report plan from its report input
(:func:`~threetears.evals.analysis.planner.generate_report_plan`), and code builds what is stored from it:

- **the analysis** (:class:`~threetears.evals.kernel.campaign.EvalAnalysis`), as a memo's is built in every part a
  memo does not write: the campaign's frozen bundle (each aspect's, and the frame across them), its declaration, the
  models in play, the provenance. It has no memo (``document`` is None) and names its plan (``report_plan_id``). Its
  ``recommended_variant_key`` is the variant key of the plan's ``recommended`` option, or None when the plan
  recommends nothing (a trade-off, a no-call, a lean);
- **the plan** (:class:`~threetears.evals.kernel.campaign.EvalReportPlan`), with how it was generated and the digest
  of the report input it was written against;
- **the insights** (:func:`planned_insights`): one per durable claim the plan states.

**Closed dependency set.** It reads the bundle, the report input built from it, the resolved prompt, the host's profile
and an injected client, and stores nothing: storing is the service's, which holds the attempt.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from threetears.evals.analysis.bundle.insights import insight_restatement_key
from threetears.evals.analysis.errors import SoundnessRefusal
from threetears.evals.analysis.generator import CallAdmission, GenerationTally, aspect_analysis
from threetears.evals.analysis.planner import PlanGeneration, generate_report_plan
from threetears.evals.analysis.report.figure_keys import key_subjects, render
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.campaign import (
    EvalAnalysis,
    EvalInsight,
    EvalReportPlan,
    GenerationProvenance,
    ReportPlanProvenance,
    report_plan_id,
)
from threetears.evals.kernel.evidence_tiers import JUDGED_TIER_RULE
from threetears.evals.kernel.report_plan import ReportPlan
from threetears.evals.schema.hashing import bytes_digest
from threetears.evals.schema.models import utc_now_iso

if TYPE_CHECKING:
    from threetears.evals.analysis.campaign_bundle import CampaignBundle
    from threetears.evals.kernel.host.profile import HostProfile
    from threetears.evals.schema.completion import CompletionGenerator

__all__ = [
    "PlannedAnalysis",
    "generate_planned_analysis",
    "planned_insights",
    "plan_recommended_arm",
    "planned_analysis",
    "report_input_digest",
    "report_plan_of",
]


class PlannedAnalysis(NamedTuple):
    """What one planned generation stores: the analysis, its plan, and the insights it mints."""

    analysis: EvalAnalysis
    plan: EvalReportPlan
    insights: list[EvalInsight]


def report_input_digest(report_input: ReportInput) -> str:
    """The digest of a report input: sha256 of its canonical JSON (keys sorted, by alias).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: A 64-character lowercase hex digest.
    :rtype: str
    """
    canonical = json.dumps(
        report_input.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return bytes_digest(canonical.encode("utf-8"))


def report_plan_of(stored: EvalReportPlan) -> ReportPlan:
    """The plan a stored report plan holds, as the plan model.

    :param stored: The stored plan.
    :ptype stored: EvalReportPlan
    :return: Its plan.
    :rtype: ReportPlan
    """
    return ReportPlan.model_validate(stored.plan)


def plan_recommended_arm(plan: ReportPlan) -> str | None:
    """The arm id of the plan's ``recommended`` option, or None when the plan recommends nothing.

    :param plan: The plan.
    :ptype plan: ReportPlan
    :return: The arm id, or None.
    :rtype: str | None
    """
    return next((row.option for row in plan.options.rows if row.role == "recommended"), None)


def _recommended_variant_key(plan: ReportPlan, report_input: ReportInput) -> str | None:
    """The variant key of the arm the plan recommends; None when it recommends nothing.

    :raises ValueError: The recommended option names no arm of the input, or an arm with no variant key.
    """
    arm_id = plan_recommended_arm(plan)
    if arm_id is None:
        return None
    arm = next((arm for arm in report_input.arms if arm.id == arm_id), None)
    if arm is None or arm.variant_key is None:
        raise ValueError(f"the plan recommends {arm_id!r}, which names no arm of the report input with a variant key")
    return arm.variant_key


def planned_insights(
    analysis: EvalAnalysis, plan: ReportPlan, bundle: CampaignBundle, report_input: ReportInput
) -> list[EvalInsight]:
    """Mint one durable insight per claim the plan states, citing the runs behind its figure keys.

    The planner names what is durable (``durable_claims``); the runs a claim rests on are read off its keys: every run
    of each arm a key names, on each aspect it names (every aspect when it names none), and the control's too where the
    figure is a difference from it (:func:`~threetears.evals.analysis.report.figure_keys.key_subjects`). A key naming
    no arm rests on every arm of the aspects it names. So an insight cannot cite a run its figures were not read from.
    The statement is the claim with each figure key written out, as a reader of a later campaign's prior insights
    reads it. Two claims stating one thing
    (:func:`~threetears.evals.analysis.bundle.insights.insight_restatement_key`) mint it once, from the first. Each
    insight's ``invalidation_trigger`` names what retires it, which the engine carries out: archiving this analysis,
    or a later analysis restating the claim.

    :param analysis: The stored analysis the insights trace back to.
    :ptype analysis: EvalAnalysis
    :param plan: Its plan.
    :ptype plan: ReportPlan
    :param bundle: The campaign bundle the plan was written over.
    :ptype bundle: CampaignBundle
    :param report_input: The report input built from it, which the claims' keys resolve against.
    :ptype report_input: ReportInput
    :return: The insights, in the plan's order.
    :rtype: list[EvalInsight]
    :raises UnresolvableReference: A claim's key does not resolve (a checked plan's always do).
    """
    variant_of = {arm.id: arm.variant_key for arm in report_input.arms if arm.variant_key is not None}
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    every_aspect = [aspect.aspect for aspect in bundle.aspects]
    runs_by = {
        (aspect.aspect, cell.variant_key): set(cell.run_ids)
        for aspect in bundle.aspects
        for cell in aspect.cell_measures
    }
    trigger = (
        f"Retracted if analysis {analysis.id} is archived; replaced in place when a later analysis of this "
        "subject states the same claim."
    )
    insights: list[EvalInsight] = []
    stated: set[str] = set()
    for claim in plan.durable_claims:
        statement = render(claim.text, report_input, {})
        key = insight_restatement_key(statement)
        if not statement.strip() or key in stated:
            continue
        stated.add(key)
        runs: set[str] = set()
        for path in claim.keys:
            subjects = key_subjects(path, report_input)
            arms: Sequence[str] = subjects.arms or list(variant_of)
            if subjects.against_control and control is not None:
                arms = [*arms, control]
            for aspect in subjects.aspects or every_aspect:
                for arm in arms:
                    runs |= runs_by.get((aspect, variant_of.get(arm, "")), set())
        insights.append(
            EvalInsight(
                scope_id=analysis.scope_id,
                subject_id=bundle.subject_id,
                subject_kind=bundle.subject_kind,
                statement=statement,
                confidence=claim.confidence,
                evidence_run_ids=sorted(runs),
                model_versions=analysis.model_versions,
                invalidation_trigger=trigger,
                source_campaign_id=bundle.campaign_id,
                source_analysis_id=analysis.id,
            )
        )
    return insights


def planned_analysis(
    bundle: CampaignBundle,
    report_input: ReportInput,
    generation: PlanGeneration,
    *,
    prompt_id: str,
    calls: int,
    bundle_assembled_at: str,
) -> PlannedAnalysis:
    """Build what a planned generation stores from the plan it accepted (this module's account).

    :param bundle: The campaign bundle the plan was written over, frozen with the analysis.
    :ptype bundle: CampaignBundle
    :param report_input: The report input built from it.
    :ptype report_input: ReportInput
    :param generation: The accepted plan and how it was generated.
    :ptype generation: PlanGeneration
    :param prompt_id: The registry key the planner prompt was resolved under.
    :ptype prompt_id: str
    :param calls: How many provider calls the plan took.
    :ptype calls: int
    :param bundle_assembled_at: When the bundle was assembled (ISO-8601), the cutoff a re-assembly reads the insight
        ledger as of.
    :ptype bundle_assembled_at: str
    :return: The analysis, its plan and its insights.
    :rtype: PlannedAnalysis
    :raises ValueError: The plan recommends an arm the input holds no variant key for.
    """
    surfaces = bundle.surfaces()
    model_versions: dict[str, str] = {}
    for aspect in bundle.aspects:
        model_versions.update(aspect.model_versions)
    refusals = [refusal for refusal in (generation.structure_refusal, generation.prose_refusal) if refusal]
    first = bundle.aspects[0]
    provenance = GenerationProvenance(
        prompt_id=prompt_id,
        prompt_version=generation.prompt_version,
        generator_model=generation.generator_model,
        bundle_fingerprint=bundle.fingerprint(),
        generated_at=generation.generated_at,
        bundle_assembled_at=bundle_assembled_at,
        token_cost=generation.token_cost,
        repair_attempts=generation.repair_attempts,
        repaired_refusal="\n\n".join(refusals) if refusals else None,
        user_message_digest=generation.user_message_digest,
        cell_model_version=first.cell_model_version,
        bundle_schema_version=first.schema_version,
        host_declarations_digest=first.host_declarations_digest,
    )
    analysis_id = str(uuid.uuid7())
    analysis = EvalAnalysis(
        id=analysis_id,
        recommended_variant_key=_recommended_variant_key(generation.plan, report_input),
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
        report_plan_id=report_plan_id(analysis_id),
        judged_tier_rule=JUDGED_TIER_RULE,
        aspects=[aspect_analysis(aspect, surfaces[aspect.aspect]) for aspect in bundle.aspects],
        campaign_bundle=bundle.frame(),
    )
    return PlannedAnalysis(
        analysis,
        _plan_doc(analysis, report_input, generation, prompt_id, calls),
        planned_insights(analysis, generation.plan, bundle, report_input),
    )


async def generate_planned_analysis(
    bundle: CampaignBundle,
    report_input: ReportInput,
    *,
    prompt: str,
    model: str,
    client: CompletionGenerator,
    prompt_id: str,
    bundle_assembled_at: str,
    profile: HostProfile,
    tally: GenerationTally | None = None,
    admit: CallAdmission | None = None,
) -> PlannedAnalysis:
    """Plan one campaign's report and build what is stored from it: the analysis, its plan and its insights.

    The planner's account holds (:mod:`~threetears.evals.analysis.planner`): two calls, each repaired at most once, a
    model the host does not allow refused before anything is sent, and every call put to ``admit`` first.

    :param bundle: The campaign bundle, frozen with the analysis.
    :ptype bundle: CampaignBundle
    :param report_input: The report input built from it (:func:`~threetears.evals.analysis.report_input_build.report_input`).
    :ptype report_input: ReportInput
    :param prompt: The resolved ``eval_report_plan`` prompt.
    :ptype prompt: str
    :param model: The planner model, as the host's client resolved it.
    :ptype model: str
    :param client: The injected completion client, bound to ``model``.
    :ptype client: CompletionGenerator
    :param prompt_id: The registry key ``prompt`` was resolved under, recorded as provenance.
    :ptype prompt_id: str
    :param bundle_assembled_at: When the bundle was assembled (ISO-8601).
    :ptype bundle_assembled_at: str
    :param profile: The host: its allowed writer models and its register.
    :ptype profile: HostProfile
    :param tally: A caller-owned record of what was sent, spent and refused, exact however the generation ends.
    :ptype tally: GenerationTally | None
    :param admit: Asked with each call's exact request before it is sent; it raises to refuse the call.
    :ptype admit: CallAdmission | None
    :return: What the generation stores.
    :rtype: PlannedAnalysis
    :raises GenerationError: The model is not one the host allows, or the provider cut a call short.
    :raises SoundnessRefusal: A call's reply was refused twice; or the accepted plan recommends an arm the input holds
        no variant key for (never repaired: the plan checks refuse an unknown arm first).
    """
    tally = tally if tally is not None else GenerationTally()
    generation = await generate_report_plan(
        report_input, prompt=prompt, model=model, client=client, profile=profile, tally=tally, admit=admit
    )
    try:
        return planned_analysis(
            bundle,
            report_input,
            generation,
            prompt_id=prompt_id,
            calls=tally.calls,
            bundle_assembled_at=bundle_assembled_at,
        )
    except ValueError as refused:
        raise SoundnessRefusal(f"the accepted plan could not be stored as an analysis: {refused}") from refused


def _plan_doc(
    analysis: EvalAnalysis, report_input: ReportInput, generation: PlanGeneration, prompt_id: str, calls: int
) -> EvalReportPlan:
    return EvalReportPlan(
        id=report_plan_id(analysis.id),
        scope_id=analysis.scope_id,
        analysis_id=analysis.id,
        campaign_id=analysis.campaign_id,
        report_input_digest=report_input_digest(report_input),
        plan=generation.plan.model_dump(mode="json", by_alias=True),
        generation=ReportPlanProvenance(
            prompt_id=prompt_id,
            prompt_version=generation.prompt_version,
            generator_model=generation.generator_model,
            calls=calls,
            repair_attempts=generation.repair_attempts,
            structure_refusal=generation.structure_refusal,
            prose_refusal=generation.prose_refusal,
            token_cost=generation.token_cost,
            user_message_digest=generation.user_message_digest,
            generated_at=generation.generated_at,
        ),
        created_at=utc_now_iso(),
    )
