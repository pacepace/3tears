"""Every closed vocabulary of the package that could be a word about a reading, found by structure, and the register
of the ones that are not.

**What is found.** A vocabulary is a closed set of words: a module-level ``Literal`` alias, or a ``Literal`` written
on a model's field. Two rules find the ones that could say whether a reading counts:

- **by name**: every module-level alias, anywhere in the package, whose name ends as a state, a reason, a verdict,
  an outcome, a decision, a status, a cause, a tier, a standing, a reading or a rule;
- **by reach**: every closed set of two or more words on any field reachable from the stored models (the aspect's
  bundle, the campaign's bundle, the report input, a run and a result), whether it has a name or not.

Each one found is either mapped word for word in ``threetears.evals.analysis.state_words`` or listed here
(:data:`NOT_ABOUT_A_READING`) under the reason it is no word about a reading. ``test_reading_states.py`` holds that
there is no third kind, so a vocabulary added later fails until someone maps it or lists it.

A vocabulary is named ``<path below the package>:<Alias>`` for an alias and ``<Model>.<field>`` for a set written on
a field with no alias of its words.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel

import threetears.evals
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.schema.models import EvalResult, EvalRun

SOURCE = Path(threetears.evals.__file__).resolve().parent

#: The stored models a vocabulary is reachable from.
ROOTS: tuple[type[BaseModel], ...] = (AnalysisContextBundle, CampaignBundle, ReportInput, EvalRun, EvalResult)

#: How a vocabulary that could be about a reading is named.
STATE_LIKE = re.compile(r"(State|Reason|Verdict|Outcome|Decision|Status|Cause|Tier|Standing|Reading|Rule)$")


def aliases() -> dict[str, frozenset[str]]:
    """Every module-level ``Literal`` alias of strings in the package, as ``<path>:<Alias>``, with its words."""
    found: dict[str, frozenset[str]] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name, value = node.targets[0].id, node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                name, value = node.target.id, node.value
            else:
                continue
            if not (
                isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and value.value.id == "Literal"
            ):
                continue
            members = value.slice.elts if isinstance(value.slice, ast.Tuple) else [value.slice]
            words = frozenset(m.value for m in members if isinstance(m, ast.Constant) and isinstance(m.value, str))
            if words:
                found[f"{path.relative_to(SOURCE).as_posix()}:{name}"] = words
    return found


def reachable() -> dict[frozenset[str], str]:
    """Every closed set of two or more words on a field reachable from :data:`ROOTS`, with the first field found
    holding it, as ``<Model>.<field>``."""
    seen: set[type[BaseModel]] = set()
    found: dict[frozenset[str], str] = {}

    def walk(annotation: Any, where: str) -> None:
        if get_origin(annotation) is Literal:
            words = frozenset(member for member in get_args(annotation) if isinstance(member, str))
            if len(words) >= 2:
                found.setdefault(words, where)
            return
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation not in seen:
                seen.add(annotation)
                for name, field in annotation.model_fields.items():
                    walk(field.annotation, f"{annotation.__name__}.{name}")
            return
        for inner in get_args(annotation):
            walk(inner, where)

    for root in ROOTS:
        walk(root, root.__name__)
    assert len(seen) > 150, "the walk did not reach the stored models' parts"
    return found


def discovered() -> dict[str, frozenset[str]]:
    """Every vocabulary the two rules find, by its name, with its words."""
    named = aliases()
    by_words: dict[frozenset[str], list[str]] = {}
    for name, words in named.items():
        by_words.setdefault(words, []).append(name)
    found = {name: words for name, words in named.items() if STATE_LIKE.search(name.rsplit(":", 1)[1].lstrip("_"))}
    for words, where in reachable().items():
        homes = by_words.get(words)
        if homes is None:
            found[where] = words
        else:
            found.update(dict.fromkeys(homes, words))
    return found


_SHARED = "the shared sets themselves, and the closed words of the shapes that hold them"
_CHECK = "a check's result or the rule it was read by: the instrument-level set the reading states are fed by"
_CLAUSE = (
    "a declared line's result on a counted reading, or an arm's standing: a clause state, never a way of not counting"
)
_KIND = "names what a thing is, the part it plays, how it was declared or how it was tested"
_TRIAL = "how one trial, call or delivery ended: below a reading, as a lost trial is"
_OPERATION = "how a job, a launch, a stored write or a host's conformance went: an operation's outcome, not evidence"
_DESIGN = "what a campaign's design ran, varied or recorded: raised as a caveat where it qualifies an answer, never a reading's state"
_ANALYST = "an analyst tool's answer about a key or a price, or how sure an authored call is"
_PAGE = "how a page draws a mark"

#: Every discovered vocabulary that is no word about whether a reading counts, under the reason it is not.
NOT_ABOUT_A_READING: dict[str, tuple[str, ...]] = {
    _SHARED: (
        "schema/reading_states.py:AttestableCoordinate",
        "schema/reading_states.py:CaveatClass",
        "schema/reading_states.py:CaveatReach",
        "schema/reading_states.py:Coordinate",
        "schema/reading_states.py:IdentityState",
        "schema/reading_states.py:Maker",
        "schema/reading_states.py:NotAvailableReason",
        "schema/reading_states.py:ReadingState",
        "schema/reading_states.py:SampleKind",
        "schema/reading_states.py:SetAsideReason",
        "schema/reading_states.py:WithheldReason",
        "Corrected.against",
        "JudgeCheck.against",
        "FamilyComparison.delta_note",
        "analysis/completeness.py:GateRule",
        "analysis/report_input.py:NoActionReason",
        "analysis/report_input.py:Remedy",
        "analysis/report_input.py:UnsizedReason",
        "RemedyStep.after",
        "RemedyStep.limits",
    ),
    _CHECK: (
        "kernel/evidence_tiers.py:CriterionState",
        "kernel/evidence_tiers.py:JudgedTierRule",
        "schema/models.py:GoalCheckProof",
    ),
    _CLAUSE: (
        "analysis/campaign_readings.py:BarState",
        "analysis/campaign_readings.py:ShownReading",
        "analysis/floors.py:FloorOutcome",
        "analysis/floors.py:FloorState",
        "analysis/floors.py:_FloorCase",
        "analysis/gate.py:GateOutcome",
        "analysis/report/blocks.py:GuardrailOutcome",
        "analysis/criteria.py:CriterionStatus",
        "analysis/criteria.py:GateStatus",
        "kernel/surface.py:GuardrailDecision",
        "ops/report_writer/writer_promotion.py:PromotionDecision",
    ),
    _KIND: (
        "analysis/bundle/schema.py:ComparisonReading",
        "analysis/bundle/schema.py:CriterionKind",
        "analysis/bundle/schema.py:FloorContrastReading",
        "analysis/kinds.py:CampaignKind",
        "analysis/answer.py:ReportKind",
        "kernel/authored.py:Reading",
        "kernel/campaign.py:ReadingKind",
        "kernel/cassettes.py:CassetteMode",
        "kernel/declaration.py:ArmRole",
        "kernel/host/bars.py:PassCriterion",
        "kernel/metrics.py:AttributionScope",
        "kernel/metrics.py:Materiality",
        "kernel/metrics.py:MeasurePopulation",
        "kernel/metrics.py:MeasureScale",
        "kernel/metrics.py:MetricDataType",
        "kernel/metrics.py:TransferabilityClass",
        "kernel/surface.py:TimeAxisBasis",
        "quick/guardrails.py:GuardrailDirection",
        "schema/judge_attribution.py:JudgeAttributionSource",
        "schema/models.py:ApparatusProvenance",
        "schema/models.py:ArmSchedule",
        "schema/models.py:BudgetStopCurrency",
        "schema/models.py:CompletenessSource",
        "schema/models.py:DeclaredMeritAxis",
        "schema/models.py:RubricAxis",
        "schema/models.py:RubricScale",
        "schema/models.py:WorldPlacement",
        "schema/world_events.py:WorldEventCause",
        "schema/world_events.py:WorldEventKind",
        "Arm.role",
        "Aspect.graded_by",
        "AspectDesign.crossing",
        "BarAdjudication.source",
        "Confound.kind",
        "ConfusionTable.unit",
        "ControlDeclaration.stimulus",
        "Criterion.kind",
        "Criterion.scope",
        "Criterion.statistic",
        "FamilyComparison.basis",
        "analysis/significance.py:SeparationTest",
        "analysis/significance.py:Unquoted",
        "FamilyComparison.margin_source",
        "InterJudgeDimension.kappa_weighting",
        "LeverCoverageInput.cohort_scope",
        "Measure.grade",
        "MeasureMovement.test",
        "RealizedDesign.shape",
        "SubjectKeyInstability.kind",
    ),
    _TRIAL: (
        "analysis/bundle/schema.py:FailCause",
        "kernel/cassettes.py:DeliveryOutcome",
        "kernel/judge_cases.py:JudgeTrialOutcome",
        "kernel/result_condition.py:CandidateFailureCause",
        "schema/models.py:AsyncDeliveryStatus",
        "schema/models.py:CassetteDeliveryOutcome",
        "schema/models.py:CellTermination",
        "schema/models.py:StopReason",
        "schema/turns.py:TurnStopReason",
    ),
    _OPERATION: (
        "kernel/campaign.py:AttemptOutcome",
        "kernel/campaign.py:SweepOutcome",
        "kernel/host/profile.py:CoverageState",
        "kernel/host/world_conformance.py:Outcome",
        "kernel/job_leases.py:RenewalVerdict",
        "kernel/prompts/contract.py:ContractState",
        "kernel/prompts/drift.py:DriftState",
        "ops/analysis.py:ReportState",
        "ops/jobs.py:JobState",
        "ops/results.py:TraceState",
        "run/job_work.py:JobOutcome",
        "run/launch.py:ArmOutcome",
        "run/run_document.py:RunUpdateOutcome",
        "schema/out_of_run_spend.py:OutOfRunOutcome",
    ),
    _DESIGN: (
        "analysis/bundle/measures.py:_Verdict",
        "MembershipChange.change",
        "analysis/reporting.py:SentSettingsState",
        "schema/judge_attribution.py:JudgeAttributionState",
        "Confound.status",
        "DeclaredCellCoverage.state",
        "DeclaredLevelCoverage.state",
        "FactorPairCell.status",
        "LeverCoverageInput.status",
        "RealizedDesign.control_excluded",
    ),
    _ANALYST: (
        "analysis/analyst/keys.py:KeyStatus",
        "analysis/analyst/price.py:OffReason",
        "kernel/authored.py:Confidence",
    ),
    _PAGE: ("analysis/report/charts.py:ChartStatus",),
}
