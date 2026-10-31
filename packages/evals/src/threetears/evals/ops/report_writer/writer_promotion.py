"""The promotion read: whether a report writer campaign's candidate arm is shown better than its control.

A change to how the report writer works (its skill, its reasoning effort, its revisions, or its model) ships only on a
measured decision, and this is the read that decides it. It adds no statistics: it reads the campaign's corrected
verdicts as every other gate does (:func:`~threetears.evals.analysis.campaign_verdicts`, the code-only page's typed
verdicts, each family corrected), keeps the candidate's, and gates them
(:func:`~threetears.evals.analysis.gate.gate_verdicts`, failing on a regression, a breached guardrail and a guardrail
not shown held):

- ``keep`` when the gate failed: the candidate is shown worse on some reading, or a guardrail is not shown held;
- ``undecided`` when it failed on nothing and left some verdict undecided, or there was none. **Undecided never
  promotes**: a reading the evidence could not separate is not shown safe;
- ``promote`` when every verdict was decided, none failing, and at least one shows the candidate better;
- ``keep`` when every verdict was decided and none shows the candidate better: nothing is gained by the change.

**Generic over levers.** The candidate is an arm, and what it changed is read off the arms' recorded lever maps, so a
model, an effort or a host's own lever is decided as a skill text is.

**What it refuses.** A campaign whose arms were not scored by one judge on one rubric
(:func:`~threetears.evals.ops.report_writer.writer_contract.refuse_unfixed_scoring_judge`), and one that ran over
anything but held-out writer cases: each run must name its case set, and each case it ran must be a writer case its
mint put on the held-out side of its split, in the set the run names. The dev side is where a change is tuned, so a
decision read there would be read on the cases it was fitted to.

**The read never writes.** Promotion is a person's act: they record it as the campaign's call (``campaign_record``,
naming the candidate) and commit the change with its entry in the skill ledger
(:mod:`~threetears.evals.analysis.analyst.skill_ledger`). Each read states the basis the build's own skill shipped on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.analyst.skill_ledger import ShippedSkill, skill_record
from threetears.evals.analysis.gate import GateResult, gate_verdicts
from threetears.evals.analysis.report.blocks import Verdict
from threetears.evals.analysis.service import assemble_campaign_bundle, campaign_verdicts
from threetears.evals.kernel.declaration import CampaignArm, resolve_control_arm
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.kernel.remedies import FIND_CAMPAIGN
from threetears.evals.ops.report_writer.writer_bank import writer_case_bank
from threetears.evals.ops.report_writer.writer_contract import refuse_unfixed_scoring_judge
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import CaseSetRef, EvalRun

#: What a promotion read answers.
PromotionDecision = Literal["promote", "keep", "undecided"]


class WriterPromotion(EvalBaseModel):
    """What the promotion read answered for one candidate arm against the control, and what it read."""

    campaign_id: str = Field(description="The campaign read.")
    decision: PromotionDecision = Field(
        description="promote, keep or undecided; undecided never promotes.",
    )
    why: str = Field(description="The decision in one sentence.")
    candidate: str = Field(description="The candidate arm, by its label.")
    candidate_key: str = Field(description="The candidate arm's variant key, which a call records.")
    control: str = Field(description="The control arm, by its label.")
    levers: dict[str, str] = Field(description="The candidate's lever map: each lever's level, as a reader names it.")
    changed: list[str] = Field(description="The levers whose level differs from the control's, by name.")
    verdicts: list[Verdict] = Field(description="The candidate's corrected verdicts, each reading's, in page order.")
    cases: int = Field(description="The held-out cases the campaign's runs ran.")
    case_sets: list[CaseSetRef] = Field(description="The held-out case sets the runs named.")
    cost_usd: float | None = Field(description="What the candidate's trials cost; None when none was priced.")
    control_cost_usd: float | None = Field(description="What the control's trials cost; None when none was priced.")
    unpriced_trials: int = Field(description="Trials of either arm whose spend went unpriced, so not in the costs.")
    writer_model: str = Field(description="The model the candidate arm wrote with.")
    shipped: ShippedSkill = Field(description="The skill ledger's entry for the skill this build ships.")


def _arm(arms: Sequence[CampaignArm], named: str, campaign_id: str) -> CampaignArm:
    """The arm a caller named, by variant key or label."""
    arm = next((each for each in arms if each.variant_key == named), None) or next(
        (each for each in arms if each.label == named), None
    )
    if arm is None:
        raise ValidationFailedError(
            f"campaign {campaign_id} has no arm {named!r}",
            remedy=f"name one of its arms by label or variant key: {[each.label for each in arms]}",
        )
    return arm


def _refuse_unheld_out(host: EvalHost, runs: Sequence[EvalRun], scope_id: str) -> tuple[list[CaseSetRef], int]:
    """The held-out sets the runs named and how many cases they ran, refusing a run or case off the held-out side."""
    faults = [f"run {run.id} named no case set" for run in runs if run.case_set is None]
    named = {run.case_set.name for run in runs if run.case_set is not None}
    ids = sorted({case_id for run in runs for case_id in run.test_case_ids})
    stored = host.storage.load_test_cases_by_ids(ids, scope_id)
    bank = writer_case_bank(stored)
    origins = {stored_case.id: case.origin for stored_case, case in bank.cases}
    unreadable = dict(bank.unreadable)
    for case_id in ids:
        origin = origins.get(case_id)
        if case_id in unreadable:
            faults.append(f"case {case_id} does not read as a writer case ({unreadable[case_id]})")
        elif case_id not in origins:
            faults.append(f"case {case_id} is no stored writer case")
        elif origin is None:
            faults.append(f"case {case_id} was put in no split")
        elif origin.side != "held_out":
            faults.append(f"case {case_id} is on the {origin.side} side of its split")
        elif origin.case_set not in named:
            faults.append(f"case {case_id} was minted into {origin.case_set!r}, which no run names")
    if faults:
        raise ValidationFailedError(
            "a promotion is decided on held-out writer cases alone, and " + "; ".join(faults),
            remedy="launch the arms over the held-out writer set (case_set), and decide only on that campaign",
        )
    refs = sorted({run.case_set for run in runs if run.case_set is not None}, key=lambda ref: (ref.name, ref.version))
    return refs, len(ids)


def _decided(gate: GateResult, candidate: str) -> tuple[PromotionDecision, str]:
    """The decision a gate over the candidate's verdicts comes to, and its sentence."""
    total = len(gate.gated)
    if gate.failed:
        return "keep", (
            f"Keep the control: {candidate} is shown worse, or a guardrail is not shown held, on "
            f"{len(gate.failures)} of {total} readings."
        )
    if gate.outcome == "undecided":
        if not total:
            return "undecided", f"Undecided: the campaign holds no verdict on {candidate}, so nothing was shown."
        return "undecided", (
            f"Undecided: {len(gate.undecided)} of {total} readings did not decide {candidate} against the control, "
            "and an undecided reading never promotes."
        )
    better = sum(verdict.outcome == "improved" for verdict in gate.gated)
    if better:
        return (
            "promote",
            f"Promote {candidate}: it is shown better on {better} of {total} readings and no worse on any.",
        )
    return "keep", f"Keep the control: {candidate} is shown better on none of {total} readings."


def writer_promotion(host: EvalHost, campaign_id: str, scope_id: str, *, candidate: str) -> WriterPromotion:
    """Read whether a report writer campaign's candidate arm should replace its control (the module docstring).

    :param host: The host whose store holds the campaign and its runs and cases.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param candidate: The candidate arm, by its label or its variant key.
    :ptype candidate: str
    :return: The decision and what it read.
    :rtype: WriterPromotion
    :raises NotFoundError: No campaign with that id in the scope.
    :raises ValidationFailedError: The campaign has no control, the candidate is none of its arms or is the control,
        an arm's lever map was not recorded, its arms were not scored by one judge on one rubric, or it ran over
        anything but held-out writer cases; or the build's skill has no ledger entry.
    """
    campaign = host.storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id, remedy=FIND_CAMPAIGN)
    runs = [run for run_id in campaign.run_ids if (run := host.storage.load_eval_run(run_id, scope_id)) is not None]
    refuse_unfixed_scoring_judge(runs)
    case_sets, cases = _refuse_unheld_out(host, runs, scope_id)
    bundle = assemble_campaign_bundle(host, campaign_id, scope_id)
    control = next((arm for arm in bundle.arms if arm.role == "control"), None)
    if control is None:
        raise ValidationFailedError(
            f"campaign {campaign_id} names no control, so no arm is read against one",
            remedy="name the control arm (campaign_update), then read the candidate against it",
        )
    arm = _arm(bundle.arms, candidate, campaign_id)
    if arm.variant_key == control.variant_key:
        raise ValidationFailedError(
            f"{arm.label} is the campaign's control, and a promotion reads a candidate against it",
            remedy="name a candidate arm",
        )
    if arm.levers is None or control.levers is None or CANDIDATE_MODEL_LEVER not in arm.levers:
        raise ValidationFailedError(
            f"campaign {campaign_id}'s arm {arm.label if arm.levers is None else control.label} records no lever map "
            "with its model, so what the candidate changed cannot be read",
            remedy="address each arm from a run (campaign_update), which records its lever map",
        )
    keys = {arm.variant_key, *resolve_control_arm(arm).variant_keys}
    control_keys = {control.variant_key, *resolve_control_arm(control).variant_keys}
    verdicts = [verdict for verdict in campaign_verdicts(host, campaign_id, scope_id) if verdict.variant_key in keys]
    decision, why = _decided(gate_verdicts(verdicts), arm.label)
    boards = [board for aspect in bundle.aspects for board in aspect.scoreboard]
    costs = {
        side: [
            board.test_cost_usd for board in boards if board.variant_key in members and board.test_cost_usd is not None
        ]
        for side, members in (("candidate", keys), ("control", control_keys))
    }
    return WriterPromotion(
        campaign_id=campaign_id,
        decision=decision,
        why=why,
        candidate=arm.label,
        candidate_key=arm.variant_key,
        control=control.label,
        levers={name: value.display for name, value in sorted(arm.levers.items())},
        changed=sorted(
            name
            for name in arm.levers.keys() | control.levers.keys()
            if (mine := arm.levers.get(name)) is None
            or (theirs := control.levers.get(name)) is None
            or mine.content_hash != theirs.content_hash
        ),
        verdicts=verdicts,
        cases=cases,
        case_sets=case_sets,
        cost_usd=sum(costs["candidate"]) if costs["candidate"] else None,
        control_cost_usd=sum(costs["control"]) if costs["control"] else None,
        unpriced_trials=sum(board.n_cost_unpriced for board in boards if board.variant_key in keys | control_keys),
        writer_model=arm.levers[CANDIDATE_MODEL_LEVER].display,
        shipped=skill_record(load_skill()),
    )


#: The engine's ``writer_promotion`` action reads a decision for a host when a person asks whether to promote; the
#: decision's literal waits for a caller (``tests/test_report_writer_exports.py``).
__all__ = ["WriterPromotion", "writer_promotion"]
