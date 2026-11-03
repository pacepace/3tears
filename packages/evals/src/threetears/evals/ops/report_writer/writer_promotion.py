"""The promotion read: whether a report writer campaign's candidate arm is shown better than its control.

A change to how the report writer works (its skill, its reasoning effort, its revisions, or its model) ships only on a
measured decision, and this is the read that decides it. It adds no statistics. It reads one pre-registered primary
reading, the pairwise pass's preference (``report_writer.preferred``, at alpha 0.05, a family of one; its report is
the completed ``writer_pairwise`` job's result), and the campaign's corrected verdicts
(:func:`~threetears.evals.analysis.campaign_verdicts`), which gate on regression only:

- ``promote`` when the primary reads improved, no gated reading reads regressed, and every guardrail is held;
- ``keep`` when the primary reads regressed, a gated reading reads regressed, or a guardrail is breached;
- ``undecided`` otherwise: the primary not separated, read on too few cases, untested or not read at all, or a
  guardrail not shown held. **Undecided never promotes.**

A reading on fewer cases than a rate needs (reason ``too_few_cases``) reads as that, whatever its outcome, as its page
prints it. A gated reading that is not separated decides nothing either way: the campaign's family of readings is
too wide to separate on a small held-out set, so only the primary is asked to.

**Generic over levers.** The candidate is an arm, and what it changed is read off the arms' recorded lever maps, so a
model, an effort or a host's own lever is decided as a skill text is.

**What it refuses.** A campaign whose arms were not scored by one judge on one rubric
(:func:`~threetears.evals.ops.report_writer.writer_contract.refuse_unfixed_scoring_judge`), and one that ran over
anything but held-out writer cases: each run must name its case set, and each case it ran must be a writer case its
mint put on the held-out side of its split, in the set the run names. The dev side is where a change is tuned, so a
decision read there would be read on the cases it was fitted to. And a pairwise job that is not a completed pass, or
whose report names another campaign, candidate, control, judge or preference criterion than the read's.

**The read never writes.** Promotion is a person's act: they record it as the campaign's call (``campaign_record``,
naming the candidate) and commit the change with its entry in the skill ledger
(:mod:`~threetears.evals.analysis.analyst.skill_ledger`). Each read states the basis the build's own skill shipped on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, ValidationError

from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.analyst.skill_ledger import ShippedSkill, skill_record
from threetears.evals.analysis.report.blocks import Verdict
from threetears.evals.analysis.service import assemble_campaign_bundle, campaign_verdicts
from threetears.evals.kernel.declaration import CampaignArm, resolve_control_arm
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.kernel.numbers import format_number
from threetears.evals.kernel.remedies import FIND_CAMPAIGN
from threetears.evals.kernel.run_pages import results_without_payloads
from threetears.evals.ops.jobs import JobStatus
from threetears.evals.ops.report_writer.preference import preference_rubric
from threetears.evals.ops.report_writer.writer_bank import writer_case_bank
from threetears.evals.ops.report_writer.writer_contract import refuse_unfixed_scoring_judge
from threetears.evals.ops.report_writer.writer_pairwise import WriterPairwiseReport
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import CaseSetRef, EvalRun, fingerprint_criterion

#: What a promotion read answers.
PromotionDecision = Literal["promote", "keep", "undecided"]


class WriterPromotion(EvalBaseModel):
    """What the promotion read answered for one candidate arm against the control, and what it read."""

    campaign_id: str = Field(description="The campaign read.")
    decision: PromotionDecision = Field(
        description="promote, keep or undecided; undecided never promotes.",
    )
    why: str = Field(description="The decision in one sentence; an undecided one names the primary reading's reason.")
    readout: str = Field(description="Every reading the decision read, its role and what it reads, as a table.")
    candidate: str = Field(description="The candidate arm, by its label.")
    candidate_key: str = Field(description="The candidate arm's variant key, which a call records.")
    control: str = Field(description="The control arm, by its label.")
    levers: dict[str, str] = Field(description="The candidate's lever map: each lever's level, as a reader names it.")
    changed: list[str] = Field(description="The levers whose level differs from the control's, by name.")
    pairwise_job: str | None = Field(description="The pairwise pass's job read; None when none was named.")
    primary: Verdict | None = Field(
        description="The primary reading: the pass's verdict on report_writer.preferred; None when no pass was read."
    )
    verdicts: list[Verdict] = Field(
        description="The candidate's corrected verdicts, each reading's, in page order: gated on regression only."
    )
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


def _primary(
    pairwise_job: JobStatus,
    *,
    campaign_id: str,
    candidate: CampaignArm,
    control: CampaignArm,
    judge_model: str,
    results: frozenset[str],
) -> Verdict:
    """The pass's verdict over every pair, refusing a job that is no completed pass over what this read reads:
    ``results`` are the ids of both arms' results in the campaign's runs as they stand."""
    job = pairwise_job.job_id
    if pairwise_job.kind != "action" or pairwise_job.state != "completed" or pairwise_job.result is None:
        raise ValidationFailedError(
            f"job {job} ({pairwise_job.kind}, {pairwise_job.state}) is not a completed writer_pairwise pass, and "
            "only a completed pass's report holds the primary reading",
            remedy="start the pass (writer_pairwise), poll it until it completes (job_poll), then read with its job id",
        )
    try:
        report = WriterPairwiseReport.model_validate(pairwise_job.result)
    except ValidationError as other:
        raise ValidationFailedError(
            f"job {job}'s result is no writer_pairwise report",
            remedy="name the job writer_pairwise started for this campaign's candidate",
        ) from other
    named = [
        f"{what} {theirs}, where the read's is {ours}"
        for what, theirs, ours in (
            ("campaign", report.campaign_id, campaign_id),
            ("candidate", report.candidate_key, candidate.variant_key),
            ("control", report.control_key, control.variant_key),
            ("judge", report.judge_model, judge_model),
            ("criterion", report.criterion, fingerprint_criterion(preference_rubric())),
        )
        if theirs != ours
    ]
    if named:
        raise ValidationFailedError(
            f"job {job}'s pass names " + "; ".join(named) + ", so its preference is no reading of this campaign",
            remedy="run the pass over this campaign's candidate (writer_pairwise), then read with its job id",
        )
    read = {result for pair in report.pairs for result in (pair.candidate_result_id, pair.control_result_id)}
    gone, unread = read - results, results - read - set(report.unpaired)
    if gone or unread:
        raise ValidationFailedError(
            f"job {job}'s pass read {len(gone)} result(s) the campaign's runs no longer hold and left {len(unread)} "
            "of theirs unread, so its preference is a reading of other runs",
            remedy="run the pass again over the campaign as it stands (writer_pairwise), then read with its job id",
        )
    return report.whole.verdict


def _reads(verdict: Verdict) -> str:
    """What a verdict reads: its outcome, or ``too_few_cases`` when it was read on fewer cases than a rate needs."""
    return "too_few_cases" if verdict.reason == "too_few_cases" else verdict.outcome


def _decided(
    primary: Verdict | None, verdicts: Sequence[Verdict], *, candidate: str, control: str
) -> tuple[PromotionDecision, str]:
    """The decision rule (the module docstring), and its sentence."""
    reads = None if primary is None else _reads(primary)
    regressed = [verdict for verdict in verdicts if verdict.kind == "contrast" and _reads(verdict) == "regressed"]
    guardrails = [verdict for verdict in verdicts if verdict.kind == "guardrail"]
    worse = [
        *(["the primary reading"] if reads == "regressed" else []),
        *(verdict.heading for verdict in regressed),
        *(f"guardrail {verdict.heading} (breached)" for verdict in guardrails if verdict.outcome == "breached"),
    ]
    if worse:
        return "keep", f"Keep the control: {candidate} is shown worse than {control} on {', '.join(worse)}."
    unheld = sum(verdict.outcome != "held" for verdict in guardrails)
    if reads == "improved" and not unheld:
        return "promote", (
            f"Promote {candidate}: the judge prefers its pages to {control}'s on the primary reading, no gated "
            "reading is shown worse and every guardrail is held."
        )
    if reads == "improved":
        return "undecided", (
            f"Undecided: the judge prefers {candidate}'s pages on the primary reading, but {unheld} of "
            f"{len(guardrails)} guardrails are not shown held."
        )
    detail = "" if primary is None or primary.reason_detail is None else f" ({primary.reason_detail})"
    because = {
        None: "no pairwise pass was read, so the primary reading is absent",
        "too_few_cases": f"the primary reading has too few cases to decide{detail}",
        "not_separated": f"the primary reading does not separate {candidate} from {control}",
    }.get(reads, f"the primary reading is {str(reads).replace('_', ' ')}{detail}")
    return "undecided", f"Undecided: {because}, and only an improved primary reading promotes."


def _readout(primary: Verdict | None, verdicts: Sequence[Verdict]) -> str:
    """Each reading the decision read as one table row: its role, what it reads, its difference and interval."""
    role = {"contrast": "gated", "guardrail": "guardrail", "bar": "reported"}
    rows = [
        [preference_rubric().name, "primary", "not read", "", "", ""]
        if primary is None
        else [preference_rubric().name, *_row(primary, "primary")[1:]],
        *(_row(verdict, role[verdict.kind]) for verdict in verdicts),
    ]
    lines = ["| Reading | Role | Reads | Difference | Interval | Level |", "|---|---|---|---|---|---|"]
    return "\n".join([*lines, *("| " + " | ".join(row) + " |" for row in rows)])


def _row(verdict: Verdict, role: str) -> list[str]:
    """One verdict's row of the readout."""
    interval = verdict.interval
    return [
        verdict.heading,
        role,
        _reads(verdict).replace("_", " "),
        "" if verdict.delta is None else format_number(verdict.delta),
        "" if interval is None else f"{format_number(interval[0])} to {format_number(interval[1])}",
        "" if verdict.interval_level is None or interval is None else format_number(verdict.interval_level),
    ]


def writer_promotion(
    host: EvalHost, campaign_id: str, scope_id: str, *, candidate: str, pairwise_job: JobStatus | None = None
) -> WriterPromotion:
    """Read whether a report writer campaign's candidate arm should replace its control (the module docstring).

    :param host: The host whose store holds the campaign and its runs and cases.
    :ptype host: EvalHost
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param candidate: The candidate arm, by its label or its variant key.
    :ptype candidate: str
    :param pairwise_job: The completed ``writer_pairwise`` job over this campaign's candidate (``job_poll``'s status),
        whose report holds the primary reading; ``None`` reads no primary, so the read never promotes.
    :ptype pairwise_job: JobStatus | None
    :return: The decision and what it read.
    :rtype: WriterPromotion
    :raises NotFoundError: No campaign with that id in the scope, or a run it names is not stored.
    :raises ValidationFailedError: The campaign has no control, the candidate is none of its arms or is the control,
        an arm's lever map was not recorded, its arms were not scored by one judge on one rubric, or it ran over
        anything but held-out writer cases; the pairwise job is no completed pass, or its report names another
        campaign, candidate, control, judge or criterion than the read's; or the build's skill has no ledger entry.
    """
    campaign = host.storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id, remedy=FIND_CAMPAIGN)
    loaded = {run_id: host.storage.load_eval_run(run_id, scope_id) for run_id in campaign.run_ids}
    if missing := [run_id for run_id, run in loaded.items() if run is None]:
        # A run the campaign names and the store cannot return would leave its arm's cases and judge unchecked, and
        # the decision read on fewer runs than the campaign ran with nothing saying so.
        raise NotFoundError(
            f"campaign {campaign_id}'s run",
            ", ".join(missing),
            remedy="restore the run, or take it off the campaign (campaign_runs_remove), then read the campaign again",
        )
    runs = [run for run in loaded.values() if run is not None]
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
    judge_model = runs[0].judge_model
    assert judge_model is not None  # refuse_unfixed_scoring_judge refuses a run that recorded none
    primary = None
    if pairwise_job is not None:
        arms = keys | {control.variant_key, *resolve_control_arm(control).variant_keys}
        results = frozenset(
            result.id
            for run in runs
            for result in results_without_payloads(host.storage, run.id, scope_id)
            if result.variant_key in arms
        )
        primary = _primary(
            pairwise_job,
            campaign_id=campaign_id,
            candidate=arm,
            control=control,
            judge_model=judge_model,
            results=results,
        )
    verdicts = [verdict for verdict in campaign_verdicts(host, campaign_id, scope_id) if verdict.variant_key in keys]
    decision, why = _decided(primary, verdicts, candidate=arm.label, control=control.label)
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
        readout=_readout(primary, verdicts),
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
        pairwise_job=None if pairwise_job is None else pairwise_job.job_id,
        primary=primary,
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
