"""The promotion read: whether a report writer campaign's candidate arm is shown better than its control.

A change to how the report writer works (its skill, its reasoning effort, its revisions, or its model) ships only on a
measured decision, and this is the read that decides it. It adds no statistics. It reads one pre-registered primary
reading, the pairwise pass's preference (``report_writer.preferred``, at alpha 0.05, a family of one; its report is
the completed ``writer_pairwise`` job's result), and the campaign's corrected verdicts
(:func:`~threetears.evals.analysis.campaign_verdicts`), which gate on regression only:

- ``promote`` when the primary reads improved on the cases a rate needs, no gated reading reads regressed, and every
  guardrail is held;
- ``keep`` when the primary reads regressed, a gated reading reads regressed, or a guardrail is breached;
- ``undecided`` otherwise: the primary not separated, improved on too few cases, untested, void or not read at all, or
  a guardrail not shown held. **Undecided never promotes.**

A reading the package's gate calls regressed keeps the control whatever its reason, one read on fewer cases than a
rate needs (reason ``too_few_cases``) included, and its row states that reason beside it: too few cases only stops an
improved primary from promoting. A gated reading that is not separated decides nothing either way: the campaign's
family of readings is too wide to separate on a small held-out set, so only the primary is asked to.

**The instrument checks.** Order consistency is the share of the cases the judge scored in both orders whose two
orders picked the same side; under 0.7 the primary reading is void, since a judge that contradicts itself that often
has read the order of the pages and not the pages, so it neither keeps nor promotes, and the read is undecided unless
another reading keeps. A case scored in one order only is counted apart, in no rate. The reading over the pairs whose
pages were both stored finished is reported beside the primary and decides nothing: a draft's page carries its open
findings, so a primary the finished-only reading does not repeat may be the judge preferring the page without them.
A primary that does not separate states how many held-out cases like these would settle it, by the bounded interval's
own projection (:func:`~threetears.evals.analysis.stats.bounded_cases_to_clear`, which the campaign report's ``To show
the floors hold`` reads): a size to plan the next campaign at, never a reason to read this one again.

**A pass that stopped early.** A pass judging under a declared stopping rule
(:class:`~threetears.evals.ops.report_writer.writer_pairwise.StoppingRule`) may stop once its preference separates;
each look is a prefix of one bet fixed for the rule's planned cases, so the primary holds its level at the stop. Every
arm ran every case launched before the pass began, so the gated readings and guardrails are read as always, on every
case run: under a writer eval that launches in batches, once, at the stop, on the cases launched by then. The read
says what the pass did in one sentence (``stopping_line``).

**Generic over levers.** The candidate is an arm, and what it changed is read off the arms' recorded lever maps, so a
model, an effort or a host's own lever is decided as a skill text is.

**Two editors.** A candidate whose ``editor.md`` differs from the control's has each arm's own editor decide its
must-fix and should-fix findings open and whether its report finished, which then measure two instruments. The rule
still reads them; the decision's sentence never names one
(:func:`~threetears.evals.ops.report_writer.writer_contract.uncompared_readings`), and says instead what the readings
the comparison counts came to. The caller says whether the arms ran one editor (``one_editor``), as the writer eval
does; a caller that does not say (the ``writer_promotion`` action) is answered for the arms' recorded skill levels: one
editor when the skills are the same, else two, since a skill that differs may differ in its ``editor.md``.

**What it refuses.** A campaign whose arms were not scored by one judge on one rubric
(:func:`~threetears.evals.ops.report_writer.writer_contract.refuse_unfixed_scoring_judge`), and one that ran over
anything but held-out writer cases: each run must name its case set, and each case it ran must be a writer case its
mint put on the held-out side of its split, in the set the run names or in a batch of it: a set made for one launch,
named for the held-out set version it is cut from (:func:`held_out_batch_name`), which lists every case of it. The dev side is where a change is tuned, so a
decision read there would be read on the cases it was fitted to. And a pairwise job that is not a completed pass, or
whose report names another campaign, candidate, control, judge or preference criterion than the read's.

**The read never writes.** Promotion is a person's act: they record it as the campaign's call (``campaign_record``,
naming the candidate) and commit the change with its entry in the skill ledger
(:mod:`~threetears.evals.analysis.analyst.skill_ledger`). Each read states the basis the build's own skill shipped on.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Final, Literal

from pydantic import Field, ValidationError

from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.analyst.skill_ledger import ShippedSkill, skill_record
from threetears.evals.analysis.report.blocks import Verdict
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.service import assemble_campaign_bundle, campaign_verdicts
from threetears.evals.analysis.stats import INTERVAL_LEVEL, PROJECTION_CASE_CAP, bounded_cases_to_clear
from threetears.evals.analysis.style import CANNOT_TELL_APART, cannot_tell_apart
from threetears.evals.kernel.declaration import CampaignArm, resolve_control_arm
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.kernel.remedies import FIND_CAMPAIGN
from threetears.evals.kernel.run_pages import results_without_payloads
from threetears.evals.ops.jobs import JobStatus
from threetears.evals.ops.report_writer.preference import preference_rubric
from threetears.evals.ops.report_writer.writer_bank import writer_case_bank
from threetears.evals.ops.report_writer.writer_contract import (
    SKILL_LEVER,
    refuse_unfixed_scoring_judge,
    uncompared_readings,
)
from threetears.evals.ops.report_writer.writer_pairwise import (
    PREFERRED_HEADING,
    PairwiseReliability,
    WriterPairwiseReport,
)
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import CaseSetRef, EvalRun, fingerprint_criterion

#: What a promotion read answers.
PromotionDecision = Literal["promote", "keep", "undecided"]

#: The order consistency under which the primary reading is void (the reset design's instrument check).
_ORDER_CONSISTENCY_FLOOR = 0.7

#: What a void primary's reason says, the same in the decision's sentence and in the readout.
_VOID = f"order consistency below {_ORDER_CONSISTENCY_FLOOR}"

#: The row the finished-only reading is read by.
_FINISHED_ONLY = f"{PREFERRED_HEADING}, both pages finished"


class WriterPromotion(EvalBaseModel):
    """What the promotion read answered for one candidate arm against the control, and what it read."""

    campaign_id: str = Field(description="The campaign read.")
    decision: PromotionDecision = Field(
        description="promote, keep or undecided; undecided never promotes.",
    )
    why: str = Field(description="The decision in one sentence; an undecided one names the primary reading's reason.")
    readout: str = Field(
        description="Every reading the decision read, its role and what it reads, then the judge's order checks, as "
        "tables."
    )
    candidate: str = Field(description="The candidate arm, by its label.")
    candidate_key: str = Field(description="The candidate arm's variant key, which a call records.")
    control: str = Field(description="The control arm, by its label.")
    levers: dict[str, str] = Field(description="The candidate's lever map: each lever's level, as a reader names it.")
    changed: list[str] = Field(description="The levers whose level differs from the control's, by name.")
    pairwise_job: str | None = Field(description="The pairwise pass's job read; None when none was named.")
    primary: Verdict | None = Field(
        description="The primary reading: the pass's verdict on report_writer.preferred; None when no pass was read."
    )
    void: str | None = Field(
        description="Why the primary reading is void, when it is (order consistency below 0.7); None when it stands "
        "or none was read."
    )
    finished_only: Verdict | None = Field(
        description="The pass's verdict over the pairs whose two pages were both stored finished, reported beside the "
        "primary and deciding nothing; None when no pass was read."
    )
    reliability: PairwiseReliability | None = Field(
        description="The pass's order consistency, one-order cases counted apart, first-slot lean, can't-tell and "
        "failed calls; None when no pass was read."
    )
    cases_to_settle: int | None = Field(
        description="The held-out cases like these at which a primary that does not separate would, by its bounded "
        "interval at the spread its cases showed; one over the projection's cap when even that many would not; None "
        "for a primary that separated, is untested or void, or whose estimate is zero."
    )
    verdicts: list[Verdict] = Field(
        description="The candidate's corrected verdicts, each reading's, in page order: gated on regression only."
    )
    cases: int = Field(description="The held-out cases the campaign's runs ran.")
    stopping_line: str | None = Field(
        default=None,
        description="What the pass did against its stopping rule, in one sentence; None when no pass was read or its "
        "report records none.",
    )
    case_sets: list[CaseSetRef] = Field(description="The held-out case sets the runs named.")
    cost_usd: float | None = Field(description="What the candidate's trials cost; None when none was priced.")
    control_cost_usd: float | None = Field(description="What the control's trials cost; None when none was priced.")
    unpriced_trials: int = Field(description="Trials of either arm whose spend went unpriced, so not in the costs.")
    writer_model: str = Field(description="The model the candidate arm wrote with.")
    shipped: ShippedSkill = Field(description="The skill ledger's entry for the skill this build ships.")
    one_editor: bool = Field(
        default=True,
        description="Whether the candidate and the control ran one editor.md; with two, each arm's own editor's "
        "readings (must-fix and should-fix findings open, report finished) measure two instruments, so the decision's "
        "sentence never names one, though the rule still reads them.",
    )


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
    batched = _batched(host, runs, scope_id)
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
        elif origin.case_set not in named and origin.case_set not in batched.get(case_id, ()):
            faults.append(f"case {case_id} was minted into {origin.case_set!r}, which no run names")
    if faults:
        raise ValidationFailedError(
            "a promotion is decided on held-out writer cases alone, and " + "; ".join(faults),
            remedy="launch the arms over the held-out writer set (case_set), and decide only on that campaign",
        )
    refs = sorted({run.case_set for run in runs if run.case_set is not None}, key=lambda ref: (ref.name, ref.version))
    return refs, len(ids)


#: A batch's set name: the held-out set's label it is a batch of, then its place among the batches.
_BATCH_NAME: Final = re.compile(r"^(?P<name>.+) v(?P<version>[1-9][0-9]*) batch [1-9][0-9]* of [1-9][0-9]*$")


def held_out_batch_name(held_out: CaseSetRef, index: int, batches: int) -> str:
    """The name of the set one batch of a launch made in batches runs: the held-out set's label it is a batch of, and
    its place (``writer-held-out v7 batch 2 of 3``). The promotion read accepts a case run from such a set, made for
    one launch, only as a case of the held-out set version it names, and only when that version lists every case of it.

    :param held_out: The held-out set version the batches are cut from.
    :ptype held_out: CaseSetRef
    :param index: The batch, from 0.
    :ptype index: int
    :param batches: How many batches the launch is cut into.
    :ptype batches: int
    :return: The name.
    :rtype: str
    """
    return f"{held_out.label} batch {index + 1} of {batches}"


def _batched(host: EvalHost, runs: Sequence[EvalRun], scope_id: str) -> dict[str, set[str]]:
    """Each case a run ran from a batch of a held-out set, with the held-out sets those batches belong to: a set made
    for one launch, named as a batch of a held-out set version (:func:`held_out_batch_name`) that lists every one of its
    cases. Any other set made for one launch belongs to no held-out set, so its cases are refused as before."""
    batched: dict[str, set[str]] = {}
    for run in runs:
        ref = run.case_set
        named = None if ref is None else _BATCH_NAME.match(ref.name)
        if ref is None or named is None:
            continue
        listed = host.storage.load_case_set(ref.name, ref.version, scope_id)
        parent = host.storage.load_case_set(named["name"], int(named["version"]), scope_id)
        if (
            listed is None
            or listed.tracked
            or parent is None
            or not set(listed.test_case_ids) <= set(parent.test_case_ids)
        ):
            continue
        for case_id in run.test_case_ids:
            batched.setdefault(case_id, set()).add(parent.name)
    return batched


def _pass_report(
    pairwise_job: JobStatus,
    *,
    campaign_id: str,
    candidate: CampaignArm,
    control: CampaignArm,
    judge_model: str,
    results: frozenset[str],
    outside: Callable[[Sequence[str]], frozenset[str]],
) -> WriterPairwiseReport:
    """The pass's report, refusing a job that is no completed pass over what this read reads:
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
    # A batch that stored no page on either arm is left out of the campaign's readings and read by the pass beside
    # them: its results are the pass's to hold, never the campaign's.
    aside = outside(report.outside_run_ids)
    gone, unread = read - results - aside, results - read - set(report.unpaired)
    if gone or unread:
        raise ValidationFailedError(
            f"job {job}'s pass read {len(gone)} result(s) the campaign's runs no longer hold and left {len(unread)} "
            "of theirs unread, so its preference is a reading of other runs",
            remedy="run the pass again over the campaign as it stands (writer_pairwise), then read with its job id",
        )
    return report


def _void(reliability: PairwiseReliability) -> str | None:
    """Why the pass's primary reading is void, when its order consistency is under the floor; else None."""
    rate = reliability.order_consistency
    if rate is None or rate >= _ORDER_CONSISTENCY_FLOOR:
        return None
    return (
        f"{_VOID} ({reliability.order_consistent} of {reliability.both_orders} cases scored in both orders picked the "
        "same side)"
    )


def _cases_to_settle(report: WriterPairwiseReport) -> int | None:
    """The held-out cases like these at which a primary that does not separate would: the fewest at which its bounded
    interval, on its cases' own spread, reaches no further toward zero than its estimate lies from it.

    None for a primary that separated or is untested, an estimate of zero, and an interval that already excludes zero
    on the cases read while the test does not separate, where no count of cases like these is the interval's to name.
    """
    verdict = report.whole.verdict
    if verdict.outcome != "not_separated" or not verdict.delta:
        return None
    shares = [case.share for case in report.cases if case.share is not None]
    control = [1.0 - share for share in shares]
    # Oriented so the estimate is above zero: the projection reads how far below its estimate the interval reaches.
    lower, upper = (control, shares) if verdict.delta > 0 else (shares, control)
    needed = bounded_cases_to_clear(
        lower,
        upper,
        paired=True,
        value_range=(0.0, 1.0),
        confidence=verdict.interval_level or INTERVAL_LEVEL,
        room=abs(verdict.delta),
    )
    return None if needed is None or needed <= len(shares) else needed


def _reads(verdict: Verdict) -> str:
    """What a verdict reads, as its row prints it: its outcome (the undecided one in the glossary's words, ``cannot tell
    apart``), with too few cases beside it when it was read on fewer cases than a rate needs."""
    outcome = CANNOT_TELL_APART if verdict.outcome == "not_separated" else verdict.outcome.replace("_", " ")
    if verdict.reason == "too_few_cases":
        return f"{outcome} (too few cases)"
    # An untested reading names why, in the one sentence its row has.
    return f"{outcome}: {verdict.reason_detail}" if verdict.outcome == "untested" and verdict.reason_detail else outcome


def _worse(standing: Verdict | None, verdicts: Sequence[Verdict]) -> list[str]:
    """What keeps the control: the standing primary regressed, each gated reading regressed, each guardrail breached."""
    return [
        *(
            [_named("the primary reading", standing)]
            if standing is not None and standing.outcome == "regressed"
            else []
        ),
        *(
            _named(verdict.heading, verdict)
            for verdict in verdicts
            if verdict.kind == "contrast" and verdict.outcome == "regressed"
        ),
        *(
            f"guardrail {verdict.heading} (breached)"
            for verdict in verdicts
            if verdict.kind == "guardrail" and verdict.outcome == "breached"
        ),
    ]


def _improved(standing: Verdict | None) -> bool:
    """Whether the standing primary reads improved on the cases a rate needs."""
    return standing is not None and standing.outcome == "improved" and standing.reason != "too_few_cases"


def _decided(
    primary: Verdict | None,
    void: str | None,
    verdicts: Sequence[Verdict],
    *,
    candidate: str,
    control: str,
    uncompared: Sequence[str] = (),
) -> tuple[PromotionDecision, str]:
    """The decision rule (the module docstring), and its sentence.

    The rule reads every verdict. The sentence names only the readings the comparison counts, never one in
    ``uncompared`` (with two editors, each arm's own editor's): a keep that only those decide says what the counted
    readings came to (``Keep the control: v2 was not shown better than v1; on the primary reading the evidence cannot
    tell them apart.``).
    """
    standing = None if void is not None else primary
    unheld_any = any(verdict.kind == "guardrail" and verdict.outcome != "held" for verdict in verdicts)
    decision: PromotionDecision = (
        "keep" if _worse(standing, verdicts) else "promote" if _improved(standing) and not unheld_any else "undecided"
    )
    counted = [verdict for verdict in verdicts if verdict.name not in uncompared]
    if worse := _worse(standing, counted):
        return decision, f"Keep the control: {candidate} is shown worse than {control} on {', '.join(worse)}."
    lead = "Keep the control:" if decision == "keep" else "Undecided:"
    guardrails = [verdict for verdict in counted if verdict.kind == "guardrail"]
    unheld = sum(verdict.outcome != "held" for verdict in guardrails)
    if _improved(standing) and not unheld:
        if decision == "promote":
            return decision, (
                f"Promote {candidate}: the judge prefers its pages to {control}'s on the primary reading, no gated "
                "reading is shown worse and every guardrail is held."
            )
        return decision, (
            f"{lead} the judge prefers {candidate}'s pages to {control}'s on the primary reading, and only readings "
            "each arm's own editor decides, which are not compared, stand in its way."
        )
    if _improved(standing):
        return decision, (
            f"{lead} the judge prefers {candidate}'s pages on the primary reading, but {unheld} of "
            f"{len(guardrails)} guardrails are not shown held."
        )
    detail = "" if primary is None or primary.reason_detail is None else f" ({primary.reason_detail})"
    if primary is None:
        because = "no pairwise pass was read, so the primary reading is absent"
    elif void is not None:
        because = f"{void}, so the primary reading is void"
    elif primary.reason == "too_few_cases":
        because = f"the primary reading has too few cases to decide{detail}"
    elif primary.outcome == "not_separated":
        # A keep's sentence has named both arms already, so its clause says them once.
        apart = "the evidence cannot tell them apart" if decision == "keep" else cannot_tell_apart([candidate], control)
        because = f"on the primary reading {apart}"
    else:
        because = f"the primary reading is {primary.outcome.replace('_', ' ')}{detail}"
    if decision == "keep":
        return decision, f"Keep the control: {candidate} was not shown better than {control}; {because}."
    return decision, f"Undecided: {because}, and only an improved primary reading promotes."


def _named(label: str, verdict: Verdict) -> str:
    """A reading the decision names, with too few cases beside it when it was read on fewer than a rate needs."""
    return f"{label} (on too few cases)" if verdict.reason == "too_few_cases" else label


def _readout(
    report: WriterPairwiseReport | None, void: str | None, settle: int | None, verdicts: Sequence[Verdict]
) -> str:
    """Each reading the decision read as one table row (its role, what it reads, its difference and interval), then
    the judge's order checks as a second table when a pass was read."""
    role = {"contrast": "gated", "guardrail": "guardrail", "bar": "reported"}
    forms = PageForms.off_page()
    if report is None:
        head = [[PREFERRED_HEADING, "primary", "not read", "", "", ""]]
    else:
        primary = _row(report.whole.verdict, "primary", forms)
        if void is not None:
            primary[2] = f"{primary[2]} (void, {_VOID})"
        head = [primary, [_FINISHED_ONLY, *_row(report.finished_only.verdict, "reported", forms)[1:]]]
    rows = [*head, *(_row(verdict, role[verdict.kind], forms) for verdict in verdicts)]
    lines = ["| Reading | Role | Reads | Difference | Interval | Level |", "|---|---|---|---|---|---|"]
    readings = "\n".join([*lines, *("| " + " | ".join(row) + " |" for row in rows)])
    if report is None:
        return readings
    reliability = report.reliability
    rate = reliability.order_consistency
    checks = [
        ["Cases scored in both orders", str(reliability.both_orders)],
        ["Of those, both orders picked the same side", str(reliability.order_consistent)],
        [
            f"Order consistency, the primary void below {_ORDER_CONSISTENCY_FLOOR}",
            "not measured" if rate is None else forms.lone(rate).text,
        ],
        ["Cases scored in one order only, counted apart", str(reliability.one_order)],
        ["Answers the judge could not tell", str(reliability.cannot_tell)],
        ["Calls that failed", str(reliability.failed)],
        *(
            []
            if settle is None
            else [
                [
                    "Held-out cases like these that would settle the primary",
                    f"more than {PROJECTION_CASE_CAP}" if settle > PROJECTION_CASE_CAP else str(settle),
                ]
            ]
        ),
    ]
    table = ["| Order check | Figure |", "|---|---|", *("| " + " | ".join(row) + " |" for row in checks)]
    return readings + "\n\n" + "\n".join(table)


def _row(verdict: Verdict, role: str, forms: PageForms) -> list[str]:
    """One verdict's row of the readout, its figures as ``forms`` states them."""
    interval, level = verdict.interval, verdict.interval_level
    return [
        verdict.heading,
        role,
        _reads(verdict),
        "" if verdict.delta is None else forms.lone(verdict.delta).text,
        "" if interval is None else f"{forms.lone(interval[0]).text} to {forms.lone(interval[1]).text}",
        "" if level is None or interval is None else forms.lone(level).text,
    ]


def writer_promotion(
    host: EvalHost,
    campaign_id: str,
    scope_id: str,
    *,
    candidate: str,
    pairwise_job: JobStatus | None = None,
    one_editor: bool | None = None,
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
    :param one_editor: Whether the candidate and the control ran one editor (its ``editor.md``, model and effort); with
        two, the decision's sentence names only the readings the comparison counts
        (:func:`~threetears.evals.ops.report_writer.writer_contract.uncompared_readings`). ``None`` answers it for the
        arms' recorded skill levels: one editor when they are the same skill, else two.
    :ptype one_editor: bool | None
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
    if one_editor is None:
        # A skill that differs may differ in its editor.md, so an unsaid editor is read as two: its readings leave the
        # sentence rather than stand in it as a verdict on the writing.
        mine, theirs = arm.levers.get(SKILL_LEVER), control.levers.get(SKILL_LEVER)
        one_editor = (None if mine is None else mine.content_hash) == (None if theirs is None else theirs.content_hash)
    keys = {arm.variant_key, *resolve_control_arm(arm).variant_keys}
    control_keys = {control.variant_key, *resolve_control_arm(control).variant_keys}
    judge_model = runs[0].judge_model
    assert judge_model is not None  # refuse_unfixed_scoring_judge refuses a run that recorded none
    report = None
    if pairwise_job is not None:
        arms = keys | {control.variant_key, *resolve_control_arm(control).variant_keys}
        results = frozenset(
            result.id
            for run in runs
            for result in results_without_payloads(host.storage, run.id, scope_id)
            if result.variant_key in arms
        )
        report = _pass_report(
            pairwise_job,
            campaign_id=campaign_id,
            candidate=arm,
            control=control,
            judge_model=judge_model,
            results=results,
            outside=lambda run_ids: frozenset(
                result.id for run_id in run_ids for result in results_without_payloads(host.storage, run_id, scope_id)
            ),
        )
    verdicts = [verdict for verdict in campaign_verdicts(host, campaign_id, scope_id) if verdict.variant_key in keys]
    primary = None if report is None else report.whole.verdict
    void = None if report is None else _void(report.reliability)
    settle = None if report is None or void is not None else _cases_to_settle(report)
    decision, why = _decided(
        primary,
        void,
        verdicts,
        candidate=arm.label,
        control=control.label,
        uncompared=uncompared_readings(one_editor=one_editor),
    )
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
        readout=_readout(report, void, settle, verdicts),
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
        void=void,
        finished_only=None if report is None else report.finished_only.verdict,
        reliability=None if report is None else report.reliability,
        cases_to_settle=settle,
        verdicts=verdicts,
        cases=cases,
        stopping_line=None if report is None or report.stopping is None else report.stopping.line,
        case_sets=case_sets,
        cost_usd=sum(costs["candidate"]) if costs["candidate"] else None,
        control_cost_usd=sum(costs["control"]) if costs["control"] else None,
        unpriced_trials=sum(board.n_cost_unpriced for board in boards if board.variant_key in keys | control_keys),
        writer_model=arm.levers[CANDIDATE_MODEL_LEVER].display,
        shipped=skill_record(load_skill()),
        one_editor=one_editor,
    )


#: The engine's ``writer_promotion`` action reads a decision for a host when a person asks whether to promote; the
#: decision's literal waits for a caller (``tests/test_report_writer_exports.py``).
__all__ = ["WriterPromotion", "held_out_batch_name", "writer_promotion"]
