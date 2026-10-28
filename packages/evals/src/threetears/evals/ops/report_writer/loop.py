"""The writing loop's rounds: the analyst drafts, the editor reviews, the analyst revises, until a review clears it.

:func:`write_rounds` runs plan §4.2's steps 1 to 6 over one frozen snapshot:

1. The analyst drafts (:meth:`~threetears.evals.ops.report_writer.analyst.AnalystSession.draft`): its submission
   passes the document's validity and the invariants before anything else sees it; a fault goes back to it as a
   bounce, which costs no editor call.
2. The editor reviews the document in a session of its own
   (:func:`~threetears.evals.ops.report_writer.editor.review_report`), given the page, ``omitted``, the computed facts
   and its earlier findings, and nothing of the analyst's session.
3. A review that passes with no ``must_fix`` finding ends the loop ``stored``. Otherwise the analyst revises in its
   own conversation, sent the review (each item's score and reason, and every finding), and the editor reviews the
   revision. :attr:`WriterLimits.revisions` revisions at most (2, so at most 3 reviews); a review that fails after the
   last ends the loop ``stored_with_findings``.

**Endings** (plan §4.4), each a value the caller stores or reports:

- ``stored``: a review cleared the document, which is stored with the review's findings (``should_fix`` ones only);
- ``stored_with_findings``: the revisions ran out; the last document is stored with its review's findings;
- ``over_budget``: the report's budget refused a turn, or the analyst called a tool after it was told to
  submit. The last document a review was read for is stored with that review's findings; with none, nothing;
- ``refused``: a submission still had faults after its round's bounces; nothing is stored, the faults are returned;
- ``review_failed``: the editor's answer could not be read twice; nothing is stored;
- ``failed``: the model refused, a reply stopped short, or the client failed (the provider, after its own retries);
  nothing is stored. The tally is the caller's and keeps every turn sent.

**The loop reads no environment and chooses nothing by kind.** The model, its sessions' clients, the mounts, the
skill texts, the checker and every limit come from the caller, and no condition here reads the document's story, a
claim, or what kind of eval the snapshot holds: what ends a round is the checks, the review's scores and its findings'
severities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from threetears.evals.analysis.analyst.skill import SkillText
from threetears.evals.analysis.report.md_parts import one_line
from threetears.evals.analysis.report_decision import every_caveat
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc, ReportFinding
from threetears.evals.ops.report_writer.analyst import (
    ANALYST_SESSION,
    AnalystLimits,
    AnalystOutcome,
    AnalystSession,
    InvariantCheck,
    SubmissionFault,
)
from threetears.evals.ops.report_writer.editor import (
    PASS_TOTAL,
    EditorLimits,
    EditorOutcome,
    Review,
    RubricScore,
    editor_session,
    review_report,
)
from threetears.evals.ops.report_writer.session import ToolMount
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.schema.turns import ToolTurnClient

__all__ = ["TurnModel", "WriterEnding", "WriterLimits", "WritingOutcome", "write_rounds"]

log = get_logger(__name__)

#: How the writing loop ended (the module docstring says when, and what is stored).
WriterEnding = Literal["stored", "stored_with_findings", "over_budget", "refused", "review_failed", "failed"]

#: The most revisions a report is given (ANALYST §4): so at most three reviews.
_MOST_REVISIONS: Final = 2


class TurnModel(Protocol):
    """The model a report is written with: its id, and the client each named session's turns go through.

    ``threetears.evals.providers.Model`` is one: built by the host with its key, and with a ``turn_recording`` its
    sessions are recorded or replayed by name.

    :ivar model_name: The model id every session runs on, which its replies name.
    """

    model_name: str

    async def turn_pricing_refusal(self) -> str | None:
        """Why a turn on this model could not be bounded before it is sent, or ``None`` when every one can be.

        Asked before the report is started, so a run its budget could not admit a turn of is refused before it
        freezes or spends anything. It may bring the client's prices up to date first.

        :return: ``None``, or the pricing refusal, naming the model and what to do.
        :rtype: str | None
        :raises ValueError: The model sends no turn at all (its route or settings), which is no pricing refusal.
        """
        ...

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        """The client one named session's turns go through.

        :param name: The session's name (``analyst``, ``editor-1``).
        :ptype name: str
        :param snapshot: The snapshot the session reads.
        :ptype snapshot: str
        :return: The client.
        :rtype: ToolTurnClient
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class WriterLimits:
    """What the writing loop is held to (plan §4.3).

    :ivar analyst: The analyst's rounds: its tool calls per draft and revision, its bounces, every turn's limits.
    :ivar editor: Each review: its tool calls, every turn's limits.
    :ivar revisions: The revisions the analyst is given after a review that does not clear its report: 0 to 2.
    """

    analyst: AnalystLimits = field(default_factory=AnalystLimits)
    editor: EditorLimits = field(default_factory=EditorLimits)
    revisions: int = _MOST_REVISIONS

    def __post_init__(self) -> None:
        """Refuse a revision count outside 0 to 2.

        :raises ValueError: The count.
        """
        if isinstance(self.revisions, bool) or not isinstance(self.revisions, int):
            raise ValueError(f"revisions is a whole number, not {self.revisions!r}")
        if not 0 <= self.revisions <= _MOST_REVISIONS:
            raise ValueError(f"a report is given 0 to {_MOST_REVISIONS} revisions, not {self.revisions}")


@dataclass(frozen=True, slots=True, kw_only=True)
class WritingOutcome:
    """How the writing loop ended, and what it leaves to store.

    :ivar ending: How it ended (:data:`WriterEnding`).
    :ivar doc: The document to store; ``None`` for an ending that stores nothing.
    :ivar findings: The findings stored with it: the last review's of that document.
    :ivar reviews: Every review read, in round order.
    :ivar analyst_rounds: The analyst's rounds begun: the draft, then each revision.
    :ivar editor_rounds: The review rounds begun.
    :ivar faults: The faults of a submission that ended the loop ``refused``; empty otherwise.
    :ivar reason: Why the loop ended other than ``stored``, in a sentence.
    :ivar sessions: Every session begun, by name, as a stored value, whatever the ending: the analyst's
        (``AnalystSession.to_json()``, under ``analyst``) and each review round's (``ToolSession.to_json()``, under
        ``editor-<round>``).
    :ivar failure: The client's failure a ``failed`` loop ended on, when one did (the analyst's or a review's), so a
        caller that records it can ask the host what of it is safe to keep: its text, which ``reason`` carries, may
        hold a provider's account. Never stored.
    """

    ending: WriterEnding
    doc: ReportDoc | None = None
    findings: tuple[ReportFinding, ...] = ()
    reviews: tuple[Review, ...] = ()
    analyst_rounds: int = 0
    editor_rounds: int = 0
    faults: tuple[SubmissionFault, ...] = ()
    reason: str = ""
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)  # Any: JSON objects
    failure: Exception | None = field(default=None, compare=False, repr=False)


@dataclass
class _Rounds:
    """What the loop has seen so far: each review read, the document each was of, and each review round's session."""

    reviews: list[Review] = field(default_factory=list)
    reviewed: list[ReportDoc] = field(default_factory=list)
    editors: dict[str, dict[str, Any]] = field(default_factory=dict)  # Any: each editor session's JSON
    analyst_rounds: int = 0
    editor_rounds: int = 0


async def write_rounds(
    *,
    model: TurnModel,
    snapshot: str,
    report_input: ReportInput,
    analyst_mount: ToolMount,
    editor_mount: ToolMount,
    analyst_skill: SkillText,
    editor_skill: SkillText,
    check: InvariantCheck,
    tally: Tally,
    budget: ReportBudget,
    limits: WriterLimits | None = None,
) -> WritingOutcome:
    """Draft, review and revise a report over one frozen snapshot, until a review clears it or a limit ends it.

    :param model: The model every session runs on, and the client of each named session.
    :ptype model: TurnModel
    :param snapshot: The frozen snapshot's id.
    :ptype snapshot: str
    :param report_input: Its report input, which validity, the editor's page and its facts read.
    :ptype report_input: ReportInput
    :param analyst_mount: The ``evals_analyst`` tool on the analyst's cut, read-only, bound to its host and caller.
    :ptype analyst_mount: ToolMount
    :param editor_mount: The tool on the editor's cut, bound the same way.
    :ptype editor_mount: ToolMount
    :param analyst_skill: The analyst's system text.
    :ptype analyst_skill: SkillText
    :param editor_skill: The editor's system text.
    :ptype editor_skill: SkillText
    :param check: The invariants' checker on the snapshot, which every submission passes before review.
    :ptype check: InvariantCheck
    :param tally: The report's tally, which every turn of every session is counted in.
    :ptype tally: Tally
    :param budget: The report's budget (dollars, or tokens and turns), which admits every turn.
    :ptype budget: ReportBudget
    :param limits: What the loop is held to; plan §4.3's defaults when ``None``.
    :ptype limits: WriterLimits | None
    :return: How it ended, and what to store.
    :rtype: WritingOutcome
    """
    held = limits if limits is not None else WriterLimits()
    analyst = AnalystSession.start(
        model=model.model_name,
        client=model.turn_session(ANALYST_SESSION, snapshot=snapshot),
        mount=analyst_mount,
        skill=analyst_skill,
        snapshot=snapshot,
        arms=[arm.id for arm in report_input.arms],
        caveats=[caveat.id for caveat in every_caveat(report_input)],
        check=check,
        tally=tally,
        budget=budget,
        limits=held.analyst,
    )
    seen = _Rounds()
    try:
        seen.analyst_rounds += 1
        drafted = await analyst.draft()
        while True:
            if drafted.ending != "submitted" or drafted.doc is None:
                return _not_submitted(drafted, seen, analyst)
            doc = drafted.doc
            seen.editor_rounds += 1
            reviewed = await review_report(
                doc,
                report_input,
                round=drafted.round,
                snapshot=snapshot,
                earlier=seen.reviews[-1].findings if seen.reviews else (),
                model=model.model_name,
                client=model.turn_session(editor_session(drafted.round), snapshot=snapshot),
                mount=editor_mount,
                skill=editor_skill,
                tally=tally,
                budget=budget,
                limits=held.editor,
            )
            seen.editors[editor_session(drafted.round)] = reviewed.session
            if reviewed.ending != "reviewed" or reviewed.review is None:
                return _not_reviewed(reviewed, seen, analyst)
            review = reviewed.review
            seen.reviews.append(review)
            seen.reviewed.append(doc)
            if review.clears:
                return _outcome("stored", seen, analyst, stores=True)
            if drafted.round > held.revisions:
                return _outcome(
                    "stored_with_findings",
                    seen,
                    analyst,
                    stores=True,
                    reason=f"the review of round {drafted.round} did not clear the report, and its {held.revisions} "
                    "revisions are used",
                )
            seen.analyst_rounds += 1
            drafted = await analyst.revise(_revision_request(review))
    except Exception as exc:
        # A failure the loop reports as its `failed` ending: the client's own (a provider's, after its retries) or
        # a refused reply, which the engine cannot name by type. The tally, the caller's, keeps every turn sent.
        contain(exc, log=log, what="the report writing loop failed and ends failed", level="error")
        return _outcome("failed", seen, analyst, reason=f"{type(exc).__name__}: {exc}", failure=exc)


def _not_submitted(drafted: AnalystOutcome, seen: _Rounds, analyst: AnalystSession) -> WritingOutcome:
    """The loop's ending when an analyst's round ends with no document to review."""
    if drafted.ending == "over_budget":
        return _outcome("over_budget", seen, analyst, stores=bool(seen.reviewed), reason=drafted.reason)
    if drafted.ending == "refused":
        return _outcome("refused", seen, analyst, faults=drafted.faults, reason=drafted.reason)
    return _outcome("failed", seen, analyst, reason=drafted.reason)


def _not_reviewed(reviewed: EditorOutcome, seen: _Rounds, analyst: AnalystSession) -> WritingOutcome:
    """The loop's ending when a review round ends with no review."""
    if reviewed.ending == "over_budget":
        return _outcome("over_budget", seen, analyst, stores=bool(seen.reviewed), reason=reviewed.reason)
    if reviewed.ending == "review_failed":
        faults = "; ".join(reviewed.faults)
        return _outcome("review_failed", seen, analyst, reason=f"{reviewed.reason}: {faults}")
    return _outcome("failed", seen, analyst, reason=reviewed.reason, failure=reviewed.failure)


def _outcome(
    ending: WriterEnding,
    seen: _Rounds,
    analyst: AnalystSession,
    *,
    stores: bool = False,
    faults: tuple[SubmissionFault, ...] = (),
    reason: str = "",
    failure: Exception | None = None,
) -> WritingOutcome:
    """The outcome, storing (where the ending stores) the last document a review was read for, with its findings."""
    return WritingOutcome(
        ending=ending,
        doc=seen.reviewed[-1] if stores else None,
        findings=seen.reviews[-1].findings if stores else (),
        reviews=tuple(seen.reviews),
        analyst_rounds=seen.analyst_rounds,
        editor_rounds=seen.editor_rounds,
        faults=faults,
        reason=reason,
        sessions={ANALYST_SESSION: analyst.to_json(), **seen.editors},
        failure=failure,
    )


def _revision_request(review: Review) -> str:
    """What the analyst is sent to revise by: the review as the editor wrote it, its total and pass as code read it."""
    verdict = (
        "passes on its scores, and holds a must_fix finding"
        if review.passes
        else f"does not pass: a report passes with no item at 0 and a total of at least {PASS_TOTAL}"
    )
    scores = "\n".join(_scored(item) for item in review.items)
    findings = (
        "\n".join(
            f"- [{finding.severity}] {finding.claim or 'report'}: {one_line(finding.finding)}"
            for finding in review.findings
        )
        or "- none"
    )
    return (
        f"The report editor reviewed your report (round {review.round}). Its review totals {review.total} and "
        f"{verdict}.\n\nScores:\n{scores}\n\nFindings:\n{findings}\n\nRevise the report: resolve every must_fix "
        "finding and every item scored 0 or 1 you can, and say in the document what you weighed and kept."
    )


def _scored(item: RubricScore) -> str:
    return f"- {item.n}. {item.name}: {item.score} of 2. {one_line(item.reason)}"
