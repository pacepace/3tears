"""The writing loop's rounds: the analyst drafts, the editor reviews, the analyst revises, until a review clears it.

:func:`write_rounds` runs plan §4.2's steps 1 to 6 over one frozen snapshot:

1. The analyst drafts (:meth:`~threetears.evals.ops.report_writer.analyst.AnalystSession.draft`): its submission
   passes the document's validity and the invariants before anything else sees it; a fault goes back to it as a
   bounce, which costs no editor call.
2. The editor reviews the document in a session of its own
   (:func:`~threetears.evals.ops.report_writer.editor.review_report`), given the page, ``omitted``, the computed facts
   and its earlier findings, and nothing of the analyst's session.
3. A review that passes with no finding open
   (:attr:`~threetears.evals.ops.report_writer.editor.Review.settles`) ends the loop ``stored``. Otherwise the analyst
   revises in its own conversation, sent the review (each item's score and reason, and every finding, the blocking
   ones marked), and the editor reviews the revision. :attr:`WriterLimits.revisions` revisions at most (2, so at most
   3 reviews). After the last, a review that passes with no blocking finding open
   (:attr:`~threetears.evals.ops.report_writer.editor.Review.blocking`) ends the loop ``stored``, its findings listed
   open, and any other ends it ``stored_with_findings``. Every ``must_fix`` blocks, and every finding first raised in
   round 1; a ``should_fix`` first raised in a later review never does, new or carried (owner rulings 2026-10-29 and
   2026-10-31): it is given a revision while one remains, and alone it never makes the report a draft.

**Endings** (plan §4.4), each a value the caller stores or reports:

- ``stored``: a review cleared the document, which is stored finished, with any ``should_fix`` findings of its last
  review that block nothing listed open;
- ``stored_with_findings``: the revisions ran out; the last document is stored with its review's findings, a draft;
- ``over_budget``: the report's budget refused a turn, or the analyst called a tool after it was told to
  submit. The last document a review was read for is stored with that review's findings, a draft; with none,
  nothing;
- ``refused``: a submission still had faults after its round's bounces; nothing is stored, the faults are returned;
- ``review_failed``: the editor's answer could not be read twice; nothing is stored;
- ``failed``: the model refused, a reply stopped short, or the client failed (the provider, after its own retries);
  nothing is stored. The tally is the caller's and keeps every turn sent.

**The loop reads no environment and chooses nothing by kind.** The model, its sessions' clients, the mounts, the
skill texts, the checker and every limit come from the caller, and no condition here reads the document's story, a
claim, or what kind of eval the snapshot holds: what ends a round is the checks, the review's scores and its
structured findings.

**A revision a cleared review sends is a chance, not a gate.** Where the review that sent it cleared the report (it
passed with only findings open that block nothing), a revision that ends without a review of its own on the
budget, on its faults or on an unreadable review (``over_budget``, ``refused``, ``review_failed``) ends the loop
``stored`` instead, with that review's document and findings: what cleared is never lost, nor made a draft, by the
chance taken to improve it. A ``failed`` round still ends the loop ``failed``, its failure reaching the caller: a
client's failure, or a recorded run asked a turn its recording lacks, is never stored over.

**A stored document is finished or a draft, and code says which.** Owner rulings 2026-10-29: the loop revises until
no blocking finding is open, and a page is never served as finished while one is. Only ``stored`` stores a finished
document; every other ending that stores one stores a draft
(:attr:`~threetears.evals.kernel.report_doc.EvalReportDoc.draft`), which its page says on its first screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from threetears.evals.analysis.analyst.skill import SkillText
from threetears.evals.analysis.report.md_parts import one_line
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

__all__ = ["MOST_REVISIONS", "TurnModel", "WriterEnding", "WriterLimits", "WritingOutcome", "write_rounds"]

log = get_logger(__name__)

#: How the writing loop ended (the module docstring says when, and what is stored).
WriterEnding = Literal["stored", "stored_with_findings", "over_budget", "refused", "review_failed", "failed"]

#: How a revision a cleared review sent may end and still leave that review's report stored finished.
_CHANCE_ENDINGS: Final[frozenset[WriterEnding]] = frozenset({"over_budget", "refused", "review_failed"})

#: The most revisions a report is given (ANALYST §4): so at most three reviews.
MOST_REVISIONS: Final = 2


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
    revisions: int = MOST_REVISIONS

    def __post_init__(self) -> None:
        """Refuse a revision count outside 0 to 2.

        :raises ValueError: The count.
        """
        if isinstance(self.revisions, bool) or not isinstance(self.revisions, int):
            raise ValueError(f"revisions is a whole number, not {self.revisions!r}")
        if not 0 <= self.revisions <= MOST_REVISIONS:
            raise ValueError(f"a report is given 0 to {MOST_REVISIONS} revisions, not {self.revisions}")


@dataclass(frozen=True, slots=True, kw_only=True)
class WritingOutcome:
    """How the writing loop ended, and what it leaves to store.

    :ivar ending: How it ended (:data:`WriterEnding`).
    :ivar doc: The document to store; ``None`` for an ending that stores nothing.
    :ivar findings: The findings stored with it, open: the last review's of that document. For ``stored`` they are
        the ones that review raised without blocking (a new ``should_fix``), often none; empty for an ending that
        stores nothing.
    :ivar reviews: Every review read, in round order.
    :ivar analyst_rounds: The analyst's rounds begun: the draft, then each revision.
    :ivar editor_rounds: The review rounds begun.
    :ivar faults: The faults of a submission that ended the loop ``refused``; empty otherwise.
    :ivar reason: Why the loop ended other than ``stored``, in a sentence; for a ``stored`` the revision of a cleared
        review did not improve on, how that revision ended.
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
    :param report_input: Its report input, which the editor's page and its facts read.
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
                return _kept_if_cleared(_not_submitted(drafted, seen, analyst), seen, analyst)
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
                return _kept_if_cleared(_not_reviewed(reviewed, seen, analyst), seen, analyst)
            review = reviewed.review
            seen.reviews.append(review)
            seen.reviewed.append(doc)
            if review.settles:
                return _outcome("stored", seen, analyst, stores=True)
            if drafted.round > held.revisions:
                if review.clears:
                    return _outcome("stored", seen, analyst, stores=True)
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


def _kept_if_cleared(ended: WritingOutcome, seen: _Rounds, analyst: AnalystSession) -> WritingOutcome:
    """``ended``, or ``stored`` with the last reviewed document where the review that sent this revision cleared it
    and the revision ended on the budget, on its faults or on an unreadable review; a ``failed`` one is kept.

    Only a review that clears with findings open sends a revision it does not need (:attr:`Review.settles`), so a
    cleared last review means the round that ended short was that revision.
    """
    if ended.ending not in _CHANCE_ENDINGS or not seen.reviews or not seen.reviews[-1].clears:
        return ended
    cleared = seen.reviews[-1].round
    log.warning(
        "the revision after the review of round %d, which cleared the report, ended %s (%s); the cleared report is "
        "stored",
        cleared,
        ended.ending,
        ended.reason,
    )
    return _outcome(
        "stored",
        seen,
        analyst,
        stores=True,
        reason=f"the revision after the review of round {cleared}, which cleared the report, ended {ended.ending}: "
        f"{ended.reason}",
    )


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
    if not review.passes:
        verdict = f"does not pass: a report passes with no item at 0 and a total of at least {review.pass_total}"
    elif review.blocking:
        verdict = "passes on its scores, and holds findings that block it"
    else:
        verdict = "passes on its scores, and holds findings that do not block it"
    scores = "\n".join(_scored(item) for item in review.items)
    blocking = {finding.id for finding in review.blocking}
    findings = (
        "\n".join(
            f"- {finding.id} [{finding.severity}{', blocking' if finding.id in blocking else ''}] "
            f"{finding.claim or 'report'}: {one_line(finding.finding)}"
            for finding in review.findings
        )
        or "- none"
    )
    return (
        f"The report editor reviewed your report (round {review.round}). Its review totals {review.total} and "
        f"{verdict}.\n\nScores:\n{scores}\n\nFindings:\n{findings}\n\nRevise the report: resolve every finding, "
        "must_fix and should_fix alike, and every item scored 0 or 1 you can, and say in the document what you weighed "
        "and kept. A report stored with a blocking finding open is marked a draft."
    )


def _scored(item: RubricScore) -> str:
    return f"- {item.n}. {item.name}: {item.score} of 2. {one_line(item.reason)}"
