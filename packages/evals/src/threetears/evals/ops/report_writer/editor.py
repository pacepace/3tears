"""The report editor: one review of a report document, in a session of its own (plan §4.2 step 4).

Each review round is a fresh :class:`~threetears.evals.ops.report_writer.session.ToolSession`: its system text is
the skill's ``editor.md`` (``load_skill().editor()``), its tool the ``evals_analyst`` tool on the editor's cut, and its
one user message holds what :func:`review_report` builds from the document and the snapshot's report input alone:

- the page as a reader sees it, every block with its values table
  (:func:`~threetears.evals.analysis.report.doc_page.text.doc_page_text` over a strict
  :func:`~threetears.evals.analysis.report.doc_page.build.build`), a draft's page: no made line, which is written
  only for a stored page, so it is the page the analyst's ``report_check`` shows, line for line;
- the document's ``omitted``, what the analyst saw and left out, with why;
- the computed facts code finds for the editor to weigh
  (:func:`~threetears.evals.analysis.report_doc_facts.editor_facts`), labelled as facts and never as verdicts;
- from the second round, the editor's own earlier findings.

**Nothing of the analyst's session reaches the editor.** :func:`review_report` takes the document, the snapshot's
report input and the earlier findings, and no message, thinking block or tool result of the analyst's: its signature
has no parameter that could carry one, so the editor judges the page as a reader would, not the reasoning behind it.

**Code scores the review.** The editor's answer is its last fenced JSON block (prose may come before it): seven
rubric items, each keyed by ``n`` and named as the rubric names it, scored 0, 1 or 2 with a reason; and findings,
each ``{claim, severity, finding}``. :class:`Review` adds the scores up and says whether the review passes (no item
scores 0 and the total is at least :data:`PASS_TOTAL`); a total or a verdict the editor writes is never read. An
answer that cannot be read is asked for once more, saying what was wrong; a second ends the round ``review_failed``.

**A session is kept.** Every round's session comes back on its outcome as a stored value
(``ToolSession.to_json()``), whatever the ending, a client's failure included, so a later question can resume it.

**Budgets.** A review has :attr:`EditorLimits.calls` tool calls, held by the session before anything is sent, and
every turn is admitted by the report's budget; a turn it refuses ends the round ``over_budget``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

from threetears.evals.analysis.analyst.skill import SkillText
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report.md_parts import one_line
from threetears.evals.analysis.report_doc_facts import EditorFact, editor_facts
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc, ReportFinding
from threetears.evals.ops.report_writer.session import SessionLimits, SessionStop, ToolMount, ToolSession
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.schema.turns import ToolTurnClient

__all__ = ["EditorLimits", "EditorOutcome", "PASS_TOTAL", "Review", "RubricScore", "editor_session", "review_report"]

log = get_logger(__name__)

#: The rubric's items, in the order ``editor.md`` numbers them: item ``n`` is ``_RUBRIC[n - 1]``.
_RUBRIC: Final = (
    "First screen",
    "Coherent story",
    "Honest emphasis",
    "Sound elision",
    "Confounds handled",
    "Actionable next steps",
    "Evidence fits claims",
)

#: The highest score an item takes.
_TOP_SCORE: Final = 2

#: The least total a review passes with, as ``editor.md`` states it, when no item scores 0.
PASS_TOTAL: Final = 11

#: What a finding about the page as a whole names in place of a claim id.
_REPORT: Final = "report"

#: A fenced block: three backticks, an optional language word, the body, three backticks.
_FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n(?P<body>.*?)\n?```", re.DOTALL)

#: How many answers a round reads before it ends ``review_failed``: the first, and the one asked for again.
_ANSWERS: Final = 2


def editor_session(round: int) -> str:
    """The name of the editor's session for a round, which its tally entries and recorded turns are kept under.

    :param round: The round reviewed.
    :ptype round: int
    :return: ``editor-<round>``.
    :rtype: str
    """
    return f"editor-{round}"


@dataclass(frozen=True, slots=True, kw_only=True)
class EditorLimits:
    """What each review round is held to (plan §4.3).

    :ivar calls: The tool calls a review may make.
    :ivar session: What every turn is held to.
    """

    calls: int = 40
    session: SessionLimits = field(default_factory=SessionLimits)

    def __post_init__(self) -> None:
        """Refuse a call budget below 1.

        :raises ValueError: The budget, named.
        """
        if isinstance(self.calls, bool) or not isinstance(self.calls, int) or self.calls < 1:
            raise ValueError(f"the editor's calls is a whole number of at least 1, not {self.calls!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class RubricScore:
    """One rubric item as the editor scored it.

    :ivar n: The item's number, 1 to 7.
    :ivar name: Its name, as the rubric names it.
    :ivar score: 0, 1 or 2.
    :ivar reason: Why, in the editor's words.
    """

    n: int
    name: str
    score: int
    reason: str

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The item as JSON.

        :return: ``{"n", "name", "score", "reason"}``.
        :rtype: dict[str, Any]
        """
        return {"n": self.n, "name": self.name, "score": self.score, "reason": self.reason}


@dataclass(frozen=True, slots=True, kw_only=True)
class Review:
    """One review of a report document: the seven items as scored, and the findings; code adds the scores up.

    :ivar round: The round it reviewed.
    :ivar items: The rubric items, in rubric order.
    :ivar findings: What the editor found for the analyst to change, in its order.
    """

    round: int
    items: tuple[RubricScore, ...]
    findings: tuple[ReportFinding, ...]

    @property
    def total(self) -> int:
        """The items' scores added up, by code."""
        return sum(item.score for item in self.items)

    @property
    def passes(self) -> bool:
        """Whether the scores pass: no item scores 0 and the total is at least :data:`PASS_TOTAL`."""
        return all(item.score > 0 for item in self.items) and self.total >= PASS_TOTAL

    @property
    def must_fix(self) -> bool:
        """Whether any finding is a ``must_fix``, which sends the report back however it scored."""
        return any(finding.severity == "must_fix" for finding in self.findings)

    @property
    def clears(self) -> bool:
        """Whether the report may be stored as written: the scores pass and no finding is a ``must_fix``."""
        return self.passes and not self.must_fix

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The review as the stored record keeps it, the total and the pass as code computed them.

        :return: ``{"round", "items", "findings", "total", "passes", "must_fix"}``.
        :rtype: dict[str, Any]
        """
        return {
            "round": self.round,
            "items": [item.to_json() for item in self.items],
            "findings": [finding.model_dump(mode="json") for finding in self.findings],
            "total": self.total,
            "passes": self.passes,
            "must_fix": self.must_fix,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class EditorOutcome:
    """How one review round ended.

    :ivar ending: ``reviewed``, a review was read; ``review_failed``, the answer could not be read twice;
        ``over_budget``, the report's budget refused a turn; ``failed``, the model refused, its reply stopped
        short for a cause other than the output cap, or the client failed (the provider's failure, after its
        retries, or a reply the client refused after it was paid for).
    :ivar round: The round reviewed.
    :ivar review: The review; ``None`` unless ``reviewed``.
    :ivar faults: What kept the last answer from being read; empty unless ``review_failed``.
    :ivar reason: Why a round ended other than ``reviewed``, in a sentence.
    :ivar session: The round's session as a stored value (``ToolSession.to_json()``), whatever the ending: a session
        is resumable, never thrown away.
    """

    ending: Literal["reviewed", "review_failed", "over_budget", "failed"]
    round: int
    review: Review | None = None
    faults: tuple[str, ...] = ()
    reason: str = ""
    session: dict[str, Any] = field(default_factory=dict)  # Any: a JSON object


async def review_report(
    doc: ReportDoc,
    report_input: ReportInput,
    *,
    round: int,
    snapshot: str,
    earlier: Sequence[ReportFinding],
    model: str,
    client: ToolTurnClient,
    mount: ToolMount,
    skill: SkillText,
    tally: Tally,
    budget: ReportBudget,
    limits: EditorLimits | None = None,
) -> EditorOutcome:
    """Review a report document in a fresh editor session: the page, ``omitted``, the facts and earlier findings.

    The parameters are everything the editor is given; none carries the analyst's session (the module docstring).

    :param doc: The document, which passed validity and the invariants.
    :ptype doc: ReportDoc
    :param report_input: The report input of the snapshot it was written from.
    :ptype report_input: ReportInput
    :param round: The round reviewed: the analyst's round that submitted the document.
    :ptype round: int
    :param snapshot: The snapshot's id, which every tool call names.
    :ptype snapshot: str
    :param earlier: The previous round's findings; empty in round 1.
    :ptype earlier: Sequence[ReportFinding]
    :param model: The model id the client sends to.
    :ptype model: str
    :param client: The client this round's session goes through (``turn_session(editor_session(round), ...)``).
    :ptype client: ToolTurnClient
    :param mount: The ``evals_analyst`` tool on the editor's cut, bound to its host and caller.
    :ptype mount: ToolMount
    :param skill: The editor's system text (``load_skill().editor()``).
    :ptype skill: SkillText
    :param tally: The report's tally.
    :ptype tally: Tally
    :param budget: The report's budget (dollars, or tokens and turns).
    :ptype budget: ReportBudget
    :param limits: What the round is held to; plan §4.3's defaults when ``None``.
    :ptype limits: EditorLimits | None
    :return: How the round ended, with its session as a stored value. A failure of the client is the ``failed``
        ending, contained and logged as an error, so the session that led to it is kept; every turn it paid for is in
        the tally.
    :rtype: EditorOutcome
    """
    held = limits if limits is not None else EditorLimits()
    session = ToolSession(
        name=editor_session(round),
        role="editor",
        model=model,
        client=client,
        mount=mount,
        system=skill,
        tally=tally,
        budget=budget,
        limits=held.session,
    )
    session.open_round(round, calls=held.calls)
    try:
        ended = await _reviewed(
            session, doc, report_input, round=round, snapshot=snapshot, earlier=earlier, limits=held
        )
    except Exception as exc:
        # The client's failure (a provider's, after its retries; a reply refused after it was paid for), which the
        # engine cannot name by type: reported as this round's `failed` ending, so the session is kept beside it.
        contain(exc, log=log, what=f"the report editor's round {round} failed and ends failed", level="error")
        ended = EditorOutcome(ending="failed", round=round, reason=f"{type(exc).__name__}: {exc}")
    return replace(ended, session=session.to_json())


async def _reviewed(
    session: ToolSession,
    doc: ReportDoc,
    report_input: ReportInput,
    *,
    round: int,
    snapshot: str,
    earlier: Sequence[ReportFinding],
    limits: EditorLimits,
) -> EditorOutcome:
    """The round run in its session: the request, the answer, and the one answer asked for again."""
    claims = frozenset(claim.id for claim in doc.claims)
    stop = await session.send(_request(doc, report_input, snapshot=snapshot, earlier=earlier, calls=limits.calls))
    faults: tuple[str, ...] = ()
    for answer in range(1, _ANSWERS + 1):
        if stop.kind == "over_budget" and stop.reply is None:
            return EditorOutcome(ending="over_budget", round=round, reason=stop.reason)
        if stop.kind == "failed":
            return EditorOutcome(ending="failed", round=round, reason=stop.reason)
        read = _answer(stop, claims, round, limits)
        if isinstance(read, Review):
            return EditorOutcome(ending="reviewed", round=round, review=read)
        faults = read
        if answer < _ANSWERS:
            stop = await session.send(_ask_again(faults))
    return EditorOutcome(
        ending="review_failed",
        round=round,
        faults=faults,
        reason=f"the editor's answer could not be read after it was asked {_ANSWERS} times",
    )


def _request(
    doc: ReportDoc, report_input: ReportInput, *, snapshot: str, earlier: Sequence[ReportFinding], calls: int
) -> str:
    """The editor's one user message: built from the document and the snapshot's input alone."""
    # A draft's page, as the analyst's report_check shows it: no made line (a draft has no generation record until it
    # is stored) and no findings in Method (the earlier ones are listed below, to check against the page).
    page = doc_page_text(build(doc, report_input, strict=True, findings=(), made=""))
    parts = [
        f"Review the report written from snapshot {snapshot}: every call you make names it as snapshot_id. This "
        f"review has {calls} tool calls.",
        f"The page, as a reader sees it:\n\n{page}",
        "What the analyst left out, and why (omitted):\n" + _omitted(doc),
        "Computed facts, for you to weigh. Code computed each from the snapshot; none is a verdict:\n"
        + _facts(editor_facts(doc, report_input)),
    ]
    if earlier:
        parts.append("Your findings from the previous round, to check against the revised page:\n" + _listed(earlier))
    parts.append(
        "Your final message, with no tool call in it, is the review: one fenced JSON object, as editor.md says."
    )
    return "\n\n".join(parts)


def _omitted(doc: ReportDoc) -> str:
    if not doc.omitted:
        return "- nothing"
    return "\n".join(f"- {one_line(item.what)}: {one_line(item.why)}" for item in doc.omitted)


def _facts(facts: Sequence[EditorFact]) -> str:
    if not facts:
        return "- none"
    return "\n".join(
        f"- [{fact.kind}] {fact.path or _REPORT}: {one_line(fact.fact)}"
        + (f" (keys: {', '.join(fact.figures)})" if fact.figures else "")
        for fact in facts
    )


def _listed(findings: Sequence[ReportFinding]) -> str:
    return "\n".join(
        f"- [{finding.severity}] {finding.claim or _REPORT}: {one_line(finding.finding)}" for finding in findings
    )


def _ask_again(faults: Sequence[str]) -> str:
    lines = "\n".join(f"- {one_line(fault)}" for fault in faults)
    return (
        f"Your review could not be read:\n{lines}\n\nAnswer again with the review: one fenced JSON object, as "
        "editor.md says, in a message that calls no tool."
    )


def _answer(stop: SessionStop, claims: frozenset[str], round: int, limits: EditorLimits) -> Review | tuple[str, ...]:
    """The review a stop carries, or what kept it from being read."""
    if stop.kind == "cut":
        return (
            f"your answer reached the {limits.session.max_tokens}-token output cap before it ended; keep the "
            "thinking shorter",
        )
    if stop.kind == "over_budget":
        return (f"this review's {limits.calls} tool calls are used up, and you called a tool again",)
    return _read_review(stop.reply.text if stop.reply is not None else "", claims=claims, round=round)


def _read_review(text: str, *, claims: frozenset[str], round: int) -> Review | tuple[str, ...]:
    """The review an editor's answer holds: its last fenced JSON block, read against the rubric.

    :param text: The answer's text.
    :ptype text: str
    :param claims: The claim ids of the document reviewed, which a finding may name.
    :ptype claims: frozenset[str]
    :param round: The round reviewed.
    :ptype round: int
    :return: The review, or every fault that keeps it from being read.
    :rtype: Review | tuple[str, ...]
    """
    blocks = [match["body"] for match in _FENCE.finditer(text)]
    if not blocks:
        return ("the answer holds no fenced block; end it with the review as one fenced JSON object",)
    try:
        value = json.loads(blocks[-1])
    except json.JSONDecodeError as unread:
        return (f"the last fenced block is not JSON ({unread.msg} at line {unread.lineno}, column {unread.colno})",)
    if not isinstance(value, dict):
        return ("the last fenced block is JSON but not an object; answer {items, findings}",)
    faults: list[str] = []
    items = _items(value.get("items"), faults)
    findings = _findings(value.get("findings"), claims, faults)
    if faults:
        return tuple(faults)
    return Review(round=round, items=items, findings=findings)


def _items(value: object, faults: list[str]) -> tuple[RubricScore, ...]:
    if not isinstance(value, list):
        faults.append(f"items is not a list; give all {len(_RUBRIC)} rubric items")
        return ()
    scored: dict[int, RubricScore] = {}
    for index, item in enumerate(value):
        read = _item(item, index, faults)
        if read is None:
            continue
        if read.n in scored:
            faults.append(f"items[{index}]: item {read.n} is scored twice")
            continue
        scored[read.n] = read
    missing = [str(n) for n in range(1, len(_RUBRIC) + 1) if n not in scored]
    if missing:
        faults.append(f"items has no readable item {', '.join(missing)}: score every rubric item")
    return tuple(scored[n] for n in sorted(scored))


def _item(item: object, index: int, faults: list[str]) -> RubricScore | None:
    where = f"items[{index}]"
    if not isinstance(item, Mapping):
        faults.append(f"{where} is not an object {{n, name, score, reason}}")
        return None
    n, name, score, reason = item.get("n"), item.get("name"), item.get("score"), item.get("reason")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(_RUBRIC):
        faults.append(f"{where}.n is {n!r}, not a rubric item's number (1 to {len(_RUBRIC)})")
        return None
    fault_count = len(faults)
    expected = _RUBRIC[n - 1]
    if not isinstance(name, str) or name.strip().casefold() != expected.casefold():
        faults.append(f"{where}.name is {name!r}; item {n} is named {expected!r}")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= _TOP_SCORE:
        faults.append(f"{where}.score is {score!r}, not 0, 1 or 2")
    if not isinstance(reason, str) or not reason.strip():
        faults.append(f"{where}.reason is empty; give the reason for the score")
    if len(faults) > fault_count:
        return None
    assert isinstance(score, int) and isinstance(reason, str)
    return RubricScore(n=n, name=expected, score=score, reason=reason.strip())


def _findings(value: object, claims: frozenset[str], faults: list[str]) -> tuple[ReportFinding, ...]:
    if not isinstance(value, list):
        faults.append("findings is not a list; give [] for none")
        return ()
    read: list[ReportFinding] = []
    for index, finding in enumerate(value):
        where = f"findings[{index}]"
        if not isinstance(finding, Mapping):
            faults.append(f"{where} is not an object {{claim, severity, finding}}")
            continue
        claim, severity, said = finding.get("claim"), finding.get("severity"), finding.get("finding")
        fault_count = len(faults)
        if not isinstance(claim, str) or (claim != _REPORT and claim not in claims):
            faults.append(
                f"{where}.claim is {claim!r}: name a claim of the report by its id, or {_REPORT!r} for the page as a "
                "whole"
            )
        if severity not in ("must_fix", "should_fix"):
            faults.append(f"{where}.severity is {severity!r}, not must_fix or should_fix")
        if not isinstance(said, str) or not said.strip():
            faults.append(f"{where}.finding is empty; say what to change, where, and what would fix it")
        if len(faults) > fault_count:
            continue
        assert isinstance(claim, str) and isinstance(said, str)
        read.append(
            ReportFinding.model_validate(
                {"claim": "" if claim == _REPORT else claim, "severity": severity, "finding": said.strip()}
            )
        )
    return tuple(read)
