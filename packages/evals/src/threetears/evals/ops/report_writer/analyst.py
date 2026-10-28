"""The analyst's session: the brief, the investigation, the submission, and the bounces until it passes (plan §4.2).

:class:`AnalystSession` runs the analyst over one frozen snapshot through a
:class:`~threetears.evals.ops.report_writer.session.ToolSession`. Its system text is the analyst skill
(``load_skill().analyst(withhold=...)``); its first user message is the brief: the snapshot id, how to submit, the
``report-doc/1`` JSON schema, and the snapshot's ``campaign_overview`` as the tool answers it, held to the result limit. The analyst then calls
the tool as it needs, and a reply that calls no tool is its submission: one fenced JSON object.

**A submission is read in four steps, and the first that finds a fault sends it back** (a bounce, which costs no
editor call): the fenced JSON (exactly one fenced block, holding one JSON object); its shape
(:class:`~threetears.evals.kernel.report_doc.ReportDoc`); its validity on the snapshot
(:func:`~threetears.evals.kernel.report_doc.report_doc_validity`, over the snapshot's arm ids and the caveat ids code
assigns on it); and the invariants, by the checker the caller hands in (``analysis/report_doc_checks.py``'s, on the
snapshot's report input). A reply stopped at its output cap is a bounce too. Each bounce tells the analyst every fault
found, as text. A round allows :attr:`AnalystLimits.bounces` bounces; a fault after the last ends the round
``refused``, carrying the faults.

**Code reads nothing it would choose by.** Nothing here reads the document's story, a claim's weight, a block's kind
or what kind of eval the snapshot holds: the checks are the document's model, its validity and the invariants, each
the same for every report.

**Rounds.** :meth:`AnalystSession.draft` is round 1, with :attr:`AnalystLimits.draft_calls` tool calls; each
:meth:`AnalystSession.revise` (the editor's findings, sent back by the caller) is the next round, in the same
conversation, with :attr:`AnalystLimits.revision_calls`. A round ends ``submitted`` (the document passed every
check), ``refused``, ``over_budget`` (the report's dollar budget refused a turn, or the analyst called a tool after
its calls ran out and it was told to submit) or ``failed`` (the model refused, or its reply stopped short).

**Stored and resumed.** :meth:`AnalystSession.to_json` is the session as a value (the tool session's record, with
the snapshot and where the round's bounces stand); :meth:`AnalystSession.resume` takes it up again, so a later
question resumes the conversation that wrote the report.

The session reads no environment: the model, its client and every limit come from its caller.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from pydantic import ValidationError

from threetears.evals.analysis.analyst.skill import SkillText
from threetears.evals.kernel.report_doc import ReportDoc, report_doc_json_schema, report_doc_validity
from threetears.evals.ops.report_writer.session import SessionLimits, SessionStop, ToolMount, ToolSession
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.schema.turns import ToolTurnClient

#: The names the writing loop calls. ``InvariantVerdict`` joins them with its first production caller outside this
#: module (``tests/test_report_writer_exports.py`` holds the list).
__all__ = [
    "ANALYST_SESSION",
    "AnalystLimits",
    "AnalystOutcome",
    "AnalystSession",
    "InvariantCheck",
    "SubmissionFault",
]

#: The analyst session's name, which its tally entries and its recorded turns are kept under.
ANALYST_SESSION: Final = "analyst"

#: The storage format :meth:`AnalystSession.to_json` writes and :meth:`AnalystSession.resume` reads.
_FORMAT: Final = 1

#: A fenced block: three backticks, an optional language word, the body, three backticks.
_FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n(?P<body>.*?)\n?```", re.DOTALL)

#: The most faults of a shape a bounce lists one by one.
_LISTED: Final = 30


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalystLimits:
    """What the analyst's rounds are held to (plan §4.3).

    :ivar draft_calls: The tool calls the first draft may make.
    :ivar revision_calls: The tool calls each revision may make.
    :ivar bounces: The submissions a round may send back before a fault ends it ``refused``.
    :ivar session: What every turn is held to.
    """

    draft_calls: int = 100
    revision_calls: int = 30
    bounces: int = 3
    session: SessionLimits = field(default_factory=SessionLimits)

    def __post_init__(self) -> None:
        """Refuse a call budget below 1 or a negative bounce count.

        :raises ValueError: The limit, named.
        """
        for name, floor in (("draft_calls", 1), ("revision_calls", 1), ("bounces", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(f"the analyst's {name} is a whole number of at least {floor}, not {value!r}")


class InvariantVerdict(Protocol):
    """One fault the invariants find in a valid document: where, what, and how to fix it."""

    @property
    def path(self) -> str:
        """The field it is in, as a path (``claims[2].evidence[0]``)."""
        ...

    @property
    def fault(self) -> str:
        """What is wrong."""
        ...

    @property
    def remedy(self) -> str:
        """How to fix it."""
        ...


#: The invariants' checker, on the snapshot the session reads: the faults it finds in a valid document, none for one
#: that passes.
InvariantCheck = Callable[[ReportDoc], Sequence[InvariantVerdict]]


@dataclass(frozen=True, slots=True)
class SubmissionFault:
    """One fault a submission is sent back for.

    :ivar kind: Which check found it: ``unparsed`` (no single fenced JSON object), ``shape`` (the document's model),
        ``validity`` (what the snapshot holds), ``invariant`` (the invariants), or ``max_tokens`` (the reply was cut).
    :ivar path: The field it is in, or ``""`` for the submission as a whole.
    :ivar fault: What is wrong, and how to fix it where the check says.
    """

    kind: Literal["unparsed", "shape", "validity", "invariant", "max_tokens"]
    path: str
    fault: str

    def line(self) -> str:
        """The fault as one line of a bounce.

        :return: ``- [kind] path: fault``, or ``- [kind] fault`` with no path.
        :rtype: str
        """
        return f"- [{self.kind}] {self.path}: {self.fault}" if self.path else f"- [{self.kind}] {self.fault}"


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalystOutcome:
    """How one round of the analyst's session ended.

    :ivar ending: ``submitted``, ``refused``, ``over_budget`` or ``failed`` (the module docstring says when).
    :ivar round: The round it ended.
    :ivar doc: The document that passed every check; ``None`` unless ``submitted``.
    :ivar faults: The faults of the submission that ended the round ``refused``; empty otherwise.
    :ivar reason: Why a round ended other than ``submitted``, in a sentence.
    """

    ending: Literal["submitted", "refused", "over_budget", "failed"]
    round: int
    doc: ReportDoc | None = None
    faults: tuple[SubmissionFault, ...] = ()
    reason: str = ""


class AnalystSession:
    """The analyst over one frozen snapshot: its rounds of investigation and submission, held to their limits.

    Build one with :meth:`start` and run :meth:`draft`, then :meth:`revise` for each revision; or take a stored one
    up again with :meth:`resume`.

    :ivar snapshot: The snapshot it reads, by id.
    :ivar limits: What its rounds are held to.
    """

    def __init__(
        self,
        session: ToolSession,
        *,
        snapshot: str,
        arms: Collection[str],
        caveats: Collection[str],
        check: InvariantCheck,
        limits: AnalystLimits,
        bounces: int = 0,
    ) -> None:
        """Wrap a tool session (:meth:`start` and :meth:`resume` build it).

        :param session: The tool session the analyst's turns go through.
        :ptype session: ToolSession
        :param snapshot: The snapshot it reads, by id.
        :ptype snapshot: str
        :param arms: The snapshot's arm ids, which validity reads.
        :ptype arms: Collection[str]
        :param caveats: The caveat ids code assigns on the snapshot, stated and implied, which validity reads.
        :ptype caveats: Collection[str]
        :param check: The invariants' checker on the snapshot.
        :ptype check: InvariantCheck
        :param limits: What its rounds are held to.
        :ptype limits: AnalystLimits
        :param bounces: The bounces the current round has sent.
        :ptype bounces: int
        :raises ValueError: An empty snapshot id.
        """
        if not snapshot:
            raise ValueError("an analyst session reads one snapshot, named by its id")
        self.snapshot = snapshot
        self.limits = limits
        self._session = session
        self._arms = frozenset(arms)
        self._caveats = frozenset(caveats)
        self._check = check
        self._bounces = bounces

    @classmethod
    def start(
        cls,
        *,
        model: str,
        client: ToolTurnClient,
        mount: ToolMount,
        skill: SkillText,
        snapshot: str,
        arms: Collection[str],
        caveats: Collection[str],
        check: InvariantCheck,
        tally: Tally,
        budget: ReportBudget,
        limits: AnalystLimits | None = None,
    ) -> AnalystSession:
        """A new analyst session, before its draft.

        :param model: The model id the client sends to (``Model.model_name``), which the stored session keeps.
        :ptype model: str
        :param client: The client its turns go through (``Model.turn_session(ANALYST_SESSION, snapshot=...)``).
        :ptype client: ToolTurnClient
        :param mount: The ``evals_analyst`` tool on the analyst's cut, bound to its host and caller.
        :ptype mount: ToolMount
        :param skill: The analyst's system text (``load_skill().analyst(withhold=...)``).
        :ptype skill: SkillText
        :param snapshot: The frozen snapshot it reads, by id.
        :ptype snapshot: str
        :param arms: The snapshot's arm ids.
        :ptype arms: Collection[str]
        :param caveats: The caveat ids code assigns on the snapshot, stated and implied.
        :ptype caveats: Collection[str]
        :param check: The invariants' checker on the snapshot.
        :ptype check: InvariantCheck
        :param tally: The report's tally.
        :ptype tally: Tally
        :param budget: The report's dollar budget.
        :ptype budget: ReportBudget
        :param limits: What its rounds are held to; plan §4.3's defaults when ``None``.
        :ptype limits: AnalystLimits | None
        :return: The session.
        :rtype: AnalystSession
        """
        held = limits if limits is not None else AnalystLimits()
        session = ToolSession(
            name=ANALYST_SESSION,
            role="analyst",
            model=model,
            client=client,
            mount=mount,
            system=skill,
            tally=tally,
            budget=budget,
            limits=held.session,
        )
        return cls(session, snapshot=snapshot, arms=arms, caveats=caveats, check=check, limits=held)

    @property
    def session(self) -> ToolSession:
        """The tool session the analyst's turns go through."""
        return self._session

    async def draft(self) -> AnalystOutcome:
        """Round 1: send the brief with the snapshot's overview, and run until a submission passes or the round ends.

        :return: How the round ended.
        :rtype: AnalystOutcome
        :raises ValueError: The session has drafted already, or the tool refused the snapshot's overview.
        :raises TurnReplyRefused: The client refused a reply it was paid for; the turn is tallied first.
        """
        if self._session.round:
            raise ValueError("the analyst has drafted already; send the editor's findings with revise()")
        overview = await self._session.mount.call({"action": "campaign_overview", "snapshot_id": self.snapshot})
        if overview.is_error:
            raise ValueError(
                f"the tool refused snapshot {self.snapshot!r}'s overview, so the analyst cannot be briefed: "
                f"{overview.text}"
            )
        self._open(1, self.limits.draft_calls)
        return await self._run(self._brief(self._session.capped(overview.text)))

    async def revise(self, request: str) -> AnalystOutcome:
        """The next round: send the caller's request (the editor's findings) in the same conversation.

        :param request: What the analyst is asked to revise, as the caller words it.
        :ptype request: str
        :return: How the round ended.
        :rtype: AnalystOutcome
        :raises ValueError: The session has not drafted yet, or an empty request.
        :raises TurnReplyRefused: The client refused a reply it was paid for; the turn is tallied first.
        """
        if not self._session.round:
            raise ValueError("the analyst revises a draft: run draft() first")
        if not request.strip():
            raise ValueError("a revision is asked for with what to revise; the request is empty")
        self._open(self._session.round + 1, self.limits.revision_calls)
        return await self._run(
            f"{request.strip()}\n\nThis revision has {self.limits.revision_calls} tool calls. Submit the whole "
            "revised document again: one fenced JSON object, in a message that calls no tool."
        )

    def _open(self, round: int, calls: int) -> None:
        self._session.open_round(round, calls=calls)
        self._bounces = 0

    def _brief(self, overview: str) -> str:
        schema = json.dumps(report_doc_json_schema(), ensure_ascii=False, sort_keys=True)
        return (
            f"Write the report on snapshot {self.snapshot}: every call you make names it as snapshot_id.\n\n"
            "Submit the report as one report-doc/1 document: a single fenced JSON object (```json ... ```) in a "
            "message that calls no tool. That message is your submission. Code checks it (the JSON, its shape, the "
            "ids it names, then the invariants) and sends it back with every fault until it passes, at most "
            f"{self.limits.bounces} times; then it goes to the editor. This draft has {self.limits.draft_calls} tool "
            f"calls, and you are warned when {self.limits.session.warn_before} are left.\n\n"
            f"The report-doc/1 JSON schema:\n```json\n{schema}\n```\n\n"
            f"The snapshot's campaign_overview, as the tool answers it:\n\n{overview}"
        )

    async def _run(self, user: str) -> AnalystOutcome:
        round = self._session.round
        stop = await self._session.send(user)
        while True:
            if stop.kind in ("over_budget", "failed"):
                return AnalystOutcome(ending=_ending(stop), round=round, reason=stop.reason)
            doc, faults = self._read(stop)
            if doc is not None:
                return AnalystOutcome(ending="submitted", round=round, doc=doc)
            if self._bounces >= self.limits.bounces:
                return AnalystOutcome(
                    ending="refused",
                    round=round,
                    faults=faults,
                    reason=f"the submission still had faults after the round's {self.limits.bounces} bounces",
                )
            self._bounces += 1
            self._session.tally.record_bounce("analyst", round, faults[0].kind)
            stop = await self._session.send(self._bounce(faults))

    def _read(self, stop: SessionStop) -> tuple[ReportDoc | None, tuple[SubmissionFault, ...]]:
        """The submission's document, or the faults the first failing check found in it."""
        if stop.kind == "cut":
            return None, (
                SubmissionFault(
                    "max_tokens",
                    "",
                    f"your reply reached the {self.limits.session.max_tokens}-token output cap before it ended, and "
                    "was not kept: reply again, keeping the thinking shorter, or submitting before the cap",
                ),
            )
        text = stop.reply.text if stop.reply is not None else ""
        found = _read_submission(text)
        if not isinstance(found, ReportDoc):
            return None, found
        validity = tuple(
            SubmissionFault("validity", path, why)
            for path, why in report_doc_validity(found, arms=self._arms, caveats=self._caveats)
        )
        if validity:
            return None, validity
        verdicts = tuple(
            SubmissionFault("invariant", verdict.path, f"{verdict.fault}; {verdict.remedy}")
            for verdict in self._check(found)
        )
        if verdicts:
            return None, verdicts
        return found, ()

    def _bounce(self, faults: Sequence[SubmissionFault]) -> str:
        lines = "\n".join(fault.line() for fault in faults)
        return (
            f"Your submission is sent back (bounce {self._bounces} of {self.limits.bounces}). Fix every fault "
            "below, then submit the whole document again: one fenced JSON object, in a message that calls no "
            f"tool.\n\n{lines}"
        )

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The session as a stored value, which :meth:`resume` takes up again.

        :return: ``{"format", "snapshot", "bounces", "limits", "session"}``; ``session`` is the tool session's record
            (:meth:`ToolSession.to_json`).
        :rtype: dict[str, Any]
        """
        return {
            "format": _FORMAT,
            "snapshot": self.snapshot,
            "bounces": self._bounces,
            "limits": {
                "draft_calls": self.limits.draft_calls,
                "revision_calls": self.limits.revision_calls,
                "bounces": self.limits.bounces,
            },
            "session": self._session.to_json(),
        }

    @classmethod
    async def resume(
        cls,
        record: Mapping[str, Any],  # Any: a JSON object
        *,
        model: str,
        client: ToolTurnClient,
        mount: ToolMount,
        skill: SkillText,
        arms: Collection[str],
        caveats: Collection[str],
        check: InvariantCheck,
        tally: Tally,
        budget: ReportBudget,
    ) -> AnalystSession:
        """Take a stored analyst session up again, ready for its next round.

        :param record: What :meth:`to_json` wrote.
        :ptype record: Mapping[str, Any]
        :param model: The model id the client sends to, which must be the one the session ran on.
        :ptype model: str
        :param client: The client its next turns go through.
        :ptype client: ToolTurnClient
        :param mount: The tool, which re-derives each result body the record dropped.
        :ptype mount: ToolMount
        :param skill: The analyst's system text, the one the session started with.
        :ptype skill: SkillText
        :param arms: The snapshot's arm ids.
        :ptype arms: Collection[str]
        :param caveats: The caveat ids code assigns on the snapshot.
        :ptype caveats: Collection[str]
        :param check: The invariants' checker on the snapshot.
        :ptype check: InvariantCheck
        :param tally: The tally its next turns are counted in.
        :ptype tally: Tally
        :param budget: The budget that admits its next turns.
        :ptype budget: ReportBudget
        :return: The session.
        :rtype: AnalystSession
        :raises ValueError: A record of another format or out of shape, or one :meth:`ToolSession.from_json` refuses (a
            model, skill text or tool other than the stored one included).
        """
        if not isinstance(record, Mapping) or record.get("format") != _FORMAT:
            found = record.get("format") if isinstance(record, Mapping) else type(record).__name__
            raise ValueError(f"a stored analyst session of format {found!r}; this build reads format {_FORMAT}")
        try:
            stored = record["session"]
            snapshot, bounces = record["snapshot"], record["bounces"]
            held = AnalystLimits(**record["limits"], session=SessionLimits(**stored["limits"]))
            if isinstance(bounces, bool) or not isinstance(bounces, int) or not 0 <= bounces <= held.bounces:
                raise TypeError(f"bounces is {bounces!r}, not a count within the round's {held.bounces}")
            if not isinstance(snapshot, str):
                raise TypeError(f"snapshot is {snapshot!r}, not an id")
        except (KeyError, TypeError) as missing:
            raise ValueError(
                f"a stored analyst session out of shape: {missing!r} is missing or of the wrong type"
            ) from missing
        session = await ToolSession.from_json(
            stored, model=model, client=client, mount=mount, system=skill, tally=tally, budget=budget
        )
        return cls(session, snapshot=snapshot, arms=arms, caveats=caveats, check=check, limits=held, bounces=bounces)


def _read_submission(text: str) -> ReportDoc | tuple[SubmissionFault, ...]:
    """A submission's document: the one fenced JSON object in ``text``, read as ``report-doc/1``.

    :param text: The reply's text.
    :ptype text: str
    :return: The document, or what keeps it from being read: no fenced block, more than one, a block that is not a
        JSON object (``unparsed``), or one the document's model refuses (``shape``, a fault per field).
    :rtype: ReportDoc | tuple[SubmissionFault, ...]
    """
    blocks = [match["body"] for match in _FENCE.finditer(text)]
    if len(blocks) != 1:
        found = "no fenced block" if not blocks else f"{len(blocks)} fenced blocks"
        return (SubmissionFault("unparsed", "", f"the submission holds {found}; submit exactly one, the document"),)
    try:
        value = json.loads(blocks[0])
    except json.JSONDecodeError as unread:
        where = f"line {unread.lineno}, column {unread.colno}"
        return (SubmissionFault("unparsed", "", f"the fenced block is not JSON ({unread.msg} at {where})"),)
    if not isinstance(value, dict):
        return (SubmissionFault("unparsed", "", "the fenced block is JSON but not an object; submit the document"),)
    try:
        return ReportDoc.model_validate(value)
    except ValidationError as invalid:
        errors = invalid.errors()
        faults = [
            SubmissionFault("shape", ".".join(str(part) for part in error["loc"]), error["msg"])
            for error in errors[:_LISTED]
        ]
        if len(errors) > _LISTED:
            faults.append(SubmissionFault("shape", "", f"and {len(errors) - _LISTED} more faults of shape"))
        return tuple(faults)


def _ending(stop: SessionStop) -> Literal["over_budget", "failed"]:
    return "over_budget" if stop.kind == "over_budget" else "failed"
