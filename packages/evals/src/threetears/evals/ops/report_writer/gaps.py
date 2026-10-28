"""The gaps a report write noted: each ``capability_gap`` call its sessions made, and code's two notices.

A write's sessions call ``capability_gap`` through the report writer's own mount, which stores nothing
(``analyst_actions(gaps_kept_by="write")``), so the write stores them itself when it ends, whatever the ending, from
what it already holds (:func:`write_gaps`):

- **each call** a session made whose result was not a refusal, read off the session's stored value (the tool call's
  arguments exactly as the model sent them, its result's error flag), with the session's role and name, the model and
  the session's transcript digest as the tally holds it; it is linked to a ``must_fix`` finding when any review's
  ``must_fix`` finding names one of the claims it touches;
- **an unresolved key** (``unresolved_key``): each figure key a checked draft wrote that does not resolve on the
  snapshot, at a field where that draft's check found a verdict, with the key and those verdicts. The drafts and
  their verdicts are the write's own checker's (:class:`CheckedDrafts`, wrapping the checker the write hands the loop),
  and each key is read by the renderer the check reads it by (``figure_keys.render``);
- **a low score** (``editor_score``): each rubric item a review scored 0 or 1, with its score and reason, and whether
  that review held a ``must_fix`` finding.

A notice's need is code's words (``_NOTICE_NEEDS``), so notices of one item, or of any key, group exactly. Nothing
here reads a need, a reason or a finding's words, or chooses anything by them: the records hold what was said.

**A crash mid-write loses that write's gaps**, as it loses the write: they are stored at its end.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import keys_in, render
from threetears.evals.analysis.report_doc_checks import DocVerdict
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.kernel.analyst_gap import (
    EvalAnalystGap,
    GapBounce,
    GapEditorItem,
    GapToolCall,
    GapVerdict,
    analyst_gap_id,
    call_gap_id,
)
from threetears.evals.kernel.report_doc import ReportDoc, report_doc_prose
from threetears.evals.ops.report_writer.analyst import ANALYST_SESSION
from threetears.evals.ops.report_writer.editor import Review, editor_session
from threetears.evals.ops.report_writer.loop import WritingOutcome

__all__ = ["CheckedDrafts", "write_gaps"]

#: The action a gap call names.
_GAP_ACTION: Final = "capability_gap"

#: Each notice's need, in code's words: what the digest groups notices by. ``{key}`` and ``{n}`` / ``{name}`` are
#: filled from the notice; the digest reads a key as a key, so every key notice is one group, and each rubric item's
#: low scores another.
_NOTICE_NEEDS: Final[Mapping[str, str]] = {
    "unresolved_key": "a draft cited a figure key that does not resolve on its snapshot: {key}",
    "editor_score": "the editor scored rubric item {n}, {name}, below 2",
}

#: What a low score's ``why`` reads when the editor gave no reason.
_NO_REASON: Final = "the editor gave no reason"


@dataclass
class CheckedDrafts:
    """The write's checker, keeping every draft it checked and the verdicts it found, in order.

    :ivar check: The checker the drafts are checked by.
    :ivar checked: Each checked draft and its verdicts, oldest first.
    """

    check: Callable[[ReportDoc], list[DocVerdict]]
    checked: list[tuple[ReportDoc, tuple[DocVerdict, ...]]] = field(default_factory=list)

    def __call__(self, doc: ReportDoc) -> list[DocVerdict]:
        """Check ``doc`` and keep it with what was found.

        :param doc: A draft.
        :ptype doc: ReportDoc
        :return: The checker's verdicts.
        :rtype: list[DocVerdict]
        """
        verdicts = self.check(doc)
        self.checked.append((doc, tuple(verdicts)))
        return verdicts


@dataclass(frozen=True, slots=True, kw_only=True)
class _Provenance:
    """Where a write's records come from, the same for each but its session."""

    scope_id: str
    snapshot_id: str
    report_doc_id: str | None
    model: str
    created_by: str
    transcripts: Mapping[str, str]

    def of(self, session: str, role: str) -> dict[str, Any]:  # Any: model fields
        return {
            "scope_id": self.scope_id,
            "snapshot_id": self.snapshot_id,
            "report_doc_id": self.report_doc_id,
            "role": role,
            "session": session,
            "model": self.model,
            "transcript_digest": self.transcripts.get(session),
            "created_by": self.created_by,
        }


def write_gaps(
    *,
    scope_id: str,
    snapshot_id: str,
    report_doc_id: str | None,
    model: str,
    created_by: str,
    outcome: WritingOutcome,
    tally: Mapping[str, Any],  # Any: the tally's JSON
    drafts: CheckedDrafts,
    report_input: ReportInput,
) -> list[EvalAnalystGap]:
    """Every gap and notice a write noted (the module docstring), ready to store.

    :param scope_id: The scope the report was written in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot it was written from.
    :ptype snapshot_id: str
    :param report_doc_id: The report the write stored; ``None`` for a write that stored none.
    :ptype report_doc_id: str | None
    :param model: The model every session ran on.
    :ptype model: str
    :param created_by: Who asked for the report, as the host names the caller.
    :ptype created_by: str
    :param outcome: The writing loop's outcome: its sessions and its reviews.
    :ptype outcome: WritingOutcome
    :param tally: The write's tally as JSON, whose ``transcripts`` hold each session's digest.
    :ptype tally: Mapping[str, Any]
    :param drafts: The write's checker, holding every draft it checked.
    :ptype drafts: CheckedDrafts
    :param report_input: The snapshot's report input, which a key is read against.
    :ptype report_input: ReportInput
    :return: The records, the calls first (session by session, in call order), then the key notices, then the low
        scores; each id once.
    :rtype: list[EvalAnalystGap]
    """
    provenance = _Provenance(
        scope_id=scope_id,
        snapshot_id=snapshot_id,
        report_doc_id=report_doc_id,
        model=model,
        created_by=created_by,
        transcripts=tally.get("transcripts", {}),
    )
    records = [
        *_calls(outcome.sessions, outcome.reviews, provenance),
        *_unresolved_keys(drafts, report_input, provenance),
        *_low_scores(outcome.reviews, provenance),
    ]
    first: dict[str, EvalAnalystGap] = {}
    for record in records:
        first.setdefault(record.id, record)
    return list(first.values())


def _calls(
    sessions: Mapping[str, Mapping[str, Any]],  # Any: each session's JSON
    reviews: Sequence[Review],
    provenance: _Provenance,
) -> Iterator[EvalAnalystGap]:
    """Each ``capability_gap`` call a session made that its tool answered, in session and call order."""
    must_fix_claims = {
        finding.claim for review in reviews for finding in review.findings if finding.severity == "must_fix"
    }
    for name, stored in sessions.items():
        session = stored["session"] if name == ANALYST_SESSION else stored
        messages = [json.loads(text) for text in session["messages"]]
        refused = {
            block["tool_use_id"]
            for message in messages
            for block in message["content"]
            if block.get("type") == "tool_result" and block.get("is_error")
        }
        answered = {
            block["tool_use_id"]
            for message in messages
            for block in message["content"]
            if block.get("type") == "tool_result"
        }
        for message in messages:
            for block in message["content"]:
                if block.get("type") != "tool_use" or block["id"] in refused or block["id"] not in answered:
                    continue
                arguments = block["input"]
                if arguments.get("action") != _GAP_ACTION:
                    continue
                yield _called(arguments, must_fix_claims, provenance.of(name, str(session["role"])))


def _called(
    arguments: Mapping[str, Any],  # Any: the call's JSON arguments, which the action validated
    must_fix_claims: set[str],
    where: dict[str, Any],  # Any: model fields
) -> EvalAnalystGap:
    """One answered gap call as its record: its arguments read as the action read them, whitespace trimmed."""
    kind = str(arguments.get("kind", "missing")).strip()
    need, why = str(arguments["need"]).strip(), str(arguments["why"]).strip()
    snapshot_id = str(arguments["snapshot_id"]).strip()
    claims = [str(claim).strip() for claim in arguments.get("claim_ids", [])]
    return EvalAnalystGap(
        id=call_gap_id(snapshot_id, kind, need, why),
        kind=kind,
        need=need,
        why=why,
        workaround=str(arguments["workaround"]).strip(),
        claim_ids=claims,
        figure_keys=[str(key).strip() for key in arguments.get("figure_keys", [])],
        calls=[
            GapToolCall(
                action=str(call["action"]).strip(),
                arguments={key: value for key, value in call.items() if key != "action"},
            )
            for call in arguments.get("calls", [])
        ],
        must_fix=bool(must_fix_claims & set(claims)),
        **{**where, "snapshot_id": snapshot_id},
    )


def _unresolved_keys(
    drafts: CheckedDrafts, report_input: ReportInput, provenance: _Provenance
) -> Iterator[EvalAnalystGap]:
    """Each figure key a checked draft wrote that does not resolve, at a field its check found a verdict at."""
    names = {arm.id: arm.name for arm in fact_arms(report_input)}
    for submission, (doc, verdicts) in enumerate(drafts.checked, start=1):
        if not verdicts:
            continue
        for path, text in report_doc_prose(doc):
            at = [verdict for verdict in verdicts if verdict.path == path]
            if not at:
                continue
            try:
                keys = keys_in(text)
            except UnresolvableReference:
                continue  # the field's braces name no key at all; its verdict says so, and there is no key to note
            for key in keys:
                assert key.span is not None
                written = text[key.span[0] : key.span[1]]
                try:
                    render(written, report_input, names)
                except UnresolvableReference as refused:
                    yield _unresolved(submission, path, written, str(refused), at, provenance)


def _unresolved(
    submission: int, path: str, key: str, refused: str, at: Sequence[DocVerdict], provenance: _Provenance
) -> EvalAnalystGap:
    kind = "unresolved_key"
    return EvalAnalystGap(
        id=analyst_gap_id(provenance.snapshot_id, kind, ANALYST_SESSION, str(submission), path, key),
        kind=kind,
        need=_NOTICE_NEEDS[kind].format(key=key),
        why=refused,
        figure_keys=[key],
        bounce=GapBounce(
            submission=submission,
            path=path,
            key=key,
            verdicts=[
                GapVerdict(invariant=verdict.invariant, path=verdict.path, fault=verdict.fault) for verdict in at
            ],
        ),
        **provenance.of(ANALYST_SESSION, "analyst"),
    )


def _low_scores(reviews: Sequence[Review], provenance: _Provenance) -> Iterator[EvalAnalystGap]:
    """Each rubric item a review scored below 2, in round and rubric order."""
    kind = "editor_score"
    for review in reviews:
        session = editor_session(review.round)
        for item in review.items:
            if item.score >= 2:
                continue
            yield EvalAnalystGap(
                id=analyst_gap_id(provenance.snapshot_id, kind, session, str(item.n)),
                kind=kind,
                need=_NOTICE_NEEDS[kind].format(n=item.n, name=item.name),
                why=item.reason.strip() or _NO_REASON,
                editor_item=GapEditorItem(
                    round=review.round,
                    n=item.n,
                    name=item.name,
                    score=item.score,
                    reason=item.reason,
                    must_fix=review.must_fix,
                ),
                must_fix=review.must_fix,
                **provenance.of(session, "editor"),
            )
