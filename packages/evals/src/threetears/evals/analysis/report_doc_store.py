"""The report document's store gate: the verdicts a document is checked by, the record it is stored as, its made line.

**One checker.** :func:`doc_verdicts` is what a ``report-doc/1`` document is checked by, wherever it is checked: the
analyst's submissions in the writing loop, the store gate, and the analyst's ``report_check`` (``ops/analyst.py``),
which reads it through :func:`doc_verdicts_and_page` for the page it built too. It builds the page strict
(:func:`~threetears.evals.analysis.report.doc_page.build.build`) and runs the invariants
(:func:`~threetears.evals.analysis.report_doc_checks.check_report_doc`, validity first) over the document and the
page's text above its appendix. A page that cannot be built strict is a verdict too, naming the field it refused,
never a crash.

**The gate.** :func:`store_report_doc` re-runs that checker before it stores, on the page as it will be read (the
editor's findings left unresolved in Method, the made line at its foot), and refuses a document with any verdict, so
nothing that fails validity or an invariant is ever stored, whatever the loop that wrote it believed.

**The generation record** (:func:`generation_record`, format 2) is what the stored document's ``generation`` holds
(plan §4.5): the model the report was written with, the skill texts' digests, the writing loop's tally (every turn,
then the sums by role, by round and in all, tool calls, bounces, wall time, a transcript digest per session), every
review as code scored it, and each session (the analyst's and each review round's) by ``{digest, turns, ref}``: its
transcript digest, the turns it sent, and the id of the ``eval_report_session`` document that holds it
(:func:`~threetears.evals.kernel.report_doc.report_session_id`). It holds digests and refs, never a message, so a
listing of report documents loads no session, and a later question resumes the conversation that wrote the report, or
the review of it, from the session's own document. A record for a report that is not stored names no ref, and holds
each session inline (``session``) instead, so a paid run that stores nothing keeps every session in memory.
:func:`made_line` writes the page's made line from it, the one place that line is written.

**The store gate writes the sessions, then the document** (``EvalStorage.save_report_doc``): a stored document never
names a session that was not written. A write that fails part way leaves the sessions written before it, which no
document names and ``drop_derived`` sweeps, and no document; the writing loop raises it with the whole record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report.doc_page.text import above_appendix_text
from threetears.evals.analysis.report_doc_checks import DocVerdict, InvariantNumber, check_report_doc
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.numbers import spell_dollars
from threetears.evals.kernel.report_doc import (
    EvalReportDoc,
    EvalReportSession,
    ReportDoc,
    ReportFinding,
    report_session_id,
)
from threetears.evals.kernel.storage import EvalStorage

__all__ = ["doc_verdicts", "doc_verdicts_and_page", "generation_record", "made_line", "store_report_doc"]

#: The format of the generation record :func:`generation_record` writes and :func:`made_line` reads.
_GENERATION_FORMAT: Final = 2

#: Which check a strict build's refusal is reported under, by the field it names: the first prefix it starts with.
_REFUSED_UNDER: Final[tuple[tuple[str, InvariantNumber], ...]] = (
    ("recommends", 0),
    ("confidence.caveats", 3),
    ("claims[", 2),
)

#: What a strict build's refusal is mended with.
_BUILD_REMEDY = (
    "the page cannot be drawn as written: block_render draws a block as the page will and figure_keys_check reads a "
    "key as the page prints it; mend the field named until both do"
)

#: How many hex digits of a digest the made line prints.
_DIGEST_SHOWN: Final = 12


def doc_verdicts(
    doc: ReportDoc, report_input: ReportInput, *, findings: Sequence[ReportFinding] = (), made: str = ""
) -> list[DocVerdict]:
    """Every verdict against a document on its snapshot: validity's and the invariants', over the strict page too.

    :param doc: The document.
    :ptype doc: ReportDoc
    :param report_input: The report input of the snapshot it was written from.
    :ptype report_input: ReportInput
    :param findings: The editor's findings the page prints in Method; none for a draft.
    :ptype findings: Sequence[ReportFinding]
    :param made: The page's made line; empty for a draft.
    :ptype made: str
    :return: The verdicts, in check order; empty for a document that may be stored. A strict build's refusal is the
        last verdict, under the check its field belongs to.
    :rtype: list[DocVerdict]
    """
    verdicts, _page = doc_verdicts_and_page(doc, report_input, findings=findings, made=made)
    return verdicts


def doc_verdicts_and_page(
    doc: ReportDoc, report_input: ReportInput, *, findings: Sequence[ReportFinding] = (), made: str = ""
) -> tuple[list[DocVerdict], DocPage | None]:
    """:func:`doc_verdicts`, with the strict page it built, so a caller that reads the page builds it once.

    :param doc: The document.
    :ptype doc: ReportDoc
    :param report_input: The report input of the snapshot it was written from.
    :ptype report_input: ReportInput
    :param findings: The editor's findings the page prints in Method; none for a draft.
    :ptype findings: Sequence[ReportFinding]
    :param made: The page's made line; empty for a draft.
    :ptype made: str
    :return: The verdicts, as :func:`doc_verdicts` returns them, and the strict page, or ``None`` where it does not
        build (its refusal is then the last verdict).
    :rtype: tuple[list[DocVerdict], DocPage | None]
    """
    try:
        page = build(doc, report_input, strict=True, findings=findings, made=made)
    except UnresolvableReference as refused:
        return [*check_report_doc(doc, report_input), _refusal(str(refused))], None
    return check_report_doc(doc, report_input, above_appendix_text=above_appendix_text(page)), page


def _refusal(said: str) -> DocVerdict:
    """A strict build's refusal (``<field path> <why>``) as a verdict at its field."""
    path, _, why = said.partition(" ")
    invariant: InvariantNumber = next((number for prefix, number in _REFUSED_UNDER if path.startswith(prefix)), 1)
    return DocVerdict(invariant=invariant, path=path, fault=why or said, remedy=_BUILD_REMEDY)


def generation_record(
    *,
    report_doc_id: str | None,
    model: str,
    tally: Mapping[str, Any],  # Any: the tally's JSON
    skill: Mapping[str, str],
    reviews: Sequence[Mapping[str, Any]],  # Any: each review's JSON
    sessions: Mapping[str, Mapping[str, Any]],  # Any: each session's JSON
) -> dict[str, Any]:  # Any: a JSON object
    """The stored document's ``generation``: how the report was written and what it cost (the module docstring).

    :param report_doc_id: The id the report is stored under, which names each session's document; ``None`` for a
        report that is not stored, whose record holds each session inline.
    :ptype report_doc_id: str | None
    :param model: The model id the report was written with.
    :ptype model: str
    :param tally: The writing loop's tally (``Tally.to_json()``), whose transcript digests and turns are each session's.
    :ptype tally: Mapping[str, Any]
    :param skill: The digest of each system text the sessions started from, by role, and the skill's own.
    :ptype skill: Mapping[str, str]
    :param reviews: Every review, in round order, as code scored it.
    :ptype reviews: Sequence[Mapping[str, Any]]
    :param sessions: Each session's JSON (its ``to_json()``), by name: the analyst's and each review round's.
    :ptype sessions: Mapping[str, Mapping[str, Any]]
    :return: ``{"format", "model", "skill", "tally", "reviews", "sessions"}``, ``sessions`` by name ``{"digest",
        "turns", "ref"}``; with no ``report_doc_id``, ``ref`` is ``None`` and the session is held under ``session``.
    :rtype: dict[str, Any]
    """
    return {
        "format": _GENERATION_FORMAT,
        "model": model,
        "skill": dict(skill),
        "tally": dict(tally),
        "reviews": [dict(review) for review in reviews],
        "sessions": {name: _session_entry(name, session, tally, report_doc_id) for name, session in sessions.items()},
    }


def _session_entry(
    name: str,
    session: Mapping[str, Any],  # Any: the session's JSON
    tally: Mapping[str, Any],  # Any: the tally's JSON
    report_doc_id: str | None,
) -> dict[str, Any]:  # Any: a JSON object
    """One session's entry in the generation record: its digest and turns from the tally, then its ref or itself."""
    turns: Sequence[Mapping[str, Any]] = tally.get("turns", [])
    transcripts: Mapping[str, str] = tally.get("transcripts", {})
    entry: dict[str, Any] = {
        "digest": transcripts.get(name),
        "turns": sum(1 for turn in turns if turn.get("session") == name),
    }
    if report_doc_id is None:
        return entry | {"ref": None, "session": dict(session)}
    return entry | {"ref": report_session_id(report_doc_id, name)}


def made_line(record: EvalReportDoc) -> str:
    """How a stored report was made, in one line, for its page's foot: models, served models, cost, skill, date.

    :param record: The stored document, its ``generation`` as :func:`generation_record` wrote it.
    :ptype record: EvalReportDoc
    :return: The line.
    :rtype: str
    :raises ValueError: A generation record of another format, or one missing what the line reads.
    """
    generation: Mapping[str, Any] = record.generation  # Any: the record's JSON, read by its keys below
    if generation.get("format") != _GENERATION_FORMAT:
        raise ValueError(
            f"a generation record of format {generation.get('format')!r}; this build reads format {_GENERATION_FORMAT}"
        )
    try:
        turns: list[Mapping[str, Any]] = generation["tally"]["turns"]
        total: Mapping[str, Any] = generation["tally"]["total"]
        skill: str = generation["skill"]["package"]
        model: str = generation["model"]
    except (KeyError, TypeError) as missing:
        raise ValueError(f"a generation record out of shape: {missing!r} is missing") from missing
    models = sorted({str(turn["model"]) for turn in turns} | {model})
    served = sorted({str(turn["served_model"]) for turn in turns if turn.get("served_model")})
    cost = total.get("cost_usd")
    spent = (
        f"for {spell_dollars(cost)}"
        if isinstance(cost, int | float) and not isinstance(cost, bool)
        else f"at a cost not known ({total.get('unpriced_turns')} turns unpriced)"
    )
    served_as = f", served as {', '.join(served)}" if served else ""
    return (
        f"Written by {', '.join(models)}{served_as}, {spent}, with analyst skill {skill[:_DIGEST_SHOWN]}, on "
        f"{record.created_at[:10]}."
    )


def store_report_doc(
    storage: EvalStorage,
    *,
    report_doc_id: str,
    scope_id: str,
    snapshot_id: str,
    campaign_id: str | None,
    doc: ReportDoc,
    report_input: ReportInput,
    ending: Literal["stored", "stored_with_findings", "over_budget"],
    findings: Sequence[ReportFinding],
    generation: Mapping[str, Any],  # Any: the generation record's JSON
    sessions: Mapping[str, Mapping[str, Any]],  # Any: each session's JSON
    created_by: str,
) -> EvalReportDoc:
    """Store a report document and its sessions, after checking it again on the page as it will be read. Blocking.

    :param storage: Where it is stored.
    :ptype storage: EvalStorage
    :param report_doc_id: The id it is stored under, which the generation record's refs name.
    :ptype report_doc_id: str
    :param scope_id: The scope it is stored in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot it was written from.
    :ptype snapshot_id: str
    :param campaign_id: The snapshot's campaign, or ``None`` for a bare report input.
    :ptype campaign_id: str | None
    :param doc: The document.
    :ptype doc: ReportDoc
    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param ending: How the writing loop ended.
    :ptype ending: Literal["stored", "stored_with_findings", "over_budget"]
    :param findings: The editor's findings left unresolved, which the page prints in Method.
    :ptype findings: Sequence[ReportFinding]
    :param generation: The generation record (:func:`generation_record` with ``report_doc_id``), naming each session
        by ref.
    :ptype generation: Mapping[str, Any]
    :param sessions: Each session's JSON, by name: exactly the sessions the record names.
    :ptype sessions: Mapping[str, Mapping[str, Any]]
    :param created_by: Who asked for the report, as the host named the caller.
    :ptype created_by: str
    :return: The stored record.
    :rtype: EvalReportDoc
    :raises ValueError: The document has a verdict on its page as it will be read, each listed; or the record's
        sessions are not the ones handed in, each by the ref this report gives it. Nothing is stored.
    :raises ConflictError: A document or session is already stored under its id (see ``EvalStorage.save_report_doc``).
    :raises StorageError: Any other write failure, which leaves only sessions no document names.
    """
    record = EvalReportDoc(
        id=report_doc_id,
        scope_id=scope_id,
        snapshot_id=snapshot_id,
        campaign_id=campaign_id,
        doc=doc.model_dump(mode="json", by_alias=True),
        ending=ending,
        findings=[finding.model_dump(mode="json") for finding in findings],
        generation=dict(generation),
        created_by=created_by,
    )
    stored = _stored_sessions(record, generation, sessions)
    verdicts = doc_verdicts(doc, report_input, findings=findings, made=made_line(record))
    if verdicts:
        listed = "\n".join(f"- [{verdict.invariant}] {verdict.path}: {verdict.fault}" for verdict in verdicts)
        raise ValueError(
            f"the report document is not stored: it has {len(verdicts)} verdicts on snapshot {snapshot_id}, and "
            f"a stored document has none\n{listed}"
        )
    storage.save_report_doc(record, sessions=stored)
    return record


def _stored_sessions(
    record: EvalReportDoc,
    generation: Mapping[str, Any],  # Any: the generation record's JSON
    sessions: Mapping[str, Mapping[str, Any]],  # Any: each session's JSON
) -> list[EvalReportSession]:
    """Each session as its own document, refusing a record whose entries do not name exactly these, by this report."""
    entries: Mapping[str, Mapping[str, Any]] = generation.get("sessions", {})
    refs = {name: entry.get("ref") for name, entry in entries.items()}
    expected = {name: report_session_id(record.id, name) for name in sessions}
    if refs != expected:
        raise ValueError(
            f"the report document is not stored: its generation record names the sessions {refs}, and the sessions "
            f"handed in are stored as {expected}"
        )
    return [
        EvalReportSession(
            id=expected[name],
            scope_id=record.scope_id,
            report_doc_id=record.id,
            name=name,
            digest=entries[name]["digest"],
            turns=entries[name]["turns"],
            session=dict(session),
        )
        for name, session in sessions.items()
    ]
