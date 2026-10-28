"""The gap digest: stored analyst gaps and notices grouped exactly, ranked, and written as issue-ready text.

:func:`gap_digest` reads :class:`~threetears.evals.kernel.analyst_gap.EvalAnalystGap` records (what a model said in a
``capability_gap`` call, or a notice code wrote) and decides nothing about them: it groups, counts, orders and quotes.

**Grouping is exact and mechanical.** Two records are one group when their kinds are equal and their needs normalize
to the same text. The normalization, in order:

1. Unicode NFKC, then case folded;
2. every figure key (``{path|filter}``) and every backticked span becomes ``_key_``;
3. every prefixed id (``report-snapshot:0193…``, ``gap:ab12…``, ``arm:1017``: a word, a colon, then a run holding a
   digit) and every UUID or run of eight or more hex digits holding a digit becomes ``_id_``;
4. every dotted path (a word starting with a letter, then one or more ``.part``: ``cost.unreported.qwen``) becomes
   ``_key_``;
5. every character that is not a letter, a digit, an underscore or whitespace becomes a space;
6. runs of whitespace become one space, and the ends are trimmed.

So a need that differs only in case, spacing, punctuation, or the ids and keys it names joins its group, and one that
differs in any word stays apart. No model reads a need: a paraphrase is a group of its own (D7: model grouping
waits for a digest that shows paraphrase noise).

**Ranking.** ``suspect`` groups first (a result that looks wrong), then groups any of whose records came with a
``must_fix`` finding, then the rest. Within each, the larger count first, then the later last-seen time, then the
kind and the normalized need in alphabetical order, so one store's digest reads in one order every time.

**Issue text.** Each group carries a title and a body a host may post as an issue as written: what was needed and
why, every distinct workaround, and each record's evidence (a ``suspect`` call's reproducing keys and tool calls, an
``unresolved_key`` notice's key and its bounce's verdicts, an ``editor_score`` notice's item, score and reason), with
where each came from. The engine posts nothing: filing the issue is the host's, with its own credentials.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Final

from threetears.evals.kernel.analyst_gap import EvalAnalystGap, GapDigest, GapGroup

#: What each kind this build knows records, as the digest's body says it; a kind a newer build added reads as itself.
_KIND_WORDS: Final[dict[str, str]] = {
    "missing": "a capability the analyst needed and no tool gave",
    "suspect": "a result that looked wrong to the analyst or the editor",
    "unresolved_key": "a figure key that did not resolve on its snapshot, which sent a draft back",
    "editor_score": "a rubric item the editor scored below 2",
}

#: Where a group stands in the digest's order: suspect first, then must_fix-linked, then the rest.
_SUSPECT, _MUST_FIX, _REST = 0, 1, 2

#: The placeholders a normalized need writes for what it names.
_KEY_MARK: Final = " _key_ "
_ID_MARK: Final = " _id_ "

_BRACED_KEY = re.compile(r"\{[^{}]*\}")
_BACKTICKED = re.compile(r"`[^`]*`")
_PREFIXED_ID = re.compile(r"\b[\w-]+:[\w.-]*\d[\w.-]*")
_HEX_ID = re.compile(r"\b(?=[0-9a-f-]*\d)[0-9a-f]{8,}(?:-[0-9a-f]{4,})*\b")
_DOTTED_PATH = re.compile(r"\b[^\W\d_][\w-]*(?:\.[\w-]+)+\b")
_NOT_A_WORD = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")

#: The most characters of a need a title carries.
_TITLE_NEED: Final = 100


def _normalized_need(need: str) -> str:
    """A need as the digest groups it (the module docstring states each step).

    :param need: The need as written.
    :ptype need: str
    :return: The normalized text.
    :rtype: str
    """
    text = unicodedata.normalize("NFKC", need).casefold()
    text = _BRACED_KEY.sub(_KEY_MARK, text)
    text = _BACKTICKED.sub(_KEY_MARK, text)
    text = _PREFIXED_ID.sub(_ID_MARK, text)
    text = _HEX_ID.sub(_ID_MARK, text)
    text = _DOTTED_PATH.sub(_KEY_MARK, text)
    text = _NOT_A_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def gap_digest(records: Sequence[EvalAnalystGap], *, since: str | None = None) -> GapDigest:
    """Group, rank and write up ``records`` (the module docstring).

    :param records: The records, oldest first (``EvalStorage.list_analyst_gaps``).
    :ptype records: Sequence[EvalAnalystGap]
    :param since: The window's start as the caller gave it, echoed; the records are already read from it.
    :ptype since: str | None
    :return: The digest.
    :rtype: GapDigest
    """
    grouped: dict[tuple[str, str], list[EvalAnalystGap]] = {}
    for record in records:
        grouped.setdefault((record.kind, _normalized_need(record.need)), []).append(record)
    groups = [_group(kind, normalized, members) for (kind, normalized), members in grouped.items()]
    # Stable sorts, least significant key first: alphabetical, then the later last-seen, then the count, then the rank.
    groups.sort(key=lambda group: (group.kind, group.normalized))
    groups.sort(key=lambda group: group.last_seen, reverse=True)
    groups.sort(key=lambda group: group.count, reverse=True)
    groups.sort(key=_rank)
    return GapDigest(since=since, records=len(records), groups=groups)


def _rank(group: GapGroup) -> int:
    if group.kind == "suspect":
        return _SUSPECT
    return _MUST_FIX if group.must_fix else _REST


def _group(kind: str, normalized: str, members: list[EvalAnalystGap]) -> GapGroup:
    first, last = members[0], members[-1]
    must_fix = any(member.must_fix for member in members)
    return GapGroup(
        kind=kind,
        need=first.need,
        normalized=normalized,
        count=len(members),
        first_seen=first.created_at,
        last_seen=last.created_at,
        must_fix=must_fix,
        snapshots=_distinct(member.snapshot_id for member in members),
        reports=_distinct(member.report_doc_id for member in members if member.report_doc_id is not None),
        gap_ids=[member.id for member in members],
        title=_title(kind, first.need),
        body=_body(kind, members, must_fix=must_fix),
    )


def _distinct(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _title(kind: str, need: str) -> str:
    line = _SPACES.sub(" ", need).strip()
    if len(line) > _TITLE_NEED:
        line = line[: _TITLE_NEED - 1].rstrip() + "…"
    return f"evals analyst gap ({kind}): {line}"


def _body(kind: str, members: list[EvalAnalystGap], *, must_fix: bool) -> str:
    first, last = members[0], members[-1]
    what = _KIND_WORDS.get(kind, f"a gap of kind {kind}, which this build does not know")
    seen = (
        f"Noted {len(members)} times, first {first.created_at}, last {last.created_at}."
        if len(members) > 1
        else f"Noted once, {first.created_at}."
    )
    lines = [
        f"**Kind:** {kind}: {what}. {seen}",
        f"**Came with a must_fix finding:** {'yes' if must_fix else 'no'}.",
        "",
        "**Need:**",
        *_bullets(member.need for member in members),
        "",
        "**Why:**",
        *_bullets(member.why for member in members),
    ]
    workarounds = _distinct(member.workaround for member in members if member.workaround)
    if workarounds:
        lines += ["", "**Workarounds:**", *_bullets(workarounds)]
    lines += ["", "**Evidence:**"]
    for member in members:
        lines += _evidence(member)
    reports = _distinct(member.report_doc_id for member in members if member.report_doc_id is not None)
    lines += [
        "",
        f"**Snapshots:** {', '.join(_distinct(member.snapshot_id for member in members))}",
        f"**Reports:** {', '.join(reports) if reports else 'none stored'}",
    ]
    return "\n".join(lines)


def _bullets(values: Iterable[str]) -> list[str]:
    return [f"- {value}" for value in _distinct(values)]


def _evidence(record: EvalAnalystGap) -> list[str]:
    where = [f"snapshot {record.snapshot_id}"]
    where.append(f"report {record.report_doc_id}" if record.report_doc_id is not None else "no report stored")
    if record.session is not None:
        where.append(f"session {record.session}" + (f" ({record.role})" if record.role else ""))
    if record.model is not None:
        where.append(f"model {record.model}")
    if record.transcript_digest is not None:
        where.append(f"transcript {record.transcript_digest}")
    where.append(f"asked by {record.created_by}")
    lines = [f"- {record.id}, {record.created_at}: {'; '.join(where)}"]
    if record.claim_ids:
        lines.append(f"  - claims: {', '.join(record.claim_ids)}")
    if record.figure_keys:
        lines.append(f"  - figure keys: {', '.join(f'`{key}`' for key in record.figure_keys)}")
    for call in record.calls:
        arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
        lines.append(f"  - reproduce: action `{call.action}` with `{arguments}`")
    if record.bounce is not None:
        bounce = record.bounce
        lines.append(f"  - bounced submission {bounce.submission} at `{bounce.path}` on key `{bounce.key}`:")
        lines += [f"    - [invariant {verdict.invariant}] {verdict.path}: {verdict.fault}" for verdict in bounce.verdicts]
    if record.editor_item is not None:
        item = record.editor_item
        with_must_fix = "with" if item.must_fix else "without"
        lines.append(
            f"  - review round {item.round}, item {item.n} ({item.name}): {item.score} of 2, {with_must_fix} a must_fix "
            f"finding in that review: {item.reason}"
        )
    return lines


__all__ = ["gap_digest"]
