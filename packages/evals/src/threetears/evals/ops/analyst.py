"""The analyst's operations: freeze a campaign's evidence as a snapshot, read it back, and note a missing capability.

Every analyst tool reads one frozen report input, by the id of the snapshot that holds it
(:class:`~threetears.evals.kernel.report_snapshot.EvalReportSnapshot`). :func:`report_snapshot_freeze` makes one from a
campaign, assembling its evidence as it stands, or from a stored analysis, from the bundle frozen with it; it is the
one write. :func:`snapshot_input` reads a snapshot's input back for a tool, refusing one whose input moved or no
longer reads. :func:`report_check` reads a draft report document on its snapshot as the store and the editor would:
every verdict code finds, and the page as text. :func:`capability_gap` stores nothing: it acknowledges a capability
the analyst reasoned around, with a gap id derived from what it says, and writes one record on the
``threetears.evals.analyst`` logger, so the call sits in the transcript beside the calls that led to it and in the
server's log.

**A freeze runs in its call**, not as a job: it assembles one campaign's bundle, as ``analysis_estimate`` does in its
call, and makes no model call. The assembly and the report input it builds run on the host's blocking executor.

**A draft check stores nothing.** It builds the draft's page strict, as generation does
(:func:`~threetears.evals.analysis.report.doc_page.build.build`), runs the invariants over the draft and that page's
text above the appendix (:func:`~threetears.evals.analysis.report_doc_checks.check_report_doc`), and returns the
verdicts whole, then a page of the page's text (:func:`~threetears.evals.analysis.report.doc_page.text.doc_page_text`,
the function the editor's message is written by). It reads the snapshot and nothing else, so the same draft on the
same snapshot always returns the same result.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from threetears.evals.analysis.analyst.cited import CitedFigure
from threetears.evals.analysis.analyst.snapshot import analyst_input, frozen_input, report_snapshot
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.build import analysis_campaign_bundle
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.text import above_appendix_text, doc_page_text
from threetears.evals.analysis.report_doc_checks import DocVerdict, InvariantNumber, check_report_doc
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.service import assemble_campaign_bundle, get_analysis, require_evidence
from threetears.evals.kernel.errors import ConflictError, NotFoundError, RefusedInput, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.kernel.report_snapshot import (
    FIND_SNAPSHOT,
    FREEZE_AGAIN_REMEDY,
    EvalReportSnapshot,
    SnapshotSource,
)
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.hashing import bytes_digest
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import utc_now_iso

#: The logger every noted capability gap is recorded on, so a server's operator can read them in one place.
_ANALYST_LOGGER = "threetears.evals.analyst"

log = get_logger(_ANALYST_LOGGER)

#: The hex digits of the digest a gap id keeps.
_GAP_DIGITS = 16

#: What a reused snapshot's fields report: the first freeze's evidence, not this call's.
_REUSED_NOTE = (
    "the same report input was frozen before: the snapshot named is that one, and its source, analysis, bundle "
    "fingerprint and freeze time are that freeze's, which may differ from what this call named"
)

#: What a freeze named neither or both of.
_ONE_SOURCE_REMEDY = "name a campaign_id (campaigns_list) or an analysis_id (analyses_list), and not both"


class SnapshotFrozen(EvalBaseModel):
    """A snapshot a freeze made, or found already stored for the same evidence."""

    snapshot_id: str = Field(description="The snapshot's id, which every analyst tool takes as snapshot_id.")
    campaign_id: str | None = Field(description="The campaign frozen; None for a bare report input.")
    source: SnapshotSource = Field(
        description="campaign, analysis or report_input: where the stored snapshot's report input came from."
    )
    analysis_id: str | None = Field(
        description="The analysis the stored snapshot was frozen from, for source analysis."
    )
    report_input_digest: str = Field(description="sha256 of the frozen report input, re-checked on every read.")
    bundle_fingerprint: str | None = Field(description="The fingerprint of the bundle it was built from.")
    frozen_at: str = Field(description="When the stored snapshot was frozen (ISO-8601).")
    reused: bool = Field(
        description="True when the same evidence was already frozen: the snapshot stored then is the one named, and "
        "its source, analysis, bundle and freeze time are that freeze's, which may differ from this call's."
    )
    replaced: bool = Field(
        description="True when a snapshot stored for the same evidence no longer read, or no longer matched its "
        "digest, and this freeze wrote it again in its place."
    )
    note: str = Field(description="What a reused snapshot's fields are of; empty for a snapshot this call stored.")


class GapNoted(EvalBaseModel):
    """A capability the analyst reasoned around, acknowledged: nothing is stored, and the work goes on."""

    gap_id: str = Field(description="The gap's id: the same snapshot, need and why always name the same gap.")
    snapshot_id: str = Field(description="The snapshot the analyst was reading.")
    recorded_in: Literal["transcript"] = Field(
        description="Where the gap is kept: the call itself, in the transcript, beside the calls that led to it."
    )
    text: str = Field(description="The acknowledgement, naming the gap and the workaround the analyst goes on with.")


def _snapshot_of(storage: EvalStorage, snapshot_id: str, scope_id: str) -> EvalReportSnapshot:
    """The stored snapshot. Blocking.

    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The stored snapshot does not read under this build's model.
    """
    try:
        snapshot = storage.load_report_snapshot(snapshot_id, scope_id)
    except ValidationError as unread:
        raise ValidationFailedError(
            f"snapshot '{snapshot_id}' was stored by a build whose snapshot this build does not read "
            f"({unread.error_count()} fault(s))",
            remedy=FREEZE_AGAIN_REMEDY,
        ) from unread
    if snapshot is None:
        raise NotFoundError("report snapshot", snapshot_id, remedy=FIND_SNAPSHOT)
    return snapshot


def _read_input(storage: EvalStorage, snapshot_id: str, scope_id: str) -> ReportInput:
    """The snapshot's report input, checked and validated. Blocking."""
    snapshot = _snapshot_of(storage, snapshot_id, scope_id)
    try:
        return frozen_input(snapshot)
    except RefusedInput as refused:
        raise ValidationFailedError(refused.fault, remedy=refused.remedy) from refused


async def snapshot_input(host: OpsHost, scope_id: str, snapshot_id: str) -> ReportInput:
    """The report input a snapshot froze: what every analyst tool reads.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot, as the freeze named it.
    :ptype snapshot_id: str
    :return: Its report input, checked against its digest and read under this build's model.
    :rtype: ReportInput
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot, or its input, does not read under this build's model, or the input
        does not digest to the one recorded beside it; the remedy is to freeze again.
    """
    eval_host = host.eval_host
    return await run_blocking(eval_host.blocking_executor, _read_input, eval_host.storage, snapshot_id, scope_id)


def _frozen(
    bundle: CampaignBundle,
    *,
    scope_id: str,
    source: SnapshotSource,
    analysis_id: str | None,
    assembled_at: str,
    created_by: str,
) -> EvalReportSnapshot:
    """The snapshot of one bundle's report input, built with the analyst's example pool. Blocking."""
    return report_snapshot(
        analyst_input(bundle),
        scope_id=scope_id,
        campaign_id=bundle.campaign_id,
        source=source,
        created_by=created_by,
        bundle=bundle,
        analysis_id=analysis_id,
        assembled_at=assembled_at,
    )


def _sound(storage: EvalStorage, snapshot_id: str, scope_id: str) -> tuple[EvalReportSnapshot | None, str | None]:
    """The snapshot stored under an id, when it is sound, and why it is not when one is stored and is not. Blocking.

    A stored snapshot is sound when it reads under this build's model and its report input digests to the digest
    recorded beside it and reads as this build's report input: exactly what every analyst tool's read checks.

    :return: ``(snapshot, None)`` for a sound one, ``(None, None)`` for none stored, ``(None, why)`` for one stored
        that a read would refuse.
    """
    try:
        stored = storage.load_report_snapshot(snapshot_id, scope_id)
    except ValidationError as unread:
        return None, f"it does not read under this build's snapshot model ({unread.error_count()} fault(s))"
    if stored is None:
        return None, None
    try:
        frozen_input(stored)
    except RefusedInput as refused:
        return None, refused.fault
    return stored, None


def _kept(storage: EvalStorage, snapshot: EvalReportSnapshot) -> SnapshotFrozen:
    """Store the snapshot unless the same evidence is already frozen soundly, and say which is stored. Blocking.

    A snapshot stored under the same id that a read would refuse (it no longer reads, or its input no longer matches
    its digest) is deleted and this one written in its place: the id is derived from the evidence, so the fresh
    snapshot is the one the id has always named, and a broken one would otherwise be returned to every freeze.
    """
    stored, broken = _sound(storage, snapshot.id, snapshot.scope_id)
    if broken is not None:
        log.warning("report snapshot %s is replaced: %s", snapshot.id, broken)
        storage.delete_report_snapshot(snapshot.id, snapshot.scope_id)
    reused = stored is not None
    if stored is None:
        try:
            storage.save_report_snapshot(snapshot)
            stored = snapshot
        except ConflictError:
            # Another freeze of the same evidence stored it between the read and the write: its content is this one's.
            stored = _snapshot_of(storage, snapshot.id, snapshot.scope_id)
            reused = True
    return SnapshotFrozen(
        snapshot_id=stored.id,
        campaign_id=stored.campaign_id,
        source=stored.source,
        analysis_id=stored.analysis_id,
        report_input_digest=stored.report_input_digest,
        bundle_fingerprint=stored.bundle_fingerprint,
        frozen_at=stored.created_at,
        reused=reused,
        replaced=broken is not None,
        note=_REUSED_NOTE if reused else "",
    )


async def report_snapshot_freeze(
    host: OpsHost,
    scope_id: str,
    *,
    campaign_id: str | None = None,
    analysis_id: str | None = None,
    created_by: str,
) -> SnapshotFrozen:
    """Freeze a campaign's evidence, or a stored analysis's, as the snapshot every analyst tool reads.

    From a campaign, its evidence is assembled as it stands, the instant stamped before the assembly reads the insight
    ledger; from an analysis, its frozen bundle is put back together. Either way the report input is built with the
    analyst's example pool (:func:`~threetears.evals.analysis.analyst.snapshot.analyst_input`) and stored only
    when its id is not: the same evidence frozen twice names one snapshot (``reused``). A snapshot stored under the
    id that a read would refuse, because it no longer reads or no longer matches its digest, is written again in its
    place (``replaced``), so freezing again is the remedy every such refusal names.

    :param host: The host whose store holds the campaign and the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope they live in.
    :ptype scope_id: str
    :param campaign_id: The campaign to freeze; exactly one of it and ``analysis_id``.
    :ptype campaign_id: str | None
    :param analysis_id: The stored analysis to freeze.
    :ptype analysis_id: str | None
    :param created_by: Who freezes it, as the host names the caller.
    :ptype created_by: str
    :return: The snapshot, made or found.
    :rtype: SnapshotFrozen
    :raises ValidationFailedError: Both or neither of the campaign and the analysis were named, the campaign resolves
        no runs, or the analysis was written over a bundle of a shape this build does not read.
    :raises NotFoundError: No such campaign or analysis in the scope.
    """
    eval_host = host.eval_host
    executor, storage = eval_host.blocking_executor, eval_host.storage
    source: SnapshotSource
    if campaign_id is not None and analysis_id is None:
        source = "campaign"
        # Stamped BEFORE assembly, as a generation stamps it: the instant is the insight ledger's cutoff.
        assembled_at = utc_now_iso()
        bundle = await run_blocking(executor, assemble_campaign_bundle, eval_host, campaign_id, scope_id)
        require_evidence(bundle, campaign_id, scope_id)
    elif analysis_id is not None and campaign_id is None:
        source = "analysis"
        analysis = await run_blocking(executor, get_analysis, storage, analysis_id, scope_id)
        assembled_at = analysis.generation.bundle_assembled_at
        try:
            bundle = await run_blocking(executor, analysis_campaign_bundle, analysis)
        except RefusedInput as refused:
            raise ValidationFailedError(refused.fault, remedy=refused.remedy) from refused
    else:
        raise ValidationFailedError(
            "a freeze names the campaign whose evidence it freezes, or the analysis whose bundle it freezes",
            remedy=_ONE_SOURCE_REMEDY,
        )
    snapshot = await run_blocking(
        executor,
        lambda: _frozen(
            bundle,
            scope_id=scope_id,
            source=source,
            analysis_id=analysis_id,
            assembled_at=assembled_at,
            created_by=created_by,
        ),
    )
    return await run_blocking(executor, _kept, storage, snapshot)


def sized_page[T](compute: Callable[[int], T], limit: int, fits: Callable[[T], bool] | None) -> T:
    """The page of at most ``limit`` rows ``compute`` builds, as large as ``fits`` passes. Blocking.

    A page grows with its rows, so the largest that fits is found by halving between one row and ``limit``. A page
    of one row is returned even where it does not fit: one row is what a page always holds. With no ``fits`` (the
    caller named the page's size), the page of ``limit`` rows.

    :param compute: Builds the page holding at most the rows it is given.
    :ptype compute: Callable[[int], T]
    :param limit: The most rows the page holds.
    :ptype limit: int
    :param fits: Whether a page is short enough; ``None`` for a page of ``limit`` rows.
    :ptype fits: Callable[[T], bool] | None
    :return: The page.
    :rtype: T
    """
    page = compute(limit)
    if fits is None or fits(page):
        return page
    low, high = 1, limit - 1
    best = compute(1)
    while low <= high:
        middle = (low + high) // 2
        candidate = compute(middle)
        if fits(candidate):
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    return best


#: The most lines of a draft's page one result holds unless the caller asks for fewer: as many as fit one result.
CHECK_LINES = 200

#: What every draft check says it is.
_CHECK_READS_AS = (
    "a draft report document read on its snapshot as the store and the editor read it: every verdict code finds, "
    "each remedy stated once beside the faults it mends, then the page a reader and the editor read, every block as "
    "its values table"
)

#: The path the verdict of a page that does not build sits at: the page, which names the field it refused.
_PAGE = "page"

#: What a draft whose page does not build is told to do.
_UNBUILT_REMEDY = (
    "the page is drawn strict, as generation draws it, and stops at the first field it cannot draw: mend the field "
    "this names (the verdicts before it say how), then check the draft again to read its page"
)


class VerdictAt(BaseModel):
    """One verdict's place and fault, under the remedy it shares with the others of its group."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(description="The field path it sits at, as `claims[2].evidence[0].query.arms`.")
    fault: str = Field(description="What is wrong there.")


class VerdictGroup(BaseModel):
    """The verdicts of one check that share a remedy: the remedy stated once, then each place and its fault."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    invariant: InvariantNumber = Field(
        description="The check: 0 validity (a page that does not build among it), else the invariant's number."
    )
    remedy: str = Field(description="How to mend every fault below, and which tool shows what to mend it with.")
    faults: list[VerdictAt] = Field(description="Each verdict's field path and fault, in the order found.")


class DraftChecked(BaseModel):
    """A draft report document checked on its snapshot: its verdicts, then one page of the page it draws, as text.

    A plain model, not the engine's base: the page's lines are kept as the page writes them, blank ones at either end
    included, which the base's whitespace stripping would drop.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reads_as: str = Field(description="What this is.")
    passes: bool = Field(
        description="True when code finds nothing wrong: no verdict, and the page builds. The editor still reads it."
    )
    verdicts_total: int = Field(description="How many verdicts code finds on the draft; none for a draft that passes.")
    verdict_offset: int = Field(description="The first verdict this result holds.")
    verdicts: list[VerdictGroup] = Field(
        description="This result's verdicts in the order found: validity (0), then each invariant, a page that does "
        "not build last, at path page. Those that share a remedy are grouped under it, so each remedy is stated once."
    )
    verdicts_unread_from: int | None = Field(
        description="Where the verdicts this result could not hold start: send the same draft with this "
        "verdict_offset. None when the result holds every verdict from verdict_offset."
    )
    progress: str = Field(
        description="Which verdicts and which lines of the page this result holds, and how to read the rest: the same "
        "draft again, from verdicts_unread_from or unread_from; or that every line is shown; or that there is no page, "
        "since the draft does not build."
    )
    page: str = Field(
        description="These lines of the page as a reader and the editor read it, joined by newlines: each claim by its "
        "id and weight, each block by what code fixed about it and its values table; empty where the draft does not "
        "build, or where verdicts are still unread."
    )
    #: What its total counts, one and many: a page is sized as the tool would write it, in this unit.
    page_unit: ClassVar[tuple[str, str]] = ("line", "lines")
    total: int = Field(description="The lines of the whole page; none where the draft does not build.")
    offset: int = Field(description="The first line of the page this result holds.")
    unread_from: int | None = Field(
        description="Where the page's lines this call left unread start: send the same draft with this offset. None "
        "when the result reads to the page's last line, or there is no page."
    )


def _grouped(verdicts: list[DocVerdict]) -> list[VerdictGroup]:
    """The verdicts grouped by check and remedy, each group where its first verdict falls, each fault in its order."""
    groups: dict[tuple[InvariantNumber, str], list[VerdictAt]] = {}
    for verdict in verdicts:
        groups.setdefault((verdict.invariant, verdict.remedy), []).append(
            VerdictAt(path=verdict.path, fault=verdict.fault)
        )
    return [
        VerdictGroup(invariant=invariant, remedy=remedy, faults=faults)
        for (invariant, remedy), faults in groups.items()
    ]


def _again(verdict_offset: int, offset: int) -> str:
    """The call that reads on: the same draft, with the offsets that name where."""
    named = f"verdict_offset={verdict_offset} and offset={offset}" if verdict_offset else f"offset={offset}"
    return f"send the same draft again with {named}"


def _verdicts_said(verdict_offset: int, end: int, count: int) -> str:
    """Which verdicts a result holds, where it does not hold them all from the first; empty where it does."""
    if count == 0 or (verdict_offset == 0 and end == count):
        return ""
    if verdict_offset >= count:
        return f"verdicts 1 to {count} of {count} passed over (verdict_offset={verdict_offset}); verdict_offset=0 reads them"
    shown = f"verdicts {verdict_offset + 1} to {end} of {count} shown"
    if end < count:
        return (
            f"{shown}; {count - end} unread: send the same draft again with verdict_offset={end}; the page follows the "
            "last verdict"
        )
    return f"{shown}, the last of them"


def _page_said(verdict_offset: int, offset: int, end: int, total: int, *, built: bool) -> str:
    """Which lines of the page a result holds, and how to read the rest, without the draft written back."""
    if not built:
        return (
            "no page: the draft does not build, so there is none to read (the verdict at path page says where it "
            "stopped); mend the verdicts and check the draft again"
        )
    if offset >= total:
        return f"no line from offset={offset}: the page has {total} lines, read from offset=0"
    if offset == 0 and end == total:
        return f"all {total} lines of the page shown"
    shown = f"lines {offset + 1} to {end} of {total} of the page shown"
    if end < total:
        return f"{shown}; {total - end} unread: {_again(verdict_offset, end)}"
    return f"{shown}, the last of them"


def _draft_read(doc: ReportDoc, report_input: ReportInput) -> tuple[list[DocVerdict], list[str] | None]:
    """Every verdict on a draft, and its page's lines; ``None`` for the lines of a page that does not build. Blocking.

    The page is built strict, as generation builds it; its text above the appendix is what invariant 5 reads beside
    the draft's prose. A page that refuses to build is one more verdict, after the checks', which read the draft's own
    fields alone.
    """
    try:
        page = build(doc, report_input, strict=True)
    except UnresolvableReference as refused:
        unbuilt = DocVerdict(
            invariant=0, path=_PAGE, fault=f"the page does not build: {refused}", remedy=_UNBUILT_REMEDY
        )
        return [*check_report_doc(doc, report_input), unbuilt], None
    verdicts = check_report_doc(doc, report_input, above_appendix_text=above_appendix_text(page))
    return verdicts, doc_page_text(page).splitlines()


def _checked_page(
    verdicts: list[DocVerdict],
    lines: list[str] | None,
    *,
    verdict_offset: int,
    held: int,
    offset: int,
    limit: int,
) -> DraftChecked:
    """One result of a draft check: ``held`` verdicts from ``verdict_offset``, then, where every verdict from it is
    held, up to ``limit`` of the page's lines from ``offset`` (none where ``limit`` is 0)."""
    window = verdicts[verdict_offset : verdict_offset + held]
    read_to = verdict_offset + len(window)
    unread = read_to < len(verdicts)
    page = lines if lines is not None else []
    shown = [] if unread else page[offset : offset + limit]
    end = offset + len(shown)
    said = [_verdicts_said(verdict_offset, read_to, len(verdicts))]
    if not unread:
        if lines is not None and limit == 0 and offset < len(page):
            said.append(f"the page follows the last verdict: {_again(len(verdicts), offset)}")
        else:
            said.append(_page_said(verdict_offset, offset, end, len(page), built=lines is not None))
    return DraftChecked(
        reads_as=_CHECK_READS_AS,
        passes=not verdicts,
        verdicts_total=len(verdicts),
        verdict_offset=verdict_offset,
        verdicts=_grouped(window),
        verdicts_unread_from=read_to if unread else None,
        progress="; ".join(part for part in said if part),
        page="\n".join(shown),
        total=len(page),
        offset=offset,
        unread_from=end if lines is not None and end < len(page) else None,
    )


def _draft_checked(
    doc: ReportDoc,
    report_input: ReportInput,
    *,
    verdict_offset: int,
    limit: int,
    offset: int,
    fits: Callable[[DraftChecked], bool],
) -> DraftChecked:
    """The draft read once, then the result that fits: as many verdicts as fit from ``verdict_offset``, and, where
    every one from it fits, as many of the page's lines as fit beside them. Blocking."""
    verdicts, lines = _draft_read(doc, report_input)
    left = max(len(verdicts) - verdict_offset, 0)

    def result(held: int, rows: int) -> DraftChecked:
        return _checked_page(verdicts, lines, verdict_offset=verdict_offset, held=held, offset=offset, limit=rows)

    if left:
        bare = sized_page(lambda held: result(held, 0), left, fits)
        if bare.verdicts_unread_from is not None or lines is None or offset >= len(lines):
            return bare
        if not fits(result(left, 1)):
            return bare
    return sized_page(lambda rows: result(left, rows), limit, fits)


async def report_check(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    doc: ReportDoc,
    fits: Callable[[DraftChecked], bool],
    limit: int = CHECK_LINES,
    offset: int = 0,
    verdict_offset: int = 0,
) -> DraftChecked:
    """Read a draft report document on its snapshot as the store and the editor would; store nothing.

    Every verdict the invariants find, on the draft and on its page's text above the appendix, with a page that does
    not build as one more; then a page of the page's text as the editor reads it
    (:func:`~threetears.evals.analysis.report.doc_page.text.doc_page_text`). A result always passes ``fits``: the
    verdicts come first, grouped so each remedy is stated once, as many as fit from ``verdict_offset``; where some are
    left over it holds no line of the page and names the ``verdict_offset`` that reads them, so no verdict is ever cut
    unsaid. Where every verdict from it fits, the page's lines from ``offset`` fill what room is left. The same draft on
    the same snapshot always returns the same result.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot the draft was written from.
    :ptype snapshot_id: str
    :param doc: The draft.
    :ptype doc: ReportDoc
    :param fits: Whether a result's text is short enough; a result holds the most verdicts, then the most lines, that
        pass, one at least.
    :ptype fits: Callable[[DraftChecked], bool]
    :param limit: The most lines of the page one result holds.
    :ptype limit: int
    :param offset: The first line it holds.
    :ptype offset: int
    :param verdict_offset: The first verdict it holds.
    :ptype verdict_offset: int
    :return: The verdicts and the page's lines.
    :rtype: DraftChecked
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read; the remedy is to freeze again.
    """
    held = await snapshot_input(host, scope_id, snapshot_id)
    return await run_blocking(
        host.eval_host.blocking_executor,
        lambda: _draft_checked(doc, held, verdict_offset=verdict_offset, limit=limit, offset=offset, fits=fits),
    )


def _gap_id(snapshot_id: str, need: str, why: str) -> str:
    """The id of one noted gap: the same snapshot, need and why always name the same gap.

    :param snapshot_id: The snapshot the analyst was reading.
    :ptype snapshot_id: str
    :param need: What the analyst needed.
    :ptype need: str
    :param why: Why it needed it.
    :ptype why: str
    :return: ``gap:`` and 16 hex digits of the sha256 of the three.
    :rtype: str
    """
    return "gap:" + bytes_digest(f"{snapshot_id}\n{need}\n{why}".encode())[:_GAP_DIGITS]


async def capability_gap(
    host: OpsHost,
    scope_id: str,
    *,
    snapshot_id: str,
    need: str,
    why: str,
    workaround: str,
    claim_ids: list[str],
) -> GapNoted:
    """Acknowledge a capability the analyst reasoned around; store nothing.

    The snapshot is read, so a gap is noted against evidence that exists. The record is the call itself, in the
    transcript; this writes one INFO line on the ``threetears.evals.analyst`` logger as well, for the server's
    operator.

    :param host: The host whose store holds the snapshot.
    :ptype host: OpsHost
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param snapshot_id: The snapshot the analyst was reading.
    :ptype snapshot_id: str
    :param need: What the analyst needed and no tool gave.
    :ptype need: str
    :param why: What it needed it for.
    :ptype why: str
    :param workaround: What it did instead.
    :ptype workaround: str
    :param claim_ids: The claims of its report the workaround touches; empty for none yet.
    :ptype claim_ids: list[str]
    :return: The acknowledgement.
    :rtype: GapNoted
    :raises NotFoundError: No snapshot with that id in the scope.
    :raises ValidationFailedError: The snapshot does not read; the remedy is to freeze again.
    """
    await snapshot_input(host, scope_id, snapshot_id)
    noted = _gap_id(snapshot_id, need, why)
    log.info(
        "capability gap %s on %s: need %r; why %r; workaround %r; claims %s",
        noted,
        snapshot_id,
        need,
        why,
        workaround,
        claim_ids,
    )
    touches = f" It touches {', '.join(claim_ids)}." if claim_ids else ""
    return GapNoted(
        gap_id=noted,
        snapshot_id=snapshot_id,
        recorded_in="transcript",
        text=f"Noted {noted}: {need}. Go on with the workaround: {workaround}.{touches}",
    )


__all__ = [
    "CHECK_LINES",
    "CitedFigure",
    "DraftChecked",
    "GapNoted",
    "SnapshotFrozen",
    "capability_gap",
    "report_check",
    "report_snapshot_freeze",
    "sized_page",
    "snapshot_input",
]
