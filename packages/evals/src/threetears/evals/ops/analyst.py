"""The analyst's operations: freeze a campaign's evidence as a snapshot, read it back, and note a missing capability.

Every analyst tool reads one frozen report input, by the id of the snapshot that holds it
(:class:`~threetears.evals.kernel.report_snapshot.EvalReportSnapshot`). :func:`report_snapshot_freeze` makes one from a
campaign, assembling its evidence as it stands, or from a stored analysis, from the bundle frozen with it; it is the
one write. :func:`snapshot_input` reads a snapshot's input back for a tool, refusing one whose input moved or no
longer reads. :func:`capability_gap` stores nothing: it acknowledges a capability the analyst reasoned around, with a
gap id derived from what it says, and writes one record on the ``threetears.evals.analyst`` logger, so the call sits in the
transcript beside the calls that led to it and in the server's log.

**A freeze runs in its call**, not as a job: it assembles one campaign's bundle, as ``analysis_estimate`` does in its
call, and makes no model call. The assembly and the report input it builds run on the host's blocking executor.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationError

from threetears.evals.analysis.analyst.cited import CitedFigure
from threetears.evals.analysis.analyst.snapshot import ANALYST_EXAMPLES, frozen_input, report_snapshot
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report.build import analysis_campaign_bundle
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.service import assemble_campaign_bundle, get_analysis, require_evidence
from threetears.evals.kernel.errors import ConflictError, NotFoundError, RefusedInput, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
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
        report_input(bundle, examples=ANALYST_EXAMPLES),
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
    analyst's example pool (:data:`~threetears.evals.analysis.analyst.snapshot.ANALYST_EXAMPLES`) and stored only
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
    "CitedFigure",
    "GapNoted",
    "SnapshotFrozen",
    "capability_gap",
    "report_snapshot_freeze",
    "snapshot_input",
]
