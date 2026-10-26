"""Freezing a report input as a snapshot, and reading it back as the one input every analyst tool reads.

:func:`report_snapshot` builds the stored :class:`~threetears.evals.kernel.report_snapshot.EvalReportSnapshot` from a
report input and, where there is one, the campaign bundle it was built from. :func:`frozen_input` reads the input back,
checking it against the digest frozen beside it and validating it under this build's model, as a reporter plan case's
input is read: a snapshot whose input moved, or no longer reads, is refused with the remedy to freeze again, never
read as something else.

**The analyst's example pool is wider than the planner's.** A planner was sent at most
:data:`~threetears.evals.analysis.report_input_build.EXAMPLES` failed trials, one per aspect, arm and stratum, each
excerpt the start of what the judge read; an analyst pages through them (``examples_list``), so a snapshot's input is
built by :func:`analyst_input`: up to :data:`ANALYST_PER_SLICE` failed trials per aspect, arm and stratum, at most
:data:`ANALYST_EXAMPLES` in all, each excerpt cut around the failure the judge's reason quotes. More than one per slice
shows whether a failure is typical of it or a one-off. A report rendered against a snapshot is rendered against the
snapshot's own input, never against a fresh ``report_input(bundle)``, whose example pool differs.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.planned_analysis import report_input_document_digest
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input as built_input
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.report_snapshot import (
    FREEZE_AGAIN_REMEDY,
    EvalReportSnapshot,
    SnapshotSource,
    report_snapshot_id,
)

#: The most failed trials a snapshot's report input holds, eight times the planner's, which ``examples_list`` pages.
ANALYST_EXAMPLES = 96

#: The most failed trials a snapshot's report input holds per aspect, arm and stratum: enough to see whether a failure
#: is typical of its slice.
ANALYST_PER_SLICE = 3


def analyst_input(bundle: CampaignBundle) -> ReportInput:
    """The report input a snapshot freezes for one campaign bundle: the planner's, with the analyst's example pool.

    :param bundle: The campaign's bundle.
    :ptype bundle: CampaignBundle
    :return: The input, up to :data:`ANALYST_PER_SLICE` failed trials per aspect, arm and stratum and
        :data:`ANALYST_EXAMPLES` in all, each excerpt cut around the failure.
    :rtype: ReportInput
    """
    return built_input(bundle, examples=ANALYST_EXAMPLES, per_slice=ANALYST_PER_SLICE, excerpt="failure")


def report_snapshot(
    report_input: ReportInput,
    *,
    scope_id: str,
    campaign_id: str | None,
    source: SnapshotSource,
    created_by: str,
    bundle: CampaignBundle | None = None,
    analysis_id: str | None = None,
    assembled_at: str | None = None,
) -> EvalReportSnapshot:
    """The snapshot freezing one report input, and the bundle it was built from when there is one.

    :param report_input: The report input every tool will read.
    :ptype report_input: ReportInput
    :param scope_id: The scope the snapshot lives in.
    :ptype scope_id: str
    :param campaign_id: The campaign, or ``None`` for a bare report input.
    :ptype campaign_id: str | None
    :param source: Where the input came from: ``campaign``, ``analysis`` or ``report_input``.
    :ptype source: SnapshotSource
    :param created_by: Who froze it.
    :ptype created_by: str
    :param bundle: The campaign bundle the input was built from; ``None`` for a bare input.
    :ptype bundle: CampaignBundle | None
    :param analysis_id: The stored analysis it was taken from, for source ``analysis``.
    :ptype analysis_id: str | None
    :param assembled_at: When the bundle was assembled (ISO-8601), with a bundle.
    :ptype assembled_at: str | None
    :return: The snapshot, its id derived from the campaign and the input's digest.
    :rtype: EvalReportSnapshot
    :raises pydantic.ValidationError: The source contradicts what is given (a bare input with a bundle, a campaign's
        without one, an analysis id on another source).
    """
    document = report_input.model_dump(mode="json", by_alias=True)
    digest = report_input_document_digest(document)
    return EvalReportSnapshot(
        id=report_snapshot_id(campaign_id, digest),
        scope_id=scope_id,
        campaign_id=campaign_id,
        source=source,
        analysis_id=analysis_id,
        report_input=document,
        report_input_digest=digest,
        bundle=bundle.to_dict() if bundle is not None else None,
        bundle_fingerprint=bundle.fingerprint() if bundle is not None else None,
        bundle_assembled_at=assembled_at if bundle is not None else None,
        example_result_ids=_example_result_ids(report_input, bundle) if bundle is not None else {},
        created_by=created_by,
    )


def _example_result_ids(report_input: ReportInput, bundle: CampaignBundle) -> dict[str, str]:
    """Each example's trial id, to the stored result its bundle trial records."""
    named = bundle.trial_ids()
    results: Mapping[tuple[str, str], str] = {
        (aspect.aspect, trial.trial_id): trial.result_id for aspect in bundle.aspects for trial in aspect.trials
    }
    held: dict[str, str] = {}
    for example in report_input.examples:
        if example.trial is None:
            continue
        where = named.get(example.trial)
        if where is not None and where in results:
            held[example.trial] = results[where]
    return held


def frozen_input(snapshot: EvalReportSnapshot) -> ReportInput:
    """The report input a snapshot froze, checked against its digest and read under this build's model.

    The digest is checked on the raw document first, so an input changed in the store is refused as the input the
    snapshot does not hold, before any model reads it.

    :param snapshot: The stored snapshot.
    :ptype snapshot: EvalReportSnapshot
    :return: Its report input.
    :rtype: ReportInput
    :raises RefusedInput: The input does not digest to the one recorded beside it, or this build's report input model
        does not read it; the remedy is to freeze again.
    """
    held = report_input_document_digest(snapshot.report_input)
    if held != snapshot.report_input_digest:
        raise RefusedInput(
            f"snapshot {snapshot.id}'s report input digests to {held} and the snapshot recorded "
            f"{snapshot.report_input_digest}: it does not hold the input it names",
            remedy=FREEZE_AGAIN_REMEDY,
        )
    try:
        return ReportInput.model_validate(snapshot.report_input)
    except ValidationError as unread:
        raise RefusedInput(
            f"snapshot {snapshot.id}'s report input does not read as this build's report input "
            f"({unread.error_count()} fault(s); the first at {_first_place(unread)})",
            remedy=FREEZE_AGAIN_REMEDY,
        ) from unread


def _first_place(unread: ValidationError) -> str:
    """Where the first fault of a validation error sits, as a dotted path."""
    (first, *_rest) = unread.errors()
    return ".".join(str(part) for part in first["loc"]) or "(the input)"


__all__ = ["ANALYST_EXAMPLES", "ANALYST_PER_SLICE", "analyst_input", "frozen_input", "report_snapshot"]
