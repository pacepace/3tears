"""The report snapshot: one campaign's evidence frozen as the report input every analyst tool reads.

The analyst and its editor investigate a campaign through tools, and each tool takes a ``snapshot_id``: the id of an
:class:`EvalReportSnapshot`, a stored, immutable document holding the report input (``report-input/1``) whole, as it
was built at freeze. Every tool is a pure function of that input and its own parameters, so an analyst's calls, an
editor's in a fresh context, and a replay of either read the same figures.

**Content-addressed.** A snapshot's id is :func:`report_snapshot_id` of its campaign and its report input's digest,
so freezing unchanged evidence names the snapshot already stored, and a snapshot can never hold another input than
the one its id names. A snapshot seeded from a bare report input, with no campaign behind it, is addressed by its
digest alone.

**Held as JSON.** The kernel cannot read the report input's model, which is the analysis package's, so the input is
kept as its JSON by alias, and the bundle it was built from as ``CampaignBundle.to_dict()``. The analysis side
validates the input and re-checks its digest on every read, as a reporter plan case does, and refuses a snapshot
whose input no longer reads: the remedy is to freeze again (:data:`FREEZE_AGAIN_REMEDY`).

**Derived and regenerable**, like an analysis: freezing the campaign again makes it again, ``EvalStorage.drop_derived``
sweeps it, and a regenerable schema bump refuses it on read.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from threetears.evals.schema.base import EvalDocumentModel, VerbatimJsonObject
from threetears.evals.schema.hashing import bytes_digest
from threetears.evals.schema.models import SchemaVersion, utc_now_iso
from threetears.evals.schema.versioning import REGENERABLE_SCHEMA_VERSION

#: Where a snapshot's report input came from: a campaign's evidence assembled at freeze (``campaign``), a stored
#: analysis's frozen bundle (``analysis``), or a report input handed over bare, with no campaign behind it
#: (``report_input``).
SnapshotSource = Literal["campaign", "analysis", "report_input"]

#: The prefix every snapshot id carries, so an id names its document type where it is read.
_ID_PREFIX = "report-snapshot:"

#: The hex digits of the digest a snapshot id keeps.
_ID_DIGITS = 32

#: How a caller finds a snapshot's id: it is what the freeze returned.
FIND_SNAPSHOT = (
    "a snapshot's id is the one report_snapshot_freeze returned; freeze the campaign or the analysis to get one"
)

#: What a caller does about a snapshot whose report input no longer reads, or no longer matches its digest.
FREEZE_AGAIN_REMEDY = (
    "freeze the campaign or the analysis again with report_snapshot_freeze, which builds its report input as this "
    "build reads it, and use the snapshot id it returns"
)


def report_snapshot_id(campaign_id: str | None, report_input_digest: str) -> str:
    """The id of the snapshot holding one campaign's report input: the same evidence always names the same snapshot.

    :param campaign_id: The campaign, or ``None`` for a report input handed over bare.
    :ptype campaign_id: str | None
    :param report_input_digest: The digest of the report input's JSON (``report_input_document_digest``).
    :ptype report_input_digest: str
    :return: ``report-snapshot:`` and 32 hex digits of the sha256 of the campaign (``-`` for none) and the digest.
    :rtype: str
    """
    digest = bytes_digest(f"{campaign_id or '-'}\n{report_input_digest}".encode())
    return _ID_PREFIX + digest[:_ID_DIGITS]


class EvalReportSnapshot(EvalDocumentModel):
    """One campaign's report input, frozen whole, which every analyst tool reads by the snapshot's id.

    Immutable once stored: a freeze writes it only when its id is not stored yet, and the id is derived from its
    content (:func:`report_snapshot_id`). ``bundle`` is the campaign bundle the input was built from, kept so a
    stored report can be traced to its evidence without assembling it again; no tool reads it.
    """

    id: str = Field(min_length=1, description="`report_snapshot_id(campaign_id, report_input_digest)`.")
    doc_type: Literal["eval_report_snapshot"] = "eval_report_snapshot"
    schema_version: SchemaVersion = REGENERABLE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    campaign_id: str | None = Field(
        description="Reference → EvalCampaign.id: the campaign frozen. None for a report input handed over bare."
    )
    source: SnapshotSource = Field(
        description="Where the report input came from: the campaign's evidence assembled at freeze, a stored "
        "analysis's frozen bundle, or a bare report input."
    )
    analysis_id: str | None = Field(
        default=None, description="Reference → EvalAnalysis.id: the analysis frozen, for source `analysis` alone."
    )
    report_input: VerbatimJsonObject = Field(
        description="The report input every tool reads, as `report-input/1` JSON by alias, frozen whole."
    )
    report_input_digest: str = Field(
        min_length=1, description="sha256 of the report input as canonical JSON, re-checked on every read."
    )
    bundle: VerbatimJsonObject | None = Field(
        default=None,
        description="The campaign bundle the input was built from (`CampaignBundle.to_dict()`); None for a bare input.",
    )
    bundle_fingerprint: str | None = Field(default=None, description="The bundle's fingerprint; None for a bare input.")
    bundle_assembled_at: str | None = Field(
        default=None, description="When the bundle was assembled (ISO-8601); None for a bare input."
    )
    example_result_ids: dict[str, str] = Field(
        default_factory=dict,
        description="Each example's trial id in the report input, to the stored result it shows (for `result_get`).",
    )
    created_by: str = Field(min_length=1, description="Who froze it, as the host named the caller.")
    created_at: str = Field(default_factory=utc_now_iso, description="When it was frozen (ISO-8601).")

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_report_snapshot":
            raise ValueError(f"doc_type must be 'eval_report_snapshot', got '{v}'")
        return v

    @model_validator(mode="after")
    def check_identity(self) -> Self:
        """Refuse an id other than the one the content derives, and a source its references contradict.

        :return: This snapshot, unchanged.
        :rtype: EvalReportSnapshot
        :raises ValueError: ``id`` is not :func:`report_snapshot_id` of the campaign and the digest; a bare input
            names a campaign, an analysis or a bundle; a campaign's or an analysis's names no campaign or no bundle;
            only an analysis's names an analysis; or the bundle's fingerprint and assembly time are not set exactly
            when the bundle is.
        """
        expected = report_snapshot_id(self.campaign_id, self.report_input_digest)
        if self.id != expected:
            raise ValueError(f"a report snapshot's id is {expected!r}, not {self.id!r}")
        if self.source == "report_input":
            if self.campaign_id is not None or self.bundle is not None:
                raise ValueError("a snapshot of a bare report input names no campaign and holds no bundle")
        elif self.campaign_id is None or self.bundle is None:
            raise ValueError(
                f"a snapshot from a {self.source} names its campaign and holds the bundle it was built from"
            )
        if (self.analysis_id is not None) != (self.source == "analysis"):
            raise ValueError("a snapshot names an analysis exactly when it was frozen from one")
        held = self.bundle is not None
        if (self.bundle_fingerprint is not None) != held or (self.bundle_assembled_at is not None) != held:
            raise ValueError("a snapshot carries its bundle's fingerprint and assembly time exactly when it holds one")
        if self.example_result_ids and not held:
            raise ValueError("a snapshot names stored results for its examples only when it holds their bundle")
        return self


__all__ = [
    "FIND_SNAPSHOT",
    "FREEZE_AGAIN_REMEDY",
    "EvalReportSnapshot",
    "SnapshotSource",
    "report_snapshot_id",
]
