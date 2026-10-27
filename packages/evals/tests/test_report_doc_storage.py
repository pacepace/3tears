"""The stored report document (``eval_report_doc``): it round-trips, lists newest first, is never written over, is
swept with the derived documents, and is refused on read after a regenerable bump."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import make_campaign, memory_storage
from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from threetears.evals.kernel.errors import ConflictError
from threetears.evals.kernel.report_doc import EvalReportDoc, ReportDoc
from threetears.evals.schema.doc_types import DOC_TYPE_FACTS
from threetears.evals.schema.versioning import REGENERABLE_SCHEMA_VERSION

_SCOPE = "uni-1"
_KAIRO = "prototype3/kairo"

#: A tally as the writing loop records one; the store keeps it as written.
_TALLY: dict[str, Any] = {"turns": [{"role": "analyst", "round": 1, "cost_usd": None}], "bounces": 0}


def _stored(name: str = _KAIRO, **fields: Any) -> EvalReportDoc:
    snapshot = snapshot_of(name)
    values: dict[str, Any] = {
        "scope_id": _SCOPE,
        "snapshot_id": snapshot.id,
        "campaign_id": snapshot.campaign_id,
        "doc": converted_doc(name),
        "ending": "stored_with_findings",
        "findings": [{"claim": "c4", "severity": "should_fix", "finding": "The off-lane share is a lead."}],
        "generation": _TALLY,
        "created_by": "agent:test",
    }
    return EvalReportDoc(**(values | fields))


def test_a_report_document_is_derived_and_regenerable() -> None:
    """It is computed from its snapshot, so writing the report again remakes it and a regenerable bump refuses it."""
    facts = DOC_TYPE_FACTS["eval_report_doc"]
    assert (facts.origin, facts.lifecycle) == ("derived", "regenerable")


def test_a_report_document_round_trips_through_the_store() -> None:
    """What is saved is what loads: the document, its findings, its tally and its references, unchanged."""
    storage, _ = memory_storage()
    report = _stored()

    storage.save_report_doc(report)
    loaded = storage.load_report_doc(report.id, _SCOPE)

    assert loaded == report
    assert ReportDoc.model_validate(loaded.doc).recommends == ["September"]
    assert loaded.generation == _TALLY
    assert storage.load_report_doc(report.id, "another-scope") is None


def test_a_stored_report_document_is_never_written_over() -> None:
    """Writing a report again stores a new document beside the old one; the same id twice is a conflict."""
    storage, _ = memory_storage()
    report = _stored()
    storage.save_report_doc(report)

    with pytest.raises(ConflictError):
        storage.save_report_doc(report)


def test_report_documents_list_newest_first_by_snapshot_and_by_campaign() -> None:
    """Every document of a scope lists newest first, and a snapshot or a campaign narrows the list."""
    storage, _ = memory_storage()
    first = _stored(created_at="2026-10-27T10:00:00+00:00")
    again = _stored(created_at="2026-10-27T11:00:00+00:00")
    other = _stored("prototype3/r1", ending="stored", findings=[], created_at="2026-10-27T12:00:00+00:00")
    for report in (first, again, other):
        storage.save_report_doc(report)

    assert [r.id for r in storage.list_report_docs(_SCOPE)] == [other.id, again.id, first.id]
    assert [r.id for r in storage.list_report_docs(_SCOPE, snapshot_id=first.snapshot_id)] == [again.id, first.id]
    assert first.campaign_id is not None
    assert [r.id for r in storage.list_report_docs(_SCOPE, campaign_id=first.campaign_id)] == [again.id, first.id]


def test_a_report_document_deletes_raw() -> None:
    """Delete reports whether one was there, and reads nothing first."""
    storage, _ = memory_storage()
    report = _stored()
    storage.save_report_doc(report)

    assert storage.delete_report_doc(report.id, _SCOPE) is True
    assert storage.delete_report_doc(report.id, _SCOPE) is False
    assert storage.load_report_doc(report.id, _SCOPE) is None


def test_drop_derived_sweeps_report_documents_with_their_snapshots_and_keeps_the_campaign() -> None:
    """The operator step after a regenerable bump deletes every stored report document, and nothing kept."""
    storage, _ = memory_storage()
    report = _stored()
    storage.save_report_snapshot(snapshot_of(_KAIRO))
    storage.save_report_doc(report)
    campaign = make_campaign(scope_id=_SCOPE)
    storage.save_campaign(campaign)

    dropped = storage.drop_derived([_SCOPE])

    assert dropped.counts["eval_report_doc"] == 1
    assert dropped.counts["eval_report_snapshot"] == 1
    assert storage.load_report_doc(report.id, _SCOPE) is None
    assert storage.list_report_docs(_SCOPE) == []
    assert storage.load_campaign(campaign.id, _SCOPE) is not None


def test_a_report_document_written_before_a_regenerable_bump_is_refused_on_read() -> None:
    """A document under another regenerable version is refused, never read as this build's, and is still swept."""
    storage, store = memory_storage()
    old = _stored().to_dict() | {"schema_version": REGENERABLE_SCHEMA_VERSION - 1}
    store.upsert(old)

    with pytest.raises(ValidationError, match="schema_version"):
        storage.load_report_doc(old["id"], _SCOPE)
    assert storage.drop_derived([_SCOPE]).counts["eval_report_doc"] == 1


def test_a_stored_document_in_a_shape_this_build_no_longer_reads_is_refused_on_read() -> None:
    """The document is checked against the model on every read, so a stored one that moved shape is refused."""
    storage, store = memory_storage()
    stored = _stored().to_dict()
    stored["doc"]["claims"][0]["evidence"] = []
    store.upsert(stored)

    with pytest.raises(ValidationError, match="at least 1 item"):
        storage.load_report_doc(stored["id"], _SCOPE)


@pytest.mark.parametrize(
    ("fields", "refusal"),
    [
        ({"findings": [{"claim": "c9", "severity": "should_fix", "finding": "No such claim."}]}, "claim 'c9'"),
        (
            {"ending": "stored", "findings": [{"claim": "", "severity": "must_fix", "finding": "Wrong call."}]},
            "no must_fix finding",
        ),
        ({"ending": "stored_with_findings", "findings": []}, "carries the findings left open"),
        ({"findings": [{"claim": "c1", "severity": "nit", "finding": "Tighten it."}]}, "severity"),
        ({"ending": "refused"}, "ending"),
        ({"doc": {"schema": "report-plan/1"}}, "doc"),
    ],
    ids=[
        "finding on an unknown claim",
        "stored with a must_fix",
        "stored with findings but none",
        "a finding out of shape",
        "unstored ending",
        "not a report document",
    ],
)
def test_a_stored_document_that_contradicts_itself_is_refused(fields: dict[str, Any], refusal: str) -> None:
    """A finding names a claim the document holds, and the ending agrees with the findings left open."""
    with pytest.raises(ValidationError, match=refusal):
        _stored(**fields)


def test_a_page_wide_finding_names_no_claim() -> None:
    """A finding about the page as a whole names no claim, and is kept."""
    report = _stored(findings=[{"claim": "", "severity": "should_fix", "finding": "The flag repeats the lede."}])

    assert report.findings == [{"claim": "", "severity": "should_fix", "finding": "The flag repeats the lede."}]
