"""``report_read`` serves the campaign's stored report document, and never refuses one it no longer reads.

Contracts held here (``threetears.evals.ops.report_read``, ``threetears.evals.analysis.campaign_report``,
``threetears.evals.analysis.service.report_doc_report``, ``threetears.evals.analysis.report_doc_store.read_report_doc``):

- **each form** of a ``doc`` analysis is its stored document's page on its own snapshot, read as a stored document is,
  its made line at its foot: Markdown and HTML are that page as its writers write it, the static page runs, links and
  loads nothing (no ``<script>``, no ``<link>``, no URL), and JSON is the campaign's JSON, every member it carried
  before, plus ``report_doc``: the stored document, its id and its generation record, with no transcript;
- **a ``plan`` analysis** reads as the code-only page with the one regenerate line, in every form, and no planner page
  is reached;
- **a document this version no longer reads** (a stale version, a model that no longer validates, a snapshot gone, a
  document deleted apart from its analysis) reads the same way, never refused;
- **the line is one constant**, which the page, ``ReportDocument.needs_regeneration`` and the listing print byte for
  byte;
- **``include_sessions``** adds each stored session, equal to the stored one, for any stored version read by
  ``report_doc_id``; a read without it, and ``list_report_docs``, read no session; a page form refuses it;
- **``report_doc_id``** of another campaign, or of none, is refused as not found;
- **``ReportDocument.doc``** replaces ``plan``;
- **the render checks** (integration): the static page of Kairo and of each misfit, its accepted recording replayed and
  stored, then read back as the report read reads a stored document, passes ``scripts/check.py`` at 1200 and 390 px,
  light and dark.
"""

from __future__ import annotations

import json
from html import escape
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

import threetears.evals.analysis.service as analysis_service
from packages.evals.tests.doc_writer_support import TOY_DOC
from packages.evals.tests.fixtures.report_plans.analyst_cli import RECORDING, REPLAY, seeded_host, snapshot_ids
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import (
    CALLER,
    TOYHOST_SCOPE,
    OpsFixture,
    generate,
    ops_fixture,
    planned_generation,
    settled,
)
from packages.evals.tests.page_contract_support import parsed, what_runs
from packages.evals.tests.report_kit_support import chromium_unavailable, render_checked
from packages.evals.tests.report_writer_support import RecordedModel, written
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.actions.engine import ReportReadParams
from threetears.evals.analysis.report.campaign.memo_blocks import NO_LONGER_READS
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report_doc_store import read_report_doc
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.report_doc import EvalReportSession
from threetears.evals.kernel.report_snapshot import report_snapshot_id
from threetears.evals.ops import JobStatus, ReportDocument, analyses_list, report_read
from threetears.evals.schema.versioning import REPORT_DOC_SCHEMA_VERSION
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.testing import RecordedTurns

#: The accepted runs: Kairo round 2 and the six misfits, each a recording replayed here.
_RUNS = Path(__file__).parent / "fixtures" / "report_docs" / "prototype4"

#: The members the campaign's JSON carried before the report document's, which every page's JSON keeps.
_CAMPAIGN_MEMBERS = {"kind", "shape", "campaign_id", "basis", "analysis_id", "recommendation", "readings", "call"}


class _ReadsCounted(InMemoryDocumentStore):
    """The reference store, keeping the doc type of every bulk read and every id read by point."""

    def __init__(self) -> None:
        super().__init__()
        self.read: list[str] = []

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **options: Any,
    ) -> list[dict[str, Any]]:
        self.read.append(doc_type)
        return super().by_doc_type(doc_type, scope_id, field_in=field_in, **options)

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        self.read.append(f"get:{doc_id}")
        return super().get(doc_id, scope_id)

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **options: Any) -> list[dict[str, Any]]:
        self.read.append(doc_type)
        return super().get_many(doc_type, doc_ids, scope_id, **options)

    def sessions_read(self) -> list[str]:
        return [read for read in self.read if read == "eval_report_session" or ":session:" in read]


async def _generated(fixture: OpsFixture) -> EvalAnalysis:
    """The corpus campaign's report written and stored by ``analysis_generate``: its ``doc`` analysis."""
    (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
    status: JobStatus = await settled(fixture.host, job.job_id)
    assert (status.state, status.status) == ("completed", "stored"), status
    assert status.analysis_id is not None
    analysis = off_loop(fixture.host.eval_host.storage.load_analysis, status.analysis_id, TOYHOST_SCOPE)
    assert analysis is not None and analysis.report_doc_id is not None
    return analysis


def _read(fixture: OpsFixture, form: Any = "markdown", **asked: Any) -> ReportDocument:
    return off_loop(report_read, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format=form, **asked)


# -- each form ----------------------------------------------------------------------------------------------------------


async def test_markdown_and_html_are_the_stored_document_s_page_on_its_snapshot_with_its_made_line() -> None:
    fixture = off_loop(ops_fixture)
    analysis = await _generated(fixture)
    assert analysis.report_doc_id is not None
    stored = off_loop(read_report_doc, fixture.host.eval_host.storage, analysis.report_doc_id, TOYHOST_SCOPE)

    markdown, html = _read(fixture, "markdown"), _read(fixture, "html")

    for read in (markdown, html):
        assert (read.basis, read.analysis_id, read.analysis_basis) == ("analysis", analysis.id, "doc")
        assert read.doc == stored.record.doc and read.needs_regeneration is None
    assert markdown.body == doc_page_markdown(stored.page)
    assert html.body == doc_page_html(stored.page)
    assert stored.page.made == stored.made and stored.made.startswith("Written by ")
    assert stored.made in markdown.body and TOY_DOC["headline"] in markdown.body
    assert 'data-page="doc"' in html.body and 'data-part="made"' in html.body
    assert parsed(html.body).tags[-1] == ("script", []), "the default page ends in the kit's script"
    assert "A report was written over this campaign's evidence" not in markdown.body, "the interim line is gone"


async def test_the_static_page_has_no_script_no_link_and_no_url() -> None:
    fixture = off_loop(ops_fixture)
    analysis = await _generated(fixture)
    assert analysis.report_doc_id is not None
    stored = off_loop(read_report_doc, fixture.host.eval_host.storage, analysis.report_doc_id, TOYHOST_SCOPE)

    static = _read(fixture, "html", static=True).body

    assert static == doc_page_html(stored.page, static=True)
    tags = [tag for tag, _ in parsed(static).tags]
    assert what_runs(static) == []
    assert "script" not in tags and "link" not in tags
    assert "<script" not in static and "<link" not in static
    assert "://" not in static and "url(" not in "".join(static.split()).lower(), "nothing a browser would fetch"
    assert 'data-page="doc"' in static


async def test_json_keeps_every_member_it_carried_and_gains_the_report_doc_without_a_transcript() -> None:
    fixture = off_loop(ops_fixture)
    code_only = json.loads(_read(fixture, "json").body)
    analysis = await _generated(fixture)

    read = _read(fixture, "json")
    document = json.loads(read.body)

    assert set(document) == set(code_only) == _CAMPAIGN_MEMBERS | {"aspects", "report_doc", "needs_regeneration"}
    assert code_only["report_doc"] is None, "a page with no stored document carries none"
    assert (code_only["needs_regeneration"], document["needs_regeneration"]) == (None, None)
    assert (document["kind"], document["basis"], document["analysis_id"]) == ("campaign", "analysis", analysis.id)
    assert document["readings"]["aspects"] == code_only["readings"]["aspects"]
    served = document["report_doc"]
    assert set(served) == {"id", "snapshot_id", "created_at", "ending", "doc", "findings", "made", "generation"}
    assert (served["id"], served["doc"]) == (analysis.report_doc_id, read.doc)
    sessions = served["generation"]["sessions"]
    assert sessions and all(set(entry) == {"digest", "turns", "ref"} for entry in sessions.values()), (
        "the generation record names each session by ref, never its transcript"
    )
    assert "sessions" not in served


# -- what no longer reads -----------------------------------------------------------------------------------------------


def _code_only_with_the_line(
    fixture: OpsFixture, analysis_id: str | None, basis: str | None, **asked: Any
) -> list[ReportDocument]:
    """Every form of the report is the code-only page with the one regenerate line, naming what it stands in for:
    the campaign's analysis, or the report document asked for by its id. Never refused."""
    named = analysis_id if analysis_id is not None else asked["report_doc_id"]
    said = (
        f"This campaign's analysis {analysis_id} was {NO_LONGER_READS}."
        if analysis_id is not None
        else f"Report document {named} was {NO_LONGER_READS}."
    )
    reads = []
    for form, static in (("markdown", False), ("html", False), ("html", True), ("json", False)):
        read = _read(fixture, form, static=static, **asked)
        assert (read.basis, read.analysis_id, read.analysis_basis) == ("code_only", analysis_id, basis), form
        assert read.doc is None and read.needs_regeneration is not None, form
        assert read.needs_regeneration.startswith(f"{NO_LONGER_READS} ("), form
        if form != "json":
            assert (escape(said) if form == "html" else said) in read.body, form
            assert NO_LONGER_READS in read.body, form
            assert "No recommendation was written" not in read.body, form
        else:
            body = json.loads(read.body)
            assert body["report_doc"] is None and body["needs_regeneration"] == read.needs_regeneration
        reads.append(read)
    return reads


async def test_a_plan_analysis_reads_code_only_with_the_line_and_no_planner_page_is_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = off_loop(ops_fixture)
    planned, _ = await planned_generation(fixture)

    def planner(*_: Any, **__: Any) -> Any:
        raise AssertionError("report_read reached the planner's page")

    monkeypatch.setattr(analysis_service, "plan_page", planner)
    monkeypatch.setattr(analysis_service, "get_report_plan", planner)

    _code_only_with_the_line(fixture, planned.id, "plan")
    assert 'data-page="campaign"' in _read(fixture, "html").body, "the code-only page, not the plan's"


def _stored_record(store: InMemoryDocumentStore, report_doc_id: str) -> dict[str, Any]:
    raw = store.get(report_doc_id, TOYHOST_SCOPE)
    assert raw is not None
    return raw


#: Every way a stored document stops reading: in its record, in its snapshot, or in its snapshot's bundle.
_UNREAD = [
    "stale_version",
    "model_moved",
    "generation_moved",
    "snapshot_gone",
    "snapshot_unreadable",
    "input_moved",
    "no_bundle",
    "bundle_moved",
    "doc_deleted",
]

#: Why each reads code-only, as the page's needs_regeneration says after the line.
_WHY = {
    "stale_version": "does not read under this version",
    "model_moved": "does not read under this version",
    "generation_moved": "carries a generation record this version does not read",
    "snapshot_gone": "which is gone",
    "snapshot_unreadable": "which does not read",
    "input_moved": "was written from a snapshot that no longer reads",
    "no_bundle": "holds no bundle",
    "bundle_moved": "report snapshot",
    "doc_deleted": "not found",
}


def _no_longer_reads(store: InMemoryDocumentStore, fixture: OpsFixture, report_doc_id: str, unread: str) -> None:
    """Make the stored document ``report_doc_id`` one this version no longer reads, as ``unread`` names."""
    storage = fixture.host.eval_host.storage
    record = _stored_record(store, report_doc_id)
    snapshot = _stored_record(store, record["snapshot_id"])
    match unread:
        case "stale_version":
            store.upsert({**record, "schema_version": REPORT_DOC_SCHEMA_VERSION - 1})
        case "model_moved":
            store.upsert({**record, "doc": {**record["doc"], "claims": "no longer a list"}})
        case "generation_moved":
            store.upsert({**record, "generation": {**record["generation"], "format": 1}})
        case "snapshot_gone":
            assert off_loop(storage.delete_report_snapshot, record["snapshot_id"], TOYHOST_SCOPE)
        case "snapshot_unreadable":
            store.upsert({**snapshot, "source": "no longer a source"})
        case "input_moved":
            store.upsert({**snapshot, "report_input": {**snapshot["report_input"], "moved": True}})
        case "no_bundle":
            # The same input frozen bare, which holds no bundle, and the document pointed at it.
            bare_id = report_snapshot_id(None, snapshot["report_input_digest"])
            bare = {
                **snapshot,
                "id": bare_id,
                "campaign_id": None,
                "source": "report_input",
                "analysis_id": None,
                "bundle": None,
                "bundle_fingerprint": None,
                "bundle_assembled_at": None,
                "example_result_ids": {},
            }
            store.upsert(bare)
            store.upsert({**record, "snapshot_id": bare_id})
        case "bundle_moved":
            store.upsert({**snapshot, "bundle": {**snapshot["bundle"], "schema_version": 999}})
        case "doc_deleted":
            assert off_loop(storage.delete_report_doc, report_doc_id, TOYHOST_SCOPE)


@pytest.mark.parametrize("unread", _UNREAD)
async def test_a_document_that_no_longer_reads_gets_the_code_only_page_and_the_line_never_a_refusal(
    unread: str,
) -> None:
    store = InMemoryDocumentStore()
    fixture = off_loop(ops_fixture, store=store)
    analysis = await _generated(fixture)
    assert analysis.report_doc_id is not None
    _no_longer_reads(store, fixture, analysis.report_doc_id, unread)

    (said, *_) = _code_only_with_the_line(fixture, analysis.id, "doc")

    assert said.needs_regeneration is not None and _WHY[unread] in said.needs_regeneration


@pytest.mark.parametrize("unread", [name for name in _UNREAD if name != "doc_deleted"])
async def test_a_document_read_by_its_id_that_no_longer_reads_gets_the_code_only_page_naming_it(unread: str) -> None:
    store = InMemoryDocumentStore()
    fixture = off_loop(ops_fixture, store=store)
    older = await _generated(fixture)
    await _generated(fixture)
    assert older.report_doc_id is not None
    _no_longer_reads(store, fixture, older.report_doc_id, unread)

    (said, *_) = _code_only_with_the_line(fixture, None, None, report_doc_id=older.report_doc_id)

    assert said.needs_regeneration is not None and _WHY[unread] in said.needs_regeneration


async def test_a_stand_in_s_json_never_reads_as_a_campaign_never_analysed() -> None:
    fixture = off_loop(ops_fixture)
    never = json.loads(_read(fixture, "json").body)
    planned, _ = await planned_generation(fixture)

    stands_in = json.loads(_read(fixture, "json").body)

    assert never["needs_regeneration"] is None
    assert (
        stands_in["needs_regeneration"].startswith(f"{NO_LONGER_READS} (")
        and planned.id in (stands_in["needs_regeneration"])
    )
    assert stands_in != never


# -- one line ------------------------------------------------------------------------------------------------------------


async def test_the_line_is_one_constant_every_surface_prints_byte_for_byte() -> None:
    fixture = off_loop(ops_fixture)
    planned, _ = await planned_generation(fixture)
    evals = eval_catalogue().mount_all(standard_tools())[0]

    (line,) = off_loop(analyses_list, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
    called = await evals.call(
        {"action": "report_read", "campaign_id": fixture.campaign.id, "format": "markdown"},
        host=fixture.host,
        caller=CALLER,
    )

    assert NO_LONGER_READS == "written in a shape this version no longer reads; regenerate with analysis_generate"
    assert line.id == planned.id and line.needs_regeneration == NO_LONGER_READS
    assert called.structured is not None and called.structured["needs_regeneration"].startswith(NO_LONGER_READS)
    for form in ("markdown", "html"):
        assert _read(fixture, form).body.count(NO_LONGER_READS) == 1, form
    assert NO_LONGER_READS in called.text


# -- transcripts -----------------------------------------------------------------------------------------------------------


async def test_include_sessions_returns_every_stored_session_of_any_version_by_report_doc_id() -> None:
    store = _ReadsCounted()
    fixture = off_loop(ops_fixture, store=store)
    first = await _generated(fixture)
    newest = await _generated(fixture)
    assert first.report_doc_id is not None and newest.report_doc_id is not None
    storage = fixture.host.eval_host.storage

    read = _read(fixture, "json", report_doc_id=first.report_doc_id, include_sessions=True)
    served = json.loads(read.body)["report_doc"]

    assert served["id"] == first.report_doc_id != newest.report_doc_id, "the version asked for, not the newest"
    assert read.analysis_id is None and read.basis == "analysis"
    stored = off_loop(storage.load_report_sessions, first.report_doc_id, TOYHOST_SCOPE)
    assert set(served["sessions"]) == set(stored) == set(served["generation"]["sessions"])
    for name, session in served["sessions"].items():
        assert EvalReportSession.model_validate(session) == stored[name]

    store.read.clear()
    without = json.loads(_read(fixture, "json", report_doc_id=first.report_doc_id).body)["report_doc"]
    assert "sessions" not in without and store.sessions_read() == [], "the page's read loads no session"
    store.read.clear()
    listed = off_loop(storage.list_report_docs, TOYHOST_SCOPE, campaign_id=fixture.campaign.id)
    assert {record.id for record in listed} == {first.report_doc_id, newest.report_doc_id}
    assert store.read == ["eval_report_doc"], "a listing of report documents loads no session"


async def test_include_sessions_reads_the_newest_document_s_sessions_and_a_page_form_refuses_it() -> None:
    fixture = off_loop(ops_fixture)
    analysis = await _generated(fixture)
    assert analysis.report_doc_id is not None

    served = json.loads(_read(fixture, "json", include_sessions=True).body)["report_doc"]
    assert served["id"] == analysis.report_doc_id and served["sessions"]

    for form in ("markdown", "html"):
        with pytest.raises(ValidationFailedError, match="only the json form carries") as refused:
            _read(fixture, form, include_sessions=True)
        assert "format='json'" in (refused.value.remedy or "")


async def test_a_report_doc_id_of_another_campaign_or_of_none_is_refused_as_not_found() -> None:
    store = InMemoryDocumentStore()
    fixture = off_loop(ops_fixture, store=store)
    analysis = await _generated(fixture)
    assert analysis.report_doc_id is not None
    record = _stored_record(store, analysis.report_doc_id)
    elsewhere = {**record, "id": "another-report", "campaign_id": "elsewhere"}
    store.upsert(elsewhere)

    for report_doc_id in ("another-report", "no-such-report"):
        with pytest.raises(NotFoundError, match="report document"):
            _read(fixture, "markdown", report_doc_id=report_doc_id)
    # The campaign is checked as soon as the record loads, before its snapshot: one gone is still not this campaign's.
    assert off_loop(fixture.host.eval_host.storage.delete_report_snapshot, record["snapshot_id"], TOYHOST_SCOPE)
    with pytest.raises(NotFoundError, match="report document 'another-report'"):
        _read(fixture, "markdown", report_doc_id="another-report")


async def test_an_older_document_read_by_its_id_reads_as_written_after_the_campaign_moved_on() -> None:
    """#977's audit read: a version written before an aspect was relabelled is laid out over the declaration it was
    written against, so it reads as its own page, never as one to regenerate."""
    fixture = off_loop(ops_fixture)
    older = await _generated(fixture)
    assert older.report_doc_id is not None
    storage = fixture.host.eval_host.storage
    campaign = off_loop(storage.load_campaign, fixture.campaign.id, TOYHOST_SCOPE)
    assert campaign is not None
    (aspect,) = campaign.aspects
    moved = campaign.model_copy(update={"aspects": [aspect.model_copy(update={"label": "renamed_after_v1"})]})
    off_loop(storage.save_campaign, moved)

    for form in ("markdown", "json"):
        read = _read(fixture, form, report_doc_id=older.report_doc_id)
        assert (read.basis, read.analysis_id, read.needs_regeneration) == ("analysis", None, None), form
        assert read.doc is not None and read.doc["headline"] == TOY_DOC["headline"]
    document = json.loads(_read(fixture, "json", report_doc_id=older.report_doc_id).body)
    assert document["report_doc"]["id"] == older.report_doc_id
    assert [fold["aspect"] for fold in document["aspects"]] == [aspect.label], "laid out as the document was written"


# -- every trial's record --------------------------------------------------------------------------------------------------


async def test_trials_all_lays_the_folds_after_the_document_which_reads_as_it_was_stored() -> None:
    fixture = off_loop(ops_fixture)
    analysis = await _generated(fixture)
    assert analysis.report_doc_id is not None
    stored = off_loop(read_report_doc, fixture.host.eval_host.storage, analysis.report_doc_id, TOYHOST_SCOPE)

    for static in (False, True):
        table = _read(fixture, "html", static=static).body
        everything = _read(fixture, "html", static=static, trials="all").body
        document, marker = doc_page_html(stored.page, static=static), '<details class="section" data-part="evidence">'
        assert table == document and marker not in table, "the table read lays the document out alone"
        assert everything.split(marker)[0] == document.split("</article>")[0], "the document above it, unchanged"
        record = everything.split(marker)[1]
        assert 'id="aspect-' in record and '<details class="trial"' in record, (
            "each aspect's fold, its trials' records in it"
        )
    markdown = _read(fixture, "markdown", trials="all").body
    assert markdown.startswith(doc_page_markdown(stored.page).rstrip("\n")) and "\n## Evidence record\n" in markdown
    assert _read(fixture, "json", trials="all").body == _read(fixture, "json").body, "JSON carries no records"


# -- the action and the result ----------------------------------------------------------------------------------------------


async def test_the_action_takes_report_doc_id_and_include_sessions() -> None:
    assert (
        ReportReadParams.model_validate({"campaign_id": "c"}).report_doc_id,
        ReportReadParams.model_validate({"campaign_id": "c"}).include_sessions,
    ) == (None, False)
    fixture = off_loop(ops_fixture)
    analysis = await _generated(fixture)
    evals = eval_catalogue().mount_all(standard_tools())[0]

    called = await evals.call(
        {
            "action": "report_read",
            "campaign_id": fixture.campaign.id,
            "format": "json",
            "report_doc_id": analysis.report_doc_id,
            "include_sessions": True,
        },
        host=fixture.host,
        caller=CALLER,
    )

    assert not called.is_error, called.text
    assert json.loads(called.text)["report_doc"]["sessions"], "the transcripts reach the caller"


def test_the_result_carries_the_document_as_doc_and_no_plan() -> None:
    fields = ReportDocument.model_fields
    assert "doc" in fields and "plan" not in fields
    assert fields["analysis_basis"].description is not None and "`doc`" in fields["analysis_basis"].description


# -- the accepted runs, read back -----------------------------------------------------------------------------------------

#: Every accepted run, by its directory name.
_ACCEPTED = sorted(path.parent.name for path in _RUNS.glob(f"*/{REPLAY}"))


async def _read_back(name: str) -> Any:
    """The accepted run's recording replayed and stored, then read back as the report read reads a stored document."""
    replay = json.loads((_RUNS / name / REPLAY).read_text())
    host = off_loop(seeded_host)
    result = await written(
        RecordedModel(RecordedTurns.replay(_RUNS / name / RECORDING), replay["model"]),
        snapshot=snapshot_ids()[replay["snapshot_of"]],
        budget_usd=replay["budget_usd"],
        withhold=replay["withhold"],
        host=host,
    )
    assert result.stored is not None, f"{name}: the replay ended {result.ending} and stored nothing"
    return off_loop(read_report_doc, host.eval_host.storage, result.stored.id, CALLER.scope_id)


@pytest.mark.parametrize("name", _ACCEPTED)
async def test_each_accepted_document_reads_back_whole_and_its_static_page_runs_and_loads_nothing(name: str) -> None:
    stored = await _read_back(name)

    assert stored.page.degraded == "", f"{name}: the page read back degraded, so it no longer reads as it was stored"
    static = doc_page_html(stored.page, static=True)
    assert what_runs(static) == [] and "<script" not in static and "<link" not in static and "://" not in static


@pytest.mark.integration
@pytest.mark.parametrize("name", _ACCEPTED)
async def test_each_accepted_document_s_static_page_as_the_report_read_reads_it_passes_the_render_checks(
    name: str, tmp_path: Path
) -> None:
    """``scripts/check.py`` at 1200 and 390 px, light and dark, on the static page of each accepted run, replayed and
    stored, then read back as the report read reads a stored document (``read_report_doc``, then the static writer
    ``campaign_report_html`` calls for a document's page)."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = tmp_path / f"{name}.html"
    page.write_text(doc_page_html((await _read_back(name)).page, static=True), encoding="utf-8")

    checked, runs = render_checked(page, tmp_path / "screenshots")

    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert len(runs) == 4 and all(line.startswith("ok") for line in runs), checked.stdout
