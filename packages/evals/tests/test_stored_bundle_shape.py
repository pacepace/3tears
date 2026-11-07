"""A stored analysis frozen over a bundle of another shape is refused, one analysis at a time, naming the regeneration.

A bundle is regenerated across a shape change, never migrated, and a stored read is strict. v0.75.0 and v0.76.0 froze
analyses over bundle shape 60, which carries fields later shapes dropped. Such an analysis is listed as needing
regeneration rather than refusing the listing of every other analysis, its line saying the shape after the retired
writer's line (every analysis frozen then is a memo), and the campaign's report reads code-only, led by the retired
writer's line (it was refused before; it is a fallback now, as a report plan's is).

An earlier shape this build reads as stored (``READ_AS_STORED``, 64 today: the bump to 65 added only optional fields,
and a shape-64 contrast's interval names its own test) is not regenerated. A report written over it renders whole on
``report_read``, the one line saying it is rendered from evidence of that shape first under the masthead and nowhere
else; ``analyses_list`` lists it as written, the line after its verdict, never as needing regeneration; and the
code-only page of the analysis by its id opens with the same line. A bundle of a shape outside the set is still
refused, as before. The shape-64 bundles are a generated report's own and the Kairo round 2 campaign's, with what 65 added taken out.
"""

from __future__ import annotations

import copy
import gzip
import json
from collections.abc import Mapping
from html import escape
from typing import Any

import pytest

from threetears.evals.analysis.bundle.schema import (
    READ_AS_STORED,
    AnalysisContextBundle,
    evidence_digest,
    refuse_a_bundle_of_another_shape,
    stored_shape_line,
)
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report.blocks import DisclosureBlock
from threetears.evals.analysis.report.build import build_code_only_report
from threetears.evals.analysis.report.campaign.answer_blocks import NO_LONGER_READS, retired_generation
from threetears.evals.analysis.report.serialize_md import report_markdown
from threetears.evals.analysis.service import analysis_report
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.ops.analysis import analyses_list, report_read
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import analysed_aspect, empty_bundle, make_analysis
from packages.evals.tests.fixtures.report_inputs.dataset import KAIRO_BUNDLE
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture, generate, ops_fixture, settled

#: The shape v0.75.0 and v0.76.0 froze analyses over.
SHIPPED_SHAPE = 60


def _frozen_at(shape: int, campaign_id: str) -> dict:
    """An aspect bundle as an older build froze it: its shape, and a field that shape carried and this one dropped."""
    bundle = empty_bundle(campaign_id=campaign_id).to_dict()
    return {**bundle, "schema_version": shape, "measurement_window_disclosure": None}


def test_a_stored_analysis_frozen_at_the_shipped_shape_is_listed_as_needing_regeneration_beside_one_that_reads():
    fixture = off_loop(ops_fixture)
    host = fixture.host.eval_host
    campaign_id = fixture.campaign.id
    readable = empty_bundle(campaign_id=campaign_id)
    current = make_analysis(
        campaign_id=campaign_id,
        scope_id=TOYHOST_SCOPE,
        aspects=[analysed_aspect(readable)],
        campaign_bundle=campaign_bundle(readable).frame(),
    )
    old_aspect = analysed_aspect(empty_bundle(campaign_id=campaign_id)).model_copy(
        update={"bundle": _frozen_at(SHIPPED_SHAPE, campaign_id)}
    )
    old = make_analysis(
        campaign_id=campaign_id,
        scope_id=TOYHOST_SCOPE,
        aspects=[old_aspect],
        campaign_bundle=campaign_bundle(readable).frame(),
    )
    assert AnalysisContextBundle.model_fields["schema_version"].default != SHIPPED_SHAPE
    host.storage.save_analysis(current)
    host.storage.save_analysis(old)

    listing = analyses_list(host, campaign_id, TOYHOST_SCOPE)

    lines = {line.id: line for line in listing.analyses}
    # Both are memos, so both lines lead with the memo writer's line; only the one frozen at the old shape says why.
    retired = retired_generation("memo")
    assert retired is not None
    assert lines[current.id].needs_regeneration == retired
    unreadable = lines[old.id].needs_regeneration
    assert unreadable is not None and unreadable.startswith(f"{retired} (")
    assert f"shape {SHIPPED_SHAPE}" in unreadable and "analysis_generate" in unreadable
    assert lines[old.id].verdict.startswith("Needs regeneration: ")


@pytest.mark.parametrize("format", ["markdown", "html", "json"])
def test_the_report_of_a_campaign_whose_newest_analysis_is_frozen_at_the_shipped_shape_reads_code_only(format: str):
    fixture = off_loop(ops_fixture)
    host = fixture.host.eval_host
    campaign_id = fixture.campaign.id
    old_aspect = analysed_aspect(empty_bundle(campaign_id=campaign_id)).model_copy(
        update={"bundle": _frozen_at(SHIPPED_SHAPE, campaign_id)}
    )
    frame = campaign_bundle(empty_bundle(campaign_id=campaign_id)).frame()
    old = make_analysis(campaign_id=campaign_id, scope_id=TOYHOST_SCOPE, aspects=[old_aspect], campaign_bundle=frame)
    host.storage.save_analysis(old)

    # A deliberate change: this read raised RefusedInput; it now returns the code-only page, as a plan's does. A memo
    # is a retired writer's, so its frozen bundle is never put back together on this door: the line names the writer.
    read = report_read(host, campaign_id, TOYHOST_SCOPE, format=format, static=(format == "html"))

    retired = retired_generation("memo")
    assert retired is not None
    assert read.basis == "code_only" and read.analysis_id == old.id
    assert read.needs_regeneration == f"{retired} (analysis {old.id})"
    assert "analysis_generate" in read.needs_regeneration
    assert (escape(retired) if format == "html" else retired) in read.body


# -- an earlier shape read as stored --------------------------------------------------------------------------------------

#: This build's aspect bundle shape.
CURRENT_SHAPE = AnalysisContextBundle.model_fields["schema_version"].default

#: The earlier shape the production report was written over, which this build reads as stored.
STORED_SHAPE = 64

#: The line a page read from shape-64 evidence says, as shipped.
STORED_LINE = (
    "Rendered from evidence assembled at shape 64; this build assembles shape 65 (a rate's contrast is now drawn on "
    "its bounded interval). Regenerate with analysis_generate to read it at 65."
)


def _at_shape(bundle: Mapping[str, Any], shape: int) -> tuple[dict[str, Any], int]:
    """An aspect bundle of this build's shape as an earlier build stored it: without the fields the bump to 65 added
    (each comparison's ``interval_basis``, a late criterion's ``aspect`` and ``reading``), at ``shape``; and how many
    comparisons lost their ``interval_basis``."""
    moved = copy.deepcopy(dict(bundle))
    dropped = 0
    for family in moved["multiple_comparisons"]["families"]:
        for comparison in family["comparisons"]:
            dropped += comparison.pop("interval_basis", None) is not None
    for late in moved["reading_scope"]["late_criteria"]:
        late.pop("aspect", None)
        late.pop("reading", None)
    moved["schema_version"] = shape
    return moved, dropped


def _stored(store: InMemoryDocumentStore, doc_id: str) -> dict[str, Any]:
    raw = store.get(doc_id, TOYHOST_SCOPE)
    assert raw is not None
    return raw


def _frozen_again_at(store: InMemoryDocumentStore, analysis: EvalAnalysis, shape: int) -> None:
    """Store the report's analysis and its snapshot as if an earlier build had frozen them at ``shape``: each aspect's
    bundle in both, and each fingerprint the analysis records over them, as that build took it."""
    assert analysis.report_doc_id is not None
    snapshot = _stored(store, _stored(store, analysis.report_doc_id)["snapshot_id"])
    aspects = [_at_shape(aspect, shape)[0] for aspect in snapshot["bundle"]["aspects"]]
    store.upsert({**snapshot, "bundle": {**snapshot["bundle"], "aspects": aspects}})
    raw = _stored(store, analysis.id)
    held = []
    for entry in raw["aspects"]:
        moved, _ = _at_shape(entry["bundle"], shape)
        held.append({**entry, "bundle": moved, "bundle_fingerprint": evidence_digest(moved)})
    frame = raw["campaign_bundle"]
    named = [
        {**entry, "fingerprint": evidence_digest(each["bundle"], like=entry["fingerprint"])}
        for entry, each in zip(frame["aspects"], held, strict=True)
    ]
    store.upsert({**raw, "aspects": held, "campaign_bundle": {**frame, "aspects": named}})


async def _written(store: InMemoryDocumentStore) -> tuple[OpsFixture, EvalAnalysis]:
    """The corpus campaign's report written and stored by ``analysis_generate``: its ``doc`` analysis, at this build's
    shape."""
    fixture = off_loop(ops_fixture, store=store)
    (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
    status = await settled(fixture.host, job.job_id)
    assert (status.state, status.status) == ("completed", "stored"), status
    assert status.analysis_id is not None
    analysis = off_loop(fixture.host.eval_host.storage.load_analysis, status.analysis_id, TOYHOST_SCOPE)
    assert analysis is not None and analysis.report_doc_id is not None
    return fixture, analysis


def test_64_is_read_as_stored_and_the_line_names_the_shape_and_what_moved():
    assert CURRENT_SHAPE == 65 and STORED_SHAPE in READ_AS_STORED
    assert stored_shape_line([CURRENT_SHAPE]) is None
    assert stored_shape_line([CURRENT_SHAPE, STORED_SHAPE]) == STORED_LINE
    refuse_a_bundle_of_another_shape({"schema_version": STORED_SHAPE}, what="this analysis", remedy="regenerate")
    with pytest.raises(RefusedInput, match="holds an evidence bundle of shape 63, and this build reads shape 65"):
        refuse_a_bundle_of_another_shape({"schema_version": 63}, what="this analysis", remedy="regenerate")


def test_the_kairo_bundle_at_shape_64_reads_as_stored_byte_for_byte_and_each_aspect_s_page_lays_out():
    # The real two-arm campaign, its contrasts carrying what 65 added, taken back to 64. Read as stored, each aspect is
    # the stored document byte for byte, so a fingerprint taken over it at 64 still names it, and its page lays out
    # over the comparisons that lack an interval basis, reading their stored verdicts.
    with gzip.open(KAIRO_BUNDLE, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    moved = [_at_shape(aspect, STORED_SHAPE) for aspect in raw["aspects"]]
    assert sum(dropped for _, dropped in moved) > 0

    bundle = CampaignBundle.from_dict({**raw, "aspects": [aspect for aspect, _ in moved]})

    assert stored_shape_line(aspect.schema_version for aspect in bundle.aspects) == STORED_LINE
    for read, (stored, _) in zip(bundle.aspects, moved, strict=True):
        assert read.schema_version == STORED_SHAPE
        assert read.to_dict() == stored and read.fingerprint() == evidence_digest(stored)
        page = report_markdown(build_code_only_report(read, assembled_at="2026-11-07T00:00:00+00:00"))
        verdicts = [
            comparison.verdict for family in read.multiple_comparisons.families for comparison in family.comparisons
        ]
        assert page and verdicts == [
            comparison["verdict"]
            for family in stored["multiple_comparisons"]["families"]
            for comparison in family["comparisons"]
        ]


def _without_line(body: str, form: str) -> str:
    if form == "html":
        return body.replace(f'\n<p class="callout" data-part="stored-shape">{escape(STORED_LINE)}</p>', "")
    return body.replace(f"_{STORED_LINE}_\n\n", "")


@pytest.mark.parametrize("form", ["markdown", "html"])
async def test_a_report_over_shape_64_evidence_renders_whole_its_line_once_and_first(form: str) -> None:
    store = InMemoryDocumentStore()
    fixture, analysis = await _written(store)
    host = fixture.host.eval_host
    static = form == "html"
    at_current = off_loop(report_read, host, fixture.campaign.id, TOYHOST_SCOPE, format=form, static=static)
    _frozen_again_at(store, analysis, STORED_SHAPE)

    read = off_loop(report_read, host, fixture.campaign.id, TOYHOST_SCOPE, format=form, static=static)

    assert read.basis == "analysis" and read.analysis_id == analysis.id
    assert read.needs_regeneration is None
    said = escape(STORED_LINE) if form == "html" else STORED_LINE
    assert read.body.count(said) == 1
    assert NO_LONGER_READS not in read.body
    # First under the masthead, before the headline the report leads with.
    lead = '<div class="bluf">' if form == "html" else "\n# "
    assert read.body.index(said) < read.body.index(lead)
    # The written report renders whole: the page at 65 with the one line added, nothing else moved.
    assert at_current.body.count(said) == 0
    assert _without_line(read.body, form) == at_current.body


async def test_a_report_over_shape_64_evidence_reads_whole_as_json_naming_its_shape() -> None:
    store = InMemoryDocumentStore()
    fixture, analysis = await _written(store)
    _frozen_again_at(store, analysis, STORED_SHAPE)

    read = off_loop(report_read, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="json")

    document = json.loads(read.body)
    assert read.basis == "analysis" and read.needs_regeneration is None and document["needs_regeneration"] is None
    assert document["report_doc"] is not None
    assert {aspect["bundle_schema_version"] for aspect in document["aspects"]} == {STORED_SHAPE}


async def test_the_listing_reads_a_report_over_shape_64_evidence_as_written_its_verdict_carrying_the_line() -> None:
    store = InMemoryDocumentStore()
    fixture, analysis = await _written(store)
    host = fixture.host.eval_host
    (before,) = off_loop(analyses_list, host, fixture.campaign.id, TOYHOST_SCOPE).analyses
    _frozen_again_at(store, analysis, STORED_SHAPE)

    (line,) = off_loop(analyses_list, host, fixture.campaign.id, TOYHOST_SCOPE).analyses

    assert line.id == analysis.id and line.needs_regeneration is None
    assert line.verdict == f"{before.verdict} {STORED_LINE}"
    assert line.verdict.count(STORED_LINE) == 1
    assert line.findings == before.findings > 0


async def test_the_analysis_read_by_its_id_over_shape_64_evidence_opens_with_the_line() -> None:
    store = InMemoryDocumentStore()
    fixture, analysis = await _written(store)
    _frozen_again_at(store, analysis, STORED_SHAPE)

    page = off_loop(analysis_report, fixture.host.eval_host.storage, analysis.id, TOYHOST_SCOPE)

    first = page.blocks[0]
    assert isinstance(first, DisclosureBlock) and first.text == STORED_LINE
    assert [block.text for block in page.blocks if isinstance(block, DisclosureBlock)].count(STORED_LINE) == 1


async def test_a_report_over_a_shape_outside_the_set_is_still_refused_as_it_was() -> None:
    store = InMemoryDocumentStore()
    fixture, analysis = await _written(store)
    host = fixture.host.eval_host
    _frozen_again_at(store, analysis, 63)

    read = off_loop(report_read, host, fixture.campaign.id, TOYHOST_SCOPE, format="markdown")
    (line,) = off_loop(analyses_list, host, fixture.campaign.id, TOYHOST_SCOPE).analyses

    assert read.basis == "code_only" and read.needs_regeneration is not None
    assert read.needs_regeneration.startswith(NO_LONGER_READS) and "shape 63" in read.needs_regeneration
    assert STORED_LINE not in read.body
    assert line.needs_regeneration is not None and "shape 63" in line.needs_regeneration
    assert line.verdict.startswith("Needs regeneration: ")
