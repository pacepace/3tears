"""A stored analysis frozen over a bundle of another shape is refused, one analysis at a time, naming the regeneration.

A bundle is regenerated across a shape change, never migrated, and a stored read is strict. v0.75.0 and v0.76.0 froze
analyses over bundle shape 60, which carries fields later shapes dropped. Such an analysis is listed as needing
regeneration rather than refusing the listing of every other analysis, its line saying the shape after the retired
writer's line (every analysis frozen then is a memo), and the campaign's report reads code-only, led by the retired
writer's line (it was refused before; it is a fallback now, as a report plan's is).

No earlier shape is read as stored since shape 66 took names out of the evidence (owner ruling, 2026-11-11): a bundle
frozen before it carries sentences and arm labels worded when it was frozen, which this build words and names when a
page is rendered. 64 was read as stored until then; now a stored analysis frozen at 64 or 65 is regenerated once, and
every door says why: ``report_read`` stands in code-only, ``analyses_list`` and ``campaigns_list`` list it as needing
regeneration, ``analysis_report`` by its id refuses it, and a call, a snapshot freeze and a writer case mint over it are
refused, each naming the bundle as frozen before names left the evidence. The shape-65 bundles are a generated
report's own, with the words 66 dropped put back, and the Kairo round 2 campaign's.
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
    AnalysisContextBundle,
    evidence_digest,
    refuse_a_bundle_of_another_shape,
)
from threetears.evals.analysis.campaign_bundle import CAMPAIGN_BUNDLE_SHAPE, CampaignBundle
from threetears.evals.analysis.report.build import analysis_campaign_bundle, build_code_only_report
from threetears.evals.analysis.report.campaign.answer_blocks import NO_LONGER_READS, retired_generation
from threetears.evals.analysis.report.serialize_md import report_markdown
from threetears.evals.analysis.service import analysis_report, record_campaign_call
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.errors import RefusedInput, ValidationFailedError
from threetears.evals.ops.analysis import analyses_list, campaigns_list, report_read
from threetears.evals.ops.analyst import report_snapshot_freeze
from threetears.evals.ops.report_writer.writer_bank import WriterCaseMint, writer_case_mint
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import analysed_aspect, empty_bundle, make_analysis
from packages.evals.tests.fixtures.report_inputs.dataset import KAIRO_BUNDLE
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, generate, ops_fixture, settled
from packages.evals.tests.writer_host_support import WRITER_TEMPLATE

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


# -- frozen before names left the evidence --------------------------------------------------------------------------

#: This build's aspect bundle shape.
CURRENT_SHAPE = AnalysisContextBundle.model_fields["schema_version"].default

#: The shape every analysis stored before names left the evidence was frozen over, and the one before it, which this
#: build read as stored until then.
NAMED_SHAPE, LAST_READ_AS_STORED = 65, 64

#: What every door says of a bundle frozen before names left the evidence.
WHY = "frozen before names left the evidence"


def _at_shape(bundle: Mapping[str, Any], shape: int) -> dict[str, Any]:
    """An aspect bundle of this build's shape as a build before 66 stored it: at ``shape``, with the sentences that
    build worded into it (the pair scan's completeness, each lockstep group's sentence)."""
    moved = copy.deepcopy(dict(bundle))
    if isinstance(pairs := moved.get("factor_pairs"), dict):
        pairs["completeness"] = f"{len(pairs['factors'])} factor varied, so no pair of factors could co-vary."
    for group in moved.get("aliased_factors") or []:
        group["sentence"] = f"{' and '.join(group['factors'])} move together."
    moved["schema_version"] = shape
    return moved


def _stored(store: InMemoryDocumentStore, doc_id: str) -> dict[str, Any]:
    raw = store.get(doc_id, TOYHOST_SCOPE)
    assert raw is not None
    return raw


def _frozen_again_at(store: InMemoryDocumentStore, analysis: EvalAnalysis, shape: int) -> None:
    """Store the report's analysis and its snapshot as if an earlier build had frozen them at ``shape``: each aspect's
    bundle in both, and each fingerprint the analysis records over them, as that build took it."""
    assert analysis.report_doc_id is not None
    snapshot = _stored(store, _stored(store, analysis.report_doc_id)["snapshot_id"])
    aspects = [_at_shape(aspect, shape) for aspect in snapshot["bundle"]["aspects"]]
    store.upsert({**snapshot, "bundle": {**snapshot["bundle"], "aspects": aspects}})
    raw = _stored(store, analysis.id)
    held = []
    for entry in raw["aspects"]:
        moved = _at_shape(entry["bundle"], shape)
        held.append({**entry, "bundle": moved, "bundle_fingerprint": evidence_digest(moved)})
    frame = raw["campaign_bundle"]
    named = [
        {**entry, "fingerprint": evidence_digest(each["bundle"])}
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


@pytest.mark.parametrize("shape", [LAST_READ_AS_STORED, NAMED_SHAPE])
def test_no_earlier_shape_is_read_as_stored_and_one_frozen_before_names_left_the_evidence_says_so(shape: int):
    # A deliberate change (owner ruling, 2026-11-11): 64 was read as stored; every shape before 66 is now refused.
    assert CURRENT_SHAPE == 66
    refuse_a_bundle_of_another_shape({"schema_version": CURRENT_SHAPE}, what="this analysis", remedy="regenerate")
    with pytest.raises(RefusedInput) as refused:
        refuse_a_bundle_of_another_shape({"schema_version": shape}, what="this analysis", remedy="regenerate")
    assert f"shape {shape}, {WHY}" in refused.value.fault and f"(shape {CURRENT_SHAPE})" in refused.value.fault
    assert refused.value.remedy == "regenerate"


@pytest.mark.parametrize("shape", [2, 3])
def test_a_campaign_bundle_frozen_before_names_left_the_readings_is_refused_saying_so(shape: int) -> None:
    """Its floors' reasons quoted measures' reader names and its derived arms carried no spelling: shape 2 was read as
    3, and 3 as itself, until the readings dropped every name (campaign bundle shape 4). Both are refused now, whole or
    as a frame, naming why, and what holds one is regenerated."""
    assert CAMPAIGN_BUNDLE_SHAPE == 4
    current = campaign_bundle(empty_bundle(campaign_id="c"))
    assert CampaignBundle.from_dict(current.to_dict()) == current
    for read in (
        lambda: CampaignBundle.from_dict({**current.to_dict(), "schema_version": shape}),
        lambda: CampaignBundle.from_frame({**current.frame(), "schema_version": shape}, current.aspects),
    ):
        with pytest.raises(RefusedInput, match="frozen before names left the readings") as refused:
            read()
        assert f"shape {shape}" in refused.value.fault and f"({CAMPAIGN_BUNDLE_SHAPE})" in refused.value.fault


def test_the_kairo_bundle_at_shape_65_is_refused_as_frozen_before_names_left_the_evidence():
    # The real two-arm campaign, taken back to 65 with the sentence 66 dropped: it no longer reads, where the same
    # bundle at 66 reads and lays out.
    with gzip.open(KAIRO_BUNDLE, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    current = CampaignBundle.from_dict(raw)
    assert {aspect.schema_version for aspect in current.aspects} == {CURRENT_SHAPE}
    for aspect in current.aspects:
        assert report_markdown(build_code_only_report(aspect, assembled_at="2026-11-11T00:00:00+00:00"))

    with pytest.raises(RefusedInput, match=WHY):
        CampaignBundle.from_dict({**raw, "aspects": [_at_shape(aspect, NAMED_SHAPE) for aspect in raw["aspects"]]})


@pytest.mark.parametrize("form", ["markdown", "html", "json"])
async def test_a_report_over_shape_65_evidence_stands_in_code_only_saying_why(form: str) -> None:
    store = InMemoryDocumentStore()
    fixture, analysis = await _written(store)
    _frozen_again_at(store, analysis, NAMED_SHAPE)

    read = off_loop(
        report_read, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format=form, static=form == "html"
    )

    assert read.basis == "code_only" and read.needs_regeneration is not None
    assert read.needs_regeneration.startswith(NO_LONGER_READS)
    assert WHY in read.needs_regeneration and f"shape {NAMED_SHAPE}" in read.needs_regeneration


async def test_the_listings_list_a_report_over_shape_65_evidence_as_needing_regeneration_saying_why() -> None:
    store = InMemoryDocumentStore()
    fixture, analysis = await _written(store)
    host = fixture.host.eval_host
    _frozen_again_at(store, analysis, NAMED_SHAPE)

    (line,) = off_loop(analyses_list, host, fixture.campaign.id, TOYHOST_SCOPE).analyses
    (campaign,) = [
        one for one in off_loop(campaigns_list, host, TOYHOST_SCOPE).campaigns if one.id == fixture.campaign.id
    ]

    assert line.id == analysis.id and line.needs_regeneration is not None and WHY in line.needs_regeneration
    assert line.verdict.startswith("Needs regeneration: ")
    assert campaign.report_state == "needs_regeneration"
    assert campaign.report is not None and campaign.report.why_regenerate is not None
    assert WHY in campaign.report.why_regenerate


async def test_the_analysis_read_by_its_id_over_shape_65_evidence_is_refused_saying_why() -> None:
    store = InMemoryDocumentStore()
    fixture, analysis = await _written(store)
    _frozen_again_at(store, analysis, NAMED_SHAPE)

    with pytest.raises(RefusedInput) as refused:
        off_loop(analysis_report, fixture.host.eval_host.storage, analysis.id, TOYHOST_SCOPE)

    assert WHY in refused.value.fault and "analysis_generate" in (refused.value.remedy or "")


async def test_a_call_a_freeze_and_a_writer_case_over_a_shape_65_analysis_are_refused_saying_why() -> None:
    store = InMemoryDocumentStore()
    fixture, analysis = await _written(store)
    host = fixture.host.eval_host
    arm = analysis_campaign_bundle(analysis).arms[0]
    _frozen_again_at(store, analysis, NAMED_SHAPE)
    off_loop(host.storage.save_template, WRITER_TEMPLATE)
    call = {"arm": arm.variant_key, "called_by": CALLER.identity, "reason": "kept on the stored report"}

    with pytest.raises(ValidationFailedError, match=WHY):
        off_loop(record_campaign_call, host, fixture.campaign.id, TOYHOST_SCOPE, analysis_id=analysis.id, **call)
    with pytest.raises(ValidationFailedError, match=WHY):
        await report_snapshot_freeze(fixture.host, TOYHOST_SCOPE, analysis_id=analysis.id, created_by=CALLER.identity)
    with pytest.raises(ValidationFailedError, match=WHY):
        await writer_case_mint(
            fixture.host,
            WriterCaseMint(template_id=WRITER_TEMPLATE.id, analysis_id=analysis.id),
            TOYHOST_SCOPE,
            created_by=CALLER.identity,
        )
