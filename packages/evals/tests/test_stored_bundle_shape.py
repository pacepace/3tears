"""A stored analysis frozen over a bundle of another shape is refused, one analysis at a time, naming the regeneration.

A bundle is regenerated across a shape change, never migrated, and a stored read is strict. v0.75.0 and v0.76.0 froze
analyses over bundle shape 60, which carries fields later shapes dropped. Such an analysis is listed as needing
regeneration rather than refusing the listing of every other analysis, and its report is refused with the step that
regenerates it.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.kernel.errors import RefusedInput, ValidationFailedError
from threetears.evals.ops.analysis import analyses_list, report_read
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import analysed_aspect, empty_bundle, make_analysis
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture

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
    assert lines[current.id].needs_regeneration is None
    unreadable = lines[old.id].needs_regeneration
    assert unreadable is not None
    assert f"shape {SHIPPED_SHAPE}" in unreadable and "analysis_generate" in unreadable
    assert lines[old.id].verdict.startswith("Needs regeneration: ")


def test_the_report_of_a_campaign_whose_newest_analysis_is_frozen_at_the_shipped_shape_says_to_regenerate():
    fixture = off_loop(ops_fixture)
    host = fixture.host.eval_host
    campaign_id = fixture.campaign.id
    old_aspect = analysed_aspect(empty_bundle(campaign_id=campaign_id)).model_copy(
        update={"bundle": _frozen_at(SHIPPED_SHAPE, campaign_id)}
    )
    frame = campaign_bundle(empty_bundle(campaign_id=campaign_id)).frame()
    host.storage.save_analysis(
        make_analysis(campaign_id=campaign_id, scope_id=TOYHOST_SCOPE, aspects=[old_aspect], campaign_bundle=frame)
    )

    with pytest.raises((RefusedInput, ValidationFailedError)) as refused:
        report_read(host, campaign_id, TOYHOST_SCOPE, format="markdown")

    said = str(refused.value)
    assert f"shape {SHIPPED_SHAPE}" in said
    assert "generate the analysis again (analysis_generate)" in said
