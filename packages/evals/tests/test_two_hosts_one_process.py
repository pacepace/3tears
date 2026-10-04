"""Two hosts run and analyse side by side in one event loop, and neither sees the other.

The engine holds no host of its own, so each entrypoint is handed one. This runs the toy host
(invoice extraction) and the courier (route planning) concurrently — each drives its own runs
through the real trial loop, files a campaign in its own storage and assembles that campaign's
bundle through the analysis service — and then reads each bundle for the other host's vocabulary.

The two vocabularies share one word: different levers, apparatus and world dimensions, different
subjects, behaviors and scopes, and different measures but for ``field_accuracy``, which both
register with different meanings. A bundle carrying one word of the other host's would be the leak
an installed or ambient host produces, and the overlap assertion is what makes "side by side" a fact
rather than a schedule: the second drive starts before the first one ends.

The shared name is what lets the second test see a leak of MEANING. A host's measure described from
the other host's registry prints no foreign word, and with no name in common a description cached by
name across hosts is never visibly wrong — so each catalog entry is checked against the descriptor
its own host gives, and the one name both describe differently is where a cross-host answer shows.
"""

from __future__ import annotations

import asyncio

from threetears.evals.analysis import BundleInspection, inspect_campaign_bundle
from threetears.evals.contracts import MetricDescriptor
from threetears.evals.contracts.host import EvalHost
from threetears.evals.contracts.metrics import describe_reported_measure
from packages.evals.tests.fixtures.courierhost import (
    COURIER_MEASURES,
    COURIER_SCOPE,
    FIELD_ACCURACY,
    courier_host,
    run_courier_campaign,
)
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES
from packages.evals.tests.fixtures.toyhost.run import execute_toyhost_run, toyhost_run_campaign

#: Words only the toy host uses, and that its run-path bundle carries: two levers, a third lever's
#: name and the campaign's behavior. Its one reported measure is the shared name, below.
TOY_CARRIED = ("chunk_tokens", "retriever_top_k", "extraction_schema", "extract_invoice_fields")

#: Words only the courier uses and its bundle carries: a lever, the apparatus input its arms differed
#: on, both measures and the behavior.
COURIER_CARRIED = ("search_depth", "traffic_feed", "on_time_rate", "detour_km", "plan_delivery_round")

#: Each host's whole vocabulary — every lever, apparatus input, measure and world dimension it
#: registers — of which none may appear in the OTHER host's bundle. Wider than what each bundle
#: carries: a bundle names a world dimension or an apparatus input only where it moved, and a leak
#: would arrive by name whether or not the owning host's own bundle printed it.
TOY_VOCABULARY = (
    *TOY_CARRIED,
    "ocr_engine_version",
    "reviewer_pool",
    "grader_version",
    "manual_review_rate",
    "document_language",
    "scan_quality",
    "vendor_template",
    "ingest_backlog",
)
COURIER_VOCABULARY = (*COURIER_CARRIED, "road_closures")


def _registered(measures: tuple[MetricDescriptor, ...], name: str) -> MetricDescriptor:
    """The descriptor a host's own measure tuple declares under ``name``."""
    (descriptor,) = [measure for measure in measures if measure.name == name]
    return descriptor


#: The one measure name both hosts register — each with its own meaning.
TOY_FIELD_ACCURACY = _registered(TOYHOST_MEASURES, FIELD_ACCURACY)
COURIER_FIELD_ACCURACY = _registered(COURIER_MEASURES, FIELD_ACCURACY)


async def _toy(host: EvalHost, timeline: list[str]) -> BundleInspection:
    timeline.append("toy started")
    path = await execute_toyhost_run(host=host)
    campaign = toyhost_run_campaign(path)
    host.storage.save_campaign(campaign)
    inspection = inspect_campaign_bundle(host, campaign.id, TOYHOST_SCOPE)
    timeline.append("toy ended")
    return inspection


async def _courier(host: EvalHost, timeline: list[str]) -> BundleInspection:
    timeline.append("courier started")
    campaign = await run_courier_campaign(host)
    inspection = inspect_campaign_bundle(host, campaign.id, COURIER_SCOPE)
    timeline.append("courier ended")
    return inspection


async def test_two_hosts_run_and_analyse_concurrently_and_each_bundle_speaks_only_its_own_vocabulary():
    toy, courier = toyhost_host(), courier_host()
    timeline: list[str] = []

    toy_inspection, courier_inspection = await asyncio.gather(_toy(toy, timeline), _courier(courier, timeline))

    assert timeline.index("courier started") < timeline.index("toy ended"), timeline
    assert timeline.index("toy started") < timeline.index("courier ended"), timeline

    toy_bundle, courier_bundle = toy_inspection.bundle, courier_inspection.bundle
    assert toy_bundle.run_ids and courier_bundle.run_ids, "each host's campaign resolved its runs"
    toy_text, courier_text = toy_bundle.model_dump_json(), courier_bundle.model_dump_json()

    assert [word for word in TOY_CARRIED if word not in toy_text] == [], "the toy bundle lost its own vocabulary"
    assert [word for word in COURIER_CARRIED if word not in courier_text] == [], "the courier bundle lost its own"
    assert [word for word in COURIER_VOCABULARY if word in toy_text] == [], "the courier's words reached the toy bundle"
    assert [word for word in TOY_VOCABULARY if word in courier_text] == [], "the toy's words reached the courier bundle"
    # The shared name is in both, each under its own host's sentence and never the other's.
    assert FIELD_ACCURACY in toy_text and FIELD_ACCURACY in courier_text
    assert TOY_FIELD_ACCURACY.description in toy_text and COURIER_FIELD_ACCURACY.description in courier_text
    assert COURIER_FIELD_ACCURACY.description not in toy_text, "the courier's meaning reached the toy bundle"
    assert TOY_FIELD_ACCURACY.description not in courier_text, "the toy's meaning reached the courier bundle"


async def test_each_host_measures_with_its_own_registry():
    """The measure catalogue — the surface a paid generator reads — is each host's own, described by it.

    Key presence cannot show this: a bundle names its measures whichever registry described them. So
    every entry, on both sides, must be the descriptor the host's OWN registry resolves that name to,
    and the name both hosts register must arrive with each host's own meaning — the one place a
    description taken from the other host, or cached by name across hosts, differs from the right one.
    """
    toy, courier = toyhost_host(), courier_host()
    # The precondition that makes the shared name a discriminator: the two meanings differ where a
    # reader would act on them.
    assert TOY_FIELD_ACCURACY.higher_is_better != COURIER_FIELD_ACCURACY.higher_is_better
    assert (TOY_FIELD_ACCURACY.unit, TOY_FIELD_ACCURACY.description) != (
        COURIER_FIELD_ACCURACY.unit,
        COURIER_FIELD_ACCURACY.description,
    )

    toy_inspection, courier_inspection = await asyncio.gather(_toy(toy, []), _courier(courier, []))

    toy_catalog, courier_catalog = toy_inspection.bundle.measure_catalog, courier_inspection.bundle.measure_catalog
    assert toy_catalog[FIELD_ACCURACY] == TOY_FIELD_ACCURACY
    assert courier_catalog[FIELD_ACCURACY] == COURIER_FIELD_ACCURACY
    for host, catalog in ((toy, toy_catalog), (courier, courier_catalog)):
        own = {name: describe_reported_measure(name, host.profile.measures) for name in catalog}
        wrong = sorted(name for name in catalog if catalog[name] != own[name])
        assert wrong == [], f"{host.profile.host_id} described {wrong} from a registry not its own"
    assert courier_catalog["on_time_rate"].merit_axis == "quality"
    assert courier_catalog["detour_km"].unit == "km"
    assert "on_time_rate" not in toy_catalog and "detour_km" not in toy_catalog
