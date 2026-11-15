"""A campaign's bundle read back from a store that keeps no key order builds what the live one built (#978).

The store port promises a document's fields and values back, never the order of a mapping's keys
(:class:`~threetears.evals.schema.store_port.DocumentStore`), and a Postgres ``jsonb`` column, where a host keeps its
documents, keeps none: it stores an object's keys shortest first, then bytewise. What a reader builds from a stored
bundle must therefore not depend on that order. These tests hold that:

- **on a real campaign's evidence**: the Kairo round 2 bundle read back from a ``jsonb`` store, or with its mappings in
  any order, builds the report input the live one built, mapping order included;
- **runs created out of id order**: with runs whose creation order is not their id order, a bundle's short and
  incomplete runs come back in the order assembly read the runs, which a store keeping key order already holds, so
  the report input of a stored analysis does not move;
- **on the page**: a declared crossing's missing cells are named by their levels in the design's axis order, read
  back from any store.
"""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Callable
from typing import Any

import pytest

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle, build_code_only_report
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report import DisclosureBlock
from threetears.evals.analysis.report.input_build import report_input
from threetears.evals.kernel import AspectDesign, ControlDeclaration, EvalStorage, SweptAxis
from threetears.evals.schema import IntervalScale, SweepableValue
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_aspect_campaign
from packages.evals.tests.fixtures.report_inputs.dataset import KAIRO_BUNDLE
from packages.evals.tests.fixtures.toyhost.corpus import toyhost_batch, toyhost_measurements
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.ops_support import TOYHOST_SCOPE

#: How a store orders an object's keys: a function from the keys as written to the keys as read back.
KeyOrder = Callable[[list[str]], list[str]]


def jsonb_order(keys: list[str]) -> list[str]:
    """The order a Postgres ``jsonb`` value holds an object's keys in: shorter keys first, then bytewise."""
    return sorted(keys, key=lambda key: (len(key.encode("utf-8")), key.encode("utf-8")))


def reversed_order(keys: list[str]) -> list[str]:
    """Every object's keys the other way round: what a store keeping no order may hand back."""
    return keys[::-1]


def _shuffled(seed: int) -> KeyOrder:
    rng = random.Random(seed)

    def order(keys: list[str]) -> list[str]:
        return rng.sample(keys, len(keys))

    return order


def reordered(node: Any, order: KeyOrder) -> Any:
    """``node`` with every object's keys, at every depth, in ``order``; its values and every list's order kept."""
    if isinstance(node, dict):
        return {key: reordered(node[key], order) for key in order(list(node))}
    if isinstance(node, list):
        return [reordered(value, order) for value in node]
    return node


def _kairo_document() -> dict[str, Any]:
    with gzip.open(KAIRO_BUNDLE, "rt", encoding="utf-8") as stream:
        document: dict[str, Any] = json.load(stream)
    return document


def _dumped(bundle: CampaignBundle) -> str:
    """The report input built from ``bundle``, as JSON keeping each mapping's order."""
    return json.dumps(report_input(bundle).model_dump(mode="json", by_alias=True))


# --- on a real campaign's evidence ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "order",
    [jsonb_order, reversed_order, _shuffled(1), _shuffled(2), _shuffled(3)],
    ids=["jsonb", "reversed", "shuffled-1", "shuffled-2", "shuffled-3"],
)
def test_a_real_campaigns_bundle_read_back_in_any_key_order_builds_the_same_input(order: KeyOrder) -> None:
    document = _kairo_document()
    live = CampaignBundle.from_dict(document)
    held = CampaignBundle.from_dict(reordered(document, order))

    # The dump keeps each mapping's order, so an input equal by value and not by order differs here.
    assert _dumped(held) == _dumped(live)


# --- runs whose creation order is not their id order ------------------------------------------------------------------


def _created_out_of_id_order() -> dict[str, Any]:
    """The Kairo bundle with its runs created in the reverse of their id order, as assembly would hold them.

    Assembly reads a campaign's runs by creation time, then id, and summarizes them in that order
    (``run_summaries``); the short and incomplete runs are keyed in it. Here the runs are created latest id first, two
    are short and three ended cancelled.
    """
    document = _kairo_document()
    (aspect,) = document["aspects"]
    summaries: list[dict[str, Any]] = aspect["run_summaries"]
    instants = sorted(summary["created_at"] for summary in summaries)
    created = list(reversed(summaries))
    for summary, instant in zip(created, instants, strict=True):
        summary["created_at"] = instant
    cancelled = {summary["run_id"] for summary in created[:3]}
    for summary in created:
        if summary["run_id"] in cancelled:
            summary["status"] = "cancelled"
    order = [summary["run_id"] for summary in created]
    aspect["run_summaries"] = created
    aspect["short_runs"] = {run: aspect["short_runs"][run] for run in order if run in aspect["short_runs"]}
    aspect["incomplete_runs"] = {run: "cancelled" for run in order if run in cancelled}
    return document


@pytest.mark.parametrize("order", [jsonb_order, reversed_order, _shuffled(4)], ids=["jsonb", "reversed", "shuffled-4"])
def test_runs_created_out_of_id_order_come_back_in_creation_order_and_a_stored_analysis_does_not_move(
    order: KeyOrder,
) -> None:
    document = _created_out_of_id_order()
    (written,) = document["aspects"]
    kept = CampaignBundle.from_dict(document)
    held = CampaignBundle.from_dict(reordered(document, order))

    (as_kept,) = kept.aspects
    assert list(as_kept.short_runs) != sorted(as_kept.short_runs), "the fixture's creation order is not its id order"
    # A store that keeps key order hands back what assembly wrote, and the read leaves it there.
    assert (list(as_kept.short_runs), list(as_kept.incomplete_runs)) == (
        list(written["short_runs"]),
        list(written["incomplete_runs"]),
    )
    (as_held,) = held.aspects
    assert (list(as_held.short_runs), list(as_held.incomplete_runs)) == (
        list(as_kept.short_runs),
        list(as_kept.incomplete_runs),
    )
    assert _dumped(held) == _dumped(kept)


# --- on the page: a declared crossing's cells -------------------------------------------------------------------------


def _axis(axis_id: str, levels: list[int], unit: str) -> SweptAxis:
    return SweptAxis(
        axis_id=axis_id,
        values=[
            SweepableValue.of(level, scale=IntervalScale(value=level, unit=unit), keep_raw=True) for level in levels
        ],
    )


def _crossing_bundle() -> AnalysisContextBundle:
    """A full crossing of two axes, declared longer name first so a ``jsonb`` store would turn them round, with one
    cell never run."""
    profile = toyhost_profile()
    storage = EvalStorage(InMemoryDocumentStore())
    design = AspectDesign(
        axes=[_axis("retriever_top_k", [3, 5], "chunks"), _axis("chunk_tokens", [256, 512], "tok")],
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="witnessed"),
        crossing="full",
    )
    runs = []
    for top_k, chunk in [(3, 256), (5, 256), (3, 512)]:
        run = toyhost_batch(chunk_tokens=chunk, retriever_top_k=top_k, grader_version="grade-2.1.0")
        storage.save_eval_run(run)
        for result in toyhost_measurements(run, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8):
            storage.save_eval_result(result)
        runs.append(run)
    campaign = make_aspect_campaign(
        label="extract_invoice_fields",
        run_ids=[run.id for run in runs],
        design=design,
        id="crossing",
        scope_id=TOYHOST_SCOPE,
        name="crossing",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="extractor_config",
    )
    storage.save_campaign(campaign)
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _never_run(bundle: AnalysisContextBundle) -> str:
    report = build_code_only_report(bundle, assembled_at="2026-10-10T00:00:00Z")
    (gaps,) = [
        block.text
        for block in report.blocks
        if isinstance(block, DisclosureBlock) and block.text.startswith("Declared arms never run:")
    ]
    return gaps


@pytest.mark.parametrize("order", [jsonb_order, reversed_order], ids=["jsonb", "reversed"])
def test_the_page_names_a_missing_cell_by_its_levels_in_the_designs_axis_order(order: KeyOrder) -> None:
    live = _crossing_bundle()
    held = AnalysisContextBundle.from_dict(reordered(live.to_dict(), order))

    assert _never_run(live) == "Declared arms never run: 5chunks × 512tok."
    assert _never_run(held) == _never_run(live)
