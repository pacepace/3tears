"""A planned analysis stored in a store that keeps no key order freezes into a plan case its message check verifies.

The store port promises a document's fields and values back, never the order of a mapping's keys
(:class:`~threetears.evals.schema.store_port.DocumentStore`), and a Postgres ``jsonb`` column, where a host keeps its
documents, keeps none: it stores an object's keys shortest first, then bytewise. A plan case's ``writer_message`` is
rendered at the freeze from the bundle the analysis stores, and checked against the digest of what its planner was sent
from the bundle assembled live. So every mapping whose key order the planner reads must come back from the stored
bundle in the order the live one held it, or a plan the planner wrote over exactly that evidence is frozen saying its
message ``differs``. These tests hold that:

- **end to end, through the host's own operations**: a generation started by ``analysis_generate`` over the toy
  host's campaign, assembled live from its stored runs and stored in a store that keeps no key order, frozen by
  ``reporter_case_freeze``, is ``verified`` with no limit, on the ``jsonb`` order and on the reverse of every mapping;
- **a memo case too**: a memo the writer wrote over the toy host's campaign, stored in such a store and frozen, is
  ``verified`` with no limit;
- **on a real campaign's evidence**: the Kairo round 2 bundle read back from a ``jsonb`` store, or with its mappings in
  any order, builds the report input the live one built, mapping order included, and the planner's message from it;
  and with runs whose creation order is not their id order, its short and incomplete runs come back in the order
  assembly read the runs, which a store keeping key order already holds, so a stored analysis does not move;
- **on the page**: a declared crossing's missing cells are named by their levels in the design's axis order, read
  back from any store;
- **what a case says when its message does differ**: a plan case states the planner's sentence
  (``WRITER_MESSAGE_DIFFERS[PLAN_CASE_VERSION]``), never the memo writer's, and a memo case the memo writer's.
"""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Callable
from typing import Any

import pytest

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle, build_code_only_report
from threetears.evals.analysis.campaign_bundle import CampaignBundle, writer_view
from threetears.evals.analysis.generator import user_message_digest
from threetears.evals.analysis.planner import plan_user_message
from threetears.evals.analysis.report import DisclosureBlock
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.reporter_bank import case_limits
from threetears.evals.analysis.reporter_kind import (
    MEMO_CASE_VERSION,
    PLAN_CASE_VERSION,
    WRITER_MESSAGE_DIFFERS,
    ReporterKindVersion,
)
from threetears.evals.analysis.service import get_report_plan
from threetears.evals.kernel import AspectDesign, ControlDeclaration, EvalStorage, SweptAxis
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import FrozenReporterCase, ReporterCaseFreeze, reporter_case_freeze
from threetears.evals.schema import IntervalScale, SweepableValue
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_aspect_campaign
from packages.evals.tests.fixtures.report_plans.dataset import KAIRO_BUNDLE, kairo_bundle
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import toyhost_batch, toyhost_measurements
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture, planned_generation, settled
from packages.evals.tests.report_support import stored_memo_analysis, toy_campaign_host
from packages.evals.tests.reporter_case_support import reporter_template

#: A digest no message rendered here has: what a generation recorded sending, where it sent something else.
_ANOTHER_DIGEST = user_message_digest("a message no freeze renders")

#: How a store orders an object's keys: a function from the keys as written to the keys as read back.
KeyOrder = Callable[[list[str]], list[str]]


def jsonb_order(keys: list[str]) -> list[str]:
    """The order a Postgres ``jsonb`` value holds an object's keys in: shorter keys first, then bytewise."""
    return sorted(keys, key=lambda key: (len(key.encode("utf-8")), key.encode("utf-8")))


def reversed_order(keys: list[str]) -> list[str]:
    """Every object's keys the other way round: what a store keeping no order may hand back."""
    return keys[::-1]


def reordered(node: Any, order: KeyOrder) -> Any:
    """``node`` with every object's keys, at every depth, in ``order``; its values and every list's order kept."""
    if isinstance(node, dict):
        return {key: reordered(node[key], order) for key in order(list(node))}
    if isinstance(node, list):
        return [reordered(value, order) for value in node]
    return node


class _StoreKeepingNoKeyOrder(InMemoryDocumentStore):
    """The in-memory store, holding each document as JSON with its keys in ``order``: a store as the port allows."""

    def __init__(self, order: KeyOrder) -> None:
        super().__init__()
        self._order = order

    def _as_held(self, document: dict[str, Any]) -> dict[str, Any]:
        held: dict[str, Any] = reordered(json.loads(json.dumps(document)), self._order)
        return held

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        super().upsert(self._as_held(document), if_match=if_match, create_only=create_only)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        return super().merge_fields(doc_id, scope_id, self._as_held(fields))


@pytest.mark.parametrize("order", [jsonb_order, reversed_order], ids=["jsonb", "reversed"])
async def test_a_plan_generated_through_the_host_and_frozen_from_its_store_is_verified(order: KeyOrder) -> None:
    fixture = off_loop(ops_fixture, store=_StoreKeepingNoKeyOrder(order))
    storage = fixture.host.eval_host.storage
    off_loop(storage.save_template, reporter_template())

    await planned_generation(fixture)
    (analysis,) = off_loop(storage.list_analyses_by_campaign, fixture.campaign.id, TOYHOST_SCOPE)
    assert analysis.report_plan_id is not None, "the generation is the report planner's"

    receipt = _frozen(fixture.host.eval_host, analysis)

    stored = off_loop(get_report_plan, storage, analysis)
    assert stored is not None
    assert receipt.report_input_digest == stored.report_input_digest
    assert (receipt.writer_message_check, receipt.limits) == ("verified", [])


@pytest.mark.parametrize("order", [jsonb_order, reversed_order], ids=["jsonb", "reversed"])
async def test_a_memo_written_over_a_campaign_in_that_store_and_frozen_from_it_is_verified(order: KeyOrder) -> None:
    host = toyhost_host(storage=EvalStorage(_StoreKeepingNoKeyOrder(order)))
    campaign, corpus = toyhost_campaign()

    def store() -> None:
        for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
            host.storage.save_eval_run(run)
            for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
                host.storage.save_eval_result(result)
        host.storage.save_campaign(campaign)
        host.storage.save_template(reporter_template())

    off_loop(store)
    memo, _ = await stored_memo_analysis(host, campaign)

    receipt = _frozen(host, memo)

    assert (receipt.kind_version, receipt.writer_message_check, receipt.limits) == (MEMO_CASE_VERSION, "verified", [])


def _frozen(host: EvalHost, analysis: EvalAnalysis) -> FrozenReporterCase:
    """``analysis`` frozen into the reporter template through the operation a host calls, with no label."""
    return off_loop(
        reporter_case_freeze,
        host,
        ReporterCaseFreeze.model_validate(
            {
                "template_id": reporter_template().id,
                "campaign_id": analysis.campaign_id,
                "recorded_analysis_id": analysis.id,
            }
        ),
        TOYHOST_SCOPE,
        collected_by=CALLER.identity,
    )


def _kairo_document() -> dict[str, Any]:
    with gzip.open(KAIRO_BUNDLE, "rt", encoding="utf-8") as stream:
        document: dict[str, Any] = json.load(stream)
    return document


def _shuffled(seed: int) -> KeyOrder:
    rng = random.Random(seed)

    def order(keys: list[str]) -> list[str]:
        return rng.sample(keys, len(keys))

    return order


@pytest.mark.parametrize(
    "order",
    [jsonb_order, reversed_order, _shuffled(1), _shuffled(2), _shuffled(3)],
    ids=["jsonb", "reversed", "shuffled-1", "shuffled-2", "shuffled-3"],
)
def test_a_real_campaigns_bundle_read_back_in_any_key_order_builds_the_same_input_and_message(order: KeyOrder) -> None:
    document = _kairo_document()
    live = report_input(CampaignBundle.from_dict(document))
    held = report_input(CampaignBundle.from_dict(reordered(document, order)))

    # The dump keeps each mapping's order, so an input equal by value and not by order differs here.
    assert json.dumps(held.model_dump(mode="json", by_alias=True)) == json.dumps(
        live.model_dump(mode="json", by_alias=True)
    )
    assert user_message_digest(plan_user_message(held)) == user_message_digest(plan_user_message(live))


# --- what a case says when its message does differ --------------------------------------------------------------------


@pytest.mark.parametrize("kind_version", [MEMO_CASE_VERSION, PLAN_CASE_VERSION])
def test_the_limit_on_a_message_that_differs_is_the_one_for_what_the_case_judges(
    kind_version: ReporterKindVersion,
) -> None:
    bundle = kairo_bundle()

    limits = case_limits(
        bundle,
        recorded_fingerprint=bundle.fingerprint(),
        writer_message="a message",
        recorded_message_digest=_ANOTHER_DIGEST,
        kind_version=kind_version,
    )

    assert limits[0] == WRITER_MESSAGE_DIFFERS[kind_version].format(recorded=_ANOTHER_DIGEST)


def test_the_two_kinds_state_two_sentences() -> None:
    assert WRITER_MESSAGE_DIFFERS[MEMO_CASE_VERSION] != WRITER_MESSAGE_DIFFERS[PLAN_CASE_VERSION]


async def test_a_plan_case_whose_planner_was_sent_another_message_states_the_planners_sentence() -> None:
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    off_loop(storage.save_template, reporter_template())
    await planned_generation(fixture)
    (analysis,) = off_loop(storage.list_analyses_by_campaign, fixture.campaign.id, TOYHOST_SCOPE)
    stored = off_loop(get_report_plan, storage, analysis)
    assert stored is not None
    sent_another = stored.generation.model_copy(update={"user_message_digest": _ANOTHER_DIGEST})
    off_loop(storage.save_report_plan, stored.model_copy(update={"generation": sent_another}))

    receipt = _frozen(fixture.host.eval_host, analysis)

    assert (receipt.kind_version, receipt.writer_message_check) == (PLAN_CASE_VERSION, "differs")
    assert receipt.limits == [WRITER_MESSAGE_DIFFERS[PLAN_CASE_VERSION].format(recorded=_ANOTHER_DIGEST)]


async def test_a_memo_case_whose_writer_was_sent_another_message_states_the_memo_writers_sentence() -> None:
    host, campaign = off_loop(toy_campaign_host)
    off_loop(host.storage.save_template, reporter_template())
    memo, _ = await stored_memo_analysis(host, campaign)
    sent_another = memo.generation.model_copy(update={"user_message_digest": _ANOTHER_DIGEST})
    off_loop(host.storage.save_analysis, memo.model_copy(update={"generation": sent_another}))

    receipt = _frozen(host, memo)

    assert (receipt.kind_version, receipt.writer_message_check) == (MEMO_CASE_VERSION, "differs")
    assert receipt.limits[0] == WRITER_MESSAGE_DIFFERS[MEMO_CASE_VERSION].format(recorded=_ANOTHER_DIGEST)


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
    assert json.dumps(report_input(held).model_dump(mode="json", by_alias=True)) == json.dumps(
        report_input(kept).model_dump(mode="json", by_alias=True)
    )
    assert plan_user_message(report_input(held)) == plan_user_message(report_input(kept))
    assert _runs_disclosed(held) == _runs_disclosed(kept)


def _runs_disclosed(bundle: CampaignBundle) -> list[list[str]]:
    """The short and incomplete runs the memo writer's view discloses, each in its order."""
    (aspect,) = writer_view(bundle)["aspects"]
    disclosures: dict[str, dict[str, str]] = aspect["disclosures"]
    return [list(disclosures["short_runs"]), list(disclosures["incomplete_runs"])]


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
