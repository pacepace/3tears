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
- **on a real campaign's evidence**: the Kairo round 2 bundle read back from a ``jsonb`` store, or with its mappings in
  any order, builds the report input the live one built, mapping order included, and the planner's message from it;
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

from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.generator import user_message_digest
from threetears.evals.analysis.planner import plan_user_message
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.reporter_bank import case_limits
from threetears.evals.analysis.reporter_kind import (
    MEMO_CASE_VERSION,
    PLAN_CASE_VERSION,
    WRITER_MESSAGE_DIFFERS,
    ReporterKindVersion,
)
from threetears.evals.analysis.service import get_report_plan
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import FrozenReporterCase, ReporterCaseFreeze, analysis_generate, reporter_case_freeze
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.report_plans.dataset import KAIRO_BUNDLE, kairo_bundle
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture, settled
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

    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"
    (analysis,) = off_loop(storage.list_analyses_by_campaign, fixture.campaign.id, TOYHOST_SCOPE)
    assert analysis.report_plan_id is not None, "the generation is the report planner's"

    receipt = _frozen(fixture.host.eval_host, analysis)

    stored = off_loop(get_report_plan, storage, analysis)
    assert stored is not None
    assert receipt.report_input_digest == stored.report_input_digest
    assert (receipt.writer_message_check, receipt.limits) == ("verified", [])


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
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"
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
