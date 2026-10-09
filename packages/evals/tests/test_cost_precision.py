"""Spend is stored and summed at full float precision; only the text a person reads is rounded.

A cheap model call costs around a hundred-thousandth of a dollar, so a total rounded to six places keeps
two significant figures of it, and a call under half a micro-dollar is stored as $0. Real campaigns run
a thousand times the calls these tests make, so every such loss is multiplied. These tests follow one
cheap price, $1.2345678e-05 a call, from the usage row to the result's ``cost_usd``, the run's summary and
the analysis bundle, and require it intact at each — the old rounding stored 1.2e-05 and summed 3.7e-05.

Mutations that turn this file red: restoring ``round(..., 6)`` in ``RoleUsageLedger.rows``,
``blended_cost``, the production-replicating sum or ``ExternalRateTable.money_for``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from threetears.evals.analysis import inspect_campaign_bundle
from threetears.evals.contracts import RoleUsageLedger
from threetears.evals.contracts.host.spend import ExternalSpend
from threetears.evals.contracts.spend import ExternalRateTable
from threetears.evals.contracts.usage_capture import blended_cost, production_replicating_cost
from threetears.evals.quick import Answer, compare
from threetears.evals.run import list_results

SCOPE = "cost-precision-tests"

#: One cheap call's price: eight significant figures, below what six decimal places can hold.
PRICE = 1.2345678e-05

#: Three of them, summed without loss — the nearest float to the exact sum, which ``3 * PRICE`` also is.
THREE = math.fsum([PRICE] * 3)

CASES = [
    {"text": "a cat", "label": "animal"},
    {"text": "a fir", "label": "plant"},
    {"text": "an oak", "label": "plant"},
]


def _expected(case: Mapping[str, Any]) -> str:
    return str(case["label"])


async def cheap(case: Mapping[str, Any]) -> Answer:
    """Answers right, at one cheap call's price."""
    return Answer(case["label"], model="m-cheap", input_tokens=10, output_tokens=1, cost_usd=PRICE)


async def dearer(case: Mapping[str, Any]) -> Answer:
    """Answers right, at two cheap calls' price."""
    return Answer(case["label"], model="m-dear", input_tokens=10, output_tokens=1, cost_usd=2 * PRICE)


def test_the_fixture_price_is_one_six_places_would_have_lost() -> None:
    assert round(PRICE, 6) != PRICE and round(THREE, 6) != THREE
    assert THREE == 3.7037034e-05


def test_a_row_folding_three_cheap_calls_keeps_every_digit() -> None:
    ledger = RoleUsageLedger(role="candidate")
    for _ in range(3):
        ledger.add(model="m", prompt_tokens=10, completion_tokens=1, reasoning_tokens=None, cost_usd=PRICE)
    (row,) = ledger.rows()
    assert row.cost_usd == THREE
    assert blended_cost([row], ("candidate",)) == THREE
    assert production_replicating_cost([row], substituted_deliveries=0) == THREE


def test_a_call_under_half_a_micro_dollar_is_not_stored_as_free() -> None:
    ledger = RoleUsageLedger(role="candidate")
    ledger.add(model="m", prompt_tokens=1, completion_tokens=1, reasoning_tokens=None, cost_usd=4e-7)
    (row,) = ledger.rows()
    assert row.cost_usd == 4e-7
    assert blended_cost([row], ("candidate",)) == 4e-7


def test_a_rate_priced_external_call_keeps_its_full_price() -> None:
    table = ExternalRateTable(rates={("search", "credit"): PRICE})
    spend = ExternalSpend(provider="search", provider_unit="credit", provider_units=3, calls=1, money=None)
    assert table.money_for(spend) == 3 * PRICE


async def test_cheap_spend_survives_the_result_the_run_summary_and_the_bundle() -> None:
    comparison = await compare(
        CASES, {"cheap": cheap, "dearer": dearer}, expected=_expected, control="cheap", scope_id=SCOPE, k=1
    )
    summary = comparison.arms["cheap"]

    # Each result: the row and the total derived from it carry the price exactly.
    results = list_results(comparison.host.storage, summary.run_id, SCOPE)
    assert len(results) == 3
    for result in results:
        (row,) = result.usage
        assert row.cost_usd == PRICE and result.cost_usd == PRICE

    # The run summary: three calls, summed without loss.
    assert summary.candidate_cost_usd == THREE

    # The bundle: the run's total, the cell's mean, and both after a JSON round trip.
    bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, SCOPE).bundle
    (run_summary,) = [s for s in bundle.run_summaries if s.run_id == summary.run_id]
    assert run_summary.cost_usd == THREE
    assert run_summary.prod_cost_usd == THREE
    means = {
        cell.variant_key: next(m.mean for m in cell.measures.measures if m.name == "cost_usd")
        for cell in bundle.cell_measures
    }
    assert sorted(means.values()) == [PRICE, 2 * PRICE]
    reread = json.loads(bundle.model_dump_json())
    assert [s["cost_usd"] for s in reread["run_summaries"] if s["run_id"] == summary.run_id] == [THREE]

    # Full precision still fingerprints deterministically: the same evidence assembles to the same bytes.
    again = inspect_campaign_bundle(comparison.host, comparison.campaign_id, SCOPE).bundle
    assert again.fingerprint() == bundle.fingerprint()
