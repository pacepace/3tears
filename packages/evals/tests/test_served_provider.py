"""A token-metered call's usage row names the upstream that served it, as the reply named it, and a cost names it.

OpenRouter's reply names the upstream provider it routed a call to (its top-level ``provider``), which the wire reads
as ``served_provider``. Contracts held here:

- **the wire to the row**: a reply naming its upstream records it on the call's usage row
  (``RoleUsage.served_provider``), beside ``served_model``; a reply naming none, and a direct provider's, record none;
- **a row per upstream**: calls of one model served by two upstreams are two rows, each naming its own, since one
  model's cost differs by who served it;
- **stored rows**: a row naming none is stored without the key, byte for byte as before the field, and a stored row
  without it reads as not recorded;
- **the census to the page**: each arm's pricing names the upstreams its model rows recorded; where only some were,
  the options table and the writer's cost fact say the rest were not recorded; where none was, the writer's cost fact
  says so and the options table names none, as it did before any was recorded, so no page built then moves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import pytest

from packages.evals.tests.run_facts_support import CASES, aspect_bundle, campaign_over, result_of, run_of
from threetears.evals.analysis.analyst.facts import campaign_overview
from threetears.evals.analysis.bundle.schema import ArmPricing
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.parts.tables import arms_table
from threetears.evals.analysis.report.parts.words import (
    SERVED_BY_IN_PART_TEMPLATE,
    SERVED_BY_TEMPLATE,
)
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel import RoleUsageLedger
from threetears.evals.providers import Catalogue, Model
from threetears.evals.schema.models import EvalResult, EvalRun, RoleUsage

#: The model the router is asked for, and the upstream its reply names.
ROUTED, UPSTREAM = "deepseek/deepseek-v4-flash", "Parasail"

#: The arms of the planted campaign: the control and the candidate.
BASE, FAST = "base", "fast"


def _router_reply(*, provider: str | None) -> dict[str, Any]:
    """An OpenRouter chat completion reply, its cost reported, naming ``provider`` as its upstream where given."""
    body: dict[str, Any] = {
        "model": ROUTED,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "51"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 18, "completion_tokens": 34, "cost": 0.00001204},
    }
    if provider is not None:
        body["provider"] = provider
    return body


def _answering(reply: Mapping[str, Any]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=dict(reply))))


#: OpenRouter's catalogue, holding the routed model's row: what the call is bounded on before it is sent.
_CATALOGUE = {"data": [{"id": ROUTED, "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}}]}


async def _row_of_one_call(reply: Mapping[str, Any]) -> RoleUsage:
    model = Model(ROUTED, api_key="test-key", http=_answering(reply), catalogue=Catalogue(http=_answering(_CATALOGUE)))
    completion = await model.generate(system="s", user="What is 17 times 3?")
    ledger = RoleUsageLedger(role="candidate")
    ledger.add_llm_result(completion)
    (row,) = ledger.rows()
    return row


# --- the wire to the row -------------------------------------------------------------------------------------------


async def test_a_reply_naming_its_upstream_records_it_on_the_calls_usage_row() -> None:
    row = await _row_of_one_call(_router_reply(provider=UPSTREAM))
    assert (row.model, row.served_model, row.served_provider) == (ROUTED, ROUTED, UPSTREAM)
    assert row.provider is None, "provider names who was called for an external contribution, never the upstream"
    assert row.call_count == 1
    assert RoleUsage.model_validate(row.model_dump(mode="json")).served_provider == UPSTREAM


async def test_a_reply_naming_no_upstream_records_none_and_stores_no_key() -> None:
    row = await _row_of_one_call(_router_reply(provider=None))
    assert row.served_provider is None
    assert "served_provider" not in row.model_dump(), "a row naming none is stored as it was before the field"


def test_calls_served_by_two_upstreams_are_two_rows_each_naming_its_own() -> None:
    ledger = RoleUsageLedger(role="candidate")
    for upstream in ("Parasail", "Venice", "Parasail"):
        ledger.add(
            model=ROUTED,
            served_model=ROUTED,
            served_provider=upstream,
            prompt_tokens=10,
            completion_tokens=5,
            reasoning_tokens=None,
            cost_usd=0.001,
            price_source="openrouter:reported",
        )
    assert [(row.served_provider, row.call_count) for row in ledger.rows()] == [("Parasail", 2), ("Venice", 1)]


def test_a_stored_row_without_the_field_reads_as_not_recorded() -> None:
    stored = {"role": "candidate", "model": ROUTED, "served_model": ROUTED, "cost_usd": 0.001, "call_count": 1}
    row = RoleUsage.model_validate(stored)
    assert row.served_provider is None
    assert row.model_dump(mode="json", exclude_defaults=True) == stored


# --- the census to the page ----------------------------------------------------------------------------------------


def _candidate(*, upstream: str | None) -> RoleUsage:
    return RoleUsage(
        role="candidate",
        model=FAST,
        served_model=FAST,
        served_provider=upstream,
        cost_usd=0.002,
        price_source="openrouter:reported",
        call_count=2,
    )


def _trials(usage: Sequence[Sequence[RoleUsage]]) -> list[tuple[EvalRun, list[EvalResult]]]:
    """The control's trials as the support writes them; the candidate's on ``usage``, one row list per case."""
    base, fast = run_of(BASE), run_of(FAST)
    return [
        (base, [result_of(base, index) for index in range(CASES)]),
        (fast, [result_of(fast, index, usage=rows) for index, rows in enumerate(usage)]),
    ]


def _fast(built: ReportInput) -> tuple[ArmPricing | None, dict[str, Any]]:
    """The candidate arm's pricing in the report input, and its options table row."""
    table = arms_table(built, block_look(built, fact_arms(built)), (), forms=PageForms.of(built))
    (row,) = [row for row in table.rows if row["option"].value is not None and FAST in str(row["option"].value)]
    (pricing,) = [
        board.pricing for arm, board in built.scoreboard.items() if FAST in arm and board.pricing is not None
    ] or [None]
    return pricing, row


def test_the_census_names_the_recorded_upstream_and_the_options_table_says_served_by_it() -> None:
    trials = _trials([[_candidate(upstream=UPSTREAM)] for _ in range(CASES)])
    _, bundle, _ = aspect_bundle(trials)
    (board,) = [row for row in bundle.scoreboard if row.pricing is not None and row.pricing.providers]
    assert board.pricing == ArmPricing(
        sources=["openrouter:reported"], providers=[UPSTREAM], calls=2 * CASES, priced_calls=2 * CASES
    )

    built = report_input(campaign_over(trials))
    _, row = _fast(built)
    assert row["priced_by"].value == "openrouter:reported"
    assert row["priced_by"].flag == SERVED_BY_TEMPLATE.format(providers=UPSTREAM)
    assert [arm.served_by for arm in campaign_overview(built, "snap").arms if arm.served_by] == [[UPSTREAM]]


def test_an_arm_whose_rows_recorded_no_upstream_names_none_and_its_page_reads_as_before() -> None:
    """No provider is guessed, and the cell says nothing it did not say before any was recorded, so a page built over
    calls recorded then is unchanged (and a judge comparing such pages is asked what its recordings were keyed on)."""
    built = report_input(campaign_over(_trials([[_candidate(upstream=None)] for _ in range(CASES)])))
    pricing, row = _fast(built)
    assert pricing is not None and pricing.providers == [] and not pricing.providers_in_part
    assert (row["priced_by"].value, row["priced_by"].flag) == ("openrouter:reported", None)
    assert '"providers_in_part"' not in built.model_dump_json(), "left out while False"
    assert not [arm for arm in campaign_overview(built, "snap").arms if arm.served_by]


def test_an_arm_whose_rows_recorded_an_upstream_for_some_calls_says_the_rest_were_not() -> None:
    usage = [[_candidate(upstream=UPSTREAM if index % 2 else None)] for index in range(CASES)]
    trials = _trials(usage)
    _, bundle, _ = aspect_bundle(trials)
    (board,) = [row for row in bundle.scoreboard if row.pricing is not None and row.pricing.providers]
    assert board.pricing is not None
    assert (board.pricing.providers, board.pricing.providers_in_part) == ([UPSTREAM], True)

    built = report_input(campaign_over(trials))
    pricing, row = _fast(built)
    assert pricing is not None and pricing.providers_in_part
    assert row["priced_by"].flag == SERVED_BY_IN_PART_TEMPLATE.format(providers=UPSTREAM)
    (arm,) = [arm for arm in campaign_overview(built, "snap").arms if arm.served_by]
    assert (arm.served_by, arm.served_by_in_part) == ([UPSTREAM], True)


@pytest.mark.parametrize("upstream", [UPSTREAM, None])
def test_a_row_holding_only_unanswered_calls_leaves_the_providers_unread(upstream: str | None) -> None:
    lost = RoleUsage(role="candidate", model=FAST, unanswered_lost=1)
    trials = _trials([[_candidate(upstream=upstream), lost] for _ in range(CASES)])
    _, bundle, _ = aspect_bundle(trials)
    # The lost row counts no answered call, so the candidate's arm is the one whose calls read uncounted.
    (board,) = [row for row in bundle.scoreboard if row.pricing is not None and row.pricing.calls is None]
    assert board.pricing is not None
    assert (board.pricing.providers, board.pricing.providers_in_part) == ([upstream] if upstream else [], False)
