"""Per-role usage accumulation + cost-view derivation."""

from types import SimpleNamespace

from threetears.evals.schema.external_spend import ExternalSpend
import pytest
from pydantic import ValidationError

from threetears.evals.schema.models import AsyncDelivery, AsyncExternalSpend, RoleUsage
from threetears.evals.kernel.spend import ExternalRateTable
from threetears.evals.kernel.usage_capture import (
    RoleUsageLedger,
    async_delivery_usage,
    cell_cost,
    count_substituted_deliveries,
    production_replicating_cost,
    program_cost,
    resolve_result_usage,
)
from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.factories import result_capture_defaults


#: The price source the fake completions below report, as a host's client would name its own.
_SOURCE = "provider:reported"


def _result(*, model="m", inp=10, out=20, reasoning=None, cost=0.001, price_source=_SOURCE):
    """An ``LLMResult``-shaped response object, naming where its price came from."""
    return SimpleNamespace(
        model=model,
        input_tokens=inp,
        output_tokens=out,
        reasoning_tokens=reasoning,
        cost_usd=cost,
        price_source=price_source,
    )


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


def test_no_calls_yields_no_rows():
    assert RoleUsageLedger(role="judge").rows() == []


def test_single_call_row():
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_result(inp=100, out=50, reasoning=20, cost=0.002))
    (row,) = ledger.rows()
    assert (row.role, row.model) == ("judge", "m")
    assert (row.prompt_tokens, row.completion_tokens, row.reasoning_tokens) == (100, 50, 20)
    assert row.cost_usd == 0.002
    assert row.price_source == _SOURCE


def test_calls_on_the_same_model_accumulate_into_one_row():
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_result(inp=100, out=50, reasoning=20, cost=0.002))
    ledger.add_llm_result(_result(inp=10, out=5, reasoning=1, cost=0.001))
    (row,) = ledger.rows()
    assert (row.prompt_tokens, row.completion_tokens, row.reasoning_tokens) == (110, 55, 21)
    assert row.cost_usd == 0.003


def test_distinct_models_get_distinct_rows_in_first_seen_order():
    # Per-dim judge configs may each pin their own model; blending them would have to
    # drop `model`, which is what makes the dollars re-derivable.
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_result(model="big", inp=100, out=50, cost=0.01))
    ledger.add_llm_result(_result(model="small", inp=10, out=5, cost=0.001))
    ledger.add_llm_result(_result(model="big", inp=1, out=1, cost=0.002))
    rows = ledger.rows()
    assert [r.model for r in rows] == ["big", "small"]
    assert rows[0].prompt_tokens == 101
    assert rows[0].cost_usd == 0.012
    assert rows[1].prompt_tokens == 10


def test_unreported_reasoning_stays_none():
    ledger = RoleUsageLedger(role="simulator")
    ledger.add_llm_result(_result(reasoning=None))
    ledger.add_llm_result(_result(reasoning=None))
    assert ledger.rows()[0].reasoning_tokens is None


def test_reported_zero_reasoning_is_kept_as_zero():
    ledger = RoleUsageLedger(role="simulator")
    ledger.add_llm_result(_result(reasoning=0))
    assert ledger.rows()[0].reasoning_tokens == 0


def test_partially_reported_reasoning_sums_what_was_observed():
    # One silent call must not discard the other call's real measurement.
    ledger = RoleUsageLedger(role="simulator")
    ledger.add_llm_result(_result(reasoning=None))
    ledger.add_llm_result(_result(reasoning=7))
    assert ledger.rows()[0].reasoning_tokens == 7


def test_unreported_cost_leaves_cost_and_price_source_none():
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_result(cost=None))
    (row,) = ledger.rows()
    assert row.cost_usd is None
    assert row.price_source is None


def test_reported_zero_cost_is_an_observation():
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_result(cost=0.0))
    (row,) = ledger.rows()
    assert row.cost_usd == 0.0
    assert row.price_source == _SOURCE


def test_missing_attributes_on_a_test_double_read_as_unreported():
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(SimpleNamespace(content="hi"))
    (row,) = ledger.rows()
    assert row.model is None
    # Absent is unreported, not a measured zero — the same rule the token fields hold
    # everywhere else in this module.
    assert (row.prompt_tokens, row.completion_tokens) == (None, None)
    assert row.reasoning_tokens is None
    assert row.cost_usd is None


def test_empty_model_string_normalizes_to_none():
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_result(model=""))
    assert ledger.rows()[0].model is None


def test_the_price_source_is_what_the_contribution_named():
    ledger = RoleUsageLedger(role="candidate")
    ledger.add(
        model="m", prompt_tokens=0, completion_tokens=0, reasoning_tokens=None, cost_usd=0.005, price_source="rate_card"
    )
    assert ledger.rows()[0].price_source == "rate_card"


def test_a_cost_whose_client_named_no_source_keeps_none_rather_than_an_assumed_provider():
    """The engine names no provider: dollars nobody attributed stay unattributed."""
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_result(cost=0.002, price_source=None))
    ledger.add(model="n", prompt_tokens=1, completion_tokens=1, reasoning_tokens=None, cost_usd=0.001)
    assert [(row.cost_usd, row.price_source) for row in ledger.rows()] == [(0.002, None), (0.001, None)]


def test_dollars_priced_two_ways_stay_in_two_rows():
    """One model's calls priced from two sources would make one row's dollars un-re-derivable."""
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_result(cost=0.002, price_source="provider:reported"))
    ledger.add_llm_result(_result(cost=0.003, price_source="rate_card"))
    rows = ledger.rows()
    assert [(row.model, row.cost_usd, row.price_source) for row in rows] == [
        ("m", 0.002, "provider:reported"),
        ("m", 0.003, "rate_card"),
    ]


def _work(**spend):
    """One live piece of background work on its own model, reporting ``spend``."""
    return AsyncDelivery(tool="scout_ahead", status="delivered", model="inner", substituted=False, **spend)


def test_an_async_delivery_names_its_own_price_source():
    (row,) = async_delivery_usage([_work(cost_usd=0.01, price_source="rate_card")], rate_table=None)
    assert (row.role, row.model, row.cost_usd, row.price_source) == ("inner_agent", "inner", 0.01, "rate_card")


def test_an_async_delivery_naming_no_price_source_is_stored_unattributed():
    (row,) = async_delivery_usage([_work(cost_usd=0.01)], rate_table=None)
    assert (row.cost_usd, row.price_source) == (0.01, None)


def test_work_still_in_flight_when_the_cell_ended_is_costed():
    """An undelivered scout spent on its model all the same; dropping it understates the cell."""
    undelivered = AsyncDelivery(
        tool="scout_ahead", status="undelivered", model="inner", substituted=False, cost_usd=0.02, input_tokens=40
    )
    (row,) = async_delivery_usage([undelivered], rate_table=None)
    assert (row.role, row.cost_usd, row.prompt_tokens, row.completion_tokens) == ("inner_agent", 0.02, 40, None)


def test_an_unreported_call_count_stays_unknown_rather_than_one():
    (row,) = async_delivery_usage([_work(cost_usd=0.01)], rate_table=None)
    assert row.call_count is None


def test_external_calls_are_priced_at_their_own_provider_s_rate():
    table = ExternalRateTable(rates={("search_api", "credits"): 0.008})
    work = _work(
        external_spend=[AsyncExternalSpend(provider="search_api", calls=2, provider_units=4, provider_unit="credits")]
    )
    (row,) = async_delivery_usage([work], rate_table=table)
    assert (row.role, row.provider, row.call_count, row.provider_units, row.cost_usd) == (
        "external",
        "search_api",
        2,
        4,
        0.032,
    )


def test_work_that_reports_no_spend_contributes_no_row():
    assert async_delivery_usage([_work()], rate_table=None) == []
    assert async_delivery_usage(None, rate_table=None) == []


@pytest.mark.parametrize(
    "spend",
    [
        {"cost_usd": 0.01},
        {"input_tokens": 3},
        {"output_tokens": 3},
        {"reasoning_tokens": 1},
        {"llm_calls": 1},
        {"external_spend": [AsyncExternalSpend(provider="search_api", calls=1)]},
    ],
)
def test_a_substituted_delivery_cannot_carry_spend(spend):
    """A replayed or seeded payload spent nothing, so no figure of spend can ride in on one."""
    with pytest.raises(ValidationError, match="spent nothing"):
        AsyncDelivery(tool="scout_ahead", status="delivered", substituted=True, **spend)
    # The same spend on the live arm is a real report.
    assert AsyncDelivery(tool="scout_ahead", status="delivered", substituted=False, **spend).reported_spend


def test_a_price_source_needs_the_cost_it_names():
    with pytest.raises(ValidationError, match="price_source"):
        _work(price_source="rate_card")
    assert _work(price_source="rate_card", cost_usd=0.01).price_source == "rate_card"


def test_external_units_are_reported_with_their_unit_or_not_at_all():
    with pytest.raises(ValidationError, match="together"):
        AsyncExternalSpend(provider="search_api", calls=1, provider_units=2)
    with pytest.raises(ValidationError, match="together"):
        AsyncExternalSpend(provider="search_api", calls=1, provider_unit="credits")
    assert (
        AsyncExternalSpend(provider="search_api", calls=1, provider_units=2, provider_unit="credits").provider_units
        == 2
    )


# ---------------------------------------------------------------------------
# A provider-reported charge crosses the delivery seam
# ---------------------------------------------------------------------------

#: A run that prices search credits and holds no rate for images.
_SEARCH_ONLY = ExternalRateTable(rates={("search_api", "credits"): 0.008})


def _images(**charge):
    """Background work that generated two images at a provider billing per image."""
    return _work(cost_usd=0.01, external_spend=[AsyncExternalSpend(provider="image_api", calls=2, **charge)])


def _external_row(rows):
    (row,) = [row for row in rows if row.role == "external"]
    return row


def test_a_provider_reported_charge_lands_as_the_external_cost_with_no_rate_table():
    row = _external_row(async_delivery_usage([_images(money=0.08)], rate_table=None))
    assert (row.provider, row.call_count, row.cost_usd, row.price_source) == (
        "image_api",
        2,
        0.08,
        "image_api:reported",
    )


def test_a_provider_reported_charge_wins_over_the_declared_rate():
    table = ExternalRateTable(rates={("image_api", "images"): 1.0})
    work = _work(
        external_spend=[
            AsyncExternalSpend(provider="image_api", calls=2, provider_units=2, provider_unit="images", money=0.08)
        ]
    )
    row = _external_row(async_delivery_usage([work], rate_table=table))
    assert (row.cost_usd, row.price_source) == (0.08, "image_api:reported")


def test_a_reported_zero_charge_is_a_priced_zero_not_unknown():
    row = _external_row(async_delivery_usage([_images(money=0.0)], rate_table=None))
    assert (row.cost_usd, row.price_source) == (0.0, "image_api:reported")


def test_calls_with_neither_a_charge_nor_a_rate_are_unknown_never_zero():
    row = _external_row(async_delivery_usage([_images()], rate_table=_SEARCH_ONLY))
    assert row.cost_usd is None and row.price_source is None and row.call_count == 2


def test_a_cells_cost_carries_a_charge_its_run_holds_no_rate_for():
    """The run's composition claims external dollars; a reported charge prices calls no rate covers."""
    deliveries = [_images(money=0.08)]
    usage = async_delivery_usage(deliveries, rate_table=_SEARCH_ONLY)
    assert cell_cost(usage, async_deliveries=deliveries, rate_table=_SEARCH_ONLY) == pytest.approx(0.01 + 0.08)


def test_a_cells_cost_is_unknown_when_a_call_has_neither_a_charge_nor_a_rate():
    deliveries = [_images()]
    usage = async_delivery_usage(deliveries, rate_table=_SEARCH_ONLY)
    assert cell_cost(usage, async_deliveries=deliveries, rate_table=_SEARCH_ONLY) is None


def test_a_cells_cost_prices_units_at_the_rate_when_no_charge_was_reported():
    deliveries = [
        _work(
            cost_usd=0.01,
            external_spend=[
                AsyncExternalSpend(provider="search_api", calls=2, provider_units=4, provider_unit="credits")
            ],
        )
    ]
    usage = async_delivery_usage(deliveries, rate_table=_SEARCH_ONLY)
    assert cell_cost(usage, async_deliveries=deliveries, rate_table=_SEARCH_ONLY) == pytest.approx(0.01 + 0.032)


def test_a_negative_charge_is_refused():
    with pytest.raises(ValidationError, match="money"):
        AsyncExternalSpend(provider="image_api", calls=1, money=-0.01)
    assert AsyncExternalSpend(provider="image_api", calls=1, money=0.0).money == 0.0


def test_the_conversion_to_the_metering_vocabulary_carries_every_reported_field():
    """The two readers of a delivery's spend share one conversion; a field it drops reaches neither.

    Held structurally: every field the report declares must arrive under its own name, so a field
    added to ``AsyncExternalSpend`` without a home on ``ExternalSpend`` (or dropped on the way) fails here.
    """
    report = AsyncExternalSpend(provider="image_api", calls=3, provider_units=3, provider_unit="images", money=0.12)
    converted = report.as_external_spend()
    assert {name: getattr(converted, name) for name in AsyncExternalSpend.model_fields} == report.model_dump()


# ---------------------------------------------------------------------------
# Cost views — derived from role membership, never stored
# ---------------------------------------------------------------------------


def _row(role, cost):
    return RoleUsage(role=role, cost_usd=cost, price_source=_SOURCE if cost is not None else None)


def test_production_replicating_excludes_judge_and_simulator():
    usage = [
        _row("candidate", 1.0),
        _row("judge", 10.0),
        _row("simulator", 100.0),
        _row("inner_agent", 0.5),
        _row("external", 0.25),
    ]
    assert production_replicating_cost(usage, substituted_deliveries=0) == 1.75
    assert program_cost(usage) == 111.75


def test_cost_views_of_an_empty_capture_are_none_not_zero():
    assert production_replicating_cost([], substituted_deliveries=0) is None
    assert program_cost([]) is None


def test_cost_views_skip_rows_that_observed_no_cost():
    usage = [_row("candidate", None), _row("judge", 2.0)]
    assert production_replicating_cost(usage, substituted_deliveries=0) is None
    assert program_cost(usage) == 2.0


def test_cost_views_sum_multiple_rows_of_the_same_role():
    usage = [_row("judge", 1.0), _row("judge", 2.0)]
    assert program_cost(usage) == 3.0
    assert production_replicating_cost(usage, substituted_deliveries=0) is None


# ---------------------------------------------------------------------------
# Call counts + the external role (counted, never priced)
# ---------------------------------------------------------------------------


def test_call_count_tracks_calls_per_model():
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(_result(model="a"))
    ledger.add_llm_result(_result(model="a"))
    ledger.add_llm_result(_result(model="b"))
    rows = {row.model: row.call_count for row in ledger.rows()}
    assert rows == {"a": 2, "b": 1}


def test_add_accepts_a_batch_of_calls():
    # A role that reports its work in aggregate (an inner agent's rounds) contributes many
    # calls in one fold.
    ledger = RoleUsageLedger(role="inner_agent")
    ledger.add(model="m", prompt_tokens=100, completion_tokens=50, reasoning_tokens=None, cost_usd=0.01, calls=4)
    (row,) = ledger.rows()
    assert row.call_count == 4
    assert row.prompt_tokens == 100


def test_external_row_counts_calls_without_claiming_tokens_or_cost():
    # Most external providers report no per-call dollar figure. Volume is the only honest
    # quantity — tokens stay None because these aren't LLM calls at all, and a 0 would be a
    # claim about an inapplicable measure.
    ledger = RoleUsageLedger.for_external(None)
    ledger.add_external(ExternalSpend(provider="search_api", calls=3, provider_units=6, provider_unit="credits"))
    (row,) = ledger.rows()
    assert row.role == "external"
    assert row.call_count == 3
    assert row.provider == "search_api"
    assert row.provider_units == 6
    assert row.prompt_tokens is None
    assert row.completion_tokens is None
    assert row.reasoning_tokens is None
    assert row.cost_usd is None
    assert row.price_source is None


def test_external_calls_accumulate_across_deliveries():
    ledger = RoleUsageLedger.for_external(None)
    ledger.add_external(ExternalSpend(provider="search_api", calls=2, provider_units=4, provider_unit="credits"))
    ledger.add_external(ExternalSpend(provider="search_api", calls=5, provider_units=10, provider_unit="credits"))
    (row,) = ledger.rows()
    assert row.call_count == 7
    assert row.provider_units == 14


def test_external_row_never_enters_a_cost_view():
    """An uncosted external row must not drag a cost view to a wrong total."""
    usage = [
        RoleUsage(role="candidate", cost_usd=1.0, price_source=_SOURCE),
        RoleUsage(role="external", call_count=9),
    ]
    assert production_replicating_cost(usage, substituted_deliveries=0) == 1.0
    assert program_cost(usage) == 1.0


def test_partially_reported_token_counts_sum_what_was_observed():
    """One silent call must not discard another call's real token measurement — the same
    rule reasoning follows, applied to prompt/completion."""
    ledger = RoleUsageLedger(role="inner_agent")
    ledger.add(model="m", prompt_tokens=None, completion_tokens=None, reasoning_tokens=None, cost_usd=0.01)
    ledger.add(model="m", prompt_tokens=100, completion_tokens=40, reasoning_tokens=None, cost_usd=0.02)
    (row,) = ledger.rows()
    assert row.prompt_tokens == 100
    assert row.completion_tokens == 40
    assert row.cost_usd == 0.03


def test_cost_observed_with_no_tokens_keeps_tokens_unknown():
    """An evicted inner-agent trace reports dollars but no counts."""
    ledger = RoleUsageLedger(role="inner_agent")
    ledger.add(model=None, prompt_tokens=None, completion_tokens=None, reasoning_tokens=None, cost_usd=0.05)
    (row,) = ledger.rows()
    assert row.cost_usd == 0.05
    assert row.prompt_tokens is None
    assert row.completion_tokens is None


# ---------------------------------------------------------------------------
# resolve_result_usage — what a read surface should show, decided in one place
# ---------------------------------------------------------------------------

# The reconstruction these tests once guarded against read `EvalResult.trace`. That field no
# longer exists — the payload moved to a sibling `EvalTrace` document — so the property is now
# guaranteed by the type rather than by a resolver branch. A fixture handing `trace=` to
# `EvalResult` used to be dropped in silence; now it raises, which is a better failure
# but not a different conclusion — a fixture asserting against a hand-built shape still proves
# nothing about the producer. A stored document still embedding its payloads is refused on read like
# any other undeclared key (`test_base.py`'s strict-read tests).


def test_captured_rows_are_returned_untouched():
    captured = [RoleUsage(role="candidate", model="m", prompt_tokens=10, cost_usd=0.5)]
    resolved = resolve_result_usage(make_eval_result(usage=captured))
    assert resolved.source == "captured"
    assert resolved.partial is False
    assert resolved.usage == captured


def test_an_empty_capture_stays_empty():
    """``usage=[]`` is a live cell that attributed no roles — a factory failure, say.

    It must not be re-read as "nothing is known": the empty list IS the observation.
    """
    resolved = resolve_result_usage(make_eval_result(usage=[]))
    assert resolved.source == "captured"
    assert resolved.usage == []


# ---------------------------------------------------------------------------
# Substituted deliveries — the disclosure that keeps a prod-cost figure honest
# ---------------------------------------------------------------------------


def _deliveries(*, substituted: bool) -> list[AsyncDelivery]:
    """The result's record of one piece of background work, on the seed arm or the live one."""
    return [
        AsyncDelivery(
            tool="scout",
            status="delivered",
            model=None if substituted else "scout/model",
            delivered_items=2,
            substituted=substituted,
        )
    ]


def test_a_substituted_delivery_is_counted_from_the_delivery_itself():
    result = make_eval_result(async_deliveries=_deliveries(substituted=True))
    assert count_substituted_deliveries(result) == 1


def test_a_live_delivery_counts_as_no_substitution():
    result = make_eval_result(async_deliveries=_deliveries(substituted=False))
    assert count_substituted_deliveries(result) == 0


def test_a_trace_stripped_result_still_discloses_its_substitution():
    """The analytic read surfaces drop ``trace`` at the database, and this guard runs there.

    The guard's aggregate consumers — the frontier and ``run_summary``'s cost summary —
    read results that carry no trace at all. A guard that read the trace counted 0 on a
    substituted result and published a candidate-only partial sum as a production cost,
    making a replay run look genuinely cheaper than the live run it replayed. The
    substitution has to survive the strip.

    (Export, pivot, history and the cost estimate never reach this guard — they project raw
    blended ``cost_usd`` — so they are understated for replays by a separate route.)
    """
    result = make_eval_result(async_deliveries=_deliveries(substituted=True))
    assert count_substituted_deliveries(result) == 1
    assert (
        production_replicating_cost(
            [RoleUsage(role="candidate", cost_usd=0.0032)],
            substituted_deliveries=count_substituted_deliveries(result),
        )
        is None
    )


def test_the_resolved_view_carries_the_disclosure_on_the_captured_arm():
    """The captured arm is the live-run arm, and the one a seeded run takes.

    It does not otherwise walk the trace at all, so a disclosure derived only from the
    trace would be absent exactly where it matters.
    """
    result = make_eval_result(
        usage=[RoleUsage(role="candidate", model="sonnet", cost_usd=0.001)],
        async_deliveries=_deliveries(substituted=True),
    )
    resolved = resolve_result_usage(result)

    assert resolved.source == "captured"
    assert resolved.substituted_deliveries == 1


def test_a_substituted_result_withholds_its_production_cost():
    """Disclosed-unknown, not an understated number.

    The inner agent's model dollars and the search credits were never spent, so summing the
    observed rows would report a figure that understates production while claiming to be
    its subset — failing toward "cheaper than reality", the dangerous direction for a
    capacity or pricing decision.
    """
    usage = [
        RoleUsage(role="candidate", model="sonnet", cost_usd=1.0),
        RoleUsage(role="inner_agent", model="inner/model", cost_usd=0.5),
    ]

    assert production_replicating_cost(usage, substituted_deliveries=0) == 1.5
    assert production_replicating_cost(usage, substituted_deliveries=1) is None


def test_the_program_cost_still_reports_what_the_eval_really_spent():
    """Substitution withholds the production axis, never the program one.

    A seeded run genuinely spent less, and that figure is a true statement about the
    eval's own ledger — the axis phase 4 established stays correct under substitution.
    """
    usage = [
        RoleUsage(role="candidate", model="sonnet", cost_usd=1.0),
        RoleUsage(role="judge", model="judge/model", cost_usd=0.25),
    ]
    assert program_cost(usage) == 1.25


def test_count_substituted_deliveries_of_an_unobserved_lifecycle_is_zero():
    """A result whose kind reports no delivery stream had no delivery the harness could substitute."""
    from threetears.evals.schema.models import EvalResult
    from threetears.evals.kernel.usage_capture import count_substituted_deliveries

    base = {
        **result_capture_defaults(),
        "scope_id": "u",
        "eval_run_id": "r1",
        "test_case_id": "tc1",
        "model": "m1",
        "k_iteration": 1,
    }
    assert count_substituted_deliveries(EvalResult(**base)) == 0
    substituted = EvalResult(**base, async_deliveries=_deliveries(substituted=True))
    assert count_substituted_deliveries(substituted) == 1
