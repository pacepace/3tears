"""The price of a trade: the price figure keys, and the analyst's ``price_tradeoff`` over them.

Contracts held here (``threetears.evals.analysis.report.figure_keys``'s price families,
``threetears.evals.analysis.analyst.price``, ``ops.analyst_price``, ``actions.analyst_price``):

- **the keys**: ``price.<from>.<to>.<gain>.<field>`` and ``cost_per_month.<arm>`` read, on r4 (the tradeoff
  reference) and on every fixture holding costs, the values computed here from the input's own fields: the gain from
  the contrasts against the control where each arm holds one, else from the arms' means, signed toward better; the
  cost gap from the arms' costs; the price per point and its ends off the gain's interval; the monthly figures at the
  declared volume. An interval reaching zero leaves the high end unbounded, refused so, never a number; a move one
  option dominates has no price; a monthly figure without a declared volume is unstated;
- **the frontier**: on r4 it is the reference's (the cheaper option and the control, the two options a guardrail
  removes off it), and on every fixture it is the options best at some price per point, found here by sweeping the
  price; every option off it says why, with the figures that show it;
- **the tool**: an interval crossing zero gives an unbounded high end, stated in words; with no declared volume no
  monthly figure is given and a note says so; no cost is refused naming what is missing, an unknown gain axis with
  campaign_overview as the remedy; every fixture snapshot, on every axis, answers or refuses, never raises; the pages
  cover every option once; through the action, every call ``ANALYST_CALLS`` makes answers alike twice, within the
  budget, with every number a figure whose key reads it.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest

from packages.evals.tests.analyst_support import (
    TEXT_BUDGET,
    action_calls,
    fixture_inputs,
    fixture_snapshots,
)
from packages.evals.tests.fixtures.report_plans.misfits import MISFITS, reference_data
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from threetears.evals.actions import Action, MountedTool, analyst_actions, analyst_tools, eval_catalogue
from threetears.evals.actions.analyst import ANALYST_CUT
from threetears.evals.actions.analyst_analyses import ANALYSES_WORKFLOW
from threetears.evals.actions.analyst_price import analyst_price_actions
from threetears.evals.analysis.analyst.price import Price, PriceTradeoff, price_tradeoff
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import figure_key_grammar, key_subjects, render, resolve
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import RefusedInput

#: What a key's refusal says of a figure the input leaves unstated.
UNSTATED = "which the input does not state"

#: The fields of a listing that page it, which are counts of rows and never figures.
PAGING = frozenset({"total", "offset", "next_offset", "limit"})

#: The fields every cited figure carries, by which one is found in a result's JSON.
FIGURE_FIELDS = frozenset({"key", "reads", "value", "kind", "scale", "unit"})


def _value(key: str, held: ReportInput) -> Any:
    return resolve(key, held).value


def _r4() -> ReportInput:
    return fixture_inputs()["r4"]


# =============================================================================
# The keys, on r4: every figure computed here from the input's own fields
# =============================================================================


def _r4_expected(first: str, second: str) -> dict[str, Any]:
    """Every price field of the move from ``first`` to ``second`` on r4's persona, computed off the raw input: each
    arm placed by its contrast against the control (the control at zero), cost per 1,000 replies from the measure
    stated in it."""
    data = reference_data("r4")
    contrasts = {c["arm"]: c for c in data["contrasts"] if c["measure"] == "persona"}
    cost = {arm: stat["mean"] for arm, stat in data["measures"]["cost_per_1k"]["per_arm"].items()}
    volume = data["meta"]["volume_per_month"]
    place = {arm: contrasts[arm]["delta"] if arm in contrasts else 0.0 for arm in cost}
    gain = place[second] - place[first]
    gap = cost[second] - cost[first]
    expected: dict[str, Any] = {"gain": gain, "cost_gap": gap, "monthly_cost_gap": gap * volume / 1000}
    candidate = next((arm for arm in (first, second) if arm != "prod"), None)
    if "prod" in (first, second) and candidate is not None:
        low, high = contrasts[candidate]["interval"]
        interval = (low, high) if second == candidate else (-high, -low)
        expected["gain_interval"] = interval
    if gain * gap > 0:
        expected["per_point"] = gap / gain
        expected["monthly_per_point"] = gap / gain * volume / 1000
        if "gain_interval" in expected:
            low, high = expected["gain_interval"]
            far, near = (high, low) if gain > 0 else (low, high)
            expected["per_point_low"] = gap / far
            expected["monthly_per_point_low"] = gap / far * volume / 1000
            if near * gain > 0:
                expected["per_point_high"] = gap / near
                expected["monthly_per_point_high"] = gap / near * volume / 1000
    return expected


_R4_ARMS = ("prod", "f64", "f16", "cor")


@pytest.mark.parametrize("move", [(a, b) for a in _R4_ARMS for b in _R4_ARMS if a != b], ids="->".join)
def test_each_r4_price_reads_what_its_contrasts_costs_and_volume_compute(move: tuple[str, str]) -> None:
    held = _r4()
    first, second = move
    expected = _r4_expected(first, second)
    for field in (
        "gain",
        "gain_interval",
        "cost_gap",
        "per_point",
        "per_point_low",
        "per_point_high",
        "monthly_cost_gap",
        "monthly_per_point",
        "monthly_per_point_low",
        "monthly_per_point_high",
    ):
        key = f"price.{first}.{second}.persona.{field}"
        if field not in expected:
            with pytest.raises(UnresolvableReference):
                resolve(key, held)
            continue
        assert _value(key, held) == pytest.approx(expected[field]), key


def test_r4s_switch_to_the_cheaper_option_reads_its_saving_per_point_lost_a_month() -> None:
    """The reference's own trade: the cheaper option saves the control's cost less its own, 12 million replies a
    month, for the 6.1 points of persona it loses."""
    held = _r4()
    assert render("{price.f64.prod.persona.monthly_cost_gap}", held, {}) == "+$7,560.00"
    assert _value("price.f64.prod.persona.monthly_per_point", held) == pytest.approx((2.1 - 1.47) * 12_000 / 6.1)
    assert render("{price.f64.prod.persona.gain}", held, {}) == "+6.1"
    assert render("{price.f64.prod.persona.gain_interval}", held, {}) == "+4.9 to +7.3"


def test_a_lower_is_better_measure_gains_toward_its_better_side() -> None:
    """r4's latency states no mean, so it places no arm; r2's calls per task do, and an arm making fewer gains."""
    with pytest.raises(UnresolvableReference, match=UNSTATED):
        resolve("price.f64.prod.latency_s.gain", _r4())
    data = MISFITS["r2 without its gates"]()
    held = fixture_inputs()["r2 without its gates"]
    calls = data["measures"]["calls_per_task"]
    assert calls["higher_is_better"] is False
    first, second = "atlas-m.off.r0", "brisk-s.off.r0"
    gain = _value(f"price.{first}.{second}.calls_per_task.gain", held)
    assert gain == pytest.approx(calls["per_arm"][first]["mean"] - calls["per_arm"][second]["mean"])


def test_with_no_control_a_gain_reads_the_arms_means_and_states_no_interval() -> None:
    data = MISFITS["r2 without its gates"]()
    held = fixture_inputs()["r2 without its gates"]
    rate = {arm: stat["mean"] for arm, stat in data["measures"]["pass_rate"]["per_arm"].items()}
    cost = {arm: stat["mean"] for arm, stat in data["measures"]["cost_per_1k"]["per_arm"].items()}
    first, second = "brisk-s.off.r0", "atlas-m.low.r1"
    prefix = f"price.{first}.{second}.pass_rate"
    assert _value(f"{prefix}.gain", held) == pytest.approx(rate[second] - rate[first])
    assert _value(f"{prefix}.cost_gap", held) == pytest.approx(cost[second] - cost[first])
    assert _value(f"{prefix}.per_point", held) == pytest.approx(
        (cost[second] - cost[first]) / (rate[second] - rate[first])
    )
    for field in ("gain_interval", "per_point_low", "per_point_high"):
        with pytest.raises(UnresolvableReference, match=UNSTATED):
            resolve(f"{prefix}.{field}", held)


def test_a_price_per_1m_replies_is_priced_per_1000() -> None:
    data = reference_data("r3")
    held = fixture_inputs()["r3"]
    per_million = {arm: stat["mean"] for arm, stat in data["measures"]["cost_per_1m"]["per_arm"].items()}
    first, second = "brisk-s.few", "atlas-m.few"
    gap = _value(f"price.{first}.{second}.macro_f1.cost_gap", held)
    assert gap == pytest.approx((per_million[second] - per_million[first]) / 1000)
    assert _value(f"cost_per_month.{first}", held) == pytest.approx(
        per_million[first] / 1000 * data["meta"]["volume_per_month"] / 1000
    )


def test_a_share_gain_is_priced_per_point() -> None:
    """Kairo's persona dimension is a share: a point is a hundredth of it, and the contrasts place its arms."""
    held = frozen_input(fixture_snapshots("s")["kairo round 2"])
    contrast = next(
        c for c in held.contrasts if c.arm == "qwen3.8-flash" and c.measure == "persona.booth_stays_offstage"
    )
    assert contrast.delta is not None
    prefix = "price.qwen3.8-flash.September.persona.booth_stays_offstage"
    gain = _value(f"{prefix}.gain", held)
    assert gain == pytest.approx(-contrast.delta)
    gap = _value("cost_per_1k.September", held) - _value("cost_per_1k.qwen3.8-flash", held)
    assert _value(f"{prefix}.cost_gap", held) == pytest.approx(gap)
    with pytest.raises(UnresolvableReference, match="is no price"):
        resolve(f"{prefix}.per_point", held)
    other = "price.September.1017.persona.booth_stays_offstage"
    held_1017 = next(c for c in held.contrasts if c.arm == "1017" and c.measure == "persona.booth_stays_offstage")
    assert held_1017.delta is not None
    one_gap = _value("cost_per_1k.1017", held) - _value("cost_per_1k.September", held)
    assert _value(f"{other}.per_point", held) == pytest.approx(one_gap / (held_1017.delta * 100))


def _crossing_r4() -> dict[str, Any]:
    """r4 with Corvid's speculation inside its guardrail and its persona interval reaching past zero."""
    data = copy.deepcopy(reference_data("r4"))
    data["measures"]["specul"]["per_arm"]["cor"]["mean"] = 96.0
    for contrast in data["contrasts"]:
        if contrast["arm"] == "cor" and contrast["measure"] == "persona":
            contrast["interval"] = [-3.4, 0.4]
    return data


def test_an_interval_reaching_zero_leaves_the_high_end_unbounded_and_refused_as_such() -> None:
    held = ReportInput.model_validate(_crossing_r4())
    prefix = "price.cor.prod.persona"
    assert _value(f"{prefix}.gain_interval", held) == pytest.approx((-0.4, 3.4))
    assert _value(f"{prefix}.per_point_low", held) == pytest.approx((2.1 - 1.62) / 3.4)
    for field in ("per_point_high", "monthly_per_point_high"):
        with pytest.raises(UnresolvableReference, match="is unbounded"):
            resolve(f"{prefix}.{field}", held)


def test_a_monthly_figure_without_a_declared_volume_is_unstated() -> None:
    held = fixture_inputs()["r5"]
    assert held.meta.volume_per_month is None
    for key in ("cost_per_month.September", "price.September.1017.persona.holds_character.monthly_cost_gap"):
        with pytest.raises(UnresolvableReference, match=UNSTATED):
            resolve(key, held)


def test_cost_per_month_is_cost_per_1000_at_the_declared_volume() -> None:
    held = _r4()
    for arm in _R4_ARMS:
        cost = reference_data("r4")["measures"]["cost_per_1k"]["per_arm"][arm]["mean"]
        assert _value(f"cost_per_month.{arm}", held) == pytest.approx(cost * 12_000)


def test_a_price_names_both_arms_and_is_derived() -> None:
    held = _r4()
    subjects = key_subjects("price.f64.prod.persona.per_point", held)
    assert subjects.arms == ("f64", "prod")
    assert resolve("price.f64.prod.persona.per_point", held).derived is True
    assert resolve("cost_per_month.f64", held).derived is True


def test_a_gain_on_a_measure_of_no_direction_names_no_figure() -> None:
    with pytest.raises(UnresolvableReference, match="names no figure"):
        resolve("price.brisk-s.off.r0.atlas-m.off.r0.outcome.gain", fixture_inputs()["r2 without its gates"])


def test_the_grammar_teaches_both_families_with_their_readings() -> None:
    lines = {line.split("`:", 1)[0]: line for line in figure_key_grammar().splitlines() if line.startswith("- `")}
    price = lines["- `price.<from>.<to>.<measure>.<field>"]
    assert "unbounded" in price and "monthly_per_point_high" in price and 'reads "$' in price
    assert 'reads "$' in lines["- `cost_per_month.<arm>"]


# =============================================================================
# The frontier
# =============================================================================


def test_r4s_frontier_is_the_cheaper_option_and_the_control_and_the_guardrails_remove_the_others() -> None:
    trade = price_tradeoff(_r4(), "persona")
    assert [row.arm for row in trade.frontier] == ["arm:f64", "arm:prod"]
    (step,) = trade.steps
    assert (step.from_arm, step.to_arm, step.bounds) == ("arm:f64", "arm:prod", "bounded")
    expected = _r4_expected("f64", "prod")
    for field in ("gain", "cost_gap", "per_point", "per_point_low", "per_point_high", "monthly_per_point"):
        figure = getattr(step, field)
        assert figure is not None and figure.value == pytest.approx(expected[field]), field
    off = {row.arm: row for row in trade.off_frontier}
    assert set(off) == {"arm:f16", "arm:cor"}
    assert all(row.reason == "gated" for row in off.values())
    assert "Music facts" in " ".join(off["arm:f16"].says) and "speculate" in " ".join(off["arm:cor"].says)
    assert trade.against_arm == "arm:prod"
    assert [move.from_arm for move in trade.against] == ["arm:f64"]
    assert trade.volume is not None and trade.volume.value == 12_000_000


def _positions(held: ReportInput, key: str) -> dict[str, float]:
    """Each arm's place on an axis, off the input's own fields: its contrast against the control where every other
    arm holds one, else its mean; signed so that higher is better."""
    table = held.judged.get(key) or held.measures[key]
    higher = True if key in held.judged else held.measures[key].higher_is_better
    sign = 1.0 if higher else -1.0
    control = next((arm.id for arm in held.arms if arm.role == "control"), None)
    contrasts = [c for c in held.contrasts if c.measure == key and c.delta is not None and c.arm != control]
    means = {arm: stat.mean for arm, stat in table.per_arm.items() if stat.mean is not None}
    holders = ({*means} | {c.arm for c in contrasts}) - {control}
    if control is not None and contrasts and all([c.arm for c in contrasts].count(arm) == 1 for arm in holders):
        return {control: 0.0} | {c.arm: sign * float(c.delta or 0.0) for c in contrasts}
    return {arm: sign * float(mean) for arm, mean in means.items()}


def _swept_frontier(held: ReportInput, key: str) -> set[str]:
    """The options best at some price per point, found by trying every price between two pairwise break points."""
    survivors = set(surviving_options(held))
    places = _positions(held, key)
    costs: dict[str, float] = {}
    for arm in survivors & set(places):
        try:
            costs[arm] = float(resolve(f"cost_per_1k.{arm}", held).value)  # type: ignore[arg-type]
        except UnresolvableReference:
            continue
    if not costs:
        return set()
    arms = sorted(costs)
    breaks = sorted(
        {
            (costs[b] - costs[a]) / (places[b] - places[a])
            for a in arms
            for b in arms
            if places[b] != places[a] and (costs[b] - costs[a]) / (places[b] - places[a]) > 0
        }
    )
    prices = [breaks[0] / 2] if breaks else [1.0]
    prices += [(low + high) / 2 for low, high in zip(breaks, breaks[1:], strict=False)]
    prices += [breaks[-1] * 2] if breaks else []
    best: set[str] = set()
    for price in prices:
        values = {arm: price * places[arm] - costs[arm] for arm in arms}
        top = max(values.values())
        winners = [arm for arm in arms if math.isclose(values[arm], top, rel_tol=1e-12, abs_tol=1e-12)]
        best.add(min(winners, key=lambda arm: [held_arm.id for held_arm in held.arms].index(arm)))
    return best


def _axes(held: ReportInput) -> list[str]:
    measures = [key for key, measure in held.measures.items() if measure.higher_is_better is not None]
    return list(dict.fromkeys([*held.judged, *measures]))


@pytest.mark.parametrize("name", sorted(fixture_inputs()))
def test_each_fixtures_frontier_is_the_options_best_at_some_price(name: str) -> None:
    held = fixture_inputs()[name]
    for key in _axes(held):
        trade = price_tradeoff(held, key)
        assert {row.arm.removeprefix("arm:") for row in trade.frontier} == _swept_frontier(held, key), key
        prices = [step.per_point for step in trade.steps]
        assert all(price is not None for price in prices), key
        values = [float(price.value) for price in prices if price is not None]  # type: ignore[arg-type]
        assert values == sorted(values), f"{key}: the flip prices rise along the frontier"


@pytest.mark.parametrize("name", sorted(fixture_inputs()))
def test_each_option_off_the_frontier_shows_why(name: str) -> None:
    held = fixture_inputs()[name]
    for key in _axes(held):
        trade = price_tradeoff(held, key, limit=200)
        for row in trade.off_frontier:
            assert row.says, row.arm
            if row.reason == "dominated":
                gain, gap = (float(figure.value) for figure in row.figures)  # type: ignore[arg-type]
                assert gain <= 0 <= gap, row
            if row.reason == "never_best":
                into, onward = (float(figure.value) for figure in row.figures)  # type: ignore[arg-type]
                assert into >= onward, row


# =============================================================================
# The tool
# =============================================================================


def test_a_step_whose_interval_crosses_zero_says_its_high_end_is_unbounded() -> None:
    held = ReportInput.model_validate(_crossing_r4())
    trade = price_tradeoff(held, "persona")
    assert [row.arm for row in trade.frontier] == ["arm:f64", "arm:cor", "arm:prod"]
    first, second = trade.steps
    assert (first.bounds, first.per_point_low, first.per_point_high) == ("unstated", None, None)
    assert "no interval" in first.says
    assert second.bounds == "unbounded_above" and second.per_point_high is None
    assert second.monthly_per_point_high is None and "unbounded above" in second.says
    assert second.per_point_low is not None and second.per_point_low.value == pytest.approx((2.1 - 1.62) / 3.4)


def test_with_no_declared_volume_no_monthly_figure_is_given_and_a_note_says_so() -> None:
    trade = price_tradeoff(fixture_inputs()["r5"], "persona.booth_stays_offstage")
    assert trade.volume is None
    assert any("meta.volume_per_month" in note for note in trade.notes)
    assert trade.steps and all(step.monthly_cost_gap is None for step in trade.steps)
    assert all(row.monthly_cost is None for row in trade.frontier)


def test_no_cost_is_refused_naming_what_is_missing() -> None:
    data = copy.deepcopy(reference_data("r4"))
    del data["measures"]["cost_per_1k"]
    with pytest.raises(RefusedInput) as refused:
        price_tradeoff(ReportInput.model_validate(data), "persona")
    assert "usd_per_1k" in refused.value.fault and "reply_cost_usd" in refused.value.fault
    assert "options_dominance" in (refused.value.remedy or "")


def test_a_survivor_with_no_cost_is_off_the_frontier_saying_so() -> None:
    data = copy.deepcopy(reference_data("r4"))
    del data["measures"]["cost_per_1k"]["per_arm"]["f64"]
    trade = price_tradeoff(ReportInput.model_validate(data), "persona")
    assert [row.arm for row in trade.frontier] == ["arm:prod"] and trade.steps == []
    (unpriced,) = (row for row in trade.off_frontier if row.reason == "no_cost")
    assert unpriced.arm == "arm:f64" and "no cost" in unpriced.says[0]


def test_an_unknown_gain_axis_is_refused_naming_campaign_overview() -> None:
    with pytest.raises(RefusedInput) as refused:
        price_tradeoff(_r4(), "no such measure")
    assert "campaign_overview" in (refused.value.remedy or "") and "'persona'" in (refused.value.remedy or "")


def test_an_axis_of_no_direction_and_an_unpriced_arm_filter_are_refused() -> None:
    with pytest.raises(RefusedInput, match="states no direction"):
        price_tradeoff(fixture_inputs()["r2"], "outcome")
    with pytest.raises(RefusedInput, match="names no option this tool prices") as refused:
        price_tradeoff(_r4(), "persona", arm="cor")
    assert "'f64'" in (refused.value.remedy or "")


def test_against_another_option_every_other_is_priced_against_it() -> None:
    trade = price_tradeoff(_r4(), "persona", arm="f64")
    assert trade.against_arm == "arm:f64"
    assert [(move.from_arm, move.to_arm) for move in trade.against] == [("arm:f64", "arm:prod")]


@pytest.mark.parametrize("name", sorted(fixture_inputs()))
def test_every_fixture_on_every_axis_answers_or_refuses_and_never_raises(name: str) -> None:
    held = fixture_inputs()[name]
    for key in [*held.measures, *held.judged, "no such measure"]:
        for arm in [None, *(held_arm.id for held_arm in held.arms)]:
            try:
                price_tradeoff(held, key, arm=arm)
            except RefusedInput:
                continue


def _arms_shown(trade: PriceTradeoff) -> set[str]:
    moved = {move.from_arm for move in trade.against} | {move.to_arm for move in trade.against}
    return (moved - {trade.against_arm}) | {row.arm for row in trade.off_frontier}


@pytest.mark.parametrize("name", ["r2 without its gates", "r4", "kairo round 2"])
def test_the_pages_cover_every_option_once(name: str) -> None:
    held = fixture_snapshots("s")[name]
    report_input = frozen_input(held)
    key = next(key for key in _axes(report_input) if price_tradeoff(report_input, key).steps)
    whole = price_tradeoff(report_input, key, limit=200)
    seen: list[str] = []
    offset: int | None = 0
    while offset is not None:
        page = price_tradeoff(report_input, key, limit=2, offset=offset)
        assert page.frontier == whole.frontier and page.steps == whole.steps
        seen += sorted(_arms_shown(page))
        offset = page.next_offset
    assert sorted(seen) == sorted(_arms_shown(whole))
    assert len(seen) == len(set(seen))


def test_the_action_is_an_analysis_on_the_analysts_cut_once_gathered() -> None:
    (action,) = analyst_price_actions()
    assert action.workflow == ANALYSES_WORKFLOW and action.permission == "read"
    assert action.example["gain_key"] in _r4().judged


# =============================================================================
# Through the action, on every fixture snapshot
# =============================================================================


def _price_action() -> Action:
    (action,) = analyst_price_actions()
    return replace(action, cut=ANALYST_CUT)


def _tool() -> MountedTool:
    """The analyst's tool, carrying price_tradeoff: from the catalogue once ``actions.analyst`` gathers
    :func:`analyst_price_actions` beside the other analyses, and until then added to it as a host's action."""
    (spec,) = analyst_tools()
    gathered = any(action.name == "price_tradeoff" for action in analyst_actions())
    return eval_catalogue(() if gathered else (_price_action(),)).mount(spec)


def _seeded() -> tuple[OpsFixture, dict[str, Any]]:
    fixture = ops_fixture()
    snapshots = fixture_snapshots(TOYHOST_SCOPE)
    for snapshot in snapshots.values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture, snapshots


def _numbers(node: Any, path: str = "") -> Iterator[tuple[str, dict[str, Any] | None]]:
    """Every number in a result's JSON, with where it sits and the cited figure holding it, if one does."""
    if isinstance(node, dict):
        if FIGURE_FIELDS <= node.keys():
            yield path, node
            return
        for key, item in node.items():
            if key not in PAGING:
                yield from _numbers(item, f"{path}.{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _numbers(item, f"{path}[{index}]")
    elif isinstance(node, int | float) and not isinstance(node, bool):
        yield path, None


async def test_every_call_answers_alike_twice_within_its_budget_on_every_figure_by_its_key() -> None:
    fixture, snapshots = off_loop(_seeded)
    tool = _tool()
    action = _price_action()
    called = 0
    for name, snapshot in snapshots.items():
        held = frozen_input(snapshot)
        for arguments in action_calls(action, held, snapshot.id):
            first = await tool.call(arguments, host=fixture.host, caller=CALLER)
            second = await tool.call(arguments, host=fixture.host, caller=CALLER)
            where = f"price_tradeoff {arguments} on {name}"
            assert first.ending == "ok", f"{where}: {first.text}"
            assert (first.text, first.structured) == (second.text, second.structured), where
            assert len(first.text) <= TEXT_BUDGET, f"{where}: {len(first.text)} characters"
            assert first.structured is not None
            for at, figure in _numbers(first.structured):
                assert figure is not None, f"{where}: {at} is a number no figure key names"
                expected = _value(figure["key"], held)
                expected = list(expected) if isinstance(expected, tuple) else expected
                assert figure["value"] == expected, f"{where}: {figure['key']}"
            called += 1
    assert called >= len(snapshots), "every snapshot pricing something is called"


async def test_an_unknown_gain_axis_through_the_action_names_campaign_overview() -> None:
    fixture, snapshots = off_loop(_seeded)
    arguments = {"action": "price_tradeoff", "snapshot_id": snapshots["r4"].id, "gain_key": "no such measure"}
    outcome = await _tool().call(arguments, host=fixture.host, caller=CALLER)
    assert outcome.is_error and "campaign_overview" in outcome.text


def test_a_move_reads_back_under_its_row() -> None:
    (step,) = price_tradeoff(_r4(), "persona").steps
    assert isinstance(step, Price)
    assert step.per_point is not None and step.per_point.key == "price.f64.prod.persona.per_point"
