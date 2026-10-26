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
import re
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest

from packages.evals.tests.analyst_support import (
    TEXT_BUDGET,
    WIDE_FRONTIER,
    action_calls,
    budget_inputs,
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
from threetears.evals.actions.render import render_cited
from threetears.evals.analysis.analyst.analyses import gate_reads
from threetears.evals.analysis.analyst.price import Price, PriceTradeoff, price_tradeoff
from threetears.evals.analysis.report_gates import gate_findings
from threetears.evals.analysis.summary import dollars_text
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.numbers import WHOLE_FROM, format_number
from threetears.evals.analysis.report.cost_unit import REPLIES, per_count, per_noun
from threetears.evals.analysis.report.figure_keys import (
    cost_basis,
    figure_key_grammar,
    format_figure,
    key_subjects,
    render,
    resolve,
)
from threetears.evals.analysis.report_decision import surviving_options
from threetears.evals.analysis.analyst.blocks import block_catalog, block_render
from threetears.evals.analysis.report.block_spec import BLOCK_SOURCES
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_rows import PriceStep, query_rows
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.report_plan import ChartQuery

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
    """The reference's own trade, the frontier's one step: the cheaper option saves the dearer's cost less its own at
    the declared monthly volume, for the points of persona it loses; every figure computed off the raw input."""
    held = _r4()
    (step,) = price_tradeoff(held, "persona").steps
    cheaper, dearer = step.from_arm.removeprefix("arm:"), step.to_arm.removeprefix("arm:")
    expected = _r4_expected(cheaper, dearer)
    key = f"price.{cheaper}.{dearer}.persona"
    assert expected["monthly_cost_gap"] >= 1000, "the control: a saving a month of a thousand dollars or more"
    assert render(f"{{{key}.monthly_cost_gap}}", held, {}) == "+" + dollars_text(expected["monthly_cost_gap"])
    assert _value(f"{key}.monthly_per_point", held) == pytest.approx(expected["monthly_per_point"])
    assert _value(f"{key}.gain", held) == pytest.approx(expected["gain"])
    assert _value(f"{key}.gain_interval", held) == pytest.approx(expected["gain_interval"])


def test_a_lower_is_better_measure_gains_toward_its_better_side() -> None:
    """r4's latency states no mean, so it places no arm; r2's calls per task do, and an arm making fewer gains."""
    with pytest.raises(UnstatedFigure):
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
        with pytest.raises(UnstatedFigure):
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


def _crossing_figures() -> tuple[float, tuple[float, float]]:
    """The crossing input's move from its third option to the control on persona, off the raw data: what the control
    costs more per 1,000 replies, and the interval on the gain (the third option's contrast, turned around)."""
    data = _crossing_r4()
    cost = {arm: stat["mean"] for arm, stat in data["measures"]["cost_per_1k"]["per_arm"].items()}
    (low, high) = next(c["interval"] for c in data["contrasts"] if c["arm"] == "cor" and c["measure"] == "persona")
    return cost["prod"] - cost["cor"], (-high, -low)


def test_an_interval_reaching_zero_leaves_the_high_end_unbounded_and_refused_as_such() -> None:
    held = ReportInput.model_validate(_crossing_r4())
    prefix = "price.cor.prod.persona"
    gap, interval = _crossing_figures()
    assert _value(f"{prefix}.gain_interval", held) == pytest.approx(interval)
    assert _value(f"{prefix}.per_point_low", held) == pytest.approx(gap / interval[1])
    for field in ("per_point_high", "monthly_per_point_high"):
        with pytest.raises(UnresolvableReference, match="is unbounded"):
            resolve(f"{prefix}.{field}", held)


def test_a_monthly_figure_without_a_declared_volume_is_unstated() -> None:
    held = fixture_inputs()["r5"]
    assert held.meta.volume_per_month is None
    for key in ("cost_per_month.September", "price.September.1017.persona.holds_character.monthly_cost_gap"):
        with pytest.raises(UnstatedFigure):
            resolve(key, held)


def test_cost_per_month_is_cost_per_1000_at_the_declared_volume() -> None:
    held = _r4()
    data = reference_data("r4")
    for arm in _R4_ARMS:
        cost = data["measures"]["cost_per_1k"]["per_arm"][arm]["mean"]
        assert _value(f"cost_per_month.{arm}", held) == pytest.approx(cost * data["meta"]["volume_per_month"] / 1000)


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
    for arm in ("f16", "cor"):
        gates = off[f"arm:{arm}"].gates
        assert gates == gate_reads(_r4(), arm, "fail") and gates, arm
        failed = {finding.criterion for finding in gate_findings(_r4(), arm) if finding.status == "fail"}
        assert {gate.criterion for gate in gates} == failed, "each gate it fails is cited, by its keys"
    assert trade.against_arm == "arm:prod"
    assert [move.from_arm for move in trade.against] == ["arm:f64"]
    assert trade.volume is not None and trade.volume.value == reference_data("r4")["meta"]["volume_per_month"]


def _pooled_means(held: ReportInput, key: str) -> dict[str, float]:
    """Each arm's figure on a key pooled over the slices holding it, off the input's own fields: the tasks holding it,
    else the strata (of one stratification), else the cells; weighted by traffic share where every slice states one
    and they sum above zero (a slice at zero weighing nothing), else equally; an arm missing a weighed slice left out."""

    def holds(per_arm: Any) -> bool:
        return any(key in figures and figures[key].mean is not None for figures in per_arm.values())

    slices = [(task.share, task.per_arm) for task in held.tasks if holds(task.per_arm)]
    strata = [stratum for stratum in held.strata if holds(stratum.per_arm)]
    if not slices and strata and len({(stratum.aspect, stratum.dimension) for stratum in strata}) == 1:
        slices = [(stratum.share, stratum.per_arm) for stratum in strata]
    if not slices:
        slices = [(None, cell.per_arm) for cell in held.cells.values() if holds(cell.per_arm)]
    traffic = bool(slices) and all(share is not None for share, _ in slices) and sum(s or 0 for s, _ in slices) > 0
    weighed = [(float(share or 0) if traffic else 1.0, per_arm) for share, per_arm in slices]
    weighed = [(weight, per_arm) for weight, per_arm in weighed if weight > 0]
    pooled: dict[str, float] = {}
    for arm in {arm for _, per_arm in weighed for arm in per_arm}:
        means = [per_arm.get(arm, {}).get(key) for _, per_arm in weighed]
        if all(stat is not None and stat.mean is not None for stat in means):
            total = sum(weight for weight, _ in weighed)
            pooled[arm] = sum(w * float(s.mean) for (w, _), s in zip(weighed, means, strict=True)) / total  # type: ignore[union-attr,arg-type]
    return pooled


def _positions(held: ReportInput, key: str) -> dict[str, float]:
    """Each arm's place on an axis, off the input's own fields: its contrast against the control where every other
    arm holds one, else its mean, else (no arm holding a mean) its figure pooled over its slices; signed so that
    higher is better."""
    table = held.judged.get(key) or held.measures[key]
    higher = True if key in held.judged else held.measures[key].higher_is_better
    sign = 1.0 if higher else -1.0
    control = next((arm.id for arm in held.arms if arm.role == "control"), None)
    contrasts = [c for c in held.contrasts if c.measure == key and c.delta is not None and c.arm != control]
    means = {arm: stat.mean for arm, stat in table.per_arm.items() if stat.mean is not None}
    holders = ({*means} | {c.arm for c in contrasts}) - {control}
    if control is not None and contrasts and all([c.arm for c in contrasts].count(arm) == 1 for arm in holders):
        return {control: 0.0} | {c.arm: sign * float(c.delta or 0.0) for c in contrasts}
    if not means:
        return {arm: sign * mean for arm, mean in _pooled_means(held, key).items()}
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
        trade = price_tradeoff(held, key, limit=len(held.arms))
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
    gap, interval = _crossing_figures()
    assert second.per_point_low is not None and second.per_point_low.value == pytest.approx(gap / interval[1])


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
    key = next(key for key in _axes(report_input) if price_tradeoff(report_input, key, limit=200).steps)
    whole = price_tradeoff(report_input, key, limit=200)
    seen: list[str] = []
    frontier: list[Any] = []
    steps: list[Any] = []
    offset: int | None = 0
    while offset is not None:
        page = price_tradeoff(report_input, key, limit=2, offset=offset)
        assert (page.frontier_options, page.frontier_steps) == (whole.frontier_options, whole.frontier_steps)
        frontier += page.frontier
        steps += page.steps
        seen += sorted(_arms_shown(page))
        offset = page.next_offset
    assert sorted(seen) == sorted(_arms_shown(whole))
    assert len(seen) == len(set(seen))
    assert sorted(frontier, key=repr) == sorted(whole.frontier, key=repr), "each frontier option on one page"
    assert sorted(steps, key=repr) == sorted(whole.steps, key=repr), "each step on one page"


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


# =============================================================================
# W2 fixes: unpaired intervals, pooled gain axes, prices within a slice, what-ifs, a wide frontier paged
# =============================================================================


def _half_width(stat: Any) -> float | None:
    if stat.half_width is not None:
        return float(stat.half_width)
    if stat.ci is not None:
        return (float(stat.ci[1]) - float(stat.ci[0])) / 2
    return None


def _pooled_half_widths(held: ReportInput, key: str) -> dict[str, float]:
    """Each pooled arm's half-width off the raw input: the weighed slices' half-widths, each times its weight's share,
    combined as independent; only where every weighed slice states one."""
    pooled = _pooled_means(held, key)

    def holds(per_arm: Any) -> bool:
        return any(key in figures and figures[key].mean is not None for figures in per_arm.values())

    slices = [(task.share, task.per_arm) for task in held.tasks if holds(task.per_arm)]
    if not slices:
        slices = [(None, cell.per_arm) for cell in held.cells.values() if holds(cell.per_arm)]
    traffic = bool(slices) and all(share is not None for share, _ in slices) and sum(s or 0 for s, _ in slices) > 0
    weighed = [(float(share or 0) if traffic else 1.0, per_arm) for share, per_arm in slices]
    weighed = [(weight, per_arm) for weight, per_arm in weighed if weight > 0]
    total = sum(weight for weight, _ in weighed)
    found: dict[str, float] = {}
    for arm in pooled:
        widths = [_half_width(per_arm[arm][key]) for _, per_arm in weighed]
        if all(width is not None for width in widths):
            found[arm] = math.sqrt(
                sum((weight / total) ** 2 * width**2 for (weight, _), width in zip(weighed, widths, strict=True))  # type: ignore[operator]
            )
    return found


def _own_half_widths(held: ReportInput, key: str) -> dict[str, float]:
    """Each arm's own half-width on an axis off the raw input: its overall figure's, else (none overall) its pooled."""
    table = held.judged.get(key) or held.measures[key]
    if any(stat.mean is not None for stat in table.per_arm.values()):
        return {arm: width for arm, stat in table.per_arm.items() if (width := _half_width(stat)) is not None}
    return _pooled_half_widths(held, key)


@pytest.mark.parametrize("name", sorted(fixture_inputs()))
def test_an_unpaired_interval_is_the_gain_plus_or_minus_both_options_own_half_widths(name: str) -> None:
    held = fixture_inputs()[name]
    checked = 0
    for key in _axes(held):
        places = _positions(held, key)
        widths = _own_half_widths(held, key)
        arms = [arm.id for arm in held.arms if arm.id in places][:4]
        for first in arms:
            for second in arms:
                if first == second:
                    continue
                path = f"price.{first}.{second}.{key}.gain_interval_unpaired"
                if first not in widths or second not in widths:
                    with pytest.raises(UnresolvableReference):
                        resolve(path, held)
                    continue
                gain = places[second] - places[first]
                width = math.hypot(widths[first], widths[second])
                assert _value(path, held) == pytest.approx((gain - width, gain + width)), path
                checked += 1
    if name in ("r5", "r1", "r1 reading only bars"):
        assert checked, "the control: an input whose arms state their own intervals prices an unpaired one"


def test_a_move_between_two_candidates_carries_its_unpaired_interval_through_to_its_price() -> None:
    held = fixture_inputs()["r1 reading only bars"]
    key = next(key for key in _axes(held) if _own_half_widths(held, key) and price_tradeoff(held, key, limit=99).steps)
    trade = price_tradeoff(held, key, limit=len(held.arms))
    step = next(step for step in trade.steps if step.unpaired is not None and step.unpaired.bounds != "no_price")
    assert step.unpaired is not None and "unpaired" in step.unpaired.says
    first, second = step.from_arm.removeprefix("arm:"), step.to_arm.removeprefix("arm:")
    gap = _value(f"price.{first}.{second}.{key}.cost_gap", held)
    points = 100.0 if resolve(f"price.{first}.{second}.{key}.gain", held).scale == "share" else 1.0
    low, high = (end * points for end in step.unpaired.gain_interval.value)  # type: ignore[union-attr]
    assert step.unpaired.per_point_low is not None
    assert step.unpaired.per_point_low.value == pytest.approx(gap / high)
    if low > 0:
        assert step.unpaired.bounds == "bounded" and step.unpaired.per_point_high is not None
        assert step.unpaired.per_point_high.value == pytest.approx(gap / low)
    else:
        assert step.unpaired.bounds == "unbounded_above" and step.unpaired.per_point_high is None


@pytest.mark.parametrize("name", ["r1", "r1 reading only bars", "r3"])
def test_a_figure_held_only_per_slice_is_pooled_into_a_gain_axis_and_priced(name: str) -> None:
    """r1 reading only bars: its quality is held per task, so the trade put every model off the frontier as
    no_gain_figure (the W2 prototype's gap). Pooled, each is placed and the frontier is the swept one."""
    held = fixture_inputs()[name]
    pooled_keys = [
        key
        for key in _axes(held)
        if not any(stat.mean is not None for stat in (held.judged.get(key) or held.measures[key]).per_arm.values())
        and _pooled_means(held, key)
    ]
    assert pooled_keys, "the control: the input holds a gain axis only per slice"
    for key in pooled_keys:
        for arm, mean in _pooled_means(held, key).items():
            assert _value(f"pooled.{key}.{arm}.mean", held) == pytest.approx(mean), (key, arm)
        trade = price_tradeoff(held, key, limit=len(held.arms))
        assert not [row for row in trade.off_frontier if row.reason == "no_gain_figure"], key
        assert "pooled" in trade.basis
        assert {row.arm.removeprefix("arm:") for row in trade.frontier} == _swept_frontier(held, key), key


def test_pooled_weights_are_traffic_shares_where_stated_and_equal_weights_are_a_caveat() -> None:
    by_task = fixture_inputs()["r1 reading only bars"]
    assert all(task.share is not None for task in by_task.tasks), "the control: every task states its share"
    key = next(key for key in _axes(by_task) if _pooled_means(by_task, key))
    arm = next(iter(_pooled_means(by_task, key)))
    assert "traffic share" in str(_value(f"pooled.{key}.{arm}.weights", by_task))
    trade = price_tradeoff(by_task, key)
    assert not any("weighted equally" in note for note in trade.notes)
    by_cell = fixture_inputs()["r3"]
    key = next(
        key
        for key in _axes(by_cell)
        if _pooled_means(by_cell, key) and not by_cell.measures.get(key, by_cell.judged.get(key)).per_arm  # type: ignore[union-attr]
    )
    arm = next(iter(_pooled_means(by_cell, key)))
    assert "weighted equally" in str(_value(f"pooled.{key}.{arm}.weights", by_cell))
    assert _value(f"pooled.{key}.{arm}.slices", by_cell) == len(by_cell.cells)
    trade = price_tradeoff(by_cell, key)
    assert trade.notes[0].startswith("caveat:") or any(note.startswith("caveat:") for note in trade.notes)
    assert any("weighted equally" in note for note in trade.notes)


def _task_costs(held: ReportInput, task_id: str) -> dict[str, float]:
    task = next(task for task in held.tasks if task.id == task_id)
    return {
        arm: float(figures["cost_per_1k"].mean)
        for arm, figures in task.per_arm.items()
        if "cost_per_1k" in figures and figures["cost_per_1k"].mean is not None
    }  # type: ignore[arg-type]


def test_a_price_within_a_task_reads_the_tasks_own_cost_and_figures() -> None:
    held = fixture_inputs()["r1 reading only bars"]
    task = next(task for task in held.tasks if _task_costs(held, task.id))
    costs = _task_costs(held, task.id)
    key = next(
        key
        for key in _axes(held)
        if sum(key in figures and figures[key].mean is not None for figures in task.per_arm.values()) > 1
    )
    holding = [arm for arm, figures in task.per_arm.items() if key in figures and figures[key].mean is not None]
    first, second = holding[:2]
    sign = 1.0 if key in held.judged or held.measures[key].higher_is_better else -1.0
    gain = sign * (float(task.per_arm[second][key].mean) - float(task.per_arm[first][key].mean))  # type: ignore[arg-type]
    prefix = f"task.{task.id}.price.{first}.{second}.{key}"
    assert _value(f"{prefix}.gain", held) == pytest.approx(gain)
    assert _value(f"{prefix}.cost_gap", held) == pytest.approx(costs[second] - costs[first])
    assert "own cost in this slice" in str(_value(f"{prefix}.cost_basis", held))
    trade = price_tradeoff(held, key, slice_filter=task.id, limit=len(held.arms))
    assert trade.within == f"task.{task.id}" and trade.volume is None
    assert all(step.monthly_cost_gap is None and step.gain_interval is None for step in trade.steps)
    assert all(
        row.cost.key == f"task.{task.id}.{row.arm.removeprefix('arm:')}.cost_per_1k.mean" for row in trade.frontier
    )


def test_a_price_within_a_slice_holding_no_cost_reads_each_options_overall_cost() -> None:
    held = fixture_inputs()["r5"]
    stratum = held.strata[0]
    key = next(
        key
        for key in _axes(held)
        if sum(key in figures and figures[key].mean is not None for figures in stratum.per_arm.values()) > 1
    )
    arms = [arm for arm, figures in stratum.per_arm.items() if key in figures and figures[key].mean is not None]
    prefix = f"strata.{stratum.id}.price.{arms[0]}.{arms[1]}.{key}"
    assert "overall cost" in str(_value(f"{prefix}.cost_basis", held))
    expected = _value(f"cost_per_1k.{arms[1]}", held) - _value(f"cost_per_1k.{arms[0]}", held)
    assert _value(f"{prefix}.cost_gap", held) == pytest.approx(expected)


def test_a_slice_naming_nothing_is_refused_naming_the_slices() -> None:
    with pytest.raises(RefusedInput, match="names no task, stratum or cell") as refused:
        price_tradeoff(fixture_inputs()["r1"], "overall", slice_filter="no such slice")
    assert "'billing'" in (refused.value.remedy or "")


def test_a_what_if_prices_an_option_the_gates_remove_labelled_never_a_recommendation() -> None:
    held = _r4()
    gated = [row.arm.removeprefix("arm:") for row in price_tradeoff(held, "persona").off_frontier]
    assert gated, "the control: r4's guardrails remove options"
    for arm in gated:
        trade = price_tradeoff(held, "persona", what_if=arm)
        what_if = trade.what_if
        assert what_if is not None and what_if.arm == f"arm:{arm}"
        assert "never a recommendation" in what_if.label and f"arm:{arm}" in what_if.label
        assert what_if.gates == gate_reads(held, arm, "fail")
        admitted = set(surviving_options(held)) | {arm}
        assert {row.removeprefix("arm:") for row in what_if.frontier} == _swept_admitted(held, "persona", admitted)
        assert all(arm in (step.from_arm[4:], step.to_arm[4:]) for step in what_if.steps)
        assert what_if.against is not None and f"arm:{arm}" in (what_if.against.from_arm, what_if.against.to_arm)
        assert [row.arm for row in trade.frontier] == ["arm:f64", "arm:prod"], "the trade itself is unchanged"


def _swept_admitted(held: ReportInput, key: str, options: set[str]) -> set[str]:
    """The options best at some price among ``options``, swept as :func:`_swept_frontier` sweeps the survivors."""
    places = _positions(held, key)
    costs = {arm: float(resolve(f"cost_per_1k.{arm}", held).value) for arm in options if arm in places}  # type: ignore[arg-type]
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
    order = [held_arm.id for held_arm in held.arms]
    for price in prices:
        values = {arm: price * places[arm] - costs[arm] for arm in arms}
        top = max(values.values())
        winners = [arm for arm in arms if math.isclose(values[arm], top, rel_tol=1e-12, abs_tol=1e-12)]
        best.add(min(winners, key=order.index))
    return best


def test_a_what_if_naming_a_survivor_or_nothing_is_refused_naming_the_gated() -> None:
    for named in ("prod", "no such arm"):
        with pytest.raises(RefusedInput, match="a what-if prices an option the gates remove") as refused:
            price_tradeoff(_r4(), "persona", what_if=named)
        assert "'cor'" in (refused.value.remedy or "") or "'f16'" in (refused.value.remedy or "")


def test_the_wide_frontier_runs_past_one_result_whole_and_its_pages_hold_it_each_within_the_budget() -> None:
    """The control for the budget test over the wide frontier: read whole, its frontier and steps run past one
    result; each default page stays within it, and following the pages reads every frontier option and step once."""
    held = budget_inputs()[WIDE_FRONTIER]
    whole = price_tradeoff(held, "pass_rate", limit=len(held.arms))
    assert len(whole.frontier) >= 30 and len(whole.steps) == len(whole.frontier) - 1
    assert len(render_cited(whole)) > TEXT_BUDGET
    assert whole.frontier_options is not None and whole.frontier_options.value == len(whole.frontier)
    frontier: list[str] = []
    steps = 0
    offset: int | None = 0
    while offset is not None:
        page = price_tradeoff(held, "pass_rate", offset=offset)
        assert len(render_cited(page)) <= TEXT_BUDGET
        frontier += [row.arm for row in page.frontier]
        steps += len(page.steps)
        offset = page.next_offset
    assert frontier == [row.arm for row in whole.frontier] and steps == len(whole.steps)


# =============================================================================
# The prices block source: a step per row, drawn by bars and laid out by a table
# =============================================================================


def _prices_query(measures: list[str], arms: list[str] | None = None) -> ChartQuery:
    return ChartQuery.model_validate(
        {
            "source": "prices",
            "aspect": "",
            "measures": measures,
            "arms": arms or [],
            "strata": [],
            "limit": 0,
            "transform": "none",
            "small_multiples_by": "none",
        }
    )


@pytest.mark.parametrize("name", sorted(fixture_inputs()))
def test_each_price_row_is_a_frontier_step_at_the_price_its_key_reads(name: str) -> None:
    held = fixture_inputs()[name]
    monthly = held.meta.volume_per_month is not None
    for key in _axes(held):
        trade = price_tradeoff(held, key, limit=len(held.arms))
        rows = query_rows(_prices_query([key]), held)
        assert [row.label for row in rows] == [f"{s.from_arm[4:]}→{s.to_arm[4:]}" for s in trade.steps], key
        for row, step in zip(rows, trade.steps, strict=True):
            assert isinstance(row.item, PriceStep)
            price = step.monthly_per_point if monthly else step.per_point
            assert price is not None and (row.item.key, row.item.value) == (price.key, price.value)
            if row.interval is not None:
                assert row.interval == (_value(row.item.low_key, held), _value(row.item.high_key, held))


def test_the_steps_draw_as_bars_and_lay_out_as_a_table_named_by_their_options_with_their_interval() -> None:
    held = _r4()
    (step,) = price_tradeoff(held, "persona").steps
    for kind in ("bars", "table"):
        block = block_render(held, block_kind=kind, query=_prices_query(["persona"]))
        assert block.total == 1 and block.unit == "dollars", kind
        (cells,) = (row.cells for row in block.rows)
        names = {arm.id: arm.name for arm in fact_arms(held)}
        assert names["f64"] in cells[0] and names["prod"] in cells[0], cells
    bars = block_render(held, block_kind="bars", query=_prices_query(["persona"]))
    assert bars.level, "the step's paired interval is drawn, at its contrast's level"
    assert any("a month at the declared volume" in note for note in bars.notes)
    assert step.monthly_per_point is not None
    unpaired = block_render(
        fixture_inputs()["r1 reading only bars"], block_kind="bars", query=_prices_query(["overall"])
    )
    assert "unpaired" in unpaired.level and unpaired.unit == "usd_per_1k"


def test_a_prices_block_naming_no_gain_axis_or_two_is_refused_naming_the_axes() -> None:
    held = _r4()
    for measures in ([], ["persona", "facts"]):
        with pytest.raises(RefusedInput, match="reads one gain axis") as refused:
            block_render(held, block_kind="bars", query=_prices_query(measures))
        assert "query.measures" in refused.value.parameters
    catalog = {source.source: source for source in block_catalog(held, "s").sources}
    assert catalog["prices"].holds_rows and "bars" in BLOCK_SOURCES and "prices" in BLOCK_SOURCES["bars"]


def _monthly_steps(held: ReportInput) -> Iterator[tuple[str, list[Price]]]:
    """Each gain axis whose frontier takes a step, on an input that declares a monthly volume, with its steps."""
    if held.meta.volume_per_month is None:
        return
    for key in _axes(held):
        steps = price_tradeoff(held, key, limit=len(held.arms)).steps
        if steps:
            yield key, steps


def test_a_prices_tables_money_reads_with_its_thousands_grouped_and_a_months_to_the_whole_dollar() -> None:
    """A step's price a month reads as its key reads under ``usd_whole`` (``$1,239``, never ``$1239``) where every step
    on the axis is $10 or more, which whole dollars do not round away; under that, no price is rounded to the whole
    dollar: each keeps every digit the number rule spells its value with, so its cents wherever it holds any (a price
    of $1,000 or more the rule writes whole, and padding states no cent it was never rounded to). Either way no dollar
    amount prints four digits ungrouped, in the table or in the bars' values."""
    whole_seen = cents_seen = False
    for name, held in fixture_inputs().items():
        for key, steps in _monthly_steps(held):
            prices = [step.monthly_per_point for step in steps]
            assert all(price is not None for price in prices), (name, key)
            values = [float(price.value) for price in prices if price is not None]  # type: ignore[arg-type]
            table = block_render(held, block_kind="table", query=_prices_query([key]))
            bars = block_render(held, block_kind="bars", query=_prices_query([key]))
            for block in (table, bars):
                for row in block.rows:
                    assert not any(re.search(r"\$\d{4}", cell) for cell in row.cells), (name, key, row.cells)
            cells = [row.cells[1] for row in table.rows]
            if min(abs(value) for value in values) >= 10:
                whole = [format_figure(resolve(price.key, held), ("usd_whole",)) for price in prices if price]
                assert cells == whole, (name, key)
                whole_seen |= any(abs(value) >= 1000 for value in values)
            else:
                for cell, value in zip(cells, values, strict=True):
                    spelled = format_number(abs(value))
                    assert cell.lstrip("-").removeprefix("$").replace(",", "").startswith(spelled), (name, key, cell)
                    if abs(value) < WHOLE_FROM and value != int(value):
                        assert "." in cell, (name, key, cells)
                cents_seen = True
    assert whole_seen, "the control: some fixture prices a step a month at $1,000 or more"
    assert cents_seen, "the control: some fixture prices a step a month under $10"


def _stepless(held: ReportInput) -> Iterator[tuple[str, list[str]]]:
    """Each gain axis whose frontier takes no step, with the options on that frontier."""
    for key in _axes(held):
        trade = price_tradeoff(held, key, limit=len(held.arms))
        if not trade.steps:
            yield key, [row.arm.removeprefix("arm:") for row in trade.frontier]


@pytest.mark.parametrize("name", sorted(fixture_inputs()))
def test_a_prices_block_on_an_axis_whose_frontier_takes_no_step_is_refused_saying_so_and_sent_to_the_frontier(
    name: str,
) -> None:
    held = fixture_inputs()[name]
    if not _costed(held):
        return
    stepped = [key for key in _axes(held) if price_tradeoff(held, key, limit=len(held.arms)).steps]
    for key, frontier in _stepless(held):
        for kind in ("bars", "table"):
            with pytest.raises(RefusedInput) as refused:
                block_render(held, block_kind=kind, query=_prices_query([key]))
            said = str(refused.value)
            assert "selects nothing" not in said, said
            assert f"the price frontier on {key!r} holds" in said and "no step to price" in said, said
            assert all(f"arm:{arm}" in said for arm in frontier), said
            assert ("holds no option" in said) == (not frontier), said
            assert f"price_tradeoff with gain_key={key!r}" in refused.value.remedy, refused.value.remedy
            assert "query.measures" in refused.value.parameters
            assert list(refused.value.valid_values) == stepped


def _costed(held: ReportInput) -> bool:
    try:
        price_tradeoff(held, next(iter(_axes(held))), limit=1)
    except RefusedInput, StopIteration:
        return False
    return True


def test_the_reference_inputs_include_an_axis_whose_frontier_takes_no_step() -> None:
    """The control for the refusal above: r2 prices its pass rate on a frontier of one option."""
    assert "pass_rate" in dict(_stepless(fixture_inputs()["r2"]))


def test_a_prices_block_keeping_no_two_neighbours_of_its_frontier_is_refused_naming_the_frontier() -> None:
    held = fixture_inputs()["r1 reading only bars"]
    trade = price_tradeoff(held, "overall", limit=len(held.arms))
    assert len(trade.frontier) > 1, "the control: the frontier takes a step"
    first = trade.frontier[0].arm.removeprefix("arm:")
    with pytest.raises(RefusedInput) as refused:
        block_render(held, block_kind="table", query=_prices_query(["overall"], arms=[first]))
    said = str(refused.value)
    assert "selects nothing" not in said and "joins two of the arms it keeps" in said, said
    assert all(row.arm in said for row in trade.frontier), said
    assert "query.arms" in refused.value.parameters


def test_a_prices_chart_says_its_prices_are_per_1000_of_what_the_snapshots_cost_counts() -> None:
    """r1's cost counts conversations: a prices chart on it, with no monthly volume, says so, never replies."""
    held = fixture_inputs()["r1 reading only bars"]
    assert held.meta.volume_per_month is None, "the control: the steps are priced per 1,000"
    per = per_count(cost_basis(held).noun)
    assert "replies" not in per, "the control: the cost counts something else"
    bars = block_render(held, block_kind="bars", query=_prices_query(["overall"]))
    assert any(f"price per point {per}:" in note for note in bars.notes), bars.notes
    assert not any("replies" in note for note in bars.notes), bars.notes


def _reads(key: str, held: ReportInput) -> float | None:
    """The number a key reads, or None where it refuses (an end the interval leaves open, or one unstated)."""
    try:
        value = resolve(key, held).value
    except UnresolvableReference:
        return None
    return float(value) if isinstance(value, int | float) else None


def _drawn_ends(held: ReportInput, key: str, steps: list[Price]) -> list[tuple[float | None, float | None]]:
    """Each step's interval ends as a prices block draws them, read off its keys: the paired interval where every
    step states its low end, else the unpaired one; a high end None where it is unbounded (open above)."""
    field = "monthly_per_point" if held.meta.volume_per_month is not None else "per_point"
    moves = [(step.from_arm.removeprefix("arm:"), step.to_arm.removeprefix("arm:")) for step in steps]

    def ends(first: str, second: str, suffix: str) -> tuple[float | None, float | None]:
        prefix = f"price.{first}.{second}.{key}.{field}"
        return _reads(f"{prefix}_low{suffix}", held), _reads(f"{prefix}_high{suffix}", held)

    paired = all(ends(first, second, "")[0] is not None for first, second in moves)
    return [ends(first, second, "" if paired else "_unpaired") for first, second in moves]


def test_a_step_whose_gain_interval_reaches_zero_is_drawn_open_above_never_as_a_bare_price() -> None:
    """The Critic's W4: a step whose price is unbounded above was drawn with no interval at all, beside steps with
    one, so the least certain price read as the firmest. Every step whose low end is stated is drawn with it: both ends
    where its high end is bounded, else open above, in the values table and the tooltip's words and named in a note."""
    seen_open = seen_bounded_beside = 0
    for name, held in fixture_inputs().items():
        if not _costed(held):
            continue
        names = {arm.id: arm.name for arm in fact_arms(held)}
        for key in _axes(held):
            steps = price_tradeoff(held, key, limit=len(held.arms)).steps
            if not steps:
                continue
            drawn = _drawn_ends(held, key, steps)
            opened = [
                f"{names[s.from_arm[4:]]} → {names[s.to_arm[4:]]}"
                for s, (low, high) in zip(steps, drawn, strict=True)
                if low is not None and high is None
            ]
            bars = block_render(held, block_kind="bars", query=_prices_query([key]))
            table = block_render(held, block_kind="table", query=_prices_query([key]))
            for block in (bars, table):
                for row, (low, high) in zip(block.rows, drawn, strict=True):
                    text = " | ".join(row.cells)
                    if low is not None and high is None:
                        assert "no upper bound" in text, (name, key, block.block_kind, text)
                    else:
                        assert "no upper bound" not in text, (name, key, block.block_kind, text)
                    if low is not None and high is not None and block.block_kind == "bars":
                        assert " to " in text, (name, key, text)
                said = [note for note in block.notes if "No upper bound on" in note]
                if opened:
                    assert block.level, (name, key, block.block_kind, "an open interval is drawn at its level")
                    assert len(said) == 1 and all(step in said[0] for step in opened), (name, key, block.notes)
                else:
                    assert not said, (name, key, block.notes)
            seen_open += bool(opened)
            seen_bounded_beside += bool(opened) and any(high is not None for _, high in drawn)
    assert seen_open, "the control: some fixture's frontier takes a step whose gain's interval reaches zero"
    assert seen_bounded_beside, "the control: some axis draws an open step beside a bounded one"


def test_a_price_within_a_slice_is_per_what_the_slices_own_cost_measure_counts() -> None:
    """The Critic's N2: a price within a task reads the task's own cost measure, so its money is per what that measure
    counts, even where the page's cost is the scoreboard's per reply. r1 prices its tasks on a per-1k-conversations
    measure; given scoreboard reply costs, the page's cost basis turns to replies and the task's prices must not."""
    data = fixture_inputs()["r1 reading only bars"].model_dump(by_alias=True, exclude_none=True)
    for row in data["scoreboard"].values():
        row["reply_cost_usd"] = 0.001
    held = ReportInput.model_validate(data)
    assert cost_basis(held).noun == REPLIES, "the control: the page's cost is the scoreboard's, per reply"
    task = next(task for task in held.tasks if _task_costs(held, task.id))
    noun = per_noun(held.measures["cost_per_1k"])
    assert noun and noun != REPLIES, "the control: the task's cost measure counts something else"
    priced = 0
    for key in _axes(held):
        try:
            trade = price_tradeoff(held, key, slice_filter=task.id, limit=len(held.arms))
        except RefusedInput:
            continue
        for step in trade.steps:
            assert "own cost in this slice" in str(_value(step.cost_gap.key.replace(".cost_gap", ".cost_basis"), held))
            for figure in (step.cost_gap, step.per_point, step.per_point_low, step.per_point_high):
                if figure is None:
                    continue
                assert resolve(figure.key, held).per == noun, (key, figure.key)
                spelled = format_figure(resolve(figure.key, held), ("usd_per_1k",))
                assert per_count(noun) in spelled and REPLIES not in spelled, spelled
                priced += 1
    assert priced, "the control: some move within the task is priced"
