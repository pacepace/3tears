"""The price of a trade: the price figure keys.

Contracts held here (``threetears.evals.analysis.report.figure_keys``'s price families):

- **the keys**: ``price.<from>.<to>.<gain>.<field>`` and ``cost_per_month.<arm>`` read, on r4 (the tradeoff
  reference) and on every fixture holding costs, the values computed here from the input's own fields: the gain from
  the contrasts against the control where each arm holds one, else from the arms' means, signed toward better; the
  cost gap from the arms' costs; the price per point and its ends off the gain's interval; the monthly figures at the
  declared volume. An interval reaching zero leaves the high end unbounded, refused so, never a number; a move one
  option dominates has no price; a monthly figure without a declared volume is unstated.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from packages.evals.tests.analyst_support import fixture_inputs, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.misfits import MISFITS, reference_data
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import figure_key_grammar, key_subjects, render, resolve
from threetears.evals.analysis.report_input import ReportInput

#: What a key's refusal says of a figure the input leaves unstated.
UNSTATED = "which the input does not state"


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
    assert render("{price.f64.prod.persona.monthly_cost_gap}", held, {}) == "+$7560.00"
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
