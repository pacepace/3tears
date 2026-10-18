"""What code reads off a campaign: each candidate against the control on every aspect, Holm across them, and floors.

One real run gives the aspect's shape; each test sets the per-case pass shares its aspects read (``varied``), so the
evidence behind every reading is exact. Twenty cases, each passed on every repeat or on none, so a share is 0 or 1.
On these, the bounded interval on six cases gained of twenty clears zero at 95% and not at 97.5%; on seven it clears
at 97.5% and not at 98.3%; on twelve, at every level the step-down reads. A drop mirrors it.
"""

from __future__ import annotations

import random

import pytest

from threetears.evals.analysis import AnalysisContextBundle, CampaignReadings, campaign_readings
from threetears.evals.analysis.report.decision import evaluate_floor, floor_readings, read_floor
from threetears.evals.analysis.stats import INTERVAL_LEVEL, SIGNIFICANCE_ALPHA, holm_adjust
from threetears.evals.kernel import RefusedInput
from threetears.evals.kernel.declaration import Floor, RateDrop

from packages.evals.tests.campaign_support import (
    CASES,
    MeasuredAspect,
    arms_of,
    campaign_of,
    measured_aspect,
    varied,
)

#: The run every test varies: both arms pass every case, so nothing but the varied shares separates them.
_BASE: MeasuredAspect | None = None


async def _base() -> MeasuredAspect:
    """The one real run, measured once for the module."""
    global _BASE
    if _BASE is None:
        _BASE = await measured_aspect("base", production=CASES, candidate=CASES)
    return _BASE


def _gained(cases: int) -> tuple[list[float], list[float]]:
    """The control failing every case, the candidate passing the first ``cases`` of them."""
    return [0.0] * CASES, [1.0] * cases + [0.0] * (CASES - cases)


def _dropped(cases: int) -> tuple[list[float], list[float]]:
    """The control passing every case, the candidate failing the first ``cases`` of them."""
    return [1.0] * CASES, [0.0] * cases + [1.0] * (CASES - cases)


def _read(
    base: MeasuredAspect,
    shapes: dict[str, tuple[list[float], list[float]]],
    *,
    margins: dict[str, float | None] | None = None,
    floors: tuple[tuple[str, float], ...] = (),
    lost: dict[str, int] | None = None,
) -> CampaignReadings:
    """The readings of a campaign of one aspect per ``shapes`` entry, each its shares, its margin and its floors."""
    margins = margins or {}
    campaign = campaign_of(
        base,
        [(label, margins.get(label, 10.0)) for label in shapes],
        floors=[(label, RateDrop(max_points=points)) for label, points in floors],
    )
    bundles = {
        label: varied(base.bundle, aspect=label, control=control, candidate=candidate, lost=(lost or {}).get(label, 0))
        for label, (control, candidate) in shapes.items()
    }
    return campaign_readings(campaign, bundles, arms=arms_of(base))


def _level(k: int) -> float:
    return 1.0 - SIGNIFICANCE_ALPHA / k


async def test_gains_step_down_across_three_aspects_each_at_its_own_holm_level() -> None:
    read = _read(await _base(), {"tone": _gained(12), "speed": _gained(7), "billing": _gained(6)})
    by = {row.aspect: row for row in read.readings}
    assert [by[label].reading for label in ("tone", "speed", "billing")] == ["better", "better", "better"]
    # Twelve clears at a third of α; seven only once tone has left the family; six only once both have.
    assert by["tone"].gain_confidence == pytest.approx(_level(3))
    assert by["speed"].gain_confidence == pytest.approx(_level(2))
    assert by["billing"].gain_confidence == pytest.approx(_level(1))
    gains = [by[label].gain_p for label in ("tone", "speed", "billing")]
    assert all(p is not None for p in gains)
    assert [by[label].gain_p_holm for label in ("tone", "speed", "billing")] == pytest.approx(
        holm_adjust([p for p in gains if p is not None])
    )
    assert all((row.gain_p_holm or 1.0) < SIGNIFICANCE_ALPHA for row in read.readings)


async def test_a_gain_alone_shown_is_unclear_once_the_family_must_correct_for_it() -> None:
    base = await _base()
    alone = _read(base, {"billing": _gained(6)})
    assert alone.readings[0].reading == "better"
    together = _read(base, {"tone": _gained(12), "speed": _gained(6), "billing": _gained(6)})
    by = {row.aspect: row for row in together.readings}
    assert by["tone"].reading == "better"
    # With tone shown, the step reads the other two at 97.5%, where six of twenty does not clear, and stops there.
    assert by["speed"].reading == by["billing"].reading == "unclear"
    assert by["billing"].gain_confidence == pytest.approx(_level(2))
    assert by["billing"].gain_interval is not None and by["billing"].gain_interval[0] <= 0


async def test_drops_step_down_on_their_own_family_against_each_margin() -> None:
    margins: dict[str, float | None] = {"tone": 0.5, "speed": 0.5, "billing": 0.5}
    read = _read(
        await _base(),
        {"tone": _dropped(12), "speed": _dropped(7), "billing": _dropped(6)},
        margins=margins,
    )
    by = {row.aspect: row for row in read.readings}
    assert [by[label].reading for label in ("tone", "speed", "billing")] == ["worse", "worse", "worse"]
    assert by["tone"].drop_confidence == pytest.approx(_level(3))
    assert by["speed"].drop_confidence == pytest.approx(_level(2))
    assert by["billing"].drop_confidence == pytest.approx(_level(1))
    drops = [by[label].drop_p for label in ("tone", "speed", "billing")]
    assert [by[label].drop_p_holm for label in ("tone", "speed", "billing")] == pytest.approx(
        holm_adjust([p for p in drops if p is not None])
    )
    # The gains are a family of their own: none of them is shown, and their levels say so.
    assert all(row.gain_confidence == pytest.approx(_level(3)) for row in read.readings)


async def test_gains_and_drops_are_corrected_apart() -> None:
    read = _read(await _base(), {"tone": _gained(6), "billing": _dropped(6)}, margins={"tone": 0.5, "billing": 0.5})
    by = {row.aspect: row for row in read.readings}
    # Each direction's family holds two rows, so each is read at 97.5% first, where six of twenty clears neither;
    # one family of four would read them at 98.75%.
    assert by["tone"].gain_confidence == pytest.approx(_level(2))
    assert by["billing"].drop_confidence == pytest.approx(_level(2))
    assert by["tone"].reading == by["billing"].reading == "unclear"


async def test_an_aspect_with_no_margin_is_never_shown_worse_and_tests_no_drop() -> None:
    read = _read(await _base(), {"billing": _dropped(15)}, margins={"billing": None})
    (row,) = read.readings
    assert row.reading == "unclear"
    assert row.margin is None and row.drop_interval is None and row.drop_p is None and row.drop_p_holm is None
    assert row.projection is None
    assert "no margin was declared" in row.reason
    assert row.gain_interval is not None and row.gain_interval[1] < 0


async def test_floors_are_read_uncorrected_while_the_readings_are_corrected() -> None:
    read = _read(
        await _base(),
        {"tone": _dropped(12), "speed": _dropped(6), "billing": _dropped(6)},
        margins={"tone": 0.5, "speed": 0.5, "billing": 0.5},
        floors=(("billing", 0.5),),
    )
    by = {row.aspect: row for row in read.readings}
    assert by["billing"].reading == "unclear"
    (status,) = read.floors
    assert status.aspect == "billing" and status.status == "broken" and status.floor == "billing/rate_drop/*"
    assert read.broken_floors() == [status]
    assert status.interval is not None and status.interval[1] < -0.5
    # The reading's interval is wider: read at the family's level, it reaches back within the margin.
    assert by["billing"].drop_interval is not None and by["billing"].drop_interval[1] > -0.5


def _shares(rng: random.Random, cases: int, shift: float) -> tuple[list[float], list[float]]:
    """Random per-case shares in thirds, the candidate's moved by ``shift`` thirds on a random share of cases."""
    thirds = [0.0, 1 / 3, 2 / 3, 1.0]
    control = [rng.choice(thirds) for _ in range(cases)]
    candidate = [min(1.0, max(0.0, value + shift / 3)) if rng.random() < 0.6 else value for value in control]
    return control, candidate


async def test_a_one_aspect_floor_and_its_reading_on_the_same_rate_and_margin_never_disagree() -> None:
    base = await _base()
    rng = random.Random(846)
    seen = set()
    for _ in range(24):
        margin = rng.choice([2.0, 5.0, 10.0, 20.0])
        control, candidate = _shares(rng, CASES, rng.choice([-3, -2, -1, 0, 1, 2]))
        lost = rng.choice([0, 0, 0, 2, 6])
        campaign = campaign_of(base, [("billing", margin)], floors=[("billing", RateDrop(max_points=margin))])
        bundle = varied(base.bundle, aspect="billing", control=control, candidate=candidate, lost=lost)
        read = campaign_readings(campaign, {"billing": bundle}, arms=arms_of(base))
        (row,) = read.readings
        (status,) = read.floors
        # One row: one step, at the floor's own level.
        assert row.drop_confidence == INTERVAL_LEVEL and row.gain_confidence == INTERVAL_LEVEL
        readings = floor_readings(bundle, base.control, base.candidate)
        verdict = read_floor(
            Floor(aspect="billing", clause=RateDrop(max_points=margin)), readings, control="control", candidate="v2"
        )
        pass_rate = next(rate for rate in verdict.rates if rate.rate.measure is None)
        assert (pass_rate.outcome == "broken") == (row.reading == "worse")
        if row.reading == "worse":
            assert status.status == "broken"
            assert row.drop_interval is not None and status.interval == pytest.approx(row.drop_interval)
        seen.add(row.reading)
    assert seen == {"better", "worse", "unclear"}


async def test_a_reading_and_its_interval_never_contradict() -> None:
    base = await _base()
    rng = random.Random(3)
    for _ in range(12):
        shapes = {
            label: _shares(rng, CASES, rng.choice([-3, -1, 0, 1, 3]))
            for label in ("tone", "speed", "billing")[: rng.choice([1, 2, 3])]
        }
        margins = {label: rng.choice([None, 2.0, 10.0]) for label in shapes}
        read = _read(base, shapes, margins=margins)
        for row in read.readings:
            if row.reading == "better":
                assert row.gain_interval is not None and row.gain_interval[0] > 0
            if row.reading == "worse":
                assert row.margin is not None and row.drop_interval is not None
                assert row.drop_interval[1] < -row.margin
            if row.reading == "unclear":
                assert row.gain_interval is None or row.gain_interval[0] <= 0 or "lost" in row.reason
                assert row.drop_interval is None or row.drop_interval[1] >= -(row.margin or 0) or "lost" in row.reason
            # The projection is the unclear row's alone, and only where a margin bounds the drop.
            assert (row.projection is not None) <= (row.reading == "unclear" and row.margin is not None)


async def test_only_an_unclear_row_with_a_margin_carries_the_cases_that_would_settle_it() -> None:
    read = _read(
        await _base(),
        {"tone": _gained(12), "speed": _dropped(12), "billing": _gained(2)},
        margins={"tone": 10.0, "speed": 10.0, "billing": 10.0},
    )
    by = {row.aspect: row for row in read.readings}
    assert (by["tone"].reading, by["speed"].reading, by["billing"].reading) == ("better", "worse", "unclear")
    assert by["tone"].projection is None and by["speed"].projection is None
    projection = by["billing"].projection
    assert projection is not None and projection.cases == CASES
    assert projection.if_none.kind in ("more", "enough", "beyond")


async def test_a_row_counts_and_names_both_arms() -> None:
    base = await _base()
    read = _read(base, {"billing": _dropped(12)})
    (row,) = read.readings
    assert (row.arm, row.arm_label) == (base.candidate, "v2")
    assert base.campaign.control is not None and read.control == base.campaign.control.variant_key
    assert (row.control_passed, row.control_trials, row.arm_passed, row.arm_trials) == (60, 60, 24, 60)
    assert row.change == pytest.approx(-60.0)
    assert (row.control_cases, row.arm_cases) == (CASES, CASES)
    assert row.confirmatory
    assert read.aspects == ["billing"] and read.alpha == SIGNIFICANCE_ALPHA


async def test_a_campaign_with_no_control_reads_no_row_and_no_floor() -> None:
    base = await _base()
    campaign = campaign_of(base, [("billing", 10.0)], floors=[("billing", RateDrop(max_points=10))])
    campaign = campaign.model_copy(update={"arms": []})
    read = campaign_readings(
        campaign, {"billing": varied(base.bundle, aspect="billing", control=[1.0] * CASES, candidate=[0.0] * CASES)}
    )
    assert read.control is None and read.control_label == "" and read.readings == [] and read.floors == []


async def test_a_bundle_for_each_aspect_is_required_and_each_reads_its_own() -> None:
    base = await _base()
    campaign = campaign_of(base, [("billing", 10.0), ("tone", 10.0)])
    billing = varied(base.bundle, aspect="billing", control=[1.0] * CASES, candidate=[1.0] * CASES)
    with pytest.raises(RefusedInput, match="no evidence bundle for aspect"):
        campaign_readings(campaign, {"billing": billing})
    with pytest.raises(RefusedInput, match="reads aspect billing"):
        campaign_readings(campaign, {"billing": billing, "tone": billing})
    with pytest.raises(RefusedInput, match="holds no aspect labelled speed"):
        campaign_readings(
            campaign, {"billing": billing, "tone": billing.model_copy(update={"aspect": "tone"}), "speed": billing}
        )


def _floors_match(read: CampaignReadings, bundle: AnalysisContextBundle, base: MeasuredAspect) -> None:
    """Every floor status is evaluate_floor's own, at its nominal level."""
    readings = floor_readings(bundle, base.control, base.candidate)
    for status in read.floors:
        floor = Floor(aspect=status.aspect, clause=RateDrop(max_points=10))
        assert status == evaluate_floor(floor, readings, arm=base.candidate, arm_label="v2", control_label="control")


async def test_a_floor_status_is_evaluate_floor_s_own() -> None:
    base = await _base()
    bundle = varied(base.bundle, aspect="billing", control=[1.0] * CASES, candidate=[0.0] * 10 + [1.0] * 10)
    campaign = campaign_of(base, [("billing", 10.0)], floors=[("billing", RateDrop(max_points=10))])
    read = campaign_readings(campaign, {"billing": bundle}, arms=arms_of(base))
    _floors_match(read, bundle, base)
