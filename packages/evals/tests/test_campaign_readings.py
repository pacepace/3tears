"""What code reads off a campaign: each candidate against the control on every aspect, Holm across its aspects, and
floors.

One real run gives the aspect's shape; each test sets the per-case pass shares its aspects read (``varied``), so the
evidence behind every reading is exact. Twenty cases, each passed on every repeat or on none, so a share is 0 or 1.
On these, the bounded interval on six cases gained of twenty clears zero at 95% and not at 97.5%; on seven it clears
at 97.5% and not at 98.3%; on twelve, at every level the step-down reads. A drop mirrors it.
"""

from __future__ import annotations

import random
from typing import Literal

import pytest

from threetears.evals.analysis import AnalysisContextBundle, CampaignReadings, campaign_kind, campaign_readings
from threetears.evals.analysis.report.decision import evaluate_floor, floor_readings, read_floor
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    holm_adjust,
)
from threetears.evals.kernel import RefusedInput
from threetears.evals.kernel.declaration import BarOverride, CampaignArm, Floor, RateDrop

from packages.evals.tests.campaign_support import (
    CASES,
    MeasuredAspect,
    MeasuredKind,
    joined,
    measured_kind,
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
    # Each p is on its family's grid, so a row shown has an adjusted p of at most α, to rounding.
    assert all((row.gain_p_holm or 1.0) <= SIGNIFICANCE_ALPHA * (1 + 1e-9) for row in read.readings)


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


#: The run the two-candidate tests vary: a second candidate, v3, beside v2.
_BASE3: MeasuredAspect | None = None


async def _base3() -> MeasuredAspect:
    """The one real run with two candidates, measured once for the module."""
    global _BASE3
    if _BASE3 is None:
        _BASE3 = await measured_aspect("base3", production=CASES, candidate=CASES, third=CASES)
    return _BASE3


async def test_a_one_aspect_floor_and_each_candidate_s_reading_on_the_same_rate_and_margin_never_disagree() -> None:
    base = await _base3()
    assert base.third is not None
    rng = random.Random(846)
    seen = set()
    for index in range(14):
        margin = rng.choice([5.0, 10.0, 20.0])
        # Below the case floor one time in four and lost trials one time in three. The first two shapes are one
        # candidate shown worse beside one shown better, paired and then unpaired, so every reading is met both ways
        # (an unpaired row left unclear projects slowly, so the random ones are paired).
        cases = 4 if index % 4 == 3 else CASES
        paired = index != 1
        control, v2 = _shares(rng, cases, (-3, -2, -1, 0, 1, 2, 3)[index % 7])
        _, v3 = _shares(rng, cases, (3, 1, -1, -3, 2, 0, -2)[index % 7])
        if index < 2:
            control, v2, v3 = [1.0] * 10 + [0.0] * 10, [0.0] * 20, [1.0] * 20
        lost = rng.choice([0, 0, 2, 6]) if cases == CASES and index > 1 else 0
        campaign = campaign_of(base, [("billing", margin)], floors=[("billing", RateDrop(max_points=margin))])
        bundle = varied(
            base.bundle,
            aspect="billing",
            control=control,
            candidate=v2,
            also={base.third: v3},
            lost=lost,
            paired=paired,
        )
        read = campaign_readings(campaign, {"billing": bundle}, arms=arms_of(base))
        assert len(read.readings) == 2 and len(read.floors) == 2
        for row, status in zip(read.readings, read.floors, strict=True):
            assert row.arm == status.arm
            # One aspect: each candidate's family is one row, one step, at the floor's own level.
            assert row.drop_confidence == INTERVAL_LEVEL and row.gain_confidence == INTERVAL_LEVEL
            readings = floor_readings(bundle, base.control, row.arm)
            verdict = read_floor(
                Floor(aspect="billing", clause=RateDrop(max_points=margin)),
                readings,
                control="production",
                candidate=row.arm_label,
            )
            pass_rate = next(rate for rate in verdict.rates if rate.rate.measure is None)
            assert (pass_rate.outcome == "broken") == (row.reading == "worse")
            if row.reading == "worse":
                assert status.status == "broken"
                assert row.drop_interval is not None and status.interval == pytest.approx(row.drop_interval)
            seen.add(row.reading)
    assert seen == {"better", "worse", "unclear"}


async def test_two_candidates_do_not_share_a_family() -> None:
    base = await _base3()
    assert base.third is not None
    control, gained = _gained(6)
    for labels in (("billing",), ("billing", "tone")):
        campaign = campaign_of(base, [(label, 10.0) for label in labels])
        bundles = {
            label: varied(base.bundle, aspect=label, control=control, candidate=gained, also={base.third: gained})
            for label in labels
        }
        read = campaign_readings(campaign, bundles, arms=arms_of(base))
        # Each candidate's family holds its own aspects alone: one aspect reads both at 95%, where six of twenty
        # clears; one family over both candidates would read them at 97.5%, where it does not.
        rows = [row for row in read.readings if row.aspect == "billing"]
        assert {row.arm_label for row in rows} == {"v2", "v3"}
        if labels == ("billing",):
            assert [row.reading for row in rows] == ["better", "better"]
            assert all(row.gain_confidence == pytest.approx(_level(1)) for row in rows)
        else:
            assert all(row.gain_confidence == pytest.approx(_level(2)) for row in rows)
            assert all(row.reading == "unclear" for row in read.readings)


async def test_a_holm_adjusted_p_is_at_most_alpha_exactly_where_the_row_is_shown() -> None:
    base = await _base()
    rng = random.Random(851)
    for index in range(10):
        cases = 4 if index % 3 == 2 else CASES
        shapes = {label: _shares(rng, cases, rng.choice([-3, -2, 0, 2, 3])) for label in ("tone", "speed", "billing")}
        margins = {label: rng.choice([None, 10.0, 20.0]) for label in shapes}
        read = _read(base, shapes, margins=margins, lost={"speed": rng.choice([0, 9])})
        for row in read.readings:
            within = SIGNIFICANCE_ALPHA * (1 + 1e-9)
            assert ((row.gain_p_holm or 1.0) <= within) == (row.reading == "better")
            assert ((row.drop_p_holm or 1.0) <= within) == (row.reading == "worse")


async def test_a_gain_its_lost_trials_could_take_back_has_no_p_below_one() -> None:
    # The interval on the per-case shares clears zero, but with 45 of the candidate's 60 trials lost and each given its
    # worst outcome, the candidate could still trail: the gate fails, so the row is unclear and its p is 1.
    base = await _base()
    control = [1 / 3] * CASES
    candidate = [1.0] * 12 + [1 / 3] * 8
    read = _read(base, {"billing": (control, candidate)}, lost={"billing": 45})
    (row,) = read.readings
    assert row.reading == "unclear" and "lost trials" in row.reason
    assert row.gain_interval is not None and row.gain_interval[0] > 0
    assert row.gain_p == 1.0 and row.gain_p_holm == 1.0


def test_the_bounded_interval_nests_over_the_step_down_s_levels() -> None:
    # The step-down never relies on it (a row counts as shown only where every looser level shows it too), but on
    # realistic shares it holds, so that rule changes nothing a reader would see.
    rng = random.Random(7)
    for _ in range(60):
        cases = rng.choice([5, 8, 12, 20, 30])
        paired = rng.random() < 0.7
        control, candidate = _shares(rng, cases, rng.choice([-3, -1, 0, 1, 3]))
        if not paired:
            candidate = [rng.choice([0.0, 1 / 3, 2 / 3, 1.0]) for _ in range(cases + rng.choice([0, 3]))]
        wider = None
        for k in range(1, 6):
            interval = bounded_difference_interval(
                control, candidate, paired=paired, value_range=(0.0, 1.0), confidence=_level(k)
            )
            assert interval is not None
            if wider is not None:
                assert interval[0] <= wider[0] + 1e-12 and interval[1] >= wider[1] - 1e-12
            wider = interval


async def test_a_reading_and_its_interval_never_contradict() -> None:
    base = await _base()
    rng = random.Random(3)
    for _ in range(12):
        shapes = {
            label: _shares(rng, CASES, rng.choice([-3, -1, 0, 1, 3]))
            for label in ("tone", "speed", "billing")[: rng.choice([1, 2, 3])]
        }
        margins = {label: rng.choice([None, 10.0, 20.0]) for label in shapes}
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
        assert status == evaluate_floor(floor, readings, arm=base.candidate, arm_label="v2", control_label="production")


async def test_a_floor_status_is_evaluate_floor_s_own() -> None:
    base = await _base()
    bundle = varied(base.bundle, aspect="billing", control=[1.0] * CASES, candidate=[0.0] * 10 + [1.0] * 10)
    campaign = campaign_of(base, [("billing", 10.0)], floors=[("billing", RateDrop(max_points=10))])
    read = campaign_readings(campaign, {"billing": bundle}, arms=arms_of(base))
    _floors_match(read, bundle, base)


def test_a_campaign_s_kind_is_derived_from_its_arms() -> None:
    def arm(label: str, role: Literal["control", "candidate"], digit: str) -> CampaignArm:
        return CampaignArm(label=label, role=role, variant_key=digit * 64)

    assert campaign_kind([]) == "measurement"
    assert campaign_kind([arm("v1", "candidate", "a")]) == "measurement"
    assert campaign_kind([arm("v1", "control", "a")]) == "measurement"
    # The same variant named twice is one arm.
    assert campaign_kind([arm("v1", "candidate", "a"), arm("v1 again", "candidate", "a")]) == "measurement"
    assert campaign_kind([arm("v1", "candidate", "a"), arm("v2", "candidate", "b")]) == "choice"
    assert campaign_kind([arm("v1", "control", "a"), arm("v2", "candidate", "b")]) == "change"


# =============================================================================
# A choice and a measurement
# =============================================================================

_KINDS: dict[str, MeasuredKind] = {}

#: The bar every kind's aspects are held to: no reply may cost more than a tenth of a cent. Each arm costs 0.2 cents
#: a reply unless its name says otherwise, so it misses this bar.
_COST_BAR = BarOverride(measure_id="cost_usd", threshold=0.001, direction="lower_is_better")


async def _kind(label: str) -> MeasuredKind:
    """Each fixture aspect, measured once for the module: on tone alpha passes nineteen cases to beta's six; on
    billing ten to nine; solo is one arm alone. ``cheap`` arms cost a twentieth of a cent a reply."""
    if label not in _KINDS:
        shapes: dict[str, tuple[dict[str, int], bool]] = {
            "tone": ({"alpha": 19, "beta": 6}, False),
            "billing": ({"alpha": 10, "beta": 9}, False),
            "solo": ({"alpha": 15}, True),
        }
        arms, control = shapes[label]
        _KINDS[label] = await measured_kind(label, arms, control=control, bars=[_COST_BAR], costs={"beta": 0.0005})
    return _KINDS[label]


async def test_a_choice_shows_an_arm_ahead_on_one_aspect_and_no_separation_on_another() -> None:
    tone, billing = await _kind("tone"), await _kind("billing")
    campaign = joined([tone, billing])
    read = campaign_readings(campaign, {"tone": tone.bundle, "billing": billing.bundle}, arms=tone.arms())
    assert read.kind == "choice" and read.readings == [] and read.floors == [] and read.control is None
    by = {pair.aspect: pair for pair in read.pairs}
    assert (by["tone"].first_label, by["tone"].second_label) == ("alpha", "beta")
    assert by["tone"].ahead == "alpha" and by["tone"].change == pytest.approx(65.0)
    assert by["tone"].interval is not None and by["tone"].interval[0] > 0
    assert by["billing"].ahead is None and "no separation" in by["billing"].reason
    assert by["billing"].interval is not None and by["billing"].interval[0] <= 0 <= by["billing"].interval[1]
    # One direction-free family over both aspects' pairs: tone is shown at 97.5%, billing then read at 95%.
    assert by["tone"].confidence == pytest.approx(_level(2))
    assert by["billing"].confidence == pytest.approx(_level(1))
    assert (by["tone"].p_holm or 1.0) <= SIGNIFICANCE_ALPHA * (1 + 1e-9) < (by["billing"].p_holm or 1.0)


async def test_a_choice_with_nothing_separated_has_no_arm_ahead() -> None:
    billing = await _kind("billing")
    read = campaign_readings(billing.campaign, {"billing": billing.bundle}, arms=billing.arms())
    (pair,) = read.pairs
    assert read.kind == "choice" and pair.ahead is None


async def test_every_kind_reads_each_arm_against_its_bars_as_the_bundle_adjudicated_them() -> None:
    tone, solo = await _kind("tone"), await _kind("solo")
    choice = campaign_readings(tone.campaign, {"tone": tone.bundle}, arms=tone.arms())
    assert [(bar.arm_label, bar.status) for bar in choice.bars] == [("alpha", "missed"), ("beta", "met")]
    assert [bar.arm_label for bar in choice.missed_bars()] == ["alpha"]
    measurement = campaign_readings(solo.campaign, {"solo": solo.bundle}, arms=solo.arms())
    assert measurement.kind == "measurement" and measurement.pairs == [] and measurement.readings == []
    (bar,) = measurement.bars
    assert (bar.status, bar.bar, bar.direction, bar.measure) == ("missed", 0.001, "lower_is_better", "cost_usd")
    assert bar.reason.startswith("alpha misses the bar on")
    measures = {reading.measure: reading for reading in measurement.measures}
    assert measures["passed"].value == pytest.approx(0.75) and measures["passed"].cases == CASES
    assert all(reading.arm_label == "alpha" and reading.aspect == "solo" for reading in measurement.measures)
    assert all(reading.interval is not None for reading in measurement.measures if reading.measure != "passed")
