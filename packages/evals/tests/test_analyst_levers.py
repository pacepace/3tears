"""Lever effects: the lever figure keys, and the analyst's ``levers_effects`` over them.

Contracts held here (``threetears.evals.analysis.report.figure_keys``'s lever families,
``threetears.evals.analysis.analyst.levers``, ``ops.analyst_levers``, ``actions.analyst_levers``):

- **the pairs**: every effect a listing carries is between two arms whose settings, a setting an arm leaves out read
  as the control's, differ in the listed setting alone; every such pair holding a mean on the measure is listed once,
  overall and within a place;
- **the figures**: an effect, its interval, level and basis are what ``compare.<to>.<from>.<measure>.<where>`` reads
  for the same two arms (one spread rule for both; compare's own keys are held to the input in
  ``test_analyst_separation.py``), so an interval exists wherever compare's does; on r2, whose pass rates state only a
  count, every effect's interval is the binomial one computed here from the input; with no spread there is no
  interval, and its key is unstated;
- **the interactions**: each pairs two effects of one move whose starting and ending arms differ in exactly one other
  setting (``across``); its difference is the second effect minus the first, its interval recovered from theirs; its
  verdict follows from the intervals (reverses, differs, within) or is ``no_interval``; a fixture holds a reversal;
  the four arms such a pair spans are read twice, each setting's move across the other: one reading is listed where
  the two verdicts agree (the first setting's by id), both where they differ, however the arms are ordered (a move
  listed the other way round is read in the first's direction); a planted reversal one reading shows and the other
  does not is listed beside it in either order;
- **the order**: effects, then interactions; within each, the rows with an interval first;
- **the refusals**: arms varying no setting, arms none of which differ in one setting alone (naming what a pair also
  differs in), a setting, place or measure the snapshot does not hold, and a measure no lever pair holds overall, each
  refused naming the call that answers instead;
- **the tool**: it is on the analyst's and the editor's cuts, in the analyses workflow; every fixture call
  ``ANALYST_CALLS`` makes answers, within the budget, every number a figure whose key reads it
  (``tests/test_analyst_contract.py``); its pages read every row once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from statistics import NormalDist
from typing import Any

import pytest

from packages.evals.tests.analyst_support import budget_inputs, fixture_inputs, fixture_snapshots
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from threetears.evals.actions import ActionOutcome, MountedTool, analyst_tools, eval_catalogue
from threetears.evals.actions.analyst_analyses import ANALYSES_WORKFLOW
from threetears.evals.analysis.analyst.levers import LeverEffects, lever_effects
from threetears.evals.analysis.errors import UnstatedFigure
from threetears.evals.analysis.report.figure_keys import figure_key_grammar, resolve
from threetears.evals.analysis.report_input import Measure, ReportInput, Stat
from threetears.evals.kernel.errors import RefusedInput

#: Every input the walk reads: the fixtures and the shape and wide campaigns.
INPUTS: dict[str, ReportInput] = {**fixture_inputs(), **budget_inputs()}


def _settings(held: ReportInput) -> dict[str, dict[str, Any]]:
    """Each arm's settings, a setting it leaves out read as the control's (none with no control)."""
    control = next((arm for arm in held.arms if arm.role == "control"), None)
    base = dict(control.settings) if control is not None else {}
    names = list(dict.fromkeys(name for arm in held.arms for name in arm.settings))
    return {arm.id: {name: arm.settings.get(name, base.get(name)) for name in names} for arm in held.arms}


def _differ(settings: Mapping[str, Mapping[str, Any]], first: str, second: str) -> list[str]:
    return [name for name, value in settings[first].items() if settings[second][name] != value]


def _stat(held: ReportInput, key: str, arm: str, place: str | None) -> Stat | None:
    if place is None:
        table = held.measures.get(key) or held.judged.get(key)
        return table.per_arm.get(arm) if table is not None else None
    held_at = [
        *(task.per_arm for task in held.tasks if task.id == place),
        *(stratum.per_arm for stratum in held.strata if stratum.id == place),
        *(cell.per_arm for cell in held.cells.values() if f"{cell.row}.{cell.column}" == place),
    ]
    return held_at[0].get(arm, {}).get(key) if len(held_at) == 1 else None


def _recovered(one: tuple[float, float, float], two: tuple[float, float, float]) -> tuple[float, float]:
    """The design note's interval on ``two - one``: low ``d - sqrt((b - l_b)^2 + (u_a - a)^2)``, high
    ``d + sqrt((u_b - b)^2 + (a - l_a)^2)``."""
    (a, low_a, high_a), (b, low_b, high_b) = one, two
    d = b - a
    return d - math.sqrt((b - low_b) ** 2 + (high_a - a) ** 2), d + math.sqrt((high_b - b) ** 2 + (a - low_a) ** 2)


def _where(held: ReportInput, place: str | None) -> str:
    """A lever place as a separation key's ``<where>`` spells it."""
    if place is None:
        return "overall"
    if any(task.id == place for task in held.tasks):
        return f"task.{place}"
    if any(stratum.id == place for stratum in held.strata):
        return f"strata.{place}"
    return f"cells.{place}"


@dataclass(frozen=True)
class _Expected:
    """What the compare family reads for a move: its difference, interval, level and basis (None where unstated)."""

    change: float
    interval: tuple[float, float] | None
    level: object
    basis: object


def _expected_effect(held: ReportInput, key: str, first: str, second: str, place: str | None, scale: str) -> _Expected:
    """The second arm minus the first and its interval, as the compare family reads them, in the figure's own units
    on ``scale`` (compare reads a share-scale rate in points)."""
    one, two = _stat(held, key, first, place), _stat(held, key, second, place)
    assert one is not None and two is not None and one.mean is not None and two.mean is not None
    prefix = f"compare.{second}.{first}.{key}.{_where(held, place)}"
    factor = 100.0 if scale == "share" else 1.0
    difference = resolve(f"{prefix}.difference", held).value
    assert isinstance(difference, float | int)
    try:
        interval = resolve(f"{prefix}.interval", held).value
    except UnstatedFigure:
        return _Expected(float(difference) / factor, None, None, None)
    assert isinstance(interval, tuple)
    try:
        level = resolve(f"{prefix}.level", held).value
    except UnstatedFigure:
        level = None
    ends = (interval[0] / factor, interval[1] / factor)
    return _Expected(float(difference) / factor, ends, level, resolve(f"{prefix}.basis", held).value)


def _all(held: ReportInput, key: str, place: str | None) -> LeverEffects:
    return _whole(next(name for name, value in INPUTS.items() if value is held), key, place)


@cache
def _whole(name: str, key: str, place: str | None) -> LeverEffects:
    """Every effect and interaction of one read, on one page: read once, held by every walk."""
    return lever_effects(INPUTS[name], key, place_filter=place, limit=10_000)


def _reads() -> list[tuple[str, str, str | None]]:
    """For each input but the wide ones, overall and in its first two places, the first two measures whose lever
    effects read there."""
    reads: list[tuple[str, str, str | None]] = []
    for name, held in INPUTS.items():
        if name.startswith("wide"):
            continue
        places = [
            *(task.id for task in held.tasks),
            *(stratum.id for stratum in held.strata),
            *(f"{cell.row}.{cell.column}" for cell in held.cells.values()),
        ]
        for place in (None, *places[:2]):
            found = 0
            for key in dict.fromkeys([*held.measures, *held.judged]):
                if found == 2:
                    break
                try:
                    lever_effects(held, key, place_filter=place, limit=1)
                except RefusedInput:
                    continue
                reads.append((name, key, place))
                found += 1
    return reads


READS = _reads()


def test_the_walk_reads_effects_overall_and_within_a_place_on_several_inputs() -> None:
    """The control for the walks below."""
    assert {name for name, _key, _place in READS} >= {"r2", "r3", "r4", "r5", "s1"}
    assert any(place is not None for _name, _key, place in READS)


@pytest.mark.parametrize(("name", "key", "place"), READS)
def test_each_lever_pair_is_listed_once_its_effect_the_second_mean_minus_the_first(
    name: str, key: str, place: str | None
) -> None:
    held = INPUTS[name]
    settings = _settings(held)
    read = _all(held, key, place)
    listed = [(row.setting, str(row.from_arm.value), str(row.to_arm.value)) for row in read.effects]
    expected = {
        (differ[0], *sorted((first, second)))
        for first in settings
        for second in settings
        if first < second
        and len(differ := _differ(settings, first, second)) == 1
        and (one := _stat(held, key, first, place)) is not None
        and one.mean is not None
        and (two := _stat(held, key, second, place)) is not None
        and two.mean is not None
    }
    assert sorted((setting, *sorted((a, b))) for setting, a, b in listed) == sorted(expected)
    assert len(listed) == len(expected), "each pair once"
    for row, (setting, first, second) in zip(read.effects, listed, strict=True):
        assert _differ(settings, first, second) == [setting]
        expected = _expected_effect(held, key, first, second, place, row.figures[0].scale)
        assert row.figures[0].key.endswith(".effect") and row.figures[0].value == pytest.approx(expected.change)
        if expected.interval is None:
            assert len(row.figures) == 1
            with pytest.raises(UnstatedFigure):
                resolve(row.figures[0].key.removesuffix(".effect") + ".interval", held)
        else:
            assert row.figures[1].key.endswith(".interval")
            assert row.figures[1].value == pytest.approx(expected.interval)
            stem = row.figures[0].key.removesuffix(".effect")
            ends = (resolve(f"{stem}.low", held).value, resolve(f"{stem}.high", held).value)
            assert ends == pytest.approx(expected.interval)
            beside = {figure.key.rsplit(".", 1)[1]: figure.value for figure in row.figures[2:]}
            assert beside == {"basis": expected.basis} | ({} if expected.level is None else {"level": expected.level})


@pytest.mark.parametrize(("name", "key", "place"), READS)
def test_each_interaction_is_one_move_in_contexts_one_setting_apart_and_its_verdict_follows(
    name: str, key: str, place: str | None
) -> None:
    held = INPUTS[name]
    settings = _settings(held)
    read = _all(held, key, place)
    for row in read.interactions:
        arms = []
        for figure in (row.first, row.second):
            stem = figure.key.removeprefix(f"lever.{row.setting}.").removesuffix(".effect")
            match = next(
                (first, second)
                for first in settings
                for second in settings
                if stem == f"{first}.{second}.{key}" + (f".{place}" if place is not None else "")
            )
            arms.append(match)
        (a1, b1), (a2, b2) = arms
        assert _differ(settings, a1, a2) == [row.across] == _differ(settings, b1, b2)
        assert row.across != row.setting
        assert settings[a1][row.setting] == settings[a2][row.setting]
        assert settings[b1][row.setting] == settings[b2][row.setting]
        scale = row.first.scale
        one, two = _expected_effect(held, key, a1, b1, place, scale), _expected_effect(held, key, a2, b2, place, scale)
        assert row.figures[0].value == pytest.approx(two.change - one.change)
        if one.interval is None or two.interval is None or one.level != two.level:
            assert row.verdict == "no_interval" and len(row.figures) == 1
            continue
        low, high = _recovered((one.change, *one.interval), (two.change, *two.interval))
        assert row.figures[1].value == pytest.approx((low, high))
        sides = [1 if lo > 0 else -1 if hi < 0 else 0 for lo, hi in (one.interval, two.interval)]
        if sides[0] * sides[1] < 0:
            assert row.verdict == "reverses"
        else:
            assert row.verdict == ("differs" if low > 0 or high < 0 else "within")


def test_a_fixture_holds_an_effect_that_reverses_across_another_setting() -> None:
    """The control for the verdicts: one prompt helps one model and hurts another, beyond their intervals."""
    verdicts = {row.verdict for name, key, place in READS for row in _all(INPUTS[name], key, place).interactions}
    assert {"reverses", "within", "no_interval"} <= verdicts


def test_a_pass_rate_with_a_count_gives_every_effect_the_binomial_interval_compare_gives() -> None:
    """r2's pass rates state a count and no interval. Every effect between two of its arms carries the interval the two
    rates' binomial spreads give, combined as independent, as arms_compare reads them: computed here from the input."""
    held = INPUTS["r2"]
    read = _all(held, "pass_rate", None)
    measure = held.measures["pass_rate"]
    z = NormalDist().inv_cdf(0.975)
    assert read.effects
    for row in read.effects:
        first, second = str(row.from_arm.value), str(row.to_arm.value)
        assert len(row.figures) > 1, f"{row.figures[0].key}: no interval"
        means, errors = [], []
        for arm in (first, second):
            mean, count = measure.per_arm[arm].mean, held.scoreboard[arm].usable or held.scoreboard[arm].cases
            assert mean is not None and count is not None
            means.append(float(mean))
            errors.append(math.sqrt(mean / 100 * (1 - mean / 100) / count) * 100)
        change = means[1] - means[0]
        half = z * math.hypot(*errors)
        assert row.figures[1].value == pytest.approx((change - half, change + half))


def _lead_contrast(held: ReportInput, arm: str, key: str) -> tuple[float, tuple[float, float]] | None:
    """The arm's contrast against the control on the measure, read on the lead aspect (the first asked, else the
    first; a contrast naming no aspect is read on it): its difference and interval, where it states both."""
    asked = [aspect.id for aspect in held.aspects if aspect.asked]
    lead = asked[0] if asked else next((aspect.id for aspect in held.aspects), None)
    found = [
        c
        for c in held.contrasts
        if c.arm == arm and c.measure == key and c.aspect in (None, lead) and c.delta is not None and c.interval
    ]
    if len(found) != 1:
        return None
    (contrast,) = found
    assert contrast.delta is not None and contrast.interval is not None
    return float(contrast.delta), (float(contrast.interval[0]), float(contrast.interval[1]))


def test_an_effect_between_the_control_and_an_arm_it_contrasts_is_that_contrasts_paired_difference() -> None:
    """Where a lever pair is the control and an arm the input contrasts against it overall (on the lead aspect), the
    two ran the same cases: the effect is the contrast's difference and its interval the contrast's own, read the
    move's way round (negated and mirrored where the move ends at the control), and its basis says paired."""
    seen: set[str] = set()
    for name, held in INPUTS.items():
        if name.startswith("wide"):
            continue
        control = next((arm.id for arm in held.arms if arm.role == "control"), None)
        for key in dict.fromkeys([*held.measures, *held.judged]):
            try:
                read = _all(held, key, None)
            except RefusedInput:
                continue
            for row in read.effects:
                first, second = str(row.from_arm.value), str(row.to_arm.value)
                if control not in (first, second):
                    continue
                arm, sign = (second, 1.0) if first == control else (first, -1.0)
                contrast = _lead_contrast(held, arm, key)
                if contrast is None:
                    continue
                delta, (low, high) = contrast
                ends = (low, high) if sign > 0 else (-high, -low)
                figures = {figure.key.rsplit(".", 1)[1]: figure.value for figure in row.figures}
                assert figures["effect"] == pytest.approx(sign * delta), (name, key, arm)
                assert figures["interval"] == pytest.approx(ends), (name, key, arm)
                basis = figures["basis"]
                assert isinstance(basis, str) and basis.startswith("paired"), (name, key, arm, basis)
                seen.add(name)
    assert seen, "the control: some lever pair runs through the control to an arm it contrasts"


def _moves(held: ReportInput, key: str, place: str | None) -> dict[str, list[frozenset[str]]]:
    """Every lever pair holding a mean on the measure, by setting, unordered."""
    settings = _settings(held)
    found: dict[str, list[frozenset[str]]] = {}
    for first in settings:
        for second in settings:
            differ = _differ(settings, first, second)
            if first < second and len(differ) == 1:
                one, two = _stat(held, key, first, place), _stat(held, key, second, place)
                if one is not None and one.mean is not None and two is not None and two.mean is not None:
                    found.setdefault(differ[0], []).append(frozenset((first, second)))
    return found


def _expected_crossings(held: ReportInput, key: str, place: str | None) -> set[frozenset[frozenset[str]]]:
    """Every two moves of one setting between the same two values whose contexts differ in exactly one other setting,
    unordered: computed from the arms' settings, whatever order the input lists the arms in."""
    settings = _settings(held)
    found: set[frozenset[frozenset[str]]] = set()
    for setting, moves in _moves(held, key, place).items():
        for one in moves:
            for two in moves:
                if one == two:
                    continue
                values = {repr(settings[arm][setting]) for arm in one}
                if values != {repr(settings[arm][setting]) for arm in two}:
                    continue
                matched = [(a, b) for a in one for b in two if repr(settings[a][setting]) == repr(settings[b][setting])]
                differs = [_differ(settings, a, b) for a, b in matched]
                if len(differs) == 2 and differs[0] == differs[1] and len(differs[0]) == 1:
                    found.add(frozenset((one, two)))
    return found


def _listed_crossings(read: LeverEffects, key: str, place: str | None) -> list[frozenset[frozenset[str]]]:
    tail = f".{key}" + (f".{place}" if place is not None else "") + ".effect"
    listed = []
    for row in read.interactions:
        moves = []
        for figure in (row.first, row.second):
            stem = figure.key.removeprefix(f"lever.{row.setting}.").removesuffix(tail)
            moves.append(next(frozenset(pair) for pair in _arm_pairs(read, stem)))
        listed.append(frozenset(moves))
    return listed


def _arm_pairs(read: LeverEffects, stem: str) -> list[tuple[str, str]]:
    """The arm pair a stem ``<from>.<to>`` names, by the arms the listing's effects name."""
    arms = {str(row.from_arm.value) for row in read.effects} | {str(row.to_arm.value) for row in read.effects}
    return [(a, b) for a in arms for b in arms if f"{a}.{b}" == stem]


def _reordered(held: ReportInput) -> ReportInput:
    """An input with the arms of its last model listed in reverse, so a move on a setting of three or more values runs
    one way in that model's context and the other way in the rest."""
    last = held.arms[-1].settings.get("model")
    reversed_arms = [arm for arm in held.arms if arm.settings.get("model") == last][::-1]
    others = [arm for arm in held.arms if arm.settings.get("model") != last]
    return held.model_copy(update={"arms": [*others, *reversed_arms]})


def _setting_of(held: ReportInput, crossing: frozenset[frozenset[str]]) -> str:
    """The setting a crossing's two moves move."""
    move = next(iter(crossing))
    (setting,) = _differ(_settings(held), *sorted(move))
    return setting


def _expected_verdict(held: ReportInput, key: str, crossing: frozenset[frozenset[str]], scale: str) -> str:
    """A crossing's verdict, from the compare family's two effects, each move read from its setting's lower value."""
    settings = _settings(held)
    setting = _setting_of(held, crossing)
    first, second = (
        _expected_effect(held, key, *sorted(move, key=lambda arm: repr(settings[arm][setting])), None, scale)
        for move in sorted(crossing, key=sorted)
    )
    if first.interval is None or second.interval is None or first.level != second.level:
        return "no_interval"
    sides = [1 if lo > 0 else -1 if hi < 0 else 0 for lo, hi in (first.interval, second.interval)]
    if sides[0] * sides[1] < 0:
        return "reverses"
    low, high = _recovered((first.change, *first.interval), (second.change, *second.interval))
    return "differs" if low > 0 or high < 0 else "within"


def _expected_listing(held: ReportInput, key: str, scale: str) -> set[frozenset[frozenset[str]]]:
    """The crossings listed: of the two readings of each four arms, the first setting's by id where their verdicts
    agree, both where they do not."""
    squares: dict[frozenset[str], list[frozenset[frozenset[str]]]] = {}
    for crossing in _expected_crossings(held, key, None):
        squares.setdefault(frozenset(arm for move in crossing for arm in move), []).append(crossing)
    kept: set[frozenset[frozenset[str]]] = set()
    for readings in squares.values():
        if len({_expected_verdict(held, key, crossing, scale) for crossing in readings}) > 1:
            kept.update(readings)
        else:
            kept.add(min(readings, key=lambda crossing: _setting_of(held, crossing)))
    return kept


@pytest.mark.parametrize("name", ["r2", "r2 reordered"])
def test_four_arms_are_listed_once_where_their_two_readings_agree_however_the_arms_are_ordered(name: str) -> None:
    """Every two moves of one setting one context apart are an interaction, and the four arms they span are read
    twice, each setting's move across the other: one difference of differences. Where the two verdicts agree, one
    reading is listed, the first setting's by id, whatever the arms' order (the misfit review of r2 printed one
    interaction twice); where they differ, both are."""
    held = _reordered(INPUTS["r2"]) if name == "r2 reordered" else INPUTS[name]
    key = "pass_rate"
    read = lever_effects(held, key, limit=10_000)
    listed = _listed_crossings(read, key, None)
    assert _expected_crossings(held, key, None), "the control: the input holds moves one context apart"
    assert len(listed) == len(set(listed)), "each once"
    assert set(listed) == _expected_listing(held, key, read.effects[0].figures[0].scale)


def _square_planted(*, reversed_order: bool) -> ReportInput:
    """r1's four arms on a two-by-two of settings ``a`` and ``b``, a score of 0 at a1 b1, 10 at a2 b1, 20 at a1 b2 and
    15 at a2 b2, each with an interval one either side: ``a``'s effect goes +10 then -5 across ``b`` (it reverses),
    ``b``'s goes +20 then +5 across ``a`` (it differs). The arms listed as given or reversed."""
    held = INPUTS["r1"]
    cells = [("1", "1", 0.0), ("2", "1", 10.0), ("1", "2", 20.0), ("2", "2", 15.0)]
    arms = [
        arm.model_copy(update={"settings": {"model": "M", "a": a, "b": b}})
        for arm, (a, b, _) in zip(held.arms, cells, strict=True)
    ]
    score = Measure(
        label="Score",
        unit="",
        level="95%",
        per_arm={
            arm.id: Stat(mean=mean, ci=(mean - 1, mean + 1)) for arm, (_, _, mean) in zip(arms, cells, strict=True)
        },
    )
    listed = arms[::-1] if reversed_order else arms
    return held.model_copy(update={"arms": listed, "measures": {**held.measures, "score": score}})


@pytest.mark.parametrize("reversed_order", [False, True], ids=["as listed", "reversed"])
def test_a_reversal_one_reading_shows_and_the_other_does_not_is_listed_beside_it_in_either_arm_order(
    reversed_order: bool,
) -> None:
    read = lever_effects(_square_planted(reversed_order=reversed_order), "score", limit=100)
    assert sorted((row.setting, row.across, row.verdict) for row in read.interactions) == [
        ("a", "b", "reverses"),
        ("b", "a", "differs"),
    ]


def _one_arm_uncounted(held: ReportInput) -> ReportInput:
    """r2 with its first arm's scoreboard gone: that arm's pass rate rests on no count, so its effects have no
    interval while the rest do."""
    board = {arm: row for arm, row in held.scoreboard.items() if arm != held.arms[0].id}
    return held.model_copy(update={"scoreboard": board})


def test_rows_with_an_interval_lead_effects_and_interactions_alike() -> None:
    held = _one_arm_uncounted(INPUTS["r2"])
    read = lever_effects(held, "pass_rate", limit=10_000)
    for rows in (read.effects, read.interactions):
        has = [len(row.figures) > 1 for row in rows]
        assert True in has and False in has, "the control: some rows have an interval and some do not"
        assert has == sorted(has, reverse=True), "every row with an interval before every row without"
    for name, key, place in READS:
        for rows in (_all(INPUTS[name], key, place).effects, _all(INPUTS[name], key, place).interactions):
            has = [len(row.figures) > 1 for row in rows]
            assert has == sorted(has, reverse=True), (name, key, place)


def test_every_setting_the_arms_vary_is_listed_with_its_values_on_every_page() -> None:
    held = INPUTS["r2"]
    settings = _settings(held)
    varying = [
        name
        for name in dict.fromkeys(n for values in settings.values() for n in values)
        if len({repr(values[name]) for values in settings.values()}) > 1
    ]
    first = lever_effects(held, "pass_rate", limit=1)
    later = lever_effects(held, "pass_rate", limit=1, offset=first.total - 1)
    for read in (first, later):
        assert [row.setting for row in read.settings] == varying
        for row in read.settings:
            assert [figure.value for figure in row.values] == list(
                dict.fromkeys(
                    float(values[row.setting]) if not isinstance(values[row.setting], str) else values[row.setting]
                    for values in settings.values()
                )
            )


def test_pages_read_every_effect_then_every_interaction_once() -> None:
    held = INPUTS["r2"]
    whole = _all(held, "pass_rate", None)
    effects: list[str] = []
    interactions: list[str] = []
    offset: int | None = 0
    while offset is not None:
        page = lever_effects(held, "pass_rate", limit=7, offset=offset)
        effects += [row.figures[0].key for row in page.effects]
        interactions += [row.figures[0].key for row in page.interactions]
        offset = page.next_offset
    assert effects == [row.figures[0].key for row in whole.effects]
    assert interactions == [row.figures[0].key for row in whole.interactions]
    assert whole.total == len(effects) + len(interactions) and whole.next_offset is None


def _tangled() -> ReportInput:
    """r4 with each candidate given a reasoning level of its own: every two arms differ in two settings or more."""
    held = INPUTS["r4"]
    arms = [
        arm if arm.role == "control" else arm.model_copy(update={"settings": {**arm.settings, "reasoning": arm.id}})
        for arm in held.arms
    ]
    return held.model_copy(update={"arms": arms})


def test_arms_none_of_which_differ_in_one_setting_alone_are_refused_naming_what_a_pair_also_differs_in() -> None:
    held = _tangled()
    settings = _settings(held)
    assert all(len(_differ(settings, a, b)) > 1 for a in settings for b in settings if a != b), "the control"
    key = next(iter(held.measures))
    with pytest.raises(RefusedInput, match="no two arms differ in any one setting") as refused:
        lever_effects(held, key)
    assert "differ in another setting too" in refused.value.fault
    assert "measure_read" in (refused.value.remedy or "") and "campaign_overview" in (refused.value.remedy or "")


@pytest.mark.parametrize(
    ("name", "arguments", "fault", "remedy"),
    [
        ("s3", {"measure_key": "cost_usd"}, "the arms vary no setting", "campaign_overview"),
        ("r2", {"measure_key": "no-such-measure"}, "names no measure or judged dimension", "campaign_overview"),
        ("r2", {"measure_key": "pass_rate", "setting_filter": "nope"}, "names no setting the arms vary", "omit"),
        ("r2", {"measure_key": "pass_rate", "place_filter": "nope"}, "names no task, stratum or cell", "omit"),
        ("r3", {"measure_key": "accuracy"}, "both hold a mean on 'accuracy' overall", "place_filter"),
    ],
    ids=["no setting varies", "unknown measure", "unknown setting", "unknown place", "held only per cell"],
)
def test_a_read_the_snapshot_cannot_answer_is_refused_naming_what_to_read_instead(
    name: str, arguments: dict[str, Any], fault: str, remedy: str
) -> None:
    measure = arguments.pop("measure_key")
    with pytest.raises(RefusedInput) as refused:
        lever_effects(INPUTS[name], measure, **arguments)
    assert fault in refused.value.fault and remedy in (refused.value.remedy or "")


def test_the_grammar_teaches_the_lever_keys_on_a_page_of_their_own() -> None:
    grammar = figure_key_grammar()
    for form in (
        "lever.<setting>.<from>.<to>.<measure>.<field>",
        "lever.<setting>.<from>.<to>.<measure>.<place>.<field>",
        "lever.<setting>.<from>.<to>.<measure>.vs.<from>.<to>.<field>",
        "lever.<setting>.<from>.<to>.<measure>.<place>.vs.<from>.<to>.<field>",
    ):
        assert f"- `{form}`" in grammar, form


# --- the tool ------------------------------------------------------------------------------------------------------


def _tool(role: str = "analyst") -> MountedTool:
    (spec,) = analyst_tools(role=role)  # type: ignore[arg-type]
    return eval_catalogue().mount(spec)


def _seeded() -> tuple[OpsFixture, dict[str, Any]]:
    fixture = ops_fixture()
    snapshots = fixture_snapshots(TOYHOST_SCOPE)
    for snapshot in snapshots.values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture, snapshots


async def _call(fixture: OpsFixture, arguments: dict[str, Any]) -> ActionOutcome:
    return await _tool().call(arguments, host=fixture.host, caller=CALLER)


def test_levers_effects_is_an_analysis_on_the_analysts_and_the_editors_cuts() -> None:
    for role in ("analyst", "editor"):
        action = _tool(role).action("levers_effects")
        assert action is not None and action.workflow == ANALYSES_WORKFLOW and action.permission == "read", role


async def test_a_call_reads_its_snapshot_and_a_refusal_names_its_remedy() -> None:
    fixture, snapshots = off_loop(_seeded)
    r2 = snapshots["r2"].id
    outcome = await _call(fixture, {"action": "levers_effects", "snapshot_id": r2, "measure_key": "pass_rate"})
    assert outcome.ending == "ok" and outcome.structured is not None, outcome.text
    assert outcome.text.startswith("lever effects\n") and outcome.structured["settings"]
    refused = await _call(
        fixture, {"action": "levers_effects", "snapshot_id": snapshots["r3"].id, "measure_key": "accuracy"}
    )
    assert refused.ending == "refused" and refused.refusal is not None, refused.text
    assert "place_filter" in refused.text and "measure_read" in refused.text
