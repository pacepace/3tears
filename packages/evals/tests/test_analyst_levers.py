"""Lever effects: the lever figure keys, and the analyst's ``levers_effects`` over them.

Contracts held here (``threetears.evals.analysis.report.figure_keys``'s lever families,
``threetears.evals.analysis.analyst.levers``, ``ops.analyst_levers``, ``actions.analyst_levers``):

- **the pairs**: every effect a listing carries is between two arms whose settings, a setting an arm leaves out read
  as the control's, differ in the listed setting alone; every such pair holding a mean on the measure is listed once,
  overall and within a place;
- **the figures**: an effect is the second arm's mean minus the first's, read off the input's own figures; its interval
  is recovered from the two arms' own intervals (``ci``, or the mean give or take ``half_width``) by the formula the
  design note states, computed here; with either arm stating none there is no interval, and its key is unstated;
- **the interactions**: each pairs two effects of one move whose starting and ending arms differ in exactly one other
  setting (``across``); its difference is the second effect minus the first, its interval recovered from theirs; its
  verdict follows from the intervals (reverses, differs, within) or is ``no_interval``; a fixture holds a reversal;
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
from functools import cache
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
from threetears.evals.analysis.report_input import ReportInput, Stat
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


def _ends(stat: Stat) -> tuple[float, float, float] | None:
    """A figure's mean and the interval it states on it, or None for no interval."""
    assert stat.mean is not None
    mean = float(stat.mean)
    if stat.ci is not None:
        return mean, float(stat.ci[0]), float(stat.ci[1])
    if stat.half_width is not None:
        return mean, mean - float(stat.half_width), mean + float(stat.half_width)
    return None


def _recovered(one: tuple[float, float, float], two: tuple[float, float, float]) -> tuple[float, float]:
    """The design note's interval on ``two - one``: low ``d - sqrt((b - l_b)^2 + (u_a - a)^2)``, high
    ``d + sqrt((u_b - b)^2 + (a - l_a)^2)``."""
    (a, low_a, high_a), (b, low_b, high_b) = one, two
    d = b - a
    return d - math.sqrt((b - low_b) ** 2 + (high_a - a) ** 2), d + math.sqrt((high_b - b) ** 2 + (a - low_a) ** 2)


def _expected_effect(
    held: ReportInput, key: str, first: str, second: str, place: str | None
) -> tuple[float, tuple[float, float] | None]:
    one, two = _stat(held, key, first, place), _stat(held, key, second, place)
    assert one is not None and two is not None and one.mean is not None and two.mean is not None
    change = float(two.mean) - float(one.mean)
    ends = _ends(one), _ends(two)
    return change, (_recovered(ends[0], ends[1]) if ends[0] is not None and ends[1] is not None else None)


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
        change, interval = _expected_effect(held, key, first, second, place)
        assert row.figures[0].key.endswith(".effect") and row.figures[0].value == pytest.approx(change)
        if interval is None:
            assert len(row.figures) == 1
            with pytest.raises(UnstatedFigure):
                resolve(row.figures[0].key.removesuffix(".effect") + ".interval", held)
        else:
            assert row.figures[1].key.endswith(".interval")
            assert row.figures[1].value == pytest.approx(interval)
            stem = row.figures[0].key.removesuffix(".effect")
            assert (resolve(f"{stem}.low", held).value, resolve(f"{stem}.high", held).value) == pytest.approx(interval)


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
        one, two = _expected_effect(held, key, a1, b1, place), _expected_effect(held, key, a2, b2, place)
        assert row.figures[0].value == pytest.approx(two[0] - one[0])
        if one[1] is None or two[1] is None:
            assert row.verdict == "no_interval" and len(row.figures) == 1
            continue
        low, high = _recovered((one[0], *one[1]), (two[0], *two[1]))
        assert row.figures[1].value == pytest.approx((low, high))
        sides = [1 if lo > 0 else -1 if hi < 0 else 0 for lo, hi in (one[1], two[1])]
        if sides[0] * sides[1] < 0:
            assert row.verdict == "reverses"
        else:
            assert row.verdict == ("differs" if low > 0 or high < 0 else "within")


def test_a_fixture_holds_an_effect_that_reverses_across_another_setting() -> None:
    """The control for the verdicts: one prompt helps one model and hurts another, beyond their intervals."""
    verdicts = {row.verdict for name, key, place in READS for row in _all(INPUTS[name], key, place).interactions}
    assert {"reverses", "within", "no_interval"} <= verdicts


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
