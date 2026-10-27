"""Two arms compared with an interval, and the cases that would settle a reading: ``arms_compare`` and ``settle_cases``.

Contracts held here (``threetears.evals.analysis.separation``, the ``spread.``, ``compare.`` and ``settle.`` families of
``analysis.report.figure_keys``, ``analysis.analyst.compare``, ``analysis.analyst.settle``, and their operations and
actions):

- **an arm's spread** is read from its statistic in order (a stated interval or half-width at the stated level, its
  standard error, a pass/fail rate's binomial spread from its count), each checked against the statistic itself;
- **two arms compared**: the difference is the first mean minus the second (in points for a rate); its interval is a
  contrast's own where one states one arm against the other overall (paired), else centred on the difference with a
  half-width of the level's quantile times the root of the two standard errors' squares (independent), and ``basis``
  says which; ``excludes_zero`` and ``ahead`` follow from the interval and the measure's direction; a measure with no
  spread is refused, naming ``measure_read``;
- **the cases that settle a reading**: an interval at the level centred on today's estimate clears the line (one-sided)
  or zero (two-sided, each arm) at ``cases_needed`` and not one case fewer, and four in five such designs clear at
  ``cases_needed_80``; ``wrong_side`` is how often a run of today's size reads the other way; the W2 r2 editor's sum
  (the pick's pass rate against its gate) is reproduced from the input's own figures; a hypothetical line is keyed
  hypothetical; a p95, a figure without a spread, and an estimate on its line say why, and a call where nothing can
  be settled is refused;
- **every fixture snapshot**: both actions answer or refuse with a remedy over the tool, and every figure resolves.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from functools import cache
from itertools import combinations
from statistics import NormalDist
from typing import Any

import pytest

from packages.evals.tests.analyst_support import aspect_tagged, compared_pair, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.misfits import reference_data
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from threetears.evals.actions import ActionOutcome, MountedTool, analyst_tools, eval_catalogue
from threetears.evals.analysis.analyst.compare import arms_compare, mean_key
from threetears.evals.analysis.analyst.settle import SettleRow, settle_cases
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnstatedFigure
from threetears.evals.analysis.report.figure_keys import (
    figure_key_grammar,
    figure_key_pages,
    gain_axis,
    resolve,
    separation_wheres,
)
from threetears.evals.analysis.report_decision import dominant_options
from threetears.evals.analysis.report_input import ReportInput, Stat
from threetears.evals.analysis.separation import (
    NoSpread,
    cases_to_separate,
    level_share,
    stat_spread,
    wrong_side_share,
)
from threetears.evals.kernel import EvalReportSnapshot
from threetears.evals.kernel.errors import RefusedInput

#: Every page a test reads whole asks for this many rows.
WHOLE = 200

#: The fields every cited figure carries, by which one is found in a result's JSON.
FIGURE_FIELDS = frozenset({"key", "reads", "value", "kind", "scale", "unit"})

_NORMAL = NormalDist()


@cache
def _all_inputs() -> dict[str, ReportInput]:
    """Every fixture snapshot's input, by name: r1 to r5, the misfits, and Kairo round 2."""
    return {name: frozen_input(snapshot) for name, snapshot in fixture_snapshots("s").items()}


def _number(value: object) -> float:
    assert isinstance(value, int | float), value
    return float(value)


def _figures(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if FIGURE_FIELDS <= node.keys():
            yield node
            return
        for item in node.values():
            yield from _figures(item)
    elif isinstance(node, list):
        for item in node:
            yield from _figures(item)


def _each_figure_resolves(structured: dict[str, Any], held: ReportInput) -> None:
    for figure in _figures(structured):
        value = resolve(figure["key"], held).value
        assert figure["value"] == (list(value) if isinstance(value, tuple) else value), figure["key"]


def _stat(held: ReportInput, arm: str, key: str, where: str) -> Stat:
    """An arm's statistic at a separation key's place, read off the input directly."""
    if where == "overall":
        table = held.measures.get(key) or held.judged[key]
        return table.per_arm[arm]
    kind, _dot, rest = where.partition(".")
    if kind == "task":
        return next(task for task in held.tasks if task.id == rest).per_arm[arm][key]
    if kind == "strata":
        return next(stratum for stratum in held.strata if stratum.id == rest).per_arm[arm][key]
    return next(cell for cell in held.cells.values() if f"{cell.row}.{cell.column}" == rest).per_arm[arm][key]


def _points(held: ReportInput, key: str) -> float:
    """What a difference on a measure is multiplied by to read in points: 100 for a rate on a share scale, as its own
    means read."""
    table = held.measures.get(key) or held.judged[key]
    arm = next(iter(table.per_arm), None)
    if arm is None:
        return 1.0
    return 100.0 if resolve(mean_key(held, arm, key, "overall"), held).scale == "share" else 1.0


# =============================================================================
# The arithmetic
# =============================================================================


def test_a_level_reads_as_a_share_from_either_spelling() -> None:
    assert level_share("95%") == level_share(0.95) == 0.95
    assert level_share("99.69%") == pytest.approx(0.9969)
    assert level_share(None) is None and level_share("most") is None and level_share("100%") is None


def test_a_spread_is_read_from_the_first_of_interval_half_width_sem_and_a_rates_count() -> None:
    z = _NORMAL.inv_cdf(0.975)
    assert stat_spread(0.5, ci=(0.4, 0.6), sem=1.0, stated_level=0.95).se == pytest.approx(0.1 / z)
    assert stat_spread(50.0, half_width=4.0, sem=1.0, stated_level=0.95).basis == "half_width"
    assert stat_spread(50.0, half_width=4.0, sem=1.0).basis == "sem", "a half-width at no stated level is not read"
    rate = stat_spread(90.0, n=400, rate="percent")
    assert rate.basis == "binomial" and rate.level == 0.95
    assert rate.se == pytest.approx(math.sqrt(0.9 * 0.1 / 400) * 100)
    assert rate.per_case == pytest.approx(math.sqrt(0.9 * 0.1) * 100)
    with pytest.raises(NoSpread, match="no interval, half-width or standard error"):
        stat_spread(3.0)
    with pytest.raises(NoSpread, match="no count"):
        stat_spread(0.9, rate="share")
    with pytest.raises(NoSpread, match="no stated level"):
        stat_spread(0.9, ci=(0.8, 1.0))


@pytest.mark.parametrize(("sided", "level"), [(1, 0.95), (2, 0.95), (1, 0.9969), (2, 0.8)])
def test_cases_needed_is_the_least_size_an_interval_centred_on_the_estimate_clears_at(sided: int, level: float) -> None:
    per_case, distance = 16.8, 0.7
    z = _NORMAL.inv_cdf(level if sided == 1 else 0.5 + level / 2)
    needed = cases_to_separate(per_case, distance, level, sided=1 if sided == 1 else 2, power=0.5)
    assert distance - z * per_case / math.sqrt(needed) >= 0
    assert distance - z * per_case / math.sqrt(needed - 1) < 0
    powered = cases_to_separate(per_case, distance, level, sided=1 if sided == 1 else 2, power=0.8)
    # At that size a fresh estimate clears with probability at least four in five, if the truth is today's estimate.
    assert _NORMAL.cdf(distance * math.sqrt(powered) / per_case - z) >= 0.8 - 1e-9
    assert _NORMAL.cdf(distance * math.sqrt(powered - 1) / per_case - z) < 0.8
    # At that size a run of it reads the other way at most as often as the level's tail, and barely less.
    assert _NORMAL.cdf(-z) * 0.99 < wrong_side_share(distance, per_case / math.sqrt(needed)) <= _NORMAL.cdf(-z)
    with pytest.raises(NoSpread, match="exactly on its line"):
        cases_to_separate(per_case, 0.0, level, sided=1, power=0.5)
    with pytest.raises(NoSpread, match="reads alike"):
        cases_to_separate(0.0, distance, level, sided=1, power=0.5)


# =============================================================================
# The keys
# =============================================================================


def _pairs(held: ReportInput) -> list[tuple[str, str]]:
    arms = [arm.id for arm in held.arms][:4]
    return list(combinations(arms, 2))


def _spread_se(held: ReportInput, arm: str, key: str, where: str) -> float | None:
    """An arm's standard error at a place, from its spread keys, checked against the statistic it was read from."""
    try:
        basis = resolve(f"spread.{arm}.{key}.{where}.basis", held).value
    except UnstatedFigure:
        return None
    stat = _stat(held, arm, key, where)
    se = _number(resolve(f"spread.{arm}.{key}.{where}.se", held).value)
    level = _number(resolve(f"spread.{arm}.{key}.{where}.level", held).value)
    points = _points(held, key)
    z = _NORMAL.inv_cdf(0.5 + level / 2)
    if basis == "its stated interval":
        assert stat.ci is not None and se == pytest.approx((stat.ci[1] - stat.ci[0]) / 2 / z * points)
    elif basis == "its stated half-width":
        assert stat.half_width is not None and se == pytest.approx(stat.half_width / z * points)
    elif basis == "its standard error":
        assert stat.sem is not None and se == pytest.approx(stat.sem * points)
    else:
        # A binomial spread rests on the distinct cases where the count is trials over fewer stated cases (each case
        # counted once: the widest its repeats allow), else on the count.
        try:
            n = _number(resolve(f"spread.{arm}.{key}.{where}.cases", held).value)
        except UnstatedFigure:
            n = _number(resolve(f"spread.{arm}.{key}.{where}.n", held).value)
        assert stat.mean is not None
        share = stat.mean / 100 if points == 1.0 else stat.mean
        assert se == pytest.approx(math.sqrt(share * (1 - share) / n) * 100)
    return se


@pytest.mark.parametrize("name", sorted(_all_inputs()))
def test_every_comparison_key_is_the_difference_of_the_means_and_its_interval_the_spreads_or_a_contrasts(
    name: str,
) -> None:
    held = _all_inputs()[name]
    checked = 0
    for first, second in _pairs(held):
        for key in dict.fromkeys([*held.measures, *held.judged]):
            for where in separation_wheres(held, (first, second), key):
                prefix = f"compare.{first}.{second}.{key}.{where}"
                difference = _number(resolve(f"{prefix}.difference", held).value)
                try:
                    interval = resolve(f"{prefix}.interval", held).value
                except UnstatedFigure:
                    continue
                assert isinstance(interval, tuple)
                basis = resolve(f"{prefix}.basis", held).value
                assert isinstance(basis, str)
                if basis.startswith("paired"):
                    (contrast,) = [
                        c for c in held.contrasts if {c.arm} < {first, second} and c.measure == key and c.interval
                    ]
                    sign = 1.0 if contrast.arm == first else -1.0
                    assert contrast.delta is not None
                    assert difference == pytest.approx(sign * contrast.delta * _points(held, key))
                    continue
                points = _points(held, key)
                mine, theirs = _stat(held, first, key, where), _stat(held, second, key, where)
                assert mine.mean is not None and theirs.mean is not None
                assert difference == pytest.approx((mine.mean - theirs.mean) * points)
                se_first, se_second = _spread_se(held, first, key, where), _spread_se(held, second, key, where)
                assert se_first is not None and se_second is not None
                level = _number(resolve(f"{prefix}.level", held).value)
                half = _NORMAL.inv_cdf(0.5 + level / 2) * math.hypot(se_first, se_second)
                assert interval == pytest.approx((difference - half, difference + half))
                checked += 1
    if name in ("r1", "r3", "kairo round 2"):
        assert checked, f"the control: {name} states spreads to compare on"


def test_a_comparison_with_the_control_on_a_contrasted_measure_reads_the_contrast_paired() -> None:
    held = _all_inputs()["r4"]
    contrast = next(contrast for contrast in held.contrasts if contrast.interval is not None)
    control = next(arm.id for arm in held.arms if arm.role == "control")
    forward = resolve(f"compare.{contrast.arm}.{control}.{contrast.measure}.overall.interval", held).value
    backward = resolve(f"compare.{control}.{contrast.arm}.{contrast.measure}.overall.interval", held).value
    assert contrast.interval is not None and isinstance(forward, tuple) and isinstance(backward, tuple)
    points = _points(held, contrast.measure)
    assert forward == pytest.approx(tuple(points * end for end in contrast.interval))
    assert backward == pytest.approx((-forward[1], -forward[0]))
    basis = resolve(f"compare.{contrast.arm}.{control}.{contrast.measure}.overall.basis", held).value
    assert isinstance(basis, str) and basis.startswith("paired")


def test_a_contrast_on_another_aspect_is_never_read_as_the_overall_paired_difference() -> None:
    """``overall`` reads an arm's by-arm figures, the lead aspect's. Where the lead aspect's contrast of the arm is
    untested and another aspect holds a contrast on the same measure key, that contrast is another aspect's reading:
    the comparison and the gain axis read the two arms' own means, never it, and nothing is labelled paired."""
    held, arm, key = aspect_tagged(_all_inputs()["r4"])
    control = next(held_arm.id for held_arm in held.arms if held_arm.role == "control")
    (lead_untested,) = [c for c in held.contrasts if (c.arm, c.measure) == (arm, key) and c.delta is None]
    (elsewhere,) = [c for c in held.contrasts if (c.arm, c.measure) == (arm, key) and c.delta is not None]
    assert lead_untested.aspect != elsewhere.aspect, "the control: the two contrasts sit on two aspects"
    points = _points(held, key)
    mine, theirs = _stat(held, arm, key, "overall").mean, _stat(held, control, key, "overall").mean
    assert mine is not None and theirs is not None and elsewhere.delta is not None
    assert mine - theirs != pytest.approx(elsewhere.delta), "the control: the other aspect's delta reads apart"
    prefix = f"compare.{arm}.{control}.{key}.overall"
    assert _number(resolve(f"{prefix}.difference", held).value) == pytest.approx((mine - theirs) * points)
    try:
        basis = resolve(f"{prefix}.basis", held).value
    except UnstatedFigure:
        basis = ""
    assert isinstance(basis, str) and not basis.startswith("paired"), basis
    axis = gain_axis(held, key)
    sign = 1.0 if axis.higher_is_better else -1.0
    assert axis.positions[arm] - axis.positions[control] == pytest.approx(sign * (mine - theirs))
    # A lead-aspect contrast still pairs: tagging the lead aspect by name changes nothing.
    paired = next(c for c in held.contrasts if c.aspect == lead_untested.aspect and c.interval is not None)
    lead_basis = resolve(f"compare.{paired.arm}.{control}.{paired.measure}.overall.basis", held).value
    assert isinstance(lead_basis, str) and lead_basis.startswith("paired")


def test_the_grammar_teaches_each_separation_family_on_its_own_page_with_a_rendered_example() -> None:
    lines = {line.split("`:", 1)[0]: line for line in figure_key_grammar().splitlines() if line.startswith("- `")}
    page = figure_key_pages()["separation"]
    for form in (
        "spread.<arm>.<measure>.<where>.<field>",
        "compare.<arm>.<arm>.<measure>.<where>.<field>",
        "settle.<arm>.<arm>.<measure>.<where>.<field>",
        "settle.criteria.<criterion>.<arm>.<where>.<field>",
        "settle.hypothetical.<line>.<arm>.<measure>.<where>.<field>",
    ):
        line = lines[f"- `{form}"]
        assert '` reads "' in line and "never a test the pipeline ran" in line, form
        assert "no interval." not in line, "a separation figure is not marked descriptive"
        assert f"- `{form}`" in page, form


def test_a_hypothetical_line_is_any_number_the_key_spells() -> None:
    held = _all_inputs()["r1"]
    for line in ("90", "90.5", "-3", "0.25"):
        figure = resolve(f"settle.hypothetical.{line}.atlas-m.overall.task.billing.line", held)
        assert figure.value == float(line)
    with pytest.raises(Exception, match="a number"):
        resolve("settle.hypothetical.ninety.atlas-m.overall.task.billing.line", held)


# =============================================================================
# arms_compare
# =============================================================================


@pytest.mark.parametrize("name", sorted(_all_inputs()))
def test_arms_compare_reads_every_place_both_arms_hold_and_says_where_the_interval_clears_zero(name: str) -> None:
    held = _all_inputs()[name]
    found = compared_pair(held)
    if found is None:
        assert len(held.arms) == 1 or not any(
            separation_wheres(held, pair, key) for pair in _pairs(held) for key in [*held.measures, *held.judged]
        ), f"{name}: arms share a measure, yet nothing compares them"
        return
    first, second, key, result = found
    assert [row.where for row in result.rows] == separation_wheres(held, (first, second), key)
    higher = True if key in held.judged else held.measures[key].higher_is_better
    for row in result.rows:
        if row.interval is None:
            assert row.unread and row.excludes_zero is None and row.ahead is None
            continue
        assert isinstance(row.interval.value, tuple)
        low, high = row.interval.value
        assert row.excludes_zero == (low > 0 or high < 0)
        if not row.excludes_zero or higher is None:
            assert row.ahead is None
        else:
            assert row.ahead == f"arm:{first if (low > 0) == higher else second}"
        assert row.unread == ""
    _each_figure_resolves(result.model_dump(mode="json"), held)
    seen = []
    offset: int | None = 0
    while offset is not None:
        page = arms_compare(held, (first, second), key, limit=1, offset=offset)
        seen += [row.where for row in page.rows]
        offset = page.next_offset
    assert seen == [row.where for row in result.rows]


def test_the_r1_editors_two_sample_reading_is_what_arms_compare_says() -> None:
    """The W2 r1-bars editor read each task's gap between the two candidates with a two-sample z from their own
    half-widths; arms_compare states that reading as keys, independent, at the judged level."""
    held = _all_inputs()["r1 reading only bars"]
    first, second = held.arms[0].id, held.arms[1].id
    result = arms_compare(held, (first, second), "overall", limit=WHOLE)
    level = level_share(held.judged["overall"].level)
    assert level is not None
    z = _NORMAL.inv_cdf(0.5 + level / 2)
    for row in result.rows:
        mine, theirs = _stat(held, first, "overall", row.where), _stat(held, second, "overall", row.where)
        assert mine.half_width is not None and theirs.half_width is not None and mine.mean and theirs.mean
        half = math.hypot(mine.half_width, theirs.half_width)
        assert row.interval is not None and row.interval.value == pytest.approx(
            (mine.mean - theirs.mean - half, mine.mean - theirs.mean + half)
        )
        assert row.level is not None and row.level.value == pytest.approx(level)
        assert z > 0


@pytest.mark.parametrize(
    ("arms", "measure", "says"),
    [
        (["atlas-l"], "overall", "two distinct arms"),
        (["atlas-l", "atlas-l"], "overall", "names one arm twice"),
        (["atlas-l", "no-such-arm"], "overall", "holds as no arm"),
        (["atlas-l", "atlas-m"], "no_such_measure", "names no measure"),
        (["atlas-l", "atlas-m"], "cost_per_1k", "measure_read"),
    ],
)
def test_arms_compare_refuses_what_it_cannot_read_and_says_what_to_call_instead(
    arms: list[str], measure: str, says: str
) -> None:
    with pytest.raises(RefusedInput, match=says) as refused:
        arms_compare(_all_inputs()["r1"], arms, measure)
    assert refused.value.remedy


# =============================================================================
# settle_cases
# =============================================================================


def test_the_r2_editors_sum_is_what_settle_cases_says_of_the_pick_against_its_gate() -> None:
    """The W2 r2 editor sized the re-test of the pick against its pass-rate gate by hand. settle_cases states it, read
    from the input: the binomial spread of the pick's pass rate on its count, one-sided at 95%."""
    held = _all_inputs()["r2"]
    (pick,) = dominant_options(held)
    gate = next(criterion for criterion in held.criteria if criterion.measure == "pass_rate")
    result = settle_cases(held, [pick], criterion=gate.id)
    (row,) = result.rows
    mean = held.measures["pass_rate"].per_arm[pick].mean
    count = held.scoreboard[pick].usable or held.scoreboard[pick].cases
    assert mean is not None and count is not None
    share, distance = mean / 100, mean - float(gate.threshold)
    per_case = math.sqrt(share * (1 - share)) * 100
    z = _NORMAL.inv_cdf(0.95)
    assert row.cases_needed is not None and row.cases_needed.value == math.ceil((z * per_case / distance) ** 2)
    assert row.wrong_side is not None
    assert row.wrong_side.value == pytest.approx(_NORMAL.cdf(-distance / (per_case / math.sqrt(count))))
    assert row.distance is not None and row.distance.value == pytest.approx(distance)
    assert [figure.value for figure in row.counts] == [count]
    assert "one-sided" in result.sided and not result.hypothetical
    # The editor's own case: a pass rate a tenth of a point over a 97% gate needs tens of thousands of cases.
    assert cases_to_separate(math.sqrt(0.971 * 0.029) * 100, 0.1, 0.95, sided=1, power=0.5) > 70_000


def _line_rows() -> Iterator[tuple[str, ReportInput, str, SettleRow]]:
    """Every row settling one arm against the criteria it is judged by, on every fixture: (input, arm, row). A row
    against a margin or floor, which reads a reading's change rather than a measure, is held in
    ``test_analyst_settle_reach``."""
    for name, held in _all_inputs().items():
        for arm in held.arms:
            try:
                result = settle_cases(held, [arm.id], limit=WHOLE)
            except RefusedInput:
                continue
            for row in result.rows:
                if row.against.startswith("criterion "):
                    yield name, held, arm.id, row


def _measure_of(held: ReportInput, row: SettleRow) -> str:
    criterion_id = row.against.removeprefix("criterion ")
    return next(criterion.measure for criterion in held.criteria if criterion.id == criterion_id)


def _level_words(row: SettleRow) -> tuple[float, str]:
    """A row's level as its figure words it: the share, and the sidedness it states."""
    assert row.level is not None and isinstance(row.level.value, str), row.level
    percent, _space, sided = row.level.value.partition("% ")
    return float(percent) / 100, sided


def test_a_line_row_says_it_is_one_sided_and_gives_the_size_at_which_the_printed_two_sided_interval_clears() -> None:
    """A page prints two-sided intervals. A line row's one-sided level says so in its own words, and beside its
    one-sided sizes it gives the size at which the two-sided interval at that level, centred on today's estimate,
    sits wholly on today's side of the line: there and not one case fewer."""
    checked = 0
    for name, held, arm, row in _line_rows():
        if row.cases_needed is None or row.cases_needed_two_sided is None:
            continue
        level, sided = _level_words(row)
        assert sided == "one-sided", (name, row.level)
        key = _measure_of(held, row)
        assert level == pytest.approx(_number(resolve(f"spread.{arm}.{key}.{row.where}.level", held).value))
        per_case = _number(resolve(f"spread.{arm}.{key}.{row.where}.per_case", held).value)
        assert row.distance is not None
        distance = abs(_number(row.distance.value))
        one, two = _NORMAL.inv_cdf(level), _NORMAL.inv_cdf(0.5 + level / 2)
        needed, printed = _number(row.cases_needed.value), _number(row.cases_needed_two_sided.value)
        # Within rounding: a size the arithmetic lands on exactly may round up by one.
        assert distance - one * per_case / math.sqrt(needed) >= -1e-9
        assert needed == 1 or distance - one * per_case / math.sqrt(needed - 1) < 1e-9
        assert distance - two * per_case / math.sqrt(printed) >= -1e-9
        assert printed == 1 or distance - two * per_case / math.sqrt(printed - 1) < 1e-9
        assert row.cases_needed_two_sided_80 is not None
        powered = _number(row.cases_needed_two_sided_80.value)
        assert _NORMAL.cdf(distance * math.sqrt(powered) / per_case - two) >= 0.8 - 1e-9
        assert powered == 1 or _NORMAL.cdf(distance * math.sqrt(powered - 1) / per_case - two) < 0.8
        checked += 1
    assert checked, "the control: some fixture arm is sized against its line"


def test_two_arms_settled_apart_say_their_level_is_two_sided_and_give_no_second_size() -> None:
    held = _all_inputs()["r1"]
    result = settle_cases(held, ["atlas-l", "atlas-m"], measure="overall", limit=WHOLE)
    for row in result.rows:
        assert _level_words(row)[1] == "two-sided"
        assert row.cases_needed_two_sided is None and row.cases_needed_two_sided_80 is None and row.settles is None


def test_a_row_says_which_side_its_estimate_is_on_and_a_failing_arms_size_confirms_its_miss() -> None:
    """A row's sizes settle whichever side today's estimate is on. For an arm on the failing side of its line, the
    size is the one at which a bound centred on today's estimate sits wholly on the failing side: it confirms the
    miss, and the row says so (``misses``); it never says such a design would clear the line."""
    sides: set[str] = set()
    for name, held, arm, row in _line_rows():
        if row.distance is None or row.cases_needed is None:
            continue
        distance = _number(row.distance.value)
        assert row.settles == ("clears" if distance > 0 else "misses"), (name, row.where, row.against)
        sides.add(row.settles)
        if distance < 0:
            z = _NORMAL.inv_cdf(_level_words(row)[0])
            per_case = _number(resolve(f"spread.{arm}.{_measure_of(held, row)}.{row.where}.per_case", held).value)
            needed = _number(row.cases_needed.value)
            # At that size the bound toward passing stays short of the line, the design showing the miss; one case
            # fewer and it reaches the line.
            assert distance + z * per_case / math.sqrt(needed) <= 1e-9
            assert needed == 1 or distance + z * per_case / math.sqrt(needed - 1) > -1e-9
    assert sides == {"clears", "misses"}, "the control: the fixtures hold an arm on each side of a line"


def _counted(held: ReportInput, arm: str, key: str, where: str) -> str:
    """What the count an arm's spread rests on counts, read off the input: its own n (cases where its n_cases equals
    it, trials where fewer, else observations), else its n_cases, else the slice's cases, or overall the scoreboard's
    usable trials, else its cases."""
    stat = _stat(held, arm, key, where)
    if stat.n is not None:
        return "observations" if stat.n_cases is None else "cases" if stat.n <= stat.n_cases else "trials"
    if stat.n_cases is not None or where != "overall":
        return "cases"
    board = held.scoreboard[arm]
    return "trials" if board.usable is not None else "cases"


def test_each_line_row_says_what_its_count_and_so_its_sizes_count() -> None:
    """A size counts what the count it rests on counts: on a repeated design that is trials, and the row says so."""
    seen: set[str] = set()
    for name, held, arm, row in _line_rows():
        if row.cases_needed is None:
            continue
        key = _measure_of(held, row)
        assert len(row.count_of) == len(row.counts) == 1, (name, row.where)
        (counted,) = row.count_of
        assert counted.key == f"spread.{arm}.{key}.{row.where}.counts"
        assert counted.value == _counted(held, arm, key, row.where), (name, arm, row.where)
        seen.add(str(counted.value))
    assert "cases" in seen, "the control: the fixtures size on cases"


def test_on_a_repeated_design_a_line_rows_sizes_count_cases_on_the_widest_spread_and_state_the_trials() -> None:
    """r2 with each pass rate stating three trials of each case: the count the spread rests on is trials, and the row
    says so. A fresh run adds cases, so each size counts cases at today's repeats; the binomial spread is read on the
    400 distinct cases, each counted once (the widest its repeats allow, since the snapshot holds no per-case
    outcomes), so the size is the most cases the repeats could need, the trials it takes beside it, and the basis and
    the result's text say so. Read as independent trials it would be three times fewer cases: the low end."""
    data = reference_data("r2")
    for stat in data["measures"]["pass_rate"]["per_arm"].values():
        stat["n_cases"], stat["n"] = 400, 1200
    held = ReportInput.model_validate(data)
    (pick,) = dominant_options(_all_inputs()["r2"])
    gate = next(criterion for criterion in held.criteria if criterion.measure == "pass_rate")
    result = settle_cases(held, [pick], criterion=gate.id)
    (row,) = result.rows
    assert [figure.value for figure in row.count_of] == ["trials"] and [f.value for f in row.counts] == [1200]
    assert [figure.value for figure in row.cases] == [400] and [figure.value for figure in row.repeats] == [3]
    mean = held.measures["pass_rate"].per_arm[pick].mean
    assert mean is not None and row.cases_needed is not None and row.trials_needed is not None
    share = mean / 100
    per_case = math.sqrt(share * (1 - share)) * 100
    distance = mean - float(gate.threshold)
    expected = math.ceil((_NORMAL.inv_cdf(0.95) * per_case / distance) ** 2)
    assert row.cases_needed.value == expected, "a size in cases, each case counted once: the widest spread"
    assert row.trials_needed.value == 3 * expected
    independent = math.ceil((_NORMAL.inv_cdf(0.95) * per_case / math.sqrt(3) / distance) ** 2)
    assert independent < expected, "the control: read as independent trials, fewer cases"
    assert row.wrong_side is not None
    assert row.wrong_side.value == pytest.approx(_NORMAL.cdf(-abs(distance) / (per_case / math.sqrt(400))))
    (basis,) = row.basis
    assert isinstance(basis.value, str) and "each counted once" in basis.value and "widest" in basis.value
    assert "widest" in result.sided and "most cases" in result.sided


def test_settling_one_arm_reads_every_criterion_it_is_judged_by_and_says_why_a_p95_is_not_sized() -> None:
    held = _all_inputs()["r2"]
    arm = held.arms[0].id
    result = settle_cases(held, [arm], limit=WHOLE)
    assert [row.against for row in result.rows] == [f"criterion {criterion.id}" for criterion in held.criteria]
    p95 = next(row for row in result.rows if row.against == "criterion p95")
    assert p95.cases_needed is None and "p95" in p95.unsettled and "settle reads a mean" in p95.unsettled
    _each_figure_resolves(result.model_dump(mode="json"), held)


def test_a_hypothetical_line_is_keyed_hypothetical_and_read_as_a_declared_one_would_be() -> None:
    held = _all_inputs()["r2 without its gates"]
    declared = _all_inputs()["r2"]
    gate = next(criterion for criterion in declared.criteria if criterion.measure == "pass_rate")
    (pick,) = dominant_options(declared)
    result = settle_cases(held, [pick], measure="pass_rate", line=float(gate.threshold))
    assert result.hypothetical
    overall = result.rows[0]
    assert overall.where == "overall" and overall.line is not None
    for figure in (overall.line, overall.distance, overall.cases_needed, overall.cases_needed_80, overall.wrong_side):
        assert figure is not None and figure.key.startswith("settle.hypothetical.")
    (against_gate,) = settle_cases(declared, [pick], criterion=gate.id).rows
    for field in ("distance", "cases_needed", "cases_needed_80", "wrong_side", "level"):
        assert getattr(overall, field).value == getattr(against_gate, field).value, field


def test_two_arms_are_settled_apart_two_sided_each_arm_running_the_cases() -> None:
    held = _all_inputs()["r1"]
    first, second = "atlas-l", "atlas-m"
    result = settle_cases(held, [first, second], measure="overall", limit=WHOLE)
    assert "two-sided" in result.sided
    z = _NORMAL.inv_cdf(0.975)
    for row in result.rows:
        mine, theirs = _stat(held, first, "overall", row.where), _stat(held, second, "overall", row.where)
        assert mine.mean is not None and theirs.mean is not None and row.cases_needed is not None
        per_case = [
            _number(resolve(f"spread.{arm}.overall.{row.where}.per_case", held).value) for arm in (first, second)
        ]
        expected = math.ceil((z * math.hypot(*per_case) / (mine.mean - theirs.mean)) ** 2)
        assert row.cases_needed.value == expected
        assert len(row.counts) == len(row.basis) == 2


def test_an_estimate_on_its_line_says_no_number_of_cases_separates_it_and_nothing_to_settle_is_refused() -> None:
    data = reference_data("r2")
    held = ReportInput.model_validate(data)
    pick = next(arm.id for arm in held.arms)
    mean = held.measures["pass_rate"].per_arm[pick].mean
    assert mean is not None
    for criterion in data["criteria"]:
        if criterion["measure"] == "pass_rate":
            criterion["threshold"] = mean
    on_line = ReportInput.model_validate(data)
    gate = next(criterion.id for criterion in on_line.criteria if criterion.measure == "pass_rate")
    with pytest.raises(RefusedInput, match="exactly on its line"):
        settle_cases(on_line, [pick], criterion=gate)
    with pytest.raises(RefusedInput, match="p95"):
        settle_cases(held, [pick], criterion="p95")


@pytest.mark.parametrize(
    ("arms", "named", "says"),
    [
        (["atlas-m"], {"criterion": "no-such"}, "names no criterion"),
        (["atlas-m"], {"line": 90.0}, "measure_key is not given"),
        (["atlas-m"], {"line": 90.0, "criterion": "overall", "measure": "overall"}, "not both"),
        (["atlas-m", "atlas-l"], {"line": 90.0, "measure": "overall"}, "not against a line"),
        (["atlas-m", "atlas-l"], {}, "measure_key is not given"),
        (["atlas-m"], {"line": 1e-9, "measure": "overall"}, "exponent notation"),
        (["atlas-m", "atlas-l", "brisk-s"], {}, "one or two distinct arms"),
    ],
)
def test_settle_cases_refuses_what_names_nothing_to_settle_with_a_remedy(
    arms: list[str], named: dict[str, Any], says: str
) -> None:
    with pytest.raises(RefusedInput, match=says) as refused:
        settle_cases(_all_inputs()["r1"], arms, **named)
    assert refused.value.remedy


def test_one_arm_no_criterion_reads_is_refused_naming_the_hypothetical_line() -> None:
    held = _all_inputs()["r2 without its gates"]
    with pytest.raises(RefusedInput, match="no declared criterion reads") as refused:
        settle_cases(held, [held.arms[0].id])
    assert "hypothetical_line" in refused.value.remedy


# =============================================================================
# Over the tool, on every fixture snapshot
# =============================================================================


def _analyst_tool() -> MountedTool:
    (spec,) = analyst_tools()
    return eval_catalogue().mount(spec)


def _seeded() -> tuple[OpsFixture, dict[str, EvalReportSnapshot]]:
    fixture = ops_fixture()
    snapshots = fixture_snapshots(TOYHOST_SCOPE)
    for snapshot in snapshots.values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture, snapshots


async def _call_tool(tool: MountedTool, fixture: OpsFixture, arguments: dict[str, Any]) -> ActionOutcome:
    return await tool.call(arguments, host=fixture.host, caller=CALLER)


def _every_call(held: ReportInput) -> Iterator[dict[str, Any]]:
    keys = [*held.measures, *held.judged]
    arms = [arm.id for arm in held.arms][:3]
    for first, second in zip(arms, arms[1:], strict=False):
        for key in keys:
            yield {"action": "arms_compare", "arm_ids": [first, second], "measure_key": key}
            yield {"action": "settle_cases", "arm_ids": [first, second], "measure_key": key}
    for arm in arms:
        yield {"action": "settle_cases", "arm_ids": [arm]}


async def test_both_actions_on_every_fixture_snapshot_answer_or_refuse_with_a_remedy_and_never_raise() -> None:
    fixture, snapshots = off_loop(_seeded)
    tool = _analyst_tool()
    answered: set[str] = set()
    for name, snapshot in snapshots.items():
        held = frozen_input(snapshot)
        for arguments in _every_call(held):
            outcome = await _call_tool(tool, fixture, {**arguments, "snapshot_id": snapshot.id})
            where = f"{arguments} on {name}"
            assert outcome.ending in ("ok", "refused"), f"{where}: {outcome.text}"
            if outcome.ending == "refused":
                assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed", where
                assert "remedy:" in outcome.text, where
                continue
            assert outcome.structured is not None
            _each_figure_resolves(outcome.structured, held)
            answered.add(arguments["action"])
    assert answered == {"arms_compare", "settle_cases"}


async def test_the_editors_tool_carries_both_actions() -> None:
    (spec,) = analyst_tools(role="editor")
    editor = eval_catalogue().mount(spec)
    assert {"arms_compare", "settle_cases"} <= {action.name for action in editor.actions}
