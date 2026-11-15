"""The parity guard: every figure of a measure the code page states, the written page's input holds (by value and role
only: what it does not match on, and what it skips, are stated below).

The class of defect this closes is "the code page can state it, the written page cannot" (#1125, #1129, #1067): a
fact computed for one page and never carried to the other. It is closed by construction rather than instance by
instance. For each hot-path campaign, the campaign's code page is read as a host reads it, and every figure its forms
handed out is collected from their ledgers (:attr:`~threetears.evals.analysis.report.forms.PageForms.ledger`: each
figure by its measure, its kind and its role). The report input is built from the same campaign's bundle, and every
figure the code page issued for a measure must resolve from that input:

- **the same measure and kind**: a stratum's figure is filed under its measure or its judged dimension, by which of
  the input's tables names it;
- **a field of the same role** (:func:`bucket_of`): a level resolves against levels only, a change against changes, a
  half-width against half-widths and spreads, a declared margin against margins and thresholds, an interval against
  intervals; a table's column, which holds its figure rounded to the places it prints, against any of them at those
  places;
- **at a scale the measure's unit is printed at** (:func:`scales_of`): as stored, or its one printed scale (a share in
  percent, money per thousand, milliseconds in seconds), never every scale;
- **a value the input holds as it is**: nothing is derived here but a scoreboard row's passes over its usable trials,
  the pass rate the input states as two counts.

Values and roles are compared, never text. A ledger entry names no arm, aspect, stratum or contrast (``_Issued``
holds the measure, kind, role and value), so a figure resolves when ANY of its measure's figures of that role bucket,
anywhere in the input, is the same value at any scale its unit is printed at: what the written page can state, not
where. The guard does NOT match on:

- **the arm**: two arms' figures swapped pass, and so does the control's baseline printed as a candidate's level;
- **the aspect**: aspect X's figure printed under aspect Y passes, since every aspect's readings, scoreboards, floors
  and across-aspects figures pool into one held set per measure and kind;
- **the stratum**: a stratum's figure printed as the aspect's passes, since strata are filed with their measure;
- **the contrast**: one contrast's delta printed under another contrast of the same measure passes;
- **the interval's role**: a level's interval printed where a change's belongs passes, since :data:`INTERVAL_FIELDS`
  pools ``ci``, ``within_interval`` and ``gain_interval``;
- **the scale**: a unit with two printed scales (``trial``: 100 and 0.01) resolves at either.

So an arm swap or a cross-aspect swap passes today. Recording arm, aspect and contrast on the ledger, and matching on
them here, is the page forms' rework, chunk S5c.

**Only figures of a measure are checked.** A ledger entry with no measure (a count of trials, a p-value, an effect
size, a confidence, a multiple, a money total, a preference figure, a lone rate's or a scale span's interval) is
skipped, and coverage is only of the figure kinds the fixtures' pages print. A skip is never silent: the guard counts
the figures it skipped for lacking a measure, and :data:`UNMEASURED` pins that count per page, so a new unmeasured
figure fails the test until someone decides whether it needs a measure (or the guard a rule for it).

**The allowance** lists each figure the code page states that the input does not hold, by campaign, measure, kind,
role and the value issued, with what holds it apart (:data:`ALLOWANCE`): a new figure missing from the input fails
even where the same measure and role are allowed for another value. An allowance entry nothing issues any more fails
too, so the allowance never outlives its gap. Every reason is one the written page's renderer closes from fields the
input already holds (:data:`GAP_REASONS`): the input has a field for every figure.

**Nothing passes for having checked nothing**: each campaign's page must have been assembled in forms, and must have
resolved at least the figures it resolves today (:data:`RESOLVED_AT_LEAST`).

**The written page is held to its input too** (:data:`WRITTEN`): a campaign whose report was written and stored is
read as a host reads it, replayed from its capture (no model is called), and every figure its forms
(:meth:`~threetears.evals.analysis.report.forms.PageForms.of`) handed out must resolve, by the same rule and with no
allowance, from the input those forms were built from. A stored page reads the input frozen with it, never a fresh
build, so that is the input it is held to.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from packages.evals.tests.test_hot_path_oracle import REPORT_NOW, WITH_BUNDLES, subject
from threetears.evals.analysis import service as analysis_service
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report_input import SHARE_UNIT, ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.ops import report_read

#: The scale each unit is printed at beside as stored: a share in percent, money per thousand replies, milliseconds in
#: seconds. The pass rate is a share, and the input states some of its figures in percent and points.
UNIT_SCALES: dict[str, tuple[float, ...]] = {
    SHARE_UNIT: (100.0,),
    "pass_fail": (100.0,),
    "class_label": (100.0,),
    "usd": (1000.0,),
    "ms": (0.001,),
    "trial": (100.0, 0.01),
}

#: The roles whose figure is a level of its measure: an arm's figure, as a page states it.
LEVEL_ROLES = frozenset({"level", "pass_k", "lone", "reading", "echoed", "check", "money", "share", "hits"})

#: The fields of the input that hold a measure's level.
LEVEL_FIELDS = frozenset(
    {"mean", "p95", "no_variation", "value", "baseline", "arm_mean", "reply_cost_usd", "test_cost_usd"}
)

#: The fields of the input that hold a change of a measure: an arm minus the control, a drop, how far it flips.
CHANGE_FIELDS = frozenset({"change", "delta", "worst_drop", "short_by", "flip_margin"})

#: The fields of the input that hold a line a figure is read against: a margin, a threshold, a bar.
DECLARED_FIELDS = frozenset({"margin", "threshold", "bar"})

#: The fields of the input that hold the half-width of an interval, or a spread.
WIDTH_FIELDS = frozenset({"half_width", "own_half_width", "sem", "spread"})

#: The fields of the input that hold an interval, low then high.
INTERVAL_FIELDS = frozenset({"ci", "interval", "within_interval", "gain_interval"})


@dataclass
class Held:
    """What the input holds for one measure of one kind, by the role of the field that holds it, and the scales its
    unit is printed at."""

    scales: tuple[float, ...] = (1.0,)
    levels: set[float] = field(default_factory=set)
    changes: set[float] = field(default_factory=set)
    declared: set[float] = field(default_factory=set)
    widths: set[float] = field(default_factory=set)
    intervals: set[tuple[float, float]] = field(default_factory=set)

    def take(self, node: Any) -> None:
        """Every figure under ``node``, filed by the field that holds it."""
        if isinstance(node, Mapping):
            for key, value in node.items():
                self._file(key, value)
                self.take(value)
        elif isinstance(node, list | tuple):
            for value in node:
                self.take(value)

    def _file(self, key: str, value: Any) -> None:
        if key in INTERVAL_FIELDS and isinstance(value, list | tuple) and len(value) == 2:
            low, high = value
            if _number(low) and _number(high):
                self.intervals.add((float(low), float(high)))
                self.widths.add((float(high) - float(low)) / 2)
            return
        if key == "own_half_width" and isinstance(value, Mapping):
            self.widths.update(float(width) for width in value.values() if _number(width))
            return
        if not _number(value):
            return
        for fields, figures in (
            (LEVEL_FIELDS, self.levels),
            (CHANGE_FIELDS, self.changes),
            (DECLARED_FIELDS, self.declared),
            (WIDTH_FIELDS, self.widths),
        ):
            if key in fields:
                figures.add(float(value))

    def scalars(self, role: str) -> set[float]:
        """The figures a role resolves against (:func:`bucket_of`)."""
        return {
            "levels": self.levels,
            "changes": self.changes,
            "declared": self.declared,
            "widths": self.widths,
            "any": self.levels | self.changes | self.declared | self.widths,
        }[bucket_of(role)]


def bucket_of(role: str) -> str:
    """Which of the input's figures a ledger role resolves against: a table's column any, since it holds whatever
    figure its column prints."""
    if role in LEVEL_ROLES:
        return "levels"
    if role in ("change", "half_width", "declared"):
        return {"change": "changes", "half_width": "widths", "declared": "declared"}[role]
    if role == "column":
        return "any"
    raise AssertionError(f"a measure's figure of role {role!r} has no field of the input to resolve against")


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def scales_of(document: Mapping[str, Any], name: str, kind: str) -> tuple[float, ...]:
    """The scales a measure's figure is printed at: as stored, and its unit's one printed scale."""
    if kind == "trial":
        return (1.0, *UNIT_SCALES["trial"])
    if kind == "judged":
        scale = (document["judged"].get(name) or {}).get("scale")
        return (1.0, *UNIT_SCALES.get(scale or "", ()))
    measure = document["measures"].get(name) or {}
    unit, grade = measure.get("unit") or "", measure.get("grade") or ""
    printed = UNIT_SCALES.get(grade) or UNIT_SCALES.get(unit) or UNIT_SCALES.get(unit.split("_per_")[0], ())
    return (1.0, *printed)


def held_by_measure(built: ReportInput) -> dict[tuple[str, str], Held]:
    """What the input holds, by ``(measure, kind)``: the kinds a page's forms key a figure by."""
    document = built.model_dump(mode="json", by_alias=True)
    held: dict[tuple[str, str], Held] = {}

    def of(measure: str, kind: str) -> Held:
        if (measure, kind) not in held:
            held[measure, kind] = Held(scales=scales_of(document, measure, kind))
        return held[measure, kind]

    for name, measure in document["measures"].items():
        of(name, "measure").take(measure)
    for name, judged in document["judged"].items():
        of(name, "judged").take(judged)
    for stratum in document["strata"]:
        for figures in stratum["per_arm"].values():
            for name, stat in figures.items():
                of(name, "measure" if name in document["measures"] else "judged").take(stat)
    for contrast in document["contrasts"]:
        of(contrast["measure"], _kind(contrast.get("kind"))).take(contrast)
    for bar in document["bars"]:
        of(bar["measure"], "measure").take(bar)
    for criterion in document["criteria"]:
        of(criterion["measure"], "measure").take(criterion)
    for guardrail in document["guardrails"]:
        of(guardrail["name"], _kind(guardrail.get("reading"))).take(guardrail)
    # The pass rate: each aspect's readings, pairs and pass^k, each arm's scoreboard, each floor, and each arm's figures
    # across the aspects.
    passed = of("passed", "trial")
    for aspect in document["aspects"]:
        passed.take({"readings": aspect["readings"], "pairs": aspect["pairs"], "pass_k": aspect.get("pass_k") or []})
    for board in [document["scoreboard"], *(aspect.get("scoreboard") or {} for aspect in document["aspects"])]:
        for row in board.values():
            if row.get("usable"):
                passed.levels.add(row["passed"] / row["usable"])
    for floor in document["floors"]:
        passed.take(floor)
    for arm in ((document.get("across_aspects") or {}).get("per_arm") or {}).values():
        passed.take({"value": arm["pass_rate"]["value"], "interval": arm["pass_rate"]["interval"]})
        if arm.get("change") is not None:
            passed.take({"change": arm["change"]["value"], "interval": arm["change"]["interval"]})
    return held


def _kind(kind: str | None) -> str:
    """A contrast's kind as a page's forms key it: a measure's own where it names none."""
    return kind if kind in ("judged", "trial") else "measure"


def resolves(held: Held | None, role: str, value: float | tuple[float, float]) -> bool:
    """Whether the input holds ``value`` in a field of ``role``, at a scale its measure is printed at."""
    if held is None:
        return False
    if isinstance(value, tuple):
        return any(
            _same(low * scale, value[0], role) and _same(high * scale, value[1], role)
            for low, high in held.intervals
            for scale in held.scales
        )
    return any(_same(figure * scale, value, role) for figure in held.scalars(role) for scale in held.scales)


def _same(stored: float, issued: float, role: str) -> bool:
    """Whether a stored figure, scaled, is the issued one: equal to twelve figures, or, for a table's column, which
    holds its figure rounded to the places it prints, equal at those places."""
    if math.isclose(stored, issued, rel_tol=1e-9, abs_tol=1e-12):
        return True
    if role != "column":
        return False
    places = next((places for places in range(5) if round(issued, places) == issued), None)
    return places is not None and abs(stored - issued) <= 0.5 * 10**-places + 1e-9


#: A value as an allowance names it: nine significant figures, an interval's ends each.
Value = float | tuple[float, float]


def _keyed(value: float | tuple[float, float]) -> Value:
    if isinstance(value, tuple):
        return (float(f"{value[0]:.9g}"), float(f"{value[1]:.9g}"))
    return float(f"{value:.9g}")


@dataclass(frozen=True)
class Gap:
    """A figure the code page states that the input does not hold: its campaign, measure, kind, role and value."""

    campaign: str
    measure: str
    kind: str
    role: str
    value: Value


#: Why a figure the code page states is not in the input as it is printed. Each is one the written page's renderer
#: (chunk S5b) closes from fields the input holds: none is a figure the input has no field for.
GAP_REASONS: dict[str, str] = {
    "outcome_share": (
        "the share of an arm's trials drawn on the scoreboard's outcome bar where some were lost: the row's passes "
        "over every trial it ran (ScoreboardRow.passed over ran), which the renderer divides as it divides passes over "
        "usable trials"
    ),
    "equal_weight_row": (
        "the level the scorecard's across-aspects row states for each arm: the mean of the matrix's cells above it, "
        "each held at the page's printed precision, so a figure of the page's own cells, which the renderer reads off "
        "each aspect's pass rate the input holds (Aspect.readings, the scoreboards)"
    ),
}

#: Each figure the code page states that the input does not hold as printed, by campaign, measure, kind, role and
#: value, with why (:data:`GAP_REASONS`).
ALLOWANCE: dict[Gap, str] = {
    Gap("scaled", "passed", "trial", "share", 0.199760766): "outcome_share",
    Gap("scaled", "passed", "trial", "share", 0.227631579): "outcome_share",
    Gap("scaled", "passed", "trial", "share", 0.246052632): "outcome_share",
    Gap("scaled_two_aspects", "passed", "trial", "level", 16.0): "equal_weight_row",
    Gap("scaled_two_aspects", "passed", "trial", "level", 21.5): "equal_weight_row",
    Gap("scaled_two_aspects", "passed", "trial", "level", 23.5): "equal_weight_row",
    Gap("scaled_two_aspects", "passed", "trial", "level", 26.0): "equal_weight_row",
    Gap("scaled_two_aspects", "passed", "trial", "share", 0.165789474): "outcome_share",
    Gap("scaled_two_aspects", "passed", "trial", "share", 0.228947368): "outcome_share",
    Gap("scaled_two_aspects", "passed", "trial", "share", 0.239473684): "outcome_share",
    Gap("scaled_two_aspects", "passed", "trial", "share", 0.281578947): "outcome_share",
}

#: The fewest figures each campaign's code page resolves from its input: what it resolves today.
RESOLVED_AT_LEAST: dict[str, int] = {"scaled": 943, "scaled_two_aspects": 1801, "toyhost": 39, "courierhost": 7}


@dataclass(frozen=True)
class Issued:
    """One figure of a measure the code page's forms handed out."""

    measure: str
    kind: str
    role: str
    value: float | tuple[float, float]


def _read(name: str, constructors: tuple[str, ...]) -> dict[str, list[Any]]:
    """Every page forms each of ``constructors`` built while campaign ``name``'s page was read as a host reads it."""
    built: dict[str, list[Any]] = {constructor: [] for constructor in constructors}
    patch = pytest.MonkeyPatch()
    for constructor, made_by in built.items():
        original = getattr(PageForms, constructor).__func__

        def spying(
            cls: type[PageForms], *args: Any, __original: Any = original, __made: Any = made_by, **kwargs: Any
        ) -> Any:
            made = __original(cls, *args, **kwargs)
            __made.append(made)
            return made

        patch.setattr(PageForms, constructor, classmethod(spying))
    patch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)
    try:
        read = subject(name)
        report_read(read.host, read.campaign_id, read.scope_id, format="markdown")
    finally:
        patch.undo()
    return built


@dataclass(frozen=True)
class Stated:
    """What one page's forms handed out: every figure of a measure, and how many figures the guard skipped because
    they name no measure (never checked, so counted)."""

    issued: tuple[Issued, ...]
    unmeasured: int


def _issued(forms: Iterable[PageForms]) -> Stated:
    """Every figure of a measure ``forms`` handed out, and how many figures they handed out with no measure."""
    entries = [entry for one in forms for entry in one.ledger if entry.value is not None]
    return Stated(
        issued=tuple(
            Issued(entry.measure, entry.kind or "measure", entry.role, entry.value)
            for entry in entries
            if entry.measure
        ),
        unmeasured=sum(1 for entry in entries if not entry.measure),
    )


@functools.cache
def code_page(name: str) -> Stated:
    """Every figure campaign ``name``'s code page stated, read as a host reads the page."""
    built = _read(name, ("of_campaign", "of_aspects", "of_bundle"))
    assert built["of_campaign"], f"{name}: the code page was assembled in no campaign forms, so nothing is checked"
    return _issued(
        one for made in sum(built.values(), []) for one in (made.values() if isinstance(made, Mapping) else [made])
    )


#: The campaigns read from a written and stored report, and the fewest figures each one's written page resolves from
#: its input: what it resolves today.
WRITTEN: dict[str, int] = {"toyhost_doc": 30}

#: How many figures each page states that name no measure, so the guard skips them: exactly what it skips today. A
#: page stating one more (or one fewer) fails until someone decides whether that figure needs a measure.
UNMEASURED: dict[str, int] = {
    "scaled": 307,
    "scaled_two_aspects": 565,
    "toyhost": 47,
    "courierhost": 28,
    "toyhost_doc": 6,
}


@functools.cache
def written_page(name: str) -> tuple[Stated, ReportInput]:
    """Every figure campaign ``name``'s written page stated, read as a host reads the page, and the one input its forms
    were built from."""
    (made,) = _read(name, ("of",)).values()
    assert made, f"{name}: the written page was assembled in no written forms, so nothing is checked"
    inputs = {id(one.report_input): one.report_input for one in made}
    assert len(inputs) == 1, f"{name}: the written page's forms read {len(inputs)} inputs, not one"
    (page_input,) = inputs.values()
    return _issued(made), page_input


@functools.cache
def built_input(name: str) -> ReportInput:
    """The report input built from campaign ``name``'s bundle."""
    read = subject(name)
    return report_input(assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id))


def gaps_of(name: str, plant: Callable[[ReportInput], ReportInput] = lambda built: built) -> tuple[set[Gap], int]:
    """Every figure campaign ``name``'s code page states that its report input (as ``plant`` leaves it) does not hold,
    and how many it does."""
    return _gaps(name, code_page(name).issued, plant(built_input(name)))


def _gaps(name: str, issued: Iterable[Issued], built: ReportInput) -> tuple[set[Gap], int]:
    """Each of ``issued`` that ``built`` does not hold, as a gap of campaign ``name``, and how many it does."""
    held = held_by_measure(built)
    found: set[Gap] = set()
    resolved = 0
    for one in issued:
        if resolves(held.get((one.measure, one.kind)), one.role, one.value):
            resolved += 1
        else:
            found.add(Gap(name, one.measure, one.kind, one.role, _keyed(one.value)))
    return found, resolved


def test_every_allowed_gap_has_a_stated_reason() -> None:
    assert set(ALLOWANCE.values()) <= set(GAP_REASONS), sorted(set(ALLOWANCE.values()) - set(GAP_REASONS))
    assert set(GAP_REASONS) <= set(ALLOWANCE.values()), "a reason no gap holds is slack"


def test_every_campaign_is_held_to_a_count_it_resolves() -> None:
    assert set(RESOLVED_AT_LEAST) == set(WITH_BUNDLES)


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_every_figure_the_code_page_states_the_input_holds(name: str) -> None:
    found, resolved = gaps_of(name)
    allowed = {gap for gap in ALLOWANCE if gap.campaign == name}

    assert not found - allowed, (
        f"the code page states figures the input does not hold: {sorted(found - allowed, key=str)}"
    )
    assert not allowed - found, f"the allowance names gaps that are closed: {sorted(allowed - found, key=str)}"
    assert resolved >= RESOLVED_AT_LEAST[name], f"{name}: {resolved} figures resolved, fewer than it resolves today"


@pytest.mark.parametrize("name", WRITTEN)
def test_every_figure_the_written_page_states_its_input_holds(name: str) -> None:
    found, resolved = written_gaps(name)

    assert not found, f"the written page states figures its input does not hold: {sorted(found, key=str)}"
    assert resolved >= WRITTEN[name], f"{name}: {resolved} figures resolved, fewer than it resolves today"


def written_gaps(name: str, plant: Callable[[ReportInput], ReportInput] = lambda built: built) -> tuple[set[Gap], int]:
    """Every figure campaign ``name``'s written page states that its input (as ``plant`` leaves it) does not hold, and
    how many it does."""
    stated, page_input = written_page(name)
    return _gaps(name, stated.issued, plant(page_input))


def test_every_page_is_held_to_the_count_of_figures_it_skips() -> None:
    assert set(UNMEASURED) == set(WITH_BUNDLES) | set(WRITTEN)


@pytest.mark.parametrize("name", sorted(UNMEASURED))
def test_the_figures_the_guard_skips_for_naming_no_measure_are_the_ones_it_skips_today(name: str) -> None:
    """A figure with no measure is not checked (the module docstring names which), and never skipped silently: each
    page skips exactly the figures it skips today, so a new one fails here until someone decides."""
    stated = written_page(name)[0] if name in WRITTEN else code_page(name)

    assert stated.unmeasured == UNMEASURED[name], f"{name} states {stated.unmeasured} figures that name no measure"


def _without_pass_k(built: ReportInput) -> ReportInput:
    """The input with every arm's pass^k taken off its aspects, as it was before the field was filled."""
    aspects = [
        aspect.model_copy(
            update={
                "pass_k": [
                    held.model_copy(update={"per_arm": None, "at_common_depth": None, "cells": None})
                    for held in aspect.pass_k
                ]
                if aspect.pass_k
                else aspect.pass_k
            }
        )
        for aspect in built.aspects
    ]
    return built.model_copy(update={"aspects": aspects})


def test_a_figure_taken_out_of_the_input_is_a_gap_the_guard_names() -> None:
    """The guard is no tautology: taking a filled field back out of the input fails it, naming the figure."""
    found, _ = gaps_of("toyhost", _without_pass_k)
    allowed = {gap for gap in ALLOWANCE if gap.campaign == "toyhost"}

    assert {(gap.measure, gap.kind, gap.role) for gap in found - allowed} >= {("passed", "trial", "pass_k")}


def _without_measure_intervals(built: ReportInput) -> ReportInput:
    """The input with every measure figure's interval taken out."""
    measures = {
        name: measure.model_copy(
            update={"per_arm": {arm: stat.model_copy(update={"ci": None}) for arm, stat in measure.per_arm.items()}}
        )
        for name, measure in built.measures.items()
    }
    return built.model_copy(update={"measures": measures})


def test_a_figure_taken_out_of_the_written_page_s_input_is_a_gap_the_guard_names() -> None:
    """The written page's guard is no tautology either: the intervals its measures print, taken out of its input,
    fail it, naming each."""
    found, _ = written_gaps("toyhost_doc", _without_measure_intervals)

    assert {(gap.measure, gap.kind, gap.role) for gap in found} >= {("total_ms", "measure", "interval")}
