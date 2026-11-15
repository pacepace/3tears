"""The parity guard: every figure the code page states, the written page's input holds.

The class of defect this closes is "the code page can state it, the written page cannot" (#1125, #1129, #1067): a
fact computed for one page and never carried to the other. It is closed by construction rather than instance by
instance. For each hot-path campaign, the campaign's code page is read as a host reads it, and every figure its forms
handed out is collected from their ledgers (:attr:`~threetears.evals.analysis.report.forms.PageForms.ledger`: each
figure by its measure, its kind and its role). The report input is built from the same campaign's bundle, and every
figure the code page issued for a measure must resolve from that input: a value the input holds for the same measure
and kind, in a field of the same role, at the scale the page printed it at.

Values and roles are compared, never text. A ledger entry names no arm, so a figure resolves when any arm's or any
contrast's figure of its measure is the same value: what the written page can state, not where it states it.

**The allowance** lists each figure the code page states that the input does not yet hold, by campaign, measure, kind
and role, with what holds it apart (:data:`ALLOWANCE`). A figure missing from the input that the allowance does not
name fails; so does an allowance entry nothing issues any more, so the allowance never outlives its gap.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from packages.evals.tests.test_hot_path_oracle import REPORT_NOW, WITH_BUNDLES, subject
from threetears.evals.analysis import service as analysis_service
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.ops import report_read

#: The scales a page prints a stored figure at: as stored, a share in percent, money per thousand replies, and a
#: percent stored as a share.
SCALES = (1.0, 100.0, 1000.0, 0.01)

#: The roles whose figure is a level of its measure: an arm's figure, a threshold, a baseline.
LEVEL_ROLES = frozenset({"level", "pass_k", "lone", "reading", "echoed", "check", "money", "share", "hits"})

#: The fields of the input that hold a measure's level.
LEVEL_FIELDS = frozenset(
    {
        "mean",
        "p95",
        "no_variation",
        "value",
        "bar",
        "threshold",
        "baseline",
        "reply_cost_usd",
        "test_cost_usd",
    }
)

#: The fields of the input that hold a change of a measure: an arm minus the control, a drop, a margin.
CHANGE_FIELDS = frozenset({"change", "delta", "worst_drop", "margin", "short_by", "flip_margin"})

#: The fields of the input that hold the half-width of an interval, or a spread.
WIDTH_FIELDS = frozenset({"half_width", "own_half_width", "sem", "spread"})

#: The fields of the input that hold an interval, low then high.
INTERVAL_FIELDS = frozenset({"ci", "interval", "within_interval", "gain_interval"})


@dataclass
class Held:
    """What the input holds for one measure of one kind, by role."""

    levels: set[float] = field(default_factory=set)
    changes: set[float] = field(default_factory=set)
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
            if key.endswith("passed") and isinstance(value, str) and " of " in value:
                passed, _, of = value.partition(" of ")
                if passed.isdigit() and of.isdigit() and int(of):
                    self.levels.add(int(passed) / int(of))
            return
        if key in LEVEL_FIELDS:
            self.levels.add(float(value))
        elif key in CHANGE_FIELDS:
            self.changes.add(float(value))
        elif key in WIDTH_FIELDS:
            self.widths.add(float(value))

    def scalars(self, role: str) -> set[float]:
        """The figures a role resolves against."""
        if role in LEVEL_ROLES:
            return self.levels
        if role == "change":
            return self.changes
        if role == "half_width":
            return self.widths
        # A table's column holds whatever figure it prints: any of them.
        return self.levels | self.changes | self.widths


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def held_by_measure(built: ReportInput) -> dict[tuple[str, str], Held]:
    """What the input holds, by ``(measure, kind)``: the kinds a page's forms key a figure by."""
    document = built.model_dump(mode="json", by_alias=True)
    held: dict[tuple[str, str], Held] = {}

    def of(measure: str, kind: str) -> Held:
        return held.setdefault((measure, kind), Held())

    for name, measure in document["measures"].items():
        of(name, "measure").take(measure)
    for name, judged in document["judged"].items():
        of(name, "judged").take(judged)
    for stratum in document["strata"]:
        for dimensions in stratum["per_arm"].values():
            for name, stat in dimensions.items():
                of(name, "judged").take(stat)
    for contrast in document["contrasts"]:
        held_by = of(contrast["measure"], _kind(contrast.get("kind")))
        held_by.take(contrast)
        # The contrast's arm side: its delta is the difference of the two means the test read, so the arm's is the
        # paired baseline plus the delta.
        if _number(contrast.get("baseline")) and _number(contrast.get("delta")):
            held_by.levels.add(contrast["baseline"] + contrast["delta"])
    for bar in document["bars"]:
        of(bar["measure"], "measure").take(bar)
    for criterion in document["criteria"]:
        of(criterion["measure"], "measure").take(criterion)
    for guardrail in document["guardrails"]:
        of(guardrail["name"], _kind(guardrail.get("reading"))).take(guardrail)
    # The pass rate: each aspect's readings and pairs, each arm's scoreboard, each floor.
    passed = of("passed", "trial")
    for aspect in document["aspects"]:
        passed.take({"readings": aspect["readings"], "pairs": aspect["pairs"]})
    for board in [document["scoreboard"], *(aspect.get("scoreboard") or {} for aspect in document["aspects"])]:
        for row in board.values():
            if row.get("usable"):
                passed.levels.add(row["passed"] / row["usable"])
            # The one cost reading, and the reply's cost, are each arm's cost per reply.
            cost = Held()
            cost.take(row)
            for key, kind in held:
                if kind == "measure" and key.startswith(("cost", "production_")):
                    held[key, kind].levels |= cost.levels
    for floor in document["floors"]:
        passed.take(floor)
    return held


def _kind(kind: str | None) -> str:
    """A contrast's kind as a page's forms key it: a measure's own where it names none."""
    return kind if kind in ("judged", "trial") else "measure"


def resolves(held: Held | None, role: str, value: float | tuple[float, float]) -> bool:
    """Whether the input holds ``value`` in a field of ``role``, at a scale a page prints it at."""
    if held is None:
        return False
    if isinstance(value, tuple):
        return any(
            _same(low * scale, value[0], role) and _same(high * scale, value[1], role)
            for low, high in held.intervals
            for scale in SCALES
        )
    return any(_same(figure * scale, value, role) for figure in held.scalars(role) for scale in SCALES)


def _same(stored: float, issued: float, role: str) -> bool:
    """Whether a stored figure, scaled, is the issued one: equal to twelve figures, or, for a table's column, which
    holds its figure rounded to the places it prints, equal at those places."""
    if math.isclose(stored, issued, rel_tol=1e-9, abs_tol=1e-12):
        return True
    if role != "column":
        return False
    places = next((places for places in range(5) if round(issued, places) == issued), None)
    return places is not None and abs(stored - issued) <= 0.5 * 10**-places + 1e-9


@dataclass(frozen=True)
class Gap:
    """A figure the code page states that the input does not hold: its campaign, measure, kind and role."""

    campaign: str
    measure: str
    kind: str
    role: str


#: Each figure the code page states that the input does not yet hold, with what holds it apart.
ALLOWANCE: dict[Gap, str] = {}


def _forms_of_the_code_page(name: str) -> Iterator[PageForms]:
    """Every set of forms the campaign's code page was assembled in, read as a host reads the page."""
    built: list[Any] = []
    patch = pytest.MonkeyPatch()
    for constructor in ("of_campaign", "of_aspects", "of_bundle"):
        original = getattr(PageForms, constructor).__func__

        def spying(cls: type[PageForms], *args: Any, __original: Any = original, **kwargs: Any) -> Any:
            made = __original(cls, *args, **kwargs)
            built.append(made)
            return made

        patch.setattr(PageForms, constructor, classmethod(spying))
    patch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)
    try:
        read = subject(name)
        report_read(read.host, read.campaign_id, read.scope_id, format="markdown")
    finally:
        patch.undo()
    for made in built:
        yield from made.values() if isinstance(made, Mapping) else [made]


def gaps_of(name: str) -> set[Gap]:
    """Every figure campaign ``name``'s code page states that its report input does not hold."""
    read = subject(name)
    held = held_by_measure(report_input(assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)))
    found: set[Gap] = set()
    for forms in _forms_of_the_code_page(name):
        for entry in forms.ledger:
            if not entry.measure or entry.value is None:
                continue
            kind = entry.kind or "measure"
            if not resolves(held.get((entry.measure, kind)), entry.role, entry.value):
                found.add(Gap(name, entry.measure, kind, entry.role))
    return found


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_every_figure_the_code_page_states_the_input_holds(name: str) -> None:
    found = gaps_of(name)
    allowed = {gap for gap in ALLOWANCE if gap.campaign == name}

    assert not found - allowed, (
        f"the code page states figures the input does not hold: {sorted(found - allowed, key=str)}"
    )
    assert not allowed - found, f"the allowance names gaps that are closed: {sorted(allowed - found, key=str)}"
