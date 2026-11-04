"""A unit written right after a figure key that prints it, read by the prose checks every report document's prose goes
through (``threetears.evals.analysis.report_prose_checks.unit_faults``, invariant 1 of the report document's checks).

Re-pointed from the retired plan checks' unit cases, each on the same text: a field of the stored plans the report
planner left (``fixtures/plan_charts/``), edited as the case says, and read against its own input.

- **A unit after a key that prints it is refused**, named with what the key prints: milliseconds, seconds, a percent,
  money's sign or words, a count, the words of a cost's own unit, and a unit outside the package's vocabulary that
  repeats the one the key prints.
- **What passes**: a key spelled bare or as a whole number before its unit, points after a key spelled in points, a
  word that only starts like a unit, and a cost's words after a key that does not print them.
- **Money per reply under one cent** is refused where the page states its costs per 1,000, naming the key that would
  state it so, its other filters kept; keyed per 1,000, a cent or more, or where no cost is stated per 1,000, it passes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.plan_charts_support import PLAN_CHARTS
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_prose_checks import unit_faults

#: The Kairo reference plan's fields the cases edit.
_RECOMMENDATION = ("r5-reference", "decision.slots[0].text")
_WHY_NOT = ("r5-reference", "decision.slots[1].text")
_SECONDS = "{measure.reply_s.September.mean}"
_R2_P95 = "{measure.latency_ms.brisk-s.low.r1.p95|int} ms"


def _field(plan: str, path: str) -> str:
    """One prose field of a stored plan, by its path (``lede``, ``decision.slots[0].text``)."""
    node: Any = json.loads((PLAN_CHARTS / f"{plan}.json").read_text(encoding="utf-8"))
    for step in path.replace("]", "").replace("[", ".").split("."):
        node = node[int(step)] if step.isdigit() else node[step]
    assert isinstance(node, str)
    return node


def _replaced(plan: str, path: str, old: str, new: str) -> str:
    text = _field(plan, path)
    assert old in text, f"{plan} {path} no longer holds {old!r}"
    return text.replace(old, new)


def _data(name: str = "r5") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((dataset.HERE / name / "input.json").read_text(encoding="utf-8"))
    return loaded


def _why(text: str, name: str = "r5", data: dict[str, Any] | None = None) -> str:
    """Every unit fault on ``text`` against input ``name`` (or ``data``), joined; empty where it passes."""
    report_input = ReportInput.model_validate(data) if data is not None else dataset.report_input_of(name)
    return " ".join(why for _, why in unit_faults([("text", text)], report_input))


def _seconds_data() -> dict[str, Any]:
    """r5 with a measure stated in seconds, read by one arm."""
    data = _data()
    data["measures"]["reply_s"] = {"unit": "s", "higher_is_better": False, "per_arm": {"September": {"mean": 3.2}}}
    return data


@pytest.mark.parametrize("after", [" ms", " milliseconds", "ms", " MS"])
def test_a_unit_after_a_key_that_prints_it_is_refused(after: str) -> None:
    text = _replaced("r2-anchor", "lede", _R2_P95, "{measure.latency_ms.brisk-s.low.r1.p95}" + after)
    why = _why(text, "r2")
    assert "right after {measure.latency_ms.brisk-s.low.r1.p95}" in why and "'513 ms'" in why


def test_a_unit_after_a_key_spelled_bare_passes() -> None:
    text = _field("r2-anchor", "lede")
    assert _R2_P95 in text
    assert _why(text, "r2") == ""


@pytest.mark.parametrize("after", [" seconds", " second", " s.", " secs"])
def test_seconds_after_a_seconds_key_are_refused(after: str) -> None:
    assert "right after" in _why(f"The reply takes {_SECONDS}{after}", data=_seconds_data())


@pytest.mark.parametrize("after", [" so it is slow.", " slower", "."])
def test_a_word_that_only_starts_like_a_unit_passes(after: str) -> None:
    assert _why(f"The reply takes {_SECONDS}{after}", data=_seconds_data()) == ""


@pytest.mark.parametrize("after", ["%", " %", " percent", " per cent"])
def test_a_percent_after_a_percent_key_is_refused(after: str) -> None:
    why = _why("{pass_rate.September|pct}" + after + " pass.")
    assert f"writes {after.strip()!r} right after {{pass_rate.September|pct}}" in why


def test_points_after_a_key_spelled_in_points_pass() -> None:
    text = _field(*_WHY_NOT)
    assert "|pts} points" in text
    assert _why(text) == ""


def test_a_units_own_words_after_a_key_that_prints_them_are_refused() -> None:
    text = _replaced(
        *_RECOMMENDATION, "{cost_per_1k.September|usd_per_1k})", "{cost_per_1k.September|usd_per_1k} per 1,000 replies)"
    )
    assert "'per 1,000 replies'" in _why(text)


def test_the_words_after_a_cost_spelled_without_them_pass() -> None:
    text = _replaced(
        *_RECOMMENDATION, "{cost_per_1k.September|usd_per_1k})", "{cost_per_1k.September} per 1,000 replies)"
    )
    assert _why(text) == ""


@pytest.mark.parametrize("after", [" dollars", " USD", " $"])
def test_money_after_a_key_that_prints_its_sign_is_refused(after: str) -> None:
    text = _replaced(*_RECOMMENDATION, "{cost_per_1k.September|usd_per_1k})", "{cost_per_1k.September}" + after + ")")
    assert "right after {cost_per_1k.September}" in _why(text)


def test_a_unit_outside_the_vocabulary_is_refused_where_it_repeats_the_printed_one() -> None:
    text = _field("r2-anchor", "lede") + " It makes {measure.calls_per_task.atlas-m.off.r0.mean} calls a task."
    assert "writes 'calls'" in _why(text, "r2")


def test_the_per_million_words_after_a_key_that_prints_them_are_refused() -> None:
    label = dataset.report_input_of("r3").measures["cost_per_1m"].label
    assert label is not None
    counted = label.split()[-1]
    text = _field("r3-anchor", "lede") + (
        f" Zero-shot costs {{measure.cost_per_1m.atlas-m.zero.mean}} per 1,000,000 {counted}."
    )
    assert f"writes 'per 1,000,000 {counted}'" in _why(text, "r3")


def test_a_count_after_a_key_that_prints_it_is_refused_whatever_it_counts() -> None:
    text = (
        _field("r3-anchor", "lede") + " Zero-shot costs {measure.cost_per_1m.atlas-m.zero.mean} per 1,000,000 replies."
    )
    assert "writes 'per 1,000,000'" in _why(text, "r3")


def test_the_per_million_words_after_a_whole_number_pass() -> None:
    text = _field("r3-anchor", "lede") + (
        " Zero-shot costs {measure.cost_per_1m.atlas-m.zero.mean|int} dollars per 1,000,000 replies."
    )
    assert _why(text, "r3") == ""


def test_money_per_reply_under_a_cent_is_refused_where_costs_read_per_thousand() -> None:
    text = _replaced(
        *_RECOMMENDATION, "({cost_per_1k.September|usd_per_1k})", "({scoreboard.September.reply_cost_usd} a reply)"
    )
    why = _why(text)
    assert "under one cent" in why and "{scoreboard.September.reply_cost_usd|usd_per_1k}" in why


def test_money_per_reply_under_a_cent_keyed_per_thousand_passes() -> None:
    text = _replaced(
        *_RECOMMENDATION, "{cost_per_1k.September|usd_per_1k}", "{scoreboard.September.reply_cost_usd|usd_per_1k}"
    )
    assert _why(text) == ""


def test_a_change_in_money_under_a_cent_keeps_its_other_filters_in_the_key_it_asks_for() -> None:
    why = _why("It costs {contrast.1017.production_replicating_cost.delta|abs|usd} more.")
    assert "{contrast.1017.production_replicating_cost.delta|abs|usd_per_1k}" in why


def test_money_per_reply_of_a_cent_or_more_passes() -> None:
    data = _data()
    data["scoreboard"]["September"]["reply_cost_usd"] = 0.02
    text = _replaced(*_RECOMMENDATION, "{cost_per_1k.September|usd_per_1k}", "{scoreboard.September.reply_cost_usd}")
    assert _why(text, data=data) == ""


def test_money_under_a_cent_passes_where_the_page_states_no_cost_per_thousand() -> None:
    data = _data()
    for row in data["scoreboard"].values():
        row["reply_cost_usd"] = None
    for measure in data["measures"].values():
        measure.pop("merit_axis", None)
    data["measures"]["cost_usd"]["per_arm"]["September"]["mean"] = 0.004
    assert _why("A reply costs {measure.cost_usd.September.mean}.", data=data) == ""
