"""What code decides before the planner writes: caveat ids, immateriality, survivors, dominance and allowed modes.

Contracts held here (``threetears.evals.analysis.report_decision``, GENERATOR.md §1.1, §2.5, §3):

- every entry of a report input's ``caveats`` gets one stable id, with its scope;
- a reading's flip margin is its interval's worst drop (``-lo``), the one implementation figure keys print;
- r5's late margin is immaterial: the flip points (10.79 and 5.815) sit under the declared 15 and the call (keep
  September) holds at every margin; its late floor likewise; a late aspect, control or guardrail stays open; a bar only
  warns, so it is checked; a margin that decides the call is open;
- the survivors, the dominant option and the allowed modes of r1 to r5 are the reference pages' (r1 Atlas-M, r2
  Brisk-S low reasoning one retry, r3 Brisk-S few-shot, r4 a tradeoff, r5 keep September), and each mode rule fires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis.report.figure_keys import flip_margin, resolve
from threetears.evals.analysis.report_decision import (
    allowed_modes,
    caveat_materiality,
    dominant_options,
    report_caveats,
    surviving_options,
)
from threetears.evals.analysis.report_input import ReportInput

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report_plans"
FLOOR = "character adherence (v2)/rate_drop/*"
RUNS = ("01a170ab-d2da-750d-a1e3-f4b43716844d", "01a17121-6e85-732d-ab3a-cf86aed8cdc6")


def _data(name: str = "r5") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / name / "input.json").read_text(encoding="utf-8"))
    return loaded


def _input(name: str = "r5", data: dict[str, Any] | None = None) -> ReportInput:
    return ReportInput.model_validate(data if data is not None else _data(name))


def _unclear_readings(data: dict[str, Any]) -> dict[str, Any]:
    """r5 with no margin and every candidate's asked reading crossing zero, and no floor read: nothing separates."""
    data["aspects"][0]["margin"] = None
    for reading in data["aspects"][0]["readings"]:
        reading["within_interval"] = [-6.0, 2.0]
        reading["margin"] = None
    data["floors"] = []
    return data


def _p95_improved(data: dict[str, Any]) -> dict[str, Any]:
    """An asked latency aspect with no readings, on which a corrected contrast shows qwen3.8-flash's p95 improved."""
    data["aspects"].append({"id": "latency", "name": "latency", "asked": True, "readings": []})
    data["measures"]["research_p95_s"] = {"unit": "s", "higher_is_better": False, "per_arm": {}}
    data["contrasts"].append(
        {"arm": "qwen3.8-flash", "control": "September", "kind": "measure", "measure": "research_p95_s",
         "aspect": "latency", "verdict": "improved", "delta": -0.8, "interval": [-1.2, -0.4], "p_adjusted": 0.001}
    )  # fmt: skip
    return data


def _cost(data: dict[str, Any], arm: str, usd: float) -> dict[str, Any]:
    data["measures"]["production_replicating_cost"]["per_arm"][arm]["mean"] = usd
    return data


class TestCaveatIds:
    def test_every_kairo_caveat_has_its_id_and_scope(self):
        assert [(caveat.id, caveat.reach) for caveat in report_caveats(_input())] == [
            ("late:aspect", "report"),
            ("late:control", "report"),
            ("late:margin", "report"),
            (f"late:floor:{FLOOR}", "report"),
            ("judge_uncalibrated", "report"),
            ("confound:observed:reasoning_ratio", "scoped"),
            ("confound:model", "scoped"),
            ("apparatus:simulator_sent_settings", "method"),
            (f"short_run:{RUNS[0]}", "scoped"),
            (f"short_run:{RUNS[1]}", "scoped"),
            ("latency", "method"),
            ("unreported:input_tokens", "method"),
            ("unreported:llm_calls", "method"),
            ("unreported:output_tokens", "method"),
        ]

    def test_a_confound_of_an_apparatus_dimension_is_that_one_caveat(self):
        ids = [caveat.id for caveat in report_caveats(_input())]
        assert "confound:simulator_sent_settings" not in ids and "apparatus:simulator_sent_settings" in ids

    def test_an_input_with_no_caveats_has_none(self):
        assert report_caveats(_input("r1")) == []

    def test_an_id_is_stable_across_reads(self):
        assert report_caveats(_input()) == report_caveats(_input())


class TestFlipMargin:
    def test_it_is_each_readings_worst_drop(self):
        readings = _input().aspects[0].readings
        assert [flip_margin(reading) for reading in readings] == [10.79, 5.815]

    def test_the_figure_key_prints_the_same_figure(self):
        assert resolve("flip_margin.qwen3.8-flash", _input()).value == 10.79

    def test_it_is_never_below_zero_and_absent_without_an_interval(self):
        data = _data()
        data["aspects"][0]["readings"][0]["within_interval"] = [1.5, 4.0]
        data["aspects"][0]["readings"][1]["within_interval"] = None
        readings = _input(data=data).aspects[0].readings
        assert [flip_margin(reading) for reading in readings] == [0.0, None]


class TestImmateriality:
    def test_kairos_late_margin_is_checked_the_call_holds_at_every_margin(self):
        margin = caveat_materiality(_input())["late:margin"]
        assert margin.immaterial
        assert margin.checked == "The late margin was checked: the call holds at every margin."

    def test_kairos_flip_points_sit_under_the_declared_margin(self):
        report_input = _input()
        flips = [flip_margin(reading) for reading in report_input.aspects[0].readings]
        assert all(flip is not None and flip < 15 for flip in flips)

    def test_kairos_late_floor_is_checked_too(self):
        floor = caveat_materiality(_input())[f"late:floor:{FLOOR}"]
        assert floor.immaterial and floor.checked is not None and "floor" in floor.checked

    @pytest.mark.parametrize("caveat", ["late:aspect", "late:control", "judge_uncalibrated", "latency"])
    def test_what_cannot_be_checked_stays_open(self, caveat: str):
        assert not caveat_materiality(_input())[caveat].immaterial

    def test_a_late_guardrail_margin_stays_open(self):
        data = _data()
        data["caveats"]["late_criteria"].append({"criterion": "guardrail", "name": "tone", "set_at": "25 Oct 2026"})
        assert not caveat_materiality(_input(data=data))["late:guardrail:tone"].immaterial

    def test_a_late_bar_only_warns_so_it_is_checked(self):
        data = _data()
        data["caveats"]["late_criteria"].append({"criterion": "bar", "name": "passed", "set_at": "25 Oct 2026"})
        bar = caveat_materiality(_input(data=data))["late:bar:passed"]
        assert bar.immaterial and bar.checked is not None

    def test_a_margin_the_call_rests_on_is_open(self):
        """With qwen3.8-flash the cheapest, it dominates only while within the margin: below its flip margin of
        10.79 it is no longer shown no worse, so the call changes and the margin is material."""
        report_input = _input(data=_cost(_data(), "qwen3.8-flash", 0.0015))
        assert dominant_options(report_input) == ["qwen3.8-flash"]
        assert not caveat_materiality(report_input)["late:margin"].immaterial
        assert not caveat_materiality(report_input)[f"late:floor:{FLOOR}"].immaterial

    def test_a_margin_read_on_no_interval_cannot_be_checked(self):
        data = _data()
        data["aspects"][0]["readings"][1]["within_interval"] = None
        assert not caveat_materiality(_input(data=data))["late:margin"].immaterial


class TestSurvivorsAndDominance:
    @pytest.mark.parametrize(
        ("name", "survivors", "dominant"),
        [
            ("r1", ["atlas-m"], ["atlas-m"]),
            ("r2", ["brisk-s.low.r1"], ["brisk-s.low.r1"]),
            ("r3", ["brisk-s.few"], ["brisk-s.few"]),
            ("r4", ["prod", "f64"], []),
            ("r5", ["September", "qwen3.8-flash", "1017"], ["September"]),
        ],
    )
    def test_each_reference_survives_and_dominates_as_its_page_says(
        self, name: str, survivors: list[str], dominant: list[str]
    ):
        report_input = _input(name)
        assert (surviving_options(report_input), dominant_options(report_input)) == (survivors, dominant)

    def test_a_standing_guardrail_removes_the_arms_that_fail_it(self):
        """r4: f16 misses the music-facts guardrail and cor the speculation one."""
        assert set(surviving_options(_input("r4"))) == {"prod", "f64"}

    def test_a_breached_guardrail_removes_its_arm(self):
        data = _data()
        data["guardrails"] = [{"arm": "1017", "decision": "breached"}]
        assert surviving_options(_input(data=data)) == ["September", "qwen3.8-flash"]

    def test_a_gate_with_no_figure_for_an_arm_is_not_passed(self):
        data = _data("r2")
        data["measures"]["latency_ms"]["per_arm"]["brisk-s.low.r1"]["p95"] = None
        assert surviving_options(_input(data=data)) == []

    def test_a_candidate_cheaper_and_shown_no_worse_dominates_the_control(self):
        assert dominant_options(_input(data=_cost(_data(), "1017", 0.0015))) == ["1017"]

    def test_a_choice_arm_behind_on_a_pair_is_not_at_least_as_good(self):
        data = _data()
        data["arms"][0]["role"] = "challenger"
        data["aspects"][0]["pairs"] = [{"arms": ["September", "1017"], "change": -3.0, "ahead": "1017"}]
        assert "September" not in dominant_options(_input(data=data))


class TestAllowedModes:
    @pytest.mark.parametrize(
        ("name", "modes"),
        [("r1", {"decide"}), ("r2", {"decide"}), ("r3", {"decide"}), ("r4", {"tradeoff"}), ("r5", {"decide"})],
    )
    def test_each_reference_allows_its_pages_mode(self, name: str, modes: set[str]):
        assert allowed_modes(_input(name)) == modes

    def test_a_declared_ranking_rule_turns_a_tradeoff_into_a_decision(self):
        data = _data("r4")
        data["contrast_families"][0]["merit_axes"] = ["quality", "cost"]
        assert allowed_modes(_input(data=data)) == {"decide"}

    def test_no_survivor_is_a_no_call(self):
        data = _data()
        data["criteria"] = [
            {"id": "tone", "kind": "guardrail", "measure": "persona.holds_character",
             "direction": "higher_is_better", "threshold": 0.95},
        ]  # fmt: skip
        assert (surviving_options(_input(data=data)), allowed_modes(_input(data=data))) == ([], {"no_call"})

    def test_every_candidate_unseparated_is_a_no_call(self):
        data = _data()
        for reading in data["aspects"][0]["readings"]:
            reading["within_interval"] = [-16.0, 1.0]
        assert allowed_modes(_input(data=data)) == {"no_call"}

    def test_a_separated_contrast_on_an_asked_latency_aspect_separates_its_candidate(self):
        """A question a measure answers: the pass-rate readings are unclear with no margin, and a corrected contrast on
        the asked latency aspect shows qwen3.8-flash improved, so it is separated and (cheapest too) decides."""
        data = _p95_improved(_unclear_readings(_data()))
        _cost(data, "qwen3.8-flash", 0.0015)
        assert allowed_modes(_input(data=data)) == {"decide"}
        assert dominant_options(_input(data=data)) == ["qwen3.8-flash"]

    def test_a_latency_question_ranked_by_its_merit_priority_decides_by_the_rule(self):
        """Discodon's shape (reproduced at eea04a131, where every decide plan was refused on `mode`): the candidate is
        shown better on p95 and costs more, so nothing dominates, and the declared priority (latency, then cost)
        picks."""
        data = _p95_improved(_unclear_readings(_data()))
        data["contrast_families"][0]["merit_axes"] = ["latency", "cost"]
        assert (allowed_modes(_input(data=data)), dominant_options(_input(data=data))) == ({"decide"}, [])

    def test_a_regressed_candidate_counts_as_separated(self):
        by_reading = _unclear_readings(_data())
        by_reading["aspects"][0]["readings"][0]["within_interval"] = [-8.0, -2.0]
        assert allowed_modes(_input(data=by_reading)) == {"decide"}
        by_contrast = _unclear_readings(_data())
        by_contrast["contrasts"].append(
            {"arm": "qwen3.8-flash", "control": "September", "kind": "judged", "measure": "persona.holds_character",
             "aspect": "character adherence (v2)", "verdict": "regressed", "delta": -5.0, "interval": [-8.0, -2.0],
             "p_adjusted": 0.01}
        )  # fmt: skip
        assert allowed_modes(_input(data=by_contrast)) == {"decide"}

    @pytest.mark.parametrize(("change", "ahead"), [(3.0, "September"), (0.4, None)])
    def test_a_choices_pass_rate_pair_is_read_as_before(self, change: float, ahead: str | None):
        """A choice has no control, so separability, which reads a change from one, never forces it to no call."""
        data = _data()
        data["arms"][0]["role"] = "challenger"
        data["aspects"][0]["pairs"] = [{"arms": ["September", "1017"], "change": change, "ahead": ahead}]
        assert (allowed_modes(_input(data=data)), dominant_options(_input(data=data))) == ({"decide"}, ["September"])

    @pytest.mark.parametrize("verdict", ["not_separated", "untested"])
    def test_an_unclear_reading_with_no_separated_contrast_is_still_a_no_call(self, verdict: str):
        data = _unclear_readings(_data())
        data["contrasts"] = [
            {**contrast, "aspect": "character adherence (v2)", "verdict": verdict}
            for contrast in data["contrasts"]
            if contrast["arm"] != "September"
        ]
        assert allowed_modes(_input(data=data)) == {"no_call"}

    def test_one_arm_measured_is_a_no_call(self):
        data = _data()
        data["arms"] = data["arms"][:1]
        assert allowed_modes(_input(data=data)) == {"no_call"}

    def test_two_survivors_neither_dominating_is_a_tradeoff(self):
        data = _cost(_data(), "1017", 0.0015)
        data["floors"] = [floor for floor in data["floors"] if floor["arm"] != "1017"]
        assert (dominant_options(_input(data=data)), allowed_modes(_input(data=data))) == ([], {"tradeoff"})
