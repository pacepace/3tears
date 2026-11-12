"""What caps a candidate's standing on the asked aspects, read on kairo's input (r5: control September, candidates
qwen3.8-flash and 1017, one asked aspect with a rate-drop floor, every judge undetermined and insufficient).

Contracts held here (``threetears.evals.analysis.report_decision``, Dominance):

- a floor read ``broken`` on an asked aspect caps a candidate shown better at ``not_shown_no_worse``, and its ``why``
  names the floor and its reading; a floor read ``unclear`` caps nothing;
- an ``improved`` contrast on a judged dimension whose judge is stated uncalibrated does not raise the standing, and
  ``why`` names it as set aside; a calibrated judge's improvement still raises it;
- a candidate capped below the control while it costs more is dominated, and the control dominates: no tradeoff;
- the editor's adoption fact names a recommended candidate a broken floor caps, and an improvement set aside.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from threetears.evals.analysis.analyst.analyses import dominance
from threetears.evals.analysis.report_decision import asked_standing, decision_at, shown_better
from threetears.evals.analysis.report_doc_facts import editor_facts
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report_inputs"
ASPECT = "character adherence (v2)"
FLOOR = "character adherence (v2)/rate_drop/*"
CANDIDATE = "qwen3.8-flash"
ARRIVAL = 'goal_state:call_count("discord.send_message") >= 1 or call_count("music.queue_voice") >= 1'


def _data() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((FIXTURES / "r5" / "input.json").read_text(encoding="utf-8"))
    return loaded


def _improved(data: dict[str, Any], measure: str) -> dict[str, Any]:
    """The candidate's contrast on ``measure`` read improved, on the asked aspect."""
    contrast = next(c for c in data["contrasts"] if c["arm"] == CANDIDATE and c["measure"] == measure)
    contrast.update(aspect=ASPECT, verdict="improved", delta=0.2, interval=[0.1, 0.3])
    return data


def _floor(data: dict[str, Any], status: str, interval: list[float]) -> dict[str, Any]:
    """The candidate's floor read at ``status`` on ``interval``."""
    read = next(f for f in data["floors"] if f["arm"] == CANDIDATE)
    read.update(status=status, interval=interval, worst_drop=-interval[0], reason=None)
    return data


def _calibrated(data: dict[str, Any], dimension: str) -> dict[str, Any]:
    data["judged"][dimension].update(tier="calibrated", calibration="met")
    data["caveats"]["judge_tiers"][dimension] = "calibrated"
    return data


def _standing(data: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    held = asked_standing(ReportInput.model_validate(data))[CANDIDATE]
    return held.standing, held.why


class TestABrokenFloorCaps:
    def test_an_improved_candidate_with_its_floor_held_is_shown_better(self):
        assert _standing(_improved(_data(), ARRIVAL)) == ("shown_better", ())

    def test_an_improved_candidate_whose_floor_reads_broken_is_not_shown_better_and_why_names_the_floor(self):
        standing, why = _standing(_floor(_improved(_data(), ARRIVAL), "broken", [-30.0, -20.0]))

        assert standing == "not_shown_no_worse"
        assert [clause for clause in why if FLOOR in clause] == [
            f"floor {FLOOR} is broken, its whole interval past the floor, which caps it at not shown no worse though "
            "it is shown better"
        ]

    def test_a_floor_read_unclear_caps_nothing(self):
        """An interval straddling the floor's line reads unclear: it warns, and the candidate stays shown better."""
        assert _standing(_floor(_improved(_data(), ARRIVAL), "unclear", [-20.0, -5.0])) == ("shown_better", ())

    def test_a_floor_margin_the_sweep_reads_broken_caps_the_candidate_there(self):
        """The decision at a hypothetical floor margin reads the floor on its interval, and the cap with it: at a
        margin the candidate's interval clears, the cap lifts and no option dominates alone."""
        report_input = ReportInput.model_validate(_floor(_improved(_data(), ARRIVAL), "broken", [-30.0, -20.0]))

        assert decision_at(report_input, floors={FLOOR: 15.0}).dominant == ("September",)
        assert decision_at(report_input, floors={FLOOR: 40.0}) == decision_at(
            ReportInput.model_validate(_improved(_data(), ARRIVAL))
        )
        assert decision_at(report_input, floors={FLOOR: 40.0}).dominant == ()


class TestACappedCandidateThatCostsMore:
    """The candidate costs more than September on every reading of cost; capped below it, it is dominated."""

    def test_uncapped_it_is_a_tradeoff_against_the_control(self):
        assert decision_at(ReportInput.model_validate(_improved(_data(), ARRIVAL))).modes == frozenset({"tradeoff"})

    def test_capped_the_control_dominates_it_and_the_call_is_a_decision(self):
        report_input = ReportInput.model_validate(_floor(_improved(_data(), ARRIVAL), "broken", [-30.0, -20.0]))
        read = dominance(report_input)

        assert decision_at(report_input).modes == frozenset({"decide"})
        assert read.dominant == ["arm:September"]
        assert [(row.standing, row.dominated_by) for row in read.survivors if row.arm == f"arm:{CANDIDATE}"] == [
            ("not_shown_no_worse", ["arm:September", "arm:1017"])
        ]


class TestAnUncalibratedJudgeIsSetAside:
    def test_an_improvement_only_on_an_uncalibrated_judged_dimension_is_not_shown_better(self):
        data = _improved(_data(), "__outcome__")
        report_input = ReportInput.model_validate(data)
        standing, why = _standing(data)

        assert not shown_better(report_input, CANDIDATE)
        assert standing == "shown_no_worse"
        assert why == (
            "its improvement on '__outcome__' is set aside: that dimension's judge is not calibrated, so it does not "
            "raise the standing",
        )
        assert asked_standing(report_input)[CANDIDATE].set_aside == ("__outcome__",)

    def test_a_calibrated_judges_improvement_raises_the_standing(self):
        assert _standing(_calibrated(_improved(_data(), "__outcome__"), "__outcome__")) == ("shown_better", ())

    def test_another_improvement_beside_it_counts_as_usual(self):
        standing, why = _standing(_improved(_improved(_data(), "__outcome__"), ARRIVAL))

        assert standing == "shown_better"
        assert [clause.split(":")[0] for clause in why] == ["its improvement on '__outcome__' is set aside"]


def _adoption(change: Callable[[dict[str, Any]], dict[str, Any]]) -> list[str]:
    """The adoption facts on kairo's converted document recommending the candidate, on its snapshot changed."""
    doc = converted_doc("prototype3/kairo")
    doc["recommends"] = [CANDIDATE]
    held = ReportInput.model_validate(snapshot_of("prototype3/kairo").report_input)
    data = change(held.model_dump(mode="json", by_alias=True, exclude_unset=True))
    facts = editor_facts(ReportDoc.model_validate(doc), ReportInput.model_validate(data))
    return [fact.fact for fact in facts if fact.kind == "adoption"]


class TestTheAdoptionFact:
    def test_a_recommended_candidate_a_broken_floor_caps_is_an_adoption_fact(self):
        facts = _adoption(lambda data: _floor(_improved(data, ARRIVAL), "broken", [-30.0, -20.0]))

        assert ["a floor read broken on an asked aspect caps its standing" in fact for fact in facts] == [True]

    def test_a_recommended_candidate_improved_only_on_an_uncalibrated_judge_names_it_set_aside(self):
        facts = _adoption(lambda data: _floor(_improved(data, "__outcome__"), "unclear", [-20.0, -5.0]))

        assert ["save on '__outcome__', whose judge is not calibrated" in fact for fact in facts] == [True]
