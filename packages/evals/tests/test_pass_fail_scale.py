"""A pass/fail rubric criterion, end to end: authored, asked, answered, stored, aggregated, described.

New criteria are pass/fail rather than 1-5. A pass is stored as 1 and a fail as 0, so a
dimension's mean is its pass rate, and every consumer reads the score through the one predicate
pair on :class:`~threetears.evals.schema.models.RubricScore` rather than re-deriving 1-5 arithmetic. Each
rule below is asserted beside its 1-5 counterpart, so a check that ignored the scale cannot pass.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.reporting import PivotError, compute_pivot, project_score_records
from threetears.evals.kernel.declaration import resolve_bar_name
from threetears.evals.kernel.host import MeasureRegistry
from threetears.evals.schema.models import SCALES, JudgedArtifact, JudgeEvidence, RubricDim, RubricScale, RubricScore
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.kernel.scoring import compute_dimension_summary, compute_pass_hat_k, result_composite
from threetears.evals.run.judge import SCALE_READERS, run_judge_llm
from threetears.evals.run.judge_service import JudgeContext, JudgeService
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The host the projections here read their vocabulary through. Nothing here is about its levers.
_HOST = toyhost_profile()


_PF = [{"name": "arc", "scale": "pass_fail"}]
_ORD = [{"name": "arc"}]


def _answer(value: Any) -> str:
    return json.dumps({"reasoning": "r", "criteria_scores": {"arc": value}})


class TestTheAuthoredCriterion:
    def test_a_pass_fail_guide_describes_pass_and_fail(self):
        dim = RubricDim(name="x.arc", description="d", scale="pass_fail", scoring_guide={"pass": "p", "fail": "f"})
        assert dim.scale == "pass_fail"

    def test_a_guide_with_levels_its_scale_lacks_is_refused(self):
        with pytest.raises(ValidationError, match="scoring guide has levels"):
            RubricDim(name="x.arc", description="d", scale="pass_fail", scoring_guide={"3": "middling"})
        with pytest.raises(ValidationError, match="scoring guide has levels"):
            RubricDim(name="x.arc", description="d", scoring_guide={"pass": "p"}, scale="ordinal")

    def test_a_criterion_with_no_scale_is_refused_rather_than_read_as_ordinal(self):
        """A criterion states its scale: none is assumed, for a criterion or for the score it was judged on."""
        with pytest.raises(ValidationError, match="scale"):
            RubricDim.model_validate({"name": "x.arc", "description": "d", "scoring_guide": {"1": "a"}})
        with pytest.raises(ValidationError, match="scale"):
            RubricScore.model_validate({"dim": "x.arc", "score": 3})


class TestTheStoredScore:
    @pytest.mark.parametrize(("scale", "score"), [("ordinal", 0), ("ordinal", 6), ("pass_fail", 2), ("pass_fail", -1)])
    def test_a_score_off_its_scale_is_refused(self, scale, score):
        with pytest.raises(ValidationError, match="not on the"):
            RubricScore(dim="x.arc", scale=scale, score=score)

    def test_the_label_names_a_pass_or_fail_and_leaves_a_level_as_its_number(self):
        """`arc=0` reads as the lowest score, not as a judged fail."""
        assert RubricScore(dim="x.a", scale="pass_fail", score=0).label == "fail"
        assert RubricScore(dim="x.a", scale="pass_fail", score=1).label == "pass"
        assert RubricScore(dim="x.a", score=4, scale="ordinal").label == "4"

    def test_normalized_and_the_bar_follow_the_scale(self):
        assert RubricScore(dim="x.a", scale="pass_fail", score=1).normalized == 1.0
        assert RubricScore(dim="x.a", scale="pass_fail", score=0).normalized == 0.0
        assert RubricScore(dim="x.a", score=5, scale="ordinal").normalized == 1.0
        assert RubricScore(dim="x.a", score=1, scale="ordinal").normalized == 0.0
        # The ordinal threshold says nothing about a pass/fail criterion: its bar is the pass.
        assert RubricScore(dim="x.a", scale="pass_fail", score=1).clears(5)
        assert not RubricScore(dim="x.a", scale="pass_fail", score=0).clears(1)
        assert RubricScore(dim="x.a", score=3, scale="ordinal").clears(3) and not RubricScore(
            dim="x.a", score=2, scale="ordinal"
        ).clears(3)


class _Replying:
    """A judge client that answers every call with one fixed reply."""

    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        self.calls += 1
        return SimpleNamespace(
            served_model=None, stop_reason="end_turn", content=self.content, input_tokens=1, output_tokens=1,
            cost_usd=0.0, model="judge-fake",
        )  # fmt: skip


async def _judged(value: Any, criteria: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """What the judge entry point makes of one reply scoring ``arc`` as ``value``."""
    parsed = await run_judge_llm(_Replying(_answer(value)), "s", "u", criteria, "Judge", "tc-1", **kwargs)
    assert parsed is not None
    return parsed


class TestTheJudgesAnswer:
    async def test_pass_and_fail_become_one_and_zero(self):
        assert (await _judged("pass", _PF))["criteria_ordinal_scores"] == {"arc": 1}
        assert (await _judged("fail", _PF))["criteria_ordinal_scores"] == {"arc": 0}

    @pytest.mark.parametrize("value", [4, 1, 0, True, "PASS", "yes", None, ["pass"]])
    async def test_anything_else_on_a_pass_fail_criterion_is_refused(self, value):
        with pytest.raises(ValueError, match="pass/fail"):
            SCALE_READERS["pass_fail"]("arc", value)
        parsed = await _judged(value, _PF)
        assert "criteria_scores" not in parsed and parsed["error"].startswith("Failed to parse")

    async def test_a_word_on_an_ordinal_criterion_is_refused(self):
        with pytest.raises(ValueError, match="not an integer"):
            SCALE_READERS["ordinal"]("arc", "pass")
        parsed = await _judged("pass", _ORD)
        assert "criteria_scores" not in parsed and parsed["error"].startswith("Failed to parse")

    async def test_cannot_tell_is_still_offered_on_pass_fail(self):
        parsed = await _judged("cannot_tell", _PF, cannot_tell_offered=True)
        assert parsed["criteria_cannot_tell"] == ["arc"] and parsed["criteria_scores"] == {}

    async def test_the_normalised_score_is_the_answer_itself(self):
        parsed = await _judged("pass", _PF)
        assert parsed["criteria_scores"] == {"arc": 1.0}
        assert parsed["criteria_ordinal_scores"] == {"arc": 1}


class _Judge:
    def __init__(self, answer: Any):
        self.answer = answer
        self.systems: list[str] = []

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        self.systems.append(system)
        return SimpleNamespace(
            served_model=None, stop_reason="end_turn", content=_answer(self.answer).replace('"arc"', '"x.arc"'), input_tokens=1,
            output_tokens=1, cost_usd=0.0, model="judge-fake",
        )  # fmt: skip


def _context() -> JudgeContext:
    return JudgeContext(
        case_id="tc-1",
        intent="i",
        variation={},
        goal_outcomes=[],
        judged_artifact=JudgedArtifact.TRANSCRIPT,
        judge_evidence=JudgeEvidence(subject="E", case_material="the scenario", artifact="Actor (asker): hi"),
    )


class TestTheJudgeService:
    async def test_a_pass_fail_dimension_is_asked_for_pass_or_fail_and_stored_with_its_scale(self):
        judge = _Judge("pass")
        dim = RubricDim(name="x.arc", description="d", scale="pass_fail", scoring_guide={"pass": "P", "fail": "F"})

        outcome = await JudgeService(
            client_factory=lambda m, t: judge, failure_describer=withhold_failure_detail
        ).score_dimension(dim, _context())

        assert outcome.score == RubricScore(dim="x.arc", scale="pass_fail", axis="capability", score=1, reasoning="r")
        (system,) = judge.systems
        assert "Answer pass or fail." in system and "1 (worst) to 5 (best)" not in system
        assert "  pass: P" in system and "  fail: F" in system

    async def test_an_ordinal_dimension_is_still_asked_for_one_to_five(self):
        judge = _Judge(4)
        outcome = await JudgeService(
            client_factory=lambda m, t: judge, failure_describer=withhold_failure_detail
        ).score_dimension(RubricDim(name="x.arc", description="d", scale="ordinal"), _context())

        assert outcome.score == RubricScore(dim="x.arc", score=4, reasoning="r", scale="ordinal", axis="capability")
        assert "1 (worst) to 5 (best)" in judge.systems[0]

    async def test_a_number_given_to_a_pass_fail_dimension_scores_nothing(self):
        judge = _Judge(4)
        dim = RubricDim(name="x.arc", description="d", scale="pass_fail")

        outcome = await JudgeService(
            client_factory=lambda m, t: judge, failure_describer=withhold_failure_detail
        ).score_dimension(dim, _context())

        assert outcome.score is None


def _result(**overrides: Any):
    return make_eval_result(goal_state_outcomes=[], **overrides)


class TestAggregation:
    def test_the_composite_reads_a_pass_as_one_and_a_fail_as_zero(self):
        mixed = _result(
            rubric_scores=[
                RubricScore(dim="x.a", scale="pass_fail", score=1),
                RubricScore(dim="x.b", scale="pass_fail", score=0),
            ]
        )
        assert result_composite(mixed) == 0.5

    def test_pass_k_passes_a_pass_whatever_the_threshold_and_fails_a_fail(self):
        passed = _result(rubric_scores=[RubricScore(dim="x.a", scale="pass_fail", score=1)])
        failed = _result(test_case_id="tc-2", rubric_scores=[RubricScore(dim="x.a", scale="pass_fail", score=0)])

        [row] = compute_pass_hat_k([passed, failed], rubric_threshold=5).values()

        assert row["n_test_cases"] == 2 and row["pass_hat_k"] == 0.5

    def test_a_dimension_summary_names_its_scale_and_its_mean_is_the_pass_rate(self):
        results = [
            _result(test_case_id=f"tc-{i}", rubric_scores=[RubricScore(dim="x.a", scale="pass_fail", score=s)])
            for i, s in enumerate([1, 1, 0, 1])
        ]
        summary = compute_dimension_summary(results)[("sonnet", "run-1", "x.a")]
        assert summary["scale"] == "pass_fail" and summary["mean_score"] == 0.75

    def test_one_dimension_judged_on_two_scales_is_refused(self):
        results = [
            _result(rubric_scores=[RubricScore(dim="x.a", scale="pass_fail", score=1)]),
            _result(test_case_id="tc-2", rubric_scores=[RubricScore(dim="x.a", score=4, scale="ordinal")]),
        ]
        with pytest.raises(ValueError, match="more than one scale"):
            compute_dimension_summary(results)


class TestTheProjection:
    def _project(self, results, **run_overrides):
        """Project through the toy host: the projection reads the installed host's levers."""
        run = make_eval_run(id="run-1", **run_overrides)
        profile = toyhost_profile()
        return project_score_records([run], results, profile=profile, archived_run_ids=None).records

    def test_a_score_row_carries_its_scale(self):
        records = self._project([_result(rubric_scores=[RubricScore(dim="x.a", scale="pass_fail", score=0)])])
        [row] = [r for r in records if r.metric == "score"]
        assert (row.rubric_dim, row.rubric_scale, row.value) == ("x.a", "pass_fail", 0.0)

    def test_a_cannot_tell_row_takes_its_scale_from_what_the_run_recorded(self):
        records = self._project(
            [_result(rubric_scores=[], judge_cannot_tell={"x.a": "no evidence"})], rubric_scales={"x.a": "pass_fail"}
        )
        [row] = [r for r in records if r.metric == "score"]
        assert (row.rubric_dim, row.rubric_scale, row.value) == ("x.a", "pass_fail", None)

    def test_a_run_states_its_scales_and_one_missing_a_dim_is_refused(self):
        """No scale is assumed: a run records every dim's, and one it did not record is refused by name."""
        with pytest.raises(ValidationError, match="rubric_scales"):
            make_eval_run(rubric_scales=None)
        assert make_eval_run(rubric_scales={"x.a": "pass_fail"}).rubric_scale("x.a") == "pass_fail"
        with pytest.raises(ValueError, match="recorded no scale"):
            make_eval_run(rubric_scales={"x.b": "ordinal"}).rubric_scale("x.a")

    def test_a_pivot_cell_pooling_both_scales_is_refused_and_one_per_dimension_is_not(self):
        results = [
            _result(
                rubric_scores=[
                    RubricScore(dim="x.a", scale="pass_fail", score=1),
                    RubricScore(dim="x.b", score=4, scale="ordinal"),
                ]
            ),
        ]
        records = self._project(results)
        with pytest.raises(PivotError, match="different scales"):
            compute_pivot(records, row_factor="model", column_factor="run_id", metric="score", profile=_HOST)
        table = compute_pivot(records, row_factor="rubric_dim", column_factor="model", metric="score", profile=_HOST)
        assert table is not None

    def test_a_pivot_takes_its_measure_range_from_the_one_scale_its_rows_share(self):
        """The catalogue range spans every scale; a table holding one reads on that scale's own."""
        records = self._project(
            [
                _result(
                    rubric_scores=[
                        RubricScore(dim="x.a", scale="pass_fail", score=1),
                        RubricScore(dim="x.b", score=4, scale="ordinal"),
                    ]
                )
            ]
        )

        passed = compute_pivot(
            records,
            row_factor="rubric_dim",
            column_factor="model",
            metric="score",
            filters={"rubric_dim": "x.a"},
            profile=_HOST,
        )
        leveled = compute_pivot(
            records,
            row_factor="rubric_dim",
            column_factor="model",
            metric="score",
            filters={"rubric_dim": "x.b"},
            profile=_HOST,
        )
        both = compute_pivot(records, row_factor="rubric_dim", column_factor="model", metric="score", profile=_HOST)

        assert passed.measure.value_range == SCALES["pass_fail"].value_range
        assert leveled.measure.value_range == SCALES["ordinal"].value_range
        assert both.measure.value_range == (0.0, 5.0)


class TestDescriptors:
    def test_a_bar_on_a_pass_fail_dimension_is_described_on_zero_to_one(self):
        resolved = resolve_bar_name(
            "x.arc", rubric_dimensions={"x.arc": "pass_fail"}, goal_state_checks=(), measures=MeasureRegistry([])
        )
        assert resolved.descriptor.value_range == (0.0, 1.0)
        ordinal = resolve_bar_name(
            "x.arc", rubric_dimensions={"x.arc": "ordinal"}, goal_state_checks=(), measures=MeasureRegistry([])
        )
        assert ordinal.descriptor.value_range == (1.0, 5.0)


class TestEveryScaleIsKnownEverywhere:
    def test_each_per_scale_table_covers_exactly_the_scales(self):
        """A scale added to the type and forgotten in one table must fail here, not fall into another's arithmetic."""
        scales = set(get_args(RubricScale))
        assert set(SCALES) == scales
        assert set(SCALE_READERS) == scales

    def test_the_public_scale_table_cannot_be_written(self):
        """SCALES is exported, so a host writing into it would rewrite every other host's arithmetic in the process."""
        with pytest.raises(TypeError):
            SCALES["ordinal"] = SCALES["pass_fail"]  # type: ignore[index]
        with pytest.raises(TypeError):
            SCALES["pass_fail"].labels[0] = "zero"  # type: ignore[index]

    @pytest.mark.parametrize("scale", sorted(get_args(RubricScale)))
    async def test_every_scale_is_asked_for_in_its_own_words(self, scale):
        """The judge service words each scale's question; a scale it has no wording for never reaches the judge."""
        judge = _Judge("pass" if scale == "pass_fail" else 4)

        await JudgeService(
            client_factory=lambda m, t: judge, failure_describer=withhold_failure_detail
        ).score_dimension(RubricDim(name="x.arc", description="d", scale=scale), _context())

        (system,) = judge.systems
        wordings = {"ordinal": "1 (worst) to 5 (best)", "pass_fail": "Answer pass or fail."}
        assert [word for word in wordings.values() if word in system] == [wordings[scale]]
