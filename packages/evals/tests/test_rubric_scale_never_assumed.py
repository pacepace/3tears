"""A rubric dimension's scale is read off what was recorded, never assumed (engine item 6).

A pass/fail dimension's values are 1 and 0 and its mean is a pass rate; read as a 1-5 scale, every axis, bar and
reading drawn from it is mislabelled. The scale is recorded twice: on each score (``RubricScore.scale``) and on the run
(``EvalRun.rubric_scales``, captured at launch from its template), which is the only record of a dimension the judge
could not tell on, since that answer carries no score. Contracts held here:

- :func:`~threetears.evals.schema.models.rubric_dim_scale` reads a reserved dual-score axis as 1-5 (it is no template
  dim, and no run records a scale for it), every other dim as recorded, and refuses a dim no scale was recorded for;
- a trial's row shows a dimension the judge could not tell on at the scale its run recorded: a pass/fail one is
  ``pass_fail``, which the row read as ``ordinal`` before;
- the bundle's judged measure and the trial rows agree on that dimension's scale.
"""

from __future__ import annotations

import pytest

from packages.evals.tests.run_facts_support import CASES, aspect_bundle, result_of, run_of
from threetears.evals.schema.models import OUTCOME_DIM_ID, TRANSCRIPT_DIM_ID, RubricScore, rubric_dim_scale

DIM = "answer.correct"


@pytest.mark.parametrize("axis", [OUTCOME_DIM_ID, TRANSCRIPT_DIM_ID])
def test_a_reserved_axis_is_one_to_five_by_construction(axis: str) -> None:
    assert rubric_dim_scale(axis, {}) == "ordinal"


def test_a_template_dim_reads_the_scale_recorded_for_it() -> None:
    assert rubric_dim_scale(DIM, {DIM: "pass_fail"}) == "pass_fail"
    assert rubric_dim_scale(DIM, {DIM: "ordinal"}) == "ordinal"


def test_a_dim_no_scale_was_recorded_for_is_refused_never_read_as_one_to_five() -> None:
    with pytest.raises(ValueError, match="never assumed"):
        rubric_dim_scale(DIM, {})


def test_a_pass_fail_dim_the_judge_could_not_tell_on_is_shown_pass_fail_on_its_trial() -> None:
    base, fast = run_of("base", rubric_scales={DIM: "pass_fail"}), run_of("fast", rubric_scales={DIM: "pass_fail"})
    trials = [
        (
            run,
            [
                result_of(
                    run,
                    index,
                    rubric_scores=(
                        []
                        if index == 0
                        else [RubricScore(dim=DIM, score=index % 2, scale="pass_fail", axis="capability")]
                    ),
                    judge_cannot_tell={DIM: "the reply was cut off"} if index == 0 else {},
                )
                for index in range(CASES)
            ],
        )
        for run in (base, fast)
    ]
    _, bundle, _ = aspect_bundle(trials)
    could_not_tell = [
        judgement
        for row in bundle.trials
        for judgement in row.judge
        if judgement.dim == DIM and judgement.score is None
    ]
    assert len(could_not_tell) == 2
    assert {judgement.scale for judgement in could_not_tell} == {"pass_fail"}
    scored = {judgement.scale for row in bundle.trials for judgement in row.judge if judgement.score is not None}
    assert scored == {"pass_fail"}
    (measure,) = [measure for measure in bundle.judged_measures if measure.name == DIM]
    assert measure.value_range == (0.0, 1.0)
