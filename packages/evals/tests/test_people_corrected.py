"""The ``corrected`` check of a judged contrast, and the cell-mean estimate read from random samples only (judges §5.3).

The arithmetic and every outcome of the check are pinned here; whether the interval holds its level, and where the
decision floor sits, is ``test_simulated_people_corrected.py``'s. A contrast has one verdict and one ``corrected``
field, whoever it was checked against; every outcome that is not ``met`` carries a reason of the shared set, and no
field holds a sentence for one.
"""

from __future__ import annotations

import math
import typing
from collections.abc import Sequence

import pytest

from threetears.evals.analysis.blind_read import draw_blind_read
from threetears.evals.analysis.bundle.cell_reads import judged_measures
from threetears.evals.analysis.bundle.comparisons import judged_contrasts_corrected
from threetears.evals.analysis.bundle.schema import (
    ComparedCell,
    ComparisonFamily,
    FamilyComparison,
    MultipleComparisons,
)
from threetears.evals.analysis.checks import Corrected
from threetears.evals.analysis.people_corrected import (
    PPI_DECISION_MIN_LABELLED_CASES,
    PPI_REPORT_MIN_LABELLED_CASES,
    PairLabels,
    corrected_reading,
    decides,
    pair_labels,
    prediction_powered_difference,
)
from threetears.evals.analysis.report_input import Contrast
from threetears.evals.analysis.reporting import ScoreRecord
from threetears.evals.analysis.stats import PPI_MIN_LABELLED_RESULTS, t_critical_two_sided
from threetears.evals.kernel.result_condition import ResultOutcome
from threetears.evals.schema import EvalResult, JudgeEvidence, RubricDim, RubricScore, label_key_of
from threetears.evals.schema.blind_read import BlindAnswer, BlindArm, BlindRead, BlindVerdict
from threetears.evals.schema.reading_states import REASONS_OF, StateOf
from packages.evals.tests.factories import make_eval_result

DIM = RubricDim(name="answer.good", description="The answer is good.", scale="pass_fail")
RIG = "rig"


# --- the estimator --------------------------------------------------------------------------------------------------


class TestTheEstimator:
    def test_no_label_gives_nothing_to_combine(self) -> None:
        assert prediction_powered_difference({"a": 1.0, "b": 0.0}, {}, confidence=0.95) is None

    def test_it_is_the_judge_s_mean_difference_plus_the_mean_of_reader_minus_judge(self) -> None:
        differences = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0, "e": 1.0}
        labelled = {"a": (0.0, 1.0), "b": (0.0, 0.0), "c": (1.0, 1.0)}
        figures = prediction_powered_difference(differences, labelled, confidence=0.95)
        assert figures is not None
        assert figures.judge_difference == pytest.approx(0.6)
        assert figures.rectifier == pytest.approx(-1.0 / 3.0)
        assert figures.estimate == pytest.approx(0.6 - 1.0 / 3.0)
        assert figures.lean == pytest.approx(1.0 / 3.0)
        assert (figures.cases, figures.labelled_cases, figures.identical_cases) == (5, 3, 0)

    def test_its_variance_is_the_cell_mean_estimate_s_linearisation_over_cases(self) -> None:
        differences = {"a": 1.0, "b": 0.0, "c": 1.0, "d": -1.0, "e": 0.0, "f": 1.0}
        labelled = {"a": (0.0, 1.0), "b": (1.0, 0.0), "c": (1.0, 1.0), "d": (0.0, -1.0)}
        figures = prediction_powered_difference(differences, labelled, confidence=0.9)
        assert figures is not None and figures.sem is not None and figures.interval is not None
        f_bar = sum(differences.values()) / 6
        gaps = {case: reader - judge for case, (reader, judge) in labelled.items()}
        r_bar = sum(gaps.values()) / 4
        residuals = [
            (f - f_bar) / 6 + ((gaps[case] - r_bar) / 4 if case in gaps else 0.0) for case, f in differences.items()
        ]
        assert figures.sem == pytest.approx(math.sqrt(4 / 3 * sum(r * r for r in residuals)))
        half = t_critical_two_sided(0.9, 3) * figures.sem
        assert figures.interval == pytest.approx((figures.estimate - half, figures.estimate + half))

    def test_identical_cases_weigh_in_at_zero_and_their_share(self) -> None:
        plain = prediction_powered_difference(
            {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}, {"a": (1.0, 1.0), "b": (0.0, 0.0)}, confidence=0.95
        )
        stratified = prediction_powered_difference(
            {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0, "x": 1.0, "y": -1.0, "z": 1.0, "w": 0.0},
            {"a": (1.0, 1.0), "b": (0.0, 0.0)},
            identical=["x", "y", "z", "w"],
            confidence=0.95,
        )
        assert plain is not None and stratified is not None and plain.sem is not None and stratified.sem is not None
        assert stratified.estimate == pytest.approx(plain.estimate / 2)
        assert stratified.sem == pytest.approx(plain.sem / 2)
        assert stratified.identical_cases == 4

    def test_a_label_outside_the_population_or_on_an_identical_case_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside the population"):
            prediction_powered_difference({"a": 1.0}, {"b": (1.0, 1.0)}, confidence=0.95)
        with pytest.raises(ValueError, match="identical case"):
            prediction_powered_difference({"a": 1.0, "b": 0.0}, {"a": (0.0, 0.0)}, identical=["a"], confidence=0.95)


# --- the check's outcomes -------------------------------------------------------------------------------------------


def _labels(
    pairs: dict[str, tuple[float, float]],
    *,
    against: typing.Literal["people", "model"] = "people",
    sample: typing.Literal["random", "selected", "none"] = "random",
    skipped: int = 0,
    population: Sequence[str] | None = None,
    read_id: str = "read-1",
) -> PairLabels:
    return PairLabels(
        against=against,
        against_id=None if against == "people" else "sim/reviewer",
        read_id=read_id,
        sample=sample,
        answered=len(pairs) + skipped,
        skipped=skipped,
        population=frozenset(population if population is not None else pairs),
        identical=frozenset(),
        pairs=pairs,
    )


def _spread(n: int) -> dict[str, tuple[float, float]]:
    """``n`` cases whose reader differences alternate +1 and -1: a mean of zero with a wide spread."""
    return {f"c{i:03d}": (1.0 if i % 2 else -1.0, 0.0) for i in range(n)}


def _read(pairs: dict[str, tuple[float, float]], **kwargs: typing.Any) -> Corrected:
    differences = {case: judge for case, (_, judge) in pairs.items()}
    corrected = corrected_reading([_labels(pairs, **kwargs)], differences, direction=1, confidence=0.95)
    assert corrected is not None
    return corrected


N = PPI_DECISION_MIN_LABELLED_CASES

#: Each outcome, from labels built to reach it: ``(pairs, the expected result, its reason, the field's state)``.
OUTCOMES: dict[str, tuple[dict[str, tuple[float, float]], str | None, str | None, str | None]] = {
    "met": ({f"c{i:03d}": (float(i % 2), float(i % 2)) for i in range(N)}, "met", None, None),
    "people_disagree": ({f"c{i:03d}": (-1.0, float(i % 2)) for i in range(N)}, "not_met", "people_disagree", None),
    # The reader sees nothing on average; the judge reads every case a point higher than the reader.
    "instrument_leans": (
        {case: (reader, reader + 1.0) for case, (reader, _) in _spread(N).items()},
        "not_met",
        "instrument_leans",
        None,
    ),
    "undecided": ({case: (reader, reader) for case, (reader, _) in _spread(N).items()}, "undecided", None, None),
    "insufficient": (
        {f"c{i:03d}": (float(i % 2), float(i % 2)) for i in range(PPI_REPORT_MIN_LABELLED_CASES)},
        "insufficient",
        "too_few",
        None,
    ),
    "too_few": ({f"c{i:03d}": (1.0, 1.0) for i in range(PPI_REPORT_MIN_LABELLED_CASES - 1)}, None, None, "too_few"),
}


class TestTheOutcomes:
    @pytest.mark.parametrize("name", list(OUTCOMES))
    def test_each_outcome_is_a_result_and_a_shared_reason_never_a_verdict(self, name: str) -> None:
        pairs, result, reason, state = OUTCOMES[name]
        corrected = _read(pairs)
        assert corrected.result == result
        assert corrected.reason == reason
        assert (corrected.state.reason if corrected.state else None) == state
        if corrected.result not in (None, "met", "undecided"):
            assert corrected.reason in {member for reasons in REASONS_OF.values() for member in reasons}
        if corrected.state is not None:
            assert corrected.state.state == "not_available" and corrected.result is None and corrected.interval is None

    def test_figures_are_stated_from_twenty_labelled_cases_and_decide_from_the_floor(self) -> None:
        assert not decides(_read(OUTCOMES["too_few"][0]))
        assert not decides(_read(OUTCOMES["insufficient"][0]))
        assert decides(_read(OUTCOMES["met"][0]))
        assert decides(_read(OUTCOMES["people_disagree"][0]))

    def test_a_regression_is_confirmed_in_its_own_direction(self) -> None:
        pairs = {f"c{i:03d}": (-float(i % 2), -float(i % 2)) for i in range(N)}
        differences = {case: judge for case, (_, judge) in pairs.items()}
        down = corrected_reading([_labels(pairs)], differences, direction=-1, confidence=0.95)
        up = corrected_reading([_labels(pairs)], differences, direction=1, confidence=0.95)
        assert down is not None and up is not None
        assert (down.result, up.result, up.reason) == ("met", "not_met", "people_disagree")

    def test_a_sample_not_drawn_at_random_states_no_figure(self) -> None:
        corrected = _read(OUTCOMES["met"][0], sample="selected")
        assert corrected.state == StateOf(state="not_available", reason="not_drawn_at_random")
        assert (corrected.result, corrected.estimate) == (None, None)

    def test_more_than_a_tenth_skipped_states_no_figure(self) -> None:
        pairs = OUTCOMES["met"][0]
        assert _read(pairs, skipped=len(pairs) // 8).state == StateOf(
            state="not_available", reason="skipped_past_limit"
        )
        assert _read(pairs, skipped=len(pairs) // 10).state is None

    def test_a_contrast_over_an_unrecorded_coordinate_states_none(self) -> None:
        pairs = OUTCOMES["met"][0]
        unrecorded = StateOf(state="withheld", reason="identity_unrecorded", coordinate="served_model")
        corrected = corrected_reading(
            [_labels(pairs)],
            {case: judge for case, (_, judge) in pairs.items()},
            direction=1,
            confidence=0.95,
            contrast_state=unrecorded,
        )
        assert corrected is not None
        assert corrected.state == StateOf(
            state="not_available", reason="identity_unrecorded", coordinate="served_model"
        )

    def test_a_person_s_check_is_the_field_and_a_model_s_rides_beside_it(self) -> None:
        pairs = OUTCOMES["met"][0]
        differences = {case: judge for case, (_, judge) in pairs.items()}
        model = _labels(pairs, against="model", read_id="read-m")
        person = _labels(dict(list(pairs.items())[:30]), read_id="read-p")
        chosen = corrected_reading([model, person], differences, direction=1, confidence=0.95)
        assert chosen is not None and (chosen.against, chosen.read_id, chosen.result) == (
            "people",
            "read-p",
            "insufficient",
        )
        assert chosen.model_check is not None
        assert (chosen.model_check.against, chosen.model_check.result) == ("model", "met")
        assert not decides(chosen) and not decides(chosen.model_check)
        # A person's sample that was not drawn at random is still the field: a model never stands in for it.
        unusable = _labels(pairs, sample="selected", read_id="read-s")
        chosen = corrected_reading([unusable, model], differences, direction=1, confidence=0.95)
        assert chosen is not None and chosen.against == "people"
        assert chosen.state == StateOf(state="not_available", reason="not_drawn_at_random")
        assert chosen.model_check is not None and chosen.model_check.against_id == "sim/reviewer"

    def test_among_people_a_stated_figure_goes_first_then_more_labels(self) -> None:
        pairs = OUTCOMES["met"][0]
        differences = {case: judge for case, (_, judge) in pairs.items()}
        fewer = _labels(dict(list(pairs.items())[:30]), read_id="read-a")
        more = _labels(pairs, read_id="read-b")
        unusable = _labels(pairs, sample="selected", read_id="read-0")
        chosen = corrected_reading([fewer, unusable, more], differences, direction=1, confidence=0.95)
        assert chosen is not None and (chosen.read_id, chosen.model_check) == ("read-b", None)

    def test_a_model_s_lean_is_never_hidden_by_a_person_s_check_that_cannot_decide(self) -> None:
        # A person with 30 labelled cases (insufficient) and a model with 80 that finds the judge leaning.
        leaning = OUTCOMES["instrument_leans"][0]
        differences = {case: judge for case, (_, judge) in leaning.items()}
        person = _labels(dict(list(OUTCOMES["undecided"][0].items())[:30]), read_id="read-p")
        model = _labels(leaning, against="model", read_id="read-m")
        corrected = corrected_reading([person, model], differences, direction=1, confidence=0.95)
        assert corrected is not None
        assert (corrected.against, corrected.result, corrected.reason) == ("people", "insufficient", "too_few")
        assert corrected.model_check is not None
        assert (corrected.model_check.result, corrected.model_check.reason) == ("not_met", "instrument_leans")
        assert corrected.model_check.labelled_cases == PPI_DECISION_MIN_LABELLED_CASES
        # The contrast's eligibility is the person's check's alone.
        assert not decides(corrected)

    def test_a_model_alone_is_the_field_and_never_decides(self) -> None:
        pairs = OUTCOMES["met"][0]
        differences = {case: judge for case, (_, judge) in pairs.items()}
        corrected = corrected_reading([_labels(pairs, against="model")], differences, direction=1, confidence=0.95)
        assert corrected is not None
        assert (corrected.against, corrected.result, corrected.model_check) == ("model", "met", None)
        assert not decides(corrected)

    def test_a_model_s_check_rides_beside_a_person_s_only_and_one_deep(self) -> None:
        model = Corrected(against="model", against_id="sim/reviewer")
        with pytest.raises(ValueError, match="beside a person's check only"):
            Corrected(against="model", model_check=model)
        with pytest.raises(ValueError, match="beside a person's check only"):
            Corrected(against="people", model_check=Corrected(against="people"))
        with pytest.raises(ValueError, match="beside a person's check only"):
            Corrected(against="people", model_check=Corrected(against="model", model_check=model))
        carried = Corrected(against="people", model_check=model)
        assert Corrected.model_validate(carried.model_dump()) == carried
        assert "model_check" not in Corrected(against="people").model_dump(exclude_none=False)

    def test_no_reader_no_field(self) -> None:
        assert corrected_reading([], {"a": 1.0}, direction=1, confidence=0.95) is None


class TestOneVerdictOneCorrectedField:
    @pytest.mark.parametrize("model", [FamilyComparison, Contrast])
    def test_a_contrast_holds_one_verdict_and_one_corrected_field(self, model: type) -> None:
        hints = typing.get_type_hints(model)
        holding = [name for name, hint in hints.items() if Corrected in typing.get_args(hint) or hint is Corrected]
        assert holding == ["corrected"]
        assert [name for name in hints if "verdict" in name] == ["verdict"]
        assert not {"prediction_powered", "reference_corrected", "people_corrected"} & set(hints)

    def test_the_field_holds_no_sentence_and_no_stored_eligibility(self) -> None:
        hints = typing.get_type_hints(Corrected)
        assert "decision_eligible" not in hints and "unavailable_reason" not in hints
        assert "verdict" not in hints
        assert [name for name in hints if "reason" in name] == ["reason"]
        arms = [arm for arm in typing.get_args(hints["reason"]) if arm is not type(None)]
        assert str not in arms
        members = {member for arm in arms for member in (typing.get_args(arm) or (arm,))}
        assert members == {member for reasons in REASONS_OF.values() for member in reasons}


# --- the labels from a blind read, and the bundle's filler ---------------------------------------------------------


def _population(cases: int) -> tuple[list[EvalResult], dict[str, JudgeEvidence], list[BlindArm]]:
    """A control that passes every other case and a candidate that passes every case, on pass/fail."""
    arms = [
        BlindArm(key="control", role="control", run_ids=["run-a"]),
        BlindArm(key="candidate", role="candidate", run_ids=["run-b"]),
    ]
    results: list[EvalResult] = []
    evidence: dict[str, JudgeEvidence] = {}
    for case in range(cases):
        for arm in arms:
            read = JudgeEvidence(case_material=f"case {case}", artifact=f"{arm.key} answers {case}")
            key = label_key_of(read, DIM)
            result = make_eval_result(
                id=f"{arm.key}-{case}",
                eval_run_id=arm.run_ids[0],
                test_case_id=f"case-{case:04d}",
                rubric_scores=[
                    RubricScore(
                        dim=DIM.name,
                        score=1 if arm.key == "candidate" else case % 2,
                        scale="pass_fail",
                        axis="capability",
                        served_model="sim/judge",
                        judge_temperature=0.0,
                        output_fingerprint=key.output_fingerprint,
                        criterion_fingerprint=key.criterion_fingerprint,
                    )
                ],
            )
            results.append(result)
            evidence[result.id] = read
    return results, evidence, arms


def _drawn(cases: int) -> tuple[BlindRead, list[EvalResult]]:
    results, evidence, arms = _population(cases)
    read = draw_blind_read(
        results, evidence, [DIM], arms, scope_id="sim", created_by="sim", created_at="t0", seed=7, planned_items=cases
    )
    return read, results


def _answered(
    read: BlindRead,
    results: list[EvalResult],
    items: int,
    *,
    rater: str = "p",
    kind: str = "person",
    agree: bool = True,
) -> BlindRead:
    """``rater`` answers the first ``items`` items, reading what the judge read (or failing every candidate)."""
    score = {result.id: result.rubric_scores[0].score for result in results}
    answers = [
        BlindAnswer(
            item=item.item,
            rater=rater,
            rater_kind=kind,  # type: ignore[arg-type]
            reviewer_served_model=None if kind == "person" else rater,
            verdicts=[
                BlindVerdict(
                    slot=output.slot,
                    dim=DIM.name,
                    score=score[output.result_id] if agree or output.arm == "control" else 0,
                )
                for output in item.outputs
            ],
        )
        for item in read.items[:items]
    ]
    return read.model_copy(update={"answers": [*read.answers, *answers]})


def _cells(results: list[EvalResult]) -> dict[tuple[str, str], list[EvalResult]]:
    return {
        ("control", RIG): [result for result in results if result.eval_run_id == "run-a"],
        ("candidate", RIG): [result for result in results if result.eval_run_id == "run-b"],
    }


def _records(results: list[EvalResult]) -> list[ScoreRecord]:
    return [
        ScoreRecord(
            subject_id="s",
            scope_id="sim",
            run_id=result.eval_run_id,
            result_id=result.id,
            test_case_id=result.test_case_id,
            model="m",
            k_iteration=1,
            metric="score",
            value=float(result.rubric_scores[0].score),
            rubric_dim=DIM.name,
            rubric_scale="pass_fail",
            outcome=ResultOutcome.OK.value,
        )
        for result in results
    ]


def _judge_values(results: list[EvalResult]) -> dict[str, float]:
    return {result.id: float(result.rubric_scores[0].score) for result in results}


def _differences(results: list[EvalResult]) -> dict[str, float]:
    by_case: dict[str, dict[str, float]] = {}
    for result in results:
        by_case.setdefault(result.test_case_id, {})[result.eval_run_id] = float(result.rubric_scores[0].score)
    return {case: values["run-b"] - values["run-a"] for case, values in by_case.items()}


def _labels_of(read: BlindRead, results: list[EvalResult]) -> list[PairLabels]:
    cells = _cells(results)
    return pair_labels(
        read,
        DIM.name,
        control=("control", {result.id: result for result in cells[("control", RIG)]}),
        candidate=("candidate", {result.id: result for result in cells[("candidate", RIG)]}),
        judge_values=_judge_values(results),
        case_differences=_differences(results),
    )


class TestTheLabelsOfABlindRead:
    def test_a_prefix_answered_by_a_person_is_a_random_sample_of_pair_differences(self) -> None:
        read, results = _drawn(30)
        (labels,) = _labels_of(_answered(read, results, 12), results)
        assert (labels.against, labels.sample, labels.answered, labels.skipped) == ("people", "random", 12, 0)
        assert len(labels.pairs) == 12 and len(labels.population) == 30
        differences = _differences(results)
        assert all(reader == judge == differences[case] for case, (reader, judge) in labels.pairs.items())

    def test_a_model_s_answers_are_its_own_labels_never_pooled_with_a_person_s(self) -> None:
        read, results = _drawn(30)
        both = _answered(_answered(read, results, 10), results, 20, rater="sim/reviewer", kind="reference")
        labels = {one.against: one for one in _labels_of(both, results)}
        assert set(labels) == {"people", "model"}
        assert (len(labels["people"].pairs), len(labels["model"].pairs)) == (10, 20)
        assert labels["model"].against_id == "sim/reviewer"

    def test_answering_out_of_order_is_a_selected_sample(self) -> None:
        read, results = _drawn(30)
        answered = _answered(read, results, 12)
        answered = answered.model_copy(update={"answers": answered.answers[1:]})
        (labels,) = _labels_of(answered, results)
        assert labels.sample == "selected"

    def test_a_read_over_other_arms_labels_nothing(self) -> None:
        read, results = _drawn(30)
        answered = _answered(read, results, 12)
        cells = _cells(results)
        assert (
            pair_labels(
                answered,
                DIM.name,
                control=("other", {result.id: result for result in cells[("control", RIG)]}),
                candidate=("candidate", {result.id: result for result in cells[("candidate", RIG)]}),
                judge_values=_judge_values(results),
                case_differences=_differences(results),
            )
            == []
        )


def _one(rid: str, run_id: str, case: str, artifact: str, score: int) -> tuple[EvalResult, JudgeEvidence]:
    """One result judged on ``DIM`` over ``artifact``, with the evidence the judge read."""
    read = JudgeEvidence(case_material=case, artifact=artifact)
    key = label_key_of(read, DIM)
    result = make_eval_result(
        id=rid,
        eval_run_id=run_id,
        test_case_id=case,
        rubric_scores=[
            RubricScore(
                dim=DIM.name,
                score=score,
                scale="pass_fail",
                axis="capability",
                served_model="sim/judge",
                judge_temperature=0.0,
                output_fingerprint=key.output_fingerprint,
                criterion_fingerprint=key.criterion_fingerprint,
            )
        ],
    )
    return result, read


class TestTheIdenticalStratum:
    """Only a case whose outputs are byte-identical across every repeat of both arms is the zero stratum; a case whose
    outputs vary, where the two drawn ones coincide, is an ordinary pair (``test_simulated_people_corrected.py``
    holds the estimate over both)."""

    def _read(self) -> tuple[list[PairLabels], list[EvalResult]]:
        results, evidence, arms = _population(30)
        # Identical on every repeat: the stratum. The judge reads the two copies apart, which the stratum ignores.
        for result, read in (
            _one("same-a", "run-a", "case-0100", "one answer", 0),
            _one("same-b", "run-b", "case-0100", "one answer", 1),
            # Varying across repeats: the candidate's second repeat answered otherwise and is not shown (no evidence
            # stored), so the drawn pair coincides on "a shared answer" while the case's outputs differ.
            _one("vary-a", "run-a", "case-0101", "a shared answer", 0),
            _one("vary-b", "run-b", "case-0101", "a shared answer", 1),
        ):
            results.append(result)
            evidence[result.id] = read
        unshown, _ = _one("vary-b2", "run-b", "case-0101", "another answer", 0)
        results.append(unshown)
        read = draw_blind_read(
            results, evidence, [DIM], arms, scope_id="sim", created_by="sim", created_at="t0", seed=7, planned_items=32
        )
        singles = {item.case_id for item in read.items if len(item.outputs) == 1}
        assert singles == {"case-0100", "case-0101"}
        answered = _answered(read, results, len(read.items))
        return _labels_of(answered, results), results

    def test_a_case_identical_on_every_repeat_is_the_stratum_and_labels_nothing(self) -> None:
        (labels,), _ = self._read()
        assert labels.identical == frozenset({"case-0100"})
        assert "case-0100" not in labels.pairs

    def test_a_coinciding_draw_over_varying_outputs_is_a_pair_the_reader_sees_no_difference_in(self) -> None:
        (labels,), _ = self._read()
        # Reader: zero, one output seen. Judge: the candidate's results over that output, less the control's.
        assert labels.pairs["case-0101"] == (0.0, 1.0)
        assert (labels.answered, labels.skipped, len(labels.pairs)) == (32, 0, 31)


def _family(p_judged: float, p_other: float) -> MultipleComparisons:
    def compared(reading: typing.Literal["judged", "measure"], name: str, p: float) -> FamilyComparison:
        return FamilyComparison(
            reading=reading,
            name=name,
            higher_is_better=True,
            control=ComparedCell(variant_key="control", apparatus_class_id=RIG, n_cases=80),
            contrast=ComparedCell(variant_key="candidate", apparatus_class_id=RIG, n_cases=80),
            delta=0.5,
            p_raw=p,
            p_adjusted=min(1.0, 2 * p),
            verdict="improved",
        )

    family = ComparisonFamily(
        aspect="quality",
        merit_axes=[],
        family_size=2,
        interval_level=0.975,
        n_untested=0,
        comparisons=[compared("judged", DIM.name, p_judged), compared("measure", "latency_ms", p_other)],
        disclosure="two comparisons",
    )
    return MultipleComparisons(families=[family])


class TestTheFiller:
    def test_a_separated_judged_contrast_carries_the_check_at_its_holm_step_s_level(self) -> None:
        read, results = _drawn(80)
        answered = _answered(read, results, 80)
        for p_judged, level in ((0.001, "97.5%"), (0.02, "95%")):
            filled = judged_contrasts_corrected(_family(p_judged, 0.01), [answered], _cells(results), _records(results))
            judged, other = filled.families[0].comparisons
            assert other.corrected is None
            assert judged.corrected is not None
            assert (judged.corrected.result, judged.corrected.level, judged.corrected.labelled_cases) == (
                "met",
                level,
                80,
            )
            assert judged.corrected.estimate == pytest.approx(0.5)
            assert judged.verdict == "improved", "the check never touches the verdict"

    def test_people_who_fail_every_candidate_contradict_it(self) -> None:
        read, results = _drawn(80)
        answered = _answered(read, results, 80, agree=False)
        filled = judged_contrasts_corrected(_family(0.001, 0.01), [answered], _cells(results), _records(results))
        corrected = filled.families[0].comparisons[0].corrected
        assert corrected is not None and (corrected.result, corrected.reason) == ("not_met", "people_disagree")

    def test_no_read_and_no_separation_leave_the_field_unset(self) -> None:
        read, results = _drawn(80)
        family = _family(0.001, 0.01)
        assert judged_contrasts_corrected(family, [], _cells(results), _records(results)) == family
        quiet = family.families[0].comparisons[0].model_copy(update={"verdict": "not_separated"})
        unseparated = MultipleComparisons(families=[family.families[0].model_copy(update={"comparisons": [quiet]})])
        filled = judged_contrasts_corrected(
            unseparated, [_answered(read, results, 80)], _cells(results), _records(results)
        )
        assert filled.families[0].comparisons[0].corrected is None


# --- the cell-mean estimate reads random samples only ---------------------------------------------------------------


def _arm(rated: int, *, drawn: bool) -> tuple[typing.Any, int]:
    """One cell of 40 results scored 3 on 1 to 5, ``rated`` of them rated 2 by a person, on a random sample or not."""
    dim = "conversation.tone"
    results = [make_eval_result(id=f"r-{i}", test_case_id=f"case-{i:03d}") for i in range(40)]
    records = [
        ScoreRecord(
            subject_id="s",
            scope_id="uni-1",
            run_id="run-1",
            result_id=result.id,
            test_case_id=result.test_case_id,
            model="sonnet",
            k_iteration=1,
            metric="score",
            value=4.0,
            rubric_dim=dim,
            rubric_scale="ordinal",
            outcome=ResultOutcome.OK.value,
        )
        for result in results
    ]
    person_scores = {(result.id, dim): [2] for result in results[:rated]}
    (measure,) = judged_measures(
        records,
        {("arm", RIG): results},
        None,
        tiers=[],
        person_scores=person_scores,
        drawn_at_random=set(person_scores) if drawn else frozenset(),
    )
    (arm,) = measure.arms
    return arm.prediction_powered, rated


class TestTheCellMeanEstimateReadsRandomSamplesOnly:
    def test_ratings_on_a_random_sample_are_read(self) -> None:
        powered, rated = _arm(PPI_MIN_LABELLED_RESULTS, drawn=True)
        assert powered is not None and powered.mean == pytest.approx(2.0) and powered.state is None
        assert powered.n_labelled == rated

    def test_ratings_chosen_rather_than_drawn_state_no_estimate_and_say_why(self) -> None:
        powered, rated = _arm(PPI_MIN_LABELLED_RESULTS + 5, drawn=False)
        assert powered is not None and powered.mean is None
        assert powered.state == StateOf(state="not_available", reason="not_drawn_at_random")
        assert powered.n_labelled == rated

    def test_every_result_rated_is_the_population_itself_and_is_read(self) -> None:
        powered, _ = _arm(40, drawn=False)
        assert powered is not None and powered.mean == pytest.approx(2.0)

    def test_a_random_sample_below_the_minimum_is_too_few(self) -> None:
        powered, _ = _arm(PPI_MIN_LABELLED_RESULTS - 1, drawn=True)
        assert powered is not None and powered.state == StateOf(state="not_available", reason="too_few")
