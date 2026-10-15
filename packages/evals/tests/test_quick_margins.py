"""``grader(fn, margin=...)``: a margin declared on a grader's reading, so a contrast can read equivalent.

``equivalent`` is the only verdict that says two arms are alike, and the engine reads it only against a margin
declared on the measure (``materiality_threshold``). With none declared — and none ever is by default — no
comparison could answer "is the cheaper model good enough?".

A grader returning a number states no range, and with none no equivalence test holds its error rate (#695), so
the engine would refuse to test a margin on it. ``grader(fn, range=(low, high))`` declares it, and a margin on such
a grader without one is refused before anything runs.

A margin on accuracy (#698) is the runs': the engine owns accuracy's descriptor, so a margin on the classifier's
field (``grader("label", margin=...)``, the first field grader) is declared on every arm's run at launch
(``EvalRun.declared_margins``), and the analysis reads it only when every run of the campaign declares it alike,
naming it as the runs' (``margin_source`` ``run``).

Mutations that turn this file red: dropping ``materiality_threshold=margin`` in ``scorer_measure``; declaring a
default margin; dropping the no-margin disclosure; accepting a margin that is not a positive number, on a grader
returning a number with no range, or on a host of the caller's own that does not declare it; dropping the range from
the grader's descriptor; landing a score outside its declared range; not recording a run margin on the run; reading
one the runs disagree on; accepting one on a core measure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import DisclosureBlock, TableBlock, inspect_campaign_bundle
from threetears.evals import Answer, Eval, Result, grader
from threetears.evals.analysis import report_markdown
from threetears.evals.kernel import RUN_MARGIN_MEASURES, ValidationFailedError, run_margin_refusal
from threetears.evals.quick.one_call import callable_host
from packages.evals.tests.off_loop import off_loop

#: Enough cases for two arms that agree on every one to be shown within 0.1 of each other. Each expects the answer
#: ``right``, which a classifier's field grader reads.
CASES = [{"n": index, "label": "right"} for index in range(48)]


def correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is right."""
    return answer == "right"


def wordy(case: Mapping[str, Any], answer: str) -> float:
    """How many words the answer has."""
    return float(len(answer.split()))


def _answers(misses: int) -> Any:
    async def candidate(case: Mapping[str, Any]) -> str:
        return "wrong" if case["n"] < misses else "right"

    return candidate


async def _compare(
    graders: list[Any] | None = None, *, cheaper_misses: int = 1, host: Any = None, scope: str = "margins"
) -> Result:
    arms = {"current": _answers(0), "cheaper": _answers(cheaper_misses)}
    ev = Eval(CASES, arms, [correct] if graders is None else graders, k=2)
    if host is not None:
        return await ev.run(host=host, scope=scope)
    return await ev.run(store=":memory:", scope=scope)


async def _run(cases: list[dict[str, Any]], arms: Mapping[str, Any], graders: list[Any], scope: str, k: int) -> Result:
    return await Eval(cases, arms, graders, k=k).run(store=":memory:", scope=scope)


def _row(comparison: Result, measure: str) -> dict[str, Any]:
    (row,) = comparison.contrasts(measure)
    return row


def _no_margin_lines(comparison: Result) -> list[str]:
    return [
        block.text
        for block in comparison.report.blocks
        if isinstance(block, DisclosureBlock) and block.text.startswith("No margin is declared")
    ]


def _no_margin_advice(comparison: Result) -> dict[str, str]:
    """The no-margin table: each reading, named once, with how it takes a margin."""
    tables = [b for b in comparison.report.blocks if isinstance(b, TableBlock) and b.name == "no_margin"]
    return {str(row["reading"]): str(row["declare"]) for table in tables for row in table.values()}


class TestADeclaredMargin:
    async def test_two_arms_inside_it_read_equivalent_with_the_margin_named(self) -> None:
        """Alike on all 48 cases. One miss among them is shown inside 0.1 in only 18 of the 48 places the bounded
        test can meet it, so the arms here agree throughout rather than test where the miss falls."""
        comparison = await _compare([grader(correct, margin=0.1)], cheaper_misses=0)
        row = _row(comparison, "correct")
        # A margin on a pass rate is stated in points, as the change it bounds is.
        assert row["verdict"].startswith("equivalent to the control") and "(margin ±10 pts)" in row["verdict"]
        assert row["delta"] == 0.0
        assert _no_margin_lines(comparison) == [], "every tested measure declares a margin"
        assert _no_margin_advice(comparison) == {}

    async def test_a_bool_scorer_s_margin_comes_with_its_range_and_a_float_scorer_s_with_the_one_declared(self) -> None:
        host = callable_host([correct, wordy], margins={"correct": 0.1, "wordy": 2.0}, ranges={"wordy": (0, 50)})
        declared = {name: host.profile.measures.get(name) for name in ("correct", "wordy")}
        assert (declared["correct"].materiality_threshold, declared["correct"].value_range) == (0.1, (0.0, 1.0))
        assert (declared["wordy"].materiality_threshold, declared["wordy"].value_range) == (2.0, (0.0, 50.0))
        plain = callable_host([correct, wordy]).profile.measures
        assert (plain.get("correct").materiality_threshold, plain.get("correct").value_range) == (None, (0.0, 1.0)), (
            "no margin is ever assumed, and a pass/fail is on 0-1 whatever its margin"
        )
        assert plain.get("wordy").value_range is None, "nothing says what a float scorer's values can be"

    async def test_a_second_control_reads_the_same_margin(self) -> None:
        comparison = await _compare([grader(correct, margin=0.1)], cheaper_misses=0)
        assert _row(await comparison.against("cheaper"), "correct")["verdict"].startswith("equivalent to the control")


class TestHowManyCasesEquivalenceTakes:
    """The case counts ``reading-reports.md`` states, through a comparison: arms answering alike on every case."""

    @staticmethod
    async def _verdict(n: int, margin: float, *, expected: bool) -> str:
        async def alike(case: Mapping[str, Any]) -> str:
            return "right" if case["n"] % 3 else "wrong"

        graders = [*(["label"] if expected else []), grader(correct, margin=margin)]
        comparison = await _run(
            CASES[:n], {"current": alike, "cheaper": alike}, graders, f"how-many-{n}-{margin}-{expected}", 2
        )
        return str(_row(comparison, "correct")["verdict"])

    @pytest.mark.parametrize(
        ("margin", "expected", "first"), [(0.25, False, 12), (0.1, False, 33), (0.25, True, 15), (0.1, True, 41)]
    )
    async def test_the_first_count_at_which_alike_arms_read_equivalent(
        self, margin: float, expected: bool, first: int
    ) -> None:
        """One reading in the family, or two when a field grader adds accuracy and Holm divides α between them."""
        assert (await self._verdict(first - 1, margin, expected=expected)).startswith("not separated")
        assert (await self._verdict(first, margin, expected=expected)).startswith("equivalent")


class TestAPassFailInterval:
    async def test_stays_inside_the_differences_a_pass_rate_allows(self) -> None:
        """Half of six cases flip to right: the t interval on the delta runs to 1.075, past any pass-rate difference.

        No margin is declared, so only the scorer's ``bool`` annotation says the values are 0 or 1 (the
        cassettes example printed ``[-0.0748, 1.075]`` before it did).
        """
        six = CASES[:6]

        async def never(case: Mapping[str, Any]) -> str:
            return "wrong"

        async def half(case: Mapping[str, Any]) -> str:
            return "right" if case["n"] < 3 else "wrong"

        comparison = await _run(six, {"current": never, "fixed": half}, [correct], "pass-fail-bounds", 2)
        row = _row(comparison, "correct")
        # A pass rate is a share, so its change reads in points on the percent ruler.
        assert row["delta"] == pytest.approx(50.0)
        assert row["interval"] == "[-7.5, 100] pts at 95%", (
            "the t interval's lower end; its upper clipped where a pass-rate difference ends"
        )


class TestNoMargin:
    async def test_no_contrast_reads_equivalent_and_the_report_says_why_in_one_line(self) -> None:
        comparison = await _compare()
        assert _row(comparison, "correct")["verdict"].startswith("not separated")
        (line,) = _no_margin_lines(comparison)
        assert "not separated never means the arms are alike" in line
        # One sentence, then each reading once in a table, with how it takes a margin.
        advice = _no_margin_advice(comparison)
        assert "grader(correct, margin=...)" in advice["Correct score"]
        (block,) = [b for b in comparison.report.blocks if isinstance(b, DisclosureBlock) and b.text == line]
        assert block.section == "not_tested", "said with what the campaign could not test"
        (table,) = [b for b in comparison.report.blocks if isinstance(b, TableBlock) and b.name == "no_margin"]
        assert table.section == "not_tested"

    async def test_on_a_classifier_it_names_accuracy_and_the_one_way_to_declare_it(self) -> None:
        arms = {"current": _answers(0), "cheaper": _answers(1)}
        comparison = await _run(CASES, arms, ["label"], "margins-accuracy", 2)
        (line,) = _no_margin_lines(comparison)
        advice = _no_margin_advice(comparison)
        assert advice["Accuracy"] == "on the classifier's field, on every arm's run: `grader('<field>', margin=...)`"
        assert "scorer" not in line + advice["Accuracy"], "accuracy takes a margin of its own: no duplicate scorer"


async def _classify(margin: Any, *, n: int = len(CASES), scope_id: str = "acc") -> Result:
    """Two classifiers that agree on every case, one in three of them wrong, compared on accuracy."""

    async def alike(case: Mapping[str, Any]) -> str:
        return "right" if case["n"] % 3 else "wrong"

    label = "label" if margin is None else grader("label", margin=margin)
    return await _run(CASES[:n], {"current": alike, "cheaper": alike}, [label], scope_id, 2)


class TestAMarginOnAccuracy:
    """#698: a run-scoped margin on a core rate measure, with no duplicate scorer."""

    async def test_agreeing_arms_at_an_adequate_n_read_equivalent_and_the_reading_names_its_margin(self) -> None:
        comparison = await _classify(0.1)
        row = _row(comparison, "accuracy")
        assert row["verdict"] == (
            "equivalent to the control, within the measure's margin (margin ±10 pts, declared on the runs)"
        )
        (verdict,) = comparison.verdicts("accuracy")
        assert (verdict.outcome, verdict.reason, verdict.margin, verdict.margin_source) == (
            "equivalent",
            "inside_margin",
            0.1,
            "run",
        )
        assert _no_margin_lines(comparison) == []

    async def test_too_few_cases_stay_not_separated_naming_the_margin_it_could_not_show(self) -> None:
        (verdict,) = (await _classify(0.1, n=12, scope_id="acc-few")).verdicts("accuracy")
        assert (verdict.outcome, verdict.reason, verdict.margin_source) == ("not_separated", "not_inside_margin", "run")

    async def test_every_run_records_it_and_the_stored_comparison_names_it(self) -> None:
        comparison = await _classify(0.1, scope_id="acc-stored")
        storage = comparison.host.storage
        for summary in comparison.arms.values():
            (run,) = off_loop(storage.load_eval_runs, [summary.run_id], comparison.scope_id, elide_payload=frozenset())
            assert run.declared_margins == {"accuracy": 0.1}
        bundle = off_loop(inspect_campaign_bundle, comparison.host, comparison.campaign_id, comparison.scope_id).bundle
        assert bundle.run_margins == {"accuracy": 0.1} and bundle.run_margins_withheld is None
        (tested,) = bundle.multiple_comparisons.families[0].comparisons
        assert (tested.equivalence_margin, tested.margin_source, tested.verdict) == (0.1, "run", "equivalent")

    async def test_runs_that_disagree_on_it_read_none_and_say_so(self) -> None:
        comparison = await _classify(0.1, scope_id="acc-disagree")
        storage = comparison.host.storage
        cheaper = comparison.arms["cheaper"].run_id
        (run,) = off_loop(storage.load_eval_runs, [cheaper], comparison.scope_id, elide_payload=frozenset())
        off_loop(storage.save_eval_run, run.model_copy(update={"declared_margins": {"accuracy": 0.2}}))
        bundle = off_loop(inspect_campaign_bundle, comparison.host, comparison.campaign_id, comparison.scope_id).bundle
        assert bundle.run_margins == {}
        assert bundle.run_margins_withheld is not None and "do not all declare one margin on accuracy" in (
            bundle.run_margins_withheld
        )
        (tested,) = bundle.multiple_comparisons.families[0].comparisons
        assert (tested.margin_source, tested.verdict) == (None, "not_separated")

    async def test_a_run_stored_before_run_margins_declares_none(self) -> None:
        comparison = await _classify(None, scope_id="acc-none")
        storage = comparison.host.storage
        (run,) = off_loop(
            storage.load_eval_runs, [comparison.arms["current"].run_id], comparison.scope_id, elide_payload=frozenset()
        )
        stored = run.model_dump(mode="json")
        del stored["declared_margins"]
        assert type(run).model_validate(stored).declared_margins == {}


class TestARefusedMargin:
    @pytest.mark.parametrize("name", ["accuracy", "match", "precision", "cost_usd"])
    def test_on_a_core_measure_it_names_the_function_route(self, name: str) -> None:
        with pytest.raises(ValueError, match=r"engine core measure's name; grade it with a function"):
            Eval(CASES, {"current": _answers(0), "cheaper": _answers(1)}, [grader(name, margin=0.05)])

    async def test_on_accuracy_wider_than_a_rate_the_launch_refuses_it(self) -> None:
        assert run_margin_refusal("accuracy", 1) is not None
        with pytest.raises(ValidationFailedError, match="the margin on 'accuracy'"):
            await _classify(1, scope_id="acc-refused")

    @pytest.mark.parametrize("margin", [0, -0.1, float("nan"), True, "0.1"])
    def test_that_is_not_a_positive_number(self, margin: Any) -> None:
        assert run_margin_refusal("accuracy", margin) is not None
        with pytest.raises(ValueError, match="is a positive number in the reading's own units"):
            grader("label", margin=margin)
        with pytest.raises(ValueError, match="is a positive number in the reading's own units"):
            grader(correct, margin=margin)

    async def test_a_grader_margin_a_host_of_your_own_does_not_declare_is_refused_but_an_accuracy_one_works(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="without the materiality_threshold its grader declares"):
            await _compare([grader(correct, margin=0.1)], host=callable_host([correct]))
        assert run_margin_refusal("accuracy", 0.05) is None and RUN_MARGIN_MEASURES == {"accuracy"}
        result = await Eval(CASES, _answers(0), [grader("label", margin=0.05)], k=1).run(
            host=callable_host(), scope="acc-own-host"
        )
        assert result.summary.status == "completed"

    async def test_on_a_scorer_returning_a_number_with_no_range_it_teaches_ranges(self) -> None:
        """Refused at the call, never accepted and then left untested with a remedy the quick path cannot take."""
        with pytest.raises(
            ValueError, match=r"a margin on 'wordy' needs its range too.*grader\(wordy, margin=\.\.\., range=\(1, 5\)\)"
        ):
            callable_host([wordy], margins={"wordy": 2.0})


def rating(case: Mapping[str, Any], answer: str) -> float:
    """A 1-to-5 rating of the answer."""
    return 5.0 if answer == "right" else 1.0


class TestADeclaredRange:
    async def test_a_float_scorer_with_a_margin_and_its_range_can_read_equivalent(self) -> None:
        comparison = await _run(
            CASES,
            {"current": _answers(0), "cheaper": _answers(1)},
            [grader(rating, margin=0.5, range=(1, 5))],
            "ranges",
            2,
        )
        row = _row(comparison, "rating")
        assert row["verdict"].startswith("equivalent to the control") and "(margin ±0.5)" in row["verdict"]
        assert "value_range" not in report_markdown(comparison.report)

    async def test_a_score_outside_it_excludes_the_cell_naming_the_scorer(self) -> None:
        def overrated(case: Mapping[str, Any], answer: str) -> float:
            """A rating that runs past its own scale on one case."""
            return 7.0 if case["n"] == 0 else 3.0

        summary = (
            await _run(
                CASES[:4],
                {"current": _answers(0), "cheaper": _answers(0)},
                [grader(overrated, range=(1, 5))],
                "ranges-outside",
                1,
            )
        ).arms["current"]
        assert summary.n_excluded == 1
        assert any("returned 7.0, outside the range 1 to 5 declared for it" in error for error in summary.errors)

    @pytest.mark.parametrize(
        ("ranges", "match"),
        [
            ({"correct": (0, 1)}, "returns a bool, a pass/fail, which is on 0 to 1 already"),
            ({"rateing": (1, 5)}, "'rateing', which no grader reports"),
            ({"rating": (5, 1)}, "the lowest and the highest score it can return"),
            ({"rating": (1, float("inf"))}, "the lowest and the highest score it can return"),
            ({"rating": 5}, "the lowest and the highest score it can return"),
        ],
    )
    async def test_an_unusable_one_is_refused(self, ranges: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            callable_host([correct, rating], ranges=ranges)

    async def test_one_a_host_of_your_own_does_not_declare_is_refused(self) -> None:
        with pytest.raises(ValueError, match="without the value_range its grader declares"):
            await _compare([grader(rating, range=(1, 5))], host=callable_host([rating]))

    @pytest.mark.parametrize("bounds", [(5, 1), (1, float("inf")), (1, 2, 3)])
    def test_an_unusable_one_is_refused_where_the_grader_is_declared(self, bounds: Any) -> None:
        with pytest.raises(ValueError, match="the lowest and highest grade it can return"):
            grader(rating, range=bounds)

    def test_a_field_takes_none(self) -> None:
        with pytest.raises(ValueError, match="graded pass/fail, on 0 to 1 already"):
            grader("label", range=(0, 1))


class TestTheNoMarginLineAdvisesEachKindOfMeasure:
    """The line names each measure with the advice that applies to it, and never tells a core measure to take a
    scorer's margin."""

    async def test_scorers_accuracy_and_an_engine_measure_each_get_their_own_advice(self) -> None:
        async def priced(case: Mapping[str, Any]) -> Answer:
            return Answer("right", model="m", input_tokens=10, output_tokens=1, cost_usd=0.001 * (1 + case["n"] % 2))

        comparison = await _run(CASES, {"current": priced, "cheaper": priced}, ["label", correct], "no-margin-kinds", 1)
        (_line,) = _no_margin_lines(comparison)
        advice = _no_margin_advice(comparison)
        assert advice["Correct score"] == "where it is graded: `grader(correct, margin=...)`"
        assert advice["Accuracy"] == "on the classifier's field, on every arm's run: `grader('<field>', margin=...)`"
        assert advice["Production cost"] == "cannot: the engine owns its description"
