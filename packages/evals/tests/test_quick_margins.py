"""``compare(margins=...)``: the quick path declares a margin on a scorer's measure, so a contrast can read equivalent.

``equivalent`` is the only verdict that says two arms are alike, and the engine reads it only against a margin
declared on the measure (``materiality_threshold``). Before ``margins=``, :func:`~threetears.evals.quick.callable_host`
declared none on any measure, so no quick comparison could ever answer "is the cheaper model good enough?".

A scorer returning a number states no range, and with none no equivalence test holds its error rate (#695), so
the engine would refuse to test a margin on it and tell the reader to "declare value_range", which the quick path
had no way to do. ``ranges=`` is that way, and a margin on such a scorer without one is refused at the call.

A margin on accuracy (#698) is the runs': the engine owns accuracy's descriptor, so ``compare(margins={"accuracy":
...})`` declares it on every arm's run at launch (``EvalRun.declared_margins``), and the analysis reads it only when
every run of the campaign declares it alike, naming it as the runs' (``margin_source`` ``run``).

Mutations that turn this file red: dropping ``materiality_threshold=margin`` in ``scorer_measure``; declaring a
default margin; dropping the no-margin disclosure; accepting a margin on a name no scorer has, on a scorer
returning a number with no range, or beside a host of the caller's own; dropping ``ranges=`` from the scorer's
descriptor; landing a score outside its declared range; not recording a run margin on the run; reading one the runs
disagree on; accepting one on accuracy with no ``expected=`` or on a core measure that is not a rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import DisclosureBlock, TableBlock, inspect_campaign_bundle
from threetears.evals.contracts import RUN_MARGIN_MEASURES, run_margin_refusal
from threetears.evals.quick import Comparison, callable_host, compare

#: Enough cases for two arms that agree on every one to be shown within 0.1 of each other.
CASES = [{"n": index} for index in range(48)]


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


async def _compare(**overrides: Any) -> Comparison:
    arguments: dict[str, Any] = {"control": "current", "scope_id": "margins", "k": 2}
    arguments.update(overrides)
    return await compare(CASES, {"current": _answers(0), "cheaper": _answers(1)}, [correct], **arguments)


def _row(comparison: Comparison, measure: str) -> dict[str, Any]:
    (row,) = comparison.contrasts(measure)
    return row


def _no_margin_lines(comparison: Comparison) -> list[str]:
    return [
        block.text
        for block in comparison.report.blocks
        if isinstance(block, DisclosureBlock) and block.text.startswith("No margin is declared")
    ]


class TestADeclaredMargin:
    async def test_two_arms_inside_it_read_equivalent_with_the_margin_named(self) -> None:
        comparison = await _compare(margins={"correct": 0.1})
        row = _row(comparison, "correct")
        assert row["verdict"].startswith("equivalent to the control") and "(margin ±0.1)" in row["verdict"]
        assert row["interval"] is not None and row["delta"] == pytest.approx(-1 / 48)
        assert _no_margin_lines(comparison) == [], "every tested measure declares a margin"

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
        comparison = await _compare(margins={"correct": 0.1})
        assert _row(comparison.against("cheaper"), "correct")["verdict"].startswith("equivalent to the control")


class TestHowManyCasesEquivalenceTakes:
    """The case counts ``reading-reports.md`` states, through ``compare``: arms answering alike on every case."""

    @staticmethod
    async def _verdict(n: int, margin: float, *, expected: bool) -> str:
        async def alike(case: Mapping[str, Any]) -> str:
            return "right" if case["n"] % 3 else "wrong"

        comparison = await compare(
            CASES[:n],
            {"current": alike, "cheaper": alike},
            [correct],
            expected=(lambda case: "right") if expected else None,
            control="current",
            scope_id=f"how-many-{n}-{margin}-{expected}",
            k=2,
            margins={"correct": margin},
        )
        return str(_row(comparison, "correct")["verdict"])

    @pytest.mark.parametrize(
        ("margin", "expected", "first"), [(0.25, False, 12), (0.1, False, 33), (0.25, True, 15), (0.1, True, 41)]
    )
    async def test_the_first_count_at_which_alike_arms_read_equivalent(
        self, margin: float, expected: bool, first: int
    ) -> None:
        """One reading in the family, or two when ``expected=`` adds accuracy and Holm divides α between them."""
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

        comparison = await compare(
            six, {"current": never, "fixed": half}, [correct], control="current", scope_id="pass-fail-bounds", k=2
        )
        row = _row(comparison, "correct")
        assert row["delta"] == pytest.approx(0.5)
        assert row["interval"] == "[-0.0748, 1] at 95%", (
            "the t interval's lower end; its upper clipped where a pass-rate difference ends"
        )


class TestNoMargin:
    async def test_no_contrast_reads_equivalent_and_the_report_says_why_in_one_line(self) -> None:
        comparison = await _compare()
        assert _row(comparison, "correct")["verdict"].startswith("not separated")
        (line,) = _no_margin_lines(comparison)
        assert "Correct score" in line and "not separated never means the arms are alike" in line
        assert "compare(margins=...)" in line
        blocks = comparison.report.blocks
        table = next(i for i, b in enumerate(blocks) if isinstance(b, TableBlock) and b.name == "comparisons")
        assert blocks[table - 1].text == line, "the line sits directly above the contrasts it explains"

    async def test_on_a_classifier_it_names_accuracy_and_the_one_way_to_declare_it(self) -> None:
        comparison = await compare(
            CASES,
            {"current": _answers(0), "cheaper": _answers(1)},
            expected=lambda case: "right",
            control="current",
            scope_id="margins-accuracy",
            k=2,
        )
        (line,) = _no_margin_lines(comparison)
        assert "No margin is declared on Accuracy" in line and "Declare one with compare(margins=...)" in line
        assert "scorer" not in line, "accuracy takes a margin of its own: no duplicate scorer is needed"


async def _classify(margins: Mapping[str, float] | None, *, n: int = len(CASES), scope_id: str = "acc") -> Comparison:
    """Two classifiers that agree on every case, one in three of them wrong, compared on accuracy."""

    async def alike(case: Mapping[str, Any]) -> str:
        return "right" if case["n"] % 3 else "wrong"

    return await compare(
        CASES[:n],
        {"current": alike, "cheaper": alike},
        expected=lambda case: "right",
        control="current",
        scope_id=scope_id,
        k=2,
        margins=margins,
    )


class TestAMarginOnAccuracy:
    """#698: a run-scoped margin on a core rate measure, with no duplicate scorer."""

    async def test_agreeing_arms_at_an_adequate_n_read_equivalent_and_the_reading_names_its_margin(self) -> None:
        comparison = await _classify({"accuracy": 0.1})
        row = _row(comparison, "accuracy")
        assert row["verdict"] == (
            "equivalent to the control, within the measure's margin (margin ±0.1, declared on the runs)"
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
        (verdict,) = (await _classify({"accuracy": 0.1}, n=12, scope_id="acc-few")).verdicts("accuracy")
        assert (verdict.outcome, verdict.reason, verdict.margin_source) == ("not_separated", "not_inside_margin", "run")

    async def test_every_run_records_it_and_the_stored_comparison_names_it(self) -> None:
        comparison = await _classify({"accuracy": 0.1}, scope_id="acc-stored")
        storage = comparison.host.storage
        for summary in comparison.arms.values():
            (run,) = storage.load_eval_runs([summary.run_id], comparison.scope_id, elide_payload=frozenset())
            assert run.declared_margins == {"accuracy": 0.1}
        bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id).bundle
        assert bundle.run_margins == {"accuracy": 0.1} and bundle.run_margins_withheld is None
        (tested,) = bundle.multiple_comparisons.families[0].comparisons
        assert (tested.equivalence_margin, tested.margin_source, tested.verdict) == (0.1, "run", "equivalent")

    async def test_runs_that_disagree_on_it_read_none_and_say_so(self) -> None:
        comparison = await _classify({"accuracy": 0.1}, scope_id="acc-disagree")
        storage = comparison.host.storage
        cheaper = comparison.arms["cheaper"].run_id
        (run,) = storage.load_eval_runs([cheaper], comparison.scope_id, elide_payload=frozenset())
        storage.save_eval_run(run.model_copy(update={"declared_margins": {"accuracy": 0.2}}))
        bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id).bundle
        assert bundle.run_margins == {}
        assert bundle.run_margins_withheld is not None and "do not all declare one margin on accuracy" in (
            bundle.run_margins_withheld
        )
        (tested,) = bundle.multiple_comparisons.families[0].comparisons
        assert (tested.margin_source, tested.verdict) == (None, "not_separated")

    async def test_a_run_stored_before_run_margins_declares_none(self) -> None:
        comparison = await _classify(None, scope_id="acc-none")
        storage = comparison.host.storage
        (run,) = storage.load_eval_runs(
            [comparison.arms["current"].run_id], comparison.scope_id, elide_payload=frozenset()
        )
        stored = run.model_dump(mode="json")
        del stored["declared_margins"]
        assert type(run).model_validate(stored).declared_margins == {}


class TestARefusedMargin:
    async def test_on_accuracy_without_expected_it_says_only_a_classifier_measures_it(self) -> None:
        with pytest.raises(ValueError, match=r"names 'accuracy', which only a classifier's arms measure"):
            await _compare(margins={"accuracy": 0.05})

    @pytest.mark.parametrize("name", ["match", "precision", "cost_usd"])
    async def test_on_a_core_measure_that_takes_no_run_margin_it_names_the_one_that_does(self, name: str) -> None:
        with pytest.raises(ValueError, match=r"declares margins only on accuracy"):
            await _compare(margins={name: 0.05})

    @pytest.mark.parametrize("margin", [0, 1, -0.1, float("nan"), True, "0.1"])
    async def test_on_accuracy_outside_a_rate_s_width(self, margin: Any) -> None:
        assert run_margin_refusal("accuracy", margin) is not None
        with pytest.raises(ValueError, match="the margin on 'accuracy'"):
            await _classify({"accuracy": margin}, scope_id="acc-refused")

    async def test_a_scorer_margin_with_a_host_of_your_own_but_an_accuracy_margin_works_with_one(self) -> None:
        with pytest.raises(ValueError, match="pass one or the other"):
            await _compare(margins={"correct": 0.1}, host=callable_host([correct]))
        assert run_margin_refusal("accuracy", 0.05) is None and RUN_MARGIN_MEASURES == {"accuracy"}

    async def test_on_a_name_no_scorer_has_it_names_the_scorers(self) -> None:
        with pytest.raises(ValueError, match=r"'corect', which no scorer reports.*'correct'"):
            await _compare(margins={"corect": 0.05})

    @pytest.mark.parametrize("margin", [0, -0.1, float("nan"), True, "0.1"])
    async def test_that_is_not_a_positive_number(self, margin: Any) -> None:
        with pytest.raises(ValueError, match="is a positive number in the measure's own units"):
            await _compare(margins={"correct": margin})

    async def test_on_a_scorer_returning_a_number_with_no_range_it_teaches_ranges(self) -> None:
        """Refused at the call, never accepted and then left untested with a remedy the quick path cannot take."""
        with pytest.raises(ValueError, match=r"a margin on 'wordy' needs its range too.*ranges=\{'wordy': \(1, 5\)\}"):
            callable_host([wordy], margins={"wordy": 2.0})


def rating(case: Mapping[str, Any], answer: str) -> float:
    """A 1-to-5 rating of the answer."""
    return 5.0 if answer == "right" else 1.0


class TestADeclaredRange:
    async def test_a_float_scorer_with_a_margin_and_its_range_can_read_equivalent(self) -> None:
        comparison = await compare(
            CASES,
            {"current": _answers(0), "cheaper": _answers(1)},
            [rating],
            control="current",
            scope_id="ranges",
            k=2,
            margins={"rating": 0.5},
            ranges={"rating": (1, 5)},
        )
        row = _row(comparison, "rating")
        assert row["verdict"].startswith("equivalent to the control") and "(margin ±0.5)" in row["verdict"]
        assert "value_range" not in comparison.render()

    async def test_a_score_outside_it_excludes_the_cell_naming_the_scorer(self) -> None:
        def overrated(case: Mapping[str, Any], answer: str) -> float:
            """A rating that runs past its own scale on one case."""
            return 7.0 if case["n"] == 0 else 3.0

        summary = (
            await compare(
                CASES[:4],
                {"current": _answers(0), "cheaper": _answers(0)},
                [overrated],
                control="current",
                scope_id="ranges-outside",
                k=1,
                ranges={"overrated": (1, 5)},
            )
        ).arms["current"]
        assert summary.n_excluded == 1
        assert any("returned 7.0, outside the range 1 to 5 declared for it" in error for error in summary.errors)

    @pytest.mark.parametrize(
        ("ranges", "match"),
        [
            ({"correct": (0, 1)}, "returns a bool, a pass/fail, which is on 0 to 1 already"),
            ({"rateing": (1, 5)}, "'rateing', which no scorer reports"),
            ({"rating": (5, 1)}, "the lowest and the highest score it can return"),
            ({"rating": (1, float("inf"))}, "the lowest and the highest score it can return"),
            ({"rating": 5}, "the lowest and the highest score it can return"),
        ],
    )
    async def test_an_unusable_one_is_refused(self, ranges: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            callable_host([correct, rating], ranges=ranges)

    async def test_beside_a_host_of_your_own_is_refused(self) -> None:
        with pytest.raises(ValueError, match="pass one or the other"):
            await _compare(ranges={"correct": (0, 1)}, host=callable_host([correct]))
