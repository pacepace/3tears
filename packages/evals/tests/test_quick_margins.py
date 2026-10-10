"""``compare(margins=...)``: the quick path declares a margin on a scorer's measure, so a contrast can read equivalent.

``equivalent`` is the only verdict that says two arms are alike, and the engine reads it only against a margin
declared on the measure (``materiality_threshold``). Before ``margins=``, :func:`~threetears.evals.quick.callable_host`
declared none on any measure, so no quick comparison could ever answer "is the cheaper model good enough?".

A scorer returning a number states no range, and with none no equivalence test holds its error rate (#695), so
the engine would refuse to test a margin on it and tell the reader to "declare value_range", which the quick path
had no way to do. ``ranges=`` is that way, and a margin on such a scorer without one is refused at the call.

Mutations that turn this file red: dropping ``materiality_threshold=margin`` in ``scorer_measure``; declaring a
default margin; dropping the no-margin disclosure; accepting a margin on accuracy, on a name no scorer has, on a
scorer returning a number with no range, or beside a host of the caller's own; dropping ``ranges=`` from the
scorer's descriptor; landing a score outside its declared range.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import DisclosureBlock, TableBlock
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


async def _compare(*, cheaper_misses: int = 1, **overrides: Any) -> Comparison:
    arguments: dict[str, Any] = {"control": "current", "scope_id": "margins", "k": 2}
    arguments.update(overrides)
    arms = {"current": _answers(0), "cheaper": _answers(cheaper_misses)}
    return await compare(CASES, arms, [correct], **arguments)


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
        """Alike on all 48 cases. One miss among them is shown inside 0.1 in only 18 of the 48 places the bounded
        test can meet it, so the arms here agree throughout rather than test where the miss falls."""
        comparison = await _compare(margins={"correct": 0.1}, cheaper_misses=0)
        row = _row(comparison, "correct")
        assert row["verdict"].startswith("equivalent to the control") and "(margin ±0.1)" in row["verdict"]
        assert row["delta"] == 0.0
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
        comparison = await _compare(margins={"correct": 0.1}, cheaper_misses=0)
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

    async def test_on_a_classifier_it_says_accuracy_takes_none_and_what_to_do(self) -> None:
        comparison = await compare(
            CASES,
            {"current": _answers(0), "cheaper": _answers(1)},
            expected=lambda case: "right",
            control="current",
            scope_id="margins-accuracy",
            k=2,
        )
        (line,) = _no_margin_lines(comparison)
        assert "Accuracy takes no margin: grade with a scorer too" in line


class TestARefusedMargin:
    @pytest.mark.parametrize("name", ["accuracy", "match"])
    async def test_on_accuracy_it_teaches_the_scorer_route(self, name: str) -> None:
        with pytest.raises(ValueError, match=r"grade it with a scorer as well .* margins=\{'correct': 0.05\}"):
            await _compare(margins={name: 0.05})

    async def test_on_a_name_no_scorer_has_it_names_the_scorers(self) -> None:
        with pytest.raises(ValueError, match=r"'corect', which no scorer reports.*'correct'"):
            await _compare(margins={"corect": 0.05})

    @pytest.mark.parametrize("margin", [0, -0.1, float("nan"), True, "0.1"])
    async def test_that_is_not_a_positive_number(self, margin: Any) -> None:
        with pytest.raises(ValueError, match="is a positive number in the measure's own units"):
            await _compare(margins={"correct": margin})

    async def test_beside_a_host_of_your_own(self) -> None:
        with pytest.raises(ValueError, match="pass one or the other"):
            await _compare(margins={"correct": 0.1}, host=callable_host([correct]))

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
