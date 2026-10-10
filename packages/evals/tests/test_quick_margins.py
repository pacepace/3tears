"""``compare(margins=...)``: the quick path declares a margin on a scorer's measure, so a contrast can read equivalent.

``equivalent`` is the only verdict that says two arms are alike, and the engine reads it only against a margin
declared on the measure (``materiality_threshold``). Before ``margins=``, :func:`~threetears.evals.quick.callable_host`
declared none on any measure, so no quick comparison could ever answer "is the cheaper model good enough?".

Mutations that turn this file red: dropping ``materiality_threshold=margin`` in ``scorer_measure``; declaring a
default margin; dropping the no-margin disclosure; accepting a margin on accuracy, on a name no scorer has, or
beside a host of the caller's own.
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

    async def test_a_bool_scorer_s_margin_comes_with_its_range_and_a_float_scorer_s_without(self) -> None:
        host = callable_host([correct, wordy], margins={"correct": 0.1, "wordy": 2.0})
        declared = {name: host.profile.measures.get(name) for name in ("correct", "wordy")}
        assert (declared["correct"].materiality_threshold, declared["correct"].value_range) == (0.1, (0.0, 1.0))
        assert (declared["wordy"].materiality_threshold, declared["wordy"].value_range) == (2.0, None)
        plain = callable_host([correct, wordy]).profile.measures
        assert (plain.get("correct").materiality_threshold, plain.get("correct").value_range) == (None, (0.0, 1.0)), (
            "no margin is ever assumed, and a pass/fail is on 0-1 whatever its margin"
        )
        assert plain.get("wordy").value_range is None, "nothing says what a float scorer's values can be"

    async def test_a_second_control_reads_the_same_margin(self) -> None:
        comparison = await _compare(margins={"correct": 0.1})
        assert _row(comparison.against("cheaper"), "correct")["verdict"].startswith("equivalent to the control")


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
