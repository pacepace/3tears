"""The reserved ``stratum`` key: a case names its kind, and the results are read per kind of case.

A pooled accuracy is dominated by the easy majority, so the classifier guide has each case declare its kind as its
stratum, under the reserved key ``stratum``. The runner never sees it, nor does any grader; the summary reads every
measure again per stratum, and a comparison's report carries its By stratum table, which :meth:`Result.strata`
reads.

Mutations that turn this file red: the key left in the case handed to the runner or a grader; the stratum not stored
on the test case; the stratum entering what addresses the case set, so re-labelling a case splits its history; the
summary not breaking a stratified run down, or breaking an unstratified one down; the too-few-cases warning dropped;
:meth:`Result.strata` naming an arm by its words rather than its key.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import StratumSummary
from threetears.evals.analysis.report.build import NO_STRATUM, TOO_FEW_CASES
from threetears.evals.kernel.surface import STRATUM_MIN_CASES
from threetears.evals import Eval
from threetears.evals.analysis import report_markdown
from threetears.evals.quick.one_call import callable_host
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.off_loop import off_loop

#: Ten plain tickets and four lookalikes: enough plain ones to read alone, too few lookalikes.
CASES = [
    {"ticket": f"plain {index}", "queue": "a" if index % 2 else "b", "stratum": "plain"} for index in range(10)
] + [{"ticket": f"lookalike {index}", "queue": "b", "stratum": "lookalike"} for index in range(4)]

#: The field each case carries its expected queue in, graded as the classifier.
QUEUE = "queue"


async def misses_lookalikes(case: Mapping[str, Any]) -> str:
    """Right on every plain ticket, wrong on every lookalike — and never told which is which."""
    assert "stratum" not in case, "the runner was handed the stratum"
    return "a" if str(case["ticket"]).startswith("lookalike") else str(case["queue"])


async def always_b(case: Mapping[str, Any]) -> str:
    assert "stratum" not in case, "the runner was handed the stratum"
    return "b"


def answered(case: Mapping[str, Any], answer: Any) -> bool:
    assert "stratum" not in case, "a grader was handed the stratum"
    return answer is not None


async def _summary(cases: Any, runner: Any, graders: list[Any], **run: Any) -> Any:
    """The one arm's summary of an eval with ``k=1``, in memory unless a store is named."""
    run.setdefault("store", ":memory:")
    return (await Eval(cases, runner, graders, k=1).run(**run)).summary


def _by_name(strata: list[StratumSummary]) -> dict[str | None, StratumSummary]:
    return {stratum.stratum: stratum for stratum in strata}


def _mean(stratum: StratumSummary, measure: str) -> float | None:
    return next(summary.mean for summary in stratum.measures if summary.name == measure)


class TestAnEvalReadsEachKind:
    async def test_the_summary_reads_every_measure_per_stratum(self) -> None:
        summary = await _summary(CASES, misses_lookalikes, [QUEUE, answered])

        strata = _by_name(summary.strata)
        assert list(strata) == ["lookalike", "plain"]
        assert (strata["plain"].n_cases, strata["plain"].n_results) == (10, 10)
        assert (strata["lookalike"].n_cases, strata["lookalike"].n_results) == (4, 4)
        assert _mean(strata["plain"], "match") == 1.0
        assert _mean(strata["lookalike"], "match") == 0.0
        assert _mean(strata["lookalike"], "answered") == 1.0
        # The confusion matrix is the run's; a stratum's match rate is its accuracy.
        assert all(measure.name != "confusion_cell" for stratum in summary.strata for measure in stratum.measures)

    async def test_a_stratum_under_the_floor_is_said_to_be_too_few(self) -> None:
        summary = await _summary(CASES, misses_lookalikes, [QUEUE])

        strata = _by_name(summary.strata)
        assert strata["lookalike"].too_few and not strata["plain"].too_few
        rendered = summary.render()
        assert "  by stratum:" in rendered
        assert f"lookalike: 4 case(s), 4 result(s) — {TOO_FEW_CASES} (fewer than {STRATUM_MIN_CASES})" in rendered
        assert "plain: 10 case(s), 10 result(s)\n" in rendered
        assert "      match: mean 0 (n=4" in rendered
        # The served lines are what render prints, indented under its heading.
        assert all(f"    {line}" in rendered for line in summary.stratum_lines)

    async def test_each_result_carries_its_cases_stratum_and_not_in_its_input(self) -> None:
        summary = await _summary(CASES, misses_lookalikes, [QUEUE])

        results = summary.results()
        assert [result.stratum for result in results] == [case["stratum"] for case in CASES]
        assert all(isinstance(result.input, dict) and "stratum" not in result.input for result in results)
        assert results[-1].render().startswith("case 13 (lookalike, repeat 1, scored)")

    async def test_a_case_without_the_key_declares_none(self) -> None:
        cases = [*CASES, {"ticket": "plain x", "queue": "b"}, {"ticket": "plain y", "queue": "b", "stratum": None}]
        summary = await _summary(cases, misses_lookalikes, [QUEUE])

        assert [stratum.stratum for stratum in summary.strata] == ["lookalike", "plain", None]
        assert _by_name(summary.strata)[None].n_cases == 2
        assert f"    {NO_STRATUM}: 2 case(s)" in summary.render()

    async def test_no_key_reads_as_before(self) -> None:
        unkinded = [{key: value for key, value in case.items() if key != "stratum"} for case in CASES]
        summary = await _summary(unkinded, always_b, [QUEUE])

        assert summary.strata == [] and summary.stratum_lines == []
        assert "by stratum" not in summary.render()
        assert all(result.stratum is None for result in summary.results())


class TestRelabellingKeepsHistory:
    async def test_a_new_stratum_is_stored_on_the_same_case(self) -> None:
        store = InMemoryDocumentStore()
        first = await _summary(CASES, always_b, [QUEUE], store=store, scope="s")
        relabelled = [{**case, "stratum": "hard"} if case["stratum"] == "lookalike" else case for case in CASES]
        second = await _summary(relabelled, always_b, [QUEUE], store=store, scope="s")

        # The same template and the same cases: the stratum addresses nothing.
        assert first.template_id == second.template_id
        host = callable_host([], store=store)
        stored = off_loop(
            host.storage.load_case_strata, [f"{first.template_id}-{index}" for index in range(len(CASES))], "s"
        )
        assert sorted({case.stratum for case in stored if case.stratum is not None}) == ["hard", "plain"]
        assert [stratum.stratum for stratum in second.strata] == ["hard", "plain"]


class TestRefusals:
    @pytest.mark.parametrize("value", ["", " ", 3, ["plain"]])
    async def test_a_stratum_that_is_no_name_is_refused_naming_its_case(self, value: Any) -> None:
        cases = [*CASES[:2], {"ticket": "x", "queue": "b", "stratum": value}]
        with pytest.raises(ValueError, match=r"case 2's stratum is"):
            await _summary(cases, always_b, [QUEUE])


async def _compared(cases: Any, graders: list[Any]) -> Any:
    return await Eval(cases, {"baseline": always_b, "candidate": misses_lookalikes}, graders, k=1).run(store=":memory:")


class TestAComparisonReadsEachArmPerKind:
    async def test_the_report_and_the_comparison_carry_the_strata(self) -> None:
        comparison = await _compared(CASES, [QUEUE])

        rendered = report_markdown(comparison.report)
        assert "By stratum" in rendered
        assert "a stratum needs at least 10 cases" in rendered
        rows = comparison.strata(reading="Accuracy")
        assert [row["arm"] for row in rows] == ["baseline", "candidate"]
        assert all(set(row["strata"]) == {"lookalike", "plain"} for row in rows)
        candidate = comparison.strata("candidate", reading="cases")
        assert len(candidate) == 1 and candidate[0]["label"] == "candidate"
        assert TOO_FEW_CASES in candidate[0]["strata"]["lookalike"]
        assert TOO_FEW_CASES not in candidate[0]["strata"]["plain"]
        assert all(summary.strata for summary in comparison.arms.values())

    async def test_unstratified_cases_have_no_strata_rows(self) -> None:
        unkinded = [{key: value for key, value in case.items() if key != "stratum"} for case in CASES]
        comparison = await _compared(unkinded, [answered])

        assert comparison.strata() == []
        assert "By stratum" not in report_markdown(comparison.report)

    async def test_an_unknown_arm_is_refused(self) -> None:
        comparison = await _compared(CASES, [QUEUE])
        with pytest.raises(ValueError):
            comparison.strata("nobody")
