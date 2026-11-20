"""The advice beside the proof: what to do on a comparison, read off the evidence its bet holds.

Fast guards on the bars and the rule (each fails if either moves), the first screens it opens with, where it is
stored and served, the ``undo`` input and its timing, and that the gate stays on the proof. The promise itself (a
candidate that is not better is advised at most one time in three, or ten) is certified at size by
``stats_certificate.py``'s ``advice.no-better.*`` claims; here it is held on the rule's own arithmetic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import pytest

from threetears.evals import Eval, Result
from threetears.evals.analysis.report.serialize_md import report_markdown
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.analysis.campaigns import get_campaign
from threetears.evals.analysis.report.words import NO_ADVICE_SENTENCE
from threetears.evals.analysis.stats import ADVICE_BARS, EVIDENCE_STEPS, advice_path
from threetears.evals.analysis.style import GLOSSARY, UNDO_LOWERING_IGNORED_SENTENCE

from packages.evals.tests import advice_nulls

_QUESTION = "Does v2 pass more often than v1?"

#: The held-out-1 on-ramp run's primary, per app (control, candidate) trials met of three, in the plan's order.
_ONRAMP = [
    (0, 3), (2, 3), (3, 3), (3, 3), (2, 3), (1, 2), (3, 3), (2, 1), (3, 3), (3, 3), (3, 3), (1, 2),
    (1, 2), (0, 1), (3, 3), (3, 3), (3, 3), (3, 3), (0, 3), (1, 3), (2, 3), (3, 2), (2, 2), (0, 3),
]  # fmt: skip


def test_the_bars_are_the_evidence_words_own_steps() -> None:
    """Moderate (3) where the switch is easy to undo, strong (10) where it is hard: never drifting from the words."""
    assert dict(ADVICE_BARS) == {"easy": 3.0, "hard": 10.0}
    assert (ADVICE_BARS["easy"], ADVICE_BARS["hard"]) == EVIDENCE_STEPS


def test_the_promise_is_said_where_the_advice_is_defined() -> None:
    assert GLOSSARY["advice"] == (
        "What to do beside the proof: switch while the candidate leads and its evidence has reached moderate (strong "
        "where the switch is hard to undo), and keep otherwise."
    )
    assert GLOSSARY["advice promise"] == (
        "However often anyone looks, the advice says switch to a candidate no better (with a margin, worse by more "
        "than it) at most one time in three (one in ten)."
    )


def test_all_wins_switch_once_the_evidence_reaches_each_bar() -> None:
    """Eight wins of eight: moderate evidence from the third case, strong from the fifth."""
    assert advice_path([0] * 8, [1] * 8, value_range=(0, 1), bar=3) == [False, False] + [True] * 6
    assert advice_path([0] * 8, [1] * 8, value_range=(0, 1), bar=10) == [False] * 4 + [True] * 4


def test_the_advice_stands_once_reached_and_withdraws_when_the_arm_is_no_longer_ahead() -> None:
    """Read off the path, as a decision is: six wins reach the bar; ten losses after them bring the arm level at twelve
    cases, and the advice withdraws there."""
    control, candidate = [0] * 6 + [1] * 10, [1] * 6 + [0] * 10
    said = advice_path(control, candidate, value_range=(0, 1), bar=3)
    assert said == [False, False] + [True] * 9 + [False] * 5


def test_a_margin_reads_the_evidence_of_no_worse_than_it() -> None:
    """Forty agreeing cases are evidence the arm is no worse than five points below the control, from the 32nd case;
    against zero they are no evidence it is better."""
    same = [1] * 20 + [0] * 20
    with_margin = advice_path(same, same, value_range=(0, 1), bar=3, margin=0.05)
    assert with_margin is not None and with_margin.index(True) == 31 and with_margin[-1]
    assert advice_path(same, same, value_range=(0, 1), bar=3) == [False] * 40


def test_the_on_ramp_result_reads_switch_at_either_bar() -> None:
    control = [Fraction(a, 3) for a, _ in _ONRAMP]
    candidate = [Fraction(b, 3) for _, b in _ONRAMP]
    for bar in ADVICE_BARS.values():
        path = advice_path(control, candidate, value_range=(0, 1), bar=bar)
        assert path is not None and path[-1], bar


def test_an_arm_eight_cases_ahead_on_moderate_evidence_reads_switch_only_where_easy_to_undo() -> None:
    """The +0.3 sample at eight cases (v1 met 8 of 24 trials, v2 15): its evidence reached 3.5, below strong."""
    control = [Fraction(x, 3) for x in (0, 1, 1, 0, 2, 1, 2, 1)]
    candidate = [Fraction(x, 3) for x in (3, 3, 2, 1, 2, 1, 1, 2)]
    easy = advice_path(control, candidate, value_range=(0, 1), bar=ADVICE_BARS["easy"])
    hard = advice_path(control, candidate, value_range=(0, 1), bar=ADVICE_BARS["hard"])
    assert easy == [False] * 3 + [True] * 5
    assert hard == [False] * 8


@pytest.mark.parametrize(("bar", "margin"), [(1.0, 0.0), (0.5, 0.0), (3.0, -0.1)])
def test_an_unusable_bar_or_margin_is_refused(bar: float, margin: float) -> None:
    with pytest.raises(ValueError):
        advice_path([0, 1], [1, 1], value_range=(0, 1), bar=bar, margin=margin)


def test_a_small_seeded_null_reads_as_the_certificate_counted_it() -> None:
    """A smoke of the certificate's claims, 60 not-better comparisons a family read after every case: the counts the
    real path gave on these seeds, so a change to the rule, the bars or the families' draws turns this red before a
    re-certification is due. (Sixty is too few to hold the promise's bound; the certificate holds it at 4,000.)"""
    counted = {family: advice_nulls.count(family, range(60)) for family in advice_nulls.FAMILIES}
    assert {family: (totals["ever3"], totals["ever10"]) for family, totals in counted.items()} == _SMOKE


#: The smoke's counts (ever advised at 3, at 10) by family, as advice_nulls.count gave them on seeds 0-59.
_SMOKE = {"passfail": (20, 4), "onramp": (13, 4), "swing": (15, 7), "skewed": (18, 4)}


# --- the first screen ---------------------------------------------------------------------------------------------


def _eval(control: Sequence[int], candidate: Sequence[int], name: str, **declared: Any) -> Eval:
    cases = [{"id": f"c{index:02d}", "a": a, "b": b} for index, (a, b) in enumerate(zip(control, candidate))]

    def v1(case: dict[str, Any]) -> int:
        return int(case["a"])

    def v2(case: dict[str, Any]) -> int:
        return int(case["b"])

    def passed(case: dict[str, Any], out: int) -> bool:
        return out == 1

    return Eval(cases, {"v1": v1, "v2": v2}, passed, k=1, name=name, **declared)


_HALF = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]


def test_all_wins_opens_on_switch_and_its_proof_below_it() -> None:
    result = asyncio.run(_eval([0] * 8, [1] * 8, "all-wins", question=_QUESTION).run(store=":memory:"))
    first, second = result.render().splitlines()[:2]
    assert first == "all-wins: Advice: switch to v2. At worst it is 21 points ahead of v1."
    assert second.startswith("all-wins: Passed score: v2 better than v1, decided at 6 cases when the evidence reached")
    assert [row["advice"] for row in result.contrasts()] == ["switch"]


def test_moderate_evidence_switches_where_easy_and_keeps_where_hard_to_undo() -> None:
    ahead = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    easy = asyncio.run(_eval(_HALF, ahead, "moderate", question=_QUESTION).run(store=":memory:"))
    hard = asyncio.run(_eval(_HALF, ahead, "moderate-hard", question=_QUESTION, undo="hard").run(store=":memory:"))
    assert easy.render().splitlines()[0] == "moderate: Advice: switch to v2. At worst it is 11 points behind v1."
    assert hard.render().splitlines()[0] == (
        "moderate-hard: Advice: keep v1. v2 is ahead, but a switch that is hard to undo needs evidence of 10 and this "
        "is moderate (7.8)."
    )
    # The proof line says its proof is pending, and what proving it needs.
    proof = easy.render().splitlines()[1]
    assert "v2 ahead of v1, proof pending." in proof
    assert "The evidence that v2 is better is moderate (7.8); proving it needs evidence of 22." in proof


def test_a_level_arm_keeps_the_control_and_never_says_there_is_not_enough_to_say() -> None:
    level = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    result = asyncio.run(_eval(_HALF, level, "level", question=_QUESTION).run(store=":memory:"))
    first = result.render().splitlines()[0]
    assert first == "level: Advice: keep v1. v2 is level with it."
    assert "not enough" not in result.render()


def test_a_gate_reads_its_advice_against_its_margin_and_says_the_one_caveat() -> None:
    same = [1] * 20 + [0] * 20
    result = asyncio.run(
        _eval(same, same, "gate", question="Is v2 no worse than v1?", margin=0.05).run(store=":memory:")
    )
    assert result.render().splitlines()[0] == (
        "gate: Advice: switch to v2. At worst it is 10 points behind v1, past the 5 points you allow."
    )


def test_the_gate_stays_on_the_proof_whatever_the_advice_says() -> None:
    """Advised to switch, not proven: the gate reads the proof, undecided, never passed."""
    ahead = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    result = asyncio.run(_eval(_HALF, ahead, "gated", question=_QUESTION).run(store=":memory:"))
    assert [row["advice"] for row in result.contrasts()] == ["switch"]
    assert result.gate().outcome == "undecided"
    assert not result.gate().failed


def test_a_bar_raised_after_the_results_is_honoured_and_says_nothing() -> None:
    """A "hard" declared after the results can only make the advice say switch less often, so it is read: the moderate
    evidence that switched at 3 keeps at 10."""
    ahead = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]

    async def read() -> tuple[str, str, list[float | None]]:
        first = await _eval(_HALF, ahead, "raised", question=_QUESTION).run(store=":memory:")
        again = await first.against("v1", undo="hard")
        bars = [family.advice_bar for family in again.report.bundle.multiple_comparisons.families]
        return first.render(), again.render(), bars

    before, after, bars = asyncio.run(read())
    assert before.splitlines()[0] == "raised: Advice: switch to v2. At worst it is 11 points behind v1."
    assert after.splitlines()[0].startswith(
        "raised, against v1: Advice: keep v1. v2 is ahead, but a switch that is hard"
    )
    assert bars == [10.0]
    assert UNDO_LOWERING_IGNORED_SENTENCE not in after


def test_a_bar_lowered_after_the_results_is_ignored_and_the_print_and_the_page_say_so() -> None:
    """An "easy" declared after the results over an earlier "hard" would lower the bar once the evidence was seen: the
    "hard" stands, the advice reads 10, and the print and the family's disclosure, which the page states, say so."""
    ahead = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]

    async def read() -> tuple[str, str, float | None, str]:
        first = await _eval(_HALF, ahead, "lowered", question=_QUESTION, undo="hard").run(store=":memory:")
        again = await first.against("v1", undo="easy")
        (family,) = again.report.bundle.multiple_comparisons.families
        page = report_markdown(again.report)
        return first.render(), again.render(), family.advice_bar, page

    before, after, bar, page = asyncio.run(read())
    assert before.splitlines()[0].startswith("lowered: Advice: keep v1. v2 is ahead, but a switch that is hard")
    assert after.splitlines()[0].startswith(
        "lowered, against v1: Advice: keep v1. v2 is ahead, but a switch that is hard"
    )
    assert bar == 10.0
    assert UNDO_LOWERING_IGNORED_SENTENCE in after.splitlines()
    assert UNDO_LOWERING_IGNORED_SENTENCE in page


def test_an_easy_declared_after_results_that_declared_nothing_lowers_nothing_and_says_nothing() -> None:
    ahead = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]

    async def read() -> str:
        first = await _eval(_HALF, ahead, "plain", question=_QUESTION).run(store=":memory:")
        return (await first.against("v1", undo="easy")).render()

    after = asyncio.run(read())
    assert after.splitlines()[0] == "plain, against v1: Advice: switch to v2. At worst it is 11 points behind v1."
    assert UNDO_LOWERING_IGNORED_SENTENCE not in after


def _arms_eval(arms: dict[str, list[int]], name: str, **declared: Any) -> Eval:
    """An eval of several arms over ten cases, each arm passing a case where its list says 1; the first is the
    control."""
    cases = [{"id": f"c{index:02d}", **{arm: values[index] for arm, values in arms.items()}} for index in range(10)]

    def runner(arm: str) -> Any:
        def run(case: dict[str, Any]) -> int:
            return int(case[arm])

        run.__name__ = arm
        return run

    def passed(case: dict[str, Any], out: int) -> bool:
        return out == 1

    return Eval(cases, {arm: runner(arm) for arm in arms}, passed, k=1, name=name, **declared)


#: An arm ahead of the control on moderate evidence: advised switch alone, at 3; kept among three arms, at 9.
_AHEAD_SOME = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]

#: An arm level with the control.
_LEVEL = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]


@pytest.mark.parametrize(("undo", "bar"), [("easy", 9.0), ("hard", 30.0)])
def test_several_arms_raise_the_bar_by_their_count(undo: Literal["easy", "hard"], bar: float) -> None:
    """Three arms against one control: the bar is three times the switch's own, so the chance any arm that is not
    better is advised stays one in three (ten). v3, which alone is advised switch at 3, is kept at 9."""
    arms = {"v1": _HALF, "v2": _LEVEL, "v3": _AHEAD_SOME, "v4": _LEVEL}
    result = asyncio.run(_arms_eval(arms, f"three-{undo}", question=_QUESTION, undo=undo).run(store=":memory:"))
    (family,) = result.report.bundle.multiple_comparisons.families
    assert family.advice_bar == bar
    assert sorted((row["contrast"], row["advice"]) for row in result.contrasts()) == [
        ("v2", "keep"),
        ("v3", "keep"),
        ("v4", "keep"),
    ]
    if undo == "easy":
        assert result.render().splitlines()[0] == (
            "three-easy: Advice: keep v1. v3 is ahead, but a switch needs evidence of 9 and this is moderate (4.6)."
        )
    alone = asyncio.run(_arms_eval({"v1": _HALF, "v3": _AHEAD_SOME}, "one", question=_QUESTION).run(store=":memory:"))
    assert [(row["contrast"], row["advice"]) for row in alone.contrasts()] == [("v3", "switch")]


def _faded(control: list[int], arm: list[int], name: str, **declared: Any) -> Result:
    """A comparison read at its first eight cases, then continued to all of them in one store."""

    async def read() -> Result:
        store = InMemoryDocumentStore()
        await _eval(control[:8], arm[:8], name, **declared).run(store=store, scope="s")
        return await _eval(control, arm, name, **declared).run(store=store, scope="s")

    return asyncio.run(read())


def test_a_decision_of_better_leads_the_advice_after_the_lead_fades() -> None:
    """Eight wins decide v2 better at six cases; ten losses after them leave v2 behind. The decision stands, so the
    advice switches with it, saying why, and the gate, which reads the decision, passes beside it."""
    result = _faded([0] * 8 + [1] * 10, [1] * 8 + [0] * 10, "fade", question=_QUESTION)
    first, proof = result.render().splitlines()[:2]
    assert first == "fade: Advice: switch to v2. It was proven better than v1, decided at 6 cases."
    assert proof.startswith("fade: Passed score: v2 better than v1, decided at 6 cases when the evidence reached 24.")
    assert [(row["advice"], row["outcome"]) for row in result.contrasts()] == [("switch", "improved")]
    assert result.gate().outcome == "passed"


def test_a_decision_of_worse_leads_the_advice_after_the_arm_recovers() -> None:
    """On a question with a margin, eight losses decide v2 worse at six cases; thirty wins after them carry v2's
    evidence past the bar, which alone would say switch. The decision stands, so the advice keeps v1, saying why, and
    the gate fails on the regression beside it."""
    result = _faded([1] * 8 + [0] * 30, [0] * 8 + [1] * 30, "rise", question="Is v2 no worse than v1?", margin=0.05)
    (family,) = result.report.bundle.multiple_comparisons.families
    (comparison,) = family.comparisons
    assert family.advice_bar is not None
    assert (comparison.evidence_better or 0.0) >= family.advice_bar
    first, proof = result.render().splitlines()[:2]
    assert first == "rise: Advice: keep v1. v2 was proven worse than v1, decided at 6 cases."
    assert proof.startswith("rise: Passed score: v2 worse than v1, decided at 6 cases when the evidence reached 24.")
    assert [(row["advice"], row["outcome"]) for row in result.contrasts()] == [("keep", "regressed")]
    assert result.gate().failed


def test_with_several_arms_an_arm_the_proof_decided_better_is_advised_switch_whatever_its_evidence() -> None:
    """Three arms, hard to undo, so the bar is 30. v3 leads for eleven cases and then plays level; Holm decides it
    better at its last step, where its evidence (24) sits under the bar. The decision leads the advice for every arm, so
    no row says keep beside an arm the proof decided better, and the gate, which passes, never contradicts it."""
    n = 24
    control = [index % 2 for index in range(n)]
    arms = {"v1": control, "v2": [1] * n, "v3": [1] * 11 + control[11:], "v4": [1] * n}
    cases = [{"id": f"c{index:02d}", **{arm: values[index] for arm, values in arms.items()}} for index in range(n)]

    def runner(arm: str) -> Any:
        def run(case: dict[str, Any]) -> int:
            return int(case[arm])

        run.__name__ = arm
        return run

    def passed(case: dict[str, Any], out: int) -> bool:
        return out == 1

    ev = Eval(cases, {arm: runner(arm) for arm in arms}, passed, k=1, name="decided", question=_QUESTION, undo="hard")
    result = asyncio.run(ev.run(store=":memory:"))
    (family,) = result.report.bundle.multiple_comparisons.families
    assert family.advice_bar == 30.0
    rows = sorted((row["contrast"], row["outcome"], row["advice"]) for row in result.contrasts())
    assert rows == [("v2", "improved", "switch"), ("v3", "improved", "switch"), ("v4", "improved", "switch")]
    # v3 is the one decided on the least evidence, under the bar: the decision, not its evidence, switches it.
    least = min(comparison.evidence_better or 0.0 for comparison in family.comparisons if comparison.advice)
    assert least < 30
    assert result.gate().outcome == "passed"


def test_undo_is_stored_on_the_aspect_only_where_hard() -> None:
    ahead = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]

    def aspect(undo: Literal["easy", "hard"]) -> dict[str, Any]:
        result = asyncio.run(_eval(_HALF, ahead, f"stored-{undo}", question=_QUESTION, undo=undo).run(store=":memory:"))
        campaign = get_campaign(result.host.storage, result.campaign_id, result.scope_id)
        return campaign.aspects[0].model_dump(mode="json")

    easy, hard = aspect("easy"), aspect("hard")
    assert "undo" not in easy
    assert hard["undo"] == "hard"


def test_an_unknown_undo_is_refused() -> None:
    with pytest.raises(ValueError, match='undo= is "easy"'):
        _eval(_HALF, _HALF, "refused", question=_QUESTION, undo="maybe")  # type: ignore[arg-type]


def test_a_comparison_continued_with_another_undo_is_a_new_comparison(tmp_path: Path) -> None:
    ahead = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    store = tmp_path / "evals.sqlite"

    async def twice() -> str:
        await _eval(_HALF, ahead, "continued", question=_QUESTION).run(store=store)
        again = await _eval(_HALF, ahead, "continued", question=_QUESTION, undo="hard").run(store=store)
        return again.render()

    assert "how hard the switch is to undo changed" in asyncio.run(twice())


def test_an_engine_campaign_gives_no_advice_and_says_why() -> None:
    """The same runs filed by an operator are an engine campaign, read once at a fixed count, which keeps no running
    evidence: its first line says there is no advice and why, and leads on the proof under it; its rows carry none."""
    cases = [_HALF * 2, [1, 1, 1, 1, 1, 1, 0, 1, 1, 1] * 2]

    async def engine() -> tuple[list[str], list[Any]]:
        quick = await _eval(*cases, "engine", question=_QUESTION).run(store=":memory:")
        filed = await quick.against("v1", created_by="an operator")
        return filed.render().splitlines()[:2], [row["advice"] for row in filed.contrasts()]

    (first, proof), advice = asyncio.run(engine())
    assert first == f"engine, against v1: {NO_ADVICE_SENTENCE}"
    assert proof.startswith("engine, against v1: Passed score: v2 better than v1.")
    assert advice == [None]
