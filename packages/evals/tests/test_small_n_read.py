"""A small comparison gets a fair read (owner ruling, 2026-11-17; the case for it, ``.prawduct/smalln-report.md``).

A quick-path comparison whose question is asked is decided on the question's own reading, its first grader's, alone:

- the mixture bet's stakes are weighted toward the larger ones, so a few cases can grow it;
- the bet is staked nine tenths toward the better side the question asks about, the rest the other way;
- the primary is tested at α, with no correction across readings, and every other reading is secondary: shown with its
  estimate, interval and evidence, and decided on nothing;
- the overall pass rate of an eval whose one check is its pass rate is that check, so it is not tested twice.

Every bet-read contrast carries graded evidence: the e-value its bets hold now on each side, said weak, moderate or
strong. The printed lead says where the arm stands, its estimate, the interval and that evidence, never "cannot tell
apart".

Held here: the three first lines at eight cases (all wins, an effect near 0.3, none) and the family behind them, through
``Eval.run``; and a smoke check that a few newcomers under the null read, through the product, as the restatement over
the bet alone reads them (``null_newcomers.py``).

The claims under the null are the statistics certificate's (``stats_certificate.py``), each on its Clopper-Pearson 95%
bound: under three nulls (pass/fail once a case; the on-ramp's shape, apps as cases with three trials and an arm-by-app
spread; a skewed null whose mean difference is zero but whose differences are not symmetric), a single read at twelve
cases and a comparison continued to five reads, each add sized from the reading, separate at most α; under those
continuations the evidence on either side ever reaches 10 at most one time in ten (Ville); and the interval on the
primary covers the true difference at least the 90% it states. ``test_stats_guards.py`` holds the fast guards that
fail for a wrong formula, and ``docs/testing.md`` says how to re-certify.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from threetears.evals import Eval, Result
from threetears.evals.analysis import ComparisonFamily
from threetears.evals.analysis.stats import (
    DECLARED_DIRECTION_SHARE,
    EVIDENCE_STEPS,
    SIGNIFICANCE_ALPHA,
    bet_evidence,
    bounded_separation_p,
    exact_rate_interval,
)
from threetears.evals.analysis.style import evidence_grade

from packages.evals.tests.null_newcomers import (
    Null,
    came_to,
    same_reads,
    small_continued,
    small_continued_product,
    small_covers,
    small_covers_product,
    small_once,
    small_once_product,
)

#: The pass rate's range, which the per-case shares and their differences are read on.
_RATE = (0.0, 1.0)

#: The question every comparison here asks, so its first grader is the primary.
_QUESTION = "Does v2 pass more often than v1?"


# --- the bet ---------------------------------------------------------------------------------------------------------


def test_all_wins_decide_from_six_cases_and_the_tilt_costs_a_tenth_of_the_capital() -> None:
    """Every case a win: the even hedge reads p 0.0271 at eight cases and the declared direction's 0.0150, which is
    1/0.9 of the one-sided bet's 0.0135; the declared direction decides from six cases (0.0471), where the even, evenly
    weighted hedge needed ten. The upward evidence at eight is the one-sided capital, 74."""
    for n, even, tilted in ((6, 0.0847, 0.0471), (8, 0.0271, 0.0150), (10, 0.0084, 0.0047)):
        control, arm = [0.0] * n, [1.0] * n
        assert bounded_separation_p(control, arm, paired=True, value_range=_RATE, mixture=True) == pytest.approx(
            even, abs=5e-5
        )
        assert bounded_separation_p(
            control, arm, paired=True, value_range=_RATE, mixture=True, toward=DECLARED_DIRECTION_SHARE
        ) == pytest.approx(tilted, abs=5e-5)
    evidence = bet_evidence([0.0] * 8, [1.0] * 8, value_range=_RATE)
    assert evidence is not None
    assert evidence[0] == pytest.approx(73.83, abs=0.01) and evidence[1] < 0.05
    assert DECLARED_DIRECTION_SHARE * evidence[0] >= 1 / SIGNIFICANCE_ALPHA


def test_a_declared_direction_is_a_share_of_the_mixture_s_capital() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        bounded_separation_p([0.0, 1.0], [1.0, 1.0], paired=True, value_range=_RATE, mixture=True, toward=1.0)
    with pytest.raises(ValueError, match="with mixture"):
        bounded_separation_p([0.0, 1.0], [1.0, 1.0], paired=True, value_range=_RATE, toward=0.9)


def test_the_evidence_is_graded_in_one_owner_s_words_at_three_and_ten() -> None:
    assert EVIDENCE_STEPS == (3.0, 10.0)
    assert [evidence_grade(e) for e in (0.2, 2.99, 3.0, 9.99, 10.0, 1e6)] == [
        "weak",
        "weak",
        "moderate",
        "moderate",
        "strong",
        "strong",
    ]


def _tilted_newcomer(rng: random.Random, null: Null) -> bool:
    """The bet alone, staked toward the asked side, over a newcomer's differences appended and kept: read at eight
    pairs and, while it does not separate, after each add of four pairs plus up to sixteen more the nearer its p came
    to zero, at most five reads. Pass/fail once a case at a share drawn from 0.1 to 0.5 on both arms, or the skewed
    null's differences (+1/3 three cases in four, -1 the fourth)."""
    diffs: list[float] = []
    n = 8
    for _ in range(5):
        while len(diffs) < n:
            if null == "skewed":
                diffs.append(-1.0 if rng.random() < 0.25 else 1 / 3)
            else:
                share = rng.uniform(0.1, 0.5)
                diffs.append(float(rng.random() < share) - float(rng.random() < share))
        control, arm = [max(0.0, -d) for d in diffs], [max(0.0, d) for d in diffs]
        p = bounded_separation_p(
            control, arm, paired=True, value_range=_RATE, mixture=True, toward=DECLARED_DIRECTION_SHARE
        )
        assert p is not None
        if p < SIGNIFICANCE_ALPHA:
            return True
        n += 4 + int(16 * (1.0 - min(1.0, p)))
    return False


@pytest.mark.parametrize(("null", "seeded_at"), [("binary", 136), ("skewed", 149)])
def test_the_tilted_bet_alone_read_up_to_five_times_separates_under_the_null_at_most_alpha(
    null: Null, seeded_at: int
) -> None:
    """The bet itself, over many newcomers (the product's own reads are simulated below, through ``Eval.run``): 5,000
    seeded newcomers a null, each reading up to five times with adds sized from the data, separate no more often than
    α on the Clopper-Pearson 95% upper bound. As seeded: 136 of 5,000 pass/fail (2.7%, upper 3.2%) and 149 under the
    skewed null (3.0%, upper 3.5%), pinned to two either side so a change to the bet that moves them is seen."""
    rng = random.Random(f"tilted-alone-{null}")
    replicates = 5000
    separated = sum(_tilted_newcomer(rng, null) for _ in range(replicates))
    bounds = exact_rate_interval(separated, replicates)
    assert bounds is not None and bounds[1] < SIGNIFICANCE_ALPHA
    assert abs(separated - seeded_at) <= 2


# --- the first screen at eight cases ---------------------------------------------------------------------------------


def _tallied(name: str, passes: Sequence[int], calls: Counter[tuple[str, int]]) -> Callable[[Mapping[str, Any]], Any]:
    """A runner that passes case ``n``'s first ``passes[n]`` calls and fails the rest."""

    def run(case: Mapping[str, Any]) -> bool:
        index = int(case["n"])
        calls[(name, index)] += 1
        return calls[(name, index)] <= passes[index]

    run.__name__ = name
    return run


def _passed(case: Mapping[str, Any], answer: Any) -> bool:
    return bool(answer)


def _cases(n: int) -> list[dict[str, Any]]:
    return [{"id": f"c{index:03d}", "n": index} for index in range(n)]


async def _tallied_result(k: int, v1: Sequence[int], v2: Sequence[int], *, question: str | None = _QUESTION) -> Result:
    calls: Counter[tuple[str, int]] = Counter()
    arms = {"v1": _tallied("v1", v1, calls), "v2": _tallied("v2", v2, calls)}
    return await Eval(_cases(len(v1)), arms, [_passed], k=k, name="smalln", question=question).run(store=":memory:")


def _family(result: Result) -> ComparisonFamily:
    (family,) = result.report.bundle.multiple_comparisons.families
    return family


@pytest.mark.parametrize(
    ("k", "v1", "v2", "line"),
    [
        (
            1,
            [0] * 8,
            [1] * 8,
            "smalln: Passed score: v2 better than v1. Over 8 paired cases, v2 is 100 points ahead of v1. The interval "
            "puts v2 between 21 and 100 points better. The evidence that v2 is better is strong (74), enough to "
            "decide. Of 8 trials each, Passed score was met on 0 for v1 and 8 for v2.",
        ),
        (
            3,
            [1, 0, 1, 2, 0, 1, 2, 1],
            [3, 1, 2, 2, 1, 3, 1, 2],
            "smalln: Passed score: v2 ahead of v1, undecided. Over 8 paired cases, v2 is 29 points ahead of v1. The "
            "interval allows v2 to be as much as 24 points worse or 60 better. The evidence that v2 is better is "
            "moderate (3.6). Of 24 trials each, Passed score was met on 8 for v1 and 15 for v2.",
        ),
        (
            3,
            [1, 2, 1, 0, 2, 1, 1, 0],
            [2, 1, 1, 1, 2, 0, 2, 0],
            "smalln: Passed score: v2 ahead of v1, undecided. Over 8 paired cases, v2 is 4 points ahead of v1. The "
            "interval allows v2 to be as much as 39 points worse or 44 better. The evidence that v2 is better is weak "
            "(1.1). Of 24 trials each, Passed score was met on 8 for v1 and 9 for v2.",
        ),
    ],
    ids=["all-wins", "about-0.3", "none"],
)
async def test_eight_cases_lead_on_where_v2_stands_its_estimate_its_interval_and_its_evidence(
    k: int, v1: list[int], v2: list[int], line: str
) -> None:
    """At eight cases the first line leads on where v2 stands, the estimate, the interval and the graded evidence, and
    never says the evidence cannot tell the arms apart; all wins decide. Behind it, one family decided on the question's
    first grader alone, at α, its interval at 90%, and no second reading of the same check."""
    result = await _tallied_result(k, v1, v2)
    printed = str(result).splitlines()[0]
    assert printed == line
    assert "cannot tell" not in printed
    family = _family(result)
    assert family.primary == "_passed" and family.family_size == 1
    assert family.interval_level == pytest.approx(0.90)
    assert [(comparison.reading, comparison.name) for comparison in family.comparisons] == [("measure", "_passed")]


async def test_with_no_question_the_read_stays_two_sided_and_still_leads_on_the_estimate() -> None:
    """With no question asked nothing is declared: the even hedge at α, its interval at 95%, and the same lead."""
    result = await _tallied_result(3, [1, 0, 1, 2, 0, 1, 2, 1], [3, 1, 2, 2, 1, 3, 1, 2], question=None)
    printed = str(result).splitlines()[0]
    assert printed.startswith("smalln: Exploratory: Passed score: v2 ahead of v1, undecided. Over 8 paired cases")
    family = _family(result)
    assert family.primary is None and family.interval_level == pytest.approx(0.95)


def _guarded(case: Mapping[str, Any], answer: Any) -> bool:
    """A second check, which the first case of every three fails on v1 alone."""
    return bool(answer) or int(case["n"]) % 3 != 0


async def test_a_second_check_is_a_secondary_reading_shown_and_never_decided() -> None:
    """Two checks: the first is the primary and is decided; the second, which v2 holds on every case and v1 misses on a
    third of them, is shown with its estimate, interval and evidence and decided on nothing, whatever its evidence. The
    pass rate, the two checks together, is a secondary reading too."""
    calls: Counter[tuple[str, int]] = Counter()
    n = 24
    arms = {"v1": _tallied("v1", [0] * n, calls), "v2": _tallied("v2", [1] * n, calls)}
    result = await Eval(_cases(n), arms, [_passed, _guarded], k=1, name="two", question=_QUESTION).run(store=":memory:")
    family = _family(result)
    assert family.primary == "_passed"
    by_name = {(comparison.reading, comparison.name): comparison for comparison in family.comparisons}
    primary = by_name[("measure", "_passed")]
    assert primary.verdict == "improved" and primary.p_adjusted is not None
    secondaries = [comparison for key, comparison in by_name.items() if key != ("measure", "_passed")]
    assert secondaries
    for comparison in secondaries:
        assert comparison.verdict == "not_separated" and comparison.p_adjusted is None
        assert comparison.evidence_better is not None and comparison.interval is not None
    assert by_name[("measure", "_guarded")].evidence_better is not None
    assert by_name[("measure", "_guarded")].evidence_better >= 1 / SIGNIFICANCE_ALPHA


# --- under the null, through the product ---------------------------------------------------------------------------


@pytest.mark.parametrize("null", ["binary", "onramp", "skewed"])
async def test_a_few_newcomers_under_each_null_read_through_the_product_as_restated(null: Null) -> None:
    """A few newcomers under each null, read once at twelve cases and continued to five reads, through the product: each
    reads, read for read, as the restatement over the bet alone reads the same answers (``null_newcomers.same_reads``),
    and at most one of them separates, which at a rate of α happens with probability above 0.9. The counts the claims
    are held on are the certificate's (``stats_certificate.py``), read on the restatement this holds to the product."""
    separated = 0
    for index in range(4):
        tag = f"smoke-once-{null}-{index}"
        product = await small_once_product(null, tag)
        assert same_reads(product, small_once(null, tag)) is None, same_reads(product, small_once(null, tag))
        separated += product[0].decided
    for index in range(2):
        tag = f"smoke-continued-{null}-{index}"
        product = await small_continued_product(null, tag)
        assert same_reads(product, small_continued(null, tag)) is None, same_reads(product, small_continued(null, tag))
        separated += came_to(product).separated
    assert separated <= 1


async def test_a_few_newcomers_read_the_primary_s_interval_as_restated() -> None:
    """At a true difference of 0.3 over sixteen cases, the primary's interval as the product reads it is the restated
    one (the even hedge's confidence sequence at 90%), newcomer for newcomer."""
    for index in range(4):
        tag = f"smoke-cover-{index}"
        product = await small_covers_product(tag)
        assert same_reads(product, small_covers(tag)) is None, same_reads(product, small_covers(tag))
