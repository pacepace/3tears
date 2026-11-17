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

Held here, through ``Eval.run`` and its continuation, each claim on its Clopper-Pearson 95% bound:

- the three first lines at eight cases (all wins, an effect near 0.3, none) and the family behind them;
- under three nulls (pass/fail once a case; the on-ramp's shape, apps as cases with three trials and an arm-by-app
  spread; a skewed null whose mean difference is zero but whose differences are not symmetric), a single read at
  twelve cases and a comparison continued to three reads, each add sized from the reading, separate at most α;
- under those continuations the evidence on either side ever reaches 10 at most one time in ten (Ville);
- the interval on the primary covers the true difference at least the 90% it states.

The long simulations are ``long_simulation``, each with a smoke shard in the unit suite; the pool's ``simulation``
suite runs them, one request per test (``.testament.toml`` says how). Each shard's count is pinned as seeded, so a
change to the read that moves one is seen; the bound is the claim.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

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
from threetears.evals.storage import InMemoryDocumentStore

#: The pass rate's range, which the per-case shares and their differences are read on.
_RATE = (0.0, 1.0)

#: The question every comparison here asks, so its first grader is the primary.
_QUESTION = "Does v2 pass more often than v1?"

#: Which null a simulated newcomer runs under.
_Null = Literal["binary", "onramp", "skewed"]


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
            "smalln: Passed score: v2 ahead of v1, not decided. Over 8 paired cases, v2 is 29 points ahead of v1. The "
            "interval allows v2 to be as much as 24 points worse or 60 better. The evidence that v2 is better is "
            "moderate (3.6). Of 24 trials each, Passed score was met on 8 for v1 and 15 for v2.",
        ),
        (
            3,
            [1, 2, 1, 0, 2, 1, 1, 0],
            [2, 1, 1, 1, 2, 0, 2, 0],
            "smalln: Passed score: v2 ahead of v1, not decided. Over 8 paired cases, v2 is 4 points ahead of v1. The "
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
    assert printed.startswith("smalln: Exploratory: Passed score: v2 ahead of v1, not decided. Over 8 paired cases")
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
    result = await Eval(_cases(n), arms, [_passed, _guarded], k=1, name="two", question=_QUESTION).run(
        store=":memory:"
    )
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


# --- under the null, through the product -----------------------------------------------------------------------------


def _share(null: _Null, tag: str, index: int) -> tuple[float, float] | tuple[int, int]:
    """Case ``index``'s draw under ``null``, fixed by the newcomer's tag alone so a continued run reads it again
    unchanged: the two arms' pass shares (binary, on-ramp), or their pass counts of three (skewed)."""
    draw = random.Random(f"{tag}-case-{index}")
    if null == "binary":
        base = draw.uniform(0.1, 0.5)
        return base, base
    if null == "onramp":
        base = draw.uniform(0.1, 0.6)
        return (
            min(1.0, max(0.0, base + draw.gauss(0.0, 0.1))),
            min(1.0, max(0.0, base + draw.gauss(0.0, 0.1))),
        )
    if draw.random() < 0.25:
        return 3, 0
    control = draw.randrange(3)
    return control, control + 1


def _drawn(null: _Null, tag: str, arm: int, calls: Counter[tuple[int, int]]) -> Callable[[Mapping[str, Any]], Any]:
    """Arm ``arm``'s runner: each call a fresh seeded draw at its share of the case, or its count's next call."""

    def run(case: Mapping[str, Any]) -> bool:
        index = int(case["n"])
        calls[(index, arm)] += 1
        held = _share(null, tag, index)[arm]
        if isinstance(held, int):
            return calls[(index, arm)] <= held
        return random.Random(f"{tag}-{arm}-{index}-{calls[(index, arm)]}").random() < held

    run.__name__ = f"v{arm + 1}"
    return run


def _k(null: _Null) -> int:
    return 1 if null == "binary" else 3


def _primary(result: Result) -> Any:
    family = _family(result)
    return next(comparison for comparison in family.comparisons if comparison.name == family.primary)


def _decided(result: Result) -> bool:
    """Whether any comparison of the read is decided either way."""
    return any(
        comparison.verdict in ("improved", "regressed")
        for family in result.report.bundle.multiple_comparisons.families
        for comparison in family.comparisons
    )


async def _read_once(null: _Null, tag: str, n: int) -> bool:
    """One newcomer's single read of ``n`` cases under ``null``: whether it separates."""
    calls: Counter[tuple[int, int]] = Counter()
    arms = {"v1": _drawn(null, tag, 0, calls), "v2": _drawn(null, tag, 1, calls)}
    result = await Eval(_cases(n), arms, [_passed], k=_k(null), name="null", question=_QUESTION).run(
        store=InMemoryDocumentStore(), scope="s"
    )
    return _decided(result)


class Continued:
    """What one continued newcomer came to: whether a read separated, and whether the evidence on each side ever
    reached the strong step at a read."""

    def __init__(self, separated: bool, strong_better: bool, strong_worse: bool) -> None:
        self.separated = separated
        self.strong_better = strong_better
        self.strong_worse = strong_worse


async def _continued(null: _Null, tag: str) -> Continued:
    """One newcomer under ``null`` who reads at eight cases and, while nothing separates, appends cases and runs the
    comparison again into the same store, at most three reads: each add four cases plus up to sixteen more the nearer
    the primary's p came to zero. Every read is the product's continuation (only appended cases run)."""
    calls: Counter[tuple[int, int]] = Counter()
    arms = {"v1": _drawn(null, tag, 0, calls), "v2": _drawn(null, tag, 1, calls)}
    store = InMemoryDocumentStore()
    n = 8
    strong = EVIDENCE_STEPS[1]
    better = worse = False
    for _ in range(3):
        result = await Eval(_cases(n), arms, [_passed], k=_k(null), name="null", question=_QUESTION).run(
            store=store, scope="s"
        )
        primary = _primary(result)
        better = better or (primary.evidence_better or 0.0) >= strong
        worse = worse or (primary.evidence_worse or 0.0) >= strong
        if _decided(result):
            return Continued(True, better, worse)
        p = primary.p_adjusted if primary.p_adjusted is not None else 1.0
        n += 4 + int(16 * (1.0 - min(1.0, p)))
    return Continued(False, better, worse)


#: Newcomers per single-read shard, and how many of each null's shards separated, as seeded. Every read completes and
#: each case's draw is fixed by its tag, so each shard reproduces its count exactly.
_ONCE_PER_SHARD = 200
_ONCE_SEPARATED: dict[_Null, tuple[int, ...]] = {"binary": (1, 1), "onramp": (0, 0), "skewed": (0, 0)}

#: Continued newcomers per shard, and per null each shard's (separated, evidence for v2 ever strong, evidence against
#: ever strong), as seeded.
_CONTINUED_PER_SHARD = 150
_CONTINUED: dict[_Null, tuple[tuple[int, int, int], ...]] = {
    "binary": ((0, 0, 0),) * 10,
    "onramp": ((0, 0, 0),) * 10,
    "skewed": ((0, 0, 0),) * 10,
}

#: The cases a single read reads.
_ONCE_AT = 12


def _shards(per_null: Mapping[_Null, Sequence[object]]) -> list[Any]:
    """Every (null, shard), each ``long_simulation``, which the pool's ``simulation`` suite runs one request per test;
    the unit suite's smoke check is :func:`test_a_few_newcomers_under_each_null_read_through_the_product`."""
    return [
        pytest.param(null, shard, marks=pytest.mark.long_simulation, id=f"{null}-{shard}")
        for null, shards in per_null.items()
        for shard in range(len(shards))
    ]


@pytest.mark.parametrize("null", ["binary", "onramp", "skewed"])
async def test_a_few_newcomers_under_each_null_read_through_the_product(null: _Null) -> None:
    """The unit suite's smoke check of the simulations below: a few newcomers under each null, read once and continued,
    through the product; at most one of them separates, which at a rate of α happens with probability above 0.9."""
    once = [await _read_once(null, f"smoke-once-{null}-{index}", _ONCE_AT) for index in range(4)]
    continued = [await _continued(null, f"smoke-continued-{null}-{index}") for index in range(2)]
    assert sum(once) + sum(came.separated for came in continued) <= 1


@pytest.mark.timeout(900)
@pytest.mark.parametrize(("null", "shard"), _shards(_ONCE_SEPARATED))
async def test_a_single_read_under_the_null_separates_as_seeded(null: _Null, shard: int) -> None:
    """A shard of seeded newcomers, each reading twelve cases once (:func:`_read_once`): as many separate as pinned."""
    separated = 0
    for index in range(_ONCE_PER_SHARD):
        separated += await _read_once(null, f"once-{null}-{shard}-{index}", _ONCE_AT)
    assert separated == _ONCE_SEPARATED[null][shard]


@pytest.mark.timeout(900)
@pytest.mark.parametrize(("null", "shard"), _shards(_CONTINUED))
async def test_a_continued_read_under_the_null_separates_and_grows_its_evidence_as_seeded(
    null: _Null, shard: int
) -> None:
    """A shard of seeded newcomers, each continuing one comparison up to three reads (:func:`_continued`): as many
    separate, and as many see the evidence on each side reach the strong step, as pinned."""
    tally = [0, 0, 0]
    for index in range(_CONTINUED_PER_SHARD):
        came = await _continued(null, f"continued-{null}-{shard}-{index}")
        tally[0] += came.separated
        tally[1] += came.strong_better
        tally[2] += came.strong_worse
    assert tuple(tally) == _CONTINUED[null][shard]


@pytest.mark.parametrize("null", ["binary", "onramp", "skewed"])
def test_under_each_null_the_read_separates_at_most_alpha_and_the_evidence_is_strong_at_most_one_time_in_ten(
    null: _Null,
) -> None:
    """The claims, read on the shards' seeded counts (each reproduced by its shard test above), on each Clopper-Pearson
    95% upper bound: a single read at twelve cases separates at most α; a comparison continued to three reads, each add
    sized from the reading, separates at most α; and over those reads the evidence for v2, and against it, each ever
    reaches 10 at most one time in ten (Ville's inequality bounds it at 1/10 for an e-value read at any stops)."""
    once = exact_rate_interval(sum(_ONCE_SEPARATED[null]), _ONCE_PER_SHARD * len(_ONCE_SEPARATED[null]))
    replicates = _CONTINUED_PER_SHARD * len(_CONTINUED[null])
    sums = [sum(shard[column] for shard in _CONTINUED[null]) for column in range(3)]
    continued, better, worse = (exact_rate_interval(count, replicates) for count in sums)
    assert once is not None and continued is not None and better is not None and worse is not None
    said = f"{null}: once {once}, continued {continued}, strong for {better}, strong against {worse}"
    assert once[1] <= SIGNIFICANCE_ALPHA, said
    assert continued[1] <= SIGNIFICANCE_ALPHA, said
    assert better[1] <= 1 / EVIDENCE_STEPS[1] and worse[1] <= 1 / EVIDENCE_STEPS[1], said


# --- the interval ----------------------------------------------------------------------------------------------------

#: The effect the coverage is read at, on the pass rate: v2 passes each case 0.3 more often than v1.
_EFFECT = 0.3

#: Newcomers per coverage shard, and how many of each shard's intervals missed the true difference, as seeded.
_COVERAGE_PER_SHARD = 200
_COVERAGE_MISSED: tuple[int, ...] = (3, 2)


async def _covers(tag: str, n: int) -> bool:
    """One newcomer's single read of ``n`` pass/fail cases, v1 passing at a share drawn from 0.1 to 0.5 and v2 at that
    plus :data:`_EFFECT`: whether the primary's interval holds the true difference."""
    calls: Counter[tuple[int, int]] = Counter()

    def arm(which: int) -> Callable[[Mapping[str, Any]], Any]:
        def run(case: Mapping[str, Any]) -> bool:
            index = int(case["n"])
            calls[(index, which)] += 1
            base = random.Random(f"{tag}-case-{index}").uniform(0.1, 0.5)
            share = base + (_EFFECT if which else 0.0)
            return random.Random(f"{tag}-{which}-{index}-{calls[(index, which)]}").random() < share

        run.__name__ = f"v{which + 1}"
        return run

    result = await Eval(_cases(n), {"v1": arm(0), "v2": arm(1)}, [_passed], k=1, name="cover", question=_QUESTION).run(
        store=InMemoryDocumentStore(), scope="s"
    )
    interval = _primary(result).interval
    assert interval is not None
    return interval[0] <= _EFFECT <= interval[1]


@pytest.mark.long_simulation
@pytest.mark.timeout(900)
@pytest.mark.parametrize("shard", range(len(_COVERAGE_MISSED)))
async def test_the_primary_s_interval_misses_the_true_difference_as_seeded(shard: int) -> None:
    """A shard of seeded newcomers reading sixteen cases at a true difference of 0.3 (:func:`_covers`): as many
    intervals miss it as pinned."""
    missed = 0
    for index in range(_COVERAGE_PER_SHARD):
        missed += not await _covers(f"cover-{shard}-{index}", 16)
    assert missed == _COVERAGE_MISSED[shard]


def test_the_primary_s_interval_covers_the_true_difference_at_least_the_ninety_percent_it_states() -> None:
    """The claim, on the shards' seeded counts: the Clopper-Pearson 95% upper bound on the share of intervals that miss
    the true difference is at most 10%, the 90% the interval states."""
    replicates = _COVERAGE_PER_SHARD * len(_COVERAGE_MISSED)
    missed = exact_rate_interval(sum(_COVERAGE_MISSED), replicates)
    assert missed is not None
    assert missed[1] <= 1 - 0.90 + 1e-12, f"missed {sum(_COVERAGE_MISSED)} of {replicates}: {missed}"
    assert math.isclose(1 - 0.90, 2 * SIGNIFICANCE_ALPHA)
