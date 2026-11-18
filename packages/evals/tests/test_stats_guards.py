"""Fast guards for the certified claims: each in seconds, each failing for a wrong formula in the bet, its tilt, its
weights, its hedge, its interval or its threshold, so every build sees a change the certificate's long counts would
only see when certified again (``stats_certificate.py``; ``docs/testing.md`` names, for each claim, its guard).

- **Exact computations.** Paired pass/fail differences are signs (a tie moves no bet), so the bet's reading of every
  sequence of up to twelve signs is enumerated, and its count of separations in each direction pinned; from them, the
  exact chance that a single read of twelve pass/fail cases under the null separates, which bounds the product's.
- **Pinned outputs.** Golden p values (staked toward the asked side, evenly, and the other way), e-values and intervals
  at 90% and 95% for fixed samples, read through the statistics functions and through ``Eval.run``'s primary.
- **Small seeded runs.** The first shard of each restated claim, pinned at the counts the product gave for it when it
  ran through ``Eval.run`` shard by shard.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from threetears.evals import Eval
from threetears.evals.analysis.stats import (
    DECLARED_DIRECTION_SHARE,
    SIGNIFICANCE_ALPHA,
    bet_evidence,
    bounded_difference_interval,
    bounded_separation_p,
)

from packages.evals.tests.null_newcomers import (
    PLANNED_ADD,
    PLANNED_READS,
    PLANNED_START,
    QUESTION,
    RATE,
    came_to,
    horizon_crossed,
    missed_side,
    rereadable,
    small_continued,
    small_covers,
    small_once,
)

# --- exact: every sign sequence --------------------------------------------------------------------------------------


def _signed(signs: Sequence[int]) -> tuple[list[float], list[float]]:
    """Two pass/fail arms whose paired differences are ``signs``: +1 the second arm alone passed, -1 the first."""
    return [1.0 if s < 0 else 0.0 for s in signs], [1.0 if s > 0 else 0.0 for s in signs]


def _separations(toward: float, most: int) -> tuple[list[int], list[int]]:
    """For each count of signs from 2 to ``most``, how many of the sequences separate at α upward and downward under the
    bet staked ``toward`` the upward side."""
    up, down = [], []
    for j in range(2, most + 1):
        tally: Counter[bool] = Counter()
        for signs in itertools.product((1, -1), repeat=j):
            p = bounded_separation_p(*_signed(signs), paired=True, value_range=RATE, mixture=True, toward=toward)
            if p is not None and p < SIGNIFICANCE_ALPHA:
                tally[sum(signs) > 0] += 1
        up.append(tally[True])
        down.append(tally[False])
    return up, down


#: Every sequence of 2 to 12 signs: how many separate upward and downward, staked nine tenths up and evenly.
_TILTED_UP = [0, 0, 0, 0, 1, 2, 4, 8, 22, 44, 87]
_TILTED_DOWN = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 5]
_EVEN = [0, 0, 0, 0, 0, 1, 2, 4, 8, 23, 46]


def test_every_sign_sequence_separates_as_counted_staked_toward_the_asked_side_and_evenly() -> None:
    """The bet over every sequence of up to twelve signs, enumerated: staked nine tenths up, all six up first separates
    at six signs, and a move against the question needs nine; staked evenly the two sides are mirror images and need
    seven. A dropped or flipped tilt, other weights, another cap or another α each move these counts."""
    assert _separations(DECLARED_DIRECTION_SHARE, 12) == (_TILTED_UP, _TILTED_DOWN)
    assert _separations(0.5, 12) == (_EVEN, _EVEN)


def test_a_tie_moves_no_bet() -> None:
    """A tied pair is a value at the null's own mean, so the bet's reading of signs is its reading of the cases."""
    signs = [1, 1, -1, 1, 1, 1, -1, 1, 1, 1]
    a, b = _signed(signs)
    tied_a, tied_b = [*a[:3], 1.0, 0.0, *a[3:], 0.0], [*b[:3], 1.0, 0.0, *b[3:], 0.0]
    for toward in (0.5, DECLARED_DIRECTION_SHARE):
        p = bounded_separation_p(a, b, paired=True, value_range=RATE, mixture=True, toward=toward)
        tied = bounded_separation_p(tied_a, tied_b, paired=True, value_range=RATE, mixture=True, toward=toward)
        assert p == pytest.approx(tied, rel=1e-12)


def test_a_single_read_of_twelve_pass_fail_cases_under_the_null_separates_at_most_alpha_exactly() -> None:
    """Under the pass/fail null (each case passing at a share drawn from 0.1 to 0.5 on both arms), a case is a sign with
    chance q = E[2s(1 - s)] = 59/150 and a tie otherwise, each sign even; so the chance the primary's bet separates at
    twelve cases is the binomial mixture of the enumerated counts, exactly: 0.4974%, at most α (the product's decision
    also needs its interval to exclude zero, so its rate is at most this; the certificate's count is 0.475%)."""
    q = 0.6 - 2 * (0.5**3 - 0.1**3) / (3 * 0.4)
    assert q == pytest.approx(59 / 150)
    rate = sum(
        math.comb(12, j) * q**j * (1 - q) ** (12 - j) * (up + down) / 2**j
        for j, up, down in zip(range(2, 13), _TILTED_UP, _TILTED_DOWN, strict=True)
    )
    assert rate == pytest.approx(0.0049741416, rel=1e-8)
    assert rate <= SIGNIFICANCE_ALPHA


def test_each_end_of_the_interval_is_a_one_sided_bound_on_every_sign_sequence() -> None:
    """The primary's interval (the even hedge's confidence sequence at 90%, read on the capital now) over every sequence
    of up to eight signs, a true difference of zero: its lower end passes zero only on all signs up, from six signs on,
    and its upper end only on all down; each side's chance at eight signs, 1/256, is at most α, the bound each end
    states. An interval read at 1 − α, or narrowed, passes zero on more of them."""
    lows, highs = [], []
    for j in range(2, 9):
        low = high = 0
        for signs in itertools.product((1, -1), repeat=j):
            interval = bounded_difference_interval(
                *_signed(signs),
                paired=True,
                value_range=RATE,
                confidence=1 - 2 * SIGNIFICANCE_ALPHA,
                mixture=True,
                current=True,
            )
            assert interval is not None
            low += interval[0] > 0
            high += interval[1] < 0
        lows.append(low)
        highs.append(high)
    assert lows == highs == [0, 0, 0, 0, 1, 1, 1]
    assert 1 / 2**8 <= SIGNIFICANCE_ALPHA


# --- pinned outputs --------------------------------------------------------------------------------------------------

_THIRD = 1 / 3

#: Fixed samples (control, contrast): three trials a case over twelve cases; twenty pass/fail cases mostly up; sixteen
#: mostly down.
_SAMPLES: dict[str, tuple[list[float], list[float]]] = {
    "k3-12": (
        [0, _THIRD, 2 * _THIRD, _THIRD, 0, _THIRD, 2 * _THIRD, _THIRD, _THIRD, 0, 2 * _THIRD, _THIRD],
        [_THIRD, 2 * _THIRD, 2 * _THIRD, 1, _THIRD, _THIRD, 1, 2 * _THIRD, 0, _THIRD, 1, 2 * _THIRD],
    ),
    "k1-20-up": (
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    ),
    "k1-16-down": (
        [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
    ),
}

#: Each sample's p staked nine tenths up, evenly and nine tenths down; its e-values up and down; and its intervals at
#: 90% and 95% on the most the capital reached; and at 90% on the capital now, the primary's.
_GOLDEN: dict[
    str,
    tuple[
        tuple[float, float, float], tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ],
] = {
    "k3-12": (
        (0.19644264061949343, 0.3452627954157681, 1.0),
        (5.639094474959404, 0.1535945333989539),
        (-0.12548302178038284, 0.48687962407711893),
        (-0.18503202556166798, 0.5223110526567325),
        (-0.12548302178038284, 0.48687962407711893),
    ),
    "k1-20-up": (
        (0.015048295765425847, 0.027079284587393396, 0.1350532415903887),
        (73.83373527002061, 0.02345964756920471),
        (0.08922183061949873, 0.7064218416344374),
        (0.04037408474832738, 0.7283258262090384),
        (0.0847714288160204, 0.7064218416344374),
    ),
    "k1-16-down": (
        (0.23903985871514394, 0.048080926623245666, 0.026728581923085762),
        (0.029686412941786006, 41.56684968694822),
        (-0.7469679876812734, -0.0625234038598137),
        (-0.772888931169291, -0.003200092032784596),
        (-0.7667127677268581, -0.05869566732144449),
    ),
}


@pytest.mark.parametrize("sample", sorted(_SAMPLES))
def test_the_bet_reads_fixed_samples_as_pinned(sample: str) -> None:
    """Each fixed sample's p (staked toward the asked side, evenly, the other way), its e-values and its intervals at 90%
    and 95%, to the digits: any change to the bet's stakes, weights, hedge, cap or interval moves them."""
    a, b = (list(map(float, side)) for side in _SAMPLES[sample])
    ps, evidence, ninety, ninety_five, now = _GOLDEN[sample]
    for toward, pinned in zip((DECLARED_DIRECTION_SHARE, 0.5, 1 - DECLARED_DIRECTION_SHARE), ps, strict=True):
        read = bounded_separation_p(a, b, paired=True, value_range=RATE, mixture=True, toward=toward)
        assert read == pytest.approx(pinned, rel=1e-9), toward
    assert bet_evidence(a, b, value_range=RATE) == pytest.approx(evidence, rel=1e-9)
    for confidence, pinned_interval in ((0.90, ninety), (0.95, ninety_five)):
        interval = bounded_difference_interval(a, b, paired=True, value_range=RATE, confidence=confidence, mixture=True)
        assert interval == pytest.approx(pinned_interval, rel=1e-9, abs=1e-12), confidence
    current = bounded_difference_interval(
        a, b, paired=True, value_range=RATE, confidence=0.90, mixture=True, current=True
    )
    assert current == pytest.approx(now, rel=1e-9, abs=1e-12)


def _passed(case: Mapping[str, Any], answer: Any) -> bool:
    return bool(answer)


def _scripted(name: str, answers: Sequence[bool]) -> Callable[[Mapping[str, Any]], Any]:
    def run(case: Mapping[str, Any]) -> bool:
        return answers[int(case["n"])]

    run.__name__ = name
    return run


@pytest.mark.parametrize(("better", "pinned"), [(True, 0.015048295765425847), (False, 0.1350532415903887)])
async def test_the_product_s_primary_is_read_staked_toward_the_asked_side(better: bool, pinned: float) -> None:
    """Through ``Eval.run``, eight cases where v2 alone passes every one, or fails every one v1 passes: the primary's
    p is the bet staked nine tenths toward the side the question asks about (v2 better), so the move it asks about
    reads 0.0150 and the move against it 0.1351, where the even hedge would read 0.0271 either way. A tilt dropped or
    flipped where the family reads its primary moves both."""
    cases = [{"id": f"c{n:03d}", "n": n} for n in range(8)]
    arms = {"v1": _scripted("v1", [not better] * 8), "v2": _scripted("v2", [better] * 8)}
    result = await Eval(cases, arms, [_passed], k=1, name="tilt", question=QUESTION).run(store=":memory:")
    (family,) = result.report.bundle.multiple_comparisons.families
    (primary,) = [comparison for comparison in family.comparisons if comparison.name == family.primary]
    assert primary.p_adjusted == pytest.approx(pinned, rel=1e-9)
    assert primary.verdict == ("improved" if better else "not_separated")


# --- small seeded runs, restated -------------------------------------------------------------------------------------


def test_the_first_shard_of_each_small_n_claim_counts_as_the_product_did() -> None:
    """The first shard of each small-n claim, restated, at the counts the product gave it through ``Eval.run``: single
    reads at twelve cases (200 a null) separate once, never and never; continued to five reads (150 a null), (separated,
    evidence for v2 ever strong, against it) as below; at a true difference of 0.3, none of 200 intervals misses it on
    either side; and of 200 newcomers asking a gate, and 200 pointing a late question at the leader, 6 and 7 separate
    continued, and 4 and 2 read once at sixteen cases.
    The single reads, the continued reads and the coverage are the counts the product gave through ``Eval.run``."""
    once = [
        sum(small_once(null, f"once-{null}-0-{i}")[0].decided for i in range(200))
        for null in ("binary", "onramp", "skewed")
    ]
    assert once == [1, 0, 0]
    for null, pinned in (("binary", (4, 6, 5)), ("onramp", (4, 5, 4)), ("skewed", (5, 3, 4))):
        came = [came_to(small_continued(null, f"continued-{null}-0-{i}")) for i in range(150)]  # type: ignore[arg-type]
        counted = (
            sum(c.separated for c in came),
            sum(c.strong_better for c in came),
            sum(c.strong_worse for c in came),
        )
        assert counted == pinned, null
    sides = [missed_side(small_covers(f"cover-0-{i}")) for i in range(200)]
    assert (sides.count("low"), sides.count("high")) == (0, 0)
    asked = [
        sum(small_continued("binary", f"{asks}-{i}", asks)[-1].decided for i in range(200)) for asks in ("gate", "late")
    ]
    assert asked == [6, 7]
    once = [
        sum(small_continued("binary", f"{asks}-once-{i}", asks, start=16, reads=1)[-1].decided for i in range(200))
        for asks in ("gate", "late")
    ]
    assert once == [4, 2]


def test_the_first_shards_of_the_no_question_claims_count_as_the_product_did() -> None:
    """Continued with no question, the first two shards of fifty separate once and twice, as through ``Eval.run``;
    re-run whole, the first separates four times; and the drained watcher's horizon, 300 paths of 120 pass/fail pairs
    and 100 of 420 pairs of three trials, crosses 9 and 1 times."""
    continued = [sum(rereadable(f"continued-{s}-{i}", continued=True)[-1].decided for i in range(50)) for s in (0, 1)]
    assert continued == [1, 2]
    assert sum(rereadable(f"rerun-0-{i}", continued=False)[-1].decided for i in range(50)) == 4
    assert sum(horizon_crossed(f"horizon-k1-{i}", 1, 120) for i in range(300)) == 9
    assert sum(horizon_crossed(f"horizon-k3-{i}", 3, 420) for i in range(100)) == 1


def test_the_first_shards_of_the_planned_onramp_claim_count_as_restated() -> None:
    """The on-ramp's held-out plans (read at twenty-four apps, then continued once with eight more), under its null: the
    first four shards of 150 separate 0, 0, 2 and 0 times, and every newcomer not decided at twenty-four reads again at
    thirty-two. Counted by this restatement on these seeds, which ``restated-as-product.small-planned`` holds to
    ``Eval.run``."""
    planned = {"start": PLANNED_START, "reads": PLANNED_READS, "add": PLANNED_ADD}
    counts = []
    for shard in range(4):
        reads = [small_continued("onramp", f"planned-onramp-{shard}-{i}", **planned) for i in range(150)]
        assert all(len(each) == 2 and each[1].cases == 32 for each in reads if not each[0].decided), shard
        counts.append(sum(came_to(each).separated for each in reads))
    assert counts == [0, 0, 2, 0]
