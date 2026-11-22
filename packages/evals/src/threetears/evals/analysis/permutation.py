"""A stratified conditional permutation test, drawn as counts and read sequentially.

Under no difference between groups, each stratum's marked members (a case's lost cells) fall among its members as a
draw without replacement, every arrangement equally likely given the stratum's margins: how many members each group
has there, and how many are marked. :func:`stratified_hypergeometric_totals` draws that law directly, as each group's
count of marked members summed over the strata, never by shuffling members. It reads strata by their margins: strata
that share them are drawn together, as counts over the arrangements they can take where they are many, or one by
one from a table of those arrangements; a stratum with too many arrangements to tabulate is drawn group by group
from the hypergeometric law. Every count is an integer. Each law is computed to double precision.

:func:`sequential_permutation_p` reads a Monte Carlo p off such draws and stops as soon as the draws settle the
decision. Once enough draws are at least as extreme that ``(1 + at least) / (1 + draws)`` can no longer fall under
the level, the decision is the full draws' decision, so it stops there: this costs nothing. At 99, 199, 399 and 799
draws (:data:`_EARLY_LOOKS`) it also stops and rejects when so few draws were at least as extreme that a true p at the
level would give so few at most 1% of the time. That rejects where the full draws would not with a chance of at most
0.0003 at 999 draws and a level of 5%, whatever the data. The bound holds because a valid test's true p is never more
likely than uniform to sit low; ``test_permutation.py`` computes it from the binomial law.
"""

from __future__ import annotations

import functools
import math
import random
from bisect import bisect
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from typing import Final

#: The draws after which a sequential read may stop and reject: each one under the read's draws is a look.
_EARLY_LOOKS: Final = (99, 199, 399, 799)

#: How unlikely the count at a look must be, were the true p exactly the level, for the read to stop and reject.
_EARLY_REJECT_CHANCE: Final = 0.01

#: The most arrangements of one stratum's marked members drawn from a table; past this, group by group.
_TABLE_LIMIT: Final = 20_000

#: Arrangements tabulated per stratum that shares the margins: a table costs once, and saves on every draw.
_TABLE_PER_STRATUM: Final = 50

#: A binomial draw costs about as much as six table draws: margins shared by more strata than six times their
#: arrangements are drawn as counts.
_COUNTS_PER_ARRANGEMENT: Final = 6

#: How far into a hypergeometric law's tails its table runs: past a term this far under the mode's, the rest is
#: under double precision.
_TAIL: Final = 2.0**-60


def _binomial_cdf(trials: int, at_most: int, chance: float) -> float:
    return math.fsum(
        math.comb(trials, k) * chance**k * (1.0 - chance) ** (trials - k) for k in range(min(at_most, trials) + 1)
    )


@functools.cache
def _rule(draws: int, level: float) -> tuple[int, dict[int, int]]:
    """The count at least as extreme that settles a read of ``draws`` as no rejection, and the most at each early look
    under ``draws`` that stops it and rejects."""
    enough = next((g for g in range(draws + 1) if (1 + g) / (1 + draws) >= level), draws + 1)
    stops: dict[int, int] = {}
    for look in _EARLY_LOOKS:
        if look >= draws:
            break
        most = -1
        while _binomial_cdf(look, most + 1, level) <= _EARLY_REJECT_CHANCE and (2 + most) / (1 + look) < level:
            most += 1
        if most >= 0:
            stops[look] = most
    return enough, stops


def sequential_permutation_p(at_least_as_extreme: Callable[[], bool], *, draws: int, level: float) -> float:
    """The Monte Carlo p of a permutation test, read sequentially: ``(1 + at least) / (1 + draws)`` over the draws
    read, where ``at_least_as_extreme`` makes one draw and says whether its statistic is at least the observed one
    (a tie counts as at least as extreme).

    It stops early in two ways. Once the count reaches the one whose p is at least ``level``, no later draw can bring
    it under, so it returns that p, the least the full draws could read: the decision is theirs. At each of 99, 199,
    399 and 799 draws under ``draws``, a count a true p of ``level`` would give at most 1% of the time stops it with the
    p read so far, under ``level``. A p under ``level`` therefore rejects as the full draws would, but for a chance of
    at most 0.0003 at 999 draws and a level of 5%.

    :param at_least_as_extreme: Makes one draw; True when its statistic is at least the observed one.
    :param draws: The most draws read.
    :param level: The level the p is read against.
    :return: The p, under ``level`` exactly where the read rejects.
    :raises ValueError: For no draws, or a level outside (0, 1].
    """
    if draws < 1 or not 0.0 < level <= 1.0:
        raise ValueError(f"a sequential p needs at least one draw and a level in (0, 1]; got {draws} and {level}")
    enough, stops = _rule(draws, level)
    at_least = 0
    for done in range(1, draws + 1):
        at_least += at_least_as_extreme()
        if at_least >= enough:
            return (1 + at_least) / (1 + draws)
        most = stops.get(done)
        if most is not None and at_least <= most:
            return (1 + at_least) / (1 + done)
    return (1 + at_least) / (1 + draws)


def _arrangements(sizes: tuple[int, ...], marked: int, limit: int) -> list[tuple[tuple[int, ...], int]] | None:
    """Each way a stratum of ``sizes`` can hold ``marked`` marked members by group, with how many arrangements of its
    members give it (``∏ comb(size, held)``); None when there are more than ``limit``."""
    found: list[tuple[tuple[int, ...], int]] = []
    last = len(sizes) - 1

    def walk(group: int, left: int, held: tuple[int, ...], weight: int) -> bool:
        if group == last:
            if left <= sizes[group]:
                found.append(((*held, left), weight * math.comb(sizes[group], left)))
            return len(found) <= limit
        return all(
            walk(group + 1, left - x, (*held, x), weight * math.comb(sizes[group], x))
            for x in range(min(sizes[group], left) + 1)
        )

    return found if walk(0, marked, (), 1) else None


def _hypergeometric_table(total: int, size: int, marked: int) -> tuple[int, list[float]]:
    """The law of how many of ``marked`` members drawn from ``total`` without replacement fall in a group of ``size``:
    its least value and its cumulative chances from there, built outward from the mode by the terms' ratios and cut
    where a term falls under :data:`_TAIL` of the mode's."""
    low, high = max(0, marked - (total - size)), min(size, marked)
    mode = min(high, max(low, (size + 1) * (marked + 1) // (total + 2)))
    rest = total - size - marked
    terms = {mode: 1.0}
    term, x = 1.0, mode
    while x < high and term > _TAIL:
        term *= (size - x) * (marked - x) / ((x + 1) * (rest + x + 1))
        x += 1
        terms[x] = term
    term, x = 1.0, mode
    while x > low and term > _TAIL:
        term *= x * (rest + x) / ((size - x + 1) * (marked - x + 1))
        x -= 1
        terms[x] = term
    first, whole = min(terms), math.fsum(terms.values())
    running, cumulative = 0.0, []
    for value in range(first, max(terms) + 1):
        running += terms[value]
        cumulative.append(running / whole)
    cumulative[-1] = 1.0
    return first, cumulative


def stratified_hypergeometric_totals(
    strata: Sequence[tuple[Sequence[int], int]], rng: random.Random
) -> Iterator[tuple[int, ...]]:
    """Draws without end of each group's count of marked members, summed over ``strata``, under no difference between
    groups: each draw places every stratum's marked members among its members uniformly at random without
    replacement, independently across strata.

    :param strata: Each stratum's members by group (the same groups, in the same order, in every stratum) and how many
        of its members are marked.
    :param rng: The draws' source, so a seeded caller reads one stream.
    :return: An endless iterator of totals, one per group.
    :raises ValueError: For strata over different groups, or a stratum marking fewer than none or more than it has.
    """
    groups = len(strata[0][0]) if strata else 0
    for sizes, marked in strata:
        if len(sizes) != groups or min(sizes) < 0 or not 0 <= marked <= sum(sizes):
            raise ValueError(f"a stratum of {tuple(sizes)} cannot mark {marked} over {groups} groups")
    return _totals(Counter((tuple(sizes), marked) for sizes, marked in strata), groups, rng)


def _totals(kinds: Counter[tuple[tuple[int, ...], int]], groups: int, rng: random.Random) -> Iterator[tuple[int, ...]]:
    # Every group's total but the last is one digit of an integer in base ``base``, above any total, so a draw sums
    # integers and the last group's total is what the others leave of every marked member.
    base = sum(sum(sizes) * strata for (sizes, _), strata in kinds.items()) + 1
    units = [base**group for group in range(groups - 1)] + [0]
    marked_in_all = sum(marked * strata for (_, marked), strata in kinds.items())
    fixed = 0
    counts: list[tuple[int, list[tuple[int, float]]]] = []
    tables: list[tuple[int, list[int], list[float]]] = []
    chains: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for (sizes, marked), strata in sorted(kinds.items()):
        present = [(size, unit) for size, unit in zip(sizes, units, strict=True) if size]
        total = sum(sizes)
        if not marked:
            continue
        if marked == total:
            fixed += strata * sum(unit * size for size, unit in present)
            continue
        if len(present) == 1:
            fixed += strata * marked * present[0][1]
            continue
        found = _arrangements(sizes, marked, min(_TABLE_LIMIT, max(64, _TABLE_PER_STRATUM * strata)))
        if found is None:
            chains.append((strata, total, marked, present))
            continue
        whole = math.comb(total, marked)
        codes = [sum(unit * held for unit, held in zip(units, way, strict=True)) for way, _ in found]
        if _COUNTS_PER_ARRANGEMENT * len(found) < strata:
            steps, left = [], whole
            for index in sorted(range(len(found)), key=lambda each: -found[each][1]):
                weight = found[index][1]
                steps.append((codes[index], weight / left if weight < left else 1.0))
                left -= weight
            counts.append((strata, steps))
        else:
            running, cumulative = 0, []
            for _, weight in found:
                running += weight
                cumulative.append(running / whole)
            tables.append((strata, codes, cumulative))
    laws: dict[tuple[int, int, int], tuple[int, list[float]]] = {}
    uniform, binomial, choices = rng.random, rng.binomialvariate, rng.choices
    while True:
        code = fixed
        for strata, steps in counts:
            left = strata
            for each, chance in steps:
                if chance >= 1.0:
                    code += left * each
                    break
                drawn = binomial(left, chance)
                code += drawn * each
                left -= drawn
                if not left:
                    break
        for strata, codes, cumulative in tables:
            if strata == 1:
                code += codes[bisect(cumulative, uniform())]
            else:
                code += sum(choices(codes, cum_weights=cumulative, k=strata))
        for strata, total, marked, present in chains:
            last = len(present) - 1
            for _ in range(strata):
                left_total, left_marked = total, marked
                for group, (size, unit) in enumerate(present):
                    if group == last or not left_marked:
                        code += left_marked * unit
                        break
                    if left_marked == left_total:
                        code += sum(each * held_unit for each, held_unit in present[group:])
                        break
                    key = (left_total, size, left_marked)
                    law = laws.get(key)
                    if law is None:
                        law = laws[key] = _hypergeometric_table(*key)
                    held = law[0] + bisect(law[1], uniform())
                    code += held * unit
                    left_total -= size
                    left_marked -= held
        totals = []
        for _ in range(groups - 1):
            code, digit = divmod(code, base)
            totals.append(digit)
        yield (*totals, marked_in_all - sum(totals))


__all__ = [
    "sequential_permutation_p",
    "stratified_hypergeometric_totals",
]
