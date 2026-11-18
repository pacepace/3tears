"""Newcomers under the null on the quick path, each simulated two ways over the same seeded answers: through the product
(``Eval.run`` into a store, its reading as the print states it) and restated over the bet alone (the package's own
statistics functions, :func:`~threetears.evals.analysis.stats.bounded_separation_p`,
:func:`~threetears.evals.analysis.stats.bet_evidence` and
:func:`~threetears.evals.analysis.stats.bounded_difference_interval`, over the committed order
:func:`~threetears.evals.analysis.bundle.trials.committed_order`, with no ``Eval.run``).

The restatement is what the statistics certificate counts its newcomers on (``stats_certificate.py``): it is fast enough
to read tens of thousands of them. It is held to the product read for read (:func:`same_reads`): the certification runs
a few hundred newcomers through both and every reading's p and decision must agree, so a count on the restatement is the
product's count.

Each newcomer is one of:

- **small n** (a question asked, so its first grader is the primary, decided on the bet staked
  :data:`~threetears.evals.analysis.stats.DECLARED_DIRECTION_SHARE` toward the better side): read once at
  :data:`ONCE_AT` cases; continued from eight cases up to five reads (:func:`small_continued`); or read once at sixteen
  cases at a true difference of :data:`EFFECT`, where its interval should cover it (:func:`small_covers`). Its null is
  one of :data:`Null`;
- **no question** (the even hedge): two arms answering right half the time, from 20 cases up to five runs, each run
  continuing the one comparison or re-running every case in a store of its own (:func:`rereadable`);
- **watched** (product only): a newcomer who stops each run partway, so which cells a stop finds in flight depends on
  time (:func:`watched`); under one interrupt the stopped cells drain, and every reading is the bet over the longest
  complete prefix of the committed order, which :func:`watched` checks reading by reading.
"""

from __future__ import annotations

import asyncio
import math
import random
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from threetears.evals import Eval, Result
from threetears.evals.analysis import ComparisonFamily, assemble_context_bundle
from threetears.evals.analysis.bundle.trials import committed_order
from threetears.evals.analysis.campaigns import get_campaign, list_campaigns
from threetears.evals.analysis.stats import (
    DECLARED_DIRECTION_SHARE,
    EVIDENCE_STEPS,
    SIGNIFICANCE_ALPHA,
    bet_decided_at,
    bet_evidence,
    bounded_difference_interval,
    bounded_separation_p,
    holm_adjust,
    interval_permits_separation,
    paired_equivalence,
    separation_test,
)
from threetears.evals.kernel.result_condition import runs_again
from threetears.evals.quick import as_host
from threetears.evals.quick.one_call import CALLABLE_HOST_ID
from threetears.evals.run import list_results
from threetears.evals.schema import EvalResult
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.storage import InMemoryDocumentStore

from packages.evals.tests.off_loop import off_loop

#: The pass rate's range, which the per-case shares and their differences are read on.
RATE = (0.0, 1.0)

#: The question every small-n comparison asks, so its first grader is the primary.
QUESTION = "Does v2 pass more often than v1?"

#: Which null a small-n newcomer runs under: pass/fail once a case at a share drawn from 0.1 to 0.5 on both arms; the
#: on-ramp's shape (apps as cases, three trials, an arm-by-app spread); or a skewed null whose mean difference is zero
#: but whose differences are not symmetric.
Null = Literal["binary", "onramp", "skewed"]

#: The cases a small-n single read reads.
ONCE_AT = 12

#: The cases a coverage newcomer reads, and the true difference on the pass rate it reads them at.
COVER_AT, EFFECT = 16, 0.3

#: The most reads a continued newcomer takes.
READS = 5

#: The on-ramp's held-out plans: a comparison read at twenty-four apps and, not separated, continued once with eight more
#: (``packages/evals/onramp/plans/held-out-2.toml``): its start, its one add, and its reads.
PLANNED_START, PLANNED_ADD, PLANNED_READS = 24, 8, 2


@dataclass(frozen=True)
class Read:
    """One reading of a newcomer's comparison.

    :param cases: The cases launched when it was read.
    :param p: The p the decision reads: the primary's, staked toward the asked side, with a question asked; the even
        hedge's without. ``None`` where the product reads none.
    :param decided: Whether it decided either way.
    :param better: The evidence the bet holds now that the contrast is better (small n only).
    :param worse: The evidence it holds that it is worse (small n only).
    :param interval: The primary's interval, where one was read (the restatement reads it only where a coverage needs
        it).
    :param decided_at: The cases at which the primary's bet first reached its level, where it decided (small n only).
    """

    cases: int
    p: float | None
    decided: bool
    better: float | None = None
    worse: float | None = None
    interval: tuple[float, float] | None = None
    decided_at: int | None = None


def _close(read: float | None, restated: float | None) -> bool:
    """Whether two readings' figures agree, to float rounding (the product sums each case's repeats in exact
    decimals)."""
    if read is None or restated is None:
        return read is restated
    return math.isclose(read, restated, rel_tol=1e-9, abs_tol=1e-12)


def same_reads(product: Sequence[Read], restated: Sequence[Read]) -> str | None:
    """Where a newcomer's product reads and its restated reads part, as a sentence; ``None`` where they agree read for
    read: as many reads, each at the same cases, its p and its evidence to float rounding, the same decision, and the
    same interval wherever the restatement read one."""
    if len(product) != len(restated):
        return f"{len(product)} product reads against {len(restated)} restated: {product} / {restated}"
    for place, (one, two) in enumerate(zip(product, restated, strict=True)):
        if (
            one.cases != two.cases
            or one.decided != two.decided
            or not _close(one.p, two.p)
            or not _close(one.better, two.better)
            or not _close(one.worse, two.worse)
            or one.decided_at != two.decided_at
        ):
            return f"read {place + 1}: product {one} / restated {two}"
        if two.interval is not None and (
            one.interval is None or not all(_close(x, y) for x, y in zip(one.interval, two.interval, strict=True))
        ):
            return f"read {place + 1}: product interval {one.interval} / restated {two.interval}"
    return None


def case_set_seed(cases: Sequence[Mapping[str, Any]]) -> str:
    """The id a case list is addressed by, which seeds its committed order: the callable host's id and the case list's
    digest, as the quick path addresses a case set with no labels, judge or seeds. The certification's equivalence check
    holds it to the product's own ``Result.template_id``."""
    return f"{CALLABLE_HOST_ID}-{canonical_digest([dict(case) for case in cases])[:16]}"


def _ordered(seed: str, launches: Sequence[range]) -> list[int]:
    """The committed order of a comparison whose launches added ``launches``' cases, as case indexes."""
    return [
        int(case.rsplit("-", 1)[1])
        for case in committed_order(seed, [[f"{seed}-{index}" for index in added] for added in launches])
    ]


def _primary_permits(interval: tuple[float, float] | None) -> bool:
    """Whether a separation may stand beside the primary's interval: one was read and it excludes zero."""
    return interval is not None and interval_permits_separation(interval)


# --- small n ---------------------------------------------------------------------------------------------------------


def small_share(null: Null, tag: str, index: int) -> tuple[float, float] | tuple[int, int]:
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


def repeats(null: Null) -> int:
    """The trials a case runs under ``null``."""
    return 1 if null == "binary" else 3


def _passed(case: Mapping[str, Any], answer: Any) -> bool:
    return bool(answer)


def small_cases(n: int) -> list[dict[str, Any]]:
    """A small-n newcomer's first ``n`` cases."""
    return [{"id": f"c{index:03d}", "n": index} for index in range(n)]


#: Held while a plain runner takes its call's number.
_COUNTING = threading.Lock()


def _drawn(null: Null, tag: str, arm: int, calls: Counter[tuple[int, int]]) -> Callable[[Mapping[str, Any]], Any]:
    """Arm ``arm``'s runner: each call a fresh seeded draw at its share of the case, or its count's next call."""

    def run(case: Mapping[str, Any]) -> bool:
        index = int(case["n"])
        # A plain runner may be called from several threads at once, one per repeat in flight: each call takes its own
        # number under the lock, so no two repeats of a case read the same call.
        with _COUNTING:
            calls[(index, arm)] += 1
            call = calls[(index, arm)]
        held = small_share(null, tag, index)[arm]
        if isinstance(held, int):
            return call <= held
        return random.Random(f"{tag}-{arm}-{index}-{call}").random() < held

    run.__name__ = f"v{arm + 1}"
    return run


def _small_values(null: Null, tag: str, index: int) -> tuple[float, float]:
    """Case ``index``'s value on each arm, as :func:`_drawn` answers it: the share of its trials passed."""
    k = repeats(null)
    held = small_share(null, tag, index)
    values = []
    for arm in (0, 1):
        share = held[arm]
        if isinstance(share, int):
            values.append(min(k, share) / k)
        else:
            values.append(
                sum(random.Random(f"{tag}-{arm}-{index}-{call}").random() < share for call in range(1, k + 1)) / k
            )
    return values[0], values[1]


def _primary(result: Result) -> Any:
    (family,) = result.report.bundle.multiple_comparisons.families
    return next(comparison for comparison in family.comparisons if comparison.name == family.primary)


def _decided(families: Sequence[ComparisonFamily]) -> bool:
    """Whether any comparison of a reading is decided either way."""
    return any(
        comparison.verdict in ("improved", "regressed") for family in families for comparison in family.comparisons
    )


def _small_read(result: Result, cases: int) -> Read:
    primary = _primary(result)
    return Read(
        cases,
        primary.p_adjusted,
        _decided(result.report.bundle.multiple_comparisons.families),
        primary.evidence_better,
        primary.evidence_worse,
        primary.interval,
        primary.decided_at_cases,
    )


#: The share of a question's bet staked on the contrast rising: toward the better side (a question), toward the worse
#: side (a gate, a question with a margin), or evenly (a question set after the results).
Lean = Literal["better", "worse", "none"]

_TOWARD: dict[Lean, float] = {
    "better": DECLARED_DIRECTION_SHARE,
    "worse": 1.0 - DECLARED_DIRECTION_SHARE,
    "none": 0.5,
}


def _small_restated(
    a: Sequence[float],
    b: Sequence[float],
    cases: int,
    *,
    interval: bool = False,
    lean: Lean = "better",
    margin: float | None = None,
) -> Read:
    """The primary's reading of ``a`` against ``b`` in their committed order: the bet staked as ``lean`` says, its p
    corrected by Holm with the equivalence test against ``margin`` where one is declared (one arm, so at most one
    hypothesis true), decided where the corrected p is below α, at the count where the bet first reached its level; the
    evidence each side holds now; and, where ``interval``, the even hedge's confidence sequence at 1 − 2α read on the
    capital now."""
    toward = _TOWARD[lean]
    p = bounded_separation_p(list(a), list(b), paired=True, value_range=RATE, mixture=True, toward=toward)
    p = 1.0 if p is None else p
    adjusted = p
    if margin is not None:
        _, equivalence = paired_equivalence(
            [y - x for x, y in zip(a, b, strict=True)], margin, value_range=RATE, mixture=True
        )
        if equivalence is not None:
            adjusted = holm_adjust([p, equivalence], max_true=1)[0]
    evidence = bet_evidence(list(a), list(b), value_range=RATE)
    better, worse = evidence if evidence is not None else (None, None)
    point = None
    if adjusted < SIGNIFICANCE_ALPHA:
        level = SIGNIFICANCE_ALPHA * p / adjusted if adjusted > 0 else SIGNIFICANCE_ALPHA
        point = bet_decided_at(list(a), list(b), value_range=RATE, toward=toward, alpha=min(1.0, level))
    bounds = None
    if interval:
        bounds = bounded_difference_interval(
            list(a),
            list(b),
            paired=True,
            value_range=RATE,
            confidence=1.0 - 2.0 * SIGNIFICANCE_ALPHA,
            mixture=True,
            current=True,
        )
    return Read(cases, adjusted, point is not None, better, worse, bounds, point[0] if point is not None else None)


async def small_once_product(null: Null, tag: str, n: int = ONCE_AT) -> list[Read]:
    """One newcomer's single read of ``n`` cases under ``null``, through ``Eval.run``."""
    calls: Counter[tuple[int, int]] = Counter()
    arms = {"v1": _drawn(null, tag, 0, calls), "v2": _drawn(null, tag, 1, calls)}
    result = await Eval(small_cases(n), arms, [_passed], k=repeats(null), name="null", question=QUESTION).run(
        store=InMemoryDocumentStore(), scope="s"
    )
    return [_small_read(result, n)]


def small_once(null: Null, tag: str, n: int = ONCE_AT) -> list[Read]:
    """:func:`small_once_product` restated over the bet alone."""
    seed = case_set_seed(small_cases(n))
    values = [_small_values(null, tag, index) for index in _ordered(seed, [range(n)])]
    return [_small_restated([a for a, _ in values], [b for _, b in values], n)]


#: What a continued small-n newcomer asks: whether v2 does better (``question``); whether it is no worse within five
#: points, a gate, whose bet leans toward worse (``gate``); or nothing up front, then at each read a question pointed at
#: whichever arm leads, set after the results, which leans no way (``late``).
Asks = Literal["question", "gate", "late"]

#: A gate's question and its margin, in the pass rate's units.
GATE_QUESTION, GATE_MARGIN = "Is v2 no worse than v1?", 0.05


async def small_continued_product(
    null: Null, tag: str, asks: Asks = "question", *, start: int = 8, reads: int = READS, add: int | None = None
) -> list[Read]:
    """One newcomer under ``null`` who reads at eight cases and, while nothing separates, appends cases and runs the
    comparison again into the same store, at most :data:`READS` reads: each add four cases plus up to sixteen more the
    nearer the primary's p came to zero. Every read is the product's continuation (only appended cases run); what each
    read asks is ``asks``. ``start``, ``reads`` and ``add`` (a fixed size for every add) read another plan: sixteen
    cases once is a single read, and the on-ramp's held-out plans read twenty-four, then eight more."""
    calls: Counter[tuple[int, int]] = Counter()
    arms = {"v1": _drawn(null, tag, 0, calls), "v2": _drawn(null, tag, 1, calls)}
    store = InMemoryDocumentStore()
    n = start
    taken: list[Read] = []
    for _ in range(reads):
        asked: dict[str, Any] = (
            {"question": QUESTION}
            if asks == "question"
            else {"question": GATE_QUESTION, "margin": GATE_MARGIN}
            if asks == "gate"
            else {}
        )
        result = await Eval(small_cases(n), arms, [_passed], k=repeats(null), name="null", **asked).run(
            store=store, scope="s"
        )
        if asks == "late":
            (row,) = result.contrasts("_passed")
            leader, other = ("v2", "v1") if (row.get("delta") or 0.0) >= 0 else ("v1", "v2")
            result = await result.against(other, question=f"Does {leader} pass more often than {other}?")
        taken.append(_small_read(result, n))
        if taken[-1].decided:
            break
        p = taken[-1].p if taken[-1].p is not None else 1.0
        n += add if add is not None else 4 + int(16 * (1.0 - min(1.0, p)))
    return taken


def small_continued(
    null: Null, tag: str, asks: Asks = "question", *, start: int = 8, reads: int = READS, add: int | None = None
) -> list[Read]:
    """:func:`small_continued_product` restated over the bet alone: the same draws, the committed order, the bet staked
    as the question leans (the asked side, toward worse for a gate with its equivalence test beside it under Holm,
    evenly and against the trailing arm for a question set after the results), the decision rule, the add sizes and
    the evidence."""
    seed = case_set_seed(small_cases(start))
    launches: list[range] = []
    values: dict[int, tuple[float, float]] = {}
    n, held = start, 0
    taken: list[Read] = []
    for _ in range(reads):
        launches.append(range(held, n))
        held = n
        order = _ordered(seed, launches)
        for index in order:
            if index not in values:
                values[index] = _small_values(null, tag, index)
        a, b = [values[i][0] for i in order], [values[i][1] for i in order]
        if asks == "late":
            if sum(b) / len(b) - sum(a) / len(a) < 0:
                a, b = b, a
            taken.append(_small_restated(a, b, n, lean="none"))
        elif asks == "gate":
            taken.append(_small_restated(a, b, n, lean="worse", margin=GATE_MARGIN))
        else:
            taken.append(_small_restated(a, b, n))
        if taken[-1].decided:
            break
        p = taken[-1].p if taken[-1].p is not None else 1.0
        n += add if add is not None else 4 + int(16 * (1.0 - min(1.0, p)))
    return taken


@dataclass(frozen=True)
class Continued:
    """What one continued small-n newcomer came to: whether a read separated, and whether the evidence on each side ever
    reached the strong step at a read."""

    separated: bool
    strong_better: bool
    strong_worse: bool


def came_to(reads: Sequence[Read]) -> Continued:
    """A continued small-n newcomer's reads, summed up."""
    strong = EVIDENCE_STEPS[1]
    return Continued(
        any(read.decided for read in reads),
        any((read.better or 0.0) >= strong for read in reads),
        any((read.worse or 0.0) >= strong for read in reads),
    )


def _cover_values(tag: str, index: int) -> tuple[float, float]:
    base = random.Random(f"{tag}-case-{index}").uniform(0.1, 0.5)
    return tuple(  # type: ignore[return-value]
        1.0 if random.Random(f"{tag}-{which}-{index}-1").random() < base + (EFFECT if which else 0.0) else 0.0
        for which in (0, 1)
    )


async def small_covers_product(tag: str, n: int = COVER_AT) -> list[Read]:
    """One newcomer's single read of ``n`` pass/fail cases, v1 passing at a share drawn from 0.1 to 0.5 and v2 at that
    plus :data:`EFFECT`, through ``Eval.run``; its read carries the primary's interval."""
    calls: Counter[tuple[int, int]] = Counter()

    def arm(which: int) -> Callable[[Mapping[str, Any]], Any]:
        def run(case: Mapping[str, Any]) -> bool:
            index = int(case["n"])
            calls[(index, which)] += 1
            base = random.Random(f"{tag}-case-{index}").uniform(0.1, 0.5)
            share = base + (EFFECT if which else 0.0)
            return random.Random(f"{tag}-{which}-{index}-{calls[(index, which)]}").random() < share

        run.__name__ = f"v{which + 1}"
        return run

    result = await Eval(
        small_cases(n), {"v1": arm(0), "v2": arm(1)}, [_passed], k=1, name="cover", question=QUESTION
    ).run(store=InMemoryDocumentStore(), scope="s")
    return [_small_read(result, n)]


def small_covers(tag: str, n: int = COVER_AT) -> list[Read]:
    """:func:`small_covers_product` restated over the bet alone, its interval read."""
    seed = case_set_seed(small_cases(n))
    values = [_cover_values(tag, index) for index in _ordered(seed, [range(n)])]
    return [_small_restated([a for a, _ in values], [b for _, b in values], n, interval=True)]


def missed_side(reads: Sequence[Read]) -> Literal["low", "high"] | None:
    """Which end of a coverage newcomer's interval missed the true difference: ``low`` where its lower end lies above
    it, ``high`` where its upper end lies below it, ``None`` where it holds it."""
    (read,) = reads
    assert read.interval is not None, "the primary reads an interval"
    if read.interval[0] > EFFECT:
        return "low"
    if read.interval[1] < EFFECT:
        return "high"
    return None


# --- no question: continued, or re-run whole -------------------------------------------------------------------------


def _correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is right."""
    return answer == "right"


def coin(tag: str, arm: str, rate: float, calls: Counter[tuple[int, str]]) -> Callable[[Mapping[str, Any]], Any]:
    """An arm answering ``right`` at ``rate``, each call a fresh seeded draw: a case run again draws again, so a product
    that re-ran a case already run would read a new value for it. ``calls`` counts each case's calls on the arm."""

    async def run(case: Mapping[str, Any]) -> str:
        calls[case["n"], arm] += 1
        draw = random.Random(f"{tag}-{case['n']}-{arm}-{calls[case['n'], arm]}").random()
        return "right" if draw < rate else "wrong"

    run.__name__ = arm
    return run


def coin_value(tag: str, n: int, arm: str, call: int, rate: float = 0.5) -> float:
    """The answer :func:`coin` gives case ``n`` on its ``call``-th call, as the value ``_correct`` reads."""
    return 1.0 if random.Random(f"{tag}-{n}-{arm}-{call}").random() < rate else 0.0


def _cases(n: int) -> list[dict[str, Any]]:
    return [{"n": index} for index in range(n)]


def _first_add(p: float) -> int:
    """How many cases a no-question newcomer adds after a reading whose smallest p was ``p``."""
    return 10 + int(90 * (1.0 - min(1.0, p)))


async def rereadable_product(tag: str, *, continued: bool) -> list[Read]:
    """One newcomer under the null, through ``Eval.run`` itself: two arms answering right at one rate (0.5, k = 1),
    each call a fresh seeded draw. Run 20 cases; while nothing separates, append cases and run again, at most
    :data:`READS` runs in all, each add ten cases plus up to ninety more the closer the reading's p came to α.
    ``continued`` runs every time into one store, so the product continues the comparison; otherwise each run is into a
    store of its own, which re-runs every case, as the quick path did before it continued a comparison, though read by
    the same bet. Separated: a verdict of improved or regressed on any comparison the print reads."""
    calls: Counter[tuple[int, str]] = Counter()
    arms = {"a": coin(tag, "a", 0.5, calls), "b": coin(tag, "b", 0.5, calls)}
    store = InMemoryDocumentStore()
    n = 20
    reads: list[Read] = []
    for _ in range(READS):
        if not continued:
            store = InMemoryDocumentStore()
        result = await Eval(_cases(n), arms, [_correct], k=1, name="null").run(store=store, scope="s")
        families = result.report.bundle.multiple_comparisons.families
        ps = [
            c.p_raw for family in families for c in family.comparisons if c.name == "_correct" and c.p_raw is not None
        ]
        reads.append(Read(n, min(ps, default=None), _decided(families)))
        if reads[-1].decided:
            break
        n += _first_add(min(ps, default=1.0))
    return reads


def _even_read(a: Sequence[float], b: Sequence[float], cases: int) -> Read:
    """The even hedge's reading of ``a`` against ``b`` in their committed order, with no question asked: decided where
    its p is below α, its delta moved and its interval at 1 − α excludes zero."""
    if len(a) < 2:
        return Read(cases, None, False)
    p = bounded_separation_p(list(a), list(b), paired=True, value_range=RATE, mixture=True)
    p = 1.0 if p is None else p
    delta = sum(b) / len(b) - sum(a) / len(a)
    decided = False
    if p < SIGNIFICANCE_ALPHA and delta:
        bounds = bounded_difference_interval(
            list(a), list(b), paired=True, value_range=RATE, confidence=1.0 - SIGNIFICANCE_ALPHA, mixture=True
        )
        decided = _primary_permits(bounds)
    return Read(cases, p, decided)


def rereadable(tag: str, *, continued: bool) -> list[Read]:
    """:func:`rereadable_product` restated over the bet alone: the same seeded answers (a case run again draws its next
    call), the committed order (one comparison's launches appended, or each re-run a case set of its own), the same p
    and the same add sizes."""
    calls: Counter[int] = Counter()
    first = case_set_seed(_cases(20))
    launches: list[range] = []
    n, held = 20, 0
    reads: list[Read] = []
    values: dict[int, tuple[float, float]] = {}
    for _ in range(READS):
        if continued:
            launches.append(range(held, n))
            held = n
            order = _ordered(first, launches)
        else:
            order = _ordered(case_set_seed(_cases(n)), [range(n)])
            values = {}
        for index in order:
            if index not in values:
                calls[index] += 1
                values[index] = (
                    coin_value(tag, index, "a", calls[index]),
                    coin_value(tag, index, "b", calls[index]),
                )
        reads.append(_even_read([values[i][0] for i in order], [values[i][1] for i in order], n))
        if reads[-1].decided:
            break
        n += _first_add(reads[-1].p if reads[-1].p is not None else 1.0)
    return reads


def horizon_crossed(tag: str, k: int, horizon: int) -> bool:
    """Whether the even hedge's capital ever reaches 1/α over ``horizon`` seeded pairs under the null, each case the
    mean of ``k`` fair coins on each arm (:func:`coin_value`): the most any drained watcher's readings can see
    (:func:`watched`), since each reads a prefix of one committed order whose values are each cell's first draw."""
    a = [sum(coin_value(tag, n, "a", call) for call in range(1, k + 1)) / k for n in range(horizon)]
    b = [sum(coin_value(tag, n, "b", call) for call in range(1, k + 1)) / k for n in range(horizon)]
    p = bounded_separation_p(a, b, paired=True, value_range=RATE, mixture=True)
    return p is not None and p < SIGNIFICANCE_ALPHA


# --- a comparison's record, read again -------------------------------------------------------------------------------


def stored(storage: Any, campaign_id: str, scope: str = "s") -> tuple[list[Any], list[EvalResult]]:
    """A filed comparison's runs, in the order they were launched, and every result they recorded. Reads the store."""
    (aspect,) = get_campaign(storage, campaign_id, scope).aspects
    runs = sorted(
        (run for run_id in aspect.run_ids if (run := storage.load_eval_run(run_id, scope)) is not None),
        key=lambda run: (run.created_at, run.id),
    )
    return runs, [result for run in runs for result in list_results(storage, run.id, scope)]


def cases_by_state(runs: Sequence[Any], results: Sequence[EvalResult]) -> tuple[set[str], set[str], set[str]]:
    """The cases the runs held, as recorded (a row that runs again recording nothing): complete (every arm, every
    repeat), reached on some arms or repeats only, and reached on none."""
    held = {case for run in runs for case in run.test_case_ids}
    every = set(range(1, runs[0].k_runs + 1))
    variants = {result.variant_key for result in results}
    repeats_of: dict[str, dict[str, set[int]]] = {}
    for result in results:
        if not runs_again(result):
            repeats_of.setdefault(result.test_case_id, {}).setdefault(result.variant_key, set()).add(result.k_iteration)
    complete = {
        case
        for case, on in repeats_of.items()
        if set(on) == variants and len(variants) > 1 and all(r == every for r in on.values())
    }
    return complete, set(repeats_of) - complete, held - set(repeats_of)


#: The rules a reading is recomputed by here: the product's (the longest complete prefix of the committed order), and
#: the two rules before it, kept as comparisons (each case at its last cell's time; each case in the cases file's order
#: within the launch that first recorded it).
Rule = Literal["prefix", "recorded", "filed"]


def reading_p(runs: Sequence[Any], results: Sequence[EvalResult], rule: Rule = "prefix") -> float | None:
    """The mixture bet's p on ``_correct`` over a comparison's recorded cells, each case's value its arm's mean over the
    repeats recorded (a row that runs again recording nothing), read by ``rule``:

    - ``prefix`` (the product): the cases of the committed order
      (:func:`~threetears.evals.analysis.bundle.trials.committed_order` over the runs' launches) up to the first case
      some arm has not recorded at every repeat;
    - ``recorded`` (a comparison rule, never the product's): every complete case, in the order of its last cell's
      ``scored_at``, launch by launch;
    - ``filed`` (a comparison rule, never the product's): every case both arms recorded any repeat of, in the launch
      that first recorded it, each launch's in the order of the cases file.

    ``None`` with fewer than two cases."""
    launch: dict[str, int] = {}
    cases_of: dict[str, list[str]] = {}
    for run in runs:
        launch.setdefault(run.launch_group_id or run.id, len(launch))
        cases_of.setdefault(run.launch_group_id or run.id, []).extend(run.test_case_ids)
    launch_of = {run.id: launch[run.launch_group_id or run.id] for run in runs}
    filed = {case: place for place, case in enumerate(dict.fromkeys(c for run in runs for c in run.test_case_ids))}
    complete, _, _ = cases_by_state(runs, results)
    values: dict[str, dict[str, list[float]]] = {}
    last: dict[str, tuple[int, str]] = {}
    first: dict[str, int] = {}
    for result in results:
        if runs_again(result):
            continue
        case = result.test_case_id
        when = (launch_of[result.eval_run_id], result.scored_at)
        last[case] = max(when, last.get(case, when))
        first[case] = min(when[0], first.get(case, when[0]))
        value = result.host_measures.get("_correct")
        if isinstance(value, float):
            values.setdefault(case, {}).setdefault(result.variant_key, []).append(value)
    variants = sorted({result.variant_key for result in results})
    paired = [case for case, on in values.items() if all(on.get(variant) for variant in variants)]
    if rule == "prefix":
        order = []
        for case in committed_order(runs[0].template_id, list(cases_of.values())):
            if case not in complete:
                break
            order.append(case)
        order = [case for case in order if case in paired]
    elif rule == "recorded":
        order = sorted((case for case in paired if case in complete), key=lambda case: (last[case], filed[case]))
    else:
        order = sorted(paired, key=lambda case: (first[case], filed[case]))
    if len(order) < 2 or len(variants) != 2:
        return None
    a, b = ([sum(values[case][variant]) / len(values[case][variant]) for case in order] for variant in variants)
    return separation_test(a, b, paired=True, value_range=RATE, mixture=True).p


def same_p(read: float | None, recomputed: float | None) -> bool:
    """Whether a reading's p is the one recomputed here, to float rounding (the product sums each case's repeats in
    exact decimals)."""
    return _close(read, recomputed)


# --- watched and stopped ---------------------------------------------------------------------------------------------

#: How long each arm's answer takes, as a law of the answer: both arms alike, latency apart from the answer (a coin of
#: its own, 1 or 6 ms); both arms alike, a wrong answer slower (1 ms right, 6 ms wrong); or arms of different laws, arm
#: ``a`` 0.4 ms whatever it answers and arm ``b`` five times slower when right (2 ms) and four or twenty times slower
#: again when wrong (8 or 40 ms).
Latency = Literal["alike-apart", "alike-slow-wrong", "b-slower-4x", "b-slower-20x"]

#: How a watcher stops a run: when the answers landed in it favour an arm by four (after ten on each arm), or at a point
#: drawn apart from any answer, in four runs of five.
Stop = Literal["lead", "blind"]


#: The most cases a watcher's comparison launches over ``launches`` runs: 20, and each add at most a hundred.
def watched_horizon(launches: int) -> int:
    """The most cases a watcher launches over ``launches`` runs: 20, then at most a hundred an add."""
    return 20 + (launches - 1) * _first_add(0.0)


def _seconds(latency: Latency, tag: str, arm: str, n: int, call: int, right: bool) -> float:
    """How long ``arm``'s answer to case ``n`` (its ``call``-th) takes under ``latency``."""
    if latency == "alike-apart":
        return 0.006 if random.Random(f"{tag}-latency-{n}-{arm}-{call}").random() < 0.5 else 0.001
    if latency == "alike-slow-wrong":
        return 0.001 if right else 0.006
    if arm == "a":
        return 0.0004
    return 0.002 if right else 0.002 * (4 if latency == "b-slower-4x" else 20)


@dataclass
class Watched:
    """What one watcher came to: whether the product separated in any reading, and whether the record-order and
    file-order rules did over the same stored answers; and, where every interrupt drained, whether no cell drew twice
    and whether the bet over every case the comparison launched, in its committed order, each at its first draws, ever
    reached 1/α (``path_crossed``), which the product's readings, each a prefix of that path, cannot pass."""

    product: bool
    recorded: bool
    filed: bool
    drew_once: bool = True
    path_crossed: bool = False
    readings: int = 0
    launched: list[list[str]] = field(default_factory=list)


async def watched(
    tag: str, *, latency: Latency, stop: Stop, k: int, cut_share: float = 0.0, launches: int = 5
) -> Watched:
    """One newcomer under the null who watches each run's answers land and stops it, through ``Eval.run`` into one
    store: two arms answering right at one rate (0.5) each call, ``k`` repeats, each answer taking as long as
    ``latency`` says, the cells played as the product plays them (case by case in the committed order, the arms in
    turn, four at a time).

    Run 20 cases; stop each run as ``stop`` says (an interrupt: the run's task cancelled, so the cells in flight finish
    and record), and, in ``cut_share`` of those stops, interrupt again 5 ms later, cutting the cells still in flight;
    read the comparison as it stands (the run's print where it finished, the filed campaign where it was stopped);
    while nothing separates, append cases and run again, at most ``launches`` runs in all, each adding ten cases plus up
    to ninety more the closer the reading's p came to α. Each reading is checked to be the bet over the longest complete
    prefix of the committed order, and read too by the two rules before it over the same stored answers
    (:func:`reading_p`'s ``recorded`` and ``filed``), at the product's family size (Holm's ``m``, ``_correct`` counted
    where the product left it untested). It stops once all three have separated on ``_correct``.
    """
    calls: Counter[tuple[int, str]] = Counter()
    landed: list[tuple[str, float]] = []
    running: list[asyncio.Task[Result]] = []
    stop_at: list[int] = []
    rules = random.Random(f"{tag}-stops")

    def interrupt() -> None:
        task = running.pop()
        loop = asyncio.get_running_loop()
        loop.call_soon(task.cancel)
        if rules.random() < cut_share:
            loop.call_later(0.005, task.cancel)

    def watching(arm: str) -> Callable[[Mapping[str, Any]], Any]:
        draw = coin(tag, arm, 0.5, calls)

        async def run(case: Mapping[str, Any]) -> str:
            answer = await draw(case)
            await asyncio.sleep(_seconds(latency, tag, arm, case["n"], calls[case["n"], arm], answer == "right"))
            landed.append((arm, 1.0 if answer == "right" else 0.0))
            if not running:
                return answer
            if stop == "blind":
                if stop_at and len(landed) >= stop_at[0]:
                    interrupt()
            else:
                mine = [v for side, v in landed if side == "a"], [v for side, v in landed if side == "b"]
                if min(map(len, mine)) >= 10 and abs(sum(mine[1]) - sum(mine[0])) >= 4:
                    interrupt()
            return answer

        run.__name__ = arm
        return run

    arms = {arm: watching(arm) for arm in "ab"}
    store = InMemoryDocumentStore()
    n = 20
    separated = [False, False, False]
    seen = Watched(False, False, False)
    runs: list[Any] = []
    for _ in range(launches):
        compare = Eval(_cases(n), arms, [_correct], k=k, name="watched")
        landed.clear()
        stop_at[:] = [rules.randint(1, 2 * k * n)] if rules.random() < 0.8 else []
        running.append(asyncio.create_task(compare.run(store=store, scope="s")))
        task = running[0]
        host = as_host([compare], store=store)
        try:
            families: Sequence[ComparisonFamily] = (await task).report.bundle.multiple_comparisons.families
        except asyncio.CancelledError:
            filed = off_loop(list_campaigns, host.storage, "s", subject_id="watched")
            campaign = max(filed, key=lambda each: (each.created_at, each.id))
            bundle = off_loop(assemble_context_bundle, campaign, storage=host.storage, profile=host.profile)
            families = bundle.multiple_comparisons.families
        running.clear()
        product, p = _separated_on_correct(families)
        campaign = max(
            off_loop(list_campaigns, host.storage, "s", subject_id="watched"),
            key=lambda each: (each.created_at, each.id),
        )
        runs, results = off_loop(stored, host.storage, campaign.id)
        (family,) = families
        (compared,) = [c for c in family.comparisons if c.name == "_correct"]
        # The product's reading is the bet over the longest complete prefix of the committed order.
        assert same_p(compared.p_raw, reading_p(runs, results)), (compared.p_raw, reading_p(runs, results))
        seen.readings += 1
        m = family.family_size + (0 if compared.p_raw is not None else 1)
        recorded, filed_order = (reading_p(runs, results, rule) for rule in ("recorded", "filed"))
        separated = [
            separated[0] or product,
            separated[1] or (recorded is not None and recorded * m < SIGNIFICANCE_ALPHA),
            separated[2] or (filed_order is not None and filed_order * m < SIGNIFICANCE_ALPHA),
        ]
        if all(separated):
            break
        n += _first_add(p)
    seen.product, seen.recorded, seen.filed = separated
    seen.drew_once = max(calls.values(), default=0) <= k
    if runs:
        launched: dict[str, list[str]] = {}
        for run in runs:
            launched.setdefault(run.launch_group_id or run.id, []).extend(run.test_case_ids)
        order = committed_order(runs[0].template_id, list(launched.values()))
        seen.launched = list(launched.values())
        indexes = [int(case.rsplit("-", 1)[1]) for case in order]
        a = [sum(coin_value(tag, i, "a", call) for call in range(1, k + 1)) / k for i in indexes]
        b = [sum(coin_value(tag, i, "b", call) for call in range(1, k + 1)) / k for i in indexes]
        path = bounded_separation_p(a, b, paired=True, value_range=RATE, mixture=True) if len(a) >= 2 else None
        seen.path_crossed = path is not None and path < SIGNIFICANCE_ALPHA
    return seen


def drained_held(seen: Watched) -> str | None:
    """Where a watcher whose every interrupt drained breaks what the drained reading rests on, as a sentence; ``None``
    where it holds: no cell drew twice, so every value read is its cell's first draw, and the product separated only
    where the bet over its whole committed path did."""
    if not seen.drew_once:
        return "a cell drew twice under a drained interrupt"
    if seen.product and not seen.path_crossed:
        return "the product separated where the bet over its whole committed path never reached 1/α"
    return None


def _separated_on_correct(families: Sequence[ComparisonFamily]) -> tuple[bool, float]:
    """Whether a reading separates the arms on ``_correct``, and its raw p there (1 where it has none)."""
    compared = [c for family in families for c in family.comparisons if c.name == "_correct"]
    return (
        any(comparison.verdict in ("improved", "regressed") for comparison in compared),
        min((c.p_raw for c in compared if c.p_raw is not None), default=1.0),
    )
