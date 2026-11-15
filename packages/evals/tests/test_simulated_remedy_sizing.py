"""The plan that lifts a caveat: seeded simulations of the real path, and the typed states where no plan can be read.

A step that adds cases promises that, taken as planned, it lifts its caveat by the rule that raised it four times in
five if the truth is today's estimate. The planner sizes it on the per-case values the reading was read from
(``ReportInput.per_case``, :func:`~threetears.evals.analysis.separation.cases_to_lift`), or on a figure's stated spread
(:func:`~threetears.evals.analysis.separation.figure_cases_to_lift`), with the cases held fixed.

**What each simulation fixes.** A held sample is drawn from a truth (seeded), kept where its reading is unsettled past
a 10-point margin (a gate: borderline) and its per-case values vary. Its plan is read through the report's own planner
(:func:`~threetears.evals.analysis.caveats.plan_of` on the input with its per-case part). Then, for each design, that
many cases are added, drawn like the held ones (truth = today's estimate, held sample fixed), at case ids that fall
among the held ones at random, and the reading is read again by its own rule: the bounded interval at the reading's
level against the margin's line, as :func:`~threetears.evals.analysis.answer.unsettled_arms` reads it, for a contrast;
the figure's mean interval against the gate's line for a gate. Each design is read off the interval directly, and the
first designs of every held sample are also read through the report's own raiser (``unsettled_arms``; the gate's status
and the implied raiser), which must agree. A row's rate is the share of designs that lift, over every held sample of
the row. On an unpaired held contrast (each arm's own generated cases) the extend first cross-fills (each arm runs the
held cases only the other ran, in a seeded order, up to the pairs planned) and draws the rest as new shared cases; the
cross-filled half is drawn independently of the held half (no correlation between the arms).

**Measured** (seed 20261115; every row is held in [0.78, 0.84]):

=======================  ======  ======  ======  ======  ======
row                      n0=3    n0=5    n0=10   n0=30   n0=100
=======================  ======  ======  ======  ======  ======
paired, pass/fail k = 1  0.789   0.810   0.815   0.800   0.810
paired, pass/fail k = 3  0.840   0.831   0.819   0.820   0.833
paired, bounded mean     0.817   0.804   0.833   0.818   0.828
cross-filled, k = 1      0.825   0.820   0.827   0.827   0.838
cross-filled, mean       0.825   0.840   0.829   0.823   0.819
gate figure              --      0.812   0.798   0.803   0.809
=======================  ======  ======  ======  ======  ======

A gate's figure under five cases states no interval (``SMALL_N_BAND_FLOOR``), so it is a point estimate sized to the
floor and never borderline: the gate rows start at five.

**Recorded, not held.** Arms that agree case by case lift a cross-filled plan more often (``rho`` 0, 0.3, 0.6: k = 1
at thirty cases 0.815, 0.918, 0.980; a bounded mean 0.803, 0.925, 0.978). Across campaigns (the truth fixed, a fresh
held sample each time, the new cases drawn from the truth) a plan lifts 0.469 of the time on pass/fail pairs at thirty
cases and 0.812 on a bounded mean: a plan states its chance for the campaign it was read on, not for every campaign.

**What it refuses.** The size before this one (the whole design sized to clear four in five, less the cases held,
read off the reading's interval by a normal approximation) lifts these rows far more often than it says (k = 1 at
thirty cases 0.923, a bounded mean 0.998; a gate whose held cases are half the design, 0.883 at thirty and a hundred);
and a size read at the one-sided level lifts them less (0.708, 0.583; a gate 0.566 and 0.570). Each is run through the
same simulation and must fall outside the band.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from threetears.evals.analysis import answer, criteria, stats
from threetears.evals.analysis.caveats import Plan, caveat_step, implied, plan_of, raise_caveats
from threetears.evals.analysis.report_input import CaseSamples, CaveatEntry, ReportInput, report_input_digest
from threetears.evals.analysis.separation import (
    POWER,
    Spread,
    cases_to_lift,
    cases_to_separate,
    figure_cases_to_lift,
    stat_spread,
)
from threetears.evals.schema.reading_states import StateOf

FIXTURES: Final = Path(__file__).resolve().parent / "fixtures"

_SEED: Final = 20261115
_LEVEL: Final = stats.INTERVAL_LEVEL
_BAND: Final = (0.78, 0.84)

#: The aspect the contrast rows read on the reasoning-effort campaign, and its margin in points.
_ASPECT, _MARGIN = "queue discipline", 10.0
_LINE: Final = -_MARGIN / 100

#: Held samples per row, designs per held sample, and how many of each sample's designs are also read through the
#: report's own raiser (:func:`_lifted`). A row on a few cases spreads more between held samples (a plan on three
#: pass/fail pairs is coarse), so it reads more of them and fewer designs each (:func:`_paired_size`); a row whose
#: plans read thousands of pairs reads fewer, within a test's time.
_HELD, _DESIGNS, _CHECKED = 24, 200, 2

#: The gate the gate rows read (r1: resolves, in the trouble task), its arm, its place and its line.
_GATE, _GATED, _GATE_TASK, _GATE_LINE = "trouble.resolves", "atlas-m", "trouble", 90.0

#: The truths the contrast rows draw from: each case's (control, candidate) values, higher better.
Pairs = list[tuple[float, float]]

#: A pass/fail pair per case, k = 1: a 4-point drop with most pairs agreeing (60 both pass, 10 the control only, 6 the
#: candidate only, 24 neither, per 100).
_K1: Final = (((1.0, 1.0), 60), ((1.0, 0.0), 10), ((0.0, 1.0), 6), ((0.0, 0.0), 24))


def _k1(rng: random.Random, count: int) -> Pairs:
    return rng.choices([pair for pair, _ in _K1], weights=[weight for _, weight in _K1], k=count)


def _k3(rng: random.Random, count: int) -> Pairs:
    """A share of three trials per case on each arm, the case's difficulty shared by both: a 4-point drop."""
    drawn = []
    for _ in range(count):
        chance = 0.35 + 0.6 * rng.random()
        control = sum(rng.random() < chance for _ in range(3)) / 3
        candidate = sum(rng.random() < chance - 0.04 for _ in range(3)) / 3
        drawn.append((control, candidate))
    return drawn


def _mean(rng: random.Random, count: int) -> Pairs:
    """A bounded score per case on [0, 1], the case's level shared by both arms: a 3-point drop."""
    drawn = []
    for _ in range(count):
        level = rng.betavariate(4.0, 2.0)
        control = min(1.0, max(0.0, level + rng.gauss(0.0, 0.15)))
        candidate = min(1.0, max(0.0, level - 0.03 + rng.gauss(0.0, 0.15)))
        drawn.append((control, candidate))
    return drawn


_TRUTHS: Final[dict[str, Callable[[random.Random, int], Pairs]]] = {"k1": _k1, "k3": _k3, "mean": _mean}


# --- the real path ----------------------------------------------------------------------------------------------------

_BASE: list[ReportInput] = []


def _base() -> ReportInput:
    """The reasoning-effort campaign's frozen input, whose aspects each read two candidates against the control."""
    if not _BASE:
        path = FIXTURES / "writer_cases" / "held_out" / "discodon-1017-reasoning-effort-01a18d4d.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            held = ReportInput.model_validate(json.load(handle)["snapshot"]["report_input"])
        _BASE.append(held.model_copy(update={"caveat_list": None, "first_screen": None, "per_case": None}))
    return _BASE[0]


def _contrast_input(
    control: Sequence[float], contrast: Sequence[float], *, paired: bool, samples: bool = True
) -> ReportInput:
    """The campaign with both candidates' readings on :data:`_ASPECT` read from these per-case values as the pipeline
    reads them (the bounded interval at the nominal level, against a 10-point margin), and, where ``samples``, the
    per-case part the builder keeps for them."""
    held = _base()
    bounded = stats.bounded_difference_interval(
        control, contrast, paired=paired, value_range=(0.0, 1.0), confidence=_LEVEL
    )
    assert bounded is not None
    interval = (bounded[0] * 100, bounded[1] * 100)
    change = (sum(contrast) / len(contrast) - sum(control) / len(control)) * 100
    update = {
        "change": change,
        "within_interval": interval,
        "gain_interval": interval,
        "gain_confidence": _LEVEL,
        "margin": _MARGIN,
        "cases": [len(control), len(contrast)],
    }
    aspects = [
        aspect.model_copy(
            update={"margin": _MARGIN, "readings": [reading.model_copy(update=update) for reading in aspect.readings]}
        )
        if aspect.id == _ASPECT
        else aspect
        for aspect in held.aspects
    ]
    per_case = None
    if samples:
        kept = CaseSamples(
            control=list(control),
            contrast=list(contrast),
            paired=paired,
            only_control=0 if paired else len(control),
            only_contrast=0 if paired else len(contrast),
            value_range=(0.0, 1.0),
            confidence=_LEVEL,
            scale=100.0,
        )
        per_case = {
            f"{_ASPECT}/passed/{reading.arm}": kept
            for aspect in held.aspects
            if aspect.id == _ASPECT
            for reading in aspect.readings
        }
    return held.model_copy(update={"aspects": aspects, "per_case": per_case})


def _unsettled(report_input: ReportInput) -> CaveatEntry | None:
    return next(
        (
            entry
            for entry in raise_caveats(report_input)
            if entry.kind == "unsettled" and f"{_ASPECT}/passed" in entry.readings
        ),
        None,
    )


def _lifted(pairs: Pairs, *, check: bool = False) -> bool:
    """Whether the reading over ``pairs`` (in case-id order) is no longer unsettled, by the rule that raises it: its
    bounded interval at the reading's level, in points, no longer reaching both past the margin's line and short of it
    (``answer.unsettled_arms``: unsettled while the low end is at or past the line and the high end short of it).

    That rule is read here off the interval directly, since a design is read thousands of times a row; where ``check``,
    the design is also read through the report's own raiser on the whole input, and the two must agree."""
    control, contrast = [a for a, _ in pairs], [b for _, b in pairs]
    bounded = stats.bounded_difference_interval(
        control, contrast, paired=True, value_range=(0.0, 1.0), confidence=_LEVEL
    )
    assert bounded is not None
    low, high = bounded[0] * 100, bounded[1] * 100
    lifted = low > -_MARGIN or high <= -_MARGIN
    if check:
        read = _contrast_input(control, contrast, paired=True, samples=False)
        assert lifted == (f"unsettled:{_ASPECT}" not in answer.unsettled_arms(read)), (low, high)
    return lifted


def _among(rng: random.Random, held: Pairs, new: Pairs) -> Pairs:
    """``held`` and ``new`` in case-id order: the held keep their order, and new ids fall among them at random."""
    slots = set(rng.sample(range(len(held) + len(new)), len(new)))
    fresh, kept = iter(new), iter(held)
    return [next(fresh) if place in slots else next(kept) for place in range(len(held) + len(new))]


def _paired_rate(held: Pairs, cases: int, rng: random.Random, designs: int) -> float:
    return (
        sum(
            _lifted(_among(rng, held, [rng.choice(held) for _ in range(cases)]), check=design < _CHECKED)
            for design in range(designs)
        )
        / designs
    )


def _partner(value: float, own: Sequence[float], other: Sequence[float], rng: random.Random, rho: float) -> float:
    """The other arm's value on the case ``value`` was read on: with chance ``rho`` the other side's value at the same
    quantile (ties broken at random), else one drawn from the other side's held values. Either way it is drawn like the
    other side's held values, so ``rho`` couples the arms without moving either arm's truth."""
    if rho <= 0 or rng.random() >= rho:
        return rng.choice(other)
    ranked = sorted(own)
    place = rng.choice([index for index, held in enumerate(ranked) if held == value])
    quantile = (place + rng.random()) / len(ranked)
    return sorted(other)[min(int(quantile * len(other)), len(other) - 1)]


def _crossed_rate(
    control: list[float], contrast: list[float], cases: int, rng: random.Random, designs: int, rho: float = 0.0
) -> float:
    """Each design cross-fills up to ``cases`` held cases in a seeded order, the other half drawn like that side's held
    values (coupled to the held half by ``rho``, :func:`_partner`), and draws the rest as new pairs, coupled alike; the
    held cases not cross-filled leave the contrast."""
    lifted = 0
    for design in range(designs):
        order = [("control", index) for index in range(len(control))]
        order += [("contrast", index) for index in range(len(contrast))]
        rng.shuffle(order)
        pairs = [
            (control[index], _partner(control[index], control, contrast, rng, rho))
            if side == "control"
            else (_partner(contrast[index], contrast, control, rng, rho), contrast[index])
            for side, index in order[:cases]
        ]
        for _ in range(cases - len(pairs)):
            first = rng.choice(control)
            pairs.append((first, _partner(first, control, contrast, rng, rho)))
        rng.shuffle(pairs)
        lifted += _lifted(pairs, check=design < _CHECKED)
    return lifted / designs


def _paired_size(count: int) -> tuple[int, int]:
    """The held samples and designs each a paired row reads (:data:`_HELD`)."""
    return (_HELD, _DESIGNS)


def _paired_samples(truth: str, count: int, rng: random.Random, samples: int = _HELD) -> list[tuple[Pairs, Plan]]:
    """``samples`` held samples of ``count`` pairs that read unsettled, whose differences vary, with their plans."""
    found: list[tuple[Pairs, Plan]] = []
    while len(found) < samples:
        held = _TRUTHS[truth](rng, count)
        if len({b - a for a, b in held}) < 2:
            continue
        report = _contrast_input([a for a, _ in held], [b for _, b in held], paired=True)
        caveat = _unsettled(report)
        if caveat is None:
            continue
        plan = plan_of(report, caveat)
        assert plan is not None
        if plan.cases is not None:
            found.append((held, plan))
    return found


# --- the paired rows --------------------------------------------------------------------------------------------------

_ROWS: Final = [(truth, count) for truth in ("k1", "k3", "mean") for count in (3, 5, 10, 30, 100)]


@pytest.mark.parametrize(("truth", "count"), _ROWS, ids=[f"{truth}-{count}" for truth, count in _ROWS])
def test_a_paired_contrast_s_plan_lifts_it_four_times_in_five_with_the_held_cases_fixed(
    truth: str, count: int, record_property: Callable[[str, object], None]
) -> None:
    rng = random.Random(f"{_SEED}-paired-{truth}-{count}")
    held_samples, designs = _paired_size(count)
    samples = _paired_samples(truth, count, rng, held_samples)
    rate = sum(_paired_rate(held, plan.cases or 0, rng, designs) for held, plan in samples) / len(samples)
    record_property("measured", f"paired {truth} n0={count}: {rate:.3f}")
    assert _BAND[0] <= rate <= _BAND[1], rate


# --- the cross-filled rows --------------------------------------------------------------------------------------------

_CROSSED: Final = [(truth, count) for truth in ("k1", "mean") for count in (3, 5, 10, 30, 100)]


def _crossed_size(truth: str, count: int) -> tuple[int, int]:
    """The held samples and designs each a cross-filled row reads (:data:`_HELD`): pass/fail arms of a hundred cases
    each plan thousands of pairs."""
    return (12, 100) if (truth, count) == ("k1", 100) else (_HELD, _DESIGNS)


def _crossed_samples(
    truth: str, count: int, rng: random.Random, samples: int = _HELD
) -> list[tuple[list[float], list[float], int]]:
    """``samples`` unpaired held samples of ``count`` cases a side that read unsettled, with the pairs each plan
    gains, its cross-fill counted per arm as the plan states it."""
    found: list[tuple[list[float], list[float], int]] = []
    while len(found) < samples:
        control = [a for a, _ in _TRUTHS[truth](rng, count)]
        contrast = [b for _, b in _TRUTHS[truth](rng, count)]
        if len(set(control)) < 2 and len(set(contrast)) < 2:
            continue
        report = _contrast_input(control, contrast, paired=False)
        caveat = _unsettled(report)
        if caveat is None:
            continue
        plan = plan_of(report, caveat)
        assert plan is not None
        if plan.cases is None:
            continue
        held_control = answer.control_arm(report)
        assert held_control is not None
        assert dict(plan.cross_fill)[held_control] == 2 * count, "the control runs both candidates' held cases"
        assert all(dict(plan.cross_fill)[arm] == count for arm in caveat.arms)
        found.append((control, contrast, plan.cases))
    return found


@pytest.mark.parametrize(("truth", "count"), _CROSSED, ids=[f"{truth}-{count}" for truth, count in _CROSSED])
def test_an_unpaired_contrast_s_plan_cross_fills_and_lifts_it_four_times_in_five(
    truth: str, count: int, record_property: Callable[[str, object], None]
) -> None:
    """Each arm ran ``count`` cases of its own (two launches, each generating its own): the contrast reads unpaired.
    The plan counts the held cases each arm would cross-fill, and the pairs it gains lift the reading four times in
    five where the arms do not agree case by case (``rho`` 0)."""
    rng = random.Random(f"{_SEED}-crossed-{truth}-{count}")
    held_samples, designs = _crossed_size(truth, count)
    samples = _crossed_samples(truth, count, rng, held_samples)
    rate = sum(_crossed_rate(control, contrast, cases, rng, designs) for control, contrast, cases in samples) / len(
        samples
    )
    record_property("measured", f"crossed {truth} n0={count}: {rate:.3f}")
    assert _BAND[0] <= rate <= _BAND[1], rate


_COUPLED: Final = (("k1", 30), ("mean", 30))


@pytest.mark.parametrize(("truth", "count"), _COUPLED, ids=[f"{truth}-{count}" for truth, count in _COUPLED])
def test_arms_that_agree_case_by_case_lift_a_cross_filled_plan_more_often(
    truth: str, count: int, record_property: Callable[[str, object], None]
) -> None:
    """Recorded, not held: the plan assumes the arms do not agree case by case, the most a cross-filled pair can vary,
    so arms coupled at ``rho`` 0.3 and 0.6 (:func:`_partner`) lift the same plans at least as often as at 0."""
    rng = random.Random(f"{_SEED}-coupled-{truth}-{count}")
    samples = _crossed_samples(truth, count, rng, _HELD // 3)
    rates = {}
    for rho in (0.0, 0.3, 0.6):
        drawn = random.Random(f"{_SEED}-coupled-{truth}-{count}-designs")
        rates[rho] = sum(
            _crossed_rate(control, contrast, cases, drawn, _DESIGNS // 2, rho) for control, contrast, cases in samples
        ) / len(samples)
    record_property("measured", f"coupled {truth} n0={count}: {rates}")
    assert rates[0.6] >= rates[0.3] - 0.02 and rates[0.3] >= rates[0.0] - 0.02, rates
    assert rates[0.6] >= _BAND[0], rates


_UNCONDITIONAL: Final = (("k1", 30), ("mean", 30))


@pytest.mark.parametrize(("truth", "count"), _UNCONDITIONAL, ids=[f"{t}-{c}" for t, c in _UNCONDITIONAL])
def test_the_unconditional_rate_is_recorded_beside_the_conditional_one(
    truth: str, count: int, record_property: Callable[[str, object], None]
) -> None:
    """Recorded, not held: the truth fixed, a fresh held sample each replicate (kept where it reads unsettled and is
    planned), its planned cases drawn from the truth rather than from the held sample. This is the rate across
    campaigns, not the one a plan states for its own; it is held only to lie strictly between nothing and certainty."""
    rng = random.Random(f"{_SEED}-unconditional-{truth}-{count}")
    lifted = replicates = 0
    while replicates < 32:
        held = _TRUTHS[truth](rng, count)
        if len({b - a for a, b in held}) < 2:
            continue
        report = _contrast_input([a for a, _ in held], [b for _, b in held], paired=True)
        caveat = _unsettled(report)
        if caveat is None:
            continue
        plan = plan_of(report, caveat)
        assert plan is not None
        if plan.cases is None:
            continue
        replicates += 1
        lifted += _lifted(_among(rng, held, _TRUTHS[truth](rng, plan.cases)))
    rate = lifted / replicates
    record_property("measured", f"unconditional {truth} n0={count}: {rate:.3f}")
    assert 0.0 < rate < 1.0, rate


# --- the gate rows ----------------------------------------------------------------------------------------------------

_R1: list[ReportInput] = []


def _gate_input(values: Sequence[float]) -> ReportInput:
    """r1 with the gated arm's figure on the gate read from ``values`` by the pipeline's own mean interval (its
    clustered standard error and :func:`~threetears.evals.analysis.stats.mean_interval`, at the table's 95%), and the
    other survivor's figure cleared, so the borderline caveat is that arm's alone."""
    if not _R1:
        path = FIXTURES / "report_inputs" / "r1" / "input.json"
        _R1.append(ReportInput.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    held = _R1[0]
    count = len(values)
    mean = sum(values) / count
    sem = stats.clustered_standard_error(list(values), list(range(count)))
    assert sem is not None
    interval = stats.mean_interval(mean, sem, count)
    assert interval is not None
    figure = {"mean": mean, "ci": interval, "half_width": None, "sem": sem, "n": count, "n_cases": count}
    tasks = []
    for task in held.tasks:
        if task.id == _GATE_TASK:
            per_arm = {arm: dict(stats_of) for arm, stats_of in task.per_arm.items()}
            per_arm[_GATED]["resolves"] = per_arm[_GATED]["resolves"].model_copy(update=figure)
            per_arm["atlas-l"]["resolves"] = per_arm["atlas-l"]["resolves"].model_copy(update={"half_width": 0.5})
            task = task.model_copy(update={"per_arm": per_arm})
        tasks.append(task)
    return held.model_copy(update={"tasks": tasks, "caveat_list": None})


def _gate_lifted(values: Sequence[float], *, check: bool = False) -> bool:
    """Whether the gate's figure over ``values`` lifts its borderline caveat on its passing side: its mean interval
    (the pipeline's clustered standard error and :func:`~threetears.evals.analysis.stats.mean_interval`) clears the
    gate's line. Read off the interval directly; where ``check``, also through the report's own gate status and
    raiser, and the two must agree."""
    count = len(values)
    sem = stats.clustered_standard_error(list(values), list(range(count)))
    assert sem is not None
    interval = stats.mean_interval(sum(values) / count, sem, count)
    assert interval is not None
    lifted = interval[0] > _GATE_LINE
    if check:
        held = _gate_input(values)
        (read,) = criteria.criterion_reads(held, _GATED, criterion=_GATE)
        real = read.status == "pass" and f"borderline:{_GATE}" not in {entry.id for entry in implied(held)}
        assert lifted == real, interval
    return lifted


_GATES: Final = (5, 10, 30, 100)


def _gate_samples(
    count: int, rng: random.Random, keep: Callable[[list[float]], bool] = lambda held: True
) -> list[tuple[list[float], int]]:
    """:data:`_HELD` held figures of ``count`` cases scoring about 93 with a per-case spread of about 25 that read
    borderline against the gate at 90 (and that ``keep`` keeps), each with the cases its plan adds. A held figure no
    count up to the search's cap lifts is typed, never sized, and is not a row's sample."""
    found: list[tuple[list[float], int]] = []
    while len(found) < _HELD:
        held = [rng.gauss(93.0, 25.0) for _ in range(count)]
        if not keep(held):
            continue
        report = _gate_input(held)
        caveat = next((entry for entry in implied(report) if entry.id == f"borderline:{_GATE}"), None)
        if caveat is None:
            continue
        plan = plan_of(report, caveat)
        assert plan is not None
        if plan.cases is None:
            assert plan.state == StateOf(state="not_available", reason="too_few"), plan
            continue
        found.append((held, plan.cases))
    return found


def _gate_rate(samples: Sequence[tuple[list[float], int]], rng: random.Random) -> float:
    """The share of designs that lift, each held figure gaining its cases drawn like its own (truth = today's
    estimate, held cases fixed)."""
    return sum(
        sum(
            _gate_lifted(held + [rng.choice(held) for _ in range(cases)], check=design < _CHECKED)
            for design in range(_DESIGNS)
        )
        / _DESIGNS
        for held, cases in samples
    ) / len(samples)


@pytest.mark.parametrize("count", _GATES)
def test_a_borderline_gate_s_plan_lifts_it_four_times_in_five_with_the_held_cases_fixed(
    count: int, record_property: Callable[[str, object], None]
) -> None:
    """Today: ``count`` cases scoring about 93 with a per-case spread of about 25 against a gate at 90, read borderline
    on the gate's t interval."""
    rng = random.Random(f"{_SEED}-gate-{count}")
    rate = _gate_rate(_gate_samples(count, rng), rng)
    record_property("measured", f"gate n0={count}: {rate:.3f}")
    assert _BAND[0] <= rate <= _BAND[1], rate


def _gate_spread(held: Sequence[float]) -> Spread:
    """The held figure's spread as the planner reads it: from its standard error, at the table's 95%."""
    count = len(held)
    sem = stats.clustered_standard_error(list(held), list(range(count)))
    assert sem is not None
    return stat_spread(sum(held) / count, sem=sem, n=count, stated_level=0.95)


def _gate_before(held: Sequence[float]) -> int:
    """The size the planner read before: the whole design to clear four in five, less the cases held."""
    spread = _gate_spread(held)
    assert spread.per_case is not None
    needed = cases_to_separate(spread.per_case, spread.mean - _GATE_LINE, spread.level, sided=2, power=POWER)
    return max(needed - len(held), 1)


def _gate_one_sided(held: Sequence[float]) -> int:
    """The size read at the one-sided level: the two-sided interval at ``2 * level - 1``."""
    spread = _gate_spread(held)
    found = figure_cases_to_lift(replace(spread, level=2 * spread.level - 1), spread.mean - _GATE_LINE)
    assert isinstance(found, int)
    return found


_GATES_REFUSED: Final = (30, 100)


@pytest.mark.parametrize("count", _GATES_REFUSED)
def test_a_gate_s_size_before_it_and_one_read_one_sided_fall_outside_the_band(
    count: int, record_property: Callable[[str, object], None]
) -> None:
    """On held figures whose cases are at least half the design the size before this one cleared (where they are a
    small share, sizing the whole design and subtracting them over-asks by little, so a borderline gate's size before
    lifted it about 84% of the time across held figures): the size before lifts far more often than four in five, and
    one read one-sided less often, while the plan holds the band on the same figures."""
    rng = random.Random(f"{_SEED}-gate-refused-{count}")
    samples = _gate_samples(count, rng, keep=lambda held: _gate_before(held) <= len(held))
    planned = _gate_rate(samples, rng)
    before = _gate_rate([(held, _gate_before(held)) for held, _ in samples], rng)
    one_sided = _gate_rate([(held, _gate_one_sided(held)) for held, _ in samples], rng)
    record_property("measured", f"gate refused n0={count}: {planned:.3f} {before:.3f} {one_sided:.3f}")
    assert _BAND[0] <= planned <= _BAND[1], planned
    assert before > _BAND[1], before
    assert one_sided < _BAND[0], one_sided


# --- what it refuses: the sizing before it, and a one-sided one -------------------------------------------------------

_REFUSED: Final = (("k1", 30), ("mean", 30))


def _rate_of(truth: str, count: int, size: Callable[[Pairs, Plan], int]) -> float:
    """The row's rate under another sizing ``size`` of held samples the planner plans (the row's own test holds the
    planner's rate on them)."""
    rng = random.Random(f"{_SEED}-refused-{truth}-{count}")
    samples = _paired_samples(truth, count, rng)
    return sum(_paired_rate(held, size(held, plan), rng, _DESIGNS // 2) for held, plan in samples) / len(samples)


def _before(held: Pairs, plan: Plan) -> int:
    """The size the planner read before the per-case values were kept: off the reading's interval, the whole design
    to clear four in five less the cases held."""
    report = _contrast_input([a for a, _ in held], [b for _, b in held], paired=True, samples=False)
    caveat = _unsettled(report)
    assert caveat is not None
    before = plan_of(report, caveat)
    assert before is not None and before.cases is not None
    return before.cases


def _one_sided(held: Pairs, plan: Plan) -> int:
    """The size read at the one-sided level: a bound at the reading's level on the one side asked, which is the
    two-sided interval at a lower level, so it asks for fewer cases than the interval the reading prints."""
    found = cases_to_lift(
        [a for a, _ in held],
        [b for _, b in held],
        paired=True,
        value_range=(0.0, 1.0),
        confidence=2 * _LEVEL - 1,
        line=_LINE,
    )
    assert isinstance(found, int)
    return found


@pytest.mark.parametrize(("truth", "count"), _REFUSED, ids=[f"{truth}-{count}" for truth, count in _REFUSED])
def test_the_sizing_before_it_lifts_far_more_often_than_it_says(
    truth: str, count: int, record_property: Callable[[str, object], None]
) -> None:
    before = _rate_of(truth, count, _before)
    record_property("measured", f"before {truth} n0={count}: {before:.3f}")
    assert before > _BAND[1], before


@pytest.mark.parametrize(("truth", "count"), _REFUSED, ids=[f"{truth}-{count}" for truth, count in _REFUSED])
def test_a_size_read_one_sided_lifts_less_often_than_it_says(
    truth: str, count: int, record_property: Callable[[str, object], None]
) -> None:
    one_sided = _rate_of(truth, count, _one_sided)
    record_property("measured", f"one-sided {truth} n0={count}: {one_sided:.3f}")
    assert one_sided < _BAND[0], one_sided


# --- degenerate inputs are typed, never a size ------------------------------------------------------------------------

_TOO_FEW: Final = StateOf(state="not_available", reason="too_few")


def test_too_few_cases_a_contrast_on_its_line_and_one_no_count_reaches_are_typed_never_a_size() -> None:
    def lift(control: Sequence[float], contrast: Sequence[float], line: float = _LINE) -> int | StateOf:
        return cases_to_lift(control, contrast, paired=True, value_range=(0.0, 1.0), confidence=_LEVEL, line=line)

    assert lift([1.0], [0.0]) == _TOO_FEW, "one case a side reads no interval"
    assert lift([1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0], line=-0.25) == _TOO_FEW, "an estimate on its line"
    near = [(1.0, 1.0)] * 60 + [(1.0, 0.0)] * 10 + [(0.0, 1.0)] * 6 + [(0.0, 0.0)] * 24
    far = lift([a for a, _ in near], [b for _, b in near], line=-0.04 - 1e-4)
    assert far == _TOO_FEW, "a size past the search's cap is no size"


def test_a_figure_with_no_count_too_few_cases_no_spread_or_on_its_line_is_typed_never_a_size() -> None:
    no_count = Spread(93.0, 5.0, None, None, _LEVEL, "sem")
    one = Spread(93.0, 5.0, 1.0, 5.0, _LEVEL, "sem")
    alike = Spread(93.0, 0.0, 20.0, 0.0, _LEVEL, "sem")
    held = Spread(93.0, 25.0 / math.sqrt(20.0), 20.0, 25.0, _LEVEL, "sem")
    assert figure_cases_to_lift(no_count, 3.0) == StateOf(state="not_available", reason="not_recorded")
    assert figure_cases_to_lift(one, 3.0) == _TOO_FEW
    assert figure_cases_to_lift(alike, 3.0) == StateOf(state="not_available", reason="no_variation")
    assert figure_cases_to_lift(held, 0.0) == _TOO_FEW, "on its line"
    assert figure_cases_to_lift(held, 1e-3) == _TOO_FEW, "past the search's cap"
    found = figure_cases_to_lift(held, 3.0)
    assert isinstance(found, int) and 20 < found < stats.PROJECTION_CASE_CAP, "near its line: the size as computed"


def test_a_plan_with_no_size_states_why_typed_and_makes_no_call() -> None:
    """A reading on its line: the planner carries the typed state beside the step's own word for it, and the step it
    gives makes no call."""
    held = _contrast_input([1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0], paired=True)
    on_line = held.model_copy(
        update={
            "aspects": [
                aspect.model_copy(
                    update={"readings": [reading.model_copy(update={"margin": 25.0}) for reading in aspect.readings]}
                )
                if aspect.id == _ASPECT
                else aspect
                for aspect in held.aspects
            ]
        }
    )
    caveat = _unsettled(on_line)
    assert caveat is not None
    plan = plan_of(on_line, caveat)
    assert plan is not None
    assert (plan.cases, plan.unsized, plan.state) == (None, "on_the_line", _TOO_FEW)
    step = caveat_step(caveat, plan=plan)
    assert (step.action, step.calls, step.planned_cases, step.unsized) == (None, [], None, "on_the_line")


def test_every_case_differing_alike_is_planned_and_lifts_every_time() -> None:
    """A contrast whose pairs all differ by one amount still narrows as cases are added (the bounded interval's width
    is not its spread alone): it is planned, and drawn like its cases it lifts with certainty."""
    held = [(1.0, 1.0)] * 6
    found = cases_to_lift(
        [a for a, _ in held], [b for _, b in held], paired=True, value_range=(0.0, 1.0), confidence=_LEVEL, line=_LINE
    )
    assert isinstance(found, int) and found >= 1
    assert _lifted(held * math.ceil((len(held) + found) / len(held)))


# --- what the planner reads -------------------------------------------------------------------------------------------


def _reasoning() -> tuple[ReportInput, CaveatEntry]:
    held = [pair for pair, weight in _K1 for _ in range(weight)]
    report = _contrast_input([a for a, _ in held], [b for _, b in held], paired=True)
    caveat = _unsettled(report)
    assert caveat is not None
    return report, caveat


def test_a_stored_input_without_per_case_values_plans_as_before() -> None:
    """With no per-case part the plan is read off the reading's interval, as before; with one, on the values."""
    report, caveat = _reasoning()
    without = report.model_copy(update={"per_case": None})
    before = plan_of(without, caveat)
    planned = plan_of(report, caveat)
    assert before is not None and planned is not None and before.cases is not None and planned.cases is not None
    assert before.cases != planned.cases
    assert (before.state, before.cross_fill) == (None, ())


def test_the_planner_reads_the_level_the_reading_was_read_at() -> None:
    report, caveat = _reasoning()
    stricter = report.model_copy(
        update={
            "per_case": {
                key: samples.model_copy(update={"confidence": 0.99}) for key, samples in (report.per_case or {}).items()
            }
        }
    )
    at_level, at_stricter = plan_of(report, caveat), plan_of(stricter, caveat)
    assert at_level is not None and at_stricter is not None
    assert at_level.cases is not None and at_stricter.cases is not None and at_stricter.cases > at_level.cases


def test_a_paired_contrast_plans_no_cross_fill_and_an_unpaired_one_plans_it_and_the_rest() -> None:
    report, caveat = _reasoning()
    paired = plan_of(report, caveat)
    assert paired is not None and paired.cross_fill == ()
    rng = random.Random(f"{_SEED}-unpaired")
    control = [a for a, _ in _k1(rng, 20)]
    contrast = [b for _, b in _k1(rng, 20)]
    unpaired = _contrast_input(control, contrast, paired=False)
    caveat = _unsettled(unpaired)
    assert caveat is not None
    plan = plan_of(unpaired, caveat)
    assert plan is not None and plan.cases is not None
    held_control = answer.control_arm(unpaired)
    assert held_control is not None
    assert dict(plan.cross_fill) == {**{arm: 20 for arm in caveat.arms}, held_control: 40}
    assert plan.arms == (*caveat.arms, held_control)


# --- nothing that identifies an input moves ---------------------------------------------------------------------------


def _bundle_case(name: str) -> ReportInput:
    from threetears.evals.analysis.campaign_bundle import CampaignBundle
    from threetears.evals.analysis.report_input_build import report_input

    path = FIXTURES / "writer_cases" / "held_out" / f"{name}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        snapshot = json.load(handle)["snapshot"]
    return report_input(CampaignBundle.from_dict(snapshot["bundle"]))


_UNSETTLED_CASES: Final = ("discodon-1017-reasoning-effort-01a18d4d", "discodon-kairo-persona-r1-01a16bd4", "s1")


@pytest.mark.parametrize("name", _UNSETTLED_CASES)
def test_the_per_case_part_is_the_values_each_unsettled_reading_was_read_from_and_outside_the_digest(name: str) -> None:
    built = _bundle_case(name)
    kept = built.per_case or {}
    unsettled = {
        f"{reading.rpartition('/')[0]}/passed/{arm}"
        for entry in built.caveat_list or []
        if entry.kind == "unsettled"
        for reading in entry.readings
        for arm in entry.arms
    }
    assert kept and set(kept) <= unsettled
    for key, samples in kept.items():
        aspect, _, arm = key.split("/")
        (reading,) = [held for each in built.aspects if each.id == aspect for held in each.readings if held.arm == arm]
        interval = stats.bounded_difference_interval(
            samples.control,
            samples.contrast,
            paired=samples.paired,
            value_range=samples.value_range,
            confidence=samples.confidence,
        )
        assert interval is not None and reading.within_interval is not None
        assert [round(end * samples.scale, 9) for end in interval] == [round(end, 9) for end in reading.within_interval]
    assert report_input_digest(built) == report_input_digest(built.model_copy(update={"per_case": None}))
    assert "per_case" in built.model_dump(mode="json", by_alias=True)


def test_an_input_no_step_sizes_on_per_case_values_holds_no_per_case_key() -> None:
    built = _bundle_case("toyhost")
    assert built.per_case is None
    assert "per_case" not in built.model_dump(mode="json", by_alias=True)
