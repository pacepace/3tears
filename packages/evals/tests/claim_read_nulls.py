"""A claim instrument's calibration under known truth: seeded campaigns drawn and read as a claim read is.

Each draw is one campaign of arms, each arm its own: how many claims a cell holds, how often its claims are false
(each cell draws its own rate around the arm's, so a cell's claims are alike), and where the instrument's answers on
its claims fall. The instrument is **fair** wherever a row says so: it fails the claims people fail at one rate (its
recall) and the claims people pass at one rate (its false fails) on every arm alike, though arms differ in everything
else. That is the null the lean test is held to, and the rows that stress it differ the arms the way real ones do: one
arm whose unsettled claims mostly read ``supported`` (it researched) against one whose mostly read ``unverifiable``,
at different base rates and cell sizes, so the draw's strata, drawn at rates about three-fold apart, hold the arms'
claims in very different mixes.

A ``true`` row reads the truth question; a ``grounded`` row reads groundedness under the draw by truth verdict, where
people's groundedness and the arm's verdict mix are tied together arm by arm. People answer every drawn item as the
truth is. The draw is the product's (``draw_claim_read``) and so is the reading (``read_claim_calibration``).

Each row reports, per draw: whether kappa's one-sided 95% lower end sat above the campaign's own kappa, whether its
one-sided 97.5% upper end fell short of it, the same of the first arm's recall against the instrument's recall on that
arm (its rate, of which the campaign's claims are one draw), where the read gives that interval at all, and whether
the lean test across arms read a lean (p at most 5%). Read by the statistics certificate (``stats_certificate.py``),
never by the unit suite; ``tests/test_simulated_claim_read.py`` keeps a fast seeded case of each.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from threetears.evals.analysis.claim_read import draw_claim_read, read_claim_calibration
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA
from threetears.evals.schema.claim_read import ClaimAnswer, ClaimQuestion, ClaimReadArm, ClaimVerdict
from packages.evals.tests.claim_read_support import CRITERION, FINGERPRINT, graded_cell

#: The cells each arm holds, and the cases they are spread over.
CELLS_PER_ARM: Final = 120
CASES: Final = 24

#: The bootstrap draws each read takes: fewer than a host's read, which the interval's ends are stable under.
DRAWS: Final = 400

#: Where an arm's unsettled claims fall: ``even`` as a typical arm's; ``supported`` an arm that researched, whose
#: unsettled claims mostly read supported; ``unverifiable`` one that did not.
Mix = Literal["even", "supported", "unverifiable"]

#: Of the false claims the instrument misses, and of the true claims it neither fails nor confirms on the web, the share
#: that read ``unverifiable`` (the rest ``supported``), by mix.
_UNSETTLED: Final[Mapping[str, tuple[float, float]]] = MappingProxyType(
    {"even": (0.6, 0.1), "supported": (0.1, 0.05), "unverifiable": (0.9, 0.8)}
)


@dataclass(frozen=True)
class Arm:
    """One arm of a row: the instrument's recall on it, its false-claim rate, its mix and its claims per cell."""

    recall: float = 0.75
    false_rate: float = 1.0 / 12.0
    mix: Mix = "even"
    claims: tuple[int, int] = (2, 10)


@dataclass(frozen=True)
class Row:
    """One row of draws: its arms, the question read, the items planned, and how many draws the certificate reads."""

    arms: tuple[Arm, ...]
    question: ClaimQuestion
    planned: int
    draws: int


def _mixed(arms: int) -> tuple[Arm, ...]:
    """Arms alternating between a researched arm (few false claims, small cells, unsettled claims supported) and one
    that did not research (many false claims, large cells, unsettled claims unverifiable), the instrument fair on all."""
    researched = Arm(false_rate=0.05, mix="supported", claims=(2, 4))
    bare = Arm(false_rate=0.2, mix="unverifiable", claims=(6, 10))
    return tuple(researched if at % 2 == 0 else bare for at in range(arms))


#: The rows the certificate counts, by name: equal arms at two, three and four arms on the truth question; arms whose
#: verdict mixes, base rates and cell sizes differ sharply, the instrument fair, on truth and on groundedness; each
#: at the planned 200 labels, four thousand draws (a false lean's rate sits at the test's 5% level, so the 95% upper
#: bound a claim reads needs that many to show a one-point tolerance), and eight thousand for groundedness at four arms,
#: whose exact permutation sits at its level, where four thousand drew 5.4% and left the bound at 6.15%. Then one arm whose instrument catches 45% of its
#: false claims against one at 80%, at 400 labels, for the test's power.
ROWS: Final[Mapping[str, Row]] = MappingProxyType(
    {
        **{f"equal.{n}-arms": Row((Arm(),) * n, "true", 200, 4000) for n in (2, 3, 4)},
        **{f"mixed.{n}-arms": Row(_mixed(n), "true", 200, 4000) for n in (2, 3, 4)},
        **{f"mixed-grounded.{n}-arms": Row(_mixed(n), "grounded", 200, 8000 if n == 4 else 4000) for n in (2, 3, 4)},
        "lopsided-80-45": Row((Arm(recall=0.8), Arm(recall=0.45)), "true", 400, 1000),
    }
)

#: Groundedness, fair: the instrument fails 80% of the claims people find ungrounded and 10% of those they find grounded.
_GROUND_RECALL: Final = 0.8
_GROUND_FALSE: Final = 0.1


@dataclass(frozen=True)
class Outcome:
    """One draw's read against its campaign's truth."""

    kappa: float
    low_missed: bool
    high_missed: bool
    arm_low_missed: bool
    arm_high_missed: bool
    leaned: bool


def _verdict(rng: random.Random, false: bool, arm: Arm) -> str:
    """The instrument's truth verdict on a claim: contradicted at the arm's recall (2% of true claims), else, by the
    arm's mix, unverifiable or supported (5% of true claims true on the web)."""
    missed, unsettled = _UNSETTLED[arm.mix]
    roll = rng.random()
    if false:
        if roll < arm.recall:
            return "contradicted"
        return "unverifiable" if rng.random() < missed else "supported"
    if roll < 0.02:
        return "contradicted"
    if roll < 0.07:
        return "true_ungrounded"
    return "unverifiable" if rng.random() < unsettled else "supported"


def _ungrounded(rng: random.Random, verdict: str, arm: Arm) -> bool:
    """Whether people find a claim ungrounded: a researched arm's supported claims mostly grounded, the rest mostly
    not, so people's answer and the stratum it was drawn from are tied together, and differently by arm."""
    if arm.mix == "supported":
        return rng.random() < (0.15 if verdict == "supported" else 0.6)
    return rng.random() < 0.8


def _kappa(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    p_i = sum(x for x, _ in pairs) / n
    p_h = sum(y for _, y in pairs) / n
    observed = sum(1 for x, y in pairs if x == y) / n
    expected = p_i * p_h + (1.0 - p_i) * (1.0 - p_h)
    return (observed - expected) / (1.0 - expected)


def one(row: str, index: int) -> Outcome:
    """Draw one campaign of ``row``, draw a claim read over it, answer it as the truth is, and read it.

    :param row: A key of :data:`ROWS`.
    :param index: The draw's index, which seeds it.
    :return: How the read stood against the campaign's own kappa and the first arm's own recall.
    """
    held = ROWS[row]
    rng = random.Random(f"claim-read-{row}-{index}")
    cells = []
    people: dict[tuple[str, int], int] = {}
    pairs: list[list[tuple[int, int]]] = [[] for _ in held.arms]
    for at_arm, arm in enumerate(held.arms):
        for at in range(CELLS_PER_ARM):
            rate = rng.betavariate(1.0, (1.0 - arm.false_rate) / arm.false_rate)
            truths = [rng.random() < rate for _ in range(rng.randint(*arm.claims))]
            verdicts = [_verdict(rng, truth, arm) for truth in truths]
            result_id = f"r{at_arm}-{at}"
            if held.question == "grounded":
                fails = [int(_ungrounded(rng, verdict, arm)) for verdict in verdicts]
                instrument = [rng.random() < (_GROUND_RECALL if fail else _GROUND_FALSE) for fail in fails]
                grounded: list[bool | None] | None = [not flagged for flagged in instrument]
            else:
                fails = [int(truth) for truth in truths]
                grounded = None
            cell = graded_cell(f"run-{at_arm}", result_id, verdicts, case_id=f"case-{at % CASES}", grounded=grounded)
            cells.append(cell)
            for place, (claim, fail) in enumerate(zip(cell.grade.claims, fails, strict=True)):
                people[(result_id, place)] = fail
                x = int(claim.verdict == "contradicted") if held.question == "true" else int(claim.grounded is False)
                pairs[at_arm].append((x, fail))
    read = draw_claim_read(
        cells,
        [ClaimReadArm(key=f"arm-{at}", run_ids=[f"run-{at}"]) for at in range(len(held.arms))],
        criterion=CRITERION,
        criterion_fingerprint=FINGERPRINT,
        questions=[held.question],
        scope_id="sim",
        created_by="sim",
        seed=index,
        planned_items=held.planned,
        created_at="2026-11-22T00:00:00+00:00",
    )
    answers = [
        ClaimAnswer(
            item=item.item,
            rater="person",
            answered_at="2026-11-22T00:00:00+00:00",
            verdicts=[ClaimVerdict(question=held.question, score=1 - people[(item.result_id, item.claim)])],
        )
        for item in read.items
    ]
    reading = read_claim_calibration(read.model_copy(update={"answers": answers}), draws=DRAWS).questions[0]
    kappa = _kappa([pair for arm in pairs for pair in arm])
    # An arm's recall is read as the instrument's behaviour on it: its rate, not the campaign's realised share.
    recall = held.arms[0].recall if held.question == "true" else _GROUND_RECALL
    first = reading.arms[0].recall
    return Outcome(
        kappa=kappa,
        low_missed=reading.kappa.lower is None or reading.kappa.lower > kappa,
        high_missed=reading.kappa.upper is None or reading.kappa.upper < kappa,
        arm_low_missed=first.lower is not None and first.lower > recall,
        arm_high_missed=first.upper is not None and first.upper < recall,
        leaned=reading.lean_p is not None and reading.lean_p <= SIGNIFICANCE_ALPHA,
    )
