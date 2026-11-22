"""A claim instrument's calibration under known truth: seeded campaigns drawn and read as a claim read is.

Each draw is one campaign: per arm, cells of two to ten world claims each, every claim true or false by a rate the
cell draws (Beta(1, 11), about 8% false, so a cell's claims are alike), and the instrument's truth verdict drawn from
whether the claim is false and the arm's recall (a false claim it misses reads ``unverifiable`` or ``supported``; a
true claim reads ``contradicted`` 2% of the time). People answer every drawn item as the truth is, so their kappa
against the instrument over the whole campaign is the figure the read estimates. The draw is the product's
(``draw_claim_read``: stratified by verdict, allocated by expected misses, two claims a cell at most) and so is the
reading (``read_claim_calibration``: the weighted 2x2, the smoothed stratified bootstrap, the lean test).

Each row reports, per draw: whether kappa's lower end sat above the campaign's kappa, whether its upper end fell short
of it, and whether the lean test across arms read a lean (p at most 5%). Under equal recall a lean is a false one,
which the certified ``claim-read.*.lean`` claims bound; under unequal recall it is the test's power, which one claim
floors. Read by the statistics certificate (``stats_certificate.py``), never by the unit suite;
``tests/test_simulated_claim_read.py`` keeps a fast seeded case of each.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from threetears.evals.analysis.claim_read import draw_claim_read, read_claim_calibration
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA
from threetears.evals.schema.claim_read import ClaimAnswer, ClaimReadArm, ClaimVerdict
from packages.evals.tests.claim_read_support import CRITERION, FINGERPRINT, graded_cell

#: The cells each arm holds, and the cases they are spread over.
CELLS_PER_ARM: Final = 120
CASES: Final = 24

#: The bootstrap draws each read takes: fewer than a host's read, which the interval's ends are stable under.
DRAWS: Final = 400


@dataclass(frozen=True)
class Row:
    """One row of draws: each arm's recall, the items planned, and how many draws the certificate reads of it."""

    recalls: tuple[float, ...]
    planned: int
    draws: int


#: The rows the certificate counts, by name: equal recall at two, three and four arms at the planned 200 labels, then
#: one arm whose instrument catches 45% of its false claims against one at 80%, at 400 labels.
ROWS: Final[Mapping[str, Row]] = MappingProxyType(
    {
        **{f"equal.{arms}-arms": Row((0.75,) * arms, 200, 2000) for arms in (2, 3, 4)},
        "lopsided-80-45": Row((0.8, 0.45), 400, 1000),
    }
)


@dataclass(frozen=True)
class Outcome:
    """One draw's read against its campaign's truth."""

    kappa: float
    low_missed: bool
    high_missed: bool
    leaned: bool


def _verdict(rng: random.Random, false: bool, recall: float) -> str:
    """The instrument's truth verdict on a claim, by whether it is false and the arm's recall."""
    roll = rng.random()
    if false:
        if roll < recall:
            return "contradicted"
        return "unverifiable" if roll < recall + (1.0 - recall) * 0.6 else "supported"
    if roll < 0.02:
        return "contradicted"
    if roll < 0.12:
        return "unverifiable"
    return "true_ungrounded" if roll < 0.17 else "supported"


def one(row: str, index: int) -> Outcome:
    """Draw one campaign of ``row``, draw a claim read over it, answer it as the truth is, and read it.

    :param row: A key of :data:`ROWS`.
    :param index: The draw's index, which seeds it.
    :return: How the read stood against the campaign's own kappa.
    """
    held = ROWS[row]
    rng = random.Random(f"claim-read-{row}-{index}")
    cells = []
    false: dict[tuple[str, int], bool] = {}
    for arm, recall in enumerate(held.recalls):
        for at in range(CELLS_PER_ARM):
            rate = rng.betavariate(1.0, 11.0)
            truths = [rng.random() < rate for _ in range(rng.randint(2, 10))]
            result_id = f"r{arm}-{at}"
            cells.append(
                graded_cell(
                    f"run-{arm}",
                    result_id,
                    [_verdict(rng, truth, recall) for truth in truths],
                    case_id=f"case-{at % CASES}",
                )
            )
            false.update(((result_id, place), truth) for place, truth in enumerate(truths))
    pairs = [
        (int(claim.verdict == "contradicted"), int(false[(cell.result_id, place)]))
        for cell in cells
        for place, claim in enumerate(cell.grade.claims)
    ]
    n = len(pairs)
    p_i = sum(x for x, _ in pairs) / n
    p_h = sum(y for _, y in pairs) / n
    observed = sum(1 for x, y in pairs if x == y) / n
    expected = p_i * p_h + (1.0 - p_i) * (1.0 - p_h)
    kappa = (observed - expected) / (1.0 - expected)
    read = draw_claim_read(
        cells,
        [ClaimReadArm(key=f"arm-{arm}", run_ids=[f"run-{arm}"]) for arm in range(len(held.recalls))],
        criterion=CRITERION,
        criterion_fingerprint=FINGERPRINT,
        questions=["true"],
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
            verdicts=[ClaimVerdict(question="true", score=0 if false[(item.result_id, item.claim)] else 1)],
        )
        for item in read.items
    ]
    reading = read_claim_calibration(read.model_copy(update={"answers": answers}), draws=DRAWS).questions[0]
    low, high = reading.kappa.lower, reading.kappa.upper
    return Outcome(
        kappa=kappa,
        low_missed=low is None or low > kappa,
        high_missed=high is None or high < kappa,
        leaned=reading.lean_p is not None and reading.lean_p <= SIGNIFICANCE_ALPHA,
    )
