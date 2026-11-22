"""Claim losses under the null, and one lopsided loss: seeded campaigns read by the bundle's claim-loss test.

Each draw is one campaign of twelve cases: one arm per entry of a row's losses, each arm losing each of its cells to a
claim criterion (``rules.claims``, grounded in the arm's own documents) at its own share, or at a share its case sets,
independently. The draw reads the bundle's judged measures over the projection's rows, as assembly does, and reports
whether the false claims per unit is withheld (``trials_lost``). Under equal loss a withhold is a false one, which the
certified ``claim-loss.*`` claims bound at the test's 5% level at two, three and four arms; under a lopsided loss it is
the test's power, which one claim floors.

Read by the statistics certificate (``stats_certificate.py``), never by the unit suite: a row of a thousand draws runs
for minutes on the pool. ``tests/test_claim_ranking.py`` keeps one fast seeded case of the permutation branch.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from threetears.evals.analysis.bundle.cell_reads import judged_measures
from threetears.evals.analysis.reporting import project_score_records
from threetears.evals.schema import EvalResult
from threetears.evals.schema.models import ClaimJudgeCounts, ClaimTally
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The criterion every draw grades, and the cases each draw runs.
CRITERION: Final = "rules.claims"
CASES: Final = 12


def _clustered(case: int) -> float:
    """A loss three of the twelve cases carry most of: 36% of their cells, 1.3% of the rest's, about 10% overall."""
    return 0.36 if case < 3 else 0.013


@dataclass(frozen=True)
class Row:
    """One row of draws: each arm's loss (a share, or a share by case) and the repeats each case runs."""

    losses: tuple[float | Callable[[int], float], ...]
    repeats: int


#: The rows the certificate counts, by name: equal loss at two, three and four arms, at 10% over two repeats, at 3%, at
#: one repeat a case and where a few cases carry most of it; then one arm losing nothing against one losing 30%.
ROWS: Final[Mapping[str, Row]] = MappingProxyType(
    {
        **{
            f"{row}.{arms}-arms": Row((loss,) * arms, repeats)
            for row, loss, repeats in (
                ("equal-10", 0.10, 2),
                ("equal-3", 0.03, 2),
                ("one-repeat", 0.10, 1),
                ("clustered", _clustered, 2),
            )
            for arms in (2, 3, 4)
        },
        "lopsided-0-30": Row((0.0, 0.30), 2),
    }
)


def _tally() -> ClaimTally:
    """A kept cell's tally: four asserted specific world claims over one unit, none false, three grounded."""
    counts = ClaimJudgeCounts(
        served_model="judge-a",
        judge_temperature=0.0,
        judge_sent_settings=None,
        claims=4,
        asserted=4,
        hedged=0,
        attributed=0,
        asserted_specific=4,
        contradicted=0,
        asserted_contradicted=0,
        hedged_contradicted=0,
        attributed_contradicted=0,
        grounded_specific=3,
        unverifiable=0,
        true_ungrounded=0,
        web_decided=0,
        evidence_conflicts=0,
        web_checked=0,
        web_skipped=0,
    )
    return ClaimTally(
        criterion_fingerprint="c" * 64,
        documents_declared=True,
        grounded=True,
        units=1,
        units_with_false=0,
        session_claims=0,
        by_judge=(counts,),
    )


def withheld(row: str, seed: int) -> bool:
    """Whether draw ``seed`` of ``row`` withholds the false claims per unit.

    :param row: The row's name, a key of :data:`ROWS`.
    :param seed: The draw.
    :return: Whether the reading is withheld.
    """
    held = ROWS[row]
    runs = [
        make_eval_run(status="completed", candidate_model=f"arm-{index}", template_id="tpl-rules")
        for index in range(len(held.losses))
    ]
    draw = random.Random(f"claim-loss-{row}-{seed}")
    kept = _tally()
    results: list[EvalResult] = []
    for index, (run, loss) in enumerate(zip(runs, held.losses, strict=True)):
        for case in range(CASES):
            for k in range(1, held.repeats + 1):
                lost = draw.random() < (loss(case) if callable(loss) else loss)
                results.append(
                    make_eval_result(
                        id=f"arm-{index}-{case:02d}-{k}",
                        eval_run_id=run.id,
                        scope_id=run.scope_id,
                        model=f"arm-{index}",
                        test_case_id=f"tc-{case:02d}",
                        k_iteration=k,
                        goal_state_outcomes=[],
                        rubric_scores=[],
                        cost_usd=None,
                        claim_tallies={} if lost else {CRITERION: kept},
                        claim_failures={CRITERION: "cut_short"} if lost else {},
                    )
                )
    records = project_score_records(runs, results, archived_run_ids=None, profile=toyhost_profile()).records
    by_cell = {
        (f"arm-{index}", "rig"): [result for result in results if result.model == f"arm-{index}"]
        for index in range(len(held.losses))
    }
    measured = judged_measures(records, by_cell, None, tiers=[])
    return next(m for m in measured if m.name == f"{CRITERION}_false_per_unit").withheld is not None


__all__ = ["CASES", "CRITERION", "ROWS", "Row", "withheld"]
