"""A model's own check against people, checked against a known truth with a fixed seed: the shared-lean rows.

The set-up is the design review's (``design-judges-review.md`` §6): people see no difference between the arms; each
arm's answer fails people on 53% of cases; the model reviewer passes a failed candidate answer with probability ``s``
(a lean toward the candidate of about ``0.53 s``); the judge copies the reviewer with 5% noise. Every check is read
through :func:`~threetears.evals.analysis.checks.reviewer_check`, the function the close calls, and the judge against
the reviewer through the agreement computation every judge's check is.

| Row | Truth | Read over | What it holds |
|---|---|---|---|
| Kappa cannot see a shared lean | ``s`` 0.15 and 0.30 | 40 and 100 pair items | The reviewer agrees with people at kappa 0.91 and 0.83 (the review's 0.91 to 0.92 and 0.83 to 0.84, within 0.03); the judge's check against the reviewer is ``met`` in at least 95% of reads; the reviewer's kappa is never shown below the bar (at most 1%) |
| The lean is caught | lean 0.16 (``s`` 0.30) | 30 and 60 pair items | ``not_met``, ``instrument_leans``, in 75% and 99% of reads (the review's figures), within four standard errors; ``met`` at most 5% (plus four standard errors), the lean being past the bound |
| | lean 0.08 (``s`` 0.15) | 30, 60, 100 | 23%, 74%, 97% |
| No false lean | no lean; the reviewer errs both ways at 5% and 10% | 30, 60, 100 | Shown to lean at most 5%, plus four standard errors |
| ``met`` is an equivalence reading | the lean exactly at ``δ`` (0.10) | 40, 60, 105, 260 pair items | ``met`` at most 5% (the bounded test's level), plus four standard errors: measured 0%, 0%, 0.25%, 0%. The rule it replaced (the t interval inside ``±δ`` over 40 items or more) fails the row at 40: 9.35% |
| What ``met`` costs | no lean; reviewer errs both ways at 5% (pair difference sd about 0.32) | 40, 105, 260 | never at 40; about 40% at 105 (at least 30%); at least 90% at 260 |

``met`` is read by the bounded equivalence test at 5%, the house's equivalence level, not a t interval inside ``±δ``:
the t interval claimed it 9.35% of the time at the bound over 40 items and 5.0% over 60 (a few rare leans give it an
interval far too narrow). The bounded test holds its level at every n and on any distribution of leans; what it buys
is that validity, not a lower nominal rate. The statistician's arithmetic (80% at about 105 items) assumed the t
interval; the valid test costs more items.

A wrong interval fails a row: a normal multiplier on few items, or a standard error not divided by ``sqrt(n)``, moves
the false-lean rate or the power off its bar.
"""

from __future__ import annotations

import random
import pytest

from threetears.evals.analysis.agreement import instrument_agreement
from threetears.evals.analysis.checks import ReviewerCheck, reviewer_check
from threetears.evals.kernel.evidence_tiers import calibration_criterion
from packages.evals.tests.simulation_support import at_least, at_most, within

REPLICATES = 2000

#: The share of each arm's answers people fail.
FAIL_RATE = 0.53


def _read(
    rng: random.Random, items: int, *, lean_share: float, noise: float = 0.0, judge_noise: float = 0.05
) -> tuple[list[tuple[int, int, str, str]], list[float], list[tuple[int, int, str, str]]]:
    """One blind read of ``items`` pair items, every one read by a person and the model on both arms.

    :return: The model-against-person kappa pairs, the per-item leans, and the judge-against-model kappa pairs.
    """
    reviewer_vs_person: list[tuple[int, int, str, str]] = []
    judge_vs_reviewer: list[tuple[int, int, str, str]] = []
    leans: list[float] = []
    for item in range(items):
        given: dict[str, tuple[int, int]] = {}
        for arm in ("control", "candidate"):
            person = int(rng.random() >= FAIL_RATE)
            model = person
            if arm == "candidate" and person == 0 and rng.random() < lean_share:
                model = 1
            if noise and rng.random() < noise:
                model = 1 - model
            judge = 1 - model if rng.random() < judge_noise else model
            output = f"{arm}-{item}"
            reviewer_vs_person.append((model, person, "owner", output))
            judge_vs_reviewer.append((judge, model, "reviewer", output))
            given[arm] = (model, person)
        leans.append((given["candidate"][0] - given["control"][0]) - (given["candidate"][1] - given["control"][1]))
    return reviewer_vs_person, leans, judge_vs_reviewer


def _t_rule_met(check: ReviewerCheck, pairs: list[tuple[int, int, str, str]], items: int) -> bool:
    """The rule ``met`` replaced: the lean's t interval inside ``±δ`` over 40 items or more, the kappa met."""
    read = instrument_agreement("pass_fail", pairs)
    kappa = calibration_criterion(read.n, read.results, read.agreement, read.interval).state
    interval = check.lean_interval
    return (
        check.result != "not_met"
        and items >= 40
        and kappa == "met"
        and interval is not None
        and -0.10 < interval[0]
        and interval[1] < 0.10
    )


def _rates(
    seed: str, items: int, *, lean_share: float, noise: float = 0.0, judge: bool = False, t_rule: bool = False
) -> dict[str, float]:
    rng = random.Random(seed)
    counts = {"leans": 0, "below_bar": 0, "met": 0, "judge_met": 0, "t_rule_met": 0}
    kappas: list[float] = []
    for _ in range(REPLICATES):
        pairs, leans, judged = _read(rng, items, lean_share=lean_share, noise=noise)
        check = reviewer_check("reviewer", "answer.good", "pass_fail", kappa_pairs=pairs, leans=leans)
        counts["leans"] += check.reason == "instrument_leans"
        counts["below_bar"] += check.reason == "instrument_below_bar"
        counts["met"] += check.result == "met"
        if t_rule:
            counts["t_rule_met"] += _t_rule_met(check, pairs, items)
        if check.kappa is not None:
            kappas.append(check.kappa)
        if judge:
            read = instrument_agreement("pass_fail", judged)
            counts["judge_met"] += (
                calibration_criterion(read.n, read.results, read.agreement, read.interval).state == "met"
            )
    rates = {name: count / REPLICATES for name, count in counts.items()}
    rates["kappa"] = sum(kappas) / len(kappas)
    return rates


@pytest.mark.parametrize(
    ("lean_share", "items", "kappa"),
    [(0.15, 40, 0.91), (0.15, 100, 0.91), (0.30, 40, 0.83), (0.30, 100, 0.83)],
)
def test_kappa_cannot_see_a_lean_the_judge_and_the_model_share(lean_share: float, items: int, kappa: float) -> None:
    rates = _rates(f"shared-{lean_share}-{items}", items, lean_share=lean_share, judge=True)
    assert abs(rates["kappa"] - kappa) <= 0.03, rates
    assert rates["judge_met"] >= 0.95, rates
    assert rates["below_bar"] <= 0.01, rates


@pytest.mark.parametrize(
    ("lean_share", "items", "caught"),
    [(0.30, 30, 0.75), (0.30, 60, 0.99), (0.15, 30, 0.23), (0.15, 60, 0.74), (0.15, 100, 0.97)],
)
def test_the_lean_is_caught_at_the_review_s_rates(lean_share: float, items: int, caught: float) -> None:
    rates = _rates(f"caught-{lean_share}-{items}", items, lean_share=lean_share)
    assert within(rates["leans"], caught, REPLICATES), rates
    if FAIL_RATE * lean_share > 0.10:
        # A lean past the bound is read as none at most at the equivalence reading's one-sided level.
        assert rates["met"] <= at_most(0.05, REPLICATES), rates


@pytest.mark.parametrize("noise", [0.05, 0.10])
@pytest.mark.parametrize("items", [30, 60, 100])
def test_a_model_that_does_not_lean_is_shown_to_lean_at_most_at_the_level(noise: float, items: int) -> None:
    rates = _rates(f"quiet-{noise}-{items}", items, lean_share=0.0, noise=noise)
    assert rates["leans"] <= at_most(0.05, REPLICATES), rates


@pytest.mark.parametrize("items", [40, 60, 105, 260])
def test_met_at_the_bound_is_claimed_at_most_at_the_test_s_level(items: int) -> None:
    # A lean of exactly 0.10, the bound (0.53 s = 0.10). Where the interval is about as wide as the bound (60 and 105
    # items) a rule reading the point estimate inside the bound would claim met in a large share of reads.
    rates = _rates(f"bound-{items}", items, lean_share=0.10 / FAIL_RATE, t_rule=True)
    assert rates["met"] <= at_most(0.05, REPLICATES), rates
    if items == 40:
        # The row fails for the rule it replaced: on these very reads the t interval inside the bound claims met
        # about 9% of the time (measured 9.35%), past the bar.
        assert rates["t_rule_met"] > at_most(0.05, REPLICATES), rates


def test_what_met_costs_in_pair_items() -> None:
    at_40 = _rates("cost-40", 40, lean_share=0.0, noise=0.05)
    at_105 = _rates("cost-105", 105, lean_share=0.0, noise=0.05)
    at_260 = _rates("cost-260", 260, lean_share=0.0, noise=0.05)
    assert at_40["met"] == 0.0, at_40
    assert at_105["met"] >= at_least(0.30, REPLICATES), at_105
    assert at_260["met"] >= at_least(0.90, REPLICATES), at_260
