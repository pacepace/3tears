"""The blind read's statistical claims, checked against a known truth with a fixed seed.

| Claim | Set-up | Error rate held |
|---|---|---|
| The exact sign test holds its level | No true preference; tie rate 0, 0.2, 0.5; 6 to 60 items; 4,000 reads a row | Falsely separated at most 5% (plus four Monte Carlo standard errors) |
| A rater who stops when it looks good gains nothing | No true preference; the rater closes the read after every item and stops at the first separation; 40 pair items, planned length 30; 1,000 reads | Falsely separated at most 5% (plus four standard errors), since the preference is read once, at the planned length |
| The rule that was refused | The sign test re-read after every item from 6 to 60 and stopped at the first separation; 4,000 reads | Falsely separated more than 10% (measured about 18%): recorded here, never shipped |
| A prefix is a random sample | 120 cases valued by their position in the store; prefixes of 10 to 40 over 600 seeds | The prefix mean is within four standard errors of the population's; every case is in the first 10 at the rate 10/120 within 4.5 standard errors. The first rows in store order miss the mean by 40 or more |
| A skip that follows content biases what is read | The same population; a rater who skips the upper half of the values | The answered prefix's mean misses the population's by more than 25: which is why a standing carries its skips |
| Slot balance | Every prefix of ``2m`` pair items, 200 seeds | Exactly ``m`` with the control in slot A |

Every draw is through :func:`~threetears.evals.analysis.blind_read.draw_blind_read` and every verdict through
:func:`~threetears.evals.analysis.blind_read.close_blind_read`, so what is held is what ships.
"""

from __future__ import annotations

import math
import random

import pytest

from threetears.evals.analysis.blind_read import (
    PREFERENCE_MIN_ITEMS,
    close_blind_read,
    draw_blind_read,
    exact_sign_test_p,
    sample_standing,
)
from threetears.evals.schema import (
    BlindAnswer,
    BlindArm,
    BlindRead,
    EvalResult,
    JudgeEvidence,
    RubricDim,
    RubricScore,
    label_key_of,
)
from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.simulation_support import at_most

DIM = RubricDim(name="answer.good", description="The answer is good.", scale="pass_fail")
ALPHA = 0.05


def _population(cases: int, *, arms: int) -> tuple[list[EvalResult], dict[str, JudgeEvidence], list[BlindArm]]:
    """``cases`` cases in store order, each arm's answer its own text."""
    declared = [BlindArm(key="control", role="control", run_ids=["run-a"])]
    if arms == 2:
        declared.append(BlindArm(key="candidate", role="candidate", run_ids=["run-b"]))
    results: list[EvalResult] = []
    evidence: dict[str, JudgeEvidence] = {}
    for case in range(cases):
        for arm in declared:
            read = JudgeEvidence(case_material=f"case {case}", artifact=f"{arm.key} answers {case}")
            key = label_key_of(read, DIM)
            result = make_eval_result(
                id=f"{arm.key}-{case}",
                eval_run_id=arm.run_ids[0],
                test_case_id=f"case-{case:04d}",
                rubric_scores=[
                    RubricScore(
                        dim=DIM.name,
                        score=1,
                        scale="pass_fail",
                        axis="capability",
                        served_model="sim/judge",
                        judge_temperature=0.0,
                        output_fingerprint=key.output_fingerprint,
                        criterion_fingerprint=key.criterion_fingerprint,
                    )
                ],
            )
            results.append(result)
            evidence[result.id] = read
    return results, evidence, declared


def _draw(cases: int, *, arms: int, seed: int, **kwargs: object) -> tuple[BlindRead, list[EvalResult]]:
    results, evidence, declared = _population(cases, arms=arms)
    read = draw_blind_read(
        results,
        evidence,
        [DIM],
        declared,
        scope_id="sim",
        created_by="sim",
        created_at="t0",
        seed=seed,
        **kwargs,  # type: ignore[arg-type]
    )
    return read, results


def _value(read: BlindRead, position: int) -> int:
    """A case's planted value: its position in the store."""
    return int(read.items[position].case_id[-4:])


@pytest.mark.parametrize("tie_rate", [0.0, 0.2, 0.5])
def test_the_exact_sign_test_separates_at_most_five_percent_of_reads_with_no_true_preference(tie_rate: float) -> None:
    rng = random.Random(f"blind-read-sign|{tie_rate}")
    replicates = 4_000
    for items in (6, 10, 15, 20, 30, 40, 60):
        separated = 0
        for _ in range(replicates):
            decided = sum(1 for _ in range(items) if rng.random() >= tie_rate)
            wins = sum(1 for _ in range(decided) if rng.random() < 0.5)
            p = exact_sign_test_p(wins, decided)
            separated += decided >= PREFERENCE_MIN_ITEMS and p is not None and p < ALPHA
        assert separated / replicates <= at_most(ALPHA, replicates), (items, separated / replicates)


def test_a_rater_who_stops_as_soon_as_the_read_separates_is_wrong_at_most_five_percent_of_the_time() -> None:
    """The rater has no preference, closes the read after every item, and stops the moment it says one arm is better.

    Read after every item, the sign test would be falsely separated about 18% of the time. The shipped read gives no
    verdict before its planned length and reads only the planned items, so every look but one says ``withheld``.
    """
    read, results = _draw(40, arms=2, seed=11, ask=("preference",))
    assert read.draw.planned_items == 30
    rng = random.Random("blind-read-stopping")
    replicates = 1_000
    fooled = 0
    seen_early = 0
    for _ in range(replicates):
        answers: list[BlindAnswer] = []
        for item in read.items:
            answers.append(BlindAnswer(item=item.item, rater="rater", answered_at="t1", prefer=rng.choice(["A", "B"])))
            report = close_blind_read(read.model_copy(update={"answers": answers}), results, []).read.report
            assert report is not None
            (preference,) = report.preferences
            if len(answers) < 30:
                seen_early += preference.verdict != "withheld"
            if preference.verdict in ("improved", "regressed"):
                fooled += 1
                break
    assert seen_early == 0, "no verdict, count or p before the planned length"
    assert fooled / replicates <= at_most(ALPHA, replicates), fooled / replicates


def test_the_refused_rule_a_sign_test_re_read_after_every_item_is_wrong_far_more_than_five_percent() -> None:
    """Recorded, not shipped: why the preference is read once."""
    rng = random.Random("blind-read-naive-looks")
    replicates = 4_000
    fooled = 0
    for _ in range(replicates):
        wins = 0
        for answered in range(1, 61):
            wins += rng.random() < 0.5
            p = exact_sign_test_p(wins, answered)
            if answered >= PREFERENCE_MIN_ITEMS and p is not None and p < ALPHA:
                fooled += 1
                break
    assert fooled / replicates > 0.10, fooled / replicates


def test_a_prefix_of_the_drawn_order_is_an_unbiased_sample_and_the_store_s_first_rows_are_not() -> None:
    cases, seeds = 120, 600
    results, evidence, declared = _population(cases, arms=1)
    population_mean = (cases - 1) / 2
    population_sd = math.sqrt((cases * cases - 1) / 12)
    sums = {m: 0.0 for m in (10, 20, 40)}
    in_first_ten = [0] * cases
    for seed in range(seeds):
        read = draw_blind_read(results, evidence, [DIM], declared, scope_id="sim", created_by="sim", seed=seed)
        values = [_value(read, position) for position in range(cases)]
        for m in sums:
            sums[m] += sum(values[:m]) / m
        for value in values[:10]:
            in_first_ten[value] += 1
    for m, total in sums.items():
        # The standard error of a mean of m drawn without replacement from the population, over the seeds.
        se = population_sd / math.sqrt(m) * math.sqrt((cases - m) / (cases - 1)) / math.sqrt(seeds)
        assert abs(total / seeds - population_mean) <= 4 * se, (m, total / seeds)
        # The first m rows in store order: the convenience sample the old sheet invited.
        assert abs(sum(range(m)) / m - population_mean) >= 40, m
    rate = 10 / cases
    sd = math.sqrt(seeds * rate * (1 - rate))
    assert max(abs(count - seeds * rate) for count in in_first_ten) <= 4.5 * sd, "every case is as likely to be early"


def test_a_skip_that_follows_content_biases_the_answered_prefix_which_the_standing_states() -> None:
    cases, seeds = 120, 200
    results, evidence, declared = _population(cases, arms=1)
    total = 0.0
    for seed in range(seeds):
        read = draw_blind_read(results, evidence, [DIM], declared, scope_id="sim", created_by="sim", seed=seed)
        prefix = [_value(read, position) for position in range(30)]
        answered = [value for value in prefix if value < cases // 2]  # the rater skips the upper half
        total += sum(answered) / len(answered)
        if seed == 0:
            answers = [
                BlindAnswer(item=read.items[position].item, rater="rater", skipped=_value(read, position) >= cases // 2)
                for position in range(30)
            ]
            standing = sample_standing(read.model_copy(update={"answers": answers}), "rater")
            assert standing.standing == "random" and standing.skipped == len(prefix) - len(answered) > 0
    assert (cases - 1) / 2 - total / seeds > 25


def test_every_prefix_of_two_m_pairs_holds_m_with_the_control_in_slot_a() -> None:
    results, evidence, declared = _population(30, arms=2)
    lean = 0
    for seed in range(200):
        read = draw_blind_read(results, evidence, [DIM], declared, scope_id="sim", created_by="sim", seed=seed)
        control_in_a = [item.outputs[0].arm == "control" for item in read.items]
        for m in range(1, 16):
            assert sum(control_in_a[: 2 * m]) == m, (seed, m)
        lean += control_in_a[0]
    # The first slot of a block is a fair coin across seeds: within four standard errors of a half.
    assert abs(lean / 200 - 0.5) <= 4 * math.sqrt(0.25 / 200)
