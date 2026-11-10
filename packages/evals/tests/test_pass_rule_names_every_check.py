"""A check a host declares fails trials, and every surface that states the pass rule names it.

A boolean host measure with a better end is a check: on its other value it fails the trial, whatever the goal checks
and the judged dimensions read. A reader who sees only "every goal check and criterion >= 3 of 5" reads a high pass
rate into an arm that failed nearly every trial on that check. Contracts held here
(``threetears.evals.analysis.report_input_build.report_input``, as ``analyst.snapshot.analyst_input`` builds an
analyst's input; ``analyst.facts.campaign_overview``; the ``aspects.<aspect>.pass_rule`` and ``.description`` keys;
the ``count.failures`` and ``scoreboard.<arm>.failed`` keys):

- **the rule names every check that is no goal check**, by the name the page gives its measure and the value it must
  hold to pass, in the rule's label (its key), the analyst's overview row and the rule's sentence alike, and a trial
  whose goal checks and judged dimensions all pass but whose check holds the other value is a failed trial;
- **count.failures counts every trial the pass rule failed**, a check-failed one included, so it reads the
  scoreboard's failed count for each arm on each aspect.

Every expectation is computed from the bundle or the census, never typed, but the check's name and value.
"""

from __future__ import annotations

from functools import cache

from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.fixtures.report_inputs.dataset import kairo_bundle
from threetears.evals.analysis.analyst.facts import campaign_overview
from threetears.evals.analysis.analyst.snapshot import analyst_input
from threetears.evals.analysis.bundle.schema import TrialCheck
from threetears.evals.analysis.bundle.trials import read_trial
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.metrics import MetricDescriptor

#: The host measure: true on a trial whose candidate never topped the customer up, which fails it.
CHECK = "top_up_never"

#: How every surface that states the rule names the check: its reader name and the value a trial must hold to pass.
NAMED = "Never topped up is false"

DESCRIPTOR = MetricDescriptor(
    name=CHECK,
    data_type="boolean",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="Whether the candidate never topped the customer up.",
    reader_name="Never topped up",
    higher_is_better=False,
)


@cache
def _bundle() -> CampaignBundle:
    """Kairo round 2 with the check declared on its aspect, and the first arm's passed trials but one failed by it
    alone: their goal checks and judged dimensions are as they were, and the census counts them as a check's."""
    bundle = kairo_bundle()
    aspect = bundle.aspects[0]
    first = aspect.scoreboard[0].variant_key
    flipped = [trial.trial_id for trial in aspect.trials if trial.arm == first and trial.state == "passed"][1:]
    assert flipped, "the control: the first arm keeps passed trials to fail"
    failed = TrialCheck(name=DESCRIPTOR.reader_name or CHECK, expression=f"{CHECK} is True")
    trials = [
        trial.model_copy(update={"state": "failed", "cause": "check", "failed_checks": [failed]})
        if trial.trial_id in flipped
        else trial
        for trial in aspect.trials
    ]
    scoreboard = [
        row.model_copy(
            update={
                "passed": row.passed - len(flipped),
                "failed": {**row.failed, "check": row.failed.get("check", 0) + len(flipped)},
            }
        )
        if row.variant_key == first
        else row
        for row in aspect.scoreboard
    ]
    changed = aspect.model_copy(
        update={
            "measure_catalog": {**aspect.measure_catalog, CHECK: DESCRIPTOR},
            "trials": trials,
            "scoreboard": scoreboard,
        }
    )
    return bundle.model_copy(update={"aspects": [changed]})


@cache
def _held() -> ReportInput:
    return analyst_input(_bundle())


def test_a_trial_whose_goal_checks_and_judge_pass_fails_on_the_check() -> None:
    """The census fails a trial on the check's other value alone, so the rule has to say the check is in it."""
    passing = make_eval_result(host_measures={CHECK: False})
    failing = make_eval_result(host_measures={CHECK: True})
    catalog = {CHECK: DESCRIPTOR}
    assert read_trial(passing, rubric_threshold=3, catalog=catalog).state == "passed", "the control"
    read = read_trial(failing, rubric_threshold=3, catalog=catalog)
    assert (read.state, read.cause) == ("failed", "check")
    assert [check.expression for check in read.failed_checks] == [f"{CHECK} is True"]


def test_the_rule_names_the_check_in_its_label_its_key_the_overview_and_its_sentence() -> None:
    held = _held()
    (aspect,) = held.aspects
    rule = aspect.pass_rule
    assert rule is not None
    assert CHECK in rule.checks and rule.named_checks == [NAMED]
    assert rule.label == f"pass^k (k=1, every goal check, {NAMED} and criterion >= 3 of 5)"
    keyed = f"aspects.{aspect.id}"
    assert resolve(f"{keyed}.pass_rule", held).value == rule.label
    (row,) = campaign_overview(held, "s").aspects
    assert row.checks == [NAMED]
    assert row.pass_rule is not None and NAMED in row.pass_rule.reads
    assert resolve(f"{keyed}.description", held).value == (
        f"A trial passes when its candidate did not fail, every goal check passed, {NAMED} and every judged dimension "
        "scored at least 3 of 5."
    )


def test_a_rule_held_to_the_check_alone_names_it_and_not_no_criterion() -> None:
    """An aspect whose only criterion is the host's check is held to it, never stated as held to nothing."""
    bundle = _bundle()
    (aspect,) = bundle.aspects
    alone = aspect.model_copy(
        update={
            "measure_catalog": {CHECK: DESCRIPTOR},
            "judged_measures": [],
        }
    )
    held = analyst_input(bundle.model_copy(update={"aspects": [alone]}))
    rule = held.aspects[0].pass_rule
    assert rule is not None and rule.criteria == []
    assert rule.label == f"pass^k (k=1, {NAMED})"
    assert held.aspects[0].description == f"A trial passes when its candidate did not fail and {NAMED}."
    assert held.aspects[0].graded_by == "checked"


def test_count_failures_reads_the_scoreboards_failed_trials() -> None:
    """The first arm's trials the check alone failed are failures: count.failures reads each arm's scoreboard failed
    count on the aspect, whatever the primary judged dimension read on them."""
    held = _held()
    (aspect,) = held.aspects
    first = next(iter(held.scoreboard))
    assert held.scoreboard[first].failed.get("check", 0) > 0, "the control: trials the check alone failed"
    for arm, row in held.scoreboard.items():
        board = resolve(f"scoreboard.{arm}.failed", held).value
        assert board == sum(row.failed.values())
        assert resolve(f"count.failures.{aspect.id}.{arm}", held).value == board
    assert resolve("count.failures", held).value == sum(sum(row.failed.values()) for row in held.scoreboard.values())
