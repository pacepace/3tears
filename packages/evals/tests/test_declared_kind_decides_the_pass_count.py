"""Whether an aspect's first line states a pass count is read off each reading's DECLARED kind, never the values it took.

A reading is a check where its declaration says it can fail a trial (a goal check, a classifier's accuracy, a host
measure declared pass/fail or a boolean with a better end, a judged dimension that counts), and a number otherwise. A
declared reading whose kind the evidence does not state (a host measure no trial scored, which the catalog does not
describe) is read as no check, and the line says so: it is never guessed to be one, which once printed "passed 20 of 20
cases" over nothing that could fail. A check that scored on some cases and not others (the runner raised on the rest)
is held by ``test_quick_bool_grader_is_a_check``. Each test below is built as a host's aspect is: stored runs, assembled into the
aspect's bundle, its verdict line read as its page and the quick print read it.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis.report.aspect.contrasts import aspect_verdict
from threetears.evals.kernel.metrics import goal_check_measure
from packages.evals.tests.report_shapes_support import ArmTrials, AspectTrials, shape_of


def _first_line(measure_order: Sequence[str], *, checks: bool = False) -> str:
    """One arm's verdict line on a host aspect whose design lists ``measure_order``, its trials graded on the two goal
    checks where ``checks`` and on nothing but what every result carries otherwise."""
    aspect = AspectTrials(
        label="extract", arms=[ArmTrials(label="v1", passing=12)], checks=checks, measure_order=measure_order
    )
    shape = shape_of("declared", [aspect], control=False)
    return aspect_verdict(
        shape.bundles["extract"], measure_order=measure_order, arm_labels=shape.readings.arm_labels()
    ).text


def test_a_host_aspect_with_only_numeric_measures_states_no_pass_count() -> None:
    line = _first_line([])

    assert line == "v1 ran 20 of 20 cases; no grader here says pass or fail."


def test_a_declared_number_no_trial_scored_is_no_check_and_the_line_says_so() -> None:
    line = _first_line(["fields_stripped"])

    # Never "v1 passed 20 of 20 cases": the declared reading has no stated kind here, so it is no check, said so.
    assert line == (
        "v1 ran 20 of 20 cases; no grader here says pass or fail; fields_stripped scored on no trial, so it is read as "
        "no check."
    )


def test_a_host_aspect_declaring_a_goal_check_states_its_pass_count() -> None:
    line = _first_line([goal_check_measure("reply.total == case.total")], checks=True)

    assert line.startswith("v1 passed 12 of 20 cases")
