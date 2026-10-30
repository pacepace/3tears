"""The work a report read does grows with its campaign no faster than the campaign does (W4b done-when #4).

The production-shaped campaign (:mod:`~packages.evals.tests.hot_path_support`) is read once at ``scale`` 1, 2 and 4
(about 3,100, 6,200 and 12,500 results: more cases per run, never more fields per result), a code-only
``report_read`` in html with the trials table, and each operation it performs is counted, not timed
(:data:`~packages.evals.tests.hot_path_support.OPERATIONS`): ``ScoreRecord`` constructions, ``EvalResult``
validations, ``model_copy`` calls (on any model, and on an ``EvalResult``), bundle fingerprints, canonical-JSON calls,
measure descriptions and pydantic constructions of any model. A count is deterministic, so a doubling's ratio is too.

Each count may grow by at most :data:`PER_DOUBLING` per doubling of the campaign, both over the assembly (counted as
the campaign bundle is built) and over the whole read. On the base these tests were written against every count grows
about 2.0 per doubling, so none is marked as a known excess. A count found to grow faster is asserted in its own test
marked ``xfail(strict=True)`` naming its hotspot and the chunk that owns the file it lives in, so a fix flips it to
XPASS, which fails, and the fixer drops the mark.

**Not counted: result iterations per fold.** Counting how many times each fold walks the results would need a hook
inside the folds, which this test may not add to the source. The counts above bound the work those walks do.

Slow by nature (three campaigns built and read, the largest about 12,500 results), so it reads each scale once per
process.
"""

from __future__ import annotations

import functools

import pytest

from packages.evals.tests.hot_path_support import OPERATIONS, growth, hot_path_corpus, report_read_counts

#: How much any count may grow when the campaign doubles.
PER_DOUBLING = 2.2

#: The scales read: each double the one before.
SCALES = (1, 2, 4)


@functools.cache
def counts() -> dict[str, dict[int, dict[str, int]]]:
    """Each phase's counts at each scale: ``{phase: {scale: {operation: count}}}``, read once per process."""
    by_phase: dict[str, dict[int, dict[str, int]]] = {"assembly": {}, "report_read": {}}
    for scale in SCALES:
        read = report_read_counts(hot_path_corpus(scale=scale))
        for phase, counted in read.items():
            by_phase[phase][scale] = counted
    return by_phase


@pytest.mark.parametrize("phase", ["assembly", "report_read"])
def test_every_counted_operation_grows_no_faster_than_the_campaign(phase: str) -> None:
    measured = counts()[phase]
    assert measured[1]["score_records"] > 0, "the read must project scores for this to test anything"

    excess = {
        operation: [round(ratio, 3) for ratio in growth(measured, operation)]
        for operation in OPERATIONS
        if any(ratio > PER_DOUBLING for ratio in growth(measured, operation))
    }

    assert not excess, (
        f"{phase}: counts growing more than {PER_DOUBLING}x per doubling: {excess}; "
        f"the counts by scale: { {scale: measured[scale] for scale in SCALES} }"
    )


def test_the_results_read_double_with_the_campaign() -> None:
    """The guard's own precondition: the scales really double the rows read, or no ratio above says anything."""
    measured = counts()["report_read"]

    assert [round(ratio, 2) for ratio in growth(measured, "result_validations")] == pytest.approx([2.0, 2.0], abs=0.03)
