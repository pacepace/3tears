"""The work a report read does grows with its campaign no faster than the campaign does (W4b done-when #4).

The production-shaped campaign (:mod:`~packages.evals.tests.hot_path_support`) is read at ``scale`` 1, 2 and 4, a
code-only ``report_read`` in html with the trials table, and each operation it performs is counted, not timed
(:data:`~packages.evals.tests.hot_path_support.OPERATIONS`): ``ScoreRecord`` constructions, ``EvalResult``
validations, ``model_copy`` calls (on any model, and on an ``EvalResult``), bundle fingerprints, canonical-JSON calls,
measure descriptions and pydantic constructions of any model. A count is deterministic, so a doubling's ratio is too,
and it does not depend on the campaign's size: the campaign is read at a quarter of production's cases per run
(:data:`CASES_PER_RUN`, about 740, 1,480 and 2,950 results), so the test costs a quarter of the time. The oracle and
the profile read the production-sized campaign.

Each count may grow by at most :data:`PER_DOUBLING` per doubling of the campaign, both over the assembly (counted as
the campaign bundle is built) and over the whole read. On the base these tests were written against every count grows
about 2.0 per doubling, so none is marked as a known excess. A count found to grow faster is asserted in its own test
marked ``xfail(strict=True)`` naming its hotspot and the chunk that owns the file it lives in, so a fix flips it to
XPASS, which fails, and the fixer drops the mark.

**What it cannot see.** Only the cases per run double: the runs (41), the arms (4) and the folds stay fixed, so work
quadratic in runs or arms, or a per-run rescan, grows linearly here. A quadratic in results or cases gives about 4.0
per doubling and is caught. Result iterations per fold are not counted: that would need a hook inside the folds,
which this test may not add to the source. The counts above bound the work those walks do.

One test, so one worker builds and reads the three campaigns once.
"""

from __future__ import annotations

import pytest

from packages.evals.tests.hot_path_support import CASES_PER_RUN as PRODUCTION_CASES_PER_RUN
from packages.evals.tests.hot_path_support import OPERATIONS, growth, hot_path_corpus, report_read_counts

#: How much any count may grow when the campaign doubles.
PER_DOUBLING = 2.2

#: The scales read: each double the one before.
SCALES = (1, 2, 4)

#: The cases each run measures at ``scale=1``: a quarter of production's.
CASES_PER_RUN = PRODUCTION_CASES_PER_RUN // 4


def test_every_counted_operation_grows_no_faster_than_the_campaign() -> None:
    by_phase: dict[str, dict[int, dict[str, int]]] = {"assembly": {}, "report_read": {}}
    for scale in SCALES:
        read = report_read_counts(hot_path_corpus(scale=scale, cases_per_run=CASES_PER_RUN))
        for phase, counted in read.items():
            by_phase[phase][scale] = counted

    # The guard's own precondition: the scales really double the rows read, or no ratio below says anything.
    whole = by_phase["report_read"]
    assert whole[1]["score_records"] > 0, "the read must project scores for this to test anything"
    assert growth(whole, "result_validations") == pytest.approx([2.0, 2.0], abs=0.1), whole
    excess = {
        (phase, operation): [round(ratio, 3) for ratio in growth(measured, operation)]
        for phase, measured in by_phase.items()
        for operation in OPERATIONS
        if any(ratio > PER_DOUBLING for ratio in growth(measured, operation))
    }

    assert not excess, f"counts growing more than {PER_DOUBLING}x per doubling: {excess}; the counts: {by_phase}"
