"""A regression found the moment a run completes, delivered through the host's sink (#647).

Driven through a launching host's job manager — the real terminal transition — over the toy host's store: a
baseline run is stored, a second run's job writes its results and completes, and the watch the
:class:`~threetears.evals.run.launch.LaunchHost` was handed as ``on_run_end`` checks it against the history.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from threetears.evals.contracts import EvalStorage, RubricScore
from threetears.evals.contracts.models import TRANSCRIPT_DIM_ID, EvalResult, EvalRun
from threetears.evals.ops import RegressionAlert, RegressionSink, RegressionWatch
from threetears.evals.run import default_job_timeout
from threetears.evals.run.launch import LaunchHost
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_calibration_rating, make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS

_SCOPE = "uni-1"
_CASES = [f"c{i}" for i in range(12)]
_JUDGE = "judge-model"


class _RecordingSink(RegressionSink):
    """Keeps every alert it is handed."""

    def __init__(self) -> None:
        self.alerts: list[RegressionAlert] = []

    async def __call__(self, alert: RegressionAlert) -> None:
        self.alerts.append(alert)


class _BrokenSink(RegressionSink):
    """Raises on every alert, and counts how often it was asked."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, alert: RegressionAlert) -> None:
        self.calls += 1
        raise RuntimeError("the pager is down")


def _results(run: EvalRun, *, cost: float, transcript: int) -> list[EvalResult]:
    return [
        make_eval_result(
            id=f"{run.id}-{case}",
            eval_run_id=run.id,
            scope_id=_SCOPE,
            test_case_id=case,
            cost_usd=cost + index * 0.0001,
            transcript_score=RubricScore(dim=TRANSCRIPT_DIM_ID, score=transcript, scale="ordinal", served_model=_JUDGE),
        )
        for index, case in enumerate(_CASES)
    ]


def _launch_host(storage: EvalStorage, watch: RegressionWatch) -> LaunchHost:
    eval_host = toyhost_host(storage=storage)
    return LaunchHost(
        eval_host=eval_host,
        kinds={},
        settings=lambda: TOYHOST_LAUNCH_SETTINGS,
        job_timeout_factory=default_job_timeout,
        world_placements=lambda run: {},
        on_run_end=watch,
    )


async def _drive(
    sink: RegressionSink,
    *,
    measures: Sequence[str],
    calibrate: bool = False,
    declined_cost: float = 0.05,
) -> tuple[EvalStorage, EvalRun]:
    """Store a baseline run, then complete a second one through the launching host's job manager."""
    storage = EvalStorage(InMemoryDocumentStore())
    baseline = make_eval_run(
        id="run-a", scope_id=_SCOPE, test_case_ids=_CASES, status="completed", created_at="2026-10-01T00:00:00Z"
    )
    storage.save_eval_run(baseline)
    for result in _results(baseline, cost=0.01, transcript=5):
        storage.save_eval_result(result)
    later = make_eval_run(id="run-b", scope_id=_SCOPE, test_case_ids=_CASES, created_at="2026-10-02T00:00:00Z")
    declined = _results(later, cost=declined_cost, transcript=1)
    if calibrate:
        # A person rates every result of both runs exactly as the judge scored it: 24 results, perfect agreement.
        for result in [*_results(baseline, cost=0.01, transcript=5), *declined]:
            assert result.transcript_score is not None
            storage.save_calibration_rating(
                make_calibration_rating(
                    scope_id=_SCOPE,
                    run_id=result.eval_run_id,
                    result_id=result.id,
                    rubric_dim=TRANSCRIPT_DIM_ID,
                    score=result.transcript_score.score,
                )
            )

    eval_host = toyhost_host(storage=storage)
    watch = RegressionWatch(host=eval_host, sink=sink, measures=measures, min_absolute_change=0.01)
    launch_host = _launch_host(storage, watch)

    async def work(progress: object) -> None:
        for result in declined:
            storage.save_eval_result(result)

    job_ids = await launch_host.job_manager.start_group([(later, work, None)])
    await launch_host.job_manager.wait_for(job_ids)
    stored = storage.load_eval_run(later.id, _SCOPE)
    assert stored is not None
    return storage, stored


async def test_one_regression_is_delivered_and_an_uncalibrated_judged_one_is_withheld() -> None:
    sink = _RecordingSink()

    _, run = await _drive(sink, measures=("cost_usd", TRANSCRIPT_DIM_ID))

    assert run.status == "completed"
    assert len(sink.alerts) == 1, "cost regressed and was delivered; the transcript fell on an uncalibrated judge"
    alert = sink.alerts[0]
    assert (alert.measure, alert.run_id, alert.previous_run_id) == ("cost_usd", "run-b", "run-a")
    assert alert.regression.label == "regressed"
    assert alert.regression.test and alert.regression.min_absolute_change == 0.01
    assert alert.regression.delta is not None and alert.regression.delta > 0.01
    assert (alert.graded_by, alert.evidence_tier) == ("code", None)


async def test_a_judged_regression_fires_once_its_judge_is_calibrated() -> None:
    sink = _RecordingSink()

    await _drive(sink, measures=(TRANSCRIPT_DIM_ID,), calibrate=True)

    assert [(alert.measure, alert.graded_by, alert.evidence_tier) for alert in sink.alerts] == [
        (TRANSCRIPT_DIM_ID, "judge", "calibrated")
    ]


async def test_a_measure_that_did_not_move_past_the_threshold_delivers_nothing() -> None:
    sink = _RecordingSink()

    await _drive(sink, measures=("cost_usd",), declined_cost=0.012)

    assert sink.alerts == []


async def test_a_failing_sink_neither_changes_the_run_nor_stops_the_next_alert() -> None:
    sink = _BrokenSink()

    _, run = await _drive(sink, measures=("cost_usd", "total_ms", "cost_usd"))

    assert run.status == "completed"
    assert sink.calls == 2, "each cost alert was offered, the first failure notwithstanding"


def test_a_watch_refuses_a_measure_the_history_cannot_series_and_an_empty_one() -> None:
    host = toyhost_host()
    with pytest.raises(ValueError, match="cannot track 'score'"):
        RegressionWatch(host=host, sink=_RecordingSink(), measures=("score",))
    with pytest.raises(ValueError, match="at least one measure"):
        RegressionWatch(host=host, sink=_RecordingSink(), measures=())
