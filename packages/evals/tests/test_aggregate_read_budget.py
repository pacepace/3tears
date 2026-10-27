"""A read-tier aggregate reads at most ``READ_TIER_ROW_BUDGET`` results: past it, it narrows or refuses, never pages.

The aggregates (``comparison_sets``, ``pivot``, ``frontier``, ``history``, ``program_budget``, ``orphaned_runs``) read a
whole scope's results into Python, and a partial aggregate is a wrong answer that looks complete, so a scope too large
is refused rather than paged (owner, 2026-10-27; the SQL pushdown is #996). Pinned here, through each lens over a store
that counts what it is asked for, with the budget patched to a few dozen results:

- **A scope estimated past the budget is refused before anything is read**, naming the estimate, the budget and the
  filters that lens takes.
- **A call narrowed to runs that fit is answered**, reading those runs alone: one subject's runs for a pivot, a
  frontier or a history (with the same cells, points and series the whole scope's read gives that subject, and the
  same status and archival counts), the grouped runs for comparison sets. A narrowed call still past it is refused.
- **A scope that fits is read whole, as before.**
- **The safety net**: a read that returns more than the budget (results whose run was deleted) is not aggregated.
- **The regression watch**, which reads a scope's history after each completed run, logs a refused history at ERROR
  and delivers nothing; its ``check`` raises the refusal.
- **A campaign report**, which reads its runs by id, is answered past the budget.

Mutations that turn this file red: ``_aggregate_results`` reading the whole scope before the estimate; its narrowed
read made whole-scope; ``>=`` for ``>`` against the budget; the safety nets dropped; ``_subject_runs`` keeping only the
cohort (the status and archival counts then differ); the watch's ``except`` removed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from threetears.evals.analysis import (
    campaign_report,
    comparison_sets,
    frontier,
    history,
    orphaned_runs,
    pivot,
    program_budget,
    reads,
)
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.judge_profiles import EvalJudgeProfile
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.ops import RegressionAlert, RegressionSink, RegressionWatch
from threetears.evals.run.reads import list_runs
from threetears.evals.schema.models import CalibrationRating, EvalResult, EvalRun, EvalTemplate
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_subject, memory_storage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture

PROFILE = toyhost_profile(every_seat=True)
SCOPE = "uni-aggregate-budget"
#: The budget every test reads under, patched in so a scope past it is a few dozen results.
BUDGET = 20
#: Cases per run (k = 1), so a run holds this many results.
CASES = 4
#: The subject a narrowed call asks for: two completed runs, one archived, one failed (16 results).
NARROW = "subject-a"
#: The other subject: three completed runs (12 results), so the scope holds 28.
OTHER = "subject-b"


# parity-with: threetears.evals.analysis.reads.LensStore
class FakeCountingStore:
    """The engine's store, counting each read a lens makes of results: the whole scope's, and each run's."""

    def __init__(self, inner: EvalStorage) -> None:
        self._inner = inner
        #: The rows each whole-scope read returned, one entry per read.
        self.scope_reads: list[int] = []
        #: The run each by-run read was of, in call order.
        self.run_reads: list[str] = []

    def query_eval_results(self, scope_id: str, /) -> list[EvalResult]:
        rows = self._inner.query_eval_results(scope_id)
        self.scope_reads.append(len(rows))
        return rows

    def query_eval_results_by_run(self, run_id: str, scope_id: str, /) -> list[EvalResult]:
        self.run_reads.append(run_id)
        return self._inner.query_eval_results_by_run(run_id, scope_id)

    def list_campaigns(self, scope_id: str, /) -> list[EvalCampaign]:
        return self._inner.list_campaigns(scope_id)

    def load_campaign(self, campaign_id: str, scope_id: str, /) -> EvalCampaign | None:
        return self._inner.load_campaign(campaign_id, scope_id)

    def query_calibration_ratings(
        self, scope_id: str, /, *, run_ids: Sequence[str] | None = None
    ) -> list[CalibrationRating]:
        return self._inner.query_calibration_ratings(scope_id, run_ids=run_ids)

    def query_judge_profiles(self, scope_id: str, /) -> list[EvalJudgeProfile]:
        return self._inner.query_judge_profiles(scope_id)

    def load_templates(self, template_ids: Sequence[str], scope_id: str, /) -> list[EvalTemplate]:
        return self._inner.load_templates(template_ids, scope_id)

    def load_eval_run(self, run_id: str, scope_id: str, /) -> EvalRun | None:
        return self._inner.load_eval_run(run_id, scope_id)

    def read_nothing(self) -> bool:
        """No result was read, whole-scope or by run."""
        return self.scope_reads == [] and self.run_reads == []


def _seed_run(
    storage: EvalStorage,
    index: int,
    subject: str,
    *,
    status: str = "completed",
    archived: bool = False,
    results: int = CASES,
) -> EvalRun:
    """A run of ``subject`` created ``index`` minutes into the day, its matrix :data:`CASES` × 1, with ``results``."""
    run = make_eval_run(
        scope_id=SCOPE,
        candidate_model=f"model-{index % 2}",
        subject_snapshot=make_subject(subject, subject.title()),
        status=status,
        archived=archived,
        test_case_ids=[f"tc-{case}" for case in range(CASES)],
        created_at=f"2026-10-27T00:{index:02d}:00+00:00",
    )
    storage.save_eval_run(run)
    for cell in range(results):
        storage.save_eval_result(
            make_eval_result(
                id=f"{run.id}-{cell}",
                eval_run_id=run.id,
                scope_id=SCOPE,
                model=run.candidate_model,
                subject_id=subject,
                test_case_id=f"tc-{cell % CASES}",
                cost_usd=0.01,
            )
        )
    return run


@pytest.fixture
def budget(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", BUDGET)
    return BUDGET


@pytest.fixture
def scope() -> tuple[EvalStorage, list[EvalRun]]:
    """28 results: :data:`NARROW`'s four runs (one archived, one failed) and :data:`OTHER`'s three; returns the first
    subject's runs."""
    storage, _ = memory_storage()
    narrow = [
        _seed_run(storage, 0, NARROW),
        _seed_run(storage, 1, NARROW),
        _seed_run(storage, 2, NARROW, archived=True),
        _seed_run(storage, 3, NARROW, status="failed"),
    ]
    for index in range(4, 7):
        _seed_run(storage, index, OTHER)
    return storage, narrow


def _lister(storage: EvalStorage) -> Callable[..., list[EvalRun]]:
    host = toyhost_host(storage=storage)
    return lambda scope_id, **kwargs: list_runs(host, scope_id, **kwargs)


#: Each aggregate as a call over a store, with the arguments given; ``status="all"`` lets the grouping read every run.
LENSES: dict[str, Callable[..., Any]] = {
    "comparison_sets": lambda store, storage, **kw: comparison_sets(
        store, SCOPE, list_runs=_lister(storage), profile=PROFILE, **({"status": "all"} | kw)
    ),
    "pivot": lambda store, storage, **kw: pivot(
        store,
        SCOPE,
        list_runs=_lister(storage),
        row_factor="model",
        column_factor="template_id",
        metric="cost_usd",
        profile=PROFILE,
        **kw,
    ),
    "frontier": lambda store, storage, **kw: frontier(store, SCOPE, list_runs=_lister(storage), profile=PROFILE, **kw),
    "history": lambda store, storage, **kw: history(
        store, SCOPE, list_runs=_lister(storage), metric="cost_usd", profile=PROFILE, **kw
    ),
    "program_budget": lambda store, storage, **kw: program_budget(store, SCOPE, list_runs=_lister(storage), **kw),
    "orphaned_runs": lambda store, storage, **kw: orphaned_runs(store, SCOPE, list_runs=_lister(storage), **kw),
}

#: What each lens's refusal tells its caller to narrow by: the filters it takes, or that it takes none.
NARROWED_BY = {
    "comparison_sets": ("campaign_id", "run_ids", "run status"),
    "pivot": ("subject_filter", "subject_id"),
    "frontier": ("subject_filter", "subject_id"),
    "history": ("subject_filter", "subject_id"),
    "program_budget": ("takes no filter", "scope_export"),
    "orphaned_runs": ("takes no filter", "runs_list"),
}


class TestAScopePastTheBudgetIsRefusedBeforeAnythingIsRead:
    @pytest.mark.parametrize("lens", sorted(LENSES))
    def test_the_refusal_names_the_estimate_the_budget_and_how_to_narrow(
        self, lens: str, budget: int, scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, _ = scope
        store = FakeCountingStore(storage)

        with pytest.raises(ValidationFailedError) as refused:
            LENSES[lens](store, storage)

        assert store.read_nothing(), "refused before any result was read"
        # The grouping reads its listed runs (the archived one is not listed); every other lens reads the scope.
        estimate = 6 * CASES if lens == "comparison_sets" else 7 * CASES
        assert refused.value.details["results_estimated"] == estimate
        assert refused.value.details["result_budget"] == budget
        assert f"about {estimate} results, past the {budget}" in refused.value.message
        assert "never answered in part" in refused.value.message
        assert f"about {estimate} results" in refused.value.remedy and f"hold {budget} or fewer" in refused.value.remedy
        for words in NARROWED_BY[lens]:
            assert words in refused.value.remedy, (lens, words)

    def test_a_refusal_is_logged_at_warning(
        self, budget: int, scope: tuple[EvalStorage, list[EvalRun]], caplog: pytest.LogCaptureFixture
    ) -> None:
        storage, _ = scope
        with caplog.at_level(logging.INFO, logger="threetears.evals"), pytest.raises(ValidationFailedError):
            LENSES["pivot"](FakeCountingStore(storage), storage)

        (line,) = [record for record in caplog.records if record.getMessage().startswith("eval.read_tier lens=pivot")]
        assert line.levelno == logging.WARNING
        assert f"refused runs=7 results_estimated=28 over READ_TIER_ROW_BUDGET={budget}" in line.getMessage()

    def test_a_narrowed_call_still_past_it_is_refused_reading_nothing(
        self, monkeypatch: pytest.MonkeyPatch, scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, _ = scope
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 3 * CASES)
        store = FakeCountingStore(storage)

        with pytest.raises(ValidationFailedError) as refused:
            LENSES["history"](store, storage, subject_id=NARROW)

        assert store.read_nothing()
        assert refused.value.details == {"runs": 4, "results_estimated": 4 * CASES, "result_budget": 3 * CASES}


class TestACallNarrowedToRunsThatFitIsAnswered:
    def test_a_pivot_of_one_subject_reads_its_runs_alone_and_gives_the_whole_reads_cells(
        self, monkeypatch: pytest.MonkeyPatch, scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, narrow = scope
        whole = LENSES["pivot"](storage, storage, subject_id=NARROW)
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", BUDGET)
        store = FakeCountingStore(storage)

        table = LENSES["pivot"](store, storage, subject_id=NARROW)

        assert store.scope_reads == [] and store.run_reads == [run.id for run in narrow]
        assert table.cells == whole.cells and table.cells
        # Its archived and failed runs were read too, so what the cohort left out is counted exactly.
        assert table.exclusions.results_from_archived_runs == whole.exclusions.results_from_archived_runs == CASES
        assert table.exclusions.results_outside_queried_runs == whole.exclusions.results_outside_queried_runs == CASES

    def test_a_history_of_one_subject_gives_the_whole_reads_series(
        self, monkeypatch: pytest.MonkeyPatch, scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, narrow = scope
        whole = LENSES["history"](storage, storage, subject_id=NARROW)
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", BUDGET)
        store = FakeCountingStore(storage)

        result = LENSES["history"](store, storage, subject_id=NARROW)

        assert store.scope_reads == [] and sorted(store.run_reads) == sorted(run.id for run in narrow)
        assert result.series == whole.series and result.series
        assert result.exclusions.results_from_archived_runs == whole.exclusions.results_from_archived_runs

    def test_a_frontier_of_one_subject_gives_the_whole_reads_points(
        self, monkeypatch: pytest.MonkeyPatch, scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, narrow = scope
        whole = LENSES["frontier"](storage, storage, subject_id=NARROW)
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", BUDGET)
        store = FakeCountingStore(storage)

        result = LENSES["frontier"](store, storage, subject_id=NARROW)

        assert store.scope_reads == [] and sorted(store.run_reads) == sorted(run.id for run in narrow)
        assert result["subjects"] == whole["subjects"] and result["subjects"]

    def test_a_grouping_narrowed_to_completed_runs_reads_those_alone(
        self, monkeypatch: pytest.MonkeyPatch, scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, _ = scope
        whole = LENSES["comparison_sets"](storage, storage, status="completed")
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", BUDGET)
        store = FakeCountingStore(storage)

        grouped = LENSES["comparison_sets"](store, storage, status="completed")

        completed = [run.id for run in storage.query_eval_runs(SCOPE) if run.status == "completed" and not run.archived]
        assert store.scope_reads == [] and sorted(store.run_reads) == sorted(completed)
        assert grouped == whole and grouped["comparison_sets"]

    @pytest.mark.parametrize("lens", sorted(LENSES))
    def test_a_scope_that_fits_is_read_whole_as_before(
        self, lens: str, monkeypatch: pytest.MonkeyPatch, scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, _ = scope
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 7 * CASES)
        store = FakeCountingStore(storage)

        LENSES[lens](store, storage)

        assert store.scope_reads == [7 * CASES] and store.run_reads == []


class TestTheSafetyNet:
    def _orphans(self, storage: EvalStorage, count: int) -> None:
        """Results of a run since deleted: in no listing, so no estimate sees them."""
        for index in range(count):
            storage.save_eval_result(
                make_eval_result(id=f"orphan-{index}", eval_run_id="deleted", scope_id=SCOPE, test_case_id="tc-0")
            )

    @pytest.mark.parametrize("lens", ["pivot", "program_budget"])
    def test_a_whole_read_past_the_budget_is_refused_when_the_call_does_not_narrow(
        self, lens: str, budget: int
    ) -> None:
        storage, _ = memory_storage()
        _seed_run(storage, 0, NARROW)
        self._orphans(storage, budget)
        store = FakeCountingStore(storage)

        with pytest.raises(ValidationFailedError) as refused:
            LENSES[lens](store, storage)

        assert store.scope_reads == [CASES + budget], "the estimate let the read through"
        assert refused.value.details["results_returned"] == CASES + budget
        assert f"the store returned {CASES + budget}" in refused.value.message
        assert f"returned {CASES + budget} results" in refused.value.remedy

    def test_a_whole_read_past_the_budget_falls_back_to_the_subjects_runs(self, budget: int) -> None:
        storage, _ = memory_storage()
        kept = _seed_run(storage, 0, NARROW)
        self._orphans(storage, budget)
        store = FakeCountingStore(storage)

        result = LENSES["history"](store, storage, subject_id=NARROW)

        assert store.scope_reads == [CASES + budget] and store.run_reads == [kept.id]
        assert result.series and result.exclusions.results_without_run == 0, "the deleted run's results went unread"

    def test_a_narrowed_read_past_the_budget_is_refused(self, budget: int) -> None:
        storage, _ = memory_storage()
        # Its matrix says four results, and it holds more than the budget (no completeness was recorded).
        swollen = _seed_run(storage, 0, NARROW, results=budget + 1)
        for index in range(1, 6):
            _seed_run(storage, index, OTHER)
        store = FakeCountingStore(storage)

        with pytest.raises(ValidationFailedError) as refused:
            LENSES["pivot"](store, storage, subject_id=NARROW)

        assert store.scope_reads == [] and store.run_reads == [swollen.id]
        assert refused.value.details == {
            "runs": 1,
            "results_estimated": CASES,
            "result_budget": budget,
            "results_returned": budget + 1,
        }


# =============================================================================
# The regression watch, and campaign reports
# =============================================================================


class _RecordingSink(RegressionSink):
    """Keeps every alert it is handed."""

    def __init__(self) -> None:
        self.alerts: list[RegressionAlert] = []

    async def __call__(self, alert: RegressionAlert) -> None:
        self.alerts.append(alert)


def _watched() -> tuple[RegressionWatch, _RecordingSink]:
    """Two completed runs of one contestant, the later one's cost up fivefold, and a watch on cost."""
    storage, _ = memory_storage()
    cases = [f"c{index}" for index in range(12)]
    for run_id, created, cost in (("run-a", "2026-10-01T00:00:00Z", 0.01), ("run-b", "2026-10-02T00:00:00Z", 0.05)):
        run = make_eval_run(id=run_id, scope_id=SCOPE, test_case_ids=cases, status="completed", created_at=created)
        storage.save_eval_run(run)
        for index, case in enumerate(cases):
            storage.save_eval_result(
                make_eval_result(
                    id=f"{run_id}-{case}",
                    eval_run_id=run_id,
                    scope_id=SCOPE,
                    test_case_id=case,
                    cost_usd=cost + index * 0.0001,
                )
            )
    sink = _RecordingSink()
    watch = RegressionWatch(
        host=toyhost_host(storage=storage), sink=sink, measures=("cost_usd",), min_absolute_change=0.01
    )
    return watch, sink


class TestTheRegressionWatch:
    def test_its_check_raises_a_refused_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        watch, _ = _watched()
        assert [alert.run_id for alert in watch.check("run-b", SCOPE)] == ["run-b"], "under the budget it fires"
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", BUDGET)

        with pytest.raises(ValidationFailedError, match="history would aggregate about 24 results"):
            watch.check("run-b", SCOPE)

    async def test_the_hook_logs_a_refused_history_at_error_and_delivers_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        watch, sink = off_loop(_watched)
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", BUDGET)

        with caplog.at_level(logging.INFO, logger="threetears.evals"):
            await watch("run-b", SCOPE, "completed")

        assert sink.alerts == []
        (line,) = [record for record in caplog.records if record.getMessage().startswith("eval.regression watch")]
        assert line.levelno == logging.ERROR
        message = line.getMessage()
        assert f"checked nothing on run=run-b scope_id={SCOPE}" in message
        assert "history would aggregate about 24 results" in message and "subject_filter" in message


def test_a_campaign_report_reads_its_runs_by_id_past_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = off_loop(ops_fixture)
    host = fixture.host.eval_host
    before = off_loop(campaign_report, host, fixture.campaign.id, TOYHOST_SCOPE)
    monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 1)

    after = off_loop(campaign_report, host, fixture.campaign.id, TOYHOST_SCOPE)

    assert off_loop(host.storage.query_eval_results, TOYHOST_SCOPE), "the scope holds results past the budget"
    assert after.basis == before.basis == "code_only"
    assert after.verdicts == before.verdicts
