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
- **A narrowed answer says what it did not count** (``narrowed_read_disclosure``, absent from a whole read and from a
  bundle's fingerprint while unset): other subjects' results, and, on a judged frontier, the pooling of each judge's
  tier, which a narrowed read takes on one subject's results and so can read another tier.
- **A scope that fits is read whole, as before.**
- **The safety net**: a read that returns more than the budget (results whose run was deleted) is not aggregated.
- **The regression watch** reads the history of the completed run's own subject, so a scope past the budget is still
  checked where that subject fits, a judged alert saying where its tier was read; when the subject alone is past it,
  the hook logs at ERROR in its own words (nothing to narrow, #996) and delivers nothing, and ``check`` raises.
- **A campaign report**, which reads its runs by id, is answered past the budget.

Mutations that turn this file red: ``_aggregate_results`` reading the whole scope before the estimate; its narrowed
read made whole-scope; ``>=`` for ``>`` against the budget; the safety nets dropped; ``_subject_runs`` keeping only the
cohort (the status and archival counts then differ); the watch's ``except`` removed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from threetears.evals.analysis import (
    MIN_CASES,
    campaign_report,
    frontier_text,
    comparison_sets,
    frontier,
    history,
    orphaned_runs,
    pivot,
    program_budget,
    reads,
)
from threetears.evals.analysis.lenses.frontier import FrontierResult
from threetears.evals.kernel import ResultKey, ValidationFailedError
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.judge_profiles import EvalJudgeProfile
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.quick import EXIT_REFUSED, run_cli
from threetears.evals.ops import RegressionAlert, RegressionSink, RegressionWatch, history_text, pivot_text
from threetears.evals.run.reads import list_runs
from threetears.evals.kernel.evidence_tiers import CALIBRATION_MIN_RESULTS
from threetears.evals.schema import GoalStateOutcome, RoleUsage, RubricScore
from threetears.evals.schema.models import TRANSCRIPT_DIM_ID, CalibrationRating, EvalResult, EvalRun, EvalTemplate
from packages.evals.tests.factories import (
    make_calibration_rating,
    make_eval_result,
    make_eval_run,
    make_subject,
    memory_storage,
)
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.run_pages_support import BatchRecordingStore

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
#: How a narrowed answer's disclosure starts, at :data:`BUDGET`.
NARROWED_SAID = f"Read for subject {NARROW!r} alone: the scope is past the read tier's budget of {BUDGET} results"


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

    def query_eval_result_keys_by_run(self, run_id: str, scope_id: str, /) -> list[ResultKey]:
        self.run_reads.append(run_id)
        return self._inner.query_eval_result_keys_by_run(run_id, scope_id)

    def load_eval_results(self, result_ids: Sequence[str], scope_id: str, /) -> list[EvalResult]:
        return self._inner.load_eval_results(result_ids, scope_id)

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
        # Other subjects' rows were never read, so they are not counted, and the table says so where it is read.
        assert whole.n_filtered_out > 0 and table.n_filtered_out == 0
        assert whole.narrowed_read_disclosure is None
        assert table.narrowed_read_disclosure is not None
        assert table.narrowed_read_disclosure.startswith(NARROWED_SAID)
        assert "other subjects' results, and results whose run is missing, are not counted" in (
            table.narrowed_read_disclosure
        )
        assert "evidence tier" not in table.narrowed_read_disclosure
        assert table.narrowed_read_disclosure in pivot_text(table)
        assert "narrowed_read_disclosure" not in whole.model_dump()

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
        assert whole.n_filtered_out > 0 and result.n_filtered_out == 0
        assert whole.narrowed_read_disclosure is None
        assert result.narrowed_read_disclosure is not None and result.narrowed_read_disclosure.startswith(NARROWED_SAID)
        assert result.narrowed_read_disclosure in history_text(result)

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
        assert whole["n_filtered_out"] > 0 and result["n_filtered_out"] == 0
        assert "narrowed_read_disclosure" not in whole
        assert result["narrowed_read_disclosure"].startswith(NARROWED_SAID)
        assert result["narrowed_read_disclosure"] in frontier_text(FrontierResult.model_validate(result))

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
# A judged frontier read for one subject, and what an unset disclosure leaves of a fingerprint
# =============================================================================

_DIM = "reply.quality"
_JUDGE = "judge/model"
#: Judged cases per subject: half the ratings :data:`CALIBRATION_MIN_RESULTS` asks, so only the two pooled calibrate.
_JUDGED_CASES = CALIBRATION_MIN_RESULTS // 2


def _judged_scope() -> EvalStorage:
    """Two subjects of one run each, every result passing and scored by one judge that people rated in agreement
    with it: :data:`_JUDGED_CASES` ratings per subject, :data:`CALIBRATION_MIN_RESULTS` across both."""
    storage, _ = memory_storage()
    for index, subject in enumerate((NARROW, OTHER)):
        run = make_eval_run(
            id=f"judged-{subject}",
            scope_id=SCOPE,
            candidate_model="a",
            subject_snapshot=make_subject(subject, subject.title()),
            status="completed",
            test_case_ids=[f"tc{case}" for case in range(_JUDGED_CASES)],
            created_at=f"2026-10-27T00:0{index}:00+00:00",
        )
        storage.save_eval_run(run)
        for case in range(_JUDGED_CASES):
            score = 4 + case % 2
            result = make_eval_result(
                id=f"{run.id}-{case}",
                eval_run_id=run.id,
                scope_id=SCOPE,
                model="a",
                subject_id=subject,
                variant_key=f"vk-{subject}",
                test_case_id=f"tc{case}",
                rubric_scores=[
                    RubricScore(
                        dim=_DIM,
                        score=score,
                        scale="ordinal",
                        served_model=_JUDGE,
                        judge_temperature=0.0,
                        axis="capability",
                    )
                ],
                goal_state_outcomes=[GoalStateOutcome(expression="ok", passed=True)],
                usage=[RoleUsage(role="candidate", cost_usd=0.01)],
            )
            storage.save_eval_result(result)
            storage.save_calibration_rating(
                make_calibration_rating(
                    scope_id=SCOPE, run_id=run.id, result_id=result.id, rubric_dim=_DIM, score=score
                )
            )
    return storage


def _verdict_tier(answer: dict[str, Any]) -> str:
    (subject,) = answer["subjects"]
    assert subject["verdict"] is not None
    return str(subject["verdict"]["judge_evidence_tier"])


def test_a_judged_frontier_read_for_one_subject_tiers_its_judge_on_that_subject_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _judged_scope()

    def ranked() -> dict[str, Any]:
        return frontier(storage, SCOPE, list_runs=_lister(storage), bar=0.5, subject_id=NARROW)

    whole = ranked()
    monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", _JUDGED_CASES + MIN_CASES)
    narrowed = ranked()

    # Read whole, the judge's agreement pools both subjects' ratings and calibrates; read for one, half as many
    # leave it undetermined. Kept, and said.
    assert (_verdict_tier(whole), _verdict_tier(narrowed)) == ("calibrated", "undetermined")
    assert "narrowed_read_disclosure" not in whole
    said = narrowed["narrowed_read_disclosure"]
    assert said.startswith(f"Read for subject {NARROW!r} alone")
    assert "each judge's evidence tier was read on this subject's results alone" in said
    assert said in frontier_text(FrontierResult.model_validate(narrowed))


def test_an_unset_disclosure_leaves_a_bundles_fingerprint_where_it_was() -> None:
    bundle = toyhost_bundle()
    assert "narrowed_read_disclosure" not in json.dumps(bundle.to_dict())
    said = bundle.model_copy(
        update={"frontier": bundle.frontier.model_copy(update={"narrowed_read_disclosure": "read for one subject"})}
    )

    assert said.fingerprint() != bundle.fingerprint(), "set, it is evidence"
    assert bundle.model_copy(deep=True).fingerprint() == bundle.fingerprint()
    assert "narrowed_read_disclosure" not in json.dumps(
        said.model_copy(
            update={"frontier": said.frontier.model_copy(update={"narrowed_read_disclosure": None})}
        ).to_dict()
    )


# =============================================================================
# The regression watch, and campaign reports
# =============================================================================


class _RecordingSink(RegressionSink):
    """Keeps every alert it is handed."""

    def __init__(self) -> None:
        self.alerts: list[RegressionAlert] = []

    async def __call__(self, alert: RegressionAlert) -> None:
        self.alerts.append(alert)


_WATCH_CASES = [f"c{index}" for index in range(12)]


def _watched(*, others: int = 0, store: InMemoryDocumentStore | None = None) -> tuple[RegressionWatch, _RecordingSink]:
    """Two completed runs of one contestant of :data:`NARROW`, the later one's cost up fivefold and its judged
    transcript score down from 5 to 1, every judged score rated in agreement by a person (24 ratings, so the judge
    calibrates on this subject alone), beside ``others`` runs of :data:`OTHER`; and a watch on cost and the
    transcript; over ``store`` when one is given."""
    storage = memory_storage()[0] if store is None else EvalStorage(store)
    plan = [("run-a", NARROW, 0.01, 5), ("run-b", NARROW, 0.05, 1)]
    plan += [(f"other-{index}", OTHER, 0.01, 5) for index in range(others)]
    for order, (run_id, subject, cost, transcript) in enumerate(plan):
        run = make_eval_run(
            id=run_id,
            scope_id=SCOPE,
            subject_snapshot=make_subject(subject, subject.title()),
            test_case_ids=_WATCH_CASES,
            status="completed",
            created_at=f"2026-10-{order + 1:02d}T00:00:00Z",
        )
        storage.save_eval_run(run)
        for index, case in enumerate(_WATCH_CASES):
            result = make_eval_result(
                id=f"{run_id}-{case}",
                eval_run_id=run_id,
                scope_id=SCOPE,
                subject_id=subject,
                test_case_id=case,
                cost_usd=cost + index * 0.0001,
                transcript_score=RubricScore(
                    dim=TRANSCRIPT_DIM_ID,
                    score=transcript,
                    scale="ordinal",
                    served_model=_JUDGE,
                    axis="capability",
                    judge_temperature=0.0,
                ),
            )
            storage.save_eval_result(result)
            if subject == NARROW:
                storage.save_calibration_rating(
                    make_calibration_rating(
                        scope_id=SCOPE,
                        run_id=run_id,
                        result_id=result.id,
                        rubric_dim=TRANSCRIPT_DIM_ID,
                        score=transcript,
                    )
                )
    sink = _RecordingSink()
    watch = RegressionWatch(
        host=toyhost_host(storage=storage),
        sink=sink,
        measures=("cost_usd", TRANSCRIPT_DIM_ID),
        min_absolute_change=0.01,
    )
    return watch, sink


class TestTheRegressionWatch:
    def test_a_scope_past_the_budget_is_still_checked_on_the_runs_own_subject(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        watch, _ = _watched(others=3)
        whole = watch.check("run-b", SCOPE)
        # The scope holds 60 results, its subject 24.
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 2 * len(_WATCH_CASES))

        narrowed = watch.check("run-b", SCOPE)

        assert [(alert.measure, alert.regression) for alert in narrowed] == [
            (alert.measure, alert.regression) for alert in whole
        ]
        assert [alert.measure for alert in narrowed] == ["cost_usd", TRANSCRIPT_DIM_ID]
        assert all(alert.narrowed_read_disclosure is None for alert in whole)
        cost, transcript = narrowed
        assert cost.narrowed_read_disclosure is None, "code graded it: no judge was tiered"
        assert transcript.evidence_tier == "calibrated"
        assert transcript.narrowed_read_disclosure is not None
        assert transcript.narrowed_read_disclosure.startswith(f"Read for subject {NARROW!r} alone")
        assert "each judge's evidence tier was read on this subject's results alone" in (
            transcript.narrowed_read_disclosure
        )

    def test_its_check_raises_when_the_runs_subject_alone_is_past_the_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        watch, _ = _watched()
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", BUDGET)

        with pytest.raises(ValidationFailedError, match="history would aggregate about 24 results"):
            watch.check("run-b", SCOPE)

    async def test_the_hook_logs_it_at_error_in_its_own_words_and_delivers_nothing(
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
        assert f"holds about 24 results, past READ_TIER_ROW_BUDGET={BUDGET}" in message
        assert "nothing an operator can narrow" in message and "#996" in message
        assert "subject_filter" not in message and "subject_id" not in message


class TestTheRegressionWatchsJudgedTiersAreHeldToTheBudget:
    """A judged step's evidence tiers read the subject's results whole, so they are held to the budget as its history
    is: the subject's two runs hold 24 results by their matrices, beside three runs of another subject."""

    def test_one_result_past_the_budget_is_refused_reading_no_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        store = BatchRecordingStore()
        watch, _ = _watched(others=3, store=store)
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 2 * len(_WATCH_CASES) - 1)
        store.clear()

        # The history it reads first refuses on the same runs and estimate; the tiers' own hold is the same bound.
        with pytest.raises(ValidationFailedError) as refused:
            watch.check("run-b", SCOPE)

        assert refused.value.details == {"runs": 2, "results_estimated": 24, "result_budget": 23}
        assert store.typed_reads == [] and store.id_reads == [], "refused before any result was read"

    def test_at_the_budget_the_judged_step_is_tiered_on_the_subjects_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = BatchRecordingStore()
        watch, _ = _watched(others=3, store=store)
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 2 * len(_WATCH_CASES))
        store.clear()

        (_, transcript) = watch.check("run-b", SCOPE)

        assert transcript.evidence_tier == "calibrated" and transcript.narrowed_read_disclosure is not None
        subject_reads = [rows for named, _, rows in store.typed_reads if set(named) == {"run-a", "run-b"}]
        assert subject_reads and set(subject_reads) == {24}, "the history and the tiers read the subject's 24"


class TestTheFrontierCommand:
    """The quick CLI's ``frontier`` names its own flag in a refusal past the budget, never the action's filter."""

    def _refused(self, storage: EvalStorage, capsys: pytest.CaptureFixture[str], *subject: str) -> str:
        host = toyhost_host(storage=storage)
        code = run_cli(["frontier", "--scope", SCOPE, *subject], host_factory=lambda: host)
        stderr = capsys.readouterr().err
        assert code == EXIT_REFUSED, stderr
        assert "subject_filter" not in stderr and "subject_id" not in stderr
        return stderr

    def test_a_scope_past_the_budget_is_told_to_name_a_subject(
        self, budget: int, scope: tuple[EvalStorage, list[EvalRun]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        stderr = self._refused(scope[0], capsys)

        assert f"frontier would aggregate about {7 * CASES} results, past the {budget}" in stderr
        assert "filter it to one subject with --subject" in stderr
        assert f"answered when they hold {budget} results or fewer" in stderr

    def test_a_subject_past_the_budget_is_told_nothing_narrower_exists(
        self,
        monkeypatch: pytest.MonkeyPatch,
        scope: tuple[EvalStorage, list[EvalRun]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", 3 * CASES)
        stderr = self._refused(scope[0], capsys, "--subject", NARROW)

        assert (
            f"this subject alone holds about {4 * CASES} results, past the budget of {3 * CASES}, and the frontier "
            "command has no narrower filter"
        ) in stderr
        assert "--subject" not in stderr


def test_a_campaign_report_reads_its_runs_by_id_past_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scope past the budget never refuses a campaign whose own runs fit it: the bundle is held to its own runs'
    estimate (``test_bundle_read_budget``), never the scope's."""
    fixture = off_loop(ops_fixture)
    host = fixture.host.eval_host
    before = off_loop(campaign_report, host, fixture.campaign.id, TOYHOST_SCOPE)
    members = off_loop(host.storage.load_eval_runs, fixture.campaign.run_ids, TOYHOST_SCOPE)
    # The campaign's own runs, by their matrices (cases × k), are the budget; one run outside it puts the scope past.
    campaign_estimate = sum(len(run.test_case_ids) * run.k_runs for run in members)
    outside = make_eval_run(
        scope_id=TOYHOST_SCOPE, status="completed", k_runs=1, test_case_ids=[f"tc-{n}" for n in range(3)]
    )
    off_loop(host.storage.save_eval_run, outside)
    for n in range(campaign_estimate + 1):
        off_loop(
            host.storage.save_eval_result,
            make_eval_result(id=f"outside-{n}", scope_id=TOYHOST_SCOPE, eval_run_id=outside.id, test_case_id="tc-0"),
        )
    monkeypatch.setattr(reads, "READ_TIER_ROW_BUDGET", campaign_estimate)

    after = off_loop(campaign_report, host, fixture.campaign.id, TOYHOST_SCOPE)

    scope_results = off_loop(host.storage.query_eval_results, TOYHOST_SCOPE)
    assert len(scope_results) > campaign_estimate, "the scope holds results past the budget"
    assert after.basis == before.basis == "code_only"
    assert after.verdicts == before.verdicts
