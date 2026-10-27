"""A scope's export is bounded: a scope too large for one call is exported in pages of whole runs, never read whole.

An unnamed ``scope_export`` once read every result in its scope into memory and serialized them into one response,
and a large scope took its host down (Discodon, 2026-10-27). Pinned here, through
:func:`~threetears.evals.analysis.export_results` over a store that counts what it is asked for, and through the
action:

- **A scope past the ceiling is never read whole.** No whole-scope read is made, and no page reads more results than
  :data:`~threetears.evals.analysis.EXPORT_PAGE_RESULTS` (the ceiling itself, unpatched, in one test).
- **Pages cover every row exactly once**, following ``next_after_run_id`` from the first page to the last, with
  archived runs and runs of another status left out and counted, and a run added between calls joining the end.
- **A scope that fits is exported as before**: its body and exclusions are the whole-scope projection's, and it
  carries no page.
- **A listing that promised less than the store holds** (results whose run was deleted) is not serialized whole.
- **Two or more named runs past the ceiling are refused**, naming the ceiling, the results they would read and the
  remedy; one named run is exported whole however large.
- **The action teaches the next call**: its text names ``export_after_run_id`` and repeats a non-default format and
  status, and following it reads the next page.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``reads.export_results``: the whole-scope read taken without the estimate; the post-read count check dropped;
  ``_results_estimate`` returning ``len(test_case_ids)`` alone.
- ``reads._export_page``: ``>=`` for ``>`` against the ceiling; a page allowed to be empty when a run remains;
  ``next_after_run_id`` set on the last page; the order by id alone.
- ``reads._export_named``: the refusal at one run.
- ``render.render_export``: the format or status left out of the hint.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

import pytest

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis import EXPORT_PAGE_RESULTS, ScoreExport, export_results, reads
from threetears.evals.analysis.lenses.export import export_projection
from threetears.evals.analysis.reporting import ScoreProjection, project_score_records
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.judge_profiles import EvalJudgeProfile
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.reads import list_runs
from threetears.evals.schema.models import CalibrationRating, EvalResult, EvalRun, EvalTemplate, RunCompleteness
from packages.evals.tests.factories import make_eval_result, make_eval_run, memory_storage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture

PROFILE = toyhost_profile(every_seat=True)
SCOPE = "uni-export-paging"
#: The ceiling most tests export under, patched in so a scope past it is a few dozen results.
CEILING = 20
#: Cases per run, so three runs fit a page of :data:`CEILING` and a fourth does not.
CASES = 6


# parity-with: threetears.evals.analysis.reads.LensStore
class FakeCountingStore:
    """The engine's store, counting each read a lens makes of results: the whole scope's, and each run's."""

    def __init__(self, inner: EvalStorage) -> None:
        self._inner = inner
        #: The rows each whole-scope read returned, one entry per read.
        self.scope_reads: list[int] = []
        #: The rows each by-run read returned, in call order.
        self.run_reads: list[tuple[str, int]] = []

    def query_eval_results(self, scope_id: str, /) -> list[EvalResult]:
        rows = self._inner.query_eval_results(scope_id)
        self.scope_reads.append(len(rows))
        return rows

    def query_eval_results_by_run(self, run_id: str, scope_id: str, /) -> list[EvalResult]:
        rows = self._inner.query_eval_results_by_run(run_id, scope_id)
        self.run_reads.append((run_id, len(rows)))
        return rows

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

    def rows_read(self) -> int:
        """Every result row any read returned."""
        return sum(self.scope_reads) + sum(rows for _, rows in self.run_reads)


def _seed_run(
    storage: EvalStorage,
    index: int,
    *,
    cases: int = CASES,
    status: str = "completed",
    archived: bool = False,
    results: int | None = None,
    k_runs: int = 1,
    completeness: RunCompleteness | None = None,
    scope_id: str = SCOPE,
) -> EvalRun:
    """A run created ``index`` minutes into the day, its matrix ``cases`` × ``k_runs``, and that many results (or
    ``results``)."""
    run = make_eval_run(
        scope_id=scope_id,
        candidate_model=f"model-{index}",
        status=status,
        archived=archived,
        k_runs=k_runs,
        completeness=completeness,
        test_case_ids=[f"tc-{case}" for case in range(cases)],
        created_at=f"2026-10-27T{index // 60:02d}:{index % 60:02d}:00+00:00",
    )
    storage.save_eval_run(run)
    for cell in range(cases * k_runs if results is None else results):
        storage.save_eval_result(
            make_eval_result(
                id=f"{run.id}-{cell}",
                eval_run_id=run.id,
                scope_id=scope_id,
                model=run.candidate_model,
                test_case_id=f"tc-{cell % cases}",
                k_iteration=cell // cases + 1,
            )
        )
    return run


def _export(store: FakeCountingStore | EvalStorage, inner: EvalStorage, **arguments: Any) -> ScoreExport:
    host = toyhost_host(storage=inner)
    return export_results(
        store,
        SCOPE,
        list_runs=lambda scope, **kwargs: list_runs(host, scope, **kwargs),
        profile=PROFILE,
        **arguments,
    )


def _record_keys(export: ScoreExport) -> list[tuple[str, str, str | None]]:
    """Each row of a JSON export, by the result, measure and dimension it is."""
    projection = ScoreProjection.model_validate(json.loads(export.body))
    return [(record.result_id, record.metric, record.rubric_dim) for record in projection.records]


def _whole_projection(storage: EvalStorage, *, status: str | None = "completed") -> ScoreProjection:
    """The whole scope projected as an unbounded export projected it, independently of the lens."""
    every = list_runs(toyhost_host(storage=storage), SCOPE, include_archived=True)
    cohort = [run for run in every if not run.archived and (status is None or run.status == status)]
    return project_score_records(
        cohort,
        storage.query_eval_results(SCOPE),
        known_run_ids={run.id for run in every},
        archived_run_ids={run.id for run in every if run.archived},
        profile=PROFILE,
    )


@pytest.fixture
def ceiling(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(reads, "EXPORT_PAGE_RESULTS", CEILING)
    return CEILING


@pytest.fixture
def large_scope() -> tuple[EvalStorage, list[EvalRun]]:
    """Seven completed runs of six results (42, past a ceiling of 20), one archived, one failed, in creation order."""
    storage, _ = memory_storage()
    completed = [_seed_run(storage, index) for index in range(7)]
    _seed_run(storage, 7, archived=True)
    _seed_run(storage, 8, status="failed")
    return storage, completed


def _pages(
    store: FakeCountingStore, inner: EvalStorage, *, after_run_id: str | None = None, **arguments: Any
) -> list[ScoreExport]:
    """Every page of an export, following each page's next run from the first (or from ``after_run_id``)."""
    pages = [_export(store, inner, after_run_id=after_run_id, **arguments)]
    while (page := pages[-1].page) is not None and page.next_after_run_id is not None:
        assert len(pages) < 50, "the pages never ended"
        pages.append(_export(store, inner, after_run_id=page.next_after_run_id, **arguments))
    return pages


class TestAScopePastTheCeilingIsNeverReadWhole:
    def test_its_first_page_reads_its_runs_alone_and_names_the_next(
        self, ceiling: int, large_scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, completed = large_scope
        store = FakeCountingStore(storage)

        export = _export(store, storage)

        assert store.scope_reads == [], "no whole-scope read"
        assert store.rows_read() <= ceiling
        assert export.page is not None
        assert export.page.run_ids == [run.id for run in completed[:3]], "whole runs up to the ceiling, oldest first"
        assert [run_id for run_id, _ in store.run_reads] == export.page.run_ids
        assert (export.page.runs_before, export.page.runs_after) == (0, 4)
        assert export.page.next_after_run_id == completed[2].id
        assert export.page.result_ceiling == ceiling
        assert (export.page.archived_runs, export.page.runs_outside_status) == (1, 1)

    def test_at_the_real_ceiling_no_call_reads_past_it(self) -> None:
        storage, _ = memory_storage()
        per_run = EXPORT_PAGE_RESULTS // 4
        for index in range(6):
            _seed_run(storage, index, cases=per_run)
        store = FakeCountingStore(storage)

        pages = _pages(store, storage, fmt="json")

        assert store.scope_reads == []
        assert [len(page.page.run_ids) for page in pages if page.page] == [4, 2]
        assert store.rows_read() == 6 * per_run, "each result read once, across every page"
        assert all(page.n_records > 0 for page in pages)


class TestPagesCoverEveryRowExactlyOnce:
    def test_the_pages_rows_are_the_whole_scopes(
        self, ceiling: int, large_scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, completed = large_scope
        store = FakeCountingStore(storage)

        pages = _pages(store, storage, fmt="json")

        assert [page.page.run_ids for page in pages if page.page] == [
            [run.id for run in completed[:3]],
            [run.id for run in completed[3:6]],
            [completed[6].id],
        ]
        assert pages[-1].page is not None and pages[-1].page.next_after_run_id is None
        rows = Counter(key for page in pages for key in _record_keys(page))
        whole = Counter(
            (record.result_id, record.metric, record.rubric_dim) for record in _whole_projection(storage).records
        )
        assert rows == whole, "every row once, none twice, none from the archived or failed run"
        assert max(rows.values()) == 1

    def test_every_status_pages_the_failed_run_too(
        self, ceiling: int, large_scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, _ = large_scope
        pages = _pages(FakeCountingStore(storage), storage, fmt="json", status="all")

        rows = Counter(key for page in pages for key in _record_keys(page))
        assert rows == Counter(
            (record.result_id, record.metric, record.rubric_dim)
            for record in _whole_projection(storage, status=None).records
        )
        assert all(page.page is not None and page.page.runs_outside_status == 0 for page in pages)
        assert all(page.page is not None and page.page.run_status is None for page in pages)

    def test_a_run_finished_or_added_between_pages_is_neither_lost_nor_repeated(
        self, ceiling: int, large_scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, completed = large_scope
        store = FakeCountingStore(storage)
        first = _export(store, storage, fmt="json")
        assert first.page is not None and first.page.next_after_run_id is not None

        # Between the calls, an early run is archived (already exported) and a new run completes.
        storage.set_eval_run_archived(completed[0].id, SCOPE, archived=True)
        added = _seed_run(storage, 30)
        rest = _pages(store, storage, fmt="json", after_run_id=first.page.next_after_run_id)

        later = [run_id for page in rest if page.page for run_id in page.page.run_ids]
        assert later == [run.id for run in completed[3:]] + [added.id]

    def test_a_run_larger_than_the_ceiling_is_a_page_of_its_own(self, ceiling: int) -> None:
        storage, _ = memory_storage()
        small = _seed_run(storage, 0)
        big = _seed_run(storage, 1, cases=ceiling + 5)
        last = _seed_run(storage, 2)
        store = FakeCountingStore(storage)

        pages = _pages(store, storage)

        assert [page.page.run_ids for page in pages if page.page] == [[small.id], [big.id], [last.id]]
        assert store.rows_read() == 2 * CASES + ceiling + 5

    def test_runs_are_paged_in_the_order_they_were_created_not_by_id(self, ceiling: int) -> None:
        storage, _ = memory_storage()
        # Seeded newest first, so each later id (time-ordered as it is minted) is an earlier run.
        runs = [_seed_run(storage, index) for index in (6, 5, 4, 3, 2, 1, 0)]

        pages = _pages(FakeCountingStore(storage), storage)

        assert [run_id for page in pages if page.page for run_id in page.page.run_ids] == [run.id for run in runs[::-1]]

    def test_a_page_after_a_run_no_longer_in_the_scope_is_refused_teaching_a_restart(
        self, ceiling: int, large_scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, _ = large_scope
        with pytest.raises(ValidationFailedError) as refused:
            _export(FakeCountingStore(storage), storage, after_run_id="deleted-run")

        assert "'deleted-run' is not in this scope" in refused.value.message
        assert "first page" in refused.value.remedy
        assert refused.value.details == {"after_run_id": "deleted-run"}


class TestAScopeThatFitsIsExportedAsBefore:
    @pytest.mark.parametrize("fmt", ["csv", "json"])
    def test_its_body_and_exclusions_are_the_whole_projections(self, fmt: str) -> None:
        storage, _ = memory_storage()
        for index in range(3):
            _seed_run(storage, index)
        _seed_run(storage, 3, archived=True)
        _seed_run(storage, 4, status="failed")
        store = FakeCountingStore(storage)

        export = _export(store, storage, fmt=fmt)

        expected = export_projection(_whole_projection(storage), fmt=fmt)
        assert export.body == expected.body
        assert export.exclusions == expected.exclusions
        assert export.exclusions.results_from_archived_runs == CASES
        assert export.exclusions.results_outside_queried_runs == CASES
        assert export == expected and export.page is None
        assert store.scope_reads == [5 * CASES] and store.run_reads == []

    def test_a_listing_that_promised_less_than_the_store_holds_is_paged_not_serialized_whole(
        self, ceiling: int
    ) -> None:
        storage, _ = memory_storage()
        kept = _seed_run(storage, 0)
        # Results of a run since deleted: in no listing, so the estimate cannot see them.
        for index in range(ceiling):
            storage.save_eval_result(
                make_eval_result(id=f"orphan-{index}", eval_run_id="deleted", scope_id=SCOPE, test_case_id="tc-0")
            )
        store = FakeCountingStore(storage)

        export = _export(store, storage, fmt="json")

        assert store.scope_reads == [CASES + ceiling], "the estimate let the read through"
        assert export.page is not None and export.page.run_ids == [kept.id], "but not the serialization"
        assert {key[0] for key in _record_keys(export)} == {f"{kept.id}-{case}" for case in range(CASES)}


class TestNamedRunsPastTheCeilingAreRefused:
    def test_the_refusal_names_the_ceiling_the_results_and_the_remedy(
        self, ceiling: int, large_scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, completed = large_scope
        store = FakeCountingStore(storage)
        named = [run.id for run in completed[:4]]

        with pytest.raises(ValidationFailedError) as refused:
            _export(store, storage, run_ids=named)

        assert refused.value.message == (
            f"an export of 4 named runs would read about {4 * CASES} results, past the {ceiling} one export call reads"
        )
        assert "name fewer runs" in refused.value.remedy and "scope_export page by page" in refused.value.remedy
        assert refused.value.details == {
            "runs_named": 4,
            "results_estimated": 4 * CASES,
            "result_ceiling": ceiling,
        }
        assert store.rows_read() == 0, "refused before anything is read"

    def test_the_estimate_counts_every_repeat(self, ceiling: int) -> None:
        storage, _ = memory_storage()
        runs = [_seed_run(storage, index, k_runs=2) for index in range(2)]

        with pytest.raises(ValidationFailedError) as refused:
            _export(FakeCountingStore(storage), storage, run_ids=[run.id for run in runs])

        assert refused.value.details["results_estimated"] == 2 * CASES * 2

    def test_the_estimate_counts_what_a_run_recorded_persisting_past_its_matrix(self, ceiling: int) -> None:
        storage, _ = memory_storage()
        past = RunCompleteness(
            expected_cells=2, produced_cells=15, persisted_cells=15, infra_excluded_cells=0, counted_from="run_loop"
        )
        runs = [_seed_run(storage, index, cases=2, results=15, completeness=past) for index in range(2)]

        with pytest.raises(ValidationFailedError) as refused:
            _export(FakeCountingStore(storage), storage, run_ids=[run.id for run in runs])

        assert refused.value.details["results_estimated"] == 30

    def test_named_runs_within_it_are_exported_whole_and_unpaged(
        self, ceiling: int, large_scope: tuple[EvalStorage, list[EvalRun]]
    ) -> None:
        storage, completed = large_scope
        export = _export(FakeCountingStore(storage), storage, run_ids=[run.id for run in completed[:3]], fmt="json")

        assert export.page is None
        assert {key[0].rsplit("-", 1)[0] for key in _record_keys(export)} == {run.id for run in completed[:3]}

    def test_one_named_run_larger_than_the_ceiling_is_exported_whole(self, ceiling: int) -> None:
        storage, _ = memory_storage()
        big = _seed_run(storage, 0, cases=ceiling + 5)

        export = _export(FakeCountingStore(storage), storage, run_ids=[big.id], fmt="json")

        assert export.page is None
        assert len({key[0] for key in _record_keys(export)}) == ceiling + 5

    def test_named_runs_are_not_paged(self, large_scope: tuple[EvalStorage, list[EvalRun]]) -> None:
        storage, completed = large_scope
        with pytest.raises(ValidationFailedError) as refused:
            _export(FakeCountingStore(storage), storage, run_ids=[completed[0].id], after_run_id=completed[1].id)

        assert refused.value.message == "an export of named runs is not paged, so it starts after no run"


# =============================================================================
# The action
# =============================================================================


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


async def test_the_action_names_the_call_that_reads_the_next_page(
    evals: MountedTool, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    # Past whatever the toy corpus already holds, so the export pages; each added run is one page of its own.
    corpus = len(off_loop(storage.query_eval_results, TOYHOST_SCOPE))
    monkeypatch.setattr(reads, "EXPORT_PAGE_RESULTS", max(corpus, CASES))
    for index in range(3):
        off_loop(_seed_run, storage, 600 + index, status="failed", scope_id=TOYHOST_SCOPE, cases=max(corpus, CASES))

    arguments = {"action": "scope_export", "export_format": "json", "run_status": "failed"}
    first = await evals.call(arguments, host=fixture.host, caller=CALLER)

    assert not first.is_error, first.text
    page = ScoreExport.model_validate(first.structured).page
    assert page is not None and page.next_after_run_id is not None
    hint = (
        f"More: action='scope_export', export_after_run_id='{page.next_after_run_id}', export_format='json', "
        "run_status='failed'."
    )
    assert hint in first.text
    assert first.text.index("page: runs 1-1 of 3, oldest first") < first.text.index(hint)
    assert first.text.endswith(ScoreExport.model_validate(first.structured).body)

    second = await evals.call(
        {**arguments, "export_after_run_id": page.next_after_run_id}, host=fixture.host, caller=CALLER
    )
    following = ScoreExport.model_validate(second.structured).page
    assert following is not None and following.runs_before == 1 and following.run_ids != page.run_ids


async def test_the_actions_last_page_names_no_next_call(evals: MountedTool, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    corpus = len(off_loop(storage.query_eval_results, TOYHOST_SCOPE))
    monkeypatch.setattr(reads, "EXPORT_PAGE_RESULTS", corpus)
    off_loop(_seed_run, storage, 600, status="failed", scope_id=TOYHOST_SCOPE, cases=corpus)

    outcome = await evals.call({"action": "scope_export", "run_status": "failed"}, host=fixture.host, caller=CALLER)

    page = ScoreExport.model_validate(outcome.structured).page
    assert page is not None and page.next_after_run_id is None
    assert "More:" not in outcome.text
