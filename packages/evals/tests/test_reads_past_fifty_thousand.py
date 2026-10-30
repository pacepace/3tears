"""A scope of 60,000 results: the readings readers and the spend lenses answer it, the bundle and the pivot refuse it.

Each reader's read budget is its measured bytes per result against its allowance (owner, 2026-10-30): the readings
readers (``history``, ``frontier``, ``comparison_sets``, the regression watch's history and judged tiers) hold
``READING_BUDGET``, about 66,000 results, under 256 MiB; the spend lenses (``program_budget``, ``orphaned_runs``) hold
none; the planning read (``detectable_effects``) holds ``PLANNING_READ_BUDGET``; the pivot and a campaign's bundle hold
``PIVOT_READ_BUDGET`` and ``BUNDLE_READ_BUDGET``, about 27,000 and 28,000, under 1 GiB.

Held here over one scope of 30 runs of 1,000 cases × 2 (60,000 results by their matrices and in the store), built once
per module from cheap rows, read through the reference store as it is and behind
:class:`~packages.evals.tests.read_path_support.EngineOnlyStore` (the engine's path):

- **answered:** every readings reader and both spend lenses, the same answer on both paths, tracemalloc's peak over
  the call under 256 MiB;
- **refused before any result is read**, naming its own figure: the pivot, the bundle, and the planning read, whose
  budget is below 60,000.

The scope costs about ten seconds to build. The answered reads run under tracemalloc, which multiplies their time:
the module took about four minutes on the pool.
"""

from __future__ import annotations

import functools
import json
import logging
import tracemalloc
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pytest

from threetears.evals.analysis import comparison_sets, frontier, history, orphaned_runs, pivot, program_budget, reads
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.ops.lenses import detectable_effects
from threetears.evals.ops.regressions import RegressionAlert, RegressionWatch
from threetears.evals.run.reads import list_runs
from threetears.evals.schema import RubricScore
from threetears.evals.schema.models import TRANSCRIPT_DIM_ID, EvalRun
from threetears.evals.schema.store_port import DocumentStore
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_aspect_campaign, make_eval_result, make_eval_run, make_subject
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.read_path_support import EngineOnlyStore

PROFILE = toyhost_profile(every_seat=True)
#: The factories' scope, which a campaign built by them lives in.
SCOPE = "uni-1"
TEMPLATE = "tpl-past-fifty-thousand"
SUBJECT = "subject-a"
RUNS = 30
CASES = 1_000
K = 2
#: The results the scope holds, by its runs' matrices and in the store.
RESULTS = RUNS * CASES * K
#: What one call of a readings reader may take: 256 MiB.
READINGS_ALLOWANCE = 256 << 20


class _CountsResultReads(InMemoryDocumentStore):
    """The reference store, counting every read of results it serves."""

    def __init__(self) -> None:
        super().__init__()
        self.result_reads: Counter[str] = Counter()

    def by_doc_type(
        self, doc_type: str, scope_id: str, *, field_in: Mapping[str, Sequence[Any]] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        if doc_type == "eval_result":
            self.result_reads["by_doc_type"] += 1
        return super().by_doc_type(doc_type, scope_id, field_in=field_in, **kwargs)

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        if doc_type == "eval_result":
            self.result_reads["get_many"] += 1
        return super().get_many(doc_type, doc_ids, scope_id, **kwargs)

    def iter_narrowed(self, doc_type: str, scope_id: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        if doc_type == "eval_result":
            self.result_reads["iter_narrowed"] += 1
        return super().iter_narrowed(doc_type, scope_id, **kwargs)

    def grouped(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        if doc_type == "eval_result":
            self.result_reads["grouped"] += 1
        return super().grouped(doc_type, scope_id, **kwargs)


@functools.cache
def _scope() -> tuple[_CountsResultReads, list[EvalRun]]:
    """The 60,000-result scope, built once: each run's first cell saved through the engine, the rest copied from its
    stored row with their case and repeat. The last run's judge scored it lowest, so its step regresses on a judged
    measure and the watch reads its judged tiers."""
    store = _CountsResultReads()
    storage = EvalStorage(store)
    cases = [f"c{case:04d}" for case in range(CASES)]
    runs = []
    for index in range(RUNS):
        last = index == RUNS - 1
        run = make_eval_run(
            id=f"run-{index:02d}",
            scope_id=SCOPE,
            template_id=TEMPLATE,
            candidate_model="model-a",
            subject_snapshot=make_subject(SUBJECT, "Subject A"),
            status="completed",
            k_runs=K,
            test_case_ids=cases,
            created_at=f"2026-10-{index // 24 + 1:02d}T{index % 24:02d}:00:00+00:00",
        )
        storage.save_eval_run(run)
        runs.append(run)
        first = make_eval_result(
            id=f"{run.id}-{cases[0]}-1",
            eval_run_id=run.id,
            scope_id=SCOPE,
            model=run.candidate_model,
            subject_id=SUBJECT,
            test_case_id=cases[0],
            k_iteration=1,
            cost_usd=0.01,
            transcript_score=RubricScore(
                dim=TRANSCRIPT_DIM_ID,
                score=1 if last else 5,
                scale="ordinal",
                served_model="judge/model",
                axis="capability",
                judge_temperature=0.0,
            ),
        )
        storage.save_eval_result(first)
        row = store.documents[(SCOPE, first.id)]
        for case in cases:
            for k in range(1, K + 1):
                if (case, k) != (cases[0], 1):
                    store.upsert(row | {"id": f"{run.id}-{case}-{k}", "test_case_id": case, "k_iteration": k})
    store.result_reads.clear()
    return store, runs


def _host(store: DocumentStore) -> EvalHost:
    return toyhost_host(storage=EvalStorage(store))


def _paths() -> dict[str, tuple[_CountsResultReads, EvalHost]]:
    """The scope's store as it is, and behind :class:`EngineOnlyStore`, each with a host reading it."""
    store, _ = _scope()
    return {"store": (store, _host(store)), "engine": (store, _host(EngineOnlyStore(store)))}


def _lister(host: EvalHost) -> Callable[..., list[EvalRun]]:
    return lambda scope_id, **kwargs: list_runs(host, scope_id, **kwargs)


async def _unused_sink(alert: RegressionAlert) -> None:
    raise AssertionError(f"check delivers nothing, yet {alert} was delivered")


def _watch(host: EvalHost) -> list[dict[str, Any]]:
    watch = RegressionWatch(host=host, sink=_unused_sink, measures=(TRANSCRIPT_DIM_ID,), min_absolute_change=0.01)
    return [alert.model_dump(mode="json") for alert in watch.check(f"run-{RUNS - 1:02d}", SCOPE)]


#: Each reader that answers the scope, as a call over a host, its answer as JSON-safe data.
ANSWERED: dict[str, Callable[[EvalHost], Any]] = {
    "history": lambda host: history(
        host.storage, SCOPE, list_runs=_lister(host), metric="cost_usd", profile=PROFILE
    ).model_dump(mode="json"),
    "frontier": lambda host: frontier(host.storage, SCOPE, list_runs=_lister(host), profile=PROFILE),
    "comparison_sets": lambda host: comparison_sets(host.storage, SCOPE, list_runs=_lister(host), profile=PROFILE),
    "regression_watch": _watch,
    "program_budget": lambda host: program_budget(host.storage, SCOPE, list_runs=_lister(host)),
    "orphaned_runs": lambda host: orphaned_runs(host.storage, SCOPE, list_runs=_lister(host)),
}


def _peak[T](read: Callable[[], T]) -> tuple[T, int]:
    """``read``'s answer, and the peak memory it allocated above what was held before it, in bytes."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        answer = read()
        return answer, tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


def test_the_scope_is_past_fifty_thousand_and_inside_the_readings_budget() -> None:
    store, runs = _scope()

    assert len(store.by_doc_type("eval_result", SCOPE)) == RESULTS == sum(len(r.test_case_ids) * r.k_runs for r in runs)
    assert 50_000 < RESULTS <= reads.READING_BUDGET
    assert max(reads.PIVOT_READ_BUDGET, reads.BUNDLE_READ_BUDGET, reads.PLANNING_READ_BUDGET) < RESULTS


class TestTheReadingsReadersAndTheSpendLensesAnswerIt:
    @pytest.mark.parametrize("reader", sorted(ANSWERED))
    def test_alike_on_both_paths_under_the_allowance(self, reader: str, caplog: pytest.LogCaptureFixture) -> None:
        answers = {}
        for path, (_, host) in _paths().items():
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="threetears.evals"):
                answers[path], peak = _peak(lambda: ANSWERED[reader](host))
            assert peak < READINGS_ALLOWANCE, (reader, path, peak)
            if reader == "regression_watch":
                # The last run's judged step regressed, so its judges were tiered over the scope's readings, and an
                # uncalibrated judge's alert withheld.
                assert answers[path] == []
                assert any(
                    record.getMessage().startswith(f"eval.regression withheld run=run-{RUNS - 1:02d}")
                    for record in caplog.records
                ), path

        assert json.dumps(answers["engine"], sort_keys=True) == json.dumps(answers["store"], sort_keys=True)


def _refusal(read: Callable[[], object], store: _CountsResultReads) -> ValidationFailedError:
    store.result_reads.clear()
    with pytest.raises(ValidationFailedError) as refused:
        read()
    assert store.result_reads == Counter(), "refused before any result was read"
    return refused.value


#: Each reader that refuses the scope, and the budget it is refused past.
REFUSED: dict[str, tuple[Callable[[EvalHost, list[EvalRun]], object], str]] = {
    "pivot": (
        lambda host, _: pivot(
            host.storage,
            SCOPE,
            list_runs=_lister(host),
            row_factor="model",
            column_factor="test_case_id",
            metric="cost_usd",
            profile=PROFILE,
        ),
        "PIVOT_READ_BUDGET",
    ),
    "assemble_context_bundle": (
        lambda host, runs: assemble_context_bundle(
            make_aspect_campaign(run_ids=[run.id for run in runs]), storage=host.storage, profile=PROFILE
        ),
        "BUNDLE_READ_BUDGET",
    ),
    "detectable_effects": (
        lambda host, _: detectable_effects(host, SCOPE, TEMPLATE, n_cases=CASES, k_runs=K, n_arms=2),
        "PLANNING_READ_BUDGET",
    ),
}


class TestThePivotTheBundleAndThePlanningReadRefuseIt:
    @pytest.mark.parametrize("reader", sorted(REFUSED))
    def test_with_their_own_figure_reading_no_result(self, reader: str) -> None:
        call, family = REFUSED[reader]
        budget = getattr(reads, family)
        _, runs = _scope()

        for path, (store, host) in _paths().items():
            refused = _refusal(lambda: call(host, runs), store)

            assert refused.details == {"runs": RUNS, "results_estimated": RESULTS, "result_budget": budget}, path
            assert f"{reader} would aggregate about {RESULTS} results, past the {budget}" in refused.message
            assert refused.remedy.endswith(f"a call whose runs hold {budget} or fewer is answered")
            assert "#" not in refused.remedy, "a host reads plain words, not an issue number"
