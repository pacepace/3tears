"""Measure the bytes per result each budgeted reader holds at its peak, and the read budget that sets.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. Run from the repo root:

    uv run python packages/evals/scripts/measure_read_budgets.py            # every reader, every path (about 30 min)
    uv run python packages/evals/scripts/measure_read_budgets.py pivot      # the named readers only

It builds the hot-path corpus (``tests/hot_path_support.py``: 41 runs of production-shaped results) at 1x and 2x, adds
a regressed step so the regression watch reads its judged tiers, and reads each reader once under tracemalloc on every
path the package ships: the memory and SQLite stores, each as it is (``store``, ``sqlite``) and with its narrowed reads
hidden (``engine``, ``sqlite_engine``). A reader's bytes per result is the marginal of its peak, ``Δpeak / Δresults``
between the two scales, so what it holds whatever the scope's size cancels out. The table prints each reader's figure
per path, and the budget ``floor(allowance / bytes)`` at 256 MiB and at 1 GiB, then each reader family's budget.

**The rule the budgets follow** (owner ruling, 2026-10-30): a family's budget is set by the largest figure over its
readers on the paths a HOST runs, the SQLite store with and without its narrowed reads and the memory store without
them (``sqlite``, ``sqlite_engine``, ``engine``). The memory store's own narrowed read (``store``) is printed for
comparison and sets nothing: no host serves from it. ``threetears.evals.analysis.reads`` keeps each family's figure,
its path, its allowance and the date it was measured.
"""

from __future__ import annotations

import gc
import logging
import sys
import tempfile
import tracemalloc
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

# Run as a script, this file's directory is first on the path and the repo root is not; the corpus is test support.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from threetears.evals.analysis import comparison_sets, orphaned_runs, program_budget  # noqa: E402
from threetears.evals.analysis import service as analysis_service  # noqa: E402
from threetears.evals.analysis.reporting import METRIC_COMPOSITE  # noqa: E402
from threetears.evals.analysis.service import assemble_campaign_bundle  # noqa: E402
from threetears.evals.kernel.host import EvalHost  # noqa: E402
from threetears.evals.kernel.storage import EvalStorage  # noqa: E402
from threetears.evals.ops import report_read  # noqa: E402
from threetears.evals.ops.lenses import detectable_effects, scope_frontier, scope_history, scope_pivot  # noqa: E402
from threetears.evals.ops.regressions import RegressionAlert, RegressionWatch  # noqa: E402
from threetears.evals.run.reads import list_runs  # noqa: E402
from threetears.evals.schema.store_port import DocumentStore  # noqa: E402
from threetears.evals.storage import InMemoryDocumentStore, SqliteDocumentStore  # noqa: E402
from packages.evals.tests.hot_path_support import (  # noqa: E402
    ARM_MODELS,
    CASES_PER_RUN,
    HOT_PATH_SCOPE,
    K_RUNS,
    HotPathCorpus,
    hot_path_corpus,
)
from packages.evals.tests.read_path_support import EngineOnlyStore  # noqa: E402
from packages.evals.tests.test_result_reading_fields import (  # noqa: E402
    Campaign,
    add_a_regressed_step,
    regressed_steps,
)

#: The report's clock, pinned so every read is the same read.
REPORT_NOW = "2026-10-30T09:00:00+00:00"


async def _unused_sink(alert: RegressionAlert) -> None:
    raise AssertionError(f"check delivers nothing, yet {alert} was delivered")


def _readers(host: EvalHost, corpus: HotPathCorpus, cases: int, step: tuple[str, str]) -> dict[str, Callable[[], Any]]:
    """Each budgeted reader (and each spend lens, for comparison) as one call over ``host``."""

    def lister(scope_id: str, **kwargs: Any) -> Any:
        return list_runs(host, scope_id, **kwargs)

    scope, campaign_id, template_id = HOT_PATH_SCOPE, corpus.campaign.id, corpus.template_ids[0]
    measure, run_id = step
    return {
        "pivot": lambda: scope_pivot(
            host, scope, row_factor="model", column_factor="test_case_id", metric=METRIC_COMPOSITE
        ),
        "history": lambda: scope_history(host, scope, metric=METRIC_COMPOSITE),
        "frontier": lambda: scope_frontier(host, scope),
        "comparison_sets": lambda: comparison_sets(host.storage, scope, list_runs=lister, profile=host.profile),
        "regression_watch": lambda: RegressionWatch(host=host, sink=_unused_sink, measures=(measure,)).check(
            run_id, scope
        ),
        "detectable_effects": lambda: detectable_effects(
            host, scope, template_id, n_cases=cases, k_runs=K_RUNS, n_arms=len(ARM_MODELS)
        ),
        "bundle": lambda: assemble_campaign_bundle(host, campaign_id, scope),
        "report_read": lambda: report_read(host, campaign_id, scope, format="html", trials="all"),
        "program_budget": lambda: program_budget(host.storage, scope, list_runs=lister),
        "orphaned_runs": lambda: orphaned_runs(host.storage, scope, list_runs=lister),
    }


#: The paths a host runs, which set a budget; ``store`` (the memory store's own narrowed read) does not.
HOST_PATHS = ("engine", "sqlite", "sqlite_engine")

#: Each budget in ``threetears.evals.analysis.reads``: its readers here, and the memory one call may take.
FAMILIES: dict[str, tuple[tuple[str, ...], int]] = {
    "READING_BUDGET": (("history", "frontier", "comparison_sets", "regression_watch"), 256 << 20),
    "PLANNING_READ_BUDGET": (("detectable_effects",), 256 << 20),
    "PIVOT_READ_BUDGET": (("pivot",), 1 << 30),
    "BUNDLE_READ_BUDGET": (("bundle",), 1 << 30),
    "SPEND_READ_BUDGET": (("program_budget", "orphaned_runs"), 1 << 30),
}


def _peak(read: Callable[[], Any]) -> int:
    """The peak memory ``read`` allocates above what was held before it, in bytes."""
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        read()
        return tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


def _paths(store: InMemoryDocumentStore, directory: str) -> dict[str, DocumentStore]:
    """The corpus on every path the package ships."""
    sqlite = SqliteDocumentStore(Path(directory) / "corpus.db")
    for document in store.documents.values():
        sqlite.upsert(dict(document))
    return {
        "store": store,
        "engine": EngineOnlyStore(store),
        "sqlite": sqlite,
        "sqlite_engine": EngineOnlyStore(sqlite),
    }


def main(only: set[str]) -> None:
    logging.disable(logging.CRITICAL)
    analysis_service.utc_now_iso = lambda: REPORT_NOW  # type: ignore[assignment]
    peaks: dict[tuple[str, str, int], tuple[int, int]] = {}
    for scale in (1, 2):
        store = InMemoryDocumentStore()
        corpus = hot_path_corpus(scale=scale, store=store)
        add_a_regressed_step(corpus.host, HOT_PATH_SCOPE)
        results = len(store.by_doc_type("eval_result", HOT_PATH_SCOPE))
        cases = CASES_PER_RUN * scale
        campaign = Campaign(
            corpus.host, corpus.campaign.id, HOT_PATH_SCOPE, corpus.template_ids[0], cases, K_RUNS, len(ARM_MODELS)
        )
        step = regressed_steps(campaign)[0]
        with tempfile.TemporaryDirectory() as directory:
            for path, read_through in _paths(store, directory).items():
                host = replace(corpus.host, storage=EvalStorage(read_through))
                for reader, read in _readers(host, corpus, cases, step).items():
                    if not only or reader in only:
                        peak = _peak(read)
                        peaks[(reader, path, scale)] = (results, peak)
                        print(f"{scale}x {path:13s} {reader:18s} {results:6d} results {peak:>12d} B")
    print(f"\n{'reader':18s} {'path':13s} {'bytes/result':>12s} {'256 MiB':>8s} {'1 GiB':>8s}")
    marginals: dict[tuple[str, str], float] = {}
    for (reader, path, scale), (n1, p1) in sorted(peaks.items()):
        if scale == 1:
            n2, p2 = peaks[(reader, path, 2)]
            marginal = (p2 - p1) / (n2 - n1)
            marginals[(reader, path)] = marginal
            budgets = [str(int(allowance // marginal)) if marginal > 0 else "-" for allowance in (256 << 20, 1 << 30)]
            print(f"{reader:18s} {path:13s} {marginal:12.0f} {budgets[0]:>8s} {budgets[1]:>8s}")
    print(f"\n{'budget':22s} {'set by':32s} {'bytes/result':>12s} {'results':>8s}")
    for family, (readers, allowance) in FAMILIES.items():
        measured = [(marginals[(r, path)], r, path) for r in readers for path in HOST_PATHS if (r, path) in marginals]
        if measured:
            worst, reader, path = max(measured)
            print(f"{family:22s} {reader + ' on ' + path:32s} {worst:12.0f} {int(allowance // worst):8d}")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
