"""Reading one run's stored results a page at a time, so no reader holds the whole run at once.

A stored result carries its judge's reasoning, its kind's payload, its world events, its deliveries and its rescores,
so a read of every result of a run grows with the run. A reader that walks a whole run (its summary, a take-up of it,
an export of it past the export ceiling) reads it here instead, one page at a time, and keeps of each result only what
it needs before the next page is read. Peak memory is then a page of results, whatever the run's size.

**A page is whole cases of the run.** The run's cases are taken in its own order (``EvalRun.test_case_ids``, each
once) and cut into runs of cases whose results a page holds, as many as :data:`RUN_RESULTS_PAGE` (or the reader's own
ceiling) allows at the rows a case is estimated to hold: the run's ``k``, or more when the run recorded persisting more
cells than its matrix. One case holding more rows than a page is a page of its own, so a page never splits a case.
Each page is one store read (:meth:`~threetears.evals.kernel.storage.EvalStorage.query_eval_results_of_cases`: the run's
results whose case is one of the page's), and its results are put in the page order: by the case's place in the run,
then ``k``, model, when the row was stored and id. That order is the same however the run is cut into pages, so a
reader of the pages gives one answer at any page size.

**What a page read does not reach.** A result whose case is none of its run's cases: the engine writes none (the runner
runs the run's own cases, and :func:`~threetears.evals.run.witnessed.record_witnessed_cell` refuses any other), and
there is no store read for "every case but these" to find one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final, Protocol

from threetears.evals.schema.models import EvalResult, EvalRun

#: The most results a reader of a whole run holds at once, as the run's matrix estimates them before any is read.
#:
#: A result is a few kilobytes to a few hundred (a judge's reasoning, a kind's payload, a world's events), so a page of
#: this many is megabytes at most, where a large run read whole was the run's whole volume.
RUN_RESULTS_PAGE: Final[int] = 200


class CaseResultsStore(Protocol):
    """The one read a paged walk of a run makes: the run's results whose case is one of those named."""

    def query_eval_results_of_cases(
        self, run_id: str, test_case_ids: Sequence[str], scope_id: str, /
    ) -> list[EvalResult]:
        """Every result of the run whose ``test_case_id`` is one of ``test_case_ids``, in one store read."""
        ...


def _rows_per_case(run: EvalRun, cases: int) -> int:
    """The rows one case of the run is estimated to hold: its ``k``, or its persisted cells spread over its cases when
    those are more (a run that ran past its matrix)."""
    persisted = run.completeness.persisted_cells if run.completeness is not None else 0
    spread = -(-persisted // cases) if cases else 0
    return max(run.k_runs, spread, 1)


def run_case_pages(run: EvalRun, *, page_results: int, after_case_id: str | None = None) -> list[tuple[str, ...]]:
    """The run's cases cut into pages, each the cases whose estimated results ``page_results`` holds (at least one).

    :param run: The run; its cases and ``k`` are read, never its results.
    :ptype run: EvalRun
    :param page_results: The most results a page is estimated to hold.
    :ptype page_results: int
    :param after_case_id: Start with the case after this one in the run's order; ``None`` starts at its first case.
    :ptype after_case_id: str | None
    :return: The pages, in the run's case order; empty when no case is left.
    :rtype: list[tuple[str, ...]]
    :raises ValueError: ``page_results`` is below 1, or ``after_case_id`` is not one of the run's cases.
    """
    if page_results < 1:
        raise ValueError(f"a page holds at least one result, not {page_results}")
    cases = list(dict.fromkeys(run.test_case_ids))
    per_page = max(1, page_results // _rows_per_case(run, len(cases)))
    if after_case_id is not None:
        if after_case_id not in cases:
            raise ValueError(f"case {after_case_id!r} is not one of run {run.id!r}'s cases")
        cases = cases[cases.index(after_case_id) + 1 :]
    return [tuple(cases[start : start + per_page]) for start in range(0, len(cases), per_page)]


def read_case_page(storage: CaseResultsStore, run: EvalRun, case_ids: Sequence[str]) -> list[EvalResult]:
    """One page of a run's results: those of ``case_ids``, in the page order (see the module docstring).

    :param storage: Where the run's results are read.
    :ptype storage: CaseResultsStore
    :param run: The run.
    :ptype run: EvalRun
    :param case_ids: The page's cases.
    :ptype case_ids: Sequence[str]
    :return: Their results, ordered.
    :rtype: list[EvalResult]
    """
    if not case_ids:
        return []
    place = {case_id: index for index, case_id in enumerate(dict.fromkeys(run.test_case_ids))}
    rows = storage.query_eval_results_of_cases(run.id, list(case_ids), run.scope_id)
    return sorted(
        rows,
        key=lambda row: (place.get(row.test_case_id, len(place)), row.k_iteration, row.model, row.scored_at, row.id),
    )


def iter_run_results(storage: CaseResultsStore, run: EvalRun, *, page_results: int) -> Iterator[EvalResult]:
    """Every result of the run whose case is one of its cases, in the page order, read one page at a time.

    A page is read when the one before it has been walked, so a reader that keeps nothing of a result past its turn
    holds one page at most.

    :param storage: Where the run's results are read.
    :ptype storage: CaseResultsStore
    :param run: The run.
    :ptype run: EvalRun
    :param page_results: The most results a page is estimated to hold (:data:`RUN_RESULTS_PAGE` unless the reader has
        a ceiling of its own).
    :ptype page_results: int
    :return: The results.
    :rtype: Iterator[EvalResult]
    :raises ValueError: ``page_results`` is below 1.
    """
    for case_ids in run_case_pages(run, page_results=page_results):
        yield from read_case_page(storage, run, case_ids)


__all__ = ["RUN_RESULTS_PAGE", "CaseResultsStore", "iter_run_results", "read_case_page", "run_case_pages"]
