"""Reading one run's stored results a page at a time, so no reader holds the whole run at once.

A stored result carries its judge's reasoning, its kind's payload, its world events, its deliveries and its rescores,
so a read of every result of a run grows with the run. A reader that walks a whole run (its summary, a take-up of it
and the offer of one, the launch pricer reading a template's history, an export of a run past the export ceiling, a
second judge's agreement) reads it here instead, one page at a time, and keeps of each result only what it needs
before the next page is read. Peak memory is then a page of results and one small key per result, whatever the run's
size.

**One pass over the run, then reads by id.** The run's results are found once, by one read of the run that leaves out
every field that grows with what a cell did and keeps each result's key (:class:`ResultKey`: its id, case, ``k``,
model and when it was stored, :meth:`~threetears.evals.kernel.storage.EvalStorage.query_eval_result_keys_by_run`). The
keys are put in the page order, and each page is then one read of its results by id
(:meth:`~threetears.evals.kernel.storage.EvalStorage.load_eval_results`), which every store answers from its key. So
a walk of a run of ``n`` results is one read of the run plus ``n / page`` reads by id: the run is passed over once,
however many pages it takes, where a read of the run per page would pass over it once a page.

**The page order** is by the case's place in the run (``EvalRun.test_case_ids``, each once), then ``k``, model, when
the row was stored and id. A result whose case is none of its run's cases (the engine writes none) comes after
every case of the run, by its case. The order is the same however the run is cut into pages, so a reader of the
pages gives one answer at any page size.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final, NamedTuple, Protocol

from threetears.evals.schema.models import EvalResult, EvalRun

#: The most results a reader of a whole run holds at once.
#:
#: A result is a few kilobytes to a few hundred (a judge's reasoning, a kind's payload, a world's events), so a page of
#: this many is megabytes at most, where a large run read whole was the run's whole volume.
RUN_RESULTS_PAGE: Final[int] = 200


class ResultKey(NamedTuple):
    """What a walk of a run reads of each result before it reads the result: where it sits in the page order."""

    id: str
    test_case_id: str
    k_iteration: int
    model: str
    scored_at: str


class RunResultsStore(Protocol):
    """The two reads a paged walk of a run makes: the run's result keys once, then a page of results by id."""

    def query_eval_result_keys_by_run(self, run_id: str, scope_id: str, /) -> list[ResultKey]:
        """The key of every result of the run, in one read that ships no other field."""
        ...

    def load_eval_results(self, result_ids: Sequence[str], scope_id: str, /) -> list[EvalResult]:
        """The named results, in one read by id."""
        ...


def run_result_keys(storage: RunResultsStore, run: EvalRun) -> list[ResultKey]:
    """Every result key of the run, in the page order (see the module docstring), from one read of the run.

    :param storage: Where the run's results are read.
    :ptype storage: RunResultsStore
    :param run: The run; its id, scope and cases are read.
    :ptype run: EvalRun
    :return: The keys, ordered.
    :rtype: list[ResultKey]
    """
    place = {case_id: index for index, case_id in enumerate(dict.fromkeys(run.test_case_ids))}
    beyond = len(place)
    keys = storage.query_eval_result_keys_by_run(run.id, run.scope_id)
    return sorted(
        keys,
        key=lambda key: (
            place.get(key.test_case_id, beyond),
            key.test_case_id if key.test_case_id not in place else "",
            key.k_iteration,
            key.model,
            key.scored_at,
            key.id,
        ),
    )


def run_case_pages(
    keys: Sequence[ResultKey], *, page_results: int, after_case_id: str | None = None
) -> list[list[ResultKey]]:
    """A run's ordered keys cut into pages of whole cases: as many cases as ``page_results`` results hold, and one
    case alone when it holds more.

    :param keys: The run's keys, in the page order (:func:`run_result_keys`).
    :ptype keys: Sequence[ResultKey]
    :param page_results: The most results a page holds, but for a case holding more on its own.
    :ptype page_results: int
    :param after_case_id: Start with the case after this one; ``None`` starts at the first case.
    :ptype after_case_id: str | None
    :return: The pages, each its keys in the page order; empty when no case is left.
    :rtype: list[list[ResultKey]]
    :raises ValueError: ``page_results`` is below 1, or ``after_case_id`` is not a case any key names.
    """
    if page_results < 1:
        raise ValueError(f"a page holds at least one result, not {page_results}")
    cases: dict[str, list[ResultKey]] = {}
    for key in keys:
        cases.setdefault(key.test_case_id, []).append(key)
    order = list(cases)
    if after_case_id is not None:
        if after_case_id not in cases:
            raise ValueError(f"case {after_case_id!r} holds none of the run's results")
        order = order[order.index(after_case_id) + 1 :]
    pages: list[list[ResultKey]] = []
    for case_id in order:
        held = cases[case_id]
        if pages and len(pages[-1]) + len(held) <= page_results:
            pages[-1].extend(held)
        else:
            pages.append(list(held))
    return pages


def read_results(storage: RunResultsStore, scope_id: str, keys: Sequence[ResultKey]) -> list[EvalResult]:
    """The results ``keys`` name, in one read by id, in the keys' order; one removed since its key was read is absent.

    :param storage: Where the results are read.
    :ptype storage: RunResultsStore
    :param scope_id: Their scope.
    :ptype scope_id: str
    :param keys: Their keys, in the order to return them.
    :ptype keys: Sequence[ResultKey]
    :return: The results.
    :rtype: list[EvalResult]
    """
    if not keys:
        return []
    place = {key.id: index for index, key in enumerate(keys)}
    return sorted(
        storage.load_eval_results([key.id for key in keys], scope_id), key=lambda row: place.get(row.id, len(place))
    )


def iter_run_results(storage: RunResultsStore, run: EvalRun, *, page_results: int) -> Iterator[EvalResult]:
    """Every result of the run, in the page order, ``page_results`` at a time.

    The run's keys are read once, when the walk starts; each page is read when the one before it has been walked, so
    a reader that keeps nothing of a result past its turn holds one page and the keys at most.

    :param storage: Where the run's results are read.
    :ptype storage: RunResultsStore
    :param run: The run.
    :ptype run: EvalRun
    :param page_results: The most results a page holds (:data:`RUN_RESULTS_PAGE` unless the reader has a ceiling of
        its own).
    :ptype page_results: int
    :return: The results.
    :rtype: Iterator[EvalResult]
    :raises ValueError: ``page_results`` is below 1.
    """
    if page_results < 1:
        raise ValueError(f"a page holds at least one result, not {page_results}")
    keys = run_result_keys(storage, run)
    for start in range(0, len(keys), page_results):
        yield from read_results(storage, run.scope_id, keys[start : start + page_results])


__all__ = [
    "RUN_RESULTS_PAGE",
    "ResultKey",
    "RunResultsStore",
    "iter_run_results",
    "read_results",
    "run_case_pages",
    "run_result_keys",
]
