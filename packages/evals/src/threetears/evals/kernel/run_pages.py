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

**Or in the store's order.** A reader that folds a run into figures a list of results gives (a comparison of two runs,
a count of what a lost run delivered, a calibration) walks it in the order the store keeps it
(:func:`iter_stored_results`), the order a read of the whole run hands its results out in, so its figures are the
ones that read gave, to the last bit of every float sum. One that keeps every result past its page keeps it without
its verbatim fields (:func:`results_without_payloads`).

**The page order** is by the case's place in the run (``EvalRun.test_case_ids``, each once), then ``k``, model, when
the row was stored and id. A result whose case is none of its run's cases (the engine writes none) comes after
every case of the run, by its case. The order is the same however the run is cut into pages, so a reader of the
pages gives one answer at any page size.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final, NamedTuple, Protocol

from threetears.evals.kernel.errors import StorageError
from threetears.evals.schema.models import EvalResult, EvalRun

#: The most results a reader of a whole run holds at once.
#:
#: A result is a few kilobytes to a few hundred (a judge's reasoning, a kind's payload, a world's events), so a page of
#: this many is megabytes at most, where a large run read whole was the run's whole volume.
RUN_RESULTS_PAGE: Final[int] = 200


#: The fields of a stored result that hold what its cell did verbatim: a world's whole event log and its kind's own
#: payload. Each grows without bound with what the cell did, and no reading of a run's results as figures reads either:
#: the engine's comparisons, counts and calibrations read the scores, outcomes, usage and measures beside them, and a
#: host reads its kind's payload as the cell ends, into the measures it reports.
VERBATIM_RESULT_FIELDS: Final = ("world_events", "kind_payload")


class UnreadableResultKey(StorageError):
    """A stored result row whose key (id, case, ``k``, model, when stored) cannot be read, so its run cannot be walked a
    page at a time (status 503, refused as a storage error).

    Raised by the key read (:meth:`RunResultsStore.query_eval_result_keys_by_run`) for a row, never for a store that
    failed to answer: a reader that tolerates a row this build cannot read (a lost run's count of what it stored)
    catches this and lets every other storage failure through.
    """


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


def iter_stored_results(storage: RunResultsStore, run_id: str, scope_id: str) -> Iterator[EvalResult]:
    """Every result of the run in the order the store keeps them, :data:`RUN_RESULTS_PAGE` at a time.

    The store's order is the order a read of the whole run (``query_eval_results_by_run``) hands its results out in,
    so a reader that folds them into figures gives the figures that read gave, sums and all. The run's keys are read
    once, when the walk starts, in that order (:meth:`RunResultsStore.query_eval_result_keys_by_run`); each page is
    read by id when the one before it has been walked. A result removed since its key was read is absent.

    :param storage: Where the run's results are read.
    :ptype storage: RunResultsStore
    :param run_id: The run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The results.
    :rtype: Iterator[EvalResult]
    """
    return iter_keyed_results(storage, scope_id, storage.query_eval_result_keys_by_run(run_id, scope_id))


def iter_keyed_results(storage: RunResultsStore, scope_id: str, keys: Sequence[ResultKey]) -> Iterator[EvalResult]:
    """The results ``keys`` name, in the keys' order, :data:`RUN_RESULTS_PAGE` at a time: each page read by id when the
    one before it has been walked. A result removed since its key was read is absent.

    :param storage: Where the results are read.
    :ptype storage: RunResultsStore
    :param scope_id: Their scope.
    :ptype scope_id: str
    :param keys: Their keys, in the order to walk them: a run's (:meth:`RunResultsStore.query_eval_result_keys_by_run`)
        in the store's order or the reader's, or some of them.
    :ptype keys: Sequence[ResultKey]
    :return: The results.
    :rtype: Iterator[EvalResult]
    """
    page = RUN_RESULTS_PAGE
    for start in range(0, len(keys), page):
        yield from read_results(storage, scope_id, keys[start : start + page])


def without_payloads(result: EvalResult) -> EvalResult:
    """The result without its verbatim fields, ``world_events`` and ``kind_payload`` (each read as none), or itself
    when it carries neither.

    :param result: A stored result.
    :ptype result: EvalResult
    :return: The result as a reader of figures keeps it.
    :rtype: EvalResult
    """
    if all(getattr(result, name) is None for name in VERBATIM_RESULT_FIELDS):
        return result
    return result.model_copy(update=dict.fromkeys(VERBATIM_RESULT_FIELDS))


def results_without_payloads(storage: RunResultsStore, run_id: str, scope_id: str) -> list[EvalResult]:
    """Every result of the run, in the store's order, read a page at a time and each kept without its verbatim fields.

    For a reader that needs every result of the run at once to fold them into figures (the cases two runs share, a
    composite per case): it holds the run's results without what grows without bound with each cell
    (``world_events`` and ``kind_payload``), and one page of them whole. Its figures are the ones a read of the whole run
    gave (:func:`iter_stored_results`); a result kept this way is never written back.

    :param storage: Where the run's results are read.
    :ptype storage: RunResultsStore
    :param run_id: The run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The results, in the store's order.
    :rtype: list[EvalResult]
    """
    return [without_payloads(result) for result in iter_stored_results(storage, run_id, scope_id)]


__all__ = [
    "RUN_RESULTS_PAGE",
    "ResultKey",
    "RunResultsStore",
    "UnreadableResultKey",
    "iter_keyed_results",
    "iter_run_results",
    "iter_stored_results",
    "read_results",
    "results_without_payloads",
    "run_case_pages",
    "run_result_keys",
    "without_payloads",
]
