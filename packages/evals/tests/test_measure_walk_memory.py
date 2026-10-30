"""The bundle's measure walk memo stays small per result, is dropped when an assembly raises, and nests.

One assembly walks each result's measures once and holds the walk for every fold that pools the result again
(``analysis/bundle/measures.py``). What it holds grows with the results read, so it is bounded per result: at most
:data:`MEMO_BYTES_PER_RESULT`, which keeps the memo under the D8 reading ceiling (256 MiB) at 250,000 results, the
largest reading budget the pushdown plan contemplates. The bytes counted are every allocation made in ``measures.py``
still live late in the assembly, read when the assembly asks its store for the case strata, after every fold that
pools the campaign's results has run.
"""

from __future__ import annotations

import functools
import gc
import tracemalloc
from collections.abc import Callable, Sequence

import pytest

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.schema import EvalCaseStratum
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.hot_path_support import HotPathCorpus, hot_path_corpus

#: The most the memo may hold per result read: the D8 ceiling (256 MiB) over 250,000 results is about 1,074 bytes.
MEMO_BYTES_PER_RESULT = 1024

#: What ``measures.py`` may still hold once an assembly has ended, however many results it read: the walk memo is
#: gone, so nothing that grows with the results is left.
AFTER_ASSEMBLY_BYTES = 64 * 1024

#: Allocations made in the module the memo lives in.
MEASURES_MODULE = [tracemalloc.Filter(True, "*/analysis/bundle/measures.py")]


def _measures_bytes() -> int:
    """Bytes allocated in ``measures.py`` and live now."""
    return sum(stat.size for stat in tracemalloc.take_snapshot().filter_traces(MEASURES_MODULE).statistics("filename"))


@functools.cache
def _corpus() -> tuple[HotPathCorpus, InMemoryDocumentStore]:
    """The production-shaped campaign at 1x, with the document store it is held in."""
    store = InMemoryDocumentStore()
    return hot_path_corpus(scale=1, store=store), store


class _StrataHook(EvalStorage):
    """The campaign's store, running ``hook`` when an assembly asks for the case strata (after its pooling folds)."""

    def __init__(self, store: InMemoryDocumentStore, hook: Callable[[], None]) -> None:
        super().__init__(store)
        self.hook = hook

    def load_case_strata(self, test_case_ids: Sequence[str], scope_id: str, /) -> list[EvalCaseStratum]:
        self.hook()
        return super().load_case_strata(test_case_ids, scope_id)


class _Stopped(Exception):
    """Raised from inside an assembly, to see what it leaves behind."""


def _assembled(storage: EvalStorage) -> AnalysisContextBundle:
    corpus, _store = _corpus()
    return assemble_context_bundle(corpus.campaign, storage=storage, profile=corpus.host.profile)


def test_the_memo_holds_at_most_a_kibibyte_per_result() -> None:
    corpus, store = _corpus()
    held: list[int] = []
    gc.collect()
    tracemalloc.start()
    try:
        _assembled(_StrataHook(store, lambda: held.append(_measures_bytes())))
    finally:
        tracemalloc.stop()

    assert held, "the assembly never asked for the case strata, so nothing was measured"
    assert held[0] / corpus.n_results <= MEMO_BYTES_PER_RESULT, (
        f"the measure walk memo held {held[0]:,} bytes over {corpus.n_results:,} results "
        f"({held[0] / corpus.n_results:,.0f} per result, bound {MEMO_BYTES_PER_RESULT:,})"
    )


def test_the_memo_is_dropped_when_an_assembly_raises() -> None:
    _corpus_value, store = _corpus()
    held: list[int] = []

    def stop() -> None:
        held.append(_measures_bytes())
        raise _Stopped

    gc.collect()
    tracemalloc.start()
    try:
        with pytest.raises(_Stopped):
            _assembled(_StrataHook(store, stop))
        gc.collect()
        after = _measures_bytes()
    finally:
        tracemalloc.stop()

    assert held and held[0] > AFTER_ASSEMBLY_BYTES, "the memo held nothing to drop, so this tests nothing"
    assert after <= AFTER_ASSEMBLY_BYTES, f"{after:,} bytes of the walk memo outlived the assembly that raised"


def test_an_assembly_inside_an_assembly_reads_as_either_alone() -> None:
    _corpus_value, store = _corpus()
    alone = _assembled(EvalStorage(store))
    inner: list[AnalysisContextBundle] = []

    def nest() -> None:
        if not inner:
            inner.append(_assembled(EvalStorage(store)))

    outer = _assembled(_StrataHook(store, nest))

    assert inner, "the outer assembly never ran the inner one"
    for nested in (inner[0], outer):
        assert nested.fingerprint() == alone.fingerprint()
        assert nested.model_dump_json() == alone.model_dump_json()
