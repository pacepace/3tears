"""A router offers ``ResultReadings`` exactly when every store a type is placed in does, and passes the kit then.

* **It conforms.** Every case of ``RESULT_READINGS_CONFORMANCE_CASES`` runs over a router of one capable store, and
  over a router of two capable stores with the kit's types placed in one of them. The kit's backend hook is installed
  on the store the kit's types are placed in, so "one backend call" is that store's one call.
* **A read is never split.** A narrowed or grouped read reaches only the store its type is placed in.
* **Offered only when it can be honoured.** A placed store without the capability takes it from the router: the router
  then has no ``iter_narrowed`` at all, rather than one that raises. A store no type is placed in does not.
* **It composes with the clock.** All four (clock, readings) combinations of stores build a router offering exactly
  those capabilities, each of them working.
"""

from __future__ import annotations

from collections.abc import Callable, ItemsView, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.evals.schema import DocumentStore, ResultReadings, StoreClock
from threetears.evals.storage import DocTypeRoute, InMemoryDocumentStore, RoutedDocumentStore
from threetears.evals.testing import RESULT_READINGS_CONFORMANCE_CASES, ResultReadingsConformanceCase

_SCOPE = "scope-a"
_ROUTING = "evals-routing"

#: The kit writes only types the registry does not hold, so they are kept in ``scratch``: the store the hook goes on.
_KIT_IN_SCRATCH = DocTypeRoute(
    by="lifecycle", to={"kept": "evidence", "durable": "evidence", "regenerable": "scratch"}, unregistered="scratch"
)

#: Every type in one store; a third store, ``draining``, holds no placement.
_ALL_IN_EVIDENCE = DocTypeRoute(
    by="lifecycle", to={"kept": "evidence", "durable": "evidence", "regenerable": "evidence"}, unregistered="evidence"
)

_EVIDENCE_TIME = datetime(2026, 3, 1, tzinfo=UTC)


class _HookedDocuments(dict[tuple[str, str], dict[str, Any]]):
    """A store's documents, calling the installed hook on each pass a read makes over them."""

    hook: Callable[[], None] | None = None

    def items(self) -> ItemsView[tuple[str, str], dict[str, Any]]:
        if self.hook is not None:
            self.hook()
        return super().items()


def _hooked(store: InMemoryDocumentStore) -> _HookedDocuments:
    """Swap the store's documents for hooked ones, and return them."""
    documents = _HookedDocuments(store.documents)
    store.documents = documents
    return documents


def _installer(store: InMemoryDocumentStore) -> Callable[[Callable[[], None]], None]:
    documents = _hooked(store)

    def install(hook: Callable[[], None]) -> None:
        documents.hook = hook

    return install


class _PortOnlyStore(DocumentStore):
    """The in-memory store behind the port alone: no readings."""

    def __init__(self) -> None:
        self.inner = InMemoryDocumentStore()

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        return self.inner.get(doc_id, scope_id)

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        return self.inner.get_with_etag(doc_id, scope_id)

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        return self.inner.get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep)

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        return self.inner.by_doc_type(
            doc_type,
            scope_id,
            order_by=order_by,
            descending=descending,
            limit=limit,
            exclude=exclude,
            field_in=field_in,
            **field_eq,
        )

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        return self.inner.iter_by_doc_type(doc_type, scope_id)

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        self.inner.upsert(document, if_match=if_match, create_only=create_only)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        return self.inner.merge_fields(doc_id, scope_id, fields)

    def delete(self, doc_id: str, scope_id: str) -> bool:
        return self.inner.delete(doc_id, scope_id)


class _ClockedPortOnlyStore(_PortOnlyStore):
    """A port-only store that keeps a clock: the clock without the readings."""

    def __init__(self, time: datetime) -> None:
        super().__init__()
        self.time = time

    def now(self) -> datetime:
        return self.time


class _ClockedReadingStore(InMemoryDocumentStore):
    """The in-memory store, which offers the readings, keeping a clock too."""

    def __init__(self, time: datetime) -> None:
        super().__init__()
        self.time = time

    def now(self) -> datetime:
        return self.time


def _router(children: Mapping[str, DocumentStore], route: DocTypeRoute = _KIT_IN_SCRATCH) -> RoutedDocumentStore:
    """A started router over ``children``: relocated (empty, so only the record is written), then opened."""
    RoutedDocumentStore.relocate(children, route, scope_ids=[], routing_scope=_ROUTING)
    return RoutedDocumentStore(children, route, routing_scope=_ROUTING)


def _doc(doc_id: str, doc_type: str, **fields: Any) -> dict[str, Any]:
    return {"id": doc_id, "scope_id": _SCOPE, "doc_type": doc_type, **fields}


# --- it conforms ----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", RESULT_READINGS_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_a_router_of_one_capable_store_conforms(case: ResultReadingsConformanceCase) -> None:
    only = InMemoryDocumentStore()
    route = DocTypeRoute(
        by="lifecycle", to={"kept": "only", "durable": "only", "regenerable": "only"}, unregistered="only"
    )
    router = _router({"only": only}, route)
    case.run(router, _installer(only))


@pytest.mark.parametrize("case", RESULT_READINGS_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_a_router_of_two_capable_stores_conforms(case: ResultReadingsConformanceCase) -> None:
    evidence, scratch = InMemoryDocumentStore(), InMemoryDocumentStore()
    router = _router({"evidence": evidence, "scratch": scratch})
    asked: list[int] = []
    _hooked(evidence).hook = lambda: asked.append(1)
    case.run(router, _installer(scratch))
    assert asked == [], "the store the kit's types are not placed in was read"


# --- a read is never split ----------------------------------------------------------------------------


def test_a_read_reaches_only_the_store_its_type_is_placed_in() -> None:
    evidence, scratch = InMemoryDocumentStore(), InMemoryDocumentStore()
    router = _router({"evidence": evidence, "scratch": scratch})
    router.upsert(_doc("r1", "eval_run", cost=1))
    router.upsert(_doc("a1", "eval_analysis", cost=2))
    assert isinstance(router, ResultReadings)
    evidence_reads: list[int] = []
    scratch_reads: list[int] = []
    _hooked(evidence).hook = lambda: evidence_reads.append(1)
    _hooked(scratch).hook = lambda: scratch_reads.append(1)

    assert [row["id"] for row in router.iter_narrowed("eval_run", _SCOPE)] == ["r1"]
    assert router.grouped("eval_run", _SCOPE, by=[], measures={"n": ("count", None)}) == [{"n": 1}]
    assert (len(evidence_reads), len(scratch_reads)) == (2, 0)
    assert [row["id"] for row in router.iter_narrowed("eval_analysis", _SCOPE, exclude=["cost"])] == ["a1"]
    assert (len(evidence_reads), len(scratch_reads)) == (2, 1)


# --- offered only when it can be honoured ---------------------------------------------------------------


@pytest.mark.parametrize("without", ["evidence", "scratch"])
def test_a_placed_store_without_the_capability_takes_it_from_the_router(without: str) -> None:
    children: dict[str, DocumentStore] = {"evidence": InMemoryDocumentStore(), "scratch": InMemoryDocumentStore()}
    children[without] = _PortOnlyStore()
    router = _router(children)
    assert not isinstance(router, ResultReadings)
    assert not hasattr(router, "iter_narrowed") and not hasattr(router, "grouped"), "no method that raises"
    router.upsert(_doc("r1", "eval_run"))
    assert [row["id"] for row in router.by_doc_type("eval_run", _SCOPE)] == ["r1"], "the port still answers"


def test_a_store_no_type_is_placed_in_does_not_take_it() -> None:
    evidence = InMemoryDocumentStore()
    router = _router({"evidence": evidence, "draining": _PortOnlyStore()}, _ALL_IN_EVIDENCE)
    assert isinstance(router, ResultReadings)
    router.upsert(_doc("r1", "eval_run", n=3))
    assert list(router.iter_narrowed("eval_run", _SCOPE, exclude=["n"])) == [_doc("r1", "eval_run")]


def test_a_route_naming_a_store_that_is_not_there_is_still_refused_as_before() -> None:
    route = DocTypeRoute(by="lifecycle", to={"kept": "a", "durable": "a", "regenerable": "b"}, unregistered="a")
    with pytest.raises(ValueError, match="not among"):
        RoutedDocumentStore({"a": InMemoryDocumentStore()}, route, routing_scope=_ROUTING)


# --- it composes with the clock -------------------------------------------------------------------------


def _store(clock: bool, readings: bool, time: datetime) -> DocumentStore:
    if clock and readings:
        return _ClockedReadingStore(time)
    if clock:
        return _ClockedPortOnlyStore(time)
    if readings:
        return InMemoryDocumentStore()
    return _PortOnlyStore()


@pytest.mark.parametrize("clock", [False, True], ids=["no-clock", "clock"])
@pytest.mark.parametrize("readings", [False, True], ids=["no-readings", "readings"])
def test_each_clock_and_readings_combination_offers_exactly_those(clock: bool, readings: bool) -> None:
    router = _router(
        {"evidence": _store(clock, readings, _EVIDENCE_TIME), "scratch": _store(clock, readings, _EVIDENCE_TIME)}
    )
    assert isinstance(router, RoutedDocumentStore)
    assert isinstance(router, StoreClock) is clock
    assert isinstance(router, ResultReadings) is readings
    router.upsert(_doc("r1", "eval_run", cost=4, heavy="x"))
    if isinstance(router, StoreClock):
        assert router.now() == _EVIDENCE_TIME
    if isinstance(router, ResultReadings):
        assert list(router.iter_narrowed("eval_run", _SCOPE, exclude=["heavy"])) == [_doc("r1", "eval_run", cost=4)]
        assert router.grouped("eval_run", _SCOPE, by=[], measures={"cost": ("sum_int", "cost")}) == [{"cost": 4}]


def test_a_clock_on_every_store_and_readings_on_one_gives_the_clock_alone() -> None:
    router = _router(
        {"evidence": _ClockedReadingStore(_EVIDENCE_TIME), "scratch": _ClockedPortOnlyStore(_EVIDENCE_TIME)}
    )
    assert isinstance(router, StoreClock) and not isinstance(router, ResultReadings)
