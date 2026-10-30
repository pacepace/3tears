"""The readings conformance kit passes on the in-memory reference store, and each of its cases can fail.

* **The reference store conforms.** Every case of ``RESULT_READINGS_CONFORMANCE_CASES`` runs against
  ``InMemoryDocumentStore``, its backend hook installed on the store's own ``documents`` dict, whose ``items()`` is
  the one pass a read makes.
* **No case is vacuous.** :data:`_FAULTS` is a table of wrong stores, each the reference store with one rule broken,
  and the cases each must turn red. A case no fault reaches fails :func:`test_every_case_is_turned_red_by_some_fault`.
* **What is the reference store's own:** a narrowed read is a snapshot taken at the call.
"""

from __future__ import annotations

import json
from collections.abc import Callable, ItemsView, Iterator, Mapping, Sequence
from typing import Any

import pytest

from threetears.evals.schema import DocumentStore, GroupMeasure, ResultReadings, narrow_items, omit_paths
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.testing import (
    RESULT_READINGS_CONFORMANCE_CASES,
    ResultReadingsConformanceCase,
    StoreConformanceFailure,
)


class _HookedDocuments(dict[tuple[str, str], dict[str, Any]]):
    """The store's documents, calling the installed hook on each pass a read makes over them."""

    hook: Callable[[], None] | None = None

    def items(self) -> ItemsView[tuple[str, str], dict[str, Any]]:
        if self.hook is not None:
            self.hook()
        return super().items()


def _hooked(store: InMemoryDocumentStore) -> Callable[[Callable[[], None]], None]:
    """Swap the store's documents for hooked ones, and return the installer the kit calls."""
    documents = _HookedDocuments(store.documents)
    store.documents = documents

    def install(hook: Callable[[], None]) -> None:
        documents.hook = hook

    return install


def _run(case: ResultReadingsConformanceCase, store: InMemoryDocumentStore) -> None:
    case.run(store, _hooked(store))


# --- the reference store conforms ------------------------------------------------------------------


@pytest.mark.parametrize("case", RESULT_READINGS_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_the_reference_store_conforms(case: ResultReadingsConformanceCase) -> None:
    _run(case, InMemoryDocumentStore())


def test_case_names_are_unique() -> None:
    names = [case.name for case in RESULT_READINGS_CONFORMANCE_CASES]
    assert len(names) == len(set(names))


def test_it_offers_the_capability() -> None:
    assert isinstance(InMemoryDocumentStore(), ResultReadings)


def test_a_store_without_the_capability_fails_every_case_by_name() -> None:
    class _PortOnly:
        """Offers neither method of the capability."""

    case = RESULT_READINGS_CONFORMANCE_CASES[0]
    with pytest.raises(StoreConformanceFailure, match="does not offer ResultReadings") as raised:
        case.run(_PortOnly(), lambda _hook: None)  # type: ignore[arg-type]
    assert str(raised.value).startswith(f"{case.name}: {case.rule} — ")


def test_a_narrowed_read_is_a_snapshot_taken_at_the_call() -> None:
    store = InMemoryDocumentStore()
    store.upsert({"id": "a", "scope_id": "s", "doc_type": "t", "n": 1})
    rows = store.iter_narrowed("t", "s")
    store.upsert({"id": "b", "scope_id": "s", "doc_type": "t", "n": 2})
    store.merge_fields("a", "s", {"n": 3})
    assert list(rows) == [{"id": "a", "scope_id": "s", "doc_type": "t", "n": 1}]


# --- the faults: each the reference store with one rule broken -------------------------------------


def _read(document: dict[str, Any], path: str) -> Any:
    node: Any = document
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _typed(value: Any) -> str:
    return json.dumps(value)


def _sum_int(values: list[Any]) -> int:
    present = [v for v in values if v is not None]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in present):
        raise TypeError("sum_int over a non-integer")
    return sum(present)


def _extreme(pick: Callable[[list[Any]], Any]) -> Callable[[list[Any]], Any]:
    def measure(values: list[Any]) -> Any:
        present = [v for v in values if v is not None]
        if any(isinstance(v, bool) or not isinstance(v, int | float | str) for v in present):
            raise TypeError("an extreme of a value that is neither a number nor a string")
        if len({isinstance(v, str) for v in present}) > 1:
            raise TypeError("an extreme of numbers and strings")
        return pick(present) if present else None

    return measure


#: A plain fold's measures over one group's values; ``values`` is handled with the row ids.
_KINDS: dict[str, Callable[[list[Any]], Any]] = {
    "count_null": lambda values: sum(1 for v in values if v is None),
    "sum_int": _sum_int,
    "min": _extreme(min),
    "max": _extreme(max),
}


class _PlainFold(InMemoryDocumentStore):
    """``grouped`` folded over ``iter_narrowed``'s rows; each fault below changes one piece of it."""

    #: How a ``by`` value becomes a group key.
    group_key: Callable[[Any], Any] = staticmethod(_typed)
    #: The measures by kind, overridden by a fault.
    kinds: Mapping[str, Callable[[list[Any]], Any]] = _KINDS
    #: Whether no ``by`` paths make a group even when nothing matched.
    group_of_nothing = False

    def ordered_values(self, pairs: list[list[Any]]) -> list[list[Any]]:
        return pairs

    def grouped(
        self,
        doc_type: str,
        scope_id: str,
        *,
        by: Sequence[str],
        measures: Mapping[str, GroupMeasure],
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        if set(by) & measures.keys():
            raise ValueError("a measure named as a by path")
        for kind, path in measures.values():
            if kind not in {"count", "values", *self.kinds} or (kind == "count") != (path is None):
                raise ValueError(f"an undefined measure {kind!r}")
        rows = list(self.iter_narrowed(doc_type, scope_id, field_in=field_in, **field_eq))
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            values = [_read(row, path) for path in by]
            if any(isinstance(value, dict | list) for value in values):
                raise TypeError("grouping by an object")
            groups.setdefault(tuple(self.group_key(value) for value in values), []).append(row)
        if not by and not groups and self.group_of_nothing:
            groups[()] = []
        answers = []
        for members in groups.values():
            answer: dict[str, Any] = {path: _read(members[0], path) if members else None for path in by}
            for name, (kind, path) in measures.items():
                if kind == "count" or path is None:
                    answer[name] = len(members)
                elif kind == "values":
                    answer[name] = self.ordered_values([[row["id"], _read(row, path)] for row in members])
                else:
                    answer[name] = self.kinds[kind]([_read(row, path) for row in members])
            answers.append(answer)
        return answers


def test_the_plain_fold_conforms() -> None:
    """The faults' base is right, so each fault's red is its own break."""
    for case in RESULT_READINGS_CONFORMANCE_CASES:
        _run(case, _PlainFold())


class _GroupsByEquality(_PlainFold):
    """Groups by Python equality, so ``True`` and ``1`` fall in one group."""

    group_key = staticmethod(lambda value: value)


class _CountNullCountsFalsy(_PlainFold):
    """``count_null`` counts every falsy value, ``0`` and ``false`` included."""

    kinds = {**_KINDS, "count_null": lambda values: sum(1 for v in values if not v)}


class _SumsAnyNumber(_PlainFold):
    """``sum_int`` adds floats and booleans: a float measure by another name."""

    kinds = {**_KINDS, "sum_int": lambda values: sum(v for v in values if isinstance(v, int | float))}


class _ComparesAsText(_PlainFold):
    """``min`` and ``max`` compare every value as its text, so ``10`` sorts below ``9``."""

    kinds = {
        **_KINDS,
        "min": lambda values: min((v for v in values if v is not None), key=str, default=None),
        "max": lambda values: max((v for v in values if v is not None), key=str, default=None),
    }


class _ValuesInIdOrder(_PlainFold):
    """``values`` comes sorted by row id rather than in the stored order."""

    def ordered_values(self, pairs: list[list[Any]]) -> list[list[Any]]:
        return sorted(pairs, key=lambda pair: str(pair[0]))


class _AcceptsAnyMeasure(_PlainFold):
    """An unknown kind, or a measure named as a ``by`` path, answers ``None`` rather than being refused."""

    kinds = {**_KINDS, "mean": lambda values: None}

    def grouped(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        measures = {name: measure for name, measure in kwargs["measures"].items() if name not in kwargs["by"]}
        return super().grouped(doc_type, scope_id, **{**kwargs, "measures": measures})


class _GroupOfNothing(_PlainFold):
    """No ``by`` paths over no matching rows answers one group of zeros, as a bare SQL aggregate does."""

    group_of_nothing = True


class _SortsById(InMemoryDocumentStore):
    """``iter_narrowed`` sorts its rows by id rather than keeping ``by_doc_type``'s order."""

    def iter_narrowed(self, doc_type: str, scope_id: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        return iter(sorted(super().iter_narrowed(doc_type, scope_id, **kwargs), key=lambda row: str(row["id"])))


class _NarrowsThroughNonLists(InMemoryDocumentStore):
    """``exclude_in_items`` also narrows a field that is an object rather than a list."""

    def iter_narrowed(
        self,
        doc_type: str,
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        exclude_in_items: Mapping[str, Sequence[str]] = {},
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> Iterator[dict[str, Any]]:
        rows = super().iter_narrowed(
            doc_type, scope_id, exclude=exclude, exclude_in_items=exclude_in_items, field_in=field_in, **field_eq
        )
        for row in rows:
            for path, item_paths in exclude_in_items.items():
                if isinstance(row.get(path), dict):
                    row = {**row, path: omit_paths(row[path], item_paths)}
            yield row


class _ReadsTwice(InMemoryDocumentStore):
    """``iter_narrowed`` reads the matching ids first and then the rows: two backend calls."""

    def iter_narrowed(self, doc_type: str, scope_id: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        predicates = {key: value for key, value in kwargs.items() if key not in ("exclude", "exclude_in_items")}
        self.by_doc_type(doc_type, scope_id, **predicates)
        return super().iter_narrowed(doc_type, scope_id, **kwargs)


class _IgnoresScope(_PlainFold):
    """``iter_narrowed`` and ``grouped`` read every scope."""

    def iter_narrowed(
        self,
        doc_type: str,
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        exclude_in_items: Mapping[str, Sequence[str]] = {},
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> Iterator[dict[str, Any]]:
        scopes = dict.fromkeys(scope for scope, _ in self.documents)
        return iter(
            [
                row
                for scope in scopes
                for row in super().iter_narrowed(
                    doc_type,
                    scope,
                    exclude=exclude,
                    exclude_in_items=exclude_in_items,
                    field_in=field_in,
                    **field_eq,
                )
            ]
        )


class _PythonEquality(_PlainFold):
    """Predicates compared with Python's ``==``, so ``True`` matches ``1`` and an absent field never matches ``None``."""

    def iter_narrowed(
        self,
        doc_type: str,
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        exclude_in_items: Mapping[str, Sequence[str]] = {},
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> Iterator[dict[str, Any]]:
        if field_in and set(field_in) & set(field_eq):
            raise ValueError("a field named twice")
        if field_in and any(isinstance(values, str) for values in field_in.values()):
            raise TypeError("a string field_in")
        rows = self.by_doc_type(doc_type, scope_id)
        return iter(
            [
                narrow_items(omit_paths(row, exclude), exclude_in_items)
                for row in rows
                if all(field in row and row[field] == value for field, value in field_eq.items())
                and all(row.get(field) in values for field, values in (field_in or {}).items())
            ]
        )


class _SwallowsFailures(InMemoryDocumentStore):
    """A backend failure reads as an empty answer."""

    def iter_narrowed(self, doc_type: str, scope_id: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        try:
            return super().iter_narrowed(doc_type, scope_id, **kwargs)
        except Exception:  # the fault under test is exactly this swallow
            return iter(())

    def grouped(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        try:
            return super().grouped(doc_type, scope_id, **kwargs)
        except Exception:  # the fault under test is exactly this swallow
            return []


_FAULTS: dict[type[DocumentStore], frozenset[str]] = {
    _SortsById: frozenset({"narrowed.equals_by_doc_type"}),
    _NarrowsThroughNonLists: frozenset({"narrowed.list_items"}),
    _ReadsTwice: frozenset({"calls.one_per_read"}),
    _IgnoresScope: frozenset({"scope.isolated"}),
    _PythonEquality: frozenset({"predicates.match_by_doc_type"}),
    _GroupsByEquality: frozenset({"grouped.by_json_type"}),
    _CountNullCountsFalsy: frozenset({"grouped.count_and_count_null"}),
    _SumsAnyNumber: frozenset({"grouped.sum_int"}),
    _ComparesAsText: frozenset({"grouped.min_max"}),
    _ValuesInIdOrder: frozenset({"grouped.values_in_stored_order"}),
    _AcceptsAnyMeasure: frozenset({"grouped.undefined_grouping_refused"}),
    _GroupOfNothing: frozenset({"empty.answers"}),
    _SwallowsFailures: frozenset({"failure.raises"}),
}


def test_the_fault_table_names_only_real_cases() -> None:
    named = frozenset().union(*_FAULTS.values())
    assert named <= {case.name for case in RESULT_READINGS_CONFORMANCE_CASES}


def test_every_case_is_turned_red_by_some_fault() -> None:
    covered = frozenset().union(*_FAULTS.values())
    assert {case.name for case in RESULT_READINGS_CONFORMANCE_CASES} <= covered


@pytest.mark.parametrize(
    ("fault", "case_name"),
    [(fault, name) for fault, names in _FAULTS.items() for name in sorted(names)],
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
def test_a_wrong_store_fails_the_case_for_the_rule_it_breaks(fault: type[DocumentStore], case_name: str) -> None:
    case = next(c for c in RESULT_READINGS_CONFORMANCE_CASES if c.name == case_name)
    store = fault()
    assert isinstance(store, InMemoryDocumentStore)
    with pytest.raises(StoreConformanceFailure) as raised:
        _run(case, store)
    assert str(raised.value).startswith(f"{case.name}: {case.rule} — ")
