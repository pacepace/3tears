"""The store conformance kit passes on the in-memory reference store, and each of its cases can fail.

Three things are proved here:

* **The reference store conforms.** Every case of ``STORE_CONFORMANCE_CASES`` runs against
  ``InMemoryDocumentStore``, parametrised exactly the way the kit's docstring tells an adopter to.
* **No case is vacuous.** :data:`_FAULTS` is a table of broken stores, each the reference store with one
  rule of the port broken, and the cases each one must turn red. A case that no fault reaches fails
  :func:`test_every_case_is_turned_red_by_some_fault` — a check nothing can fail is a check that proves
  nothing, and the kit is only worth what it can catch in someone else's adapter.
* **What is the reference store's own**, beyond the port: a document without its scope or id is
  refused, a sweep may delete as it iterates, and a conditional write's compare and write happen under
  one lock, so two threads holding one etag cannot both land.
"""

from __future__ import annotations

import copy
import itertools
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from threetears.evals.contracts import DocumentStore, StoreConflict
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.testing import STORE_CONFORMANCE_CASES, StoreConformanceCase, StoreConformanceFailure

_SCOPE = "scope-a"


def _doc(doc_id: str, **fields: object) -> dict[str, object]:
    return {"id": doc_id, "scope_id": _SCOPE, "doc_type": "eval_run", **fields}


# --- the reference store conforms ------------------------------------------------------------------


@pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_the_reference_store_conforms(case: StoreConformanceCase) -> None:
    case.run(InMemoryDocumentStore())


def test_case_names_are_unique() -> None:
    names = [case.name for case in STORE_CONFORMANCE_CASES]
    assert len(names) == len(set(names))


def test_a_failure_names_the_case_and_the_rule() -> None:
    case = next(c for c in STORE_CONFORMANCE_CASES if c.name == "etag.stale_write_is_refused")
    with pytest.raises(StoreConformanceFailure) as raised:
        case.run(_IgnoresIfMatch())
    message = str(raised.value)
    assert message.startswith(f"{case.name}: {case.rule} — ")
    assert "expected StoreConflict" in message


def test_a_store_raising_something_else_is_not_reported_as_a_broken_rule() -> None:
    """A backend fault propagates as itself, so an adopter reads its own error rather than the kit's."""

    class _Down(InMemoryDocumentStore):
        def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
            raise ConnectionError("database is down")

    case = next(c for c in STORE_CONFORMANCE_CASES if c.name == "scope.miss_is_an_outcome")
    with pytest.raises(ConnectionError, match="database is down"):
        case.run(_Down())


# --- the faults: each the reference store with one rule broken -------------------------------------


class _MergeCreates(InMemoryDocumentStore):
    """``merge_fields`` on a missing document creates it."""

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        if self.get(doc_id, scope_id) is None:
            self.upsert({"id": doc_id, "scope_id": scope_id, "doc_type": "conformance_doc", **fields})
            return True
        return super().merge_fields(doc_id, scope_id, fields)


class _GetIgnoresScope(InMemoryDocumentStore):
    """``get`` finds an id in whichever scope holds it."""

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        for (_, stored_id), document in self.documents.items():
            if stored_id == doc_id:
                return copy.deepcopy(document)
        return None


class _OneDocumentPerId(InMemoryDocumentStore):
    """A write evicts the same id from every other scope, as a store keyed by id alone would."""

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        for key in [key for key in self.documents if key[1] == document["id"]]:
            del self.documents[key]
        super().upsert(document, if_match=if_match)


class _IgnoresDocType(InMemoryDocumentStore):
    """The typed reads return documents of any type."""

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        return [d for i in doc_ids if (d := self.get(i, scope_id)) is not None]

    def by_doc_type(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [copy.deepcopy(d) for (scope, _), d in self.documents.items() if scope == scope_id]


class _InjectsItsEtag(InMemoryDocumentStore):
    """Reads hand back the store's own ``_etag`` column inside the document."""

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        document, etag = super().get_with_etag(doc_id, scope_id)
        return None if document is None else {**document, "_etag": etag}


class _UpsertMerges(InMemoryDocumentStore):
    """An upsert merges into the stored document instead of replacing it."""

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        stored = self.get(document["id"], document["scope_id"]) or {}
        super().upsert({**stored, **document}, if_match=if_match)


class _SharesReferences(InMemoryDocumentStore):
    """``get`` hands back the stored dict itself."""

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        return self.documents.get((scope_id, doc_id))


def _exclude_top_level(document: dict[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key not in {path.split(".")[0] for path in paths}}


class _ExcludeDropsTheTopField(InMemoryDocumentStore):
    """``exclude`` drops a path's whole top-level field, siblings and all."""

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        exclude = kwargs.pop("exclude", ())
        return [_exclude_top_level(d, exclude) for d in super().get_many(doc_type, doc_ids, scope_id, **kwargs)]

    def by_doc_type(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        exclude = kwargs.pop("exclude", ())
        return [_exclude_top_level(d, exclude) for d in super().by_doc_type(doc_type, scope_id, **kwargs)]


def _exclude_truncating(document: dict[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    for path in paths:
        head, _, rest = path.partition(".")
        if rest and not isinstance(result.get(head), dict):
            result.pop(head, None)
    return result


class _ExcludeTruncatesAtANonObject(InMemoryDocumentStore):
    """An ``exclude`` path that runs through a non-object drops the field it runs through."""

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        exclude = kwargs.get("exclude", ())
        return [_exclude_truncating(d, exclude) for d in super().get_many(doc_type, doc_ids, scope_id, **kwargs)]

    def by_doc_type(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        exclude = kwargs.get("exclude", ())
        return [_exclude_truncating(d, exclude) for d in super().by_doc_type(doc_type, scope_id, **kwargs)]


class _KeepFillsNone(InMemoryDocumentStore):
    """``keep`` hands back every named field, ``None`` for one the document does not carry."""

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        found = super().get_many(doc_type, doc_ids, scope_id, exclude=exclude)
        return [{field: d.get(field) for field in keep} if keep else d for d in found]


class _AcceptsKeepAndExclude(InMemoryDocumentStore):
    """``get_many`` applies ``keep`` and ignores ``exclude`` rather than refusing the pair."""

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        return (
            super().get_many(doc_type, doc_ids, scope_id, keep=keep)
            if keep
            else super().get_many(doc_type, doc_ids, scope_id, exclude=exclude)
        )


class _GetManyRepeats(InMemoryDocumentStore):
    """``get_many`` returns a document once per time its id is named."""

    def get_many(self, doc_type: str, doc_ids: Sequence[str], scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [d for i in doc_ids for d in super().get_many(doc_type, [i], scope_id, **kwargs)]


class _PredicatesCompareText(InMemoryDocumentStore):
    """``by_doc_type`` compares predicates as text, so ``1`` matches ``"1"``."""

    def by_doc_type(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        names = {"order_by", "descending", "limit", "exclude"}
        predicates = {key: value for key, value in kwargs.items() if key not in names}
        rows = super().by_doc_type(doc_type, scope_id, **{key: kwargs[key] for key in names & kwargs.keys()})
        return [d for d in rows if all(str(d.get(key)) == str(value) for key, value in predicates.items())]


class _NoneMatchesOnlyNull(InMemoryDocumentStore):
    """A ``None`` predicate matches a field stored as null, but not one that is absent."""

    def by_doc_type(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        rows = super().by_doc_type(doc_type, scope_id, **kwargs)
        nulls = [key for key, value in kwargs.items() if value is None]
        return [d for d in rows if all(key in d for key in nulls)]


class _OrdersAsText(InMemoryDocumentStore):
    """``order_by`` sorts every value as text, so 10 sorts before 2."""

    def by_doc_type(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        order_by, descending, limit = (
            kwargs.pop("order_by", None),
            kwargs.pop("descending", True),
            kwargs.pop("limit", None),
        )
        rows = super().by_doc_type(doc_type, scope_id, **kwargs)
        if order_by is not None:
            rows.sort(key=lambda d: (d.get(order_by) is not None, str(d.get(order_by))), reverse=descending)
        return rows[:limit] if limit is not None else rows


class _MissingSortsHighest(InMemoryDocumentStore):
    """A document without the order field sorts above every one that has it, as a SQL default can."""

    def by_doc_type(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        order_by, descending = kwargs.get("order_by"), kwargs.get("descending", True)
        rows = super().by_doc_type(doc_type, scope_id, **kwargs)
        if order_by is None:
            return rows
        present = [d for d in rows if d.get(order_by) is not None]
        absent = [d for d in rows if d.get(order_by) is None]
        return absent + present if descending else present + absent


class _LimitsBeforeOrdering(InMemoryDocumentStore):
    """``limit`` cuts the unordered rows, then orders what is left."""

    def by_doc_type(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        limit = kwargs.pop("limit", None)
        order_by, descending = kwargs.pop("order_by", None), kwargs.pop("descending", True)
        rows = super().by_doc_type(doc_type, scope_id, **kwargs)
        rows = rows[:limit] if limit is not None else rows
        if order_by is not None:
            rows.sort(key=lambda d: (d.get(order_by) is not None, d.get(order_by)), reverse=descending)
        return rows


class _NoEtags(InMemoryDocumentStore):
    """A store without conditional writes: no token, and ``if_match`` ignored."""

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        return self.get(doc_id, scope_id), None

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        super().upsert(document)


class _IgnoresIfMatch(InMemoryDocumentStore):
    """The etag compare is broken: every conditional write lands."""

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        super().upsert(document)


class _EtagNeverMoves(InMemoryDocumentStore):
    """Every write leaves the document's etag where it was."""

    def _mint_etag(self, key: tuple[str, str]) -> None:
        self._etags.setdefault(key, f"etag-{key[1]}")


class _MergeKeepsTheEtag(InMemoryDocumentStore):
    """``merge_fields`` writes without minting a new etag."""

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        _, etag = self.get_with_etag(doc_id, scope_id)
        landed = super().merge_fields(doc_id, scope_id, fields)
        if etag is not None:
            self._etags[(scope_id, doc_id)] = etag
        return landed


class _EtagSurvivesDelete(InMemoryDocumentStore):
    """``delete`` forgets the document but not its etag, so the old token still writes."""

    def delete(self, doc_id: str, scope_id: str) -> bool:
        etag = self._etags.get((scope_id, doc_id))
        deleted = super().delete(doc_id, scope_id)
        if etag is not None:
            self._etags[(scope_id, doc_id)] = etag
        return deleted


class _EtagIsAVersionFromOne(InMemoryDocumentStore):
    """The etag is a per-document version that restarts at 1, so a re-created document reuses its predecessor's."""

    def __init__(self) -> None:
        super().__init__()
        self._versions: dict[tuple[str, str], Iterator[int]] = {}

    def _mint_etag(self, key: tuple[str, str]) -> None:
        self._etags[key] = f"v{next(self._versions.setdefault(key, itertools.count(1)))}"

    def delete(self, doc_id: str, scope_id: str) -> bool:
        self._versions.pop((scope_id, doc_id), None)
        return super().delete(doc_id, scope_id)


class _RefusesUnconditionalOverwrite(InMemoryDocumentStore):
    """A write without ``if_match`` to a document that exists is refused."""

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        if if_match is None and self.get(document["id"], document["scope_id"]) is not None:
            raise StoreConflict("refusing an unconditional overwrite")
        super().upsert(document, if_match=if_match)


class _MergeReplaces(InMemoryDocumentStore):
    """``merge_fields`` replaces the document with the locating fields plus the merged ones."""

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        stored = self.get(doc_id, scope_id)
        if stored is None:
            return False
        self.upsert({"id": doc_id, "scope_id": scope_id, "doc_type": stored["doc_type"], **fields})
        return True


class _MergeSetsLocatingFields(InMemoryDocumentStore):
    """``merge_fields`` writes whatever it is handed, the locating fields included."""

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        document = self.documents.get((scope_id, doc_id))
        if document is None or not fields:
            return False
        document.update(fields)
        self._mint_etag((scope_id, doc_id))
        return True


class _DeleteAlwaysReportsTrue(InMemoryDocumentStore):
    """``delete`` reports ``True`` whether or not there was a document."""

    def delete(self, doc_id: str, scope_id: str) -> bool:
        super().delete(doc_id, scope_id)
        return True


class _SweepIgnoresDocType(InMemoryDocumentStore):
    """``iter_by_doc_type`` yields every id in the scope, whatever its type."""

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        yield from [doc_id for (scope, doc_id) in list(self.documents) if scope == scope_id]


#: Each broken store, and the cases it must turn red. A case may fail under other faults too; what is
#: asserted is that these ones go red, and that every case is listed against at least one fault.
_FAULTS: dict[type[InMemoryDocumentStore], frozenset[str]] = {
    _MergeCreates: frozenset({"scope.miss_is_an_outcome"}),
    _GetIgnoresScope: frozenset({"scope.isolates_every_method"}),
    _OneDocumentPerId: frozenset({"scope.one_id_two_scopes"}),
    _IgnoresDocType: frozenset({"scope.doc_type_is_part_of_typed_reads"}),
    _InjectsItsEtag: frozenset({"read.document_as_written"}),
    _UpsertMerges: frozenset({"write.upsert_replaces"}),
    _SharesReferences: frozenset({"write.copied_in_and_out"}),
    _ExcludeDropsTheTopField: frozenset({"projection.exclude_drops_one_leaf"}),
    _ExcludeTruncatesAtANonObject: frozenset({"projection.exclude_ignores_dead_paths"}),
    _KeepFillsNone: frozenset({"projection.keep_named_fields"}),
    _AcceptsKeepAndExclude: frozenset({"projection.keep_and_exclude_refused"}),
    _GetManyRepeats: frozenset({"projection.get_many_absent_and_repeated"}),
    _PredicatesCompareText: frozenset({"query.predicates_are_anded"}),
    _NoneMatchesOnlyNull: frozenset({"query.none_matches_absent_or_null"}),
    _OrdersAsText: frozenset({"order.by_stored_type"}),
    _MissingSortsHighest: frozenset({"order.missing_sorts_lowest"}),
    _LimitsBeforeOrdering: frozenset({"order.limit_after_ordering"}),
    _NoEtags: frozenset({"etag.found_document_has_one", "etag.stale_write_is_refused"}),
    _IgnoresIfMatch: frozenset(
        {
            "etag.stale_write_is_refused",
            "etag.current_write_lands",
            "etag.gone_document_is_refused",
            "etag.recreated_document_is_new",
            "etag.merge_moves_it",
            "retry.reread_recovers_a_lost_race",
        }
    ),
    _EtagNeverMoves: frozenset({"etag.current_write_lands", "etag.stale_write_is_refused", "etag.merge_moves_it"}),
    _MergeKeepsTheEtag: frozenset({"etag.merge_moves_it"}),
    _EtagSurvivesDelete: frozenset({"etag.gone_document_is_refused"}),
    _EtagIsAVersionFromOne: frozenset({"etag.recreated_document_is_new"}),
    _RefusesUnconditionalOverwrite: frozenset({"etag.unconditional_write_lands"}),
    _MergeReplaces: frozenset({"merge.sets_fields_keeps_rest"}),
    _MergeSetsLocatingFields: frozenset({"merge.refuses_empty_and_locating"}),
    _DeleteAlwaysReportsTrue: frozenset({"delete.removes_and_reports"}),
    _SweepIgnoresDocType: frozenset({"delete.sweep_by_doc_type"}),
}

_CASES = {case.name: case for case in STORE_CONFORMANCE_CASES}


def test_the_fault_table_names_only_real_cases() -> None:
    named = frozenset().union(*_FAULTS.values())
    assert named <= _CASES.keys(), sorted(named - _CASES.keys())


def test_every_case_is_turned_red_by_some_fault() -> None:
    covered = frozenset().union(*_FAULTS.values())
    assert set(_CASES) <= covered, f"cases no fault can fail: {sorted(set(_CASES) - covered)}"


@pytest.mark.parametrize(
    ("fault", "case_name"),
    [(fault, name) for fault, names in _FAULTS.items() for name in sorted(names)],
    ids=lambda value: value.__name__.strip("_") if isinstance(value, type) else value,
)
def test_a_broken_store_fails_the_case_for_the_rule_it_breaks(
    fault: type[InMemoryDocumentStore], case_name: str
) -> None:
    """A broken store raises; a :class:`StoreConflict` escaping a case that expected a write to land counts too."""
    with pytest.raises((StoreConformanceFailure, StoreConflict)):
        _CASES[case_name].run(fault())


# --- the reference store's own properties ----------------------------------------------------------


def test_it_satisfies_the_port() -> None:
    store: DocumentStore = InMemoryDocumentStore()
    assert store.get("absent", _SCOPE) is None


def test_a_document_without_its_scope_or_id_is_refused() -> None:
    store = InMemoryDocumentStore()
    with pytest.raises(KeyError):
        store.upsert({"id": "d1", "doc_type": "eval_run"})
    with pytest.raises(KeyError):
        store.upsert({"scope_id": _SCOPE, "doc_type": "eval_run"})


def test_a_sweep_may_delete_as_it_goes() -> None:
    store = InMemoryDocumentStore()
    for doc_id in ("d1", "d2", "d3"):
        store.upsert(_doc(doc_id))
    for doc_id in store.iter_by_doc_type("eval_run", _SCOPE):
        assert store.delete(doc_id, _SCOPE) is True
    assert store.documents == {}


class _SlowWrites(dict[tuple[str, str], dict[str, Any]]):
    """A documents table whose every write takes long enough for another thread to arrive."""

    def __setitem__(self, key: tuple[str, str], value: dict[str, Any]) -> None:
        time.sleep(0.05)
        super().__setitem__(key, value)


def test_two_threads_holding_one_etag_cannot_both_land() -> None:
    """The compare and the write are one step: of two writers presenting one etag, exactly one lands.

    Without the lock both pass the compare during the other's slow write, both land, and the first
    writer's change is lost with no conflict raised.
    """
    store = InMemoryDocumentStore()
    store.upsert(_doc("d1", v=0))
    _, etag = store.get_with_etag("d1", _SCOPE)
    store.documents = _SlowWrites(store.documents)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def write(value: int) -> None:
        barrier.wait()
        try:
            store.upsert(_doc("d1", v=value), if_match=etag)
        except StoreConflict:
            outcomes.append("conflict")
        else:
            outcomes.append("landed")

    threads = [threading.Thread(target=write, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["conflict", "landed"]
