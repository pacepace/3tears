"""The shipped in-memory reference store keeps every rule the ``DocumentStore`` port states."""

from __future__ import annotations

import pytest

from threetears.evals.contracts import DocumentStore, InMemoryDocumentStore, StoreConflict

_SCOPE = "scope-a"
_OTHER = "scope-b"


def _doc(doc_id: str, *, scope: str = _SCOPE, doc_type: str = "eval_run", **fields: object) -> dict[str, object]:
    return {"id": doc_id, "scope_id": scope, "doc_type": doc_type, **fields}


def test_it_satisfies_the_port() -> None:
    store: DocumentStore = InMemoryDocumentStore()
    assert store.get("absent", _SCOPE) is None


def test_a_read_in_another_scope_misses() -> None:
    store = InMemoryDocumentStore()
    store.upsert(_doc("d1"))
    assert store.get("d1", _OTHER) is None
    assert store.get_with_etag("d1", _OTHER) == (None, None)
    assert store.get_many("eval_run", ["d1"], _OTHER) == []
    assert store.by_doc_type("eval_run", _OTHER) == []
    assert list(store.iter_by_doc_type("eval_run", _OTHER)) == []
    assert store.delete("d1", _OTHER) is False
    assert store.get("d1", _SCOPE) == _doc("d1")


def test_doc_type_is_part_of_every_typed_read() -> None:
    store = InMemoryDocumentStore()
    store.upsert(_doc("d1", doc_type="eval_result"))
    assert store.get_many("eval_run", ["d1"], _SCOPE) == []
    assert store.by_doc_type("eval_run", _SCOPE) == []
    assert list(store.iter_by_doc_type("eval_result", _SCOPE)) == ["d1"]


def test_a_conditional_write_with_a_stale_etag_is_refused_and_the_winner_kept() -> None:
    store = InMemoryDocumentStore()
    store.upsert(_doc("d1", v=1))
    _, etag = store.get_with_etag("d1", _SCOPE)
    assert etag is not None
    store.upsert(_doc("d1", v=2))  # another writer wins the race
    with pytest.raises(StoreConflict):
        store.upsert(_doc("d1", v=3), if_match=etag)
    assert store.get("d1", _SCOPE) == _doc("d1", v=2)


def test_a_conditional_write_with_the_current_etag_lands_and_moves_it_on() -> None:
    store = InMemoryDocumentStore()
    store.upsert(_doc("d1", v=1))
    _, etag = store.get_with_etag("d1", _SCOPE)
    store.upsert(_doc("d1", v=2), if_match=etag)
    _, after = store.get_with_etag("d1", _SCOPE)
    assert after is not None and after != etag
    with pytest.raises(StoreConflict):
        store.upsert(_doc("d1", v=3), if_match=etag)


def test_a_conditional_write_to_a_deleted_document_is_refused() -> None:
    store = InMemoryDocumentStore()
    store.upsert(_doc("d1"))
    _, etag = store.get_with_etag("d1", _SCOPE)
    store.delete("d1", _SCOPE)
    with pytest.raises(StoreConflict):
        store.upsert(_doc("d1"), if_match=etag)


def test_merge_fields_moves_the_etag_so_a_conditional_writer_rereads() -> None:
    store = InMemoryDocumentStore()
    store.upsert(_doc("d1", flag=False, other="kept"))
    _, etag = store.get_with_etag("d1", _SCOPE)
    assert store.merge_fields("d1", _SCOPE, {"flag": True}) is True
    assert store.get("d1", _SCOPE) == _doc("d1", flag=True, other="kept")
    with pytest.raises(StoreConflict):
        store.upsert(_doc("d1", flag=False), if_match=etag)


def test_merge_fields_reports_a_miss_and_refuses_what_it_may_not_set() -> None:
    store = InMemoryDocumentStore()
    assert store.merge_fields("absent", _SCOPE, {"flag": True}) is False
    store.upsert(_doc("d1"))
    with pytest.raises(ValueError, match="at least one"):
        store.merge_fields("d1", _SCOPE, {})
    for locating in ("id", "scope_id", "doc_type"):
        with pytest.raises(ValueError, match="locate"):
            store.merge_fields("d1", _SCOPE, {locating: "moved"})


def test_a_document_without_its_scope_or_id_is_refused() -> None:
    store = InMemoryDocumentStore()
    with pytest.raises(KeyError):
        store.upsert({"id": "d1", "doc_type": "eval_run"})
    with pytest.raises(KeyError):
        store.upsert({"scope_id": _SCOPE, "doc_type": "eval_run"})


def test_projections_follow_the_port_helpers_and_never_combine() -> None:
    store = InMemoryDocumentStore()
    store.upsert(_doc("d1", heavy={"blob": "x", "keep": 1}, small=2))
    assert store.get_many("eval_run", ["d1"], _SCOPE, exclude=["heavy.blob"]) == [
        _doc("d1", heavy={"keep": 1}, small=2)
    ]
    assert store.get_many("eval_run", ["d1"], _SCOPE, keep=["id", "small", "absent"]) == [{"id": "d1", "small": 2}]
    assert store.by_doc_type("eval_run", _SCOPE, exclude=["heavy"]) == [_doc("d1", small=2)]
    with pytest.raises(ValueError, match="never both"):
        store.get_many("eval_run", ["d1"], _SCOPE, exclude=["heavy"], keep=["id"])


def test_by_doc_type_filters_orders_and_limits() -> None:
    store = InMemoryDocumentStore()
    store.upsert(_doc("a", at="2026-01-02", status="done"))
    store.upsert(_doc("b", at="2026-01-03", status="done"))
    store.upsert(_doc("c", status="done"))
    store.upsert(_doc("d", at="2026-01-01", status="running", note=None))

    assert [d["id"] for d in store.by_doc_type("eval_run", _SCOPE, order_by="at")] == ["b", "a", "d", "c"]
    assert [d["id"] for d in store.by_doc_type("eval_run", _SCOPE, order_by="at", descending=False)] == [
        "c",
        "d",
        "a",
        "b",
    ]
    assert [d["id"] for d in store.by_doc_type("eval_run", _SCOPE, order_by="at", limit=1)] == ["b"]
    assert {d["id"] for d in store.by_doc_type("eval_run", _SCOPE, status="done")} == {"a", "b", "c"}
    # A None predicate matches a field that is absent or null.
    assert {d["id"] for d in store.by_doc_type("eval_run", _SCOPE, note=None)} == {"a", "b", "c", "d"}


def test_what_a_caller_holds_never_reaches_the_stored_copy() -> None:
    store = InMemoryDocumentStore()
    written = _doc("d1", nested={"v": 1})
    store.upsert(written)
    written["nested"]["v"] = 2  # type: ignore[index]
    read = store.get("d1", _SCOPE)
    assert read is not None
    read["nested"]["v"] = 3
    assert store.get("d1", _SCOPE) == _doc("d1", nested={"v": 1})


def test_a_sweep_may_delete_as_it_goes() -> None:
    store = InMemoryDocumentStore()
    for doc_id in ("d1", "d2", "d3"):
        store.upsert(_doc(doc_id))
    for doc_id in store.iter_by_doc_type("eval_run", _SCOPE):
        assert store.delete(doc_id, _SCOPE) is True
    assert store.documents == {}
