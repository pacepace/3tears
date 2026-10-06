"""The store conformance kit: every rule of the ``DocumentStore`` port, as a case an adapter runs.

An app that brings its own database implements :class:`~threetears.evals.contracts.DocumentStore`
over it, and every eval document then lives or dies by that adapter. The port's rules — scoping,
the strip on read, projection, ordering, optimistic concurrency, delete — are stated in prose on the
port; this module states them again as checks, so an adapter is proved rather than read.

**The kit is plain Python, and the adopter's test runner parametrises it.** Each
:class:`StoreConformanceCase` takes a fresh, empty store and either returns or raises
:class:`StoreConformanceFailure`, whose message names the case and the rule it broke. Under pytest::

    import pytest
    from threetears.evals.testing import STORE_CONFORMANCE_CASES, StoreConformanceCase

    @pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
    def test_my_store_conforms(case: StoreConformanceCase, tmp_path) -> None:
        case.run(MyDocumentStore(tmp_path / "evals.sqlite"))

**Every case is mandatory.** There is no capability flag and no skip: a store that cannot do one of
these is a store the engine cannot run on — in particular, a store without conditional writes turns
every read-modify-write into a blind overwrite. Each case is handed its own store, writes only under
the kit's own scopes, and leaves what it wrote behind; every case but one writes only the kit's own
``doc_type`` values, and that one (``doc_types.every_engine_type_is_stored``) writes one document of
each type the engine writes, so a store that routes documents by ``doc_type`` — a table per type —
fails it for any type it has no route for. **When the engine adds a document type, that case is how an
adopter's store learns of it**: the type joins :data:`~threetears.evals.contracts.storage.EVAL_DOC_TYPES`
and the case goes red until the store routes it.

**What it cannot see.** Concurrency is checked only within one process, and only probabilistically:
``etag.racing_writers_one_lands`` races several threads holding one etag, which catches a conditional
write compared and written in two steps when the threads interleave between them — it can pass by luck,
never fail by it — and says nothing about two processes on one database; make the compare and the write
one statement. Scale, and a backend's own failure modes: a genuine failure must raise rather than read as
a miss, and no in-process check can make a backend fail.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from threetears.evals.contracts.storage import EVAL_DOC_TYPES
from threetears.evals.contracts.store_port import DocumentStore, StoreConflict

__all__ = [
    "STORE_CONFORMANCE_CASES",
    "StoreConformanceCase",
    "StoreConformanceFailure",
]

_SCOPE = "conformance-scope-a"
_OTHER_SCOPE = "conformance-scope-b"
_TYPE = "conformance_doc"
_OTHER_TYPE = "conformance_other"


class StoreConformanceFailure(AssertionError):
    """A store broke a rule of the ``DocumentStore`` port; the message names the case and the rule.

    An :class:`AssertionError`, so a test runner reports it as a failed assertion rather than an
    error in the test.
    """


@dataclass(frozen=True)
class StoreConformanceCase:
    """One rule of the port, as a check over a store.

    Attributes:
        name: A stable identifier, for a test id (``"etag.stale_write_is_refused"``).
        rule: The rule, in one sentence, as the port states it.
        check: The check. Takes a fresh, empty store; returns, or raises
            :class:`StoreConformanceFailure`.
    """

    name: str
    rule: str
    check: Callable[[DocumentStore], None]

    def run(self, store: DocumentStore) -> None:
        """Run the case against ``store``, which must be fresh and empty.

        Args:
            store: The adapter under test.

        Raises:
            StoreConformanceFailure: The store broke the rule; the message names the case, the rule
                and what was observed. Anything else the store raises propagates unchanged.
        """
        try:
            self.check(store)
        except StoreConformanceFailure as failure:
            raise StoreConformanceFailure(f"{self.name}: {self.rule} — {failure}") from None


# --- helpers ----------------------------------------------------------------------------------------


def _doc(doc_id: str, *, scope: str = _SCOPE, doc_type: str = _TYPE, **fields: Any) -> dict[str, Any]:
    return {"id": doc_id, "scope_id": scope, "doc_type": doc_type, **fields}


def _expect(condition: bool, observed: str) -> None:
    if not condition:
        raise StoreConformanceFailure(observed)


def _expect_equal(actual: object, expected: object, what: str) -> None:
    _expect(actual == expected, f"{what}: expected {expected!r}, got {actual!r}")


def _ids(documents: Sequence[dict[str, Any]]) -> list[Any]:
    return [document.get("id") for document in documents]


def _expect_conflict(write: Callable[[], None], what: str) -> None:
    try:
        write()
    except StoreConflict:
        # NOSILENT: the conflict IS the conformance check passing; the failure is the write landing
        return
    raise StoreConformanceFailure(f"{what} landed; expected StoreConflict")


def _expect_value_error(call: Callable[[], object], what: str) -> None:
    try:
        call()
    except ValueError:
        # NOSILENT: the refusal IS the conformance check passing; the failure is the call being accepted
        return
    raise StoreConformanceFailure(f"{what} was accepted; expected ValueError")


def _found(store: DocumentStore, doc_id: str) -> dict[str, Any]:
    document = store.get(doc_id, _SCOPE)
    if document is None:
        raise StoreConformanceFailure(f"get missed {doc_id!r}, which was just written")
    return document


def _etag(store: DocumentStore, doc_id: str, scope: str = _SCOPE) -> str:
    document, etag = store.get_with_etag(doc_id, scope)
    _expect(document is not None, f"get_with_etag missed {doc_id!r}, which was just written")
    if not isinstance(etag, str) or not etag:
        raise StoreConformanceFailure(f"get_with_etag returned etag {etag!r} for a found document")
    return etag


#: A document exercising every JSON shape a stored eval model can carry.
_RICH = _doc(
    "rich",
    text="naïve — ünïcode ✓",
    empty="",
    integer=9007199254740991,
    negative=-3,
    real=0.1,
    flag=True,
    off=False,
    nothing=None,
    items=[1, "two", None, {"three": [3]}],
    no_items=[],
    nested={"a": {"b": {"c": 1, "d": [1, 2]}, "e": "f"}, "g": {}},
)


# --- scoping ----------------------------------------------------------------------------------------


def _miss_is_an_outcome(store: DocumentStore) -> None:
    _expect_equal(store.get("absent", _SCOPE), None, "get of an absent id")
    _expect_equal(store.get_with_etag("absent", _SCOPE), (None, None), "get_with_etag of an absent id")
    _expect_equal(store.get_many(_TYPE, ["absent"], _SCOPE), [], "get_many of an absent id")
    _expect_equal(store.by_doc_type(_TYPE, _SCOPE), [], "by_doc_type over an empty scope")
    _expect_equal(list(store.iter_by_doc_type(_TYPE, _SCOPE)), [], "iter_by_doc_type over an empty scope")
    _expect_equal(store.delete("absent", _SCOPE), False, "delete of an absent id")
    _expect_equal(store.merge_fields("absent", _SCOPE, {"x": 1}), False, "merge_fields on an absent id")
    _expect_equal(store.get("absent", _SCOPE), None, "get after merge_fields on an absent id (it must create nothing)")


def _scope_isolates_every_method(store: DocumentStore) -> None:
    store.upsert(_doc("d1", v=1))
    _expect_equal(store.get("d1", _OTHER_SCOPE), None, "get from another scope")
    _expect_equal(store.get_with_etag("d1", _OTHER_SCOPE), (None, None), "get_with_etag from another scope")
    _expect_equal(store.get_many(_TYPE, ["d1"], _OTHER_SCOPE), [], "get_many from another scope")
    _expect_equal(store.by_doc_type(_TYPE, _OTHER_SCOPE), [], "by_doc_type over another scope")
    _expect_equal(list(store.iter_by_doc_type(_TYPE, _OTHER_SCOPE)), [], "iter_by_doc_type over another scope")
    _expect_equal(store.merge_fields("d1", _OTHER_SCOPE, {"v": 2}), False, "merge_fields from another scope")
    _expect_equal(store.delete("d1", _OTHER_SCOPE), False, "delete from another scope")
    _expect_equal(store.get("d1", _SCOPE), _doc("d1", v=1), "the document after every other scope touched it")


def _one_id_in_two_scopes_is_two_documents(store: DocumentStore) -> None:
    store.upsert(_doc("d1", v="a"))
    store.upsert(_doc("d1", scope=_OTHER_SCOPE, v="b"))
    _expect_equal(store.get("d1", _SCOPE), _doc("d1", v="a"), "scope a's document after scope b wrote the same id")
    _expect_equal(store.get("d1", _OTHER_SCOPE), _doc("d1", scope=_OTHER_SCOPE, v="b"), "scope b's document")
    _expect_equal(store.delete("d1", _OTHER_SCOPE), True, "delete in scope b")
    _expect_equal(store.get("d1", _SCOPE), _doc("d1", v="a"), "scope a's document after scope b's was deleted")


def _doc_type_is_part_of_every_typed_read(store: DocumentStore) -> None:
    store.upsert(_doc("d1", doc_type=_OTHER_TYPE))
    _expect_equal(store.get_many(_TYPE, ["d1"], _SCOPE), [], "get_many under another doc_type")
    _expect_equal(store.by_doc_type(_TYPE, _SCOPE), [], "by_doc_type under another doc_type")
    _expect_equal(list(store.iter_by_doc_type(_TYPE, _SCOPE)), [], "iter_by_doc_type under another doc_type")
    _expect_equal(list(store.iter_by_doc_type(_OTHER_TYPE, _SCOPE)), ["d1"], "iter_by_doc_type under its own doc_type")


def _reads_return_the_document_as_written(store: DocumentStore) -> None:
    store.upsert(dict(_RICH))
    _expect_equal(store.get("rich", _SCOPE), _RICH, "get")
    _expect_equal(store.get_with_etag("rich", _SCOPE)[0], _RICH, "get_with_etag's document")
    _expect_equal(store.get_many(_TYPE, ["rich"], _SCOPE), [_RICH], "get_many")
    _expect_equal(store.by_doc_type(_TYPE, _SCOPE), [_RICH], "by_doc_type")


def _upsert_replaces_the_whole_document(store: DocumentStore) -> None:
    store.upsert(_doc("d1", kept=1, dropped=2))
    store.upsert(_doc("d1", kept=3))
    _expect_equal(store.get("d1", _SCOPE), _doc("d1", kept=3), "the document after a second upsert left a field out")


def _documents_are_copied_in_and_out(store: DocumentStore) -> None:
    written = _doc("d1", nested={"v": 1}, items=[1])
    store.upsert(written)
    written["nested"]["v"] = 99
    written["items"].append(99)
    read = _found(store, "d1")
    _expect_equal(read, _doc("d1", nested={"v": 1}, items=[1]), "the stored document after the writer mutated its dict")
    read["nested"]["v"] = 42
    _expect_equal(
        store.get("d1", _SCOPE),
        _doc("d1", nested={"v": 1}, items=[1]),
        "the stored document after a reader mutated its copy",
    )


# --- projection -------------------------------------------------------------------------------------


def _exclude_drops_one_leaf_and_keeps_its_siblings(store: DocumentStore) -> None:
    store.upsert(_doc("d1", payload={"heavy": "x" * 64, "light": 1, "deep": {"drop": 1, "keep": 2}}, top=True))
    expected = _doc("d1", payload={"light": 1, "deep": {"keep": 2}}, top=True)
    exclude = ["payload.heavy", "payload.deep.drop"]
    _expect_equal(store.by_doc_type(_TYPE, _SCOPE, exclude=exclude), [expected], "by_doc_type with exclude")
    _expect_equal(store.get_many(_TYPE, ["d1"], _SCOPE, exclude=exclude), [expected], "get_many with exclude")
    _expect_equal(
        store.get("d1", _SCOPE),
        _doc("d1", payload={"heavy": "x" * 64, "light": 1, "deep": {"drop": 1, "keep": 2}}, top=True),
        "the stored document after a projected read",
    )


def _exclude_ignores_absent_and_non_object_paths(store: DocumentStore) -> None:
    store.upsert(_doc("d1", scalar=1, listed=[{"a": 1}], payload={"a": 1}))
    exclude = ["missing", "payload.missing", "scalar.inner", "listed.a", "missing.deeper.still"]
    expected = [_doc("d1", scalar=1, listed=[{"a": 1}], payload={"a": 1})]
    _expect_equal(
        store.by_doc_type(_TYPE, _SCOPE, exclude=exclude), expected, "by_doc_type excluding paths that lead nowhere"
    )
    _expect_equal(
        store.get_many(_TYPE, ["d1"], _SCOPE, exclude=exclude), expected, "get_many excluding paths that lead nowhere"
    )


def _keep_returns_only_the_named_fields_present(store: DocumentStore) -> None:
    store.upsert(_doc("d1", a=1, b={"c": 2}, nulled=None, big="x" * 64))
    got = store.get_many(_TYPE, ["d1"], _SCOPE, keep=["id", "a", "b", "nulled", "absent"])
    _expect_equal(got, [{"id": "d1", "a": 1, "b": {"c": 2}, "nulled": None}], "get_many with keep")


def _keep_and_exclude_together_are_refused(store: DocumentStore) -> None:
    store.upsert(_doc("d1", a=1))
    _expect_value_error(
        lambda: store.get_many(_TYPE, ["d1"], _SCOPE, exclude=["a"], keep=["id"]),
        "get_many with both exclude and keep",
    )


def _get_many_skips_absent_ids_and_returns_each_once(store: DocumentStore) -> None:
    store.upsert(_doc("d1"))
    store.upsert(_doc("d2"))
    got = store.get_many(_TYPE, ["d2", "absent", "d1", "d2"], _SCOPE)
    _expect_equal(sorted(_ids(got)), ["d1", "d2"], "ids get_many returned for d2, absent, d1, d2")
    _expect_equal(store.get_many(_TYPE, [], _SCOPE), [], "get_many of no ids")


# --- querying and ordering --------------------------------------------------------------------------


def _predicates_are_anded_equalities(store: DocumentStore) -> None:
    store.upsert(_doc("both", run="r1", model="m1"))
    store.upsert(_doc("run_only", run="r1", model="m2"))
    store.upsert(_doc("model_only", run="r2", model="m1"))
    store.upsert(_doc("text_one", run="r1", model="m1", n="1"))
    store.upsert(_doc("int_one", run="r1", model="m1", n=1))
    store.upsert(_doc("bool_true", run="r1", model="m1", n=True))
    store.upsert(_doc("int_zero", run="r2", model="m2", n=0))
    store.upsert(_doc("bool_false", run="r2", model="m2", n=False))
    _expect_equal(
        sorted(_ids(store.by_doc_type(_TYPE, _SCOPE, run="r1", model="m1"))),
        ["bool_true", "both", "int_one", "text_one"],
        "by_doc_type(run='r1', model='m1')",
    )
    _expect_equal(
        _ids(store.by_doc_type(_TYPE, _SCOPE, n=1)),
        ["int_one"],
        "by_doc_type(n=1) — a number never equals its text, nor a boolean",
    )
    _expect_equal(_ids(store.by_doc_type(_TYPE, _SCOPE, n=True)), ["bool_true"], "by_doc_type(n=True) — true is not 1")
    _expect_equal(_ids(store.by_doc_type(_TYPE, _SCOPE, n=0)), ["int_zero"], "by_doc_type(n=0) — 0 is not false")
    _expect_equal(
        _ids(store.by_doc_type(_TYPE, _SCOPE, n=False)), ["bool_false"], "by_doc_type(n=False) — false is not 0"
    )
    _expect_equal(
        sorted(_ids(store.by_doc_type(_TYPE, _SCOPE, run="r1"))),
        ["bool_true", "both", "int_one", "run_only", "text_one"],
        "by_doc_type(run='r1')",
    )


def _a_none_predicate_matches_absent_or_null(store: DocumentStore) -> None:
    store.upsert(_doc("absent"))
    store.upsert(_doc("null", parent=None))
    store.upsert(_doc("set", parent="p"))
    _expect_equal(
        sorted(_ids(store.by_doc_type(_TYPE, _SCOPE, parent=None))), ["absent", "null"], "by_doc_type(parent=None)"
    )
    _expect_equal(_ids(store.by_doc_type(_TYPE, _SCOPE, parent="p")), ["set"], "by_doc_type(parent='p')")


def _order_by_sorts_by_stored_type_in_both_directions(store: DocumentStore) -> None:
    for doc_id, rank, at in (
        ("two", 2, "2026-01-02T00:00:00Z"),
        ("ten", 10, "2026-01-10T00:00:00Z"),
        ("one", 1, "2025-12-31T23:59:59Z"),
    ):
        store.upsert(_doc(doc_id, rank=rank, at=at))
    _expect_equal(
        _ids(store.by_doc_type(_TYPE, _SCOPE, order_by="rank", descending=False)),
        ["one", "two", "ten"],
        "ascending by a number",
    )
    _expect_equal(
        _ids(store.by_doc_type(_TYPE, _SCOPE, order_by="rank")),
        ["ten", "two", "one"],
        "descending (the default) by a number",
    )
    _expect_equal(
        _ids(store.by_doc_type(_TYPE, _SCOPE, order_by="at", descending=False)),
        ["one", "two", "ten"],
        "ascending by a timestamp",
    )


def _a_missing_order_field_sorts_lowest(store: DocumentStore) -> None:
    store.upsert(_doc("has", rank=1))
    store.upsert(_doc("absent"))
    store.upsert(_doc("null", rank=None))
    ascending = _ids(store.by_doc_type(_TYPE, _SCOPE, order_by="rank", descending=False))
    descending = _ids(store.by_doc_type(_TYPE, _SCOPE, order_by="rank", descending=True))
    _expect(
        ascending[-1] == "has" and sorted(ascending[:2]) == ["absent", "null"],
        f"ascending by rank put {ascending}; the unranked sort first",
    )
    _expect(
        descending[0] == "has" and sorted(descending[1:]) == ["absent", "null"],
        f"descending by rank put {descending}; the unranked sort last",
    )


def _limit_applies_after_ordering(store: DocumentStore) -> None:
    for doc_id, rank in (("b", 2), ("c", 3), ("a", 1), ("d", 4)):
        store.upsert(_doc(doc_id, rank=rank))
    _expect_equal(
        _ids(store.by_doc_type(_TYPE, _SCOPE, order_by="rank", limit=2)), ["d", "c"], "the top two by rank, descending"
    )
    _expect_equal(
        _ids(store.by_doc_type(_TYPE, _SCOPE, order_by="rank", descending=False, limit=1)),
        ["a"],
        "the first by rank, ascending",
    )
    _expect_equal(len(store.by_doc_type(_TYPE, _SCOPE, limit=3)), 3, "the number of documents under limit=3")


# --- optimistic concurrency -------------------------------------------------------------------------


def _a_found_document_carries_an_etag(store: DocumentStore) -> None:
    store.upsert(_doc("d1"))
    _etag(store, "d1")


def _a_stale_etag_is_refused_and_the_winner_kept(store: DocumentStore) -> None:
    store.upsert(_doc("d1", v=1))
    stale = _etag(store, "d1")
    store.upsert(_doc("d1", v=2))
    _expect_conflict(
        lambda: store.upsert(_doc("d1", v=3), if_match=stale), "a write presenting the etag from before another write"
    )
    _expect_equal(store.get("d1", _SCOPE), _doc("d1", v=2), "the stored document after the refused write")


def _the_current_etag_lands_and_moves_on(store: DocumentStore) -> None:
    # The same content again, so an etag derived from the content alone — which would not move — fails.
    store.upsert(_doc("d1", v=1))
    first = _etag(store, "d1")
    store.upsert(_doc("d1", v=1), if_match=first)
    _expect_equal(store.get("d1", _SCOPE), _doc("d1", v=1), "the document after a write presenting the current etag")
    second = _etag(store, "d1")
    _expect(second != first, f"the etag did not move on after a write of the same content (still {first!r})")
    _expect_conflict(lambda: store.upsert(_doc("d1", v=3), if_match=first), "a second write presenting the first etag")


def _a_conditional_write_to_a_gone_document_is_refused(store: DocumentStore) -> None:
    store.upsert(_doc("d1"))
    etag = _etag(store, "d1")
    store.delete("d1", _SCOPE)
    _expect_conflict(lambda: store.upsert(_doc("d1", v=2), if_match=etag), "a conditional write to a deleted document")
    _expect_equal(store.get("d1", _SCOPE), None, "the deleted document after the refused write")
    _expect_conflict(
        lambda: store.upsert(_doc("never", v=1), if_match=etag), "a conditional write to a document never written"
    )
    _expect_equal(store.get("never", _SCOPE), None, "the never-written document after the refused write")


def _a_recreated_document_does_not_honour_an_old_etag(store: DocumentStore) -> None:
    # Re-created with the same content, so an etag derived from the content alone — which would come back — fails.
    store.upsert(_doc("d1", v=1))
    old = _etag(store, "d1")
    store.delete("d1", _SCOPE)
    store.upsert(_doc("d1", v=1))
    _expect(_etag(store, "d1") != old, f"a deleted and re-created document carries its old etag {old!r}")
    _expect_conflict(
        lambda: store.upsert(_doc("d1", v=3), if_match=old), "a write presenting the etag of the deleted predecessor"
    )


def _an_unconditional_write_always_lands(store: DocumentStore) -> None:
    store.upsert(_doc("d1", v=1))
    _etag(store, "d1")
    store.upsert(_doc("d1", v=2))
    store.upsert(_doc("d1", v=3), if_match=None)
    _expect_equal(store.get("d1", _SCOPE), _doc("d1", v=3), "the document after unconditional writes")


def _merge_fields_moves_the_etag(store: DocumentStore) -> None:
    store.upsert(_doc("d1", flag=False))
    before = _etag(store, "d1")
    _expect_equal(store.merge_fields("d1", _SCOPE, {"flag": True}), True, "merge_fields on a stored document")
    _expect(_etag(store, "d1") != before, "merge_fields left the etag where it was")
    _expect_conflict(
        lambda: store.upsert(_doc("d1", flag=False), if_match=before), "a write presenting the etag from before a merge"
    )
    _expect_equal(store.get("d1", _SCOPE), _doc("d1", flag=True), "the document after the refused write")


def _a_lost_race_is_recovered_by_rereading(store: DocumentStore) -> None:
    store.upsert(_doc("d1", counter=0, tags=[]))
    mine = _etag(store, "d1")
    theirs = _etag(store, "d1")
    store.upsert({**_found(store, "d1"), "tags": ["theirs"]}, if_match=theirs)
    _expect_conflict(lambda: store.upsert(_doc("d1", counter=1, tags=[]), if_match=mine), "the loser's first attempt")
    current, fresh = _found(store, "d1"), _etag(store, "d1")
    _expect_equal(current, _doc("d1", counter=0, tags=["theirs"]), "the re-read after the lost race")
    store.upsert({**current, "counter": current["counter"] + 1}, if_match=fresh)
    _expect_equal(
        store.get("d1", _SCOPE), _doc("d1", counter=1, tags=["theirs"]), "the document after the re-applied write"
    )


#: How many writers race in :func:`_one_of_many_racing_writers_lands`, and how many times. A store whose
#: compare and write are two steps loses the race only when two threads interleave between them, so the
#: case gives it many chances; a pass is evidence, a failure is proof.
_RACING_WRITERS = 8
_RACE_ROUNDS = 25


def _one_of_many_racing_writers_lands(store: DocumentStore) -> None:
    # The engine calls its store from a blocking-I/O executor's threads, so a store is driven from several
    # threads at once in production; this one drives it from several in one process.
    store.upsert(_doc("d1", v=-1))
    for round_number in range(_RACE_ROUNDS):
        etag = _etag(store, "d1")
        barrier = threading.Barrier(_RACING_WRITERS)
        landed: list[int] = []
        refused: list[int] = []
        failed: list[BaseException] = []

        def write(writer: int, held: str = etag) -> None:
            barrier.wait()
            try:
                store.upsert(_doc("d1", v=writer, round=round_number), if_match=held)
            except StoreConflict:
                # NOSILENT: a refusal is the outcome every losing writer must get; it is counted below
                refused.append(writer)
            except Exception as error:
                # NOSILENT: carried to the checking thread, which re-raises it as itself
                failed.append(error)
            else:
                landed.append(writer)

        threads = [threading.Thread(target=write, args=(writer,)) for writer in range(_RACING_WRITERS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failed:
            raise failed[0]
        _expect(
            len(landed) == 1,
            f"round {round_number}: {len(landed)} of {_RACING_WRITERS} writers holding one etag landed "
            f"({sorted(landed)}); the compare and the write must be one step",
        )
        _expect_equal(len(refused), _RACING_WRITERS - 1, f"round {round_number}: writers refused with StoreConflict")
        _expect_equal(
            store.get("d1", _SCOPE),
            _doc("d1", v=landed[0], round=round_number),
            f"round {round_number}: the stored document after the race",
        )


# --- merge_fields -----------------------------------------------------------------------------------


def _merge_fields_sets_fields_and_keeps_the_rest(store: DocumentStore) -> None:
    store.upsert(_doc("d1", flag=False, other={"kept": [1]}, gone=None))
    _expect_equal(store.merge_fields("d1", _SCOPE, {"flag": True, "added": {"x": 1}}), True, "merge_fields")
    _expect_equal(
        store.get("d1", _SCOPE),
        _doc("d1", flag=True, other={"kept": [1]}, gone=None, added={"x": 1}),
        "the document after merge_fields",
    )


def _merge_fields_refuses_empty_and_locating_fields(store: DocumentStore) -> None:
    store.upsert(_doc("d1", v=1))
    _expect_value_error(lambda: store.merge_fields("d1", _SCOPE, {}), "merge_fields with no fields")
    for field, value in (("id", "d2"), ("scope_id", _OTHER_SCOPE), ("doc_type", _OTHER_TYPE)):
        _expect_value_error(
            partial(store.merge_fields, "d1", _SCOPE, {field: value, "v": 2}), f"merge_fields setting {field!r}"
        )
    _expect_equal(store.get("d1", _SCOPE), _doc("d1", v=1), "the document after every refused merge")


# --- delete -----------------------------------------------------------------------------------------


def _delete_removes_the_document_and_reports_it(store: DocumentStore) -> None:
    store.upsert(_doc("d1"))
    store.upsert(_doc("d2"))
    _expect_equal(store.delete("d1", _SCOPE), True, "the first delete of a stored document")
    _expect_equal(store.delete("d1", _SCOPE), False, "a second delete of the same document")
    _expect_equal(store.get("d1", _SCOPE), None, "get after delete")
    _expect_equal(store.get_with_etag("d1", _SCOPE), (None, None), "get_with_etag after delete")
    _expect_equal(store.get_many(_TYPE, ["d1"], _SCOPE), [], "get_many after delete")
    _expect_equal(_ids(store.by_doc_type(_TYPE, _SCOPE)), ["d2"], "by_doc_type after delete")
    _expect_equal(list(store.iter_by_doc_type(_TYPE, _SCOPE)), ["d2"], "iter_by_doc_type after delete")


def _iter_by_doc_type_yields_each_id_once_for_delete(store: DocumentStore) -> None:
    for doc_id in ("a", "b", "c"):
        store.upsert(_doc(doc_id))
    store.upsert(_doc("kept", doc_type=_OTHER_TYPE))
    ids = list(store.iter_by_doc_type(_TYPE, _SCOPE))
    _expect_equal(sorted(ids), ["a", "b", "c"], "iter_by_doc_type's ids")
    _expect_equal([store.delete(doc_id, _SCOPE) for doc_id in ids], [True, True, True], "deleting each yielded id")
    _expect_equal(store.by_doc_type(_TYPE, _SCOPE), [], "the swept type after the sweep")
    _expect_equal(_ids(store.by_doc_type(_OTHER_TYPE, _SCOPE)), ["kept"], "the other type after the sweep")


def _every_engine_doc_type_is_stored(store: DocumentStore) -> None:
    for doc_type in EVAL_DOC_TYPES:
        store.upsert(_doc(f"kit-{doc_type}", doc_type=doc_type, marker=doc_type))
    for doc_type in EVAL_DOC_TYPES:
        doc_id = f"kit-{doc_type}"
        expected = _doc(doc_id, doc_type=doc_type, marker=doc_type)
        _expect_equal(store.get(doc_id, _SCOPE), expected, f"get of a {doc_type!r} document")
        _expect_equal(store.get_many(doc_type, [doc_id], _SCOPE), [expected], f"get_many of a {doc_type!r} document")
        _expect_equal(store.by_doc_type(doc_type, _SCOPE), [expected], f"by_doc_type({doc_type!r})")
        _expect_equal(list(store.iter_by_doc_type(doc_type, _SCOPE)), [doc_id], f"iter_by_doc_type({doc_type!r})")


#: Every case, in the order the port states its rules: scoping, the strip on read, projection,
#: querying, optimistic concurrency (with the re-read that recovers a lost race), merge, delete —
#: and last, that every ``doc_type`` the engine writes is one the store stores.
STORE_CONFORMANCE_CASES: tuple[StoreConformanceCase, ...] = (
    StoreConformanceCase(
        "scope.miss_is_an_outcome",
        "a miss returns None, empty or False and never raises, and creates nothing",
        _miss_is_an_outcome,
    ),
    StoreConformanceCase(
        "scope.isolates_every_method",
        "every method reads, writes and deletes only in the scope it names",
        _scope_isolates_every_method,
    ),
    StoreConformanceCase(
        "scope.one_id_two_scopes",
        "one id in two scopes is two independent documents",
        _one_id_in_two_scopes_is_two_documents,
    ),
    StoreConformanceCase(
        "scope.doc_type_is_part_of_typed_reads",
        "a typed read returns only documents of its doc_type",
        _doc_type_is_part_of_every_typed_read,
    ),
    StoreConformanceCase(
        "read.document_as_written",
        "every read returns the document exactly as written, with nothing the store injected",
        _reads_return_the_document_as_written,
    ),
    StoreConformanceCase(
        "write.upsert_replaces",
        "an upsert replaces the whole document, so a field it leaves out is gone",
        _upsert_replaces_the_whole_document,
    ),
    StoreConformanceCase(
        "write.copied_in_and_out",
        "mutating a written or read dict never reaches the stored document",
        _documents_are_copied_in_and_out,
    ),
    StoreConformanceCase(
        "projection.exclude_drops_one_leaf",
        "exclude removes each dotted path's leaf and keeps its siblings",
        _exclude_drops_one_leaf_and_keeps_its_siblings,
    ),
    StoreConformanceCase(
        "projection.exclude_ignores_dead_paths",
        "an exclude path that is absent or runs through a non-object is a no-op",
        _exclude_ignores_absent_and_non_object_paths,
    ),
    StoreConformanceCase(
        "projection.keep_named_fields",
        "keep returns only the named top-level fields the document has",
        _keep_returns_only_the_named_fields_present,
    ),
    StoreConformanceCase(
        "projection.keep_and_exclude_refused",
        "get_many refuses exclude and keep together with ValueError",
        _keep_and_exclude_together_are_refused,
    ),
    StoreConformanceCase(
        "projection.get_many_absent_and_repeated",
        "get_many skips absent ids and returns each document once",
        _get_many_skips_absent_ids_and_returns_each_once,
    ),
    StoreConformanceCase(
        "query.predicates_are_anded",
        "by_doc_type's predicates are ANDed equalities on the stored type",
        _predicates_are_anded_equalities,
    ),
    StoreConformanceCase(
        "query.none_matches_absent_or_null",
        "a None predicate matches a field that is absent or null",
        _a_none_predicate_matches_absent_or_null,
    ),
    StoreConformanceCase(
        "order.by_stored_type",
        "order_by sorts by the stored type, in either direction, descending by default",
        _order_by_sorts_by_stored_type_in_both_directions,
    ),
    StoreConformanceCase(
        "order.missing_sorts_lowest",
        "a document without the order field, or with it null, sorts below every one that has it",
        _a_missing_order_field_sorts_lowest,
    ),
    StoreConformanceCase(
        "order.limit_after_ordering",
        "limit takes the first documents of the ordered result",
        _limit_applies_after_ordering,
    ),
    StoreConformanceCase(
        "etag.found_document_has_one",
        "a found document always comes back with an etag",
        _a_found_document_carries_an_etag,
    ),
    StoreConformanceCase(
        "etag.stale_write_is_refused",
        "a write presenting a stale etag raises StoreConflict and the winner's document stands",
        _a_stale_etag_is_refused_and_the_winner_kept,
    ),
    StoreConformanceCase(
        "etag.current_write_lands",
        "a write presenting the current etag lands and mints a new one",
        _the_current_etag_lands_and_moves_on,
    ),
    StoreConformanceCase(
        "etag.gone_document_is_refused",
        "a conditional write to a deleted or never-written document raises StoreConflict",
        _a_conditional_write_to_a_gone_document_is_refused,
    ),
    StoreConformanceCase(
        "etag.recreated_document_is_new",
        "a document deleted and written again does not honour its predecessor's etag",
        _a_recreated_document_does_not_honour_an_old_etag,
    ),
    StoreConformanceCase(
        "etag.unconditional_write_lands",
        "a write with if_match=None always lands",
        _an_unconditional_write_always_lands,
    ),
    StoreConformanceCase(
        "etag.merge_moves_it",
        "merge_fields mints a new etag, so a writer holding the old one is refused",
        _merge_fields_moves_the_etag,
    ),
    StoreConformanceCase(
        "etag.racing_writers_one_lands",
        "of writers on several threads presenting one etag at once, exactly one lands and every other raises "
        "StoreConflict — probabilistic: a two-step compare-then-write can pass by luck, never fail by it",
        _one_of_many_racing_writers_lands,
    ),
    StoreConformanceCase(
        "retry.reread_recovers_a_lost_race",
        "after a StoreConflict, re-reading and re-applying lands on top of the winner's write",
        _a_lost_race_is_recovered_by_rereading,
    ),
    StoreConformanceCase(
        "merge.sets_fields_keeps_rest",
        "merge_fields sets the named top-level fields and leaves every other field as stored",
        _merge_fields_sets_fields_and_keeps_the_rest,
    ),
    StoreConformanceCase(
        "merge.refuses_empty_and_locating",
        "merge_fields refuses no fields, and id, scope_id or doc_type, with ValueError",
        _merge_fields_refuses_empty_and_locating_fields,
    ),
    StoreConformanceCase(
        "delete.removes_and_reports",
        "delete removes the document from every read and reports True once, then False",
        _delete_removes_the_document_and_reports_it,
    ),
    StoreConformanceCase(
        "delete.sweep_by_doc_type",
        "iter_by_doc_type yields each id of its type once, and deleting each sweeps the type",
        _iter_by_doc_type_yields_each_id_once_for_delete,
    ),
    StoreConformanceCase(
        "doc_types.every_engine_type_is_stored",
        "every doc_type the engine writes (EVAL_DOC_TYPES) is stored and read back — an adapter that routes "
        "documents by doc_type routes each of them, the out-of-run spend ledger (eval_out_of_run_spend) included",
        _every_engine_doc_type_is_stored,
    ),
)
