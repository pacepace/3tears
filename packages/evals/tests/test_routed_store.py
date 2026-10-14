"""``RoutedDocumentStore`` is one port over several stores, each ``doc_type`` kept where its facts say.

The conformance kit runs over a router of two stores, so the router is held to every rule of the port the way an
adapter is. The rest pin what only a router can get wrong: typed calls reach one store and id-only calls ask every
store; one id is held by one store, and a read never picks between two; a store that does not hold an id answers
"not here", never a refusal; a rule that leaves a fact unmapped, names a store that is not there, or keeps a
tombstone apart from what it suppresses is refused at construction; a conditional write reaches the owning store
unchanged; and a changed rule refuses to start until ``relocate`` has moved every document to its new store.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.schema import DOC_TYPE_FACTS, DocTypeFacts, StoreClock, StoreConflict
from threetears.evals.storage import (
    ROUTING_RECORD_DOC_TYPE,
    DocTypeRoute,
    InMemoryDocumentStore,
    MisplacedDocument,
    RouteChanged,
    RoutedDocumentStore,
)
from threetears.evals.testing import STORE_CONFORMANCE_CASES, StoreConformanceCase

from packages.evals.tests.factories import make_analysis, make_campaign, make_eval_run

_SCOPE = "scope-a"

#: The scope these tests' routers keep their placement record in.
_ROUTING = "evals-routing"

#: Evidence in one store, what the engine can remake in another; a host's own documents with the evidence.
_BY_LIFECYCLE = DocTypeRoute(
    by="lifecycle", to={"kept": "evidence", "durable": "evidence", "regenerable": "scratch"}, unregistered="evidence"
)


def _stores() -> dict[str, InMemoryDocumentStore]:
    return {"evidence": InMemoryDocumentStore(), "scratch": InMemoryDocumentStore()}


def _router(
    stores: dict[str, InMemoryDocumentStore] | None = None, route: DocTypeRoute = _BY_LIFECYCLE
) -> tuple[RoutedDocumentStore, dict[str, InMemoryDocumentStore]]:
    """A started router: its stores relocated (empty, so only the record is written), then opened."""
    children = stores if stores is not None else _stores()
    RoutedDocumentStore.relocate(children, route, scope_ids=[], routing_scope=_ROUTING)
    return RoutedDocumentStore(children, route, routing_scope=_ROUTING), children


def _doc(doc_id: str, doc_type: str, **fields: Any) -> dict[str, Any]:
    return {"id": doc_id, "scope_id": _SCOPE, "doc_type": doc_type, **fields}


def _held(store: InMemoryDocumentStore) -> set[str]:
    """The ids a store holds in the test scope, the routing record aside."""
    return {doc_id for scope, doc_id in store.documents if scope == _SCOPE}


# --- the port ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_a_router_of_two_stores_conforms(case: StoreConformanceCase) -> None:
    """The kit's own types are unregistered and reach one store; its engine-type case spreads over both."""
    router, _ = _router()
    case.run(router)


def test_the_engine_type_case_spread_over_both_stores() -> None:
    """The positive control for the conformance run: it did exercise two stores, each holding its lifecycle."""
    router, stores = _router()
    case = next(case for case in STORE_CONFORMANCE_CASES if case.name == "doc_types.every_engine_type_is_stored")
    case.run(router)

    held = {name: {doc_id for scope, doc_id in store.documents if scope != _ROUTING} for name, store in stores.items()}
    assert held["scratch"] == {f"kit-{t}" for t, facts in DOC_TYPE_FACTS.items() if facts.lifecycle == "regenerable"}
    assert held["evidence"] == {
        f"kit-{t}" for t, facts in DOC_TYPE_FACTS.items() if facts.lifecycle in {"kept", "durable"}
    }


def test_the_engine_reads_and_writes_through_a_router() -> None:
    router, stores = _router()
    storage = EvalStorage(router)
    run, campaign, analysis = (
        make_eval_run(scope_id=_SCOPE, status="completed"),
        make_campaign(scope_id=_SCOPE),
        make_analysis(scope_id=_SCOPE),
    )
    storage.save_eval_run(run)
    storage.save_campaign(campaign)
    storage.save_analysis(analysis)

    assert _held(stores["evidence"]) == {run.id, campaign.id}
    assert _held(stores["scratch"]) == {analysis.id}
    assert storage.load_eval_run(run.id, _SCOPE) == run
    assert storage.list_campaigns(_SCOPE) == [campaign]
    assert storage.load_analysis(analysis.id, _SCOPE) == analysis
    assert storage.set_eval_run_archived(run.id, _SCOPE, archived=True) is True
    assert stores["evidence"].get(run.id, _SCOPE)["archived"] is True  # type: ignore[index]


# --- typed calls route, id-only calls fan out ----------------------------------------------------


def test_typed_calls_reach_only_the_store_the_type_is_kept_in() -> None:
    router, stores = _router()
    router.upsert(_doc("a1", "eval_analysis"))
    router.upsert(_doc("r1", "eval_run"))
    stores["evidence"].upsert(_doc("stray", "eval_analysis"))

    assert _held(stores["scratch"]) == {"a1"} and "r1" in _held(stores["evidence"])
    assert [d["id"] for d in router.by_doc_type("eval_analysis", _SCOPE)] == ["a1"], "the evidence store is not asked"
    assert list(router.iter_by_doc_type("eval_analysis", _SCOPE)) == ["a1"]
    assert [d["id"] for d in router.get_many("eval_analysis", ["a1", "stray"], _SCOPE)] == ["a1"]


def test_a_host_type_goes_to_the_store_named_for_unregistered_types() -> None:
    router, stores = _router()
    router.upsert(_doc("n1", "host_note"))

    assert _held(stores["evidence"]) == {"n1"}
    assert router.get("n1", _SCOPE) == _doc("n1", "host_note")


def test_an_id_only_read_finds_the_document_in_whichever_store_holds_it() -> None:
    router, stores = _router()
    router.upsert(_doc("a1", "eval_analysis", v=1))

    assert router.get("a1", _SCOPE) == _doc("a1", "eval_analysis", v=1)
    document, etag = router.get_with_etag("a1", _SCOPE)
    assert (document, etag) == stores["scratch"].get_with_etag("a1", _SCOPE)
    assert router.get("absent", _SCOPE) is None and router.get_with_etag("absent", _SCOPE) == (None, None)


def test_not_here_is_not_a_refusal() -> None:
    """Every store but the holder answers False; the router reports the one that held it, and False for none."""
    router, stores = _router()
    router.upsert(_doc("a1", "eval_analysis", v=1))

    assert router.merge_fields("a1", _SCOPE, {"v": 2}) is True
    assert stores["scratch"].get("a1", _SCOPE) == _doc("a1", "eval_analysis", v=2)
    assert _held(stores["evidence"]) == set(), "merging asked the evidence store and created nothing there"
    assert router.merge_fields("absent", _SCOPE, {"v": 2}) is False
    assert router.delete("a1", _SCOPE) is True
    assert router.delete("a1", _SCOPE) is False


# --- one id, one store -------------------------------------------------------------------------


def _held_twice() -> tuple[RoutedDocumentStore, dict[str, InMemoryDocumentStore]]:
    router, stores = _router()
    router.upsert(_doc("d1", "eval_analysis", v="scratch"))
    stores["evidence"].upsert(_doc("d1", "eval_analysis", v="evidence"))
    return router, stores


@pytest.mark.parametrize("read", ["get", "get_with_etag"])
def test_an_id_held_by_two_stores_fails_loudly_rather_than_picking_one(read: str) -> None:
    router, _ = _held_twice()
    with pytest.raises(MisplacedDocument, match=r"held by \['evidence', 'scratch'\]"):
        getattr(router, read)("d1", _SCOPE)


@pytest.mark.parametrize("write", ["merge_fields", "delete"])
def test_an_id_only_write_reaching_two_stores_fails_loudly(write: str) -> None:
    router, _ = _held_twice()
    args: tuple[Any, ...] = ("d1", _SCOPE, {"v": "x"}) if write == "merge_fields" else ("d1", _SCOPE)
    with pytest.raises(MisplacedDocument, match="reached"):
        getattr(router, write)(*args)


def test_a_document_in_a_store_its_type_is_not_kept_in_is_refused_on_read() -> None:
    """A typed read of its type would never find it there, so an id-only read must not pretend it is in place."""
    router, stores = _router()
    stores["evidence"].upsert(_doc("a1", "eval_analysis"))

    assert router.by_doc_type("eval_analysis", _SCOPE) == []
    with pytest.raises(MisplacedDocument, match="its type is kept in 'scratch'") as refused:
        router.get("a1", _SCOPE)
    assert "relocate" not in refused.value.fault and "RoutedDocumentStore.relocate" in refused.value.remedy


# --- conditional writes ------------------------------------------------------------------------


def test_a_conditional_write_reaches_the_owning_store_with_its_etag_unchanged() -> None:
    router, stores = _router()
    router.upsert(_doc("a1", "eval_analysis", v=1))
    _, etag = router.get_with_etag("a1", _SCOPE)
    assert etag is not None

    router.upsert(_doc("a1", "eval_analysis", v=2), if_match=etag)
    assert stores["scratch"].get("a1", _SCOPE) == _doc("a1", "eval_analysis", v=2)
    with pytest.raises(StoreConflict):
        router.upsert(_doc("a1", "eval_analysis", v=3), if_match=etag)
    with pytest.raises(StoreConflict):
        router.upsert(_doc("r1", "eval_run"), if_match=etag)


def test_a_create_only_write_reaches_the_owning_store() -> None:
    router, stores = _router()
    router.upsert(_doc("a1", "eval_analysis", v=1), create_only=True)

    assert stores["scratch"].get("a1", _SCOPE) == _doc("a1", "eval_analysis", v=1)
    with pytest.raises(StoreConflict):
        router.upsert(_doc("a1", "eval_analysis", v=2), create_only=True)
    assert router.get("a1", _SCOPE) == _doc("a1", "eval_analysis", v=1)


def test_a_create_only_write_is_refused_when_another_store_holds_the_id() -> None:
    """Nothing of any doc_type may be stored under the id, in any store; a StoreConflict, as one store would raise."""
    router, stores = _router()
    router.upsert(_doc("d1", "eval_run", v="run"))

    with pytest.raises(StoreConflict, match=r"already held by \['evidence'\]"):
        router.upsert(_doc("d1", "eval_analysis", v="analysis"), create_only=True)
    assert _held(stores["scratch"]) == set(), "the owning store was never written"
    assert router.get("d1", _SCOPE) == _doc("d1", "eval_run", v="run")


def test_a_document_without_a_doc_type_cannot_be_placed() -> None:
    router, _ = _router()
    with pytest.raises(ValueError, match="by its doc_type"):
        router.upsert({"id": "x", "scope_id": _SCOPE})


def test_the_routing_record_is_invisible_to_typed_reads_of_its_scope() -> None:
    """The record lives under a doc_type no engine call names, so even a read of its scope lists nothing."""
    router, _ = _router()

    assert router.by_doc_type("eval_run", _ROUTING) == []
    assert list(router.iter_by_doc_type("host_note", _ROUTING)) == []


def test_the_record_is_written_under_the_doc_type_the_package_exports() -> None:
    """A host auditing what its tables hold names the router's record by the exported constant, not a copy of it."""
    _, stores = _router()

    for name, store in stores.items():
        assert list(store.iter_by_doc_type(ROUTING_RECORD_DOC_TYPE, _ROUTING)), f"{name} holds no record under it"


# --- the rule ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "refusal"),
    [
        (dict(by="origin", to={"input": "a", "record": "a"}, unregistered="a"), r"missing \['derived'\]"),
        (dict(by="lifecycle", to={"kept": "a", "regenerable": "a"}, unregistered="a"), r"missing \['durable'\]"),
        (
            dict(by="lifecycle", to={"kept": "a", "durable": "a", "regenerable": "a", "cached": "a"}, unregistered="a"),
            "unknown",
        ),
        (dict(by="doc_type", to={}, unregistered="a"), "keys on one of"),
        (dict(by="lifecycle", to={"kept": "a", "durable": "a", "regenerable": ""}, unregistered="a"), "never blank"),
    ],
)
def test_a_rule_that_misses_or_invents_a_value_is_refused(route: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        DocTypeRoute(**route)


def test_a_rule_naming_a_store_that_is_not_there_is_refused() -> None:
    route = DocTypeRoute(
        by="lifecycle", to={"kept": "evidence", "durable": "evidence", "regenerable": "cache"}, unregistered="evidence"
    )
    with pytest.raises(ValueError, match=r"names stores \['cache'\]"):
        RoutedDocumentStore.relocate(_stores(), route, scope_ids=[], routing_scope=_ROUTING)
    with pytest.raises(ValueError, match=r"names stores \['cache'\]"):
        RoutedDocumentStore(_stores(), route, routing_scope=_ROUTING)


def test_one_store_under_two_names_is_refused() -> None:
    store = InMemoryDocumentStore()
    with pytest.raises(ValueError, match="under two names"):
        RoutedDocumentStore.relocate(
            {"evidence": store, "scratch": store}, _BY_LIFECYCLE, scope_ids=[], routing_scope=_ROUTING
        )


def test_the_rule_resolves_to_one_store_per_registered_type() -> None:
    router, _ = _router(
        route=DocTypeRoute(
            by="origin", to={"input": "evidence", "record": "evidence", "derived": "scratch"}, unregistered="evidence"
        )
    )

    assert set(router.placements) == set(DOC_TYPE_FACTS)
    assert {t for t, store in router.placements.items() if store == "scratch"} == {
        t for t, facts in DOC_TYPE_FACTS.items() if facts.origin == "derived"
    }


def test_a_rule_keeping_a_tombstone_apart_from_what_it_suppresses_is_refused() -> None:
    """Structural: the real registry keeps each tombstone's facts its definition's, so a registry that did not
    is what the refusal is for."""
    facts = {
        "rubric_dim": DocTypeFacts(origin="input", lifecycle="kept"),
        "rubric_dim_tombstone": DocTypeFacts(origin="record", lifecycle="kept", suppresses="rubric_dim"),
    }
    route = DocTypeRoute(by="origin", to={"input": "a", "record": "b", "derived": "b"}, unregistered="a")

    with pytest.raises(ValueError, match="rubric_dim_tombstone in 'b' and the rubric_dim it suppresses in 'a'"):
        route.placements(facts)
    assert DocTypeRoute(
        by="lifecycle", to={"kept": "a", "durable": "b", "regenerable": "b"}, unregistered="a"
    ).placements(facts)


def test_a_lifecycle_rule_places_what_people_author_in_a_store_of_its_own() -> None:
    """The durable lifecycle is a value a rule maps like any other: the eight authored types go where it says."""
    stores = {**_stores(), "authored": InMemoryDocumentStore()}
    route = DocTypeRoute(
        by="lifecycle",
        to={"kept": "evidence", "durable": "authored", "regenerable": "scratch"},
        unregistered="evidence",
    )
    router, _ = _router(stores, route)

    assert {t for t, store in router.placements.items() if store == "authored"} == {
        t for t, facts in DOC_TYPE_FACTS.items() if facts.lifecycle == "durable"
    }
    router.upsert(_doc("t1", "eval_template"))
    router.upsert(_doc("r1", "eval_run"))
    assert (_held(stores["authored"]), _held(stores["evidence"])) == ({"t1"}, {"r1"})


def test_every_real_tombstone_is_kept_with_what_it_suppresses_under_every_rule() -> None:
    for by, values in (("origin", ("input", "record", "derived")), ("lifecycle", ("kept", "durable", "regenerable"))):
        for split in values:
            route = DocTypeRoute(by=by, to={v: ("b" if v == split else "a") for v in values}, unregistered="a")  # type: ignore[arg-type]
            placements = route.placements(DOC_TYPE_FACTS)
            assert placements["rubric_dim_tombstone"] == placements["rubric_dim"]
            assert placements["judge_config_tombstone"] == placements["judge_config"]


# --- changing the rule is an operation ------------------------------------------------------------

_BY_ORIGIN = DocTypeRoute(
    by="origin", to={"input": "evidence", "record": "evidence", "derived": "scratch"}, unregistered="evidence"
)
_EVERYTHING_IN_EVIDENCE = DocTypeRoute(
    by="lifecycle", to={"kept": "evidence", "durable": "evidence", "regenerable": "evidence"}, unregistered="evidence"
)


def test_stores_with_no_record_refuse_to_start() -> None:
    """Documents already in an adopted store cannot be told from an empty one through the port."""
    with pytest.raises(RouteChanged, match="records the placements it was written under"):
        RoutedDocumentStore(_stores(), _BY_LIFECYCLE, routing_scope=_ROUTING)


def test_a_changed_rule_refuses_to_start_naming_what_moved() -> None:
    _, stores = _router(route=_EVERYTHING_IN_EVIDENCE)

    with pytest.raises(RouteChanged, match="eval_analysis") as refused:
        RoutedDocumentStore(stores, _BY_LIFECYCLE, routing_scope=_ROUTING)
    # The fault and the remedy apart, so a host naming its own relocation tool replaces only the remedy.
    assert "eval_analysis" in refused.value.fault and "relocate" not in refused.value.fault
    assert "RoutedDocumentStore.relocate" in refused.value.remedy
    assert str(refused.value) == f"{refused.value.fault}; {refused.value.remedy}"


def test_a_different_rule_with_the_same_placements_starts() -> None:
    """The record is of placements, not of how the rule was spelled: nothing moved, so nothing to relocate."""
    _, stores = _router(route=_EVERYTHING_IN_EVIDENCE)
    by_origin = DocTypeRoute(
        by="origin", to={"input": "evidence", "record": "evidence", "derived": "evidence"}, unregistered="evidence"
    )

    RoutedDocumentStore(stores, by_origin, routing_scope=_ROUTING)


def test_a_rule_moving_one_type_refuses_to_start_naming_it() -> None:
    """By origin and by lifecycle differ only in the cassette, the one regenerable record; that one type is named."""
    _, stores = _router(route=_BY_LIFECYCLE)

    with pytest.raises(
        RouteChanged,
        match=r"keeps \['eval_cassette'\] in another store",
    ):
        RoutedDocumentStore(stores, _BY_ORIGIN, routing_scope=_ROUTING)


def test_stores_passed_under_each_other_s_names_refuse_to_start() -> None:
    _, stores = _router()
    swapped = {"evidence": stores["scratch"], "scratch": stores["evidence"]}

    with pytest.raises(RouteChanged, match="passed as 'evidence' was written as 'scratch'"):
        RoutedDocumentStore(swapped, _BY_LIFECYCLE, routing_scope=_ROUTING)


def test_records_that_disagree_refuse_to_start() -> None:
    """An interrupted relocation writes some stores' records and not others'."""
    _, stores = _router(route=_EVERYTHING_IN_EVIDENCE)
    RoutedDocumentStore.relocate(
        {"scratch": stores["scratch"], "evidence": InMemoryDocumentStore()},
        _BY_LIFECYCLE,
        scope_ids=[],
        routing_scope=_ROUTING,
    )

    with pytest.raises(RouteChanged, match="record different placements"):
        RoutedDocumentStore(stores, _BY_LIFECYCLE, routing_scope=_ROUTING)


def test_a_changed_unregistered_store_refuses_to_start() -> None:
    _, stores = _router()
    route = DocTypeRoute(
        by="lifecycle", to={"kept": "evidence", "durable": "evidence", "regenerable": "scratch"}, unregistered="scratch"
    )

    with pytest.raises(RouteChanged, match="unregistered types"):
        RoutedDocumentStore(stores, route, routing_scope=_ROUTING)


def test_a_store_added_with_nothing_placed_in_it_starts_and_is_recorded() -> None:
    _, stores = _router()
    grown = {**stores, "archive": InMemoryDocumentStore()}

    RoutedDocumentStore(grown, _BY_LIFECYCLE, routing_scope=_ROUTING)

    assert grown["archive"].get("placements", _ROUTING) is not None
    RoutedDocumentStore(grown, _BY_LIFECYCLE, routing_scope=_ROUTING)


def test_a_type_new_to_the_record_is_read_where_unregistered_types_were() -> None:
    """A release adding a type starts without a relocation when the type lands where an older build put unknowns."""
    _, stores = _router()
    record = stores["evidence"].get("placements", _ROUTING)
    assert record is not None
    for name, store in stores.items():
        older = {
            **record,
            "child": name,
            "placements": {k: v for k, v in record["placements"].items() if k != "eval_run"},
        }
        older["fingerprint"] = "older"
        store.upsert(older)

    RoutedDocumentStore(stores, _BY_LIFECYCLE, routing_scope=_ROUTING)

    assert "eval_run" in stores["scratch"].get("placements", _ROUTING)["placements"]  # type: ignore[index]


def test_a_type_new_to_the_record_placed_elsewhere_than_unknowns_were_refuses_to_start() -> None:
    _, stores = _router()
    for name, store in stores.items():
        record = store.get("placements", _ROUTING)
        assert record is not None
        store.upsert(
            {
                **record,
                "placements": {k: v for k, v in record["placements"].items() if k != "eval_analysis"},
                "fingerprint": "older",
            }
        )

    with pytest.raises(RouteChanged, match="eval_analysis"):
        RoutedDocumentStore(stores, _BY_LIFECYCLE, routing_scope=_ROUTING)


def test_relocate_moves_every_document_to_its_new_store_and_then_starts() -> None:
    router, stores = _router(route=_EVERYTHING_IN_EVIDENCE)
    router.upsert(_doc("a1", "eval_analysis", v=1))
    router.upsert(_doc("r1", "eval_run"))
    router.upsert(_doc("n1", "host_note"))
    router.upsert({**_doc("a2", "eval_analysis"), "scope_id": "scope-b"})

    moved = RoutedDocumentStore.relocate(
        stores, _BY_LIFECYCLE, scope_ids=[_SCOPE, "scope-b"], host_doc_types=["host_note"], routing_scope=_ROUTING
    )

    assert moved == {"eval_analysis": 2}
    assert _held(stores["scratch"]) == {"a1"} and _held(stores["evidence"]) == {"r1", "n1"}
    started = RoutedDocumentStore(stores, _BY_LIFECYCLE, routing_scope=_ROUTING)
    assert started.get("a1", _SCOPE) == _doc("a1", "eval_analysis", v=1)
    assert [d["id"] for d in started.by_doc_type("eval_analysis", "scope-b")] == ["a2"]


def test_relocate_moves_a_host_type_it_is_told_of() -> None:
    router, stores = _router()
    router.upsert(_doc("n1", "host_note"))
    to_scratch = DocTypeRoute(
        by="lifecycle", to={"kept": "evidence", "durable": "evidence", "regenerable": "scratch"}, unregistered="scratch"
    )

    assert RoutedDocumentStore.relocate(
        stores, to_scratch, scope_ids=[_SCOPE], host_doc_types=["host_note"], routing_scope=_ROUTING
    ) == {"host_note": 1}
    assert _held(stores["scratch"]) == {"n1"}


def test_an_interrupted_relocation_is_finished_by_running_it_again() -> None:
    """A copy that landed before the interruption is found identical where it belongs, and its stray removed."""
    _, stores = _router(route=_EVERYTHING_IN_EVIDENCE)
    stores["evidence"].upsert(_doc("a1", "eval_analysis", v=1))
    stores["scratch"].upsert(_doc("a1", "eval_analysis", v=1))

    assert RoutedDocumentStore.relocate(stores, _BY_LIFECYCLE, scope_ids=[_SCOPE], routing_scope=_ROUTING) == {
        "eval_analysis": 1
    }
    assert _held(stores["evidence"]) == set() and _held(stores["scratch"]) == {"a1"}


def test_empty_stores_hold_no_documents() -> None:
    assert RoutedDocumentStore.holds_documents(_stores(), [_SCOPE, "scope-b"]) is False


def test_stores_holding_only_a_placement_record_hold_no_documents() -> None:
    """The record is the router's bookkeeping in the routing scope, not a document, even when that scope is asked."""
    _, stores = _router()

    assert RoutedDocumentStore.holds_documents(stores, [_SCOPE]) is False
    assert RoutedDocumentStore.holds_documents(stores, [_ROUTING], host_doc_types=[ROUTING_RECORD_DOC_TYPE]) is False


@pytest.mark.parametrize("store", ["evidence", "scratch"])
def test_a_registered_document_in_any_store_is_found(store: str) -> None:
    stores = _stores()
    stores[store].upsert(_doc("a1", "eval_analysis"))

    assert RoutedDocumentStore.holds_documents(stores, ["scope-b", _SCOPE]) is True


def test_a_document_in_a_scope_not_named_is_not_found() -> None:
    stores = _stores()
    stores["evidence"].upsert(_doc("r1", "eval_run"))

    assert RoutedDocumentStore.holds_documents(stores, ["scope-b"]) is False


def test_a_host_type_is_found_only_when_named() -> None:
    stores = _stores()
    stores["evidence"].upsert(_doc("n1", "host_note"))

    assert RoutedDocumentStore.holds_documents(stores, [_SCOPE]) is False
    assert RoutedDocumentStore.holds_documents(stores, [_SCOPE], host_doc_types=["host_note"]) is True


def test_every_registered_type_is_asked() -> None:
    for doc_type in DOC_TYPE_FACTS:
        stores = _stores()
        stores["scratch"].upsert(_doc("d1", doc_type))
        assert RoutedDocumentStore.holds_documents(stores, [_SCOPE]) is True, doc_type


def test_holds_documents_stops_at_the_first_document() -> None:
    asked: list[str] = []

    class _CountingStore(InMemoryDocumentStore):
        def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
            asked.append(doc_type)
            return super().iter_by_doc_type(doc_type, scope_id)

    stores = {"evidence": _CountingStore(), "scratch": _CountingStore()}
    first = sorted(DOC_TYPE_FACTS)[0]
    stores["evidence"].upsert(_doc("d1", first))

    assert RoutedDocumentStore.holds_documents(stores, [_SCOPE, "scope-b"]) is True
    assert asked == [first]


def test_relocate_refuses_two_different_documents_under_one_id() -> None:
    _, stores = _router(route=_EVERYTHING_IN_EVIDENCE)
    stores["evidence"].upsert(_doc("a1", "eval_analysis", v="one"))
    stores["scratch"].upsert(_doc("a1", "eval_analysis", v="two"))

    with pytest.raises(MisplacedDocument, match="the two differ"):
        RoutedDocumentStore.relocate(stores, _BY_LIFECYCLE, scope_ids=[_SCOPE], routing_scope=_ROUTING)
    assert _held(stores["evidence"]) == {"a1"}, "nothing of that id was moved"
    with pytest.raises(RouteChanged):
        RoutedDocumentStore(stores, _BY_LIFECYCLE, routing_scope=_ROUTING)


# --- a store's clock -----------------------------------------------------------------------------


class _ClockedStore(InMemoryDocumentStore):
    """An in-memory store that keeps a clock of its own, fixed so a test can tell whose clock answered."""

    def __init__(self, at: datetime) -> None:
        super().__init__()
        self._at = at

    def now(self) -> datetime:
        return self._at


_EVIDENCE_TIME = datetime(2026, 10, 1, tzinfo=UTC)
_SCRATCH_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def test_the_router_keeps_a_clock_when_every_store_does_and_reads_the_job_records_store() -> None:
    router, _ = _router({"evidence": _ClockedStore(_EVIDENCE_TIME), "scratch": _ClockedStore(_SCRATCH_TIME)})

    assert isinstance(router, StoreClock) and isinstance(router, RoutedDocumentStore)
    assert router.placements["eval_run"] == "evidence"
    assert router.now() == _EVIDENCE_TIME


def test_the_router_keeps_no_clock_when_any_store_has_none() -> None:
    router, _ = _router({"evidence": _ClockedStore(_EVIDENCE_TIME), "scratch": InMemoryDocumentStore()})

    assert not isinstance(router, StoreClock)
    assert not hasattr(router, "now")


def test_every_job_record_type_is_kept_with_runs_under_every_rule() -> None:
    """The clocked router reads the runs' store; every leased record must be stamped by that same clock."""
    for by, values in (("origin", ("input", "record", "derived")), ("lifecycle", ("kept", "durable", "regenerable"))):
        for split in values:
            route = DocTypeRoute(by=by, to={v: ("b" if v == split else "a") for v in values}, unregistered="a")  # type: ignore[arg-type]
            placements = route.placements(DOC_TYPE_FACTS)
            assert placements["eval_sweep"] == placements["eval_analysis_attempt"] == placements["eval_run"]


# --- the host names the routing scope ----------------------------------------------------------------------------


class _RowLevelSecured:
    """One partition's view of a table several partitions share, as row-level security gives it.

    The table's key is global: a partition sees and writes only rows it owns, and a write to a key another
    partition owns is refused as the database's unique constraint would refuse it, though the writer cannot see
    the row. Two environments of one database, each behind its own policy, are two views over one table.
    """

    def __init__(self, table: InMemoryDocumentStore, owners: dict[tuple[str, str], str], partition: str) -> None:
        self._table, self._owners, self._partition = table, owners, partition

    def _mine(self, doc_id: str, scope_id: str) -> bool:
        return self._owners.get((scope_id, doc_id)) == self._partition

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        return self._table.get(doc_id, scope_id) if self._mine(doc_id, scope_id) else None

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        return self._table.get_with_etag(doc_id, scope_id) if self._mine(doc_id, scope_id) else (None, None)

    def get_many(self, doc_type: str, doc_ids: Any, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        found = self._table.get_many(doc_type, doc_ids, scope_id, **kwargs)
        return [d for d in found if self._mine(d["id"], scope_id)]

    def by_doc_type(self, doc_type: str, scope_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [d for d in self._table.by_doc_type(doc_type, scope_id, **kwargs) if self._mine(d["id"], scope_id)]

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Any:
        return (i for i in self._table.iter_by_doc_type(doc_type, scope_id) if self._mine(i, scope_id))

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        key = (document["scope_id"], document["id"])
        owner = self._owners.setdefault(key, self._partition)
        if owner != self._partition:
            raise StoreConflict(f"duplicate key {key}: the row is another partition's")
        self._table.upsert(document, if_match=if_match, create_only=create_only)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        return self._mine(doc_id, scope_id) and self._table.merge_fields(doc_id, scope_id, fields)

    def delete(self, doc_id: str, scope_id: str) -> bool:
        if not self._mine(doc_id, scope_id):
            return False
        del self._owners[(scope_id, doc_id)]
        return self._table.delete(doc_id, scope_id)


def test_two_partitions_of_one_database_each_relocate_and_start_under_their_own_routing_scope() -> None:
    """A shared table with a global key holds one placement record per routing scope, so each partition names its
    own and neither overwrites, nor is refused by, the other's."""
    tables = _stores()
    owners: dict[str, dict[tuple[str, str], str]] = {name: {} for name in tables}  # each table's row owners
    routed = {}
    for env, scope in (("env-a", "acme"), ("env-b", "globex")):
        views = {name: _RowLevelSecured(table, owners[name], env) for name, table in tables.items()}
        views["evidence"].upsert({**_doc(f"{env}-a1", "eval_analysis", v=env), "scope_id": scope})
        views["evidence"].upsert({**_doc(f"{env}-r1", "eval_run"), "scope_id": scope})

        moved = RoutedDocumentStore.relocate(
            views, _BY_LIFECYCLE, scope_ids=[scope], routing_scope=f"evals-routing.{env}"
        )

        assert moved == {"eval_analysis": 1}
        routed[env] = (RoutedDocumentStore(views, _BY_LIFECYCLE, routing_scope=f"evals-routing.{env}"), scope)

    for env, (router, scope) in routed.items():
        assert router.get(f"{env}-a1", scope) == {**_doc(f"{env}-a1", "eval_analysis", v=env), "scope_id": scope}
        assert [d["id"] for d in router.by_doc_type("eval_run", scope)] == [f"{env}-r1"]
    assert {key for key in tables["scratch"].documents if key[1] == "placements"} == {
        ("evals-routing.env-a", "placements"),
        ("evals-routing.env-b", "placements"),
    }


def test_a_router_under_another_routing_scope_finds_no_record_and_refuses_to_start() -> None:
    stores = _stores()
    RoutedDocumentStore.relocate(stores, _BY_LIFECYCLE, scope_ids=[], routing_scope="evals-routing.one")
    with pytest.raises(RouteChanged, match="records the placements"):
        RoutedDocumentStore(stores, _BY_LIFECYCLE, routing_scope="evals-routing.two")


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_routing_scope_is_refused(blank: str) -> None:
    with pytest.raises(ValueError, match="routing scope"):
        RoutedDocumentStore.relocate(_stores(), _BY_LIFECYCLE, scope_ids=[], routing_scope=blank)
    with pytest.raises(ValueError, match="routing scope"):
        RoutedDocumentStore(_stores(), _BY_LIFECYCLE, routing_scope=blank)


def test_relocate_refuses_to_move_the_routing_scope_as_a_document_scope() -> None:
    with pytest.raises(ValueError, match="routing scope"):
        RoutedDocumentStore.relocate(_stores(), _BY_LIFECYCLE, scope_ids=["acme", "r"], routing_scope="r")
