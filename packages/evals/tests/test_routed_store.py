"""``RoutedDocumentStore`` is one port over several stores, each ``doc_type`` kept where its facts say.

The conformance kit runs over a router of two stores, so the router is held to every rule of the port the way an
adapter is. The rest pin what only a router can get wrong: typed calls reach one store and id-only calls ask every
store; one id is held by one store, and a read never picks between two; a store that does not hold an id answers
"not here", never a refusal; a rule that leaves a fact unmapped, names a store that is not there, or keeps a
tombstone apart from what it suppresses is refused at construction; a conditional write reaches the owning store
unchanged; and a changed rule refuses to start until ``relocate`` has moved every document to its new store.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.schema import DOC_TYPE_FACTS, DocTypeFacts, StoreClock, StoreConflict
from threetears.evals.storage import (
    ROUTING_SCOPE,
    DocTypeRoute,
    InMemoryDocumentStore,
    MisplacedDocument,
    RouteChanged,
    RoutedDocumentStore,
)
from threetears.evals.testing import STORE_CONFORMANCE_CASES, StoreConformanceCase

from packages.evals.tests.factories import make_analysis, make_campaign, make_eval_run

_SCOPE = "scope-a"

#: Evidence in one store, what the engine can remake in another; a host's own documents with the evidence.
_BY_LIFECYCLE = DocTypeRoute(by="lifecycle", to={"kept": "evidence", "regenerable": "scratch"}, unregistered="evidence")


def _stores() -> dict[str, InMemoryDocumentStore]:
    return {"evidence": InMemoryDocumentStore(), "scratch": InMemoryDocumentStore()}


def _router(
    stores: dict[str, InMemoryDocumentStore] | None = None, route: DocTypeRoute = _BY_LIFECYCLE
) -> tuple[RoutedDocumentStore, dict[str, InMemoryDocumentStore]]:
    """A started router: its stores relocated (empty, so only the record is written), then opened."""
    children = stores if stores is not None else _stores()
    RoutedDocumentStore.relocate(children, route, scope_ids=[])
    return RoutedDocumentStore(children, route), children


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

    held = {
        name: {doc_id for scope, doc_id in store.documents if scope != ROUTING_SCOPE} for name, store in stores.items()
    }
    assert held["scratch"] == {f"kit-{t}" for t, facts in DOC_TYPE_FACTS.items() if facts.lifecycle == "regenerable"}
    assert held["evidence"] == {f"kit-{t}" for t, facts in DOC_TYPE_FACTS.items() if facts.lifecycle == "kept"}


def test_the_engine_reads_and_writes_through_a_router() -> None:
    router, stores = _router()
    storage = EvalStorage(router)
    run, campaign, analysis = (
        make_eval_run(scope_id=_SCOPE),
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
    with pytest.raises(MisplacedDocument, match="its type is kept in 'scratch'"):
        router.get("a1", _SCOPE)


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


def test_a_document_without_a_doc_type_cannot_be_placed() -> None:
    router, _ = _router()
    with pytest.raises(ValueError, match="by its doc_type"):
        router.upsert({"id": "x", "scope_id": _SCOPE})


def test_the_routing_record_is_invisible_to_typed_reads_of_its_scope() -> None:
    """The record lives under a doc_type no engine call names, so even a read of its scope lists nothing."""
    router, _ = _router()

    assert router.by_doc_type("eval_run", ROUTING_SCOPE) == []
    assert list(router.iter_by_doc_type("host_note", ROUTING_SCOPE)) == []


# --- the rule ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "refusal"),
    [
        (dict(by="origin", to={"input": "a", "record": "a"}, unregistered="a"), r"missing \['derived'\]"),
        (dict(by="lifecycle", to={"kept": "a", "regenerable": "a", "cached": "a"}, unregistered="a"), "unknown"),
        (dict(by="doc_type", to={}, unregistered="a"), "keys on one of"),
        (dict(by="lifecycle", to={"kept": "a", "regenerable": ""}, unregistered="a"), "never blank"),
    ],
)
def test_a_rule_that_misses_or_invents_a_value_is_refused(route: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        DocTypeRoute(**route)


def test_a_rule_naming_a_store_that_is_not_there_is_refused() -> None:
    route = DocTypeRoute(by="lifecycle", to={"kept": "evidence", "regenerable": "cache"}, unregistered="evidence")
    with pytest.raises(ValueError, match=r"names stores \['cache'\]"):
        RoutedDocumentStore.relocate(_stores(), route, scope_ids=[])
    with pytest.raises(ValueError, match=r"names stores \['cache'\]"):
        RoutedDocumentStore(_stores(), route)


def test_one_store_under_two_names_is_refused() -> None:
    store = InMemoryDocumentStore()
    with pytest.raises(ValueError, match="under two names"):
        RoutedDocumentStore.relocate({"evidence": store, "scratch": store}, _BY_LIFECYCLE, scope_ids=[])


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
    assert DocTypeRoute(by="lifecycle", to={"kept": "a", "regenerable": "b"}, unregistered="a").placements(facts)


def test_every_real_tombstone_is_kept_with_what_it_suppresses_under_every_rule() -> None:
    for by, values in (("origin", ("input", "record", "derived")), ("lifecycle", ("kept", "regenerable"))):
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
    by="lifecycle", to={"kept": "evidence", "regenerable": "evidence"}, unregistered="evidence"
)


def test_stores_with_no_record_refuse_to_start() -> None:
    """Documents already in an adopted store cannot be told from an empty one through the port."""
    with pytest.raises(RouteChanged, match="records the placements it was written under"):
        RoutedDocumentStore(_stores(), _BY_LIFECYCLE)


def test_a_changed_rule_refuses_to_start_naming_what_moved() -> None:
    _, stores = _router(route=_EVERYTHING_IN_EVIDENCE)

    with pytest.raises(RouteChanged, match="eval_analysis") as refused:
        RoutedDocumentStore(stores, _BY_LIFECYCLE)
    assert "relocate" in str(refused.value)


def test_a_different_rule_with_the_same_placements_starts() -> None:
    """The record is of placements, not of how the rule was spelled: nothing moved, so nothing to relocate."""
    _, stores = _router(route=_EVERYTHING_IN_EVIDENCE)
    by_origin = DocTypeRoute(
        by="origin", to={"input": "evidence", "record": "evidence", "derived": "evidence"}, unregistered="evidence"
    )

    RoutedDocumentStore(stores, by_origin)


def test_a_rule_moving_some_types_refuses_to_start_naming_them() -> None:
    """By origin and by lifecycle differ only in the regenerable records, the cassette and the generation slot."""
    _, stores = _router(route=_BY_LIFECYCLE)

    with pytest.raises(RouteChanged, match=r"keeps \['eval_cassette', 'eval_generation_slot'\] in another store"):
        RoutedDocumentStore(stores, _BY_ORIGIN)


def test_stores_passed_under_each_other_s_names_refuse_to_start() -> None:
    _, stores = _router()
    swapped = {"evidence": stores["scratch"], "scratch": stores["evidence"]}

    with pytest.raises(RouteChanged, match="passed as 'evidence' was written as 'scratch'"):
        RoutedDocumentStore(swapped, _BY_LIFECYCLE)


def test_records_that_disagree_refuse_to_start() -> None:
    """An interrupted relocation writes some stores' records and not others'."""
    _, stores = _router(route=_EVERYTHING_IN_EVIDENCE)
    RoutedDocumentStore.relocate(
        {"scratch": stores["scratch"], "evidence": InMemoryDocumentStore()}, _BY_LIFECYCLE, scope_ids=[]
    )

    with pytest.raises(RouteChanged, match="record different placements"):
        RoutedDocumentStore(stores, _BY_LIFECYCLE)


def test_a_changed_unregistered_store_refuses_to_start() -> None:
    _, stores = _router()
    route = DocTypeRoute(by="lifecycle", to={"kept": "evidence", "regenerable": "scratch"}, unregistered="scratch")

    with pytest.raises(RouteChanged, match="unregistered types"):
        RoutedDocumentStore(stores, route)


def test_a_store_added_with_nothing_placed_in_it_starts_and_is_recorded() -> None:
    _, stores = _router()
    grown = {**stores, "archive": InMemoryDocumentStore()}

    RoutedDocumentStore(grown, _BY_LIFECYCLE)

    assert grown["archive"].get("placements", ROUTING_SCOPE) is not None
    RoutedDocumentStore(grown, _BY_LIFECYCLE)


def test_a_type_new_to_the_record_is_read_where_unregistered_types_were() -> None:
    """A release adding a type starts without a relocation when the type lands where an older build put unknowns."""
    _, stores = _router()
    record = stores["evidence"].get("placements", ROUTING_SCOPE)
    assert record is not None
    for name, store in stores.items():
        older = {
            **record,
            "child": name,
            "placements": {k: v for k, v in record["placements"].items() if k != "eval_run"},
        }
        older["fingerprint"] = "older"
        store.upsert(older)

    RoutedDocumentStore(stores, _BY_LIFECYCLE)

    assert "eval_run" in stores["scratch"].get("placements", ROUTING_SCOPE)["placements"]  # type: ignore[index]


def test_a_type_new_to_the_record_placed_elsewhere_than_unknowns_were_refuses_to_start() -> None:
    _, stores = _router()
    for name, store in stores.items():
        record = store.get("placements", ROUTING_SCOPE)
        assert record is not None
        store.upsert(
            {
                **record,
                "placements": {k: v for k, v in record["placements"].items() if k != "eval_analysis"},
                "fingerprint": "older",
            }
        )

    with pytest.raises(RouteChanged, match="eval_analysis"):
        RoutedDocumentStore(stores, _BY_LIFECYCLE)


def test_relocate_moves_every_document_to_its_new_store_and_then_starts() -> None:
    router, stores = _router(route=_EVERYTHING_IN_EVIDENCE)
    router.upsert(_doc("a1", "eval_analysis", v=1))
    router.upsert(_doc("r1", "eval_run"))
    router.upsert(_doc("n1", "host_note"))
    router.upsert({**_doc("a2", "eval_analysis"), "scope_id": "scope-b"})

    moved = RoutedDocumentStore.relocate(
        stores, _BY_LIFECYCLE, scope_ids=[_SCOPE, "scope-b"], host_doc_types=["host_note"]
    )

    assert moved == {"eval_analysis": 2}
    assert _held(stores["scratch"]) == {"a1"} and _held(stores["evidence"]) == {"r1", "n1"}
    started = RoutedDocumentStore(stores, _BY_LIFECYCLE)
    assert started.get("a1", _SCOPE) == _doc("a1", "eval_analysis", v=1)
    assert [d["id"] for d in started.by_doc_type("eval_analysis", "scope-b")] == ["a2"]


def test_relocate_moves_a_host_type_it_is_told_of() -> None:
    router, stores = _router()
    router.upsert(_doc("n1", "host_note"))
    to_scratch = DocTypeRoute(by="lifecycle", to={"kept": "evidence", "regenerable": "scratch"}, unregistered="scratch")

    assert RoutedDocumentStore.relocate(stores, to_scratch, scope_ids=[_SCOPE], host_doc_types=["host_note"]) == {
        "host_note": 1
    }
    assert _held(stores["scratch"]) == {"n1"}


def test_an_interrupted_relocation_is_finished_by_running_it_again() -> None:
    """A copy that landed before the interruption is found identical where it belongs, and its stray removed."""
    _, stores = _router(route=_EVERYTHING_IN_EVIDENCE)
    stores["evidence"].upsert(_doc("a1", "eval_analysis", v=1))
    stores["scratch"].upsert(_doc("a1", "eval_analysis", v=1))

    assert RoutedDocumentStore.relocate(stores, _BY_LIFECYCLE, scope_ids=[_SCOPE]) == {"eval_analysis": 1}
    assert _held(stores["evidence"]) == set() and _held(stores["scratch"]) == {"a1"}


def test_relocate_refuses_two_different_documents_under_one_id() -> None:
    _, stores = _router(route=_EVERYTHING_IN_EVIDENCE)
    stores["evidence"].upsert(_doc("a1", "eval_analysis", v="one"))
    stores["scratch"].upsert(_doc("a1", "eval_analysis", v="two"))

    with pytest.raises(MisplacedDocument, match="the two differ"):
        RoutedDocumentStore.relocate(stores, _BY_LIFECYCLE, scope_ids=[_SCOPE])
    assert _held(stores["evidence"]) == {"a1"}, "nothing of that id was moved"
    with pytest.raises(RouteChanged):
        RoutedDocumentStore(stores, _BY_LIFECYCLE)


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
    for by, values in (("origin", ("input", "record", "derived")), ("lifecycle", ("kept", "regenerable"))):
        for split in values:
            route = DocTypeRoute(by=by, to={v: ("b" if v == split else "a") for v in values}, unregistered="a")  # type: ignore[arg-type]
            placements = route.placements(DOC_TYPE_FACTS)
            assert placements["eval_sweep"] == placements["eval_analysis_attempt"] == placements["eval_run"]
