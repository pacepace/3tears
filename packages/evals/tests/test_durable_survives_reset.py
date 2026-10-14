"""A core baseline reset deletes the evidence core and what is derived from it, and never what people author.

``schema/versioning.py``'s standing rule: the core is baseline-resettable forever, and the durable namespace (labels,
judge configs, cases, templates) is never touched by a reset. This file simulates one: a scope holding a document of
every stored type is reset, and then read under a build whose core baseline moved on, with nothing restamped. It
also holds the two operations a host's runbook calls to the shape that makes a reset safe: ``reset_core`` is one
call that names no table and leaves the store bootable, and the full wipe deletes what people author only when its
caller says so by name.

The calibration half of the promise, that a label outlives its evidence and still calibrates a judge, is driven end to
end in ``tests/test_judge_kind.py``, beside the toy judge campaign it reuses.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.stored_models import doc_type_of, stored_models, stored_sample
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.schema.base import CoreDocumentModel
from threetears.evals.schema.doc_types import DOC_TYPE_FACTS
from threetears.evals.schema.versioning import (
    CORE_RESET_DOC_TYPES,
    CORE_VERSION_SPACE,
    DURABLE_DOC_TYPES,
)
from threetears.evals.storage import DocTypeRoute, InMemoryDocumentStore, RoutedDocumentStore

#: The scope every seeded document is moved into.
_SCOPE = "reset-scope"

#: The scope a router's placement record is kept in.
_ROUTING = "evals-routing.reset"


def _every_type() -> dict[str, tuple[type[Any], dict[str, Any]]]:
    """One stored document of every registered type, in :data:`_SCOPE`, by ``doc_type``."""
    seeded = {}
    for model in stored_models():
        document = {**stored_sample(model), "scope_id": _SCOPE}
        seeded[doc_type_of(model)] = (model, document)
    assert set(seeded) == set(DOC_TYPE_FACTS), "the seed holds every registered type"
    return seeded


def _seed(store: Any) -> dict[str, tuple[type[Any], dict[str, Any]]]:
    seeded = _every_type()
    for _, document in seeded.values():
        store.upsert(document)
    return seeded


def _held(store: Any, doc_type: str) -> list[str]:
    return list(store.iter_by_doc_type(doc_type, _SCOPE))


def test_reset_core_deletes_the_core_and_the_regenerable_and_keeps_every_durable_document_whole() -> None:
    store = InMemoryDocumentStore()
    seeded = _seed(store)
    store.upsert({"id": "note-1", "doc_type": "host_note", "scope_id": _SCOPE})

    counts = EvalStorage(store, host_doc_types=["host_note"]).reset_core([_SCOPE])

    assert set(counts) == CORE_RESET_DOC_TYPES and all(count == 1 for count in counts.values())
    for doc_type, (model, document) in seeded.items():
        if doc_type in DURABLE_DOC_TYPES:
            (doc_id,) = _held(store, doc_type)
            assert model.from_dict(store.get(doc_id, _SCOPE)) == model.from_dict(document)
        else:
            assert _held(store, doc_type) == [], f"{doc_type} survived a core reset"
    assert store.get("note-1", _SCOPE) is not None, "a host's own document is the host's, never the reset's"


def test_after_the_core_baseline_moves_every_durable_document_still_reads_and_a_core_one_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core moves to v16 with no step and nothing restamped: the durable space does not notice."""
    seeded = _every_type()
    monkeypatch.setattr(
        CoreDocumentModel, "version_space", dataclasses.replace(CORE_VERSION_SPACE, version=16, baseline=16)
    )

    for doc_type, (model, document) in seeded.items():
        if doc_type in DURABLE_DOC_TYPES:
            assert model.from_dict(document).to_dict() == document
        elif DOC_TYPE_FACTS[doc_type].lifecycle == "kept":
            with pytest.raises(ValidationError, match=f"{doc_type} document was written under core v15, below"):
                model.from_dict(document)


def test_the_full_wipe_makes_its_caller_say_whether_what_people_author_goes_too() -> None:
    storage = EvalStorage(InMemoryDocumentStore())

    with pytest.raises(TypeError, match="include_durable"):
        storage.nuke_all_eval_data([_SCOPE])  # type: ignore[call-arg]


# --- a routed store: the one call, and the store boots after it --------------------------------------------------

#: The route a host ran before ``.26``: what people author kept with the evidence, as every ``kept`` type was.
_BEFORE = DocTypeRoute(
    by="lifecycle", to={"kept": "evidence", "durable": "evidence", "regenerable": "scratch"}, unregistered="evidence"
)

#: The route it runs after: what people author in a store of its own, backed up like source code.
_AFTER = DocTypeRoute(
    by="lifecycle", to={"kept": "evidence", "durable": "authored", "regenerable": "scratch"}, unregistered="evidence"
)


def _routed() -> tuple[dict[str, InMemoryDocumentStore], dict[str, tuple[type[Any], dict[str, Any]]]]:
    """Stores a router ran over under :data:`_BEFORE`, holding a document of every type."""
    stores = {
        "evidence": InMemoryDocumentStore(),
        "scratch": InMemoryDocumentStore(),
        "authored": InMemoryDocumentStore(),
    }
    RoutedDocumentStore.relocate(stores, _BEFORE, scope_ids=[], routing_scope=_ROUTING)
    seeded = _seed(RoutedDocumentStore(stores, _BEFORE, routing_scope=_ROUTING))
    return stores, seeded


@pytest.mark.parametrize("route", [_BEFORE, _AFTER], ids=["same placements", "durable placed apart"])
def test_a_routed_reset_core_leaves_durable_documents_readable_the_core_empty_and_the_store_booting(
    route: DocTypeRoute,
) -> None:
    stores, seeded = _routed()

    reset = RoutedDocumentStore.reset_core(stores, route, scope_ids=[_SCOPE], routing_scope=_ROUTING)

    router = RoutedDocumentStore(stores, route, routing_scope=_ROUTING)
    storage = EvalStorage(router)
    assert dict(reset.deleted) == dict.fromkeys(CORE_RESET_DOC_TYPES, 1)
    moved = DURABLE_DOC_TYPES if route is _AFTER else set()
    assert set(reset.relocated) == moved
    for doc_type, (model, document) in seeded.items():
        held = _held(router, doc_type)
        if doc_type in DURABLE_DOC_TYPES:
            (doc_id,) = held
            assert model.from_dict(router.get(doc_id, _SCOPE)) == model.from_dict(document)
            assert router.placements[doc_type] == route.to["durable"]
        else:
            assert held == [], f"{doc_type} survived a core reset"
    assert storage.query_templates(_SCOPE), "the engine reads what people authored through the booted router"


def test_a_routed_reset_core_restores_placement_records_a_hand_run_truncate_wiped() -> None:
    """The incident the one call exists for: a runbook truncated tables by name, records included, and the router
    refused to start. ``reset_core`` writes them back, so the store boots."""
    stores, _ = _routed()
    for store in stores.values():
        for doc_id in list(store.iter_by_doc_type("eval_routing_record", _ROUTING)):
            store.delete(doc_id, _ROUTING)

    RoutedDocumentStore.reset_core(stores, _AFTER, scope_ids=[_SCOPE], routing_scope=_ROUTING)

    RoutedDocumentStore(stores, _AFTER, routing_scope=_ROUTING)


def test_a_routed_reset_core_refuses_the_routing_scope_as_a_document_scope() -> None:
    stores, _ = _routed()

    with pytest.raises(ValueError, match="routing scope"):
        RoutedDocumentStore.reset_core(stores, _AFTER, scope_ids=[_SCOPE, _ROUTING], routing_scope=_ROUTING)
    assert _held(stores["evidence"], "eval_run"), "refused before anything was deleted"
