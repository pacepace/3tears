"""``EvalStorage`` reads and writes every eval document through one store, addressed by scope.

Definitions and runtime documents used to sit in two stores with two scope mechanisms: a
definitions scope the host bound at construction, and a per-call scope for runs. Now one
``DocumentStore`` holds them all, every document names its own ``scope_id``, and every read names
the scope it asks about. These tests pin that a read in another scope sees nothing, that the wipe
names every type the model graph can store, that the wipe and the boot reclaim reach exactly the
scopes their host names, and that a host cannot claim an engine type.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.run.jobs import EvalJobManager
from threetears.evals.run.lifecycle import sweep_abandoned_runs
from threetears.evals.contracts.storage import EVAL_DOC_TYPES, EvalStorage

from packages.evals.tests.factories import (
    make_analysis,
    make_analysis_attempt,
    make_campaign,
    make_eval_run,
    make_insight,
    make_judge_config,
    make_rubric_dim,
    make_template,
)
from packages.evals.tests.memory_store import InMemoryDocumentStore, memory_storage
from packages.evals.tests.stored_models import doc_type_of, stored_models

_HOME = "scope-a"
_ELSEWHERE = "scope-b"


def test_the_wipe_names_every_doc_type_the_model_graph_can_store() -> None:
    """A stored model missing from the wipe's list leaves documents behind an operator was told were gone."""
    graph = {doc_type_of(model) for model in stored_models()}

    assert "eval_template" in graph and "eval_run" in graph, "the derivation found no stored models at all"
    assert set(EVAL_DOC_TYPES) == graph
    assert len(EVAL_DOC_TYPES) == len(set(EVAL_DOC_TYPES))


def test_every_stored_model_is_reached_by_the_derivation() -> None:
    """The positive control for the derivation: each model the storage layer writes is in the set it walks."""
    names = {model.__name__ for model in stored_models()}

    assert {"EvalTemplate", "EvalCampaign", "EvalAnalysis", "EvalInsight", "EvalRun", "EvalCassette"} <= names


#: Each definition kind: how to build one in a scope, save it, and read it back by id in a scope.
_DEFINITIONS: dict[str, tuple[Callable[[str], EvalBaseModel], str, Callable[[EvalStorage, str, str], Any]]] = {
    "template": (lambda s: make_template(scope_id=s), "save_template", EvalStorage.load_template),
    "judge_config": (lambda s: make_judge_config(scope_id=s), "save_judge_config", EvalStorage.load_judge_config),
    "rubric_dim": (lambda s: make_rubric_dim(scope_id=s), "save_rubric_dim", EvalStorage.load_rubric_dim),
    "campaign": (lambda s: make_campaign(scope_id=s), "save_campaign", EvalStorage.load_campaign),
    "analysis": (lambda s: make_analysis(scope_id=s), "save_analysis", EvalStorage.load_analysis),
    "insight": (lambda s: make_insight(scope_id=s), "save_insight", EvalStorage.load_insight),
}


@pytest.mark.parametrize("kind", sorted(_DEFINITIONS))
def test_a_definition_is_read_back_in_its_own_scope_and_nowhere_else(kind: str) -> None:
    build, save, load = _DEFINITIONS[kind]
    storage, _ = memory_storage()
    definition = build(_HOME)
    getattr(storage, save)(definition)

    assert load(storage, definition.id, _HOME) == definition, "the positive control: the home scope reads it"
    assert load(storage, definition.id, _ELSEWHERE) is None


def test_every_definition_listing_is_confined_to_the_scope_it_names() -> None:
    storage, _ = memory_storage()
    storage.save_template(make_template(scope_id=_HOME))
    storage.save_judge_config(make_judge_config(scope_id=_HOME))
    storage.save_rubric_dim(make_rubric_dim(scope_id=_HOME))
    storage.save_campaign(make_campaign(scope_id=_HOME, id="campaign-1"))
    storage.save_analysis(make_analysis(scope_id=_HOME))
    storage.save_analysis_attempt(make_analysis_attempt(scope_id=_HOME))
    storage.save_insight(make_insight(scope_id=_HOME))

    listings: dict[str, Callable[[str], list[Any]]] = {
        "templates": storage.query_templates,
        "judge_configs": storage.query_judge_configs,
        "rubric_dims": storage.query_rubric_dims,
        "campaigns": storage.list_campaigns,
        "analyses": lambda scope: storage.list_analyses_by_campaign("campaign-1", scope),
        "attempts": lambda scope: storage.list_analysis_attempts_by_campaign("campaign-1", scope),
        "insights": storage.query_insights,
    }
    for name, listing in listings.items():
        assert len(listing(_HOME)) == 1, f"{name}: the home scope lists the one it holds"
        assert listing(_ELSEWHERE) == [], f"{name}: another scope lists nothing"


def test_definitions_and_runs_share_the_one_store() -> None:
    storage, store = memory_storage()
    template = make_template(scope_id=_HOME)
    run = make_eval_run(scope_id=_HOME, template_id=template.id)
    storage.save_template(template)
    storage.save_eval_run(run)

    assert set(store.documents) == {(_HOME, template.id), (_HOME, run.id)}


def test_the_wipe_reaches_every_tier_and_the_host_s_own_documents() -> None:
    store = InMemoryDocumentStore()
    storage = EvalStorage(store, host_doc_types=["host_note"])
    storage.save_template(make_template(scope_id=_HOME))
    storage.save_eval_run(make_eval_run(scope_id=_ELSEWHERE))
    store.upsert({"id": "note-1", "doc_type": "host_note", "scope_id": _HOME})

    counts = storage.nuke_all_eval_data([_HOME, _ELSEWHERE])

    assert store.documents == {}
    assert counts["eval_template"] == 1 and counts["eval_run"] == 1 and counts["host_note"] == 1


def test_the_wipe_touches_only_the_scopes_it_is_named() -> None:
    storage, store = memory_storage()
    storage.save_template(make_template(scope_id=_HOME))
    storage.save_template(make_template(scope_id=_ELSEWHERE))

    counts = storage.nuke_all_eval_data([_HOME])

    assert counts["eval_template"] == 1
    assert [key for key in store.documents] == [(_ELSEWHERE, storage.query_templates(_ELSEWHERE)[0].id)]


def test_the_reclaim_settles_runs_only_in_the_scopes_it_is_named() -> None:
    """The boot reclaim is told which scopes to sweep; it cannot find the others and does not try."""
    storage, _ = memory_storage()
    here = make_eval_run(id="run-here", scope_id=_HOME, status="running")
    there = make_eval_run(id="run-there", scope_id=_ELSEWHERE, status="running")
    storage.save_eval_run(here)
    storage.save_eval_run(there)

    report = sweep_abandoned_runs(storage, [_HOME], job_manager=EvalJobManager(storage))

    assert report.scanned == 1 and report.cancelled_run_ids == ["run-here"]
    assert storage.load_eval_run("run-here", _HOME).status == "cancelled"
    assert storage.load_eval_run("run-there", _ELSEWHERE).status == "running"


@pytest.mark.parametrize("claimed", ["eval_run", "eval_template"])
def test_a_host_cannot_declare_an_engine_doc_type_as_its_own(claimed: str) -> None:
    with pytest.raises(ValueError, match=claimed):
        EvalStorage(InMemoryDocumentStore(), host_doc_types=[claimed])
