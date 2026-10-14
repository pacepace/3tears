"""The doc-type registry is the one list of stored types, and every list the engine keeps is derived from it.

``DOC_TYPE_FACTS`` says, per stored ``doc_type``, where its documents come from, whether a later release reads them,
and what a tombstone suppresses. These tests tie it to the model graph from both ends, so a stored model cannot ship
without a registered type, and a registered type cannot claim a lifecycle its model does not keep: a ``kept`` type is
exactly a model on the core base read in the core's space, and a ``durable`` type exactly one read in the durable
space: what upgrades it on read, and what decides whether a core baseline reset deletes it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.stored_models import doc_type_of, stored_models, stored_sample
from threetears.evals.schema import DOC_TYPE_FACTS, DocTypeFacts
from threetears.evals.schema.base import CoreDocumentModel
from threetears.evals.schema import models as schema_models
from threetears.evals.schema.models import EVAL_DOC_TYPES, EvalCassette
from threetears.evals.schema.versioning import (
    CASSETTE_SCHEMA_VERSION,
    CORE_DOC_TYPES,
    CORE_RESET_DOC_TYPES,
    CORE_VERSION_SPACE,
    DURABLE_DOC_TYPES,
    DURABLE_VERSION_SPACE,
    REGENERABLE_SCHEMA_VERSION,
)

#: What people author, as decided at ``.26``: the eight types a core baseline reset never touches.
DURABLE = frozenset(
    {
        "eval_template",
        "judge_config",
        "rubric_dim",
        "rubric_dim_tombstone",
        "judge_config_tombstone",
        "case_set",
        "eval_test_case",
        "calibration_rating",
    }
)


def test_every_stored_model_s_doc_type_is_registered_exactly_once() -> None:
    graph = Counter(doc_type_of(model) for model in stored_models())

    assert "eval_template" in graph and "eval_campaign" in graph, "the derivation found no stored models at all"
    assert [doc_type for doc_type, count in graph.items() if count > 1] == [], "two models write one doc_type"
    assert set(graph) == set(DOC_TYPE_FACTS)


def test_the_wipe_list_and_the_core_are_derived_from_the_registry() -> None:
    assert EVAL_DOC_TYPES == tuple(DOC_TYPE_FACTS)
    assert {doc_type for doc_type, facts in DOC_TYPE_FACTS.items() if facts.lifecycle == "kept"} == CORE_DOC_TYPES
    assert {doc_type for doc_type, facts in DOC_TYPE_FACTS.items() if facts.lifecycle == "durable"} == DURABLE_DOC_TYPES


def test_the_durable_set_is_exactly_what_people_author_and_never_core_or_derived() -> None:
    assert DURABLE_DOC_TYPES == DURABLE
    assert not CORE_DOC_TYPES & DURABLE_DOC_TYPES
    assert not any(DOC_TYPE_FACTS[doc_type].origin == "derived" for doc_type in DURABLE_DOC_TYPES)
    assert CORE_RESET_DOC_TYPES == set(DOC_TYPE_FACTS) - DURABLE, "a core baseline reset deletes everything else"


@pytest.mark.parametrize("model", stored_models(), ids=lambda model: model.__name__)
def test_a_type_is_kept_or_durable_exactly_when_its_model_is_read_in_that_space(model: Any) -> None:
    """A kept or durable type upgrades on read only through the core base, in its own space; a regenerable one
    refuses another version."""
    facts = DOC_TYPE_FACTS[doc_type_of(model)]

    if facts.lifecycle == "kept":
        assert issubclass(model, CoreDocumentModel) and model.version_space is CORE_VERSION_SPACE
    elif facts.lifecycle == "durable":
        assert issubclass(model, CoreDocumentModel) and model.version_space is DURABLE_VERSION_SPACE
    else:
        assert not issubclass(model, CoreDocumentModel)
        assert model.model_fields["schema_version"].default in {REGENERABLE_SCHEMA_VERSION, CASSETTE_SCHEMA_VERSION}


def _regenerable_models() -> list[Any]:
    return [model for model in stored_models() if DOC_TYPE_FACTS[doc_type_of(model)].lifecycle == "regenerable"]


@pytest.mark.parametrize("model", _regenerable_models(), ids=lambda model: model.__name__)
def test_the_cassette_alone_is_written_under_the_cassette_version(model: Any) -> None:
    """The two regenerable version spaces stay apart: the cassette carries one constant, every other model the other."""
    if model is EvalCassette:
        assert model.model_fields["schema_version"].default == CASSETTE_SCHEMA_VERSION
    else:
        assert model.model_fields["schema_version"].default == REGENERABLE_SCHEMA_VERSION


@pytest.mark.parametrize("model", _regenerable_models(), ids=lambda model: model.__name__)
def test_each_regenerable_bump_refuses_its_own_space_and_no_other(model: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """An analysis bump must never refuse a recorded corpus, nor a cassette bump an analysis.

    A cassette is re-captured by calling the third-party tools again, so it reads against
    ``CASSETTE_SCHEMA_VERSION`` and nothing else does. Each constant is moved in turn, to a number neither space
    uses, and a stored document reads exactly when the constant it is governed by did not move: so a cassette typed
    ``SchemaVersion``, or an analysis typed ``CassetteSchemaVersion``, is named here whatever the two numbers are.
    """
    document = stored_sample(model)
    unused = max(REGENERABLE_SCHEMA_VERSION, CASSETTE_SCHEMA_VERSION) + 1
    governs = "CASSETTE_SCHEMA_VERSION" if model is EvalCassette else "REGENERABLE_SCHEMA_VERSION"

    for constant in ("REGENERABLE_SCHEMA_VERSION", "CASSETTE_SCHEMA_VERSION"):
        with monkeypatch.context() as bumped:
            bumped.setattr(schema_models, constant, unused)
            if constant == governs:
                with pytest.raises(ValidationError, match=f"v{document['schema_version']}"):
                    model.from_dict(document)
            else:
                assert model.from_dict(document).schema_version == document["schema_version"]


def test_every_tombstone_suppresses_a_registered_type_sharing_its_facts() -> None:
    """A tombstone is read with the type it suppresses; kept apart in time or place, it outlives its slot."""
    tombstones = {doc_type: facts for doc_type, facts in DOC_TYPE_FACTS.items() if facts.suppresses is not None}

    assert set(tombstones) == {doc_type for doc_type in DOC_TYPE_FACTS if doc_type.endswith("_tombstone")}
    for doc_type, facts in tombstones.items():
        assert facts.suppresses is not None
        suppressed = DOC_TYPE_FACTS[facts.suppresses]
        assert doc_type == f"{facts.suppresses}_tombstone"
        assert (suppressed.origin, suppressed.lifecycle) == (facts.origin, facts.lifecycle)


def test_the_placements_the_registry_records() -> None:
    """What each type is, as decided: a change here is a decision about what is kept, never a side effect."""
    assert {doc_type: (facts.origin, facts.lifecycle) for doc_type, facts in DOC_TYPE_FACTS.items()} == {
        "eval_template": ("input", "durable"),
        "judge_config": ("input", "durable"),
        "rubric_dim": ("input", "durable"),
        "rubric_dim_tombstone": ("input", "durable"),
        "judge_config_tombstone": ("input", "durable"),
        "case_set": ("input", "durable"),
        "eval_test_case": ("input", "durable"),
        "eval_campaign": ("input", "kept"),
        "eval_run": ("record", "kept"),
        "eval_result": ("record", "kept"),
        "eval_trace": ("record", "kept"),
        "calibration_rating": ("record", "durable"),
        "eval_out_of_run_spend": ("record", "kept"),
        "eval_sweep": ("record", "kept"),
        "eval_analysis_attempt": ("record", "kept"),
        "eval_cassette": ("record", "regenerable"),
        "eval_generation_slot": ("record", "kept"),
        "eval_launch": ("record", "kept"),
        "eval_second_judge_pass": ("record", "kept"),
        "eval_judge_temperature_check": ("record", "kept"),
        "eval_action_job": ("record", "kept"),
        "eval_analysis": ("derived", "regenerable"),
        "eval_insight": ("derived", "regenerable"),
        "eval_judge_profile": ("derived", "regenerable"),
    }


@pytest.mark.parametrize(
    ("fields", "refusal"),
    [
        ({"origin": "authored", "lifecycle": "kept"}, "origin is one of"),
        ({"origin": "input", "lifecycle": "forever"}, "lifecycle is one of"),
        ({"origin": "derived", "lifecycle": "kept"}, "a derived document is computed again"),
        ({"origin": "derived", "lifecycle": "durable"}, "a derived document is computed again"),
    ],
)
def test_facts_outside_the_vocabulary_are_refused(fields: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        DocTypeFacts(**fields)
