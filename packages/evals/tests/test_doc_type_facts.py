"""The doc-type registry is the one list of stored types, and every list the engine keeps is derived from it.

``DOC_TYPE_FACTS`` says, per stored ``doc_type``, where its documents come from, whether a later release reads them,
and what a tombstone suppresses. These tests tie it to the model graph from both ends, so a stored model cannot ship
without a registered type, and a registered type cannot claim a lifecycle its model does not keep: a ``kept`` type is
exactly a model on the core base, which is what upgrades it on read.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from packages.evals.tests.stored_models import doc_type_of, stored_models
from threetears.evals.schema import DOC_TYPE_FACTS, DocTypeFacts
from threetears.evals.schema.base import CoreDocumentModel
from threetears.evals.schema.models import EVAL_DOC_TYPES
from threetears.evals.schema.versioning import CORE_DOC_TYPES, REGENERABLE_SCHEMA_VERSION


def test_every_stored_model_s_doc_type_is_registered_exactly_once() -> None:
    graph = Counter(doc_type_of(model) for model in stored_models())

    assert "eval_template" in graph and "eval_campaign" in graph, "the derivation found no stored models at all"
    assert [doc_type for doc_type, count in graph.items() if count > 1] == [], "two models write one doc_type"
    assert set(graph) == set(DOC_TYPE_FACTS)


def test_the_wipe_list_and_the_core_are_derived_from_the_registry() -> None:
    assert EVAL_DOC_TYPES == tuple(DOC_TYPE_FACTS)
    assert {doc_type for doc_type, facts in DOC_TYPE_FACTS.items() if facts.lifecycle == "kept"} == CORE_DOC_TYPES


@pytest.mark.parametrize("model", stored_models(), ids=lambda model: model.__name__)
def test_a_type_is_kept_exactly_when_its_model_is_on_the_core_base(model: Any) -> None:
    """A kept type upgrades on read only through the core base, and a regenerable one refuses another version."""
    facts = DOC_TYPE_FACTS[doc_type_of(model)]

    if facts.lifecycle == "kept":
        assert issubclass(model, CoreDocumentModel)
    else:
        assert not issubclass(model, CoreDocumentModel)
        assert model.model_fields["schema_version"].default == REGENERABLE_SCHEMA_VERSION


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
        "eval_template": ("input", "kept"),
        "judge_config": ("input", "kept"),
        "rubric_dim": ("input", "kept"),
        "rubric_dim_tombstone": ("input", "kept"),
        "judge_config_tombstone": ("input", "kept"),
        "case_set": ("input", "kept"),
        "eval_test_case": ("input", "kept"),
        "eval_campaign": ("input", "kept"),
        "eval_run": ("record", "kept"),
        "eval_result": ("record", "kept"),
        "eval_trace": ("record", "kept"),
        "calibration_rating": ("record", "kept"),
        "eval_out_of_run_spend": ("record", "kept"),
        "eval_sweep": ("record", "kept"),
        "eval_analysis_attempt": ("record", "kept"),
        "eval_cassette": ("record", "regenerable"),
        "eval_generation_slot": ("record", "regenerable"),
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
    ],
)
def test_facts_outside_the_vocabulary_are_refused(fields: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        DocTypeFacts(**fields)
