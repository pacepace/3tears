"""A core document of this build's major at a newer minor is read, and written back whole.

``schema/versioning.py`` states the rule: a minor is additive, so a reader accepts any newer minor of its own major,
keeps every field it does not declare — at every depth — out of every attribute, dump and digest, and writes them back
where they were, stamping the newest minor any of them was read at. Strictness stays everywhere drift can be seen: an
unknown field at this build's minor or older is refused, a newer major is refused, and only the stored read is ever
tolerant — construction, payload validation and every authoring surface are as strict as before.

The rule exists for a mixed fleet: the old side of every rolling upgrade is whatever this build's reader does, so each
test here is phrased as what a document a NEWER build wrote does when it reaches this one. ``NEWER`` is that build's
minor; ``NEWER_FIELD`` a field it added.
"""

from __future__ import annotations

import json
import typing
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from packages.evals.tests.factories import make_judge_config, make_rubric_dim, memory_storage
from packages.evals.tests.stored_models import doc_type_of, stored_models, stored_sample
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.job_leases import request_job_cancel
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.authoring import update_judge_config, update_rubric_dim, update_template
from threetears.evals.run.run_document import set_run_labels
from threetears.evals.schema.base import CoreDocumentModel, CoreShapeModel
from threetears.evals.schema.models import (
    EvalResult,
    EvalRun,
    EvalTemplate,
    EvalTrace,
    JudgedArtifact,
    JudgeEvidence,
    fingerprint_judged_output,
)
from threetears.evals.schema.versioning import CORE_ADDRESSING_FIELDS, CORE_SCHEMA_MINOR, CORE_SCHEMA_VERSION
from threetears.evals.storage import InMemoryDocumentStore

#: The minor a newer build wrote at.
NEWER = CORE_SCHEMA_MINOR + 1

#: A field that newer build added, with a value only it knows the meaning of.
NEWER_FIELD = {"written_by_a_newer_build": {"kept": ["exactly", "as", "written"], "count": 3}}


def _core_models() -> list[type[CoreDocumentModel]]:
    return [model for model in stored_models() if issubclass(model, CoreDocumentModel)]


def _at_newer_minor(document: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """``document`` as the newer build would have written it: at its minor, with ``fields`` added."""
    return {**document, "schema_minor": NEWER, **fields}


def _tamper(store: InMemoryDocumentStore, doc_id: str, scope_id: str, **fields: Any) -> None:
    """Rewrite one stored document with ``fields`` merged in, as the newer build's write would have left it."""
    document = store.get(doc_id, scope_id)
    assert document is not None, f"nothing is stored under {doc_id}"
    store.upsert({**document, **fields})


# --- the read and the write ------------------------------------------------------------------------


def test_a_newer_minor_result_keeps_an_unknown_top_level_field() -> None:
    """Read, kept out of every attribute and dump, and written back with the newer minor."""
    document = _at_newer_minor(stored_sample(EvalResult), **NEWER_FIELD)

    read = EvalResult.from_dict(document)

    assert read.schema_minor == NEWER
    assert not hasattr(read, "written_by_a_newer_build")
    assert "written_by_a_newer_build" not in read.model_dump(mode="json")
    assert "written_by_a_newer_build" not in json.loads(read.model_dump_json())
    assert read.to_dict() == document


def test_a_newer_minor_result_keeps_an_unknown_field_inside_a_rubric_score() -> None:
    """The nested case: a field a newer build added to an embedded shape is kept on that shape, and put back there."""
    document = _at_newer_minor(stored_sample(EvalResult))
    document["rubric_scores"][0]["calibrated_by"] = "a newer build's calibrator"

    read = EvalResult.from_dict(document)

    assert "calibrated_by" not in read.rubric_scores[0].model_dump(mode="json")
    written = read.to_dict()
    assert written["rubric_scores"][0]["calibrated_by"] == "a newer build's calibrator"
    assert written == document


@pytest.mark.parametrize("model", _core_models(), ids=lambda model: model.__name__)
def test_every_core_model_reads_a_newer_minor_and_writes_its_unknown_fields_back(
    model: type[CoreDocumentModel],
) -> None:
    document = _at_newer_minor(stored_sample(model), **NEWER_FIELD)

    assert model.from_dict(document).to_dict() == document


def _embedded_models(model: type[BaseModel], seen: set[type[BaseModel]]) -> None:
    for field in model.model_fields.values():
        stack: list[Any] = [field.annotation]
        while stack:
            annotation = stack.pop()
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                if annotation not in seen:
                    seen.add(annotation)
                    _embedded_models(annotation, seen)
                continue
            stack.extend(typing.get_args(annotation))


def test_every_model_a_core_document_embeds_keeps_newer_minor_fields() -> None:
    """A shape a core document embeds, at any depth, carries the tolerance, or a newer field on it is refused."""
    seen: set[type[BaseModel]] = set()
    for model in _core_models():
        seen.add(model)
        _embedded_models(model, seen)

    missing = sorted(model.__name__ for model in seen if not issubclass(model, CoreShapeModel))

    assert not missing, f"embedded in a core document and not a CoreShapeModel: {missing}"
    assert len(seen) > len(_core_models()), "the walk found no embedded shape at all"


# --- strictness everywhere else --------------------------------------------------------------------


def test_an_unknown_field_at_this_build_s_minor_is_refused() -> None:
    """A writer at this build's minor cannot have written it, so it is drift, at the top level and nested."""
    document = stored_sample(EvalResult)
    nested = json.loads(json.dumps(document))
    nested["rubric_scores"][0]["calibrated_by"] = "x"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvalResult.from_dict({**document, **NEWER_FIELD})
    with pytest.raises(ValidationError, match="calibrated_by"):
        EvalResult.from_dict(nested)


@pytest.mark.parametrize("minor", [0, NEWER, 5])
def test_a_newer_major_is_refused_whatever_its_minor(minor: int) -> None:
    document = {**stored_sample(EvalResult), "schema_version": CORE_SCHEMA_VERSION + 1, "schema_minor": minor}

    with pytest.raises(ValidationError, match="by a newer build"):
        EvalResult.from_dict(document)


@pytest.mark.parametrize("minor", ["1", -1, True, 1.5, None])
def test_a_malformed_minor_is_refused_by_name(minor: object) -> None:
    with pytest.raises(ValidationError, match="schema_minor") as refused:
        EvalResult.from_dict({**stored_sample(EvalResult), "schema_minor": minor})
    assert "not a core minor number" in str(refused.value)


def test_an_absent_minor_reads_as_the_baseline_minor() -> None:
    document = stored_sample(EvalResult)
    del document["schema_minor"]

    read = EvalResult.from_dict(document)

    assert read.schema_minor == CORE_SCHEMA_MINOR
    assert read.to_dict()["schema_minor"] == CORE_SCHEMA_MINOR


def test_only_the_stored_read_tolerates_a_newer_minor() -> None:
    """Construction and payload validation at a newer minor are refused: a rebuild cannot silently drop a field."""
    document = _at_newer_minor(stored_sample(EvalResult))
    read = EvalResult.from_dict(_at_newer_minor(stored_sample(EvalResult), **NEWER_FIELD))

    with pytest.raises(ValidationError, match="built at core minor"):
        EvalResult.model_validate(document)
    with pytest.raises(ValidationError, match="extra_forbidden|built at core minor"):
        EvalResult(**read.to_dict())
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvalResult.model_validate({**stored_sample(EvalResult), **NEWER_FIELD})


def test_model_copy_and_assignment_keep_the_newer_fields() -> None:
    document = _at_newer_minor(stored_sample(EvalRun), **NEWER_FIELD)
    read = EvalRun.from_dict(document)

    copied = read.model_copy(update={"labels": ["kept"]})
    read.labels = ["assigned"]

    assert copied.to_dict() == {**document, "labels": ["kept"]}
    assert read.to_dict() == {**document, "labels": ["assigned"]}


def test_a_sub_model_carried_into_another_document_stamps_its_minor() -> None:
    """A rubric score read from a newer result, placed in a document this build wrote, takes its minor along."""
    newer = _at_newer_minor(stored_sample(EvalResult))
    newer["rubric_scores"][0]["calibrated_by"] = "x"
    score = EvalResult.from_dict(newer).rubric_scores[0]
    fresh = EvalResult.from_dict(stored_sample(EvalResult))
    assert fresh.schema_minor == CORE_SCHEMA_MINOR

    written = fresh.model_copy(update={"rubric_scores": [score]}).to_dict()

    assert written["schema_minor"] == NEWER
    assert written["rubric_scores"][0]["calibrated_by"] == "x"
    assert EvalResult.from_dict(written).to_dict() == written


def test_the_remainder_enters_no_digest() -> None:
    """A judged output's fingerprint is taken over what this build declares, whatever a newer build added to it."""
    evidence = JudgeEvidence(case_material="a customer asks", artifact="assistant: two tickets")
    document = _at_newer_minor(
        stored_sample(EvalTrace),
        judge_evidence={**evidence.model_dump(mode="json"), "rendered_with": "v2"},
        judged_artifact=JudgedArtifact.DOCUMENT.value,
    )

    read = EvalTrace.from_dict(document)

    assert read.judge_evidence is not None
    assert fingerprint_judged_output(read.judge_evidence) == fingerprint_judged_output(evidence)
    assert read.to_dict() == document


def test_the_json_pair_of_a_core_document_is_its_stored_read() -> None:
    read = EvalResult.from_dict(_at_newer_minor(stored_sample(EvalResult), **NEWER_FIELD))

    assert EvalResult.from_json(read.to_json()).to_dict() == read.to_dict()
    with pytest.raises(ValidationError):
        EvalResult.from_json("{not json")


# --- every write path ------------------------------------------------------------------------------


def _stored_newer_run() -> tuple[EvalStorage, InMemoryDocumentStore, EvalRun]:
    """A pending run saved, then rewritten as the newer build would have left it."""
    storage, store = memory_storage()
    run = EvalRun.from_dict(stored_sample(EvalRun))
    storage.save_eval_run(run)
    _tamper(store, run.id, run.scope_id, schema_minor=NEWER, **NEWER_FIELD)
    return storage, store, run


def _kept(store: InMemoryDocumentStore, run: EvalRun) -> dict[str, Any]:
    document = store.get(run.id, run.scope_id)
    assert document is not None
    assert document["written_by_a_newer_build"] == NEWER_FIELD["written_by_a_newer_build"]
    assert document["schema_minor"] == NEWER
    return document


def test_a_whole_write_of_a_read_run_keeps_the_newer_fields() -> None:
    storage, store, run = _stored_newer_run()
    read, etag = storage.load_eval_run_with_etag(run.id, run.scope_id)
    assert read is not None

    storage.save_eval_run(read.model_copy(update={"labels": ["rewritten"]}), if_match=etag)

    assert _kept(store, run)["labels"] == ["rewritten"]


def test_a_read_modify_write_keeps_the_newer_fields() -> None:
    """``update_eval_run``'s to_dict → edit → from_dict, as every run document writer does it."""
    storage, store, run = _stored_newer_run()

    set_run_labels(storage, run.id, run.scope_id, labels=["relabelled"])

    assert _kept(store, run)["labels"] == ["relabelled"]


def test_a_partial_write_keeps_the_newer_fields() -> None:
    storage, store, run = _stored_newer_run()

    storage.set_eval_run_archived(run.id, run.scope_id, archived=True)
    request_job_cancel(storage, run.id, run.scope_id, reason="rolling")

    document = _kept(store, run)
    assert document["archived"] is True
    assert document["cancel_request_reason"] == "rolling"


def test_a_settling_read_writes_the_run_lost_and_keeps_the_newer_fields() -> None:
    """A pending run with no lease is settled lost by the next read, a conditional write of the whole record."""
    storage, store, run = _stored_newer_run()

    settled = storage.load_eval_run(run.id, run.scope_id)

    assert settled is not None and settled.lost is not None
    assert _kept(store, run)["lost"] is not None


def test_every_partial_write_names_only_addressing_fields() -> None:
    """A merge cannot take the newest of a minor it never read, so it names only baseline fields: addressing ones."""
    cancel = {"cancel_requested_at", "cancel_request_reason"}
    for doc_type in (
        "eval_run",
        "eval_sweep",
        "eval_analysis_attempt",
        "eval_launch",
        "eval_second_judge_pass",
        "eval_judge_temperature_check",
        "eval_action_job",
    ):
        assert cancel <= CORE_ADDRESSING_FIELDS[doc_type], doc_type
    assert "archived" in CORE_ADDRESSING_FIELDS["eval_run"]


# --- authoring: caller input never reaches the tolerant read ---------------------------------------


def _stored_newer(store: InMemoryDocumentStore, document: CoreDocumentModel) -> None:
    store.upsert({**document.to_dict(), "schema_minor": NEWER, **NEWER_FIELD})


def test_a_rubric_dim_and_a_judge_config_a_newer_build_wrote_stay_editable_and_whole() -> None:
    storage, store = memory_storage()
    dim = make_rubric_dim()
    config = make_judge_config()
    _stored_newer(store, dim)
    _stored_newer(store, config)

    edited = update_rubric_dim(storage, dim.id, dim.scope_id, {"universal": not dim.universal})
    reauthored = update_judge_config(storage, config.id, config.scope_id, {"name": "reauthored"})

    for document in (edited, reauthored):
        stored = store.get(document.id, document.scope_id)
        assert stored is not None
        assert stored["written_by_a_newer_build"] == NEWER_FIELD["written_by_a_newer_build"]
        assert stored["schema_minor"] == NEWER
    assert edited.universal is not dim.universal
    assert reauthored.id != config.id


def _authoring_checks() -> dict[str, Any]:
    return {
        "require_known_tools_allowed": lambda _tools: None,
        "refuse_undeclared_world_seed": lambda _template: None,
        "refuse_undeliverable_template": lambda _template: None,
    }


def test_a_template_a_newer_build_wrote_stays_editable_and_whole() -> None:
    from packages.evals.tests.fixtures.toyhost.host import toyhost_host

    store = InMemoryDocumentStore()
    host = toyhost_host(storage=EvalStorage(store))
    template = EvalTemplate.from_dict(stored_sample(EvalTemplate))
    store.upsert({**template.to_dict(), "schema_minor": NEWER, **NEWER_FIELD})

    edited = update_template(host, template.id, template.scope_id, {"description": "reworded"}, **_authoring_checks())

    stored = store.get(template.id, template.scope_id)
    assert stored is not None and stored["written_by_a_newer_build"] == NEWER_FIELD["written_by_a_newer_build"]
    assert edited.description == "reworded"


@pytest.mark.parametrize("stored_minor", [CORE_SCHEMA_MINOR, NEWER])
def test_a_caller_cannot_send_a_newer_minor_or_a_field_this_build_does_not_declare(stored_minor: int) -> None:
    """An unknown name is refused before the merge, and ``schema_minor`` is a server field, ignored if sent.

    Whatever minor the stored document is at, nothing the caller sent reaches the stored read's tolerance: the
    stored read only ever sees fields a newer build wrote.
    """
    storage, store = memory_storage()
    dim = make_rubric_dim()
    store.upsert({**dim.to_dict(), "schema_minor": stored_minor})
    before = store.get(dim.id, dim.scope_id)

    with pytest.raises(ValidationFailedError, match="not a field of CatalogRubricDim: written_by_a_newer_build"):
        update_rubric_dim(storage, dim.id, dim.scope_id, {"schema_minor": 99, **NEWER_FIELD})
    assert store.get(dim.id, dim.scope_id) == before

    update_rubric_dim(storage, dim.id, dim.scope_id, {"schema_minor": 99, "universal": not dim.universal})
    stored = store.get(dim.id, dim.scope_id)
    assert stored is not None and stored["schema_minor"] == stored_minor


@pytest.mark.parametrize("stored_minor", [CORE_SCHEMA_MINOR, NEWER])
def test_a_caller_s_value_is_validated_strictly_at_every_depth_whatever_the_stored_minor(stored_minor: int) -> None:
    """A declared field the caller sends, carrying a key this build does not declare inside it, is refused.

    The case a name check cannot see: ``dim`` is a declared field, and the unknown key is inside it. On a document a
    newer build wrote, the stored read would keep it; the caller's half is validated strictly before the merge, so
    it never gets there.
    """
    storage, store = memory_storage()
    dim = make_rubric_dim()
    store.upsert({**dim.to_dict(), "schema_minor": stored_minor})
    before = store.get(dim.id, dim.scope_id)
    smuggled = {**dim.dim.model_dump(mode="json"), "calibrated_by": "a caller"}

    with pytest.raises(ValidationFailedError, match="calibrated_by"):
        update_rubric_dim(storage, dim.id, dim.scope_id, {"dim": smuggled})
    assert store.get(dim.id, dim.scope_id) == before


def test_each_core_model_names_its_doc_type() -> None:
    """The control for the parametrised tests above: every core model is a stored model with a doc type."""
    assert {doc_type_of(model) for model in _core_models()} == set(CORE_ADDRESSING_FIELDS)
