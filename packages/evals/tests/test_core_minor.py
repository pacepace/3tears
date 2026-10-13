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
from enum import Enum
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
    RoleUsage,
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


# --- open vocabularies: a newer minor's value is read and named, never folded ----------------------


def _host_over(store: InMemoryDocumentStore) -> Any:
    from dataclasses import replace

    from threetears.evals.quick import callable_host

    return replace(callable_host(), storage=EvalStorage(store))


def _stored_run_and_result(
    run_fields: dict[str, Any], usage: list[dict[str, Any]], cost_roles: list[str]
) -> tuple[InMemoryDocumentStore, EvalRun]:
    """A completed run and one result of it, stored as a newer build wrote them: at its minor, with ``*_fields``."""
    store = InMemoryDocumentStore()
    run = EvalRun.from_dict(stored_sample(EvalRun)).model_copy(update={"status": "completed"})
    store.upsert({**run.to_dict(), "schema_minor": NEWER, **run_fields})
    result = {
        **stored_sample(EvalResult),
        "eval_run_id": run.id,
        "scope_id": run.scope_id,
        "usage": usage,
        "cost_roles": cost_roles,
        "schema_minor": NEWER,
    }
    store.upsert(result)
    return store, run


def _usage_row(role: str, model: str, cost_usd: float, calls: int) -> dict[str, Any]:
    return RoleUsage(role="judge", model=model, cost_usd=cost_usd, call_count=calls).model_dump(mode="json") | {
        "role": role
    }


def test_an_unknown_usage_role_at_a_newer_minor_is_disclosed_by_name() -> None:
    """A role a newer build added is summed on a line of its own, after the roles this build knows, and folded into
    none of them — the one reader that, before, iterated the known roles and silently dropped the rest."""
    from threetears.evals.analysis.summary import summarize_run

    store, run = _stored_run_and_result(
        {},
        [_usage_row("judge", "judge-a", 0.5, 1), _usage_row("reranker", "rr-1", 0.25, 2)],
        ["candidate", "judge", "reranker"],
    )

    summary = summarize_run(_host_over(store), run.id, run.scope_id)

    roles = [spend.role for spend in summary.spend]
    assert roles == ["judge", "reranker"]
    reranker = summary.role_spend("reranker")
    judge = summary.role_spend("judge")
    assert reranker is not None and judge is not None
    assert not reranker.known and judge.known
    assert (reranker.cost_usd, reranker.calls, judge.cost_usd) == (0.25, 2, 0.5)
    assert reranker.line().startswith("reranker spend (a role this build does not know): ")


def test_an_unknown_open_vocabulary_value_at_this_build_s_minor_is_refused() -> None:
    """Drift at this build's own minor is refused, by the vocabulary's name, as an unknown field is."""
    document = {**stored_sample(EvalResult), "usage": [_usage_row("reranker", "rr-1", 0.25, 2)]}

    with pytest.raises(ValidationError, match="'reranker' is not a usage role this build knows"):
        EvalResult.from_dict(document)
    with pytest.raises(ValidationError, match="is not a usage role this build knows"):
        RoleUsage(role="reranker")


def test_an_unknown_cost_cap_origin_is_named_and_reads_its_recorded_ceiling() -> None:
    from threetears.evals.analysis.summary import summarize_run
    from threetears.evals.kernel.host.sweepables import CORE_SWEEPABLES

    store, run = _stored_run_and_result(
        {"max_cost_usd": 2.0, "max_cost_usd_origin": "host_ceiling"},
        [_usage_row("judge", "judge-a", 0.5, 1)],
        ["judge"],
    )
    host = _host_over(store)

    summary = summarize_run(host, run.id, run.scope_id)
    stored_run = host.storage.load_eval_run(run.id, run.scope_id)
    ceiling = next(sweepable for sweepable in CORE_SWEEPABLES if sweepable.name == "max_cost_usd")

    assert (
        summary.spend_cap_text
        == "spend cap: $2.00 for this run (its origin, 'host_ceiling', is one this build does not know)"
    )
    assert stored_run is not None and ceiling.read(stored_run, []) == 2.0


def test_an_unknown_candidate_origin_reads_as_cannot_say_not_as_moved() -> None:
    """``origin != "inherited"`` used to read any origin a newer build added as a departure from production."""
    from threetears.evals.kernel.host.sweepables import CANDIDATE_MODEL_LEVER, CORE_SWEEPABLES

    candidate = next(sweepable for sweepable in CORE_SWEEPABLES if sweepable.name == CANDIDATE_MODEL_LEVER)
    assert candidate.departs_production is not None
    sample = stored_sample(EvalRun)
    read = {
        origin: EvalRun.from_dict(
            _at_newer_minor(sample, apparatus_provenance="commissioned", model_role_provenance={"candidate": origin})
        )
        for origin in ("chosen", "inherited", "from_a_newer_router")
    }

    assert {origin: candidate.departs_production(run, []) for origin, run in read.items()} == {
        "chosen": True,
        "inherited": False,
        "from_a_newer_router": None,
    }


def test_an_unknown_reasoning_effort_is_read_and_a_call_that_would_send_it_is_refused() -> None:
    """A run judged at an effort this build does not know reads; re-judging it today refuses, naming the setting."""
    from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
    from threetears.evals.run.rejudge import recorded_judge_pins

    sample = stored_sample(EvalRun)
    run = EvalRun.from_dict(
        _at_newer_minor(
            sample,
            judge_request_settings={
                **JUDGE_REQUEST_SETTINGS.model_dump(mode="json"),
                "reasoning_max_tokens": None,
                "reasoning_effort": "ultra",
            },
        )
    )

    assert run.judge_request_settings is not None and run.judge_request_settings.reasoning_effort == "ultra"
    with pytest.raises(ValidationFailedError, match="ultra"):
        recorded_judge_pins(run, request_settings="today")


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        ("EvalRun", "max_metered_calls_origin", "from_a_newer_metering"),
        ("EvalRun", "cell_timeout_s_origin", "from_a_newer_router"),  # beside its deadline, set below
        ("EvalRun", "judge_config_provenance", {"conversation.tone": "from_a_newer_router"}),
        ("OutOfRunSpend", "purpose", "rerank"),
        ("EvalActionJob", "refusal", {"kind": "quota_exhausted", "status": 429}),
    ],
)
def test_every_open_vocabulary_round_trips_an_unknown_value_at_a_newer_minor(
    model: str, field: str, value: Any
) -> None:
    cls = next(candidate for candidate in _core_models() if candidate.__name__ == model)
    document = _at_newer_minor(stored_sample(cls), **{field: value})
    if model == "EvalActionJob":
        document["outcome"] = "failed"
    if field == "cell_timeout_s_origin":
        document["cell_timeout_s"] = 30.0

    read = cls.from_dict(document)

    assert read.to_dict() == document
    with pytest.raises(ValidationError, match="this build knows"):
        cls.from_dict({**document, "schema_minor": CORE_SCHEMA_MINOR})


def test_a_simulator_purpose_a_newer_build_added_is_read_on_its_row() -> None:
    row = {**_usage_row("simulator", "sim-1", 0.1, 1), "actor_id": "user", "purpose": "narrate"}
    document = _at_newer_minor(stored_sample(EvalResult), usage=[row], cost_roles=["simulator"])

    assert EvalResult.from_dict(document).to_dict() == document


def test_each_registry_holds_every_value_its_writers_produce() -> None:
    """A writer here only ever writes a value its registry names, so this build's own documents read strictly."""
    from threetears.evals.kernel.refusals import refusal_kind
    from threetears.evals.kernel.errors import EvalServiceError
    from threetears.evals.run.budget import EvalRunCostCap
    from threetears.evals.run.metering import MeteredCallLedger
    from threetears.evals.schema import (
        COST_CAP_ORIGINS,
        METERED_CALL_ORIGINS,
        REFUSAL_KINDS,
    )

    cost = {
        EvalRunCostCap.resolve_ceiling_origin(cap, enforcement_enabled=on)
        for cap in (None, 1.0)
        for on in (True, False)
    }
    assert cost <= set(COST_CAP_ORIGINS)
    metered = {
        MeteredCallLedger.resolve_ceiling_origin(cap, configured_max_metered_calls=configured, enforcement_enabled=on)
        for cap in (None, 3)
        for on in (True, False)
        for configured in (None, 10)
    }
    assert metered <= set(METERED_CALL_ORIGINS)
    assert refusal_kind(EvalServiceError("x", remedy="none")) in REFUSAL_KINDS


#: Every closed vocabulary on a core shape, by the name the versioning docstring lists it under, and its values.
def _closed_vocabularies() -> dict[str, frozenset[Any]]:
    from threetears.evals.kernel.campaign import AttemptOutcome, SweepOutcome
    from threetears.evals.kernel.judge_temperature import TemperatureSelection, TemperatureSetting
    from threetears.evals.kernel.cassettes import CassetteMode
    from threetears.evals.quick.guardrails import GuardrailDirection
    from threetears.evals.run.judge_second import SecondJudgeAskedBy
    from threetears.evals.run.job_work import JobOutcome
    from threetears.evals.schema import models
    from threetears.evals.schema.completion import StopReason
    from threetears.evals.schema.judge_attribution import JudgeAttributionSource
    from threetears.evals.schema.out_of_run_spend import OutOfRunOutcome
    from threetears.evals.schema.world_events import WorldEventCause, WorldEventKind

    def values(alias: Any) -> frozenset[Any]:
        return frozenset(typing.get_args(alias))

    return {
        "EvalRunStatus": values(models.EvalRunStatus),
        "JobOutcome": values(JobOutcome),
        "SweepOutcome": values(SweepOutcome),
        "AttemptOutcome": values(AttemptOutcome),
        "CellTermination": values(models.CellTermination),
        "GoalCheckProof": values(models.GoalCheckProof),
        "GoalCheckIntent": values(models.GoalCheckIntent),
        "ApparatusProvenance": values(models.ApparatusProvenance),
        "RubricScale": values(models.RubricScale),
        "RubricAxis": values(models.RubricAxis),
        "nominal": frozenset({"nominal"}),
        "ordinal": frozenset({"ordinal"}),
        "interval": frozenset({"interval"}),
        "model_default": frozenset({"model_default"}),
        "WorldPlacement": values(models.WorldPlacement),
        "WorldEventKind": values(WorldEventKind),
        "WorldEventCause": values(WorldEventCause),
        "AsyncDeliveryStatus": values(models.AsyncDeliveryStatus),
        "CassetteMode": values(CassetteMode),
        "TemperatureSelection": values(TemperatureSelection),
        "TemperatureSetting": values(TemperatureSetting),
        "CompletenessSource": values(models.CompletenessSource),
        "JudgeAttributionSource": values(JudgeAttributionSource),
        "DeclaredMeritAxis": values(models.DeclaredMeritAxis),
        "GuardrailDirection": values(GuardrailDirection),
        "CampaignDesign.crossing": frozenset({"full", "star"}),
        "ControlDeclaration.stimulus": frozenset({"controlled", "uncontrolled"}),
        "VariationAxis.generator": frozenset({"enum", "sample", "llm"}),
        "ConversationSpec.turn_scheduler": frozenset({"round_robin", "llm_decided"}),
        "RaterKind": values(models.RaterKind),
        "OutOfRunOutcome": values(OutOfRunOutcome),
        "SecondJudgeAskedBy": values(SecondJudgeAskedBy),
        "StopReason": values(StopReason),
    }


def _vocabularies_on_core_shapes() -> dict[str, frozenset[Any]]:
    """Every ``Literal`` and enum a core shape's field carries, at any depth, by ``Model.field``: its values."""
    seen: set[type[BaseModel]] = set()
    for model in _core_models():
        seen.add(model)
        _embedded_models(model, seen)
    found: dict[str, frozenset[Any]] = {}
    for model in seen:
        for name, field in model.model_fields.items():
            stack: list[Any] = [field.annotation]
            while stack:
                annotation = stack.pop()
                if typing.get_origin(annotation) is typing.Literal:
                    found[f"{model.__name__}.{name}"] = frozenset(typing.get_args(annotation))
                elif isinstance(annotation, type) and issubclass(annotation, Enum):
                    found[f"{model.__name__}.{name}"] = frozenset(member.value for member in annotation)
                elif not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
                    stack.extend(typing.get_args(annotation))
    return found


def test_every_vocabulary_on_a_core_shape_is_open_or_listed_closed() -> None:
    """A vocabulary a core shape gains is classified: an open ``str`` with a registry, or closed and listed.

    Every ``Literal`` and enum left on a core shape must be one the versioning docstring names as closed by design,
    by value; a single ``doc_type`` value is a type, never a vocabulary.
    """
    from threetears.evals.schema import versioning
    from threetears.evals.schema.models import ConversationStopCause, JudgedArtifact

    closed = _closed_vocabularies()
    closed["ConversationStopCause"] = frozenset(member.value for member in ConversationStopCause)
    closed["JudgedArtifact"] = frozenset(member.value for member in JudgedArtifact)
    doc_types = {frozenset({doc_type_of(model)}) for model in _core_models()}

    unclassified = {
        where: sorted(values)
        for where, values in _vocabularies_on_core_shapes().items()
        if values not in closed.values() and values not in doc_types
    }
    unlisted = sorted(name for name in closed if f"``{name}``" not in (versioning.__doc__ or ""))

    assert not unclassified, f"a vocabulary on a core shape is neither open nor listed closed: {unclassified}"
    assert not unlisted, f"closed here but not in the versioning docstring's closed list: {unlisted}"


def test_every_open_vocabulary_is_a_string_on_its_stored_fields() -> None:
    """The ten open vocabularies are not ``Literal``s anywhere on a core shape, and each is named in the policy."""
    from threetears.evals.schema import versioning

    registries = {
        "UsageRole": "USAGE_ROLES",
        "SimulatorPurpose": "SIMULATOR_PURPOSES",
        "CostCapOrigin": "COST_CAP_ORIGINS",
        "MeteredCallOrigin": "METERED_CALL_ORIGINS",
        "CellTimeoutOrigin": "CELL_TIMEOUT_ORIGINS",
        "ModelRoleOrigin": "MODEL_ROLE_ORIGINS",
        "RoleModelOrigin": "ROLE_MODEL_ORIGINS",
        "ReasoningEffort": "REASONING_EFFORTS",
        "OutOfRunPurpose": "OUT_OF_RUN_PURPOSES",
        "RefusalKind": "REFUSAL_KINDS",
    }
    import threetears.evals.schema as schema

    still_closed = {
        where
        for where, values in _vocabularies_on_core_shapes().items()
        for registry in registries.values()
        if values == frozenset(getattr(schema, registry))
    }
    assert not still_closed
    policy = " ".join((versioning.__doc__ or "").split())
    for alias, registry in registries.items():
        assert getattr(schema, alias) is str, alias
        assert f"``{alias}`` (``{registry}``)" in policy, alias
