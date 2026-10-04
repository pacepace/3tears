"""The eval engine's own model bases: their stance, pinned, and reads as strict as construction.

The stance is spelled once in ``threetears.evals.contracts.base`` and pinned literally below, so a
change to it is diff-visible on its own terms. Stored eval documents are disposable, so there is no
tolerant read: a stored document carrying a field the model does not declare, or written under
another schema version, is refused — and the last tests here assert that through real storage.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.contracts.base import EvalBaseModel, EvalDocumentModel
from threetears.evals.contracts.models import EVAL_SCHEMA_VERSION, CassetteKey, EvalCassette, EvalResult, EvalRun
from threetears.evals.contracts.storage import EvalStorage
from threetears.evals.contracts.identity import IDENTITY_VERSION
from packages.evals.tests.factories import (
    make_analysis,
    make_analysis_attempt,
    make_campaign,
    make_eval_result,
    make_eval_run,
    make_eval_trace,
    make_insight,
    make_judge_config,
    make_rubric_dim,
    make_template,
    make_test_case,
)
from packages.evals.tests.memory_store import InMemoryDocumentStore, memory_storage
from packages.evals.tests.stored_models import stored_models


class _Sample(EvalBaseModel):
    """A minimal model on the eval base, for the round-trip and stance tests."""

    name: str
    note: str = ""


class _Payload(EvalBaseModel):
    """A model with a free-form field — where externally-ingested text actually lands.

    The distinction matters for the surrogate tests below and is not obvious: a
    declared ``str`` field cannot hold a lone surrogate at all, because pydantic
    refuses it during *validation*, before serialization is ever reached. So the
    population the ``to_json`` fallback exists for is the free-form one — the
    ``dict``/``Any`` fields eval models use for provider payloads, scraped text and
    stored blobs.
    """

    blob: dict[str, Any] = {}


def test_the_base_pins_its_configuration_values_literally():
    """Eval's stance is asserted here, not inherited from the host's base.

    Each value carries a consequence, so a future edit has to argue with the
    consequence rather than with a preference:

    - ``extra="forbid"`` — an unknown key is a construction-time failure. It is
      a decision the first cross-repo consumer has to revisit, because a strict
      receiver refusing documents written by a newer peer is a total-refusal outage.
      It is safe while one version writes and reads every stored eval document.
    - ``validate_assignment=True`` — extends that to mutation, which is where a
      long-lived run record spends its life.
    - ``str_strip_whitespace=True`` — why whitespace-significant eval models are on
      plain ``BaseModel`` instead of this base.
    - ``ser_json_inf_nan="constants"`` — eval measures produce infinities and NaN,
      and the alternative serialises them as ``null``, which reads as "not measured".
    - ``populate_by_name=True`` — an aliased field is constructible by either name.

    The assertion is over the RESOLVED config rather than the declared options, so
    ``validate_by_alias``/``validate_by_name`` appear: pydantic expands
    ``populate_by_name`` into them. Pinning the resolved form means a pydantic
    release that changes that expansion shows up here as a failing equality rather
    than as a silent change in what eval's models accept.
    """
    assert EvalBaseModel.model_config == {
        "extra": "forbid",
        "validate_assignment": True,
        "str_strip_whitespace": True,
        "ser_json_inf_nan": "constants",
        "populate_by_name": True,
        "validate_by_alias": True,
        "validate_by_name": True,
    }


def test_the_document_base_derives_from_the_same_options_and_names_only_its_deltas():
    """The stance has one spelling, and this is what keeps it that way.

    ``EvalDocumentModel`` is the base of nearly every stored eval document — ``EvalRun``,
    ``EvalResult``, ``EvalTemplate``, ``EvalTrace`` and all of ``analysis/`` — while the pin test
    above asserts over ``EvalBaseModel`` alone. A document base that re-typed those values would
    take an edit to the option table and apply it to the minority, silently, with that test still
    green. So this asserts the derivation: every option keeps the table's value except the one
    delta named here. **``extra`` is not a delta** — the document base refuses unknown keys on
    every path, reads included.

    Read through the front door: the table's resolved form is ``EvalBaseModel``'s config, and the
    document base's is that of ``EvalRun``, a stored document on it.
    """
    deltas = {"json_schema_serialization_defaults_required": True}
    options = EvalBaseModel.model_config
    document = EvalRun.model_config

    assert issubclass(EvalRun, EvalDocumentModel)
    assert {k: document[k] for k in deltas} == deltas
    assert not deltas.keys() & options.keys(), "a delta is now in the base's own table, so it is no longer a delta"
    assert {k: v for k, v in options.items() if k not in deltas}.items() <= document.items(), (
        "the document base no longer carries the base's option table — if it now spells those values "
        "itself, an edit to the table reaches the strict models and skips the stored ones"
    )


def test_a_lone_surrogate_serialises_instead_of_raising():
    """The serialisation fallback, and the reason it is not decoration.

    Externally-ingested data reaches these models — scraped strings, provider
    responses — and a lone UTF-16 surrogate is unencodable in UTF-8. Without the
    fallback this raises not here but in whichever consumer first stores such a
    string.

    ``model_dump_json`` is asserted to raise first, because that is the only thing
    that makes the rest of this test about the fallback: on a model where the plain
    path succeeds, every assertion below would pass with the fallback deleted.
    """
    model = _Payload(blob={"scraped": "before\ud800after"})

    with pytest.raises(ValueError):
        model.model_dump_json()

    payload = model.to_json()

    # It round-trips as JSON at all, which is the property that was at risk.
    assert json.loads(payload)["blob"]["scraped"] == "before�after"
    # And the damage is visible in the document rather than silently dropped.
    assert "\ud800" not in payload


def test_the_dict_path_passes_a_lone_surrogate_through_unrepaired():
    """``to_dict`` is the other entry point, and it deliberately does not sanitize.

    ``model_dump(mode="json")`` produces python objects and never encodes, so there
    is nothing to fail and no repair to make — the surrogate survives intact. Pinned
    because the asymmetry is a real property callers depend on rather than an
    oversight: a caller that hands ``to_dict`` output to a JSON encoder itself is
    the one that meets the error, and ``to_json`` is the method that handles it.
    """
    model = _Payload(blob={"scraped": "before\ud800after"})

    assert model.to_dict()["blob"]["scraped"] == "before\ud800after"


def test_an_unknown_field_is_refused():
    """The ``forbid`` stance still holds after the copy."""
    with pytest.raises(ValidationError, match="unexpected_keyword_argument|extra_forbidden"):
        _Sample(name="ok", unknown_field="value")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        _Sample.from_dict({"name": "ok", "unknown_field": "value"})


def test_assignment_is_validated():
    """``validate_assignment`` is the half of the stance a construction test misses."""
    model = _Sample(name="ok")

    with pytest.raises(ValidationError):
        model.name = 123


def test_the_json_pair_round_trips():
    """``to_json``/``from_json`` compose, which is what callers actually rely on."""
    model = _Sample(name="ok", note="hello")

    assert _Sample.from_json(model.to_json()) == model


def test_the_dict_pair_round_trips():
    """``to_dict``/``from_dict`` compose the same way."""
    model = _Sample(name="ok", note="hello")

    assert _Sample.from_dict(model.to_dict()) == model


def test_the_engine_models_are_on_the_engine_base():
    """The requirement's property, asserted on real models rather than on the base.

    Every test above would pass with nothing repointed. This one is what says the
    requirement is met: the engine's models derive from eval's base, and a repoint that
    reverted would fail here rather than only shrinking a register.
    """
    from threetears.evals.analysis.reporting import ScoreRecord
    from threetears.evals.analysis.viz.models import FindingChart
    from threetears.evals.contracts.identity import DerivedContextIdentity
    from threetears.evals.contracts.metrics import MetricDescriptor
    from threetears.evals.contracts.models import EvalRun
    from threetears.evals.contracts.result_condition import ResultCondition
    from threetears.evals.contracts.usage_capture import ResolvedUsage

    for cls in (
        DerivedContextIdentity,
        MetricDescriptor,
        EvalRun,
        ScoreRecord,
        ResultCondition,
        ResolvedUsage,
        FindingChart,
    ):
        assert issubclass(cls, EvalBaseModel), f"{cls.__name__} is not on the eval base"


# =============================================================================
# Reads are strict: what storage hands back is validated exactly as construction is
# =============================================================================


def _stored_result() -> tuple[EvalResult, EvalStorage, InMemoryDocumentStore]:
    """One result saved to an in-memory store, with the storage and the raw store.

    Returns:
        The result, the ``EvalStorage`` over the store, and the store itself.
    """
    storage, store = memory_storage()
    result = EvalResult(
        scope_id="scope-1",
        eval_run_id="run-1",
        test_case_id="tc-1",
        model="m",
        k_iteration=1,
        termination="completed",
        cost_usd=0.0,
        cost_roles=["candidate", "inner_agent", "judge", "simulator"],
        usage=[],
        covariates={},
        phase_timings={},
        host_measures={},
        variant_key="vk-1",
        identity_version=IDENTITY_VERSION,
    )
    storage.save_eval_result(result)
    return result, storage, store


def _tamper(store: InMemoryDocumentStore, result: EvalResult, **fields: object) -> None:
    """Rewrite the stored document with ``fields`` merged in, as an older or foreign writer would.

    Args:
        store: The in-memory store.
        result: The result whose document to rewrite.
        **fields: Keys to set on the stored document.
    """
    document = store.get(result.id, result.scope_id)
    assert document is not None, "the result was not stored"
    store.upsert({**document, **fields})


def test_a_stored_result_reads_back_whole():
    """The control: an untampered document loads, so the refusals below are about the tampering."""
    result, storage, _ = _stored_result()

    assert storage.load_eval_result(result.id, result.scope_id) == result


def test_a_stored_field_the_model_does_not_declare_is_refused():
    """No tolerant read: a document carrying a key this build does not declare does not load."""
    result, storage, store = _stored_result()
    _tamper(store, result, retired_lifecycle=[])

    with pytest.raises(ValidationError, match="retired_lifecycle"):
        storage.load_eval_result(result.id, result.scope_id)


def test_a_stored_nested_field_the_model_does_not_declare_is_refused():
    """The refusal reaches every level, not only the top-level document."""
    result, storage, store = _stored_result()
    _tamper(store, result, usage=[{"role": "candidate", "model": "m", "credits": 3}])

    with pytest.raises(ValidationError, match="credits"):
        storage.load_eval_result(result.id, result.scope_id)


@pytest.mark.parametrize("version", [EVAL_SCHEMA_VERSION - 1, EVAL_SCHEMA_VERSION + 1])
def test_a_document_written_under_another_schema_version_is_refused(version: int):
    """A shape-compatible document from another schema still does not load: the version says it is not this one."""
    result, storage, store = _stored_result()
    _tamper(store, result, schema_version=version)

    with pytest.raises(ValidationError, match=f"eval schema v{version}"):
        storage.load_eval_result(result.id, result.scope_id)


# =============================================================================
# ...and both refusals hold for every stored model, not only the one above
# =============================================================================


def _cassette() -> EvalCassette:
    """One action-seam recording, built the way the cassette proxy builds one."""
    key = CassetteKey(
        corpus_id="run-capture",
        template_id="tpl-1",
        test_case_id="tc-1",
        tool="search",
        action="query",
        params_hash="0" * 16,
        occurrence=0,
    )
    return EvalCassette.build(key, scope_id="uni-1", seam="action", captured_model="m", response={"hits": []})


#: A valid instance of every stored model, by name. Keyed by the derived population below, so a
#: stored model added without a row here fails :func:`test_every_stored_model_has_a_sample` rather
#: than slipping past both refusals.
_SAMPLES: dict[str, Callable[[], EvalBaseModel]] = {
    "CatalogRubricDim": make_rubric_dim,
    "EvalAnalysis": make_analysis,
    "EvalAnalysisAttempt": make_analysis_attempt,
    "EvalCampaign": make_campaign,
    "EvalCassette": _cassette,
    "EvalInsight": make_insight,
    "EvalResult": make_eval_result,
    "EvalRun": make_eval_run,
    "EvalTemplate": make_template,
    "EvalTestCase": make_test_case,
    "EvalTrace": make_eval_trace,
    "JudgeConfig": make_judge_config,
}


def _stored_document(model: type[EvalBaseModel]) -> dict[str, Any]:
    """A stored document of ``model``, as the store hands it back, after checking it loads untampered."""
    sample = _SAMPLES[model.__name__]()
    assert type(sample) is model, f"the sample for {model.__name__} is a {type(sample).__name__}"
    document = sample.to_dict()
    assert model.from_dict(document) == sample, "the control: an untampered document reads back whole"
    return document


def test_every_stored_model_has_a_sample() -> None:
    """The population is derived; this keeps the sample table from silently trailing it."""
    assert sorted(_SAMPLES) == [model.__name__ for model in stored_models()]


@pytest.mark.parametrize("model", stored_models(), ids=lambda model: model.__name__)
def test_every_stored_model_refuses_a_field_it_does_not_declare(model: type[EvalBaseModel]) -> None:
    """A stored document carrying an undeclared key does not load, whichever model it is.

    Read through ``from_dict``, which is every storage load path (``EvalStorage._hydrate``), so a
    model that relaxed its ``extra`` stance on its own would be named here.
    """
    document = _stored_document(model)

    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.from_dict({**document, "written_by_a_newer_build": True})


@pytest.mark.parametrize("version", [EVAL_SCHEMA_VERSION - 1, EVAL_SCHEMA_VERSION + 1])
@pytest.mark.parametrize("model", stored_models(), ids=lambda model: model.__name__)
def test_every_stored_model_refuses_a_document_from_another_schema_version(
    model: type[EvalBaseModel], version: int
) -> None:
    """A stored document from another schema does not load, whichever model it is."""
    document = _stored_document(model)

    with pytest.raises(ValidationError, match=f"eval schema v{version}"):
        model.from_dict({**document, "schema_version": version})
