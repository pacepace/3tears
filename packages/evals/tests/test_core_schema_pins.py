"""The evidence core's shape is pinned: a core shape cannot change without a version bump and an upgrader.

``schema/versioning.py`` promises that a core document is read forever from the baseline. This holds that
promise from both ends: every core type's validation JSON schema is pinned to a digest under the version
recorded here, so a shape change without a bump goes red here; and a frozen corpus of stored core
documents, one directory per core version, must keep reading under the current build, which also catches
a validator tightening the JSON schema cannot show (the goal grammar, an ``AfterValidator``).

**When this goes red**, one of these happened:

- **A core shape moved and the version did not.** Any change to a core type's shape is a bump, an added
  optional field included. Before changing the shape, copy ``tests/fixtures/core_documents/v{N}/`` to
  ``v{N+1}/`` (the old directory stays as it is: it is what an old store holds). Then bump
  ``CORE_SCHEMA_VERSION`` with a ledger line above it, append one ``CoreUpgrader`` from ``N`` (an identity
  step for a purely additive change), re-pin the digests and the version here, and regenerate the new
  directory's fixtures from :func:`~packages.evals.tests.stored_models.stored_sample`, pinning their files.
  Give the step its own test of the fields it transforms.
- **The version moved and the pins did not.** Re-pin every digest under the new version, even where a
  shape held, so this file states what it was checked under.
- **The version moved and the ledger or the chain has no entry for it.** Write the ledger line, and the
  upgrader.
- **A frozen fixture stopped reading.** The current build no longer reads a document an older build
  stored. Either a validator tightened (a bump and a step that brings old documents into line, or a None
  for what cannot be derived), or a step is wrong. Never edit the fixture: its digest is pinned so it
  cannot be.
- **A frozen fixture's digest moved.** Someone edited a stored document of the past. Restore it.
- **The core's membership moved.** A type became core or stopped being core; that is a decision about
  what is kept, recorded in ``CORE_DOC_TYPES`` and here together.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.stored_models import doc_type_of, stored_models
from threetears.evals.schema import versioning
from threetears.evals.schema.base import CoreDocumentModel, EvalBaseModel
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.versioning import (
    CORE_ADDRESSING_FIELDS,
    CORE_BASELINE_VERSION,
    CORE_DOC_TYPES,
    CORE_SCHEMA_VERSION,
    CORE_UPGRADERS,
)

#: The ``CORE_SCHEMA_VERSION`` every digest below was taken under.
PINNED_CORE_SCHEMA_VERSION = 14

#: Each core type's validation JSON schema, normalised (:func:`core_shape_digest`), by ``doc_type``.
CORE_SHAPE_DIGESTS: dict[str, str] = {
    "calibration_rating": "825e3639a2174894ed356b5f8a75feeac145698e3cf2bb0a91af59ab38fd9342",
    "case_set": "e8b3eab78db2316c8eb8295d41b1c937a9e1ed5381b91dad7870715251d9e46e",
    "eval_action_job": "304e5c7845d124355c72ec9e42e2647513154766d4a66960e1fc606b82f8e5c4",
    "eval_analysis_attempt": "6e8b0c20ff14d13beaa77535dc0b597e8502d02b93f91ad3b6985c24a482b703",
    "eval_campaign": "4dd5daa94ed5c7298cda14cca45e08e7b4582ac3a5ff96849b0a30fe75dcccd0",
    "eval_generation_slot": "f5782b025d43c87474c10d1ccb426a3e1129a0d61043c42368959dfac64c2170",
    "eval_judge_temperature_check": "deff77f10ba6db35ed004a770079f72607ce09ef445cd920245ab1a313e0cf27",
    "eval_launch": "ccda7572311aa02dabf5790b99891182c53b8f5d9ec4eb4d6affbf0554ecbb21",
    "eval_out_of_run_spend": "27225b5f04091b13f2ea49337f8a03eac5ff6e63101919f396fd2d54cd32d9b7",
    "eval_result": "399632f58494fb6a7d94e83bafeee3ceb4208c3c15280ef9415a5addf4d70c30",
    "eval_run": "bf267da62aceecfb56e5e95f05109ecb2101184559a7f8f0315eff97606a93ec",
    "eval_second_judge_pass": "9a5ba57bff5e0c27841e80b97557e31563af1fd2b25e6f2c365e1eb5da644cb7",
    "eval_sweep": "1fb99c3636c18ebd176e2e5d6b12c74533e955123c2249d2f78b1185421e9e08",
    "eval_template": "e13e5a3192984be060d15e90de46b4dac81ab20d32553c349d57940c521adc92",
    "eval_test_case": "75ae7a69db2cfa81f7fbeff2f4066ee8f0a9ffcf91bb26830e7d929832eba6f0",
    "eval_trace": "12c45a35fe0445e86b4a84897065fb5982565e87164c9cb57810ba0594241563",
    "judge_config": "c83905f8a12b90787d56b4a9893082c52ded9501ab49b183a1848255560ab856",
    "judge_config_tombstone": "04710a7d9aa03d663fbbee62281d4a91b360fd81578bd99ecd5317008378ef57",
    "rubric_dim": "4d07547ddd66e10f5217f515b6439ba3d72c927674e0a1b32cfeb5878a4035d5",
    "rubric_dim_tombstone": "99d16d84a4918f8c0e41d4b13ff57077c01b890c55c9dd463d421e16f15fe8fc",
}

#: The fields the store addresses each core type by, pinned: a step may never touch one, so the set moving
#: is a decision about the store, never a side effect.
PINNED_ADDRESSING_FIELDS: dict[str, frozenset[str]] = {
    "eval_test_case": frozenset({"id", "doc_type", "schema_version", "scope_id", "template_id", "stratum"}),
    "eval_run": frozenset(
        {
            "id",
            "doc_type",
            "schema_version",
            "scope_id",
            "status",
            "archived",
            "created_at",
            "cancel_requested_at",
            "cancel_request_reason",
        }
    ),
    "eval_result": frozenset({"id", "doc_type", "schema_version", "scope_id", "eval_run_id", "test_case_id", "model"}),
    "eval_trace": frozenset({"id", "doc_type", "schema_version", "scope_id"}),
    "calibration_rating": frozenset(
        {
            "id",
            "doc_type",
            "schema_version",
            "scope_id",
            "run_id",
            "result_id",
            "rated_at",
            "output_fingerprint",
            "criterion_fingerprint",
        }
    ),
    "eval_out_of_run_spend": frozenset(
        {"id", "doc_type", "schema_version", "scope_id", "purpose", "launch_group_id", "template_id", "created_at"}
    ),
    "eval_template": frozenset({"id", "doc_type", "schema_version", "scope_id", "name", "archived", "universal"}),
    "judge_config": frozenset(
        {"id", "doc_type", "schema_version", "scope_id", "rubric_dim_id", "archived", "created_at"}
    ),
    "rubric_dim": frozenset({"id", "doc_type", "schema_version", "scope_id", "key", "archived", "created_at"}),
    "rubric_dim_tombstone": frozenset({"id", "doc_type", "schema_version", "scope_id", "deleted_at"}),
    "judge_config_tombstone": frozenset({"id", "doc_type", "schema_version", "scope_id", "deleted_at"}),
    "case_set": frozenset({"id", "doc_type", "schema_version", "scope_id", "name", "version"}),
    "eval_campaign": frozenset(
        {"id", "doc_type", "schema_version", "scope_id", "subject_id", "behavior", "archived", "created_at"}
    ),
    "eval_sweep": frozenset(
        {"id", "doc_type", "schema_version", "scope_id", "cancel_requested_at", "cancel_request_reason"}
    ),
    "eval_generation_slot": frozenset({"id", "doc_type", "schema_version", "scope_id", "campaign_id", "attempt_id"}),
    **{
        job_record: frozenset(
            {"id", "doc_type", "schema_version", "scope_id", "cancel_requested_at", "cancel_request_reason"}
        )
        for job_record in ("eval_launch", "eval_second_judge_pass", "eval_judge_temperature_check", "eval_action_job")
    },
    "eval_analysis_attempt": frozenset(
        {
            "id",
            "doc_type",
            "schema_version",
            "scope_id",
            "campaign_id",
            "created_at",
            "cancel_requested_at",
            "cancel_request_reason",
        }
    ),
}

#: What a step may never touch beyond the addressing fields: the host's and the kind's opaque payloads, and
#: every stored key or digest (an upgrader never recomputes identity).
UNTOUCHABLE_FIELDS = frozenset(
    {
        "host_payload",
        "kind_payload",
        "variant_key",
        "context_key",
        "context_components",
        "content_hash",
        "identity_version",
    }
)

#: The frozen corpus: one directory per core version, one stored document per core type.
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "core_documents"

#: A core type that did not exist before some version, by the version it was first stored at: no store holds one
#: from before it, so its frozen corpus starts there. A type that existed and joined the core later is not here —
#: an older store holds it, and its corpus starts at the baseline.
FIRST_STORED_AT: dict[str, int] = {}

#: Each frozen fixture's sha256, so a stored document of the past is never edited.
FIXTURE_DIGESTS: dict[str, str] = {
    "v14/calibration_rating.json": "afaccf4bb7a0f4f7a3f3f60d2a625b5a7ee672adc5c04c1a223f308adad5c8b9",
    "v14/case_set.json": "132b511a0d6934be20e4746571476cfa75a2754a6dfbc3067b9cb0c556947399",
    "v14/eval_action_job.json": "f20f12da3b15b601f3b65e65bcb948c6496546c7c62502f01037fc176520b2ef",
    "v14/eval_analysis_attempt.json": "226a09c4dc926bd5d03813c468e2d26c6877ee67b88422ddde1432210f6824f1",
    "v14/eval_campaign.json": "458a8542f18e444a056f2d70026791256af53942fc3ed9d2813d8305d4f43c50",
    "v14/eval_generation_slot.json": "d93c4447232d592abb3390052898cf1602aad991785e4deb302fe071f5466c0b",
    "v14/eval_judge_temperature_check.json": "dcae09b0fc76e8897ee4a68c45f7bd7ce230a9366370734ad68611b7bb06218a",
    "v14/eval_launch.json": "f6f4600e9a70da1e15707495115d199d17eb87179f84f745be3fe0e2ac275b01",
    "v14/eval_out_of_run_spend.json": "6623f165748aae76eb0c6a9a17cfaece3d5556c1085ed40b6632804558f2798a",
    "v14/eval_result.json": "aea1db47349557046b84370238f472cbe19ed5970962d5c9460a8cefb1bdf4e5",
    "v14/eval_run.json": "39b26d720d2ff5363bec8bc55245fb4572ab33041555cc1eaba9cff06228ae62",
    "v14/eval_second_judge_pass.json": "f257b8c8ad5cc49763eb8db1b753941d0415936322e4c60e88e45dcf8a4176f0",
    "v14/eval_sweep.json": "9f3bd42e3de42dfbcdbb8f2e1ec2bfa21ec89fb94cd6e2dc9954095cbd52b5ee",
    "v14/eval_template.json": "2eef3600943202f19c1a6189fa617ea4fac7e6dab3657c3f0d021f5e980421c3",
    "v14/eval_test_case.json": "d3d4483e951f9e02b8ecb5c9c82d51c40864a64f1c59ccda837cf10e188b6675",
    "v14/eval_trace.json": "b1bf582cbe1910a29e9c57fb27e511a2f3d1de19f79b7df7a8725ba572db6366",
    "v14/judge_config.json": "9d93ac9f3c00656c6047d40f657f8cdbc09ee7d5c5d419a875b5a6fe2c047b51",
    "v14/judge_config_tombstone.json": "abb15de0aaf8f4f9d358c08cf29c416a2e980defb4b99e70b5e4659e81b45b8b",
    "v14/rubric_dim.json": "c3611665659943d198ed8c1d4b541bf952215a31aa8f44df3771190f8a1b175b",
    "v14/rubric_dim_tombstone.json": "0a64732f5f037c1c74c3dcf119f3e9cd81cd473c2c686350c37476e63058703f",
}

#: Keys whose dict values are maps of names to schemas, not schema nodes: their own keys are field or
#: definition names (a field named ``title`` must survive), and each value is a schema node.
_NAMED_SCHEMAS = frozenset({"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"})

#: Keys whose values are data, not schema nodes, kept exactly as written.
_DATA = frozenset({"default", "const", "enum"})

#: Annotations a schema node carries for a reader, which change no document's validity.
_ANNOTATIONS = frozenset({"description", "title", "examples"})


def _normalised(node: Any) -> Any:
    """``node`` with the reader annotations dropped at schema-node level, and nothing else touched."""
    if isinstance(node, list):
        return [_normalised(item) for item in node]
    if not isinstance(node, dict):
        return node
    kept: dict[str, Any] = {}
    for key, value in node.items():
        if key in _ANNOTATIONS:
            continue
        if key in _NAMED_SCHEMAS and isinstance(value, dict):
            kept[key] = {name: _normalised(schema) for name, schema in value.items()}
        elif key in _DATA:
            kept[key] = value
        else:
            kept[key] = _normalised(value)
    return kept


def core_shape_digest(model: type[EvalBaseModel]) -> str:
    """The digest of a core type's validation JSON schema, without descriptions, titles or examples.

    Args:
        model: A core stored model.

    Returns:
        A sha256 hex digest; it moves when what the model accepts moves, and not when its prose does.
    """
    return canonical_digest(_normalised(model.model_json_schema(mode="validation")))


def _core_models() -> dict[str, type[EvalBaseModel]]:
    return {doc_type_of(model): model for model in stored_models() if issubclass(model, CoreDocumentModel)}


def _fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("v*/*.json"))


# --- the shape -------------------------------------------------------------------------------------


def test_the_pins_were_taken_under_this_build_s_core_version() -> None:
    assert PINNED_CORE_SCHEMA_VERSION == CORE_SCHEMA_VERSION, (
        "CORE_SCHEMA_VERSION moved: re-pin every digest under the new version (see the module docstring)"
    )


@pytest.mark.parametrize("doc_type", sorted(CORE_SHAPE_DIGESTS))
def test_each_core_shape_matches_its_pin(doc_type: str) -> None:
    """A core type that accepts something new, or refuses something it read, is a bump."""
    model = _core_models()[doc_type]
    assert core_shape_digest(model) == CORE_SHAPE_DIGESTS[doc_type], (
        f"{model.__name__} ({doc_type}) changed shape under core v{CORE_SCHEMA_VERSION}. A core shape change is a "
        "version bump with an upgrader, an added optional field included; see this module's docstring"
    )


def test_an_added_optional_field_moves_a_pin() -> None:
    """The case the pin exists for: an OPTIONAL field joining a core type is a shape change, and is seen."""
    from pydantic import create_model

    run = _core_models()["eval_run"]
    grown = create_model("EvalRun", __base__=run, added_note=(str | None, None))

    assert core_shape_digest(run) == CORE_SHAPE_DIGESTS["eval_run"]
    assert core_shape_digest(grown) != CORE_SHAPE_DIGESTS["eval_run"]


def test_the_digest_ignores_prose_and_keeps_names() -> None:
    """Descriptions do not move a pin; a field literally named ``title`` and every default do."""
    schema = {
        "title": "M",
        "description": "prose",
        "properties": {"title": {"type": "string", "description": "a field named title"}, "n": {"default": 1}},
    }
    assert _normalised(schema) == {"properties": {"title": {"type": "string"}, "n": {"default": 1}}}


def test_the_core_is_exactly_the_core_models() -> None:
    """Every stored model on the core base is in CORE_DOC_TYPES, and every core type is on the core base."""
    assert set(_core_models()) == CORE_DOC_TYPES
    assert set(CORE_SHAPE_DIGESTS) == CORE_DOC_TYPES


def test_the_addressing_fields_are_pinned_and_are_fields() -> None:
    assert dict(CORE_ADDRESSING_FIELDS) == PINNED_ADDRESSING_FIELDS
    for doc_type, model in _core_models().items():
        missing = CORE_ADDRESSING_FIELDS[doc_type] - set(model.model_fields)
        assert not missing, f"{model.__name__} has no field {sorted(missing)} the store addresses it by"


# --- the version and its chain ---------------------------------------------------------------------

#: A ledger entry, as the comment block above ``CORE_SCHEMA_VERSION`` writes one: ``#: - **v8** — …``.
_LEDGER_ENTRY = re.compile(r"^#: - \*\*v(\d+)\*\*", re.MULTILINE)


def test_every_core_version_has_a_ledger_entry_and_an_upgrader() -> None:
    """Every version from the baseline is explained, and every step between two of them is registered."""
    source = inspect.getsource(versioning)
    ledger = source[source.index("CORE_BASELINE_VERSION: Final") : source.index("CORE_SCHEMA_VERSION: int =")]
    recorded = {int(version) for version in _LEDGER_ENTRY.findall(ledger)}
    assert set(range(CORE_BASELINE_VERSION, CORE_SCHEMA_VERSION + 1)) <= recorded
    assert [step.from_version for step in CORE_UPGRADERS] == list(range(CORE_BASELINE_VERSION, CORE_SCHEMA_VERSION))
    assert all(step.doc_types <= CORE_DOC_TYPES and step.reason for step in CORE_UPGRADERS)


# --- the frozen corpus -----------------------------------------------------------------------------


def test_every_core_version_has_a_whole_frozen_corpus() -> None:
    """One stored document per core type, for every version from the baseline (or from the version a type was
    first stored at): what an old store holds."""
    assert all(
        doc_type in CORE_DOC_TYPES and CORE_BASELINE_VERSION < version <= CORE_SCHEMA_VERSION
        for doc_type, version in FIRST_STORED_AT.items()
    )
    for version in range(CORE_BASELINE_VERSION, CORE_SCHEMA_VERSION + 1):
        held = {path.stem for path in (FIXTURES / f"v{version}").glob("*.json")}
        stored_then = {doc_type for doc_type in CORE_DOC_TYPES if FIRST_STORED_AT.get(doc_type, 0) <= version}
        assert held == stored_then, f"v{version}'s frozen corpus holds {sorted(held)}"


def test_no_frozen_fixture_was_edited() -> None:
    found = {
        path.relative_to(FIXTURES).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _fixture_files()
    }
    assert found == FIXTURE_DIGESTS, "a stored document of the past was edited, added or removed; restore it"


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda path: path.relative_to(FIXTURES).as_posix())
def test_every_frozen_core_document_reads_under_this_build(path: Path) -> None:
    """A core document any earlier build stored still reads, at the current version."""
    document = json.loads(path.read_text(encoding="utf-8"))
    model = _core_models()[document["doc_type"]]

    read = model.from_dict(document)

    assert read.schema_version == CORE_SCHEMA_VERSION  # type: ignore[attr-defined]
    if document["schema_version"] == CORE_SCHEMA_VERSION:
        assert read.to_dict() == document, "a document at the current version reads back whole"


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda path: path.relative_to(FIXTURES).as_posix())
def test_a_frozen_document_written_before_its_optional_fields_still_reads(path: Path) -> None:
    """A document stored before an optional field existed lacks the key; it reads, with the field's default."""
    document = json.loads(path.read_text(encoding="utf-8"))
    model = _core_models()[document["doc_type"]]
    optional = {name for name, field in model.model_fields.items() if not field.is_required()}
    addressed = CORE_ADDRESSING_FIELDS[document["doc_type"]]

    sparse = {key: value for key, value in document.items() if key not in optional or key in addressed}

    assert model.from_dict(sparse).schema_version == CORE_SCHEMA_VERSION  # type: ignore[attr-defined]


def test_every_registered_step_keeps_to_the_step_rules() -> None:
    """On every frozen document at its version, a step leaves what the store reads and every key untouched.

    Empty at the baseline, where no step is registered; the mechanism tests hold the runner to the same
    rules with synthetic steps.
    """
    for step, path in (
        (step, path) for step in CORE_UPGRADERS for path in sorted((FIXTURES / f"v{step.from_version}").glob("*.json"))
    ):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["doc_type"] not in step.doc_types:
            continue
        protected = CORE_ADDRESSING_FIELDS[document["doc_type"]] | UNTOUCHABLE_FIELDS
        upgraded = step.upgrade(dict(document))
        moved = {key for key in protected if upgraded.get(key) != document.get(key)}
        assert moved <= {"schema_version"}, f"the v{step.from_version} step touched {sorted(moved)} in {path.name}"
        assert step.upgrade(dict(document)) == upgraded, "a step is pure: the same input gives the same output"
