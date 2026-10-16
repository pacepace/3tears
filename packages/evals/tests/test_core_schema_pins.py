"""The evidence core's shape is pinned: a core shape cannot change without a minor or a major, as the change is.

``schema/versioning.py`` promises that a core document of this build's major is read, a newer minor of it included,
and that the core ships no upgrader: a core major is a baseline reset. This holds the first promise from both ends: every core type's validation JSON schema
is pinned to a digest under the major and minor recorded here, so a shape change without a version goes red here; and
a frozen corpus of stored core documents, one directory per core major, must keep reading under the current build,
which also catches a validator tightening the JSON schema cannot show (the goal grammar, an ``AfterValidator``).

**When this goes red**, one of these happened:

- **A core shape moved and the version did not.** Any change to a core type's shape is a version, an added
  optional field included. If it is additive only (a new optional field with a default, a new optional
  sub-object, a new member of an open vocabulary, a new doc type) it is a **minor**: bump ``CORE_SCHEMA_MINOR``
  with a ledger line ending in ``MINOR_ADDITIVITY_SENTENCE``, and re-pin the digests and the minor here. Nothing
  else is a minor: anything else is a **major**, which for the core is a baseline reset with no upgrader. Bump
  ``CORE_SCHEMA_VERSION`` and ``CORE_BASELINE_VERSION`` together with a ledger line above them, reset
  ``CORE_SCHEMA_MINOR`` to 0, replace ``tests/fixtures/core_documents/v{N}/`` with ``v{N+1}/`` (a reset refuses and
  deletes every older core document, so no build reads the old corpus), re-pin the digests and the version here, and
  pin the new fixtures (``regenerate_core_fixtures.py``, below). Its release notes tell a host to delete its core
  documents before deploying (``EvalStorage.reset_core``).
- **The version moved and the pins did not.** Re-pin every digest under the new major or minor, even where a
  shape held, so this file states what it was checked under.
- **The version moved and the ledger has no entry for it.** Write the ledger line (a minor's carrying the
  additivity sentence), and for a major move the baseline with it.
- **A frozen fixture stopped reading.** The current build no longer reads a document an older build
  stored at this major. A validator tightened, which is a major (a baseline reset). Never edit the fixture: its
  digest is pinned so it cannot be.
- **A frozen fixture's digest moved.** Someone edited a stored document of the past. Restore it.
- **The core's membership moved.** A type became core or stopped being core; that is a decision about
  what is kept, recorded in ``CORE_DOC_TYPES`` and here together. The eight types people author left the core
  for the durable space at ``.26`` (``tests/test_durable_schema_pins.py``), and their ``.25`` fixtures moved with
  them, unedited, to ``tests/fixtures/durable_documents/v15/``.

The v15 corpus is the baseline of a release that has not been published: no store a released build wrote holds a
v15 document yet, so while that is true a baseline change regenerates it in place rather than copying it. (The v14
corpus it replaced was the pre-release baseline of ``.21`` to ``.24``, dropped with no upgrader when ``.25`` reset
the baseline, since no release reads a pre-release store.) **A
released version's corpus is never regenerated**: it is what an old store holds, and rewriting it would make the test
that reads it vouch for the current build instead of for the stored past.

**Regenerating the current corpus** is one command, from the repo root, which writes
``tests/fixtures/core_documents/v{CORE_SCHEMA_VERSION}/`` from :func:`~packages.evals.tests.stored_models.stored_sample`
and re-pins ``CORE_SHAPE_DIGESTS`` and ``FIXTURE_DIGESTS`` below (it leaves ``PINNED_CORE_SCHEMA_VERSION`` and
``PINNED_CORE_SCHEMA_MINOR`` to the author)::

    uv run python packages/evals/scripts/regenerate_core_fixtures.py

It refuses when a ``v<N>`` directory newer than the current version exists, or when the newest final release's tree
already holds the current version's directory (that corpus shipped), and on an unchanged tree it changes nothing.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.schema_pin_support import keeps, minor_entries, minors_missing_the_sentence
from packages.evals.tests.stored_models import doc_type_of, stored_models
from threetears.evals.schema import versioning
from threetears.evals.schema.base import CoreDocumentModel, EvalBaseModel
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.versioning import (
    CORE_ADDRESSING_FIELDS,
    CORE_BASELINE_VERSION,
    CORE_DOC_TYPES,
    CORE_SCHEMA_MINOR,
    CORE_SCHEMA_VERSION,
    CORE_VERSION_SPACE,
    MINOR_ADDITIVITY_SENTENCE,
)

#: The ``CORE_SCHEMA_VERSION`` every digest below was taken under.
PINNED_CORE_SCHEMA_VERSION = 15

#: The ``CORE_SCHEMA_MINOR`` every digest below was taken under.
PINNED_CORE_SCHEMA_MINOR = 4

#: Each core type's validation JSON schema, normalised (:func:`core_shape_digest`), by ``doc_type``.
CORE_SHAPE_DIGESTS: dict[str, str] = {
    "eval_action_job": "61a2e6c5cf60f39eb3251c8b4a14bd7f12262d8af6bc0de6271b534182c08641",
    "eval_analysis_attempt": "85e61884a619f0237a011392c01d6c6f5d39758a63baf06c68b35b48ae525022",
    "eval_campaign": "f82345b9daf72ae86c65be7d533deaf586cc8ebe7b9ad52e729733751863a9bb",
    "eval_generation_slot": "23007ca5ca66892b4b0e4d6f9706ef6be1308178d6d4b6b84504969bf4ad046f",
    "eval_judge_temperature_check": "13dff1159e165b0f4b0a58e447fcf59d8c0063092c0f86c26056f50496aaea73",
    "eval_launch": "de6189cbac75fb150ec2abb10bce3cd80eb0cbb117236e09cd6617050c0aeb73",
    "eval_out_of_run_spend": "db695734cf91e321ab056d563e0a81236094749c507fdcb9cf7f9be575ab7933",
    "eval_result": "5602712e847081ba98808ca9b075865187019b5378423aff6039a7b9e76a3e72",
    "eval_run": "178c461dcec85f697f52c7b897a427e744e03f24cad08941a09d88b6f20761ae",
    "eval_second_judge_pass": "d0b7a869ada1a23f1ae41a01f3d6bbeccf770275cd810182f087d6b2c126491f",
    "eval_sweep": "10dd422f33f64afb0d6083369f02e5efdab008754b7af97b443045f1516ccb98",
    "eval_trace": "d28a11fd8b69fc862588cd9b147a8d79d36fe9e6bc4dd630209dc645aa16c19b",
}

#: The fields the store addresses each core type by, pinned: a step may never touch one, so the set moving
#: is a decision about the store, never a side effect.
PINNED_ADDRESSING_FIELDS: dict[str, frozenset[str]] = {
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
    "eval_trace": frozenset(
        {
            "id",
            "doc_type",
            "schema_version",
            "scope_id",
            "result_id",
            "eval_run_id",
            "judge_evidence",
            "judged_artifact",
            "schema_minor",
        }
    ),
    "eval_out_of_run_spend": frozenset(
        {"id", "doc_type", "schema_version", "scope_id", "purpose", "launch_group_id", "template_id", "created_at"}
    ),
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

#: The frozen corpus: one directory per core version, one stored document per core type.
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "core_documents"

#: A core type that did not exist before some version, by the version it was first stored at: no store holds one
#: from before it, so its frozen corpus starts there. A type that existed and joined the core later is not here —
#: an older store holds it, and its corpus starts at the baseline.
FIRST_STORED_AT: dict[str, int] = {}

#: Each frozen fixture's sha256, so a stored document of the past is never edited.
FIXTURE_DIGESTS: dict[str, str] = {
    "v15/eval_action_job.json": "67cb26c6a769c47755648035ac83c94803440cd8ec8801eadccd31afe2f04199",
    "v15/eval_analysis_attempt.json": "98c17ced332c5143b11313c292be519420f61480c300fa39f27190aedc2fce21",
    "v15/eval_campaign.json": "edc265c933db6e61ed515d9a5bd4c2b9d912e1cd67975781c5f2223cbe300958",
    "v15/eval_generation_slot.json": "84a0b5aea0e55129a656fc8fbee4898312e76ff57abcc661741d2c0796ee0145",
    "v15/eval_judge_temperature_check.json": "403d4ce0aa93ad55fb62a173d73de6d2afff0f75580994a7f0f206e82fd1e27a",
    "v15/eval_launch.json": "4338a30f69c6fc2d8c4fc1c19a4600ac9f94c8233f0cdcfe6140d23f84f64bc8",
    "v15/eval_out_of_run_spend.json": "8f6a0d5ff459e8085e56c5f71b381250d7fba692ad69e762c8770986f00837c4",
    "v15/eval_result.json": "7f3876042b6aeaeaeb7f7614cabecdbf9c84309a3db492e819c74783e0df49f1",
    "v15/eval_run.json": "4be37ded93ce117bafc670713c6c1e773e0cfa3059b18033e2709874547a6237",
    "v15/eval_second_judge_pass.json": "ca8e46c05aeccc4dbe79c4554ba13609bf49ba97a7abb84a80fa74da01099d73",
    "v15/eval_sweep.json": "0ac48ac8e6ec9afe67e6581f860347b5dce2b73e294b95dbadbb5b267d687186",
    "v15/eval_trace.json": "ada03d5dad96a6dd3faba0cf34af88f07a6b63b48adc80213be032bcb79abec4",
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

    :param model: A core stored model.
    :ptype model: type[EvalBaseModel]
    :return: A sha256 hex digest; it moves when what the model accepts moves, and not when its prose does.
    :rtype: str
    """
    return canonical_digest(_normalised(model.model_json_schema(mode="validation")))


def _core_models() -> dict[str, type[EvalBaseModel]]:
    """The stored models read in the core's version space: the core base, less the durable types on it."""
    return {
        doc_type_of(model): model
        for model in stored_models()
        if issubclass(model, CoreDocumentModel) and model.version_space is CORE_VERSION_SPACE
    }


def _fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("v*/*.json"))


# --- the shape -------------------------------------------------------------------------------------


def test_the_pins_were_taken_under_this_build_s_core_version() -> None:
    assert PINNED_CORE_SCHEMA_VERSION == CORE_SCHEMA_VERSION, (
        "CORE_SCHEMA_VERSION moved: re-pin every digest under the new version (see the module docstring)"
    )


def test_the_pins_were_taken_under_this_build_s_core_minor() -> None:
    assert PINNED_CORE_SCHEMA_MINOR == CORE_SCHEMA_MINOR, (
        "CORE_SCHEMA_MINOR moved: re-pin every digest under the new minor (see the module docstring)"
    )


@pytest.mark.parametrize("doc_type", sorted(CORE_SHAPE_DIGESTS))
def test_each_core_shape_matches_its_pin(doc_type: str) -> None:
    """A core type that accepts something new, or refuses something it read, is a bump."""
    model = _core_models()[doc_type]
    assert core_shape_digest(model) == CORE_SHAPE_DIGESTS[doc_type], (
        f"{model.__name__} ({doc_type}) changed shape under core v{CORE_SCHEMA_VERSION}.{CORE_SCHEMA_MINOR}. A core "
        "shape change is a minor (additive only) or a major (with an upgrader), an added optional field included; see "
        "this module's docstring"
    )


def test_an_added_optional_field_moves_a_pin() -> None:
    """The case the pin exists for: an OPTIONAL field joining a core type is a shape change (a minor), and is seen."""
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


def test_the_core_reads_only_its_own_major_and_every_major_has_a_ledger_entry() -> None:
    """The core ships no upgrader: a core major is a baseline reset, so the baseline is the major, explained."""
    source = inspect.getsource(versioning)
    ledger = source[source.index("CORE_BASELINE_VERSION: Final") : source.index("CORE_SCHEMA_VERSION: int =")]
    assert CORE_BASELINE_VERSION == CORE_SCHEMA_VERSION
    assert CORE_VERSION_SPACE.upgraders == ()
    assert CORE_SCHEMA_VERSION in {int(version) for version in _LEDGER_ENTRY.findall(ledger)}


def test_every_core_minor_has_a_ledger_entry_carrying_the_additivity_sentence() -> None:
    """Every minor of this major is explained, and each one asserts in words that an older build reads it right."""
    source = inspect.getsource(versioning)
    ledger = source[source.index("CORE_SCHEMA_VERSION: int =") : source.index("CORE_SCHEMA_MINOR: int =")]

    assert (CORE_SCHEMA_VERSION, 0) in minor_entries(ledger)
    assert not minors_missing_the_sentence(ledger, CORE_SCHEMA_VERSION, CORE_SCHEMA_MINOR)


def test_a_minor_entry_without_the_additivity_sentence_is_named() -> None:
    """The rule above, on a made-up ledger, so it is seen firing while the real one has no minor past the first."""
    words = MINOR_ADDITIVITY_SENTENCE.split(" ")
    ledger = (
        "#: - **v14.0** — the baseline minor.\n"
        f"#: - **v14.1** — a run's optional note. {' '.join(words[:6])}\n"
        f"#:   {' '.join(words[6:])}\n"
        "#: - **v14.2** — a result's optional tag.\n"
    )

    assert minors_missing_the_sentence(ledger, 14, 2) == [2]
    assert minors_missing_the_sentence(ledger, 14, 3) == [2, 3]


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
    if document["schema_version"] != CORE_SCHEMA_VERSION:
        return
    written = read.to_dict()
    if document.get("schema_minor", 0) == CORE_SCHEMA_MINOR:
        assert written == document, "a document at the current version reads back whole"
        return
    # An older minor of this major: the write stamps this build's minor and adds each later minor's fields at their
    # defaults, and keeps every value the document stored, at every depth.
    assert written["schema_minor"] == CORE_SCHEMA_MINOR
    assert keeps({k: v for k, v in document.items() if k != "schema_minor"}, written), (
        "a document at an older minor of the current major reads back with every stored value"
    )


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda path: path.relative_to(FIXTURES).as_posix())
def test_a_frozen_document_written_before_its_optional_fields_still_reads(path: Path) -> None:
    """A document stored before an optional field existed lacks the key; it reads, with the field's default."""
    document = json.loads(path.read_text(encoding="utf-8"))
    model = _core_models()[document["doc_type"]]
    optional = {name for name, field in model.model_fields.items() if not field.is_required()}
    addressed = CORE_ADDRESSING_FIELDS[document["doc_type"]]

    sparse = {key: value for key, value in document.items() if key not in optional or key in addressed}

    assert model.from_dict(sparse).schema_version == CORE_SCHEMA_VERSION  # type: ignore[attr-defined]
