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
PINNED_CORE_SCHEMA_MINOR = 2

#: Each core type's validation JSON schema, normalised (:func:`core_shape_digest`), by ``doc_type``.
CORE_SHAPE_DIGESTS: dict[str, str] = {
    "eval_action_job": "9c11c196f42bb6ef9947d4fdb00fd179e5cbc5eb3f7e5a0949e413fc35e39d0d",
    "eval_analysis_attempt": "74fce5fef0e2876f62df51a7c87f364f358ec652646e4fa1efdf696a6361a2c3",
    "eval_campaign": "aed376e4d6b5f00f083f74a6ebadc545aa466de236124830c37cc228276acdca",
    "eval_generation_slot": "554e5225bf0bfa617991750089c4953fc4a1903ce095a1e046e53829819b7fc3",
    "eval_judge_temperature_check": "204d780376bb7db499179d6b63141efa45e3a7ebf45905aa6780d226b872d40e",
    "eval_launch": "e6d6bfb32c805e284b36b66871f454e8597f87bc0a3b0d3866c417851828fcd6",
    "eval_out_of_run_spend": "ca2d959bf9d9657e913b5f442730fef7a9505ec693beae87923e199831b2870f",
    "eval_result": "8bec36e2ce1efed3cf1e099df6e9e3b807b20aa8725a182cc2634990ae49bd65",
    "eval_run": "526251b8c83afd9889e97d9a86604ba4c2867a3b21ce2ce70e27db41ee14f116",
    "eval_second_judge_pass": "dfd1d452b98b6c5c7b8fbfd4ee1f4256c8496b29e6e2e770ea2e688b1a34c052",
    "eval_sweep": "2ffabe3f76ecbf64abf1e4fe196b409fb9bf7bae417aad6660f6eae7fab5910a",
    "eval_trace": "394735ef53922a7547c5e3cc883030c030be9cd22a98a8f4d527eb0776c645a2",
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
    "eval_trace": frozenset({"id", "doc_type", "schema_version", "scope_id"}),
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
    "v15/eval_action_job.json": "e1f2e92c5bf8962345488c5791f93e4698c221af53831f06aba5934a73b73e95",
    "v15/eval_analysis_attempt.json": "bd1015905b2a71c3d1d3efebbb15ce313d1c64a5281a20558fe2e685bcfc85e0",
    "v15/eval_campaign.json": "b000affa5e9c0bbea26f272ba0024270cceb1fe1720a06ed64c7916f4f9bf0a3",
    "v15/eval_generation_slot.json": "18efd267b51051a943a4edc53aeb0f57f11efd6f54fa9996791e15b5a25e9991",
    "v15/eval_judge_temperature_check.json": "178f1a5afc01c064ae7013a35bcf6528a907906e2764689a0784924aa4b70cb3",
    "v15/eval_launch.json": "3d92be4c4c5d701841ff9ab4b4192879888902d6626ab9e35cad016a9abf53e9",
    "v15/eval_out_of_run_spend.json": "b8ae34c1bb023f0c5694ad81b31345fd0be19b4d41655e4f3c1d5173d126109d",
    "v15/eval_result.json": "185ebd15dd452e4ba84527d9750caaa4ccc971055b7280b8124a6f9afb9f690e",
    "v15/eval_run.json": "05d4b8d8ab2c79d288fc952eb768d820f471fd0038476a5092ca36d0599c7474",
    "v15/eval_second_judge_pass.json": "6a3e6a6e0bdb25e904e4ee48c941b3c6366deaa1dd0373c62d43d1c8f799ed12",
    "v15/eval_sweep.json": "4b51f9b19f3cc56c2928413fe2f8314af40d00d615bf17d37b99e49f50e60cc7",
    "v15/eval_trace.json": "8fe80d888a150823a845e20750d259df302a9559a0b94fc43fa9967156a26445",
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
