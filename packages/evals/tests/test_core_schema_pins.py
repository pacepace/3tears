"""The evidence core's shape is pinned: a core shape cannot change without a minor or a major, as the change is.

``schema/versioning.py`` promises that a core document is read forever from the baseline, and that a newer minor of
this build's major is read too. This holds the first promise from both ends: every core type's validation JSON schema
is pinned to a digest under the major and minor recorded here, so a shape change without a version goes red here; and
a frozen corpus of stored core documents, one directory per core major, must keep reading under the current build,
which also catches a validator tightening the JSON schema cannot show (the goal grammar, an ``AfterValidator``).

**When this goes red**, one of these happened:

- **A core shape moved and the version did not.** Any change to a core type's shape is a version, an added
  optional field included. If it is additive only (a new optional field with a default, a new optional
  sub-object, a new member of an open vocabulary, a new doc type) it is a **minor**: bump ``CORE_SCHEMA_MINOR``
  with a ledger line ending in ``MINOR_ADDITIVITY_SENTENCE``, and re-pin the digests and the minor here. Nothing
  else is a minor: before changing the shape, copy ``tests/fixtures/core_documents/v{N}/`` to ``v{N+1}/`` (the old
  directory stays as it is: it is what an old store holds), bump ``CORE_SCHEMA_VERSION`` with a ledger line above
  it, reset ``CORE_SCHEMA_MINOR`` to 0, append one ``CoreUpgrader`` from ``N``, re-pin the digests and the version
  here, and regenerate the new directory's fixtures from
  :func:`~packages.evals.tests.stored_models.stored_sample`, pinning their files. Give the step its own test of the
  fields it transforms.
- **The version moved and the pins did not.** Re-pin every digest under the new major or minor, even where a
  shape held, so this file states what it was checked under.
- **The version moved and the ledger or the chain has no entry for it.** Write the ledger line (a minor's
  carrying the additivity sentence), and for a major the upgrader.
- **A frozen fixture stopped reading.** The current build no longer reads a document an older build
  stored. Either a validator tightened (a major and a step that brings old documents into line, or a None
  for what cannot be derived), or a step is wrong. Never edit the fixture: its digest is pinned so it
  cannot be.
- **A frozen fixture's digest moved.** Someone edited a stored document of the past. Restore it.
- **The core's membership moved.** A type became core or stopped being core; that is a decision about
  what is kept, recorded in ``CORE_DOC_TYPES`` and here together.

The v14 corpus is the baseline of a release that has not been published: no store a released build wrote holds a
v14 document yet, so while that is true a baseline change regenerates it in place rather than copying it.
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
    CORE_SCHEMA_MINOR,
    CORE_SCHEMA_VERSION,
    CORE_UPGRADERS,
    MINOR_ADDITIVITY_SENTENCE,
)

#: The ``CORE_SCHEMA_VERSION`` every digest below was taken under.
PINNED_CORE_SCHEMA_VERSION = 14

#: The ``CORE_SCHEMA_MINOR`` every digest below was taken under.
PINNED_CORE_SCHEMA_MINOR = 0

#: Each core type's validation JSON schema, normalised (:func:`core_shape_digest`), by ``doc_type``.
CORE_SHAPE_DIGESTS: dict[str, str] = {
    "calibration_rating": "eb82930180e483123fedb43115bcccc25072743a75b362ca36c6405a011ab840",
    "case_set": "cd9240cfe0ba3d5bc47a5d1745e9550d5635c97dec673a07e369ecaaa0d4cd2d",
    "eval_action_job": "34a319ad2d80e03af4230c7fd743217e5b8077612346acab5cf2046da60edf3d",
    "eval_analysis_attempt": "bf64f0740df0680db480ec6ce27bbe06706482673c866b48a22245e17782af3e",
    "eval_campaign": "80c372f8bc21bdb3dab458044f54eafe8199d1a3291a6121187b721099c6f440",
    "eval_generation_slot": "ca9d09bbe419cfa67397aa779301656c3c55e0a8d5ce682f961fa7afa9001a02",
    "eval_judge_temperature_check": "bb30f648778effb15139f519595277af05dc1f12cdbc84994a487a4091dc5a51",
    "eval_launch": "068f35d01499ee970dd142663dd5fce44ef61ac2ad57f06e66d10bc0fd69115e",
    "eval_out_of_run_spend": "f2e3cd674e1b5f8dcc7d0ef5c66cbc5f1b610ecdbcae112a3873f34712c68716",
    "eval_result": "8128b07a619c442d91030fea873063ab124c6b005a728575f169f11e2c2e4ddd",
    "eval_run": "e43ff83a6be08add5e725b5ac00e545b815ec19526a654d0322670ca8d05cf8a",
    "eval_second_judge_pass": "849b1f0bae7466e15032b71f9e620c6060b0de94aed7d107aedd76ac5cae814c",
    "eval_sweep": "36ed86de4dcd89bdc989b9bdb7542440a39cdd27f1b96d027b4b3ac4d187b3d0",
    "eval_template": "82c4f3b331d903623cff91168c442ecabbe99c5a35224b1ef26d914f93923611",
    "eval_test_case": "8b8529be10e69772fbd20a86732730a9be3002bfe8c91e94cb0adb32f3cce414",
    "eval_trace": "9a661ba216e98bf33eedaa7b1ec0677ba0c2d6de3eaa7fe50435a03338f8ba0b",
    "judge_config": "2442bfc1477ef2dc08c1e0005a1be3111dc2618342eb8bb05c1773c867dae9c9",
    "judge_config_tombstone": "9f2516ea3389b3e6d2821f30afc4c9d87281b6939bf9a6f29e4dcc5227f8d795",
    "rubric_dim": "16426934e0e2310cc31657bf770bcd2be2e5468bfc1b8f68d4fc4751af1063f6",
    "rubric_dim_tombstone": "78d56efa5301b61be7bd483a95bea43146e30492dbb5f3dcea68777064f0370f",
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
    "v14/calibration_rating.json": "bf707deb719e83eb83974f7af844b4b5e4c0b2d83edbd460586956252b17a4a3",
    "v14/case_set.json": "ccdff6b0635adf883991c89c44b3e2a604db2513790010a370a00b52bf999654",
    "v14/eval_action_job.json": "828d1eea5030b330efb7b00537ae13924941d6504c4425abad6b27d740c3c639",
    "v14/eval_analysis_attempt.json": "829b1182bea47947263627d303dd193e028f70ed1a808b94d5071c49b3c9ecca",
    "v14/eval_campaign.json": "faf6562effd8a683b9fc35fce612fc345b4e8a5efc3a31fd64129df5c4981840",
    "v14/eval_generation_slot.json": "1cd595e453b2b517fb93de7a00fbe9940acddfdccb15e79f2f15010ab04ffafc",
    "v14/eval_judge_temperature_check.json": "0da0c6f4052126c1293ae09aace9e4591c5a3d461d9171734bab964b7489ef62",
    "v14/eval_launch.json": "18d959b6f2fb54da17358e93fd9a361d5521257eb3f74ecf19f5dc7dcc2b01de",
    "v14/eval_out_of_run_spend.json": "e265120f73f4e819219777edbd4c94c33827803ec2f670c9fa37eda00dad298a",
    "v14/eval_result.json": "7e5f76f4dd5ada43e7e13dc9e635d5beb3bc2f2f0afdb99f8618e75af27b4640",
    "v14/eval_run.json": "a8e7813f8cbb4d83942482e710275cf946c1d0daf878aabe625e7c3f4352d2c0",
    "v14/eval_second_judge_pass.json": "84efd573fed017f339033d8b87625ebf151f2a282564d883f7d101955164dea9",
    "v14/eval_sweep.json": "fda0545809a15cf1f4fe33f7626259f6cc5d1b05cc18d097ecdb5ec51b6ea244",
    "v14/eval_template.json": "ed58c584ac2de9e33902877ace38d5124bd1938830f835e6d22292a35e0b1fa3",
    "v14/eval_test_case.json": "f1dbebe6ab126c26e78eba85c91ebb91bd5b59341f2396a2bdc3704104f7a74b",
    "v14/eval_trace.json": "1447db5bf777035550f7f6695a838322b8d62dd45769732fd877ecdf7e73a08b",
    "v14/judge_config.json": "953a71d18c72fafdf1fc254bed202aa4c4f86b7d8e3c7fd01643fb64ec40f3b7",
    "v14/judge_config_tombstone.json": "f5130f149870ce29e0dc668d4109868b85e623e1c57ec95cf58d336ec40818c8",
    "v14/rubric_dim.json": "8e9f903ebe838bded11813a1f48ae82b28bab6fedf90cdaa26fbe989bfcff120",
    "v14/rubric_dim_tombstone.json": "817b4efd4f7195abeb59c39d691b54fc8e29cde5b924e4d9b027584ce39a719f",
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
    """The case the pin exists for: an OPTIONAL field joining a core type is a shape change — a minor — and is seen."""
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


#: A minor's ledger entry, as the comment block above ``CORE_SCHEMA_MINOR`` writes one: ``#: - **v14.1** — …``,
#: through the end of its (possibly wrapped) comment lines.
_MINOR_ENTRY = re.compile(r"^#: - \*\*v(\d+)\.(\d+)\*\*(.*(?:\n#:   .*)*)", re.MULTILINE)


def _minor_entries(ledger: str) -> dict[tuple[int, int], str]:
    """Each minor's ledger entry in ``ledger``, keyed by its major and minor, its wrapped lines joined."""
    return {
        (int(major), int(minor)): " ".join(line.removeprefix("#:").strip() for line in text.splitlines())
        for major, minor, text in _MINOR_ENTRY.findall(ledger)
    }


def _minors_missing_the_sentence(ledger: str, major: int, newest: int) -> list[int]:
    """The minors after a major's first, up to ``newest``, whose ledger entry is absent or lacks the sentence."""
    entries = _minor_entries(ledger)
    return [minor for minor in range(1, newest + 1) if MINOR_ADDITIVITY_SENTENCE not in entries.get((major, minor), "")]


def test_every_core_minor_has_a_ledger_entry_carrying_the_additivity_sentence() -> None:
    """Every minor of this major is explained, and each one asserts in words that an older build reads it right."""
    source = inspect.getsource(versioning)
    ledger = source[source.index("CORE_SCHEMA_VERSION: int =") : source.index("CORE_SCHEMA_MINOR: int =")]

    assert (CORE_SCHEMA_VERSION, 0) in _minor_entries(ledger)
    assert not _minors_missing_the_sentence(ledger, CORE_SCHEMA_VERSION, CORE_SCHEMA_MINOR)


def test_a_minor_entry_without_the_additivity_sentence_is_named() -> None:
    """The rule above, on a made-up ledger, so it is seen firing while the real one has no minor past the first."""
    words = MINOR_ADDITIVITY_SENTENCE.split(" ")
    ledger = (
        "#: - **v14.0** — the baseline minor.\n"
        f"#: - **v14.1** — a run's optional note. {' '.join(words[:6])}\n"
        f"#:   {' '.join(words[6:])}\n"
        "#: - **v14.2** — a result's optional tag.\n"
    )

    assert _minors_missing_the_sentence(ledger, 14, 2) == [2]
    assert _minors_missing_the_sentence(ledger, 14, 3) == [2, 3]


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
