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
  here, and regenerate the new directory's fixtures and pin their files (``regenerate_core_fixtures.py``, below).
  Give the step its own test of the fields it transforms.
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
v14 document yet, so while that is true a baseline change regenerates it in place rather than copying it. **A
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
    "eval_action_job": "1f0f57286049566dddc98121e1f93051c43192e9a591ff226b2bb1fd319ba001",
    "eval_analysis_attempt": "b6818453c2c4909a9b9b8a38aa295e40b1001af589bdebf08b5e76c6b5bea436",
    "eval_campaign": "c5ce64d3f7df447c436b5f1b0313cd2464d61d1b456e1f0cb0e5fb9ff3a04c2e",
    "eval_generation_slot": "ca9d09bbe419cfa67397aa779301656c3c55e0a8d5ce682f961fa7afa9001a02",
    "eval_judge_temperature_check": "7b7329ebdc0bcc86d1ed178dceb5b27d09e87463941470a4780af3d61411a673",
    "eval_launch": "bea6ee7730e3cb901fae51173f0b9173e0c7760bd2044f61a006c1b8285d5ba8",
    "eval_out_of_run_spend": "3cf459518ff63151dc79c217df2c854e638565150a4d7cf6ce791b97cfda2488",
    "eval_result": "fcd0680de40e609fe022b711dfe654da4f4eed5d8c79d991a1f0af06cacd0fd0",
    "eval_run": "fd706ef4ac538a9a0bf26d98a960c10d5763fbdda6ef8fa60e7bad0ecb70d1fd",
    "eval_second_judge_pass": "0baec9aa24085cafa4d5c7956a9b81609262edc208069289d2cda7ad579c7436",
    "eval_sweep": "c77dc40a100b1562acd662710c1aa62a430672367fe6545fabc40035f6872332",
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
    "v14/calibration_rating.json": "13dd233fc1beb53c247fa7102caa5e548f718f09798595043981f1c3d2aafe6f",
    "v14/case_set.json": "abde1efc6cf73cdf3dc314f89df7ffc164be73edf45edd7d8dc64d9eebc70e93",
    "v14/eval_action_job.json": "11bf89d9afcf6076f0c8467ba215090183fa9949b0f9875e0a9241649b59cc3c",
    "v14/eval_analysis_attempt.json": "1d0c1d050bf3008b35defe9840a63ea873d7cd9ea616c3a7a2d96ed793851c4d",
    "v14/eval_campaign.json": "3c37ddd4d0a8568922a8a9ee28ffa642eeaa38581f9cd21ed96c6f7147b25f8c",
    "v14/eval_generation_slot.json": "eeb763fca25e650c4acd906fe6834f386b7253c82ad7d164ba0ca8c1d83e9d38",
    "v14/eval_judge_temperature_check.json": "378c623cd9cc8f9a8a2379d8d52c0c2f04a94562febe26809e4817c8016eb5cd",
    "v14/eval_launch.json": "f6032b5407687592e28f516d78b37d0f3a320f120bd681225301ad7d93990e75",
    "v14/eval_out_of_run_spend.json": "e5de0a9f9ca79c86ff094742c67b3091b58935be146f2bbd82127362dee506f0",
    "v14/eval_result.json": "bf5f4ddd5d88ffe1002a44df055fcc57521f648cdb2d08a4c12b08420b7e650c",
    "v14/eval_run.json": "38654e44816e956fc18ef6d1624df1b50ef5e6140968dc817d2b4644dd92d4bb",
    "v14/eval_second_judge_pass.json": "39255c38aa05e44d0c87e4d21e7588cc6d608acc8164dd0e060e7d0bc1c972ca",
    "v14/eval_sweep.json": "c8d9791fc3bd410fc9750696eec0a4f3c5dc6230ce08fd089f1155859dc52a68",
    "v14/eval_template.json": "5b3e6253363148f247469e514fc2ebed2ef3b061cb4e242b6596ca97a54fdac2",
    "v14/eval_test_case.json": "865b5b6209712f35c0e2df8d3c6a4e477a5bbc952c23404d3db1d0fded5b24f3",
    "v14/eval_trace.json": "1447db5bf777035550f7f6695a838322b8d62dd45769732fd877ecdf7e73a08b",
    "v14/judge_config.json": "b92e43af3ae07e90e0b184849a634e4db8a8ed503f23280e4483f9ba896ed244",
    "v14/judge_config_tombstone.json": "f317e204ef8834e9181cf273a7f7d45140b8125af6e01ff6d7fa8a99fd5dd10c",
    "v14/rubric_dim.json": "24fd1030e84b699a34f1162765159df07681e6c515e4588a72d79d528b210543",
    "v14/rubric_dim_tombstone.json": "b167e3026a5562188c7972a95ad80ae6642f85bd1418e8c97cea99fb311c157d",
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
