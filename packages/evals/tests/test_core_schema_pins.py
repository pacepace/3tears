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
PINNED_CORE_SCHEMA_VERSION = 15

#: The ``CORE_SCHEMA_MINOR`` every digest below was taken under.
PINNED_CORE_SCHEMA_MINOR = 1

#: Each core type's validation JSON schema, normalised (:func:`core_shape_digest`), by ``doc_type``.
CORE_SHAPE_DIGESTS: dict[str, str] = {
    "calibration_rating": "06d6ad5f49eaca78e679bd187748674042249d47e20cde42c51b37e7121b69bb",
    "case_set": "73215bf96f3e1162628f0291eb019aab03a9f89342ed92b797acc6f7ab396593",
    "eval_action_job": "08dde0e492e98335d953dca2198983899a70960b5bfd52b82ba1cbb64d6f3e75",
    "eval_analysis_attempt": "28cbb946c28429589a90688b593275af8278820ba6f3bc3ee414a088a67f330d",
    "eval_campaign": "62dbc461d3dc9fd40d05a255aa2446390dee7df6b04ae653109349429327f5bb",
    "eval_generation_slot": "ea6ecdf0bb0f2baf40a071ede146fbb1f194ed0fd215fa1d99bd837ad7cc8f08",
    "eval_judge_temperature_check": "239334f02911c04f433c9a21d5c4d2f00df8b65790a85573eb6fe60035f9b8e4",
    "eval_launch": "17987aee847cc79dcdf4e902f06321f22b9b93cf7a0fbfb9a7eb6310fee0d472",
    "eval_out_of_run_spend": "85d9d128c2f8ecbd5c41adf6f2cde657ac52c672adb6241d968f8647ec79f1ae",
    "eval_result": "554eb01ed0f1b6cd729471f52e8c0b744919240fe1858262b208f270d48040c3",
    "eval_run": "e44d44f881a2e9c91825dfd5def7fd5414a39979ef1c4c08aae4bda49e01f52e",
    "eval_second_judge_pass": "118f451f6e13ef3bab1c551419d3fb4daafc8603f5ca6484c42d2b0b8fd55590",
    "eval_sweep": "bb9ca087b2023cf639e413b53d70f224f72a0b6dbed690cda60876852ae13976",
    "eval_template": "f267c311333029ca4a58b3334393c7644b3c205c7f3f911ea9d7b25e64562e32",
    "eval_test_case": "43f0234687b956e73f9c858fecb19c5e3e78b09fef6c880dec8a9b78bad9c6b8",
    "eval_trace": "c8fb190d1f49e8c7b83e81ed9fe31598b910341891eb7d7721168b10453a753f",
    "judge_config": "a81b6a373d7e9b247c2f20709de345de821d0c412d83dbacd2c44fdc5792f348",
    "judge_config_tombstone": "cdb415348bffd7b1b997639da367567c2a84c84c9a3a69f5df6a8c616dcaad93",
    "rubric_dim": "cafe6253eb3b788bdf33680491c2b09ecbd733a60c85bd3e1d0012cb50dae26e",
    "rubric_dim_tombstone": "c9a4dc812590a5e339c04c4a577d4267549aa944cf837dce4e3e156815ed2801",
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
    "v15/calibration_rating.json": "333fbf1e96422b1d39b7ee668f5b8d0ac9b3d2b84dddfb2ee3a3e7a53c427bbe",
    "v15/case_set.json": "53d0200175bde582ca314576e24572887efa2bc457b1a685270bbcd9c3596175",
    "v15/eval_action_job.json": "5283f8ce71dfb97876c55ea602877f738bbf7135c976fbcd1904aa1716e66c05",
    "v15/eval_analysis_attempt.json": "855f51c733b8275beabf97b520dd752765865a8c307c78ed8642a1c669ab32a9",
    "v15/eval_campaign.json": "c9152ddab88b638891590f3909b5917629c2ff21766a54f62362703560881645",
    "v15/eval_generation_slot.json": "9a04af073babbd91668b2b294d7d20ba8c62fba6d430abcdecfcd1313b2d4c98",
    "v15/eval_judge_temperature_check.json": "4dab8c1462b5cd3dab1945624c73061677fe292f03c8fad2509dc96ef72b955b",
    "v15/eval_launch.json": "8fa9e3d0060c1f43a0717536c4a473d2cb981ba01170c277ecbe9ce8e1d97519",
    "v15/eval_out_of_run_spend.json": "0919885afaf82372b7f4bc900fb5965a7bc0e09c6d9504cbbae99633a3c79ab3",
    "v15/eval_result.json": "492c0a747f2698068e66b2bb3779aa0b059921008bb6b1e1ee7815599a58a525",
    "v15/eval_run.json": "e90dbc59bb11c2fcdd21880e8892446070d49284cc76298a39d019cd472685fb",
    "v15/eval_second_judge_pass.json": "a3d1b663aa127e77e6056849a8b046e63d1ad5ab87f9791eef67fd5589d8de53",
    "v15/eval_sweep.json": "1ac1d31e40f730f47a8e8591de9f7f7f712f4c50d56322573e8a5c4e112ca4a3",
    "v15/eval_template.json": "4ffe5e19b95d2197eccd06f94fbac03dacd1ce3ee52722bfd9cfbdb3d54933ad",
    "v15/eval_test_case.json": "fd66c87220a5769f26221d43d9fbca830e70bcb27f1ef79c4137db01bcc13d66",
    "v15/eval_trace.json": "703f57f22421d3da09ed45bcf165b05e4d1b07b190ea47990c4cdce03eabe024",
    "v15/judge_config.json": "7c7eeb6e55026a4d03769f5256368d88f7331d9468e661fe041254e90e5f9cdd",
    "v15/judge_config_tombstone.json": "1b140241aac242aaaa427dc3aaa99da7f39def10ebff175681618082c8fecd87",
    "v15/rubric_dim.json": "f0bb5c7be5e97393bb95e9b481a70f174a80ae7f1747c3f32ca9aafd465ae9af",
    "v15/rubric_dim_tombstone.json": "d429714b9ed3e3643377451ea53507c21c7daad72ba12b33a389197d770bb300",
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
