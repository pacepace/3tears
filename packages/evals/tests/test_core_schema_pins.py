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
PINNED_CORE_SCHEMA_MINOR = 0

#: Each core type's validation JSON schema, normalised (:func:`core_shape_digest`), by ``doc_type``.
CORE_SHAPE_DIGESTS: dict[str, str] = {
    "calibration_rating": "6f9ad0190d746c34a262aba8b268182e6a2d7c3ed85e8da9597f9a8d7ed365d6",
    "case_set": "10843b08827a93f835f70e91282354299b2ec81e1839609a6937fc9032473a2f",
    "eval_action_job": "b960c0abac4d73afa5bf061df97ba59e39b58d684a4f0660d7eba634de7dc7e0",
    "eval_analysis_attempt": "7f260bd77c125fe2484933c1deffac9ff6b41d5e399ae6edfb2064c4456d9d64",
    "eval_campaign": "a937b4282b7ed15f993f2d2c4d7e2bf05f9449d26a0c03db151b41b2e4a8032e",
    "eval_generation_slot": "4466adfa53f0c9b1e07f6ecfaf252e94a661eb084aa16414a849bc4a2da50f1f",
    "eval_judge_temperature_check": "89e925705d7b407d7cf1d54f74fcace96a13cc58e214e7d25aa94b50518a4ae6",
    "eval_launch": "768e476f137056c37a11383b7013d83a0b13bc4707cb5fc0ae758877201d64c2",
    "eval_out_of_run_spend": "465be395fdcddc11acfc4891e2cc5ca46ddf944a5c72e27f15d38c7beebd7390",
    "eval_result": "d1a7719eeaa2e48d0b8dbb43f285c12ebfaa85b0385da72749f7f7ad6d57dca0",
    "eval_run": "f20d70a82d79130ba7bc0eb6c2896221cb0924f0940f89f5d0f44f87191dedc6",
    "eval_second_judge_pass": "44b048a733c7914e6530aae030667d21ff6aa307a3935fc95d9aa484f8f2b4ba",
    "eval_sweep": "0614390275057c6fc7362d69033f3cbc432203e2cb228f835349f5ba9af6a996",
    "eval_template": "2f1c8793034194705e1e2889ebff5a77ce23a95c525083a7fe1b7738483c3c36",
    "eval_test_case": "933059c8cac11160a6d8ecf42bb17ff2f2dc4314eb83724277d4fd34b019dcef",
    "eval_trace": "e3fb36a574e5ca361ac0d48d886bcd835dae2f9789373c7367b3d649d0e7f073",
    "judge_config": "b6856660cd3f04fe36c88b18ab9a8edf0d352b824b3bd54de272057345f4f4b8",
    "judge_config_tombstone": "40345344b65208edcc863165a50cdd3098c8fab93b55ea38b230fd55dc072608",
    "rubric_dim": "12276b6acbbe3184fe0cc4e51ec44b0c62b2fd58f35d43947f01c27e2482a179",
    "rubric_dim_tombstone": "6c53b7851e2f75c6100899d70ef4c3067d67a3e790fad85073514801c26fde41",
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
    "v15/calibration_rating.json": "894f74e31dcdb42e26cb949b080cbcb2e76bb20d8361ed1c17bc74a9c7864e7e",
    "v15/case_set.json": "1a30692918a1dcd4ff1702bf3c2095133e6652021e30961f29a8528a673cd082",
    "v15/eval_action_job.json": "4b2ecc83cff78228a6e9ece65f1e94ea0141f4eac3423c8e63bf755759993e58",
    "v15/eval_analysis_attempt.json": "51ac3116665931de45c028104a98d709c3a71a646c15e44c0851d2983425d237",
    "v15/eval_campaign.json": "bc597f20b976172f4a75ee1cfee36727a010306ab6f97b973962415267081dbb",
    "v15/eval_generation_slot.json": "6b1df8625f6871a39db2d4bf190ac52e5867455b1c9d1e3b3aec41d8fed30623",
    "v15/eval_judge_temperature_check.json": "048bbaf27f60485f264269579b66acc7ef03a2571974bcb1053009cfce64ee19",
    "v15/eval_launch.json": "bde5f0faacca127190c68fe186dd3091c6b0539ea720dccd9734bca7f7c9415c",
    "v15/eval_out_of_run_spend.json": "962d53c87533583170a959f2319434f50b4b12a16163494e55c976c9727f6d19",
    "v15/eval_result.json": "a99434ba322273da0edf48cd394747f8b5143c937423c118555518b27947450c",
    "v15/eval_run.json": "a1f4126c13d91a07152b3ed92af863e22ef1fe84ad60cfd611f5debcdcd1190f",
    "v15/eval_second_judge_pass.json": "64f0820224a76b57d589b04d521e813da0d73f977a26b6db53effcdc45d74e64",
    "v15/eval_sweep.json": "cc15b2f7a0786064e581c29bf879251ec9b2bb23361568fa9dbbddc367e2f4de",
    "v15/eval_template.json": "4cf70858e2c14deabf3af88055d8e31dcba1bbb0253ae953baa23998be8abebf",
    "v15/eval_test_case.json": "504dcf34cff61440e782e30322cf9cc42cf681fbbb866faddff9aef48ee405cd",
    "v15/eval_trace.json": "469a20e131c67cb5c00082463bd32a148a544be182301cc645f8ecedf581769c",
    "v15/judge_config.json": "d3f29495b08a71effd16ad0b64b9f28ccff653aa2e21bf858a795099af4a1c96",
    "v15/judge_config_tombstone.json": "b75f4d2f2597723a72655afe3cf852de1bcd21bc43c90c18c841e50264d33a4b",
    "v15/rubric_dim.json": "b65c88b862dddf2723ea3972be8381cca002131d7d9ae524418602d1bc2f6378",
    "v15/rubric_dim_tombstone.json": "da3fb751ac55bf182f339faf233fc65aeeb0d467044c78ee9709844761042cef",
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
