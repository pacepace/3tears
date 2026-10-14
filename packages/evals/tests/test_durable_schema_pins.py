"""The durable space's shape is pinned, and it is the one space where a major must ship an upgrader.

``schema/versioning.py`` splits what people author (templates, rubric dims, judge configs and their tombstones, case
sets, test cases, calibration ratings) out of the evidence core into a version space of its own, forked from core
v15 at ``.26``. The core may reset its baseline (a stamp and a refusal, no steps); the durable space never does, so
this file is the gate that holds it to its promise:

- every durable type's validation JSON schema, embedded shapes included, is pinned to a digest under the durable
  major and minor recorded here, so a shape change without a version goes red here — including a change to a shape
  the core embeds too (a ``RubricDim``, a ``JudgeEvidence``), which takes the durable document's space;
- the durable baseline is 15, forever: a build that moved it would refuse labels people gave;
- every durable version from the baseline has a ledger entry and an upgrader, and every minor's entry carries
  ``MINOR_ADDITIVITY_SENTENCE``;
- the frozen corpus under ``tests/fixtures/durable_documents/`` (the ``.25`` documents, moved here unedited from the
  core corpus) keeps reading under this build.

**When this goes red**: a durable shape moved, so bump ``DURABLE_SCHEMA_MINOR`` (additive only, with a ledger line
ending in the sentence) or ``DURABLE_SCHEMA_VERSION`` (copy ``durable_documents/v{N}/`` to ``v{N+1}/`` first, append
a ``CoreUpgrader`` to ``DURABLE_UPGRADERS``, reset the minor), and re-pin the digest the failure prints. Never edit a
frozen fixture, and never move the baseline.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from packages.evals.tests.stored_models import doc_type_of, stored_models
from packages.evals.tests.test_core_schema_pins import _keeps as keeps
from packages.evals.tests.test_core_schema_pins import _minors_missing_the_sentence as minors_missing_the_sentence
from packages.evals.tests.test_core_schema_pins import core_shape_digest
from threetears.evals.schema import versioning
from threetears.evals.schema.base import CoreDocumentModel
from threetears.evals.schema.versioning import (
    CORE_DOC_TYPES,
    DURABLE_ADDRESSING_FIELDS,
    DURABLE_BASELINE_VERSION,
    DURABLE_DOC_TYPES,
    DURABLE_SCHEMA_MINOR,
    DURABLE_SCHEMA_VERSION,
    DURABLE_UPGRADERS,
    DURABLE_VERSION_SPACE,
    CoreUpgrader,
)

#: The durable baseline, which never moves.
NEVER_MOVES = 15

#: The ``DURABLE_SCHEMA_VERSION`` every digest below was taken under.
PINNED_DURABLE_SCHEMA_VERSION = 15

#: The ``DURABLE_SCHEMA_MINOR`` every digest below was taken under.
PINNED_DURABLE_SCHEMA_MINOR = 1

#: Each durable type's validation JSON schema, normalised as the core pins normalise it, by ``doc_type``.
DURABLE_SHAPE_DIGESTS: dict[str, str] = {
    "calibration_rating": "d762a9dac94ca324d18c2addd366a2a7921c70cdb1223bf1aba92cf84f5138ab",
    "case_set": "73215bf96f3e1162628f0291eb019aab03a9f89342ed92b797acc6f7ab396593",
    "eval_template": "0d8284601f6631df29d23945b36054750b590b18ee9e6625f4ceaa8f96fc952e",
    "eval_test_case": "43f0234687b956e73f9c858fecb19c5e3e78b09fef6c880dec8a9b78bad9c6b8",
    "judge_config": "a81b6a373d7e9b247c2f20709de345de821d0c412d83dbacd2c44fdc5792f348",
    "judge_config_tombstone": "cdb415348bffd7b1b997639da367567c2a84c84c9a3a69f5df6a8c616dcaad93",
    "rubric_dim": "b419d4b700256649458dc8c7b8ad646d6526bb358144e925312d79e71c19d3f1",
    "rubric_dim_tombstone": "c9a4dc812590a5e339c04c4a577d4267549aa944cf837dce4e3e156815ed2801",
}

#: The fields the store addresses each durable type by, pinned: a durable step may never touch one.
PINNED_DURABLE_ADDRESSING_FIELDS: dict[str, frozenset[str]] = {
    "eval_test_case": frozenset({"id", "doc_type", "schema_version", "scope_id", "template_id", "stratum"}),
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
    "eval_template": frozenset({"id", "doc_type", "schema_version", "scope_id", "name", "archived", "universal"}),
    "judge_config": frozenset(
        {"id", "doc_type", "schema_version", "scope_id", "rubric_dim_id", "archived", "created_at"}
    ),
    "rubric_dim": frozenset({"id", "doc_type", "schema_version", "scope_id", "key", "archived", "created_at"}),
    "rubric_dim_tombstone": frozenset({"id", "doc_type", "schema_version", "scope_id", "deleted_at"}),
    "judge_config_tombstone": frozenset({"id", "doc_type", "schema_version", "scope_id", "deleted_at"}),
    "case_set": frozenset({"id", "doc_type", "schema_version", "scope_id", "name", "version"}),
}

#: The frozen durable corpus: one directory per durable major, one stored document per durable type.
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "durable_documents"

#: Each frozen durable fixture's sha256: the ``.25`` documents, moved from the core corpus with these exact bytes.
FIXTURE_DIGESTS: dict[str, str] = {
    "v15/calibration_rating.json": "894f74e31dcdb42e26cb949b080cbcb2e76bb20d8361ed1c17bc74a9c7864e7e",
    "v15/case_set.json": "1a30692918a1dcd4ff1702bf3c2095133e6652021e30961f29a8528a673cd082",
    "v15/eval_template.json": "4cf70858e2c14deabf3af88055d8e31dcba1bbb0253ae953baa23998be8abebf",
    "v15/eval_test_case.json": "504dcf34cff61440e782e30322cf9cc42cf681fbbb866faddff9aef48ee405cd",
    "v15/judge_config.json": "d3f29495b08a71effd16ad0b64b9f28ccff653aa2e21bf858a795099af4a1c96",
    "v15/judge_config_tombstone.json": "b75f4d2f2597723a72655afe3cf852de1bcd21bc43c90c18c841e50264d33a4b",
    "v15/rubric_dim.json": "b65c88b862dddf2723ea3972be8381cca002131d7d9ae524418602d1bc2f6378",
    "v15/rubric_dim_tombstone.json": "da3fb751ac55bf182f339faf233fc65aeeb0d467044c78ee9709844761042cef",
}


def _durable_models() -> dict[str, type[CoreDocumentModel]]:
    return {
        doc_type_of(model): model
        for model in stored_models()
        if issubclass(model, CoreDocumentModel) and model.version_space is DURABLE_VERSION_SPACE
    }


def _fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("v*/*.json"))


# --- the space -------------------------------------------------------------------------------------


def test_the_durable_baseline_never_moves() -> None:
    assert DURABLE_BASELINE_VERSION == NEVER_MOVES, (
        f"DURABLE_BASELINE_VERSION is {DURABLE_BASELINE_VERSION}: the durable baseline never moves, since every label, "
        "template and case people wrote since .25 would be refused by name; a durable shape change ships an upgrader"
    )
    assert DURABLE_VERSION_SPACE.baseline == NEVER_MOVES


def test_the_durable_space_is_exactly_the_durable_models() -> None:
    assert set(_durable_models()) == DURABLE_DOC_TYPES
    assert set(DURABLE_SHAPE_DIGESTS) == DURABLE_DOC_TYPES
    assert not DURABLE_DOC_TYPES & CORE_DOC_TYPES


def test_the_pins_were_taken_under_this_build_s_durable_version() -> None:
    assert (PINNED_DURABLE_SCHEMA_VERSION, PINNED_DURABLE_SCHEMA_MINOR) == (
        DURABLE_SCHEMA_VERSION,
        DURABLE_SCHEMA_MINOR,
    ), "the durable version moved: re-pin every digest under it (see the module docstring)"


@pytest.mark.parametrize("doc_type", sorted(DURABLE_SHAPE_DIGESTS))
def test_each_durable_shape_matches_its_pin(doc_type: str) -> None:
    """A durable type that accepts something new, or refuses something it read, is a minor or a major with a step."""
    model = _durable_models()[doc_type]
    digest = core_shape_digest(model)
    assert digest == DURABLE_SHAPE_DIGESTS[doc_type], (
        f"{model.__name__} ({doc_type}) changed shape under durable v{DURABLE_SCHEMA_VERSION}.{DURABLE_SCHEMA_MINOR} "
        f"(now {digest}). A durable shape change is a minor (additive only) or a major with an upgrader; see this "
        "module's docstring"
    )


def test_a_change_to_a_shape_the_core_embeds_too_moves_a_durable_pin() -> None:
    """A template embeds ``RubricDim``, which core shapes embed too: its digest covers the embedded shape."""
    from pydantic import create_model

    template = _durable_models()["eval_template"]
    rubric = template.model_fields["rubric"].annotation
    grown = create_model("EvalTemplate", __base__=template, rubric=(rubric | None, None))

    assert core_shape_digest(grown) != DURABLE_SHAPE_DIGESTS["eval_template"]


def test_the_durable_addressing_fields_are_pinned_and_are_fields() -> None:
    assert dict(DURABLE_ADDRESSING_FIELDS) == PINNED_DURABLE_ADDRESSING_FIELDS
    for doc_type, model in _durable_models().items():
        missing = DURABLE_ADDRESSING_FIELDS[doc_type] - set(model.model_fields)
        assert not missing, f"{model.__name__} has no field {sorted(missing)} the store addresses it by"


# --- the version and its chain: a durable major always ships its step ------------------------------


def _steps_missing(steps: tuple[CoreUpgrader, ...], baseline: int, version: int) -> list[int]:
    """The majors from ``baseline`` below ``version`` with no registered step from them."""
    registered = {step.from_version for step in steps}
    return [major for major in range(baseline, version) if major not in registered]


def test_every_durable_major_ships_its_upgrader() -> None:
    assert _steps_missing(DURABLE_UPGRADERS, DURABLE_BASELINE_VERSION, DURABLE_SCHEMA_VERSION) == []
    assert [step.from_version for step in DURABLE_UPGRADERS] == list(
        range(DURABLE_BASELINE_VERSION, DURABLE_SCHEMA_VERSION)
    )
    assert all(step.doc_types <= DURABLE_DOC_TYPES and step.reason for step in DURABLE_UPGRADERS)


def test_a_durable_major_without_an_upgrader_is_named() -> None:
    """The rule above, on a made-up chain, so it is seen firing while the real chain is empty at the baseline."""
    step = CoreUpgrader(from_version=15, doc_types=frozenset({"eval_template"}), reason="r", upgrade=dict)

    assert _steps_missing((), 15, 16) == [15]
    assert _steps_missing((step,), 15, 17) == [16]
    assert _steps_missing((step,), 15, 16) == []


def _ledger(start: str, end: str) -> str:
    source = inspect.getsource(versioning)
    return source[source.index(start) : source.index(end)]


def test_every_durable_version_and_minor_has_a_ledger_entry() -> None:
    majors = _ledger("DURABLE_BASELINE_VERSION: Final", "DURABLE_SCHEMA_VERSION: int =")
    minors = _ledger("DURABLE_SCHEMA_VERSION: int =", "DURABLE_SCHEMA_MINOR: int =")

    for major in range(DURABLE_BASELINE_VERSION, DURABLE_SCHEMA_VERSION + 1):
        assert f"#: - **v{major}** —" in majors
    assert f"#: - **v{DURABLE_SCHEMA_VERSION}.0** —" in minors
    assert not minors_missing_the_sentence(minors, DURABLE_SCHEMA_VERSION, DURABLE_SCHEMA_MINOR)


def test_the_standing_rule_is_in_the_durable_ledger() -> None:
    """The owner's and the consumer's ruling, recorded where the next bump is written."""
    ledger = " ".join(
        line.removeprefix("#:").strip()
        for line in _ledger("DURABLE_BASELINE_VERSION: Final", "DURABLE_SCHEMA_VERSION: int =").splitlines()
    )
    assert "the evidence core is baseline-resettable forever" in ledger
    assert "is the ONLY place a shape change must ship an upgrader; a core baseline reset never touches it." in ledger


# --- the frozen corpus -----------------------------------------------------------------------------


def test_every_durable_version_has_a_whole_frozen_corpus() -> None:
    for version in range(DURABLE_BASELINE_VERSION, DURABLE_SCHEMA_VERSION + 1):
        held = {path.stem for path in (FIXTURES / f"v{version}").glob("*.json")}
        assert held == DURABLE_DOC_TYPES, f"durable v{version}'s frozen corpus holds {sorted(held)}"


def test_no_frozen_durable_fixture_was_edited() -> None:
    found = {
        path.relative_to(FIXTURES).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _fixture_files()
    }
    assert found == FIXTURE_DIGESTS, "a stored durable document of the past was edited, added or removed; restore it"


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda path: path.relative_to(FIXTURES).as_posix())
def test_every_frozen_durable_document_reads_under_this_build(path: Path) -> None:
    """A durable document any earlier build stored still reads, at the current durable version."""
    document = json.loads(path.read_text(encoding="utf-8"))
    model = _durable_models()[document["doc_type"]]

    read = model.from_dict(document)

    assert read.schema_version == DURABLE_SCHEMA_VERSION  # type: ignore[attr-defined]
    assert read.schema_minor == DURABLE_SCHEMA_MINOR
    if (document["schema_version"], document.get("schema_minor", 0)) == (DURABLE_SCHEMA_VERSION, DURABLE_SCHEMA_MINOR):
        assert read.to_dict() == document, "a document at the current version reads back whole"
    else:
        # A later minor's fields are written at their defaults, at any depth (a template's dims gained ``judge``);
        # every value the document stored is kept as it was.
        rewritten = read.to_dict()
        assert keeps({key: value for key, value in document.items() if key != "schema_minor"}, rewritten), (
            "a document at an older minor reads back with every stored value as it was"
        )
