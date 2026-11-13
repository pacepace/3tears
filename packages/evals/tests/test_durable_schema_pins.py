"""The durable space's shape is pinned, and it is the one space where a major must ship an upgrader.

``schema/versioning.py`` splits what people author (templates, rubric dims, judge configs and their tombstones, case
sets, test cases, calibration ratings, since durable v16 campaigns, and since a later minor the analyst's gap notes) out of
the evidence core into a version space of its own, forked from core v15 at ``.26``. The core may reset its baseline (a stamp and a refusal, no steps); the durable space never does, so
this file is the gate that holds it to its promise:

- every durable type's validation JSON schema, embedded shapes included, is pinned to a digest under the durable
  major and minor recorded here, so a shape change without a version goes red here — including a change to a shape
  the core embeds too (a ``RubricDim``, a ``JudgeEvidence``), which takes the durable document's space;
- the durable baseline is 15, forever: a build that moved it would refuse labels people gave;
- every durable version from the baseline has a ledger entry and an upgrader, and every minor's entry carries
  ``MINOR_ADDITIVITY_SENTENCE``;
- the frozen corpus under ``tests/fixtures/durable_documents/`` (the ``.25`` documents, moved here unedited from the
  core corpus, and the campaign's v15 document, moved the same way when it became durable at v16) keeps reading
  under this build, an older major's through the registered steps.

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

from packages.evals.tests.schema_pin_support import ADDED_AT_MAJOR, keeps, minors_missing_the_sentence
from packages.evals.tests.stored_models import doc_type_of, stored_models
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
    upgrade_durable_document,
)

#: The durable baseline, which never moves.
NEVER_MOVES = 15

#: The ``DURABLE_SCHEMA_VERSION`` every digest below was taken under.
PINNED_DURABLE_SCHEMA_VERSION = 17

#: The ``DURABLE_SCHEMA_MINOR`` every digest below was taken under.
PINNED_DURABLE_SCHEMA_MINOR = 0

#: Each durable type's validation JSON schema, normalised as the core pins normalise it, by ``doc_type``.
DURABLE_SHAPE_DIGESTS: dict[str, str] = {
    "calibration_rating": "8477ebd5e374d893776e65a7e731e0f38631146dc58ce107d5ee12e849d2ea46",
    "case_set": "8f0165d26d157517528f3ee5d3e09cff75405025ae3643abbbf5f661f99ae197",
    "eval_analyst_gap": "e0ac1f149ed22d466e610dff66dcc1c34af0ddc552a47dac5a153ff6e5d62836",
    "eval_campaign": "96ceb75b5776dd0c5812054dcac4d013357131035d10087ca2b43af09b556f44",
    "eval_template": "95d91727e479a9e99529ceb4bc2cc2c3dc0d2b62574cc3ac14a0d5cd6e1179b4",
    "eval_test_case": "97a970e4edfe2100ff9b65de77fc83d164a133764833f35c7b792aec890be5fb",
    "judge_config": "45695e6285c81ac8033850d9a8e62bfd227ac5cb86ab0de0e756fbb34d151faa",
    "judge_config_tombstone": "27d8ee74be598ae29c60bd1f3e865d819bded0f623cfa58ae6b1c6f92a9ae4bc",
    "rubric_dim": "c857ffe7dd2ce098fca9578474350074c9fb7a9f7e6e71601d85bf2a077a2793",
    "rubric_dim_tombstone": "9737ebc5236c74117d12d9232630daf544076e0643504096675ace2dacc44507",
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
    "eval_campaign": frozenset(
        {"id", "doc_type", "schema_version", "scope_id", "subject_id", "behavior", "archived", "created_at"}
    ),
    "eval_analyst_gap": frozenset({"id", "doc_type", "schema_version", "scope_id", "created_at"}),
}

#: The frozen durable corpus: one directory per durable major, one stored document per durable type.
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "durable_documents"

#: Each frozen durable fixture's sha256: the ``.25`` documents and the campaign's v15 one, moved from the core corpus
#: with these exact bytes, the v16 corpus as the build before the lever's reader name wrote it, and the v17 corpus
#: this build writes.
FIXTURE_DIGESTS: dict[str, str] = {
    "v15/calibration_rating.json": "894f74e31dcdb42e26cb949b080cbcb2e76bb20d8361ed1c17bc74a9c7864e7e",
    "v15/case_set.json": "1a30692918a1dcd4ff1702bf3c2095133e6652021e30961f29a8528a673cd082",
    "v15/eval_campaign.json": "b2f12dfb925d69d599de0d9040a2f1a5bfed57d268aa60ca87cc57eb4c9b46bb",
    "v15/eval_template.json": "4cf70858e2c14deabf3af88055d8e31dcba1bbb0253ae953baa23998be8abebf",
    "v15/eval_test_case.json": "504dcf34cff61440e782e30322cf9cc42cf681fbbb866faddff9aef48ee405cd",
    "v15/judge_config.json": "d3f29495b08a71effd16ad0b64b9f28ccff653aa2e21bf858a795099af4a1c96",
    "v15/judge_config_tombstone.json": "b75f4d2f2597723a72655afe3cf852de1bcd21bc43c90c18c841e50264d33a4b",
    "v15/rubric_dim.json": "b65c88b862dddf2723ea3972be8381cca002131d7d9ae524418602d1bc2f6378",
    "v15/rubric_dim_tombstone.json": "da3fb751ac55bf182f339faf233fc65aeeb0d467044c78ee9709844761042cef",
    "v16/calibration_rating.json": "085d1e7f28b15a7b6fffc850dca73c88bb7afcc543169367fd42bb1b21ee775c",
    "v16/eval_analyst_gap.json": "b1c02467eb7abad961ae645d37ddc59c9fe203f4349d007ec4f594f94b67d9c6",
    "v16/case_set.json": "cae61566ce663fc6eff2ddb4604e6b5a066e4b347bd998ea391647e2dcb7730c",
    "v16/eval_campaign.json": "b074a1ff416888f86382dd13997f788805ab8429c3a65330ffa59fb0f0d4f70e",
    "v16/eval_template.json": "0ffbc44743965bd0786cf8d8ad0d2482cc0c96a8a0628bd93ab82cca6dcdd113",
    "v16/eval_test_case.json": "11c704d4c76a2ad0fa591720bf388b7c489484c4ab29182085a8e50fe6610e77",
    "v16/judge_config.json": "62d655a09e85d57fd1142bc3ea3e2241f7293718e7300d4f53ced8406bbb36a9",
    "v16/judge_config_tombstone.json": "4831dcf1c41fa3725f578d2119d046473c832cec7fb25a3ab04fa57d5b26eea7",
    "v16/rubric_dim.json": "34a340255217cbafb3b9be7c6517d3b200b77aa07aedf77c32f06b372db3bc45",
    "v16/rubric_dim_tombstone.json": "374d32a4b7497c976dec064095209bffd224d861dac6f17b492366e60034ad09",
    "v17/calibration_rating.json": "64213ddc7847462b483759ce1ee07928b2245c1c6572deabaebe1801d3044370",
    "v17/eval_analyst_gap.json": "a3a45b37c6417984f079bc78856bfd29a852bd62a8b2971eabe91ea202220258",
    "v17/case_set.json": "60fa982745ed59772ca13d6a5637efd2abf6d8226fe101635370e0fae4a08129",
    "v17/eval_campaign.json": "0613719edaf6c065fc0a6c0a8a817201f4816b9a46228e488c7cdbc7215d9556",
    "v17/eval_template.json": "0545ce364d5b9d53bc4d43f72dcda0fe94d3c9a85df18a154c4fed2603794a33",
    "v17/eval_test_case.json": "beee7e08a6e2eb89c221f609bc0f080f5b7004427434d643f89ec14561614abd",
    "v17/judge_config.json": "7fd4653547ff67d5b376a778c16ffec8c5b822777c2e9d7a5d14665e22242743",
    "v17/judge_config_tombstone.json": "8c91225ea1b2d7aede15d8a3d5f2bd2c2053a0a1934f9b06021f7f3b38e3a230",
    "v17/rubric_dim.json": "f22602972b2a3f87e2d6e351ebd0a35a5d8641683f655724ba253932513df012",
    "v17/rubric_dim_tombstone.json": "e9bd18ac6b00fbcaeafb76f608f2e6699b45c2df227e19b7fa398da7bc7e7c07",
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
    """The rule above, on a made-up chain, so it is seen firing whatever the real chain holds."""
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
    """A major's corpus holds every type that existed at it; a type added later has no earlier document to freeze."""
    assert set(ADDED_AT_MAJOR) <= DURABLE_DOC_TYPES
    for version in range(DURABLE_BASELINE_VERSION, DURABLE_SCHEMA_VERSION + 1):
        held = {path.stem for path in (FIXTURES / f"v{version}").glob("*.json")}
        existed = {kind for kind in DURABLE_DOC_TYPES if ADDED_AT_MAJOR.get(kind, DURABLE_BASELINE_VERSION) <= version}
        assert held == existed, f"durable v{version}'s frozen corpus holds {sorted(held)}"


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
        # An older major is read through the registered steps, which rewrite only the types they name; a later
        # minor's fields are then written at their defaults, at any depth (a template's dims gained ``judge``). Every
        # value the stepped document holds is kept as it was.
        rewritten = read.to_dict()
        stepped = upgrade_durable_document(document)
        assert keeps({key: value for key, value in stepped.items() if key != "schema_minor"}, rewritten), (
            "a document at an older version reads back with every value its upgrade holds as it was"
        )


def test_a_frozen_v15_campaign_reads_as_one_aspect() -> None:
    """The one type the v15 step rewrites: its template, runs and design move into one aspect labelled behavior."""
    document = json.loads((FIXTURES / "v15" / "eval_campaign.json").read_text(encoding="utf-8"))
    campaign = _durable_models()["eval_campaign"].from_dict(document)

    (aspect,) = campaign.aspects  # type: ignore[attr-defined]
    assert aspect.label == document["behavior"]
    assert aspect.template_id == document["template_id"]
    assert aspect.run_ids == document["run_ids"]
    assert campaign.arms[0].variant_key == document["declared_design"]["control"]  # type: ignore[attr-defined]
    assert "declared_design" not in campaign.to_dict()


def test_every_other_durable_type_passes_the_v15_step_unchanged() -> None:
    for path in sorted((FIXTURES / "v15").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["doc_type"] == "eval_campaign":
            continue
        assert upgrade_durable_document(document) == {**document, "schema_version": DURABLE_SCHEMA_VERSION}


def test_a_frozen_v16_campaign_reads_with_every_axis_named() -> None:
    """The v16 step names each declared axis that stores no reader name from its id, once; nothing else moves."""
    document = json.loads((FIXTURES / "v16" / "eval_campaign.json").read_text(encoding="utf-8"))
    campaign = _durable_models()["eval_campaign"].from_dict(document)

    (axis,) = campaign.aspects[0].design.axes  # type: ignore[attr-defined]
    assert (axis.axis_id, axis.reader_name) == ("model", "model")
    stepped = upgrade_durable_document(document)
    stepped["aspects"][0]["design"]["axes"][0].pop("reader_name")
    assert stepped == {**document, "schema_version": DURABLE_SCHEMA_VERSION}, (
        "the step names axes and moves nothing else"
    )


@pytest.mark.parametrize(
    ("axis_id", "named"),
    [
        ("model", "model"),
        ("callable.prompt", "prompt"),
        ("callable-judged.effort", "effort"),
        ("extractor.prompt_style", "extractor prompt style"),
        ("max_output_tokens", "max output tokens"),
    ],
)
def test_the_v16_step_names_an_unnamed_axis_from_its_id_and_keeps_a_named_one(axis_id: str, named: str) -> None:
    """The step names an axis by the one rule that reads a lever's id as words, less the quick path's kind namespace."""
    document = json.loads((FIXTURES / "v16" / "eval_campaign.json").read_text(encoding="utf-8"))
    axis = document["aspects"][0]["design"]["axes"][0]
    axis["axis_id"] = axis_id
    kept = json.loads(json.dumps(document))
    kept["aspects"][0]["design"]["axes"][0]["reader_name"] = "Reasoning budget"

    assert upgrade_durable_document(document)["aspects"][0]["design"]["axes"][0]["reader_name"] == named
    assert upgrade_durable_document(kept)["aspects"][0]["design"]["axes"][0]["reader_name"] == "Reasoning budget"


def test_every_other_durable_type_passes_the_v16_step_unchanged() -> None:
    for path in sorted((FIXTURES / "v16").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["doc_type"] == "eval_campaign":
            continue
        assert upgrade_durable_document(document) == {**document, "schema_version": DURABLE_SCHEMA_VERSION}
