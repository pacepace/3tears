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
PINNED_CORE_SCHEMA_VERSION = 10

#: Each core type's validation JSON schema, normalised (:func:`core_shape_digest`), by ``doc_type``.
CORE_SHAPE_DIGESTS: dict[str, str] = {
    "calibration_rating": "a543edb034dbeb7a7956d697939f665cc5c9f6c380419dbcf70e95e27f343772",
    "case_set": "67cd75fe2dcb16a85fe8ecbda3ee0890addf6913d64ac1b95a93a64b06625d52",
    "eval_analysis_attempt": "5fcf17735d8a4791fa8b1cb9eed7a59458c67cc0c787f39d210ebf5b343085e8",
    "eval_campaign": "c33c6dc1d6c9fba98d9887bc20c395eac5ca6245302eb0fb32d696b155fab8d5",
    "eval_out_of_run_spend": "f1d51ffd4cb2447bc1ca32c67ac20a184a6fa17665c3f22bae7ba73bb0f46f43",
    "eval_result": "84be2b0ee738171611a7173793061d06a7297d952b5a0aba41ecdb3207487cb5",
    "eval_run": "32fdaebc9f3eaca01c2ead896dede4cd9a2a9283980b9338961578e77c3bf1c5",
    "eval_sweep": "17f38b8fb24fb416695a5ed2c4ef7f6a4702a18e0a0a1e470dd0f15769640c90",
    "eval_template": "8f9ae4cc32fe1a11fcf08007b7c6ef96767c475e3d7f220222f92b7cd9763231",
    "eval_test_case": "1e928fb0d3d646aa8c7451385d04417d0f42b70239da88ae4929d23f0f8de87f",
    "eval_trace": "fb24fe2a75c0ebe5586510326023406c98ee0a0fb701fc9db4988020b4a7b084",
    "judge_config": "77088727ea463a92164453b579715d057c8622c6dc22e757c3e42584c3760af2",
    "judge_config_tombstone": "c933c26a6b232648508cfdc75b806ab176d262d758e3a814956d7a6639d0672c",
    "rubric_dim": "6b96268625583c6b89e252223496da9e61fe13c3e840c0da5ca21919d0078fc5",
    "rubric_dim_tombstone": "5baef0da563d3d2943f0b254d13d3796d391e0f5b5dd0baf40564738966a9ae0",
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

#: Each frozen fixture's sha256, so a stored document of the past is never edited.
FIXTURE_DIGESTS: dict[str, str] = {
    "v8/calibration_rating.json": "ae0d27591a4335b1e22fcf023dd69cd5c25a5079febfcf85f65daf4e85b27214",
    "v8/case_set.json": "8dbf4dc88ffc44d95e4dc9d13a344d1479321893b635812c415defadd67c8fb4",
    "v8/eval_analysis_attempt.json": "f68876fc90a3a3a9374d949460d81eaa107f4035461a19781770c0d26a31da1e",
    "v8/eval_campaign.json": "7b240be0794eb3621310f399e90288c84717af0caec4c993fd88f26a9127f084",
    "v8/eval_out_of_run_spend.json": "234922bc7364b5c390728a54b0c37f80ded81c913ffd068bd30a21642cbb61f7",
    "v8/eval_result.json": "0ec4c7bc2001e3923c26ee79c728aaf80ce4212c7d0403c0e528c6d25ac582f8",
    "v8/eval_run.json": "859ff20570f63d78963c51471decf76218f4ef2312e8de8e889f9776513fa56d",
    "v8/eval_sweep.json": "ffe9e179d8493b8162a4bd04f867d2ed3ac130801ace6b7179bf7d6e58062f5a",
    "v8/eval_template.json": "8b9174fbcee49a61f7e0b863399e34af37644da5cc074e5e18a9e1a6c3062e3c",
    "v8/eval_test_case.json": "3a15ab6414bb2de1de0df80f8841fbbc631ecb3c7d85fccf84a3053a7e219de0",
    "v8/eval_trace.json": "1d0b69fad9e25b56764fda5c3885994e9587a89f48a01e30c728af7f11883358",
    "v8/judge_config.json": "cf87a09a0dffc6decaeac9f340a33daa276be754ef85207ee68aa8463e02ced2",
    "v8/judge_config_tombstone.json": "20a27fce783aa93b221e2fce9cccf92580ab635b8fad319829288e6931493dfb",
    "v8/rubric_dim.json": "fdbdab2e2a5283ee14b988ead71c25be8be21ea19256a5d31ce433b81a3ef36c",
    "v8/rubric_dim_tombstone.json": "4198ff3163fc2cc1b2de7b67fb0dddb4a87d370551725e4e20f19af40ade3529",
    "v9/calibration_rating.json": "41b8f8a8c87b832148090ea42e5a215980810fab4a30716f0111656b3a1c6c9f",
    "v9/case_set.json": "883b7aece975b2727571ce8cb931204112b44da58adc2dbc94eb933514585ae5",
    "v9/eval_analysis_attempt.json": "d9e693bc468d85e5dbf9cd4e5b94b88b9eb527305d17d89500f610c7f607f035",
    "v9/eval_campaign.json": "47fdd75fc22140e30d3746944f8f78305a694199a98519df2888880e8ea98bc5",
    "v9/eval_out_of_run_spend.json": "a5bb8d170690525b99b8a89d50551cf6a13e85cc35927f9ca0ce86cbbcde966f",
    "v9/eval_result.json": "add5bcd7b841d0abbd285f11390b7078abc32178aeacf48de9110c47025771ca",
    "v9/eval_run.json": "b7d71bf1aac26afb13d405046127f82e10ce5340e21da03cf766cb560ca02582",
    "v9/eval_sweep.json": "68c3fcb2b9c64837b46befb3c366ae5e46b5988a74f9edd1b0e905e07565c01b",
    "v9/eval_template.json": "c766828c84b560109267571c35987ce6d50ee30d9b0251d8bf9d8054e7180155",
    "v9/eval_test_case.json": "6e5f26208443cfd966a5bc977bbd4a4a52ba8c745fb9bd0b981fb1ea99a2ba7e",
    "v9/eval_trace.json": "eb8ffa40da641f26b0f75d86aa4f697b03e6ad895c3e15d37d8f6fabe5624e85",
    "v9/judge_config.json": "c7c35db9a714df3de6bf71e5c04264982162b6a833ba66e767a164faebdb6d79",
    "v9/judge_config_tombstone.json": "d2c5be28efc12f564ec604374f148021b15e67bb4b5ea935e171b79dae9a28a6",
    "v9/rubric_dim.json": "e81b15e51c39604bd295d3d7f49e2efb3b20e6969eefe4a4cd09e7a1cefbe09b",
    "v9/rubric_dim_tombstone.json": "3c8058911dd4cfa1a87bf3ea0e5ba7b62b4ed11ce94dca8c63b44860d242e7bd",
    "v10/calibration_rating.json": "bd29da501e7cb6a2e2c54f14783c90f3c3e5f2fa6313ad3b70e5de27fda532fe",
    "v10/case_set.json": "d5d2c352b4effea075cf6f05a7b4ece0789120c48bae5faf371d25c7117635ad",
    "v10/eval_analysis_attempt.json": "d8254f8c6a09b5e5a653b98a89db576ace68c82fc32fe4da6a162cfe7ac27bf4",
    "v10/eval_campaign.json": "ce49d7fa609312692d79d5fdb7c3661becb056d9c7bd8cb60828d97a46b3c4e2",
    "v10/eval_out_of_run_spend.json": "09d29852a0078c3750de7f0443e37927a685978a064098d3a398118184b7250b",
    "v10/eval_result.json": "072ab69a839032ca0395c91adc93aa6d0a5913cf62c138f4b7ba0df5a034e27d",
    "v10/eval_run.json": "56b1cdd79e8754004d0658d5ed926a7f0acf609bdeff4e07724ccf7d2f9f158f",
    "v10/eval_sweep.json": "ad6f534d12f194819e9a4bb9f56dde8d0ae00cfb4ac33975863131de70aaa3d5",
    "v10/eval_template.json": "5f2ebd9cfb97b037141adcfd51597a0a4cc01d33243a117320fe609bacc13e61",
    "v10/eval_test_case.json": "30c4e9297c56ab3791c98802459f83b7ad6f8246b9e885fecb561f164fe8e8c9",
    "v10/eval_trace.json": "3d99d43d31546bfdfbbd53381b660c59346d52b1c464e4c4738b3b6bcbcb6884",
    "v10/judge_config.json": "018fb9a4b3d9d7058306b0de6f11d94fef4d687fd376fcbef4afc3fa08d5ea31",
    "v10/judge_config_tombstone.json": "8a81a65775ba7e0dd4e571a2ed7bc50d80fa77a60ab4cb719a45edf1304b665e",
    "v10/rubric_dim.json": "4bb70cede631382f55912d1a2d7e31b53adcc9728b785baa09dd4558c52f04b7",
    "v10/rubric_dim_tombstone.json": "12174bd3c3ac1de46b5e45a8a72f4602fb2c8edb1e3662623ceefc15f86cfece",
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


def test_no_core_type_retires_a_field_within_a_version() -> None:
    """A core type changes only by a bump with an upgrader; ``__retired_fields__`` is the regenerable path."""
    retiring = [model.__name__ for model in _core_models().values() if getattr(model, "__retired_fields__", {})]
    assert retiring == []


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
    """One stored document per core type, for every version from the baseline: what an old store holds."""
    for version in range(CORE_BASELINE_VERSION, CORE_SCHEMA_VERSION + 1):
        held = {path.stem for path in (FIXTURES / f"v{version}").glob("*.json")}
        assert held == CORE_DOC_TYPES, f"v{version}'s frozen corpus holds {sorted(held)}"


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
