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
PINNED_CORE_SCHEMA_VERSION = 11

#: Each core type's validation JSON schema, normalised (:func:`core_shape_digest`), by ``doc_type``.
CORE_SHAPE_DIGESTS: dict[str, str] = {
    "calibration_rating": "9f29a51348412ab4d973a53602e090ab4ea8e5702096dc68135cf840c29e3b1d",
    "case_set": "b197505c75b1b2db6c0996ce20c2ae0bbbb055ce72d325d3c7d9430b39716be3",
    "eval_analysis_attempt": "85f7de49dce7526edcaf0caeaa3706c6248e2372222c1b3e3fe23873c24a6d99",
    "eval_campaign": "f3abf049a650270b1d82e65a90ef6604ab3c1944c7add140437d69995cca5477",
    "eval_generation_slot": "701de910c9d0f871d349eeae48482c3942bce0c932cc92f20a6365208407bc74",
    "eval_launch": "71bef2151fe48407e6e72b300ec5fcf090656930ea4d633324bea5ff90ee08ae",
    "eval_second_judge_pass": "e7f81e065c0039e1ba821998330e9149bcc95760e2ea1c09d2c0938244b80fb6",
    "eval_judge_temperature_check": "6c9bf5bdafc7358cee2618347f4fab5a301c14586bb7e48ff27dbd4d47f0eb46",
    "eval_action_job": "8c6841e9b249c78bae79d373137d5b164f9f38d0de0fe330ea662f75f57c77e0",
    "eval_out_of_run_spend": "6d0dfe8830220f19e50dc76ce99c9ff884405bf899f96dc93cb81b34b51479e0",
    "eval_result": "8a4d2a2b0b61c96c410d64a6a1756f610c0f9d2fe3def39e35737b43829fedde",
    "eval_run": "774a10d12790e3675d710ba69fecf1add97b5b7b4e266705164386375f7f9e33",
    "eval_sweep": "ba39dd270c17fe50295b822fb20fefc9e55a24973bfe3194b2b01b586fe5456f",
    "eval_template": "cc03471c04c39ad410a83041b6934a4e30e9305da68ec471133a6f779ee08cff",
    "eval_test_case": "c8139138501db54667407e9b4eb1796dfd79486d2c815fc694f216c072bdb3e2",
    "eval_trace": "8e1980b919b1ec0af96be8b60660a7227a61b899613bdfb7f091b1c7fd94fc8d",
    "judge_config": "96bf544c6aa654a9268c313e9894459153542f0c9850fefeccc52a77508de723",
    "judge_config_tombstone": "ec120d419da986810a631eb8b22c6c0dd6c1e077886ee57a0574310043c8caab",
    "rubric_dim": "9dceb1d549784ee0240bb615c01882aba85e7732d74bb499cad0476c443abba8",
    "rubric_dim_tombstone": "a0dbdaa3194eea2320398650c036fe9307ff7ba5ca09c1b4b430c9c34097c04c",
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
FIRST_STORED_AT: dict[str, int] = {
    "eval_generation_slot": 10,
    "eval_launch": 10,
    "eval_second_judge_pass": 10,
    "eval_judge_temperature_check": 10,
    "eval_action_job": 10,
}

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
    "v10/eval_generation_slot.json": "7de32af29779e8be2afe49db69959ff3a03017aefc843555bfab69fbe80f923b",
    "v10/eval_launch.json": "09e109cea826386249b47e9d62c5a0ac7258f965f377827db19483a8e13fc435",
    "v10/eval_second_judge_pass.json": "dcc1965334813457a0c97d23760bcd143322e1fc25b4097ee4af6a7f0b13a74d",
    "v10/eval_judge_temperature_check.json": "1e018414a97497027c6506513a466c0a48f26eda9b15658e9d4f1aa0ac38f5ab",
    "v10/eval_action_job.json": "93991981cfd6b6ee924b881666a85fb10dcad29867bcdad96ff566b202d28919",
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
    "v11/calibration_rating.json": "0a8dd33756e8b0253a1a48a0dbba7c725fe16efe130ca8ba105c1ef2c104c6cf",
    "v11/case_set.json": "84deeb6b5185d6fc28fff42b2500ccd83262f4fea67edb735e25f4026b7a6316",
    "v11/eval_action_job.json": "cd388702a22470e5a7be912e189e63a0beb0b46df0a4c8121e1847ec36527508",
    "v11/eval_analysis_attempt.json": "74f6176ccf33b48d0e17c0a97ec6f21b358bb9d41e855fbcef8a04b2364ae991",
    "v11/eval_campaign.json": "a4b58a8531c393c3bf5dcaa0c01d943fef6555f971e787709071318cb47be53c",
    "v11/eval_generation_slot.json": "e17183029adda21efa06db5ab0cb85b80c92c7e3da54d7c6d1fcafd100e6d079",
    "v11/eval_judge_temperature_check.json": "3b2eef3ea16511a097cf591b82a1510d6e10b7d2e28df40ca6980e30980d79d2",
    "v11/eval_launch.json": "9156678e1618bb9693827446da62b3156c369b18d50c7ba9e61afeb5681088be",
    "v11/eval_out_of_run_spend.json": "13bc8cd74982bc6cb55dac202e076e58b30481cee4d78e9141b82a040527759f",
    "v11/eval_result.json": "a86ffb2838bc8a95773e4fb83ccca17d42cac07ea371694950f5b7b8ae28f6e6",
    "v11/eval_run.json": "5aaca832f1b453408fa717829b51e215e0c3040f4b4e3ec4a4746594160ff397",
    "v11/eval_second_judge_pass.json": "193ff2385b962952c6cc6efeae1a102df999e1e14727a080fd096c948b54114f",
    "v11/eval_sweep.json": "2a5ce79c8f20a5a393b99f99778db907d35afed55166ba4c763acb7482ccc3a6",
    "v11/eval_template.json": "2e9dec5f128aab9658d6e1f29394d6f62ef429e5ef5ba5edeeddcab07a06442d",
    "v11/eval_test_case.json": "ecfc1fba6c64654bed87c3facd7e5d8d9662b99034117c60772c1258794ccb94",
    "v11/eval_trace.json": "78cd87b9b8c93e2ea80abdcd59d1376442471232b5cb741fe8b247bb06e6d6b2",
    "v11/judge_config.json": "d4f60ced0146774f1149856fe3836923a0cf536bea7d29eb3f4bd8b88e1efe89",
    "v11/judge_config_tombstone.json": "c163c5b49bd7f2b6052ed6d9b03f9c5511969cbb8ec6defae0bcbe3e9169e88d",
    "v11/rubric_dim.json": "434352140759ca2997d12c7c02d67cf4d77f807dbdfc53b52d92d5a6d8717a06",
    "v11/rubric_dim_tombstone.json": "6144094aa45a5a264e4fbe3c5d3a8beea636649c0e0094dd7a9b805612e6fe6b",
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
