"""The core's upgrade runner: an older core document reads through its upgraders, and nothing else reads.

The runner is driven here, apart from the steps ``CORE_UPGRADERS`` registers, through its public front door, :func:`upgrade_document`, with
synthetic chains, against a real core document from the frozen corpus and a real core model. A
test-defined core model reads through ``from_dict`` to show where the upgrade sits: before every
validator, the model's own included.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError, computed_field, model_validator

from threetears.evals.schema.base import CoreDocumentModel
from threetears.evals.schema.models import CoreSchemaVersion, EvalRun
from threetears.evals.schema.versioning import (
    CORE_SCHEMA_VERSION,
    CoreUpgrader,
    CoreVersionRefused,
    upgrade_document,
)

#: A run as the current build stored it.
_RUN = json.loads(
    (
        Path(__file__).resolve().parent / "fixtures" / "core_documents" / f"v{CORE_SCHEMA_VERSION}" / "eval_run.json"
    ).read_text(encoding="utf-8")
)


def _rename(old: str, new: str) -> Any:
    def step(document: dict[str, Any]) -> dict[str, Any]:
        document[new] = document.pop(old)
        return document

    return step


def _add(field: str, value: Any) -> Any:
    def step(document: dict[str, Any]) -> dict[str, Any]:
        document[field] = value
        return document

    return step


#: The current core version, which the synthetic chain below ends at.
_NOW = CORE_SCHEMA_VERSION

#: A synthetic two-step history for the run, ending at the current version: two versions back the judge's
#: temperature was stored as ``judge_temp``, and the next version renamed it; the current version added
#: ``cell_concurrency``, which a run one version back never recorded, so its step writes None, "not recorded".
_CHAIN = (
    CoreUpgrader(
        _NOW - 2,
        frozenset({"eval_run"}),
        "judge_temp renamed judge_temperature",
        _rename("judge_temp", "judge_temperature"),
    ),
    CoreUpgrader(_NOW - 1, frozenset({"eval_run"}), "cell_concurrency joined", _add("cell_concurrency", None)),
)


def _as_two_back(run: dict[str, Any]) -> dict[str, Any]:
    """The same run as the synthetic build two versions back would have stored it."""
    older = {key: value for key, value in run.items() if key not in ("judge_temperature", "cell_concurrency")}
    return {**older, "judge_temp": run["judge_temperature"], "schema_version": _NOW - 2}


def _upgrade(document: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return upgrade_document(document, **{"steps": _CHAIN, "current": _NOW, "baseline": _NOW - 2, **kwargs})


# --- a previous version reads through its upgraders ------------------------------------------------


def test_a_run_from_two_versions_back_reads_as_the_run_it_was() -> None:
    """Through both steps, the run from two versions back validates strictly as the run the current build stored."""
    upgraded = _upgrade(_as_two_back(_RUN))

    assert upgraded["schema_version"] == _NOW
    assert EvalRun.model_validate(upgraded) == EvalRun.from_dict(_RUN)


def test_a_run_from_the_previous_version_takes_only_the_last_step() -> None:
    previous = {key: value for key, value in _RUN.items() if key != "cell_concurrency"} | {"schema_version": _NOW - 1}

    assert EvalRun.model_validate(_upgrade(previous)) == EvalRun.from_dict(_RUN)


def test_a_type_a_step_does_not_name_gets_only_the_number() -> None:
    other = {"doc_type": "eval_result", "schema_version": _NOW - 2, "judge_temp": 0.0}

    assert _upgrade(other) == {"doc_type": "eval_result", "schema_version": _NOW, "judge_temp": 0.0}


def test_the_stored_document_is_not_modified() -> None:
    stored = _as_two_back(_RUN)
    before = json.dumps(stored, sort_keys=True)

    _upgrade(stored)

    assert json.dumps(stored, sort_keys=True) == before


def test_a_current_document_is_returned_unchanged() -> None:
    assert _upgrade(dict(_RUN)) == _RUN


# --- and nothing else reads ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "why"),
    [
        (_NOW + 1, "by a newer build"),
        (_NOW - 3, "before the first public release"),
        ("7", "not a core version number"),
        (7.0, "not a core version number"),
        (True, "not a core version number"),
        (None, "not a core version number"),
    ],
)
def test_a_version_the_chain_does_not_read_is_refused(version: object, why: str) -> None:
    with pytest.raises(CoreVersionRefused, match=why):
        _upgrade({**_RUN, "schema_version": version})


def test_a_chain_with_a_missing_step_is_refused() -> None:
    with pytest.raises(CoreVersionRefused, match=f"no core upgrader is registered from v{_NOW - 2}"):
        _upgrade(_as_two_back(_RUN), steps=_CHAIN[1:])


def test_a_step_that_leaves_a_key_behind_is_refused_by_the_strict_read() -> None:
    """Rule 4: a step's output validates strictly, so a step that copies instead of moving is caught."""
    leaky = (CoreUpgrader(_NOW - 2, frozenset({"eval_run"}), "copies", _add("judge_temperature", None)), _CHAIN[1])

    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvalRun.model_validate(_upgrade(_as_two_back(_RUN), steps=leaky))


# --- where the upgrade sits: before every validator ------------------------------------------------


class _Kept(CoreDocumentModel):
    """A core document with a validator of its own and a computed field, to see what each is handed."""

    id: str
    doc_type: Literal["kept_for_test"] = "kept_for_test"
    schema_version: CoreSchemaVersion = CORE_SCHEMA_VERSION
    scope_id: str
    note: str

    @model_validator(mode="before")
    @classmethod
    def _sees(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = {**data, "note": f"{data.get('note')} (validator saw v{data.get('schema_version')})"}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        return f"{self.scope_id}/{self.id}"


def _kept(**fields: Any) -> dict[str, Any]:
    return {
        "id": "k-1",
        "doc_type": "kept_for_test",
        "schema_version": CORE_SCHEMA_VERSION,
        "scope_id": "s",
        "note": "n",
    } | fields


def test_a_core_document_reads_through_from_dict_with_its_own_validators_after_the_upgrade() -> None:
    read = _Kept.from_dict(_kept(label="an echo, discarded"))

    assert read.note == f"n (validator saw v{CORE_SCHEMA_VERSION})"
    assert read.label == "s/k-1"


def test_a_refusal_comes_before_any_validator_and_as_a_validation_error_on_the_version() -> None:
    with pytest.raises(ValidationError) as refused:
        _Kept.from_dict(_kept(schema_version=CORE_SCHEMA_VERSION + 1, label="echo", unknown_key=True))

    (error,) = refused.value.errors()
    assert error["loc"] == ("schema_version",)
    assert "newer build" in error["msg"]


def test_a_document_with_no_version_is_refused_rather_than_read_as_current() -> None:
    document = _kept()
    del document["schema_version"]

    with pytest.raises(ValidationError, match="not a core version number"):
        _Kept.from_dict(document)


def test_only_the_stored_read_upgrades() -> None:
    """A construction or a payload at an older core version is refused: only from_dict reads the past."""
    with pytest.raises(ValidationError, match=f"not v{CORE_SCHEMA_VERSION - 1}"):
        _Kept.model_validate(_kept(schema_version=CORE_SCHEMA_VERSION - 1))
