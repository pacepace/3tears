"""Every contract type JSON round-trips (SR-L4, search-spec.md §6).

No callables, open files, or port objects in any payload; the reconstructed
instance equals the original, field for field, including Decimal money and
timezone-aware timestamps. The typed error taxonomy round-trips through its
:class:`FailureRecord` projection (D10).
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

import threetears.search.contracts as contracts
from threetears.search.contracts import (
    FAILURE_CLASSES,
    AuthFailed,
    FailureRecord,
    LocalCapExceeded,
    RateLimited,
    SearchFailure,
    SearchResultsMetadata,
    Spend,
)
from _search_instances import ALL_INSTANCES, METADATA, SPEND


@pytest.mark.parametrize("instance", ALL_INSTANCES, ids=lambda i: type(i).__name__)
def test_json_roundtrip_is_lossless(instance: BaseModel) -> None:
    """model -> JSON text -> model reproduces the instance exactly."""
    payload = instance.model_dump_json()
    rebuilt = type(instance).model_validate_json(payload)
    assert rebuilt == instance


@pytest.mark.parametrize("instance", ALL_INSTANCES, ids=lambda i: type(i).__name__)
def test_payload_is_plain_json(instance: BaseModel) -> None:
    """the serialized form parses as plain JSON -- nothing non-wire rides it."""
    parsed = json.loads(instance.model_dump_json())
    assert isinstance(parsed, dict)


def test_every_exported_contract_model_is_covered() -> None:
    """the round-trip suite covers every exported ContractModel, by construction."""
    exported = {
        obj
        for name in contracts.__all__
        if isinstance(obj := getattr(contracts, name), type) and issubclass(obj, BaseModel)
    }
    covered = {type(instance) for instance in ALL_INSTANCES}
    assert exported == covered, f"round-trip coverage drifted: missing={exported - covered}"


def test_failure_taxonomy_has_seven_distinct_classes() -> None:
    """SR-J1: seven distinguishable classes, distinct wire names."""
    assert len(FAILURE_CLASSES) == 7
    assert len({cls.failure_class for cls in FAILURE_CLASSES.values()}) == 7


@pytest.mark.parametrize("failure_class", sorted(FAILURE_CLASSES), ids=str)
def test_every_failure_roundtrips_through_its_record(failure_class: str) -> None:
    """typed error -> FailureRecord -> JSON -> FailureRecord -> typed error."""
    failure_type = FAILURE_CLASSES[failure_class]
    kwargs: dict[str, object] = {}
    if failure_type is RateLimited:
        kwargs["retry_after_seconds"] = 12.5
    if failure_type is LocalCapExceeded:
        kwargs["scope"] = "persona:capy"
    original = failure_type(
        "it broke",
        spend=SPEND,
        provider_instance="searxng-main",
        remediation="teaching text",
        **kwargs,  # type: ignore[arg-type]
    )

    record = FailureRecord.model_validate_json(original.to_record().model_dump_json())
    rebuilt = record.to_failure()

    assert type(rebuilt) is failure_type
    assert rebuilt.message == original.message
    assert rebuilt.spend == SPEND
    assert rebuilt.provider_instance == "searxng-main"
    assert rebuilt.remediation == "teaching text"
    if isinstance(original, RateLimited):
        assert isinstance(rebuilt, RateLimited)
        assert rebuilt.retry_after_seconds == original.retry_after_seconds
    if isinstance(original, LocalCapExceeded):
        assert isinstance(rebuilt, LocalCapExceeded)
        assert rebuilt.scope == original.scope


def test_every_failure_carries_spend() -> None:
    """SR-E3: the failure path cannot drop what the call consumed."""
    failure: SearchFailure = AuthFailed("denied", spend=SPEND)
    assert failure.spend == SPEND
    assert failure.to_record().spend == SPEND


def test_unknown_failure_class_is_refused_loudly() -> None:
    """an unknown wire class raises naming what this reader knows, never misreads."""
    record = FailureRecord(failure_class="not-a-thing", message="?", spend=Spend())
    with pytest.raises(ValueError, match="not-a-thing"):
        record.to_failure()


def test_metadata_projection_roundtrips_through_dict() -> None:
    """to_metadata / from_metadata is lossless, per the ObjectHandle precedent (D22)."""
    projected = METADATA.to_metadata()
    assert projected["schema_version"] == 1
    rebuilt = SearchResultsMetadata.from_metadata(json.loads(json.dumps(projected)))
    assert rebuilt == METADATA


def test_metadata_refuses_newer_schema_version() -> None:
    """D13: a reader meeting a payload newer than it understands refuses, naming both."""
    payload = METADATA.to_metadata()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="99"):
        SearchResultsMetadata.from_metadata(payload)
