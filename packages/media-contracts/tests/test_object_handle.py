"""Tests for the ObjectHandle metadata wire-codec (produce -> catalog contract)."""

from __future__ import annotations

from uuid import UUID

import pytest

from threetears.media.contracts import OBJECT_HANDLE_METADATA_KEY, ObjectHandle

_OBJ = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")


def test_metadata_key_constant() -> None:
    """The metadata key is stable -- producer + catalog must agree on it."""
    assert OBJECT_HANDLE_METADATA_KEY == "object_handle"


def test_to_metadata_is_json_safe() -> None:
    """to_metadata stringifies the UUID at the border; all values JSON-safe."""
    handle = ObjectHandle(
        object_id=_OBJ,
        s3_key="cust/scope/scans/2026/06/30/obj/scan.xml",
        mime_type="application/xml",
        size_bytes=4096,
        summary="2 hosts",
        category="scans",
    )
    assert handle.to_metadata() == {
        "object_id": str(_OBJ),
        "s3_key": "cust/scope/scans/2026/06/30/obj/scan.xml",
        "mime_type": "application/xml",
        "size_bytes": 4096,
        "summary": "2 hosts",
        "category": "scans",
    }


def test_round_trip_preserves_fields() -> None:
    """from_metadata(to_metadata(h)) reconstructs an equal handle."""
    handle = ObjectHandle(
        object_id=_OBJ,
        s3_key="k",
        mime_type="application/pdf",
        size_bytes=7,
        summary=None,
        category="reports",
    )
    assert ObjectHandle.from_metadata(handle.to_metadata()) == handle


def test_from_metadata_missing_required_field_raises() -> None:
    """A descriptor missing a required field fails closed (KeyError)."""
    with pytest.raises(KeyError):
        ObjectHandle.from_metadata({"object_id": str(_OBJ), "s3_key": "k"})


def test_from_metadata_bad_uuid_raises() -> None:
    """A non-UUID object_id fails closed (ValueError)."""
    with pytest.raises(ValueError):
        ObjectHandle.from_metadata({"object_id": "not-a-uuid", "s3_key": "k", "mime_type": "m", "size_bytes": 1})


def test_optional_fields_default_to_none() -> None:
    """summary + category are optional and round-trip as None when absent."""
    restored = ObjectHandle.from_metadata({"object_id": str(_OBJ), "s3_key": "k", "mime_type": "m", "size_bytes": 2})
    assert restored.summary is None
    assert restored.category is None
