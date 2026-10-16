"""The tolerant wire reader: a field a newer sender adds is ignored, a retired identity field is refused.

A wire model read by a separately deployed receiver must not refuse the whole message because the
sender is newer. But ignoring a field is not always safe: a field retired BECAUSE it let the caller
name its own authority must still be refused, or a stale client keeps asserting it and is silently
granted something broader than it asked for.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ConfigDict, ValidationError

from threetears.core.wire import TolerantWireModel


class _Door(TolerantWireModel):
    """a request model as a door reads it."""

    refused_fields: ClassVar[frozenset[str]] = frozenset({"user_id"})

    sql: str


class _FrozenDoor(TolerantWireModel):
    """a tolerant model that also freezes, to prove config merges rather than replaces."""

    model_config = ConfigDict(frozen=True)

    sql: str


def test_a_newer_senders_field_is_ignored() -> None:
    """an older receiver accepts a message carrying a field it does not know."""
    model = _Door.model_validate({"sql": "select 1", "identity_proof": "abc"})
    assert model.sql == "select 1"
    assert "identity_proof" not in model.model_dump()


def test_a_retired_field_is_refused() -> None:
    """a field retired for authority reasons is refused, not ignored."""
    with pytest.raises(ValidationError, match="may not be sent"):
        _Door.model_validate({"sql": "select 1", "user_id": "someone-else"})


def test_a_retired_field_is_refused_from_json_too() -> None:
    """the json border refuses it the same way."""
    with pytest.raises(ValidationError, match="may not be sent"):
        _Door.model_validate_json('{"sql": "select 1", "user_id": "someone-else"}')


def test_a_subclass_config_merges_with_tolerance() -> None:
    """a subclass that sets other config keys stays tolerant."""
    model = _FrozenDoor.model_validate({"sql": "x", "new": 1})
    assert model.sql == "x"
    with pytest.raises(ValidationError):
        model.sql = "y"  # type: ignore[misc]
