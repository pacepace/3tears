"""unit tests for :class:`threetears.epoch.wire.EpochBumpMessage`.

covers field-set freezing, JSON round-trip via Pydantic
``model_dump_json`` / ``model_validate_json`` (the wire path used
by :meth:`NatsClient.publish` + :meth:`NatsClient.subscribe_typed`),
and the immutability contract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from threetears.epoch.wire import EpochBumpMessage


class TestEpochBumpMessageShape:
    """field set, defaults, and frozen-instance contract."""

    def test_required_fields_only(self) -> None:
        """subject_path + epoch construct cleanly; payload defaults to None."""
        msg = EpochBumpMessage(subject_path="app.capabilities.epoch", epoch=7)
        assert msg.subject_path == "app.capabilities.epoch"
        assert msg.epoch == 7
        assert msg.payload is None

    def test_payload_accepts_arbitrary_json(self) -> None:
        """payload is dict[str, Any] | None and accepts mixed primitives."""
        payload = {"model_id": "abc", "tags": [1, 2], "meta": {"nested": True}}
        msg = EpochBumpMessage(
            subject_path="app.capabilities.epoch",
            epoch=42,
            payload=payload,
        )
        assert msg.payload == payload

    def test_subject_path_required(self) -> None:
        """missing subject_path raises ValidationError."""
        with pytest.raises(ValidationError):
            EpochBumpMessage(epoch=1)  # type: ignore[call-arg]

    def test_epoch_required(self) -> None:
        """missing epoch raises ValidationError."""
        with pytest.raises(ValidationError):
            EpochBumpMessage(subject_path="x.y.z")  # type: ignore[call-arg]

    def test_frozen_rejects_mutation(self) -> None:
        """frozen=True forbids attribute assignment after construction."""
        msg = EpochBumpMessage(subject_path="x.y.z", epoch=1)
        with pytest.raises(ValidationError):
            msg.epoch = 2  # type: ignore[misc]

    def test_unknown_fields_silently_dropped_for_forward_compat(self) -> None:
        """unknown fields are silently dropped per EPOCH-09 forward compatibility.

        ``extra="ignore"`` is locked in :class:`EpochBumpMessage`'s
        ``model_config``. flipping to ``extra="forbid"`` is a
        wire-protocol break: every publisher would have to roll
        forward together. this test enforces the lock so a
        well-meaning future maintainer cannot accidentally tighten
        validation and silently deadletter every older publisher's
        broadcasts.
        """
        msg = EpochBumpMessage.model_validate(
            {"subject_path": "x.y.z", "epoch": 1, "unknown_field": "ignored"},
        )
        assert msg.subject_path == "x.y.z"
        assert not hasattr(msg, "unknown_field")
        # explicit assertion on the config policy -- the regression
        # gate. flipping this to "forbid" must be a deliberate,
        # platform-wide decision recorded in a wire-protocol break note.
        assert EpochBumpMessage.model_config["extra"] == "ignore"


class TestEpochBumpMessageWireRoundTrip:
    """``model_dump_json`` -> ``model_validate_json`` is the wire shape."""

    def test_round_trip_with_payload(self) -> None:
        """typed envelope survives the publish/subscribe wire path byte-for-byte."""
        original = EpochBumpMessage(
            subject_path="3tears.gateway.catalog.epoch",
            epoch=99,
            payload={"action": "create", "model_id": "deepseek-v3.2"},
        )
        wire = original.model_dump_json()
        decoded = EpochBumpMessage.model_validate_json(wire)
        assert decoded == original

    def test_round_trip_without_payload(self) -> None:
        """payload=None round-trips without leaking into the JSON."""
        original = EpochBumpMessage(subject_path="x.y.z", epoch=1)
        decoded = EpochBumpMessage.model_validate_json(original.model_dump_json())
        assert decoded.payload is None

    def test_malformed_json_validation_error(self) -> None:
        """corrupt wire payload surfaces ValidationError (caught by typed dispatch)."""
        with pytest.raises(ValidationError):
            EpochBumpMessage.model_validate_json('{"subject_path": "x", "epoch": "not-an-int"}')
