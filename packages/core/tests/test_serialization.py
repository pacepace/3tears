"""Tests for threetears.core.serialization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4

import pytest

from threetears.core.serialization import (
    deserialize_from_json,
    serialize_to_json,
)


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class TestSerializeToJson:
    def test_basic_types(self):
        data = {"name": "test", "count": 42, "active": True}
        result = json.loads(serialize_to_json(data))
        assert result == data

    def test_uuid(self):
        uid = uuid4()
        data = {"id": uid}
        result = json.loads(serialize_to_json(data))
        assert result["id"] == str(uid)

    def test_datetime(self):
        dt = datetime(2026, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        data = {"created": dt}
        result = json.loads(serialize_to_json(data))
        assert result["created"] == dt.isoformat()

    def test_decimal(self):
        data = {"price": Decimal("19.99")}
        result = json.loads(serialize_to_json(data))
        assert result["price"] == "19.99"

    def test_bytes(self):
        data = {"payload": b"\xde\xad\xbe\xef"}
        result = json.loads(serialize_to_json(data))
        assert result["payload"] == "deadbeef"

    def test_enum(self):
        data = {"color": Color.RED}
        result = json.loads(serialize_to_json(data))
        assert result["color"] == "red"

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="not JSON serializable"):
            serialize_to_json({"bad": object()})

    def test_returns_bytes(self):
        result = serialize_to_json({"x": 1})
        assert isinstance(result, bytes)

    def test_none_values(self):
        data = {"name": "test", "value": None}
        result = json.loads(serialize_to_json(data))
        assert result["value"] is None


class TestDeserializeFromJson:
    def test_uuid_restoration(self):
        uid = uuid4()
        raw = serialize_to_json({"id": uid})
        result = deserialize_from_json(raw, {"id": UUID})
        assert result["id"] == uid
        assert isinstance(result["id"], UUID)

    def test_datetime_restoration(self):
        dt = datetime(2026, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        raw = serialize_to_json({"created": dt})
        result = deserialize_from_json(raw, {"created": datetime})
        assert result["created"] == dt

    def test_decimal_restoration(self):
        raw = serialize_to_json({"price": Decimal("19.99")})
        result = deserialize_from_json(raw, {"price": Decimal})
        assert result["price"] == Decimal("19.99")

    def test_bytes_restoration(self):
        raw = serialize_to_json({"payload": b"\xde\xad"})
        result = deserialize_from_json(raw, {"payload": bytes})
        assert result["payload"] == b"\xde\xad"

    def test_int_restoration(self):
        raw = serialize_to_json({"count": 42})
        result = deserialize_from_json(raw, {"count": int})
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_bool_restoration(self):
        raw = serialize_to_json({"active": True})
        result = deserialize_from_json(raw, {"active": bool})
        assert result["active"] is True

    def test_list_passthrough(self):
        raw = serialize_to_json({"tags": ["a", "b"]})
        result = deserialize_from_json(raw, {"tags": list[str]})
        assert result["tags"] == ["a", "b"]

    def test_none_preserved(self):
        raw = serialize_to_json({"value": None})
        result = deserialize_from_json(raw, {"value": UUID})
        assert result["value"] is None

    def test_optional_uuid(self):
        uid = uuid4()
        raw = serialize_to_json({"id": uid})
        result = deserialize_from_json(raw, {"id": UUID | None})
        assert result["id"] == uid

    def test_unknown_field_passthrough(self):
        raw = serialize_to_json({"extra": "hello"})
        result = deserialize_from_json(raw, {})
        assert result["extra"] == "hello"

    def test_round_trip_complex(self):
        uid = uuid4()
        dt = datetime(2026, 3, 12, tzinfo=timezone.utc)
        original = {
            "id": uid,
            "name": "test",
            "created": dt,
            "price": Decimal("9.99"),
            "active": True,
            "tags": ["a", "b"],
            "meta": None,
        }
        field_types = {
            "id": UUID,
            "name": str,
            "created": datetime,
            "price": Decimal,
            "active": bool,
            "tags": list[str],
            "meta": str | None,
        }
        raw = serialize_to_json(original)
        restored = deserialize_from_json(raw, field_types)
        assert restored["id"] == uid
        assert restored["created"] == dt
        assert restored["price"] == Decimal("9.99")
        assert restored["active"] is True
        assert restored["tags"] == ["a", "b"]
        assert restored["meta"] is None
