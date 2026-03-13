"""JSON serialization helpers for cache storage (L2 NATS KV).

Provides a custom JSON encoder and type-aware deserializer that handles
UUID, datetime, Decimal, bytes, and Enum round-trips through JSON.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, get_args, get_origin
from uuid import UUID


def _json_serializer(obj: object) -> str | int | float | bool | None:
    """Serialize non-JSON-native types for ``json.dumps``.

    Called as the ``default`` parameter of ``json.dumps``.
    """
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def serialize_to_json(data: dict[str, Any]) -> bytes:
    """Serialize entity data dictionary to JSON bytes for cache storage."""
    return json.dumps(data, default=_json_serializer).encode("utf-8")


def _resolve_base_type(type_hint: Any) -> type | None:
    """Extract the concrete type from a possibly-Optional type hint.

    For ``UUID | None`` returns ``UUID``. For ``list[float]`` returns ``list``.
    """
    origin = get_origin(type_hint)
    if origin is not None:
        import types

        if origin is types.UnionType:
            args = get_args(type_hint)
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                inner = non_none[0]
                inner_origin = get_origin(inner)
                return inner_origin if inner_origin is not None else inner
            return None
        return origin
    return type_hint


def deserialize_from_json(
    data: bytes, field_types: dict[str, Any]
) -> dict[str, Any]:
    """Deserialize JSON bytes from cache back to entity data dictionary.

    Converts string representations back to their native Python types
    based on the entity's field type annotations.
    """
    raw: dict[str, Any] = json.loads(data.decode("utf-8"))
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            result[key] = None
            continue
        base_type = _resolve_base_type(field_types.get(key))
        if base_type is UUID and isinstance(value, str):
            result[key] = UUID(value)
        elif base_type is datetime and isinstance(value, str):
            result[key] = datetime.fromisoformat(value)
        elif base_type is Decimal and isinstance(value, str):
            result[key] = Decimal(value)
        elif base_type is bytes and isinstance(value, str):
            result[key] = bytes.fromhex(value)
        elif base_type is int and isinstance(value, (int, float)):
            result[key] = int(value)
        elif base_type is bool and isinstance(value, (bool, int)):
            result[key] = bool(value)
        elif base_type is list and isinstance(value, list):
            result[key] = value
        else:
            result[key] = value
    return result
