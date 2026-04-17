"""JSON serialization helpers and pluggable format-handler registry.

Provides a custom JSON encoder and type-aware deserializer that handles
UUID, datetime, Decimal, bytes, and Enum round-trips through JSON, plus
a runtime-checkable :class:`FormatHandler` Protocol and extension-keyed
registry that external packages use to plug in YAML, TOML, .env, or any
other structural document format.

No concrete handlers are registered here — each format lives in its own
package and self-registers on module import.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, get_args, get_origin, runtime_checkable
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
        return obj.value  # type: ignore[no-any-return]
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
                return inner_origin if inner_origin is not None else inner  # type: ignore[no-any-return]
            return None
        return origin  # type: ignore[no-any-return]
    return type_hint  # type: ignore[no-any-return]


def deserialize_from_json(data: bytes, field_types: dict[str, Any]) -> dict[str, Any]:
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


class UnknownFormatError(LookupError):
    """raised when no :class:`FormatHandler` is registered for given extension.

    subclasses :class:`LookupError` so callers may catch broadly or narrowly.
    """


@runtime_checkable
class FormatHandler(Protocol):
    """structural contract for pluggable serialization format handlers.

    implementations own parsing, serialization, and path-based access for
    one or more file extensions. path expressions are interpreted by each
    handler — no jsonpath grammar is imposed at protocol level.

    :cvar extensions: tuple of file extensions handler owns, leading-dot
        form (e.g. ``(".yaml", ".yml")``); registry normalizes to lowercase
        without leading dot when indexing
    """

    extensions: tuple[str, ...]

    def load(self, text: str) -> Any:
        """parse serialized document body into in-memory tree.

        :param text: serialized document body
        :ptype text: str
        :return: in-memory document tree
        :rtype: Any
        :raises ValueError: if text cannot be parsed as this format
        """
        ...

    def dump(self, tree: Any) -> str:
        """serialize in-memory tree back to document body text.

        :param tree: in-memory document tree
        :ptype tree: Any
        :return: serialized document body
        :rtype: str
        :raises TypeError: if tree contains types handler cannot serialize
        """
        ...

    def get(self, tree: Any, path: str) -> Any:
        """resolve handler-interpreted path expression against tree.

        :param tree: in-memory document tree
        :ptype tree: Any
        :param path: handler-interpreted path expression
        :ptype path: str
        :return: value at path within tree
        :rtype: Any
        :raises KeyError: if path does not resolve within tree
        """
        ...

    def set(self, tree: Any, path: str, value: Any) -> Any:
        """assign value at handler-interpreted path within tree.

        handler may mutate tree in place and return same tree, or return
        a new structure — callers must use the returned tree.

        :param tree: in-memory document tree
        :ptype tree: Any
        :param path: handler-interpreted path expression
        :ptype path: str
        :param value: value to assign at path
        :ptype value: Any
        :return: possibly new document tree with value set at path
        :rtype: Any
        :raises KeyError: if path cannot be constructed within tree
        """
        ...

    def merge(self, tree: Any, partial: dict[str, Any]) -> Any:
        """merge partial document into tree according to handler's rules.

        handler may mutate tree in place and return same tree, or return
        a new structure — callers must use the returned tree.

        :param tree: in-memory document tree
        :ptype tree: Any
        :param partial: partial document to merge into tree
        :ptype partial: dict[str, Any]
        :return: possibly new document tree with partial merged
        :rtype: Any
        """
        ...


_HANDLERS: dict[str, FormatHandler] = {}


def register_handler(handler: FormatHandler) -> None:
    """install handler in module-level registry under each of its extensions.

    extension keys are normalized to lowercase with leading dot stripped.
    registering the same extension twice replaces the prior handler.

    :param handler: concrete handler implementing :class:`FormatHandler`
    :ptype handler: FormatHandler
    :return: None
    :rtype: None
    """
    for ext in handler.extensions:
        _HANDLERS[ext.lstrip(".").lower()] = handler


def handler_for(path: str | Path) -> FormatHandler:
    """resolve path's extension to registered handler.

    extension matching is case-insensitive and strips leading dot.

    :param path: filesystem path or string whose extension selects handler
    :ptype path: str | Path
    :return: registered handler for path's extension
    :rtype: FormatHandler
    :raises UnknownFormatError: if no handler is registered for extension
    """
    ext = Path(path).suffix.lstrip(".").lower()
    try:
        result = _HANDLERS[ext]
    except KeyError as e:
        raise UnknownFormatError(
            f"no FormatHandler registered for extension {ext!r}"
        ) from e
    return result
