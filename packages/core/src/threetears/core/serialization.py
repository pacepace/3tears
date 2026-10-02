"""JSON serialization helpers and pluggable format-handler registry.

Provides a custom JSON encoder and type-aware deserializer that handles
UUID, datetime, Decimal, bytes, and Enum round-trips through JSON, the one
stored form of a datetime inside JSON (:func:`json_datetime`) that every
storage encoder writes, :func:`to_stored_json` for a model or structure about
to be stored, plus a runtime-checkable :class:`FormatHandler` Protocol and extension-keyed
registry that external packages use to plug in YAML, TOML, .env, or any
other structural document format.

No concrete handlers are registered here — each format lives in its own
package and self-registers on module import.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, get_args, get_origin, runtime_checkable
from uuid import UUID

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

__all__ = [
    "FormatHandler",
    "UnknownFormatError",
    "deserialize_from_json",
    "handler_for",
    "json_datetime",
    "register_handler",
    "serialize_to_json",
    "to_stored_json",
]


def json_datetime(value: datetime, *, field: str | None = None) -> str:
    """the one stored form of a datetime inside JSON, at every tier.

    ISO 8601 extended format with the ``T`` separator, always six fraction digits, and an
    explicit UTC offset: ``2026-10-01T12:30:00.000000+00:00``. Fixed width, so stored strings
    compare and sort as the instants they name. An aware value in another zone is converted to
    UTC first, so one instant has one spelling.

    Every storage encoder's handler -- L3's jsonb codec and L2's payload encoders through
    :func:`threetears.core.backends.schema_sql.json_default`, the entity codec through
    :func:`serialize_to_json`, L1's caches, and a model stored through :func:`to_stored_json` --
    writes this form, so a nested datetime reads back as the same string whichever tier answered.
    Readers parse it with :meth:`datetime.fromisoformat`, which also accepts every form stored
    before this existed (``str(dt)``'s space separator, ``isoformat()`` without a fraction,
    pydantic's ``Z`` suffix).

    **A naive value is refused.** It names no instant, so storing it would make every reader guess
    the zone it was measured in. A collection's declared timestamp column is normalised before it
    reaches here (:meth:`~threetears.core.collections.base.BaseCollection.serialize` callers stamp
    a naive value UTC, as its L3 write does); anything else that arrives naive is a producer bug,
    and the error names the field when the caller knows it.

    :param value: the datetime to store
    :ptype value: datetime
    :param field: where the value sits (a column, or a dotted path into a document), named in the
        refusal; ``None`` when the caller cannot know (a ``json.dumps`` default handler)
    :ptype field: str | None
    :return: the canonical text
    :rtype: str
    :raises ValueError: if ``value`` is naive
    """
    if value.tzinfo is None or value.utcoffset() is None:
        where = f" in {field!r}" if field is not None else ""
        raise ValueError(
            f"refusing to store a naive datetime{where} ({value.isoformat()}): it names no instant. "
            f"produce it timezone-aware -- datetime.now(UTC), or attach the zone it was measured in"
        )
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def to_stored_json(value: Any, *, field: str | None = None) -> Any:
    """the JSON-ready form of a value that is about to be STORED: JSONB, a KV value, a cached payload.

    Exactly what :meth:`pydantic.BaseModel.model_dump` in ``mode="json"`` writes -- enums as their
    values, UUIDs and Decimals as strings, sets as lists, a model's own serializers honoured --
    except that every datetime is written in :func:`json_datetime`'s one form. Pydantic's JSON mode
    writes ``2026-10-01T12:30:00Z`` and drops a zero fraction, a second spelling of an instant every
    other storage tier writes the one way; route a stored model through here, never through
    ``model_dump(mode="json")`` / ``model_dump_json()``. An API response is not storage and keeps
    pydantic's form.

    Accepts a model, or any structure of mappings, sequences and models. A naive datetime anywhere
    in it is refused, naming its path (``runs[0].date_started``).

    :param value: the model or structure to store
    :ptype value: Any
    :param field: the path the value sits at, prefixed to the path a refusal names
    :ptype field: str | None
    :return: a structure of JSON-native values
    :rtype: Any
    :raises ValueError: if any datetime in ``value`` is naive
    """
    return _stored_json(value, to_jsonable_python(value), field)


def _stored_json(native: Any, encoded: Any, path: str | None) -> Any:
    """walk the python-side value and its pydantic JSON encoding together, re-spelling datetimes.

    The two walk in step because pydantic's JSON encoding keeps the python shape (a mapping stays a
    mapping, a sequence a sequence, keys in the same order). Where a model's own serializer changed
    the shape, the encoded value is kept as pydantic wrote it.

    :param native: the value as the caller holds it (a model is dumped in python mode first)
    :ptype native: Any
    :param encoded: the same value as pydantic's JSON mode encodes it
    :ptype encoded: Any
    :param path: where ``native`` sits, for a refusal's message
    :ptype path: str | None
    :return: ``encoded`` with every datetime re-spelled in :func:`json_datetime`'s form
    :rtype: Any
    :raises ValueError: if a datetime is naive
    """
    if isinstance(native, BaseModel):
        native = native.model_dump(mode="python")
    result: Any = encoded
    if isinstance(native, datetime) and isinstance(encoded, str):
        result = json_datetime(native, field=path)
    elif isinstance(native, Mapping) and isinstance(encoded, dict) and len(native) == len(encoded):
        result = {
            encoded_key: _stored_json(item, encoded_item, _child_path(path, encoded_key))
            for (_, item), (encoded_key, encoded_item) in zip(native.items(), encoded.items(), strict=True)
        }
    elif (
        isinstance(native, (list, tuple, set, frozenset)) and isinstance(encoded, list) and len(native) == len(encoded)
    ):
        result = [
            _stored_json(item, encoded_item, f"{path or ''}[{index}]")
            for index, (item, encoded_item) in enumerate(zip(native, encoded, strict=True))
        ]
    return result


def _child_path(path: str | None, key: Any) -> str:
    """the path of a mapping member, for a refusal's message.

    :param path: the mapping's own path, or ``None`` at the top
    :ptype path: str | None
    :param key: the member's key as encoded
    :ptype key: Any
    :return: ``path.key``, or ``key`` at the top
    :rtype: str
    """
    return f"{path}.{key}" if path is not None else str(key)


def _json_serializer(obj: object) -> str | int | float | bool | None:
    """Serialize non-JSON-native types for ``json.dumps``.

    Called as the ``default`` parameter of ``json.dumps``.
    """
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return json_datetime(obj)
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, Enum):
        return obj.value  # type: ignore[no-any-return]
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def serialize_to_json(data: dict[str, Any]) -> bytes:
    """serialize entity data dictionary to JSON bytes for cache storage.

    a datetime is written in :func:`json_datetime`'s form, as every storage tier writes it.

    :param data: row dict keyed by column name
    :ptype data: dict[str, Any]
    :return: UTF-8 JSON bytes
    :rtype: bytes
    """
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
        raise UnknownFormatError(f"no FormatHandler registered for extension {ext!r}") from e
    return result
