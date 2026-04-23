"""Shared LangGraph checkpoint serializer with uuid_utils sanitization.

Both :class:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver`
and :class:`~threetears.langgraph.proxy_checkpoint.ProxyCheckpointSaver`
serialize LangGraph checkpoint data to bytes for storage. The
underlying :class:`~langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`
delegates to ``ormsgpack`` which cannot encode
:class:`uuid_utils.UUID` instances (those are returned by ``asyncpg``
column readers on modern drivers). The wrapper walks the structure
and substitutes plain :class:`str` for every ``uuid_utils.UUID`` before
passing the object down to the inner serializer.

Previously each saver carried its own private copy of this helper; the
consolidation lets both implementations share the one truth.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

try:
    import uuid_utils as _uuid_utils
except ImportError:
    _uuid_utils = None  # type: ignore[assignment]


__all__ = ["UUIDSafeSerializer"]


class UUIDSafeSerializer:
    """wraps ``JsonPlusSerializer`` to sanitize ``uuid_utils.UUID`` values.

    :class:`asyncpg` returns ``uuid_utils.UUID`` instances (not the
    stdlib :class:`uuid.UUID`) which ``ormsgpack`` cannot serialize.
    this wrapper walks the input data structure and converts every
    encountered ``uuid_utils.UUID`` to a plain :class:`str` before
    passing the object to the inner serializer, preserving every
    other value unchanged.

    :ivar _inner: underlying :class:`JsonPlusSerializer` instance
    """

    def __init__(self) -> None:
        """initialize the wrapper with a fresh inner serializer."""
        self._inner = JsonPlusSerializer()

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        """recursively convert ``uuid_utils.UUID`` values to strings.

        :param obj: arbitrary object to sanitize
        :ptype obj: Any
        :return: same structure with UUID-v7 objects replaced by strings
        :rtype: Any
        """
        result: Any
        if _uuid_utils is not None and isinstance(obj, _uuid_utils.UUID):
            result = str(obj)
        elif isinstance(obj, dict):
            result = {k: UUIDSafeSerializer._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            result = [UUIDSafeSerializer._sanitize(x) for x in obj]
        elif isinstance(obj, tuple):
            result = tuple(UUIDSafeSerializer._sanitize(x) for x in obj)
        else:
            result = obj
        return result

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """serialize an object, sanitizing UUIDs first.

        :param obj: object to serialize
        :ptype obj: Any
        :return: tuple of (type tag, encoded bytes)
        :rtype: tuple[str, bytes]
        """
        return self._inner.dumps_typed(self._sanitize(obj))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """deserialize typed bytes back into a python object.

        :param data: tuple of (type tag, encoded bytes)
        :ptype data: tuple[str, bytes]
        :return: decoded object
        :rtype: Any
        """
        return self._inner.loads_typed(data)
