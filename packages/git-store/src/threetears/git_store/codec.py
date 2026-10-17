"""how a row becomes a file's bytes and back: the codec half of the plug-in boundary.

A product brings a codec and a layout (:mod:`threetears.git_store.layout`); everything else is shared.
:class:`JsonCodec` ships here. A markdown-with-frontmatter codec (scriob's) is the product's own.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from threetears.core.serialization import json_datetime

__all__ = ["JsonCodec", "RowCodec"]


@runtime_checkable
class RowCodec(Protocol):
    """turns a row into a file's bytes, and a file's bytes back into the row.

    :ivar suffix: the file name suffix files of this codec carry (``".json"``, ``".md"``)
    """

    suffix: str

    def encode(self, row: Mapping[str, Any]) -> bytes:
        """the file's bytes for ``row``.

        :param row: the row, column to value
        :ptype row: Mapping[str, Any]
        :return: the bytes
        :rtype: bytes
        """
        ...

    def decode(self, data: bytes) -> dict[str, Any]:
        """the row a file's bytes hold.

        :param data: the bytes
        :ptype data: bytes
        :return: the row
        :rtype: dict[str, Any]
        """
        ...


def _json_default(value: object) -> Any:
    """the JSON form of a value :mod:`json` cannot write itself.

    :param value: the value
    :ptype value: object
    :return: its JSON form
    :rtype: Any
    :raises TypeError: for a value with no agreed form
    :raises ValueError: for a naive datetime, which names no instant
    """
    if isinstance(value, datetime):
        return json_datetime(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID | Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} has no JSON form in a git-store row")


class JsonCodec:
    """rows as pretty JSON files, keys in the row's own order.

    A file this codec wrote decodes and encodes back to exactly the same bytes, so a row read and
    saved unchanged makes no commit. Datetimes are written as UTC ISO-8601 with microseconds (the
    form :func:`threetears.core.serialization.json_datetime` gives every stored datetime in 3tears)
    and the columns named in ``datetime_columns`` are read back as aware datetimes, so a
    compare-and-swap on ``date_updated`` compares instants, not strings.

    :ivar suffix: ``".json"``
    """

    suffix = ".json"

    def __init__(
        self,
        *,
        indent: int | None = 2,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        trailing_newline: bool = True,
        datetime_columns: Collection[str] = ("date_created", "date_updated"),
    ) -> None:
        """choose the file's form.

        :param indent: spaces per level; ``None`` for one line
        :ptype indent: int | None
        :param sort_keys: write keys sorted rather than in the row's order
        :ptype sort_keys: bool
        :param ensure_ascii: escape every non-ASCII character
        :ptype ensure_ascii: bool
        :param trailing_newline: end the file with a newline
        :ptype trailing_newline: bool
        :param datetime_columns: top-level columns decoded as datetimes
        :ptype datetime_columns: Collection[str]
        :return: None
        :rtype: None
        """
        self._indent = indent
        self._sort_keys = sort_keys
        self._ensure_ascii = ensure_ascii
        self._trailing_newline = trailing_newline
        self._datetime_columns = frozenset(datetime_columns)

    def encode(self, row: Mapping[str, Any]) -> bytes:
        """the file's bytes for ``row``.

        :param row: the row
        :ptype row: Mapping[str, Any]
        :return: UTF-8 JSON
        :rtype: bytes
        :raises TypeError: for a value with no JSON form
        :raises ValueError: for a naive datetime
        """
        text = json.dumps(
            dict(row),
            indent=self._indent,
            sort_keys=self._sort_keys,
            ensure_ascii=self._ensure_ascii,
            default=_json_default,
        )
        if self._trailing_newline:
            text += "\n"
        return text.encode("utf-8")

    def decode(self, data: bytes) -> dict[str, Any]:
        """the row a file holds.

        :param data: UTF-8 JSON of an object
        :ptype data: bytes
        :return: the row, its datetime columns as aware datetimes
        :rtype: dict[str, Any]
        :raises ValueError: when the file is not a JSON object
        """
        loaded = json.loads(data.decode("utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"a git-store JSON row must be an object, found {type(loaded).__name__}")
        for column in self._datetime_columns:
            value = loaded.get(column)
            if isinstance(value, str):
                loaded[column] = datetime.fromisoformat(value)
        return loaded
