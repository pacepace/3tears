"""a read cut by the bytes of its encoded rows: what :meth:`Driver.fetch_within` keeps, and the one rule that cuts it.

A caller that answers with a byte-capped reply (a responder whose JSON reply must fit a message, an HTTP reply
sized for a client) names its cap and its own encoder. The driver reads pages and hands each row to
:class:`RowBudget`, which encodes it ONCE with the caller's encoder, keeps it while the kept rows, as one JSON
array, fit the cap, and stops the read at the first row that does not. The caller gets the rows and their
encodings back, so its reply is the encodings spliced into an array (:meth:`BoundedRows.json_array`), never a
second encode of the same rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel

__all__ = [
    "DEFAULT_PAGE_ROWS",
    "BoundedRows",
    "RowBudget",
    "json_with_rows",
    "start_budget",
]

#: rows a driver reads a page at a time when the caller names none: small enough that the page past the cap
#: costs little, large enough that a big read is a few round trips
DEFAULT_PAGE_ROWS: Final = 500


def _check_max_bytes(max_bytes: int) -> None:
    """refuse a byte cap that cannot hold an empty array.

    :param max_bytes: the cap
    :ptype max_bytes: int
    :return: nothing
    :rtype: None
    :raises ValueError: if ``max_bytes`` is below 2 (``[]``)
    """
    if max_bytes < 2:
        raise ValueError(f"max_bytes must be at least 2, got {max_bytes}")


def _check_page_rows(page_rows: int) -> None:
    """refuse a page below one row.

    :param page_rows: the rows a page reads
    :ptype page_rows: int
    :return: nothing
    :rtype: None
    :raises ValueError: if ``page_rows`` is below 1
    """
    if page_rows < 1:
        raise ValueError(f"page_rows must be at least 1, got {page_rows}")


@dataclass(frozen=True, slots=True)
class BoundedRows:
    """the rows a byte-capped read kept, with the encoding its caller made of them.

    :ivar rows: the rows kept, in the statement's order
    :ivar encoded: the kept rows' JSON, in order, as the caller's encoder made it: each piece one or more rows
        (an array's inside, without its brackets), so :meth:`json_array` joins them by ``,``
    :ivar truncated: whether the result had rows past the ones kept
    :ivar size: the bytes of :meth:`json_array`
    """

    rows: list[dict[str, Any]]
    encoded: list[bytes]
    truncated: bool
    size: int

    def json_array(self) -> bytes:
        """the kept rows' encodings as one JSON array, ``size`` bytes long.

        :return: ``[`` the pieces joined by ``,`` ``]``
        :rtype: bytes
        """
        return b"[" + b",".join(self.encoded) + b"]"


class RowBudget:
    """the cut a byte-capped read makes: rows kept while their JSON array fits, encoded a page at a time.

    The caller's encoder turns rows into a JSON array, as its reply carries them. A page is encoded whole, once; a
    page that fits what is left of ``max_bytes`` is kept whole. Only the page that crosses the cap is encoded again
    a row at a time, to find the last row that fits, so a read encodes its rows about once: never a second pass
    over every row. The first row is kept whatever its size, so a page always moves its reader on; ``size`` then
    says it is over the cap. ``max_rows`` bounds the kept rows as :meth:`Driver.fetch_at_most` does: a row reached
    past it ends the read, truncated.

    :param max_bytes: the most bytes the kept rows' array may take
    :ptype max_bytes: int
    :param encode: the caller's encoder, rows to their JSON array
    :ptype encode: Callable[[list[dict[str, Any]]], bytes]
    :param max_rows: the most rows kept; None for no row bound
    :ptype max_rows: int | None
    """

    __slots__ = ("_encode", "_encoded", "_max_bytes", "_max_rows", "_rows", "_size", "_truncated")

    def __init__(self, max_bytes: int, encode: Callable[[list[dict[str, Any]]], bytes], max_rows: int | None) -> None:
        self._max_bytes = max_bytes
        self._encode = encode
        self._max_rows = max_rows
        self._rows: list[dict[str, Any]] = []
        self._encoded: list[bytes] = []
        self._size = 2
        self._truncated = False

    def wanted(self, page_rows: int) -> int:
        """how many rows the next page should read: ``page_rows``, or fewer where the row bound is near.

        One row past ``max_rows`` is read, so a result with exactly that many rows is told from a longer one.

        :param page_rows: the rows a page reads
        :ptype page_rows: int
        :return: the rows to ask for, at least 1
        :rtype: int
        """
        wanted = page_rows
        if self._max_rows is not None:
            wanted = max(1, min(page_rows, self._max_rows + 1 - len(self._rows)))
        return wanted

    def take(self, page: Iterable[dict[str, Any]]) -> bool:
        """keep the page's rows while they fit; say whether the read should go on.

        :param page: the next rows of the result, in order
        :ptype page: Iterable[dict[str, Any]]
        :return: True while every row so far was kept (a further page may be read), False once one was not
        :rtype: bool
        """
        rows = list(page)
        if self._truncated:
            rows = []
        elif self._max_rows is not None and len(self._rows) + len(rows) > self._max_rows:
            rows = rows[: self._max_rows - len(self._rows)]
            self._truncated = True
        if rows:
            inner = self._encode(rows)[1:-1]
            grown = self._size + len(inner) + (1 if self._rows else 0)
            if grown <= self._max_bytes or (not self._rows and len(rows) == 1):
                self._keep(rows, inner, grown)
            else:
                self._truncated = True
                self._keep_what_fits(rows)
        return not self._truncated

    def _keep(self, rows: list[dict[str, Any]], inner: bytes, grown: int) -> None:
        """keep rows whose JSON (an array's inside) brings the array to ``grown`` bytes.

        :param rows: the rows
        :ptype rows: list[dict[str, Any]]
        :param inner: their JSON, without the array's brackets
        :ptype inner: bytes
        :param grown: the array's bytes with them
        :ptype grown: int
        :return: nothing
        :rtype: None
        """
        self._rows.extend(rows)
        self._encoded.append(inner)
        self._size = grown

    def _keep_what_fits(self, rows: list[dict[str, Any]]) -> None:
        """the page that crosses the cap: its rows a row at a time, to the last that fits (the first row always).

        :param rows: the page's rows
        :ptype rows: list[dict[str, Any]]
        :return: nothing
        :rtype: None
        """
        for row in rows:
            inner = self._encode([row])[1:-1]
            grown = self._size + len(inner) + (1 if self._rows else 0)
            if self._rows and grown > self._max_bytes:
                break
            self._keep([row], inner, grown)

    def result(self) -> BoundedRows:
        """the rows kept so far.

        :return: the kept rows, their encodings, whether rows were left, and the array's bytes
        :rtype: BoundedRows
        """
        return BoundedRows(rows=self._rows, encoded=self._encoded, truncated=self._truncated, size=self._size)


def start_budget(
    *, max_bytes: int, encode: Callable[[list[dict[str, Any]]], bytes], max_rows: int | None, page_rows: int
) -> RowBudget:
    """check a byte-capped read's bounds and start its budget: the one entry every driver's ``fetch_within`` uses.

    :param max_bytes: the most bytes the kept rows' array may take
    :ptype max_bytes: int
    :param encode: the caller's encoder, rows to their JSON array
    :ptype encode: Callable[[list[dict[str, Any]]], bytes]
    :param max_rows: the most rows kept; None for no row bound
    :ptype max_rows: int | None
    :param page_rows: the rows a page reads
    :ptype page_rows: int
    :return: an empty budget
    :rtype: RowBudget
    :raises ValueError: if ``max_bytes`` is below 2, ``max_rows`` or ``page_rows`` below 1
    """
    _check_max_bytes(max_bytes)
    _check_page_rows(page_rows)
    if max_rows is not None and max_rows < 1:
        raise ValueError(f"max_rows must be at least 1, got {max_rows}")
    return RowBudget(max_bytes, encode, max_rows)


def json_with_rows(model: BaseModel, rows_json: bytes, *, field: str = "rows") -> bytes:
    """a reply model's JSON with ``rows_json`` as its ``field``, the rows' encodings spliced in, never re-encoded.

    For a reply whose rows were encoded once while they were read (:meth:`BoundedRows.json_array`): the rest of
    the model's JSON with that array as ``field``. A reader parses it exactly as it parses the model's own
    ``model_dump_json``; whatever ``field`` holds on the model is left out, so the rows may stay on it for the
    caller's own use.

    :param model: the reply
    :ptype model: BaseModel
    :param rows_json: the rows as one JSON array
    :ptype rows_json: bytes
    :param field: the reply's rows field
    :ptype field: str
    :return: the reply's JSON
    :rtype: bytes
    """
    rest = model.model_dump_json(exclude={field}).encode("utf-8")
    head = b'{"' + field.encode("utf-8") + b'":' + rows_json
    return head + (b"," + rest[1:] if len(rest) > 2 else b"}")
