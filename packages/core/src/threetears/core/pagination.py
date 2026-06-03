"""Keyset (seek) pagination — stable cursor paging over a composite sort key.

Reusable across 3tears: page any large, append-heavy ordered list (conversation messages,
audit log, wake fires) WITHOUT ``LIMIT``/``OFFSET`` drift. ``OFFSET`` silently skips or
repeats rows when the underlying list changes between page fetches (a new row shifts every
later page by one); keyset anchors on the last row's sort key instead, so paging stays
stable while the tail grows.

The caller owns the SQL (table, filters, and the column allow-list -- the ``Keyset``
``columns`` are TRUSTED identifiers, never user input). This module provides the three
fiddly, easy-to-get-wrong parts:

- an opaque ``cursor`` (encode/decode of the composite sort-key tuple),
- the ``ORDER BY`` clause and the keyset ``WHERE`` predicate for that key + direction,
- page assembly: fetch ``page_size + 1``, trim the sentinel, emit the ``next_cursor``.

Usage::

    KS = Keyset(columns=("date_created", "message_id"), casts=("timestamptz", "uuid"))
    pred, pred_params = KS.predicate(cursor, first_param=len(params) + 1)
    where = base_where + (f" AND {pred}" if pred else "")
    rows = await pool.fetch(
        f"SELECT ... FROM messages WHERE {where} ORDER BY {KS.order_by()} LIMIT ${n}",
        *params, *pred_params, page_size + 1,
    )
    page = KS.page(rows, page_size, key_of=lambda r: (r["date_created"], r["message_id"]))
    # page.items -> this page; page.next_cursor -> opaque token for the next page (or None)
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

__all__ = ["CursorError", "Keyset", "Page", "decode_cursor", "encode_cursor"]

T = TypeVar("T")


class CursorError(ValueError):
    """A cursor token could not be decoded (malformed, tampered, or wrong arity)."""


def encode_cursor(key: Sequence[Any]) -> str:
    """Encode a composite sort-key tuple into an opaque, URL-safe cursor token.

    Values are JSON-encoded with ``default=str`` so datetimes / UUIDs become their string
    form; the matching column ``casts`` on :class:`Keyset` turn them back into typed
    placeholders at query time. The token is opaque (clients must not parse it), not secret.
    """
    raw = json.dumps(list(key), separators=(",", ":"), default=str).encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(token: str) -> list[Any]:
    """Decode an opaque cursor token back into its sort-key value list.

    :raises CursorError: if the token is malformed or not a JSON array.
    """
    try:
        raw = base64.urlsafe_b64decode(token.encode())
        value = json.loads(raw.decode())
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise CursorError(f"invalid cursor token: {token!r}") from exc
    if not isinstance(value, list):
        raise CursorError(f"cursor payload is not an array: {value!r}")
    return value


@dataclass(frozen=True)
class Page(Generic[T]):
    """One page of results plus the cursor to fetch the next page.

    :param items: the rows for this page (at most ``page_size``)
    :param next_cursor: opaque token for the next page, or ``None`` when exhausted
    """

    items: list[T]
    next_cursor: str | None


@dataclass(frozen=True)
class Keyset:
    """A composite sort key + direction for keyset pagination.

    ``columns`` are TRUSTED SQL identifiers (a caller-controlled allow-list -- NEVER user
    input; they are interpolated into SQL). Include a unique tiebreaker last (e.g. the
    primary key, or a UUIDv7 which is itself time-ordered) so the key is total and a page
    boundary is unambiguous across ties. All columns share one ``descending`` direction
    (mixed directions need the expanded OR-form and are intentionally out of scope).

    ``casts`` optionally gives a Postgres type per column for the keyset placeholders (e.g.
    ``("timestamptz", "uuid")``); since cursor values arrive as strings, the cast restores
    the column type for the comparison (``$1::timestamptz``). Empty = no casts.

    :param columns: composite sort-key columns, tiebreaker last
    :param casts: optional Postgres type per column for placeholder casts
    :param descending: ``True`` for ``DESC`` (newest-first / scroll-back), the common case
    """

    columns: tuple[str, ...]
    casts: tuple[str, ...] = field(default=())
    descending: bool = True

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("Keyset requires at least one column")
        if self.casts and len(self.casts) != len(self.columns):
            raise ValueError(f"casts arity {len(self.casts)} != columns arity {len(self.columns)}")

    def order_by(self) -> str:
        """The ``ORDER BY`` clause body (no ``ORDER BY`` keyword), e.g. ``a DESC, b DESC``."""
        direction = "DESC" if self.descending else "ASC"
        return ", ".join(f"{c} {direction}" for c in self.columns)

    def _placeholder(self, index: int, position: int) -> str:
        ph = f"${index}"
        if self.casts:
            ph = f"{ph}::{self.casts[position]}"
        return ph

    def predicate(self, cursor: str | None, first_param: int) -> tuple[str, list[Any]]:
        """Keyset ``WHERE`` fragment selecting rows strictly AFTER ``cursor``.

        Emits a row-value comparison ``(c1, c2) < ($n::t1, $n+1::t2)`` (``>`` when
        ascending) -- exactly the seek predicate for a composite key. Returns ``("", [])``
        for the first page (``cursor is None``).

        :param cursor: opaque token from a prior :class:`Page`, or ``None`` for page one
        :param first_param: the next free ``$N`` index in the caller's param list
        :returns: ``(sql_fragment, params)`` to splice into the ``WHERE`` and bind in order
        :raises CursorError: if the cursor's arity does not match ``columns``
        """
        if cursor is None:
            return "", []
        values = decode_cursor(cursor)
        if len(values) != len(self.columns):
            raise CursorError(f"cursor arity {len(values)} != keyset arity {len(self.columns)}")
        op = "<" if self.descending else ">"
        cols = ", ".join(self.columns)
        placeholders = ", ".join(self._placeholder(first_param + i, i) for i in range(len(self.columns)))
        return f"({cols}) {op} ({placeholders})", values

    def page(
        self,
        rows: Sequence[T],
        page_size: int,
        key_of: Callable[[T], Sequence[Any]],
    ) -> Page[T]:
        """Assemble a :class:`Page` from rows fetched with ``LIMIT page_size + 1``.

        The extra ``+ 1`` row is a sentinel: if it came back there is a next page, so we drop
        it and set ``next_cursor`` from the last KEPT row's sort key. ``key_of`` extracts that
        sort-key tuple from a row (the same columns, in the same order, as ``columns``).

        :param rows: rows from a query that used ``LIMIT page_size + 1`` and this ``order_by``
        :param page_size: the requested page size (the ``+ 1`` is the sentinel)
        :param key_of: extracts the sort-key tuple from a row, for the next cursor
        :returns: the trimmed page + its ``next_cursor`` (``None`` when no sentinel row)
        """
        items = list(rows[:page_size])
        has_more = len(rows) > page_size
        next_cursor = encode_cursor(key_of(items[-1])) if has_more and items else None
        return Page(items=items, next_cursor=next_cursor)
