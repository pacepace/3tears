"""a read of changed rows' vocabulary: the columns a row's hash covers and how each is written, a group's hash, and
what a read answers. The hashes are spelled and the reads run in ``threetears.core.keyed.changed``."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

__all__ = ["ROW_HASH_COLUMN", "ChangedRows", "GroupHash", "HashColumn", "HashKind"]

#: the name a row's hash is read under, in both phases and in a full read that stores it
ROW_HASH_COLUMN: Final = "row_hash"

#: the most digits a hashed NUMERIC's scale may name (DECIMAL(38, scale) on every dialect)
_MAX_SCALE: Final = 37


class HashKind(StrEnum):
    """how a hashed column's value is written as canonical text, the same on every dialect."""

    #: the text itself
    TEXT = "text"
    #: an integer's decimal digits
    INTEGER = "integer"
    #: a decimal at a fixed scale (``HashColumn.scale``): ``1.50`` at scale 2, whatever the column's own. Over a
    #: double precision column the text is per engine: Redshift and PostgreSQL cast a double to DECIMAL(38, scale)
    #: differently past 15 significant digits, so such a hash is compared only with one the same engine computed
    NUMERIC = "numeric"
    #: ``t`` or ``f``
    BOOLEAN = "boolean"
    #: ``YYYY-MM-DD`` (PostgreSQL writes an infinite date as no text: it hashes as NULL there)
    DATE = "date"
    #: a timestamp without time zone, ``YYYY-MM-DD HH:MM:SS.ffffff`` (six fractional digits always)
    TIMESTAMP = "timestamp"


@dataclass(frozen=True)
class HashColumn:
    """one column a row's hash covers.

    A floating-point column is hashed as :attr:`HashKind.NUMERIC` at the scale its values carry; a float's own text
    differs between databases. Its hash is still per engine: Redshift and PostgreSQL cast a double to
    DECIMAL(38, scale) differently past 15 significant digits (live parity 2026-11-10: every text, integer, timestamp
    and boolean column equal on both, 0 of 130 double column-samples), so a double's hash is compared only with a hash
    the same engine computed, never across engines.

    :param column: the column, a TRUSTED identifier
    :ptype column: str
    :param kind: how its value is written before hashing
    :ptype kind: HashKind
    :param scale: a NUMERIC's fractional digits (0 to 37); None for every other kind
    :ptype scale: int | None
    """

    column: str
    kind: HashKind = HashKind.TEXT
    scale: int | None = None

    def __post_init__(self) -> None:
        if self.kind is HashKind.NUMERIC:
            if self.scale is None or not 0 <= self.scale <= _MAX_SCALE:
                raise ValueError(f"a hashed numeric {self.column!r} names its scale, 0 to {_MAX_SCALE}")
        elif self.scale is not None:
            raise ValueError(f"only a hashed numeric takes a scale, {self.column!r} is {self.kind}")


@dataclass(frozen=True)
class GroupHash:
    """an order-free hash of a group's rows: how many, and four 32-bit lanes each summing one slice of every row's
    hash (:func:`row_hash_sql`) modulo 2**32. The same rows give the same group hash in any order; a changed, added or
    removed row moves it.

    :param rows: the rows in the group
    :ptype rows: int
    :param lanes: the four lane sums, in :data:`_LANES` order
    :ptype lanes: tuple[int, int, int, int]
    """

    rows: int
    lanes: tuple[int, int, int, int]


@dataclass(frozen=True)
class ChangedRows:
    """what a read of a scope's changed rows answered.

    :param rows: the full rows whose hash differs from the held one or that were not held, each with its hash
        under :data:`ROW_HASH_COLUMN` (store it beside the row)
    :ptype rows: list[dict[str, Any]]
    :param gone_keys: the held keys the scope no longer holds
    :ptype gone_keys: list[tuple[Any, ...]]
    :param changed_keys: the keys whose hash moved or that were not held, as the scope answers them
    :ptype changed_keys: list[tuple[Any, ...]]
    :param capped: True when more keys moved than ``full_rows_at_most`` allows: ``rows`` is empty, no full row was
        read, and ``changed_keys`` names them (the caller reads them its own way: one export); or when the moved
        groups held more rows than ``group_rows_at_most`` allows: no key or full row was read, ``changed_keys`` is
        empty and ``moved_groups`` names them
    :ptype capped: bool
    :param moved_groups: of a grouped read (``group``), the groups whose hash moved, as the scope answers them, and the
        held groups the scope no longer answers, as ``held`` names them
    :ptype moved_groups: list[Any]
    :param moved_group_rows: the rows the moved groups hold, as phase 0 answered them
    :ptype moved_group_rows: int
    """

    rows: list[dict[str, Any]]
    gone_keys: list[tuple[Any, ...]]
    changed_keys: list[tuple[Any, ...]] = field(default_factory=list)
    capped: bool = False
    moved_groups: list[Any] = field(default_factory=list)
    moved_group_rows: int = 0
