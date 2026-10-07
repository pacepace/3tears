"""A relation's fingerprint: how many rows it holds, and a digest of which keys they are.

**Two questions, and a count answers only the first.** A count says whether a copy holds as
many rows as its source. The digest says whether they are the same rows: a delete and an insert
leave the count unchanged, so a copy half old and half new passes a count while being a state
the source was never in. Each row's key is rendered as one text value, hashed with MD5, and the
first eight hex digits of each hash are summed.

**The rendering is one rule everywhere.** A NULL renders as ``CHR(30)`` (the ASCII record
separator), never as an empty string, so a key that went NULL is not one that went ``''``;
``CHR(31)`` (the unit separator) joins the columns, so ``('a', 'bc')`` is not ``('ab', 'c')``.
:func:`relation_key_expression` is that rule in SQL, and every engine the platform admits
spells ``CAST``, ``CHR`` and ``||`` the same way. :func:`key_fingerprint` is the same rule in
Python, for keys already read. Turning a hash into a summable number has no portable SQL
spelling, so each engine's digest SQL lives with its driver; :func:`postgres_fingerprint_sql`
is Postgres' (and the L3 tier's).

**Comparable within one rendering only.** For text keys the Postgres digest and the Python
digest are the same number. A non-text key renders as its engine casts it (Postgres writes a
timestamp ``2026-11-04 03:57:11+00``, Python ``2026-11-04 03:57:11+00:00``), so compare a
digest only with one taken the same way over values held the same way.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "KeyFingerprint",
    "key_fingerprint",
    "postgres_fingerprint_sql",
    "relation_key_expression",
]

#: what a NULL key value renders as: a NULL is not an empty string
_NULL_TEXT: Final = chr(30)

#: what joins a key's columns, so a value moving across the boundary is seen
_COLUMN_SEPARATOR: Final = chr(31)

#: how many leading hex digits of each row's MD5 the digest sums
_HASH_HEX_DIGITS: Final = 8


@dataclass(frozen=True)
class KeyFingerprint:
    """how many rows a relation holds and a digest of their keys.

    :ivar row_count: the rows
    :ivar digest: the sum, as text, of each key's hash; compared for equality, never parsed
    """

    row_count: int
    digest: str


def relation_key_expression(key: Sequence[str], *, boolean_columns: Collection[str] = ()) -> str:
    """render the key of one row as a single text value, NULLs distinguished, in SQL.

    A boolean column renders through a ``CASE`` as ``'true'`` or ``'false'`` -- the text Postgres's
    own cast gives -- because Redshift refuses to cast a boolean to text at all. Name every boolean
    column of ``key`` in ``boolean_columns``; any other column is cast.

    :param key: the key's columns, TRUSTED identifiers
    :ptype key: Sequence[str]
    :param boolean_columns: which of them are booleans
    :ptype boolean_columns: Collection[str]
    :return: a SQL expression producing one text value per row
    :rtype: str
    :raises ValueError: when ``key`` is empty, which would render a constant and
        fingerprint every relation of the same size identically
    """
    columns = list(key)
    if not columns:
        raise ValueError("a relation fingerprint needs at least one ordering column")
    rendered = [
        f"CASE WHEN {column} IS NULL THEN CHR(30) WHEN {column} THEN 'true' ELSE 'false' END"
        if column in boolean_columns
        else f"CASE WHEN {column} IS NULL THEN CHR(30) ELSE CAST({column} AS VARCHAR) END"
        for column in columns
    ]
    return " || CHR(31) || ".join(rendered)


def postgres_fingerprint_sql(relation: str, key: Sequence[str], filters: str = "") -> str:
    """one Postgres statement counting ``relation`` and digesting its keys.

    Postgres turns a hash into a summable number by casting its leading hex digits through
    ``bit(32)``. ``SUM`` over ``bigint`` widens to ``numeric``, so a large relation cannot wrap,
    and a wrapped sum would fingerprint two different relations identically.

    :param relation: the relation, a TRUSTED identifier (quoted by the caller where it must be)
    :ptype relation: str
    :param key: the key's columns, TRUSTED identifiers
    :ptype key: Sequence[str]
    :param filters: a `` WHERE ...`` fragment naming the rows to fingerprint; every row when empty
    :ptype filters: str
    :return: the statement, answering ``row_count`` and ``digest``
    :rtype: str
    :raises ValueError: when ``key`` is empty
    """
    return (
        "SELECT COUNT(*) AS row_count, "  # noqa: S608 - relation and key are trusted identifiers
        "COALESCE(SUM(('x' || SUBSTR(MD5(k), 1, 8))::bit(32)::bigint), 0) AS digest "
        f"FROM (SELECT {relation_key_expression(key)} AS k FROM {relation}{filters}) AS fingerprint_source"
    )


def key_fingerprint(keys: Iterable[Sequence[Any]]) -> KeyFingerprint:
    """the fingerprint of keys already read, rendered as :func:`relation_key_expression` renders them.

    Each value renders as ``str(value)``, so give values in the form they are held: the digest of
    two copies agrees only when their values do.

    :param keys: each row's key values, in the key's column order
    :ptype keys: Iterable[Sequence[Any]]
    :return: the count and digest
    :rtype: KeyFingerprint
    """
    count = 0
    total = 0
    for key in keys:
        text = _COLUMN_SEPARATOR.join(_NULL_TEXT if value is None else str(value) for value in key)
        total += int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:_HASH_HEX_DIGITS], 16)
        count += 1
    return KeyFingerprint(row_count=count, digest=str(total))
