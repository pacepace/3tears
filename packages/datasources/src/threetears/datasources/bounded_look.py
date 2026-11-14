"""a bounded look at an external datasource's table: how many rows an author's predicate matches, up to a cap, and a
few of them.

For a check run on its own (the hub's concept binding validation), never a read of the table: both statements are
bounded by their ``LIMIT``, whatever the predicate, so the no-whole-reads gate judges a call to either as bounded and
needs no key. They speak to the datasource's own :class:`~threetears.datasources.drivers.base.Driver`, in its
dialect: identifiers quoted, the ``LIMIT`` an integer written into the statement (some warehouse drivers bind no
placeholder there), the predicate the author's text, already checked by the caller (the hub's ``validate_read_sql``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from threetears.core.sql_fragments import quote_identifier
from threetears.datasources.drivers.base import Driver

__all__ = ["SAMPLE_LIMIT_MAX", "BoundedCount", "count_matching_at_most", "sample_matching"]

#: the most rows a sample may ask for: a look, never a read
SAMPLE_LIMIT_MAX: Final[int] = 100


@dataclass(frozen=True, slots=True)
class BoundedCount:
    """how many rows matched, counted no further than the cap.

    :ivar count: the rows counted, at most the cap
    :ivar more: whether the cap was reached (as many rows as the cap, or more, match)
    """

    count: int
    more: bool


def _relation(schema: str, table: str) -> str:
    """the table, schema-qualified and each part quoted."""
    return f"{quote_identifier(schema)}.{quote_identifier(table)}"


def _matching(predicate: str | None) -> str:
    """the ``WHERE`` the author's predicate makes, or none."""
    return f" WHERE {predicate}" if predicate and predicate.strip() else ""


def _bound(value: int, name: str, most: int | None = None) -> int:
    """a cap or limit as the integer written into the statement.

    :raises ValueError: when it is not a positive integer, or over ``most``
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if most is not None and value > most:
        raise ValueError(f"{name} is at most {most}, got {value}")
    return value


async def count_matching_at_most(
    driver: Driver, schema: str, table: str, predicate: str | None, *, at_most: int
) -> BoundedCount:
    """count the rows ``predicate`` matches, reading no more than ``at_most`` of them.

    ``SELECT COUNT(*) FROM (SELECT 1 FROM <schema.table> [WHERE <predicate>] LIMIT <at_most>) AS capped``

    :param driver: the datasource's driver
    :ptype driver: Driver
    :param schema: the table's schema, TRUSTED as a name (quoted here)
    :ptype schema: str
    :param table: the table, TRUSTED as a name (quoted here)
    :ptype table: str
    :param predicate: the author's predicate, already checked by the caller; None or blank for every row
    :ptype predicate: str | None
    :param at_most: the cap
    :ptype at_most: int
    :return: the count, and whether the cap was reached
    :rtype: BoundedCount
    :raises ValueError: when ``at_most`` is not a positive integer
    """
    cap = _bound(at_most, "at_most")
    sql = (
        "SELECT COUNT(*) AS match_count FROM "
        f"(SELECT 1 AS matched FROM {_relation(schema, table)}{_matching(predicate)} LIMIT {cap}) AS capped"
    )
    rows = await driver.fetch(sql)
    count = int(next(iter(rows[0].values()))) if rows else 0
    return BoundedCount(count=count, more=count >= cap)


async def sample_matching(
    driver: Driver, schema: str, table: str, predicate: str | None, *, limit: int
) -> list[dict[str, Any]]:
    """a few rows ``predicate`` matches, in no order.

    ``SELECT * FROM <schema.table> [WHERE <predicate>] LIMIT <limit>``

    :param driver: the datasource's driver
    :ptype driver: Driver
    :param schema: the table's schema, TRUSTED as a name (quoted here)
    :ptype schema: str
    :param table: the table, TRUSTED as a name (quoted here)
    :ptype table: str
    :param predicate: the author's predicate, already checked by the caller; None or blank for any row
    :ptype predicate: str | None
    :param limit: the rows wanted, at most :data:`SAMPLE_LIMIT_MAX`
    :ptype limit: int
    :return: at most ``limit`` rows
    :rtype: list[dict[str, Any]]
    :raises ValueError: when ``limit`` is not a positive integer or is over :data:`SAMPLE_LIMIT_MAX`
    """
    wanted = _bound(limit, "limit", SAMPLE_LIMIT_MAX)
    sql = f"SELECT * FROM {_relation(schema, table)}{_matching(predicate)} LIMIT {wanted}"  # noqa: S608 - quoted names, a checked predicate
    return list(await driver.fetch_at_most(sql, max_rows=wanted))
