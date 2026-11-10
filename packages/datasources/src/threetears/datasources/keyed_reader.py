"""a hub-served datasource as the asyncpg-shaped reader the keyed helpers take, so they read a warehouse too.

:mod:`threetears.core.keyed` awaits ``fetch(query, *params)`` on what it is given. :class:`DatasourceReader` answers
that with one :meth:`~threetears.datasources.query_client.DatasourceQueryClient.query` on a named datasource. A
warehouse has no arrays (Redshift): its callers bind sets as lists (``AnyOf(..., as_list=True)``,
``KeyBatch(..., as_list=True)``). An answer the hub cut at its row cap is refused (:class:`DatasourceAnswerCut`),
never taken as every row: a keyed helper sizes its batches under the cap, so a cut answer is a batch too large.

A warehouse's change reads and grouped counts go through the same helpers::

    reader = DatasourceReader(client, "redshift-results", decode={"updated_at": datetime.fromisoformat})
    moved = await read_past_marks(reader, "results", columns=[...], marks={"updated_at": mark})
    held = await counts_by(reader, "results", column="race_id", values=AnyOf(race_ids, as_list=True))

Parameters cross the rail as JSON (a date or time in ISO form, a UUID or a decimal as text, a list item by item);
a reply's values come back as JSON, so ``decode`` turns a column back into its type (the stamp
:func:`~threetears.core.keyed.read_past_marks` compares with its mark).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

from threetears.datasources.query_client import DatasourceQueryClient

__all__ = ["DatasourceAnswerCut", "DatasourceReader"]


class DatasourceAnswerCut(RuntimeError):
    """the hub cut a keyed read's answer at its row cap: the batch named more rows than one answer holds."""


def _wire(value: Any) -> Any:
    """a parameter as the rail's JSON carries it: a date or time in ISO form, a UUID or a decimal as text."""
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, UUID | Decimal):
        return str(value)
    if isinstance(value, list | tuple):
        return [_wire(item) for item in value]
    return value


class DatasourceReader:
    """reads a datasource through the query client, one query a ``fetch``.

    :param client: the process's query client
    :ptype client: DatasourceQueryClient
    :param datasource: the datasource's name, as the hub holds it
    :ptype datasource: str
    :param user_identity_token: the per-turn user assertion when a human is in the loop; the principal's own grants
        alone when None
    :ptype user_identity_token: str | None
    :param decode: column -> how its JSON value is turned back into its type (``datetime.fromisoformat``); a NULL
        stays None
    :ptype decode: Mapping[str, Callable[[Any], Any]] | None
    """

    #: the rail carries JSON: arrays bind as text, and a warehouse binds none (``as_list=True``)
    binds_native_arrays = False

    def __init__(
        self,
        client: DatasourceQueryClient,
        datasource: str,
        *,
        user_identity_token: str | None = None,
        decode: Mapping[str, Callable[[Any], Any]] | None = None,
    ) -> None:
        self._client = client
        self._datasource = datasource
        self._token = user_identity_token
        self._decode = dict(decode or {})

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        """the query's rows.

        :param query: the SELECT, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters (a date, time, UUID or decimal is sent in its JSON form)
        :ptype params: Any
        :return: the rows, ``decode``'s columns decoded
        :rtype: list[dict[str, Any]]
        :raises DatasourceAnswerCut: when the hub cut the answer at its row cap
        """
        result = await self._client.query(
            self._datasource, query, [_wire(param) for param in params], user_identity_token=self._token
        )
        if result.truncated:
            raise DatasourceAnswerCut(
                f"{self._datasource}: the answer was cut at the hub's row cap after {result.row_count} rows; "
                "read fewer keys a statement"
            )
        rows = [dict(row) for row in result.rows]
        for row in rows:
            for column, turn in self._decode.items():
                if row.get(column) is not None:
                    row[column] = turn(row[column])
        return rows
