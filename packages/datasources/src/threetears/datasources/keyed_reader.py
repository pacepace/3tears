"""a hub-served datasource as the asyncpg-shaped reader the keyed helpers take, so they read a warehouse too.

:mod:`threetears.core.keyed` awaits ``fetch(query, *params)`` on what it is given. :class:`DatasourceReader` answers
that with one :meth:`~threetears.datasources.query_client.DatasourceQueryClient.query` on a named datasource. A
warehouse has no arrays (Redshift): its callers bind sets as lists (``AnyOf(..., as_list=True)``,
``KeyBatch(..., as_list=True)``). An answer the hub cut at its row cap is refused (:class:`DatasourceAnswerCut`),
never taken as every row: a keyed helper sizes its batches under the cap, so a cut answer is a batch too large.
"""

from __future__ import annotations

from typing import Any

from threetears.datasources.query_client import DatasourceQueryClient

__all__ = ["DatasourceAnswerCut", "DatasourceReader"]


class DatasourceAnswerCut(RuntimeError):
    """the hub cut a keyed read's answer at its row cap: the batch named more rows than one answer holds."""


class DatasourceReader:
    """reads a datasource through the query client, one query a ``fetch``.

    :param client: the process's query client
    :ptype client: DatasourceQueryClient
    :param datasource: the datasource's name, as the hub holds it
    :ptype datasource: str
    """

    def __init__(self, client: DatasourceQueryClient, datasource: str) -> None:
        self._client = client
        self._datasource = datasource

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        """the query's rows.

        :param query: the SELECT, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters, JSON-native values
        :ptype params: Any
        :return: the rows
        :rtype: list[dict[str, Any]]
        :raises DatasourceAnswerCut: when the hub cut the answer at its row cap
        """
        result = await self._client.query(self._datasource, query, list(params))
        if result.truncated:
            raise DatasourceAnswerCut(
                f"{self._datasource}: the answer was cut at the hub's row cap after {result.row_count} rows; "
                "read fewer keys a statement"
            )
        return [dict(row) for row in result.rows]
