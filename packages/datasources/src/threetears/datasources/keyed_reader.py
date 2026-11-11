"""a hub-served datasource as the asyncpg-shaped reader the keyed helpers take, so they read a warehouse too.

:mod:`threetears.core.keyed` awaits ``fetch(query, *params)`` on what it is given. :class:`DatasourceReader` answers
that with one :meth:`~threetears.datasources.query_client.DatasourceQueryClient.query` on a named datasource. A
warehouse has no arrays (Redshift): its callers bind sets as lists (``AnyOf(..., as_list=True)``,
``KeyBatch(..., as_list=True)``). An answer the hub cut at its row cap is refused (:class:`DatasourceAnswerCut`),
never taken as every row: a keyed read (:func:`~threetears.core.keyed.read_keyed`) reads a cut batch again in halves,
the cut sized from then on for the rest of its batches.

A warehouse's change reads and grouped counts go through the same helpers::

    reader = DatasourceReader(client, "redshift-results", decode={"updated_at": datetime.fromisoformat})
    moved = await read_past_marks(reader, "results", columns=[...], marks={"updated_at": mark})
    held = await counts_by(reader, "results", column="race_id", values=AnyOf(race_ids, as_list=True))

Parameters cross the rail as JSON (a date or time in ISO form, a UUID or a decimal as text, a list item by item);
a reply's values come back as JSON, so ``decode`` turns a column back into its type (the stamp
:func:`~threetears.core.keyed.read_past_marks` compares with its mark).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

from threetears.core.backends.protocol import L3AnswerCutError
from threetears.core.keyed import Dialect, Fragment, Selected, by_conditions
from threetears.datasources.entities import DataSourceType
from threetears.datasources.query_client import DatasourceQueryClient, read_scope_pages

__all__ = ["DatasourceAnswerCut", "DatasourceReader"]


#: each engine whose statements a row hash is spelled for, onto that spelling (an agent-internal datasource reads
#: through the L3 broker: PostgreSQL's, YugabyteDB's alike)
_ENGINE_DIALECTS: Mapping[DataSourceType, Dialect] = {
    DataSourceType.REDSHIFT: Dialect.REDSHIFT,
    DataSourceType.POSTGRES: Dialect.POSTGRES,
    DataSourceType.YUGABYTE: Dialect.YUGABYTE,
    DataSourceType.AGENT_INTERNAL: Dialect.POSTGRES,
}


#: the pages a scope read takes before it refuses to go on (a relation growing faster than it is read)
_MAX_SCOPE_PAGES = 10_000


class DatasourceAnswerCut(L3AnswerCutError):
    """the hub cut a keyed read's answer (at its row cap, or at its reply's bytes) and said so: the batch named more
    than one answer holds. An :class:`~threetears.core.backends.protocol.L3AnswerCutError`, so a keyed read
    (:func:`~threetears.core.keyed.read_keyed`) reads the batch again in halves rather than raising."""


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
    :param engine: the datasource's engine, from its record (its config's ``datasource_type``): the dialect its
        statements are spelled in (:attr:`sql_dialect`, a row hash's spelling); None when the caller reads nothing
        spelled by engine
    :ptype engine: DataSourceType | None
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
        engine: DataSourceType | None = None,
    ) -> None:
        self._client = client
        self._datasource = datasource
        self._token = user_identity_token
        self._decode = dict(decode or {})
        self._engine = engine

    @property
    def sql_dialect(self) -> Dialect | str:
        """the dialect this datasource's statements are spelled in
        (:data:`~threetears.core.keyed.SQL_DIALECT_ATTRIBUTE`): its engine's, or, for an engine no row hash is
        spelled for or none named, that engine's name, which :func:`~threetears.core.keyed.dialect_of` refuses.

        :return: the dialect, or the engine's name
        :rtype: Dialect | str
        """
        if self._engine is None:
            return "unnamed"
        return _ENGINE_DIALECTS.get(self._engine, str(self._engine))

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        """the query's rows.

        :param query: the SELECT, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters (a date, time, UUID or decimal is sent in its JSON form)
        :ptype params: Any
        :return: the rows, ``decode``'s columns decoded
        :rtype: list[dict[str, Any]]
        :raises DatasourceAnswerCut: when the hub cut the answer (its row cap, or its reply's bytes)
        """
        result = await self._client.query(
            self._datasource, query, [_wire(param) for param in params], user_identity_token=self._token
        )
        if result.truncated:
            raise DatasourceAnswerCut(
                f"{self._datasource}: the hub cut the answer after {result.row_count} rows (its row cap or its "
                f"reply's bytes, {result.max_reply_bytes}); read fewer keys a statement",
                rows=result.row_count,
            )
        rows = [dict(row) for row in result.rows]
        for row in rows:
            for column, turn in self._decode.items():
                if row.get(column) is not None:
                    row[column] = turn(row[column])
        return rows

    async def read_scope(
        self,
        relation: str,
        *,
        columns: Sequence[str | Selected],
        key: Sequence[str],
        by: Mapping[str, Any],
        where: Sequence[Fragment],
    ) -> list[dict[str, Any]]:
        """every row of ``relation`` that ``by`` and ``where`` name, in ``key``'s order, through the datasource pager
        :func:`~threetears.datasources.query_client.read_all` pages by: sized from the reply cap the hub announces,
        reading on past a reply the hub cut, never a cut answer refused or taken as every row. A keyed helper reading
        a scope (:func:`~threetears.core.keyed.read_changed`'s keys and hashes) calls it in place of its own pages.

        :param relation: the relation, a TRUSTED name; read as ``t``
        :ptype relation: str
        :param columns: the columns, as written, or :class:`~threetears.core.keyed.Selected` expressions
        :ptype columns: Sequence[str | Selected]
        :param key: the relation's key, TRUSTED identifiers among ``columns``, the pages' order
        :ptype key: Sequence[str]
        :param by: pinned columns, each a value or an :class:`~threetears.core.keyed.AnyOf`
        :ptype by: Mapping[str, Any]
        :param where: filters, each written as ``t."col"`` from ``$1``
        :ptype where: Sequence[Fragment]
        :return: the rows, ``decode``'s columns decoded
        :rtype: list[dict[str, Any]]
        :raises IncompleteReadError: when paging cannot step past every row
        """
        pinned, values = by_conditions(by, alias="t")
        fragments = [Fragment(" AND ".join(pinned), tuple(_wire(value) for value in values))] if pinned else []
        fragments += [Fragment(fragment.sql, tuple(_wire(value) for value in fragment.params)) for fragment in where]
        rows = await read_scope_pages(
            self._client,
            self._datasource,
            columns=columns,
            relation=relation,
            key=key,
            conditions=fragments,
            user_identity_token=self._token,
            max_pages=_MAX_SCOPE_PAGES,
        )
        for row in rows:
            for column, turn in self._decode.items():
                if row.get(column) is not None:
                    row[column] = turn(row[column])
        return rows
