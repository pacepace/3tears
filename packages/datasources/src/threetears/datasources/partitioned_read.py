"""Reading a relation by parts: fingerprint each part, read the parts that moved, several at once.

**Why parts.** A relation that a source rebuilds whole, with no change marker of its own, can still
be kept current cheaply when it divides into parts that change independently (one state's results,
one customer's rows): fingerprint every part, compare each with the fingerprint recorded when it was
last read, and read only the parts whose fingerprint moved. :func:`fingerprint_parts` and
:func:`read_parts` are the two halves; ``where`` names each part as equality filters, exactly as
:func:`~threetears.datasources.query_client.read_all` takes them.

**A fingerprint over values, not only keys.** The digest covers the columns it is given. Given the
relation's key it says which rows are there; given every column read it says whether any value in
the part changed too, which is what a caller comparing parts needs.

**Several at once, never more than asked.** A read through the datasource rail costs mostly waiting
(the round trip and the warehouse's planning), not bytes, so parts are asked for side by side, at
most ``concurrency`` at a time. The hub runs at most the datasource driver's own connection cap at
once (:data:`DEFAULT_PART_CONCURRENCY`, Redshift's default), lets a couple more wait a few seconds,
and refuses the rest ``DATASOURCE_BUSY`` (nothing ran; the client asks again briefly, then raises).
So asking for more than the cap does not only queue: past the hub's short queue it fails the read.
Keep ``concurrency`` at or under the datasource's cap.

**One failure fails the read.** The first part that raises cancels the parts still running and is
raised as itself (a refused statement, an incomplete part), so a caller never takes some parts as
read when the read as a whole was not.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final, TypeVar

from threetears.observe import get_logger

from threetears.datasources.query_client import DatasourceQueryClient, RelationFingerprintResult, read_all

__all__ = ["DEFAULT_PART_CONCURRENCY", "fingerprint_parts", "read_parts"]

log = get_logger(__name__)

#: parts asked for at once unless the caller says otherwise: the hub's default number of open
#: warehouse connections per datasource (``connection_cache_size``)
DEFAULT_PART_CONCURRENCY: Final = 5

_ResultT = TypeVar("_ResultT")


async def _bounded(calls: Sequence[Callable[[], Awaitable[_ResultT]]], concurrency: int) -> list[_ResultT]:
    """run every call, at most ``concurrency`` at once, and answer their results in the calls' order.

    :param calls: the calls
    :ptype calls: Sequence[Callable[[], Awaitable[_ResultT]]]
    :param concurrency: the most running at once
    :ptype concurrency: int
    :return: each call's result, in order
    :rtype: list[_ResultT]
    :raises ValueError: when ``concurrency`` is below one
    :raises Exception: the first failure, the calls still running cancelled
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    gate = asyncio.Semaphore(concurrency)

    async def gated(call: Callable[[], Awaitable[_ResultT]]) -> _ResultT:
        async with gate:
            return await call()

    tasks: list[asyncio.Task[_ResultT]] = []
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(gated(call)) for call in calls]
    except* Exception as failures:
        # the group has cancelled the rest; the caller learns what failed, not that a group did
        raise failures.exceptions[0] from None
    return [task.result() for task in tasks]


async def fingerprint_parts(
    client: DatasourceQueryClient,
    datasource_name: str,
    *,
    relation: str,
    columns: Sequence[str],
    parts: Sequence[Mapping[str, str]],
    concurrency: int = DEFAULT_PART_CONCURRENCY,
) -> list[RelationFingerprintResult]:
    """count and fingerprint each part of a relation over ``columns``, several parts at once.

    :param client: the connected query client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource, as the hub names it
    :ptype datasource_name: str
    :param relation: the relation, a TRUSTED identifier
    :ptype relation: str
    :param columns: the columns the digest covers, TRUSTED identifiers: every column read, for a
        digest that moves when any value does
    :ptype columns: Sequence[str]
    :param parts: each part's equality filters, column -> value
    :ptype parts: Sequence[Mapping[str, str]]
    :param concurrency: the most parts asked for at once
    :ptype concurrency: int
    :return: each part's fingerprint, in the order of ``parts``
    :rtype: list[RelationFingerprintResult]
    :raises ValueError: when ``concurrency`` is below one
    :raises DatasourceQueryError: the first part the hub or the warehouse refused
    """

    def one(part: Mapping[str, str]) -> Callable[[], Awaitable[RelationFingerprintResult]]:
        return lambda: client.relation_fingerprint(datasource_name, relation=relation, key=columns, where=part)

    results = await _bounded([one(part) for part in parts], concurrency)
    log.debug(
        "fingerprinted a relation by parts",
        extra={"extra_data": {"relation": relation, "parts": len(parts), "concurrency": concurrency}},
    )
    return results


async def read_parts(
    client: DatasourceQueryClient,
    datasource_name: str,
    *,
    columns: Sequence[str],
    relation: str,
    key: Sequence[str],
    parts: Sequence[Mapping[str, str]],
    concurrency: int = DEFAULT_PART_CONCURRENCY,
    page_size: int | None = None,
) -> list[list[dict[str, Any]]]:
    """read each part of a relation whole (:func:`~threetears.datasources.query_client.read_all`), several at once.

    :param client: the connected query client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource, as the hub names it
    :ptype datasource_name: str
    :param columns: the columns to read, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :param relation: the relation, a TRUSTED identifier
    :ptype relation: str
    :param key: the ordering key each part pages by; unique within a part
    :ptype key: Sequence[str]
    :param parts: each part's equality filters, column -> value
    :ptype parts: Sequence[Mapping[str, str]]
    :param concurrency: the most parts read at once
    :ptype concurrency: int
    :param page_size: rows per page; ``read_all``'s default when None
    :ptype page_size: int | None
    :return: each part's rows, in the order of ``parts``
    :rtype: list[list[dict[str, Any]]]
    :raises ValueError: when ``concurrency`` is below one, or the arguments cannot describe a read
    :raises IncompleteReadError: the first part that could not be shown whole
    :raises DatasourceQueryError: the first part the hub or the warehouse refused
    """
    paging: dict[str, Any] = {} if page_size is None else {"page_size": page_size}

    def one(part: Mapping[str, str]) -> Callable[[], Awaitable[list[dict[str, Any]]]]:
        return lambda: read_all(
            client, datasource_name, columns=columns, relation=relation, key=key, where=part, **paging
        )

    results = await _bounded([one(part) for part in parts], concurrency)
    log.info(
        "read a relation by parts",
        extra={
            "extra_data": {
                "relation": relation,
                "parts": len(parts),
                "rows": sum(len(rows) for rows in results),
                "concurrency": concurrency,
            }
        },
    )
    return results
