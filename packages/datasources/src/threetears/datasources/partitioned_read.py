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
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any, Final, TypeVar

from threetears.core.keyed import Selected
from threetears.observe import get_logger

from threetears.datasources.query_client import (
    CHANGED_READ_ATTEMPTS,
    RESULT_TOO_LARGE,
    DatasourceQueryClient,
    DatasourceQueryError,
    IncompleteReadError,
    RelationFingerprintResult,
    keyset_condition,
    plain_columns,
    read_all,
    select_list,
)

__all__ = ["DEFAULT_PART_CONCURRENCY", "fingerprint_parts", "read_partitions", "read_parts"]

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
    columns: Sequence[str | Selected],
    relation: str,
    key: Sequence[str],
    parts: Sequence[Mapping[str, str]],
    concurrency: int = DEFAULT_PART_CONCURRENCY,
    page_size: int | None = None,
    prove: bool = False,
) -> list[list[dict[str, Any]]]:
    """read each part of a relation whole (:func:`~threetears.datasources.query_client.read_all`), several at once.

    Each part is read by keyset pages with no fingerprint; ``prove=True`` asks ``read_all``'s old proof
    (each part counted and fingerprinted before and after its pages) for each part.

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
    :param prove: fingerprint each part before and after its read (``read_all``'s ``prove``); off by default
    :ptype prove: bool
    :return: each part's rows, in the order of ``parts``
    :rtype: list[list[dict[str, Any]]]
    :raises ValueError: when ``concurrency`` is below one, or the arguments cannot describe a read
    :raises IncompleteReadError: the first part that could not be shown whole
    :raises DatasourceQueryError: the first part the hub or the warehouse refused
    """
    paging: dict[str, Any] = {} if page_size is None else {"page_size": page_size}

    def one(part: Mapping[str, str]) -> Callable[[], Awaitable[list[dict[str, Any]]]]:
        return lambda: read_all(
            client, datasource_name, columns=columns, relation=relation, key=key, where=part, prove=prove, **paging
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


#: rows per page of a read by partitions: under the hub's row cap with room for the extra row each
#: page asks for, as :func:`~threetears.datasources.query_client.read_all`'s pages
_PARTITION_PAGE_ROWS: Final = 900

#: the most rows one batch of parts holds in memory between its read and its check
_BATCH_ROWS: Final = 50_000

#: the most page boundaries one statement may answer: under the hub's row cap
_MAX_BOUNDARIES: Final = 990


def _bound(key: Sequence[str], values: Sequence[Any], *, below: bool, first: int) -> tuple[str, list[Any]]:
    """the condition keeping keys at or above ``values`` (``below`` False) or strictly below them.

    The one keyset builder :func:`~threetears.datasources.query_client.read_all` pages with too.

    :param key: the key's columns, TRUSTED identifiers
    :ptype key: Sequence[str]
    :param values: the bound's key values
    :ptype values: Sequence[Any]
    :param below: strictly below the bound; otherwise at or above it
    :ptype below: bool
    :param first: the first placeholder's number
    :ptype first: int
    :return: the condition, parenthesised, and its parameters
    :rtype: tuple[str, list[Any]]
    """
    condition, params = keyset_condition(
        key, values, earlier="<" if below else ">", last="<" if below else ">=", first=first
    )
    return f"({condition})", params


async def _boundaries(
    client: DatasourceQueryClient,
    datasource_name: str,
    *,
    relation: str,
    key: Sequence[str],
    partition_by: str,
    parts: Sequence[str],
    page_size: int,
) -> dict[str, list[tuple[Any, ...]]]:
    """each part's page starts: the key of its first row and of every ``page_size``-th row after.

    :return: part -> its page starts, in key order
    :rtype: dict[str, list[tuple[Any, ...]]]
    """
    keys = ", ".join(key)
    # the part column leads the read's key: every row read is one of the named parts'
    part_key = partition_by
    placeholders = ", ".join(f"${index + 1}" for index in range(len(parts)))
    starts: dict[str, list[tuple[Any, ...]]] = {part: [] for part in parts}
    # a reply the hub cut (its row cap or its bytes) holds the first starts: the read goes on past the last one it
    # holds (each start is its own row, so none is split), until a reply comes back whole
    after: tuple[Any, ...] | None = None
    while True:
        past, more = (
            ("", [])
            if after is None
            else keyset_condition(["part__", *key], after, earlier=">", last=">", first=len(parts) + 1)
        )
        sql = (
            f"SELECT part__, {keys} FROM (SELECT {part_key} AS part__, {keys}, "  # noqa: S608 - trusted identifiers
            f"ROW_NUMBER() OVER (PARTITION BY {part_key} ORDER BY {keys}) AS row__ "
            f"FROM {relation} WHERE {part_key} IN ({placeholders})) AS paged "
            f"WHERE MOD(row__ - 1, {int(page_size)}) = 0{f' AND ({past})' if past else ''} ORDER BY part__, {keys}"
        )
        result = await client.query(datasource_name, sql, params=[*parts, *more])
        for row in result.rows:
            starts[str(row["part__"])].append(tuple(row[column] for column in key))
        if not result.truncated:
            return starts
        if not result.rows:
            raise IncompleteReadError(f"{datasource_name}: {relation}'s page starts came back cut and empty")
        last = result.rows[-1]
        after = (last["part__"], *(last[column] for column in key))


async def _read_pages(
    client: DatasourceQueryClient,
    datasource_name: str,
    *,
    columns: Sequence[str | Selected],
    relation: str,
    key: Sequence[str],
    partition_by: str,
    part: str,
    starts: Sequence[tuple[Any, ...]],
    page_size: int,
) -> list[Callable[[], Awaitable[list[dict[str, Any]]]]]:
    """one call per page of a part, each reading from its start up to the next page's.

    :return: the calls, in key order
    :rtype: list[Callable[[], Awaitable[list[dict[str, Any]]]]]
    """
    selected = select_list(columns)
    ordering = ", ".join(key)

    def one(index: int) -> Callable[[], Awaitable[list[dict[str, Any]]]]:
        async def page() -> list[dict[str, Any]]:
            # a reply the hub cut (its bytes) holds a prefix of the page: the page reads on past the cut reply's
            # last whole key group until a reply comes back whole
            collected: list[dict[str, Any]] = []
            after: tuple[Any, ...] | None = None
            while True:
                if after is None:
                    lower, params = _bound(key, starts[index], below=False, first=2)
                else:
                    past, params = keyset_condition(key, after, earlier=">", last=">", first=2)
                    lower = f"({past})"
                conditions = [f"{partition_by} = $1", lower]
                if index + 1 < len(starts):
                    upper, more = _bound(key, starts[index + 1], below=True, first=2 + len(params))
                    conditions.append(upper)
                    params += more
                sql = (
                    f"SELECT {selected} FROM {relation} WHERE {' AND '.join(conditions)} "  # noqa: S608 - trusted identifiers
                    f"ORDER BY {ordering} LIMIT {int(page_size) + 1}"
                )
                result = await client.query(datasource_name, sql, params=[part, *params])
                rows = result.rows
                if len(collected) + len(rows) > page_size:
                    raise IncompleteReadError(
                        f"{datasource_name}: {relation} {partition_by}={part}: a page held more than {page_size} "
                        "rows; rows were written during the read, or the key is not unique"
                    )
                if not result.truncated:
                    return collected + rows
                kept = list(rows)
                cut_key = tuple(kept[-1][column] for column in key) if kept else None
                while kept and tuple(kept[-1][column] for column in key) == cut_key:
                    kept.pop()
                if not kept and cut_key is not None:
                    # one key a reply (wide rows): whole when the part holds no more rows of it, one keyed count says
                    same = " AND ".join(f"{column} = ${index}" for index, column in enumerate(key, 2))
                    counted = await client.query(
                        datasource_name,
                        f"SELECT count(*) AS n FROM {relation} WHERE {partition_by} = $1 AND {same}",  # noqa: S608
                        params=[part, *cut_key],
                    )
                    if counted.rows and int(counted.rows[0]["n"]) == len(rows):
                        kept = list(rows)
                if not kept:
                    raise IncompleteReadError(
                        f"{datasource_name}: {relation} {partition_by}={part}: the hub cut a page within one value of "
                        f"{tuple(key)}: one key group is wider than one reply, so the page cannot be read whole"
                    )
                collected.extend(kept)
                after = tuple(kept[-1][column] for column in key)

        return page

    return [one(index) for index in range(len(starts))]


def _batches(counts: Mapping[str, int], *, batch_rows: int, page_size: int) -> list[list[str]]:
    """the parts in sorted order, grouped so a group's rows and page starts stay bounded.

    :return: the groups
    :rtype: list[list[str]]
    """
    batches: list[list[str]] = []
    rows = starts = 0
    for part in sorted(counts):
        pages = math.ceil(counts[part] / page_size)
        if batches and (rows + counts[part] > batch_rows or starts + pages > _MAX_BOUNDARIES):
            batches.append([])
            rows = starts = 0
        if not batches:
            batches.append([])
        batches[-1].append(part)
        rows += counts[part]
        starts += pages
    return batches


async def read_partitions(
    client: DatasourceQueryClient,
    datasource_name: str,
    *,
    columns: Sequence[str | Selected],
    relation: str,
    key: Sequence[str],
    partition_by: str,
    parts: Sequence[str] | None = None,
    concurrency: int = DEFAULT_PART_CONCURRENCY,
    page_size: int = _PARTITION_PAGE_ROWS,
    batch_rows: int = _BATCH_ROWS,
    counts: Mapping[str, int] | None = None,
    prove: bool = False,
) -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    """every part of a relation (or the ``parts`` asked for) over the read rail, pages read side by side, one part at a time.

    **Why not part by part.** :func:`read_parts` pages each part from its start, one page after
    another (keyset paging cannot know where page two begins until page one is back). Here one
    statement answers where every page of a batch of parts begins (each part's key at every
    ``page_size``-th row), and then every page is read at once, at most ``concurrency`` at a time:
    the round trips run side by side instead of one after another.

    **Named parts are read by key, with no fingerprint.** Given ``parts``, each part's page starts are
    asked in one statement and its pages read; nothing is fingerprinted. Given ``counts`` too (each
    part's rows as the source's change record or the caller's own count reported them), a part is
    handed back only if its pages hold exactly its count, and raised otherwise; without them, a page
    holding more rows than it should is still refused. A read that crosses a write is picked up by
    the next load, which the source's change record names.

    **With ``prove=True``, or no ``parts`` (the parts are then found by the grouped fingerprint), the
    old proof runs.** Every part is fingerprinted in ONE grouped ask before
    the read; once a batch's pages are read, its parts are fingerprinted again in one ask, and a part
    is handed back only if its fingerprint did not move and its pages hold exactly its count. A page
    holding more rows than it should (rows written mid-read, or a key that is not unique) is refused
    at once. A part asked for that holds no rows comes back empty.

    **A batch whose part changed while it was read is read again.** The fingerprints taken after the
    read are the next read's "before", and the whole batch is read again from its page starts, up to
    three reads in all (each re-read logged at warning). Nothing of the batch has been handed back
    yet, so the caller only ever sees parts proven whole. A part still moving on the last read is
    raised as before (logged at error). No other failure is read again.

    The batches bound what is held: a batch's rows wait for its check; the next batch is read while
    the caller works on this one.

    :param client: the connected query client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param columns: the columns to read; every fingerprint covers all of them
    :ptype columns: Sequence[str]
    :param relation: the relation, a TRUSTED identifier
    :ptype relation: str
    :param key: the key each part's rows are unique by and ordered by
    :ptype key: Sequence[str]
    :param partition_by: the column whose values are the parts, never NULL
    :ptype partition_by: str
    :param parts: the parts to read; every part the relation holds when None
    :ptype parts: Sequence[str] | None
    :param concurrency: pages read at once: at or under the datasource's cap
    :ptype concurrency: int
    :param page_size: rows per page, under the hub's row cap
    :ptype page_size: int
    :param batch_rows: the most rows one batch of parts holds
    :ptype batch_rows: int
    :param counts: each named part's rows as counted by the caller, checked against its pages; a part
        it leaves out is not checked
    :ptype counts: Mapping[str, int] | None
    :param prove: fingerprint every part before and after its batch's read (the old proof); off by
        default, and always on when ``parts`` is None
    :ptype prove: bool
    :return: each part's value and rows, in sorted order
    :rtype: AsyncIterator[tuple[str, list[dict[str, Any]]]]
    :raises IncompleteReadError: when a part moved during each of its batch's reads, or its pages do not
        hold exactly its rows, or a part's value is NULL
    :raises DatasourceQueryError: when the hub refuses a statement (an older hub refuses the grouped
        fingerprint as ``MALFORMED_REQUEST``)
    """
    if parts is not None and not prove:
        async for part, rows in _read_named_partitions(
            client,
            datasource_name,
            columns=columns,
            relation=relation,
            key=key,
            partition_by=partition_by,
            parts=parts,
            concurrency=concurrency,
            page_size=page_size,
            batch_rows=batch_rows,
            counts=counts or {},
        ):
            yield part, rows
        return
    where_in = None if parts is None else {partition_by: list(parts)}
    before = await client.relation_fingerprint_groups(
        datasource_name, relation=relation, key=plain_columns(columns), group_by=partition_by, where_in=where_in
    )
    if None in before:
        raise IncompleteReadError(f"{datasource_name}: {relation} holds rows whose {partition_by} is NULL")
    counts = {str(part): fingerprint.row_count for part, fingerprint in before.items() if part is not None}
    for part in parts or ():
        counts.setdefault(part, 0)

    async def read_batch(
        members: list[str],
    ) -> tuple[list[tuple[str, list[dict[str, Any]]]], dict[str | None, RelationFingerprintResult], str | None]:
        """one read of a batch: its parts' rows, their fingerprints after the read, and the first part that moved."""
        holding = [part for part in members if counts[part] > 0]
        starts = (
            await _boundaries(
                client,
                datasource_name,
                relation=relation,
                key=key,
                partition_by=partition_by,
                parts=holding,
                page_size=page_size,
            )
            if holding
            else {}
        )
        calls: list[Callable[[], Awaitable[list[dict[str, Any]]]]] = []
        spans: dict[str, tuple[int, int]] = {}
        for part in holding:
            pages = await _read_pages(
                client,
                datasource_name,
                columns=columns,
                relation=relation,
                key=key,
                partition_by=partition_by,
                part=part,
                starts=starts[part],
                page_size=page_size,
            )
            spans[part] = (len(calls), len(calls) + len(pages))
            calls += pages
        try:
            pages_read = await _bounded(calls, concurrency)
        except DatasourceQueryError as exc:
            # a page too large for one reply: this batch's parts are read part by part instead,
            # with read_all halving its pages until they fit
            if exc.error_code != RESULT_TOO_LARGE:
                raise
            pages_read = []
            spans = {}
            for part in holding:
                rows = await read_all(
                    client,
                    datasource_name,
                    columns=columns,
                    relation=relation,
                    key=key,
                    where={partition_by: part},
                    page_size=page_size,
                )
                spans[part] = (len(pages_read), len(pages_read) + 1)
                pages_read.append(rows)
        after = (
            await client.relation_fingerprint_groups(
                datasource_name,
                relation=relation,
                key=plain_columns(columns),
                group_by=partition_by,
                where_in={partition_by: members},
            )
            if holding
            else {}
        )
        read: list[tuple[str, list[dict[str, Any]]]] = []
        for part in members:
            start, end = spans.get(part, (0, 0))
            rows = [row for page in pages_read[start:end] for row in page]
            if after.get(part) != before.get(part):
                return read, after, part
            if len(rows) != counts[part]:
                raise IncompleteReadError(
                    f"{datasource_name}: {relation} {partition_by}={part} counted {counts[part]} rows and the read "
                    f"returned {len(rows)}"
                )
            read.append((part, rows))
        return read, after, None

    async def batch(members: list[str]) -> list[tuple[str, list[dict[str, Any]]]]:
        moved: str | None = None
        for attempt in range(1, CHANGED_READ_ATTEMPTS + 1):
            read, after, moved = await read_batch(members)
            if moved is None:
                return read
            if attempt < CHANGED_READ_ATTEMPTS:
                log.warning(
                    "a part changed while it was read; reading its batch again",
                    extra={
                        "extra_data": {
                            "relation": relation,
                            "part": moved,
                            "attempt": attempt,
                            "attempts": CHANGED_READ_ATTEMPTS,
                        }
                    },
                )
                # taken after the read that moved: what the next read must find unchanged, and count
                for part in members:
                    if part in after:
                        before[part] = after[part]
                        counts[part] = after[part].row_count
                    else:
                        before.pop(part, None)
                        counts[part] = 0
        log.error(
            "a part changed during every read of its batch",
            extra={"extra_data": {"relation": relation, "part": moved, "attempts": CHANGED_READ_ATTEMPTS}},
        )
        raise IncompleteReadError(
            f"{datasource_name}: {relation} {partition_by}={moved} changed while it was read; re-read once it is settled"
        )

    batches = _batches(counts, batch_rows=batch_rows, page_size=page_size)
    # the next batch is read while the caller works on this one, once this one's reads are done: two
    # batches reading at once would ask the datasource for twice its cap
    pending = asyncio.ensure_future(batch(batches[0])) if batches else None
    try:
        for index in range(len(batches)):
            if pending is None:
                break
            done = await pending
            pending = asyncio.ensure_future(batch(batches[index + 1])) if index + 1 < len(batches) else None
            for part, rows in done:
                yield part, rows
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.wait({pending})
    log.info(
        "read a relation by partitions",
        extra={
            "extra_data": {
                "relation": relation,
                "parts": len(counts),
                "rows": sum(counts.values()),
                "batches": len(batches),
                "concurrency": concurrency,
            }
        },
    )


async def _read_named_partitions(
    client: DatasourceQueryClient,
    datasource_name: str,
    *,
    columns: Sequence[str | Selected],
    relation: str,
    key: Sequence[str],
    partition_by: str,
    parts: Sequence[str],
    concurrency: int,
    page_size: int,
    batch_rows: int,
    counts: Mapping[str, int],
) -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    """:func:`read_partitions` over named parts, by key, with no fingerprint.

    A part with a count is batched by it; a part without one is a batch of its own, so a batch's rows
    stay bounded by what is known of them. Each batch's page starts come in one statement, its pages are
    read side by side, and each part is checked against its count when it has one.

    :return: each part's value and rows: the counted parts in sorted order, then the others in sorted order
    :rtype: AsyncIterator[tuple[str, list[dict[str, Any]]]]
    :raises IncompleteReadError: when a counted part's pages do not hold exactly its count, or a page holds
        more rows than it should
    """
    counted = {part: counts[part] for part in parts if part in counts}
    batches = _batches(counted, batch_rows=batch_rows, page_size=page_size) + [
        [part] for part in sorted(set(parts) - set(counted))
    ]
    total = 0
    for members in batches:
        starts = await _boundaries(
            client,
            datasource_name,
            relation=relation,
            key=key,
            partition_by=partition_by,
            parts=members,
            page_size=page_size,
        )
        calls: list[Callable[[], Awaitable[list[dict[str, Any]]]]] = []
        spans: dict[str, tuple[int, int]] = {}
        for part in members:
            pages = await _read_pages(
                client,
                datasource_name,
                columns=columns,
                relation=relation,
                key=key,
                partition_by=partition_by,
                part=part,
                starts=starts[part],
                page_size=page_size,
            )
            spans[part] = (len(calls), len(calls) + len(pages))
            calls += pages
        try:
            pages_read = await _bounded(calls, concurrency)
        except DatasourceQueryError as exc:
            # a page too large for one reply: this batch's parts are read part by part instead,
            # with read_all halving its pages until they fit
            if exc.error_code != RESULT_TOO_LARGE:
                raise
            pages_read = []
            for part in members:
                rows = await read_all(
                    client,
                    datasource_name,
                    columns=columns,
                    relation=relation,
                    key=key,
                    where={partition_by: part},
                    page_size=page_size,
                )
                spans[part] = (len(pages_read), len(pages_read) + 1)
                pages_read.append(rows)
        for part in members:
            start, end = spans[part]
            rows = [row for page in pages_read[start:end] for row in page]
            if part in counted and len(rows) != counted[part]:
                raise IncompleteReadError(
                    f"{datasource_name}: {relation} {partition_by}={part} counted {counted[part]} rows and the read "
                    f"returned {len(rows)}"
                )
            total += len(rows)
            yield part, rows
    log.info(
        "read named partitions by key",
        extra={
            "extra_data": {
                "relation": relation,
                "parts": len(parts),
                "rows": total,
                "batches": len(batches),
                "concurrency": concurrency,
            }
        },
    )
