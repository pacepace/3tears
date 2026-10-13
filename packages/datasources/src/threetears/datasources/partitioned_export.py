"""A relation read whole, or its chosen parts, through ONE warehouse export divided by a column.

**Why one export.** Reading a relation part by part through exports costs one round trip per part
(a fingerprint, an ``UNLOAD``, a fingerprint, a read, a delete), and the round trips, not the bytes,
are the time. :func:`export_partitions` asks for one ``UNLOAD`` of every part at once, divided by
the part's column (``PARTITION BY``: one directory of files per value), and hands the parts back one
at a time, so a caller holds one part's rows at a time and not the relation's.

**Proven before the first part is handed back**, every check a refusal
(:class:`~threetears.datasources.export_read.IncompleteExportError`), as for one part
(:func:`~threetears.datasources.export_read.export_part`):

- every part is fingerprinted, in ONE grouped ask, before the export and again after it; a part
  whose fingerprint moved is refused, so the files are one state of every part;
- the warehouse wrote as many rows as the parts count together;
- the manifest is the export's own, every file it lists sits in its own part's directory under the
  export's prefix in the bucket the caller reads, it names no part that was not asked for, and the
  rows it counts for each part are that part's count.

Each part's files are then read when the caller reaches it, and the part is handed back only if they
hold exactly its count. A part asked for that holds no rows is handed back empty, so a caller can
tell "now empty" from "not read".

**Then deleted, always**: once every part was handed back, when the caller stops early or fails, and
when a proof refuses it -- from the moment the warehouse answered the export, no way out leaves a
copy of warehouse rows behind (Pace's ruling: delete after load, no timer). A refused export is
logged at WARNING with what was refused (the proof, the counts and fingerprints) so an operator can
tell why without the files.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any, Final
from urllib.parse import unquote

from threetears.observe import get_logger

from threetears.datasources.export_read import (
    ExportStore,
    IncompleteExportError,
    _delete_after_load,
    _parquet_rows,
    _read_object,
    export_select,
)
from threetears.datasources.query_client import (
    DatasourceExportResult,
    DatasourceQueryClient,
    RelationFingerprintResult,
)

__all__ = ["HIVE_NULL_PARTITION", "export_partitions"]

log = get_logger(__name__)

#: the directory the warehouse writes a NULL partition value as
HIVE_NULL_PARTITION: Final = "__HIVE_DEFAULT_PARTITION__"

#: files of one part read at once
_DEFAULT_FILE_CONCURRENCY: Final = 8


def _partition_files(
    manifest: Mapping[str, Any], result: DatasourceExportResult, partition_by: str
) -> dict[str | None, tuple[list[str], int | None]]:
    """each part's files, as the manifest lists them, and the rows it counts for each part.

    :param manifest: the parsed manifest
    :ptype manifest: Mapping[str, Any]
    :param result: the export
    :ptype result: DatasourceExportResult
    :param partition_by: the column the export is divided by
    :ptype partition_by: str
    :return: part value (``None`` for NULL) -> its keys, and the rows counted (None when the
        manifest does not count every file)
    :rtype: dict[str | None, tuple[list[str], int | None]]
    :raises IncompleteExportError: when an entry is not a file of one part's directory under the export
    """
    where = f"s3://{result.bucket}/{result.object_prefix}"
    parts: dict[str | None, tuple[list[str], int | None]] = {}
    for entry in manifest.get("entries", []):
        url = str(entry.get("url", ""))
        relative = url.removeprefix(where)
        directory, _, file_name = relative.partition("/")
        column, equals, encoded = directory.partition("=")
        if (
            relative == url
            or not file_name
            or "/" in file_name
            or file_name in (".", "..")
            or column != partition_by
            or not equals
        ):
            raise IncompleteExportError(
                f"the export's manifest names {url!r}, which is not a file of a part of {where}"
            )
        value = None if encoded == HIVE_NULL_PARTITION else unquote(encoded)
        keys, counted = parts.get(value, ([], 0))
        keys.append(f"{result.object_prefix}{relative}")
        meta = entry.get("meta") or {}
        if counted is not None and isinstance(meta.get("record_count"), int):
            counted += meta["record_count"]
        else:
            counted = None
        parts[value] = (keys, counted)
    return parts


async def _read_part(store: ExportStore, keys: Sequence[str], concurrency: int) -> list[dict[str, Any]]:
    """every row of one part's files, several files at once.

    :return: the rows
    :rtype: list[dict[str, Any]]
    """
    gate = asyncio.Semaphore(concurrency)

    async def one(key: str) -> list[dict[str, Any]]:
        async with gate:
            data = await _read_object(store, key)
        return await asyncio.to_thread(_parquet_rows, data)

    rows: list[dict[str, Any]] = []
    for part in await asyncio.gather(*(one(key) for key in keys)):
        rows.extend(part)
    return rows


def _moved(
    before: Mapping[str | None, RelationFingerprintResult],
    after: Mapping[str | None, RelationFingerprintResult],
) -> list[str | None]:
    """the parts whose fingerprint differs between two grouped readings (a part gone or new counts).

    :return: the parts
    :rtype: list[str | None]
    """
    return [value for value in set(before) | set(after) if before.get(value) != after.get(value)]


async def export_partitions(
    client: DatasourceQueryClient,
    store: ExportStore,
    datasource_name: str,
    *,
    relation: str,
    columns: Sequence[str],
    partition_by: str,
    destination: str,
    bucket: str,
    parts: Sequence[str] | None = None,
    file_concurrency: int = _DEFAULT_FILE_CONCURRENCY,
) -> AsyncIterator[tuple[str | None, list[dict[str, Any]]]]:
    """every part of a relation (or the ``parts`` asked for), through one export, proven, one part at a time.

    :param client: the caller's datasource client
    :ptype client: DatasourceQueryClient
    :param store: an object store over the export bucket, with the caller's credentials
    :ptype store: ExportStore
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param relation: the relation
    :ptype relation: str
    :param columns: the columns to read; every fingerprint covers all of them
    :ptype columns: Sequence[str]
    :param partition_by: the column whose values are the parts; among ``columns``
    :ptype partition_by: str
    :param destination: a fresh relative path under the datasource's export prefix
    :ptype destination: str
    :param bucket: the bucket the store reads
    :ptype bucket: str
    :param parts: the parts to read; every part the relation holds when None
    :ptype parts: Sequence[str] | None
    :param file_concurrency: one part's files read at once
    :ptype file_concurrency: int
    :return: each part's value and rows, in the parts' sorted order (NULL first); a part asked for
        that holds no rows comes back empty
    :rtype: AsyncIterator[tuple[str | None, list[dict[str, Any]]]]
    :raises ValueError: when ``partition_by`` is not among ``columns``
    :raises IncompleteExportError: when a part moved during the export, or the export does not hold
        exactly the parts' rows
    :raises ExportNotDeletedError: when the export could not be deleted after it was read
    :raises DatasourceQueryError: when the hub refuses a fingerprint or the export (an older hub
        refuses ``partition_by`` or ``group_by`` as ``MALFORMED_REQUEST``)
    """
    if partition_by not in columns:
        raise ValueError(f"the partition column {partition_by!r} must be one of the columns read")
    where_in = None if parts is None else {partition_by: list(parts)}

    async def grouped() -> dict[str | None, RelationFingerprintResult]:
        return await client.relation_fingerprint_groups(
            datasource_name, relation=relation, key=columns, group_by=partition_by, where_in=where_in
        )

    before = await grouped()
    select = export_select(relation, columns, where_in=where_in)
    result = await client.export(datasource_name, select, destination=destination, partition_by=partition_by)
    # from here every way out deletes the export, refused or read, failed or stopped: no copy of
    # warehouse rows outlives the load (delete after load, Pace's ruling)
    read = 0
    asked: list[str | None] = []
    refused: IncompleteExportError | None = None
    unwinding: BaseException | None = None
    pending: asyncio.Task[list[dict[str, Any]]] | None = None
    try:
        try:
            files = await _proven(
                client,
                store,
                result,
                before,
                grouped,
                bucket=bucket,
                destination=destination,
                relation=relation,
                partition_by=partition_by,
            )
        except IncompleteExportError as exc:
            refused = exc
            raise
        asked = sorted(set(before) | set(parts or ()), key=lambda value: (value is not None, value or ""))

        def start(value: str | None) -> asyncio.Task[list[dict[str, Any]]]:
            keys = files.get(value, ([], 0))[0]
            return asyncio.ensure_future(_read_part(store, keys, file_concurrency))

        # the next part is read while the caller works on this one
        pending = start(asked[0]) if asked else None
        for index, value in enumerate(asked):
            current, pending = pending, (start(asked[index + 1]) if index + 1 < len(asked) else None)
            rows = await current if current is not None else []
            held = before[value].row_count if value in before else 0
            if len(rows) != held:
                refused = IncompleteExportError(
                    f"{relation} {partition_by}={value}: the export's files hold {len(rows)} rows, the part holds {held}"
                )
                raise refused
            read += len(rows)
            yield value, rows
    except BaseException as exc:
        unwinding = exc
        raise
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.wait({pending})
        deleted = await _delete_after_load(
            client,
            datasource_name,
            result,
            destination=destination,
            what=relation,
            refused=refused,
            unwinding=unwinding,
            refusal_detail={
                "warehouse_rows": result.row_count,
                "parts": {str(v): [f.row_count, f.digest] for v, f in before.items()},
            },
        )
        if refused is None and unwinding is None:
            log.info(
                "partitioned export read",
                extra={
                    "extra_data": {
                        "datasource": datasource_name,
                        "relation": relation,
                        "partition_by": partition_by,
                        "parts": len(asked),
                        "rows": read,
                        "object_prefix": result.object_prefix,
                        "versions_deleted": deleted,
                    }
                },
            )


async def _proven(
    client: DatasourceQueryClient,
    store: ExportStore,
    result: DatasourceExportResult,
    before: Mapping[str | None, RelationFingerprintResult],
    grouped: Callable[[], Awaitable[dict[str | None, RelationFingerprintResult]]],
    *,
    bucket: str,
    destination: str,
    relation: str,
    partition_by: str,
) -> dict[str | None, tuple[list[str], int | None]]:
    """every proof an export must pass before its first part is handed back; its parts' files.

    :return: part -> its files and the rows the manifest counts for it
    :rtype: dict[str | None, tuple[list[str], int | None]]
    :raises IncompleteExportError: naming the proof that failed
    """
    if not result.object_prefix.endswith(f"/{destination}/"):
        raise IncompleteExportError(
            f"the hub answered an export at {result.object_prefix!r}, not the destination {destination!r} asked for"
        )
    moved = _moved(before, await grouped())
    if moved:
        raise IncompleteExportError(
            f"{relation}: {len(moved)} part(s) of {partition_by} changed while they were exported "
            f"({sorted(str(v) for v in moved)[:10]}); the export is not one state of them"
        )
    expected = sum(fingerprint.row_count for fingerprint in before.values())
    if result.row_count != expected:
        raise IncompleteExportError(
            f"{relation}: the warehouse exported {result.row_count} rows, the parts hold {expected}"
        )
    if result.bucket != bucket or result.manifest_path != f"{result.object_prefix}manifest":
        raise IncompleteExportError(
            f"the export is at s3://{result.bucket}/{result.manifest_path}, not this reader's {bucket!r} under its prefix"
        )
    files: dict[str | None, tuple[list[str], int | None]] = {}
    if result.row_count > 0:
        files = _partition_files(json.loads(await _read_object(store, result.manifest_path)), result, partition_by)
    stray = [value for value in files if value not in before]
    if stray:
        raise IncompleteExportError(f"{relation}: the export holds parts no fingerprint counted: {stray[:10]}")
    for value, (_, counted) in files.items():
        if counted is not None and counted != before[value].row_count:
            raise IncompleteExportError(
                f"{relation} {partition_by}={value}: the manifest counts {counted} rows, the part holds "
                f"{before[value].row_count}"
            )
    return files
