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

**Or proven against the caller's own counts** (:func:`export_counted_partitions`): for rows that
never change once written (a model run's output), the caller counts each part before the export and
the export is held to those counts, with no fingerprint; it runs the caller's own ``SELECT``. The
proven export (:class:`CountedPartitionedExport`) is the caller's to hold until it is done: each
part read when it is reached (:func:`read_counted_partition`) or handed to L3 as presigned files
(:func:`presigned_counted_partition`), then deleted (:func:`delete_counted_export`). A refused one
is deleted before the refusal is raised.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import unquote

from threetears.observe import get_logger

from threetears.core.backends.protocol import LoadFile

from threetears.datasources.export_read import (
    DEFAULT_PRESIGN_SECONDS,
    ExportStore,
    IncompleteExportError,
    PresigningExportStore,
    _check_export_location,
    _counted_total,
    _delete_after_load,
    _manifest_entries,
    _ManifestFile,
    _parquet_rows,
    _presign_load_files,
    _read_object,
    export_select,
)
from threetears.datasources.query_client import (
    DatasourceExportResult,
    DatasourceQueryClient,
    RelationFingerprintResult,
)

__all__ = [
    "HIVE_NULL_PARTITION",
    "CountedFile",
    "CountedPartitionedExport",
    "delete_counted_export",
    "export_counted_partitions",
    "export_partitions",
    "presigned_counted_partition",
    "read_counted_partition",
]

log = get_logger(__name__)

#: the directory the warehouse writes a NULL partition value as
HIVE_NULL_PARTITION: Final = "__HIVE_DEFAULT_PARTITION__"

#: files of one part read at once
_DEFAULT_FILE_CONCURRENCY: Final = 8


def _partition_files(
    manifest: Mapping[str, Any], result: DatasourceExportResult, partition_by: str
) -> dict[str | None, list[_ManifestFile]]:
    """each part's files, as the manifest lists them.

    :param manifest: the parsed manifest
    :ptype manifest: Mapping[str, Any]
    :param result: the export
    :ptype result: DatasourceExportResult
    :param partition_by: the column the export is divided by
    :ptype partition_by: str
    :return: part value (``None`` for NULL) -> its files, in the manifest's order
    :rtype: dict[str | None, list[_ManifestFile]]
    :raises IncompleteExportError: when an entry is not a file of one part's directory under the export
    """
    where = f"s3://{result.bucket}/{result.object_prefix}"
    parts: dict[str | None, list[_ManifestFile]] = {}
    for file in _manifest_entries(manifest, result, of="a file of a part of"):
        directory, _, file_name = file.relative.partition("/")
        column, equals, encoded = directory.partition("=")
        if not file_name or "/" in file_name or file_name in (".", "..") or column != partition_by or not equals:
            raise IncompleteExportError(
                f"the export's manifest names {file.url!r}, which is not a file of a part of {where}"
            )
        value = None if encoded == HIVE_NULL_PARTITION else unquote(encoded)
        parts.setdefault(value, []).append(file)
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
            keys = [file.object_name for file in files.get(value, [])]
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
) -> dict[str | None, list[_ManifestFile]]:
    """every proof an export must pass before its first part is handed back; its parts' files.

    :return: part -> its files
    :rtype: dict[str | None, list[_ManifestFile]]
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
    _check_export_location(result, bucket=bucket)
    files: dict[str | None, list[_ManifestFile]] = {}
    if result.row_count > 0:
        files = _partition_files(json.loads(await _read_object(store, result.manifest_path)), result, partition_by)
    stray = [value for value in files if value not in before]
    if stray:
        raise IncompleteExportError(f"{relation}: the export holds parts no fingerprint counted: {stray[:10]}")
    for value, part_files in files.items():
        counted = _counted_total(part_files)
        if counted is not None and counted != before[value].row_count:
            raise IncompleteExportError(
                f"{relation} {partition_by}={value}: the manifest counts {counted} rows, the part holds "
                f"{before[value].row_count}"
            )
    return files


@dataclass(frozen=True)
class CountedFile:
    """one file of a counted export's part: its key, and the rows the manifest counts for it.

    :ivar path: the object's key in the export bucket
    :ivar rows: the rows the manifest counts for it; None when the manifest does not count it
    """

    path: str
    rows: int | None


@dataclass(frozen=True)
class CountedPartitionedExport:
    """a partitioned export proven against the caller's counts: where it is, and each part's files and rows.

    :ivar destination: the destination it was asked at (what the hub deletes under)
    :ivar result: the hub's answer: the bucket, the object prefix, the manifest and the rows written
    :ivar partition_by: the column its files are divided by
    :ivar files: part value -> its files, in the manifest's order; every counted part with rows has some
    :ivar rows: part value -> its rows, as the caller counted them (a part counted with none left out)
    """

    destination: str
    result: DatasourceExportResult
    partition_by: str
    files: Mapping[str, tuple[CountedFile, ...]]
    rows: Mapping[str, int]


async def _counted_files(
    store: ExportStore,
    result: DatasourceExportResult,
    *,
    destination: str,
    bucket: str,
    partition_by: str,
    counts: Mapping[str, int],
) -> dict[str, tuple[CountedFile, ...]]:
    """every proof a counted export must pass before a row of it is read; each part's files.

    :return: part value -> its files
    :rtype: dict[str, tuple[CountedFile, ...]]
    :raises IncompleteExportError: naming the proof that failed
    """
    if not result.object_prefix.endswith(f"/{destination}/"):
        raise IncompleteExportError(
            f"the hub answered an export at {result.object_prefix!r}, not the destination {destination!r} asked for"
        )
    _check_export_location(result, bucket=bucket)
    counted = sum(counts.values())
    if result.row_count != counted:
        raise IncompleteExportError(f"the warehouse exported {result.row_count} rows, the counts add up to {counted}")
    files: dict[str, tuple[CountedFile, ...]] = {}
    if result.row_count > 0:
        manifest = json.loads(await _read_object(store, result.manifest_path))
        for value, part_files in _partition_files(manifest, result, partition_by).items():
            rows = _counted_total(part_files)
            # a warehouse char column's values come back padded; the caller's counts are not
            part = None if value is None else value.strip()
            if part is None or part not in counts:
                raise IncompleteExportError(f"the export holds a {partition_by} no count named: {value!r}")
            if rows is not None and rows != counts[part]:
                raise IncompleteExportError(
                    f"{partition_by}={part}: the manifest counts {rows} rows, the caller {counts[part]}"
                )
            files[part] = (
                *files.get(part, ()),
                *(CountedFile(path=file.object_name, rows=file.rows) for file in part_files),
            )
    missing = sorted(part for part in set(counts) - set(files))
    if missing:
        raise IncompleteExportError(f"the export holds no file of {partition_by} {missing[:10]}")
    return files


async def export_counted_partitions(
    client: DatasourceQueryClient,
    store: ExportStore,
    datasource_name: str,
    *,
    select: str,
    partition_by: str,
    counts: Mapping[str, int],
    destination: str,
    bucket: str,
) -> CountedPartitionedExport:
    """export the caller's ``SELECT`` divided by a column, proven against the caller's own count of each part.

    For rows that never change once written, which the caller counted part by part before asking
    (the export's counterpart to :func:`~threetears.datasources.query_client.read_counted`). Every
    check a refusal (:class:`~threetears.datasources.export_read.IncompleteExportError`) before any
    row is read:

    - the export is the one asked for: under ``destination``, in ``bucket``, its manifest its
      prefix's own;
    - the warehouse wrote as many rows as the counts add up to;
    - the manifest names no part that was not counted, and counts each part's rows as counted;
    - each counted part with rows has files.

    A part's files are held to its count again when it is read (:func:`read_counted_partition`).
    The proven export is the caller's to delete (:func:`delete_counted_export`); a refused one, or
    one the caller stopped waiting for, is deleted here before the error is raised (a failed delete
    logged at ERROR, the error raised all the same).

    :param client: the caller's datasource client (its identity asks for the export)
    :ptype client: DatasourceQueryClient
    :param store: an object store over the export bucket, with the caller's credentials
    :ptype store: ExportStore
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param select: the ``SELECT`` exported: a plain statement, its values written in as literals
    :ptype select: str
    :param partition_by: the column its files are divided by; among ``select``'s columns
    :ptype partition_by: str
    :param counts: part value -> its rows, as the caller counted them; a part with none needs no file
    :ptype counts: Mapping[str, int]
    :param destination: a fresh relative path under the datasource's export prefix
    :ptype destination: str
    :param bucket: the bucket the store reads
    :ptype bucket: str
    :return: the proven export
    :rtype: CountedPartitionedExport
    :raises IncompleteExportError: naming the proof that refused it
    :raises DatasourceQueryError: when the hub refuses the export (an older hub refuses
        ``partition_by`` as ``MALFORMED_REQUEST``; a datasource with no export location as
        ``EXPORT_UNSUPPORTED``): nothing was written
    """
    expected = {part: rows for part, rows in counts.items() if rows}
    result = await client.export(datasource_name, select, destination=destination, partition_by=partition_by)
    # from here the export exists: a refusal, or any error, deletes it before it is raised
    refused: IncompleteExportError | None = None
    try:
        try:
            files = await _counted_files(
                store, result, destination=destination, bucket=bucket, partition_by=partition_by, counts=expected
            )
        except IncompleteExportError as exc:
            refused = exc
            raise
    except BaseException as exc:
        await _delete_after_load(
            client,
            datasource_name,
            result,
            destination=destination,
            what=destination,
            refused=refused,
            unwinding=exc,
            refusal_detail={"warehouse_rows": result.row_count, "counted": sum(expected.values())},
        )
        raise
    log.info(
        "counted partitioned export proven",
        extra={
            "extra_data": {
                "datasource": datasource_name,
                "partition_by": partition_by,
                "parts": len(expected),
                "rows": result.row_count,
                "object_prefix": result.object_prefix,
            }
        },
    )
    return CountedPartitionedExport(
        destination=destination, result=result, partition_by=partition_by, files=files, rows=expected
    )


async def read_counted_partition(
    store: ExportStore,
    export: CountedPartitionedExport,
    part: str,
    *,
    file_concurrency: int = _DEFAULT_FILE_CONCURRENCY,
) -> list[dict[str, Any]]:
    """one part's rows of a counted export, its files read several at once and held to its count.

    :param store: an object store over the export bucket
    :ptype store: ExportStore
    :param export: the proven export
    :ptype export: CountedPartitionedExport
    :param part: the part's value; one not counted, or counted with none, has no rows
    :ptype part: str
    :param file_concurrency: its files read at once
    :ptype file_concurrency: int
    :return: the rows, as the files hold them (the partition column among them)
    :rtype: list[dict[str, Any]]
    :raises IncompleteExportError: when the files do not hold exactly the part's count
    """
    files = export.files.get(part, ())
    rows = await _read_part(store, [file.path for file in files], file_concurrency)
    expected = export.rows.get(part, 0)
    if len(rows) != expected:
        raise IncompleteExportError(
            f"{export.partition_by}={part}: the export's files hold {len(rows)} rows, the caller counted {expected}"
        )
    return rows


async def presigned_counted_partition(
    store: PresigningExportStore,
    export: CountedPartitionedExport,
    part: str,
    *,
    expires_in: int = DEFAULT_PRESIGN_SECONDS,
) -> list[LoadFile]:
    """one part's files of a counted export, presigned, each with its manifest count: a load's files.

    For :meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.load_files`, so the
    L3 broker streams the files and the rows never pass through the caller. The files keep the
    partition column (the export ``INCLUDE`` s it).

    :param store: an object store over the export bucket that presigns
    :ptype store: PresigningExportStore
    :param export: the proven export
    :ptype export: CountedPartitionedExport
    :param part: the part's value; one not counted, or counted with none, has no files
    :ptype part: str
    :param expires_in: how long each URL is valid, in seconds
    :ptype expires_in: int
    :return: the files, in the manifest's order
    :rtype: list[LoadFile]
    :raises IncompleteExportError: when the manifest does not count one of the files, or its counts
        are not the part's
    """
    files = export.files.get(part, ())
    uncounted = [file.path for file in files if file.rows is None]
    if uncounted:
        raise IncompleteExportError(
            f"{export.partition_by}={part}: the manifest counts no rows for {uncounted[0]!r}; a file handed "
            f"to L3 needs its count"
        )
    keyed = [(file.path, file.rows) for file in files if file.rows is not None]
    counted = sum(rows for _, rows in keyed)
    if counted != export.rows.get(part, 0):
        raise IncompleteExportError(
            f"{export.partition_by}={part}: the manifest counts {counted} rows, the caller {export.rows.get(part, 0)}"
        )
    return await _presign_load_files(store, keyed, expires_in=expires_in)


async def delete_counted_export(
    client: DatasourceQueryClient,
    datasource_name: str,
    export: CountedPartitionedExport,
    *,
    unwinding: BaseException | None = None,
) -> int | None:
    """delete a counted export once the caller is done with it: every version under its destination.

    The rule every export reader keeps (delete after load, no timer): a delete that fails is raised
    as :class:`~threetears.datasources.export_read.ExportNotDeletedError` when nothing else is; when
    the caller is already unwinding an error, that error wins and the failed delete is logged at ERROR.

    :param client: the caller's datasource client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param export: the export
    :ptype export: CountedPartitionedExport
    :param unwinding: the error the caller is unwinding, if one is
    :ptype unwinding: BaseException | None
    :return: the versions deleted; None when the delete failed under an unwinding error
    :rtype: int | None
    :raises ExportNotDeletedError: when the delete failed and nothing else is unwinding
    """
    return await _delete_after_load(
        client,
        datasource_name,
        export.result,
        destination=export.destination,
        what=export.destination,
        refused=None,
        unwinding=unwinding,
        refusal_detail={},
    )
