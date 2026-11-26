"""Reading a part of a relation through a warehouse export: proven whole before a row is handed back.

**The read.** :func:`export_part` fingerprints the part, asks the hub to export it
(:meth:`~threetears.datasources.query_client.DatasourceQueryClient.export`), fingerprints it again,
and reads the export's parquet files from the bucket through an object store the caller built (with
its own credentials: keys from its environment, or its role). It is the export's counterpart to
:func:`~threetears.datasources.query_client.read_all` and returns the same thing, the part's rows as
dicts, so a caller can take either path.

**Proven before it is returned**, every check a refusal (:class:`IncompleteExportError`), never a
warning:

- the export is the one asked for: its files sit under the destination this call chose, and its
  manifest is that prefix's own;
- the part did not move while it was exported: its fingerprint after equals its fingerprint before,
  over the same columns, so the files are one state of the relation and not a mix of two;
- the warehouse wrote as many rows as the fingerprint counts;
- the files are the ones the manifest lists, every one under the export's own prefix in the bucket
  the caller expects (a manifest naming any other location is refused, not followed);
- the files hold exactly that many rows (and, where the manifest counts each file, as many as it
  says).

**Then it is deleted, always** (Pace's ruling, 2026-10-09: delete after load, no timer, no lifecycle
rule): from the moment the export is asked, every way out -- the part proven and read, a
proof refusing it, a read failing, the ask failing or the caller cancelled while it was in flight (the
hub's answer is still waited for, so the delete comes after the last file) -- asks the hub to delete
every version under the destination
(``export_delete``), with a delete-only grant the reader does not hold, so no copy of warehouse rows
outlives the load. A refused export is logged at WARNING with what was refused (the proof and the
counts) so an operator can tell why without the files. A delete that fails is raised
(:class:`ExportNotDeletedError`) when nothing else is; when a refusal or another error is already
unwinding, that error wins and the failed delete is logged at ERROR.

A digest over the parquet itself is not compared: the fingerprint is the warehouse's own hash,
spelled in its dialect, and is opaque to everything else. The before-and-after pair is what proves
the relation held still, and the counts prove nothing was lost on the way.

**Values arrive typed** (a decimal as :class:`~decimal.Decimal`, a timestamp as
:class:`~datetime.datetime`), where the read rail carries JSON values; a caller coercing rows to its
columns takes both.

**Or handed to L3 as files**, for a counted partitioned export
(:func:`~threetears.datasources.partitioned_export.presigned_counted_partition`): the files
presigned, each with its own row count, for the L3 broker to stream and ``COPY`` itself
(:meth:`~threetears.core.collections.schema_backed.SchemaBackedCollection.load_files`), so the rows
never pass through the pod.

Needs ``pyarrow`` (the ``export`` extra), imported only when a file is read.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from threetears.core.backends.protocol import LOAD_FILES_MAX_WAIT_SECONDS, LoadFile
from threetears.core.keyed import Selected
from threetears.observe import get_logger

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _PARQUET = True
except ImportError:  # NOSILENT: the export extra is optional; reading an export's files says so without it
    _PARQUET = False

from threetears.datasources.export import sql_string_literal
from threetears.datasources.query_client import (
    _ACCESS_DENIED,
    _MALFORMED_REQUEST,
    DATASOURCE_BUSY,
    EXPORT_NOT_GRANTED,
    EXPORT_REFUSED,
    EXPORT_UNSUPPORTED,
    DatasourceExportResult,
    DatasourceQueryClient,
    DatasourceQueryError,
    RelationFingerprintRequest,
    plain_columns,
    select_list,
)

__all__ = [
    "DEFAULT_PRESIGN_SECONDS",
    "ExportNotDeletedError",
    "ExportStore",
    "IncompleteExportError",
    "PresigningExportStore",
    "export_part",
    "export_part_arrow",
    "export_select",
    "read_export",
]

#: how long a presigned export file stays fetchable by default, in seconds: longer than the most a pod
#: waits for one request (:data:`~threetears.core.backends.protocol.LOAD_FILES_MAX_WAIT_SECONDS`), with
#: a quarter of an hour over for the request to be sent. A load sent as several requests reaches its
#: later files after the earlier requests' actual time, minutes at a binary COPY's pace; a file that
#: expired first fails its request, and the load drops and advances as for any failure. Presign
#: each load's files just before it is sent.
DEFAULT_PRESIGN_SECONDS = LOAD_FILES_MAX_WAIT_SECONDS + 900

log = get_logger(__name__)

#: the hub's answers to an export ask that it gives before the warehouse runs anything, so nothing sits
#: under the destination and there is nothing to delete: not granted, not exportable, a ``SELECT`` or
#: destination refused (or one overlapping an export not yet deleted), busy (the hub released its claim
#: unrun), a request it could not read, a caller with no read. Every other end of the ask -- a transport
#: failure or timeout, a warehouse error midway (an ``UNLOAD`` that failed can leave some of its files),
#: an unknown code, a cancel while it was in flight -- may have left files, and is deleted
_REFUSED_BEFORE_WRITING: Final = frozenset(
    {
        EXPORT_NOT_GRANTED,
        EXPORT_UNSUPPORTED,
        EXPORT_REFUSED,
        DATASOURCE_BUSY,
        _MALFORMED_REQUEST,
        _ACCESS_DENIED,
        "INVALID_DATASOURCE_NAME",
    }
)


class IncompleteExportError(RuntimeError):
    """an export could not be shown to hold exactly the part's rows, at one moment; nothing is returned."""


class ExportNotDeletedError(RuntimeError):
    """a proven export could not be deleted; its files are still in the bucket, under the prefix named.

    Raised, failing the part although its rows were read and proven, on purpose: the intent is that
    no copy of warehouse rows outlives the load that read it. Logging and carrying on would load the
    part and leave the copy for nobody to own; failing makes the leftover visible, and the next
    refresh reads the part again through a fresh export and deletes that one.
    """


class ExportStore(Protocol):
    """what reading an export needs from an object store: one object's bytes, streamed.

    :class:`threetears.object_store.S3ObjectStore` is one, built over the export bucket.
    """

    def open_read(self, key: str) -> AsyncIterator[bytes]:
        """stream the object at ``key``.

        :param key: the object's key
        :ptype key: str
        :return: its bytes, in chunks
        :rtype: AsyncIterator[bytes]
        """
        ...


class PresigningExportStore(ExportStore, Protocol):
    """an export store that also presigns: what handing an export to L3 as files needs.

    :class:`threetears.object_store.S3ObjectStore` is one; a
    :class:`~threetears.object_store.FilesystemObjectStore` presigns as a ``file://`` URL.
    """

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        """a GET URL for the object at ``key`` that needs no credential, valid ``expires_in`` seconds.

        :param key: the object's key
        :ptype key: str
        :param expires_in: how long the URL is valid, in seconds
        :ptype expires_in: int
        :return: the URL
        :rtype: str
        """
        ...


def export_select(
    relation: str,
    columns: Sequence[str | Selected],
    where: Mapping[str, str] | None = None,
    where_in: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """the ``SELECT`` an export of one part runs: the columns, the relation, equality filters as literals.

    An export binds no parameters (the warehouse's ``UNLOAD`` cannot), so each filter value is
    written in as a quoted literal; the relation and every column must be plain identifiers, checked
    by the same grammar a fingerprint request is.

    :param relation: the relation, ``name`` or ``schema.name``
    :ptype relation: str
    :param columns: the columns to export, in order: plain identifiers, or
        :class:`~threetears.core.keyed.Selected` expressions under their names (a row's hash, ``alias=None``)
    :ptype columns: Sequence[str | Selected]
    :param where: column -> value; every row when None
    :ptype where: Mapping[str, str] | None
    :param where_in: column -> the values it may hold, each written in as a literal
    :ptype where_in: Mapping[str, Sequence[str]] | None
    :return: the statement
    :rtype: str
    :raises ValueError: when a name is not a plain identifier
    :raises ExportRefusedError: when a value cannot be written as a literal
    """
    checked = RelationFingerprintRequest(relation=relation, key_columns=plain_columns(columns), where=dict(where or {}))
    sets = {column: list(values) for column, values in (where_in or {}).items()}
    if sets:
        # the set filters' columns, checked by the same grammar as the key's
        RelationFingerprintRequest(relation=relation, key_columns=list(sets))
    # the plain columns as the model checked them; a Selected expression is the caller's TRUSTED SQL
    statement = f"SELECT {select_list(columns)} FROM {checked.relation}"  # noqa: S608 - identifiers checked
    # the values as given (text), the columns as the model checked them
    conditions = [f"{column} = {sql_string_literal(value)}" for column, value in dict(where or {}).items()]
    for column, values in sets.items():
        members = ", ".join(sql_string_literal(value) for value in values)
        conditions.append(f"{column} IN ({members})" if values else "1 = 0")
    if conditions:
        statement += f" WHERE {' AND '.join(conditions)}"
    return statement


async def _read_object(store: ExportStore, key: str) -> bytes:
    """the whole object at ``key``.

    :param store: the store
    :ptype store: ExportStore
    :param key: the key
    :ptype key: str
    :return: its bytes
    :rtype: bytes
    """
    chunks = [chunk async for chunk in store.open_read(key)]
    return b"".join(chunks)


@dataclass(frozen=True)
class _ManifestFile:
    """one file a manifest lists, under the export's prefix.

    :ivar url: the entry's ``url``, as the manifest names it
    :ivar relative: its path under the export's prefix
    :ivar object_name: its object's name in the export bucket (its key there)
    :ivar rows: the rows the manifest counts for it; None when it counts none
    """

    url: str
    relative: str
    object_name: str
    rows: int | None


def _manifest_entries(manifest: Mapping[str, Any], result: DatasourceExportResult, *, of: str) -> list[_ManifestFile]:
    """every file a manifest lists, each checked to sit under the export's prefix: the one reading of an entry.

    :param manifest: the parsed manifest
    :ptype manifest: Mapping[str, Any]
    :param result: the export
    :ptype result: DatasourceExportResult
    :param of: what a file must be a file of, for the refusal (``"a file of"``, ``"a file of a part of"``)
    :ptype of: str
    :return: the files, in the manifest's order
    :rtype: list[_ManifestFile]
    :raises IncompleteExportError: when an entry names anything outside the export
    """
    where = f"s3://{result.bucket}/{result.object_prefix}"
    files: list[_ManifestFile] = []
    for entry in manifest.get("entries", []):
        url = str(entry.get("url", ""))
        relative = url.removeprefix(where)
        if relative == url or not relative:
            raise IncompleteExportError(f"the export's manifest names {url!r}, which is not {of} {where}")
        rows = (entry.get("meta") or {}).get("record_count")
        counted = rows if isinstance(rows, int) and not isinstance(rows, bool) and rows >= 0 else None
        files.append(
            _ManifestFile(url=url, relative=relative, object_name=f"{result.object_prefix}{relative}", rows=counted)
        )
    return files


def _manifest_files(manifest: Mapping[str, Any], result: DatasourceExportResult) -> list[_ManifestFile]:
    """the files of an unpartitioned export's manifest, each directly under the export's prefix.

    :param manifest: the parsed manifest
    :ptype manifest: Mapping[str, Any]
    :param result: the export
    :ptype result: DatasourceExportResult
    :return: the files, in the manifest's order
    :rtype: list[_ManifestFile]
    :raises IncompleteExportError: when an entry names anything outside the export
    """
    where = f"s3://{result.bucket}/{result.object_prefix}"
    files = _manifest_entries(manifest, result, of="a file of")
    for file in files:
        if "/" in file.relative or file.relative in (".", ".."):
            raise IncompleteExportError(f"the export's manifest names {file.url!r}, which is not a file of {where}")
    return files


def _counted_total(files: Sequence[_ManifestFile]) -> int | None:
    """the rows a manifest counts for ``files``; None when it does not count every one.

    :param files: the files
    :ptype files: Sequence[_ManifestFile]
    :return: the total, or None
    :rtype: int | None
    """
    counts = [file.rows for file in files]
    return None if any(rows is None for rows in counts) else sum(rows for rows in counts if rows is not None)


def _parquet_rows(data: bytes) -> list[dict[str, Any]]:
    """the rows of one parquet file.

    :param data: the file
    :ptype data: bytes
    :return: its rows
    :rtype: list[dict[str, Any]]
    """
    if not _PARQUET:
        raise ImportError("reading an export's parquet files needs pyarrow: install 3tears-datasources[export]")
    rows: list[dict[str, Any]] = _read_parquet(data).to_pylist()
    return rows


def _parquet_table(data: bytes) -> pa.Table:
    """one parquet file as an Arrow table, its rows never made Python objects.

    :param data: the file
    :ptype data: bytes
    :return: its rows
    :rtype: pyarrow.Table
    """
    if not _PARQUET:
        raise ImportError("reading an export's parquet files needs pyarrow: install 3tears-datasources[export]")
    return _read_parquet(data)


def _read_parquet(data: bytes) -> pa.Table:
    """one parquet file's rows, read natively over its bytes: blocking, on a worker thread.

    Over ``pa.BufferReader`` the reader never calls back into Python. Over ``io.BytesIO`` (and through
    ``pq.read_table``'s dataset machinery) every read of the file went back through a Python file object, taking the
    interpreter lock each time: six threads reading an export's files at once held the event loop 0.1 s
    (cobalt prod, 2026-10-31).

    :param data: the file
    :ptype data: bytes
    :return: its rows
    :rtype: pyarrow.Table
    """
    table: pa.Table = pq.ParquetFile(pa.BufferReader(data)).read()
    return table


async def read_export(store: ExportStore, result: DatasourceExportResult, *, bucket: str) -> list[dict[str, Any]]:
    """every row of an export, read from the files its manifest lists, counted against the warehouse's count.

    :param store: an object store over the export bucket
    :ptype store: ExportStore
    :param result: the export
    :ptype result: DatasourceExportResult
    :param bucket: the bucket the caller reads; an export in any other is refused
    :ptype bucket: str
    :return: the rows
    :rtype: list[dict[str, Any]]
    :raises IncompleteExportError: when the export is elsewhere, its manifest names a file outside it,
        or its files hold a different number of rows than the warehouse wrote
    """
    rows: list[dict[str, Any]] = []
    for data in await _export_files(store, result, bucket=bucket):
        rows.extend(_parquet_rows(data))
    _held_to_count(len(rows), result)
    return rows


async def _read_export_arrow(store: ExportStore, result: DatasourceExportResult, *, bucket: str) -> pa.Table:
    """:func:`read_export`, its rows one Arrow table as the files hold them (each file read on a thread), never one
    Python object a row: for a caller that hands the rows on (a worker process) rather than reads them.

    :param store: an object store over the export bucket
    :ptype store: ExportStore
    :param result: the export
    :ptype result: DatasourceExportResult
    :param bucket: the bucket the caller reads; an export in any other is refused
    :ptype bucket: str
    :return: the rows; a table of no columns when the export wrote none
    :rtype: pyarrow.Table
    :raises IncompleteExportError: as :func:`read_export`
    """
    tables = [
        await asyncio.to_thread(_parquet_table, data) for data in await _export_files(store, result, bucket=bucket)
    ]
    # each file's own types, widened where two files differ (a decimal's precision is each file's own): no value moves
    table = pa.concat_tables(tables, promote_options="permissive") if tables else pa.table({})
    _held_to_count(table.num_rows, result)
    return table


async def _export_files(store: ExportStore, result: DatasourceExportResult, *, bucket: str) -> list[bytes]:
    """the bytes of every file an export's manifest lists, the manifest's count held to the warehouse's.

    :param store: an object store over the export bucket
    :ptype store: ExportStore
    :param result: the export
    :ptype result: DatasourceExportResult
    :param bucket: the bucket the caller reads
    :ptype bucket: str
    :return: each file's bytes, in the manifest's order
    :rtype: list[bytes]
    :raises IncompleteExportError: when the export is elsewhere, its manifest names a file outside it, or counts
        other rows than the warehouse wrote
    """
    _check_export_location(result, bucket=bucket)
    files: list[bytes] = []
    if result.row_count > 0:
        manifest = json.loads(await _read_object(store, result.manifest_path))
        listed = _manifest_files(manifest, result)
        counted = _counted_total(listed)
        if counted is not None and counted != result.row_count:
            raise IncompleteExportError(
                f"the export's manifest counts {counted} rows, the warehouse says it wrote {result.row_count}"
            )
        for file in listed:
            files.append(await _read_object(store, file.object_name))
    return files


def _held_to_count(rows: int, result: DatasourceExportResult) -> None:
    """refuse an export whose files hold other than the rows the warehouse says it wrote.

    :param rows: the rows the files hold
    :ptype rows: int
    :param result: the export
    :ptype result: DatasourceExportResult
    :return: nothing
    :rtype: None
    :raises IncompleteExportError: naming both counts
    """
    if rows != result.row_count:
        raise IncompleteExportError(
            f"the export's files hold {rows} rows, the warehouse says it wrote {result.row_count}"
        )


def _check_export_location(result: DatasourceExportResult, *, bucket: str) -> None:
    """refuse an export outside the reader's bucket, or whose manifest is not its prefix's own.

    :param result: the export
    :ptype result: DatasourceExportResult
    :param bucket: the bucket the caller reads
    :ptype bucket: str
    :return: nothing
    :rtype: None
    :raises IncompleteExportError: when either is so
    """
    if result.bucket != bucket:
        raise IncompleteExportError(f"the export is in bucket {result.bucket!r}, not this reader's {bucket!r}")
    if result.manifest_path != f"{result.object_prefix}manifest":
        raise IncompleteExportError(
            f"the export's manifest {result.manifest_path!r} is not this reader's: it is not its prefix "
            f"{result.object_prefix!r}'s own"
        )


async def _presign_load_files(
    store: PresigningExportStore,
    keyed: Sequence[tuple[str, int]],
    *,
    expires_in: int,
) -> list[LoadFile]:
    """each ``(key, rows)`` as a :class:`LoadFile`, its key presigned.

    :param store: the store over the export bucket
    :ptype store: PresigningExportStore
    :param keyed: each file's key and rows, in order
    :ptype keyed: Sequence[tuple[str, int]]
    :param expires_in: how long each URL is valid, in seconds
    :ptype expires_in: int
    :return: the files, in order
    :rtype: list[LoadFile]
    """
    return [LoadFile(url=await store.presigned_get_url(key, expires_in=expires_in), rows=rows) for key, rows in keyed]


async def export_part(
    client: DatasourceQueryClient,
    store: ExportStore,
    datasource_name: str,
    *,
    relation: str,
    columns: Sequence[str | Selected],
    destination: str,
    bucket: str,
    where: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """one part of a relation, read through an export and proven whole and unmoved (:func:`_export_part`).

    :return: the part's rows
    :rtype: list[dict[str, Any]]
    """
    rows: list[dict[str, Any]] = await _export_part(
        client,
        store,
        datasource_name,
        relation=relation,
        columns=columns,
        destination=destination,
        bucket=bucket,
        where=where,
        read=read_export,
    )
    return rows


async def export_part_arrow(
    client: DatasourceQueryClient,
    store: ExportStore,
    datasource_name: str,
    *,
    relation: str,
    columns: Sequence[str | Selected],
    destination: str,
    bucket: str,
    where: Mapping[str, str] | None = None,
) -> pa.Table:
    """:func:`export_part`, its rows one Arrow table as the export's files hold them (:func:`_read_export_arrow`): for a
    caller that hands the rows on to a worker process rather than reads them.

    :return: the part's rows
    :rtype: pyarrow.Table
    """
    table: pa.Table = await _export_part(
        client,
        store,
        datasource_name,
        relation=relation,
        columns=columns,
        destination=destination,
        bucket=bucket,
        where=where,
        read=_read_export_arrow,
    )
    return table


async def _export_part(
    client: DatasourceQueryClient,
    store: ExportStore,
    datasource_name: str,
    *,
    relation: str,
    columns: Sequence[str | Selected],
    destination: str,
    bucket: str,
    where: Mapping[str, str] | None = None,
    read: Callable[..., Awaitable[Any]] = read_export,
) -> Any:
    """one part of a relation, read through an export and proven whole and unmoved.

    :param client: the caller's datasource client (its identity asks for the export)
    :ptype client: DatasourceQueryClient
    :param store: an object store over the export bucket, with the caller's credentials
    :ptype store: ExportStore
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param relation: the relation
    :ptype relation: str
    :param columns: the columns to read; the fingerprint covers every one read as written (a
        :class:`~threetears.core.keyed.Selected` expression is computed from them)
    :ptype columns: Sequence[str | Selected]
    :param destination: a fresh relative path under the datasource's export prefix
    :ptype destination: str
    :param bucket: the bucket the store reads
    :ptype bucket: str
    :param where: the part, as equality filters; the whole relation when None
    :ptype where: Mapping[str, str] | None
    :param read: how the export's files are read: :func:`read_export` (dicts) or :func:`_read_export_arrow`
    :ptype read: Callable[..., Awaitable[Any]]
    :return: the part's rows, as ``read`` answers them
    :rtype: Any
    :raises IncompleteExportError: when the part moved during the export, the export is not the one
        asked for, or it does not hold exactly its rows
    :raises ExportNotDeletedError: when the proven export could not be deleted
    :raises DatasourceQueryError: when the hub refuses the fingerprint or the export
    """
    select = export_select(relation, columns, where)
    filters = dict(where or {})
    before = await client.relation_fingerprint(
        datasource_name, relation=relation, key=plain_columns(columns), where=filters
    )
    what = f"{relation} {filters or ''}".rstrip()
    # an ask that ends in no answer (a cancel while it was in flight among them) is deleted before it is raised
    result = await _asked_export(
        client,
        datasource_name,
        client.export(datasource_name, select, destination=destination),
        destination=destination,
        what=what,
    )
    # from here every way out deletes the export, refused or read: no copy of warehouse rows
    # outlives the load (delete after load, Pace's ruling)
    rows: Any = []
    refused: IncompleteExportError | None = None
    unwinding: BaseException | None = None
    try:
        try:
            if not result.object_prefix.endswith(f"/{destination}/"):
                raise IncompleteExportError(
                    f"the hub answered an export at {result.object_prefix!r}, not the destination {destination!r} "
                    "asked for"
                )
            after = await client.relation_fingerprint(
                datasource_name, relation=relation, key=plain_columns(columns), where=filters
            )
            if after != before:
                raise IncompleteExportError(
                    f"{relation} {filters or ''} changed while it was exported ({before.row_count} rows before, "
                    f"{after.row_count} after); the export is not one state of it"
                )
            if result.row_count != before.row_count:
                raise IncompleteExportError(
                    f"{relation} {filters or ''}: the warehouse exported {result.row_count} rows, the relation "
                    f"holds {before.row_count}"
                )
            rows = await read(store, result, bucket=bucket)
        except IncompleteExportError as exc:
            refused = exc
            raise
    except BaseException as exc:
        unwinding = exc
        raise
    finally:
        deleted = await _delete_after_load(
            client,
            datasource_name,
            result,
            destination=destination,
            what=what,
            refused=refused,
            unwinding=unwinding,
            refusal_detail={"warehouse_rows": result.row_count, "fingerprint": [before.row_count, before.digest]},
        )
    log.info(
        "export read",
        extra={
            "extra_data": {
                "datasource": datasource_name,
                "relation": relation,
                "where": filters,
                "rows": len(rows),
                "object_prefix": result.object_prefix,
                "versions_deleted": deleted,
            }
        },
    )
    return rows


async def _to_its_end[T](work: Awaitable[T]) -> T:
    """run ``work`` to its end even when the caller is cancelled, and only then let the cancel through.

    A bare ``asyncio.shield`` lets the cancel out at once while the shielded work goes on: the caller
    unwinds (a pod's ``close`` returns, its loop ends) before the work is done. Here the cancel waits
    for it, a second cancel too, and is raised once it has finished; a failure of the work under a
    cancel is read (the work logs its own), the cancel is what is raised.

    :param work: what must not be cut short
    :ptype work: Awaitable[T]
    :return: its result
    :rtype: T
    """
    task = asyncio.ensure_future(work)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                # NOSILENT: cancelled again while the work finishes; the cancel is raised once it has
                continue
        if not task.cancelled():
            task.exception()  # read, so a failure under a cancel isn't reported as never retrieved
        raise


async def _asked_export(
    client: DatasourceQueryClient,
    datasource_name: str,
    ask: Awaitable[DatasourceExportResult],
    *,
    destination: str,
    what: str,
) -> DatasourceExportResult:
    """the hub's answer to an export ask, and the export deleted when the ask does not end in one.

    The export's location is known before it is asked: the hub writes it under ``destination``, and
    deletes it by that destination alone, so an ask that ends in anything but an answer -- the caller
    cancelled while it was in flight, a transport failure or a timeout, a warehouse error midway -- is
    deleted there too, unless the hub refused it before anything ran (:data:`_REFUSED_BEFORE_WRITING`).

    A cancel does not stop the hub: the ``UNLOAD`` goes on writing after the caller stops waiting, and
    a delete sent meanwhile would delete what was there so far and the hub's record of it, leaving the
    files written after it with nothing to find them. So under a cancel the hub's answer is still
    waited for (the client's export deadline outlasts the hub's statement deadline, so a timeout too
    means the ``UNLOAD`` is over), then the export deleted, then the cancel raised.

    :param client: the caller's datasource client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param ask: the export ask (``client.export(...)``), not yet awaited
    :ptype ask: Awaitable[DatasourceExportResult]
    :param destination: the destination it is asked at (what the hub deletes under)
    :ptype destination: str
    :param what: what is read, for the messages
    :ptype what: str
    :return: the hub's answer
    :rtype: DatasourceExportResult
    :raises asyncio.CancelledError: when the caller was cancelled, once the export is deleted
    :raises DatasourceQueryError: when the hub refused the export, or the ask failed (deleted first)
    """
    asking = asyncio.ensure_future(ask)
    stopped: asyncio.CancelledError | None = None
    while not asking.done():
        try:
            await asyncio.wait({asking})
        except asyncio.CancelledError as exc:
            # NOSILENT: the caller stopped waiting while the hub may still be writing; the cancel is
            # raised once the hub answered and the export was deleted
            stopped = stopped or exc
    if asking.cancelled():
        stopped = stopped or asyncio.CancelledError()
    failure = None if asking.cancelled() else asking.exception()
    handed_back = failure is None and stopped is None
    nothing_written = isinstance(failure, DatasourceQueryError) and failure.error_code in _REFUSED_BEFORE_WRITING
    if not handed_back and not nothing_written:
        await _delete_after_load(
            client,
            datasource_name,
            asking.result() if failure is None and not asking.cancelled() else None,
            destination=destination,
            what=what,
            refused=None,
            unwinding=stopped or failure,
            refusal_detail={},
        )
    if stopped is not None:
        raise stopped
    if failure is not None:
        raise failure
    return asking.result()


async def _delete_after_load(
    client: DatasourceQueryClient,
    datasource_name: str,
    result: DatasourceExportResult | None,
    *,
    destination: str,
    what: str,
    refused: IncompleteExportError | None,
    unwinding: BaseException | None,
    refusal_detail: Mapping[str, Any],
) -> int | None:
    """delete an export on its way out, whatever the way: the one rule every export reader keeps.

    A refused export is logged at WARNING with its proof first, then deleted all the same. A delete
    that fails is raised as :class:`ExportNotDeletedError` when nothing is unwinding; when an error
    already is (a refusal among them), that error wins and the failed delete is logged at ERROR.

    The delete runs to its end even when the caller is cancelled meanwhile (:func:`_to_its_end`): a
    cancel landing on the delete would otherwise leave the export it was deleting. The cancel is
    raised once the delete is done.

    :param client: the caller's datasource client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param result: the export the hub answered; None when the ask ended without an answer
    :ptype result: DatasourceExportResult | None
    :param destination: the destination the export was asked at (what the hub deletes under)
    :ptype destination: str
    :param what: what was read, for the messages
    :ptype what: str
    :param refused: the proof that refused the export, if one did
    :ptype refused: IncompleteExportError | None
    :param unwinding: the error leaving the read, if one is
    :ptype unwinding: BaseException | None
    :param refusal_detail: what the refusal is logged with besides its proof (counts, fingerprints)
    :ptype refusal_detail: Mapping[str, Any]
    :return: the versions deleted; None when the delete failed under an unwinding error
    :rtype: int | None
    :raises ExportNotDeletedError: when the delete failed and nothing else is unwinding
    :raises asyncio.CancelledError: when the caller was cancelled during the delete, once it is done
    """
    object_prefix = None if result is None else result.object_prefix
    where = (
        f"under its destination {destination!r}"
        if result is None
        else f"at s3://{result.bucket}/{result.object_prefix}"
    )
    if refused is not None:
        log.warning(
            "an export was refused; it is deleted all the same",
            extra={
                "extra_data": {
                    "datasource": datasource_name,
                    "read": what,
                    "object_prefix": object_prefix,
                    "destination": destination,
                    "proof": str(refused),
                    **refusal_detail,
                }
            },
        )

    async def delete() -> int | None:
        try:
            return await client.delete_export(datasource_name, destination=destination)
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- raised as the named failure when nothing else is; logged when an error already unwinds, which wins
            log.error(
                "an export could not be deleted; its rows are still in the bucket",
                extra={
                    "extra_data": {
                        "datasource": datasource_name,
                        "object_prefix": object_prefix,
                        "destination": destination,
                        "error": f"{type(exc).__name__}: {exc}",
                        "unwinding": None if unwinding is None else type(unwinding).__name__,
                    }
                },
            )
            if unwinding is None:
                raise ExportNotDeletedError(
                    f"{what}: the export was read, but it could not be deleted and is still {where}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            return None

    return await _to_its_end(delete())
