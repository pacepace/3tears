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

**Then it is deleted** (Pace's ruling, 2026-10-09: delete after load, no timer, no lifecycle rule):
once the part is proven and its rows are in hand, the hub is asked to delete every version under
the destination (``export_delete``), with a delete-only grant the reader does not hold. A delete
that fails is raised (:class:`ExportNotDeletedError`) rather than left as a stray copy of warehouse
rows nobody owns; a refused export is not deleted, so an operator can look at it.

A digest over the parquet itself is not compared: the fingerprint is the warehouse's own hash,
spelled in its dialect, and is opaque to everything else. The before-and-after pair is what proves
the relation held still, and the counts prove nothing was lost on the way.

**Values arrive typed** (a decimal as :class:`~decimal.Decimal`, a timestamp as
:class:`~datetime.datetime`), where the read rail carries JSON values; a caller coercing rows to its
columns takes both.

Needs ``pyarrow`` (the ``export`` extra), imported only when a file is read.
"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

from threetears.observe import get_logger

from threetears.datasources.export import sql_string_literal
from threetears.datasources.query_client import (
    DatasourceExportResult,
    DatasourceQueryClient,
    RelationFingerprintRequest,
)

__all__ = [
    "ExportNotDeletedError",
    "ExportStore",
    "IncompleteExportError",
    "export_part",
    "export_select",
    "read_export",
]

log = get_logger(__name__)


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


def export_select(
    relation: str,
    columns: Sequence[str],
    where: Mapping[str, str] | None = None,
    where_in: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """the ``SELECT`` an export of one part runs: the columns, the relation, equality filters as literals.

    An export binds no parameters (the warehouse's ``UNLOAD`` cannot), so each filter value is
    written in as a quoted literal; the relation and every column must be plain identifiers, checked
    by the same grammar a fingerprint request is.

    :param relation: the relation, ``name`` or ``schema.name``
    :ptype relation: str
    :param columns: the columns to export, in order
    :ptype columns: Sequence[str]
    :param where: column -> value; every row when None
    :ptype where: Mapping[str, str] | None
    :param where_in: column -> the values it may hold, each written in as a literal
    :ptype where_in: Mapping[str, Sequence[str]] | None
    :return: the statement
    :rtype: str
    :raises ValueError: when a name is not a plain identifier
    :raises ExportRefusedError: when a value cannot be written as a literal
    """
    checked = RelationFingerprintRequest(
        relation=relation,
        key_columns=list(columns),
        where=dict(where or {}),
        where_in={column: list(values) for column, values in (where_in or {}).items()},
    )
    statement = f"SELECT {', '.join(checked.key_columns)} FROM {checked.relation}"  # noqa: S608 - identifiers checked
    conditions = [f"{column} = {sql_string_literal(value)}" for column, value in checked.where.items()]
    for column, values in checked.where_in.items():
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


def _manifest_files(manifest: Mapping[str, Any], result: DatasourceExportResult) -> tuple[list[str], int | None]:
    """the keys a manifest lists, each checked to sit under the export's prefix, and its row total.

    :param manifest: the parsed manifest
    :ptype manifest: Mapping[str, Any]
    :param result: the export
    :ptype result: DatasourceExportResult
    :return: the keys, and the rows the manifest counts (None when it counts none)
    :rtype: tuple[list[str], int | None]
    :raises IncompleteExportError: when an entry names anything outside the export
    """
    where = f"s3://{result.bucket}/{result.object_prefix}"
    keys: list[str] = []
    counted: int | None = 0
    for entry in manifest.get("entries", []):
        url = str(entry.get("url", ""))
        relative = url.removeprefix(where)
        if relative == url or not relative or "/" in relative or relative in (".", ".."):
            raise IncompleteExportError(f"the export's manifest names {url!r}, which is not a file of {where}")
        keys.append(f"{result.object_prefix}{relative}")
        meta = entry.get("meta") or {}
        if counted is not None and isinstance(meta.get("record_count"), int):
            counted += meta["record_count"]
        else:
            counted = None
    return keys, counted


def _parquet_rows(data: bytes) -> list[dict[str, Any]]:
    """the rows of one parquet file.

    :param data: the file
    :ptype data: bytes
    :return: its rows
    :rtype: list[dict[str, Any]]
    """
    import pyarrow.parquet as pq  # noqa: PLC0415 - the export extra, needed only once a file is read

    rows: list[dict[str, Any]] = pq.read_table(io.BytesIO(data)).to_pylist()
    return rows


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
    if result.bucket != bucket:
        raise IncompleteExportError(f"the export is in bucket {result.bucket!r}, not the {bucket!r} this reader reads")
    if result.manifest_path != f"{result.object_prefix}manifest":
        raise IncompleteExportError(
            f"the export's manifest {result.manifest_path!r} is not its prefix {result.object_prefix!r}'s own"
        )
    rows: list[dict[str, Any]] = []
    if result.row_count > 0:
        manifest = json.loads(await _read_object(store, result.manifest_path))
        keys, counted = _manifest_files(manifest, result)
        if counted is not None and counted != result.row_count:
            raise IncompleteExportError(
                f"the export's manifest counts {counted} rows, the warehouse says it wrote {result.row_count}"
            )
        for key in keys:
            rows.extend(_parquet_rows(await _read_object(store, key)))
    if len(rows) != result.row_count:
        raise IncompleteExportError(
            f"the export's files hold {len(rows)} rows, the warehouse says it wrote {result.row_count}"
        )
    return rows


async def export_part(
    client: DatasourceQueryClient,
    store: ExportStore,
    datasource_name: str,
    *,
    relation: str,
    columns: Sequence[str],
    destination: str,
    bucket: str,
    where: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """one part of a relation, read through an export and proven whole and unmoved.

    :param client: the caller's datasource client (its identity asks for the export)
    :ptype client: DatasourceQueryClient
    :param store: an object store over the export bucket, with the caller's credentials
    :ptype store: ExportStore
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param relation: the relation
    :ptype relation: str
    :param columns: the columns to read; the fingerprint covers all of them
    :ptype columns: Sequence[str]
    :param destination: a fresh relative path under the datasource's export prefix
    :ptype destination: str
    :param bucket: the bucket the store reads
    :ptype bucket: str
    :param where: the part, as equality filters; the whole relation when None
    :ptype where: Mapping[str, str] | None
    :return: the part's rows
    :rtype: list[dict[str, Any]]
    :raises IncompleteExportError: when the part moved during the export, the export is not the one
        asked for, or it does not hold exactly its rows
    :raises ExportNotDeletedError: when the proven export could not be deleted
    :raises DatasourceQueryError: when the hub refuses the fingerprint or the export
    """
    select = export_select(relation, columns, where)
    filters = dict(where or {})
    before = await client.relation_fingerprint(datasource_name, relation=relation, key=columns, where=filters)
    result = await client.export(datasource_name, select, destination=destination)
    if not result.object_prefix.endswith(f"/{destination}/"):
        raise IncompleteExportError(
            f"the hub answered an export at {result.object_prefix!r}, not the destination {destination!r} asked for"
        )
    after = await client.relation_fingerprint(datasource_name, relation=relation, key=columns, where=filters)
    if after != before:
        raise IncompleteExportError(
            f"{relation} {filters or ''} changed while it was exported ({before.row_count} rows before, "
            f"{after.row_count} after); the export is not one state of it"
        )
    if result.row_count != before.row_count:
        raise IncompleteExportError(
            f"{relation} {filters or ''}: the warehouse exported {result.row_count} rows, the relation holds "
            f"{before.row_count}"
        )
    rows = await read_export(store, result, bucket=bucket)
    try:
        deleted = await client.delete_export(datasource_name, destination=destination)
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- re-raised as the named failure, cause chained
        raise ExportNotDeletedError(
            f"{relation} {filters or ''}: the export was read and proven, but it could not be deleted and is "
            f"still at s3://{result.bucket}/{result.object_prefix}: {type(exc).__name__}: {exc}"
        ) from exc
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
