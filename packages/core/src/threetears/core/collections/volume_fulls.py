"""a full on the delta-files volume: per (table, state) part files and a manifest naming them, fenced by term.

**Why.** A table that opts in to delta files (:attr:`~threetears.core.collections.schema_backed.TableSchema.delta_files`)
keeps every change on the shared volume, and nothing there expires on a timer, so a pod copy can always catch up
(``design-pod-delta-files.md``). Its starting point, the full, lives on the same volume instead of the pod's NATS
Object Store (decision 7): a copy loads the full, then applies every delta newer than each part. L3 is never read
whole by a copy: with no full on the volume it waits for the writer to publish one (decision 14).

**Parts and a manifest** (decision 8). A full is one Arrow IPC file (zstd inside, decision 18) per (table, state), its
rows the table's columns typed as the L3 rail answers them, plus a manifest naming, per table, each part's file and the
version (the table's generation token, ``{incarnation}:{count}``) the part is current to, and the version the table as
a whole is current to (a state with no part holds no rows at it). A delta file may be named as a part (decision 16):
its delete markers are dropped as it loads, and any column beyond the table's own is left out. Files never change once
written; a new full is new parts for the states that changed plus a new manifest naming every part.

**Fenced by term** (decision 15). Every manifest is named by the writer's term (its write lease's, only ever going up
across handovers) and a sequence within the term: ``full-<term>-<seq>.json``. Readers take the highest term, then the
newest sequence in it. A writer that lost its lease and wakes to publish writes an older term's manifest, which no
reader takes and :func:`deletable_full_files` lists; nothing is overwritten, so a stale writer never replaces the
current writer's work. Where the term comes from is the writer's to settle; this module takes it as given.

**Published by rename** (decision 8). A manifest is written under a temporary name, flushed to disk, renamed to its
own name and the directory flushed: a reader sees the whole manifest or none, never part of one. A name already taken
is refused, never replaced: one writer holds a term (the lease), so nothing else makes a name of it.

**Kept and deleted.** The current manifest and the one it replaced (its ``previous``, which a reader that took it
before the publish may still be loading) are kept with every file they name; :func:`deletable_full_files` lists the
rest for the writer to delete after its own publish. Nothing calls it on a timer.

**Where.** Under the delta files' root (``THREETEARS_DELTA_FILES_DIR``), in ``.fulls/``: ``.fulls/manifests/`` and
``.fulls/parts/<table>/``. A table's delta files are ``<table>/<write>.arrow``; a table is an identifier, which never
begins with a dot, so no table's directory is ``.fulls``, and the delta files' orphan sweep, which looks only at the
files directly under each directory of the root, never reaches a part or a manifest.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

from threetears.observe import get_logger

from threetears.core.cache.duckdb import STAGE_PREFIX, StagedChange
from threetears.core.collections.generation import split_generation_token

if TYPE_CHECKING:
    from threetears.core.collections.copy_follower import FollowedTable, FollowerBackend

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a full refuses without it, saying so
    _PYARROW = False

__all__ = [
    "FullAbsent",
    "FullManifest",
    "FullPart",
    "FullTable",
    "StagedFull",
    "StaleTermError",
    "VolumeFulls",
    "current_full_manifest",
    "deletable_full_files",
    "publish_full_manifest",
    "write_full_part",
]

log = get_logger(__name__)

#: the fulls' directory under the delta files' root: a name no table's directory can take (a table is an identifier)
_FULLS: Final = ".fulls"
#: the manifests' directory, under :data:`_FULLS`
_MANIFESTS: Final = "manifests"
#: the parts' directory, under :data:`_FULLS`: one directory a table
_PARTS: Final = "parts"
#: a manifest's name: its term and sequence, zero-padded so a listing reads in order
_MANIFEST_NAME: Final = re.compile(r"^full-(\d{20})-(\d{20})\.json$")
#: the manifest's format; one of another is refused
_FORMAT: Final = 1
#: a part's suffix: the delta files' own, so a delta named as a part reads the same way
_SUFFIX: Final = ".arrow"
#: the suffix of a file being written, before its rename; never a name a manifest holds
_PARTIAL_SUFFIX: Final = ".partial"
#: a table name as it may become a directory
_TABLE_NAME: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: the column a delta file marks its delete markers by (``delta_files._DELETED_COLUMN``): a delta named as a part
#: loads its rows, never its markers
_DELETED_COLUMN: Final = "_3tears_deleted"
#: a part's schema metadata: its table, its state and the version it is current to
_TABLE_METADATA: Final = b"threetears.full.table"
_SCOPE_METADATA: Final = b"threetears.full.scope"
_VERSION_METADATA: Final = b"threetears.full.version"
#: how often a copy with no full to load looks for one again (seconds): one listing of the manifests' directory
_DEFAULT_POLL_SECONDS: Final = 1.0


class StaleTermError(RuntimeError):
    """a writer would publish a full under a term older than the current manifest's: another writer took the lease
    since, so this one's full is refused before it is written (readers would ignore it anyway)."""


@dataclass(frozen=True, slots=True)
class FullPart:
    """one (table, state) part of a full.

    :ivar scope: the state, as the writer's seqlock names it (``state:TX``); None for a table with no scope
    :ivar file: the file, relative to the delta files' root: a part's own (``.fulls/parts/<table>/<id>.arrow``) or a
        delta file named as the part (``<table>/<write>.arrow``)
    :ivar version: the table's generation token the part is current to: deltas of this version or older are in it
    :ivar rows: the rows it holds (a delta named as a part: its rows written, delete markers not counted)
    """

    scope: str | None
    file: str
    version: str
    rows: int

    def encode(self) -> dict[str, Any]:
        return {"scope": self.scope, "file": self.file, "version": self.version, "rows": self.rows}

    @classmethod
    def decode(cls, body: Mapping[str, Any]) -> FullPart:
        scope = body["scope"]
        return cls(
            scope=None if scope is None else str(scope),
            file=_relative_file(str(body["file"])),
            version=str(body["version"]),
            rows=int(body["rows"]),
        )


@dataclass(frozen=True, slots=True)
class FullTable:
    """one table in a full.

    :ivar version: the generation token the table as a whole is current to: a state with no part holds no rows at it
    :ivar parts: its parts, one a state
    """

    version: str
    parts: tuple[FullPart, ...] = ()

    def encode(self) -> dict[str, Any]:
        return {"version": self.version, "parts": [part.encode() for part in self.parts]}

    @classmethod
    def decode(cls, body: Mapping[str, Any]) -> FullTable:
        return cls(version=str(body["version"]), parts=tuple(FullPart.decode(part) for part in body["parts"]))


@dataclass(frozen=True, slots=True)
class FullManifest:
    """a published full: its term and sequence, the manifest it replaced, and each table's parts.

    :ivar term: the writer's term
    :ivar seq: its sequence within the term, from 1
    :ivar previous: the name of the manifest that was current when this one was published (kept beside it, since a
        reader may still be loading it); None for the first
    :ivar tables: each table's entry
    """

    term: int
    seq: int
    previous: str | None
    tables: Mapping[str, FullTable]

    @property
    def name(self) -> str:
        """the manifest's file name: its term, then its sequence.

        :return: the name
        :rtype: str
        """
        return _manifest_name(self.term, self.seq)

    def files(self) -> set[str]:
        """every file it names, relative to the root.

        :return: the files
        :rtype: set[str]
        """
        return {part.file for entry in self.tables.values() for part in entry.parts}

    def encode(self) -> bytes:
        body = {
            "format": _FORMAT,
            "term": self.term,
            "seq": self.seq,
            "previous": self.previous,
            "tables": {name: entry.encode() for name, entry in sorted(self.tables.items())},
        }
        return json.dumps(body, sort_keys=True).encode()

    @classmethod
    def decode(cls, raw: bytes) -> FullManifest:
        """a manifest read back.

        :raises ValueError: when it is not a manifest of this format
        :raises KeyError: when a field is missing
        """
        body = json.loads(raw)
        if body.get("format") != _FORMAT:
            raise ValueError(f"a full manifest of format {body.get('format')!r}, not {_FORMAT}")
        previous = body.get("previous")
        return cls(
            term=int(body["term"]),
            seq=int(body["seq"]),
            previous=None if previous is None else str(previous),
            tables={str(name): FullTable.decode(entry) for name, entry in body["tables"].items()},
        )


def _manifest_name(term: int, seq: int) -> str:
    return f"full-{term:020d}-{seq:020d}.json"


def _relative_file(file: str) -> str:
    """a manifest's file name, checked to stay under the root.

    :raises ValueError: when it is absolute or climbs out
    """
    path = PurePosixPath(file)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"a full names a file outside its root: {file!r}")
    return file


def _manifests_dir(root: Path) -> Path:
    return root / _FULLS / _MANIFESTS


def _fsync_directory(directory: Path) -> None:
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _write_renamed(final: Path, write: Any) -> None:
    """write ``final`` under a temporary name beside it, flush it, rename it into place and flush the directory, so a
    reader that opens the name sees the whole file or none. Refuses a name already taken: nothing is overwritten.

    :param final: the file
    :param write: writes the file's bytes to the open sink
    :raises FileExistsError: when ``final`` exists
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"{final} exists; a full's files are never overwritten")
    partial = final.parent / f".{final.name}.{uuid.uuid4().hex}{_PARTIAL_SUFFIX}"
    renamed = False
    try:
        with partial.open("wb") as sink:
            write(sink)
            sink.flush()
            os.fsync(sink.fileno())
        os.rename(partial, final)
        renamed = True
    finally:
        if not renamed:
            partial.unlink(missing_ok=True)
    _fsync_directory(final.parent)


def write_full_part(root: Path, table: str, scope: str | None, rows: Any, version: str) -> FullPart:
    """save one (table, state) part of a full under ``root``: Arrow IPC with zstd inside, under a new name, written
    and renamed into place. Blocking: run it on a thread.

    :param root: the delta files' root
    :ptype root: Path
    :param table: the table
    :ptype table: str
    :param scope: the state; None for a table with no scope
    :ptype scope: str | None
    :param rows: the part's rows, the table's columns
    :ptype rows: pyarrow.Table
    :param version: the generation token the rows are current to
    :ptype version: str
    :return: the part, to name in a manifest
    :rtype: FullPart
    :raises ValueError: when the table is not an identifier
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a full needs pyarrow: install 3tears[snapshot]")
    if not _TABLE_NAME.match(table):
        raise ValueError(f"a full's table must be an identifier, got {table!r}")
    file = f"{_FULLS}/{_PARTS}/{table}/{uuid.uuid4().hex}{_SUFFIX}"
    metadata = {_TABLE_METADATA: table.encode(), _VERSION_METADATA: version.encode()}
    if scope is not None:
        metadata[_SCOPE_METADATA] = scope.encode()
    held = rows.replace_schema_metadata(metadata)

    def write(sink: Any) -> None:
        options = pa_ipc.IpcWriteOptions(compression="zstd")
        with pa_ipc.new_file(sink, held.schema, options=options) as writer:
            writer.write_table(held)

    _write_renamed(root / file, write)
    return FullPart(scope=scope, file=file, version=version, rows=held.num_rows)


def _listed_manifests(root: Path) -> list[tuple[int, int, str]]:
    """every published manifest under ``root``, as (term, seq, name), oldest first; a temporary one is not listed."""
    directory = _manifests_dir(root)
    if not directory.is_dir():
        return []
    listed = []
    for path in directory.iterdir():
        matched = _MANIFEST_NAME.match(path.name)
        if matched is not None:
            listed.append((int(matched.group(1)), int(matched.group(2)), path.name))
    return sorted(listed)


def _read_manifest(root: Path, name: str) -> FullManifest:
    return FullManifest.decode((_manifests_dir(root) / name).read_bytes())


def current_full_manifest(root: Path) -> FullManifest | None:
    """the manifest readers take: the highest term, then the newest sequence in it; None when none is published.
    Blocking: run it on a thread.

    :param root: the delta files' root
    :ptype root: Path
    :return: the manifest
    :rtype: FullManifest | None
    :raises ValueError: when the current manifest does not read back (never skipped for an older one: an older term's
        may be a stale writer's)
    """
    listed = _listed_manifests(root)
    if not listed:
        return None
    term, seq, name = listed[-1]
    manifest = _read_manifest(root, name)
    if (manifest.term, manifest.seq) != (term, seq):
        raise ValueError(f"full manifest {name} says term {manifest.term} seq {manifest.seq}")
    return manifest


def publish_full_manifest(root: Path, *, term: int, tables: Mapping[str, FullTable]) -> FullManifest:
    """publish a full: a manifest of ``tables`` at ``term``, the next sequence in it, naming the current manifest as
    its previous; written under a temporary name, flushed, renamed into place. Blocking: run it on a thread.

    Every part it names must already be written (:func:`write_full_part`, or a delta file). Only the writer holding the
    write lease publishes, under that lease's term.

    :param root: the delta files' root
    :ptype root: Path
    :param term: the writer's term, from its write lease; only ever higher for a later holder
    :ptype term: int
    :param tables: every table the full holds, each with every part (those it keeps of the previous full, and new ones)
    :ptype tables: Mapping[str, FullTable]
    :return: the manifest published
    :rtype: FullManifest
    :raises StaleTermError: when the current manifest's term is higher: another writer took the lease since
    :raises FileExistsError: when the name is taken (two writers of one term: the lease was not held)
    """
    if term < 0:
        raise ValueError(f"a term is not negative, got {term}")
    for name in tables:
        if not _TABLE_NAME.match(name):
            raise ValueError(f"a full's table must be an identifier, got {name!r}")
    listed = _listed_manifests(root)
    if listed and listed[-1][0] > term:
        raise StaleTermError(
            f"the current full is of term {listed[-1][0]}, above this writer's {term}: another writer took the lease"
        )
    seq = max((s for t, s, _ in listed if t == term), default=0) + 1
    manifest = FullManifest(term=term, seq=seq, previous=listed[-1][2] if listed else None, tables=dict(tables))
    for file in manifest.files():
        _relative_file(file)
    raw = manifest.encode()
    _write_renamed(_manifests_dir(root) / manifest.name, lambda sink: sink.write(raw))
    log.info(
        "full published",
        extra={
            "extra_data": {
                "manifest": manifest.name,
                "previous": manifest.previous,
                "tables": len(manifest.tables),
                "parts": len(manifest.files()),
            }
        },
    )
    return manifest


def deletable_full_files(root: Path, *, covered_deltas: Collection[str] = ()) -> list[str]:
    """the files of the fulls the writer may delete: every manifest but the current one and its previous, every
    part neither names, every temporary file left by a write that died, and each delta in ``covered_deltas`` neither
    names (a delta named as a part is kept, decision 16). Blocking: run it on a thread.

    Call it only from the writer, after its own publish and before it writes the next full's parts: a part written for
    a manifest not yet published is named by none. With no manifest published nothing is listed (a rebuild's parts
    are written before its first manifest).

    :param root: the delta files' root
    :ptype root: Path
    :param covered_deltas: delta files (``<table>/<write>.arrow``) the current full covers
    :ptype covered_deltas: Collection[str]
    :return: the files, relative to ``root``
    :rtype: list[str]
    """
    current = current_full_manifest(root)
    if current is None:
        return []
    kept_manifests = {current.name}
    kept_files = current.files()
    if current.previous is not None and (_manifests_dir(root) / current.previous).is_file():
        kept_manifests.add(current.previous)
        kept_files |= _read_manifest(root, current.previous).files()
    deletable: list[str] = []
    manifests = _manifests_dir(root)
    for path in sorted(manifests.iterdir()):
        if path.is_file() and path.name not in kept_manifests:
            deletable.append(f"{_FULLS}/{_MANIFESTS}/{path.name}")
    parts = root / _FULLS / _PARTS
    if parts.is_dir():
        for directory in sorted(p for p in parts.iterdir() if p.is_dir()):
            for path in sorted(directory.iterdir()):
                file = f"{_FULLS}/{_PARTS}/{directory.name}/{path.name}"
                if path.is_file() and file not in kept_files:
                    deletable.append(file)
    deletable.extend(sorted(file for file in covered_deltas if file not in kept_files))
    return deletable


def _versions_floor(table: str, versions: Iterable[str]) -> str:
    """the oldest of a table's versions: where a copy of every part of it catches up from.

    :raises ValueError: when a version is not ``{incarnation}:{count}``, or two are of different incarnations (a table
        reset or rebuilt starts fresh: every part is of the new one)
    """
    split = []
    for version in versions:
        parsed = split_generation_token(version)
        if parsed is None:
            raise ValueError(f"{table}: a full's version must be a generation token, got {version!r}")
        split.append((parsed, version))
    incarnations = {incarnation for (incarnation, _), _ in split}
    if len(incarnations) != 1:
        raise ValueError(f"{table}: a full's parts are of incarnations {sorted(incarnations)}, not one")
    return min(split)[1]


@dataclass(frozen=True, slots=True)
class FullAbsent:
    """no full a copy can load yet: it waits for the writer to publish one, never reading L3 whole.

    :ivar reason: why, for the follower's status
    """

    reason: str


@dataclass(frozen=True, slots=True)
class StagedFull:
    """what :meth:`VolumeFulls.stage` staged beside the copy from the current full.

    :ivar manifest: the manifest's name
    :ivar versions: each table's version per state: the part's, or the table's for a state with no part (key None
        for a table with no scope, or a table with no part). A delta at or below a state's version is in the copy
    :ivar marks: each table's oldest version, of the table and its parts: where the copy catches up from
    :ivar changes: each table's stage, whole, to apply in one transaction
    :ivar rows: the rows staged
    """

    manifest: str
    versions: Mapping[str, Mapping[str | None, str]]
    marks: Mapping[str, str]
    changes: list[StagedChange]
    rows: int


class VolumeFulls:
    """a copy follower's start from the full on the delta-files volume, for the tables that opt in.

    :param root: the delta files' root (``delta_files_root()``)
    :ptype root: Path
    :param tables: the copy's tables that opt in to delta files; every other table starts as today
    :ptype tables: Sequence[FollowedTable]
    :param poll_seconds: how often a copy with no full to load looks for one again
    :ptype poll_seconds: float
    """

    def __init__(self, root: Path, tables: Sequence[FollowedTable], *, poll_seconds: float = _DEFAULT_POLL_SECONDS):
        if not _PYARROW:
            raise ImportError("a full needs pyarrow: install 3tears[snapshot]")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.root = root
        self.tables: Final = {table.name: table for table in tables}
        self.poll_seconds = poll_seconds

    async def stage(self, backend: FollowerBackend, *, manifest: FullManifest | None = None) -> StagedFull | FullAbsent:
        """stage each table's parts of the current full beside the copy, a part at a time; :class:`FullAbsent` when no
        full is published, or the current one does not hold every table.

        :param backend: the copy
        :ptype backend: FollowerBackend
        :param manifest: the manifest to load, read already; None reads the current one
        :ptype manifest: FullManifest | None
        :return: what was staged, or why nothing was
        :rtype: StagedFull | FullAbsent
        :raises FileNotFoundError: when a part the manifest names is not on the volume
        """
        if manifest is None:
            manifest = await asyncio.to_thread(current_full_manifest, self.root)
        if manifest is None:
            return FullAbsent("no full on the volume yet; the writer publishes one")
        missing = sorted(set(self.tables) - set(manifest.tables))
        if missing:
            return FullAbsent(f"full {manifest.name} does not hold {missing} yet; the writer publishes one that does")
        versions: dict[str, dict[str | None, str]] = {}
        marks: dict[str, str] = {}
        changes: list[StagedChange] = []
        rows = 0
        for name, table in self.tables.items():
            entry = manifest.tables[name]
            marks[name] = _versions_floor(name, [entry.version, *(part.version for part in entry.parts)])
            held: dict[str | None, str] = {None: entry.version}
            held.update((part.scope, part.version) for part in entry.parts)
            versions[name] = held
            stage = f"{STAGE_PREFIX}{name}__rows"
            await asyncio.to_thread(backend.drop_stage, stage)
            try:
                await asyncio.to_thread(backend.stage_rows, stage, name, pa.table({}), table.key)
                for part in entry.parts:
                    rows += await asyncio.to_thread(self._stage_part, backend, stage, table, part)
            except BaseException:
                await asyncio.to_thread(backend.drop_stage, *[c.rows for c in changes if c.rows], stage)
                raise
            # a start from the full fills an empty copy: its apply empties the table as a cold start, as a checkpoint's
            changes.append(StagedChange(table=name, key=table.key, rows=stage, whole=True, whole_kind="cold_start"))
        log.info(
            "full staged from the volume",
            extra={"extra_data": {"manifest": manifest.name, "tables": len(changes), "rows": rows}},
        )
        return StagedFull(manifest=manifest.name, versions=versions, marks=marks, changes=changes, rows=rows)

    def _stage_part(self, backend: FollowerBackend, stage: str, table: FollowedTable, part: FullPart) -> int:
        """one part read off the volume and staged, on the caller's worker thread.

        :return: the rows staged
        :raises ValueError: when the part holds other than the rows its manifest says
        """
        with pa.OSFile(str(self.root / part.file), "rb") as source:
            rows = pa_ipc.open_file(source).read_all()
        if _DELETED_COLUMN in rows.column_names:
            # a delta named as the part: its rows written, never its delete markers
            rows = rows.filter(pc.invert(rows.column(_DELETED_COLUMN)))
        rows = rows.select([column.name for column in table.columns])
        if rows.num_rows != part.rows:
            raise ValueError(f"part {part.file} holds {rows.num_rows} rows, its manifest says {part.rows}")
        return int(backend.stage_rows(stage, table.name, rows, table.key)) if rows.num_rows else 0
