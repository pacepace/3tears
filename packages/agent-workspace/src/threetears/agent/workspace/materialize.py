"""materialize, bind, recover -- runtime primitives bridging L3 <-> disk.

three public entrypoints are offered:

- :func:`materialize` dumps every file from a workspace into a fresh
  temp directory using :func:`atomic_write`. no lease is acquired (the
  tempdir is ephemeral and per-caller so there is no cross-pod
  contention), and the caller owns cleanup. intended for builders that
  accept a path argument.

- :func:`bind` is an async context manager that syncs L3 files onto a
  configured sandbox named root (the ``bind`` root by default), yields
  the disk path, and on clean exit captures disk changes back to L3.
  the bind window is serialized across pods via
  :class:`WorkspaceFileLease` keyed on ``"bind:{root_name}"``. on
  exception the lease releases through the ``async with`` scope and the
  capture-back phase is skipped by design (disk state after a crashed
  body is suspect; partial progress is intentionally lost).

- :func:`recover` performs the capture-back phase explicitly. operators
  who inspect a disk root after a crashed bind and decide the state is
  good call it to flush disk into L3 without the lease. since there is
  no enter-snapshot available in a recovery scenario, recover compares
  each on-disk file's sha256 against the current L3 head row and writes
  only the differences.

writes during capture-back intentionally bypass
:func:`threetears.agent.workspace.tools.helpers._write_file_atomic`
because that helper enforces optimistic-concurrency against the head
row's sha256 -- bind owns the workspace during its window, so OCC
against a sha the bind process itself produced would merely forbid
legitimate writes. the bind transaction opens one asyncpg transaction
per cycle and emits ``workspace_file_versions`` journal inserts plus
``workspace_files`` head upserts plus a ``workspaces.current_version``
bump in a single statement batch.

the workspace-validator hook (shard 16) is likewise NOT run on
capture-back. bind owns the disk window: the operator or builder
process writes files through whatever tooling it likes, and the
validator interface's ``Callable[[str, bytes], None]`` contract was
designed for the LLM-driven fs_* / doc_* tool surface where a single
tool call produces one write we can intercept cleanly. capture-back
mops up an arbitrary batch of disk changes at end-of-window, and
running validators there would either (a) reject half a bind's worth
of legitimate work because one emitted file happened to fail schema,
or (b) force partial commits that break the atomicity the bind
transaction was written to deliver. the shard explicitly records this
exclusion: validate through the LLM tools that produce the writes,
let bind own the window.

anti-patterns deliberately avoided:

- capture-back on exception -- never. disk state is suspect after a
  body crash, and losing the partial progress is the design.
- :func:`_write_file_atomic` inside capture-back -- never, per above.
- skipping lease acquisition inside :func:`bind` -- never; multi-pod
  safety depends on it.
- automatic deletion of the disk root after bind exit -- never; the
  next bind call benefits from a warm disk state, and explicit cleanup
  is an administrative concern.
- :func:`os.walk` for disk enumeration -- use :meth:`pathlib.Path.rglob`
  to match the rest of the codebase.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal
from uuid import UUID, uuid7

from watchfiles import Change, awatch

from threetears.agent.audit import AuditEvent, publish_audit
from threetears.core.utils.atomic_write import atomic_write
from threetears.observe import get_logger

from threetears.agent.workspace.bind_policy import BindConflictPolicy
from threetears.agent.workspace.tools.helpers import _next_journal_version

__all__ = [
    "bind",
    "materialize",
    "recover",
]

if TYPE_CHECKING:
    from threetears.agent.workspace.collections import (
        WorkspaceCollection,
        WorkspaceFileCollection,
        WorkspaceFileVersionCollection,
    )
    from threetears.agent.workspace.lease import WorkspaceFileLease
    from threetears.agent.workspace.sandbox import WorkspaceSandbox


log = get_logger(__name__)


_INSERT_WORKSPACE_FILE_VERSION_SQL = """
INSERT INTO workspace_file_versions (
    id, workspace_id, relative_path, version, content,
    sha256, action, label, actor_id, correlation_id, date_created
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
"""

_UPSERT_WORKSPACE_FILE_SQL = """
INSERT INTO workspace_files (
    id, workspace_id, relative_path, content, sha256, version, date_updated
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (workspace_id, relative_path) DO UPDATE SET
    content = EXCLUDED.content,
    sha256 = EXCLUDED.sha256,
    version = EXCLUDED.version,
    date_updated = EXCLUDED.date_updated
"""

_DELETE_WORKSPACE_FILE_SQL = """
DELETE FROM workspace_files
WHERE workspace_id = $1 AND relative_path = $2
"""

_UPDATE_WORKSPACE_VERSION_SQL = """
UPDATE workspaces
SET current_version = GREATEST(current_version, $1),
    date_updated = $2
WHERE id = $3
"""


async def materialize(
    *,
    workspace_id: UUID,
    workspace_file_collection: WorkspaceFileCollection,
    parent_dir: Path | None = None,
) -> Path:
    """write every file in workspace to fresh temp dir, return dir path.

    queries all head-state file rows for ``workspace_id`` through
    :meth:`WorkspaceFileCollection.find_by_workspace` and writes each one
    to the temp directory at its ``relative_path`` using
    :func:`atomic_write`. parent directories are created as needed.

    no lease is acquired: the tempdir is unique per call (created via
    :func:`tempfile.mkdtemp`) and ephemeral, so no cross-pod contention
    is possible. the caller owns cleanup -- typical pattern::

        tempdir = await materialize(
            workspace_id=ws_id,
            workspace_file_collection=files,
        )
        try:
            run_builder(tempdir)
        finally:
            shutil.rmtree(tempdir, ignore_errors=True)

    :param workspace_id: identifier of workspace whose files to dump
    :ptype workspace_id: UUID
    :param workspace_file_collection: head-state file collection used to
        enumerate the workspace
    :ptype workspace_file_collection: WorkspaceFileCollection
    :param parent_dir: optional parent directory under which the temp
        dir is created; defaults to the system temp location
    :ptype parent_dir: Path | None
    :return: absolute path to the fresh temp directory containing all
        files from workspace
    :rtype: Path
    """
    tempdir_str = tempfile.mkdtemp(
        prefix=f"workspace-{workspace_id.hex[:8]}-",
        dir=None if parent_dir is None else str(parent_dir),
    )
    tempdir = Path(tempdir_str)
    files = await workspace_file_collection.find_by_workspace(workspace_id)
    for file_entity in files:
        target = tempdir / file_entity.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        await atomic_write(target, file_entity.content)
    log.info(
        "workspace.materialize.done",
        extra={
            "workspace_id": str(workspace_id),
            "file_count": len(files),
            "tempdir": str(tempdir),
        },
    )
    return tempdir


def _sha256_bytes(payload: bytes) -> str:
    """compute hex sha256 digest of raw bytes.

    :param payload: raw bytes to digest
    :ptype payload: bytes
    :return: 64-character hex digest
    :rtype: str
    """
    return hashlib.sha256(payload).hexdigest()


def _snapshot_disk_sync(disk_root: Path) -> dict[str, tuple[bytes, str]]:
    """sync walker: enumerate every file under ``disk_root``, return ``{relpath: (bytes, sha256)}``.

    walks via :meth:`Path.rglob` (not :func:`os.walk`) and filters to
    regular files. symlinks that resolve OUTSIDE the root are dropped
    defensively so an attacker who planted a symlink into ``/etc`` cannot
    exfiltrate its contents via the capture-back path.

    this is the sync, blocking body. async callers must invoke via
    :func:`asyncio.to_thread` so the event loop is not blocked on the
    filesystem walk.

    :param disk_root: absolute path to sandboxed root directory
    :ptype disk_root: Path
    :return: mapping from POSIX-style relative path to (content, sha256)
    :rtype: dict[str, tuple[bytes, str]]
    """
    out: dict[str, tuple[bytes, str]] = {}
    resolved_root = disk_root.resolve()
    for candidate in disk_root.rglob("*"):
        if not candidate.is_file():
            continue
        # reject symlinks (or any entry) whose resolved target escapes
        # the root. is_file() follows symlinks; .resolve() + parentage
        # check closes the symlink-escape hole explicitly.
        try:
            resolved_candidate = candidate.resolve()
            resolved_candidate.relative_to(resolved_root)
        except OSError, ValueError:
            log.warning(
                "workspace.snapshot_disk.skip_escape",
                extra={
                    "extra_data": {
                        "candidate": str(candidate),
                        "disk_root": str(disk_root),
                    },
                },
            )
            continue
        rel = candidate.relative_to(disk_root).as_posix()
        data = candidate.read_bytes()
        out[rel] = (data, _sha256_bytes(data))
    return out


async def _snapshot_disk(disk_root: Path) -> dict[str, tuple[bytes, str]]:
    """async wrapper around :func:`_snapshot_disk_sync` via :func:`asyncio.to_thread`.

    :param disk_root: absolute path to sandboxed root directory
    :ptype disk_root: Path
    :return: mapping from POSIX-style relative path to (content, sha256)
    :rtype: dict[str, tuple[bytes, str]]
    """
    return await asyncio.to_thread(_snapshot_disk_sync, disk_root)


async def _seed_l3_from_disk(
    *,
    workspace: Any,
    disk_root: Path,
    workspace_file_collection: WorkspaceFileCollection,
    db_pool: Any,
    actor_id: UUID,
    correlation_id: UUID,
    on_conflict: BindConflictPolicy,
) -> int:
    """seed L3 head-state from disk under the chosen conflict policy.

    branches on ``on_conflict``:

    - :attr:`BindConflictPolicy.L3_WINS` -- L3 is authoritative.
      runs the original "seed if empty" gate: when
      :meth:`WorkspaceFileCollection.find_by_workspace` returns rows
      the helper is a no-op; otherwise every file under ``disk_root``
      is imported as a ``create`` at version 1 in one transaction and
      ``workspaces.current_version`` is bumped via ``GREATEST``. this
      preserves the historical behavior of
      ``_import_disk_to_l3_if_empty``.

    - :attr:`BindConflictPolicy.DISK_WINS` -- disk is authoritative.
      always walks disk. every disk path becomes a ``create`` journal
      row + head upsert (when no head row exists) or an ``update``
      journal row + head upsert (when the existing head sha differs).
      every L3-only path (present in head-state, absent from disk)
      becomes a ``delete`` journal row + head delete. unchanged files
      (matching sha) are skipped. all writes land in a single asyncpg
      transaction and ``workspaces.current_version`` is bumped via
      ``GREATEST``.

    validator dispatch is intentionally bypassed in both modes: the
    bind-enter contract mirrors disk <-> L3 while validators target the
    LLM-tool write surface, so re-seeding an existing directory does
    not fail because some legacy file violates a schema the agent
    author added later; the validator will reject the next LLM write
    instead.

    :param workspace: target workspace entity whose head-state to seed
    :ptype workspace: Any
    :param disk_root: absolute path to sandboxed bind root
    :ptype disk_root: Path
    :param workspace_file_collection: head-state collection used for
        emptiness gate in L3_WINS mode
    :ptype workspace_file_collection: WorkspaceFileCollection
    :param db_pool: asyncpg pool supplying acquire + transaction
    :ptype db_pool: Any
    :param actor_id: identifier of actor running the seed
    :ptype actor_id: UUID
    :param correlation_id: originating tool-call envelope identifier
    :ptype correlation_id: UUID
    :param on_conflict: policy selecting the seed strategy
    :ptype on_conflict: BindConflictPolicy
    :return: count of files created + updated + deleted during seeding
        (0 when the L3_WINS gate short-circuits and the workspace
        already has head rows)
    :rtype: int
    """
    n_touched = 0
    if on_conflict is BindConflictPolicy.L3_WINS:
        existing = await workspace_file_collection.find_by_workspace(
            workspace.id,
        )
        if not existing:
            disk = await _snapshot_disk(disk_root)
            if disk:
                n_touched = await _seed_l3_import_all(
                    workspace=workspace,
                    disk=disk,
                    db_pool=db_pool,
                    actor_id=actor_id,
                    correlation_id=correlation_id,
                )
                log.info(
                    "workspace.bind.seed_l3_wins",
                    extra={
                        "workspace_id": str(workspace.id),
                        "file_count": n_touched,
                        "disk_root": str(disk_root),
                    },
                )
    else:
        disk = await _snapshot_disk(disk_root)
        existing_rows = await workspace_file_collection.find_by_workspace(
            workspace.id,
        )
        existing_by_path: dict[str, tuple[str, int]] = {
            row.relative_path: (row.sha256, row.version) for row in existing_rows
        }
        n_touched = await _seed_l3_disk_wins(
            workspace=workspace,
            disk=disk,
            existing_by_path=existing_by_path,
            db_pool=db_pool,
            actor_id=actor_id,
            correlation_id=correlation_id,
        )
        log.info(
            "workspace.bind.seed_disk_wins",
            extra={
                "workspace_id": str(workspace.id),
                "file_count": n_touched,
                "disk_root": str(disk_root),
            },
        )
    return n_touched


async def _seed_l3_import_all(
    *,
    workspace: Any,
    disk: dict[str, tuple[bytes, str]],
    db_pool: Any,
    actor_id: UUID,
    correlation_id: UUID,
) -> int:
    """bulk-import every disk path as a ``create`` journal row + head upsert.

    L3_WINS-empty helper: the caller has already verified the workspace
    head-state is empty, so every disk path is a legitimate new
    ``create`` at version 1 and no sha-diff check is needed. the
    transactional envelope matches the DISK_WINS variant so the two
    share commit semantics.

    :param workspace: target workspace entity
    :ptype workspace: Any
    :param disk: mapping of ``relative_path`` to ``(content, sha256)``
    :ptype disk: dict[str, tuple[bytes, str]]
    :param db_pool: asyncpg pool supplying acquire + transaction
    :ptype db_pool: Any
    :param actor_id: identifier of actor running the seed
    :ptype actor_id: UUID
    :param correlation_id: originating tool-call envelope identifier
    :ptype correlation_id: UUID
    :return: count of files imported
    :rtype: int
    """
    now = datetime.now(UTC)
    action_create: Literal["create"] = "create"
    max_version = 0
    async with db_pool.acquire() as conn:
        async with conn.transaction(namespace=workspace.namespace_name):
            for rel, (content, sha) in disk.items():
                new_version = await _next_journal_version(
                    conn,
                    workspace.id,
                    rel,
                )
                if new_version > max_version:
                    max_version = new_version
                await conn.execute(
                    _INSERT_WORKSPACE_FILE_VERSION_SQL,
                    uuid7(),
                    workspace.id,
                    rel,
                    new_version,
                    content,
                    sha,
                    action_create,
                    None,
                    actor_id,
                    correlation_id,
                    now,
                )
                await conn.execute(
                    _UPSERT_WORKSPACE_FILE_SQL,
                    uuid7(),
                    workspace.id,
                    rel,
                    content,
                    sha,
                    new_version,
                    now,
                )
            await conn.execute(
                _UPDATE_WORKSPACE_VERSION_SQL,
                max_version,
                now,
                workspace.id,
            )
    return len(disk)


async def _seed_l3_disk_wins(
    *,
    workspace: Any,
    disk: dict[str, tuple[bytes, str]],
    existing_by_path: dict[str, tuple[str, int]],
    db_pool: Any,
    actor_id: UUID,
    correlation_id: UUID,
) -> int:
    """clobber L3 head-state with disk contents: create + update + delete.

    DISK_WINS-enter helper. compares ``disk`` against
    ``existing_by_path`` (head-state ``{rel: (sha, version)}``) and
    inside a single asyncpg transaction:

    - emits a ``create`` journal row + head upsert for every disk path
      with no existing head row
    - emits an ``update`` journal row + head upsert for every disk
      path whose existing head sha differs
    - emits a ``delete`` journal row + head delete for every existing
      head path absent from disk
    - bumps ``workspaces.current_version`` via ``GREATEST`` when any
      of the above fired

    unchanged files (matching sha) are skipped entirely.

    :param workspace: target workspace entity
    :ptype workspace: Any
    :param disk: mapping of ``relative_path`` to ``(content, sha256)``
        produced by :func:`_snapshot_disk`
    :ptype disk: dict[str, tuple[bytes, str]]
    :param existing_by_path: existing head-state indexed by relative
        path, mapping to ``(sha256, version)``
    :ptype existing_by_path: dict[str, tuple[str, int]]
    :param db_pool: asyncpg pool supplying acquire + transaction
    :ptype db_pool: Any
    :param actor_id: identifier of actor running the seed
    :ptype actor_id: UUID
    :param correlation_id: originating tool-call envelope identifier
    :ptype correlation_id: UUID
    :return: count of journal rows emitted (create + update + delete)
    :rtype: int
    """
    creates: list[tuple[str, bytes, str]] = []
    updates: list[tuple[str, bytes, str]] = []
    deletes: list[str] = []
    for rel, (content, sha) in disk.items():
        prior = existing_by_path.get(rel)
        if prior is None:
            creates.append((rel, content, sha))
        elif prior[0] != sha:
            updates.append((rel, content, sha))
    for rel in existing_by_path:
        if rel not in disk:
            deletes.append(rel)

    n_touched = 0
    if creates or updates or deletes:
        now = datetime.now(UTC)
        action_create: Literal["create"] = "create"
        action_update: Literal["update"] = "update"
        action_delete: Literal["delete"] = "delete"
        max_version = 0
        async with db_pool.acquire() as conn:
            async with conn.transaction(namespace=workspace.namespace_name):
                for rel, content, sha in creates:
                    new_version = await _next_journal_version(
                        conn,
                        workspace.id,
                        rel,
                    )
                    if new_version > max_version:
                        max_version = new_version
                    await conn.execute(
                        _INSERT_WORKSPACE_FILE_VERSION_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        new_version,
                        content,
                        sha,
                        action_create,
                        None,
                        actor_id,
                        correlation_id,
                        now,
                    )
                    await conn.execute(
                        _UPSERT_WORKSPACE_FILE_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        content,
                        sha,
                        new_version,
                        now,
                    )
                for rel, content, sha in updates:
                    new_version = await _next_journal_version(
                        conn,
                        workspace.id,
                        rel,
                    )
                    if new_version > max_version:
                        max_version = new_version
                    await conn.execute(
                        _INSERT_WORKSPACE_FILE_VERSION_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        new_version,
                        content,
                        sha,
                        action_update,
                        None,
                        actor_id,
                        correlation_id,
                        now,
                    )
                    await conn.execute(
                        _UPSERT_WORKSPACE_FILE_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        content,
                        sha,
                        new_version,
                        now,
                    )
                for rel in deletes:
                    new_version = await _next_journal_version(
                        conn,
                        workspace.id,
                        rel,
                    )
                    if new_version > max_version:
                        max_version = new_version
                    await conn.execute(
                        _INSERT_WORKSPACE_FILE_VERSION_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        new_version,
                        b"",
                        _sha256_bytes(b""),
                        action_delete,
                        None,
                        actor_id,
                        correlation_id,
                        now,
                    )
                    await conn.execute(
                        _DELETE_WORKSPACE_FILE_SQL,
                        workspace.id,
                        rel,
                    )
                await conn.execute(
                    _UPDATE_WORKSPACE_VERSION_SQL,
                    max_version,
                    now,
                    workspace.id,
                )
        n_touched = len(creates) + len(updates) + len(deletes)
    return n_touched


def _resolve_under_root(
    candidate: Path,
    disk_root: Path,
    resolved_root: Path,
) -> str | None:
    """return posix relative path under ``disk_root`` or ``None`` if escape.

    reuses the symlink-escape guard used by :func:`_snapshot_disk_sync`:
    resolves the candidate and verifies its parentage against the
    resolved root, dropping entries whose resolved target lives outside
    the root (symlink exfiltration).

    :param candidate: absolute filesystem path reported by a watch event
    :ptype candidate: Path
    :param disk_root: raw sandboxed root (pre-resolve)
    :ptype disk_root: Path
    :param resolved_root: :meth:`Path.resolve` of ``disk_root``, computed
        once by the caller
    :ptype resolved_root: Path
    :return: posix relative path when safe, ``None`` when escape detected
    :rtype: str | None
    """
    result: str | None = None
    try:
        resolved_candidate = candidate.resolve()
        resolved_candidate.relative_to(resolved_root)
        result = candidate.relative_to(disk_root).as_posix()
    except OSError, ValueError:
        log.warning(
            "workspace.watch.skip_escape",
            extra={
                "extra_data": {
                    "candidate": str(candidate),
                    "disk_root": str(disk_root),
                },
            },
        )
    return result


def _read_file_sync(path: Path) -> bytes:
    """sync body for :func:`asyncio.to_thread` dispatch of :meth:`Path.read_bytes`.

    :param path: absolute path to file to read
    :ptype path: Path
    :return: raw file bytes
    :rtype: bytes
    """
    return path.read_bytes()


async def _handle_watch_batch(
    *,
    batch: set[tuple[Change, str]],
    workspace: Any,
    disk_root: Path,
    resolved_root: Path,
    db_pool: Any,
    actor_id: UUID,
    correlation_id: UUID,
    just_wrote: deque[tuple[str, str]],
    on_conflict: BindConflictPolicy = BindConflictPolicy.DISK_WINS,
) -> list[str]:
    """apply one batch of :func:`watchfiles.awatch` changes to L3 under policy.

    coalesces the incoming change set by relative path so multiple
    events on the same file in one batch collapse to a single DB write,
    skips paths that escape ``disk_root`` via symlink, and filters the
    "just-wrote-by-us" set so round-trip writes by the bind process
    itself do not re-journal.

    behavior under ``on_conflict``:

    - :attr:`BindConflictPolicy.DISK_WINS` -- disk is authoritative
      during the bind window. ``added`` and ``modified`` events read
      the current bytes off disk (via :func:`asyncio.to_thread`),
      compute the sha256, and emit either ``create`` (no head row
      exists) or ``update`` (head row differs) as a single
      transactional pair of ``INSERT INTO workspace_file_versions`` +
      ``INSERT INTO workspace_files ... ON CONFLICT DO UPDATE``.
      ``deleted`` events emit ``action="delete"`` + ``DELETE FROM
      workspace_files``.

    - :attr:`BindConflictPolicy.L3_WINS` -- L3 is authoritative during
      the bind window. branches per event kind:

      * ``added`` for a path NOT in L3: imports as ``create`` (the
        agent never wrote this file, so external creation is new
        content L3 must carry);
      * ``added`` for a path ALREADY in L3: skipped (the disk copy is
        almost certainly our L3 -> disk sync echoing back; the real
        round-trip guard via ``just_wrote`` covers the common case
        but the event can still arrive without a sha match, e.g.
        editor save of identical content);
      * ``modified``: skipped entirely (L3 holds truth; the external
        modification is discarded and will be overwritten on the next
        bind's L3 -> disk projection);
      * ``deleted``: skipped (L3 holds truth; the file will be
        re-materialized via ``atomic_write`` on the next bind enter).

    validator dispatch is intentionally bypassed here in both modes:
    the watcher observes disk mutations from arbitrary external
    processes, and the bind contract places validators on the LLM-tool
    write surface instead. document + fs tools still gate their writes.

    :param batch: set of ``(Change, absolute_path_string)`` tuples
        yielded by :func:`watchfiles.awatch`
    :ptype batch: set[tuple[Change, str]]
    :param workspace: target workspace entity whose pointer to advance
    :ptype workspace: Any
    :param disk_root: absolute path to sandboxed bind root
    :ptype disk_root: Path
    :param resolved_root: pre-resolved root for symlink-escape detection
    :ptype resolved_root: Path
    :param db_pool: asyncpg pool supplying acquire + transaction
    :ptype db_pool: Any
    :param actor_id: identifier of actor owning the bind window
    :ptype actor_id: UUID
    :param correlation_id: originating tool-call envelope identifier
    :ptype correlation_id: UUID
    :param just_wrote: bounded deque of ``(relative_path, sha256)``
        entries recording writes the bind process itself performed; each
        event whose ``(rel, sha)`` pair appears in the deque is skipped
        so disk-round-trip writes do not re-journal
    :ptype just_wrote: deque[tuple[str, str]]
    :param on_conflict: policy selecting event-handling semantics;
        defaults to :attr:`BindConflictPolicy.DISK_WINS`
    :ptype on_conflict: BindConflictPolicy
    :return: list of ``relative_path`` keys the batch mutated in L3
    :rtype: list[str]
    """
    # coalesce by relative path: if a file is both added and modified
    # inside the same batch we write once, using the disk state we
    # observe at handle-time (the last event wins naturally).
    coalesced: dict[str, Change] = {}
    for change, abs_path in batch:
        rel = _resolve_under_root(Path(abs_path), disk_root, resolved_root)
        if rel is None:
            continue
        coalesced[rel] = change

    action_create: Literal["create"] = "create"
    action_update: Literal["update"] = "update"
    action_delete: Literal["delete"] = "delete"
    changed: list[str] = []
    if coalesced:
        now = datetime.now(UTC)
        async with db_pool.acquire() as conn:
            async with conn.transaction(namespace=workspace.namespace_name):
                max_version = 0
                for rel, change in coalesced.items():
                    if change is Change.deleted:
                        if on_conflict is BindConflictPolicy.L3_WINS:
                            # L3_WINS: external deletion is ignored. the
                            # file is still authoritative in L3 and will
                            # be re-projected onto disk on the next bind
                            # enter via atomic_write.
                            continue
                        # DISK_WINS delete-path: skip when nothing was
                        # there; otherwise emit a delete row.
                        head = await conn.fetchrow(
                            _SELECT_HEAD_SQL,
                            workspace.id,
                            rel,
                        )
                        if head is None:
                            continue
                        new_version = await _next_journal_version(
                            conn,
                            workspace.id,
                            rel,
                        )
                        if new_version > max_version:
                            max_version = new_version
                        await conn.execute(
                            _INSERT_WORKSPACE_FILE_VERSION_SQL,
                            uuid7(),
                            workspace.id,
                            rel,
                            new_version,
                            b"",
                            _sha256_bytes(b""),
                            action_delete,
                            None,
                            actor_id,
                            correlation_id,
                            now,
                        )
                        await conn.execute(
                            _DELETE_WORKSPACE_FILE_SQL,
                            workspace.id,
                            rel,
                        )
                        changed.append(rel)
                    else:
                        candidate = disk_root / rel
                        if not candidate.is_file():
                            # raced: file vanished before we could read.
                            # the next cycle will pick up the delete.
                            continue
                        content = await asyncio.to_thread(
                            _read_file_sync,
                            candidate,
                        )
                        sha = _sha256_bytes(content)
                        if (rel, sha) in just_wrote:
                            continue
                        head = await conn.fetchrow(
                            _SELECT_HEAD_SQL,
                            workspace.id,
                            rel,
                        )
                        current_sha = None if head is None else head["sha256"]
                        if current_sha == sha:
                            continue
                        if on_conflict is BindConflictPolicy.L3_WINS:
                            # L3_WINS added for path already in L3:
                            # skip (L3 holds truth; the agent would not
                            # have created this path via external means
                            # during the window, so any diff is either
                            # an echo beyond our just_wrote window or
                            # an external mutation we ignore).
                            # L3_WINS modified: skip (external
                            # modification discarded).
                            # L3_WINS added for NEW path (head is None):
                            # import as create -- net-new content the
                            # agent cannot have produced.
                            if change is Change.added and head is None:
                                new_version = await _next_journal_version(
                                    conn,
                                    workspace.id,
                                    rel,
                                )
                                if new_version > max_version:
                                    max_version = new_version
                                await conn.execute(
                                    _INSERT_WORKSPACE_FILE_VERSION_SQL,
                                    uuid7(),
                                    workspace.id,
                                    rel,
                                    new_version,
                                    content,
                                    sha,
                                    action_create,
                                    None,
                                    actor_id,
                                    correlation_id,
                                    now,
                                )
                                await conn.execute(
                                    _UPSERT_WORKSPACE_FILE_SQL,
                                    uuid7(),
                                    workspace.id,
                                    rel,
                                    content,
                                    sha,
                                    new_version,
                                    now,
                                )
                                changed.append(rel)
                            continue
                        # DISK_WINS: import every add / modify event.
                        new_version = await _next_journal_version(
                            conn,
                            workspace.id,
                            rel,
                        )
                        if new_version > max_version:
                            max_version = new_version
                        action_to_use = action_create if head is None else action_update
                        await conn.execute(
                            _INSERT_WORKSPACE_FILE_VERSION_SQL,
                            uuid7(),
                            workspace.id,
                            rel,
                            new_version,
                            content,
                            sha,
                            action_to_use,
                            None,
                            actor_id,
                            correlation_id,
                            now,
                        )
                        await conn.execute(
                            _UPSERT_WORKSPACE_FILE_SQL,
                            uuid7(),
                            workspace.id,
                            rel,
                            content,
                            sha,
                            new_version,
                            now,
                        )
                        changed.append(rel)
                if changed:
                    await conn.execute(
                        _UPDATE_WORKSPACE_VERSION_SQL,
                        max_version,
                        now,
                        workspace.id,
                    )
    return changed


_SELECT_HEAD_SQL = "SELECT content, sha256, version FROM workspace_files WHERE workspace_id = $1 AND relative_path = $2"


async def _watch_loop(
    *,
    workspace: Any,
    disk_root: Path,
    db_pool: Any,
    actor_id: UUID,
    correlation_id: UUID,
    just_wrote: deque[tuple[str, str]],
    on_conflict: BindConflictPolicy = BindConflictPolicy.DISK_WINS,
) -> None:
    """drive :func:`watchfiles.awatch` over ``disk_root`` while bind is open.

    each yielded batch is handed to :func:`_handle_watch_batch` along
    with the configured ``on_conflict`` policy. inner iteration
    exceptions are caught and logged at :meth:`log.exception` level so
    a single bad file (transient I/O, symlink cycle, DB blip) does not
    kill the task; the loop yields back to the event loop via
    :func:`asyncio.sleep` on the error path so a tight-spinning failure
    cannot starve other tasks. :class:`asyncio.CancelledError` cleanly
    exits the loop without re-raising; bind's teardown code already
    awaits the task with a bounded timeout and expects the task to
    return rather than raise :class:`CancelledError`.

    :param workspace: target workspace entity
    :ptype workspace: Any
    :param disk_root: absolute path to sandboxed bind root
    :ptype disk_root: Path
    :param db_pool: asyncpg pool supplying acquire + transaction
    :ptype db_pool: Any
    :param actor_id: identifier of actor running the bind window
    :ptype actor_id: UUID
    :param correlation_id: originating tool-call envelope identifier
    :ptype correlation_id: UUID
    :param just_wrote: bounded deque of ``(rel, sha)`` pairs recording
        writes the bind process itself performed
    :ptype just_wrote: deque[tuple[str, str]]
    :param on_conflict: policy forwarded to each batch handler
    :ptype on_conflict: BindConflictPolicy
    :return: None
    :rtype: None
    """
    resolved_root = disk_root.resolve()
    try:
        async for batch in awatch(disk_root, recursive=True):
            try:
                changed = await _handle_watch_batch(
                    batch=batch,
                    workspace=workspace,
                    disk_root=disk_root,
                    resolved_root=resolved_root,
                    db_pool=db_pool,
                    actor_id=actor_id,
                    correlation_id=correlation_id,
                    just_wrote=just_wrote,
                    on_conflict=on_conflict,
                )
                if changed:
                    log.info(
                        "workspace.watch.batch",
                        extra={
                            "workspace_id": str(workspace.id),
                            "changed_count": len(changed),
                        },
                    )
            # NOSILENT: per-batch failure must not kill the watcher task.
            # log at exception-level so SRE sees programmer-error
            # regressions; yield to the loop so a persistent failure
            # cannot starve other tasks.
            except Exception as iter_exc:
                log.exception(
                    "bind watcher iteration failed: %s",
                    iter_exc,
                )
                await asyncio.sleep(0)
    except asyncio.CancelledError:
        # bind's teardown cancels us on clean or exception-path exit;
        # returning instead of re-raising lets the awaiting wait_for
        # observe a completed task rather than a CancelledError that
        # would propagate into bind's outer frame.
        return


async def _capture_back(
    *,
    workspace: Any,
    disk_root: Path,
    snapshot: dict[str, tuple[str, int]],
    workspace_file_collection: WorkspaceFileCollection,
    workspace_file_version_collection: WorkspaceFileVersionCollection,
    workspace_collection: WorkspaceCollection,
    db_pool: Any,
    actor_id: UUID,
    correlation_id: UUID,
    nats_client: Any = None,
    namespace: str | None = None,
    root_name: str = "bind",
) -> list[str]:
    """write disk state changes relative to ``snapshot`` into L3.

    compares the on-enter snapshot (``{relpath: (sha256, version)}``) to
    the current disk state and emits, inside a single asyncpg
    transaction:

    - ``create`` journal row + head upsert for files new on disk
    - ``update`` journal row + head upsert for files whose sha changed
    - ``delete`` journal row + head delete for files missing from disk
    - ``workspaces.current_version`` bump (GREATEST) + ``date_updated``
      refresh when any of the above fired

    when no changes are detected the function returns an empty list
    without opening a transaction.

    new-file version starts at 1; updated-file version is
    ``snapshot[relpath].version + 1``; deleted-file version is likewise
    ``snapshot[relpath].version + 1``. the ``correlation_id`` and
    ``actor_id`` are stamped on every journal row so the capture phase
    is traceable back to the bind caller.

    :param workspace: target workspace entity whose pointer to advance
    :ptype workspace: Any
    :param disk_root: absolute path to the sandboxed bind root
    :ptype disk_root: Path
    :param snapshot: on-enter mapping ``{relative_path: (sha256, version)}``
    :ptype snapshot: dict[str, tuple[str, int]]
    :param workspace_file_collection: head-state collection, unused
        here but accepted for signature symmetry with the public tools
    :ptype workspace_file_collection: WorkspaceFileCollection
    :param workspace_file_version_collection: journal collection,
        unused here but accepted for signature symmetry
    :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
    :param workspace_collection: workspace collection, unused here but
        accepted for signature symmetry
    :ptype workspace_collection: WorkspaceCollection
    :param db_pool: asyncpg pool supplying acquire + transaction
    :ptype db_pool: Any
    :param actor_id: identifier of the actor that owned the bind window
    :ptype actor_id: UUID
    :param correlation_id: identifier of the originating tool-call envelope
    :ptype correlation_id: UUID
    :return: list of ``relative_path`` keys whose L3 rows were written
    :rtype: list[str]
    """
    del workspace_file_collection
    del workspace_file_version_collection
    del workspace_collection

    disk = await _snapshot_disk(disk_root)

    creates: list[tuple[str, bytes, str]] = []
    updates: list[tuple[str, bytes, str]] = []
    deletes: list[str] = []

    for rel, (content, sha) in disk.items():
        prior = snapshot.get(rel)
        if prior is None:
            creates.append((rel, content, sha))
        elif prior[0] != sha:
            updates.append((rel, content, sha))

    for rel in snapshot:
        if rel not in disk:
            deletes.append(rel)

    changed: list[str] = []
    change_kinds: dict[str, str] = {}
    if creates or updates or deletes:
        now = datetime.now(UTC)
        max_version = 0
        action_create: Literal["create"] = "create"
        action_update: Literal["update"] = "update"
        action_delete: Literal["delete"] = "delete"
        async with db_pool.acquire() as conn:
            async with conn.transaction(namespace=workspace.namespace_name):
                for rel, content, sha in creates:
                    # derive version from journal so re-creating a path
                    # that was previously bind-deleted does not collide
                    # on (workspace_id, relative_path, version).
                    new_version = await _next_journal_version(conn, workspace.id, rel)
                    max_version = max(max_version, new_version)
                    await conn.execute(
                        _INSERT_WORKSPACE_FILE_VERSION_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        new_version,
                        content,
                        sha,
                        action_create,
                        None,
                        actor_id,
                        correlation_id,
                        now,
                    )
                    await conn.execute(
                        _UPSERT_WORKSPACE_FILE_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        content,
                        sha,
                        new_version,
                        now,
                    )
                    changed.append(rel)
                    change_kinds[rel] = "create"

                for rel, content, sha in updates:
                    # derive version from the journal rather than from
                    # snapshot[rel].version + 1. the live watcher spawned
                    # inside the bind window may already have committed
                    # journal rows for this path, so the snapshot's
                    # version is no longer a safe basis for "prior + 1"
                    # -- a naive bump would collide with the watcher's
                    # INSERT on (workspace_id, relative_path, version).
                    new_version = await _next_journal_version(
                        conn,
                        workspace.id,
                        rel,
                    )
                    max_version = max(max_version, new_version)
                    await conn.execute(
                        _INSERT_WORKSPACE_FILE_VERSION_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        new_version,
                        content,
                        sha,
                        action_update,
                        None,
                        actor_id,
                        correlation_id,
                        now,
                    )
                    await conn.execute(
                        _UPSERT_WORKSPACE_FILE_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        content,
                        sha,
                        new_version,
                        now,
                    )
                    changed.append(rel)
                    change_kinds[rel] = "update"

                for rel in deletes:
                    # same watcher-collision rationale as updates above.
                    new_version = await _next_journal_version(
                        conn,
                        workspace.id,
                        rel,
                    )
                    max_version = max(max_version, new_version)
                    await conn.execute(
                        _INSERT_WORKSPACE_FILE_VERSION_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        new_version,
                        b"",
                        _sha256_bytes(b""),
                        action_delete,
                        None,
                        actor_id,
                        correlation_id,
                        now,
                    )
                    await conn.execute(
                        _DELETE_WORKSPACE_FILE_SQL,
                        workspace.id,
                        rel,
                    )
                    changed.append(rel)
                    change_kinds[rel] = "delete"

                await conn.execute(
                    _UPDATE_WORKSPACE_VERSION_SQL,
                    max_version,
                    now,
                    workspace.id,
                )
        # defense-in-depth: publish one additive audit event per
        # changed file on top of the baseline ``tool.call`` emitted
        # by ToolServer when the caller is inside a tool dispatch.
        # outside the transaction; audit is best-effort and must not
        # undo committed capture-back on transient NATS failure.
        #
        # WS-ACL-10: the envelope carries the identity tuple. the
        # scope is the authoritative source for actor_user_id /
        # calling_agent_id / customer_id; bind() is called inside a
        # dispatch so scope is set. owner_agent_id + resource
        # namespace id come from the workspace entity. when scope is
        # unavailable (e.g. unit tests invoking _capture_back
        # directly), the helper RuntimeError propagates into the
        # outer swallow.
        try:
            if namespace is not None and nats_client is not None and changed:
                from threetears.agent.tools.call_scope import (
                    current_scope as _current_scope,
                )

                _scope = _current_scope()
                if _scope is None:
                    raise RuntimeError(
                        "workspace.materialize audit: no ToolCallScope "
                        "installed; bind must run under enter_call_scope."
                    )
                _ctx = _scope.context
                if (
                    _ctx.user_id is None
                    or _ctx.agent_id is None
                    or _ctx.customer_id is None
                ):
                    raise RuntimeError(
                        "workspace.materialize audit: scope missing "
                        "identity dimension; cannot publish envelope."
                    )
                for rel in changed:
                    event = AuditEvent(
                        id=uuid7(),
                        timestamp=datetime.now(UTC),
                        event_type="workspace.materialize",
                        actor_user_id=_ctx.user_id,
                        calling_agent_id=_ctx.agent_id,
                        owner_agent_id=workspace.owner_agent_id,
                        customer_id=_ctx.customer_id,
                        resource_namespace_id=workspace.id,
                        resource_namespace_type="workspace_file",
                        action="materialize",
                        outcome="success",
                        correlation_id=correlation_id,
                        details={
                            "workspace_resource_id": f"{workspace.id}/{rel}",
                            "root_name": root_name,
                            "change_kind": change_kinds.get(rel, "update"),
                        },
                    )
                    await publish_audit(
                        event,
                        nats_client=nats_client,
                        namespace=namespace,
                    )
        # NOSILENT: audit failure must never taint the successful
        # capture-back commit above. log at exception-level so SRE
        # still sees programmer-error regressions (AttributeError,
        # NameError) — only NATS-publish failure is already
        # swallowed inside publish_audit at WARN.
        except Exception as audit_exc:
            log.exception(
                "workspace.materialize audit publish swallow caught: %s",
                audit_exc,
            )
    return changed


@asynccontextmanager
async def bind(
    *,
    workspace_id: UUID,
    sandbox: WorkspaceSandbox,
    lease: WorkspaceFileLease,
    workspace_collection: WorkspaceCollection,
    workspace_file_collection: WorkspaceFileCollection,
    workspace_file_version_collection: WorkspaceFileVersionCollection,
    db_pool: Any,
    actor_id: UUID,
    correlation_id: UUID,
    root_name: str = "bind",
    lease_ttl_seconds: int = 30,
    lease_max_wait_seconds: int = 60,
    nats_client: Any = None,
    namespace: str | None = None,
    on_conflict: BindConflictPolicy = BindConflictPolicy.DISK_WINS,
) -> AsyncIterator[Path]:
    """async context manager that sync L3 -> disk, yields path, captures on clean exit.

    resolves ``disk_root`` through
    :meth:`WorkspaceSandbox.resolve_fs_path(workspace.name, root_name)`,
    acquires a :class:`WorkspaceFileLease` scoped to
    ``"bind:{root_name}"`` so two pods binding the same workspace at the
    same root serialize cleanly, writes every head-state file from L3
    onto disk via :func:`atomic_write`, snapshots ``{relpath: (sha256,
    version)}``, yields ``disk_root``, and on clean exit captures disk
    changes back to L3 in one transaction through :func:`_capture_back`.

    on body exception the lease releases through the ``async with``
    scope but capture-back is skipped by design -- disk state after a
    crashed body is suspect and the partial progress is intentionally
    lost. use :func:`recover` after crash if the operator judges disk
    state to be good.

    the named root must be configured on the sandbox (a
    :class:`KeyError` on the ``fs_roots`` dict propagates out
    unchanged). absolute or escape paths on ``workspace.name`` raise
    :class:`SandboxDenied` through
    :meth:`PathSandbox.resolve_fs_path`.

    :param workspace_id: identifier of workspace to bind
    :ptype workspace_id: UUID
    :param sandbox: workspace sandbox carrying the named fs roots
    :ptype sandbox: WorkspaceSandbox
    :param lease: workspace file lease wrapper over NATS KV
    :ptype lease: WorkspaceFileLease
    :param workspace_collection: workspace collection for id lookup
    :ptype workspace_collection: WorkspaceCollection
    :param workspace_file_collection: head-state file collection
    :ptype workspace_file_collection: WorkspaceFileCollection
    :param workspace_file_version_collection: journal collection
    :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
    :param db_pool: asyncpg pool supplying acquire + transaction
    :ptype db_pool: Any
    :param actor_id: identifier of actor running the bind window
    :ptype actor_id: UUID
    :param correlation_id: originating tool-call envelope identifier
    :ptype correlation_id: UUID
    :param root_name: sandbox named root to bind under; lease key is
        ``"bind:{root_name}"`` so multiple bind roots on one agent use
        independent leases
    :ptype root_name: str
    :param lease_ttl_seconds: lease TTL forwarded to :meth:`acquire`
    :ptype lease_ttl_seconds: int
    :param lease_max_wait_seconds: max total wait forwarded to :meth:`acquire`
    :ptype lease_max_wait_seconds: int
    :param on_conflict: policy governing L3 vs disk authority during
        the bind window; controls both seed-on-enter strategy and
        live-watcher event handling; defaults to
        :attr:`BindConflictPolicy.DISK_WINS`
    :ptype on_conflict: BindConflictPolicy
    :return: async context manager yielding the sandboxed disk root
    :rtype: AsyncIterator[Path]
    :raises ValueError: if ``workspace_id`` does not resolve to a live workspace
    :raises KeyError: if ``root_name`` is not configured on sandbox
    :raises SandboxDenied: if ``workspace.name`` escapes the named root
    """
    workspace = await workspace_collection.find_by_id(workspace_id)
    if workspace is None:
        raise ValueError(f"workspace {workspace_id} not found or is soft-deleted")
    disk_root = sandbox.resolve_fs_path(workspace.name, root_name)
    disk_root.mkdir(parents=True, exist_ok=True)

    handle = await lease.acquire(
        workspace_id,
        f"bind:{root_name}",
        ttl_seconds=lease_ttl_seconds,
        max_wait_seconds=lease_max_wait_seconds,
    )
    async with handle:
        # step 1: seed L3 from disk under the conflict policy. L3_WINS
        # runs the historical "import if empty" gate; DISK_WINS walks
        # disk unconditionally and mirrors create/update/delete rows.
        # the gate re-queries find_by_workspace so the L3_WINS path
        # does not re-import on a populated workspace.
        await _seed_l3_from_disk(
            workspace=workspace,
            disk_root=disk_root,
            workspace_file_collection=workspace_file_collection,
            db_pool=db_pool,
            actor_id=actor_id,
            correlation_id=correlation_id,
            on_conflict=on_conflict,
        )
        # step 2: sync L3 -> disk (after import, this is a no-op on a
        # freshly-empty workspace because L3 now matches disk; on a
        # pre-populated workspace it projects head state over whatever
        # disk state happens to be present, per the bind contract).
        files = await workspace_file_collection.find_by_workspace(workspace_id)
        snapshot: dict[str, tuple[str, int]] = {}
        just_wrote: deque[tuple[str, str]] = deque(maxlen=256)
        for file_entity in files:
            target = disk_root / file_entity.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            await atomic_write(target, file_entity.content)
            just_wrote.append(
                (file_entity.relative_path, file_entity.sha256),
            )
            snapshot[file_entity.relative_path] = (
                file_entity.sha256,
                file_entity.version,
            )
        log.info(
            "workspace.bind.enter",
            extra={
                "workspace_id": str(workspace_id),
                "root_name": root_name,
                "disk_root": str(disk_root),
                "file_count": len(files),
            },
        )
        # step 3: spawn watcher task so external writes landing on disk
        # during the bind window mirror into L3 in real time. the task
        # owns its own awatch generator; cancellation on exit is
        # bounded by asyncio.wait_for below.
        watcher_task = asyncio.create_task(
            _watch_loop(
                workspace=workspace,
                disk_root=disk_root,
                db_pool=db_pool,
                actor_id=actor_id,
                correlation_id=correlation_id,
                just_wrote=just_wrote,
                on_conflict=on_conflict,
            ),
            name=f"workspace.bind.watch:{workspace_id.hex}:{root_name}",
        )
        body_raised: BaseException | None = None
        try:
            yield disk_root
        except BaseException as body_exc:
            body_raised = body_exc
        finally:
            # tear the watcher down deterministically whether or not the
            # body raised; the bounded wait_for guarantees bind exit does
            # not hang on a stuck filesystem watcher.
            watcher_task.cancel()
            try:
                await asyncio.wait_for(watcher_task, timeout=2.0)
            except TimeoutError:
                log.warning(
                    "workspace.bind.watcher_cancel_timeout",
                    extra={
                        "workspace_id": str(workspace_id),
                        "root_name": root_name,
                    },
                )
            # NOSILENT: CancelledError on the awaited task is expected;
            # any other exception is programmer error we want visible.
            except asyncio.CancelledError:
                pass
            except Exception as cancel_exc:
                log.exception(
                    "workspace.bind.watcher_cancel_error: %s",
                    cancel_exc,
                )
        if body_raised is not None:
            log.warning(
                "workspace.bind.exception_skip_capture",
                extra={
                    "workspace_id": str(workspace_id),
                    "root_name": root_name,
                },
            )
            raise body_raised
        changed = await _capture_back(
            workspace=workspace,
            disk_root=disk_root,
            snapshot=snapshot,
            workspace_file_collection=workspace_file_collection,
            workspace_file_version_collection=workspace_file_version_collection,
            workspace_collection=workspace_collection,
            db_pool=db_pool,
            actor_id=actor_id,
            correlation_id=correlation_id,
            nats_client=nats_client,
            namespace=namespace,
            root_name=root_name,
        )
        log.info(
            "workspace.bind.capture",
            extra={
                "workspace_id": str(workspace_id),
                "root_name": root_name,
                "changed_count": len(changed),
            },
        )


async def recover(
    *,
    workspace_id: UUID,
    sandbox: WorkspaceSandbox,
    workspace_collection: WorkspaceCollection,
    workspace_file_collection: WorkspaceFileCollection,
    workspace_file_version_collection: WorkspaceFileVersionCollection,
    db_pool: Any,
    actor_id: UUID,
    correlation_id: UUID,
    root_name: str = "bind",
) -> list[str]:
    """capture-back escape hatch: write on-disk state to L3 without lease.

    used after a crashed bind when the operator inspects the disk root
    and decides the state is good. resolves ``disk_root`` the same way
    :func:`bind` does, walks the disk, and compares each file's sha256
    against the current L3 head row. only genuinely-different files are
    written; unchanged files are skipped. no lease is acquired because
    the operator opted in explicitly (the lease would be a foot-gun if
    the crashed bind still held it).

    the L3 head row sha256 is used as the comparison baseline (not an
    enter-snapshot, which recover does not have). files present on disk
    but absent from L3 are written as ``create`` at version 1. files on
    disk whose sha matches L3 are no-ops. files absent from disk are
    NOT deleted -- recover is additive; deleting files from L3 requires
    a deliberate action through the tool layer.

    :param workspace_id: identifier of workspace to recover
    :ptype workspace_id: UUID
    :param sandbox: workspace sandbox carrying the named fs roots
    :ptype sandbox: WorkspaceSandbox
    :param workspace_collection: workspace collection for id lookup
    :ptype workspace_collection: WorkspaceCollection
    :param workspace_file_collection: head-state file collection
    :ptype workspace_file_collection: WorkspaceFileCollection
    :param workspace_file_version_collection: journal collection
    :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
    :param db_pool: asyncpg pool supplying acquire + transaction
    :ptype db_pool: Any
    :param actor_id: identifier of actor running recovery
    :ptype actor_id: UUID
    :param correlation_id: correlation id stamped on journal rows
    :ptype correlation_id: UUID
    :param root_name: sandbox named root to recover from
    :ptype root_name: str
    :return: list of ``relative_path`` keys whose L3 rows were written
    :rtype: list[str]
    :raises ValueError: if ``workspace_id`` does not resolve to a live workspace
    :raises KeyError: if ``root_name`` is not configured on sandbox
    :raises SandboxDenied: if ``workspace.name`` escapes the named root
    """
    workspace = await workspace_collection.find_by_id(workspace_id)
    if workspace is None:
        raise ValueError(f"workspace {workspace_id} not found or is soft-deleted")
    disk_root = sandbox.resolve_fs_path(workspace.name, root_name)

    head_rows = await workspace_file_collection.find_by_workspace(workspace_id)
    head_by_path: dict[str, tuple[str, int]] = {row.relative_path: (row.sha256, row.version) for row in head_rows}

    disk = await _snapshot_disk(disk_root)

    creates: list[tuple[str, bytes, str]] = []
    updates: list[tuple[str, bytes, str]] = []

    for rel, (content, sha) in disk.items():
        prior = head_by_path.get(rel)
        if prior is None:
            creates.append((rel, content, sha))
        elif prior[0] != sha:
            updates.append((rel, content, sha))

    changed: list[str] = []
    if creates or updates:
        now = datetime.now(UTC)
        max_version = 0
        action_create: Literal["create"] = "create"
        action_update: Literal["update"] = "update"
        async with db_pool.acquire() as conn:
            async with conn.transaction(namespace=workspace.namespace_name):
                for rel, content, sha in creates:
                    # derive version from journal, not the head cache,
                    # so re-created paths never collide with deleted
                    # history on the unique-constraint triple.
                    new_version = await _next_journal_version(conn, workspace.id, rel)
                    max_version = max(max_version, new_version)
                    await conn.execute(
                        _INSERT_WORKSPACE_FILE_VERSION_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        new_version,
                        content,
                        sha,
                        action_create,
                        None,
                        actor_id,
                        correlation_id,
                        now,
                    )
                    await conn.execute(
                        _UPSERT_WORKSPACE_FILE_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        content,
                        sha,
                        new_version,
                        now,
                    )
                    changed.append(rel)

                for rel, content, sha in updates:
                    # derive from journal for the same reason as
                    # _capture_back: recover runs without the bind lease,
                    # so a concurrent bind on another pod could bump the
                    # journal in parallel; the unique constraint would
                    # reject a naive prior_version+1 INSERT.
                    new_version = await _next_journal_version(
                        conn,
                        workspace.id,
                        rel,
                    )
                    max_version = max(max_version, new_version)
                    await conn.execute(
                        _INSERT_WORKSPACE_FILE_VERSION_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        new_version,
                        content,
                        sha,
                        action_update,
                        None,
                        actor_id,
                        correlation_id,
                        now,
                    )
                    await conn.execute(
                        _UPSERT_WORKSPACE_FILE_SQL,
                        uuid7(),
                        workspace.id,
                        rel,
                        content,
                        sha,
                        new_version,
                        now,
                    )
                    changed.append(rel)

                await conn.execute(
                    _UPDATE_WORKSPACE_VERSION_SQL,
                    max_version,
                    now,
                    workspace.id,
                )
    log.info(
        "workspace.recover.done",
        extra={
            "workspace_id": str(workspace_id),
            "root_name": root_name,
            "changed_count": len(changed),
        },
    )
    return changed
