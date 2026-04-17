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
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal
from uuid import UUID, uuid7

from threetears.core.utils.atomic_write import atomic_write
from threetears.observe import get_logger

from threetears.agent.workspace import audit
from threetears.agent.workspace.tools.helpers import _next_journal_version

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
        except (OSError, ValueError):
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
    updates: list[tuple[str, bytes, str, int]] = []
    deletes: list[tuple[str, int]] = []

    for rel, (content, sha) in disk.items():
        prior = snapshot.get(rel)
        if prior is None:
            creates.append((rel, content, sha))
        elif prior[0] != sha:
            updates.append((rel, content, sha, prior[1]))

    for rel, (prior_sha, prior_version) in snapshot.items():
        if rel not in disk:
            deletes.append((rel, prior_version))

    changed: list[str] = []
    change_kinds: dict[str, str] = {}
    if creates or updates or deletes:
        now = datetime.now(UTC)
        max_version = 0
        action_create: Literal["create"] = "create"
        action_update: Literal["update"] = "update"
        action_delete: Literal["delete"] = "delete"
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                for rel, content, sha in creates:
                    # derive version from journal so re-creating a path
                    # that was previously bind-deleted does not collide
                    # on (workspace_id, relative_path, version).
                    new_version = await _next_journal_version(
                        conn, workspace.id, rel
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

                for rel, content, sha, prior_version in updates:
                    new_version = prior_version + 1
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

                for rel, prior_version in deletes:
                    new_version = prior_version + 1
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
        # defense-in-depth: publish one audit event per changed file.
        # outside the transaction; audit is best-effort and must not
        # undo committed capture-back on transient NATS failure.
        try:
            if namespace is not None and nats_client is not None:
                for rel in changed:
                    await audit.publish_workspace_event(
                        nats_client=nats_client,
                        namespace=namespace,
                        event_type="workspace.bind",
                        actor_id=actor_id,
                        agent_id=actor_id,
                        resource_type="workspace_file",
                        resource_id=f"{workspace.id}/{rel}",
                        action="bind",
                        details={
                            "root_name": root_name,
                            "change_kind": change_kinds.get(rel, "update"),
                        },
                        correlation_id=correlation_id,
                    )
        # NOSILENT: audit failure must never taint the successful
        # capture-back commit above. log at exception-level so SRE
        # still sees programmer-error regressions (AttributeError,
        # NameError) — only NATS-publish failure is already swallowed
        # inside publish_workspace_event at WARN.
        except Exception as audit_exc:
            log.exception(
                "workspace.bind audit publish swallow caught: %s",
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
    :return: async context manager yielding the sandboxed disk root
    :rtype: AsyncIterator[Path]
    :raises ValueError: if ``workspace_id`` does not resolve to a live workspace
    :raises KeyError: if ``root_name`` is not configured on sandbox
    :raises SandboxDenied: if ``workspace.name`` escapes the named root
    """
    workspace = await workspace_collection.find_by_id(workspace_id)
    if workspace is None:
        raise ValueError(
            f"workspace {workspace_id} not found or is soft-deleted"
        )
    disk_root = sandbox.resolve_fs_path(workspace.name, root_name)
    disk_root.mkdir(parents=True, exist_ok=True)

    handle = await lease.acquire(
        workspace_id,
        f"bind:{root_name}",
        ttl_seconds=lease_ttl_seconds,
        max_wait_seconds=lease_max_wait_seconds,
    )
    async with handle:
        files = await workspace_file_collection.find_by_workspace(workspace_id)
        snapshot: dict[str, tuple[str, int]] = {}
        for file_entity in files:
            target = disk_root / file_entity.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            await atomic_write(target, file_entity.content)
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
        try:
            yield disk_root
        except BaseException:
            log.warning(
                "workspace.bind.exception_skip_capture",
                extra={
                    "workspace_id": str(workspace_id),
                    "root_name": root_name,
                },
            )
            raise
        else:
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
        raise ValueError(
            f"workspace {workspace_id} not found or is soft-deleted"
        )
    disk_root = sandbox.resolve_fs_path(workspace.name, root_name)

    head_rows = await workspace_file_collection.find_by_workspace(workspace_id)
    head_by_path: dict[str, tuple[str, int]] = {
        row.relative_path: (row.sha256, row.version) for row in head_rows
    }

    disk = await _snapshot_disk(disk_root)

    creates: list[tuple[str, bytes, str]] = []
    updates: list[tuple[str, bytes, str, int]] = []

    for rel, (content, sha) in disk.items():
        prior = head_by_path.get(rel)
        if prior is None:
            creates.append((rel, content, sha))
        elif prior[0] != sha:
            updates.append((rel, content, sha, prior[1]))

    changed: list[str] = []
    if creates or updates:
        now = datetime.now(UTC)
        max_version = 0
        action_create: Literal["create"] = "create"
        action_update: Literal["update"] = "update"
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                for rel, content, sha in creates:
                    # derive version from journal, not the head cache,
                    # so re-created paths never collide with deleted
                    # history on the unique-constraint triple.
                    new_version = await _next_journal_version(
                        conn, workspace.id, rel
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

                for rel, content, sha, prior_version in updates:
                    new_version = prior_version + 1
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
