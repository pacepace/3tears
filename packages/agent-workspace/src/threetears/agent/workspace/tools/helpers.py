"""shared helpers for fs_*, doc_*, history/diff/rollback workspace tools.

three primitives live here:

- :func:`_resolve_workspace` -- turn an optional ``workspace`` kwarg into
  a live :class:`Workspace` entity (explicit name, or the conversation's
  pin). raises typed exceptions so each tool's ``execute`` can translate
  them into ``ToolResult(success=False, ...)`` without its own glue.
- :func:`_write_file_atomic` -- the three-row transaction (journal +
  head-state + workspace version pointer) that every byte-level write
  tool shares. enforces optimistic concurrency via ``expected_sha256``
  by reading the current head inside the same transaction the writes
  run in, so read-and-write share SERIALIZABLE semantics and a racing
  update is caught cleanly.
- :func:`_resolve_ref` -- translate the history-tool ``ref`` vocabulary
  (``"head"``, integer version, checkpoint label) into a concrete journal
  row. returns a plain row ``dict`` (mirrors asyncpg's ``Record`` shape)
  rather than a hydrated entity so the caller already holds an open
  connection; ``None`` means "not found", which diff/rollback treat as
  clean-error and skip-file respectively.

no tool class lives here; this module is import-cheap and side-effect
free so test modules and sibling tool modules can pull in just what they
need.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid7

if TYPE_CHECKING:
    from threetears.agent.tools.context import ToolContextManager

    from threetears.agent.workspace.collections import (
        WorkspaceCollection,
        WorkspaceFileCollection,
        WorkspaceFileVersionCollection,
    )
    from threetears.agent.workspace.config import ValidatorEntry
    from threetears.agent.workspace.entities import Workspace

from threetears.agent.workspace import pin as _pin
from threetears.agent.workspace.validators import dispatch_validators


def _resolve_validators(
    deps: dict[str, Any],
) -> list[ValidatorEntry] | None:
    """
    extract validator entries from the factory dependency bundle.

    every write-class tool's ``_build`` calls this so the factory and
    direct-construct code paths stay in lockstep. precedence:

    1. explicit ``validators`` kwarg wins when present (tests and
       callers wiring dispatch directly).
    2. otherwise falls back to ``config.validators`` when ``config`` is
       a :class:`WorkspaceConfig` (or duck-typed equivalent exposing a
       ``.validators`` sequence).

    returns ``None`` when neither source is available so
    :func:`_write_file_atomic` can short-circuit the dispatch step
    entirely on workspaces that haven't declared any validators.

    :param deps: factory dependency bundle (``**kwargs`` passed to each
        registered ``_build``)
    :ptype deps: dict[str, Any]
    :return: list of validator entries or None
    :rtype: list[ValidatorEntry] | None
    """
    result: list[ValidatorEntry] | None = None
    explicit = deps.get("validators")
    if explicit is not None:
        result = list(explicit) if explicit else None
    else:
        config = deps.get("config")
        if config is not None:
            entries = getattr(config, "validators", None)
            if entries:
                result = list(entries)
    return result


class WorkspaceNotFound(ValueError):
    """raised by :func:`_resolve_workspace` when no live workspace matches.

    fires for an explicit name lookup miss, a pin whose workspace_id no
    longer resolves under the agent, or a resolved row whose
    ``date_deleted`` is set (soft-deleted workspaces are not valid
    targets for read or write through the fs_*/doc_* tools).
    """


class NoWorkspacePinned(LookupError):
    """raised by :func:`_resolve_workspace` when caller omitted ``workspace``
    and no workspace is pinned to the current conversation.
    """


class Sha256Mismatch(RuntimeError):
    """raised inside :func:`_write_file_atomic` when OCC check fails.

    carries ``expected`` (the ``expected_sha256`` the caller supplied) and
    ``current`` (the sha256 actually present on the head row at read
    time, or ``None`` when caller expected the file to be absent). tool
    wrappers translate the pair into a clean agent-visible error so the
    LLM can re-read and retry.

    :ivar expected: sha256 hex digest caller supplied, or None when
        caller supplied None (no OCC attempted -- in that case this
        exception is never raised)
    :ivar current: sha256 hex digest currently on head row, or None when
        no head row exists for the path
    """

    def __init__(self, expected: str | None, current: str | None) -> None:
        """capture the expected/current pair and build a readable message.

        :param expected: sha256 hex digest caller supplied
        :ptype expected: str | None
        :param current: sha256 hex digest currently on head row, or None
        :ptype current: str | None
        :return: None
        :rtype: None
        """
        self.expected = expected
        self.current = current
        super().__init__(
            f"sha256 mismatch: expected {expected!r}, current {current!r}"
        )


async def _resolve_workspace(
    workspace_arg: str | None,
    context: ToolContextManager,
    workspace_collection: WorkspaceCollection,
    agent_id: UUID,
) -> Workspace:
    """resolve an optional ``workspace`` kwarg to a live workspace entity.

    precedence:

    1. non-empty ``workspace_arg`` -- look up by ``(agent_id, name)``.
    2. otherwise -- read the pin via :func:`pin.get_pin` and resolve
       by the pinned ``workspace_id`` under the agent (so a rename of
       the pinned workspace still resolves correctly).

    soft-deleted workspaces (``date_deleted is not None``) are treated as
    not found; fs_*/doc_* tools must not operate on tombstones.

    :param workspace_arg: explicit workspace name from tool kwargs, or None
    :ptype workspace_arg: str | None
    :param context: conversation-scoped context manager for pin lookup
    :ptype context: ToolContextManager
    :param workspace_collection: collection providing
        ``find_by_agent_and_name`` and ``find_by_id_and_agent``
    :ptype workspace_collection: WorkspaceCollection
    :param agent_id: identifier of agent owning the workspace
    :ptype agent_id: UUID
    :return: matched live workspace entity
    :rtype: Workspace
    :raises WorkspaceNotFound: if name lookup misses, pin points at a
        workspace no longer resolvable, or the resolved row is
        soft-deleted
    :raises NoWorkspacePinned: if caller omitted ``workspace_arg`` and
        no pin exists on the current conversation
    """
    result: Workspace | None
    if workspace_arg:
        result = await workspace_collection.find_by_agent_and_name(
            agent_id, workspace_arg
        )
        if result is None:
            raise WorkspaceNotFound(
                f"workspace {workspace_arg!r} not found"
            )
    else:
        snapshot = await _pin.get_pin(context)
        if snapshot is None:
            raise NoWorkspacePinned(
                "no workspace pinned; call workspace.use(name) first"
            )
        result = await workspace_collection.find_by_id_and_agent(
            snapshot.workspace_id, agent_id
        )
        if result is None:
            raise WorkspaceNotFound(
                f"pinned workspace {snapshot.workspace_name!r} not found"
            )
    if result.date_deleted is not None:
        raise WorkspaceNotFound(
            f"workspace {result.name!r} is deleted"
        )
    return result


_SELECT_HEAD_SQL = (
    "SELECT content, sha256, version "
    "FROM workspace_files "
    "WHERE workspace_id = $1 AND relative_path = $2"
)

_SELECT_JOURNAL_MAX_VERSION_SQL = (
    "SELECT COALESCE(MAX(version), 0) AS max_version "
    "FROM workspace_file_versions "
    "WHERE workspace_id = $1 AND relative_path = $2"
)


async def _next_journal_version(
    conn: Any, workspace_id: UUID, relative_path: str,
) -> int:
    """
    compute the next journal version number for a workspace-relative path.

    scans the journal (not the head cache) so paths that were deleted
    from head by bind capture-back still get monotonically increasing
    version numbers on re-create. this prevents unique-constraint
    collisions on ``(workspace_id, relative_path, version)`` when the
    head row has been removed but journal history persists.

    :param conn: asyncpg connection enlisted in the caller's transaction
    :ptype conn: Any
    :param workspace_id: target workspace identifier
    :ptype workspace_id: UUID
    :param relative_path: workspace-relative path
    :ptype relative_path: str
    :return: next version number to use for an insert on this path
    :rtype: int
    """
    row = await conn.fetchrow(
        _SELECT_JOURNAL_MAX_VERSION_SQL, workspace_id, relative_path
    )
    max_version = 0 if row is None else int(row["max_version"])
    return max_version + 1

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

_UPDATE_WORKSPACE_VERSION_SQL = """
UPDATE workspaces
SET current_version = GREATEST(current_version, $1),
    date_updated = $2
WHERE id = $3
"""


async def _write_file_atomic(
    *,
    db_pool: Any,
    workspace: Workspace,
    relative_path: str,
    content: bytes,
    action: Literal["create", "update", "delete", "revert"],
    actor_id: UUID,
    correlation_id: UUID,
    expected_sha256: str | None,
    workspace_file_collection: WorkspaceFileCollection,
    workspace_file_version_collection: WorkspaceFileVersionCollection,
    workspace_collection: WorkspaceCollection,
    validators: list[ValidatorEntry] | None = None,
) -> tuple[int, str]:
    """write a workspace file atomically; return ``(new_version, new_sha256)``.

    the three-row transaction every byte-level write tool shares:

    1. SELECT current head row inside the transaction (read-and-write
       in one tx so OCC is actually serializable -- doing the read via
       a collection outside this transaction would let a racing writer
       slip in between).
    2. OCC check: if ``expected_sha256 is not None`` and the current sha
       does not match, raise :class:`Sha256Mismatch` (asyncpg rolls back
       on exception).
    3. VALIDATE: if ``validators`` is non-empty, dispatch each entry
       that matches ``relative_path`` via
       :func:`dispatch_validators`. a validation failure raises
       :class:`WorkspaceValidationError` which aborts the transaction
       before any INSERT/UPSERT (asyncpg rolls back on exception). the
       validator step runs inside the transaction to guarantee no
       journal row, head row, or workspace version bump is committed
       when content is rejected.
    4. INSERT new journal row at ``new_version = existing.version + 1``
       (or 1 when no row exists).
    5. UPSERT head-state row on ``(workspace_id, relative_path)``.
    6. UPDATE workspace ``current_version = GREATEST(current, new)`` so
       concurrent writers on different paths do not regress the pointer.

    collection parameters are accepted for signature symmetry with
    sibling tool helpers; the actual writes bypass the collection cache
    and go through the connection so every statement sits in the same
    transaction. L2 cache warms on the next read.

    ``validators`` takes a list directly (not the full
    :class:`WorkspaceConfig`) because the helper otherwise has no use
    for the config surface; tool wrappers pull the list off their
    injected config once at construct time and pass it through on every
    write. a ``None`` or empty list is a no-op.

    :param db_pool: asyncpg pool (or pool-like) supplying acquire+transaction
    :ptype db_pool: Any
    :param workspace: target workspace entity
    :ptype workspace: Workspace
    :param relative_path: workspace-relative path being written
    :ptype relative_path: str
    :param content: new file content bytes
    :ptype content: bytes
    :param action: journal action verb recorded on the new version row
    :ptype action: Literal["create", "update", "delete", "revert"]
    :param actor_id: identifier of actor performing the write
    :ptype actor_id: UUID
    :param correlation_id: identifier of originating tool-call envelope
    :ptype correlation_id: UUID
    :param expected_sha256: prior sha256 caller expects on head row, or
        None to skip the OCC check
    :ptype expected_sha256: str | None
    :param workspace_file_collection: head-state file collection
    :ptype workspace_file_collection: WorkspaceFileCollection
    :param workspace_file_version_collection: journal collection
    :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
    :param workspace_collection: workspace collection
    :ptype workspace_collection: WorkspaceCollection
    :param validators: optional list of per-pattern validator entries
        resolved from ``WorkspaceConfig.validators``; first matching
        failure aborts the write
    :ptype validators: list[ValidatorEntry] | None
    :return: pair of (new_version, new_sha256_hex)
    :rtype: tuple[int, str]
    :raises Sha256Mismatch: if OCC check fails
    :raises WorkspaceValidationError: if a matching validator rejects
        ``content``
    """
    # parameters present for signature symmetry with sibling write
    # helpers and to document dependency flow; direct conn usage keeps
    # every write in one transaction.
    del workspace_file_collection
    del workspace_file_version_collection
    del workspace_collection

    new_sha256 = hashlib.sha256(content).hexdigest()
    now = datetime.now(UTC)

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            head = await conn.fetchrow(
                _SELECT_HEAD_SQL, workspace.id, relative_path
            )
            current_sha: str | None = None if head is None else head["sha256"]
            if expected_sha256 is not None and current_sha != expected_sha256:
                raise Sha256Mismatch(
                    expected=expected_sha256, current=current_sha
                )
            if validators:
                dispatch_validators(validators, relative_path, content)
            # derive next version from the journal, not the head row:
            # bind's capture-back may DELETE the head row on "delete"
            # action while leaving the journal intact. a later re-create
            # at the same path would otherwise collide on the
            # (workspace_id, relative_path, version) unique constraint.
            new_version = await _next_journal_version(
                conn, workspace.id, relative_path
            )
            await conn.execute(
                _INSERT_WORKSPACE_FILE_VERSION_SQL,
                uuid7(),
                workspace.id,
                relative_path,
                new_version,
                content,
                new_sha256,
                action,
                None,
                actor_id,
                correlation_id,
                now,
            )
            await conn.execute(
                _UPSERT_WORKSPACE_FILE_SQL,
                uuid7(),
                workspace.id,
                relative_path,
                content,
                new_sha256,
                new_version,
                now,
            )
            await conn.execute(
                _UPDATE_WORKSPACE_VERSION_SQL,
                new_version,
                now,
                workspace.id,
            )
    return new_version, new_sha256


_SELECT_LATEST_VERSION_ROW_SQL = (
    "SELECT id, workspace_id, relative_path, version, content, sha256, "
    "action, label, actor_id, correlation_id, date_created "
    "FROM workspace_file_versions "
    "WHERE workspace_id = $1 AND relative_path = $2 "
    "ORDER BY version DESC LIMIT 1"
)

_SELECT_VERSION_BY_NUMBER_SQL = (
    "SELECT id, workspace_id, relative_path, version, content, sha256, "
    "action, label, actor_id, correlation_id, date_created "
    "FROM workspace_file_versions "
    "WHERE workspace_id = $1 AND relative_path = $2 AND version = $3"
)

_SELECT_CHECKPOINT_ROW_SQL = (
    "SELECT id, workspace_id, relative_path, version, content, sha256, "
    "action, label, actor_id, correlation_id, date_created "
    "FROM workspace_file_versions "
    "WHERE workspace_id = $1 AND relative_path = $2 "
    "AND action = 'checkpoint' AND label = $3 "
    "ORDER BY version DESC LIMIT 1"
)


async def _resolve_ref(
    conn: Any,
    workspace_id: UUID,
    relative_path: str,
    ref: str | int,
) -> dict[str, Any] | None:
    """resolve a history-tool ``ref`` against the journal for a single path.

    ``ref`` vocabulary:

    - ``"head"`` (string literal) -- newest journal row for
      ``(workspace_id, relative_path)``; picked via ``ORDER BY version
      DESC LIMIT 1`` rather than reading ``workspace_files`` so the same
      code path serves diff/rollback uniformly.
    - ``int`` (or digit-only string) -- exact version lookup on
      ``(workspace_id, relative_path, version)``.
    - any other string -- checkpoint label; matches rows with
      ``action='checkpoint'`` and ``label=ref`` for the same path. when
      no matching checkpoint exists for this path (e.g. the file did not
      exist when the checkpoint was taken), ``None`` is returned so
      rollback can skip and diff can emit a clean error.

    the row is returned as a plain ``dict[str, Any]`` (the callers
    already hold an open connection and need only the columns; hydrating
    an entity would force an extra cache path and make rollback's
    ``_write_file_atomic`` call awkward when the row comes from a
    checkpoint that never touched the head cache).

    :param conn: live asyncpg-like connection already enrolled in the
        caller's transaction (or a raw acquired connection for read-only
        callers)
    :ptype conn: Any
    :param workspace_id: identifier of workspace owning the file
    :ptype workspace_id: UUID
    :param relative_path: workspace-relative path whose journal to query
    :ptype relative_path: str
    :param ref: ``"head"``, integer version, digit-only string, or
        checkpoint label
    :ptype ref: str | int
    :return: journal row as dict, or None when no row matches
    :rtype: dict[str, Any] | None
    """
    result: dict[str, Any] | None
    row: Any
    if isinstance(ref, int):
        row = await conn.fetchrow(
            _SELECT_VERSION_BY_NUMBER_SQL, workspace_id, relative_path, ref
        )
    elif ref == "head":
        row = await conn.fetchrow(
            _SELECT_LATEST_VERSION_ROW_SQL, workspace_id, relative_path
        )
    elif ref.isdigit():
        row = await conn.fetchrow(
            _SELECT_VERSION_BY_NUMBER_SQL,
            workspace_id,
            relative_path,
            int(ref),
        )
    else:
        row = await conn.fetchrow(
            _SELECT_CHECKPOINT_ROW_SQL, workspace_id, relative_path, ref
        )
    if row is None:
        result = None
    else:
        result = dict(row)
    return result
