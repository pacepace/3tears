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

from threetears.agent.acl import AclCache

__all__ = [
    "NoWorkspacePinned",
    "Sha256Mismatch",
    "WorkspaceNotFound",
    "WorkspaceAuditIdentity",
    "authorize_workspace",
    "authorize_workspace_file",
    "enrich_workspace_identity",
    "workspace_audit_identity",
]


class WorkspaceAuditIdentity:
    """resolved identity fields needed to publish a workspace audit event.

    audit-task-01 Phase 3: fields map onto
    :class:`threetears.agent.audit.AuditEvent` as follows:
    ``actor_user_id`` -> ``actor_user_id``, ``calling_agent_id`` ->
    ``calling_agent_id``, ``owner_agent_id`` -> ``owner_agent_id``,
    ``customer_id`` -> ``customer_id``, ``namespace_id`` ->
    ``resource_namespace_id``. this plain value class carries the five
    UUIDs every tool's additive audit-publish block pulls from one
    place; the per-tool ``resource_namespace_type`` + domain-specific
    details are stamped at the call site.

    construction goes through :func:`workspace_audit_identity`, which
    reads the current :class:`ToolCallScope` and the resolved
    :class:`Workspace`; passing the value around keeps the publish call
    sites terse and uniform.

    :ivar actor_user_id: invoking user identifier (from
        ``scope.context.user_id``)
    :ivar calling_agent_id: agent whose process ran the tool (from
        ``scope.context.agent_id``)
    :ivar owner_agent_id: workspace owner agent (from
        ``workspace.owner_agent_id``)
    :ivar customer_id: owning customer (from ``scope.context.customer_id``)
    :ivar namespace_id: workspace id (shared PK with the
        ``platform.namespaces`` row); maps onto the
        :attr:`AuditEvent.resource_namespace_id` column at publish
        time
    """

    __slots__ = (
        "actor_user_id",
        "calling_agent_id",
        "owner_agent_id",
        "customer_id",
        "namespace_id",
    )

    def __init__(
        self,
        *,
        actor_user_id: UUID,
        calling_agent_id: UUID,
        owner_agent_id: UUID,
        customer_id: UUID,
        namespace_id: UUID,
    ) -> None:
        """
        bind the five audit-identity UUIDs as attributes.

        :param actor_user_id: invoking user's UUID
        :ptype actor_user_id: UUID
        :param calling_agent_id: calling agent's UUID
        :ptype calling_agent_id: UUID
        :param owner_agent_id: owning agent's UUID
        :ptype owner_agent_id: UUID
        :param customer_id: owning customer's UUID
        :ptype customer_id: UUID
        :param namespace_id: workspace/namespace UUID
        :ptype namespace_id: UUID
        :return: None
        :rtype: None
        """
        self.actor_user_id = actor_user_id
        self.calling_agent_id = calling_agent_id
        self.owner_agent_id = owner_agent_id
        self.customer_id = customer_id
        self.namespace_id = namespace_id


def workspace_audit_identity(workspace: Workspace) -> WorkspaceAuditIdentity:
    """build a :class:`WorkspaceAuditIdentity` from the current scope + workspace.

    WS-ACL-10: every workspace-mutating tool's audit publish carries the
    full five-UUID identity tuple. this helper pulls scope + workspace
    into the value object the :func:`publish_audit` caller forwards as
    :class:`AuditEvent` fields. every required dimension is mandatory;
    missing any field raises so the call site cannot silently publish
    a partial envelope under ``extra='forbid'``.

    :param workspace: resolved workspace entity (must have
        :attr:`owner_agent_id` populated and :attr:`customer_id` stamped
        via :func:`enrich_workspace_identity`)
    :ptype workspace: Workspace
    :return: audit-identity value for forwarding to publish helpers
    :rtype: WorkspaceAuditIdentity
    :raises RuntimeError: when no :class:`ToolCallScope` is installed,
        or when ``workspace.customer_id`` is None (the scope did not
        enrich), or when ``scope.context`` lacks user_id / agent_id /
        customer_id
    """
    from threetears.agent.tools.call_scope import current_scope

    scope = current_scope()
    if scope is None:
        raise RuntimeError(
            "workspace_audit_identity called outside a ToolCallScope; "
            "every tool dispatch must run under enter_call_scope so the "
            "invoking user / calling agent / customer identities are "
            "available to the audit envelope."
        )
    ctx = scope.context
    if ctx.user_id is None:
        raise RuntimeError(
            "workspace_audit_identity: scope.context.user_id is None; "
            "the tool dispatch envelope omitted the invoking user."
        )
    if ctx.agent_id is None:
        raise RuntimeError(
            "workspace_audit_identity: scope.context.agent_id is None; "
            "the tool dispatch envelope omitted the calling agent."
        )
    if ctx.customer_id is None:
        raise RuntimeError(
            "workspace_audit_identity: scope.context.customer_id is "
            "None; the tool dispatch envelope omitted the owning "
            "customer (WS-ACL-08 threading required)."
        )
    ws_customer = workspace.customer_id
    if ws_customer is None:
        raise RuntimeError(
            "workspace_audit_identity: workspace.customer_id is None; "
            "call enrich_workspace_identity before publishing the "
            "audit envelope."
        )
    return WorkspaceAuditIdentity(
        actor_user_id=ctx.user_id,
        calling_agent_id=ctx.agent_id,
        owner_agent_id=workspace.owner_agent_id,
        customer_id=ws_customer,
        namespace_id=workspace.id,
    )

if TYPE_CHECKING:
    from threetears.agent.tools.context import ToolContextManager

    from threetears.agent.workspace.collections import (
        WorkspaceCollection,
        WorkspaceFileCollection,
        WorkspaceFileVersionCollection,
    )
    from threetears.agent.workspace.config import ValidatorEntry
    from threetears.agent.workspace.entities import Workspace

from threetears.agent.workspace import pin as pin_module
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
        super().__init__(f"sha256 mismatch: expected {expected!r}, current {current!r}")


_SELECT_NAMESPACE_CUSTOMER_SQL = (
    "SELECT customer_id FROM platform.namespaces WHERE id = $1"
)


async def authorize_workspace(
    workspace: Workspace,
    operation: Literal["read", "write"],
    *,
    db_pool: Any,
    acl_cache: AclCache,
) -> None:
    """convenience wrapper: enrich identity then authorize via shared cache.

    every workspace tool's ``execute`` calls this once per dispatch,
    immediately after :func:`_resolve_workspace`. both ``acl_cache``
    and an installed :class:`ToolCallScope` are REQUIRED -- there is
    no "skip authorization" path (WS-ACL-05 is a hard requirement).
    tests must inject a real :class:`AclCache` wired with loader
    stubs and wrap the dispatch in :func:`enter_call_scope`;
    production wiring always supplies both. ``db_pool`` may be None
    to skip the identity-enrichment fetch -- the cache then sees a
    ``customer_id`` of ``None`` and rejects the call as unroutable,
    which is the desired behavior for tools called from harnesses
    that pre-stamped the entity.

    :param workspace: resolved workspace entity
    :ptype workspace: Workspace
    :param operation: ``"read"`` or ``"write"``; passed verbatim to
        :func:`authorize_workspace_access`
    :ptype operation: Literal["read", "write"]
    :param db_pool: asyncpg-like pool for the platform.namespaces
        lookup; may be ``None`` to skip enrichment
    :ptype db_pool: Any
    :param acl_cache: shared :class:`AclCache` wired with membership
        + grant loaders; REQUIRED
    :ptype acl_cache: AclCache
    :return: None
    :rtype: None
    :raises RuntimeError: when no :class:`ToolCallScope` is installed
        (the tool was dispatched outside a call-scope context)
    :raises WorkspaceAccessDenied: on any denial path
    """
    # local imports keep this module cheap when tools only need
    # _resolve_workspace or _write_file_atomic.
    from threetears.agent.tools.call_scope import current_scope

    from threetears.agent.workspace.authorize import authorize_workspace_access

    scope = current_scope()
    if scope is None:
        raise RuntimeError(
            "authorize_workspace called outside a ToolCallScope; every tool "
            "dispatch must enter_call_scope before executing. tests should "
            "wrap the tool invocation in enter_call_scope(ToolCallScope(...))."
        )
    if db_pool is not None:
        await enrich_workspace_identity(workspace, db_pool)
    await authorize_workspace_access(
        scope,
        workspace,
        operation,
        acl_cache=acl_cache,
    )
    return None


async def authorize_workspace_file(
    workspace: Workspace,
    relative_path: str,
    direction: Literal["read", "write"],
    *,
    db_pool: Any,
    acl_cache: AclCache,
) -> None:
    """per-file rbac gate replacing the retired ``sandbox.enforce`` call.

    namespace-task-01 phase 7 wrapper: the workspace file-access
    enforcement path is now (1) :meth:`WorkspaceSandbox.validate_syntax`
    for syntactic sanity, (2) this helper for the path-glob rbac
    decision. see
    :func:`threetears.agent.workspace.authorize.authorize_workspace_file_access`
    for the underlying evaluator wiring; this helper performs the
    identity-enrichment round-trip and installs the call-scope guard
    so each tool's execute path reads as a one-liner.

    tests must wrap the dispatch in
    :func:`threetears.agent.tools.call_scope.enter_call_scope`;
    production wiring always supplies both ``acl_cache`` and the
    call-scope. ``db_pool`` may be ``None`` when the caller has
    pre-stamped ``workspace.customer_id`` (harness tests).

    :param workspace: resolved workspace entity
    :ptype workspace: Workspace
    :param relative_path: workspace-relative path being authorized
    :ptype relative_path: str
    :param direction: ``"read"`` for a read call, ``"write"`` for a
        mutation (must map to the corresponding
        ``read_file_matching:`` / ``write_file_matching:`` action
        prefix in the evaluator)
    :ptype direction: Literal["read", "write"]
    :param db_pool: asyncpg-like pool for the platform.namespaces
        lookup; may be ``None`` to skip enrichment
    :ptype db_pool: Any
    :param acl_cache: shared :class:`AclCache` wired with membership +
        grant loaders; REQUIRED
    :ptype acl_cache: AclCache
    :return: None
    :rtype: None
    :raises RuntimeError: when no :class:`ToolCallScope` is installed
    :raises WorkspaceAccessDenied: on any denial path (missing
        customer, cross-customer, no matching glob)
    """
    from threetears.agent.tools.call_scope import current_scope

    from threetears.agent.workspace.authorize import (
        authorize_workspace_file_access,
    )

    scope = current_scope()
    if scope is None:
        raise RuntimeError(
            "authorize_workspace_file called outside a ToolCallScope; every "
            "tool dispatch must enter_call_scope before executing. tests "
            "should wrap the tool invocation in "
            "enter_call_scope(ToolCallScope(...)).",
        )
    if db_pool is not None:
        await enrich_workspace_identity(workspace, db_pool)
    await authorize_workspace_file_access(
        scope,
        workspace,
        relative_path,
        direction,
        acl_cache=acl_cache,
    )
    return None


async def enrich_workspace_identity(
    workspace: Workspace,
    db_pool: Any,
) -> Workspace:
    """stamp ``workspace.customer_id`` from platform.namespaces in-place.

    workspace-task-19 (WS-ACL-03) keeps the customer dimension on
    :class:`platform.namespaces` rather than duplicating it onto every
    agent-schema ``workspaces`` row. resolving the customer requires a
    single platform-level fetch; this helper performs that lookup and
    stamps the result onto the in-memory entity via the
    :attr:`Workspace.customer_id` setter so the authorize helper can
    read it back on the next statement.

    the query uses ``namespace=`` so it lands on the platform pool
    regardless of the caller's default agent schema: the L3 proxy
    recognizes ``namespace="platform"`` (or any platform-typed
    namespace) and binds ``search_path`` accordingly. pools that lack
    ``namespace=`` support (tests, direct asyncpg) fall back to the
    fully-qualified ``platform.namespaces`` table reference.

    :param workspace: workspace entity to enrich
    :ptype workspace: Workspace
    :param db_pool: asyncpg pool (or pool-like) that can reach the
        platform schema; the v014 migration places namespaces on
        every agent's reachable path
    :ptype db_pool: Any
    :return: the same ``workspace`` instance (returned for chaining)
    :rtype: Workspace
    """
    row = await db_pool.fetchrow(_SELECT_NAMESPACE_CUSTOMER_SQL, workspace.id)
    if row is not None:
        raw = row["customer_id"] if isinstance(row, dict) else row["customer_id"]
        workspace.customer_id = UUID(str(raw)) if raw is not None else None
    return workspace


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
        result = await workspace_collection.find_by_agent_and_name(agent_id, workspace_arg)
        if result is None:
            raise WorkspaceNotFound(f"workspace {workspace_arg!r} not found")
    else:
        snapshot = await pin_module.get_pin(context)
        if snapshot is None:
            raise NoWorkspacePinned("no workspace pinned; call workspace.use(name) first")
        result = await workspace_collection.find_by_id_and_agent(snapshot.workspace_id, agent_id)
        if result is None:
            raise WorkspaceNotFound(f"pinned workspace {snapshot.workspace_name!r} not found")
    if result.date_deleted is not None:
        raise WorkspaceNotFound(f"workspace {result.name!r} is deleted")
    return result


_SELECT_HEAD_SQL = "SELECT content, sha256, version FROM workspace_files WHERE workspace_id = $1 AND relative_path = $2"

_SELECT_JOURNAL_MAX_VERSION_SQL = (
    "SELECT COALESCE(MAX(version), 0) AS max_version "
    "FROM workspace_file_versions "
    "WHERE workspace_id = $1 AND relative_path = $2"
)


async def _next_journal_version(
    conn: Any,
    workspace_id: UUID,
    relative_path: str,
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
    row = await conn.fetchrow(_SELECT_JOURNAL_MAX_VERSION_SQL, workspace_id, relative_path)
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
WHERE id = $3 AND agent_id = $4
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

    # WS-ACL-06: bind the tx session to the workspace's namespace so
    # the broker resolves ``SET search_path`` to the OWNER agent's
    # schema on every statement below. owner and grantee paths run
    # the same SQL against the same physical storage; only the ACL
    # check at authorize time differentiates them. pools lacking the
    # ``namespace=`` kwarg (test fakes, direct asyncpg) ignore the
    # keyword because they bind search_path elsewhere; the production
    # NatsProxyL3Backend honors it.
    async with db_pool.acquire() as conn:
        async with conn.transaction(namespace=workspace.namespace_name):
            head = await conn.fetchrow(_SELECT_HEAD_SQL, workspace.id, relative_path)
            current_sha: str | None = None if head is None else head["sha256"]
            if expected_sha256 is not None and current_sha != expected_sha256:
                raise Sha256Mismatch(expected=expected_sha256, current=current_sha)
            if validators:
                dispatch_validators(validators, relative_path, content)
            # derive next version from the journal, not the head row:
            # bind's capture-back may DELETE the head row on "delete"
            # action while leaving the journal intact. a later re-create
            # at the same path would otherwise collide on the
            # (workspace_id, relative_path, version) unique constraint.
            new_version = await _next_journal_version(conn, workspace.id, relative_path)
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
                workspace.agent_id,
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
    *,
    namespace_name: str | None = None,
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

    WS-ACL-06: ``namespace_name`` routes the outside-tx lookups to the
    owner agent's schema. when the connection is already pinned to a
    tx (``conn.tx_id`` set), per-statement ``namespace=`` is rejected
    by the proxy; the helper detects that case and omits the kwarg so
    the tx session's already-bound namespace applies. outside a tx the
    kwarg rides on every fetchrow so grantee rollback / diff queries
    land in the owner's schema.

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
    :param namespace_name: canonical workspace namespace
        (``workspace.<uuid>``) passed as ``namespace=`` to the
        outside-tx fetchrow calls so grantee reads route correctly
    :ptype namespace_name: str | None
    :return: journal row as dict, or None when no row matches
    :rtype: dict[str, Any] | None
    """
    # inside-tx callers must not pass namespace= on per-statement calls
    # (the proxy rejects it). detect the pinned state by reading
    # ``tx_id``; connections without that attribute (raw asyncpg) are
    # always outside-tx so the kwarg flows through.
    inside_tx = getattr(conn, "tx_id", None) is not None
    kwargs: dict[str, Any] = (
        {} if inside_tx or namespace_name is None
        else {"namespace": namespace_name}
    )
    result: dict[str, Any] | None
    row: Any
    if isinstance(ref, int):
        row = await conn.fetchrow(
            _SELECT_VERSION_BY_NUMBER_SQL,
            workspace_id,
            relative_path,
            ref,
            **kwargs,
        )
    elif ref == "head":
        row = await conn.fetchrow(
            _SELECT_LATEST_VERSION_ROW_SQL,
            workspace_id,
            relative_path,
            **kwargs,
        )
    elif ref.isdigit():
        row = await conn.fetchrow(
            _SELECT_VERSION_BY_NUMBER_SQL,
            workspace_id,
            relative_path,
            int(ref),
            **kwargs,
        )
    else:
        row = await conn.fetchrow(
            _SELECT_CHECKPOINT_ROW_SQL,
            workspace_id,
            relative_path,
            ref,
            **kwargs,
        )
    if row is None:
        result = None
    else:
        result = dict(row)
    return result
