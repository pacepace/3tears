"""identity + authorization primitives for the MCP framework.

three concerns separated cleanly so v2 (HTTP transport + per-call
bearer-token identity) plugs in without disturbing v1 wiring:

- :class:`Identity` -- who is the caller? in v1 stdio this is fixed
  for the server's lifetime (env-var creds). in v2 HTTP it varies
  per request.
- :class:`IdentityProvider` Protocol -- where does the identity come
  from? :class:`EnvVarIdentityProvider` is the v1 impl; a future
  :class:`BearerTokenIdentityProvider` slots in for v2 with the same
  interface.
- :class:`Authorizer` Protocol -- given an identity and a permission
  string, allow or deny. :class:`LocalGrantAuthorizer` is the
  framework default backed by :class:`McpToolGrantCollection` (kept
  current by following the table's write generation). per-product impls can plug in a different
  evaluator (e.g. one backed by ``threetears.agent.acl.
  evaluate_decision``) without changing per-tool registrations.
  :class:`LocalGrantAuthorizer` needs the ``grants`` extra; everything
  else here imports without it.

default-deny: no admit-by-default codepath. the framework refuses to
dispatch a tool unless the configured :class:`Authorizer` returns
True for ``(identity, tool.required_permission)``.

admin auto-grant: the configured admin :class:`Identity` is granted
every tool **in memory only** at server startup. nothing writes to
``mcp_tool_grants`` for the admin grant -- the table stays truthful
to operator-added grants. the auto-grant is logged at INFO so it's
auditable.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from threetears.observe import PeriodicTask, cancel_and_wait, contain, get_logger, spawn_background

# LocalGrantAuthorizer's collaborators come from the ``grants`` extra. Nothing else in this module
# needs them, and a server with its own Authorizer must import this module without them, so their
# absence is recorded here and LocalGrantAuthorizer refuses to build, naming the extra.
_GRANTS_IMPORT_ERROR: ImportError | None = None
try:
    from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
    from threetears.epoch import EpochGenerationReader, TableFollower, follow_tables
except ImportError as _exc:  # NOSILENT: the grants extra is optional; LocalGrantAuthorizer names it
    _GRANTS_IMPORT_ERROR = _exc

__all__ = [
    "Authorizer",
    "BearerTokenIdentityProvider",
    "BearerTokenResolver",
    "EnvVarIdentityProvider",
    "GrantCacheUnavailableError",
    "Identity",
    "IdentityProvider",
    "LocalGrantAuthorizer",
    "PrincipalType",
    "TokenSource",
]

log = get_logger(__name__)


PrincipalType = Literal["user", "group", "role"]
"""shape of a principal in :class:`McpToolGrantCollection`.

users are the dominant case in v1; groups + roles unlock the
"grant the role X this permission" pattern that the hub's RBAC
already uses for HTTP endpoints.
"""


@dataclass(frozen=True, slots=True)
class Identity:
    """the resolved caller identity for one MCP server (v1) or one
    MCP request (v2).

    :ivar principal_type: which principal table the identity belongs
        to -- v1 always uses ``"user"`` since stdio servers run as
        a single env-var-derived user
    :ivar principal_id: UUID of the user/group/role
    :ivar groups: pre-resolved group memberships of this principal;
        used by :class:`LocalGrantAuthorizer` when checking
        group-scoped grants. empty for v1's env-var identity unless
        the provider does the resolve at startup
    :ivar roles: pre-resolved role assignments of this principal;
        used by :class:`LocalGrantAuthorizer` when checking
        role-scoped grants
    :ivar is_admin: convenience flag set by the
        :class:`IdentityProvider` when the identity has admin role.
        used by the auto-grant short-circuit; never persisted
    """

    principal_type: PrincipalType
    principal_id: UUID
    groups: frozenset[UUID] = frozenset()
    roles: frozenset[UUID] = frozenset()
    is_admin: bool = False


@runtime_checkable
class IdentityProvider(Protocol):
    """resolve the calling identity for a tool dispatch.

    v1 stdio: returns one fixed identity per server (env-var creds).
    v2 HTTP: returns the bearer-token-resolved identity per request.

    :class:`McpServer` calls :meth:`identify` once per dispatch; v1
    impls cache; v2 impls re-resolve.
    """

    async def identify(self) -> Identity:
        """resolve the calling identity.

        :return: caller identity
        :rtype: Identity
        :raises RuntimeError: when no identity can be resolved
            (e.g. env var missing in v1, bearer token absent in v2)
        """
        ...


class EnvVarIdentityProvider:
    """v1 stdio :class:`IdentityProvider` -- one identity per server lifetime.

    reads the admin user UUID from env vars (``MCP_ADMIN_USER_ID``
    by default; configurable). future v2 :class:`BearerTokenIdentityProvider`
    will resolve per request from the MCP-client-supplied bearer
    token using the same :class:`Identity` shape.

    the admin flag is default-deny: ``is_admin`` defaults to False so a
    wiring that forgets the flag yields an ordinary (non-admin) identity
    subject to the same grant checks as any other principal. admin
    authority is a privilege that must be granted explicitly -- a
    consumer wiring the admin-equivalent env-var creds passes
    ``is_admin=True`` at construction (the two stdio launchers do). the
    previous True-by-default admitted every tool for any caller whose
    wirer omitted the flag, a total RBAC bypass.

    :param principal_id: user UUID; if ``None`` reads from
        ``user_id_env_var``
    :ptype principal_id: UUID | None
    :param user_id_env_var: env var name that holds the admin user
        UUID when ``principal_id`` is unset
    :ptype user_id_env_var: str
    :param is_admin: whether this identity has admin role; default-deny
        False, admin granted only when passed explicitly
    :ptype is_admin: bool
    """

    def __init__(
        self,
        *,
        principal_id: UUID | None = None,
        user_id_env_var: str = "MCP_ADMIN_USER_ID",
        is_admin: bool = False,
    ) -> None:
        """capture identity source; resolve at :meth:`identify` call.

        :param principal_id: explicit UUID; bypasses env var when set
        :ptype principal_id: UUID | None
        :param user_id_env_var: env var consulted when ``principal_id``
            is ``None``
        :ptype user_id_env_var: str
        :param is_admin: admin flag; default-deny False, granted only
            when the wirer passes True explicitly
        :ptype is_admin: bool
        :return: nothing
        :rtype: None
        """
        self._principal_id = principal_id
        self._env_var = user_id_env_var
        self._is_admin = is_admin

    async def identify(self) -> Identity:
        """resolve the configured identity.

        :return: caller identity
        :rtype: Identity
        :raises RuntimeError: when neither ``principal_id`` was set
            at construction nor the configured env var is populated
            with a valid UUID
        """
        if self._principal_id is not None:
            principal_id = self._principal_id
        else:
            raw = os.environ.get(self._env_var)
            if not raw:
                raise RuntimeError(
                    f"identity provider requires {self._env_var} or an explicit principal_id at construction",
                )
            try:
                principal_id = UUID(raw)
            except ValueError as exc:
                raise RuntimeError(
                    f"{self._env_var} must be a valid UUID; got {raw!r}",
                ) from exc
        return Identity(
            principal_type="user",
            principal_id=principal_id,
            is_admin=self._is_admin,
        )


BearerTokenResolver = Callable[[str], Awaitable["Identity"]]
"""signature for the bearer-token -> :class:`Identity` resolver.

injected into :class:`BearerTokenIdentityProvider` so the 3tears mcp
package never imports hub auth code: the hub wires its own
``APIKeyAuthStrategy`` / ``JWTAuthStrategy``-backed resolver. the
resolver validates the token and returns the resolved caller
:class:`Identity`; it raises when the token cannot be resolved.
"""


TokenSource = Callable[[], "str | None"]
"""signature for the per-request bearer-token source.

returns the current request's bearer token (or ``None`` when absent).
in production the transport layer populates a request-scoped
contextvar and passes its reader here (see
:func:`threetears.mcp.http_server.current_bearer_token`); tests pass a
simple mutable holder. keeping this injectable is what lets
:meth:`BearerTokenIdentityProvider.identify` honour the fixed
zero-argument Protocol signature while still re-resolving per request.
"""


class BearerTokenIdentityProvider:
    """v2 HTTP :class:`IdentityProvider` -- one identity per request.

    unlike :class:`EnvVarIdentityProvider` (one fixed identity per
    server lifetime), this provider re-resolves the caller on every
    :meth:`identify` call from the request-scoped bearer token. the
    token is obtained from the injected ``token_source`` (a
    request-scoped contextvar reader in production) and mapped to an
    :class:`Identity` by the injected ``resolver``. neither the token
    nor the resolved identity is cached across requests.

    the 3tears mcp package deliberately does not decode JWTs or read a
    user table itself: both the token source and the resolver are
    injected so the hub can supply its own credential-validation
    backend without this package importing hub code.

    :param resolver: async token -> :class:`Identity` resolver
    :ptype resolver: BearerTokenResolver
    :param token_source: callable returning the current request's
        bearer token (or ``None`` when absent)
    :ptype token_source: TokenSource
    """

    def __init__(
        self,
        *,
        resolver: BearerTokenResolver,
        token_source: TokenSource,
    ) -> None:
        """capture the resolver + token source; resolve at :meth:`identify`.

        :param resolver: async token -> :class:`Identity` resolver
        :ptype resolver: BearerTokenResolver
        :param token_source: current-request bearer-token reader
        :ptype token_source: TokenSource
        :return: nothing
        :rtype: None
        """
        self._resolver = resolver
        self._token_source = token_source

    async def identify(self) -> Identity:
        """resolve the calling identity from the current request's bearer token.

        re-resolves per call (v2 semantics): reads the token from the
        injected source, then maps it via the injected resolver. both
        are injected, so neither has an error contract of its own: an
        exception of any type from either is surfaced as
        ``RuntimeError`` to honour the :class:`IdentityProvider`
        Protocol contract, which is the one type
        :meth:`McpServer._dispatch` maps to the ``IDENTITY_UNAVAILABLE``
        error result.

        :return: resolved caller identity
        :rtype: Identity
        :raises RuntimeError: when the token source fails, the bearer
            token is absent, or the resolver cannot resolve it to an
            identity
        """
        try:
            token = self._token_source()
            if not token:
                raise RuntimeError(
                    "bearer token absent from request context; caller identity cannot be resolved",
                )
            identity = await self._resolver(token)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"bearer token could not be read or resolved to an identity: {type(exc).__name__}: {exc}",
            ) from exc
        return identity


@runtime_checkable
class Authorizer(Protocol):
    """permission evaluator -- given identity and permission, allow or deny.

    :class:`LocalGrantAuthorizer` is the framework default backed by
    :class:`~threetears.mcp.rbac.McpToolGrantCollection`. per-product
    impls plug in for different evaluators (e.g. backed by the
    existing ``threetears.agent.acl.evaluate_decision``); the
    interface is unchanged.
    """

    async def allows(self, identity: Identity, permission: str) -> bool:
        """return True iff ``identity`` is granted ``permission``.

        :param identity: caller identity
        :ptype identity: Identity
        :param permission: permission string the tool requires
        :ptype permission: str
        :return: True when allowed; False when denied
        :rtype: bool
        """
        ...

    async def start(self) -> None:
        """initialize background state (cache prime, listener subscribe).

        called once by :meth:`McpServer.serve` before accepting any
        dispatch. authorizers without background state implement as
        a no-op.

        :return: nothing
        :rtype: None
        """
        ...

    async def stop(self) -> None:
        """tear down background state (cancel periodic tasks, etc.).

        called by :meth:`McpServer.stop` (or the server's lifespan
        teardown). authorizers without background state implement
        as a no-op.

        :return: nothing
        :rtype: None
        """
        ...


# ---------------------------------------------------------------------
# LocalGrantAuthorizer -- framework default impl
# ---------------------------------------------------------------------


GrantLoader = Callable[[], Awaitable[list[dict[str, Any]]]]
"""signature for the function that reads every grant from L3: the cold-start fill.

returns a list of dict rows shaped like ``mcp_tool_grants`` table columns (``grant_id``,
``principal_id``, ``permission``; others are ignored). :class:`LocalGrantAuthorizer` calls it only
to fill its empty cache at :meth:`~LocalGrantAuthorizer.start`, and when the table is dropped here
because a write changed the whole of it (a truncate, a load, an announced migration: the operator's
reset). a changed grant never reaches it.

abstracted as a function (not a direct collection reference) so tests can substitute a fake
without spinning up Postgres + the full :class:`McpToolGrantCollection` plumbing.
"""


GrantReader = Callable[[UUID], Awaitable[dict[str, Any] | None]]
"""signature for the function that reads ONE grant from L3 by its key.

returns the grant's row (shaped as a :data:`GrantLoader` row), or ``None`` when no grant has that
key -- it was revoked. :class:`LocalGrantAuthorizer` calls it for each grant the table's row
messages and generation walk name (:meth:`McpToolGrantCollection.read_grant`).
"""


#: the table the grant cache is derived from
_GRANTS_TABLE = "mcp_tool_grants"


def _require_grants_extra() -> None:
    """refuse to build the grant store's authorizer where its extra is not installed.

    :return: nothing
    :rtype: None
    :raises ImportError: when ``threetears.core`` or ``threetears.epoch`` could not be imported
    """
    if _GRANTS_IMPORT_ERROR is not None:
        raise ImportError(
            "LocalGrantAuthorizer is the MCP grant store, which needs the grants extra: install 3tears-mcp[grants]",
        ) from _GRANTS_IMPORT_ERROR


def _held_by(row: dict[str, Any]) -> tuple[UUID, tuple[UUID, str]]:
    """one grant row as the cache holds it: its key, and the ``(principal_id, permission)`` it gives.

    :param row: a ``mcp_tool_grants`` row
    :ptype row: dict[str, Any]
    :return: ``(grant_id, (principal_id, permission))``
    :rtype: tuple[UUID, tuple[UUID, str]]
    :raises KeyError: when the row lacks one of the three columns
    :raises ValueError: when a key is not a uuid
    """
    grant_id = row["grant_id"]
    principal_id = row["principal_id"]
    if not isinstance(grant_id, UUID):
        grant_id = UUID(str(grant_id))
    if not isinstance(principal_id, UUID):
        principal_id = UUID(str(principal_id))
    return grant_id, (principal_id, str(row["permission"]))


class GrantCacheUnavailableError(RuntimeError):
    """the grant cache is not known to match ``mcp_tool_grants``, so no grant in it is trusted.

    raised by :meth:`LocalGrantAuthorizer.allows` for a non-admin identity while a read the cache
    needs failed and is owed -- the fill, or a changed grant's keyed read -- so the cache may hold a
    grant revoked since. answering ``False`` instead would read as "you lack this permission", which
    is not what happened; answering from the cache would keep a revoked grant allowed. the owed read
    is retried every ``retry_interval_seconds``; the MCP server turns this into an ``AUTHZ_ERROR``
    result and logs it.
    """


class LocalGrantAuthorizer:
    """framework default :class:`Authorizer` backed by ``mcp_tool_grants``.

    holds an in-memory grant cache: each grant by its key, and how many grants give each
    ``(principal_id, permission)``. :meth:`start` fills it once from the supplied
    :class:`GrantLoader` -- the cold start.

    **cross-pod coherence is the table's write generation.** ``mcp_tool_grants`` is a switched-on
    table (:class:`McpToolGrantCollection`), so every committed grant write advances its generation and
    records the keys it changed. the authorizer is a cache derived from the table
    (``CollectionRegistry.register_derived_cache``) and follows it (``threetears.epoch.follow_tables``):
    each row message -- a peer's broadcast, this process's own write, or an epoch the generation walk
    finds this process missed (a lost broadcast, one out of order, a write that bumped nothing) --
    names a grant, and the authorizer reads that one grant by key (:class:`GrantReader`) and inserts,
    updates or removes it. nothing but the cold start, and a write that changed the whole table
    (the operator's reset: a truncate, a load), reads every grant.

    a process the follow has fallen behind on (its change record lost with the broker) keeps serving
    what it holds, as every follower does, said by the registry at ERROR; a restart fills it again.

    admin auto-grant: when an :class:`Identity` arrives with ``is_admin=True``, every permission
    check returns True without consulting the cache. the auto-grant is logged at INFO at
    :meth:`start`-time so it's auditable; nothing is written to ``mcp_tool_grants``.

    :param grant_loader: reads every grant row from L3; called only for the cold start and a write
        that changed the whole table
    :ptype grant_loader: GrantLoader
    :param grant_reader: reads one grant row from L3 by key; required with a registry
    :ptype grant_reader: GrantReader | None
    :param registry: the registry whose invalidation listener hears the table's rows and which
        follows its generation. Optional: when ``None``, the cache is filled once at :meth:`start` and
        never changed -- correct for single-process modes (stdio MCP servers, tests) where there is no
        other writer.
    :ptype registry: CollectionRegistry | None
    :param generation_reader: reads the table's generation for the mark start() takes, and watches it
        for the follow; given with ``registry``
    :ptype generation_reader: EpochGenerationReader | None
    :param admin_principal_ids: optional set of principal UUIDs to log explicitly at start-time as
        auto-granted. logging the specific principal_id (not just "admins get everything") keeps the
        audit trail concrete for the v1 single-identity stdio mode
    :ptype admin_principal_ids: set[UUID] | None
    :param retry_interval_seconds: how often an owed read (a failed fill or keyed read) is retried
    :ptype retry_interval_seconds: float
    """

    def __init__(
        self,
        *,
        grant_loader: GrantLoader,
        grant_reader: GrantReader | None = None,
        registry: CollectionRegistry | None = None,
        generation_reader: EpochGenerationReader | None = None,
        admin_principal_ids: set[UUID] | None = None,
        retry_interval_seconds: float = 5.0,
    ) -> None:
        """capture deps; no I/O until :meth:`start`.

        ``registry`` and ``generation_reader`` are jointly optional -- single-process modes (stdio
        MCP servers, tests) pass neither; multi-pod modes pass both, and a ``grant_reader`` to apply
        each changed grant with.

        :param grant_loader: L3 read of every grant, for the cold start and a whole-table change
        :ptype grant_loader: GrantLoader
        :param grant_reader: L3 read of one grant by key; required with a registry
        :ptype grant_reader: GrantReader | None
        :param registry: the registry that follows the table; optional
        :ptype registry: CollectionRegistry | None
        :param generation_reader: the follow's epoch-bucket reader; optional, with ``registry``
        :ptype generation_reader: EpochGenerationReader | None
        :param admin_principal_ids: principals to log as auto-granted
        :ptype admin_principal_ids: set[UUID] | None
        :param retry_interval_seconds: how often an owed read is retried (only with a registry)
        :ptype retry_interval_seconds: float
        :return: nothing
        :rtype: None
        :raises ValueError: when exactly one of registry / generation_reader is provided (must be both
            or neither), when a registry comes without a grant reader, or when an interval is not
            positive
        :raises ImportError: when the ``grants`` extra is not installed
        """
        _require_grants_extra()
        if (registry is None) != (generation_reader is None):
            raise ValueError(
                "registry and generation_reader must be provided together; passing exactly one is a usage error",
            )
        if registry is not None and grant_reader is None:
            raise ValueError(
                "a changed grant is applied by reading it by key: pass grant_reader "
                "(McpToolGrantCollection.read_grant) with the registry",
            )
        if registry is not None and not (0 < retry_interval_seconds < float("inf")):
            # refused HERE, not when the loop is built: by then start() has already registered and
            # filled, and a failure there would leave a registration behind with no loop.
            raise ValueError(f"retry_interval_seconds must be positive, got {retry_interval_seconds}")
        self._grant_loader = grant_loader
        self._grant_reader = grant_reader
        self._registry = registry
        self._generation_reader = generation_reader
        self._admin_principal_ids = admin_principal_ids or set()
        self._retry_interval_seconds = retry_interval_seconds
        # each grant by its key -> the (principal_id, permission) it gives, and how many grants
        # give each pair: two grants for two tools can give one permission, and revoking one of
        # them must not take the permission away. presence is the grant; absence is default-deny.
        self._grants: dict[UUID, tuple[UUID, str]] = {}
        self._held: dict[tuple[UUID, str], int] = {}
        # reads (a fill's, a key's) overlap, so each is numbered when it STARTS. a read lands only
        # when nothing that started later has already landed for what it read, and clears what is
        # owed only when it started after the failure that owed it: otherwise a slow read that
        # began before a revocation could finish last and put the revoked grant back.
        self._reads_started = 0
        # the newest fill landed, and the newest whose failure is still owed; while a fill is
        # owed every non-admin check is refused.
        self._filled_generation = 0
        self._fill_failed_generation = 0
        self._fill_owed = False
        # per grant: the newest keyed read landed, and a failed one still owed
        self._key_landed: dict[UUID, int] = {}
        self._keys_owed: dict[UUID, int] = {}
        # the keyed reads (and a whole-table change's fill) the registry's synchronous callbacks
        # started, held so stop() can end them
        self._reads: set[asyncio.Task[None]] = set()
        self._unregister: Callable[[], None] | None = None
        self._follower: TableFollower | None = None
        self._started = False
        self._retry_task: PeriodicTask | None = None

    async def start(self) -> None:
        """register as a cache derived from the table, take its mark, fill the cache once, spawn the retry.

        idempotent: subsequent calls log a warning and short-circuit. the table's generation is read
        and taken as this registry's mark BEFORE the fill reads its rows, so every write the fill may
        not hold is an epoch after the mark: the follow walks those and reads each grant by key. the
        follow itself needs the registry's invalidation listener running; when it already runs it
        starts here, otherwise the host calls :meth:`follow` once it does.

        :return: nothing
        :rtype: None
        :raises RuntimeError: in single-process mode, when the fill fails: nothing would retry it
        """
        if self._started:
            log.warning("LocalGrantAuthorizer.start called twice; ignoring")
            return
        following = self._registry is not None
        if self._registry is not None:
            assert self._generation_reader is not None  # constructed together
            self._unregister = self._registry.register_derived_cache(
                _GRANTS_TABLE, on_row=self._on_grant_row, on_table_dropped=self._on_table_dropped
            )
            # the generation as it stands, read before the fill: everything at or below it is in the
            # rows the fill reads, and the follow walks every epoch after it. ``None`` (no change log
            # yet) is a mark too: every epoch of the first incarnation minted after it is walked
            marked = await self._generation_reader.read(_GRANTS_TABLE)
            self._registry.follow_generation(_GRANTS_TABLE)
            self._registry.generation_marks.adopt_generation(_GRANTS_TABLE, marked)
        try:
            await self._fill()
        except Exception as exc:
            # the grant loader is the caller's and has no error contract of its own; whatever it
            # raised, the fill is already marked owed (in _fill) and every non-admin check refused.
            # single-process mode has no retry to land it, so this authorizer must not start.
            if not following:
                raise RuntimeError("MCP grant cache could not be loaded at start, and nothing would retry it") from exc
            contain(
                exc,
                log=log,
                what="MCP grant cache could not be filled at start; refusing non-admin checks until the retry lands",
                level="error",
            )
        log_extras: dict[str, Any] = {
            "grant_count": len(self._grants),
            "mode": "follows mcp_tool_grants" if following else "single-process",
        }
        if self._admin_principal_ids:
            log_extras["admin_principal_ids"] = sorted(str(pid) for pid in self._admin_principal_ids)
            log.info(
                "MCP authorizer started; admin principals auto-granted in memory (not persisted to mcp_tool_grants)",
                extra={"extra_data": log_extras},
            )
        else:
            log.info(
                "MCP authorizer started; no explicit admin principals registered "
                "(any Identity arriving with is_admin=True still short-circuits)",
                extra={"extra_data": log_extras},
            )
        if following:
            # sleeps first; a pass that errors is logged and retried on the next interval.
            self._retry_task = PeriodicTask(
                self._retry_owed,
                interval=self._retry_interval_seconds,
                name="mcp-grant-retry-loop",
                logger=log,
                failure_message="MCP grant read retry failed; will retry on next interval",
            )
            self._retry_task.start()
            if self._registry is not None and self._registry.invalidation_listener_running:
                self.follow()
        self._started = True

    def follow(self) -> None:
        """start following ``mcp_tool_grants``: each epoch after the mark start() took is walked and read by key.

        call once the registry's invalidation listener runs (:meth:`start` calls it itself when it already
        does). idempotent. until it runs, a grant written by another process is not seen here.

        :return: nothing
        :rtype: None
        :raises RuntimeError: without a registry, or when its invalidation listener is not running
        """
        if self._registry is None or self._generation_reader is None:
            raise RuntimeError("a single-process LocalGrantAuthorizer has no table to follow")
        if self._follower is not None:
            return
        self._follower = follow_tables(self._registry, self._generation_reader, (_GRANTS_TABLE,))
        log.info("MCP grant cache follows mcp_tool_grants")

    async def stop(self) -> None:
        """stop following, unregister, cancel the retry and every read in flight.

        idempotent: subsequent calls are no-ops. ``_started`` is cleared FIRST, on every path, so a
        stopped authorizer can be started again.

        :return: nothing
        :rtype: None
        """
        self._started = False
        unregister, self._unregister = self._unregister, None
        if unregister is not None:
            # first, so nothing the registry hears from here starts another read
            unregister()
        follower, self._follower = self._follower, None
        if follower is not None:
            await follower.stop()
        task, self._retry_task = self._retry_task, None
        if task is not None:
            await task.stop()
        reads, self._reads = self._reads, set()
        for read in reads:
            await cancel_and_wait(read)

    def _on_grant_row(self, message: CacheInvalidationMessage) -> None:
        """registry callback: one grant changed -- a broadcast, this process's write, or a walked epoch.

        synchronous, as the registry calls it, so the keyed read runs as its own task. until it lands
        the cache answers from what it held, as every follower serves what it holds while it catches up.

        :param message: the row message; ``ids[0]`` is the grant's key
        :ptype message: CacheInvalidationMessage
        :return: nothing
        :rtype: None
        :raises ValueError: when the message's key is not a uuid
        """
        grant_id = UUID(message.ids[0])
        self._spawn(self._apply_grant(grant_id), f"mcp-grant-read-{grant_id}")

    def _on_table_dropped(self) -> None:
        """registry callback: a write changed the whole table (a truncate, a load) -- fill again.

        the operator's reset, and with the cold start the only time every grant is read.

        :return: nothing
        :rtype: None
        """
        log.warning(
            "mcp_tool_grants changed as a whole table; filling the grant cache again",
            extra={"extra_data": {"grant_count": len(self._grants)}},
        )
        self._spawn(self._fill(), "mcp-grant-fill")

    def _spawn(self, read: Coroutine[Any, Any, None], name: str) -> None:
        """run one read as a task this authorizer holds until it ends; its failure is logged.

        :param read: the read
        :ptype read: Coroutine[Any, Any, None]
        :param name: the task's name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        task = spawn_background(read, name=name, logger=log)
        self._reads.add(task)
        task.add_done_callback(self._reads.discard)

    async def _retry_owed(self) -> None:
        """one retry pass: the fill when it is owed, then every grant whose keyed read is owed, by key.

        a retry that fails again raises out of the pass, so ``PeriodicTask`` logs each one.

        :return: nothing
        :rtype: None
        """
        if self._fill_owed:
            await self._fill()
        for grant_id in tuple(self._keys_owed):
            await self._apply_grant(grant_id)

    async def allows(self, identity: Identity, permission: str) -> bool:
        """return True iff ``identity`` holds ``permission``.

        admin short-circuit fires before the cache lookup. otherwise consults the in-memory cache
        of ``mcp_tool_grants``. group/role-scoped grants are matched against the pre-resolved
        ``groups`` / ``roles`` on the :class:`Identity` (the :class:`IdentityProvider` is
        responsible for resolving them).

        :param identity: caller identity
        :ptype identity: Identity
        :param permission: permission string
        :ptype permission: str
        :return: True when the grant exists; False otherwise
        :rtype: bool
        :raises GrantCacheUnavailableError: a read the cache needs failed and is owed (admins are
            unaffected)
        """
        if identity.is_admin:
            return True
        why = self._why_not_current()
        if why is not None:
            raise GrantCacheUnavailableError(
                f"MCP grant cache is not current ({why}); refusing {permission!r} "
                "rather than answering from grants that may have been revoked",
            )
        if (identity.principal_id, permission) in self._held:
            return True
        for group_id in identity.groups:
            if (group_id, permission) in self._held:
                return True
        for role_id in identity.roles:
            if (role_id, permission) in self._held:
                return True
        return False

    def _why_not_current(self) -> str | None:
        """why the cache may hold a revoked grant, or ``None`` when every read it needs has landed.

        :return: the reason, or ``None``
        :rtype: str | None
        """
        if self._fill_owed:
            return "the last load of the grants failed"
        if self._keys_owed:
            return f"the read of {len(self._keys_owed)} changed grant(s) failed"
        return None

    def _put(self, grant_id: UUID, held: tuple[UUID, str] | None) -> None:
        """set one grant in the cache: ``held`` is what it gives, ``None`` that it is gone.

        :param grant_id: the grant
        :ptype grant_id: UUID
        :param held: ``(principal_id, permission)``, or ``None`` for a revoked grant
        :ptype held: tuple[UUID, str] | None
        :return: nothing
        :rtype: None
        """
        before = self._grants.pop(grant_id, None)
        if before is not None:
            remaining = self._held[before] - 1
            if remaining:
                self._held[before] = remaining
            else:
                del self._held[before]
        if held is not None:
            self._grants[grant_id] = held
            self._held[held] = self._held.get(held, 0) + 1

    async def _apply_grant(self, grant_id: UUID) -> None:
        """read one grant by key and put it in the cache: inserted, updated, or removed when gone.

        a read that fails is owed -- :meth:`allows` refuses non-admin checks until a read of that
        grant lands (the retry pass reads it again) -- unless a read that started later already
        landed. a read that lands after one that started later is dropped: it read older rows.

        :param grant_id: the grant a row message named
        :ptype grant_id: UUID
        :return: nothing
        :rtype: None
        :raises Exception: whatever the grant reader raised, after the read is marked owed
        """
        assert self._grant_reader is not None  # required with the registry that calls this
        self._reads_started += 1
        generation = self._reads_started
        landed = False
        try:
            row = await self._grant_reader(grant_id)
            held = None if row is None else _held_by(row)[1]
            landed = True
        finally:
            # no except: the reader is the caller's, so its error has no type this module can
            # name. the read is owed whatever it raised, and the error goes on to the caller's
            # boundary (the read's task, the retry pass), which logs it.
            if not landed and generation > max(self._key_landed.get(grant_id, 0), self._filled_generation):
                self._keys_owed[grant_id] = max(self._keys_owed.get(grant_id, 0), generation)
        if generation < max(self._key_landed.get(grant_id, 0), self._filled_generation):
            return
        self._key_landed[grant_id] = generation
        self._put(grant_id, held)
        if generation > self._keys_owed.get(grant_id, 0):
            self._keys_owed.pop(grant_id, None)

    async def _fill(self) -> None:
        """read every grant into the cache: the cold start, or the fill after a whole-table change.

        a fill that fails is owed and refuses non-admin checks until one lands. a grant whose keyed
        read started after this fill and landed keeps what that read found: it read newer rows.

        :return: nothing
        :rtype: None
        :raises Exception: whatever the loader raised, after the fill is marked owed
        """
        self._reads_started += 1
        generation = self._reads_started
        landed = False
        try:
            rows = await self._grant_loader()
            filled = dict(_held_by(row) for row in rows)
            landed = True
        finally:
            # no except: see _apply_grant. a fill that a later one already superseded owes nothing
            if not landed and generation > self._filled_generation:
                self._fill_failed_generation = max(self._fill_failed_generation, generation)
                self._fill_owed = True
        if generation < self._filled_generation:
            return
        for grant_id, landed_at in self._key_landed.items():
            if landed_at > generation:
                newer = self._grants.get(grant_id)
                if newer is None:
                    filled.pop(grant_id, None)
                else:
                    filled[grant_id] = newer
        self._grants = {}
        self._held = {}
        for grant_id, held in filled.items():
            self._put(grant_id, held)
        self._key_landed = {key: at for key, at in self._key_landed.items() if at > generation}
        self._keys_owed = {key: at for key, at in self._keys_owed.items() if at > generation}
        self._filled_generation = generation
        if generation > self._fill_failed_generation:
            self._fill_owed = False
        log.info("MCP grant cache filled", extra={"extra_data": {"grant_count": len(self._grants)}})
