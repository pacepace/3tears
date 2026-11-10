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
  framework default backed by :class:`McpToolGrantCollection` (with
  task-02 epoch refresh). per-product impls can plug in a different
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

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from threetears.observe import PeriodicTask, contain, get_logger

# LocalGrantAuthorizer's collaborators come from the ``grants`` extra. Nothing else in this module
# needs them, and a server with its own Authorizer must import this module without them, so their
# absence is recorded here and LocalGrantAuthorizer refuses to build, naming the extra.
_GRANTS_IMPORT_ERROR: ImportError | None = None
try:
    from threetears.epoch import EpochClient, EpochListener, catchup_tick
    from threetears.nats import Subject, Subjects
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
to fill its empty cache at :meth:`~LocalGrantAuthorizer.start`, and after an operator reset of the
rbac epoch (the counter was replaced, so no bump says what changed). a bump never reaches it.

abstracted as a function (not a direct collection reference) so tests can substitute a fake
without spinning up Postgres + the full :class:`McpToolGrantCollection` plumbing.
"""


GrantReader = Callable[[UUID], Awaitable[dict[str, Any] | None]]
"""signature for the function that reads ONE grant from L3 by its key.

returns the grant's row (shaped as a :data:`GrantLoader` row), or ``None`` when no grant has that
key -- it was revoked. :class:`LocalGrantAuthorizer` calls it for the grant each rbac bump names
(:meth:`McpToolGrantCollection.read_grant`).
"""


def _require_grants_extra() -> None:
    """refuse to build the grant store's authorizer where its extra is not installed.

    :return: nothing
    :rtype: None
    :raises ImportError: when ``threetears.epoch`` or ``threetears.nats`` could not be imported
    """
    if _GRANTS_IMPORT_ERROR is not None:
        raise ImportError(
            "LocalGrantAuthorizer is the MCP grant store, which needs the grants extra: install 3tears-mcp[grants]",
        ) from _GRANTS_IMPORT_ERROR


def _rbac_epoch_subject() -> Subject:
    """the rbac epoch subject, resolved when it is used so the caller's namespace applies.

    :return: the subject
    :rtype: Subject
    """
    return Subjects.mcp_rbac_epoch()


def _grant_named_by(payload: dict[str, Any] | None) -> UUID | None:
    """the grant a bump's payload names, or ``None`` when it names none it can be read by.

    the hub's grant endpoints bump with ``{"grant_id": "<uuid>", "action": "create" | "delete"}``.

    :param payload: the bump's payload
    :ptype payload: dict[str, Any] | None
    :return: the grant's key, or ``None``
    :rtype: UUID | None
    """
    raw = None if payload is None else payload.get("grant_id")
    if not isinstance(raw, str):
        return None
    try:
        return UUID(raw)
    except ValueError:  # NOSILENT: a key that is no uuid names no grant; the caller counts the epoch unheard
        return None


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

    raised by :meth:`LocalGrantAuthorizer.allows` for a non-admin identity while the cache may
    hold a grant revoked since: a load or a keyed read failed and is owed, or rbac epochs went
    unheard and the grants they changed are unknown. answering ``False`` instead would read as
    "you lack this permission", which is not what happened; answering from the cache would keep a
    revoked grant allowed. the MCP server turns it into an ``AUTHZ_ERROR`` result and logs it.
    """


class LocalGrantAuthorizer:
    """framework default :class:`Authorizer` backed by ``mcp_tool_grants``.

    holds an in-memory grant cache: each grant by its key, and how many grants give each
    ``(principal_id, permission)``. :meth:`start` fills it once from the supplied
    :class:`GrantLoader` -- the cold start. cross-pod coherence: subscribes to
    :func:`Subjects.mcp_rbac_epoch` via :class:`EpochListener`; each bump names the grant it
    changed, and the authorizer reads that one grant by key (:class:`GrantReader`) and inserts,
    updates or removes it. nothing but the cold start and an operator reset (the epoch counter
    was REPLACED) reads every grant.

    **an epoch whose grant is unknown is not reloaded around.** a bump that names no grant, an
    epoch skipped between two bumps, or a bump the catch-up tick finds the broadcast never
    delivered: the rbac epoch is a NATS KV counter that keeps no record of what each bump
    changed, so the grant it changed cannot be read by key. the cache may then hold a revoked
    grant, so non-admin checks are refused (:class:`GrantCacheUnavailableError`) until an
    operator reset or a restart fills the cache again; the gap is logged at ERROR.

    admin auto-grant: when an :class:`Identity` arrives with ``is_admin=True``, every
    permission check returns True without consulting the cache. the auto-grant is logged at
    INFO at :meth:`start`-time so it's auditable; nothing is written to ``mcp_tool_grants``.

    :param grant_loader: reads every grant row from L3; called only for the cold start and an
        operator reset
    :ptype grant_loader: GrantLoader
    :param grant_reader: reads one grant row from L3 by key; required with an epoch listener
    :ptype grant_reader: GrantReader | None
    :param epoch_client: :class:`EpochClient`; read for the epoch a fill reflects. Optional:
        when ``None``, the cache is filled once at :meth:`start` and never changed -- correct for
        single-process modes (stdio MCP servers, tests) where there is no other writer to
        broadcast against.
    :ptype epoch_client: EpochClient | None
    :param epoch_listener: :class:`EpochListener`; subscribes to the rbac epoch. Optional: when
        ``None`` (or when ``epoch_client`` is None), :meth:`start` skips the broadcast
        subscription and the periodic catch-up loop. Same single-process rationale as
        ``epoch_client``.
    :ptype epoch_listener: EpochListener | None
    :param admin_principal_ids: optional set of principal UUIDs to log explicitly at
        start-time as auto-granted. logging the specific principal_id (not just "admins get
        everything") keeps the audit trail concrete for the v1 single-identity stdio mode
    :ptype admin_principal_ids: set[UUID] | None
    :param catchup_interval_seconds: how often the periodic catch-up tick compares the counter
        with what the listener heard, and retries owed reads. default 60 matches the
        capabilities-epoch tick
    :ptype catchup_interval_seconds: float
    """

    def __init__(
        self,
        *,
        grant_loader: GrantLoader,
        grant_reader: GrantReader | None = None,
        epoch_client: EpochClient | None = None,
        epoch_listener: EpochListener | None = None,
        admin_principal_ids: set[UUID] | None = None,
        catchup_interval_seconds: float = 60.0,
    ) -> None:
        """capture deps; no I/O until :meth:`start`.

        ``epoch_client`` and ``epoch_listener`` are jointly optional -- single-process modes
        (stdio MCP servers, tests) pass neither; multi-pod modes pass both, and a
        ``grant_reader`` to apply each bump with.

        :param grant_loader: L3 read of every grant, for the cold start and an operator reset
        :ptype grant_loader: GrantLoader
        :param grant_reader: L3 read of one grant by key, for each bump; required with an
            epoch listener
        :ptype grant_reader: GrantReader | None
        :param epoch_client: epoch client; optional
        :ptype epoch_client: EpochClient | None
        :param epoch_listener: epoch listener; optional
        :ptype epoch_listener: EpochListener | None
        :param admin_principal_ids: principals to log as auto-granted
        :ptype admin_principal_ids: set[UUID] | None
        :param catchup_interval_seconds: periodic catch-up interval (only consulted when
            ``epoch_listener`` is provided)
        :ptype catchup_interval_seconds: float
        :return: nothing
        :rtype: None
        :raises ValueError: when exactly one of epoch_client / epoch_listener is provided (must
            be both or neither), when an epoch listener comes without a grant reader, or when
            the catch-up interval is not positive
        :raises ImportError: when the ``grants`` extra is not installed
        """
        _require_grants_extra()
        if (epoch_client is None) != (epoch_listener is None):
            raise ValueError(
                "epoch_client and epoch_listener must be provided together; passing exactly one is a usage error",
            )
        if epoch_listener is not None and grant_reader is None:
            raise ValueError(
                "an rbac bump is applied by reading the grant it names: pass grant_reader "
                "(McpToolGrantCollection.read_grant) with the epoch listener",
            )
        if epoch_listener is not None and not (0 < catchup_interval_seconds < float("inf")):
            # refused HERE, not when the loop is built: by then start() has already primed the cache
            # and subscribed, and a failure there would leave a registration behind with no loop.
            raise ValueError(f"catchup_interval_seconds must be positive, got {catchup_interval_seconds}")
        self._grant_loader = grant_loader
        self._grant_reader = grant_reader
        self._epoch_client = epoch_client
        self._epoch_listener = epoch_listener
        self._admin_principal_ids = admin_principal_ids or set()
        self._catchup_interval_seconds = catchup_interval_seconds
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
        # owed every non-admin check is refused, since the bump before it may have been a
        # revocation.
        self._filled_generation = 0
        self._fill_failed_generation = 0
        self._fill_owed = False
        # per grant: the newest keyed read landed, and a failed one still owed
        self._key_landed: dict[UUID, int] = {}
        self._keys_owed: dict[UUID, int] = {}
        # the newest rbac epoch the cache holds the change of, with every epoch before it
        self._epoch = 0
        # set to the reads started so far when an epoch's grant goes unknown; a fill that
        # starts after it clears it. ``None`` while every epoch's grant is known.
        self._unheard_at: int | None = None
        # a counter value the catch-up tick saw ahead of what the listener heard, given one
        # interval for its broadcast to arrive before the tick advances the listener past it
        self._awaiting_broadcast: int | None = None
        self._started = False
        self._catchup_task: PeriodicTask | None = None

    async def start(self) -> None:
        """subscribe to the rbac epoch, fill the cache once, spawn the catch-up tick.

        idempotent: subsequent calls log a warning and short-circuit. the counter is read and
        the subscription made BEFORE the fill, so a bump landing during the fill is heard and
        applied by key (or is already in the fill, which read the counter before its rows), and
        none falls between the fill and the subscription. the catch-up tick is the safety net
        for a broadcast lost anyway, and retries an owed read.

        :return: nothing
        :rtype: None
        :raises RuntimeError: in single-process mode, when the fill fails: nothing would retry it
        """
        if self._started:
            log.warning("LocalGrantAuthorizer.start called twice; ignoring")
            return
        epoch_mode = self._epoch_listener is not None
        if epoch_mode:
            assert self._epoch_listener is not None  # narrowed by epoch_mode
            assert self._epoch_client is not None  # constructed together
            primed_epoch = await self._epoch_client.current(_rbac_epoch_subject())
            self._epoch = primed_epoch
            # sender-posture: trigger-only -- a bump names the grant it changed, read here by key
            await self._epoch_listener.subscribe(
                _rbac_epoch_subject(),
                self._on_rbac_bump,
                primed_epoch=primed_epoch,
                on_reset=self._on_rbac_reset,
            )
        try:
            await self._fill()
        except Exception as exc:
            # the grant loader is the caller's and has no error contract of its own; whatever it
            # raised, the fill is already marked owed (in _fill) and every non-admin check refused.
            # single-process mode has no tick to retry it, so this authorizer must not start.
            if not epoch_mode:
                raise RuntimeError("MCP grant cache could not be loaded at start, and nothing would retry it") from exc
            contain(
                exc,
                log=log,
                what="MCP grant cache could not be filled at start; refusing non-admin checks "
                "until the catch-up tick's retry lands",
                level="error",
            )
        log_extras: dict[str, Any] = {
            "grant_count": len(self._grants),
            "epoch_mode": "multi-pod" if epoch_mode else "single-process",
        }
        if epoch_mode:
            log_extras["catchup_interval_seconds"] = self._catchup_interval_seconds
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
        if epoch_mode:
            # sleeps first; a tick that errors is logged and retried on the next interval.
            self._catchup_task = PeriodicTask(
                self._catchup_once,
                interval=self._catchup_interval_seconds,
                name="mcp-rbac-catchup-loop",
                logger=log,
                failure_message="MCP rbac catch-up tick errored; will retry on next interval",
            )
            self._catchup_task.start()
        self._started = True

    async def stop(self) -> None:
        """cancel the periodic catch-up tick and await its exit.

        idempotent: subsequent calls are no-ops. callers (typically
        :meth:`McpServer.stop` or a lifespan teardown) invoke once
        on shutdown.

        ``_started`` is cleared FIRST, on every path. It used to be set only
        after the task teardown, below an early return taken whenever there was
        no task -- so a single-process authorizer (no epoch listener, hence no
        catch-up task) stayed marked started forever, and :meth:`start`'s
        double-start guard then refused to bring it back up. A stop that leaves
        the object unable to start again is not idempotent, it is one-way.

        :return: nothing
        :rtype: None
        """
        self._started = False
        self._awaiting_broadcast = None
        if self._catchup_task is None:
            return
        # the handle is released BEFORE the await. clearing it afterwards means
        # a raise during teardown leaves a cancelled task attached to an object
        # that already reads as not-started, and the next start() overwrites it.
        task = self._catchup_task
        self._catchup_task = None
        await task.stop()
        if self._epoch_listener is not None:
            # the listener's registration outlives the task otherwise, so a
            # later reset would call back into a stopped authorizer and ask it
            # to reload a cache it no longer maintains.
            #
            # Scoped to OUR callback: the listener keeps one registration list
            # per subject because two consumers on one subject are supported,
            # and a bare deregister would unregister every other authorizer
            # sharing this listener along with this one.
            self._epoch_listener.deregister(_rbac_epoch_subject(), self._on_rbac_bump)
        log.info("MCP authorizer catch-up loop stopped")

    async def _catchup_once(self) -> None:
        """one safety-net pass: retry owed reads, then find a broadcast the listener never heard.

        an owed fill (the cold start's or a reset's, which failed) is retried whole; an owed
        grant is retried by its key. then the counter is compared with what the listener heard.
        one ahead of it is given ONE interval for its broadcast to arrive -- advancing the
        listener at once would make it drop that broadcast, payload and all, when it lands a
        moment later. still unheard a whole interval on, the broadcast was lost (subscriber
        blip, dropped message) and the framework's pass advances the listener: the epoch's
        grant is unknown, which :meth:`_on_rbac_bump` records. the framework's pass is also
        what finds a REPLACED counter, which every KV operation survives silently. cheap when
        nothing has changed -- one counter read. cadence, failure isolation and shutdown are
        ``PeriodicTask``'s; a retry that fails again raises out of the pass, so each is logged.

        :return: nothing
        :rtype: None
        """
        # only ever started under epoch_mode, so both are non-None here.
        assert self._epoch_listener is not None
        assert self._epoch_client is not None
        if self._fill_owed:
            await self._fill()
        for grant_id in tuple(self._keys_owed):
            await self._apply_grant(grant_id)
        subject = _rbac_epoch_subject()
        current = await self._epoch_client.current(subject)
        heard = self._epoch_listener.last_seen(subject)
        if current > heard and (self._awaiting_broadcast is None or heard >= self._awaiting_broadcast):
            self._awaiting_broadcast = current
            return
        self._awaiting_broadcast = None
        # the pass itself is the framework's: which subjects to poll, and whether one failing
        # subject abandons the rest, are not decisions this consumer should hold its own opinion about.
        await catchup_tick(self._epoch_listener, [(subject, self._on_rbac_bump)])

    async def allows(self, identity: Identity, permission: str) -> bool:
        """return True iff ``identity`` holds ``permission``.

        admin short-circuit fires before the cache lookup. otherwise
        consults the in-memory cache of ``mcp_tool_grants``. group/role-scoped
        grants are matched against the pre-resolved ``groups`` / ``roles`` on the
        :class:`Identity` (the :class:`IdentityProvider` is responsible
        for resolving them).

        :param identity: caller identity
        :ptype identity: Identity
        :param permission: permission string
        :ptype permission: str
        :return: True when the grant exists; False otherwise
        :rtype: bool
        :raises GrantCacheUnavailableError: the cache may hold a revoked grant -- a fill or a
            keyed read failed, or an epoch's grant is unknown (admins are unaffected)
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
        """why the cache may hold a revoked grant, or ``None`` when it holds every change heard.

        :return: the reason, or ``None``
        :rtype: str | None
        """
        if self._fill_owed:
            return "the last load of the grants failed"
        if self._keys_owed:
            return f"the read of {len(self._keys_owed)} changed grant(s) failed"
        if self._unheard_at is not None:
            return "rbac epochs went unheard and the grants they changed are unknown"
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

        a read that fails is owed -- :meth:`allows` refuses non-admin checks until a read of
        that grant lands (the catch-up tick retries it) -- unless a read that started later
        already landed. a read that lands after one that started later is dropped: it read
        older rows.

        :param grant_id: the grant a bump named
        :ptype grant_id: UUID
        :return: nothing
        :rtype: None
        :raises Exception: whatever the grant reader raised, after the read is marked owed
        """
        assert self._grant_reader is not None  # required with the listener that calls this
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
            # boundary (the subscription's dispatcher, the catch-up tick), which logs it.
            if not landed and generation > max(self._key_landed.get(grant_id, 0), self._filled_generation):
                self._keys_owed[grant_id] = max(self._keys_owed.get(grant_id, 0), generation)
        if generation < max(self._key_landed.get(grant_id, 0), self._filled_generation):
            return
        self._key_landed[grant_id] = generation
        self._put(grant_id, held)
        if generation > self._keys_owed.get(grant_id, 0):
            self._keys_owed.pop(grant_id, None)
        log.info(
            "MCP grant applied by key",
            extra={"extra_data": {"grant_id": str(grant_id), "present": held is not None}},
        )

    async def _fill(self) -> None:
        """read every grant into the cache: the cold start, or the fill after an operator reset.

        the counter is read first, so the fill holds every epoch up to it. a fill that fails is
        owed and refuses non-admin checks until one lands. a grant whose keyed read started
        after this fill and landed keeps what that read found: it read newer rows.

        :return: nothing
        :rtype: None
        :raises Exception: whatever the loader or the counter read raised, after the fill is
            marked owed
        """
        self._reads_started += 1
        generation = self._reads_started
        landed = False
        try:
            if self._epoch_client is not None:
                # read BEFORE the rows: every change at or below it is in them, so a bump at or
                # below it needs no read of its own
                self._epoch = max(self._epoch, await self._epoch_client.current(_rbac_epoch_subject()))
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
        if self._unheard_at is not None and generation > self._unheard_at:
            self._unheard_at = None
        log.info(
            "MCP grant cache filled",
            extra={"extra_data": {"grant_count": len(self._grants), "epoch": self._epoch}},
        )

    async def _on_rbac_reset(self) -> None:
        """epoch-listener callback: the epoch counter was REPLACED, so fill the cache again.

        the operator reset, and with the cold start the only time every grant is read: the new
        counter numbers nothing this pod heard, so no bump can say what changed while the old
        one went away. Distinct from :meth:`_on_rbac_bump` because it carries no epoch, and
        deliberately so: every number this pod compared against belongs to a counter that no
        longer exists, and a reset dressed as a bump would be swallowed by epoch dedupe.

        :return: nothing
        :rtype: None
        """
        log.warning(
            "MCP rbac epoch counter was replaced; filling the grant cache again",
            extra={"extra_data": {"grant_count": len(self._grants)}},
        )
        self._epoch = 0
        self._awaiting_broadcast = None
        await self._fill()

    async def _on_rbac_bump(self, epoch: int, payload: dict[str, Any] | None) -> None:
        """epoch-listener callback: apply the grant the bump names, by key.

        an epoch at or below the one the cache holds is already in it (a fill read the counter
        past it before reading its rows). an epoch whose grant is unknown -- one skipped since
        the last, or this one when its payload names no grant -- is recorded and logged at
        ERROR, and non-admin checks are refused until a fill: nothing durable says which grant
        it changed, so it cannot be read by key, and no whole reload stands in for it.

        :param epoch: new epoch returned by the bump
        :ptype epoch: int
        :param payload: the publisher's ``{"grant_id": ..., "action": ...}``; ``None`` from a
            catch-up pass
        :ptype payload: dict[str, Any] | None
        :return: nothing
        :rtype: None
        """
        if epoch <= self._epoch:
            return
        grant_id = _grant_named_by(payload)
        first_unheard = self._epoch + 1
        last_unheard = epoch if grant_id is None else epoch - 1
        self._epoch = epoch
        if last_unheard >= first_unheard:
            self._unheard_at = self._reads_started
            log.error(
                "MCP rbac epochs went unheard; the grants they changed are unknown, so non-admin "
                "checks are refused until an operator reset or a restart fills the grant cache",
                extra={
                    "extra_data": {
                        "first_unheard_epoch": first_unheard,
                        "last_unheard_epoch": last_unheard,
                        "payload": payload,
                    },
                },
            )
        if grant_id is not None:
            await self._apply_grant(grant_id)
