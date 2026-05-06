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

from threetears.epoch import EpochClient, EpochListener
from threetears.nats import Subjects
from threetears.observe import get_logger

__all__ = [
    "Authorizer",
    "EnvVarIdentityProvider",
    "Identity",
    "IdentityProvider",
    "LocalGrantAuthorizer",
    "PrincipalType",
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

    reads the admin user UUID from env vars (``METALLM_ADMIN_USER_ID``
    by default; configurable). future v2 :class:`BearerTokenIdentityProvider`
    will resolve per request from the MCP-client-supplied bearer
    token using the same :class:`Identity` shape.

    the admin flag is True by default -- the env-var creds in v1 are
    admin-equivalent (same as the existing prototype's behaviour).
    consumers wiring a non-admin identity should pass ``is_admin=False``
    explicitly.

    :param principal_id: user UUID; if ``None`` reads from
        ``user_id_env_var``
    :ptype principal_id: UUID | None
    :param user_id_env_var: env var name that holds the admin user
        UUID when ``principal_id`` is unset
    :ptype user_id_env_var: str
    :param is_admin: whether this identity has admin role; v1 default
        True matches the prototype's env-var-admin behaviour
    :ptype is_admin: bool
    """

    def __init__(
        self,
        *,
        principal_id: UUID | None = None,
        user_id_env_var: str = "METALLM_ADMIN_USER_ID",
        is_admin: bool = True,
    ) -> None:
        """capture identity source; resolve at :meth:`identify` call.

        :param principal_id: explicit UUID; bypasses env var when set
        :ptype principal_id: UUID | None
        :param user_id_env_var: env var consulted when ``principal_id``
            is ``None``
        :ptype user_id_env_var: str
        :param is_admin: admin flag; default True for v1 env-var creds
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
                    f"identity provider requires {self._env_var} "
                    "or an explicit principal_id at construction",
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


# ---------------------------------------------------------------------
# LocalGrantAuthorizer -- framework default impl
# ---------------------------------------------------------------------


GrantLoader = Callable[[], Awaitable[list[dict[str, Any]]]]
"""signature for the function that loads all grants from L3.

returns a list of dict rows shaped like ``mcp_tool_grants`` table
columns (``principal_type``, ``principal_id``, ``permission``).
:class:`LocalGrantAuthorizer` calls this on cold start and on every
:meth:`mcp.rbac` epoch bump to rebuild its in-memory cache.

abstracted as a function (not a direct collection reference) so
tests can substitute a fake without spinning up Postgres + the full
:class:`McpToolGrantCollection` plumbing.
"""


class LocalGrantAuthorizer:
    """framework default :class:`Authorizer` backed by ``mcp_tool_grants``.

    holds an in-memory grant cache keyed by ``(principal_id, permission)``.
    cold-start loads via the supplied :class:`GrantLoader`. cross-pod
    coherence: subscribes to :func:`Subjects.mcp_rbac_epoch` via
    task-02's :class:`EpochListener`; on bump reloads the cache from L3.

    admin auto-grant: when an :class:`Identity` arrives with
    ``is_admin=True``, every permission check returns True without
    consulting the cache. the auto-grant is logged at INFO at
    :meth:`start`-time so it's auditable; nothing is written to
    ``mcp_tool_grants``.

    :param grant_loader: function that returns the current grant
        rows from L3
    :ptype grant_loader: GrantLoader
    :param epoch_client: task-02 :class:`EpochClient`; used by the
        listener for cold-start prime via ``current(subject)``
    :ptype epoch_client: EpochClient
    :param epoch_listener: task-02 :class:`EpochListener`; subscribes
        to the rbac epoch
    :ptype epoch_listener: EpochListener
    """

    def __init__(
        self,
        *,
        grant_loader: GrantLoader,
        epoch_client: EpochClient,
        epoch_listener: EpochListener,
    ) -> None:
        """capture deps; no I/O until :meth:`start`.

        :param grant_loader: L3 grant loader
        :ptype grant_loader: GrantLoader
        :param epoch_client: task-02 epoch client
        :ptype epoch_client: EpochClient
        :param epoch_listener: task-02 epoch listener
        :ptype epoch_listener: EpochListener
        :return: nothing
        :rtype: None
        """
        self._grant_loader = grant_loader
        self._epoch_client = epoch_client
        self._epoch_listener = epoch_listener
        # cache key shape: (principal_id, permission) -> True. presence
        # is the grant; we don't store False entries because absence
        # means "no grant" by default-deny semantics.
        self._cache: set[tuple[UUID, str]] = set()
        self._started = False

    async def start(self) -> None:
        """prime the cache from L3 and subscribe to the rbac epoch.

        idempotent: subsequent calls log a warning and short-circuit.
        production code calls once at server startup; tests may
        re-construct.

        :return: nothing
        :rtype: None
        """
        if self._started:
            log.warning("LocalGrantAuthorizer.start called twice; ignoring")
            return
        await self._reload_cache()
        await self._epoch_listener.subscribe(
            Subjects.mcp_rbac_epoch(),
            self._on_rbac_bump,
        )
        log.info(
            "MCP authorizer started; admin identities are auto-granted "
            "every tool in memory (not persisted to mcp_tool_grants)",
            extra={"extra_data": {"grant_count": len(self._cache)}},
        )
        self._started = True

    async def allows(self, identity: Identity, permission: str) -> bool:
        """return True iff ``identity`` holds ``permission``.

        admin short-circuit fires before the cache lookup. otherwise
        consults the in-memory cache populated from
        ``mcp_tool_grants``. group/role-scoped grants are matched
        against the pre-resolved ``groups`` / ``roles`` on the
        :class:`Identity` (the :class:`IdentityProvider` is responsible
        for resolving them).

        :param identity: caller identity
        :ptype identity: Identity
        :param permission: permission string
        :ptype permission: str
        :return: True when the grant exists; False otherwise
        :rtype: bool
        """
        if identity.is_admin:
            return True
        if (identity.principal_id, permission) in self._cache:
            return True
        for group_id in identity.groups:
            if (group_id, permission) in self._cache:
                return True
        for role_id in identity.roles:
            if (role_id, permission) in self._cache:
                return True
        return False

    async def _reload_cache(self) -> None:
        """rebuild the in-memory grant cache from L3.

        called on cold start and on every ``mcp.rbac`` epoch bump.
        narrow exception scope: a load failure logs and leaves the
        prior cache in place rather than emptying it (denying every
        grant on a transient L3 hiccup would be a worse outcome
        than serving stale grants for one tick).

        :return: nothing
        :rtype: None
        """
        try:
            rows = await self._grant_loader()
        except Exception:
            log.warning(
                "MCP grant cache reload failed; keeping prior cache",
                exc_info=True,
                extra={"extra_data": {"prior_grant_count": len(self._cache)}},
            )
            return
        new_cache: set[tuple[UUID, str]] = set()
        for row in rows:
            principal_id = row["principal_id"]
            permission = row["permission"]
            if not isinstance(principal_id, UUID):
                principal_id = UUID(str(principal_id))
            new_cache.add((principal_id, permission))
        self._cache = new_cache
        log.info(
            "MCP grant cache reloaded",
            extra={"extra_data": {"grant_count": len(self._cache)}},
        )

    async def _on_rbac_bump(self, epoch: int, payload: dict[str, Any] | None) -> None:
        """epoch-listener callback: reload cache on rbac bump.

        :param epoch: new epoch returned by the bump
        :ptype epoch: int
        :param payload: opaque hint from the publisher; framework
            does not inspect
        :ptype payload: dict[str, Any] | None
        :return: nothing
        :rtype: None
        """
        log.info(
            "MCP rbac epoch bump received; reloading grant cache",
            extra={"extra_data": {"epoch": epoch, "payload": payload}},
        )
        await self._reload_cache()
