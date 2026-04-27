"""registry-side rbac stack for the standalone ``_run_server()`` entrypoint.

mirrors the agent SDK's
:func:`aibots_agents.runtime.three_tier_stack.build_three_tier_stack`
with the agent-specific bits stripped: there is no agent identity
on the registry process, no agent main pool, no agent-owned
Collections. what remains is the rbac surface the
:class:`~threetears.registry.rbac_authorizer.RbacEvaluatorAuthorizer`
needs to resolve ``tool.call`` decisions:

- one :class:`NatsProxyL3Backend` pinned to
  :data:`PLATFORM_RBAC_READ_NAMESPACE` (``system.platform.rbac``).
  the hub broker's read-only carve-out admits SELECT against this
  namespace regardless of the SQL target table -- the registry uses
  this single pool for the four rbac metadata tables AND for
  ``platform.namespaces`` (read-only lookups during authorization).
- five canonical Collections from :mod:`threetears.agent.acl`
  (``NamespaceCollection`` + ``Group`` / ``GroupMember`` / ``Role`` /
  ``RoleAssignment``).
- :class:`CollectionMembershipLoader` + :class:`CollectionGrantLoader`
  fronting the four rbac metadata Collections.
- :class:`AclCache` three-layer ttl cache with default 60s TTL.
- NATS subscriptions on ``{ns}.acl.membership.invalidate`` /
  ``{ns}.acl.assignment.invalidate`` / ``{ns}.acl.role.invalidate``
  so cross-process rbac mutations purge the cache promptly instead
  of waiting on TTL.

construction is synchronous; callers invoke
:meth:`RegistryRbacStack.subscribe_invalidations` after start so the
sync-vs-async split mirrors the agent stack pattern.

note: the registry pod has no real agent identity. the
:class:`NatsProxyL3Backend` requires an ``agent_id`` string which the
broker stamps on logs but does NOT use to gate
``system.platform.rbac`` reads (the carve-out keys on namespace +
action only). the registry passes a service sentinel UUID so log
correlation has a stable identifier; the broker still admits the
read because of the namespace+action match.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_DNS, UUID, uuid5

from pydantic import ValidationError
from threetears.agent.acl import (
    AclCache,
    AssignmentInvalidatePayload,
    CollectionGrantLoader,
    CollectionMembershipLoader,
    GroupCollection,
    GroupMemberCollection,
    MembershipInvalidatePayload,
    NamespaceCollection,
    RoleAssignmentCollection,
    RoleCollection,
    RoleInvalidatePayload,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.nats import IncomingMessage, NatsClient, Subjects
from threetears.observe import get_logger

__all__ = [
    "REGISTRY_SERVICE_SENTINEL_AGENT_ID",
    "PLATFORM_RBAC_READ_NAMESPACE",
    "RegistryRbacStack",
    "build_registry_rbac_stack",
]


log = get_logger(__name__)


#: name of the system namespace the hub broker's read-only carve-out
#: admits SELECT traffic on. the value is duplicated here (instead of
#: imported from ``aibots.hub.broker.acl``) so the 3tears registry
#: package retains its hub-independent dependency graph -- the hub is
#: free to evolve the carve-out internally as long as the canonical
#: name stays the same.
PLATFORM_RBAC_READ_NAMESPACE: str = "system.platform.rbac"


#: deterministic uuid5 sentinel used as the ``agent_id`` on every
#: :class:`NatsProxyL3Backend` request the registry issues. the broker
#: stamps this id on logs for traceability but does NOT gate
#: ``system.platform.rbac`` reads on it -- the carve-out keys on
#: namespace + action. keeping the value deterministic across registry
#: restarts means broker logs always show the same originator string
#: for registry-side reads, which simplifies operator triage.
REGISTRY_SERVICE_SENTINEL_AGENT_ID: UUID = uuid5(
    NAMESPACE_DNS,
    "threetears.registry.service-sentinel",
)


@dataclass
class RegistryRbacStack:
    """bundle of rbac primitives constructed for the registry process.

    :param l1_backend: in-process SQLite backend the rbac Collections
        snap as their L1 tier (warm reads stay in-process)
    :ptype l1_backend: SQLiteBackend
    :param registry: collection registry with the rbac pool bound and
        the L1 backend wired
    :ptype registry: CollectionRegistry
    :param namespace_collection: canonical
        :class:`NamespaceCollection`; consumed by
        :class:`RbacEvaluatorAuthorizer` for the
        ``get_by_name(canonical_tool_name)`` lookup
    :ptype namespace_collection: NamespaceCollection
    :param acl_cache: canonical :class:`AclCache` with membership +
        grant loaders fronting the four rbac metadata Collections;
        consumed by :class:`RbacEvaluatorAuthorizer` for the
        ``evaluate_decision`` hot path
    :ptype acl_cache: AclCache
    """

    l1_backend: SQLiteBackend
    registry: CollectionRegistry
    namespace_collection: NamespaceCollection
    group_collection: GroupCollection
    group_member_collection: GroupMemberCollection
    role_collection: RoleCollection
    role_assignment_collection: RoleAssignmentCollection
    membership_loader: CollectionMembershipLoader
    grant_loader: CollectionGrantLoader
    acl_cache: AclCache
    nats_client: NatsClient
    subject_namespace: str
    _membership_subscription: Any = None
    _assignment_subscription: Any = None
    _role_subscription: Any = None

    async def subscribe_invalidations(self) -> None:
        """bind the three rbac invalidation subjects to handlers.

        cross-process rbac mutations (admin tools rewriting
        ``role_assignments`` / ``group_members`` / ``roles``)
        publish typed payloads on
        ``{ns}.acl.membership.invalidate`` /
        ``{ns}.acl.assignment.invalidate`` /
        ``{ns}.acl.role.invalidate``. without these subscriptions the
        registry's :class:`AclCache` stays warm with stale tuples for
        up to ``ttl_seconds`` after a mutation. each handler logs
        unparseable payloads at WARNING + invalidates the cache (the
        canonical "fail safe" behaviour shared with the agent stack).

        :return: nothing
        :rtype: None
        """
        membership_subject = Subjects.acl_invalidate(kind="membership")
        assignment_subject = Subjects.acl_invalidate(kind="assignment")
        role_subject = Subjects.acl_invalidate(kind="role")

        self._membership_subscription = await self.nats_client.subscribe(
            subject=membership_subject,
            cb=self._handle_membership_invalidation,
        )
        self._assignment_subscription = await self.nats_client.subscribe(
            subject=assignment_subject,
            cb=self._handle_assignment_invalidation,
        )
        self._role_subscription = await self.nats_client.subscribe(
            subject=role_subject,
            cb=self._handle_role_invalidation,
        )
        log.info(
            "registry rbac stack subscribed to invalidations",
            extra={
                "extra_data": {
                    "membership_subject": membership_subject.path,
                    "assignment_subject": assignment_subject.path,
                    "role_subject": role_subject.path,
                }
            },
        )

    async def close(self) -> None:
        """unsubscribe invalidation handlers and reset the L1 backend.

        the registry owns the NATS client lifecycle separately; this
        method only releases the rbac stack's own resources.

        :return: nothing
        :rtype: None
        """
        for sub in (
            self._membership_subscription,
            self._assignment_subscription,
            self._role_subscription,
        ):
            if sub is not None:
                try:
                    await self.nats_client.unsubscribe(sub)
                except Exception as exc:
                    log.warning(
                        "registry rbac stack unsubscribe failed",
                        extra={"extra_data": {"error": str(exc)}},
                    )
        self._membership_subscription = None
        self._assignment_subscription = None
        self._role_subscription = None
        self.l1_backend.reset()
        log.info("registry rbac stack closed")

    async def _handle_membership_invalidation(self, msg: IncomingMessage) -> None:
        """drop cached memberships for the actor named in the payload.

        per-actor invalidation: the canonical evaluator caches one
        tuple per ``(actor_type, actor_id)``; the loader's
        ``invalidate(actor_type, actor_id)`` purges only that key
        which is much cheaper than a full sweep when many actors
        share the cache.

        :param msg: incoming wrapper envelope carrying the payload
        :ptype msg: IncomingMessage
        :return: nothing
        :rtype: None
        """
        try:
            payload = MembershipInvalidatePayload.model_validate_json(msg.data)
        except ValidationError:
            log.warning(
                "registry acl cache: membership.invalidate payload unparseable size=%d",
                len(msg.data),
            )
            return
        self.acl_cache.invalidate_memberships(
            actor_type=payload.actor_type,
            actor_id=payload.actor_id,
        )

    async def _handle_assignment_invalidation(self, msg: IncomingMessage) -> None:
        """drop cached grant tuples for the group named in the payload.

        :param msg: incoming wrapper envelope
        :ptype msg: IncomingMessage
        :return: nothing
        :rtype: None
        """
        try:
            payload = AssignmentInvalidatePayload.model_validate_json(msg.data)
        except ValidationError:
            log.warning(
                "registry acl cache: assignment.invalidate payload unparseable size=%d",
                len(msg.data),
            )
            return
        self.acl_cache.invalidate_grants(group_id=payload.group_id)

    async def _handle_role_invalidation(self, msg: IncomingMessage) -> None:
        """drop the entire grant cache when role permissions mutate.

        role permissions are denormalized into the cached grant
        tuples; a per-role invalidation would require walking every
        cached grant looking for matches. nuking the cache is
        coarser but keeps the invariant that no stale role
        permissions ever survive a role mutation.

        :param msg: incoming wrapper envelope
        :ptype msg: IncomingMessage
        :return: nothing
        :rtype: None
        """
        try:
            RoleInvalidatePayload.model_validate_json(msg.data)
        except ValidationError:
            log.warning(
                "registry acl cache: role.invalidate payload unparseable size=%d",
                len(msg.data),
            )
            return
        self.acl_cache.invalidate_all()


def _resolve_acl_ttl_seconds() -> int:
    """resolve the AclCache TTL from the registry's env knob.

    :data:`THREETEARS_REGISTRY_ACL_TTL_SECONDS` overrides the default
    (60 seconds, matching the hub-side cache). the env-var path is
    bounded -- values <= 0 fall back to the default rather than
    disabling the cache entirely (which would defeat the rbac
    fast-path).

    :return: TTL in seconds (>= 1)
    :rtype: int
    """
    raw = os.environ.get("THREETEARS_REGISTRY_ACL_TTL_SECONDS", "")
    result = 60
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                result = parsed
        except ValueError:
            log.warning(
                "THREETEARS_REGISTRY_ACL_TTL_SECONDS unparseable, falling back to default 60s: value=%s",
                raw,
            )
    return result


def build_registry_rbac_stack(
    *,
    nats_client: NatsClient,
    subject_namespace: str,
    l1_backend: SQLiteBackend,
) -> RegistryRbacStack:
    """construct the registry-side rbac stack.

    builds a single :class:`NatsProxyL3Backend` pinned to
    :data:`PLATFORM_RBAC_READ_NAMESPACE` and binds it as the default
    L3 pool on a :class:`CollectionRegistry`. all five rbac
    Collections (namespace + four metadata) snap that pool at
    construction time. the rbac authorizer's hot path is one
    :meth:`NamespaceCollection.get_by_name` plus one
    :func:`evaluate_decision` against the cache, both served from
    the in-process L1 mirror after the first warm-up.

    :param nats_client: connected canonical NATS wrapper client; used
        for the proxy backend AND for the invalidation subscriptions
        (bound by :meth:`RegistryRbacStack.subscribe_invalidations`)
    :ptype nats_client: NatsClient
    :param subject_namespace: NATS subject namespace prefix; threaded
        into both the proxy backend and the invalidation subjects
    :ptype subject_namespace: str
    :param l1_backend: shared SQLite L1 backend; wired into the
        collection registry as the default L1 tier so all rbac
        Collections snap it at construction. caller is responsible
        for adding the rbac metadata tables to the backend's schema
        (see :data:`registry.l1_cache.REGISTRY_L1_METADATA`).
    :ptype l1_backend: SQLiteBackend
    :return: populated rbac stack ready for downstream consumers
    :rtype: RegistryRbacStack
    """
    core_config = DefaultCoreConfig()
    registry = CollectionRegistry()

    rbac_pool = NatsProxyL3Backend(
        nats_client=nats_client.raw,
        namespace_prefix=subject_namespace,
        agent_id=str(REGISTRY_SERVICE_SENTINEL_AGENT_ID),
        default_namespace=PLATFORM_RBAC_READ_NAMESPACE,
    )

    registry.configure(l1_backend=l1_backend, l3_pool=rbac_pool)

    namespace_collection = NamespaceCollection(
        registry=registry,
        config=core_config,
        nats_client=nats_client,
    )
    group_collection = GroupCollection(
        registry=registry,
        config=core_config,
        nats_client=nats_client,
    )
    group_member_collection = GroupMemberCollection(
        registry=registry,
        config=core_config,
        nats_client=nats_client,
    )
    role_collection = RoleCollection(
        registry=registry,
        config=core_config,
        nats_client=nats_client,
    )
    role_assignment_collection = RoleAssignmentCollection(
        registry=registry,
        config=core_config,
        nats_client=nats_client,
    )

    membership_loader = CollectionMembershipLoader(
        collection=group_member_collection,
    )
    grant_loader = CollectionGrantLoader(
        assignment_collection=role_assignment_collection,
        role_collection=role_collection,
        group_collection=group_collection,
    )
    acl_cache = AclCache(
        membership_loader=membership_loader,
        grant_loader=grant_loader,
        ttl_seconds=_resolve_acl_ttl_seconds(),
    )

    return RegistryRbacStack(
        l1_backend=l1_backend,
        registry=registry,
        namespace_collection=namespace_collection,
        group_collection=group_collection,
        group_member_collection=group_member_collection,
        role_collection=role_collection,
        role_assignment_collection=role_assignment_collection,
        membership_loader=membership_loader,
        grant_loader=grant_loader,
        acl_cache=acl_cache,
        nats_client=nats_client,
        subject_namespace=subject_namespace,
    )
