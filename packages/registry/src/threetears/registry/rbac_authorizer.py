"""rbac-backed :class:`AgentToolAuthorizer` for the registry proxy.

namespace-task-01 phase 2 retires :class:`KvAgentToolAuthorizer` in
favour of the unified rbac evaluator from
:mod:`threetears.agent.acl`. every tool is a namespace of type
``tool`` (emitted by :class:`threetears.agent.tools.ToolServer` on
registration); a per-call authorization decision resolves to
"evaluator.evaluate(user_id, agent_id, namespace_id, action=tool.call)"
via the same machinery workspaces + datasources use.

wiring shape:

- one :class:`RbacEvaluatorAuthorizer` instance per registry pod
- constructed with a :class:`MembershipLoader`, a
  :class:`GrantLoader`, and a :class:`NamespaceByNameResolver` callable
- call path: :class:`~threetears.registry.proxy.CallProxy` invokes
  :meth:`is_authorized` on every tool dispatch; the authorizer
  resolves the tool's namespace id via the resolver, then asks the
  evaluator for a boolean decision

defense in depth: when ``user_id`` is ``None`` the authorizer
returns ``False`` (a tool dispatch without an identified user is
refused even when the agent is privileged). when the namespace
resolver returns ``None`` (tool not yet materialized; registration
race) the authorizer also returns ``False``.
"""

from __future__ import annotations

from typing import Awaitable, Callable
from uuid import UUID

from threetears.agent.acl import (
    EvaluationContext,
    GrantLoader,
    MembershipLoader,
    Namespace as AclNamespace,
    evaluate_decision,
)
from threetears.observe import get_logger

__all__ = [
    "NamespaceByNameResolver",
    "RbacEvaluatorAuthorizer",
    "ToolNamespaceRow",
]


log = get_logger(__name__)


TOOL_CALL_ACTION = "tool.call"


class ToolNamespaceRow:
    """resolved ``platform.namespaces`` row fields for a single tool.

    the Registry authorizer only needs the subset of
    ``platform.namespaces`` that the evaluator reads:

    - ``id`` to key per-namespace grant entries
    - ``namespace_type`` set to ``"tool"`` (structural invariant)
    - ``owner_agent_id`` + ``customer_id`` for the evaluator's owner
      short-circuit and cross-customer guard

    :ivar id: namespace UUID
    :ivar namespace_type: always ``"tool"`` for rows surfaced here
    :ivar owner_agent_id: owning agent (``None`` for platform tools)
    :ivar customer_id: owning customer (``None`` for platform tools)
    """

    __slots__ = ("id", "namespace_type", "owner_agent_id", "customer_id")

    def __init__(
        self,
        *,
        id: UUID,
        namespace_type: str,
        owner_agent_id: UUID | None,
        customer_id: UUID | None,
    ) -> None:
        """initialize a resolved tool namespace row.

        :param id: namespace UUID
        :ptype id: UUID
        :param namespace_type: namespace type (always ``"tool"``)
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID or ``None``
        :ptype owner_agent_id: UUID | None
        :param customer_id: owning customer UUID or ``None``
        :ptype customer_id: UUID | None
        """
        self.id = id
        self.namespace_type = namespace_type
        self.owner_agent_id = owner_agent_id
        self.customer_id = customer_id


NamespaceByNameResolver = Callable[[str], Awaitable[ToolNamespaceRow | None]]


class RbacEvaluatorAuthorizer:
    """authorize tool dispatch via the unified rbac evaluator.

    implements the :class:`~threetears.registry.auth.AgentToolAuthorizer`
    protocol. on each call, resolves the tool's ``platform.namespaces``
    row (caches the lookup), then asks
    :func:`~threetears.agent.acl.evaluate_decision` for a boolean on
    ``(user_id, agent_id, namespace, action="tool.call")``.

    platform built-in tools (``owner_agent_id=NULL``,
    ``customer_id=NULL``) require an explicit grant — the evaluator
    does not special-case them. an admin seeds one assignment on
    ``scope=all`` (or ``type_customer`` with ``namespace_type=tool``)
    binding the "default tool access" group to the caller's customer.
    this matches the shared-workspace pattern (no implicit grants on
    shared-type rows).

    :param membership_loader: actor -> memberships resolver
    :ptype membership_loader: MembershipLoader
    :param grant_loader: groups -> assignments + roles resolver
    :ptype grant_loader: GrantLoader
    :param namespace_resolver: async callable ``name -> ToolNamespaceRow | None``
        resolving a tool namespace by its canonical name
        (``tool:<mcp_name>:<version>``) to the row the evaluator
        reads. caller supplies whichever implementation matches the
        Registry's wiring (NATS-proxied L3 query, direct asyncpg pool,
        in-memory test fixture)
    :ptype namespace_resolver: NamespaceByNameResolver
    """

    def __init__(
        self,
        *,
        membership_loader: MembershipLoader,
        grant_loader: GrantLoader,
        namespace_resolver: NamespaceByNameResolver,
    ) -> None:
        """wire the authorizer to its loaders + namespace resolver.

        :param membership_loader: actor -> memberships resolver
        :ptype membership_loader: MembershipLoader
        :param grant_loader: groups -> assignments + roles resolver
        :ptype grant_loader: GrantLoader
        :param namespace_resolver: async callable resolving tool
            namespace by name to a :class:`ToolNamespaceRow`
        :ptype namespace_resolver: NamespaceByNameResolver
        """
        self._membership_loader = membership_loader
        self._grant_loader = grant_loader
        self._namespace_resolver = namespace_resolver

    async def is_authorized(
        self,
        agent_id: str,
        user_id: str | None,
        tool_name: str,
    ) -> bool:
        """resolve an authorization decision for a tool dispatch.

        the Registry's :class:`~threetears.registry.auth.AgentToolAuthorizer`
        protocol was widened from the two-argument signature used by
        :class:`KvAgentToolAuthorizer` to the three-argument shape
        here: the unified rbac evaluator needs the invoking user
        identity to resolve the user side of an intersection
        decision. the proxy sources ``user_id`` from
        ``ProxyCallRequest.context.user_id``.

        :param agent_id: calling agent UUID in string form (border
            conversion happens here)
        :ptype agent_id: str
        :param user_id: invoking user UUID in string form, or
            ``None`` when the dispatch carries no user identity
        :ptype user_id: str | None
        :param tool_name: fully qualified tool name (the
            ``mcp_name`` from the proxy request); resolved against
            the tool namespace registry via the configured resolver
        :ptype tool_name: str
        :return: True iff the evaluator grants the ``tool.call``
            action on the resolved tool namespace
        :rtype: bool
        """
        result = False
        try:
            agent_uuid = UUID(agent_id)
        except ValueError:
            log.warning(
                "rbac authorizer: invalid agent_id, denying",
                extra={"extra_data": {"agent_id": agent_id, "tool_name": tool_name}},
            )
            return result

        user_uuid: UUID | None = None
        if user_id is not None:
            try:
                user_uuid = UUID(user_id)
            except ValueError:
                log.warning(
                    "rbac authorizer: invalid user_id, denying",
                    extra={
                        "extra_data": {
                            "user_id": user_id,
                            "tool_name": tool_name,
                        }
                    },
                )
                return result

        # defense in depth: a tool dispatch without an identified
        # user cannot be authorized. the workspace side short-
        # circuits to the agent-owner shortcut instead, but tool
        # grants are always two-sided (user must have permission).
        if user_uuid is None:
            log.info(
                "rbac authorizer: no user_id on tool dispatch, denying",
                extra={
                    "extra_data": {
                        "agent_id": agent_id,
                        "tool_name": tool_name,
                    }
                },
            )
            return result

        ns_row = await self._namespace_resolver(tool_name)
        if ns_row is None:
            log.info(
                "rbac authorizer: no tool namespace row, denying",
                extra={
                    "extra_data": {
                        "agent_id": agent_id,
                        "user_id": user_id,
                        "tool_name": tool_name,
                    }
                },
            )
            return result

        evaluator_namespace = AclNamespace(
            id=ns_row.id,
            customer_id=ns_row.customer_id,
            namespace_type=ns_row.namespace_type,
            owner_agent_id=ns_row.owner_agent_id,
        )
        ctx = EvaluationContext(
            namespace=evaluator_namespace,
            action=TOOL_CALL_ACTION,
            user_id=user_uuid,
            agent_id=agent_uuid,
        )
        result = await evaluate_decision(
            ctx,
            membership_loader=self._membership_loader,
            grant_loader=self._grant_loader,
        )
        return result
