"""rbac-backed :class:`AgentToolAuthorizer` for the registry proxy.

namespace-task-01 phase 2 retires :class:`KvAgentToolAuthorizer` in
favour of the unified rbac evaluator from
:mod:`threetears.agent.acl`. every tool is a namespace of type
``tool`` (emitted by :class:`threetears.agent.tools.ToolServer` on
registration); a per-call authorization decision resolves to
"evaluator.evaluate(user_id, agent_id, namespace_id, action=tool.call)"
via the same machinery workspaces + datasources use.

three-tier-task-01 phase D retired the bespoke resolver callable
alias and the parallel tool-namespace-row value object. the
authorizer now takes a ``NamespaceCollection`` handle directly and
calls :meth:`NamespaceCollection.get_by_name`; the returned entity
surfaces the four fields the evaluator reads through normal
attribute access. the Collection parameter is typed ``Any`` because
the concrete class lives in :mod:`aibots.hub.broker.namespaces` and
this package sits a layer below aibots in the dependency graph; the
three-tier-task-01 shard documents the import path as the callers'
responsibility rather than a registry-side layering edit.

wiring shape:

- one :class:`RbacEvaluatorAuthorizer` instance per registry pod
- constructed with a :class:`MembershipLoader`, a
  :class:`GrantLoader`, and a ``NamespaceCollection``
- call path: :class:`~threetears.registry.proxy.CallProxy` invokes
  :meth:`is_authorized` on every tool dispatch; the authorizer
  resolves the tool's namespace via the Collection's
  ``get_by_name`` method, then asks the evaluator for a boolean
  decision

defense in depth: when ``user_id`` is ``None`` the authorizer
returns ``False`` (a tool dispatch without an identified user is
refused even when the agent is privileged). when the namespace
lookup returns ``None`` (tool not yet materialized; registration
race) the authorizer also returns ``False``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.agent.acl import (
    AclCache,
    EvaluationContext,
    Namespace as AclNamespace,
    evaluate_decision,
)
from threetears.core.namespaces import (
    PLURAL_PREFIX_TOOL,
    build_namespace_name,
)
from threetears.observe import get_logger

__all__ = [
    "RbacEvaluatorAuthorizer",
]


log = get_logger(__name__)


TOOL_CALL_ACTION = "tool.call"


class RbacEvaluatorAuthorizer:
    """authorize tool dispatch via the unified rbac evaluator.

    implements the :class:`~threetears.registry.auth.AgentToolAuthorizer`
    protocol. on each call, looks up the tool's ``platform.namespaces``
    row via :meth:`NamespaceCollection.get_by_name` (Collection hits
    its L1 cache on hot paths), then asks
    :func:`~threetears.agent.acl.evaluate_decision` for a boolean on
    ``(user_id, agent_id, namespace, action="tool.call")``.

    platform built-in tools (``owner_agent_id=NULL``,
    ``customer_id=NULL``) require an explicit grant — the evaluator
    does not special-case them. an admin seeds one assignment on
    ``scope=all`` (or ``type_customer`` with ``namespace_type=tool``)
    binding the "default tool access" group to the caller's customer.
    this matches the shared-workspace pattern (no implicit grants on
    shared-type rows).

    :param acl_cache: shared :class:`AclCache` carrying loaders +
        ttl-bounded layers; consulted on every authorize hit so the
        bulk of decisions stay in process memory
    :ptype acl_cache: AclCache
    :param namespace_collection: three-tier ``NamespaceCollection``
        whose ``get_by_name(name)`` method resolves tool namespaces
        by canonical name (``tool:<mcp_name>:<version>``). typed
        ``Any`` because concrete Collection class lives in
        :mod:`aibots.hub.broker.namespaces`, a layer above this
        package in dependency graph; caller's wiring code
        passes real Collection instance
    :ptype namespace_collection: Any
    """

    def __init__(
        self,
        *,
        acl_cache: AclCache,
        namespace_collection: Any,
    ) -> None:
        """wire authorizer to its cache + namespace collection.

        :param acl_cache: shared :class:`AclCache`
        :ptype acl_cache: AclCache
        :param namespace_collection: three-tier ``NamespaceCollection``
            whose :meth:`get_by_name` surfaces tool namespace row
        :ptype namespace_collection: Any
        """
        self._acl_cache = acl_cache
        self._namespace_collection = namespace_collection

    async def is_authorized(
        self,
        agent_id: str,
        user_id: str | None,
        tool_name: str,
        tool_version: str,
    ) -> bool:
        """resolve an authorization decision for a tool dispatch.

        the Registry's :class:`~threetears.registry.auth.AgentToolAuthorizer`
        protocol carries the calling user identity so the unified
        rbac evaluator can resolve the user side of the
        intersection decision, and the dispatch's
        ``(tool_name, tool_version)`` tuple so this implementation
        can construct the canonical
        ``platform.namespaces.name`` shape
        (``tools.<sanitized-mcp>.<sanitized-version>``) that the
        emitter writes the row under. the proxy sources
        ``user_id`` from ``ProxyCallRequest.context.user_id`` and
        ``tool_name`` / ``tool_version`` from the request directly.

        :param agent_id: calling agent UUID in string form (border
            conversion happens here)
        :ptype agent_id: str
        :param user_id: invoking user UUID in string form, or
            ``None`` when the dispatch carries no user identity
        :ptype user_id: str | None
        :param tool_name: ``mcp_name`` from the proxy request;
            sanitized + plural-prefixed via
            :func:`build_namespace_name` before lookup
        :ptype tool_name: str
        :param tool_version: ``mcp_version`` from the proxy
            request; second segment of the canonical namespace name
        :ptype tool_version: str
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

        # canonicalize: the dispatch arrives as the natural
        # ``(mcp_name, mcp_version)`` pair; the namespace ``name``
        # column is the sanitized plural-prefix shape produced by
        # :func:`build_namespace_name`. constructing the canonical
        # form here keeps every consumer (this lookup, hub
        # access materializer, namespace emitter) agreed on the
        # ``platform.namespaces.name`` value without the call site
        # needing to reverse the sanitization rules.
        canonical_name = build_namespace_name(
            PLURAL_PREFIX_TOOL,
            tool_name,
            tool_version,
        )
        ns_entity = await self._namespace_collection.get_by_name(canonical_name)
        if ns_entity is None:
            log.info(
                "rbac authorizer: no tool namespace row, denying",
                extra={
                    "extra_data": {
                        "agent_id": agent_id,
                        "user_id": user_id,
                        "tool_name": tool_name,
                        "tool_version": tool_version,
                        "canonical_name": canonical_name,
                    }
                },
            )
            return result

        evaluator_namespace = AclNamespace(
            id=ns_entity.id,
            customer_id=ns_entity.customer_id,
            namespace_type=ns_entity.namespace_type,
            owner_agent_id=ns_entity.owner_agent_id,
        )
        ctx = EvaluationContext(
            namespace=evaluator_namespace,
            action=TOOL_CALL_ACTION,
            user_id=user_uuid,
            agent_id=agent_uuid,
        )
        result = await evaluate_decision(ctx, cache=self._acl_cache)
        return result
