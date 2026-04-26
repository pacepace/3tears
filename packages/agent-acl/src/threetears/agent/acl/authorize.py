"""canonical rbac authorization primitive shared across every 3tears app.

every resource-typed authorize helper (memory, datasource, channel,
customer, audit, api_key, model, conversation, workspace,
workspace_file, shared_agent, ...) collapses to a 3-line wrapper
around :func:`authorize` that:

1. resolves resource identity to a canonical namespace name
2. picks the action vocabulary specific to the resource
3. catches :class:`AccessDenied` and re-raises a typed
   resource-specific subclass

the primitive itself is resource-agnostic: it takes a
:class:`NamespaceCollection` handle, a namespace name, an action
string, the calling user + agent ids, and a shared :class:`AclCache`.
it looks up the namespace, builds an :class:`EvaluationContext`,
calls :func:`evaluate_decision` (which serves from the cache's
membership and per-namespace layers), and either returns the
:class:`EvaluationResult` or raises :class:`AccessDenied` on a deny.

generalization rationale: per the 3tears platform vision, RBAC is a
cross-cutting concern the SDK owns. one canonical path keeps every
consumer's behavior identical — same cache layers, same denial shape,
same trace span — so a fix landed here propagates without per-app
audit. resource-specific helpers exist only to pin (a) the action
vocabulary and (b) the typed exception class their callers catch
on; they do not re-implement the lookup or the evaluator call.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.agent.acl.cache import AclCache
from threetears.agent.acl.evaluator import evaluate_with_trail
from threetears.agent.acl.types import (
    EvaluationContext,
    EvaluationResult,
    Namespace as AclNamespace,
)
from threetears.observe import get_logger, traced

__all__ = [
    "AccessDenied",
    "NamespaceNotFound",
    "authorize",
    "authorize_with_trail",
]

log = get_logger(__name__)


class AccessDenied(Exception):
    """raised when the unified evaluator denies an access request.

    carries the action, namespace name, and caller identity so a
    resource-specific wrapper can preserve the contextual fields when
    re-raising as a typed subclass. callers that catch this generic
    base catch every per-resource denial transparently; callers that
    need to dispatch on resource type catch the typed subclass.

    :ivar action: action string evaluated (e.g. ``"memory.read"``)
    :ivar namespace_name: canonical name of namespace evaluated against
    :ivar user_id: invoking user UUID, or ``None`` for agent-only
        evaluations
    :ivar agent_id: invoking agent UUID, or ``None`` for user-only
        evaluations
    :ivar reason: short classification string for log / audit fan-out
    """

    def __init__(
        self,
        message: str,
        *,
        action: str | None = None,
        namespace_name: str | None = None,
        user_id: UUID | None = None,
        agent_id: UUID | None = None,
        reason: str | None = None,
    ) -> None:
        """initialize the denial exception.

        :param message: human-readable denial message
        :ptype message: str
        :param action: action string evaluated
        :ptype action: str | None
        :param namespace_name: namespace name evaluated against
        :ptype namespace_name: str | None
        :param user_id: invoking user UUID
        :ptype user_id: UUID | None
        :param agent_id: invoking agent UUID
        :ptype agent_id: UUID | None
        :param reason: short classification string
        :ptype reason: str | None
        """
        super().__init__(message)
        self.action = action
        self.namespace_name = namespace_name
        self.user_id = user_id
        self.agent_id = agent_id
        self.reason = reason


class NamespaceNotFound(AccessDenied):
    """raised when the authorize primitive cannot resolve namespace by name.

    distinct subclass of :class:`AccessDenied` so resource-specific
    wrappers can surface "namespace row missing" as a wiring-gap
    diagnostic separately from a "user lacks grant" denial. the
    typed subclass keeps callers that catch :class:`AccessDenied`
    backwards-compatible: every namespace-not-found is still an
    access denial.
    """


@traced
async def authorize(
    *,
    namespace_collection: Any,
    namespace_name: str,
    action: str,
    user_id: UUID | None,
    agent_id: UUID | None,
    cache: AclCache,
) -> EvaluationResult:
    """canonical rbac authorization primitive.

    looks up namespace by name via ``namespace_collection.get_by_name``,
    builds an :class:`EvaluationContext`, and runs the unified evaluator
    via :func:`evaluate_decision` against the supplied :class:`AclCache`.
    raises :class:`NamespaceNotFound` when the namespace row is absent
    and :class:`AccessDenied` when the evaluator denies; returns the
    full :class:`EvaluationResult` on allow so callers that need the
    namespace entity, contributing trails, or the effective action set
    do not pay for a second lookup.

    :param namespace_collection: a Collection exposing
        ``async def get_by_name(name: str) -> entity | None``;
        typed ``Any`` because concrete Collection class lives in
        consumer apps' layers (hub, agent pod) above this package
    :ptype namespace_collection: Any
    :param namespace_name: canonical namespace name to evaluate
        against (e.g. ``"datasource:my_warehouse"``,
        ``"memories.<agent_id_hex>.<customer_id_hex>"``)
    :ptype namespace_name: str
    :param action: canonical action string (e.g. ``"memory.read"``,
        ``"datasource.write"``)
    :ptype action: str
    :param user_id: invoking user UUID, or ``None`` for agent-only
        evaluation
    :ptype user_id: UUID | None
    :param agent_id: invoking agent UUID, or ``None`` for user-only
        evaluation
    :ptype agent_id: UUID | None
    :param cache: shared :class:`AclCache` carrying loaders + ttl
        layers
    :ptype cache: AclCache
    :return: full evaluation result on allow (carries effective
        actions, contributing trails, and the resolved namespace
        entity via ``result.context.namespace`` is not surfaced;
        callers that need the entity pass it back in via separate
        lookup or use :func:`authorize_with_trail` and the included
        result fields)
    :rtype: EvaluationResult
    :raises NamespaceNotFound: when ``namespace_collection.get_by_name``
        returns None for ``namespace_name``
    :raises AccessDenied: when the evaluator denies the action
    """
    ns_entity = await namespace_collection.get_by_name(namespace_name)
    if ns_entity is None:
        log.warning(
            "authorize: namespace row missing",
            extra={
                "extra_data": {
                    "action": action,
                    "namespace_name": namespace_name,
                    "user_id": str(user_id) if user_id else None,
                    "agent_id": str(agent_id) if agent_id else None,
                },
            },
        )
        raise NamespaceNotFound(
            f"access denied: namespace {namespace_name} not found",
            action=action,
            namespace_name=namespace_name,
            user_id=user_id,
            agent_id=agent_id,
            reason="namespace_not_found",
        )

    acl_namespace = AclNamespace(
        id=ns_entity.id,
        customer_id=ns_entity.customer_id,
        namespace_type=ns_entity.namespace_type,
        owner_agent_id=ns_entity.owner_agent_id,
    )
    eval_ctx = EvaluationContext(
        namespace=acl_namespace,
        action=action,
        user_id=user_id,
        agent_id=agent_id,
    )
    result = await evaluate_with_trail(eval_ctx, cache=cache)
    if not result.decision:
        log.info(
            "authorize: denied",
            extra={
                "extra_data": {
                    "action": action,
                    "namespace_name": namespace_name,
                    "namespace_id": str(ns_entity.id),
                    "user_id": str(user_id) if user_id else None,
                    "agent_id": str(agent_id) if agent_id else None,
                },
            },
        )
        raise AccessDenied(
            f"access denied: {action} on namespace {namespace_name}",
            action=action,
            namespace_name=namespace_name,
            user_id=user_id,
            agent_id=agent_id,
            reason="evaluator_deny",
        )
    return result


@traced
async def authorize_with_trail(
    *,
    namespace_collection: Any,
    namespace_name: str,
    action: str,
    user_id: UUID | None,
    agent_id: UUID | None,
    cache: AclCache,
) -> tuple[EvaluationResult, Any]:
    """authorize variant that also returns resolved namespace entity.

    several resource wrappers (datasource, customer, memory) need the
    entity itself for downstream audit envelopes or assignment-ensure
    paths. this variant performs the same lookup + evaluator call as
    :func:`authorize` and returns ``(result, ns_entity)`` so callers
    do not pay for a second namespace lookup.

    :param namespace_collection: a Collection exposing
        ``async def get_by_name(name: str) -> entity | None``
    :ptype namespace_collection: Any
    :param namespace_name: canonical namespace name to evaluate against
    :ptype namespace_name: str
    :param action: canonical action string
    :ptype action: str
    :param user_id: invoking user UUID, or ``None``
    :ptype user_id: UUID | None
    :param agent_id: invoking agent UUID, or ``None``
    :ptype agent_id: UUID | None
    :param cache: shared :class:`AclCache`
    :ptype cache: AclCache
    :return: ``(result, ns_entity)`` pair
    :rtype: tuple[EvaluationResult, Any]
    :raises NamespaceNotFound: when ``namespace_collection.get_by_name``
        returns None for ``namespace_name``
    :raises AccessDenied: when the evaluator denies the action
    """
    ns_entity = await namespace_collection.get_by_name(namespace_name)
    if ns_entity is None:
        log.warning(
            "authorize_with_trail: namespace row missing",
            extra={
                "extra_data": {
                    "action": action,
                    "namespace_name": namespace_name,
                    "user_id": str(user_id) if user_id else None,
                    "agent_id": str(agent_id) if agent_id else None,
                },
            },
        )
        raise NamespaceNotFound(
            f"access denied: namespace {namespace_name} not found",
            action=action,
            namespace_name=namespace_name,
            user_id=user_id,
            agent_id=agent_id,
            reason="namespace_not_found",
        )

    acl_namespace = AclNamespace(
        id=ns_entity.id,
        customer_id=ns_entity.customer_id,
        namespace_type=ns_entity.namespace_type,
        owner_agent_id=ns_entity.owner_agent_id,
    )
    eval_ctx = EvaluationContext(
        namespace=acl_namespace,
        action=action,
        user_id=user_id,
        agent_id=agent_id,
    )
    result = await evaluate_with_trail(eval_ctx, cache=cache)
    if not result.decision:
        log.info(
            "authorize_with_trail: denied",
            extra={
                "extra_data": {
                    "action": action,
                    "namespace_name": namespace_name,
                    "namespace_id": str(ns_entity.id),
                    "user_id": str(user_id) if user_id else None,
                    "agent_id": str(agent_id) if agent_id else None,
                },
            },
        )
        raise AccessDenied(
            f"access denied: {action} on namespace {namespace_name}",
            action=action,
            namespace_name=namespace_name,
            user_id=user_id,
            agent_id=agent_id,
            reason="evaluator_deny",
        )
    return result, ns_entity


# evaluate_decision is intentionally not re-exported here; callers
# that want the bool-only fast path import from the evaluator module
# directly. the canonical user-facing surface for application code
# is :func:`authorize` / :func:`authorize_with_trail`.
