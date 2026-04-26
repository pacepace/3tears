"""ACL-integrated cross-agent memory retrieval service.

collections-task-04. canonical pattern for cross-partition retrieval:
the Collection layer knows about partitions; the service layer knows
about authorization. :class:`MemoryAccessService` composes them:

1. enumerate candidate ``memory`` namespaces under a customer.
2. evaluate ``memory.read`` for the caller against each namespace via
   :func:`threetears.agent.acl.evaluate_decision` (the unified evaluator).
3. extract the authorized ``owner_agent_id`` from each surviving
   namespace into a tuple.
4. fan out a single SQL via
   :meth:`MemoriesCollection.find_for_user_in_agents` (decorated
   ``@spans_partitions``) with the resolved tuple.

the Collection method NEVER evaluates ACL — it accepts a resolved
tuple and trusts the caller. the service layer NEVER hand-rolls SQL —
it composes the Collection's authorized-set-only contract. the two
layers compose cleanly: Collection is testable in isolation against a
mock pool; service is testable in isolation against mock evaluator +
mock Collection.

future cross-partition retrieval surfaces (cross-agent workspaces,
cross-agent conversations) follow the same shape. ``partition-column-
pattern.md`` documents the principle.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.agent.acl import (
    EvaluationContext,
    Namespace as AclNamespace,
    evaluate_decision,
)
from threetears.agent.memory.authorize import (
    ACTION_MEMORY_READ,
    MEMORY_NAMESPACE_TYPE,
)
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.entities import MemoryEntity
from threetears.observe import get_logger

__all__ = [
    "MemoryAccessService",
]

log = get_logger(__name__)


class MemoryAccessService:
    """service-layer composer for cross-agent memory retrieval.

    constructed with the unified RBAC cache, a
    :class:`NamespaceCollection`, and a :class:`MemoriesCollection`.
    callers invoke
    :meth:`find_for_user_across_authorized_agents` to retrieve a
    user's memories across every agent partition the caller has been
    granted ``memory.read`` on.

    the service's only responsibility is composing authorization with
    the Collection's ``@spans_partitions`` fan-out method. it carries
    no state of its own and is safe to construct per-request.
    """

    __slots__ = (
        "acl_cache",
        "namespace_collection",
        "memories_collection",
    )

    def __init__(
        self,
        *,
        acl_cache: Any,
        namespace_collection: Any,
        memories_collection: MemoriesCollection,
    ) -> None:
        """initialize with the three dependencies the composer threads.

        :param acl_cache: shared RBAC cache exposing ``membership_loader``
            and ``grant_loader`` attributes (an
            :class:`threetears.agent.acl.AclCache` instance in
            production; a permissive fixture in tests). the loaders
            are passed straight to :func:`evaluate_decision`
        :ptype acl_cache: Any
        :param namespace_collection: three-tier ``NamespaceCollection``
            exposing :meth:`find_by_type_and_customer`. typed ``Any``
            because the Collection lives in
            :mod:`aibots.hub.broker.namespaces` (a higher layer than
            ``agent-memory``); production wiring constructs the bundle
            with the concrete instance, this module only uses the
            documented method surface
        :ptype namespace_collection: Any
        :param memories_collection: three-tier ``MemoriesCollection``
            exposing the ``@spans_partitions``-decorated
            :meth:`MemoriesCollection.find_for_user_in_agents` method
        :ptype memories_collection: MemoriesCollection
        """
        self.acl_cache = acl_cache
        self.namespace_collection = namespace_collection
        self.memories_collection = memories_collection

    async def find_for_user_across_authorized_agents(
        self,
        *,
        user_id: UUID,
        caller_user_id: UUID,
        customer_id: UUID,
    ) -> list[MemoryEntity]:
        """find ``user_id``'s memories across every agent the caller can read.

        composition shape:

        1. enumerate every ``memory`` namespace under ``customer_id``
           via :meth:`NamespaceCollection.find_by_type_and_customer`.
        2. for each candidate namespace, evaluate ``memory.read`` for
           ``caller_user_id`` via :func:`evaluate_decision` against the
           unified ACL loaders. authorized namespaces contribute their
           ``owner_agent_id`` to the resolved set; denied namespaces
           are silently skipped (a denial is not an error here, it's a
           normal scope contraction).
        3. when no agents resolve, return ``[]`` early — the
           ``@spans_partitions`` Collection method refuses an empty
           tuple and that refusal is correct, but the service layer
           short-circuits at the same boundary so the noise stays out
           of the call stack.
        4. otherwise call
           :meth:`MemoriesCollection.find_for_user_in_agents` with the
           resolved ``tuple[UUID, ...]`` and return the merged
           projection.

        :param user_id: user whose memories to surface (row filter
            applied uniformly within each authorized partition)
        :ptype user_id: UUID
        :param caller_user_id: invoking user UUID — the ``memory.read``
            evaluator decides authorization on this identity
        :ptype caller_user_id: UUID
        :param customer_id: customer UUID; cross-customer namespaces
            are filtered out at the namespace enumeration step
        :ptype customer_id: UUID
        :return: list of memory entities across every authorized
            agent partition, ordered by ``date_created`` DESC. empty
            list when the caller has no ``memory.read`` grants in the
            customer
        :rtype: list[MemoryEntity]
        """
        candidate = await self.namespace_collection.find_by_type_and_customer(
            namespace_type=MEMORY_NAMESPACE_TYPE,
            customer_id=customer_id,
        )

        authorized: list[UUID] = []
        for ns in candidate:
            owner_agent_id = getattr(ns, "owner_agent_id", None)
            if owner_agent_id is None:
                # a memory namespace must own an agent to participate
                # in cross-agent retrieval; namespaces with no owner
                # cannot contribute (they would surface every user's
                # memories indiscriminately). skip silently.
                continue
            ctx = EvaluationContext(
                namespace=AclNamespace(
                    id=ns.id,
                    customer_id=ns.customer_id,
                    namespace_type=ns.namespace_type,
                    owner_agent_id=owner_agent_id,
                ),
                action=ACTION_MEMORY_READ,
                user_id=caller_user_id,
            )
            allowed = await evaluate_decision(ctx, cache=self.acl_cache)
            if allowed:
                authorized.append(owner_agent_id)

        if not authorized:
            log.debug(
                "memory cross-agent retrieval: no authorized agents",
                extra={
                    "extra_data": {
                        "caller_user_id": str(caller_user_id),
                        "customer_id": str(customer_id),
                        "candidate_count": len(candidate),
                    },
                },
            )
            return []

        log.debug(
            "memory cross-agent retrieval: fan-out to authorized agents",
            extra={
                "extra_data": {
                    "caller_user_id": str(caller_user_id),
                    "customer_id": str(customer_id),
                    "authorized_count": len(authorized),
                },
            },
        )
        return await self.memories_collection.find_for_user_in_agents(
            user_id=user_id,
            agent_ids=tuple(authorized),
            customer_id=customer_id,
        )
