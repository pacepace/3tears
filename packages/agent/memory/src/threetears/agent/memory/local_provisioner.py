"""Materialize a memory namespace where there is no hub to ask.

The authorizer asks its ``namespace_provisioner`` for a memory namespace it cannot find. A hub deployment
answers through :class:`~threetears.agent.memory.HubMemoryNamespaceProvisioner`: agents never write the
control plane, the hub does. A deployment with NO hub -- one application that owns its own control plane
and its own ``namespaces`` table -- had nothing to plug in, so it wrote the row with raw SQL, and got it
subtly wrong (``owner_namespace`` NULL, so no agent owned it; ``schema_name`` NULL).

:class:`LocalMemoryNamespaceProvisioner` is that deployment's provisioner. It writes exactly the row the hub
writes -- every field from the same public helpers -- through ``NamespaceCollection.ensure_namespace``, the
one write the hub is moving onto too. Like the hub it resolves by (type, owner agent, customer) first, reads
its row back, and refuses a row that is not the pair it was asked for.

**Whose customer it trusts.** The hub checks the requested customer against the caller's VERIFIED identity;
there is no forwarded identity here. The provisioner trusts the ``(agent_id, customer_id)`` it is handed --
which is right only because, in a hub-less deployment, the caller is the application that owns the control
plane and already decided which customer the request is for. Never construct this in a process that serves
untrusted agents: that is what the hub is for.

``test_no_namespace_writes.py`` exempts this module, and only for ``ensure_namespace``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from threetears.agent.memory.authorize import (
    MEMORY_NAMESPACE_TYPE,
    memory_namespace_id,
    memory_namespace_name,
    memory_namespace_schema_name,
)
from threetears.agent.memory.namespace_client import MemoryNamespaceRef, MemoryNamespaceUnavailableError
from threetears.core.namespaces import build_agent_namespace_name
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.agent.acl.entities import NamespaceEntity

__all__ = ["LocalMemoryNamespaceProvisioner"]

log = get_logger(__name__)


class _Namespaces(Protocol):
    """the two ``NamespaceCollection`` calls the provisioner makes."""

    async def get_by_owner_and_customer(
        self, *, namespace_type: str, owner_agent_id: UUID | None, customer_id: UUID | None
    ) -> NamespaceEntity | None:
        """the row of this type for the owner and customer, or ``None``.

        :param namespace_type: the type discriminator
        :ptype namespace_type: str
        :param owner_agent_id: the owning agent
        :ptype owner_agent_id: UUID | None
        :param customer_id: the customer
        :ptype customer_id: UUID | None
        :return: the row
        :rtype: NamespaceEntity | None
        """
        ...

    async def ensure_namespace(self, **fields: object) -> NamespaceEntity:
        """get-or-create the row these fields describe.

        :param fields: the ``NamespaceCollection.ensure_namespace`` keywords
        :ptype fields: object
        :return: the row
        :rtype: NamespaceEntity
        """
        ...


class LocalMemoryNamespaceProvisioner:
    """the :class:`~threetears.agent.memory.MemoryNamespaceProvisioner` for a deployment with no hub.

    plug it into ``MemoryAuthorizerDependencies(namespace_provisioner=...)``.

    :param namespace_collection: the deployment's ``NamespaceCollection``
    :ptype namespace_collection: NamespaceCollection
    """

    def __init__(self, namespace_collection: _Namespaces) -> None:
        self._namespaces = namespace_collection

    async def ensure(self, *, agent_id: UUID, customer_id: UUID) -> MemoryNamespaceRef:
        """the memory namespace for ``(agent_id, customer_id)``, created if absent.

        :param agent_id: the memory's owning agent
        :ptype agent_id: UUID
        :param customer_id: the customer the memory belongs to (trusted -- see the module docstring)
        :ptype customer_id: UUID
        :return: the resolved namespace
        :rtype: MemoryNamespaceRef
        :raises MemoryNamespaceUnavailableError: the row could not be written, read back, or is not
            this pair's -- the authorizer maps it to a denial
        """
        row = await self._resolve(agent_id, customer_id)
        if row is None:
            try:
                await self._namespaces.ensure_namespace(
                    namespace_id=memory_namespace_id(agent_id, customer_id),
                    name=memory_namespace_name(agent_id, customer_id),
                    namespace_type=MEMORY_NAMESPACE_TYPE,
                    owner_agent_id=agent_id,
                    customer_id=customer_id,
                    owner_namespace=build_agent_namespace_name(agent_id),
                    schema_name=memory_namespace_schema_name(agent_id, customer_id),
                    metadata={},
                )
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- any write failure is "unavailable", which the authorizer turns into a denial rather than a crash
                raise MemoryNamespaceUnavailableError(f"could not ensure the memory namespace: {exc}") from exc
            row = await self._resolve(agent_id, customer_id)
            if row is None:
                raise MemoryNamespaceUnavailableError("the memory namespace was written but cannot be read back")
            log.info(
                "memory namespace provisioned locally",
                extra={"extra_data": {"agent_id": str(agent_id), "customer_id": str(customer_id)}},
            )
        return _as_ref(row, agent_id=agent_id, customer_id=customer_id)

    async def _resolve(self, agent_id: UUID, customer_id: UUID) -> NamespaceEntity | None:
        """the existing row for the pair, looked up the way the hub looks it up.

        :param agent_id: the owning agent
        :ptype agent_id: UUID
        :param customer_id: the customer
        :ptype customer_id: UUID
        :return: the row, or ``None``
        :rtype: NamespaceEntity | None
        """
        return await self._namespaces.get_by_owner_and_customer(
            namespace_type=MEMORY_NAMESPACE_TYPE, owner_agent_id=agent_id, customer_id=customer_id
        )


def _as_ref(row: NamespaceEntity, *, agent_id: UUID, customer_id: UUID) -> MemoryNamespaceRef:
    """the evaluator's view of ``row``, refused unless it is the pair that was asked for.

    :param row: the resolved row
    :ptype row: NamespaceEntity
    :param agent_id: the agent asked for
    :ptype agent_id: UUID
    :param customer_id: the customer asked for
    :ptype customer_id: UUID
    :return: the ref
    :rtype: MemoryNamespaceRef
    :raises MemoryNamespaceUnavailableError: the row belongs to another pair
    """
    actual = (row.namespace_type, row.owner_agent_id, row.customer_id)
    if actual != (MEMORY_NAMESPACE_TYPE, agent_id, customer_id):
        raise MemoryNamespaceUnavailableError(f"resolved namespace {row.id} is {actual}, not the pair asked for")
    return MemoryNamespaceRef(
        id=row.id,
        customer_id=customer_id,
        owner_agent_id=agent_id,
        namespace_type=MEMORY_NAMESPACE_TYPE,
        owner_namespace=row.owner_namespace,
    )
