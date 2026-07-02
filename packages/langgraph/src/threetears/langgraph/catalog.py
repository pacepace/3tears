"""Seam for cataloging a produced object into the agent's object catalog.

When a producing tool streams a large artifact out-of-band (Path-2) and returns
an :class:`~threetears.media.contracts.ObjectHandle` in its result metadata, the
agent must persist a catalog record so the object is discoverable and is not an
orphan. This module defines the :class:`ObjectCataloger` Protocol the graph's
``tool_node`` calls; the host app (the SDK) supplies the concrete writer (which
commits the object to the hub-owned ``objects`` catalog over NATS).

This Protocol is deliberately table-agnostic: it carries only the handle + the
verified per-call identity, so the host is free to back it with any store (the
platform's hub-owned ``objects`` table; a test double; a future backend).

Mirrors :class:`~threetears.langgraph.offload.ToolResultOffloader` (the offload
seam): an optional per-call integration injected via ``config["configurable"]``,
invoked by the tool node after a tool returns. A catalog failure is logged and
tolerated by the caller -- the object stays in the store but uncataloged, so the
hub-side reconciler garbage-collects it past its grace window; a failed catalog
must never break the tool result.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from threetears.media.contracts import ObjectHandle

__all__ = ["ObjectCataloger"]


@runtime_checkable
class ObjectCataloger(Protocol):
    """Persists a produced :class:`ObjectHandle` as a catalog record.

    The concrete implementation (host-side) commits the object to the catalog
    under the verified per-call identity, so the record is stamped with the right
    tenant + conversation.
    """

    async def catalog(
        self,
        handle: ObjectHandle,
        *,
        conversation_id: UUID,
        customer_id: UUID,
        user_id: UUID | None,
    ) -> None:
        """Persist ``handle`` as a catalog record for the current agent + tenant.

        :param handle: the produced object's handle (object_id, s3_key,
            mime_type, size_bytes, summary, category)
        :ptype handle: ObjectHandle
        :param conversation_id: the verified conversation the object was
            produced in
        :ptype conversation_id: UUID
        :param customer_id: the verified owning customer (tenant)
        :ptype customer_id: UUID
        :param user_id: the verified invoking user, or ``None`` for an
            agent-initiated turn
        :ptype user_id: UUID | None
        :return: nothing
        :rtype: None
        """
        ...
