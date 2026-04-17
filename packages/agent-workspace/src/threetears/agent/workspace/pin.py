"""conversation-scoped workspace pin stored as a ContextItem.

records "which workspace is currently pinned for this conversation" by
writing a context item with ``context_type="workspace_pin"`` and
``key="current"`` through the public
:class:`threetears.agent.tools.context.ToolContextManager` API
(:meth:`get_item_by_type_and_key`, :meth:`save_item_by_type_and_key`,
:meth:`delete_item_by_type_and_key`). reuses the three-tier (L1 SQLite
-> L2 NATS KV -> L3 PostgreSQL) storage and cross-pod cache-coherency
already provided by the underlying :class:`ContextItemCollection`.

this module ships only the storage adapter: no tool definitions, no new
entity, no workspace-existence checks (callers verify the workspace
before pinning).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from threetears.agent.tools.context import ToolContextManager


_CONTEXT_TYPE = "workspace_pin"
_PIN_KEY = "current"


@dataclass
class PinnedWorkspace:
    """snapshot of the workspace currently pinned to a conversation.

    :param workspace_id: identifier of pinned workspace
    :ptype workspace_id: UUID
    :param workspace_name: display name captured at pin time
    :ptype workspace_name: str
    :param date_pinned: timezone-aware UTC instant the pin was written
    :ptype date_pinned: datetime
    :param pinned_by_actor_id: identifier of actor that pinned workspace
    :ptype pinned_by_actor_id: UUID
    """

    workspace_id: UUID
    workspace_name: str
    date_pinned: datetime
    pinned_by_actor_id: UUID


async def set_pin(
    context: ToolContextManager,
    workspace_id: UUID,
    workspace_name: str,
    pinned_by_actor_id: UUID,
) -> None:
    """pin ``workspace_id`` to the current conversation.

    overwrites any existing pin (only one pin per conversation). UUIDs
    are converted to ``str`` and the timestamp to ISO-8601 at the
    storage border per CLAUDE.md. caller is responsible for verifying
    that ``workspace_id`` refers to an extant workspace.

    :param context: conversation-scoped context manager
    :ptype context: ToolContextManager
    :param workspace_id: identifier of workspace to pin
    :ptype workspace_id: UUID
    :param workspace_name: display name to capture at pin time
    :ptype workspace_name: str
    :param pinned_by_actor_id: identifier of actor performing pin
    :ptype pinned_by_actor_id: UUID
    :return: None
    :rtype: None
    """
    date_pinned = datetime.now(UTC)
    metadata = {
        "workspace_name": workspace_name,
        "date_pinned": date_pinned.isoformat(),
        "pinned_by_actor_id": str(pinned_by_actor_id),
    }
    await context.save_item_by_type_and_key(
        context_type=_CONTEXT_TYPE,
        key=_PIN_KEY,
        content=str(workspace_id),
        short_desc=workspace_name[:200],
        long_desc="",
        metadata=metadata,
    )


async def get_pin(context: ToolContextManager) -> PinnedWorkspace | None:
    """return the currently pinned workspace for this conversation, or ``None``.

    parses stored border representations back into typed Python objects:
    UUID strings rehydrate to :class:`UUID`, ISO timestamps rehydrate to
    timezone-aware UTC :class:`datetime` (naive timestamps are attached
    to UTC defensively).

    :param context: conversation-scoped context manager
    :ptype context: ToolContextManager
    :return: pin snapshot or None if unset
    :rtype: PinnedWorkspace | None
    """
    result: PinnedWorkspace | None = None
    item = await context.get_item_by_type_and_key(_CONTEXT_TYPE, _PIN_KEY)
    if item is not None:
        metadata = item.get("metadata") or {}
        date_pinned = datetime.fromisoformat(metadata["date_pinned"])
        if date_pinned.tzinfo is None:
            date_pinned = date_pinned.replace(tzinfo=UTC)
        result = PinnedWorkspace(
            workspace_id=UUID(item["content"]),
            workspace_name=metadata["workspace_name"],
            date_pinned=date_pinned,
            pinned_by_actor_id=UUID(metadata["pinned_by_actor_id"]),
        )
    return result


async def clear_pin(context: ToolContextManager) -> None:
    """remove the pin for this conversation. idempotent no-op if absent.

    :param context: conversation-scoped context manager
    :ptype context: ToolContextManager
    :return: None
    :rtype: None
    """
    await context.delete_item_by_type_and_key(_CONTEXT_TYPE, _PIN_KEY)
