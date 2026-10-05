"""Changing what an agent remembers without destroying anything it remembered.

Three moves, used by every writer that changes a stored memory -- dream's
consolidation, post-turn extraction, ``memory_add``'s near-duplicate:

- :func:`supersede` writes the new memory, links each memory it replaces to it
  in ``memory_consolidations`` with the reason, and marks those superseded. The
  old ones drop out of ambient recall and stay recallable by id.
- :func:`retract` takes a memory out of recall without a replacement: it is
  tagged retracted, with why, which every search, dedup and dream skips, and
  its salience goes to the bottom. It stays recallable by id.
- A permanent (``evergreen``) memory is touched by neither: :func:`is_permanent`
  is what each writer asks first, and a change to one is written as a new
  memory beside it.

Nothing here deletes a row.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from uuid_utils import uuid7

from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.agent.memory.collections import MemoriesCollection, MemoryConsolidationsCollection
    from threetears.agent.memory.entities import MemoryEntity

__all__ = ["RETRACTED_TAG", "is_permanent", "retract", "supersede"]

log = get_logger(__name__)

#: The tag a retracted memory carries; searches, dedup and dream skip it.
RETRACTED_TAG = "retracted"
#: The tag that carries why, after the colon.
_BECAUSE = "retracted_because:"


def is_permanent(memory: MemoryEntity | dict[str, Any] | None) -> bool:
    """Whether a memory is pinned permanent: never decayed, merged, superseded or retracted.

    :param memory: the memory, as an entity or a row
    :ptype memory: MemoryEntity | dict[str, Any] | None
    :return: True for an evergreen memory
    :rtype: bool
    """
    if memory is None:
        return False
    if isinstance(memory, dict):
        return bool(memory.get("evergreen"))
    return bool(getattr(memory, "evergreen", False))


async def supersede(
    memories: MemoriesCollection,
    consolidations: MemoryConsolidationsCollection,
    *,
    agent_id: UUID,
    source_ids: list[UUID],
    fields: dict[str, Any],
    rationale: str | None,
) -> UUID:
    """Write the memory that replaces ``source_ids``, link each to it, and mark them superseded.

    :param memories: the memories collection
    :ptype memories: MemoriesCollection
    :param consolidations: the provenance edges collection
    :ptype consolidations: MemoryConsolidationsCollection
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param source_ids: the memories it replaces; none of them permanent
    :ptype source_ids: list[UUID]
    :param fields: the new memory's columns other than its id, agent and dates:
        ``customer_id``, ``user_id``, ``conversation_id``, ``type_memory``,
        ``content``, ``embedding``, ``salience`` and, when set, ``evergreen``
    :ptype fields: dict[str, Any]
    :param rationale: why, kept on each edge
    :ptype rationale: str | None
    :return: the new memory's id
    :rtype: UUID
    :raises ValueError: when asked to replace a permanent memory
    """
    for source_id in source_ids:
        if is_permanent(await memories.get((agent_id, source_id))):
            raise ValueError(f"memory {source_id} is permanent and is never superseded")
    new_id = UUID(str(uuid7()))
    # the new id is fresh, so nothing reaches it yet; the guard also proves no
    # source descends from an earlier memory in the chain
    await consolidations.assert_no_cycle(agent_id, consolidated_memory_id=new_id, source_memory_ids=source_ids)
    now = datetime.now(UTC)
    entity = memories.create(
        {**fields, "memory_id": new_id, "agent_id": agent_id, "date_created": now, "date_updated": now}
    )
    await memories.save_entity(entity)
    # the edges after the new memory commits, so their keys have a target
    for source_id in source_ids:
        edge = consolidations.create(
            {
                "agent_id": agent_id,
                "consolidated_memory_id": new_id,
                "source_memory_id": source_id,
                "rationale": rationale,
                "date_created": now,
                "date_updated": now,
            }
        )
        await consolidations.save_entity(edge)
    await memories.mark_superseded(agent_id, source_memory_ids=source_ids, gist_id=new_id)
    return new_id


async def retract(memories: MemoriesCollection, *, agent_id: UUID, memory_id: UUID, reason: str) -> bool:
    """Take a memory out of ambient recall, keeping it, and say why.

    :param memories: the memories collection
    :ptype memories: MemoriesCollection
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param memory_id: the memory
    :ptype memory_id: UUID
    :param reason: why it no longer holds, kept as its tag
    :ptype reason: str
    :return: False when it is permanent or gone, and nothing changed
    :rtype: bool
    """
    entity = await memories.get((agent_id, memory_id))
    if entity is None or is_permanent(entity):
        return False
    tags = [t for t in (entity.tags or []) if t != RETRACTED_TAG and not t.startswith(_BECAUSE)]
    entity.tags = [*tags, RETRACTED_TAG, f"{_BECAUSE}{reason[:200]}"]
    await memories.save_entity(entity)
    # salience is not written by an entity save; the raw pass is its one writer
    await memories.set_salience(agent_id, memory_ids=[memory_id], salience=0.0)
    return True
