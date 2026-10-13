"""Changing what an agent remembers without destroying anything it remembered.

The moves every writer that changes a stored memory uses -- dream's
consolidation, post-turn extraction, ``memory_add``'s near-duplicate, and the
tools an agent corrects and retires its own memories with:

- :func:`supersede` writes the new memory, links each memory it replaces to it
  in ``memory_consolidations`` with the reason, and marks those superseded. The
  old ones drop out of ambient recall and stay recallable by id.
- :func:`correct_beside` writes the new memory and links it to the ones it
  corrects without replacing them: how a correction to a permanent memory is
  kept next to it.
- :func:`write_once` writes a new memory and, when a memory from the same
  turn or the one before already says nearly the same thing, replaces that
  one with it: one event is stored once, whichever writers wrote it.
- :func:`retract` takes a memory out of recall without a replacement: it is
  tagged retracted, with why, which every search, dedup and dream skips, and
  its salience goes to the bottom. It stays recallable by id, and
  :func:`restore` undoes it.
- A permanent (``evergreen``) memory is touched by neither replacement nor
  retraction unless the caller says the person asked (``allow_permanent``):
  :func:`is_permanent` is what each writer asks first, and otherwise a change
  to one is written as a new memory beside it.

:func:`marks_of` reads a memory row's state the way a line shown to an agent
says it: permanent, retired and why, replaced and by what, corrected by what.

Nothing here deletes a row.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from uuid_utils import uuid7

from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.agent.memory.collections import MemoriesCollection, MemoryConsolidationsCollection
    from threetears.agent.memory.entities import MemoryEntity

__all__ = [
    "RETRACTED_BECAUSE_PREFIX",
    "RETRACTED_SALIENCE_PREFIX",
    "RETRACTED_TAG",
    "SAME_EVENT_RATIONALE",
    "MemoryMarks",
    "Written",
    "correct_beside",
    "is_permanent",
    "marks_of",
    "restore",
    "retract",
    "supersede",
    "write_once",
]

log = get_logger(__name__)

#: The tag a retracted memory carries; searches, dedup and dream skip it.
RETRACTED_TAG = "retracted"
#: The tag that carries why, after the colon.
RETRACTED_BECAUSE_PREFIX = "retracted_because:"
#: The tag that carries what the memory's salience was, so a restore puts it back.
RETRACTED_SALIENCE_PREFIX = "retracted_salience:"
_BECAUSE = RETRACTED_BECAUSE_PREFIX
#: Why a memory was replaced by :func:`write_once`, kept on the edge.
SAME_EVENT_RATIONALE = "the same event, written again in the same turn or the next"


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


@dataclass(frozen=True)
class MemoryMarks:
    """What a memory's line says of it besides its text.

    :ivar permanent: pinned; never decayed, merged, replaced or retired by an agent alone
    :ivar retired: retracted; out of search and of ambient recall
    :ivar retired_because: the reason kept with the retraction, when one was
    :ivar replaced_by: the memory that stands in for it, when that memory exists
    :ivar replaced_at: when the replacement was written
    :ivar corrected_by: a correction written beside it and linked, when it was not replaced
    :ivar alias: the name it opens by, if it has one
    """

    permanent: bool = False
    retired: bool = False
    retired_because: str | None = None
    replaced_by: UUID | None = None
    replaced_at: datetime | None = None
    corrected_by: UUID | None = None
    alias: str | None = None

    @property
    def current(self) -> bool:
        """Whether the memory still holds: neither retired nor replaced."""
        return not self.retired and self.replaced_by is None


def _tag_list(tags: Any) -> list[str]:
    """A ``tags`` value as a list: a raw read hands the column back as JSON text."""
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except ValueError:
            return []
    return [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []


def marks_of(row: dict[str, Any]) -> MemoryMarks:
    """Read a memory row's state from the mark columns the collection selects beside it.

    A row whose replacement no longer exists (``superseded_at`` is ``None``) is
    current again, as every search treats it.

    :param row: a row from ``search_by_ids``, ``search_by_semantic``,
        ``search_by_fts``, ``find_by_alias``, ``find_for_recall`` or
        ``list_for_user``
    :ptype row: dict[str, Any]
    :return: the marks
    :rtype: MemoryMarks
    """
    tags = _tag_list(row.get("tags"))
    because = next((t[len(_BECAUSE) :] for t in tags if t.startswith(_BECAUSE)), None)
    replaced = row.get("superseded_by") if row.get("superseded_at") is not None else None
    corrected = row.get("corrected_by") if replaced is None else None
    return MemoryMarks(
        permanent=bool(row.get("evergreen")),
        retired=RETRACTED_TAG in tags,
        retired_because=because or None,
        replaced_by=UUID(str(replaced)) if replaced is not None else None,
        replaced_at=row.get("superseded_at") if replaced is not None else None,
        corrected_by=UUID(str(corrected)) if corrected is not None else None,
        alias=row.get("alias") or None,
    )


async def _write_linked(
    memories: MemoriesCollection,
    consolidations: MemoryConsolidationsCollection,
    *,
    agent_id: UUID,
    source_ids: list[UUID],
    fields: dict[str, Any],
    rationale: str | None,
) -> UUID:
    """Write a new memory and an edge from each of ``source_ids`` to it; mark nothing."""
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
    return new_id


async def supersede(
    memories: MemoriesCollection,
    consolidations: MemoryConsolidationsCollection,
    *,
    agent_id: UUID,
    source_ids: list[UUID],
    fields: dict[str, Any],
    rationale: str | None,
    allow_permanent: bool = False,
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
    :param allow_permanent: replace a permanent memory too: the caller's word
        that the person the memory is about asked for the change
    :ptype allow_permanent: bool
    :return: the new memory's id
    :rtype: UUID
    :raises ValueError: when asked to replace a permanent memory without ``allow_permanent``
    """
    if not allow_permanent:
        for source_id in source_ids:
            if is_permanent(await memories.get((agent_id, source_id))):
                raise ValueError(f"memory {source_id} is permanent and is never superseded")
    new_id = await _write_linked(
        memories, consolidations, agent_id=agent_id, source_ids=source_ids, fields=fields, rationale=rationale
    )
    await memories.mark_superseded(agent_id, source_memory_ids=source_ids, gist_id=new_id)
    return new_id


@dataclass(frozen=True)
class Written:
    """What :func:`write_once` did.

    :ivar memory_id: the memory written, or ``None`` when nothing was
    :ivar replaced: the earlier writes of the same event it replaced, as
        :meth:`MemoriesCollection.find_same_event` returned them
    :ivar standing: a permanent earlier write of the same event, left as it is
    """

    memory_id: UUID | None
    replaced: tuple[dict[str, Any], ...] = ()
    standing: dict[str, Any] | None = None


async def write_once(
    memories: MemoriesCollection,
    consolidations: MemoryConsolidationsCollection | None,
    *,
    agent_id: UUID,
    fields: dict[str, Any],
    near_message_ids: Sequence[UUID] = (),
    threshold: float | None = None,
    when_permanent: Literal["beside", "skip"] = "beside",
) -> Written:
    """Write a new memory, replacing an earlier write of the same event from this turn or the one before.

    The one place a new memory is written by the writers that can each write
    the same event: extraction after a turn, and the agent's own ``memory_add``.
    One event was stored three and four times in a few minutes (extraction,
    then the agent's own save, then extraction again), each a separate memory.

    "The same event" is two things together: the earlier memory's source
    message is one of ``near_message_ids``, and it is at least ``threshold``
    like the new one. Both, because either alone is wrong: two facts from one
    turn are two memories, and two memories from different months that read
    alike may be two events.

    The new write replaces the earlier ones through :func:`supersede`, so they
    stay readable by id and the edge says why. A permanent earlier write is
    never replaced: with ``when_permanent="skip"`` the new one is not written
    (the permanent one already says it), and with ``"beside"`` it is written
    beside it.

    With no ``threshold``, no ``near_message_ids`` or no edges collection the
    rule is off and this is a plain write.

    :param memories: the memories collection
    :ptype memories: MemoriesCollection
    :param consolidations: the provenance edges collection; ``None`` turns the rule off
    :ptype consolidations: MemoryConsolidationsCollection | None
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param fields: the new memory's columns, as :func:`supersede` takes them;
        ``user_id`` and ``embedding`` are read for the search
    :ptype fields: dict[str, Any]
    :param near_message_ids: the source messages of this turn and the one before it
    :ptype near_message_ids: Sequence[UUID]
    :param threshold: the cosine similarity at which two such memories are one event
    :ptype threshold: float | None
    :param when_permanent: what a permanent earlier write of the event means for the new one
    :ptype when_permanent: Literal["beside", "skip"]
    :return: what was written, replaced and left standing
    :rtype: Written
    """
    same: list[dict[str, Any]] = []
    if threshold is not None and near_message_ids and consolidations is not None and fields.get("embedding"):
        same = await memories.find_same_event(
            user_id=fields["user_id"],
            agent_id=agent_id,
            embedding=fields["embedding"],
            message_ids=list(near_message_ids),
            threshold=threshold,
        )
    standing = next((row for row in same if row["evergreen"]), None)
    replaceable = [row for row in same if not row["evergreen"]]
    if standing is not None and when_permanent == "skip":
        return Written(memory_id=None, standing=standing)
    if replaceable:
        assert consolidations is not None  # ``same`` is empty without it
        new_id = await supersede(
            memories,
            consolidations,
            agent_id=agent_id,
            source_ids=[row["memory_id"] for row in replaceable],
            fields=fields,
            rationale=SAME_EVENT_RATIONALE,
        )
        return Written(memory_id=new_id, replaced=tuple(replaceable), standing=standing)
    new_id = UUID(str(uuid7()))
    now = datetime.now(UTC)
    entity = memories.create(
        {**fields, "memory_id": new_id, "agent_id": agent_id, "date_created": now, "date_updated": now}
    )
    await memories.save_entity(entity)
    return Written(memory_id=new_id, standing=standing)


async def correct_beside(
    memories: MemoriesCollection,
    consolidations: MemoryConsolidationsCollection,
    *,
    agent_id: UUID,
    source_ids: list[UUID],
    fields: dict[str, Any],
    rationale: str | None,
) -> UUID:
    """Write a memory that corrects ``source_ids`` and link each to it, leaving them standing.

    How a correction to a permanent memory is kept: the old one is not marked
    superseded and stays in recall, and a read of either shows the other.

    :param memories: the memories collection
    :ptype memories: MemoriesCollection
    :param consolidations: the provenance edges collection
    :ptype consolidations: MemoryConsolidationsCollection
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param source_ids: the memories it corrects
    :ptype source_ids: list[UUID]
    :param fields: the new memory's columns, as :func:`supersede` takes them
    :ptype fields: dict[str, Any]
    :param rationale: why, kept on each edge
    :ptype rationale: str | None
    :return: the new memory's id
    :rtype: UUID
    """
    return await _write_linked(
        memories, consolidations, agent_id=agent_id, source_ids=source_ids, fields=fields, rationale=rationale
    )


async def retract(
    memories: MemoriesCollection, *, agent_id: UUID, memory_id: UUID, reason: str, allow_permanent: bool = False
) -> bool:
    """Take a memory out of search and ambient recall, keeping it, and say why.

    :param memories: the memories collection
    :ptype memories: MemoriesCollection
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param memory_id: the memory
    :ptype memory_id: UUID
    :param reason: why it no longer holds, kept as its tag
    :ptype reason: str
    :param allow_permanent: retract a permanent memory too: the caller's word
        that the person the memory is about asked for it
    :ptype allow_permanent: bool
    :return: False when it is permanent, already retracted or gone, and nothing changed
    :rtype: bool
    """
    return await memories.mark_retracted(agent_id, memory_id=memory_id, reason=reason, allow_permanent=allow_permanent)


async def restore(memories: MemoriesCollection, *, agent_id: UUID, memory_id: UUID, salience: float = 0.5) -> bool:
    """Undo :func:`retract`: the memory is found by search and ambient recall again.

    Its salience returns to what it was when it was retracted; ``salience`` is
    used only for a memory retracted before that was recorded.

    :param memories: the memories collection
    :ptype memories: MemoriesCollection
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param memory_id: the memory
    :ptype memory_id: UUID
    :param salience: the salience to give a memory whose earlier one is not recorded
    :ptype salience: float
    :return: False when it was not retracted or is gone, and nothing changed
    :rtype: bool
    """
    return await memories.clear_retracted(agent_id, memory_id=memory_id, default_salience=salience)
