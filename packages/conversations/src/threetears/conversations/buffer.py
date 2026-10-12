"""
ConversationWriteBuffer -- cross-conversation pod-wide batched writes.

every agent pod multiplexes many conversations concurrently. each
inbound message bumps the conversation row's ``message_count`` and
``date_last_message`` columns; without batching, every inbound
envelope produces its own L3 round-trip (50ms-100ms typical). per
Pace's DQ-D2 nuance on the audit response: per-conversation buffers
do NOT help here -- a multi-conversation pod amortizes the L3 write
cost across conversations via batched-across-conversations writes,
not by waiting longer on any one conversation.

the buffer holds an in-memory map of pending deltas keyed by
``(agent_id, conversation_id)``. flush triggers (whichever fires
first):

- timer (default 30s) -- bounded latency for low-traffic conversations
- message-count threshold (default 100) -- bounded queue size
- explicit :meth:`stop` / :meth:`flush` -- drain on pod shutdown

each flush walks the queued deltas and routes them through the
collection's :meth:`save_entity` path; the collection's own write
buffer takes over for the cross-pod-coherence side. failure of any
one entry is logged and skipped; the buffer never blocks the pod.

platform-agnostic: any 3tears app whose pods multiplex multiple
conversations benefits. the buffer is constructed in the agent
bootstrap once per pod and shared by every inbound envelope on
that pod.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.conversations.collection import ConversationsCollection

__all__ = [
    "ConversationWriteBuffer",
]

log = get_logger(__name__)


_DEFAULT_FLUSH_INTERVAL_SECONDS: float = 30.0
_DEFAULT_FLUSH_THRESHOLD_MESSAGES: int = 100


class _ConversationDelta:
    """accumulating delta for one conversation between flushes.

    captures the running ``message_count`` increment, the most recent
    ``date_last_message`` timestamp, and the role of the most recent
    message. the buffer collapses every observed message for the same
    ``conversation_id`` into one delta; on flush, one ``save_entity``
    call applies all of them at once.

    :param conversation_id: conversation UUID this delta belongs to
    :ptype conversation_id: UUID
    :param agent_id: agent partition the conversation lives in
    :ptype agent_id: UUID
    """

    __slots__ = (
        "agent_id",
        "conversation_id",
        "increment",
        "latest_at",
        "latest_role",
    )

    def __init__(self, agent_id: UUID, conversation_id: UUID) -> None:
        """initialize an empty delta for the given partition + conversation.

        :param agent_id: agent partition the conversation lives in
        :ptype agent_id: UUID
        :param conversation_id: conversation UUID this delta belongs to
        :ptype conversation_id: UUID
        :return: nothing
        :rtype: None
        """
        self.agent_id = agent_id
        self.conversation_id = conversation_id
        self.increment = 0
        self.latest_at: datetime | None = None
        self.latest_role: str | None = None

    def merge(self, at: datetime, role: str) -> None:
        """fold one observed message into the running delta.

        increments the counter, advances ``latest_at`` only when
        ``at`` is newer than the prior entry (out-of-order arrivals
        do not roll the timestamp backwards), and overwrites
        ``latest_role`` with the most-recent role.

        :param at: timestamp the message was observed at
        :ptype at: datetime
        :param role: short actor token (``user`` / ``assistant`` / ...)
        :ptype role: str
        :return: nothing
        :rtype: None
        """
        self.increment += 1
        if self.latest_at is None or at > self.latest_at:
            self.latest_at = at
            self.latest_role = role


class ConversationWriteBuffer:
    """cross-conversation pod-wide batched write buffer.

    constructed once per agent pod, shared by every inbound envelope.
    callers invoke :meth:`enqueue` per observed message; the buffer
    coalesces deltas across conversations and flushes them on a timer
    (default 30s) or once the running message count exceeds the
    threshold (default 100). :meth:`stop` drains the buffer on pod
    shutdown.

    flush policy decisions (data-layer-task-01 sub-task 3):

    - timer window 30s: bounds latency for the conversation list
      admin surface; slow enough that a low-traffic pod amortizes
      one round-trip across many conversations.
    - message threshold 100: bounds the in-memory queue depth on a
      busy pod and gives a fast-flush path for traffic spikes.
    - shutdown drain: every queued delta lands before the pod exits
      so admin queries reflect the conversation state at exit time.

    :param collection: canonical
        :class:`~threetears.conversations.ConversationsCollection`
        the buffer routes flushes through; the collection's own write
        buffer takes over for cross-pod L1 invalidation
    :ptype collection: ConversationsCollection
    :param flush_interval_seconds: timer window between scheduled
        flushes; default 30s. zero disables the timer (only
        threshold + shutdown trigger flushes)
    :ptype flush_interval_seconds: float
    :param flush_threshold_messages: maximum running message count
        before an immediate flush fires; default 100
    :ptype flush_threshold_messages: int
    """

    def __init__(
        self,
        *,
        collection: ConversationsCollection,
        flush_interval_seconds: float = _DEFAULT_FLUSH_INTERVAL_SECONDS,
        flush_threshold_messages: int = _DEFAULT_FLUSH_THRESHOLD_MESSAGES,
    ) -> None:
        """initialize the buffer with empty state and configured policy.

        :param collection: canonical conversations collection
        :ptype collection: ConversationsCollection
        :param flush_interval_seconds: timer window between flushes
        :ptype flush_interval_seconds: float
        :param flush_threshold_messages: count threshold for fast flush
        :ptype flush_threshold_messages: int
        :return: nothing
        :rtype: None
        """
        self._collection = collection
        self._flush_interval_seconds = flush_interval_seconds
        self._flush_threshold = flush_threshold_messages
        self._deltas: dict[tuple[UUID, UUID], _ConversationDelta] = {}
        self._lock = asyncio.Lock()
        self._running_count = 0
        self._timer_task: asyncio.Task[None] | None = None
        self._stopped = False

    async def start(self) -> None:
        """start the background flush timer.

        idempotent. when ``flush_interval_seconds`` is zero the
        timer is skipped (flush only fires on threshold or stop).
        the timer runs until :meth:`stop` is called.

        :return: nothing
        :rtype: None
        """
        if self._timer_task is not None:
            return
        if self._flush_interval_seconds <= 0:
            return
        self._timer_task = asyncio.create_task(
            self._timer_loop(),
            name="conversation-write-buffer-timer",
        )

    async def stop(self) -> None:
        """drain the buffer and stop the timer.

        called from agent pod shutdown. cancels the timer, performs
        one last flush so every queued delta lands, then sets the
        ``_stopped`` flag so subsequent :meth:`enqueue` calls drop
        their entries (the pod is going away; recording a delta
        we cannot persist would just leak memory).

        :return: nothing
        :rtype: None
        """
        if self._stopped:
            return
        self._stopped = True
        if self._timer_task is not None:
            self._timer_task.cancel()
            try:
                await self._timer_task
            # NOSILENT: cancellation is the expected end of the timer loop
            except asyncio.CancelledError:
                pass
            self._timer_task = None
        await self.flush()

    async def enqueue(
        self,
        *,
        agent_id: UUID,
        conversation_id: UUID,
        at: datetime,
        role: str,
    ) -> None:
        """enqueue one observed message for batched flush.

        coalesces with any prior delta for the same
        ``(agent_id, conversation_id)`` pair: the counter accumulates
        and ``date_last_message`` advances to the most recent ``at``.
        when the running count crosses the threshold an immediate
        flush is scheduled (without blocking the caller).

        :param agent_id: agent partition the conversation lives in
        :ptype agent_id: UUID
        :param conversation_id: conversation UUID this delta belongs to
        :ptype conversation_id: UUID
        :param at: timestamp the message was observed at
        :ptype at: datetime
        :param role: short actor token (``user`` / ``assistant`` / ...)
        :ptype role: str
        :return: nothing
        :rtype: None
        """
        if self._stopped:
            return
        normalized = at.astimezone(UTC) if at.tzinfo else at.replace(tzinfo=UTC)
        key = (agent_id, conversation_id)
        async with self._lock:
            delta = self._deltas.get(key)
            if delta is None:
                delta = _ConversationDelta(agent_id, conversation_id)
                self._deltas[key] = delta
            delta.merge(normalized, role)
            self._running_count += 1
            should_flush = self._running_count >= self._flush_threshold
        if should_flush:
            asyncio.create_task(
                self.flush(),
                name="conversation-write-buffer-threshold-flush",
            )

    async def flush(self) -> None:
        """drain the buffer and apply every queued delta.

        per-delta flow: load the conversation entity (lazy create if
        missing -- the entity must already exist in normal operation
        because the inbound envelope path lazy-creates it), apply the
        running delta via :meth:`Conversation.record_message` once
        per accumulated message, persist via
        :meth:`ConversationsCollection.save_entity`. failures on any
        one delta are logged and skipped so a single bad row never
        blocks the rest.

        :return: nothing
        :rtype: None
        """
        async with self._lock:
            pending = self._deltas
            self._deltas = {}
            self._running_count = 0
        if not pending:
            return
        log.debug(
            "conversation write buffer flush",
            extra={
                "extra_data": {
                    "delta_count": len(pending),
                }
            },
        )
        for delta in pending.values():
            try:
                await self._apply_delta(delta)
            except Exception as exc:
                log.warning(
                    "conversation write buffer apply failed (soft-fail): %s",
                    exc,
                    extra={
                        "extra_data": {
                            "agent_id": str(delta.agent_id),
                            "conversation_id": str(delta.conversation_id),
                            "increment": delta.increment,
                        }
                    },
                )

    async def _apply_delta(self, delta: _ConversationDelta) -> None:
        """apply one accumulated delta to the persisted conversation row.

        the conversation entity must already exist (the inbound
        envelope path is the lazy-creator). when the row is missing
        the delta is dropped with a debug log: a missing row at flush
        time means the conversation never landed in the first place,
        which is a higher-level bug, not a buffer concern.

        :param delta: accumulated delta to apply
        :ptype delta: _ConversationDelta
        :return: nothing
        :rtype: None
        """
        if delta.increment == 0 or delta.latest_at is None:
            return
        pk = (delta.agent_id, delta.conversation_id)
        entity = await self._collection.get(pk)
        if entity is None:
            log.debug(
                "conversation write buffer: missing entity at flush; dropping",
                extra={
                    "extra_data": {
                        "agent_id": str(delta.agent_id),
                        "conversation_id": str(delta.conversation_id),
                    }
                },
            )
            return
        for _ in range(delta.increment):
            entity.record_message(
                delta.latest_at,
                delta.latest_role or "unknown",
            )
        await self._collection.save_entity(entity)

    async def _timer_loop(self) -> None:
        """background loop: flush every ``flush_interval_seconds``.

        sleeps the configured window between flushes. cancellation
        ends the loop; the final drain happens in :meth:`stop`.

        :return: nothing
        :rtype: None
        """
        try:
            while True:
                await asyncio.sleep(self._flush_interval_seconds)
                try:
                    await self.flush()
                except Exception as exc:
                    log.warning(
                        "conversation write buffer timer flush failed (soft-fail): %s",
                        exc,
                    )
        # NOSILENT: cancellation ends the timer loop on pod shutdown
        except asyncio.CancelledError:
            return
