"""PresenceSweeper — heartbeat-driven self-heal for presence state.

channels-task-01's self-heal mechanism. modelled on
:class:`~threetears.registry.health.HeartbeatSubscriber`: a periodic
sweep asks the per-connection Collection for each tracked connection and
evicts any whose ``date_last_heartbeat`` falls outside the liveness
window, pruning it from its room-index member set. a dead pod's
connections vanish automatically — no KV-TTL, no manual cleanup.

the sweeper holds no persistent state; ``_known_connection_ids`` is an
in-memory set mirroring the per-connection Collection's primary keys for
this process's lifetime. join/leave on the local
:class:`~threetears.channels.presence.room_state.RoomState` seeds + clears
the set via :meth:`track` / :meth:`untrack`; cross-pod, every pod sweeps
the connections IT tracks, and the eviction's room-index prune + L2
invalidation make the removal visible fleet-wide.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime

from threetears.observe import get_logger

from threetears.channels.presence.collection import PresenceCollection

__all__ = ["PresenceSweeper"]

log = get_logger(__name__)

#: default seconds between sweeps and the default staleness threshold.
#: deliberately conservative defaults; callers tune per deployment.
_DEFAULT_CHECK_INTERVAL = 15.0
_DEFAULT_TIMEOUT = 45.0


class PresenceSweeper:
    """periodic staleness sweep over tracked presence connections.

    :param collection: the presence state surface
    :ptype collection: PresenceCollection
    :param check_interval: seconds between sweeps
    :ptype check_interval: float
    :param timeout: seconds after which a connection with no fresh
        heartbeat is considered stale and evicted
    :ptype timeout: float
    """

    def __init__(
        self,
        collection: PresenceCollection,
        check_interval: float = _DEFAULT_CHECK_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._collection = collection
        self._check_interval = check_interval
        self._timeout = timeout
        self._running = False
        self._check_task: asyncio.Task[None] | None = None
        self._known_connection_ids: set[str] = set()

    @property
    def known_connection_ids(self) -> set[str]:
        """return a copy of the tracked connection-id set.

        :return: tracked connection ids
        :rtype: set[str]
        """
        return set(self._known_connection_ids)

    @property
    def has_active_check_task(self) -> bool:
        """return whether the periodic sweep task is running.

        :return: ``True`` between :meth:`start` and :meth:`stop`
        :rtype: bool
        """
        return self._check_task is not None

    def track(self, connection_id: str) -> None:
        """start tracking a connection for the staleness sweep.

        :param connection_id: opaque per-socket id
        :ptype connection_id: str
        :return: nothing
        :rtype: None
        """
        self._known_connection_ids.add(connection_id)

    def track_many(self, connection_ids: Iterable[str]) -> None:
        """start tracking several connections (recovery / multi-join).

        :param connection_ids: connection ids to track
        :ptype connection_ids: Iterable[str]
        :return: nothing
        :rtype: None
        """
        self._known_connection_ids.update(connection_ids)

    def untrack(self, connection_id: str) -> None:
        """stop tracking a connection (clean leave).

        :param connection_id: opaque per-socket id
        :ptype connection_id: str
        :return: nothing
        :rtype: None
        """
        self._known_connection_ids.discard(connection_id)

    async def start(self) -> None:
        """spawn the periodic sweep loop.

        :return: nothing
        :rtype: None
        """
        self._running = True
        self._check_task = asyncio.create_task(self._sweep_loop())
        log.info(
            "presence sweeper started",
            extra={"extra_data": {"check_interval": self._check_interval, "timeout": self._timeout}},
        )

    async def stop(self) -> None:
        """cancel the sweep loop; presence state in the collection is untouched.

        :return: nothing
        :rtype: None
        """
        self._running = False
        if self._check_task is not None:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
        log.info("presence sweeper stopped")

    async def _sweep_loop(self) -> None:
        """run the staleness sweep on the configured interval until stopped.

        a single sweep failure is logged and does not brick the loop —
        one mis-timed sweep must not stop self-heal.

        :return: nothing
        :rtype: None
        """
        while self._running:
            await asyncio.sleep(self._check_interval)
            if not self._running:
                break
            try:
                await self.run_sweep()
            except Exception as exc:
                # supervisor loop: surface any sweep failure to the log
                # without stopping self-heal -- a single mis-timed sweep
                # must not brick presence eviction (mirrors
                # HeartbeatSubscriber._health_check_loop).
                log.warning(
                    "presence sweep failed",
                    extra={"extra_data": {"error": str(exc)}},
                )

    async def run_sweep(self) -> list[str]:
        """execute one staleness sweep across tracked connections.

        for each tracked connection: a pk-get of its per-connection row;
        a row already gone (peer evicted it) is dropped from the tracked
        set; a row whose ``date_last_heartbeat`` is older than ``timeout``
        is pruned from its room-index member set, its per-connection row
        deleted, and it is dropped from the tracked set. fresh
        connections are left untouched.

        :return: the connection ids evicted by this sweep
        :rtype: list[str]
        """
        if not self._known_connection_ids:
            return []
        now = datetime.now(UTC)
        evicted: list[str] = []
        gone: list[str] = []
        for connection_id in list(self._known_connection_ids):
            entity = await self._collection.connections.get(connection_id)
            if entity is None:
                gone.append(connection_id)
                continue
            elapsed = (now - entity.date_last_heartbeat).total_seconds()
            if elapsed <= self._timeout:
                continue
            room_id = entity.room_id
            await self._collection.rooms.remove_member(room_id, connection_id)
            await self._collection.connections.delete(connection_id)
            evicted.append(connection_id)
            log.warning(
                "evicting stale presence connection",
                extra={
                    "extra_data": {
                        "connection_id": connection_id,
                        "room_id": room_id,
                        "elapsed_seconds": elapsed,
                    }
                },
            )
        for connection_id in (*evicted, *gone):
            self._known_connection_ids.discard(connection_id)
        return evicted
