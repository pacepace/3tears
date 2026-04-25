"""heartbeat subscriber orchestration for tool pod liveness tracking.

subscribes to heartbeat subjects, drives per-pod liveness state
through :class:`~threetears.registry.heartbeat_collection.HeartbeatCollection`,
and periodically removes catalog endpoints for pods that exceed the
liveness timeout. endpoint removal is fine-grained (only the stale
pod's endpoints), so tools served by surviving pods continue serving.

namespace-task-01 phase 8.5l-3 split the former :class:`HeartbeatMonitor`
into two focused classes:

- :class:`HeartbeatCollection`
  (:mod:`threetears.registry.heartbeat_collection`) owns persistent
  per-pod state (L1 + L2 NATS KV). the Collection is the data surface
  for any caller that needs to read a pod's status (hub admin
  dashboards, other registry processes after an L1 restart).
- :class:`HeartbeatSubscriber` (this module) owns orchestration:
  NATS subscription on ``{ns}.tools.heartbeat.>``, the periodic
  health-check sweep, and tool-catalog invalidation when a pod
  exceeds the liveness timeout. it consumes the Collection through
  constructor injection and never reaches into L1 / L2 directly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from threetears.agent.tools.server import HeartbeatMessage
from threetears.nats import Subjects
from threetears.observe import get_logger
from threetears.registry.catalog import ToolCatalog
from threetears.registry.entities import HeartbeatEntity
from threetears.registry.heartbeat_collection import HeartbeatCollection

if TYPE_CHECKING:
    from threetears.nats import NatsClient, Subscription

__all__ = ["HeartbeatSubscriber"]

log = get_logger(__name__)


_STATUS_HEALTHY = "healthy"
_STATUS_UNRESPONSIVE = "unresponsive"


class HeartbeatSubscriber:
    """orchestrate heartbeat subscription, health-check loop, catalog eviction.

    subscribes to the registry's heartbeat wildcard subject, tracks
    the set of pods this registry has ever observed, and runs a
    periodic sweep that asks the :class:`HeartbeatCollection` for each
    known pod's current state. pods whose ``date_last_heartbeat`` is
    outside the configured liveness timeout have their endpoints
    dropped from the :class:`ToolCatalog` and are marked
    ``"unresponsive"`` in the Collection before being purged.

    the subscriber holds no persistent state itself. ``_known_pod_ids``
    is an in-memory set mirroring the Collection's primary keys for
    the lifetime of this process; after a restart the set repopulates
    from incoming heartbeats (and, incidentally, from L2 pull-through
    the first time the Collection resolves a known pod). on clean
    shutdown the subscriber unsubscribes from NATS and cancels the
    health-check task; the Collection's rows stay in L1/L2 so a
    restarting registry can pick up the catalog view without waiting
    for the next heartbeat.
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        collection: HeartbeatCollection,
        namespace: str = "aibots",
        check_interval: float | None = None,
        timeout: float | None = None,
    ) -> None:
        """initialize heartbeat subscriber.

        :param catalog: tool catalog for marking endpoints
            available / deregistering stale pods
        :ptype catalog: ToolCatalog
        :param collection: persistent heartbeat state surface; every
            pod save / read flows through this Collection
        :ptype collection: HeartbeatCollection
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param check_interval: seconds between health-check sweeps;
            sourced from THREETEARS_REGISTRY_HEARTBEAT_CHECK_INTERVAL
            env var when ``None``
        :ptype check_interval: float | None
        :param timeout: seconds after which a pod is considered
            unresponsive; sourced from
            THREETEARS_REGISTRY_HEARTBEAT_TIMEOUT env var when
            ``None``
        :ptype timeout: float | None
        """
        from threetears.registry.config import (
            get_heartbeat_check_interval,
            get_heartbeat_timeout,
        )

        self._catalog = catalog
        self._collection = collection
        self._namespace = namespace
        self._check_interval = (
            check_interval
            if check_interval is not None
            else get_heartbeat_check_interval()
        )
        self._timeout = timeout if timeout is not None else get_heartbeat_timeout()
        self._nc: "NatsClient | None" = None
        self._sub: "Subscription | None" = None
        self._check_task: asyncio.Task[None] | None = None
        self._running = False
        self._known_pod_ids: set[str] = set()

    @property
    def has_active_check_task(self) -> bool:
        """return whether the periodic health-check task is currently running.

        :return: ``True`` after :meth:`start` creates the task and
            before :meth:`stop` cancels it; ``False`` otherwise
        :rtype: bool
        """
        return self._check_task is not None

    @property
    def known_pod_ids(self) -> set[str]:
        """return pod ids currently tracked by the subscriber.

        :return: a copy of the in-memory known-pods set
        :rtype: set[str]
        """
        return set(self._known_pod_ids)

    def track_pod(self, pod_id: str) -> None:
        """seed a pod id into the in-memory known-pods set.

        normally :meth:`handle_heartbeat` populates the set on first
        heartbeat; this method lets tests and recovery paths inject a
        pod id directly so the next :meth:`run_health_check` sweep
        will examine it.

        :param pod_id: pod identifier to start tracking
        :ptype pod_id: str
        :return: nothing
        :rtype: None
        """
        self._known_pod_ids.add(pod_id)

    def track_pods(self, pod_ids: Iterable[str]) -> None:
        """seed multiple pod ids into the in-memory known-pods set.

        batch variant of :meth:`track_pod`; convenient for restoring
        tracking state from a Collection scan at pod startup or for
        multi-pod test setup.

        :param pod_ids: iterable of pod identifiers to start tracking
        :ptype pod_ids: Iterable[str]
        :return: nothing
        :rtype: None
        """
        self._known_pod_ids.update(pod_ids)

    async def start(self, nc: "NatsClient") -> None:
        """start heartbeat monitoring.

        subscribes to the heartbeat wildcard subject and spawns the
        periodic health-check loop.

        DQ-B7 queue-group note: heartbeat subscription is intentionally
        NOT in a queue group -- every registry replica must observe
        every pod's heartbeats so its local catalog stays warm. the
        Collection's invalidation pipeline keeps cross-replica state
        consistent.

        :param nc: connected canonical NATS wrapper client
        :ptype nc: NatsClient
        :return: nothing
        :rtype: None
        """
        self._nc = nc
        self._running = True
        subject = Subjects.tools_heartbeat_wildcard()
        self._sub = await nc.subscribe(subject=subject, cb=self.handle_heartbeat)
        self._check_task = asyncio.create_task(self._health_check_loop())
        log.info(
            "heartbeat subscriber started",
            extra={
                "extra_data": {
                    "subject": subject.path,
                    "check_interval": self._check_interval,
                    "timeout": self._timeout,
                }
            },
        )

    async def stop(self) -> None:
        """stop heartbeat monitoring.

        unsubscribes from the heartbeat subject and cancels the
        health-check loop. persistent pod state in the Collection is
        untouched; a peer registry process continues to track the
        pods via the L2 tier.

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
        if self._sub is not None and self._nc is not None:
            await self._nc.unsubscribe(self._sub)
            self._sub = None
        log.info("heartbeat subscriber stopped")

    async def handle_heartbeat(self, msg: Any) -> None:
        """public NATS-subject handler for incoming heartbeat from a tool pod.

        bound by :meth:`start` as the ``cb`` callback on
        ``{namespace}.tools.heartbeat.>`` so every heartbeat-publishing
        tool pod's message arrives here. tests exercise this surface
        directly; the name, the single ``msg`` parameter, and the lack
        of return value are part of the stability contract.

        the parameter type stays ``Any`` (not :class:`IncomingMessage`)
        because the unit tests call this handler directly with raw
        :class:`HeartbeatMessage` -shaped wire bytes wrapped in a
        :class:`MagicMock`; ``msg.data`` is the only field read in the
        body.

        updates the pod's state in the Collection (creating it if
        this is the pod's first heartbeat), marks every endpoint for
        this pod available in the catalog, and records the pod id
        in the in-memory known-pods set so the health-check loop
        will examine it on the next sweep.

        malformed payloads are logged and dropped; no persistent
        state is mutated.

        :param msg: wrapper :class:`IncomingMessage` envelope OR raw
            NATS-shaped object exposing ``.data`` (for legacy test
            doubles)
        :ptype msg: Any
        :return: nothing
        :rtype: None
        """
        try:
            heartbeat = HeartbeatMessage.model_validate_json(msg.data)
        except Exception as exc:
            log.warning(
                "malformed heartbeat message",
                extra={"extra_data": {"error": str(exc)}},
            )
            return

        now = datetime.now(UTC)
        pod_id = heartbeat.pod_id
        marked_tools = self._catalog.mark_pod_endpoints_available(pod_id)
        existing = await self._collection.get(pod_id)
        entity: HeartbeatEntity
        if existing is None:
            entity = self._collection.create(
                {
                    "pod_id": pod_id,
                    "date_last_heartbeat": now,
                    "tools": marked_tools,
                    "tools_count": heartbeat.tools_count,
                    "status": _STATUS_HEALTHY,
                    "consecutive_misses": 0,
                }
            )
        else:
            entity = existing
            entity.date_last_heartbeat = now
            entity.tools = marked_tools
            entity.tools_count = heartbeat.tools_count
            entity.status = _STATUS_HEALTHY
            entity.consecutive_misses = 0
        await self._collection.save_entity(entity)
        self._known_pod_ids.add(pod_id)

    async def _health_check_loop(self) -> None:
        """run the periodic health-check sweep until :meth:`stop`.

        sleeps for ``check_interval`` seconds between sweeps and
        surfaces any sweep exception to the logger without stopping
        the loop; a single mis-timed sweep must not brick liveness
        tracking.

        :return: nothing
        :rtype: None
        """
        while self._running:
            await asyncio.sleep(self._check_interval)
            if not self._running:
                break
            try:
                await self.run_health_check()
            except Exception as exc:
                log.warning(
                    "health check sweep failed",
                    extra={"extra_data": {"error": str(exc)}},
                )

    async def run_health_check(self) -> None:
        """execute one health-check sweep across known pods.

        asks the Collection for each known pod and compares its
        ``date_last_heartbeat`` against the configured liveness
        timeout. pods that exceed the timeout are marked
        ``"unresponsive"`` in the Collection, their endpoints are
        dropped from the :class:`ToolCatalog`, and they are purged
        from the Collection + the in-memory known-pods set so the
        next sweep does not re-examine them. the Collection delete
        publishes an L2 invalidation so peer registry processes
        evict their own L1 copy.

        :return: nothing
        :rtype: None
        """
        if not self._known_pod_ids:
            return
        now = datetime.now(UTC)
        to_remove: list[str] = []
        for pod_id in list(self._known_pod_ids):
            entity = await self._collection.get(pod_id)
            if entity is None:
                # peer registry already evicted; drop from our set
                to_remove.append(pod_id)
                continue
            last = entity.date_last_heartbeat
            if isinstance(last, datetime) and last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            elapsed = (now - last).total_seconds()
            if elapsed <= self._timeout:
                continue
            entity.consecutive_misses = int(entity.consecutive_misses) + 1
            entity.status = _STATUS_UNRESPONSIVE
            await self._collection.save_entity(entity)
            log.warning(
                "pod heartbeat timeout",
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "elapsed_seconds": elapsed,
                        "consecutive_misses": entity.consecutive_misses,
                    }
                },
            )
            removed = await self._catalog.deregister_pod(pod_id)
            log.info(
                "deregistered timed-out pod endpoints",
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "removed_tools": removed,
                    }
                },
            )
            to_remove.append(pod_id)
        for pod_id in to_remove:
            self._known_pod_ids.discard(pod_id)
            await self._collection.delete(pod_id)
