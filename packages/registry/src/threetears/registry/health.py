"""heartbeat monitor for tool pod liveness tracking.

subscribes to heartbeat subjects, tracks per-pod status, and
periodically removes endpoints for pods that exceed timeout
threshold. only removes individual endpoints, not entire tool
entries, so surviving pods continue serving.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, MetaData, String, Table, Text

from threetears.agent.tools.server import HeartbeatMessage
from threetears.observe import get_logger
from threetears.registry.catalog import ToolCatalog

__all__ = [
    "HeartbeatMonitor",
    "PodStatus",
]

if TYPE_CHECKING:
    from threetears.core.cache.sqlite import SQLiteBackend

_HEALTH_L1_METADATA = MetaData()

_pod_health_table = Table(
    "pod_health",
    _HEALTH_L1_METADATA,
    Column("key", String, primary_key=True),
    Column("value", Text, nullable=True),
    Column("date_updated", String, nullable=True),
)

_logger = get_logger(__name__)


@dataclass
class PodStatus:
    """tracked status for single tool pod.

    :param pod_id: unique identifier for tool pod
    :ptype pod_id: str
    :param date_last_heartbeat: timestamp of last received heartbeat
    :ptype date_last_heartbeat: datetime
    :param tools: list of full_name values served by this pod
    :ptype tools: list[str]
    :param consecutive_misses: number of consecutive missed health checks
    :ptype consecutive_misses: int
    """

    pod_id: str
    date_last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    tools: list[str] = field(default_factory=list)
    consecutive_misses: int = 0


class HeartbeatMonitor:
    """monitors tool pod health via heartbeat messages.

    subscribes to heartbeat wildcard subject, tracks per-pod
    liveness, marks pod endpoints available on heartbeat, and
    periodically removes endpoints for pods that exceed
    configured timeout. only removes individual endpoints so
    tools with surviving pods remain available.
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        namespace: str = "aibots",
        check_interval: float | None = None,
        timeout: float | None = None,
        l1_backend: SQLiteBackend | None = None,
    ) -> None:
        """initialize heartbeat monitor.

        :param catalog: tool catalog for marking endpoints available/unavailable
        :ptype catalog: ToolCatalog
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param check_interval: seconds between health check sweeps.
            sourced from THREETEARS_REGISTRY_HEARTBEAT_CHECK_INTERVAL env var if not provided.
        :ptype check_interval: float | None
        :param timeout: seconds after which pod is considered dead.
            sourced from THREETEARS_REGISTRY_HEARTBEAT_TIMEOUT env var if not provided.
        :ptype timeout: float | None
        :param l1_backend: optional SQLiteBackend for persistent pod health state
        :ptype l1_backend: SQLiteBackend | None
        """
        from threetears.registry.config import get_heartbeat_check_interval, get_heartbeat_timeout

        self._catalog = catalog
        self._namespace = namespace
        self._check_interval = check_interval if check_interval is not None else get_heartbeat_check_interval()
        self._timeout = timeout if timeout is not None else get_heartbeat_timeout()
        self._l1 = l1_backend
        if self._l1 is not None and not self._l1.is_initialized():
            self._l1.initialize(_HEALTH_L1_METADATA)
        self._pods: dict[str, PodStatus] = {}
        self._nc: Any | None = None
        self._sub: Any | None = None
        self._check_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def pods(self) -> dict[str, PodStatus]:
        """return tracked pod statuses.

        :return: dictionary of pod_id to PodStatus
        :rtype: dict[str, PodStatus]
        """
        return self._pods

    def _persist_pod_to_l1(self, pod_id: str, status: PodStatus) -> None:
        """persist pod status to L1 SQLiteBackend if available.

        :param pod_id: unique pod identifier
        :ptype pod_id: str
        :param status: pod status to persist
        :ptype status: PodStatus
        """
        if self._l1 is None:
            return
        serialized = json.dumps(
            {
                "pod_id": status.pod_id,
                "date_last_heartbeat": status.date_last_heartbeat.isoformat(),
                "tools": status.tools,
                "consecutive_misses": status.consecutive_misses,
            }
        )
        self._l1.upsert(
            "pod_health",
            {
                "key": pod_id,
                "value": serialized,
                "date_updated": datetime.now(UTC).isoformat(),
            },
            primary_key="key",
        )

    async def start(self, nc: Any) -> None:
        """start heartbeat monitoring.

        subscribes to heartbeat wildcard subject and starts
        periodic health check loop.

        :param nc: connected NATS client
        :ptype nc: Any
        """
        self._nc = nc
        self._running = True
        subject = f"{self._namespace}.tools.heartbeat.>"
        self._sub = await nc.subscribe(subject, cb=self._handle_heartbeat)
        self._check_task = asyncio.create_task(self._health_check_loop())
        _logger.info(
            "heartbeat monitor started",
            extra={
                "extra_data": {
                    "subject": subject,
                    "check_interval": self._check_interval,
                    "timeout": self._timeout,
                }
            },
        )

    async def stop(self) -> None:
        """stop heartbeat monitoring.

        unsubscribes from heartbeat subject and cancels health
        check loop.
        """
        self._running = False
        if self._check_task is not None:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
        if self._sub is not None:
            await self._sub.unsubscribe()
            self._sub = None
        _logger.info("heartbeat monitor stopped")

    async def _handle_heartbeat(self, msg: Any) -> None:
        """handle incoming heartbeat message from tool pod.

        updates pod status with current timestamp, resets
        consecutive miss counter, and marks all endpoints for
        this pod as available in catalog.

        :param msg: incoming NATS message containing heartbeat
        :ptype msg: Any
        """
        try:
            heartbeat = HeartbeatMessage.model_validate_json(msg.data)
        except Exception as exc:
            _logger.warning(
                "malformed heartbeat message",
                extra={"extra_data": {"error": str(exc)}},
            )
            return

        now = datetime.now(UTC)
        pod_status = self._pods.get(heartbeat.pod_id)

        if pod_status is None:
            pod_status = PodStatus(
                pod_id=heartbeat.pod_id,
                date_last_heartbeat=now,
            )
            self._pods[heartbeat.pod_id] = pod_status
        else:
            pod_status.date_last_heartbeat = now
            pod_status.consecutive_misses = 0

        marked = self._catalog.mark_pod_endpoints_available(heartbeat.pod_id)
        pod_status.tools = marked
        self._persist_pod_to_l1(heartbeat.pod_id, pod_status)

    async def _health_check_loop(self) -> None:
        """periodically check pod health and deregister timed-out pods.

        runs at configured interval, deregistering endpoints for
        pods that exceed timeout.
        """
        while self._running:
            await asyncio.sleep(self._check_interval)
            await self._run_health_check()

    async def _run_health_check(self) -> None:
        """execute single health check sweep across all tracked pods.

        removes endpoints for pods whose last heartbeat exceeds
        timeout threshold. only removes individual endpoints so
        tools with surviving pods remain available.
        """
        now = datetime.now(UTC)
        to_remove: list[str] = []
        for pod_id, pod_status in self._pods.items():
            elapsed = (now - pod_status.date_last_heartbeat).total_seconds()
            if elapsed > self._timeout:
                pod_status.consecutive_misses += 1
                _logger.warning(
                    "pod heartbeat timeout",
                    extra={
                        "extra_data": {
                            "pod_id": pod_id,
                            "elapsed_seconds": elapsed,
                            "consecutive_misses": pod_status.consecutive_misses,
                        }
                    },
                )
                removed = await self._catalog.deregister_pod(pod_id)
                _logger.info(
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
            self._pods.pop(pod_id, None)
            if self._l1 is not None:
                self._l1.delete_by_id("pod_health", pod_id, primary_key="key")
