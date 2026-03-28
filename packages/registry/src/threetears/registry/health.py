"""heartbeat monitor for tool pod liveness tracking.

subscribes to heartbeat subjects, tracks per-pod status, and
periodically deregisters pods that exceed timeout threshold.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from threetears.agent.tools.server import HeartbeatMessage
from threetears.core.logging import get_logger
from threetears.registry.catalog import ToolCatalog

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
    liveness, and periodically deregisters pods that exceed
    configured timeout.
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        namespace: str = "aibots",
        check_interval: float = 5.0,
        timeout: float = 30.0,
    ) -> None:
        """initialize heartbeat monitor.

        :param catalog: tool catalog for deregistering timed-out pods
        :ptype catalog: ToolCatalog
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param check_interval: seconds between health check sweeps
        :ptype check_interval: float
        :param timeout: seconds after which pod is considered dead
        :ptype timeout: float
        """
        self._catalog = catalog
        self._namespace = namespace
        self._check_interval = check_interval
        self._timeout = timeout
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
            extra={"extra_data": {
                "subject": subject,
                "check_interval": self._check_interval,
                "timeout": self._timeout,
            }},
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

        updates pod status with current timestamp and resets
        consecutive miss counter. marks all tools from pod as
        available.

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

        self._mark_pod_tools_available(heartbeat.pod_id)

    def _mark_pod_tools_available(self, pod_id: str) -> None:
        """mark all tools from specified pod as available in catalog.

        :param pod_id: identifier of pod whose tools to mark available
        :ptype pod_id: str
        """
        pod_status = self._pods.get(pod_id)
        if pod_status is None:
            return
        for full_name in pod_status.tools:
            self._catalog.mark_available(full_name)

    async def _health_check_loop(self) -> None:
        """periodically check pod health and deregister timed-out pods.

        runs at configured interval, incrementing consecutive_misses
        for each pod and deregistering those exceeding timeout.
        """
        while self._running:
            await asyncio.sleep(self._check_interval)
            await self._run_health_check()

    async def _run_health_check(self) -> None:
        """execute single health check sweep across all tracked pods.

        deregisters pods whose last heartbeat exceeds timeout
        threshold.
        """
        now = datetime.now(UTC)
        to_remove: list[str] = []
        for pod_id, pod_status in self._pods.items():
            elapsed = (now - pod_status.date_last_heartbeat).total_seconds()
            if elapsed > self._timeout:
                pod_status.consecutive_misses += 1
                _logger.warning(
                    "pod heartbeat timeout",
                    extra={"extra_data": {
                        "pod_id": pod_id,
                        "elapsed_seconds": elapsed,
                        "consecutive_misses": pod_status.consecutive_misses,
                    }},
                )
                removed = await self._catalog.deregister_pod(pod_id)
                _logger.info(
                    "deregistered timed-out pod",
                    extra={"extra_data": {
                        "pod_id": pod_id,
                        "removed_tools": removed,
                    }},
                )
                to_remove.append(pod_id)
        for pod_id in to_remove:
            self._pods.pop(pod_id, None)
