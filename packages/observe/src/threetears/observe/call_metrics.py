"""``CallMetrics`` -- every finished call counted, with its duration, by what was called and how it ended.

An in-flight gauge says how many calls a pod is running at the moment it is scraped. A call that starts and
ends between two scrapes is never seen by it: on cobalt dev on 2026-10-30 the reports tool pods answered
thousands of calls in six hours and their gauge read 0 at every scrape. A counter and a histogram keep every
call, so a dashboard reads the rate (``rate(<prefix>_calls_total[5m])``), the latency, and the mean
concurrency (the rate of ``<prefix>_call_duration_seconds_sum``), at any scrape interval.

Like :class:`~threetears.observe.inflight.InflightRequestsGauge`, prometheus is optional: without
``prometheus_client`` every method is a silent no-op. Pass the gauge's registry so both are served on the
pod's one ``/metrics`` route::

    gauge = InflightRequestsGauge("threetears_tools_inflight_requests")
    calls = CallMetrics("threetears_tools", registry=gauge.registry)
    calls.observe(name="report_rows", outcome="success", seconds=0.012)

The caller keeps both labels bounded: ``name`` from what the pod serves (never what a caller asked for),
``outcome`` from a fixed set.
"""

from __future__ import annotations

from typing import Any

from threetears.observe.logging import get_logger

try:
    from prometheus_client import CollectorRegistry, Counter, Histogram

    _PROMETHEUS = True
except ImportError:  # NOSILENT: the prometheus extra is optional; the metrics are disabled without it (logged)
    _PROMETHEUS = False

__all__ = ["CALL_DURATION_BUCKETS", "CallMetrics"]

log = get_logger(__name__)

#: seconds: a cached answer (milliseconds) through a long tool (minutes)
CALL_DURATION_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    300.0,
)


class CallMetrics:
    """a calls counter and a duration histogram, labelled by what was called and how it ended.

    :param prefix: metric name prefix, a prometheus identifier (``threetears_tools``); the metrics are
        ``<prefix>_calls_total`` and ``<prefix>_call_duration_seconds``
    :ptype prefix: str
    :param label: the name of the label naming what was called (``tool``)
    :ptype label: str
    :param registry: the collector registry to register on, normally the pod's in-flight gauge's, so one
        ``/metrics`` route serves both; ``None`` makes a private one
    :ptype registry: CollectorRegistry | None
    """

    def __init__(self, prefix: str, *, label: str = "tool", registry: "CollectorRegistry | None" = None) -> None:
        """register the counter and the histogram, or stay a no-op without prometheus.

        :param prefix: metric name prefix
        :ptype prefix: str
        :param label: the label naming what was called
        :ptype label: str
        :param registry: collector registry to register on, or ``None`` for a private one
        :ptype registry: CollectorRegistry | None
        :return: nothing
        :rtype: None
        """
        self._label = label
        self._calls: Any = None
        self._duration: Any = None
        if not _PROMETHEUS:
            log.debug("prometheus_client not installed -- call metrics disabled")
            return
        target = registry if registry is not None else CollectorRegistry()
        labels = (label, "outcome")
        self._calls = Counter(
            f"{prefix}_calls", "calls finished, by what was called and how it ended", labels, registry=target
        )
        self._duration = Histogram(
            f"{prefix}_call_duration_seconds",
            "seconds from a call's arrival to its end, by what was called and how it ended",
            labels,
            buckets=CALL_DURATION_BUCKETS,
            registry=target,
        )

    def observe(self, *, name: str, outcome: str, seconds: float) -> None:
        """count one finished call and its duration.

        :param name: what was called, from a bounded set (``""`` for a call that named nothing served)
        :ptype name: str
        :param outcome: how it ended, from a fixed set (``success`` / ``failure`` / ``error``)
        :ptype outcome: str
        :param seconds: how long it took
        :ptype seconds: float
        :return: nothing
        :rtype: None
        """
        if self._calls is None:
            return
        labels = {self._label: name, "outcome": outcome}
        self._calls.labels(**labels).inc()
        self._duration.labels(**labels).observe(max(seconds, 0.0))
