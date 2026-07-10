"""``InflightRequestsGauge`` -- leak-safe in-flight request gauge primitive.

the request/reply RPC pods (gateway, registry, tool pods, hub) scale
horizontally via core-NATS queue groups (one-of-N delivery) rather than
JetStream. KEDA's ``prometheus`` scaler therefore cannot read a stream
backlog to decide when to scale; instead each pod exposes an
IN-FLIGHT-REQUESTS gauge at a prometheus ``/metrics`` endpoint and KEDA
scales the Deployment on the aggregate / average in-flight load.

this module owns the one shared primitive every RPC pod uses to publish
that gauge. the design goals:

- **leak-safe bracket.** :meth:`InflightRequestsGauge.track` is a context
  manager that increments the gauge on entry and decrements it on exit
  via ``try/finally`` -- the decrement runs even when the wrapped request
  raises, so a failing request never strands the counter above its true
  value (which would make KEDA over-scale forever).
- **prometheus is OPTIONAL.** mirroring
  :mod:`threetears.scheduled_jobs.metrics`, the primitive imports
  ``prometheus_client`` lazily and degrades to a silent no-op when the
  library is absent, so :mod:`threetears.observe` keeps its zero-hard-
  dependency contract. pods that actually scale (registry, tool pods)
  declare ``prometheus-client`` as a hard dependency so the gauge is live
  in production.
- **low cardinality.** the gauge carries NO labels; each pod publishes a
  distinct metric NAME (``threetears_registry_inflight_requests``,
  ``gateway_inflight_requests``, ...) so a per-pod KEDA ScaledObject
  queries exactly one time series. prometheus identifier rules reject a
  leading digit, so 3tears-side names use the ``threetears_`` prefix
  (``3tears_`` is illegal), matching the scheduled-jobs convention.

usage::

    from threetears.observe.inflight import InflightRequestsGauge

    gauge = InflightRequestsGauge("threetears_registry_inflight_requests")

    async def handle_call(msg):
        with gauge.track():
            await process(msg)

    # served on the shared HealthServer's /metrics route:
    content_type, body = gauge.render()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from threetears.observe.logging import get_logger

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

__all__ = ["InflightRequestsGauge"]

log = get_logger(__name__)


# prometheus exposition content-type used when the library is present.
# the fallback (library absent) is a plain text/plain so the /metrics
# route still returns a well-formed empty response.
_FALLBACK_CONTENT_TYPE = "text/plain; charset=utf-8"


class InflightRequestsGauge:
    """leak-safe in-flight request gauge with an optional prometheus backend.

    wraps a single unlabelled ``prometheus_client.Gauge`` and exposes a
    :meth:`track` context manager that brackets one request: increment on
    entry, decrement on exit (``try/finally``, so the decrement survives
    an exception raised inside the bracket). when ``prometheus_client`` is
    not installed every method is a silent no-op and :meth:`render`
    returns an empty body, so the primitive is safe to import and call
    from a package that does not pin the prometheus extra.

    :param metric_name: prometheus metric name (no labels). MUST be a
        valid prometheus identifier (``[a-zA-Z_][a-zA-Z0-9_]*``); 3tears
        callers use the ``threetears_`` prefix because a leading digit is
        illegal
    :ptype metric_name: str
    :param registry: existing collector registry to register the gauge on
        (gateway / hub pass their dedicated registry so the gauge joins
        their existing ``/metrics`` surface); ``None`` creates a private
        registry the pod serves through the shared HealthServer
    :ptype registry: CollectorRegistry | None
    :param description: gauge help text; defaults to a description derived
        from ``metric_name``
    :ptype description: str | None
    """

    def __init__(
        self,
        metric_name: str,
        *,
        registry: "CollectorRegistry | None" = None,
        description: str | None = None,
    ) -> None:
        """initialize gauge, registering it on the resolved registry.

        :param metric_name: prometheus metric name (no labels)
        :ptype metric_name: str
        :param registry: collector registry to register on, or ``None``
            to own a private registry
        :ptype registry: CollectorRegistry | None
        :param description: gauge help text (derived from name when None)
        :ptype description: str | None
        :return: nothing
        :rtype: None
        """
        self._metric_name = metric_name
        self._description = description if description is not None else f"{metric_name} in-flight request count"
        self._registry: Any = registry
        self._gauge: Any = None
        self._available = False
        self._initialise(registry)

    def _initialise(self, registry: "CollectorRegistry | None") -> None:
        """import ``prometheus_client`` and register the gauge.

        on import failure (extra not installed) leaves the primitive in
        its no-op state: :meth:`track` still brackets execution but
        mutates nothing, and :meth:`render` returns an empty body.

        :param registry: collector registry supplied by the caller, or
            ``None`` to own a private one
        :ptype registry: CollectorRegistry | None
        :return: nothing
        :rtype: None
        """
        try:
            from prometheus_client import CollectorRegistry, Gauge
        except ImportError:
            log.debug("prometheus_client not installed -- in-flight gauge disabled")
            return
        self._registry = registry if registry is not None else CollectorRegistry()
        self._gauge = Gauge(self._metric_name, self._description, registry=self._registry)
        self._available = True

    @property
    def metric_name(self) -> str:
        """return the prometheus metric name this gauge publishes.

        :return: metric name
        :rtype: str
        """
        return self._metric_name

    @property
    def available(self) -> bool:
        """return whether the gauge registered against a live prometheus backend.

        :return: ``True`` when ``prometheus_client`` is installed and the
            gauge registered cleanly; ``False`` in the no-op state
        :rtype: bool
        """
        return self._available

    @property
    def value(self) -> float:
        """return the gauge's current value (``0.0`` in the no-op state).

        reads the sample straight off the registry so tests can assert
        the gauge returns to its baseline after a bracketed request.

        :return: current in-flight count
        :rtype: float
        """
        result = 0.0
        if self._registry is not None:
            sample = self._registry.get_sample_value(self._metric_name)
            if sample is not None:
                result = float(sample)
        return result

    @contextmanager
    def track(self) -> Iterator[None]:
        """bracket one in-flight request: increment on entry, decrement on exit.

        the decrement runs in a ``finally`` so it survives an exception
        raised inside the ``with`` body -- a failing request decrements
        the gauge exactly as a succeeding one does, so the counter never
        leaks upward. a no-op (yields without mutating) when the gauge is
        in the no-op state.

        :return: iterator yielding once for the ``with`` body
        :rtype: Iterator[None]
        """
        if self._gauge is not None:
            self._gauge.inc()
        try:
            yield
        finally:
            if self._gauge is not None:
                self._gauge.dec()

    def render(self) -> tuple[str, bytes]:
        """render the registry in prometheus text exposition format.

        returned as ``(content_type, body)`` so the shared HealthServer's
        ``/metrics`` route can serve it without importing
        ``prometheus_client`` itself. in the no-op state (library absent)
        returns a plain-text empty body.

        :return: tuple of prometheus content-type and exposition body
        :rtype: tuple[str, bytes]
        """
        if not self._available or self._registry is None:
            result: tuple[str, bytes] = (_FALLBACK_CONTENT_TYPE, b"")
        else:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            result = (CONTENT_TYPE_LATEST, generate_latest(self._registry))
        return result
