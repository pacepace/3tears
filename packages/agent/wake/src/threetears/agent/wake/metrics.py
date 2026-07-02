"""Prometheus instruments for the agent-wake lifecycle.

Bounded-cardinality discipline (PLACEMENT §1.15 + OBS-10 spec
requirement) is enforced by the drift-guard test
:mod:`threetears.agent.wake.tests.unit.test_metrics_cardinality`. The
forbidden label set is exposed as :data:`FORBIDDEN_LABEL_NAMES` so the
walker can read it back via Python rather than via a docstring grep.

Instrument names use the ``threetears_agent_wake_`` prefix. The
PLACEMENT memo §3.1 calls for the literal ``3tears_`` prefix, but
Prometheus identifier rules (RFC: ``[a-zA-Z_][a-zA-Z0-9_]*``) reject
a leading digit -- ``prometheus_client`` silently rewrites the leading
``3`` to ``_``, yielding ``_tears_agent_wake_*`` on /metrics. Using
``threetears_`` mirrors the existing
:mod:`threetears.models.tracking` (``threetears_llm_*``) pattern that
consumer Grafana dashboards already grep against, keeps the
``threetears`` Python namespace and the Prometheus prefix coherent,
and stays Prometheus-spec-legal. Names are locked: renaming any
instrument breaks consumer dashboards. Adding a new instrument is
fine; renaming an existing one is a contract break.

Pattern follows :mod:`threetears.models.tracking` -- ``prometheus_client``
is an OPTIONAL runtime dependency. When the library is not installed
the emitter no-ops every call gracefully, so the platform stays usable
in unit-test contexts that don't pin the extra. The package's
``prometheus`` optional-extras-install gives the consumer the right
default. Consumers shipping their own ``CollectorRegistry`` pass it
via :func:`get_wake_emitter`; the per-registry cache keeps registration
idempotent across re-init paths.

Spec ref: ``docs/agent-wake/shard-05-observability-and-models.md``
OBS-01 .. OBS-10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from threetears.observe import get_logger

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

__all__ = [
    "FORBIDDEN_LABEL_NAMES",
    "WAKE_DRIFT_SECONDS",
    "WAKE_FAILURES_TOTAL",
    "WAKE_FIRES_TOTAL",
    "WAKE_PROMETHEUS_NAMES",
    "WAKE_RATE_LIMIT_REJECTIONS_TOTAL",
    "WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL",
    "WAKE_TICK_DURATION_SECONDS",
    "WAKE_WEBHOOK_RECEIVED_TOTAL",
    "WAKE_YIELD_DURATION_SECONDS",
    "WakeMetricsEmitter",
    "get_wake_emitter",
    "reset_wake_emitter_for_testing",
]


log = get_logger(__name__)


# Labels that MUST NOT appear on any wake Prometheus instrument
# (PLACEMENT §1.15 + OBS-10). These are the unbounded-cardinality
# fields whose use as Prometheus labels would explode the time-series
# database. Enforced by ``test_metrics_cardinality.py``.
FORBIDDEN_LABEL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "conversation_id",
        "user_id",
        "schedule_id",
        "subscription_id",
        "fire_id",
        "agent_id",
    }
)


# Locked instrument names. Renaming breaks consumer dashboards;
# adding a new instrument is fine. The companion ``_LABELS_*``
# tuples pin the bounded-cardinality label sets.
WAKE_FIRES_TOTAL: Final[str] = "threetears_agent_wake_fires_total"
WAKE_FAILURES_TOTAL: Final[str] = "threetears_agent_wake_failures_total"
WAKE_TICK_DURATION_SECONDS: Final[str] = "threetears_agent_wake_tick_duration_seconds"
WAKE_RATE_LIMIT_REJECTIONS_TOTAL: Final[str] = "threetears_agent_wake_rate_limit_rejections_total"
WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL: Final[str] = "threetears_agent_wake_schedule_cap_rejections_total"
WAKE_DRIFT_SECONDS: Final[str] = "threetears_agent_wake_drift_seconds"
WAKE_YIELD_DURATION_SECONDS: Final[str] = "threetears_agent_wake_yield_duration_seconds"
WAKE_WEBHOOK_RECEIVED_TOTAL: Final[str] = "threetears_agent_wake_webhook_received_total"


# Public roll-up of every instrument name shard-05 ships. Lets the
# consumer (and the cardinality drift-guard test) iterate without
# duplicating the list.
WAKE_PROMETHEUS_NAMES: Final[tuple[str, ...]] = (
    WAKE_FIRES_TOTAL,
    WAKE_FAILURES_TOTAL,
    WAKE_TICK_DURATION_SECONDS,
    WAKE_RATE_LIMIT_REJECTIONS_TOTAL,
    WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL,
    WAKE_DRIFT_SECONDS,
    WAKE_YIELD_DURATION_SECONDS,
    WAKE_WEBHOOK_RECEIVED_TOTAL,
)


# Locked bounded-label sets. Each label value's enum is documented at
# the emit site. Cardinality budget per OBS-02 (fires_total) = 9 statuses
# x 7 schedule_types x 2 modes = 126 series. Acceptable.
_LABELS_FIRES: Final[tuple[str, ...]] = ("status", "schedule_type", "execution_mode")
_LABELS_FAILURES: Final[tuple[str, ...]] = ("reason",)
_LABELS_RATE_LIMIT: Final[tuple[str, ...]] = ("scope",)
_LABELS_WEBHOOK_RECEIVED: Final[tuple[str, ...]] = ("outcome",)


# Public mapping of instrument name -> declared labelnames. The
# drift-guard test reads this off to assert no forbidden label slipped
# in. Histograms with no labels record as empty tuple.
WAKE_LABEL_SETS: Final[dict[str, tuple[str, ...]]] = {
    WAKE_FIRES_TOTAL: _LABELS_FIRES,
    WAKE_FAILURES_TOTAL: _LABELS_FAILURES,
    WAKE_TICK_DURATION_SECONDS: (),
    WAKE_RATE_LIMIT_REJECTIONS_TOTAL: _LABELS_RATE_LIMIT,
    WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL: (),
    WAKE_DRIFT_SECONDS: (),
    WAKE_YIELD_DURATION_SECONDS: (),
    WAKE_WEBHOOK_RECEIVED_TOTAL: _LABELS_WEBHOOK_RECEIVED,
}


class WakeMetricsEmitter:
    """Holder + emitter for the locked agent-wake Prometheus instruments.

    Mirrors the :class:`threetears.models.tracking._PrometheusEmitter`
    pattern: lazily registers instruments against an optional
    :class:`prometheus_client.CollectorRegistry`; degrades gracefully
    to no-op when ``prometheus_client`` is not installed.

    Construction does the registration once. Subsequent calls
    :meth:`inc_fire`, :meth:`observe_tick_duration`, etc. emit samples;
    they are silent no-ops when the emitter is in the unavailable
    state.

    :param registry: optional Prometheus collector registry to register
        the locked instruments against; ``None`` uses the default
        global registry
    :ptype registry: CollectorRegistry | None
    """

    def __init__(self, registry: "CollectorRegistry | None" = None) -> None:
        self._registry = registry
        self._available = False
        self._resolved_registry: Any = None
        self._fires: Any = None
        self._failures: Any = None
        self._tick_duration: Any = None
        self._rate_limit: Any = None
        self._schedule_cap: Any = None
        self._drift: Any = None
        self._yield_duration: Any = None
        self._webhook_received: Any = None
        self._initialise()

    def _initialise(self) -> None:
        """Import ``prometheus_client`` + register the locked instruments.

        On import failure (extra not installed) leaves ``_available``
        as ``False``; subsequent emit calls then become no-ops.
        Resolves the registry once so :meth:`unregister_from_registry`
        survives ``sys.modules`` monkeypatching in tests.
        """
        try:
            from prometheus_client import REGISTRY, Counter, Histogram
        except ImportError:
            log.debug(
                "prometheus_client not installed -- agent-wake metrics disabled",
            )
            return

        self._resolved_registry = self._registry if self._registry is not None else REGISTRY

        # prometheus_client treats ``registry=None`` specially -- pass
        # through verbatim so the default-registry path is identical
        # to pre-knob behaviour.
        def _kwargs(labels: tuple[str, ...]) -> dict[str, Any]:
            kw: dict[str, Any] = {}
            if labels:
                kw["labelnames"] = labels
            if self._registry is not None:
                kw["registry"] = self._registry
            return kw

        self._fires = Counter(
            WAKE_FIRES_TOTAL,
            "Total wake fires by status, schedule_type, and execution_mode",
            **_kwargs(_LABELS_FIRES),
        )
        self._failures = Counter(
            WAKE_FAILURES_TOTAL,
            "Wake fire failures by reason",
            **_kwargs(_LABELS_FAILURES),
        )
        self._tick_duration = Histogram(
            WAKE_TICK_DURATION_SECONDS,
            "Wake tick body duration in seconds",
            **_kwargs(()),
        )
        self._rate_limit = Counter(
            WAKE_RATE_LIMIT_REJECTIONS_TOTAL,
            "Wake rate-limit rejections by scope",
            **_kwargs(_LABELS_RATE_LIMIT),
        )
        self._schedule_cap = Counter(
            WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL,
            "Wake per-conv active-schedule cap rejections",
            **_kwargs(()),
        )
        self._drift = Histogram(
            WAKE_DRIFT_SECONDS,
            "Wake fire drift (actual_fired_at - scheduled_fire_at) in seconds",
            **_kwargs(()),
        )
        self._yield_duration = Histogram(
            WAKE_YIELD_DURATION_SECONDS,
            "Wake-yield duration (actual_fired_at -> yield point) in seconds",
            **_kwargs(()),
        )
        self._webhook_received = Counter(
            WAKE_WEBHOOK_RECEIVED_TOTAL,
            "Webhook receiver outcomes",
            **_kwargs(_LABELS_WEBHOOK_RECEIVED),
        )
        self._available = True

    @property
    def available(self) -> bool:
        """Return ``True`` when the instruments registered cleanly."""
        return self._available

    def unregister_from_registry(self) -> None:
        """Unregister every instrument so a test can rebuild the emitter.

        Used only by :func:`reset_wake_emitter_for_testing`; production
        callers leave the emitter for the process lifetime. ``KeyError``
        / ``ValueError`` raised by the registry's ``unregister`` are
        swallowed -- the cleanup is best-effort.
        """
        if not self._available or self._resolved_registry is None:
            return
        for collector in (
            self._fires,
            self._failures,
            self._tick_duration,
            self._rate_limit,
            self._schedule_cap,
            self._drift,
            self._yield_duration,
            self._webhook_received,
        ):
            if collector is None:
                continue
            try:
                self._resolved_registry.unregister(collector)
            except KeyError:
                # collector wasn't actually on the registry -- benign
                # race with another test's teardown path; continue.
                continue
            except ValueError:
                # prometheus_client raises ValueError when a collector
                # is registered against a different registry than ours;
                # also benign for the unregister-best-effort flow.
                continue
        self._fires = None
        self._failures = None
        self._tick_duration = None
        self._rate_limit = None
        self._schedule_cap = None
        self._drift = None
        self._yield_duration = None
        self._webhook_received = None
        self._available = False

    # ------------------------------------------------------------------
    # Emit helpers -- bounded enums documented inline. Each emitter is
    # a no-op when the emitter is unavailable (prometheus_client not
    # installed in the consumer environment).
    # ------------------------------------------------------------------

    def inc_fire(
        self,
        *,
        status: str,
        schedule_type: str,
        execution_mode: str,
    ) -> None:
        """Increment :data:`WAKE_FIRES_TOTAL`.

        :param status: one of the :data:`FireStatus` enum values
        :ptype status: str
        :param schedule_type: one of the :data:`ScheduleType` enum
            values (plus ``'webhook'`` for webhook-source fires)
        :ptype schedule_type: str
        :param execution_mode: ``'inline'`` or ``'spawn'``
        :ptype execution_mode: str
        """
        if not self._available:
            return
        self._fires.labels(
            status=status,
            schedule_type=schedule_type,
            execution_mode=execution_mode,
        ).inc()

    def inc_failure(self, *, reason: str) -> None:
        """Increment :data:`WAKE_FAILURES_TOTAL` by reason.

        Reasons (bounded): ``conv_deleted``, ``rate_limited``,
        ``handler_exception``, ``cap_exceeded``, ``no_handler``,
        ``other``.

        :param reason: bounded failure reason string
        :ptype reason: str
        """
        if not self._available:
            return
        self._failures.labels(reason=reason).inc()

    def observe_tick_duration(self, seconds: float) -> None:
        """Observe one tick body's duration.

        :param seconds: tick wall-clock duration in seconds
        :ptype seconds: float
        """
        if not self._available:
            return
        self._tick_duration.observe(seconds)

    def inc_rate_limit_rejection(self, *, scope: str) -> None:
        """Increment :data:`WAKE_RATE_LIMIT_REJECTIONS_TOTAL`.

        :param scope: ``'conv'`` | ``'user'`` | ``'webhook'``
        :ptype scope: str
        """
        if not self._available:
            return
        self._rate_limit.labels(scope=scope).inc()

    def inc_schedule_cap_rejection(self) -> None:
        """Increment :data:`WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL`."""
        if not self._available:
            return
        self._schedule_cap.inc()

    def observe_drift(self, seconds: float) -> None:
        """Observe one fire's drift (``actual_fired_at - scheduled_fire_at``).

        :param seconds: drift in seconds (always non-negative -- the
            wake clock never fires early)
        :ptype seconds: float
        """
        if not self._available:
            return
        self._drift.observe(seconds)

    def observe_yield_duration(self, seconds: float) -> None:
        """Observe one yielded fire's wake-to-yield duration.

        :param seconds: duration in seconds between the wake start
            instant and the yield call instant (PLACEMENT §8.5.1)
        :ptype seconds: float
        """
        if not self._available:
            return
        self._yield_duration.observe(seconds)

    def inc_webhook_received(self, *, outcome: str) -> None:
        """Increment :data:`WAKE_WEBHOOK_RECEIVED_TOTAL`.

        :param outcome: ``'accepted'`` | ``'auth_failed'`` |
            ``'rate_limited'`` | ``'bad_template'`` |
            ``'source_rejected'`` | ``'not_found'`` | ``'failed'``
        :ptype outcome: str
        """
        if not self._available:
            return
        self._webhook_received.labels(outcome=outcome).inc()


# Per-registry emitter cache -- registering the same instrument twice
# on the same registry raises in prometheus_client, so we cache per
# ``id(registry)``. Sentinel key ``0`` indexes the default global
# registry path.
_EMITTERS: dict[int, WakeMetricsEmitter] = {}


def get_wake_emitter(
    registry: "CollectorRegistry | None" = None,
) -> WakeMetricsEmitter:
    """Return the cached :class:`WakeMetricsEmitter` for ``registry``.

    Pass-through to a per-process / per-registry singleton so emission
    sites don't have to thread the registry through every call path.
    The consumer typically calls this once at startup with its
    ``CollectorRegistry`` (or omits the arg to use the default global
    one) and stashes the result.

    :param registry: optional Prometheus collector registry
    :ptype registry: CollectorRegistry | None
    :return: shared emitter (no-op when ``prometheus_client`` missing)
    :rtype: WakeMetricsEmitter
    """
    key = 0 if registry is None else id(registry)
    emitter = _EMITTERS.get(key)
    if emitter is None:
        emitter = WakeMetricsEmitter(registry=registry)
        _EMITTERS[key] = emitter
    return emitter


def reset_wake_emitter_for_testing() -> None:
    """Tear down + recycle the per-registry emitter cache.

    Only intended for the test suite -- production callers never run
    this. For each cached emitter, unregisters the locked instruments
    from the underlying registry, then drops the cache entry so the
    next :func:`get_wake_emitter` call builds fresh.

    Without this, a test that hides ``prometheus_client`` from
    ``sys.modules`` would leave stale collectors on the registry and
    the next emitter's ``_initialise`` would raise ``ValueError:
    Duplicated timeseries``.
    """
    for emitter in _EMITTERS.values():
        emitter.unregister_from_registry()
    _EMITTERS.clear()
