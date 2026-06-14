"""Prometheus instruments for the scheduled-jobs lifecycle.

Generalized from :mod:`threetears.agent.wake.metrics`. The instrument
set is trimmed to the payload-agnostic tick + fire + drift surface (the
webhook / rate-limit / schedule-cap / yield instruments were agent-
specific and are dropped), and the forbidden-label guard is generalized
to the generic id columns. The cardinality discipline is preserved: the
forbidden label set is exposed as :data:`FORBIDDEN_LABEL_NAMES` so a
drift-guard test can read it back via Python rather than via a docstring
grep, and the bounded label sets are pinned per instrument.

Instrument names use the ``threetears_scheduled_jobs_`` prefix.
Prometheus identifier rules (``[a-zA-Z_][a-zA-Z0-9_]*``) reject a leading
digit, so the literal ``3tears_`` prefix is illegal; ``threetears_``
mirrors the rest of the platform and stays spec-legal. Names are locked:
renaming an instrument breaks consumer dashboards; adding one is fine.

``prometheus_client`` is an OPTIONAL runtime dependency. When the
library is absent the emitter no-ops every call gracefully, so the
package stays usable in unit-test contexts that don't pin the extra. The
``prometheus`` optional-extra installs it for consumers that want the
instruments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from threetears.observe import get_logger

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

__all__ = [
    "FORBIDDEN_LABEL_NAMES",
    "SCHEDULED_JOBS_DRIFT_SECONDS",
    "SCHEDULED_JOBS_FAILURES_TOTAL",
    "SCHEDULED_JOBS_FIRES_TOTAL",
    "SCHEDULED_JOBS_LABEL_SETS",
    "SCHEDULED_JOBS_PROMETHEUS_NAMES",
    "SCHEDULED_JOBS_TICK_DURATION_SECONDS",
    "ScheduledJobsMetricsEmitter",
    "get_scheduled_jobs_emitter",
    "reset_scheduled_jobs_emitter_for_testing",
]


log = get_logger(__name__)


# Labels that MUST NOT appear on any scheduled-jobs Prometheus
# instrument. These are the unbounded-cardinality id fields whose use as
# labels would explode the time-series database. The generic id columns
# replace agent-wake's ``conversation_id`` / ``user_id`` / ``agent_id``
# etc. ``payload`` / ``kind`` are also forbidden -- ``kind`` because a
# consumer may mint unbounded kinds; the bounded labels below carry only
# ``status`` / ``schedule_type`` / ``reason``.
FORBIDDEN_LABEL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "partition_key",
        "job_id",
        "fire_id",
        "payload",
        "kind",
    }
)


# Locked instrument names. Renaming breaks consumer dashboards; adding a
# new instrument is fine.
SCHEDULED_JOBS_FIRES_TOTAL: Final[str] = "threetears_scheduled_jobs_fires_total"
SCHEDULED_JOBS_FAILURES_TOTAL: Final[str] = "threetears_scheduled_jobs_failures_total"
SCHEDULED_JOBS_TICK_DURATION_SECONDS: Final[str] = "threetears_scheduled_jobs_tick_duration_seconds"
SCHEDULED_JOBS_DRIFT_SECONDS: Final[str] = "threetears_scheduled_jobs_drift_seconds"


# Public roll-up of every instrument name. Lets the consumer (and the
# cardinality drift-guard test) iterate without duplicating the list.
SCHEDULED_JOBS_PROMETHEUS_NAMES: Final[tuple[str, ...]] = (
    SCHEDULED_JOBS_FIRES_TOTAL,
    SCHEDULED_JOBS_FAILURES_TOTAL,
    SCHEDULED_JOBS_TICK_DURATION_SECONDS,
    SCHEDULED_JOBS_DRIFT_SECONDS,
)


# Locked bounded-label sets. ``schedule_type`` is bounded (a small
# app-evolvable enum); ``reason`` is a small documented enum. The
# ``status`` label on fires_total is fed the consumer's
# ``JobFireResult.status`` -- an OPEN value at this layer, so its
# cardinality is NOT a fixed 3. The cardinality guard instead relies on
# the consumer constraining ``status`` to a bounded terminal-status enum
# (a bounded ``Literal``): fires_total stays at (bounded statuses) x
# (bounded schedule_types) series. The guard's intent -- keep this
# instrument's series count bounded -- holds as long as the consumer
# passes a bounded status enum, never a free-form string.
_LABELS_FIRES: Final[tuple[str, ...]] = ("status", "schedule_type")
_LABELS_FAILURES: Final[tuple[str, ...]] = ("reason",)


# Public mapping of instrument name -> declared labelnames. The
# drift-guard test reads this off to assert no forbidden label slipped
# in. Histograms with no labels record as empty tuple.
SCHEDULED_JOBS_LABEL_SETS: Final[dict[str, tuple[str, ...]]] = {
    SCHEDULED_JOBS_FIRES_TOTAL: _LABELS_FIRES,
    SCHEDULED_JOBS_FAILURES_TOTAL: _LABELS_FAILURES,
    SCHEDULED_JOBS_TICK_DURATION_SECONDS: (),
    SCHEDULED_JOBS_DRIFT_SECONDS: (),
}


class ScheduledJobsMetricsEmitter:
    """Holder + emitter for the locked scheduled-jobs Prometheus instruments.

    Lazily registers instruments against an optional
    :class:`prometheus_client.CollectorRegistry`; degrades gracefully to
    no-op when ``prometheus_client`` is not installed. Construction does
    the registration once; subsequent emit calls are silent no-ops when
    the emitter is in the unavailable state.

    :param registry: optional Prometheus collector registry to register
        the locked instruments against; ``None`` uses the default global
        registry
    :ptype registry: CollectorRegistry | None
    """

    def __init__(self, registry: "CollectorRegistry | None" = None) -> None:
        self._registry = registry
        self._available = False
        self._resolved_registry: Any = None
        self._fires: Any = None
        self._failures: Any = None
        self._tick_duration: Any = None
        self._drift: Any = None
        self._initialise()

    def _initialise(self) -> None:
        """Import ``prometheus_client`` + register the locked instruments.

        On import failure (extra not installed) leaves ``_available`` as
        ``False``; subsequent emit calls become no-ops.
        """
        try:
            from prometheus_client import REGISTRY, Counter, Histogram
        except ImportError:
            log.debug("prometheus_client not installed -- scheduled-jobs metrics disabled")
            return

        self._resolved_registry = self._registry if self._registry is not None else REGISTRY

        def _kwargs(labels: tuple[str, ...]) -> dict[str, Any]:
            kw: dict[str, Any] = {}
            if labels:
                kw["labelnames"] = labels
            if self._registry is not None:
                kw["registry"] = self._registry
            return kw

        self._fires = Counter(
            SCHEDULED_JOBS_FIRES_TOTAL,
            "Total job fires by status and schedule_type",
            **_kwargs(_LABELS_FIRES),
        )
        self._failures = Counter(
            SCHEDULED_JOBS_FAILURES_TOTAL,
            "Job fire failures by reason",
            **_kwargs(_LABELS_FAILURES),
        )
        self._tick_duration = Histogram(
            SCHEDULED_JOBS_TICK_DURATION_SECONDS,
            "Tick body duration in seconds",
            **_kwargs(()),
        )
        self._drift = Histogram(
            SCHEDULED_JOBS_DRIFT_SECONDS,
            "Fire drift (actual_fired_at - scheduled_fire_at) in seconds",
            **_kwargs(()),
        )
        self._available = True

    @property
    def available(self) -> bool:
        """Return ``True`` when the instruments registered cleanly."""
        return self._available

    def unregister_from_registry(self) -> None:
        """Unregister every instrument so a test can rebuild the emitter.

        Used only by :func:`reset_scheduled_jobs_emitter_for_testing`;
        production callers leave the emitter for the process lifetime.
        ``KeyError`` / ``ValueError`` raised by the registry's
        ``unregister`` are swallowed -- the cleanup is best-effort.
        """
        if not self._available or self._resolved_registry is None:
            return
        for collector in (self._fires, self._failures, self._tick_duration, self._drift):
            if collector is None:
                continue
            try:
                self._resolved_registry.unregister(collector)
            except KeyError:
                # collector wasn't on the registry -- benign race; continue.
                continue
            except ValueError:
                # registered against a different registry than ours; also
                # benign for the unregister-best-effort flow.
                continue
        self._fires = None
        self._failures = None
        self._tick_duration = None
        self._drift = None
        self._available = False

    # ------------------------------------------------------------------
    # Emit helpers -- bounded enums documented inline. Each emitter is a
    # no-op when the emitter is unavailable.
    # ------------------------------------------------------------------

    def inc_fire(self, *, status: str, schedule_type: str) -> None:
        """Increment :data:`SCHEDULED_JOBS_FIRES_TOTAL`.

        :param status: one of the
            :data:`~threetears.scheduled_jobs.types.ScheduleFireStatus`
            values
        :ptype status: str
        :param schedule_type: one of the
            :data:`~threetears.scheduled_jobs.types.ScheduleType` values
        :ptype schedule_type: str
        """
        if not self._available:
            return
        self._fires.labels(status=status, schedule_type=schedule_type).inc()

    def inc_failure(self, *, reason: str) -> None:
        """Increment :data:`SCHEDULED_JOBS_FAILURES_TOTAL` by reason.

        Reasons (bounded): ``handler_exception``, ``claim_lost``,
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

    def observe_drift(self, seconds: float) -> None:
        """Observe one fire's drift (``actual_fired_at - scheduled_fire_at``).

        :param seconds: drift in seconds (always non-negative -- the
            clock never fires early)
        :ptype seconds: float
        """
        if not self._available:
            return
        self._drift.observe(seconds)


# Per-registry emitter cache -- registering the same instrument twice on
# the same registry raises in prometheus_client, so we cache per
# ``id(registry)``. Sentinel key ``0`` indexes the default global
# registry path.
_EMITTERS: dict[int, ScheduledJobsMetricsEmitter] = {}


def get_scheduled_jobs_emitter(
    registry: "CollectorRegistry | None" = None,
) -> ScheduledJobsMetricsEmitter:
    """Return the cached :class:`ScheduledJobsMetricsEmitter` for ``registry``.

    Pass-through to a per-process / per-registry singleton so emission
    sites don't have to thread the registry through every call path.

    :param registry: optional Prometheus collector registry
    :ptype registry: CollectorRegistry | None
    :return: shared emitter (no-op when ``prometheus_client`` missing)
    :rtype: ScheduledJobsMetricsEmitter
    """
    key = 0 if registry is None else id(registry)
    emitter = _EMITTERS.get(key)
    if emitter is None:
        emitter = ScheduledJobsMetricsEmitter(registry=registry)
        _EMITTERS[key] = emitter
    return emitter


def reset_scheduled_jobs_emitter_for_testing() -> None:
    """Tear down + recycle the per-registry emitter cache.

    Only intended for the test suite -- production callers never run
    this. For each cached emitter, unregisters the locked instruments
    from the underlying registry, then drops the cache entry so the next
    :func:`get_scheduled_jobs_emitter` call builds fresh.
    """
    for emitter in _EMITTERS.values():
        emitter.unregister_from_registry()
    _EMITTERS.clear()
