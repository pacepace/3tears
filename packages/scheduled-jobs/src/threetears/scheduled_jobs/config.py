"""Tick-engine config Protocol + platform defaults.

Generalized from :mod:`threetears.agent.wake.config`. agent-wake's
``WakeConfig`` carried agent-specific policy (per-conversation fire caps,
per-user caps, webhook caps, HTTP allow-lists, Loki/Postgres named-query
registries). None of that is generic. What IS generic is the tick
engine's own operational knobs: the cross-pod lock key and the per-tick
due-row scan cap. :class:`JobConfig` declares that read-side shape and
the platform ships ``DEFAULT_*`` constants the consumer can fall back to
or override per-deployment.

The Protocol is pure-read; the engine consults it once per tick. A
consumer that wants the platform-baseline behaviour can use
:data:`DEFAULT_JOB_CONFIG` directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

__all__ = [
    "DEFAULT_DISPATCH_REAP_AFTER_SECONDS",
    "DEFAULT_DISPATCH_REAP_AFTER_SECONDS_BY_KIND",
    "DEFAULT_JOB_CONFIG",
    "DEFAULT_TICK_DUE_LIMIT",
    "DEFAULT_TICK_LOCK_KEY",
    "JobConfig",
    "reap_after_seconds_for_kind",
]


# Default cross-pod lock key the tick engine acquires. A held lock means
# another pod is already running this tick -- the engine skips silently.
# Consumers running multiple independent tick pumps in one process supply
# distinct keys so the pumps do not exclude each other.
DEFAULT_TICK_LOCK_KEY: str = "scheduled_jobs_tick"


# Default per-tick cap on the due-row scan. The engine pages no further
# than this many rows per tick; a larger backlog drains over subsequent
# ticks. Mirrors agent-wake's ``list_due_for_tick`` default ``limit=200``.
DEFAULT_TICK_DUE_LIMIT: int = 200


# Default age (seconds) after which a ``job_fires`` row still stuck in
# ``'dispatching'`` is reaped to ``'failed'`` by the tick's sweep. A
# dispatch is expected to stage its work and return promptly, so a row
# in-flight for many minutes signals a pod that died mid-dispatch; the
# occurrence's schedule already advanced, so the row would otherwise
# remain a zombie forever. 15 minutes is generous headroom over a normal
# dispatch while still surfacing the loss the same operational day.
DEFAULT_DISPATCH_REAP_AFTER_SECONDS: int = 900


# Default per-``kind`` reap-threshold overrides: none. Every kind falls
# back to :data:`DEFAULT_DISPATCH_REAP_AFTER_SECONDS`, so a deployment
# that configures nothing behaves exactly as it did before per-kind
# thresholds existed.
#
# A kind whose work legitimately runs for hours needs a larger value
# here -- a dataset build carries a multi-hour statement ceiling, and the
# 15-minute baseline would reap it mid-run and record the loss as a
# failure. **A larger threshold alone does not fix false reaping**: the
# age is measured from dispatch start, not last activity, so a bigger
# number only moves the cliff. The complete fix is this threshold PLUS
# progress-conditioned renewal keyed on a ``date_last_progress`` column
# on the run row (``dsh-task-04b`` adds the column, ``dsh-task-09``
# renews on it). Neither half is sufficient alone.
DEFAULT_DISPATCH_REAP_AFTER_SECONDS_BY_KIND: Mapping[str, int] = MappingProxyType({})


@runtime_checkable
class JobConfig(Protocol):
    """Read-side operational config for the tick engine.

    Pure read protocol -- no mutation methods. Every property has a
    corresponding ``DEFAULT_*`` constant at module scope so a consumer
    that wants the platform baseline can delegate to the defaults from
    its own implementation.

    :ivar tick_lock_key: the cross-pod lock key the tick acquires.
        Consumers running more than one pump in a process MUST vary this
        -- two pumps sharing a key serialise against each other for no
        reason
    :ivar tick_due_limit: per-tick cap on the due-row scan
    :ivar dispatch_reap_after_seconds: fallback age after which a stuck
        ``'dispatching'`` fire row is reaped to ``'failed'``
    :ivar dispatch_reap_after_seconds_by_kind: per-``kind`` overrides of
        that age; a kind absent from the mapping uses the fallback.
        Resolve through :func:`reap_after_seconds_for_kind` rather than
        reading the mapping directly
    """

    @property
    def tick_lock_key(self) -> str: ...

    @property
    def tick_due_limit(self) -> int: ...

    @property
    def dispatch_reap_after_seconds(self) -> int: ...

    @property
    def dispatch_reap_after_seconds_by_kind(self) -> Mapping[str, int]: ...


class _DefaultJobConfig:
    """Concrete :class:`JobConfig` returning only the platform defaults.

    Every property delegates to the corresponding module-level
    ``DEFAULT_*`` constant so an operator changing a default sees it land
    everywhere at once. Consumers wanting per-deployment overrides supply
    their own :class:`JobConfig` impl.
    """

    @property
    def tick_lock_key(self) -> str:
        return DEFAULT_TICK_LOCK_KEY

    @property
    def tick_due_limit(self) -> int:
        return DEFAULT_TICK_DUE_LIMIT

    @property
    def dispatch_reap_after_seconds(self) -> int:
        return DEFAULT_DISPATCH_REAP_AFTER_SECONDS

    @property
    def dispatch_reap_after_seconds_by_kind(self) -> Mapping[str, int]:
        return DEFAULT_DISPATCH_REAP_AFTER_SECONDS_BY_KIND


# Platform-default :class:`JobConfig` singleton. Returns the ``DEFAULT_*``
# constants; used by :func:`threetears.scheduled_jobs.tick.scheduled_tick_job`
# when no consumer-supplied config is passed.
DEFAULT_JOB_CONFIG: JobConfig = _DefaultJobConfig()


def reap_after_seconds_for_kind(config: JobConfig, kind: str) -> int:
    """Resolve the stale-dispatch reap age for one job ``kind``.

    Consults ``config.dispatch_reap_after_seconds_by_kind`` and falls
    back to ``config.dispatch_reap_after_seconds`` when the kind
    configures no override, so an unconfigured kind behaves exactly as
    it did before per-kind thresholds existed.

    :param config: operational config carrying both the fallback and the
        per-kind overrides
    :ptype config: JobConfig
    :param kind: the job kind whose threshold to resolve
    :ptype kind: str
    :return: reap age in seconds
    :rtype: int
    """
    return config.dispatch_reap_after_seconds_by_kind.get(kind, config.dispatch_reap_after_seconds)
