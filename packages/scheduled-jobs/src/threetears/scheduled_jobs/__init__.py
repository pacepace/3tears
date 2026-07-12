"""3tears-scheduled-jobs -- generic, payload-agnostic scheduled-jobs core.

Extracted and generalized from ``3tears-agent-wake``'s scheduling
machinery with every agent/skill/webhook/conversation-specific concept
stripped out. The public surface:

- :func:`scheduled_tick_job` -- the cross-pod-locked tick pump. Takes the
  injected store(s) + dispatch callback + NATS client; no domain
  knowledge.
- :func:`compute_next_fire_at` -- the pure reschedule math for every
  schedule type + both missed-fire policies.
- :class:`ScheduleStore` / :class:`FireStore` / :class:`DueSchedule` --
  the store Protocols the tick engine depends on (and nothing else).
- :class:`ScheduledJobEntity` / :class:`JobFireEntity` +
  :class:`ScheduledJobCollection` / :class:`JobFireCollection` +
  :func:`scheduled_jobs_table` / :func:`job_fires_table` +
  :func:`register` -- the default ``kind`` + ``payload`` store a simple
  consumer can use as-is.
- :class:`JobConfig` / :class:`JobTrigger` / :class:`JobFireResult` + the
  schedule-type / fire-status Literals -- the vocabulary.
- the event-name constants + the cardinality-bounded metrics emitter.

Version is sourced from the installed package metadata so a release that
bumps ``pyproject.toml`` without touching this file cannot drift the
runtime ``__version__`` reporting.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-scheduled-jobs")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.scheduled_jobs.collections import (
    JobFireCollection,
    ScheduledJobCollection,
)
from threetears.scheduled_jobs.config import (
    DEFAULT_DISPATCH_REAP_AFTER_SECONDS,
    DEFAULT_JOB_CONFIG,
    DEFAULT_TICK_DUE_LIMIT,
    DEFAULT_TICK_LOCK_KEY,
    JobConfig,
)
from threetears.scheduled_jobs.entities import (
    JobFireEntity,
    ScheduledJobEntity,
)
from threetears.scheduled_jobs.events import (
    EVENT_FIRE_DISPATCHED,
    EVENT_FIRE_DRIFT,
    EVENT_FIRE_FAILED,
    EVENT_FIRE_SKIPPED_BUSY,
    EVENT_TICK_COMPLETED,
    EVENT_TICK_STARTED,
)
from threetears.scheduled_jobs.metrics import (
    FORBIDDEN_LABEL_NAMES,
    SCHEDULED_JOBS_DRIFT_SECONDS,
    SCHEDULED_JOBS_FAILURES_TOTAL,
    SCHEDULED_JOBS_FIRES_TOTAL,
    SCHEDULED_JOBS_LABEL_SETS,
    SCHEDULED_JOBS_PROMETHEUS_NAMES,
    SCHEDULED_JOBS_TICK_DURATION_SECONDS,
    ScheduledJobsMetricsEmitter,
    get_scheduled_jobs_emitter,
    reset_scheduled_jobs_emitter_for_testing,
)
from threetears.scheduled_jobs.migrations import register
from threetears.scheduled_jobs.protocols import (
    DueSchedule,
    FireStore,
    ScheduleStore,
)
from threetears.scheduled_jobs.reschedule import compute_next_fire_at
from threetears.scheduled_jobs.tables import (
    job_fires_table,
    scheduled_jobs_table,
)
from threetears.scheduled_jobs.tick import (
    DispatchCallback,
    scheduled_tick_job,
)
from threetears.scheduled_jobs.types import (
    JobFireResult,
    JobTrigger,
    MissedFirePolicy,
    ScheduleFireStatus,
    ScheduleType,
)

__all__ = [
    "DEFAULT_DISPATCH_REAP_AFTER_SECONDS",
    "DEFAULT_JOB_CONFIG",
    "DEFAULT_TICK_DUE_LIMIT",
    "DEFAULT_TICK_LOCK_KEY",
    "EVENT_FIRE_DISPATCHED",
    "EVENT_FIRE_DRIFT",
    "EVENT_FIRE_FAILED",
    "EVENT_FIRE_SKIPPED_BUSY",
    "EVENT_TICK_COMPLETED",
    "EVENT_TICK_STARTED",
    "FORBIDDEN_LABEL_NAMES",
    "SCHEDULED_JOBS_DRIFT_SECONDS",
    "SCHEDULED_JOBS_FAILURES_TOTAL",
    "SCHEDULED_JOBS_FIRES_TOTAL",
    "SCHEDULED_JOBS_LABEL_SETS",
    "SCHEDULED_JOBS_PROMETHEUS_NAMES",
    "SCHEDULED_JOBS_TICK_DURATION_SECONDS",
    "DispatchCallback",
    "DueSchedule",
    "FireStore",
    "JobConfig",
    "JobFireCollection",
    "JobFireEntity",
    "JobFireResult",
    "JobTrigger",
    "MissedFirePolicy",
    "ScheduleFireStatus",
    "ScheduleStore",
    "ScheduleType",
    "ScheduledJobCollection",
    "ScheduledJobEntity",
    "ScheduledJobsMetricsEmitter",
    "compute_next_fire_at",
    "get_scheduled_jobs_emitter",
    "job_fires_table",
    "register",
    "reset_scheduled_jobs_emitter_for_testing",
    "scheduled_jobs_table",
    "scheduled_tick_job",
]
