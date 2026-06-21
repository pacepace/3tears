"""Unit tests for the scheduled-jobs metrics emitter + cardinality guard.

Two concerns:

- **Cardinality discipline**: no instrument declares a forbidden
  (unbounded-cardinality) label. The forbidden set is read off
  :data:`FORBIDDEN_LABEL_NAMES` so the guard stays in sync with the
  module rather than a docstring. Mirrors agent-wake's
  ``test_metrics_cardinality``.
- **Emit smoke**: the emitter registers + emits against a private
  registry without raising, and degrades to no-op cleanly.
"""

from __future__ import annotations

import pytest

from threetears.scheduled_jobs.metrics import (
    FORBIDDEN_LABEL_NAMES,
    SCHEDULED_JOBS_LABEL_SETS,
    SCHEDULED_JOBS_PROMETHEUS_NAMES,
    ScheduledJobsMetricsEmitter,
)


class TestCardinalityGuard:
    """No instrument carries a forbidden unbounded-cardinality label."""

    def test_no_forbidden_labels(self) -> None:
        for name, labels in SCHEDULED_JOBS_LABEL_SETS.items():
            for label in labels:
                assert label not in FORBIDDEN_LABEL_NAMES, f"{name} declares forbidden label {label!r}"

    def test_label_sets_cover_every_instrument(self) -> None:
        """Every published instrument name has a declared label set."""
        for name in SCHEDULED_JOBS_PROMETHEUS_NAMES:
            assert name in SCHEDULED_JOBS_LABEL_SETS

    def test_forbidden_set_covers_the_id_columns(self) -> None:
        """The forbidden set names the generic id + opaque columns."""
        assert {"partition_key", "job_id", "fire_id", "payload", "kind"} <= FORBIDDEN_LABEL_NAMES


class TestEmitterSmoke:
    """The emitter registers + emits against a private registry."""

    def test_emit_against_private_registry(self) -> None:
        pytest.importorskip("prometheus_client")
        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        emitter = ScheduledJobsMetricsEmitter(registry=registry)
        assert emitter.available is True
        # every emit path runs without raising
        emitter.inc_fire(status="succeeded", schedule_type="interval")
        emitter.inc_fire(status="failed", schedule_type="cron")
        emitter.inc_failure(reason="handler_exception")
        emitter.observe_tick_duration(0.01)
        emitter.observe_drift(3.2)
        emitter.unregister_from_registry()
        assert emitter.available is False
