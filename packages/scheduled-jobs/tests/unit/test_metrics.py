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
    get_scheduled_jobs_emitter,
    reset_scheduled_jobs_emitter_for_testing,
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


class TestConcurrentFirstUse:
    def test_emitters_first_asked_for_on_several_threads_register_once(self) -> None:
        """concurrent first callers for one registry share one emitter and raise nothing.

        the per-registry cache is checked and then filled; without a lock every thread that
        checked before the first one filled it built its own emitter, and the second one's
        registration raised ``Duplicated timeseries``. the registry registers slowly so every
        thread is certainly inside that window at once.
        """
        import threading
        import time

        prometheus_client = pytest.importorskip("prometheus_client")

        class _SlowRegistry(prometheus_client.CollectorRegistry):  # type: ignore[misc]
            """a registry that holds each registration open long enough for every thread to arrive."""

            def __init__(self) -> None:
                super().__init__()
                self.registrations = 0

            def register(self, collector: object) -> None:
                """count, wait, register.

                :param collector: the collector being registered
                :ptype collector: object
                :return: None
                :rtype: None
                """
                self.registrations += 1
                time.sleep(0.05)
                super().register(collector)

        registry = _SlowRegistry()
        threads_count = 8
        start = threading.Barrier(threads_count)
        emitters: list[object] = []
        errors: list[BaseException] = []
        record = threading.Lock()

        def first_use() -> None:
            start.wait()
            try:
                emitter = get_scheduled_jobs_emitter(registry)
            except BaseException as exc:  # noqa: BLE001 -- the assertion below reports every one
                with record:
                    errors.append(exc)
                return
            with record:
                emitters.append(emitter)

        try:
            workers = [threading.Thread(target=first_use) for _ in range(threads_count)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=30)

            assert errors == [], f"concurrent first callers raised: {errors!r}"
            assert len({id(emitter) for emitter in emitters}) == 1
            once = registry.registrations
            get_scheduled_jobs_emitter(registry)
            assert registry.registrations == once, "a later call registered again"
        finally:
            reset_scheduled_jobs_emitter_for_testing()
