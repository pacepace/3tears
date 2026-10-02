"""Tests for threetears.observe.metrics."""

from __future__ import annotations

import importlib
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import patch

import pytest

from threetears.observe.metrics import (
    counter,
    gauge,
    histogram,
    metered,
)


@contextmanager
def _metrics_without_prometheus(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """``threetears.observe.metrics`` imported afresh into a process where prometheus_client is absent.

    The availability check runs once per process and remembers its answer, so the module the
    rest of this suite shares has already seen prometheus_client installed. A fresh import with
    the distribution unimportable is the state an install without it is in. ``patch.dict``
    restores ``sys.modules`` on exit and ``monkeypatch`` the package attribute the import rebinds,
    so the rest of the process never sees this copy; the copy itself stays usable after the block.

    :param monkeypatch: pytest's patcher
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the freshly imported module
    :rtype: Iterator[ModuleType]
    """
    import threetears.observe as observe_pkg

    monkeypatch.setattr(observe_pkg, "metrics", importlib.import_module("threetears.observe.metrics"))
    with patch.dict(sys.modules, {"prometheus_client": None}):
        sys.modules.pop("threetears.observe.metrics", None)
        yield importlib.import_module("threetears.observe.metrics")


def _unique(stem: str) -> str:
    """a metric name no other test registers, so each assertion reads only its own samples."""
    return f"{stem}.{uuid.uuid4().hex[:8]}"


class TestPrometheusCheck:
    """prometheus_client availability detection, observed through what an accessor records."""

    def test_prometheus_available_when_installed(self):
        from prometheus_client import REGISTRY

        # prometheus-client is a dev dependency, so an accessor records into the registry
        name = _unique("test.check.available")
        counter(name).inc()

        assert REGISTRY.get_sample_value(f"{name.replace('.', '_')}_total") == 1.0

    def test_prometheus_cached_after_first_check(self, monkeypatch: pytest.MonkeyPatch):
        """the first check decides for the process: prometheus_client arriving later does not change it."""
        from prometheus_client import REGISTRY

        with _metrics_without_prometheus(monkeypatch) as metrics:
            metrics.counter(_unique("test.check.before")).inc()  # the check runs here, absent

        name = _unique("test.check.after")
        metrics.counter(name).inc()  # prometheus_client importable again

        assert REGISTRY.get_sample_value(f"{name.replace('.', '_')}_total") is None


def _registered_as(name: str) -> str:
    """the prometheus family name a counter registered under *name* records into.

    :param name: the raw name given to :func:`counter`
    :ptype name: str
    :return: the family name whose ``_total`` sample is 1.0 after one increment
    :rtype: str
    """
    from prometheus_client import REGISTRY

    counter(name).inc()
    families = [
        family.name
        for family in REGISTRY.collect()
        if any(sample.name == f"{family.name}_total" and sample.value == 1.0 for sample in family.samples)
        and family.name.endswith(name.rsplit(".", 1)[-1].translate(str.maketrans("<>", "__")))
    ]
    assert len(families) == 1, f"expected one family for {name!r}, found {families}"
    return families[0]


class TestSanitizeMetricName:
    """metric name sanitization, observed as the name the registry records under."""

    def test_dots_replaced(self):
        suffix = uuid.uuid4().hex[:8]
        assert _registered_as(f"my.module.func{suffix}") == f"my_module_func{suffix}"

    def test_angle_brackets_replaced(self):
        suffix = uuid.uuid4().hex[:8]
        assert _registered_as(f"my.module.<locals>.func{suffix}") == f"my_module__locals__func{suffix}"

    def test_already_clean_name_unchanged(self):
        name = f"already_clean_{uuid.uuid4().hex[:8]}"
        assert _registered_as(name) == name


class TestAccessors:
    """counter/histogram/gauge get-or-create accessors."""

    def test_counter_records_arbitrary_value(self):
        from prometheus_client import REGISTRY

        c = counter("test.accessor.counter")
        c.inc()
        c.inc(2)

        assert REGISTRY.get_sample_value("test_accessor_counter_total") == 3.0

    def test_counter_with_labels(self):
        from prometheus_client import REGISTRY

        c = counter("test.accessor.labelled.counter", label_names=("outcome",))
        c.labels(outcome="win").inc()
        c.labels(outcome="loss").inc()
        c.labels(outcome="loss").inc()

        assert REGISTRY.get_sample_value("test_accessor_labelled_counter_total", {"outcome": "win"}) == 1.0
        assert REGISTRY.get_sample_value("test_accessor_labelled_counter_total", {"outcome": "loss"}) == 2.0

    def test_histogram_records_result_derived_value(self):
        """the whole point: a value only known after a function returns."""
        from prometheus_client import REGISTRY

        def process_response():
            return {"completion_rate": 0.75}

        result = process_response()
        histogram("test.accessor.completion_rate").observe(result["completion_rate"])

        assert REGISTRY.get_sample_value("test_accessor_completion_rate_sum") == 0.75

    def test_gauge_set_and_inc_dec(self):
        from prometheus_client import REGISTRY

        g = gauge("test.accessor.gauge")
        g.set(5)
        assert REGISTRY.get_sample_value("test_accessor_gauge") == 5.0

        g.inc()
        assert REGISTRY.get_sample_value("test_accessor_gauge") == 6.0

        g.dec(2)
        assert REGISTRY.get_sample_value("test_accessor_gauge") == 4.0

    def test_accessor_cached_across_calls(self):
        """a second accessor call under the same name reuses the instrument.

        prometheus_client raises ValueError on duplicate registration, so
        calling counter() twice with the same name without an exception is
        itself sufficient proof the cache works.
        """
        first = counter("test.accessor.cached")
        second = counter("test.accessor.cached")

        assert first is second

    def test_first_calls_on_several_threads_at_once_register_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """concurrent first calls for one name get one instrument and no error.

        the cache is checked and then filled; without a lock every thread that checked
        before the first one filled it constructed its own instrument, and the second
        registration raised ``Duplicated timeseries``. registration is slowed here so every
        thread is certainly inside that window at once.
        """
        import threading
        import time

        from prometheus_client import REGISTRY

        real_register = REGISTRY.register
        registrations: list[object] = []

        def slow_register(collector: object) -> None:
            registrations.append(collector)
            time.sleep(0.05)
            real_register(collector)  # type: ignore[arg-type]

        monkeypatch.setattr(REGISTRY, "register", slow_register)
        threads_count = 8
        start = threading.Barrier(threads_count)
        results: list[object] = []
        errors: list[BaseException] = []
        record = threading.Lock()

        def first_call() -> None:
            start.wait()
            try:
                instrument = counter("test.accessor.concurrent_first")
            except BaseException as exc:  # noqa: BLE001 -- the assertion below reports every one
                with record:
                    errors.append(exc)
                return
            with record:
                results.append(instrument)

        workers = [threading.Thread(target=first_call) for _ in range(threads_count)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)

        assert errors == [], f"concurrent first calls raised: {errors!r}"
        assert len({id(instrument) for instrument in results}) == 1
        assert len(registrations) == 1

    def test_different_kinds_under_the_same_name_raise(self):
        """counter/histogram/gauge each register under the exact name given;

        prometheus's registry conflicts purely on metric name regardless of
        instrument type, so requesting two different kinds under the
        identical name is a genuine caller error that must surface as
        prometheus_client's own ValueError, not be silently papered over
        by the accessor cache (which correctly does not prevent this --
        each (kind, name, label_names) key only dedupes identical requests).
        """
        counter("test.accessor.shared_name_conflict")
        with pytest.raises(ValueError, match="Duplicated timeseries"):
            histogram("test.accessor.shared_name_conflict")

    def test_accessor_passthrough_without_prometheus(self, monkeypatch: pytest.MonkeyPatch):
        with _metrics_without_prometheus(monkeypatch) as metrics:
            c = metrics.counter("test.accessor.passthrough")
            h = metrics.histogram("test.accessor.passthrough.histogram")
            g = metrics.gauge("test.accessor.passthrough.gauge")

            # every method is safe to call and does nothing
            c.inc()
            c.labels(status="x").inc()
            h.observe(1.0)
            g.set(1.0)
            g.inc()
            g.dec()


class TestMeteredDecorator:
    """@metered decorator behavior."""

    def test_metered_bare_sync(self):
        @metered
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_metered_parameterised_sync(self):
        @metered(name="test.custom.metric")
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    async def test_metered_bare_async(self):
        @metered
        async def add(a, b):
            return a + b

        assert await add(1, 2) == 3

    async def test_metered_parameterised_async(self):
        @metered(name="test.custom.async.metric")
        async def add(a, b):
            return a + b

        assert await add(1, 2) == 3

    def test_metered_preserves_function_name(self):
        @metered
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_metered_sync_exception_propagates(self):
        @metered(name="test.explode.sync")
        def explode():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            explode()

    async def test_metered_async_exception_propagates(self):
        @metered(name="test.explode.async")
        async def explode():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await explode()

    def test_metered_passthrough_without_prometheus(self, monkeypatch: pytest.MonkeyPatch):
        with _metrics_without_prometheus(monkeypatch) as metrics:

            @metrics.metered(name="test.passthrough")
            def add(a, b):
                return a + b

            assert add(1, 2) == 3

    async def test_metered_async_passthrough_without_prometheus(self, monkeypatch: pytest.MonkeyPatch):
        with _metrics_without_prometheus(monkeypatch) as metrics:

            @metrics.metered(name="test.async.passthrough")
            async def add(a, b):
                return a + b

            assert await add(1, 2) == 3


class TestMeteredRecording:
    """@metered actually records counter/histogram values.

    reads recorded values through prometheus_client's own public
    REGISTRY.get_sample_value() API (the same pattern
    InflightRequestsGauge's own tests use), not by reaching into any
    private module or instrument state.
    """

    def test_success_increments_success_counter(self):
        from prometheus_client import REGISTRY

        @metered(name="test.recording.success")
        def add(a, b):
            return a + b

        add(1, 2)
        add(3, 4)

        assert REGISTRY.get_sample_value("test_recording_success_calls_total", {"status": "success"}) == 2.0
        assert REGISTRY.get_sample_value("test_recording_success_calls_total", {"status": "error"}) is None

    def test_error_increments_error_counter_not_success(self):
        from prometheus_client import REGISTRY

        @metered(name="test.recording.error")
        def explode():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                explode()

        assert REGISTRY.get_sample_value("test_recording_error_calls_total", {"status": "error"}) == 3.0
        assert REGISTRY.get_sample_value("test_recording_error_calls_total", {"status": "success"}) is None

    def test_duration_histogram_records_observations(self):
        from prometheus_client import REGISTRY

        @metered(name="test.recording.duration")
        def add(a, b):
            return a + b

        add(1, 2)
        add(3, 4)

        assert REGISTRY.get_sample_value("test_recording_duration_duration_seconds_count") == 2.0

    def test_instruments_created_once_and_reused(self):
        """
        a second call under the same metric name must not re-register the
        instrument pair -- prometheus_client raises ValueError on duplicate
        registration, so simply calling the decorated function twice
        without an exception is itself sufficient proof the cache works,
        with no need to inspect any private instrument-cache state.
        """
        from prometheus_client import REGISTRY

        @metered(name="test.recording.reuse")
        def add(a, b):
            return a + b

        add(1, 2)
        add(3, 4)  # would raise ValueError here if not cached

        assert REGISTRY.get_sample_value("test_recording_reuse_calls_total", {"status": "success"}) == 2.0

    async def test_async_success_increments_success_counter(self):
        from prometheus_client import REGISTRY

        @metered(name="test.recording.async.success")
        async def add(a, b):
            return a + b

        await add(1, 2)

        assert REGISTRY.get_sample_value("test_recording_async_success_calls_total", {"status": "success"}) == 1.0
