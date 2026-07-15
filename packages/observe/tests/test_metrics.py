"""Tests for threetears.observe.metrics."""

from __future__ import annotations

import pytest

from threetears.observe.metrics import (
    _check_prometheus,
    _sanitize_metric_name,
    counter,
    gauge,
    histogram,
    metered,
)


@pytest.fixture(autouse=True)
def _reset_prometheus_check():
    """Reset the prometheus_client availability cache between tests."""
    import threetears.observe.metrics as mod

    original = mod._prometheus_available
    yield
    mod._prometheus_available = original


class TestPrometheusCheck:
    """prometheus_client availability detection."""

    def test_prometheus_available_when_installed(self):
        import threetears.observe.metrics as mod

        mod._prometheus_available = None
        result = _check_prometheus()
        # prometheus-client is a dev dependency so should be available
        assert result is True

    def test_prometheus_cached_after_first_check(self):
        import threetears.observe.metrics as mod

        mod._prometheus_available = None
        _check_prometheus()
        assert mod._prometheus_available is not None
        cached = mod._prometheus_available
        _check_prometheus()
        assert mod._prometheus_available is cached


class TestSanitizeMetricName:
    """metric name sanitization."""

    def test_dots_replaced(self):
        assert _sanitize_metric_name("my.module.func") == "my_module_func"

    def test_angle_brackets_replaced(self):
        assert _sanitize_metric_name("my.module.<locals>.func") == "my_module__locals__func"

    def test_already_clean_name_unchanged(self):
        assert _sanitize_metric_name("already_clean") == "already_clean"


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

    def test_accessor_passthrough_without_prometheus(self):
        import threetears.observe.metrics as mod

        mod._prometheus_available = False

        c = counter("test.accessor.passthrough")
        h = histogram("test.accessor.passthrough.histogram")
        g = gauge("test.accessor.passthrough.gauge")

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

    def test_metered_passthrough_without_prometheus(self):
        import threetears.observe.metrics as mod

        mod._prometheus_available = False

        @metered(name="test.passthrough")
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    async def test_metered_async_passthrough_without_prometheus(self):
        import threetears.observe.metrics as mod

        mod._prometheus_available = False

        @metered(name="test.async.passthrough")
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
