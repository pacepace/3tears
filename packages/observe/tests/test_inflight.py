"""tests for ``threetears.observe.inflight.InflightRequestsGauge``.

the leak-safe bracket is the whole point of the primitive: the gauge
must return to its baseline after a request whether that request
succeeds OR raises. these tests pin both paths plus the rendering surface
the shared HealthServer's ``/metrics`` route depends on.
"""

from __future__ import annotations

import pytest

from prometheus_client import CollectorRegistry

from threetears.observe.inflight import InflightRequestsGauge


class TestTrackBracket:
    """``track()`` increments on entry and decrements on exit."""

    def test_value_starts_at_zero(self) -> None:
        """a fresh gauge reads ``0`` before any request."""
        gauge = InflightRequestsGauge("test_inflight_requests")
        assert gauge.value == 0.0

    def test_increments_inside_bracket(self) -> None:
        """the gauge reads ``1`` while a request is in flight."""
        gauge = InflightRequestsGauge("test_inflight_requests")
        with gauge.track():
            assert gauge.value == 1.0

    def test_returns_to_baseline_on_success(self) -> None:
        """after a clean request the gauge is back to ``0``."""
        gauge = InflightRequestsGauge("test_inflight_requests")
        with gauge.track():
            pass
        assert gauge.value == 0.0

    def test_returns_to_baseline_on_exception(self) -> None:
        """LEAK-SAFETY: an exception inside the bracket still decrements.

        this is the bug the primitive exists to prevent -- a failing
        request that strands the gauge above baseline would make KEDA
        over-scale the Deployment forever.
        """
        gauge = InflightRequestsGauge("test_inflight_requests")
        with pytest.raises(RuntimeError, match="boom"), gauge.track():
            assert gauge.value == 1.0
            raise RuntimeError("boom")
        assert gauge.value == 0.0

    def test_nested_concurrent_requests_stack(self) -> None:
        """two overlapping requests read ``2`` then unwind to ``0``.

        models the queue-group pod handling multiple concurrent calls:
        each ``track()`` is independent and the counts add.
        """
        gauge = InflightRequestsGauge("test_inflight_requests")
        with gauge.track():
            with gauge.track():
                assert gauge.value == 2.0
            assert gauge.value == 1.0
        assert gauge.value == 0.0


class TestRegistryWiring:
    """the gauge registers on the supplied registry (gateway / hub path)."""

    def test_registers_on_supplied_registry(self) -> None:
        """a caller-owned registry exposes the gauge sample by name."""
        registry = CollectorRegistry()
        gauge = InflightRequestsGauge("test_inflight_requests", registry=registry)
        with gauge.track():
            assert registry.get_sample_value("test_inflight_requests") == 1.0

    def test_metric_name_property(self) -> None:
        """the primitive echoes back its metric name."""
        gauge = InflightRequestsGauge("gateway_inflight_requests")
        assert gauge.metric_name == "gateway_inflight_requests"

    def test_available_when_prometheus_installed(self) -> None:
        """with ``prometheus_client`` present the gauge is live."""
        gauge = InflightRequestsGauge("test_inflight_requests")
        assert gauge.available is True


class TestRender:
    """``render()`` produces prometheus text exposition for /metrics."""

    def test_render_returns_content_type_and_body(self) -> None:
        """render yields the prometheus content-type and a bytes body."""
        gauge = InflightRequestsGauge("test_inflight_requests")
        content_type, body = gauge.render()
        assert "text/plain" in content_type
        assert isinstance(body, bytes)

    def test_render_includes_metric_name_and_value(self) -> None:
        """the exposition names the metric and carries its current value."""
        gauge = InflightRequestsGauge("test_inflight_requests")
        with gauge.track():
            _content_type, body = gauge.render()
        text = body.decode("utf-8")
        assert "test_inflight_requests" in text
