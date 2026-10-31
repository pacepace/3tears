"""``CallMetrics``: every finished call counted, with its duration, on the registry the caller shares."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, generate_latest

from threetears.observe import CallMetrics, InflightRequestsGauge


def test_a_call_is_counted_with_its_duration_under_its_name_and_outcome() -> None:
    """Flips if a finished call leaves no count or no duration under the labels it was observed with."""
    registry = CollectorRegistry()
    calls = CallMetrics("demo", registry=registry)
    calls.observe(name="a", outcome="success", seconds=0.2)
    calls.observe(name="a", outcome="success", seconds=0.3)
    calls.observe(name="a", outcome="error", seconds=1.0)
    assert registry.get_sample_value("demo_calls_total", {"tool": "a", "outcome": "success"}) == 2.0
    assert registry.get_sample_value("demo_calls_total", {"tool": "a", "outcome": "error"}) == 1.0
    assert registry.get_sample_value("demo_call_duration_seconds_sum", {"tool": "a", "outcome": "success"}) == 0.5
    assert (
        registry.get_sample_value(
            "demo_call_duration_seconds_bucket", {"tool": "a", "outcome": "success", "le": "0.25"}
        )
        == 1.0
    )


def test_the_metrics_are_served_beside_the_in_flight_gauge_they_share_a_registry_with() -> None:
    """Flips if the counter lands on a registry the pod's /metrics route does not render."""
    gauge = InflightRequestsGauge("demo_inflight_requests")
    calls = CallMetrics("demo", label="op", registry=gauge.registry)
    calls.observe(name="b", outcome="failure", seconds=0.01)
    _content_type, body = gauge.render()
    text = body.decode("utf-8")
    assert 'demo_calls_total{op="b",outcome="failure"} 1.0' in text
    assert "demo_inflight_requests 0.0" in text
    assert gauge.registry is not None
    assert generate_latest(gauge.registry) == body
