"""Unit tests for the wake Prometheus emitter + WakeConfig defaults.

The emitter uses an isolated :class:`prometheus_client.CollectorRegistry`
per test so the locked instrument names are registered exactly once per
registry without leaking between tests. The pattern mirrors
:mod:`packages.models.tests.unit.models.test_tracking`.

Three concerns drive the tests:

1. The emitter instantiates cleanly against a private registry, every
   declared instrument is reachable, and ``available=True``.
2. Each emit helper writes the expected sample values + labels.
3. The :class:`WakeConfig` Protocol defaults match the PLACEMENT
   §1.9 / §3.5 numbers (regression guard if a future PR weakens the
   defaults silently).
"""

from __future__ import annotations

import pytest

from threetears.agent.wake.config import (
    DEFAULT_HTTP_ALLOWED_HOSTS,
    DEFAULT_LOKI_NAMED_QUERIES,
    DEFAULT_MAX_FIRES_PER_CONV_PER_DAY,
    DEFAULT_MAX_FIRES_PER_USER_PER_DAY,
    DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
    DEFAULT_MAX_WEBHOOK_FIRES_PER_SUBSCRIPTION_PER_HOUR,
    DEFAULT_POSTGRES_NAMED_QUERIES,
)
from threetears.agent.wake.metrics import (
    WAKE_DRIFT_SECONDS,
    WAKE_FAILURES_TOTAL,
    WAKE_FIRES_TOTAL,
    WAKE_RATE_LIMIT_REJECTIONS_TOTAL,
    WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL,
    WAKE_TICK_DURATION_SECONDS,
    WAKE_WEBHOOK_RECEIVED_TOTAL,
    WAKE_YIELD_DURATION_SECONDS,
    WakeMetricsEmitter,
)


@pytest.fixture()
def private_registry() -> object:
    """Return a fresh :class:`CollectorRegistry` per test."""
    from prometheus_client import CollectorRegistry

    return CollectorRegistry()


def test_emitter_registers_every_declared_instrument(private_registry: object) -> None:
    """Construction populates every locked instrument name.

    Probes via ``generate_latest`` so the assertion is independent of
    prometheus_client's sample-name suffix convention -- which differs
    between counters (no extra suffix because the name already ends
    in ``_total``) and histograms (``_count`` / ``_sum`` / ``_bucket``).
    """
    from prometheus_client import generate_latest

    emitter = WakeMetricsEmitter(registry=private_registry)
    try:
        assert emitter.available is True
        scraped = generate_latest(private_registry).decode("utf-8")  # type: ignore[arg-type]
        expected_names = (
            WAKE_FIRES_TOTAL,
            WAKE_FAILURES_TOTAL,
            WAKE_TICK_DURATION_SECONDS,
            WAKE_RATE_LIMIT_REJECTIONS_TOTAL,
            WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL,
            WAKE_DRIFT_SECONDS,
            WAKE_YIELD_DURATION_SECONDS,
            WAKE_WEBHOOK_RECEIVED_TOTAL,
        )
        for name in expected_names:
            assert name in scraped, f"{name} not in scraped output: {scraped}"
    finally:
        emitter.unregister_from_registry()


def test_inc_fire_records_labelled_increment(private_registry: object) -> None:
    """``inc_fire`` writes one sample with the declared label triple."""
    emitter = WakeMetricsEmitter(registry=private_registry)
    try:
        emitter.inc_fire(status="fired", schedule_type="daily_at", execution_mode="inline")
        emitter.inc_fire(status="fired", schedule_type="daily_at", execution_mode="inline")
        emitter.inc_fire(status="yielded", schedule_type="webhook", execution_mode="inline")
        fired_count = private_registry.get_sample_value(  # type: ignore[attr-defined]
            WAKE_FIRES_TOTAL,
            {"status": "fired", "schedule_type": "daily_at", "execution_mode": "inline"},
        )
        yielded_count = private_registry.get_sample_value(  # type: ignore[attr-defined]
            WAKE_FIRES_TOTAL,
            {"status": "yielded", "schedule_type": "webhook", "execution_mode": "inline"},
        )
        assert fired_count == 2.0
        assert yielded_count == 1.0
    finally:
        emitter.unregister_from_registry()


def test_inc_rate_limit_and_schedule_cap(private_registry: object) -> None:
    """``inc_rate_limit_rejection`` + ``inc_schedule_cap_rejection`` count."""
    emitter = WakeMetricsEmitter(registry=private_registry)
    try:
        emitter.inc_rate_limit_rejection(scope="conv")
        emitter.inc_rate_limit_rejection(scope="user")
        emitter.inc_schedule_cap_rejection()
        emitter.inc_schedule_cap_rejection()
        assert (
            private_registry.get_sample_value(  # type: ignore[attr-defined]
                WAKE_RATE_LIMIT_REJECTIONS_TOTAL,
                {"scope": "conv"},
            )
            == 1.0
        )
        assert (
            private_registry.get_sample_value(  # type: ignore[attr-defined]
                WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL,
            )
            == 2.0
        )
    finally:
        emitter.unregister_from_registry()


def test_observe_histograms_record_samples(private_registry: object) -> None:
    """The three histograms record observation counts and sums."""
    emitter = WakeMetricsEmitter(registry=private_registry)
    try:
        emitter.observe_tick_duration(0.5)
        emitter.observe_tick_duration(1.5)
        emitter.observe_drift(0.0)
        emitter.observe_drift(120.0)
        emitter.observe_yield_duration(2.0)

        tick_count = private_registry.get_sample_value(  # type: ignore[attr-defined]
            WAKE_TICK_DURATION_SECONDS + "_count"
        )
        tick_sum = private_registry.get_sample_value(  # type: ignore[attr-defined]
            WAKE_TICK_DURATION_SECONDS + "_sum"
        )
        drift_count = private_registry.get_sample_value(  # type: ignore[attr-defined]
            WAKE_DRIFT_SECONDS + "_count"
        )
        yield_count = private_registry.get_sample_value(  # type: ignore[attr-defined]
            WAKE_YIELD_DURATION_SECONDS + "_count"
        )

        assert tick_count == 2.0
        assert tick_sum == 2.0
        assert drift_count == 2.0
        assert yield_count == 1.0
    finally:
        emitter.unregister_from_registry()


def test_inc_webhook_received_writes_outcome(private_registry: object) -> None:
    """Webhook outcomes increment the labelled counter."""
    emitter = WakeMetricsEmitter(registry=private_registry)
    try:
        emitter.inc_webhook_received(outcome="accepted")
        emitter.inc_webhook_received(outcome="auth_failed")
        accepted = private_registry.get_sample_value(  # type: ignore[attr-defined]
            WAKE_WEBHOOK_RECEIVED_TOTAL,
            {"outcome": "accepted"},
        )
        auth_failed = private_registry.get_sample_value(  # type: ignore[attr-defined]
            WAKE_WEBHOOK_RECEIVED_TOTAL,
            {"outcome": "auth_failed"},
        )
        assert accepted == 1.0
        assert auth_failed == 1.0
    finally:
        emitter.unregister_from_registry()


def test_inc_failure_records_reason(private_registry: object) -> None:
    """``inc_failure`` increments the reason-labelled counter."""
    emitter = WakeMetricsEmitter(registry=private_registry)
    try:
        emitter.inc_failure(reason="handler_exception")
        emitter.inc_failure(reason="handler_exception")
        emitter.inc_failure(reason="cap_exceeded")
        ex = private_registry.get_sample_value(  # type: ignore[attr-defined]
            WAKE_FAILURES_TOTAL,
            {"reason": "handler_exception"},
        )
        ce = private_registry.get_sample_value(  # type: ignore[attr-defined]
            WAKE_FAILURES_TOTAL,
            {"reason": "cap_exceeded"},
        )
        assert ex == 2.0
        assert ce == 1.0
    finally:
        emitter.unregister_from_registry()


# ---------------------------------------------------------------------------
# WakeConfig defaults (PLACEMENT §1.9 / §3.5 regression guard)
# ---------------------------------------------------------------------------


def test_default_wake_caps_match_placement_lock() -> None:
    """The platform defaults match the locked values from PLACEMENT.

    Catches the failure mode where a future PR silently weakens a
    default (e.g. ``DEFAULT_MAX_SCHEDULES_PER_CONVERSATION = 100``)
    without updating PLACEMENT.
    """
    assert DEFAULT_MAX_FIRES_PER_CONV_PER_DAY == 24
    assert DEFAULT_MAX_FIRES_PER_USER_PER_DAY == 100
    assert DEFAULT_MAX_WEBHOOK_FIRES_PER_SUBSCRIPTION_PER_HOUR == 60
    assert DEFAULT_MAX_SCHEDULES_PER_CONVERSATION == 10
    assert DEFAULT_HTTP_ALLOWED_HOSTS == ()
    assert DEFAULT_LOKI_NAMED_QUERIES == {}
    assert DEFAULT_POSTGRES_NAMED_QUERIES == {}
