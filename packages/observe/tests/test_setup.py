"""Tests for threetears.observe.setup."""

from __future__ import annotations

from threetears.observe.setup import TelemetryConfig, reset_telemetry


class TestTelemetryConfig:
    """TelemetryConfig dataclass defaults and construction."""

    def test_defaults(self):
        config = TelemetryConfig()
        assert config.enabled is False
        assert config.endpoint == "http://localhost:4317"
        assert config.service_name == "threetears"
        assert config.service_version == "0.1.0"
        assert config.sample_rate == 1.0
        assert config.export_timeout_seconds == 10
        assert config.loki_endpoint is None
        assert len(config.suppressed_env_vars) > 0

    def test_custom_config(self):
        config = TelemetryConfig(
            enabled=True,
            endpoint="http://tempo:4317",
            service_name="myapp",
            service_version="2.0.0",
            sample_rate=0.5,
            loki_endpoint="loki:3100",
        )
        assert config.enabled is True
        assert config.service_name == "myapp"
        assert config.sample_rate == 0.5
        assert config.loki_endpoint == "loki:3100"

    def test_frozen(self):
        config = TelemetryConfig()
        import dataclasses

        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            config.enabled = True  # type: ignore[misc]


class TestInitDisabled:
    """init_telemetry when disabled."""

    def test_init_returns_false_when_disabled(self):
        from threetears.observe.setup import init_telemetry

        config = TelemetryConfig(enabled=False)
        assert init_telemetry(config) is False


class TestResetTelemetry:
    """reset_telemetry for test isolation."""

    def test_reset_is_safe_when_not_initialized(self):
        # Should not raise even when nothing was initialized
        reset_telemetry()


class TestCallSiteEnrichingHandler:
    """_CallSiteEnrichingHandler enriches log records for OTel export."""

    def test_enriches_pathname(self):
        import logging

        from threetears.observe.setup import _CallSiteEnrichingHandler

        inner = logging.Handler()
        records: list[logging.LogRecord] = []
        inner.emit = lambda r: records.append(r)  # type: ignore[assignment]

        handler = _CallSiteEnrichingHandler(inner)

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="/full/path/module.py",
            lineno=10, msg="hello", args=(), exc_info=None,
        )
        record.call_site_file = "module.py"  # type: ignore[attr-defined]
        record.call_site_class = "MyClass"  # type: ignore[attr-defined]
        record.call_site_func = "handle"  # type: ignore[attr-defined]
        record.call_site_line = 42  # type: ignore[attr-defined]

        handler.emit(record)

        assert len(records) == 1
        assert records[0].pathname == "module.py"
        assert records[0].funcName == "MyClass.handle"
        assert records[0].lineno == 42

    def test_enriches_without_class(self):
        import logging

        from threetears.observe.setup import _CallSiteEnrichingHandler

        inner = logging.Handler()
        records: list[logging.LogRecord] = []
        inner.emit = lambda r: records.append(r)  # type: ignore[assignment]

        handler = _CallSiteEnrichingHandler(inner)

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="/full/path/module.py",
            lineno=10, msg="hello", args=(), exc_info=None,
        )
        record.call_site_func = "my_function"  # type: ignore[attr-defined]

        handler.emit(record)

        assert records[0].funcName == "my_function"
