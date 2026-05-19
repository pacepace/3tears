"""Smoke tests -- verify the package imports cleanly."""

from __future__ import annotations


def test_package_imports():
    """All public symbols import without error."""
    from threetears.observe import (
        ContextFormatter,
        ThreeTearsLogger,
        clear_context,
        configure_logging,
        configure_third_party_logging,
        get_context,
        get_logger,
        set_context,
        traced,
    )

    assert callable(get_logger)
    assert callable(set_context)
    assert callable(clear_context)
    assert callable(get_context)
    assert callable(configure_logging)
    assert callable(configure_third_party_logging)
    assert callable(traced)
    assert ContextFormatter is not None
    assert ThreeTearsLogger is not None


def test_setup_imports():
    """Setup module imports without requiring OTel SDK."""
    from threetears.observe.setup import (
        TelemetryConfig,
        init_telemetry,
        reset_telemetry,
        shutdown_telemetry,
    )

    assert callable(init_telemetry)
    assert callable(shutdown_telemetry)
    assert callable(reset_telemetry)
    assert TelemetryConfig is not None


def test_version():
    """Package version is set."""
    from threetears.observe import __version__

    assert __version__ == "0.8.6"
