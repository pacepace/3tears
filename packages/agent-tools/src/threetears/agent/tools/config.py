"""agent-tools config layer -- sanctioned home for platform default timeouts.

keeping every timeout default in this module satisfies the project's
enforcement test (``tests/enforcement/test_no_hardcoded_timeouts.py``)
which allowlists only explicit config layers as valid sources of
numeric timeout literals. runtime callers invoke the ``get_*`` helpers
to read values; tests and operators can override via the listed
environment variables.
"""

from __future__ import annotations

import os

from threetears.observe import get_logger

log = get_logger(__name__)

_PLATFORM_DEFAULT_READY_TIMEOUT = 10.0
_PLATFORM_DEFAULT_READY_POLL_INTERVAL = 0.05
_PLATFORM_DEFAULT_SERVE_READY_TIMEOUT = 30.0


def _env_float(name: str, fallback: float) -> float:
    """read an env var as a float with a fallback default.

    :param name: environment variable name
    :ptype name: str
    :param fallback: returned when the variable is missing or malformed
    :ptype fallback: float
    :return: resolved float
    :rtype: float
    """
    raw = os.environ.get(name)
    if raw is None:
        result = fallback
    else:
        try:
            result = float(raw)
        except ValueError:
            log.warning("invalid %s=%r, using default", name, raw)
            result = fallback
    return result


def get_ready_timeout() -> float:
    """return ToolServer readiness wait timeout in seconds.

    :return: timeout from THREETEARS_TOOLSERVER_READY_TIMEOUT or platform default
    :rtype: float
    """
    return _env_float(
        "THREETEARS_TOOLSERVER_READY_TIMEOUT",
        _PLATFORM_DEFAULT_READY_TIMEOUT,
    )


def get_ready_poll_interval() -> float:
    """return ToolServer readiness discovery poll interval in seconds.

    :return: interval from THREETEARS_TOOLSERVER_READY_POLL_INTERVAL or platform default
    :rtype: float
    """
    return _env_float(
        "THREETEARS_TOOLSERVER_READY_POLL_INTERVAL",
        _PLATFORM_DEFAULT_READY_POLL_INTERVAL,
    )


def get_serve_ready_timeout() -> float:
    """return ToolServer.wait_ready() default timeout in seconds.

    :return: timeout from THREETEARS_TOOLSERVER_SERVE_READY_TIMEOUT or platform default
    :rtype: float
    """
    return _env_float(
        "THREETEARS_TOOLSERVER_SERVE_READY_TIMEOUT",
        _PLATFORM_DEFAULT_SERVE_READY_TIMEOUT,
    )
