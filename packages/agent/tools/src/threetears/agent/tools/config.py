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

__all__ = [
    "get_connect_retry_backoff_cap",
    "get_connect_retry_budget",
    "get_engagement_scope_request_timeout",
    "get_jwks_request_timeout",
    "get_object_resolve_request_timeout",
    "get_ready_poll_interval",
    "get_ready_timeout",
    "get_serve_ready_timeout",
]

log = get_logger(__name__)

_PLATFORM_DEFAULT_READY_TIMEOUT = 10.0
_PLATFORM_DEFAULT_READY_POLL_INTERVAL = 0.05
_PLATFORM_DEFAULT_SERVE_READY_TIMEOUT = 30.0
_PLATFORM_DEFAULT_JWKS_REQUEST_TIMEOUT = 5.0
_PLATFORM_DEFAULT_OBJECT_RESOLVE_REQUEST_TIMEOUT = 5.0
_PLATFORM_DEFAULT_ENGAGEMENT_SCOPE_REQUEST_TIMEOUT = 5.0
# a standalone tool pod that opens its OWN connection retries the initial connect for this long before
# giving up (fail-visible crash -> k8s CrashLoopBackoff). generous by design: k8s starts pods in any
# order, so the hub (auth-callout + the pod's seeded tool_pods row) may not be up yet -- the pod must
# wait it out rather than depend on start ordering.
_PLATFORM_DEFAULT_CONNECT_RETRY_BUDGET = 180.0
_PLATFORM_DEFAULT_CONNECT_RETRY_BACKOFF_CAP = 15.0


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


def get_jwks_request_timeout() -> float:
    """return the Hub JWKS fetch request/reply timeout in seconds.

    :return: timeout from THREETEARS_TOOLSERVER_JWKS_REQUEST_TIMEOUT or platform default
    :rtype: float
    """
    return _env_float(
        "THREETEARS_TOOLSERVER_JWKS_REQUEST_TIMEOUT",
        _PLATFORM_DEFAULT_JWKS_REQUEST_TIMEOUT,
    )


def get_object_resolve_request_timeout() -> float:
    """return the Hub object-resolve request/reply timeout in seconds.

    :return: timeout from THREETEARS_TOOLSERVER_OBJECT_RESOLVE_REQUEST_TIMEOUT or platform default
    :rtype: float
    """
    return _env_float(
        "THREETEARS_TOOLSERVER_OBJECT_RESOLVE_REQUEST_TIMEOUT",
        _PLATFORM_DEFAULT_OBJECT_RESOLVE_REQUEST_TIMEOUT,
    )


def get_engagement_scope_request_timeout() -> float:
    """return the Hub engagement-scope request/reply timeout in seconds.

    :return: timeout from THREETEARS_TOOLSERVER_ENGAGEMENT_SCOPE_REQUEST_TIMEOUT or platform default
    :rtype: float
    """
    return _env_float(
        "THREETEARS_TOOLSERVER_ENGAGEMENT_SCOPE_REQUEST_TIMEOUT",
        _PLATFORM_DEFAULT_ENGAGEMENT_SCOPE_REQUEST_TIMEOUT,
    )


def get_connect_retry_budget() -> float:
    """return how long a standalone tool pod retries its INITIAL NATS connect before failing loud.

    :return: seconds from THREETEARS_TOOL_POD_CONNECT_RETRY_SECONDS or the platform default
    :rtype: float
    """
    return _env_float(
        "THREETEARS_TOOL_POD_CONNECT_RETRY_SECONDS",
        _PLATFORM_DEFAULT_CONNECT_RETRY_BUDGET,
    )


def get_connect_retry_backoff_cap() -> float:
    """return the max backoff (seconds) between initial-connect retry attempts.

    :return: seconds from THREETEARS_TOOL_POD_CONNECT_RETRY_BACKOFF_CAP or the platform default
    :rtype: float
    """
    return _env_float(
        "THREETEARS_TOOL_POD_CONNECT_RETRY_BACKOFF_CAP",
        _PLATFORM_DEFAULT_CONNECT_RETRY_BACKOFF_CAP,
    )
