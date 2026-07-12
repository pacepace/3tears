"""registry configuration sourced from environment variables.

single source of truth for all registry timeout defaults.
components read from this config rather than hardcoding values.

hierarchy: env var -> platform default (120s for tool calls).
"""

from __future__ import annotations

import os

from threetears.observe import get_logger

__all__ = [
    "get_call_timeout",
    "get_heartbeat_check_interval",
    "get_heartbeat_max_misses",
    "get_heartbeat_timeout",
    "get_jwks_request_timeout",
    "get_mcp_timeout",
    "get_proxy_assertion_signing_key_ref",
    "get_nats_proxy_timeout_ms",
    "get_probe_timeout",
]

log = get_logger(__name__)

# platform default for tool call timeout (seconds)
_PLATFORM_DEFAULT_CALL_TIMEOUT = 120.0
# platform default for heartbeat liveness timeout (seconds)
_PLATFORM_DEFAULT_HEARTBEAT_TIMEOUT = 45.0
# platform default for heartbeat check sweep interval (seconds)
_PLATFORM_DEFAULT_HEARTBEAT_CHECK_INTERVAL = 5.0
# platform default for the number of CONSECUTIVE missed sweeps a pod
# tolerates before its endpoints are fully evicted from the catalog. a
# value of 1 restores immediate eviction on the first miss; the default
# of 3 gives a transient blip (a slow sweep, a brief GC pause, a network
# hiccup) room to recover before the pod is torn out of the catalog.
_PLATFORM_DEFAULT_HEARTBEAT_MAX_MISSES = 3
# platform default for tool pod reachability probe timeout (seconds)
_PLATFORM_DEFAULT_PROBE_TIMEOUT = 3.0
# platform default for the Hub JWKS fetch request/reply timeout (seconds)
_PLATFORM_DEFAULT_JWKS_REQUEST_TIMEOUT = 5.0


def get_call_timeout() -> float:
    """read tool call timeout from environment or return platform default.

    env var: THREETEARS_REGISTRY_CALL_TIMEOUT

    :return: call timeout in seconds
    :rtype: float
    """
    raw = os.environ.get("THREETEARS_REGISTRY_CALL_TIMEOUT")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            log.warning(
                "invalid THREETEARS_REGISTRY_CALL_TIMEOUT=%r, using default %.1f",
                raw,
                _PLATFORM_DEFAULT_CALL_TIMEOUT,
            )
    return _PLATFORM_DEFAULT_CALL_TIMEOUT


def get_heartbeat_timeout() -> float:
    """read heartbeat liveness timeout from environment or return platform default.

    env var: THREETEARS_REGISTRY_HEARTBEAT_TIMEOUT

    :return: heartbeat timeout in seconds
    :rtype: float
    """
    raw = os.environ.get("THREETEARS_REGISTRY_HEARTBEAT_TIMEOUT")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            log.warning(
                "invalid THREETEARS_REGISTRY_HEARTBEAT_TIMEOUT=%r, using default %.1f",
                raw,
                _PLATFORM_DEFAULT_HEARTBEAT_TIMEOUT,
            )
    return _PLATFORM_DEFAULT_HEARTBEAT_TIMEOUT


def get_heartbeat_check_interval() -> float:
    """read heartbeat check interval from environment or return platform default.

    env var: THREETEARS_REGISTRY_HEARTBEAT_CHECK_INTERVAL

    :return: check interval in seconds
    :rtype: float
    """
    raw = os.environ.get("THREETEARS_REGISTRY_HEARTBEAT_CHECK_INTERVAL")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            log.warning(
                "invalid THREETEARS_REGISTRY_HEARTBEAT_CHECK_INTERVAL=%r, using default %.1f",
                raw,
                _PLATFORM_DEFAULT_HEARTBEAT_CHECK_INTERVAL,
            )
    return _PLATFORM_DEFAULT_HEARTBEAT_CHECK_INTERVAL


def get_heartbeat_max_misses() -> int:
    """read heartbeat max consecutive misses from environment or return platform default.

    env var: THREETEARS_REGISTRY_HEARTBEAT_MAX_MISSES

    a pod's endpoints are quarantined (marked unavailable, kept in the
    catalog) on each missed sweep and only fully deregistered once its
    consecutive-miss count reaches this threshold, so a transient miss
    recovers on the next heartbeat instead of forcing re-registration.
    values below 1 are clamped to 1 (evict on the first miss).

    :return: max consecutive missed sweeps before full eviction
    :rtype: int
    """
    raw = os.environ.get("THREETEARS_REGISTRY_HEARTBEAT_MAX_MISSES")
    result = _PLATFORM_DEFAULT_HEARTBEAT_MAX_MISSES
    if raw is not None:
        try:
            result = int(raw)
        except ValueError:
            log.warning(
                "invalid THREETEARS_REGISTRY_HEARTBEAT_MAX_MISSES=%r, using default %d",
                raw,
                _PLATFORM_DEFAULT_HEARTBEAT_MAX_MISSES,
            )
    if result < 1:
        result = 1
    return result


def get_probe_timeout() -> float:
    """read tool pod reachability probe timeout from environment or return platform default.

    env var: THREETEARS_REGISTRY_PROBE_TIMEOUT

    :return: probe timeout in seconds
    :rtype: float
    """
    raw = os.environ.get("THREETEARS_REGISTRY_PROBE_TIMEOUT")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            log.warning(
                "invalid THREETEARS_REGISTRY_PROBE_TIMEOUT=%r, using default %.1f",
                raw,
                _PLATFORM_DEFAULT_PROBE_TIMEOUT,
            )
    return _PLATFORM_DEFAULT_PROBE_TIMEOUT


def get_mcp_timeout() -> float:
    """read MCP HTTP client timeout from environment or return platform default.

    env var: THREETEARS_MCP_TIMEOUT

    :return: MCP timeout in seconds
    :rtype: float
    """
    raw = os.environ.get("THREETEARS_MCP_TIMEOUT")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            log.warning(
                "invalid THREETEARS_MCP_TIMEOUT=%r, using default %.1f",
                raw,
                _PLATFORM_DEFAULT_CALL_TIMEOUT,
            )
    return _PLATFORM_DEFAULT_CALL_TIMEOUT


def get_nats_proxy_timeout_ms() -> int:
    """read NATS proxy query timeout from environment or return platform default.

    env var: THREETEARS_NATS_PROXY_TIMEOUT_MS

    :return: query timeout in milliseconds
    :rtype: int
    """
    raw = os.environ.get("THREETEARS_NATS_PROXY_TIMEOUT_MS")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            log.warning(
                "invalid THREETEARS_NATS_PROXY_TIMEOUT_MS=%r, using default 5000",
                raw,
            )
    return 5000


def get_proxy_assertion_signing_key_ref() -> str:
    """read the secret reference to the proxy's assertion-signing key.

    env var: ``THREETEARS_REGISTRY_PROXY_ASSERTION_SIGNING_KEY_REF`` (a ``scheme://locator``
    secret reference); defaults to ``env://THREETEARS_PROXY_ASSERTION_SIGNING_KEY`` -- the
    SAME key the Hub publishes in its JWKS. When the reference does not resolve, the proxy mints
    no assertion (the binding is inert until the key is provisioned).

    :return: the secret reference
    :rtype: str
    """
    return os.environ.get(
        "THREETEARS_REGISTRY_PROXY_ASSERTION_SIGNING_KEY_REF",
        "env://THREETEARS_PROXY_ASSERTION_SIGNING_KEY",
    )


def get_jwks_request_timeout() -> float:
    """read the Hub JWKS fetch request timeout from environment or return platform default.

    env var: THREETEARS_REGISTRY_JWKS_REQUEST_TIMEOUT

    :return: JWKS fetch request/reply timeout in seconds
    :rtype: float
    """
    raw = os.environ.get("THREETEARS_REGISTRY_JWKS_REQUEST_TIMEOUT")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            log.warning(
                "invalid THREETEARS_REGISTRY_JWKS_REQUEST_TIMEOUT=%r, using default %.1f",
                raw,
                _PLATFORM_DEFAULT_JWKS_REQUEST_TIMEOUT,
            )
    return _PLATFORM_DEFAULT_JWKS_REQUEST_TIMEOUT
