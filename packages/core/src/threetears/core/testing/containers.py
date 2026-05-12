"""bare testcontainer + connectivity primitives every test repo can pull.

NOT a pytest plugin -- this module exposes plain functions every
caller can wrap into their own fixture shape. the canonical fixture
shapes built on top of these primitives live in
:mod:`threetears.core.testing.fixtures` (registered via
``pytest_plugins``).

separation of concerns:

- :func:`check_docker_available` (memoised) is the docker-daemon
  ping. callers gate their fixtures on this and call
  :func:`pytest.skip` when the daemon is unreachable so a fresh
  checkout without docker installed does not hard-fail every test.
- :func:`nats_reachable` is the equivalent for "is the long-running
  NATS at ``localhost:4222`` (or wherever) up". consumers running
  against the devx-compose stack gate on this; consumers spinning
  their own NATS testcontainer ignore it.
- :func:`skip_without_docker_marker` /
  :func:`skip_without_nats_marker` build ``pytest.mark.skipif``
  marks the test author can apply at file or class level.

all heavy imports (``testcontainers``, ``docker``, ``nats``) happen
inside the function bodies so this module stays cheap to import from
non-test code paths.
"""

from __future__ import annotations

from typing import Any

import pytest

__all__ = [
    "check_docker_available",
    "nats_reachable",
    "skip_without_docker_marker",
    "skip_without_nats_marker",
]


_DOCKER_AVAILABLE: bool | None = None
_NATS_REACHABLE: dict[str, bool] = {}


def check_docker_available() -> bool:
    """check whether docker daemon is reachable for testcontainer use.

    memoised so repeated probes do not re-ping the daemon. an
    unreachable daemon is the dominant failure mode on a fresh
    developer checkout (docker desktop not running) and we want
    the check to be cheap when N integration tests all gate on it.

    :return: True when docker is reachable, False otherwise
    :rtype: bool
    """
    global _DOCKER_AVAILABLE  # noqa: PLW0603
    result = _DOCKER_AVAILABLE
    if result is None:
        try:
            import docker  # noqa: PLC0415

            client = docker.from_env()  # type: ignore[attr-defined]
            client.ping()
            result = True
        except Exception:
            result = False
        _DOCKER_AVAILABLE = result
    return result


def nats_reachable(
    *,
    host: str = "localhost",
    port: int = 4222,
    timeout_seconds: float = 0.25,
) -> bool:
    """probe whether a NATS server is accepting TCP connections at host:port.

    memoised per (host, port) pair so a test session that runs
    dozens of NATS-backed tests pays the connect cost once. the
    timeout is intentionally short because a "no NATS" verdict is
    the common case on a fresh checkout and we don't want every
    skipped test to wait a full second.

    used by tests that target the long-running devx NATS at
    ``nats://localhost:4222``. tests that spin their own NATS
    testcontainer should gate on :func:`check_docker_available`
    instead -- the container will provide its own URI.

    :param host: NATS host to probe
    :ptype host: str
    :param port: NATS port to probe
    :ptype port: int
    :param timeout_seconds: per-probe TCP connect timeout
    :ptype timeout_seconds: float
    :return: True when a TCP socket can connect to host:port
    :rtype: bool
    """
    import socket  # noqa: PLC0415

    cache_key = f"{host}:{port}"
    if cache_key in _NATS_REACHABLE:
        return _NATS_REACHABLE[cache_key]

    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            verdict = True
    except OSError:
        verdict = False
    _NATS_REACHABLE[cache_key] = verdict
    return verdict


def skip_without_docker_marker() -> Any:
    """build a ``pytest.mark.skipif`` that fires when docker is unreachable.

    wrap a class or test function with this when the test body
    ASSUMES docker is available (e.g., directly constructs a
    testcontainer outside the canonical fixture path). the canonical
    fixtures in :mod:`threetears.core.testing.fixtures` already
    do their own ``pytest.skip`` calls; this marker is for the
    handful of tests that build containers ad-hoc.

    :return: pytest mark
    :rtype: Any
    """
    return pytest.mark.skipif(
        not check_docker_available(),
        reason="Docker not available",
    )


def skip_without_nats_marker(
    *,
    host: str = "localhost",
    port: int = 4222,
) -> Any:
    """build a ``pytest.mark.skipif`` that fires when no NATS is at host:port.

    use at file-level (``pytestmark = skip_without_nats_marker()``)
    or class-level for tests that hit the long-running devx NATS
    rather than a per-test container. matches the gold-standard
    pattern in :mod:`threetears.agent.tools.tests.integration
    .test_tool_server_nats`.

    :param host: NATS host to probe
    :ptype host: str
    :param port: NATS port to probe
    :ptype port: int
    :return: pytest mark
    :rtype: Any
    """
    return pytest.mark.skipif(
        not nats_reachable(host=host, port=port),
        reason=f"NATS not reachable at {host}:{port}",
    )
