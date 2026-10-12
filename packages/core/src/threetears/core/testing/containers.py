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
- :func:`stagger_container_start` spreads the first container start
  of each xdist worker out in time. every shared container fixture
  calls it; a fixture that starts its own container should too.

all heavy imports (``testcontainers``, ``docker``, ``nats``) happen
inside the function bodies so this module stays cheap to import from
non-test code paths.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from typing import Any

import pytest

from threetears.observe import BuildOnce

__all__ = [
    "CONTAINER_STAGGER_ENV",
    "ContainerStartStagger",
    "check_docker_available",
    "nats_reachable",
    "skip_without_docker_marker",
    "skip_without_nats_marker",
    "stagger_container_start",
]


#: probe verdicts, memoised through ``BuildOnce`` so concurrent first callers probe once: docker
#: under :data:`_DOCKER`, each NATS ``host:port`` under its own (a colon never appears in the former).
_PROBES: BuildOnce[str, bool] = BuildOnce()

#: the :data:`_PROBES` key the docker verdict is held under.
_DOCKER = "docker"

#: seconds between xdist workers' first container starts; ``0`` disables the stagger.
CONTAINER_STAGGER_ENV = "THREETEARS_TEST_CONTAINER_STAGGER_SECONDS"
_DEFAULT_STAGGER_SECONDS = 2.0


class ContainerStartStagger:
    """the once-per-process wait an xdist worker takes before its first container start.

    several workers starting their first container in the same instant is a burst the Docker
    daemon does not always survive -- on ZFS-backed storage it leaves half-created containers
    (``zfs destroy ... dataset does not exist``) and the session fixture errors. worker ``gwN``
    therefore waits ``N * stagger`` seconds before its first start, once per instance: ``gw0``
    never waits, a worker that never starts a container pays nothing, and without xdist nothing
    changes. the stagger is ``THREETEARS_TEST_CONTAINER_STAGGER_SECONDS`` (default 2.0; ``0``
    disables it).

    the process holds one instance, which :func:`stagger_container_start` uses; a separate
    instance is a separate "first start", which is what a test of the rule needs.
    """

    def __init__(self) -> None:
        """start with no wait taken."""
        self._waited = False

    def wait_before_first_start(self, *, sleep: Callable[[float], None] = time.sleep) -> None:
        """wait this worker's share of the stagger, the first time only.

        :param sleep: the blocking sleep
        :ptype sleep: Callable[[float], None]
        :return: None
        :rtype: None
        :raises ValueError: the stagger is not a finite, non-negative number
        """
        worker = os.environ.get("PYTEST_XDIST_WORKER", "")
        if self._waited or not worker.startswith("gw") or not worker[2:].isdigit():
            return
        raw = os.environ.get(CONTAINER_STAGGER_ENV, str(_DEFAULT_STAGGER_SECONDS))
        try:
            stagger = float(raw)
        except ValueError:
            stagger = -1.0
        if not math.isfinite(stagger) or stagger < 0:
            raise ValueError(f"{CONTAINER_STAGGER_ENV} must be a finite, non-negative number of seconds, got {raw!r}")
        self._waited = True
        delay = int(worker[2:]) * stagger
        if delay > 0:
            sleep(delay)


#: the process's stagger: a worker waits before its FIRST container only.
_PROCESS_STAGGER = ContainerStartStagger()


def stagger_container_start() -> None:
    """delay this xdist worker's first container start by its index times the stagger.

    once per process, through the process's :class:`ContainerStartStagger`; see it for the rule.

    :return: None
    :rtype: None
    :raises ValueError: the stagger is not a finite, non-negative number
    """
    _PROCESS_STAGGER.wait_before_first_start()


def check_docker_available() -> bool:
    """check whether docker daemon is reachable for testcontainer use.

    memoised so repeated probes do not re-ping the daemon. an
    unreachable daemon is the dominant failure mode on a fresh
    developer checkout (docker desktop not running) and we want
    the check to be cheap when N integration tests all gate on it.

    :return: True when docker is reachable, False otherwise
    :rtype: bool
    """
    return _PROBES.get(_DOCKER, _ping_docker)


def _ping_docker() -> bool:
    """ping the docker daemon once.

    :return: True when docker is reachable, False otherwise
    :rtype: bool
    """
    try:
        import docker  # noqa: PLC0415

        client = docker.from_env()  # type: ignore[attr-defined]
        client.ping()
        result = True
    except Exception:
        result = False
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

    def _connect() -> bool:
        """
        opens and closes one TCP connection to host:port.

        :return: True when it connected
        :rtype: bool
        """
        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                verdict = True
        except OSError:
            verdict = False
        return verdict

    return _PROBES.get(f"{host}:{port}", _connect)


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
