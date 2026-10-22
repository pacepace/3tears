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
from dataclasses import dataclass
from typing import Any

import pytest

from threetears.observe import BuildOnce, contain, get_logger

log = get_logger(__name__)

__all__ = [
    "CONTAINER_STAGGER_ENV",
    "ContainerStartStagger",
    "check_docker_available",
    "nats_reachable",
    "skip_without_docker_marker",
    "skip_without_nats_marker",
    "stagger_container_start",
]


#: NATS probe verdicts, memoised through ``BuildOnce`` so concurrent first callers probe once, each
#: ``host:port`` under its own key.
_PROBES: BuildOnce[str, bool] = BuildOnce()


@dataclass(frozen=True, slots=True)
class _DockerVerdict:
    """what the one docker probe of this process found: an answer, or the error it raised.

    :param available: whether docker is reachable, when the probe answered
    :ptype available: bool
    :param error: what the probe raised instead of answering, or ``None``
    :ptype error: Exception | None
    """

    available: bool
    error: Exception | None = None


#: the docker verdict, memoised through ``BuildOnce`` so concurrent first callers probe once. A
#: failed probe is memoised too: every later check raises the same error rather than probing again.
_DOCKER_VERDICT: BuildOnce[str, _DockerVerdict] = BuildOnce()

#: the :data:`_DOCKER_VERDICT` key the docker verdict is held under.
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

    ``False`` -- which skips every docker-backed test -- is the answer only for docker that is
    genuinely not there: the docker SDK is not installed, the daemon's socket does not exist, or
    nothing is listening on it. any other probe failure (a socket this user may not open, a
    ``DOCKER_HOST`` that does not resolve, a daemon that answers with an error) raises, so a
    broken docker fails the tests that need it instead of skipping them and leaving CI green.

    a probe that raised is memoised like one that answered: every later call raises that same
    error, so one broken docker is one failure per check, not a fresh probe each time.

    :return: True when docker is reachable, False when docker is not installed or not running
    :rtype: bool
    :raises Exception: the probe's own error, for any failure other than docker being absent
    """
    verdict = _DOCKER_VERDICT.get(_DOCKER, _probe_docker_once)
    if verdict.error is not None:
        raise verdict.error
    return verdict.available


def _probe_docker_once() -> _DockerVerdict:
    """run :func:`_ping_docker`, keeping what it raised as the verdict rather than letting it escape.

    :return: the answer, or the error to raise from every :func:`check_docker_available`
    :rtype: _DockerVerdict
    """
    try:
        verdict = _DockerVerdict(available=_ping_docker())
    except Exception as exc:  # noqa: BLE001 -- not absorbed: every check_docker_available() raises it
        contain(
            exc,
            log=log,
            what="docker probe failed with something other than docker being absent; every docker check raises it",
            level="error",
        )
        verdict = _DockerVerdict(available=False, error=exc)
    return verdict


def _docker_is_absent(exc: BaseException) -> bool:
    """whether a probe failure says docker is not running, rather than that it is broken.

    the docker SDK wraps the socket error under a ``DockerException`` / ``requests`` error, so the
    whole cause chain is read: no socket file (:class:`FileNotFoundError`) or nothing listening
    (:class:`ConnectionRefusedError`) is absence; a :class:`PermissionError` anywhere in it is a
    docker that runs and this user may not reach.

    :param exc: the probe's error
    :ptype exc: BaseException
    :return: True when the error means the daemon is not there
    :rtype: bool
    """
    causes: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in causes:
        causes.append(current)
        current = current.__cause__ or current.__context__
    if any(isinstance(cause, PermissionError) for cause in causes):
        return False
    return any(isinstance(cause, FileNotFoundError | ConnectionRefusedError) for cause in causes)


def _ping_docker() -> bool:
    """ping the docker daemon once.

    :return: True when docker is reachable, False when it is not installed or not running
    :rtype: bool
    :raises Exception: any other probe failure; see :func:`check_docker_available`
    """
    try:
        import docker  # noqa: PLC0415
    except ImportError:
        log.info("the docker SDK is not installed; treating docker as unavailable")
        return False
    try:
        client = docker.from_env()  # type: ignore[attr-defined]
        client.ping()
    except (docker.errors.DockerException, OSError) as exc:
        # requests' errors are OSErrors; the SDK wraps them in DockerException when it reads the version
        if not _docker_is_absent(exc):
            raise
        log.info("docker daemon is not running; treating docker as unavailable: %s", exc)
        return False
    return True


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

    the probe runs when this is CALLED, which for a decorator is when its module is collected: a
    docker that is installed but broken (see :func:`check_docker_available`) raises here, so the
    module fails to collect and the run reports it as an error, rather than its tests skipping.

    :return: pytest mark
    :rtype: Any
    :raises Exception: the docker probe's error, for any failure other than docker being absent
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
