"""The docker probe skips docker-backed tests only when docker is genuinely absent.

``check_docker_available`` read ANY probe error as "no docker", so a docker that was installed and
running but broken -- a socket this user may not open, a ``DOCKER_HOST`` that does not resolve, a
daemon answering with an error -- skipped every docker-backed test and left CI green. Absent (no
socket, nothing listening) still skips; everything else now raises.

Each case runs the probe in a fresh interpreter, because its verdict is memoised for the process
and this process may already hold one; ``DOCKER_HOST`` points the probe at the condition under test.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("docker")

_PROBE = """
from threetears.core.testing.containers import check_docker_available
try:
    print("verdict", check_docker_available())
except Exception as exc:
    print("raised", type(exc).__name__)
"""


def _probe(docker_host: str) -> str:
    """the probe's answer for ``docker_host``: ``verdict True|False`` or ``raised <type>``."""
    env = dict(os.environ, DOCKER_HOST=docker_host)
    env.pop("DOCKER_TLS_VERIFY", None)
    done = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1]


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """a directory with a path short enough for a unix socket (108 bytes)."""
    with tempfile.TemporaryDirectory(prefix="dp-", dir="/tmp") as path:
        yield Path(path)


@pytest.fixture
def erroring_daemon() -> Iterator[int]:
    """a TCP port where something answers every request with ``500``."""
    server = socket.create_server(("127.0.0.1", 0))
    stop = threading.Event()

    def _serve() -> None:
        server.settimeout(0.1)
        while not stop.is_set():
            try:
                conn, _addr = server.accept()
            except TimeoutError:
                continue
            with conn:
                conn.recv(65536)
                conn.sendall(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        yield server.getsockname()[1]
    finally:
        stop.set()
        thread.join(timeout=5)
        server.close()


def test_no_socket_reads_as_no_docker(short_dir: Path) -> None:
    assert _probe(f"unix://{short_dir / 'missing.sock'}") == "verdict False"


def test_nothing_listening_reads_as_no_docker() -> None:
    with socket.create_server(("127.0.0.1", 0)) as reserved:
        port = reserved.getsockname()[1]
    assert _probe(f"tcp://127.0.0.1:{port}") == "verdict False"


@pytest.mark.skipif(os.geteuid() == 0, reason="root opens a mode-000 socket, so the refusal cannot be made")
def test_a_socket_this_user_may_not_open_raises(short_dir: Path) -> None:
    path = short_dir / "docker.sock"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(path))
        listener.listen()
        path.chmod(0)
        assert _probe(f"unix://{path}") == "raised DockerException"


def test_a_daemon_that_answers_with_an_error_raises(erroring_daemon: int) -> None:
    assert _probe(f"tcp://127.0.0.1:{erroring_daemon}") == "raised DockerException"


def test_a_docker_host_that_does_not_resolve_raises() -> None:
    assert _probe("tcp://no-such-docker-host.invalid:2375") == "raised DockerException"


_PROBE_TWICE = """
import os
from threetears.core.testing.containers import check_docker_available
outcomes = []
for host in ("tcp://no-such-docker-host.invalid:2375", "unix:///nonexistent/docker.sock"):
    os.environ["DOCKER_HOST"] = host
    try:
        outcomes.append(check_docker_available())
    except Exception as exc:
        outcomes.append(exc)
first, second = outcomes
print("same-error" if isinstance(first, Exception) and first is second else f"reprobed {outcomes!r}")
"""


def test_a_failed_probe_is_memoised_and_raised_again_rather_than_probing_again() -> None:
    """the second check would answer False against the missing socket if it probed again."""
    done = subprocess.run([sys.executable, "-c", _PROBE_TWICE], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "same-error"


def test_the_skip_marker_raises_at_collection_on_a_broken_docker(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a module decorated with the marker fails to collect, so a broken docker is an error, not skipped tests."""
    pytester.makepyfile(
        test_needs_docker="""
        from threetears.core.testing.containers import skip_without_docker_marker

        @skip_without_docker_marker()
        def test_uses_docker():
            pass
        """
    )
    monkeypatch.setenv("DOCKER_HOST", "tcp://no-such-docker-host.invalid:2375")
    # the workspace's entry-point plugins are not what is under test, and loading them is slow
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*ERROR collecting*test_needs_docker.py*"])
