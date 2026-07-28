"""Lifecycle tests for the on-demand VNC path.

No real ``x11vnc`` here -- it is not installed outside the container, and requiring it would
make this suite unrunnable on the machine it is written on. What IS real
is everything this module actually owns: the process spawn, the wait-for-a-listening-port, the
termination, and the idempotence. The stubs are real executables that really bind the port
they are told to, so a test that says "nothing listens after stop" has genuinely connected to
something before and genuinely failed to afterwards.

That distinction matters here more than usual. The failure this chunk must not ship is a
session that reports success and shows a black rectangle, and every way of reaching that state
runs through "we thought a process was up and it was not" -- which is precisely what a mocked
``create_subprocess_exec`` would assert away.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import hitl
import pytest
from hitl import VncLifecycle, VncUnavailable

#: Ports well away from anything a developer machine runs, since these bind for real.
_RFB_TEST_PORT = 55901


def _free_port_is_free(port: int) -> bool:
    """Whether *port* has no listener, asked the only way that cannot be wrong."""
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def _write_stub(directory: Path, name: str, *, listens: bool = True, exit_code: int = 0) -> None:
    """Install a fake *name* on PATH that binds whichever port its argv names.

    Parses the port out of the real argv shape this module builds -- ``-rfbport N`` -- so the
    stub only listens if the production code actually passed a port where it claims to.

    :param listens: when False, exits immediately without binding, which is the
        started-then-died failure the port wait exists to catch
    """
    body = f"""#!{sys.executable}
import socket, sys, time
argv = sys.argv[1:]
if not {listens}:
    sys.exit({exit_code})
port = None
for i, a in enumerate(argv):
    if a == "-rfbport" and i + 1 < len(argv):
        port = int(argv[i + 1])
        break
if port is None:
    for a in argv:
        if ":" in a and a.split(":")[-1].isdigit() and not a.startswith("-"):
            port = int(a.split(":")[-1])
            break
if port is None:
    sys.exit(3)
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port))
s.listen(5)
while True:
    time.sleep(0.05)
"""
    path = directory / name
    path.write_text(body)
    path.chmod(0o755)


@pytest.fixture()
def stub_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a working x11vnc stub at the front of PATH, on a free RFB port."""
    monkeypatch.setattr(hitl, "_RFB_PORT", _RFB_TEST_PORT)
    _write_stub(tmp_path, "x11vnc")
    monkeypatch.setenv("PATH", str(tmp_path), prepend=":")
    return tmp_path


@pytest.fixture()
async def lifecycle(stub_path: Path):
    """A lifecycle on test ports, always torn down even when the test fails."""
    del stub_path
    vnc = VncLifecycle(display_num=99)
    try:
        yield vnc
    finally:
        await vnc.stop()


async def test_nothing_listens_before_the_first_start(lifecycle: VncLifecycle) -> None:
    """The steady state of an unattended container is no VNC surface at all.

    Starting at boot would mean the display is reachable for the whole life of a container
    that, almost all of the time, has nobody looking at it.
    """
    assert not lifecycle.health()
    assert _free_port_is_free(_RFB_TEST_PORT)


async def test_start_serves_the_display_and_names_it(lifecycle: VncLifecycle) -> None:
    """The display on the RFB port is now the whole of what starting produces.

    There used to be a second process here serving a client tree and proxying the stream. Both
    jobs belong to the MIT container sharing this pod, so what remains is the one thing only
    this container can do: put its own X display on a loopback socket.
    """
    session = await lifecycle.start()

    assert lifecycle.health()
    assert not _free_port_is_free(_RFB_TEST_PORT), "x11vnc is not accepting connections"
    assert session.display == ":99"


async def test_start_is_idempotent(lifecycle: VncLifecycle) -> None:
    """ "Open a session" is the operation a human-facing queue retries.

    A second start that spawned a second ``x11vnc`` would have it lose the RFB port race and
    exit, leaving a lifecycle holding a handle to a dead process while reporting healthy.
    """
    first = await lifecycle.start()
    x11vnc_pid = lifecycle._x11vnc.pid

    second = await lifecycle.start()

    assert second == first
    assert lifecycle._x11vnc.pid == x11vnc_pid, "a second x11vnc was spawned over the running one"
    assert lifecycle.health()


async def test_nothing_survives_teardown(lifecycle: VncLifecycle) -> None:
    """A stopped session must leave the container as it was before anyone arrived."""
    await lifecycle.start()
    assert not _free_port_is_free(_RFB_TEST_PORT)

    await lifecycle.stop()

    assert not lifecycle.health()
    for _ in range(50):
        if _free_port_is_free(_RFB_TEST_PORT):
            break
        await asyncio.sleep(0.1)
    assert _free_port_is_free(_RFB_TEST_PORT), "x11vnc outlived the teardown"


async def test_stop_is_safe_to_call_twice_and_before_any_start(lifecycle: VncLifecycle) -> None:
    """Teardown is the path an error handler takes, so it must not raise a second error."""
    await lifecycle.stop()
    await lifecycle.start()
    await lifecycle.stop()
    await lifecycle.stop()
    assert not lifecycle.health()


async def test_start_after_stop_works(lifecycle: VncLifecycle) -> None:
    """A human who left should be able to come back without restarting the container."""
    await lifecycle.start()
    await lifecycle.stop()
    session = await lifecycle.start()
    assert lifecycle.health()
    assert session.display == ":99"


async def test_a_process_that_never_listens_fails_loudly_and_leaves_nothing_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Started-then-died is the failure mode that otherwise reaches a human as a black screen.

    Waiting on the PORT rather than on the spawn is what catches it; a bare sleep would call
    this a success. And the teardown on failure is what keeps ``start`` to two outcomes
    instead of leaving a half-up pair for the next caller to find.
    """
    monkeypatch.setattr(hitl, "_RFB_PORT", _RFB_TEST_PORT)
    monkeypatch.setattr(hitl, "_START_TIMEOUT_SECONDS", 1.0)
    _write_stub(tmp_path, "x11vnc", listens=False, exit_code=1)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=":")

    vnc = VncLifecycle(display_num=99)
    with pytest.raises(VncUnavailable, match="x11vnc"):
        await vnc.start()

    assert not vnc.health()
    assert _free_port_is_free(_RFB_TEST_PORT), "a failed start left x11vnc holding the RFB port"


async def test_a_missing_binary_says_the_image_lacks_vnc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The message should point at the Dockerfile, not at this module.

    ``FileNotFoundError: x11vnc`` sends whoever reads it looking for a bug in the spawn code,
    when the actual cause is an image built without the packages.
    """
    monkeypatch.setenv("PATH", str(tmp_path))
    vnc = VncLifecycle(display_num=99)
    with pytest.raises(VncUnavailable, match="built without VNC support"):
        await vnc.start()


async def test_the_display_number_is_a_parameter_not_a_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-display today, display pool later -- as configuration, not a rewrite.

    One Xvfb means one operator at a time. Concurrency needs :100, :101 and so on, each with
    its own Chromium and x11vnc, and the seam for that is the number never being hardcoded.
    """
    monkeypatch.setenv("DISPLAY_NUM", "101")
    assert VncLifecycle().display == ":101"
    assert VncLifecycle(display_num=7).display == ":7"

    monkeypatch.delenv("DISPLAY_NUM", raising=False)
    assert VncLifecycle().display == ":99", "the default stopped matching entrypoint.sh's Xvfb"


def test_x11vnc_is_bound_to_loopback_so_the_pod_is_the_boundary(stub_path: Path) -> None:
    """``-localhost`` IS the access control on this port, not a hardening extra.

    Containers in one Kubernetes pod share a network namespace, so 127.0.0.1 is reachable by
    the MIT container beside this one and by nothing else. Bound wide, the RFB port would be
    reachable by anything that can route to the container, going straight around the capability
    check in front of the relay -- and silently, because the operator's path would keep working.
    """
    del stub_path
    argv = VncLifecycle(display_num=99)._x11vnc_argv()
    assert "-localhost" in argv
    assert "-display" in argv and ":99" in argv
    assert "-nopw" in argv, "a password prompt with no password to check would stall the connection"
    assert "-xrandr" in argv and "resize" in argv, "a server-side geometry change would leave viewers on a stale size"


async def test_the_child_never_gets_an_undrained_pipe(stub_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pipe nobody reads is a 64 KiB ceiling on how long the child survives.

    x11vnc logs per connection and this session is explicitly built for reconnects
    (``-forever``, ``-shared``), so a long operator session fills the buffer and blocks x11vnc
    inside a write. The visible result is a display that stops painting, which is precisely the
    failure this module claims to prevent -- so the claim and the plumbing have to agree.
    """
    del stub_path
    captured: dict[str, object] = {}

    async def _fake_exec(*argv: str, **kwargs: object) -> object:
        captured.update(kwargs)
        raise OSError("not actually spawning")

    monkeypatch.setattr(hitl.asyncio, "create_subprocess_exec", _fake_exec)
    vnc = VncLifecycle(display_num=99)
    with pytest.raises(VncUnavailable):
        await vnc._spawn(["x11vnc"], what="x11vnc")

    assert captured.get("stderr") is not asyncio.subprocess.PIPE, (
        "stderr is a pipe nobody reads, which caps the child's life at 64 KiB of output"
    )
