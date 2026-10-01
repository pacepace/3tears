"""Fixtures shared by more than one sidecar suite.

The ``x11vnc`` stubs live here because two suites need a display that genuinely comes up: the
lifecycle suite, which is about the process, and the render-path suite, which opens a real HITL
session over HTTP so the session's own teardown is what drives the render-path healer.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import hitl
import pytest

#: Well away from anything a developer machine runs, since the stub binds it for real -- and not
#: 5900, which a Mac's own Screen Sharing holds.
RFB_TEST_PORT = 55901


@dataclass(frozen=True)
class X11vncStub:
    """A fake ``x11vnc`` on PATH that records every launch and binds the port its argv names."""

    #: Directory holding the stub, at the front of PATH.
    directory: Path

    @property
    def launch_log(self) -> Path:
        """One JSON line per launch: the argv the stub was started with."""
        return self.directory / "x11vnc.launches"

    def launches(self) -> list[list[str]]:
        """Every argv the stub has been launched with, oldest first.

        :return: one argv list per launch
        :rtype: list[list[str]]
        """
        if not self.launch_log.exists():
            return []
        return [json.loads(line) for line in self.launch_log.read_text().splitlines() if line]


def _write_stub(directory: Path, *, listens: bool = True, exit_code: int = 0) -> X11vncStub:
    """Install a fake ``x11vnc`` in *directory* that binds whichever port its argv names.

    Parses the port out of the real argv shape the lifecycle builds -- ``-rfbport N`` -- so the
    stub only listens if the production code actually passed a port where it claims to. Every
    launch appends its argv to the stub's launch log before anything else, so a test can count
    spawns and read the real arguments without reaching into the lifecycle.

    :param directory: where to write the stub
    :ptype directory: Path
    :param listens: when False, exits immediately without binding, which is the
        started-then-died failure the port wait exists to catch
    :ptype listens: bool
    :param exit_code: the exit status of a stub that does not listen
    :ptype exit_code: int
    :return: a handle on the installed stub
    :rtype: X11vncStub
    """
    stub = X11vncStub(directory=directory)
    body = f"""#!{sys.executable}
import json, socket, sys, time
argv = sys.argv[1:]
with open({str(stub.launch_log)!r}, "a") as launches:
    launches.write(json.dumps(argv) + "\\n")
if not {listens}:
    sys.exit({exit_code})
port = None
for i, a in enumerate(argv):
    if a == "-rfbport" and i + 1 < len(argv):
        port = int(argv[i + 1])
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
    path = directory / "x11vnc"
    path.write_text(body)
    path.chmod(0o755)
    return stub


@pytest.fixture()
def x11vnc_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> X11vncStub:
    """Put a working ``x11vnc`` stub at the front of PATH, on a free RFB port."""
    monkeypatch.setattr(hitl, "_RFB_PORT", RFB_TEST_PORT)
    stub = _write_stub(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=":")
    return stub


@pytest.fixture()
def dead_x11vnc_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> X11vncStub:
    """Put an ``x11vnc`` stub on PATH that exits 1 without ever listening."""
    monkeypatch.setattr(hitl, "_RFB_PORT", RFB_TEST_PORT)
    monkeypatch.setattr(hitl, "_START_TIMEOUT_SECONDS", 1.0)
    stub = _write_stub(tmp_path, listens=False, exit_code=1)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=":")
    return stub
