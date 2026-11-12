"""Every socket connect refused for an offline test, each one kept with the thread and the frames that made it.

A replay-only run makes no connection, and a test proves it by replacing ``socket.socket.connect`` (and
``connect_ex``) and asserting no connect was tried. The replacement is the process's, so it also sees a connect the
run never made, from a thread an earlier test left running. It cannot tell the two apart by thread: work the run
submits may run on a pool worker, or a loop thread, that was started before the guard went in. So every connect
counts against the test, and each is recorded with what is needed to find its maker: the thread's name, where that
thread came from, and the whole stack that asked for it. The record is also written to stderr as the connect is
refused, in full: pytest shows a failed test's captured stderr whole, where it shortens a failed assertion's values.
"""

from __future__ import annotations

import socket
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Final, NoReturn

import pytest

__all__ = ["Connect", "refuse_connects"]

#: :attr:`Connect.origin` of a connect made on the thread that put the guard in.
GUARD_THREAD: Final = "the thread that put the guard in"

#: :attr:`Connect.origin` of a connect made on a thread already running when the guard went in: one an earlier test
#: left behind, or a long-lived worker the run's own work was handed to. A lead, never an excuse.
EARLIER_THREAD: Final = "a thread already running when the guard went in"

#: :attr:`Connect.origin` of a connect made on a thread started after the guard went in.
LATER_THREAD: Final = "a thread started after the guard went in"


@dataclass(frozen=True)
class Connect:
    """One refused connect: where it went, the thread that made it, and the frames that made it.

    :ivar address: The address the socket was asked to connect to, ``(host, port)`` for an IP socket.
    :ivar thread: The name of the thread that made it.
    :ivar origin: Where that thread came from: :data:`GUARD_THREAD`, :data:`EARLIER_THREAD` or :data:`LATER_THREAD`.
    :ivar stack: Every frame that led to the connect, outermost first, each ``path:line in function``.
    """

    address: object
    thread: str
    origin: str
    stack: tuple[str, ...]

    def __str__(self) -> str:
        frames = "\n".join(f"    {frame}" for frame in self.stack)
        return f"a connect to {self.address} from thread {self.thread!r} ({self.origin}), made at:\n{frames}"


def _stack() -> tuple[str, ...]:
    """The frames that asked the socket to connect, outermost first, without the guard's own three."""
    frames = traceback.extract_stack()[:-3]
    return tuple(f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in frames)


def refuse_connects(monkeypatch: pytest.MonkeyPatch, *, run: str = "a replay-only run") -> list[Connect]:
    """Refuse every socket connect in the process until ``monkeypatch`` is undone, and keep each one.

    Every ``connect`` and ``connect_ex`` raises :class:`AssertionError`, is written to stderr, and is appended to the
    list returned, which the test asserts empty, whichever thread made it.

    :param monkeypatch: The monkeypatch whose undoing gives the process its ``connect`` back.
    :ptype monkeypatch: pytest.MonkeyPatch
    :param run: What the test runs, as the refusal names it: ``"a replay-only run"``.
    :ptype run: str
    :return: Every connect tried while the guard was in, in the order they were tried.
    :rtype: list[Connect]
    """
    here = threading.current_thread()
    before = frozenset(thread for thread in threading.enumerate() if thread is not here)
    attempts: list[Connect] = []

    def refuse(address: object) -> NoReturn:
        thread = threading.current_thread()
        origin = GUARD_THREAD if thread is here else EARLIER_THREAD if thread in before else LATER_THREAD
        made = Connect(address=address, thread=thread.name, origin=origin, stack=_stack())
        attempts.append(made)
        sys.stderr.write(f"refused: {run} tried {made}\n")
        raise AssertionError(f"{run} tried {made}")

    def connect(_self: socket.socket, address: object) -> NoReturn:
        refuse(address)

    def connect_ex(_self: socket.socket, address: object) -> NoReturn:
        refuse(address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    return attempts
