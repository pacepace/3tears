"""Every socket connect refused for an offline test, each one kept with the thread and the frames that made it.

A replay-only run makes no connection, and a test proves it by replacing ``socket.socket.connect`` (and
``connect_ex``) and asserting no connect was tried. The replacement is the process's, so it also sees a connect the
run never made, from a thread an earlier test left running. It cannot tell the two apart by thread: work the run
submits may run on a pool worker, or a loop thread, that was started before the guard went in. So every connect
counts against the test, and each is recorded with what is needed to find its maker: the host it was resolved from,
the thread's name, where that thread came from, and the whole stack that asked for it. The record is also written to
stderr as the connect is refused, in full. A pool run keeps neither the stderr nor more than the head and tail of a
failed assertion's values, so a record's ``repr`` leads with the host and the frames that made the connect, innermost
first and past the socket and HTTP client's own: what survives the shortening names the maker. A test asserts none
was made with :func:`assert_no_connects`, which fails with every record whole, frames and all: pytest shortens an
``== []`` comparison's values, never a failure's own message.
"""

from __future__ import annotations

import socket
import sys
import sysconfig
import threading
import traceback
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Final, NoReturn

import pytest

__all__ = ["Connect", "assert_no_connects", "refuse_connects"]

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
    :ivar host: The name the thread last resolved before the connect (``socket.getaddrinfo``), or ``None``.
    :ivar thread: The name of the thread that made it.
    :ivar origin: Where that thread came from: :data:`GUARD_THREAD`, :data:`EARLIER_THREAD` or :data:`LATER_THREAD`.
    :ivar stack: Every frame that led to the connect, outermost first, each ``path:line in function``.
    """

    address: object
    host: str | None
    thread: str
    origin: str
    stack: tuple[str, ...]

    def __str__(self) -> str:
        frames = "\n".join(f"    {frame}" for frame in self.stack)
        return (
            f"a connect to {self.host or '?'} {self.address} from thread {self.thread!r} ({self.origin}), made at:\n"
            f"{frames}"
        )

    def __repr__(self) -> str:
        inner = [frame for frame in reversed(self.stack) if not frame.startswith(_LIBRARIES)]
        via = [
            frame for frame in reversed(self.stack) if frame.startswith(_LIBRARIES) and not frame.startswith(_TRANSPORT)
        ]
        return (
            f"Connect({self.host or '?'} {self.address} on {self.thread!r}, made by {inner[:6]!r}, via {via[-4:]!r}, "
            f"{self.origin})"
        )


#: Frames of installed libraries and the standard library: what a record's maker is called through, never the maker.
_LIBRARIES: Final = tuple({sysconfig.get_paths()["purelib"], sysconfig.get_paths()["stdlib"]})

#: Frames of the socket and HTTP client libraries a connect goes through, skipped where a record names its maker.
_TRANSPORT: Final = tuple(
    f"{root}/{name}"
    for root in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["stdlib"]}
    for name in ("socket.py", "ssl.py", "httpcore/", "httpx/", "urllib3/", "requests/", "http/client.py")
)


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
    resolved = threading.local()
    getaddrinfo = socket.getaddrinfo

    def resolve(host: object, *args: object, **kwargs: object) -> object:
        resolved.host = host if isinstance(host, str) else repr(host)
        return getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]

    def refuse(address: object) -> NoReturn:
        thread = threading.current_thread()
        origin = GUARD_THREAD if thread is here else EARLIER_THREAD if thread in before else LATER_THREAD
        made = Connect(
            address=address, host=getattr(resolved, "host", None), thread=thread.name, origin=origin, stack=_stack()
        )
        attempts.append(made)
        sys.stderr.write(f"refused: {run} tried {made}\n")
        raise AssertionError(f"{run} tried {made}")

    def connect(_self: socket.socket, address: object) -> NoReturn:
        refuse(address)

    def connect_ex(_self: socket.socket, address: object) -> NoReturn:
        refuse(address)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    return attempts


def assert_no_connects(attempts: Sequence[Connect]) -> None:
    """Fail the test when any connect was tried, with every record whole as the failure's message.

    pytest shortens the values of a failed ``attempts == []`` to their head and tail, which cuts a record's frames;
    a failure's own message is kept as written, so the frames that made each connect reach the log.

    :param attempts: The connects :func:`refuse_connects` kept.
    :ptype attempts: Sequence[Connect]
    :return: nothing
    :rtype: None
    """
    if attempts:
        records = "\n\n".join(str(made) for made in attempts)
        pytest.fail(f"{len(attempts)} connect(s) tried while every connect was refused:\n\n{records}")
