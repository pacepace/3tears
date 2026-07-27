"""An in-memory NATS bus, so the control-message tests route through the real primitives.

``serve_owner`` and ``forward`` are what make a control message find the pod holding a display,
and the way they can go wrong is by disagreeing about the subject a session maps to. Stubbing
either of them out would remove the only thing worth testing: both ends deriving the same
subject from the same session id, independently, through ``Subjects.forward``.

So the fake sits underneath both. It carries a payload from a request to a subscriber and the
reply back, and knows nothing about sessions, tabs or ownership.

Queue-group delivery is modelled as "exactly one subscriber", which is the property
``serve_owner`` relies on during the window where two pods briefly believe they own a key.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from nats.errors import NoRespondersError
from threetears.nats.errors import RequestError
from threetears.nats.transport import IncomingMessage
from threetears.scrape.operator_control import SessionDisplay


@dataclass
class _Registration:
    """One subscriber, and the queue group it shares with the key's other owners."""

    subject: str
    cb: Any
    queue: str | None


class FakeBus:
    """Just enough NATS for ``serve_owner`` and ``forward`` to meet over.

    Deliberately narrow: a message goes to one subscriber and its reply comes back, which is the
    whole of what owner-routed forwarding needs from a broker.
    """

    def __init__(self) -> None:
        self.registrations: list[_Registration] = []
        #: Every payload that reached the bus, for asserting on what did and did not travel.
        self.sent: list[bytes] = []
        self._replies: dict[str, asyncio.Future[bytes]] = {}
        self._inbox = 0

    async def subscribe(
        self,
        *,
        subject: Any,
        cb: Any,
        queue: str | None = None,
        deadletter_on_failure: bool = True,
        **_: Any,
    ) -> _Registration:
        """Register a subscriber and hand back the token used to drop it."""
        del deadletter_on_failure
        registration = _Registration(subject=subject.path, cb=cb, queue=queue)
        self.registrations.append(registration)
        return registration

    async def unsubscribe(self, sub: _Registration) -> None:
        """Drop a subscriber. Dropping one twice is not an error."""
        if sub in self.registrations:
            self.registrations.remove(sub)

    async def flush(self, timeout: float = 2.0) -> None:
        """Nothing is buffered here, so registration is already visible."""
        del timeout

    async def publish_raw_reply(self, *, reply_subject: str, payload: bytes) -> None:
        """Complete the request waiting on *reply_subject*."""
        pending = self._replies.get(reply_subject)
        if pending is not None and not pending.done():
            pending.set_result(payload)

    async def request_raw(self, *, subject: Any, payload: bytes, timeout: timedelta) -> bytes:
        """Deliver to exactly one subscriber on *subject* and wait for its reply.

        No subscriber raises the same shape the real client does -- a ``RequestError`` whose
        cause is ``NoRespondersError`` -- because ``forward`` reads the CAUSE's type to tell "no
        live owner" apart from a transport fault, and a fake that raised a bare error would let
        that branch pass untested.
        """
        self.sent.append(payload)
        matched = [r for r in self.registrations if r.subject == subject.path]
        if not matched:
            raise RequestError(f"no responders for {subject.path}") from NoRespondersError()
        self._inbox += 1
        inbox = f"_INBOX.fake.{self._inbox}"
        reply: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self._replies[inbox] = reply
        message = IncomingMessage(data=payload, reply_subject=inbox, subject=subject.path)
        # One subscriber, never all of them: distinct owners of one key share a queue group, and
        # the whole point of that group is that an overlap dispatches to exactly one.
        await matched[0].cb(message)
        return await asyncio.wait_for(reply, timeout=timeout.total_seconds())


@dataclass
class RecordedTab:
    """One tab a fake display was asked to open."""

    target_id: str
    url: str
    nav_steps: list[dict[str, Any]] | None
    session_state: dict[str, Any] | None


class RecordingDisplay(SessionDisplay):
    """A stand-in for the pod's display, recording what it was asked to do.

    Declared by subclassing the protocol rather than by marker comment, so mypy checks it still
    satisfies ``SessionDisplay`` and a method added to the protocol fails here rather than at
    the first test that happens to call it.

    Not named ``Fake...``: it is not standing in for a production implementation, because there
    is none in this package. Driving the display is the AGPL sidecar's job and reaching it is
    the platform's, so this is the reference behaviour the protocol describes.
    """

    #: What ``complete_tab`` hands back as the human's raw work. A live cookie jar in
    #: production, and the thing the control surface must never let onto the bus.
    raw_state: dict[str, Any] | None = None

    def __init__(self, raw_state: dict[str, Any] | None = None) -> None:
        self.raw_state = raw_state
        self.opened: list[RecordedTab] = []
        self.completed: list[str] = []
        self.closed = 0
        self.reads = 0

    async def open_tab(
        self,
        *,
        target_id: str,
        url: str,
        nav_steps: list[dict[str, Any]] | None,
        session_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Record the request and answer as the sidecar's tab endpoint does."""
        self.opened.append(RecordedTab(target_id=target_id, url=url, nav_steps=nav_steps, session_state=session_state))
        return {"tab_id": f"tab-{len(self.opened)}", "target_id": target_id, "url": url, "free_slots": 3}

    async def complete_tab(self, *, tab_id: str) -> dict[str, Any]:
        """Answer as the sidecar does, including the RAW export under ``session_state``."""
        self.completed.append(tab_id)
        return {
            "tab_id": tab_id,
            "target_id": "target-1",
            "free_slots": 4,
            "session_state": self.raw_state,
        }

    async def close_session(self) -> dict[str, Any]:
        """Record the teardown."""
        self.closed += 1
        return {"closed": True}

    async def read_state(self) -> dict[str, Any]:
        """Report slots and tabs."""
        self.reads += 1
        return {"free_slots": 4, "max_slots": 4, "tabs": []}
