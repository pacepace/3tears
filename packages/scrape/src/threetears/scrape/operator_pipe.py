"""This pod's display, served over the bus for as long as this pod owns the session.

A display lives in one pod, and a tool pod has no inbound network path: no Service, no Ingress,
nothing dials in. So a process terminating the operator's WebSocket somewhere else cannot connect
to the display, and has no way of being given an address that would work. The bytes travel the
other way instead -- out of the pod, over the bus it already holds open -- and this module is the
end of that stream which sits beside the display.

**Which deployment reaches a display this way, and which does not.** Two routes to one display
exist and they belong to different arrangements rather than being alternatives within one:

- **The operator's socket and the display share a pod.** The MIT container beside the AGPL sidecar
  terminates the WebSocket, and :func:`threetears.scrape.operator.relay_stream` reaches ``x11vnc``
  by connecting to the pod's loopback, which is what it does when no transport is supplied. The
  compose file in this repository is exactly this deployment. Nothing in this module runs.
- **The socket is terminated where no route into the pod exists.** A hub holding the only Ingress
  is that case, and it is the deployed one. The pod enters :func:`serve_display`, and the process
  holding the operator's socket passes ``relay_stream`` a transport that attaches to the pipe
  instead of opening a TCP connection.

The loopback hop to ``x11vnc`` exists either way. What changes is which process makes it: the one
the operator is connected to, or this one.

**The claim is what entitles this pod to serve, exactly as it entitles the control subject.**
:func:`threetears.scrape.operator_control.serve_session` and this pair with the same claim and
refuse on the same terms::

    async with claim_session(lease, session_id) as claim, serve_session(
        nats, claim, display_api, session_state_key=key, owned_node=node
    ), serve_display(nats, claim, owned_node=node, pod_id=pod_id, display=where):
        await claim.until_lost()

``node`` is the value the pod's registration reply named
(:attr:`threetears.agent.tools.server.ToolServer.owned_namespaces`), and BOTH halves take it:
the control plane and the display stream ride separate families derived from the same node, so
naming it on one and not the other leaves the other on a subject granted to nobody.

A pod holding no claim never puts its display on a stream, and a claim lost under a live stream
ends it. The second half is not a check written here: the relay already takes a stop signal for
precisely this, so losing the claim ends a display stream by the same mechanism that ends the
co-located one.

**The stream ends cleanly rather than with an error when a handover takes the session**, because
that is what it is: the display is still there and another pod is serving it now. The operator's
client sees end of stream and reconnects, which is the path the operator page already has copy
for, and the reconnect lands on whichever pod holds the claim.

**Letting go of what this pod was holding is not this module's job.** ``serve_session`` closes the
local display when a handover takes the session away, and doing it here as well would mean two
teardowns racing each other over one sidecar.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from threetears.nats import (
    DEFAULT_CREDIT_WINDOW_BYTES,
    DEFAULT_MAX_CHUNK_BYTES,
    Subjects,
    serve_pipe,
)
from threetears.observe import get_logger

from .operator import relay_stream

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from collections.abc import AsyncIterator

    from threetears.nats import NatsClient, PipeStream

    from .operator import DisplayEndpoint, DisplayTransport
    from .operator_session import SessionClaim

__all__ = ["serve_display"]

log = get_logger(__name__)


class _OperatorWentAway(Exception):
    """The far end half-closed its half of the pipe, which is how a session ordinarily ends.

    Exists because :func:`threetears.scrape.operator.relay_stream` reads "the other end is
    finished" from a pump RAISING, while :meth:`threetears.nats.PipeStream.receive` reports it by
    returning ``None``. Translating one into the other is the whole of this type's job, and it is
    named in ``benign`` so the relay treats it as the ordinary end rather than a fault.
    """


@asynccontextmanager
async def serve_display(
    nats: NatsClient,
    claim: SessionClaim,
    *,
    owned_node: str,
    pod_id: str,
    display: DisplayEndpoint,
    transport: DisplayTransport | None = None,
    credit: int = DEFAULT_CREDIT_WINDOW_BYTES,
    max_chunk: int = DEFAULT_MAX_CHUNK_BYTES,
) -> AsyncIterator[None]:
    """Serve this session's display over the byte pipe while this pod owns it.

    Each attach opens its own connection to the display and bridges it to that stream, so a
    reconnecting operator gets a fresh connection rather than inheriting a half-used one, and the
    connection is released when the stream ends.

    **A pod that has lost the claim refuses an attach rather than serving it.** Dropping the
    subscription is prompt but not instantaneous, so an attach can arrive at a pod that has
    stopped being the owner in between, and serving it would put a second live view on a display
    the new owner is driving. The refusal reaches the caller as
    :class:`threetears.nats.PipeRemoteError`, which is what tells it to attach again and find the
    pod that does hold the session.

    :param nats: the platform's connected client
    :ptype nats: NatsClient
    :param claim: this pod's live claim on the session, which decides both the key served and
        whether an attach may be answered
    :ptype claim: SessionClaim
    :param owned_node: the tool-name NODE this pod owns -- ``pentest``, or the canonical
        ``tools.pentest`` its registration reply carried back. It is the ONE input the subject
        family and the stream subjects are derived from, so there is no second identifier that
        can disagree with what this pod is granted. NOT a registered tool namespace name: those
        are minted at registration, while the grants are minted at connect from the NODES on the
        pod's ``tool_pods`` row, so a leaf-keyed family names a subject no grant covers -- and an
        ungranted subscription receives nothing rather than raising.
    :ptype owned_node: str
    :param pod_id: this pod's identifier, which leads the stream subjects and must be the one this
        pod's publish grant names
    :ptype pod_id: str
    :param display: resolves the session to the host and port of its RFB server, which inside a
        pod is ``x11vnc`` on loopback. Resolved per attach rather than once, so a display that
        moved is found again on the next one.
    :ptype display: DisplayEndpoint
    :param transport: opens the reader and writer the bytes move between, defaulting to a plain TCP
        connection to what *display* resolved. Present for the same reason it is present on
        :func:`threetears.scrape.operator.relay_stream`, and passed straight through to it.
    :ptype transport: DisplayTransport | None
    :param credit: unacknowledged bytes an operator may be behind by before this pod stops reading
        the display. The ceiling this pod will agree to; a caller asking for less gets less.
    :ptype credit: int
    :param max_chunk: largest body a single frame carries, lowered to *credit* if it exceeds it
    :ptype max_chunk: int
    :return: an async iterator yielding while the display is served
    :rtype: AsyncIterator[None]
    :raises RuntimeError: if the claim has already been lost, since serving would be showing a
        display this pod does not own
    """
    if not claim.held:
        raise RuntimeError(f"session {claim.session_id!r} is not held by this pod, so its display is not ours to serve")

    async def _bridge(stream: PipeStream) -> None:
        """Carry one attached stream's bytes to and from this pod's display."""
        if not claim.held:
            # Raised rather than logged and ignored, for the reason `serve_session` raises on a
            # control message that arrives too late: the caller has to learn that the pod it
            # reached is not the owner, or it waits on a stream nobody is feeding.
            raise RuntimeError(f"session {claim.session_id!r} moved to another pod before this stream started")
        host, port = await display(claim.session_id)
        log.info(
            "operator: bridging this pod's display onto the pipe",
            extra={
                "extra_data": {
                    "session_id": claim.session_id,
                    "owned_node": owned_node,
                    "pod_id": pod_id,
                    "nonce": stream.endpoint.nonce,
                }
            },
        )

        async def _from_operator() -> bytes:
            """Read the operator's next bytes, translating end of stream into a benign raise."""
            chunk = await stream.receive()
            if chunk is None:
                raise _OperatorWentAway(f"session {claim.session_id!r}: the operator's end half-closed")
            return chunk

        await relay_stream(
            stream.send,
            _from_operator,
            host,
            port,
            transport=transport,
            until=claim.until_lost,
            benign=(_OperatorWentAway,),
        )

    async with serve_pipe(
        nats,
        claim.session_id,
        tool=owned_node,
        pod_id=pod_id,
        handler=_bridge,
        # Derived here rather than taken as a parameter, so a pod cannot serve under a family that
        # names a node other than the one it owns. The caller derives the same family from the
        # same node, which is what makes the two ends meet at all.
        family=Subjects.hitl_pipe_family(owned_node),
        credit=credit,
        max_chunk=max_chunk,
    ):
        yield
