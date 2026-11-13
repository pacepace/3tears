"""the one owner of every nats-py private attribute this package touches.

nats-py has no public way to do five things :class:`threetears.nats.NatsClient` needs, and each
was reached for where it was needed until the private surface was spread across ``client.py``
with no owner. This module is now the only place in ``threetears.nats`` that reads or writes a
nats-py name with a leading underscore. ``client.py`` calls the functions below, never the
attributes, so a nats-py release that renames one breaks here, by name, and in the test that
checks this module's declared surface (``tests/unit/test_nats_py_internals.py``) -- not as an
``AttributeError`` that a health probe catches and turns into a fleet-wide restart loop.

The module carries the workspace's per-file SLF001 ignore for nats-py, and every access below has
an entry with its reason in ``tests/enforcement/_underscore_exemptions.txt``; nothing else in the
workspace reads or writes them on a nats-py object, and no access carries an inline pragma.

**Verified against nats-py 2.16.0** (``nats/aio/client.py``, ``nats/aio/subscription.py``,
``nats/js/client.py`` and ``nats/nuid.py``). 2.15.0 lacks ``Client._skip_subject_validation``, so the floor is
2.16: ``packages/nats/pyproject.toml`` admits the one minor verified here (floor and cap), the minor the lock installs
and every suite runs, so no declared version goes unexercised. Moving the cap: read the same four files in the new release, and every other
nats-py module that calls ``_nuid.next()`` (``js/object_store.py``, ``micro/service.py``), check every
entry of :data:`PRIVATE_SURFACE` against them, run the unit and integration suites on it, and
add the version to this paragraph.

The surface, and why each is used:

``Client._pending``, ``Client._pending_data_size``, ``Client._transport``, ``Client._pongs``,
``Client._flush_queue``
    :func:`write_pending_then_ping`. nats-py's ``flush`` writes its ``PING`` straight to the
    socket ahead of anything still in the pending buffer, so its ``PONG`` proves nothing about a
    ``SUB`` or ``UNSUB`` sent just before; and a timed-out or cancelled ``flush`` leaves a
    cancelled future in ``_pongs`` that ends the read loop when the late ``PONG`` arrives. The
    round trip writes the pending buffer and the ``PING`` in one synchronous step, queues its own
    ``PONG`` future in that step, and wakes the flusher -- the same fields nats-py's own
    ``_flusher`` and ``_send_ping`` use, in the same order.
``Client._send_unsubscribe``, ``Subscription._id``
    :func:`send_unsubscribe`. Removes a subscription's interest at the server while it stays
    registered here, so the messages the server routed before the ``UNSUB`` are still delivered
    and handled. nats-py's ``unsubscribe`` forgets the subscription first; its ``drain`` races
    its own ``PING`` ahead of the ``UNSUB``.
``JetStreamContext.PullSubscription._sub``, ``Subscription._pending_queue``,
``Subscription._pending_size``
    :func:`pull_subscription_inbox` and :func:`take_queued_messages`. A pull subscription's
    inbox has no callback, and nats-py exposes neither the inbox subscription nor its queue. A
    stopping pull consumer takes what the server delivered to a fetch request still live there,
    keeping nats-py's pending-bytes accounting right as it does.
``Client._process_op_err``
    :func:`force_reconnect`. nats-py has no public force-reconnect; this is the entry its own
    read loop uses when a connection goes stale.
``Client._skip_subject_validation``, ``Client._max_pending_size``, ``Client._flush_pending``,
``Client._resp_sub_prefix``, ``Client._init_request_sub``, ``Client._resp_map`` (and ``_pending``,
``_pending_data_size``, ``_flush_queue``, ``_nuid`` above)
    :func:`publish_unjoined` and :func:`request_unjoined`. nats-py's ``_send_publish`` builds a
    ``PUB`` as one ``bytes``: the command line, then the payload, then CRLF, concatenated on the
    event loop (``nats/protocol/command.py``). For a payload of many megabytes that is one copy
    of all of it holding the loop (4.6-7.2 ms per 12 MB measured on a pool container). These two
    queue the command line, the payload object itself and the CRLF as three pending items, which
    the flusher hands to ``writelines`` and the socket sends from memoryviews, with no copy; the
    rest is nats-py's own publish and request, step for step: the same checks, stats, pending-size
    accounting and flush, and the same response mux, tokens and futures as ``_request_new_style``.
``Client._nuid``
    :func:`install_csprng_nuid`. nats-py names every inbox and request token from one NUID per
    connection: a random 12-character prefix, then a sequence that grows by a fixed step. One
    observed reply subject predicts that connection's next inboxes, and any client in the account
    can get bytes delivered onto a subject it can name. :class:`CsprngNuid` draws every token from
    ``secrets`` instead. Every nats-py consumer of ``_nuid`` calls only ``next()``: the request mux
    prefix and request tokens and ``new_inbox()`` (``aio/client.py``), JetStream's async-publish
    reply prefix and token, ephemeral pull-consumer names and push deliver subjects
    (``js/client.py``), object-store chunk ids (``js/object_store.py``) and a micro service's id
    (``micro/service.py``). Two of them slice ``msg.subject[len(inbox_prefix) + 22 + 2:]`` and one
    calls ``.extend()`` on the result, so ``next()`` returns a fresh, mutable ``bytearray`` of
    exactly 22 bytes.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from nats.aio.client import NO_RESPONDERS_STATUS, PING_PROTO, Client
from nats.aio.subscription import Subscription
from nats.errors import (
    BadSubjectError,
    ConnectionClosedError,
    ConnectionDrainingError,
    MaxPayloadError,
    NoRespondersError,
    OutboundBufferLimitError,
)
from nats.errors import TimeoutError as NatsTimeoutError
from nats.js.api import Header
from nats.js.client import JetStreamContext
from nats.nuid import NUID
from nats.protocol.command import PUB_OP

if TYPE_CHECKING:
    from nats.aio.msg import Msg

__all__ = [
    "NUID_LENGTH",
    "PRIVATE_SURFACE",
    "UNJOINED_FROM_BYTES",
    "CsprngNuid",
    "PrivateAttribute",
    "force_reconnect",
    "install_csprng_nuid",
    "missing_private_attributes",
    "publish_unjoined",
    "pull_subscription_inbox",
    "request_unjoined",
    "send_unsubscribe",
    "take_queued_messages",
    "write_pending_then_ping",
]


@dataclass(frozen=True)
class PrivateAttribute:
    """one nats-py private attribute this module depends on.

    :ivar owner: the nats-py class that carries it
    :ivar name: the attribute name
    :ivar kind: what it must be on a freshly constructed instance -- a type, ``"method"`` for a
        coroutine method, or ``"declared"`` for an instance attribute nats-py sets to ``None``
        until the connection opens
    """

    owner: str
    name: str
    kind: type | str


#: every nats-py private attribute this module reads, writes or calls.
#: :func:`missing_private_attributes` checks each against the installed nats-py.
PRIVATE_SURFACE: Final[tuple[PrivateAttribute, ...]] = (
    PrivateAttribute("Client", "_pending", list),
    PrivateAttribute("Client", "_pending_data_size", int),
    PrivateAttribute("Client", "_transport", "declared"),
    PrivateAttribute("Client", "_pongs", list),
    PrivateAttribute("Client", "_flush_queue", "declared"),
    PrivateAttribute("Client", "_send_unsubscribe", "method"),
    PrivateAttribute("Client", "_process_op_err", "method"),
    PrivateAttribute("Subscription", "_id", int),
    PrivateAttribute("Subscription", "_pending_queue", asyncio.Queue),
    PrivateAttribute("Subscription", "_pending_size", int),
    PrivateAttribute("PullSubscription", "_sub", Subscription),
    PrivateAttribute("Client", "_nuid", NUID),
    PrivateAttribute("Client", "_skip_subject_validation", bool),
    PrivateAttribute("Client", "_max_pending_size", int),
    PrivateAttribute("Client", "_flush_pending", "method"),
    PrivateAttribute("Client", "_resp_sub_prefix", "declared"),
    PrivateAttribute("Client", "_init_request_sub", "method"),
    PrivateAttribute("Client", "_resp_map", dict),
)

#: the payload size from which :class:`threetears.nats.NatsClient` publishes through
#: :func:`publish_unjoined` / :func:`request_unjoined` instead of nats-py's own ``publish`` and
#: ``request``. Below it nats-py's one concatenated ``bytes`` costs microseconds; from it the copy
#: grows with the payload and holds the event loop while it runs.
UNJOINED_FROM_BYTES: Final[int] = 64 * 1024

#: nats-py's publish-path subject check (``nats.aio.client._SUBJECT_INVALID_RE``): no whitespace
#: or CRLF, which would let a subject inject protocol commands.
_SUBJECT_INVALID: Final[re.Pattern[str]] = re.compile(r"[ \t\r\n]")

#: the CRLF that ends a ``PUB``'s payload.
_CRLF: Final[bytes] = b"\r\n"

#: the length nats-py slices every inbox token at (``msg.subject[len(prefix) + 22 + 2:]``).
NUID_LENGTH: Final[int] = 22

#: nats-py's NUID alphabet: base62, so a token is a valid subject token.
_NUID_DIGITS: Final[bytes] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class CsprngNuid(NUID):
    """a drop-in for nats-py's ``NUID`` whose every token is drawn from ``secrets``.

    Each token is :data:`NUID_LENGTH` base62 characters, about 131 bits, with no shared prefix and
    no step between one token and the next: seeing any number of them says nothing about another.
    It subclasses ``NUID`` so the client's attribute keeps its declared type; the prefix and
    sequence state ``NUID.__init__`` sets up is never read, because ``next`` is the only method
    nats-py calls.
    """

    def next(self) -> bytearray:
        """a fresh token, as nats-py's ``NUID.next`` returns one.

        :return: a new, mutable ``bytearray`` of :data:`NUID_LENGTH` base62 bytes
        :rtype: bytearray
        """
        return bytearray(secrets.choice(_NUID_DIGITS) for _ in range(NUID_LENGTH))


def install_csprng_nuid(connection: Client) -> None:
    """make ``connection`` name every inbox, request token and consumer from :class:`CsprngNuid`.

    Called between constructing the client and ``connect()``, before anything has drawn a token.

    :param connection: a nats-py client not yet connected
    :ptype connection: Client
    :return: nothing
    :rtype: None
    """
    connection._nuid = CsprngNuid()


def write_pending_then_ping(connection: Client, pong: asyncio.Future[bool]) -> None:
    """hand the pending buffer then a ``PING`` to the transport in one step, queuing ``pong`` for it.

    Synchronous on purpose: nothing can be written between the buffer and the ``PING``, and
    ``Client._pongs`` stays in the order the ``PING`` s were written, so every ``PONG`` resolves
    the future of its own ``PING``. The flusher is then woken without waiting, for a transport
    that sends only when drained (the websocket one).

    :param connection: a connected nats-py client
    :ptype connection: Client
    :param pong: the future the ``PONG`` for this ``PING`` resolves
    :ptype pong: asyncio.Future[bool]
    :return: nothing
    :rtype: None
    :raises RuntimeError: when the connection has no transport or flusher (never opened)
    """
    transport = connection._transport
    flush_queue = connection._flush_queue
    if transport is None or flush_queue is None:
        raise RuntimeError("cannot write to a nats-py connection that has never been opened")
    pending = connection._pending
    if pending:
        transport.writelines(pending[:])
        connection._pending = []
        connection._pending_data_size = 0
    connection._pongs.append(pong)
    transport.write(PING_PROTO)
    wake: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    try:
        flush_queue.put_nowait(wake)
    except asyncio.QueueFull:
        # NOSILENT: the flusher already holds wake-ups it has not taken, and each one drains the transport
        pass


def _check_subject(subject: str) -> None:
    """refuse a subject nats-py's publish path refuses: empty, or holding whitespace or CRLF.

    :param subject: the subject to send on
    :ptype subject: str
    :return: nothing
    :rtype: None
    :raises BadSubjectError: when the subject is malformed
    """
    if not subject or _SUBJECT_INVALID.search(subject):
        raise BadSubjectError


async def publish_unjoined(connection: Client, subject: str, payload: bytes, reply: str = "") -> None:
    """publish ``payload`` as nats-py's ``publish`` does, without copying it into one command.

    nats-py's ``publish`` then ``_send_publish`` (no headers), step for step: subject checks
    unless the connection skips them, closed / draining / reconnect-buffer / max-payload refusals,
    ``out_msgs`` / ``out_bytes`` stats, the pending-size flush past the limit and the flusher
    wake. The one difference: the ``PUB`` line, the payload object and the CRLF go into the
    pending buffer as three items, appended in one synchronous step so no other command can land
    between them, and the payload is never copied on the event loop.

    :param connection: a nats-py client
    :ptype connection: Client
    :param subject: the subject to publish on
    :ptype subject: str
    :param payload: the message data; a mutable buffer is copied first, as nats-py's
        concatenation copied it, so a later change to it cannot reach the wire
    :ptype payload: bytes
    :param reply: the reply subject, empty for none
    :ptype reply: str
    :return: nothing
    :rtype: None
    :raises BadSubjectError: for an empty subject or one holding whitespace
    :raises ConnectionClosedError: when the connection is closed
    :raises ConnectionDrainingError: when the connection is draining its publishes
    :raises OutboundBufferLimitError: while reconnecting with buffering off or the buffer full
    :raises MaxPayloadError: when the payload is over the server's limit
    """
    if not connection._skip_subject_validation:
        _check_subject(subject)
        if reply:
            _check_subject(reply)
    if connection.is_closed:
        raise ConnectionClosedError
    if connection.is_draining_pubs:
        raise ConnectionDrainingError
    data = payload if type(payload) is bytes else bytes(payload)
    payload_size = len(data)
    max_pending = connection._max_pending_size
    if not connection.is_connected and (max_pending <= 0 or payload_size + connection._pending_data_size > max_pending):
        raise OutboundBufferLimitError
    if payload_size > connection.max_payload:
        raise MaxPayloadError
    if subject == "":
        raise BadSubjectError
    head = f"{PUB_OP} {subject} {reply} {payload_size}\r\n".encode()
    connection.stats["out_msgs"] += 1
    connection.stats["out_bytes"] += payload_size
    pending = connection._pending
    pending.append(head)
    pending.append(data)
    pending.append(_CRLF)
    connection._pending_data_size += len(head) + payload_size + len(_CRLF)
    if max_pending > 0 and connection._pending_data_size > max_pending:
        await connection._flush_pending(force_flush=True)
    flush_queue = connection._flush_queue
    if flush_queue is not None and flush_queue.empty():
        await connection._flush_pending()


async def request_unjoined(connection: Client, subject: str, payload: bytes, timeout: float) -> Msg:
    """nats-py's ``request`` (new style, no headers), publishing through :func:`publish_unjoined`.

    The same response mux as nats-py's ``_request_new_style``: one wildcard inbox subscription per
    connection (made on first use), a fresh token per request, a future in the mux's map that
    removes itself when done. A ``503`` status reply is a :class:`NoRespondersError`, as there.

    :param connection: a connected nats-py client
    :ptype connection: Client
    :param subject: the subject to request on
    :ptype subject: str
    :param payload: the request data
    :ptype payload: bytes
    :param timeout: seconds to wait for the reply once the request is published
    :ptype timeout: float
    :return: the reply
    :rtype: Msg
    :raises ConnectionDrainingError: when the connection is draining its publishes
    :raises NatsTimeoutError: when no reply arrives within ``timeout``
    :raises NoRespondersError: when the server reports no subscriber for ``subject``
    """
    if not connection._skip_subject_validation:
        _check_subject(subject)
    if connection.is_draining_pubs:
        raise ConnectionDrainingError
    if not connection._resp_sub_prefix:
        await connection._init_request_sub()
    prefix = connection._resp_sub_prefix
    if prefix is None:
        raise ConnectionClosedError
    token = connection._nuid.next()
    token.extend(secrets.token_hex(2).encode())
    inbox = prefix[:]
    inbox.extend(token)
    key = token.decode()
    future: asyncio.Future[Msg] = asyncio.get_running_loop().create_future()
    future.add_done_callback(lambda _done: connection._resp_map.pop(key, None))
    connection._resp_map[key] = future
    await publish_unjoined(connection, subject, payload, reply=inbox.decode())
    try:
        msg = await asyncio.wait_for(future, timeout)
    except TimeoutError as exc:
        raise NatsTimeoutError from exc
    if msg.headers and msg.headers.get(Header.STATUS) == NO_RESPONDERS_STATUS:
        raise NoRespondersError
    return msg


async def send_unsubscribe(connection: Client, subscription: Subscription) -> None:
    """queue an ``UNSUB`` for ``subscription`` while it stays registered on ``connection``.

    :param connection: the nats-py client carrying the subscription
    :ptype connection: Client
    :param subscription: the nats-py subscription whose interest the server is to drop
    :ptype subscription: Subscription
    :return: nothing
    :rtype: None
    :raises Exception: whatever nats-py raises for a connection that cannot send
    """
    await connection._send_unsubscribe(subscription._id)


def pull_subscription_inbox(psub: JetStreamContext.PullSubscription) -> Subscription:
    """the plain subscription on a pull subscription's fetch inbox.

    :param psub: a nats-py pull subscription
    :ptype psub: JetStreamContext.PullSubscription
    :return: the subscription its fetch responses arrive on
    :rtype: Subscription
    """
    inbox: Subscription = psub._sub
    return inbox


def take_queued_messages(subscription: Subscription) -> list[Msg]:
    """take every message queued on ``subscription``, keeping nats-py's pending-bytes count right.

    :param subscription: a nats-py subscription with no callback consuming its queue
    :ptype subscription: Subscription
    :return: the queued messages, oldest first, status messages included
    :rtype: list[Msg]
    """
    queue = subscription._pending_queue
    taken: list[Msg] = []
    while not queue.empty():
        msg = queue.get_nowait()
        queue.task_done()
        subscription._pending_size -= len(msg.data)
        taken.append(msg)
    return taken


async def force_reconnect(connection: Client, error: Exception) -> None:
    """make nats-py treat ``error`` as a failure of the live connection and reconnect.

    :param connection: the nats-py client to reconnect
    :ptype connection: Client
    :param error: the error it is told it hit
    :ptype error: Exception
    :return: nothing
    :rtype: None
    """
    await connection._process_op_err(error)


def _instances() -> dict[str, object]:
    """one freshly constructed instance of each nats-py class in :data:`PRIVATE_SURFACE`.

    :return: instances by class name
    :rtype: dict[str, object]
    """
    client = Client()
    subscription = Subscription(client, id=1, subject="threetears.surface.check")
    pull = JetStreamContext.PullSubscription(JetStreamContext(client), subscription, "stream", "consumer", b"inbox")
    return {"Client": client, "Subscription": subscription, "PullSubscription": pull}


def missing_private_attributes(surface: tuple[PrivateAttribute, ...] = PRIVATE_SURFACE) -> list[str]:
    """every entry of ``surface`` the installed nats-py does not provide as declared.

    Constructs each class without connecting; nothing touches the network.

    :param surface: the attributes to check; :data:`PRIVATE_SURFACE` unless a test proves the
        check can fail
    :ptype surface: tuple[PrivateAttribute, ...]
    :return: one line per missing or changed attribute, empty when the surface is intact
    :rtype: list[str]
    """
    instances = _instances()
    missing: list[str] = []
    for attribute in surface:
        instance = instances[attribute.owner]
        label = f"{attribute.owner}.{attribute.name}"
        if not hasattr(instance, attribute.name):
            missing.append(f"{label}: absent")
            continue
        value = getattr(instance, attribute.name)
        if attribute.kind == "method":
            if not inspect.iscoroutinefunction(value):
                missing.append(f"{label}: no longer a coroutine method")
        elif attribute.kind == "declared":
            continue
        elif isinstance(attribute.kind, type) and not isinstance(value, attribute.kind):
            missing.append(f"{label}: expected {attribute.kind.__name__}, found {type(value).__name__}")
    return missing
