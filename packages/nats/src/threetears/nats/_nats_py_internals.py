"""the one owner of every nats-py private attribute this package touches.

nats-py has no public way to do four things :class:`threetears.nats.NatsClient` needs, and each
was reached for where it was needed until the private surface was spread across ``client.py``
with no owner. This module is now the only place in ``threetears.nats`` that reads or writes a
nats-py name with a leading underscore. ``client.py`` calls the functions below, never the
attributes, so a nats-py release that renames one breaks here, by name, and in the test that
checks this module's declared surface (``tests/unit/test_nats_py_internals.py``) -- not as an
``AttributeError`` that a health probe catches and turns into a fleet-wide restart loop.

**Verified against nats-py 2.15.0 and 2.16.0** (``nats/aio/client.py``,
``nats/aio/subscription.py`` and ``nats/js/client.py`` read in both; every attribute below is
unchanged between them). ``packages/nats/pyproject.toml`` caps nats-py below the first release
not verified here. Moving the cap: read the same three files in the new release, check every
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
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from nats.aio.client import PING_PROTO, Client
from nats.aio.subscription import Subscription
from nats.js.client import JetStreamContext

if TYPE_CHECKING:
    from nats.aio.msg import Msg

__all__ = [
    "PRIVATE_SURFACE",
    "PrivateAttribute",
    "force_reconnect",
    "missing_private_attributes",
    "pull_subscription_inbox",
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
)


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
    transport = connection._transport  # noqa: SLF001 -- see the module docstring
    flush_queue = connection._flush_queue  # noqa: SLF001 -- see the module docstring
    if transport is None or flush_queue is None:
        raise RuntimeError("cannot write to a nats-py connection that has never been opened")
    pending = connection._pending  # noqa: SLF001 -- see the module docstring
    if pending:
        transport.writelines(pending[:])
        connection._pending = []  # noqa: SLF001 -- see the module docstring
        connection._pending_data_size = 0  # noqa: SLF001 -- see the module docstring
    connection._pongs.append(pong)  # noqa: SLF001 -- see the module docstring
    transport.write(PING_PROTO)
    wake: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    try:
        flush_queue.put_nowait(wake)
    except asyncio.QueueFull:
        # NOSILENT: the flusher already holds wake-ups it has not taken, and each one drains the transport
        pass


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
    await connection._send_unsubscribe(subscription._id)  # noqa: SLF001 -- see the module docstring


def pull_subscription_inbox(psub: JetStreamContext.PullSubscription) -> Subscription:
    """the plain subscription on a pull subscription's fetch inbox.

    :param psub: a nats-py pull subscription
    :ptype psub: JetStreamContext.PullSubscription
    :return: the subscription its fetch responses arrive on
    :rtype: Subscription
    """
    inbox: Subscription = psub._sub  # noqa: SLF001 -- see the module docstring
    return inbox


def take_queued_messages(subscription: Subscription) -> list[Msg]:
    """take every message queued on ``subscription``, keeping nats-py's pending-bytes count right.

    :param subscription: a nats-py subscription with no callback consuming its queue
    :ptype subscription: Subscription
    :return: the queued messages, oldest first, status messages included
    :rtype: list[Msg]
    """
    queue = subscription._pending_queue  # noqa: SLF001 -- see the module docstring
    taken: list[Msg] = []
    while not queue.empty():
        msg = queue.get_nowait()
        queue.task_done()
        subscription._pending_size -= len(msg.data)  # noqa: SLF001 -- see the module docstring
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
    await connection._process_op_err(error)  # noqa: SLF001 -- see the module docstring


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
