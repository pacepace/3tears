"""owner-routed request forwarding over NATS request/reply.

a **generic, payload-agnostic** primitive for "send a request to
whichever pod currently *serves* a key, and get its reply back". this is
the *messaging* half of a single-writer pattern: a separate election
mechanism (:func:`threetears.nats.nats_distributed_lock`,
:class:`threetears.core.coordination.KVLease`) decides *who* serves a
given ``key``; this module only carries a request from any caller to
that owner and the owner's reply back. the consumer ties the two
together — hold the lease, then :func:`serve_owner`; lose it, exit the
context.

design
------

- **subject derivation** — every ``key`` maps deterministically to one
  forward subject via :meth:`Subjects.forward`. the key is hashed
  (SHA-256 hex) into a subject-safe token, mirroring
  :meth:`Subjects.room`: app-supplied keys may contain ``.``, spaces,
  ``*`` or ``>`` (all illegal/ambiguous in a NATS subject token), and a
  one-way digest sidesteps every one of them while staying collision-
  resistant and deterministic.

- **owner side** — :func:`serve_owner` subscribes the key's forward
  subject in a **queue group keyed by that subject**. the queue group is
  the safety valve for the lease-handoff window: if two pods briefly
  believe they own the key (old holder not yet exited, new holder
  already serving), NATS still delivers each forwarded request to
  *exactly one* of them. the handler runs and its reply bytes (or a
  structured error frame on raise) go back on the request's reply
  inbox.

- **caller side** — :func:`forward` issues a request to the key's
  forward subject and returns the owner's reply bytes. no current owner
  (no subscriber, or the handoff window swallowed the request) surfaces
  as :class:`NoOwnerError`; a handler that *raised* surfaces as
  :class:`ForwardedHandlerError` carrying the original exception's type
  name + message, so a consumer can map a forwarded failure back onto
  its own typed exception (e.g. an op-log CAS conflict) instead of
  losing it.

wire framing
------------

the reply payload is a **1-byte tag** followed by a body:

- ``0x00`` (``_TAG_OK``) — body is the handler's raw reply ``bytes``,
  verbatim.
- ``0x01`` (``_TAG_ERR``) — body is UTF-8 JSON
  ``{"type": <exc-type-name>, "message": <str(exc)>}``.

:func:`forward` reads the tag, returns the body on ``0x00``, and raises
:class:`ForwardedHandlerError` reconstructed from the JSON on ``0x01``.
the tag keeps a handler that legitimately returns empty / arbitrary
bytes unambiguous from an error frame.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Final

from nats.errors import NoRespondersError as _NatsNoRespondersError, TimeoutError as _NatsTimeoutError

from threetears.observe import get_logger

from threetears.nats.errors import NatsClientError, RequestError
from threetears.nats.subjects import Subjects

if TYPE_CHECKING:
    from threetears.nats.client import NatsClient, Subscription
    from threetears.nats.transport import IncomingMessage

__all__ = [
    "DEFAULT_FORWARD_TIMEOUT",
    "ForwardError",
    "ForwardedHandlerError",
    "ForwardHandler",
    "NoOwnerError",
    "forward",
    "serve_owner",
]


log = get_logger(__name__)


#: default request timeout for :func:`forward` when the caller does not
#: pass one. deliberately short — the common "no owner during handoff"
#: case should surface fast so the consumer can retry / self-promote.
DEFAULT_FORWARD_TIMEOUT: Final[timedelta] = timedelta(seconds=5)

#: reply-frame tag: body is the handler's verbatim reply bytes.
_TAG_OK: Final[int] = 0x00

#: reply-frame tag: body is UTF-8 JSON {"type": ..., "message": ...}.
_TAG_ERR: Final[int] = 0x01


ForwardHandler = Callable[[bytes], Awaitable[bytes]]
"""owner-side handler shape for :func:`serve_owner`.

receives the forwarded request payload bytes and returns the reply
payload bytes. raising propagates to the caller's :func:`forward` as a
:class:`ForwardedHandlerError` carrying the exception's type name +
message.
"""


class ForwardError(NatsClientError):
    """base class for owner-routed-forward conditions.

    subclasses :class:`threetears.nats.NatsClientError` so a consumer
    catching the whole wrapper surface still catches these, while a
    consumer that cares about routing can catch this narrower base.
    """


class NoOwnerError(ForwardError):
    """raised by :func:`forward` when no pod currently serves the key.

    covers both "no subscriber at all" (NATS reports no responders) and
    "the request timed out inside the lease-handoff window" — from the
    caller's perspective both mean *there is no live owner right now*.
    distinct from a generic transport failure (which surfaces as
    :class:`threetears.nats.RequestError`), so a consumer can branch on
    it to retry or attempt to self-promote.
    """


class ForwardedHandlerError(ForwardError):
    """raised by :func:`forward` when the owner's handler raised.

    carries the original exception's **type name** and **message** so a
    consumer can map a forwarded failure back onto its own typed
    exception (e.g. reconstruct an op-log CAS conflict). the original
    exception object itself does not cross the wire — only its type name
    and ``str()`` do.

    :param type_name: ``type(exc).__name__`` from the owner side
    :ptype type_name: str
    :param message: ``str(exc)`` from the owner side
    :ptype message: str
    """

    __slots__ = ("type_name", "message")

    def __init__(self, type_name: str, message: str) -> None:
        self.type_name = type_name
        self.message = message
        super().__init__(f"forwarded handler raised {type_name}: {message}")


def _encode_ok(payload: bytes) -> bytes:
    """frame a successful handler reply: ``0x00`` + verbatim bytes.

    :param payload: handler's reply bytes
    :ptype payload: bytes
    :return: tagged reply frame
    :rtype: bytes
    """
    return bytes([_TAG_OK]) + payload


def _encode_err(exc: BaseException) -> bytes:
    """frame a handler exception: ``0x01`` + UTF-8 JSON ``{type, message}``.

    :param exc: exception the handler raised
    :ptype exc: BaseException
    :return: tagged error frame
    :rtype: bytes
    """
    body = json.dumps({"type": type(exc).__name__, "message": str(exc)}).encode("utf-8")
    return bytes([_TAG_ERR]) + body


def _decode_reply(frame: bytes) -> bytes:
    """decode a reply frame; raise :class:`ForwardedHandlerError` on an error frame.

    :param frame: tagged reply frame from the owner
    :ptype frame: bytes
    :return: handler's reply bytes (OK frame)
    :rtype: bytes
    :raises ForwardError: if the frame is empty or carries an unknown tag
    :raises ForwardedHandlerError: if the frame is an error frame
    """
    if not frame:
        raise ForwardError("owner replied with an empty frame (no tag byte)")
    tag, body = frame[0], frame[1:]
    if tag == _TAG_OK:
        return body
    if tag == _TAG_ERR:
        try:
            decoded = json.loads(body.decode("utf-8"))
            type_name = str(decoded["type"])
            message = str(decoded["message"])
        except (UnicodeDecodeError, ValueError, KeyError) as exc:
            raise ForwardError(f"owner sent a malformed error frame: {exc}") from exc
        raise ForwardedHandlerError(type_name=type_name, message=message)
    raise ForwardError(f"owner replied with an unknown frame tag: 0x{tag:02x}")


@asynccontextmanager
async def serve_owner(
    nats: NatsClient,
    key: str,
    handler: ForwardHandler,
) -> AsyncIterator[None]:
    """serve forwarded requests for ``key`` while this context is active.

    subscribe the key's forward subject in a queue group keyed by that
    subject (so a brief two-owner overlap during lease handoff still
    dispatches each request to exactly one owner). each forwarded
    request runs ``handler(payload)`` and replies with the framed
    result: the handler's bytes on success, a structured error frame on
    raise. the subscription is dropped on exit.

    forwarded requests for one owner are handled **serially** (the
    subscription does not set ``max_in_flight``), which is the intended
    behaviour for the single-writer pattern this serves — writes to an
    owned key are processed one at a time, not fanned out concurrently.

    intended call site: a consumer that has *just* acquired the lease /
    lock for ``key`` enters this context, and exits it the moment it
    loses ownership::

        async with nats_distributed_lock(nats, key):
            async with serve_owner(nats, key, handler):
                ...  # serve while we own the key

    :param nats: connected canonical :class:`threetears.nats.NatsClient`
    :ptype nats: NatsClient
    :param key: ownership key; mapped deterministically to a forward
        subject via :meth:`Subjects.forward`
    :ptype key: str
    :param handler: async ``bytes -> bytes`` callback run per forwarded
        request; raising surfaces to the caller as
        :class:`ForwardedHandlerError`
    :ptype handler: ForwardHandler
    :return: async iterator yielding ``None`` while serving
    :rtype: AsyncIterator[None]
    :raises SubscribeError: if subscription registration fails
    """
    subject = Subjects.forward(key)

    async def _on_request(msg: IncomingMessage) -> None:
        """run the handler and reply with a framed result.

        a missing reply subject means the caller used fire-and-forget on
        a request subject — there is nobody to reply to, so we just run
        the handler for its effect and skip the reply.
        """
        try:
            reply = await handler(msg.data)
            frame = _encode_ok(reply)
        except Exception as exc:  # noqa: BLE001 -- boundary: every handler failure is framed back to the caller, never swallowed
            frame = _encode_err(exc)
            log.info(
                "owner handler raised; framing error back to caller",
                extra={
                    "extra_data": {
                        "key": key,
                        "subject": subject.path,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                },
            )
        if msg.reply_subject is not None:
            await nats.publish_raw_reply(reply_subject=msg.reply_subject, payload=frame)

    # queue group = the forward subject itself: every owner of the SAME
    # key shares one queue group, so an overlap dispatches to exactly
    # one; distinct keys ride distinct subjects + groups and never mix.
    # deadletter_on_failure is off: handler failures are framed back to
    # the caller (above), they are not transport faults to deadletter.
    sub: Subscription = await nats.subscribe(
        subject=subject,
        cb=_on_request,
        queue=subject.path,
        deadletter_on_failure=False,
    )
    # flush so the subscription is registered server-side before we hand
    # control back to the consumer (and before any forward can race in).
    await nats.flush()
    try:
        yield
    finally:
        await nats.unsubscribe(sub)


async def forward(
    nats: NatsClient,
    key: str,
    payload: bytes,
    *,
    timeout: timedelta = DEFAULT_FORWARD_TIMEOUT,
) -> bytes:
    """forward ``payload`` to whichever pod currently serves ``key``.

    issues a request to the key's forward subject and returns the
    owner's reply bytes. if no pod currently serves the key (no
    subscriber, or the request fell inside a lease-handoff window),
    raises :class:`NoOwnerError`. if the owner's handler raised, raises
    :class:`ForwardedHandlerError` carrying the original type name +
    message.

    :param nats: connected canonical :class:`threetears.nats.NatsClient`
    :ptype nats: NatsClient
    :param key: ownership key; mapped to a forward subject via
        :meth:`Subjects.forward`
    :ptype key: str
    :param payload: opaque request bytes (the consumer serializes its
        own domain payload)
    :ptype payload: bytes
    :param timeout: max wait for the owner's reply
    :ptype timeout: timedelta
    :return: the owner handler's reply bytes
    :rtype: bytes
    :raises NoOwnerError: if no pod currently serves the key (no
        responders / timeout in the handoff window)
    :raises ForwardedHandlerError: if the owner's handler raised
    :raises RequestError: on any other transport failure
    """
    subject = Subjects.forward(key)
    try:
        frame = await nats.request_raw(subject=subject, payload=payload, timeout=timeout)
    except RequestError as exc:
        # request_raw collapses timeout / no-responders / transport into
        # RequestError but preserves the original nats-py exception as
        # __cause__. inspect the cause's TYPE (never the message string)
        # to tell "no live owner right now" apart from a real transport
        # fault: NoRespondersError = nobody subscribed; TimeoutError =
        # request fell inside the handoff window. both are NoOwnerError.
        cause = exc.__cause__
        if isinstance(cause, _NatsNoRespondersError | _NatsTimeoutError | TimeoutError):
            raise NoOwnerError(f"no owner currently serves key {key!r}") from exc
        raise
    return _decode_reply(frame)
