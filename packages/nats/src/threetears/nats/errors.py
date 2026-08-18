"""error hierarchy for the 3tears-nats wrapper.

every public method on :class:`threetears.nats.NatsClient` raises one
of these typed exceptions on failure rather than leaking
``nats-py``-specific exception types to callers. this keeps consumer
code decoupled from the underlying client library and makes it
possible to rev ``nats-py`` without breaking every catch site.
"""

from __future__ import annotations

__all__ = [
    "KvError",
    "NamespaceNotConfiguredError",
    "NatsClientError",
    "NoRespondersError",
    "OpLogError",
    "OpLogSequenceConflict",
    "PayloadTooLargeError",
    "PublishError",
    "PublishTimeoutError",
    "RequestError",
    "RequestTimeoutError",
    "StreamSubjectsOverlapError",
    "SubscribeError",
]


class NatsClientError(Exception):
    """base class for every 3tears-nats wrapper error.

    callers wanting a single ``except`` clause that catches every
    wrapper-raised condition should catch this base.
    """


class NamespaceNotConfiguredError(NatsClientError):
    """raised when a subject namespace is required but none is configured.

    the subject namespace prefix has no default. configure it explicitly
    by calling :func:`threetears.nats.set_default_namespace`, by
    connecting a :class:`threetears.nats.NatsClient` (which binds its
    configured ``nats_subject_namespace``), or by setting the
    ``THREETEARS_NATS_SUBJECT_NAMESPACE`` environment variable. requiring
    an explicit value prevents two unconfigured deployments from silently
    colliding on the same subjects when they share a NATS cluster.
    """


class StreamSubjectsOverlapError(NatsClientError):
    """raised when a JetStream stream cannot be created because its subjects
    are already claimed by a DIFFERENT stream on the same NATS account.

    a subject belongs to exactly one stream. this is NOT the "stream already
    exists under the same name" case (which is reconciled by updating the
    existing stream's subject set): the requested stream name was never
    created, so updating it would raise a confusing ``NotFoundError`` and mask
    this actionable error. the usual cause is another connection publishing
    over these subjects under the WRONG subject namespace -- e.g. a client that
    did not pass its own ``nats_subject_namespace`` and fell back to a shared
    default, minting a stream that squats the subject space. resolve the
    conflicting stream or correct the namespace. ``__cause__`` carries the
    underlying nats-py exception for diagnostics.
    """


class PublishError(NatsClientError):
    """raised when a publish operation fails.

    wraps any underlying nats-py exception. ``__cause__`` carries
    original exception for diagnostics.
    """


class PublishTimeoutError(PublishError):
    """raised when a publish did not complete inside its deadline.

    Distinct from :class:`PublishError` because the outcome is genuinely
    unknown rather than failed: a JetStream publish that times out waiting for
    its ``PubAck`` may still have been persisted by the broker. A caller
    retrying must therefore treat the message as possibly-already-delivered --
    use a deduplication header (``Nats-Msg-Id``) if exactly-once matters.

    **The publish may also still be running.** ``nats-py``'s flush path
    swallows ``asyncio.CancelledError``, so a publish that has stopped
    responding to cancellation cannot be awaited to completion without
    reproducing the wedge this error exists to break. When that happens the
    wrapper abandons the in-flight task and says so in the log rather than
    blocking its caller on a coroutine that has refused to stop.
    """


class PayloadTooLargeError(PublishError):
    """raised when a publish is bigger than the broker will accept.

    ``nats-py`` refuses an oversized publish **client-side**, before anything
    leaves the process, by comparing the payload against the ``max_payload`` the
    server advertised in its INFO -- 1 MB on an untuned broker. That refusal
    arrived here as a generic :class:`PublishError` carrying a stringified
    cause, which is the one classification a caller most needs to tell apart:
    "the frame we built is too big" is a bug in what we built, and "the broker
    went away" is an outage. They deserve different log lines, and only one of
    them is worth retrying -- **retrying an oversized publish is an infinite
    loop that logs**, because nothing about the next attempt is smaller.

    The failure is quiet in exactly the deployments that produce it. A room
    frame crossing NATS is published once and fanned out; the publisher's own
    socket already has the content, so the person who caused the frame sees
    everything and the room sees nothing. Single-pod tests never reach the
    publish at all.

    Both numbers are attributes rather than only message text, so a caller can
    act on them -- pick a smaller shape, or say by how much it missed --
    without parsing prose. :attr:`max_payload` is ``None`` when the client
    could not say what the broker advertised.

    :ivar subject: the subject the publish targeted
    :ivar size_bytes: the size of the payload that was refused
    :ivar max_payload: the broker's advertised limit, when known
    """

    def __init__(
        self,
        *,
        subject: str,
        size_bytes: int,
        max_payload: int | None,
    ) -> None:
        """build the error, naming both numbers in the message and on the instance.

        :param subject: the subject the publish targeted
        :ptype subject: str
        :param size_bytes: size of the refused payload
        :ptype size_bytes: int
        :param max_payload: the broker's advertised limit, when known
        :ptype max_payload: int | None
        :return: nothing
        :rtype: None
        """
        limit = f"{max_payload} bytes" if max_payload is not None else "unknown (not connected)"
        super().__init__(
            f"publish refused as too large: subject={subject}: "
            f"payload is {size_bytes} bytes, broker max_payload is {limit}"
        )
        self.subject = subject
        self.size_bytes = size_bytes
        self.max_payload = max_payload


class SubscribeError(NatsClientError):
    """raised when a subscription operation fails to register.

    not raised for callback-side exceptions during message processing
    (those are routed to deadletter or re-raised by the dispatcher).
    """


class RequestError(NatsClientError):
    """raised when a request/reply round-trip fails.

    covers timeout, no-responders, transport failure, and response
    decode failure. the two failures a caller most often needs to tell
    apart -- nobody is subscribed, versus somebody is but did not answer
    in time -- have their own subclasses below. every one of them still
    IS a ``RequestError``, so an existing ``except RequestError`` keeps
    catching all of them unchanged.
    """


class NoRespondersError(RequestError):
    """raised when a request reaches NATS but nothing is subscribed to the subject.

    distinct from :class:`RequestTimeoutError`, and the distinction is
    operationally load-bearing: "nobody is listening" points at a service
    that never started, was never deployed, or is subscribed on a different
    subject/namespace, while a timeout points at one that IS there and is
    slow or wedged. those send an operator to different places.

    the caller that motivated splitting these apart is an agent registering
    with the agent router at boot: on a cold rollout the router may not have
    subscribed yet, which is an ordinary startup race worth waiting out,
    where a transport failure is not. collapsing both into ``RequestError``
    left "wait and retry" indistinguishable from "give up" except by
    matching on message text.
    """


class RequestTimeoutError(RequestError):
    """raised when a request is delivered but no reply arrives within the timeout.

    the responder exists (contrast :class:`NoRespondersError`) and either did
    not answer, answered too slowly, or answered to a reply subject that never
    got back. retrying is often right; retrying forever is not.
    """


class KvError(NatsClientError):
    """raised when a JetStream KV bucket operation fails.

    covers bucket creation, get/put/delete/create/update, and CAS
    conflicts. CAS-specific outcomes (revision-mismatch on update)
    surface as a return-value of ``None`` from
    :meth:`threetears.nats.NatsKvBucket.update`; this exception is
    reserved for transport / bucket-existence failures.
    """


class OpLogError(NatsClientError):
    """raised when a JetStream op-log (WAL) operation fails.

    covers stream creation/binding, append transport failures, and
    replay failures on :class:`threetears.nats.OpLog`. the
    expected-last-sequence CAS conflict (fence #1) is *not* a generic
    op-log failure -- it surfaces as the dedicated
    :class:`OpLogSequenceConflict` subclass so callers can catch the
    fence specifically. this base is reserved for transport /
    stream-existence failures.
    """


class OpLogSequenceConflict(OpLogError):
    """raised when an op-log append fails the expected-last-sequence CAS.

    this is fence #1 of the durable write path: an appender whose
    ``expected_last_seq`` no longer matches the stream's current last
    sequence is rejected **in-band** (a stale / split-brain second
    writer). it is distinct from a duplicate-``op_id`` retry, which is
    an at-most-once no-op and returns
    :class:`threetears.nats.AppendResult` with ``deduplicated=True``
    rather than raising.
    """
