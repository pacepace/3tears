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
    "OpLogError",
    "OpLogSequenceConflict",
    "PublishError",
    "RequestError",
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


class PublishError(NatsClientError):
    """raised when a publish operation fails.

    wraps any underlying nats-py exception. ``__cause__`` carries
    original exception for diagnostics.
    """


class SubscribeError(NatsClientError):
    """raised when a subscription operation fails to register.

    not raised for callback-side exceptions during message processing
    (those are routed to deadletter or re-raised by the dispatcher).
    """


class RequestError(NatsClientError):
    """raised when a request/reply round-trip fails.

    covers timeout, no-responders, transport failure, and response
    decode failure. distinct subclasses may be added later if callers
    need to disambiguate.
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
