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
    "NatsClientError",
    "PublishError",
    "RequestError",
    "SubscribeError",
]


class NatsClientError(Exception):
    """base class for every 3tears-nats wrapper error.

    callers wanting a single ``except`` clause that catches every
    wrapper-raised condition should catch this base.
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
