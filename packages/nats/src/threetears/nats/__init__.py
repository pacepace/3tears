"""3tears-nats — typed NATS client wrapper, subject builders, JetStream KV bucket primitives.

re-exports the public surface every consumer should bind to. callers
should NOT reach into submodules (``threetears.nats.client`` etc.) for
public types — the re-exports here are the stable api.
"""

from __future__ import annotations

from threetears.nats.client import (
    DEFAULT_DRAIN_TIMEOUT,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_STARTUP_TIMEOUT,
    NatsClient,
    Subscription,
)
from threetears.nats.errors import (
    KvError,
    NatsClientError,
    PublishError,
    RequestError,
    SubscribeError,
)
from threetears.nats.kv import NatsKvBucket
from threetears.nats.subjects import (
    DEFAULT_NAMESPACE,
    Subject,
    SubjectKind,
    Subjects,
    get_default_namespace,
    set_default_namespace,
)
from threetears.nats.transport import (
    IncomingMessage,
    MessageCallback,
    RawMessageCallback,
    StreamTransport,
)

__all__ = [
    # client + lifecycle
    "DEFAULT_DRAIN_TIMEOUT",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_STARTUP_TIMEOUT",
    "NatsClient",
    "Subscription",
    # subjects
    "DEFAULT_NAMESPACE",
    "Subject",
    "SubjectKind",
    "Subjects",
    "get_default_namespace",
    "set_default_namespace",
    # KV
    "NatsKvBucket",
    # transport Protocols + message envelope
    "IncomingMessage",
    "MessageCallback",
    "RawMessageCallback",
    "StreamTransport",
    # errors
    "KvError",
    "NatsClientError",
    "PublishError",
    "RequestError",
    "SubscribeError",
]
