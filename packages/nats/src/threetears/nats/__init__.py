"""3tears-nats — typed NATS client wrapper, subject builders, JetStream KV bucket primitives.

re-exports the public surface every consumer should bind to. callers
should NOT reach into submodules (``threetears.nats.client`` etc.) for
public types — the re-exports here are the stable api.
"""

from __future__ import annotations

from threetears.nats.client import (
    DEFAULT_DRAIN_TIMEOUT,
    DEFAULT_FLUSHER_QUEUE_SIZE,
    DEFAULT_PENDING_SIZE_BYTES,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_STARTUP_TIMEOUT,
    NatsClient,
    Subscription,
    TokenCallback,
)
from threetears.nats.cross_worker_cancel import (
    CrossWorkerCanceller,
    TaskCancelEnvelope,
)
from threetears.nats.distributed_lock import LockHeld, nats_distributed_lock
from threetears.nats.forward import (
    DEFAULT_FORWARD_TIMEOUT,
    ForwardError,
    ForwardedHandlerError,
    ForwardHandler,
    NoOwnerError,
    forward,
    serve_owner,
)
from threetears.nats.errors import (
    KvError,
    NamespaceNotConfiguredError,
    NatsClientError,
    OpLogError,
    OpLogSequenceConflict,
    PublishError,
    RequestError,
    SubscribeError,
)
from threetears.nats.kv import NatsKvBucket
from threetears.nats.oplog import AppendResult, OpLog, OpRecord
from threetears.nats.auth_callout import (
    AuthCalloutRequest,
    decode_auth_request,
    mint_auth_response,
)
from threetears.nats.auth_callout_responder import (
    AUTH_CALLOUT_SUBJECT,
    DEFAULT_AUTH_CALLOUT_QUEUE_GROUP,
    DEFAULT_NATS_USER_JWT_TTL_SECONDS,
    AuthAccountKeyError,
    AuthCalloutResponder,
    GrantPolicy,
    PrincipalResolver,
    ResolvedPrincipal,
)
from threetears.nats.subject_permissions import (
    CROSS_PLATFORM_CACHE_INVALIDATE,
    Principal,
    PrincipalPermissions,
    build_permissions,
    inbox_prefix_for,
)
from threetears.nats.subjects import (
    Subject,
    SubjectKind,
    Subjects,
    get_default_namespace,
    set_default_namespace,
)
from threetears.nats.user_jwt import (
    account_public_key,
    generate_account_seed,
    mint_user_jwt,
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
    "DEFAULT_FLUSHER_QUEUE_SIZE",
    "DEFAULT_PENDING_SIZE_BYTES",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_STARTUP_TIMEOUT",
    "NatsClient",
    "Subscription",
    "TokenCallback",
    # subjects
    "Subject",
    "SubjectKind",
    "Subjects",
    "get_default_namespace",
    "set_default_namespace",
    # subject permissions (decentralized-auth allow-lists)
    "CROSS_PLATFORM_CACHE_INVALIDATE",
    "Principal",
    "PrincipalPermissions",
    "build_permissions",
    "inbox_prefix_for",
    # NATS v2 user-JWT minting (decentralized auth)
    "account_public_key",
    "generate_account_seed",
    "mint_user_jwt",
    # NATS auth-callout request/response codecs
    "AuthCalloutRequest",
    "decode_auth_request",
    "mint_auth_response",
    # NATS auth-callout responder (generalized: PrincipalResolver + GrantPolicy seams)
    "AUTH_CALLOUT_SUBJECT",
    "DEFAULT_AUTH_CALLOUT_QUEUE_GROUP",
    "DEFAULT_NATS_USER_JWT_TTL_SECONDS",
    "AuthAccountKeyError",
    "AuthCalloutResponder",
    "GrantPolicy",
    "PrincipalResolver",
    "ResolvedPrincipal",
    # KV
    "NatsKvBucket",
    # op-log (durable write-path WAL)
    "AppendResult",
    "OpLog",
    "OpRecord",
    # distributed lock
    "LockHeld",
    "nats_distributed_lock",
    # cross-worker cancel-by-key (keyed task registry + routed cancel)
    "CrossWorkerCanceller",
    "TaskCancelEnvelope",
    # owner-routed forward
    "DEFAULT_FORWARD_TIMEOUT",
    "ForwardError",
    "ForwardedHandlerError",
    "ForwardHandler",
    "NoOwnerError",
    "forward",
    "serve_owner",
    # transport Protocols + message envelope
    "IncomingMessage",
    "MessageCallback",
    "RawMessageCallback",
    "StreamTransport",
    # errors
    "KvError",
    "NamespaceNotConfiguredError",
    "NatsClientError",
    "OpLogError",
    "OpLogSequenceConflict",
    "PublishError",
    "RequestError",
    "SubscribeError",
]
