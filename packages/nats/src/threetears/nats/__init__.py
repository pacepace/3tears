"""3tears-nats — typed NATS client wrapper, subject builders, JetStream KV bucket primitives.

re-exports the public surface every consumer should bind to. callers
should NOT reach into submodules (``threetears.nats.client`` etc.) for
public types — the re-exports here are the stable api.

some of those submodules reach ``nats-py`` or ``nkeys``; the rest are pure
python. those names are therefore resolved **lazily** (PEP 562) so
that importing this package — which happens transitively for anyone
touching ``threetears.core.collections`` — does not drag the NATS client
and its ``nkeys`` dependency into a process that only uses the L1 SQLite
tier. see ``_LAZY_SUBMOD_ATTRS`` below. the eager block that follows is
deliberately eager: those modules cost nothing.

.. note::
   when this package requires python >= 3.15, delete ``_LAZY_SUBMOD_ATTRS``,
   ``_LazyReexportModule``, ``__getattr__``, ``__dir__`` and the
   ``TYPE_CHECKING`` block, and simply prefix each of those plain imports with PEP
   810's ``lazy`` keyword instead. ``_LazyReexportModule`` goes with them:
   a ``lazy from`` binds the name at the import statement, restoring the
   eager ordering that kept the submodule from shadowing its re-export.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # the lazy names, re-imported so type checkers resolve them
    from threetears.nats.client import (
        DEFAULT_DRAIN_TIMEOUT,
        DEFAULT_FLUSHER_QUEUE_SIZE,
        DEFAULT_PENDING_SIZE_BYTES,
        DEFAULT_REQUEST_TIMEOUT,
        DEFAULT_STARTUP_TIMEOUT,
        JetStreamPullConsumer,
        JetStreamPushConsumer,
        JetStreamResultWaiter,
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
    from threetears.nats.pipe import (
        DEFAULT_ATTACH_TIMEOUT,
        DEFAULT_CREDIT_WINDOW_BYTES,
        DEFAULT_MAX_CHUNK_BYTES,
        DEFAULT_READY_TIMEOUT,
        DEFAULT_IDLE_TIMEOUT,
        MIN_PIPE_PROTOCOL_VERSION,
        PipeIdleTimeout,
        PIPE_PROTOCOL_VERSION,
        PipeEndpoint,
        PipeError,
        PipeProtocolError,
        PipeRemoteError,
        PipeSequenceGapError,
        PipeStream,
        PipeStreamHandler,
        PipeTransport,
        attach_pipe,
        open_pipe,
        serve_pipe,
    )
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

    from threetears.nats.user_jwt import (
        account_public_key,
        generate_account_seed,
        mint_user_jwt,
    )

    from threetears.nats.kv import NatsKvBucket
    from threetears.nats.oplog import AppendResult, OpLog, OpRecord

from threetears.nats.errors import (
    KvError,
    NamespaceNotConfiguredError,
    NatsClientError,
    NoRespondersError,
    OpLogError,
    OpLogSequenceConflict,
    PayloadTooLargeError,
    PublishError,
    RequestError,
    RequestTimeoutError,
    SubscribeError,
)
from threetears.nats.result_delivery import (
    RESULT_ACK_TIMEOUT_SECONDS,
    RESULT_RETENTION_SECONDS,
    RESULT_STREAM_SUFFIX,
    SYNC_REPLY_BUDGET_SECONDS,
    reply_subject_is_owned_by_agent,
    reply_subject_prefix_for_agent,
    requires_async_result,
    result_stream_name,
    result_subject_is_owned_by_pod,
    result_subject_prefix_for_pod,
)
from threetears.nats.subject_permissions import (
    CROSS_PLATFORM_CACHE_INVALIDATE,
    Principal,
    PrincipalPermissions,
    build_permissions,
    inbox_prefix_for,
)
from threetears.nats.subjects import (
    PipeDirection,
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

#: submodule -> public names, for every submodule that reaches ``nats-py`` or
#: ``nkeys`` (directly or transitively). everything here resolves on first
#: attribute access rather than at package import, so an L1-only consumer never
#: loads the client. keep in sync with the ``TYPE_CHECKING`` block above and
#: with ``__all__``; ``test_lazy_surface.py`` asserts all three agree.
_LAZY_SUBMOD_ATTRS: Final[dict[str, tuple[str, ...]]] = {
    "client": (
        "DEFAULT_DRAIN_TIMEOUT",
        "DEFAULT_FLUSHER_QUEUE_SIZE",
        "DEFAULT_PENDING_SIZE_BYTES",
        "DEFAULT_REQUEST_TIMEOUT",
        "DEFAULT_STARTUP_TIMEOUT",
        "JetStreamPullConsumer",
        "JetStreamPushConsumer",
        "JetStreamResultWaiter",
        "NatsClient",
        "Subscription",
        "TokenCallback",
    ),
    "cross_worker_cancel": ("CrossWorkerCanceller", "TaskCancelEnvelope"),
    "distributed_lock": ("LockHeld", "nats_distributed_lock"),
    "forward": (
        "DEFAULT_FORWARD_TIMEOUT",
        "ForwardError",
        "ForwardedHandlerError",
        "ForwardHandler",
        "NoOwnerError",
        "forward",
        "serve_owner",
    ),
    "pipe": (
        "DEFAULT_ATTACH_TIMEOUT",
        "DEFAULT_CREDIT_WINDOW_BYTES",
        "DEFAULT_MAX_CHUNK_BYTES",
        "DEFAULT_IDLE_TIMEOUT",
        "DEFAULT_READY_TIMEOUT",
        "MIN_PIPE_PROTOCOL_VERSION",
        "PIPE_PROTOCOL_VERSION",
        "PipeEndpoint",
        "PipeError",
        "PipeIdleTimeout",
        "PipeProtocolError",
        "PipeRemoteError",
        "PipeSequenceGapError",
        "PipeStream",
        "PipeStreamHandler",
        "PipeTransport",
        "attach_pipe",
        "open_pipe",
        "serve_pipe",
    ),
    "auth_callout": ("AuthCalloutRequest", "decode_auth_request", "mint_auth_response"),
    "auth_callout_responder": (
        "AUTH_CALLOUT_SUBJECT",
        "DEFAULT_AUTH_CALLOUT_QUEUE_GROUP",
        "DEFAULT_NATS_USER_JWT_TTL_SECONDS",
        "AuthAccountKeyError",
        "AuthCalloutResponder",
        "GrantPolicy",
        "PrincipalResolver",
        "ResolvedPrincipal",
    ),
    "kv": ("NatsKvBucket",),
    "oplog": ("AppendResult", "OpLog", "OpRecord"),
    "user_jwt": ("account_public_key", "generate_account_seed", "mint_user_jwt"),
}

_LAZY_ATTR_TO_SUBMOD: Final[dict[str, str]] = {
    attr: submod for submod, attrs in _LAZY_SUBMOD_ATTRS.items() for attr in attrs
}

#: distributions whose absence means "the client extra is not installed", as
#: opposed to a genuine broken import inside one of our own submodules.
_CLIENT_DISTS: Final[frozenset[str]] = frozenset({"nats", "nkeys"})


class _LazyReexportModule(ModuleType):
    """Keeps a re-export from being shadowed by its same-named submodule.

    ``forward`` is both a submodule of this package and the function that
    submodule exports. Importing that submodule -- directly, or as a side
    effect of resolving any *other* lazy name from it, such as
    ``ForwardedHandlerError`` -- makes the import machinery bind the module
    onto this package under the same name. That write lands in the package
    dict, so :func:`__getattr__` stops firing and ``threetears.nats.forward``
    silently becomes the module: a call then fails with ``TypeError: 'module'
    object is not callable``, far from the import that caused it.

    Eager re-exports had no such problem, because the ``from ... import``
    that bound the function ran *after* the machinery bound the module. Under
    lazy resolution that order inverts, so the collision is refused here
    instead. The submodule stays reachable via
    ``sys.modules['threetears.nats.forward']`` and via ``from
    threetears.nats.forward import ...``, which is how the internals reach it.
    """

    def __setattr__(self, name: str, value: object) -> None:
        shadows_reexport = (
            name in _LAZY_ATTR_TO_SUBMOD
            and isinstance(value, ModuleType)
            and getattr(value, "__name__", None) == f"{__name__}.{name}"
        )
        if shadows_reexport:
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _LazyReexportModule


def __getattr__(name: str) -> object:
    """Resolve a nats-py-backed re-export on first access (PEP 562)."""
    submod = _LAZY_ATTR_TO_SUBMOD.get(name)
    if submod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{submod}")
    except ImportError as exc:
        # only claim "extra not installed" when that is actually what happened;
        # a broken import inside our own submodule must surface as itself.
        if (exc.name or "").split(".")[0] not in _CLIENT_DISTS:
            raise
        raise ImportError(
            f"{name} requires the NATS client, which is not installed. "
            f"Install it with: pip install '3tears-nats[client]' "
            f"(or '3tears[nats]' when installing via core)."
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # cache: __getattr__ will not fire again for this name
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_ATTR_TO_SUBMOD})


__all__ = [
    # client + lifecycle
    "DEFAULT_DRAIN_TIMEOUT",
    "DEFAULT_FLUSHER_QUEUE_SIZE",
    "DEFAULT_PENDING_SIZE_BYTES",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_STARTUP_TIMEOUT",
    "JetStreamPullConsumer",
    "JetStreamPushConsumer",
    "JetStreamResultWaiter",
    "NatsClient",
    "Subscription",
    "TokenCallback",
    # subjects
    "PipeDirection",
    "Subject",
    "SubjectKind",
    "Subjects",
    "get_default_namespace",
    "set_default_namespace",
    # asynchronous result delivery (answers that outlive their receiving connection)
    "RESULT_ACK_TIMEOUT_SECONDS",
    "RESULT_RETENTION_SECONDS",
    "RESULT_STREAM_SUFFIX",
    "SYNC_REPLY_BUDGET_SECONDS",
    "reply_subject_is_owned_by_agent",
    "reply_subject_prefix_for_agent",
    "requires_async_result",
    "result_stream_name",
    "result_subject_is_owned_by_pod",
    "result_subject_prefix_for_pod",
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
    # byte pipe (a stream to whichever pod owns a key)
    "DEFAULT_ATTACH_TIMEOUT",
    "DEFAULT_CREDIT_WINDOW_BYTES",
    "DEFAULT_MAX_CHUNK_BYTES",
    "DEFAULT_IDLE_TIMEOUT",
    "DEFAULT_READY_TIMEOUT",
    "MIN_PIPE_PROTOCOL_VERSION",
    "PIPE_PROTOCOL_VERSION",
    "PipeEndpoint",
    "PipeError",
    "PipeIdleTimeout",
    "PipeProtocolError",
    "PipeRemoteError",
    "PipeSequenceGapError",
    "PipeStream",
    "PipeStreamHandler",
    "PipeTransport",
    "attach_pipe",
    "open_pipe",
    "serve_pipe",
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
    "PayloadTooLargeError",
    "PublishError",
    "NoRespondersError",
    "RequestError",
    "RequestTimeoutError",
    "SubscribeError",
]
