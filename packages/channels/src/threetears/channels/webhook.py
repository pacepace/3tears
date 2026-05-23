"""Generic HTTP webhook receiver for the 3tears channels package.

Sibling of the Slack / Discord / WebSocket inbound adapters: receives
HTTP POSTs at a configurable mount point on a host FastAPI app, verifies
the HMAC signature, and delegates to
:func:`threetears.agent.wake.webhook_adapter.webhook_receive` for the
verify + rate-limit + trigger-construct + dispatch flow. Maps the
:class:`WebhookReceiveResult` outcome to a JSON HTTP response.

Spec ref: ``docs/agent-wake/shard-06-channels-webhook-receiver.md``.
PLACEMENT §1.13 (webhook receiver platform-side) + §3.3 (locked:
``3tears-channels``).

The receiver framework owns the HTTP routing-and-response plumbing
ONLY. The verify + rate-limit + dispatch flow is owned by
:func:`webhook_receive` in ``3tears-agent-wake`` (shard 04) so the
wake invariants stay localised. Vendor-specific signature schemes
(GitHub ``X-Hub-Signature-256``, Stripe ``Stripe-Signature``, etc.)
plug in via :meth:`WebhookReceiver.register_verifier` without
modifying this module.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Final
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.agent.wake.config import WakeConfig
    from threetears.agent.wake.entities import EncryptionService
    from threetears.agent.wake.types import DeliveryAdapter, HandlerCallback

__all__ = [
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "DEFAULT_SIGNATURE_HEADER",
    "Verifier",
    "WebhookReceiver",
    "verify_generic_hmac_sha256",
]

log = get_logger(__name__)


# Default header the platform expects to carry the HMAC signature. The
# ``X-3Tears-Webhook-Signature`` name keeps the platform brand-neutral
# (a consumer product can override via the constructor for
# backwards-compatibility with its existing webhook senders -- e.g.
# metallm passes ``X-MetaLLM-Signature`` to preserve its legacy
# senders' headers).
DEFAULT_SIGNATURE_HEADER: Final[str] = "X-3Tears-Webhook-Signature"


# Maximum body size accepted by the receiver before short-circuiting
# with a 413. 1 MiB headroom for the largest realistic JSON webhook
# payload (typical webhook payloads are <100 KB); larger payloads
# would put memory pressure on HMAC verification (the entire body must
# sit in memory for the constant-time compare). Configurable per
# receiver via the ``max_payload_bytes`` constructor kwarg.
DEFAULT_MAX_PAYLOAD_BYTES: Final[int] = 1 << 20  # 1 MiB


# Default value of the ``Retry-After`` response header on a 429. The
# wake-side per-subscription rate limit uses a 60s rolling window
# (``DEFAULT_RATE_WINDOW_SECONDS`` in
# :mod:`threetears.agent.wake.webhook_adapter`); 60s is the worst-case
# wait for the oldest fire in the window to age out. Could be smarter
# by reading the exact window rollover from the adapter result, but
# the static value matches the documented spec simplification.
_DEFAULT_RETRY_AFTER_SECONDS: Final[str] = "60"


Verifier = Callable[[bytes, bytes, dict[str, str]], bool]
"""Signature-verification callable plugged into the receiver.

The :class:`WebhookReceiver` looks up the verifier for an inbound
subscription by its ``verification_scheme`` (e.g. ``'generic_hmac_sha256'``
or a vendor name) and invokes it with ``(secret, payload_bytes, headers)``.
A truthy return means the signature is valid; the receiver then hands
off to :func:`webhook_receive` for the rest of the pipeline.

Vendor adapters (``'github'``, ``'stripe'``, ``'slack'``) register their
own verifier via :meth:`WebhookReceiver.register_verifier`. Verifier
implementations MUST use :func:`hmac.compare_digest` (or equivalent
constant-time compare) to guard against timing attacks.
"""


def verify_generic_hmac_sha256(
    secret: bytes,
    payload: bytes,
    headers: dict[str, str],
) -> bool:
    """Verify an HMAC-SHA256 signature in the platform-default header format.

    The default scheme uses an ``X-3Tears-Webhook-Signature`` header
    carrying ``sha256=<hex>``. The HMAC is computed over the raw
    request bytes with :func:`hmac.compare_digest` for constant-time
    comparison (timing-attack defence). Returns ``False`` for any
    structural problem (missing header, wrong prefix, length mismatch)
    rather than raising; the receiver maps the boolean to a 403.

    :param secret: subscription's decrypted HMAC secret
    :ptype secret: bytes
    :param payload: raw HTTP body to verify against
    :ptype payload: bytes
    :param headers: request headers as a case-folded dict (the receiver
        passes lowercase header names per HTTP/2 convention; verifiers
        SHOULD look up by lowercase)
    :ptype headers: dict[str, str]
    :return: ``True`` when the computed HMAC matches the header
        verbatim, ``False`` otherwise
    :rtype: bool
    """
    sig_header = headers.get(DEFAULT_SIGNATURE_HEADER.lower(), "")
    if not sig_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret, payload, sha256).hexdigest()
    return hmac.compare_digest(expected, sig_header)


class WebhookReceiver:
    """Generic HMAC-verified inbound webhook receiver.

    Routes ``POST {mount_path}/{subscription_id}`` into
    :func:`threetears.agent.wake.webhook_adapter.webhook_receive`. The
    receiver framework owns the HTTP boundary (body read, size cap,
    signature header extraction, source-IP detection, response shape);
    the wake-side adapter owns the verify + rate-limit + trigger +
    dispatch flow.

    Consumers construct the receiver at app-startup time and register
    it on their FastAPI app via :meth:`register`. All dependencies
    (asyncpg pool, encryption service, handler callback, wake config,
    delivery adapters) are constructor args -- no global state.

    Per PLACEMENT §1.13 the receiver does NOT host subscription CRUD
    endpoints. CRUD belongs in the consumer's REST router or the
    agent-tool surfaces from shard 04 (``webhook_subscription_create``,
    etc.). The receiver is receive-side only.

    :ivar _pool: asyncpg pool the wake collections + dispatcher share
    :ivar _encryption_service: consumer's :class:`EncryptionService`
        impl; used to decrypt the per-subscription HMAC secret on each
        inbound fire
    :ivar _handler: consumer's :class:`HandlerCallback`; the wake
        dispatcher hands the constructed :class:`WakeTrigger` to this
        callback for product-specific processing
    :ivar _wake_config: consumer's :class:`WakeConfig`; carries the
        per-conv + per-user caps the dispatcher enforces at fire time
    :ivar _delivery_adapters: mapping of ``DeliveryTarget`` to adapter;
        forwarded to the dispatcher for non-conversation targets
    :ivar _signature_header: header carrying the HMAC signature
        (default ``'X-3Tears-Webhook-Signature'``; consumers with an
        existing brand can override -- e.g. ``'X-MetaLLM-Signature'``)
    :ivar _max_payload_bytes: short-circuit cap on request body size;
        anything larger returns 413 without invoking the wake adapter
    :ivar _verifiers: scheme name -> :class:`Verifier` registry; the
        receiver looks up by the subscription row's
        ``verification_scheme`` column
    """

    def __init__(
        self,
        *,
        pool: Any,
        encryption_service: EncryptionService,
        handler: HandlerCallback,
        wake_config: WakeConfig,
        delivery_adapters: dict[str, DeliveryAdapter] | None = None,
        signature_header: str = DEFAULT_SIGNATURE_HEADER,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        """Construct a webhook receiver bound to its consumer-supplied wiring.

        Per Requirement WEBHOOK-04 every dependency is a constructor
        arg; no global state. This lets a single host process mount
        multiple receivers (different mount paths, different signature
        headers, etc.) for products that operate multiple webhook
        surfaces.

        :param pool: asyncpg pool the wake collections + dispatcher
            share. Forwarded verbatim to :func:`webhook_receive`.
        :ptype pool: Any
        :param encryption_service: consumer's :class:`EncryptionService`
            implementation
        :ptype encryption_service: EncryptionService
        :param handler: consumer's :class:`HandlerCallback`
        :ptype handler: HandlerCallback
        :param wake_config: consumer's :class:`WakeConfig`
        :ptype wake_config: WakeConfig
        :param delivery_adapters: optional mapping of delivery target
            name to :class:`DeliveryAdapter`; ``None`` means
            conversation-only delivery (the platform default)
        :ptype delivery_adapters: dict[str, DeliveryAdapter] | None
        :param signature_header: header name carrying the HMAC
            signature (default
            ``DEFAULT_SIGNATURE_HEADER``)
        :ptype signature_header: str
        :param max_payload_bytes: request body size cap; larger bodies
            short-circuit to 413
        :ptype max_payload_bytes: int
        """
        self._pool = pool
        self._encryption_service = encryption_service
        self._handler = handler
        self._wake_config = wake_config
        self._delivery_adapters: dict[str, DeliveryAdapter] = (
            dict(delivery_adapters) if delivery_adapters is not None else {}
        )
        self._signature_header = signature_header
        self._max_payload_bytes = max_payload_bytes
        self._verifiers: dict[str, Verifier] = {
            "generic_hmac_sha256": verify_generic_hmac_sha256,
        }

    def register_verifier(self, scheme: str, verifier: Verifier) -> None:
        """Register or replace a signature-verification scheme.

        Vendor-specific schemes (``'github'``, ``'stripe'``,
        ``'slack'``, etc.) plug in via this method without modifying
        the receiver. The subscription row's ``verification_scheme``
        column drives the lookup; ``'generic_hmac_sha256'`` is
        pre-registered with :func:`verify_generic_hmac_sha256`.

        Overriding the default scheme is supported (e.g. a consumer
        wanting a non-standard header format can register a custom
        ``'generic_hmac_sha256'`` impl), but production deployments
        typically register vendor schemes alongside the default.

        :param scheme: scheme name matching the
            ``webhook_subscriptions.verification_scheme`` column
        :ptype scheme: str
        :param verifier: callable with the :data:`Verifier` signature
        :ptype verifier: Verifier
        """
        self._verifiers[scheme] = verifier

    def register(self, app: FastAPI, *, mount_path: str = "/webhooks") -> None:
        """Mount the receiver as a ``POST`` route on a FastAPI app.

        Adds a route at ``{mount_path}/{subscription_id}`` accepting
        ``POST`` with the receiver's :meth:`_handle` as the endpoint.
        The path parameter ``subscription_id`` is typed as :class:`UUID`
        so FastAPI rejects malformed paths with 422 before invoking
        the receiver.

        :param app: FastAPI app to mount on. Starlette apps are NOT
            supported here (use ``app.add_route`` directly with
            :meth:`_handle` if a starlette-only target is required).
        :ptype app: FastAPI
        :param mount_path: URL prefix the subscription id is appended
            to. Defaults to ``/webhooks``; consumers serving multiple
            receivers can use distinct prefixes.
        :ptype mount_path: str
        """
        app.add_api_route(
            f"{mount_path}/{{subscription_id}}",
            self._handle,
            methods=["POST"],
            tags=["webhooks"],
        )

    async def _handle(self, subscription_id: UUID, request: Request) -> Response:
        """FastAPI route handler -- the receive-side HTTP boundary.

        Pipeline:

        1. Read body; reject with 413 if it exceeds the size cap.
        2. Extract the signature header verbatim (NO trimming or
           transformation per Requirement WEBHOOK-12 -- the verifier
           gets the raw header value).
        3. Resolve source IP via ``X-Forwarded-For`` first-hop, falling
           back to socket address.
        4. Hand off to :func:`webhook_receive` for subscription lookup,
           verification, rate-limit, trigger build, dispatch.
        5. Map the :class:`WebhookReceiveResult` to a
           :class:`JSONResponse` with the corresponding status code
           and (for 429) a ``Retry-After`` header.

        The webhook adapter's per-subscription verifier-lookup happens
        inside :func:`webhook_receive` against the subscription's
        ``verification_scheme`` column; the receiver framework's
        :attr:`_verifiers` registry is consulted only when a future
        revision extends the adapter to dispatch via the registry.
        For v1 the adapter hardcodes ``generic_hmac_sha256``; the
        registry exists so vendor schemes can land without an adapter
        change.

        :param subscription_id: bare subscription UUID from the path
        :ptype subscription_id: UUID
        :param request: starlette/fastapi request object
        :ptype request: Request
        :return: JSON response carrying ``fire_id`` (when set) and a
            diagnostic ``message`` field
        :rtype: Response
        """
        # Lazy import keeps this module's load cost cheap when the
        # consumer doesn't actually mount webhooks (CLI tools, test
        # runners that only touch other channel adapters). Same
        # pattern as agent-wake's dispatch module uses for its
        # CollectionRegistry import.
        from threetears.agent.wake.webhook_adapter import webhook_receive  # noqa: PLC0415

        body = await request.body()
        if len(body) > self._max_payload_bytes:
            return JSONResponse(
                status_code=413,
                content={"fire_id": None, "message": "payload too large"},
            )

        signature = request.headers.get(self._signature_header)
        source_ip = self._resolve_source_ip(request)

        result = await webhook_receive(
            subscription_id=subscription_id,
            payload_bytes=body,
            signature_header=signature,
            source_ip=source_ip,
            pool=self._pool,
            encryption_service=self._encryption_service,
            handler=self._handler,
            delivery_adapters=self._delivery_adapters,
            wake_config=self._wake_config,
        )

        headers: dict[str, str] = {}
        if result.status_code == 429:
            headers["Retry-After"] = _DEFAULT_RETRY_AFTER_SECONDS
        return JSONResponse(
            status_code=result.status_code,
            content={
                "fire_id": str(result.fire_id) if result.fire_id is not None else None,
                "message": result.message,
            },
            headers=headers,
        )

    def _resolve_source_ip(self, request: Request) -> str | None:
        """Resolve the source IP per the 3tears reverse-proxy convention.

        Reads ``X-Forwarded-For`` first-hop (the leftmost address in
        the comma-separated list, which is the original client behind
        any proxy chain). Falls back to the socket address when no
        ``X-Forwarded-For`` is present. Returns ``None`` when neither
        is available (rare, but possible in test contexts).

        :param request: starlette/fastapi request object
        :ptype request: Request
        :return: source IP string, or ``None`` when undetectable
        :rtype: str | None
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return None
