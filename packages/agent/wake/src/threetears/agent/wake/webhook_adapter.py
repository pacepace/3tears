"""Webhook receiver adapter -- verify, rate-limit, dispatch.

Shard 06's :class:`WebhookReceiver` framework lives in ``3tears-channels``
and owns the HTTP routing, but the verify -> construct -> dispatch flow
is owned here so the platform's wake invariants stay localised.

:func:`webhook_receive` is the single entry point: looks up the
subscription, verifies HMAC, applies rate-limit + source-IP allow-list,
renders the Jinja2 template against the inbound payload, constructs a
:class:`WakeTrigger`, inserts the in-flight ``wake_fires`` row, and
hands off to :func:`dispatch_wake`. Returns
:class:`WebhookReceiveResult` so the receiver can map outcomes to HTTP
status codes without leaking implementation detail.

Spec ref: ``docs/agent-wake/shard-04-agent-tools-and-webhook-adapter.md``
Requirements TOOL-15 / TOOL-16 + PLACEMENT §1.13.
"""

from __future__ import annotations

import hmac
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Final
from uuid import UUID

from jinja2 import TemplateError
from jinja2.sandbox import SandboxedEnvironment
from uuid_utils import uuid7

from threetears.agent.wake.collections import (
    WakeFireCollection,
    WebhookSubscriptionCollection,
)
from threetears.agent.wake.config import DEFAULT_WAKE_CONFIG, WakeConfig
from threetears.agent.wake.dispatch import dispatch_wake
from threetears.agent.wake.entities import EncryptionService
from threetears.agent.wake.events import (
    EVENT_WEBHOOK_AUTH_FAILED,
    EVENT_WEBHOOK_RATE_LIMITED,
    EVENT_WEBHOOK_RECEIVED,
    EVENT_WEBHOOK_REJECTED,
)
from threetears.agent.wake.metrics import get_wake_emitter
from threetears.agent.wake.types import (
    DeliveryAdapter,
    HandlerCallback,
    WakeTrigger,
)
from threetears.observe import get_logger

__all__ = [
    "DEFAULT_RATE_WINDOW_SECONDS",
    "WebhookReceiveResult",
    "webhook_receive",
]


log = get_logger(__name__)


# Sliding window the rate-limit query covers. 60s mirrors the spec's
# "rate_limit_per_minute" naming; the receiver counts fires in this
# window against the subscription's per-minute cap.
DEFAULT_RATE_WINDOW_SECONDS: Final[int] = 60


# Same env construction as the create/update validators. Autoescape
# is False so the rendered output is plain text, not HTML.
_jinja_env = SandboxedEnvironment(autoescape=False)


@dataclass(frozen=True)
class WebhookReceiveResult:
    """Outcome envelope returned by :func:`webhook_receive`.

    Frozen so the HTTP-side receiver in ``3tears-channels`` cannot
    accidentally mutate fields the platform owns. ``status_code``
    matches the HTTP status the receiver should return:

    - 202 -- accepted + dispatched (``fire_id`` is set)
    - 400 -- malformed payload / template render error
    - 401 -- missing or invalid signature
    - 403 -- source IP not in allow-list
    - 404 -- subscription not found or paused
    - 429 -- per-subscription rate limit exceeded
    - 500 -- dispatch / persistence failed downstream

    :ivar status_code: HTTP status the receiver should return
    :ivar fire_id: created ``wake_fires`` row id (202 / 500 only)
    :ivar message: human-readable diagnostic for the receiver to log
    """

    status_code: int
    fire_id: UUID | None
    message: str


async def webhook_receive(
    *,
    subscription_id: UUID,
    payload_bytes: bytes,
    signature_header: str | None,
    source_ip: str | None,
    pool: Any,
    encryption_service: EncryptionService,
    handler: HandlerCallback,
    delivery_adapters: dict[str, DeliveryAdapter] | None = None,
    default_rate_limit_per_minute: int = 60,
    rate_window_seconds: int = DEFAULT_RATE_WINDOW_SECONDS,
    now: datetime | None = None,
    wake_config: WakeConfig = DEFAULT_WAKE_CONFIG,
) -> WebhookReceiveResult:
    """Verify, rate-limit, and dispatch an inbound webhook.

    Flow (per PLACEMENT §1.13 + Requirement TOOL-16):

    1. Look up the subscription by bare ``subscription_id`` (the
       receiver has no conversation context until this lookup).
    2. Verify HMAC over the raw ``payload_bytes`` against the
       subscription's decrypted secret (constant-time compare).
    3. Apply the optional ``allowed_source_pattern`` regex against
       ``source_ip``.
    4. Count recent fires for the subscription and reject when the
       per-minute cap (subscription override or ``default``) is
       exceeded.
    5. Render the Jinja2 template against the decoded payload to
       build the per-fire task prompt.
    6. Build a :class:`WakeTrigger` with the subscription's
       ``default_skill_id`` attached.
    7. INSERT the ``wake_fires`` row in ``status='dispatching'`` via
       :meth:`WakeFireCollection.create_dispatching`.
    8. Hand off to :func:`dispatch_wake`; on success the dispatcher's
       caller finalizes the row, on failure we mark it failed here.

    :param subscription_id: bare subscription UUID (path param)
    :ptype subscription_id: UUID
    :param payload_bytes: raw HTTP body for HMAC verification + JSON
        decode (the HMAC is computed over the un-decoded bytes per the
        constant-time-compare requirement)
    :ptype payload_bytes: bytes
    :param signature_header: ``X-Hub-Signature-256`` / equivalent
        header carrying ``"sha256=<hex>"``
    :ptype signature_header: str | None
    :param source_ip: client IP for the allow-list check; ``None``
        bypasses the regex check (caller has no IP info)
    :ptype source_ip: str | None
    :param pool: asyncpg pool the Collections + dispatcher share
    :ptype pool: Any
    :param encryption_service: consumer-supplied encryption service
        used to decrypt ``secret_ciphertext``
    :ptype encryption_service: EncryptionService
    :param handler: consumer-supplied :class:`HandlerCallback` for
        :func:`dispatch_wake`
    :ptype handler: HandlerCallback
    :param delivery_adapters: optional mapping of non-conversation
        delivery targets to adapters (forwarded to
        :func:`dispatch_wake`)
    :ptype delivery_adapters: dict[str, DeliveryAdapter] | None
    :param default_rate_limit_per_minute: cap used when the
        subscription has no per-row override
    :ptype default_rate_limit_per_minute: int
    :param rate_window_seconds: sliding window the rate count covers
        (defaults to :data:`DEFAULT_RATE_WINDOW_SECONDS`)
    :ptype rate_window_seconds: int
    :param now: reference instant for the rate-limit window + fire
        timestamps (defaults to ``datetime.now(UTC)``)
    :ptype now: datetime | None
    :param wake_config: consumer's :class:`WakeConfig` impl forwarded to
        :func:`dispatch_wake` for the per-conv + per-user rate-limit
        check at dispatch time. The webhook-side per-subscription cap
        is enforced inline above (using ``default_rate_limit_per_minute``
        / the subscription row override) so callers passing the default
        config still get full coverage.
    :ptype wake_config: WakeConfig
    :return: outcome envelope the receiver translates to an HTTP
        response
    :rtype: WebhookReceiveResult
    """
    receive_at = now if now is not None else datetime.now(UTC)
    emitter = get_wake_emitter()
    # local imports keep the receiver's module-load cost cheap when
    # the platform isn't actually serving webhooks (test runners,
    # CLI tools, etc.). Same pattern as dispatch_wake's lazy
    # CollectionRegistry construction.
    from threetears.core.collections.registry import CollectionRegistry  # noqa: PLC0415
    from threetears.core.config import DefaultCoreConfig  # noqa: PLC0415

    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")

    subs = WebhookSubscriptionCollection(registry=registry, config=cfg)
    sub = await subs.find_by_id(subscription_id)
    if sub is None or sub.status != "active":
        log.info(
            EVENT_WEBHOOK_REJECTED,
            extra={"extra_data": {"subscription_id": str(subscription_id), "reason": "not_found"}},
        )
        emitter.inc_webhook_received(outcome="not_found")
        return WebhookReceiveResult(
            status_code=404,
            fire_id=None,
            message="subscription not found or paused",
        )

    # HMAC verification ------------------------------------------------
    if not signature_header:
        log.info(
            EVENT_WEBHOOK_AUTH_FAILED,
            extra={"extra_data": {"subscription_id": str(subscription_id), "reason": "missing_header"}},
        )
        emitter.inc_webhook_received(outcome="auth_failed")
        return WebhookReceiveResult(
            status_code=401,
            fire_id=None,
            message="missing signature header",
        )
    try:
        secret = sub.decrypt_secret(encryption_service)
    except Exception as exc:  # noqa: BLE001 - encryption boundary
        log.warning(
            "webhook_receive secret decrypt failed",
            extra={"extra_data": {"subscription_id": str(subscription_id), "error": str(exc)}},
        )
        return WebhookReceiveResult(
            status_code=500,
            fire_id=None,
            message=f"secret decrypt failed: {exc}",
        )
    expected = (
        "sha256="
        + hmac.new(
            secret.encode("utf-8"),
            payload_bytes,
            sha256,
        ).hexdigest()
    )
    if not hmac.compare_digest(expected, signature_header):
        log.info(
            EVENT_WEBHOOK_AUTH_FAILED,
            extra={"extra_data": {"subscription_id": str(subscription_id), "reason": "bad_signature"}},
        )
        emitter.inc_webhook_received(outcome="auth_failed")
        return WebhookReceiveResult(
            status_code=401,
            fire_id=None,
            message="invalid signature",
        )

    # Source IP allow-list --------------------------------------------
    if sub.allowed_source_pattern is not None and source_ip is not None:
        try:
            if not re.match(sub.allowed_source_pattern, source_ip):
                log.info(
                    EVENT_WEBHOOK_REJECTED,
                    extra={
                        "extra_data": {
                            "subscription_id": str(subscription_id),
                            "reason": "source_rejected",
                        }
                    },
                )
                emitter.inc_webhook_received(outcome="source_rejected")
                return WebhookReceiveResult(
                    status_code=403,
                    fire_id=None,
                    message="source IP not allowed",
                )
        except re.error as exc:
            # Subscription row stored an invalid regex; treat as 500
            # so the operator notices via Prometheus rather than
            # silently rejecting every webhook.
            log.warning(
                "webhook_receive allowed_source_pattern regex invalid",
                extra={
                    "extra_data": {
                        "subscription_id": str(subscription_id),
                        "pattern": sub.allowed_source_pattern,
                        "error": str(exc),
                    }
                },
            )
            return WebhookReceiveResult(
                status_code=500,
                fire_id=None,
                message=f"allowed_source_pattern regex invalid: {exc}",
            )

    # Rate-limit -------------------------------------------------------
    fires = WakeFireCollection(registry=registry, config=cfg)
    cap = sub.rate_limit_per_minute or default_rate_limit_per_minute
    try:
        window_count = await _count_recent_fires_for_subscription(
            fires_collection=fires,
            subscription_id=subscription_id,
            conversation_id=sub.conversation_id,
            since=receive_at.timestamp() - rate_window_seconds,
            pool=pool,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "webhook_receive rate-limit query failed",
            extra={"extra_data": {"subscription_id": str(subscription_id), "error": str(exc)}},
        )
        return WebhookReceiveResult(
            status_code=500,
            fire_id=None,
            message=f"rate-limit query failed: {exc}",
        )
    if window_count >= cap:
        log.info(
            EVENT_WEBHOOK_RATE_LIMITED,
            extra={
                "extra_data": {
                    "subscription_id": str(subscription_id),
                    "conversation_id": str(sub.conversation_id),
                    "count": window_count,
                    "cap": cap,
                    "window_seconds": rate_window_seconds,
                }
            },
        )
        emitter.inc_webhook_received(outcome="rate_limited")
        emitter.inc_rate_limit_rejection(scope="webhook")
        return WebhookReceiveResult(
            status_code=429,
            fire_id=None,
            message=f"rate limit exceeded: {window_count} fires in last {rate_window_seconds}s (cap {cap})",
        )

    # Decode + render template ----------------------------------------
    try:
        payload_obj = _decode_payload(payload_bytes)
    except ValueError as exc:
        return WebhookReceiveResult(
            status_code=400,
            fire_id=None,
            message=f"payload decode error: {exc}",
        )

    template_text = sub.task_prompt_template or ""
    try:
        template = _jinja_env.from_string(template_text)
        rendered = template.render(event=payload_obj)
    except TemplateError as exc:
        log.warning(
            "webhook_receive template render failed",
            extra={"extra_data": {"subscription_id": str(subscription_id), "error": str(exc)}},
        )
        emitter.inc_webhook_received(outcome="bad_template")
        return WebhookReceiveResult(
            status_code=400,
            fire_id=None,
            message=f"template render error: {exc}",
        )

    # Build trigger ---------------------------------------------------
    trigger = WakeTrigger(
        schedule_id=None,  # webhook fires carry no source schedule
        user_id=sub.user_id,
        agent_id=sub.agent_id,
        conversation_id=sub.conversation_id,
        fire_source="webhook",
        execution_mode=sub.execution_mode,
        schedule_type="webhook",
        fired_at=receive_at,
        schedule_name=sub.name,
        task_prompt=rendered,
        context_from_schedule_id=None,
        delivery_target=sub.delivery_target,
        delivery_config=dict(sub.delivery_config),
        skill_id=sub.default_skill_id,
    )

    fire_id = UUID(str(uuid7()))
    try:
        await fires.create_dispatching(
            fire_id=fire_id,
            schedule_id=None,
            webhook_subscription_id=subscription_id,
            conversation_id=sub.conversation_id,
            scheduled_fire_at=None,
            actual_fired_at=receive_at,
            fire_source=trigger.fire_source,
            execution_mode=trigger.execution_mode,
            delivery_target_resolved=trigger.delivery_target,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "webhook_receive create_dispatching failed",
            extra={"extra_data": {"subscription_id": str(subscription_id), "error": str(exc)}},
        )
        return WebhookReceiveResult(
            status_code=500,
            fire_id=None,
            message=f"persist failed: {exc}",
        )

    # Dispatch --------------------------------------------------------
    try:
        result = await dispatch_wake(
            trigger,
            fire_id,
            pool,
            handler=handler,
            delivery_adapters=delivery_adapters,
            wake_config=wake_config,
        )
    except Exception as exc:  # noqa: BLE001 - dispatch boundary
        log.exception(
            "webhook_receive dispatch failed",
            extra={"extra_data": {"subscription_id": str(subscription_id), "fire_id": str(fire_id)}},
        )
        emitter.inc_webhook_received(outcome="failed")
        emitter.inc_fire(
            status="failed",
            schedule_type="webhook",
            execution_mode=trigger.execution_mode,
        )
        emitter.inc_failure(reason="handler_exception")
        try:
            await fires.finalize_failed(
                sub.conversation_id,
                fire_id,
                error=str(exc),
                latency_ms=None,
            )
        except Exception as finalize_exc:  # noqa: BLE001 - best-effort
            log.warning(
                "webhook_receive finalize_failed raised",
                extra={"extra_data": {"fire_id": str(fire_id), "error": str(finalize_exc)}},
            )
        return WebhookReceiveResult(
            status_code=500,
            fire_id=fire_id,
            message=f"dispatch failed: {exc}",
        )

    # Finalize on success path (the dispatcher returns a typed
    # WakeDispatchResult; the platform writes the terminal row here
    # because the webhook receiver owns the per-fire write window).
    try:
        await fires.finalize_success(
            sub.conversation_id,
            fire_id,
            status=result.status,
            output_text=result.output_text,
            latency_ms=result.latency_ms,
            display_suppressed=result.display_suppressed,
        )
        await subs.record_fire(
            sub.conversation_id,
            subscription_id,
            fired_at=receive_at,
        )
    except Exception as exc:  # noqa: BLE001 - finalize boundary
        log.warning(
            "webhook_receive finalize_success raised",
            extra={"extra_data": {"fire_id": str(fire_id), "error": str(exc)}},
        )
        # We already dispatched; surface a 500 so the receiver knows
        # post-dispatch persistence failed even though the wake ran.
        return WebhookReceiveResult(
            status_code=500,
            fire_id=fire_id,
            message=f"post-dispatch persist failed: {exc}",
        )

    log.info(
        EVENT_WEBHOOK_RECEIVED,
        extra={
            "extra_data": {
                "subscription_id": str(subscription_id),
                "conversation_id": str(sub.conversation_id),
                "fire_id": str(fire_id),
                "status": result.status,
                "execution_mode": trigger.execution_mode,
                "delivery_target": trigger.delivery_target,
            }
        },
    )
    emitter.inc_webhook_received(outcome="accepted")
    emitter.inc_fire(
        status=result.status,
        schedule_type="webhook",
        execution_mode=trigger.execution_mode,
    )
    return WebhookReceiveResult(
        status_code=202,
        fire_id=fire_id,
        message="dispatched",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_payload(payload_bytes: bytes) -> Any:
    """Decode the HTTP body. Tries JSON first; falls back to UTF-8 text.

    Webhook senders typically post JSON; non-JSON payloads (raw text,
    form-encoded) reach the agent as the ``event`` variable's string
    form. Malformed UTF-8 raises ``ValueError`` so the receiver can
    surface a 400.

    :param payload_bytes: raw HTTP body
    :ptype payload_bytes: bytes
    :return: decoded JSON object/array/scalar, or a string fallback
    :rtype: Any
    :raises ValueError: when neither JSON nor UTF-8 decode succeed
    """
    if not payload_bytes:
        return {}
    try:
        decoded = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"payload is not valid UTF-8: {exc}"
        raise ValueError(msg) from exc
    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        # Sender posted plain text; expose it as a string under
        # ``{{event}}``. The template author can render verbatim or
        # ignore.
        return decoded


async def _count_recent_fires_for_subscription(
    *,
    fires_collection: WakeFireCollection,
    subscription_id: UUID,
    conversation_id: UUID,
    since: float,
    pool: Any,
) -> int:
    """Count fires for ``subscription_id`` in the given window.

    The :class:`WakeFireCollection` ``count_in_window`` method is
    conversation-scoped, not subscription-scoped, so we issue a
    targeted COUNT here. Kept as a module-level helper rather than
    folded into the Collection because the per-subscription
    rate-limit query is a webhook-receiver concern, not part of the
    fire collection's general API.

    Implementation note: we hit the ``idx_wake_fires_conv_time`` index
    via the ``conversation_id`` predicate + filter on
    ``webhook_subscription_id`` server-side.

    :param fires_collection: only used to suppress lint warnings about
        unused params; reserved for future cache-aware paths
    :ptype fires_collection: WakeFireCollection
    :param subscription_id: target subscription
    :ptype subscription_id: UUID
    :param conversation_id: partition column (lets the query hit the
        per-conv index)
    :ptype conversation_id: UUID
    :param since: lower bound on ``actual_fired_at`` as a POSIX
        timestamp
    :ptype since: float
    :param pool: asyncpg-compatible pool
    :ptype pool: Any
    :return: number of fires in the window
    :rtype: int
    """
    del fires_collection  # reserved for future cache integration
    if pool is None:
        return 0
    since_dt = datetime.fromtimestamp(since, tz=UTC)
    value = await pool.fetchval(
        "SELECT COUNT(*) FROM wake_fires "
        "WHERE conversation_id = $1 AND webhook_subscription_id = $2 "
        "AND actual_fired_at >= $3",
        conversation_id,
        subscription_id,
        since_dt,
    )
    return int(value or 0)
