# agent-wake-06: Channels webhook receiver framework

> **REMOVED 2026-05-24:** the outbound delivery framework was removed as an undesigned parallel abstraction. `WebhookReceiver` no longer takes a `delivery_adapters` argument and no longer imports `DeliveryAdapter`. The inbound webhook receiver itself (HMAC verification, subscription CRUD, rate-limit, payload templating) is UNCHANGED and fully supported — only the delivery-adapter wiring is gone. Wake fires always deliver into the conversation. The text below retains the `delivery_adapters` plumbing for history — do NOT rebuild it.

## 2026-05-19 revision deltas (apply BEFORE implementing)

Canonical source: `<metallm>/docs/long_running/PLACEMENT.md`.

**Minor changes only:**
- When constructing the `WakeTrigger` from an inbound webhook, set `trigger.skill_id = subscription.default_skill_id` (if any). PLACEMENT §1.1.
- No other changes — `WebhookReceiver` framework (HMAC, subscription CRUD primitive, rate-limit, payload templating with sandboxed Jinja2) is unchanged in shape.

## Objective

Land the generic inbound webhook receiver in `3tears-channels`,
alongside the existing Slack / Discord / WebSocket adapters. The
receiver:

- Accepts POST at a configurable mount point (consumer registers
  `WebhookReceiver` on their FastAPI / Starlette app).
- Decodes the request body + signature header.
- Calls into `3tears-agent-wake`'s `webhook_receive(...)` adapter from
  shard 04 — which does the verification, rate-limit, trigger
  construction, dispatch.
- Returns the HTTP response shape (202 / 401 / 403 / 404 / 429 / 500)
  to the sender.

The receiver framework is platform; the FastAPI router file mounting
it is product (metallm's `api/src/api/v1/webhooks.py` shim is one
line). Future products mount their own.

This shard's home in `3tears-channels` (not `3tears-agent-wake`) is
deliberate: `3tears-channels` already hosts inbound-message adapters
(Slack, Discord, WebSocket); a generic HTTP webhook adapter fits
naturally alongside them. The wake-trigger construction logic lives
in `3tears-agent-wake.webhook_adapter` (shard 04); this shard owns
the HTTP routing-and-response plumbing.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| WEBHOOK-01 | New module `packages/channels/src/threetears/channels/webhook.py` exporting `WebhookReceiver` class with `register(app, mount_path: str = "/webhooks")` method that adds a `POST {mount_path}/{subscription_id}` route. | P0 |
| WEBHOOK-02 | The receiver delegates to `threetears.agent.wake.webhook_receive(...)` (shard 04) for the verify+rate-limit+dispatch flow. It does NOT reimplement HMAC verification or rate-limit logic. | P0 |
| WEBHOOK-03 | HTTP response mapping: `webhook_receive` returns a `WebhookReceiveResult`; the framework maps to FastAPI/Starlette `JSONResponse(status_code=result.status_code, content={"fire_id": str(result.fire_id) or None, "message": result.message})`. 429 includes a `Retry-After` header pointing to the window rollover. | P0 |
| WEBHOOK-04 | `WebhookReceiver` is constructed with `(pool, encryption_service, handler, wake_config)` (~~the `delivery_adapters` arg is REMOVED 2026-05-24~~). All wiring happens at consumer construction time; no global state. | P0 |
| WEBHOOK-05 | Signature header name is configurable; default `"X-3Tears-Webhook-Signature"` (NOT `X-MetaLLM-Signature` — the platform doesn't carry the consumer's brand). metallm's existing endpoints set the override to `"X-MetaLLM-Signature"` for backwards-compat. | P0 |
| WEBHOOK-06 | `verification_scheme` from the subscription row determines the verification path. v1 ships `'generic_hmac_sha256'` only. Future vendor adapters (`'github'`, `'stripe'`, `'slack'`) extend by registering a verifier callable in `_VERIFIERS: dict[str, Verifier]`. | P0 |
| WEBHOOK-07 | Source IP detection follows the existing 3tears reverse-proxy convention: read from `X-Forwarded-For` header (first hop), fall back to socket address. Existing convention shared across Slack/Discord adapters. | P0 |
| WEBHOOK-08 | Payload decoding: try `json.loads(payload_bytes)` first; on JSONDecodeError, treat as opaque string. `webhook_receive` always receives the raw bytes; the framework passes them through. The Jinja template render in `webhook_receive` reads either the decoded dict OR the raw string into `event`. | P0 |
| WEBHOOK-09 | Request size cap: 1 MiB. Anything larger returns 413 without invoking `webhook_receive`. Configurable via the constructor (`max_payload_bytes`). | P0 |
| WEBHOOK-10 | Integration test: receiver mounted on a test FastAPI app, POST with valid HMAC signature → 202 + fire_id; POST with invalid signature → 401; POST exceeding size cap → 413. | P0 |
| WEBHOOK-11 | The `WebhookReceiver` does NOT host the subscription CRUD endpoints. Those are agent-tool surfaces (shard 04 tools) + the product's REST router. The receiver is the receive-side only. | P0 |
| WEBHOOK-12 | Future-vendor signature schemes documented as extension points: register via `WebhookReceiver.register_verifier(scheme_name, verifier_func)`. Verifier signature: `Callable[[secret: bytes, payload: bytes, headers: dict[str, str]], bool]`. Vendor-specific schemes add zero core changes. | P0 |

---

## Why `3tears-channels` not `3tears-agent-wake`

Considered:

- **`3tears-agent-wake`** — rejected. The wake package's job is the
  wake runtime (schema, tick, dispatch, agent tools). The webhook
  receiver is one source of wake triggers but is its own HTTP-adapter
  concern. Mixing HTTP routing into the wake package's surface area
  conflates "receive inbound HTTP" with "manage wake schedules."
- **A new `3tears-webhook` package** — considered. Could work. The
  argument for is "small clean package for the receive primitive."
  The argument against: `3tears-channels` already hosts
  inbound-message adapters with the same shape (Slack/Discord adapters
  receive payloads, verify, route to a handler) — adding HTTP webhook
  as another adapter is symmetrical. We don't introduce a new package
  unless the boundary is clear, and "HTTP-inbound channel" is the same
  bounded context as "Slack-inbound channel."
- **`3tears-channels`** — chosen. Symmetric to existing adapters.

Flagged in the placement memo as a confidence assessment item — Pace
can override.

---

## API specification

```python
# packages/channels/src/threetears/channels/webhook.py
from __future__ import annotations

import hmac
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final
from uuid import UUID

from asyncpg import Pool
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from threetears.observe import get_logger

from threetears.agent.wake.config import WakeConfig
from threetears.agent.wake.types import HandlerCallback  # DeliveryAdapter REMOVED 2026-05-24 — no outbound delivery framework
from threetears.agent.wake.webhook_adapter import webhook_receive

__all__ = ["WebhookReceiver", "Verifier"]

log = get_logger(__name__)

_DEFAULT_SIGNATURE_HEADER: Final[str] = "X-3Tears-Webhook-Signature"
_DEFAULT_MAX_PAYLOAD_BYTES: Final[int] = 1 << 20  # 1 MiB

Verifier = Callable[[bytes, bytes, dict[str, str]], bool]
"""(secret, payload_bytes, headers) -> True if valid"""


def _verify_generic_hmac_sha256(secret: bytes, payload: bytes, headers: dict[str, str]) -> bool:
    sig_header = headers.get(_DEFAULT_SIGNATURE_HEADER.lower(), "")
    if not sig_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret, payload, sha256).hexdigest()
    return hmac.compare_digest(expected, sig_header)


class WebhookReceiver:
    """Generic HMAC-verified webhook receiver.

    Routes POST {mount_path}/{subscription_id} into
    threetears.agent.wake.webhook_receive. Consumers register the receiver
    on their FastAPI app and supply the handler / wake_config / encryption
    service.
    """

    def __init__(
        self,
        *,
        pool: Pool,
        encryption_service: Any,
        handler: HandlerCallback,
        wake_config: WakeConfig,
        # delivery_adapters: dict[str, DeliveryAdapter] REMOVED 2026-05-24 — no outbound delivery framework
        signature_header: str = _DEFAULT_SIGNATURE_HEADER,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        self._pool = pool
        self._encryption_service = encryption_service
        self._handler = handler
        self._wake_config = wake_config
        # self._delivery_adapters REMOVED 2026-05-24 — no outbound delivery framework
        self._signature_header = signature_header
        self._max_payload_bytes = max_payload_bytes
        self._verifiers: dict[str, Verifier] = {
            "generic_hmac_sha256": _verify_generic_hmac_sha256,
        }

    def register_verifier(self, scheme: str, verifier: Verifier) -> None:
        """Add or override a verification scheme for vendor adapters."""
        self._verifiers[scheme] = verifier

    def register(self, app: FastAPI, *, mount_path: str = "/webhooks") -> None:
        app.add_api_route(
            f"{mount_path}/{{subscription_id}}",
            self._handle,
            methods=["POST"],
            tags=["webhooks"],
        )

    async def _handle(self, subscription_id: UUID, request: Request) -> Response:
        body = await request.body()
        if len(body) > self._max_payload_bytes:
            return JSONResponse(
                status_code=413,
                content={"message": "payload too large"},
            )

        signature = request.headers.get(self._signature_header)
        source_ip = self._resolve_source_ip(request)

        # The agent-wake adapter does the rest: subscription lookup,
        # verification, source-IP check, rate-limit, trigger build, dispatch.
        result = await webhook_receive(
            subscription_id=subscription_id,
            payload_bytes=body,
            signature_header=signature,
            source_ip=source_ip,
            pool=self._pool,
            encryption_service=self._encryption_service,
            handler=self._handler,
            wake_config=self._wake_config,
            # delivery_adapters REMOVED 2026-05-24 — no outbound delivery framework
        )

        headers: dict[str, str] = {}
        if result.status_code == 429:
            headers["Retry-After"] = "60"
        return JSONResponse(
            status_code=result.status_code,
            content={
                "fire_id": str(result.fire_id) if result.fire_id else None,
                "message": result.message,
            },
            headers=headers,
        )

    def _resolve_source_ip(self, request: Request) -> str | None:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return None
```

---

## Patterns to Follow

- Existing inbound-channel adapter shape: `3tears-channels` Slack / Discord adapters (look at the protocol → handler dispatch shape).
- FastAPI route registration on a constructed object: existing patterns in metallm's `main.py:include_router(...)`. The receiver's `register(app)` is the analog for "I'm a channel that adds routes to a host app."
- HMAC `compare_digest`: existing `3tears` security utilities pattern.
- `X-Forwarded-For` parsing: existing 3tears reverse-proxy convention (look for it in Slack adapter source).

---

## Files to Create

### Inside `3tears-channels`

- `packages/channels/src/threetears/channels/webhook.py` — `WebhookReceiver`, `Verifier`, `_verify_generic_hmac_sha256`.
- `packages/channels/tests/unit/test_webhook_verification.py` — generic HMAC verifier unit tests (valid, invalid, missing header, malformed header).
- `packages/channels/tests/integration/test_webhook_receiver.py` — mount on a test FastAPI app; POST scenarios for 202 / 401 / 413 / 429 / 404. Uses a stubbed `webhook_receive` to test the routing-and-response plumbing.
- `packages/channels/src/threetears/channels/__init__.py` — export `WebhookReceiver`.

### Inside `3tears-channels` package metadata

- `packages/channels/pyproject.toml` — add `3tears-agent-wake>=0.9.0` to deps (channels now depends on agent-wake for the adapter target). This creates an edge: `channels → agent_wake → conversations + agent_skills`.

---

## Implementation Notes

1. **Lazy import of `webhook_receive`.** To avoid circular-import edge cases during package init, the framework can lazy-import `webhook_receive`:

   ```python
   def __init__(...):
       ...
       from threetears.agent.wake.webhook_adapter import webhook_receive  # lazy
       self._webhook_receive = webhook_receive
   ```

   In practice, `3tears-channels` declares `3tears-agent-wake` as a runtime dependency, so the import-time edge is fine. Use whichever pattern keeps mypy happy.

2. **The dependency edge `channels → agent-wake` is acceptable.** It's a one-direction edge (agent-wake doesn't depend on channels). No cycle. The alternative — agent-wake depending on channels — would be worse.

3. **`max_payload_bytes` default 1 MiB.** Most webhook payloads are <100 KB. 1 MiB headroom; configurable per receiver. Larger means bigger memory pressure during HMAC compute (the entire body sits in memory during verify).

4. **The receiver does NOT batch.** One request = one fire. Burst handling is via the per-subscription rate-limit (shard 04's `webhook_receive`).

5. **`Retry-After` default 60s.** Could be smarter (calculate exact rollover) but 60s is a reasonable heuristic. Document as a known simplification.

6. **`signature_header` configurable for backwards-compat.** metallm's existing webhook receiver uses `X-MetaLLM-Signature`; on bump, metallm passes that as the override. The platform default is `X-3Tears-Webhook-Signature` so new consumers don't inherit metallm's brand.

7. **`verification_scheme` extension.** Subscription row carries the scheme name; the receiver looks up the verifier in `self._verifiers`. New vendor adapters (GitHub `X-Hub-Signature-256`, Stripe `Stripe-Signature`, etc.) register their verifier function at receiver construction time:

   ```python
   receiver.register_verifier("github", verify_github_signature)
   ```

   These adapters can live in `3tears-channels` as additional modules (`webhook_github.py`, etc.) or in product packages.

8. **No body parsing inside the receiver.** The receiver passes raw bytes to `webhook_receive`. The adapter (shard 04) is responsible for `json.loads(payload_bytes)` if it wants structured access. Keeps signature verification semantics correct: the signature is over the exact bytes received, not a re-serialized version.

9. **No request logging at info level.** Webhook payloads may contain secrets / PII. Debug-level only, with credential-shaped field redaction.

10. **Integration test uses a real test FastAPI app.** Mount the receiver, send `httpx.AsyncClient`-based requests, assert status codes + response bodies. Stub `webhook_receive` to return canned `WebhookReceiveResult`s for the routing tests; use the real `webhook_receive` against a testcontainer for the end-to-end test.

---

## Anti-patterns

- DO NOT reimplement HMAC verification in the receiver. Delegate to the registered verifier function.
- DO NOT parse the JSON payload before verification. Sign-then-parse; otherwise an attacker can craft payloads that succeed in the verification step but fail the parse, leaking timing info.
- DO NOT log request bodies. PII risk.
- DO NOT couple the receiver to a specific FastAPI version. Use the `add_api_route` API; works across versions.
- DO NOT hardcode the mount path. `register(app, mount_path=...)` is configurable.
- DO NOT add subscription CRUD endpoints to this module. Subscription CRUD is the consumer's REST router's job (it imports the Pydantic models from `3tears-agent-wake.api_models` + delegates to `WebhookSubscriptionCollection`). The receiver is receive-side only.
- DO NOT batch multiple fires per request. One POST = one fire.
- DO NOT make `Retry-After` dynamic without a clear win. The 60s default is fine for v1.
- DO NOT verify the signature against a non-constant-time compare. `hmac.compare_digest` is mandatory.

---

## Success Criteria

- [ ] `WebhookReceiver.register(app)` adds the route at the configured mount path.
- [ ] Valid HMAC POST → 202 + `{fire_id, message}`.
- [ ] Invalid HMAC POST → 401.
- [ ] Missing signature header → 401.
- [ ] Body over `max_payload_bytes` → 413 (no `webhook_receive` invocation).
- [ ] Source IP from `X-Forwarded-For` first hop when present.
- [ ] Rate limit hit → 429 with `Retry-After: 60`.
- [ ] Vendor-specific verifier registration works via `register_verifier`.
- [ ] Integration test: end-to-end with `webhook_receive` and a real testcontainer Postgres.
- [ ] `./scripts/check-all.sh` clean.
