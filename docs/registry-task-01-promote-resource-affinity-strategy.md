# registry-task-01: Promote resource-keyed soft-affinity routing into 3tears

**Status:** DRAFT — design pending. The *trigger* is real (second adopter); the *exact shape Scriob consumes* is still being designed (see "Open / pending" below). Do not start until the Scriob affinity design settles.
**Scope:** `3tears-registry` (`routing.py` — joins the existing `RoutingStrategy` Protocol + `LeastConnectionsStrategy`), promoting an implementation that today lives in a product. Net-new framework code + a product re-export.
**Origin:** Two adopters now want the same "route requests for a resource back to its warm home pod" behavior, which triggers the workspace's promotion rule (promote out of a product into 3tears when a second package needs it):
1. **aibots** built it product-side as `LeastStickyConnectionsStrategy` + `PodAffinityCollection` (`14-eng-ai-bot/src/aibots/agent_router/routing.py`) — routes a conversation back to the pod with its warm L1 cache. Keyed hard to `conversation_id`; L1+L2 only ("ephemeral by design", L3 raises); cross-pod coherence via the `threetears.cache.invalidate` stream; LRU-bounded.
2. **Scriob** wants the same shape for connection/conversation warmth on its WebSocket layer.

---

## Objective

Promote aibots' soft sticky-affinity routing into `3tears-registry` as a **generic, resource-keyed** strategy implementing the existing `RoutingStrategy` Protocol — parameterized by an arbitrary resource key (not hardcoded to `conversation_id`) — plus the backing affinity collection as a reusable, L1+L2-only `BaseCollection`. aibots' impl becomes a re-export (its tests are the behavior contract the generalized version must not break).

## Design constraints (the promotion must stay single-purpose)

- **Keep it SOFT.** This is a *cache-warmth optimization*, lossy-OK: losing affinity costs a cold cache, never correctness. It must NOT acquire a lease or any exclusivity. (The hard, exclusive single-owner case — e.g. a git working tree — is a **different** primitive, `nats_distributed_lock`; do not fold the two together. Conflating soft affinity with a hard lease is the exact "kitchen-sink" outcome this promotion must avoid.)
- **Strip product policy from mechanism.** aibots' version carries LRU sizing + `conversation`-specific assumptions + "L3 raises loudly". The generalized version exposes the *mechanism* (resource-key → home pod, soft, ephemeral); policy (LRU bound, what the key means) is the caller's. If policy cannot be cleanly separated, that is 3tears telling you it is not general yet — stop.
- **Reuse the existing seam.** The framework already ships `RoutingStrategy(Protocol)` + `LeastConnectionsStrategy` (`registry/routing.py`). The promoted strategy is a peer implementation, selectable via the existing `CallProxy` `routing_strategy` injection — not a new subsystem.

## Open / pending (why this is DRAFT, not READY)

The Scriob affinity design is mid-flight and changes what Scriob actually consumes here:
- Scriob's **connection/chat** layer is **reconcile-anywhere** (local sockets + NATS fanout + shared NATS-KV state — metallm's `ConnectionManager` pattern). Soft sticky affinity is an *optimization* on top, not required for correctness — so Scriob's dependence on this strategy is real but **not load-bearing**.
- Scriob's **load-bearing** affinity is the **repo working-tree lease** (`nats_distributed_lock`, repo-as-actor) — a *different* primitive, explicitly NOT this.
- A genuinely **net-new** Scriob need surfaced that may belong here too (or in channels): a **queryable cross-pod presence roster** (collaborative "who is in (branch,file) room X"), which products built as 1:1 chat never needed — likely an ephemeral L1+L2 `BaseCollection` keyed by room, the same shape as `PodAffinityCollection`. Decide during the Scriob design whether the roster and the affinity strategy share machinery or stay separate.

Confirm the second-adopter case is the *soft routing strategy specifically* (not just "Scriob uses affinity somewhere") before promoting — otherwise this is speculative generality.

## Acceptance (when promoted)
- Generic resource-keyed strategy + affinity collection in `3tears-registry`, single-purpose (no lease coupling), conforming to `RoutingStrategy`.
- aibots' `LeastStickyConnectionsStrategy`/`PodAffinityCollection` reduced to a thin re-export; aibots' existing tests green unchanged (behavior contract).
- Policy (LRU/sizing/key meaning) is caller-supplied, not baked in.
