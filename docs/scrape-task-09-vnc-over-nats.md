# scrape-task-09: A byte pipe over NATS, and the operator's reach into a tool pod

**Status:** design, ready to build. Successor to
[`scrape-task-08`](scrape-task-08-hitl-vnc-and-fetch-health.md), which shipped the operator
surface in v0.20.0. Parent backlog item: `SCR-1FK5` (points 1, 2 and the residue of 3 remain
open there; this document does not reopen the no-identity decision recorded in task-08 section 6).

**Every claim in "Verified, not assumed" was read out of running code this session.** Anything
reasoned rather than read is in "What is assumed" and carries its mitigation.

---

## The requirement

An operator must be able to reach a live VNC display that is running inside a Kubernetes tool
pod, from a browser, over TLS, with the platform's existing identity and RBAC deciding who gets
in and which tenant's display they can see.

Today they cannot, and the reason is structural rather than a missing configuration.
`relay_stream` reaches the display with `asyncio.open_connection(host, port)` -- a TCP connect to
loopback. That requires the process terminating the operator's WebSocket to share a network
namespace with `x11vnc`. In a deployed cluster it does not: the display lives in a tool pod, and
a tool pod has no Service, no Ingress and no inbound path of any kind. It is an egress-only NATS
client by design.

`relay_stream`'s own docstring anticipated this and named the alternative: relay the same bytes
to the pod that holds the display, over NATS, keyed on the session id. That alternative is now
the only door, so the trade it described ("settle it on measurements") no longer has two sides.

**The deliverable is not a VNC feature.** It is a general-purpose, payload-agnostic byte pipe
between any caller and whichever pod owns a key, riding the same owner-routing that
`threetears.nats.forward` already provides for request/reply. RFB is its first consumer. The
second is obvious and already wanted: an interactive shell into a tool pod, which has the
identical no-inbound-path problem.

---

## Verified, not assumed

**The display path is loopback-only, and that is deliberate.**
`packages/scrape/sidecar/hitl.py` binds `x11vnc` to `127.0.0.1` on `_RFB_PORT = 5900` and its
module docstring states that the binding *is* the access control. `relay_stream`
(`packages/scrape/src/threetears/scrape/operator.py`) opens a plain TCP connection to it.

**A tool pod has no inbound surface.**
`.devops/argocd/dev/app/templates/tool-pods.yaml` in `14-eng-ai-bot` renders a Deployment with
optional `ports`, no Service and no Ingress. The only `Ingress` templates in that chart are the
hub's and the admin website's. Tool pods self-mint a NATS connect JWT from their own Ed25519 key
and dial out; nothing dials in.

**The existing control plane is already blocked by permissions, before any of this.**
`threetears.nats.Subjects.forward` renders `{ns}.forward.{sha256hex(key)}`. No principal in
`packages/nats/src/threetears/nats/subject_permissions.py` grants publish or subscribe on that
family -- not `_tool_pod`, not `_hub`, not `_registry`. So `operator_control.serve_session`, and
with it `open_tab` / `complete_tab` / `close_session` / `read_state`, is denied by auth-callout in
any deployment that enforces these grants.

**The session claim is blocked the same way.** `KVLease` defaults its bucket to `{ns}_leases`
(`packages/core/src/threetears/core/coordination/lease.py`). `_tool_pod`'s `kv_buckets` grant is
`({ns}-proxy_assertion_nonces,)` alone. A platform that cannot open the bucket passes `lease=None`,
and `claim_session` then logs its two-pods-can-serve-one-display warning and yields anyway.

**NATS publish has no backpressure, and the client already measures the consequence.**
`packages/nats/src/threetears/nats/client.py` sets an explicit bounded `pending_size`
(`DEFAULT_PENDING_SIZE_BYTES`) and `flusher_queue_size` (`DEFAULT_FLUSHER_QUEUE_SIZE`), counts
consecutive outbound overflows on `overflow_events`, and folds a sustained run of them into
`is_healthy` so a `/healthz` trips and the pod is restarted. What it does not expose is any
per-subscription pending limit: `NatsClient.subscribe` takes `subject`, `cb`, `queue`,
`max_in_flight` and `deadletter_on_failure`, and nothing else.

**A stream subject leads with its authenticated segment.** `Subjects.gateway_stream` and
`Subjects.hub_stream` both put the authenticated principal id first, and both docstrings state
why: a bare `stream.*` grant would let one principal sniff or inject onto a peer's in-flight
stream.

**The hub already terminates WebSockets and already owns the one ingress path.**
`14-eng-ai-bot`'s `src/aibots/hub/app.py` registers `add_api_websocket_route("/ws/chat/{agent_slug}")`.
That repo's `src/aibots/hub/ingress/dispatch_core.py` documents itself as the ONE ingress-to-tool-mesh
dispatch path: it mints the identity-bound `CallContext` plus PoP, wraps a `ProxyCallRequest`, and
originates through the registry `CallProxy` -- which is where `LimitGuard` and
`EndpointUsageEmitter` are already attached.

**The RBAC engine that role-group-people needs is already shipped.**
`packages/agent/acl/src/threetears/agent/acl/` carries `GroupCollection`,
`GroupMemberCollection`, `RoleCollection` (permissions shaped `{resource_type: [action, ...]}`),
`RoleAssignmentCollection`, `NamespaceCollection`, the `evaluate_decision` evaluator, an
`AclCache` and a NATS invalidation bus. `authorize()` takes `action: str`, so the action
vocabulary is data.

**Namespaces carry the tenant boundary, and platform tool namespaces deliberately do not.**
The `namespaces` table declares `customer_id` (nullable, immutable).
`14-eng-ai-bot`'s `src/aibots/hub/security/bootstrap_rbac.py` records that platform web-tool
namespaces materialize
with `customer_id=NULL`, which is why the built-in `ToolCaller` grant had to be `namespace`-scoped
rather than `type_customer`-scoped.

**A latent tool-pod fan-out exists and this is the workload that would trip it.**
`packages/agent/tools/src/threetears/agent/tools/server.py` subscribes
`Subjects.tools_internal(pod_id)` with no queue group, reasoning that the subject is pod-specific
so "only this pod's connection binds them".
That repo's `tests/integration/test_scale_tool_pod_distribution.py` states the premise that
makes it true: "N pods register DISTINCT `pod_id`s". The chart supplies one literal `toolPodId`
per Deployment to every replica. Every tool pod runs `replicaCount: 1` today, so this is latent,
not live.

**The test shapes this needs already exist.**
`packages/nats/tests/integration/test_forward_round_trip.py` is the round-trip precedent, and
`packages/nats/tests/integration/test_user_jwt_scoped_grant_live.py` is the precedent for proving
a grant admits what it should and denies what it should against a live broker, by applying the
built permission set as config-mode `authorization`. Both use
`threetears.core.testing.fixtures.nats_container` and skip cleanly without docker.

---

## What is assumed

- **[ASSUMPTION: the in-cluster NATS `max_payload` is at least a few hundred KB | LOW impact]**
  The compose configs in `14-eng-ai-bot/docker/config/nats/` set 16 MB and the prod config's
  comment says it MUST be raised above the 1 MB default. The cluster's NATS comes from a separate
  chart (`nats.nats.svc.cluster.local`) that was not read. Mitigation: the pipe's chunk size is
  negotiated at attach and defaults to `_RELAY_CHUNK_BYTES` (64 KiB), which is under even the
  untuned 1 MB default. Verify the deployed value before tuning upward.

- **[ASSUMPTION: steady-state RFB bandwidth for a human clearing a challenge is tens of KB/s, not
  megabytes per second | MED impact]** Reasoned from encoding, not measured: noVNC negotiates
  Tight/ZRLE, `x11vnc` serves them, and a person reading a challenge is a mostly-static screen
  with small dirty rectangles. The multi-MB bursts are the first frame and page loads.
  `relay_stream`'s docstring asserts the pessimistic reading ("a full-screen update is megabytes,
  continuous"). Mitigation: the credit window bounds the damage either way, and the first chunk's
  integration test drives a deliberately slow consumer so the behaviour under a bad assumption is
  the tested path rather than the surprise.

- **[ASSUMPTION: core NATS message loss on this path is confined to slow-consumer conditions |
  HIGH impact]** This is what makes core NATS acceptable instead of JetStream. It is the
  documented NATS behaviour rather than something probed here. Mitigation is the mechanism
  itself: the sequence check makes any gap, whatever its cause, a detected teardown rather than a
  silently corrupted stream. Correctness does not depend on the assumption; only the frequency of
  reconnects does.

---

## What already exists -- reuse inventory

Nothing below is to be reimplemented. Where a gap is named, the fix is an enhancement to the
listed thing, not a parallel one.

| Need | Reuse | Gap to close |
|---|---|---|
| Route to the pod owning a key | `threetears.nats.forward` / `serve_owner` | none for the rendezvous; the subject family is ungranted |
| Subject construction | `threetears.nats.Subjects` | needs a `pipe` builder; `forward` needs a scoped variant so grants can be family-scoped |
| Bounded outbound buffer, overflow telemetry | `NatsClient` `pending_size` / `overflow_events` / `is_healthy` | no per-subscription inbound limit |
| Ownership with compare-and-swap renewal | `threetears.core.coordination.KVLease` | `{ns}_leases` ungranted to tool pods |
| Which pod owns a display | `threetears.scrape.operator_session.claim_session` | none |
| Control messages to that pod | `threetears.scrape.operator_control` | none, once `forward` is granted |
| Operator page, noVNC, WebSocket route | `threetears.scrape.operator.build_operator_router` | `relay_stream` hardcodes a TCP transport |
| Hub-side subscribe-then-forward shape | `aibots.hub.router.stream_bridge` | its unbounded queue is wrong for bytes; reuse the shape, not the buffering |
| Authenticated, metered, audited call into a tool pod | `aibots.hub.ingress.dispatch_core` -> `CallProxy` | none; the attach rides it |
| WebSocket on the hub | `app.add_api_websocket_route` | none |
| Roles, groups, members, assignments, evaluation, cache invalidation | `threetears.agent.acl` | action vocabulary and a tenant-scoped namespace type |
| Namespace naming and tenant column | `threetears.core.namespaces`, `namespaces.customer_id` | no HITL namespace type or plural prefix |
| Round-trip and live-grant test harness | `packages/nats/tests/integration/`, `threetears.core.testing.fixtures` | none |

---

## Design

### 1. The pipe primitive

A new module beside `packages/nats/src/threetears/nats/forward.py`, in `3tears-nats`, payload-agnostic in exactly the way `forward`
is. It carries bytes; it interprets none of them.

**Rendezvous reuses `forward` verbatim.** The caller sends an attach request to the key's forward
subject. Whichever pod holds the key answers with the concrete stream coordinates:

```
attach  -> {"op": "attach", "credit": <bytes>, "max_chunk": <bytes>}
reply   <- {"pod_id": "...", "nonce": "...", "max_chunk": <bytes>, "credit": <bytes>}
```

The owner mints the nonce per attach. Using the nonce rather than the application key in the
stream subject buys two things: a stale reconnect cannot land on a live stream, and an
application key (a session id, a tenant-bearing string) never rides a subject where a wildcard
subscriber could enumerate it.

**Streaming rides pod-id-led subjects**, for the reason `gateway_stream` and `hub_stream` already
state:

```
{ns}.pipe.{pod_id}.{nonce}.down    owner publishes, caller subscribes
{ns}.pipe.{pod_id}.{nonce}.up      caller publishes, owner subscribes
```

`pod_id` leads because it is the authenticated segment. A session-id-led subject with the
wildcard publish grant the hub needs would let any pod paint frames onto any session.

**Framing is a fixed-width header and a body.** A monotonic per-direction sequence, and a small
tag distinguishing data from a credit acknowledgement, so an ack needs no second subject:

```
byte 0      tag: 0x00 data, 0x01 credit-ack
bytes 1..4  uint32 big-endian sequence (data) or acked-through sequence (credit-ack)
bytes 5..    body (data only)
```

**A gap is a teardown, never a delivery.** RFB has no resynchronisation, so a receiver that sees
`seq != last + 1` raises rather than skipping. This is the whole reason core NATS is safe here:
loss under slow-consumer conditions is detectable, and detection converts a permanently corrupted
stream into the clean-reconnect path the operator page already has copy for
(the `everConnected` branch in
`packages/scrape/src/threetears/scrape/operator_assets/operator.html` distinguishes a drop from a
refusal).

**Credit replaces the backpressure TCP was giving away for free.** This is the part with no
existing equivalent and the part most likely to be wrong if it is skipped. Reading from a socket
and publishing is unbounded: a slow operator does not slow the producer, it fills the producer's
pending buffer until the connection wedges. So the consumer publishes a credit-ack every half
window, and the producer stops reading its source once unacked bytes exceed the window. The fat
direction (`down`, the display) needs it; `up` carries keystrokes and pointer events and does not.

**Core NATS, not JetStream, and the reasons are specific.** JetStream would persist the pixels of
a human's authenticated session to disk; at-least-once delivery produces duplicates that corrupt
RFB exactly as badly as gaps unless deduplicated on the same sequence anyway; and the write
amplification of a continuous stream is real cost for no gained property.

### 2. Scoped forward subjects

Granting `{ns}.forward.>` to tool pods would let any tool pod serve any owner-routed key in the
namespace. The key is a SHA-256 digest of an arbitrary application string, so the subject carries
nothing a grant can discriminate on.

The fix is an enhancement to the existing builder rather than a coarse grant: a family segment
between the prefix and the digest, so `{ns}.forward.{family}.{sha256hex(key)}` is grantable per
family. `Subjects.forward(key)` keeps its current shape and behaviour for every existing caller;
the scoped variant is additive.

**[DECISION: add a family segment to the forward subject family rather than granting
`{ns}.forward.>` | the coarse grant makes every owner-routed key in the namespace serveable by
any tool pod, and the digest leaves nothing else to discriminate on | user can veto and accept
the broad grant]**

### 3. `relay_stream` gets a transport seam

`relay_stream` currently opens its own TCP connection. It gains a transport parameter whose
default preserves today's behaviour exactly, so the loopback deployment (the compose file in this
repo, and the two-container pod) is unchanged and its tests keep passing untouched.

The seam is deliberately narrow, matching the existing docstring's promise that only this
function changes: something that yields a reader-like and writer-like pair. The TCP opener is one
implementation; the pipe is the other.

### 4. Where the operator lands

The hub. It holds the only Ingress and already terminates WebSockets, and the ingest path is
where a NATS-fabric call belongs. `build_operator_router` is mounted there with a pipe-backed
transport, and the pod-side of the pipe is served by the scrape tool pod for as long as it holds
the session claim.

The attach goes through `dispatch_core`, not around it. That is what makes the capability check
inherit identity verification, RBAC, rate limiting, metering and audit rather than growing a
second copy of each.

### 5. Authorization and multi-tenancy

Role, group, members. No new engine, and the action vocabulary is data:

- A `hitl` resource type with a `hitl.attach` action, seeded the way that repo's
  `bootstrap_rbac.py` already seeds `PlatformSuperAdmin` and `ToolCaller`.
- A role carrying `{"hitl": ["hitl.attach"]}`.
- A group holding that role by assignment; people in the group.

**The tenant boundary is the namespace, and getting this wrong is the one security defect this
design can produce.** A HITL session must NOT be authorized against the scrape tool's own
namespace: `tools.<mcp_name>.<version>` is a platform-scoped row with `customer_id=NULL`, which is
precisely why the `ToolCaller` seed had to be namespace-scoped. Authorizing there would let one
customer's operator group attach to another customer's live display, and that display is a browser
holding the target's authenticated session.

So the session is authorized against a customer-scoped namespace, in the shape the platform
already uses for `memories.<agent_id_hex>.<customer_id_hex>`: a new plural prefix and namespace
type whose rows carry `customer_id`. `type_customer`-scoped role assignments then do the tenant
isolation the evaluator already performs everywhere else, with no bespoke check anywhere on the
path.

### 6. What this does not decide

The replica-identity question. One `toolPodId` per Deployment shared by every replica breaks the
"N pods register DISTINCT `pod_id`s" premise, and the identity model is one Ed25519 key per tool
pod name against one `tool_pods` row. HITL is the workload that most wants `replicaCount > 1`, so
this needs an answer, but it is a platform-identity decision rather than a transport one and it
does not block anything here at `replicaCount: 1`.

---

## Explicitly out of scope

- Any change to the no-identity position for the sidecar's own 8088 surface. Task-08 section 6
  struck the in-package RBAC gate as an error of layer and that stands. `SCR-1FK5` points 1 and 2
  own the remaining per-route posture question.
- Replacing `forward`'s request/reply semantics, or introducing a second owner-election mechanism.
- Pooling displays, or more than one Xvfb per pod.
- The tool-pod replica identity model (section 6 above).

---

## Files to create / modify

**`3tears-nats`**
- new `packages/nats/src/threetears/nats/pipe.py` -- the primitive
- `packages/nats/src/threetears/nats/subjects.py` -- `Subjects.pipe`, scoped `forward`
- `packages/nats/src/threetears/nats/subject_permissions.py` -- `_tool_pod` and `_hub` grants,
  plus the `{ns}_leases` bucket for `_tool_pod`
- `packages/nats/src/threetears/nats/__init__.py` -- lazy re-exports in the existing hand-rolled
  PEP 562 shape
- new `packages/nats/tests/unit/test_pipe.py`, new
  `packages/nats/tests/integration/test_pipe_round_trip.py`, new
  `packages/nats/tests/integration/test_forward_grants_live.py` (which covers the pipe subjects
  too, rather than a second grants file)

**`3tears` core**
- `packages/core/src/threetears/core/namespaces.py` -- the HITL plural prefix and type

**`3tears-scrape`**
- `packages/scrape/src/threetears/scrape/operator.py` -- the transport seam on `relay_stream`
- new `packages/scrape/src/threetears/scrape/operator_pipe.py` -- the pod-side display bridge
- `packages/scrape/README.md` -- the deployment section currently describes only the
  two-container pod

**`14-eng-ai-bot`** (consumer-side, planned here and built there)
- the operator router mounted on the hub with a pipe-backed transport
- the attach as a `dispatch_core` call
- `src/aibots/hub/security/bootstrap_rbac.py` -- the `hitl` role seed
- the chart -- the scrape tool pod entry

---

## Anti-patterns to avoid

- **Copying `stream_bridge`'s unbounded queue.** It is right for LLM tokens and wrong for bytes.
  Reuse its subscribe-before-dispatch ordering and its cleanup-on-disconnect, not its buffering.
- **Making the pipe know what RFB is.** It carries bytes. Anything it understood, it could get
  wrong, and RFB does not resynchronise.
- **Skipping the credit window because it works in a lab.** Loopback and a fast laptop hide the
  entire failure mode; a real operator on a bad connection is the case it exists for.
- **Proving the primitive only against a testcontainer with no authorization.** That is the
  "unit-correct parts, composed failure" learning: the subjects must be proven admitted by the
  built permission set, not merely working on a server that enforces nothing.
- **Authorizing the display against the tool namespace.** See section 5. This is the tenant leak.
- **Letting a gap be skipped rather than raised.** A receiver that tolerates a missing sequence
  turns a detectable fault into a frozen screen, which is the failure mode the whole design is
  arranged to avoid.

---

## Acceptance criteria

1. A byte stream moves between two NATS clients through the pipe, in both directions, with the
   receiver reconstructing the producer's bytes exactly.
2. A dropped or reordered chunk raises at the receiver rather than being delivered, and the error
   names the stream.
3. A consumer that stops acknowledging causes the producer to stop reading its source, proven by
   instrumenting the producer's own reads and asserting they halt at the window boundary and
   resume on the next acknowledgement.

   **Not** by asserting `NatsClient.overflow_events` stays at zero. That counter increments only
   when `OutboundBufferLimitError` is raised at the publish boundary while the client is
   disconnected or reconnecting with a full pending buffer, so on a healthy connected test it
   reads zero whether or not any credit accounting exists at all. An assertion on it would pass
   against a pipe with the whole mechanism deleted, which is the "a test that asserts a refusal
   must prove the code ran" learning wearing the costume of rigour.
4. The permission set built for a tool-pod principal admits the pipe and forward subjects that
   principal actually uses, and denies another pod's stream subjects, proven against a live broker.
5. A tool-pod principal can open the lease bucket `claim_session` needs.
6. `relay_stream`'s existing loopback behaviour is unchanged, proven by its existing tests passing
   without modification.
7. An operator reaches a real display through the hub, over TLS, end to end, and drives it.
8. An operator entitled to one customer's session is refused another customer's, and the refusal
   comes from the shipped evaluator rather than from a check written for this path.

---

## Open questions for review

1. **Package placement.** `3tears-nats` beside the existing
   `packages/nats/src/threetears/nats/forward.py` is the recommendation: same class of
   primitive, same key derivation, same owner-routing. The alternative is its own distribution,
   which buys separable release cadence and costs a cross-family dependency bound.
2. **Whether the attach should be a tool call or a forward.** Routing it through `dispatch_core`
   inherits the whole authorization and metering stack; sending it on `forward` directly is
   simpler and inherits none of it. The recommendation is `dispatch_core`, and the cost is that
   the tool-mesh call shape has to carry a stream handle in its reply.
3. **Credit window default.** Wants a number chosen against a measurement rather than taste. The
   first chunk's slow-consumer test is where that measurement comes from.
