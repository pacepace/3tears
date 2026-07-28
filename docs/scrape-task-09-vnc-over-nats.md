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
in, which tenant's display they can see, and which network zone they may reach into.

**The zone half is not a later refinement.** The deployment has pod sets with different network
reach -- some inside a firewall with visibility of targets nobody else can resolve, some with plain
internet egress -- and only certain customers are entitled to certain zones. Section 5 shows that
zone entitlement and tenant entitlement are the SAME role assignment rather than two mechanisms,
which is why the requirement states both here rather than treating one as a special case of the
other.

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

**The session claim is blocked the same way.** `KVLease` defaults its bucket suffix to `leases`,
which the transport prefixes to `{ns}-leases`
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

**Routing cannot see who is calling, so it cannot be the zone mechanism.**
`threetears.registry.routing.RoutingStrategy` declares `select(self, endpoints: list[ToolEndpoint])
-> ToolEndpoint | None` and receives nothing else: not the caller, not the customer, not the call
arguments. `ToolEndpoint` (`packages/registry/src/threetears/registry/catalog.py`) carries
`pod_id`, `status`, `in_flight` and `date_last_heartbeat`, with no labels or attributes. A
zone-aware router would therefore need a widened endpoint model, a breaking change to a shipped
Protocol, and a policy engine inside a load balancer.

**A tool's registered name is the seam, and it is documented as one.**
`ScrapeTool.mcp_name()` returns `3tears.scrape`, and its docstring states that "a consuming wrapper
that registers it is free to override this with its own namespaced name".
`threetears.agent.tools.server.tool_namespace_name` turns that into
`tools.<mcp_name>.<version>` with dots sanitized to dashes.

**A pod cannot register a tool it is not entitled to.**
`threetears.registry.registration.RegistrationHandler._authenticate_and_filter` verifies the pod's
presented identity token against its stored key, then filters `manifest.tools` against
`pod_auth.allowed_namespaces` by prefix match. Tools outside those namespaces are dropped and
logged with the pod name and the rejected list, and a manifest with nothing left is refused
outright with "no tools authorized for this pod's namespaces".

**A tool's registered name is completely unvalidated, and both sanitizers only handle dots.**
`ToolManifestEntry.name` is a bare `str` with no pattern.
`RegistrationHandler._validate_manifest` checks exactly two things: `pod_id` non-empty and `tools`
non-empty. Meanwhile `threetears.nats.subjects._sanitize` and
`threetears.core.namespaces.sanitize_segment` are both `value.replace(".", "-")` and nothing more,
so a space, a `*` or a `>` in a tool name survives into whatever is built from it. This is the
hazard `Subjects.forward` and `Subjects.room` already document and already solve by hashing.

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
| Ownership with compare-and-swap renewal | `threetears.core.coordination.KVLease` | `{ns}-leases` ungranted to tool pods |
| Which pod owns a display | `threetears.scrape.operator_session.claim_session` | none |
| Control messages to that pod | `threetears.scrape.operator_control` | none, once `forward` is granted |
| Operator page, noVNC, WebSocket route | `threetears.scrape.operator.build_operator_router` | `relay_stream` hardcodes a TCP transport |
| Hub-side subscribe-then-forward shape | `aibots.hub.router.stream_bridge` | its unbounded queue is wrong for bytes; reuse the shape, not the buffering |
| Authenticated, metered, audited call into a tool pod | `aibots.hub.ingress.dispatch_core` -> `CallProxy` | none; the attach rides it |
| WebSocket on the hub | `app.add_api_websocket_route` | none |
| Roles, groups, members, assignments, evaluation, cache invalidation | `threetears.agent.acl` | action vocabulary and a tenant-scoped namespace type |
| Namespace naming and tenant column | `threetears.core.namespaces`, `namespaces.customer_id` | no HITL namespace type or plural prefix |
| Round-trip and live-grant test harness | `packages/nats/tests/integration/`, `threetears.core.testing.fixtures` | none |
| Zone as a tool identity | `ScrapeTool.mcp_name()` override seam, `tool_namespace_name` | none; a consuming wrapper names its own tool |
| Refusing a pod that claims the wrong tool | `RegistrationHandler._authenticate_and_filter`, `allowed_namespaces` on the tool-pods row | none |

---

## Design

### 1. The pipe primitive

A new module beside `packages/nats/src/threetears/nats/forward.py`, in `3tears-nats`, payload-agnostic in exactly the way `forward`
is. It carries bytes; it interprets none of them.

**Rendezvous reuses `forward` verbatim.** The caller sends an attach request to the key's forward
subject. Whichever pod holds the key answers with the concrete stream coordinates:

```
attach  -> {"op": "attach", "credit": <bytes>, "max_chunk": <bytes>}
reply   <- {"tool": "tools.scrape-zone_alpha.1-0-0", "pod_id": "...", "nonce": "...",
            "max_chunk": <bytes>, "credit": <bytes>}
```

**The owner states which tool it is serving, and cannot usefully lie about it.** The caller derives
`{tool_digest}` by hashing this value, so it needs the readable form and the owner is the
authoritative source. Carrying the READABLE name rather than the digest is deliberate: it is what
makes a trace correlatable, and it costs nothing because the caller hashes it anyway.

An owner naming a tool it does not serve gains nothing. Its publish grant is
`pipe.{digest of its own tool}.{its own pod_id}.>` -- an exact literal minted from the tool-pods
row -- so a subject built from a false name is one it cannot publish on, and the stream dies at the
first frame instead of crossing a boundary. The property is worth stating because it is what makes
a self-declared field safe HERE and would not make one safe elsewhere.

The owner mints the nonce per attach. Using the nonce rather than the application key in the
stream subject buys two things: a stale reconnect cannot land on a live stream, and an
application key (a session id, a tenant-bearing string) never rides a subject where a wildcard
subscriber could enumerate it.

**Streaming rides pod-id-led subjects**, for the reason `gateway_stream` and `hub_stream` already
state:

```
{ns}.pipe.{tool_digest}.{pod_id}.{nonce}.down    owner publishes, caller subscribes
{ns}.pipe.{tool_digest}.{pod_id}.{nonce}.up      caller publishes, owner subscribes
```

`{tool_digest}` is `sha256hex(tool_namespace_name)` -- see section 5 for why it is a digest of the
tool identity rather than a readable zone name, and why that is what makes the whole scheme need no
naming convention.

`pod_id` is the authenticated segment, and an owner's publish grant names its OWN pod id exactly
rather than a wildcard -- the same shape `tools.internal.{pod_id}` already uses, and the reason a
session-id-led subject would be wrong (the wildcard publish grant the caller side needs would let
any pod paint frames onto any session).

**What the `{tool_digest}` segment does and does not buy, stated honestly.** The strong isolation
is the exact pod-id grant above; the digest adds nothing to it. What it buys is a SUBSCRIBE grant
separable per pod set (`pipe.a3f9c2....>` held apart from another tool's), so a future per-zone
gateway is a grant change rather than a subject migration -- and, because the digest is computable
from the tool-pods row's `allowed_namespaces`, a pod's publish grant can be an EXACT literal with
no wildcard in it at all. It is cheap now and expensive to retrofit, which is the argument for it.

What it explicitly does NOT buy is legibility: `pipe.a3f9c2...` tells a reader nothing about which
zone they are looking at. That cost is real, it is the same cost `Subjects.room` already accepted,
and the mitigation is the same -- the readable tool identity rides in the attach reply and in every
log line, never in the subject.

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
between the prefix and the digest, so `{ns}.forward.{sha256hex(family)}.{sha256hex(key)}` is
grantable per family. The family is supplied by the CALLING MODULE as a deployment constant rather
than by a request-time caller -- but it is NOT trusted input, because it is derived from a
registered tool name that nothing validates, which is exactly why the builder hashes it. `Subjects.forward(key)` keeps its current shape and behaviour
for every existing caller; the scoped variant is additive, and the two never collide because one
carries a single segment after `forward` and the other carries two.

**`Subjects.forward_scoped` hashes the family too, rather than accepting a ready-made token.** As
built, the family is a raw string and the builder digests it, so `[0-9a-f]`-only is a property of
the one classmethod every caller goes through rather than a convention each call site is trusted to
follow. That matters precisely because the family carries an unvalidated tool name (below): a caller
allowed to precompute the token could put a space, a `*` or a `>` into the subject, and into the
GRANT minted from it. It is the same reason the digest sits inside `Subjects.forward` and
`Subjects.room` rather than at their call sites. `Subjects.forward_scoped_wildcard(family)` renders
the grant, `{ns}.forward.{sha256hex(family)}.*`, from the same derivation -- exact on the family,
wildcard on the key alone. A NATS wildcard matches a whole token, so there is no partial-token form
available here: the family segment is either one exact literal or `*`.

**The family carries the serving tool's identity, not just the concern.** For the operator surface
it is `Subjects.hitl_forward_family(tool_namespace_name)`, which is `hitl-` followed by the tool's
registered namespace name, hashed by the builders into the subject's first segment. A flat `hitl`
family would let a pod serving one tool join the queue group for another tool's session control
subject -- and `serve_owner` queue-groups on the subject deliberately, so that is not eavesdropping
but a SHARE OF THE MESSAGES.

The damage from the flat form is bounded rather than catastrophic, and the bound is worth
recording so nobody over-corrects later: the interloper would have to already know the session id
(the token is a digest it cannot enumerate), and `serve_session` re-checks `claim.held` on every
message against a KV lease it does not hold, so it can only refuse messages, not act on them. That
is a denial of service on a live human session rather than a breach. It is free to close now and
awkward to close once callers exist.

**[DECISION: add a family segment to the forward subject family, carrying a digest of the serving
tool's namespace, rather than granting `{ns}.forward.>` | the coarse grant makes every owner-routed
key in the namespace serveable by any tool pod, and the key digest leaves nothing else to
discriminate on | user can veto and accept the broad grant]**

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

### 5. Network zones, tenancy, and why they are one decision

The deployment this has to serve is not one undifferentiated pool of scrape pods. Some pod sets sit
inside a firewall with reach to targets nobody else can see; others have plain internet egress. Only
certain customers may use certain zones.

**A zone is a tool identity, not a pod attribute.** Each pod set registers a DIFFERENT tool:

```
pod set A  ->  scrape.zone_alpha   ->  namespace  tools.scrape-zone_alpha.1-0-0
pod set B  ->  scrape.zone_beta    ->  namespace  tools.scrape-zone_beta.1-0-0
pod set C  ->  scrape.internet     ->  namespace  tools.scrape-internet.1-0-0
```

The seam already exists and is already documented as one: `ScrapeTool.mcp_name()` returns
`3tears.scrape` and says a consuming wrapper is free to override it with its own namespaced name.

**Why this and not endpoint labels plus a smarter router.** `RoutingStrategy.select` receives only
the endpoint list -- no caller, no customer, no target -- and `ToolEndpoint` carries no attributes.
A zone-aware router would mean widening the endpoint model, a breaking change to a shipped
Protocol so it can see call context, and a customer-to-zone policy engine living inside a load
balancer. That last one is the defect: authorization and routing become two facts that can
disagree, and the disagreement is invisible until it routes wrong.

With zone-as-tool, the registry only ever sees endpoints for the tool that was actually called, so
every endpoint in that list is in the right zone by construction. Routing stays a load balancer.

**Entitlement is then one role assignment, and it is the shipped `ToolCaller` shape.** A role
carrying `{"tool": ["tool.call"]}`, assigned to a group at
`scope_namespace_id = tools.scrape-zone_alpha.1-0-0`. Customer X's people and agents are in that
group; customer Z's are not. Zone entitlement and customer entitlement collapse into the same fact,
evaluated by the same evaluator, with no second mechanism to keep in sync.

**And a pod cannot claim a zone it was not given.**
`RegistrationHandler._authenticate_and_filter` verifies the pod's identity token against its stored
key and then filters its manifest against `allowed_namespaces` on the tool-pods row. A
general-internet pod misconfigured to register `scrape.zone_alpha` has those tools dropped
server-side and logged, and is refused entirely if nothing survives. The enforcement is on the row,
not on the pod's honesty.

**The display inherits both facts, and this is where the naive shape is wrong.** A HITL session
must NOT be authorized against the scrape tool's own namespace: `tools.<mcp_name>.<version>` is a
platform-scoped row with `customer_id=NULL`, which is precisely why the `ToolCaller` seed had to be
namespace-scoped. Authorizing there would let one customer's operator group attach to another
customer's live display, and that display is a browser holding the target's authenticated session.

Nor is a customer-only namespace enough. `hitl.<customer_id_hex>` isolates tenants but NOT zones:
an operator entitled to customer X on the general internet could attach to customer X's zone-alpha
display, which is a reach into the firewalled network they were never granted.

So the session namespace derives from BOTH, in the shape the platform already uses for
`memories.<agent_id_hex>.<customer_id_hex>`:

```
hitl.<tool-namespace-name minus its `tools.` prefix>.<customer_id_hex>

e.g.  tools.scrape-zone_alpha.1-0-0  ->  hitl.scrape-zone_alpha.1-0-0.7f3c...
```

Spelled out rather than left as "the tool segment", because the version is part of it and an
earlier phrasing here left that ambiguous. The paragraph below on version granularity says why it
has to be.

One `authorize()` call then carries both facts: you may attach to a zone-alpha display, and only
for your own tenant. The rows carry `customer_id`, so `type_customer`-scoped role assignments do
the tenant isolation the evaluator already performs everywhere else, with no bespoke check anywhere
on the path.

**The subject token is DERIVED from the tool identity, and it is a digest.** This is the decision
that makes everything above need no naming convention, so it is recorded rather than left implied.

`{tool_digest} = sha256hex(tool_namespace_name)`, e.g. `tools.scrape-zone_alpha.1-0-0` becomes a
64-character hex token. Deriving rather than configuring means there is no second identifier that
can disagree with the tool a pod actually registered -- the same argument that rejected endpoint
labels above. Hashing rather than sanitizing is what removes the convention: tool names are
UNVALIDATED (`ToolManifestEntry.name` is a bare `str`; `_validate_manifest` checks only that
`pod_id` and `tools` are non-empty) and both sanitizers replace dots and nothing else, so a name
containing a space, a `*` or a `>` would produce an illegal subject or, worse, inject a wildcard
into one. A digest is `[0-9a-f]` only, collision-resistant and deterministic, so every pod and the
hub derive the same token from the same tool. It is the identical move
`Subjects.forward` and `Subjects.room` already make, for the identical reason.

**And it makes the grant exact rather than wildcarded.** The hub mints a pod's permissions at
connect time from the `allowed_namespaces` list on its tool-pods row -- the same list that filters
its registration. It can hash each entry and emit literal subject grants, so a pod's publish
permission names precisely the tools it was authorized to serve and nothing else. One source of
truth feeding both registration filtering and subject permissions.

**[DECISION: derive the subject token as a digest of the tool's namespace name, rather than
configuring a zone per pod or sanitizing the tool name into the subject | configuring adds a second
identifier that can disagree with the registered tool; sanitizing an unvalidated name is a
wildcard-injection hazard the existing forward/room builders already document and avoid | user can
veto in favour of validating `mcp_name` against a strict pattern and using it raw, which buys
legible subjects at the cost of constraining a shipped public surface]**

**The subject takes the digest; the NAMESPACE does not.** These are different surfaces with
different constraints, and treating them the same would be the mistake:

```
subject     {ns}.pipe.a3f9c2....{pod_id}.{nonce}
namespace   hitl.scrape-zone_alpha.1-0-0.<customer_id_hex>
```

A namespace name is a database row value. Subject-safety does not apply to it, and legibility is
worth having in a table a human administers. This is the same split `Subjects.room` already makes:
digest in the subject, raw identity in the wire envelope.

**The namespace carries the tool VERSION, because entitlement does.** `tools.scrape-zone_alpha.1-0-0`
is its own namespace row with its own role assignments. If the session namespace dropped the
version, a version bump would produce the mismatch where somebody can call the tool but not attach
to the display it raised, or the reverse. Matching the granularity is what keeps the two grants
telling the same story.

**Operators are scoped the same way**, which falls out rather than needing design: an operator's
group needs the `hitl.attach` role on the zone's session namespace, so a zone-beta operator simply
has no assignment that reaches a zone-alpha display.

**What is NOT verified, and should be known.** Nothing confirms that a pod registering
`scrape.zone_alpha` is ACTUALLY placed on the zone-alpha network. `allowed_namespaces` proves it was
authorized to claim that identity; the network placement is a chart and NetworkPolicy fact. That is
the same trust model as the rest of the deployment, but it means the chart entry and the
`tool_pods` row must be provisioned as a pair, and a mismatch is silent. A later guard could have a
zone's pod assert its egress address at registration and have the hub check it against the zone's
expected range; that is not in this scope.

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
- Verifying that a pod registering a zone's tool is genuinely placed on that zone's network. The
  registration filter proves entitlement to the identity, not the placement; see the closing note
  of section 5 for the guard that would close it.

---

## Files to create / modify

**`3tears-nats`**
- new `packages/nats/src/threetears/nats/pipe.py` -- the primitive
- `packages/nats/src/threetears/nats/subjects.py` -- `Subjects.pipe`, scoped `forward`
- `packages/nats/src/threetears/nats/subject_permissions.py` -- `_tool_pod` and `_hub` grants,
  plus the lease bucket for `_tool_pod`, which is `{ns}-leases`: `KVLease` returns the bare suffix
  and `NatsClient.kv_bucket` layers the connection's `{ns}-` over it. Written out because the wrong
  spelling fails SILENTLY -- a pod that cannot open the bucket is handed `lease=None` and serves the
  display unclaimed. `test_forward_grants_live.py` proves the grant against a real broker, and
  `tests/enforcement/test_kv_bucket_grant_naming.py` pins the grant and the lease's own default as a
  pair so they cannot drift.

  **Do not infer a naming rule from that.** Three conventions are live: `kv_bucket()` prefixes a
  suffix, a direct `js.key_value(bucket=...)` prefixes nothing, and a component that builds its own
  `f"{namespace}_thing"` name and creates it directly owns that name verbatim. A grant's spelling
  cannot tell you which applies, so a chunk touching grants reads the opener, never the shape
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
  two-container pod, and says nothing about naming a per-zone tool via the `mcp_name()` override

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
- **Authorizing the display against a customer-only namespace.** Also section 5, and a subtler
  version of the same defect: it isolates tenants while leaving every zone reachable by anyone
  entitled to that tenant anywhere.
- **Conflating egress with network zone.** `EgressDriver` (task-08 section 7) is "which exit does
  this request leave by" -- TOR, WARP, a proxy -- chosen PER REQUEST, inside a pod, from a caller's
  argument. A zone is where a pod is PLACED on the network, decided at deploy time. If zone ever
  becomes caller-selectable the way egress is, a customer can ask for a zone they are not entitled
  to, and the entitlement in section 5 stops meaning anything.
- **Putting an unvalidated tool name into a subject.** Tool names are not validated anywhere and
  both sanitizers replace only dots, so a name carrying a space, a `*` or a `>` produces an illegal
  subject or injects a wildcard into one. Hash it, the way `Subjects.forward` and `Subjects.room`
  already do. The same applies to any future app-supplied value that reaches a subject.
- **Adding labels to `ToolEndpoint` so the router can pick a zone.** That is the design section 5
  rejected, and its cost is that authorization and routing become two facts that can disagree.
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
9. An operator entitled to a customer in one zone is refused that SAME customer's display in
   another zone, from the same evaluator. This is the criterion a customer-only namespace passes
   criterion 8 while failing, which is why it is written separately rather than folded in.
10. A pod presenting a manifest for a tool outside its `allowed_namespaces` has that tool rejected
    at registration, so a zone cannot be joined by a pod that was not granted it.
11. A tool whose name contains a space, a `*` or a `>` yields a subject token of `[0-9a-f]` only,
    and the permission minted for it is an exact literal rather than a wildcard. The hostile name is
    the test input, not a hypothetical: nothing validates tool names, so this is the property that
    has to hold rather than a convention that has to be followed.

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
4. **RESOLVED 2026-07-28, kept here because the reasoning outlives the question.** Whether the
   subject's zone token is derived or configured: DERIVED, as a digest of the tool's namespace name
   (section 5 carries the decision). The concern that deriving forces a naming convention is what
   selected the digest over a sanitized name -- a digest constrains nothing, because it launders any
   name into `[0-9a-f]`. One consequence to accept knowingly: a single pod serving two tools has two
   digests and therefore two subject families, so "one pod, one zone" stops being an assumption the
   code can make. That is a property rather than a limitation, but it is not free -- anything that
   wants to enumerate "this pod's streams" has to enumerate per tool.
