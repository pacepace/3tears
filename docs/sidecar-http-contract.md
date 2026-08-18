# The nodriver sidecar's HTTP contract

**Status:** decided and built (2026-08-18). The error model, the versioning rule and the
access posture are all settled — `SCR-1FK5` point 3 was ruled on the same day and is
implemented below.

**Closes:** `SCR-0HB4` (this contract had no recorded versioning or error-model decision)
and `SCR-1FK5` points 1, 2 and 3 — points 4, 5 and 6 were closed when `SCR-7VD2` shipped on
2026-07-27.

**Scope.** Port 8088 on `packages/scrape/sidecar`. That service imports nothing from
`threetears.*`, is AGPL-licensed separately from the rest of this repo, and carries its own
version — none of which changes because it now has a written contract.

---

## 1. Why this was missing, which is the part worth keeping

Chunks 05, 06 and 08 added a cross-process HTTP surface — `/v1/hitl/session`, `/v1/hitl/tab`,
`/v1/hitl/tab/{id}/complete`, `/v1/hitl/vnc`, plus `session_state` and
`egress_proxy`/`egress_name` on `RenderRequest` and `egress` on `RenderResponse` — and no
chunk carried an `**Exposed API:**` declaration. The produced-surface rule never fired, so
neither a versioning approach nor an error model was recorded for any of it.

The gap was never theoretical. `nodriver_sidecar.py` conditionally omits both new request
fields, with the comment *"so a sidecar built before this existed still accepts the payload"*.
That is a real compatibility policy — invented at one call site, applied nowhere else, and
written down in no artifact. Section 3 below promotes it to the contract it always was.

## 2. The error model — DECIDED, and it was broken

**Every error response on this service is:**

```json
{"error": {"code": "<stable machine-readable>", "message": "<human, may be reworded>"}}
```

`code` is the stable half and the only part a client may branch on. `message` is for a human
and can change without notice.

**What it was.** Three shapes on one `/v1` surface:

| Routes | Shape |
|---|---|
| `/v1/render`, `/v1/download` | `{"error": {"code", "message"}}` |
| every `/v1/hitl/*` | `{"error": "<free text>"}` |
| any request failing validation | FastAPI's `{"detail": [...]}` |

So a client could not write one error handler, and the machine-readable half was missing from
precisely the routes that hand back a human's solved session — the ones whose failures a
caller most needs to distinguish. Normalised 2026-08-18; `main._error()` is now the only
constructor, and a sweep test drives the session-scoped routes into failure and asserts the
shape rather than trusting a list someone must remember to extend.

**FastAPI's `{"detail": ...}` remains** for request-validation failures. Left deliberately: it
is emitted by the framework before any handler runs, and intercepting it repays little — a
malformed request is a caller bug, not a runtime condition to branch on. A client should treat
a 422 without an `error` key as "I sent something malformed".

### The code vocabulary is open, not closed

A client meeting an unrecognised `code` must fall back on the status, not reject the response.
This is the same rule, one level down, that made `structured_kind` a `str` rather than a
`Literal` (`stream-protocol-structured-results.md`): a closed vocabulary means a reader
predating a new member *rejects* a response it could have handled.

## 3. Versioning — DECIDED

**`/v1` is a real version, and only a BREAKING change bumps it.** Breaking means: removing a
route, removing a response field, changing a field's type, changing the meaning of an existing
`code`, or making a previously-optional request field required.

**These are NOT breaking and ship without a bump:**

- Adding a route.
- Adding a response field.
- Adding an **optional** request field.
- Adding a new `error.code` value (see the open-vocabulary rule above).

**What a client is entitled to assume when a field it sends is ignored.** This is the policy
`nodriver_sidecar.py` invented at one call site, now stated once for everyone:

> **The sidecar ignores request fields it does not recognise, and never fails a request for
> carrying one.** A client may therefore send a newer payload to an older sidecar and get the
> older behaviour — *not* an error. What it may **not** assume is that the field took effect.
> Where that distinction matters, the response says so: `RenderResponse.egress` reports the
> exit actually used, so a client that sent `egress_name` to a sidecar too old to honour it
> sees `null` rather than a silent direct fetch.

That last sentence is the load-bearing one. "Ignore unknown fields" is only safe when the
response can be asked what actually happened; a field whose effect is unobservable must not be
added under this rule.

### The 2026-08-18 error-shape change did NOT bump `/v1`, and here is why

Changing `error` from a string to an object on the HITL routes is, by the rule stated
directly above, a **breaking change to a field's type**. It shipped inside `/v1` anyway.
That is an exception, and it is written down rather than left as a silent inconsistency in
the first document to state the rule.

The reasons, in order of weight:

1. **There are no deployed consumers.** The sidecar ships as a container image that no
   automated pipeline had ever published, and no application uses it. A migration path
   protects users who exist.
2. **`/v2/hitl` alongside `/v1/render` would fragment the surface permanently** to fix an
   inconsistency, which is a worse outcome than the inconsistency.
3. **The service carries its own version, and that is the signal that moved.** The sidecar
   went `0.1.0` → `0.2.0` in the same change. It is deliberately outside the family
   lockstep precisely so it can say "I broke something" without a framework release.

**This exception does not generalise.** Once anything depends on this API, a type change
bumps `/v1` as the rule says. The exception is available exactly while the consumer count
is zero, and that will not be true twice.

**Version skew is expected to be small but real**: the sidecar ships as a container on its own
cadence and is not part of the family lockstep, so a deployment can have a newer library
against an older sidecar. The rules above are what make that survivable.

## 4. Access posture — RECOMMENDATION, needs a ruling

This is `SCR-1FK5` points 1–3. **It does not reopen the no-identity decision**, which stands:
the sidecar authenticates nobody, deciding *who should hold a token* happens where identity
lives, and `docs/scrape-task-08-hitl-vnc-and-fetch-health.md` §6 struck the in-package RBAC
gate as an error of layer. What follows is about what the surface is *worth*, so whoever
fronts it knows what they are protecting.

### What reaching 8088 is currently equivalent to

`POST /v1/hitl/session/{id}/tab/{tab_id}/complete` returns **the raw cookie jar and storage of
a human's solved session, unsealed** — the sidecar holds no key, by design. So reaching this
port is equivalent to holding a browser plus a target's authenticated session.

The one control that exists is a **binding, not a policy**: compose publishes
`127.0.0.1:8088:8088`, but `entrypoint.sh` runs `uvicorn --host 0.0.0.0`, so any other
container on the same compose network reaches it by service name with nothing in the way. On
Kubernetes the pod boundary answers this; under compose — which is what this repo ships — it
does not. The session token only proves the caller holds something *this container minted*.

### Point 1 — per-route posture. DECIDED: three tiers, not one gate

A single allow/deny over the port is the wrong shape, because these routes are not worth the
same:

| Tier | Routes | Why |
|---|---|---|
| **Public within the pod** | `GET /healthz` | No data. Must stay reachable by orchestration probes that hold no token. |
| **Capability-gated** | `POST/GET/DELETE /v1/hitl/vnc`, `POST /v1/hitl/session`, `/v1/render`, `/v1/download` | Drive a browser. Already gated by the minted token where a session exists. |
| **Sensitive** | `GET /v1/hitl/session/{id}`, `POST .../tab`, `POST .../complete`, `DELETE /v1/hitl/session/{id}` | Return or act on a solved session's credentials. `.../complete` is the one that hands back the cookie jar. |

`SessionManager` and its token check were deliberately **kept** when `SCR-7VD2` shipped —
removing the last capability check from endpoints that return raw cookie jars is a security
reduction. That decision belongs to this point and is hereby settled: **the token check stays
on the sensitive tier regardless of what fronts the port.**

### Point 2 — what a minted token entitles its holder to. DECIDED contract

> A session token is a **capability over exactly one session**: it permits driving that
> session's display, opening tabs within it, completing it (which discloses its cookie jar),
> and closing it. It carries **no identity**, is valid until `expires_at`, is not
> transferable in any way the sidecar can detect, and confers nothing over any other session
> or over the container.

Stating it this way is what lets a platform gate against it: the platform decides *who* may be
issued one, and the sidecar honours it as a bearer capability and nothing more.

### Point 3 — the container-network path. DECIDED 2026-08-18: safe by default, widened out loud

`entrypoint.sh` bound `0.0.0.0` unconditionally, so "everything on this network is trusted"
was an assumption nothing stated and nothing enforced.

**The bind address is now `BIND_HOST`, defaulting to `127.0.0.1`.**

- **On Kubernetes** — the shipping shape since `SCR-7VD2` — the front-door container shares
  the pod's network namespace, so the loopback default is exactly right and nothing outside
  the pod can reach 8088 at all. A real reduction in exposure, taken by default.
- **Under compose**, each container has its own namespace, so a loopback bind makes the port
  unreachable even through a published port. `docker-compose.yml` therefore sets
  `BIND_HOST=0.0.0.0` explicitly, with a comment naming what that admits: any other container
  on the network reaches a port that mints session tokens and returns raw cookie jars.

Binding loopback unconditionally was considered and rejected. It does not merely restrict
compose, it breaks compose outright, and a default that cannot work in one of the two
supported shapes is not a security improvement — it is an outage with a security-shaped
justification. Making the wider bind **explicit and annotated** is what converts an inherited
assumption into a stated one, which is what this point asked for.

## 5. What is deliberately not decided here

- **Rate limiting and quotas.** No route has any. Out of scope for a contract document; it is
  a deployment concern until a consumer reports a need.
- **Who may be issued a token.** Belongs to the platform, permanently. §6 of the HITL design
  doc is the authority.
