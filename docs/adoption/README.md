# 3tears -- Adoption Guide

This is the entry point for deciding what to adopt from 3tears and why. Each
module has its own doc in this directory: the problem it solves, its design
philosophy, when to adopt it, and what it composes with. This file is the map.

Written for both humans and AI systems evaluating the platform. Every module
doc follows the same fixed structure so it can be parsed as reliably as it can
be read.

## What 3tears is

3tears is a **Python library suite, not a service**, requiring **Python
3.14+**. There is no daemon, container, or API to deploy for 3tears itself.
You `pip install` the packages you need and construct objects inside your
host application's process. It brings infrastructure dependencies --
PostgreSQL always, NATS JetStream once you scale past one pod -- but those
are services you stand up and own. 3tears just talks to them.

It ships as ~26 independently-versioned packages under one `threetears.*`
import namespace, each installable and each pinned on its own. Take the whole
stack, or take one package and ignore the rest. Nothing in the family requires
the rest of the family.

## The problem it solves

Two problems, not one. First: a new app, script, or agent gets built against
ad hoc plumbing -- a hand-rolled SQLite cache, a bespoke tool-dispatch loop,
one-off RBAC checks -- because building it from primitives is slower than
building it from scratch. Second: when that app succeeds and needs to scale
past one process, the ad hoc plumbing doesn't scale, and state-scatter,
cache-drift, and cross-pod coordination set in. Most projects solve problem
one fast and dirty, then pay for it at problem two.

3tears is built to solve both with the same primitives, not one at the cost
of the other. Start a project against `core` and it behaves like a normal
local data layer -- L1 SQLite, no NATS connection, no pod concept. Nothing
about that early code has to be rewritten when the project grows into a
multi-pod deployment: you add a NATS deployment and pass an L2 client into
the same `CollectionRegistry`. The three-tier design isn't scaffolding you
build later -- it's the shape the code already has.

At scale, the second problem takes over: state scatters across pods, caches
drift out of sync, cross-pod coordination becomes a second job. Add an LLM
agent on top and it compounds -- the agent needs memory, tools, RBAC, and
model access that all have to stay coherent across every pod running it.

3tears turns your data into three-tier cached objects -- L1 in-process SQLite,
L2 NATS JetStream KV, L3 PostgreSQL -- so every pod reads local and fast, every
write flows through to durable storage, and every pod stays coherent without
hand-rolled invalidation. Agent primitives (memory, tools, RBAC, model
adapters, channel integrations, LangGraph checkpointing) are built on that same
foundation, not bolted onto it.

## The core mental model

> **3tears makes no connections. The host supplies every external dependency
> by dependency injection.**

3tears never opens a connection, reads an environment variable, or installs a
log handler on your behalf. You construct the backing clients -- a PostgreSQL
pool, optionally a NATS client -- and hand them to 3tears through one
dependency-injection seam, the `CollectionRegistry`. 3tears owns none of your
infrastructure's lifecycle. You do.

A read walks L1 -> L2 -> L3 and re-promotes on the way back. A write goes to
L3 (the source of truth), then promotes into L1 and L2, then broadcasts a
cross-pod invalidation. A miss at any tier falls through to the next tier --
the stack degrades gracefully rather than failing. Losing every cache is a
performance event, not a correctness event, because L3 is the only tier that
is ever authoritative.

Single pod? Skip L2 entirely and run L1 + L3 -- you don't connect a NATS
server, though `3tears-nats` still ships as a package dependency of `core`
either way. Add a real NATS deployment only when you scale out.

**Migrations compose across packages, but registration is manual.** Every
package that owns schema exports a `register(runner)` function from its
`migrations` subpackage. A host app installing more than one such package
must import and call each installed package's `register()` against one
shared `MigrationRunner` before running it -- there's no auto-discovery.
See [`docs/how-to-add-a-migration.md`](../how-to-add-a-migration.md) for the
full pattern.

Full wiring detail, decision tables, and copy-pasteable code: see
[`docs/integration-guide.md`](../integration-guide.md).

## Platform-wide design principles

These recur across modules. Treat them as defaults, not suggestions, when
extending or integrating with 3tears:

1. **No implicit connections; dependency injection only.** 3tears never
   opens a socket, reads an environment variable, or constructs an
   infrastructure client on its own. The host owns lifecycle, secrets, and
   logging destinations for everything it hands in. (The mechanism -- one
   seam per concern -- is principle 4.)
2. **The durable tier is the source of truth; caches are disposable.**
   Reads are read-through (a miss at L1 or L2 falls through to L3 and
   promotes the result back up); writes are write-through (a write commits
   to L3 first, then promotes into L1/L2). Neither tier is ever written to
   independently of L3. This reasoning applies beyond L1/L2/L3 -- it is the
   same principle behind every cache-invalidation and reload path in the
   platform.
3. **Graceful degradation.** A missing or unhealthy dependency narrows to a
   specific failure mode and falls through, rather than taking the whole path
   down. Transport errors degrade; programming errors still surface.
4. **Explicit dependency injection.** One configuration seam per concern,
   configured once. No package reaches out and constructs its own
   infrastructure client behind your back.
5. **Compose existing primitives before building new ones.** A new package
   justifies itself only if the capability genuinely does not exist yet
   elsewhere in the platform.
6. **Do the structurally-best thing.** Backward compatibility is a
   consideration, not a constraint that blocks a better design. This
   platform is built and consumed with AI-assisted development, where
   updating every call site of a changed interface is fast and mechanical
   rather than slow and risky -- so a breaking change is weighed against
   that lower cost, not against a hand-maintained codebase's cost. A
   consumer is expected to update, not shielded indefinitely from a better
   design.
7. **Fire-and-forget side channels never break the caller.** Audit,
   telemetry, and cache invalidation are side effects. A failure in one of
   them must never fail the operation that triggered it.
8. **Default deny, explicit grant.** Authorization and RBAC paths fail closed
   -- a missing actor, missing namespace row, or ambiguous state is treated
   as denied, never as allowed.
9. **One blessed path per concern.** One migration runner, one audit
   envelope, one ACL evaluator, one NATS client. Where the platform already
   has a canonical way to do something, packages use it instead of
   reimplementing it.
10. **Python is the root language, not an assumption baked into the
    design.** 3tears ships as Python libraries; every integration today is
    Python. A non-Python integration isn't architecturally excluded -- it
    would be a purpose-built, thin client for that language (talking to the
    same wire protocols, not a port of the Python object model), added only
    when a real consumer justifies it. Python stays canonical; other
    languages are additive, not a rewrite target.

These principles are enforced, not just stated: `3tears-enforcement` runs
static-analysis scanners across the ecosystem that check pattern compliance
at commit time.

## The four families

| Family | Packages | Role |
|---|---|---|
| **Core data** | `core`, `conversations`, `datasources` | The three-tier entity/collection/caching layer and the data model every other family builds on |
| **Infrastructure** | `nats`, `observe`, `epoch`, `mcp`, `registry`, `scheduled-jobs`, `media-contracts`, `enforcement`, `backup`, `object-store` | Cross-cutting platform services: transport, telemetry, coherence, tool routing, scheduling, storage, and static verification |
| **Agent framework** | `agent-tools`, `agent-memory`, `agent-skills`, `agent-workspace`, `agent-acl`, `agent-audit`, `agent-wake`, `agent-identity`, `agent-intention`, `agent-knowledge` | Everything an LLM agent needs to act, remember, and evolve safely at scale |
| **Models, channels, LangGraph** | `models`, `channels`, `langgraph` | The surface that connects an agent to LLM providers, chat channels, and LangGraph orchestration |

## Module index

### Core data

| Module | Solves |
|---|---|
| [`core`](core.md) | Three-tier entities and collections. `DataStore`, schema, migrations. The foundation everything else sits on. |
| [`conversations`](conversations.md) | Owns the shared `conversations` table so no single consumer package has to. |
| [`datasources`](datasources.md) | One canonical model for "what is a datasource," with a swappable driver abstraction. |

### Infrastructure

| Module | Solves |
|---|---|
| [`nats`](nats.md) | A single, mistake-proofed NATS client so every package and host app shares one wrapper. |
| [`observe`](observe.md) | Structured logging, tracing, and correlation context, silent until the host opts in. |
| [`epoch`](epoch.md) | Cross-pod cache coherence for in-memory config, combining push and pull. |
| [`mcp`](mcp.md) | A shared Model Context Protocol server framework with per-tool RBAC baked in. |
| [`registry`](registry.md) | Multi-pod tool discovery and load-balanced call routing over NATS. |
| [`scheduled-jobs`](scheduled-jobs.md) | A generic, multi-pod-safe scheduling core with zero domain concepts baked in. |
| [`media-contracts`](media-contracts.md) | Zero-dependency contracts that decouple media providers from media consumers. |
| [`enforcement`](enforcement.md) | Shared static-analysis scanners that enforce architectural invariants at commit time. |
| [`backup`](backup.md) | Encrypted, GFS-rotated, restore-verified database backups to any object store. |
| [`object-store`](object-store.md) | Streaming S3-compatible storage for large binary artifacts. |

### Agent framework

| Module | Solves |
|---|---|
| [`agent-tools`](agent-tools.md) | Tool definition, dispatch, and audit for LLM agents. |
| [`agent-memory`](agent-memory.md) | Extraction, hybrid retrieval, and salience-ranked long-term memory for agents. |
| [`agent-skills`](agent-skills.md) | Procedural memory: reusable, labeled agent skills and invocation history. |
| [`agent-workspace`](agent-workspace.md) | Sandboxed workspace entities and format handlers for agent file work. |
| [`agent-acl`](agent-acl.md) | One RBAC evaluator so every consumer answers authorization identically. |
| [`agent-audit`](agent-audit.md) | One audit envelope and wire format across every domain. |
| [`agent-wake`](agent-wake.md) | Schedules, fires, and dispatches wakes so long-running agents can resume themselves later. |
| [`agent-identity`](agent-identity.md) | Versioned, consent-gated self-evolution of an agent's own identity. |
| [`agent-intention`](agent-intention.md) | A standing-wants corpus so an agent can hold a want without repeating itself. |
| [`agent-knowledge`](agent-knowledge.md) | Governed-knowledge retrieval and injection for LangGraph agents. |

### Models, channels, LangGraph

| Module | Solves |
|---|---|
| [`models`](models.md) | One factory surface for LLM/embedding providers, with circuit breakers and usage tracking. |
| [`channels`](channels.md) | Write agent logic once, deliver it across Slack, Discord, and WebSocket. |
| [`langgraph`](langgraph.md) | Three-tier LangGraph checkpointing that scales the same on a laptop or a fleet. |

## How to adopt

There is no required order and no all-or-nothing install. A few common
entry points:

- **Just the cache.** Install `core`. `nats` comes along as a dependency
  either way; you only need a real NATS deployment once you run more than
  one pod.
- **Add an LLM agent to an existing 3tears app.** Add `agent-tools` and
  `agent-memory` first -- most other agent packages depend on one or both.
  You still need [`models`](models.md) separately for the actual LLM
  provider call -- it has no dependency link to or from the agent-framework
  packages, so it won't turn up by following "Composes with" chains from
  them.
- **Expose your app's tools over MCP.** Add `mcp`. `registry` is a separate,
  not-pre-wired multi-pod routing layer -- add it yourself if you need
  routing on top of MCP tools, don't assume the two packages talk to each
  other already.
- **You are not building an agent at all.** `core`, `nats`, `observe`, and
  `epoch` are useful as a general-purpose horizontally-scalable data layer on
  their own.

Every module doc states its dependencies under "Composes with." Follow that
chain to find the minimal install for what you need -- but a few packages
(`models` notably) are meant to be added independently rather than pulled in
transitively; each doc calls this out where it applies.

## Testing your integration

3tears ships reusable pytest fixtures for local integration tests --
`threetears.core.testing.fixtures` spins up Postgres (and pgvector, via a
parametrized image) in testcontainers, and skips cleanly when Docker isn't
available. Reuse these in your own `conftest.py` rather than hand-rolling a
Postgres fixture per consuming app. Full setup: `docs/integration-guide.md`,
"Local integration testing" (§8.1).

If your app defines its own test fake standing in for a 3tears protocol --
an ACL evaluator, a store client, a workspace shell -- declare what
production protocol it stands in for: a subclass, a `# parity-with:` marker
comment, or an explicit exemption. This is the same fake-parity discipline
3tears enforces on itself (see this repo's own `CLAUDE.md`, "Test Fakes");
it's what keeps a fake from silently rotting out of sync with the real
protocol it's supposed to mirror.

## External references

Curated links for the underlying technology 3tears sits on, not 3tears
itself:

| Technology | What it's for here | Docs |
|---|---|---|
| PostgreSQL | L3, the source of truth | <https://www.postgresql.org/docs/> |
| pgvector | Semantic search for `agent-memory` / `agent-knowledge` | <https://github.com/pgvector/pgvector> |
| NATS JetStream | L2 cache, cross-pod pub/sub, audit and epoch transport | <https://docs.nats.io/nats-concepts/jetstream> |
| OpenTelemetry | Tracing backend for `observe` | <https://opentelemetry.io/docs/> |
| Model Context Protocol | The protocol `mcp` implements | <https://modelcontextprotocol.io/> |
| LangGraph | Agent orchestration `langgraph` checkpoints into | <https://docs.langchain.com/oss/python/langgraph/overview> |
| S3 API | The protocol `object-store` speaks (any S3-compatible backend) | <https://docs.aws.amazon.com/AmazonS3/latest/API/Welcome.html> |

## Keeping this guide current

This directory is maintained by a saved prompt, not by hand-editing on
recall. See [`PROMPT.md`](PROMPT.md) for the update procedure -- run it any
time packages are added, removed, or meaningfully changed, and whenever this
guide is suspected to be stale.
