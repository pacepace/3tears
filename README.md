# 3tears

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

3tears is a family of Python packages for apps that run on many pods and host LLM agents. Its core keeps your data in objects over three cache tiers, so every pod reads locally and writes stay coherent across the fleet.

Running an app on more than one pod scatters its state: caches drift, and keeping pods in step becomes a second job. The core handles that once:

- **L1** is SQLite, in the process: reads are local and synchronous.
- **L2** is NATS KV, shared by every pod: it keeps the pods' caches coherent.
- **L3** is PostgreSQL, or a git working tree: the durable source of truth.

The agent packages (memory, tools, RBAC, identity, skills and more) build on the same tiers, so an agent spread across pods is built the same way as one on a laptop. Each package installs on its own: take the whole family, or one package and only what it depends on.

**Contents:** [Status](#status) · [Install](#install) · [Quick start](#quick-start) · [Packages](#packages) · [Versioning](#versioning) · [Adopting a package](#adopting-a-package) · [Where to go next](#where-to-go-next) · [Development](#development) · [Contributing](#contributing) · [How it's built](#how-its-built) · [License](#license)

## Status

3tears is alpha. The `0.x` line runs in production internally, but the public API can change between minor versions until `1.0.0`, so pin the family to one version. Python 3.14+ is required.

## Install

```bash
pip install 3tears                 # the core: three-tier entities and collections
pip install 3tears-agent-memory    # or any package from the tables below
```

The core needs an L3: PostgreSQL, or a git working tree. On one pod with PostgreSQL, NATS (L2) is optional. More than one pod, or a git L3, needs NATS and `pip install "3tears[nats]"`.

## Quick start

A collection is a table seen through the three tiers, and an entity is one row of it. Reading an entity pulls it up through the tiers on a miss; writing to it is change-tracked, and `save()` writes through every tier.

```python
# users is a UsersCollection; the core README sets one up
entity = await users.get(user_id)      # L1, then L2, then L3; promoted on a miss
entity.name_display = "Alice"          # change-tracked
await entity.save()                    # written through L3, L1 and L2

users[user_id, "name_display"]         # synchronous read of one field
```

The [core README](packages/core/README.md) sets up the registry, defines a collection, and covers flush strategies, optimistic locking and the git-backed L3.

## Packages

Every package shares the `threetears.*` import namespace. The `agent-*` packages live under `packages/agent/`.

### Core

| Package | Import | What it does |
|---|---|---|
| [`3tears`](packages/core/) | `threetears.core` | Three-tier entities and collections (L1 SQLite, L2 NATS KV, L3 PostgreSQL or a git working tree), `DataStore` for dynamic tables, and `MigrationRunner` for versioned schema migrations |

### Agent framework

| Package | Import | What it does |
|---|---|---|
| [`3tears-agent-tools`](packages/agent/tools/) | `threetears.agent.tools` | Tool framework: the `TearsTool` base, `ToolServer` for NATS registration, dispatch and audit, context management, and built-in tools |
| [`3tears-agent-memory`](packages/agent/memory/) | `threetears.agent.memory` | Memory extraction, hybrid retrieval (semantic, full-text, recency) and MMR reranking |
| [`3tears-agent-knowledge`](packages/agent/knowledge/) | `threetears.agent.knowledge` | Governed-knowledge retrieval: concepts and playbook entries, injected through `before_model` middleware |
| [`3tears-agent-skills`](packages/agent/skills/) | `threetears.agent.skills` | Procedural memory: skill definitions and invocation history |
| [`3tears-agent-identity`](packages/agent/identity/) | `threetears.agent.identity` | Versioned identity blocks: the agent proposes a change, a person consents, and earlier versions are superseded, never lost |
| [`3tears-agent-intention`](packages/agent/intention/) | `threetears.agent.intention` | Standing wants: an intention lifecycle, salience decay, embedding dedup, and private deliberation tools |
| [`3tears-agent-workspace`](packages/agent/workspace/) | `threetears.agent.workspace` | Workspace entities, a sandbox, format handlers, and workspace tools |
| [`3tears-agent-acl`](packages/agent/acl/) | `threetears.agent.acl` | RBAC evaluator and cache: groups, roles, assignments, and evaluation trails |
| [`3tears-agent-audit`](packages/agent/audit/) | `threetears.agent.audit` | One audit envelope and a fire-and-forget `publish_audit`, plus the erasure rule that anonymizes audit records |
| [`3tears-agent-wake`](packages/agent/wake/) | `threetears.agent.wake` | Wake schedules, fires and webhook subscriptions for long-running agents |

### Infrastructure and integrations

| Package | Import | What it does |
|---|---|---|
| [`3tears-conversations`](packages/conversations/) | `threetears.conversations` | The per-agent `conversations` table, its entity and three-tier collection, shared by memory, tools and workspaces |
| [`3tears-datasources`](packages/datasources/) | `threetears.datasources` | Datasource entities and a driver abstraction for PostgreSQL, Redshift, Snowflake and BigQuery |
| [`3tears-object-store`](packages/object-store/) | `threetears.object_store` | Streaming S3-compatible store for large binaries (pcaps, dumps, rendered reports), kept out of the entity tiers |
| [`3tears-backup`](packages/backup/) | `threetears.backup` | Encrypted, grandfather-father-son rotated database backups to any `ObjectStore`, with restore verification |
| [`3tears-geo`](packages/geo/) | `threetears.geo` | Slippy-map tile geometry: WKB decoding, zoom bands and MVT encoding, with no PostGIS |
| [`3tears-nats`](packages/nats/) | `threetears.nats` | Typed NATS client, subject builders, and JetStream KV helpers |
| [`3tears-observe`](packages/observe/) | `threetears.observe` | Structured logging, OpenTelemetry tracing, the `@traced` decorator, and ASGI correlation middleware |
| [`3tears-epoch`](packages/epoch/) | `threetears.epoch` | Generation-stamped config epochs broadcast over NATS, so every pod reloads its caches coherently |
| [`3tears-iam`](packages/iam/) | `threetears.iam` | Passwords, OAuth2/OIDC, SAML, GitHub sign-in, session tokens, DPoP, TOTP, WebAuthn, and the anti-automation controls that guard them |
| [`3tears-registry`](packages/registry/) | `threetears.registry` | Multi-pod tool routing: a NATS KV-backed catalog, discovery, and a load-balancing call proxy |
| [`3tears-scheduled-jobs`](packages/scheduled-jobs/) | `threetears.scheduled_jobs` | Multi-pod-safe scheduled jobs: a cross-pod-locked tick loop, reschedule math, and store protocols |
| [`3tears-langgraph`](packages/langgraph/) | `threetears.langgraph` | A three-tier LangGraph checkpoint saver, agent middleware (prompt caching, summarizing, tool-result offload, tool-loop budgets), and streaming |
| [`3tears-mcp`](packages/mcp/) | `threetears.mcp` | Model Context Protocol framework: an RBAC-gated `McpServer`, `McpTool`, and an auth-aware HTTP client |
| [`3tears-tool-schema`](packages/tool-schema/) | `threetears.tool_schema` | Dependency-free helpers that make a tool's JSON Schema self-contained for models and validators |
| [`3tears-models`](packages/models/) | `threetears.models` | LangChain-native model adapters (Anthropic, OpenAI, OpenRouter, VoyageAI, Whisper, image providers, and Claude through the Claude CLI) with capability metadata, circuit breakers and usage tracking |
| [`3tears-channels`](packages/channels/) | `threetears.channels` | One message protocol with Slack, Discord and WebSocket adapters, plus email sending and an HTTP webhook receiver |
| [`3tears-search`](packages/search/) | `threetears.search` | Provider-agnostic web and media search, with SearXNG, Tavily and Brave Answers adapters |
| [`3tears-scrape`](packages/scrape/) | `threetears.scrape` | Self-healing web scraping: a model proposes extractions, each is checked against the real page, and the winner is kept as a reusable recipe |
| [`3tears-media-contracts`](packages/media-contracts/) | `threetears.media.contracts` | Dependency-free media capability contracts shared by providers and tools |
| [`3tears-evals`](packages/evals/) | `threetears.evals` | Evals for LLM features, from a classifier to a tool-using agent: each version runs over the same cases, and the report says what moved and whether the move is real. Its public API is still changing |
| [`3tears-enforcement`](packages/enforcement/) | `threetears.enforcement` | AST-based enforcement scanners and shared test utilities, run from pytest in any repo |

## Versioning

Every `3tears*` package releases at the same version, and every dependency between them is bounded to that minor line. Install them as a set, pinned to one release:

```bash
pip install "3tears==X.Y.Z" "3tears-agent-memory==X.Y.Z"   # the same X.Y.Z for every 3tears package
```

Mixing versions is not supported. The bounds make a mixed set fail at install time instead of at runtime. [docs/releasing.md](docs/releasing.md) has the reasoning and the release procedure.

## Adopting a package

Each 3tears package asks as little of its host as it can. Where a package needs something from the app around it, it asks for it as a small protocol (a **port**), and the host fills each port with an adapter of its own. Everything else is optional.

An app that already runs 3tears can build those adapters on 3tears modules instead of its own systems:

- its database, through `threetears.core`;
- tracing, through `threetears.observe`;
- model calls, through `3tears-models`.

### Evals, as the worked example

`3tears-evals` depends on no other 3tears package, only `httpx` and `pydantic`. A host writes four things: a host value, a store over its own storage, a kind that calls the code under test, and a launcher per kind. It also brings its own UI and model clients. The store, the model clients and tracing may be built on a 3tears module instead. The rest (the trial loop, judging, pricing, the report) is the engine's.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/evals-overview-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/evals-overview-light.svg">
  <img alt="The evals architecture, simplified. The host app brings its own UI, storage, tracing and model clients through small adapters, each of which may be built on an optional 3tears module (core, observe, models) instead. The evals engine depends on no other 3tears package. Its report writer loops: the analyst writes, code checks bounce a faulty draft, the editor reviews, the analyst revises, and the report is stored. Offline, a capability gap the analyst reports becomes a change in the 3tears repo that ships back to the host." src="docs/images/evals-overview-light.svg">
</picture>

The diagram shows the designed architecture: the analyst and editor loop and the offline loop are not on `develop` yet. [The full architecture](docs/evals-architecture.md) has every arrow, what is built today, and a table of the ports. [Adopting the engine](packages/evals/docs/adopting-a-host.md) is the full host guide.

## Where to go next

- [Adoption guide](docs/adoption/README.md): what most packages solve and when to adopt them.
- [Integration guide](docs/integration-guide.md): wiring the core into a host as its data layer, locally and across pods.
- [Evals README](packages/evals/README.md): a first eval in one TOML file, then one idea per rung.
- [Observability](docs/observability.md): what 3tears logs and traces, and what the host configures.
- [Exceptions](docs/exceptions.md): how failures reach a caller or an operator.
- [Adding a migration](docs/how-to-add-a-migration.md): the one path for a schema change.
- [CHANGELOG](CHANGELOG.md): what changed in each release.

## Development

3tears is a [uv](https://docs.astral.sh/uv/) workspace, so one `uv sync` installs every package in dev mode:

```bash
git clone https://github.com/pacepace/3tears.git && cd 3tears
uv sync
./scripts/install-hooks.sh    # once per clone: a commit-msg hook that refuses agent attribution
```

Use the scripts; never run `pytest`, `ruff` or `mypy` directly. Extra arguments pass through to the tool.

| Script | What it runs |
|---|---|
| `./scripts/check-all.sh` | Lint, typecheck, the unit tests and the sidecar tests |
| `./scripts/test.sh [package]` | The unit tests, for every package or one: `./scripts/test.sh core` |
| `./scripts/test-integration.sh` | The integration tests, against real containers such as NATS and Postgres (needs Docker) |
| `./scripts/test-sidecar.sh` | The scrape package's nodriver sidecar tests |
| `./scripts/lint.sh` | `ruff check` and the format check; `--fix` fixes |
| `./scripts/typecheck.sh [package]` | `mypy`, strict |

`check-all.sh` does not run the integration tests, and cross-pod behaviour is tested only there. Run `./scripts/test-integration.sh -rs` before a pull request and account for every skip, since a skip reads like a pass. [docs/testing.md](docs/testing.md) has the details, including building the sidecar image first.

## Contributing

1. Cut a feature branch from `develop`, in a fork if you can't push here.
2. Write tests for new behaviour. Tests are contracts: fix the code, never weaken a test.
3. Run `./scripts/check-all.sh`, then `./scripts/test-integration.sh -rs`.
4. Open a pull request into `develop`. It merges with a merge commit, never a squash.

[CLAUDE.md](CLAUDE.md) has the rest of the repo's rules, for people and coding agents alike.

## How it's built

3tears is built with [Enforcement-Accelerated Development](https://doi.org/10.5281/zenodo.17968797) (EAD), [Mark Pace](https://pace.org)'s methodology: the rules a reviewer would otherwise enforce by hand are written as tests. Work is planned in task docs under `docs/`, one per unit of work; the rules run as static and AST checks in `tests/enforcement/` and the `3tears-enforcement` package; and `3tears-observe` puts the call site in every log line.

## License

MIT. See [LICENSE](LICENSE).
