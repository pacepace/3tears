# Separating Concerns: Architecture Proposal for Modular Package Use

**Status:** Proposal / RFC
**Author:** Architecture review (autonomous)
**Date:** 2026-06-06
**Scope:** All 19 workspace packages; primary focus on the dependency seams that
force heavyweight installs on lightweight consumers.

---

## 1. Executive summary

3tears is a 19-package uv workspace. The packaging boundaries are drawn around
*features* ("tools", "memory", "models", "channels") but the **dependency
graph is not drawn around *weight***. As a result, a consumer who wants one
small capability is forced to install — and at import time, to *execute* — the
transitive closure of a much larger feature.

The reported symptom is exact and reproducible:

> To use `3tears-models` you have to pull in `3tears-agent-tools`, which pulls
> in half the universe.

This is true, and the root cause is sharper than "models depends on tools."
**`3tears-models` depends on the entire `3tears-agent-tools` distribution to
import two pure-stdlib contract types** — `GeneratedImage` (a dataclass) and
`ImageGenerationBackend` (a `typing.Protocol`) — from
`threetears.agent.tools.protocols`. Those two types have **zero** heavy
dependencies. The package they happen to live in has many.

`pip install 3tears-models` therefore resolves this closure:

```
3tears-models
└── 3tears-agent-tools          ← needed only for 2 protocol classes
    ├── 3tears-agent-memory     → pgvector, langchain-core
    │   ├── 3tears-agent-acl
    │   ├── 3tears-conversations  (also a *stale* edge — see §4.4)
    │   └── 3tears-langgraph    → langgraph, langgraph-checkpoint
    ├── 3tears-agent-audit
    ├── 3tears-conversations    (stale edge — declared, never imported)
    ├── 3tears-langgraph
    └── 3tears (core)           → sqlalchemy, asyncpg, nats-py, aiosqlite
```

A LangChain model-factory library ends up shipping **pgvector, asyncpg,
SQLAlchemy, NATS, and the entire agent-memory subsystem**. None of it is used
by the model factories.

This document:

1. Names the **four architectural patterns** that produce the coupling, each
   with file:line evidence.
2. Separates the two *distinct* kinds of coupling — **distribution coupling**
   (what `pip` installs) and **import-time coupling** (what executes when you
   `import`) — because they have different fixes.
3. Proposes a **phased remediation** that is **transparent to consuming apps**
   wherever possible (re-export shims + namespace packages + lazy loading with
   type stubs), and is explicit about the one place where a deeper, less
   transparent split (core) is worth considering.
4. Recommends **enforcement** so the boundaries cannot silently rot — fitting
   the repo's existing `packages/enforcement` culture.

The good news: **every technique this proposal relies on already exists in the
codebase.** The fix is to apply proven local patterns consistently and to
introduce one new structural element (contract distributions), not to invent
anything.

---

## 2. The two kinds of coupling

These are constantly conflated, and they have different remedies. A correct
proposal has to treat them separately.

| | **Distribution coupling** | **Import-time coupling** |
|---|---|---|
| **What it is** | What `pip install X` drags onto disk | What executes the moment you `import X.y` |
| **Driven by** | `[project] dependencies` in `pyproject.toml` | `__init__.py` re-exports + module-top imports |
| **Symptom** | "pulls in half the universe" (wheels) | slow first import; optional-dep `ImportError`; can't import a leaf without the trunk |
| **Fix family** | contract distributions, optional extras, dependency hygiene | thin `__init__`, PEP 562 / `lazy_loader`, `TYPE_CHECKING`, deferred imports |

The user's pain ("pulls in half the universe") is primarily **distribution
coupling** — pgvector and asyncpg wheels installed for a model factory. But the
two interact: even if every wheel were installed, the **eager `__init__`**
(§4.2) means `from threetears.agent.tools.protocols import GeneratedImage`
*executes* `threetears.agent.tools.__init__`, which imports `agent.memory`,
which imports core, langgraph, etc. So both layers must be addressed to fully
deliver "modular use."

---

## 3. Current dependency reality (verified)

The declared graph and the *actual* (import-level, `src/` only) graph disagree
in many places. The actual graph below was extracted mechanically from imports
and hand-verified (one false positive — a docstring code-example in
`core/testing/sqla_parity.py` referencing `agent.memory` — was removed; **there
is no `core → agent.memory` import cycle**).

### 3.1 Actual internal dependency edges (real code)

```
observe        → (none)                                   ← true universal leaf
nats           → observe
core (3tears)  → nats, observe
langgraph      → core, observe
epoch          → core, nats, observe
enforcement    → core
agent-acl      → core, observe
agent-audit    → nats, observe                            ← does NOT import core
agent-skills   → core, observe
datasources    → core, observe
mcp            → core, epoch, nats, observe
conversations  → core, agent-acl, langgraph, observe
agent-memory   → core, agent-acl, langgraph, observe      ← NOT agent-tools, NOT conversations
agent-tools    → core, agent-audit, agent-memory, langgraph, nats, observe   ← NOT conversations
agent-wake     → core, agent-skills, nats, observe
agent-workspace→ core, agent-acl, agent-audit, agent-tools, nats, observe
channels       → observe (+ core via deferred/in-function, + agent-wake via guarded webhook)
registry       → core, agent-acl, agent-tools, nats, observe
models         → observe, agent-tools (ONLY threetears.agent.tools.protocols)
```

### 3.2 Layer view (depth = transitive distance from leaves)

```
L0  observe
L1  nats
L2  core
L3  langgraph · epoch · enforcement · agent-acl · agent-audit · agent-skills · datasources · mcp
L4  conversations · agent-memory · agent-wake
L5  agent-tools
L6  agent-workspace · registry · models · channels
```

`models` sits at L6 only because of its single L5 edge to `agent-tools`. Cut
that one edge correctly and `models` drops to ~L1 (it needs only `observe` plus
its own LangChain stack).

---

## 4. Root-cause patterns

Four patterns, in priority order. Each is a *general* smell with a *specific*
instance.

### 4.1 Pattern A — Contracts co-located with implementation (no contract seam)

**The smell:** pure interface types (Protocols, dataclasses, enums, capability
metadata) live in the **same distribution** as the heavy implementation that
satisfies them. Any consumer that needs only the *contract* is forced to take
on the *implementation's* entire dependency closure. This is a violation of the
**Dependency Inversion Principle** (depend on abstractions) and the **Interface
Segregation Principle** (don't depend on things you don't use), expressed at the
*packaging* layer rather than the class layer.

**The instance (the headline):**
`packages/agent/tools/src/threetears/agent/tools/protocols.py` is *pure* — its
only imports are `dataclasses`, `typing`, and `uuid`:

```python
# protocols.py — lines 9–12
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from uuid import UUID
```

It defines the shared media contracts: `GeneratedImage`, `MediaInfo`,
`ImageGenerationBackend`, `MediaStorage`, `VisionProvider`, `TextProvider`,
`TranscriptionProvider`. Its own docstring states the intent precisely:

> "These define the contracts that host applications implement. Tool
> implementations can depend on these protocols without coupling to any
> specific infrastructure."

That is a *contracts module*. But it is shipped inside `3tears-agent-tools`,
whose `dependencies` are `3tears`, `3tears-agent-audit`, `3tears-agent-memory`,
`3tears-conversations`, `3tears-langgraph`, `langchain-core`, `httpx`.

Five files in `models` import from it, and **nothing else** from `agent-tools`:

```
packages/models/src/threetears/models/providers/image/openai.py:9
packages/models/src/threetears/models/providers/image/modelslab.py:10
packages/models/src/threetears/models/providers/image/a1111.py:9
packages/models/src/threetears/models/providers/image/huggingface.py:7
packages/models/src/threetears/models/providers/image/comfyui.py:10
    → from threetears.agent.tools.protocols import GeneratedImage, ImageGenerationBackend
```

These contracts are **genuinely shared** between `models` (image *providers*)
and `agent-tools` (the `image_generation` *builtin tool* that consumes a
provider). They belong to *neither* feature package — they are a third thing: a
contract both sides agree on. That is the textbook signal to extract a contracts
distribution (§6.2).

> **On the user's hypothesis** — "dataclasses inside capability modules create
> unnecessary dependencies." This is correct, and now precise: the problem is
> not the dataclass itself (the module is pure). The problem is a *pure contract
> module living in a heavy distribution*, compounded by an eager package
> `__init__` (Pattern B) that means even importing the pure module executes the
> heavy package. The dataclass is the *thing you wanted*; the distribution
> boundary is the *thing that taxed you*.

**Other contract modules in the same position** (pure, but trapped in heavy
distributions — these are latent versions of the same bug):

| Module | Purity | Trapped in distribution that forces |
|---|---|---|
| `agent/tools/.../protocols.py` | PURE (stdlib) | agent-tools closure (memory, pgvector, langchain, asyncpg, …) |
| `agent/tools/.../types.py` (`ChatModelFactory`) | PURE (typing) | same — imported by `registry`, `workspace` |
| `agent/memory/.../types.py` (`MemoryType`, `MemoryConfig`) | PURE | core closure (sqlalchemy/asyncpg/nats) |
| `agent/acl/.../types.py` (`EvaluationContext`, `Role`, …) | PURE | core closure |
| `agent/skills/.../types.py` | PURE (Literals) | core closure |
| `agent/wake/.../types.py` | PURE | core + apscheduler + jinja2 closure |
| `models/.../enums.py`, `models/.../capabilities.py` | PURE / pydantic-only | langchain closure (via eager `__init__`, Pattern B) |
| `langgraph/.../protocols.py` | PURE (`asyncpg` under `TYPE_CHECKING`) | **none — already done right** |

`langgraph/protocols.py` is the in-repo proof that the right pattern is already
understood: it puts `asyncpg` behind `if TYPE_CHECKING:` so the contract is
importable with zero runtime cost. The proposal generalizes that instinct.

### 4.2 Pattern B — Eager `__init__.py` re-export aggregation

**The smell:** a package's top-level `__init__.py` imports *all* its submodules
to present a flat public API (`from threetears.agent.tools import ToolRouter`).
The side effect: **you cannot import any submodule without executing the whole
package.** Python imports parent packages before children, so
`from threetears.agent.tools.protocols import GeneratedImage` first runs
`threetears.agent.tools.__init__`, which eagerly imports `context`, which
imports `threetears.agent.memory.collections`, which runs `agent.memory`'s
(also eager) `__init__`, and so on. The pure leaf detonates the whole trunk.

**The instance:** `agent/tools/__init__.py` eagerly imports 18 submodules,
including the line that bridges into memory:

```python
# agent/tools/__init__.py
from threetears.agent.tools.context import ToolContextManager   # line 22
# context.py, lines 10–11:
#   from threetears.agent.memory.collections import MemoryRefsCollection
#   from threetears.agent.memory.entities  import MemoryRefEntity
```

So `import threetears.agent.tools.protocols` → executes `agent.tools.__init__`
→ imports `context` → imports `agent.memory` (eager `__init__`) → core,
langgraph, observe. The two stdlib protocol classes drag the L2–L5 trunk at
import time, on top of the install-time closure.

**Prevalence** (top-level leaf `__init__.py`, eager vs thin):

| EAGER (detonates on submodule import) | THIN / guarded |
|---|---|
| core, models, registry, agent-tools, agent-memory, agent-acl, agent-skills, agent-wake, agent-workspace, langgraph | observe, nats, conversations, datasources, mcp, epoch, agent-audit, **channels** |

No package uses PEP 562 module `__getattr__` today. `channels` is the in-repo
model citizen: its `__init__.py` wraps optional adapters in `try/except
ImportError` so importing the package never requires slack/discord/fastapi —
exactly the resilience the eager packages lack.

### 4.3 Pattern C — Distribution granularity tracks features, not dependency weight

**The smell:** `3tears-agent-tools` is a *god distribution*. Inside one wheel it
bundles: pure media protocols, the tool registry/router/executor, MCP client,
document parsing (PyMuPDF/docx/openpyxl), media analysis, image generation,
the agent-memory bridge, todo storage, workflow tools, and the chunker. These
have wildly different dependency weights, yet a consumer must take the union of
*all* their dependencies to use *any* of them.

`agent-tools` already uses optional extras for the *leaf* heavy libs
(`calculator`, `units`, `fetch`, `pdf`, `docx`, `xlsx`, `vision`, `ocr`) — good.
But the *structural* deps (memory, conversations, langgraph) are mandatory, and
the *contract* surface (protocols, `ChatModelFactory`) is welded to the
implementation. Extras solve the optional-leaf problem; they do not solve the
contract-seam or the mandatory-structural-dep problem.

The same shape, milder, appears in **core**: it hard-depends on `sqlalchemy`,
`asyncpg`, `nats-py`, `aiosqlite` (`packages/core/pyproject.toml`), so *every*
package downstream of core takes all four — even `agent-acl`, `agent-skills`, or
a contract-only consumer that needs none of them. Yet core *has* a clean
internal seam (verified, §6.4): `cache/base.py`, `config.py`, `serialization.py`,
`namespaces.py`, `entities/base.py`, `exceptions.py` are stdlib/typing-only;
only `collections/*` pulls NATS and `data/*` + `models.py` pull SQLAlchemy/asyncpg.

### 4.4 Pattern D — Dependency-declaration drift (the monorepo hides it)

**The smell:** because uv installs the whole workspace together, a package can
import modules it never declared, or declare deps it never imports, and tests
still pass. The declarations stop describing reality. Each drift is a latent
bug for any *standalone* `pip install` of that package.

**Verified drift** (declared vs. actually imported in `src/`):

| Package | STALE (declared, never imported) | MISSING (imported, not declared) |
|---|---|---|
| `langgraph` | — | **core, observe** (declares *no* 3tears deps; imports `core.data.migrations`, `core.data.store`, `observe`) |
| `registry` | — | **core, nats, observe** (declares only agent-tools, agent-acl) |
| `agent-memory` | **agent-tools, conversations** | observe |
| `agent-tools` | **conversations** | nats, observe |
| `agent-audit` | core (declared, not imported) | nats, observe |
| `agent-acl`, `agent-skills`, `datasources`, `conversations` | — | observe |

Two consequences worth calling out:

- **`observe` and `nats` are imported almost everywhere but declared almost
  nowhere** — they ride in transitively through core. Standalone installs of
  `langgraph`/`registry` would `ImportError`.
- **The `agent-tools ↔ agent-memory` declared cycle is half-stale.**
  `agent-tools → agent-memory` is real (context.py). `agent-memory → agent-tools`
  is **declared but never imported** — a stale edge that creates a *declared*
  dependency cycle with no code behind it. Removing it breaks the cycle for free.

Fixing drift is pure hygiene (no API change) and is the zero-risk first step.
It does **not**, by itself, fix the headline: even with the stale
`conversations` edge removed, `models → agent-tools → agent-memory → pgvector`
remains. The contract seam (Pattern A) is required to actually lighten `models`.

---

## 5. Design principles for the target state

1. **Depend on contracts, not distributions.** A consumer that needs an
   interface should import a *contracts* package whose dependency closure is
   essentially empty (stdlib + pydantic/typing). Implementations depend on the
   same contracts and on their heavy backends.
2. **Distribution boundaries follow dependency weight, not just features.**
   Split where the dependency cost changes, not only where the feature name
   changes.
3. **Optional by default for heavy backends.** A backend lib (pgvector,
   redshift, PyMuPDF, a specific LangChain provider) is an *extra* unless every
   consumer genuinely needs it. The repo already does this well for leaf libs;
   extend it to structural ones.
4. **Lazy at the import layer, honest at the dependency layer.** Importing a
   package's namespace must not execute its whole implementation graph
   (PEP 562 / `lazy_loader`), and every import must be backed by a declared
   dependency.
5. **Transparency is a P1 goal, not a constraint.** Prefer changes that keep
   `from threetears.agent.tools.protocols import GeneratedImage` working
   verbatim (via re-export shims and lazy stubs). Accept a non-transparent
   change only where the payoff is large and the shim cost is high (core split).
6. **Make the invariant enforceable.** A boundary that isn't tested will rot
   (Pattern D is the proof). Encode the layering and contract-purity rules in
   `packages/enforcement`.

---

## 6. Proposal

Four phases, ordered by (value ÷ risk). Phases 0–2 deliver the headline win and
are transparent to consumers. Phase 3 is a deeper, optional split with explicit
trade-offs.

### Phase 0 — Dependency hygiene (zero API change, do immediately)

Make declarations match reality (§4.4). For every package: add the `observe` /
`nats` / `core` deps it actually imports; remove the stale `conversations` edges
from `agent-tools` and `agent-memory`; remove the stale `agent-tools` edge from
`agent-memory` (breaks the declared cycle); drop the unused `core` dep from
`agent-audit` (it imports only `nats` + `observe`).

- **Risk:** negligible. No code moves; only `pyproject.toml` edits.
- **Benefit:** standalone installs work; the *real* graph becomes visible (and
  greppable); the declared cycle disappears; sets the baseline for enforcement.
- **Transparency:** total.

This is necessary but not sufficient — it does not lighten `models`.

### Phase 1 — Extract contract distributions (the headline fix)

Create thin, dependency-free **contracts packages** and have *both* the
implementation and the lightweight consumers depend on them. Keep the old
import paths working with re-export shims so consumers don't change.

**1a. `3tears-media-contracts`** (`threetears.media.contracts` — or, to preserve
paths exactly, see the shim note below).

- Move the *body* of `agent/tools/protocols.py` (the pure media
  Protocols/dataclasses) into this new package. Its only deps: stdlib + (if any
  DTO uses it) pydantic.
- `agent-tools` re-exports them for back-compat:
  ```python
  # agent/tools/protocols.py  (now a shim)
  from threetears.media.contracts import (
      GeneratedImage, ImageGenerationBackend, MediaInfo, MediaStorage,
      VisionProvider, TextProvider, TranscriptionProvider,
  )
  __all__ = [...]  # unchanged
  ```
- `models` depends on `3tears-media-contracts` **instead of** `3tears-agent-tools`
  and changes its five imports to `from threetears.media.contracts import …`
  (or keeps the old path if the shim lives below an eager `__init__` — but see
  Phase 2; the shim path only becomes truly free once `agent-tools.__init__` is
  lazy).

**Result for `models`:** its only remaining inter-package edge is `observe`
(L0) plus `3tears-media-contracts` (deps: none). `pip install 3tears-models` no
longer pulls agent-tools, agent-memory, pgvector, conversations, agent-acl,
agent-audit, langgraph, asyncpg, sqlalchemy, nats. **This is the user's "half
the universe" deleted.**

**1b. `3tears-agent-contracts`** for the cross-package agent interfaces that are
currently trapped: `ChatModelFactory` (`agent/tools/types.py`), and the
lightweight tool-context contracts (`base_tool`'s ABCs, `call_scope`,
`context_envelope`) that `registry` and `agent-workspace` import. These
consumers can then depend on contracts rather than the full `agent-tools`
distribution. (Scope this to what is actually imported across boundaries — see
the registry/workspace edges in §3.1 — not the whole surface.)

- **Risk:** low–moderate. Mechanical moves + shims; behavior identical.
- **Benefit:** breaks distribution coupling at the seam that matters; `models`
  becomes genuinely standalone.
- **Transparency:** high. Old import paths preserved via shims. The only
  visible change is that `models`' *metadata* no longer lists `agent-tools`.

> **Naming/transparency option:** if keeping the literal path
> `threetears.agent.tools.protocols` importable *without* installing
> `3tears-agent-tools` matters, the contracts package can itself own that
> namespace path via the implicit-namespace mechanism. Simpler and clearer is a
> new neutral path (`threetears.media.contracts`) plus the shim above; the shim
> keeps the old path working *when agent-tools is installed*, and `models`
> moves to the new path. Recommend the neutral path.

### Phase 2 — De-eager the `__init__` files (fix import-time coupling)

Convert the EAGER `__init__.py` files (§4.2) to **lazy** ones using the
Scientific-Python `lazy_loader` (`lazy.attach` + `lazy.attach_stub` with a
`.pyi`), the codified community standard (SPEC-0001 / PEP 562). This keeps the
flat public API (`from threetears.agent.tools import ToolRouter` still works)
while ensuring that importing a *submodule* (or touching one attribute) no
longer executes the entire package graph.

```python
# agent/tools/__init__.py  (lazy)
import lazy_loader as lazy
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
```

with a generated `__init__.pyi` listing the same public symbols so mypy/IDEs see
the full, statically-typed API. (mypy is strict here; the stub is what keeps
`--strict` and editor completion intact — this is the explicit answer to the
"lazy loses type info" objection.)

- **Priority targets:** `agent-tools`, `agent-memory`, `core` (these are the
  ones on the hot path of the headline and the most-depended-on).
- **Risk:** moderate. Import *ordering* side effects matter where modules rely
  on import-time registration (e.g. `models` deliberately eager-imports its
  provider modules so `register_capabilities()` runs — see
  `models/__init__.py`). Such intentional eager registrations must be preserved
  (keep those specific imports eager, or convert to an explicit
  `register_all()` called once). Audit each package for import-time side effects
  before flipping it.
- **Benefit:** importing `threetears.agent.tools.protocols` (or the shim) no
  longer detonates memory/core/langgraph; faster cold imports across the board;
  optional-dep `ImportError`s stop leaking from unrelated submodules.
- **Transparency:** high — public API and import paths unchanged; backed by
  `.pyi` stubs so static typing is preserved.

> **In-repo precedents that prove the technique:** `models/providers/*` keep
> `langchain_*` under `TYPE_CHECKING` + factory bodies (so importing `models` is
> metadata-only); `langgraph/protocols.py` guards `asyncpg`; `core/cache/duckdb.py`
> uses `try/except ImportError`; `channels/__init__.py` guards optional adapters;
> `core[testing]` lazy-imports testcontainers inside fixtures. Phase 2 simply
> makes lazy the default for package namespaces instead of an ad-hoc per-module
> habit.

### Phase 3 — (Optional, deeper) Split `core` into contracts + backends

Core is the most-depended-on package and the widest blast radius, so this is
last and optional. But it has a real, verified seam (§6.4): a light
contracts/entities layer vs. a heavy backends layer.

**Option 3a — extract `3tears-core-contracts`** (light): `cache/base.py`
(`L1Backend` Protocol + `MISSING`), `config.py` (`CoreConfig` Protocol),
`serialization.py`, `namespaces.py`, `entities/base.py`, `exceptions.py`. Deps:
stdlib + pydantic + observe. `3tears` (core) then depends on
`3tears-core-contracts` and keeps the heavy runtime (`collections/*` → NATS,
`data/*` + `models.py` → SQLAlchemy/asyncpg, the cache backends). Re-export
shims preserve every `threetears.core.*` path.

- Consumers that only touch entity/contract types (e.g. a future
  contract-only consumer, or `agent-acl`/`agent-skills` if their entity needs
  can be met by the contracts layer) stop pulling SQLAlchemy/asyncpg/NATS.

**Option 3b — backends behind extras**: keep one `3tears` distribution but move
`sqlalchemy`/`asyncpg` to a `[postgres]`/`[sqlalchemy]` extra and `nats-py` to a
`[nats]` extra, with `try/except ImportError` guards and clear errors when a
backend is used without its extra. Less re-architecture, more guard code; risks
runtime surprises if a guard is missed.

- **Risk:** moderate–high (touches the foundation). Must preserve the SQLAlchemy
  table-parity guarantees (`core/testing/sqla_parity.py`) and the migration
  framework. Recommend doing this only after Phases 0–2 prove the contracts +
  lazy patterns in the smaller packages.
- **Benefit:** lets the *whole* lower stack (acl, skills, audit, langgraph,
  datasources) become installable without the Postgres/NATS wheels unless they
  use them.
- **Transparency:** high *for import paths* (shims), but it *does* change the
  dependency metadata and the install story — this is the one place where the
  "transparent" goal is partially traded for the modularity goal, which the
  brief explicitly permits as P1-secondary.

Recommendation: do **3a** if a concrete contract-only consumer emerges or the
lower-stack install weight becomes a real complaint; otherwise defer. Phases
0–2 already resolve the reported problem.

### 6.4 Verified core seam (supports Phase 3)

```
LIGHT (stdlib / typing / pydantic only — candidate 3tears-core-contracts)
  core/cache/base.py        L1Backend Protocol, MISSING        (typing only)
  core/config.py            CoreConfig Protocol                (dataclass, typing)
  core/serialization.py     serializers                        (stdlib)
  core/namespaces.py        namespace helpers                  (stdlib)
  core/entities/base.py     BaseEntity                         (→ cache.MISSING, observe)
  core/exceptions.py        exception types                    (stdlib)

HEAVY (must stay in 3tears runtime)
  core/collections/base.py  BaseCollection      → threetears.nats (nats-py)
  core/collections/*        schema_backed, flush, registry
  core/data/store.py        DataStore           → sqlalchemy (via sql_builder)
  core/data/migrations/*                        → asyncpg
  core/models.py            SA ORM mixins       → sqlalchemy + pg dialect
  core/cache/sqlite.py      SQLiteBackend       (stdlib sqlite3 — light, but lives with backends)
  core/cache/duckdb.py      DuckDBBackend       (optional, already lazy)
  core/backends/*           nats_proxy          → nats-py
  core/coordination/*       leases              → nats-py
```

One small refinement worth making regardless of Phase 3: `entities/base.py`
imports `MISSING` from the `core.cache` *package* (`__init__`), which also
imports the sqlite backend. Importing `MISSING` from `core.cache.base` directly
keeps the entities layer minimal.

---

## 7. Target dependency picture

After Phases 0–2 (the recommended baseline):

```
        observe (L0)
           │
   ┌───────┼─────────────────────────────────────┐
   │       │                                       │
 nats     media-contracts (new, deps: ~none)   agent-contracts (new)
   │       │         ▲                              ▲
 core      │         │                              │
   │       │     models ── depends on ──────────────┘ (only observe + contracts)
   ...     └──── agent-tools (impl) ── re-exports media-contracts
```

`models` (and any future contract-only consumer) depends on contracts + observe,
not on the agent-tools implementation closure. `agent-tools` keeps its rich API
and re-exports the contracts so existing callers don't change.

---

## 8. Worked example: the reported case, before/after

**Before**

```toml
# packages/models/pyproject.toml
dependencies = ["3tears-observe", "3tears-agent-tools", "langchain-core",
                "langchain-anthropic", "langchain-openai", "langchain-openrouter", ...]
```
`pip install 3tears-models` → observe, nats, core, agent-acl, agent-audit,
agent-memory, agent-tools, conversations, langgraph + sqlalchemy, asyncpg,
nats-py, aiosqlite, **pgvector**, httpx, langchain-* . (~half the universe.)

**After (Phases 0–1)**

```toml
# packages/models/pyproject.toml
dependencies = ["3tears-observe", "3tears-media-contracts", "langchain-core",
                "langchain-anthropic", "langchain-openai", "langchain-openrouter", ...]
```
`pip install 3tears-models` → observe, media-contracts + langchain-* only. No
pgvector, no asyncpg, no SQLAlchemy, no NATS, no agent-memory.

**Consumer code:** unchanged if the shim keeps the old path; one-line import
swap to the neutral `threetears.media.contracts` path if preferred. Either way,
the image providers behave identically.

---

## 9. Enforcement (so it can't rot again)

The repo already enforces architecture invariants in `packages/enforcement`
(cache-primitive contracts, underscore-access discipline, fake/protocol parity).
Add three checks in the same spirit:

1. **Declared-vs-actual dependency check.** Fail CI if a package imports a
   `threetears.*` module not in its declared deps, or declares a 3tears dep it
   never imports. (This is exactly the §4.4 audit, automated. `import-linter`'s
   *independence*/*forbidden* contracts or a small AST walker both work.)
2. **Contract-purity check.** Fail CI if a designated contracts package imports
   anything outside an allowlist (stdlib, typing, pydantic, dataclasses). Keeps
   the seam from silently re-acquiring heavy deps.
3. **Layering check.** Encode the L0→L6 layering (§3.2) as an `import-linter`
   *layers* contract so no lower layer imports a higher one and no new cycle
   appears.

A focused **import-cost regression test** (assert that importing
`threetears.models` does not import `pgvector`/`asyncpg`/`sqlalchemy` —
`sys.modules` introspection) cheaply guards the headline win.

---

## 10. Risks, trade-offs, and non-goals

- **More distributions to version.** The repo already versions 19 packages in
  lockstep; +2–3 contract packages fit that model. Contract packages change
  rarely (they're interfaces), so churn is low.
- **Lazy `__init__` + import-time registration.** The real hazard of Phase 2.
  `models` relies on eager provider import for `register_capabilities()`. Each
  package must be audited for such side effects and either keep those imports
  eager or move to an explicit `register_all()`. Do not flip a package to lazy
  blind.
- **Shim sprawl.** Re-export shims are cheap but must be tracked; mark them
  `# back-compat shim — re-exports threetears.media.contracts` and consider a
  deprecation horizon if/when consumers migrate to the neutral paths.
- **Core split (Phase 3) is genuinely invasive.** Recommended only on demand.
- **Non-goal:** changing runtime behavior, the three-tier cache semantics, the
  migration framework, or any public API *behavior*. This proposal moves
  *where types live* and *when modules load*, not *what they do*.

---

## 11. Recommended sequencing

1. **Phase 0** (hygiene) — small PR, immediate, unblocks accurate enforcement.
2. **Enforcement check #1** (declared-vs-actual) — lock in Phase 0.
3. **Phase 1a** (`3tears-media-contracts`) — delivers the reported fix for
   `models`; ship + add the import-cost regression test.
4. **Phase 1b** (`3tears-agent-contracts`) — relieves `registry` / `workspace`.
5. **Phase 2** (lazy `__init__`, starting with agent-tools/agent-memory/core) —
   removes residual import-time coupling; add contract-purity + layering checks.
6. **Phase 3** (core split) — only if/when justified; behind the same shim +
   enforcement playbook proven in 1–2.

Each phase is independently shippable and independently valuable. Phases 0–1
alone close the reported issue.

---

## Appendix A — Method & evidence

- Declared graph: read from every `packages/*/pyproject.toml` `[project]
  dependencies` + `[tool.uv.sources]`.
- Actual graph: mechanical extraction of `from threetears.* import` /
  `import threetears.*` across all `src/` trees, then hand-verified. One false
  positive removed: `core/testing/sqla_parity.py:31` references
  `threetears.agent.memory.collections` **inside a docstring code-block**, not
  in code — so there is **no** `core → agent.memory` import cycle.
- Purity classification: inspection of module-top imports per file (cited inline
  with file:line throughout §4 and §6.4).

## Appendix B — Selected references

- Scientific Python **SPEC-0001**, *Lazy Loading of Submodules and Functions* —
  the codified community standard this proposal's Phase 2 follows:
  https://scientific-python.org/specs/spec-0001/
- `lazy_loader` (PEP 562 `__getattr__` + `.pyi` stub support; used by
  scikit-image, NetworkX, MNE): https://github.com/scientific-python/lazy-loader
- **PEP 562** — module `__getattr__` / `__dir__`: https://peps.python.org/pep-0562/
- **PEP 810** — *Explicit lazy imports* (emerging; future-facing, not required
  here): https://peps.python.org/pep-0810/
- uv — optional-dependencies (extras, published) vs. dependency-groups
  (dev-only, PEP 735): https://docs.astral.sh/uv/concepts/projects/dependencies/
- uv — workspaces: https://docs.astral.sh/uv/concepts/projects/workspaces/
- `import-linter` — enforce layered/independent/forbidden contracts in CI:
  https://import-linter.readthedocs.io/
