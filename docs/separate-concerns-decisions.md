# Separating Concerns: Decisions and Status

**Status:** Decided / partially implemented
**Relates to:** `docs/separate-concerns.md` (the RFC, branch
`feat/separate-concerns-proposal`)
**Scope:** records what the maintainers decided on each open question in
the RFC, what has shipped, and what is deferred with evidence.

---

## Shipped (this branch)

| RFC item | Status |
|---|---|
| Phase 0 — dependency hygiene | **Done.** Every workspace package's declarations now match its actual imports (verified mechanically). The declared `agent-tools ↔ agent-memory` cycle is gone. `channels` is intentionally untouched: its agent-wake edge is declared via the `webhook` extra and its core imports are deferred — both sanctioned shapes (see enforcement semantics below). |
| Enforcement check #1 — declared-vs-actual | **Done.** `threetears.enforcement.dependency_alignment`: `dependency.missing` / `dependency.stale`, strict by default, exemptions require rationales. |
| Phase 1a — `3tears-media-contracts` | **Done.** Pure media protocols extracted to a dependency-free package (`threetears.media.contracts`); `threetears.agent.tools.protocols` is a re-export shim; `3tears-models` depends on contracts + observe only. |
| Enforcement check #2 — contract purity | **Done.** `contract.impure`: designated contracts packages may import only stdlib, their own namespace, and configured extras. `TYPE_CHECKING` imports are exempt. |
| Import-cost regression test | **Done.** `packages/models/tests/test_import_cost.py` asserts in a fresh interpreter that `import threetears.models` loads no `threetears.agent*`, `threetears.core`, `threetears.nats`, sqlalchemy, asyncpg, pgvector, or nats modules. |

### Enforcement import-context semantics (decided)

An import requires a declared hard dependency only when it is an
**unguarded module-top** import. Three shapes are sanctioned for optional
or deferred dependencies and are never flagged missing:

1. `try: ... except ImportError:` guards,
2. function-body (deferred) imports,
3. `if TYPE_CHECKING:` blocks.

Optional-extra declarations satisfy the requirement (the `channels`
`webhook` extra is the model citizen). All shapes count as *usage* when
deciding staleness.

---

## Decision 1 — lazy `__init__`: hand-rolled PEP 562, not `lazy_loader`

The RFC's Phase 2 proposes the `lazy_loader` package (`attach_stub` +
generated `.pyi`). **Decided: hand-rolled PEP 562 with a `TYPE_CHECKING`
block instead.** Pattern:

```python
# package __init__.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from threetears.example.api import PublicThing  # real imports: mypy/IDE see full types

_LAZY = {"PublicThing": "threetears.example.api"}

def __getattr__(name: str) -> object:
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value          # cache: __getattr__ fires once per name
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
```

Rationale:

- **Zero added runtime dependency.** 3tears is a 19-package library
  suite; every dependency it adds is inherited by every consumer.
  `lazy_loader` is well-maintained, but a ~10-line stdlib pattern does
  not justify a supply-chain edge on the package that exists to *remove*
  dependency weight.
- **No stub drift.** `attach_stub` requires a generated `.pyi` that
  duplicates the public API; under strict mypy the stub and the module
  can drift silently. The `TYPE_CHECKING` block lives in the same file
  as the lazy map, so drift is visible in one diff and a trivial test
  (`__all__`/`_LAZY`/`TYPE_CHECKING` agreement) can pin it.
- **House precedent.** The repo already uses `TYPE_CHECKING` guards and
  deferred imports as idiom (`langgraph/protocols.py`, models providers,
  `channels/__init__.py`); this generalizes the existing instinct rather
  than importing a new one.

Trade-off accepted: per-package boilerplate (~15 lines) instead of two
lines of `lazy_loader`. With ~3 priority packages (Phase 2 below), the
total is small.

## Decision 2 — Phase 1b (`3tears-agent-contracts`): deferred, with evidence

The RFC scoped 1b loosely ("`ChatModelFactory`, the lightweight
tool-context contracts that `registry` and `agent-workspace` import").
A purity audit of what those consumers *actually* import:

| Consumer | Imports | Defining module | Weight |
|---|---|---|---|
| registry | `RegistrationManifest`, `HeartbeatMessage`, `CallRequest` | `agent/tools/server.py` | **HEAVY** — server.py imports agent-audit, nats, core at module top |
| registry | `CallContext`, `bind_log_context` | `context_envelope.py` | LIGHT (observe only) |
| workspace | `TearsTool`, `ToolResult`, `MCPToolDefinition` (14 files) | `base_tool.py` | LIGHT |
| workspace | `ToolCallScope`, `current_scope` (8 files) | `call_scope.py` | LIGHT |
| workspace | `ToolContextManager` (11 files) | `context.py` | **HEAVY** — context.py imports agent-memory at module top |

**Verdict: a contracts package cannot deliver what 1b promises.**
Both consumers' heaviest edges (`server.py` wire envelopes for registry,
`ToolContextManager` for workspace) are genuinely implementation-coupled —
extracting the LIGHT modules would shrink neither consumer's install
closure, because the HEAVY edges remain. The real fixes are design
changes (move the registration/heartbeat wire envelopes out of
`server.py`; break `ToolContextManager`'s memory coupling), which belong
in the RFC discussion, not in a mechanical extraction. Revisit after
Phase 2: lazy `__init__`s remove the *import-time* half of the pain for
free, which may be all these consumers need.

## Decision 3 — Phase 2 (de-eager `__init__`s): own PR, with this checklist

Phase 2 proceeds (priority order: `agent-tools`, `agent-memory`, `core`)
as a separate PR after the current branches land. Flipping a package to
lazy requires this audit, recorded in the PR description per package:

1. **Import-time side effects.** Grep the package for module-top calls
   that register state: `register_capabilities()`, codec registration,
   `sqlite3.register_adapter`, `logging` handler installation, subclass
   registries / `__init_subclass__`, module-level singletons. Each one
   either stays eagerly imported or moves to an explicit `register_all()`
   invoked by the consumer.
2. **`__all__` / `_LAZY` / `TYPE_CHECKING` agreement test.** Add the
   three-way consistency test alongside the flip.
3. **Import-cost regression test.** Add a `sys.modules` probe (the
   models test is the template) asserting the package's submodule
   imports no longer detonate the trunk: e.g.
   `import threetears.agent.tools.protocols` must not load
   `threetears.agent.memory`.
4. **Downstream smoke.** Run the full workspace suite plus one consumer
   app (metallm) against the flipped package before merge.
5. **No blind flips.** One package per commit; revert unit is one
   package.

Known Phase 2 hazard already catalogued: `models/__init__` eager-imports
its provider modules so `register_capabilities()` runs at import time —
that registration becomes the test case for checklist item 1.

## Phase 3 (core split) — unchanged from RFC

Deferred until a concrete contract-only consumer emerges. The §6.4
seam analysis stands. The one no-regret refinement (import `MISSING`
from `core.cache.base` instead of the `core.cache` package `__init__`)
can ride any future core PR.
