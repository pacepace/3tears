# Build plan: the tool-call wire protocol moves into core, and registry stops depending on agent-tools

Branch: `tools/wire-in-core` (off origin/develop c65ecdf8c). One PR into develop, its own area (tools/registry).
Critic mode: cumulative at the end.

## Goals

`3tears-registry` installs agent-tools' whole closure (12 family packages, 51 third-party, among them
langchain, langgraph, pgvector, httpx) only to read the tool server's wire messages. Those messages are
cleanly separable from `agent/tools/server.py`'s implementation. Moving them into core cuts registry's
closure to 6 family packages and 17 third-party, and stops registry moving whenever agent-tools does.

Success: registry's `pyproject.toml` no longer names `3tears-agent-tools`; nothing in `packages/registry/src`
imports `threetears.agent.tools`; every existing import path of a moved name still works and resolves to the
same object; the whole unit suite is green apart from known reds.

Out of scope: removing registry's mirror models (`ProbeResponse`, `DiscoverRequest`/`DiscoverResponse`,
`ProxyCallResponse`); moving `TolerantWireModel` or `CacheClass` out of core; a core-free contracts leaf.

## Owner rulings (2026-10-25, binding)

- The home is core: `threetears.core.tool_wire`, following `core.http_cache` (promoted when a second consumer
  arrived and neither package may import the other). Not a new leaf: a leaf that depends on core is a leaf in
  name only, and costs a lockstep package and a PyPI publisher.
- `CallResponse` moves with `CallRequest`.
- No version bump (work PRs never bump; the family sits at 0.78.0 and v0.77.0 is the latest tag).
- Do not merge develop into other area PRs.

## Decisions taken by the main session (inferred; flag at the boundary)

- [DECISION: `bind_log_context` moves with `CallContext` | it is the canonical projection of a call's context
  to log tags; leaving it would make registry duplicate the mapping]
- [DECISION: registry's tests that drive real pods (`ToolServer`, `TearsTool`, builtins) keep importing
  agent-tools, which the workspace install provides | `imports_declared` and dependency alignment scan `src`
  only; moving them is churn with no consumer]
- [DECISION: the pinned `context_envelope` error string (agent-tools test_context_envelope.py:147,
  registry proxy.py:241) keeps its text | the old path still works]
- [DECISION: agent-tools' `test_context_envelope.py` moves whole to core as `test_tool_wire_context.py`;
  the tool-free classes and methods of `test_rest_affordance.py` move to core's `test_tool_wire_rest.py`
  and the rest stay | every case in the first file is pure, so a move, not a copy, keeps one home per
  assertion]
- [DECISION: at chunk 01's commit core and agent-tools both define the names, so the tolerant-reader
  inventory carries both `CallRequest` keys and `test_public_api_has_a_caller` is red for the five core
  names only agent-tools' own definitions referenced; chunk 02's re-export imports are their caller, and
  chunk 02 drops the old inventory key | a chunk boundary inside one move; the branch is judged at its tip]

## What moves (line numbers on develop c65ecdf8c)

Into `packages/core/src/threetears/core/tool_wire/`:
- `context.py`: all of `agent/tools/context_envelope.py` (`CallContext`, `bind_log_context`, ...; needs pydantic,
  `observe.set_context`/`clear_context`).
- `rest.py`: all of `agent/tools/http_operation.py` (8 names incl. `RestAffordance`, `RestAffordanceError`,
  `PathTemplateBinding`, `QUERY_METHODS`; needs `core.http_cache`). It moves because
  `ToolManifestEntry.face_rest: RestAffordance | None` (server.py:411).
- `messages.py` (or split by concern): from `agent/tools/server.py`'s wire block (336-946):
  `ToolManifestEntry`, `RegistrationManifest`, `RefusedTool`, `RegistrationResponse`, `CallRequest` (with its
  private `_LEGACY_FLAT_IDENTITY_FIELDS`, used only by its own validator), `CallAccepted`, `CallResponse`,
  `HeartbeatMessage`.
- `__init__.py` re-exporting the public names.

Stays in agent-tools (pod policy and machinery): `ToolRegistrationRefused`, `FINAL_REFUSAL_CODES`,
`refusal_is_final`, `refusals_in_reply`, `ProbeAck`, the `DiscoveryProbe*` models, `ToolCallFailure`,
`TOOL_RESULT_TOO_LARGE`, `tool_namespace_*`, `nats_connect`, `ToolServer`.

Re-export rule: `server.py`, `context_envelope.py` and `http_operation.py` import the moved names from core at
module top and keep them in `__all__` (strict mypy explicit re-export). The same class objects, so isinstance,
nested validation and `is` hold. `__module__` changes: say so in the CHANGELOG.

## Chunks

### 01: core.tool_wire (new code in core)
- Create the package, moving code verbatim; rewrite docstrings that say "defined here because registry depends
  on agent-tools" (server.py:485-491; registry registration.py:57-64) to the new reality.
- Core must not import agent-tools or registry (check `tests/enforcement` layering and core's own boundary
  tests). No function-level imports.
- Tests in `packages/core/tests/`: move the pure cases of agent-tools' `test_rest_affordance.py` and the
  `CallRequest`/`CallContext` cases of `test_context_envelope.py`, or add equivalents; a test that the moved
  names are core's public surface.
- `tests/enforcement/test_wire_models_are_tolerant_readers.py:27`: inventory keys become the core FQNs.
- Done when: lint, typecheck core, and the touched test files pass.

### 02: agent-tools re-exports from core
- Replace the definitions in `server.py`, `context_envelope.py`, `http_operation.py` with core imports.
- Extend `packages/agent/tools/tests/enforcement/test_rest_declaration_shares_binding.py` (its `_SRC_ROOT`
  at :33) to read core's `tool_wire/rest.py` too, so the check still sees the binding.
- Add a test that each old path's name `is` core's object.
- Done when: lint, typecheck agent-tools, and the touched test files pass.

### 03: registry switches over
- Rewrite registry's 5 `src` import lines (proxy.py:28-29, client.py:59, health.py:32, registration.py:27-33)
  to `threetears.core.tool_wire`.
- `packages/registry/pyproject.toml`: drop `3tears-agent-tools` (dependency and uv source); its floor in
  `tests/enforcement/test_dependency_alignment.py` (~:375) moves with it.
- Registry tests: point wire-name imports at core; tests needing pod machinery keep agent-tools.
- Add `packages/registry/tests/test_import_cost.py`: (i) an AST scan proving no `threetears.agent.tools` import
  anywhere in registry `src`, TYPE_CHECKING included; (ii) a fresh-interpreter probe
  (`threetears.enforcement.import_floor.modules_loaded_by_importing`, if present on this branch, else a local
  subprocess probe) that importing `threetears.registry` loads no `threetears.agent.tools*`, `langchain*`,
  `langgraph*` or `pgvector*`.
- Done when: lint, typecheck registry, and the touched test files pass.

### 04: docs
- CHANGELOG Unreleased entry: names moved into `threetears.core.tool_wire`, old paths still work, `__module__`
  changed, registry no longer depends on agent-tools (an app that reached agent-tools through registry declares
  it itself).
- `docs/separate-concerns-decisions.md` Decision 2 (:86-109): record that the wire envelopes left server.py,
  into core.
- README package table/diagram if it shows registry → agent-tools.

## Gates (main session)
Full unit suite on the testament pool; integration `./scripts/test-integration.sh -rs` for registry and
agent-tools after telling discodon; cumulative Critic.

## Risks
- `test_public_api_has_a_caller`: moved names gain new definitions in core; each needs a src caller outside
  its defining module (registry and agent-tools provide them) or a ledger entry.
- `test_api_growth_requires_a_minor_bump`: new core surface; 0.78.0 vs v0.77.0 is already a minor.
- Merge conflict with #947 (iam) is unlikely: #947 does not touch core's tool code, registry or agent-tools.
