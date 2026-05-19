# workspace-task-01: workspace_create idempotency on (agent_id, name)

## Objective

Make `WorkspaceCreateTool` idempotent on `(agent_id, name)` when the requested source matches the existing workspace's source. Same name + same source → return the existing `workspace_id` with `success=True` and re-pin it; same name + different source → return `success=False` with a clean conflict message. Eliminates the L3 ERROR log for "duplicate key value violates unique constraint uq_workspaces_agent_name" on legitimate re-creates while preserving fail-loud behavior on genuinely conflicting calls.

---

## Background

`workspace_create` currently fails on any `(agent_id, name)` collision with `asyncpg.exceptions.UniqueViolationError`, caught at `tools/workspace_create.py:394` and returned to the LLM as `ToolResult(success=False, error="workspace 'X' already exists for this agent")`. The behaviour is fail-clear, but two real-world patterns make it noisier than it should be:

1. **Same source, repeated invoke** — an agent's LLM, mid-conversation, calls `workspace_create("audience_test", from_template="audience")` twice (recursion path, retry, or just re-issuing the call after a partial failure). The second call hits the unique constraint and fails despite asking for *exactly* what the first call already produced. Reproduced in production on 2026-04-30 18:18:51 (hub log: `proxy.py/QueryProxy.handle_tx_execute.2739: tx.execute failed`).
2. **L3 proxy logging at ERROR** — the L3 query proxy logs every tx failure at ERROR level, including this expected domain conflict. Visible noise even when the tool itself catches the violation cleanly.

Platform pattern to align with: `agent_create` is idempotent on slug — re-creating the same agent returns the existing record with `201`, not `409`. The README documents this. `workspace_create` should follow the same shape so the platform's "idempotent create" contract is uniform.

Locked decision (Pace 2026-04-30): **option (a2) — same name + same source = idempotent return; same name + different source = clean error.** Not (a1) "first create wins, ignore source" (silent loss of caller intent) and not (a3) "second call resets workspace" (destroys file state). Strict "what you asked for, with what you asked for, or a clear no."

---

## Design context

**Identity definition:**

- "Source" for conflict comparison is the **effective template_name** stored on the workspaces row at create time. This is set from `from_template` directly, or from the source workspace's `template_name` when the caller passed `from_workspace`. The tool already computes this as `effective_template` at `workspace_create.py:284-286`.
- A call with no `from_template` and no `from_workspace` produces `template_name=None`. A subsequent call with no source against the same name matches (both `None`). A subsequent call with `from_template="X"` against the same name conflicts (`None != "X"`).
- `from_workspace` is **not** persisted as such; it resolves to whatever `template_name` the source workspace itself carries. Two creates that both resolve to the same `effective_template_name` are "same source" by design — the `from_workspace` *name* is not part of the identity comparison.
- `description` is metadata, not source identity. A second create with a different description still matches; the existing description is preserved (no update).

**Conflict semantics on race:** The pre-check `find_by_agent_and_name` may race with a concurrent create. The existing `except asyncpg.exceptions.UniqueViolationError` at line 394 already covers this. New behaviour: inside the catch, re-fetch the now-committed row, run the same source-comparison, branch on idempotent-return vs. clean-conflict — never bubble a raw violation to the LLM.

**Side-effects on idempotent return:**

- DB INSERT: skipped (no new row). Existing workspace + files + versions unchanged.
- `pin.set_pin`: invoked. The agent asked to make this its workspace; honour the intent regardless of who originally created it.
- Per-tool audit publish on `{ns}.audit.workspace.>`: invoked, with `metadata.reused=True` so audit can distinguish reuse from fresh create.
- `WorkspaceCreateEvent` on `{ns}.workspaces.create`: **skipped**. The hub's `WorkspaceNamespaceEmitter` already created the paired `platform.namespaces` row on the original create; re-publishing would either no-op (safe) or trigger a uniqueness collision on `platform.namespaces` (logged-as-error noise we are explicitly trying to eliminate). The event is a "create notification," not a "presence beacon."

---

## Patterns to follow

- Existing tool execute + exception ladder: `packages/agent-workspace/src/threetears/agent/workspace/tools/workspace_create.py:249-407`. The new idempotency branch slots into the same try/except shape.
- `WorkspaceCollection.find_by_agent_and_name`: `packages/agent-workspace/src/threetears/agent/workspace/collections.py:236`. Returns `Workspace | None`. Use unchanged.
- `Workspace.template_name` field: `collections.py:66` (`Column("template_name", STRING_TYPE, nullable=True)`). Read this directly off the entity returned by `find_by_agent_and_name`.
- Error envelope shape (`_CreateError`, `WorkspaceValidationError`, `UniqueViolationError`, generic `Exception`): preserve the catch-ladder ordering. Add the new logic at the top of `execute()` (pre-check) and inside the existing UniqueViolationError handler (race fallback). No new exception type needed.
- Sphinx docstrings + lowercase + `:ptype` not `:type` per CLAUDE.md (3tears repo follows the same conventions as the hub).
- Tests under `packages/agent-workspace/tests/unit/test_workspace_create.py` (find via `grep -rn "WorkspaceCreateTool" tests/`); new tests extend that file.

---

## Files to modify

- `packages/agent-workspace/src/threetears/agent/workspace/tools/workspace_create.py` — three surgical edits:
  1. In `execute()` immediately after computing `effective_template` (line 286), pre-check for an existing workspace via `find_by_agent_and_name(agent_id, name)`. If found:
     - source matches → set `result = ToolResult(success=True, content=<reused message>, metadata={"workspace_id": str(existing.id), "reused": True})`, run `pin.set_pin`, publish audit with `reused=True`, **do not** publish `WorkspaceCreateEvent`, return.
     - source mismatches → return `ToolResult(success=False, content="", error=f"workspace {name!r} already exists with a different source (existing template={existing.template_name!r}, requested template={effective_template!r}); pick a new name or call workspace.use to switch to the existing one")`.
  2. In the `except asyncpg.exceptions.UniqueViolationError` handler at line 394, **replace** the bare `success=False` return with the same idempotent-vs-conflict branching (re-fetch via `find_by_agent_and_name`, compare `template_name`). Race-safe: a concurrent create that won the INSERT now appears in the fetch.
  3. The `WorkspaceCreateEvent` publish (around line 380, search for `Subjects.workspaces_create()` or `WorkspaceCreateEvent`) gates on `result.metadata.get("reused") is not True`.
- `packages/agent-workspace/tests/unit/test_workspace_create.py` (or wherever the existing tool tests live — confirm via `grep -rn "class WorkspaceCreateTool\|workspace_create_tool" tests/`) — add seven tests (see Implementation Notes below).
- `packages/agent-workspace/README.md` — under whatever section documents `workspace_create` (search for "workspace_create" in the README), one paragraph: "create is idempotent on `(agent_id, name)` when the requested source matches the existing workspace's `template_name`. Mismatched source returns a clean conflict; matching source re-pins the existing workspace and emits an audit event with `reused=True`."

---

## Implementation notes

1. **Test-first.** Per CLAUDE.md, write each test before its implementation. Test list:

   - `test_create_idempotent_same_template` — create with `from_template="X"`, then create again with same name + `from_template="X"`; assert second call returns `success=True`, `metadata.reused=True`, same `workspace_id` as first call.
   - `test_create_idempotent_both_empty` — create with no template, no source; create again same name with no template, no source; assert idempotent return.
   - `test_create_idempotent_from_workspace_resolves_to_same_template` — create A with `from_template="X"`; create B forking A (`from_workspace="A"`); create B again with `from_workspace="A"`; assert B's second call is idempotent (both resolve to `template_name="X"`).
   - `test_create_conflict_different_template` — create with `from_template="X"`; create again same name with `from_template="Y"`; assert `success=False`, error names both templates.
   - `test_create_conflict_template_vs_empty` — create with `from_template="X"`; create again same name with no template; assert `success=False`.
   - `test_create_idempotent_repins_existing` — first create pins workspace A. Pin a different workspace B. Second create against A's name returns idempotent; assert pin is now A again.
   - `test_create_idempotent_publishes_audit_skips_namespace_event` — second idempotent create publishes one audit event with `reused=True` and zero `WorkspaceCreateEvent`s.
   - `test_create_race_unique_violation_resolved_to_idempotent` — patch `find_by_agent_and_name` to return None on first call (pre-check) but the row appears between pre-check and INSERT; assert the `UniqueViolationError` is caught, re-fetch finds the row, source compares equal, returns idempotent.
   - `test_create_race_unique_violation_resolved_to_conflict` — same race shape but the concurrent winner used a different source; assert clean conflict error.

2. **The pre-check is an optimisation, not a guarantee.** YugabyteDB allows two concurrent transactions to both pass `find_by_agent_and_name` (returns None for both) and both attempt INSERT; one wins, the other hits `UniqueViolationError`. The race-handling branch in the `except UniqueViolationError` is the actual correctness mechanism. The pre-check exists only to skip the INSERT round-trip in the common single-caller case and to avoid an L3 ERROR log entry every time.

3. **Description handling.** A second idempotent call with a different `description` than the existing row's: the existing description wins. We do NOT update on idempotent return — that would mutate state via a tool whose contract is "create or get," confusing replay semantics. If the agent wants to update description, that's a different tool's concern (workspace_update if it exists, or a future shard).

4. **`pin.set_pin` on idempotent path.** Run unconditionally on idempotent return. The agent's invocation says "I want this workspace pinned for this conversation"; honour it whether we created the workspace or merely confirmed it. Use the same call shape as the create-success path (`workspace_id=existing.id, workspace_name=name, pinned_by_actor_id=self._agent_id`).

5. **Audit publish on idempotent path.** The existing audit publish around line 324-380 builds an audit event with the per-tool scope. Add a `reused` boolean to the audit envelope's metadata. The audit collector on the hub side does not need to change to consume this — the field is additive and the audit envelope already accepts arbitrary metadata. Leave the same `try/except` swallow shape; audit failures are non-fatal for create.

6. **`WorkspaceCreateEvent` skip on idempotent path.** The publish runs only when the `result.metadata.get("reused")` is falsy. Place the skip-check at the publish call site, not inside the publisher — keeps the publisher dumb and makes the tool's intent clear at the call point.

7. **Result message strings.** Lock these so tests can match:

   - Idempotent success content: `f"workspace {name!r} already exists with the same source ({effective_template!r}); reusing existing workspace"`
   - Conflict error: `f"workspace {name!r} already exists with a different source (existing template={existing.template_name!r}, requested template={effective_template!r}); pick a new name or call workspace.use to switch to the existing one"`

   These are the user-facing messages the LLM reads. Keep them precise and actionable so the agent can decide between "rename" and "switch."

8. **Source-identity comparison.** Define a small private helper in the same module (no separate file — keep the change surgical):

   ```python
   def _same_source(existing: Workspace, requested_template: str | None) -> bool:
       """source identity for idempotency: effective template_name match.
       both ``None`` matches; ``str`` equality matches; ``None`` vs ``str`` is conflict.
       """
       return existing.template_name == requested_template
   ```

   Used by both the pre-check branch and the race-recovery branch in the UniqueViolationError handler. Single comparison rule, two call sites.

---

## Anti-patterns

- DO NOT change the workspaces schema to add a `source_kind` discriminator column. The `template_name` column already encodes effective source identity for both template-derived and fork-derived workspaces (forks inherit their parent's `template_name`). Schema change would be migration churn for no extra discrimination.
- DO NOT compare `description` or any non-source metadata in the conflict check. Description is mutable; comparing it would make the tool's idempotency surface depend on caller-passed metadata that has nothing to do with workspace identity.
- DO NOT remove the `except asyncpg.exceptions.UniqueViolationError` handler thinking the pre-check makes it unnecessary. The pre-check is a fast-path; the race is real (two concurrent agents, same name, same template — the second one's pre-check sees None, the INSERT fails on the violation). Both branches converge through the same `_same_source` helper.
- DO NOT re-publish `WorkspaceCreateEvent` on the idempotent path. The hub-side namespace_emitter created the matching `platform.namespaces` row on the original publish; re-publishing would either be a no-op upsert (safe but semantically off) or would race against the namespaces table's uniqueness (the exact log noise this shard is trying to eliminate). Strictly: emit "create notification" only when a row was actually inserted.
- DO NOT update the existing workspace's `description`, `template_name`, or `date_updated` on idempotent return. The tool's contract is "create or get" — get must not mutate.
- DO NOT widen idempotency to include `(agent_id, name)` *across* sources by silently picking the existing source. That's option (a1), explicitly rejected — silent semantic divergence between caller intent and stored state is a bug source. Conflict must be loud and named.
- DO NOT change the L3 QueryProxy's ERROR-level logging of `UniqueViolationError` in this shard. That's option (B) from the design discussion and was explicitly scoped out — Pace's call was option (A) only. The pre-check eliminates 99% of the ERROR-log occurrences indirectly; the rare remaining race-path violation logs once and is handled cleanly. A separate shard can address proxy log levels if it remains a concern.
- DO NOT introduce a new exception type for the conflict case. The `ToolResult(success=False, error=...)` envelope is the LLM-facing contract; raising would force a different code path through the existing `except Exception` ladder and lose the precision of the error message. Use the result envelope, not exceptions.

---

## Success criteria

- [ ] Same name + same template_name (including both `None`) → `ToolResult(success=True, metadata.reused=True)`, same `workspace_id` returned, no DB INSERT, existing pin replaced with this workspace.
- [ ] Same name + different template_name → `ToolResult(success=False)`, error names both existing and requested templates, no DB mutation, pin unchanged.
- [ ] `from_workspace` calls compare via the source workspace's `template_name`; two creates that resolve to the same effective template_name are idempotent.
- [ ] Audit event published on idempotent return carries `metadata.reused=True`.
- [ ] `WorkspaceCreateEvent` on `{ns}.workspaces.create` is NOT published on the idempotent return path.
- [ ] `pin.set_pin` IS invoked on idempotent return path with the existing `workspace_id`.
- [ ] Race path: pre-check returns None, INSERT raises `UniqueViolationError`, re-fetch + source-compare resolves to either idempotent return or clean conflict — never propagates the raw violation.
- [ ] All seven new tests pass: `uv run pytest packages/agent-workspace/tests/unit/test_workspace_create.py -v -k idempotent`
- [ ] Existing workspace_create tests still pass: `uv run pytest packages/agent-workspace/tests/unit/test_workspace_create.py -v`
- [ ] Type checking passes: `uv run mypy packages/agent-workspace/src/threetears/agent/workspace/tools/workspace_create.py`
- [ ] Lint passes: `uv run ruff check packages/agent-workspace/`
- [ ] No back-compat shims, alias re-exports, or dual code paths introduced.
- [ ] README documents the idempotency contract in one paragraph.

---

## Verification

```bash
cd /path/to/3tears

# Unit tests
uv run pytest packages/agent-workspace/tests/unit/test_workspace_create.py -v -k idempotent
uv run pytest packages/agent-workspace/tests/unit/test_workspace_create.py -v

# Type + lint
uv run mypy packages/agent-workspace/src/threetears/agent/workspace/tools/workspace_create.py
uv run ruff check packages/agent-workspace/

# Manual smoke (requires a running devx stack and an agent with workspace block)
# 1. Bring up the stack against an agent.yaml with workspace.bind_root configured.
# 2. From the chat UI, ask the agent to create a workspace twice with the same name + template.
# 3. Tail the hub log: `docker logs -f aibots-devx-hub-1 | grep -E "tx.execute failed|workspace"`
#    Expected: zero "duplicate key value violates unique constraint uq_workspaces_agent_name" lines.
# 4. Tail the agent log; the agent's tool result for the second call shows
#    `metadata.reused=true` and the `content` references "reusing existing workspace".
```

---

## Out of scope

- L3 QueryProxy ERROR-level logging of `UniqueViolationError` (option (B) from the design discussion). The pre-check elides the common case; the rare race path logs once and is handled. If the residual logging remains an issue post-(A), a follow-up shard addresses proxy log policy.
- Schema changes to record source provenance more precisely (e.g. distinguishing template vs fork at the row level). Current `template_name` column suffices for source-identity comparison.
- Updating the existing workspace's metadata (description, etc.) on idempotent re-create. "Create or get" must not mutate; mutation is a separate tool's concern.
- Changing `agent_create`, `customer_create`, or other "create" surfaces to share a unified idempotency primitive. Worth doing if a third create path needs the same shape, but premature with two examples.
- Cross-agent workspace sharing. The unique constraint is `(agent_id, name)` — workspaces are agent-scoped by design. This shard preserves that scope.

---

## Enforcement test suggestions

After completing this task, consider whether enforcement tests are needed for:

- [ ] **`WorkspaceCreateEvent` is gated on `reused=False`** — suggested test: AST-walk `workspace_create.py` to assert the `WorkspaceCreateEvent` publish call site is wrapped in a conditional that reads `reused` from the result metadata. Catches future refactors that drop the gate and re-introduce the namespace-uniqueness race.
- [ ] **`UniqueViolationError` handled by re-fetch, not propagation** — suggested test: AST-walk the `except asyncpg.exceptions.UniqueViolationError` block to assert it calls `find_by_agent_and_name` (the re-fetch pattern). Catches future regressions that strip the race-recovery branch.
- [ ] **Pre-check + race-recovery share the same comparison helper** — suggested test: AST-walk to assert both code paths call `_same_source` (or whatever the helper is named). Catches the bug class where the two paths drift apart and accept different-source idempotency in one branch but reject in the other.

**Note:** Do not implement enforcement tests without approval. Document suggestions here for review.
