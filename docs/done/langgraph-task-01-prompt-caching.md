# langgraph-task-01: Prompt Caching in the 3tears LangGraph Layer

**Status:** 3tears + SDK hook DONE (2026-04-24). Gateway wire protocol tracked as a follow-up shard (`14-eng-ai-bot/docs/gateway-task-01-prompt-caching-wire.md`).
**Scope:** primarily `3tears/packages/langgraph`; touches `3tears/packages/agent-tools` (wire protocol) and the downstream consumers in `14-eng-ai-bot-agents` / `14-eng-ai-bot` (Gateway service) where noted (`(3tears)` label).

## Closeout

Design pivot after consolidation-task-01 landed: caching lives on a dedicated `PromptCachingHook` (implementing `AgentNodeHook`) rather than inline in `agent_node`. The node body stays small and cache-agnostic; callers that want caching install the hook. The canonical `agent_node` was extended to re-read `chat_model` / `tools` from the post-hook configurable so the hook can swap in a memoized pre-bound model.

Landed commits:

- 3tears `4eb76da` — `threetears.langgraph.caching` (`ChatModelCapabilities`, `detect_capabilities`, `annotate_system_prompt`, `extract_cache_usage`, `compute_tool_key`, `should_bind_tools_fresh`) + 22 unit tests.
- 3tears `1e3ccad` — `PromptCachingHook` in `hooks.py` + `nodes.py` re-reads post-hook configurable + 9 integration tests + 3 AST enforcement guards (`tests/enforcement/test_prompt_caching_ast_guards.py`).
- aibots-agents `9aeba22` — SDK `agent_node` wrapper installs `PromptCachingHook` (gated by `_prompt_caching_enabled` flag + capability detection) + 5 integration tests (summarization/caching compose, second-call cache_read, explicit-disable).

Landed docs: `3tears/docs/prompt-caching.md` (contract, summarization interaction, downstream wiring checklist, worked example, capability map, enforcement guards, known limitations).

Gateway wire protocol (request-side `cache_control_enabled` + response-side `cache_read_input_tokens` / `cache_creation_input_tokens` + hub usage schema columns + cost-accounting discount math) is follow-up work in `14-eng-ai-bot/docs/gateway-task-01-prompt-caching-wire.md`.

Success criteria below still reflect the original inline-in-agent_node plan; the hook-based implementation satisfies every P0 requirement (CACHE-01 through CACHE-04) while keeping the node body single-responsibility.

---

## Objective

Add first-class prompt-caching support to the shared `agent_node` so every agent built on the 3tears LangGraph primitives pays less for repeated prefixes (system prompt + tool definitions + prior conversation). Today the node rebuilds the system prompt, binds tools, and calls the chat model with no `cache_control` hints — every turn is a full-price input. We need to cache at least the system prompt + tool schemas (and optionally the prior-turn transcript) across turns, expose cache hit/miss telemetry on the response, and keep the API drop-in for every chat model family we already support (Anthropic, OpenAI-compatible, and whatever lands next).

---

## Requirements

Operating across `3tears` shared libraries; no `requirements.md` IDs exist here, so requirements are expressed inline.

| ID | Requirement | Priority |
|----|-------------|----------|
| CACHE-01 | `agent_node` must attach provider-appropriate `cache_control` to the last stable prefix block (system prompt; optionally system-prompt-plus-tools) so Anthropic-family models hit the cached prefix on turn ≥2. | P0 |
| CACHE-02 | Bound tools must be attached to the model once per workspace/config-lifetime, not rebuilt per turn, so tool schema bytes are bit-stable across invocations (a prerequisite for cache hits on the tool definitions). | P0 |
| CACHE-03 | Summarization, when enabled, must emit its summary block in a position that does NOT break the cached prefix (the summary replaces history *after* the cache breakpoint, not before it). | P0 |
| CACHE-04 | The response path must surface `cache_read_input_tokens` and `cache_creation_input_tokens` (Anthropic) / equivalent (other providers) on the `AIMessage.usage_metadata` so the Hub's gateway telemetry can bill correctly and tests can assert on cache behavior. | P0 |
| CACHE-05 | Providers that do not support prompt caching (a new Bedrock adapter, a local model, a fake test double) must pass through cleanly — `cache_control` annotations are no-ops when the provider ignores them. | P1 |
| CACHE-06 | A model-family capability map (supports-caching / TTL options / minimum prefix length) must live in one place on the agent-tools side so agents can ask "is caching viable for this model?" without hard-coding provider names in `nodes.py`. | P1 |
| CACHE-07 | Docs in `3tears/docs/` describe the caching contract, the summarization interaction, the downstream wiring checklist (Gateway + SDK), and a worked example that shows cache-read tokens climbing across three turns. | P1 |

---

## Design Context

### Current `agent_node` (the canonical primitive)

From `3tears/packages/langgraph/src/threetears/langgraph/nodes.py:22-59`:

```python
async def agent_node(state, config):
    configurable = config.get("configurable", {})
    chat_model = configurable["chat_model"]
    system_prompt = configurable.get("system_prompt", "")
    tools = configurable.get("tools", [])

    messages = list(state["messages"])
    if system_prompt and (not messages or not isinstance(messages[0], SystemMessage)):
        messages.insert(0, SystemMessage(content=system_prompt))

    model = chat_model.bind_tools(tools) if tools else chat_model
    response = await model.ainvoke(messages)
    return {"messages": [response]}
```

Three problems for caching:

1. The `SystemMessage` is constructed with plain `content=system_prompt` — no `cache_control` hint reaches the Anthropic SDK.
2. `chat_model.bind_tools(tools)` runs every turn; even if tool objects are identical, the bind is a fresh call path and provider libraries typically don't cache it. More importantly, tools can change between turns (the configurable tool list is rebuilt upstream), invalidating the cache prefix.
3. No `usage_metadata` inspection — callers have no way to know whether the cache hit.

### SDK fork

The SDK at `14-eng-ai-bot-agents/src/aibots_agents/runtime/graph.py` has a richer copy of `agent_node` that wraps the 3tears primitive with summarization (`maybe_summarize`), streaming callbacks, and per-call `user_id` injection. The fork imports neither 3tears primitive nor calls through the shared builders — it reimplemented the node to add fields. Any caching change in 3tears must also flow into this SDK fork (or — better — the SDK should delegate to the 3tears primitive once caching lands; that consolidation is a follow-up, not part of this shard).

### Summarization breaks cached prefixes

`14-eng-ai-bot-agents/src/aibots_agents/runtime/nodes/summarization.py` at summary time replaces the first N messages with a fresh `SystemMessage(content=f"Conversation summary: {new_summary}")`. The Anthropic cache keys on the exact byte prefix; injecting a different system message at position 0 invalidates every cached block downstream. The summary must be appended *after* the cached system prompt (e.g. inserted as a second `SystemMessage` or as a dedicated assistant turn) or the shard is worthless.

### Gateway path (cross-repo flag)

Production agents don't talk to Anthropic/OpenAI directly — they route through `14-eng-ai-bot/src/aibots/gateway/` via NATS. The gateway's `completion` handler converts the wire messages back to LangChain and calls `model.ainvoke(lc_messages)`. Today's wire format (`GatewayCompletionRequest` in `aibots/gateway/wire.py`) has no field for cache-control annotations or for the cache-read / cache-write tokens in the response. Part of this shard is adding those fields to the wire protocol and threading them both ways.

### Langchain-anthropic prompt caching semantics

`langchain-anthropic ≥ 0.3.x` accepts `cache_control={"type": "ephemeral"}` on individual content blocks or on `additional_kwargs` of a message; `ChatAnthropic(model_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}})` is not required for current Sonnet/Opus/Haiku 4 families but is required for some transitional models. The *correct* shape is `SystemMessage(content=[{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}])` — i.e. move the system prompt to a structured content block, not a bare string. `ChatAnthropic` then emits the `cache_control` on the provider request.

OpenAI-family models ignore `cache_control` silently; Bedrock uses a different opt-in (`inferenceConfig.cachingConfig`). The abstraction in this shard is: expose a `ChatModelCapabilities` record per model, let `agent_node` consult it, and skip cache annotations on models that don't support it.

---

## Research

- Anthropic prompt caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching — read the "How to use" and "Cache lifetime" sections; the breakpoint shape and the 5-minute TTL (extendable per-turn) are the constraints that drive the node logic.
- LangChain-anthropic caching guide: https://python.langchain.com/docs/integrations/chat/anthropic/#prompt-caching — the canonical way to attach `cache_control` via a structured content list on `SystemMessage` / `HumanMessage`.
- LangGraph config `configurable` docs: https://langchain-ai.github.io/langgraph/how-tos/pass-config/ — we add capability / caching hints to `config["configurable"]` and read them in `agent_node` without changing the MessagesState graph shape.
- OpenAI prompt caching (automatic, no opt-in): https://platform.openai.com/docs/guides/prompt-caching — note the `usage.prompt_tokens_details.cached_tokens` field on the response; `ChatOpenAI` surfaces this on `usage_metadata["input_token_details"]["cache_read"]`.

Not applicable this shard but useful context:
- Amazon Bedrock prompt caching (different wire shape; leave behind a TODO comment pointing at the Bedrock adapter when one lands).

---

## Patterns to Follow

- Reusable-node pattern: `3tears/packages/langgraph/src/threetears/langgraph/nodes.py:22` — keep `agent_node` single-responsibility; new logic goes into helper modules it imports.
- Configurable-injection pattern: same file, the `config["configurable"]["chat_model"]` lookup — every new knob should ride `configurable`, not a new node parameter, so the public node signature stays `(state, config)`.
- Capability-map pattern: model-metadata lookup in `14-eng-ai-bot/src/aibots/hub/gateway/model_metadata.py:73` (cache cost columns already exist in that catalog — treat it as the downstream consumer for cost accounting and mirror the set of model IDs).
- Sphinx docstring + single-return style: `3tears/packages/agent-workspace/src/threetears/agent/workspace/materialize.py` is a good example of the house style the reviewer expects.

---

## Files to Create / Modify

### Create (in `3tears/packages/langgraph/src/threetears/langgraph/`)

- `caching.py` — new module with:
  - `@dataclass ChatModelCapabilities` (`supports_anthropic_cache_control: bool`, `supports_openai_auto_cache: bool`, `min_cacheable_tokens: int`, `cache_ttl_seconds: int`).
  - `detect_capabilities(chat_model: BaseChatModel) -> ChatModelCapabilities` — inspect `type(chat_model).__name__` + `chat_model.model` (or `model_name`) against a model→capability map.
  - `annotate_system_prompt(prompt: str, caps: ChatModelCapabilities) -> SystemMessage` — returns the structured-content form with `cache_control={"type": "ephemeral"}` attached to the last block when `caps.supports_anthropic_cache_control` is True; returns the bare-string form otherwise.
  - `extract_cache_usage(response: AIMessage) -> dict[str, int]` — pulls `cache_read_input_tokens`, `cache_creation_input_tokens`, and (OpenAI) `cached_tokens` from `response.usage_metadata` into a normalized dict shape.
  - `should_bind_tools_fresh(prev_tool_key: str | None, current_tool_key: str) -> bool` — decide whether to rebind (tool set changed) or reuse a cached binding. Tool-key helper: stable hash over every tool's `name + args_schema JSON`.

### Modify (in `3tears/packages/langgraph/src/threetears/langgraph/`)

- `nodes.py::agent_node` — use `detect_capabilities`, `annotate_system_prompt`, and `extract_cache_usage`; stash the bound model on `config["configurable"]["_bound_model"]` keyed by a stable tool-key so the next invocation of the same node reuses the binding when tools haven't changed.
- `builders.py` — if the builders there wire up `configurable` defaults, add a `with_prompt_caching(enabled=True)` option and thread it into `configurable`.

### Modify (cross-repo `(3tears)` coordination)

*These are not part of this shard's acceptance but MUST be called out in the shard's Anti-patterns/Success sections so the subagent doesn't forget them when they pair with the downstream PR.*

- `14-eng-ai-bot-agents/src/aibots_agents/runtime/graph.py::agent_node` — either (a) replace the SDK's fork with a thin wrapper around the 3tears primitive, or (b) mirror the caching logic. (a) is the goal; (b) is acceptable interim.
- `14-eng-ai-bot-agents/src/aibots_agents/runtime/nodes/summarization.py` — change summary-injection so it does NOT replace the first `SystemMessage`; emit the summary as a second `SystemMessage` appended after the prefix, or as a tool/assistant message.
- `14-eng-ai-bot/src/aibots/gateway/wire.py::GatewayCompletionRequest` — add optional `cache_control_enabled: bool` field; `GatewayCompletionResponse.usage` grows `cache_read_input_tokens` + `cache_creation_input_tokens`.
- `14-eng-ai-bot/src/aibots/gateway/handlers/completion.py` — forward the cache-control annotations through `wire_to_langchain()` → `model.ainvoke()` → `langchain_to_wire()` on the response.

### Create tests (in `3tears/packages/langgraph/tests/unit/`)

- `test_caching.py` — pin `detect_capabilities` on a mock `ChatAnthropic` / `ChatOpenAI`; pin `annotate_system_prompt` returning structured-content with cache_control; pin `extract_cache_usage` happy-path + missing-field path.
- `test_agent_node_caching.py` — use a fake chat model returning `AIMessage` with `usage_metadata={"input_tokens": 100, "input_token_details": {"cache_read": 90}}`; assert `agent_node` wired the cache annotations and the response propagates `usage_metadata`. Include a test for the "tool set changed → rebind" branch.

---

## Implementation Notes

1. **Capability detection first.** The single biggest source of bugs in prompt-caching integrations is "attach cache_control unconditionally and hope the provider ignores it" — OpenAI does ignore it but some provider clients raise. Write `detect_capabilities` first; test it against stub chat-model classes before writing the annotate path.

2. **Structured content is the Anthropic contract.** Don't send `SystemMessage(content="...", additional_kwargs={"cache_control": ...})` — `ChatAnthropic` does not read that. Send `SystemMessage(content=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}])`. The token count is the same.

3. **Cache breakpoint goes at the END of the prefix.** Anthropic caches from the start of the request up to (and including) each block carrying `cache_control`. Put the breakpoint on the last stable thing: after the system prompt if that is all we cache, or after the last tool-schema block if we cache both. Putting breakpoints mid-way fragments the cache and often costs more than not caching.

4. **Tool schemas are part of the cached prefix.** `ChatAnthropic.bind_tools()` serializes tools into the request body before the system prompt in some wire shapes; in others they come after. Either way, the tool-JSON bytes must be stable across turns. Two guard rails:
   - Stable ordering — sort the tools list by `tool.name` before binding.
   - Stable schema — our `args_schema.model_json_schema()` output can differ between pydantic versions; capture the JSON at bind time and reuse it on subsequent binds rather than re-materializing.

5. **Summarization strategy.** Summarization runs as a separate node in the SDK; after this shard, that node must not insert a system message at index 0. Suggested shape: first message stays `SystemMessage(content=[... structured, cache_control ...])`; summary goes into a second message (either a second `SystemMessage` or a `HumanMessage` prefixed with `"[summary of earlier turns]"`). Both satisfy Claude's conversation contract.

6. **Usage telemetry is the only ground truth.** Do not assume caching is working because the code "looks right." Every PR must include a test that calls the node twice and asserts `cache_read > 0` on the second turn. The `extract_cache_usage` helper is the mandatory readout.

7. **Do NOT rebind tools inside the node when the set hasn't changed.** Use `should_bind_tools_fresh` + a cached bound-model reference on `configurable`. Caveat: `configurable` is a fresh dict per invocation in LangGraph; the cached reference has to live elsewhere — either on the agent's bootstrap state or, for test simplicity, on a module-level WeakKeyDictionary keyed by `id(chat_model)`. Pick the WeakKeyDictionary for this shard; the bootstrap plumbing is a later polish.

8. **Cross-provider degradation is deliberate.** When `detect_capabilities().supports_anthropic_cache_control is False`, the node still emits a plain `SystemMessage(content=prompt)` — no errors, no warnings. That's the correct behavior.

---

## Anti-patterns

- **DO NOT** attach `cache_control` via `additional_kwargs` — `ChatAnthropic` silently ignores it. Use structured content blocks.
- **DO NOT** place the cache breakpoint on the last human message — the cache is provider-side and stops at the breakpoint; putting it there means we never cache anything on the reply path.
- **DO NOT** insert a fresh `SystemMessage` at position 0 for summarization — it invalidates the cached prefix every time.
- **DO NOT** hard-code provider strings in `nodes.py` (`if "anthropic" in type(model).__name__:`) — use `ChatModelCapabilities`. Hard-coding couples the shared library to specific LangChain adapter names.
- **DO NOT** silently swallow `usage_metadata=None` — surface it as an observability warning (once, per-model, not per-turn).
- **DO NOT** skip the SDK fork sync — if `14-eng-ai-bot-agents/runtime/graph.py` still calls its own copy of the node, the shared caching is dead code in production. The cross-repo follow-up is NOT optional; acceptance of this shard includes filing the SDK-side task with a pointer.
- **DO NOT** ship any back-compat shim, alias, or feature-flag toggle as part of this work. Per `14-eng-ai-bot/CLAUDE.md` "NO BACKWARDS-COMPATIBILITY SHIMS": if a helper signature changes (e.g. `agent_node` gains a kwarg), every caller updates in the same PR. The capabilities interface is added once and used from day one.

---

## Success Criteria

- [ ] `threetears.langgraph.caching` module exists with the four helpers enumerated above; each has Sphinx docstrings following the CLAUDE.md style.
- [ ] `agent_node` attaches `cache_control` to the system prompt on Anthropic-family chat models and propagates `usage_metadata.cache_*` on the response.
- [ ] Tool rebinding is guarded by `should_bind_tools_fresh` — repeated calls with the same tool set do NOT re-call `bind_tools`.
- [ ] New tests pass: `uv run pytest packages/langgraph/tests/unit/test_caching.py packages/langgraph/tests/unit/test_agent_node_caching.py -v`.
- [ ] Fake chat model driving `agent_node` twice in a row reports `cache_read > 0` on the second call.
- [ ] Non-Anthropic chat model (stub) drives `agent_node` cleanly with zero cache annotations and zero warnings.
- [ ] `3tears/docs/prompt-caching.md` documents the contract (what gets cached, what doesn't, summarization interaction, downstream wiring checklist).
- [ ] `ruff check` + `mypy` clean on the touched files.
- [ ] Cross-repo follow-up task(s) filed (one in `14-eng-ai-bot-agents` for SDK-fork consolidation + summarization fix, one in `14-eng-ai-bot` for Gateway wire-protocol + handler changes). Link IDs recorded in this shard before marking complete.

---

## Verification

```bash
# unit tests for the new caching helpers
uv run pytest packages/langgraph/tests/unit/test_caching.py -v

# unit tests exercising the annotated agent_node
uv run pytest packages/langgraph/tests/unit/test_agent_node_caching.py -v

# type check + lint
uv run mypy packages/langgraph/src/threetears/langgraph/caching.py
uv run mypy packages/langgraph/src/threetears/langgraph/nodes.py
uv run ruff check packages/langgraph/

# full package regression
uv run pytest packages/langgraph/ -v
```

Live-agent verification (post-cross-repo follow-up merge, NOT part of this shard's acceptance):

```bash
# Prime the agent with a long system prompt (≥1024 tokens) and send two
# messages in a row. Inspect the Gateway's usage log for
# cache_read_input_tokens > 0 on the second response.
#
# Expected shape in the response usage_metadata:
#   { "input_tokens": N, "input_token_details": { "cache_read": >0 } }
```

---

## Enforcement Test Suggestions

Considerations after this shard lands (pending reviewer approval):

- [ ] AST test: every call to `SystemMessage(content=...)` in `3tears/packages/langgraph/` passes a LIST, not a bare string — guards against a regression that reintroduces a plain-string system message and silently breaks cache annotations.
- [ ] AST test: no call to `chat_model.bind_tools(...)` appears inside `agent_node` without a preceding `should_bind_tools_fresh` check — guards against the "rebind every turn" regression.
- [ ] AST test: `extract_cache_usage` is called on every code path that produces an `AIMessage` inside the node — guards against losing telemetry when the node is refactored.

Not implementing any of these in this shard; flag them in the final review so the reviewer can green-light them as a follow-up.
