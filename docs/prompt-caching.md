# Prompt Caching in the 3tears LangGraph Layer

**Status:** Implemented in v0.5.0 (langgraph-task-01)
**Scope:** `3tears/packages/langgraph` — shared helpers + hook.
**Downstream wiring:** `14-eng-ai-bot-agents` SDK installs the hook on the thin `agent_node` wrapper; `14-eng-ai-bot` Gateway wire protocol carries the cache counters end-to-end (see "Downstream wiring checklist" below).

---

## What gets cached

Anthropic's ephemeral prompt cache keys on the exact byte prefix of a request. The 3tears caching path caches:

1. **The system prompt** — always, on caching-capable Anthropic models (Claude 3/4 Sonnet, Opus, Haiku).
2. **Tool schemas** — indirectly, via the tool-binding memoization path. `chat_model.bind_tools(sorted_tools)` is called once per unique tool set (keyed on sha256 of sorted `(name, args_schema JSON)` pairs) and the resulting bound model is reused across turns. This keeps the tool-JSON bytes byte-stable, which keeps the cached prefix hot.
3. **Prior conversation turns** are NOT currently explicitly cached (no breakpoint on HumanMessage/AIMessage content blocks). They ride on the same cached prefix only up to the point where the first new turn appears.

What is NOT cached:

- The final user message on each turn (by design — the cache breakpoint is at the end of the *stable* prefix).
- Summaries produced by the `SummarizationHook` (they land after the cache breakpoint in the message list).
- Non-Anthropic adapters silently degrade: OpenAI's automatic prompt caching kicks in at the provider level (zero opt-in required) and surfaces on `usage_metadata.input_token_details.cache_read`. Bedrock, local models, and unknown adapters produce plain `SystemMessage(content=<string>)` with no annotations.

---

## The contract

The `threetears.langgraph.PromptCachingHook` implements `AgentNodeHook` and rides the `configurable["_hooks"]["agent"]` slot. Its two phases:

**`before_invoke(messages, config, state)`**

1. Reads `chat_model` from `configurable`.
2. Runs `detect_capabilities(chat_model)`. Conservative: unknown class names return the all-False record, so the degradation path is silent and zero-warning.
3. If caps flag anthropic caching support AND `messages[0]` is a bare-string `SystemMessage`, rewrites index 0 to the structured-content form:

   ```python
   SystemMessage(content=[
       {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
   ])
   ```

   Idempotent: if `messages[0].content` is already a list, the hook leaves it alone.

4. Computes the tool-key digest via `compute_tool_key(sorted_tools)`.
5. Looks up the previous `_BoundModelCache` on the module-level `_BOUND_MODEL_CACHE` (a `WeakKeyDictionary` keyed on the original `chat_model` instance).
6. When `should_bind_tools_fresh(prev_key, current_key)` returns False, reuses the cached bound model; otherwise calls `chat_model.bind_tools(sorted_tools)` fresh and stores the result.
7. Rewrites `configurable["chat_model"]` to the bound handle and sets `configurable["tools"] = []` so the node's own binding path (`chat_model.bind_tools(tools) if tools else chat_model`) becomes a no-op.

**`after_invoke(response, config, state)`**

1. Runs `extract_cache_usage(response)` to pull normalized cache counters from `response.usage_metadata` and/or `response.response_metadata.usage`.
2. Stashes the normalized dict under `response.usage_metadata["cache_usage"]`:

   ```python
   {
       "cache_read_input_tokens": int,     # anthropic cache hits (or openai cache_read)
       "cache_creation_input_tokens": int, # anthropic cache writes
       "cached_tokens": int,               # openai-style alias for cache_read
   }
   ```

   Downstream consumers (tests, Gateway telemetry) read the key unconditionally — the hook always attaches it, with zeros when no telemetry was produced.

---

## Interaction with summarization

`SummarizationHook` and `PromptCachingHook` are designed to compose cleanly. Install order in the SDK wrapper (`aibots_agents/runtime/graph.py`):

1. `SummarizationHook` first — runs BEFORE the caching hook so its summarization-internal `chat_model.ainvoke([summarize_msg])` call sees the original *unbound* chat_model. If caching ran first, the swap to `configurable["chat_model"] = bound_model` would force the summarization helper to talk to the tool-bound model, which is semantically wrong (summarization is a plain LLM chore that takes no tools).

2. `PromptCachingHook` second — runs as the last transformation before the node's actual `ainvoke`. At this point:
   - The message list has already had older history collapsed (when summarization triggered).
   - `messages[0]` is still a `SystemMessage` (summarization preserves the index-0 system prefix explicitly — see `graph_hooks.py::SummarizationHook.before_invoke`).
   - The caching hook annotates `messages[0]` and swaps in the pre-bound model.

Both hooks explicitly keep the system message at index 0 stable. The summarization helper, when triggered, does NOT replace the index-0 `SystemMessage` — it operates on `messages[1:]` and produces a replacement for that tail, leaving the prefix untouched. This is the single most important invariant for prompt caching: the byte prefix of every request must be stable across turns, or the cache never hits.

---

## Worked example

```python
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from threetears.langgraph import PromptCachingHook, agent_node

chat = ChatAnthropic(model="claude-sonnet-4-6")
state = {"messages": [HumanMessage(content="What is the capital of France?")]}
config = {
    "configurable": {
        "chat_model": chat,
        "system_prompt": "You are an encyclopedic assistant. " * 200,  # ~1500 tokens
        "_hooks": {"agent": [PromptCachingHook()]},
    },
}

# Turn 1: cache miss, cache write
result_1 = await agent_node(state, config)
usage_1 = result_1["messages"][0].usage_metadata["cache_usage"]
# usage_1 == {"cache_read_input_tokens": 0, "cache_creation_input_tokens": ~1500, "cached_tokens": 0}

# Turn 2: cache hit on the system prefix
state["messages"].append(result_1["messages"][0])
state["messages"].append(HumanMessage(content="What about Germany?"))
result_2 = await agent_node(state, config)
usage_2 = result_2["messages"][0].usage_metadata["cache_usage"]
# usage_2 == {"cache_read_input_tokens": ~1500, "cache_creation_input_tokens": 0, "cached_tokens": ~1500}
```

The key observation: on turn 2, the tokens billed for input are ~90% cheaper because ~1500 of them come from the cached prefix.

---

## Downstream wiring checklist

Components that need to cooperate for end-to-end caching telemetry:

- [x] **3tears `agent_node`** re-reads `chat_model` and `tools` from post-hook configurable. ([`nodes.py`](../packages/langgraph/src/threetears/langgraph/nodes.py))
- [x] **`threetears.langgraph.caching`** provides the pure helpers. ([`caching.py`](../packages/langgraph/src/threetears/langgraph/caching.py))
- [x] **`threetears.langgraph.PromptCachingHook`** rides `AgentNodeHook`. ([`hooks.py`](../packages/langgraph/src/threetears/langgraph/hooks.py))
- [x] **SDK agent_node wrapper** installs `PromptCachingHook` on recognized models. (`aibots_agents/runtime/graph.py`)
- [ ] **Gateway wire protocol** (`aibots/gateway/wire.py`) grows a `GatewayCompletionRequest.cache_control_enabled` field and `GatewayCompletionResponse.usage.cache_read_input_tokens` / `cache_creation_input_tokens` fields. *(Follow-up shard filed in `14-eng-ai-bot`: see `docs/gateway-prompt-caching-wire.md`.)*
- [ ] **Gateway completion handler** (`aibots/gateway/handlers/completion.py`) forwards the cache annotations from wire into LangChain messages and lifts the cache counters off the response back into the wire usage block. *(Same follow-up shard.)*
- [ ] **Hub usage persistence** (`aibots/hub/usage/`) records cache_read / cache_creation separately from input_tokens so cost accounting reflects the discount. *(Same follow-up shard; touches the schema.)*

---

## Capability map

Today's detection (in `threetears.langgraph.caching._ANTHROPIC_CACHE_MODEL_PREFIXES`):

| Adapter class | Model prefix | `supports_anthropic_cache_control` | `supports_openai_auto_cache` |
|---------------|--------------|------------------------------------|------------------------------|
| `ChatAnthropic` | `claude-3-*`, `claude-sonnet-*`, `claude-opus-*`, `claude-haiku-*`, `claude-4-*` | True | False |
| `ChatAnthropic` | other | False | False |
| `ChatOpenAI` | any | False | True |
| `AzureChatOpenAI` | any | False | True |
| anything else | any | False | False |

Minimum cacheable prefix length for Anthropic: **1024 tokens** (`ANTHROPIC_MIN_CACHEABLE_TOKENS`). Cache TTL: **5 minutes**, extended per read (`ANTHROPIC_EPHEMERAL_TTL_SECONDS`).

Adding a new provider: extend `detect_capabilities` in `caching.py` to recognize the adapter class name and return the appropriate record. Avoid adding provider-specific imports at module level — the capability map is intentionally string-based so workspaces that ship one adapter do not pay the import cost of the others.

---

## Enforcement guards

Three AST tests in `packages/langgraph/tests/enforcement/test_prompt_caching_ast_guards.py`:

1. **`test_system_message_in_caching_path_uses_list_content`** — every `SystemMessage(content=...)` in `hooks.py` passes a list, guarding against a regression that reintroduces a bare-string system message and silently breaks cache annotations.
2. **`test_bind_tools_in_hook_module_is_guarded_by_should_bind_tools_fresh`** — every `.bind_tools(...)` call in `hooks.py` lives in a function that references `should_bind_tools_fresh`, guarding against a "rebind every turn" regression.
3. **`test_after_invoke_of_prompt_caching_hook_calls_extract_cache_usage`** — `PromptCachingHook.after_invoke` always calls `extract_cache_usage`, guarding against dropping cache telemetry.

These guards are scoped to `hooks.py` because the canonical `agent_node` deliberately stays cache-agnostic (it produces a bare-string `SystemMessage` for callers that do NOT install the caching hook; this is the degradation path).

---

## Known limitations / follow-ups

- The hook does not yet cache the conversation tail (prior human + assistant turns). Adding a second breakpoint on the last tool-result message would push the cacheable region further — tracked as follow-up once the Gateway wire protocol lands and we can measure whether the extra prefix bytes are worth the complexity.
- Bedrock prompt caching uses a different wire shape (`inferenceConfig.cachingConfig`); when the Bedrock adapter lands, add a new branch in `detect_capabilities` and teach `annotate_system_prompt` to emit the Bedrock-shaped block instead.
- The `WeakKeyDictionary` cache is process-local. A pod restart drops the memoization (first turn after restart rebuilds the binding); this is acceptable because the Anthropic ephemeral cache is provider-side and lives across our pod restarts.
