# 3tears-langgraph

Three-tier LangGraph checkpoint saver: L1 (SQLite) -> L2 (NATS KV) -> L3 (PostgreSQL).

L1 and L2 are optional cache layers that degrade gracefully on failure. L3 (PostgreSQL) is the source of truth, reached through the `AsyncQueryExecutor` protocol so the same saver serves trusted services (direct asyncpg pool) and sandboxed agents (NATS L3 proxy).

## Installation

```bash
pip install 3tears-langgraph
```

## Usage

```python
from threetears.langgraph import (
    AsyncpgPoolAdapter,
    ThreeTierCheckpointSaver,
)

# Trusted service with direct asyncpg.Pool: wrap once
saver = ThreeTierCheckpointSaver(executor=AsyncpgPoolAdapter(pool))

# Sandboxed agent: NatsProxyL3Backend already implements
# AsyncQueryExecutor, pass it straight through
saver = ThreeTierCheckpointSaver(executor=nats_l3_backend)

graph = builder.compile(checkpointer=saver)
```

## Middleware

The package ships platform-level [`AgentMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware) for `langchain.agents.create_agent` — the framework-aligned successor to the old hand-rolled `AgentNodeHook` / `ToolNodeHook` protocols. Consumer-specific policy lives in each consumer as its own middleware; only the reusable platform seams live here:

- `PromptCachingMiddleware` (`wrap_model_call`) — annotates a leading bare-string system message with Anthropic `cache_control={"type": "ephemeral"}` when the model supports it, then normalizes cache-hit/creation counters onto `usage_metadata["cache_usage"]`. Non-Anthropic adapters degrade silently to bare-string system messages.
- `ToolResultOffloadMiddleware` (`wrap_tool_call`) — when a `ToolResultOffloader` is injected on `config["configurable"]` and a tool result exceeds `offload_threshold_chars`, stores the full content out-of-band and shows the model `"<summary>\n\n[ctx:<handle>]"` (the structured `artifact` is preserved). Opt-in: no offloader ⇒ byte-for-byte no-op.
- `ObjectCatalogMiddleware` (`wrap_tool_call`) — when a tool returns an `ObjectHandle` in its result artifact and an `ObjectCataloger` is injected, persists a catalog record under the verified call identity. Soft-fail side-effect: a catalog error never breaks the tool result.

```python
from langchain.agents import create_agent

from threetears.langgraph import (
    ObjectCatalogMiddleware,
    PromptCachingMiddleware,
    ToolResultOffloadMiddleware,
)

agent = create_agent(
    model=chat_anthropic,
    tools=tools,
    middleware=[
        PromptCachingMiddleware(),
        ToolResultOffloadMiddleware(),
        ObjectCatalogMiddleware(),
    ],
)
# after a run, PromptCachingMiddleware has stamped:
# message.usage_metadata["cache_usage"]
# == {"cache_read_input_tokens": ..., "cache_creation_input_tokens": ..., "cached_tokens": ...}
```

The offload / catalog contracts (`ToolResultOffloader`, `ObjectCataloger`) are pure structural `Protocol`s exported from `threetears.langgraph.offload` / `threetears.langgraph.catalog`; a consumer injects a concrete implementation on `config["configurable"]` (e.g. `tool_result_offloader`, `object_cataloger`) without the package taking any dependency on the consumer's context store.

See [`3tears/docs/prompt-caching.md`](../../docs/prompt-caching.md) for the full caching contract, summarization interaction, downstream wiring checklist, and a worked example.

## Streaming

The package ships `StreamingResponse`, a transport-agnostic primitive that owns the lifecycle of one streaming response: `start` -> any number of `emit_token` / `emit_tool_call_*` -> mutually-exclusive `end` (success) or `error` (failure) terminal. `run_graph(compiled_graph, state, config)` consumes a LangGraph `astream_events(version="v2")` loop with the start/end ordering managed; on graph exception it fires `error(code="AGENT_FAILED", ...)` and re-raises so the caller still sees the failure on the synchronous path.

The wire vocabulary is fixed: `StreamStartEvent` / `StreamTokenEvent` / `StreamEndEvent` / `StreamErrorEvent` / `ToolCallStartEvent` / `ToolCallEndEvent` / `ToolCallProgressEvent`, dispatched via the `StreamEvent` discriminated union and the `parse_stream_event(payload)` adapter. The transport seam is the `StreamTransport` Protocol -- one method, `async def publish(self, payload: bytes) -> None`. Any wire (NATS subject, websocket, chunked HTTP body) satisfies it.

```python
from threetears.langgraph import StreamingResponse, StreamTransport

class WebSocketStreamTransport:
    """example transport for a websocket consumer."""
    def __init__(self, ws): self._ws = ws
    async def publish(self, payload: bytes) -> None:
        await self._ws.send_bytes(payload)

stream = StreamingResponse(
    transport=WebSocketStreamTransport(ws),
    correlation_id=correlation_id,
    conversation_id=conversation_id,
    start_time_monotonic=request_start,
)
final_state = await stream.run_graph(compiled_graph, state, config)
```

A reference adapter can bind the primitive to a per-correlation-id stream subject via `nc.publish_raw`. Tool-call observation envelopes flow through `ToolCallProgressHook` reading the active `StreamingResponse` from `config["configurable"]["streaming_response"]`.
