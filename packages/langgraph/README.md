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

## Prompt caching

The package ships `PromptCachingHook`, an `AgentNodeHook` implementation that rewrites the system prompt with Anthropic `cache_control={"type": "ephemeral"}` annotations and memoizes tool binding across turns. Non-Anthropic adapters degrade silently to bare-string system messages.

```python
from threetears.langgraph import PromptCachingHook, agent_node

config = {
    "configurable": {
        "chat_model": chat_anthropic,
        "system_prompt": long_prompt,
        "_hooks": {"agent": [PromptCachingHook()]},
    },
}
result = await agent_node(state, config)
usage = result["messages"][0].usage_metadata["cache_usage"]
# {"cache_read_input_tokens": ..., "cache_creation_input_tokens": ..., "cached_tokens": ...}
```

See [`3tears/docs/prompt-caching.md`](../../docs/prompt-caching.md) for the full contract, summarization interaction, downstream wiring checklist, and a worked example.
