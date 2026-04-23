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
