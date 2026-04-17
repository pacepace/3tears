# 3tears-langgraph

Three-tier LangGraph checkpoint saver: L1 (SQLite) -> L2 (NATS KV) -> L3 (PostgreSQL).

L1 and L2 are optional cache layers that degrade gracefully on failure. L3 (PostgreSQL) is the source of truth.

## Installation

```bash
pip install 3tears-langgraph
```

## Usage

```python
from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver

saver = ThreeTierCheckpointSaver(postgres_pool=pool)
graph = builder.compile(checkpointer=saver)
```
