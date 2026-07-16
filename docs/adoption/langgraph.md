# 3tears-langgraph

`threetears.langgraph` -- LangGraph integration. Three-tier checkpoint
savers plus opt-in offload/catalog middleware.

## Problem

LangGraph's checkpointing needs a backend, and a naive one either doesn't
scale across pods or doesn't work for sandboxed agents that hold no direct
database credentials. Building that backend, and making it work for both a
trusted service and a sandboxed worker, is a substantial piece of
infrastructure to get right on its own.

## What it does

- `ThreeTierCheckpointSaver` -- L1 SQLite -> L2 NATS KV -> L3 PostgreSQL,
  the same tiering `core` uses for entities.
- `ContextMergeMiddleware` for merging context across graph steps.
- Offload and catalog middleware for large tool outputs, strictly opt-in.

## Design philosophy

Cache layers degrade gracefully on failure, matching `core`'s tiering
philosophy: losing L1 or L2 is a performance event, not a correctness
event. One `AsyncQueryExecutor` protocol serves both trusted services
(direct asyncpg) and sandboxed agents (NATS-proxied L3) -- the same saver
works in both topologies without a fork. Offload and catalog features are
strictly opt-in: no offloader configured means byte-for-byte no-op, and
catalog failures are soft-fail side effects that must never break a tool
result.

## When to adopt

Any LangGraph agent running on the 3tears data layer, especially across
more than one pod where checkpoints need to stay coherent.

## Composes with

- [`core`](core.md) -- the same three-tier caching model, applied to
  checkpoints.
- [`nats`](nats.md) -- the L2 tier and the sandboxed-agent proxy path.
- [`agent-knowledge`](agent-knowledge.md) -- its `wrap_model_call`
  middleware targets LangGraph agents built here.
- [`media-contracts`](media-contracts.md) -- the catalog middleware's
  media-handling contract.

## Install

```bash
pip install 3tears-langgraph
```
