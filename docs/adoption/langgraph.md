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
- `threetears.langgraph.fence` -- the fence around material in a prompt (below).

## The fence around material

A model reads everything in its prompt the same way, so a stored memory, a
document or a tool's preview that says "ignore your instructions" can be
followed. `untrusted_fence(nonce, text)` wraps material in
`<untrusted nonce=X>` ... `</untrusted nonce=X>` and disarms any fence tag
inside it, so planted text cannot close its own fence; the nonce therefore need
not be secret. `untrusted_rule(nonce)` says what the fence means.

- **Put the rule where the messages go to the model.** `with_fence_rules(messages)`
  adds the rule for every fence the call carries to its system prompt;
  `rules_missing(texts, explained=prompt)` does the same for a prompt handed over as
  a string. A site then only fences.
- **A block placed in a prompt someone else assembles carries its own rule:**
  `explained_fence(text)`. Its nonce is derived from the text (`nonce_for`), so an
  unchanged block renders byte-identical and a cached system prompt stays cached.

What the platform fences itself: the memory block (`MemoryRetriever.retrieve`'s
`RetrievalResult.context`: memories, media excerpts, chunk headlines), the memory
ledger and tool-result previews (`ToolContextManager`), the stored memories the
dream and extraction's resolution step read, and a document under
`media_analyze`.

What it does not:

- **A tool's return** reaches the model through the consumer's tool loop, which
  should fence it there. The platform's own agent path does not yet.
- **Text a model wrote** -- an extracted memory, a conversation variable, a
  summary -- is passed on as that model's words. It was written from fenced
  material, and fencing it would tell the next model to distrust its own
  pipeline. This is metallm's owner ruling of 2026-09-23, adopted here.

## Design philosophy

Cache layers degrade gracefully on failure, matching `core`'s tiering
philosophy: losing L1 or L2 is a performance event, not a correctness
event. L3 is a correctness event and reaches the caller -- except that
`aput_writes` runs in LangGraph's executor teardown, where raising ends a
turn that has already answered, so a lost *crash-recovery* write degrades
with a warning. That carve-out is scoped by channel and not by tier: a
write on a control channel (`__interrupt__`, `__resume__`, `__error__`,
`__scheduled__`) still raises, because degrading one would turn a
human-approval pause into an ordinary end of turn.
One `AsyncQueryExecutor` protocol serves both trusted services
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
