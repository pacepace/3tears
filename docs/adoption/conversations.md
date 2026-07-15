# 3tears-conversations

`threetears.conversations` -- `Conversation` entity, three-tier
`ConversationsCollection`, and per-agent schema migrations.

## Problem

Multiple downstream packages (`agent-memory`, `agent-tools`'s context items,
`agent-workspace` bindings) all key off `conversation_id`. Without a single
owner, each consumer either duplicates the table definition or takes on an
awkward dependency on whichever package happened to define it first.

## What it does

- `Conversation` entity and `ConversationsCollection` built on `core`'s
  three-tier caching.
- Per-agent schema migrations that create and evolve the `conversations`
  table.

## Design philosophy

`conversations` exists because no single consumer package is the natural
owner of the table. Centralizing it here lets `agent-tools` and
`agent-workspace` each depend on `conversations` without pulling in one
another's unrelated pipelines. (`agent-memory` does not take a package
dependency on it -- it threads a bare `conversation_id` parameter through
instead.)

## When to adopt

Any app using more than one of the agent-framework packages that reference
conversations -- adopt this first so they share one table instead of each
assuming ownership.

## Composes with

- [`core`](core.md) -- the three-tier entity/collection base.
- [`agent-acl`](agent-acl.md) -- authorization for conversation access.
- [`langgraph`](langgraph.md) -- conversation events integrate with
  LangGraph-built agents.
- [`agent-tools`](agent-tools.md), [`agent-workspace`](agent-workspace.md) --
  typical consumers.

## Install

```bash
pip install 3tears-conversations
```
