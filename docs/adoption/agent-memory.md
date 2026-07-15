# 3tears-agent-memory

`threetears.agent.memory` -- memory extraction, retrieval, hybrid search,
and MMR reranking for LLM agents.

## Problem

An agent that can't remember prior conversations re-asks the same questions
and re-learns the same facts every session. Building extraction, storage,
and relevant-fact retrieval from scratch -- and getting the access control
right, so an agent can freely manage its own memory but a user-initiated
action still needs an explicit grant -- is a substantial, easy-to-get-wrong
subsystem on its own.

## What it does

- Extraction of memorable facts from conversations.
- Hybrid retrieval (keyword + semantic) with Maximal Marginal Relevance
  (MMR) reranking for relevance without redundancy.
- A single `Collection` entry point for all memory-table SQL, so no
  consumer holds a raw pool reference.
- Owner short-circuit access control: an agent freely reads/writes/extracts
  its own memory namespace; user-initiated actions require an explicit RBAC
  grant.

## Design philosophy

Memory is hard-delete-only: a fact is true, or it's gone -- never
soft-deleted, archived, or put in a "pending" status. Retrieved facts do
carry a decaying salience *ranking* weight that fades unless reinforced,
but that affects retrieval ranking, not whether the fact exists. This
distinction is why agent-internal deliberation state (see
`agent-intention`) is deliberately kept out of this package rather than
added as a memory `kind`: intentions have a genuine status lifecycle
(open/acted/dismissed), and that's a fundamentally different,
agent-internal control construct from a fact taxonomy that only ever grows
or gets hard-deleted.

## When to adopt

Any LLM agent that needs to persist and recall facts across sessions.
Requires pgvector for semantic search.

## Composes with

- [`core`](core.md) -- the three-tier collection base.
- [`agent-acl`](agent-acl.md) -- authorization for non-owner access.

Extraction takes a bare `conversation_id` parameter -- it does not carry a
package dependency on `conversations`.
- [`agent-intention`](agent-intention.md) -- deliberately separate rather
  than a memory `kind`; see that doc for why.

## Install

```bash
pip install 3tears-agent-memory
# requires: CREATE EXTENSION vector;
```
