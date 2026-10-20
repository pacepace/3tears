# 3tears Agent Intention

Standing-wants corpus for LLM agents. An intention is agent-authored deliberation
output -- a want the agent is holding onto ("ask Pace about the migration", "check
whether the wake threads are firing") -- with its own status lifecycle, salience
decay, and restraint brakes so the agent surfaces a want at most occasionally
rather than every turn.

Part of the [3tears](https://github.com/pacepace/3tears) framework.

## Installation

```bash
pip install 3tears-agent-intention
```

## Why a separate package (not a memory of `kind=intention`)

Memory is a hard-delete-only user-fact taxonomy with immutable scope columns and no
lifecycle. An intention is all lifecycle: `open -> asked -> granted / dropped`, a
partial index for "find the open wants", and a validated status value set. Riding the
shared `memory_type` PG enum would pollute a user-fact taxonomy with an agent-internal
control construct and force an irreversible enum add on six consumers.

Enhance-first is still honoured: intention **consumes** the reusable salience/decay
substrate the `agent/memory` enhancement built (the same `salience` column shape and
the shared `apply_salience_decay` helper), rather than reinventing it.

## Components

### `intentions` table

Partition on `agent_id`; composite PK `(agent_id, intention_id)`; CAS on `date_updated`
(mirrors memory). Columns: `status` (PG enum `intention_status`), `content`,
`embedding` (pgvector, 1024 wide unless configured -- see below; for dedup on log), `salience` (NUMERIC(5,4), reuses the
decay substrate), `last_decayed_at` (decay anchor), `last_surfaced_at` (cooldown
anchor), `source_memory_id` / `source_conversation_id` (soft-ref provenance, no FK).

### Embedding width

`intentions.embedding` is a plain pgvector `vector(N)`, and N is a deployment setting: agent-memory's width, so one embedding provider serves both corpora. It defaults to **1024**, so a host that configures nothing is unaffected. Give the same `embedding_dim` to `threetears.agent.intention.migrations.register(runner, embedding_dim=...)`, `IntentionsCollection(..., embedding_dim=...)` and `intentions_table(metadata, embedding_dim=...)`.

- **Ceilings.** PostgreSQL's pgvector HNSW index takes at most **2,000** dimensions; YugabyteDB's `ybhnsw` up to **16,000** (no `halfvec` there, so the column stays `vector`). `register` refuses a width below 1 or above 16,000; v001 refuses a width above 2,000 on PostgreSQL before any statement runs.
- **Writes are checked.** A save whose embedding is not the collection's width is refused with a `ValueError` naming both widths, before any SQL runs.
- **An existing schema keeps its width.** `await verify_embedding_width(store, embedding_dim)` (`threetears.agent.intention.embedding_width`, also exported from the package root) raises `EmbeddingWidthMismatch` when `intentions.embedding` is another width.
- **Changing it is an operator action.** `await resize_embedding_width(store, embedding_dim)` drops `ix_intentions_embedding_hnsw`, alters the column with `USING NULL`, recreates the index under its own name and options, and returns how many embeddings it cleared. Every want keeps its content and loses its embedding, so re-embed them. Nothing runs it automatically. **Run it with writes to these tables stopped:** on YugabyteDB the column change is a table rewrite, and writes made while it runs may be lost ("Concurrent DMLs may not be reflected in the new table", yugabyte-db issue 19860).

### User isolation is a `user_id` WHERE clause, not RBAC

Every metallm user shares one `agent_id`, so the partition isolates nothing and the
agent-owner RBAC short-circuit sees every want. `user_id` is the only boundary: the
collection's user-facing reads take `user_id` as a **required** parameter, and every
consumer read filters on it.

## Versioning policy

The package version moves in **lockstep** with the rest of the 3tears monorepo -- every
package tracks the framework git tag.
