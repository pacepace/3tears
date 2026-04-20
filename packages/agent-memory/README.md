# 3tears Agent Memory

Memory system for LLM agents. Handles extraction of memorable facts from conversations, hybrid retrieval (semantic + full-text + recency), and memory lifecycle management.

Part of the [3tears](https://github.com/pacepace/3tears) framework.

## Installation

```bash
pip install 3tears-agent-memory
```

## Components

### MemoryExtractor

Extracts memorable facts from conversation turns. Uses a multi-stage pipeline: candidate extraction via LLM, deduplication against existing memories via embedding similarity, and action resolution (ADD / UPDATE / DELETE).

```python
from threetears.agent.memory import MemoryExtractor, MemoryConfig, EmbeddingProvider, ChatModelFactory

extractor = MemoryExtractor(
    config=MemoryConfig(),
    embedding_provider=my_embedding_provider,
    chat_model_factory=my_chat_model_factory,
    summary_callback=on_new_memory,
)

await extractor.extract(
    pool=db_pool,
    user_id=user_id,
    conversation_id=conv_id,
    message_id_source=msg_id,
    user_message="I just moved to Portland",
    assistant_response="That's exciting! Portland has great food...",
    turn_count=5,
)
```

### MemoryRetriever

Retrieves relevant memories using hybrid search: pgvector semantic similarity, PostgreSQL full-text search, recency decay, and MMR reranking for diversity.

```python
from threetears.agent.memory import MemoryRetriever, MemoryConfig

retriever = MemoryRetriever(config=MemoryConfig(), embedding_provider=my_embedding_provider)

result = await retriever.retrieve_with_candidates(pool, user_id, "Tell me about Portland")

# result.context     — formatted string for injection into system prompt
# result.memories    — raw memory dicts with similarity scores
# result.media_content — matched media content
# result.memory_chunks — matched document chunks
```

### Protocols

Implement these to integrate with your infrastructure:

```python
from threetears.agent.memory import EmbeddingProvider, ChatModelFactory

class MyEmbeddingProvider(EmbeddingProvider):
    async def embed(self, text: str) -> tuple[list[float], int, UUID]:
        # Returns (embedding_vector, token_count, model_id)
        ...

class MyChatModelFactory(ChatModelFactory):
    async def create_chat_model(self, purpose: str = "extraction"):
        # Returns a langchain BaseChatModel
        ...
```

### Tools

LangChain tools for agent use — memory search and recall:

```python
from threetears.agent.memory import load_memory_search_tool, load_recall_memory_tool

search_tool = load_memory_search_tool(pool, user_id, embedding_provider, tool_context)
recall_tool = load_recall_memory_tool(pool, user_id)
```

### Configuration

```python
from threetears.agent.memory import MemoryConfig

config = MemoryConfig(
    similarity_threshold=0.4,      # minimum cosine similarity for retrieval
    detail_threshold=0.85,         # threshold for including full memory detail
    context_budget=15,             # max memories in context
    dedup_threshold=0.85,          # similarity threshold for deduplication
    max_candidates=10,             # max candidates per extraction
)
```

## Database Schema

Requires PostgreSQL with the `pgvector` extension. The package's own migration runner (`threetears.agent.memory.migrations.register`) produces the full schema per agent schema. Registered versions:

- **v001** — `memories` (PK `memory_id`, pgvector `embedding`, scoping ids, content, summary, lifecycle timestamps).
- **v002** — `conversation_memory_refs` (ledger of per-conversation surfaced items).
- **v003** — column reconciliation: renames PK and discriminator to match the package code (`id`→`memory_id`, `memory_type`→`type_memory`), drops columns the code does not read (`embedding_model`, `importance`, `metadata`, `date_accessed`), loosens `agent_id`/`customer_id` to NULL.
- **v004** — lifecycle + conversation-link columns on `memories` (`conversation_id`, `message_id_source`, `is_deleted`, `media_id`, `date_deleted`, `summary`) with indexes.
- **v005** — FTS: `search_vector TSVECTOR` + GIN index + maintenance trigger on `memories`.
- **v006** — `media` (parent) + `media_content` (chunked extracted text with embedding + FTS).
- **v007** — `memory_chunks` (document-style chunks with heading / page metadata + embedding + FTS).

Every FTS column is trigger-maintained from `content` + `summary` (weighted A/B) — callers do not have to populate `search_vector` manually. Integration tests under `tests/integration/` exercise the full chain + every public API surface against `pgvector/pgvector:pg16` via testcontainers.

## RBAC Enforcement (namespace-task-01 Phase 3)

Memory reads, writes, and extractions flow through the unified rbac evaluator in `threetears.agent.acl`. Every (agent, customer) pair is a `memory`-type namespace in `platform.namespaces`; each access resolves the namespace and evaluates one of three canonical actions against the caller's `(user_id, agent_id)` pair:

- `memory.read` — retrieval / search / recall. Guarded on `MemoryRetriever.retrieve*`, `MemoriesCollection.find_by_user`, `MemoriesCollection.find_by_scope`, the `memory_search` + `recall_memory` LangChain tools.
- `memory.write` — user-initiated writes. Guarded on `MemoriesCollection.save_memory` and the `add_memory` LangChain tool.
- `memory.extract` — agent-internal extraction path. Guarded on `MemoryExtractor.extract`; the owner short-circuit keeps the common case (agent emitting memories on its own namespace) grant-free.

Owner short-circuit: the evaluator allows any action when the calling agent owns the memory namespace. Agent-internal retrieval and extraction therefore work without explicit grants; user-initiated reads and writes require evaluator assignments.

Auto-assignment on first user-write: `add_memory` ensures a `MemoryOwner` assignment for the calling user on their first write (idempotent-by-state — the ensurer only fires when the user has zero memory rows in the target schema). Subsequent writes authorize against the materialized grant; admin-revoked grants stay revoked (the ensurer does not resurrect them).

Wiring shape: every consumer of the memory surface REQUIRES a `MemoryAuthorizerDependencies` bundle exposing:

- `membership_loader` + `grant_loader` — the evaluator's loaders (`threetears.agent.acl.MembershipLoader` / `GrantLoader`);
- `namespace_resolver` — async `(agent_id, customer_id) -> MemoryNamespaceRow | None` (create-if-absent is the resolver's responsibility; typically NATS-request-to-hub in production, in-memory fixture in tests);
- `assignment_ensurer` — async `(user_id, memory_namespace_row) -> None` for the auto-assignment path.

There is no bypass. Every `MemoriesCollection`, `MemoryRetriever`, `MemoryExtractor`, and LangChain tool factory (`load_memory_search_tool`, `load_add_memory_tool`, `load_recall_memory_tool`) takes the bundle as a required constructor/factory argument; every code path that touches a memory row runs `authorize_memory_access` first. Callers that omit the bundle fail at the type checker and the Python signature boundary.

- Production wiring builds the bundle from hub-side loaders + a namespace resolver (NATS request-reply against the hub broker) + the first-write assignment ensurer; the canonical example lives in `MemoryIntegration` in the `aibots-agents` runtime.
- Test wiring injects a permissive fixture `permissive_memory_authorizer` (see `tests/conftest.py`) that allows every evaluate and no-ops the ensurer. Fixture usage is explicit in every test file that constructs a memory surface.
- Back-office / admin tooling that genuinely needs to read or write memories without an identity must construct its own bundle (typically reusing the hub-side loaders directly and a no-op ensurer) — there is no global escape hatch.

See `threetears.agent.memory.authorize` for the full public surface.

Hub migrations v020 (role seeds) and v021 (cross-agent-schema backfill) land the platform-side rbac rows required for evaluator resolution; the three platform roles (`MemoryOwner` / `MemoryReader` / `MemoryWriter`) carry the canonical action vocabulary.
