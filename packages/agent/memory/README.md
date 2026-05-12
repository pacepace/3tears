# 3tears Agent Memory

Memory system for LLM agents. Handles extraction of memorable facts from conversations, hybrid retrieval (semantic + full-text + recency), and memory lifecycle management.

Part of the [3tears](https://github.com/pacepace/3tears) framework.

## Installation

```bash
pip install 3tears-agent-memory
```

## Components

### Collections are the single entry point for memory-table SQL

Every memory-table write, single-row read, batch read, and hybrid-search query goes through one of four `BaseCollection` subclasses; no consumer of this package holds an `asyncpg.Pool` reference directly (namespace-task-01 phase 8.5b).

- `MemoriesCollection` — `memories` table. CRUD through `get` / `save_entity` / `delete`; complex queries through `hybrid_search`, `search_by_ids`, `search_by_semantic`, `search_by_fts`, `find_similar_for_dedup`, `count_by_user`, `fetch_content_for_recall`.
- `MediaCollection` — `media` parent table. CRUD only.
- `MediaContentCollection` — `media_content` child table. CRUD + `hybrid_search`, `search_by_ids`, `search_by_semantic`, `search_by_fts`, `fetch_content_for_recall`.
- `MemoryChunkCollection` — `memory_chunks` child table. CRUD + `hybrid_search`, `search_by_ids`, `search_by_semantic`, `fetch_content_for_recall`.

All four resolve their L3 pool through `CollectionRegistry` (same pattern `ConversationCollection` uses); an L1 `SQLiteBackend` attached to the registry populates on `save_entity` and serves subsequent by-id `get` calls without an L3 round-trip. The orphan tables `media` / `media_content` / `memory_chunks` — introduced by migrations v006 / v007 — have Collection coverage for the first time in this release.

Hybrid-search methods carry documented `# cache-bypass: <reason>` inline comments because the query shape (vector distance, FTS rank, multi-table joins) is not primary-key-addressable and the L1 row cache cannot serve the lookup. Keeping the SQL on the Collection preserves the single entry point — the cache-primitive enforcement walker recognises in-Collection bypass sites as legitimate and reports any bypass that leaks back into `retrieval.py` / `extraction.py` / `tools.py` as a violation.

### MemoryExtractor

Extracts memorable facts from conversation turns. Uses a multi-stage pipeline: candidate extraction via LLM, deduplication against existing memories via embedding similarity, and action resolution (ADD / UPDATE / DELETE).

```python
from threetears.agent.memory import (
    MemoriesCollection,
    MemoryConfig,
    MemoryExtractor,
)

extractor = MemoryExtractor(
    config=MemoryConfig(),
    embedding_provider=my_embedding_provider,
    chat_model_factory=my_chat_model_factory,
    authorizer=authorizer_bundle,
    memories_collection=memories_collection,
    summary_callback=on_new_memory,
)

await extractor.extract(
    user_id=user_id,
    conversation_id=conv_id,
    message_id_source=msg_id,
    user_message="I just moved to Portland",
    assistant_response="That's exciting! Portland has great food...",
    turn_count=5,
    agent_id=agent_id,
    customer_id=customer_id,
)
```

### MemoryRetriever

Retrieves relevant memories using hybrid search: pgvector semantic similarity, PostgreSQL full-text search, recency decay, and MMR reranking for diversity. Takes the three search-bearing Collections at construction — no pool.

```python
from threetears.agent.memory import MemoryRetriever, MemoryConfig

retriever = MemoryRetriever(
    config=MemoryConfig(),
    embedding_provider=my_embedding_provider,
    authorizer=authorizer_bundle,
    memories_collection=memories_collection,
    media_content_collection=media_content_collection,
    memory_chunk_collection=memory_chunk_collection,
)

result = await retriever.retrieve_with_candidates(
    user_id,
    "Tell me about Portland",
    agent_id=agent_id,
    customer_id=customer_id,
    caller_user_id=user_id,
    caller_agent_id=agent_id,
)

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

LangChain tools for agent use — memory search, recall, and explicit add. Factories take Collection references; no pool:

```python
from threetears.agent.memory import (
    load_add_memory_tool,
    load_memory_search_tool,
    load_recall_memory_tool,
)

search_tool = await load_memory_search_tool(
    user_id=user_id,
    embedding_provider=embedding_provider,
    agent_id=agent_id,
    customer_id=customer_id,
    authorizer=authorizer_bundle,
    memories_collection=memories_collection,
    media_content_collection=media_content_collection,
    memory_chunk_collection=memory_chunk_collection,
)
recall_tool = await load_recall_memory_tool(
    user_id=user_id,
    agent_id=agent_id,
    customer_id=customer_id,
    authorizer=authorizer_bundle,
    memories_collection=memories_collection,
    media_content_collection=media_content_collection,
    memory_chunk_collection=memory_chunk_collection,
)
add_tool = await load_add_memory_tool(
    user_id=user_id,
    conversation_id=conv_id,
    message_id=msg_id,
    embedding_provider=embedding_provider,
    agent_id=agent_id,
    customer_id=customer_id,
    authorizer=authorizer_bundle,
    memories_collection=memories_collection,
)
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

- `acl_cache` — shared `threetears.agent.acl.AclCache` instance;
- `membership_loader` + `grant_loader` — the evaluator's loaders (`threetears.agent.acl.MembershipLoader` / `GrantLoader`);
- `namespace_collection` — three-tier `NamespaceCollection` used to resolve the memory namespace via `get_by_owner_and_customer(namespace_type="memory", owner_agent_id, customer_id)` (create-if-absent flows through `save_entity`);
- `group_collection` + `group_member_collection` + `role_collection` + `role_assignment_collection` — the rbac Collections the first-write owner-assignment path uses via `ensure_memory_owner_assignment(...)`.

There is no bypass. Every `MemoriesCollection`, `MemoryRetriever`, `MemoryExtractor`, and LangChain tool factory (`load_memory_search_tool`, `load_add_memory_tool`, `load_recall_memory_tool`) takes the bundle as a required constructor/factory argument; every code path that touches a memory row runs `authorize_memory_access` first. Callers that omit the bundle fail at the type checker and the Python signature boundary.

- Production wiring builds the bundle directly from the agent-side three-tier stack's Collections (`NatsProxyL3Backend`-backed `NamespaceCollection` / `GroupCollection` / ...); the canonical example lives in `MemoryIntegration` in the `aibots-agents` runtime.
- Test wiring injects a permissive fixture `permissive_memory_authorizer` (see `tests/conftest.py`) that carries in-memory Collection stand-ins and a permissive evaluator. Fixture usage is explicit in every test file that constructs a memory surface.
- Back-office / admin tooling that genuinely needs to read or write memories without an identity must construct its own bundle with hub-side Collections (directly bound to the hub's asyncpg pool) — there is no global escape hatch.

See `threetears.agent.memory.authorize` for the full public surface.

Hub migrations v020 (role seeds) and v021 (cross-agent-schema backfill) land the platform-side rbac rows required for evaluator resolution; the three platform roles (`MemoryOwner` / `MemoryReader` / `MemoryWriter`) carry the canonical action vocabulary.
