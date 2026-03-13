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

Requires PostgreSQL with pgvector. Memories are stored with embeddings for semantic search, tsvector columns for full-text search, and timestamps for recency decay. See the 3tears documentation for the full schema.
