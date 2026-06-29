# 3tears-models

LangChain-native AI model factories for the 3tears framework. Build chat and embedding models from a single call, with capability metadata, circuit breakers, error translation, and usage tracking wired in.

```bash
pip install 3tears-models
```

## What you get

- **Factories** -- `create_chat_model()` and `create_embedding_model()` return standard LangChain `BaseChatModel` / `Embeddings` instances. No custom runtime protocols to learn.
- **Providers** -- Anthropic, OpenAI, OpenRouter, VoyageAI, Whisper, and image backends (OpenAI Images, HuggingFace, A1111, ModelsLab, ComfyUI).
- **Capability registry** -- `get_capabilities()`, `register_capabilities()`, and per-model overrides describe context windows, vision support, tool support, and tier.
- **Circuit breakers** -- `CircuitBreaker` and `CircuitBreakerRegistry` trip on repeated provider failures and recover on a timer.
- **Usage tracking** -- `UsageTracker.record()` emits an OpenTelemetry span plus Prometheus instruments, with optional per-application audit and counter sinks.
- **Message hygiene** -- `preprocess_messages()`, `enforce_alternating_roles()`, `filter_invalid_tool_calls()`, and streaming chunk helpers (`parse_chunk`, `merge_chunks`).

## Quickstart

```python
from threetears.models import create_chat_model, create_embedding_model

chat = create_chat_model("claude-sonnet-4-6")
reply = await chat.ainvoke("Summarize three-tier caching in one sentence.")

embedder = create_embedding_model("voyage-3")
vectors = await embedder.aembed_documents(["first doc", "second doc"])
```

Capabilities, circuit breakers, and usage tracking attach automatically. Reach for the registry and tracker APIs directly when you need to inspect or override them.

## License

MIT. See [LICENSE](LICENSE).
