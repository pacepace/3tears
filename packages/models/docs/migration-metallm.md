# Migration Guide: metallm to threetears.models

## Overview

metallm has the most to migrate. The `threetears.models` package replaces metallm's provider layer, embedding service, whisper service, image backends, circuit breaker, and streaming recovery logic. After migration, metallm focuses on personality, NSFW handling, content pipeline, and frontend API.

**Moves to threetears.models:**
- Chat providers (Anthropic, OpenAI, OpenRouter)
- Embedding service (OpenAI, VoyageAI)
- Whisper transcription service
- Image generation backends (OpenAI, HuggingFace, A1111, ModelsLab, ComfyUI)
- Circuit breaker
- Streaming chunk merging and tool call recovery
- Message preprocessing (alternating roles, vision content)

**Stays in metallm:**
- NSFW content detection and filtering
- Personality system and persona management
- Content pipeline (summarization, extraction, formatting)
- Frontend HTTP API
- DuckDB local caching layer

## Dependency update

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "3tears-models[anthropic,openai,openrouter,voyageai,whisper,image]",
    # ... existing deps
]
```

## Files to delete

### `graph/providers.py`

Provider creation and management replaced by `threetears.models` adapters and `ModelCache`.

### `services/embedding.py`

Replaced by `OpenAIEmbeddingProvider` and `VoyageAIEmbeddingProvider`.

**Before:**
```python
from metallm.services.embedding import EmbeddingService

svc = EmbeddingService(provider="openai", model="text-embedding-3-small", api_key=key)
vector = await svc.embed(text)
```

**After:**
```python
from threetears.models.providers.openai import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider("text-embedding-3-small", api_key)
result = await provider.embed(text)
vector = result.vector
```

### `services/whisper.py`

Replaced by `WhisperTranscriptionProvider`.

**Before:**
```python
from metallm.services.whisper import WhisperService

svc = WhisperService(api_key=key)
text = await svc.transcribe(audio_bytes, mime_type="audio/mp3")
```

**After:**
```python
from threetears.models.providers.whisper import WhisperTranscriptionProvider

provider = WhisperTranscriptionProvider(api_key=key)
result = await provider.transcribe(audio_bytes, "audio/mp3")
text = result.text
segments = result.segments  # optional time-aligned segments
```

### `services/image_backends/*.py`

All image generation backends replaced by `threetears.models.providers.image` module.

**Before:**
```python
from metallm.services.image_backends.openai_backend import OpenAIImageBackend

backend = OpenAIImageBackend(api_key=key)
image_bytes = await backend.generate(prompt, size="1024x1024")
```

**After:**
```python
from threetears.models.providers.image import OpenAIImageProvider

provider = OpenAIImageProvider(api_key=key)
# provider implements ImageGenerationProvider protocol
```

### `cache/circuit_breaker.py`

Replaced by `CircuitBreakerRegistry`.

## Files to modify

### `graph/nodes/personality.py`

Replace inline streaming recovery with shared helpers.

**Before:**
```python
# inline chunk accumulation and tool call fixup
content_parts = []
tool_calls = []
for chunk in chunks:
    content_parts.append(chunk.content)
    if chunk.tool_calls:
        tool_calls.extend(chunk.tool_calls)
# manual split detection
if len(tool_calls) >= 2:
    fixed = _fix_split_tool_calls(tool_calls)
```

**After:**
```python
from threetears.models.streaming import merge_chunks, recover_split_tool_calls

result = merge_chunks(chunks)
if result.tool_calls:
    result = ChatResult(
        content=result.content,
        tool_calls=recover_split_tool_calls(result.tool_calls),
        model=result.model,
        usage=result.usage,
    )
```

Use `preprocess_messages()` before sending to providers that require alternating roles:

```python
from threetears.models.preprocessing import preprocess_messages
from threetears.models.capabilities import ModelCapabilities

capabilities = ModelCapabilities(
    model_name="claude-sonnet-4-6",
    model_type=ModelType.CHAT,
    model_tier=ModelTier.LARGE,
    model_status=ModelStatus.ACTIVE,
    requires_alternating_roles=True,
)
processed = preprocess_messages(messages, capabilities)
result = await provider.complete(processed)
```

### `services/media_adapters.py`

Replace inline base64 encoding with `format_vision_content()`.

**Before:**
```python
import base64
b64 = base64.b64encode(image_bytes).decode()
content = [
    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    {"type": "text", "text": prompt},
]
```

**After:**
```python
from threetears.models.preprocessing import format_vision_content

content = format_vision_content(image_bytes, mime_type, prompt)
```

## What stays in metallm

These concerns remain because they are application-specific:

- **NSFW handling**: Content detection, filtering, and policy enforcement
- **Personality system**: Persona definitions, system prompt generation, personality switching
- **Content pipeline**: Summarization chains, extraction pipelines, output formatting
- **Frontend API**: HTTP routes, WebSocket handlers, session management
- **DuckDB caching**: Local query cache, conversation history, analytics aggregation
