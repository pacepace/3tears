# Migration Guide: aibots to threetears.models

## Overview

The `threetears.models` package extracts provider protocols, circuit breakers, error translation, and caching from the aibots Gateway into a shared library. After migration, the aibots Gateway becomes thinner -- it routes NATS requests, enforces ACLs, and handles billing, but delegates model lifecycle to `threetears.models`.

**Moves to threetears.models:**
- Circuit breaker state machine (`CircuitBreakerRegistry`)
- Error translation (`friendly_api_error`, `identify_provider`)
- Provider instance caching (`ModelCache`)
- Chat/Embedding provider adapters and protocols

**Stays in aibots:**
- Wire error codes (`ERROR_PROVIDER_RATE_LIMITED`, `ERROR_PROVIDER_OVERLOADED`, etc.)
- Billing and cost recording
- NATS request routing and subject handling
- ACL enforcement (customer/agent access control)
- Agent lifecycle management

## Dependency update

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "3tears-models[anthropic,openai,voyageai]",
    # ... existing deps
]
```

## Files to delete

### `gateway/circuit_breaker.py`

Replaced entirely by `CircuitBreakerRegistry` from `threetears.models`.

**Before (aibots):**
```python
from aibots.gateway.circuit_breaker import CircuitBreaker, CIRCUIT_OPEN

breaker = CircuitBreaker(provider_name="anthropic", threshold=5, timeout=30)
if breaker.is_open():
    raise CircuitOpenException(provider_name)
```

**After (threetears.models):**
```python
from threetears.models import CircuitBreakerRegistry, CircuitOpenError

registry = CircuitBreakerRegistry(failure_threshold=5, recovery_timeout_seconds=30.0)
breaker = registry.get("anthropic")
breaker.check()  # raises CircuitOpenError if open
```

## Files to modify

### `gateway/provider_pool.py`

Replace hand-rolled provider pool with `ModelCache` and provider adapters.

**Before:**
```python
from aibots.gateway.provider_pool import ProviderPool

pool = ProviderPool()
model = pool.get_or_create("anthropic", "claude-sonnet-4-20250514", api_key)
response = await model.ainvoke(messages)
```

**After:**
```python
from threetears.models import ModelCache
from threetears.models.providers.anthropic import AnthropicChatProvider

cache = ModelCache()
provider = cache.get("anthropic", "claude-sonnet-4-20250514")
if provider is None:
    provider = AnthropicChatProvider("claude-sonnet-4-20250514", api_key)
    cache.put("anthropic", "claude-sonnet-4-20250514", provider)

messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
result = await provider.complete(messages)
```

### `gateway/handlers/completion.py`

Add `friendly_api_error()` for user-facing error text in NATS error responses.

**Before:**
```python
except Exception as exc:
    error_text = f"Provider error: {exc}"
    return NatsErrorResponse(code=ERROR_PROVIDER_UNKNOWN, message=error_text)
```

**After:**
```python
from threetears.models import friendly_api_error

except Exception as exc:
    error_text = friendly_api_error(exc)
    return NatsErrorResponse(code=ERROR_PROVIDER_UNKNOWN, message=error_text)
```

### `gateway/handlers/embedding.py`

Replace raw LangChain embedding calls with typed providers.

**Before:**
```python
from langchain_openai import OpenAIEmbeddings

model = OpenAIEmbeddings(model=model_name, api_key=api_key)
vectors = await model.aembed_documents([text])
result = {"vector": vectors[0], "dimensions": len(vectors[0])}
```

**After:**
```python
from threetears.models.providers.openai import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider(model_name, api_key)
result = await provider.embed(text)
# result is EmbeddingResult with .vector, .dimensions, .model, .token_count
```

## What stays in aibots

These concerns remain in the Gateway because they are platform-specific:

- **Wire error codes**: `ERROR_PROVIDER_RATE_LIMITED`, `ERROR_PROVIDER_OVERLOADED`, `ERROR_PROVIDER_AUTH`, `ERROR_PROVIDER_UNAVAILABLE` -- these are NATS envelope codes, not model-layer concerns
- **Billing**: Cost recording, credit deduction, usage aggregation per customer
- **ACL**: Which agents can access which providers, model-level allowlists
- **NATS routing**: Subject patterns, queue groups, request/reply envelopes
- **Agent lifecycle**: Agent-to-provider mapping, hot-swap on key rotation
