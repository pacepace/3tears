"""integration tests for cache and registry working together."""

from __future__ import annotations

from threetears.models.cache import ModelCache
from threetears.models.registry import ProviderRegistry


class TestCacheAndRegistry:
    """integration tests for ModelCache and ProviderRegistry cooperation."""

    def test_cache_and_registry_together(self) -> None:
        """provider registry resolves module, cache stores provider instance."""
        registry = ProviderRegistry()

        # register a stdlib module as stand-in provider (avoids real SDK imports)
        registry.register("test-json", "json")
        module = registry.get_provider("test-json")

        cache = ModelCache()
        cache.put("test-json", "json-model", module)

        cached = cache.get("test-json", "json-model")
        assert cached is module
        assert cache.size() == 1

    def test_registry_resolves_multiple_then_cache_stores(self) -> None:
        """multiple resolved modules stored in cache independently."""
        registry = ProviderRegistry()
        registry.register("stdlib-json", "json")
        registry.register("stdlib-os", "os")

        json_module = registry.get_provider("stdlib-json")
        os_module = registry.get_provider("stdlib-os")

        cache = ModelCache()
        cache.put("stdlib-json", "model-a", json_module)
        cache.put("stdlib-os", "model-b", os_module)

        assert cache.get("stdlib-json", "model-a") is json_module
        assert cache.get("stdlib-os", "model-b") is os_module
        assert cache.size() == 2

    def test_cache_evict_provider_clears_all_models(self) -> None:
        """evicting provider removes all models for that provider from cache."""
        cache = ModelCache()
        sentinel_a = object()
        sentinel_b = object()
        sentinel_c = object()

        cache.put("anthropic", "claude-sonnet-4-20250514", sentinel_a)
        cache.put("anthropic", "claude-haiku", sentinel_b)
        cache.put("openai", "gpt-4o", sentinel_c)

        assert cache.size() == 3

        removed = cache.evict_provider("anthropic")
        assert removed == 2
        assert cache.get("anthropic", "claude-sonnet-4-20250514") is None
        assert cache.get("anthropic", "claude-haiku") is None
        assert cache.get("openai", "gpt-4o") is sentinel_c
        assert cache.size() == 1

    def test_cache_size_tracking(self) -> None:
        """cache accurately tracks size across put and evict operations."""
        cache = ModelCache()
        assert cache.size() == 0

        cache.put("p1", "m1", object())
        assert cache.size() == 1

        cache.put("p1", "m2", object())
        assert cache.size() == 2

        cache.put("p2", "m1", object())
        assert cache.size() == 3

        cache.evict("p1", "m1")
        assert cache.size() == 2

        cache.evict_provider("p1")
        assert cache.size() == 1

        cache.clear()
        assert cache.size() == 0

    def test_registry_custom_provider_cached(self) -> None:
        """custom registered provider resolves and caches correctly."""
        registry = ProviderRegistry()
        registry.register("custom", "collections")
        module = registry.get_provider("custom")

        cache = ModelCache()
        cache.put("custom", "custom-v1", module)

        # second resolve returns same cached module from registry
        module_again = registry.get_provider("custom")
        assert module_again is module

        # cache returns same instance
        assert cache.get("custom", "custom-v1") is module

    def test_evict_then_re_cache(self) -> None:
        """evicted entry can be re-cached with new instance."""
        cache = ModelCache()
        first = object()
        second = object()

        cache.put("anthropic", "claude-sonnet-4-20250514", first)
        assert cache.get("anthropic", "claude-sonnet-4-20250514") is first

        cache.evict("anthropic", "claude-sonnet-4-20250514")
        assert cache.get("anthropic", "claude-sonnet-4-20250514") is None

        cache.put("anthropic", "claude-sonnet-4-20250514", second)
        assert cache.get("anthropic", "claude-sonnet-4-20250514") is second
        assert cache.size() == 1
