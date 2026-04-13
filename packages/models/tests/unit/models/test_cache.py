"""tests for ModelCache thread-safe instance cache."""

from __future__ import annotations

import threading

from threetears.models.cache import ModelCache


class TestModelCache:
    """tests for ModelCache thread-safe instance cache."""

    # -- construction --

    def test_new_cache_is_empty(self) -> None:
        """newly created cache has size zero."""
        cache = ModelCache()
        assert cache.size() == 0

    # -- get / put --

    def test_put_and_get(self) -> None:
        """stored instance is retrievable by same key."""
        cache = ModelCache()
        sentinel = object()
        cache.put("anthropic", "claude-sonnet-4-20250514", sentinel)
        assert cache.get("anthropic", "claude-sonnet-4-20250514") is sentinel

    def test_get_missing_returns_none(self) -> None:
        """get for non-existent key returns None."""
        cache = ModelCache()
        assert cache.get("nonexistent", "nonexistent") is None

    def test_put_overwrites_existing(self) -> None:
        """second put with same key replaces first instance."""
        cache = ModelCache()
        first = object()
        second = object()
        cache.put("anthropic", "claude-sonnet-4-20250514", first)
        cache.put("anthropic", "claude-sonnet-4-20250514", second)
        assert cache.get("anthropic", "claude-sonnet-4-20250514") is second
        assert cache.size() == 1

    def test_multiple_providers(self) -> None:
        """different providers do not collide."""
        cache = ModelCache()
        anthropic_instance = object()
        openai_instance = object()
        cache.put("anthropic", "claude-sonnet-4-20250514", anthropic_instance)
        cache.put("openai", "gpt-4", openai_instance)
        assert cache.get("anthropic", "claude-sonnet-4-20250514") is anthropic_instance
        assert cache.get("openai", "gpt-4") is openai_instance
        assert cache.size() == 2

    def test_same_provider_different_models(self) -> None:
        """different models for same provider are stored separately."""
        cache = ModelCache()
        sonnet = object()
        haiku = object()
        cache.put("anthropic", "claude-sonnet-4-20250514", sonnet)
        cache.put("anthropic", "claude-haiku", haiku)
        assert cache.get("anthropic", "claude-sonnet-4-20250514") is sonnet
        assert cache.get("anthropic", "claude-haiku") is haiku
        assert cache.size() == 2

    # -- eviction --

    def test_evict_existing_returns_true(self) -> None:
        """evict returns True and removes entry when key exists."""
        cache = ModelCache()
        cache.put("anthropic", "claude-sonnet-4-20250514", object())
        assert cache.evict("anthropic", "claude-sonnet-4-20250514") is True
        assert cache.get("anthropic", "claude-sonnet-4-20250514") is None
        assert cache.size() == 0

    def test_evict_missing_returns_false(self) -> None:
        """evict returns False when key does not exist."""
        cache = ModelCache()
        assert cache.evict("nonexistent", "nonexistent") is False

    def test_evict_provider_removes_all(self) -> None:
        """evict_provider removes all entries for given provider and returns count."""
        cache = ModelCache()
        cache.put("anthropic", "claude-sonnet-4-20250514", object())
        cache.put("anthropic", "claude-haiku", object())
        cache.put("openai", "gpt-4", object())
        count = cache.evict_provider("anthropic")
        assert count == 2
        assert cache.get("anthropic", "claude-sonnet-4-20250514") is None
        assert cache.get("anthropic", "claude-haiku") is None
        assert cache.size() == 1

    def test_evict_provider_returns_zero_for_unknown(self) -> None:
        """evict_provider returns 0 for non-existent provider."""
        cache = ModelCache()
        assert cache.evict_provider("nonexistent") == 0

    def test_evict_provider_leaves_other_providers(self) -> None:
        """evict_provider does not affect entries from other providers."""
        cache = ModelCache()
        openai_instance = object()
        cache.put("anthropic", "claude-sonnet-4-20250514", object())
        cache.put("openai", "gpt-4", openai_instance)
        cache.evict_provider("anthropic")
        assert cache.get("openai", "gpt-4") is openai_instance
        assert cache.size() == 1

    def test_clear_removes_all(self) -> None:
        """clear empties entire cache."""
        cache = ModelCache()
        cache.put("anthropic", "claude-sonnet-4-20250514", object())
        cache.put("openai", "gpt-4", object())
        cache.clear()
        assert cache.size() == 0

    def test_clear_on_empty_cache(self) -> None:
        """clear on empty cache completes without error."""
        cache = ModelCache()
        cache.clear()
        assert cache.size() == 0

    # -- thread safety --

    def test_concurrent_puts(self) -> None:
        """multiple threads putting different keys simultaneously do not corrupt cache."""
        cache = ModelCache()
        errors: list[Exception] = []

        def put_entry(index: int) -> None:
            try:
                cache.put(f"provider-{index}", f"model-{index}", index)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=put_entry, args=(i,))
            for i in range(100)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert cache.size() == 100
        for i in range(100):
            assert cache.get(f"provider-{i}", f"model-{i}") == i

    def test_concurrent_get_and_evict(self) -> None:
        """reader and evicter operating concurrently do not corrupt cache."""
        cache = ModelCache()
        for i in range(100):
            cache.put("provider", f"model-{i}", i)

        results: list[int | None] = []
        evict_results: list[bool] = []
        errors: list[Exception] = []

        def reader(index: int) -> None:
            try:
                val = cache.get("provider", f"model-{index}")
                results.append(val)
            except Exception as exc:
                errors.append(exc)

        def evicter(index: int) -> None:
            try:
                found = cache.evict("provider", f"model-{index}")
                evict_results.append(found)
            except Exception as exc:
                errors.append(exc)

        threads: list[threading.Thread] = []
        for i in range(100):
            threads.append(threading.Thread(target=reader, args=(i,)))
            threads.append(threading.Thread(target=evicter, args=(i,)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert cache.size() == 0
