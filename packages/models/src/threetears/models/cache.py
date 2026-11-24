"""thread-safe instance cache for model provider objects."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

__all__ = [
    "ModelCache",
]


class ModelCache:
    """thread-safe cache for model provider instances.

    stores instantiated model objects keyed by (provider_name, model_name)
    tuple to avoid re-instantiation on every request. supports selective
    eviction for API key rotation and provider shutdown scenarios.
    """

    def __init__(self) -> None:
        # a read-only snapshot replaced whole under the lock: one entry per (provider, model), a handful
        self._cache: Mapping[tuple[str, str], Any] = MappingProxyType({})
        self._lock: threading.Lock = threading.Lock()

    def get(self, provider_name: str, model_name: str) -> Any | None:
        """retrieve cached model instance.

        :param provider_name: provider identifier (e.g. "anthropic")
        :ptype provider_name: str
        :param model_name: model identifier (e.g. "claude-sonnet-4-6")
        :ptype model_name: str
        :return: cached instance or None if not found
        :rtype: Any | None
        """
        with self._lock:
            result = self._cache.get((provider_name, model_name))
        return result

    def put(self, provider_name: str, model_name: str, instance: Any) -> None:
        """store model instance in cache.

        :param provider_name: provider identifier
        :ptype provider_name: str
        :param model_name: model identifier
        :ptype model_name: str
        :param instance: model instance to cache
        :ptype instance: Any
        """
        with self._lock:
            self._cache = MappingProxyType({**self._cache, (provider_name, model_name): instance})

    def evict(self, provider_name: str, model_name: str) -> bool:
        """remove specific model instance from cache.

        :param provider_name: provider identifier
        :ptype provider_name: str
        :param model_name: model identifier
        :ptype model_name: str
        :return: True if entry was found and removed, False if not found
        :rtype: bool
        """
        with self._lock:
            key = (provider_name, model_name)
            if key in self._cache:
                self._cache = MappingProxyType({held: value for held, value in self._cache.items() if held != key})
                found = True
            else:
                found = False
        return found

    def evict_provider(self, provider_name: str) -> int:
        """remove all cached instances for given provider.

        :param provider_name: provider identifier
        :ptype provider_name: str
        :return: number of entries removed
        :rtype: int
        """
        with self._lock:
            kept = {key: value for key, value in self._cache.items() if key[0] != provider_name}
            count = len(self._cache) - len(kept)
            self._cache = MappingProxyType(kept)
        return count

    def clear(self) -> None:
        """remove all entries from cache."""
        with self._lock:
            self._cache = MappingProxyType({})

    def size(self) -> int:
        """return number of cached entries.

        :return: current cache size
        :rtype: int
        """
        with self._lock:
            result = len(self._cache)
        return result
