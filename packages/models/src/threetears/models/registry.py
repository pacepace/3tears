"""provider registry mapping provider names to factory module paths.

The registry is consumed by :func:`threetears.models.factory.create_chat_model`
and :func:`threetears.models.factory.create_embedding_model` to resolve a
provider name to its factory module. Host applications can register
custom providers via :meth:`ProviderRegistry.register`.
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType

__all__ = [
    "BUILTIN_PROVIDERS",
    "ProviderRegistry",
]

BUILTIN_PROVIDERS: dict[str, str] = {
    "anthropic": "threetears.models.providers.anthropic",
    "openai": "threetears.models.providers.openai",
    "openrouter": "threetears.models.providers.openrouter",
    "voyageai": "threetears.models.providers.voyageai",
    "whisper": "threetears.models.providers.whisper",
}


class ProviderRegistry:
    """registry mapping provider names to adapter module import paths.

    supports lazy loading -- adapter modules are only imported when first
    requested via :meth:`get_provider`. host applications can register
    custom providers for project-specific adapters.
    """

    def __init__(self) -> None:
        self._providers: dict[str, str] = dict(BUILTIN_PROVIDERS)
        self._modules: dict[str, ModuleType] = {}
        self._lock = threading.Lock()

    def register(self, name: str, import_path: str) -> None:
        """register provider name to module import path mapping.

        overwrites existing registration if name already exists. clears
        any cached module so the next access re-imports.

        :param name: provider name (e.g. ``"custom-llm"``)
        :ptype name: str
        :param import_path: full dotted import path to provider module
        :ptype import_path: str
        """
        with self._lock:
            self._providers[name] = import_path
            self._modules.pop(name, None)

    def get_provider(self, name: str) -> ModuleType:
        """resolve provider name to imported module.

        lazily imports module on first access, caches for subsequent calls.

        :param name: registered provider name
        :ptype name: str
        :return: imported provider module
        :rtype: ModuleType
        :raises KeyError: if provider name not registered
        :raises ImportError: if provider module or SDK dependencies not installed
        """
        with self._lock:
            registered = self._providers.get(name)
            cached = self._modules.get(name)

        if registered is None:
            registered_keys = sorted(self._providers.keys())
            raise KeyError(f"Unknown provider: '{name}'. Registered: {registered_keys}")

        if cached is not None:
            return cached

        try:
            module = importlib.import_module(registered)
        except ImportError as exc:
            if name in BUILTIN_PROVIDERS:
                raise ImportError(
                    f"Failed to import provider '{name}' from "
                    f"'{registered}': {exc}. "
                    f"Install 3tears-models[{name}] to use the {name} provider"
                ) from exc
            raise ImportError(f"Failed to import provider '{name}' from '{registered}': {exc}") from exc

        with self._lock:
            self._modules[name] = module

        return module

    def list_providers(self) -> list[str]:
        """return sorted list of all registered provider names.

        :return: sorted provider name list
        :rtype: list[str]
        """
        with self._lock:
            result = sorted(self._providers.keys())
        return result

    def is_registered(self, name: str) -> bool:
        """check if provider name is registered.

        :param name: provider name to check
        :ptype name: str
        :return: True if registered
        :rtype: bool
        """
        with self._lock:
            result = name in self._providers
        return result
