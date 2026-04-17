"""registry mapping provider names to adapter module import paths."""

from __future__ import annotations

import importlib
from types import ModuleType

_BUILTIN_PROVIDERS: dict[str, str] = {
    "anthropic": "threetears.models.providers.anthropic",
    "openai": "threetears.models.providers.openai",
    "openrouter": "threetears.models.providers.openrouter",
    "voyageai": "threetears.models.providers.voyageai",
    "whisper": "threetears.models.providers.whisper",
}


class ProviderRegistry:
    """registry mapping provider names to adapter module import paths.

    supports lazy loading -- adapter modules are only imported when first
    requested via get_provider(). host applications can register custom
    providers for project-specific adapters.
    """

    def __init__(self) -> None:
        self._providers: dict[str, str] = dict(_BUILTIN_PROVIDERS)
        self._modules: dict[str, ModuleType] = {}

    def register(self, name: str, import_path: str) -> None:
        """register provider name to module import path mapping.

        overwrites existing registration if name already exists.
        clears any cached module for name to force re-import.

        :param name: provider name (e.g. "custom-llm")
        :ptype name: str
        :param import_path: full dotted import path to provider module
        :ptype import_path: str
        """
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
        if name not in self._providers:
            raise KeyError(f"Unknown provider: '{name}'. Registered: {sorted(self._providers.keys())}")

        if name in self._modules:
            result = self._modules[name]
        else:
            import_path = self._providers[name]
            try:
                module = importlib.import_module(import_path)
            except ImportError as exc:
                if name in _BUILTIN_PROVIDERS:
                    raise ImportError(
                        f"Failed to import provider '{name}' from "
                        f"'{import_path}': {exc}. "
                        f"Install 3tears-models[{name}] to use "
                        f"the {name} provider"
                    ) from exc
                raise ImportError(f"Failed to import provider '{name}' from '{import_path}': {exc}") from exc
            self._modules[name] = module
            result = module
        return result

    def list_providers(self) -> list[str]:
        """return sorted list of all registered provider names.

        :return: sorted provider name list
        :rtype: list[str]
        """
        result = sorted(self._providers.keys())
        return result

    def is_registered(self, name: str) -> bool:
        """check if provider name is registered.

        :param name: provider name to check
        :ptype name: str
        :return: True if registered
        :rtype: bool
        """
        result = name in self._providers
        return result
