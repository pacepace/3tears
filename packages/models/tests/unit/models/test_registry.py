"""tests for ProviderRegistry lazy-loading provider module registry."""

from __future__ import annotations

import json
from types import ModuleType

import pytest

from threetears.models.registry import ProviderRegistry, BUILTIN_PROVIDERS


class TestProviderRegistry:
    """tests for ProviderRegistry lazy-loading provider module registry."""

    # -- construction --

    def test_new_registry_has_builtin_providers(self) -> None:
        """new registry includes all builtin providers."""
        registry = ProviderRegistry()
        providers = registry.list_providers()
        for name in BUILTIN_PROVIDERS:
            assert name in providers

    def test_builtin_provider_count(self) -> None:
        """new registry has exactly 5 builtin providers."""
        registry = ProviderRegistry()
        assert len(registry.list_providers()) == 5

    # -- registration --

    def test_register_custom_provider(self) -> None:
        """custom provider is registered and discoverable."""
        registry = ProviderRegistry()
        registry.register("custom-llm", "some.module.path")
        assert registry.is_registered("custom-llm") is True

    def test_register_overwrites_existing(self) -> None:
        """re-registration replaces import path."""
        registry = ProviderRegistry()
        registry.register("custom", "path.one")
        registry.register("custom", "path.two")
        assert registry.is_registered("custom") is True

    def test_register_clears_cached_module(self) -> None:
        """re-registration invalidates previously cached module."""
        registry = ProviderRegistry()
        registry.register("test-json", "json")
        first = registry.get_provider("test-json")
        registry.register("test-json", "json")
        second = registry.get_provider("test-json")
        assert isinstance(first, ModuleType)
        assert isinstance(second, ModuleType)

    # -- lookup --

    def test_is_registered_true_for_builtin(self) -> None:
        """is_registered returns True for builtin provider."""
        registry = ProviderRegistry()
        assert registry.is_registered("anthropic") is True

    def test_is_registered_false_for_unknown(self) -> None:
        """is_registered returns False for unknown provider."""
        registry = ProviderRegistry()
        assert registry.is_registered("nonexistent") is False

    def test_list_providers_sorted(self) -> None:
        """list_providers returns alphabetically sorted list."""
        registry = ProviderRegistry()
        providers = registry.list_providers()
        assert providers == sorted(providers)

    # -- get_provider --

    def test_get_provider_unknown_raises_keyerror(self) -> None:
        """unknown provider name raises KeyError with helpful message."""
        registry = ProviderRegistry()
        with pytest.raises(KeyError, match="Unknown provider: 'nonexistent'"):
            registry.get_provider("nonexistent")

    def test_get_provider_caches_module(self) -> None:
        """second call returns same module object."""
        registry = ProviderRegistry()
        registry.register("test-json", "json")
        first = registry.get_provider("test-json")
        second = registry.get_provider("test-json")
        assert first is second

    def test_get_provider_import_error_has_install_hint(self) -> None:
        """builtin provider ImportError message includes install hint."""
        # inject a fake builtin that cannot be imported to test hint message
        from threetears.models import registry as registry_mod

        original = registry_mod.BUILTIN_PROVIDERS.copy()
        registry_mod.BUILTIN_PROVIDERS["fake-builtin"] = "totally.nonexistent.module"
        try:
            fresh = ProviderRegistry()
            with pytest.raises(ImportError, match=r"3tears-models\[fake-builtin\]"):
                fresh.get_provider("fake-builtin")
        finally:
            registry_mod.BUILTIN_PROVIDERS.clear()
            registry_mod.BUILTIN_PROVIDERS.update(original)

    def test_get_provider_real_module(self) -> None:
        """registered real stdlib module loads successfully."""
        registry = ProviderRegistry()
        registry.register("test-json", "json")
        module = registry.get_provider("test-json")
        assert module is json

    def test_get_provider_custom_import_error(self) -> None:
        """custom provider ImportError propagates with path info."""
        registry = ProviderRegistry()
        registry.register("bad-provider", "totally.fake.module.path")
        with pytest.raises(ImportError, match="totally.fake.module.path"):
            registry.get_provider("bad-provider")
