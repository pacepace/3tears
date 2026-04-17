"""tests for threetears.core.serialization format-handler registry.

covers multi-extension registration, case-insensitive lookup, and
UnknownFormatError raising for unregistered extensions.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from threetears.core.serialization import (
    FormatHandler,
    UnknownFormatError,
    _HANDLERS,
    handler_for,
    register_handler,
)


class _FakeYamlHandler:
    """stand-in FormatHandler implementation for registry tests.

    implements the structural protocol shape but does no real parsing;
    tests only verify identity-preserving registration and lookup.
    """

    extensions: tuple[str, ...] = (".yaml", ".yml")

    def load(self, text: str) -> Any:
        """return text unchanged as stand-in parse result.

        :param text: serialized document body
        :ptype text: str
        :return: input text as-is
        :rtype: Any
        """
        return text

    def dump(self, tree: Any) -> str:
        """return tree coerced to str as stand-in serialization.

        :param tree: in-memory document tree
        :ptype tree: Any
        :return: serialized document body
        :rtype: str
        """
        return str(tree)

    def get(self, tree: Any, path: str) -> Any:
        """return tree unchanged as stand-in path lookup.

        :param tree: in-memory document tree
        :ptype tree: Any
        :param path: handler-interpreted path expression
        :ptype path: str
        :return: value at path (stubbed as tree itself)
        :rtype: Any
        """
        return tree

    def set(self, tree: Any, path: str, value: Any) -> Any:
        """return tree unchanged as stand-in set operation.

        :param tree: in-memory document tree
        :ptype tree: Any
        :param path: handler-interpreted path expression
        :ptype path: str
        :param value: value to assign at path
        :ptype value: Any
        :return: possibly new tree
        :rtype: Any
        """
        return tree

    def merge(self, tree: Any, partial: dict[str, Any]) -> Any:
        """return tree unchanged as stand-in merge operation.

        :param tree: in-memory document tree
        :ptype tree: Any
        :param partial: partial document to merge into tree
        :ptype partial: dict[str, Any]
        :return: possibly new tree
        :rtype: Any
        """
        return tree


class _FakeTomlHandler:
    """second stand-in handler used to verify disjoint registry entries."""

    extensions: tuple[str, ...] = (".toml",)

    def load(self, text: str) -> Any:
        """return text unchanged.

        :param text: serialized document body
        :ptype text: str
        :return: input text as-is
        :rtype: Any
        """
        return text

    def dump(self, tree: Any) -> str:
        """return tree coerced to str.

        :param tree: in-memory document tree
        :ptype tree: Any
        :return: serialized document body
        :rtype: str
        """
        return str(tree)

    def get(self, tree: Any, path: str) -> Any:
        """return tree unchanged.

        :param tree: in-memory document tree
        :ptype tree: Any
        :param path: handler-interpreted path expression
        :ptype path: str
        :return: value at path (stubbed as tree itself)
        :rtype: Any
        """
        return tree

    def set(self, tree: Any, path: str, value: Any) -> Any:
        """return tree unchanged.

        :param tree: in-memory document tree
        :ptype tree: Any
        :param path: handler-interpreted path expression
        :ptype path: str
        :param value: value to assign at path
        :ptype value: Any
        :return: possibly new tree
        :rtype: Any
        """
        return tree

    def merge(self, tree: Any, partial: dict[str, Any]) -> Any:
        """return tree unchanged.

        :param tree: in-memory document tree
        :ptype tree: Any
        :param partial: partial document to merge into tree
        :ptype partial: dict[str, Any]
        :return: possibly new tree
        :rtype: Any
        """
        return tree


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """snapshot and restore the module-level handler registry per test.

    :return: iterator yielding once while the registry is cleared, then
        restoring the prior snapshot to avoid cross-test contamination
    :rtype: Iterator[None]
    """
    snapshot = dict(_HANDLERS)
    _HANDLERS.clear()
    try:
        yield
    finally:
        _HANDLERS.clear()
        _HANDLERS.update(snapshot)


class TestRegisterHandler:
    """register_handler installs all declared extensions as lookup keys."""

    def test_registers_all_declared_extensions(self, clean_registry: None) -> None:
        handler = _FakeYamlHandler()
        register_handler(handler)
        assert _HANDLERS["yaml"] is handler
        assert _HANDLERS["yml"] is handler

    def test_keys_are_stored_lowercase_and_without_dot(
        self,
        clean_registry: None,
    ) -> None:
        class _UpperHandler:
            extensions: tuple[str, ...] = (".JSON",)

            def load(self, text: str) -> Any:
                return text

            def dump(self, tree: Any) -> str:
                return str(tree)

            def get(self, tree: Any, path: str) -> Any:
                return tree

            def set(self, tree: Any, path: str, value: Any) -> Any:
                return tree

            def merge(self, tree: Any, partial: dict[str, Any]) -> Any:
                return tree

        register_handler(_UpperHandler())
        assert "json" in _HANDLERS
        assert ".json" not in _HANDLERS
        assert ".JSON" not in _HANDLERS

    def test_fake_handler_satisfies_protocol(self, clean_registry: None) -> None:
        handler = _FakeYamlHandler()
        assert isinstance(handler, FormatHandler)


class TestHandlerFor:
    """handler_for resolves paths to registered handlers case-insensitively."""

    def test_returns_registered_handler(self, clean_registry: None) -> None:
        handler = _FakeYamlHandler()
        register_handler(handler)
        assert handler_for("config.yaml") is handler
        assert handler_for("config.yml") is handler

    def test_is_case_insensitive_on_extension(self, clean_registry: None) -> None:
        handler = _FakeYamlHandler()
        register_handler(handler)
        assert handler_for("CONFIG.YAML") is handler
        assert handler_for("config.YmL") is handler

    def test_accepts_path_object(self, clean_registry: None) -> None:
        from pathlib import Path

        handler = _FakeYamlHandler()
        register_handler(handler)
        assert handler_for(Path("/tmp/x.yaml")) is handler

    def test_disjoint_handlers_resolve_independently(
        self,
        clean_registry: None,
    ) -> None:
        yaml_handler = _FakeYamlHandler()
        toml_handler = _FakeTomlHandler()
        register_handler(yaml_handler)
        register_handler(toml_handler)
        assert handler_for("a.yaml") is yaml_handler
        assert handler_for("b.toml") is toml_handler

    def test_raises_unknown_format_error(self, clean_registry: None) -> None:
        with pytest.raises(UnknownFormatError) as exc_info:
            handler_for("mystery.xyz")
        assert "xyz" in str(exc_info.value)

    def test_unknown_format_error_is_lookup_error(
        self,
        clean_registry: None,
    ) -> None:
        with pytest.raises(LookupError):
            handler_for("mystery.xyz")
