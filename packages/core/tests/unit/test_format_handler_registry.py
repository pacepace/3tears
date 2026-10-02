"""tests for threetears.core.serialization format-handler registry.

covers multi-extension registration, case-insensitive lookup, and
UnknownFormatError raising for unregistered extensions.

the registry is process-wide and has no unregister, so every test registers
its fakes under extensions minted for that test alone. no test can then see
another's handler, or replace a real handler (``.yaml``, ``.json``) that the
rest of the session relies on, and the registry is observed only through
:func:`handler_for`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from threetears.core.serialization import (
    FormatHandler,
    UnknownFormatError,
    handler_for,
    register_handler,
)


# parity-exempt: format-handler stand-in; FormatHandler Protocol surface is parse+dump only and the fake exercises both, but we exempt rather than mark because the test is the format-handler-registry test itself
class _FakeHandler:
    """stand-in FormatHandler implementation for registry tests.

    implements the structural protocol shape but does no real parsing;
    tests only verify identity-preserving registration and lookup.
    """

    def __init__(self, *extensions: str) -> None:
        """declare the extensions this handler registers under.

        :param extensions: dotted extensions, as a real handler declares them
        :ptype extensions: str
        """
        self.extensions: tuple[str, ...] = extensions

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


@pytest.fixture
def ext() -> str:
    """a lowercase extension nothing else in the process registers, without its dot.

    :return: the extension
    :rtype: str
    """
    return f"t3fake{uuid4().hex[:12]}"


class TestRegisterHandler:
    """register_handler installs all declared extensions as lookup keys."""

    def test_registers_all_declared_extensions(self, ext: str) -> None:
        handler = _FakeHandler(f".{ext}a", f".{ext}b")
        register_handler(handler)
        assert handler_for(f"doc.{ext}a") is handler
        assert handler_for(f"doc.{ext}b") is handler

    def test_a_declared_extension_is_matched_lowercase_and_without_its_dot(self, ext: str) -> None:
        handler = _FakeHandler(f".{ext.upper()}")
        register_handler(handler)
        assert handler_for(f"doc.{ext}") is handler
        assert handler_for(f"doc.{ext.upper()}") is handler

    def test_registering_an_extension_again_replaces_the_prior_handler(self, ext: str) -> None:
        first = _FakeHandler(f".{ext}")
        second = _FakeHandler(f".{ext}")
        register_handler(first)
        register_handler(second)
        assert handler_for(f"doc.{ext}") is second

    def test_fake_handler_satisfies_protocol(self, ext: str) -> None:
        handler = _FakeHandler(f".{ext}")
        assert isinstance(handler, FormatHandler)


class TestHandlerFor:
    """handler_for resolves paths to registered handlers case-insensitively."""

    def test_returns_registered_handler(self, ext: str) -> None:
        handler = _FakeHandler(f".{ext}", f".{ext}x")
        register_handler(handler)
        assert handler_for(f"config.{ext}") is handler
        assert handler_for(f"config.{ext}x") is handler

    def test_is_case_insensitive_on_extension(self, ext: str) -> None:
        handler = _FakeHandler(f".{ext}")
        register_handler(handler)
        assert handler_for(f"CONFIG.{ext.upper()}") is handler
        assert handler_for(f"config.{ext.capitalize()}") is handler

    def test_accepts_path_object(self, ext: str) -> None:
        handler = _FakeHandler(f".{ext}")
        register_handler(handler)
        assert handler_for(Path(f"/tmp/x.{ext}")) is handler

    def test_disjoint_handlers_resolve_independently(self, ext: str) -> None:
        first = _FakeHandler(f".{ext}a")
        second = _FakeHandler(f".{ext}b")
        register_handler(first)
        register_handler(second)
        assert handler_for(f"a.{ext}a") is first
        assert handler_for(f"b.{ext}b") is second

    def test_raises_unknown_format_error(self, ext: str) -> None:
        with pytest.raises(UnknownFormatError) as exc_info:
            handler_for(f"mystery.{ext}")
        assert ext in str(exc_info.value)

    def test_unknown_format_error_is_lookup_error(self, ext: str) -> None:
        with pytest.raises(LookupError):
            handler_for(f"mystery.{ext}")
