"""The fidelity mechanism, proven on a package this test writes and no host owns.

:mod:`threetears.evals.run.fidelity` is the engine half of a fidelity contract: it resolves a dotted
constructor path and reads each declared caller's source for a reference to it. The toy host
declares one contract, over a module-level function (``test_fidelity_adoption.py`` is its canary),
which reaches neither property below: a method constructor, and a module that exists but fails to
import. The evidence for those is a throwaway package under ``tmp_path`` holding a document parser.
It has no host type, no adapter import and no host vocabulary.

Two properties are pinned, each from both sides:

* the module boundary of a dotted path is FOUND by import rather than guessed by splitting, so
  a constructor that is a method on a class resolves, and a module that exists but fails to
  import is reported as broken rather than walked past as absent;
* a METHOD contract is reached only through its class name outside the defining module, while
  a module-level function is reached by its bare name.
"""

from __future__ import annotations

import sys
import textwrap
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from threetears.evals.run.fidelity import FidelityContract, callers_missing_the_constructor, resolve_constructor


#: The throwaway package's modules, by name within the package. Inside ``parsing`` nothing spells
#: ``Parser.from_text``: its own code reaches the method only as ``cls.from_text`` or through a
#: call, which is the case where the bare name has to be the bar.
_MODULES = {
    "__init__": "",
    "parsing": """
        def make_parser(text):
            return Parser().reparse(text)


        class Parser:
            @classmethod
            def from_text(cls, text):
                return cls()

            def reparse(self, text):
                return type(self).from_text(text)

            @classmethod
            def again(cls, text):
                return cls.from_text(text)
    """,
    "via_class": """
        from .parsing import Parser

        def load(text):
            return Parser.from_text(text)
    """,
    "via_bare_attribute": """
        def load(factory, text):
            return factory.from_text(text)
    """,
    "via_subscript": """
        def load(registry, text):
            return registry["parser"].from_text(text)
    """,
    "via_function": """
        from .parsing import make_parser

        def load(text):
            return make_parser(text)
    """,
    "unrelated": """
        def load(text):
            return text.upper()
    """,
    "broken": """
        import hostneutral_dependency_that_is_not_installed

        def make():
            return None
    """,
}


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Write the throwaway package, put it on the path, and forget it afterwards.

    Yields:
        The package's unique top-level name.
    """
    name = f"hostneutral_fidelity_{uuid.uuid4().hex[:12]}"
    root = tmp_path / name
    root.mkdir()
    for module, source in _MODULES.items():
        (root / f"{module}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    for loaded in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
        del sys.modules[loaded]


def _contract(constructor: str, *callers: str) -> FidelityContract:
    return FidelityContract(behavior="document.parse", constructor=constructor, callers=callers, why="one parser")


class TestTheModuleBoundaryIsFoundNotGuessed:
    """``resolve_constructor`` imports the longest prefix that imports and walks the rest."""

    def test_a_method_on_a_class_resolves(self, package: str) -> None:
        resolved = resolve_constructor(_contract(f"{package}.parsing.Parser.from_text"))

        parser_class = sys.modules[f"{package}.parsing"].Parser
        assert resolved == parser_class.from_text

    def test_a_module_level_function_still_resolves(self, package: str) -> None:
        resolved = resolve_constructor(_contract(f"{package}.parsing.make_parser"))

        assert resolved is sys.modules[f"{package}.parsing"].make_parser

    def test_a_module_that_exists_and_fails_to_import_is_reported_as_broken(self, package: str) -> None:
        """Walking past it would call the constructor absent from a module that is right there."""
        with pytest.raises(ModuleNotFoundError) as raised:
            resolve_constructor(_contract(f"{package}.broken.make"))

        assert raised.value.name == "hostneutral_dependency_that_is_not_installed"

    def test_a_path_with_no_importable_prefix_is_refused(self) -> None:
        missing = f"hostneutral_absent_{uuid.uuid4().hex[:12]}"
        with pytest.raises(ImportError, match="no importable module prefix"):
            resolve_constructor(_contract(f"{missing}.parsing.make_parser"))

    def test_an_attribute_the_module_lacks_is_an_attribute_error(self, package: str) -> None:
        with pytest.raises(AttributeError):
            resolve_constructor(_contract(f"{package}.parsing.Parser.no_such_method"))


class TestWhatCountsAsReachingTheConstructor:
    """``callers_missing_the_constructor`` asks a stricter question of a method than of a function."""

    def test_a_method_is_reached_through_its_class_name(self, package: str) -> None:
        contract = _contract(f"{package}.parsing.Parser.from_text", f"{package}.via_class")

        assert callers_missing_the_constructor(contract) == []

    def test_a_bare_attribute_of_the_same_name_does_not_reach_a_method(self, package: str) -> None:
        """``factory.from_text`` could be any object's attribute, so it is not the class's method."""
        contract = _contract(f"{package}.parsing.Parser.from_text", f"{package}.via_bare_attribute")

        assert callers_missing_the_constructor(contract) == [f"{package}.via_bare_attribute"]

    def test_a_receiver_the_walk_cannot_name_does_not_reach_a_method(self, package: str) -> None:
        contract = _contract(f"{package}.parsing.Parser.from_text", f"{package}.via_subscript")

        assert callers_missing_the_constructor(contract) == [f"{package}.via_subscript"]

    def test_inside_the_defining_module_the_bare_name_reaches_the_method(self, package: str) -> None:
        """The class's own methods reach it as ``cls.from_text``, which names the class as surely."""
        contract = _contract(f"{package}.parsing.Parser.from_text", f"{package}.parsing")

        assert callers_missing_the_constructor(contract) == []

    def test_a_module_level_function_is_reached_by_its_bare_name(self, package: str) -> None:
        contract = _contract(f"{package}.parsing.make_parser", f"{package}.via_function", f"{package}.unrelated")

        assert callers_missing_the_constructor(contract) == [f"{package}.unrelated"]

    def test_a_constructor_that_is_gone_is_a_broken_contract_not_missing_callers(self) -> None:
        missing = f"hostneutral_absent_{uuid.uuid4().hex[:12]}"
        with pytest.raises(ImportError):
            callers_missing_the_constructor(_contract(f"{missing}.parsing.make_parser", "json"))
