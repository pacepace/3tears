"""enforcement: one path-template type, one cache vocabulary.

two drift classes this package has already paid for once, encoded as static
checks so the second occurrence fails at CI rather than at review:

1. **a parallel path-template descriptor.** the outbound
   ``HttpOperationDescriptor`` and the inbound ``RestAffordance`` are the same
   idea in two directions. they share
   :class:`~threetears.agent.tools.http_operation.PathTemplateBinding` so a
   template is read identically on both sides. a THIRD type that declares its
   own ``path_template`` field without inheriting that base would re-derive
   placeholders its own way, and the two derivations would disagree the first
   time either changed.

2. **a second cache-exposure vocabulary.** ``CacheClass`` lives once, in
   ``threetears.core.http_cache``, because a per-package copy would let one
   package widen what another narrowed. a local enum spelling PUBLIC /
   AUTHENTICATED / PRIVATE is that copy.

pure AST, no imports of the code under test, so a syntax-level regression
cannot hide behind a working import.

FALLIBILITY PROOF: each check below is exercised against a synthetic violating
source in ``TestTheseChecksCanFail``. a guard nobody has ever seen fail is a
guard nobody knows is wired up.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "tools"

#: the sanctioned base every path-template-carrying type must inherit.
_SHARED_BASE = "PathTemplateBinding"

#: member names that identify a cache-exposure vocabulary. any enum in this
#: package carrying all of them is a copy of the promoted one.
_CACHE_VOCABULARY = frozenset({"PUBLIC", "AUTHENTICATED", "PRIVATE"})


def _collect_src_files() -> list[Path]:
    """collect production Python source files to scan.

    :return: sorted list of source file paths under agent/tools src
    :rtype: list[Path]
    """
    return sorted(_SRC_ROOT.rglob("*.py"))


def _base_names(node: ast.ClassDef) -> set[str]:
    """collect the simple names of a class's bases.

    matches bare ``Name`` and ``module.Attr`` forms; structural only, no
    import resolution.

    :param node: class definition node
    :ptype node: ast.ClassDef
    :return: set of base names
    :rtype: set[str]
    """
    names: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _declares_field(node: ast.ClassDef, field_name: str) -> bool:
    """return True when a class body annotates ``field_name`` at class level.

    :param node: class definition node
    :ptype node: ast.ClassDef
    :param field_name: attribute name to look for
    :ptype field_name: str
    :return: True when the class annotates that attribute
    :rtype: bool
    """
    found = False
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.target.id == field_name:
                found = True
    return found


def find_unshared_path_template_types(tree: ast.Module) -> list[str]:
    """find classes declaring ``path_template`` without the shared base.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: names of offending classes
    :rtype: list[str]
    """
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name == _SHARED_BASE:
            continue
        if _declares_field(node, "path_template") and _SHARED_BASE not in _base_names(node):
            offenders.append(node.name)
    return offenders


def find_duplicate_cache_vocabularies(tree: ast.Module) -> list[str]:
    """find enums re-spelling the promoted cache-exposure vocabulary.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: names of offending enum classes
    :rtype: list[str]
    """
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not {"StrEnum", "Enum", "IntEnum"} & _base_names(node):
            continue
        members = {
            statement.targets[0].id
            for statement in node.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        }
        if _CACHE_VOCABULARY <= members:
            offenders.append(node.name)
    return offenders


class TestOnePathTemplateType:
    """every path-template carrier inherits the shared binding."""

    def test_no_parallel_descriptor_type(self) -> None:
        """a new ``path_template`` field means inheriting the shared base."""
        violations: list[str] = []
        for path in _collect_src_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for name in find_unshared_path_template_types(tree):
                violations.append(f"{path.relative_to(_SRC_ROOT)}: {name}")
        assert not violations, (
            f"class(es) declare a path_template without inheriting {_SHARED_BASE} "
            f"({len(violations)}):\n" + "\n".join(violations)
        )


class TestOneCacheVocabulary:
    """the exposure vocabulary is spelled once, upstream in core."""

    def test_no_local_cache_class_enum(self) -> None:
        """a local PUBLIC/AUTHENTICATED/PRIVATE enum is a second vocabulary."""
        violations: list[str] = []
        for path in _collect_src_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for name in find_duplicate_cache_vocabularies(tree):
                violations.append(f"{path.relative_to(_SRC_ROOT)}: {name}")
        assert not violations, (
            "cache-exposure vocabulary re-spelled locally; import "
            "threetears.core.http_cache.CacheClass instead "
            f"({len(violations)}):\n" + "\n".join(violations)
        )


class TestTheseChecksCanFail:
    """in-file proof that both walkers report a real violation."""

    def test_parallel_descriptor_is_detected(self) -> None:
        """a hand-rolled path-template dataclass is reported by name."""
        source = (
            "from dataclasses import dataclass\n"
            "@dataclass(frozen=True)\n"
            "class RivalDescriptor:\n"
            "    method: str\n"
            "    path_template: str\n"
        )
        assert find_unshared_path_template_types(ast.parse(source)) == ["RivalDescriptor"]

    def test_inheriting_the_shared_base_is_not_detected(self) -> None:
        """the check passes what it is supposed to pass."""
        source = (
            "from dataclasses import dataclass\n"
            "@dataclass(frozen=True)\n"
            "class GoodDescriptor(PathTemplateBinding):\n"
            "    path_template: str\n"
        )
        assert find_unshared_path_template_types(ast.parse(source)) == []

    def test_duplicate_cache_vocabulary_is_detected(self) -> None:
        """a local copy of the exposure enum is reported by name."""
        source = (
            "from enum import StrEnum\n"
            "class RivalCacheClass(StrEnum):\n"
            "    PUBLIC = 'public'\n"
            "    AUTHENTICATED = 'authenticated'\n"
            "    PRIVATE = 'private'\n"
        )
        assert find_duplicate_cache_vocabularies(ast.parse(source)) == ["RivalCacheClass"]

    def test_an_unrelated_enum_is_not_detected(self) -> None:
        """the check does not fire on any enum that happens to be nearby."""
        source = "from enum import StrEnum\nclass Colour(StrEnum):\n    RED = 'red'\n    PUBLIC = 'public'\n"
        assert find_duplicate_cache_vocabularies(ast.parse(source)) == []
