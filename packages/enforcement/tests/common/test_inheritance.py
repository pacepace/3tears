"""tests for ``inheritance`` module."""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.common.inheritance import (
    collect_class_base_graph,
    extract_base_names,
    transitively_subclasses_any,
)


def _classdef(source: str) -> ast.ClassDef:
    """parse ``source`` (a single ``class ...:`` block) to a ClassDef."""
    tree = ast.parse(source)
    cls = tree.body[0]
    assert isinstance(cls, ast.ClassDef)
    return cls


class TestExtractBaseNames:
    def test_bare_name(self) -> None:
        cls = _classdef("class Foo(Base): pass")
        assert extract_base_names(cls) == ["Base"]

    def test_attribute(self) -> None:
        cls = _classdef("class Foo(pkg.Base): pass")
        assert extract_base_names(cls) == ["Base"]

    def test_subscript_name(self) -> None:
        cls = _classdef("class Foo(Base[int]): pass")
        assert extract_base_names(cls) == ["Base"]

    def test_subscript_attribute(self) -> None:
        cls = _classdef("class Foo(pkg.Base[int]): pass")
        assert extract_base_names(cls) == ["Base"]

    def test_nested_subscript(self) -> None:
        cls = _classdef("class Foo(pkg.outer.Base[T]): pass")
        assert extract_base_names(cls) == ["Base"]

    def test_multiple_bases(self) -> None:
        cls = _classdef("class Foo(A, B, pkg.C): pass")
        assert extract_base_names(cls) == ["A", "B", "C"]

    def test_no_bases(self) -> None:
        cls = _classdef("class Foo: pass")
        assert extract_base_names(cls) == []

    def test_call_base_skipped(self) -> None:
        # ``class Foo(make_base()):`` cannot be reduced textually
        cls = _classdef("class Foo(make_base()): pass")
        assert extract_base_names(cls) == []


def _write_module(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


class TestCollectClassBaseGraph:
    def test_single_module(self, tmp_path: Path) -> None:
        _write_module(
            tmp_path / "a.py",
            "class A: pass\nclass B(A): pass\n",
        )
        graph = collect_class_base_graph((tmp_path,))
        assert graph["A"] == set()
        assert graph["B"] == {"A"}

    def test_multiple_modules(self, tmp_path: Path) -> None:
        _write_module(tmp_path / "a.py", "class A: pass\n")
        _write_module(tmp_path / "b.py", "class B(A): pass\n")
        graph = collect_class_base_graph((tmp_path,))
        assert "A" in graph
        assert graph["B"] == {"A"}

    def test_namespace_collision_merges_bases(self, tmp_path: Path) -> None:
        _write_module(tmp_path / "x.py", "class Base(Foo): pass\n")
        _write_module(tmp_path / "y.py", "class Base(Bar): pass\n")
        graph = collect_class_base_graph((tmp_path,))
        # documented limitation: collisions union their bases
        assert graph["Base"] == {"Foo", "Bar"}

    def test_skips_unparseable_files(self, tmp_path: Path) -> None:
        _write_module(tmp_path / "good.py", "class A: pass\n")
        _write_module(tmp_path / "bad.py", "class A(:")
        graph = collect_class_base_graph((tmp_path,))
        # bad.py is silently skipped
        assert "A" in graph

    def test_skips_skipped_dirs(self, tmp_path: Path) -> None:
        _write_module(tmp_path / ".venv" / "skip.py", "class Hidden: pass\n")
        _write_module(tmp_path / "good.py", "class Visible: pass\n")
        graph = collect_class_base_graph((tmp_path,))
        assert "Visible" in graph
        assert "Hidden" not in graph


class TestTransitivelySubclassesAny:
    def test_direct_match(self) -> None:
        graph: dict[str, set[str]] = {"Foo": {"Base"}}
        assert transitively_subclasses_any("Foo", frozenset({"Base"}), graph) is True

    def test_self_match(self) -> None:
        # if class itself is a target, we say yes
        assert (
            transitively_subclasses_any(
                "Base",
                frozenset({"Base"}),
                {"Base": set()},
            )
            is True
        )

    def test_depth_two(self) -> None:
        graph = {"Foo": {"Mid"}, "Mid": {"Base"}}
        assert transitively_subclasses_any("Foo", frozenset({"Base"}), graph) is True

    def test_depth_three(self) -> None:
        graph = {"Foo": {"L1"}, "L1": {"L2"}, "L2": {"Base"}}
        assert transitively_subclasses_any("Foo", frozenset({"Base"}), graph) is True

    def test_no_match(self) -> None:
        graph = {"Foo": {"Other"}, "Other": {"Else"}}
        assert transitively_subclasses_any("Foo", frozenset({"Base"}), graph) is False

    def test_unknown_class_no_match(self) -> None:
        graph: dict[str, set[str]] = {}
        assert transitively_subclasses_any("Ghost", frozenset({"Base"}), graph) is False

    def test_cycle_returns_false_safely(self) -> None:
        graph = {"A": {"B"}, "B": {"A"}}
        # neither reaches "Base"; the cycle must terminate
        assert transitively_subclasses_any("A", frozenset({"Base"}), graph) is False
        assert transitively_subclasses_any("B", frozenset({"Base"}), graph) is False

    def test_cycle_with_target(self) -> None:
        graph = {"A": {"B"}, "B": {"A", "Base"}}
        assert transitively_subclasses_any("A", frozenset({"Base"}), graph) is True

    def test_multi_target_set(self) -> None:
        graph = {"Foo": {"Mid"}, "Mid": {"Schemed"}}
        assert (
            transitively_subclasses_any(
                "Foo",
                frozenset({"Base", "Schemed"}),
                graph,
            )
            is True
        )


class TestSchemaBackedExample:
    """smoke-test for the cache-primitive use case: the originally-broken chain."""

    def test_collection_chain_resolves_transitively(self, tmp_path: Path) -> None:
        _write_module(
            tmp_path / "base.py",
            "class BaseCollection: pass\n",
        )
        _write_module(
            tmp_path / "schema.py",
            "class SchemaBackedCollection(BaseCollection): pass\n",
        )
        _write_module(
            tmp_path / "memories.py",
            "class MemoriesCollection(SchemaBackedCollection):\n    pass\n",
        )
        graph = collect_class_base_graph((tmp_path,))
        assert (
            transitively_subclasses_any(
                "MemoriesCollection",
                frozenset({"BaseCollection"}),
                graph,
            )
            is True
        )
