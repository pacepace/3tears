"""tests for the coercion-coverage ``find_run_overrides`` walker."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.coercion_coverage.walkers import (
    base_looks_toolish,
    find_run_overrides,
)


_DEFAULT_SUFFIXES: frozenset[str] = frozenset({"Tool"})


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo_with_pyproject(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


# ------------------------------------------------------------------
# base_looks_toolish helper
# ------------------------------------------------------------------

class TestBaseLooksToolish:
    def test_bare_name_with_tool_suffix(self) -> None:
        import ast
        node = ast.parse("class X(FooTool): pass").body[0]
        assert isinstance(node, ast.ClassDef)
        assert base_looks_toolish(node.bases[0], _DEFAULT_SUFFIXES) is True

    def test_attribute_with_tool_suffix(self) -> None:
        import ast
        node = ast.parse("class X(pkg.FooTool): pass").body[0]
        assert isinstance(node, ast.ClassDef)
        assert base_looks_toolish(node.bases[0], _DEFAULT_SUFFIXES) is True

    def test_bare_tool_name_matches(self) -> None:
        # full name "Tool" ends with "Tool" — matches the suffix rule.
        import ast
        node = ast.parse("class X(Tool): pass").body[0]
        assert isinstance(node, ast.ClassDef)
        assert base_looks_toolish(node.bases[0], _DEFAULT_SUFFIXES) is True

    def test_non_tool_base_rejected(self) -> None:
        import ast
        node = ast.parse("class X(BaseClass): pass").body[0]
        assert isinstance(node, ast.ClassDef)
        assert base_looks_toolish(node.bases[0], _DEFAULT_SUFFIXES) is False

    def test_subscript_base_rejected(self) -> None:
        # Generic[T]-style subscript bases cannot be name-matched.
        import ast
        node = ast.parse("class X(Generic[T]): pass").body[0]
        assert isinstance(node, ast.ClassDef)
        assert base_looks_toolish(node.bases[0], _DEFAULT_SUFFIXES) is False

    def test_call_base_rejected(self) -> None:
        # call-shaped bases (``make_base()``) are not name-matchable.
        import ast
        node = ast.parse("class X(make_base()): pass").body[0]
        assert isinstance(node, ast.ClassDef)
        assert base_looks_toolish(node.bases[0], _DEFAULT_SUFFIXES) is False

    def test_custom_suffix_set_honoured(self) -> None:
        import ast
        node = ast.parse("class X(MyAction): pass").body[0]
        assert isinstance(node, ast.ClassDef)
        suffixes = frozenset({"Action"})
        assert base_looks_toolish(node.bases[0], suffixes) is True
        # default suffix set rejects the same base.
        assert base_looks_toolish(node.bases[0], _DEFAULT_SUFFIXES) is False


# ------------------------------------------------------------------
# walker — positive cases
# ------------------------------------------------------------------

class TestRunOverrideFlagged:
    def test_tool_subclass_with_run_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "tool.py",
            "class FooTool(Tool):\n"
            "    def run(self, **kwargs):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "coercion_coverage.run_override"
        assert v.symbol == "FooTool"
        assert v.file == src / "pkg" / "tool.py"
        # the violation line points at the offending method, not the class
        assert v.line == 2
        assert "FooTool" in v.reason
        assert "execute()" in v.reason

    def test_async_run_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "tool.py",
            "class FooTool(Tool):\n"
            "    async def run(self, **kwargs):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert len(violations) == 1
        assert violations[0].symbol == "FooTool"

    def test_attribute_base_with_run_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "tool.py",
            "class FooTool(threetears.Tool):\n"
            "    def run(self):\n"
            "        return 1\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert len(violations) == 1
        assert violations[0].symbol == "FooTool"

    def test_bare_tool_base_with_run_flagged(self, tmp_path: Path) -> None:
        # the bare class name "Tool" is itself caught by the suffix
        # match (``"Tool".endswith("Tool")`` is true).
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "tool.py",
            "class FooTool(Tool):\n"
            "    def run(self):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert len(violations) == 1

    def test_multi_base_with_one_toolish_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "tool.py",
            "class FooTool(SomeMixin, Tool, OtherMixin):\n"
            "    def run(self):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert len(violations) == 1
        assert violations[0].symbol == "FooTool"

    def test_multiple_tool_subclasses_mixed(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "good.py",
            "class GoodTool(Tool):\n"
            "    def execute(self):\n"
            "        return None\n",
        )
        _write(
            src / "pkg" / "bad.py",
            "class BadTool(Tool):\n"
            "    def run(self):\n"
            "        return None\n",
        )
        _write(
            src / "pkg" / "another_bad.py",
            "class AnotherBadTool(Tool):\n"
            "    async def run(self):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        symbols = {v.symbol for v in violations}
        assert symbols == {"BadTool", "AnotherBadTool"}

    def test_init_py_scanned(self, tmp_path: Path) -> None:
        # canonical didn't skip __init__.py; preserve that.
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "__init__.py",
            "class FooTool(Tool):\n"
            "    def run(self):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert len(violations) == 1
        assert violations[0].symbol == "FooTool"


# ------------------------------------------------------------------
# walker — negative cases
# ------------------------------------------------------------------

class TestRunOverrideNotFlagged:
    def test_tool_subclass_with_execute_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "tool.py",
            "class FooTool(Tool):\n"
            "    def execute(self, **kwargs):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert violations == []

    def test_non_toolish_base_with_run_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "task.py",
            "class TaskRunner(BaseTask):\n"
            "    def run(self):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert violations == []

    def test_class_without_bases_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "thing.py",
            "class Thing:\n"
            "    def run(self):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert violations == []

    def test_nested_class_ignored(self, tmp_path: Path) -> None:
        # only top-level classes are scanned. a nested Tool subclass
        # with a run override must NOT be flagged because the canonical
        # walker only inspects ``tree.body`` directly.
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "outer.py",
            "class Outer:\n"
            "    class Inner(Tool):\n"
            "        def run(self):\n"
            "            return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert violations == []

    def test_run_method_on_unrelated_class_not_flagged(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class JobRunner:\n"
            "    def run(self):\n"
            "        return None\n"
            "\n"
            "class WorkerThread(threading.Thread):\n"
            "    def run(self):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert violations == []

    def test_method_other_than_run_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "tool.py",
            "class FooTool(Tool):\n"
            "    def runner(self):\n"
            "        return None\n"
            "    def runs(self):\n"
            "        return None\n",
        )
        violations = find_run_overrides((src,), repo, _DEFAULT_SUFFIXES)
        assert violations == []


# ------------------------------------------------------------------
# walker — path-dep / multi-root behaviour
# ------------------------------------------------------------------

class TestPathDepWalking:
    def test_two_repo_fixture_flags_subclass(self, tmp_path: Path) -> None:
        # synthetic "package A defines Tool, package B subclasses it"
        # layout. both src roots get passed to the walker (mimicking
        # discover_src_roots's output for a path-dep workspace) — the
        # subclass in B with a run override must be flagged. the AST
        # walker doesn't actually need to see the Tool base's
        # definition; it matches by base name suffix. but this test
        # confirms multi-root walking does not double-count or drop
        # violations.
        lib_src = tmp_path / "lib" / "src"
        consumer_src = tmp_path / "consumer" / "src"
        _write(
            lib_src / "lib" / "base.py",
            "class Tool:\n"
            "    def run(self):\n"
            "        return None\n",
        )
        _write(
            consumer_src / "consumer_pkg" / "concrete.py",
            "from lib.base import Tool\n"
            "\n"
            "class FooTool(Tool):\n"
            "    def run(self):\n"
            "        return None\n",
        )
        violations = find_run_overrides(
            (lib_src, consumer_src), tmp_path, _DEFAULT_SUFFIXES,
        )
        # Tool itself in lib_src has no Tool-ish base, so it is not
        # flagged; FooTool in consumer_src has Tool as a base and a
        # run override, so it is flagged.
        symbols = {v.symbol for v in violations}
        assert symbols == {"FooTool"}
