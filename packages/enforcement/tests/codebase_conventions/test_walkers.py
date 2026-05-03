"""tests for the codebase-conventions walkers."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.codebase_conventions.walkers import (
    find_missing_future_annotations,
    find_missing_return_types,
    find_print_calls,
    find_stdlib_getlogger_calls,
)


_GETLOGGER_MARKER = "# stdlib-getlogger: ok"
_DEFAULT_SKIP = frozenset({"__init__.py"})


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


# ----------------------------------------------------------------------
# find_print_calls
# ----------------------------------------------------------------------


class TestFindPrintCalls:
    def test_bare_print_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\ndef foo() -> None:\n    print('hi')\n",
        )
        violations = find_print_calls((src,), repo, {})
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "codebase_conventions.print"
        assert v.symbol == "print"
        assert v.line == 3

    def test_print_no_args_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\nprint()\n",
        )
        violations = find_print_calls((src,), repo, {})
        assert len(violations) == 1

    def test_print_reference_not_flagged(self, tmp_path: Path) -> None:
        # ``print`` (no call) — assignment, attribute reference, etc.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\nalias = print\n",
        )
        assert find_print_calls((src,), repo, {}) == []

    def test_obj_print_attribute_call_not_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        # ``obj.print(...)`` — attribute call, not the bare builtin.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\ndef foo(obj) -> None:\n    obj.print('hi')\n",
        )
        assert find_print_calls((src,), repo, {}) == []

    def test_logger_print_method_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\ndef foo(logger) -> None:\n    logger.print('hi')\n",
        )
        assert find_print_calls((src,), repo, {}) == []

    def test_exempt_file_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "cli.py",
            "from __future__ import annotations\ndef main() -> None:\n    print('hi')\n",
        )
        exempt = {
            "src/pkg/cli.py": ("command-line entry point intentionally writes to stdout"),
        }
        assert find_print_calls((src,), repo, exempt) == []

    def test_multiple_prints_one_per_call(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\ndef foo() -> None:\n    print('a')\n    print('b')\n    print('c')\n",
        )
        violations = find_print_calls((src,), repo, {})
        assert len(violations) == 3
        lines = sorted(v.line for v in violations)
        assert lines == [3, 4, 5]

    def test_no_prints_no_violations(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\nx = 1\n",
        )
        assert find_print_calls((src,), repo, {}) == []


# ----------------------------------------------------------------------
# find_stdlib_getlogger_calls
# ----------------------------------------------------------------------


class TestFindStdlibGetloggerCalls:
    def test_logging_getlogger_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\nimport logging\nlog = logging.getLogger(__name__)\n",
        )
        violations = find_stdlib_getlogger_calls(
            (src,),
            repo,
            {},
            _GETLOGGER_MARKER,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "codebase_conventions.stdlib_getlogger"
        assert v.symbol == "getLogger"
        assert v.line == 3

    def test_marker_on_same_line_exempts(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "import logging\n"
            "logging.getLogger('watchfiles').setLevel(40)  "
            "# stdlib-getlogger: ok — quiet third-party\n",
        )
        violations = find_stdlib_getlogger_calls(
            (src,),
            repo,
            {},
            _GETLOGGER_MARKER,
        )
        assert violations == []

    def test_marker_on_above_line_does_not_exempt(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "import logging\n"
            "# stdlib-getlogger: ok\n"
            "log = logging.getLogger(__name__)\n",
        )
        violations = find_stdlib_getlogger_calls(
            (src,),
            repo,
            {},
            _GETLOGGER_MARKER,
        )
        assert len(violations) == 1
        assert violations[0].line == 4

    def test_exempt_file_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "config.py",
            "from __future__ import annotations\nimport logging\nlog = logging.getLogger(__name__)\n",
        )
        exempt = {
            "src/pkg/config.py": ("platform bootstrap; stdlib logging used pre-observe init"),
        }
        violations = find_stdlib_getlogger_calls(
            (src,),
            repo,
            exempt,
            _GETLOGGER_MARKER,
        )
        assert violations == []

    def test_bare_getlogger_call_flagged(self, tmp_path: Path) -> None:
        # ``from logging import getLogger; log = getLogger(...)``
        # — the bare name call still counts.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\nfrom logging import getLogger\nlog = getLogger(__name__)\n",
        )
        violations = find_stdlib_getlogger_calls(
            (src,),
            repo,
            {},
            _GETLOGGER_MARKER,
        )
        assert len(violations) == 1
        assert violations[0].line == 3

    def test_obj_getlogger_attribute_not_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        # ``obj.getLogger(...)`` — receiver is not ``logging``.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\ndef foo(obj) -> None:\n    obj.getLogger('x')\n",
        )
        assert (
            find_stdlib_getlogger_calls(
                (src,),
                repo,
                {},
                _GETLOGGER_MARKER,
            )
            == []
        )

    def test_attr_chain_getlogger_not_flagged(self, tmp_path: Path) -> None:
        # ``foo.bar.getLogger(...)`` — receiver is Attribute, not Name.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\ndef f(x) -> None:\n    x.foo.getLogger('a')\n",
        )
        assert (
            find_stdlib_getlogger_calls(
                (src,),
                repo,
                {},
                _GETLOGGER_MARKER,
            )
            == []
        )

    def test_no_getlogger_no_violations(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "from threetears.observe import get_logger\n"
            "log = get_logger(__name__)\n",
        )
        assert (
            find_stdlib_getlogger_calls(
                (src,),
                repo,
                {},
                _GETLOGGER_MARKER,
            )
            == []
        )

    def test_marker_on_one_line_does_not_exempt_other(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "import logging\n"
            "logging.getLogger('a').setLevel(40)  # stdlib-getlogger: ok\n"
            "log = logging.getLogger(__name__)\n",
        )
        violations = find_stdlib_getlogger_calls(
            (src,),
            repo,
            {},
            _GETLOGGER_MARKER,
        )
        assert len(violations) == 1
        assert violations[0].line == 4


# ----------------------------------------------------------------------
# find_missing_future_annotations
# ----------------------------------------------------------------------


class TestFindMissingFutureAnnotations:
    def test_module_with_future_no_violation(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\nx = 1\n",
        )
        assert (
            find_missing_future_annotations(
                (src,),
                repo,
                {},
                _DEFAULT_SKIP,
            )
            == []
        )

    def test_module_without_future_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "x = 1\n")
        violations = find_missing_future_annotations(
            (src,),
            repo,
            {},
            _DEFAULT_SKIP,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "codebase_conventions.future_annotations"
        assert v.symbol == "__future__"

    def test_init_py_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "__init__.py", "x = 1\n")
        assert (
            find_missing_future_annotations(
                (src,),
                repo,
                {},
                _DEFAULT_SKIP,
            )
            == []
        )

    def test_exempt_file_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "legacy.py", "x = 1\n")
        exempt = {
            "src/pkg/legacy.py": ("legacy module pending migration to PEP 563"),
        }
        assert (
            find_missing_future_annotations(
                (src,),
                repo,
                exempt,
                _DEFAULT_SKIP,
            )
            == []
        )

    def test_only_other_future_imports_flagged(self, tmp_path: Path) -> None:
        # has a __future__ import but not for ``annotations``.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import division\nx = 1\n",
        )
        violations = find_missing_future_annotations(
            (src,),
            repo,
            {},
            _DEFAULT_SKIP,
        )
        assert len(violations) == 1

    def test_multiple_future_imports_with_annotations_passes(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations, division\nx = 1\n",
        )
        assert (
            find_missing_future_annotations(
                (src,),
                repo,
                {},
                _DEFAULT_SKIP,
            )
            == []
        )

    def test_nested_future_import_does_not_count(
        self,
        tmp_path: Path,
    ) -> None:
        # nested ``__future__`` imports are SyntaxErrors in real
        # python; we use a shape that parses but is nested under a
        # function. ast.iter_child_nodes only walks module top-level.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "if True:\n    from __future__ import annotations\n",
        )
        violations = find_missing_future_annotations(
            (src,),
            repo,
            {},
            _DEFAULT_SKIP,
        )
        assert len(violations) == 1


# ----------------------------------------------------------------------
# find_missing_return_types
# ----------------------------------------------------------------------


class TestFindMissingReturnTypes:
    def test_function_without_return_type_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\ndef foo():\n    return 1\n",
        )
        violations = find_missing_return_types(
            (src,),
            repo,
            {},
            _DEFAULT_SKIP,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "codebase_conventions.return_type"
        assert v.symbol == "foo"
        assert v.line == 2

    def test_function_with_return_type_passes(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\ndef foo() -> int:\n    return 1\n",
        )
        assert (
            find_missing_return_types(
                (src,),
                repo,
                {},
                _DEFAULT_SKIP,
            )
            == []
        )

    def test_dunder_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "class Foo:\n"
            "    def __init__(self):\n"
            "        pass\n"
            "    def __repr__(self):\n"
            "        return 'x'\n",
        )
        assert (
            find_missing_return_types(
                (src,),
                repo,
                {},
                _DEFAULT_SKIP,
            )
            == []
        )

    def test_test_function_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        # NOT in a test_*.py file (which would be skipped wholesale);
        # name starts with test_ so it should still be skipped.
        _write(
            src / "pkg" / "helpers.py",
            "from __future__ import annotations\ndef test_helper():\n    pass\n",
        )
        assert (
            find_missing_return_types(
                (src,),
                repo,
                {},
                _DEFAULT_SKIP,
            )
            == []
        )

    def test_test_file_skipped(self, tmp_path: Path) -> None:
        # the walker skips files whose basename starts with ``test_``.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "test_thing.py",
            "from __future__ import annotations\ndef helper():\n    pass\n",
        )
        assert (
            find_missing_return_types(
                (src,),
                repo,
                {},
                _DEFAULT_SKIP,
            )
            == []
        )

    def test_init_py_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "__init__.py",
            "from __future__ import annotations\ndef re_export():\n    pass\n",
        )
        assert (
            find_missing_return_types(
                (src,),
                repo,
                {},
                _DEFAULT_SKIP,
            )
            == []
        )

    def test_exempt_file_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "legacy.py",
            "from __future__ import annotations\ndef foo():\n    pass\n",
        )
        exempt = {
            "src/pkg/legacy.py": ("legacy module pending migration to typed signatures"),
        }
        assert (
            find_missing_return_types(
                (src,),
                repo,
                exempt,
                _DEFAULT_SKIP,
            )
            == []
        )

    def test_async_function_without_return_type_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\nasync def foo():\n    return 1\n",
        )
        violations = find_missing_return_types(
            (src,),
            repo,
            {},
            _DEFAULT_SKIP,
        )
        assert len(violations) == 1
        assert violations[0].symbol == "foo"

    def test_method_without_return_type_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\nclass Foo:\n    def bar(self):\n        return 1\n",
        )
        violations = find_missing_return_types(
            (src,),
            repo,
            {},
            _DEFAULT_SKIP,
        )
        assert len(violations) == 1
        assert violations[0].symbol == "bar"

    def test_name_mangled_double_underscore_prefix_not_dunder(
        self,
        tmp_path: Path,
    ) -> None:
        # ``__foo`` (starts with __, doesn't end with __) is name-mangled,
        # NOT a dunder. it should be flagged.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\nclass Foo:\n    def __mangled(self):\n        return 1\n",
        )
        violations = find_missing_return_types(
            (src,),
            repo,
            {},
            _DEFAULT_SKIP,
        )
        assert len(violations) == 1
        assert violations[0].symbol == "__mangled"


# ----------------------------------------------------------------------
# multi-root path-dep walking
# ----------------------------------------------------------------------


class TestPathDepWalking:
    def test_two_package_workspace_finds_violations(
        self,
        tmp_path: Path,
    ) -> None:
        a_src = tmp_path / "a" / "src"
        b_src = tmp_path / "b" / "src"
        _write(
            a_src / "pkg_a" / "mod.py",
            "from __future__ import annotations\n",
        )
        _write(
            b_src / "pkg_b" / "mod.py",
            "from __future__ import annotations\ndef foo():\n    return 1\n",
        )
        violations = find_missing_return_types(
            (a_src, b_src),
            tmp_path,
            {},
            _DEFAULT_SKIP,
        )
        assert len(violations) == 1
        assert violations[0].file == b_src / "pkg_b" / "mod.py"
        assert violations[0].symbol == "foo"
