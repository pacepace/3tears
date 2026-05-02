"""tests for the five underscore-access shape walkers."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.underscore_access.walkers import (
    package_id,
    parse_ruff_slf001_output,
    same_package,
    shape_a_violations,
    shape_c_violations,
    shape_d_violations,
    shape_e_violations,
)


_DEFAULT_SHAPE_C_SKIP: frozenset[str] = frozenset({
    "conftest.py", "__main__.py", "_version.py",
})


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


# ------------------------------------------------------------------
# package-identity helpers
# ------------------------------------------------------------------

class TestPackageIdentity:
    def test_package_id_under_root(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        file = _write(src / "pkg" / "mod.py", "")
        ident = package_id(file, [src])
        assert ident == (str(src.resolve()), "pkg")

    def test_package_id_outside_root(self, tmp_path: Path) -> None:
        other = tmp_path / "other"
        file = _write(other / "x.py", "")
        ident = package_id(file, [tmp_path / "src"])
        assert ident == ()

    def test_same_package_true(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        a = _write(src / "pkg" / "a.py", "")
        b = _write(src / "pkg" / "sub" / "b.py", "")
        assert same_package(a, b, [src]) is True

    def test_same_package_false_different_top(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        a = _write(src / "alpha" / "a.py", "")
        b = _write(src / "beta" / "b.py", "")
        assert same_package(a, b, [src]) is False

    def test_same_package_false_outside_root(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        a = _write(src / "pkg" / "a.py", "")
        b = _write(tmp_path / "outer" / "b.py", "")
        assert same_package(a, b, [src]) is False


# ------------------------------------------------------------------
# shape A — cross-module private import
# ------------------------------------------------------------------

class TestShapeA:
    def test_cross_package_private_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "pkg" / "__init__.py", "")
        _write(src / "pkg" / "_helper.py", "_name = 1\n")
        _write(
            src / "consumer" / "__init__.py",
            "from pkg._helper import _name\n",
        )
        violations = shape_a_violations((src,), tmp_path, (src,))
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "underscore_access.A"
        assert v.symbol == "_name"
        assert v.file == src / "consumer" / "__init__.py"

    def test_same_package_private_allowed(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "pkg" / "__init__.py", "")
        _write(src / "pkg" / "_helper.py", "_name = 1\n")
        _write(
            src / "pkg" / "consumer.py",
            "from pkg._helper import _name\n",
        )
        violations = shape_a_violations((src,), tmp_path, (src,))
        assert violations == []

    def test_relative_import_allowed(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "pkg" / "__init__.py", "")
        _write(src / "pkg" / "_helper.py", "_name = 1\n")
        _write(
            src / "pkg" / "consumer.py",
            "from ._helper import _name\n",
        )
        violations = shape_a_violations((src,), tmp_path, (src,))
        assert violations == []

    def test_unresolved_module_skipped(self, tmp_path: Path) -> None:
        # third-party module not in any src root must be skipped.
        src = tmp_path / "src"
        _write(
            src / "pkg" / "consumer.py",
            "from third_party import _name\n",
        )
        violations = shape_a_violations((src,), tmp_path, (src,))
        assert violations == []

    def test_public_import_not_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "pkg" / "__init__.py", "")
        _write(src / "pkg" / "helper.py", "name = 1\n")
        _write(
            src / "consumer" / "__init__.py",
            "from pkg.helper import name\n",
        )
        violations = shape_a_violations((src,), tmp_path, (src,))
        assert violations == []

    def test_path_dep_walking_treated_as_same_package_under_combined_roots(
        self, tmp_path: Path,
    ) -> None:
        # option-B behaviour: when shape A receives the union of all
        # src roots from the runner, an importer in repo A and a
        # private module in path-dep package B that share a top-level
        # package name should still be flagged because they live under
        # different src roots. this asserts the (src_root, top_pkg)
        # identity rule.
        consumer_src = tmp_path / "consumer" / "src"
        lib_src = tmp_path / "lib" / "src"
        _write(lib_src / "lib" / "__init__.py", "")
        _write(lib_src / "lib" / "_internals.py", "_helper = 1\n")
        _write(
            consumer_src / "consumer_pkg" / "__init__.py",
            "from lib._internals import _helper\n",
        )
        # both src roots provided to the walker, mimicking
        # discover_src_roots's output for a repo with a path-dep.
        violations = shape_a_violations(
            (consumer_src, lib_src), tmp_path, (consumer_src, lib_src),
        )
        assert len(violations) == 1
        assert violations[0].symbol == "_helper"


# ------------------------------------------------------------------
# shape B — ruff SLF001 parsing
# ------------------------------------------------------------------

class TestShapeBParsing:
    def test_parses_ruff_concise_output(self, tmp_path: Path) -> None:
        stdout = (
            "src/pkg/file.py:42:5: SLF001 Private member accessed: `_internal`\n"
            "src/pkg/other.py:11:3: SLF001 Private member accessed: `_state`\n"
        )
        violations = parse_ruff_slf001_output(stdout, tmp_path)
        assert len(violations) == 2
        v1, v2 = violations
        assert v1.category == "underscore_access.B"
        assert v1.line == 42
        assert v1.symbol == "_internal"
        assert v1.file == (tmp_path / "src/pkg/file.py").resolve()
        assert v2.symbol == "_state"
        assert v2.line == 11

    def test_unrelated_lines_ignored(self, tmp_path: Path) -> None:
        stdout = (
            "noise\n"
            "src/pkg/file.py:1:1: F401 unused import\n"
            "src/pkg/file.py:42:5: SLF001 Private member accessed: `_x`\n"
        )
        violations = parse_ruff_slf001_output(stdout, tmp_path)
        assert len(violations) == 1
        assert violations[0].symbol == "_x"

    def test_message_without_backtick_falls_back(self, tmp_path: Path) -> None:
        stdout = "src/p.py:1:1: SLF001 some unusual message\n"
        violations = parse_ruff_slf001_output(stdout, tmp_path)
        assert len(violations) == 1
        assert violations[0].symbol == "_<unknown>"

    def test_absolute_path_preserved(self, tmp_path: Path) -> None:
        abs_file = tmp_path / "abs.py"
        stdout = f"{abs_file}:3:2: SLF001 Private member accessed: `_z`\n"
        violations = parse_ruff_slf001_output(stdout, tmp_path)
        assert violations[0].file == abs_file


# ------------------------------------------------------------------
# shape C — missing __all__
# ------------------------------------------------------------------

class TestShapeC:
    def test_missing_all_with_public_names_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            "def public_fn():\n    pass\n\nclass Foo:\n    pass\n",
        )
        violations = shape_c_violations((src,), tmp_path, _DEFAULT_SHAPE_C_SKIP)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "underscore_access.C"
        assert v.symbol == "__all__"
        assert "public_fn" in v.reason

    def test_with_all_not_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            '__all__ = ["public_fn"]\n\ndef public_fn():\n    pass\n',
        )
        violations = shape_c_violations((src,), tmp_path, _DEFAULT_SHAPE_C_SKIP)
        assert violations == []

    def test_skip_basename_not_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "pkg" / "conftest.py", "def fixture():\n    pass\n")
        violations = shape_c_violations((src,), tmp_path, _DEFAULT_SHAPE_C_SKIP)
        assert violations == []

    def test_private_only_module_not_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "internal.py",
            "_private_var = 1\n\ndef _helper():\n    pass\n",
        )
        violations = shape_c_violations((src,), tmp_path, _DEFAULT_SHAPE_C_SKIP)
        assert violations == []

    def test_empty_init_not_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(src / "pkg" / "__init__.py", "")
        violations = shape_c_violations((src,), tmp_path, _DEFAULT_SHAPE_C_SKIP)
        assert violations == []

    def test_annotated_assignment_recognised(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            "VALUE: int = 1\n",
        )
        violations = shape_c_violations((src,), tmp_path, _DEFAULT_SHAPE_C_SKIP)
        assert len(violations) == 1
        assert "VALUE" in violations[0].reason

    def test_annotated_all_satisfies_check(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            '__all__: list[str] = ["public_fn"]\n\ndef public_fn():\n    pass\n',
        )
        violations = shape_c_violations((src,), tmp_path, _DEFAULT_SHAPE_C_SKIP)
        assert violations == []

    def test_custom_skip_basenames_honoured(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "weird.py",
            "def public_fn():\n    pass\n",
        )
        violations = shape_c_violations(
            (src,), tmp_path, frozenset({"weird.py"}),
        )
        assert violations == []


# ------------------------------------------------------------------
# shape D — subclass shadows base private
# ------------------------------------------------------------------

class TestShapeD:
    def test_subclass_shadows_base_method(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "base.py",
            "class Base:\n    def _internal(self):\n        return 1\n",
        )
        _write(
            src / "pkg" / "sub.py",
            "from pkg.base import Base\n\nclass Sub(Base):\n"
            "    def _internal(self):\n        return 2\n",
        )
        violations = shape_d_violations((src,), tmp_path, (src,))
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "underscore_access.D"
        assert v.symbol == "_internal"
        assert "Sub" in v.reason
        assert "Base" in v.reason

    def test_subclass_shadows_base_attribute(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "base.py",
            "class Base:\n    _state = 0\n",
        )
        _write(
            src / "pkg" / "sub.py",
            "class Sub(Base):\n    _state = 1\n",
        )
        violations = shape_d_violations((src,), tmp_path, (src,))
        assert len(violations) == 1
        assert violations[0].symbol == "_state"

    def test_subclass_with_distinct_private_not_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "base.py",
            "class Base:\n    def _alpha(self):\n        return 1\n",
        )
        _write(
            src / "pkg" / "sub.py",
            "class Sub(Base):\n    def _beta(self):\n        return 2\n",
        )
        violations = shape_d_violations((src,), tmp_path, (src,))
        assert violations == []

    def test_class_without_base_not_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Solo:\n    _x = 1\n",
        )
        violations = shape_d_violations((src,), tmp_path, (src,))
        assert violations == []

    def test_subclass_definition_does_not_flag_self(self, tmp_path: Path) -> None:
        # ensure a class doesn't flag itself when its own private attr
        # also lives in the class_privates map under its name.
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    _x = 1\n",
        )
        violations = shape_d_violations((src,), tmp_path, (src,))
        assert violations == []

    def test_attribute_base_recognised(self, tmp_path: Path) -> None:
        # class Sub(pkg.Base): -- the attribute base resolves textually
        # to "Base" via the last-segment rule.
        src = tmp_path / "src"
        _write(
            src / "pkg" / "base.py",
            "class Base:\n    def _hidden(self):\n        return 1\n",
        )
        _write(
            src / "pkg" / "sub.py",
            "import pkg\n\nclass Sub(pkg.Base):\n"
            "    def _hidden(self):\n        return 2\n",
        )
        violations = shape_d_violations((src,), tmp_path, (src,))
        assert len(violations) == 1
        assert violations[0].symbol == "_hidden"


# ------------------------------------------------------------------
# shape E — __all__ lists private names
# ------------------------------------------------------------------

class TestShapeE:
    def test_private_name_in_all_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            '__all__ = ["_x"]\n\n_x = 1\n',
        )
        violations = shape_e_violations((src,), tmp_path)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "underscore_access.E"
        assert v.symbol == "_x"

    def test_public_name_in_all_not_flagged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            '__all__ = ["x"]\n\nx = 1\n',
        )
        violations = shape_e_violations((src,), tmp_path)
        assert violations == []

    def test_dunder_not_flagged(self, tmp_path: Path) -> None:
        # ``__name__`` -ish entries: starts and ends with underscore;
        # is_private_name excludes them.
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            '__all__ = ["__init__"]\n',
        )
        violations = shape_e_violations((src,), tmp_path)
        assert violations == []

    def test_tuple_form_recognised(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            '__all__ = ("_x",)\n',
        )
        violations = shape_e_violations((src,), tmp_path)
        assert len(violations) == 1
        assert violations[0].symbol == "_x"

    def test_set_form_recognised(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            '__all__ = {"_x"}\n',
        )
        violations = shape_e_violations((src,), tmp_path)
        assert len(violations) == 1

    def test_computed_form_skipped(self, tmp_path: Path) -> None:
        # computed/non-literal forms can't be statically analysed.
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            "__all__ = list(['_x'])\n",
        )
        violations = shape_e_violations((src,), tmp_path)
        assert violations == []

    def test_annotated_assignment_form(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            '__all__: list[str] = ["_x"]\n',
        )
        violations = shape_e_violations((src,), tmp_path)
        assert len(violations) == 1
        assert violations[0].symbol == "_x"

    def test_multiple_private_entries(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write(
            src / "pkg" / "mod.py",
            '__all__ = [\n    "_a",\n    "public",\n    "_b",\n]\n',
        )
        violations = shape_e_violations((src,), tmp_path)
        assert {v.symbol for v in violations} == {"_a", "_b"}
