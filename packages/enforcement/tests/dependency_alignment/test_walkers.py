"""walker tests for dependency-alignment enforcement.

builds synthetic uv-workspace shapes under ``tmp_path`` and asserts
the four contract rules:

1. an unguarded module-top import of an undeclared sibling -> missing
2. a declared workspace dep with zero imports anywhere -> stale
3. guarded / deferred / TYPE_CHECKING imports are never missing
4. extras-declared and guarded-used deps are neither missing nor stale
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.dependency_alignment import (
    DependencyAlignmentConfig,
    dependency_alignment_violations,
)


def _make_package(
    root: Path,
    dist: str,
    namespace: str,
    body: str,
    dependencies: list[str] | None = None,
    extras: dict[str, list[str]] | None = None,
) -> None:
    """scaffold one workspace package with a single module.

    :param root: workspace root
    :ptype root: Path
    :param dist: distribution name for ``[project] name``
    :ptype dist: str
    :param namespace: dotted import namespace (``threetears.alpha``)
    :ptype namespace: str
    :param body: source for the package's ``api.py`` module
    :ptype body: str
    :param dependencies: ``[project] dependencies`` entries
    :ptype dependencies: list[str] | None
    :param extras: optional-dependency groups
    :ptype extras: dict[str, list[str]] | None
    """
    pkg_dir = root / "packages" / dist
    deps = "".join(f'    "{d}",\n' for d in (dependencies or []))
    toml = f'[project]\nname = "{dist}"\nversion = "0.0.1"\ndependencies = [\n{deps}]\n'
    if extras:
        toml += "\n[project.optional-dependencies]\n"
        for group, items in extras.items():
            listing = ", ".join(f'"{i}"' for i in items)
            toml += f"{group} = [{listing}]\n"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "pyproject.toml").write_text(toml)
    module_dir = pkg_dir / "src"
    for part in namespace.split("."):
        module_dir = module_dir / part
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text("")
    (module_dir / "api.py").write_text(body)


def _violations(root: Path) -> list:
    config = DependencyAlignmentConfig(repo_root=root, package_globs=("packages/*",))
    return dependency_alignment_violations(config)


class TestMissingDetection:
    def test_unguarded_top_import_of_undeclared_sibling_is_missing(self, tmp_path: Path) -> None:
        _make_package(tmp_path, "tt-alpha", "threetears.alpha", "VALUE = 1\n")
        _make_package(
            tmp_path,
            "tt-beta",
            "threetears.beta",
            "from threetears.alpha.api import VALUE\n",
        )
        found = _violations(tmp_path)
        assert [v.category for v in found] == ["dependency.missing"]
        assert found[0].symbol == "tt_alpha"
        assert "tt-beta imports threetears.alpha.api" in found[0].reason

    def test_declared_dependency_is_not_missing(self, tmp_path: Path) -> None:
        _make_package(tmp_path, "tt-alpha", "threetears.alpha", "VALUE = 1\n")
        _make_package(
            tmp_path,
            "tt-beta",
            "threetears.beta",
            "from threetears.alpha.api import VALUE\n",
            dependencies=["tt-alpha"],
        )
        assert _violations(tmp_path) == []

    def test_extra_declared_dependency_is_not_missing(self, tmp_path: Path) -> None:
        _make_package(tmp_path, "tt-alpha", "threetears.alpha", "VALUE = 1\n")
        _make_package(
            tmp_path,
            "tt-beta",
            "threetears.beta",
            "from threetears.alpha.api import VALUE\n",
            extras={"alpha": ["tt-alpha"]},
        )
        assert _violations(tmp_path) == []


class TestShieldedImportsAreNotMissing:
    def test_type_checking_import(self, tmp_path: Path) -> None:
        _make_package(tmp_path, "tt-alpha", "threetears.alpha", "VALUE = 1\n")
        _make_package(
            tmp_path,
            "tt-beta",
            "threetears.beta",
            ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from threetears.alpha.api import VALUE\n"),
        )
        assert _violations(tmp_path) == []

    def test_import_error_guarded_import(self, tmp_path: Path) -> None:
        _make_package(tmp_path, "tt-alpha", "threetears.alpha", "VALUE = 1\n")
        _make_package(
            tmp_path,
            "tt-beta",
            "threetears.beta",
            ("try:\n    from threetears.alpha.api import VALUE\nexcept ImportError:\n    VALUE = None\n"),
        )
        assert _violations(tmp_path) == []

    def test_function_body_import(self, tmp_path: Path) -> None:
        _make_package(tmp_path, "tt-alpha", "threetears.alpha", "VALUE = 1\n")
        _make_package(
            tmp_path,
            "tt-beta",
            "threetears.beta",
            ("def load():\n    from threetears.alpha.api import VALUE\n    return VALUE\n"),
        )
        assert _violations(tmp_path) == []


class TestStaleDetection:
    def test_declared_but_never_imported_is_stale(self, tmp_path: Path) -> None:
        _make_package(tmp_path, "tt-alpha", "threetears.alpha", "VALUE = 1\n")
        _make_package(
            tmp_path,
            "tt-beta",
            "threetears.beta",
            "VALUE = 2\n",
            dependencies=["tt-alpha"],
        )
        found = _violations(tmp_path)
        assert [v.category for v in found] == ["dependency.stale"]
        assert found[0].symbol == "tt_alpha"
        assert found[0].file.name == "pyproject.toml"

    def test_guarded_usage_counts_against_staleness(self, tmp_path: Path) -> None:
        _make_package(tmp_path, "tt-alpha", "threetears.alpha", "VALUE = 1\n")
        _make_package(
            tmp_path,
            "tt-beta",
            "threetears.beta",
            ("def load():\n    from threetears.alpha.api import VALUE\n    return VALUE\n"),
            dependencies=["tt-alpha"],
        )
        assert _violations(tmp_path) == []

    def test_non_workspace_deps_are_ignored(self, tmp_path: Path) -> None:
        _make_package(
            tmp_path,
            "tt-beta",
            "threetears.beta",
            "VALUE = 2\n",
            dependencies=["pydantic>=2.0"],
        )
        assert _violations(tmp_path) == []


class TestContractPurity:
    def _purity(self, root: Path, contract: str) -> list:
        from threetears.enforcement.dependency_alignment import (
            DependencyAlignmentConfig,
        )
        from threetears.enforcement.dependency_alignment.walkers import (
            contract_purity_violations,
        )

        config = DependencyAlignmentConfig(
            repo_root=root,
            package_globs=("packages/*",),
            contract_packages=(f"packages/{contract}",),
        )
        return contract_purity_violations(config)

    def test_stdlib_only_contract_is_clean(self, tmp_path: Path) -> None:
        _make_package(
            tmp_path,
            "tt-contracts",
            "threetears.contracts",
            (
                "from dataclasses import dataclass\nfrom typing import Protocol\n\n@dataclass\nclass Thing:\n    name: str\n"
            ),
        )
        assert self._purity(tmp_path, "tt-contracts") == []

    def test_third_party_top_import_is_impure(self, tmp_path: Path) -> None:
        _make_package(
            tmp_path,
            "tt-contracts",
            "threetears.contracts",
            "import sqlalchemy\n",
        )
        found = self._purity(tmp_path, "tt-contracts")
        assert [v.category for v in found] == ["contract.impure"]
        assert found[0].symbol == "sqlalchemy"

    def test_own_namespace_import_is_clean(self, tmp_path: Path) -> None:
        _make_package(
            tmp_path,
            "tt-contracts",
            "threetears.contracts",
            "from threetears.contracts.api import VALUE\nVALUE2 = VALUE\n",
        )
        # rewrite api.py so the self-import has a target symbol
        api = tmp_path / "packages" / "tt-contracts" / "src" / "threetears" / "contracts" / "api.py"
        api.write_text("VALUE = 1\nfrom threetears.contracts.api import VALUE\n")
        assert self._purity(tmp_path, "tt-contracts") == []

    def test_type_checking_shielded_import_is_clean(self, tmp_path: Path) -> None:
        _make_package(
            tmp_path,
            "tt-contracts",
            "threetears.contracts",
            ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import sqlalchemy\n"),
        )
        assert self._purity(tmp_path, "tt-contracts") == []

    def test_extra_allowed_prefix_is_clean(self, tmp_path: Path) -> None:
        from threetears.enforcement.dependency_alignment import (
            DependencyAlignmentConfig,
        )
        from threetears.enforcement.dependency_alignment.walkers import (
            contract_purity_violations,
        )

        _make_package(
            tmp_path,
            "tt-contracts",
            "threetears.contracts",
            "from pydantic import BaseModel\n",
        )
        config = DependencyAlignmentConfig(
            repo_root=tmp_path,
            package_globs=("packages/*",),
            contract_packages=("packages/tt-contracts",),
            contract_extra_allowed=("pydantic",),
        )
        assert contract_purity_violations(config) == []
