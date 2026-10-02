"""the declared-imports gate: each reader, and the gate as a whole, seen to fail.

The comparison has to be seen to fail, or it is indistinguishable from one that cannot. Every
finding the runner can report is produced here from a repo tree the test authored, against the
real installed environment (``pytest`` is installed wherever this runs, owned by ``pytest``).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from threetears.enforcement.imports_declared import (
    ImportsDeclaredConfig,
    imported_modules,
    imports_declared_findings,
    is_governed,
    module_name,
    module_owners,
    requirement_name,
    run_imports_declared_enforcement,
    undeclared_imports,
)


def _repo(tmp_path: Path, *, source: str, dependencies: list[str]) -> Path:
    """a one-module repo: ``src/probe_pkg/mod.py`` and a manifest declaring ``dependencies``.

    :param tmp_path: scratch directory
    :ptype tmp_path: Path
    :param source: the module's source
    :ptype source: str
    :param dependencies: the manifest's ``[project] dependencies``
    :ptype dependencies: list[str]
    :return: the repo root
    :rtype: Path
    """
    module = tmp_path / "src" / "probe_pkg" / "mod.py"
    module.parent.mkdir(parents=True)
    module.write_text(source, encoding="utf-8")
    quoted = ", ".join(f'"{dep}"' for dep in dependencies)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "probe"\nversion = "0"\ndependencies = [{quoted}]\n', encoding="utf-8"
    )
    return tmp_path


def _config(root: Path, **overrides: object) -> ImportsDeclaredConfig:
    """the probe repo's config, guards pointed at ``pytest``, which the probe imports.

    :param root: the probe repo root
    :ptype root: Path
    :param overrides: fields to replace
    :ptype overrides: object
    :return: the config
    :rtype: ImportsDeclaredConfig
    """
    fields: dict[str, object] = {
        "repo_root": root,
        "first_party": frozenset({"probe_pkg"}),
        "required_import_roots": frozenset({"pytest"}),
        "required_owned_modules": frozenset({"pytest"}),
    }
    fields.update(overrides)
    return ImportsDeclaredConfig(**fields)  # type: ignore[arg-type]


class TestTheGateAsAWhole:
    def test_an_import_whose_distribution_is_undeclared_fails_the_gate(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, source="import pytest\nimport probe_pkg.other\n", dependencies=[])
        with pytest.raises(pytest.fail.Exception, match=r"'pytest': \(\['pytest'\], \['src/probe_pkg/mod.py'\]\)"):
            run_imports_declared_enforcement(_config(root))

    def test_a_declared_import_passes(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, source="import pytest\nimport asyncio\n", dependencies=["pytest>=8"])
        run_imports_declared_enforcement(_config(root))

    def test_an_empty_walk_is_a_finding_not_a_pass(self, tmp_path: Path) -> None:
        """the first non-vacuity guard: a tree importing nothing governed checks nothing."""
        root = _repo(tmp_path, source="import asyncio\n", dependencies=[])
        findings = imports_declared_findings(_config(root, required_import_roots=frozenset()))
        assert any("checking nothing" in finding for finding in findings), findings

    def test_a_required_import_root_the_walk_missed_is_a_finding(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, source="import pytest\n", dependencies=["pytest"])
        findings = imports_declared_findings(_config(root, required_import_roots=frozenset({"pytest", "fastapi"})))
        assert any("not ['fastapi']" in finding for finding in findings), findings

    def test_an_owner_map_blind_to_a_required_module_is_a_finding(self, tmp_path: Path) -> None:
        """the OTHER input's guard: the half that silently emptied under editable installs."""
        root = _repo(tmp_path, source="import pytest\n", dependencies=["pytest"])
        findings = imports_declared_findings(_config(root, required_owned_modules=frozenset({"no_such_module_x"})))
        assert any("no_such_module_x" in finding for finding in findings), findings

    def test_an_import_nothing_installed_provides_is_a_finding(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, source="import pytest\nimport no_such_distribution_y\n", dependencies=["pytest"])
        findings = imports_declared_findings(_config(root))
        assert any("no_such_distribution_y" in finding and "no installed" in finding for finding in findings)

    def test_only_the_configured_source_roots_are_walked(self, tmp_path: Path) -> None:
        """a ``src/tests`` tree beside the package is not held to the runtime rule."""
        root = _repo(tmp_path, source="import pytest\n", dependencies=["pytest"])
        stray = root / "src" / "tests" / "test_x.py"
        stray.parent.mkdir(parents=True)
        stray.write_text("import no_such_distribution_z\n", encoding="utf-8")
        assert imports_declared_findings(_config(root, source_roots=("src/probe_pkg",))) == []
        assert "no_such_distribution_z" in imported_modules(_config(root))


class TestUndeclarableRoots:
    """an import that must NOT be declared (a dependency cycle) is set aside, with a reason."""

    def test_an_undeclarable_root_is_set_aside(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, source="import pytest\nimport no_such_hub_q.app\n", dependencies=["pytest"])
        config = _config(root, undeclarable_import_roots={"no_such_hub_q": "the hub depends on us; a cycle"})
        assert imports_declared_findings(config) == []

    def test_an_entry_no_import_uses_is_a_finding(self, tmp_path: Path) -> None:
        """a stale entry would excuse a future import nobody reviewed."""
        root = _repo(tmp_path, source="import pytest\n", dependencies=["pytest"])
        config = _config(root, undeclarable_import_roots={"no_such_hub_q": "the hub depends on us; a cycle"})
        findings = imports_declared_findings(config)
        assert any("['no_such_hub_q'] match no import" in finding for finding in findings), findings

    def test_an_entry_without_a_reason_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="carry no rationale"):
            _config(tmp_path, undeclarable_import_roots={"no_such_hub_q": "  "})


class TestTheComparison:
    def test_an_import_owned_only_by_an_undeclared_distribution_is_reported(self) -> None:
        offenders = undeclared_imports({"PIL": {Path("src/x/media.py")}}, {"PIL": {"pillow"}}, {"fastapi"})
        assert offenders == {"PIL": (["pillow"], ["src/x/media.py"])}

    def test_a_declared_owner_satisfies_the_import(self) -> None:
        assert undeclared_imports({"PIL": {Path("a.py")}}, {"PIL": {"pillow"}}, {"pillow"}) == {}

    def test_one_declared_owner_of_a_shared_namespace_satisfies_it(self) -> None:
        owners = {"google.cloud": {"google-cloud-bigquery", "google-cloud-storage"}}
        assert undeclared_imports({"google.cloud": {Path("a.py")}}, owners, {"google-cloud-storage"}) == {}


class TestTheReaders:
    def test_requirement_names_are_read_through_extras_markers_and_bounds(self) -> None:
        assert requirement_name("3tears-iam[saml]>=0.22,<1.0") == "3tears-iam"
        assert requirement_name("uvicorn[standard]>=0.30") == "uvicorn"
        assert requirement_name("PyJWT>=2.10") == "pyjwt"
        assert requirement_name("psycopg[binary] >=3.2 ; python_version>'3'") == "psycopg"
        assert requirement_name("aibots-agent-admin") == "aibots-agent-admin"
        assert requirement_name("zope.interface>=5") == "zope-interface"

    def test_module_names_come_from_files_not_metadata(self) -> None:
        assert module_name(PurePosixPath("PIL/Image.py")) == "PIL.Image"
        assert module_name(PurePosixPath("PIL/_imaging.cpython-314-darwin.so")) == "PIL._imaging"
        assert module_name(PurePosixPath("threetears/iam/__init__.py")) == "threetears.iam"
        assert module_name(PurePosixPath("pillow-12.0.0.dist-info/RECORD")) is None
        assert module_name(PurePosixPath("../../bin/uvicorn")) is None
        assert module_name(PurePosixPath("yaml-stubs/__init__.pyi")) is None

    def test_stdlib_and_first_party_imports_are_not_governed(self) -> None:
        first_party = frozenset({"aibots"})
        assert not is_governed("asyncio", first_party)
        assert not is_governed("collections.abc", first_party)
        assert not is_governed("aibots.hub.app", first_party)
        assert not is_governed("__future__", first_party)
        assert is_governed("PIL", first_party)
        assert is_governed("threetears.nats", first_party)

    def test_a_shared_namespace_resolves_at_the_full_module_path(self) -> None:
        """``threetears.agent`` is shipped by several distributions; its leaf modules by one each."""
        owners = module_owners()
        assert len(owners["threetears"]) > 1
        assert owners["threetears.enforcement.imports_declared"] == {"3tears-enforcement"}
