"""tests for ``run_underscore_enforcement`` orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.underscore_access import (
    UnderscoreAccessConfig,
    run_underscore_enforcement,
)


_VALID_RATIONALE = "framework-stable internal sentinel that test legitimately reads"


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo_with_pyproject(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _build_shape_a_repo(tmp_path: Path) -> tuple[Path, Path]:
    """build a synthetic repo with one shape-A violation.

    :return: (repo_root, src_root)
    :rtype: tuple[Path, Path]
    """
    repo = _make_repo_with_pyproject(tmp_path / "repo")
    src = repo / "src"
    _write(src / "pkg" / "__init__.py", "")
    _write(src / "pkg" / "_helper.py", "_name = 1\n")
    _write(
        src / "consumer" / "__init__.py",
        "from pkg._helper import _name\n",
    )
    return repo, src


# ------------------------------------------------------------------
# input validation
# ------------------------------------------------------------------


class TestWalkerArgValidation:
    def test_unknown_walker_raises(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        config = UnderscoreAccessConfig(
            repo_root=repo,
            scan_roots=(),
            inheritance_roots=(),
        )
        with pytest.raises(ValueError, match="walker must be one of"):
            run_underscore_enforcement(config, walker="bogus")


# ------------------------------------------------------------------
# strict/report mode behaviour
# ------------------------------------------------------------------


class TestModes:
    def test_strict_fails_on_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_shape_a_repo(tmp_path)
        monkeypatch.setenv("UNDERSCORE_AUDIT_MODE_TEST", "strict")
        config = UnderscoreAccessConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            mode_env_var="UNDERSCORE_AUDIT_MODE_TEST",
            enable_shape_b_ruff=False,
        )
        with pytest.raises(pytest.fail.Exception):
            run_underscore_enforcement(config, walker="shape_a")

    def test_report_mode_returns_silently(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_shape_a_repo(tmp_path)
        monkeypatch.setenv("UNDERSCORE_AUDIT_MODE_TEST", "report")
        config = UnderscoreAccessConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            mode_env_var="UNDERSCORE_AUDIT_MODE_TEST",
            enable_shape_b_ruff=False,
        )
        # must not raise even though violations exist.
        run_underscore_enforcement(config, walker="shape_a")

    def test_clean_run_does_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "__init__.py", "")
        _write(src / "pkg" / "_helper.py", "_name = 1\n")
        _write(src / "pkg" / "consumer.py", "from pkg._helper import _name\n")
        monkeypatch.setenv("UNDERSCORE_AUDIT_MODE_TEST", "strict")
        config = UnderscoreAccessConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            mode_env_var="UNDERSCORE_AUDIT_MODE_TEST",
            enable_shape_b_ruff=False,
        )
        run_underscore_enforcement(config, walker="shape_a")

    def test_default_mode_is_strict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_shape_a_repo(tmp_path)
        monkeypatch.delenv("UNDERSCORE_AUDIT_MODE_TEST", raising=False)
        config = UnderscoreAccessConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            mode_env_var="UNDERSCORE_AUDIT_MODE_TEST",
            enable_shape_b_ruff=False,
        )
        with pytest.raises(pytest.fail.Exception):
            run_underscore_enforcement(config, walker="shape_a")


# ------------------------------------------------------------------
# exemption application
# ------------------------------------------------------------------


class TestExemptions:
    def test_exemption_removes_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_shape_a_repo(tmp_path)
        # the shape-A violation is at src/consumer/__init__.py:1:_name.
        # the exemption uses the repo-relative path with forward slashes.
        ex_path = repo / "_exemptions.txt"
        ex_path.write_text(f"# rationale: {_VALID_RATIONALE}\nsrc/consumer/__init__.py:1:_name\n")
        monkeypatch.setenv("UNDERSCORE_AUDIT_MODE_TEST", "strict")
        config = UnderscoreAccessConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            exemptions_path=ex_path,
            mode_env_var="UNDERSCORE_AUDIT_MODE_TEST",
            enable_shape_b_ruff=False,
        )
        # with the exemption, strict mode should pass.
        run_underscore_enforcement(config, walker="shape_a")

    def test_exemption_path_none_skips_loading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        monkeypatch.setenv("UNDERSCORE_AUDIT_MODE_TEST", "strict")
        config = UnderscoreAccessConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            exemptions_path=None,
            mode_env_var="UNDERSCORE_AUDIT_MODE_TEST",
            enable_shape_b_ruff=False,
        )
        run_underscore_enforcement(config, walker="shape_a")


# ------------------------------------------------------------------
# walker == "all" aggregates correctly
# ------------------------------------------------------------------


class TestWalkerAll:
    def test_all_aggregates_shape_a_c_e(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # build a repo that exercises shapes A, C, and E together.
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "__init__.py", "")
        # shape A: consumer imports private from a different package.
        _write(src / "pkg" / "_helper.py", "_name = 1\n")
        _write(
            src / "consumer" / "__init__.py",
            "from pkg._helper import _name\n",
        )
        # shape C: module with a public name and no __all__.
        _write(src / "pkg" / "shape_c_mod.py", "def public_fn():\n    pass\n")
        # shape E: __all__ lists a private name.
        _write(
            src / "pkg" / "shape_e_mod.py",
            '__all__ = ["_x"]\n\n_x = 1\n',
        )
        monkeypatch.setenv("UNDERSCORE_AUDIT_MODE_TEST", "report")
        config = UnderscoreAccessConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            mode_env_var="UNDERSCORE_AUDIT_MODE_TEST",
            enable_shape_b_ruff=False,
        )
        # run "all" — must not raise (report mode).
        run_underscore_enforcement(config, walker="all")
        # also run "all" in strict mode to confirm aggregation produces
        # a failure with multiple categories surfaced.
        monkeypatch.setenv("UNDERSCORE_AUDIT_MODE_TEST", "strict")
        with pytest.raises(pytest.fail.Exception) as exc_info:
            run_underscore_enforcement(config, walker="all")
        msg = str(exc_info.value)
        assert "underscore_access.A" in msg
        assert "underscore_access.C" in msg
        assert "underscore_access.E" in msg


# ------------------------------------------------------------------
# explicit roots vs discovery
# ------------------------------------------------------------------


class TestRootsDiscovery:
    def test_falls_back_to_discover(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # build a repo with a single src root and no path-deps; with
        # both root fields = None the runner should call
        # find_local_src_roots (scan) and discover_src_roots
        # (inheritance) against the pyproject and find the src tree.
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "__init__.py", "")
        _write(src / "pkg" / "ok.py", "")
        monkeypatch.setenv("UNDERSCORE_AUDIT_MODE_TEST", "strict")
        config = UnderscoreAccessConfig(
            repo_root=repo,
            scan_roots=None,
            inheritance_roots=None,
            mode_env_var="UNDERSCORE_AUDIT_MODE_TEST",
            enable_shape_b_ruff=False,
        )
        # clean repo, no violations expected.
        run_underscore_enforcement(config, walker="shape_a")


# ------------------------------------------------------------------
# shape-B ruff toggle
# ------------------------------------------------------------------


class TestShapeBToggle:
    def test_disabled_shape_b_yields_no_violations(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "__init__.py", "")
        _write(
            src / "pkg" / "obj.py",
            "class Obj:\n    def __init__(self):\n        self._x = 1\n",
        )
        # arbitrary cross-class private access that ruff would flag.
        _write(
            src / "pkg" / "consumer.py",
            "from pkg.obj import Obj\n\ndef caller():\n    o = Obj()\n    return o._x\n",
        )
        monkeypatch.setenv("UNDERSCORE_AUDIT_MODE_TEST", "strict")
        config = UnderscoreAccessConfig(
            repo_root=repo,
            scan_roots=(src,),
            inheritance_roots=(src,),
            mode_env_var="UNDERSCORE_AUDIT_MODE_TEST",
            enable_shape_b_ruff=False,
        )
        # disabled shape B: zero violations, strict passes.
        run_underscore_enforcement(config, walker="shape_b")
