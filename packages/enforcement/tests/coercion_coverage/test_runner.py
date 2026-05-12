"""tests for ``run_coercion_enforcement`` orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.coercion_coverage import (
    CoerceCoverageConfig,
    run_coercion_enforcement,
)


_VALID_RATIONALE = "intentional override under controlled fixture for unit-test use only"


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo_with_pyproject(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _build_violation_repo(tmp_path: Path) -> tuple[Path, Path]:
    """build a synthetic repo with one run-override violation.

    :return: (repo_root, src_root)
    :rtype: tuple[Path, Path]
    """
    repo = _make_repo_with_pyproject(tmp_path / "repo")
    src = repo / "src"
    _write(
        src / "pkg" / "tool.py",
        "class FooTool(Tool):\n    def run(self):\n        return None\n",
    )
    return repo, src


# ------------------------------------------------------------------
# input validation
# ------------------------------------------------------------------


class TestWalkerArgValidation:
    def test_unknown_walker_raises(self, tmp_path: Path) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        config = CoerceCoverageConfig(repo_root=repo, src_roots=())
        with pytest.raises(ValueError, match="walker must be one of"):
            run_coercion_enforcement(config, walker="bogus")

    def test_default_walker_is_all(self, tmp_path: Path) -> None:
        # bare invocation with no walker argument should work.
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        config = CoerceCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="COERCE_TEST_MODE",
        )
        run_coercion_enforcement(config)


# ------------------------------------------------------------------
# strict / report mode behaviour
# ------------------------------------------------------------------


class TestModes:
    def test_strict_fails_on_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.setenv("COERCE_TEST_MODE", "strict")
        config = CoerceCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="COERCE_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_coercion_enforcement(config)

    def test_report_mode_returns_silently(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.setenv("COERCE_TEST_MODE", "report")
        config = CoerceCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="COERCE_TEST_MODE",
        )
        # must not raise even though violations exist.
        run_coercion_enforcement(config)

    def test_clean_run_does_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "tool.py",
            "class FooTool(Tool):\n    def execute(self):\n        return None\n",
        )
        monkeypatch.setenv("COERCE_TEST_MODE", "strict")
        config = CoerceCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="COERCE_TEST_MODE",
        )
        run_coercion_enforcement(config)

    def test_default_mode_is_strict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.delenv("COERCE_TEST_MODE", raising=False)
        config = CoerceCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="COERCE_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_coercion_enforcement(config)


# ------------------------------------------------------------------
# exemption application
# ------------------------------------------------------------------


class TestExemptions:
    def test_exemption_removes_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        # the violation lives at src/pkg/tool.py:2 with symbol FooTool.
        ex_path = repo / "_coercion_exemptions.txt"
        ex_path.write_text(f"# rationale: {_VALID_RATIONALE}\nsrc/pkg/tool.py:2:FooTool\n")
        monkeypatch.setenv("COERCE_TEST_MODE", "strict")
        config = CoerceCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
            mode_env_var="COERCE_TEST_MODE",
        )
        # with the exemption, strict mode should pass.
        run_coercion_enforcement(config)

    def test_exemption_path_none_skips_loading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        monkeypatch.setenv("COERCE_TEST_MODE", "strict")
        config = CoerceCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=None,
            mode_env_var="COERCE_TEST_MODE",
        )
        run_coercion_enforcement(config)


# ------------------------------------------------------------------
# explicit src_roots vs discovery
# ------------------------------------------------------------------


class TestSrcRootsDiscovery:
    def test_falls_back_to_discover(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # build a repo with a single src root and no path-deps; with
        # src_roots=None the runner should call discover_src_roots
        # against the pyproject and find the src tree itself.
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "tool.py",
            "class FooTool(Tool):\n    def execute(self):\n        return None\n",
        )
        monkeypatch.setenv("COERCE_TEST_MODE", "strict")
        config = CoerceCoverageConfig(
            repo_root=repo,
            src_roots=None,
            mode_env_var="COERCE_TEST_MODE",
        )
        run_coercion_enforcement(config)


# ------------------------------------------------------------------
# custom base_class_suffixes
# ------------------------------------------------------------------


class TestCustomSuffixes:
    def test_custom_suffix_flags_action_class(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "thing.py",
            "class FooAction(BaseAction):\n    def run(self):\n        return None\n",
        )
        monkeypatch.setenv("COERCE_TEST_MODE", "strict")
        config = CoerceCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="COERCE_TEST_MODE",
            base_class_suffixes=frozenset({"Action"}),
        )
        with pytest.raises(pytest.fail.Exception):
            run_coercion_enforcement(config)

    def test_default_suffix_does_not_flag_action_class(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo_with_pyproject(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "thing.py",
            "class FooAction(BaseAction):\n    def run(self):\n        return None\n",
        )
        monkeypatch.setenv("COERCE_TEST_MODE", "strict")
        config = CoerceCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="COERCE_TEST_MODE",
        )
        # default suffix set is {"Tool"} — Action is not flagged.
        run_coercion_enforcement(config)
