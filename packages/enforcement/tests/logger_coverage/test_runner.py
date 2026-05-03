"""tests for ``run_logger_coverage_enforcement`` orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.logger_coverage import (
    LoggerCoverageConfig,
    run_logger_coverage_enforcement,
)


_VALID_RATIONALE = "intentional silent module under controlled fixture for unit-test use only"


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _build_violation_repo(tmp_path: Path) -> tuple[Path, Path]:
    """build a synthetic repo with one missing-logger violation.

    :return: (repo_root, src_root)
    :rtype: tuple[Path, Path]
    """
    repo = _make_repo(tmp_path / "repo")
    src = repo / "src"
    _write(src / "pkg" / "mod.py", "x = 1\n")
    return repo, src


# ------------------------------------------------------------------
# input validation
# ------------------------------------------------------------------


class TestWalkerArgValidation:
    def test_unknown_walker_raises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        config = LoggerCoverageConfig(repo_root=repo, src_roots=())
        with pytest.raises(ValueError, match="walker must be one of"):
            run_logger_coverage_enforcement(config, walker="bogus")

    def test_default_walker_is_all(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "__init__.py", "")
        monkeypatch.setenv("LC_TEST_MODE", "strict")
        config = LoggerCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="LC_TEST_MODE",
        )
        run_logger_coverage_enforcement(config)


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
        monkeypatch.setenv("LC_TEST_MODE", "strict")
        config = LoggerCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="LC_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_logger_coverage_enforcement(config)

    def test_report_mode_returns_silently(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.setenv("LC_TEST_MODE", "report")
        config = LoggerCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="LC_TEST_MODE",
        )
        # must not raise even though violations exist.
        run_logger_coverage_enforcement(config)

    def test_clean_run_does_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.observe import get_logger\nlog = get_logger(__name__)\n",
        )
        monkeypatch.setenv("LC_TEST_MODE", "strict")
        config = LoggerCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="LC_TEST_MODE",
        )
        run_logger_coverage_enforcement(config)

    def test_default_mode_is_strict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.delenv("LC_TEST_MODE", raising=False)
        config = LoggerCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="LC_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_logger_coverage_enforcement(config)


# ------------------------------------------------------------------
# exemption application — exempt_files (file-level, in-walker)
# ------------------------------------------------------------------


class TestExemptFiles:
    def test_exempt_file_silences_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.setenv("LC_TEST_MODE", "strict")
        config = LoggerCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="LC_TEST_MODE",
            exempt_files={
                "src/pkg/mod.py": ("pure pydantic model module, no runtime behaviour"),
            },
        )
        run_logger_coverage_enforcement(config)


# ------------------------------------------------------------------
# exemption application — exemptions_path (line-level, parsed file)
# ------------------------------------------------------------------


class TestExemptionsPath:
    def test_exemption_file_removes_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # the walker emits ``symbol`` = relative posix path. the
        # exemption parser's grammar does not accept ``/`` or ``.`` in
        # the symbol slot (it requires an identifier). this domain
        # therefore relies on ``exempt_files`` for real-world use; we
        # still verify the exemptions_path plumbing is wired by
        # passing a file with no matching lines and confirming the
        # violation is unchanged.
        repo, src = _build_violation_repo(tmp_path)
        ex_path = repo / "_logger_coverage_exemptions.txt"
        ex_path.write_text(f"# rationale: {_VALID_RATIONALE}\nsrc/other/unrelated.py:1:placeholder\n")
        monkeypatch.setenv("LC_TEST_MODE", "strict")
        config = LoggerCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
            mode_env_var="LC_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_logger_coverage_enforcement(config)

    def test_exemption_path_none_skips_loading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "__init__.py", "")
        monkeypatch.setenv("LC_TEST_MODE", "strict")
        config = LoggerCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=None,
            mode_env_var="LC_TEST_MODE",
        )
        run_logger_coverage_enforcement(config)


# ------------------------------------------------------------------
# explicit src_roots vs discovery
# ------------------------------------------------------------------


class TestSrcRootsDiscovery:
    def test_falls_back_to_discover(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # build a clean repo with src/, no path-deps; with
        # src_roots=None the runner should call discover_src_roots
        # and find the src tree itself.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.observe import get_logger\nlog = get_logger(__name__)\n",
        )
        monkeypatch.setenv("LC_TEST_MODE", "strict")
        config = LoggerCoverageConfig(
            repo_root=repo,
            src_roots=None,
            mode_env_var="LC_TEST_MODE",
        )
        run_logger_coverage_enforcement(config)


# ------------------------------------------------------------------
# custom factory / var configuration through the runner
# ------------------------------------------------------------------


class TestCustomConfig:
    def test_custom_factory_name_accepted_end_to_end(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from somewhere import make_logger\nlog = make_logger(__name__)\n",
        )
        monkeypatch.setenv("LC_TEST_MODE", "strict")
        config = LoggerCoverageConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="LC_TEST_MODE",
            logger_factory_names=frozenset({"get_logger", "make_logger"}),
        )
        run_logger_coverage_enforcement(config)
