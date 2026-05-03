"""tests for ``run_no_stdlib_logging_enforcement`` orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.no_stdlib_logging import (
    NoStdlibLoggingConfig,
    run_no_stdlib_logging_enforcement,
)


_VALID_RATIONALE = "intentional stdlib logging import under controlled fixture for unit-test use only"


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _build_violation_repo(tmp_path: Path) -> tuple[Path, Path]:
    """build a synthetic repo with one stdlib-logging violation.

    :return: (repo_root, src_root)
    :rtype: tuple[Path, Path]
    """
    repo = _make_repo(tmp_path / "repo")
    src = repo / "src"
    _write(src / "pkg" / "mod.py", "import logging\n")
    return repo, src


# ------------------------------------------------------------------
# input validation
# ------------------------------------------------------------------


class TestWalkerArgValidation:
    def test_unknown_walker_raises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        config = NoStdlibLoggingConfig(repo_root=repo, src_roots=())
        with pytest.raises(ValueError, match="walker must be one of"):
            run_no_stdlib_logging_enforcement(config, walker="bogus")

    def test_default_walker_is_all(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        config = NoStdlibLoggingConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSL_TEST_MODE",
        )
        run_no_stdlib_logging_enforcement(config)


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
        monkeypatch.setenv("NSL_TEST_MODE", "strict")
        config = NoStdlibLoggingConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSL_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_no_stdlib_logging_enforcement(config)

    def test_report_mode_returns_silently(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.setenv("NSL_TEST_MODE", "report")
        config = NoStdlibLoggingConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSL_TEST_MODE",
        )
        # must not raise even though violations exist.
        run_no_stdlib_logging_enforcement(config)

    def test_clean_run_does_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.observe import get_logger\n",
        )
        monkeypatch.setenv("NSL_TEST_MODE", "strict")
        config = NoStdlibLoggingConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSL_TEST_MODE",
        )
        run_no_stdlib_logging_enforcement(config)

    def test_default_mode_is_strict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.delenv("NSL_TEST_MODE", raising=False)
        config = NoStdlibLoggingConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSL_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_no_stdlib_logging_enforcement(config)


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
        monkeypatch.setenv("NSL_TEST_MODE", "strict")
        config = NoStdlibLoggingConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSL_TEST_MODE",
            exempt_files={
                "src/pkg/mod.py": ("platform bootstrap; stdlib logging used pre-observe init"),
            },
        )
        run_no_stdlib_logging_enforcement(config)


# ------------------------------------------------------------------
# exemption application — exemptions_path (line-level, parsed file)
# ------------------------------------------------------------------


class TestExemptionsPath:
    def test_exemption_file_removes_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        # the violation lives at src/pkg/mod.py:1 with symbol "*". the
        # exemption parser does not accept "*" as a symbol (the regex
        # requires an identifier), so we test with a from-import which
        # has a real identifier symbol.
        _write(
            src / "pkg" / "mod.py",
            "from logging import getLogger\n",
        )
        ex_path = repo / "_no_stdlib_logging_exemptions.txt"
        ex_path.write_text(f"# rationale: {_VALID_RATIONALE}\nsrc/pkg/mod.py:1:getLogger\n")
        monkeypatch.setenv("NSL_TEST_MODE", "strict")
        config = NoStdlibLoggingConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
            mode_env_var="NSL_TEST_MODE",
        )
        run_no_stdlib_logging_enforcement(config)

    def test_exemption_path_none_skips_loading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        monkeypatch.setenv("NSL_TEST_MODE", "strict")
        config = NoStdlibLoggingConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=None,
            mode_env_var="NSL_TEST_MODE",
        )
        run_no_stdlib_logging_enforcement(config)


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
            "from threetears.observe import get_logger\n",
        )
        monkeypatch.setenv("NSL_TEST_MODE", "strict")
        config = NoStdlibLoggingConfig(
            repo_root=repo,
            src_roots=None,
            mode_env_var="NSL_TEST_MODE",
        )
        run_no_stdlib_logging_enforcement(config)


# ------------------------------------------------------------------
# custom line marker
# ------------------------------------------------------------------


class TestCustomConfig:
    def test_custom_line_marker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # consumer using a different marker convention.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "import logging  # logging-stdlib-allowed\n",
        )
        monkeypatch.setenv("NSL_TEST_MODE", "strict")
        config = NoStdlibLoggingConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSL_TEST_MODE",
            line_marker="# logging-stdlib-allowed",
        )
        run_no_stdlib_logging_enforcement(config)
