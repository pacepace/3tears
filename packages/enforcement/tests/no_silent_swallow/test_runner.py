"""tests for ``run_no_silent_swallow_enforcement`` orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.no_silent_swallow import (
    NoSilentSwallowConfig,
    run_no_silent_swallow_enforcement,
)


_VALID_RATIONALE = (
    "intentional silent swallow under controlled fixture for unit-test use only"
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _build_violation_repo(tmp_path: Path) -> tuple[Path, Path]:
    """build a synthetic repo with one silent-swallow violation.

    :return: (repo_root, src_root)
    :rtype: tuple[Path, Path]
    """
    repo = _make_repo(tmp_path / "repo")
    src = repo / "src"
    _write(
        src / "pkg" / "mod.py",
        "def f():\n"
        "    try:\n"
        "        do()\n"
        "    except ValueError:\n"
        "        pass\n",
    )
    return repo, src


# ------------------------------------------------------------------
# input validation
# ------------------------------------------------------------------

class TestWalkerArgValidation:
    def test_unknown_walker_raises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        config = NoSilentSwallowConfig(repo_root=repo, src_roots=())
        with pytest.raises(ValueError, match="walker must be one of"):
            run_no_silent_swallow_enforcement(config, walker="bogus")

    def test_default_walker_is_all(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSS_TEST_MODE",
        )
        run_no_silent_swallow_enforcement(config)


# ------------------------------------------------------------------
# strict / report mode behaviour
# ------------------------------------------------------------------

class TestModes:
    def test_strict_fails_on_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.setenv("NSS_TEST_MODE", "strict")
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSS_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_no_silent_swallow_enforcement(config)

    def test_report_mode_returns_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.setenv("NSS_TEST_MODE", "report")
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSS_TEST_MODE",
        )
        # must not raise even though violations exist.
        run_no_silent_swallow_enforcement(config)

    def test_clean_run_does_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        log.error('x')\n",
        )
        monkeypatch.setenv("NSS_TEST_MODE", "strict")
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSS_TEST_MODE",
        )
        run_no_silent_swallow_enforcement(config)

    def test_default_mode_is_strict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.delenv("NSS_TEST_MODE", raising=False)
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSS_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_no_silent_swallow_enforcement(config)


# ------------------------------------------------------------------
# exemption application
# ------------------------------------------------------------------

class TestExemptions:
    def test_exemption_removes_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        # the violation lives at src/pkg/mod.py:4 with symbol ValueError.
        ex_path = repo / "_no_silent_swallow_exemptions.txt"
        ex_path.write_text(
            f"# rationale: {_VALID_RATIONALE}\n"
            "src/pkg/mod.py:4:ValueError\n"
        )
        monkeypatch.setenv("NSS_TEST_MODE", "strict")
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
            mode_env_var="NSS_TEST_MODE",
        )
        run_no_silent_swallow_enforcement(config)

    def test_exemption_path_none_skips_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        monkeypatch.setenv("NSS_TEST_MODE", "strict")
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=None,
            mode_env_var="NSS_TEST_MODE",
        )
        run_no_silent_swallow_enforcement(config)


# ------------------------------------------------------------------
# explicit src_roots vs discovery
# ------------------------------------------------------------------

class TestSrcRootsDiscovery:
    def test_falls_back_to_discover(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # build a clean repo with src/, no path-deps; with
        # src_roots=None the runner should call discover_src_roots
        # and find the src tree itself.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        log.error('x')\n",
        )
        monkeypatch.setenv("NSS_TEST_MODE", "strict")
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=None,
            mode_env_var="NSS_TEST_MODE",
        )
        run_no_silent_swallow_enforcement(config)


# ------------------------------------------------------------------
# custom logger_names / nosilent_marker
# ------------------------------------------------------------------

class TestCustomConfig:
    def test_custom_logger_names_recognised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # a repo using ``slog`` as the receiver name should pass with
        # an extended logger_names set.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        slog.error('x')\n",
        )
        monkeypatch.setenv("NSS_TEST_MODE", "strict")
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSS_TEST_MODE",
            logger_names=frozenset({"slog", "log", "logger"}),
        )
        run_no_silent_swallow_enforcement(config)

    def test_default_logger_names_does_not_match_slog(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # without the override, ``slog`` is not a recognised logger.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:\n"
            "        return None\n",
        )
        monkeypatch.setenv("NSS_TEST_MODE", "strict")
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSS_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_no_silent_swallow_enforcement(config)

    def test_custom_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # consumer using a different marker convention.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def f():\n"
            "    try:\n"
            "        do()\n"
            "    except ValueError:  # SILENT-OK: shutdown path\n"
            "        pass\n",
        )
        monkeypatch.setenv("NSS_TEST_MODE", "strict")
        config = NoSilentSwallowConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NSS_TEST_MODE",
            nosilent_marker="# SILENT-OK:",
        )
        run_no_silent_swallow_enforcement(config)
