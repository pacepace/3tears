"""tests for ``run_dict_state_enforcement`` orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.common import ExemptionError
from threetears.enforcement.dict_state_detection import (
    DictStateAllowlistEntry,
    DictStateConfig,
    run_dict_state_enforcement,
)


_VALID_RATIONALE = "live LangChain ChatModel instances; non-serializable, process-local by design"


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _build_violation_repo(tmp_path: Path) -> tuple[Path, Path]:
    """build a synthetic repo with one dict-state violation.

    :return: (repo_root, src_root)
    :rtype: tuple[Path, Path]
    """
    repo = _make_repo(tmp_path / "repo")
    src = repo / "src"
    _write(
        src / "pkg" / "mod.py",
        "class Foo:\n    def __init__(self) -> None:\n        self._cache = {}\n",
    )
    return repo, src


# ------------------------------------------------------------------
# input validation
# ------------------------------------------------------------------


class TestWalkerArgValidation:
    def test_unknown_walker_raises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        config = DictStateConfig(repo_root=repo, src_roots=())
        with pytest.raises(ValueError, match="walker must be one of"):
            run_dict_state_enforcement(config, walker="bogus")

    def test_default_walker_is_all(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="DICT_TEST_MODE",
        )
        # walker="all" by default; clean source -> no work.
        run_dict_state_enforcement(config)


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
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="DICT_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_dict_state_enforcement(config, walker="detect")

    def test_report_mode_returns_silently(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.setenv("DICT_TEST_MODE", "report")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="DICT_TEST_MODE",
        )
        run_dict_state_enforcement(config, walker="detect")

    def test_clean_run_does_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._x = []\n",
        )
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="DICT_TEST_MODE",
        )
        run_dict_state_enforcement(config, walker="detect")

    def test_default_mode_is_strict(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        monkeypatch.delenv("DICT_TEST_MODE", raising=False)
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="DICT_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_dict_state_enforcement(config, walker="detect")


# ------------------------------------------------------------------
# walker selection
# ------------------------------------------------------------------


class TestWalkerSelection:
    def test_detect_only_skips_stale_check(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        pass\n",
        )
        # stale entry would surface under "all" / "allowlist_integrity",
        # but "detect" skips the integrity walker.
        stale = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=99,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            allowlist=(stale,),
            mode_env_var="DICT_TEST_MODE",
        )
        # detect only: stale entry never checked.
        run_dict_state_enforcement(config, walker="detect")

    def test_allowlist_integrity_only_skips_detection(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        # detect violation exists; integrity walker doesn't surface it.
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="DICT_TEST_MODE",
        )
        run_dict_state_enforcement(config, walker="allowlist_integrity")

    def test_all_runs_both_walkers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        # detection violation present + stale allowlist entry.
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._x = {}\n",
        )
        stale = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=99,
            attr_name="_phantom",
            rationale=_VALID_RATIONALE,
        )
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            allowlist=(stale,),
            mode_env_var="DICT_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception) as excinfo:
            run_dict_state_enforcement(config, walker="all")
        msg = str(excinfo.value)
        assert "dict_state_detection.dict_in_init" in msg
        assert "dict_state_detection.stale_allowlist" in msg


# ------------------------------------------------------------------
# allowlist integration
# ------------------------------------------------------------------


class TestAllowlistIntegration:
    def test_allowlist_silences_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        entry = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            allowlist=(entry,),
            mode_env_var="DICT_TEST_MODE",
        )
        run_dict_state_enforcement(config, walker="detect")

    def test_known_violations_silences_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        entry = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            known_violations=(entry,),
            mode_env_var="DICT_TEST_MODE",
        )
        run_dict_state_enforcement(config, walker="detect")

    def test_stale_allowlist_entry_fails_in_all_walker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        pass\n",
        )
        entry = DictStateAllowlistEntry(
            file="src/pkg/mod.py",
            line=3,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            allowlist=(entry,),
            mode_env_var="DICT_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception) as excinfo:
            run_dict_state_enforcement(config, walker="all")
        assert "stale_allowlist" in str(excinfo.value)


# ------------------------------------------------------------------
# exemption file (symmetry with sibling domains)
# ------------------------------------------------------------------


class TestExemptionFile:
    def test_exemption_file_silences_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        ex_path = repo / "_dict_state_exemptions.txt"
        ex_path.write_text(
            "# rationale: legitimately ephemeral, documented in the module docstring\nsrc/pkg/mod.py:3:_cache\n"
        )
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
            mode_env_var="DICT_TEST_MODE",
        )
        run_dict_state_enforcement(config, walker="detect")

    def test_exemption_file_missing_raises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=repo / "missing.txt",
        )
        with pytest.raises(FileNotFoundError):
            run_dict_state_enforcement(config, walker="detect")

    def test_exemption_blanket_rationale_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        repo, src = _build_violation_repo(tmp_path)
        ex_path = repo / "_dict_state_exemptions.txt"
        ex_path.write_text("# rationale: tests need this\nsrc/pkg/mod.py:3:_cache\n")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
        )
        with pytest.raises(ExemptionError):
            run_dict_state_enforcement(config, walker="detect")


# ------------------------------------------------------------------
# explicit src_roots vs discovery
# ------------------------------------------------------------------


class TestSrcRootsDiscovery:
    def test_falls_back_to_discover(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "class Foo:\n    def __init__(self) -> None:\n        self._x = []\n",
        )
        monkeypatch.setenv("DICT_TEST_MODE", "strict")
        config = DictStateConfig(
            repo_root=repo,
            src_roots=None,
            mode_env_var="DICT_TEST_MODE",
        )
        run_dict_state_enforcement(config, walker="detect")
