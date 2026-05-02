"""tests for ``run_nats_enforcement`` orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.common import ExemptionError
from threetears.enforcement.nats_wrapper_usage import (
    NatsWrapperConfig,
    run_nats_enforcement,
)


_VALID_RATIONALE = (
    "integration test against live nats containers; wrapper migration "
    "deferred to a follow-up shard"
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _build_production_violation_repo(tmp_path: Path) -> tuple[Path, Path]:
    """build a synthetic repo with one direct-nats import in production.

    :return: (repo_root, src_root)
    :rtype: tuple[Path, Path]
    """
    repo = _make_repo(tmp_path / "repo")
    src = repo / "src"
    _write(src / "pkg" / "mod.py", "import nats\n")
    return repo, src


def _build_test_violation_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """build a synthetic repo with one direct-nats import in tests.

    :return: (repo_root, src_root, tests_root)
    :rtype: tuple[Path, Path, Path]
    """
    repo = _make_repo(tmp_path / "repo")
    src = repo / "src"
    tests = repo / "tests"
    _write(src / "pkg" / "mod.py", "from threetears.nats import NatsClient\n")
    _write(tests / "test_x.py", "import nats\n")
    return repo, src, tests


# ------------------------------------------------------------------
# input validation
# ------------------------------------------------------------------

class TestWalkerArgValidation:
    def test_unknown_walker_raises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        config = NatsWrapperConfig(repo_root=repo, src_roots=())
        with pytest.raises(ValueError, match="walker must be one of"):
            run_nats_enforcement(config, walker="bogus")

    def test_default_walker_is_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NATS_TEST_MODE",
        )
        # walker="all" by default; both walkers no-op on empty input.
        run_nats_enforcement(config)


# ------------------------------------------------------------------
# strict / report mode behaviour
# ------------------------------------------------------------------

class TestModes:
    def test_strict_fails_on_production_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_production_violation_repo(tmp_path)
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NATS_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_nats_enforcement(config, walker="production")

    def test_report_mode_returns_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_production_violation_repo(tmp_path)
        monkeypatch.setenv("NATS_TEST_MODE", "report")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NATS_TEST_MODE",
        )
        # must not raise even though a violation exists.
        run_nats_enforcement(config, walker="production")

    def test_clean_run_does_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.nats import NatsClient\n",
        )
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NATS_TEST_MODE",
        )
        run_nats_enforcement(config, walker="production")

    def test_default_mode_is_strict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_production_violation_repo(tmp_path)
        monkeypatch.delenv("NATS_TEST_MODE", raising=False)
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="NATS_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_nats_enforcement(config, walker="production")


# ------------------------------------------------------------------
# walker selection
# ------------------------------------------------------------------

class TestWalkerSelection:
    def test_production_only_skips_tests(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src, tests = _build_test_violation_repo(tmp_path)
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            tests_root=tests,
            mode_env_var="NATS_TEST_MODE",
        )
        # production-only: ignores the tests violation.
        run_nats_enforcement(config, walker="production")

    def test_tests_only_skips_production(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_production_violation_repo(tmp_path)
        # production violation exists; no tests tree configured.
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            tests_root=None,
            mode_env_var="NATS_TEST_MODE",
        )
        # tests-only: production is ignored, no tests root => no work.
        run_nats_enforcement(config, walker="tests")

    def test_all_runs_both_walkers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # production has a violation; tests has a different violation.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        tests = repo / "tests"
        _write(src / "pkg" / "p.py", "import nats\n")
        _write(tests / "test_t.py", "from nats.aio import Client\n")
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            tests_root=tests,
            mode_env_var="NATS_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception) as excinfo:
            run_nats_enforcement(config, walker="all")
        # both categories appear in the failure report.
        msg = str(excinfo.value)
        assert "nats_wrapper_usage.production_import" in msg
        assert "nats_wrapper_usage.test_import" in msg

    def test_tests_only_strict_fails_on_test_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src, tests = _build_test_violation_repo(tmp_path)
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            tests_root=tests,
            mode_env_var="NATS_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_nats_enforcement(config, walker="tests")


# ------------------------------------------------------------------
# exemption file handling
# ------------------------------------------------------------------

class TestExemptionsFile:
    def test_path_exemption_silences_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_production_violation_repo(tmp_path)
        ex_path = repo / "_nats_exemptions.txt"
        ex_path.write_text(
            f"# rationale: {_VALID_RATIONALE}\n"
            "src/pkg/mod.py\n"
        )
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
            mode_env_var="NATS_TEST_MODE",
        )
        run_nats_enforcement(config, walker="production")

    def test_path_exemption_only_silences_listed_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "exempt.py", "import nats\n")
        _write(src / "pkg" / "violator.py", "import nats\n")
        ex_path = repo / "_nats_exemptions.txt"
        ex_path.write_text(
            f"# rationale: {_VALID_RATIONALE}\n"
            "src/pkg/exempt.py\n"
        )
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
            mode_env_var="NATS_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception) as excinfo:
            run_nats_enforcement(config, walker="production")
        msg = str(excinfo.value)
        assert "violator.py" in msg
        assert "exempt.py" not in msg

    def test_exemption_path_none_skips_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=None,
            mode_env_var="NATS_TEST_MODE",
        )
        run_nats_enforcement(config, walker="production")

    def test_exemption_file_missing_raises(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "ok.py", "")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=repo / "missing.txt",
        )
        with pytest.raises(FileNotFoundError):
            run_nats_enforcement(config, walker="production")

    def test_exemption_entry_without_rationale_raises(
        self, tmp_path: Path,
    ) -> None:
        repo, src = _build_production_violation_repo(tmp_path)
        ex_path = repo / "_nats_exemptions.txt"
        ex_path.write_text("src/pkg/mod.py\n")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
        )
        with pytest.raises(ExemptionError, match="no preceding"):
            run_nats_enforcement(config, walker="production")

    def test_exemption_blanket_rationale_rejected(
        self, tmp_path: Path,
    ) -> None:
        repo, src = _build_production_violation_repo(tmp_path)
        ex_path = repo / "_nats_exemptions.txt"
        ex_path.write_text(
            "# rationale: tests need this\n"
            "src/pkg/mod.py\n"
        )
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
        )
        # rejected because the rationale is < 30 chars and a blanket phrase.
        with pytest.raises(ExemptionError):
            run_nats_enforcement(config, walker="production")

    def test_exemption_short_rationale_rejected(
        self, tmp_path: Path,
    ) -> None:
        repo, src = _build_production_violation_repo(tmp_path)
        ex_path = repo / "_nats_exemptions.txt"
        ex_path.write_text(
            "# rationale: too short\n"
            "src/pkg/mod.py\n"
        )
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
        )
        with pytest.raises(
            ExemptionError, match="rationale must be at least",
        ):
            run_nats_enforcement(config, walker="production")

    def test_exemption_empty_rationale_rejected(
        self, tmp_path: Path,
    ) -> None:
        repo, src = _build_production_violation_repo(tmp_path)
        ex_path = repo / "_nats_exemptions.txt"
        ex_path.write_text(
            "# rationale:\n"
            "src/pkg/mod.py\n"
        )
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
        )
        with pytest.raises(
            ExemptionError, match="non-empty reason",
        ):
            run_nats_enforcement(config, walker="production")

    def test_exemption_with_other_comments_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # non-rationale comments and blank lines are allowed.
        repo, src = _build_production_violation_repo(tmp_path)
        ex_path = repo / "_nats_exemptions.txt"
        ex_path.write_text(
            "# header comment about nats exemptions\n"
            "\n"
            f"# rationale: {_VALID_RATIONALE}\n"
            "src/pkg/mod.py\n"
        )
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
            mode_env_var="NATS_TEST_MODE",
        )
        run_nats_enforcement(config, walker="production")


# ------------------------------------------------------------------
# explicit src_roots vs discovery
# ------------------------------------------------------------------

class TestSrcRootsDiscovery:
    def test_falls_back_to_discover(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # build a clean repo with src/, no path-deps.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.nats import NatsClient\n",
        )
        monkeypatch.setenv("NATS_TEST_MODE", "strict")
        config = NatsWrapperConfig(
            repo_root=repo,
            src_roots=None,
            mode_env_var="NATS_TEST_MODE",
        )
        run_nats_enforcement(config, walker="production")
