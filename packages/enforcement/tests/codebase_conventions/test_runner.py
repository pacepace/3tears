"""tests for ``run_codebase_conventions_enforcement`` orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.codebase_conventions import (
    CodebaseConventionsConfig,
    run_codebase_conventions_enforcement,
)


_VALID_RATIONALE = (
    "synthetic test fixture; intentional convention violation for unit-test use only"
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _build_print_violation(tmp_path: Path) -> tuple[Path, Path]:
    repo = _make_repo(tmp_path / "repo")
    src = repo / "src"
    _write(
        src / "pkg" / "mod.py",
        "from __future__ import annotations\n"
        "def foo() -> None:\n"
        "    print('hi')\n",
    )
    return repo, src


def _build_clean_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = _make_repo(tmp_path / "repo")
    src = repo / "src"
    _write(
        src / "pkg" / "mod.py",
        "from __future__ import annotations\n"
        "def foo() -> int:\n"
        "    return 1\n",
    )
    return repo, src


# ----------------------------------------------------------------------
# input validation
# ----------------------------------------------------------------------


class TestWalkerArgValidation:
    def test_unknown_walker_raises(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        config = CodebaseConventionsConfig(repo_root=repo, src_roots=())
        with pytest.raises(ValueError, match="walker must be one of"):
            run_codebase_conventions_enforcement(config, walker="bogus")

    def test_default_walker_is_all(self, tmp_path: Path) -> None:
        repo, src = _build_clean_repo(tmp_path)
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
        )
        run_codebase_conventions_enforcement(config)

    def test_each_walker_name_accepted(self, tmp_path: Path) -> None:
        repo, src = _build_clean_repo(tmp_path)
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
        )
        for walker in (
            "print",
            "stdlib_getlogger",
            "future_annotations",
            "return_type",
            "all",
        ):
            run_codebase_conventions_enforcement(config, walker=walker)


# ----------------------------------------------------------------------
# strict / report mode behaviour
# ----------------------------------------------------------------------


class TestModes:
    def test_strict_fails_on_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_print_violation(tmp_path)
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_codebase_conventions_enforcement(config)

    def test_report_mode_returns_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_print_violation(tmp_path)
        monkeypatch.setenv("CC_TEST_MODE", "report")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
        )
        run_codebase_conventions_enforcement(config)

    def test_clean_run_does_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_clean_repo(tmp_path)
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
        )
        run_codebase_conventions_enforcement(config)

    def test_default_mode_is_strict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_print_violation(tmp_path)
        monkeypatch.delenv("CC_TEST_MODE", raising=False)
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_codebase_conventions_enforcement(config)


# ----------------------------------------------------------------------
# walker dispatching
# ----------------------------------------------------------------------


class TestWalkerDispatching:
    def test_print_walker_finds_only_print_violations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # mod has both a print violation and a missing-return-type
        # violation; choosing walker="print" should only fail on print.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "def foo():\n"
            "    print('x')\n",
        )
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
            future_annotations_exempt_files={
                "src/pkg/mod.py": _VALID_RATIONALE,
            },
            return_type_exempt_files={
                "src/pkg/mod.py": _VALID_RATIONALE,
            },
        )
        with pytest.raises(pytest.fail.Exception, match="print"):
            run_codebase_conventions_enforcement(config, walker="print")

    def test_return_type_walker_finds_only_return_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "def foo():\n"
            "    return 1\n",
        )
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
        )
        # walker="print" — print is fine, no fail.
        run_codebase_conventions_enforcement(config, walker="print")
        # walker="return_type" — fails.
        with pytest.raises(pytest.fail.Exception):
            run_codebase_conventions_enforcement(
                config, walker="return_type",
            )

    def test_future_annotations_walker_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "x = 1\n")
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_codebase_conventions_enforcement(
                config, walker="future_annotations",
            )

    def test_stdlib_getlogger_walker_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "import logging\n"
            "log = logging.getLogger(__name__)\n",
        )
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
        )
        with pytest.raises(pytest.fail.Exception):
            run_codebase_conventions_enforcement(
                config, walker="stdlib_getlogger",
            )


# ----------------------------------------------------------------------
# exemption application — exempt_files (file-level, in-walker)
# ----------------------------------------------------------------------


class TestExemptFiles:
    def test_print_exempt_file_silences(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_print_violation(tmp_path)
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
            print_exempt_files={"src/pkg/mod.py": _VALID_RATIONALE},
        )
        run_codebase_conventions_enforcement(config, walker="print")

    def test_getlogger_exempt_file_silences(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "config.py",
            "from __future__ import annotations\n"
            "import logging\n"
            "log = logging.getLogger(__name__)\n",
        )
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
            getlogger_exempt_files={"src/pkg/config.py": _VALID_RATIONALE},
        )
        run_codebase_conventions_enforcement(
            config, walker="stdlib_getlogger",
        )

    def test_future_annotations_exempt_file_silences(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "x = 1\n")
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
            future_annotations_exempt_files={
                "src/pkg/mod.py": _VALID_RATIONALE,
            },
        )
        run_codebase_conventions_enforcement(
            config, walker="future_annotations",
        )

    def test_return_type_exempt_file_silences(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "def foo():\n"
            "    return 1\n",
        )
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
            return_type_exempt_files={"src/pkg/mod.py": _VALID_RATIONALE},
        )
        run_codebase_conventions_enforcement(config, walker="return_type")


# ----------------------------------------------------------------------
# exemption application — exemptions_path (line-level, parsed file)
# ----------------------------------------------------------------------


class TestExemptionsPath:
    def test_exemption_file_removes_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_print_violation(tmp_path)
        ex_path = repo / "_codebase_conventions_exemptions.txt"
        ex_path.write_text(
            f"# rationale: {_VALID_RATIONALE}\n"
            "src/pkg/mod.py:3:print\n"
        )
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=ex_path,
            mode_env_var="CC_TEST_MODE",
        )
        run_codebase_conventions_enforcement(config, walker="print")

    def test_exemption_path_none_skips_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo, src = _build_clean_repo(tmp_path)
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            exemptions_path=None,
            mode_env_var="CC_TEST_MODE",
        )
        run_codebase_conventions_enforcement(config)


# ----------------------------------------------------------------------
# explicit src_roots vs discovery
# ----------------------------------------------------------------------


class TestSrcRootsDiscovery:
    def test_falls_back_to_discover(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "def foo() -> int:\n"
            "    return 1\n",
        )
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=None,
            mode_env_var="CC_TEST_MODE",
        )
        run_codebase_conventions_enforcement(config)


# ----------------------------------------------------------------------
# custom getlogger marker
# ----------------------------------------------------------------------


class TestCustomConfig:
    def test_custom_getlogger_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "import logging\n"
            "log = logging.getLogger(__name__)  # stdlogger-allowed\n",
        )
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
            getlogger_marker="# stdlogger-allowed",
        )
        run_codebase_conventions_enforcement(
            config, walker="stdlib_getlogger",
        )

    def test_custom_skip_basenames(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # also skip conftest.py.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "conftest.py", "x = 1\n")
        monkeypatch.setenv("CC_TEST_MODE", "strict")
        config = CodebaseConventionsConfig(
            repo_root=repo,
            src_roots=(src,),
            mode_env_var="CC_TEST_MODE",
            skip_basenames=frozenset({"__init__.py", "conftest.py"}),
        )
        run_codebase_conventions_enforcement(config)
