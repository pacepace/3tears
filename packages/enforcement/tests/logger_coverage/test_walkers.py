"""tests for the logger-coverage ``find_modules_without_logger`` walker."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.logger_coverage.walkers import (
    find_modules_without_logger,
)


_FACTORY_NAMES = frozenset({"get_logger"})
_VAR_NAMES = frozenset({"log", "_logger"})
_SKIP_BASENAMES = frozenset({"__init__.py"})


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _scan(
    src: Path,
    repo: Path,
    exempt_files: dict[str, str] | None = None,
    factory_names: frozenset[str] = _FACTORY_NAMES,
    var_names: frozenset[str] = _VAR_NAMES,
    skip_basenames: frozenset[str] = _SKIP_BASENAMES,
) -> list:
    return find_modules_without_logger(
        (src,), repo, exempt_files or {},
        factory_names, var_names, skip_basenames,
    )


# ------------------------------------------------------------------
# walker — recognises canonical module-level logger
# ------------------------------------------------------------------

class TestRecognisesLogger:
    def test_log_get_logger_no_violation(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.observe import get_logger\n"
            "log = get_logger(__name__)\n",
        )
        assert _scan(src, repo) == []

    def test_underscore_logger_alias_no_violation(
        self, tmp_path: Path,
    ) -> None:
        # legacy alias accepted by default config.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.observe import get_logger\n"
            "_logger = get_logger(__name__)\n",
        )
        assert _scan(src, repo) == []

    def test_observe_get_logger_attribute_callee_no_violation(
        self, tmp_path: Path,
    ) -> None:
        # ``log = observe.get_logger(__name__)`` — Attribute callee.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears import observe\n"
            "log = observe.get_logger(__name__)\n",
        )
        assert _scan(src, repo) == []

    def test_deeper_namespaced_attribute_callee_no_violation(
        self, tmp_path: Path,
    ) -> None:
        # ``log = threetears.observe.get_logger(__name__)`` — still
        # an Attribute callee with attr ``"get_logger"``.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "import threetears.observe\n"
            "log = threetears.observe.get_logger(__name__)\n",
        )
        assert _scan(src, repo) == []

    def test_logger_below_other_statements_still_recognised(
        self, tmp_path: Path,
    ) -> None:
        # the logger does not have to be the first statement; any
        # module-level position is accepted.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from __future__ import annotations\n"
            "from threetears.observe import get_logger\n"
            "X = 1\n"
            "log = get_logger(__name__)\n",
        )
        assert _scan(src, repo) == []


# ------------------------------------------------------------------
# walker — flags missing logger
# ------------------------------------------------------------------

class TestMissingLogger:
    def test_module_with_no_logger_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "x = 1\n")
        violations = _scan(src, repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "logger_coverage.missing"
        assert v.symbol == "src/pkg/mod.py"
        assert v.line == 1
        assert "get_logger" in v.reason
        assert "exempt_files" in v.reason

    def test_non_canonical_var_name_flagged(self, tmp_path: Path) -> None:
        # ``something = get_logger(__name__)`` is not canonical.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.observe import get_logger\n"
            "something = get_logger(__name__)\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1

    def test_unknown_factory_function_flagged(self, tmp_path: Path) -> None:
        # ``log = make_logger(__name__)`` — factory name not accepted.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from somewhere import make_logger\n"
            "log = make_logger(__name__)\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1

    def test_string_value_flagged(self, tmp_path: Path) -> None:
        # ``log = "string"`` — value is not a Call, so not a logger.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            'log = "string"\n',
        )
        violations = _scan(src, repo)
        assert len(violations) == 1

    def test_nested_logger_assignment_does_not_count(
        self, tmp_path: Path,
    ) -> None:
        # logger assignment inside a function body is not a
        # module-level assignment and should not satisfy the check.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.observe import get_logger\n"
            "def f():\n"
            "    log = get_logger(__name__)\n"
            "    return log\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1

    def test_chained_assignment_not_recognised(
        self, tmp_path: Path,
    ) -> None:
        # ``a = log = get_logger(__name__)`` — multiple targets are
        # explicitly rejected as ambiguous.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.observe import get_logger\n"
            "a = log = get_logger(__name__)\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1


# ------------------------------------------------------------------
# walker — file-level skip rules
# ------------------------------------------------------------------

class TestSkipRules:
    def test_init_py_skipped_by_default(self, tmp_path: Path) -> None:
        # __init__.py never flagged regardless of content.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "__init__.py", "x = 1\n")
        assert _scan(src, repo) == []

    def test_custom_skip_basenames_honoured(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "_compat.py", "x = 1\n")
        violations = _scan(
            src,
            repo,
            skip_basenames=frozenset({"__init__.py", "_compat.py"}),
        )
        assert violations == []

    def test_empty_file_skipped(self, tmp_path: Path) -> None:
        # zero-size file is not flagged — placeholder modules are
        # outside the contract.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "empty.py", "")
        assert _scan(src, repo) == []

    def test_exempt_file_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "models.py", "x = 1\n")
        exempt = {
            "src/pkg/models.py": (
                "pure pydantic model module, no runtime behaviour"
            ),
        }
        assert _scan(src, repo, exempt_files=exempt) == []

    def test_exempt_match_is_exact_no_globbing(
        self, tmp_path: Path,
    ) -> None:
        # listing ``src/pkg/`` does not exempt ``src/pkg/mod.py``.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "x = 1\n")
        exempt = {
            "src/pkg/": "directory-level exemption attempt — not supported",
        }
        violations = _scan(src, repo, exempt_files=exempt)
        assert len(violations) == 1


# ------------------------------------------------------------------
# walker — custom factory / var name configuration
# ------------------------------------------------------------------

class TestCustomNames:
    def test_extended_factory_set_accepts_make_logger(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from somewhere import make_logger\n"
            "log = make_logger(__name__)\n",
        )
        violations = _scan(
            src,
            repo,
            factory_names=frozenset({"get_logger", "make_logger"}),
        )
        assert violations == []

    def test_restricted_var_set_rejects_underscore_logger(
        self, tmp_path: Path,
    ) -> None:
        # only ``log`` accepted; ``_logger`` flagged.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.observe import get_logger\n"
            "_logger = get_logger(__name__)\n",
        )
        violations = _scan(
            src,
            repo,
            var_names=frozenset({"log"}),
        )
        assert len(violations) == 1


# ------------------------------------------------------------------
# walker — multi-file / multi-root behaviour
# ------------------------------------------------------------------

class TestMultipleViolations:
    def test_one_violation_per_offending_module(
        self, tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "a.py", "x = 1\n")
        _write(src / "pkg" / "b.py", "y = 2\n")
        _write(
            src / "pkg" / "c.py",
            "from threetears.observe import get_logger\n"
            "log = get_logger(__name__)\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 2
        symbols = sorted(v.symbol for v in violations)
        assert symbols == ["src/pkg/a.py", "src/pkg/b.py"]

    def test_clean_repo_no_violations(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "a.py",
            "from threetears.observe import get_logger\n"
            "log = get_logger(__name__)\n",
        )
        _write(
            src / "pkg" / "b.py",
            "from threetears import observe\n"
            "log = observe.get_logger(__name__)\n",
        )
        _write(src / "pkg" / "__init__.py", "")
        assert _scan(src, repo) == []


class TestPathDepWalking:
    def test_two_package_workspace_finds_violation(
        self, tmp_path: Path,
    ) -> None:
        # synthetic two-package workspace: A clean, B silent. with
        # both src roots passed (mimicking discover_src_roots's
        # output), the violation in B is found.
        a_src = tmp_path / "a" / "src"
        b_src = tmp_path / "b" / "src"
        _write(
            a_src / "pkg_a" / "mod.py",
            "from threetears.observe import get_logger\n"
            "log = get_logger(__name__)\n",
        )
        _write(
            b_src / "pkg_b" / "mod.py", "x = 1\n",
        )
        violations = find_modules_without_logger(
            (a_src, b_src), tmp_path, {},
            _FACTORY_NAMES, _VAR_NAMES, _SKIP_BASENAMES,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.file == b_src / "pkg_b" / "mod.py"
