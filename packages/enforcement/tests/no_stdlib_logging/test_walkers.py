"""tests for the no-stdlib-logging ``find_stdlib_logging_imports`` walker."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.no_stdlib_logging.walkers import (
    find_stdlib_logging_imports,
    is_stdlib_logging_module,
)


_LINE_MARKER = "# stdlib-logging: ok"


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
) -> list:
    return find_stdlib_logging_imports(
        (src,),
        repo,
        exempt_files or {},
        _LINE_MARKER,
    )


# ------------------------------------------------------------------
# helpers — is_stdlib_logging_module
# ------------------------------------------------------------------


class TestIsStdlibLoggingModule:
    def test_logging_root_matches(self) -> None:
        assert is_stdlib_logging_module("logging") is True

    def test_logging_submodule_matches(self) -> None:
        assert is_stdlib_logging_module("logging.handlers") is True
        assert is_stdlib_logging_module("logging.config") is True

    def test_third_party_loggish_does_not_match(self) -> None:
        assert is_stdlib_logging_module("my_logging") is False
        assert is_stdlib_logging_module("loggingx") is False
        assert is_stdlib_logging_module("logger") is False

    def test_none_does_not_match(self) -> None:
        assert is_stdlib_logging_module(None) is False

    def test_empty_does_not_match(self) -> None:
        assert is_stdlib_logging_module("") is False


# ------------------------------------------------------------------
# walker — `import logging` (Import nodes)
# ------------------------------------------------------------------


class TestImportLogging:
    def test_plain_import_logging_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import logging\n")
        violations = _scan(src, repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "no_stdlib_logging.import"
        assert v.symbol == "*"
        assert v.line == 1
        assert "import logging" in v.reason

    def test_import_logging_handlers_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import logging.handlers\n")
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "logging.handlers"

    def test_import_logging_config_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import logging.config\n")
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "logging.config"

    def test_import_logging_with_alias_flagged(self, tmp_path: Path) -> None:
        # ``import logging as stdlog`` — alias does not change what
        # was imported.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import logging as stdlog\n")
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "*"

    def test_third_party_logger_lib_not_flagged(self, tmp_path: Path) -> None:
        # only the literal ``logging`` package matters.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import logger\n")
        assert _scan(src, repo) == []

    def test_third_party_my_logging_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import my_logging\n")
        assert _scan(src, repo) == []

    def test_loggingx_not_flagged(self, tmp_path: Path) -> None:
        # name starts with ``logging`` but is not a submodule
        # (``logging.x`` would have a dot).
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import loggingx\n")
        assert _scan(src, repo) == []

    def test_combined_import_flags_only_logging(
        self,
        tmp_path: Path,
    ) -> None:
        # ``import os, logging, sys`` — only the logging alias is flagged.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import os, logging, sys\n")
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "*"


# ------------------------------------------------------------------
# walker — `from logging import ...` (ImportFrom nodes)
# ------------------------------------------------------------------


class TestFromLoggingImport:
    def test_from_logging_import_one_name_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from logging import getLogger\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.symbol == "getLogger"
        assert "from logging import getLogger" in v.reason

    def test_from_logging_import_multiple_names(
        self,
        tmp_path: Path,
    ) -> None:
        # one violation per imported name.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from logging import getLogger, INFO, DEBUG\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 3
        symbols = sorted(v.symbol for v in violations)
        assert symbols == ["DEBUG", "INFO", "getLogger"]

    def test_from_logging_import_star(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "from logging import *\n")
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].symbol == "*"

    def test_from_logging_handlers_import_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from logging.handlers import RotatingFileHandler\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.symbol == "RotatingFileHandler"
        assert "from logging.handlers import" in v.reason

    def test_from_my_logging_import_not_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from my_logging import x\n",
        )
        assert _scan(src, repo) == []

    def test_relative_from_dot_import_not_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        # ``from . import logging`` — relative import; ``node.level == 1``.
        # the walker ignores relative imports because they cannot
        # refer to the stdlib root package.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "__init__.py",
            "",
        )
        _write(
            src / "pkg" / "mod.py",
            "from . import logging\n",
        )
        assert _scan(src, repo) == []


# ------------------------------------------------------------------
# walker — per-line marker
# ------------------------------------------------------------------


class TestLineMarker:
    def test_marker_on_same_line_exempts_import(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "import logging  # stdlib-logging: ok — quiet uvicorn\n",
        )
        assert _scan(src, repo) == []

    def test_marker_on_from_import_exempts(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from logging import getLogger  # stdlib-logging: ok — third-party config\n",
        )
        assert _scan(src, repo) == []

    def test_marker_on_above_line_does_not_exempt(
        self,
        tmp_path: Path,
    ) -> None:
        # the marker must be on the offending line, not above.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "# stdlib-logging: ok — quiet uvicorn\nimport logging\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].line == 2

    def test_marker_on_one_line_does_not_exempt_other(
        self,
        tmp_path: Path,
    ) -> None:
        # per-line marker is line-scoped: line 1 is exempt, line 2 is
        # still flagged.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "import logging  # stdlib-logging: ok — quiet uvicorn\nimport logging.handlers\n",
        )
        violations = _scan(src, repo)
        assert len(violations) == 1
        assert violations[0].line == 2
        assert violations[0].symbol == "logging.handlers"


# ------------------------------------------------------------------
# walker — file-level exempt_files
# ------------------------------------------------------------------


class TestExemptFiles:
    def test_exempt_file_skipped_entirely(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "bootstrap.py",
            "import logging\nimport logging.handlers\nfrom logging import getLogger\n",
        )
        exempt = {
            "src/pkg/bootstrap.py": ("platform bootstrap; stdlib logging used pre-observe init"),
        }
        violations = _scan(src, repo, exempt_files=exempt)
        assert violations == []

    def test_exempt_match_is_exact_no_globbing(
        self,
        tmp_path: Path,
    ) -> None:
        # listing ``src/pkg/`` does not exempt ``src/pkg/bootstrap.py``;
        # the match is on the full relative posix path.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "bootstrap.py", "import logging\n")
        exempt = {
            "src/pkg/": "directory-level exemption attempt — not supported",
        }
        violations = _scan(src, repo, exempt_files=exempt)
        assert len(violations) == 1

    def test_exempt_uses_forward_slashes(self, tmp_path: Path) -> None:
        # exemption keys are forward-slash relative posix paths.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "deeply" / "nested" / "mod.py",
            "import logging\n",
        )
        exempt = {
            "src/pkg/deeply/nested/mod.py": ("transient stdlib reference for fixture-only use"),
        }
        violations = _scan(src, repo, exempt_files=exempt)
        assert violations == []


# ------------------------------------------------------------------
# walker — multiple violations and ordering
# ------------------------------------------------------------------


class TestMultipleViolations:
    def test_multiple_imports_on_different_lines(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "import logging\n"
            "import logging.handlers\n"
            "from logging import getLogger\n"
            "from logging.config import fileConfig\n",
        )
        violations = _scan(src, repo)
        # 1 + 1 + 1 + 1 = 4 violations, one per line.
        assert len(violations) == 4
        symbols = [v.symbol for v in violations]
        # source-order traversal — Import / ImportFrom both yield in
        # textual order via ast.walk.
        assert "*" in symbols
        assert "logging.handlers" in symbols
        assert "getLogger" in symbols
        assert "fileConfig" in symbols

    def test_violations_across_multiple_files(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "a.py", "import logging\n")
        _write(src / "pkg" / "b.py", "from logging import getLogger\n")
        _write(src / "pkg" / "c.py", "x = 1\n")  # clean
        violations = _scan(src, repo)
        assert len(violations) == 2
        files = sorted(str(v.file) for v in violations)
        assert files[0].endswith("a.py")
        assert files[1].endswith("b.py")


# ------------------------------------------------------------------
# walker — clean code
# ------------------------------------------------------------------


class TestCleanCode:
    def test_no_imports_no_violations(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "x = 1\n")
        assert _scan(src, repo) == []

    def test_only_threetears_observe_no_violations(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.observe import get_logger\nlog = get_logger(__name__)\n",
        )
        assert _scan(src, repo) == []


# ------------------------------------------------------------------
# walker — path-dep / multi-root behaviour
# ------------------------------------------------------------------


class TestPathDepWalking:
    def test_two_package_workspace_finds_violation(
        self,
        tmp_path: Path,
    ) -> None:
        # synthetic two-package workspace: A clean, B has a stdlib
        # logging import. with both src roots passed (mimicking
        # discover_src_roots's output), the violation in B is found.
        a_src = tmp_path / "a" / "src"
        b_src = tmp_path / "b" / "src"
        _write(
            a_src / "pkg_a" / "mod.py",
            "from threetears.observe import get_logger\n",
        )
        _write(
            b_src / "pkg_b" / "mod.py",
            "import logging\n",
        )
        violations = find_stdlib_logging_imports(
            (a_src, b_src),
            tmp_path,
            {},
            _LINE_MARKER,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.symbol == "*"
        assert v.file == b_src / "pkg_b" / "mod.py"
