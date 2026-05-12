"""tests for the nats-wrapper-usage walkers.

covers both the production walker (:func:`find_direct_nats_imports`)
and the test-tree walker (:func:`find_test_nats_imports`), plus the
``is_forbidden_module`` helper.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.nats_wrapper_usage.walkers import (
    find_direct_nats_imports,
    find_test_nats_imports,
    is_forbidden_module,
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


# ------------------------------------------------------------------
# helpers — is_forbidden_module
# ------------------------------------------------------------------


class TestIsForbiddenModule:
    def test_root_matches(self) -> None:
        assert is_forbidden_module("nats", "nats") is True

    def test_submodule_matches(self) -> None:
        assert is_forbidden_module("nats.aio", "nats") is True
        assert is_forbidden_module("nats.errors", "nats") is True
        assert is_forbidden_module("nats.js.api", "nats") is True

    def test_substring_overlap_does_not_match(self) -> None:
        # only the literal package root counts.
        assert is_forbidden_module("natural", "nats") is False
        assert is_forbidden_module("natalie", "nats") is False
        assert is_forbidden_module("natsumi", "nats") is False

    def test_none_does_not_match(self) -> None:
        assert is_forbidden_module(None, "nats") is False

    def test_empty_does_not_match(self) -> None:
        assert is_forbidden_module("", "nats") is False

    def test_custom_forbidden_module(self) -> None:
        assert is_forbidden_module("kafka", "kafka") is True
        assert is_forbidden_module("kafka.admin", "kafka") is True
        assert is_forbidden_module("nats", "kafka") is False


# ------------------------------------------------------------------
# production walker — Import nodes
# ------------------------------------------------------------------


class TestProductionImport:
    def test_plain_import_nats_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import nats\n")
        violations = find_direct_nats_imports((src,), repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "nats_wrapper_usage.production_import"
        assert v.symbol == "*"
        assert v.line == 1
        assert "import nats" in v.reason

    def test_import_nats_aio_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import nats.aio\n")
        violations = find_direct_nats_imports((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "nats.aio"

    def test_import_nats_errors_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import nats.errors\n")
        violations = find_direct_nats_imports((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "nats.errors"

    def test_import_nats_with_alias_flagged(self, tmp_path: Path) -> None:
        # alias does not change what was imported.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import nats as natz\n")
        violations = find_direct_nats_imports((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "*"

    def test_import_natural_not_flagged(self, tmp_path: Path) -> None:
        # substring overlap is not a match.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import natural\n")
        assert find_direct_nats_imports((src,), repo) == []

    def test_import_natsumi_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import natsumi\n")
        assert find_direct_nats_imports((src,), repo) == []

    def test_combined_import_flags_only_nats(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import os, nats, sys\n")
        violations = find_direct_nats_imports((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "*"


# ------------------------------------------------------------------
# production walker — ImportFrom nodes
# ------------------------------------------------------------------


class TestProductionImportFrom:
    def test_from_nats_import_one_name_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "from nats import connect\n")
        violations = find_direct_nats_imports((src,), repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.symbol == "connect"
        assert "from nats import connect" in v.reason

    def test_from_nats_import_multiple_names(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from nats import connect, NATS, JetStream\n",
        )
        violations = find_direct_nats_imports((src,), repo)
        assert len(violations) == 3
        symbols = sorted(v.symbol for v in violations)
        assert symbols == ["JetStream", "NATS", "connect"]

    def test_from_nats_import_star(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "from nats import *\n")
        violations = find_direct_nats_imports((src,), repo)
        assert len(violations) == 1
        assert violations[0].symbol == "*"

    def test_from_nats_aio_import_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from nats.aio import Client\n",
        )
        violations = find_direct_nats_imports((src,), repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.symbol == "Client"
        assert "from nats.aio import" in v.reason

    def test_from_natalie_import_not_flagged(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "from natalie import x\n")
        assert find_direct_nats_imports((src,), repo) == []

    def test_from_threetears_nats_import_not_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        # the wrapper itself.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.nats import NatsClient\n",
        )
        assert find_direct_nats_imports((src,), repo) == []

    def test_relative_from_dot_import_not_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        # relative imports cannot reach the top-level nats package.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "__init__.py", "")
        _write(src / "pkg" / "mod.py", "from . import nats\n")
        assert find_direct_nats_imports((src,), repo) == []


# ------------------------------------------------------------------
# multi-file production scan
# ------------------------------------------------------------------


class TestProductionMultiFile:
    def test_violations_across_multiple_files(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "a.py", "import nats\n")
        _write(src / "pkg" / "b.py", "from nats.aio import Client\n")
        _write(src / "pkg" / "c.py", "from threetears.nats import NatsClient\n")
        violations = find_direct_nats_imports((src,), repo)
        assert len(violations) == 2
        files = sorted(str(v.file) for v in violations)
        assert files[0].endswith("a.py")
        assert files[1].endswith("b.py")

    def test_clean_file_no_violations(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "pkg" / "mod.py",
            "from threetears.nats import NatsClient, Subjects\nclient = NatsClient()\n",
        )
        assert find_direct_nats_imports((src,), repo) == []


# ------------------------------------------------------------------
# production / tests separation
# ------------------------------------------------------------------


class TestProductionVsTestsSeparation:
    def test_production_walker_does_not_scan_tests(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        tests = repo / "tests"
        _write(src / "pkg" / "mod.py", "from threetears.nats import NatsClient\n")
        _write(tests / "test_x.py", "import nats\n")
        # production walker only sees the src tree.
        assert find_direct_nats_imports((src,), repo) == []

    def test_tests_walker_does_not_scan_src(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        tests = repo / "tests"
        _write(src / "pkg" / "mod.py", "import nats\n")
        _write(tests / "test_x.py", "from threetears.nats import NatsClient\n")
        # tests walker only sees the tests tree.
        assert find_test_nats_imports(tests, repo) == []


# ------------------------------------------------------------------
# tests walker — same matched shapes, distinct category
# ------------------------------------------------------------------


class TestTestsWalker:
    def test_tests_walker_emits_test_import_category(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        tests = repo / "tests"
        _write(tests / "test_x.py", "import nats\n")
        violations = find_test_nats_imports(tests, repo)
        assert len(violations) == 1
        v = violations[0]
        assert v.category == "nats_wrapper_usage.test_import"
        assert v.symbol == "*"

    def test_tests_walker_with_none_returns_empty(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        assert find_test_nats_imports(None, repo) == []

    def test_tests_walker_missing_directory_returns_empty(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        # tests dir does not exist on disk
        assert find_test_nats_imports(repo / "tests", repo) == []

    def test_tests_walker_finds_from_imports(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        tests = repo / "tests"
        _write(
            tests / "test_x.py",
            "from nats.aio import Client\n",
        )
        violations = find_test_nats_imports(tests, repo)
        assert len(violations) == 1
        assert violations[0].category == "nats_wrapper_usage.test_import"
        assert violations[0].symbol == "Client"


# ------------------------------------------------------------------
# custom forbidden module
# ------------------------------------------------------------------


class TestCustomForbiddenModule:
    def test_custom_forbidden_module_flags_kafka(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import kafka\n")
        violations = find_direct_nats_imports(
            (src,),
            repo,
            forbidden_module="kafka",
        )
        assert len(violations) == 1
        assert violations[0].symbol == "*"

    def test_custom_forbidden_module_does_not_flag_nats(
        self,
        tmp_path: Path,
    ) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "pkg" / "mod.py", "import nats\n")
        # forbidden_module="kafka" — nats is allowed.
        violations = find_direct_nats_imports(
            (src,),
            repo,
            forbidden_module="kafka",
        )
        assert violations == []


# ------------------------------------------------------------------
# multi-root walking (simulates discover_src_roots output)
# ------------------------------------------------------------------


class TestPathDepWalking:
    def test_two_package_workspace_finds_violation(
        self,
        tmp_path: Path,
    ) -> None:
        a_src = tmp_path / "a" / "src"
        b_src = tmp_path / "b" / "src"
        _write(
            a_src / "pkg_a" / "mod.py",
            "from threetears.nats import NatsClient\n",
        )
        _write(b_src / "pkg_b" / "mod.py", "import nats\n")
        violations = find_direct_nats_imports((a_src, b_src), tmp_path)
        assert len(violations) == 1
        v = violations[0]
        assert v.symbol == "*"
        assert v.file == b_src / "pkg_b" / "mod.py"
