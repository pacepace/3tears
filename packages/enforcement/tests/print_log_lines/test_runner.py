"""tests for ``run_print_log_lines_enforcement``: modes, exemptions file, and the non-vacuity floor."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.print_log_lines import PrintLogLinesConfig, run_print_log_lines_enforcement

_ENV = "PLL_TEST_MODE"
_RATIONALE = "the bootstrap prints one line to stderr before observe is configured, by design"


def _repo(tmp_path: Path, source: str) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    src = repo / "src"
    (src / "pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    (src / "pkg" / "mod.py").write_text(source)
    return repo, src


_BAD = "import sys\nimport threetears.observe\nprint('x', file=sys.stderr)\n"
_CLEAN = "from threetears.observe import get_logger\nlog = get_logger(__name__)\n"


def test_the_defaults_scope_to_3tears_with_a_floor_of_one(tmp_path: Path) -> None:
    config = PrintLogLinesConfig(repo_root=tmp_path)
    assert config.scope_import_roots == ("threetears",)
    assert config.minimum_scoped_modules == 1
    assert config.line_marker == "# print-log: ok"
    assert config.mode_env_var == "PRINT_LOG_LINES_ENFORCEMENT_MODE"


def test_an_unknown_walker_is_refused(tmp_path: Path) -> None:
    repo, src = _repo(tmp_path, _CLEAN)
    with pytest.raises(ValueError, match="walker must be one of"):
        run_print_log_lines_enforcement(PrintLogLinesConfig(repo_root=repo, src_roots=(src,)), walker="bogus")


def test_a_clean_tree_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    repo, src = _repo(tmp_path, _CLEAN)
    run_print_log_lines_enforcement(PrintLogLinesConfig(repo_root=repo, src_roots=(src,), mode_env_var=_ENV))


def test_strict_mode_fails_on_a_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "strict")
    repo, src = _repo(tmp_path, _BAD)
    with pytest.raises(pytest.fail.Exception, match="found 1 violation"):
        run_print_log_lines_enforcement(PrintLogLinesConfig(repo_root=repo, src_roots=(src,), mode_env_var=_ENV))


def test_report_mode_reports_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_ENV, "report")
    repo, src = _repo(tmp_path, _BAD)
    run_print_log_lines_enforcement(PrintLogLinesConfig(repo_root=repo, src_roots=(src,), mode_env_var=_ENV))
    assert "print_log_lines.stderr_print" in capsys.readouterr().err


def test_an_exemptions_file_entry_clears_the_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "strict")
    repo, src = _repo(tmp_path, _BAD)
    exemptions = repo / "_print_log_lines_exemptions.txt"
    exemptions.write_text(f"# rationale: {_RATIONALE}\nsrc/pkg/mod.py:3:print\n")
    run_print_log_lines_enforcement(
        PrintLogLinesConfig(repo_root=repo, src_roots=(src,), exemptions_path=exemptions, mode_env_var=_ENV)
    )


def test_a_scan_reaching_no_3tears_module_fails_the_floor_even_in_report_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_ENV, "report")
    repo, src = _repo(tmp_path, "import os\n")
    with pytest.raises(pytest.fail.Exception, match="BROKEN SCAN"):
        run_print_log_lines_enforcement(PrintLogLinesConfig(repo_root=repo, src_roots=(src,), mode_env_var=_ENV))


def test_a_floor_of_zero_lets_an_empty_scan_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "strict")
    repo, src = _repo(tmp_path, "import os\n")
    run_print_log_lines_enforcement(
        PrintLogLinesConfig(repo_root=repo, src_roots=(src,), mode_env_var=_ENV, minimum_scoped_modules=0)
    )
