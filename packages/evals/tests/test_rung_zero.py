"""Rung zero holds: a product with a case list and a classifier or scorer function integrates in one short file.

``examples/rung_zero.py`` is that file. These tests execute it, read the summary it returns, and
hold it to the two properties that make it rung zero rather than merely an example: it stays under
:data:`RUNG_ZERO_MAX_LINES` lines, and it imports nothing but the engine's public roots and the
standard library — no test fixture, no third-party package, no module below a root.

Each structural check is run in both directions on the same source: the example passes it, and the
example with the defect added fails it, so neither check can pass for the reason a broken one would.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from threetears.evals.quick import EvalSummary
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
RUNG_ZERO = Path(__file__).resolve().parents[1] / "examples" / "rung_zero.py"

#: Rung zero is one file under sixty lines.
RUNG_ZERO_MAX_LINES = 59


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("rung_zero_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _too_long(source: str) -> bool:
    return len(source.splitlines()) > RUNG_ZERO_MAX_LINES


def _foreign_imports(source: str) -> list[str]:
    """Every import in ``source`` that is neither the standard library nor the engine."""
    foreign = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top not in sys.stdlib_module_names and top != "__future__" and not name.startswith("threetears.evals"):
                foreign.append(name)
    return foreign


async def test_the_example_runs_and_its_summary_reads_what_it_measured(capsys: pytest.CaptureFixture[str]) -> None:
    """Six cases, two repeats, a classifier and a scorer: the summary counts every cell and reads both."""
    summary = await _load(RUNG_ZERO).main()
    assert isinstance(summary, EvalSummary)
    assert summary.status == "completed"
    assert summary.candidate_model == "classify"
    assert (summary.n_cases, summary.k_runs, summary.n_results, summary.n_scored) == (6, 2, 12, 12)
    assert summary.n_candidate_failed == summary.n_excluded == 0
    by_name = {measure.name: measure for measure in summary.measures}
    # Two cases of six are misread: "Not bad at all." comes back neutral, and the late-but-loved one negative.
    assert by_name["match"].mean == pytest.approx(4 / 6)
    # Only the second is a false alarm: a positive review labelled negative.
    assert by_name["no_false_alarm"].mean == pytest.approx(5 / 6)
    assert by_name["match"].n == by_name["no_false_alarm"].n == by_name["confusion_cell"].n == 12
    assert [(cell.expected, cell.predicted, cell.count) for cell in summary.confusion] == [
        ("negative", "negative", 4),
        ("neutral", "neutral", 2),
        ("positive", "negative", 2),
        ("positive", "neutral", 2),
        ("positive", "positive", 2),
    ]
    labels = {statistics.label: statistics for statistics in summary.labels}
    assert sorted(labels) == ["negative", "neutral", "positive"]
    assert (labels["neutral"].precision, labels["neutral"].recall) == (0.5, 1.0)
    assert labels["positive"].precision == 1.0 and labels["positive"].recall == pytest.approx(1 / 3)
    assert labels["negative"].precision == pytest.approx(2 / 3) and labels["negative"].recall == 1.0
    assert summary.errors == []
    out = capsys.readouterr().out
    assert summary.render() in out
    assert "positive → neutral: 2" in out
    # k=2: the four neutral predictions are two cases, each answered twice, and the count says so.
    assert "neutral: precision 0.5 (2/4 over 2 cases, 95% CI" in out
    # The misses are read back, each with every reason it missed: the false alarm misses twice over.
    misses = summary.misses()
    assert [(miss.case, miss.repeat) for miss in misses] == [("4", 1), ("4", 2), ("5", 1), ("5", 2)]
    assert "'It arrived a day late, but I love it.', repeat 1: answered 'negative', expected 'positive'; " in out
    assert "no_false_alarm gave 0" in out


def test_the_example_stays_under_sixty_lines() -> None:
    """Rung zero's whole claim is its size; padding the file past the bound turns this red."""
    source = RUNG_ZERO.read_text(encoding="utf-8")
    assert not _too_long(source), f"{RUNG_ZERO.name} is {len(source.splitlines())} lines; rung zero is under 60"
    padded = source + "\n" * (RUNG_ZERO_MAX_LINES + 1 - len(source.splitlines()))
    assert _too_long(padded), "the bound admitted a 60-line file"


def test_the_example_imports_nothing_but_the_engine_and_the_standard_library() -> None:
    """No third-party package and no test fixture: a product reproduces it with the wheel alone."""
    source = RUNG_ZERO.read_text(encoding="utf-8")
    assert _foreign_imports(source) == []
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "threetears.evals.quick"
        for node in ast.walk(ast.parse(source))
    ), "the example no longer imports run_eval from its root, so the checks above say nothing about it"
    assert _foreign_imports(source + "\nimport pytest\n") == ["pytest"]
    assert _foreign_imports(source + "\nfrom packages.evals.tests import factories\n") == ["packages.evals.tests"]


def test_the_example_reaches_the_engine_only_through_public_roots(tmp_path: Path) -> None:
    """The engine names it uses are each in a public root's ``__all__``, as for every other consumer."""
    assert public_root_violations(SOURCE_ROOT, [("rung_zero.py", RUNG_ZERO)], consumer_root=REPO_ROOT) == []
    reaching = tmp_path / "rung_zero.py"
    reaching.write_text(
        RUNG_ZERO.read_text(encoding="utf-8") + "\nfrom threetears.evals.quick.one_call import CallableKind\n",
        encoding="utf-8",
    )
    assert public_root_violations(SOURCE_ROOT, [("rung_zero.py", reaching)], consumer_root=REPO_ROOT)
