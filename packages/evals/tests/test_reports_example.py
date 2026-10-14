"""``examples/reports.py`` writes a finished campaign's files where it is told, as a newcomer would run it.

The campaign is the example's own offline stand-ins, so its verdict is fixed: the candidate rule triages every ticket
right and the baseline two of seven. The files are checked for presence and substance, not byte for byte: what they
say is the report's own tests' job.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from threetears.evals.analysis import AnalysisContextBundle
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
REPORTS = Path(__file__).resolve().parents[1] / "examples" / "reports.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reports_example", REPORTS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_it_writes_the_page_its_markdown_and_its_bundle_into_the_directory_it_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = await _load().main(tmp_path)

    assert sorted(path.name for path in files.values()) == ["bundle.json", "report.html", "report.md"]
    assert all(path.parent == tmp_path and path.stat().st_size > 0 for path in files.values())
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<script" not in html and "<svg" in html
    # The page names the fingerprint of the bundle written beside it.
    bundle = AnalysisContextBundle.from_json((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    assert bundle.fingerprint() in html
    assert "Each arm against the control" in (tmp_path / "report.md").read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "gate passed: all 1 verdict(s) decided" in out
    assert str(tmp_path) in out


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [("reports.py", REPORTS)], consumer_root=REPO_ROOT) == []
