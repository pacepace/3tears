"""``examples/reports.py`` writes a finished campaign's files where it is told, as a newcomer would run it.

The campaign is the example's own offline stand-ins, so its verdicts are fixed: the candidate rule triages every
ticket right and the baseline seven of twelve, which separates on accuracy, and neither reports any spend. The
files are checked for presence and substance, not byte for byte: what they say is the report's own tests' job.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
REPORTS = Path(__file__).resolve().parents[1] / "examples" / "reports.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reports_example", REPORTS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_it_writes_the_report_and_its_evidence_into_the_directory_it_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    verdicts = await _load().main(tmp_path)

    # Neither arm reports its spend, so cost was not measured and is not tested: accuracy is the one verdict.
    assert verdicts == {("more-keywords", "Accuracy"): "improved"}
    for name in ("report.md", "report.html", "bundle.json"):
        assert (tmp_path / name).stat().st_size > 0, name
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Contrasts against the control" in markdown
    # The report cites the bundle written beside it.
    bundle = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["fingerprint"] in markdown

    out = capsys.readouterr().out
    assert (
        "\nmore-keywords on Accuracy: improved (improved on the control)\ngate passed: all 1 verdict(s) decided" in out
    )
    assert str(tmp_path.resolve()) in out


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [("reports.py", REPORTS)], consumer_root=REPO_ROOT) == []
