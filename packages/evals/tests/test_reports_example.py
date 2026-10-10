"""``examples/reports.py`` writes a finished campaign's files where it is told, as a newcomer would run it.

The campaign is the example's own offline stand-ins, so its verdicts are fixed: the candidate rule triages every
ticket right and the baseline seven of twelve, which separates on accuracy, and neither reports any spend. The
files are checked for presence and substance, not byte for byte: what they say is the report's own tests' job.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
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


async def test_it_writes_the_report_its_evidence_and_its_charts_into_the_directory_it_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    verdicts = await _load().main(tmp_path)

    # Neither arm reports its spend, so cost was not measured and is not tested: accuracy is the one verdict.
    assert verdicts == {("candidate", "Accuracy"): "improved on the control"}
    for name in ("report.md", "report.html", "bundle.json"):
        assert (tmp_path / name).stat().st_size > 0, name
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Contrasts against the control" in markdown
    # The report cites the bundle written beside it.
    bundle = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["fingerprint"] in markdown

    specs = sorted((tmp_path / "charts").glob("*.vl.json"))
    assert specs and specs[0].name == "00-accuracy.vl.json"
    for path in specs:
        spec = json.loads(path.read_text(encoding="utf-8"))
        assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/") and "config" in spec
    if importlib.util.find_spec("vl_convert") is not None:
        svgs = sorted((tmp_path / "charts").glob("*.svg"))
        assert [svg.name.removesuffix(".svg") for svg in svgs] == [s.name.removesuffix(".vl.json") for s in specs]
        assert all(svg.read_text(encoding="utf-8").startswith("<svg") for svg in svgs)
        assert (tmp_path / "charts.html").stat().st_size > 0
    out = capsys.readouterr().out
    assert "\ncandidate on Accuracy: improved on the control" in out
    assert str(tmp_path.resolve()) in out


async def test_without_the_vega_extra_it_writes_the_specs_and_says_why_there_are_no_svgs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = builtins.__import__

    def no_rasteriser(name: str, *args: object, **kwargs: object) -> object:
        if name == "vl_convert":
            raise ModuleNotFoundError("No module named 'vl_convert'", name=name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, "vl_convert", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_rasteriser)
    await _load().main(tmp_path)

    assert list((tmp_path / "charts").glob("*.vl.json"))
    assert not list((tmp_path / "charts").glob("*.svg"))
    assert not (tmp_path / "charts.html").exists()
    assert 'No SVGs: install the [vega] extra (pip install "3tears-evals[vega]")' in capsys.readouterr().out


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [("reports.py", REPORTS)], consumer_root=REPO_ROOT) == []
