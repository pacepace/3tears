"""Every door into a campaign's report returns the same page: one builder, one disclosure set, no door adding a block.

The quick path's ``Result.page``, the command line's ``report`` and the ``report_read`` operation behind the MCP
action all read :func:`~threetears.evals.analysis.campaign_report`, the campaign's page, whose one fold is the
aspect's page :func:`~threetears.evals.analysis.report.build_code_only_report` lays out — so with the clock fixed they
return byte-identical HTML, and the same disclosures: the three the quick path once added on its own (a judge
grading its own model's answers, no margin to be equivalent on, a guardrail not shown held) are the builder's now, so
the command line and an agent read them too. ``CampaignReport.write`` puts the page, its Markdown, its JSON and each
aspect's bundle in one directory, and the page names each bundle's fingerprint.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis import (
    AnalysisContextBundle,
    DisclosureBlock,
    TableBlock,
    campaign_report,
    campaign_report_html,
    campaign_report_markdown,
)
from threetears.evals.analysis import service as analysis_service
from threetears.evals.ops import report_read
from threetears.evals import Eval, Result
from threetears.evals.quick import run_cli
from packages.evals.tests.off_loop import off_loop

#: The instant every door's report is assembled at, so their pages can be compared byte for byte.
NOW = "2026-10-14T09:00:00+00:00"

CASES = [{"text": f"refund {n}", "label": "billing"} for n in range(4)] + [
    {"text": f"refund, it crashed {n}", "label": "billing"} for n in range(4)
]


async def _careful(case: Mapping[str, Any]) -> str:
    return "billing" if "refund" in case["text"] else "bug"


async def _hasty(case: Mapping[str, Any]) -> str:
    return "bug" if "crash" in case["text"] else "billing"


@pytest.fixture
async def comparison(monkeypatch: pytest.MonkeyPatch) -> Result:
    monkeypatch.setattr(analysis_service, "utc_now_iso", lambda: NOW)
    ev = Eval(CASES, {"hasty": _hasty, "careful": _careful}, ["label"], name="doors")
    return await ev.run(store=":memory:")


def _disclosures(page_blocks: list[Any]) -> set[str]:
    return {block.text for block in page_blocks if isinstance(block, DisclosureBlock)}


async def test_compare_the_service_and_report_read_return_one_page_byte_for_byte(comparison: Result) -> None:
    host = comparison.host
    by_service = off_loop(campaign_report, host, comparison.campaign_id, comparison.scope_id)
    by_operation = off_loop(report_read, host, comparison.campaign_id, comparison.scope_id, format="html").body
    assert campaign_report_html(comparison.page) == campaign_report_html(by_service) == by_operation
    assert campaign_report_markdown(comparison.page) == campaign_report_markdown(by_service)
    assert _disclosures(comparison.page.blocks) == _disclosures(by_service.blocks)
    assert _disclosures(comparison.page.aspects[0].report.blocks) == _disclosures(by_service.aspects[0].report.blocks)


async def test_the_command_line_prints_the_same_page(comparison: Result, capsys: pytest.CaptureFixture[str]) -> None:
    host = comparison.host
    capsys.readouterr()
    args = ["report", comparison.campaign_id, "--scope", comparison.scope_id, "--format", "html"]
    assert off_loop(run_cli, args, host_factory=lambda: host) == 0
    assert capsys.readouterr().out == campaign_report_html(comparison.page)


async def test_report_read_json_names_the_bundle_the_page_was_computed_from(comparison: Result) -> None:
    body = off_loop(report_read, comparison.host, comparison.campaign_id, comparison.scope_id, format="json").body
    document = json.loads(body)
    assert (document["kind"], document["basis"], document["analysis_id"]) == ("campaign", "code_only", None)
    (aspect,) = document["aspects"]
    assert aspect["bundle_fingerprint"] == comparison.report.bundle.fingerprint()
    assert aspect["bundle_schema_version"] == comparison.report.bundle.schema_version


async def test_write_puts_one_self_contained_page_its_markdown_and_its_bundle_in_one_directory(
    comparison: Result, tmp_path: Path
) -> None:
    files = comparison.write(tmp_path / "out")
    (aspect,) = comparison.page.aspects
    (fold_id,) = comparison.page.fold_ids
    assert {role: path.name for role, path in files.items()} == {
        "html": "report.html",
        "markdown": "report.md",
        "json": "campaign.json",
        f"bundle:{aspect.aspect}": f"bundle-{fold_id}.json",
    }
    page = files["html"].read_text(encoding="utf-8")
    # A host serves the page in a sandboxed frame under a strict content policy: it reaches nothing outside itself.
    assert not re.search(r"<(script|link|img|iframe|object|embed|video|audio|source)\b", page, re.I)
    assert not re.search(r"<[^>]*\s(on[a-z]+|src|srcset|action|formaction)\s*=", page, re.I), "no handler, no fetch"
    hrefs = re.findall(r"<[^>]*\shref\s*=\s*\"([^\"]*)\"", page, re.I)
    assert hrefs and all(ref.startswith("#") for ref in hrefs), "the only links are anchors within the page"
    assert not re.search(r"https?://|//[a-z0-9.-]+\.[a-z]{2,}/", page, re.I), "no absolute or protocol-relative URL"
    assert not re.search(r"url\(|@import|@font-face", page, re.I), "no stylesheet fetch and no web font"
    (style,) = re.findall(r"<style>(.*?)</style>", page, re.S)
    assert "--font-sans: system-ui," in style, "the system's own fonts, named once as a variable"
    fonts = re.findall(r"\bfont(?:-family)?:\s*([^;}]+)", style)
    assert fonts and all("var(--font-" in declared for declared in fonts), fonts
    bundle = AnalysisContextBundle.from_json(files[f"bundle:{aspect.aspect}"].read_text(encoding="utf-8"))
    assert bundle.fingerprint() in page
    assert files["markdown"].read_text(encoding="utf-8") == campaign_report_markdown(comparison.page)


async def test_the_command_line_writes_the_directory(
    comparison: Result, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = comparison.host
    args = ["report", comparison.campaign_id, "--scope", comparison.scope_id, "--out", str(tmp_path / "cli")]
    assert off_loop(run_cli, args, host_factory=lambda: host) == 0
    assert (tmp_path / "cli" / "report.html").read_text(encoding="utf-8") == campaign_report_html(comparison.page)
    (fold_id,) = comparison.page.fold_ids
    assert f"report.html, report.md, campaign.json, bundle-{fold_id}.json" in capsys.readouterr().out


async def test_compare_s_disclosures_are_the_builder_s(comparison: Result) -> None:
    """A measure with no margin is said on the page whichever door built it: the builder says it, not the quick path."""
    by_service = off_loop(campaign_report, comparison.host, comparison.campaign_id, comparison.scope_id)
    (fold,) = by_service.aspects
    said = [text for text in _disclosures(fold.report.blocks) if text.startswith("No margin is declared")]
    assert len(said) == 1
    assert "not separated never means the arms are alike" in said[0]
    (table,) = [b for b in fold.report.blocks if isinstance(b, TableBlock) and b.name == "no_margin"]
    assert [row["reading"] for row in table.values()].count("Accuracy") == 1
