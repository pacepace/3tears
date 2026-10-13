"""The headline and each summary bullet are shown at the weakest of the findings they rest on.

A finding is shown at its confidence and evidence tier, and the headline — what most readers read alone — was shown at
none, so a headline over a medium finding read as settled. The writer names the findings each rests on, and code
derives the standing (:mod:`threetears.evals.analysis.standing`): the weakest confidence among them, and the weakest
evidence tier of every reading they cite. The report serves it beside the headline and every bullet, the listing
beside the headline, and the memo the reporter judge reads beside both. An analysis stored before the links existed
says its standing was not recorded, and is never given a guessed one.

Driven over the toy campaign through :func:`~threetears.evals.analysis.generator.generate_analysis`, with two findings:
0 on latency at high confidence, 1 on cost at medium.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, alias_at, memo_payload
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.analysis.report import build_report
from threetears.evals.analysis.report.model import TextBlock
from threetears.evals.analysis.report.serialize_html import report_html
from threetears.evals.analysis.report.serialize_md import report_markdown
from threetears.evals.analysis.reporter_kind import render_memo_as_written
from threetears.evals.analysis.standing import headline_standing, summary_standings
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.ops import analyses_list
from threetears.evals.schema.models import utc_now_iso

_SUMMARY = "- The wide width is slower.\n- It also costs more, though less surely."


def _generated() -> EvalAnalysis:
    """The toy analysis: the headline on both findings, the first bullet on finding 0, the second on finding 1."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    payload = memo_payload(bundle)
    latency = payload["findings"][0]
    cost = latency | {
        "title": "the wider chunk costs more per document",
        "evidence": [
            {"cell": alias_at(bundle, level), "measure_id": "production_replicating_cost", "reading": "measure"}
            for level in (TOYHOST_NARROW, TOYHOST_WIDE)
        ],
        "confidence": "medium",
    }
    payload |= {
        "findings": [latency, cost],
        "headline_rests_on": [0, 1],
        "summary": _SUMMARY,
        "summary_rests_on": [[0], [1]],
    }
    analysis, _insights = asyncio.run(
        generate_analysis(
            bundle,
            prompt=PROMPT,
            model=MODEL,
            client=FixturedClient(json.dumps(payload)),
            prompt_id=PROMPT_ID,
            bundle_assembled_at=utc_now_iso(),
            profile=profile,
        )
    )
    return analysis


@pytest.fixture(scope="module")
def analysis() -> EvalAnalysis:
    return _generated()


def test_an_analysis_over_findings_that_names_no_link_is_refused(analysis: EvalAnalysis) -> None:
    """Over findings, the headline and each bullet name what they rest on, or the analysis is not a document."""
    document = analysis.model_dump(mode="json")
    del document["document"]["headline_rests_on"], document["document"]["summary_rests_on"]
    with pytest.raises(ValidationError, match="headline_rests_on is empty over findings"):
        EvalAnalysis.model_validate(document)


class TestTheStandingIsTheWeakestOfTheFindingsUnderIt:
    def test_the_headline_stands_at_the_weakest_confidence_of_its_findings(self, analysis: EvalAnalysis) -> None:
        standing = headline_standing(analysis)
        assert standing is not None
        assert (standing.rests_on, standing.confidence, standing.evidence_tier) == ([0, 1], "medium", "mechanical")

    def test_each_bullet_stands_on_its_own_findings(self, analysis: EvalAnalysis) -> None:
        bullets = summary_standings(analysis)
        assert bullets is not None
        assert [(bullet, s.confidence) for bullet, s in bullets] == [
            ("The wide width is slower.", "high"),
            ("It also costs more, though less surely.", "medium"),
        ]


class TestTheReportServesIt:
    def test_the_headline_block_leads_with_its_findings_and_their_weakest_tier(self, analysis: EvalAnalysis) -> None:
        report = build_report(analysis)
        headline = report.blocks[0]
        assert isinstance(headline, TextBlock) and headline.role == "headline"
        assert headline.body == analysis.document.headline and headline.rests_on_numbers == [1, 2]
        assert {fact.name: fact.value for fact in headline.facts} == {
            "Confidence": "medium",
            "Stands on": "checks and measures",
        }

    def test_the_summary_is_one_block_per_bullet_each_with_its_own_standing(self, analysis: EvalAnalysis) -> None:
        summary = [b for b in build_report(analysis).blocks if isinstance(b, TextBlock) and b.role == "summary"]
        assert [(b.body, b.rests_on, b.facts[0].value) for b in summary] == [
            ("The wide width is slower.", [0], "high"),
            ("It also costs more, though less surely.", [1], "medium"),
        ]

    def test_markdown_and_html_print_the_standing_beside_the_words(self, analysis: EvalAnalysis) -> None:
        report = build_report(analysis)
        markdown = report_markdown(report)
        assert "_The headline: Confidence: medium · Stands on: checks and measures. Rests on finding 1, 2._" in markdown
        assert (
            "- The wide width is slower. — Confidence: high · Stands on: checks and measures. Rests on finding 1."
            in (markdown)
        )
        assert '<div data-role="headline"><p class="facts">The headline: Confidence: medium' in report_html(report)


def test_the_listing_states_the_headlines_standing(analysis: EvalAnalysis) -> None:
    host = toyhost_host()
    host.storage.save_analysis(analysis)
    lines = {line.id: line for line in analyses_list(host, analysis.campaign_id, analysis.scope_id).analyses}
    assert lines[analysis.id].headline_standing == (
        "Confidence: medium · Stands on: checks and measures · rests on finding 1, 2"
    )


def test_the_memo_the_judge_reads_carries_the_standing(analysis: EvalAnalysis) -> None:
    memo = render_memo_as_written(analysis)
    assert "The headline: Confidence: medium · Stands on: checks and measures. Rests on finding 1, 2." in memo
    assert "- It also costs more, though less surely. — Confidence: medium" in memo
