"""A reading over a run that did not finish opens on what stopped it, with no verdict, on every surface (#1173).

A verdict is read first and acted on, and one over a run that stopped early is a figure over whichever cells happened
to be stored. Whether a reading may open with one is a single decision,
:func:`~threetears.evals.analysis.run_ending.stopped_short`, read off the run's stored status alone; these hold every
surface to it, for every way a run ends short:

- the decision itself, over every status a run can carry;
- the quick print (``Result.render``), one arm and several;
- the aspect's page, in Markdown and HTML;
- the campaign's page, in Markdown and HTML, under code's lead and under a caller's own;
- what a stored analysis is listed by;
- the report input a written page is built from, and the written page in Markdown, HTML and text.

The evidence of a run that finished reads as it did: its pages, its input and its line carry nothing of this.
"""

from __future__ import annotations

import dataclasses
import html
import re
from collections.abc import Mapping
from typing import Any

import pytest

from packages.evals.tests.campaign_page_support import first_screen_markdown, header_html, kairo_page, opening
from packages.evals.tests.factories import analysed_aspect, make_analysis
from packages.evals.tests.fixtures.report_inputs.dataset import kairo_bundle
from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from threetears.evals import Eval
from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignBundle,
    build_campaign_report,
    build_code_only_report,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.report.build import analysis_lead, build_code_only_report_in_forms
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.input_build import report_input
from threetears.evals.analysis.report.model import DisclosureBlock
from threetears.evals.analysis.report.verdict import bundle_stopped_short, verdict_line
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.run_ending import RunEnding, stopped_short
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.schema.models import RUN_STATUSES

#: Every way a run ends short, with how each reads after the run's name: every status but ``completed``. A run
#: interrupted and never taken up again is still ``running`` when it is read.
ENDINGS: dict[str, str] = {
    "failed": "failed",
    "exhausted": "stopped when its provider refused the account or the model",
    "cancelled": "was cancelled",
    "budget_stopped": "was stopped by its budget",
    "running": "has not finished",
    "pending": "has not started",
}

#: Why the run ended, as its record gives it: over two lines and closed with a stop, as an operator's reason can be.
REASON = "the on-call\nstopped it."
REASON_SAID = "the on-call stopped it."

NO_VERDICT = ", so there is no verdict: "

ASSEMBLED_AT = "2026-10-31T00:00:00+00:00"


def test_the_table_of_endings_is_every_status_but_completed() -> None:
    assert set(ENDINGS) == RUN_STATUSES - {"completed"}


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", sorted(RUN_STATUSES))
def test_only_a_completed_run_may_open_with_a_verdict(status: str) -> None:
    short = stopped_short([RunEnding("r1", "completed"), RunEnding("r2", status, REASON)])
    if status == "completed":
        assert short is None
        return
    assert short is not None and [ending.run_id for ending in short.endings] == ["r2"]
    assert short.sentences(lambda ending: "the run") == [f"the run {ENDINGS[status]}{NO_VERDICT}{REASON_SAID}"]


def test_a_status_this_build_has_no_words_for_is_unfinished_and_named() -> None:
    short = stopped_short([RunEnding("r1", "paused")])
    assert short is not None
    assert short.sentences(lambda ending: "the run") == ["the run ended paused, so there is no verdict"]
    assert short.covers(3) == "the readings below cover only the 3 result(s) recorded before it stopped"


# ---------------------------------------------------------------------------
# The quick print
# ---------------------------------------------------------------------------

CASES = [{"n": 1}, {"n": 2}, {"n": 3}]


async def _double(case: Mapping[str, Any]) -> int:
    return int(case["n"]) * 2


async def _triple(case: Mapping[str, Any]) -> int:
    return int(case["n"]) * 3


def _even(case: Mapping[str, Any], answer: Any) -> bool:
    return answer % 2 == 0


def _ended(result: Any, arm: str, status: str, *, reason: str | None = REASON) -> Any:
    """``result`` with one arm's run ended ``status``: a designed stop's reason is its stop's, a failure's its last
    error."""
    summary = result.arms[arm]
    failed = status in ("failed", "exhausted")
    changed = summary.model_copy(
        update={
            "status": status,
            "stopped_because": None if failed else reason,
            "errors": [*summary.errors, *([reason] if failed and reason is not None else [])],
        }
    )
    return dataclasses.replace(result, arms={**result.arms, arm: changed})


@pytest.mark.parametrize("status", sorted(ENDINGS))
async def test_the_print_of_one_arm_opens_on_what_stopped_its_run_and_prints_no_verdict(status: str) -> None:
    result = await Eval(CASES, _double, [_even], k=1, name="doubles").run(store=":memory:", scope="unfinished")
    finished = result.render().splitlines()
    assert finished[0].startswith("doubles: ") and "no verdict" not in finished[0]

    printed = _ended(result, next(iter(result.arms)), status).render().splitlines()
    first, second = printed[:2]
    assert first == f"doubles: the run {ENDINGS[status]}{NO_VERDICT}{REASON_SAID}"
    assert re.fullmatch(r"the readings below cover only the 3 result\(s\) recorded (before it \w+|so far)", second)
    assert finished[0] not in printed, "the finished run's verdict line is printed nowhere"


async def test_a_failed_run_that_recorded_no_error_still_prints_no_verdict() -> None:
    result = await Eval(CASES, _double, [_even], k=1, name="doubles").run(store=":memory:", scope="unfinished")
    verdict = result.render().splitlines()[0]
    printed = _ended(result, next(iter(result.arms)), "failed", reason=None).render().splitlines()
    assert printed[0] == "doubles: the run failed, so there is no verdict"
    assert verdict not in printed


async def test_the_print_of_a_comparison_names_the_arm_whose_run_stopped() -> None:
    compared = Eval(CASES, {"double": _double, "triple": _triple}, [_even], k=1, name="multiples")
    result = await compared.run(store=":memory:", scope="unfinished")
    verdict = result.render().splitlines()[0]
    printed = _ended(result, "triple", "budget_stopped").render().splitlines()
    assert printed[0] == f"multiples: triple was stopped by its budget{NO_VERDICT}{REASON_SAID}"
    assert printed[1] == "the readings below cover only the 6 result(s) recorded before it stopped"
    assert verdict not in printed


# ---------------------------------------------------------------------------
# The pages, over Kairo round 2's committed bundle with one candidate run ended short
# ---------------------------------------------------------------------------


def _stopped_aspect(aspect: AnalysisContextBundle, status: str, *, reason: str | None = REASON) -> tuple[Any, str]:
    """``aspect`` with its first candidate's first run ended ``status``, and that run's id."""
    run_id = aspect.design.contrasts[0].run_ids[0]
    summaries = [
        summary.model_copy(update={"ended_because": reason}) if summary.run_id == run_id else summary
        for summary in aspect.run_summaries
    ]
    return aspect.model_copy(update={"incomplete_runs": {run_id: status}, "run_summaries": summaries}), run_id


def _stopped_kairo(status: str) -> tuple[CampaignBundle, str]:
    """Kairo's campaign bundle with one candidate run ended ``status``, and what its pages open with."""
    bundle = kairo_bundle()
    aspect, run_id = _stopped_aspect(bundle.aspects[0], status)
    opens = f"qwen3.8-flash's run {run_id} {ENDINGS[status]}{NO_VERDICT}{REASON_SAID}"
    return bundle.model_copy(update={"aspects": [aspect]}), opens


def _campaign(bundle: CampaignBundle) -> EvalCampaign:
    declared = bundle.declaration
    return EvalCampaign.model_validate(
        {
            "id": bundle.campaign_id,
            "scope_id": bundle.scope_id,
            "name": bundle.name,
            "subject_id": bundle.subject_id,
            "subject_kind": bundle.subject_kind,
            "behavior": bundle.behavior,
            "created_by": declared.declared_by or "unknown",
            "question": declared.question,
            "declared_at": declared.declared_at,
            "declared_by": declared.declared_by or "",
            "arms": list(declared.arms),
            "aspects": list(declared.aspects),
            "floors": list(declared.floors),
        }
    )


def _campaign_page(bundle: CampaignBundle, **laid_out: Any) -> Any:
    labels = bundle.readings.arm_labels()
    forms = PageForms.of_aspects(bundle.aspects, bundle.readings)
    reports = {
        aspect.aspect: build_code_only_report_in_forms(
            aspect, assembled_at=ASSEMBLED_AT, arm_labels=labels, forms=forms[aspect.aspect]
        )
        for aspect in bundle.aspects
    }
    return build_campaign_report(_campaign(bundle), bundle.readings, reports, **laid_out)


def _verdict_paragraph(page_html: str) -> str:
    """The text of the page's first verdict paragraph, its markup dropped."""
    found = re.search(r'<p class="verdict"[^>]*>(.*?)</p>', page_html, flags=re.S)
    assert found is not None, "the page carries no verdict paragraph"
    return html.unescape(re.sub(r"<[^>]+>", "", found.group(1)))


def test_kairo_s_runs_all_finished_and_its_page_opens_with_its_verdict() -> None:
    bundle = kairo_bundle()
    assert all(bundle_stopped_short(aspect) is None for aspect in bundle.aspects)
    (source, text), *_ = opening(kairo_page())
    assert source == "verdict" and text.startswith("Pass rate: ") and "no verdict" not in text


@pytest.mark.parametrize("status", sorted(ENDINGS))
def test_the_aspect_s_page_opens_on_what_stopped_the_run_in_markdown_and_html(status: str) -> None:
    bundle, opens = _stopped_kairo(status)
    (aspect,) = bundle.aspects
    labels = bundle.readings.arm_labels()
    line = verdict_line(aspect, lambda key: labels.get(key, key))
    assert line.rule == "unfinished" and line.decisions == () and line.lead == opens
    assert line.text.startswith(opens) and "The readings below cover only the results recorded" in line.text

    page = build_code_only_report(aspect, assembled_at=ASSEMBLED_AT, arm_labels=labels)
    (verdict,) = [
        block
        for block in page.blocks
        if isinstance(block, DisclosureBlock) and block.section == "header" and block.source == "verdict"
    ]
    assert verdict.text.startswith(opens) and verdict.tone == "failed" and not verdict.emphasis
    bold = [line for line in report_markdown(page).splitlines() if line.startswith("**")]
    assert bold[0].startswith(f"**{opens}"), bold[0]
    assert _verdict_paragraph(report_html(page)).startswith(opens)


def test_an_arm_s_only_run_is_named_by_its_arm_alone_and_a_bundle_carrying_no_reason_says_none() -> None:
    aspect = kairo_bundle().aspects[0]
    placed = aspect.design.contrasts[0]
    run_id = placed.run_ids[0]
    one_run = placed.model_copy(update={"run_ids": [run_id]})
    design = aspect.design.model_copy(update={"contrasts": [one_run, *aspect.design.contrasts[1:]]})
    stopped, _ = _stopped_aspect(aspect.model_copy(update={"design": design}), "cancelled", reason=None)
    line = verdict_line(stopped, lambda key: "candidate" if key == placed.variant_key else "other")
    assert line.lead == "candidate's run was cancelled, so there is no verdict."


@pytest.mark.parametrize("status", sorted(ENDINGS))
def test_the_campaign_s_page_opens_on_what_stopped_the_run_in_markdown_and_html(status: str) -> None:
    bundle, opens = _stopped_kairo(status)
    page = _campaign_page(bundle)
    (source, text), *_ = opening(page)
    assert source == "verdict" and text.startswith(opens)
    # No lead on the aspect follows it: the page states no decision over the results recorded before the stop.
    assert [source for source, _ in opening(page)].count("verdict") == 1
    assert "Pass rate: " not in text
    first_screen = first_screen_markdown(page)
    assert opens in first_screen and "Pass rate: qwen3.8-flash against September" not in first_screen
    header = header_html(page)
    assert _verdict_paragraph(header).startswith(opens) and 'data-tone="failed"' in header


def test_a_caller_s_own_lead_comes_after_what_stopped_the_run() -> None:
    bundle, opens = _stopped_kairo("cancelled")
    own = DisclosureBlock(section="answer", source="verdict", text="Adopt 1017.", tone="decided")
    said = [text for source, text in opening(_campaign_page(bundle, lead=[own])) if source == "verdict"]
    assert len(said) == 2 and said[0].startswith(opens) and said[1] == "Adopt 1017."
    # Over runs that all finished the caller's lead is the page's one opening, as it was.
    finished = [text for source, text in opening(_campaign_page(kairo_bundle(), lead=[own])) if source == "verdict"]
    assert finished == ["Adopt 1017."]


@pytest.mark.parametrize("status", sorted(ENDINGS))
def test_a_stored_analysis_is_listed_by_what_stopped_the_run_before_any_caveat(status: str) -> None:
    bundle, _ = _stopped_kairo(status)
    stored = make_analysis(
        campaign_id=bundle.campaign_id,
        scope_id=bundle.scope_id,
        aspects=[analysed_aspect(aspect) for aspect in bundle.aspects],
        campaign_bundle=bundle.frame(),
        observation_refs=bundle.run_ids,
    )
    lead = analysis_lead(stored, bundle=bundle)
    assert lead.stopped and f" {ENDINGS[status]}{NO_VERDICT}{REASON_SAID}" in lead.verdict
    assert lead.caveats, "the fixture's page warns, so the order is tested"
    assert lead.line().startswith(lead.verdict) and lead.line().endswith(lead.caveats[-1])

    whole = kairo_bundle()
    finished = analysis_lead(
        make_analysis(
            campaign_id=whole.campaign_id,
            scope_id=whole.scope_id,
            aspects=[analysed_aspect(aspect) for aspect in whole.aspects],
            campaign_bundle=whole.frame(),
            observation_refs=whole.run_ids,
        ),
        bundle=whole,
    )
    assert not finished.stopped and finished.line().endswith(finished.verdict)


# ---------------------------------------------------------------------------
# The written page
# ---------------------------------------------------------------------------


def test_the_report_input_carries_what_stopped_the_run_and_nothing_where_every_run_finished() -> None:
    bundle, opens = _stopped_kairo("budget_stopped")
    (said,) = report_input(bundle).caveats.unfinished
    assert said.startswith(opens)
    finished = report_input(kairo_bundle())
    assert finished.caveats.unfinished == []
    assert "unfinished" not in finished.model_dump(mode="json", by_alias=True)["caveats"]


def test_the_written_page_says_what_stopped_the_run_before_its_headline_in_every_form() -> None:
    name = "prototype/kairo"
    doc = ReportDoc.model_validate(converted_doc(name))
    held = ReportInput.model_validate(snapshot_of(name).report_input)
    finished = build(doc, held, strict=True)
    assert finished.stopped == "" and "stopped" not in finished.model_dump(mode="json")
    assert 'data-part="stopped"' not in doc_page_html(finished)

    line = "qwen3.8-flash's run was cancelled, so there is no verdict: the on-call stopped it."
    caveats = held.caveats.model_copy(update={"unfinished": [line]})
    page = build(doc, held.model_copy(update={"caveats": caveats}), strict=True)
    assert page.stopped == line

    markdown = doc_page_markdown(page).splitlines()
    said = markdown.index(f"**{line}**")
    assert said < next(index for index, text in enumerate(markdown) if text.startswith("# "))
    text = doc_page_text(page).splitlines()
    assert text.index(line) < next(index for index, said_line in enumerate(text) if said_line.startswith("# "))
    page_html = doc_page_html(page)
    callout = page_html.index('data-part="stopped"')
    assert html.escape(line) in page_html[callout : callout + len(html.escape(line)) + 40]
    assert callout < page_html.index('class="bluf"')
