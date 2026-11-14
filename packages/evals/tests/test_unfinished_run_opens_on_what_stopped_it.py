"""A reading over a run that did not finish opens on what stopped it, with no verdict, on every surface (#1173).

A verdict is read first and acted on, and one over a run that stopped early is a figure over whichever cells happened
to be stored. Whether a reading may open with one is a single decision,
:func:`~threetears.evals.analysis.run_ending.stopped_short`, read off the run's stored status alone; these hold every
surface to it, for every way a run ends short:

- the decision itself, over every status a run can carry;
- the quick print (``Result.render``), one arm and several;
- the aspect's page, in Markdown and HTML;
- the campaign's page, in Markdown and HTML, under code's lead and under a caller's own;
- what a stored analysis is listed by, and a campaign's listing line: its headline, its short outcome and its
  short verdict, and that the line says to regenerate once the run has finished;
- the report input a written page is built from, and the written page in Markdown, HTML and text;
- ``evals gate``, which reads no verdict over such a campaign and exits 1;
- a campaign of several aspects, where only the aspect holding the run loses its verdict;
- a failed setup beside an unfinished run, which keeps the setup's own line and still states no verdict.

Where a run is ended in the store (the gate, the assembled bundle, the listing), the reason on the page is the one
the stored run records, read by the engine and never set on the page's evidence by hand.

The evidence of a run that finished reads as it did: its pages, its input and its line carry nothing of this.
"""

from __future__ import annotations

import dataclasses
import html
import re
from collections.abc import Mapping
from typing import Any

import pytest

from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.campaign_page_support import (
    aspect_fixture,
    first_screen_markdown,
    header_html,
    kairo_page,
    opening,
    page_of,
)
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, generate, ops_fixture, settled
from packages.evals.tests.factories import analysed_aspect, make_analysis
from packages.evals.tests.fixtures.report_inputs.dataset import kairo_bundle
from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from threetears.evals import Eval
from threetears.evals.analysis import (
    AnalysisContextBundle,
    CampaignBundle,
    assemble_campaign_bundle,
    build_campaign_report,
    build_code_only_report,
    campaign_report,
    campaign_report_markdown,
    campaign_stopped_short,
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
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops import campaigns_list
from threetears.evals.providers.failures import error_from_response
from threetears.evals.quick import EXIT_OK, run_cli
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
    # What follows the opening says what is true of a campaign's readings, and the remedy, in one sentence.
    recorded = "recorded so far" if status in ("running", "pending") else "recorded before the stop"
    assert line.text == (
        f"{opens} The readings below include the partial results {recorded}, so resume the run or leave it out of the "
        "campaign to read a verdict."
    )

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


# ---------------------------------------------------------------------------
# A run ended in the store: the gate, the assembled evidence, the campaign's report
# ---------------------------------------------------------------------------

GATED = [{"n": index} for index in range(48)]


def _correct(case: Mapping[str, Any], answer: str) -> bool:
    return answer == "right"


def _wrong_below(count: int) -> Any:
    async def candidate(case: Mapping[str, Any]) -> str:
        return "wrong" if case["n"] < count else "right"

    return candidate


async def _better(scope: str) -> Any:
    """A comparison whose candidate is shown better: a campaign the gate passes while every run is finished."""
    arms = {"current": _wrong_below(30), "better": _wrong_below(0)}
    return await Eval(GATED, arms, [_correct], k=1, name="current vs better").run(store=":memory:", scope=scope)


#: What a stored run records of each ending, beside its status: the field its reason is read from.
STORED: dict[str, dict[str, Any]] = {
    "failed": {"error_details": [REASON]},
    "cancelled": {"cancellation_reason": REASON},
    "budget_stopped": {"budget_stop_reason": REASON},
    "running": {},
}


def _end_stored(host: EvalHost, run_id: str, scope: str, status: str) -> None:
    """End the stored run ``status``, as its record would read; ``completed`` puts it back as a finished run."""
    run = host.storage.load_eval_run(run_id, scope)
    assert run is not None
    cleared = {"error_details": [], "cancellation_reason": None, "budget_stop_reason": None}
    host.storage.save_eval_run(run.model_copy(update={**cleared, "status": status, **STORED.get(status, {})}))


def _gate(comparison: Any) -> int:
    return run_cli(
        ["gate", comparison.campaign_id, "--scope", comparison.scope_id], host_factory=lambda: comparison.host
    )


@pytest.mark.parametrize("status", sorted(STORED))
async def test_the_gate_reads_no_verdict_over_an_unfinished_run_and_exits_1(
    status: str, capsys: pytest.CaptureFixture[str]
) -> None:
    comparison = await _better(f"gate-{status}")
    run_id = comparison.arms["better"].run_id
    assert off_loop(_gate, comparison) == EXIT_OK
    assert capsys.readouterr().out.startswith("gate passed: ")

    off_loop(_end_stored, comparison.host, run_id, comparison.scope_id, status)
    assert off_loop(_gate, comparison) == 1
    printed = capsys.readouterr().out
    reason = f": {REASON_SAID}" if STORED[status] else "."
    assert printed.startswith(f"better's run {ENDINGS[status]}, so there is no verdict{reason}"), printed
    assert "gate passed" not in printed and "gate FAILED" not in printed

    # Finished again, the campaign is gated as it was.
    off_loop(_end_stored, comparison.host, run_id, comparison.scope_id, "completed")
    assert off_loop(_gate, comparison) == EXIT_OK


@pytest.mark.parametrize("status", sorted(STORED))
async def test_the_assembled_evidence_carries_why_the_stored_run_ended_and_its_report_opens_on_it(status: str) -> None:
    comparison = await _better(f"assembled-{status}")
    host, scope, campaign_id = comparison.host, comparison.scope_id, comparison.campaign_id
    run_id = comparison.arms["better"].run_id

    def read() -> tuple[CampaignBundle, list[str], str]:
        return (
            assemble_campaign_bundle(host, campaign_id, scope),
            campaign_stopped_short(host, campaign_id, scope),
            campaign_report_markdown(campaign_report(host, campaign_id, scope)),
        )

    whole, stopped, page = off_loop(read)
    assert stopped == [] and "no verdict" not in page
    for summary in whole.aspects[0].run_summaries:
        assert summary.ended_because is None and "ended_because" not in summary.model_dump(mode="json")

    off_loop(_end_stored, host, run_id, scope, status)
    ended, stopped, page = off_loop(read)
    (aspect,) = ended.aspects
    assert aspect.incomplete_runs == {run_id: status}
    reasons = {summary.run_id: summary.ended_because for summary in aspect.run_summaries}
    assert reasons.pop(run_id) == (REASON if STORED[status] else None)
    assert set(reasons.values()) == {None}, "a finished run carries no reason"
    opens = f"better's run {ENDINGS[status]}, so there is no verdict" + (f": {REASON_SAID}" if STORED[status] else ".")
    (said,) = stopped
    assert said.startswith(opens)
    assert f"**{opens}" in page.split("\n## ", 1)[0], page[:600]


# ---------------------------------------------------------------------------
# A failed setup beside an unfinished run
# ---------------------------------------------------------------------------

_OVERLOADED = '{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}'


async def test_a_run_that_failed_measuring_nothing_keeps_the_setup_s_line_on_every_page_and_states_no_verdict() -> None:
    error = error_from_response("anthropic", 529, _OVERLOADED)

    async def declined(case: Mapping[str, Any]) -> int:
        raise error

    result = await Eval(CASES, declined, [_even], k=1, name="declined").run(store=":memory:", scope="setup-failed")
    assert result.summary.status == "failed" and bundle_stopped_short(result.report.bundle) is not None
    setup = "The test setup failed or the provider declined every call, so every trial was lost."
    # The print opens on the run's failure; the aspect's page and the campaign's keep the setup's line, which names
    # what lost the trials and the remedy. Neither states a verdict, and the campaign's page adds no lead of its own.
    assert result.render().splitlines()[0].startswith("declined: the run failed, so there is no verdict: ")
    assert verdict_line(result.report.bundle, lambda key: "declined").rule == "setup"
    said = [text for source, text in opening(result.page) if source == "verdict"]
    assert len(said) == 1 and said[0].startswith(setup), said
    assert "passed" not in said[0]


# ---------------------------------------------------------------------------
# Several aspects: only the aspect holding the run loses its verdict
# ---------------------------------------------------------------------------


def _stop_billing(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
    return _stopped_aspect(bundle, "cancelled")[0] if bundle.aspect == "billing" else bundle


async def test_a_campaign_of_two_aspects_opens_on_the_stopped_one_and_keeps_the_other_s_lead() -> None:
    whole = [text for source, text in opening(await page_of("billing", "tone")) if source == "verdict"]
    assert len(whole) == 1 and "on billing" in whole[0] and "on tone" in whole[0]

    stopped, lead = [
        text for source, text in opening(await page_of("billing", "tone", bundle=_stop_billing)) if source == "verdict"
    ]
    assert stopped.startswith("On billing: ") and f" was cancelled{NO_VERDICT}{REASON_SAID}" in stopped
    assert "on tone" in lead and "on billing" not in lead, "the aspect that finished keeps its lead; the other has none"


async def test_two_aspects_are_listed_and_read_into_the_report_input_by_the_stopped_one_first() -> None:
    billing = (await aspect_fixture("billing")).bundle
    # Each fixture aspect was measured as a campaign of its own; here they are one campaign's.
    tone = (await aspect_fixture("tone")).bundle.model_copy(update={"campaign_id": billing.campaign_id})
    bundle = campaign_bundle(_stop_billing(billing), tone)
    stored = make_analysis(
        campaign_id=bundle.campaign_id,
        scope_id=bundle.scope_id,
        aspects=[analysed_aspect(aspect) for aspect in bundle.aspects],
        campaign_bundle=bundle.frame(),
        observation_refs=bundle.run_ids,
    )
    lead = analysis_lead(stored, bundle=bundle)
    assert lead.stopped
    assert lead.verdict.startswith("billing: ") and f" was cancelled{NO_VERDICT}{REASON_SAID}" in lead.verdict
    # The aspect that finished is still listed by what code read on it, after the stopped one; billing by no reading.
    after = lead.verdict.split("to read a verdict.", 1)[1]
    assert after.strip().startswith("tone: ") and "billing: " not in after

    (said,) = report_input(bundle).caveats.unfinished
    assert said.startswith("On billing: ") and f" was cancelled{NO_VERDICT}{REASON_SAID}" in said
    assert report_input(campaign_bundle(billing, tone)).caveats.unfinished == []


# ---------------------------------------------------------------------------
# A campaign's listing line
# ---------------------------------------------------------------------------


async def test_a_report_written_over_an_unfinished_run_is_listed_by_what_stopped_it_until_the_run_finishes() -> None:
    fixture = off_loop(ops_fixture)
    host = fixture.host.eval_host
    run_id = fixture.campaign.run_ids[-1]
    off_loop(_end_stored, host, run_id, TOYHOST_SCOPE, "cancelled")
    (job,) = (await generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    status = await settled(fixture.host, job.job_id)
    await run_blocking(
        host.blocking_executor, _listed_then_finished, host, fixture.campaign.id, status.analysis_id, run_id
    )


def _listed_then_finished(host: EvalHost, campaign_id: str, analysis_id: str | None, run_id: str) -> None:
    analysis = host.storage.load_reported_analysis(campaign_id, TOYHOST_SCOPE)
    assert analysis is not None and analysis.id == analysis_id
    # The report recommends an arm, so a short verdict read off it would say to adopt or keep one.
    arm = analysis.aspects[0].bundle["scoreboard"][0]["variant_key"]
    host.storage.save_analysis(analysis.model_copy(update={"recommended_variant_key": arm}))

    def line() -> Any:
        (found,) = [each for each in campaigns_list(host, TOYHOST_SCOPE).campaigns if each.id == campaign_id]
        return found

    listed = line()
    assert listed.report is not None and listed.report_state == "written"
    assert f" was cancelled{NO_VERDICT}{REASON_SAID}" in listed.report.headline
    assert listed.outcome_short == "Did not finish"
    assert listed.verdict_short is not None and listed.verdict_short.removeprefix("Draft: ") == "No call"

    # The run is taken up again and completes: the line no longer says it did not finish, and says to regenerate.
    _end_stored(host, run_id, TOYHOST_SCOPE, "completed")
    finished = line()
    assert finished.report is not None and finished.report_state == "needs_regeneration"
    assert finished.outcome_short is None
    assert finished.report.why_regenerate is not None and "has since finished" in finished.report.why_regenerate
    assert finished.report.headline.startswith("Needs regeneration: ")
    assert "was cancelled" not in finished.report.headline
