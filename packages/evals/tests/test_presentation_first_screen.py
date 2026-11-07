"""A campaign's first screen says what the page is, never what it is not (presentation build, chunk 1).

The owner's readings of 2026-11-06 (pages B and D) and the review's A2 and A6 hold the first screen to five things,
each held here on the ladder's replayed pages, rungs 0 to 4, and the G2 fixtures:

- every page has an answer block, a lead, whether or not anything warns (W2, P1);
- no first-screen line outside what warns and the report-wide caveats opens or turns on an absence
  (:func:`~packages.evals.tests.style_support.absence_violations`; P6, H5, H12);
- a rate the first screen shows carries the count it is computed over, in its line or its cell (H11);
- every report-wide caveat code assigns lands on the first screen, once (A2's caveat clause);
- rung 2's uncalibrated judge is one of them (P2), its detail one step down in Design & provenance.

A caller that states its own answer (a harness reading a figure of its own) hands it in as the page's lead, which
replaces code's and changes nothing else. :func:`absence_violations` has its own direct tests at the end.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from threetears.evals.analysis import build_campaign_report, campaign_report_html, campaign_report_markdown
from threetears.evals.analysis.report.campaign.provenance import caveat_blocks
from threetears.evals.analysis.report.model import (
    FIRST_SCREEN,
    CampaignReport,
    DisclosureBlock,
    Report,
    ReportBlock,
    TableBlock,
)
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.quick.files import load_evals

from packages.evals.tests.g2_support import g2_report, g2_ruled_report
from packages.evals.tests.ladder_support import load, offline_copy
from packages.evals.tests.report_shapes_support import s4_change
from packages.evals.tests.style_support import absence_violations, reading_names


@dataclass(frozen=True)
class _Rung:
    """One rung replayed offline: its campaign's page and its aspect's own page."""

    page: CampaignReport
    report: Report


#: Each rung: its directory, the file declaring its eval, and the module attribute that holds it (None for TOML).
_RUNGS: dict[str, tuple[str, str, str | None]] = {
    "rung0": ("1-triage", "triage.toml", None),
    "rung1": ("1-triage", "triage.py", "triage_eval"),
    "rung2": ("2-replies", "replies.py", "replies_eval"),
    "rung3": ("3-invoices", "extract.py", "invoices"),
    "rung4": ("4-booking", "agent.py", "booking"),
}


@pytest.fixture(scope="module")
def rungs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, _Rung]]:
    """Every rung replayed once, from its recordings, with no key."""
    replayed: dict[str, _Rung] = {}
    for name, (directory, file, attribute) in _RUNGS.items():
        with pytest.MonkeyPatch.context() as monkeypatch:
            copy = offline_copy(directory, tmp_path_factory.mktemp(name), monkeypatch)
            if attribute is None:
                (runnable,) = load_evals(copy / file)
            else:
                runnable = getattr(load(copy / file, Path(file).stem, monkeypatch), attribute)
            result = asyncio.run(runnable.run(store=":memory:"))
            replayed[name] = _Rung(page=result.page, report=result.report)
    yield replayed


def _first_screen(blocks: list[ReportBlock]) -> list[ReportBlock]:
    return [block for block in blocks if block.section in FIRST_SCREEN]


def _said(blocks: list[ReportBlock]) -> list[DisclosureBlock]:
    return [block for block in _first_screen(blocks) if isinstance(block, DisclosureBlock)]


# --- every page has an answer block ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_every_rung_s_campaign_page_has_a_lead(rungs: dict[str, _Rung], name: str) -> None:
    page = rungs[name].page
    (lead,) = [block for block in _said(page.blocks) if block.source == "verdict"]
    assert lead.section == "answer" and lead.text.strip()
    # The lead sits under the question, above what warns and the caveats.
    order = [block for block in page.blocks if block.section in ("answer", "warnings")]
    assert order.index(lead) == next(index for index, block in enumerate(order) if block.section == "answer") + (
        1 if any(isinstance(block, DisclosureBlock) and block.source == "basis" for block in order) else 0
    )


def test_one_arm_measured_leads_on_its_pass_rate_as_a_count(rungs: dict[str, _Rung]) -> None:
    (lead,) = [block for block in _said(rungs["rung2"].page.blocks) if block.source == "verdict"]
    assert lead.text == "reply passed 40 of 40 trials on 20 cases."


async def test_both_g2_fixtures_have_an_answer_block() -> None:
    for report in (await g2_report(), (await g2_ruled_report())[1]):
        assert [block for block in _said(report.blocks) if block.source == "verdict" and block.text.strip()]


# --- no first-screen line states an absence ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_no_first_screen_line_of_a_rung_states_an_absence(rungs: dict[str, _Rung], name: str) -> None:
    # Rung 3's undecided reading says what it means ("the evidence cannot tell v2 and v1 apart"), never "not separated".
    page = rungs[name].page
    names = reading_names(fold.report.bundle for fold in page.aspects)
    assert absence_violations(_first_screen(page.blocks), names=names) == []


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_no_first_screen_line_of_a_rung_s_aspect_page_states_an_absence(rungs: dict[str, _Rung], name: str) -> None:
    # The aspect page's verdict line is the campaign's lead sentence, and its guardrail what it means for the decision.
    report = rungs[name].report
    assert absence_violations(_first_screen(report.blocks), names=reading_names([report.bundle])) == []


async def test_no_first_screen_line_of_either_g2_fixture_states_an_absence() -> None:
    # The setup gate says what the run measured ("The test setup failed: it measured the setup instead of the
    # models."), never "No verdict".
    for report in (await g2_report(), (await g2_ruled_report())[1]):
        assert absence_violations(_first_screen(report.blocks), names=reading_names([report.bundle])) == []


# --- a first-screen rate carries its count ----------------------------------------------------------------------------

_RATE = re.compile(r"\d%")
_COUNT = re.compile(r"\b\d+ of \d+\b")


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_every_first_screen_rate_carries_its_count(rungs: dict[str, _Rung], name: str) -> None:
    page = rungs[name].page
    for block in _said(page.blocks):
        if _RATE.search(block.text) and block.section != "warnings":
            assert _COUNT.search(block.text), block.text
    # The scorecard's matrix, which a desktop's first screen shows under the caveats: each rate cell its count.
    (matrix,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "scorecard"]
    rates = [column.key for column in matrix.columns if column.header.endswith("(%)")]
    assert rates, "every rung's matrix holds a rate"
    for row in matrix.rows:
        for key in rates:
            if row[key].value is not None:
                assert _COUNT.search(row[key].text), row[key].text


# --- every report-wide caveat lands on the first screen ---------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_every_report_wide_caveat_lands_on_the_first_screen_once(rungs: dict[str, _Rung], name: str) -> None:
    rung = rungs[name]
    bundles = [fold.report.bundle for fold in rung.page.aspects]
    for page, section in ((rung.page, "warnings"), (rung.report, "header")):
        assigned = [block.anchor for block in caveat_blocks(bundles, section=section)]
        landed = [block.anchor for block in _said(page.blocks) if block.anchor in assigned]
        assert landed == assigned
    html = campaign_report_html(rung.page)
    header = html[html.index("<header>") : html.index("</header>")]
    for anchor in (block.anchor for block in caveat_blocks(bundles, section="warnings")):
        assert header.count(f'id="{anchor}"') == 1 and html.count(f'id="{anchor}"') == 1


def test_rung_2_s_first_screen_carries_its_judge_caveat_and_its_detail_one_step_down(rungs: dict[str, _Rung]) -> None:
    page = rungs["rung2"].page
    (caveat,) = [block for block in _said(page.blocks) if block.anchor == "caveat-judge"]
    assert caveat.section == "warnings"
    assert caveat.text == "Judged scores on score are directional until people's labels calibrate the judge."
    (setup,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "test_setup"]
    (row,) = [row for row in setup.rows if row["setup"].text == "judge scores directional"]
    assert row["detail"].text.startswith("Judge google/gemini-2.5-flash-lite is uncalibrated on ")
    assert re.search(r"Rate about \d+ of its outputs to calibrate it\.$", row["detail"].text)
    # The fold says neither again: the campaign states both once.
    for fold in page.aspects:
        assert not [block for block in fold.report.blocks if block.anchor in ("caveat-judge", "setup-judge")]
    # The aspect's own page carries the same line, its detail under Not tested.
    own = rungs["rung2"].report
    assert [block.text for block in _said(own.blocks) if block.anchor == "caveat-judge"] == [caveat.text]
    assert [block.section for block in own.blocks if block.anchor == "setup-judge"] == ["not_tested"]


# --- a caller's own answer replaces code's lead -----------------------------------------------------------------------


def test_a_lead_the_caller_supplies_replaces_code_s_in_place_and_the_rest_of_the_page_is_byte_identical() -> None:
    shape = s4_change()
    computed = build_campaign_report(shape.campaign, shape.readings, shape.reports)
    given = DisclosureBlock(
        section="answer",
        source="verdict",
        text="Undecided, so production stays: 10 cases cannot tell v2 apart from it.",
        tone="open",
    )
    supplied = build_campaign_report(shape.campaign, shape.readings, shape.reports, lead=[given])
    (code_lead,) = [
        block for block in computed.blocks if isinstance(block, DisclosureBlock) and block.source == "verdict"
    ]
    index = computed.blocks.index(code_lead)
    # Where code's lead stood, directly after the question line, the caller's block stands, as given, its tone kept.
    assert supplied.blocks[index] == given and supplied.blocks[index - 1].source == "basis"
    assert (
        supplied.blocks[:index] + supplied.blocks[index + 1 :] == computed.blocks[:index] + computed.blocks[index + 1 :]
    )
    # Rendered in code's tone, linking code's term, the page differs only by the lead's words: every other byte is the
    # same.
    alike = given.model_copy(update={"tone": code_lead.tone, "term": code_lead.term})
    same_tone = build_campaign_report(shape.campaign, shape.readings, shape.reports, lead=[alike])
    for render in (campaign_report_markdown, campaign_report_html):
        before = render(computed)
        after = render(same_tone)
        assert before.count(code_lead.text) == 1 and after.count(given.text) == 1
        assert after.replace(given.text, code_lead.text) == before


def test_a_lead_outside_the_answer_or_empty_is_refused() -> None:
    shape = s4_change()
    stray = DisclosureBlock(section="warnings", source="verdict", text="Production stays.")
    with pytest.raises(RefusedInput, match="not the answer"):
        build_campaign_report(shape.campaign, shape.readings, shape.reports, lead=[stray])
    with pytest.raises(RefusedInput, match="no block at all"):
        build_campaign_report(shape.campaign, shape.readings, shape.reports, lead=[])


# --- absence_violations, directly ------------------------------------------------------------------------------------


def _line(text: str, *, section: str = "answer", source: str = "memo", anchor: str | None = None) -> DisclosureBlock:
    return DisclosureBlock.model_validate({"section": section, "source": source, "text": text, "anchor": anchor})


@pytest.mark.parametrize(
    ("text", "detail"),
    [
        ("No recommendation was written: no analysis has been generated over this campaign.", "No recommendation"),
        ("No question was declared, so every reading is a lead.", "No question"),
        ("One arm is measured, so nothing is compared.", "nothing is"),
        ("Accuracy: v2 and v1 not separated.", "not separated"),
        ("The control never ran there.", "never ran"),
        ("Of the floors, none held.", "none held"),
        ("Nothing was declared.", "Nothing was"),
    ],
)
def test_a_line_opening_or_turning_on_an_absence_is_one_violation(text: str, detail: str) -> None:
    (found,) = absence_violations([_line(text)])
    assert (found.text, found.detail) == (text, detail)


def test_a_line_saying_what_the_page_is_passes() -> None:
    lines = [
        _line("Exploratory comparison.", source="comparisons"),
        _line("reply passed 40 of 40 trials on 20 cases.", source="verdict"),
        _line("Notes on the arm are in its fold; another arm cannot be compared.", source="verdict"),
    ]
    assert absence_violations(lines) == []


def test_a_page_s_own_reading_name_states_no_absence_and_a_stray_no_still_does() -> None:
    # A guardrail named "No invented total score" is a name the page prints, masked as code is; a "No" anywhere else
    # in the line still opens an absence.
    named = _line("v2 may do worse than v1 on the No invented total score guardrail.", source="verdict")
    assert absence_violations([named], names=["No invented total score"]) == []
    (found,) = absence_violations([named])
    assert found.detail == "No invented"
    (stray,) = absence_violations([_line("v2 made No call.", source="verdict")], names=["No invented total score"])
    assert stray.detail == "No call"


def test_what_warns_the_caveats_and_the_quoted_question_may_state_an_absence() -> None:
    lines = [
        _line("v2 is not yet shown safe on the guardrail.", section="warnings", source="guardrails"),
        _line(
            "Request settings were not confirmed as sent.", section="header", source="measurement", anchor="caveat-x"
        ),
        _line("Under 5 cases no gap is estimated.", section="scoreboard", source="caveat"),
        _line("No floor holds on billing.", section="header", source="floor"),
        _line('"Does v2 never invent a total?"', source="basis"),
    ]
    assert absence_violations(lines) == []


def test_a_block_with_no_words_and_a_code_span_hold_no_absence() -> None:
    (table,) = [
        block for block in s4_change().page().blocks if isinstance(block, TableBlock) and block.name == "scorecard"
    ]
    assert absence_violations([table, _line("Launch with `measure_latency=None` to read it clean.")]) == []
