"""The report a G2-shaped campaign gets: what the failed production memo should have said, on its first screen.

The G2 campaign (2026-10) compared two models over 4 cases × 3 repeats. The judge's reply was cut off at its output
cap on most of the challenger's trials, so the control kept 8 of its 12 trials and the challenger 2 of its 12, and
no question was declared. Its memo headlined "Keep the current model — adopted", printed the control as the arm
table's "winner", printed twenty chart blocks of "4 samples" stubs and a percent change on two trials, and said
"excluded" where it meant that the test setup had failed — and the word "truncated" appeared nowhere on its first
screen.

Its judge was new, with no human labels — as every judge on a first campaign is — so the page says so on every
report: as a state with its next step (rate outputs to calibrate it), set apart from the losses, never as a failure.

The fixture (:mod:`g2_support`) is a real campaign through the quick path, two arms over 4 cases × 3, whose stored
results are then given G2's losses — a judge reply cut off at the cap it was sent (``judge_cut_short``, as the
judge's cut-short path records it) and test-setup faults — and whose bundle is assembled again over them, so the
page reads the assembler's own ``exclusions`` (``truncated`` with ``truncated_sent_caps``), scoreboard and trials.
Both doors' output, HTML and Markdown, are held to the same lines.
"""

from __future__ import annotations

import html
import re

import pytest

from threetears.evals.analysis import DisclosureBlock, Report, report_html, report_markdown
from threetears.evals.analysis.report.model import FIRST_SCREEN
from packages.evals.tests.g2_support import JUDGE, g2_report

#: Words that would make a judge's calibration state read as something that broke.
FAILURE_WORDS = ("fail", "error", "broken", "invalid")


@pytest.fixture
async def g2() -> Report:
    return await g2_report()


def _header_lines(page: str) -> list[str]:
    """The HTML page's first screen, as the lines a reader sees: each element of its ``<header>`` one line."""
    header = page[page.index("<header>") : page.index("</header>")]
    text = re.sub(r"<(svg|table)\b.*?</\1>", "", header, flags=re.S)
    pieces = re.split(r"</(?:h1|p|div|figcaption)>", text)
    lines = [html.unescape(re.sub(r"<[^>]+>", "", line)).strip() for line in pieces]
    return [line for line in lines if line]


def _first_screen_md(markdown: str) -> str:
    """The Markdown's first screen: everything before the first section heading."""
    return markdown.split("\n## ", 1)[0]


@pytest.mark.parametrize("door", ["html", "markdown"])
def test_the_first_screen_says_the_judge_was_truncated_and_at_which_cap(g2: Report, door: str) -> None:
    first = "\n".join(_header_lines(report_html(g2))) if door == "html" else _first_screen_md(report_markdown(g2))
    assert "truncated" in first
    assert "10240 tokens" in first and "max_tokens=" not in first
    assert "No verdict: the test setup failed, not the models." in first
    assert "v4-flash lost 10 of 12 trials, most because the judge's reply was truncated" in first


#: The first screen's opening lines, golden: the question as the title, then each arm named once with the control
#: marked, and the date.
TITLE = "Kairo queue discipline: does v4-flash do better than v4.1-flash?"
ARMS = "v4.1-flash (control) vs v4-flash · 14 Oct 2026"
SCOREBOARD_HEADER = "| Arm | Ran | Lost | Usable | Passed | Failed by check | Failed by judge | Cost per reply |"


def test_html_opens_on_the_question_then_the_arms(g2: Report) -> None:
    lines = _header_lines(report_html(g2))
    assert lines[:2] == [TITLE, ARMS]
    assert lines[2].startswith("No verdict: the test setup failed")


def test_markdown_opens_on_the_question_then_the_arms_then_the_verdict(g2: Report) -> None:
    lines = report_markdown(g2).splitlines()
    assert lines[:4] == [f"# {TITLE}", ARMS, "", lines[3]] and lines[3].startswith("**No verdict: ")
    assert lines[5:8] == ["**Scoreboard**", "", SCOREBOARD_HEADER]


@pytest.mark.parametrize("door", ["html", "markdown"])
def test_no_arm_is_a_winner_and_no_change_is_a_percent(g2: Report, door: str) -> None:
    page = report_html(g2) if door == "html" else report_markdown(g2)
    assert "winner" not in page.lower()
    assert not re.search(r"[+-]\d+(\.\d+)?%", page), "a percent change"
    assert "change vs" not in page


def test_exactly_one_chart_is_drawn_and_it_is_the_outcome_bar(g2: Report) -> None:
    page = report_html(g2)
    assert page.count("<svg") == 1
    assert page.count('data-chart="outcome_bar"') == 1
    markdown = report_markdown(g2)
    assert markdown.count("**Outcomes**") == 1
    # The case floor is a caveat under the scoreboard, in normal weight, after the result.
    first = _first_screen_md(markdown)
    caveat = "Too few cases to estimate a rate. Usable cases: v4.1-flash 3, v4-flash 1; a rate needs 5."
    assert caveat in first.splitlines() and f"**{caveat}" not in first
    assert first.index(SCOREBOARD_HEADER) < first.index(caveat)


@pytest.mark.parametrize("door", ["html", "markdown"])
def test_no_samples_stub_is_printed(g2: Report, door: str) -> None:
    page = report_html(g2) if door == "html" else report_markdown(g2)
    assert "samples" not in page


def test_the_scoreboard_is_a_bare_table_of_counts(g2: Report) -> None:
    markdown = report_markdown(g2)
    assert "| ▸ v4.1-flash | 12 | 4 | 8 | 8 | 0 | 0 |" in markdown
    assert "| v4-flash | 12 | 10 | 2 | 2 | 0 | 0 |" in markdown
    assert "lost: judge truncated" in markdown and "lost: setup fault" in markdown, "the causes, in the bar"


@pytest.mark.parametrize("door", ["html", "markdown"])
def test_the_contrasts_collapse_to_one_line_under_the_case_floor(g2: Report, door: str) -> None:
    # The challenger kept 2 usable trials, both of one case: no contrast can resolve, and none is a row.
    page = report_html(g2) if door == "html" else report_markdown(g2)
    assert "No contrast can resolve: a side has 1 usable case, and a difference needs 5." in page
    assert 'data-table="comparisons"' not in page and "| Reading | Arm | Change |" not in page


def test_both_doors_say_max_tokens(g2: Report) -> None:
    assert "max_tokens" in report_html(g2)
    assert "max_tokens" in report_markdown(g2)


def _calibration(report: Report) -> list[DisclosureBlock]:
    return [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "calibration"]


def test_an_unlabelled_judge_is_a_state_with_its_next_step_never_a_failure(g2: Report) -> None:
    (said,) = _calibration(g2)
    assert said.text.startswith(f"Judge {JUDGE} is uncalibrated on ")
    assert "no human labels yet" in said.text and "directional" in said.text
    assert re.search(r"Rate about \d+ of its outputs to calibrate it\.$", said.text), "it names the next step"
    assert not [word for word in FAILURE_WORDS if word in said.text.lower()]


def test_an_unlabelled_judge_is_said_apart_from_the_losses_and_in_no_alarm_colour(g2: Report) -> None:
    (said,) = _calibration(g2)
    assert said.section not in FIRST_SCREEN | {"scoreboard"}, "never beside what the test setup lost"
    page = report_html(g2)
    assert f'<p class="disclosure" data-source="calibration">{html.escape(said.text, quote=True)}' in page
    assert "Rate about" in report_markdown(g2)
