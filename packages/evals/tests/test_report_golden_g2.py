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
    assert "max_tokens=10240" in first
    assert "NO VERDICT. The test setup failed, not the models" in first
    assert "v4-flash lost 10 of 12 trials, most because the judge's reply was truncated" in first


def test_html_line_2_is_the_exploratory_banner(g2: Report) -> None:
    lines = _header_lines(report_html(g2))
    assert lines[0].startswith("Kairo queue discipline · ")
    assert lines[1].startswith("EXPLORATORY — no question was declared before the run")


def test_markdown_line_2_is_the_exploratory_banner(g2: Report) -> None:
    lines = report_markdown(g2).splitlines()
    assert lines[0].startswith("# Kairo queue discipline · ")
    assert lines[1].startswith("EXPLORATORY — no question was declared before the run")


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
    assert markdown.count("**Chart: ") == 1
    assert "No rate is estimated or charted: no arm has 5 usable cases." in markdown


@pytest.mark.parametrize("door", ["html", "markdown"])
def test_no_samples_stub_is_printed(g2: Report, door: str) -> None:
    page = report_html(g2) if door == "html" else report_markdown(g2)
    assert "samples" not in page


@pytest.mark.parametrize("door", ["html", "markdown"])
def test_a_usable_count_is_never_printed_without_saying_usable(g2: Report, door: str) -> None:
    page = report_html(g2) if door == "html" else report_markdown(g2)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", page))
    for sentence in re.split(r"[.\n|]", text):
        if re.search(r"\b8 of 12\b", sentence):
            assert "usable" in sentence, sentence


def test_the_scoreboard_counts_every_trial_and_reads_as_a_count(g2: Report) -> None:
    markdown = report_markdown(g2)
    assert "8 of 12 trials usable (4 lost to the test setup)" in markdown
    assert "2 of 12 trials usable (10 lost to the test setup)" in markdown
    assert "10 (judge truncated 7, test setup fault 3)" in markdown


@pytest.mark.parametrize("door", ["html", "markdown"])
def test_the_contrasts_state_no_difference_on_the_one_usable_case(g2: Report, door: str) -> None:
    # The challenger kept 2 usable trials, both of one case: the contrast is read over what the arms kept.
    page = report_html(g2) if door == "html" else report_markdown(g2)
    assert "not tested (n=1)" in page


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
    assert f'<div class="disclosure" data-source="calibration">{said.text}</div>' in page
    assert "Rate about" in report_markdown(g2)
