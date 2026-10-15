"""A report's first two screens speak a reader's words, and every way a trial is lost or fails has words and a remedy.

A trial is passed, failed (the candidate's failure) or lost (the test setup's). The engine's nouns for the same
things — a cell, an observation, an exclusion, the rig, the harness, the apparatus, a reserved judge axis's stored
id — never reach the first screen or what-it-means, and no arm is called a winner. Held over three reports: a G2-shaped
code-only report, a classifier comparison that separated, and the toy host's analysis report.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, get_args

import pytest

from threetears.evals.analysis import Report, report_html, report_markdown
from threetears.evals.analysis.report.model import FIRST_SCREEN
from threetears.evals.analysis.report.verdict import FAIL_CAUSE_WORDS, LOSS_CAUSE_WORDS, fail_words, loss_words
from threetears.evals.analysis.report.words import FIRST_SCREEN_BANNED
from threetears.evals.analysis.bundle.schema import FailCause
from threetears.evals.kernel.result_condition import CandidateFailureCause, ExclusionCause
from threetears.evals import Eval
from packages.evals.tests.report_support import toy_report
from packages.evals.tests.g2_support import g2_report


@pytest.fixture
async def g2() -> Report:
    return await g2_report()


#: The sections a reader reads before deciding to read on: the first screen, and what it means.
_SCREENS_ONE_AND_TWO = FIRST_SCREEN | {"memo", "contrasts"}


def _screens_one_and_two(report: Report) -> str:
    """The Markdown of the first two screens' blocks alone, title included."""
    kept = report.model_copy(update={"blocks": [b for b in report.blocks if b.section in _SCREENS_ONE_AND_TWO]})
    return report_markdown(kept)


def _banned(text: str) -> list[str]:
    """Every banned word the text uses, matched whole (and plural), case-blind."""
    return [word for word in FIRST_SCREEN_BANNED if re.search(rf"(?<![\w-]){re.escape(word)}s?(?![\w-])", text, re.I)]


async def _classify(case: Mapping[str, Any]) -> str:
    return "billing" if "refund" in case["text"] or "charged" in case["text"] else "bug"


async def _hasty(case: Mapping[str, Any]) -> str:
    return "bug" if "crash" in case["text"] else "billing"


async def test_a_g2_report_s_first_two_screens_use_none_of_the_engine_s_nouns(g2: Report) -> None:
    assert _banned(_screens_one_and_two(g2)) == []


async def test_a_comparison_s_first_two_screens_use_none_of_the_engine_s_nouns() -> None:
    cases = [{"text": f"refund {n}", "label": "billing"} for n in range(5)] + [
        {"text": f"refund crash {n}", "label": "billing"} for n in range(5)
    ]
    comparison = await Eval(cases, {"hasty": _hasty, "careful": _classify}, ["label"]).run(store=":memory:")
    text = _screens_one_and_two(comparison.report)
    assert _banned(text) == []
    assert "improved on the control" in text


async def test_an_analysis_report_s_first_two_screens_use_none_of_the_engine_s_nouns() -> None:
    _, analysis, report = await toy_report()
    text = _screens_one_and_two(report)
    assert _banned(text) == []
    # The memo's findings are on screen two, each by its headline, unnumbered.
    assert "Written by" not in text and analysis.document.findings[0].title in text and "F1." not in text


async def test_no_page_calls_any_arm_a_winner(g2: Report) -> None:
    _, _, report = await toy_report()
    for page in (report_html(g2), report_markdown(g2), report_html(report), report_markdown(report)):
        assert "winner" not in page.lower()


@pytest.mark.parametrize("cause", get_args(ExclusionCause))
def test_every_way_a_trial_is_lost_has_words_a_short_name_and_a_remedy(cause: str) -> None:
    worded = loss_words(cause)
    assert worded.words and worded.remedy and worded.short
    assert "{" not in worded.words + worded.remedy, "every placeholder is filled"


@pytest.mark.parametrize("cause", [*get_args(CandidateFailureCause), *get_args(FailCause)])
def test_every_way_a_trial_fails_has_words_a_short_name_and_a_remedy(cause: str) -> None:
    worded = fail_words(cause)
    assert worded.words and worded.remedy and worded.short


def test_the_word_tables_cover_exactly_their_causes() -> None:
    assert set(LOSS_CAUSE_WORDS) == set(get_args(ExclusionCause))
    assert set(FAIL_CAUSE_WORDS) == set(get_args(FailCause))


def test_a_truncated_judge_names_the_cap_it_was_sent_or_says_it_was_unreported() -> None:
    from threetears.evals.analysis.bundle.schema import ArmExclusions

    sent = ArmExclusions(
        variant_key="v",
        apparatus_class_id="rig-1",
        n_trials=12,
        n_usable=2,
        by_cause={"truncated": 10},
        truncated_sent_caps=[10240],
    )
    unsaid = sent.model_copy(update={"truncated_sent_caps": [], "truncated_cap_unreported": True})
    other_rig = sent.model_copy(update={"apparatus_class_id": "rig-2", "truncated_cap_unreported": True})
    assert "(10240 tokens)" in loss_words("truncated", exclusions=[sent]).words
    remedy = loss_words("truncated", exclusions=[sent]).remedy
    assert remedy == "raise the judge's max_tokens (sent 10240), then rerun both arms"
    assert "max_tokens not reported" in loss_words("truncated", exclusions=[unsaid]).words
    assert "=" not in loss_words("truncated", exclusions=[sent]).words, "a cap in words, never key=value"
    # An arm under two rigs reads every cell's caps: one reported, one not.
    assert loss_words("truncated", exclusions=[sent, other_rig]).words.endswith(
        "(10240 tokens, and unreported on some)"
    )


def test_a_cause_this_build_has_no_words_for_is_named_as_itself_and_says_so() -> None:
    assert loss_words("rate_limited").words == "rate_limited (a cause this build has no words for)"
