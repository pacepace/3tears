"""The style a report is held to: the names it gives identifiers, and the lint every rendered sentence passes.

The owner's verdict on the first production report named each construction below; each test holds one of them.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.style import (
    BANNED_PHRASES,
    GLOSSARY,
    arm_name,
    discounts_by_age,
    fix_prose,
    humanize,
    lint_page,
    lint_prose,
    measure_name,
)


def _rules(text: str, **kwargs: object) -> list[str]:
    return [violation.rule for violation in lint_prose(text, **kwargs)]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("display", "short", "first"),
    [
        ("model=deepseek-v4.1-flash (production)", "production", "deepseek-v4.1-flash (production, control)"),
        ("candidate=v1", "v1", "v1 (control)"),
        ("model=deepseek/deepseek-v4-flash", "deepseek-v4-flash", "deepseek/deepseek-v4-flash (control)"),
        ("prompt=v1, effort=low", "prompt v1, effort low", "prompt v1, effort low (control)"),
        # The candidate's kind is no name: the other lever names the arm.
        ("candidate_kind=assistant, model=m", "m", "m (control)"),
    ],
)
def test_an_arm_is_named_in_full_once_then_by_its_given_label(display: str, short: str, first: str) -> None:
    name = arm_name(display, control=True)
    assert name.label == short and humanize("arm", display) == short
    assert name.first() == first
    assert "=" not in name.first() and ") (" not in name.first()


def test_snake_case_names_read_as_words_and_reserved_axes_by_their_plain_name() -> None:
    assert humanize("dimension", "no_invented_total") == "no invented total"
    assert humanize("dimension", "__outcome__") == "outcome"
    assert humanize("model", "deepseek/deepseek-v4-flash") == "deepseek-v4-flash"
    assert humanize("case", 'The world as it stood before the persona\'s first turn: ## music { "catalog": [') == ""
    assert humanize("case", "clean") == "clean"
    # A case is named by what its author called it; a derived label, a generated id or a bare number is no name.
    assert humanize("case", "double-charge") == "double-charge" and humanize("case", "c01") == "c01"
    assert humanize("case", "plain: eval-6dad555b3db40e35-0") == ""
    assert humanize("case", "eval-6dad555b3db40e35-0") == "" and humanize("case", "7") == ""


def test_the_v26_scoreboard_header_breaks_the_rules_it_was_ruled_on() -> None:
    header = (
        "Scoreboard: what each arm ran, and how each trial ended (the control first, then the other arms by name; "
        "a count, not a ranking)"
    )
    rules = _rules(header, header=True)
    assert rules.count("banned") == 3 and "header" in rules
    assert _rules("Scoreboard", header=True) == []


def test_meta_prose_hedges_and_key_value_are_flagged() -> None:
    assert "banned" in _rules(
        "Findings are numbered F1 onward, and each recommendation names the findings it rests on."
    )
    assert "banned" in _rules("Candidate model showed no measurable cost reduction over production.")
    assert "key_value" in _rules("The judge was sent max_tokens=10240.")
    assert "facts" in _rules("Confidence: low · About: model · Trials: T4, T6")
    assert "dash_chain" in _rules("G2 — the queue — the cheaper model")
    assert all(phrase == phrase.casefold() for phrase in BANNED_PHRASES)


def test_a_long_or_number_laden_sentence_is_flagged() -> None:
    long = " ".join(["word"] * 29) + "."
    assert "long_sentence" in _rules(long)
    assert "numerals" in _rules("Candidate passed 6 of 12 runs against 11 of 12.")
    assert _rules("Candidate passed 6 of 12 runs.") == []
    # An identifier carrying digits is not a numeral, and not a word of prose.
    assert _rules("deepseek-v4.1-flash and T4 and v2 ran.") == []


def test_the_declared_question_repeated_in_the_body_is_flagged_and_dropped() -> None:
    question = "Does the cheaper model hold every queue dimension within ten points of production's pass rate?"
    body = f"We asked: {question} The answer is no."
    assert "question" in _rules(body, question=question)
    fixed, remaining = fix_prose(body + " Every number is code's.", question=question)
    assert fixed == "The answer is no." and remaining == []


@pytest.mark.parametrize(
    "sentence",
    [
        "The question was set after the runs began, so these readings are weaker.",
        "Because the runs predate the campaign, treat the gain as provisional.",
        "These readings rest on older runs and should be confirmed by fresh ones.",
        "Read post hoc, the tone gain is only a lead.",
        "The arms ran on different days, so they are not one experiment.",
        "Control and candidate were measured hours apart, which confounds the comparison.",
        "The margin was set after the readings were in, so the gain is provisional.",
        "The control runs were collected before the question, so read them with lower confidence.",
        # An arm named by its model is no change: the noun alone never keeps an age discount.
        "The candidate model's gain rests on older runs, so it is provisional.",
    ],
)
def test_a_sentence_discounting_a_reading_by_data_age_or_time_apart_is_named(sentence: str) -> None:
    """Owner rulings (2026-10-23): a run whose measurement context is unchanged reads exactly like a fresh one, and arms
    measured apart are one experiment, so the writer never discounts a reading for when it ran."""
    assert discounts_by_age(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        # A real config change between the arms, which is what the owner ruled still matters.
        "The judge was re-pinned after the runs started, so the arms were graded by different judges.",
        "The older runs were scored by a judge since replaced.",
        "Older runs used the v1 prompt.",
        "Older runs used the v1 prompt, so they are not directly comparable.",
        "The arms ran on different days and the judge changed between them, so the comparison is confounded.",
        # The late-criterion disclosure the writer is asked for, in its own words and in code's.
        "The margin was set after the results were in.",
        "The margin on billing was set on 20 Oct 2026, after its readings were visible.",
    ],
)
def test_a_sentence_naming_a_change_or_a_late_criterion_is_never_read_as_a_discount(sentence: str) -> None:
    assert not discounts_by_age(sentence)
    assert fix_prose(sentence) == (sentence, [])


def test_a_page_lints_its_prose_and_headers_not_its_data() -> None:
    page = "\n".join(
        [
            "# A title that is a whole question and may run long, as a title does?",
            "",
            "| Arm | Cost per reply |",
            "|---|---|",
            "| production | rests on 4 of 5 and 6 of 7 |",
            "",
            "<details><summary>T1 · case 1 · production · passed</summary>",
            "",
            "```",
            "Written by anyone, inside a quoted dump.",
            "```",
            "",
            "</details>",
            "",
            "## Contrasts against the control",
        ]
    )
    assert [violation.rule for violation in lint_page(page)] == ["header"]


def test_a_check_expression_is_code_and_never_humanized() -> None:
    expression = 'call_count("music.queue_track") >= 2'
    # Underscores and dots are part of the code a reader copies: humanize leaves an expression as written.
    assert humanize("dimension", expression) == expression
    nested = 'all(it.segue_text.length > 0 for it in calls("music.queue_track"))'
    assert humanize("dimension", nested) == nested
    # A check's measure is named by its expression, as code, never "Pass rate of"; any other by its reader name.
    measure = f"goal_state:{expression}"
    assert measure_name(measure, f"Pass rate of {expression}") == f"`{expression}`"
    assert measure_name("failed_reply_notices", "Failed-reply notices") == "Failed-reply notices"
    assert measure_name("turn_time") == "turn time"


def test_every_glossary_entry_is_one_short_plain_sentence() -> None:
    for term, sentence in GLOSSARY.items():
        assert lint_prose(sentence) == [], term
        assert sentence.endswith(".") and sentence.count(". ") == 0, term
