"""Pin the exact prose a model reads for free text in a web search listing.

Titles, snippets, filter details, and notices are provider text of any number of
lines. Each must stay under its own result or footnote: a snippet's lines all
indented, no blank line inside a result, a title or detail joined onto one line.
"""

from __future__ import annotations

from threetears.search.bind import render_prose
from threetears.search.contracts import CandidateSet, CriterionDisposition
from packages.search.tests.search_instances import CANDIDATE

FACTS = "published 2026-03-04 09:00 UTC; found by duckduckgo"


def _prose(**update: object) -> str:
    """Render one candidate with ``update`` applied to the shared fixture."""
    return render_prose(CandidateSet(candidates=(CANDIDATE.model_copy(update=update),)))


def test_a_many_line_snippet_keeps_every_line_indented_under_its_result() -> None:
    """Blank lines vanish, an indented line keeps its relative indent, no line lands at the margin."""
    prose = _prose(title="Capybara", snippet="First line.\n\nSecond line.\n    indented detail\nThird line.")

    assert prose == (
        "1. Capybara\n"
        f"   URL: {CANDIDATE.identity}\n"
        f"   {FACTS}\n"
        "   First line.\n"
        "   Second line.\n"
        "       indented detail\n"
        "   Third line."
    )


def test_two_results_with_many_line_snippets_are_separated_only_between_results() -> None:
    """The one blank line is the gap between results; none sits inside either."""
    first = CANDIDATE.model_copy(update={"title": "One", "snippet": "a\n\nb"})
    second = CANDIDATE.model_copy(update={"title": "Two", "identity": "https://example.org/two", "snippet": "c\n\nd"})
    prose = render_prose(CandidateSet(candidates=(first, second)))

    assert prose == (
        "1. One\n"
        f"   URL: {CANDIDATE.identity}\n"
        f"   {FACTS}\n"
        "   a\n"
        "   b\n"
        "\n"
        "2. Two\n"
        "   URL: https://example.org/two\n"
        f"   {FACTS}\n"
        "   c\n"
        "   d"
    )


def test_a_many_line_title_is_joined_onto_the_heading_line() -> None:
    prose = _prose(title="  Capybara\n\n   habitats  \n", snippet=None)

    assert prose == f"1. Capybara habitats\n   URL: {CANDIDATE.identity}\n   {FACTS}"


def test_a_blank_title_shows_the_untitled_heading() -> None:
    prose = _prose(title=" \n ", snippet=None)

    assert prose == f"1. Untitled\n   URL: {CANDIDATE.identity}\n   {FACTS}"


def test_a_whitespace_only_snippet_still_shows_the_result_heading() -> None:
    prose = _prose(title="Capybara", snippet=" \n\n  \n")

    assert prose == f"1. Capybara\n   URL: {CANDIDATE.identity}\n   {FACTS}"


def test_an_unmet_filter_detail_is_joined_onto_its_footnote_line() -> None:
    result = CandidateSet(
        dispositions=(
            CriterionDisposition(
                criterion_key="time-range",
                disposition="unsatisfied",
                detail="the engine\n\n  ignores dates\n    entirely",
            ),
        ),
    )

    assert render_prose(result).endswith(
        "What this search could not do:\n- the time-range filter was not applied: the engine ignores dates entirely"
    )


def test_a_many_line_notice_is_joined_onto_its_footnote_line() -> None:
    candidate = CANDIDATE.model_copy(update={"title": "T", "snippet": None})
    result = CandidateSet(candidates=(candidate,), notices=("engines down:\n\n brave\n\twikidata",))

    assert render_prose(result) == (
        f"1. T\n   URL: {CANDIDATE.identity}\n   {FACTS}\n\n"
        "What this search could not do:\n"
        "- engines down: brave wikidata"
    )


def test_a_filter_without_detail_still_shows_its_footnote_line() -> None:
    result = CandidateSet(
        dispositions=(CriterionDisposition(criterion_key="time-range", disposition="unsatisfied", detail=" \n "),),
    )

    assert render_prose(result).endswith("What this search could not do:\n- the time-range filter was not applied")
