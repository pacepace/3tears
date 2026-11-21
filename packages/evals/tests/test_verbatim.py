"""The verbatim check: a quoted stretch is found in its text through the fold, and its offsets slice the original.

Each character class the fold forgives is driven on its own (curly quotes, dashes, the soft hyphen, zero-width
characters, the no-break space, case, whitespace runs, a decomposed accent, a case fold that expands), each asserting
that the offsets returned slice the original text, typography and all. What it never forgives is held too: a changed
word, a blank needle, a match from or to the middle of a fold unit, and an elided quote. An ellipsis the text itself
holds is found like any other character (revision 1 refused it, which made such a stretch impossible to quote). The
fold's identity is a digest of its revision, both its tables and the Unicode version: changing any table moves it, and
it is pinned.
"""

from __future__ import annotations

import unicodedata
from dataclasses import replace

import pytest

from threetears.evals.kernel.claims.verbatim import VERBATIM_FOLD, verbatim_span


def _slice(needle: str, haystack: str) -> str | None:
    span = verbatim_span(needle, haystack)
    return None if span is None else haystack[span[0] : span[1]]


@pytest.mark.parametrize(
    ("needle", "haystack", "original"),
    [
        pytest.param(
            "the guide's favourite",
            "Then came the guide’s favourite room.",
            "the guide’s favourite",
            id="curly-apostrophe",
        ),
        pytest.param(
            'called it "the long hall"',
            "Locals called it “the long hall” for years.",
            "called it “the long hall”",
            id="curly-double-quotes",
        ),
        pytest.param(
            "built 1890-1894 by Orsk",
            "The Varn Bridge, built 1890–1894 by Orsk, still stands.",
            "built 1890–1894 by Orsk",
            id="en-dash",
        ),
        pytest.param(
            "the bridge - and the mill",
            "There was the bridge — and the mill.",
            "the bridge — and the mill",
            id="em-dash",
        ),
        pytest.param(
            "a cooperative venture",
            "It began as a co­operative venture.",
            "a co­operative venture",
            id="soft-hyphen",
        ),
        pytest.param(
            "the Varn Bridge",
            "Cross the Varn​ Bridge at dusk.",
            "the Varn​ Bridge",
            id="zero-width-space",
        ),
        pytest.param(
            "Lisbon",
            "Through Lis‍bon and on.",
            "Lis‍bon",
            id="zero-width-joiner",
        ),
        pytest.param(
            "a 12 km walk",
            "It is a 12 km walk.",
            "a 12 km walk",
            id="no-break-space",
        ),
        pytest.param(
            "the varn bridge OPENED",
            "And the Varn Bridge opened in 1894.",
            "the Varn Bridge opened",
            id="case",
        ),
        pytest.param(
            "opened in 1894",
            "The bridge opened  \n\t in   1894, late.",
            "opened  \n\t in   1894",
            id="whitespace-runs",
        ),
        pytest.param(
            "  opened in 1894 ",
            "The bridge opened in 1894.",
            "opened in 1894",
            id="needle-ends-trimmed",
        ),
        pytest.param(
            "café by the river",
            "The café by the river opened first.",
            "café by the river",
            id="decomposed-accent",
        ),
        pytest.param(
            "the strasse",
            "Down the Straße to the square.",
            "the Straße",
            id="expanding-case-fold",
        ),
    ],
)
def test_each_forgiven_class_is_found_and_its_offsets_slice_the_original(
    needle: str, haystack: str, original: str
) -> None:
    span = verbatim_span(needle, haystack)

    assert span is not None
    start, end = span
    assert haystack[start:end] == original
    assert haystack.index(original) == start


def test_the_first_match_is_the_one_returned() -> None:
    haystack = "It opened in 1894. The other opened in 1894 too."

    assert verbatim_span("opened in 1894", haystack) == (3, 17)


@pytest.mark.parametrize(
    ("needle", "haystack"),
    [
        pytest.param("opened in 1895", "The bridge opened in 1894.", id="a-changed-word"),
        pytest.param("opened, in 1894", "The bridge opened in 1894.", id="added-punctuation"),
        pytest.param("   ", "The bridge opened in 1894.", id="blank"),
        pytest.param("", "The bridge opened in 1894.", id="empty"),
        pytest.param("the stras", "Down the Straße.", id="ends-inside-a-unit"),
        pytest.param("cafe", "The café opened.", id="drops-a-combining-mark"),
    ],
)
def test_what_is_not_verbatim_is_not_found(needle: str, haystack: str) -> None:
    assert verbatim_span(needle, haystack) is None


@pytest.mark.parametrize(
    ("needle", "haystack"),
    [
        pytest.param("opened \u2026 1894", "The bridge opened in 1894.", id="unicode-ellipsis-eliding"),
        pytest.param("opened ... 1894", "The bridge opened in 1894.", id="three-stops-eliding"),
    ],
)
def test_an_elided_quote_is_not_found(needle: str, haystack: str) -> None:
    assert verbatim_span(needle, haystack) is None


@pytest.mark.parametrize(
    ("needle", "haystack", "original"),
    [
        pytest.param(
            "opened\u2026 in 1894", "The bridge opened\u2026 in 1894.", "opened\u2026 in 1894", id="unicode-ellipsis"
        ),
        pytest.param(
            "opened... in 1894", "The bridge opened\u2026 in 1894.", "opened\u2026 in 1894", id="as-three-stops"
        ),
        pytest.param("opened... in 1894", "The bridge opened... in 1894.", "opened... in 1894", id="three-stops"),
    ],
)
def test_an_ellipsis_the_text_holds_is_found_like_any_other_character(
    needle: str, haystack: str, original: str
) -> None:
    span = verbatim_span(needle, haystack)

    assert span is not None
    assert haystack[span[0] : span[1]] == original


#: The fold's identity on this build's tables and Unicode database, produced by ``VerbatimFold.identity`` itself. It
#: moves with an edited table or a Python whose Unicode data differs: that is a new definition, written here on purpose.
_PINNED_FOLD = "verbatim-fold-v2:ee4fe759dc40"


def test_the_folds_identity_is_pinned_and_moves_with_each_table_and_the_revision() -> None:
    assert VERBATIM_FOLD.identity == _PINNED_FOLD, f"pinned at Unicode {unicodedata.unidata_version}"
    moved = {
        replace(VERBATIM_FOLD, to_ascii={**VERBATIM_FOLD.to_ascii, "\u2026": "."}).identity,
        replace(VERBATIM_FOLD, to_ascii={**VERBATIM_FOLD.to_ascii, "\u2014": "--"}).identity,
        replace(VERBATIM_FOLD, removed=VERBATIM_FOLD.removed | {"\u00a0"}).identity,
        replace(VERBATIM_FOLD, removed=VERBATIM_FOLD.removed - {"\u00ad"}).identity,
        replace(VERBATIM_FOLD, revision="verbatim-fold-v3").identity,
    }
    assert VERBATIM_FOLD.identity not in moved
    assert len(moved) == 5
    reordered = replace(VERBATIM_FOLD, to_ascii=dict(reversed(list(VERBATIM_FOLD.to_ascii.items()))))
    assert reordered.identity == VERBATIM_FOLD.identity, "a table's order is no part of it"


def test_a_changed_table_changes_what_matches() -> None:
    kept = replace(VERBATIM_FOLD, removed=VERBATIM_FOLD.removed - {"\u00ad"})

    assert VERBATIM_FOLD.span("cooperative", "a co\u00adoperative") is not None
    assert kept.span("cooperative", "a co\u00adoperative") is None
