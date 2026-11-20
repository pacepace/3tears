"""A change chart of one row is refused (#1124, owner ruling 2026-11-14): one change belongs in a sentence or a table,
which state it with its interval, never in a chart whose one row compares it with nothing.

The rule is structural, the count of the rows the chart draws, so it holds for a forest and diverging bars alike, and
for the dot strips drawn through the diverging bars' rows. Kairo's written page draws its pass-rate contrasts; the same
chart narrowed to one arm draws one change. Only a chart of what changed is refused: two arms compared on one measure
are one difference by what they are, and stay drawn (owner ruling, 2026-11-14).
"""

from __future__ import annotations

import copy

import pytest

from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc

#: Kairo's written page, whose third claim draws a chart its tests may redraw.
_KAIRO = "prototype3/kairo"

#: Kairo's pass-rate contrasts on its one aspect: two arms against the control, so two changes.
_CONTRASTS: dict[str, object] = {
    "source": "contrasts",
    "aspect": "character adherence (v2)",
    "measures": ["passed"],
    "arms": [],
    "strata": [],
}


def _page(kind: str, query: dict[str, object], *, one_candidate: bool = False) -> str:
    """Kairo's written page with its third claim's chart redrawn as ``kind`` over ``query``, as Markdown; with
    ``one_candidate``, its input holding the pass rate's contrast of one candidate alone, as a campaign of one candidate
    does."""
    doc = copy.deepcopy(converted_doc(_KAIRO))
    evidence = doc["claims"][3]["evidence"][1]
    evidence["block"] = kind
    evidence["query"].update(query)
    evidence["caption"] = "The chart drawn here."
    held = ReportInput.model_validate(snapshot_of(_KAIRO).report_input)
    if one_candidate:
        kept = [c for c in held.contrasts if not (c.arm == "1017" and c.measure == "passed")]
        held = held.model_copy(update={"contrasts": kept})
    return doc_page_markdown(build(ReportDoc.model_validate(doc), held, strict=True))


@pytest.mark.parametrize("kind", ["forest", "diverging_bars", "dot_interval"])
def test_a_change_chart_of_two_rows_is_drawn(kind: str) -> None:
    assert "The chart drawn here." in _page(kind, _CONTRASTS)


@pytest.mark.parametrize("kind", ["forest", "diverging_bars", "dot_interval"])
def test_a_change_chart_of_one_row_is_refused_naming_where_the_change_belongs(kind: str) -> None:
    with pytest.raises(UnresolvableReference, match="it draws one change") as refused:
        _page(kind, {**_CONTRASTS, "arms": ["qwen3.8-flash"]})
    assert "in the prose" in str(refused.value) and "in a table" in str(refused.value)


def test_the_refusal_names_a_compare_chart_and_more_rows_only_where_the_source_holds_them() -> None:
    """The refusal's advice can be followed (Critic D-3): it names a ``compare`` chart of the two arms, which draws one
    difference, and selecting more rows only where the source holds another change of the measure. Kairo's two
    candidates hold two; a campaign of one candidate holds one, and is told no more."""
    with pytest.raises(UnresolvableReference) as two:
        _page("forest", {**_CONTRASTS, "arms": ["qwen3.8-flash"]})
    with pytest.raises(UnresolvableReference) as one:
        _page("forest", {**_CONTRASTS, "arms": ["qwen3.8-flash"]}, one_candidate=True)
    assert "a 'compare' chart" in str(two.value) and "a 'compare' chart" in str(one.value)
    assert "select more rows" in str(two.value)
    assert "select more rows" not in str(one.value)


@pytest.mark.parametrize("kind", ["diverging_bars", "dot_interval"])
def test_two_arms_compared_on_one_measure_are_drawn_as_their_one_difference(kind: str) -> None:
    compare = {"source": "compare", "aspect": "", "measures": ["failed_reply_notices"], "arms": ["1017", "September"]}
    assert "The chart drawn here." in _page(kind, {**_CONTRASTS, **compare})
