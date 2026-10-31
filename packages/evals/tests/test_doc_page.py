"""The page a report document draws: built from the document and its snapshot's report input alone.

Held over the 18 converted prototype documents (``fixtures/report_docs``), each against the snapshot it was written
from. The expected side is read from somewhere else than the page: ``block_render`` (the block the analyst saw), the
document itself, the input's arms, the figure keys and code's caveats.

- **Every document renders.** The 12 documents that name only what their snapshot holds (the second and third
  prototypes', and the first prototype's Kairo run) build strict, to HTML (default and static) and Markdown. The other
  6 of the first prototype name caveat ids their analyst made up, which code does not assign (kept as planted
  negatives), so strict refuses them by that caveat's path, and they render read as stored, degraded.
- **Every block is the block the analyst saw**: each evidence block's values table on the page is ``block_render``'s
  for the same kind, query, reference line and emphasis, row for row; a claim of two blocks draws both, in order; a
  reference line is code's, at the figure its key reads; emphasis is drawn.
- **The options table** holds every arm of the input in its order, the ones ``recommends`` names marked and set apart,
  no other; the four tradeoff documents recommend nothing, so none is.
- **The appendix and Method** are always on the page, in every form, after every claim.
- **No internal id above the appendix**: no caveat id, measure key, claim id or arm id (where its name differs) is in
  the text above the appendix, which holds each block's values and every string its drawn chart shows a reader.
- **The appendix's own parts**: Method under its own heading, after what was left out; the editor's findings left
  open after it, each by its severity and the claim's place (linked) or the page; the made line last; none of it
  above the appendix. A caveat landed on a claim or a Method term is printed beside it by its label.
- **``story`` is read by nothing**: the page, its Markdown and its text are byte-identical whatever it says.
"""

from __future__ import annotations

import html
import json
import re
from functools import cache
from typing import Any, get_args

import pytest

from packages.evals.tests.report_doc_support import LATER_DOCS, PROTOTYPE1_DOCS, converted_doc, snapshot_of
from threetears.evals.analysis.analyst.blocks import block_render, values_table
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report.doc_page.text import above_appendix_text, doc_page_text
from threetears.evals.analysis.report.doc_page.words import (
    APPENDIX_TITLE,
    CAVEAT_LABELS,
    CAVEAT_REPEATED,
    CAVEAT_REPEATED_ON,
    CAVEATS_TEMPLATE,
    DEGRADED_PAGE,
    FINDINGS_TITLE,
    METHOD_TITLE,
    OMITTED_TITLE,
    SEVERITY_WORDS,
)
from threetears.evals.analysis.report.figure_keys import format_figure, resolve
from threetears.evals.analysis.report_decision import every_caveat
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.kernel.report_doc import ReportDoc, ReportFinding, Story

#: Every converted document.
_ALL = PROTOTYPE1_DOCS + LATER_DOCS

#: The documents that name only what their snapshot holds, which build strict.
_VALID = ("prototype/kairo", *LATER_DOCS)

#: The first prototype's documents that name caveat ids their analyst made up: planted negatives.
_PLANTED = tuple(name for name in PROTOTYPE1_DOCS if name not in _VALID)

#: More rows than any fixture block holds, so a block's values table comes back whole.
_EVERY_ROW = 10_000


@cache
def _input(name: str) -> ReportInput:
    return ReportInput.model_validate(snapshot_of(name).report_input)


def _doc(name: str, **changes: Any) -> ReportDoc:
    return ReportDoc.model_validate({**converted_doc(name), **changes})


@cache
def _page(name: str) -> DocPage:
    return build(_doc(name), _input(name), strict=name in _VALID)


# --- every document renders ----------------------------------------------------------------------------------------


def test_twelve_documents_build_strict_and_six_are_planted_negatives() -> None:
    assert (len(_VALID), len(_PLANTED)) == (12, 6)


@pytest.mark.parametrize("name", _VALID)
def test_each_valid_document_renders_strict_to_html_and_markdown(name: str) -> None:
    """A document whose ids its snapshot holds builds strict, with nothing degraded, and every form is written."""
    page = _page(name)

    written, static, markdown = doc_page_html(page), doc_page_html(page, static=True), doc_page_markdown(page)

    assert page.degraded == ""
    assert written.startswith("<!doctype html>") and "<script>" in written
    assert "<script" not in static and "<link" not in static
    assert markdown.startswith(page.masthead.kind)
    assert all(evidence.chart is not None and not evidence.degraded for c in page.claims for evidence in c.evidence)


@pytest.mark.parametrize("name", _PLANTED)
def test_a_first_prototype_document_is_refused_strict_by_its_made_up_caveat_and_reads_degraded(name: str) -> None:
    """Strict refuses a caveat id code does not assign, naming its path; read as stored, the page renders and says so."""
    doc = _doc(name)
    assigned = {caveat.id for caveat in every_caveat(_input(name))}
    made_up = [i for i, caveat in enumerate(doc.confidence.caveats) if caveat.id not in assigned]
    assert made_up, "the first prototype's documents keep their made-up caveat ids as planted negatives"

    with pytest.raises(UnresolvableReference, match=re.escape(f"confidence.caveats[{made_up[0]}]")):
        build(doc, _input(name), strict=True)
    page = _page(name)

    assert page.degraded == DEGRADED_PAGE
    assert DEGRADED_PAGE in html.unescape(doc_page_html(page))
    assert DEGRADED_PAGE in doc_page_markdown(page)


# --- every block is the block the analyst saw ----------------------------------------------------------------------


@pytest.mark.parametrize("name", _VALID)
def test_every_block_is_the_values_block_render_returns(name: str) -> None:
    """Each block's values on the page are ``block_render``'s for its kind, query, reference line and emphasis."""
    doc, page, held = _doc(name), _page(name), _input(name)

    for written, drawn in zip(doc.claims, page.claims, strict=True):
        assert [evidence.block for evidence in drawn.evidence] == [evidence.block for evidence in written.evidence]
        for evidence, shown in zip(written.evidence, drawn.evidence, strict=True):
            seen = block_render(
                held,
                block_kind=evidence.block,
                query=evidence.query,
                reference=evidence.reference,
                emphasis=evidence.emphasis,
                limit=_EVERY_ROW,
                fits=lambda _block: True,
            )
            assert shown.chart is not None
            columns, rows = values_table(shown.chart.chart)
            assert (columns, rows) == (seen.columns, [row.cells for row in seen.rows])
            assert (shown.level, shown.unit) == (seen.level, seen.unit)


def test_a_claim_of_two_blocks_draws_both_in_its_order() -> None:
    """Kairo's lead claim rests on a bars block and a forest; the page draws both under it, in that order."""
    name = "prototype3/kairo"
    (lead, *_rest) = _page(name).claims
    written = doc_page_html(_page(name))
    claim = written[written.index('id="claim-1"') : written.index('id="claim-2"')]

    assert [evidence.block for evidence in lead.evidence] == ["bars", "forest"]
    assert [figure for figure in re.findall(r'data-figure="([^"]+)"', claim)] == ["bars", "forest"]
    assert claim.index(lead.evidence[0].caption[:20]) < claim.index(lead.evidence[1].caption[:20])


def test_a_reference_line_is_codes_at_the_figure_its_key_reads() -> None:
    """Kairo's forest draws the margin: the page names it at the figure ``block_render`` keys it to."""
    name = "prototype3/kairo"
    doc = _doc(name)
    (at, evidence) = next(
        (place, evidence)
        for place, claim in enumerate(doc.claims)
        for evidence in claim.evidence
        if evidence.reference == "margin"
    )
    seen = block_render(_input(name), block_kind=evidence.block, query=evidence.query, reference="margin")
    assert seen.reference is not None

    (shown,) = [one for one in _page(name).claims[at].evidence if one.block == evidence.block]

    assert shown.reference == f"Margin {format_figure(resolve(seen.reference.key, _input(name)))}"


def test_emphasis_is_drawn() -> None:
    """The fourth tradeoff's lead block stresses one arm; drawn without it, the chart the kit draws differs."""
    name = "prototype3/r4"
    doc = _doc(name)
    (stressed, *_rest) = [evidence for evidence in doc.claims[0].evidence if evidence.emphasis]
    plain = doc.model_copy(
        update={
            "claims": [
                doc.claims[0].model_copy(
                    update={
                        "evidence": [
                            evidence.model_copy(update={"emphasis": ""}) if evidence is stressed else evidence
                            for evidence in doc.claims[0].evidence
                        ]
                    }
                ),
                *doc.claims[1:],
            ]
        }
    )
    place = doc.claims[0].evidence.index(stressed)

    def kit(page: DocPage) -> str:
        chart = page.claims[0].evidence[place].chart
        assert chart is not None and isinstance(chart.chart, ChartBlock)
        return json.dumps(chart.chart.kit_chart(), sort_keys=True)

    assert kit(_page(name)) != kit(build(plain, _input(name), strict=True))


# --- the options table -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL)
def test_every_arm_is_in_the_options_table_and_only_the_recommended_are_marked(name: str) -> None:
    """A row per arm of the input, in its order, each named by its arm; the recommended rows are set apart and say so."""
    doc, page, held = _doc(name), _page(name), _input(name)
    options = page.options
    marked = [index for index, row in enumerate(options.rows) if (row["role"].flag or "") == "recommended"]

    assert [row["option"].tag for row in options.rows] == [row["arm"].tag for row in page.arms_key.table.rows]
    assert len(options.rows) == len(held.arms)
    assert page.recommended_rows == marked == [i for i, arm in enumerate(held.arms) if arm.id in doc.recommends]
    highlighted = re.findall(
        r"<tr( class=\"hl\")?>",
        doc_page_html(page).split('data-table="options"')[1].split("<tbody>")[1].split("</tbody>")[0],
    )
    assert [index for index, css in enumerate(highlighted) if css] == marked


def test_the_four_tradeoff_documents_recommend_nothing_and_mark_no_arm() -> None:
    """A tradeoff's lean stays in its prose: no arm is recommended, so no row is set apart."""
    tradeoffs = [name for name in _ALL if _doc(name).story == "tradeoff"]

    assert len(tradeoffs) == 4
    assert all(_page(name).recommended_rows == [] for name in tradeoffs)


# --- the appendix and Method ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL)
def test_the_appendix_and_method_are_always_on_the_page_after_every_claim(name: str) -> None:
    """The appendix divider and Method are in the HTML, the Markdown and the page text, after the last claim."""
    page = _page(name)
    written, markdown, text = doc_page_html(page), doc_page_markdown(page), doc_page_text(page)
    last = page.claims[-1].text.split("{")[0][:24]

    assert page.method and page.full_results and page.full_results[0].name == "arms_by_measure"
    assert (
        written.index(f'id="claim-{len(page.claims)}"')
        < written.index('data-part="appendix"')
        < written.index('data-part="method"')
    )
    assert markdown.index(last) < markdown.index(f"## {APPENDIX_TITLE}")
    assert text.index(last) < text.index(f"# {APPENDIX_TITLE}") < text.index(f"- {page.method[0].term}")
    assert f"# {APPENDIX_TITLE}" not in above_appendix_text(page)
    for form, heading in (
        (written, f"<h3>{METHOD_TITLE}</h3><dl"),
        (markdown, f"### {METHOD_TITLE}\n\n- **"),
        (text, f"## {METHOD_TITLE}\n- "),
    ):
        assert form.index(f"{APPENDIX_TITLE}") < form.index(heading)
        if page.omitted:
            assert form.index(OMITTED_TITLE) < form.index(heading)


# --- no internal id above the appendix ----------------------------------------------------------------------------


def _internal_ids(name: str) -> set[str]:
    """The ids a reader is never shown above the appendix: caveat ids, measure keys and claim ids that read as ids,
    and arm ids where the arm's name is not its id."""
    held = _input(name)
    ids = {caveat.id for caveat in every_caveat(held)} | {caveat.id for caveat in _doc(name).confidence.caveats}
    ids |= {key for key in [*held.measures, *held.judged] if re.search(r"[_.:]", key)}
    ids |= {claim.id for claim in _doc(name).claims}
    ids |= {arm.id for arm in fact_arms(held) if arm.id not in arm.name}
    return ids


@pytest.mark.parametrize("name", _VALID)
def test_no_internal_id_is_in_the_text_above_the_appendix(name: str) -> None:
    """Above the appendix the page names caveats by label and measures by reader name; the editor's text alone names a
    claim by its id."""
    above = above_appendix_text(_page(name))

    leaked = sorted(
        found for found in _internal_ids(name) if re.search(rf"(?<![\w:]){re.escape(found)}(?![\w:])", above)
    )

    assert not leaked
    assert all(f"Claim {claim.id} (" in doc_page_text(_page(name)) for claim in _doc(name).claims)


# --- story is read by nothing ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _VALID)
def test_the_page_is_byte_identical_whatever_the_story_says(name: str) -> None:
    """Every story a document could tell draws the same HTML, Markdown and text."""
    forms = {
        (doc_page_html(page), doc_page_markdown(page), doc_page_text(page))
        for page in (build(_doc(name, story=story), _input(name), strict=True) for story in get_args(Story))
    }

    assert len(forms) == 1


# --- the editor's findings, the made line, and caveats landed beside a claim or a term ------------------------------

#: The made line a caller passes.
_MADE = "Written by the analyst loop on 2026-10-28 from this campaign's frozen snapshot."

#: One finding on the second claim, one on the page as a whole.
_FINDINGS = (
    ReportFinding(claim="c2", severity="must_fix", finding="The second claim reads a drop the interval does not show."),
    ReportFinding(claim="", severity="should_fix", finding="The lede repeats the headline."),
)


def test_the_findings_left_open_and_the_made_line_close_the_appendix_in_every_form() -> None:
    """After Method: each finding by its severity, the claim by its place (linked in HTML) or the page; then the made
    line, last. None of it is above the appendix."""
    name = "prototype3/kairo"
    page = build(_doc(name), _input(name), strict=True, findings=_FINDINGS, made=_MADE)
    written, markdown, text = html.unescape(doc_page_html(page)), doc_page_markdown(page), doc_page_text(page)
    must, should = SEVERITY_WORDS["must_fix"], SEVERITY_WORDS["should_fix"]

    assert '<li><b>Must fix</b> · <a href="#claim-2">Claim 2</a>: The second claim' in written
    assert f"<li><b>{should}</b> · The page: The lede repeats the headline.</li>" in written
    assert f"- **{must}** · Claim 2: The second claim" in markdown
    assert f"- **{should}** · The page: The lede repeats the headline." in markdown
    assert f"- {must} · Claim 2: The second claim" in text and f"- {should} · The page: The lede" in text
    for form, method in (
        (written, f"<h3>{METHOD_TITLE}</h3>"),
        (markdown, f"### {METHOD_TITLE}"),
        (text, f"## {METHOD_TITLE}"),
    ):
        assert form.index(method) < form.index(FINDINGS_TITLE) < form.index(must) < form.index(should)
        assert form.index(should) < form.index(_MADE)
    assert written.index(_MADE) > written.index("The lede repeats") and "</article>" in written[written.index(_MADE) :]
    assert markdown.rstrip().endswith(f"_{_MADE}_") and text.rstrip().endswith(_MADE)
    above = above_appendix_text(page)
    assert not any(said in above for said in (FINDINGS_TITLE, must, should, _MADE))


def _landed(doc: ReportDoc) -> ReportDoc:
    """Kairo's document with a code caveat landed on its second claim and an analyst caveat on its first Method term."""
    data = doc.model_dump(mode="json", by_alias=True)
    data["confidence"]["caveats"].append(
        {"id": "analyst:case-mix", "label": "Case mix differs from traffic", "material_if": "Only if traffic differs."}
    )
    data["claims"][1]["caveats"] = ["confound:model"]
    data["method"][0]["caveats"] = ["analyst:case-mix"]
    return ReportDoc.model_validate(data)


def test_a_caveat_landed_on_a_claim_or_a_method_term_is_printed_beside_it_by_its_label() -> None:
    """Code's label for a code caveat, the analyst's for its own, beside the claim and the term in every form."""
    name = "prototype3/kairo"
    page = build(_landed(_doc(name)), _input(name), strict=True)
    on_claim = CAVEATS_TEMPLATE.format(labels=CAVEAT_LABELS["confound"].format(name="model"))
    on_term = CAVEATS_TEMPLATE.format(labels="Case mix differs from traffic")
    written, markdown, text = html.unescape(doc_page_html(page)), doc_page_markdown(page), doc_page_text(page)

    assert page.claims[1].caveats == [CAVEAT_LABELS["confound"].format(name="model")]
    assert page.method[0].caveats == ["Case mix differs from traffic"]
    claim = written[written.index('id="claim-2"') : written.index('id="claim-3"')]
    assert on_claim in claim and on_claim not in written.replace(claim, "")
    assert on_term in written[written.index(f"<h3>{METHOD_TITLE}</h3>") :]
    for form in (markdown, text):
        assert form.index(page.claims[1].text[:30]) < form.index(on_claim) < form.index(page.claims[2].text[:30])
        assert form.index(f"{METHOD_TITLE}") < form.index(on_term)
    assert on_claim in above_appendix_text(page) and on_term not in above_appendix_text(page)
    assert "confound:model" not in above_appendix_text(page)


def test_a_label_two_caveats_share_is_printed_once_naming_what_tells_them_apart_in_every_form() -> None:
    """Kairo's character claim lands two short runs, both of qwen3.8-flash, which code labels alike: the line beside it
    says the label once, how many caveats carry it (in words) and the arm their records name, in HTML, Markdown and
    text alike. A claim landing one of them prints its label alone."""
    name = "prototype4/kairo"
    doc = _doc(name)
    runs = doc.claims[1].caveats
    label = CAVEAT_LABELS["short_run"]
    said = CAVEAT_REPEATED_ON.format(label=label, count="two", arms="qwen3.8-flash (A)")
    page = build(doc, _input(name), strict=True)

    assert len(runs) == 2 and all(run.startswith("short_run:") for run in runs)
    assert page.claims[1].caveats == [said]
    _printed_once(page, CAVEATS_TEMPLATE.format(labels=said), label)
    one = doc.model_copy(update={"claims": [doc.claims[0], doc.claims[1].model_copy(update={"caveats": runs[:1]})]})
    assert build(one, _input(name), strict=True).claims[1].caveats == [label]


def test_a_shared_label_whose_records_name_no_arm_says_only_how_many() -> None:
    """The same two short runs with no arm recorded on either: the line says the label once and how many, no arm."""
    name = "prototype4/kairo"
    held = _input(name)
    runs = [run.model_copy(update={"arm": None}) for run in held.caveats.short_runs]
    armless = held.model_copy(update={"caveats": held.caveats.model_copy(update={"short_runs": runs})})
    label = CAVEAT_LABELS["short_run"]
    said = CAVEAT_REPEATED.format(label=label, count="two")
    page = build(_doc(name), armless, strict=True)

    assert page.claims[1].caveats == [said]
    _printed_once(page, CAVEATS_TEMPLATE.format(labels=said), label)


def _printed_once(page: DocPage, line: str, label: str) -> None:
    """The line is printed once in every form, and the label is never printed twice in a row."""
    for form in (html.unescape(doc_page_html(page)), doc_page_markdown(page), doc_page_text(page)):
        assert form.count(line) == 1
        assert f"{label}; {label}" not in form
