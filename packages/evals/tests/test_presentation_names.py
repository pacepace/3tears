"""Each name once, each term defined where it is first used, each heading once, and one order on every replay.

The review's A3 term clause holds a page to terms defined before use and one name per thing. Held here on the
ladder's replayed rungs (offline, no key), the Kairo capture, the stored report documents and the listing:

- no page heading names an arm twice: a page heads with the campaign's name, its arms the line under it, and a
  listing's title names each part once;
- no prose line a page prints is a ``key=value``, read through :func:`~threetears.evals.analysis.style.lint_prose`,
  and a list of unchecked settings prints once, for the arms it covers;
- every glossary term a page defines is linked where the page first uses it in its prose, or by a header
  before that; the first screen's judge line links the judged scale, which the glossary states once;
- no heading is printed twice: a table in a section of its own name is titled by the section, and an aspect's fold
  heads its methods apart from the page's;
- a case row's trial ids are in their own order, and two replays of one recording agree on every case;
- a measurement's fold opens on its count; a lever run at one level, or too thinly, is said in words; how a reading
  takes a margin is said in words; a page with no analysis says how one is written;
- a report document's masthead dates in the one form every page dates in, and names its aspect once;
- no review finding prints an engine key, dotted or with its value, or a claim's id;
- a settle table says which side of its line today's estimate is on in a sentence, never the enum's word;
- a draft is listed by its own headline, marked a draft, and no listing line holds a cut question.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import html
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_inputs.dataset import kairo_bundle
from packages.evals.tests.doc_writer_support import (
    TOY_DOC,
    FakeAnalystClient,
    passing_review,
    scripted_reply,
    submission,
)
from packages.evals.tests.factories import fixture_variant_key
from packages.evals.tests.ladder_support import load, offline_copy
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, WRITER, ops_fixture, settled
from packages.evals.tests.report_doc_support import REPORT_DOCS, converted_doc, snapshot_of
from threetears.evals.analysis import (
    build_code_only_report,
    campaign_report_html,
    campaign_report_markdown,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.report.build import NO_ANALYSIS
from threetears.evals.analysis.campaign_title import arms_line, declared_title
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.doc_page.model import masthead_parts
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report.doc_page.model import DocFinding
from threetears.evals.analysis.report.figure_keys import is_figure_key, render
from threetears.evals.analysis.report.model import (
    CampaignReport,
    DisclosureBlock,
    Report,
    ReportBlock,
    TableBlock,
    TrialBlock,
)
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.style import CANNOT_TELL_APART, CHECK_LINK_TITLE, GLOSSARY, lint_prose
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm
from threetears.evals.kernel.report_doc import ReportDoc, ReportFinding
from threetears.evals.ops import WriterLimits, analysis_generate, campaigns_list
from threetears.evals.ops.analysis import QUESTION_SHORT_CHARS, CampaignDefinition, campaign_create

#: The rungs held here: one arm named as its campaign is (rung 1), a judge (rung 2), a classifier with a guardrail
#: (rung 3), each with the file declaring its eval and the attribute holding it.
_RUNGS: dict[str, tuple[str, str, str]] = {
    "rung1": ("1-triage", "triage.py", "triage_eval"),
    "rung2": ("2-replies", "replies.py", "replies_eval"),
    "rung3": ("3-invoices", "extract.py", "invoices"),
}


@dataclass(frozen=True)
class _Rung:
    """One rung replayed offline: its campaign's page, rendered."""

    page: CampaignReport
    report: Report
    markdown: str
    html: str


def _replayed(name: str, tmp_path_factory: pytest.TempPathFactory) -> _Rung:
    directory, file, attribute = _RUNGS[name]
    with pytest.MonkeyPatch.context() as monkeypatch:
        copy = offline_copy(directory, tmp_path_factory.mktemp(name), monkeypatch)
        runnable = getattr(load(copy / file, Path(file).stem, monkeypatch), attribute)
        result = asyncio.run(runnable.run(store=":memory:"))
    page = result.page
    return _Rung(
        page=page, report=result.report, markdown=campaign_report_markdown(page), html=campaign_report_html(page)
    )


@pytest.fixture(scope="module")
def rungs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, _Rung]]:
    """Each rung replayed once, and rung 3 a second time, from its recordings, with no key."""
    replayed = {name: _replayed(name, tmp_path_factory) for name in _RUNGS}
    replayed["rung3-again"] = _replayed("rung3", tmp_path_factory)
    yield replayed


def _blocks(page: CampaignReport) -> list[ReportBlock]:
    """Every block a campaign's page lays out: its own, then each fold's."""
    return [*page.blocks, *(block for fold in page.aspects for block in fold.report.blocks)]


#: The tables whose cells are sentences: what the page discloses of its setup and how it was made.
_PROSE_TABLES = frozenset({"test_setup", "criteria_set", "methods", "no_margin"})


def _prose(rung: _Rung) -> list[str]:
    """Every sentence a rung prints: each disclosure on its campaign page, its folds and its aspect's own page, and
    each cell of a table whose cells are sentences."""
    blocks = [*_blocks(rung.page), *rung.report.blocks]
    said = [block.text for block in blocks if isinstance(block, DisclosureBlock)]
    said += [
        cell.text
        for block in blocks
        if isinstance(block, TableBlock) and block.name in _PROSE_TABLES
        for row in block.rows
        for cell in row.values()
    ]
    return said


def _arms(page: CampaignReport) -> list[str]:
    return list(page.readings.arm_labels().values())


def _named(name: str, text: str) -> int:
    """How many times ``name`` stands in ``text`` as a whole word."""
    return len(re.findall(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", text))


# --- titles --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_a_page_heads_with_the_campaign_s_name_and_names_each_arm_once_under_it(
    rungs: dict[str, _Rung], name: str
) -> None:
    rung = rungs[name]
    heading = rung.markdown.splitlines()[0]
    assert heading == f"# {rung.page.campaign_name}", "the heading is the campaign's name, never a stack of names"
    assert f"<title>{html.escape(rung.page.campaign_name or '')}</title>" in rung.html
    arms_lines = [
        block.text for block in rung.page.blocks if isinstance(block, DisclosureBlock) and block.source == "arms"
    ]
    for arm in _arms(rung.page):
        assert _named(arm, heading) <= 1, (arm, heading)
        assert all(_named(arm, line) <= 1 for line in arms_lines), (arm, arms_lines)
    # An engine-given aspect label never heads a fold or a cell: the campaign's name stands for it.
    assert not re.search(r"^## (classify|score)$", rung.markdown, flags=re.MULTILINE)


def _titled(name: str, control: list[str], candidates: list[str], aspects: list[tuple[str, str | None]]) -> str:
    """The listing title over arms of these labels, read through the title a frozen bundle's arms are given."""
    arms = [
        CampaignArm(label=label, role=role, variant_key=fixture_variant_key(label))
        for role, labels in (("control", control), ("candidate", candidates))
        for label in labels
    ]
    return declared_title(name, arms, aspects)


def test_a_listing_title_names_each_part_once() -> None:
    # An arm named as its campaign is, on an aspect the engine labelled: the name alone.
    assert _titled("triage", [], ["triage"], [("classify", None)]) == "triage"
    assert _titled("invoices", ["v1"], ["v2"], [("classify", None)]) == "v1 vs v2 (invoices)"
    assert (
        _titled("support queue", ["production"], ["v2"], [("billing", None), ("tone", None)])
        == "production vs v2 · billing, tone (support queue)"
    )
    # A name that names every arm (Kairo's round 2) is not followed by the arms again.
    kairo = "Kairo round 2: September vs qwen3.8-flash and 1017 on character v2"
    title = _titled(kairo, ["September"], ["qwen3.8-flash", "1017"], [("character adherence (v2)", None)])
    assert title == f"{kairo} · character adherence (v2)"
    for arm in ("September", "qwen3.8-flash", "1017"):
        assert _named(arm, title) == 1
    assert arms_line("invoices", control=["v1"], candidates=["v2"], aspects=[("classify", None)]) == "v1 vs v2"


# --- no key=value in prose -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_no_prose_line_on_a_page_is_a_key_and_its_value(rungs: dict[str, _Rung], name: str) -> None:
    said = _prose(rungs[name])
    found = {text: hits for text in said if (hits := [v for v in lint_prose(text) if v.rule == "key_value"])}
    assert not found, found
    # The judge's lines say its settings in words, and each judge that reads alike once.
    own = [block.text for block in rungs[name].report.blocks if isinstance(block, DisclosureBlock)]
    judged = [text for text in own if text.startswith("Judge: ")]
    assert len(judged) == len(set(judged))
    assert all("{" not in text and "builtin:" not in text for text in judged), judged


# --- glossary ------------------------------------------------------------------------------------------------------


#: A tag in the page's markup, and the elements a glossary term is never linked inside (a link, code, a heading, a
#: fold's summary, a table, the navigation, a chart, the head, the quoted question).
_TAG = re.compile(r"(<[^>]*>)")
_TAG_NAME = re.compile(r"</?\s*([a-zA-Z][a-zA-Z0-9]*)")
_LINK_TITLE = re.compile(r'<a[^>]*title="(?P<title>[^"]*)"[^>]*>')
_UNREAD = frozenset(
    {"a", "code", "pre", "summary", "h1", "h2", "h3", "h4", "h5", "h6", "th", "td", "nav", "svg", "script", "style"}
    | {"title", "question"}
)
_VOID = frozenset({"meta", "br", "hr", "img", "input", "link", "wbr"})


def _first_uses(page: str) -> dict[str, str]:
    """Each glossary term the page defines, by how the page first meets it: ``linked`` where a link to it comes
    before or at its first use in text a reader reads, ``bare`` where its words stand unlinked first, ``unused``
    where its words never stand in such text."""
    body, _, glossary = page.partition('<section id="glossary">')
    defined = [html.unescape(term) for term in re.findall(r'<dt id="g-[^"]*">([^<]*)</dt>', glossary)]
    # A term of art's link carries its sentence; a goal check's name (defined by the page, not GLOSSARY) carries one
    # hover for every check, so its link is read by the name it shows.
    by_definition = {GLOSSARY[term]: term for term in defined if term in GLOSSARY}
    seen: dict[str, str] = {}
    opened: list[str] = []
    naming = False
    for token in _TAG.split(body):
        if naming and not token.startswith("<"):
            if (shown := html.unescape(token)) in defined:
                seen.setdefault(shown, "linked")
            naming = False
        if token.startswith("<"):
            found = _TAG_NAME.match(token)
            if found is None:
                continue
            name = found.group(1).lower()
            if token.startswith("</"):
                while opened and opened.pop() not in (name, *(("question", "verdict") if name == "p" else ())):
                    pass
                continue
            if name == "a" and (title := _LINK_TITLE.match(token)) is not None:
                said = html.unescape(title.group("title"))
                term = by_definition.get(said)
                if term is not None:
                    seen.setdefault(term, "linked")
                naming = said == CHECK_LINK_TITLE
            if name not in _VOID and not token.endswith("/>"):
                # The declared question is the author's words, quoted whole and never linked.
                # The verdict line links no term of art but the undecided reading: no other is first used there.
                quoted = "question" if 'class="question"' in token else "verdict" if 'class="verdict"' in token else ""
                opened.append(quoted or name)
            continue
        if any(name in _UNREAD for name in opened):
            continue
        text = html.unescape(token)
        for term in [
            term for term in defined if "verdict" not in opened or term == CANNOT_TELL_APART or term not in GLOSSARY
        ]:
            if term not in seen and re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text, re.IGNORECASE):
                seen[term] = "bare"
    return {term: seen.get(term, "unused") for term in defined}


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_every_glossary_term_on_a_page_is_linked_at_its_first_use(rungs: dict[str, _Rung], name: str) -> None:
    uses = _first_uses(rungs[name].html)
    assert uses, "the page has a glossary"
    assert [term for term, use in uses.items() if use == "bare"] == []
    # The terms the page itself uses in prose, the undecided reading among them, are linked where first read.
    if name == "rung3":
        for term in ("interval", "margin", "cannot tell apart", "guardrail"):
            assert uses.get(term) == "linked", (term, uses.get(term))


@functools.cache
def _kairo() -> Report:
    """Kairo round 2's aspect page, from the committed capture."""
    (aspect,) = kairo_bundle().aspects
    return build_code_only_report(aspect, assembled_at="2026-10-31T00:00:00+00:00")


def test_the_kairo_page_links_every_term_at_its_first_use() -> None:
    assert [term for term, use in _first_uses(report_html(_kairo())).items() if use == "bare"] == []


def test_the_lead_links_the_undecided_reading_and_the_verdict_header_links_verdict(rungs: dict[str, _Rung]) -> None:
    page = rungs["rung3"].html
    header = page[page.index("<header>") : page.index("</header>")]
    (lead,) = re.findall(r'<p class="verdict"[^>]*>.*?</p>', header, flags=re.DOTALL)
    undecided = html.escape(GLOSSARY["cannot tell apart"])
    assert re.search(rf'<a href="#[^"]*" title="{re.escape(undecided)}">cannot tell v2 and v1 apart</a>', lead), lead
    headers = re.findall(r"<th[^>]*>(.*?)</th>", page, flags=re.DOTALL)
    verdicts = [cell for cell in headers if re.sub(r"<[^>]+>", "", cell).strip() == "Verdict"]
    assert verdicts and all(f'title="{html.escape(GLOSSARY["verdict"])}"' in cell for cell in verdicts)


def test_the_verdict_line_links_no_term_but_the_undecided_reading(rungs: dict[str, _Rung]) -> None:
    """Discodon's 0.94.0 pass: a one-arm page's bold verdict ("deepseek-v4.1-flash passed 152 of 180 trials on 60
    cases") linked "passed" and "cases", a link inside the page's answer. The verdict links no term of art but the
    undecided reading (rung 3's, which the lead must define where it is first read); each other term is linked at its
    next use. A goal check's name there still links the check's entry
    (``test_a_lead_s_name_links_its_glossary_entry_and_no_attribute_carries_the_expression``)."""
    page = rungs["rung1"].html
    header = page[page.index("<header>") : page.index("</header>")]
    verdicts = re.findall(r'<p class="verdict"[^>]*>.*?</p>', header, flags=re.DOTALL)
    assert verdicts and any(" passed " in re.sub(r"<[^>]+>", "", verdict) for verdict in verdicts), verdicts
    definitions = {html.escape(said) for said in GLOSSARY.values()}
    assert not [
        link for verdict in verdicts for link in re.findall(r'<a [^>]*title="([^"]*)"', verdict) if link in definitions
    ], verdicts
    assert _first_uses(page).get("passed") == "linked", "passed is linked where it is next read"
    undecided = html.escape(GLOSSARY[CANNOT_TELL_APART])
    rung3 = rungs["rung3"].html
    (lead,) = re.findall(r'<p class="verdict"[^>]*>.*?</p>', rung3[: rung3.index("</header>")], flags=re.DOTALL)
    links = [link for link in re.findall(r'<a [^>]*title="([^"]*)"', lead) if link in definitions]
    assert links and set(links) == {undecided}, links


def test_the_first_screen_s_judge_line_links_the_judged_scale(rungs: dict[str, _Rung]) -> None:
    page = rungs["rung2"].html
    header = page[page.index("<header>") : page.index("</header>")]
    (line,) = re.findall(r'<p[^>]*id="caveat-judge"[^>]*>.*?</p>', header, flags=re.DOTALL)
    assert f'title="{html.escape(GLOSSARY["judged score"])}"' in line


@pytest.mark.parametrize("name", ["rung2", "rung3"])
def test_a_measure_a_table_names_by_its_reader_name_is_defined(rungs: dict[str, _Rung], name: str) -> None:
    page = rungs[name].page
    assert "Measuring spend, all roles" in rungs[name].markdown
    assert "measuring spend, all roles" in page.terms
    assert f"- **measuring spend, all roles**: {GLOSSARY['measuring spend, all roles']}" in rungs[name].markdown


def test_the_judged_scale_is_stated_once_in_the_glossary(rungs: dict[str, _Rung]) -> None:
    page = rungs["rung2"].markdown
    assert "judged score" in rungs["rung2"].page.terms
    assert page.count("from 1 to 5") == 1
    assert f"- **judged score**: {GLOSSARY['judged score']}" in page


# --- headings once -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_no_heading_is_printed_twice(rungs: dict[str, _Rung], name: str) -> None:
    # One page of one aspect heads each section once: its fold heads its methods as the aspect's.
    headings = [line.lstrip("#").strip() for line in rungs[name].markdown.splitlines() if re.match(r"^#{1,6} ", line)]
    assert len(headings) == len(set(headings)), [heading for heading in headings if headings.count(heading) > 1]
    pages = [rungs[name].markdown, report_markdown(rungs[name].page.aspects[0].report)]
    for page in pages:
        heading: str | None = None
        for line in page.splitlines():
            if found := re.match(r"^#{2,6} (.+)$", line):
                heading = found.group(1)
            elif heading is not None and line == f"**{heading}**":
                pytest.fail(f"{heading!r} is printed as its section's heading and again as a table's title")


# --- order ---------------------------------------------------------------------------------------------------------


def _cases_table(page: str) -> list[str]:
    start = page.index("| Case | Flag |")
    return page[start : page.index("\n\n", start)].splitlines()[2:]


def _row_counts(row: str) -> tuple[str, ...]:
    """A case row with each arm's marks counted rather than ordered: what two replays of one recording must agree on."""
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    return tuple(
        "".join(sorted(cell.split(" (")[0].split())) + (f" ({cell.split(' (', 1)[1]}" if " (" in cell else "")
        if set(cell.split(" (")[0].split()) <= {"✓", "✗", "–"}
        else cell
        for cell in cells
    )


def test_a_case_row_s_trial_ids_are_in_order_and_two_replays_lay_the_cases_out_alike(
    rungs: dict[str, _Rung],
) -> None:
    """Ids in their own order, and one bundle laid out byte for byte alike every time. Two replays of one recording
    agree on every case, its ids and how many of each mark; which repeat of a case failed is the replay's (its calls
    of one case ask alike, and run concurrently), so the order of a row's marks is the evidence's, not the page's."""
    page = rungs["rung3"].page
    rows = _cases_table(rungs["rung3"].markdown)
    assert rows
    for row in rows:
        ids = row.rstrip(" |").rsplit("|", 1)[1].split()
        numbers = [int(trial.removeprefix("T")) for trial in ids]
        assert numbers == sorted(numbers), row
    assert campaign_report_markdown(page) == rungs["rung3"].markdown, "one page renders alike every time"
    again = _cases_table(rungs["rung3-again"].markdown)
    assert [_row_counts(row) for row in rows] == [_row_counts(row) for row in again]


# --- report documents ----------------------------------------------------------------------------------------------


def _record(name: str) -> dict[str, object]:
    record: dict[str, object] = json.loads((REPORT_DOCS / name / "record.json").read_text(encoding="utf-8"))
    return record


def _input(name: str) -> ReportInput:
    return ReportInput.model_validate(snapshot_of(name).report_input)


def test_a_document_s_masthead_dates_in_one_form_and_names_its_aspect_once() -> None:
    """A reader's page dates the run as every page dates (``Run 25 Oct 2026``) and names each part once; the masthead
    the editor and the judge read (``doc_page_text``) stays as recorded reviews were keyed on it."""
    for name in ("prototype5/kairo", "prototype2/s2"):
        record_input = _input(name)
        doc = ReportDoc.model_validate(_record(name)["doc"]) if name.startswith("prototype5") else None
        page = build(doc if doc is not None else _converted(name), record_input, strict=False)
        parts = masthead_parts(page.masthead)
        assert len(parts) == len(set(parts)), parts
        (run,) = [part for part in parts if part.startswith("Run ")]
        assert re.fullmatch(r"Run \d{1,2} [A-Z][a-z]{2} \d{4}", run), run
        assert " · ".join(parts) in doc_page_markdown(page)
        assert all(f">{html.escape(part)}<" in doc_page_html(page) for part in parts[2:])
        assert page.title == record_input.meta.campaign, "the page's name is the campaign's"
    # The editor's view is the masthead as recorded: its date as stored.
    kairo = build(
        ReportDoc.model_validate(_record("prototype5/kairo")["doc"]), _input("prototype5/kairo"), strict=False
    )
    assert re.search(r"Run \d{4}-\d{2}-\d{2}", doc_page_text(kairo))


def test_a_document_s_masthead_opens_on_the_campaign_s_name() -> None:
    """Discodon's 0.94.0 pass: a written report's masthead named the kind, the system and the counts but never the
    campaign, which only the browser's tab carried. Its first line is the campaign's name, as a code page's title is,
    on its own line; the kind and the counts follow."""
    for name in ("prototype5/kairo", "prototype2/s2"):
        record_input = _input(name)
        doc = ReportDoc.model_validate(_record(name)["doc"]) if name.startswith("prototype5") else _converted(name)
        page = build(doc, record_input, strict=False)
        campaign = record_input.meta.campaign
        assert campaign and campaign == page.title, "the control: the campaign has a name"
        written = doc_page_html(page)
        mast = written[
            written.index('<div class="mast">') : written.index("</div>", written.index('<div class="mast">'))
        ]
        spans = re.findall(r'<span(?: class="([^"]*)")?>(.*?)</span>', mast)
        assert spans[0] == ("name", html.escape(campaign)), spans
        assert spans[1][0] == "kind", spans


def _converted(name: str) -> ReportDoc:
    return ReportDoc.model_validate(converted_doc(name))


def test_no_review_finding_prints_an_engine_key_or_a_claim_s_id() -> None:
    name = "prototype5/kairo"
    record = _record(name)
    report_input = _input(name)
    doc = ReportDoc.model_validate(record["doc"])
    key = next(
        key for key in ("control.name", "aspect.name", "meta.campaign", "meta.run_date") if _resolves(key, report_input)
    )
    value = render(f"{{{key}}}", report_input, {})
    planted = ReportFinding.model_validate(
        {
            "claim": doc.claims[0].id,
            "severity": "should_fix",
            "finding": f"The page reads `{key} = 62` beside {doc.claims[0].id}'s block.",
        }
    )
    held = record["findings"]
    assert isinstance(held, list)
    findings = [*(ReportFinding.model_validate(finding) for finding in held), planted]
    page = build(doc, report_input, strict=False, findings=findings, draft=True)
    claim_ids = {claim.id for claim in doc.claims}
    for finding in [*page.findings, *_recorded_findings()]:
        assert not [v for v in lint_prose(finding.said) if v.rule == "key_value"], finding.said
        # No dotted engine key, with its value or without: a key that names no figure here reads in words.
        assert not [
            word for word in re.findall(r"[a-z_][\w<>*-]*(?:\.[\w<>*-]+)+", finding.said) if is_figure_key(word)
        ], finding.said
        assert not any(
            re.search(rf"\b{re.escape(claim)}\b", finding.said) for claim in claim_ids if re.fullmatch(r"c\d+", claim)
        ), finding.said
    said = page.findings[-1].said
    assert value in said and "claim 1's block" in said, said
    for form in (doc_page_html(page), doc_page_markdown(page)):
        assert f"{key} = 62" not in form
    # The editor and the judge read the finding as it was written, so a recorded review keeps its key.
    assert f"{key} = 62" in doc_page_text(page)


def _recorded_findings() -> list[DocFinding]:
    """Every review finding the stored prototype 5 records leave open, as each page prints it."""
    shown: list[DocFinding] = []
    for record_path in sorted(REPORT_DOCS.glob("prototype5/*/record.json")):
        name = f"prototype5/{record_path.parent.name}"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        try:
            report_input = _input(name)
        except KeyError:
            # This record's snapshot is not among the committed ones; the others carry its kinds of finding.
            continue
        findings = [ReportFinding.model_validate(finding) for finding in record["findings"]]
        page = build(ReportDoc.model_validate(record["doc"]), report_input, strict=False, findings=findings)
        shown += page.findings
    assert any("settle." in finding.finding for finding in shown), "a recorded finding quotes a settle key"
    return shown


def _resolves(key: str, report_input: ReportInput) -> bool:
    try:
        render(f"{{{key}}}", report_input, {})
    except UnresolvableReference:
        return False
    return True


def test_a_settle_table_never_prints_which_side_of_its_line_today_s_estimate_is_on_by_the_enum_s_word() -> None:
    # The settle table holds four columns (chunk 7: `Arm`, `Against` and its two sizes) and no `Settles` column, so
    # no cell of it prints `clears`, `misses` or `on_the_line`.
    name = "prototype5/r2-with-one-gate"
    record = _record(name)
    markdown = doc_page_markdown(build(ReportDoc.model_validate(record["doc"]), _input(name), strict=False))
    lines = markdown.splitlines()
    cells: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith("| Arm | Against |"):
            for row in lines[index + 2 :]:
                if not row.startswith("|"):
                    break
                cells += [cell.strip() for cell in row.strip("|").split("|")]
    assert cells, "the document draws a settle table"
    assert not set(cells) & {"clears", "misses", "on_the_line"}, set(cells)


# --- the listing ---------------------------------------------------------------------------------------------------


async def test_a_draft_is_listed_by_its_own_headline_marked_a_draft() -> None:
    fixture = off_loop(ops_fixture)
    # A must_fix: a round-1 should_fix left open no longer makes a draft (owner ruling 2026-11-09).
    open_finding = {"claim": "report", "severity": "must_fix", "finding": "Give the cost column its dollars."}
    script = {
        "analyst": [scripted_reply(submission())],
        "editor-1": [scripted_reply(passing_review(findings=[open_finding]))],
    }
    object.__setattr__(
        fixture.host.eval_host, "clients", lambda role, model, *, temperature=None: FakeAnalystClient(script)
    )
    assert fixture.host.generation is not None
    host = dataclasses.replace(
        fixture.host, generation=dataclasses.replace(fixture.host.generation, limits=WriterLimits(revisions=0))
    )
    (job,) = (await analysis_generate(host, fixture.campaign.id, TOYHOST_SCOPE, writer=WRITER)).jobs
    status = await settled(host, job.job_id)
    assert status.status == "stored", status
    (line,) = [
        line
        for line in off_loop(campaigns_list, host.eval_host, TOYHOST_SCOPE).campaigns
        if line.id == fixture.campaign.id
    ]
    assert line.report is not None and line.report.draft
    assert line.report.headline == f"Draft: {TOY_DOC['headline']}"
    # The row leads with the short verdict (the host's re-shoot, 0.93), so it is marked as the headline is.
    assert line.verdict_short is not None and line.verdict_short.startswith("Draft: "), line.verdict_short
    assert line.verdict_short.removeprefix("Draft: ").split(" ", 1)[0] in {"Keep", "Adopt", "No"}, line.verdict_short


def _declared_aspect(campaign: EvalCampaign) -> dict[str, Any]:
    """The fixture campaign's one aspect as an author declares it on a new campaign: its design, and no run."""
    (aspect,) = campaign.aspects
    assert aspect.design is not None
    return {
        "label": aspect.label,
        "run_ids": [],
        "design": aspect.design.model_dump(mode="json", exclude={"merit_axes"}),
    }


def test_no_listing_line_holds_a_cut_question() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    long = " ".join(["Can the cheaper model replace the one in production without passing character"] * 3)
    assert len(long) > QUESTION_SHORT_CHARS
    campaign_create(
        host,
        CampaignDefinition(
            name="long question",
            subject_id=TOYHOST_SUBJECT.subject_id,
            behavior=fixture.campaign.behavior,
            question=long,
            aspects=[_declared_aspect(fixture.campaign)],
        ),
        TOYHOST_SCOPE,
        created_by=CALLER.identity,
    )
    lines = campaigns_list(host, TOYHOST_SCOPE).campaigns
    assert any(line.question == long for line in lines)
    for line in lines:
        assert "…" not in line.question_short
        assert line.question_short in ("", " ".join(line.question.split()))


def test_a_trial_s_world_state_is_counted_and_folded_never_printed_open(rungs: dict[str, _Rung]) -> None:
    """A host that writes the case's world into what it asked (``## <part>`` and its JSON) has it counted and folded
    under the case's name, the case's own words kept open."""
    bundle = rungs["rung3"].page.aspects[0].report.bundle
    world = '## music\n{\n  "catalog": [{"artist": "Two Shell"}]\n}\n'
    trials = [
        trial.model_copy(update={"first_input": f"The world as it stood:\n\n{world}## Jules\nwhat time is it?"})
        for trial in bundle.trials
    ]
    page = build_code_only_report(
        bundle.model_copy(update={"trials": trials}), assembled_at="2026-11-07T00:00:00+00:00", trials="all"
    )
    records = [block for block in page.blocks if isinstance(block, TrialBlock)]
    assert records
    for record in records:
        assert record.world_parts == ["music"] and '"catalog"' in record.world
        assert '"catalog"' not in record.first_input and "what time is it?" in record.first_input
    markdown, written = report_markdown(page), report_html(page)
    assert "World state: 1 part (music), folded:" in markdown
    assert f"<summary>{records[0].case_label}'s world state</summary>" in markdown
    assert f"{html.escape(records[0].case_label)}'s world state" in written
    # Opened nowhere: every copy of the state sits inside its fold.
    assert markdown.count('"catalog"') == markdown.count("'s world state</summary>")
    for said in (markdown, written):
        assert "result_get" not in said


# --- words for what a page cannot test, and how it was made ---------------------------------------------------------


def test_a_list_of_unchecked_settings_prints_once_for_the_arms_it_covers() -> None:
    page = report_markdown(_kairo())
    (sentence,) = [line for line in page.splitlines() if line.startswith("A per-reply cost is what production")]
    assert sentence.count("left unchecked whether") == 1 and "every arm left unchecked whether" in sentence
    # Each arm's own move follows, one short clause each, its setting said as words.
    assert "arm September set model to deepseek/deepseek-v4.1-flash" in sentence
    assert "=" not in sentence


def test_a_measurement_s_fold_opens_on_its_count_never_a_bare_word(rungs: dict[str, _Rung]) -> None:
    for name in ("rung1", "rung2"):
        (fold,) = rungs[name].page.aspects
        assert re.fullmatch(r".+ passed \d+ of \d+ trials on \d+ cases", fold.shows), fold.shows
        assert f"## {fold.heading}\n\n{fold.shows}.\n" in rungs[name].markdown


def test_a_lever_run_at_one_level_or_too_thinly_is_said_in_words(rungs: dict[str, _Rung]) -> None:
    report = rungs["rung3"].report
    said = [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]
    assert "One model ran, so this page says nothing of changing the model." in said
    assert not any("unswept" in text or "Lever " in text for text in said)
    bundle = report.bundle
    thin = bundle.model_copy(
        update={"coverage": [row.model_copy(update={"status": "thin"}) for row in bundle.coverage]}
    )
    rebuilt = build_code_only_report(thin, assembled_at="2026-11-07T00:00:00+00:00")
    texts = [block.text for block in rebuilt.blocks if isinstance(block, DisclosureBlock)]
    assert "Each model ran too few repeats to read what changing the model does." in texts


def _no_margin(report: Report) -> dict[str, str]:
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "no_margin"]
    return {row["reading"].text: row["declare"].text for row in table.rows}


def test_how_a_reading_takes_a_margin_is_said_in_words(rungs: dict[str, _Rung]) -> None:
    rung3 = _no_margin(rungs["rung3"].report)
    assert rung3["Due score"] == "a margin on its grader"
    assert rung3["Accuracy"] == "a margin on the classifier's field's grader, on every arm's run"
    assert rung3["Production cost"] == "cannot: the engine owns its description"
    # A goal check takes its margin on the check; a graded reading with no declared range is told to declare one.
    assert "a margin on its check" in _no_margin(_kairo()).values()
    bundle = rungs["rung3"].report.bundle
    catalog = {
        name: described.model_copy(update={"family": "mechanical", "value_range": None}) if name == "due" else described
        for name, described in bundle.measure_catalog.items()
    }
    ranged = build_code_only_report(
        bundle.model_copy(update={"measure_catalog": catalog}), assembled_at="2026-11-07T00:00:00+00:00"
    )
    assert _no_margin(ranged)["Due score"] == "a margin on its grader, with the range its values take"
    for advice in (*rung3.values(), *_no_margin(_kairo()).values()):
        assert "grader(" not in advice and "margin=" not in advice


def test_a_page_with_no_analysis_says_how_one_is_written(rungs: dict[str, _Rung]) -> None:
    assert NO_ANALYSIS == "none written yet; generating an analysis writes one from this same evidence"
    assert f"| Analysis | {NO_ANALYSIS} |" in rungs["rung3"].markdown
    assert "analysis_generate" not in rungs["rung3"].markdown
