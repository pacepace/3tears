"""Each name once, each term defined where it is first used, each heading once, and one order on every replay
(presentation build, chunk 4).

The review's A3 term clause holds a page to terms defined before use and one name per thing. Held here on the
ladder's replayed rungs (offline, no key), the stored report documents and the listing:

- no page heading names an arm twice: a page heads with the campaign's name, its arms the line under it, and a
  listing's title names each part once (P10, the host's finding 6a);
- no prose line a page prints is a ``key=value`` (P20), read through :func:`~threetears.evals.analysis.style.lint_prose`;
- every glossary term a page defines is linked from the page, and a header naming a term links it the first time it
  is named (P21, P22); the judged scale is stated once, in the glossary;
- no heading is printed twice: a table in a section of its own name is titled by the section (P23);
- a case row's trial ids are in their own order, and two replays of one recording lay the cases out byte for byte
  alike (P28);
- a report document's masthead dates in the one form every page dates in, and names its aspect once (P42, P37);
- no review finding prints an engine key or a claim's id (the host's written-report finding 9);
- a settle table says which side of its line today's estimate is on in a sentence, never the enum's word;
- a draft is listed by its own headline, marked a draft, and no listing line holds a cut question (finding 14).
"""

from __future__ import annotations

import asyncio
import dataclasses
import html
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.doc_writer_support import (
    TOY_DOC,
    FakeAnalystClient,
    passing_review,
    scripted_reply,
    submission,
)
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
from threetears.evals.analysis.analyst.settle import SETTLES_WORDS
from threetears.evals.analysis.campaign_title import arms_line, named_title
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.doc_page.model import masthead_parts
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report.model import (
    CampaignReport,
    DisclosureBlock,
    Report,
    ReportBlock,
    TableBlock,
    TrialBlock,
)
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.style import GLOSSARY, lint_prose
from threetears.evals.kernel.campaign import EvalCampaign
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


def test_a_listing_title_names_each_part_once() -> None:
    # An arm named as its campaign is, on an aspect the engine labelled: the name alone.
    assert named_title("triage", control=[], candidates=["triage"], aspects=[("classify", None)]) == "triage"
    assert (
        named_title("invoices", control=["v1"], candidates=["v2"], aspects=[("classify", None)])
        == "v1 vs v2 (invoices)"
    )
    assert (
        named_title(
            "support queue", control=["production"], candidates=["v2"], aspects=[("billing", None), ("tone", None)]
        )
        == "production vs v2 · billing, tone (support queue)"
    )
    # A name that names every arm (Kairo's round 2) is not followed by the arms again.
    kairo = "Kairo round 2: September vs qwen3.8-flash and 1017 on character v2"
    title = named_title(
        kairo, control=["September"], candidates=["qwen3.8-flash", "1017"], aspects=[("character adherence (v2)", None)]
    )
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


#: A header cell as the page writes it: its text, linked or not, and the term its link defines (by its title).
_HEADER = re.compile(r"<th[^>]*>(?P<inner>.*?)</th>", re.DOTALL)
_LINK_TITLE = re.compile(r'<a[^>]*title="(?P<title>[^"]*)"[^>]*>')
_TAG = re.compile(r"<[^>]+>")


def _terms_by_definition() -> dict[str, str]:
    return {html.escape(definition): term for term, definition in GLOSSARY.items()}


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_every_glossary_term_on_a_page_is_linked_and_a_header_links_it_where_first_named(
    rungs: dict[str, _Rung], name: str
) -> None:
    page = rungs[name].html
    body, _, glossary = page.partition('<section id="glossary">')
    defined = re.findall(r'<dt id="g-[^"]*">([^<]*)</dt>', glossary)
    assert defined, "the page has a glossary"
    by_definition = _terms_by_definition()
    linked = {by_definition[title] for title in _LINK_TITLE.findall(body) if title in by_definition}
    assert {html.unescape(term) for term in defined} <= linked, "a term the glossary defines is linked from the page"
    first: dict[str, bool] = {}
    for header in _HEADER.finditer(body):
        inner = header.group("inner")
        text = html.unescape(_TAG.sub("", inner)).strip().casefold()
        term = next((term for term in GLOSSARY if term.casefold() == text), None)
        if term is not None and term not in first:
            titles = [html.unescape(title) for title in _LINK_TITLE.findall(inner)]
            first[term] = GLOSSARY[term] in titles
    assert all(first.values()), {term: ok for term, ok in first.items() if not ok}
    # A header naming a cost per reply links the term in the words the header says.
    for header in _HEADER.finditer(body):
        inner = header.group("inner")
        text = html.unescape(_TAG.sub("", inner)).strip()
        if text.startswith("Cost per "):
            assert GLOSSARY[text.casefold()] in [html.unescape(title) for title in _LINK_TITLE.findall(inner)], text


def test_the_judged_scale_is_stated_once_in_the_glossary(rungs: dict[str, _Rung]) -> None:
    page = rungs["rung2"].markdown
    assert "judged score" in rungs["rung2"].page.terms
    assert page.count("from 1 to 5") == 1
    assert f"- **judged score**: {GLOSSARY['judged score']}" in page


# --- headings once -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_no_heading_is_printed_twice(rungs: dict[str, _Rung], name: str) -> None:
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
    for finding in page.findings:
        assert not [v for v in lint_prose(finding.said) if v.rule == "key_value"], finding.said
        assert not re.search(r"\b[a-z_][\w-]*\.[\w.-]+\s*=", finding.said), finding.said
        assert not any(
            re.search(rf"\b{re.escape(claim)}\b", finding.said) for claim in claim_ids if re.fullmatch(r"c\d+", claim)
        ), finding.said
    said = page.findings[-1].said
    assert value in said and "claim 1's block" in said, said
    for form in (doc_page_html(page), doc_page_markdown(page)):
        assert f"{key} = 62" not in form
    # The editor and the judge read the finding as it was written, so a recorded review keeps its key.
    assert f"{key} = 62" in doc_page_text(page)


def _resolves(key: str, report_input: ReportInput) -> bool:
    try:
        render(f"{{{key}}}", report_input, {})
    except UnresolvableReference:
        return False
    return True


def test_a_settle_table_says_which_side_of_its_line_today_s_estimate_is_on_in_a_sentence() -> None:
    name = "prototype5/r2-with-one-gate"
    record = _record(name)
    markdown = doc_page_markdown(build(ReportDoc.model_validate(record["doc"]), _input(name), strict=False))
    lines = markdown.splitlines()
    cells: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith("|") and "| Settles |" in line:
            column = [cell.strip() for cell in line.strip("|").split("|")].index("Settles")
            for row in lines[index + 2 :]:
                if not row.startswith("|"):
                    break
                cells.append([cell.strip() for cell in row.strip("|").split("|")][column])
    assert cells, "the document draws a settle table"
    assert set(cells) <= {*SETTLES_WORDS.values(), "—"}, set(cells)


# --- the listing ---------------------------------------------------------------------------------------------------


async def test_a_draft_is_listed_by_its_own_headline_marked_a_draft() -> None:
    fixture = off_loop(ops_fixture)
    open_finding = {"claim": "report", "severity": "should_fix", "finding": "Give the cost column its dollars."}
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
