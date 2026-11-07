"""One result, one reading, one sentence, on every surface a reader meets it.

A first-time reader took "not separated" for "the two are roughly equal", and small figures beside an undecided
reading for "the same". Held here on the ladder's replayed rungs, offline, and on Kairo round 2's committed bundle:

- the campaign page's lead, the aspect page's verdict, the print's first line and the exit line carry one sentence:
  the decision, what each arm passed, and what the interval allows the candidate to be, in points (rungs 3 and 4);
- a lead on a row under a margin states the count its To settle row states, and no other lead counts cases to settle;
- no surface code writes prints the retired phrase ("not separated", "nothing separated"); the undecided reading is
  :data:`~threetears.evals.analysis.style.CANNOT_TELL_APART`, defined once in the glossary and linked at its first use;
- each guardrail and measure reads one name on every surface, and the undecided guardrail's line says what it means
  for the decision before the count of cases that could fail it;
- rung 4's page and print say one thing of a goal check a candidate that did nothing could pass.
"""

from __future__ import annotations

import asyncio
import re
from html import escape
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from threetears.evals.analysis import (
    GLOSSARY,
    build_campaign_report,
    build_code_only_report,
    campaign_report_html,
    campaign_report_markdown,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.report.build import CHECK_PROOF_WORDS, NOT_SHOWN_TO_MEASURE
from threetears.evals.analysis.report.campaign.names import measure_name_of
from threetears.evals.analysis.report.words import judged_heading
from threetears.evals.analysis.report.model import CampaignReport, DisclosureBlock, ReportBlock, TableBlock
from threetears.evals.analysis.style import CANNOT_TELL_APART, cannot_tell_apart, lint_prose
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.quick.files import load_evals
from threetears.evals.quick.result import Result

from packages.evals.tests.fixtures.report_inputs.dataset import kairo_bundle
from packages.evals.tests.g2_support import g2_report, g2_ruled_report
from packages.evals.tests.ladder_support import load, offline_copy
from packages.evals.tests.report_shapes_support import s1_change, s2_choice, s4_change

#: Each rung: its directory, the file declaring its eval, and the module attribute that holds it (None for TOML).
_RUNGS: dict[str, tuple[str, str, str | None]] = {
    "rung0": ("1-triage", "triage.toml", None),
    "rung1": ("1-triage", "triage.py", "triage_eval"),
    "rung2": ("2-replies", "replies.py", "replies_eval"),
    "rung3": ("3-invoices", "extract.py", "invoices"),
    "rung4": ("4-booking", "agent.py", "booking"),
}

#: The words the undecided reading retired: a first-time reader took them for "the same".
_RETIRED = re.compile(r"\b(?:not|nothing(?: else)?) separated\b", re.IGNORECASE)


@dataclass(frozen=True)
class _Surfaces:
    """One rung replayed offline, and the four places its result is said."""

    result: Result
    lead: str
    verdict: str
    first_line: str
    exit_line: str | None
    exit_code: int


def _said(blocks: list[ReportBlock], source: str, section: str | None = None) -> list[DisclosureBlock]:
    return [
        block
        for block in blocks
        if isinstance(block, DisclosureBlock) and block.source == source and section in (None, block.section)
    ]


@pytest.fixture(scope="module")
def rungs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, _Surfaces]]:
    """Every rung replayed once, from its recordings, with no key."""
    replayed: dict[str, _Surfaces] = {}
    for name, (directory, file, attribute) in _RUNGS.items():
        with pytest.MonkeyPatch.context() as monkeypatch:
            copy = offline_copy(directory, tmp_path_factory.mktemp(name), monkeypatch)
            if attribute is None:
                (runnable,) = load_evals(copy / file)
            else:
                runnable = getattr(load(copy / file, Path(file).stem, monkeypatch), attribute)
            result = asyncio.run(runnable.run(store=":memory:"))
        (lead,) = _said(result.page.blocks, "verdict", "answer")
        (verdict,) = _said(result.report.blocks, "verdict", "header")
        replayed[name] = _Surfaces(
            result=result,
            lead=lead.text,
            verdict=verdict.text,
            first_line=str(result).splitlines()[0],
            exit_line=result.why_failed(),
            exit_code=result.exit_code(),
        )
    yield replayed


# --- one result, one sentence ----------------------------------------------------------------------------------------


#: Rung 3's result as every surface says it: the decision, what each arm passed, what the interval allows.
_RUNG_3_LEAD = (
    "Accuracy: the evidence cannot tell v2 and v1 apart. Of 90 trials each, v1 passed 90 and v2 89. Over 30 paired "
    "cases, the interval allows v2 to be as much as 19.7 points worse or 17.8 better."
)

#: What rung 3's undecided guardrail means for the decision, as its verdict line says it.
_RUNG_3_GUARDRAIL = "v2 may do worse than v1 on the No invented total score guardrail."


def test_rung_3_s_four_surfaces_say_one_sentence_in_one_phrase_and_one_measure_name(
    rungs: dict[str, _Surfaces],
) -> None:
    rung = rungs["rung3"]
    # The page leads with the result; the guardrail is its warning, one line down.
    assert rung.lead == _RUNG_3_LEAD
    # The aspect page and the print say the same sentences, then what the guardrail means for the decision.
    assert rung.verdict == f"{_RUNG_3_LEAD} {_RUNG_3_GUARDRAIL}"
    assert rung.first_line == f"invoices: {_RUNG_3_LEAD} {_RUNG_3_GUARDRAIL}"
    # The exit line opens with the result, then why the gate failed, on the guardrail by the same name.
    assert rung.exit_code == 4
    assert rung.exit_line == (
        f"{_RUNG_3_LEAD} The gate failed on guardrail No invented total score, v2 vs v1: undecided: v2 may be worse "
        "than v1 by more than the margin (the interval reaches both sides of -2 pts, the declared margin, so the arm is "
        "shown neither within it nor beyond it)."
    )
    # The print names each arm's measures as the page does: the classifier's match is the Accuracy the page leads on.
    printed = str(rung.result)
    assert "  Accuracy: mean " in printed and "  currency: mean" not in printed
    assert "  No invented total score: mean " in printed and "  no_invented_total:" not in printed


def test_page_b_s_lead_says_what_the_interval_allows_each_arm_the_subject(rungs: dict[str, _Surfaces]) -> None:
    """A first-time reader of an undecided lead takes small figures for "the same"; the lead says what each arm passed
    and how far the interval lets the candidate fall or rise, in points, its ends the contrasts table's."""
    rung = rungs["rung3"]
    lead = rung.lead
    assert "v1 passed 90" in lead and "v2 89" in lead and "Of 90 trials each" in lead
    assert "as much as 19.7 points worse or 17.8 better" in lead
    # No level the first screen leaves unexplained, no subject but an arm, no count of cases to settle without a margin.
    assert "%" not in lead and " It " not in lead and "settle" not in lead
    (contrasts,) = [
        block for block in rung.result.report.blocks if isinstance(block, TableBlock) and block.name == "comparisons"
    ]
    (row,) = [row for row in contrasts.rows if row["reading"].text == "Accuracy"]
    assert (row["interval"].text, row["cases"].text) == ("[-19.7, 17.8] pts", "30 paired")
    assert lint_prose(lead) == []
    # The lead links the phrase's one definition at its first use, in the header, before any table says it.
    html = campaign_report_html(rung.result.page)
    header = html[: html.index("</header>")]
    assert f'title="{escape(GLOSSARY[CANNOT_TELL_APART])}"' in header


def _kairo_page() -> CampaignReport:
    """Kairo round 2's campaign page over its committed bundle, as the campaign declared itself then."""
    bundle = kairo_bundle()
    declared = bundle.declaration
    campaign = EvalCampaign.model_validate(
        {
            "id": bundle.campaign_id,
            "scope_id": bundle.scope_id,
            "name": bundle.name,
            "subject_id": bundle.subject_id,
            "subject_kind": bundle.subject_kind,
            "behavior": bundle.behavior,
            "created_by": declared.declared_by or "unknown",
            "question": declared.question,
            "declared_at": declared.declared_at,
            "declared_by": declared.declared_by or "",
            "arms": list(declared.arms),
            "aspects": list(declared.aspects),
            "floors": list(declared.floors),
        }
    )
    labels = bundle.readings.arm_labels()
    reports = {
        aspect.aspect: build_code_only_report(aspect, assembled_at="2026-10-31T00:00:00+00:00", arm_labels=labels)
        for aspect in bundle.aspects
    }
    return build_campaign_report(campaign, bundle.readings, reports)


def test_a_lead_says_an_arm_shown_worse_is_worse_whatever_its_margin_allows() -> None:
    """qwen3.8-flash's interval on its drop lies wholly below zero and within the 15-point margin: the lead says both,
    September the subject of its own count."""
    (lead,) = _said(_kairo_page().blocks, "verdict", "answer")
    assert lead.text == (
        "Pass rate: qwen3.8-flash shown worse than September, but within the 15-point margin. Of 897 trials, September "
        "passed 765. Of 895, qwen3.8-flash passed 695. The interval puts qwen3.8-flash between 3.4 and 10.8 points "
        "worse. Pass rate: 1017 against September no worse than the margin. Of 897 trials each, September passed 765 "
        "and 1017 738. The interval allows 1017 to be as much as 5.8 points worse or 1.1 better."
    )


def test_a_lead_under_a_margin_states_the_count_its_to_settle_row_states() -> None:
    page = s1_change().page()
    (lead,) = _said(page.blocks, "verdict", "answer")
    (settle,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "to_settle"]
    counts = [
        found.group(1)
        for row in settle.rows
        if row["shows"].text.startswith("no worse") and (found := re.match(r"(\d+) cases", row["observed"].text))
    ]
    said = re.findall(r"settling it takes (\d+) cases in all, as To settle counts it\.", lead.text)
    assert said == counts and counts[:2] == ["233", "31"]


def test_rung_4_s_three_surfaces_say_one_sentence_and_it_exits_clean(rungs: dict[str, _Surfaces]) -> None:
    rung = rungs["rung4"]
    sentence = "agent passed 31 of 36 trials on 12 cases."
    # The page leads with the count; the aspect page and the print say it, then which cases failed a trial.
    missed = "Cases that failed a trial: r08, r09, r10."
    assert rung.lead == sentence and rung.verdict == f"{sentence} {missed}"
    assert rung.first_line == f"booking: {sentence} {missed}"
    # One arm has nothing to gate, so no exit line is printed beside the print.
    assert (rung.exit_code, rung.exit_line) == (0, None)


@pytest.mark.parametrize("name", ["rung0", "rung1", "rung2"])
def test_every_one_arm_rung_s_page_lead_is_its_aspect_s_verdict_and_its_print_s_first_line(
    rungs: dict[str, _Surfaces], name: str
) -> None:
    rung = rungs[name]
    assert rung.verdict.startswith(rung.lead) and rung.first_line == f"{rung.result.name}: {rung.verdict}"
    assert re.fullmatch(r"\S+ passed \d+ of \d+ trials on \d+ cases\.", rung.lead), rung.lead


# --- the retired phrase ----------------------------------------------------------------------------------------------


def _rendered(rung: _Surfaces) -> dict[str, str]:
    page, report = rung.result.page, rung.result.report
    return {
        "campaign.md": campaign_report_markdown(page),
        "campaign.html": campaign_report_html(page),
        "campaign-static.html": campaign_report_html(page, static=True),
        "aspect.md": report_markdown(report),
        "aspect.html": report_html(report),
        "aspect-static.html": report_html(report, static=True),
        "print": str(rung.result),
        "print-full": rung.result.render(full=True),
        "exit": rung.exit_line or "",
    }


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_no_surface_of_a_rung_prints_the_retired_phrase(rungs: dict[str, _Surfaces], name: str) -> None:
    for surface, text in _rendered(rungs[name]).items():
        assert not _RETIRED.search(text), (surface, _RETIRED.findall(text))


async def test_no_shape_s_page_prints_the_retired_phrase() -> None:
    pages = {
        "s1": campaign_report_markdown(s1_change().page()),
        "s2": campaign_report_markdown(s2_choice().page()),
        "s4": campaign_report_markdown(s4_change().page()),
        "s4.html": campaign_report_html(s4_change().page()),
        "g2": report_markdown(await g2_report()),
        "g2_ruled": report_markdown((await g2_ruled_report())[1]),
    }
    for name, text in pages.items():
        assert not _RETIRED.search(text), (name, _RETIRED.findall(text))


def test_the_undecided_phrase_is_one_glossary_entry_and_the_verdict_column_links_it(
    rungs: dict[str, _Surfaces],
) -> None:
    assert CANNOT_TELL_APART in GLOSSARY and "same" in GLOSSARY[CANNOT_TELL_APART]
    assert not any(_RETIRED.search(words) for words in GLOSSARY.values())
    assert "separated" not in GLOSSARY
    # A table cell says the short form; the column it sits in links the one definition.
    report = rungs["rung3"].result.report
    (contrasts,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "comparisons"]
    (column,) = [column for column in contrasts.columns if column.key == "verdict"]
    assert column.term == CANNOT_TELL_APART
    assert {row["verdict"].text.split(";")[0] for row in contrasts.rows} == {CANNOT_TELL_APART}
    assert CANNOT_TELL_APART in report.terms
    html = report_html(report)
    assert html.count('<dt id="g-cannot-tell-apart">') == 1


def test_the_sentence_form_names_every_arm_and_the_control_last() -> None:
    assert cannot_tell_apart(["v2"], "v1") == "the evidence cannot tell v2 and v1 apart"
    assert cannot_tell_apart(["v2", "v3"], "v1") == "the evidence cannot tell v2 or v3 apart from v1"
    assert cannot_tell_apart(["b", "c", "d"], "a") == "the evidence cannot tell b, c or d apart from a"


# --- one name per guardrail and measure ------------------------------------------------------------------------------


def test_one_guardrail_key_reads_one_name_on_every_surface(rungs: dict[str, _Surfaces]) -> None:
    rung = rungs["rung3"]
    (fold,) = rung.result.page.aspects
    name = measure_name_of(fold.report.bundle, "no_invented_total")
    assert name == "No invented total score"
    (warning,) = _said(rung.result.page.blocks, "guardrails", "warnings")
    surfaces = {
        "warning": warning.text,
        "verdict": rung.verdict,
        "first line": rung.first_line,
        "exit": rung.exit_line or "",
        "print": str(rung.result),
        "aspect page": report_markdown(rung.result.report),
        "campaign page": campaign_report_markdown(rung.result.page),
    }
    for surface, text in surfaces.items():
        assert name in text, surface
        # The key humanized ("no invented total") is a second name for it; only the one name is printed.
        assert not re.search(r"\bno invented total(?! score)", text, re.IGNORECASE), surface


def test_the_undecided_guardrail_says_what_it_means_first_then_the_cases_that_could_fail_it(
    rungs: dict[str, _Surfaces],
) -> None:
    (warning,) = _said(rungs["rung3"].result.page.blocks, "guardrails", "warnings")
    assert warning.text == (
        "v2 may do worse than v1 on the No invented total score guardrail on classify, by more than its 2-point "
        "margin. Of the 30 cases that could fail it, v2 failed 1 and v1 failed 1, too few to settle it."
    )


def test_a_measurement_s_scorecard_names_the_pass_rate_as_a_name_and_one_figure_once(
    rungs: dict[str, _Surfaces],
) -> None:
    page = rungs["rung0"].result.page
    (matrix,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "scorecard"]
    headers = [column.header for column in matrix.columns]
    assert "Pass rate (%)" in headers and "pass rate (%)" not in headers
    # The classifier's label match is its accuracy: one figure, one column, under the name the lead reads.
    assert "Accuracy" in headers and "Label matched" not in headers
    (measures,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "measures"]
    named = [row["measure"].text for row in measures.rows]
    assert "Pass rate" in named and "Label matched" not in named and named.count("Accuracy") == 1


# --- a goal check a candidate that did nothing could pass -----------------------------------------------------------


def test_rung_4_s_page_and_print_say_one_thing_of_each_unproven_goal_check(rungs: dict[str, _Surfaces]) -> None:
    rung = rungs["rung4"]
    page = campaign_report_markdown(rung.result.page)
    printed = str(rung.result)
    check_lines = [line for line in printed.splitlines() if line.strip().startswith("goal check ")]
    assert len(check_lines) == 3
    unproven = CHECK_PROOF_WORDS["unproven"]
    for line in check_lines:
        # The page's words for what the check is and what follows; then what the launch measured, which the page
        # cannot know, said as the launch's.
        assert f"{unproven}; {NOT_SHOWN_TO_MEASURE}; at launch one passed it in " in line, line
    assert "NOT A MEASUREMENT" not in printed and "does not beat doing nothing" not in printed
    sentence = f"{NOT_SHOWN_TO_MEASURE[:1].upper()}{NOT_SHOWN_TO_MEASURE[1:]}."
    assert page.count(f"is {unproven}. {sentence}") == 3


# --- reading names, never identifiers --------------------------------------------------------------------------------

#: An identifier the engine reserves (a judge's ``__outcome__`` axis), which no sentence or heading prints.
_DUNDER = re.compile(r"__[a-z_]+__")


def test_no_code_written_page_prints_a_reserved_judge_axis_by_its_stored_id() -> None:
    """Each aspect's own page over Kairo's committed bundle names its judge's reserved axes as every surface does,
    ``outcome (judged)``, never ``__outcome__``, in every line and every table's headers and readings code writes; a
    trial's stored error, quoted as recorded, is the run's words."""
    pages = [
        build_code_only_report(bundle, assembled_at="2026-10-31T00:00:00+00:00") for bundle in kairo_bundle().aspects
    ]
    assert any("outcome (judged)" in report_markdown(page) for page in pages)
    for page in pages:
        said: list[str] = []
        for block in page.blocks:
            if isinstance(block, DisclosureBlock):
                said.append(block.text)
            elif isinstance(block, TableBlock):
                said += [column.header for column in block.columns]
                said += [row["reading"].text for row in block.rows if "reading" in row]
        assert said and not [text for text in said if _DUNDER.search(text)]


#: Code a reader copies (a key quoted in a trial record), not a name.
_CODE = re.compile(r"`[^`]*`")

#: A judged dimension, a classifier's label match and the overall pass rate under a second name: what one reading
#: name per measure rules out.
_SECOND_NAMES = re.compile(r"\| the pass rate \||[a-z]+\.[a-z_]+ \(judged|\| Label matched \|")


@pytest.mark.parametrize("name", ["rung0", "rung2", "rung3"])
def test_every_measure_reads_one_name_on_page_aspect_page_print_and_exit_line(
    rungs: dict[str, _Surfaces], name: str
) -> None:
    surfaces = _rendered(rungs[name])
    for surface, text in surfaces.items():
        # A trial's record quotes each dimension by its key, as code: that is the key, not a name.
        said = _CODE.sub("", text)
        assert not _SECOND_NAMES.search(said), (surface, _SECOND_NAMES.findall(said))


def test_a_judged_dimension_reads_one_name_on_the_page_and_in_the_print(rungs: dict[str, _Surfaces]) -> None:
    surfaces = _rendered(rungs["rung2"])
    for dimension in ("answer.grounded", "answer.answers", "answer.safe"):
        named = judged_heading(dimension)
        assert named == f"{dimension.replace('.', ' ')} (judged)"
        assert f"| {named} |" in surfaces["aspect.md"] and f"  {named}, scored pass/fail: mean " in surfaces["print"]
        assert dimension not in surfaces["print"]


def test_kairo_s_judged_dimension_reads_one_name_in_its_contrasts_and_floors() -> None:
    # Every table header and reading cell, and every line code writes, names it one way; a trial's record quotes the
    # judge's own rubric, keyed by the dimension, as stored.
    page = _kairo_page()
    said: list[str] = []
    for report in [fold.report for fold in page.aspects]:
        for block in [*page.blocks, *report.blocks]:
            if isinstance(block, DisclosureBlock):
                said.append(block.text)
            elif isinstance(block, TableBlock):
                said += [column.header for column in block.columns]
                said += [row[key].text for row in block.rows for key in ("reading", "rate", "guardrail") if key in row]
    assert any("persona holds character (judged)" in text for text in said)
    assert not [text for text in said if "persona.holds_character" in _CODE.sub("", text)]
    assert "| the pass rate |" not in campaign_report_markdown(page)
