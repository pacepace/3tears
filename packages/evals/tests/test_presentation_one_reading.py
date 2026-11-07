"""One result, one reading, one sentence, on every surface a reader meets it (presentation build, chunk 2).

The owner's reading of page B (2026-11-06) took "not separated" for "the two are roughly equal" (H7): the review's A7
holds a page to a phrase a first-time reader cannot take for "the same". The review's P41 found one rung 3 result
phrased four ways (the campaign page's lead, the aspect page's verdict, the quick print's first line, the exit line)
and a measure named two ways (A3's single-name clause). Held here on the ladder's replayed rungs, offline:

- the campaign page's lead, the aspect page's verdict, the print's first line and the exit line carry one sentence,
  one phrase and one measure name (rungs 3 and 4);
- no surface code writes prints the retired phrase ("not separated", "nothing separated"); the undecided reading is
  :data:`~threetears.evals.analysis.style.CANNOT_TELL_APART`, defined once in the glossary;
- a guardrail key and a measure key read one name on every surface (P14, P15), and the undecided guardrail's line says
  what it means for the decision before the count of cases that could fail it (H6, P26);
- rung 4's page and print say one thing of a goal check a candidate that did nothing could pass (P25).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from threetears.evals.analysis import (
    GLOSSARY,
    build_code_only_report,
    campaign_report_html,
    campaign_report_markdown,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.report.build import CHECK_PROOF_WORDS, NOT_SHOWN_TO_MEASURE
from threetears.evals.analysis.report.campaign.names import measure_name_of
from threetears.evals.analysis.report.model import DisclosureBlock, ReportBlock, TableBlock
from threetears.evals.analysis.style import CANNOT_TELL_APART, cannot_tell_apart, lint_prose
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


def test_rung_3_s_four_surfaces_say_one_sentence_in_one_phrase_and_one_measure_name(
    rungs: dict[str, _Surfaces],
) -> None:
    rung = rungs["rung3"]
    decision = "Accuracy: the evidence cannot tell v2 and v1 apart."
    # The page leads with the decision and the figures it rests on; the guardrail is its warning, one line down.
    assert rung.lead.startswith(f"{decision} ")
    # The aspect page and the print say the same sentences, then what the guardrail means for the decision.
    guardrail = "v2 may do worse than v1 on the No invented total score guardrail."
    assert rung.verdict == f"{rung.lead} {guardrail}"
    assert rung.first_line == f"invoices: {rung.lead} {guardrail}"
    # The exit line opens with them, then why the gate failed, on the guardrail by the same name.
    assert rung.exit_code == 4 and rung.exit_line is not None
    assert rung.exit_line.startswith(f"{rung.lead} The gate failed on guardrail No invented total score, v2 vs v1: ")
    # One measure name for the lead, never the classifier field's ("currency") or the key's.
    for said in (rung.lead, rung.verdict, rung.first_line, rung.exit_line):
        assert said.count(cannot_tell_apart(["v2"], "v1")) == 1 and "Accuracy" in said
        assert "currency" not in said and "accuracy" not in said
    # The print names each arm's measures as the page does: the classifier's match is the Accuracy the page leads on.
    printed = str(rung.result)
    assert "  Accuracy: mean " in printed and "  currency: mean" not in printed
    assert "  No invented total score: mean " in printed and "  no_invented_total:" not in printed


def test_rung_3_s_lead_carries_both_arms_figures_the_change_and_the_cases_to_settle(
    rungs: dict[str, _Surfaces],
) -> None:
    """The host's iframe pass (finding 1): a lead that says only "undecided" leaves the figures five screens down. The
    decision comes first, then both arms' readings with their counts, the change over the cases it was read on and its
    interval, each as the contrasts table one step down prints it, and the cases that would settle it."""
    rung = rungs["rung3"]
    assert rung.lead == (
        "Accuracy: the evidence cannot tell v2 and v1 apart. It reads 100% on v1, 90 of 90 trials. It reads 98.9% on "
        "v2, 89 of 90 trials. The change is -1.1 pts over 30 paired cases. Its 99.2% interval is [-19.7, 17.8] pts. "
        "About "
        f"{_cases_to_settle(rung.lead)} cases per arm would settle it, if the arms differ as observed."
    )
    # Every figure is the aspect page's contrasts row, at the same value and precision.
    (contrasts,) = [
        block for block in rung.result.report.blocks if isinstance(block, TableBlock) and block.name == "comparisons"
    ]
    (row,) = [row for row in contrasts.rows if row["reading"].text == "Accuracy"]
    assert row["control"].text.startswith("100% ") and row["arm"].text.startswith("98.9% ")
    assert (row["change"].text, row["interval"].text, row["cases"].text) == (
        "-1.1 pts",
        "[-19.7, 17.8] pts",
        "30 paired",
    )
    assert "Interval (99.2%)" in [column.header for column in contrasts.columns]
    # Each sentence keeps to the style's three figures.
    assert lint_prose(rung.lead) == []


def _cases_to_settle(lead: str) -> str:
    """The cases-to-settle count a lead states, read back: a thousands-grouped whole number of at least the case floor."""
    found = re.search(r"About ([\d,]+) cases per arm would settle it", lead)
    assert found is not None, lead
    assert int(found.group(1).replace(",", "")) > 30
    return found.group(1)


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
        # The page's words for what the check is and what follows; the baseline the run measured after them.
        assert f"{unproven}, {NOT_SHOWN_TO_MEASURE}; a candidate that did nothing passes it in " in line, line
    assert "NOT A MEASUREMENT" not in printed and "does not beat doing nothing" not in printed
    assert page.count(f"is {unproven}.") == 3 and page.count(NOT_SHOWN_TO_MEASURE) == 3


# --- reading names, never identifiers --------------------------------------------------------------------------------

#: An identifier the engine reserves (a judge's ``__outcome__`` axis), which no sentence or heading prints.
_DUNDER = re.compile(r"__[a-z_]+__")


def test_no_code_written_page_prints_a_reserved_judge_axis_by_its_stored_id() -> None:
    """The host's iframe pass (finding 8): Kairo's judged axes headed ``__outcome__ (judged)``. Each aspect's own page
    over Kairo's committed bundle names them as every surface does, ``outcome (judged)``, in every line and every
    table's headers and readings code writes; a trial's stored error, quoted as recorded, is the run's words."""
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
