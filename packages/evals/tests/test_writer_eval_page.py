"""The writer eval's page: its answer leads, its side-by-side tables are rows with columns, and a replay of a declared
eval reads its criteria as set before its results.

A two-arm replay over Kairo's case, both arms laid Kairo's report, so both pages are stored, scored and shown to the
scripted judge side by side in both orders. The judge scores 4 every time, so it prefers the page it reads second in
both orders: its orders disagree, the promotion read voids its primary reading, and the decision is undecided. Rung 5's
replayed page, where the preference is read and does not separate, is held in ``test_rung_five_writer.py``.

Mutations that turn this file red: the answer missing, or not directly under the question; its decision word, share or reason
not the result's; the promotion's own sentence missing from the page; the side-by-side table laid out as key and value;
the campaign filed dated after its arms ran, so its criteria read as set after its results; an interval end
rounded onto even, or two ends rounded alike; a share printed at fewer decimals than its interval's ends, so the two
shares miss 100 or a share just off even reads as even; a share drawn with no interval as a solid mark.
"""

from __future__ import annotations

import socket
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import pytest

from packages.evals.tests.writer_eval_support import FakeScoringJudge, kairo_case, kairo_report, lay_recording
from threetears.evals.analysis.stats import PROJECTION_CASE_CAP
from threetears.evals.kernel.numbers import spell_count
from threetears.evals.ops.report_writer.writer_pairwise import resolution_line
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.report import ChartBlock, DisclosureBlock, DotIntervalChart, TableBlock
from threetears.evals.harness import Writer, WriterEval, WriterResult
from threetears.evals.harness import writer_eval as writer_eval_module
from threetears.evals.harness.writer_turns import writer_turns
from threetears.evals.ops.report_writer.writer_pairwise import WriterPairwiseReport

#: The writer every arm names.
WRITER: Final = "report-writer-candidate"

#: The question the eval declares, so its aspect and its control are criteria the page times.
QUESTION: Final = "Does a blank line in SKILL.md change the reports?"


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key in the environment, and every socket connect refused."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def connect(_self: socket.socket, address: object) -> None:
        raise AssertionError(f"a replay-only run connected to {address}")

    monkeypatch.setattr(socket.socket, "connect", connect)


async def _replayed(tmp_path: Path) -> WriterResult:
    """The two-arm replay over Kairo's case, declaring :data:`QUESTION`."""
    files = dict(load_skill().files)
    files["SKILL.md"] = f"{files['SKILL.md']}\n"
    arms = {"shipped": Writer(WRITER, skill=load_skill()), "variant": Writer(WRITER, skill=AnalystSkill(files=files))}
    kairo = kairo_case(tmp_path / "cases")
    for arm, writer in arms.items():
        routed = writer_turns(
            tmp_path / "turns", arm=arm, skill=writer.analyst_skill, cases={kairo.snapshot.id: "kairo"}
        )
        lay_recording(routed.path(WRITER, None, kairo.snapshot.id), kairo_report(WRITER, WRITER))
    writer_eval = WriterEval(
        tmp_path / "cases",
        arms=arms,
        judge=FakeScoringJudge(),
        recordings=tmp_path / "turns",
        budget_usd=0.75,
        max_cost_usd=8.0,
        held_out_fraction=0.35,
        question=QUESTION,
        store=tmp_path / "evals.sqlite",
    )
    return await writer_eval.run()


async def test_the_answer_leads_the_page_in_the_results_own_words_and_the_tables_are_rows_with_columns(
    tmp_path: Path, offline: None
) -> None:
    result = await _replayed(tmp_path)
    read = result.pairwise["variant"]
    assert result.decisions == {"variant": "undecided"} and read.void is not None and read.share is not None

    # The answer is the page's lead, under the question and the line of what the page compares (presentation chunk
    # 4), set apart as the verdict line.
    question, compares, answer = result.page.blocks[:3]
    assert isinstance(compares, DisclosureBlock) and compares.text == "shipped vs variant · report-writing"
    assert isinstance(answer, DisclosureBlock) and (answer.section, answer.source) == ("answer", "verdict")
    assert answer.tone == "open", "undecided is no decision"
    assert isinstance(question, DisclosureBlock) and question.text == f'"{QUESTION}"'
    # Its words are the result's: the decision, the judge's share as the table prints it, and why it cannot count, the
    # reason after a colon, so its own count's parentheses never nest in another pair.
    share = round(read.share * 100)
    assert answer.text == (
        f"Undecided, so shipped stays: across 1 held-out case the judge gave variant {share}% of its preference over "
        f"shipped (50% is even), but its two reading orders disagreed too often for that to count: {read.void}."
    )
    # The promotion's own sentence is on the page, one step down, as the print says it.
    said = [block.text for block in result.page.blocks if isinstance(block, DisclosureBlock)]
    assert f"Promotion of variant: {result.promotion_why['variant']}" in said
    assert f"promotion of variant: {result.promotion_why['variant']}" in result.render()

    # The preference figure opens the Scorecard, its line under it, then the side-by-side table: a row per candidate
    # and the pairs it reads, a column per figure.
    tables = {block.name: block for block in result.page.blocks if isinstance(block, TableBlock)}
    side = tables["side_by_side"]
    scorecard = [block for block in result.page.blocks if block.section == "scorecard"]
    figure, line, terms = scorecard[:3]
    assert scorecard.index(side) == 3, "the figure, its line, the sentence defining its terms, then the table"
    assert isinstance(terms, DisclosureBlock) and terms.text.startswith("Side by side, the judge read")
    # One case reads no interval: the share is an open mark, and the figure states no level.
    assert isinstance(figure, ChartBlock) and isinstance(figure.data, DotIntervalChart)
    assert [(row.value, row.lo, row.hi, row.hollow) for row in figure.data.rows or []] == [(share, None, None, True)]
    assert figure.data.level is None
    assert isinstance(line, DisclosureBlock) and line.text == (
        f"variant {share}% · shipped {100 - share}% (50% is even); untested: {read.primary.reason_detail} (void)."
    )
    assert [column.key for column in side.columns] == [
        "candidate",
        "pairs",
        "reads",
        "share",
        "control_share",
        "interval",
    ]
    values = side.values()
    assert [(row["candidate"], row["pairs"]) for row in values] == [
        ("variant", "every pair"),
        ("variant", "both pages finished"),
    ]
    # An untested reading names why in its row: one case, under the two the test needs; and void besides.
    assert values[0]["reads"] == f"untested: {read.primary.reason_detail} (void)" and values[0]["share"] == share
    assert read.primary.reason_detail == "1 case with a share, and the test needs two"
    checks = tables["judge_checks"].values()
    assert [row["candidate"] for row in checks] == ["variant"]
    assert checks[0]["order_consistency"] == f"0% ({read.order_consistent} of {read.both_orders})"
    # The same tables in the print and in pairwise.md: no key-and-value row.
    written = result.write(tmp_path / "report")
    pairwise = written["pairwise"].read_text(encoding="utf-8")
    assert result.pairwise_text() in pairwise and result.pairwise_text() in result.render()
    assert "| Candidate | Pairs | Reads | Candidate's share (%) | Control's share (%) |" in pairwise
    assert "| share |" not in pairwise and "| reads |" not in pairwise

    # The written page opens on the question under the title, then what it compares (presentation chunk 4), then the
    # answer, before anything else.
    page = written["markdown"].read_text(encoding="utf-8").splitlines()
    assert page[1] == f'"{QUESTION}"' and page[3] == compares.text and page[5] == f"**{answer.text}**"


async def test_a_replay_of_a_declared_eval_reads_its_criteria_as_set_before_its_results(
    tmp_path: Path, offline: None
) -> None:
    result = await _replayed(tmp_path)

    # The eval asks a question, so its aspect and its control are timed against its first result; both were fixed as
    # the arms launched, before any result existed.
    (fold,) = result.page.aspects
    assert fold.report.bundle.question_declared
    assert fold.report.bundle.reading_scope.late_criteria == []
    page = result.write(tmp_path / "report")["markdown"].read_text(encoding="utf-8")
    assert "set after the results" not in page and "Criteria set late" not in page


async def test_both_shares_print_at_the_interval_s_decimals_sum_to_100_and_stay_off_even(
    tmp_path: Path, offline: None
) -> None:
    result = await _replayed(tmp_path)
    read = result.pairwise["variant"]

    def shares(share: float, interval: tuple[float, float]) -> tuple[str, str, str]:
        primary = read.primary.model_copy(update={"interval": interval})
        shown = replace(result, pairwise={"variant": replace(read, share=share, primary=primary)}).pairwise_text()
        (line,) = [line for line in shown.splitlines() if line.startswith("| variant | every pair |")]
        candidate, control, ends = line.removesuffix(" |").split(" | ")[-3:]
        return candidate, control, ends

    # An interval end of 99.998% needs three decimals to stay short of all; both shares print at those decimals, past
    # the four significant figures a lone figure keeps, so the two sum to exactly 100 as printed. The three are one
    # set, and a set reads at one precision, so the whole end keeps the set's decimals too.
    candidate, control, ends = shares(0.910714, (0.6, 0.99996))
    assert (candidate, control, ends) == ("91.071", "8.929", "80.000 to 99.998")
    assert Decimal(candidate) + Decimal(control) == 100
    # A share just under even stays under it, and the control's just over, never both printed as 50.
    candidate, control, ends = shares(0.499996, (-0.2, 0.2))
    assert (candidate, control, ends) == ("49.9996", "50.0004", "40.0000 to 60.0000")
    assert Decimal(candidate) + Decimal(control) == 100


async def test_the_share_reads_alike_in_the_table_the_figure_s_values_its_line_and_the_answer(
    tmp_path: Path, offline: None
) -> None:
    """V6's review: the table's share is one set with the figure drawn above it, the line under that figure and the
    page's answer, so each prints it alike: the figure's values table and tooltips in its format's digits (``pct:0``
    where the set reads whole, ``pct:3`` at three decimals), never coarser than the line, and both share columns of a
    row one kind."""
    result = await _replayed(tmp_path)
    side = next(block for block in result.page.blocks if isinstance(block, TableBlock) and block.name == "side_by_side")
    row = side.rows[0]
    share, control = row["share"].text, row["control_share"].text
    assert type(row["share"].value) is type(row["control_share"].value), "both share columns one kind"
    figure = next(block for block in result.page.blocks if isinstance(block, ChartBlock))
    assert isinstance(figure.data, DotIntervalChart)
    places = len(share.partition(".")[2])
    assert figure.data.fmt == f"pct:{places}"
    _, values = figure.values_table()
    assert values[0][1] == f"{share}%", (values, share)
    said = [block.text for block in result.page.blocks if isinstance(block, DisclosureBlock)]
    assert f"variant {share}% · shipped {control}% (50% is even)" in " ".join(said)
    answer = next(text for text in said if text.startswith("Undecided"))
    assert f"gave variant {share}% of its preference" in answer, answer


#: Two candidates' readings as their passes report them, shaped so their shares need different precisions: whole
#: percents for one, three decimals for the other, a share 0.002 points under even.
_SHAPED: Final = {"wide": (0.61, (-0.4, 0.8)), "near": (0.49998, (-0.2, 0.2))}


async def _two_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WriterResult:
    """The replay over Kairo's case with two candidates beside the control, each pass's reading shaped as
    :data:`_SHAPED` says where the completed job hands it to the eval (``await_job``), so the page is built on it."""
    skill = load_skill()
    arms = {"shipped": Writer(WRITER, skill=skill)}
    for name, suffix in (("wide", "\n"), ("near", "\n\n")):
        files = dict(skill.files)
        files["SKILL.md"] = f"{files['SKILL.md']}{suffix}"
        arms[name] = Writer(WRITER, skill=AnalystSkill(files=files))
    kairo = kairo_case(tmp_path / "cases")
    for arm, writer in arms.items():
        routed = writer_turns(
            tmp_path / "turns", arm=arm, skill=writer.analyst_skill, cases={kairo.snapshot.id: "kairo"}
        )
        lay_recording(routed.path(WRITER, None, kairo.snapshot.id), kairo_report(WRITER, WRITER))
    awaited = writer_eval_module.await_job

    async def shaped(*args: Any, **kwargs: Any) -> Any:
        statuses = await awaited(*args, **kwargs)
        out = []
        for status in statuses:
            result = status.result
            if isinstance(result, dict) and result.get("candidate") in _SHAPED and "whole" in result:
                report = WriterPairwiseReport.model_validate(result)
                share, interval = _SHAPED[report.candidate]
                verdict = report.whole.verdict.model_copy(update={"interval": interval, "interval_level": 0.95})
                whole = report.whole.model_copy(update={"candidate_share": share, "verdict": verdict})
                status = status.model_copy(update={"result": report.model_copy(update={"whole": whole}).model_dump()})
            out.append(status)
        return out

    monkeypatch.setattr(writer_eval_module, "await_job", shaped)
    writer_eval = WriterEval(
        tmp_path / "cases",
        arms=arms,
        judge=FakeScoringJudge(),
        recordings=tmp_path / "turns",
        budget_usd=0.75,
        max_cost_usd=12.0,
        held_out_fraction=0.35,
        question=QUESTION,
        store=tmp_path / "evals.sqlite",
    )
    return await writer_eval.run()


async def test_each_candidate_s_share_reads_at_its_own_precision_in_its_panel_its_line_its_row_and_the_answer(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V5's review (O-9): with two candidates the figure drew every share at the finest one's decimals, so ``61``
    read ``61.000%`` beside the table's ``61``. Where the candidates' precisions differ, each is a panel of its own in
    its own format, and each share reads alike in its panel's values table, its line, its table row and the answer."""
    result = await _two_candidates(tmp_path, monkeypatch)
    figure = next(block for block in result.page.blocks if isinstance(block, ChartBlock))
    assert isinstance(figure.data, DotIntervalChart) and figure.data.panels is not None, "one panel per candidate"
    assert [(panel.title, panel.fmt) for panel in figure.data.panels] == [("wide", "pct:0"), ("near", "pct:3")]
    expected = {"wide": ("61", "39", "30", "90"), "near": ("49.998", "50.002", "40.000", "60.000")}
    _, values = figure.values_table()
    assert values == [
        ["wide", "wide", "61%", "30 to 90%"],
        ["near", "near", "49.998%", "40.000 to 60.000%"],
    ], values
    side = next(block for block in result.page.blocks if isinstance(block, TableBlock) and block.name == "side_by_side")
    said = " ".join(block.text for block in result.page.blocks if isinstance(block, DisclosureBlock))
    (answer,) = [
        block.text
        for block in result.page.blocks
        if isinstance(block, DisclosureBlock) and (block.section, block.source) == ("answer", "verdict")
    ]
    for arm, (share, control, low, high) in expected.items():
        (row,) = [row for row in side.rows if row["candidate"].text == arm and row["pairs"].text == "every pair"]
        assert (row["share"].text, row["control_share"].text, row["interval"].text) == (
            share,
            control,
            f"{low} to {high}",
        )
        assert f"{arm} {share}% · shipped {control}% (50% is even), 95% interval {low} to {high}%" in said, said
        assert f"gave {arm} {share}% of its preference" in answer, answer
    # Each end of the axis names the candidate in words, never one candidate's name for both.
    assert figure.data.better == ("prefers shipped", "prefers the candidate")


async def test_an_interval_end_near_even_keeps_its_side_and_two_near_ends_stay_apart(
    tmp_path: Path, offline: None
) -> None:
    result = await _replayed(tmp_path)
    read = result.pairwise["variant"]

    def row(share: float, interval: tuple[float, float]) -> str:
        primary = read.primary.model_copy(update={"interval": interval})
        shown = replace(result, pairwise={"variant": replace(read, share=share, primary=primary)}).pairwise_text()
        (line,) = [line for line in shown.splitlines() if line.startswith("| variant | every pair |")]
        return line

    # The pass reads its interval on the difference of the two shares; the page prints it on the candidate's share.
    # A low end of -0.004 points of difference is 49.998% of the preference: never printed as 50, so the crossing of
    # even stays on the page, and the shares print at the interval's precision, summing to 100.
    assert row(0.6, (-0.00004, 0.4)).endswith("| 60.000 | 40.000 | 49.998 to 70.000 |")
    # Two ends that whole percents would print alike take the decimals that tell them apart.
    assert row(0.6702, (0.3402, 0.3406)).endswith("| 67.02 | 32.98 | 67.01 to 67.03 |")


async def test_a_count_of_cases_to_settle_reads_with_its_thousands_separated(tmp_path: Path, offline: None) -> None:
    """After rung 5's judge was recorded again its page printed "Held-out cases to settle | 2969": a count on the page
    reads as every count the page spells does, thousands separated (``2,969``), through the one count speller."""
    result = await _replayed(tmp_path)
    read = result.pairwise["variant"]
    shown = replace(result, pairwise={"variant": replace(read, cases_to_settle=2969)}).pairwise_text()
    (line,) = [line for line in shown.splitlines() if line.startswith("| variant | 0% ")]
    assert "| 2,969 |" in line and "2969" not in line, line
    capped = replace(result, pairwise={"variant": replace(read, cases_to_settle=PROJECTION_CASE_CAP + 1)})
    (line,) = [line for line in capped.pairwise_text().splitlines() if line.startswith("| variant | 0% ")]
    assert "| more than 10,000 |" in line, line


def test_the_count_speller_separates_thousands_and_the_resolution_line_reads_through_it() -> None:
    """Finding N-6: one count speller (``spell_count``) for every count a page spells."""
    assert [spell_count(count) for count in (7, 44, 999, 2969, 10_000, 6224)] == [
        "7",
        "44",
        "999",
        "2,969",
        "10,000",
        "6,224",
    ]
    assert resolution_line(11, projected=2969).endswith("the last run on these cases projected 2,969 to settle.")
    assert resolution_line(11, projected=PROJECTION_CASE_CAP + 1).endswith("projected more than 10,000 to settle.")
