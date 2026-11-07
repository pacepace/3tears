"""The writer eval's page: its answer leads, its side-by-side tables are rows with columns, and a replay of a declared
eval reads its criteria as set before its results.

A two-arm replay over Kairo's case, both arms laid Kairo's report, so both pages are stored, scored and shown to the
scripted judge side by side in both orders. The judge scores 4 every time, so it prefers the page it reads second in
both orders: its orders disagree, the promotion read voids its primary reading, and the decision is undecided. Rung 5's
replayed page, where the preference is read and does not separate, is held in ``test_rung_five_writer.py``.

Mutations that turn this file red: the answer missing, or not directly under the question; its decision word, share or reason
not the result's; the promotion's own sentence missing from the page; the side-by-side table laid out as key and value;
the campaign filed dated after its arms ran, so its criteria read as set after its results; an interval end
rounded onto zero, or two ends rounded alike.
"""

from __future__ import annotations

import socket
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from packages.evals.tests.writer_eval_support import FakeScoringJudge, kairo_case, kairo_report, lay_recording
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.report import DisclosureBlock, TableBlock
from threetears.evals.harness import Writer, WriterEval, WriterResult
from threetears.evals.harness.writer_turns import writer_turns

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

    # The answer is the page's lead, directly under the question, set apart as the verdict line.
    question, answer = result.page.blocks[:2]
    assert isinstance(answer, DisclosureBlock) and (answer.section, answer.source) == ("answer", "verdict")
    assert answer.tone == "open", "undecided is no decision"
    assert isinstance(question, DisclosureBlock) and question.text == f'"{QUESTION}"'
    # Its words are the result's: the decision, the judge's share as the table prints it, and why it cannot count.
    share = round(read.share * 100)
    assert answer.text == (
        f"Undecided, so shipped stays: across 1 held-out case the judge gave variant {share}% of its preference over "
        f"shipped (50% is even), but its two reading orders disagreed too often for that to count ({read.void})."
    )
    # The promotion's own sentence is on the page, one step down, as the print says it.
    said = [block.text for block in result.page.blocks if isinstance(block, DisclosureBlock)]
    assert f"Promotion of variant: {result.promotion_why['variant']}" in said
    assert f"promotion of variant: {result.promotion_why['variant']}" in result.render()

    # The side-by-side table opens the Scorecard: a row per candidate and the pairs it reads, a column per figure.
    tables = {block.name: block for block in result.page.blocks if isinstance(block, TableBlock)}
    side = tables["side_by_side"]
    scorecard = [block for block in result.page.blocks if block.section == "scorecard"]
    assert scorecard.index(side) == 1, "the sentence defining its terms, then the table"
    assert [column.key for column in side.columns] == ["candidate", "pairs", "reads", "share", "difference", "interval"]
    values = side.values()
    assert [(row["candidate"], row["pairs"]) for row in values] == [
        ("variant", "every pair"),
        ("variant", "both pages finished"),
    ]
    assert values[0]["reads"] == "untested (void)" and values[0]["share"] == share
    checks = tables["judge_checks"].values()
    assert [row["candidate"] for row in checks] == ["variant"]
    assert checks[0]["order_consistency"] == f"0% ({read.order_consistent} of {read.both_orders})"
    # The same tables in the print and in pairwise.md: no key-and-value row.
    written = result.write(tmp_path / "report")
    pairwise = written["pairwise"].read_text(encoding="utf-8")
    assert result.pairwise_text() in pairwise and result.pairwise_text() in result.render()
    assert "| Candidate | Pairs | Reads | Share (%) | Difference (pts) |" in pairwise
    assert "| share |" not in pairwise and "| reads |" not in pairwise

    # The written page opens on the question under the title, then the answer, before anything else.
    page = written["markdown"].read_text(encoding="utf-8").splitlines()
    assert page[1] == f'"{QUESTION}"' and page[3] == f"**{answer.text}**"


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


async def test_an_interval_end_near_zero_keeps_its_sign_and_two_near_ends_stay_apart(
    tmp_path: Path, offline: None
) -> None:
    result = await _replayed(tmp_path)
    read = result.pairwise["variant"]

    def row(delta: float, interval: tuple[float, float]) -> str:
        primary = read.primary.model_copy(update={"delta": delta, "interval": interval})
        shown = replace(result, pairwise={"variant": replace(read, primary=primary)}).pairwise_text()
        (line,) = [line for line in shown.splitlines() if line.startswith("| variant | every pair |")]
        return line

    # A low end of -0.004 points crosses zero: it is never printed as 0, so the crossing stays on the page.
    assert row(0.2, (-0.00004, 0.4)).endswith("| 20 | -0.004 to 40 |")
    # Two ends that whole points would print alike take the decimals that tell them apart.
    assert row(0.3402, (0.3401, 0.3404)).endswith("| 34.02 | 34.01 to 34.04 |")
