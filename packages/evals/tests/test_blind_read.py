"""The blind read: one seeded random order over a campaign's judged outputs, read with no arm, run or judge score.

Held here on small made-up campaigns, through the pure core (``analysis/blind_read.py``) and the quick path's files
(``quick/blind.py``):

- the draw depends on the answers and the seed, never on the order results are listed in or on their ids;
- the sheet, as the page's bytes and the CSV's, names no arm, run, result or case and carries no judge score;
- answers from the top are a ``random`` sample and any other set is not drawn at random;
- closing gives person ratings that name the read, the judge's scores against each reader stored as counts (no kappa,
  bar or tier on the document), and a preference with its exact test, withheld until the read reaches the length
  fixed when it was drawn; the print states each check as counts, the catch and false-fail rates with exact
  intervals, and no kappa, bar, tier word or count of results still needed;
- a result judged on other wording of a criterion, or whose stored evidence is other text, is never shown: a rating
  is bound to what its rater read;
- a model that answers the sheet is read apart from people, never as one, never when it is the judge's own model
  or cannot be shown to be another, once per model that served its answers, and its calls are held to the cost cap
  and written to the ledger;
- the page is one static file that loads nothing, and its saved labels read back as answers to the draw.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import blind_read as core
from threetears.evals.analysis.report.blind_rates import blind_rates
from threetears.evals.kernel.out_of_run import OutOfRunBudget
from threetears.evals.analysis.blind_read import (
    blind_sheet,
    close_blind_read,
    draw_blind_read,
    exact_sign_test_p,
    read_preference,
    read_sample,
)
from threetears.evals.quick import blind as files
from threetears.evals.quick.blind import (
    SHEET_COLUMNS,
    answer_sheet,
    blind_read_csv,
    blind_read_html,
    judge_lines,
    read_labels,
    write_blind_read,
)
from threetears.evals.run.blind_review import plan_review
from threetears.evals.schema import (
    EvalResult,
    JudgeEvidence,
    OutOfRunSpend,
    RubricDim,
    RubricScore,
    StopReason,
    label_key_of,
)
from threetears.evals.schema.blind_read import (
    BlindAnswer,
    BlindArm,
    BlindPreference,
    BlindRead,
    BlindReview,
    BlindVerdict,
)
from threetears.evals.schema.reading_states import StateOf
from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.report_kit_support import chromium_unavailable, rendered

GROUNDED = RubricDim(name="answer.grounded", description="Every fact is in the policy.", scale="pass_fail")
TONE = RubricDim(name="answer.tone", description="The reply is warm.", scale="ordinal")
JUDGE_MODEL = "acme/judge-1"

#: Words that appear in no output, so finding one on the sheet means an id leaked onto it.
CONTROL_RUN, CANDIDATE_RUN = "run-ctrl-7f3a", "run-cand-9b2c"


def _result(
    run: str, case: int, text: str, *, passes: bool, repeat: int = 1, tone: int | None = None
) -> tuple[EvalResult, JudgeEvidence]:
    evidence = JudgeEvidence(case_material=f"Customer question number {case}.", artifact=text)
    scores = [
        RubricScore(
            dim=GROUNDED.name,
            score=int(passes),
            scale="pass_fail",
            axis="capability",
            reasoning="JUDGE-REASONING-MARK",
            served_model=JUDGE_MODEL,
            judge_temperature=0.0,
            output_fingerprint=label_key_of(evidence, GROUNDED).output_fingerprint,
            criterion_fingerprint=label_key_of(evidence, GROUNDED).criterion_fingerprint,
        )
    ]
    if tone is not None:
        scores.append(
            RubricScore(
                dim=TONE.name,
                score=tone,
                scale="ordinal",
                axis="capability",
                served_model=JUDGE_MODEL,
                judge_temperature=0.0,
                output_fingerprint=label_key_of(evidence, TONE).output_fingerprint,
                criterion_fingerprint=label_key_of(evidence, TONE).criterion_fingerprint,
            )
        )
    result = make_eval_result(
        id=f"res-{run}-{case}-{repeat}",
        eval_run_id=run,
        test_case_id=f"case-{case:03d}",
        k_iteration=repeat,
        rubric_scores=scores,
    )
    return result, evidence


def _campaign(
    cases: int = 24, *, arms: int = 2, judge_fails: frozenset[int] = frozenset()
) -> tuple[list[EvalResult], dict[str, JudgeEvidence]]:
    """``cases`` cases, each arm's answer its own text; the judge fails the control's answer on ``judge_fails``."""
    results: list[EvalResult] = []
    evidence: dict[str, JudgeEvidence] = {}
    for case in range(cases):
        made = [_result(CONTROL_RUN, case, f"old reply to {case}", passes=case not in judge_fails)]
        if arms == 2:
            made.append(_result(CANDIDATE_RUN, case, f"new reply to {case}", passes=True))
        for result, read in made:
            results.append(result)
            evidence[result.id] = read
    return results, evidence


def _arms(count: int = 2) -> list[BlindArm]:
    arms = [BlindArm(key="v1", role="control", run_ids=[CONTROL_RUN])]
    if count == 2:
        arms.append(BlindArm(key="v2", role="candidate", run_ids=[CANDIDATE_RUN]))
    return arms


def _draw(results: list[EvalResult], evidence: dict[str, JudgeEvidence], *, arms: int = 2, **kwargs: Any) -> BlindRead:
    return draw_blind_read(
        results, evidence, [GROUNDED], _arms(arms), scope_id="scope", created_by="test", created_at="t0", **kwargs
    )


def _answer(
    read: BlindRead, position: int, *, rater: str = "owner", prefer_arm: str | None = "v2", **kwargs: Any
) -> BlindAnswer:
    """An answer to the item at ``position``: every output passed, the output of ``prefer_arm`` preferred."""
    item = read.items[position]
    prefer = None
    if len(item.outputs) == 2 and "preference" in read.ask:
        prefer = "tie" if prefer_arm is None else next(o.slot for o in item.outputs if o.arm == prefer_arm)
    scores = kwargs.pop("scores", {})
    return BlindAnswer(
        item=item.item,
        rater=rater,
        answered_at="t1",
        verdicts=[
            BlindVerdict(slot=output.slot, dim=GROUNDED.name, score=scores.get(output.arm, 1))
            for output in item.outputs
        ],
        prefer=prefer,
        **kwargs,
    )


def _with(read: BlindRead, answers: list[BlindAnswer]) -> BlindRead:
    return read.model_copy(update={"answers": answers})


# --- the draw -------------------------------------------------------------------------------------------------------


def test_the_draw_is_the_same_whatever_order_the_store_lists_results_in_and_whatever_their_ids() -> None:
    results, evidence = _campaign()
    first = _draw(results, evidence)
    shuffled = list(results)
    random.Random(3).shuffle(shuffled)
    # Another replay: every result under a new id and its run under a new id, the answers the same.
    renamed = [r.model_copy(update={"id": f"again-{r.id}", "eval_run_id": f"again-{r.eval_run_id}"}) for r in shuffled]
    arms = [arm.model_copy(update={"run_ids": [f"again-{run}" for run in arm.run_ids]}) for arm in _arms()]
    again = draw_blind_read(
        renamed,
        {f"again-{key}": read for key, read in evidence.items()},
        [GROUNDED],
        arms,
        scope_id="scope",
        created_by="test",
    )
    assert [item.item for item in again.items] == [item.item for item in first.items]
    assert [[(o.slot, o.arm, o.output_fingerprint) for o in item.outputs] for item in again.items] == [
        [(o.slot, o.arm, o.output_fingerprint) for o in item.outputs] for item in first.items
    ]
    assert again.id == first.id and again.draw == first.draw
    assert first.draw.population == 24 and first.draw.planned_items == 24
    assert _draw(results, evidence, seed=7).items != first.items, "another seed draws another order"


def test_the_order_is_not_the_store_s_and_every_block_of_two_pairs_holds_the_control_in_a_once() -> None:
    results, evidence = _campaign(40)
    read = _draw(results, evidence)
    assert [item.case_id for item in read.items] != sorted(item.case_id for item in read.items)
    assert sorted(item.case_id for item in read.items) == [f"case-{n:03d}" for n in range(40)]
    control_in_a = [item.outputs[0].arm == "v1" for item in read.items]
    for start in range(0, 40, 2):
        assert sum(control_in_a[start : start + 2]) == 1


def test_one_result_per_arm_per_case_is_drawn_among_its_repeats_by_its_output() -> None:
    results, evidence = _campaign(6, arms=1)
    for case in range(6):
        second, read = _result(CONTROL_RUN, case, f"another reply to {case}", passes=True, repeat=2)
        results.append(second)
        evidence[second.id] = read
    read = _draw(results, evidence, arms=1)
    assert len(read.items) == 6 and all(len(item.outputs) == 1 for item in read.items)
    picked = {item.outputs[0].evidence.artifact.split(" reply")[0] for item in read.items}
    assert picked == {"old", "another"}, "the pick varies by case; it is not always the first repeat"
    assert read.ask == ["verdicts"], "one arm has no pair to prefer between"


def test_a_case_an_arm_did_not_answer_and_identical_outputs_are_counted_and_never_paired() -> None:
    results, evidence = _campaign(8)
    # Case 0: the candidate's answer was never judged. Case 1: no evidence was stored for it. Case 2: both arms said
    # the same thing, byte for byte.
    results = [r for r in results if not (r.eval_run_id == CANDIDATE_RUN and r.test_case_id == "case-000")]
    lost = next(r for r in results if r.eval_run_id == CANDIDATE_RUN and r.test_case_id == "case-001")
    del evidence[lost.id]
    same = next(r for r in results if r.eval_run_id == CANDIDATE_RUN and r.test_case_id == "case-002")
    twin, twin_read = _result(CANDIDATE_RUN, 2, "old reply to 2", passes=True)
    results[results.index(same)] = twin
    evidence[twin.id] = twin_read
    read = _draw(results, evidence)
    assert read.draw.excluded == {
        core.EXCLUDED_NO_ANSWER: 1,
        core.EXCLUDED_NO_EVIDENCE: 1,
        core.EXCLUDED_IDENTICAL: 1,
    }
    assert read.draw.population == 6
    single = next(item for item in read.items if item.case_id == "case-002")
    assert [output.slot for output in single.outputs] == ["A"]
    sheet = blind_sheet(read)
    asked = {item.item: item.ask.preference for item in sheet.items}
    assert asked[single.item] is False and sum(asked.values()) == 5


def test_a_read_is_one_arm_or_a_control_against_one_candidate() -> None:
    results, evidence = _campaign(4)
    with pytest.raises(ValueError, match="a control against one candidate"):
        draw_blind_read(
            results,
            evidence,
            [GROUNDED],
            [BlindArm(key="a", role="candidate", run_ids=["x"]), BlindArm(key="b", role="candidate", run_ids=["y"])],
            scope_id="scope",
            created_by="test",
        )
    with pytest.raises(ValueError, match="named by two arms"):
        draw_blind_read(
            results,
            evidence,
            [GROUNDED],
            [BlindArm(key="a", role="control", run_ids=["x"]), BlindArm(key="b", role="candidate", run_ids=["x"])],
            scope_id="scope",
            created_by="test",
        )


# --- the sheet ------------------------------------------------------------------------------------------------------


def test_the_sheet_s_bytes_hold_no_arm_run_result_or_case_id_and_no_judge_score(tmp_path: Path) -> None:
    results, evidence = _campaign(12, judge_fails=frozenset({1, 4}))
    read = _draw(results, evidence)
    written = write_blind_read(tmp_path, read)
    assert sorted(path.name for path in written.values()) == ["blind_read.csv", "blind_read.html"]
    forbidden = [
        CONTROL_RUN,
        CANDIDATE_RUN,
        "JUDGE-REASONING-MARK",
        JUDGE_MODEL,
        *(result.id for result in results),
        *(result.test_case_id for result in results),
        '"v1"',
        '"v2"',
        "variant",
        "result_id",
        "run_id",
        "case_id",
        '"arm"',
        "score",
    ]
    for path in written.values():
        data = path.read_bytes()
        for mark in forbidden:
            assert mark.encode() not in data, f"{path.name} carries {mark!r}"
    # The CSV's columns are the sheet's, and an arm's key appears in no cell of them.
    rows = list(csv.DictReader(io.StringIO(written["blind_read_csv"].read_text(encoding="utf-8"))))
    assert list(rows[0]) == list(SHEET_COLUMNS) and len(rows) == 24
    assert {cell for row in rows for cell in row.values()}.isdisjoint({"v1", "v2"})
    assert {row["label"] for row in rows} == {""} and {row["prefer"] for row in rows} == {""}
    assert [row["item"] for row in rows[::2]] == [item.item for item in read.items], "rows are in the drawn order"
    sheet = json.loads(re.search(r'id="sheet">(.*?)</script>', blind_read_html(blind_sheet(read)), re.S).group(1))  # type: ignore[union-attr]
    assert set(sheet) == {"read_id", "name", "instructions", "criteria", "items", "answered", "next"}
    assert {key for item in sheet["items"] for key in item} == {"item", "case_material", "outputs", "ask"}
    assert {key for item in sheet["items"] for output in item["outputs"] for key in output} == {
        "slot",
        "text",
        "output_fingerprint",
        "criterion_fingerprints",
    }


def test_the_sheet_reports_one_rater_s_progress_and_the_first_unanswered_item() -> None:
    results, evidence = _campaign(6)
    read = _draw(results, evidence)
    read = _with(read, [_answer(read, 0), _answer(read, 1), _answer(read, 0, rater="someone else")])
    sheet = blind_sheet(read, rater="owner")
    assert sheet.answered == [read.items[0].item, read.items[1].item] and sheet.next == read.items[2].item
    assert blind_sheet(read).answered == [] and blind_sheet(read).next == read.items[0].item
    assert [level.value for level in sheet.criteria[0].levels] == ["pass", "fail"]


def test_the_page_is_one_static_file_that_loads_nothing_and_guards_every_use_of_storage() -> None:
    results, evidence = _campaign(4)
    page = blind_read_html(blind_sheet(_draw(results, evidence)))
    assert "http://" not in page and "https://" not in page and "//cdn" not in page
    assert not re.search(r"<(link|img|iframe|object|embed|video|audio)\b", page)
    assert not re.search(r"<script[^>]*\bsrc=", page) and "@import" not in page and "url(" not in page
    assert "fetch(" not in page and "XMLHttpRequest" not in page and "sendBeacon" not in page
    assert "Download labels.csv" in page
    script = page.split("<script>", 1)[1]
    uses = [match.start() for match in re.finditer(r"localStorage", script)]
    assert len(uses) == 2, "one read and one write"
    for at in uses:
        before = script[:at]
        assert before.rfind("try {") > before.rfind("catch ("), "each use sits inside a try"
        assert script.find("catch (unavailable)", at) != -1
    # An output that tries to close the script element stays text.
    hostile, read = _result(CONTROL_RUN, 0, "</script><script>alert(1)</script>", passes=True)
    hostile_page = blind_read_html(
        blind_sheet(draw_blind_read([hostile], {hostile.id: read}, [GROUNDED], _arms(1), scope_id="s", created_by="t"))
    )
    assert hostile_page.count("<script") == 2 and "alert(1)" in hostile_page


@pytest.mark.browser
def test_the_page_opened_from_disk_saves_labels_that_read_back_as_answers_from_the_top(tmp_path: Path) -> None:
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    results, evidence = _campaign(6)
    read = _draw(results, evidence)
    page = write_blind_read(tmp_path, read)["blind_read"]
    # Storage is taken away first, as a locked-down browser does; then the first three cases are answered through
    # the page's own controls and the labels saved.
    probe = """() => {
      Object.defineProperty(window, "localStorage", { get() { throw new Error("denied"); } });
      const pick = (value) => {
        const groups = {};
        document.querySelectorAll("#item input[type=radio]").forEach((input) => { (groups[input.name] ||= []).push(input); });
        Object.values(groups).forEach((inputs) => {
          const wanted = inputs.find((input) => input.value === value) || inputs[0];
          wanted.click();
        });
      };
      const name = document.getElementById("rater");
      name.value = "B. Reader"; name.dispatchEvent(new Event("input"));
      for (let n = 0; n < 3; n += 1) { pick("pass"); document.getElementById("next").click(); }
      document.getElementById("download").click();
      return {
        csv: document.getElementById("csv").value,
        status: document.getElementById("status").textContent,
        focusable: Array.from(document.querySelectorAll("#item input, button")).every((node) => node.tabIndex >= 0),
        text: document.body.innerText,
      };
    }"""
    shown, errors = rendered(page, probe)
    assert not errors, errors
    assert "Answered 3, 3 of them from the top" in shown["status"] and shown["focusable"]
    assert "v1" not in shown["text"] and "v2" not in shown["text"] and CONTROL_RUN not in shown["text"]
    labels = tmp_path / "labels.csv"
    labels.write_text(shown["csv"], encoding="utf-8")
    # The saved labels carry what each was given on, so they still pair when the answers change.
    saved = list(csv.DictReader(io.StringIO(shown["csv"])))
    assert {row["output_fingerprint"] for row in saved} == {
        output.output_fingerprint for item in read.items[:3] for output in item.outputs
    }
    assert all(row["criterion_fingerprint"] for row in saved)
    answers = read_labels(labels, read, results).answers
    assert [answer.item for answer in answers] == [item.item for item in read.items[:3]]
    assert {answer.rater for answer in answers} == {"B. Reader"}
    assert all(len(answer.verdicts) == 2 and answer.prefer == "A" for answer in answers)
    assert read_sample(_with(read, answers), "B. Reader").sample == "random"


# --- the standing ---------------------------------------------------------------------------------------------------


def test_answers_from_the_top_are_a_random_sample_and_any_other_set_is_not_drawn_at_random() -> None:
    results, evidence = _campaign(10)
    read = _draw(results, evidence)
    top = _with(read, [_answer(read, n) for n in range(4)])
    standing = read_sample(top, "owner")
    assert (standing.sample, standing.answered, standing.items, standing.skipped) == ("random", 4, 10, 0)
    assert read_sample(_with(read, [_answer(read, n) for n in (0, 1, 3)]), "owner").sample == "selected"
    assert read_sample(_with(read, [_answer(read, 5)]), "owner").sample == "selected"
    assert read_sample(read, "owner").sample == "none"
    # A skip is an answer: the prefix stays one, and the standing says how many were skipped.
    skipped = _with(
        read, [_answer(read, 0), BlindAnswer(item=read.items[1].item, rater="owner", skipped=True), _answer(read, 2)]
    )
    standing = read_sample(skipped, "owner")
    assert (standing.sample, standing.answered, standing.skipped) == ("random", 3, 1)
    with pytest.raises(ValidationError, match="skipped"):
        BlindAnswer(item="x", rater="owner", skipped=True, prefer="A")


# --- the close ------------------------------------------------------------------------------------------------------


def test_closing_gives_person_ratings_that_name_the_read_agreement_with_a_tier_and_a_preference() -> None:
    fails = frozenset({0, 3, 5, 8, 11, 14, 17, 20})
    results, evidence = _campaign(24, judge_fails=fails)
    read = _draw(results, evidence)
    # The person fails exactly what the judge failed, and prefers the candidate on every pair.
    answers = [_answer(read, n, scores={"v1": int(int(read.items[n].case_id[-3:]) not in fails)}) for n in range(24)]
    closed = close_blind_read(_with(read, answers), results, closed_by="owner", closed_at="t2")
    assert len(closed.ratings) == 48 and closed.random == frozenset(rating.id for rating in closed.ratings)
    assert {(r.rater, r.rater_kind, r.source, r.collected_by) for r in closed.ratings} == {
        ("owner", "person", f"blind_read:{read.id}", "owner")
    }
    assert all(rating.label_key is not None and rating.rated_withheld for rating in closed.ratings)
    report = closed.read.report
    assert report is not None and closed.read.closed_by == "owner" and closed.read.closed_at == "t2"
    (standing,) = report.samples
    assert (standing.sample, standing.answered) == ("random", 24)
    (stored,) = report.agreement
    (reading,) = closed.agreement
    assert reading.counts == stored, "the close's reading is the stored counts, with what the print reads beside them"
    assert (reading.dim, reading.anchor, reading.decides, reading.anchor_names) == (
        GROUNDED.name,
        "person",
        True,
        ["owner"],
    )
    assert (reading.pairs, reading.results, reading.cases, reading.blind_pairs, reading.random_pairs) == (
        48,
        48,
        24,
        48,
        48,
    )
    assert reading.kappa == pytest.approx(1.0) and reading.exact_agreement == 1.0
    assert (reading.tier_before, reading.tier, reading.calibration) == ("undetermined", "calibrated", "met")
    table = reading.table
    assert (table.scores, table.counts) == ([0, 1], [[8, 0], [0, 40]]), (
        "fail and pass, the reader down, the judge across"
    )
    cells = core.pass_fail_counts(table)
    assert cells is not None
    assert (cells.both_pass, cells.both_fail, cells.reader_fail_judge_pass, cells.reader_pass_judge_fail) == (
        40,
        8,
        0,
        0,
    )
    assert core.agreed_pairs(table) == sum(map(sum, table.counts)) == 48
    (preference,) = report.preferences
    assert (preference.anchor, preference.random, preference.control, preference.candidate) == (
        "person",
        True,
        "v1",
        "v2",
    )
    assert (preference.prefer_candidate, preference.prefer_control, preference.ties, preference.test) == (
        24,
        0,
        0,
        "exact_sign",
    )
    assert preference.p == pytest.approx(2 / 2**24) and preference.verdict == "improved"
    assert preference.slot_a_share == 0.5 and preference.slot_choices == 24, "balanced slots, so no lean"


def test_a_judge_that_passes_everything_is_shown_by_the_table_where_kappa_says_nothing() -> None:
    """The finding the only hand-built blind read made: 27 of 30 agreement, the judge catching none of 3 failures."""
    results, evidence = _campaign(30, arms=1)
    read = _draw(results, evidence, arms=1)
    answers = [_answer(read, n, scores={"v1": int(n not in (2, 9, 20))}) for n in range(30)]
    closed = close_blind_read(_with(read, answers), results)
    (reading,) = closed.agreement
    assert reading.exact_agreement == pytest.approx(27 / 30) and reading.kappa == pytest.approx(0.0)
    assert reading.tier != "calibrated"
    cells = core.pass_fail_counts(reading.table)
    assert cells is not None and (cells.reader_fail_judge_pass, cells.both_fail) == (3, 0), "caught 0 of 3"
    assert core.single_verdicts(reading.table) == (None, 1), "the judge gave every answer one verdict: pass"
    lines = judge_lines(closed, labels_name="labels.csv", labels_there=True)
    assert lines[0] == "judge: 30 of 30 cases read blind by owner, in the drawn order (a random sample)"
    # #1208: counts, the catch rate and the false-fail rate with exact intervals, and the one verdict the judge gave.
    assert lines[1].endswith(
        "30 results from 30 cases: the judge caught 0 of the 3 the reader failed (exact 95% interval 0 to "
        "71%); the judge failed 0 of the 27 the reader passed (exact 95% interval 0 to 13%); the judge gave every "
        "answer one verdict, pass"
    )
    # No kappa, bar, tier word or estimate of results still needed is printed.
    for word in ("kappa", "calibrated", "undetermined", "bar of", "more results", "read further"):
        assert not any(word in line for line in lines), word
    # One owner of the rate statements: the console says each in the words the host's response serves.
    rates = blind_rates(reading.table, reader="the reader", judge="the judge")
    assert rates.catch is not None and rates.false_fail is not None
    assert f"the judge {rates.catch.said}; the judge {rates.false_fail.said}; {'; '.join(rates.said)}" in lines[1]


def test_the_console_and_the_host_response_state_a_check_s_rates_in_one_set_of_words() -> None:
    """Every rate statement the console prints is one :func:`blind_rates` makes, which the host's response serves:
    where the reader failed nothing (no catch rate), and where the reader passed nothing (no false-fail rate)."""
    results, evidence = _campaign(30, arms=1)
    read = _draw(results, evidence, arms=1)
    for score in (1, 0):
        answers = [_answer(read, n, scores={"v1": score}) for n in range(30)]
        closed = close_blind_read(_with(read, answers), results)
        (reading,) = closed.agreement
        line = judge_lines(closed, labels_name="labels.csv", labels_there=True)[1]
        rates = blind_rates(reading.table, reader="the reader", judge="the judge")
        assert (rates.catch is None) == (score == 1) and (rates.false_fail is None) == (score == 0)
        said = [f"the judge {rate.said}" for rate in (rates.catch, rates.false_fail) if rate is not None]
        for statement in [*said, *rates.said]:
            assert statement in line, (score, statement, line)


def test_the_preference_is_withheld_until_the_planned_length_and_reads_only_those_items() -> None:
    results, evidence = _campaign(40)
    read = _draw(results, evidence, planned_items=12)
    assert read.draw.planned_items == 12
    partial = close_blind_read(_with(read, [_answer(read, n) for n in range(11)]), results).read.report
    assert partial is not None
    (withheld,) = partial.preferences
    assert (withheld.verdict, withheld.planned, withheld.answered) == (None, 12, 11)
    assert withheld.withheld and withheld.state == StateOf(state="withheld", reason="unfinished")
    assert (withheld.items, withheld.prefer_candidate, withheld.p, withheld.slot_a_share) == (0, 0, None, None)
    lines = judge_lines(
        close_blind_read(_with(read, [_answer(read, n) for n in range(11)]), results),
        labels_name="labels.csv",
        labels_there=True,
        shown={"v1": "v1", "v2": "v2"},
    )
    assert "preference: shown when the first 12 cases are answered (11 of 12)" in lines
    assert lines[-1] == "next: keep going from case 12 in blind_read.html"
    assert not any("p 0" in line or "over v1" in line for line in lines), "no direction or p before the planned length"
    # At the planned length it is read; answers past it change nothing in it.
    at = [_answer(read, n, prefer_arm="v2" if n < 9 else "v1") for n in range(12)]
    past = [*at, *(_answer(read, n, prefer_arm="v1") for n in range(12, 40))]
    first = close_blind_read(_with(read, at), results).read.report
    later = close_blind_read(_with(read, past), results).read.report
    assert first is not None and later is not None
    assert first.preferences == later.preferences
    (preference,) = first.preferences
    assert (preference.prefer_candidate, preference.prefer_control, preference.planned) == (9, 3, 12)
    assert preference.p == pytest.approx(exact_sign_test_p(9, 12)) and preference.verdict == "not_separated"
    # Out of order, the planned items are not all answered, so it stays withheld.
    hunted = close_blind_read(_with(read, [_answer(read, n) for n in range(20, 40)]), results).read.report
    assert hunted is not None and hunted.preferences[0].withheld and hunted.preferences[0].verdict is None
    with pytest.raises(ValidationError, match="withheld"):
        BlindPreference.model_validate({**withheld.model_dump(), "prefer_candidate": 3})


def test_the_exact_sign_test_is_the_design_s_table_and_five_items_cannot_separate() -> None:
    fewest = {6: 6, 10: 9, 15: 12, 20: 15, 30: 21, 40: 27, 60: 39}
    for n, k in fewest.items():
        p, short = exact_sign_test_p(k, n), exact_sign_test_p(k - 1, n)
        assert p is not None and short is not None and p < 0.05 <= short, (n, k)
    assert exact_sign_test_p(12, 15) == pytest.approx(0.0352, abs=1e-4)
    assert exact_sign_test_p(5, 5) == pytest.approx(0.0625) and core.PREFERENCE_MIN_ITEMS == 6
    assert exact_sign_test_p(0, 0) is None and exact_sign_test_p(3, 6) == 1.0
    results, evidence = _campaign(5)
    read = _draw(results, evidence)
    report = close_blind_read(_with(read, [_answer(read, n) for n in range(5)]), results).read.report
    assert report is not None and (report.preferences[0].verdict, report.preferences[0].p) == (None, None)
    assert report.preferences[0].too_few
    assert report.preferences[0].state == StateOf(state="not_available", reason="too_few")


def test_ties_are_counted_and_never_split_and_skips_are_said() -> None:
    results, evidence = _campaign(12)
    read = _draw(results, evidence)
    answers = [_answer(read, n, prefer_arm=None if n in (0, 1) else "v2") for n in range(11)]
    answers.append(BlindAnswer(item=read.items[11].item, rater="owner", skipped=True))
    closing = close_blind_read(_with(read, answers), results)
    closed = closing.read
    assert closed.report is not None
    (preference,) = closed.report.preferences
    assert (preference.prefer_candidate, preference.prefer_control, preference.ties, preference.missing) == (9, 0, 2, 1)
    assert preference.p == pytest.approx(exact_sign_test_p(9, 9))
    lines = judge_lines(closing, labels_name="labels.csv", labels_there=True, shown={})
    assert lines[0] == (
        "judge: 12 of 12 cases read blind by owner, in the drawn order (a random sample of the cases answered; 1 skipped)"
    )
    assert (
        "preference: v2 over v1 on 9 of 9 (2 ties), 1 with no preference given, sign test p 0.004: v2 preferred"
        in lines
    )


def test_several_people_s_shares_are_read_by_the_bounded_test() -> None:
    results, evidence = _campaign(12)
    read = _draw(results, evidence)
    answers = [_answer(read, n, rater=rater) for n in range(12) for rater in ("ann", "bo")]
    report = close_blind_read(_with(read, answers), results).read.report
    assert report is not None
    (preference,) = report.preferences
    assert preference.test == "bounded_share" and preference.anchor_names == ["ann", "bo"]
    assert preference.prefer_candidate == 12 and preference.p is not None


#: Blind reads a durable 17.4 build closed and stored, written by that build (``evals/main`` at ``be8de80cb``) through
#: ``packages/evals/scripts/write_blind_read_v17_4.py``, which names how to run it there: every path a 17.4 close stored
#: a preference by (the exact sign test, the bounded test on several raters' shares, the anytime test on a read closed
#: early), skipped items on each, a model reviewer's preference under the model that served it, and each outcome.
#: Each carries the p, verdict and state its close computed, as 17.4 stored them.
STORED_AT_17_4 = Path(__file__).resolve().parent / "fixtures" / "blind_read_preferences" / "v17_4.json"

#: The frozen durable corpus's 17.4 blind read, whose preference is too few to read.
DURABLE_17_4 = Path(__file__).resolve().parent / "fixtures" / "durable_documents" / "v17" / "blind_read.json"


def _stored_at_17_4() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = json.loads(STORED_AT_17_4.read_text(encoding="utf-8"))
    return [*documents, json.loads(DURABLE_17_4.read_text(encoding="utf-8"))]


def _closed_again(read: BlindRead) -> BlindRead:
    """``read``'s answers closed again by this build, by whom and when it was closed."""
    reopened = read.model_copy(update={"report": None, "closed_at": None, "closed_by": None, "closed_early": False})
    return close_blind_read(reopened, [], closed_by=read.closed_by, closed_at=read.closed_at).read


def _preferences(read: BlindRead) -> list[BlindPreference]:
    assert read.report is not None
    return list(read.report.preferences)


def test_the_17_4_documents_cover_every_path_a_close_stores_a_preference_by() -> None:
    """The corpus the recomputation is held on: each test, a read closed early, a skip on each test, a model's
    preference under its served model, and every outcome, each as 17.4 stored it."""
    stored = [preference for document in _stored_at_17_4() for preference in document["report"]["preferences"]]
    assert all(document["schema_version"] == 17 and document["schema_minor"] == 4 for document in _stored_at_17_4())
    assert {preference["test"] for preference in stored} == {"exact_sign", "bounded_share", "anytime"}
    assert {preference["test"] for preference in stored if preference["missing"]} == {
        "exact_sign",
        "bounded_share",
        "anytime",
    }, "a skipped item on each test"
    assert any(document.get("closed_early") for document in _stored_at_17_4())
    assert any(preference["anchor"] == "model" and preference["served_model"] for preference in stored)
    assert {preference["verdict"] for preference in stored} == {"improved", "regressed", "not_separated", None}
    assert {(preference["state"] or {}).get("reason") for preference in stored} == {None, "too_few"}
    assert {preference["test"] for preference in stored if preference["state"]} >= {"exact_sign", "anytime"}


def test_a_stored_preference_holds_its_counts_and_no_p_verdict_or_too_few_state() -> None:
    """#1236: a preference stores what its test reads (each item's share, and the anytime test's planned pairs) and no
    p, verdict or too-few state. A read stored that way reads them off its counts, equal to what its close read."""
    tests_seen: set[str] = set()
    for document in _stored_at_17_4():
        closed = _closed_again(BlindRead.from_dict(document))
        built_all = _preferences(closed)
        written_all = closed.to_dict()["report"]["preferences"]
        loaded = _preferences(BlindRead.from_dict(closed.to_dict()))
        assert len(built_all) == len(written_all) == len(loaded) == len(document["report"]["preferences"])
        for built, written, stored in zip(built_all, written_all, loaded, strict=True):
            assert "p" not in written and "verdict" not in written, "the judgement is read off the counts, never stored"
            assert written["state"] is None, "too few to test is read off the counts, never stored"
            assert written["shares"] == built.shares and len(written["shares"]) == written["items"]
            assert ("stake_items" in written) == (built.test == "anytime")
            assert (stored.p, stored.verdict, stored.state) == (None, None, None), "it carries no judgement"
            read = read_preference(stored)
            assert (read.p, read.verdict, read.state) == (built.p, built.verdict, built.state)
            tests_seen.add(built.test)
    assert tests_seen == {"exact_sign", "bounded_share", "anytime"}


def test_p_verdict_and_state_read_off_the_counts_equal_what_a_17_4_build_stored() -> None:
    """Every blind read a 17.4 build stored, closed again from its answers and stored as counts: the p, verdict and
    state read off those counts are the ones 17.4 stored, to the bit, on every preference of every document."""
    verdicts: set[str | None] = set()
    for document in _stored_at_17_4():
        again = BlindRead.from_dict(_closed_again(BlindRead.from_dict(document)).to_dict())
        for as_stored, recomputed in zip(document["report"]["preferences"], _preferences(again), strict=True):
            read = read_preference(recomputed)

            state = None if read.state is None else read.state.model_dump()
            assert (read.p, read.verdict, state) == (as_stored["p"], as_stored["verdict"], as_stored["state"]), (
                document["name"]
            )
            assert (read.anchor, read.served_model, read.test, read.missing) == (
                as_stored["anchor"],
                as_stored["served_model"],
                as_stored["test"],
                as_stored["missing"],
            )
            assert (read.prefer_candidate, read.prefer_control, read.ties) == (
                as_stored["prefer_candidate"],
                as_stored["prefer_control"],
                as_stored["ties"],
            )
            verdicts.add(read.verdict)
    assert verdicts == {"improved", "regressed", "not_separated", None}


def test_a_preference_a_17_4_build_stored_reads_and_writes_the_p_and_verdict_it_carries() -> None:
    """A 17.4 preference holds no shares, so nothing is recomputed: it reads with the p, verdict and state it stored,
    even where its counts would read otherwise, and writes them back as stored."""
    for document in _stored_at_17_4():
        read = BlindRead.from_dict(document)
        for as_stored, stored in zip(document["report"]["preferences"], _preferences(read), strict=True):
            assert stored.shares is None and (stored.p, stored.verdict) == (as_stored["p"], as_stored["verdict"])
            assert read_preference(stored) is stored
        assert read.to_dict()["report"]["preferences"] == document["report"]["preferences"], "written back as stored"
    sign = json.loads(STORED_AT_17_4.read_text(encoding="utf-8"))[0]
    (preference,) = sign["report"]["preferences"]
    assert (preference["test"], preference["verdict"]) == ("exact_sign", "improved")
    preference.update(p=0.5, verdict="not_separated")
    (believed,) = _preferences(BlindRead.from_dict(sign))
    assert (read_preference(believed).p, read_preference(believed).verdict) == (0.5, "not_separated"), (
        "what 17.4 stored is what it reads, never recomputed"
    )


def test_a_stored_too_few_state_its_counts_contradict_is_refused() -> None:
    """Too few to test is a fact of the counts: under six items preferring a side. A 17.4 preference carries the state
    as stored, so one its counts contradict either way is refused; one holding its shares is read too few off them,
    and a stored too-few state beside counts that are not too few is refused."""
    documents = json.loads(STORED_AT_17_4.read_text(encoding="utf-8"))
    sign = next(document for document in documents if document["name"] == "sign, improved, one skipped")
    (separated,) = sign["report"]["preferences"]
    with pytest.raises(ValidationError, match="too few to test only below 6"):
        BlindPreference.model_validate(
            {**separated, "p": None, "verdict": None, "state": {"state": "not_available", "reason": "too_few"}}
        )
    few = next(document for document in documents if document["name"] == "sign, too few")
    (too_few,) = few["report"]["preferences"]
    assert too_few["state"] == {"state": "not_available", "reason": "too_few"}
    with pytest.raises(ValidationError, match="fewer than 6 items that prefer a side is too few"):
        BlindPreference.model_validate({**too_few, "p": 0.06, "verdict": "not_separated", "state": None})

    # Stored as counts: the close writes no state, and the read decides too few off them.
    (counted,) = _preferences(_closed_again(BlindRead.from_dict(few)))
    written = counted.to_dict()
    assert written["state"] is None and counted.too_few
    loaded = BlindPreference.model_validate(written)
    assert not loaded.too_few and read_preference(loaded).too_few
    assert (read_preference(loaded).p, read_preference(loaded).verdict) == (None, None)
    assert BlindPreference.model_validate({**written, "state": too_few["state"]}).too_few, "agrees with its counts"
    (read_off,) = _preferences(_closed_again(BlindRead.from_dict(sign)))
    with pytest.raises(ValidationError, match="too few to test only below 6"):
        BlindPreference.model_validate({**read_off.to_dict(), "state": too_few["state"]})


def test_a_sign_test_preference_holds_only_shares_one_answer_can_give() -> None:
    """The exact sign test reads one answer per item, so each share is 1, 0 or a half; a mean of several raters'
    answers on it is refused, never scored by the sign test."""
    results, evidence = _campaign(12)
    read = _draw(results, evidence)
    closed = close_blind_read(_with(read, [_answer(read, n) for n in range(12)]), results).read
    (built,) = _preferences(closed)
    written = built.to_dict()
    assert built.test == "exact_sign" and set(written["shares"]) == {1.0}

    with pytest.raises(ValidationError, match="one answer per item, so each share is 1, 0 or a half"):
        BlindPreference.model_validate({**written, "shares": [0.75, *written["shares"][1:]]})
    shared = BlindPreference.model_validate(
        {**written, "test": "bounded_share", "shares": [0.75, *written["shares"][1:]]}
    )
    assert shared.shares is not None and shared.shares[0] == 0.75, "a mean of several raters' answers is a share"


def test_a_p_or_verdict_beside_the_shares_is_an_echo_and_counts_must_be_the_shares() -> None:
    results, evidence = _campaign(12)
    read = _draw(results, evidence)
    closed = close_blind_read(_with(read, [_answer(read, n) for n in range(12)]), results).read
    assert closed.report is not None
    (built,) = closed.report.preferences
    written = built.to_dict()

    echoed = BlindPreference.model_validate({**written, "p": 0.9, "verdict": "regressed"})

    assert (echoed.p, echoed.verdict) == (None, None), "discarded, never believed"
    assert (read_preference(echoed).p, read_preference(echoed).verdict) == (built.p, built.verdict)
    with pytest.raises(ValidationError, match="counts for each side"):
        BlindPreference.model_validate({**written, "prefer_candidate": written["prefer_candidate"] - 1})
    with pytest.raises(ValidationError, match="anytime test states the planned pair items"):
        BlindPreference.model_validate({**written, "stake_items": 12})


def test_an_answer_the_read_does_not_ask_is_refused() -> None:
    results, evidence = _campaign(4)
    read = _draw(results, evidence)
    stray = BlindAnswer(
        item="not-an-item", rater="owner", verdicts=[BlindVerdict(slot="A", dim=GROUNDED.name, score=1)]
    )
    with pytest.raises(ValueError, match="no item of blind read"):
        close_blind_read(_with(read, [stray]), results)
    off = BlindAnswer(
        item=read.items[0].item, rater="owner", verdicts=[BlindVerdict(slot="A", dim=GROUNDED.name, score=4)]
    )
    with pytest.raises(ValueError, match="not on the pass_fail scale"):
        close_blind_read(_with(read, [off]), results)


def test_the_stored_report_holds_counts_and_no_kappa_bar_tier_or_results_still_needed() -> None:
    """#1208: a judge's agreement is stored as counts. The one real read gave kappa 0 or undefined at 10 to 20 of 20
    agreed, because one side gave a single verdict; a stored kappa, bar verdict or tier would freeze that judgement
    into a durable document. The counts are stored, and everything else is read off them."""
    results, evidence = _campaign(20, arms=1, judge_fails=frozenset())
    read = _draw(results, evidence, arms=1)
    # The reader fails two answers; the judge passed every one: a constant judge.
    answers = [_answer(read, n, scores={"v1": int(n not in (3, 11))}) for n in range(20)]
    closed = close_blind_read(_with(read, answers), results, closed_by="owner")
    stored = closed.read.to_dict()["report"]
    (agreement,) = stored["agreement"]
    assert set(agreement) == {
        "dim",
        "scale",
        "judge_model",
        "anchor",
        "anchor_names",
        "served_model",
        "pairs",
        "results",
        "cases",
        "blind_pairs",
        "random_pairs",
        "table",
    }
    assert agreement["table"] == {"scores": [0, 1], "counts": [[0, 2], [0, 18]]}
    text = json.dumps(stored)
    for word in ("kappa", "tier", "calibration", "results_needed", "decides", "exact_agreement", "lower", "detection"):
        assert word not in text, word
    report = closed.read.report
    assert report is not None
    assert core.single_verdicts(report.agreement[0].table) == (None, 1), "the judge gave every answer one verdict"


def test_a_1_5_criterion_stores_its_full_rating_cross_table() -> None:
    results: list[EvalResult] = []
    evidence: dict[str, JudgeEvidence] = {}
    for case in range(10):
        result, read_of = _result(CONTROL_RUN, case, f"reply {case}", passes=True, tone=1 + case % 5)
        results.append(result)
        evidence[result.id] = read_of
    read = draw_blind_read(results, evidence, [TONE], _arms(1), scope_id="scope", created_by="test", created_at="t0")
    by_case = {item.item: int(item.case_id[-3:]) for item in read.items}
    # The reader gives one level above the judge's, capped at 5.
    answers = [
        BlindAnswer(
            item=item.item,
            rater="owner",
            answered_at="t1",
            verdicts=[BlindVerdict(slot="A", dim=TONE.name, score=min(5, 2 + by_case[item.item] % 5))],
        )
        for item in read.items
    ]
    closed = close_blind_read(_with(read, answers), results)
    report = closed.read.report
    assert report is not None
    (agreement,) = report.agreement
    assert agreement.table.scores == [1, 2, 3, 4, 5]
    # Rows: the reader's level; columns: the judge's. Each judge level held twice.
    assert agreement.table.counts == [
        [0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 2],
    ]
    assert core.pass_fail_counts(agreement.table) is None, "a 1-5 criterion has no fixed pass"
    assert (core.agreed_pairs(agreement.table), sum(map(sum, agreement.table.counts))) == (2, 10)
    # The console's 1-5 line: why neither rate exists, in the owner's words, then the agreed count and the cross-table.
    lines = judge_lines(closed, labels_name="labels.csv", labels_there=True)
    assert lines[1] == (
        "  answer tone (judged)  10 results from 10 cases: a 1-5 criterion has no fixed pass, so no catch rate: the "
        "table counts every pair; the same score on 2 of 10; by the reader's score, the judge's 1 to 5: 1: 0 0 0 0 0; "
        "2: 2 0 0 0 0; 3: 0 2 0 0 0; 4: 0 0 2 0 0; 5: 0 0 0 2 2"
    )


def test_a_result_judged_on_other_wording_of_a_criterion_is_never_shown_under_the_new_wording() -> None:
    """A template is edited in place. A rating is bound to the criterion the judge read, so a result judged on the
    old wording, shown under the new, would store the reader's answer to the new question as a label of the old one.
    Such a result is left out of the draw and counted; every output shown is one the judge read in the words shown."""
    results, evidence = _campaign(8, arms=1)
    reworded = GROUNDED.model_copy(update={"description": "Every fact is in the policy, and the policy is cited."})
    nothing = draw_blind_read(
        results, evidence, [reworded], _arms(1), scope_id="scope", created_by="test", created_at="t0"
    )
    assert nothing.items == [] and nothing.draw.excluded == {core.EXCLUDED_REWORDED: 8}
    # Three answers judged before the edit, on the old wording; the rest after it, on the words the read shows.
    older = GROUNDED.model_copy(update={"description": "Every fact is in the policy text."})
    mixed: list[EvalResult] = []
    for result in results:
        if result.test_case_id in {"case-001", "case-004", "case-006"}:
            stale = label_key_of(evidence[result.id], older).criterion_fingerprint
            result = result.model_copy(
                update={
                    "rubric_scores": [
                        score.model_copy(update={"criterion_fingerprint": stale}) for score in result.rubric_scores
                    ]
                }
            )
        mixed.append(result)
    read = _draw(mixed, evidence, arms=1)
    assert read.draw.excluded == {core.EXCLUDED_REWORDED: 3}
    assert {item.case_id for item in read.items}.isdisjoint({"case-001", "case-004", "case-006"})
    shown = label_key_of(evidence[results[0].id], GROUNDED).criterion_fingerprint
    assert {fp for item in read.items for output in item.outputs for fp in output.criterion_fingerprints.values()} == {
        shown
    }
    closed = close_blind_read(_with(read, [_answer(read, n) for n in range(len(read.items))]), results)
    assert {rating.criterion_fingerprint for rating in closed.ratings} == {shown}
    # Stored evidence that is not the text the judge read is left out the same way, and counted.
    altered = dict(evidence)
    altered[results[2].id] = JudgeEvidence(case_material="Customer question number 2.", artifact="another reply")
    other = _draw(results, altered, arms=1)
    assert other.draw.excluded == {core.EXCLUDED_OTHER_TEXT: 1}


# --- a model as the reader -------------------------------------------------------------------------------------------


def _reference(read: BlindRead, position: int, model: str, **kwargs: Any) -> BlindAnswer:
    kwargs.setdefault("reviewer_served_model", f"{model}-0613")
    return _answer(read, position, rater=model, made_by="model", **kwargs)


def test_a_reference_reviewer_is_read_apart_and_its_ratings_are_never_a_person_s() -> None:
    fails = frozenset({0, 3, 5, 8, 11, 14, 17, 20})
    results, evidence = _campaign(24, judge_fails=fails)
    read = _draw(results, evidence)
    reviewer = "other/reviewer-9"
    answers = [
        _reference(read, n, reviewer, scores={"v1": int(int(read.items[n].case_id[-3:]) not in fails)})
        for n in range(24)
    ]
    closed = close_blind_read(_with(read, answers), results)
    # Never a person's: an agent's rating, under the model's own id, with no relay record. That it is this read's
    # model reviewer's is held beside the ratings, by id, and is in no rater's name.
    assert {(r.rater, r.rater_kind, r.source) for r in closed.ratings} == {(reviewer, "agent", None)}
    assert dict(closed.reviewed) == {rating.id: reviewer for rating in closed.ratings}
    report = closed.read.report
    assert report is not None
    (stored,) = report.agreement
    assert stored.served_model == f"{reviewer}-0613", "a model's reading names the model that served it"
    (reading,) = closed.agreement
    assert (reading.anchor, reading.anchor_names, reading.decides, reading.tier) == (
        "model",
        [reviewer],
        True,
        "calibrated",
    )
    assert not reading.shares_family and report.reviewer_checks == [] and report.not_checked == {}
    (preference,) = report.preferences
    assert (preference.anchor, preference.anchor_names) == ("model", [reviewer])
    lines = judge_lines(closed, labels_name=None, labels_there=False, shown={})
    # No person has read, so the person's first step leads, never dropped beside a model's read.
    assert lines[0].startswith("judge: 0 of 24 cases read blind by a person: judge not checked (")
    lines = lines[1:]
    assert lines[0] == (
        f"judge: 24 of 24 cases read blind by a model ({reviewer}), in the drawn order (a random sample)"
    )
    assert lines[1] == f"  checked against a model ({reviewer}), not yet against people:"
    # A pair read: two results of each case, so the exact intervals' independence is said not to hold.
    assert lines[2].endswith(
        "48 results from 24 cases: the judge caught 8 of the 8 the model failed (exact 95% interval 63 to "
        "100%); the judge failed 0 of the 40 the model passed (exact 95% interval 0 to 9%); the intervals count "
        "each answer as independent, which answers on one case are not"
    )
    # No tier value or kappa is printed for a judge checked against a model: the line says what it was checked against.
    assert not any("calibrated" in line or "kappa" in line for line in lines)
    assert lines[3].startswith(f"preference of a model ({reviewer}), not a person: v2 over v1 on 24 of 24")
    assert not any(line.startswith("next:") for line in lines)


def test_a_model_that_is_the_judge_s_own_is_refused_and_one_of_its_family_is_said() -> None:
    results, evidence = _campaign(24)
    read = _draw(results, evidence)
    owned = close_blind_read(_with(read, [_reference(read, n, JUDGE_MODEL) for n in range(24)]), results)
    own = owned.read
    assert own.report is not None and own.report.agreement == [] and owned.agreement == ()
    assert own.report.not_checked == {JUDGE_MODEL: [core.REFUSED_SAME_MODEL]}
    assert (
        f"  not checked against a model ({JUDGE_MODEL}): the reviewer is the judge's own model, so its answers repeat "
        "the judge and are not read as a check of it"
    ) in judge_lines(owned, labels_name=None, labels_there=False)
    # Named as another model, and served by the judge's on ONE case of the 24: refused all the same.
    served = [
        _reference(read, n, "alias/reviewer", reviewer_served_model=JUDGE_MODEL if n == 5 else "alias/reviewer-1")
        for n in range(24)
    ]
    aliased = close_blind_read(_with(read, served), results).read.report
    assert aliased is not None and aliased.agreement == []
    assert aliased.not_checked == {"alias/reviewer": [core.REFUSED_SAME_MODEL]}
    # Named as the id the judge was ASKED as, which its scores' served model does not spell.
    asked = close_blind_read(
        _with(read, [_reference(read, n, "judge-alias") for n in range(24)]), results, judge_models=["judge-alias"]
    ).read.report
    assert asked is not None and asked.not_checked == {"judge-alias": [core.REFUSED_SAME_MODEL]}
    family = close_blind_read(_with(read, [_reference(read, n, "acme/reviewer-2") for n in range(24)]), results)
    assert family.agreement[0].shares_family
    assert core.shares_family("acme/reviewer-2", family.agreement[0].judge_model), "read off the stored names at load"
    assert any("one provider's models" in line for line in judge_lines(family, labels_name=None, labels_there=False))


def test_a_model_whose_served_model_is_not_recorded_on_every_answer_is_not_a_check_and_says_why() -> None:
    results, evidence = _campaign(24)
    read = _draw(results, evidence)
    # One answer of the 24 names no served model: the reviewer cannot be shown to be another model than the judge.
    partly = [
        _reference(read, n, "other/reviewer-9", **({"reviewer_served_model": None} if n == 7 else {}))
        for n in range(24)
    ]
    closing = close_blind_read(_with(read, partly), results)
    closed = closing.read
    assert closed.report is not None and closed.report.agreement == []
    assert closed.report.not_checked == {
        "other/reviewer-9": [core.REFUSED_UNRECORDED.format(side="the reviewer's answers")]
    }
    lines = judge_lines(closing, labels_name=None, labels_there=False)
    assert (
        "  not checked against a model (other/reviewer-9): which model served the reviewer's answers was not "
        "recorded, so the reviewer cannot be shown to be another model than the judge, and its answers are not read "
        "as a check of it"
    ) in lines
    # The judge's own served model unrecorded: the same, from the other side.
    unnamed = [
        result.model_copy(
            update={
                "rubric_scores": [score.model_copy(update={"served_model": None}) for score in result.rubric_scores]
            }
        )
        for result in results
    ]
    blank = close_blind_read(
        _with(read, [_reference(read, n, "other/reviewer-9") for n in range(24)]), unnamed
    ).read.report
    assert blank is not None and blank.agreement == []
    assert blank.not_checked == {"other/reviewer-9": [core.REFUSED_UNRECORDED.format(side="the judge's scores")]}


def test_a_person_s_reading_decides_once_it_is_decided_and_the_anchor_is_measured_against_people() -> None:
    fails = frozenset({0, 3, 5, 8, 11, 14, 17, 20})
    results, evidence = _campaign(24, judge_fails=fails)
    read = _draw(results, evidence)
    reviewer = "other/reviewer-9"

    def scores(n: int) -> dict[str, int]:
        return {"v1": int(int(read.items[n].case_id[-3:]) not in fails)}

    reference = [_reference(read, n, reviewer, scores=scores(n)) for n in range(24)]
    # Three person labels decide nothing: the reference tier stands, the person reading beside it.
    few = close_blind_read(_with(read, [*reference, *(_answer(read, n, scores=scores(n)) for n in range(3))]), results)
    assert few.read.report is not None
    by_anchor = {reading.anchor: reading for reading in few.agreement}
    assert (by_anchor["person"].calibration, by_anchor["person"].decides) == ("insufficient", False)
    assert (by_anchor["model"].tier, by_anchor["model"].decides) == ("calibrated", True)
    # The two are never pooled: each reading counts its own rater's pairs alone.
    assert (by_anchor["person"].pairs, by_anchor["model"].pairs) == (6, 48)
    (check,) = few.read.report.reviewer_checks
    assert (check.reviewer, check.served_model, check.dim, check.outputs) == (
        reviewer,
        f"{reviewer}-0613",
        GROUNDED.name,
        6,
    )
    assert core.agreed_pairs(check.table) == 6, "the reviewer gave every output the person's score"
    lines = judge_lines(few, labels_name="labels.csv", labels_there=True)
    assert f"  beside it, checked against a model ({reviewer}):" in lines
    assert any(
        line.startswith(f"    {reviewer} against people on answer grounded (judged): 6 outputs") for line in lines
    )
    # A whole person read decides it, whatever the reference says.
    full = close_blind_read(
        _with(read, [*reference, *(_answer(read, n, scores=scores(n)) for n in range(24))]), results
    )
    assert full.read.report is not None
    decided = {reading.anchor: reading.decides for reading in full.agreement}
    assert decided == {"person": True, "model": False}
    assert {p.anchor for p in full.read.report.preferences} == {"person", "model"}


def test_a_reviewer_s_answers_two_served_models_gave_are_read_apart_and_never_pooled() -> None:
    """A review can run again for the items left unanswered, and another model may serve it under the same reviewer
    id. Each served model's answers are one reading and one check against people; none pools the two."""
    fails = frozenset({0, 3, 5, 8, 11, 14, 17, 20})
    results, evidence = _campaign(24, judge_fails=fails)
    read = _draw(results, evidence)
    reviewer = "other/reviewer-9"

    def scores(n: int) -> dict[str, int]:
        return {"v1": int(int(read.items[n].case_id[-3:]) not in fails)}

    first = [_reference(read, n, reviewer, scores=scores(n), reviewer_served_model="other/a-1") for n in range(12)]
    # The second served model passes everything: pooled, it would hide behind the first.
    second = [_reference(read, n, reviewer, reviewer_served_model="other/b-2") for n in range(12, 24)]
    people = [_answer(read, n, scores=scores(n)) for n in range(24)]
    closed = close_blind_read(_with(read, [*first, *second, *people]), results)
    report = closed.read.report
    assert report is not None
    models = sorted(
        (entry.served_model, entry.anchor_names, entry.pairs) for entry in report.agreement if entry.anchor == "model"
    )
    assert models == [("other/a-1", [reviewer], 24), ("other/b-2", [reviewer], 24)]
    checks = {check.served_model: check for check in report.reviewer_checks}
    assert sorted(checks) == ["other/a-1", "other/b-2"]
    assert core.agreed_pairs(checks["other/a-1"].table) == 24, "the first served model agreed with people throughout"
    b_cells = core.pass_fail_counts(checks["other/b-2"].table)
    assert b_cells is not None and b_cells.reader_fail_judge_pass > 0, "the second's misses are its own"
    assert all(entry.served_model is None for entry in report.agreement if entry.anchor == "person")
    # #1231: printed, each served model heads its own block, with its own reading and its own check against people,
    # so no two lines on one criterion stand unlabelled.
    lines = judge_lines(closed, labels_name="labels.csv", labels_there=True)
    headings = [line for line in lines if "checked against a model" in line]
    assert headings == [
        f"  beside it, checked against a model ({reviewer}), served by other/a-1:",
        f"  beside it, checked against a model ({reviewer}), served by other/b-2:",
    ]
    for heading, cases in zip(headings, ("0 to 11", "12 to 23"), strict=True):
        block = lines[lines.index(heading) + 1 : lines.index(heading) + 3]
        assert block[0].startswith("    answer grounded (judged)  24 results from 12 cases: "), cases
        assert block[1].startswith(f"    {reviewer} against people on answer grounded (judged): 24 outputs: "), cases
    served = [line for line in lines if line.startswith("    answer grounded (judged)")]
    assert len(served) == len(set(served)) == 2, "the two served models' readings differ and each is printed once"


def test_a_reviewer_s_preference_is_read_once_per_served_model_and_never_pooled() -> None:
    """One reviewer id, two served models that disagree: the first prefers the candidate, the second the control. Pooled,
    they would read as one reviewer who cannot tell the arms apart. Each served model's answers are one preference,
    naming its model; a skip, which no model served, counts as unanswered in each, and the print names which is which."""
    results, evidence = _campaign(24)
    read = _draw(results, evidence)
    reviewer = "other/reviewer-9"
    first = [_reference(read, n, reviewer, prefer_arm="v2", reviewer_served_model="other/a-1") for n in range(11)]
    skip = BlindAnswer(item=read.items[11].item, rater=reviewer, made_by="model", skipped=True)
    second = [_reference(read, n, reviewer, prefer_arm="v1", reviewer_served_model="other/b-2") for n in range(12, 24)]
    closed = close_blind_read(_with(read, [*first, skip, *second]), results, closed_by="owner")
    report = closed.read.report
    assert report is not None

    read_as = sorted((p.anchor_names, p.prefer_candidate, p.prefer_control, p.missing) for p in report.preferences)
    assert read_as == [([reviewer], 0, 12, 1), ([reviewer], 11, 0, 1)], "two preferences, never one pooled"
    named = {p.served_model: (p.prefer_candidate, p.prefer_control) for p in report.preferences}
    assert named == {"other/a-1": (11, 0), "other/b-2": (0, 12)}
    lines = judge_lines(closed, labels_name="labels.csv", labels_there=True, shown={"v1": "v1", "v2": "v2"})
    said = [line for line in lines if line.startswith("preference of")]
    assert [line.split(":")[0] for line in said] == [
        f"preference of a model ({reviewer}), served by other/a-1, not a person",
        f"preference of a model ({reviewer}), served by other/b-2, not a person",
    ]
    # One served model throughout: one preference, naming it, and the print as it was.
    whole = [_reference(read, n, reviewer, prefer_arm="v2") for n in range(24)]
    once = close_blind_read(_with(read, whole), results).read.report
    assert once is not None
    (alone,) = once.preferences
    assert (alone.served_model, alone.prefer_candidate, alone.test) == (f"{reviewer}-0613", 24, "exact_sign")


# parity-with: threetears.evals.schema.completion.PricedCompletion
class FakeReviewer:
    """A reviewer that passes every output and prefers B, and keeps what it was sent; ``replies`` scripts a reply by
    the call's position."""

    model_name = "scripted/reviewer"

    def __init__(self, replies: dict[int, str] | None = None) -> None:
        self.sent: list[tuple[str, str]] = []
        self.formats: list[dict[str, Any] | None] = []
        self.replies = replies or {}

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        self.sent.append((system, user))
        self.formats.append(response_format)
        scripted = self.replies.get(len(self.sent) - 1)
        if scripted is not None:
            return _Reply(scripted)
        slots = re.findall(r"^# Output ([AB])$", user, re.M)
        verdicts = {slot: {GROUNDED.name: "pass"} for slot in slots}
        prefer = "B" if len(slots) == 2 else "none"
        return _Reply("```json\n" + json.dumps({"verdicts": verdicts, "prefer": prefer, "reason": "read it"}) + "\n```")

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        return None


@dataclass(frozen=True)
class _Reply:
    content: str
    model: str = "scripted/reviewer"
    served_model: str | None = "scripted/reviewer-0613"
    input_tokens: int | None = 10
    output_tokens: int | None = 5
    cost_usd: float | None = 0.0001
    price_source: str | None = "scripted rate"
    reasoning_tokens: int | None = None
    stop_reason: StopReason = "end_turn"
    temperature: float | None = None


# parity-with: threetears.evals.schema.out_of_run_spend.OutOfRunSpendStore
class FakeLedger:
    """An out-of-run ledger that keeps its rows."""

    def __init__(self) -> None:
        self.rows: list[OutOfRunSpend] = []

    def save_out_of_run_spend(self, spend: OutOfRunSpend, /) -> None:
        self.rows.append(spend)


def _budget(ledger: FakeLedger) -> OutOfRunBudget:
    return OutOfRunBudget(ledger, scope_id="scope", cap_usd=None, blocking_executor=None)


async def _reviewed(
    read: BlindRead, reviewer: FakeReviewer, ledger: FakeLedger, *, items: set[str] | None = None, **kwargs: Any
) -> tuple[list[BlindAnswer], BlindReview]:
    """A model's reading as the quick result makes it: planned, admitted, then made."""
    sheet, budget = blind_sheet(read), _budget(ledger)
    planned = plan_review(sheet, items)
    admitted = dict(zip(planned, budget.admit(reviewer, "reviewer", list(planned.values())), strict=True))
    return await answer_sheet(sheet, reviewer, budget, admitted, **kwargs)


async def test_a_model_answers_the_sheet_one_call_per_case_shown_only_what_a_person_is() -> None:
    results, evidence = _campaign(6)
    read = _draw(results, evidence)
    reviewer, ledger = FakeReviewer(), FakeLedger()
    answers, review = await _reviewed(read, reviewer, ledger)
    assert len(reviewer.sent) == 6 and [answer.item for answer in answers] == [item.item for item in read.items]
    assert {(a.rater, a.made_by, a.reviewer_served_model, a.prefer) for a in answers} == {
        ("scripted/reviewer", "model", "scripted/reviewer-0613", "B")
    }
    for _, user in reviewer.sent:
        for mark in (CONTROL_RUN, CANDIDATE_RUN, "res-", "case-0", "v1", "v2", JUDGE_MODEL, "JUDGE-REASONING-MARK"):
            assert mark not in user
        assert "fingerprint" not in user and user.count("Customer question number") == 1, "one case per call"
    # Each call asks for the strict reply shape, every answer a string enum.
    (shape,) = {json.dumps(sent, sort_keys=True) for sent in reviewer.formats}
    schema = json.loads(shape)["json_schema"]
    assert schema["strict"] is True and schema["schema"]["properties"]["prefer"]["enum"] == ["A", "B", "tie"]
    assert read_sample(_with(read, answers), "scripted/reviewer", "model").sample == "random"
    # Every call is on the ledger under its own purpose, and the reading states what they cost.
    assert [(row.purpose, row.model, row.outcome) for row in ledger.rows] == [
        ("reviewer", "scripted/reviewer", "completed")
    ] * 6
    assert (review.reviewer, review.calls, review.unpriced_calls, review.served_models) == (
        "scripted/reviewer",
        6,
        0,
        ["scripted/reviewer-0613"],
    )
    assert review.cost_usd == pytest.approx(0.0006) and review.unanswered == [] and review.stopped is None
    lines = judge_lines(
        close_blind_read(_with(read, answers).model_copy(update={"reviews": [review]}), results),
        labels_name=None,
        labels_there=False,
    )
    assert (
        "reviewer: a model (scripted/reviewer) read the blind read for $0.00060 over 6 call(s); the cost of checking "
        "the judge, in no candidate's cost"
    ) in lines


async def test_a_reply_that_cannot_be_used_leaves_its_case_not_answered_and_the_rest_are_asked() -> None:
    results, evidence = _campaign(6)
    read = _draw(results, evidence)
    # The third reply is prose and the fifth leaves a verdict out; the other four are usable.
    reviewer = FakeReviewer({2: "I would rather not.", 4: '{"verdicts": {}, "prefer": "A", "reason": ""}'})
    answers, review = await _reviewed(read, reviewer, FakeLedger())
    assert [answer.skipped for answer in answers] == [False, False, True, False, True, False]
    assert review.calls == 6 and review.stopped is None
    assert review.unanswered == [
        "case 3: its reply is not JSON (Expecting value)",
        "case 5: it gave None for output A on answer.grounded, whose answer is pass or fail",
    ]
    closing = close_blind_read(_with(read, answers).model_copy(update={"reviews": [review]}), results)
    closed = closing.read
    assert closed.report is not None
    (sample,) = closed.report.samples
    assert (sample.sample, sample.answered, sample.skipped) == ("random", 6, 2)
    lines = judge_lines(closing, labels_name=None, labels_there=False)
    assert lines[0].startswith("judge: 0 of 6 cases read blind by a person: judge not checked (")
    assert lines[1] == (
        "judge: 6 of 6 cases read blind by a model (scripted/reviewer), in the drawn order (a random sample of the "
        "cases answered; 2 skipped)"
    )
    assert (
        "  2 case(s) not answered: case 3: its reply is not JSON (Expecting value); case 5: it gave None for output A "
        "on answer.grounded, whose answer is pass or fail"
    ) in lines


async def test_a_cost_cap_below_the_sheet_s_cost_stops_the_model_and_the_reading_says_so() -> None:
    results, evidence = _campaign(6)
    read = _draw(results, evidence)
    reviewer, ledger = FakeReviewer(), FakeLedger()
    # The eval has landed $0.0010 against a cap of $0.00115: one call fits whole, the second crosses it, no third.
    answers, review = await _reviewed(read, reviewer, ledger, max_cost_usd=0.00115, landed_usd=0.001)
    assert (len(answers), len(reviewer.sent), len(ledger.rows), review.calls) == (2, 2, 2, 2)
    assert review.stopped is not None
    assert review.stopped.startswith("the eval's cost cap stopped it, the eval having landed $0.0012, over its ")
    lines = judge_lines(
        close_blind_read(_with(read, answers).model_copy(update={"reviews": [review]}), results),
        labels_name=None,
        labels_there=False,
    )
    assert f"  stopped before every case was asked, since {review.stopped}" in lines
    # Spend nobody priced cannot be bounded by a cap, so nothing is asked.
    none, unpriced = await _reviewed(read, FakeReviewer(), FakeLedger(), max_cost_usd=1.0, landed_usd=None)
    assert none == [] and unpriced.calls == 0
    assert unpriced.stopped == "the eval spent what nobody priced, which its cost cap cannot bound"


async def test_only_the_cases_asked_for_are_asked() -> None:
    results, evidence = _campaign(6)
    read = _draw(results, evidence)
    reviewer = FakeReviewer()
    wanted = {read.items[1].item, read.items[4].item}
    answers, review = await _reviewed(read, reviewer, FakeLedger(), items=wanted, reused=4)
    assert {answer.item for answer in answers} == wanted and (review.calls, review.reused) == (2, 4)


# --- the labels file ------------------------------------------------------------------------------------------------


def _fill(read: BlindRead, path: Path, positions: list[int], **cells: str) -> None:
    rows = list(csv.DictReader(io.StringIO(blind_read_csv(read))))
    seen: set[str] = set()
    for row in rows:
        if int(row["position"]) - 1 in positions:
            row.update({"label": "pass", **cells})
            if row["item"] not in seen and "prefer" not in cells:
                row["prefer"] = "B"
            seen.add(row["item"])
    with path.open("w", encoding="utf-8", newline="") as written:
        writer = csv.DictWriter(written, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_a_sheet_filled_in_from_the_top_reads_as_a_random_sample_and_out_of_order_as_not(tmp_path: Path) -> None:
    results, evidence = _campaign(10)
    read = _draw(results, evidence)
    labels = tmp_path / "labels.csv"
    _fill(read, labels, [0, 1, 2, 3])
    got = read_labels(labels, read, results)
    assert (len(got.answers), got.loose, got.unread) == (4, [], [])
    assert {(a.rater, a.made_by, a.collected_by, a.prefer) for a in got.answers} == {
        ("labels.csv", "person", "labels.csv", "B")
    }
    assert read_sample(_with(read, got.answers), "labels.csv").sample == "random"
    _fill(read, labels, [0, 2, 7])
    assert read_sample(_with(read, read_labels(labels, read, results).answers), "labels.csv").sample == "selected"
    # The kind is per row, a person unless it says reference (the file's word for a model), which names its model.
    _fill(read, labels, [0, 1], rater="other/reviewer-9", rater_kind="reference")
    assert {(a.rater, a.made_by, a.collected_by) for a in read_labels(labels, read, results).answers} == {
        ("other/reviewer-9", "model", None)
    }
    _fill(read, labels, [0], rater_kind="robot")
    with pytest.raises(ValueError, match="rater_kind 'robot'"):
        read_labels(labels, read, results)
    _fill(read, labels, [0], rater_kind="reference")
    with pytest.raises(ValueError, match="names no model"):
        read_labels(labels, read, results)
    _fill(read, labels, [0], label="maybe")
    with pytest.raises(ValueError, match="its scale is pass or fail"):
        read_labels(labels, read, results)
    _fill(read, labels, [0], prefer="C")
    with pytest.raises(ValueError, match="a preference is A, B or tie"):
        read_labels(labels, read, results)
    _fill(read, labels, [0, 1], label="skip", prefer="")
    skipped = read_labels(labels, read, results).answers
    assert [(a.skipped, a.verdicts, a.prefer) for a in skipped] == [(True, [], None)] * 2


def test_the_files_a_rung_2_read_wrote_before_answers_said_who_made_them_still_read(tmp_path: Path) -> None:
    """The two files the rung-2 blind read stores on disk, as that build wrote them (it stored no read document), each
    read into answers that carry ``made_by``. Their word for a model stays ``reference``: it is the file's word, which a
    labeller types and an older kept file holds, and it reads as a model."""
    results, evidence = _campaign(6)
    read = _draw(results, evidence)
    # The labels a person saved from the page or filled into blind_read.csv: a blank rater_kind is a person.
    labels = tmp_path / "labels.csv"
    _fill(read, labels, [0, 1, 2])
    header = labels.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("position,item,slot,dimension,") and ",rater,rater_kind,prefer," in header
    people = read_labels(labels, read, results).answers
    assert {(a.rater, a.made_by, a.collected_by) for a in people} == {("labels.csv", "person", "labels.csv")}
    # A model's answers the quick path kept under blind_reads/, in that build's columns, reference and all.
    kept = tmp_path / "kept.csv"
    columns = (
        "item,slot,dimension,label,reason,rater,rater_kind,served_model,prefer,output_fingerprint,criterion_fingerprint"
    )
    lines = [columns]
    for item in read.items[:4]:
        for number, output in enumerate(item.outputs):
            fingerprints = f"{output.output_fingerprint},{output.criterion_fingerprints[GROUNDED.name]}"
            prefer = "B" if number == 0 and len(item.outputs) == 2 else ""
            lines.append(
                f"{item.item},{output.slot},{GROUNDED.name},pass,,scripted/reviewer,reference,scripted/reviewer-0613,"
                f"{prefer},{fingerprints}"
            )
    kept.write_text("\n".join(lines) + "\n", encoding="utf-8")
    model = read_labels(kept, read, results).answers
    assert {(a.rater, a.made_by, a.reviewer_served_model, a.collected_by) for a in model} == {
        ("scripted/reviewer", "model", "scripted/reviewer-0613", None)
    }
    both = _with(read, [*people, *model])
    assert read_sample(both, "scripted/reviewer", "model").sample == "random"
    assert read_sample(both, "labels.csv").sample == "random"
    closed = close_blind_read(both, results)
    assert closed.read.report is not None
    assert {(s.rater, s.made_by) for s in closed.read.report.samples} == {
        ("labels.csv", "person"),
        ("scripted/reviewer", "model"),
    }
    assert {reading.anchor for reading in closed.read.report.agreement} == {"person", "model"}
    # The print names a model as a model wherever it names who answered, never by its bare id as a person is named.
    lines = judge_lines(closed, labels_name="labels.csv", labels_there=True, shown={})
    assert any("by a model (scripted/reviewer)" in line for line in lines), lines
    bare = [line for line in lines for lead in ("by ", "of ", "against ") if f"{lead}scripted/reviewer" in line]
    assert bare == [], bare


def test_an_older_to_label_sheet_and_another_draw_s_labels_still_read_as_not_drawn_at_random(tmp_path: Path) -> None:
    results, evidence = _campaign(24, arms=1)
    read = _draw(results, evidence, arms=1)
    old = tmp_path / "labels.csv"
    columns = ["case", "repeat", "dimension", "case_material", "answer", "label", "reason", "rater", "run_id",
               "result_id", "output_fingerprint", "criterion_fingerprint"]  # fmt: skip
    with old.open("w", encoding="utf-8", newline="") as written:
        writer = csv.DictWriter(written, fieldnames=columns)
        writer.writeheader()
        for result in results:
            score = result.rubric_scores[0]
            writer.writerow(
                {
                    "case": result.test_case_id,
                    "repeat": "1",
                    "dimension": score.dim,
                    "case_material": "m",
                    "answer": "a",
                    "label": "pass",
                    "reason": "",
                    "rater": "",
                    # an earlier run's ids, which this run's results do not carry
                    "run_id": "an-older-run",
                    "result_id": f"older-{result.id}",
                    "output_fingerprint": score.output_fingerprint,
                    "criterion_fingerprint": score.criterion_fingerprint,
                }
            )
    got = read_labels(old, read, results)
    assert (got.answers, len(got.loose), got.unread) == ([], 24, [])
    closing = close_blind_read(read, results, other_ratings=got.loose)
    closed = closing.read
    assert closed.report is not None and closed.report.samples == [] and closed.report.other_ratings == 24
    (reading,) = closed.report.agreement
    assert (reading.pairs, reading.blind_pairs, reading.random_pairs) == (24, 0, 0), "paired, and none of it random"
    lines = judge_lines(closing, labels_name="labels.csv", labels_there=True, loose=24)
    assert lines[0] == (
        # Whose they are, as a sample line says it: a file with no rater column is a person's, unnamed.
        "judge: 24 label(s) in labels.csv by a person are not on this run's draw (the answers changed since, or it is "
        "an older sheet): kept, each paired with the same output, and not drawn at random"
    )
    # A label for another draw that carries no fingerprint cannot be paired, and is said.
    other = tmp_path / "other.csv"
    other.write_text("item,slot,dimension,label\nnot-this-draw,A,answer.grounded,pass\n", encoding="utf-8")
    assert read_labels(other, read, results).unread == [
        "line 2: it is not on this run's draw and carries no fingerprint to pair it by"
    ]
    missing = tmp_path / "missing.csv"
    missing.write_text("item,slot\nx,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks the column"):
        read_labels(missing, read, results)


def _page_labels(read: BlindRead, positions: list[int], path: Path, *, prefer_arm: str = "v2") -> None:
    """Labels as the page saves them for the cases at ``positions``: every output passed, with its fingerprints."""
    rows = ["item,slot,dimension,label,reason,rater,rater_kind,prefer,output_fingerprint,criterion_fingerprint"]
    for position in positions:
        item = read.items[position]
        prefer = next((o.slot for o in item.outputs if o.arm == prefer_arm), "") if len(item.outputs) == 2 else ""
        for number, output in enumerate(item.outputs):
            rows.append(
                f"{item.item},{output.slot},{GROUNDED.name},pass,,owner,person,{prefer if number == 0 else ''},"
                f"{output.output_fingerprint},{output.criterion_fingerprints[GROUNDED.name]}"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_labels_saved_from_the_page_survive_a_change_to_the_answers_as_not_drawn_at_random(tmp_path: Path) -> None:
    results, evidence = _campaign(10)
    read = _draw(results, evidence)
    labels = tmp_path / "labels.csv"
    _page_labels(read, list(range(6)), labels)
    assert len(read_labels(labels, read, results).answers) == 6, "on its own draw the file is six answers"
    # The candidate's answer to one labelled case changes, and two cases are added: another population, another draw.
    changed_case = read.items[2].case_id
    old = next(r for r in results if r.eval_run_id == CANDIDATE_RUN and r.test_case_id == changed_case)
    new, new_read = _result(CANDIDATE_RUN, int(changed_case[-3:]), "a rewritten reply", passes=True)
    later = [new if result is old else result for result in results]
    later_evidence = {**{k: v for k, v in evidence.items() if k != old.id}, new.id: new_read}
    for case in (10, 11):
        for run, text in ((CONTROL_RUN, "old"), (CANDIDATE_RUN, "new")):
            added, added_read = _result(run, case, f"{text} reply to {case}", passes=True)
            later.append(added)
            later_evidence[added.id] = added_read
    redrawn = _draw(later, later_evidence)
    assert {item.item for item in redrawn.items}.isdisjoint({item.item for item in read.items})
    got = read_labels(labels, redrawn, later)
    # Eleven of the twelve labelled outputs are still among the answers and stay read; the rewritten one is not.
    assert (got.answers, len(got.loose)) == ([], 11)
    assert {rating.rater_kind for rating in got.loose} == {"person"}
    gone = [entry for entry in got.unread if "no longer among this run's answers" in entry]
    moved = [entry for entry in got.unread if "a preference given on another draw" in entry]
    assert (len(gone), len(moved), len(got.unread)) == (1, 6, 7)
    closing = close_blind_read(redrawn, later, other_ratings=got.loose, unread=got.unread)
    closed = closing.read
    assert closed.report is not None and closed.report.agreement[0].random_pairs == 0
    lines = judge_lines(closing, labels_name="labels.csv", labels_there=True, loose=11, unread=got.unread)
    assert lines[0].startswith("judge: 11 label(s) in labels.csv by owner are not on this run's draw")
    assert lines[1].startswith("  7 label(s) in labels.csv not read: line 2: a preference given on another draw")
    assert lines[1].endswith("; and 4 more") and len([line for line in lines if "not read" in line]) == 1


def test_a_case_answered_in_part_is_counted_and_a_missing_preference_is_said(tmp_path: Path) -> None:
    results, evidence = _campaign(8)
    read = _draw(results, evidence)
    labels = tmp_path / "labels.csv"
    _fill(read, labels, list(range(8)))
    rows = list(csv.DictReader(labels.open(encoding="utf-8", newline="")))
    for row in rows:
        # Case 3: only output A is given a verdict. Case 6: both verdicts, and the preference left blank.
        if int(row["position"]) == 3 and row["slot"] == "B":
            row["label"] = ""
        if int(row["position"]) == 6:
            row["prefer"] = ""
    with labels.open("w", encoding="utf-8", newline="") as written:
        writer = csv.DictWriter(written, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    closing = close_blind_read(_with(read, read_labels(labels, read, results).answers), results)
    closed = closing.read
    assert closed.report is not None
    (sample,) = closed.report.samples
    assert (sample.sample, sample.answered, sample.skipped, sample.partial) == ("random", 8, 0, 2)
    (preference,) = closed.report.preferences
    assert (preference.items, preference.missing) == (7, 1)
    lines = judge_lines(closing, labels_name="labels.csv", labels_there=True)
    assert lines[0] == (
        "judge: 8 of 8 cases read blind by a person, in the drawn order (a random sample of the cases answered; 2 "
        "answered in part)"
    )
    assert any(line.startswith("preference: ") and ", 1 with no preference given, " in line for line in lines)
    # A whole answer is not a partial one, and a skip is its own count.
    assert read_sample(_with(read, [_answer(read, 0)]), "owner").partial == 0


def test_with_no_labels_the_line_says_the_judge_is_not_checked_and_the_one_next_step() -> None:
    """No tier word and no count of answers still needed (#1208): the line says the judge is not checked, and how."""
    results, evidence = _campaign(4, arms=1)
    closed = close_blind_read(_draw(results, evidence, arms=1), results)
    assert judge_lines(closed, labels_name="labels.csv", labels_there=False) == [
        "judge: 0 of 4 cases read blind: judge not checked (labels.csv is not there yet); next step: open the "
        "blind_read.html the report writes, answer its cases from the top, and save its labels as labels.csv"
    ]
    assert judge_lines(None, labels_name=None, labels_there=False) == [
        "judge: no judged answer to read blind: judge not checked"
    ]


# --- the boundary ---------------------------------------------------------------------------------------------------


def test_the_pure_core_imports_no_page_form_or_quick_path_module() -> None:
    tree = ast.parse(Path(core.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    engine = {name for name in imported if name.startswith("threetears.")}
    assert engine == {
        "threetears.evals.analysis.agreement",
        "threetears.evals.analysis.stats",
        "threetears.evals.kernel.evidence_tiers",
        "threetears.evals.schema.blind_read",
        "threetears.evals.schema.hashing",
        "threetears.evals.schema.models",
        "threetears.evals.schema.reading_states",
    }
    assert files.blind_sheet is core.blind_sheet, "the quick path calls the core's sheet, and builds none of its own"
