"""The blind read: one seeded random order over a campaign's judged outputs, read with no arm, run or judge score.

Held here on small made-up campaigns, through the pure core (``analysis/blind_read.py``) and the quick path's files
(``quick/blind.py``):

- the draw depends on the answers and the seed, never on the order results are listed in or on their ids;
- the sheet, as the page's bytes and the CSV's, names no arm, run, result or case and carries no judge score;
- answers from the top are a ``random`` sample and any other set is ``selected``;
- closing gives person ratings that name the read, agreement per criterion with its tier, and a preference with its
  exact test, withheld until the read reaches the length fixed when it was drawn;
- a reference reviewer (a model) is read apart from people, never as one, and never when it is the judge's own model;
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
from typing import Any, Self

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import blind_read as core
from threetears.evals.analysis.blind_read import (
    blind_sheet,
    close_blind_read,
    draw_blind_read,
    exact_sign_test_p,
    sample_standing,
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
from threetears.evals.schema import (
    BlindAnswer,
    BlindArm,
    BlindPreference,
    BlindRead,
    BlindVerdict,
    EvalResult,
    JudgeEvidence,
    RubricDim,
    RubricScore,
    StopReason,
    label_key_of,
)
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
    assert {key for item in sheet["items"] for output in item["outputs"] for key in output} == {"slot", "text"}


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
    # Storage is taken away first, as a locked-down browser does; then the first three items are answered through
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
    answers = read_labels(labels, read, results).answers
    assert [answer.item for answer in answers] == [item.item for item in read.items[:3]]
    assert {answer.rater for answer in answers} == {"B. Reader"}
    assert all(len(answer.verdicts) == 2 and answer.prefer == "A" for answer in answers)
    assert sample_standing(_with(read, answers), "B. Reader").standing == "random"


# --- the standing ---------------------------------------------------------------------------------------------------


def test_answers_from_the_top_are_a_random_sample_and_any_other_set_is_selected() -> None:
    results, evidence = _campaign(10)
    read = _draw(results, evidence)
    top = _with(read, [_answer(read, n) for n in range(4)])
    standing = sample_standing(top, "owner")
    assert (standing.standing, standing.answered, standing.items, standing.skipped) == ("random", 4, 10, 0)
    assert sample_standing(_with(read, [_answer(read, n) for n in (0, 1, 3)]), "owner").standing == "selected"
    assert sample_standing(_with(read, [_answer(read, 5)]), "owner").standing == "selected"
    assert sample_standing(read, "owner").standing == "none"
    # A skip is an answer: the prefix stays one, and the standing says how many were skipped.
    skipped = _with(
        read, [_answer(read, 0), BlindAnswer(item=read.items[1].item, rater="owner", skipped=True), _answer(read, 2)]
    )
    standing = sample_standing(skipped, "owner")
    assert (standing.standing, standing.answered, standing.skipped) == ("random", 3, 1)
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
    (standing,) = report.standings
    assert (standing.standing, standing.answered) == ("random", 24)
    (reading,) = report.agreement
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
    assert table is not None
    assert (table.both_pass, table.both_fail, table.rater_fail_judge_pass, table.rater_pass_judge_fail) == (40, 8, 0, 0)
    assert table.detection == 1.0 and table.detection_interval is not None
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
    report = close_blind_read(_with(read, answers), results).read.report
    assert report is not None
    (reading,) = report.agreement
    assert reading.exact_agreement == pytest.approx(27 / 30) and reading.kappa == pytest.approx(0.0)
    assert reading.tier != "calibrated"
    table = reading.table
    assert table is not None and (table.rater_fail_judge_pass, table.both_fail, table.detection) == (3, 0, 0.0)
    lines = judge_lines(
        close_blind_read(_with(read, answers), results).read, labels_name="labels.csv", labels_there=True
    )
    assert lines[0] == "judge: 30 of 30 items read blind, in the drawn order (a random sample)"
    assert "30 results from 30 cases" in lines[1] and lines[1].endswith(
        "the judge failed 0 of the 3 answers you failed"
    )


def test_the_preference_is_withheld_until_the_planned_length_and_reads_only_those_items() -> None:
    results, evidence = _campaign(40)
    read = _draw(results, evidence, planned_items=12)
    assert read.draw.planned_items == 12
    partial = close_blind_read(_with(read, [_answer(read, n) for n in range(11)]), results).read.report
    assert partial is not None
    (withheld,) = partial.preferences
    assert (withheld.verdict, withheld.planned, withheld.answered) == ("withheld", 12, 11)
    assert (withheld.items, withheld.prefer_candidate, withheld.p, withheld.slot_a_share) == (0, 0, None, None)
    lines = judge_lines(
        close_blind_read(_with(read, [_answer(read, n) for n in range(11)]), results).read,
        labels_name="labels.csv",
        labels_there=True,
        shown={"v1": "v1", "v2": "v2"},
    )
    assert "preference: shown when the first 12 items are answered (11 of 12)" in lines
    assert lines[-1] == "next: keep going from item 12 in blind_read.html"
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
    assert hunted is not None and hunted.preferences[0].verdict == "withheld"
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
    assert report is not None and (report.preferences[0].verdict, report.preferences[0].p) == ("too_few", None)


def test_ties_are_counted_and_never_split_and_skips_are_said() -> None:
    results, evidence = _campaign(12)
    read = _draw(results, evidence)
    answers = [_answer(read, n, prefer_arm=None if n in (0, 1) else "v2") for n in range(11)]
    answers.append(BlindAnswer(item=read.items[11].item, rater="owner", skipped=True))
    closed = close_blind_read(_with(read, answers), results).read
    assert closed.report is not None
    (preference,) = closed.report.preferences
    assert (preference.prefer_candidate, preference.prefer_control, preference.ties, preference.skipped) == (9, 0, 2, 1)
    assert preference.p == pytest.approx(exact_sign_test_p(9, 9))
    lines = judge_lines(closed, labels_name="labels.csv", labels_there=True, shown={})
    assert lines[0] == (
        "judge: 12 of 12 items read blind, in the drawn order (a random sample of the cases answered; 1 skipped)"
    )
    assert "preference: v2 over v1 on 9 of 9 (2 ties), 1 skipped, sign test p 0.004: v2 preferred" in lines


def test_several_people_s_shares_are_read_by_the_bounded_test() -> None:
    results, evidence = _campaign(12)
    read = _draw(results, evidence)
    answers = [_answer(read, n, rater=rater) for n in range(12) for rater in ("ann", "bo")]
    report = close_blind_read(_with(read, answers), results).read.report
    assert report is not None
    (preference,) = report.preferences
    assert preference.test == "bounded_share" and preference.anchor_names == ["ann", "bo"]
    assert preference.prefer_candidate == 12 and preference.p is not None


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


# --- a reference reviewer -------------------------------------------------------------------------------------------


def _reference(read: BlindRead, position: int, model: str, **kwargs: Any) -> BlindAnswer:
    return _answer(read, position, rater=model, rater_kind="reference", **kwargs)


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
    assert {(r.rater, r.rater_kind, r.source) for r in closed.ratings} == {(f"reference:{reviewer}", "agent", None)}
    report = closed.read.report
    assert report is not None
    (reading,) = report.agreement
    assert (reading.anchor, reading.anchor_names, reading.decides, reading.tier) == (
        "reference",
        [reviewer],
        True,
        "calibrated",
    )
    assert not reading.shares_family and report.anchor_checks == [] and report.refused_anchors == []
    (preference,) = report.preferences
    assert (preference.anchor, preference.anchor_names) == ("reference", [reviewer])
    lines = judge_lines(closed.read, labels_name=None, labels_there=False, shown={})
    assert lines[0] == (
        f"judge: 24 of 24 items read blind by {reviewer}, a reference reviewer, in the drawn order (a random sample)"
    )
    assert lines[1] == f"  checked against {reviewer}, not yet against people:"
    assert f"calibrated against a reference model ({reviewer})" in lines[2]
    assert lines[3].startswith(f"preference of {reviewer} (a reference reviewer, not a person): v2 over v1 on 24 of 24")
    assert not any(line.startswith("next:") for line in lines)


def test_a_reference_reviewer_that_is_the_judge_s_own_model_is_refused_and_one_of_its_family_is_said() -> None:
    results, evidence = _campaign(24)
    read = _draw(results, evidence)
    own = close_blind_read(_with(read, [_reference(read, n, JUDGE_MODEL) for n in range(24)]), results).read
    assert own.report is not None and own.report.agreement == []
    (refused,) = own.report.refused_anchors
    assert (refused.reviewer, refused.dim) == (JUDGE_MODEL, GROUNDED.name) and "repeat the judge" in refused.reason
    assert (
        f"  {JUDGE_MODEL} is the judge's own model: its answers repeat the judge and are not read as calibration"
        in judge_lines(own, labels_name=None, labels_there=False)
    )
    # Named as another model, served by the judge's: refused all the same.
    served = [_reference(read, n, "alias/reviewer", reviewer_served_model=JUDGE_MODEL) for n in range(24)]
    aliased = close_blind_read(_with(read, served), results).read.report
    assert aliased is not None and aliased.agreement == [] and len(aliased.refused_anchors) == 1
    family = close_blind_read(_with(read, [_reference(read, n, "acme/reviewer-2") for n in range(24)]), results).read
    assert family.report is not None and family.report.agreement[0].shares_family
    assert any("one provider's models" in line for line in judge_lines(family, labels_name=None, labels_there=False))


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
    by_anchor = {reading.anchor: reading for reading in few.read.report.agreement}
    assert (by_anchor["person"].calibration, by_anchor["person"].decides) == ("insufficient", False)
    assert (by_anchor["reference"].tier, by_anchor["reference"].decides) == ("calibrated", True)
    # The two are never pooled: each reading counts its own rater's pairs alone.
    assert (by_anchor["person"].pairs, by_anchor["reference"].pairs) == (6, 48)
    (check,) = few.read.report.anchor_checks
    assert (check.reviewer, check.dim, check.outputs, check.exact_agreement) == (reviewer, GROUNDED.name, 6, 1.0)
    lines = judge_lines(few.read, labels_name="labels.csv", labels_there=True)
    assert f"  beside it, checked against {reviewer}:" in lines
    assert any(line.startswith(f"    {reviewer} against people on answer.grounded: 6 outputs") for line in lines)
    # A whole person read decides it, whatever the reference says.
    full = close_blind_read(
        _with(read, [*reference, *(_answer(read, n, scores=scores(n)) for n in range(24))]), results
    )
    assert full.read.report is not None
    decided = {reading.anchor: reading.decides for reading in full.read.report.agreement}
    assert decided == {"person": True, "reference": False}
    assert {p.anchor for p in full.read.report.preferences} == {"person", "reference"}


# parity-with: threetears.evals.schema.CompletionClient
class FakeReviewer:
    """A reviewer that passes every output and prefers B, and keeps what it was sent."""

    model_name = "scripted/reviewer"

    def __init__(self, reply: str | None = None) -> None:
        self.sent: list[tuple[str, str]] = []
        self.reply = reply

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        self.sent.append((system, user))
        if self.reply is not None:
            return _Reply(self.reply)
        slots = re.findall(r"^# Output ([AB])$", user, re.M)
        verdicts = {slot: {GROUNDED.name: "pass"} for slot in slots}
        prefer = "B" if len(slots) == 2 else None
        return _Reply("```json\n" + json.dumps({"verdicts": verdicts, "prefer": prefer, "reason": "read it"}) + "\n```")

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


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


async def test_a_model_answers_the_sheet_one_call_per_item_shown_only_what_a_person_is() -> None:
    results, evidence = _campaign(6)
    read = _draw(results, evidence)
    reviewer = FakeReviewer()
    answers = await answer_sheet(blind_sheet(read), reviewer, model="scripted/reviewer")
    assert len(reviewer.sent) == 6 and [answer.item for answer in answers] == [item.item for item in read.items]
    assert {(a.rater, a.rater_kind, a.reviewer_served_model, a.prefer) for a in answers} == {
        ("scripted/reviewer", "reference", "scripted/reviewer-0613", "B")
    }
    for _, user in reviewer.sent:
        for mark in (CONTROL_RUN, CANDIDATE_RUN, "res-", "case-0", "v1", "v2", JUDGE_MODEL, "JUDGE-REASONING-MARK"):
            assert mark not in user
        assert user.count("Customer question number") == 1, "one item per call, nothing of any other"
    assert sample_standing(_with(read, answers), "scripted/reviewer").standing == "random"
    with pytest.raises(ValueError, match="item 1 is not JSON"):
        await answer_sheet(blind_sheet(read), FakeReviewer("I would rather not."), model="scripted/reviewer")
    with pytest.raises(ValueError, match="gives None for output A"):
        await answer_sheet(blind_sheet(read), FakeReviewer('{"verdicts": {}}'), model="scripted/reviewer")


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


def test_a_sheet_filled_in_from_the_top_reads_as_a_random_sample_and_out_of_order_as_selected(tmp_path: Path) -> None:
    results, evidence = _campaign(10)
    read = _draw(results, evidence)
    labels = tmp_path / "labels.csv"
    _fill(read, labels, [0, 1, 2, 3])
    got = read_labels(labels, read, results)
    assert (len(got.answers), got.loose, got.unread) == (4, [], [])
    assert {(a.rater, a.rater_kind, a.collected_by, a.prefer) for a in got.answers} == {
        ("labels.csv", "person", "labels.csv", "B")
    }
    assert sample_standing(_with(read, got.answers), "labels.csv").standing == "random"
    _fill(read, labels, [0, 2, 7])
    assert sample_standing(_with(read, read_labels(labels, read, results).answers), "labels.csv").standing == "selected"
    # The kind is per row, a person unless it says reference, and a reference names its model.
    _fill(read, labels, [0, 1], rater="other/reviewer-9", rater_kind="reference")
    assert {(a.rater, a.rater_kind, a.collected_by) for a in read_labels(labels, read, results).answers} == {
        ("other/reviewer-9", "reference", None)
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


def test_an_older_to_label_sheet_and_another_draw_s_labels_still_read_as_selected(tmp_path: Path) -> None:
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
    closed = close_blind_read(read, results, other_ratings=got.loose).read
    assert closed.report is not None and closed.report.standings == [] and closed.report.other_ratings == 24
    (reading,) = closed.report.agreement
    assert (reading.pairs, reading.blind_pairs, reading.random_pairs) == (24, 0, 0), "paired, and none of it random"
    lines = judge_lines(closed, labels_name="labels.csv", labels_there=True, loose=24)
    assert lines[0] == (
        "judge: 24 label(s) in labels.csv are not on this run's draw (an older sheet, or another run's): counted "
        "toward agreement as selected, not a random sample"
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


def test_with_no_labels_the_line_says_the_judge_is_uncalibrated_and_the_one_next_step() -> None:
    results, evidence = _campaign(4, arms=1)
    read = close_blind_read(_draw(results, evidence, arms=1), results).read
    assert judge_lines(read, labels_name="labels.csv", labels_there=False) == [
        "judge: 0 of 4 items read blind: judge uncalibrated (labels.csv is not there yet); next step: open the "
        "blind_read.html the report writes, answer its items from the top, and save its labels as labels.csv; the "
        "calibrated tier's floor is 20 judged answers per dimension and the read holds 4, so it needs 16 more"
    ]
    assert judge_lines(None, labels_name=None, labels_there=False) == [
        "judge: no judged answer to read blind: judge uncalibrated"
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
    }
    assert files.blind_sheet is core.blind_sheet, "the quick path calls the core's sheet, and builds none of its own"
