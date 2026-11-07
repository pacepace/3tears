"""The writer calibration set, in its dry mode: the Opus editor's reviews as judge cases, and a judge arm that reads them.

``fixtures/writer_cases/writer_calibration.py`` is run live only with an owner's approval (reset P2, chunk B). Its dry
mode builds everything the live run judges, on a scripted judge that calls no model. These tests hold that:

- every page comes from a replay through the writer kind at revisions 0, 1 or 2, paired with its own round's review;
- each page gives seven cases on the gate's own criteria, its labels the Opus scores;
- each pair of rounds gives two pair cases, the second the mirror of the first;
- both case sets are frozen, a judge arm is read per criterion, and no judge profile is recorded;
- the summary a live run writes carries every row, arm and left-out round, and its cap is split by planned calls.

Prototype 4's two second rounds no longer replay: their recorded round-2 reviews carry no ``continues`` or
``resolved``, which the editor's reader now asks for, so the replay asks a turn the recording lacks.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from math import comb

import pytest

from packages.evals.tests.fixtures.writer_cases.writer_calibration import (
    ABSOLUTE_SET,
    DRY_JUDGE,
    PAIR_SET,
    REPORT_DOCS,
    Calibration,
    absolute_cases,
    calibration,
    pair_cases,
    arm_caps,
    pairs,
    preferred,
)
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.off_loop import off_loop
from threetears.evals.kernel import judge_case_of, judge_criterion_digest
from threetears.evals.ops.report_writer.preference import pair_evidence, preference_rubric
from threetears.evals.ops.report_writer.writer_contract import (
    EDITOR_ROUNDS_MEASURE,
    MUST_FIX_OPEN_MEASURE,
    SHOULD_FIX_OPEN_MEASURE,
    writer_scoring_rubric,
)

#: The rounds that no longer replay, each with the session its replay ran out in.
_UNREPLAYED = {
    ("prototype4/r2-with-its-gates-read-in-each-task", 2): "editor-2",
    ("prototype4/r3-without-its-configs", 2): "editor-2",
}


@pytest.fixture(scope="module")
def dry() -> Calibration:
    return asyncio.run(calibration(dry=True, k=1, cap_usd=0.0, judges=()))


def test_every_page_is_a_replay_at_revisions_0_1_or_2_paired_with_its_own_rounds_review(dry: Calibration) -> None:
    assert 28 <= len(dry.found) <= 30
    assert {(out.run, out.round) for out in dry.left} == set(_UNREPLAYED)
    for out in dry.left:
        assert f"session {_UNREPLAYED[(out.run, out.round)]!r}" in out.why, out.why
    rounds = Counter(page.round for page in dry.found)
    assert set(rounds) == {1, 2, 3}, "revisions 0, 1 and 2"
    for page in dry.found:
        generation = json.loads((REPORT_DOCS / page.run / "generation.json").read_text())
        (review,) = [review for review in generation["reviews"] if review["round"] == page.round]
        assert page.review == review
        assert (page.editor, page.model) == (generation["skill"]["editor"], generation["model"])
        # The replay ended on that round, holding exactly that review's findings open.
        assert page.measures[EDITOR_ROUNDS_MEASURE] == page.round
        severities = Counter(finding["severity"] for finding in review["findings"])
        assert page.measures[MUST_FIX_OPEN_MEASURE] == severities["must_fix"]
        assert page.measures[SHOULD_FIX_OPEN_MEASURE] == severities["should_fix"]
    # Every round of one recording is its own page.
    by_run: dict[str, set[str]] = {}
    for page in dry.found:
        by_run.setdefault(page.run, set()).add(page.evidence.artifact)
    assert all(len(texts) == len([p for p in dry.found if p.run == run]) for run, texts in by_run.items())
    assert sum(1 for page in dry.found if page.run.startswith("prototype5/")) == 21


def test_each_page_gives_seven_cases_on_the_gates_criteria_labelled_by_the_opus_scores(dry: Calibration) -> None:
    rubric = writer_scoring_rubric()
    cases = absolute_cases(dry.found)
    assert len(cases) == 7 * len(dry.found)
    for index, case in enumerate(cases):
        page, item = dry.found[index // 7], index % 7
        scored = sorted(page.review["items"], key=lambda each: each["n"])[item]
        assert case.criterion == rubric[item]
        assert case.expected == {2: 5, 1: 3, 0: 1}[scored["score"]]
        assert case.note == scored["reason"]
        assert case.author == f"agent:{page.model} report editor @{page.editor}"
        # The page and its facts as the gate's judge reads them, but for the trailing whitespace a case trims.
        assert (case.output, case.case_material) == (
            page.evidence.artifact.strip(),
            page.evidence.case_material.strip(),
        )
    # Frozen, each case carries the gate's criterion digest.
    gate = {dim.name: judge_criterion_digest(dim.name, dim.scale, dim) for dim in rubric}
    frozen = dry.frozen[ABSOLUTE_SET]
    assert len(frozen.cases) == len(cases) and not frozen.skipped
    per_dim = Counter(case.dim for case in frozen.cases)
    assert set(per_dim) == set(gate) and set(per_dim.values()) == {len(dry.found)}
    for case in frozen.cases:
        stored = off_loop(dry.host.storage.load_test_case, case.test_case_id, TOYHOST_SCOPE)
        assert stored is not None
        judge_case = judge_case_of(stored)
        assert judge_case is not None and judge_case.criterion_digest() == gate[case.dim]


def test_each_pair_of_rounds_gives_two_cases_the_swapped_one_mirrored(dry: Calibration) -> None:
    rounds = Counter(page.run for page in dry.found)
    ordered = pairs(dry.found)
    cases = pair_cases(dry.found)
    assert len(cases) == len(ordered) == 2 * sum(comb(n, 2) for n in rounds.values()) == 42
    criterion = preference_rubric()
    for index in range(0, len(cases), 2):
        first, swapped = ordered[index], ordered[index + 1]
        assert (swapped.a, swapped.b) == (first.b, first.a)
        assert first.a.round < first.b.round
        assert cases[index].expected == preferred(first.b.total - first.a.total)
        assert cases[index].expected + cases[index + 1].expected == 6
        assert cases[index + 1].output == pair_evidence(first.b.evidence, first.a.evidence).artifact.strip()
        assert cases[index].criterion == criterion
    # Kairo's third round scored 12 against its second's 14: the earlier page is the one preferred.
    kairo = next(p for p in ordered if p.a.run == "prototype5/kairo" and (p.a.round, p.b.round) == (2, 3))
    assert (kairo.a.total, kairo.b.total, kairo.expected) == (14, 12, 1)
    assert [preferred(delta) for delta in (-5, -2, -1, 0, 1, 2, 5)] == [1, 1, 2, 3, 4, 5, 5]
    frozen = dry.frozen[PAIR_SET]
    assert len(frozen.cases) == len(cases) and {case.dim for case in frozen.cases} == {criterion.name}
    # Frozen, each pair case carries the digest of the criterion the pass asks, as an absolute case carries its own.
    gate = judge_criterion_digest(criterion.name, criterion.scale, criterion)
    for case in frozen.cases:
        stored = off_loop(dry.host.storage.load_test_case, case.test_case_id, TOYHOST_SCOPE)
        assert stored is not None
        judge_case = judge_case_of(stored)
        assert judge_case is not None and judge_case.criterion_digest() == gate


def test_both_case_sets_are_frozen_and_listed_by_name(dry: Calibration) -> None:
    for name in (ABSOLUTE_SET, PAIR_SET):
        report = dry.frozen[name]
        assert report.case_set is not None and report.case_set.name == name
        case_set = off_loop(dry.host.storage.load_case_set, name, report.case_set.version, TOYHOST_SCOPE)
        assert case_set is not None
        assert list(case_set.test_case_ids) == [case.test_case_id for case in report.cases]


def test_the_dry_judge_arm_is_read_per_criterion_and_no_profile_is_recorded(dry: Calibration) -> None:
    by_dim = {reading.rubric_dim: reading for reading in dry.readings.readings}
    assert set(by_dim) == {dim.name for dim in writer_scoring_rubric()} | {preference_rubric().name}
    assert not dry.readings.unread
    for dim, reading in by_dim.items():
        assert reading.cases == (42 if dim == preference_rubric().name else len(dry.found))
        assert reading.parse_validity.rate == 1.0 and reading.judge_model == DRY_JUDGE
        assert reading.label_agreement is not None and reading.label_agreement.n == reading.cases
    assert off_loop(dry.host.storage.query_judge_profiles, TOYHOST_SCOPE) == []


def test_the_summary_the_live_run_writes_carries_every_arm_row_and_left_out_round(dry: Calibration) -> None:
    summary = json.loads(json.dumps(dry.summary(dry=True)))
    assert summary["case_sets"] == {
        ABSOLUTE_SET: {"version": 1, "cases": 7 * len(dry.found)},
        PAIR_SET: {"version": 1, "cases": 42},
    }
    assert [(arm["case_set"], arm["cases"], arm["trials"], arm["stopped"]) for arm in summary["arms"]] == [
        (ABSOLUTE_SET, 7 * len(dry.found), 7 * len(dry.found), None),
        (PAIR_SET, 42, 42, None),
    ]
    rows = summary["agreement_with_the_opus_editor"]
    assert len(rows) == 8 and {row["judge"] for row in rows} == {DRY_JUDGE}
    assert len(summary["pages"]) == len(dry.found) and len(summary["left_out"]) == len(_UNREPLAYED)
    lines = summary["readout"].splitlines()
    assert lines[0].startswith("Agreement with the Opus editor")
    # One table row per reading and per arm, each as wide as its header.
    agreement = [line for line in lines if line.startswith(f"| {DRY_JUDGE} | report_writer.")]
    assert len(agreement) == 8 and all(line.count("|") == 9 for line in agreement)
    assert sum(1 for line in lines if line.startswith(f"| {DRY_JUDGE} | writer-opus-")) == 2
    assert all(f"| {run} round {round_} |" in summary["readout"] for run, round_ in _UNREPLAYED)
    # The anchor a later reread compares with: the pair arm's trials alone, the absolute set's left out, one row per
    # pair case.
    anchor = summary["anchor"]
    assert len(anchor) == 42 and {row["judge"] for row in anchor} == {DRY_JUDGE}
    assert [(row["run"], row["first"], row["second"]) for row in anchor] == [
        (pair.a.run, pair.a.round, pair.b.round) for pair in pairs(dry.found)
    ]
    assert all(row["score"] in range(1, 6) and row["served_model"] == DRY_JUDGE for row in anchor)


def test_a_live_runs_cap_is_split_by_the_calls_each_arm_plans(dry: Calibration) -> None:
    caps = arm_caps(dry.frozen, ["one", "two"], 10.0)
    assert sum(caps.values()) == pytest.approx(5.0)
    # 196 absolute calls against 42 pair calls that read two pages each.
    assert caps[ABSOLUTE_SET] / caps[PAIR_SET] == pytest.approx(7 * len(dry.found) / (2 * 42))
