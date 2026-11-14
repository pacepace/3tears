"""Rung 2, ``examples/2-replies/``: a judge as a grader, checked two ways: on its examples, and against a person.

The run grades the judge's four examples as it grades the replies and says how many it got right, from the recording.
The rung ships with no person's labels, so the run it records says so, by name, with the step that changes it; that
state is the lesson and is asserted here. The ``blind_read.html`` and ``blind_read.csv`` beside it are what the
recorded run wrote: the replies in one seeded random order, with no run, result or judge score on them, and a run
replayed from the recordings writes the same bytes, since the draw depends on the answers alone. Two tests fill the
sheet in to show ``Judge(labels=...)`` reading it back, from the top and out of order: their labels are a test's,
never shipped, and say nothing about how good the judge is.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from threetears.evals import Example, Judge
from threetears.evals.quick.files import main
from threetears.evals.testing import RecordedCompletions, RecordedPrompt
from packages.evals.tests.example_loader import EXAMPLES
from packages.evals.tests.report_support import page_violations
from packages.evals.tests.rung_support import (
    program_lines,
    recorded,
    recorded_cost,
    recorded_settings,
    root_only_imports,
)
from packages.evals.tests.ladder_support import offline_copy
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

RUNG = EXAMPLES / "2-replies"

#: What the honest first run prints about its judge: no labels, and the step that calibrates it.
UNCALIBRATED = (
    "judge: 0 of 20 items read blind: judge uncalibrated (labels.csv is not there yet); next step: open the "
    "blind_read.html the report writes, answer its items from the top, and save its labels as labels.csv"
)

#: What the run prints about the judge's examples: each graded twice, as the recording holds it.
EXAMPLES_GRADED = "judge: 4 of 4 examples graded as they must be on every one of 2 repeats ($0.00086 over 8 call(s))"


@pytest.fixture
def replies_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of ``examples/2-replies/`` as the working directory, with no key: every call must replay."""
    return offline_copy("2-replies", tmp_path, monkeypatch)


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("replies_rung", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as read:
        return list(csv.DictReader(read))


def test_no_person_s_labels_ship_with_the_rung() -> None:
    assert not (RUNG / "labels.csv").exists(), "labels are a person's to write; none may ship"
    assert not (RUNG / "to_label.csv").exists(), "the sheet in store order is retired"
    rows = _rows(RUNG / "blind_read.csv")
    assert {row["label"] for row in rows} == {""} and {row["rater"] for row in rows} == {""}
    assert {row["prefer"] for row in rows} == {""} and {row["rater_kind"] for row in rows} == {""}


def test_the_judge_is_three_pass_fail_dimensions_four_hard_negatives_and_another_family(replies_dir: Path) -> None:
    module = _load(replies_dir / "replies.py")
    judge = module.judge
    assert isinstance(judge, Judge)
    assert [(dim.name, dim.scale) for dim in judge.dims] == [
        ("answer.grounded", "pass_fail"),
        ("answer.answers", "pass_fail"),
        ("answer.safe", "pass_fail"),
    ]
    assert len(judge.examples) == 4 and all(isinstance(example, Example) for example in judge.examples)
    # Each hard negative pairs with one that looks like it and must get the other verdict.
    assert sorted((example.dimension, example.expected) for example in judge.examples) == [
        ("grounded", "fail"),
        ("grounded", "pass"),
        ("safe", "fail"),
        ("safe", "pass"),
    ]
    assert judge.labels == Path("labels.csv")
    drafter_family, judge_family = module.drafter.model_name.split("/")[0], judge.model.split("/")[0]
    assert (drafter_family, judge_family) == ("deepseek", "google")


def test_the_runner_s_recording_covers_every_reply(replies_dir: Path) -> None:
    """Checked before any replay. The judge's prompts are the engine's, over each reply: a run that misses one
    excludes its cell, which the run test below would print."""
    module = _load(replies_dir / "replies.py")
    system = f"Reply to this customer email for Acme Notes.\n\n{module.POLICY}"
    emails = [json.loads(line) for line in (replies_dir / "emails.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(emails) == 20
    path = replies_dir / "recordings" / "replies.jsonl"
    calls = [RecordedPrompt(system, email["email"]) for email in emails for _ in range(2)]
    RecordedCompletions.check_covers(path, calls, settings=recorded_settings(path), rerecord="RECORD=1")
    judged = recorded(replies_dir / "recordings" / "judge.jsonl")
    assert len(judged) == 20 * 2 * 3 + 4 * 2, "every reply on every dimension, and every example, each twice"
    assert {line["result"]["model"] for line in judged} == {"google/gemini-2.5-flash-lite"}  # type: ignore[index]
    assert {line["result"]["temperature"] for line in judged} == {0.0}  # type: ignore[index]


def test_the_run_replays_offline_and_says_its_judge_is_uncalibrated_with_the_next_step(
    replies_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["replies.py"]) == 0

    printed = capsys.readouterr().out
    assert "40 result(s): 40 scored, 0 excluded" in printed
    for dimension in ("grounded", "answers", "safe"):
        assert f"answer {dimension} (judged), scored pass/fail: 100%, 40 of 40 trials" in printed
    assert UNCALIBRATED in printed.splitlines()
    # Read right under it: the judge on the answers it must grade a known way, each twice, every one as it must be.
    assert printed.splitlines()[printed.splitlines().index(UNCALIBRATED) + 1] == EXAMPLES_GRADED
    assert "  missed: " not in printed
    page = (replies_dir / ".evals" / "reports" / "replies" / "report.md").read_text(encoding="utf-8")
    assert (
        "The judge graded 4 of 4 of its examples as they must be on every one of 2 repeats ($0.00086 over 8 call(s))."
        in page.splitlines()
    ), "the page states what the examples cost, as the print does"
    # The judge is another family, so nothing warns that a model judged its own answers.
    assert "self-judging:" not in printed

    report_dir = replies_dir / ".evals" / "reports" / "replies"
    rows = _rows(report_dir / "blind_read.csv")
    # One reply per email, drawn among its two repeats, on each of the three dimensions: 20 items, in the drawn order.
    assert len(rows) == 60 and [int(row["position"]) for row in rows[::3]] == list(range(1, 21))
    assert {row["label"] for row in rows} == {""}, "no verdict is on the sheet"
    # The draw depends on the answers alone, so a replayed run writes the shipped page and sheet, byte for byte.
    for name in ("blind_read.html", "blind_read.csv"):
        written = (report_dir / name).read_bytes()
        assert written == (RUNG / name).read_bytes(), f"the shipped {name} is not what a replayed run writes"
        # Neither names a run or a result, and the judge's model and reasons are nowhere on it.
        for mark in (b"run_id", b"result_id", b"gemini", b"eval_run"):
            assert mark not in written, f"{name} carries {mark!r}"


async def test_the_judge_s_spend_is_its_own_and_its_model_judged_none_of_its_own_answers(replies_dir: Path) -> None:
    module = _load(replies_dir / "replies.py")

    result = await module.replies_eval.run(store=":memory:")

    summary = result.summary
    assert (summary.n_cases, summary.k_runs, summary.n_results, summary.n_scored) == (20, 2, 40, 40)
    assert summary.measures == [], "the judge is the only grade"
    assert summary.role_spend("judge").calls == 120, "20 emails x 2 repeats x 3 dimensions, one call each"
    examples = result.judge_examples
    assert examples is not None and examples.repeats == 2
    assert [(grade.example.dimension, grade.expected, grade.verdicts) for grade in examples.grades] == [
        ("grounded", "fail", ("fail", "fail")),
        ("grounded", "pass", ("pass", "pass")),
        ("safe", "fail", ("fail", "fail")),
        ("safe", "pass", ("pass", "pass")),
    ]
    # The examples' calls are the judge's, apart from the replies' results: 4 examples x 2 repeats, one call each.
    assert (examples.spend.role, examples.spend.calls, examples.spend.unpriced_calls) == ("judge", 8, 0)
    # They were recorded after the replies' 120, onto the same file, and cost what those 8 lines say.
    added = [line["result"] for line in recorded(replies_dir / "recordings" / "judge.jsonl")[120:]]
    costs = [result["cost_usd"] for result in added if isinstance(result, dict)]
    assert len(costs) == 8 and examples.spend.cost_usd == pytest.approx(sum(costs))
    assert summary.judge_shares_candidate_model == [] and summary.self_judging_warning is None
    assert result.calibration_lines() == [UNCALIBRATED]
    assert page_violations(result.report) == []


def _label(replies_dir: Path, positions: set[int]) -> None:
    """Save ``labels.csv``: the shipped sheet with every reply at ``positions`` marked as the judge marked it."""
    rows = _rows(replies_dir / "blind_read.csv")
    for row in rows:
        if int(row["position"]) in positions:
            row["label"], row["rater"] = "pass", "this test, not a person"
    with (replies_dir / "labels.csv").open("w", encoding="utf-8", newline="") as written:
        writer = csv.DictWriter(written, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_a_sheet_answered_from_the_top_is_read_back_as_a_random_sample_and_the_judge_s_agreement(
    replies_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mechanics only: a test's labels (every reply marked as the judge marked it), never a person's."""
    _label(replies_dir, set(range(1, 15)))

    assert main(["replies.py"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert UNCALIBRATED not in lines
    at = lines.index("judge: 14 of 20 items read blind, in the drawn order (a random sample)")
    # The judge passed every reply and so did the labels, so kappa has nothing to measure: said, with what to do.
    undefined = (
        "14 results from 14 cases  kappa undefined: you and the judge gave every answer the same verdict; read further"
    )
    assert lines[at + 1 : at + 5] == [
        f"  answer.answers   {undefined}",
        f"  answer.grounded  {undefined}",
        f"  answer.safe      {undefined}",
        "next: keep going from item 15 in blind_read.html",
    ]
    assert lines[at + 5] == EXAMPLES_GRADED
    # The ladder shows these lines as what a read from the top prints; it is held to them here.
    ladder = (REPO_ROOT / "packages" / "evals" / "docs" / "ladder.md").read_text(encoding="utf-8")
    assert "\n".join(lines[at : at + 5]) in ladder, "docs/ladder.md's rung 2 labelled print is not what the run prints"
    assert not any(line.startswith("preference") for line in lines), "one arm has no pair to prefer between"


def test_a_sheet_answered_out_of_order_reads_as_selected(replies_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _label(replies_dir, {2, 5, 11})

    assert main(["replies.py"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert "judge: 3 of 20 items read blind, out of the drawn order (selected, not a random sample)" in lines
    assert "next: for a random sample, answer blind_read.html from item 1, in order" in lines


async def test_the_result_carries_the_read_and_each_standing(replies_dir: Path) -> None:
    _label(replies_dir, set(range(1, 21)))
    module = _load(replies_dir / "replies.py")

    result = await module.replies_eval.run(store=":memory:")

    read = result.blind_read
    assert read is not None and read.report is not None
    assert (len(read.items), read.draw.population, read.draw.excluded) == (20, 20, {})
    assert [arm.role for arm in read.arms] == ["control"] and read.ask == ["verdicts"]
    (standing,) = read.report.standings
    assert (standing.standing, standing.answered, standing.items) == ("random", 20, 20)
    assert [(c.dimension, c.labels, c.kappa) for c in result.calibrations] == [
        ("answer.answers", 20, None),
        ("answer.grounded", 20, None),
        ("answer.safe", 20, None),
    ]
    assert not any(line.startswith("next:") for line in result.calibration_lines())


def test_the_docstring_states_the_cost_its_recordings_record() -> None:
    docstring = ast.get_docstring(ast.parse((RUNG / "replies.py").read_text(encoding="utf-8")))
    assert docstring is not None
    said = " ".join(docstring.split())
    for recording, calls_word in (("replies.jsonl", "reply calls"), ("judge.jsonl", "judge calls")):
        calls, cost = recorded_cost(RUNG / "recordings" / recording)
        assert f"{calls} {calls_word}" in said and cost in said, f"replies.py does not state {recording}'s {cost}"


def test_the_program_is_about_forty_lines_and_imports_only_the_package_root() -> None:
    path = RUNG / "replies.py"
    assert program_lines(path) <= 45
    assert root_only_imports(path) == []
    assert public_root_violations(SOURCE_ROOT, [("replies.py", path)], consumer_root=REPO_ROOT) == []
