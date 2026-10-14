"""Rung 2, ``examples/2-replies/``: a judge as a grader, its hard negatives, and calibration that starts uncalibrated.

The rung ships with no person's labels, so the run it records says so, by name, with the step that changes it; that
state is the lesson and is asserted here. The ``to_label.csv`` beside it is what the recorded run wrote, and a run
replayed from the recordings writes the same rows. One test fills that file in to show ``Judge(labels=...)`` reading
it back: its labels are a test's, never shipped, and say nothing about how good the judge is.
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
from packages.evals.tests.rung_support import (
    program_lines,
    recorded,
    recorded_cost,
    recorded_settings,
    root_only_imports,
    rung_copy,
)
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

RUNG = EXAMPLES / "2-replies"

#: What the honest first run prints about its judge: no labels, and the step that calibrates it.
UNCALIBRATED = (
    "judge: 0 labels: judge uncalibrated (labels.csv is not there yet); next step: rate about 20 outputs in the "
    "to_label.csv the report writes, without looking at the judge's verdict"
)

#: The columns of a ``to_label.csv`` row that are the same whichever run wrote it. The ids are each run's own, and a
#: case's two repeats may take its two recorded replies in either order, so ``repeat`` is left out too.
_STABLE = ("case", "dimension", "case_material", "answer", "output_fingerprint", "criterion_fingerprint")


@pytest.fixture
def replies_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of ``examples/2-replies/`` as the working directory, with no key: every call must replay."""
    return rung_copy("2-replies", tmp_path, monkeypatch)


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
    assert {row["label"] for row in _rows(RUNG / "to_label.csv")} == {""}
    assert {row["rater"] for row in _rows(RUNG / "to_label.csv")} == {""}


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
    writer_family, judge_family = module.writer.model_name.split("/")[0], judge.model.split("/")[0]
    assert (writer_family, judge_family) == ("deepseek", "google")


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
    assert len(judged) == 20 * 2 * 3
    assert {line["result"]["model"] for line in judged} == {"google/gemini-2.5-flash-lite"}  # type: ignore[index]
    assert {line["result"]["temperature"] for line in judged} == {0.0}  # type: ignore[index]


def test_the_run_replays_offline_and_says_its_judge_is_uncalibrated_with_the_next_step(
    replies_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["replies.py"]) == 0

    printed = capsys.readouterr().out
    assert "40 result(s): 40 scored, 0 excluded" in printed
    for dimension in ("grounded", "answers", "safe"):
        assert f"answer.{dimension} (judged pass/fail): mean 1 (n=40" in printed
    assert UNCALIBRATED in printed.splitlines()
    # The judge is another family, so nothing warns that a model judged its own answers.
    assert "self-judging:" not in printed

    written = replies_dir / ".evals" / "reports" / "replies" / "to_label.csv"
    rows = _rows(written)
    assert len(rows) == 120 and {row["label"] for row in rows} == {""}, "the judge's verdict is withheld"
    shipped = _rows(RUNG / "to_label.csv")
    # The same rows, in whatever order the store lists its results.
    assert sorted([row[column] for column in _STABLE] for row in rows) == sorted(
        [row[column] for column in _STABLE] for row in shipped
    ), "the shipped to_label.csv is not what a replayed run writes"


async def test_the_judge_s_spend_is_its_own_and_its_model_judged_none_of_its_own_answers(replies_dir: Path) -> None:
    module = _load(replies_dir / "replies.py")

    result = await module.replies_eval.run(store=":memory:")

    summary = result.summary
    assert (summary.n_cases, summary.k_runs, summary.n_results, summary.n_scored) == (20, 2, 40, 40)
    assert summary.measures == [], "the judge is the only grade"
    assert summary.role_spend("judge").calls == 120, "20 emails x 2 repeats x 3 dimensions, one call each"
    assert summary.judge_shares_candidate_model == [] and summary.self_judging_warning is None
    assert result.calibration_lines() == [UNCALIBRATED]


def test_a_filled_in_to_label_is_read_back_as_the_judge_s_calibration(
    replies_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mechanics only: a test's labels (every reply marked as the judge marked it), never a person's."""
    rows = _rows(replies_dir / "to_label.csv")
    for row in rows[:60]:
        row["label"], row["rater"] = "pass", "this test, not a person"
    with (replies_dir / "labels.csv").open("w", encoding="utf-8", newline="") as written:
        writer = csv.DictWriter(written, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert main(["replies.py"]) == 0

    printed = capsys.readouterr().out
    assert UNCALIBRATED not in printed
    read = [line for line in printed.splitlines() if line.startswith("judge against labels.csv: answer.")]
    assert len(read) == 3 and all(" label(s), " in line and "; tier " in line for line in read)


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
