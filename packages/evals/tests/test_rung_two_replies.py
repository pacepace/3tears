"""Rung 2, ``examples/2-replies/``: a judge as a grader, checked on its examples and against a reference read.

The run grades the judge's four examples as it grades the replies and says how many it got right, from the recording.
The reference is ``labels.csv``: a person's blind read of 20 replies, adjudicated by the person reader where a model's
blind read disagreed, with no independent adjudicator (owner rulings, 2026-11-16), and two items re-ruled by the
owner on 2026-11-17 under the new wording. Its construction from ``adjudication/`` (both reads, the rulings and the
re-checks) is checked here with every exclusion counted, and it binds to the
``grounded`` wording the owner ruled under: the person's own read, bound to the earlier wording, does not pair on it.
A second eval judges the same replies on the same words one tier up, against the same reference. The judge's agreement
is printed as counts with exact intervals (#1208). The ``blind_read.html`` and ``blind_read.csv`` beside it are what
the recorded run writes; the reference was read on an earlier draw of the same replies, so its labels pair by the
reply they were given on, as not drawn at random. Two tests overwrite the labels to show ``Judge(labels=...)`` reading
a partial sheet back, from the top and out of order: their labels are a test's and say nothing about the judge.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

from threetears.evals import Example, Judge
from threetears.evals.quick.files import main
from threetears.evals.quick.result import read_blind
from packages.evals.tests.off_loop import off_loop
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

#: Whose the reference is, with its limit, as its rows name it and the print says it.
REFERENCE = "Brooks (adjudicated by the person reader; no independent adjudicator)"

#: The reference's labels, said once, for each judge's print.
REFERENCE_LINE = (
    f"judge: 55 label(s) in labels.csv by {REFERENCE} are not on this run's draw (the answers changed since, or it is "
    "an older sheet): kept, each paired with the same output, and not drawn at random"
)

#: The safe line, the same for both judges: the reference failed no reply on it.
SAFE = (
    "  answer safe (judged)      19 results from 19 cases: the reader failed no answer, so no catch rate; the judge "
    "failed 0 of the 19 the reader passed (exact 95% interval 0% to 18%); the reader gave every answer one verdict, "
    "pass; the judge gave every answer one verdict, pass"
)

#: What the run prints about its judge against the reference: counts, the catch and false-fail rates, exact intervals.
JUDGE_READ = [
    REFERENCE_LINE,
    "  answer answers (judged)   20 results from 20 cases: the judge failed 0 of the 1 answers the reader failed "
    "(exact 95% interval 0% to 97%); the judge failed 0 of the 19 the reader passed (exact 95% interval 0% to 18%); "
    "the judge gave every answer one verdict, pass",
    "  answer grounded (judged)  16 results from 16 cases: the judge failed 1 of the 2 answers the reader failed "
    "(exact 95% interval 1% to 99%); the judge failed 0 of the 14 the reader passed (exact 95% interval 0% to 23%)",
    SAFE,
]

#: The stronger judge, on the same replies, words and reference.
STRONGER_READ = [
    REFERENCE_LINE,
    "  answer answers (judged)   20 results from 20 cases: the judge failed 0 of the 1 answers the reader failed "
    "(exact 95% interval 0% to 97%); the judge failed 3 of the 19 the reader passed (exact 95% interval 3% to 40%)",
    "  answer grounded (judged)  16 results from 16 cases: the judge failed 2 of the 2 answers the reader failed "
    "(exact 95% interval 16% to 100%); the judge failed 4 of the 14 the reader passed (exact 95% interval 8% to 58%)",
    SAFE,
]

#: What the run prints about the judge's examples: each graded twice, as the recording holds it.
EXAMPLES_GRADED = "judge: 4 of 4 examples graded as they must be on every one of 2 repeats ($0.00090 over 8 call(s))"


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


def _reference(
    person: list[dict[str, str]], model: list[dict[str, str]], rulings: list[dict[str, str]]
) -> Counter[str]:
    """How each of the 60 (reply, dimension) pairs comes into the reference, by the owner's rulings of 2026-11-16."""
    said = {(row["item"], row["dimension"]): row["label"] for row in model}
    ruled = {(row["item"], row["dimension"]): row["ruling"] for row in rulings}
    how: Counter[str] = Counter()
    for row in person:
        key = (row["item"], row["dimension"])
        if row["label"] == said[key]:
            # Agreed, except a both-fail on grounded: failed under the exhaustive wording, read again under this one.
            failed = row["dimension"] == "answer.grounded" and row["label"] == "fail"
            how["agreed, re-check" if failed else "agreed"] += 1
        else:
            how[f"ruled {ruled[key]}"] += 1
    return how


def test_the_reference_is_the_agreed_labels_and_the_rulings_with_every_exclusion_counted() -> None:
    person, model = _rows(RUNG / "adjudication" / "person.csv"), _rows(RUNG / "adjudication" / "model.csv")
    rulings = _rows(RUNG / "adjudication" / "rulings.csv")
    assert {row["rater"] for row in person} == {"Brooks"} and {row["rater"] for row in model} == {"claude-fable-5-2"}
    # 13 disagreements, each ruled blind to who said what: 9 decided, 4 unclear; one agreed fail set aside.
    assert len(rulings) == 13 and _reference(person, model, rulings) == {
        "agreed": 46,
        "agreed, re-check": 1,
        "ruled pass": 7,
        "ruled fail": 2,
        "ruled unclear": 4,
    }
    sided = Counter(
        "person" if ruling["ruling"] == by_person else "model"
        for ruling in rulings
        if ruling["ruling"] != "unclear"
        for by_person in [
            next(r["label"] for r in person if (r["item"], r["dimension"]) == (ruling["item"], ruling["dimension"]))
        ]
    )
    assert sided == {"person": 7, "model": 2}, "the rulings sided with the person on 7 and the model on 2"
    labels = _rows(RUNG / "labels.csv")
    assert {row["rater"] for row in labels} == {REFERENCE} and {row["rater_kind"] for row in labels} == {"person"}
    # Every reply the person read, on every dimension, once; the reference is each pair's label as ruled.
    assert sorted((row["item"], row["dimension"]) for row in labels) == sorted(
        (row["item"], row["dimension"]) for row in person
    )
    said = {(row["item"], row["dimension"]): row for row in labels}
    ruled = {(row["item"], row["dimension"]): row["ruling"] for row in rulings}
    # The owner's re-checks of 2026-11-17 under the new wording, which stand over the earlier rulings.
    rechecks = _rows(RUNG / "adjudication" / "rechecks.csv")
    assert {(row["item"], row["dimension"], row["ruling"], row["ruled_on"]) for row in rechecks} == {
        ("ded39e41e5fa", "answer.grounded", "fail", "2026-11-17"),
        ("5050d620fcb5", "answer.grounded", "pass", "2026-11-17"),
        ("897bcf0a55b3", "answer.grounded", "excluded", "2026-11-17"),
    }
    rechecked = {(row["item"], row["dimension"]): row["ruling"] for row in rechecks}
    for row in person:
        key = (row["item"], row["dimension"])
        agreed = next(r["label"] for r in model if (r["item"], r["dimension"]) == key) == row["label"]
        expected = rechecked.get(key, row["label"] if agreed else ruled[key])
        if expected in ("unclear", "excluded"):
            expected = ""
        assert said[key]["label"] == expected, key
        assert said[key]["output_fingerprint"] == row["output_fingerprint"], "each label is bound to the reply read"
    # The locked-out reply's grounded is reversed to fail; its safe stays excluded as unclear.
    assert (said[("ded39e41e5fa", "answer.grounded")]["label"], said[("ded39e41e5fa", "answer.safe")]["label"]) == (
        "fail",
        "",
    )
    assert said[("ded39e41e5fa", "answer.grounded")]["reason"] == (
        "re-ruled by the owner on 2026-11-17 under the new wording"
    )
    # 55 labels, 5 excluded, each with why on its row: none is counted, since a row with no label pairs with nothing.
    counted = Counter((row["dimension"], row["label"]) for row in labels if row["label"])
    assert counted == {
        ("answer.grounded", "pass"): 14,
        ("answer.grounded", "fail"): 2,
        ("answer.answers", "pass"): 19,
        ("answer.answers", "fail"): 1,
        ("answer.safe", "pass"): 19,
    }
    assert Counter(row["reason"] for row in labels if not row["label"]) == {
        "excluded: unclear on adjudication": 4,
        "excluded: borderline under the new wording; excluded by the owner, 2026-11-17": 1,
    }


def test_the_reference_binds_to_the_grounded_wording_ruled_on_and_the_earlier_read_does_not_pair_on_it(
    replies_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = {row["dimension"]: row["criterion_fingerprint"] for row in _rows(RUNG / "blind_read.csv")}
    person = {row["dimension"]: row["criterion_fingerprint"] for row in _rows(RUNG / "adjudication" / "person.csv")}
    labels = {row["dimension"]: row["criterion_fingerprint"] for row in _rows(RUNG / "labels.csv")}
    assert labels == sheet, "the reference is bound to the words the judge is asked now"
    assert person["answer.grounded"] != sheet["answer.grounded"], "the person's read answered the earlier words"
    assert {dim: person[dim] for dim in ("answer.answers", "answer.safe")} == {
        dim: sheet[dim] for dim in ("answer.answers", "answer.safe")
    }
    # The person's read as it was saved, bound to the earlier grounded wording: it pairs on answers and safe only.
    (replies_dir / "labels.csv").write_bytes((RUNG / "adjudication" / "person.csv").read_bytes())

    assert main(["replies.py"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("judge: 40 label(s) in labels.csv by Brooks are not on this run's draw")
    # Its 20 grounded labels answered other words: not read, and said, never counted.
    assert lines[2].startswith("  20 label(s) in labels.csv not read: line 2: it answered other wording of ")
    assert lines[2].endswith("than the judge is asked now; and 17 more")
    # The check reads answers and safe alone, and the line after them is the examples'.
    assert [line.split("  ")[1] for line in lines[3:5]] == ["answer answers (judged)", "answer safe (judged)"]
    assert lines[5] == EXAMPLES_GRADED


def test_the_judge_is_three_pass_fail_dimensions_four_hard_negatives_and_another_family(replies_dir: Path) -> None:
    module = _load(replies_dir / "replies.py")
    judge = module.judge
    assert isinstance(judge, Judge)
    assert [(dim.name, dim.scale) for dim in judge.dims] == [
        ("answer.grounded", "pass_fail"),
        ("answer.answers", "pass_fail"),
        ("answer.safe", "pass_fail"),
    ]
    # Owner ruling, 2026-11-16: grounded is consistent with the policy, not exhaustive of it.
    grounded = judge.dims[0].description
    assert (
        "contradicts the support policy" in grounded and "Where the policy says nothing, that is no failure" in grounded
    )
    assert len(judge.examples) == 4 and all(isinstance(example, Example) for example in judge.examples)
    # Each hard negative pairs with one that looks like it and must get the other verdict.
    assert sorted((example.dimension, example.expected) for example in judge.examples) == [
        ("grounded", "fail"),
        ("grounded", "pass"),
        ("safe", "fail"),
        ("safe", "pass"),
    ]
    assert judge.labels == Path("labels.csv") and judge.reviewer is None
    drafter_family, judge_family = module.drafter.model_name.split("/")[0], judge.model.split("/")[0]
    assert (drafter_family, judge_family) == ("deepseek", "google")
    # The stronger judge: one tier up, sent to Anthropic directly, on the same words, examples and reference.
    stronger = module.stronger
    assert stronger.model == "claude-sonnet-5-5"
    assert [(dim.name, dim.description) for dim in stronger.dims] == [(dim.name, dim.description) for dim in judge.dims]
    assert stronger.examples == judge.examples and stronger.labels == judge.labels and stronger.reviewer is None


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
    stronger = recorded(replies_dir / "recordings" / "judge_stronger.jsonl")
    assert len(stronger) == 20 * 2 * 3 + 4 * 2
    assert {line["result"]["served_model"] for line in stronger} == {"claude-sonnet-5-5"}  # type: ignore[index]
    assert {line["result"]["temperature"] for line in stronger} == {None}, "no temperature is sent to Claude"  # type: ignore[index]
    assert sorted(path.name for path in (replies_dir / "recordings").iterdir()) == [
        "judge.jsonl",
        "judge_stronger.jsonl",
        "replies.jsonl",
    ]


def test_the_run_replays_offline_and_prints_each_judge_against_the_reference(
    replies_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["replies.py"]) == 0

    printed = capsys.readouterr().out
    lines = printed.splitlines()
    assert "40 result(s): 40 scored, 0 excluded" in printed
    assert lines[0] == "replies: reply passed 39 of 40 trials on 20 cases. Cases that failed a trial: locked-out."
    # The judge against the reference: it failed 1 of the 2 replies the reference failed on grounded, none it passed.
    assert lines[1 : 1 + len(JUDGE_READ)] == JUDGE_READ
    # Read right under it: the judge on the answers it must grade a known way, each twice, every one as it must be.
    assert lines[1 + len(JUDGE_READ)] == EXAMPLES_GRADED
    assert "  missed: " not in printed
    # One tier up, the same replies, words and reference: it failed both grounded replies the reference failed, and
    # 4 of the 14 it passed.
    stronger = lines.index(
        "replies-stronger: reply_again passed 30 of 40 trials on 20 cases. Cases that failed a trial: monthly-refund, "
        "annual-late-refund, annual-early-refund, locked-out, send-password, and 5 more."
    )
    assert lines[stronger + 1 : stronger + 1 + len(STRONGER_READ)] == STRONGER_READ
    # No kappa, bar, tier word or count of results still needed (#1208).
    for line in [*lines[1 : 1 + len(JUDGE_READ)], *lines[stronger + 1 : stronger + 1 + len(STRONGER_READ)]]:
        for word in ("kappa", "calibrated", "undetermined", "bar of", "more results"):
            assert word not in line, line
    page = (replies_dir / ".evals" / "reports" / "replies" / "report.md").read_text(encoding="utf-8")
    assert (
        "The judge graded 4 of 4 of its examples as they must be on every one of 2 repeats ($0.00090 over 8 call(s))."
        in page.splitlines()
    ), "the page states what the examples cost, as the print does"
    # Neither judge is the drafter's family, so nothing warns that a model judged its own answers.
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
    assert result.calibration_lines() == JUDGE_READ
    assert page_violations(result.report) == []


def _label(replies_dir: Path, positions: set[int]) -> None:
    """Save ``labels.csv``: the shipped sheet with every reply at ``positions`` marked pass, in the reference's place."""
    rows = _rows(replies_dir / "blind_read.csv")
    for row in rows:
        if int(row["position"]) in positions:
            row["label"], row["rater"] = "pass", "a test"
    with (replies_dir / "labels.csv").open("w", encoding="utf-8", newline="") as written:
        writer = csv.DictWriter(written, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_a_sheet_answered_from_the_top_is_read_back_as_a_random_sample_and_the_judge_s_agreement(
    replies_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mechanics only: a test's labels (every reply marked pass) in place of the reference."""
    _label(replies_dir, set(range(1, 15)))

    assert main(["replies.py"]) == 0

    lines = capsys.readouterr().out.splitlines()
    at = lines.index("judge: 14 of 20 cases read blind by a test, in the drawn order (a random sample)")
    # The labels failed nothing, so no catch rate exists: said, with the one verdict each side gave, and no kappa.
    for line in lines[at + 1 : at + 4]:
        assert "14 results from 14 cases: the reader failed no answer, so no catch rate; " in line, line
        assert "the reader gave every answer one verdict, pass" in line and "kappa" not in line, line
    assert "next: keep going from case 15 in blind_read.html" in lines
    assert not any(line.startswith("preference") for line in lines), "one arm has no pair to prefer between"


def test_a_sheet_answered_out_of_order_reads_as_not_drawn_at_random(
    replies_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _label(replies_dir, {2, 5, 11})

    assert main(["replies.py"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert "judge: 3 of 20 cases read blind by a test, out of the drawn order (not drawn at random)" in lines
    assert "next: for a random sample, answer blind_read.html from case 1, in order" in lines


async def test_the_result_carries_the_read_and_each_standing(replies_dir: Path) -> None:
    _label(replies_dir, set(range(1, 21)))
    module = _load(replies_dir / "replies.py")

    result = await module.replies_eval.run(store=":memory:")

    closed, _ = off_loop(
        read_blind, result.judge, result.host, result.arms, result.control, result.scope_id, result.name
    )
    assert closed is not None
    read = closed.read
    assert read.report is not None
    assert sorted(result.blind_read_files) == ["blind_read.csv", "blind_read.html"]
    assert (len(read.items), read.draw.population, read.draw.excluded) == (20, 20, {})
    assert [arm.role for arm in read.arms] == ["control"] and read.ask == ["verdicts"]
    (standing,) = [sample for sample in read.report.samples if sample.made_by == "person"]
    assert (standing.sample, standing.answered, standing.items) == ("random", 20, 20)
    assert [(c.dimension, c.labels) for c in result.calibrations] == [
        ("answer.answers", 20),
        ("answer.grounded", 20),
        ("answer.safe", 20),
    ]
    assert not any(line.startswith("next:") for line in result.calibration_lines())


def test_the_docstring_states_the_cost_its_recordings_record() -> None:
    docstring = ast.get_docstring(ast.parse((RUNG / "replies.py").read_text(encoding="utf-8")))
    assert docstring is not None
    said = " ".join(docstring.split())
    for recording, calls_word in (
        ("replies.jsonl", "reply calls"),
        ("judge.jsonl", "judge calls on google/gemini-2.5-flash-lite"),
        ("judge_stronger.jsonl", "judge calls on claude-sonnet-5-5"),
    ):
        path = RUNG / "recordings" / recording
        calls, cost = recorded_cost(path)
        sources = {line["result"]["price_source"] for line in recorded(path)}  # type: ignore[index]
        # Every rate is a published or a provider-reported one: none is a catalogue's estimate.
        assert not any(str(source).startswith("openrouter-catalogue:") for source in sources), recording
        assert f"{calls} {calls_word}" in said and cost in said, f"replies.py does not state {recording}'s {cost}"


def test_the_program_is_about_forty_lines_and_imports_only_the_package_root() -> None:
    path = RUNG / "replies.py"
    assert program_lines(path) <= 45
    assert root_only_imports(path) == []
    assert public_root_violations(SOURCE_ROOT, [("replies.py", path)], consumer_root=REPO_ROOT) == []
