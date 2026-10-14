"""Rung 0: an eval in one TOML file, read into the same ``Eval`` a Python file builds, and run with no Python.

The file is read relative to itself; its runner is a prompt to a model replaying a recording made here by the
package's own recorder, so every test runs offline. What the loader refuses it refuses by name, before anything runs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from threetears.evals import Eval
from threetears.evals.quick.files import main
from threetears.evals.quick.toml_file import load_toml
from packages.evals.tests.recorded_model_support import record_model

MODEL = "deepseek/deepseek-v4-flash"

#: The rung-0 file: three slots and a recording.
TRIAGE_TOML = '''\
# python -m threetears.evals triage.toml
cases = "emails.jsonl"
grader = ["category", "urgent", "needs_human"]
recording = "recordings/triage.jsonl"

[runner]
model = "deepseek/deepseek-v4-flash"
prompt = """
Triage this support email:

{email}
"""

[runner.output]
category = ["billing", "bug", "account", "sales", "spam"]
urgent = "bool"
needs_human = "bool"
'''

EMAILS = [
    {"id": "charged", "email": "I was charged twice.", "category": "billing", "urgent": False, "needs_human": True},
    {"id": "crash", "email": "The app crashes {always}.", "category": "bug", "urgent": True, "needs_human": True},
    {"id": "prize", "email": "You WON a prize.", "category": "spam", "urgent": False, "needs_human": False},
]


class Output(BaseModel):
    """The answer the file's ``[runner.output]`` declares, named as the loader names it."""

    category: Literal["billing", "bug", "account", "sales", "spam"]
    urgent: bool
    needs_human: bool


def _prompt(email: str) -> str:
    return f"Triage this support email:\n\n{email}"


@pytest.fixture
def rung_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory holding the file, its cases and the model's recording; the working directory is elsewhere."""
    folder = tmp_path / "triage"
    folder.mkdir()
    (folder / "triage.toml").write_text(TRIAGE_TOML, encoding="utf-8")
    (folder / "emails.jsonl").write_text("".join(json.dumps(email) + "\n" for email in EMAILS), encoding="utf-8")
    replies = {
        _prompt(str(email["email"])): json.dumps(
            {"category": email["category"], "urgent": email["urgent"], "needs_human": False}
        )
        for email in EMAILS
    }
    asyncio.run(
        record_model(
            folder / "recordings" / "triage.jsonl", MODEL, list(replies), replies.__getitem__, schema=Output, repeats=3
        )
    )
    monkeypatch.delenv("RECORD", raising=False)
    monkeypatch.chdir(tmp_path)
    return folder


def test_the_file_builds_the_eval_a_python_file_would(rung_zero: Path) -> None:
    ev = load_toml(rung_zero / "triage.toml")

    assert isinstance(ev, Eval)
    assert (ev.name, ev.k, len(ev.cases)) == ("triage", 3, 3)
    assert ev.plan.classifier == "category" and ev.plan.fields == ["category", "urgent", "needs_human"]
    runner = ev.arms[None]
    assert runner.__doc__ == "Triage this support email:", "the prompt's first line is what each case asks"


def test_the_command_line_runs_the_file_from_another_directory_and_fills_braces_from_the_case(
    rung_zero: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(rung_zero / "triage.toml")]) == 0

    printed = capsys.readouterr().out
    assert "category: mean 1 (n=9" in printed
    assert "urgent: mean 1 (n=9" in printed
    # The recorded model never sets needs_human, so the two emails that need a person miss on it.
    assert "needs_human: mean 0.3333 (n=9" in printed
    assert Path(".evals/reports/triage/report.html").is_file()


def test_a_file_with_no_output_answers_with_the_reply_s_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RECORD", raising=False)
    (tmp_path / "words.toml").write_text(
        f"""\
cases = [{{ id = "a", word = "billing", label = "billing" }}, {{ id = "b", word = "bug", label = "bug" }}]
grader = ["label"]
recording = "words.jsonl"
k = 1

[runner]
model = "{MODEL}"
prompt = "Echo {{word}}"
""",
        encoding="utf-8",
    )
    asyncio.run(record_model(tmp_path / "words.jsonl", MODEL, ["Echo billing", "Echo bug"], lambda p: p.split()[1]))

    ev = load_toml(tmp_path / "words.toml")
    result = asyncio.run(ev.run(store=":memory:"))

    assert result.summary.n_scored == 2 and result.summary.confusion[0].expected == "billing"


@pytest.mark.parametrize(
    ("edit", "refusal"),
    [
        (lambda text: 'colour = "red"\n' + text, "takes no key colour"),
        (lambda text: text.replace('grader = ["category", "urgent", "needs_human"]\n', ""), "lacks grader"),
        (lambda text: text.replace('urgent = "bool"', 'urgent = "boolean"'), "runner.output.urgent is 'boolean'"),
        (lambda text: text.replace('category = ["billing"', 'category = [1, "billing"'), "each a non-blank string"),
        (lambda text: text.replace("{email}", "{mail}"), r"fills \{mail\}, which case\(s\) charged, crash, prize lack"),
        (lambda text: text.replace("{email}", "{email!r}"), "names no case field"),
        (lambda text: text.replace("{email}", "{email"), "do not parse"),
        (
            lambda text: text.replace('"needs_human"]', '"needs_human", {judge = {rubric = {a = "b"}}}]'),
            "a judge is",
        ),
        (lambda text: text.replace("cases = ", "cases = 3 #"), "a list of JSON objects or a file"),
    ],
)
def test_what_the_loader_refuses_it_names_with_the_file(rung_zero: Path, edit: object, refusal: str) -> None:
    path = rung_zero / "triage.toml"
    path.write_text(edit(TRIAGE_TOML), encoding="utf-8")  # type: ignore[operator]

    with pytest.raises(ValueError, match=refusal):
        load_toml(path)


def test_a_file_that_is_not_toml_is_refused_by_the_command_line_with_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "broken.toml").write_text("cases = [", encoding="utf-8")

    assert main([str(tmp_path / "broken.toml")]) == 2
    assert "cannot be read as TOML" in capsys.readouterr().err
