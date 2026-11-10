"""Rung 0: an eval in one TOML file, read into the same ``Eval`` a Python file builds, and run with no Python.

The file is read relative to itself; its runner is a prompt to a model replaying a recording made here by the
package's own recorder, so every test runs offline. What the loader refuses it refuses by name, before anything runs.

The runner sends what the app sends: a prompt and a system message, inline or read from the app's own files beside
the TOML, under the app's ``max_tokens`` and effort. A recording is keyed by the request (system, prompt and
settings), so a replay that succeeds proves the request is the one recorded, and an edit to a prompt file is refused
at replay exactly as an edit to an inline prompt is.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from threetears.evals import Eval
from threetears.evals.quick.files import main
from threetears.evals.quick.toml_file import load_toml
from threetears.evals.testing import RecordingRefused
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
    # Each measure printed as the page names it: the classifier's label match as its Accuracy.
    assert "Accuracy: 100%, 9 of 9 trials" in printed
    assert "Urgent score: 100%, 9 of 9 trials" in printed
    # The recorded model never sets needs_human, so the two emails that need a person miss on it.
    assert "Needs human score: 33%, 3 of 9 trials" in printed
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


#: The app's own prompt and system message, as files beside the TOML hold them (each ending in a newline).
SYSTEM = "You triage a support inbox for a small software company.\nAnswer only in the shape asked for."

#: The rung-0 file sending what the app sends, its texts read from the app's files.
FROM_FILES_TOML = """\
cases = "emails.jsonl"
grader = ["category", "urgent", "needs_human"]
recording = "recordings/triage.jsonl"

[runner]
model = "deepseek/deepseek-v4-flash"
prompt_file = "prompts/triage.md"
system_file = "prompts/system.md"
max_tokens = 600
effort = "minimal"

[runner.output]
category = ["billing", "bug", "account", "sales", "spam"]
urgent = "bool"
needs_human = "bool"
"""

#: The same eval with both texts inline.
INLINE_TOML = FROM_FILES_TOML.replace(
    'prompt_file = "prompts/triage.md"', 'prompt = """\nTriage this support email:\n\n{email}\n"""'
).replace('system_file = "prompts/system.md"', f'system = """\n{SYSTEM}\n"""')


async def _record_as_the_app_sends(path: Path, *, max_tokens: int | None, effort: str | None) -> None:
    """Record the model answering each email three times, sent as the app sends it: its system message, its prompt,
    and its settings, which the recording's key covers."""
    replies = {
        _prompt(str(email["email"])): json.dumps(
            {"category": email["category"], "urgent": email["urgent"], "needs_human": email["needs_human"]}
        )
        for email in EMAILS
    }
    await record_model(
        path,
        MODEL,
        list(replies),
        replies.__getitem__,
        schema=Output,
        repeats=3,
        system=SYSTEM,
        max_tokens=max_tokens,
        effort=effort,
    )


@pytest.fixture
def app_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A rung-0 directory whose prompt and system message are the app's files, recorded under the app's settings."""
    folder = tmp_path / "triage"
    (folder / "prompts").mkdir(parents=True)
    (folder / "prompts" / "triage.md").write_text(_prompt("{email}") + "\n", encoding="utf-8")
    (folder / "prompts" / "system.md").write_text(SYSTEM + "\n", encoding="utf-8")
    (folder / "emails.jsonl").write_text("".join(json.dumps(email) + "\n" for email in EMAILS), encoding="utf-8")
    (folder / "triage.toml").write_text(FROM_FILES_TOML, encoding="utf-8")
    (folder / "inline.toml").write_text(INLINE_TOML, encoding="utf-8")
    asyncio.run(_record_as_the_app_sends(folder / "recordings" / "triage.jsonl", max_tokens=600, effort="minimal"))
    monkeypatch.delenv("RECORD", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    return folder


def _first_case() -> dict[str, Any]:
    return dict(EMAILS[0])


def test_prompt_and_system_files_build_the_eval_the_inline_form_builds(app_files: Path) -> None:
    from_files, inline = load_toml(app_files / "triage.toml"), load_toml(app_files / "inline.toml")

    # Each replays the one recording, keyed by system, prompt and settings: both send the request the app recorded.
    results = [asyncio.run(ev.run(store=":memory:")) for ev in (from_files, inline)]
    assert [result.summary.n_scored for result in results] == [9, 9]
    assert [result.summary.n_excluded for result in results] == [0, 0]
    graded = [{measure.name: measure.mean for measure in result.summary.measures} for result in results]
    assert graded[0] == graded[1]
    # The classifier (category) is read as ``match``; every answer was the case's own, in either form.
    assert [graded[0][name] for name in ("match", "urgent", "needs_human")] == [1, 1, 1], graded[0]
    assert from_files.arms[None].__doc__ == inline.arms[None].__doc__ == "Triage this support email:"


def test_the_placeholders_check_reads_the_prompt_file_s_text(app_files: Path) -> None:
    (app_files / "prompts" / "triage.md").write_text("Triage this support email:\n\n{mail}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"triage.toml: the prompt fills \{mail\}, which case\(s\) charged"):
        load_toml(app_files / "triage.toml")


@pytest.mark.parametrize("edited", ["prompts/triage.md", "inline.toml"])
def test_an_edited_prompt_file_is_refused_at_replay_as_an_edited_inline_prompt_is(app_files: Path, edited: str) -> None:
    target = app_files / edited
    target.write_text(target.read_text(encoding="utf-8").replace("support email", "customer email"), encoding="utf-8")
    ev = load_toml(app_files / ("inline.toml" if edited == "inline.toml" else "triage.toml"))

    with pytest.raises(RecordingRefused, match="Triage this customer email"):
        asyncio.run(ev.arms[None](_first_case()))


def test_max_tokens_and_effort_reach_the_model(app_files: Path) -> None:
    """The recording holds the calls only under ``max_tokens = 600`` and effort ``minimal``: a file sending them
    replays, and the same file without them is refused naming the settings that differ."""
    path = app_files / "triage.toml"
    answered: Any = asyncio.run(load_toml(path).arms[None](_first_case()))
    assert answered.category == "billing"

    path.write_text(FROM_FILES_TOML.replace("max_tokens = 600\n", "").replace('effort = "minimal"\n', ""))
    with pytest.raises(RecordingRefused, match="recorded only under other request settings") as refused:
        asyncio.run(load_toml(path).arms[None](_first_case()))
    assert "max_tokens" in str(refused.value) and "minimal" in str(refused.value)


#: Every key ``[runner]`` takes, as each of its refusals names them.
RUNNER_KEYS = ("model", "prompt", "prompt_file", "system", "system_file", "max_tokens", "effort", "output")


@pytest.mark.parametrize(
    ("edit", "refusal"),
    [
        (
            lambda text: text.replace("[runner]\n", '[runner]\nprompt = "Triage {email}"\n'),
            "runner takes prompt or prompt_file, not both",
        ),
        (
            lambda text: text.replace("[runner]\n", '[runner]\nsystem = "Be brief."\n'),
            "runner takes system or system_file, not both",
        ),
        (lambda text: text.replace("[runner]\n", "[runner]\ntemperature = 0.2\n"), "runner takes no key temperature"),
        (lambda text: text.replace('prompt_file = "prompts/triage.md"\n', ""), "runner lacks prompt"),
        (lambda text: text.replace("prompts/triage.md", "prompts/gone.md"), "runner.prompt_file 'prompts/gone.md'"),
        (lambda text: text.replace("max_tokens = 600", "max_tokens = 0"), "runner: max_tokens 0 is not a positive"),
        (lambda text: text.replace('"minimal"', '"some"'), "runner: effort 'some' is not one of"),
        (lambda text: text.replace('"minimal"', "600"), r"runner: a reasoning budget \(600\) not below max_tokens"),
    ],
)
def test_what_runner_refuses_it_names_with_the_file_and_every_key_runner_takes(
    app_files: Path, edit: Callable[[str], str], refusal: str
) -> None:
    path = app_files / "triage.toml"
    path.write_text(edit(FROM_FILES_TOML), encoding="utf-8")

    with pytest.raises(ValueError, match=refusal) as refused:
        load_toml(path)
    said = str(refused.value)
    assert said.startswith(f"{path}: ")
    if refusal.startswith(("runner takes", "runner lacks")):
        assert all(key in said for key in RUNNER_KEYS), said
