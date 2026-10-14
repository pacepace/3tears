"""Rungs 0 and 1, ``examples/1-triage/``: one email-triage eval, as a TOML file and as a 15-line program, offline.

Both run from a copy of the directory against the recordings committed beside them, through the same command a
newcomer types (``python -m threetears.evals FILE``), so the test reads what that prints and the report it writes.
Nothing here assumes what the model said beyond what its recording holds: the figures asserted are the recording's.
"""

from __future__ import annotations

import re

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from threetears.evals.analysis.style import lint_page

from threetears.evals.kernel import strict_response_format
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

RUNG = EXAMPLES / "1-triage"

#: What the rung's README promises: the Python program is 15 lines, the TOML file no more.
MAX_PROGRAM_LINES = 15


@pytest.fixture
def triage_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of ``examples/1-triage/`` as the working directory, with no key: every call must replay."""
    return rung_copy("1-triage", tmp_path, monkeypatch)


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("triage_rung", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _emails(directory: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (directory / "emails.jsonl").read_text(encoding="utf-8").splitlines()]


def test_the_cases_are_thirty_emails_with_lookalike_hard_negatives() -> None:
    emails = _emails(RUNG)
    assert len(emails) == 30 and len({email["id"] for email in emails}) == 30
    assert {email["stratum"] for email in emails} == {"plain", "lookalike"}
    lookalikes = [email for email in emails if email["stratum"] == "lookalike"]
    assert len(lookalikes) >= 10
    # A lookalike looks like another category: a phishing mail shouting about the account is spam.
    assert {email["category"] for email in lookalikes} == {"billing", "bug", "account", "sales", "spam"}
    assert {email["category"] for email in emails} == {"billing", "bug", "account", "sales", "spam"}
    assert all(set(email) == {"id", "stratum", "email", "category", "urgent", "needs_human"} for email in emails)


def test_the_program_is_fifteen_lines_and_the_toml_file_no_longer() -> None:
    assert program_lines(RUNG / "triage.py") <= MAX_PROGRAM_LINES
    assert len((RUNG / "triage.toml").read_text(encoding="utf-8").splitlines()) <= MAX_PROGRAM_LINES


def test_the_line_count_reads_the_program_and_not_its_docstring(tmp_path: Path) -> None:
    padded = tmp_path / "triage.py"
    padded.write_text((RUNG / "triage.py").read_text(encoding="utf-8") + "\nprint('one line more')\n", encoding="utf-8")
    assert program_lines(padded) == program_lines(RUNG / "triage.py") + 1


def test_each_recording_covers_every_call_its_file_makes(triage_dir: Path) -> None:
    """Checked before any replay, so a prompt that changed under its recording is named rather than met as a miss."""
    module = _load(triage_dir / "triage.py")
    prompts = [f"Triage this support email:\n\n{email['email']}" for email in _emails(triage_dir)]
    for recording, schema_name in (("triage.jsonl", "Triage"), ("triage-toml.jsonl", "Output")):
        path = triage_dir / "recordings" / recording
        directive = strict_response_format(schema_name, module.Triage.model_json_schema() | {"title": schema_name})
        calls = [RecordedPrompt("", prompt, directive) for prompt in prompts for _ in range(3)]
        RecordedCompletions.check_covers(path, calls, settings=recorded_settings(path), rerecord="RECORD=1")
        assert {line["result"]["model"] for line in recorded(path)} == {"deepseek/deepseek-v4-flash"}  # type: ignore[index]


def test_the_program_runs_offline_prints_its_readings_and_writes_its_report(
    triage_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["triage.py"]) == 0

    printed = capsys.readouterr().out
    assert "90 result(s): 90 scored" in printed and "excluded" not in printed
    # The classifier is the first field, read per label; the other field and the code grader are pass rates.
    assert "category: mean 0.9 (n=90" in printed and "per label:" in printed
    # Each label's precision and recall count answers over cases, since k=3 answers each case three times.
    assert (
        "    account: precision 1 (17/17 over 7 cases, 95% CI 0.5687-1), recall 0.8095 (17/21 over 7 cases" in printed
    )
    assert "urgent: mean 0.8333 (n=90" in printed
    assert "escalates: mean 0.9889 (n=90" in printed
    assert "top misses (5 of 20):" in printed
    assert "total spend: $0.00192, the results' cost" in printed
    assert "stored in .evals/evals.sqlite, scope 'emails'" in printed
    assert (triage_dir / ".evals" / "reports" / "emails" / "report.html").is_file()
    page = (triage_dir / ".evals" / "reports" / "emails" / "report.md").read_text(encoding="utf-8")
    # One arm's verdict is how its cases came out, first, under the title and the arm.
    assert re.fullmatch(r"\*\*emails: \d+/\d+ cases passed; \d+ misses: .+\*\*", page.splitlines()[3])
    assert lint_page(page, names={page.splitlines()[1].split(" · ")[0]}) == []


def test_the_toml_file_runs_from_another_directory_and_its_symmetric_field_misses_what_escalates_forgives(
    triage_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rung 0 grades ``needs_human`` both ways; rung 1's ``escalates`` fails only an email left alone."""
    monkeypatch.chdir(triage_dir.parent)
    assert main([str(triage_dir / "triage.toml")]) == 0

    printed = capsys.readouterr().out
    assert "90 result(s): 90 scored" in printed and "excluded" not in printed
    assert "needs_human: mean 0.8111 (n=90" in printed
    assert "total spend: $0.00199, the results' cost" in printed
    assert (triage_dir.parent / ".evals" / "reports" / "triage" / "report.html").is_file()


def test_the_program_runs_from_another_directory_through_the_module_entry(
    triage_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One door: ``python -m threetears.evals examples/1-triage/triage.py`` from the package's directory, not the rung's.

    Its cases and its recording are read beside the file, and the store and the report land where it is run.
    """
    monkeypatch.chdir(triage_dir.parent)
    assert main([f"{triage_dir.name}/triage.py"]) == 0

    printed = capsys.readouterr().out
    assert "90 result(s): 90 scored" in printed and "excluded" not in printed
    assert "escalates: mean 0.9889 (n=90" in printed
    assert (triage_dir.parent / ".evals" / "reports" / "emails" / "report.html").is_file()
    assert not (triage_dir / ".evals").exists()


def test_an_email_the_recording_does_not_hold_is_excluded_rather_than_sent_to_a_model(
    triage_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    added = {"id": "new", "stratum": "plain", "email": "Is there a student discount?", "category": "sales"}
    with (triage_dir / "emails.jsonl").open("a", encoding="utf-8") as cases:
        cases.write(json.dumps(added | {"urgent": False, "needs_human": False}) + "\n")

    assert main(["triage.py"]) == 0

    printed = capsys.readouterr().out
    assert "excluded: 3 x " in printed and "Is there a student discount" in printed


def test_each_docstring_states_the_cost_its_recordings_record() -> None:
    docstring = _load_docstring(RUNG / "triage.py")
    for recording in ("triage.jsonl", "triage-toml.jsonl"):
        calls, cost = recorded_cost(RUNG / "recordings" / recording)
        assert f"{calls} calls, {cost}" in docstring, f"triage.py's docstring does not state {recording}'s {cost}"


def _load_docstring(path: Path) -> str:
    docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert docstring is not None
    return " ".join(docstring.split())


def test_the_program_imports_only_the_package_root() -> None:
    path = RUNG / "triage.py"
    assert root_only_imports(path) == []
    assert public_root_violations(SOURCE_ROOT, [("triage.py", path)], consumer_root=REPO_ROOT) == []


def test_the_inventory_guide_s_last_step_writes_the_rung_s_own_files() -> None:
    """``docs/building-an-inventory.md`` ends by writing rung 0: its blocks are the shipped files, verbatim."""
    guide = (RUNG.parents[1] / "docs" / "building-an-inventory.md").read_text(encoding="utf-8")
    toml_block = guide.split("```toml\n", 1)[1].split("```", 1)[0]
    assert toml_block == (RUNG / "triage.toml").read_text(encoding="utf-8")
    json_lines = guide.split("```json\n", 1)[1].split("```", 1)[0].splitlines()
    shipped = (RUNG / "emails.jsonl").read_text(encoding="utf-8").splitlines()
    assert json_lines and all(line in shipped for line in json_lines)
