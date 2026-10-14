"""Rung 1, the whole program: email triage in one file, run by ``python -m threetears.evals triage.py``, offline.

The program below is the ladder's first Python rung as a newcomer writes it (``docs`` and the example directory carry
the same text): a pydantic answer, a runner calling ``Model``, one asymmetric grader, and ``Eval`` over a cases file.
It runs here exactly as written, against a recording of the model made by the package's own recorder, so the test
reads what the command line prints, the report it writes and the spend the model's calls reported.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from typing import Literal

import pytest
from pydantic import BaseModel

from threetears.evals.quick.files import main
from packages.evals.tests.recorded_model_support import CALL_COST_USD, record_model

#: The program, verbatim: 20 lines, 15 not blank (the brief counts 19; its block has 20).
TRIAGE_PY = '''\
from typing import Literal
from pydantic import BaseModel
from threetears.evals import Eval, Model

model = Model("deepseek/deepseek-v4-flash", recording="recordings/triage.jsonl")

class Triage(BaseModel):
    category: Literal["billing", "bug", "account", "sales", "spam"]
    urgent: bool
    needs_human: bool

async def triage(case: dict) -> Triage:
    """Triage an inbound support email."""
    return await model.json(f"Triage this support email:\\n\\n{case['email']}", Triage)

def escalates(case: dict, out: Triage) -> bool:
    """Never leaves alone an email a person must see."""
    return out.needs_human or not case["needs_human"]

triage_eval = Eval("emails.jsonl", runner=triage, grader=["category", "urgent", escalates])
'''

#: Each email: its id, its text, then the category, urgency and need for a person a correct triage gives it.
_EMAILS = [
    ("double-charge", "I was charged twice this month.", "billing", False, True),
    ("crash", "The app crashes when I log in.", "bug", True, True),
    ("seats", "Can we buy ten more seats?", "sales", False, False),
    ("prize", "You WON a prize, click here.", "spam", False, False),
    ("lookalike", "Refund the crash, I want my money.", "billing", True, True),
]
EMAILS = [
    {"id": id_, "email": text, "category": category, "urgent": urgent, "needs_human": human}
    | ({"stratum": "lookalike"} if id_ == "lookalike" else {})
    for id_, text, category, urgent, human in _EMAILS
]

#: What the recorded model answered for each email: right on all but the lookalike, which it calls a bug and leaves.
ANSWERS = {
    "double-charge": {"category": "billing", "urgent": False, "needs_human": True},
    "crash": {"category": "bug", "urgent": True, "needs_human": True},
    "seats": {"category": "sales", "urgent": False, "needs_human": False},
    "prize": {"category": "spam", "urgent": False, "needs_human": False},
    "lookalike": {"category": "bug", "urgent": True, "needs_human": False},
}


class Triage(BaseModel):
    """The program's answer, declared alike so the recording is keyed by the schema the program sends."""

    category: Literal["billing", "bug", "account", "sales", "spam"]
    urgent: bool
    needs_human: bool


def _prompt(email: dict[str, object]) -> str:
    return f"Triage this support email:\n\n{email['email']}"


@pytest.fixture
def rung_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The example directory: the program, its cases and the model's recording, as the working directory."""
    (tmp_path / "triage.py").write_text(TRIAGE_PY, encoding="utf-8")
    (tmp_path / "emails.jsonl").write_text("".join(json.dumps(email) + "\n" for email in EMAILS), encoding="utf-8")
    by_prompt = {_prompt(email): json.dumps(ANSWERS[str(email["id"])]) for email in EMAILS}
    asyncio.run(
        record_model(
            tmp_path / "recordings" / "triage.jsonl",
            "deepseek/deepseek-v4-flash",
            list(by_prompt),
            by_prompt.__getitem__,
            schema=Triage,
            repeats=3,
        )
    )
    monkeypatch.delenv("RECORD", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_program_is_the_rung_as_written() -> None:
    lines = TRIAGE_PY.splitlines()
    assert (len(lines), sum(1 for line in lines if line.strip())) == (20, 15)


def test_the_command_line_runs_the_program_offline_prints_it_quietly_and_writes_its_report(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["triage.py"]) == 0

    printed = capsys.readouterr().out
    # The classifier is the first field, by its own name; every other grader is a pass rate of its own.
    assert "category: mean 0.8 (n=15" in printed
    assert "urgent: mean 1 (n=15" in printed
    assert "escalates: mean 0.8 (n=15" in printed
    # The lookalike's miss names both readings it failed, and the store and report are where the defaults put them.
    assert "case lookalike (repeat 1):" in printed
    assert "stored in .evals/evals.sqlite, scope 'emails'" in printed
    assert "report: .evals/reports/emails/report.html" in printed
    # Every call was the model's, so its spend is summed with no Answer: 15 calls at the recorded price.
    assert f"total spend: ${15 * CALL_COST_USD:.6f}, the results' cost" in printed
    assert (rung_one / ".evals" / "reports" / "emails" / "report.html").is_file()


def test_a_prompt_the_recording_does_not_hold_excludes_its_cells_rather_than_calling_a_model(
    rung_one: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    added = {"id": "new", "email": "Is there a student discount?", "category": "sales", "urgent": False}
    edited = [*EMAILS, {**added, "needs_human": False}]
    (rung_one / "emails.jsonl").write_text("".join(json.dumps(email) + "\n" for email in edited), encoding="utf-8")

    assert main(["triage.py"]) == 0

    printed = capsys.readouterr().out
    assert "excluded: 3 x " in printed and "Is there a student discount" in printed
