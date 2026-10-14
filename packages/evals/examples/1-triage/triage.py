"""Rung 1: point it at your own function. Run ``python -m threetears.evals examples/1-triage/triage.py``.

The same eval as ``triage.toml`` (rung 0), with your own function as the runner and one grader TOML cannot say:
``escalates``, an asymmetric rule. Each email is triaged k=3 times; it replays ``recordings/triage.jsonl`` offline,
and ``RECORD=1`` with ``OPENROUTER_API_KEY`` set records it again. Recorded on deepseek/deepseek-v4-flash, as the
recordings' cost fields say: 90 calls, $0.0019 (``triage.toml``: 90 calls, $0.0020).
"""

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
    return await model.json(f"Triage this support email:\n\n{case['email']}", Triage)


def escalates(case: dict, out: Triage) -> bool:
    """Never leaves alone an email a person must see."""
    return out.needs_human or not case["needs_human"]


triage_eval = Eval("emails.jsonl", runner=triage, grader=["category", "urgent", escalates], k=3)
