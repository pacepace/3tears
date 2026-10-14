"""Rung 3: changing a prompt? Compare it against the current one, and get a report you can hand to a reviewer.

Two prompts read 30 invoices (clean, multi-currency, noisy OCR); the first is the control. From this directory,
``python extract.py`` (or ``python -m threetears.evals extract.py``) replays ``recordings/invoices.jsonl`` offline,
and ``RECORD=1`` with ``OPENROUTER_API_KEY`` set records it again. As recorded on deepseek/deepseek-v4-flash:
180 calls, $0.0103 (the recording's own cost_usd fields, summed).
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from threetears.evals import Eval, Model, grader

model = Model("deepseek/deepseek-v4-flash", max_tokens=4096, effort="low", recording="recordings/invoices.jsonl")


class Invoice(BaseModel):
    number: str
    currency: Literal["USD", "EUR", "GBP", "JPY", "CHF", "CAD"]
    due: str
    total: float | None


V1 = "Extract the invoice number, currency, due date (YYYY-MM-DD) and total due from this invoice:\n\n{text}"
V2 = V1 + (
    "\n\nThe currency is the one the total is charged in, never one quoted for reference. Read dates in the "
    "issuer's own order (09/03 is 9 March in Europe, 3 September in the US). Undo OCR slips (O for 0, l for 1). "
    "Leave total null unless the invoice prints the amount due: never add up the lines yourself."
)


@grader(margin=0.05)  # a difference under 5 points is too small to act on, so v2 can read "equivalent"
def total(case: dict, out: Invoice) -> bool:
    """The amount due to the cent, or none when the invoice prints none."""
    if case["total"] is None or out.total is None:
        return case["total"] == out.total
    return abs(out.total - case["total"]) < 0.005


@grader(guard=0.02)  # a guardrail: v2 must not do worse here by more than 2 points, whatever else it gains
def no_invented_total(case: dict, out: Invoice) -> bool:
    """Never states a total the invoice does not print."""
    return case["total"] is not None or out.total is None


invoices = Eval(
    "invoices.jsonl",  # each case's stratum splits every reading by kind of invoice in the report
    runner={"v1": model.prompt(V1, Invoice), "v2": model.prompt(V2, Invoice)},  # the first key is the control
    grader=["currency", "number", "due", total, no_invented_total],  # a field each, then two functions
)

if __name__ == "__main__":
    result = asyncio.run(invoices.run())
    print(result)  # v2 against v1 on every reading: the delta, its interval and a verdict
    result.write("report")  # report.html for the reviewer, beside report.md and bundle.json
    raise SystemExit(4 if result.gate().failed else 0)  # in CI: a regression or a guardrail not shown held fails

# Two levers at once? Name the arms as k=v pairs ({"prompt=v1, effort=low": ..., "prompt=v2, effort=high": ...}) and
# each pair is a factor of the design; `await result.against("prompt=v1, effort=high")` then reads the same runs
# against another control, with no run repeated.
