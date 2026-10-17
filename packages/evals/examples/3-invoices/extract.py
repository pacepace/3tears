"""Rung 3: changing a prompt? Compare it against the current one, and get a report you can hand to a reviewer.

Two prompts read 30 invoices (clean, multi-currency, noisy OCR); the first is the control.
``python -m threetears.evals examples/3-invoices/extract.py`` replays ``recordings/invoices.jsonl`` offline, and
``RECORD=1`` with ``OPENROUTER_API_KEY`` set records it again; the block at the end is that run as your own script.
As recorded on deepseek/deepseek-v4-flash: 180 calls, $0.0105 (the recording's own cost_usd fields, summed).

Two levers at once? Name the arms as k=v pairs ("prompt=v2, effort=high") and each pair is a factor of the design;
``await result.against("prompt=v1, effort=high")`` reads the same runs against another control, with no run repeated.
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from threetears.evals import Eval, Model, grader

model = Model("deepseek/deepseek-v4-flash", max_tokens=8192, effort="low", recording="recordings/invoices.jsonl")


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


@grader(guard=0.02)  # a guardrail: v2 must not do worse here by more than 2 points, whatever else it gains
def no_invented_total(case: dict, out: Invoice) -> bool:
    """Never states a total the invoice does not print."""
    return case["total"] is not None or out.total is None


invoices = Eval(
    "invoices.jsonl",  # each case's stratum splits every reading by kind of invoice in the report
    runner={"v1": model.prompt(V1, Invoice), "v2": model.prompt(V2, Invoice)},  # the first key is the control
    # A field each; a case's null total means none may be stated. A difference under 5 points on total is too small
    # to act on, so v2 can read "equivalent" there.
    grader=["currency", "number", "due", grader("total", margin=0.05), no_invented_total],
)

if __name__ == "__main__":
    result = asyncio.run(invoices.run())
    print(result)  # the verdict line, then each arm's readings; every delta and its interval is in the report
    result.write("report")  # report.html for the reviewer, beside report.md and bundle.json
    raise SystemExit(4 if result.gate().failed else 0)  # in CI: a regression or a guardrail not shown held fails
