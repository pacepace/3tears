"""Rung 3a: changing a prompt? Compare it against the current one, and read the verdict.

Two prompts read 30 invoices (clean, multi-currency, noisy OCR); the first is the control.
``python -m threetears.evals examples/3-invoices/compare.py`` replays ``recordings/invoices.jsonl`` offline, and
``RECORD=1`` with ``OPENROUTER_API_KEY`` set records it again. As recorded on deepseek/deepseek-v4-flash:
180 calls, $0.0105 (the recording's own cost_usd fields, summed).
"""

from invoice import V1, V2, Invoice

from threetears.evals import Eval, Model

model = Model("deepseek/deepseek-v4-flash", max_tokens=8192, effort="low", recording="recordings/invoices.jsonl")

compare = Eval(
    "invoices.jsonl",  # each case's stratum splits every reading by kind of invoice in the report
    runner={"v1": model.prompt(V1, Invoice), "v2": model.prompt(V2, Invoice)},  # the first key is the control
    # What the comparison answers; the first grader (currency) is its measure, which the verdict and the page lead on.
    question="Does v2 read the currency an invoice charges in right more often than v1?",
    grader=["currency", "number", "due", "total"],  # a field each; a case's null total means none may be stated
    name="compare",
)
