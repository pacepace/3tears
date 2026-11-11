"""Rung 3b: a margin, so a reading can come out equivalent: the one verdict that says "good enough".

compare.py's two prompts, with a margin on total, fixed before any reading from what the decision is for: a person
checks every total before it is paid, so a prompt worse on total by under 20 points would still ship on its other
fields. Thirty invoices can show equivalence only at a margin this wide; a tighter one needs more cases. Replays
``recordings/invoices.jsonl``, as compare.py does. As recorded on deepseek/deepseek-v4-flash: 180 calls, $0.0105
(the recording's own cost_usd fields, summed).
"""

from invoice import V1, V2, Invoice

from threetears.evals import Eval, Model, grader

model = Model("deepseek/deepseek-v4-flash", max_tokens=8192, effort="low", recording="recordings/invoices.jsonl")

within_margin = Eval(
    "invoices.jsonl",
    runner={"v1": model.prompt(V1, Invoice), "v2": model.prompt(V2, Invoice)},
    question="Does v2 read the currency an invoice charges in right more often than v1?",
    # Fixed before the run: totals are checked by hand, so a gap under 20 points would not change which prompt ships.
    grader=["currency", "number", "due", grader("total", margin=0.20)],
    name="margin",
)
