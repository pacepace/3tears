"""Rung 3b: a margin, so a reading can come out equivalent: the one verdict that says "good enough".

compare.py's two prompts, with a margin on total: the largest difference too small to act on. Thirty invoices narrow
v2's interval on total to 19 points either side, so a 20-point margin is the tightest they can show it inside; a
tighter one needs more cases. Replays ``recordings/invoices.jsonl``, as compare.py does. As recorded on
deepseek/deepseek-v4-flash: 180 calls, $0.0105 (the recording's own cost_usd fields, summed).
"""

from invoice import V1, V2, Invoice

from threetears.evals import Eval, Model, grader

model = Model("deepseek/deepseek-v4-flash", max_tokens=8192, effort="low", recording="recordings/invoices.jsonl")

within_margin = Eval(
    "invoices.jsonl",
    runner={"v1": model.prompt(V1, Invoice), "v2": model.prompt(V2, Invoice)},
    question="Does v2 read the currency an invoice charges in right more often than v1?",
    # A difference on total under 20 points is too small to act on, so v2 can read "equivalent" there.
    grader=["currency", "number", "due", grader("total", margin=0.20)],
    name="margin",
)
