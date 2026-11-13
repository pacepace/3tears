"""Rung 3c: a guardrail, decided on its own: something v2 must not do worse on, whatever else it gains.

compare.py's two prompts, with a check that v2 never states a total the invoice does not print. The guardrail is read
for v2 against the control as held, breached or undecided, apart from the contrasts; a trial that fails the check is
failed too, in the pass rate and in pass^k. Undecided is never read as held, so the gate fails and the door exits 4.
Replays ``recordings/invoices.jsonl``, as compare.py does. As recorded on deepseek/deepseek-v4-flash:
180 calls, $0.0105 (the recording's own cost_usd fields, summed).
"""

from invoice import V1, V2, Invoice

from threetears.evals import Eval, Model, grader

model = Model("deepseek/deepseek-v4-flash", max_tokens=8192, effort="low", recording="recordings/invoices.jsonl")


@grader(guard=0.02)  # a guardrail: v2 must not do worse here by more than 2 points, whatever else it gains
def no_invented_total(case: dict, out: Invoice) -> bool:
    """Never states a total the invoice does not print."""
    return case["total"] is not None or out.total is None


invoices = Eval(
    "invoices.jsonl",
    runner={"v1": model.prompt(V1, Invoice), "v2": model.prompt(V2, Invoice)},
    question="Does v2 read the currency an invoice charges in right more often than v1?",
    grader=["currency", "number", "due", "total", no_invented_total],  # compare.py's fields, then the guardrail
)
