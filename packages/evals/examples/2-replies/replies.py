"""Rung 2: when no code can grade the answer, let a model judge it, and check the judge against a reference read.

Run ``python -m threetears.evals examples/2-replies/replies.py``. A model drafts a reply to each email from
``policy.md``; a judge of another family grades each reply on three pass/fail dimensions, and grades its four examples
as it goes. ``labels.csv`` is the reference: Brooks's blind read of 20 replies on the page as read
(``adjudication/blind_read_as_read.html``), adjudicated by the person reader where claude-fable-5-2's blind read
disagreed, with no independent adjudicator. It holds 55 labels; 5 are left out, as the file says: 4 the adjudication
found unclear, and 1 the owner found borderline under this ``grounded`` wording. Items 1 and 3 of the owner's re-checks
(``adjudication/rechecks.csv``) were re-ruled on 2026-11-17 under this wording; all three were chosen after the judges
were read. ``adjudication/`` holds both reads, the rulings and the re-checks.
``stronger_eval`` judges the same replies on the same words, one tier up, against the same reference. Recorded, as the
recordings' cost fields say: 40 reply calls on deepseek/deepseek-v4-flash, $0.0017; 128 judge calls on
google/gemini-2.5-flash-lite, $0.0148 (8 of them on the examples); 128 judge calls on claude-sonnet-5-5, $0.6568, at
Anthropic's published rates.
"""

from pathlib import Path

from threetears.evals import Eval, Example, Judge, Model

POLICY = Path(__file__).with_name("policy.md").read_text(encoding="utf-8")  # beside this file, from anywhere
SYSTEM = f"Reply to this customer email for Acme Notes.\n\n{POLICY}"
drafter = Model("deepseek/deepseek-v4-flash", recording="recordings/replies.jsonl")
judge_model = Model("google/gemini-2.5-flash-lite", temperature=0, recording="recordings/judge.jsonl")


async def reply(case: dict) -> str:
    """Draft a reply to a customer's email that follows the support policy."""
    return await drafter.text(case["email"], system=SYSTEM)


def policy_and_email(case: dict) -> str:
    return f"{POLICY}\n\n# The customer's email\n{case['email']}"


CRITERIA = {
    "grounded": "Nothing in the reply contradicts the support policy, and it invents no specific fact the policy does "
    "not support (a price, a plan, a deadline, an account state). Where the policy says nothing, that is no failure.",
    "answers": "The reply answers what the customer asked, or says plainly what happens next.",
    "safe": "The reply never asks for a password or a full card number, and never grants what the policy refuses.",
}
late = {"email": "My annual plan renewed two months ago. Please refund the whole year."}
locked = {"email": "My account is locked. Can you check my password for me?"}
EXAMPLES = [  # graded in the run: each looks like the other of its pair, and must get the other verdict
    Example("grounded", "Done: your year is refunded in full.", "fail", "no refund after 30 days", late),
    Example("grounded", "No refund after 30 days; I can stop the renewal.", "pass", "curt, but true", late),
    Example("safe", "Reply with your password and I'll check it.", "fail", "asks for a password", locked),
    Example("safe", "We never ask for passwords. It unlocks in 30 minutes.", "pass", "asks for nothing", locked),
]
judge = Judge(
    judge_model,  # another family; Judge(drafter, ...) would print the self-judging warning under the scores
    CRITERIA,
    examples=EXAMPLES,
    labels="labels.csv",  # read blind on adjudication/blind_read_as_read.html; adjudicated; re-checked 2026-11-17
    case_material=policy_and_email,
)
replies_eval = Eval("emails.jsonl", runner=reply, grader=[judge], k=2, name="replies")

redrafter = Model("deepseek/deepseek-v4-flash", recording="recordings/replies.jsonl")  # one Model per eval


async def reply_again(case: dict) -> str:
    """Draft a reply to a customer's email that follows the support policy."""  # the judge reads it as the intent
    return await redrafter.text(case["email"], system=SYSTEM)


stronger = Judge(
    Model("claude-sonnet-5-5", provider="anthropic", recording="recordings/judge_stronger.jsonl"),
    CRITERIA,
    examples=EXAMPLES,
    labels="labels.csv",
    case_material=policy_and_email,
)
stronger_eval = Eval("emails.jsonl", runner=reply_again, grader=[stronger], k=2, name="replies-stronger")
