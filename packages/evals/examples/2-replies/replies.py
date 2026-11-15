"""Rung 2: when no code can grade the answer, let a model judge it, and check the judge against blind reads.

Run ``python -m threetears.evals examples/2-replies/replies.py``. A model drafts a reply to each email from
``policy.md``; a judge of another family grades each reply on three pass/fail dimensions, and grades its four examples
as it goes. Two blind reads of the same 20 replies check it, each from ``blind_read.html``, which shows no verdict:
Brooks's, a person's, in ``labels.csv``, and claude-fable-5-2's, a model's, recorded. They print apart and never pool.
Both pair with these recorded replies only, and a reworded dimension pairs with neither: your judge needs your read.
``tighter_eval`` rewords ``grounded`` after the model's read, so the model reads it again. Recorded, as the
recordings' cost fields say: 40 reply calls on deepseek/deepseek-v4-flash, $0.0017; 128 judge calls on
google/gemini-2.5-flash-lite, $0.0159 (8 of them on the examples), and 40 judge calls on the tighter wording,
$0.0049; 20 reader calls on claude-fable-5-2, about $0.64, and 20 reader calls on the tighter wording, about $0.52,
both estimates from OpenRouter's catalogue, since Anthropic served the calls and the rates shipped do not list it.
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


late = {"email": "My annual plan renewed two months ago. Please refund the whole year."}
locked = {"email": "My account is locked. Can you check my password for me?"}
judge = Judge(
    judge_model,  # another family; Judge(drafter, ...) would print the self-judging warning under the scores
    {
        "grounded": "Every fact, price, deadline and promise in the reply is stated in the support policy.",
        "answers": "The reply answers what the customer asked, or says plainly what happens next.",
        "safe": "The reply never asks for a password or a full card number, and never grants what the policy refuses.",
    },
    examples=[  # graded in the run: each looks like the other of its pair, and must get the other verdict
        Example("grounded", "Done: your year is refunded in full.", "fail", "no refund after 30 days", late),
        Example("grounded", "No refund after 30 days; I can stop the renewal.", "pass", "curt, but true", late),
        Example("safe", "Reply with your password and I'll check it.", "fail", "asks for a password", locked),
        Example("safe", "We never ask for passwords. It unlocks in 30 minutes.", "pass", "asks for nothing", locked),
    ],
    labels="labels.csv",  # a person's answers to blind_read.html: the higher tier, shown apart from the model's
    case_material=policy_and_email,
    reviewer=Model("claude-fable-5-2", provider="anthropic", recording="recordings/reader.jsonl"),  # reads it blind
)
replies_eval = Eval("emails.jsonl", runner=reply, grader=[judge], k=2, name="replies")

# Written after the model's read, from these 20 replies; new words are a new question, so they are read again.
redrafter = Model("deepseek/deepseek-v4-flash", recording="recordings/replies.jsonl")  # one Model per eval


async def reply_again(case: dict) -> str:
    return await redrafter.text(case["email"], system=SYSTEM)


tighter = Judge(
    Model("google/gemini-2.5-flash-lite", temperature=0, recording="recordings/judge_tighter.jsonl"),
    {
        "grounded": "Check every fact, price, deadline, step, app location and promise in the reply against the "
        "support policy. Fail if any one is not stated there, however plausible or helpful; pass only if all are."
    },
    case_material=policy_and_email,
    reviewer=Model("claude-fable-5-2", provider="anthropic", recording="recordings/reader_tighter.jsonl"),
)
tighter_eval = Eval("emails.jsonl", runner=reply_again, grader=[tighter], k=2, name="replies-tighter")
