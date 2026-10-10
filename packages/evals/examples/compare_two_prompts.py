"""What does the new prompt change, and is the difference real or noise?

Two versions of a support-ticket classifier, differing only in their prompt, run over the same tickets, and the
engine tests the new one against the old one, the control. New here: ``compare``, which runs each candidate as
one arm and reports every arm against the control: the difference (delta), its interval, the Holm-adjusted p,
and a verdict. Then read where the arms disagree, through ``comparison.results(arm)``. The verdicts and what
each licenses: ``docs/reading-reports.md``.

Run it with ``python packages/evals/examples/compare_two_prompts.py``. With ``ANTHROPIC_API_KEY`` set it calls
Claude 48 times (12 tickets x 2 repeats x 2 prompts) for well under a cent; without it, keyword stand-ins play
the prompts and say nothing about Claude. The stand-ins are built to differ. Live, a current model may get every
ticket right under both prompts, and then "not separated" is the right answer: these tickets cannot tell the
prompts apart, and harder ones are needed. It never means the prompts are equally good.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from _live import claude, online
from threetears.evals.quick import Comparison, compare

MODEL = "claude-haiku-5-5"

# -----------------------------------------------------------------------------
# 1. The cases: each ticket, and the queue a person filed it under.
# -----------------------------------------------------------------------------

CASES = [
    {"ticket": "I was charged twice for my March invoice.", "queue": "billing"},
    {"ticket": "Can I get a refund for the annual plan I bought yesterday?", "queue": "billing"},
    {"ticket": "The export button does nothing when I click it in Firefox.", "queue": "bug"},
    {"ticket": "The app crashes every time I upload a photo larger than 10 MB.", "queue": "bug"},
    {"ticket": "I can't log in; the reset-password email never arrives.", "queue": "account"},
    {"ticket": "Please change the email address on my account to my work one.", "queue": "account"},
    {"ticket": "It would be great if reports could be scheduled weekly.", "queue": "feature_request"},
    {"ticket": "Do you plan to add a dark mode?", "queue": "feature_request"},
    # The hard ones: each mentions a second queue's words.
    {"ticket": "My card was charged but the upgrade page shows an error and nothing changed.", "queue": "billing"},
    {"ticket": "Since the update, the login page loops back to itself after I sign in.", "queue": "bug"},
    {"ticket": "I was charged for a seat after removing that user from my account.", "queue": "billing"},
    {"ticket": "Could you add an option to pay by invoice?", "queue": "feature_request"},
]

# -----------------------------------------------------------------------------
# 2. The two prompts under comparison.
# -----------------------------------------------------------------------------

BASELINE_PROMPT = (
    "Classify the support ticket into one of: billing, bug, account, feature_request. Answer with the label only."
)

CANDIDATE_PROMPT = """Classify the support ticket into exactly one queue.

- billing: charges, refunds, invoices, plans and payments.
- bug: the product does something wrong or fails — errors, crashes, broken pages.
- account: access to and details of the customer's own account, when nothing is broken.
- feature_request: something the product does not do yet.

When a ticket touches two queues, choose the one whose team must act first: money taken
without the service delivered is billing; a login that fails because the product misbehaves is a bug.

Answer with the label only."""

# -----------------------------------------------------------------------------
# 3. The live candidate: an async function from a case to a label.
# -----------------------------------------------------------------------------

Candidate = Callable[[Mapping[str, Any]], Awaitable[str]]


def claude_classifier(prompt: str) -> Candidate:
    """A candidate that asks Claude to classify each ticket under ``prompt``."""
    client = claude(MODEL)

    async def classify(case: Mapping[str, Any]) -> str:
        return (await client.generate(system=prompt, user=case["ticket"])).content.strip().lower()

    return classify


# -----------------------------------------------------------------------------
# 4. The OFFLINE stand-in: keyword rules, not a model.
#
# The baseline takes the first queue whose words appear; the candidate first applies its prompt's tie-break.
# -----------------------------------------------------------------------------

WORDS = {
    "account": ("log in", "login", "password", "email address", "account"),
    "bug": ("error", "crash", "does nothing", "loops", "broken"),
    "billing": ("charged", "refund", "invoice", "plan", "card"),
    "feature_request": ("would be great", "plan to add", "could you add", "could be"),
}


def offline_classifier(prompt: str) -> Candidate:
    """A keyword stand-in for one of the two prompts, so the example runs with no API key."""

    async def classify(case: Mapping[str, Any]) -> str:
        text = case["ticket"].lower()
        found = [queue for queue, words in WORDS.items() if any(word in text for word in words)]
        if prompt is CANDIDATE_PROMPT:
            if "feature_request" in found:
                return "feature_request"
            if "charged" in text:
                return "billing"  # money taken without the service delivered
            if "bug" in found:
                return "bug"  # a failure outranks the account words around it
        return found[0] if found else "bug"

    return classify


# -----------------------------------------------------------------------------
# 5. Run both prompts over every ticket twice, test the candidate against the baseline, print the verdict.
# -----------------------------------------------------------------------------


async def main() -> Comparison:
    make = claude_classifier if online() else offline_classifier
    if online():
        print(f"Running against Claude ({MODEL}).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a keyword stand-in for each prompt.\n")

    comparison = await compare(
        CASES,
        {"baseline": make(BASELINE_PROMPT), "candidate": make(CANDIDATE_PROMPT)},  # each arm by the name you give it
        expected=lambda case: case["queue"],  # a classifier eval: accuracy and a confusion matrix per arm
        control="baseline",  # the arm every other arm is tested against
        name="ticket triage: baseline vs candidate prompt" + ("" if online() else " (offline)"),  # titles the report
        scope_id="compare-two-prompts",
        k=2,  # repeats per case: a model's answer can change between calls
    )

    # The verdict: each arm against the control on each reading, row["arm"] being the name you gave it.
    for row in comparison.contrasts():
        p = (
            "" if row["p_adjusted"] is None else f", Holm-adjusted p {row['p_adjusted']:.2g}"
        )  # none when nothing varied
        print(f"{row['arm']} vs {comparison.control} on {row['reading']}: {row['control_mean']:.2f} -> ", end="")
        print(f"{row['arm_mean']:.2f}, delta {row['delta']:+.2f}, interval {row['interval']}{p}: {row['verdict']}")

    # Where the arms disagree: each arm's answer to a ticket, on every repeat.
    print("\nWhere the arms disagree:")
    for index, case in enumerate(CASES):
        said = {arm: [r.answer for r in comparison.results(arm) if r.case == str(index)] for arm in comparison.arms}
        if len({str(answers) for answers in said.values()}) > 1:
            print(f"  {case['ticket']!r}, filed under {case['queue']}:")
            print("    " + "; ".join(f"{arm} said {', '.join(answers)}" for arm, answers in said.items()))
    print("\nThe full report: print(comparison.render()), or reports.py to write it to files.")
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
