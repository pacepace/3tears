"""What does the new prompt change, and is the difference real or noise?

Two versions of a support-ticket classifier, differing only in their prompt, run over the same tickets,
and the engine tests the new one against the old one, the control, and gives a verdict. New here:
``compare``, which runs each candidate as one arm and reports every arm against the control.

Run it with ``python packages/evals/examples/compare_two_prompts.py``. With ``ANTHROPIC_API_KEY`` set it calls
Claude 48 times for well under a cent; without it, keyword stand-ins play the prompts and say nothing about Claude.
The stand-ins are built to differ; live, a current model may already get every ticket right, and then "not
separated" is the correct answer: these cases cannot tell the prompts apart, and harder ones are needed.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

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
    import anthropic  # imported here so the offline path does not need the package

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY

    async def classify(case: Mapping[str, Any]) -> str:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            output_config={"effort": "low"},  # a four-way label needs little thought
            system=prompt,
            messages=[{"role": "user", "content": case["ticket"]}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip().lower()

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
    online = bool(os.environ.get("ANTHROPIC_API_KEY"))
    make = claude_classifier if online else offline_classifier
    if online:
        print(f"Running against Claude ({MODEL}).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a keyword stand-in for each prompt.\n")

    comparison = await compare(
        CASES,
        # Each arm by name; the report calls it model=<name>.
        {"baseline": make(BASELINE_PROMPT), "candidate": make(CANDIDATE_PROMPT)},
        expected=lambda case: case["queue"],  # a classifier eval: accuracy and a confusion matrix
        control="baseline",  # the arm every other arm is tested against
        name="ticket triage: baseline vs candidate prompt" + ("" if online else " (offline)"),
        scope_id="compare-two-prompts",
        k=2,  # repeats per case: a model's answer can change between calls
    )

    for arm, summary in comparison.arms.items():
        print(f"--- {arm} ---\n{summary.render()}\n")

    # The verdict: each arm against the control, the difference, its Holm-adjusted p, and whether it separated.
    for row in comparison.contrasts():
        arm, control = (row[key].removeprefix("model=") for key in ("contrast", "control"))
        p = "" if row["p_adjusted"] is None else f" (p={row['p_adjusted']:.2g})"  # none when nothing varied
        print(f"{arm} vs {control} on {row['reading']}: {row['delta']:+.2g}{p}: {row['verdict']}")
    print("\nThe full report: comparison.render(), or reports.py to write it to files.")
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
