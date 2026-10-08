"""Compare two prompts for a support-ticket classifier, and ask the engine which one is better.

This is the step after ``rung_zero.py``. There, one function was measured; here two versions of
one classifier are measured over the same cases, and the engine says whether the second does
better than the first or whether the difference is noise.

The classifier sorts a customer's support ticket into one of four queues. The two versions
differ only in their prompt:

- ``baseline`` is the prompt you have today: a bare list of the queue names.
- ``candidate`` is the prompt you want to ship: each queue defined, with the rule for the cases
  that look like two queues at once.

Each version is called on every ticket a few times, graded against the queue a person filed it
under, and the two are tested against each other with ``baseline`` as the control.

Run it with ``python packages/evals/examples/compare_two_prompts.py``. With ``ANTHROPIC_API_KEY``
set, every answer comes from Claude (``claude-haiku-5-5``). Without it, the example runs OFFLINE: a
keyword stand-in plays each prompt, so the script always runs, and the output says so. The
stand-in is only there to show the shape of the result; its numbers say nothing about Claude.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from threetears.evals.quick import Comparison, compare

# -----------------------------------------------------------------------------
# 1. The cases: each ticket, and the queue a person filed it under.
#
# A case is any JSON object. The candidate is handed the whole case; the engine
# reads the expected label through ``expected=`` below, so the key names are yours.
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

QUEUES = ("billing", "bug", "account", "feature_request")

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
# 3. How one prompt becomes a candidate: an async function from a case to a label.
#
# The engine calls the candidate once per case and repeat. Whatever string it
# returns is the predicted label; anything that is not a label counts as an
# unusable answer, which is never a match.
# -----------------------------------------------------------------------------

Candidate = Callable[[Mapping[str, Any]], Awaitable[str]]


def claude_classifier(prompt: str) -> Candidate:
    """A candidate that asks Claude to classify each ticket under ``prompt``."""
    import anthropic  # imported here so the offline path does not need the package

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY

    async def classify(case: Mapping[str, Any]) -> str:
        response = await client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=1024,
            output_config={"effort": "low"},  # a four-way label needs little thought
            system=prompt,
            messages=[{"role": "user", "content": case["ticket"]}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text.strip().lower()

    return classify


# The OFFLINE stand-ins: keyword rules that play each prompt when there is no API key.
# The baseline takes the first queue whose words appear, in this order; the candidate
# first applies the tie-break rule its prompt states. Nothing here is a model.
_WORDS = {
    "account": ("log in", "login", "password", "email address", "account"),
    "bug": ("error", "crash", "does nothing", "loops", "broken"),
    "billing": ("charged", "refund", "invoice", "plan", "card"),
    "feature_request": ("would be great", "plan to add", "could you add", "could be"),
}


def offline_classifier(prompt: str) -> Candidate:
    """A keyword stand-in for one of the two prompts, so the example runs with no API key."""

    def hits(text: str, queue: str) -> bool:
        return any(word in text for word in _WORDS[queue])

    async def classify(case: Mapping[str, Any]) -> str:
        text = case["ticket"].lower()
        if prompt is CANDIDATE_PROMPT:
            if hits(text, "feature_request"):
                return "feature_request"
            if hits(text, "billing") and "charged" in text:
                return "billing"  # money taken without the service delivered
            if hits(text, "bug"):
                return "bug"  # a failure outranks the account words around it
        for queue in _WORDS:  # the baseline: whichever queue's words come first
            if hits(text, queue):
                return queue
        return "bug"

    return classify


# -----------------------------------------------------------------------------
# 4. Run both prompts, test the candidate against the baseline, print the verdict.
# -----------------------------------------------------------------------------


async def main() -> Comparison:
    """Compare the two prompts over every ticket, twice each, and print the campaign's report."""
    online = bool(os.environ.get("ANTHROPIC_API_KEY"))
    make = claude_classifier if online else offline_classifier
    if online:
        print("Running against Claude (claude-haiku-5-5).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a keyword stand-in for each prompt.\n")

    comparison = await compare(
        CASES,
        # Each arm is named; the name is how the report refers to it (``model=<name>``).
        {"baseline": make(BASELINE_PROMPT), "candidate": make(CANDIDATE_PROMPT)},
        # Handing over each case's expected label makes this a classifier eval:
        # accuracy, a confusion matrix, and per-label precision/recall/F1.
        expected=lambda case: case["queue"],
        # The control is the arm every other arm is tested against.
        control="baseline",
        name="ticket triage: baseline vs candidate prompt" + ("" if online else " (offline)"),
        # Where the runs and the campaign are stored. The default store is in memory.
        scope_id="compare-two-prompts",
        # Repeats per case. A model's answer can change between calls; repeats show by how much.
        k=2,
    )

    # Each arm's own summary: accuracy and the confusion matrix, as rung zero prints it.
    for arm, summary in comparison.arms.items():
        print(f"--- {arm} ---")
        print(summary.render())
        print()

    # The campaign's report. Its "Contrasts against the control" table is the verdict:
    # for each reading (accuracy here), the candidate's difference from the baseline,
    # the p-value after correcting for every comparison made, and whether that difference
    # separated from the control or is within what chance produces.
    #
    # A note on cost: the report lists a ``cost_usd`` reading, and it is 0 for both arms.
    # The candidate here is a plain function, and the engine does not see what it spends
    # inside — so the two prompts compare on accuracy here, not on price.
    print(comparison.render())
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
