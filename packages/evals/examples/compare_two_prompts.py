"""What does the new prompt change, and is the difference real or noise?

Two versions of a support-ticket classifier, differing only in their prompt, run over the same tickets, and the
engine tests the new one against the old one, the control. New here: ``compare``, which runs each candidate as
one arm and reports every arm against the control: the difference (delta), its interval, the Holm-adjusted p,
and a verdict. Then read where the arms disagree, through ``comparison.results(arm)``. The verdicts and what
each licenses: ``docs/reading-reports.md``.

The tickets are a file beside this one, ``tickets.jsonl``, not a list in the script, so the next prompt version,
and every one after it, runs against exactly the same cases. When production gets a ticket wrong, add it there
(``docs/designing-classifier-evals.md``, section 8, "Maintaining the set"). When the suite needs versions and
history, store it as a case set with ``mint_case_set`` or the ``case_set_mint`` action (``docs/concepts.md``,
"Case set").

Run it with ``python packages/evals/examples/compare_two_prompts.py``. With ``ANTHROPIC_API_KEY`` set it calls
Claude 48 times (12 tickets x 2 repeats x 2 prompts) for well under a cent; without it, keyword stand-ins play
the prompts and say nothing about Claude. The stand-ins are built to differ. Live, a current model may get every
ticket right under both prompts, and then "not separated" is the right answer: these tickets cannot tell the
prompts apart, and harder ones are needed. It never means the prompts are equally good.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from _live import claude, online
from threetears.evals.quick import Comparison, compare

MODEL = "claude-haiku-5-5"

# -----------------------------------------------------------------------------
# 1. The cases: each ticket, and the queue a person filed it under, kept in a file beside this script.
#
# The test set is data, not code: the next prompt version runs against exactly these cases, and when production
# gets a ticket wrong, it becomes a new line here (docs/designing-classifier-evals.md, section 8, "Maintaining the
# set"). Each line's "kind" marks the hard ones, the lookalikes that mention a second queue's words; it is for you
# and the disagreement printout, never the candidate, which sees only the ticket. When the suite needs versions
# and history, store it as a case set (mint_case_set, or the case_set_mint action: docs/concepts.md, "Case set").
# -----------------------------------------------------------------------------

TICKETS = Path(__file__).with_name("tickets.jsonl")
CASES = [json.loads(line) for line in TICKETS.read_text().splitlines() if line.strip()]

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
            print(f"  {case['ticket']!r}, filed under {case['queue']} ({case['kind']}):")
            print("    " + "; ".join(f"{arm} said {', '.join(answers)}" for arm, answers in said.items()))
    print("\nThe full report: print(comparison.render()), or reports.py to write it to files.")
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
