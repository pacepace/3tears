"""What does changing the prompt do on each model?

Two things vary at once, the prompt and the model: four arms, one per combination. New here: ``factors=``, which
keys each arm by its level of each thing varied (``(model, "v2")``), and ``Comparison.against``. Every arm is
tested against one control, (older model, v1), so the prompt's effect on the newer model is read against a
second control over the same runs, with nothing run again. Levers and arms: ``docs/concepts.md``.

Run it with ``python packages/evals/examples/prompt_x_model.py``. With ``ANTHROPIC_API_KEY`` set it calls
Claude 112 times (14 tickets x 2 repeats x 4 arms) for about two US cents. Without it, keyword stand-ins for an
older and a newer model play each combination and say nothing about Claude. The stand-ins are built to differ.
Live, current models may get every ticket right, and then "not separated" is the right answer: these tickets
cannot tell the arms apart, and harder ones are needed.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from _live import claude, online
from threetears.evals.quick import Comparison, compare

LIVE = ("claude-haiku-4-5", "claude-haiku-5-5")  # an older cheap model and a newer one
STAND_INS = ("older-stand-in", "newer-stand-in")

# -----------------------------------------------------------------------------
# 1. The cases: the tickets from compare_two_prompts.py, and two more hard ones.
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
    {"ticket": "My card was charged but the upgrade page shows an error and nothing changed.", "queue": "billing"},
    {"ticket": "Since the update, the login page loops back to itself after I sign in.", "queue": "bug"},
    {"ticket": "I was charged for a seat after removing that user from my account.", "queue": "billing"},
    {"ticket": "Could you add an option to pay by invoice?", "queue": "feature_request"},
    {"ticket": "Will you add a monthly plan for small teams?", "queue": "feature_request"},
    {"ticket": "I was charged again after my account was closed.", "queue": "billing"},
]

# -----------------------------------------------------------------------------
# 2. The two things varied: these two prompts, and two models.
# -----------------------------------------------------------------------------

PROMPTS = {
    "v1": "Classify the support ticket into one of: billing, bug, account, feature_request. "
    "Answer with the label only.",
    "v2": "Classify the support ticket into one of: billing, bug, account, feature_request. Money taken without "
    "the service delivered is billing; a login the product breaks is a bug; anything it does not do yet is a "
    "feature_request. Answer with the label only.",
}

# -----------------------------------------------------------------------------
# 3. One candidate per combination: an async function from a case to a label.
# -----------------------------------------------------------------------------

Candidate = Callable[[Mapping[str, Any]], Awaitable[str]]


def claude_classifier(model: str, prompt: str) -> Candidate:
    """A candidate that asks ``model`` to classify each ticket under prompt ``prompt``."""
    client = claude(model, max_tokens=256)

    async def classify(case: Mapping[str, Any]) -> str:
        return (await client.generate(system=PROMPTS[prompt], user=case["ticket"])).content.strip().lower()

    return classify


# -----------------------------------------------------------------------------
# 4. The OFFLINE stand-ins: keyword rules, not models.
#
# Under v1 the older model takes the first queue whose words appear; v2's tie-breaks fix that, and the
# newer model applies them unprompted but for one ticket.
# -----------------------------------------------------------------------------

FIRST_WORDS = {"account": ("log in", "login", "account", "email"), "bug": ("error", "crash", "does nothing")}
FIRST_WORDS["billing"] = ("charged", "refund", "invoice", "plan")
TIE_BREAKS = {" add ": "feature_request", "would be great": "feature_request", "charged": "billing", "loops": "bug"}


def offline_classifier(model: str, prompt: str) -> Candidate:
    """A keyword stand-in for one prompt on one model, so the example runs with no API key."""

    async def classify(case: Mapping[str, Any]) -> str:
        text = case["ticket"].lower()
        newer = model == "newer-stand-in"
        if newer and prompt == "v1" and "invoice" in text:
            return "billing"
        if newer or prompt == "v2":
            for word, queue in TIE_BREAKS.items():
                if word in text:
                    return queue
        return next((queue for queue, words in FIRST_WORDS.items() if any(w in text for w in words)), "feature_request")

    return classify


# -----------------------------------------------------------------------------
# 5. Run all four combinations over every ticket twice, then read v2 against v1 on each model.
# -----------------------------------------------------------------------------


async def main() -> Comparison:
    older, newer = LIVE if online() else STAND_INS
    make = claude_classifier if online() else offline_classifier
    if online():
        print(f"Running against Claude ({older}, {newer}).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with keyword stand-ins for each combination.\n")

    comparison = await compare(
        CASES,
        {(model, prompt): make(model, prompt) for model in (older, newer) for prompt in PROMPTS},
        factors=("model", "prompt"),  # each arm is keyed by its level of each factor, in this order
        control=(older, "v1"),
        expected=lambda case: case["queue"],
        name="prompt x model" + ("" if online() else " (offline)"),
        scope_id="prompt-x-model",
        k=2,
    )

    # Against (older, v1), v2 on the older model is the prompt's effect there. Its effect on the newer model needs
    # v1 on the newer model as the control: against() reads the same runs that way, running nothing again.
    for model, read in ((older, comparison), (newer, comparison.against((newer, "v1")))):
        (row,) = [row for row in read.contrasts("accuracy") if row["arm"] == (model, "v2")]  # each arm by its key
        p = "" if row["p_adjusted"] is None else f", Holm-adjusted p {row['p_adjusted']:.2g}"  # none if nothing varied
        print(f"On {model}, v2 vs v1 on {row['reading']}: {row['control_mean']:.2f} -> {row['arm_mean']:.2f}, ", end="")
        print(f"delta {row['delta']:+.2f}, interval {row['interval']}{p}: {row['verdict']}")
        for index, case in enumerate(CASES):  # where the two prompts disagree on this model
            said = [[r.answer for r in comparison.results((model, v)) if r.case == str(index)] for v in PROMPTS]
            if said[0] != said[1]:
                print(f"  {case['ticket']!r} ({case['queue']}): v1 said {', '.join(said[0])}; v2 {', '.join(said[1])}")
    print("\nThe full report: print(comparison.render()), or reports.py to write it to files.")
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
