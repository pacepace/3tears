"""Does the better prompt help on both models, or only on one?

Two factors vary at once, the prompt and the model: four arms, one per combination, each named in the report
by both (``callable.prompt=v2, model=...``). Arms are tested against one control, so the prompt's effect on the
second model is read against a second control over the same runs. New here: ``factors=``, which keys each arm
by its level of each factor, and ``Comparison.against``. Levers and arms: ``docs/concepts.md``.

Run it with ``python packages/evals/examples/prompt_x_model.py``. With ``ANTHROPIC_API_KEY`` set it calls
Claude 112 times for about two US cents; without it, keyword stand-ins play each combination and say nothing
about Claude.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from threetears.evals.quick import Comparison, compare

MODELS = ("claude-haiku-4-5", "claude-haiku-5-5")
OLD, NEW = MODELS  # the older cheap model and the newer one

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
# 2. The two factors: these two prompts, and the two models in MODELS.
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
    import anthropic  # imported here so the offline path does not need the package

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    effort: dict[str, Any] = {"output_config": {"effort": "low"}} if model == NEW else {}  # Haiku 4.5 has no effort

    async def classify(case: Mapping[str, Any]) -> str:
        response = await client.messages.create(
            model=model,
            max_tokens=256,
            system=PROMPTS[prompt],
            messages=[{"role": "user", "content": case["ticket"]}],
            **effort,
        )
        return "".join(block.text for block in response.content if block.type == "text").strip().lower()

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
        if model == NEW and prompt == "v1" and "invoice" in text:
            return "billing"
        if model == NEW or prompt == "v2":
            for word, queue in TIE_BREAKS.items():
                if word in text:
                    return queue
        return next((queue for queue, words in FIRST_WORDS.items() if any(w in text for w in words)), "feature_request")

    return classify


# -----------------------------------------------------------------------------
# 5. Run all four combinations over every ticket twice, then read v2 against v1 on each model.
# -----------------------------------------------------------------------------


async def main() -> Comparison:
    online = bool(os.environ.get("ANTHROPIC_API_KEY"))
    make = claude_classifier if online else offline_classifier
    if online:
        print(f"Running against Claude ({OLD}, {NEW}).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with keyword stand-ins for each combination.\n")

    comparison = await compare(
        CASES,
        {(model, prompt): make(model, prompt) for model in MODELS for prompt in PROMPTS},
        factors=("model", "prompt"),  # each arm is keyed by its level of each factor, in this order
        control=(OLD, "v1"),
        expected=lambda case: case["queue"],
        name="prompt x model" + ("" if online else " (offline)"),
        scope_id="prompt-x-model",
        k=2,
    )
    print(comparison.render())

    # Against (OLD, v1), v2 on OLD is the prompt's effect there. Its effect on NEW needs v1 on NEW as the
    # control: against() reads the same runs that way, running nothing again.
    print("\nDoes v2 beat v1?")
    for model, read in ((OLD, comparison), (NEW, comparison.against((NEW, "v1")))):
        row = next(r for r in read.contrasts("accuracy") if r["contrast"] == f"callable.prompt=v2, model={model}")
        print(f"  on {model}: accuracy {row['delta']:+.2f}, {row['verdict']}")
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
