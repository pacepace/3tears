"""Two factors at once: two prompts on two models. Does the better prompt help on both?

``compare_two_prompts.py`` varied one thing. Here two vary, prompt (``v1``, ``v2``) and model (the older
Haiku, the newer), so there are four arms, one per combination. Each factor is a *lever*, a setting every
run records (``docs/concepts.md``), so the report names each arm by both: ``callable.prompt=v2, model=...``.

Run it with ``python packages/evals/examples/prompt_x_model.py``. With ``ANTHROPIC_API_KEY`` set it calls
Claude: 4 arms x 14 tickets x 2 repeats, 112 short calls, about two US cents. Without the key it runs
OFFLINE on keyword stand-ins, so it always runs, but those numbers say nothing about Claude.

The last table answers the question: v2 against v1 on each model. The engine tests arms against one
control (v1 on the older model), so the newer model's row comes from ``comparison.against(...)``, which
reads the same runs against v1 on the newer model without running anything again.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from threetears.evals.quick import Comparison, compare

# 1. The cases: the tickets from compare_two_prompts.py, and two more hard ones. -----------------------

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

# 2. The two factors: two prompts, and two cheap models ($1/$5 and $0.10/$0.50 per million tokens). ----

PROMPTS = {
    "v1": "Classify the support ticket into one of: billing, bug, account, feature_request. Answer with the label only.",
    "v2": "Classify the support ticket into one of: billing, bug, account, feature_request. Money taken without "
    "the service delivered is billing; a login the product breaks is a bug; anything it does not do yet is a "
    "feature_request. Answer with the label only.",
}
OLD, NEW = "claude-haiku-4-5", "claude-haiku-5-5"

# 3. One candidate per combination: an async function from a case to a label. -------------------------

Candidate = Callable[[Mapping[str, Any]], Awaitable[str]]


def claude_classifier(model: str, prompt: str) -> Candidate:
    """Ask ``model`` to classify each ticket under ``prompt``."""
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


# The OFFLINE stand-ins: keyword rules, not models. Under v1 the older model takes the first queue whose
# words appear. v2 states tie-break rules that fix that; the newer model follows them unprompted, except
# that under v1 it still files "pay by invoice" as billing.
FIRST_WORDS = {"account": ("log in", "login", "account", "email"), "bug": ("error", "crash", "does nothing")}
FIRST_WORDS["billing"] = ("charged", "refund", "invoice", "plan")
TIE_BREAKS = {" add ": "feature_request", "would be great": "feature_request", "charged": "billing", "loops": "bug"}


def offline_classifier(model: str, prompt: str) -> Candidate:
    """A keyword stand-in for one combination, so the example runs with no API key."""

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


# 4. Run the 2x2, then read the prompt's effect on each model. ----------------------------------------


async def main() -> Comparison:
    online = bool(os.environ.get("ANTHROPIC_API_KEY"))
    make = claude_classifier if online else offline_classifier
    print(
        "Running against Claude." if online else "No ANTHROPIC_API_KEY: OFFLINE stand-ins, saying nothing about Claude."
    )

    comparison = await compare(
        CASES,
        # Each arm is keyed by its level of each factor, in the order factors= names them.
        {(model, prompt): make(model, prompt) for model in (OLD, NEW) for prompt in PROMPTS},
        factors=("model", "prompt"),
        control=(OLD, "v1"),
        expected=lambda case: case["queue"],
        name="prompt x model" + ("" if online else " (offline)"),
        scope_id="prompt-x-model",
        k=2,
    )

    print(f"\nAccuracy{OLD:>20}{NEW:>20}")
    for prompt in PROMPTS:
        means = [{m.name: m.mean for m in comparison.arms[(model, prompt)].measures}["match"] for model in (OLD, NEW)]
        print(f"{prompt:8}" + "".join(f"{mean:>20.2f}" for mean in means))

    # The answer: v2 against v1 on each model, each Holm-corrected within its own campaign.
    print("\nDoes v2 beat v1?")
    for model, read in ((OLD, comparison), (NEW, comparison.against((NEW, "v1")))):
        row = next(r for r in read.contrasts("accuracy") if r["contrast"] == f"callable.prompt=v2, model={model}")
        p = "-" if row["p_adjusted"] is None else f"{row['p_adjusted']:.3f}"
        print(f"  on {model}: accuracy {row['delta']:+.2f}, p={p}, {row['verdict']}")
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
