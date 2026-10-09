"""Run one prompt on two Claude models, and weigh what each gets right against what it costs.

``compare_two_prompts.py`` changed the prompt and kept the model. This keeps the prompt and changes
the model: is the cheaper model good enough? To answer that, each candidate returns an ``Answer``,
its label plus the tokens and dollars the call spent, and the engine records that spend. Each arm's
summary then prints a ``candidate spend`` line, and the report tests the arms' ``cost_usd`` against
the control the same way it tests their accuracy.

The task is labelling a short email ``phishing``, ``spam`` or ``legit``. The control is
``claude-haiku-4-5`` ($1 / $5 per million input / output tokens); the other arm is
``claude-haiku-5-5`` ($0.10 / $0.50).

Run it with ``python packages/evals/examples/compare_two_models.py``. With ``ANTHROPIC_API_KEY`` set,
it makes 48 short calls (12 emails, 2 repeats, 2 models), which cost well under a cent. Without it,
the example runs OFFLINE: a keyword stand-in plays each model and reports made-up token counts, so
the script always runs and the output says so. Its numbers say nothing about Claude.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from threetears.evals.quick import Answer, Comparison, compare

# --- 1. The cases: each email, and the label a person gave it --------------------------------------

CASES = [
    {"email": "Account suspended: verify your password within 24 hours at paypa1-secure.com.", "label": "phishing"},
    {"email": "IT department: your mailbox is full. Log in here to keep receiving email.", "label": "phishing"},
    {"email": "Invoice overdue. Please wire the payment to our new bank account today. - the CEO", "label": "phishing"},
    {"email": "A parcel could not be delivered. Pay the $1.99 redelivery fee at this link.", "label": "phishing"},
    {"email": "Congratulations! You've won a $1,000 gift card. Click to claim your prize.", "label": "spam"},
    {"email": "Limited time: 70% off designer watches, today only!!!", "label": "spam"},
    {"email": "Earn $5,000 a week from home, no experience needed.", "label": "spam"},
    {"email": "Hi Sam, attaching the slides for Thursday's review. Shout if the numbers look off.", "label": "legit"},
    {"email": "Your order #44821 has shipped and should arrive on Tuesday.", "label": "legit"},
    {"email": "Reminder: your dentist appointment is tomorrow at 3pm. Reply C to confirm.", "label": "legit"},
    {"email": "We noticed a sign-in from a new device. If this was you, no action is needed.", "label": "legit"},
    {"email": "Your free trial ends in 3 days. You can manage your subscription in the app.", "label": "legit"},
]

# --- 2. The one prompt, the two models, and what each charges --------------------------------------

PROMPT = """Label the email as exactly one of:
- phishing: it tries to get credentials, payment or a transfer by pretending to be someone it is not.
- spam: unsolicited advertising or a too-good-to-be-true offer that asks for nothing sensitive.
- legit: anything else, including real notices about the reader's own accounts and orders.
Answer with the label only."""

OLDER, NEWER = "claude-haiku-4-5", "claude-haiku-5-5"

# Anthropic's list prices, (input, output) USD per million tokens; check them before you trust the dollars.
RATES_PER_MILLION = {OLDER: (1.00, 5.00), NEWER: (0.10, 0.50)}


def priced(label: str, model: str, input_tokens: int, output_tokens: int) -> Answer:
    """The label, with the call's tokens and what they cost at the model's list price."""
    input_rate, output_rate = RATES_PER_MILLION[model]
    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1e6
    return Answer(label, model=model, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)


# --- 3. A candidate per model: an async function from a case to a priced Answer --------------------

Candidate = Callable[[Mapping[str, Any]], Awaitable[Answer]]


def claude_classifier(model: str) -> Candidate:
    """A candidate that asks ``model`` to label each email, priced from the usage the API reports."""
    import anthropic  # imported here so the offline path does not need the package

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    # Haiku 4.5 takes no effort setting; a low effort keeps Haiku 5.5's thinking short.
    extras: dict[str, Any] = {"output_config": {"effort": "low"}} if model == NEWER else {}

    async def classify(case: Mapping[str, Any]) -> Answer:
        response = await client.messages.create(
            model=model,
            max_tokens=256,  # room for Haiku 5.5's thinking, which counts as output, and the label
            system=PROMPT,
            messages=[{"role": "user", "content": case["email"]}],
            **extras,
        )
        label = "".join(block.text for block in response.content if block.type == "text").strip().lower()
        return priced(label, model, response.usage.input_tokens, response.usage.output_tokens)

    return classify


def offline_classifier(model: str) -> Candidate:
    """An OFFLINE keyword stand-in for ``model``, with made-up token counts. Not a model."""
    careful = model == OLDER  # the stand-in for the dearer model knows two more rules

    async def classify(case: Mapping[str, Any]) -> Answer:
        text = case["email"].lower()
        if any(word in text for word in ("password", "log in", "wire the payment")):
            label = "phishing"
        elif careful and "fee at this link" in text:
            label = "phishing"
        elif any(word in text for word in ("won", "% off", "earn $", "free")) and not (careful and "trial" in text):
            label = "spam"
        else:
            label = "legit"
        return priced(label, model, input_tokens=80 + len(text) // 4, output_tokens=3 if careful else 20)

    return classify


# --- 4. Run both models over every email, twice each, with the older one as the control ------------


async def main() -> Comparison:
    """Compare the two models and print each arm's accuracy and spend, then the report's verdicts."""
    online = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if online:
        print(f"Running against Claude: {OLDER} (the control) vs {NEWER}.\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a keyword stand-in for each model.")
        print("The output shows its shape; its accuracy and dollars say nothing about Claude.\n")
    make = claude_classifier if online else offline_classifier

    comparison = await compare(
        CASES,
        {OLDER: make(OLDER), NEWER: make(NEWER)},  # the arm names are the model ids
        expected=lambda case: case["label"],
        control=OLDER,
        name=f"email triage: {OLDER} vs {NEWER}" + ("" if online else " (offline)"),
        scope_id="compare-two-models",
        k=2,
    )

    # Each arm's accuracy ("match") and, on its last line, what its calls cost.
    for arm, summary in comparison.arms.items():
        print(f"--- {arm} ---\n{summary.render()}\n")

    # The verdict is the report's "Contrasts against the control" table, one row per reading:
    # accuracy and cost_usd, each with its difference from the control, a Holm-adjusted p, and
    # whether it separated. A cheaper arm whose accuracy is "not separated from the control" is
    # the case for switching; more hard cases would tell you whether that tie is real.
    print(comparison.render())
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
