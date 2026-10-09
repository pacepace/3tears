"""Is the cheaper model good enough for this prompt, given what it saves?

``compare_two_prompts.py`` changed the prompt; this keeps the prompt and changes the model, so accuracy is
weighed against cost. New here: each candidate returns an ``Answer``, its label plus the tokens and dollars
the call spent, so each arm's summary prints its spend and the report tests the arms' ``cost_usd`` against
the control as it tests their accuracy. How spend is counted: ``docs/cost-and-budgets.md``.

Run it with ``python packages/evals/examples/compare_two_models.py``. With ``ANTHROPIC_API_KEY`` set it
calls Claude 48 times (12 emails, 2 repeats, 2 models) for well under a cent; without it, keyword
stand-ins with made-up token counts play the models, and their numbers say nothing about Claude.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from threetears.evals.quick import Answer, Comparison, compare

# Model id -> Anthropic's list price, (input, output) USD per million tokens. Check it before you trust the dollars.
MODELS = {"claude-haiku-4-5": (1.00, 5.00), "claude-haiku-5-5": (0.10, 0.50)}
CONTROL, CHEAPER = MODELS  # the model you run today, and the one you might switch to

# -----------------------------------------------------------------------------
# 1. The cases: each email, and the label a person gave it.
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 2. The one prompt both models are given.
# -----------------------------------------------------------------------------

PROMPT = """Label the email as exactly one of:
- phishing: it tries to get credentials, payment or a transfer by pretending to be someone it is not.
- spam: unsolicited advertising or a too-good-to-be-true offer that asks for nothing sensitive.
- legit: anything else, including real notices about the reader's own accounts and orders.
Answer with the label only."""

# -----------------------------------------------------------------------------
# 3. The live candidate: one model's label, returned as an Answer that carries what the call cost.
# -----------------------------------------------------------------------------

Candidate = Callable[[Mapping[str, Any]], Awaitable[Answer]]


def priced_answer(label: str, model: str, input_tokens: int, output_tokens: int) -> Answer:
    """The label, with the call's tokens and their cost at the model's list price."""
    input_rate, output_rate = MODELS[model]
    cost_usd = (input_tokens * input_rate + output_tokens * output_rate) / 1e6
    # The engine sees only what the candidate returns, so the spend travels with the label.
    return Answer(label, model=model, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd)


def claude_classifier(model: str) -> Candidate:
    """A candidate that asks ``model`` to label each email."""
    import anthropic  # imported here so the offline path does not need the package

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    effort: dict[str, Any] = {} if model == CONTROL else {"output_config": {"effort": "low"}}  # Haiku 4.5 takes none

    async def classify(case: Mapping[str, Any]) -> Answer:
        response = await client.messages.create(
            model=model,
            max_tokens=256,  # room for Haiku 5.5's thinking, which is billed as output
            system=PROMPT,
            messages=[{"role": "user", "content": case["email"]}],
            **effort,
        )
        label = "".join(block.text for block in response.content if block.type == "text").strip().lower()
        return priced_answer(label, model, response.usage.input_tokens, response.usage.output_tokens)

    return classify


# -----------------------------------------------------------------------------
# 4. The OFFLINE stand-in: keyword rules with made-up token counts, not a model.
#
# The control's stand-in knows two rules the cheaper one does not.
# -----------------------------------------------------------------------------


def offline_classifier(model: str) -> Candidate:
    """A keyword stand-in for ``model``, so the example runs with no API key."""
    careful = model == CONTROL

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
        return priced_answer(label, model, input_tokens=80 + len(text) // 4, output_tokens=3 if careful else 20)

    return classify


# -----------------------------------------------------------------------------
# 5. Run both models over every email twice, test the cheaper against the control, print the verdict.
# -----------------------------------------------------------------------------


async def main() -> Comparison:
    online = bool(os.environ.get("ANTHROPIC_API_KEY"))
    make = claude_classifier if online else offline_classifier
    if online:
        print(f"Running against Claude ({CONTROL}, {CHEAPER}).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a keyword stand-in for each model.\n")

    comparison = await compare(
        CASES,
        {CONTROL: make(CONTROL), CHEAPER: make(CHEAPER)},  # each arm is named by its model id
        expected=lambda case: case["label"],
        # The control is the arm every other arm is tested against.
        control=CONTROL,
        name=f"email triage: {CONTROL} vs {CHEAPER}" + ("" if online else " (offline)"),
        scope_id="compare-two-models",
        k=2,
    )

    # Each arm's accuracy ("match") and, on its last line, what its calls cost.
    for arm, summary in comparison.arms.items():
        print(f"--- {arm} ---\n{summary.render()}\n")

    # "Contrasts against the control" holds the verdict on accuracy and on cost_usd. A cheaper arm whose
    # accuracy is "not separated from the control" is the case for switching, once there are enough hard cases.
    print(comparison.render())
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
