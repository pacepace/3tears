"""Is the cheaper model good enough for this prompt, given what it saves?

``compare_two_prompts.py`` changed the prompt; this keeps the prompt and changes the model, so accuracy is weighed
against cost. New here: spend, and a margin.

- **Spend.** Each candidate returns an ``Answer``: its label plus the tokens and dollars the call spent. The report
  tests the arms' spend (Production cost, the candidate's own, never a judge's) against the control, as it tests
  their accuracy. How spend is counted: ``docs/cost-and-budgets.md``.
- **A margin.** "Good enough" means "no more than ``MARGIN`` less accurate". Only the verdict ``equivalent``
  supports that: an equivalence test shows the difference inside a margin declared on the measure.
  "Not separated" never does: it means these cases could not tell the models apart. ``margins=`` declares the
  margin on the ``correct`` scorer; with none declared, no contrast can read ``equivalent``.

Twelve emails cannot show a margin of 0.05. Two models that agree on every email need about a hundred
emails (in this package's runs, 80 did not show it and 120 did). So expect "not separated" here, and read the
interval: it is how much worse the cheaper model could plausibly be. The verdicts: ``docs/reading-reports.md``.

Run it with ``python packages/evals/examples/compare_two_models.py``. With ``ANTHROPIC_API_KEY`` set it calls
Claude 48 times (12 emails x 2 repeats x 2 models), for well under a cent. Without it, two keyword stand-ins,
with made-up token counts and prices, play a current model and a cheaper one. Their numbers say nothing about Claude.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from _live import claude, online, spent
from threetears.evals.quick import Answer, Comparison, compare

LIVE = ("claude-haiku-4-5", "claude-haiku-5-5")  # the model you run today, and the cheaper one you might switch to
STAND_INS = {"current-stand-in": (1.00, 5.00), "cheaper-stand-in": (0.10, 0.50)}  # made-up prices, USD per M tokens
MARGIN = 0.05  # the most accuracy you would give up for the saving: declare it before you look at the results

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

PROMPT = """Label the email as exactly one of:
- phishing: it tries to get credentials, payment or a transfer by pretending to be someone it is not.
- spam: unsolicited advertising or a too-good-to-be-true offer that asks for nothing sensitive.
- legit: anything else, including real notices about the reader's own accounts and orders.
Answer with the label only."""

# -----------------------------------------------------------------------------
# 2. The grade: a scorer, since a margin is declared on a scorer's measure.
# -----------------------------------------------------------------------------


def correct(case: Mapping[str, Any], label: str) -> bool:
    """Whether the label is the one a person gave the email."""
    return label == case["label"]


# -----------------------------------------------------------------------------
# 3. The candidates: a model's label, returned as an Answer that carries what the call cost.
# -----------------------------------------------------------------------------

Candidate = Callable[[Mapping[str, Any]], Awaitable[Answer]]


def claude_classifier(model: str) -> Candidate:
    client = claude(model, max_tokens=256)

    async def classify(case: Mapping[str, Any]) -> Answer:
        reply = await client.generate(system=PROMPT, user=case["email"])
        return spent(reply, reply.content.strip().lower())  # the engine sees only what the candidate returns

    return classify


def offline_classifier(model: str) -> Candidate:
    """A keyword stand-in, not a model; the current one knows two rules the cheaper one does not."""
    careful, (input_rate, output_rate) = model == "current-stand-in", STAND_INS[model]

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
        tokens_in, tokens_out = 80 + len(text) // 4, 3 if careful else 20
        cost = (tokens_in * input_rate + tokens_out * output_rate) / 1e6
        return Answer(label, model=model, input_tokens=tokens_in, output_tokens=tokens_out, cost_usd=cost)

    return classify


# -----------------------------------------------------------------------------
# 4. Run both models over every email twice, test the cheaper against the current one, and decide.
# -----------------------------------------------------------------------------

WHAT_TO_DO = {
    "improved": "the cheaper model is more accurate on these cases: switch, if the spend row shows the saving.",
    "equivalent": f"the cheaper model is shown within {MARGIN} of the current one: it is good enough, so switch "
    "if the spend row shows the saving.",
    "regressed": "the cheaper model is shown less accurate: switch only if the interval's loss is worth the saving.",
    "not separated": "these cases could not tell the models apart, which does not show the cheaper one is good "
    "enough. Keep the current model, and add cases (hard ones first) until the verdict is decided.",
    "untested": "no test could decide; the verdict says why.",
}


async def main() -> Comparison:
    current, cheaper = LIVE if online() else STAND_INS
    make = claude_classifier if online() else offline_classifier
    if online():
        print(f"Running against Claude ({current}, {cheaper}).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with keyword stand-ins for both models.\n")

    comparison = await compare(
        CASES,
        {current: make(current), cheaper: make(cheaper)},
        [correct],
        factors=("model",),  # the arms ARE models: each key is its run's model, and the report reads model=<key>
        margins={"correct": MARGIN},  # declared on the scorer, so a contrast on it can read "equivalent"
        control=current,
        scope_id="compare-two-models",
        k=2,
    )
    for arm, summary in comparison.arms.items():
        print(f"{arm}: {summary.candidate_calls} calls, candidate spend ${summary.candidate_cost_usd:.5f}")

    # The verdicts: accuracy, then spend (lower is better, so a saving reads "improved").
    print("\nEach reading against the control (Production cost is US dollars per answer):")
    for row in comparison.contrasts():
        p = "" if row["p_adjusted"] is None else f", Holm-adjusted p {row['p_adjusted']:.2g}"  # none if nothing varied
        print(f"{row['arm']} vs {comparison.control} on {row['reading']}: {row['control_mean']:.3g} -> ", end="")
        print(f"{row['arm_mean']:.3g}, delta {row['delta']:+.3g}, interval {row['interval']}{p}: {row['verdict']}")

    (accuracy,) = comparison.contrasts("correct")
    verdict = next(verdict for verdict in WHAT_TO_DO if accuracy["verdict"].startswith(verdict))
    print(f"\nOn accuracy, {verdict}: {WHAT_TO_DO[verdict]}")

    print("\nWhere the models disagree:")
    for index, case in enumerate(CASES):
        said = {arm: [r.answer for r in comparison.results(arm) if r.case == str(index)] for arm in comparison.arms}
        if len({str(answers) for answers in said.values()}) > 1:
            print(f"  {case['email']!r}, labelled {case['label']}:")
            print("    " + "; ".join(f"{arm} said {', '.join(answers)}" for arm, answers in said.items()))
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
