"""A campaign assembled twice is one bundle: the same fingerprint, and a frozen copy that reloads to it.

The campaign is two keyword classifiers over a dozen support tickets, plain rules that always run offline, so it is
deterministic in everything but its ids and times. (It was the retired memo writer's recorded example's campaign.)
"""

from __future__ import annotations

from typing import Any

from threetears.evals import Eval
from threetears.evals.analysis import CampaignBundle, assemble_campaign_bundle
from packages.evals.tests.off_loop import off_loop

TICKETS = [
    ("I was charged twice for my March invoice.", "billing"),
    ("Can I get a refund for the plan I bought yesterday?", "billing"),
    ("My card was charged but the upgrade page shows an error.", "billing"),
    ("I was charged for a seat after removing that user from my account.", "billing"),
    ("The export button does nothing in Firefox.", "bug"),
    ("The app crashes when I upload a large photo.", "bug"),
    ("Since the update, the login page loops back to itself.", "bug"),
    ("The reset-password email never arrives.", "account"),
    ("Please change the email address on my account.", "account"),
    ("It would be great if reports could be scheduled weekly.", "feature_request"),
    ("Do you plan to add a dark mode?", "feature_request"),
    ("Could you add an option to pay by invoice?", "feature_request"),
]
CASES = [{"ticket": ticket, "queue": queue} for ticket, queue in TICKETS]

# Each arm takes the first queue whose words appear; the candidate checks the ambiguous queues in a better order.
BASELINE_RULES = {"account": ("account", "email", "login"), "billing": ("charged", "refund", "invoice")}
CANDIDATE_RULES = {"feature_request": ("would be great", "plan to add", "could you add"), "billing": ("charged",)}
CANDIDATE_RULES |= {"bug": ("error", "crash", "nothing", "loops"), "account": ("account", "email", "password")}


def keyword_classifier(rules: dict[str, tuple[str, ...]]) -> Any:
    async def classify(case: dict[str, Any]) -> str:
        text = case["ticket"].lower()
        return next((queue for queue, words in rules.items() if any(word in text for word in words)), "bug")

    return classify


async def test_assembling_the_same_campaign_twice_gives_the_same_fingerprint() -> None:
    arms = {"baseline": keyword_classifier(BASELINE_RULES), "candidate": keyword_classifier(CANDIDATE_RULES)}
    comparison = await Eval(CASES, arms, ["queue"], k=2).run(store=":memory:", scope="determinism")
    once, twice = (
        off_loop(assemble_campaign_bundle, comparison.host, comparison.campaign_id, comparison.scope_id)
        for _ in range(2)
    )
    assert once.fingerprint() == twice.fingerprint()
    assert CampaignBundle.from_json(once.to_json(indent=2)).fingerprint() == once.fingerprint()
