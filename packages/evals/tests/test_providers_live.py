"""The client against OpenRouter for real: recorded live once, replayed offline from then on.

Each test builds its :class:`~threetears.evals.providers.Model` with ``recording=`` a file under
``fixtures/providers/``, so the suite replays what OpenRouter answered and calls nothing. To record them again, with
``OPENROUTER_API_KEY`` set (four calls on the cheapest models, well under a cent)::

    RECORD=1 ./scripts/test.sh evals -k test_providers_live

What only a live call can show is asserted here, on the recorded answer: that the request each wire builds is one
OpenRouter serves (plain, under a strict schema, and resent after a refusal), and what it reports back (the upstream
that served it, the stop reason, and its own cost).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from threetears.evals.kernel import strict_response_format
from threetears.evals.providers import Catalogue, Model
from threetears.evals.testing import check_completion_conformance

#: Where the recordings live.
RECORDINGS = Path(__file__).resolve().parent / "fixtures" / "providers"

#: The runner model the ladder records on: the cheapest DeepSeek flash OpenRouter prices.
MODEL = "deepseek/deepseek-v4-flash"

#: A cheap model no endpoint of which takes a temperature, so a request carrying one under enforcement is refused.
NO_TEMPERATURE = "openai/gpt-5-nano"


class Triage(BaseModel):
    """A support email, triaged."""

    category: Literal["billing", "bug", "account", "sales", "spam"]
    urgent: bool


EMAIL = "Subject: charged twice\n\nMy card was charged twice for this month's plan. Please refund one today."


def recorded(name: str, model_id: str = MODEL, **options: Any) -> Model:
    """``model_id`` replaying (or, under ``RECORD=1``, recording) ``fixtures/providers/<name>.jsonl``."""
    return Model(model_id, recording=RECORDINGS / f"{name}.jsonl", **options)


def extra(completion: object, name: str) -> object:
    """A member of the client's own beyond the port's: on the live completion, or kept under a replay's ``extra``."""
    kept = getattr(completion, "extra", None)
    return kept[name] if kept is not None else getattr(completion, name)


async def test_a_plain_call_is_served_and_says_who_served_it() -> None:
    model = recorded("plain", max_tokens=1000)
    answer = await model.text("Name the colour of a clear daytime sky in one word.")
    assert "blue" in answer.lower()


async def test_a_strict_schema_is_sent_enforced_and_the_reply_is_the_schema() -> None:
    model = recorded("json_schema", max_tokens=1000)
    directive = strict_response_format("Triage", Triage.model_json_schema())
    completion = await model.generate(system="Triage the support email.", user=EMAIL, response_format=directive)
    assert completion.stop_reason == "end_turn"
    assert completion.sent_settings is not None and completion.sent_settings.strict_output is True
    assert Triage.model_validate_json(completion.content).category == "billing"


async def test_a_refused_temperature_is_resent_without_it_and_the_reply_says_none_was_sent() -> None:
    # The catalogue knows nothing of the model, as for one listed after it was read, so the temperature goes out and
    # the router refuses the enforced request (404, no endpoint); the resend without it is served, still enforced.
    unknown = Catalogue(
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": []})))
    )
    model = recorded(
        "negotiation", NO_TEMPERATURE, max_tokens=2000, effort="minimal", temperature=0.2, catalogue=unknown
    )
    directive = strict_response_format("Triage", Triage.model_json_schema())
    completion = await model.generate(system="Triage the support email.", user=EMAIL, response_format=directive)
    assert completion.temperature is None
    assert completion.sent_settings is not None and completion.sent_settings.strict_output is True
    assert Triage.model_validate_json(completion.content).category == "billing"


async def test_an_answer_is_priced_by_openrouters_own_figure() -> None:
    model = recorded("priced", max_tokens=1000)
    completion = await model.generate(system="Answer in one short sentence.", user="What is 17 times 3?")
    check_completion_conformance(completion, require_sent_settings=True)
    assert completion.cost_usd is not None and completion.cost_usd > 0
    assert completion.price_source == "openrouter:reported"
    assert completion.input_tokens and completion.output_tokens
    assert completion.served_model is not None and extra(completion, "served_provider")
    assert extra(completion, "pricing_failure") is None
    assert "51" in completion.content
