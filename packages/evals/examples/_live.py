"""Claude as a completion client: the live path the examples share. Not an example of its own.

The engine never names a provider. A completion client is anything with an async
``generate(system=, user=, response_format=)`` that returns a reply carrying the ``Completion`` fields below.
Pricing is the client's job, and a cost of None means unpriced, never $0. ``llm_judge.py`` hands this client
to a judge, ``llm_analysis.py`` to the analysis writer, and the comparisons call it as their candidate.
It reads ``ANTHROPIC_API_KEY``, and ``anthropic`` is imported only when a live client is built.
"""

import os
from collections import namedtuple
from types import SimpleNamespace
from typing import Any

from threetears.evals.quick import Answer

#: Model id -> Anthropic's (input, output) list price, USD per million tokens. Check it before you trust the dollars.
PRICES = {"claude-haiku-4-5": (1.00, 5.00), "claude-haiku-5-5": (0.10, 0.50)}

#: Models that take an effort setting; the examples ask for low effort, since their answers are short.
TAKES_EFFORT = {"claude-haiku-5-5"}

Completion = namedtuple(
    "Completion",
    "content input_tokens output_tokens reasoning_tokens cost_usd price_source model served_model stop_reason "
    "temperature",
)


def online() -> bool:
    """Whether an API key is set, so the examples call Claude rather than their offline stand-ins."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def claude(model: str, *, max_tokens: int = 1024) -> Any:
    """``model`` behind the engine's ``CompletionClient`` protocol, priced at its list price."""
    import anthropic  # imported here so the offline path does not need the package

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    effort = {"effort": "low"} if model in TAKES_EFFORT else {}
    input_rate, output_rate = PRICES[model]

    async def generate(*, system: str, user: str, response_format: dict | None = None) -> Completion:
        config = dict(effort)
        if (response_format or {}).get("type") == "json_schema":  # the analysis writer sends the shape it wants
            config["format"] = {"type": "json_schema", "schema": response_format["json_schema"]["schema"]}
        # A judge asks for JSON in its prompt and sends {"type": "json_object"}, which Claude needs no flag for.
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,  # thinking is billed as output, so leave it room
            system=system,
            messages=[{"role": "user", "content": user}],
            **({"output_config": config} if config else {}),
        )
        usage, details = response.usage, response.usage.output_tokens_details
        stopped = {"end_turn": "end_turn", "max_tokens": "max_tokens", "refusal": "content_filter"}
        return Completion(
            content="".join(block.text for block in response.content if block.type == "text"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=details.thinking_tokens if details is not None else None,
            cost_usd=(usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1e6,
            price_source="anthropic list price, from examples/_live.py",
            model=model,
            served_model=response.model,
            temperature=None,  # none is sent, so the model's own sampling applies
            stop_reason=stopped.get(response.stop_reason or "", "error"),
        )

    return SimpleNamespace(generate=generate, model=model, aclose=client.close)


def spent(reply: Completion, value: Any) -> Answer:
    """``value``, with the tokens and dollars ``reply`` cost: the engine sees only what a candidate returns."""
    return Answer(
        value,
        model=reply.model,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        cost_usd=reply.cost_usd,
    )
