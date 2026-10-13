"""What the models adapter can read off each provider's reply, pinned: the "what LangChain hides" report as a contract.

For each provider the adapter supports, one full canned reply goes through a mocked transport (a reply carrying
every field the provider sends: usage with a reasoning split, the served model, a finish reason, OpenRouter's
cost and serving upstream), and the table below says which of the port's ten
:class:`~threetears.evals.schema.CompletionResult` members, plus the adapter's own ``served_provider``, the
adapter populates from it and which it leaves ``None``. It is the design's table
(``eval-providers-design.md``, "What LangChain hides"), probed against 3tears-models 0.66.0 with langchain-core
1.6.3, langchain-anthropic 1.4.8, langchain-openai 1.3.5 and langchain-openrouter 0.2.9.

A member moving either way is a fact about the provider layer, not a bug to fix here: a LangChain release that
stops copying a field hides a member, one that starts copying it exposes one, and models can do either. So the
failure names the member and the direction, and the commit that moves the pin says which upgrade moved it.

``stop_reason``, ``content`` and ``model`` are populated on every reply by construction (an unnamed stop reads
``truncated``, never ``None``); they are pinned anyway, because the table is the whole member set or it is not a
report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from threetears.evals.schema.completion import COMPLETION_RESULT_ATTRIBUTES

from packages.evals.tests.models_adapter_support import ProviderTransport, build, mocked_providers, plan_reply

#: Populated or ``None``, per provider and member, for a full reply.
POPULATED, ABSENT = "populated", "None"

#: The pin. Anthropic: reasoning from the response's own usage (LangChain leaves the thinking split out of
#: ``usage_metadata``), cost from the catalogue, no serving upstream (the provider is called directly). OpenAI:
#: the same, its reasoning split in ``usage_metadata``. OpenRouter: everything, its cost its own figure and its
#: serving upstream named.
PINNED: dict[str, dict[str, str]] = {
    "anthropic": {
        "content": POPULATED,
        "input_tokens": POPULATED,
        "output_tokens": POPULATED,
        "reasoning_tokens": POPULATED,
        "cost_usd": POPULATED,
        "price_source": POPULATED,
        "model": POPULATED,
        "served_model": POPULATED,
        "temperature": POPULATED,
        "stop_reason": POPULATED,
        "served_provider": ABSENT,
    },
    "openai": {
        "content": POPULATED,
        "input_tokens": POPULATED,
        "output_tokens": POPULATED,
        "reasoning_tokens": POPULATED,
        "cost_usd": POPULATED,
        "price_source": POPULATED,
        "model": POPULATED,
        "served_model": POPULATED,
        "temperature": POPULATED,
        "stop_reason": POPULATED,
        "served_provider": ABSENT,
    },
    "openrouter": {
        "content": POPULATED,
        "input_tokens": POPULATED,
        "output_tokens": POPULATED,
        "reasoning_tokens": POPULATED,
        "cost_usd": POPULATED,
        "price_source": POPULATED,
        "model": POPULATED,
        "served_model": POPULATED,
        "temperature": POPULATED,
        "stop_reason": POPULATED,
        "served_provider": POPULATED,
    },
}

#: Each provider's model for the full reply: one whose temperature LangChain keeps (``gpt-5`` drops it at build,
#: which ``test_models_adapter.py`` holds), and an OpenRouter id that is not an alias, whose reply names the model.
_MODELS = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o", "openrouter": "anthropic/claude-sonnet-4.6"}


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> Iterator[ProviderTransport]:
    """The mocked network for one test."""
    yield from mocked_providers(monkeypatch)


def test_the_pin_covers_every_member_of_the_port() -> None:
    """Non-vacuity: every row names exactly the port's members and the adapter's ``served_provider``."""
    for provider, row in PINNED.items():
        assert set(row) == {*COMPLETION_RESULT_ATTRIBUTES, "served_provider"}, provider


@pytest.mark.parametrize("provider", sorted(PINNED))
async def test_what_the_adapter_reads_off_a_full_reply_is_pinned(net: ProviderTransport, provider: str) -> None:
    plan_reply(net, provider)
    client = build(net, provider, model=_MODELS[provider], temperature=0.2)
    result: Any = await client.generate(system="You grade.", user="Grade this.")
    seen = {member: ABSENT if getattr(result, member) is None else POPULATED for member in PINNED[provider]}
    moved = {
        member: f"{PINNED[provider][member]} -> {now}"
        for member, now in seen.items()
        if now != PINNED[provider][member]
    }
    assert not moved, (
        f"What the adapter can read off a {provider} reply moved: {moved}. A LangChain or 3tears-models upgrade "
        "moved the pin (a member it stopped exposing reads None now; one it started exposing is populated). Check "
        "the member's value is right, move PINNED and the design's table in the same commit, and say in the commit "
        "message which upgrade moved which member."
    )
