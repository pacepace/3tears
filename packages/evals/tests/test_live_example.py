"""``examples/_live.py``, the Claude client every live example shares, reads the real SDK's reply.

No test here calls the API. Each builds a reply from the ``anthropic`` SDK's own types and points
``AsyncAnthropic`` at a client returning it, so a field the SDK renames turns this red rather than a
newcomer's first live run. Model ids and prices are read off the helper, never written here.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.quick import Answer
from packages.evals.tests.example_loader import EXAMPLES, load_example
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations


def _reply(anthropic: ModuleType, model: str, **overrides: Any) -> Any:
    """A reply as the SDK types it: a thinking block, then the text."""
    return anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": f"{model}-served",
            "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": " Spam\n"}],
            "stop_reason": "max_tokens",
            "stop_sequence": None,
            "usage": {"input_tokens": 1_000, "output_tokens": 200, "output_tokens_details": {"thinking_tokens": 50}},
        }
        | overrides
    )


def _fake_sdk(monkeypatch: pytest.MonkeyPatch, anthropic: ModuleType, reply: Any) -> list[dict[str, Any]]:
    """Point ``anthropic.AsyncAnthropic`` at a client whose ``create`` records each request and returns ``reply``."""
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append(request)
        return reply

    async def close() -> None:
        return None

    monkeypatch.setattr(
        anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create), close=close)
    )
    return sent


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-haiku-5-5"])
async def test_a_reply_is_priced_at_its_model_s_list_price_and_read_field_for_field(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    anthropic = pytest.importorskip("anthropic")
    live = load_example("_live.py")
    assert model in live.PRICES, "the model ids are the helper's own"
    sent = _fake_sdk(monkeypatch, anthropic, _reply(anthropic, model))
    client = live.claude(model)

    completion = await client.generate(system="be brief", user="hello", response_format={"type": "json_object"})

    (request,) = sent
    assert (request["model"], request["system"], request["messages"]) == (
        model,
        "be brief",
        [{"role": "user", "content": "hello"}],
    )
    assert "response_format" not in request, "a judge's json_object format is asked for in its prompt, not sent"
    if model in live.TAKES_EFFORT:
        assert request["output_config"] == {"effort": "low"}
    else:
        assert "output_config" not in request, f"{model} takes no effort setting"
    input_rate, output_rate = live.PRICES[model]
    assert completion.cost_usd == pytest.approx((1_000 * input_rate + 200 * output_rate) / 1e6)
    assert (completion.input_tokens, completion.output_tokens, completion.reasoning_tokens) == (1_000, 200, 50)
    assert (completion.content, completion.stop_reason) == (" Spam\n", "max_tokens")
    assert (completion.model, completion.served_model, completion.temperature) == (model, f"{model}-served", None)
    await client.aclose()


async def test_a_json_schema_format_is_sent_as_a_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    live = load_example("_live.py")
    sent = _fake_sdk(monkeypatch, anthropic, _reply(anthropic, "claude-haiku-5-5", stop_reason="refusal"))
    schema = {"type": "object", "properties": {"headline": {"type": "string"}}}
    completion = await live.claude("claude-haiku-5-5").generate(
        system="s", user="u", response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}
    )
    assert sent[0]["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": schema}}
    assert completion.stop_reason == "content_filter"


async def test_spent_carries_the_reply_s_spend_beside_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    live = load_example("_live.py")
    _fake_sdk(monkeypatch, anthropic, _reply(anthropic, "claude-haiku-5-5"))
    reply = await live.claude("claude-haiku-5-5").generate(system="s", user="u")
    answer = live.spent(reply, "spam")
    assert isinstance(answer, Answer)
    assert (answer.value, answer.model, answer.input_tokens, answer.output_tokens, answer.cost_usd) == (
        "spam",
        "claude-haiku-5-5",
        1_000,
        200,
        reply.cost_usd,
    )


def test_online_reads_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    live = load_example("_live.py")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not live.online()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert live.online()


def test_every_example_reaches_the_engine_only_through_public_roots() -> None:
    examples = [(path.name, path) for path in sorted(EXAMPLES.glob("*.py"))]
    assert len(examples) >= 10
    assert public_root_violations(SOURCE_ROOT, examples, consumer_root=REPO_ROOT) == []
