"""Tool turns over scripted HTTP: the request bodies pinned against fixtures, the reply read and priced.

Every turn sends through ``httpx.MockTransport``. Each case of ``fixtures/providers/turns.jsonl`` holds the body a turn
must send and the Messages API reply it is answered with, so the tool definitions, the ``tool_result`` blocks, the
three cache breakpoints and the thinking echo are each pinned as the provider reads them. Nothing here reaches a
live API.
"""

from __future__ import annotations

import json
from dataclasses import replace
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from threetears.evals.providers import Catalogue, Model, ProviderError
from threetears.evals.schema import (
    TurnReply,
    ThinkingBinding,
    ToolResultClearing,
    TurnText,
    TurnThinking,
    TurnToolResult,
    TurnToolSpec,
    TurnToolUse,
    TurnMessage,
    block_from_json,
)

#: The pinned cases, by name: each one's request body and the reply it is answered with.
CASES: Final[dict[str, dict[str, Any]]] = {
    case["case"]: case
    for case in map(
        json.loads,
        (Path(__file__).parent / "fixtures" / "providers" / "turns.jsonl").read_text(encoding="utf-8").splitlines(),
    )
}

MODEL: Final = "claude-opus-5-5"

SYSTEM: Final = "You are the analyst. Read the snapshot through the tool, then submit the report."

OPENING: Final = "Write the report for snapshot snap-0001."

TOOLS: Final = (
    TurnToolSpec(
        "evals_analyst",
        "Read the frozen snapshot: one action per call.",
        {
            "type": "object",
            "properties": {"action": {"type": "string"}, "snapshot": {"type": "string"}},
            "required": ["action"],
        },
    ),
    TurnToolSpec("evals_help", "Explain an action.", {"type": "object", "properties": {"topic": {"type": "string"}}}),
)

#: Opus 5.5's row as the catalogue lists it: $4 and $20 a million, cache reads at $0.40 and writes at $5.
CATALOGUE: Final[dict[str, Any]] = {
    "data": [
        {
            "id": "anthropic/claude-opus-5.5",
            "pricing": {
                "prompt": "0.000004",
                "completion": "0.00002",
                "input_cache_read": "0.0000004",
                "input_cache_write": "0.000005",
            },
        }
    ]
}


class Server:
    """A Messages API that answers each request with the next reply, and keeps each body it was sent."""

    def __init__(self, *replies: Mapping[str, Any]) -> None:
        self._replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        self.urls: list[str] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        self.headers.append(request.headers)
        self.bodies.append(json.loads(request.content))
        return httpx.Response(200, json=self._replies.pop(0))


def catalogue(rows: Mapping[str, Any] = CATALOGUE) -> Catalogue:
    """A catalogue read from ``rows`` over a scripted fetch."""
    return Catalogue(http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=rows))))


def opus(server: Server, /, **options: Any) -> Model:
    """Opus on Anthropic at high effort, sending through ``server``."""
    options.setdefault("catalogue", catalogue())
    options.setdefault("effort", "high")
    return Model(
        MODEL,
        provider="anthropic",
        api_key="test-key",
        http=httpx.AsyncClient(transport=httpx.MockTransport(server)),
        **options,
    )


async def test_the_opening_turn_sends_the_system_prompt_the_tools_and_three_cache_breakpoints() -> None:
    server = Server(CASES["opening"]["reply"])
    model = opus(server)

    reply = await model.turn(system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=32000)

    assert server.urls == ["https://api.anthropic.com/v1/messages"]
    assert server.bodies == [CASES["opening"]["body"]]
    marked = json.dumps(server.bodies[0]).count('"cache_control"')
    assert marked == 3, "system, last tool and last message: one breakpoint each, and no other"
    assert reply.stop_reason == "tool_use"
    assert reply.provider_stop_reason == "tool_use"
    assert [(call.id, call.name, dict(call.input)) for call in reply.tool_uses] == [
        ("toolu_01A", "evals_analyst", {"action": "campaign_overview", "snapshot": "snap-0001"}),
        ("toolu_01B", "evals_analyst", {"action": "no_such_action"}),
    ]
    assert reply.text == "Reading the overview first."
    assert reply.model == MODEL
    assert reply.served_model == "claude-opus-5-5-20261001"


async def test_usage_counts_the_cache_and_the_turn_is_priced_at_its_rates() -> None:
    server = Server(CASES["opening"]["reply"])
    reply = await opus(server).turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=32000
    )

    assert reply.input_tokens == 6200, "1200 fresh and 5000 written to the cache"
    assert reply.output_tokens == 300
    assert reply.cache_write_tokens == 5000
    assert reply.cache_read_tokens == 0, "a reported 0 is 0, not unknown"
    # 1200 x $4/M + 5000 x $5/M (cache write) + 300 x $20/M
    assert reply.cost_usd == pytest.approx(0.0358)
    assert reply.price_source == "openrouter-catalogue:anthropic/claude-opus-5.5"
    assert reply.pricing_failure is None


async def test_tool_results_go_back_with_is_error_on_a_refusal_and_the_thinking_echoed_exactly() -> None:
    server = Server(CASES["opening"]["reply"], CASES["tool_results"]["reply"])
    model = opus(server)
    opening = TurnMessage("user", OPENING)
    first = await model.turn(system=SYSTEM, messages=[opening], tools=TOOLS, max_tokens=32000)
    overview, refused = first.tool_uses
    results = TurnMessage(
        "user",
        [
            TurnToolResult(overview.id, "Campaign overview for snap-0001: three arms, two aspects."),
            TurnToolResult(
                refused.id,
                "evals_analyst has no action no_such_action: call campaign_overview to list them.",
                is_error=True,
            ),
        ],
    )

    second = await model.turn(
        system=SYSTEM, messages=[opening, first.message(), results], tools=TOOLS, max_tokens=32000
    )

    assert server.bodies[1] == CASES["tool_results"]["body"]
    echoed = server.bodies[1]["messages"][1]["content"]
    assert echoed == CASES["opening"]["reply"]["content"], "the reply's blocks, signature included, go back unchanged"
    assert "cache_control" not in json.dumps(server.bodies[1]["messages"][:2]), "only the last message is marked"
    assert second.stop_reason == "end_turn"
    assert isinstance(second.content[0], TurnThinking) and second.content[0].text == ""
    assert second.content[0].to_json() == {"type": "redacted_thinking", "data": "cmVkYWN0ZWQtdGhpbmtpbmctYnl0ZXM="}
    assert second.cache_read_tokens == 5000 and second.cache_write_tokens == 800
    assert second.input_tokens == 5850
    # 50 x $4/M + 5000 x $0.40/M (cache read) + 800 x $5/M (cache write) + 400 x $20/M
    assert second.cost_usd == pytest.approx(0.0142)


async def test_a_turn_cut_at_its_cap_is_named_max_tokens() -> None:
    server = Server(CASES["cut_short"]["reply"])
    reply = await opus(server).turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=32000
    )

    assert server.bodies == [CASES["cut_short"]["body"]]
    assert reply.stop_reason == "max_tokens"
    assert reply.provider_stop_reason == "max_tokens"


@pytest.mark.parametrize(
    ("raw", "read"),
    [
        ("end_turn", "end_turn"),
        ("stop_sequence", "end_turn"),
        ("tool_use", "tool_use"),
        ("max_tokens", "max_tokens"),
        ("refusal", "content_filter"),
        ("pause_turn", "truncated"),
        ("model_context_window_exceeded", "truncated"),
        ("something_new", "truncated"),
        (None, "truncated"),
    ],
)
async def test_each_stop_reason_is_read_in_the_turn_s_words(raw: str | None, read: str) -> None:
    reply = {**CASES["cut_short"]["reply"], "stop_reason": raw}
    turned = await opus(Server(reply)).turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=32000
    )

    assert turned.stop_reason == read
    assert turned.provider_stop_reason == raw


async def test_an_unpriced_turn_costs_none_and_says_why_never_zero() -> None:
    server = Server(CASES["opening"]["reply"])
    reply = await opus(server, catalogue=catalogue({"data": []})).turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=32000
    )

    assert reply.cost_usd is None
    assert reply.price_source is None
    assert reply.pricing_failure and MODEL in reply.pricing_failure


async def test_unreported_counts_are_none_never_zero() -> None:
    bare = {**CASES["opening"]["reply"], "usage": {}}
    reply = await opus(Server(bare)).turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=32000
    )

    assert (reply.input_tokens, reply.output_tokens) == (None, None)
    assert (reply.cache_read_tokens, reply.cache_write_tokens) == (None, None)
    assert reply.cost_usd is None and reply.pricing_failure


async def test_a_turn_without_tools_or_system_marks_only_the_last_message() -> None:
    server = Server(CASES["opening"]["reply"])
    await opus(server, effort=None).turn(system="", messages=[TurnMessage("user", OPENING)], tools=(), max_tokens=512)

    assert server.bodies == [
        {
            "model": MODEL,
            "max_tokens": 512,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": OPENING, "cache_control": {"type": "ephemeral"}}]}
            ],
        }
    ]


def claude(model_id: str, server: Server, /, **options: Any) -> Model:
    """``model_id`` on Anthropic, sending through ``server``."""
    return Model(
        model_id,
        provider="anthropic",
        api_key="test-key",
        catalogue=catalogue(),
        http=httpx.AsyncClient(transport=httpx.MockTransport(server)),
        **options,
    )


async def test_a_thinking_budget_goes_as_enabled_thinking_and_must_sit_below_the_turn_s_cap() -> None:
    server = Server(CASES["opening"]["reply"])
    model = claude("claude-sonnet-4-6", server, effort=2048)

    await model.turn(system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=4096)
    with pytest.raises(ValueError, match=r"max_tokens \(2048\) must exceed the thinking budget \(2048\)"):
        await model.turn(system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=2048)

    assert server.bodies[0]["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in server.bodies[0]
    assert len(server.bodies) == 1, "the refused turn sent nothing"


@pytest.mark.parametrize(
    ("model_id", "effort", "refusal"),
    [
        ("claude-opus-5-5", 2048, r"claude-opus-5-5 takes no thinking budget \(2048 tokens\)"),
        ("claude-haiku-5-5", 2048, r"claude-haiku-5-5 takes no thinking budget"),
        ("claude-sonnet-5", 2048, r"claude-sonnet-5 takes no thinking budget"),
        ("claude-opus-4-8", 2048, r"claude-opus-4-8 takes no thinking budget"),
        ("claude-opus-4-7", 2048, r"claude-opus-4-7 takes no thinking budget"),
        ("claude-opus-5-5", "none", r"claude-opus-5-5 cannot have thinking disabled \(effort='none'\)"),
        ("claude-sonnet-5-5", "none", r"claude-sonnet-5-5 cannot have thinking disabled"),
        ("claude-fable-5-1", "none", r"claude-fable-5-1 cannot have thinking disabled"),
        ("claude-mythos-5", "none", r"claude-mythos-5 cannot have thinking disabled"),
    ],
)
async def test_a_setting_the_model_s_facts_say_it_refuses_is_refused_before_anything_is_sent(
    model_id: str, effort: int | str, refusal: str
) -> None:
    server = Server(CASES["opening"]["reply"])

    with pytest.raises(ValueError, match=refusal):
        await claude(model_id, server, effort=effort).turn(
            system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=4096
        )
    with pytest.raises(ValueError, match=refusal):
        Model(model_id, provider="anthropic", api_key="k", effort=effort, turn_recording="unused-turns.jsonl")
    assert server.bodies == []


@pytest.mark.parametrize("model_id", ["claude-haiku-5-5", "claude-sonnet-5", "claude-opus-5"])
async def test_effort_none_sends_thinking_disabled_where_the_model_takes_it(model_id: str) -> None:
    server = Server(CASES["opening"]["reply"])
    await claude(model_id, server, effort="none").turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=4096
    )

    assert server.bodies[0]["thinking"] == {"type": "disabled"}
    assert "output_config" not in server.bodies[0]


async def test_a_model_the_facts_do_not_cover_is_sent_the_setting_and_its_refusal_comes_back_unresent() -> None:
    sent: list[dict[str, Any]] = []

    def refusing(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        error = {"type": "invalid_request_error", "message": "thinking.type.enabled is not supported for this model"}
        return httpx.Response(400, json={"type": "error", "error": error})

    model = Model(
        "claude-opus-6",
        provider="anthropic",
        api_key="test-key",
        effort=2048,
        catalogue=catalogue(),
        http=httpx.AsyncClient(transport=httpx.MockTransport(refusing)),
    )
    with pytest.raises(ProviderError):
        await model.turn(system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=4096)

    assert [body["thinking"] for body in sent] == [{"type": "enabled", "budget_tokens": 2048}], "sent once, as given"


async def test_a_completion_reads_the_same_facts_and_sends_a_budget_as_its_effort_word() -> None:
    server = Server(
        {
            "model": "claude-opus-5-5-20261001",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
    )
    completion = await claude("claude-opus-5-5", server, effort=4096, max_tokens=8192).generate(system="", user="x")

    assert "budget_tokens" not in json.dumps(server.bodies[0])
    assert server.bodies[0]["thinking"] == {"type": "adaptive"}
    assert completion.sent_settings is not None and completion.sent_settings.unsent_reasoning_max_tokens == 4096


async def test_a_last_message_of_only_thinking_is_refused_before_anything_is_sent() -> None:
    server = Server(CASES["opening"]["reply"])
    thinking = TurnThinking({"type": "thinking", "thinking": "x", "signature": "s"})
    with pytest.raises(ValueError, match="holds only thinking"):
        await opus(server).turn(
            system=SYSTEM, messages=[TurnMessage("assistant", [thinking])], tools=TOOLS, max_tokens=512
        )
    assert server.bodies == []


@pytest.mark.parametrize(
    ("model_id", "options"),
    [
        ("deepseek/deepseek-v4-flash", {}),
        ("gpt-5-nano", {"provider": "openai"}),
        ("local-model", {"base_url": "http://localhost:8000/v1"}),
    ],
)
async def test_every_route_but_anthropic_s_is_refused_naming_the_one_that_works(
    model_id: str, options: dict[str, Any]
) -> None:
    server = Server(CASES["opening"]["reply"])
    model = Model(model_id, api_key="k", http=httpx.AsyncClient(transport=httpx.MockTransport(server)), **options)
    teaching = r"tool turns go to Anthropic's Messages API alone.*provider='anthropic'"

    with pytest.raises(ValueError, match=teaching):
        await model.turn(system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=512)
    with pytest.raises(ValueError, match=teaching):
        model.turn_price_ceiling(system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=512)
    with pytest.raises(ValueError, match=teaching):
        model.turn_session("analyst", snapshot="snap-0001")
    with pytest.raises(ValueError, match=teaching):
        Model(model_id, api_key="k", turn_recording="unused.jsonl", **options)
    assert server.bodies == [], "a refused route sends nothing"


async def test_a_model_built_with_a_temperature_sends_no_turn() -> None:
    server = Server(CASES["opening"]["reply"])
    with pytest.raises(ValueError, match="tool turns send no temperature"):
        await opus(server, temperature=0.2).turn(
            system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=512
        )
    assert server.bodies == []


async def test_the_turn_price_ceiling_bounds_the_turn_and_is_none_before_the_catalogue_is_read() -> None:
    shared = catalogue()
    server = Server(CASES["opening"]["reply"])
    model = opus(server, catalogue=shared)
    messages = [TurnMessage("user", OPENING)]

    assert model.turn_price_ceiling(system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=32000) is None
    await shared.load()
    ceiling = model.turn_price_ceiling(system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=32000)
    reply = await model.turn(system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=32000)

    assert ceiling is not None and reply.cost_usd is not None
    assert ceiling >= reply.cost_usd
    assert ceiling >= 32000 * 0.00002, "the whole output cap at the output rate, at least"
    assert server.bodies, "a ceiling sends nothing; the turn after it sent one body"
    assert len(server.bodies) == 1


async def test_a_session_without_a_recording_is_the_model_itself() -> None:
    model = opus(Server(CASES["opening"]["reply"]))
    assert model.turn_session("analyst", snapshot="snap-0001") is model


def test_blocks_read_back_from_their_json_and_an_unknown_block_is_refused() -> None:
    for block in (
        TurnText("t"),
        TurnToolUse("id", "name", {"a": [1, {"b": 2}]}),
        TurnThinking({"type": "thinking", "thinking": "x", "signature": "s"}),
        TurnToolResult("id", "out", is_error=True),
        TurnToolResult("id", "out"),
    ):
        assert block_from_json(block.to_json()) == block
    with pytest.raises(ValueError, match="'server_tool_use' block is not one a tool turn carries"):
        block_from_json({"type": "server_tool_use", "id": "x"})


async def test_a_reply_carrying_a_block_a_turn_does_not_read_is_a_provider_error() -> None:
    odd = {**CASES["opening"]["reply"], "content": [{"type": "server_tool_use", "id": "x", "name": "web_search"}]}
    with pytest.raises(ProviderError, match="server_tool_use"):
        await opus(Server(odd)).turn(
            system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=512
        )


async def test_a_thinking_block_in_a_user_message_is_refused_before_anything_is_sent() -> None:
    server = Server(CASES["opening"]["reply"])
    thinking = TurnThinking({"type": "thinking", "thinking": "x", "signature": "s"})
    with pytest.raises(ValueError, match="only in the assistant message"):
        await opus(server).turn(
            system=SYSTEM, messages=[TurnMessage("user", [thinking, TurnText(OPENING)])], tools=TOOLS, max_tokens=512
        )
    assert server.bodies == []


async def test_a_clearing_goes_as_context_management_under_its_beta_and_nothing_else_changes() -> None:
    """Server-side context editing: the one way a long tool loop keeps its context in bounds without editing what it
    sent (an edited earlier turn invalidates every later thinking block)."""
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=CASES["opening"]["reply"])

    model = Model(
        MODEL,
        provider="anthropic",
        api_key="test-key",
        catalogue=catalogue(),
        effort="high",
        http=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
    )
    clearing = ToolResultClearing(trigger_input_tokens=150_000, keep_tool_uses=12, clear_at_least_input_tokens=36_000)

    await model.turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=32000, clearing=clearing
    )
    await model.turn(system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=32000)

    cleared, plain = (json.loads(request.content) for request in sent)
    assert cleared["context_management"] == {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 150000},
                "keep": {"type": "tool_uses", "value": 12},
                "clear_at_least": {"type": "input_tokens", "value": 36000},
            }
        ]
    }
    assert sent[0].headers["anthropic-beta"] == "context-management-2025-06-27"
    assert {key: value for key, value in cleared.items() if key != "context_management"} == plain
    assert plain == CASES["opening"]["body"], "a turn with no clearing sends the body it always did"
    assert "anthropic-beta" not in sent[1].headers


async def test_a_binding_goes_as_thinking_block_binding_under_its_beta_beside_a_clearing_s() -> None:
    """The check of every replayed thinking block's conversation: set, it opts the turn in whatever the account's age,
    so an edit to the history is the provider's 400 and never a block quietly dropped."""
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=CASES["opening"]["reply"])

    model = Model(
        MODEL,
        provider="anthropic",
        api_key="test-key",
        catalogue=catalogue(),
        effort="high",
        http=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
    )
    opening = [TurnMessage("user", OPENING)]

    await model.turn(system=SYSTEM, messages=opening, tools=TOOLS, max_tokens=32000, binding=ThinkingBinding())
    await model.turn(
        system=SYSTEM,
        messages=opening,
        tools=TOOLS,
        max_tokens=32000,
        clearing=ToolResultClearing(),
        binding=ThinkingBinding(prefix_mismatch_behavior="drop_block"),
    )

    bound, both = (json.loads(request.content) for request in sent)
    assert bound["thinking"] == {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}}
    assert sent[0].headers["anthropic-beta"] == "thinking-binding-controls-2026-08-01"
    assert {**bound, "thinking": {"type": "adaptive"}} == CASES["opening"]["body"], "only the thinking gains it"
    assert both["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert sent[1].headers["anthropic-beta"] == "context-management-2025-06-27,thinking-binding-controls-2026-08-01"


async def test_a_binding_rides_a_thinking_budget_and_no_turn_that_sends_thinking_disabled() -> None:
    budget, disabled = Server(CASES["opening"]["reply"]), Server(CASES["opening"]["reply"])

    await claude("claude-sonnet-4-6", budget, effort=2048).turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=4096, binding=ThinkingBinding()
    )
    await claude("claude-opus-5", disabled, effort="none").turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=4096, binding=ThinkingBinding()
    )

    assert budget.bodies[0]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 2048,
        "block_binding": {"prefix_mismatch_behavior": "error"},
    }
    assert budget.headers[0]["anthropic-beta"] == "thinking-binding-controls-2026-08-01"
    assert disabled.bodies[0]["thinking"] == {"type": "disabled"}, "thinking disabled replays no block to check"
    assert "anthropic-beta" not in disabled.headers[0]


async def test_a_binding_on_a_model_built_with_no_effort_goes_where_the_model_thinks_anyway() -> None:
    """Claude Opus 5.5 cannot turn thinking off, so sent none it thinks adaptively and replays its blocks: it is sent
    adaptive thinking to carry the check. A model that may not think when sent none is sent neither."""
    always, maybe = Server(CASES["opening"]["reply"], CASES["opening"]["reply"]), Server(CASES["opening"]["reply"])
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return always(request)

    model = Model(
        MODEL,
        provider="anthropic",
        api_key="test-key",
        catalogue=catalogue(),
        http=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
    )
    await model.turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=4096, binding=ThinkingBinding()
    )
    await model.turn(system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=4096)
    await claude("claude-opus-4-8", maybe).turn(
        system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=4096, binding=ThinkingBinding()
    )

    assert always.bodies[0]["thinking"] == {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}}
    assert sent[0].headers["anthropic-beta"] == "thinking-binding-controls-2026-08-01"
    assert "thinking" not in always.bodies[1], "with no binding, no effort sends no thinking, as it always did"
    assert "anthropic-beta" not in sent[1].headers
    assert "thinking" not in maybe.bodies[0]
    assert "anthropic-beta" not in maybe.headers[0]


async def test_a_reply_says_which_binding_its_request_carried_and_what_the_provider_reported_doing() -> None:
    """The evidence a run keeps of the check: the behaviour sent, and the response's ``input_transformations``."""
    dropped = {"type": "thinking_dropped", "reason": "prefix_binding_mismatch", "path": "messages.3.content.0"}
    server = Server(
        {**CASES["opening"]["reply"], "input_transformations": []},
        {**CASES["opening"]["reply"], "input_transformations": [dropped]},
        CASES["opening"]["reply"],
    )
    model = opus(server)
    opening = [TurnMessage("user", OPENING)]

    checked = await model.turn(system=SYSTEM, messages=opening, tools=TOOLS, max_tokens=4096, binding=ThinkingBinding())
    dropping = await model.turn(
        system=SYSTEM,
        messages=opening,
        tools=TOOLS,
        max_tokens=4096,
        binding=ThinkingBinding(prefix_mismatch_behavior="drop_block"),
    )
    plain = await model.turn(system=SYSTEM, messages=opening, tools=TOOLS, max_tokens=4096)

    assert (checked.sent_binding, checked.input_transformations) == ("error", ())
    assert (dropping.sent_binding, dropping.input_transformations) == ("drop_block", (dropped,))
    assert (plain.sent_binding, plain.input_transformations) == (None, None)
    assert TurnReply.from_json(dropping.to_json()) == dropping
    earlier = {
        key: value for key, value in checked.to_json().items() if key not in {"sent_binding", "input_transformations"}
    }
    assert TurnReply.from_json(earlier) == replace(checked, sent_binding=None, input_transformations=None), (
        "a reply recorded before the members reads them as none"
    )


def test_a_binding_refuses_a_behaviour_the_provider_does_not_take() -> None:
    with pytest.raises(ValueError, match="'error' or 'drop_block', not 'warn'"):
        ThinkingBinding(prefix_mismatch_behavior="warn")  # type: ignore[arg-type]


def test_a_clearing_refuses_a_limit_that_is_not_a_count_above_0() -> None:
    with pytest.raises(ValueError, match="keep_tool_uses is a whole number above 0"):
        ToolResultClearing(keep_tool_uses=0)
