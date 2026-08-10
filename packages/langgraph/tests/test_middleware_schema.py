"""Behavioral tests for :class:`SchemaPrimingMiddleware`.

Verifies the ``awrap_model_call`` seam: the honesty preamble + documented-schema block
are FOLDED into ``request.system_message`` (never appended as a second, non-consecutive
``SystemMessage`` -- the pattern that crashes LangChain's Anthropic binding), the
documented-schema block is persisted onto ``metadata`` via a returned
:class:`~langchain.agents.middleware.types.ExtendedModelResponse` ``Command``, the
honesty rule ships even when no digest documents tables (schema block empty, honesty
still folded), a non-datasource agent and a missing integration pass the call through,
a digest read that faults soft-fails to the honesty rule alone, the token budget drops
the tail with a footer, and the sync mirror passes through. A REAL ``create_agent`` +
capturing fake model regression proves the model receives exactly one leading
``SystemMessage`` (the crash guard the stub-model unit tests cannot see).

Two more properties are load-bearing and covered here. WHICH tables survive truncation
is decided by the priority rule (hazard notes, then documentation weight, then name),
not by the order the digest yielded, so the same block renders from any permutation of
the same tables. And the budget is DERIVED from the documented-table count between a
floor and a ceiling, so the 35-table corpus that used to prime 3 tables now primes all
35 while a runaway corpus still stops at the ceiling.

The integration is read off ``config["configurable"]``; the direct-drive tests set the
runnable-config contextvar so the middleware's ``get_config`` sees it, and the
real-agent test passes it via ``ainvoke``'s ``config``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from random import Random
from types import SimpleNamespace
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables.config import var_child_runnable_config

from threetears.langgraph.middleware_schema import (
    _SCHEMA_PRIMING_TOKEN_CEILING,
    _SCHEMA_PRIMING_TOKEN_FLOOR,
    _SCHEMA_PRIMING_TOKENS_PER_TABLE,
    SchemaPrimingMiddleware,
    SchemaPrimingState,
    _HONESTY_PREAMBLE,
    _render_schema_block,
    _scaled_budget,
)


class _StubIntegration:
    """opaque schema-priming integration standing in for the real one.

    exposes the duck-typed surface the middleware calls: ``datasource_ids`` and
    ``get_digest``. ``ids`` are returned as-is; ``digests`` maps id -> digest entity;
    ``raise_on_digest`` forces ``get_digest`` to raise (soft-fail path).
    """

    def __init__(
        self,
        ids: list[Any],
        digests: dict[Any, Any] | None = None,
        *,
        raise_on_digest: bool = False,
    ) -> None:
        self._ids = ids
        self._digests = digests or {}
        self._raise_on_digest = raise_on_digest

    async def datasource_ids(self) -> list[Any]:
        return self._ids

    async def get_digest(self, datasource_id: Any) -> Any:
        if self._raise_on_digest:
            raise RuntimeError("digest read boom")
        return self._digests.get(datasource_id)


def _digest(tables: list[dict[str, Any]]) -> Any:
    """build a digest entity exposing a ``tables`` projection (duck-typed)."""
    return SimpleNamespace(tables=tables)


@contextmanager
def _configured(configurable: dict[str, Any]) -> Iterator[None]:
    """set the runnable-config contextvar so ``get_config`` sees ``configurable``."""
    token = var_child_runnable_config.set({"configurable": configurable})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _request(system: SystemMessage | None) -> ModelRequest:
    """build a minimal ``ModelRequest`` for the direct-drive tests."""
    return ModelRequest(
        model=cast("BaseChatModel", SimpleNamespace()),
        messages=[HumanMessage(content="hi")],
        system_message=system,
    )


def _drive(
    mw: SchemaPrimingMiddleware,
    request: ModelRequest,
    configurable: dict[str, Any],
) -> tuple[ModelRequest, Any]:
    """drive ``awrap_model_call``; return the request the handler saw + the result."""
    captured: dict[str, ModelRequest] = {}

    async def _handler(req: ModelRequest) -> Any:
        captured["req"] = req
        return SimpleNamespace(result=[AIMessage(content="ok")])

    async def _run() -> Any:
        with _configured(configurable):
            return await mw.awrap_model_call(request, _handler)

    out = asyncio.run(_run())
    return captured["req"], out


_TABLE = {
    "schema": "public",
    "table": "orders",
    "description": "customer orders",
    "columns": [{"name": "id", "type": "int", "description": "pk"}],
}


class TestInjection:
    def test_folds_honesty_and_schema_into_system(self) -> None:
        integration = _StubIntegration(["ds-1"], {"ds-1": _digest([_TABLE])})
        req, _out = _drive(
            SchemaPrimingMiddleware(),
            _request(SystemMessage(content="base")),
            {"schema_priming_integration": integration},
        )
        assert req.system_message is not None
        content = req.system_message.content
        assert isinstance(content, str)
        # folded into the SINGLE system message, base first then the honesty + schema.
        assert content.startswith("base\n\n" + _HONESTY_PREAMBLE)
        assert "# Documented schema" in content
        assert "public.orders" in content

    def test_returns_extended_response_with_metadata_block(self) -> None:
        integration = _StubIntegration(["ds-1"], {"ds-1": _digest([_TABLE])})
        _req, out = _drive(
            SchemaPrimingMiddleware(),
            _request(SystemMessage(content="base")),
            {"schema_priming_integration": integration},
        )
        assert isinstance(out, ExtendedModelResponse)
        assert out.command is not None
        update = out.command.update
        assert isinstance(update, dict)
        block = update["metadata"]["documented_schema_block"]
        # the stash carries the schema block WITHOUT the honesty preamble.
        assert "# Documented schema" in block
        assert _HONESTY_PREAMBLE not in block

    def test_honesty_ships_when_no_digest_documents_tables(self) -> None:
        # datasource resolves but no digest entity exists -> schema block empty, but
        # the honesty rule still folds (imperatives are emitted by any datasource read).
        integration = _StubIntegration(["ds-1"], {})
        req, out = _drive(SchemaPrimingMiddleware(), _request(None), {"schema_priming_integration": integration})
        assert isinstance(out, ExtendedModelResponse)
        assert req.system_message is not None
        assert req.system_message.content == _HONESTY_PREAMBLE
        assert out.command is not None
        assert out.command.update["metadata"]["documented_schema_block"] == ""

    def test_digest_read_fault_drops_block_but_keeps_honesty(self) -> None:
        integration = _StubIntegration(["ds-1"], raise_on_digest=True)
        req, out = _drive(SchemaPrimingMiddleware(), _request(None), {"schema_priming_integration": integration})
        assert isinstance(out, ExtendedModelResponse)
        assert req.system_message is not None
        assert req.system_message.content == _HONESTY_PREAMBLE
        assert out.command is not None
        assert out.command.update["metadata"]["documented_schema_block"] == ""


class TestNoop:
    def test_noop_without_integration(self) -> None:
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(SchemaPrimingMiddleware(), req_in, {})
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_noop_for_non_datasource_agent(self) -> None:
        # integration present but resolves no datasource ids -> pass-through.
        integration = _StubIntegration([])
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(SchemaPrimingMiddleware(), req_in, {"schema_priming_integration": integration})
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_noop_outside_runnable_context(self) -> None:
        # no contextvar set -> get_config raises -> soft-fail to pass-through.
        mw = SchemaPrimingMiddleware()
        req = _request(SystemMessage(content="base"))
        captured: dict[str, ModelRequest] = {}

        async def _handler(r: ModelRequest) -> Any:
            captured["req"] = r
            return SimpleNamespace(result=[])

        out = asyncio.run(mw.awrap_model_call(req, _handler))
        assert captured["req"] is req
        assert not isinstance(out, ExtendedModelResponse)


def _rich(name: str, columns: int = 4) -> dict[str, Any]:
    """a heavily documented table: table prose plus described columns."""
    return {
        "schema": "public",
        "table": name,
        "description": "what this table is for. " * 8,
        "columns": [
            {"name": f"c{i}", "type": "int", "description": "what this column means. " * 3} for i in range(columns)
        ],
    }


def _bare(name: str) -> dict[str, Any]:
    """a minimally documented table: one short line, undescribed columns."""
    return {
        "schema": "public",
        "table": name,
        "description": "x",
        "columns": [{"name": "c0", "type": "int"}],
    }


def _hazardous(name: str) -> dict[str, Any]:
    """a thinly documented table carrying an unloaded-column warning."""
    entry = _bare(name)
    entry["unloaded_columns"] = ["never_loaded"]
    return entry


def _rendered_names(block: str) -> list[str]:
    """the qualified table names in the block, in rendered order."""
    return [line.removeprefix("### ") for line in block.splitlines() if line.startswith("### ")]


class TestBudget:
    def test_tail_dropped_with_footer(self) -> None:
        tables = [{"schema": "public", "table": f"t{i}", "description": "x" * 200, "columns": []} for i in range(10)]
        block = _render_schema_block([_digest(tables)], budget=120)
        assert "not shown here" in block
        # at least one table always renders even under a tight budget.
        assert "public.t0" in block

    def test_empty_when_no_tables(self) -> None:
        assert _render_schema_block([_digest([])], budget=1500) == ""

    def test_single_oversized_table_still_renders(self) -> None:
        # the at-least-one rule: a budget smaller than one table still primes that table.
        block = _render_schema_block([_digest([_rich("only", columns=20)])], budget=10)
        assert _rendered_names(block) == ["public.only"]
        assert "not shown here" not in block

    def test_footer_count_matches_tables_dropped(self) -> None:
        tables = [_rich(f"t{i:02d}") for i in range(20)]
        block = _render_schema_block([_digest(tables)], budget=400)
        dropped = len(tables) - len(_rendered_names(block))
        assert dropped > 0
        assert f"_{dropped} more documented table(s) not shown here" in block

    def test_footer_states_the_selection_rule(self) -> None:
        # a table missing from the block must read as a stated decision, not a fault.
        tables = [_rich(f"t{i:02d}") for i in range(20)]
        block = _render_schema_block([_digest(tables)], budget=400)
        assert "keeps the tables carrying the most documentation" in block


class TestScaledBudget:
    def test_budget_scales_with_table_count(self) -> None:
        assert _scaled_budget(10) == 10 * _SCHEMA_PRIMING_TOKENS_PER_TABLE
        assert _scaled_budget(20) > _scaled_budget(10)

    def test_floor_holds_for_a_small_corpus(self) -> None:
        assert _scaled_budget(0) == _SCHEMA_PRIMING_TOKEN_FLOOR
        assert _scaled_budget(1) == _SCHEMA_PRIMING_TOKEN_FLOOR
        assert _scaled_budget(2) == _SCHEMA_PRIMING_TOKEN_FLOOR

    def test_ceiling_holds_for_a_huge_corpus(self) -> None:
        assert _scaled_budget(1000) == _SCHEMA_PRIMING_TOKEN_CEILING
        assert _scaled_budget(100_000) == _SCHEMA_PRIMING_TOKEN_CEILING

    def test_wide_corpus_renders_whole_where_the_old_constant_truncated(self) -> None:
        # the measured regression: 35 documented tables, of which the fixed 1500-token
        # budget primed 3. the derived budget primes all 35.
        tables = [_rich(f"t{i:02d}", columns=12) for i in range(35)]
        scaled = _render_schema_block([_digest(tables)])
        assert len(_rendered_names(scaled)) == 35
        assert "not shown here" not in scaled
        fixed = _render_schema_block([_digest(tables)], budget=_SCHEMA_PRIMING_TOKEN_FLOOR)
        assert len(_rendered_names(fixed)) < 10

    def test_middleware_defaults_to_the_derived_budget(self) -> None:
        assert SchemaPrimingMiddleware().token_budget is None


class TestPriority:
    def test_documented_tables_survive_over_bare_ones(self) -> None:
        # the lottery this replaces: the bare tables led the input, so they used to be
        # the ones that survived. selection is by documentation now, not by position.
        tables = [_bare(f"b{i:02d}") for i in range(20)] + [_rich(f"r{i}") for i in range(3)]
        block = _render_schema_block([_digest(tables)], budget=300)
        names = _rendered_names(block)
        assert names, "at least one table must always render"
        assert all(name.startswith("public.r") for name in names)
        assert "not shown here" in block

    def test_selection_is_independent_of_input_order(self) -> None:
        tables = [_bare(f"b{i:02d}") for i in range(20)] + [_rich(f"r{i}") for i in range(3)]
        forward = _render_schema_block([_digest(list(tables))], budget=300)
        backward = _render_schema_block([_digest(list(reversed(tables)))], budget=300)
        rotated = _render_schema_block([_digest(tables[7:] + tables[:7])], budget=300)
        assert forward == backward == rotated

    def test_output_is_byte_stable_across_shuffled_input(self) -> None:
        tables = [_rich(f"r{i}") for i in range(4)] + [_bare(f"b{i}") for i in range(4)] + [_hazardous("h0")]
        shuffled = Random(1234)  # seeded so a failure is reproducible
        orders = []
        for _ in range(5):
            permutation = list(tables)
            shuffled.shuffle(permutation)
            orders.append(_render_schema_block([_digest(permutation)]))
        assert len(set(orders)) == 1

    def test_hazard_tables_outrank_richer_documentation(self) -> None:
        # an unloaded column reads as a measured 0; nothing in the live catalog says
        # otherwise, so that warning outranks even a heavily documented table.
        tables = [_rich("r0"), _rich("r1"), _hazardous("h0")]
        names = _rendered_names(_render_schema_block([_digest(tables)]))
        assert names[0] == "public.h0"

    def test_caveats_count_as_hazard_documentation(self) -> None:
        # forward-compatible: the projection carries no caveats today, but when it does
        # they rank with the unloaded-column overlay rather than falling to the tail.
        with_caveats = _bare("c0") | {"caveats": "never sum across geo_level"}
        names = _rendered_names(_render_schema_block([_digest([_rich("r0"), with_caveats])]))
        assert names[0] == "public.c0"

    def test_equal_documentation_is_tie_broken_by_name(self) -> None:
        tables = [_rich("zebra"), _rich("alpha"), _rich("mango")]
        names = _rendered_names(_render_schema_block([_digest(tables)]))
        assert names == ["public.alpha", "public.mango", "public.zebra"]

    def test_tables_from_every_digest_are_ranked_together(self) -> None:
        # ranking spans datasources: a rich table on the second digest beats a bare one
        # on the first, which per-digest ordering would have preferred.
        block = _render_schema_block([_digest([_bare("b0")]), _digest([_rich("r0")])])
        assert _rendered_names(block) == ["public.r0", "public.b0"]


class TestSyncMirror:
    def test_sync_wrap_model_call_passes_through(self) -> None:
        mw = SchemaPrimingMiddleware()
        req = _request(SystemMessage(content="base"))
        captured: dict[str, ModelRequest] = {}

        def _handler(r: ModelRequest) -> Any:
            captured["req"] = r
            return SimpleNamespace(result=[])

        integration = _StubIntegration(["ds-1"], {"ds-1": _digest([_TABLE])})
        with _configured({"schema_priming_integration": integration}):
            out = mw.wrap_model_call(req, _handler)
        assert captured["req"] is req  # un-primed
        assert not isinstance(out, ExtendedModelResponse)


class TestShape:
    def test_is_agent_middleware(self) -> None:
        m = SchemaPrimingMiddleware()
        assert isinstance(m, AgentMiddleware)
        assert m.name == "SchemaPrimingMiddleware"
        assert hasattr(m, "awrap_model_call")
        assert hasattr(m, "wrap_model_call")

    def test_state_schema_declares_metadata_channel(self) -> None:
        # the metadata stash only persists if the state schema declares the channel.
        assert SchemaPrimingMiddleware().state_schema is SchemaPrimingState
        assert "metadata" in SchemaPrimingState.__annotations__

    def test_token_budget_is_configurable(self) -> None:
        assert SchemaPrimingMiddleware(token_budget=99).token_budget == 99


# --------------------------------------------------------------------------- #
# real create_agent regression: the crash guard the stub-model tests can't see
# --------------------------------------------------------------------------- #
class _CapturingChatModel(BaseChatModel):
    """fake chat model that records the messages every model call receives.

    Drives a REAL ``create_agent`` so the prompt-assembly path
    (``[request.system_message, *messages]``) is exercised end to end -- the path
    the direct-drive stub tests never reach.

    :ivar sink: accumulates one message-list snapshot per model call.
    """

    sink: list[list[BaseMessage]]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Record the received messages and return a fixed no-tool-call reply.

        :param messages: the assembled prompt for this model call.
        :ptype messages: list[BaseMessage]
        :param stop: stop sequences (unused).
        :ptype stop: list[str] | None
        :param run_manager: callback manager (unused).
        :ptype run_manager: Any
        :return: a single-generation result ending the agent loop.
        :rtype: ChatResult
        """
        self.sink.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    @property
    def _llm_type(self) -> str:
        """Return the model type discriminator.

        :return: fixed type string.
        :rtype: str
        """
        return "capturing-fake"


class TestRealAgentRegression:
    def test_model_receives_exactly_one_leading_system_message(self) -> None:
        model = _CapturingChatModel(sink=[])
        integration = _StubIntegration(["ds-1"], {"ds-1": _digest([_TABLE])})
        agent = create_agent(
            model=model,
            tools=[],
            system_prompt="base prompt",
            middleware=[SchemaPrimingMiddleware()],
            state_schema=SchemaPrimingState,
        )
        final = asyncio.run(
            agent.ainvoke(
                {"messages": [HumanMessage(content="who are the active users")]},
                config={"configurable": {"schema_priming_integration": integration}},
            )
        )
        # the model was called; inspect the FIRST call's assembled prompt.
        assert model.sink, "the model was never called"
        first_call = model.sink[0]
        systems = [m for m in first_call if isinstance(m, SystemMessage)]
        # EXACTLY ONE system message, and it is FIRST -> no non-consecutive system
        # message (which langchain_anthropic would reject).
        assert len(systems) == 1
        assert isinstance(first_call[0], SystemMessage)
        assert "Ground every answer in tool results" in str(systems[0].content)
        assert "# Documented schema" in str(systems[0].content)
        assert "base prompt" in str(systems[0].content)
        # the block survived onto the metadata channel via the Command update.
        assert "# Documented schema" in final.get("metadata", {}).get("documented_schema_block", "")
