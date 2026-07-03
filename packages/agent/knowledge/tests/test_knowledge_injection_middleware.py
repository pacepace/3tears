"""Behavioral tests for :class:`KnowledgeInjectionMiddleware`.

Verifies the ``awrap_model_call`` seam: retrieved governed knowledge is FOLDED into
``request.system_message`` (never appended as a second, non-consecutive
``SystemMessage`` -- the pattern that crashes LangChain's Anthropic binding), the
rendered block + shadow ledgers are persisted onto ``metadata`` via a returned
:class:`~langchain.agents.middleware.types.ExtendedModelResponse` ``Command``, and a
missing integration / missing verified ``call_context`` / nothing-retrieved /
retrieval-fault all pass the call through un-merged. A REAL ``create_agent`` +
capturing fake model regression proves the model receives exactly one leading
``SystemMessage`` (the crash guard the stub-model unit tests cannot see).

The integration + verified identity are read off ``config["configurable"]``; the
direct-drive tests set the runnable-config contextvar so the middleware's
``get_config`` sees them, and the real-agent test passes them via ``ainvoke``'s
``config``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid7

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables.config import var_child_runnable_config
from threetears.knowledge import ConceptEffective, ConceptSnapshot, EntryEffective, EntrySnapshot, Scope

from threetears.agent.knowledge.integration import KnowledgeIntegration
from threetears.agent.knowledge.middleware import (
    KnowledgeInjectionMiddleware,
    KnowledgeInjectionState,
    _concept_shadow_disclosures,
    _cosine_similarity,
    _entry_shadow_disclosures,
    _estimate_tokens,
    _rank_and_trim_shared,
    _render_block,
    _render_entry,
    _split_invariant_concepts,
    _split_invariant_entries,
)


# --------------------------------------------------------------------------- #
# snapshot / effective-view builders
# --------------------------------------------------------------------------- #
def _entry_snapshot(*, title: str = "Filter", body: str = "use active", always_inject: bool = False) -> EntrySnapshot:
    """Build an :class:`EntrySnapshot` with a fresh id for merge input."""
    return EntrySnapshot(
        id=uuid7(),
        scope=Scope.PLATFORM,
        title=title,
        body=body,
        always_inject=always_inject,
        datasource_id=None,
    )


def _concept_snapshot(
    *,
    name: str = "active users",
    definition: str = "seen in last 30 days",
    always_inject: bool = False,
) -> ConceptSnapshot:
    """Build a :class:`ConceptSnapshot` with a fresh id for merge input."""
    return ConceptSnapshot(
        id=uuid7(),
        scope=Scope.PLATFORM,
        name=name,
        definition=definition,
        always_inject=always_inject,
    )


def _entry_effective(*, always_inject: bool = False, shadows: Scope | None = None) -> EntryEffective:
    """Build an :class:`EntryEffective` directly (bypassing the merge)."""
    return EntryEffective(entry=_entry_snapshot(always_inject=always_inject), shadows_scope=shadows)


def _concept_effective(
    *,
    always_inject: bool = False,
    shadows: Scope | None = None,
    ambiguous: bool = False,
) -> ConceptEffective:
    """Build a :class:`ConceptEffective` directly (bypassing the merge)."""
    return ConceptEffective(
        concept=_concept_snapshot(always_inject=always_inject),
        shadows_scope=shadows,
        ambiguous=ambiguous,
    )


# --------------------------------------------------------------------------- #
# stub collections + integration wiring
# --------------------------------------------------------------------------- #
class _StubEntryCollection:
    """opaque entry collection exposing the surface ``retrieve_entries`` reads."""

    def __init__(self, snapshots: Sequence[EntrySnapshot] = (), *, raises: bool = False) -> None:
        self._snapshots = list(snapshots)
        self._raises = raises

    async def list_visible_to_user(
        self,
        user_id: Any,
        *,
        datasource_id: Any = None,
        customer_scope: Any,
    ) -> list[EntrySnapshot]:
        if self._raises:
            raise RuntimeError("entry backend boom")
        return list(self._snapshots)

    async def fetch_embeddings(self, ids: Any, *, customer_scope: Any) -> dict[Any, Any]:
        return {}


class _StubConceptCollection:
    """opaque concept collection exposing the surface ``retrieve_concepts`` reads."""

    def __init__(self, snapshots: Sequence[ConceptSnapshot] = (), *, raises: bool = False) -> None:
        self._snapshots = list(snapshots)
        self._raises = raises

    async def list_visible_to_user(
        self,
        user_id: Any,
        *,
        datasource_id: Any = None,
        datasource_table_id: Any = None,
        customer_scope: Any,
    ) -> list[ConceptSnapshot]:
        if self._raises:
            raise RuntimeError("concept backend boom")
        return list(self._snapshots)

    async def fetch_embeddings(self, ids: Any, *, customer_scope: Any) -> dict[Any, Any]:
        return {}


class _RaisingCallContext:
    """verified-identity stand-in whose ``customer_id`` access raises (soft-fail path)."""

    @property
    def user_id(self) -> Any:
        return uuid7()

    @property
    def customer_id(self) -> Any:
        raise RuntimeError("identity boom")


def _integration(
    entries: Sequence[EntrySnapshot] = (),
    concepts: Sequence[ConceptSnapshot] = (),
    *,
    entry_raises: bool = False,
    concept_raises: bool = False,
) -> KnowledgeIntegration:
    """Build a real :class:`KnowledgeIntegration` over stub collections."""
    return KnowledgeIntegration(
        entry_collection=_StubEntryCollection(entries, raises=entry_raises),
        concept_collection=_StubConceptCollection(concepts, raises=concept_raises),
        embedding_model=None,
    )


def _call_context() -> Any:
    """Build a verified per-call identity exposing user/customer ids."""
    return SimpleNamespace(user_id=uuid7(), customer_id=uuid7())


def _configurable(integration: KnowledgeIntegration, *, with_call_context: bool = True) -> dict[str, Any]:
    """Build a configurable carrying a knowledge integration + verified identity."""
    cfg: dict[str, Any] = {"knowledge_integration": integration}
    if with_call_context:
        cfg["call_context"] = _call_context()
    return cfg


@contextmanager
def _configured(configurable: dict[str, Any]) -> Iterator[None]:
    """Set the runnable-config contextvar so ``get_config`` sees ``configurable``."""
    token = var_child_runnable_config.set({"configurable": configurable})
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _request(system: SystemMessage | None) -> ModelRequest:
    """Build a minimal ``ModelRequest`` for the direct-drive tests."""
    return ModelRequest(
        model=cast("BaseChatModel", SimpleNamespace()),
        messages=[HumanMessage(content="who are the active users")],
        system_message=system,
    )


def _drive(
    mw: KnowledgeInjectionMiddleware,
    request: ModelRequest,
    configurable: dict[str, Any],
) -> tuple[ModelRequest, Any]:
    """Drive ``awrap_model_call``; return the request the handler saw + the result."""
    captured: dict[str, ModelRequest] = {}

    async def _handler(req: ModelRequest) -> Any:
        captured["req"] = req
        return SimpleNamespace(result=[AIMessage(content="ok")])

    async def _run() -> Any:
        with _configured(configurable):
            return await mw.awrap_model_call(request, _handler)

    out = asyncio.run(_run())
    return captured["req"], out


class TestInjection:
    def test_folds_block_into_system_message(self) -> None:
        integration = _integration(
            entries=[_entry_snapshot(title="Filter deleted", body="exclude deleted rows")],
            concepts=[_concept_snapshot(name="active users", definition="seen in last 30 days")],
        )
        req, out = _drive(
            KnowledgeInjectionMiddleware(), _request(SystemMessage(content="base")), _configurable(integration)
        )
        assert req.system_message is not None
        content = req.system_message.content
        assert isinstance(content, str)
        # folded into the SINGLE system message, base first then the governed block.
        assert content.startswith("base\n\n# Governed data knowledge")
        assert "active users" in content
        assert "Filter deleted" in content

    def test_returns_extended_response_with_metadata(self) -> None:
        integration = _integration(
            entries=[_entry_snapshot()],
            concepts=[_concept_snapshot()],
        )
        _req, out = _drive(
            KnowledgeInjectionMiddleware(), _request(SystemMessage(content="base")), _configurable(integration)
        )
        assert isinstance(out, ExtendedModelResponse)
        assert out.command is not None
        update = out.command.update
        assert isinstance(update, dict)
        metadata = update["metadata"]
        assert metadata["governed_knowledge_block"].startswith("# Governed data knowledge")
        assert len(metadata["knowledge_injected_entries"]) == 1
        assert len(metadata["knowledge_injected_concepts"]) == 1

    def test_glossary_renders_before_procedures(self) -> None:
        integration = _integration(entries=[_entry_snapshot()], concepts=[_concept_snapshot()])
        req, _out = _drive(
            KnowledgeInjectionMiddleware(), _request(SystemMessage(content="base")), _configurable(integration)
        )
        content = req.system_message.content
        assert isinstance(content, str)
        # definitions (concept glossary) before procedures (entries).
        assert content.index("## Data concepts") < content.index("## Data knowledge")

    def test_no_base_system_message_becomes_block(self) -> None:
        integration = _integration(entries=[_entry_snapshot()])
        req, _out = _drive(KnowledgeInjectionMiddleware(), _request(None), _configurable(integration))
        assert req.system_message is not None
        assert isinstance(req.system_message.content, str)
        assert req.system_message.content.startswith("# Governed data knowledge")


class TestNoop:
    def test_noop_without_integration(self) -> None:
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(KnowledgeInjectionMiddleware(), req_in, {})
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_noop_without_call_context(self) -> None:
        integration = _integration(entries=[_entry_snapshot()])
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(KnowledgeInjectionMiddleware(), req_in, {"knowledge_integration": integration})
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_noop_when_nothing_retrieved(self) -> None:
        integration = _integration()  # empty collections
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(KnowledgeInjectionMiddleware(), req_in, _configurable(integration))
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_noop_outside_runnable_context(self) -> None:
        # no contextvar set -> get_config raises -> soft-fail to pass-through.
        mw = KnowledgeInjectionMiddleware()
        req = _request(SystemMessage(content="base"))
        captured: dict[str, ModelRequest] = {}

        async def _handler(r: ModelRequest) -> Any:
            captured["req"] = r
            return SimpleNamespace(result=[])

        out = asyncio.run(mw.awrap_model_call(req, _handler))
        assert captured["req"] is req
        assert not isinstance(out, ExtendedModelResponse)


class TestSoftFail:
    def test_retrieval_fault_passes_through(self) -> None:
        # both backends raise -> retrieve_* soft-fail internally to ([], []) ->
        # nothing retrieved -> the call proceeds on the un-merged request.
        integration = _integration(entry_raises=True, concept_raises=True)
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(KnowledgeInjectionMiddleware(), req_in, _configurable(integration))
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_identity_fault_passes_through(self) -> None:
        # reading the verified identity raises INSIDE the middleware try -> the
        # seam swallows it and proceeds on the un-merged request.
        integration = _integration(entries=[_entry_snapshot()])
        cfg = {"knowledge_integration": integration, "call_context": _RaisingCallContext()}
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(KnowledgeInjectionMiddleware(), req_in, cfg)
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)


class TestSyncMirror:
    def test_sync_wrap_model_call_passes_through(self) -> None:
        mw = KnowledgeInjectionMiddleware()
        req = _request(SystemMessage(content="base"))
        captured: dict[str, ModelRequest] = {}

        def _handler(r: ModelRequest) -> Any:
            captured["req"] = r
            return SimpleNamespace(result=[])

        with _configured(_configurable(_integration(entries=[_entry_snapshot()]))):
            out = mw.wrap_model_call(req, _handler)
        assert captured["req"] is req  # un-merged
        assert not isinstance(out, ExtendedModelResponse)


class TestShape:
    def test_is_agent_middleware(self) -> None:
        m = KnowledgeInjectionMiddleware()
        assert isinstance(m, AgentMiddleware)
        assert m.name == "KnowledgeInjectionMiddleware"
        assert hasattr(m, "awrap_model_call")
        assert hasattr(m, "wrap_model_call")

    def test_state_schema_declares_metadata_channel(self) -> None:
        # the governed-block ledger only persists if the state schema declares the
        # channel (with the merge reducer).
        assert KnowledgeInjectionMiddleware().state_schema is KnowledgeInjectionState
        assert "metadata" in KnowledgeInjectionState.__annotations__


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
        integration = _integration(
            entries=[_entry_snapshot(title="Filter deleted", body="exclude deleted rows")],
            concepts=[_concept_snapshot(name="active users", definition="seen in last 30 days")],
        )
        agent = create_agent(
            model=model,
            tools=[],
            system_prompt="base prompt",
            middleware=[KnowledgeInjectionMiddleware()],
            state_schema=KnowledgeInjectionState,
        )
        final = asyncio.run(
            agent.ainvoke(
                {"messages": [HumanMessage(content="who are the active users")]},
                config={"configurable": _configurable(integration)},
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
        assert "# Governed data knowledge" in str(systems[0].content)
        assert "base prompt" in str(systems[0].content)
        # the ledger survived onto the metadata channel via the Command update.
        assert final.get("metadata", {}).get("governed_knowledge_block", "").startswith("# Governed data knowledge")


# --------------------------------------------------------------------------- #
# render / trim / cosine helper unit tests
# --------------------------------------------------------------------------- #
class TestRenderAndTrim:
    def test_split_invariants(self) -> None:
        views = [_entry_effective(always_inject=True), _entry_effective(always_inject=False)]
        inv, sit = _split_invariant_entries(views)
        assert len(inv) == 1
        assert len(sit) == 1

    def test_split_invariant_concepts(self) -> None:
        views = [_concept_effective(always_inject=True), _concept_effective(always_inject=False)]
        inv, sit = _split_invariant_concepts(views)
        assert len(inv) == 1
        assert len(sit) == 1

    def test_render_block_orders_glossary_before_procedures(self) -> None:
        block = _render_block(
            invariant_concepts=[_concept_effective(always_inject=True)],
            situational_concepts=[],
            invariant_entries=[_entry_effective(always_inject=True)],
            situational_entries=[],
        )
        assert block.startswith("# Governed data knowledge")
        assert block.index("Data concepts") < block.index("Data knowledge")

    def test_render_block_empty_when_no_views(self) -> None:
        assert (
            _render_block(
                invariant_concepts=[],
                situational_concepts=[],
                invariant_entries=[],
                situational_entries=[],
            )
            == ""
        )

    def test_rank_and_trim_tiny_budget_keeps_nothing(self) -> None:
        # the situational tail is fully trimmable (unlike invariants): a budget below
        # the first item's cost keeps NOTHING.
        entries = [_entry_effective() for _ in range(3)]
        kept_c, kept_e = _rank_and_trim_shared(
            situational_concepts=[],
            situational_entries=entries,
            budget=1,
        )
        assert kept_c == []
        assert kept_e == []

    def test_rank_and_trim_partial_budget_trims_the_tail(self) -> None:
        # a budget for exactly two items keeps two of five (greedy prefix, then break).
        entries = [_entry_effective() for _ in range(5)]
        per_item = _estimate_tokens(_render_entry(entries[0]))
        _kept_c, kept_e = _rank_and_trim_shared(
            situational_concepts=[],
            situational_entries=entries,
            budget=per_item * 2,
        )
        assert len(kept_e) == 2

    def test_rank_and_trim_keeps_all_under_generous_budget(self) -> None:
        entries = [_entry_effective() for _ in range(3)]
        _kept_c, kept_e = _rank_and_trim_shared(
            situational_concepts=[],
            situational_entries=entries,
            budget=10_000,
        )
        assert len(kept_e) == 3


class TestCosine:
    def test_identical_vectors_score_one(self) -> None:
        assert _cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0

    def test_empty_or_mismatched_score_zero(self) -> None:
        assert _cosine_similarity([], [1.0]) == 0.0
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestShadowLedgers:
    def test_entry_shadow_disclosure_recorded(self) -> None:
        views = [_entry_effective(shadows=Scope.PLATFORM), _entry_effective(shadows=None)]
        ledger = _entry_shadow_disclosures(views)
        assert len(ledger) == 1
        assert ledger[0]["shadows_scope"] == "platform"
        assert "entry_id" in ledger[0]
        assert "title" in ledger[0]

    def test_concept_shadow_and_ambiguity_recorded(self) -> None:
        views = [
            _concept_effective(shadows=Scope.CUSTOMER),
            _concept_effective(ambiguous=True),
            _concept_effective(),
        ]
        ledger = _concept_shadow_disclosures(views)
        assert len(ledger) == 2
        by_scope = {d["shadows_scope"] for d in ledger}
        assert "customer" in by_scope
        assert any(d["ambiguous"] == "true" for d in ledger)
