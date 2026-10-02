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

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables.config import var_child_runnable_config
from threetears.knowledge import ConceptEffective, ConceptSnapshot, EntryEffective, EntrySnapshot, Scope

from threetears.agent.knowledge.integration import (
    GovernedKnowledgeUnavailableError,
    KnowledgeIntegration,
)
from threetears.agent.knowledge.middleware import (
    GovernedKnowledgeRenderError,
    KnowledgeInjectionMiddleware,
    KnowledgeInjectionState,
)

#: the situational-budget contract, stated as the numbers it was measured at (see
#: the middleware's budget notes): a 3500-token floor, 175 tokens per situational
#: candidate, and a 16000-token ceiling. a change to any of them is a deliberate
#: re-measurement, so it lands here too.
BUDGET_FLOOR = 3500
TOKENS_PER_CANDIDATE = 175
BUDGET_CEILING = 16000

#: the four section headers the block renders, glossary before procedures.
INVARIANT_GLOSSARY = "## Data concepts that ALWAYS apply"
SITUATIONAL_GLOSSARY = "## Data concepts relevant to this question"
INVARIANT_PROCEDURES = "## Data knowledge that ALWAYS applies"
SITUATIONAL_PROCEDURES = "## Data knowledge relevant to this question"


def _rendered_tokens(snapshot: EntrySnapshot) -> int:
    """estimate what one unshadowed entry costs the budget: its rendered text over four chars a token."""
    return len(f"### {snapshot.title}\n{snapshot.body}") // 4


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


def _sized_entry_snapshot(index: int, *, body_chars: int = 800) -> EntrySnapshot:
    """Build a situational entry with a uniform, predictable rendered token cost.

    The default entry builders render to a handful of tokens, which no realistic
    budget ever trims; the corpus-scaling tests need items whose cost is the same
    order as a real authored procedure so the budget is what decides the cut.
    """
    return EntrySnapshot(
        id=uuid7(),
        scope=Scope.PLATFORM,
        title=f"Rule {index:02d}",
        body="x" * body_chars,
        always_inject=False,
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


class _BoomEntrySnapshot:
    """Entry snapshot whose ``title`` access raises, to force a single-item render fault.

    Every other attribute is a real :class:`EntrySnapshot`'s, so the shared merge,
    the invariant split and the stable-order sort all see a well-formed row; only
    rendering -- which reads ``title`` first -- faults.
    """

    def __init__(self, *, always_inject: bool = False) -> None:
        self.real = _entry_snapshot(always_inject=always_inject)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)

    @property
    def title(self) -> str:
        """Raise to simulate a render fault on this one item.

        :return: never returns.
        :rtype: str
        :raises RuntimeError: always.
        """
        raise RuntimeError("entry render boom")


def _boom_entry(*, always_inject: bool = False) -> EntrySnapshot:
    """Build an entry snapshot whose render raises on the ``title`` access."""
    return cast("EntrySnapshot", _BoomEntrySnapshot(always_inject=always_inject))


class _FixedEmbedder:
    """embedding model that embeds every turn query as one fixed vector."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    async def aembed_query(self, text: str) -> list[float]:
        _ = text
        return list(self.vector)


# --------------------------------------------------------------------------- #
# stub collections + integration wiring
# --------------------------------------------------------------------------- #
class _StubEntryCollection:
    """opaque entry collection exposing the surface ``retrieve_entries`` reads."""

    def __init__(
        self,
        snapshots: Sequence[EntrySnapshot] = (),
        *,
        raises: bool = False,
        embeddings: dict[Any, list[float]] | None = None,
    ) -> None:
        self._snapshots = list(snapshots)
        self._raises = raises
        self._embeddings = embeddings or {}

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
        return {i: self._embeddings[i] for i in ids if i in self._embeddings}


class _StubConceptCollection:
    """opaque concept collection exposing the surface ``retrieve_concepts`` reads."""

    def __init__(
        self,
        snapshots: Sequence[ConceptSnapshot] = (),
        *,
        raises: bool = False,
        embeddings: dict[Any, list[float]] | None = None,
    ) -> None:
        self._snapshots = list(snapshots)
        self._raises = raises
        self._embeddings = embeddings or {}

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
        return {i: self._embeddings[i] for i in ids if i in self._embeddings}


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
    embedding_model: Any = None,
    embeddings: dict[Any, list[float]] | None = None,
) -> KnowledgeIntegration:
    """Build a real :class:`KnowledgeIntegration` over stub collections.

    ``embeddings`` is the stored-vector table both stub collections answer
    ``fetch_embeddings`` from; ``embedding_model`` embeds the turn query.
    """
    return KnowledgeIntegration(
        entry_collection=_StubEntryCollection(entries, raises=entry_raises, embeddings=embeddings),
        concept_collection=_StubConceptCollection(concepts, raises=concept_raises, embeddings=embeddings),
        embedding_model=embedding_model,
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


def _injected_entry_count(out: Any) -> int:
    """Count the entries a drive actually injected, off the metadata ledger."""
    assert isinstance(out, ExtendedModelResponse)
    assert out.command is not None
    update = out.command.update
    assert isinstance(update, dict)
    return len(update["metadata"]["knowledge_injected_entries"])


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
    def test_retrieval_fault_fails_closed(self) -> None:
        """THIS TEST PREVIOUSLY ASSERTED THE BUG.

        It required a retrieval fault to pass through on the un-merged request --
        the turn proceeding with NO governed knowledge at all, and nothing
        anywhere saying so. That is the fail-open the governance layer exists to
        prevent; ``GovernedKnowledgeRenderError`` already refuses it for an
        invariant that failed to RENDER, and a fault that happens one step
        earlier is the same hole.

        Seen live: an L3 timeout produced exactly this, and the answer was
        indistinguishable from a governed one.
        """
        integration = _integration(entry_raises=True, concept_raises=True)
        req_in = _request(SystemMessage(content="base"))
        with pytest.raises(GovernedKnowledgeUnavailableError):
            _drive(KnowledgeInjectionMiddleware(), req_in, _configurable(integration))

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
# split / trim / budget / ranking / render, driven through the middleware
# --------------------------------------------------------------------------- #
def _run_turn(
    *,
    entries: Sequence[EntrySnapshot] = (),
    concepts: Sequence[ConceptSnapshot] = (),
    token_budget: int | None = None,
    embedding_model: Any = None,
    embeddings: dict[Any, list[float]] | None = None,
) -> tuple[ModelRequest, Any]:
    """Drive one turn through the middleware over the given governed rows."""
    return _drive(
        KnowledgeInjectionMiddleware(token_budget=token_budget),
        _request(SystemMessage(content="base")),
        _configurable(
            _integration(
                entries=entries,
                concepts=concepts,
                embedding_model=embedding_model,
                embeddings=embeddings,
            )
        ),
    )


def _system_text(req: ModelRequest) -> str:
    """The system prompt the model received."""
    assert req.system_message is not None
    content = req.system_message.content
    assert isinstance(content, str)
    return content


def _metadata(out: Any) -> dict[str, Any]:
    """The metadata ledger a drive persisted."""
    assert isinstance(out, ExtendedModelResponse)
    assert out.command is not None
    update = out.command.update
    assert isinstance(update, dict)
    metadata = update["metadata"]
    assert isinstance(metadata, dict)
    return metadata


def _warnings_naming(caplog: pytest.LogCaptureFixture, needle: str) -> list[str]:
    """WARNING lines whose message contains ``needle``."""
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and needle in r.getMessage()]


def _all_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def _huge_invariant(tokens: int) -> EntrySnapshot:
    """An always-inject entry costing at least ``tokens`` -- enough to trip the D9 overflow line."""
    return _entry_snapshot(title="Huge", body="x" * (tokens * 4 + 4), always_inject=True)


class TestInvariantSplit:
    def test_entries_split_into_always_and_relevant_sections(self) -> None:
        req, out = _run_turn(
            entries=[
                _entry_snapshot(title="Hard rule", always_inject=True),
                _entry_snapshot(title="Soft rule", always_inject=False),
            ]
        )
        content = _system_text(req)
        assert content.index(INVARIANT_PROCEDURES) < content.index("Hard rule") < content.index(SITUATIONAL_PROCEDURES)
        assert content.index(SITUATIONAL_PROCEDURES) < content.index("Soft rule")
        assert len(_metadata(out)["knowledge_injected_entries"]) == 2

    def test_concepts_split_into_always_and_relevant_sections(self) -> None:
        req, _out = _run_turn(
            concepts=[
                _concept_snapshot(name="hard term", always_inject=True),
                _concept_snapshot(name="soft term", always_inject=False),
            ]
        )
        content = _system_text(req)
        assert content.index(INVARIANT_GLOSSARY) < content.index("hard term") < content.index(SITUATIONAL_GLOSSARY)
        assert content.index(SITUATIONAL_GLOSSARY) < content.index("soft term")


class TestRenderAndTrim:
    def test_render_orders_glossary_before_procedures(self) -> None:
        req, _out = _run_turn(
            entries=[_entry_snapshot(always_inject=True)],
            concepts=[_concept_snapshot(always_inject=True)],
        )
        content = _system_text(req)
        assert content.startswith("base\n\n# Governed data knowledge")
        assert content.index(INVARIANT_GLOSSARY) < content.index(INVARIANT_PROCEDURES)

    def test_nothing_left_after_the_trim_passes_the_call_through(self) -> None:
        # every retrieved item is situational and a zero budget keeps none of them,
        # so the rendered block is empty and the request goes through un-merged.
        req_in = _request(SystemMessage(content="base"))
        req_out, out = _drive(
            KnowledgeInjectionMiddleware(token_budget=0),
            req_in,
            _configurable(_integration(entries=[_entry_snapshot(), _entry_snapshot()])),
        )
        assert req_out is req_in
        assert not isinstance(out, ExtendedModelResponse)

    def test_tiny_budget_keeps_no_situational_item(self) -> None:
        # the situational tail is fully trimmable (unlike invariants): a budget below
        # the first item's cost keeps NOTHING of it.
        req, out = _run_turn(
            entries=[_entry_snapshot(title="Kept hard", always_inject=True)]
            + [_entry_snapshot(title=f"Dropped {i}") for i in range(3)],
            token_budget=1,
        )
        content = _system_text(req)
        assert "Kept hard" in content
        assert "Dropped" not in content
        assert SITUATIONAL_PROCEDURES not in content
        assert len(_metadata(out)["knowledge_injected_entries"]) == 1

    def test_partial_budget_trims_the_tail(self) -> None:
        # a budget for two and a half items keeps two of five (greedy prefix, then break).
        entries = [_sized_entry_snapshot(i) for i in range(5)]
        _req, out = _run_turn(entries=entries, token_budget=_rendered_tokens(entries[0]) * 5 // 2)
        assert _injected_entry_count(out) == 2

    def test_generous_budget_keeps_all(self) -> None:
        _req, out = _run_turn(entries=[_entry_snapshot() for _ in range(3)], token_budget=10_000)
        assert _injected_entry_count(out) == 3


class TestBudgetScalesWithCorpus:
    """The situational budget grows with the corpus instead of sitting at a constant.

    A constant budget makes every item authored past its capacity evict another
    item from the SAME turn: ots authored 90 situational items and a flat 3500 kept
    14 to 25 of them per turn, so the same eval suite scored 48-50/51 with a
    different set of cases failing each run. Documenting more made the agent know
    less, which is the opposite of what the FIX_PROTOCOL asks for.

    The effective budget is read where an operator reads it: the D9 overflow line
    and the shared-trim drop line both name the budget that actually ran.
    """

    def test_empty_pool_keeps_the_floor(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            _run_turn(entries=[_huge_invariant(BUDGET_FLOOR + 100)])
        (line,) = _warnings_naming(caplog, "invariant set alone exceeds situational budget")
        assert f"budget={BUDGET_FLOOR})" in line

    def test_small_corpus_keeps_the_floor(self, caplog: pytest.LogCaptureFixture) -> None:
        # a corpus too small to fill the floor gains nothing from scaling, and the
        # floor is the value the 50/51 correctness evidence was earned on.
        with caplog.at_level("WARNING"):
            _run_turn(entries=[_huge_invariant(BUDGET_FLOOR + 100)] + [_entry_snapshot() for _ in range(3)])
        (line,) = _warnings_naming(caplog, "invariant set alone exceeds situational budget")
        assert f"budget={BUDGET_FLOOR})" in line

    def test_large_corpus_scales_above_the_floor(self, caplog: pytest.LogCaptureFixture) -> None:
        scaled = 90 * TOKENS_PER_CANDIDATE
        assert BUDGET_FLOOR < scaled <= BUDGET_CEILING
        with caplog.at_level("WARNING"):
            _run_turn(entries=[_sized_entry_snapshot(i) for i in range(90)])
        (line,) = _warnings_naming(caplog, "shared trim dropped")
        assert f"budget={scaled} tokens" in line

    def test_ceiling_caps_an_unbounded_corpus(self, caplog: pytest.LogCaptureFixture) -> None:
        # unbounded scaling is a context bomb: without the cap a runaway corpus
        # puts the whole library in every system prompt on every turn.
        assert 200 * TOKENS_PER_CANDIDATE > BUDGET_CEILING
        with caplog.at_level("WARNING"):
            _run_turn(entries=[_sized_entry_snapshot(i) for i in range(200)])
        (line,) = _warnings_naming(caplog, "shared trim dropped")
        assert f"budget={BUDGET_CEILING} tokens" in line

    def test_explicit_budget_wins_over_scaling(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            _run_turn(entries=[_sized_entry_snapshot(i) for i in range(200)], token_budget=512)
        (line,) = _warnings_naming(caplog, "shared trim dropped")
        assert "budget=512 tokens" in line

    def test_explicit_zero_budget_is_honoured_not_treated_as_unset(self, caplog: pytest.LogCaptureFixture) -> None:
        # zero is a real deployment setting (invariants only), which is why the
        # "scale me" signal is None and not an in-range sentinel.
        with caplog.at_level("WARNING"):
            _run_turn(
                entries=[_entry_snapshot(always_inject=True)] + [_sized_entry_snapshot(i) for i in range(90)],
                token_budget=0,
            )
        (line,) = _warnings_naming(caplog, "shared trim dropped")
        assert "dropped 90 of 90" in line
        assert "budget=0 tokens" in line

    def test_constructor_default_defers_to_scaling(self) -> None:
        assert KnowledgeInjectionMiddleware().token_budget is None
        assert KnowledgeInjectionMiddleware(token_budget=99).token_budget == 99

    def test_default_admits_more_of_a_large_corpus_than_the_floor(self) -> None:
        snapshots = [_sized_entry_snapshot(i) for i in range(30)]
        _req, scaled_out = _run_turn(entries=snapshots)
        _req2, pinned_out = _run_turn(entries=snapshots, token_budget=BUDGET_FLOOR)
        assert _injected_entry_count(scaled_out) > _injected_entry_count(pinned_out)

    def test_explicit_budget_still_trims_a_large_corpus(self) -> None:
        # the override is honoured END TO END: a deployment that asks for a lean
        # cut gets one no matter how big the pool.
        snapshots = [_sized_entry_snapshot(i) for i in range(30)]
        _req, out = _run_turn(entries=snapshots, token_budget=_rendered_tokens(snapshots[0]) * 5 // 2)
        assert _injected_entry_count(out) == 2

    def test_scaling_counts_only_situational_candidates(self, caplog: pytest.LogCaptureFixture) -> None:
        # invariants inject in full regardless (D9) and never enter the trim pool,
        # so counting them would buy budget for items that do not spend it: 30
        # situational candidates scale to 30 * 175, not (30 + 40) * 175.
        invariants = [_entry_snapshot(title=f"Hard {i}", body="b", always_inject=True) for i in range(40)]
        situational = [_sized_entry_snapshot(i) for i in range(30)]
        assert sum(_rendered_tokens(s) for s in situational) > 30 * TOKENS_PER_CANDIDATE
        with caplog.at_level("WARNING"):
            _run_turn(entries=invariants + situational)
        (line,) = _warnings_naming(caplog, "shared trim dropped")
        assert f"budget={30 * TOKENS_PER_CANDIDATE} tokens" in line

    def test_invariants_inject_in_full_under_a_zero_budget(self) -> None:
        # the budget governs the situational tail ONLY; hard rules are exempt, so
        # even the leanest explicit budget cannot cull one.
        snapshots = [_entry_snapshot(title=f"Hard {i}", body="always apply", always_inject=True) for i in range(3)]
        req, out = _run_turn(entries=snapshots, token_budget=0)
        content = _system_text(req)
        for index in range(3):
            assert f"Hard {index}" in content
        assert _injected_entry_count(out) == 3

    def test_trim_drop_warning_names_the_effective_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        # the drop warning is how this bug was found; it stays honest only if it
        # reports the budget that actually did the cutting, not the floor.
        snapshots = [_sized_entry_snapshot(i) for i in range(30)]
        with caplog.at_level("WARNING"):
            _run_turn(entries=snapshots)
        drops = _warnings_naming(caplog, "shared trim dropped")
        assert drops
        assert f"budget={30 * TOKENS_PER_CANDIDATE}" in drops[0]


class TestSilentDegradationSignals:
    """SDS-02/03: every ranking soft-fail leaves a fingerprint a human can find.

    The stable-order fallback and the situational starvation both change what the
    model sees without failing the turn, which is exactly how a retrieval outage
    shaped every answer for a product's whole life and announced nothing.
    """

    def test_unembedded_query_warns_naming_candidate_count(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            _run_turn(entries=[_entry_snapshot() for _ in range(3)], token_budget=10_000)
        warnings = _all_warnings(caplog)
        assert len(warnings) == 1
        assert "fell back to stable order" in warnings[0]
        assert "did not embed" in warnings[0]
        assert "candidates=3" in warnings[0]

    def test_no_stored_vectors_warns_naming_the_cause(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            _run_turn(
                entries=[_entry_snapshot() for _ in range(2)],
                token_budget=10_000,
                embedding_model=_FixedEmbedder([1.0, 0.0]),
                embeddings={},
            )
        warnings = _all_warnings(caplog)
        assert len(warnings) == 1
        assert "none of the 2 situational candidates carry a stored embedding" in warnings[0]

    def test_healthy_ranking_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        entries = [_entry_snapshot() for _ in range(2)]
        with caplog.at_level("WARNING"):
            _run_turn(
                entries=entries,
                token_budget=10_000,
                embedding_model=_FixedEmbedder([1.0, 0.0]),
                embeddings={e.id: [1.0, 0.0] for e in entries},
            )
        assert _all_warnings(caplog) == []

    def test_empty_pool_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        # nothing situational was retrieved, so nothing degraded -- warning here would
        # fire on every turn of an agent that governs no situational knowledge at all.
        with caplog.at_level("WARNING"):
            _run_turn(entries=[_entry_snapshot(always_inject=True)], token_budget=10_000)
        assert _all_warnings(caplog) == []

    def test_partial_embedding_coverage_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        # one stored vector still ranks; only a TOTAL absence is the fallback.
        entries = [_entry_snapshot() for _ in range(2)]
        with caplog.at_level("WARNING"):
            _run_turn(
                entries=entries,
                token_budget=10_000,
                embedding_model=_FixedEmbedder([1.0, 0.0]),
                embeddings={entries[0].id: [1.0, 0.0]},
            )
        assert _all_warnings(caplog) == []

    def test_starvation_warns_with_counts(self, caplog: pytest.LogCaptureFixture) -> None:
        entries = [_sized_entry_snapshot(i, body_chars=4000) for i in range(4)]
        with caplog.at_level("WARNING"):
            _run_turn(entries=entries, token_budget=512)
        (line,) = _warnings_naming(caplog, "kept no situational entries")
        assert "candidates=4" in line
        assert "budget=512" in line

    def test_starvation_silent_when_something_survived(self, caplog: pytest.LogCaptureFixture) -> None:
        entries = [_sized_entry_snapshot(i) for i in range(4)]
        with caplog.at_level("WARNING"):
            _req, out = _run_turn(entries=entries, token_budget=_rendered_tokens(entries[0]) * 3 // 2)
        assert _injected_entry_count(out) == 1
        assert _warnings_naming(caplog, "kept no situational entries") == []

    def test_starvation_silent_when_nothing_was_offered(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            _run_turn(entries=[_entry_snapshot(always_inject=True)], token_budget=512)
        assert _warnings_naming(caplog, "kept no situational entries") == []

    def test_fallback_prefers_the_query_cause(self, caplog: pytest.LogCaptureFixture) -> None:
        # both causes hold at once (no query vector AND no stored vectors); the query
        # outage is the upstream one and is the only line logged, so a turn never
        # carries two lines for one degradation.
        with caplog.at_level("WARNING"):
            _run_turn(entries=[_entry_snapshot() for _ in range(5)], token_budget=10_000, embeddings={})
        (line,) = _warnings_naming(caplog, "fell back to stable order")
        assert "did not embed" in line


class TestRenderFaultIsolation:
    """One item's render fault must NOT drop the whole governed block.

    A governance layer that fails open on its own hard rules is not governance: a
    situational (best-effort) item's render fault is isolated to that item, while an
    invariant (always-inject) item's render fault fails CLOSED rather than silently
    proceeding ungoverned.
    """

    def test_situational_render_fault_skips_only_the_bad_item(self, caplog: pytest.LogCaptureFixture) -> None:
        # one situational entry booms on render; the surviving situational entry
        # (and the whole block) still reaches the agent -- the fault does NOT nuke
        # the block.
        with caplog.at_level("WARNING"):
            req, out = _run_turn(
                entries=[_boom_entry(), _entry_snapshot(title="Filter deleted", body="exclude deleted rows")]
            )
        content = _system_text(req)
        assert "# Governed data knowledge" in content
        assert "Filter deleted" in content
        assert _injected_entry_count(out) == 1
        assert _warnings_naming(caplog, "skipping this item")

    def test_situational_all_faulting_yields_no_section_not_a_crash(self) -> None:
        # if EVERY situational item faults, the section is simply empty (skipped),
        # never a raised exception -- best-effort context degrades to nothing.
        req, _out = _run_turn(entries=[_boom_entry(), _entry_snapshot(title="Hard", always_inject=True)])
        content = _system_text(req)
        assert "Hard" in content
        assert SITUATIONAL_PROCEDURES not in content

    def test_middleware_fails_closed_on_invariant_render_fault(self) -> None:
        # an invariant (always-inject) hard rule that cannot render must FAIL CLOSED:
        # silently dropping it would let the agent proceed ungoverned on a rule it
        # must always apply. the seam must not swallow the signal into a
        # pass-through either: the handler is never reached and the error surfaces.
        integration = _integration(entries=[_boom_entry(always_inject=True), _entry_snapshot()])
        mw = KnowledgeInjectionMiddleware()
        request = _request(SystemMessage(content="base"))
        handler_calls: list[ModelRequest] = []

        async def _handler(req: ModelRequest) -> Any:
            handler_calls.append(req)
            return SimpleNamespace(result=[AIMessage(content="ok")])

        async def _run() -> Any:
            with _configured(_configurable(integration)):
                return await mw.awrap_model_call(request, _handler)

        with pytest.raises(GovernedKnowledgeRenderError):
            asyncio.run(_run())
        assert handler_calls == []


class TestSimilarityRanking:
    """situational items rank by cosine similarity to the turn query, highest first."""

    def test_identical_direction_ranks_first(self) -> None:
        near = _entry_snapshot(title="Near", body="b")
        far = _entry_snapshot(title="Far", body="b")
        req, _out = _run_turn(
            entries=[far, near],
            token_budget=10_000,
            embedding_model=_FixedEmbedder([1.0, 2.0, 3.0]),
            embeddings={near.id: [1.0, 2.0, 3.0], far.id: [3.0, -2.0, 0.5]},
        )
        content = _system_text(req)
        assert content.index("### Near") < content.index("### Far")

    def test_empty_mismatched_and_zero_vectors_score_zero(self, caplog: pytest.LogCaptureFixture) -> None:
        # a mismatched dimension and a zero vector score 0.0 rather than crashing the
        # rank: below a positive match, above an opposed one.
        positive = _entry_snapshot(title="Positive", body="b")
        mismatched = _entry_snapshot(title="Mismatched", body="b")
        zero = _entry_snapshot(title="Zero", body="b")
        opposed = _entry_snapshot(title="Opposed", body="b")
        with caplog.at_level("WARNING"):
            req, out = _run_turn(
                entries=[opposed, zero, mismatched, positive],
                token_budget=10_000,
                embedding_model=_FixedEmbedder([1.0, 0.0]),
                embeddings={
                    positive.id: [1.0, 0.0],
                    mismatched.id: [1.0, 0.0, 0.0],
                    zero.id: [0.0, 0.0],
                    opposed.id: [-1.0, 0.0],
                },
            )
        content = _system_text(req)
        assert content.index("### Positive") < content.index("### Mismatched") < content.index("### Opposed")
        assert content.index("### Positive") < content.index("### Zero") < content.index("### Opposed")
        assert _injected_entry_count(out) == 4
        assert _warnings_naming(caplog, "wrong dimension")


class TestShadowLedgers:
    def test_entry_shadow_disclosure_recorded(self) -> None:
        platform = _entry_snapshot(title="Platform rule")
        override = EntrySnapshot(
            id=uuid7(),
            scope=Scope.CUSTOMER,
            title="Customer override",
            body="use archived too",
            always_inject=False,
            datasource_id=None,
            origin_entry_id=platform.id,
        )
        unrelated = _entry_snapshot(title="Unrelated")
        _req, out = _run_turn(entries=[platform, override, unrelated], token_budget=10_000)
        ledger = _metadata(out)["knowledge_shadow_disclosures"]
        assert len(ledger) == 1
        assert ledger[0]["shadows_scope"] == "platform"
        assert ledger[0]["entry_id"] == str(override.id)
        assert ledger[0]["title"] == "Customer override"

    def test_concept_shadow_and_ambiguity_recorded(self) -> None:
        platform = _concept_snapshot(name="revenue", definition="gross")
        override = ConceptSnapshot(
            id=uuid7(),
            scope=Scope.CUSTOMER,
            name="revenue",
            definition="net of refunds",
            always_inject=False,
            origin_concept_id=platform.id,
        )
        twin_a = _concept_snapshot(name="active users", definition="seen in 30 days")
        twin_b = _concept_snapshot(name="active users", definition="seen in 7 days")
        plain = _concept_snapshot(name="churn", definition="lost")
        _req, out = _run_turn(concepts=[platform, override, twin_a, twin_b, plain], token_budget=10_000)
        ledger = _metadata(out)["knowledge_concept_shadow_disclosures"]
        by_name = {d["name"] for d in ledger}
        assert "churn" not in by_name
        assert any(d["shadows_scope"] == "platform" and d["name"] == "revenue" for d in ledger)
        assert any(d["ambiguous"] == "true" and d["name"] == "active users" for d in ledger)


class TestTurnQueryText:
    """the situational ranker embeds what the person asked."""

    class _RecordingEmbedder:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def aembed_query(self, text: str) -> list[float]:
            self.queries.append(text)
            return [1.0, 0.0]

    def _embedded_query(self, messages: list[BaseMessage]) -> list[str]:
        embedder = self._RecordingEmbedder()
        _drive(
            KnowledgeInjectionMiddleware(),
            ModelRequest(
                model=cast("BaseChatModel", SimpleNamespace()),
                messages=messages,
                system_message=SystemMessage(content="base"),
            ),
            _configurable(_integration(entries=[_entry_snapshot()], embedding_model=embedder)),
        )
        return embedder.queries

    def test_a_plain_turn_is_its_text(self) -> None:
        assert self._embedded_query([HumanMessage(content="which roofs need work")]) == ["which roofs need work"]

    def test_a_turn_carrying_an_image_is_its_text_blocks(self) -> None:
        # an attached image makes the turn a block list. the query is the question, not
        # an empty string (which dropped ranking to stable order) and not the list's repr.
        turn = HumanMessage(
            content=[
                {"type": "object_reference", "object_id": str(uuid7()), "mime_type": "image/png"},
                {"type": "text", "text": "which roofs need work"},
            ]
        )
        assert self._embedded_query([AIMessage(content="earlier"), turn]) == ["which roofs need work"]
