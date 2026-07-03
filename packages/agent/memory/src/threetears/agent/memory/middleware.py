"""3tears memory-injection priming middleware for ``langchain.agents.create_agent``.

The framework-aligned successor to the old ``memory_retrieval_node`` (a hand-rolled
graph node that ran before the agent node). Re-expressed onto
:meth:`langchain.agents.middleware.AgentMiddleware.awrap_model_call`: before each
model call, when a memory integration is injected on
``config["configurable"]["memory_integration"]`` and the verified per-call identity
is known (read off ``config["configurable"]["call_context"]``), the last
:class:`~langchain_core.messages.HumanMessage` is used as the retrieval query. Any
memories the retriever surfaces are wrapped under the memory-context prefix
(:data:`_MEMORY_CONTEXT_PREFIX`) and FOLDED into ``request.system_message``, and the
surfaced ids are deduped into a ``surfaced_memory_ids`` metadata ledger so the same
memory is not re-surfaced across turns.

Why the fold and not an extra ``SystemMessage`` in the message list: ``create_agent``
assembles the model prompt as ``[request.system_message, *request.messages]`` with NO
consolidation, and LangChain's Anthropic binding then RAISES on multiple
non-consecutive system messages (``ValueError("Received multiple non-consecutive
system messages")``). Injecting the memory block as a second ``SystemMessage`` (the
pattern an ``abefore_model`` hook would use) triggers exactly that crash against the
real gateway. Folding the block into the single ``request.system_message`` -- the
proven :class:`~threetears.langgraph.middleware_context.ContextMergeMiddleware`
pattern -- keeps one system message and sidesteps the failure.

Identity comes from the VERIFIED ``call_context`` (the realign handler makes
``call_context`` the canonical verified identity), NOT from graph state -- the old
node read ``state["agent_id"]`` / ``state["customer_id"]`` / ``state["user_id"]``,
which the framework no longer threads through state.

The deduped surfaced-ids ledger is persisted onto the ``metadata`` state channel via
a returned :class:`~langchain.agents.middleware.types.ExtendedModelResponse` carrying
a :class:`~langgraph.types.Command` state update; the dedup unions the newly surfaced
ids over the ids already on ``request.state["metadata"]`` so the ledger accumulates
across turns. The ``metadata`` channel carries a
:func:`threetears.langgraph.merge_metadata` reducer (declared on
:class:`MemoryInjectionState`) so this injector's key composes with the sibling
schema / knowledge injectors rather than clobbering them; an update to a channel the
state schema does not declare would be silently dropped.

The integration is read OPAQUELY off ``config["configurable"]`` as a duck-type: no
concrete import binds the framework to the host's memory wiring. Opt-in by
construction: with no integration on ``config["configurable"]`` the seam is a
pass-through. Every operation soft-fails -- a retrieval fault logs a warning and
proceeds on the un-merged request rather than failing the turn, exactly like the
original node.

Async is the real path -- the retriever's ``retrieve`` is ``async def``. The
synchronous :meth:`~MemoryInjectionMiddleware.wrap_model_call` mirror cannot drive it,
so it degrades to a pass-through (no injection this turn) and warns when an
integration was in fact configured so the misconfiguration is visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, NotRequired, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import AgentState, ExtendedModelResponse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_config
from langgraph.types import Command
from threetears.langgraph.state import merge_metadata
from threetears.observe import get_logger

from threetears.agent.memory.integration import retrieve_memories

__all__ = ["MemoryInjectionMiddleware", "MemoryInjectionState"]

log = get_logger(__name__)

#: Standing framing prepended to the injected memories. Homed here (the framework
#: middleware) rather than each agent's system prompt so it ships ONCE to every
#: memory-backed agent. It tells the model to treat the block as personalization
#: context, used only when relevant to the current turn.
_MEMORY_CONTEXT_PREFIX = (
    "## Relevant memories about this user\nUse these memories to personalize your response when relevant.\n\n"
)

#: metadata channel key for the surfaced-memory-id ledger. deduped across turns so a
#: memory the agent has already been shown is not re-injected.
_SURFACED_MEMORY_IDS_KEY = "surfaced_memory_ids"


class MemoryInjectionState(AgentState[Any]):
    """Agent state extended with the ``surfaced_memory_ids`` metadata channel.

    Extends the base :class:`~langchain.agents.middleware.types.AgentState` with a
    ``metadata`` dict channel so :class:`MemoryInjectionMiddleware` can stash the
    deduped surfaced-memory-id ledger. The channel carries the
    :func:`threetears.langgraph.merge_metadata` reducer so this injector's key
    composes with the sibling schema / knowledge injectors (each writes disjoint
    top-level keys) rather than the last writer clobbering the others; LangGraph
    silently drops an update to a channel the state schema does not declare, so the
    channel must be declared here for the ledger to persist.

    :ivar metadata: per-run metadata bag; the middleware writes the deduped
        surfaced-memory ids under ``surfaced_memory_ids``.
    """

    metadata: NotRequired[Annotated[dict[str, Any], merge_metadata]]


def _configurable() -> dict[str, Any]:
    """Return the active run's ``configurable`` dict, or empty when unavailable.

    Reads the :class:`~langchain_core.runnables.RunnableConfig` for the active run
    via :func:`langgraph.config.get_config` -- the canonical way a ``wrap_model_call``
    hook reaches ``configurable`` (the ``Runtime`` on a ``ModelRequest`` does not carry
    ``config`` for the model-call seam). Outside a runnable context ``get_config``
    raises ``RuntimeError``; with no config there is no injected integration, so this
    soft-fails to an empty dict and the seam becomes a pass-through rather than
    crashing.

    :return: the run's ``configurable`` mapping, or an empty dict.
    :rtype: dict[str, Any]
    """
    try:
        config = get_config()
    except RuntimeError:
        return {}
    return config.get("configurable") or {}


def _fold_into_system(
    system_message: SystemMessage | None,
    block: str,
) -> SystemMessage:
    """Return a single ``SystemMessage`` carrying the base prompt plus the block.

    Preserves the base system prompt and appends the memory block after a blank line.
    When the base message has structured (list) content -- e.g. a prior middleware
    already annotated it -- the block is appended as an extra text part so existing
    parts (and any ``cache_control`` on them) are left intact. When there is no base
    system message, the block becomes the whole system message. Mirrors
    :func:`threetears.langgraph.middleware_context._fold_context_into_system`.

    :param system_message: the request's current system message, or ``None``.
    :ptype system_message: SystemMessage | None
    :param block: the non-empty memory block to fold in.
    :ptype block: str
    :return: the merged system message.
    :rtype: SystemMessage
    """
    if system_message is None:
        return SystemMessage(content=block)
    base_content = system_message.content
    if isinstance(base_content, list):
        merged_parts = [*base_content, {"type": "text", "text": block}]
        return system_message.model_copy(update={"content": merged_parts})
    base_text = base_content if isinstance(base_content, str) else str(base_content)
    return system_message.model_copy(update={"content": f"{base_text}\n\n{block}"})


def _extract_last_user_message(messages: Sequence[BaseMessage]) -> str:
    """Extract text content from the last user (human) message in the list.

    Iterates the messages in reverse to find the most recent
    :class:`~langchain_core.messages.HumanMessage`.

    :param messages: sequence of conversation messages (covariant in
        :class:`BaseMessage` so the LangGraph ``AnyMessage`` union type passes
        without losing typing precision).
    :ptype messages: Sequence[BaseMessage]
    :return: text content of the last user message, or an empty string if none found.
    :rtype: str
    """
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = str(msg.content)
            break
    return query


class MemoryInjectionMiddleware(AgentMiddleware[MemoryInjectionState, Any, Any]):
    """Fold retrieved user memories into the system prompt before the model.

    The ``create_agent`` successor to the ``memory_retrieval_node``. On every model
    call, when a memory integration is injected on
    ``config["configurable"]["memory_integration"]`` and the verified per-call
    identity is present on ``call_context``, the last ``HumanMessage`` is used as the
    retrieval query; any surfaced memories are FOLDED into ``request.system_message``
    under :data:`_MEMORY_CONTEXT_PREFIX` (yielding exactly one system message), and the
    surfaced ids are deduped into ``metadata["surfaced_memory_ids"]`` via a returned
    ``Command`` state update. Opt-in: no integration (or no verified identity) on
    ``config["configurable"]`` -> the seam is a pass-through. Every operation
    soft-fails: a retrieval fault logs and proceeds on the un-merged request.
    """

    name = "MemoryInjectionMiddleware"

    state_schema: type[MemoryInjectionState] = MemoryInjectionState

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        """Retrieve memories and fold them into the system prompt before the model.

        Reads the memory integration + the verified ``call_context`` identity off
        ``config["configurable"]``. Uses the last ``HumanMessage`` as the query,
        retrieves memories under the verified agent/customer/user identity, wraps them
        under :data:`_MEMORY_CONTEXT_PREFIX`, folds the block into
        ``request.system_message``, runs the call, and returns an
        :class:`ExtendedModelResponse` whose ``Command`` persists the deduped surfaced
        ids under ``metadata["surfaced_memory_ids"]``. Proceeds on the un-merged
        request (a plain response) when no integration/identity is configured, when
        there is no query, when no memory is surfaced, or when retrieval faults
        (soft-fail).

        :param request: the model request (carries ``system_message`` + ``messages``
            + ``state``).
        :ptype request: ModelRequest
        :param handler: the next handler in the middleware chain.
        :ptype handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
        :return: the model response, wrapped with a metadata ``Command`` when memories
            were injected.
        :rtype: ModelResponse[Any] | ExtendedModelResponse[Any]
        """
        configurable = _configurable()
        integration = configurable.get("memory_integration")
        # identity comes from the VERIFIED call_context, never from graph state.
        # a missing integration or call_context means no injection this turn.
        call_context = configurable.get("call_context") if integration is not None else None
        memory_text = ""
        surfaced_ids: list[str] = []
        if integration is not None and call_context is not None:
            try:
                agent_id = call_context.agent_id
                customer_id = call_context.customer_id
                user_id = call_context.user_id

                query = _extract_last_user_message(request.messages)

                memories: list[str] = []
                if query:
                    memories = await retrieve_memories(
                        integration,
                        agent_id=agent_id,
                        customer_id=customer_id,
                        user_id=user_id,
                        query=query,
                    )

                if memories:
                    memory_text = _MEMORY_CONTEXT_PREFIX + "\n".join(f"- {m}" for m in memories)
                    # dedup across turns: union the newly surfaced ids over the ids
                    # already on state so a memory shown on a prior turn is not
                    # double-counted in the ledger.
                    state = cast("MemoryInjectionState", request.state)
                    existing = state.get("metadata") or {}
                    surfaced: set[str] = set(existing.get(_SURFACED_MEMORY_IDS_KEY, []))
                    surfaced.update(memories)
                    surfaced_ids = list(surfaced)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- memory injection is best-effort context enrichment; a fault proceeds on the un-merged request rather than failing the turn
                log.warning("memory injection middleware failed (soft-fail): %s", exc)
                memory_text, surfaced_ids = "", []
        if not memory_text:
            passthrough: ModelResponse[Any] = await handler(request)
            return passthrough
        merged = _fold_into_system(request.system_message, memory_text)
        response: ModelResponse[Any] = await handler(request.override(system_message=merged))
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"metadata": {_SURFACED_MEMORY_IDS_KEY: surfaced_ids}}),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse[Any]:
        """Sync mirror: cannot drive the async retriever, so pass the call through.

        The retriever's ``retrieve`` is ``async def``; a synchronous
        ``wrap_model_call`` path cannot await it. Rather than fail the run, memory
        injection is skipped this turn (the request passes through un-merged) and a
        warning is logged when an integration was in fact configured so the
        misconfiguration is visible.

        :param request: the model request (passed through unchanged).
        :ptype request: ModelRequest
        :param handler: the next handler in the middleware chain.
        :ptype handler: Callable[[ModelRequest], ModelResponse]
        :return: the model response from the un-merged request.
        :rtype: ModelResponse[Any]
        """
        if _configurable().get("memory_integration") is not None:
            log.warning(
                "memory injection skipped: an integration is configured but the agent "
                "ran on the synchronous model path; retrieval is async-only, so no "
                "memories are injected this turn",
            )
        result: ModelResponse[Any] = handler(request)
        return result
