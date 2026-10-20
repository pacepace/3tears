"""A per-turn tool-loop budget for ``langchain.agents.create_agent``, so a looping model still answers.

A model that keeps calling tools runs a ``create_agent`` graph into LangGraph's recursion limit,
and the turn ends on ``GraphRecursionError`` with no reply. :class:`ToolLoopBudgetMiddleware` ends
the loop first, and always with text:

- **A per-turn cap.** At most ``max_tool_calls`` tool calls per turn (a turn is one invocation of
  the agent; resuming an interrupted one continues it). The cap is absolute: a response whose
  calls would take the turn past it is discarded and the forced answer pass runs instead.
- **Escalating tiers.** Two pressures are measured before every model call -- tool calls against
  the cap, and the prompt against the model's context window -- and the larger sets the tier.
  At ``warn`` and ``urgent`` the model is shown a wrap-up nudge; at ``force`` it is told to
  answer and any tool call it makes is dropped. Calls reach ``warn``/``urgent`` at ``warn_pct``/``urgent_pct`` of the cap and
  ``force`` AT the cap; the window reaches each tier at its pct, ``force_pct`` below 100 so the
  answer has room to generate.
- **A forced answer pass.** At ``force`` the model is told to answer from what it already has.
  Its tools stay declared -- Anthropic refuses a request whose history holds tool calls and
  results but declares no tools -- with ``forced_tool_choice`` (``"none"`` where the provider
  takes it) as the request's ``tool_choice``. The reply is rebuilt as text only: no tool call
  and no tool-use content block survives, so the turn ends.
- **An empty-answer guard.** A final answer with no visible text (empty, or reasoning only) is
  retried once, with a stricter instruction and the model's own reasoning; if that is empty too
  the turn ends with ``fallback_text``, run through the model chain as a one-message stand-in
  model so it streams and lands in the history like any answer. Never an empty turn.
- **Metered.** The step's answer carries the summed ``usage_metadata`` of every model call the
  step made (a discarded round and a retry included) and the last call's ``response_metadata``.
- **A runaway guard.** ``runaway_rounds`` consecutive rounds that wrote no text and whose every
  tool call failed force the answer pass, however far below the cap the turn is.

**The turn's record lives in agent state, not in the messages.** The budget declares a private
state field (:class:`ToolLoopBudgetState`), resets it in ``before_agent`` -- once per invocation,
not on a resume -- and updates it with each model step's response. The count is never inferred
from the ``messages`` channel, because other middleware rewrites that channel mid-turn:
:class:`~threetears.langgraph.SummarizationMiddleware` replaces the history with a summary
``HumanMessage`` and a tail, which a "turn starts at the last user message" rule would read as a
new turn, and any middleware may append a ``HumanMessage``. The instance itself holds nothing, so
one serves every thread, and a checkpoint carries the record. Nudges and instructions are request
overrides, never written to state. Install it first in the ``middleware`` list, so its window
pressure measures the request before other middleware adds to it.

:func:`recursion_limit_for` gives the graph a ``recursion_limit`` the budget can never reach, so
the budget, not LangGraph's recursion error, is what ends a loop.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Annotated, Any, Literal, NotRequired, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import AgentState, ExtendedModelResponse, PrivateStateAttr
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import Command

from threetears.observe import get_logger

__all__ = [
    "DEFAULT_MAX_TOOL_CALLS_PER_TURN",
    "DEFAULT_TOOL_BUDGET_FALLBACK_TEXT",
    "ToolBudgetTier",
    "ToolLoopBudgetMiddleware",
    "ToolLoopBudgetState",
    "recursion_limit_for",
]

log = get_logger(__name__)

#: Tool calls one turn may make. metallm's ``max_tool_calls_per_turn``, which its research turns run under.
DEFAULT_MAX_TOOL_CALLS_PER_TURN = 16

#: The turn's reply when even the stricter retry wrote nothing.
DEFAULT_TOOL_BUDGET_FALLBACK_TEXT = (
    "I wasn't able to write an answer this time. I looked things up with my tools but didn't manage "
    "to turn what I found into a reply. Please ask again, or narrow the question."
)

_WARN_TEXT = (
    "Start wrapping up: answer instead of making more tool calls. "
    "You have made {tool_calls} of the {max_tool_calls} tool calls this turn allows."
)
_URGENT_TEXT = (
    "Stop calling tools and write your answer now, from what you already have. "
    "You have made {tool_calls} of the {max_tool_calls} tool calls this turn allows."
)
_FORCE_TEXT = (
    "You can make no more tool calls this turn. Answer the user now using only the information "
    "you already have. If it is incomplete, say what you found and what is missing."
)
_RUNAWAY_TEXT = (
    "Your recent tool calls failed and you have written nothing. Stop calling tools and answer "
    "the user now with what you have, saying plainly what you could not do."
)
_RETRY_TEXT = (
    "Your last reply was empty. Write your answer to the user now, as plain text, using only "
    "what you already have. Do not call tools and do not think it through again."
)
_RETRY_REASONING_PREFIX = "You thought it through but wrote no reply. Your thinking:\n\n{reasoning}\n\n"

#: Supersteps a ``create_agent`` round costs with no hook middleware: the model node and the tools node.
_MODEL_AND_TOOLS_STEPS = 2
#: Supersteps a run spends taking its input, before the first node runs.
_INPUT_STEPS = 1


class ToolBudgetTier(IntEnum):
    """How hard the budget is pressing on the turn; a higher tier is more insistent."""

    NONE = 0
    WARN = 1
    URGENT = 2
    FORCE = 3


#: The agent-state key the budget keeps its turn record under.
_STATE_KEY = "tool_loop_budget"


class ToolLoopBudgetState(AgentState[Any]):
    """Agent state with the budget's private turn record.

    The record is reset by the budget's ``before_agent`` hook, which runs once when an invocation
    starts and not when an interrupted one resumes, so a turn is one invocation. Nothing that
    rewrites the ``messages`` channel can move it.

    :ivar tool_loop_budget: the current turn's record (:class:`_TurnTally` as a dict)
    """

    tool_loop_budget: NotRequired[Annotated[dict[str, Any], PrivateStateAttr]]


@dataclass(frozen=True)
class _TurnTally:
    """What the current turn has done so far.

    :ivar tool_calls: tool calls the turn's rounds made
    :ivar rounds: model responses in the turn whose tool calls were dispatched
    :ivar failing_rounds: trailing consecutive rounds with no text whose every tool result failed
    :ivar tier: the highest tier the turn has reached (so a transition is logged once)
    :ivar open_call_ids: the latest round's tool-call ids, until the next model call judges its results
    :ivar open_wrote_text: whether the latest round's response wrote text
    """

    tool_calls: int = 0
    rounds: int = 0
    failing_rounds: int = 0
    tier: int = 0
    open_call_ids: tuple[str, ...] = ()
    open_wrote_text: bool = False

    @classmethod
    def from_state(cls, state: Any) -> _TurnTally:
        """The record in agent state; a fresh one when there is none.

        :param state: the agent state
        :ptype state: Any
        :return: the record
        :rtype: _TurnTally
        """
        raw = state.get(_STATE_KEY) if isinstance(state, Mapping) else None
        if not isinstance(raw, Mapping):
            return cls()
        return cls(
            tool_calls=int(raw.get("tool_calls", 0)),
            rounds=int(raw.get("rounds", 0)),
            failing_rounds=int(raw.get("failing_rounds", 0)),
            tier=int(raw.get("tier", 0)),
            open_call_ids=tuple(str(i) for i in raw.get("open_call_ids", ())),
            open_wrote_text=bool(raw.get("open_wrote_text", False)),
        )

    def to_state(self) -> dict[str, Any]:
        """The record as the dict agent state holds.

        :return: the dict
        :rtype: dict[str, Any]
        """
        return {
            "tool_calls": self.tool_calls,
            "rounds": self.rounds,
            "failing_rounds": self.failing_rounds,
            "tier": self.tier,
            "open_call_ids": list(self.open_call_ids),
            "open_wrote_text": self.open_wrote_text,
        }

    def settled(self, messages: Sequence[BaseMessage]) -> _TurnTally:
        """Judge the latest round by its tool results, now that they are in ``messages``.

        A round fails when its response wrote no text and every one of its tool results is an
        error -- in ``create_agent`` terms, metallm's round with no text and no valid dispatch.
        Results that cannot be found count as not failing.

        :param messages: the conversation
        :ptype messages: Sequence[BaseMessage]
        :return: the record with the round judged and closed
        :rtype: _TurnTally
        """
        if not self.open_call_ids:
            return self
        wanted = set(self.open_call_ids)
        statuses = [m.status for m in messages if isinstance(m, ToolMessage) and m.tool_call_id in wanted]
        failed = bool(statuses) and all(s == "error" for s in statuses) and not self.open_wrote_text
        return replace(
            self, failing_rounds=self.failing_rounds + 1 if failed else 0, open_call_ids=(), open_wrote_text=False
        )

    def dispatched(self, answer: AIMessage) -> _TurnTally:
        """The record after a response whose tool calls go on to run.

        :param answer: the response
        :ptype answer: AIMessage
        :return: the record with the round counted and left open
        :rtype: _TurnTally
        """
        return replace(
            self,
            tool_calls=self.tool_calls + len(answer.tool_calls),
            rounds=self.rounds + 1,
            open_call_ids=tuple(str(c.get("id")) for c in answer.tool_calls if c.get("id")),
            open_wrote_text=bool(_visible_text(answer)),
        )


def _visible_text(message: BaseMessage) -> str:
    """The text a reader would see in ``message``: its text blocks, reasoning excluded.

    :param message: the message
    :ptype message: BaseMessage
    :return: the stripped visible text, empty when there is none
    :rtype: str
    """
    return str(message.text).strip()


def _reasoning_text(message: BaseMessage) -> str:
    """The reasoning a message carries, wherever its provider puts it.

    Anthropic-style ``thinking`` blocks, standard ``reasoning`` blocks, and the
    ``reasoning_content`` / ``reasoning`` keys OpenAI-compatible adapters (DeepSeek, OpenRouter)
    leave in ``additional_kwargs``.

    :param message: the message
    :ptype message: BaseMessage
    :return: the reasoning, stripped; empty when there is none
    :rtype: str
    """
    parts: list[str] = []
    if isinstance(message.content, list):
        for block in message.content:
            if not isinstance(block, Mapping):
                continue
            for key in ("thinking", "reasoning"):
                value = block.get(key)
                if block.get("type") in ("thinking", "reasoning") and isinstance(value, str):
                    parts.append(value)
    for key in ("reasoning_content", "reasoning"):
        value = message.additional_kwargs.get(key)
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(p.strip() for p in parts if p.strip())


def _response_messages(response: ModelResponse) -> list[BaseMessage]:
    """The messages a model response carries.

    :param response: the response
    :ptype response: ModelResponse
    :return: the messages, as a new list
    :rtype: list[BaseMessage]
    """
    return list(response.result)


def _structured(response: ModelResponse | None) -> bool:
    """Whether the response carries a structured response (the consumer's response format answered).

    :param response: the response
    :ptype response: ModelResponse | None
    :return: whether a structured response is present
    :rtype: bool
    """
    return response is not None and response.structured_response is not None


def _last_ai(response: ModelResponse | None) -> AIMessage | None:
    """The last ``AIMessage`` in a model response.

    :param response: the response
    :ptype response: ModelResponse | None
    :return: the message, or ``None`` when there is none
    :rtype: AIMessage | None
    """
    for message in reversed(_response_messages(response) if response is not None else []):
        if isinstance(message, AIMessage):
            return message
    return None


def _with_answer(response: ModelResponse, answer: AIMessage) -> ModelResponse:
    """``response`` with ``answer`` in place of its last ``AIMessage`` and no tool call left open.

    Drops every tool call, invalid tool call and tool-use content block from ``answer`` (its
    usage and response metadata stay), and with them any ``ToolMessage``
    in the response that answers one (a structured-output parse error comes back as one): a
    tool result whose call is gone is rejected by providers on the next call.

    :param response: the response
    :ptype response: ModelResponse
    :param answer: the answer to end the step with
    :ptype answer: AIMessage
    :return: the rebuilt response
    :rtype: ModelResponse
    """
    original = _last_ai(response)
    dropped = {c["id"] for c in (original.tool_calls if original is not None else []) if c.get("id")}
    dropped |= {c["id"] for c in answer.tool_calls if c.get("id")}
    content: Any = answer.content
    if isinstance(content, list):
        content = [b for b in content if not (isinstance(b, Mapping) and _is_tool_block(b))]
    answer = answer.model_copy(update={"content": content, "tool_calls": [], "invalid_tool_calls": []})
    messages: list[BaseMessage] = []
    placed = False
    for message in reversed(_response_messages(response)):
        if isinstance(message, ToolMessage) and message.tool_call_id in dropped:
            continue
        if isinstance(message, AIMessage) and not placed:
            message, placed = answer, True
        messages.append(message)
    messages.reverse()
    if not placed:
        messages.append(answer)
    return ModelResponse(result=messages, structured_response=response.structured_response)


def _is_tool_block(block: Mapping[str, Any]) -> bool:
    """Whether a content block is a tool call in some provider's shape (``tool_use``, ``tool_call``...).

    :param block: the content block
    :ptype block: Mapping[str, Any]
    :return: whether it names a tool call
    :rtype: bool
    """
    kind = block.get("type")
    return isinstance(kind, str) and ("tool" in kind or kind == "function_call")


def _metered(result: ExtendedModelResponse, spent: Sequence[ModelResponse]) -> ExtendedModelResponse:
    """``result`` with its answer carrying the usage of every call the step made.

    :param result: the step's result
    :ptype result: ExtendedModelResponse
    :param spent: every response the step's model calls returned, in order
    :ptype spent: Sequence[ModelResponse]
    :return: the result, its answer's ``usage_metadata`` summed and ``response_metadata`` filled
    :rtype: ExtendedModelResponse
    """
    answer = _last_ai(result.model_response)
    if answer is None or len(spent) < 2:
        return result
    replies = [ai for r in spent if (ai := _last_ai(r)) is not None]
    usage: UsageMetadata | None = None
    for ai in replies:
        if ai.usage_metadata is not None:
            usage = ai.usage_metadata if usage is None else add_usage(usage, ai.usage_metadata)
    metadata = answer.response_metadata or next(
        (ai.response_metadata for ai in reversed(replies) if ai.response_metadata), {}
    )
    update: dict[str, Any] = {"response_metadata": dict(metadata)}
    if usage is not None:
        update["usage_metadata"] = usage
    messages = [answer.model_copy(update=update) if m is answer else m for m in result.model_response.result]
    response = ModelResponse(result=messages, structured_response=result.model_response.structured_response)
    return ExtendedModelResponse(model_response=response, command=result.command)


class _FallbackChatModel(BaseChatModel):
    """A stand-in model that says one fixed text, so the fallback reply runs the model chain and streams."""

    text: str

    @property
    def _llm_type(self) -> str:
        return "tool-loop-budget-fallback"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[LanguageModelInput, AIMessage]:
        """Accept the agent's tools and ignore them: this model only ever says its text.

        :param tools: the tools
        :ptype tools: Sequence[Any]
        :return: this model
        :rtype: Runnable[LanguageModelInput, AIMessage]
        """
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content=self.text))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content=self.text))


def _middleware_overrides(middleware: AgentMiddleware, hook: str) -> bool:
    """Whether ``middleware`` implements ``hook`` (sync or async), as ``create_agent`` decides it.

    :param middleware: the middleware
    :ptype middleware: AgentMiddleware
    :param hook: the hook's sync name, e.g. ``"before_model"``
    :ptype hook: str
    :return: whether the factory adds a graph node for it
    :rtype: bool
    """
    cls = type(middleware)
    return getattr(cls, hook) is not getattr(AgentMiddleware, hook) or getattr(cls, f"a{hook}") is not getattr(
        AgentMiddleware, f"a{hook}"
    )


def recursion_limit_for(
    max_tool_calls: int | None = None,
    *,
    middleware: Sequence[AgentMiddleware] = (),
    margin: int = 3,
) -> int:
    """A ``recursion_limit`` for a ``create_agent`` graph that the tool budget can never reach.

    Pass the same ``middleware`` list the agent is built with. The cap is the
    :class:`ToolLoopBudgetMiddleware`'s in that list, so it is stated once; ``max_tool_calls`` is
    for a list with no budget in it, and if given alongside one it must agree.

    Under the budget a turn makes at most ``max_tool_calls`` tool rounds (each round makes at
    least one call) and one more model call that answers. Each model call costs the model node
    plus one node per ``before_model`` and ``after_model`` hook; each tool round costs the tools
    node; ``before_agent`` and ``after_agent`` hooks (the budget's own included) cost one node
    each, once; and the run's input is one step of its own. With ``margin=0`` it is exactly the
    steps of the longest turn the budget allows.

    Put the result on the run config: ``{"recursion_limit": recursion_limit_for(middleware=...)}``.

    :param max_tool_calls: the cap, for a list with no budget in it; default 16
    :ptype max_tool_calls: int | None
    :param middleware: the agent's middleware
    :ptype middleware: Sequence[AgentMiddleware]
    :param margin: slack above the worst case
    :ptype margin: int
    :return: the limit
    :rtype: int
    :raises ValueError: two budgets in the list, a ``max_tool_calls`` that disagrees with the
        budget's, or a cap below 1
    """
    budgets = [m for m in middleware if isinstance(m, ToolLoopBudgetMiddleware)]
    if len(budgets) > 1:
        raise ValueError(f"the middleware list holds {len(budgets)} ToolLoopBudgetMiddleware; an agent takes one")
    if budgets:
        cap = budgets[0].max_tool_calls
        if max_tool_calls is not None and max_tool_calls != cap:
            raise ValueError(f"max_tool_calls={max_tool_calls} disagrees with the budget's max_tool_calls={cap}")
    else:
        cap = max_tool_calls if max_tool_calls is not None else DEFAULT_MAX_TOOL_CALLS_PER_TURN
    if cap < 1:
        raise ValueError(f"max_tool_calls must be at least 1, got {cap}")
    per_model_call = 1 + sum(
        _middleware_overrides(m, "before_model") + _middleware_overrides(m, "after_model") for m in middleware
    )
    once = sum(_middleware_overrides(m, "before_agent") + _middleware_overrides(m, "after_agent") for m in middleware)
    tools_steps = (_MODEL_AND_TOOLS_STEPS - 1) * cap
    return _INPUT_STEPS + (cap + 1) * per_model_call + tools_steps + once + margin


class ToolLoopBudgetMiddleware(AgentMiddleware):
    """Bound a turn's tool loop, nudge the model to wrap up, and always end the turn with text.

    See the module docstring for the behaviour; every threshold and text is a constructor argument.

    :param max_tool_calls: tool calls one turn may make; ``force`` is reached at this count
    :ptype max_tool_calls: int
    :param warn_pct: percent of the cap, and of the context window, at which ``warn`` begins
    :ptype warn_pct: float
    :param urgent_pct: percent of the cap, and of the context window, at which ``urgent`` begins
    :ptype urgent_pct: float
    :param force_pct: percent of the context window at which ``force`` begins
    :ptype force_pct: float
    :param runaway_rounds: consecutive failing rounds with no text that force the answer
    :ptype runaway_rounds: int
    :param context_window_tokens: the model's input window; ``None`` reads the chat model's
        ``profile["max_input_tokens"]``, and with neither the window pressure is off
    :ptype context_window_tokens: int | None
    :param count_tokens: counts a prompt's tokens; approximate by default
    :ptype count_tokens: Callable[[Sequence[BaseMessage]], int]
    :param nudge_role: ``"human"`` appends nudges as a trailing user message (accepted by every
        provider after tool results); ``"system"`` as a trailing system message
    :ptype nudge_role: Literal["human", "system"]
    :param warn_text: the ``warn`` nudge; may use ``{tool_calls}`` and ``{max_tool_calls}``
    :ptype warn_text: str
    :param urgent_text: the ``urgent`` nudge; may use ``{tool_calls}`` and ``{max_tool_calls}``
    :ptype urgent_text: str
    :param force_text: the forced answer pass's instruction
    :ptype force_text: str
    :param runaway_text: the forced answer pass's instruction when the runaway guard forced it
    :ptype runaway_text: str
    :param retry_text: the stricter instruction when an answer came back empty
    :ptype retry_text: str
    :param fallback_text: the turn's reply when the retry is empty too
    :ptype fallback_text: str
    :param forced_tool_choice: the ``tool_choice`` of the forced pass and the retry, whose tools
        stay declared; ``"none"`` where the provider adapter takes it, ``None`` to steer by the
        instruction alone (any call is dropped either way)
    :ptype forced_tool_choice: str | None
    """

    name = "ToolLoopBudgetMiddleware"
    state_schema = ToolLoopBudgetState

    def __init__(
        self,
        *,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS_PER_TURN,
        warn_pct: float = 70,
        urgent_pct: float = 85,
        force_pct: float = 90,
        runaway_rounds: int = 3,
        context_window_tokens: int | None = None,
        count_tokens: Callable[[Sequence[BaseMessage]], int] = count_tokens_approximately,
        nudge_role: Literal["human", "system"] = "human",
        warn_text: str = _WARN_TEXT,
        urgent_text: str = _URGENT_TEXT,
        force_text: str = _FORCE_TEXT,
        runaway_text: str = _RUNAWAY_TEXT,
        retry_text: str = _RETRY_TEXT,
        fallback_text: str = DEFAULT_TOOL_BUDGET_FALLBACK_TEXT,
        forced_tool_choice: str | None = None,
    ) -> None:
        super().__init__()
        if max_tool_calls < 1:
            raise ValueError(f"max_tool_calls must be at least 1, got {max_tool_calls}")
        if not 0 < warn_pct <= urgent_pct <= force_pct <= 100:
            raise ValueError(
                f"need 0 < warn_pct <= urgent_pct <= force_pct <= 100, got {warn_pct}, {urgent_pct}, {force_pct}"
            )
        if runaway_rounds < 1:
            raise ValueError(f"runaway_rounds must be at least 1, got {runaway_rounds}")
        if context_window_tokens is not None and context_window_tokens < 1:
            raise ValueError(f"context_window_tokens must be positive, got {context_window_tokens}")
        if not fallback_text.strip():
            raise ValueError("fallback_text must not be blank: it is the reply when nothing else is")
        self.max_tool_calls = max_tool_calls
        self.warn_pct = warn_pct
        self.urgent_pct = urgent_pct
        self.force_pct = force_pct
        self.runaway_rounds = runaway_rounds
        self.context_window_tokens = context_window_tokens
        self.count_tokens = count_tokens
        self.nudge_role = nudge_role
        self.warn_text = warn_text
        self.urgent_text = urgent_text
        self.force_text = force_text
        self.runaway_text = runaway_text
        self.retry_text = retry_text
        self.fallback_text = fallback_text
        self.forced_tool_choice = forced_tool_choice

    # --- the turn ---------------------------------------------------------------------------------

    def before_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any]:
        """Start the turn's record: an invocation is a turn.

        :param state: the agent state
        :ptype state: AgentState[Any]
        :param runtime: the run's runtime
        :ptype runtime: Runtime[Any]
        :return: the state update resetting the record
        :rtype: dict[str, Any]
        """
        return {_STATE_KEY: _TurnTally().to_state()}

    async def abefore_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any]:
        """Start the turn's record: an invocation is a turn.

        :param state: the agent state
        :ptype state: AgentState[Any]
        :param runtime: the run's runtime
        :ptype runtime: Runtime[Any]
        :return: the state update resetting the record
        :rtype: dict[str, Any]
        """
        return self.before_agent(state, runtime)

    # --- assessment -----------------------------------------------------------------------------

    def _calls_tier(self, tool_calls: int) -> ToolBudgetTier:
        """The tier the turn's tool-call count alone puts it in.

        :param tool_calls: the turn's tool calls so far
        :ptype tool_calls: int
        :return: the tier
        :rtype: ToolBudgetTier
        """
        if tool_calls <= 0:
            return ToolBudgetTier.NONE
        if tool_calls >= self.max_tool_calls:
            return ToolBudgetTier.FORCE
        if tool_calls >= math.ceil(self.max_tool_calls * self.urgent_pct / 100):
            return ToolBudgetTier.URGENT
        if tool_calls >= math.ceil(self.max_tool_calls * self.warn_pct / 100):
            return ToolBudgetTier.WARN
        return ToolBudgetTier.NONE

    def _window_tier(self, pct: float) -> ToolBudgetTier:
        """The tier the prompt's share of the context window alone puts the turn in.

        :param pct: the prompt as a percent of the window
        :ptype pct: float
        :return: the tier
        :rtype: ToolBudgetTier
        """
        if pct >= self.force_pct:
            return ToolBudgetTier.FORCE
        if pct >= self.urgent_pct:
            return ToolBudgetTier.URGENT
        if pct >= self.warn_pct:
            return ToolBudgetTier.WARN
        return ToolBudgetTier.NONE

    def _window(self, request: ModelRequest) -> int | None:
        """The model's input window: the configured one, else the chat model's profile.

        :param request: the model request
        :ptype request: ModelRequest
        :return: the window in tokens, or ``None`` when unknown
        :rtype: int | None
        """
        if self.context_window_tokens is not None:
            return self.context_window_tokens
        profile = getattr(request.model, "profile", None)
        window = profile.get("max_input_tokens") if isinstance(profile, Mapping) else None
        return window if isinstance(window, int) and window > 0 else None

    def _window_pct(self, request: ModelRequest) -> float:
        """The prompt the request sends, with its system message, as a percent of the window.

        :param request: the model request
        :ptype request: ModelRequest
        :return: the percent; ``0.0`` when the window is unknown
        :rtype: float
        """
        window = self._window(request)
        if window is None:
            return 0.0
        messages = request.messages
        prompt = [request.system_message, *messages] if request.system_message is not None else list(messages)
        return self.count_tokens(prompt) / window * 100

    def _assess(self, request: ModelRequest, tally: _TurnTally) -> tuple[ToolBudgetTier, str, float]:
        """The turn's tier, which pressure set it, and the window percent.

        :param request: the model request
        :ptype request: ModelRequest
        :param tally: the turn's record
        :ptype tally: _TurnTally
        :return: ``(tier, cause, window_pct)``; cause is ``"tool_calls"`` or ``"context_window"``
        :rtype: tuple[ToolBudgetTier, str, float]
        """
        if tally.rounds == 0:
            return ToolBudgetTier.NONE, "tool_calls", 0.0
        pct = self._window_pct(request)
        by_calls, by_window = self._calls_tier(tally.tool_calls), self._window_tier(pct)
        if by_window > by_calls:
            return by_window, "context_window", pct
        return by_calls, "tool_calls", pct

    @staticmethod
    def _conversation(request: ModelRequest) -> Sequence[BaseMessage]:
        """The state's messages, which outer middleware cannot have rewritten; else the request's.

        :param request: the model request
        :ptype request: ModelRequest
        :return: the conversation
        :rtype: Sequence[BaseMessage]
        """
        state = request.state
        messages = state.get("messages") if isinstance(state, Mapping) else None
        return messages if messages else request.messages

    def _instruction(self, text: str) -> HumanMessage | SystemMessage:
        """``text`` as the message role this budget appends.

        :param text: the instruction
        :ptype text: str
        :return: the message
        :rtype: HumanMessage | SystemMessage
        """
        return HumanMessage(content=text) if self.nudge_role == "human" else SystemMessage(content=text)

    def _answer_only(self, request: ModelRequest, text: str) -> ModelRequest:
        """``request`` told to answer: ``text`` appended, tools still declared, ``forced_tool_choice`` set.

        :param request: the model request
        :ptype request: ModelRequest
        :param text: the instruction
        :ptype text: str
        :return: the overridden request
        :rtype: ModelRequest
        """
        return request.override(
            tool_choice=self.forced_tool_choice, messages=[*request.messages, self._instruction(text)]
        )

    # --- the plan both seams drive ---------------------------------------------------------------

    def _plan(self, request: ModelRequest) -> Generator[ModelRequest, ModelResponse, ExtendedModelResponse]:
        """The budget's model calls for one ``create_agent`` model step.

        Yields each request to run; receives its response (or has the call's exception thrown in);
        returns the step's response with the turn's updated record as its state command.
        :meth:`wrap_model_call` and :meth:`awrap_model_call` drive it, so the two seams cannot drift.

        :param request: the model request
        :ptype request: ModelRequest
        :return: the step's result
        :rtype: Generator[ModelRequest, ModelResponse, ExtendedModelResponse]
        """
        tally = _TurnTally.from_state(request.state).settled(self._conversation(request))
        return (yield from self._step(request, tally))

    def _step(
        self, request: ModelRequest, tally: _TurnTally
    ) -> Generator[ModelRequest, ModelResponse, ExtendedModelResponse]:
        """Assess the turn, run the model as the tier says, and record what the response did.

        :param request: the model request
        :ptype request: ModelRequest
        :param tally: the turn's record, its latest round judged
        :ptype tally: _TurnTally
        :return: the step's result
        :rtype: Generator[ModelRequest, ModelResponse, ExtendedModelResponse]
        """
        tier, cause, pct = self._assess(request, tally)
        if tier > tally.tier:
            log.warning(
                "tool-loop budget tier reached",
                extra={
                    "extra_data": {
                        "tier": tier.name.lower(),
                        "cause": cause,
                        "tool_calls": tally.tool_calls,
                        "max_tool_calls": self.max_tool_calls,
                        "rounds": tally.rounds,
                        "window_pct": round(pct, 1),
                    }
                },
            )
            tally = replace(tally, tier=int(tier))

        if tally.failing_rounds >= self.runaway_rounds:
            answered = yield from self._forced_answer(request, tally.tool_calls, "runaway", self.runaway_text)
            return self._recorded(answered, tally)
        if tier is ToolBudgetTier.FORCE:
            answered = yield from self._forced_answer(request, tally.tool_calls, cause, self.force_text)
            return self._recorded(answered, tally)

        to_run = request
        if tier in (ToolBudgetTier.WARN, ToolBudgetTier.URGENT):
            template = self.warn_text if tier is ToolBudgetTier.WARN else self.urgent_text
            nudge = template.format(tool_calls=tally.tool_calls, max_tool_calls=self.max_tool_calls)
            to_run = request.override(messages=[*request.messages, self._instruction(nudge)])
        response = yield to_run

        answer = _last_ai(response)
        if _structured(response) or answer is None:
            return self._recorded(response, tally)
        if answer.tool_calls:
            projected = tally.tool_calls + len(answer.tool_calls)
            if projected <= self.max_tool_calls:
                return self._recorded(response, tally.dispatched(answer))
            log.warning(
                "tool-loop budget discarded a round that would pass the cap",
                extra={
                    "extra_data": {
                        "tool_calls": tally.tool_calls,
                        "round_tool_calls": len(answer.tool_calls),
                        "max_tool_calls": self.max_tool_calls,
                    }
                },
            )
            answered = yield from self._forced_answer(request, tally.tool_calls, "tool_calls", self.force_text)
            return self._recorded(answered, tally)
        if _visible_text(answer):
            return self._recorded(response, tally)
        answered = yield from self._answer_or_fallback(request, response, answer, tally.tool_calls)
        return self._recorded(answered, tally)

    @staticmethod
    def _recorded(response: ModelResponse, tally: _TurnTally) -> ExtendedModelResponse:
        """``response`` carrying the turn's record as a state update.

        :param response: the step's response
        :ptype response: ModelResponse
        :param tally: the turn's record after this step
        :ptype tally: _TurnTally
        :return: the response with its command
        :rtype: ExtendedModelResponse
        """
        return ExtendedModelResponse(model_response=response, command=Command(update={_STATE_KEY: tally.to_state()}))

    def _forced_answer(
        self, request: ModelRequest, tool_calls: int, cause: str, instruction: str
    ) -> Generator[ModelRequest, ModelResponse, ModelResponse]:
        """Call the model told to answer; rebuild its reply as text and guard it against being empty.

        :param request: the model request
        :ptype request: ModelRequest
        :param tool_calls: the turn's tool calls so far
        :ptype tool_calls: int
        :param cause: what forced it: ``"tool_calls"``, ``"context_window"`` or ``"runaway"``
        :ptype cause: str
        :param instruction: the instruction appended for the pass
        :ptype instruction: str
        :return: the step's response
        :rtype: Generator[ModelRequest, ModelResponse, ModelResponse]
        """
        log.warning(
            "tool-loop budget forced an answer",
            extra={"extra_data": {"cause": cause, "tool_calls": tool_calls, "max_tool_calls": self.max_tool_calls}},
        )
        response = yield self._answer_only(request, instruction)
        if _structured(response):
            return response
        answer = _last_ai(response) or AIMessage(content="")
        # Told to answer, the model can still emit a call; dropping it (and any result it carries)
        # is what ends the turn.
        response = _with_answer(response, answer)
        if _visible_text(answer):
            return response
        return (yield from self._answer_or_fallback(request, response, answer, tool_calls))

    def _answer_or_fallback(
        self, request: ModelRequest, response: ModelResponse, empty: AIMessage, tool_calls: int
    ) -> Generator[ModelRequest, ModelResponse, ModelResponse]:
        """Retry an empty answer once, strictly; else end the turn with the fallback text.

        :param request: the model request (the original, before any instruction was appended)
        :ptype request: ModelRequest
        :param response: the response whose answer was empty
        :ptype response: ModelResponse
        :param empty: the empty answer
        :ptype empty: AIMessage
        :param tool_calls: the turn's tool calls so far
        :ptype tool_calls: int
        :return: the step's response
        :rtype: Generator[ModelRequest, ModelResponse, ModelResponse]
        """
        reasoning = _reasoning_text(empty)
        log.warning(
            "tool-loop budget retrying an empty answer",
            extra={"extra_data": {"tool_calls": tool_calls, "had_reasoning": bool(reasoning)}},
        )
        retry = self.retry_text
        if reasoning:
            retry = _RETRY_REASONING_PREFIX.format(reasoning=reasoning) + retry
        retried: ModelResponse | None
        try:
            retried = yield self._answer_only(request, retry)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            log.warning(
                "tool-loop budget retry failed; answering with the fallback",
                extra={"extra_data": {"tool_calls": tool_calls, "error_type": type(exc).__name__}},
            )
            retried = None
        if retried is not None and _structured(retried):
            return retried
        answer = _last_ai(retried)
        if retried is not None and answer is not None and _visible_text(answer):
            return _with_answer(retried, answer)
        log.warning(
            "tool-loop budget answered with the fallback text",
            extra={"extra_data": {"tool_calls": tool_calls, "max_tool_calls": self.max_tool_calls}},
        )
        stand_in = request.override(model=_FallbackChatModel(text=self.fallback_text), tool_choice=None)
        try:
            said = yield stand_in
        except GraphBubbleUp:
            raise
        except Exception as exc:
            # The chain refused even the stand-in; the reply still lands in the history, unstreamed.
            log.warning(
                "tool-loop budget fallback call failed; answering without streaming",
                extra={"extra_data": {"tool_calls": tool_calls, "error_type": type(exc).__name__}},
            )
            return _with_answer(response, AIMessage(content=self.fallback_text, name=empty.name))
        return _with_answer(said, _last_ai(said) or AIMessage(content=self.fallback_text, name=empty.name))

    # --- the seams ------------------------------------------------------------------------------

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ExtendedModelResponse:
        """Run the budget's plan for one model step, synchronously.

        :param request: the model request
        :ptype request: ModelRequest
        :param handler: runs a request through the rest of the chain and the model
        :ptype handler: Callable[[ModelRequest], ModelResponse]
        :return: the model step's response, with the turn's record as its command
        :rtype: ExtendedModelResponse
        """
        plan = self._plan(request)
        spent: list[ModelResponse] = []
        try:
            to_run = next(plan)
            while True:
                try:
                    outcome = handler(to_run)
                except Exception as exc:  # noqa: BLE001 - handed to the plan, which re-raises all but a retry's failure
                    to_run = plan.throw(exc)
                else:
                    spent.append(outcome)
                    to_run = plan.send(outcome)
        except StopIteration as done:
            return _metered(cast("ExtendedModelResponse", done.value), spent)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ExtendedModelResponse:
        """Run the budget's plan for one model step.

        :param request: the model request
        :ptype request: ModelRequest
        :param handler: runs a request through the rest of the chain and the model
        :ptype handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
        :return: the model step's response, with the turn's record as its command
        :rtype: ExtendedModelResponse
        """
        plan = self._plan(request)
        spent: list[ModelResponse] = []
        try:
            to_run = next(plan)
            while True:
                try:
                    outcome = await handler(to_run)
                except Exception as exc:  # noqa: BLE001 - handed to the plan, which re-raises all but a retry's failure
                    to_run = plan.throw(exc)
                else:
                    spent.append(outcome)
                    to_run = plan.send(outcome)
        except StopIteration as done:
            return _metered(cast("ExtendedModelResponse", done.value), spent)
