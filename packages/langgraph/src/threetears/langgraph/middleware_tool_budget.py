"""A per-turn tool-loop budget for ``langchain.agents.create_agent``, so a looping model still answers.

A model that keeps calling tools runs a ``create_agent`` graph into LangGraph's recursion limit,
and the turn ends on ``GraphRecursionError`` with no reply. :class:`ToolLoopBudgetMiddleware` ends
the loop first, and always with text:

- **A per-turn cap.** At most ``max_tool_calls`` tool calls per turn (a turn is everything after
  the last ``HumanMessage``). The cap is absolute: a response whose calls would take the turn past
  it is discarded and the forced answer pass runs instead.
- **Escalating tiers.** Two pressures are measured before every model call -- tool calls against
  the cap, and the prompt against the model's context window -- and the larger sets the tier.
  At ``warn`` and ``urgent`` the model is shown a wrap-up nudge; at ``force`` it is not offered
  tools at all. Calls reach ``warn``/``urgent`` at ``warn_pct``/``urgent_pct`` of the cap and
  ``force`` AT the cap; the window reaches each tier at its pct, ``force_pct`` below 100 so the
  answer has room to generate.
- **A forced answer pass.** At ``force`` the model is called with no tools and told to answer
  from what it already has. Any tool call it still returns is dropped, so the turn ends.
- **An empty-answer guard.** A final answer with no visible text (empty, or reasoning only) is
  retried once, with no tools, a stricter instruction and the model's own reasoning; if that is
  empty too the turn ends with ``fallback_text``. Never an empty turn.
- **A runaway guard.** ``runaway_rounds`` consecutive rounds that wrote no text and whose every
  tool call failed force the answer pass, however far below the cap the turn is.

It is stateless: everything is derived from the state's messages on each call, so one instance
serves every thread, and a resumed checkpoint sees the same count. Nudges and instructions are
request overrides, never written to state. Install it first in the ``middleware`` list, so it sees
the state's messages before any other middleware rewrites the request.

:func:`recursion_limit_for` gives the graph a ``recursion_limit`` the budget can never reach, so
the budget, not LangGraph's recursion error, is what ends a loop.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.errors import GraphBubbleUp

from threetears.observe import get_logger

__all__ = [
    "DEFAULT_MAX_TOOL_CALLS_PER_TURN",
    "DEFAULT_TOOL_BUDGET_FALLBACK_TEXT",
    "ToolBudgetTier",
    "ToolLoopBudgetMiddleware",
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


@dataclass(frozen=True)
class _TurnTally:
    """What the current turn has done so far.

    :ivar tool_calls: tool calls the turn's rounds made
    :ivar rounds: model responses in the turn that called tools
    :ivar failing_rounds: trailing consecutive rounds with no text whose every tool result failed
    """

    tool_calls: int
    rounds: int
    failing_rounds: int


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


def _turn_messages(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """The current turn: every message after the last ``HumanMessage``.

    :param messages: the conversation
    :ptype messages: Sequence[BaseMessage]
    :return: the turn's messages
    :rtype: Sequence[BaseMessage]
    """
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index + 1 :]
    return messages


def _tally(messages: Sequence[BaseMessage]) -> _TurnTally:
    """Count the current turn's tool calls, its tool rounds, and its trailing failing rounds.

    A round fails when its model response wrote no text and every one of its tool results is an
    error -- in ``create_agent`` terms, metallm's round with no text and no valid dispatch.

    :param messages: the conversation
    :ptype messages: Sequence[BaseMessage]
    :return: the tally
    :rtype: _TurnTally
    """
    turn = _turn_messages(messages)
    statuses: dict[str, str] = {m.tool_call_id: m.status for m in turn if isinstance(m, ToolMessage)}
    tool_calls = rounds = failing = 0
    for message in turn:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        rounds += 1
        tool_calls += len(message.tool_calls)
        results = [statuses[cid] for c in message.tool_calls if (cid := c.get("id")) is not None and cid in statuses]
        all_failed = bool(results) and all(s == "error" for s in results)
        failing = failing + 1 if all_failed and not _visible_text(message) else 0
    return _TurnTally(tool_calls=tool_calls, rounds=rounds, failing_rounds=failing)


def _without_last_round(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """The conversation as it stood before the turn's latest tool round.

    :param messages: the conversation
    :ptype messages: Sequence[BaseMessage]
    :return: the messages before the last tool-calling ``AIMessage``
    :rtype: Sequence[BaseMessage]
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.tool_calls:
            return messages[:index]
    return messages


def _response_messages(result: Any) -> list[BaseMessage]:
    """The messages a handler result carries, whatever shape the handler returned.

    :param result: a ``ModelResponse``, ``ExtendedModelResponse`` or ``AIMessage``
    :ptype result: Any
    :return: the messages
    :rtype: list[BaseMessage]
    """
    if isinstance(result, ExtendedModelResponse):
        return list(result.model_response.result)
    if isinstance(result, ModelResponse):
        return list(result.result)
    if isinstance(result, AIMessage):
        return [result]
    return []


def _structured(result: Any) -> bool:
    """Whether the result carries a structured response (the consumer's response format answered).

    :param result: the handler result
    :ptype result: Any
    :return: whether a structured response is present
    :rtype: bool
    """
    if isinstance(result, ExtendedModelResponse):
        result = result.model_response
    return isinstance(result, ModelResponse) and result.structured_response is not None


def _last_ai(result: Any) -> AIMessage | None:
    """The last ``AIMessage`` in a handler result.

    :param result: the handler result
    :ptype result: Any
    :return: the message, or ``None`` when the result has none
    :rtype: AIMessage | None
    """
    for message in reversed(_response_messages(result)):
        if isinstance(message, AIMessage):
            return message
    return None


def _with_ai(result: Any, replacement: AIMessage) -> Any:
    """``result`` with its last ``AIMessage`` replaced, in the shape it came in.

    :param result: the handler result
    :ptype result: Any
    :param replacement: the message to put in place of the last ``AIMessage``
    :ptype replacement: AIMessage
    :return: the result, rebuilt
    :rtype: Any
    """
    messages = _response_messages(result)
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage):
            messages[index] = replacement
            break
    else:
        messages.append(replacement)
    if isinstance(result, ExtendedModelResponse):
        inner = ModelResponse(result=messages, structured_response=result.model_response.structured_response)
        return ExtendedModelResponse(model_response=inner, command=result.command)
    if isinstance(result, ModelResponse):
        return ModelResponse(result=messages, structured_response=result.structured_response)
    return replacement


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
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS_PER_TURN,
    *,
    middleware: Sequence[AgentMiddleware] = (),
    margin: int = 3,
) -> int:
    """A ``recursion_limit`` for a ``create_agent`` graph that the tool budget can never reach.

    Under :class:`ToolLoopBudgetMiddleware` a turn makes at most ``max_tool_calls`` tool rounds
    (each round makes at least one call) and one more model call that answers. Each model call
    costs the model node plus one node per ``before_model`` and ``after_model`` hook; each tool
    round costs the tools node; ``before_agent`` and ``after_agent`` hooks cost one node each,
    once; and the run's input is one step of its own. With ``margin=0`` it is exactly the steps
    of the longest turn the budget allows. Pass the same ``middleware`` list the agent is built
    with so the limit follows it.

    Put the result on the run config: ``{"recursion_limit": recursion_limit_for(...)}``.

    :param max_tool_calls: the budget's per-turn cap
    :ptype max_tool_calls: int
    :param middleware: the agent's middleware, whose hooks add graph nodes
    :ptype middleware: Sequence[AgentMiddleware]
    :param margin: slack above the worst case
    :ptype margin: int
    :return: the limit
    :rtype: int
    """
    if max_tool_calls < 1:
        raise ValueError(f"max_tool_calls must be at least 1, got {max_tool_calls}")
    per_model_call = 1 + sum(
        _middleware_overrides(m, "before_model") + _middleware_overrides(m, "after_model") for m in middleware
    )
    once = sum(_middleware_overrides(m, "before_agent") + _middleware_overrides(m, "after_agent") for m in middleware)
    tools_steps = (_MODEL_AND_TOOLS_STEPS - 1) * max_tool_calls
    return _INPUT_STEPS + (max_tool_calls + 1) * per_model_call + tools_steps + once + margin


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
    """

    name = "ToolLoopBudgetMiddleware"

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

    def recursion_limit(self, *, middleware: Sequence[AgentMiddleware] = (), margin: int = 3) -> int:
        """:func:`recursion_limit_for` at this budget's cap.

        :param middleware: the agent's middleware, whose hooks add graph nodes
        :ptype middleware: Sequence[AgentMiddleware]
        :param margin: slack above the worst case
        :ptype margin: int
        :return: the limit
        :rtype: int
        """
        return recursion_limit_for(self.max_tool_calls, middleware=middleware, margin=margin)

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

    def _window_pct(self, request: ModelRequest, messages: Sequence[BaseMessage]) -> float:
        """The prompt ``messages`` make, with the system message, as a percent of the window.

        :param request: the model request
        :ptype request: ModelRequest
        :param messages: the conversation
        :ptype messages: Sequence[BaseMessage]
        :return: the percent; ``0.0`` when the window is unknown
        :rtype: float
        """
        window = self._window(request)
        if window is None:
            return 0.0
        prompt = [request.system_message, *messages] if request.system_message is not None else list(messages)
        return self.count_tokens(prompt) / window * 100

    def _assess(self, request: ModelRequest, messages: Sequence[BaseMessage]) -> tuple[ToolBudgetTier, str, float]:
        """The turn's tier, which pressure set it, and the window percent.

        :param request: the model request
        :ptype request: ModelRequest
        :param messages: the conversation
        :ptype messages: Sequence[BaseMessage]
        :return: ``(tier, cause, window_pct)``; cause is ``"tool_calls"`` or ``"context_window"``
        :rtype: tuple[ToolBudgetTier, str, float]
        """
        tally = _tally(messages)
        if tally.rounds == 0:
            return ToolBudgetTier.NONE, "tool_calls", 0.0
        pct = self._window_pct(request, messages)
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

    def _tool_less(self, request: ModelRequest, text: str) -> ModelRequest:
        """``request`` with no tools offered and ``text`` appended as the last message.

        :param request: the model request
        :ptype request: ModelRequest
        :param text: the instruction
        :ptype text: str
        :return: the overridden request
        :rtype: ModelRequest
        """
        return request.override(tools=[], tool_choice=None, messages=[*request.messages, self._instruction(text)])

    # --- the plan both seams drive ---------------------------------------------------------------

    def _plan(self, request: ModelRequest) -> Generator[ModelRequest, Any, Any]:
        """The budget's model calls for one ``create_agent`` model step.

        Yields each request to run; receives its result (or has the call's exception thrown in);
        returns what the model step returns. :meth:`wrap_model_call` and :meth:`awrap_model_call`
        drive it, so the two seams cannot drift.

        :param request: the model request
        :ptype request: ModelRequest
        :return: the step's result
        :rtype: Generator[ModelRequest, Any, Any]
        """
        conversation = self._conversation(request)
        tally = _tally(conversation)
        tier, cause, pct = self._assess(request, conversation)
        if tier > ToolBudgetTier.NONE:
            before, _, _ = self._assess(request, _without_last_round(conversation))
            if tier > before:
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

        if tally.failing_rounds >= self.runaway_rounds:
            return (yield from self._forced_answer(request, tally.tool_calls, "runaway", self.runaway_text))
        if tier is ToolBudgetTier.FORCE:
            return (yield from self._forced_answer(request, tally.tool_calls, cause, self.force_text))

        to_run = request
        if tier in (ToolBudgetTier.WARN, ToolBudgetTier.URGENT):
            template = self.warn_text if tier is ToolBudgetTier.WARN else self.urgent_text
            nudge = template.format(tool_calls=tally.tool_calls, max_tool_calls=self.max_tool_calls)
            to_run = request.override(messages=[*request.messages, self._instruction(nudge)])
        result = yield to_run

        if _structured(result):
            return result
        answer = _last_ai(result)
        if answer is None:
            return result
        if answer.tool_calls:
            projected = tally.tool_calls + len(answer.tool_calls)
            if projected <= self.max_tool_calls:
                return result
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
            return (yield from self._forced_answer(request, tally.tool_calls, "tool_calls", self.force_text))
        if _visible_text(answer):
            return result
        return (yield from self._answer_or_fallback(request, result, answer, tally.tool_calls))

    def _forced_answer(
        self, request: ModelRequest, tool_calls: int, cause: str, instruction: str
    ) -> Generator[ModelRequest, Any, Any]:
        """Call the model with no tools, told to answer; guard the answer against being empty.

        :param request: the model request
        :ptype request: ModelRequest
        :param tool_calls: the turn's tool calls so far
        :ptype tool_calls: int
        :param cause: what forced it: ``"tool_calls"``, ``"context_window"`` or ``"runaway"``
        :ptype cause: str
        :param instruction: the instruction appended for the pass
        :ptype instruction: str
        :return: the step's result
        :rtype: Generator[ModelRequest, Any, Any]
        """
        log.warning(
            "tool-loop budget forced an answer",
            extra={"extra_data": {"cause": cause, "tool_calls": tool_calls, "max_tool_calls": self.max_tool_calls}},
        )
        result = yield self._tool_less(request, instruction)
        if _structured(result):
            return result
        answer = _last_ai(result)
        if answer is None:
            answer = AIMessage(content="")
        if answer.tool_calls or answer.invalid_tool_calls:
            # Offered no tools, the model can still emit a call; dropping it is what ends the turn.
            answer = answer.model_copy(update={"tool_calls": [], "invalid_tool_calls": []})
            result = _with_ai(result, answer)
        if _visible_text(answer):
            return result
        return (yield from self._answer_or_fallback(request, result, answer, tool_calls))

    def _answer_or_fallback(
        self, request: ModelRequest, result: Any, empty: AIMessage, tool_calls: int
    ) -> Generator[ModelRequest, Any, Any]:
        """Retry an empty answer once, strictly and with no tools; else end the turn with the fallback.

        :param request: the model request (the original, before any instruction was appended)
        :ptype request: ModelRequest
        :param result: the result whose answer was empty
        :ptype result: Any
        :param empty: the empty answer
        :ptype empty: AIMessage
        :param tool_calls: the turn's tool calls so far
        :ptype tool_calls: int
        :return: the step's result
        :rtype: Generator[ModelRequest, Any, Any]
        """
        reasoning = _reasoning_text(empty)
        log.warning(
            "tool-loop budget retrying an empty answer",
            extra={"extra_data": {"tool_calls": tool_calls, "had_reasoning": bool(reasoning)}},
        )
        retry = self.retry_text
        if reasoning:
            retry = _RETRY_REASONING_PREFIX.format(reasoning=reasoning) + retry
        try:
            retried = yield self._tool_less(request, retry)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            log.warning(
                "tool-loop budget retry failed; answering with the fallback",
                extra={"extra_data": {"tool_calls": tool_calls, "error_type": type(exc).__name__}},
            )
            retried = None
        answer = _last_ai(retried) if retried is not None else None
        if answer is not None and _visible_text(answer):
            return _with_ai(retried, answer.model_copy(update={"tool_calls": [], "invalid_tool_calls": []}))
        log.warning(
            "tool-loop budget answered with the fallback text",
            extra={"extra_data": {"tool_calls": tool_calls, "max_tool_calls": self.max_tool_calls}},
        )
        return _with_ai(result, AIMessage(content=self.fallback_text, name=empty.name))

    # --- the seams ------------------------------------------------------------------------------

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        """Run the budget's plan for one model step, synchronously.

        :param request: the model request
        :ptype request: ModelRequest
        :param handler: runs a request through the rest of the chain and the model
        :ptype handler: Callable[[ModelRequest], ModelResponse]
        :return: the model step's result
        :rtype: ModelResponse | AIMessage | ExtendedModelResponse
        """
        plan = self._plan(request)
        try:
            to_run = next(plan)
            while True:
                try:
                    outcome = handler(to_run)
                except Exception as exc:  # noqa: BLE001 - handed to the plan, which re-raises all but a retry's failure
                    to_run = plan.throw(exc)
                else:
                    to_run = plan.send(outcome)
        except StopIteration as done:
            return done.value

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]) -> Any:
        """Run the budget's plan for one model step.

        :param request: the model request
        :ptype request: ModelRequest
        :param handler: runs a request through the rest of the chain and the model
        :ptype handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
        :return: the model step's result
        :rtype: ModelResponse | AIMessage | ExtendedModelResponse
        """
        plan = self._plan(request)
        try:
            to_run = next(plan)
            while True:
                try:
                    outcome = await handler(to_run)
                except Exception as exc:  # noqa: BLE001 - handed to the plan, which re-raises all but a retry's failure
                    to_run = plan.throw(exc)
                else:
                    to_run = plan.send(outcome)
        except StopIteration as done:
            return done.value
