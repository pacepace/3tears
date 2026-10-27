"""The tool-turn port: one turn of a model that calls tools, and what it hands back.

:class:`ToolTurnClient` is what a loop that lets a model call tools is handed: ``turn`` sends the conversation so far
(a system prompt, the messages, the tool definitions and an output cap) and answers the model's next message as a
:class:`TurnReply`; ``turn_price_ceiling`` bounds that turn's cost before it is sent, so a budget can refuse a turn
that could overrun it and send nothing. The loop appends the reply's own message (:meth:`TurnReply.message`) to the
conversation, runs each tool call it holds, and answers them in one user message of :class:`TurnToolResult`\\ s.

**The blocks are Anthropic's content blocks.** Tool turns go to Anthropic's Messages API alone (the shipped client
refuses any other route), so a block's JSON (:meth:`TurnText.to_json` and its siblings, read back by
:func:`block_from_json`) is the block as that API writes and reads it. A :class:`TurnThinking` keeps the block exactly
as the reply carried it, signature included, because the API checks a thinking block sent back against what it wrote.

**Unknown is not zero.** Every count and the cost of a :class:`TurnReply` is ``None`` where the provider reported
nothing or nothing priced it, never ``0``; ``pricing_failure`` says why a cost is ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self, get_args

__all__ = [
    "TurnText",
    "TurnThinking",
    "TurnToolResult",
    "TurnToolSpec",
    "ToolTurnClient",
    "TurnToolUse",
    "TurnBlock",
    "TurnMessage",
    "TurnReply",
    "TurnReplyRefused",
    "TurnStopReason",
    "block_from_json",
]

#: Why a turn stopped, in the engine's words. ``tool_use`` is the model waiting on the tool calls its message holds;
#: ``end_turn`` is a message that is done (a stop sequence included); ``max_tokens`` is the output cap reached;
#: ``content_filter`` is the model refusing; ``truncated`` is output that stopped early for a cause that is neither
#: (the context window filling, a server pause) or an absent or unknown reason, which ``provider_stop_reason`` names.
TurnStopReason = Literal["end_turn", "tool_use", "max_tokens", "content_filter", "truncated"]

#: Every :data:`TurnStopReason`, which :meth:`TurnReply.from_json` checks a stored one against.
_TURN_STOP_REASONS: frozenset[str] = frozenset(get_args(TurnStopReason))


@dataclass(frozen=True, slots=True)
class TurnText:
    """Text, written by the model or by the loop.

    :ivar text: The text.
    """

    text: str

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The block as the Messages API writes it.

        :return: ``{"type": "text", "text": ...}``.
        :rtype: dict[str, Any]
        """
        return {"type": "text", "text": self.text}


@dataclass(frozen=True, slots=True)
class TurnToolUse:
    """One tool call the model's message makes.

    :ivar id: The call's id, which its :class:`TurnToolResult` names.
    :ivar name: The tool called.
    :ivar input: The call's arguments, as the model wrote them.
    """

    id: str
    name: str
    input: Mapping[str, Any]  # Any: the model's JSON arguments

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The block as the Messages API writes it.

        :return: ``{"type": "tool_use", "id", "name", "input"}``.
        :rtype: dict[str, Any]
        """
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": dict(self.input)}


@dataclass(frozen=True, slots=True)
class TurnThinking:
    """The model's thinking, kept exactly as the reply carried it, to be sent back unchanged.

    The Messages API checks a thinking block sent back against the signature it wrote, so the block is held as the
    JSON object it arrived as (a ``thinking`` block with its ``signature``, or a ``redacted_thinking`` block with its
    ``data``), and :meth:`to_json` answers that object, never a rebuilt one.

    :ivar block: The block as the reply carried it.
    """

    block: Mapping[str, Any]  # Any: the provider's JSON object, echoed unchanged

    @property
    def text(self) -> str:
        """The thinking's text, or ``""`` for a redacted block."""
        thinking = self.block.get("thinking")
        return thinking if isinstance(thinking, str) else ""

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The block exactly as the reply carried it.

        :return: The block.
        :rtype: dict[str, Any]
        """
        return _copied(self.block)


@dataclass(frozen=True, slots=True)
class TurnToolResult:
    """What one tool call answered, sent back in the next user message.

    :ivar tool_use_id: The id of the :class:`TurnToolUse` it answers.
    :ivar content: The tool's result text.
    :ivar is_error: Whether the tool refused the call, which the model reads as a failure to recover from.
    """

    tool_use_id: str
    content: str
    is_error: bool = False

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The block as the Messages API reads it; ``is_error`` is written only on a refusal.

        :return: ``{"type": "tool_result", "tool_use_id", "content"}``, plus ``"is_error": true`` on a refusal.
        :rtype: dict[str, Any]
        """
        written: dict[str, Any] = {"type": "tool_result", "tool_use_id": self.tool_use_id, "content": self.content}
        if self.is_error:
            written["is_error"] = True
        return written


#: One block of a turn's message.
TurnBlock = TurnText | TurnToolUse | TurnThinking | TurnToolResult


def block_from_json(value: Mapping[str, Any]) -> TurnBlock:  # Any: a JSON object
    """A block read from its JSON, as the Messages API writes it.

    :param value: The block's JSON object.
    :ptype value: Mapping[str, Any]
    :return: The block.
    :rtype: TurnBlock
    :raises ValueError: A block type a tool turn does not carry, or a block missing a member, named.
    """
    kind = value.get("type")
    try:
        if kind == "text":
            return TurnText(_text(value, "text"))
        if kind == "tool_use":
            arguments = value.get("input")
            if not isinstance(arguments, Mapping):
                raise ValueError("its input is not a JSON object")
            return TurnToolUse(_text(value, "id"), _text(value, "name"), _copied(arguments))
        if kind in ("thinking", "redacted_thinking"):
            return TurnThinking(_copied(value))
        if kind == "tool_result":
            return TurnToolResult(
                _text(value, "tool_use_id"), _text(value, "content"), is_error=value.get("is_error") is True
            )
    except ValueError as missing:
        raise ValueError(f"a {kind} block that cannot be read: {missing}") from missing
    raise ValueError(
        f"a {kind!r} block is not one a tool turn carries (text, tool_use, thinking, redacted_thinking, tool_result)"
    )


def _text(value: Mapping[str, Any], member: str) -> str:  # Any: a JSON object
    found = value.get(member)
    if not isinstance(found, str):
        raise ValueError(f"its {member} is not a string")
    return found


def _copied(value: Mapping[str, Any]) -> dict[str, Any]:  # Any: a JSON object
    """A deep copy of a JSON object, so a block never shares a nested mapping with what it was read from."""
    return {key: _copied_value(member) for key, member in value.items()}


def _copied_value(value: object) -> Any:  # Any: a JSON value
    if isinstance(value, Mapping):
        return _copied(value)
    if isinstance(value, list | tuple):
        return [_copied_value(member) for member in value]
    return value


@dataclass(frozen=True, slots=True)
class TurnMessage:
    """One message of the conversation a turn sends.

    :ivar role: ``user`` (the loop: the opening brief, tool results) or ``assistant`` (a reply sent back).
    :ivar content: Its blocks; a string is one :class:`TurnText`.
    """

    role: Literal["user", "assistant"]
    content: tuple[TurnBlock, ...]

    def __init__(self, role: Literal["user", "assistant"], content: str | Sequence[TurnBlock]) -> None:
        """Build the message.

        :param role: ``user`` or ``assistant``.
        :ptype role: Literal["user", "assistant"]
        :param content: Its blocks, or a string for one text block.
        :ptype content: str | Sequence[TurnBlock]
        :raises ValueError: A role that is neither, or no blocks.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"a turn message's role is user or assistant, not {role!r}")
        blocks = (TurnText(content),) if isinstance(content, str) else tuple(content)
        if not blocks:
            raise ValueError("a turn message holds at least one block")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", blocks)

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The message as the Messages API reads it.

        :return: ``{"role", "content": [block, ...]}``.
        :rtype: dict[str, Any]
        """
        return {"role": self.role, "content": [block.to_json() for block in self.content]}


@dataclass(frozen=True, slots=True)
class TurnToolSpec:
    """One tool the model may call: its name, what it does, and its arguments' JSON schema.

    :ivar name: The tool's name, which the model's calls name.
    :ivar description: What the tool does, as the model reads it.
    :ivar input_schema: The JSON schema of its arguments.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]  # Any: a JSON schema

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The definition as the Messages API reads it.

        :return: ``{"name", "description", "input_schema"}``.
        :rtype: dict[str, Any]
        """
        return {"name": self.name, "description": self.description, "input_schema": _copied(self.input_schema)}


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnReply:
    """The model's next message, why it stopped, what it used and what it cost.

    :ivar content: The message's blocks, in order: thinking, text and tool calls.
    :ivar stop_reason: Why it stopped, in the engine's words (:data:`TurnStopReason`).
    :ivar provider_stop_reason: The reason as the provider named it, or ``None``.
    :ivar model: The model id the turn was sent to.
    :ivar served_model: The model the reply named, or ``None`` when it named none.
    :ivar input_tokens: Prompt tokens, cache reads and writes included; ``None`` when unreported.
    :ivar output_tokens: Output tokens, thinking included; ``None`` when unreported.
    :ivar reasoning_tokens: The thinking share of ``output_tokens``; ``None`` when unsplit.
    :ivar cache_read_tokens: The prompt tokens read from the cache; ``None`` when unreported.
    :ivar cache_write_tokens: The prompt tokens written to the cache; ``None`` when unreported.
    :ivar cost_usd: What the turn cost, or ``None`` when nothing priced it (never ``0`` for unpriced).
    :ivar price_source: Who priced it, or ``None`` when unpriced.
    :ivar pricing_failure: Why ``cost_usd`` is ``None``; ``None`` when priced.
    """

    content: tuple[TurnBlock, ...]
    stop_reason: TurnStopReason
    provider_stop_reason: str | None
    model: str
    served_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    pricing_failure: str | None

    @property
    def text(self) -> str:
        """The message's text blocks, joined; never a thinking block."""
        return "".join(block.text for block in self.content if isinstance(block, TurnText))

    @property
    def tool_uses(self) -> tuple[TurnToolUse, ...]:
        """The tool calls the message makes, in order."""
        return tuple(block for block in self.content if isinstance(block, TurnToolUse))

    def message(self) -> TurnMessage:
        """The reply as the assistant message the next turn sends back, every block (thinking included) unchanged.

        :return: The message.
        :rtype: TurnMessage
        """
        return TurnMessage("assistant", self.content)

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The reply as JSON, read back by :meth:`from_json`.

        :return: Every member, the blocks as the Messages API writes them.
        :rtype: dict[str, Any]
        """
        return {
            "content": [block.to_json() for block in self.content],
            "stop_reason": self.stop_reason,
            "provider_stop_reason": self.provider_stop_reason,
            "model": self.model,
            "served_model": self.served_model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": self.cost_usd,
            "price_source": self.price_source,
            "pricing_failure": self.pricing_failure,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Self:  # Any: a JSON object
        """A reply read back from :meth:`to_json`'s JSON.

        :param value: The JSON object.
        :ptype value: Mapping[str, Any]
        :return: The reply.
        :rtype: TurnReply
        :raises ValueError: A member missing, extra or of the wrong type, named.
        """
        members = set(value)
        if members != _REPLY_MEMBERS:
            missing, extra = sorted(_REPLY_MEMBERS - members), sorted(members - _REPLY_MEMBERS)
            raise ValueError(f"a turn reply's members differ: missing {missing}, unexpected {extra}")
        content = value["content"]
        if not isinstance(content, list) or not all(isinstance(block, Mapping) for block in content):
            raise ValueError("a turn reply's content is not a list of blocks")
        stop = value["stop_reason"]
        if stop not in _TURN_STOP_REASONS:
            raise ValueError(f"a turn reply's stop_reason {stop!r} is not one of {sorted(_TURN_STOP_REASONS)}")
        cost = value["cost_usd"]
        if cost is not None and (isinstance(cost, bool) or not isinstance(cost, int | float)):
            raise ValueError(f"a turn reply's cost_usd {cost!r} is neither a number nor null")
        return cls(
            content=tuple(block_from_json(block) for block in content),
            stop_reason=stop,
            provider_stop_reason=_optional_text(value, "provider_stop_reason"),
            model=_text(value, "model"),
            served_model=_optional_text(value, "served_model"),
            input_tokens=_optional_count(value, "input_tokens"),
            output_tokens=_optional_count(value, "output_tokens"),
            reasoning_tokens=_optional_count(value, "reasoning_tokens"),
            cache_read_tokens=_optional_count(value, "cache_read_tokens"),
            cache_write_tokens=_optional_count(value, "cache_write_tokens"),
            cost_usd=None if cost is None else float(cost),
            price_source=_optional_text(value, "price_source"),
            pricing_failure=_optional_text(value, "pricing_failure"),
        )


#: The members :meth:`TurnReply.to_json` writes, and :meth:`TurnReply.from_json` reads.
_REPLY_MEMBERS = frozenset(
    {
        "content",
        "stop_reason",
        "provider_stop_reason",
        "model",
        "served_model",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd",
        "price_source",
        "pricing_failure",
    }
)


def _optional_text(value: Mapping[str, Any], member: str) -> str | None:  # Any: a JSON object
    found = value.get(member)
    if found is not None and not isinstance(found, str):
        raise ValueError(f"a turn reply's {member} is neither a string nor null")
    return found


def _optional_count(value: Mapping[str, Any], member: str) -> int | None:  # Any: a JSON object
    found = value.get(member)
    if found is not None and (isinstance(found, bool) or not isinstance(found, int)):
        raise ValueError(f"a turn reply's {member} is neither a count nor null")
    return found


class TurnReplyRefused(Exception):
    """A turn that was sent and answered, so paid for, whose reply the client then refuses to hand back.

    The loop never reads the reply, but the turn was billed, so the reply rides on the refusal and the loop counts its
    tokens and cost before the refusal goes on to its caller. A recording raises one (as its own
    ``RecordingRefused`` too) for a reply holding the literal ``{snapshot}``, which a replay could not tell from the
    placeholder.

    :ivar reply: The reply refused, as the provider answered it; set by whoever raises it.
    """

    reply: TurnReply


class ToolTurnClient(Protocol):
    """The client a tool-calling loop is handed: one turn at a time, priced before it is sent.

    The shipped implementation is ``threetears.evals.providers.Model`` (``Model.turn``, and ``Model.turn_session``
    for a recorded session); a replay serves recorded replies through the same port
    (``threetears.evals.testing.RecordedTurns``).
    """

    async def turn(
        self, *, system: str, messages: Sequence[TurnMessage], tools: Sequence[TurnToolSpec], max_tokens: int
    ) -> TurnReply:
        """Send the conversation so far and answer the model's next message.

        :param system: The system prompt.
        :ptype system: str
        :param messages: The conversation, oldest first, ending on the message the model answers.
        :ptype messages: Sequence[TurnMessage]
        :param tools: The tools the model may call.
        :ptype tools: Sequence[TurnToolSpec]
        :param max_tokens: The turn's output cap, thinking included.
        :ptype max_tokens: int
        :return: The model's reply.
        :rtype: TurnReply
        """
        ...

    def turn_price_ceiling(
        self, *, system: str, messages: Sequence[TurnMessage], tools: Sequence[TurnToolSpec], max_tokens: int
    ) -> float | None:
        """The most this turn can cost, in dollars, or ``None`` when the client cannot say (never a lower figure).

        :param system: The system prompt the turn would send.
        :ptype system: str
        :param messages: The conversation it would send.
        :ptype messages: Sequence[TurnMessage]
        :param tools: The tools it would send.
        :ptype tools: Sequence[TurnToolSpec]
        :param max_tokens: Its output cap.
        :ptype max_tokens: int
        :return: The ceiling, or ``None``.
        :rtype: float | None
        """
        ...
