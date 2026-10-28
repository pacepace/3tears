"""A session of tool turns: one model's conversation with one mounted tool, held to its budgets, stored and resumed.

The report writer's analyst and each of its editor rounds is a :class:`ToolSession`: a system text (a role of the
analyst skill, :class:`~threetears.evals.analysis.analyst.skill.SkillText`), one tool (a :class:`ToolMount`, the
``evals_analyst`` tool cut for the role and bound to its host and caller by whoever mounts it), and the messages so
far. :meth:`ToolSession.send` appends a user message and runs turns until the model answers without calling a tool,
or something stops it; each turn's tool calls are run on the mount and answered in the next user message.

**Every limit is checked before anything is sent.** Before each turn the session asks the client for its price
ceiling and the report's :class:`~threetears.evals.ops.report_writer.tally.ReportBudget` to admit it; a refusal sends
nothing and stops the session ``over_budget``. A round has a call budget (:meth:`ToolSession.open_round`): a warning
rides on the tool results once ``warn_before`` calls are left, the results that spend the last call tell the model to
submit now, any call past the budget is answered as not run, and a reply that calls a tool after the model was told to
submit stops the session ``over_budget`` without another turn. A tool result's text is cut at ``result_chars``,
saying so.

**Every turn is tallied**, whatever its reply (:class:`~threetears.evals.ops.report_writer.tally.Tally`), and a reply
the client refuses after it was paid for (:class:`~threetears.evals.schema.turns.TurnReplyRefused`) is tallied before
the refusal goes on to the caller.

**Context elision stays legal and rare.** Once a turn's input passes ``elide_over_tokens``, tool-result bodies older
than the last ``keep_results`` are replaced by ``[elided; call again]``, but only in user messages before the last
assistant message that carries thinking, and only tool results: no assistant message, so no thinking block and no
tool call, is ever changed, and every tool call keeps its result block. An elision clears at least ``keep_results``
bodies at once and stays as it is afterwards, so the prompt is rewritten (and its cache re-read) once per that many
calls, never every turn.

**A session is a stored value, and a resumed one sends what the uninterrupted one would.** :meth:`ToolSession.to_json`
keeps the system text's digest, the tool's, the limits, every message in order with each thinking block exactly as its
reply carried it, every tool call with its result, and where the round stands. A result body the mount answered is
dropped from storage only when the mount says its results are re-derivable (a read of a frozen snapshot), and the
record names each one dropped with the sha256 of the text the model was sent. :meth:`ToolSession.from_json` refuses a
system text or tool other than the stored ones, re-derives each dropped body by calling the tool again with the
call's own arguments, and refuses one that no longer reads the same; so the resumed session's next request is the
uninterrupted session's, byte for byte.

A reply cut at its output cap (``max_tokens``) is tallied and not kept: its thinking may stop short of its signature,
and the session's caller decides what to tell the model instead. A model refusal or a reply cut for another cause
stops the session ``failed``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from threetears.evals.analysis.analyst.skill import SkillText
from threetears.evals.kernel.capped_completion import refresh_prices
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally, TurnCost, WriterRole
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.turns import (
    ToolTurnClient,
    TurnBlock,
    TurnMessage,
    TurnReply,
    TurnReplyRefused,
    TurnText,
    TurnThinking,
    TurnToolResult,
    TurnToolSpec,
    TurnToolUse,
    block_from_json,
)

#: The names the analyst's session calls. ``ToolOutcome``, what a mount's call answers, joins them with the mount the
#: report writer's entry point builds (``tests/test_report_writer_exports.py`` holds the list).
__all__ = ["SessionLimits", "SessionStop", "ToolMount", "ToolSession"]

#: What an elided tool result's body reads as.
_ELIDED: Final = "[elided; call again]"

#: The storage format :meth:`ToolSession.to_json` writes and :meth:`ToolSession.from_json` reads.
_FORMAT: Final = 1


class ToolOutcome(Protocol):
    """What one call to a mounted tool answered: the text the model reads, and whether it was refused."""

    @property
    def text(self) -> str:
        """The text the model reads: the result, or the refusal that teaches."""
        ...

    @property
    def is_error(self) -> bool:
        """Whether the call was refused or failed, which the model reads as a failure to recover from."""
        ...


class ToolMount(Protocol):
    """The one tool a session's model may call, bound to the host and caller its calls run as.

    The report writer mounts the ``evals_analyst`` tool (``analyst_tools(writes=False)`` for the analyst, the editor's
    cut for the editor) over a snapshot it froze first, so no call writes.
    """

    @property
    def name(self) -> str:
        """The tool's name, which the model's calls name."""
        ...

    @property
    def description(self) -> str:
        """What the model reads about the tool: its description and the catalogue's primer."""
        ...

    @property
    def rederivable(self) -> bool:
        """Whether every result is a function of the call's arguments alone (a read of a frozen snapshot), so a
        stored session may drop a result's body and call the tool again for it."""
        ...

    def input_schema(self) -> dict[str, Any]:  # Any: a JSON schema
        """The JSON schema of the tool's arguments.

        :return: The schema.
        :rtype: dict[str, Any]
        """
        ...

    async def call(self, arguments: Mapping[str, Any]) -> ToolOutcome:  # Any: the model's JSON arguments
        """Run one call.

        :param arguments: The call's arguments, as the model wrote them.
        :ptype arguments: Mapping[str, Any]
        :return: What it answered.
        :rtype: ToolOutcome
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionLimits:
    """What every turn of a session is held to (plan §4.3).

    :ivar max_tokens: Each turn's output cap, thinking included.
    :ivar result_chars: The most characters of a tool result the model is sent; the rest is cut, saying so.
    :ivar keep_results: How many of the latest tool results an elision always keeps, and the fewest it clears.
    :ivar elide_over_tokens: The input a turn may reach before older tool-result bodies are elided.
    :ivar warn_before: How many calls before the round's budget runs out the model is warned.
    """

    max_tokens: int = 32_000
    result_chars: int = 12_000
    keep_results: int = 12
    elide_over_tokens: int = 150_000
    warn_before: int = 10

    def __post_init__(self) -> None:
        """Refuse a limit that is not a whole number above 0 (``warn_before`` may be 0).

        :raises ValueError: The limit, named.
        """
        for name in ("max_tokens", "result_chars", "keep_results", "elide_over_tokens", "warn_before"):
            value = getattr(self, name)
            floor = 0 if name == "warn_before" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(f"a session's {name} is a whole number of at least {floor}, not {value!r}")

    def to_json(self) -> dict[str, int]:
        """The limits as JSON.

        :return: Each limit by name.
        :rtype: dict[str, int]
        """
        return {
            "max_tokens": self.max_tokens,
            "result_chars": self.result_chars,
            "keep_results": self.keep_results,
            "elide_over_tokens": self.elide_over_tokens,
            "warn_before": self.warn_before,
        }


@dataclass(frozen=True, slots=True)
class SessionStop:
    """Why :meth:`ToolSession.send` stopped running turns.

    :ivar kind: ``answered``, a reply that calls no tool (the submission); ``cut``, a reply stopped at the output cap
        and not kept; ``over_budget``, a turn the dollar budget refused (nothing sent) or a tool call after the
        model was told to submit; ``failed``, the model refused, or its reply stopped short for another cause.
    :ivar reply: The last reply, or ``None`` when no turn was sent.
    :ivar reason: Why, in a sentence; empty for ``answered``.
    """

    kind: Literal["answered", "cut", "over_budget", "failed"]
    reply: TurnReply | None
    reason: str = ""


class ToolSession:
    """One model's conversation with one mounted tool: its turns, its budgets, and its stored form.

    :ivar name: The session's name (``analyst``, ``editor-1``), which its tally entries carry.
    :ivar role: The role it writes as.
    :ivar limits: What every turn is held to.
    :ivar tools: The one tool definition every turn sends.
    :ivar skill_digest: The digest of the system text it was started with.
    """

    def __init__(
        self,
        *,
        name: str,
        role: WriterRole,
        client: ToolTurnClient,
        mount: ToolMount,
        system: SkillText,
        tally: Tally,
        budget: ReportBudget,
        limits: SessionLimits | None = None,
    ) -> None:
        """Start an empty session.

        :param name: The session's name.
        :ptype name: str
        :param role: The role it writes as.
        :ptype role: WriterRole
        :param client: The client its turns go through (``Model.turn_session(name, snapshot=...)``).
        :ptype client: ToolTurnClient
        :param mount: The tool its model may call.
        :ptype mount: ToolMount
        :param system: The role's system text.
        :ptype system: SkillText
        :param tally: The report's tally, which every turn is counted in.
        :ptype tally: Tally
        :param budget: The report's dollar budget, which admits every turn.
        :ptype budget: ReportBudget
        :param limits: What every turn is held to; plan §4.3's defaults when ``None``.
        :ptype limits: SessionLimits | None
        :raises ValueError: An empty name.
        """
        if not name:
            raise ValueError("a session needs a name, which its turns are tallied and recorded under")
        self.name = name
        self.role: WriterRole = role
        self.limits = limits if limits is not None else SessionLimits()
        self.tools = (TurnToolSpec(mount.name, mount.description, mount.input_schema()),)
        self.skill_digest = system.digest
        self._system = system.text
        self._client = client
        self._mount = mount
        self._tally = tally
        self._budget = budget
        self._messages: list[TurnMessage] = []
        #: Each result the mount answered, by its call's id: the sha256 of the text the model was sent.
        self._answered: dict[str, str] = {}
        #: The results elided in the context, by their call's id.
        self._elided: set[str] = set()
        self._sent = 0
        self._last_input: int | None = None
        self._round = 0
        self._calls = 0
        self._used = 0
        self._warned = False
        self._closing = False

    @property
    def mount(self) -> ToolMount:
        """The tool the session's model calls."""
        return self._mount

    @property
    def tally(self) -> Tally:
        """The report's tally, which every turn of the session is counted in."""
        return self._tally

    @property
    def messages(self) -> tuple[TurnMessage, ...]:
        """The conversation so far, oldest first."""
        return tuple(self._messages)

    @property
    def round(self) -> int:
        """The round the session is in; 0 before its first."""
        return self._round

    @property
    def calls(self) -> int:
        """The most tool calls the current round may make."""
        return self._calls

    @property
    def calls_used(self) -> int:
        """The tool calls the current round has made."""
        return self._used

    def tools_digest(self) -> str:
        """The sha256 of the tool definitions every turn sends.

        :return: The digest.
        :rtype: str
        """
        return canonical_digest([tool.to_json() for tool in self.tools])

    def transcript_digest(self) -> str:
        """The sha256 of the messages as they now stand.

        :return: The digest.
        :rtype: str
        """
        return canonical_digest([message.to_json() for message in self._messages])

    def open_round(self, round: int, *, calls: int) -> None:
        """Start a round with its own call budget: the draft, a revision, an editor's review.

        :param round: The round's number, above the current one.
        :ptype round: int
        :param calls: The most tool calls the round may make.
        :ptype calls: int
        :raises ValueError: A round not above the current one, or a call budget below 1.
        """
        if round <= self._round:
            raise ValueError(f"round {round} does not follow round {self._round}")
        if calls < 1:
            raise ValueError(f"a round's call budget is at least 1, not {calls}")
        self._round, self._calls, self._used, self._warned, self._closing = round, calls, 0, False, False

    async def send(self, content: str | Sequence[TurnBlock]) -> SessionStop:
        """Append a user message and run turns until the model answers without a tool call, or is stopped.

        :param content: The message: a text, or its blocks.
        :ptype content: str | Sequence[TurnBlock]
        :return: Why it stopped.
        :rtype: SessionStop
        :raises ValueError: No round is open.
        :raises TurnReplyRefused: The client refused a reply it was paid for; the turn is tallied first.
        """
        if self._round == 0:
            raise ValueError("open a round (open_round) before sending: a round's call budget holds every turn")
        self._append_user(list((TurnText(content),) if isinstance(content, str) else content))
        while True:
            self._elide_if_over()
            await refresh_prices(self._client)
            ceiling = self._client.turn_price_ceiling(
                system=self._system, messages=self._messages, tools=self.tools, max_tokens=self.limits.max_tokens
            )
            refused = self._budget.admit(ceiling)
            if refused is not None:
                return SessionStop("over_budget", None, f"the report's budget refused the next turn: {refused}")
            reply = await self._turn(ceiling)
            if reply.stop_reason == "max_tokens":
                return SessionStop(
                    "cut", reply, f"the reply reached the {self.limits.max_tokens}-token output cap before it ended"
                )
            if reply.stop_reason in ("content_filter", "truncated"):
                cause = reply.provider_stop_reason or reply.stop_reason
                return SessionStop("failed", reply, f"the reply stopped short ({cause}) and cannot be read")
            if reply.content:
                self._messages.append(reply.message())
            self._tally.record_transcript(self.name, self.transcript_digest())
            if not reply.tool_uses:
                return SessionStop("answered", reply)
            if self._closing:
                return SessionStop(
                    "over_budget",
                    reply,
                    f"the round's {self._calls} tool calls were used up, the model was told to submit, and it called "
                    "a tool again",
                )
            await self._answer(reply.tool_uses)

    async def _turn(self, ceiling: float | None) -> TurnReply:
        """Send one turn, tally it and charge it, a reply refused after it was paid for included."""
        started, turn = time.monotonic(), self._sent
        try:
            reply = await self._client.turn(
                system=self._system, messages=self._messages, tools=self.tools, max_tokens=self.limits.max_tokens
            )
        except TurnReplyRefused as refused:
            self._sent += 1
            self._count(refused.reply, turn, time.monotonic() - started, ceiling, refused=str(refused))
            raise
        self._sent += 1
        self._count(reply, turn, time.monotonic() - started, ceiling)
        self._last_input = reply.input_tokens
        return reply

    def _count(
        self, reply: TurnReply, turn: int, wall_s: float, ceiling: float | None, *, refused: str | None = None
    ) -> None:
        self._tally.record_turn(
            TurnCost.of(
                reply, session=self.name, role=self.role, round=self._round, turn=turn, wall_s=wall_s, refused=refused
            )
        )
        self._budget.charge(reply.cost_usd, ceiling=ceiling)

    async def _answer(self, uses: Sequence[TurnToolUse]) -> None:
        """Run the reply's tool calls within the round's budget and answer each, with the round's notes after."""
        results: list[TurnBlock] = []
        ran = 0
        for use in uses:
            if self._used >= self._calls:
                results.append(TurnToolResult(use.id, self._not_run(), is_error=True))
                continue
            self._used += 1
            ran += 1
            if use.name != self._mount.name:
                refusal = f"refused: there is no tool {use.name!r}; the one tool is {self._mount.name!r}"
                results.append(TurnToolResult(use.id, refusal, is_error=True))
                continue
            outcome = await self._mount.call(use.input)
            text = self._capped(outcome.text)
            self._answered[use.id] = canonical_digest(text)
            results.append(TurnToolResult(use.id, text, is_error=outcome.is_error))
        self._tally.record_calls(self.role, self._round, ran)
        left = self._calls - self._used
        if left <= 0:
            self._closing = True
            results.append(TurnText(self._submit_now()))
        elif not self._warned and left <= self.limits.warn_before:
            self._warned = True
            results.append(
                TurnText(
                    f"[{self._used} of this round's {self._calls} tool calls used; {left} left. Finish reading and "
                    "write your answer.]"
                )
            )
        self._messages.append(TurnMessage("user", results))

    def _not_run(self) -> str:
        return f"not run: this round's {self._calls} tool calls are used up; answer now, without another call"

    def _submit_now(self) -> str:
        return (
            f"[This round's {self._calls} tool calls are used up. Answer now, in a message that calls no tool; "
            "another tool call ends the session with no answer.]"
        )

    def _capped(self, text: str) -> str:
        cap = self.limits.result_chars
        if len(text) <= cap:
            return text
        return (
            f"{text[:cap]}\n[cut at {cap:,} of {len(text):,} characters: narrow the call (a filter, a smaller "
            "limit, the next page) to read the rest]"
        )

    def _append_user(self, blocks: list[TurnBlock]) -> None:
        """Append a user message, answering first any tool call the last reply left open, and joining a user
        message the conversation already ends on (a reply that was not kept leaves one)."""
        if self._messages and self._messages[-1].role == "assistant":
            open_calls = [block.id for block in self._messages[-1].content if isinstance(block, TurnToolUse)]
            blocks = [*(TurnToolResult(call, self._not_run(), is_error=True) for call in open_calls), *blocks]
        if self._messages and self._messages[-1].role == "user":
            self._messages[-1] = TurnMessage("user", [*self._messages[-1].content, *blocks])
            return
        self._messages.append(TurnMessage("user", blocks))

    def _elide_if_over(self) -> None:
        """Elide older tool-result bodies once the input passes its limit (the module docstring's rule)."""
        size = self._last_input if self._last_input is not None else self._estimated_input()
        if size <= self.limits.elide_over_tokens:
            return
        thinking = [
            index
            for index, message in enumerate(self._messages)
            if message.role == "assistant" and any(isinstance(block, TurnThinking) for block in message.content)
        ]
        assistants = [index for index, message in enumerate(self._messages) if message.role == "assistant"]
        if not assistants:
            return
        bound = thinking[-1] if thinking else assistants[-1]
        results = [
            block.tool_use_id
            for message in self._messages
            for block in message.content
            if isinstance(block, TurnToolResult)
        ]
        kept = set(results[-self.limits.keep_results :])
        eligible = {
            block.tool_use_id
            for message in self._messages[:bound]
            for block in message.content
            if isinstance(block, TurnToolResult)
            and block.tool_use_id in self._answered
            and block.tool_use_id not in self._elided
            and block.tool_use_id not in kept
        }
        if len(eligible) < self.limits.keep_results:
            return
        self._messages = [
            TurnMessage(
                message.role, [_elided(block) if _answers(block, eligible) else block for block in message.content]
            )
            if message.role == "user"
            else message
            for message in self._messages
        ]
        self._elided |= eligible

    def _estimated_input(self) -> int:
        """The input's size where no reply has reported one: a token per three characters of what is sent."""
        sent = json.dumps([message.to_json() for message in self._messages] + [tool.to_json() for tool in self.tools])
        return (len(self._system) + len(sent)) // 3

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The session as a stored value, which :meth:`from_json` resumes (the module docstring says what it holds).

        :return: ``{"format", "name", "role", "skill_digest", "tools_digest", "limits", "messages", "results",
            "state"}``.
        :rtype: dict[str, Any]
        """
        dropped = {call for call in self._answered if call not in self._elided} if self._mount.rederivable else set()
        messages = [
            {
                "role": message.role,
                "content": [
                    {**block.to_json(), "content": ""} if _answers(block, dropped) else block.to_json()
                    for block in message.content
                ],
            }
            for message in self._messages
        ]
        return {
            "format": _FORMAT,
            "name": self.name,
            "role": self.role,
            "skill_digest": self.skill_digest,
            "tools_digest": self.tools_digest(),
            "limits": self.limits.to_json(),
            "messages": messages,
            "results": [
                {
                    "tool_use_id": call,
                    "sha256": digest,
                    "context": "elided" if call in self._elided else "kept",
                    "stored": "rederive" if call in dropped else "whole",
                }
                for call, digest in self._answered.items()
            ],
            "state": {
                "turns": self._sent,
                "last_input_tokens": self._last_input,
                "round": self._round,
                "calls": self._calls,
                "calls_used": self._used,
                "warned": self._warned,
                "closing": self._closing,
            },
        }

    @classmethod
    async def from_json(
        cls,
        record: Mapping[str, Any],  # Any: a JSON object
        *,
        client: ToolTurnClient,
        mount: ToolMount,
        system: SkillText,
        tally: Tally,
        budget: ReportBudget,
    ) -> ToolSession:
        """Resume a stored session: the same system text and tool, its messages, and where its round stood.

        :param record: What :meth:`to_json` wrote.
        :ptype record: Mapping[str, Any]
        :param client: The client its next turns go through.
        :ptype client: ToolTurnClient
        :param mount: The tool, which re-derives each result body the record dropped.
        :ptype mount: ToolMount
        :param system: The role's system text, which must be the one the session started with.
        :ptype system: SkillText
        :param tally: The tally its next turns are counted in.
        :ptype tally: Tally
        :param budget: The budget that admits its next turns.
        :ptype budget: ReportBudget
        :return: The session, ready for its next user message.
        :rtype: ToolSession
        :raises ValueError: A record of another format or out of shape; a system text or tool other than the
            stored one; a dropped result body the tool no longer answers the same.
        """
        if record.get("format") != _FORMAT:
            raise ValueError(f"a stored session of format {record.get('format')!r}; this build reads format {_FORMAT}")
        try:
            limits = SessionLimits(**record["limits"])
            session = cls(
                name=record["name"],
                role=record["role"],
                client=client,
                mount=mount,
                system=system,
                tally=tally,
                budget=budget,
                limits=limits,
            )
            stored_skill, stored_tools = record["skill_digest"], record["tools_digest"]
            messages = [
                TurnMessage(message["role"], [block_from_json(block) for block in message["content"]])
                for message in record["messages"]
            ]
            results = list(record["results"])
            state = record["state"]
        except (KeyError, TypeError) as missing:
            raise ValueError(f"a stored session out of shape: {missing!r}") from missing
        if stored_skill != session.skill_digest:
            raise ValueError(
                f"session {session.name!r} was started with system text {stored_skill[:12]}, and it is resumed with "
                f"{session.skill_digest[:12]}: resume it with the skill text it started from"
            )
        if stored_tools != session.tools_digest():
            raise ValueError(
                f"session {session.name!r} was started with another definition of tool {mount.name!r}: resume it "
                "with the tool it started with"
            )
        uses = {block.id: block for message in messages for block in message.content if isinstance(block, TurnToolUse)}
        bodies: dict[str, str] = {}
        for result in results:
            call, digest = result["tool_use_id"], result["sha256"]
            session._answered[call] = digest
            if result["context"] == "elided":
                session._elided.add(call)
            if result["stored"] != "rederive":
                continue
            if call not in uses:
                raise ValueError(f"a stored session drops the result of call {call!r}, which it holds no call for")
            outcome = await mount.call(uses[call].input)
            text = session._capped(outcome.text)
            if canonical_digest(text) != digest:
                raise ValueError(
                    f"call {call!r} ({uses[call].input!r}) no longer answers as it did when the session was stored, "
                    "so the session cannot be resumed as it was: the snapshot or the tool has changed"
                )
            bodies[call] = text
        session._messages = [
            TurnMessage(
                message.role,
                [_with_body(block, bodies) for block in message.content],
            )
            for message in messages
        ]
        session._sent = state["turns"]
        session._last_input = state["last_input_tokens"]
        session._round = state["round"]
        session._calls = state["calls"]
        session._used = state["calls_used"]
        session._warned = state["warned"]
        session._closing = state["closing"]
        return session


def _answers(block: TurnBlock, calls: Mapping[str, object] | set[str]) -> bool:
    """Whether ``block`` is the result of one of ``calls``."""
    return isinstance(block, TurnToolResult) and block.tool_use_id in calls


def _elided(block: TurnBlock) -> TurnBlock:
    """A tool result with its body elided; any other block unchanged."""
    if not isinstance(block, TurnToolResult):
        return block
    return TurnToolResult(block.tool_use_id, _ELIDED, is_error=block.is_error)


def _with_body(block: TurnBlock, bodies: Mapping[str, str]) -> TurnBlock:
    """A tool result with its body re-derived, where ``bodies`` holds one for its call; any other block unchanged."""
    if not isinstance(block, TurnToolResult) or block.tool_use_id not in bodies:
        return block
    return TurnToolResult(block.tool_use_id, bodies[block.tool_use_id], is_error=block.is_error)
