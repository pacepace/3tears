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
saying so (:meth:`ToolSession.capped`).

**Every turn is tallied**, whatever its reply (:class:`~threetears.evals.ops.report_writer.tally.Tally`), and a reply
the client refuses after it was paid for (:class:`~threetears.evals.schema.turns.TurnReplyRefused`) is tallied before
the refusal goes on to the caller.

**The history is append-only.** No message is changed once it is sent: each request's messages are the previous
request's, then what was added since. On the models that bind a thinking block to the conversation that produced it,
rewriting an earlier turn (eliding an old tool result's body, dropping an exchange) invalidates every later thinking
block, and the provider refuses a request replaying one; no client-side shape avoids it. So the context is kept in
bounds by the provider instead: every turn carries a :class:`~threetears.evals.schema.turns.ToolResultClearing`
(server-side context editing), which clears old tool results once the input passes ``clear_over_tokens``, keeping the
latest ``keep_tool_uses`` calls, and only when it can clear at least ``clear_at_least_tokens``, so the cached prompt is
rewritten in batches, never every turn.

**A session is a stored value, and a resumed one sends what the uninterrupted one would.** :meth:`ToolSession.to_json`
keeps the model it runs on, the system text's digest, the tool's, the limits, every message in order with each
thinking block exactly as its reply carried it, every tool call with its result, and where the round stands. A result
body the mount answered is dropped from storage only when the mount says its results are re-derivable (a read of a
frozen snapshot), and the record names each one dropped with the sha256 of the text the model was sent: the stored
record is shorter, and the history it rebuilds is the one sent. :meth:`ToolSession.from_json` refuses a model, system
text or tool other than the stored ones, re-derives each dropped body by calling the tool again with the call's own
arguments, and refuses one that no longer reads the same; so the resumed session's next request is the uninterrupted
session's, byte for byte.

**Each message is stored as its JSON text, not as a nested object.** A tool call's input and a thinking block are
sent back exactly as they were, and a store is free to hand an object's keys back in another order (a Postgres
``jsonb`` column returns them shortest first): a nested message read back from such a store would rebuild a request
whose bytes differ from the one sent, which misses the prompt cache. A string comes back as it was written, so the
history a resume rebuilds is the one sent, byte for byte, whatever order the store keeps keys in.

A reply cut at its output cap (``max_tokens``) is tallied and not kept (its thinking may stop short of its
signature); it was never sent back, so leaving it out edits nothing, and the session's caller says what it needs as a
new user message. A model refusal or a reply cut for another cause stops the session ``failed``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from threetears.evals.analysis.analyst.skill import SkillText
from threetears.evals.kernel.numbers import group_thousands
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally, TurnCost, WriterRole
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.turns import (
    ToolResultClearing,
    ToolTurnClient,
    TurnBlock,
    TurnMessage,
    TurnReply,
    TurnReplyRefused,
    TurnText,
    TurnToolResult,
    TurnToolSpec,
    TurnToolUse,
    block_from_json,
)

#: The names the analyst's and the editor's sessions call, and ``ToolOutcome``, what the mount the report writer's
#: entry point builds answers.
__all__ = ["SessionLimits", "SessionStop", "ToolMount", "ToolOutcome", "ToolSession"]

#: The storage format :meth:`ToolSession.to_json` writes and :meth:`ToolSession.from_json` reads.
_FORMAT: Final = 3


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
    :ivar clear_over_tokens: The input a turn may reach before the provider clears old tool results.
    :ivar keep_tool_uses: How many of the latest tool calls, with their results, a clearing always keeps.
    :ivar clear_at_least_tokens: The fewest input tokens a clearing must remove, or it is not applied: about twelve
        results at the ``result_chars`` cap, so the cached prompt is rewritten in batches.
    :ivar warn_before: How many calls before the round's budget runs out the model is warned.
    """

    max_tokens: int = 32_000
    result_chars: int = 12_000
    clear_over_tokens: int = 150_000
    keep_tool_uses: int = 12
    clear_at_least_tokens: int = 36_000
    warn_before: int = 10

    def __post_init__(self) -> None:
        """Refuse a limit that is not a whole number above 0 (``warn_before`` may be 0).

        :raises ValueError: The limit, named.
        """
        for name, value in self.to_json().items():
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
            "clear_over_tokens": self.clear_over_tokens,
            "keep_tool_uses": self.keep_tool_uses,
            "clear_at_least_tokens": self.clear_at_least_tokens,
            "warn_before": self.warn_before,
        }

    def clearing(self) -> ToolResultClearing:
        """The server-side clearing every turn carries.

        :return: The clearing, from these limits.
        :rtype: ToolResultClearing
        """
        return ToolResultClearing(
            trigger_input_tokens=self.clear_over_tokens,
            keep_tool_uses=self.keep_tool_uses,
            clear_at_least_input_tokens=self.clear_at_least_tokens,
        )


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
    :ivar model: The model id its turns go to, which every reply must name and a resumed session must keep.
    :ivar limits: What every turn is held to.
    :ivar tools: The one tool definition every turn sends.
    :ivar skill_digest: The digest of the system text it was started with.
    """

    def __init__(
        self,
        *,
        name: str,
        role: WriterRole,
        model: str,
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
        :param model: The model id the client sends to (``Model.model_name``).
        :ptype model: str
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
        :raises ValueError: An empty name or model id.
        """
        if not name:
            raise ValueError("a session needs a name, which its turns are tallied and recorded under")
        if not model:
            raise ValueError("a session names the model it runs on, which its thinking blocks are bound to")
        self.name = name
        self.role: WriterRole = role
        self.model = model
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
        self._sent = 0
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

    def capped(self, text: str) -> str:
        """``text`` held to the session's result limit: cut at ``result_chars``, saying so.

        :param text: A tool's text, as it answered.
        :ptype text: str
        :return: The text the model is sent.
        :rtype: str
        """
        cap = self.limits.result_chars
        if len(text) <= cap:
            return text
        return (
            f"{text[:cap]}\n[cut at {group_thousands(str(cap))} of {group_thousands(str(len(text)))} characters: narrow the call (a filter, a smaller "
            "limit, the next page) to read the rest]"
        )

    async def send(self, content: str | Sequence[TurnBlock]) -> SessionStop:
        """Append a user message and run turns until the model answers without a tool call, or is stopped.

        :param content: The message: a text, or its blocks.
        :ptype content: str | Sequence[TurnBlock]
        :return: Why it stopped.
        :rtype: SessionStop
        :raises ValueError: No round is open, or a reply came from another model than the session's.
        :raises TurnReplyRefused: The client refused a reply it was paid for; the turn is tallied first.
        """
        if self._round == 0:
            raise ValueError("open a round (open_round) before sending: a round's call budget holds every turn")
        self._append_user(list((TurnText(content),) if isinstance(content, str) else content))
        while True:
            # The port declares it, so no wrapper can drop it: a turn is never priced from prices never read.
            await self._client.refresh_prices()
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
        """Send one turn, tally it and charge it, a reply refused after it was paid for included; a reply from
        another model than the session's is refused, after it is counted."""
        started, turn = time.monotonic(), self._sent
        # Counted as sent before it is awaited: a turn whose reply never comes (the client fails, or the writing is
        # cancelled or runs out of time) is still a turn sent, which may have been billed.
        self._tally.record_sent()
        try:
            reply = await self._client.turn(
                system=self._system,
                messages=self._messages,
                tools=self.tools,
                max_tokens=self.limits.max_tokens,
                clearing=self.limits.clearing(),
            )
        except TurnReplyRefused as refused:
            self._sent += 1
            self._count(refused.reply, turn, time.monotonic() - started, ceiling, refused=str(refused))
            raise
        self._sent += 1
        self._count(reply, turn, time.monotonic() - started, ceiling)
        if reply.model != self.model:
            raise ValueError(
                f"session {self.name!r} runs on {self.model!r}, and its client sent the turn to {reply.model!r}: "
                "hand it the client of the model it runs on"
            )
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
            text = self.capped(outcome.text)
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

    def _append_user(self, blocks: list[TurnBlock]) -> None:
        """Append a user message, answering first any tool call the last reply left open. A user message the
        conversation already ends on (after a reply that was not kept, or a turn the budget refused) stays as it
        was, and the new one follows it, which the provider reads as one turn: nothing sent is ever changed.

        A call is left open only by the hard stop, after which the session is next sent to in a new round (whose
        :meth:`open_round` ends the stop), so its answer names the round that ran out and this one's fresh budget."""
        if self._messages and self._messages[-1].role == "assistant":
            open_calls = [block.id for block in self._messages[-1].content if isinstance(block, TurnToolUse)]
            answer = self._not_run() if self._closing else self._not_run_before()
            blocks = [*(TurnToolResult(call, answer, is_error=True) for call in open_calls), *blocks]
        self._messages.append(TurnMessage("user", blocks))

    def _not_run_before(self) -> str:
        return (
            "not run: the previous round's tool calls were used up when this call was made; this round starts with a "
            f"fresh budget of {self._calls} tool calls"
        )

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The session as a stored value, which :meth:`from_json` resumes (the module docstring says what it holds).

        :return: ``{"format", "name", "role", "model", "skill_digest", "tools_digest", "limits", "messages",
            "results", "state"}``; each of ``messages`` is that message's JSON text, so no store's key order reaches
            what a resume sends again.
        :rtype: dict[str, Any]
        """
        dropped = set(self._answered) if self._mount.rederivable else set()
        messages = [
            _message_text(
                {
                    "role": message.role,
                    "content": [
                        {**block.to_json(), "content": ""} if _answers(block, dropped) else block.to_json()
                        for block in message.content
                    ],
                }
            )
            for message in self._messages
        ]
        return {
            "format": _FORMAT,
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "skill_digest": self.skill_digest,
            "tools_digest": self.tools_digest(),
            "limits": self.limits.to_json(),
            "messages": messages,
            "results": [
                {"tool_use_id": call, "sha256": digest, "stored": "rederive" if call in dropped else "whole"}
                for call, digest in self._answered.items()
            ],
            "state": {
                "turns": self._sent,
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
        model: str,
        client: ToolTurnClient,
        mount: ToolMount,
        system: SkillText,
        tally: Tally,
        budget: ReportBudget,
    ) -> ToolSession:
        """Resume a stored session: the same model, system text and tool, its messages, and where its round stood.

        :param record: What :meth:`to_json` wrote.
        :ptype record: Mapping[str, Any]
        :param model: The model id the client sends to, which must be the one the session ran on.
        :ptype model: str
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
        :raises ValueError: A record of another format or out of shape; a model, system text or tool other than the
            stored one; a dropped result body the tool no longer answers the same.
        """
        if not isinstance(record, Mapping) or record.get("format") != _FORMAT:
            found = record.get("format") if isinstance(record, Mapping) else type(record).__name__
            raise ValueError(f"a stored session of format {found!r}; this build reads format {_FORMAT}")
        try:
            stored_model = record["model"]
            session = cls(
                name=record["name"],
                role=record["role"],
                model=model,
                client=client,
                mount=mount,
                system=system,
                tally=tally,
                budget=budget,
                limits=SessionLimits(**record["limits"]),
            )
            stored_skill, stored_tools = record["skill_digest"], record["tools_digest"]
            messages = [
                TurnMessage(message["role"], [block_from_json(block) for block in message["content"]])
                for message in (_message_of(text) for text in record["messages"])
            ]
            results = [(item["tool_use_id"], item["sha256"], item["stored"]) for item in record["results"]]
            state = record["state"]
            counters = (
                _count_of(state, "turns"),
                _count_of(state, "round"),
                _count_of(state, "calls"),
                _count_of(state, "calls_used"),
                _flag_of(state, "warned"),
                _flag_of(state, "closing"),
            )
        except (KeyError, TypeError, AttributeError) as missing:
            raise ValueError(f"a stored session out of shape: {missing!r} is missing or of the wrong type") from missing
        if stored_model != model:
            raise ValueError(
                f"session {session.name!r} ran on {stored_model!r}, and it is resumed on {model!r}: its thinking "
                "blocks are bound to the model that wrote them, so resume it on that model"
            )
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
        for call, digest, stored in results:
            session._answered[call] = digest
            if stored != "rederive":
                continue
            if call not in uses:
                raise ValueError(f"a stored session drops the result of call {call!r}, which it holds no call for")
            outcome = await mount.call(uses[call].input)
            text = session.capped(outcome.text)
            if canonical_digest(text) != digest:
                raise ValueError(
                    f"call {call!r} ({uses[call].input!r}) no longer answers as it did when the session was stored, "
                    "so the session cannot be resumed as it was: the snapshot or the tool has changed"
                )
            bodies[call] = text
        session._messages = [
            TurnMessage(message.role, [_with_body(block, bodies) for block in message.content]) for message in messages
        ]
        (
            session._sent,
            session._round,
            session._calls,
            session._used,
            session._warned,
            session._closing,
        ) = counters
        return session


def _message_text(message: Mapping[str, Any]) -> str:  # Any: a JSON object
    """A message as the text it is stored as: its JSON, keys in the order they were sent (the module docstring)."""
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def _message_of(text: Any) -> dict[str, Any]:  # Any: a stored value, checked here; a JSON object
    """A stored message read back from its text, keys in the order it was written; refused unless it is one.

    :raises TypeError: The stored value is not the text of a JSON object.
    """
    if not isinstance(text, str):
        raise TypeError(f"a stored message is its JSON text, not {type(text).__name__}")
    try:
        message = json.loads(text)
    except json.JSONDecodeError as unread:
        raise TypeError(f"a stored message is not JSON ({unread.msg})") from unread
    if not isinstance(message, dict):
        raise TypeError(f"a stored message is the text of a JSON object, not of {type(message).__name__}")
    return message


def _count_of(state: Mapping[str, Any], member: str) -> int:  # Any: a JSON object
    """A stored count, refused as out of shape unless it is a whole number of at least 0."""
    value = state[member]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"state.{member} is {value!r}, not a count")
    return value


def _flag_of(state: Mapping[str, Any], member: str) -> bool:  # Any: a JSON object
    """A stored flag, refused as out of shape unless it is a boolean."""
    value = state[member]
    if not isinstance(value, bool):
        raise TypeError(f"state.{member} is {value!r}, not true or false")
    return value


def _answers(block: TurnBlock, calls: Mapping[str, object] | set[str]) -> bool:
    """Whether ``block`` is the result of one of ``calls``."""
    return isinstance(block, TurnToolResult) and block.tool_use_id in calls


def _with_body(block: TurnBlock, bodies: Mapping[str, str]) -> TurnBlock:
    """A tool result with its body re-derived, where ``bodies`` holds one for its call; any other block unchanged."""
    if not isinstance(block, TurnToolResult) or block.tool_use_id not in bodies:
        return block
    return TurnToolResult(block.tool_use_id, bodies[block.tool_use_id], is_error=block.is_error)
