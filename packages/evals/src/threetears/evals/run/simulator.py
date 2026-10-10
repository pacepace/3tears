"""Multi-actor user simulator for the eval substrate.

Drives one (or several) simulated actors against a conversing candidate
inside an eval scenario. A template's :class:`~threetears.evals.contracts.models.ConversationSpec`
lists them; each :class:`~threetears.evals.contracts.models.ActorPolicy` declares a policy and an
intent, and the simulator turns those into one LLM call per turn-it-speaks, conditioned on the
variation params and the transcript so far.

Architecture
------------

A conversation is a sequence of **speaker rounds**. In each round one or more simulated actors
speak, then the candidate answers the round once. A kind normally hands the loop to
:func:`~threetears.evals.run.conversation.drive_conversation`, which is this, written once — less the
run's cost cap, asked before every paid call, and the answer the last actor's departure still gets::

   opener = driver.initial_utterance()          # the first actor's template, if any
   while driver.should_continue():
       while (actor := await driver.next_speaker(llm)) is not None:
           turn = await driver.next_user_turn(llm, actor)
           ...deliver turn unless turn.done...
       response = await candidate.answer(the round's turns)
       driver.record_candidate_turn(response)

The simulator is responsible for *which actor speaks next* and *what they say*. The kind is
responsible for the candidate side and goal-state evaluation. This split keeps the simulator agnostic
of the LLM client implementation: any object with an async
``generate(system, user, response_format) -> object`` method works.

Stopping
--------

A conversation stops only on a structural signal, recorded as a
:class:`~threetears.evals.contracts.models.ConversationStopCause`: the turn budget,
every simulated actor's structured ``done``, or a rig fault. One actor saying ``done`` leaves the
conversation (the scheduler stops offering it); the conversation ends ``user_done`` when the last
one leaves. Goal checks grade the end state and never end a conversation: one that holds early
("never called X") would stop a pressure conversation at its first refusal. Nothing here reads what
the candidate said — whether a refusal was a good one is a judged property of the transcript, not a
reason to end it.

Session breaks
--------------

``ConversationSpec.sessions`` spreads the candidate turns over that many sittings. After the
candidate turn that closes a session the driver marks a break itself, and the next utterance it
DELIVERS carries ``session_break=True`` and the new ``session_index``. The kind clears the history
its candidate sees, or starts whatever its product calls a new session, while its world persists:
what persists is the kind's to decide. Nothing outside the driver marks a break, so the count the
template declares is the count a run holds.

Turn scheduler
--------------

``round_robin`` takes the actors in list order, wrapping, until the round holds
``max_speakers_per_round`` utterances. ``llm_decided`` asks a simulator-role model, through a strict
structured reply, which actor speaks next or whether the round is done (offered only once the round
holds an utterance). A reply that does not validate gets one repair call naming what was wrong; a
second bad reply is a :class:`SimulatorReplyInvalid`. When only one answer is legal (one actor left
and nothing said yet) no call is made, because there is nothing to decide.

Spend
-----

Every simulator-role call this driver makes, utterance or scheduling, is one :class:`SimulatorCall`
in :attr:`TurnDriver.calls`, attributed to the actor it produced or chose, and recorded before its
reply is validated: a malformed reply was billed like any other. That list is the one record of the
simulator's spend; :meth:`TurnDriver.fold_usage` folds it into the cell's ``simulator`` ledger, one
stored row per actor and purpose (``RoleUsage.actor_id`` and ``RoleUsage.purpose``), so what each
actor's lines and each pick of the scheduler cost survives the cell. :attr:`TurnDriver.cost_usd` is
its total, which :func:`~threetears.evals.run.conversation.drive_conversation` asks the run's cost cap
about before every further call.

**The worst case one conversation can spend.** With ``T = max_turns``, ``S = max_speakers_per_round`` and ``A``
actors: at most ``T·S`` lines are delivered and at most ``A`` replies are departures, so at most ``T·S + A`` utterance
calls are made; every ``llm_decided`` decision ends in one of those or in one ``round_done`` per round, and makes at
most :data:`SCHEDULER_CALL_ATTEMPTS` calls, so at most ``2·(T·S + A + T)`` scheduling calls. That is ``3·(T·S + A) +
2·T`` simulator calls under ``llm_decided`` (``T·S + A`` under ``round_robin``) — at the schema's maxima (``T = 100``,
``S = 20``) over 6,000 calls, each capped at :data:`SIMULATOR_REQUEST_SETTINGS`'s ``max_tokens``. Under an enforcing
cost cap the run's ceiling binds first: the loop stops ``budget_stopped`` before any call once the run's recorded
spend plus this conversation's simulator spend exceeds the cap. It asks once per decision, and one ``llm_decided``
decision can be :data:`SCHEDULER_CALL_ATTEMPTS` calls, so the overshoot is at most two simulator calls (one under
``round_robin``) plus the cell's candidate spend. With no cap enforced, the structure above is the only bound.

Why this is engine API
----------------------

:class:`TurnDriver`, :class:`SimulatorTurn`, :class:`CandidateTurn` and
:class:`SimulatorReplyInvalid` are driven by a host's conversing kind, and they live in ``run``
because what they drive is the engine's own template: the actors, their policies and the turn
budget are the template's ``conversation`` block, the call goes through the ``SimulatorLLM``
port, and nothing here knows what the candidate is. Any kind whose candidate holds a
conversation is simulated against the same block, so it drives the same simulator rather than
re-implementing one beside itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from threetears.evals.contracts.authored import strict_schema
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.models import (
    CANDIDATE_SPEAKER,
    ROUND_DONE,
    WORLD_SPEAKER,
    ActorPolicy,
    ClientRequestSettings,
    ConversationSpec,
    ConversationStopCause,
    ReasoningEffort,
    SimulatorPurpose,
    WorldRound,
)
from threetears.evals.contracts.provider import SimulatorLLM
from threetears.evals.contracts.world_events import WorldEvent
from threetears.evals.contracts.usage_capture import CallUsage, RoleUsageLedger

#: How hard the simulated user reasons: the router's ``reasoning.effort``, at its lowest level that
#: still reasons. A turn is a line of dialogue and a scheduling pick is one id, so the reasoning
#: they need is small; the setting exists because a reasoning model's default spends far more.
#:
#: **An effort, not a token budget — measured, twice.** ``openai/gpt-5-nano`` at its default
#: (``medium``) effort spent all of a flat 4096-token cap reasoning on a rules lawyer's objection
#: and returned an empty or cut reply, three cells in six of one template (2026-10-06). A token
#: budget (``reasoning.max_tokens`` 1024 under a 5120 cap) did not fix it: gpt-5-nano is
#: effort-only, so the router mapped the budget to an effort by its share of the cap (a fifth is
#: ``low``), which bounds nothing in tokens, and the same utterance still reasoned to the whole
#: cap and came back empty in two cells of 24. So the level is named directly. ``minimal`` is the
#: lowest the model lists (the router's ``/models`` gives gpt-5-nano ``supported_efforts``
#: high/medium/low/minimal, reasoning ``mandatory``, so ``none`` would be refused).
#:
#: **What it does not guarantee.** An effort level bounds nothing in tokens on any model; a model
#: that takes reasoning as a token budget maps the level back to one by its own rule, and a model
#: that does not reason ignores it. The cap below is the only hard stop.
SIMULATOR_REASONING_EFFORT: ReasoningEffort = "minimal"

#: Room the output cap leaves for the simulated user's private reasoning at
#: :data:`SIMULATOR_REASONING_EFFORT`. Never sent — the request carries the effort, not a budget —
#: so it bounds nothing; it is what the cap is sized from, so a turn that reasons a little more than
#: ``minimal`` usually does still has room to answer.
SIMULATOR_REASONING_ALLOWANCE_TOKENS = 1024

#: Output room kept for the simulated user's visible reply: the strict-schema JSON object holding one
#: utterance or one scheduling pick. That is at most a few hundred tokens, so this is headroom; unused
#: output tokens are not billed.
SIMULATOR_ANSWER_BUDGET_TOKENS = 4096

#: The simulator's output cap: the reasoning allowance plus the answer budget, derived rather than
#: chosen. It is the only bound in tokens a simulator call has.
SIMULATOR_MAX_TOKENS = SIMULATOR_REASONING_ALLOWANCE_TOKENS + SIMULATOR_ANSWER_BUDGET_TOKENS

#: The simulator role's request settings as ONE value: what the host's client builder applies to
#: the simulated user's client, and what a run launched with a simulated user records as
#: ``simulator_request_settings``. Recorded on the run for the reason
#: :data:`threetears.evals.run.judge.JUDGE_REQUEST_SETTINGS` is — moving it changes the conversation
#: every later candidate is handed, with ``simulator_model`` unchanged. The setting has moved twice:
#: runs launched before the first move record a flat 4096-token cap with no reasoning parameter,
#: runs between the two record a 1024-token reasoning budget under a 5120 cap, and runs after record
#: this effort level. The three stamps differ, so the ``simulator_request_settings`` apparatus
#: dimension tells a campaign pooling runs from either side of either move that its simulated user
#: was asked differently.
SIMULATOR_REQUEST_SETTINGS = ClientRequestSettings(
    max_tokens=SIMULATOR_MAX_TOKENS,
    reasoning_effort=SIMULATOR_REASONING_EFFORT,
)

#: How many calls one ``llm_decided`` scheduling decision may make: the first, and one repair that
#: names what was wrong with it. A second bad reply is a rig fault, not a reason to keep paying.
SCHEDULER_CALL_ATTEMPTS = 2


class SimulatedUserReply(EvalBaseModel):
    """What the simulated user returns for one turn, as the strict schema it is sent.

    ``done`` is how a simulated actor leaves the conversation on its own terms: it is a field the
    provider's schema enforcement fills, not a phrase anything searches the utterance for. A
    reply that does not validate against this model is a simulator fault, never a guess.
    """

    utterance: str
    done: bool


class NextSpeakerReply(EvalBaseModel):
    """What the ``llm_decided`` scheduler returns for one pick, as the strict schema it is sent.

    ``next`` is an actor id or :data:`~threetears.evals.contracts.models.ROUND_DONE`. The schema
    sent with each call narrows it to an ``enum`` of exactly the answers legal at that point, and
    the reply is checked against the same set on return, whatever the provider claims to enforce.
    """

    next: str


def _response_format(name: str, model: type[EvalBaseModel]) -> dict[str, Any]:
    """The ``response_format`` directive for ``model``, through the strict projection.

    The model's docstring is dropped from what is sent: it is written for the next developer, and the
    user prompt is what tells the model what the fields mean.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                key: value for key, value in strict_schema(model.model_json_schema()).items() if key != "description"
            },
        },
    }


#: The ``response_format`` directive every simulated-user call sends. Derived from
#: :class:`SimulatedUserReply` through the same strict projection the analysis writer's schema
#: goes through, so the shape sent and the shape validated on return are one declaration.
SIMULATED_USER_RESPONSE_FORMAT: dict[str, Any] = _response_format("simulated_user_reply", SimulatedUserReply)


def next_speaker_response_format(choices: list[str]) -> dict[str, Any]:
    """The ``response_format`` one scheduling call sends: :class:`NextSpeakerReply` with ``next`` narrowed.

    Args:
        choices: The answers legal at this pick, in the order they are offered.

    Returns:
        The strict directive, its ``next`` property an ``enum`` of ``choices``.
    """
    directive = _response_format("next_speaker", NextSpeakerReply)
    directive["json_schema"]["schema"]["properties"]["next"]["enum"] = list(choices)
    return directive


class SimulatorReplyInvalid(ValueError):
    """A simulator-role reply did not match the schema it was sent.

    Raised for a simulated user's reply that breaks :class:`SimulatedUserReply`, and for an
    ``llm_decided`` scheduler whose reply and repair both failed :class:`NextSpeakerReply`. A rig
    fault: the conversation stops ``simulator_error`` and the malformed reply never reaches the
    candidate as if someone had said it. The calls behind it are already in
    :attr:`TurnDriver.calls`, which is where their spend is read.
    """


# =============================================================================
# Transcript / turn shapes
# =============================================================================


@dataclass
class SimulatorTurn:
    """One user-side utterance produced by the simulator.

    ``actor_id`` identifies which :class:`ActorPolicy` produced this utterance. ``content`` is the
    verbatim text the candidate sees. ``session_break`` says this is the first utterance of a new
    session, whose 0-based number is ``session_index``; ``round_index`` is the 0-based speaker round
    it belongs to.
    """

    actor_id: str
    content: str
    session_break: bool = False
    session_index: int = 0
    round_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    #: The actor declared itself done with this reply. Such a turn is not delivered to the
    #: candidate: the actor has left the conversation, and carries no session break.
    done: bool = False


@dataclass
class CandidateTurn:
    """One candidate-side response captured by the kind and fed back into the simulator.

    It carries the reply's words and nothing else, because the words are all the simulator reads:
    an actor answers what the candidate said. What the candidate DID — its tool calls — reaches the
    engine through the cell's call ledger and world session, which the goal checks read; a second copy
    here would be a channel nothing reads.
    """

    content: str


@dataclass(frozen=True)
class SimulatorCall:
    """One simulator-role model call the driver made, for the cell's ``simulator`` usage row.

    ``purpose`` says which call it was: an actor's ``utterance``, or a ``schedule`` pick by the
    ``llm_decided`` scheduler. ``actor_id`` is the actor the call produced or chose — ``None`` for a
    pick that ended the round or whose reply was refused, which spoke for nobody. ``usage`` is read
    off the response whatever the reply held, with ``None`` fields where the client reported nothing.
    """

    purpose: SimulatorPurpose
    actor_id: str | None
    round_index: int
    usage: CallUsage


# =============================================================================
# Driver
# =============================================================================


def _session_starts(conversation: ConversationSpec) -> frozenset[int]:
    """The candidate-turn counts after which a new session begins, spread as evenly as integers allow.

    ``ConversationSpec`` holds ``sessions <= max_turns``, so each boundary is at least one turn past
    the one before it and every session holds a turn.
    """
    return frozenset(
        index * conversation.max_turns // conversation.sessions for index in range(1, conversation.sessions)
    )


@dataclass
class TurnDriver:
    """Multi-actor round scheduler and utterance generator.

    Construct one per conversation, over the template's ``conversation`` block. The kind asks who
    speaks next and what they say, records the candidate's answer to each round, and the driver keeps
    the rounds, the sessions, the departures and the spend.
    """

    conversation: ConversationSpec
    variation: dict[str, Any]
    transcript: list[tuple[str, str]] = field(default_factory=list)
    candidate_turns: int = 0
    user_turns: int = 0
    #: Every simulator-role call made, in order. See the module docstring's *Spend*.
    calls: list[SimulatorCall] = field(default_factory=list)
    _next_actor_idx: int = field(default=0, init=False)
    _pending_session_break: bool = field(default=False, init=False)
    _stop_cause: ConversationStopCause | None = field(default=None, init=False)
    _round_index: int = field(default=0, init=False)
    _round_slots: int = field(default=0, init=False)
    _round_delivered: int = field(default=0, init=False)
    _session_index: int = field(default=0, init=False)
    _departed: set[str] = field(default_factory=set, init=False)
    _session_starts: frozenset[int] = field(default_factory=frozenset, init=False)

    def __post_init__(self) -> None:
        if self.conversation.turn_scheduler not in ("round_robin", "llm_decided"):
            raise ValueError(f"Unknown turn_scheduler: {self.conversation.turn_scheduler}")
        self._session_starts = _session_starts(self.conversation)

    # ---- Continuation -------------------------------------------------------

    def should_continue(self) -> bool:
        """True iff another round should be produced.

        Stops on the turn budget, or on any cause already recorded through :meth:`stop` — every
        actor's ``done`` or a rig fault.
        """
        if self._stop_cause is not None:
            return False
        if self.candidate_turns >= self.conversation.max_turns:
            self._stop_cause = ConversationStopCause.MAX_TURNS
            return False
        return True

    def stop(self, cause: ConversationStopCause) -> None:
        """Halt the conversation under ``cause``. Idempotent: the first cause recorded stands.

        Args:
            cause: Why the conversation stopped.
        """
        if self._stop_cause is None:
            self._stop_cause = cause

    @property
    def stop_cause(self) -> ConversationStopCause | None:
        """Why the conversation stopped, or ``None`` while it is still running."""
        return self._stop_cause

    @property
    def session_index(self) -> int:
        """The 0-based session the conversation is in."""
        return self._session_index

    @property
    def round_index(self) -> int:
        """The 0-based speaker round the conversation is in."""
        return self._round_index

    @property
    def departed(self) -> frozenset[str]:
        """The ids of the actors that have said ``done`` and left the conversation."""
        return frozenset(self._departed)

    def _present(self) -> list[ActorPolicy]:
        return [actor for actor in self.conversation.actors if actor.id not in self._departed]

    # ---- Initial utterance --------------------------------------------------

    def initial_utterance(self) -> SimulatorTurn | None:
        """Return the opening turn from the first actor's template, if it has one.

        The first actor's :attr:`ActorPolicy.initial_utterance_template` is rendered with the
        variation params (placeholders like ``{variation.topic}`` substituted), under either
        scheduler: the author placed that actor first and gave it the opening line. It is the first
        utterance of the first round. Returns ``None`` when the first actor has no template — the kind
        then asks :meth:`next_speaker` for the opener.

        Raises:
            ValueError: The conversation has already taken a turn; an opener opens.
        """
        if self.user_turns or self.candidate_turns:
            raise ValueError("initial_utterance opens a conversation, and this one has already taken turns")
        if not self.conversation.actors:
            return None
        actor = self.conversation.actors[0]
        template_text = actor.initial_utterance_template
        if not template_text:
            return None
        # Advance the rotation past this actor since they just spoke.
        self._next_actor_idx = 1 % len(self.conversation.actors)
        self._round_slots += 1
        self.user_turns += 1
        rendered = _render_variation_placeholders(template_text, self.variation)
        return self._deliver(actor, rendered)

    # ---- Scheduling ---------------------------------------------------------

    async def next_speaker(self, llm: SimulatorLLM) -> ActorPolicy | None:
        """Who speaks next in the current round, or ``None`` when the round is done.

        A round is done when it holds ``max_speakers_per_round`` utterance calls, or, under
        ``llm_decided``, when the scheduler says so. A round whose calls all ended in departures holds
        nothing for the candidate to answer, so it starts over rather than ending: every such call
        removes an actor, so this is bounded by the actor count.

        Args:
            llm: The simulator-role client; called only by ``llm_decided``, and only when more than one
                answer is legal.

        Returns:
            The actor to hand to :meth:`next_user_turn`, or ``None``.

        Raises:
            SimulatorReplyInvalid: The scheduler's reply and its repair both broke the schema.
            ValueError: The conversation has already stopped.
        """
        if self._stop_cause is not None:
            raise ValueError(f"the conversation stopped ({self._stop_cause.value}); nobody speaks next")
        if self._round_slots >= self.conversation.max_speakers_per_round:
            if self._round_delivered:
                return None
            self._round_slots = 0
        present = self._present()
        if self.conversation.turn_scheduler == "round_robin":
            actors = self.conversation.actors
            while True:
                actor = actors[self._next_actor_idx % len(actors)]
                self._next_actor_idx += 1
                if actor.id not in self._departed:
                    return actor
        choices = [actor.id for actor in present]
        if self._round_delivered:
            choices.append(ROUND_DONE)
        if len(choices) == 1:
            return present[0]
        chosen = await self._ask_scheduler(llm, present, choices)
        if chosen == ROUND_DONE:
            return None
        return next(actor for actor in present if actor.id == chosen)

    async def _ask_scheduler(self, llm: SimulatorLLM, present: list[ActorPolicy], choices: list[str]) -> str:
        """One ``llm_decided`` decision: a call, and one repair call if its reply is refused."""
        system = _build_scheduler_system_prompt(present, self.variation)
        user = _build_scheduler_user_prompt(
            self.transcript, choices, delivered=self._round_delivered, cap=self.conversation.max_speakers_per_round
        )
        response_format = next_speaker_response_format(choices)
        refusal = ""
        for _attempt in range(SCHEDULER_CALL_ATTEMPTS):
            prompt = user if not refusal else f"{user}\n\nYour previous reply was refused: {refusal}. Reply again."
            response = await llm.generate(system=system, user=prompt, response_format=response_format)
            usage = _call_usage(response)
            content = getattr(response, "content", None)
            try:
                reply = NextSpeakerReply.model_validate_json(content or "", strict=True)
            except ValidationError:
                refusal = "it was not a JSON object holding exactly the field `next`"
                self.calls.append(SimulatorCall("schedule", None, self._round_index, usage))
                continue
            if reply.next not in choices:
                refusal = f"`next` was {json.dumps(reply.next)}, which is not one of {json.dumps(choices)}"
                self.calls.append(SimulatorCall("schedule", None, self._round_index, usage))
                continue
            actor_id = None if reply.next == ROUND_DONE else reply.next
            self.calls.append(SimulatorCall("schedule", actor_id, self._round_index, usage))
            return reply.next
        raise SimulatorReplyInvalid(
            f"the turn scheduler's reply was refused {SCHEDULER_CALL_ATTEMPTS} times; the last: {refusal}"
        )

    # ---- Per-turn LLM-driven utterance -------------------------------------

    async def next_user_turn(self, llm: SimulatorLLM, actor: ActorPolicy) -> SimulatorTurn:
        """Produce ``actor``'s next utterance via one structured LLM call.

        The call sends :data:`SIMULATED_USER_RESPONSE_FORMAT` and validates the reply against
        :class:`SimulatedUserReply` on return, whatever the provider claims to enforce. The call is in
        :attr:`calls` before the reply is validated. A reply with ``done`` set removes the actor from
        the conversation and is not appended to the transcript: it is the actor leaving, not a line
        the candidate is asked to answer. The last actor leaving stops the driver ``user_done``.

        Args:
            llm: An object satisfying :class:`SimulatorLLM` — needs only
                ``await llm.generate(system=..., user=..., response_format=...)`` returning
                an object with a ``content`` attribute.
            actor: The speaker, as :meth:`next_speaker` named it.

        Returns:
            A :class:`SimulatorTurn`.

        Raises:
            SimulatorReplyInvalid: The reply was not the structured shape it was sent.
            ValueError: ``actor`` is not one of this conversation's actors, or has already left.
        """
        if not any(actor is candidate for candidate in self.conversation.actors):
            raise ValueError(f"actor {actor.id!r} is not one of this conversation's actors")
        if actor.id in self._departed:
            raise ValueError(f"actor {actor.id!r} has left the conversation and cannot speak")
        self._round_slots += 1
        response = await llm.generate(
            system=_build_actor_system_prompt(actor, self.variation),
            user=_build_actor_user_prompt(actor, self.transcript),
            response_format=SIMULATED_USER_RESPONSE_FORMAT,
        )
        # Spend is recorded before the reply is validated: a malformed reply was still billed.
        self.calls.append(SimulatorCall("utterance", actor.id, self._round_index, _call_usage(response)))
        reply = _parse_reply(getattr(response, "content", None))
        self.user_turns += 1
        if reply.done:
            self._departed.add(actor.id)
            if not self._present():
                self.stop(ConversationStopCause.USER_DONE)
            return SimulatorTurn(
                actor_id=actor.id,
                content=reply.utterance,
                session_index=self._session_index,
                round_index=self._round_index,
                done=True,
            )
        return self._deliver(actor, reply.utterance)

    def _deliver(self, actor: ActorPolicy, content: str) -> SimulatorTurn:
        """Record a delivered utterance; it carries the pending session break, if any."""
        self.transcript.append((actor.id, content))
        self._round_delivered += 1
        flag = self._pending_session_break
        self._pending_session_break = False
        return SimulatorTurn(
            actor_id=actor.id,
            content=content,
            session_break=flag,
            session_index=self._session_index,
            round_index=self._round_index,
        )

    # ---- World rounds -------------------------------------------------------

    def world_round(self) -> WorldRound | None:
        """The world round the current round is, or ``None`` when the actors speak in it."""
        return self.conversation.world_round(self.candidate_turns + 1)

    def record_world_event(self, world_round: WorldRound, event: WorldEvent) -> SimulatorTurn:
        """Record that the current round's stimulus fired, and return it as the round's one turn.

        The turn is spoken by :data:`~threetears.evals.contracts.models.WORLD_SPEAKER`, so the kind answering the
        round knows the world moved and nobody spoke. It carries a due session break, as a delivered line would.
        It is not delivered through ``post_user_turn``: the host's fire handle already moved the candidate's
        world. The transcript records it, so an actor speaking in a later round reads that it happened.

        Args:
            world_round: The current round, as :meth:`world_round` named it.
            event: What the world session recorded when it fired.

        Returns:
            The round's turn.

        Raises:
            ValueError: ``world_round`` is not the current round.
        """
        if world_round != self.world_round():
            raise ValueError(f"turn {world_round.turn} is not the current round's world event")
        content = f"[world] {world_round.dimension} fired ({event.condition or event.kind})"
        self.transcript.append((WORLD_SPEAKER, content))
        flag = self._pending_session_break
        self._pending_session_break = False
        return SimulatorTurn(
            actor_id=WORLD_SPEAKER,
            content=content,
            session_break=flag,
            session_index=self._session_index,
            round_index=self._round_index,
            metadata={"world_event": event.model_dump(mode="json")},
        )

    def record_candidate_turn(self, turn: CandidateTurn) -> None:
        """Append the candidate's answer to the round, close the round, and mark a due session break.

        Nothing here reads the response's words: see the module docstring's *Stopping*.
        """
        self.candidate_turns += 1
        self.transcript.append((CANDIDATE_SPEAKER, turn.content))
        self._round_index += 1
        self._round_slots = 0
        self._round_delivered = 0
        if self.candidate_turns in self._session_starts:
            self._mark_session_break()

    # ---- Session breaks -----------------------------------------------------

    def _mark_session_break(self) -> None:
        """Begin the next session: the next delivered utterance carries ``session_break=True``.

        Private, and called only from :meth:`record_candidate_turn` at the boundaries
        ``ConversationSpec.sessions`` places, so the breaks a run holds are the ones its template
        declared.
        """
        self._session_index += 1
        self._pending_session_break = True

    # ---- Spend --------------------------------------------------------------

    @property
    def cost_usd(self) -> float | None:
        """What every call in :attr:`calls` cost, or ``None`` once any one of them went unpriced.

        Unpriced is a state, never zero (:func:`~threetears.evals.contracts.usage_capture.blended_cost`):
        a total over a call nobody priced is unknown, and a cost cap stops on it rather than counting it free.
        """
        if any(call.usage.cost_usd is None for call in self.calls):
            return None
        return sum(call.usage.cost_usd or 0.0 for call in self.calls)

    def fold_usage(self, ledger: RoleUsageLedger) -> None:
        """Fold every call in :attr:`calls` into the cell's ``simulator`` ledger, one row per actor and purpose.

        Each call lands under the actor it spoke for or chose and its purpose (``utterance`` or
        ``schedule``), so the stored rows say what each actor and the scheduler spent; a pick that chose
        nobody (``round_done``, a refused reply) lands under no actor.

        Args:
            ledger: The cell's simulator-role ledger.

        Raises:
            ValueError: ``ledger`` is another role's. Simulator spend is the program's cost and never
                the candidate's, so landing it on another role would misstate what the candidate costs.
        """
        if ledger.role != "simulator":
            raise ValueError(f"simulator calls fold into the simulator ledger, not the {ledger.role!r} one")
        for call in self.calls:
            ledger.add_llm_result(call.usage, actor_id=call.actor_id, purpose=call.purpose)


def _call_usage(response: Any) -> CallUsage:
    """The spend one simulator-role response reports; a field the client did not report stays ``None``."""
    return CallUsage.of(response)


# =============================================================================
# Prompt building
# =============================================================================


_VARIATION_PLACEHOLDER = re.compile(r"\{variation\.([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _render_variation_placeholders(text: str, variation: dict[str, Any]) -> str:
    """Substitute ``{variation.<field>}`` tokens in ``text``.

    Unknown fields are left in place verbatim (a templating error is
    surfaced visibly in the transcript rather than silently dropped).
    """

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variation:
            return match.group(0)
        return str(variation[key])

    return _VARIATION_PLACEHOLDER.sub(replace, text)


def _build_actor_system_prompt(actor: ActorPolicy, variation: dict[str, Any]) -> str:
    """Assemble the system prompt for one actor's LLM call.

    Includes the actor's policy and the variation params so the actor's
    utterance stays coherent across the scenario's parameter space. The
    actor's intent is in the user prompt (where the LLM pays it more
    attention) — not here.

    The prompt frames the actor against a generic "candidate under
    evaluation", so a template of any domain works without rewording.
    """
    var_block = _render_variation_block(variation)
    return (
        f"You are simulating a user interacting with a candidate under evaluation. "
        f"Speak ONLY as your assigned actor — do not narrate, do not address yourself "
        f"in third person.\n\n"
        f"## Actor policy (your tone & manner)\n{actor.policy}\n\n"
        f"## Scenario parameters\n{var_block}"
    )


def _build_actor_user_prompt(actor: ActorPolicy, transcript: list[tuple[str, str]]) -> str:
    """Assemble the user prompt — intent + the transcript so far.

    The transcript is rendered turn-by-turn with role labels so the LLM
    sees who has spoken. The actor's *own* prior turns are labeled
    ``You (<id>)``; the candidate's turns are labeled ``Candidate``;
    other actors' turns are labeled ``Actor (<id>)``.
    """
    rendered = _render_transcript(transcript, current_actor_id=actor.id)
    return (
        f"Your intent in this conversation: {actor.intent}\n\n"
        f"Transcript so far:\n{rendered}\n\n"
        f"Reply with your next utterance, in character, one short paragraph, as `utterance`. "
        f"Set `done` to true instead when you have nothing more to say in this conversation; "
        f"you then leave it, and `utterance` is not delivered."
    )


def _render_transcript(transcript: list[tuple[str, str]], *, current_actor_id: str | None) -> str:
    """Render transcript with stable role labels that name no kind of candidate.

    ``current_actor_id`` is the actor reading it, whose own lines read ``You (<id>)``; ``None`` for a
    reader that is no actor (the scheduler), to whom every actor reads ``Actor (<id>)``.
    """
    if not transcript:
        return "(empty — you speak first)" if current_actor_id is not None else "(empty)"
    lines = []
    for speaker, content in transcript:
        if speaker == CANDIDATE_SPEAKER:
            label = "Candidate"
        elif speaker == current_actor_id:
            label = f"You ({speaker})"
        else:
            label = f"Actor ({speaker})"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def _render_variation_block(variation: dict[str, Any]) -> str:
    if not variation:
        return "(none)"
    lines = [f"- {k}: {v}" for k, v in variation.items()]
    return "\n".join(lines)


def _build_scheduler_system_prompt(present: list[ActorPolicy], variation: dict[str, Any]) -> str:
    """Assemble the ``llm_decided`` scheduler's system prompt: its job, and who is in the conversation.

    Each actor is listed with its policy and intent, so the pick can follow who would plausibly speak
    next. The scheduler never speaks, and nothing here names a kind of candidate.
    """
    roster = "\n".join(f"- {actor.id}: {actor.policy} Wants: {actor.intent}" for actor in present)
    return (
        "You schedule the simulated side of a conversation with a candidate under evaluation. The "
        "simulated actors speak in rounds; the candidate answers each round once it is done. You decide "
        "who speaks next in the current round, or that the round is done. You never speak yourself.\n\n"
        f"## Actors still in the conversation\n{roster}\n\n"
        f"## Scenario parameters\n{_render_variation_block(variation)}"
    )


def _build_scheduler_user_prompt(
    transcript: list[tuple[str, str]], choices: list[str], *, delivered: int, cap: int
) -> str:
    """Assemble one scheduling call's user prompt: the transcript, the round so far, the legal answers."""
    rendered = _render_transcript(transcript, current_actor_id=None)
    if ROUND_DONE in choices:
        done_line = (
            f"Answer {ROUND_DONE!r} when the candidate should answer now; this round holds {delivered} "
            f"utterance(s) and may hold at most {cap}."
        )
    else:
        done_line = "Nobody has spoken in this round yet, so an actor must speak."
    return f"Transcript so far:\n{rendered}\n\n{done_line}\nReply with `next`: exactly one of {json.dumps(choices)}."


# =============================================================================
# Structured reply parsing
# =============================================================================


def _parse_reply(content: str | None) -> SimulatedUserReply:
    """Validate the simulated user's reply against the schema it was sent.

    A protocol parse of a JSON answer, not a reading of prose: the only thing inspected is
    whether the object has the two declared fields.

    Args:
        content: The completion's text.

    Returns:
        The validated reply.

    Raises:
        SimulatorReplyInvalid: The text is not JSON, or does not match the model.
    """
    try:
        # Strict: ``"done": "yes"`` is a broken reply, not a truthy one.
        return SimulatedUserReply.model_validate_json(content or "", strict=True)
    except ValidationError as exc:
        raise SimulatorReplyInvalid(f"simulated user reply did not match its schema: {exc}") from exc


__all__ = [
    "SCHEDULER_CALL_ATTEMPTS",
    "SIMULATED_USER_RESPONSE_FORMAT",
    "SIMULATOR_ANSWER_BUDGET_TOKENS",
    "SIMULATOR_MAX_TOKENS",
    "SIMULATOR_REASONING_ALLOWANCE_TOKENS",
    "SIMULATOR_REASONING_EFFORT",
    "SIMULATOR_REQUEST_SETTINGS",
    "CandidateTurn",
    "NextSpeakerReply",
    "SimulatedUserReply",
    "SimulatorCall",
    "SimulatorReplyInvalid",
    "SimulatorTurn",
    "TurnDriver",
    "next_speaker_response_format",
]
