"""What writing a report cost: every paid turn, the tool calls and bounces each round made, and the dollar budget.

A report is written in sessions of tool turns (the analyst's, then each editor round's), and every turn is paid for.
:class:`Tally` keeps one :class:`TurnCost` per turn sent, whichever session sent it: the role and round it belongs to,
the model asked and the model that answered, its tokens (input, output, the thinking share, cache reads and cache
writes), its cost and who priced it, and how long it took. Beside the turns it counts each round's tool calls and its
bounces. :meth:`Tally.to_json` is what the stored report carries as its ``generation``: the turns, then the sums by
role, by round and in all.

**Unknown is not zero.** A turn nothing priced has ``cost_usd`` ``None`` and says why (``pricing_failure``); a count
the provider did not report is ``None``. A sum over turns any of which is ``None`` is ``None`` too, beside how many
turns were unpriced, so a total never reads lower than what was spent.

**A paid reply is counted even when it is refused.** A client may refuse a reply it was sent and paid for (a
recording refuses one holding the literal ``{snapshot}``); the turn is still tallied, with why it was refused, before
the refusal reaches the caller.

:class:`ReportBudget` is the host's dollar cap for one report, shared by all of its sessions. Before every turn a
session asks it to admit the turn on the client's price ceiling for it; a refusal sends nothing. A turn that is sent
is charged what it cost, or, when nothing priced it, its ceiling: the most it can have cost, so an unpriced turn
never lets the budget read more room than there is.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from threetears.evals.kernel.numbers import spell_dollars
from threetears.evals.schema.turns import TurnReply

__all__ = ["ReportBudget", "Tally", "TurnCost", "WriterRole"]

#: Who a session writes as: the analyst, who drafts and revises, or the report editor, who reviews.
WriterRole = Literal["analyst", "editor"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnCost:
    """One paid turn: whose it was, what it used and what it cost.

    :ivar session: The session that sent it (``analyst``, ``editor-1``).
    :ivar role: The role the session writes as.
    :ivar round: The round it belongs to: the analyst's draft is round 1 and each revision the next; an editor's
        review is the round it reviews.
    :ivar turn: Its index in its session, from 0.
    :ivar model: The model id the turn was sent to.
    :ivar served_model: The model the reply named, or ``None`` when it named none.
    :ivar input_tokens: Prompt tokens, cache reads and writes included; ``None`` when unreported.
    :ivar output_tokens: Output tokens, thinking included; ``None`` when unreported.
    :ivar reasoning_tokens: The thinking share of ``output_tokens``; ``None`` when unsplit.
    :ivar cache_read_tokens: Prompt tokens read from the cache; ``None`` when unreported.
    :ivar cache_write_tokens: Prompt tokens written to the cache; ``None`` when unreported.
    :ivar cost_usd: What the turn cost, or ``None`` when nothing priced it; never ``0`` for unpriced.
    :ivar price_source: Who priced it, or ``None`` when unpriced.
    :ivar pricing_failure: Why ``cost_usd`` is ``None``; ``None`` when priced.
    :ivar stop_reason: Why the reply stopped, in a turn's words.
    :ivar wall_s: Seconds from sending the turn to its reply.
    :ivar refused: Why the client refused the reply after it was paid for, or ``None`` for a reply the session read.
    """

    session: str
    role: WriterRole
    round: int
    turn: int
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
    stop_reason: str
    wall_s: float
    refused: str | None = None

    @classmethod
    def of(
        cls,
        reply: TurnReply,
        *,
        session: str,
        role: WriterRole,
        round: int,
        turn: int,
        wall_s: float,
        refused: str | None = None,
    ) -> TurnCost:
        """The tally entry for one reply.

        :param reply: The reply the turn was answered with.
        :ptype reply: TurnReply
        :param session: The session that sent it.
        :ptype session: str
        :param role: The role the session writes as.
        :ptype role: WriterRole
        :param round: The round the turn belongs to.
        :ptype round: int
        :param turn: Its index in its session.
        :ptype turn: int
        :param wall_s: Seconds from sending it to its reply.
        :ptype wall_s: float
        :param refused: Why the client refused the reply after it was paid for; ``None`` for one it handed back.
        :ptype refused: str | None
        :return: The entry.
        :rtype: TurnCost
        """
        return cls(
            session=session,
            role=role,
            round=round,
            turn=turn,
            model=reply.model,
            served_model=reply.served_model,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            reasoning_tokens=reply.reasoning_tokens,
            cache_read_tokens=reply.cache_read_tokens,
            cache_write_tokens=reply.cache_write_tokens,
            cost_usd=reply.cost_usd,
            price_source=reply.price_source,
            pricing_failure=reply.pricing_failure,
            stop_reason=reply.stop_reason,
            wall_s=wall_s,
            refused=refused,
        )

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The entry as JSON, every member written, ``None`` as ``null``.

        :return: The entry.
        :rtype: dict[str, Any]
        """
        return {
            "session": self.session,
            "role": self.role,
            "round": self.round,
            "turn": self.turn,
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
            "stop_reason": self.stop_reason,
            "wall_s": self.wall_s,
            "refused": self.refused,
        }


#: The token counts a sum is taken of, by their member names.
_TOKENS = ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens", "cache_write_tokens")


class Tally:
    """The writing loop's cost record: every turn, and the tool calls and bounces of each role's rounds.

    One per report, handed to each session it writes with; :meth:`to_json` is the stored report's ``generation``.
    """

    def __init__(self) -> None:
        """Start an empty tally."""
        self._turns: list[TurnCost] = []
        self._calls: Counter[tuple[WriterRole, int]] = Counter()
        self._bounces: dict[tuple[WriterRole, int], list[str]] = {}
        self._transcripts: dict[str, str] = {}
        self._sent = 0

    @property
    def turns(self) -> tuple[TurnCost, ...]:
        """Every turn recorded, in the order they were sent."""
        return tuple(self._turns)

    @property
    def sent(self) -> int:
        """Every turn sent, whether its reply came back or not: at least ``len(turns)``. A turn whose reply never
        came (the client failed after its retries, or the writing was cancelled or ran out of time while it was in
        flight) is counted here and in no :class:`TurnCost`, since nothing reported what it used or cost."""
        return max(self._sent, len(self._turns))

    def record_sent(self) -> None:
        """Count one turn as sent, before its reply is awaited (:attr:`sent`)."""
        self._sent += 1

    def record_turn(self, cost: TurnCost) -> None:
        """Keep one paid turn.

        :param cost: The turn.
        :ptype cost: TurnCost
        """
        self._turns.append(cost)

    def record_calls(self, role: WriterRole, round: int, calls: int) -> None:
        """Count tool calls a role's round made.

        :param role: The role.
        :ptype role: WriterRole
        :param round: The round.
        :ptype round: int
        :param calls: How many calls to add.
        :ptype calls: int
        """
        self._calls[(role, round)] += calls

    def record_bounce(self, role: WriterRole, round: int, kind: str) -> None:
        """Count one bounce of a role's round: a submission sent back with what was wrong with it.

        :param role: The role.
        :ptype role: WriterRole
        :param round: The round.
        :ptype round: int
        :param kind: What was wrong (``unparsed``, ``shape``, ``validity``, ``invariant``, ``max_tokens``).
        :ptype kind: str
        """
        self._bounces.setdefault((role, round), []).append(kind)

    def record_transcript(self, session: str, digest: str) -> None:
        """Keep the digest of a session's messages as they now stand, replacing the one kept before.

        :param session: The session.
        :ptype session: str
        :param digest: The sha256 of its messages' canonical JSON.
        :ptype digest: str
        """
        self._transcripts[session] = digest

    def spent_usd(self) -> float | None:
        """What every turn cost together, or ``None`` when any turn was unpriced, a turn sent whose reply never came
        included.

        :return: The sum, or ``None``.
        :rtype: float | None
        """
        return None if self.sent > len(self._turns) else _sum_cost(self._turns)

    def priced_usd(self) -> float:
        """What the turns that reported a cost cost together: the priced share, a floor whenever a sent turn is
        unpriced (:meth:`unpriced_turns`).

        :return: The sum.
        :rtype: float
        """
        return math.fsum(turn.cost_usd for turn in self._turns if turn.cost_usd is not None)

    def unpriced_turns(self) -> int:
        """The turns sent whose cost nothing reported: replies with no cost, and turns whose reply never came.

        :return: The count.
        :rtype: int
        """
        return self.sent - len(self._turns) + sum(1 for turn in self._turns if turn.cost_usd is None)

    def to_json(self) -> dict[str, Any]:  # Any: a JSON object
        """The tally as the stored report's ``generation``: the turns, then the sums by role, by round and in all.

        :return: ``{"turns", "by_role", "by_round", "total", "transcripts"}``; ``by_round`` keyed ``<role>:<round>``.
        :rtype: dict[str, Any]
        """
        roles = sorted({turn.role for turn in self._turns} | {role for role, _ in (*self._calls, *self._bounces)})
        rounds = sorted({(turn.role, turn.round) for turn in self._turns} | set(self._calls) | set(self._bounces))
        return {
            "turns": [turn.to_json() for turn in self._turns],
            "by_role": {
                role: self._rollup(
                    [turn for turn in self._turns if turn.role == role],
                    [key for key in rounds if key[0] == role],
                )
                for role in roles
            },
            "by_round": {
                f"{role}:{round}": self._rollup(
                    [turn for turn in self._turns if (turn.role, turn.round) == (role, round)], [(role, round)]
                )
                for role, round in rounds
            },
            "total": self._rollup(self._turns, rounds),
            "transcripts": dict(sorted(self._transcripts.items())),
        }

    def _rollup(self, turns: list[TurnCost], rounds: Iterable[tuple[WriterRole, int]]) -> dict[str, Any]:
        keys = list(rounds)
        rolled: dict[str, Any] = {"turns": len(turns)}
        rolled.update({name: _sum_count(getattr(turn, name) for turn in turns) for name in _TOKENS})
        rolled["cost_usd"] = _sum_cost(turns)
        rolled["unpriced_turns"] = sum(1 for turn in turns if turn.cost_usd is None)
        rolled["tool_calls"] = sum(self._calls[key] for key in keys)
        bounces = [kind for key in keys for kind in self._bounces.get(key, [])]
        rolled["bounces"] = len(bounces)
        rolled["bounce_kinds"] = dict(sorted(Counter(bounces).items()))
        rolled["wall_s"] = sum(turn.wall_s for turn in turns)
        return rolled


def _sum_count(counts: Iterable[int | None]) -> int | None:
    """The sum, or ``None`` when any count is unknown."""
    total = 0
    for count in counts:
        if count is None:
            return None
        total += count
    return total


def _sum_cost(turns: Iterable[TurnCost]) -> float | None:
    """What the turns cost together, or ``None`` when any of them is unpriced."""
    total = 0.0
    for turn in turns:
        if turn.cost_usd is None:
            return None
        total += turn.cost_usd
    return total


class ReportBudget:
    """The host's dollar cap for one report, shared by every session that writes it.

    :meth:`admit` is asked before every turn with the client's price ceiling for it, and refuses a turn that could
    take the spend past the cap, or one the client cannot bound; a refused turn is never sent. ``report_write``
    refuses a model whose turns cannot be bounded before it starts, so that refusal is a backstop. :meth:`charge` counts a
    sent turn at what it cost, or at its ceiling when nothing priced it.

    :ivar usd: The cap, in dollars.
    """

    def __init__(self, *, usd: float) -> None:
        """Set the cap.

        :param usd: The most the report may spend, in dollars: finite and above 0.
        :ptype usd: float
        :raises ValueError: A cap that is not a finite amount above 0.
        """
        if isinstance(usd, bool) or not isinstance(usd, int | float) or not math.isfinite(usd) or usd <= 0:
            raise ValueError(f"a report's budget is a finite number of dollars above 0, not {usd!r}")
        self.usd = float(usd)
        self._spent = 0.0

    @property
    def spent(self) -> float:
        """What the turns charged so far count for: each at its cost, or its ceiling when it was unpriced."""
        return self._spent

    def admit(self, ceiling: float | None) -> str | None:
        """Whether a turn may be sent: ``None`` admits it; otherwise why it is refused, as a sentence.

        :param ceiling: The client's ceiling for the turn, or ``None`` when it cannot bound it.
        :ptype ceiling: float | None
        :return: ``None``, or the refusal.
        :rtype: str | None
        """
        if ceiling is None:
            return (
                f"the client cannot bound the next turn's cost, so the report's {spell_dollars(self.usd)} budget cannot admit it "
                "(load the price catalogue, hand the model a pricer, or use a model it prices)"
            )
        if self._spent + ceiling > self.usd:
            return (
                f"the next turn can cost up to {spell_dollars(ceiling)}, and {spell_dollars(self._spent)} of the report's "
                f"{spell_dollars(self.usd)} budget is spent"
            )
        return None

    def charge(self, cost: float | None, *, ceiling: float | None) -> None:
        """Count a sent turn: at its cost, or at its ceiling when nothing priced it.

        :param cost: What the turn cost, or ``None`` when unpriced.
        :ptype cost: float | None
        :param ceiling: The ceiling it was admitted on.
        :ptype ceiling: float | None
        """
        counted = cost if cost is not None else ceiling
        if counted is not None:
            self._spent += counted
