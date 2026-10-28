"""What writing a report cost: every paid turn, the tool calls and bounces each round made, and the report's budget.

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

:class:`ReportBudget` is the host's cap for one report, shared by all of its sessions, in one of two forms the host
declares. **A dollar budget** (``ReportBudget(usd=)``) admits every turn on the client's price ceiling for it, and
charges a sent turn what it cost, or, when nothing priced it, its ceiling: the most it can have cost, so an unpriced
turn never lets the budget read more room than there is. A ceiling built on rates the host's pricer does not
guarantee still admits the turn, and the turn is tallied as admitted on an estimate (``admitted_on_estimate``), which
the made line discloses. **A token and turn budget** (``ReportBudget(tokens=, turns=)``) admits a turn by its tokens
and the turns sent, never by its price, so a model nothing prices (a local model, a subscription) still writes a
report, tallied unpriced and never as ``$0``. Either way, a refusal sends nothing.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from threetears.evals.kernel.numbers import group_thousands, spell_dollars
from threetears.evals.schema.turns import TurnCeiling, TurnReply

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
    :ivar admitted_on_estimate: Whether a dollar budget admitted the turn on a ceiling built on rates the host's
        pricer does not guarantee: its cap was checked against an estimate.
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
    admitted_on_estimate: bool = False

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
        admitted_on_estimate: bool = False,
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
        :param admitted_on_estimate: Whether a dollar budget admitted it on an estimated ceiling.
        :ptype admitted_on_estimate: bool
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
            admitted_on_estimate=admitted_on_estimate,
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
            "admitted_on_estimate": self.admitted_on_estimate,
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
        if self.unpriced_turns():
            return None
        # One sum for both: the total is the priced share when every turn is priced, never a second summation that
        # could differ from it in the last place.
        return self.priced_usd()

    def priced_usd(self) -> float:
        """What the turns that reported a cost cost together: the priced share, a floor whenever a sent turn is
        unpriced (:meth:`unpriced_turns`).

        :return: The sum.
        :rtype: float
        """
        return math.fsum(turn.cost_usd for turn in self._turns if turn.cost_usd is not None)

    def unreturned_turns(self) -> int:
        """The turns sent whose reply never came: the client raised, or the writing was cancelled or ran out of time
        while they were in flight. Each is unpriced (:meth:`unpriced_turns` counts it too).

        :return: The count.
        :rtype: int
        """
        return self.sent - len(self._turns)

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
        rolled["unpriced_reasons"] = sorted({turn.pricing_failure for turn in turns if turn.pricing_failure})
        rolled["admitted_on_estimate"] = sum(1 for turn in turns if turn.admitted_on_estimate)
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
    """The host's cap for one report, shared by every session that writes it: dollars, or tokens and turns.

    Exactly one form, declared by the host. ``ReportBudget(usd=)`` admits a turn on the client's price ceiling for it,
    and refuses one that could take the spend past the cap, or one the client cannot bound; ``report_write`` refuses a
    model whose turns cannot be bounded before it starts, so that refusal is a backstop. ``ReportBudget(tokens=,
    turns=)``, either cap or both, admits a turn by a bound on its tokens (its prompt's, as its price ceiling bounds it,
    and its output cap) and by the turns already sent, and never asks for a price. A refused turn is never sent.
    :meth:`charge` counts a sent turn: at what it cost or, unpriced, its ceiling; at the tokens it reported or, where
    it reported none, its bound; and as one turn.

    :ivar usd: The dollar cap, or ``None`` for a token and turn budget.
    :ivar tokens: The token cap (prompt and output tokens, cache reads and writes included), or ``None``.
    :ivar turns: The turn cap, or ``None``.
    """

    def __init__(self, *, usd: float | None = None, tokens: int | None = None, turns: int | None = None) -> None:
        """Set the cap, in one form.

        :param usd: The most the report may spend, in dollars: finite and above 0.
        :ptype usd: float | None
        :param tokens: The most tokens its turns may use, in all: a whole number above 0.
        :ptype tokens: int | None
        :param turns: The most turns it may send: a whole number above 0.
        :ptype turns: int | None
        :raises ValueError: Both forms, or neither; a dollar cap that is not a finite amount above 0; a token or turn
            cap that is not a whole number above 0.
        """
        if usd is not None and (tokens is not None or turns is not None):
            raise ValueError(
                "a report's budget is dollars (usd=) or tokens and turns (tokens=, turns=), declared by the host, "
                "never both"
            )
        if usd is None and tokens is None and turns is None:
            raise ValueError(
                "a report's budget declares a cap: dollars (usd=), or tokens and turns (tokens=, turns=) for a model "
                "nothing prices"
            )
        if usd is not None and (
            isinstance(usd, bool) or not isinstance(usd, int | float) or not math.isfinite(usd) or usd <= 0
        ):
            raise ValueError(f"a report's budget is a finite number of dollars above 0, not {usd!r}")
        for name, cap in (("tokens", tokens), ("turns", turns)):
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
                raise ValueError(f"a report's {name} budget is a whole number above 0, not {cap!r}")
        self.usd = float(usd) if usd is not None else None
        self.tokens = tokens
        self.turns = turns
        self._spent = 0.0
        self._spent_tokens = 0
        self._sent = 0

    @property
    def spelled(self) -> str:
        """The cap in words: ``$20.00``, or ``2,000,000-token, 60-turn``."""
        if self.usd is not None:
            return spell_dollars(self.usd)
        caps = [f"{group_thousands(str(self.tokens))}-token"] if self.tokens is not None else []
        caps += [f"{self.turns}-turn"] if self.turns is not None else []
        return ", ".join(caps)

    @property
    def spent(self) -> float:
        """What the turns charged so far count for in dollars, under either form: each at its cost, or its ceiling when
        it was unpriced, and nothing for a turn with neither. A token and turn budget admits by tokens and turns alone
        and never reads it."""
        return self._spent

    def admit(self, ceiling: TurnCeiling | None, *, tokens: int) -> str | None:
        """Whether a turn may be sent: ``None`` admits it; otherwise why it is refused, as a sentence.

        :param ceiling: The client's price ceiling for the turn, or ``None`` when it cannot bound it; read by a dollar
            budget alone.
        :ptype ceiling: TurnCeiling | None
        :param tokens: A bound on the tokens the turn can use: its prompt's bound and its output cap; read by a token
            budget alone.
        :ptype tokens: int
        :return: ``None``, or the refusal.
        :rtype: str | None
        """
        if self.usd is None:
            if self.turns is not None and self._sent >= self.turns:
                return f"the report's {self.turns}-turn budget is spent"
            if self.tokens is not None and self._spent_tokens + tokens > self.tokens:
                return (
                    f"the next turn can use up to {group_thousands(str(tokens))} tokens, and "
                    f"{group_thousands(str(self._spent_tokens))} of the report's "
                    f"{group_thousands(str(self.tokens))}-token budget are used"
                )
            return None
        if ceiling is None:
            return (
                f"the client cannot bound the next turn's cost, so the report's {spell_dollars(self.usd)} budget cannot admit it "
                "(load the price catalogue, hand the model a pricer, use a model it prices, or declare a token and "
                "turn budget for a model nothing prices)"
            )
        if self._spent + ceiling.usd > self.usd:
            return (
                f"the next turn can cost up to {spell_dollars(ceiling.usd)}, and {spell_dollars(self._spent)} of the report's "
                f"{spell_dollars(self.usd)} budget is spent"
            )
        return None

    def admitted_on_estimate(self, ceiling: TurnCeiling | None) -> bool:
        """Whether admitting a turn on ``ceiling`` checks the cap against an estimate: a dollar budget, and a ceiling
        built on rates the host's pricer does not guarantee.

        :param ceiling: The ceiling the turn is admitted on.
        :ptype ceiling: TurnCeiling | None
        :return: Whether it is an estimate the cap is checked against.
        :rtype: bool
        """
        return self.usd is not None and ceiling is not None and ceiling.estimate

    def charge(self, reply: TurnReply, *, ceiling: TurnCeiling | None, tokens: int) -> None:
        """Count a sent turn: its dollars, its tokens, and the turn itself.

        :param reply: The reply it was answered with, a refused one included: it was paid for.
        :ptype reply: TurnReply
        :param ceiling: The price ceiling it was admitted on, counted for its dollars when nothing priced it.
        :ptype ceiling: TurnCeiling | None
        :param tokens: The token bound it was admitted on, counted when the reply reported no token count.
        :ptype tokens: int
        """
        counted = reply.cost_usd if reply.cost_usd is not None else ceiling.usd if ceiling is not None else None
        if counted is not None:
            self._spent += counted
        used = (
            reply.input_tokens + reply.output_tokens
            if reply.input_tokens is not None and reply.output_tokens is not None
            else tokens
        )
        self._spent_tokens += used
        self._sent += 1
