"""Listing the factual claims a text states: the step that turns a cell's segments into claims to grade.

:class:`ClaimExtractor` reads a cell's text as the host cut it, an ordered list of
:class:`~threetears.evals.schema.claims.Segment` values, each a unit with its index or another piece with none. A model
lists every checkable factual proposition the segments state, one proposition per claim, each with the excerpt it
stands in and its tags; code then checks every item and keeps what it checked (:class:`ExtractedClaim`). Each claim
has a scope: ``world``, a fact a source could settle, or ``session``, a statement about this output or run itself (what
was just presented, what comes next, a result within the session). A line that does both is split into one of each,
and a ``restates`` never crosses scopes. The extractor sees the segments' text and nothing else: no key, no document,
no segment's tags. So the list is not primed by the evidence that grades it.

**What a claim is** is stated in domain-neutral terms in the prompt: opinions, impressions, questions and greetings
are not claims; compounds are split, a mixed line into its session and its world claim; what the wording takes for
granted counts as stated. A host whose domain needs more says so in ``guidance``, in its own words and with its own
examples, which stay in the host's code and enter the identity.

**The excerpt is the segment's own slice.** Each listed excerpt is found in its segment by
:func:`~threetears.evals.kernel.claims.verbatim.verbatim_span`, which forgives typography and case and nothing else,
and the excerpt kept is the segment's text at the offsets found, never the model's string.

**A restatement names its original by quoting it, never by counting.** A claim that states a fact an earlier claim
already stated gives, as ``restates``, that earlier claim's whole excerpt, and code resolves it among the claims of
the same scope listed before it (in an earlier segment, or earlier in the list within the same segment); the claim it
names is the first claim of that one's chain. A quote equal to an earlier excerpt, under the verbatim fold, names that
claim, ahead of any claim whose longer excerpt merely holds it. A quote equal to none names a claim only where it
stands, on word boundaries, in exactly one earlier excerpt and holds no other earlier excerpt whole, so a quote of one
claim's excerpt with a neighbour's words added never silently names the neighbour. The quote is matched against
excerpts rather than segments, so a line said word for word in two segments resolves to the one said first, and two
claims cut from one sentence stay apart. A number or a fragment of fewer than :data:`_SHORTEST_QUOTE` characters, a
quote no earlier excerpt matches, and one that could name claims stating different facts name no single claim, and
each is a failed check. A position in a list of a hundred claims is easily miscounted, and a miscounted one would pass
every check and lend the restatement another claim's verdict, since a restatement is never asked about on its own.

**Repair, not refusal.** A reply that will not parse is asked again in full, naming why, twice, then raises. An item
that fails a check (an excerpt not in its segment, a ``restates`` quote that names no single earlier claim) is asked
again alone, with its failure named, in up to two repair rounds; one still failing then raises. A failed quote can
always be repaired: to the whole excerpt of the claim it means, which names that claim wherever its excerpt is no other
claim's, or to null, which leaves the claim to be checked on its own and is the right answer where two claims stating
different facts share one excerpt. Under a criterion's
:class:`~threetears.evals.schema.models.RepairBudget` the rounds are its own and every item asked again counts against
its ``max_items_per_cell``, which the extraction holds as its own step's, apart from the verify steps'. A failing item is never dropped or
downgraded, since dropping the claims a model found hardest to quote would bias every reading counted over them. A
reply cut short raises at once, since the same request would meet the same cap. Every failure is
:class:`~threetears.evals.kernel.claims.failures.ClaimGradingFailed`, which costs the cell only this criterion's
readings.

**It records what it spent, and how the list was made.** Every completion, parsed or not, is folded into the cell's
``judge`` ledger, and recorded as a :class:`~threetears.evals.schema.models.ClaimCall` where the caller keeps the
grade's calls. :attr:`ClaimExtractor.identity` names the model and a digest of everything that shapes the list: the
prompts with the host's ``guidance`` in them, the reply schemas, the verbatim fold's identity (its tables and Unicode
version), and the retry and repair budgets. Claims listed two ways never pool, so a grade whose restatements were
named by position never pools with one whose restatements were named by quote.

Engine-internal: the judge phase builds and calls it, and nothing here is exported.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from pydantic import ConfigDict, Field, ValidationError

from threetears.evals.kernel.authored import strict_response_format
from threetears.evals.kernel.claims.failures import ClaimGradingFailed
from threetears.evals.kernel.claims.verbatim import VERBATIM_FOLD, verbatim_span
from threetears.evals.kernel.provider import describe_incomplete_completion, extract_json
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.capped_completion import call_ceiling
from threetears.evals.schema.claims import Segment
from threetears.evals.schema.completion import (
    CompletionClient,
    CompletionProtocolDefect,
    CompletionResult,
    sent_settings_of,
)
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import MODEL_DEFAULT_TEMPERATURE, ClaimCall, RepairBudget

#: What a claim is about: the world, which truth grading reads, or this output or session itself, which it never does.
type ClaimScope = Literal["world", "session"]

#: How many times the whole cell is asked about: the first call and two more when a finished reply will not parse.
_ATTEMPTS: Final = 3

#: How many rounds a failing item is asked again alone before the extraction raises.
_REPAIR_ROUNDS: Final = 2

#: The role every extraction call is billed under: it reads the candidate's output, as a judge does.
_ROLE: Final = "judge"

#: The fewest characters, folded, a ``restates`` quote holds: fewer is a number or a fragment of a word, which an
#: excerpt can hold by chance (a counted place written as a string, a digit of a year), never a claim's excerpt.
_SHORTEST_QUOTE: Final = 4

#: Something other than whitespace: what a claim's text and an excerpt must hold.
_NOT_BLANK: Final = r"\S"

_RULES: Final = (
    "A claim is one checkable factual proposition the text states: that something is so, about a name, a date, a "
    "place, a number or quantity, a rule, or how things relate. List every one.\n"
    "- One proposition per claim: a sentence that states two facts is two claims.\n"
    "- What the wording takes for granted counts as stated: calling a thing the second of its kind states that "
    "there was a first.\n"
    "- A sweeping general or causal statement (that one event led to another) is a claim.\n"
    "- Not claims: opinions, evaluations and impressions; questions and greetings; and a bare instruction with no "
    "factual content. An instruction or how-to that states a rule or a fact (what a thing does, what a step "
    "requires) states that fact as a claim.\n"
    "- A claim is about the world, or about this output or session itself: what was just presented or played, what "
    "comes next, a result or count within the session (a roll, a score), a pointer to another part of this output "
    '("as noted above"). A sentence that does both is split: introducing what was just presented together with a '
    "fact about it is a session claim that it was presented and a world claim stating the fact.\n"
    "For each claim give:\n"
    "- claim: the proposition, stated on its own in one sentence.\n"
    "- excerpt: the shortest unbroken stretch of the segment that carries the claim, copied exactly as it is written, "
    "never shortened with an ellipsis.\n"
    "- specific: true when the claim states a particular detail a source could confirm or refute (a name, a number, "
    "a date, a place, a rule); false for a general statement.\n"
    "- hedged: true only when the text marks the claim as uncertain; false otherwise. A hedged claim is still a "
    "claim.\n"
    "- attributed: true when the text reports the claim without vouching for it (a rumour, another party's or a "
    'character\'s statement, "some say"); false otherwise. A statement the text makes and backs with a source '
    '("the rules say", "according to its notes") is the text\'s own and not attributed. An attributed claim is '
    "still a claim.\n"
    '- scope: "session" when the claim is about this output or session itself; "world" otherwise.\n'
    "- restates: when the claim states a fact an earlier claim of the same scope in the list already stated, that "
    "earlier claim's whole excerpt, copied exactly as you gave it, no more and no less; otherwise null. Never a number: "
    "the excerpt is what names the claim. When that excerpt is also the excerpt of another claim stating a different "
    "fact, give null: the claim is then checked on its own."
)

_GUIDANCE_HEADING: Final = "\n\n# Notes on what counts as a claim here\n\n"

_SYSTEM_PROMPT: Final = (
    "You list the factual claims a text states. The text is given in segments numbered 0, 1, and so on, in order. "
    "List the claims in the order they appear, each with the number of the segment it is in.\n\n"
    + _RULES
    + "\n\nAnswer with JSON only, in the shape the response format states."
)

_USER_TEMPLATE: Final = "# Segments\n\n{segments}"

_SEGMENT_TEMPLATE: Final = "Segment {index}:\n{text}"

_AGAIN_TEMPLATE: Final = "{user}\n\n# Your last reply\n\nIt could not be used: {failure}. Answer again, in full."

_REPAIR_SYSTEM_PROMPT: Final = (
    "You correct one claim from a list of the factual claims a text states. You are given the segment the claim is "
    "in, the claims listed before it, each with its excerpt and scope, the claim as it was listed, and what is wrong "
    "with it. Give the claim again with that fixed, as the same proposition.\n\n"
    + _RULES
    + "\n\nAnswer with JSON only, in the shape the response "
    "format states."
)

_REPAIR_TEMPLATE: Final = (
    "# Segment {segment}\n\n{text}\n\n# Claims listed before it\n\n{earlier}\n\n"
    "# The claim, at position {position}\n\n{listed}\n\n# What is wrong\n\n{failure}"
)

#: One claim listed before the one repaired: its excerpt first, the words a ``restates`` copies to name it.
_EARLIER_TEMPLATE: Final = "- excerpt: {excerpt}\n  claim: {claim}\n  scope: {scope}"

#: How much of a model's own earlier reply a later call may echo, priced before any call is made: a call's claims, a
#: listed item, an answer or a failure it is shown again. The claims a text states are bounded by the text itself, so
#: the allowance is this many characters of the text the claims come from, plus :data:`ECHO_FLOOR_CHARS`. Bounded by the
#: text rather than by the reply's output cap, because that cap is the judge client's own setting, which the completion
#: port does not expose: a prompt carrying an earlier reply can be priced from its source text, never from the cap. A
#: ceiling, not a guess at the bill; a call whose echo outgrows it is refused by the hold that priced it, never sent
#: past it, and the criterion is lost as ``hold_exceeded``, the price's miss.
ECHO_FACTOR: Final = 3

#: The characters of echo every allowance holds however short its text: a claim's framing, a failure named, an answer.
ECHO_FLOOR_CHARS: Final = 2000

#: The character a priced prompt's allowance is filled with: four bytes in UTF-8, the most one character takes, so a
#: token bound counted in bytes over the filler is never below one over the real text.
_FILLER: Final = "\U0001d465"


#: A number as wide as any claim's place in a list is printed: what a priced prompt prints where a place will stand.
_WIDEST_PLACE: Final = 999_999_999


def echo_chars(*texts: str) -> int:
    """The allowance for what a call may echo of the claims ``texts`` state (:data:`ECHO_FACTOR`).

    :param texts: The text the claims come from.
    :return: The characters.
    """
    return ECHO_FACTOR * sum(len(text) for text in texts) + ECHO_FLOOR_CHARS


def filler(chars: int) -> str:
    """``chars`` characters standing in, when a call is priced, for text a model has not written yet.

    :param chars: How many.
    :return: The filler.
    """
    return _FILLER * chars


def claim_call(role: Literal["extract", "truth", "ground", "repair", "web"], completion: CompletionResult) -> ClaimCall:
    """One completion as a claim grade records its call: the role, and the judge's identity as the reply reported it.

    :param role: What the call was asked.
    :param completion: The completion.
    :return: The call.
    :raises CompletionProtocolDefect: The completion carries no ``temperature``, or a ``sent_settings`` of the wrong
        type.
    """
    try:
        sent = completion.temperature
    except AttributeError as missing:
        raise CompletionProtocolDefect(
            completion, "temperature", "a claim's grade records the temperature each call was sent at"
        ) from missing
    return ClaimCall(
        role=role,
        served_model=getattr(completion, "served_model", None) or None,
        judge_temperature=MODEL_DEFAULT_TEMPERATURE if sent is None else float(sent),
        judge_sent_settings=sent_settings_of(completion),
    )


_CLAIM_PROPERTIES: Final[dict[str, Any]] = {
    "claim": {"type": "string"},
    "excerpt": {"type": "string"},
    "specific": {"type": "boolean"},
    "hedged": {"type": "boolean"},
    "attributed": {"type": "boolean"},
    "scope": {"type": "string", "enum": ["world", "session"]},
    "restates": {"anyOf": [{"type": "string"}, {"type": "null"}]},
}


class ExtractedClaim(EvalBaseModel):
    """One claim the extractor listed, checked: where it stands, the proposition, its excerpt and its tags."""

    model_config = ConfigDict(frozen=True)

    segment: int = Field(ge=0, description="The position of the segment it stands in, among the cell's.")
    unit: int | None = Field(ge=0, description="That segment's unit, or None when the segment is no unit.")
    claim: str = Field(min_length=1, description="The proposition, stated on its own: one fact.")
    excerpt: str = Field(
        min_length=1,
        description="The segment's own text at start..end: the stretch the claim stands in, never the model's copy "
        "of it.",
    )
    start: int = Field(
        ge=0,
        description="Where the excerpt starts in the segment's text: its first occurrence there, when the segment "
        "holds it twice.",
    )
    end: int = Field(ge=1, description="Where it ends (exclusive).")
    specific: bool = Field(
        description="Whether the claim states a particular detail a source could settle (a name, a number, a date, a "
        "place, a rule). The model's tag; no human label calibrates it yet."
    )
    hedged: bool = Field(
        description="Whether the text marked it as uncertain. A hedged claim is still graded for truth."
    )
    attributed: bool = Field(
        description="Whether the text reported it without vouching for it (a rumour, another party's statement). A "
        "statement the text backs with a source is its own, not attributed. An attributed claim is still graded for "
        "truth."
    )
    scope: ClaimScope = Field(
        description="What the claim is about: ``world``, a fact a source could settle, graded for truth; or "
        "``session``, this output or run itself (what was just presented, what comes next, a result within the "
        "session), which only a host's reading of its own record grades."
    )
    restates: int | None = Field(
        ge=0,
        description="The position, among the cell's claims, of the first claim stating the same fact, or None. Always "
        "earlier than this claim's own position, of the same scope, and never itself a restatement.",
    )


class _ListedClaim(EvalBaseModel):
    """One claim as the extractor's reply lists it."""

    segment: int = Field(ge=0)
    claim: str = Field(pattern=_NOT_BLANK)
    excerpt: str = Field(pattern=_NOT_BLANK)
    specific: bool
    hedged: bool
    attributed: bool
    scope: ClaimScope
    #: The excerpt of the earlier claim it restates, as the model quoted it, or None.
    restates: str | None


class _Listing(EvalBaseModel):
    """The extractor's whole reply."""

    claims: list[_ListedClaim]


class _Repaired(EvalBaseModel):
    """One claim as a repair reply gives it again: the same item, its segment fixed by the request."""

    claim: str = Field(pattern=_NOT_BLANK)
    excerpt: str = Field(pattern=_NOT_BLANK)
    specific: bool
    hedged: bool
    attributed: bool
    scope: ClaimScope
    restates: str | None


def _check_units(segments: Sequence[Segment]) -> None:
    """Refuse a cell whose units are not numbered 0..n-1 with no gap, each in at least one segment.

    A unit may span several segments (an entry's description and each of its facts), so a unit index may repeat.

    :param segments: The cell's segments.
    :raises ValueError: They are not.
    """
    units = sorted({segment.unit for segment in segments if segment.unit is not None})
    if units != list(range(len(units))):
        raise ValueError(
            f"a cell's units are numbered 0..n-1 with no gap, each in at least one segment, and its segments carry the "
            f"unit indices {units}: the host's segmentation numbers each unit"
        )


@dataclass(frozen=True)
class ClaimExtractor:
    """A model step that lists the factual claims a cell's segments state, each checked against its segment.

    :ivar client: The completion client the step calls. It must name its model (``model_name``), since an unnamed
        extractor would let two models' lists pool.
    :ivar guidance: The host's own notes on what counts as a claim in its domain, with its own examples, appended to
        the prompts. Empty by default.
    :ivar repairs: The criterion's repair budget, whose rounds the extraction takes and whose ``max_items_per_cell``
        bounds the items it asks again, as its own step's; None (a bare extractor) asks up to two rounds of every
        failing item.
    """

    client: CompletionClient
    guidance: str = ""
    repairs: RepairBudget | None = None

    def __post_init__(self) -> None:
        """Refuse a client that names no model.

        :raises ValueError: It names none.
        """
        model = getattr(self.client, "model_name", None)
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"a claim extractor's client names its model (model_name), and {type(self.client).__name__} names "
                "none: the grade records which model listed the claims, so an unnamed one would pool with any other"
            )

    @property
    def model_name(self) -> str:
        """The model the client calls."""
        # Checked present and a non-blank string at construction.
        return str(getattr(self.client, "model_name", ""))

    @property
    def response_format(self) -> dict[str, Any]:
        """The strict ``json_schema`` directive the listing call sends: ``{"claims": [{"segment", "claim",
        "excerpt", "specific", "hedged", "attributed", "scope", "restates"}]}``."""
        item = {"type": "object", "properties": {"segment": {"type": "integer"}, **_CLAIM_PROPERTIES}}
        schema = {"type": "object", "properties": {"claims": {"type": "array", "items": item}}}
        return strict_response_format("claim_listing", schema)

    @property
    def repair_response_format(self) -> dict[str, Any]:
        """The strict ``json_schema`` directive a repair call sends: one claim, without its segment, which the
        request fixes."""
        return strict_response_format("claim_repair", {"type": "object", "properties": dict(_CLAIM_PROPERTIES)})

    @property
    def identity(self) -> str:
        """``<model>|claims:<sha12>``: the model, and :attr:`definition`'s first twelve characters."""
        return f"{self.model_name}|claims:{self.definition[:12]}"

    @property
    def definition(self) -> str:
        """A digest of everything but the model that shapes the list: the prompts (with the host's guidance), the
        reply schemas, the verbatim fold's identity and the retry and repair budgets. What a criterion's fingerprint
        reads, since which model listed the claims is recorded on each call."""
        shape: dict[str, Any] = {
            "system": self._system_prompt,
            "user": [_USER_TEMPLATE, _SEGMENT_TEMPLATE, _AGAIN_TEMPLATE],
            "repair_system": self._repair_system_prompt,
            "repair_user": [_REPAIR_TEMPLATE, _EARLIER_TEMPLATE],
            "schema": self.response_format,
            "repair_schema": self.repair_response_format,
            "fold": VERBATIM_FOLD.identity,
            "attempts": _ATTEMPTS,
            "repair_rounds": _REPAIR_ROUNDS,
        }
        if self.repairs is not None:
            # Only when a budget is given, so a bare extractor's identity is the one it always was.
            shape["repairs"] = self.repairs.model_dump(mode="json")
        return canonical_digest(shape)

    def price_ceiling(self, segments: Sequence[Segment]) -> float | None:
        """The most extracting ``segments`` can cost on this extractor's client, before any call is made.

        Every attempt of the listing call, its retries priced with a failure named at the echo allowance
        (:func:`echo_chars`), and the most items the repair budget lets it ask again, each priced at its parts: its
        largest segment; the claims listed before it, at one allowance for their propositions and labels, as a list of
        claims is priced, and one more for the excerpts shown beside them; the item, at the allowance of its largest
        segment; and its failure, at one allowance for the excerpts a quote that could name several claims lists, and
        one more for the quote itself and the fixed text of every check.

        :param segments: The cell's segments.
        :return: The ceiling in dollars, or None when the client cannot price a call, or no repair budget bounds the
            items asked again.
        """
        cell = tuple(segments)
        if not cell:
            return 0.0
        if self.repairs is None:
            return None
        rendered = "\n\n".join(
            _SEGMENT_TEMPLATE.format(index=index, text=segment.text) for index, segment in enumerate(cell)
        )
        echo = filler(ECHO_FLOOR_CHARS)
        listing = call_ceiling(
            self.client,
            system=self._system_prompt,
            user=_AGAIN_TEMPLATE.format(user=_USER_TEMPLATE.format(segments=rendered), failure=echo),
            response_format=self.response_format,
            attempts=_ATTEMPTS,
        )
        longest = max(cell, key=lambda segment: len(segment.text.encode("utf-8")))
        texts = [segment.text for segment in cell]
        repair = call_ceiling(
            self.client,
            system=self._repair_system_prompt,
            user=_REPAIR_TEMPLATE.format(
                segment=len(cell),
                text=longest.text,
                # The propositions with their labels, and the excerpts beside them: two allowances of the cell's text.
                earlier=filler(echo_chars(*texts) + echo_chars(*texts)),
                # The widest place a claim can be printed at: a bound on its digits, never a count of claims.
                position=_WIDEST_PLACE,
                listed=filler(echo_chars(longest.text)),
                # The excerpts an ambiguous quote lists, and the quote with every check's fixed text.
                failure=filler(echo_chars(*texts) + echo_chars(longest.text)),
            ),
            response_format=self.repair_response_format,
        )
        if listing is None or repair is None:
            return None
        return listing + repair * self.repairs.max_items_per_cell

    @property
    def _system_prompt(self) -> str:
        return self._with_guidance(_SYSTEM_PROMPT)

    @property
    def _repair_system_prompt(self) -> str:
        return self._with_guidance(_REPAIR_SYSTEM_PROMPT)

    def _with_guidance(self, prompt: str) -> str:
        return prompt + _GUIDANCE_HEADING + self.guidance.strip() if self.guidance.strip() else prompt

    async def extract(
        self, segments: Sequence[Segment], *, ledger: RoleUsageLedger, calls: list[ClaimCall] | None = None
    ) -> tuple[ExtractedClaim, ...]:
        """List the claims ``segments`` state, in the order they appear.

        The claims come back ordered by segment, and a claim's ``restates`` names the earlier position, in the tuple
        returned, of the first claim stating the same fact: the claim its quote resolved to, and the first of that
        one's chain. No segments make no call. Every completion is added to ``ledger``, a cut-short or unreadable one
        included, since each was billed. What the client itself raises (a
        provider failure, a recording that holds no answer) is not caught: it is the caller's to classify.

        :param segments: The cell's segments, in order.
        :param ledger: The cell's ``judge`` ledger.
        :param calls: Where each call is recorded as the grade keeps it, in order: ``extract`` for the listing and its
            retries, ``repair`` for an item asked again; None records none.
        :return: The checked claims.
        :raises ValueError: ``ledger`` is not the ``judge`` role's, or the segments' units are not numbered 0..n-1,
            with no gap.
        :raises ClaimGradingFailed: A reply was cut short, no attempt's reply parsed, a round would ask more items again
            than the repair budget has left, or an item still failed its checks after the repair rounds.
        """
        if ledger.role != _ROLE:
            raise ValueError(
                f"a claim extractor's calls are billed to the {_ROLE!r} role, and the ledger is {ledger.role!r}'s"
            )
        cell = tuple(segments)
        _check_units(cell)
        if not cell:
            return ()
        recorded: list[ClaimCall] = calls if calls is not None else []
        listed = await self._list(cell, ledger, recorded)
        failures = _faults(listed, cell)
        rounds = _REPAIR_ROUNDS if self.repairs is None else self.repairs.rounds
        asked = 0
        for _round in range(rounds):
            if not failures:
                break
            if self.repairs is not None and asked + len(failures) > self.repairs.max_items_per_cell:
                named = "; ".join(f"claim {position}: {failure}" for position, failure in sorted(failures.items()))
                raise ClaimGradingFailed(
                    f"the claim extractor would ask {len(failures)} claim(s) again with {asked} of the step's "
                    f"{self.repairs.max_items_per_cell} repairs spent: {named}",
                    loss="repair_budget_spent",
                )
            asked += len(failures)
            unreadable: dict[int, str] = {}
            for position in sorted(failures):
                repaired, failure = await self._repair(position, listed, cell, failures[position], ledger, recorded)
                listed[position] = repaired
                if failure is not None:
                    unreadable[position] = failure
            # Every item is checked again, not only the repaired ones: a repair that changes a claim's scope or excerpt
            # can leave another's quote naming no claim. A repaired item keeps the failure its repair named.
            failures = {position: unreadable.get(position, fault) for position, fault in _faults(listed, cell).items()}
        if failures:
            named = "; ".join(f"claim {position}: {failure}" for position, failure in sorted(failures.items()))
            raise ClaimGradingFailed(
                f"the claim extractor could not repair {len(failures)} claim(s) in {rounds} rounds: {named}",
                loss="repair_budget_spent",
            )
        return _checked(listed, cell)

    async def _complete(
        self,
        *,
        role: Literal["extract", "repair"],
        system: str,
        user: str,
        response_format: dict[str, Any],
        ledger: RoleUsageLedger,
        calls: list[ClaimCall],
    ) -> str:
        """One billed and recorded call's finished reply.

        :raises ClaimGradingFailed: The reply was cut short.
        """
        completion = await self.client.generate(system=system, user=user, response_format=response_format)
        ledger.add_llm_result(completion)
        calls.append(claim_call(role, completion))
        cut_short = describe_incomplete_completion(completion)
        if cut_short is not None:
            raise ClaimGradingFailed(f"the claim extractor's reply was cut short: {cut_short}", loss="cut_short")
        return completion.content

    async def _list(
        self, cell: tuple[Segment, ...], ledger: RoleUsageLedger, calls: list[ClaimCall]
    ) -> list[_ListedClaim]:
        """The claims one reply lists, asked again in full, naming why, while a finished reply will not parse.

        :raises ClaimGradingFailed: A reply was cut short, or none of the attempts' replies parsed.
        """
        rendered = "\n\n".join(
            _SEGMENT_TEMPLATE.format(index=index, text=segment.text) for index, segment in enumerate(cell)
        )
        first = _USER_TEMPLATE.format(segments=rendered)
        user = first
        unreadable: ValueError | None = None
        for _attempt in range(_ATTEMPTS):
            content = await self._complete(
                role="extract",
                system=self._system_prompt,
                user=user,
                response_format=self.response_format,
                ledger=ledger,
                calls=calls,
            )
            try:
                return _read_listing(content, len(cell))
            except ValueError as refused:
                unreadable = refused
                user = _AGAIN_TEMPLATE.format(user=first, failure=_unreadable(refused))
        raise ClaimGradingFailed(
            f"the claim extractor's reply did not parse in {_ATTEMPTS} attempts", loss="unreadable"
        ) from unreadable

    async def _repair(
        self,
        position: int,
        listed: list[_ListedClaim],
        cell: tuple[Segment, ...],
        failure: str,
        ledger: RoleUsageLedger,
        calls: list[ClaimCall],
    ) -> tuple[_ListedClaim, str | None]:
        """The item at ``position`` asked again alone, with ``failure`` named, and what is still wrong with it.

        :return: The item (the old one when the reply will not parse), and its failure or None.
        :raises ClaimGradingFailed: The reply was cut short.
        """
        item = listed[position]
        kept = _kept_excerpts(listed, cell)
        earlier = [
            _EARLIER_TEMPLATE.format(excerpt=kept[other], claim=listed[other].claim, scope=listed[other].scope)
            for other in _order(listed)
            if (listed[other].segment, other) < (item.segment, position)
        ]
        user = _REPAIR_TEMPLATE.format(
            segment=item.segment,
            text=cell[item.segment].text,
            earlier="\n".join(earlier) or "none",
            position=position,
            listed=json.dumps(item.model_dump(exclude={"segment"}), ensure_ascii=False),
            failure=failure,
        )
        content = await self._complete(
            role="repair",
            system=self._repair_system_prompt,
            user=user,
            response_format=self.repair_response_format,
            ledger=ledger,
            calls=calls,
        )
        try:
            repaired = _Repaired.model_validate(extract_json(content))
        except ValueError as refused:
            return item, f"{failure}; and the last correction could not be used: {_unreadable(refused)}"
        fixed = _ListedClaim(segment=item.segment, **repaired.model_dump())
        listed_after = [*listed[:position], fixed, *listed[position + 1 :]]
        return fixed, _faults(listed_after, cell).get(position)


def _unreadable(refused: ValueError) -> str:
    """Why a reply could not be read, in a line a model can act on."""
    if isinstance(refused, ValidationError):
        problems = "; ".join(
            f"{'/'.join(str(part) for part in error['loc']) or 'the reply'}: {error['msg']}"
            for error in refused.errors()[:5]
        )
        return f"it is not the shape the response format states ({problems})"
    return str(refused)


def _read_listing(content: str, segments: int) -> list[_ListedClaim]:
    """The claims one finished reply lists.

    :raises ValueError: The reply is not the listing's shape, or names a segment the cell does not have.
    """
    listing = _Listing.model_validate(extract_json(content))
    for position, item in enumerate(listing.claims):
        if item.segment >= segments:
            raise ValueError(f"claim {position} names segment {item.segment}, and the segments are 0 to {segments - 1}")
    return list(listing.claims)


def _order(listed: Sequence[_ListedClaim]) -> list[int]:
    """The items' places in the order they appear: by segment, and by place in the list within one."""
    return sorted(range(len(listed)), key=lambda at: (listed[at].segment, at))


def _kept_excerpts(listed: Sequence[_ListedClaim], cell: tuple[Segment, ...]) -> list[str]:
    """Each item's excerpt as it would be kept: its segment's own slice, or the model's string where the segment does
    not hold it (an item that fails its excerpt check, still shown and quotable until it is repaired)."""
    kept: list[str] = []
    for item in listed:
        text = cell[item.segment].text
        span = verbatim_span(item.excerpt, text)
        kept.append(item.excerpt if span is None else text[span[0] : span[1]])
    return kept


def _quoted(words: str) -> str:
    return json.dumps(words, ensure_ascii=False)


def _on_word_boundaries(quote: str, excerpt: str) -> bool:
    """Whether ``quote`` stands verbatim in ``excerpt`` (its first occurrence), starting and ending on word boundaries."""
    span = verbatim_span(quote, excerpt)
    if span is None:
        return False
    start, end = span
    return (start == 0 or not excerpt[start - 1].isalnum()) and (end == len(excerpt) or not excerpt[end].isalnum())


def _restated(listed: Sequence[_ListedClaim], cell: tuple[Segment, ...]) -> list[tuple[int | None, str | None]]:
    """For each item, the first claim of the chain its ``restates`` quote names, by place in ``listed``, and what is
    wrong with the quote: ``(None, None)`` for an item that restates nothing.

    Only claims of the item's scope listed before it can be named, and each stands for the first claim of its own chain,
    so claims of one chain are one name. A quote equal, folded, to some of their excerpts names those; equal to none, it
    names those whose excerpt holds it on word boundaries together with those whose whole excerpt it holds. It names a
    claim when all it names stand for one first claim. A quote of fewer than :data:`_SHORTEST_QUOTE` characters or with
    no letter names nothing. Items are resolved in the order they appear, so every claim a quote can name was resolved
    before it. A claim whose own quote failed stands for itself until it is repaired.
    """
    order = _order(listed)
    kept = _kept_excerpts(listed, cell)
    folded = [VERBATIM_FOLD.fold(excerpt)[0].strip() for excerpt in kept]
    resolved: dict[int, tuple[int | None, str | None]] = {}
    for rank, position in enumerate(order):
        item = listed[position]
        if item.restates is None:
            resolved[position] = (None, None)
            continue
        quote = _quoted(item.restates)
        sought = VERBATIM_FOLD.fold(item.restates)[0].strip()
        if len(sought) < _SHORTEST_QUOTE or not any(character.isalpha() for character in sought):
            resolved[position] = (
                None,
                f"its restates quotes {quote}, a number or a fragment, which names no claim; restates copies the whole "
                f"excerpt of the earlier {item.scope} claim this one restates, exactly as listed, or is null",
            )
            continue
        earlier = [other for other in order[:rank] if listed[other].scope == item.scope]
        named = [other for other in earlier if folded[other] == sought]
        whole = bool(named)
        if not whole:
            named = [
                other
                for other in earlier
                if folded[other]
                # The folded containment is a cheap first pass; the verbatim check on word boundaries is the rule.
                and (
                    (sought in folded[other] and _on_word_boundaries(item.restates, kept[other]))
                    or folded[other] in sought
                )
            ]
        # Each claim named stands for the first claim of its chain, itself where it restates nothing.
        firsts = sorted({first if (first := resolved[other][0]) is not None else other for other in named})
        if not firsts:
            resolved[position] = (
                None,
                f"its restates quotes {quote}, and no {item.scope} claim listed before it has that excerpt; restates "
                f"copies the whole excerpt of the earlier {item.scope} claim this one restates, exactly as listed, or "
                "is null",
            )
        elif len(firsts) > 1 and whole:
            resolved[position] = (
                None,
                f"its restates quotes {quote}, the excerpt of {len(firsts)} earlier {item.scope} claims stating "
                "different facts, so no quote can name one of them: give null, which is the right answer here and "
                "leaves this claim to be checked on its own",
            )
        elif len(firsts) > 1:
            excerpts = "; ".join(dict.fromkeys(_quoted(kept[other]) for other in named))
            resolved[position] = (
                None,
                f"its restates quotes {quote}, which could name {len(firsts)} earlier {item.scope} claims stating "
                f"different facts ({excerpts}); copy the whole excerpt of the one this claim restates, exactly as "
                "listed and no more, or give null",
            )
        else:
            resolved[position] = (firsts[0], None)
    return [resolved[position] for position in range(len(listed))]


def _faults(listed: Sequence[_ListedClaim], cell: tuple[Segment, ...]) -> dict[int, str]:
    """What is wrong with each item that fails a check, by its place in ``listed``.

    An item's excerpt must stand verbatim in its segment, and its ``restates`` quote, when it gives one, must name a
    single earlier claim of its scope (:func:`_restated`).
    """
    faults: dict[int, str] = {}
    for position, (item, (_first, quote_fault)) in enumerate(zip(listed, _restated(listed, cell), strict=True)):
        found: list[str] = []
        if verbatim_span(item.excerpt, cell[item.segment].text) is None:
            found.append(
                f"its excerpt is not in segment {item.segment}; an excerpt is one unbroken stretch of the segment, "
                "copied as it is written"
            )
        if quote_fault is not None:
            found.append(quote_fault)
        if found:
            faults[position] = "; ".join(found)
    return faults


def _checked(listed: Sequence[_ListedClaim], cell: tuple[Segment, ...]) -> tuple[ExtractedClaim, ...]:
    """The checked items, ordered by segment (the list's order within one), each ``restates`` quote resolved to the
    first claim of its chain and renumbered to match.

    :param listed: Items that all pass :func:`_faults`, so every quote names one earlier claim.
    :param cell: The cell's segments.
    """
    order = _order(listed)
    renumbered = {old: new for new, old in enumerate(order)}
    restated = _restated(listed, cell)
    claims: list[ExtractedClaim] = []
    for old in order:
        item = listed[old]
        first, quote_fault = restated[old]
        if quote_fault is not None:  # pragma: no cover - every item passed _faults, which checks this quote
            raise AssertionError(f"claim {old}'s restates was checked and names no single earlier claim")
        segment = cell[item.segment]
        span = verbatim_span(item.excerpt, segment.text)
        if span is None:  # pragma: no cover - every item passed _faults, which checks this span
            raise AssertionError(f"claim {old}'s excerpt was checked and is not in its segment")
        start, end = span
        claims.append(
            ExtractedClaim(
                segment=item.segment,
                unit=segment.unit,
                claim=item.claim.strip(),
                excerpt=segment.text[start:end],
                start=start,
                end=end,
                specific=item.specific,
                hedged=item.hedged,
                attributed=item.attributed,
                scope=item.scope,
                restates=None if first is None else renumbered[first],
            )
        )
    return tuple(claims)


__all__: list[str] = []
