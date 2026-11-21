"""Listing the factual claims a speaker made: the model step that turns a cell's utterances into claims to grade.

A host hands :class:`ClaimExtractor` the cell's utterances in the order they were said (:class:`Utterance`): an
utterance that introduces one of the output's stops (the numbered items it presents in turn) carries that stop's index,
and any other utterance (a message, an aside) carries None. A model lists
every checkable factual proposition the utterances assert, one proposition per claim, each with the excerpt it stands
in and two tags; code then checks every item and stores what it checked (:class:`ExtractedClaim`). The extractor
never sees the case's key, its sources or the candidate's own findings, so the list is not primed by the evidence that
grades it.

**What a claim is** is stated generically in the prompt: descriptions, opinions, talk about the programme itself and
echoes of the request are not claims; compounds are split; a framing's implied claim counts. A host whose domain
needs more says so in ``guidance``, in its own words and with its own examples, which stay in the host's code.

**The excerpt is the utterance's own slice.** Each listed excerpt is found in its utterance by
:func:`~threetears.evals.kernel.verbatim.verbatim_span`, which forgives typography and case and nothing else, and the
stored excerpt is the utterance's text at the offsets found, never the model's string.

**Repair, not refusal.** A reply that will not parse is asked again in full, naming why, twice, then raises. An item
that fails a check (an excerpt not in its utterance or shortened with an ellipsis, a ``restates`` that names no
earlier claim) is asked again alone, with the failure named, in up to two repair rounds; one still failing then raises.
A failing item is never dropped or downgraded, since dropping the claims a model found hardest to quote would bias
every reading counted over them. A reply cut short raises at once, since the same request would meet the same cap.
Every failure is :class:`ClaimExtractionFailed`, an :class:`~threetears.evals.kernel.host.apparatus.ApparatusError`:
raised out of a kind's ``invoke``, the runner excludes the cell as ``apparatus_failed`` and never scores it.

**It records what it spent, and how the list was made.** Every completion, parsed or not, is folded into the cell's
``judge`` ledger. :attr:`ClaimExtractor.identity` names the model and a digest of everything that shapes the list: the
prompts (the host's ``guidance`` in them), the reply schemas, the verbatim fold, the retry and repair budgets, and the
host's ``segmentation``, the name of how it cut its output into utterances and stops, which sets every per-stop
denominator. Claims listed two ways never pool.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import ConfigDict, Field, ValidationError

from threetears.evals.kernel.authored import strict_response_format
from threetears.evals.kernel.host.apparatus import ApparatusError
from threetears.evals.kernel.provider import describe_incomplete_completion, extract_json
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.kernel.verbatim import VERBATIM_FOLD, is_elided, verbatim_span
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.completion import CompletionClient
from threetears.evals.schema.hashing import canonical_digest

#: How many times the whole cell is asked about: the first call and two more when a finished reply will not parse.
_ATTEMPTS: Final = 3

#: How many rounds a failing item is asked again alone before the extraction raises.
_REPAIR_ROUNDS: Final = 2

#: The role every extraction call is billed under: it reads the candidate's output, as a judge does.
_ROLE: Final = "judge"

#: Something other than whitespace: what a claim's text and an excerpt must hold.
_NOT_BLANK: Final = r"\S"

_RULES: Final = (
    "A claim is one checkable factual proposition the speaker asserts: a name, a date or year, a place, who made or "
    "took part in something, what something belongs to or is part of, a first, a record, or a number or duration "
    "stated as fact. List every one.\n"
    "- One proposition per claim: a sentence that states two facts is two claims.\n"
    "- A framing's implied claim counts: introducing a work as someone's asserts that it is theirs.\n"
    "- A sweeping historical or causal statement (that one event started a trend) is a claim.\n"
    "- Not claims: descriptions of how something sounds, looks or feels; opinions and evaluations, a familiar "
    "epithet included; talk about the programme itself (what was just presented, what comes next); and echoes of "
    "the request that add no fact.\n"
    "For each claim give:\n"
    "- claim: the proposition, stated on its own in one sentence.\n"
    "- excerpt: the shortest unbroken stretch of the utterance that carries the claim, copied exactly as it is "
    "written, never shortened with an ellipsis.\n"
    "- specific: true when the claim states a particular detail a source could confirm or refute (a name, a number, "
    "a date, a place, a credit); false for a general statement.\n"
    '- hedged: true only when the speaker marks the claim as their own uncertain recollection ("I believe", '
    '"if memory serves", "roughly, if I recall"); false otherwise. A hedged claim is still a claim.\n'
    "- restates: when the claim states a fact an earlier claim in the list already stated, that claim's position in "
    "the list, counting from 0; otherwise null."
)

_GUIDANCE_HEADING: Final = "\n\n# Notes on what counts as a claim here\n\n"

_SYSTEM_PROMPT: Final = (
    "You list the factual claims in what a speaker said. The utterances are numbered U0, U1, and so on, in the "
    "order they were said. List the claims in the order they are heard, each with the number of the utterance it "
    "is in.\n\n" + _RULES + "\n\nAnswer with JSON only, in the shape the response format states."
)

_USER_TEMPLATE: Final = "# Utterances\n\n{utterances}"

_UTTERANCE_TEMPLATE: Final = "U{index}:\n{text}"

_AGAIN_TEMPLATE: Final = "{user}\n\n# Your last reply\n\nIt could not be used: {failure}. Answer again, in full."

_REPAIR_SYSTEM_PROMPT: Final = (
    "You correct one claim from a list of the factual claims in what a speaker said. You are given the utterance "
    "the claim is in, the claims listed before it, the claim as it was listed, and what is wrong with it. Give the "
    "claim again with that fixed, as the same proposition.\n\n" + _RULES + "\n\nAnswer with JSON only, in the shape "
    "the response format states."
)

_REPAIR_TEMPLATE: Final = (
    "# Utterance U{utterance}\n\n{text}\n\n# Claims listed before it\n\n{earlier}\n\n"
    "# The claim, at position {position}\n\n{listed}\n\n# What is wrong\n\n{failure}"
)

_CLAIM_PROPERTIES: Final[dict[str, Any]] = {
    "claim": {"type": "string"},
    "excerpt": {"type": "string"},
    "specific": {"type": "boolean"},
    "hedged": {"type": "boolean"},
    "restates": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
}


class ClaimExtractionFailed(ApparatusError):
    """The extractor could not list a cell's claims: a reply was cut short, a reply would not parse after retries,
    or an item still failed its checks after the repair rounds.

    A fault in the measuring rig, never the candidate's. Out of a kind's ``invoke`` the runner records the cell
    excluded as ``apparatus_failed`` and goes on, so it is never scored as a cell that said nothing false.
    """


class Utterance(EvalBaseModel):
    """One thing the candidate said, as the host cut its output: a stop's introduction with its index, or anything else."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(pattern=_NOT_BLANK, description="What was said, as the candidate's output holds it.")
    stop_index: int | None = Field(
        default=None,
        ge=0,
        description="The index of the stop this utterance introduces, when it introduces one: the cell's stops are "
        "numbered 0..n-1, each once. None for any other utterance (a message, an aside), whose claims count but "
        "which is no stop.",
    )


class ExtractedClaim(EvalBaseModel):
    """One claim the extractor listed, checked: where it was said, the proposition, its excerpt and its tags."""

    model_config = ConfigDict(frozen=True)

    utterance: int = Field(ge=0, description="The position of the utterance it was said in, among the cell's.")
    stop_index: int | None = Field(
        ge=0, description="That utterance's stop, or None when it introduces none (a message, an aside)."
    )
    claim: str = Field(min_length=1, description="The proposition, stated on its own: one fact.")
    excerpt: str = Field(
        min_length=1,
        description="The utterance's own text at start..end: the stretch the claim stands in, never the model's "
        "copy of it.",
    )
    start: int = Field(ge=0, description="Where the excerpt starts in the utterance's text.")
    end: int = Field(ge=1, description="Where it ends (exclusive).")
    specific: bool = Field(
        description="Whether the claim states a particular detail a source could settle (a name, a number, a date, a "
        "place, a credit). The model's tag; no human label calibrates it yet."
    )
    hedged: bool = Field(
        description="Whether the speaker marked it as their own uncertain recollection. A hedged claim is still graded "
        "for truth."
    )
    restates: int | None = Field(
        ge=0,
        description="The position, among the cell's claims, of an earlier claim stating the same fact, or None. "
        "Always earlier than this claim's own position.",
    )


class _ListedClaim(EvalBaseModel):
    """One claim as the extractor's reply lists it."""

    utterance: int = Field(ge=0)
    claim: str = Field(pattern=_NOT_BLANK)
    excerpt: str = Field(pattern=_NOT_BLANK)
    specific: bool
    hedged: bool
    restates: int | None


class _Listing(EvalBaseModel):
    """The extractor's whole reply."""

    claims: list[_ListedClaim]


class _Repaired(EvalBaseModel):
    """One claim as a repair reply gives it again: the same item, its utterance fixed by the request."""

    claim: str = Field(pattern=_NOT_BLANK)
    excerpt: str = Field(pattern=_NOT_BLANK)
    specific: bool
    hedged: bool
    restates: int | None


def _check_stops(utterances: Sequence[Utterance]) -> None:
    """Refuse a cell whose stops are not numbered 0..n-1, each once.

    :param utterances: The cell's utterances.
    :raises ValueError: They are not.
    """
    stops = [utterance.stop_index for utterance in utterances if utterance.stop_index is not None]
    if sorted(stops) != list(range(len(stops))):
        raise ValueError(
            f"a cell's stops are numbered 0..n-1, each once, and its utterances carry the stop indices {stops}: the "
            "host's segmentation numbers each stop's introduction by its stop"
        )


@dataclass(frozen=True)
class ClaimExtractor:
    """A model step that lists the factual claims a cell's utterances assert, each checked against its utterance.

    :ivar client: The completion client the step calls. It must name its model (``model_name``), since an unnamed
        extractor would let two models' lists pool.
    :ivar segmentation: The host's name for how it cuts a candidate's output into utterances and stops. It enters the
        identity, since it sets every per-stop denominator: change it whenever the cutting changes.
    :ivar guidance: The host's own notes on what counts as a claim in its domain, with its own examples, appended to
        the prompts. Empty by default.
    """

    client: CompletionClient
    segmentation: str
    guidance: str = ""

    def __post_init__(self) -> None:
        """Refuse a client that names no model, and a blank segmentation.

        :raises ValueError: One of those.
        """
        model = getattr(self.client, "model_name", None)
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"a claim extractor's client names its model (model_name), and {type(self.client).__name__} names "
                "none: the measures record which model listed the claims, so an unnamed one would pool with any other"
            )
        if not self.segmentation.strip():
            raise ValueError(
                "a claim extractor names the host's segmentation, how its output is cut into utterances and stops, "
                "and this one is blank: it sets every per-stop denominator, so it is part of the identity"
            )

    @property
    def model_name(self) -> str:
        """The model the client calls."""
        # Checked present and a non-blank string at construction.
        return str(getattr(self.client, "model_name", ""))

    @property
    def response_format(self) -> dict[str, Any]:
        """The strict ``json_schema`` directive the listing call sends: ``{"claims": [{"utterance", "claim",
        "excerpt", "specific", "hedged", "restates"}]}``."""
        item = {"type": "object", "properties": {"utterance": {"type": "integer"}, **_CLAIM_PROPERTIES}}
        schema = {"type": "object", "properties": {"claims": {"type": "array", "items": item}}}
        return strict_response_format("claim_listing", schema)

    @property
    def repair_response_format(self) -> dict[str, Any]:
        """The strict ``json_schema`` directive a repair call sends: one claim, without its utterance, which the
        request fixes."""
        return strict_response_format("claim_repair", {"type": "object", "properties": dict(_CLAIM_PROPERTIES)})

    @property
    def identity(self) -> str:
        """``<model>|claims:<sha12>``: the model, and a digest of the prompts (with the host's guidance), the reply
        schemas, the verbatim fold, the retry and repair budgets and the host's segmentation."""
        digest = canonical_digest(
            {
                "system": self._system_prompt,
                "user": [_USER_TEMPLATE, _UTTERANCE_TEMPLATE, _AGAIN_TEMPLATE],
                "repair_system": self._repair_system_prompt,
                "repair_user": _REPAIR_TEMPLATE,
                "schema": self.response_format,
                "repair_schema": self.repair_response_format,
                "fold": VERBATIM_FOLD,
                "attempts": _ATTEMPTS,
                "repair_rounds": _REPAIR_ROUNDS,
                "segmentation": self.segmentation,
            }
        )
        return f"{self.model_name}|claims:{digest[:12]}"

    @property
    def _system_prompt(self) -> str:
        return self._with_guidance(_SYSTEM_PROMPT)

    @property
    def _repair_system_prompt(self) -> str:
        return self._with_guidance(_REPAIR_SYSTEM_PROMPT)

    def _with_guidance(self, prompt: str) -> str:
        return prompt + _GUIDANCE_HEADING + self.guidance.strip() if self.guidance.strip() else prompt

    async def extract(self, utterances: Sequence[Utterance], *, ledger: RoleUsageLedger) -> tuple[ExtractedClaim, ...]:
        """List the claims ``utterances`` assert, in the order they are heard.

        The claims come back ordered by utterance, and a claim's ``restates`` names an earlier position in the tuple
        returned. No utterances make no call. Every completion is added to ``ledger``, a cut-short or unreadable one
        included, since each was billed. What the client itself raises (a provider failure, a recording that holds no
        answer) is not caught: it is the caller's to classify.

        :param utterances: The cell's utterances, in the order they were said.
        :param ledger: The cell's ``judge`` ledger.
        :return: The checked claims.
        :raises ValueError: ``ledger`` is not the ``judge`` role's, or the utterances' stops are not numbered 0..n-1,
            each once.
        :raises ClaimExtractionFailed: A reply was cut short, no attempt's reply parsed, or an item still failed its
            checks after the repair rounds.
        """
        if ledger.role != _ROLE:
            raise ValueError(
                f"a claim extractor's calls are billed to the {_ROLE!r} role, and the ledger is {ledger.role!r}'s"
            )
        cell = tuple(utterances)
        _check_stops(cell)
        if not cell:
            return ()
        listed = await self._list(cell, ledger)
        failures = {position: failure for position in range(len(listed)) if (failure := _fault(position, listed, cell))}
        for _round in range(_REPAIR_ROUNDS):
            if not failures:
                break
            still: dict[int, str] = {}
            for position in sorted(failures):
                repaired, failure = await self._repair(position, listed, cell, failures[position], ledger)
                listed[position] = repaired
                if failure is not None:
                    still[position] = failure
            failures = still
        if failures:
            named = "; ".join(f"claim {position}: {failure}" for position, failure in sorted(failures.items()))
            raise ClaimExtractionFailed(
                f"the claim extractor could not repair {len(failures)} claim(s) in {_REPAIR_ROUNDS} rounds: {named}"
            )
        return _checked(listed, cell)

    async def _complete(
        self, *, system: str, user: str, response_format: dict[str, Any], ledger: RoleUsageLedger
    ) -> str:
        """One billed call's finished reply.

        :raises ClaimExtractionFailed: The reply was cut short.
        """
        completion = await self.client.generate(system=system, user=user, response_format=response_format)
        ledger.add_llm_result(completion)
        cut_short = describe_incomplete_completion(completion)
        if cut_short is not None:
            raise ClaimExtractionFailed(f"the claim extractor's reply was cut short: {cut_short}")
        return completion.content

    async def _list(self, cell: tuple[Utterance, ...], ledger: RoleUsageLedger) -> list[_ListedClaim]:
        """The claims one reply lists, asked again in full, naming why, while a finished reply will not parse.

        :raises ClaimExtractionFailed: A reply was cut short, or none of the attempts' replies parsed.
        """
        rendered = "\n\n".join(
            _UTTERANCE_TEMPLATE.format(index=index, text=utterance.text) for index, utterance in enumerate(cell)
        )
        first = _USER_TEMPLATE.format(utterances=rendered)
        user = first
        unreadable: ValueError | None = None
        for _attempt in range(_ATTEMPTS):
            content = await self._complete(
                system=self._system_prompt, user=user, response_format=self.response_format, ledger=ledger
            )
            try:
                return _read_listing(content, len(cell))
            except ValueError as refused:
                unreadable = refused
                user = _AGAIN_TEMPLATE.format(user=first, failure=_unreadable(refused))
        raise ClaimExtractionFailed(
            f"the claim extractor's reply did not parse in {_ATTEMPTS} attempts"
        ) from unreadable

    async def _repair(
        self,
        position: int,
        listed: list[_ListedClaim],
        cell: tuple[Utterance, ...],
        failure: str,
        ledger: RoleUsageLedger,
    ) -> tuple[_ListedClaim, str | None]:
        """The item at ``position`` asked again alone, with ``failure`` named, and what is still wrong with it.

        :return: The item (the old one when the reply will not parse), and its failure or None.
        :raises ClaimExtractionFailed: The reply was cut short.
        """
        item = listed[position]
        earlier = [
            f"{other}: {listed[other].claim}"
            for other in sorted(range(len(listed)), key=lambda at: (listed[at].utterance, at))
            if (listed[other].utterance, other) < (item.utterance, position)
        ]
        user = _REPAIR_TEMPLATE.format(
            utterance=item.utterance,
            text=cell[item.utterance].text,
            earlier="\n".join(earlier) or "none",
            position=position,
            listed=json.dumps(item.model_dump(exclude={"utterance"}), ensure_ascii=False),
            failure=failure,
        )
        content = await self._complete(
            system=self._repair_system_prompt, user=user, response_format=self.repair_response_format, ledger=ledger
        )
        try:
            repaired = _Repaired.model_validate(extract_json(content))
        except ValueError as refused:
            return item, f"{failure}; and the last correction could not be used: {_unreadable(refused)}"
        fixed = _ListedClaim(utterance=item.utterance, **repaired.model_dump())
        listed_after = [*listed[:position], fixed, *listed[position + 1 :]]
        return fixed, _fault(position, listed_after, cell)


def _unreadable(refused: ValueError) -> str:
    """Why a reply could not be read, in a line a model can act on."""
    if isinstance(refused, ValidationError):
        problems = "; ".join(
            f"{'/'.join(str(part) for part in error['loc']) or 'the reply'}: {error['msg']}"
            for error in refused.errors()[:5]
        )
        return f"it is not the shape the response format states ({problems})"
    return str(refused)


def _read_listing(content: str, utterances: int) -> list[_ListedClaim]:
    """The claims one finished reply lists.

    :raises ValueError: The reply is not the listing's shape, or names an utterance the cell does not have.
    """
    listing = _Listing.model_validate(extract_json(content))
    for position, item in enumerate(listing.claims):
        if item.utterance >= utterances:
            raise ValueError(
                f"claim {position} names utterance U{item.utterance}, and the utterances are U0 to U{utterances - 1}"
            )
    return list(listing.claims)


def _fault(position: int, listed: Sequence[_ListedClaim], cell: tuple[Utterance, ...]) -> str | None:
    """What is wrong with the item at ``position``, or None.

    Its excerpt must stand verbatim in its utterance, unelided, and its ``restates`` must name a claim heard before
    it: one in an earlier utterance, or earlier in the list within the same utterance.
    """
    item = listed[position]
    faults: list[str] = []
    if is_elided(item.excerpt):
        faults.append(
            "its excerpt is shortened with an ellipsis; an excerpt is one unbroken stretch of the utterance, copied as "
            "it is written"
        )
    elif verbatim_span(item.excerpt, cell[item.utterance].text) is None:
        faults.append(
            f"its excerpt is not in utterance U{item.utterance}; an excerpt is copied from the utterance as it is "
            "written"
        )
    if item.restates is not None:
        target = item.restates
        if not 0 <= target < len(listed) or target == position:
            faults.append(f"its restates names claim {target}, and the list has no such other claim")
        elif (listed[target].utterance, target) > (item.utterance, position):
            faults.append(
                f"its restates names claim {target}, which is heard after it; restates names an earlier claim, or null"
            )
    return "; ".join(faults) or None


def _checked(listed: Sequence[_ListedClaim], cell: tuple[Utterance, ...]) -> tuple[ExtractedClaim, ...]:
    """The checked items, ordered by utterance (the list's order within one), with ``restates`` renumbered to match.

    :param listed: Items that all pass :func:`_fault`.
    :param cell: The cell's utterances.
    """
    order = sorted(range(len(listed)), key=lambda at: (listed[at].utterance, at))
    renumbered = {old: new for new, old in enumerate(order)}
    claims: list[ExtractedClaim] = []
    for old in order:
        item = listed[old]
        utterance = cell[item.utterance]
        span = verbatim_span(item.excerpt, utterance.text)
        if span is None:  # pragma: no cover - every item passed _fault, which checks this span
            raise AssertionError(f"claim {old}'s excerpt was checked and is not in its utterance")
        start, end = span
        claims.append(
            ExtractedClaim(
                utterance=item.utterance,
                stop_index=utterance.stop_index,
                claim=item.claim.strip(),
                excerpt=utterance.text[start:end],
                start=start,
                end=end,
                specific=item.specific,
                hedged=item.hedged,
                restates=None if item.restates is None else renumbered[item.restates],
            )
        )
    return tuple(claims)


__all__ = ["ClaimExtractionFailed", "ClaimExtractor", "ExtractedClaim", "Utterance"]
