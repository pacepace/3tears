"""The ``restates`` scorer over the claim corpus's labelled repeats, on scripted extractor replies.

Nothing is spent: each corpus result is listed by a scripted client, through the real extractor, quote resolution and
all, and :func:`score_restates` counts what came back. The script tags each labelled repeat one of three ways in turn,
quoting its original's stretch (a hit), quoting another earlier claim's (a wrong target) or leaving it untagged (a
miss), and the score must give back exactly the script's counts. A hand-made cell holds every outcome once, the
unlisted repeat and the spurious tag among them.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packages.evals.tests.claim_corpus_restates import (
    CorpusCell,
    LabelledClaim,
    RestatesScore,
    corpus_cells,
    score_restates,
)
from threetears.evals.kernel import RoleUsageLedger
from threetears.evals.kernel.claims.extraction import ClaimExtractor, ExtractedClaim
from threetears.evals.kernel.claims.verbatim import verbatim_span

_MODEL = "scripted/claims"


@dataclass(frozen=True)
class _Reply:
    """One completion, shaped as the engine's completion port describes it."""

    content: str
    stop_reason: str = "end_turn"
    model: str = _MODEL
    served_model: str | None = _MODEL
    input_tokens: int | None = 900
    output_tokens: int | None = 600
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.0
    price_source: str | None = "scripted"
    temperature: float | None = None


# parity-with: threetears.evals.schema.CompletionClient
@dataclass
class _FakeListingClient:
    """Answers the one listing call with its scripted reply; any second call (a repair) fails the test."""

    listing: str
    model_name: str = _MODEL
    calls: list[str] = field(default_factory=list)

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        del system, response_format
        self.calls.append(user)
        if len(self.calls) > 1:
            raise AssertionError(f"the script quotes so every tag names one claim, and a repair was asked: {user}")
        return _Reply(content=self.listing)

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeListingClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


@dataclass(frozen=True)
class _Stretch:
    """One distinct labelled stretch of a cell: what the script lists as one claim."""

    segment: int
    start: int
    excerpt: str
    claim: str


def _stretches(cell: CorpusCell) -> list[_Stretch]:
    seen: dict[tuple[int, int], _Stretch] = {}
    for labelled in cell.labelled:
        place = (labelled.segment, labelled.start)
        seen.setdefault(place, _Stretch(labelled.segment, labelled.start, labelled.excerpt, labelled.claim))
    return [seen[place] for place in sorted(seen)]


def _holders(quote: str, earlier: list[_Stretch]) -> list[_Stretch]:
    return [stretch for stretch in earlier if verbatim_span(quote, stretch.excerpt) is not None]


def _script(cell: CorpusCell, turn: list[int], plan: Counter[str]) -> str:
    """The listing a scripted extractor gives for ``cell``: each stretch once, each labelled repeat tagged in turn.

    A hit quotes the original's stretch; a wrong target quotes the nearest earlier untagged stretch that is not the
    original's; a miss gives null. A quote more than one earlier stretch holds would be asked again, so a repeat whose
    turn's quote is not held by exactly one stretch is left a miss. A stretch carrying several labelled repeats is
    scripted once, for the first: the others are misses where it is left untagged, and hits only where its tag names
    their own original's stretch. ``plan`` counts what was scripted.
    """
    stretches = _stretches(cell)
    at = {(stretch.segment, stretch.start): rank for rank, stretch in enumerate(stretches)}
    tags: dict[int, tuple[str, _Stretch]] = {}
    scripted: set[int] = set()
    for repeat in sorted((c for c in cell.labelled if c.repeat_of), key=lambda c: (c.segment, c.start)):
        original = cell.by_id(repeat.repeat_of or "")
        rank = at[(repeat.segment, repeat.start)]
        if rank in scripted:
            named = tags[rank][1] if rank in tags else None
            if named is None:
                plan["miss"] += 1
            else:
                plan["hit" if (named.segment, named.start) == (original.segment, original.start) else "wrong"] += 1
            continue
        scripted.add(rank)
        earlier = stretches[:rank]
        outcome = ("hit", "wrong", "miss")[turn[0] % 3]
        turn[0] += 1
        target: _Stretch | None = None
        if outcome == "hit":
            target = stretches[at[(original.segment, original.start)]]
        elif outcome == "wrong":
            others = [
                stretch
                for back, stretch in reversed(list(enumerate(earlier)))
                if back not in tags
                and (stretch.segment, stretch.start) != (original.segment, original.start)
                and _holders(stretch.excerpt, earlier) == [stretch]
            ]
            target = others[0] if others else None
        if target is None or len(_holders(target.excerpt, earlier)) != 1:
            outcome, target = "miss", None
        if target is not None:
            tags[rank] = (target.excerpt, target)
        plan[outcome] += 1
    items = [
        {
            "segment": stretch.segment,
            "claim": stretch.claim,
            "excerpt": stretch.excerpt,
            "specific": True,
            "hedged": False,
            "attributed": False,
            "scope": "world",
            "restates": tags[rank][0] if rank in tags else None,
        }
        for rank, stretch in enumerate(stretches)
    ]
    return json.dumps({"claims": items})


async def test_the_score_over_the_corpus_gives_back_the_scripted_hits_misses_and_wrong_targets() -> None:
    cells = [cell for cell in corpus_cells() if any(claim.repeat_of for claim in cell.labelled)]
    turn = [0]
    plan: Counter[str] = Counter()
    scored: list[tuple[CorpusCell, tuple[ExtractedClaim, ...]]] = []
    for cell in cells:
        client = _FakeListingClient(_script(cell, turn, plan))
        claims = await ClaimExtractor(client).extract(cell.segments, ledger=RoleUsageLedger(role="judge"))
        scored.append((cell, claims))

    score = score_restates(scored)

    assert score == RestatesScore(
        hits=plan["hit"], wrong_targets=plan["wrong"], misses=plan["miss"], unlisted=0, spurious=0
    )
    assert min(plan["hit"], plan["wrong"], plan["miss"]) > 0, "the script reaches every outcome"
    labelled = sum(1 for cell in cells for claim in cell.labelled if claim.repeat_of)
    assert sum(plan.values()) == labelled, "every labelled repeat is scored once"


def _hand_made(tmp_path: Path) -> CorpusCell:
    """One result of three lines: two facts said, then each said again, then a third repeated twice over."""
    rows = [
        ("c-01", "U0", "built in 1894", None),
        ("c-02", "U0", "by Hela Orsk", None),
        ("c-03", "U1", "the 1894 bridge", "c-01"),
        ("c-04", "U1", "Orsk's work", "c-02"),
        ("c-05", "U1", "spans the harbour", None),
        ("c-06", "U2", "harbour span", "c-05"),
        ("c-07", "U2", "the mill is older", None),
        ("c-08", "U2", "still standing", "c-01"),
    ]
    path = tmp_path / "claims.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "claim_id": claim_id,
                    "result_id": "r-1",
                    "utterance_ref": ref,
                    "utterance_excerpt": excerpt,
                    "claim": f"Claim {claim_id}.",
                    "repeat_of": repeat_of,
                }
            )
            for claim_id, ref, excerpt, repeat_of in rows
        ),
        encoding="utf-8",
    )
    (cell,) = corpus_cells(path)
    return cell


def _listed(at: int, labelled: LabelledClaim, restates: int | None = None) -> tuple[int, ExtractedClaim]:
    return at, ExtractedClaim(
        segment=labelled.segment,
        unit=labelled.segment,
        claim=labelled.claim,
        excerpt=labelled.excerpt,
        start=labelled.start,
        end=labelled.end,
        specific=True,
        hedged=False,
        attributed=False,
        scope="world",
        restates=restates,
    )


def test_each_outcome_is_counted_once_on_a_hand_made_cell(tmp_path: Path) -> None:
    cell = _hand_made(tmp_path)
    claim = {labelled.claim_id: labelled for labelled in cell.labelled}
    listed = dict(
        [
            _listed(0, claim["c-01"]),
            _listed(1, claim["c-02"]),
            _listed(2, claim["c-03"], restates=0),  # a hit: names the claim standing for c-01
            _listed(3, claim["c-04"], restates=0),  # a wrong target: c-02 is the original
            _listed(4, claim["c-05"], restates=1),  # spurious: c-05 is no repeat
            _listed(5, claim["c-06"]),  # a miss: listed, untagged
            _listed(6, claim["c-07"]),
            # c-08 is said again and nothing stands for it: unlisted.
        ]
    )

    score = score_restates([(cell, tuple(listed[at] for at in sorted(listed)))])

    assert score == RestatesScore(hits=1, wrong_targets=1, misses=1, unlisted=1, spurious=1)
    assert cell.segments[1].text == "the 1894 bridge | Orsk's work | spans the harbour"
    assert cell.segments[1].text[claim["c-04"].start : claim["c-04"].end] == "Orsk's work"
