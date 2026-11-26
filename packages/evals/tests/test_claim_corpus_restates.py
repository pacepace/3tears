"""The ``restates`` scorer over the claim corpus's labelled repeats, on scripted extractor replies.

Nothing is spent: each corpus result is listed by a scripted client, one claim per labelled claim, through the real
extractor, quote resolution and repairs and all, and :func:`score_restates` counts what came back. The script tags
each labelled repeat one of three ways in turn, quoting its original's excerpt (a hit), quoting another earlier
claim's (a wrong target) or leaving it untagged (a miss), and the score must give back exactly the script's counts. An
original whose excerpt the corpus labels for several facts cannot be named by any quote, so its repair is answered
null, and it scores a miss. A hand-made cell holds every outcome once, the
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
from threetears.evals.kernel.claims.verbatim import VERBATIM_FOLD

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


#: What a repair prompt says where the only right answer is null: the claim restated shares its excerpt with another.
_NULL_IS_THE_ANSWER = "give null, which is the right answer here"


# parity-with: threetears.evals.schema.CompletionClient
@dataclass
class _FakeListingClient:
    """Answers the listing call with its scripted reply, and a repair asked where null is the only right answer with
    the item given again untagged; any other repair fails the test."""

    listing: str
    model_name: str = _MODEL
    calls: list[str] = field(default_factory=list)
    nulled: int = 0

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        del system, response_format
        self.calls.append(user)
        if len(self.calls) == 1:
            return _Reply(content=self.listing)
        if _NULL_IS_THE_ANSWER not in user:
            raise AssertionError(
                f"the script quotes so every other tag names one claim, and a repair was asked: {user}"
            )
        listed = user.split("\n\n# What is wrong")[0].rsplit("\n\n", 1)[1]
        self.nulled += 1
        return _Reply(content=json.dumps({**json.loads(listed), "restates": None}))

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeListingClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _folded(text: str) -> str:
    return VERBATIM_FOLD.fold(text)[0].strip()


def _overlap(a: LabelledClaim, b: LabelledClaim) -> bool:
    return a.segment == b.segment and a.start < b.end and b.start < a.end


def _quotable(excerpt: str) -> bool:
    folded = _folded(excerpt)
    return len(folded) >= 4 and any(character.isalpha() for character in folded)


def _script(cell: CorpusCell, turn: list[int], plan: Counter[str]) -> str:
    """The listing a scripted extractor gives for ``cell``: each labelled claim once, each labelled repeat tagged in turn.

    A hit quotes its original's whole excerpt; a wrong target quotes the whole excerpt of the nearest earlier untagged
    claim apart from the original whose excerpt no other earlier claim has; a miss gives null. A quote that the excerpts
    of claims of different facts equal is answered null on its repair, the only right answer, so it is a miss, and
    ``plan["shared"]`` counts those. A stretch carrying several labelled repeats is tagged once, on the first, and the
    others score as that tag names their own originals. ``plan`` counts what is expected.
    """
    claims = sorted(cell.labelled, key=lambda claim: (claim.segment, claim.start))
    firsts: dict[int, int] = {}
    tagged: dict[tuple[int, int], int | None] = {}
    tags: dict[int, str] = {}
    for at, repeat in enumerate(claims):
        if repeat.repeat_of is None:
            continue
        original = cell.by_id(repeat.repeat_of)
        place = (repeat.segment, repeat.start)
        if place not in tagged:
            earlier = list(range(at))
            outcome = ("hit", "wrong", "miss")[turn[0] % 3]
            turn[0] += 1
            target: int | None = None
            if outcome == "hit" and _quotable(original.excerpt):
                equal = {
                    firsts.get(other, other)
                    for other in earlier
                    if _folded(claims[other].excerpt) == _folded(original.excerpt)
                }
                if len(equal) == 1:
                    target = equal.pop()
                else:
                    plan["shared"] += 1
                tags[at] = original.excerpt
            elif outcome == "wrong":
                for other in reversed(earlier):
                    candidate = claims[other]
                    same = [o for o in earlier if _folded(claims[o].excerpt) == _folded(candidate.excerpt)]
                    if (
                        other not in firsts
                        and not _overlap(candidate, original)
                        and same == [other]
                        and _quotable(candidate.excerpt)
                    ):
                        target = other
                        tags[at] = candidate.excerpt
                        break
            tagged[place] = target
            if target is not None:
                firsts[at] = target
        target = tagged[place]
        if target is None:
            plan["miss"] += 1
        else:
            plan["hit" if _overlap(claims[target], original) else "wrong"] += 1
    items = [
        {
            "segment": claim.segment,
            "claim": claim.claim,
            "excerpt": claim.excerpt,
            "specific": True,
            "hedged": False,
            "attributed": False,
            "scope": "world",
            "restates": tags.get(at),
        }
        for at, claim in enumerate(claims)
    ]
    return json.dumps({"claims": items})


async def test_the_score_over_the_corpus_gives_back_the_scripted_hits_misses_and_wrong_targets() -> None:
    cells = [cell for cell in corpus_cells() if any(claim.repeat_of for claim in cell.labelled)]
    turn = [0]
    plan: Counter[str] = Counter()
    scored: list[tuple[CorpusCell, tuple[ExtractedClaim, ...]]] = []
    nulled = 0
    for cell in cells:
        client = _FakeListingClient(_script(cell, turn, plan))
        claims = await ClaimExtractor(client).extract(cell.segments, ledger=RoleUsageLedger(role="judge"))
        scored.append((cell, claims))
        nulled += client.nulled

    score = score_restates(scored)

    assert score == RestatesScore(
        hits=plan["hit"], wrong_targets=plan["wrong"], misses=plan["miss"], unlisted=0, spurious=0
    )
    assert min(plan["hit"], plan["wrong"], plan["miss"]) > 0, "the script reaches every outcome"
    # The hits the script could not name: their originals share an excerpt with a claim of another fact, so their
    # quotes were asked again and answered null. Counted by this test's own run over the corpus, at this rotation.
    assert nulled == plan["shared"] == 2, "an original sharing its excerpt with another fact is named only by null"
    labelled = sum(1 for cell in cells for claim in cell.labelled if claim.repeat_of)
    assert sum(plan[outcome] for outcome in ("hit", "wrong", "miss")) == labelled, "every labelled repeat scored once"


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
