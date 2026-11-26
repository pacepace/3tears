"""Scoring a listing's ``restates`` tags against the claim corpus's labelled repeats.

The corpus (``fixtures/claim_corpus/claimset/claims.jsonl``, CC-BY 4.0, Discodon / Kairo (Radio Discodon); its README
says what it holds) labels each claim a broadcast line made, and a claim that says again what an earlier claim of the
same result said carries ``repeat_of``, the earlier claim's id. Those rows are the only human labels of a restatement
this package holds, so they are what the extractor's ``restates`` tag is scored on.

**The cells are the corpus's own words.** The corpus stores the stretch each claim was said in, never the whole line,
so a result becomes a cell of one segment per line (``utterance_ref``), in the order the lines first appear, whose text
is that line's labelled stretches in the order they appear, joined by :data:`JOIN`, each added only when the text so
far does not already hold it (labelled stretches of one line often overlap). Each labelled claim then stands where its
stretch first stands in its segment, found by the same verbatim check the extractor keeps an excerpt by
(:func:`corpus_cells`). An extractor run over these cells reads fragments,
not the lines as said; a run over the lines themselves needs the host's stored results.

**A listed claim stands for a labelled one** when it is in the same segment and its kept span overlaps the labelled
claim's. The corpus labels a stretch that often carries several claims (a run of band names), so a listed claim
standing for a labelled original may state another of that stretch's facts: the score is as fine as the labels.

:func:`score_restates` counts each labelled repeat once (:class:`RestatesScore`): a hit when a claim standing for it
is tagged and the claim the tag names stands for its original; a wrong target when one is tagged and none of those
tags names a claim standing for the original; a miss when claims stand for it and none is tagged; unlisted when no
claim stands for it. Apart from those, it counts the tagged claims that stand for no labelled repeat and for at least
one labelled claim (spurious), the tag whose harm is lending a claim someone else's verdict.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from threetears.evals.kernel.claims.extraction import ExtractedClaim
from threetears.evals.kernel.claims.verbatim import verbatim_span
from threetears.evals.schema import Segment

CLAIMS: Final = Path(__file__).resolve().parent / "fixtures" / "claim_corpus" / "claimset" / "claims.jsonl"

#: What a cell's segment puts between two labelled stretches of one line.
JOIN: Final = " | "


@dataclass(frozen=True)
class LabelledClaim:
    """One corpus claim, placed in its cell.

    :ivar claim_id: The corpus's id.
    :ivar claim: The proposition as labelled.
    :ivar excerpt: The stretch it was said in.
    :ivar segment: The cell segment its line became.
    :ivar start: Where the stretch starts in that segment's text.
    :ivar end: Where it ends (exclusive).
    :ivar repeat_of: The id of the earlier claim it says again, or None.
    """

    claim_id: str
    claim: str
    excerpt: str
    segment: int
    start: int
    end: int
    repeat_of: str | None


@dataclass(frozen=True)
class CorpusCell:
    """One corpus result as a cell: its segments, and its labelled claims in corpus order.

    :ivar result_id: The corpus's result id.
    :ivar segments: One per line, each its own unit.
    :ivar labelled: The result's claims.
    """

    result_id: str
    segments: tuple[Segment, ...]
    labelled: tuple[LabelledClaim, ...]

    def by_id(self, claim_id: str) -> LabelledClaim:
        """The labelled claim of that id.

        :raises KeyError: The cell has none.
        """
        for claim in self.labelled:
            if claim.claim_id == claim_id:
                return claim
        raise KeyError(f"cell {self.result_id} labels no claim {claim_id}")


@dataclass(frozen=True)
class RestatesScore:
    """How a listing's ``restates`` tags stood against the labelled repeats, as counts.

    :ivar hits: Labelled repeats tagged with a target standing for the labelled original.
    :ivar wrong_targets: Labelled repeats tagged, with no tag naming a claim standing for the original.
    :ivar misses: Labelled repeats listed and left untagged.
    :ivar unlisted: Labelled repeats no listed claim stands for.
    :ivar spurious: Tagged claims standing for labelled claims, none of them a repeat.
    """

    hits: int
    wrong_targets: int
    misses: int
    unlisted: int
    spurious: int


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_cells(path: Path = CLAIMS) -> tuple[CorpusCell, ...]:
    """Every result of the corpus at ``path`` as a cell, in the order its results first appear.

    :param path: A claims file of the corpus's shape.
    :return: The cells.
    :raises ValueError: A ``repeat_of`` names a claim of another result, or one said no earlier.
    """
    by_result: dict[str, list[dict[str, Any]]] = {}
    for row in _rows(path):
        by_result.setdefault(row["result_id"], []).append(row)
    cells: list[CorpusCell] = []
    for result_id, rows in by_result.items():
        lines: dict[str, str] = {}
        for row in rows:
            text = lines.get(row["utterance_ref"])
            if text is None:
                lines[row["utterance_ref"]] = row["utterance_excerpt"]
            elif verbatim_span(row["utterance_excerpt"], text) is None:
                lines[row["utterance_ref"]] = text + JOIN + row["utterance_excerpt"]
        refs = list(lines)
        segments = tuple(Segment(text=lines[ref], unit=at) for at, ref in enumerate(refs))
        labelled: list[LabelledClaim] = []
        for row in rows:
            at = refs.index(row["utterance_ref"])
            span = verbatim_span(row["utterance_excerpt"], lines[row["utterance_ref"]])
            if span is None:  # pragma: no cover - every stretch was added unless the text already held it
                raise AssertionError(f"{row['claim_id']}'s stretch is not in the line built from it")
            start, end = span
            labelled.append(
                LabelledClaim(
                    claim_id=row["claim_id"],
                    claim=row["claim"],
                    excerpt=lines[row["utterance_ref"]][start:end],
                    segment=at,
                    start=start,
                    end=end,
                    repeat_of=row["repeat_of"],
                )
            )
        cell = CorpusCell(result_id=result_id, segments=segments, labelled=tuple(labelled))
        for claim in cell.labelled:
            if claim.repeat_of is None:
                continue
            original = cell.by_id(claim.repeat_of)
            if (original.segment, original.start) >= (claim.segment, claim.start):
                raise ValueError(f"{claim.claim_id} repeats {original.claim_id}, which is said no earlier")
        cells.append(cell)
    return tuple(cells)


def _stands_for(listed: ExtractedClaim, labelled: LabelledClaim) -> bool:
    return listed.segment == labelled.segment and listed.start < labelled.end and labelled.start < listed.end


def score_restates(scored: Sequence[tuple[CorpusCell, Sequence[ExtractedClaim]]]) -> RestatesScore:
    """Score each cell's listed claims' ``restates`` against its labelled repeats.

    :param scored: Each cell, with the claims an extractor listed for its segments.
    :return: The counts over every cell.
    """
    hits = wrong = misses = unlisted = spurious = 0
    for cell, listed in scored:
        repeats = [claim for claim in cell.labelled if claim.repeat_of is not None]
        for repeat in repeats:
            original = cell.by_id(repeat.repeat_of or "")
            standing = [claim for claim in listed if _stands_for(claim, repeat)]
            targets = [listed[claim.restates] for claim in standing if claim.restates is not None]
            if not standing:
                unlisted += 1
            elif not targets:
                misses += 1
            elif any(_stands_for(target, original) for target in targets):
                hits += 1
            else:
                wrong += 1
        spurious += sum(
            1
            for claim in listed
            if claim.restates is not None
            and any(_stands_for(claim, labelled) for labelled in cell.labelled)
            and not any(_stands_for(claim, repeat) for repeat in repeats)
        )
    return RestatesScore(hits=hits, wrong_targets=wrong, misses=misses, unlisted=unlisted, spurious=spurious)
