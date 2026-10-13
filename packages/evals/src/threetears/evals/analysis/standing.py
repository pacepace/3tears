"""What the headline and each summary bullet stand on: the weakest of the findings they rest on, derived on read.

Most readers read only the headline and the summary, and neither carried a confidence: a finding is shown at its
tier, while the sentence summarizing it was shown at none, so a headline over a medium finding read as settled. The
writer now names the findings each rests on (``headline_rests_on`` and ``summary_rests_on`` on
:class:`~threetears.evals.kernel.authored.AuthoredAnalysis`), and code derives what each can bear, as it derives a
finding's evidence tier from the readings it cites (:func:`~threetears.evals.kernel.campaign.evidence_tier_of`):

- **confidence** — the weakest confidence among those findings, so a headline is never shown surer than the least
  sure finding under it;
- **evidence tier** — the weakest tier of every reading those findings cite.

Derived on every read, never stored: the links are the writer's and are stored as written, and the tiers follow
from them and the resolutions beside them. The generator refuses an analysis that leaves the links empty over
findings. A sentence resting on no finding — every one, over an analysis with none — reads ``None`` here: "not
recorded", which every surface says in words, never a tier guessed from the findings.
"""

from __future__ import annotations

from pydantic import Field

from threetears.evals.kernel.authored import Confidence, summary_bullets
from threetears.evals.kernel.campaign import EvalAnalysis, EvidenceTier, evidence_tier_of
from threetears.evals.schema.base import EvalBaseModel

#: Confidence tiers, weakest first.
_CONFIDENCE_ORDER: tuple[Confidence, ...] = ("low", "medium", "high", "very_high")


class Standing(EvalBaseModel):
    """What one headline or summary bullet stands on — derived, never stored."""

    rests_on: list[int] = Field(description="The finding positions it rests on, as the writer linked them.")
    confidence: Confidence | None = Field(
        description="The weakest confidence among those findings; None when it rests on no finding."
    )
    evidence_tier: EvidenceTier | None = Field(
        description=(
            "The weakest evidence tier of every reading those findings cite (`none` when they cite none); None for an "
            "analysis stored with no resolutions, whose findings' tiers were never read."
        )
    )


def standing_of(analysis: EvalAnalysis, positions: list[int]) -> Standing:
    """What a sentence resting on ``positions`` stands on: the weakest of those findings.

    Args:
        analysis: The analysis the findings are in; its positions were checked on construction.
        positions: Finding positions.

    Returns:
        The standing.
    """
    findings = [analysis.document.findings[position] for position in positions]
    confidence = min((f.confidence for f in findings), key=_CONFIDENCE_ORDER.index) if findings else None
    tier: EvidenceTier | None = None
    if analysis.resolutions:
        tier = evidence_tier_of(
            [row.judged_tier for position in positions for row in analysis.resolutions[position].evidence]
        )
    return Standing(rests_on=list(positions), confidence=confidence, evidence_tier=tier)


def headline_standing(analysis: EvalAnalysis) -> Standing:
    """What the headline stands on.

    Args:
        analysis: The analysis.

    Returns:
        The standing.
    """
    return standing_of(analysis, list(analysis.document.headline_rests_on))


def summary_standings(analysis: EvalAnalysis) -> list[tuple[str, Standing]]:
    """Each summary bullet with what it stands on.

    Args:
        analysis: The analysis.

    Returns:
        ``(bullet, standing)`` per bullet (:func:`~threetears.evals.kernel.authored.summary_bullets`), in order. Over
        no findings the links are empty and each bullet stands on nothing.
    """
    document = analysis.document
    bullets = summary_bullets(document.summary)
    links = document.summary_rests_on if document.findings else [[] for _ in bullets]
    if len(links) != len(bullets):
        raise ValueError(
            f"analysis {analysis.id!r} links {len(links)} summary bullet(s) and its summary has {len(bullets)}; the "
            "generator refuses that before it is stored, so this analysis was not written by it"
        )
    return [(bullet, standing_of(analysis, list(positions))) for bullet, positions in zip(bullets, links, strict=True)]


__all__ = ["Standing", "headline_standing", "standing_of", "summary_standings"]
