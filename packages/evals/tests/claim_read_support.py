"""Graded cells built by hand, for claim reads: each claim a world claim with the verdict asked for, and the case
passage, the web record or nothing that verdict needs, so a grade is valid as a stored one is.

The text is invented: a reading room's almanac and its shelf of pamphlets. Read by ``test_claim_read.py``,
``test_simulated_claim_read.py`` and ``claim_read_nulls.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis.claim_read import GradedCell
from threetears.evals.kernel.claims.grading import claim_evidence_digest
from threetears.evals.schema import Segment
from threetears.evals.schema.models import (
    ClaimCall,
    ClaimCitation,
    ClaimEvidence,
    ClaimGrade,
    EvidenceDocument,
    EvidenceRef,
    GradedClaim,
    TextSpan,
    WebCheckRecord,
    WebWindow,
)

__all__ = ["CRITERION", "FINGERPRINT", "graded_cell"]

#: The criterion every built grade is graded under, and its fingerprint.
CRITERION = "almanac.claims"
FINGERPRINT = "a" * 64

_CATALOGUE = "The catalogue lists each pamphlet of the shelf by its number and its year."


def _sentence(cell: str, at: int) -> str:
    return f"Pamphlet {at} of shelf {cell} was printed in {1900 + at}."


def graded_cell(
    run_id: str,
    result_id: str,
    verdicts: Sequence[str],
    *,
    case_id: str = "case-1",
    grounded: Sequence[bool | None] | None = None,
    restates: Sequence[int | None] | None = None,
    replayed: bool = False,
    fingerprint: str = FINGERPRINT,
) -> GradedCell:
    """A graded cell of one segment per claim, each a world claim reading the verdict given.

    :param run_id: Its run.
    :param result_id: Its result.
    :param verdicts: Each claim's truth verdict, in order.
    :param case_id: Its case.
    :param grounded: Each claim's groundedness, or None for no ground step.
    :param restates: Each claim's original, by place, or None.
    :param replayed: Whether its web results came from a recording.
    :param fingerprint: The criterion fingerprint the grade was made under.
    :return: The cell.
    """
    texts = [_sentence(result_id, at) for at in range(len(verdicts))]
    segments = tuple(Segment(text=text, unit=at) for at, text in enumerate(texts))
    catalogue = EvidenceDocument(id="catalogue", provenance="case", title="Catalogue", text=_CATALOGUE)
    shelf = EvidenceDocument(id="shelf", provenance="run", title="Shelf notes", text=" ".join(texts))
    evidence = ClaimEvidence(segmentation="sentences-v1", segments=segments, documents=(catalogue, shelf))
    refs = (
        EvidenceRef(ref="S1", provenance="case", document="catalogue", span=TextSpan(start=0, end=len(_CATALOGUE))),
        EvidenceRef(ref="S2", provenance="run", document="shelf", span=TextSpan(start=0, end=len(texts[0]))),
    )
    claims: list[GradedClaim] = []
    records: list[WebCheckRecord] = []
    for at, (text, verdict) in enumerate(zip(texts, verdicts, strict=True)):
        ground = None if grounded is None else grounded[at]
        original = None if restates is None else restates[at]
        fields: dict[str, object] = {
            "segment": at,
            "unit": at,
            "claim": text,
            "excerpt": text,
            "start": 0,
            "end": len(text),
            "scope": "world",
            "specific": True,
            "hedged": False,
            "attributed": False,
            "restates": original,
            "verdict": verdict,
            "decided_by": 0,
            "grounded": ground,
            "ground_citations": (ClaimCitation(ref="S2", stance="supports", quote=texts[0]),) if ground else (),
        }
        if verdict in ("supported", "contradicted"):
            stance = "supports" if verdict == "supported" else "contradicts"
            fields.update(
                graded_against="case",
                truth_citations=(ClaimCitation(ref="S1", stance=stance, quote=_CATALOGUE),),
            )
        elif verdict == "true_ungrounded":
            fields.update(graded_against="web", suspect=True, nominated_by="judge", web="checked")
            if original is None:
                window = WebWindow(url=f"https://example.org/{result_id}/{at}", text=text)
                records.append(
                    WebCheckRecord(
                        claim=at,
                        queries=(f"pamphlet {at} year",),
                        urls=(window.url,),
                        windows=(window,),
                        outcome="true",
                        deciding_url=window.url,
                        quote=text,
                        value_span=TextSpan(start=len(text) - 5, end=len(text) - 1),
                    )
                )
        claims.append(GradedClaim.model_validate(fields))
    grade = ClaimGrade(
        criterion_fingerprint=fingerprint,
        claim_evidence_digest=claim_evidence_digest(evidence),
        calls=(ClaimCall(role="truth", served_model="judge-a", judge_temperature=0.0),),
        evidence=refs,
        claims=tuple(claims),
        web=tuple(records),
        web_replayed=replayed and bool(records),
    )
    return GradedCell(run_id=run_id, result_id=result_id, case_id=case_id, grade=grade, evidence=evidence)
