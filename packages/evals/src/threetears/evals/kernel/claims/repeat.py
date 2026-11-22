"""Repeating a stored claim grade: the instrument's agreement with itself, or a second judge's with it.

A claim grade is a measurement of a cell, and the instrument behind it is checked two ways besides against people:
asked again by the same judge (its agreement with itself), and asked by another judge (whether two judges read the
same claims alike). Both re-read **what the first grade read**, so what is measured is the reading, never the
listing or the searching:

- **The extraction is fixed.** The claims are the stored grade's, as listed; only their verdicts are asked again.
- **Truth and groundedness are verified again** over the same case evidence and documents, by the criterion's own
  verify step, under the same definition (the criterion's fingerprint must be the grade's).
- **The web verdict is asked again over the windows the grade stored, and nothing is searched**
  (:meth:`~threetears.evals.kernel.claims.web.WebChecker.reread`). A claim the repeat doubts that the first grade
  never checked has no stored window to read, so it reads unchecked (``skipped_cap``) and unverifiable, never
  searched; one the first grade checked and the repeat no longer doubts is decided by its truth step, as any claim
  is. So the web step is measured wherever both readings reached it.

**A replayed grade is not repeated** (:func:`replayed_state`): its web results came from a recording, so no live
search of its run stands behind it, and every check over it reads not available (``replayed``).

:func:`repeat_agreement` pairs a first grade's claims with a repeat's, per question: the pass/fail pairs on truth (the
claim read false or not) and on groundedness, and the web outcomes of the claims both readings decided on the web.

:func:`repeat_ceiling` prices a repeat before it is made, at its worst case under the criterion's budgets.

Engine-internal: nothing here is exported.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from threetears.evals.kernel.claims.extraction import ExtractedClaim
from threetears.evals.kernel.claims.grading import (
    claim_evidence_digest,
    criterion_fingerprint,
    criterion_steps,
    web_checker,
)
from threetears.evals.kernel.claims.verification import WebGraded, graded_claims, truth_verdict
from threetears.evals.kernel.claims.web import WebCheck
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.schema.completion import CompletionClient
from threetears.evals.schema.models import ClaimCriterion, ClaimEvidence, ClaimGrade, GradedClaim
from threetears.evals.schema.reading_states import StateOf
from threetears.evals.schema.reference import ReferenceKey

_REPLAYED = StateOf(state="not_available", reason="replayed")


def replayed_state(grade: ClaimGrade) -> StateOf | None:
    """The state every check over ``grade`` is in: not available (``replayed``) when a recording served its web
    results, so no live search of its run stands behind it; None (counted) otherwise.

    :param grade: The stored grade.
    :return: The state, or None.
    """
    return _REPLAYED if grade.web_replayed else None


def repeat_ceiling(
    criterion: ClaimCriterion, evidence: ClaimEvidence, *, key: ReferenceKey | None, client: CompletionClient
) -> float | None:
    """The most repeating one stored grade under ``criterion`` can cost on ``client``, priced before any call.

    A repeat's prompts are not known before it is made: a repair asks again with what the reply got wrong. So it is
    priced at its worst case under the criterion's own budgets: every verify step asked in full, each with its repair
    budget (:meth:`~threetears.evals.kernel.claims.verification.ClaimVerifier.price_ceiling`), and, for a criterion
    with a web budget, the web step with every search free, since a repeat searches nothing and reads only the
    windows the grade stored (:meth:`~threetears.evals.kernel.claims.web.WebChecker.price_ceiling`). The extraction
    is not priced: the repeat keeps the grade's claims as listed.

    :param criterion: The criterion, as the template declares it.
    :param evidence: The cell's claim evidence, as the trace stores it.
    :param key: The case's reference key, or None.
    :param client: The client the repeat is asked on, its prices brought up to date.
    :return: The ceiling in dollars, or None when the client cannot price a call.
    """
    _, verifier = criterion_steps(criterion, client)
    verification = verifier.price_ceiling(evidence.segments, key=key, documents=evidence.documents)
    checker = web_checker(criterion, client)
    web = 0.0 if checker is None else checker.price_ceiling(evidence.segments, query_usd=0.0)
    if verification is None or web is None:
        return None
    return verification + web


@dataclass(frozen=True)
class ClaimRepeat:
    """A stored grade read again.

    :ivar grade: The repeat's grade: the first grade's claims, verdicts asked again; None where nothing was asked
        (the grade was replayed, or holds no world claim).
    :ivar state: Not available (``replayed``) for a replayed grade; None otherwise.
    :ivar reread: The claims whose web verdict was asked again over the first grade's stored windows, in claim order.
    :ivar unsearched: The claims the repeat doubted that the first grade never checked on the web: no window to read,
        nothing searched, so they read unchecked.
    """

    grade: ClaimGrade | None
    state: StateOf | None = None
    reread: tuple[int, ...] = ()
    unsearched: tuple[int, ...] = ()


async def repeat_grade(
    criterion: ClaimCriterion,
    evidence: ClaimEvidence,
    grade: ClaimGrade,
    *,
    key: ReferenceKey | None,
    client: CompletionClient,
) -> ClaimRepeat:
    """Read a stored grade again on ``client``: the same claims, verified again, the web verdicts asked again over
    the stored windows, and nothing searched.

    The same judge's client gives the instrument's agreement with itself; another judge's client, a second judge's
    reading of the same claims and windows. What the client raises (a provider failure, a replay's miss, a protocol
    defect) is not caught, as a truth call's is not: it is the caller's to classify. A failure the verify step makes
    its own (a ground step lost) or the web step makes its own (``web_loss``) is the repeat grade's, as it would be the
    first's.

    :param criterion: The criterion, as the template declares it.
    :param evidence: The cell's claim evidence, as the trace stores it.
    :param grade: The stored grade.
    :param key: The case's reference key, or None.
    :param client: The judge's client the repeat is asked on.
    :return: The repeat.
    :raises ValueError: The criterion grades under another fingerprint than the grade's, or the grade does not grade
        ``evidence``.
    :raises ClaimGradingFailed: A truth reply was cut short, unreadable or past its repairs.
    """
    if (state := replayed_state(grade)) is not None:
        return ClaimRepeat(grade=None, state=state)
    extractor, verifier = criterion_steps(criterion, client)
    fingerprint = criterion_fingerprint(criterion, extractor, verifier)
    if fingerprint != grade.criterion_fingerprint:
        raise ValueError(
            f"claim criterion {criterion.name!r} grades under fingerprint {fingerprint[:12]}, and the grade was made "
            f"under {grade.criterion_fingerprint[:12]}: a repeat under another definition measures another instrument"
        )
    if grade.claim_evidence_digest != claim_evidence_digest(evidence):
        raise ValueError(
            f"claim criterion {criterion.name!r}'s grade names claim evidence {grade.claim_evidence_digest[:12]}, and "
            f"the trace stores {claim_evidence_digest(evidence)[:12]}"
        )
    # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
    if not any(claim.scope == "world" for claim in grade.claims):
        return ClaimRepeat(grade=None)
    fields = set(ExtractedClaim.model_fields)
    claims = tuple(ExtractedClaim.model_validate(claim.model_dump(include=fields)) for claim in grade.claims)
    ledger = RoleUsageLedger(role="judge")
    verification = await verifier.verify(claims, key=key, documents=evidence.documents, ledger=ledger)
    checker = web_checker(criterion, client)
    # With no web budget nothing is nominated for the web, as in the first grade: a doubted claim reads unverifiable.
    nominated = [
        position
        for position, (claim, finding) in enumerate(zip(claims, verification.findings, strict=True))
        if checker is not None
        and finding is not None
        and claim.restates is None
        and truth_verdict(finding.truth_citations, suspect=finding.suspect).nominated_by is not None
    ]
    stored = {record.claim: record for record in grade.web}
    reread = tuple(position for position in nominated if position in stored)
    unsearched = tuple(position for position in nominated if position not in stored)
    web = WebCheck(calls=(), records=(), outcomes={})
    if checker is not None and reread:
        web = await checker.reread(
            claims, [stored[position] for position in reread], ledger=ledger, calls_before=len(verification.calls)
        )
    outcomes: dict[int, WebGraded] = {position: WebGraded("skipped_cap") for position in unsearched}
    outcomes.update(web.outcomes)
    again = ClaimGrade(
        criterion_fingerprint=grade.criterion_fingerprint,
        claim_evidence_digest=grade.claim_evidence_digest,
        calls=(*verification.calls, *web.calls),
        evidence=verification.evidence,
        documents_truncated=verification.documents_truncated,
        claims=graded_claims(claims, verification, calls_before=0, web=outcomes),
        web=web.records,
        ground_loss=verification.ground_loss,
        web_loss=web.loss,
    )
    return ClaimRepeat(grade=again, reread=reread, unsearched=unsearched)


@dataclass(frozen=True)
class RepeatAgreement:
    """How a repeat's claims stood against the first grade's, as counts; nothing derived is kept.

    :ivar truth: (first fails, repeat fails) -> claims, over world claims that restate nothing: a claim fails when it
        reads ``contradicted``.
    :ivar grounded: The same on groundedness, over claims both readings' ground steps read.
    :ivar web: (first outcome, repeat outcome) -> claims, over claims both readings decided on the web.
    :ivar state: Not available (``replayed``) where either grade was replayed, and nothing is counted.
    """

    truth: Mapping[tuple[int, int], int]
    grounded: Mapping[tuple[int, int], int]
    web: Mapping[tuple[str, str], int]
    state: StateOf | None = None


def repeat_agreement(first: ClaimGrade, repeat: ClaimRepeat) -> RepeatAgreement:
    """Pair a first grade's claims with its repeat's, per question.

    :param first: The stored grade.
    :param repeat: Its repeat (:func:`repeat_grade`).
    :return: The counts; empty, with the repeat's state, where nothing was repeated.
    :raises ValueError: The repeat lists other claims than the first grade.
    """
    state = replayed_state(first) or repeat.state
    if state is not None or repeat.grade is None:
        return RepeatAgreement(truth={}, grounded={}, web={}, state=state)
    again = repeat.grade
    if [(c.segment, c.start, c.end, c.claim) for c in again.claims] != [
        (c.segment, c.start, c.end, c.claim) for c in first.claims
    ]:
        raise ValueError("a repeat reads the first grade's claims, as listed, and this one lists others")
    truth: Counter[tuple[int, int]] = Counter()
    grounded: Counter[tuple[int, int]] = Counter()
    web: Counter[tuple[str, str]] = Counter()
    first_web = {record.claim: record.outcome for record in first.web}
    again_web = {record.claim: record.outcome for record in again.web}
    for place, (a, b) in enumerate(zip(first.claims, again.claims, strict=True)):
        # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
        if a.scope != "world" or a.restates is not None:
            continue
        truth[(_fails(a), _fails(b))] += 1
        if a.grounded is not None and b.grounded is not None:
            grounded[(int(not a.grounded), int(not b.grounded))] += 1
        if place in first_web and place in again_web and b.web == "checked":
            web[(first_web[place], again_web[place])] += 1
    return RepeatAgreement(truth=dict(truth), grounded=dict(grounded), web=dict(web))


def _fails(claim: GradedClaim) -> int:
    return int(claim.verdict == "contradicted")


__all__: list[str] = []
