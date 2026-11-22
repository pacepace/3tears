"""One claim criterion's grade of one cell: its steps run in order, its fingerprint, its price, and its tally.

The judge phase calls :func:`grade_criterion` once per criterion the template declares, after the rubric dimensions,
each under its own hold against the run's cost cap. A criterion lists the claims the cell's segments state
(:class:`~threetears.evals.kernel.claims.extraction.ClaimExtractor`), checks each against the case's evidence and,
where it names ``grounded_in``, against the documents the candidate itself was given
(:class:`~threetears.evals.kernel.claims.verification.ClaimVerifier`), and keeps what it found as a
:class:`~threetears.evals.schema.models.ClaimGrade` with the counts derived from it
(:func:`~threetears.evals.kernel.claims.readings.claim_tally`).

**Every way a criterion fails is its own loss**, raised as
:class:`~threetears.evals.kernel.claims.failures.ClaimGradingFailed` with the loss the cell records: a step's reply
cut short, unreadable or past its repairs, and a grade whose payload would not fit the trace
(``evidence_over_cap``). A lost ground step is not a failure here: it rides on the tally, and truth stands.

**The web check** runs last, for a criterion with a web budget, over the claims the truth step nominated
(:class:`~threetears.evals.kernel.claims.web.WebChecker`), on the caller's search. A claim it did not check (past the
cell's budget, refused by the cost cap, or cut by the cell's deadline) reads ``skipped_cap`` and ``unverifiable``; a
criterion with no web budget checks nothing, and a nominated claim reads ``unverifiable`` with no web state.

Engine-internal: the judge phase calls it, and nothing here is exported.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Final

from threetears.evals.kernel.claims.extraction import ClaimExtractor
from threetears.evals.kernel.claims.failures import ClaimGradingFailed
from threetears.evals.kernel.claims.readings import claim_tally
from threetears.evals.kernel.claims.verification import (
    ClaimVerifier,
    Verification,
    WebGraded,
    graded_claims,
    truth_verdict,
)
from threetears.evals.kernel.claims.web import Search, WebCheck, WebChecker, web_definition
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.schema.completion import CompletionClient
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    CLAIM_TRACE_BYTES,
    ClaimCall,
    ClaimCriterion,
    ClaimEvidence,
    ClaimGrade,
    ClaimTally,
)
from threetears.evals.schema.reference import ReferenceKey

#: The revision of how a criterion grades, in its fingerprint: raised with any change to the order of the steps, to
#: what each is handed, or to how a grade and its tally are built from them.
CLAIM_GRADE_REVISION: Final = "claim grade v1"


@dataclass(frozen=True)
class GradedCriterion:
    """What one criterion found in one cell: the grade the trace stores, and the tally the result stores.

    :ivar grade: The grade.
    :ivar tally: Its counts, with a lost ground step recorded on them.
    :ivar ground_failure: What failed in the ground step when it was lost, named; None when it was not.
    """

    grade: ClaimGrade
    tally: ClaimTally
    ground_failure: str | None = None


def criterion_steps(criterion: ClaimCriterion, client: CompletionClient) -> tuple[ClaimExtractor, ClaimVerifier]:
    """The extraction and the verify step a criterion runs, on ``client``.

    :param criterion: The criterion.
    :param client: The judge's client, which names its model.
    :return: Its extractor and its verifier.
    :raises ValueError: The client names no model.
    """
    extractor = ClaimExtractor(client, guidance=criterion.guidance, repairs=criterion.repairs)
    verifier = ClaimVerifier(
        client,
        pages=criterion.pages,
        repairs=criterion.repairs,
        grounded_in=criterion.grounded_in,
        guidance=criterion.guidance,
    )
    return extractor, verifier


def criterion_fingerprint(criterion: ClaimCriterion, extractor: ClaimExtractor, verifier: ClaimVerifier) -> str:
    """What a criterion grades under: every field it declares, and every prompt, schema and rule its steps follow.

    The model is not in it: which model answered is recorded on each of the grade's calls, as a rubric dimension's
    built-in identity leaves it out. Two grades of one fingerprint were asked the same questions in the same way.

    :param criterion: The criterion.
    :param extractor: Its extractor (:func:`criterion_steps`).
    :param verifier: Its verifier.
    :return: A sha256 hex digest.
    """
    definition: dict[str, object] = {
        "revision": CLAIM_GRADE_REVISION,
        "criterion": criterion.model_dump(mode="json"),
        "extraction": extractor.definition,
        "verification": verifier.definition,
    }
    if criterion.web is not None:
        # Only where it is asked, so a criterion with no web budget keeps the fingerprint it had before the web check.
        definition["web"] = web_definition(criterion.web, criterion.repairs, criterion.guidance)
    return canonical_digest(definition)


def web_checker(criterion: ClaimCriterion, client: CompletionClient) -> WebChecker | None:
    """The web step a criterion runs, on ``client``; None for a criterion with no web budget.

    :param criterion: The criterion.
    :param client: The judge's client.
    :return: Its web checker, or None.
    """
    if criterion.web is None:
        return None
    return WebChecker(client, budget=criterion.web, repairs=criterion.repairs, guidance=criterion.guidance)


def claim_evidence_digest(evidence: ClaimEvidence) -> str:
    """The digest a grade names the evidence it graded by.

    :param evidence: The cell's claim evidence.
    :return: A sha256 hex digest of every field of it.
    """
    return canonical_digest(evidence.model_dump(mode="json"))


def claim_payload_bytes(evidence: ClaimEvidence, grades: Mapping[str, ClaimGrade]) -> int:
    """How many bytes a trace's claims payload takes, serialized as the trace's own cap counts it.

    :param evidence: The cell's claim evidence.
    :param grades: Every criterion's grade the trace would hold.
    :return: The bytes, as UTF-8 JSON.
    """
    payload = {
        "claim_evidence": evidence.model_dump(mode="json", exclude_none=False),
        "claim_grades": {name: grade.model_dump(mode="json", exclude_none=False) for name, grade in grades.items()},
    }
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def criterion_ceiling(
    criterion: ClaimCriterion,
    evidence: ClaimEvidence,
    *,
    key: ReferenceKey | None,
    client: CompletionClient,
    query_usd: float | None = None,
) -> float | None:
    """The most grading one cell under ``criterion`` can cost, priced before any call is made.

    The extraction's, each verify step's and, for a criterion with a web budget, the web step's ceilings, each with
    its own repair budget (:meth:`ClaimExtractor.price_ceiling`, :meth:`ClaimVerifier.price_ceiling`,
    :meth:`~threetears.evals.kernel.claims.web.WebChecker.price_ceiling`).

    :param criterion: The criterion.
    :param evidence: The cell's claim evidence.
    :param key: The case's reference key, or None.
    :param client: The judge's client, its prices brought up to date.
    :param query_usd: What one web search costs, at most; None when it cannot be said, or no search is asked.
    :return: The ceiling in dollars, or None when the client cannot price a call, or a criterion with a web budget
        cannot price a search.
    """
    extractor, verifier = criterion_steps(criterion, client)
    extraction = extractor.price_ceiling(evidence.segments)
    verification = verifier.price_ceiling(evidence.segments, key=key, documents=evidence.documents)
    checker = web_checker(criterion, client)
    web = 0.0 if checker is None else checker.price_ceiling(evidence.segments, query_usd=query_usd)
    if extraction is None or verification is None or web is None:
        return None
    return extraction + verification + web


async def grade_criterion(
    criterion: ClaimCriterion,
    evidence: ClaimEvidence,
    *,
    key: ReferenceKey | None,
    client: CompletionClient,
    graded: Mapping[str, ClaimGrade],
    truth_kept: Callable[[GradedCriterion], None] | None = None,
    search: Search | None = None,
    web_replayed: bool = False,
) -> GradedCriterion:
    """Grade one cell under one criterion: list its claims, verify them, check the nominated ones on the web where it
    declares a budget, and keep the grade with its tally.

    What the client raises in the extraction, a truth call or a web call (a provider failure, a replay's miss, a
    completion that breaks the protocol), and what ``search`` raises but a cost-cap refusal, is not caught: it is the
    judge phase's to classify.

    :param criterion: The criterion.
    :param evidence: The cell's claim evidence.
    :param key: The case's reference key, whose facts are case evidence; None when the case has none.
    :param client: The judge's client.
    :param graded: The grades the cell's trace already holds from the criteria graded before this one, which the
        trace's cap counts beside this one.
    :param truth_kept: Told what the cell keeps if its deadline strikes in a later step: before a ground step starts,
        the grade and tally its truth step leaves, its ground step lost to the deadline (``ground_loss``
        ``deadline``), since the ground step's length grows with what the arm fetched; and before a web step starts,
        the grade and tally the verify step leaves, every nominated claim unchecked (``skipped_cap``). In both, every
        claim the criterion would check on the web reads ``skipped_cap``. Not told when that grade would not fit the
        trace.
    :param search: The host's search, billed and bounded by the caller; required for a criterion with a web budget
        that nominates a claim.
    :param web_replayed: Whether ``search`` serves recorded results, which the grade records
        (:attr:`~threetears.evals.schema.models.ClaimGrade.web_replayed`) for a grade that checked any claim.
    :return: The grade and its tally.
    :raises ClaimGradingFailed: A step failed (``cut_short``, ``unreadable``, ``repair_budget_spent``, or
        ``cost_cap`` from a client its hold refuses outside the web step), or the grade would take the trace past
        :data:`~threetears.evals.schema.models.CLAIM_TRACE_BYTES` (``evidence_over_cap``).
    :raises ValueError: A criterion with a web budget nominated a claim, and no search was given.
    """
    extractor, verifier = criterion_steps(criterion, client)
    checker = web_checker(criterion, client)
    # Each step folds its completions into a ledger it requires; the judge phase bills the calls from the client.
    ledger = RoleUsageLedger(role="judge")
    calls: list[ClaimCall] = []
    claims = await extractor.extract(evidence.segments, ledger=ledger, calls=calls)
    fingerprint = criterion_fingerprint(criterion, extractor, verifier)

    def nominated(verification: Verification) -> list[int]:
        """The claims the truth step nominated for the web check, each one that restates nothing, in claim order."""
        if checker is None:
            return []
        return [
            position
            for position, (claim, finding) in enumerate(zip(claims, verification.findings, strict=True))
            if finding is not None
            and claim.restates is None
            and truth_verdict(finding.truth_citations, suspect=finding.suspect).nominated_by is not None
        ]

    def unchecked(verification: Verification) -> WebCheck:
        """The web step as a cell keeps it when it never ran: every nominated claim unchecked."""
        return WebCheck(calls=(), records=(), outcomes=dict.fromkeys(nominated(verification), WebGraded("skipped_cap")))

    def kept(verification: Verification, web: WebCheck) -> GradedCriterion:
        grade = ClaimGrade(
            criterion_fingerprint=fingerprint,
            claim_evidence_digest=claim_evidence_digest(evidence),
            calls=(*calls, *verification.calls, *web.calls),
            evidence=verification.evidence,
            documents_truncated=verification.documents_truncated,
            claims=graded_claims(claims, verification, calls_before=len(calls), web=web.outcomes),
            web=web.records,
            web_replayed=web_replayed and bool(web.records),
        )
        held = claim_payload_bytes(evidence, {**graded, criterion.name: grade})
        if held > CLAIM_TRACE_BYTES:
            raise ClaimGradingFailed(
                f"claim criterion {criterion.name!r}'s grade would take the trace's claims payload to {held} bytes, "
                f"and CLAIM_TRACE_BYTES allows {CLAIM_TRACE_BYTES}",
                loss="evidence_over_cap",
            )
        tally = claim_tally(grade, evidence, grounded_in=criterion.grounded_in, ground_loss=verification.ground_loss)
        return GradedCriterion(grade=grade, tally=tally, ground_failure=verification.ground_failure)

    def fallback(verification: Verification) -> None:
        """Tell ``truth_kept`` what the cell keeps if its deadline strikes in the next step."""
        if truth_kept is None:
            return
        try:
            truth_kept(kept(verification, unchecked(verification)))
        except ClaimGradingFailed:
            # NOSILENT: a fallback grade that outgrows the trace is no fallback for a deadline; the full grade meets
            # the same cap, which records the criterion's loss (evidence_over_cap) where it is decided.
            return

    def truth_done(truth: Verification) -> None:
        fallback(
            replace(truth, ground_loss="deadline", ground_failure="the cell's deadline struck during the ground step")
        )

    verification = await verifier.verify(
        claims, key=key, documents=evidence.documents, ledger=ledger, truth_done=truth_done
    )
    to_check = nominated(verification)
    if checker is None or not to_check:
        return kept(verification, unchecked(verification))
    if search is None:
        raise ValueError(
            f"claim criterion {criterion.name!r} declares a web budget and nominated claims {to_check}, and no web "
            "search was given to check them"
        )
    fallback(verification)
    web = await checker.check(
        claims,
        to_check,
        search=search,
        ledger=ledger,
        calls_before=len(calls) + len(verification.calls),
    )
    return kept(verification, web)


__all__: list[str] = []
