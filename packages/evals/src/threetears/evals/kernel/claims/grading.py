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
(``evidence_over_cap``). :func:`grade_or_loss` turns each, and a provider's failure, into the loss the cell records,
for the judge phase and a re-judge alike. A lost ground step is not a failure here: it rides on the grade and its
tally (``ground_loss``), and truth stands, so a tally recomputed from a stored grade (:func:`grade_tally`) reads a lost
groundedness as exactly that.

**The web check is not built yet.** A criterion with a web budget is refused before any call
(:func:`refuse_unbuilt_claim_steps`), so a template stating one never runs as if it stated none; every claim a
criterion nominates reads ``unverifiable``, the verdict table's reading with no web outcome.

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
from threetears.evals.kernel.claims.verification import ClaimVerifier, Verification, graded_claims
from threetears.evals.kernel.host.apparatus import ApparatusError
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.schema.completion import CompletionClient, CompletionProtocolDefect
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.schema.models import (
    CLAIM_TRACE_BYTES,
    ClaimCall,
    ClaimCriterion,
    ClaimEvidence,
    ClaimGrade,
    ClaimTally,
    EvalTemplate,
    KnownClaimLoss,
)
from threetears.evals.schema.reference import ReferenceKey

log = get_logger(__name__)

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
    return canonical_digest(
        {
            "revision": CLAIM_GRADE_REVISION,
            "criterion": criterion.model_dump(mode="json"),
            "extraction": extractor.definition,
            "verification": verifier.definition,
        }
    )


def criterion_fingerprints(template: EvalTemplate, client: CompletionClient) -> dict[str, str]:
    """Each claim criterion's fingerprint (:func:`criterion_fingerprint`), by name, in the template's order.

    What a cell records for every criterion its claims phase is due (``EvalResult.claim_fingerprints``), graded or
    lost, so a re-judge of a lost one can tell whether it would still grade under the same definition.

    :param template: The template.
    :param client: The judge's client, which names its model; the fingerprint does not read it.
    :return: criterion -> fingerprint.
    :raises ValueError: The client names no model.
    """
    return {
        criterion.name: criterion_fingerprint(criterion, *criterion_steps(criterion, client))
        for criterion in template.claim_criteria
    }


def grade_tally(grade: ClaimGrade, evidence: ClaimEvidence, criterion: ClaimCriterion) -> ClaimTally:
    """The tally a stored grade gives, recomputed from it alone: what the cell's result stores for the criterion.

    The grade records its ground step's loss (:attr:`ClaimGrade.ground_loss`), so a recomputed tally loses
    groundedness exactly where the grading did, and never truth.

    :param grade: The criterion's grade of the cell, as the trace stores it.
    :param evidence: The claim evidence it graded, as the trace stores it.
    :param criterion: The criterion, whose ``grounded_in`` the counts read.
    :return: The tally.
    :raises ValueError: The grade does not grade ``evidence`` (another digest), or its claims do not give a tally
        (:func:`~threetears.evals.kernel.claims.readings.claim_tally`).
    """
    if grade.claim_evidence_digest != claim_evidence_digest(evidence):
        raise ValueError(
            f"claim criterion {criterion.name!r}'s grade names claim evidence {grade.claim_evidence_digest[:12]}, "
            f"and the trace stores {claim_evidence_digest(evidence)[:12]}"
        )
    return claim_tally(grade, evidence, grounded_in=criterion.grounded_in, ground_loss=grade.ground_loss)


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


def refuse_unbuilt_claim_steps(template: EvalTemplate) -> None:
    """Refuse a template whose claim criterion declares a step this build does not run.

    A criterion with a web budget asks for a web check, which this build does not have: run as declared, it would
    grade every nominated claim ``unverifiable`` under a fingerprint that says the web was asked.

    :param template: The template.
    :raises ValueError: A criterion declares a web budget.
    """
    if webbed := [criterion.name for criterion in template.claim_criteria if criterion.web is not None]:
        raise ValueError(
            f"claim criteria {', '.join(map(repr, webbed))} of template {template.name!r} declare a web budget, and "
            "this build has no web check; declare web None to grade against the case's evidence alone"
        )


def criterion_ceiling(
    criterion: ClaimCriterion, evidence: ClaimEvidence, *, key: ReferenceKey | None, client: CompletionClient
) -> float | None:
    """The most grading one cell under ``criterion`` can cost, priced before any call is made.

    The extraction's and each verify step's ceilings, each with its own repair budget
    (:meth:`ClaimExtractor.price_ceiling`, :meth:`ClaimVerifier.price_ceiling`). The web check adds nothing: no
    criterion this build runs declares one.

    :param criterion: The criterion.
    :param evidence: The cell's claim evidence.
    :param key: The case's reference key, or None.
    :param client: The judge's client, its prices brought up to date.
    :return: The ceiling in dollars, or None when the client cannot price a call.
    """
    extractor, verifier = criterion_steps(criterion, client)
    extraction = extractor.price_ceiling(evidence.segments)
    verification = verifier.price_ceiling(evidence.segments, key=key, documents=evidence.documents)
    if extraction is None or verification is None:
        return None
    return extraction + verification


async def grade_criterion(
    criterion: ClaimCriterion,
    evidence: ClaimEvidence,
    *,
    key: ReferenceKey | None,
    client: CompletionClient,
    graded: Mapping[str, ClaimGrade],
    truth_kept: Callable[[GradedCriterion], None] | None = None,
) -> GradedCriterion:
    """Grade one cell under one criterion: list its claims, verify them, and keep the grade with its tally.

    What the client raises in the extraction or a truth call (a provider failure, a replay's miss, a completion that
    breaks the protocol) is not caught: it is the judge phase's to classify.

    :param criterion: The criterion.
    :param evidence: The cell's claim evidence.
    :param key: The case's reference key, whose facts are case evidence; None when the case has none.
    :param client: The judge's client.
    :param graded: The grades the cell's trace already holds from the criteria graded before this one, which the
        trace's cap counts beside this one.
    :param truth_kept: Told, before a ground step starts, the grade and tally its truth step leaves, its ground step
        lost to the deadline (``ground_loss`` ``deadline``): what the cell keeps when a deadline strikes there, since
        the ground step's length grows with what the arm fetched. Not told when that grade would not fit the trace.
    :return: The grade and its tally.
    :raises ClaimGradingFailed: A step failed (``cut_short``, ``unreadable``, ``repair_budget_spent``, or
        ``cost_cap`` or ``hold_exceeded`` from a client its hold refuses), or the grade would take the trace past
        :data:`~threetears.evals.schema.models.CLAIM_TRACE_BYTES` (``evidence_over_cap``).
    """
    extractor, verifier = criterion_steps(criterion, client)
    # Each step folds its completions into a ledger it requires; the judge phase bills the calls from the client.
    ledger = RoleUsageLedger(role="judge")
    calls: list[ClaimCall] = []
    claims = await extractor.extract(evidence.segments, ledger=ledger, calls=calls)
    fingerprint = criterion_fingerprint(criterion, extractor, verifier)

    def kept(verification: Verification) -> GradedCriterion:
        grade = ClaimGrade(
            criterion_fingerprint=fingerprint,
            claim_evidence_digest=claim_evidence_digest(evidence),
            calls=(*calls, *verification.calls),
            evidence=verification.evidence,
            documents_truncated=verification.documents_truncated,
            claims=graded_claims(claims, verification, calls_before=len(calls)),
            ground_loss=verification.ground_loss,
        )
        held = claim_payload_bytes(evidence, {**graded, criterion.name: grade})
        if held > CLAIM_TRACE_BYTES:
            raise ClaimGradingFailed(
                f"claim criterion {criterion.name!r}'s grade would take the trace's claims payload to {held} bytes, "
                f"and CLAIM_TRACE_BYTES allows {CLAIM_TRACE_BYTES}",
                loss="evidence_over_cap",
            )
        tally = grade_tally(grade, evidence, criterion)
        return GradedCriterion(grade=grade, tally=tally, ground_failure=verification.ground_failure)

    def truth_done(truth: Verification) -> None:
        if truth_kept is None:
            return
        try:
            truth_kept(
                kept(
                    replace(
                        truth,
                        ground_loss="deadline",
                        ground_failure="the cell's deadline struck during the ground step",
                    )
                )
            )
        except ClaimGradingFailed:
            # NOSILENT: a grade of truth alone that outgrows the trace is no fallback for a deadline, so none is kept.
            # The criterion is recorded all the same: evidence_over_cap once the full grade meets the same cap, or
            # deadline when the deadline strikes in the ground step first.
            return

    verification = await verifier.verify(
        claims, key=key, documents=evidence.documents, ledger=ledger, truth_done=truth_done
    )
    return kept(verification)


async def grade_or_loss(
    criterion: ClaimCriterion,
    evidence: ClaimEvidence,
    *,
    key: ReferenceKey | None,
    client: CompletionClient,
    graded: Mapping[str, ClaimGrade],
    what: str,
    truth_kept: Callable[[GradedCriterion], None] | None = None,
) -> GradedCriterion | KnownClaimLoss:
    """Grade one cell under one criterion (:func:`grade_criterion`), or name the loss the cell records instead.

    Every way the criterion fails is its own loss, logged: a step's (:class:`ClaimGradingFailed`), and the provider
    failing a call (``provider_error``). Raised instead, failing what asked, since neither is an outcome of the cell:
    a completion that breaks the protocol in the extraction or a truth call, and a call the rig could not serve (a
    replay's miss).

    :param criterion: The criterion.
    :param evidence: The cell's claim evidence.
    :param key: The case's reference key, or None.
    :param client: The judge's client.
    :param graded: The grades the cell's trace already holds from criteria graded before this one.
    :param what: What is graded, for the log line a loss writes ("Eval run r: claim criterion c for case=k").
    :param truth_kept: See :func:`grade_criterion`.
    :return: The grade and its tally, or the loss.
    :raises CompletionProtocolDefect: A completion in the extraction or a truth call breaks the protocol.
    :raises ApparatusError: The rig could not serve a call.
    """
    try:
        return await grade_criterion(criterion, evidence, key=key, client=client, graded=graded, truth_kept=truth_kept)
    except ClaimGradingFailed as failed:
        # Recorded, never swallowed: the loss is the criterion's on the cell, which keeps every other reading.
        log.warning("%s lost (%s): %s", what, failed.loss, failed)
        return failed.loss
    except CompletionProtocolDefect, ApparatusError:
        raise
    except (
        Exception
    ) as e:  # prawduct:ok-broad-except — judge boundary; a provider's failure is this criterion's loss alone
        # NOSILENT: returned as provider_error, which the cell records as the criterion's loss.
        contain(e, log=log, what=f"{what} failed", level="error")
        return "provider_error"


__all__: list[str] = []
