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

**The web check** runs last, for a criterion with a web budget, over the claims the truth step nominated
(:class:`~threetears.evals.kernel.claims.web.WebChecker`), on the caller's search. A claim it did not check (past the
cell's budget, refused by the cost cap, or cut by the cell's deadline) reads ``skipped_cap`` and ``unverifiable``. A
web step that fails is its own loss (``web_loss``), as a ground step's is: the claims it left undecided read ``lost``
and ``unverifiable``, and truth stands. A criterion with no web budget checks nothing, and a nominated claim reads
``unverifiable`` with no web state.

Engine-internal: the judge phase calls it, and nothing here is exported.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from threetears.evals.kernel.claims.derived import ClaimDerivation, ClaimReaderDefect, derive_claim_readings
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
from threetears.evals.kernel.claims.web import (
    QUERY_UNIT,
    Search,
    WebCheck,
    WebChecker,
    query_price,
    web_definition,
)
from threetears.evals.kernel.usage_capture import ExternalRateTable
from threetears.evals.schema.web_search import WebSearch
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
    :ivar web_failure: What failed in the web step when it failed (its loss is the grade's ``web_loss``), named; None
        when it did not.
    """

    grade: ClaimGrade
    tally: ClaimTally
    ground_failure: str | None = None
    web_failure: str | None = None


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


def grade_tally(
    grade: ClaimGrade, evidence: ClaimEvidence, criterion: ClaimCriterion, *, derivation: ClaimDerivation | None
) -> ClaimTally:
    """The tally a stored grade gives, recomputed from it alone: what the cell's result stores for the criterion.

    The grade records its ground step's loss (:attr:`ClaimGrade.ground_loss`), so a recomputed tally loses
    groundedness exactly where the grading did, and never truth. The host's readings of the grade
    (:mod:`~threetears.evals.kernel.claims.derived`) are made here too, so no tally is ever built without them.

    :param grade: The criterion's grade of the cell, as the trace stores it.
    :param evidence: The claim evidence it graded, as the trace stores it.
    :param criterion: The criterion, whose ``grounded_in`` the counts read.
    :param derivation: The host's readings of the criterion (its reader and measures), or None for a criterion no
        measure derives from. Required, so every caller says which.
    :return: The tally.
    :raises ValueError: The grade does not grade ``evidence`` (another digest), its claims do not give a tally
        (:func:`~threetears.evals.kernel.claims.readings.claim_tally`), or ``derivation`` is another criterion's.
    :raises ClaimReaderDefect: The host's reader raised, or answered what its measures cannot hold.
    """
    if derivation is not None and derivation.criterion != criterion.name:
        raise ValueError(
            f"claim criterion {criterion.name!r} is tallied with the readings derived from {derivation.criterion!r}"
        )
    if grade.claim_evidence_digest != claim_evidence_digest(evidence):
        raise ValueError(
            f"claim criterion {criterion.name!r}'s grade names claim evidence {grade.claim_evidence_digest[:12]}, "
            f"and the trace stores {claim_evidence_digest(evidence)[:12]}"
        )
    tally = claim_tally(grade, evidence, grounded_in=criterion.grounded_in, ground_loss=grade.ground_loss)
    if derivation is None:
        return tally
    return tally.model_copy(update={"derived": derive_claim_readings(derivation, grade, evidence)})


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


def refuse_web_without_search(
    criteria: Sequence[ClaimCriterion],
    *,
    template: str,
    web: WebSearch | None,
    rates: ExternalRateTable | None = None,
    enforced: bool = False,
    witnessed: bool = False,
) -> None:
    """Refuse grading ``criteria`` where one with a web budget could not search as declared, before any call.

    Run without a web search, such a criterion would grade every nominated claim ``unverifiable`` under a fingerprint
    that says the web was asked. Under an enforcing cost cap its searches must be priced, by a rate the run declares
    for ``(identity, "query")`` or by the port's own ``price_ceiling``: an unpriced search could spend past what the
    cap set aside. The run's launch, a re-judge and a take-up refuse by this one rule.

    :param criteria: The criteria about to be graded.
    :param template: The template's name, for the refusal.
    :param web: The web search, or ``None``.
    :param rates: The run's rate table, or ``None``.
    :param enforced: Whether an enforcing cost cap bounds the grading.
    :param witnessed: Whether the cells are witnessed, which are graded with no web search.
    :raises ValueError: A criterion with a web budget meets a witnessed cell or no web search, or one with no
        identity; the port's ceiling is no amount; or the search is unpriced under an enforcing cap.
    """
    webbed = [criterion.name for criterion in criteria if criterion.web is not None]
    if not webbed:
        return
    named = ", ".join(map(repr, webbed))
    if witnessed:
        raise ValueError(
            f"claim criteria {named} of template {template!r} declare a web budget, and a witnessed cell is graded "
            "with no web search: web checks are not supported on witnessed cells yet, so declare web None for this "
            "template"
        )
    if web is None:
        raise ValueError(
            f"claim criteria {named} of template {template!r} declare a web budget, and this run has no web search; "
            "hand one in (RunnerOptions.web_search, KindWiring.web_search at launch, or web_search to a re-judge), or "
            "declare web None to grade against the case's evidence alone"
        )
    identity = getattr(web, "identity", None)
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(
            f"the web search ({type(web).__name__}) names no identity, which the spend records its searches under; "
            "give it a non-blank identity"
        )
    # Priced once here, whatever the cap, so a port whose ceiling is no amount is refused before anything is asked.
    if query_price(web, rates)[0] is None and enforced:
        raise ValueError(
            f"claim criteria {named} of template {template!r} check claims on the web search {identity!r}, which "
            "states no price_ceiling, and the run declares no rate for it (ExternalRateTable "
            f"{(identity, QUERY_UNIT)!r}); under an enforcing cost cap every search is priced before it is sent, so "
            "declare that rate, give the port a price_ceiling, or run without the cap"
        )


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
    derivation: ClaimDerivation | None,
    truth_kept: Callable[[GradedCriterion], None] | None = None,
    search: Search | None = None,
) -> GradedCriterion:
    """Grade one cell under one criterion: list its claims, verify them, check the nominated ones on the web where it
    declares a budget, and keep the grade with its tally.

    What the client raises in the extraction or a truth call (a provider failure, a replay's miss, a completion that
    breaks the protocol) is not caught: it is the judge phase's to classify. The web step's failures are its own
    (:meth:`~threetears.evals.kernel.claims.web.WebChecker.check`): they ride on the grade's ``web_loss`` and the
    tally's, and truth stands; a replay's miss or a protocol defect there is raised too.

    :param criterion: The criterion.
    :param evidence: The cell's claim evidence.
    :param key: The case's reference key, whose facts are case evidence; None when the case has none.
    :param client: The judge's client.
    :param graded: The grades the cell's trace already holds from the criteria graded before this one, which the
        trace's cap counts beside this one.
    :param derivation: The host's readings of the criterion, made with every tally (:func:`grade_tally`), or None.
    :param truth_kept: Told what the cell keeps if its deadline strikes in a later step: before a ground step starts,
        the grade and tally its truth step leaves, its ground step lost to the deadline (``ground_loss``
        ``deadline``), since the ground step's length grows with what the arm fetched; and before a web step starts,
        the grade and tally the verify step leaves, every nominated claim unchecked (``skipped_cap``). In both, every
        claim the criterion would check on the web reads ``skipped_cap``. Not told when that grade would not fit the
        trace.
    :param search: The host's search, billed and bounded by the caller; required for a criterion with a web budget
        that nominates a claim. A grade that checked a claim on a result a recording served is marked
        (:attr:`~threetears.evals.schema.models.ClaimGrade.web_replayed`).
    :return: The grade and its tally.
    :raises ClaimGradingFailed: A step failed (``cut_short``, ``unreadable``, ``repair_budget_spent``, or
        ``cost_cap`` or ``hold_exceeded`` from a client its hold refuses outside the web step), or the grade would
        take the trace past
        :data:`~threetears.evals.schema.models.CLAIM_TRACE_BYTES` (``evidence_over_cap``).
    :raises ValueError: A criterion with a web budget nominated a claim, and no search was given.
    :raises ClaimReaderDefect: The host's reader of the criterion raised, or answered what its measures cannot hold.
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
            ground_loss=verification.ground_loss,
            web_loss=web.loss,
            web_replayed=web.replayed and bool(web.records),
        )
        held = claim_payload_bytes(evidence, {**graded, criterion.name: grade})
        if held > CLAIM_TRACE_BYTES:
            raise ClaimGradingFailed(
                f"claim criterion {criterion.name!r}'s grade would take the trace's claims payload to {held} bytes, "
                f"and CLAIM_TRACE_BYTES allows {CLAIM_TRACE_BYTES}",
                loss="evidence_over_cap",
            )
        tally = grade_tally(grade, evidence, criterion, derivation=derivation)
        return GradedCriterion(
            grade=grade, tally=tally, ground_failure=verification.ground_failure, web_failure=web.failure
        )

    def fallback(verification: Verification) -> None:
        """Tell ``truth_kept`` what the cell keeps if its deadline strikes in the next step."""
        if truth_kept is None:
            return
        try:
            truth_kept(kept(verification, unchecked(verification)))
        except ClaimGradingFailed:
            # NOSILENT: a fallback grade that outgrows the trace is no fallback for a deadline, so none is kept. The
            # criterion is recorded all the same: evidence_over_cap once the full grade meets the same cap, or deadline
            # when the deadline strikes in a later step first.
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


async def grade_or_loss(
    criterion: ClaimCriterion,
    evidence: ClaimEvidence,
    *,
    key: ReferenceKey | None,
    client: CompletionClient,
    graded: Mapping[str, ClaimGrade],
    what: str,
    derivation: ClaimDerivation | None,
    truth_kept: Callable[[GradedCriterion], None] | None = None,
    search: Search | None = None,
) -> GradedCriterion | KnownClaimLoss:
    """Grade one cell under one criterion (:func:`grade_criterion`), or name the loss the cell records instead.

    Every way the criterion fails is its own loss, logged: a step's (:class:`ClaimGradingFailed`), and the provider
    failing a call (``provider_error``). Raised instead, failing what asked, since none is an outcome of the cell:
    a completion that breaks the protocol in the extraction or a truth call, a call the rig could not serve (a
    replay's miss), and the host's claim reader failing (:class:`ClaimReaderDefect`).

    :param criterion: The criterion.
    :param evidence: The cell's claim evidence.
    :param key: The case's reference key, or None.
    :param client: The judge's client.
    :param graded: The grades the cell's trace already holds from criteria graded before this one.
    :param what: What is graded, for the log line a loss writes ("Eval run r: claim criterion c for case=k").
    :param derivation: See :func:`grade_criterion`.
    :param truth_kept: See :func:`grade_criterion`.
    :param search: The host's search, for a criterion with a web budget (:func:`grade_criterion`). A caller refuses
        such a criterion with no search before it asks anything (the run's launch refusal), so a missing one never
        reaches here as a loss.
    :return: The grade and its tally, or the loss.
    :raises CompletionProtocolDefect: A completion in the extraction or a truth call breaks the protocol.
    :raises ApparatusError: The rig could not serve a call.
    :raises ClaimReaderDefect: The host's reader of the criterion raised, or answered what its measures cannot hold.
    """
    try:
        return await grade_criterion(
            criterion,
            evidence,
            key=key,
            client=client,
            graded=graded,
            derivation=derivation,
            truth_kept=truth_kept,
            search=search,
        )
    except ClaimGradingFailed as failed:
        # Recorded, never swallowed: the loss is the criterion's on the cell, which keeps every other reading.
        log.warning("%s lost (%s): %s", what, failed.loss, failed)
        return failed.loss
    except CompletionProtocolDefect, ApparatusError, ClaimReaderDefect:
        raise
    except (
        Exception
    ) as e:  # prawduct:ok-broad-except — judge boundary; a provider's failure is this criterion's loss alone
        # NOSILENT: returned as provider_error, which the cell records as the criterion's loss.
        contain(e, log=log, what=f"{what} failed", level="error")
        return "provider_error"


__all__: list[str] = []
