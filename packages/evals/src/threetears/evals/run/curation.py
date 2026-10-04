"""Eval curation — the surfaces that retire, detach and destroy stored observations.

The eval corpus is append-only by default. Curation is the family that walks that
back: **archive** excludes a run from every cohort while destroying nothing, and
**delete** is the bounded exception for data that must actually go away. They belong
together because they share one contract — :func:`require_delete_confirmation`, the
echo-the-id discipline every destructive entry point here is gated on. **Detach** —
removing campaign membership without touching the run — is the campaign family's
(:func:`threetears.evals.analysis.campaigns.remove_runs_from_campaign`); the delete
cascade here detaches a destroyed run from every campaign itself, under the same
campaign write lock (:mod:`threetears.evals.contracts.campaign_writes`).

**Functions over storage, not methods on a service.** Each entry point takes a
:class:`CurationStore` and typed parameters, so a host's service delegates here and
the family itself can be exercised against a storage double with no service
constructed at all — or against a double implementing nothing but the port, which
is what makes the port's narrowness a fact rather than a claim.

**A delete that fails halfway logs what it destroyed before it raises.** These
cascades are irreversible and there is no import path for runs or results, so the
exception body — which is transient — is never the only record. Every partial path
emits an ``eval.delete_run PARTIAL …`` line naming how far it got, because an
operator reading ``logs(action='errors')`` after a failed delete needs to know what
survived, and re-reading the run document will tell them it is intact when its
results are already gone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from threetears.evals.contracts.campaign_writes import serialized_campaign_write
from threetears.evals.contracts.errors import NotFoundError, StorageError, ValidationFailedError
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.host.profile import HostProfile
    from threetears.evals.contracts.campaign import EvalAnalysis, EvalCampaign, EvalInsight
    from threetears.evals.contracts.models import EvalResult, EvalRun

log = get_logger(__name__)


class CurationStore(Protocol):
    """Everything the curation family reads, writes and destroys — and nothing else.

    The widest of the engine's storage ports, because curation is the family
    that reaches across every kind of stored observation: runs and their
    results, analyses, campaigns and insights. It is still cut to the calls this
    module makes rather than to what a storage layer offers, which is the whole
    difference between a port and a re-export — a host adopting this family
    implements only methods it can trace to a line here, not every method a
    full storage layer happens to carry. (A reporter case's retirement is the
    reporter kind's, with its own port:
    :class:`~threetears.evals.analysis.reporter_curation.ReporterCaseStore`.)

    Structural, so a host's own storage satisfies it by having the methods.
    :class:`~threetears.evals.contracts.storage.EvalStorage` does, with no
    inheritance and no registration.

    **Two members are redeclared rather than inherited, and that is the choice.**
    ``load_eval_runs`` / ``query_eval_results_by_run`` are
    :class:`~threetears.evals.analysis.bundle.CampaignReadStore`'s, and composing from
    it would be legal here. It would also hand a consumer of this family a
    contract naming the bundle-assembly port it has no concept of,
    which is the union-port problem one size down. Interface segregation wins over
    de-duplication at a seam whose two sides ship as different packages.

    Positional parameters are positional-only, so an implementation's own parameter
    names never have to match the port's. ``scope_id`` is the engine's word for a
    partition it never interprets, in the port and in this module's public signatures
    alike.
    """

    def load_eval_runs(
        self, run_ids: Sequence[str], scope_id: str, /, *, elide_payload: frozenset[str]
    ) -> list[EvalRun]:
        """Load the named runs within a scope in one read, leaving ``elide_payload`` out of each payload.

        A run that does not resolve there is absent from the answer.

        **Every returned run must record what the read left out**: an implementation calls
        :meth:`~threetears.evals.contracts.models.EvalRun.note_elided_payload` with ``elide_payload`` on each run
        it returns. The store's projection cannot say so itself — a document with a path left out
        looks exactly like one stored without it — and an unmarked run reads as whole, so rebuilding
        the host's subject from it, or writing it back, proceeds with the value silently gone
        instead of refusing.
        """
        ...

    def set_eval_run_archived(self, run_id: str, scope_id: str, /, *, archived: bool) -> bool:
        """Write one run's ``archived`` flag in place; ``False`` when the run does not resolve. Raises ``StorageError``."""
        ...

    def delete_eval_run(self, run_id: str, scope_id: str, /) -> bool:
        """Destroy one run document, reporting whether it was there."""
        ...

    def query_eval_results_by_run(self, run_id: str, scope_id: str, /) -> list[EvalResult]:
        """Every result belonging to one run within a scope."""
        ...

    def delete_eval_result(self, result_id: str, scope_id: str, /) -> bool:
        """Destroy one result and its trace sibling, reporting whether it was there."""
        ...

    def load_analysis(self, analysis_id: str, scope_id: str, /) -> EvalAnalysis | None:
        """Load one analysis within a scope, or ``None``."""
        ...

    def save_analysis(self, analysis: EvalAnalysis, /) -> None:
        """Write an analysis; raises ``StorageError`` (``ConflictError`` on a lost ``if_match``) rather than returning a flag."""
        ...

    def delete_analysis(self, analysis_id: str, scope_id: str, /) -> bool:
        """Destroy one analysis, reporting whether it was there."""
        ...

    def save_campaign(self, campaign: EvalCampaign, /) -> None:
        """Write a campaign; raises ``StorageError`` (``ConflictError`` on a lost ``if_match``) rather than returning a flag."""
        ...

    def list_campaigns(self, scope_id: str, /) -> list[EvalCampaign]:
        """Every campaign in a scope — the delete cascade's detach scans membership in Python."""
        ...

    def load_insight(self, insight_id: str, scope_id: str, /) -> EvalInsight | None:
        """Load one insight within a scope, or ``None``."""
        ...

    def delete_insight(self, insight_id: str, scope_id: str, /) -> bool:
        """Destroy one insight, reporting whether it was there."""
        ...


#: The alternative an ARCHIVABLE eval object offers instead of destruction.
#:
#: Named rather than inlined because it is **not universal**, and the exact list
#: matters: an ``archived`` field exists on ``EvalRun``, ``EvalAnalysis`` and
#: ``EvalCampaign``, and does **not** exist on ``EvalResult``, ``EvalInsight`` or a
#: classifier snapshot. Every call site that destroys one of the second group must
#: pass its own ``alternative``, or a refused caller is sent to a surface the object
#: does not have — advice they cannot act on, in the one message whose whole job is
#: to tell them what to do instead.
#:
#: This constant previously claimed the property was "true of runs, results,
#: analyses and insights", which was wrong about two of the four and shipped that
#: wrongness into the live refusal text for both. ``test_curation_seam.py`` now pins
#: the membership against the models, because a comment naming a set is a comment
#: that goes stale the first time the set changes.
_ARCHIVE_INSTEAD = "archive it instead to exclude it from cohorts without destroying it"

#: What to do with a bad OBSERVATION instead of deleting it.
#:
#: There is deliberately no per-result archive (see :func:`delete_result`): a run is
#: the unit an operator reasons about, and a hidden subset of a run's results would
#: silently change that run's own aggregates. So the reversible move exists one level
#: up, and a refused caller is pointed at the surface that actually exists.
_ARCHIVE_THE_RUN_INSTEAD = (
    "archive the whole run instead (run_archive), which excludes every one of its observations from "
    "cohorts without destroying anything — there is no per-result archive, deliberately"
)

#: Why an insight has no archive to be pointed at, said rather than implied.
#:
#: Not an oversight and not a gap to close. An insight is fed back to the generator
#: as PRIOR CONTEXT on every later generation over the same subject, so a wrong one
#: keeps steering analyses until it is gone. Marking it archived without every reader
#: honouring the mark — the bundle's ``prior_insights`` read included — would leave
#: the falsehood in the generator's input while *looking* handled, which is strictly
#: worse than either deleting it or leaving it alone. So the ANALYSIS is archived, and
#: the insights it minted are deleted. The asymmetry is the rule, not a shortfall against it.
#:
#: The one retirement an insight does have is its analysis's: archiving the analysis
#: RETRACTS what it minted, read from the analysis at every use
#: (:func:`~threetears.evals.analysis.bundle.retracted_insights`) rather than stamped on
#: the insight — so the one mark is honoured by the bundle and the ledger listing alike,
#: and there is no second flag for a reader to miss.
_NO_INSIGHT_ARCHIVE = (
    "leave it in the ledger — an insight has no archive, because it is fed back to the generator as "
    "prior context and a retained-but-marked one would go on steering later analyses. Deleting a wrong "
    "insight is the intended answer"
)


def require_delete_confirmation(
    kind: str,
    object_id: str,
    confirm: str | None,
    *,
    cascade: str | None = None,
    alternative: str = _ARCHIVE_INSTEAD,
) -> None:
    """Refuse a destructive eval delete unless the caller echoed the target's id.

    Stored eval observations are otherwise append-only: an operator retires one by
    ARCHIVING it, which excludes it from every cohort while destroying nothing.
    Hard delete is the bounded exception for a corpus that must actually go away,
    so it is gated the same way ``nuke_all_eval_data`` is — explicitly, and loudly.

    The confirmation is the object's own id rather than a constant word, because a
    constant is memorised and then travels: ``confirm="yes"`` typed against the
    wrong id destroys the wrong object with the caller's full confidence. An
    echoed id can only confirm the thing it names.

    Args:
        kind: Human-readable object kind, used in the refusal message.
        object_id: The id being destroyed — also the required confirmation value.
        confirm: What the caller passed; surrounding whitespace is ignored.
        cascade: What else goes with the object, as a phrase reading on from its
            id (``"and its 3 stored eval results"``). A caller who is told the
            count before confirming can weigh the delete; one told only the id
            is weighing the wrong object. Omit when the delete takes nothing else.
        alternative: What to do instead of destroying it, as a phrase reading on
            from "or". Defaults to archiving, which is the answer for the eval
            objects that HAVE an ``archived`` field — ``EvalRun``, ``EvalAnalysis``,
            ``EvalCampaign``. ``EvalResult``, ``EvalInsight`` and a classifier
            snapshot do not, so those call sites pass a replacement; taking the
            default there sends a refused caller to a surface the object has no
            access to, which is worse than saying nothing.

    Raises:
        ValidationFailedError: ``confirm`` does not echo ``object_id``.
    """
    if (confirm or "").strip() != object_id:
        target = f"{kind} '{object_id}'" + (f" {cascade}" if cascade else "")
        raise ValidationFailedError(
            f"deleting {target} is unrecoverable and there is no import path to restore it. "
            f"Pass confirm='{object_id}' to proceed, or {alternative}."
        )


def load_run_as_listed(storage: CurationStore, run_id: str, scope_id: str, *, profile: HostProfile) -> EvalRun:
    """Load one run the way a listing loads it: without the payload paths the host declares a listing leaves out.

    For a reader of a run's scalars. The returned run records what its payload is missing
    (:attr:`~threetears.evals.contracts.models.EvalRun.elided_payload_paths`), so a reader that needs a left-out
    value refuses rather than reading its absence; :func:`~threetears.evals.run.lifecycle.get_run` is the whole read.

    Args:
        storage: Eval storage backend.
        run_id: The run to load.
        scope_id: Partition key — the scope the run belongs to.
        profile: The host, whose ``listing_elisions`` the read leaves out.

    Returns:
        The run, with the host's listing elisions (``HostProfile.listing_elisions``) left out and marked.

    Raises:
        NotFoundError: No run with that id in the scope.
    """
    runs = storage.load_eval_runs([run_id], scope_id, elide_payload=profile.listing_elisions)
    if not runs:
        raise NotFoundError("run", run_id)
    return runs[0]


def set_run_archived(
    storage: CurationStore, run_id: str, scope_id: str, *, archived: bool, profile: HostProfile
) -> EvalRun:
    """Archive or un-archive a run — reversible exclusion from every cohort.

    An archived run keeps its document, its results and its campaign memberships;
    what changes is that aggregating surfaces stop counting it (see
    :func:`~threetears.evals.run.reads.list_runs`) and analysis bundles
    report it as deliberately excluded rather than pooling it. Un-archiving
    restores it everywhere, which is the property that makes this the safe answer
    to a contaminated cohort and hard delete the last resort.

    **The flag is written in place, and the run is read the way a listing reads it**
    (:func:`load_run_as_listed`). Nothing here needs the host's payload, which is most
    of a run document, so the write sets the one field and never sends the document
    back — and so cannot lose a concurrent writer's edit to any other field.

    Idempotent: setting the state a run already carries writes nothing.

    Args:
        storage: Eval storage backend.
        run_id: Run to curate.
        scope_id: Partition key — the scope the run belongs to.
        archived: Target state. ``True`` excludes, ``False`` restores.
        profile: The host, whose listing elisions the run is read without.

    Returns:
        The run as persisted, read as a listing reads it: its payload is missing the host's
        listing elisions and says so (:attr:`~threetears.evals.contracts.models.EvalRun.elided_payload_paths`).
        Read it through ``get_run`` for the whole document.

    Raises:
        NotFoundError: No run with that id in the scope, including one deleted between the
            read and the write.
        StorageError: The write failed.
    """
    current = load_run_as_listed(storage, run_id, scope_id, profile=profile)
    if current.archived == archived:
        return current
    if not storage.set_eval_run_archived(run_id, scope_id, archived=archived):
        raise NotFoundError("run", run_id)
    log.info(
        "eval.set_run_archived run=%s scope=%s archived=%s",
        run_id,
        scope_id,
        archived,
    )
    # Re-read rather than returning ``current`` with the flag flipped: the caller is told this is
    # "the run as persisted", and only storage can say what that is.
    return load_run_as_listed(storage, run_id, scope_id, profile=profile)


def set_analysis_archived(
    storage: CurationStore,
    analysis_id: str,
    scope_id: str,
    *,
    archived: bool,
    reason: str | None = None,
) -> EvalAnalysis:
    """Archive or un-archive a stored analysis — the alternative to destroying it.

    An analysis is archived when it was shown FALSE — its claims contradicted by
    the evidence it was generated over — and archiving rather than deleting is
    deliberate: the record that a falsehood was generated is itself evidence, and
    a system that measures how often its generator is wrong needs the denominator.
    The document, its findings and the insights it minted all survive; what
    changes is that every reading surface marks it, so no reader mistakes it for a
    live report — and the insights it minted are RETRACTED for as long as it stays
    archived: left out of every generation's prior context and marked where the
    ledger is listed. That is read from this flag at each use, so this write is the
    whole of it and un-archiving restores them.

    **Un-archiving clears the reason.** A restored analysis carrying "archived
    because its recommendation is unsupported" describes a state it is no longer
    in, and a stale justification reads as a live one. The two fields therefore
    move together in a single write.

    **The reason is optional, not required.** Most retirements are routine — an
    analysis superseded by a later one over the same campaign — and forcing prose
    onto those produces placeholder text that dilutes the entries carrying real
    evidence. It is still the field that makes the archive legible, so every
    surface that offers this asks for one.

    Idempotent: when the stored document already carries the requested state and
    reason, nothing is written, so a repeated call cannot lose a concurrent edit.

    **A whole-document write with no optimistic-concurrency guard.** A run has
    other writers, which is why :func:`set_run_archived` sets its one field in
    place instead. An analysis is written once, by the generation that produced it,
    and is never mutated again — this is its only writer, so there is no concurrent
    field for a blind write to clobber. Storage exposes no ``load_analysis_with_etag`` for the
    same reason. A second analysis writer would change that and would have to bring
    one with it.

    Args:
        storage: Eval storage backend.
        analysis_id: Analysis to curate.
        scope_id: The scope it lives in.
        archived: Target state. ``True`` marks it everywhere, ``False`` restores it.
        reason: Why, in the operator's words. Recorded when archiving; ignored when
            restoring, which clears whatever reason was there.

    Returns:
        The analysis as persisted.

    Raises:
        NotFoundError: No analysis with that id in the scope.
        StorageError: The updated analysis failed to persist.
    """
    from threetears.evals.contracts.campaign import EvalAnalysis

    current = storage.load_analysis(analysis_id, scope_id)
    if current is None:
        raise NotFoundError("analysis", analysis_id)

    target_reason = reason if archived else None
    if current.archived == archived and current.archived_reason == target_reason:
        return current

    data = current.to_dict()
    data["archived"] = archived
    data["archived_reason"] = target_reason
    storage.save_analysis(EvalAnalysis.from_dict(data))
    log.info(
        "eval.set_analysis_archived analysis=%s campaign=%s archived=%s reason=%s",
        analysis_id,
        current.campaign_id,
        archived,
        target_reason or "(none given)",
    )
    # Re-read rather than returning the constructed document, for the reason
    # set_run_archived does: the caller is told this is "the analysis as
    # persisted", and only storage can say what that is.
    persisted = storage.load_analysis(analysis_id, scope_id)
    if persisted is None:
        raise NotFoundError("analysis", analysis_id)
    return persisted


def delete_run(storage: CurationStore, run: EvalRun, scope_id: str, *, confirm: str | None = None) -> dict[str, Any]:
    """Delete a run, the results it owns, and its campaign memberships.

    **Cascades to results by design.** ``EvalResult`` carries its run id and
    nothing else identifies the conditions it was produced under, so a result
    outliving its run is unreadable data that still enters every scope-wide
    aggregate — a leak of exactly the contamination the delete was performed to
    remove. Results are destroyed first: if any of them fails, the run document
    survives, so the survivors still belong to something and the caller sees
    the partial state named rather than inferring it.

    **A non-terminal run is refused.** Results are persisted incrementally as
    each cell finishes, so a live job goes on calling ``save_eval_result``
    after the cascade has run — every one of those writes references a run id
    that no longer resolves, which is the exact orphan class the cascade
    exists to prevent, and the job manager tolerates the missing document
    silently. Cancel first, then delete.

    **Insight back-references are left dangling, deliberately.** Any
    :class:`~threetears.evals.contracts.campaign.EvalInsight` citing this run in
    ``evidence_run_ids`` keeps the id, which will no longer resolve. The
    campaign detach above exists because a dangling *member* is
    indistinguishable from a lookup failure; an insight's evidence citation is
    not — insights are the durable ledger and destroying conclusions to tidy a
    reference would lose more than it repairs. So the id stays and this states
    that it dangles, rather than the reader discovering it as a 404.

    Prefer :func:`set_run_archived` — deletion is unrecoverable and there is no
    import path for runs or results.

    Args:
        storage: Eval storage backend.
        run: The run to destroy, already loaded — the caller performs the
            not-found check, so a missing run is refused before ``confirm`` is
            even examined.
        scope_id: Partition key — the scope the run belongs to.
        confirm: Must echo the run's id; see :func:`require_delete_confirmation`.

    Returns:
        What was removed: ``{"run_id", "results_deleted", "campaigns_detached"}``.

    Raises:
        ValidationFailedError: ``confirm`` does not echo the run id, or the run
            is still ``pending`` / ``running``.
        StorageError: A result, a campaign detach, or the run itself failed to
            persist; the message names how far the delete got.
    """
    from threetears.evals.contracts.models import NON_TERMINAL_RUN_STATUSES

    run_id = run.id
    require_delete_confirmation("run", run_id, confirm)
    if run.status in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(
            f"run '{run_id}' is {run.status} — a live job persists each result as it finishes, so "
            f"deleting now orphans every result written after this call. Cancel it first "
            f"(cancel_run), confirm it reached a terminal status, then delete."
        )

    results = storage.query_eval_results_by_run(run_id, scope_id)
    failed = [r.id for r in results if not storage.delete_eval_result(r.id, scope_id)]
    if failed:
        # Destruction already happened for the results that did delete. The success
        # path below logs what it removed; so must every path that removes something
        # and then raises, or an irreversible partial delete leaves no trace in
        # logs(action='errors') and the operator cannot tell what survived.
        log.error(
            "eval.delete_run PARTIAL run=%s scope=%s results_deleted=%d results_failed=%d campaigns_detached=(none)",
            run_id,
            scope_id,
            len(results) - len(failed),
            len(failed),
        )
        raise StorageError(
            f"run '{run_id}' left intact: {len(failed)} of {len(results)} result(s) could not be deleted "
            f"({', '.join(failed[:3])}{'…' if len(failed) > 3 else ''}) — retry to finish the cascade"
        )

    campaigns = _detach_run_from_all_campaigns(storage, run_id, scope_id, results_already_deleted=len(results))

    if not storage.delete_eval_run(run_id, scope_id):
        log.error(
            "eval.delete_run PARTIAL run=%s scope=%s results_deleted=%d campaigns_detached=%s run_document=SURVIVED",
            run_id,
            scope_id,
            len(results),
            campaigns or "(none)",
        )
        raise StorageError(
            f"deleted {len(results)} result(s) and detached {len(campaigns)} campaign(s), but run "
            f"'{run_id}' itself could not be deleted — retry to finish"
        )
    log.warning(
        "eval.delete_run run=%s scope=%s results_deleted=%d campaigns_detached=%s",
        run_id,
        scope_id,
        len(results),
        campaigns or "(none)",
    )
    return {"run_id": run_id, "results_deleted": len(results), "campaigns_detached": campaigns}


def delete_result(
    storage: CurationStore, result: EvalResult, scope_id: str, *, confirm: str | None = None
) -> dict[str, Any]:
    """Delete a single eval result, leaving its run and siblings in place.

    The per-observation eraser, for a cell whose measurement is known bad while
    the rest of the run stands. Prefer archiving the whole run when the run is
    what went wrong — there is no per-result archive, deliberately: a run is the
    unit an operator can reason about, and a hidden subset of a run's results
    would silently change that run's own aggregates.

    As with :func:`delete_run`, an insight citing this result in
    ``evidence_result_ids`` keeps the id and it stops resolving; the ledger is
    durable on purpose and is not cascaded.

    Args:
        storage: Eval storage backend.
        result: The result to destroy, already loaded — the caller performs the
            not-found check.
        scope_id: Partition key — the scope the result belongs to.
        confirm: Must echo the result's id.

    Returns:
        ``{"result_id": ..., "run_id": ...}`` — the run the result belonged to,
        so the caller can re-read what its aggregates now say.

    Raises:
        ValidationFailedError: ``confirm`` does not echo the result id.
        StorageError: The result failed to delete.
    """
    result_id = result.id
    require_delete_confirmation("result", result_id, confirm, alternative=_ARCHIVE_THE_RUN_INSTEAD)
    if not storage.delete_eval_result(result_id, scope_id):
        raise StorageError(f"failed to delete result '{result_id}'")
    log.warning("eval.delete_result result=%s run=%s scope=%s", result_id, result.eval_run_id, scope_id)
    return {"result_id": result_id, "run_id": result.eval_run_id}


def delete_analysis(storage: CurationStore, analysis: EvalAnalysis, *, confirm: str | None = None) -> dict[str, Any]:
    """Delete a stored analysis, leaving the insights it minted in place.

    Insights are the durable ledger — they outlive the analysis that observed
    them and are referenced by later work — so cascading here would destroy
    conclusions rather than the (regenerable) report that carried them. Delete
    an insight explicitly with :func:`delete_insight` when it is the insight
    that is wrong.

    The surviving insights therefore carry a ``source_analysis_id`` that no
    longer resolves. That is the cost of keeping them, not an oversight:
    ``insight_list`` renders the column so the dangle is visible where the
    ledger is read, rather than surfacing as an unexplained 404 later.

    **Prefer :func:`set_analysis_archived`.** An analysis shown false is archived,
    never silently deleted:
    the record that a falsehood was generated is the denominator a closed loop
    measures itself against, and destroying it destroys that number. Hard delete
    remains available for an analysis nobody needs the record of.

    Args:
        storage: Eval storage backend.
        analysis: The analysis to destroy, already loaded — the caller performs
            the not-found check.
        confirm: Must echo the analysis's id.

    Returns:
        ``{"analysis_id": ..., "campaign_id": ...}``.

    Raises:
        ValidationFailedError: ``confirm`` does not echo the analysis id.
        StorageError: The analysis failed to delete.
    """
    analysis_id = analysis.id
    require_delete_confirmation("analysis", analysis_id, confirm)
    if not storage.delete_analysis(analysis_id, analysis.scope_id):
        raise StorageError(f"failed to delete analysis '{analysis_id}'")
    log.warning("eval.delete_analysis analysis=%s campaign=%s", analysis_id, analysis.campaign_id)
    return {"analysis_id": analysis_id, "campaign_id": analysis.campaign_id}


def delete_insight(
    storage: CurationStore, insight_id: str, scope_id: str, *, confirm: str | None = None
) -> dict[str, Any]:
    """Delete a single insight from the ledger.

    **The one eval object where delete is the INTENDED answer rather than the last
    resort.** Every sibling here prefers archiving; an insight has no archive, and
    that asymmetry is deliberate. An insight is fed back to the generator as prior
    context on every later generation over the same subject, so a wrong one keeps
    steering analyses until it is gone — marking it retired without every reader
    honouring the mark would leave the falsehood in the generator's input while
    looking handled. So the ANALYSIS shown false is archived, and the insights it minted are
    deleted. The refusal message therefore points at no archive, and says why.

    Loads its own target, unlike the deletes above: there is no ``get_insight``
    read surface for a caller to have gone through first.

    Args:
        storage: Eval storage backend.
        insight_id: Insight to destroy.
        scope_id: The scope it lives in.
        confirm: Must echo ``insight_id``.

    Returns:
        ``{"insight_id": ..., "source_campaign_id": ...}``.

    Raises:
        NotFoundError: No insight with that id in the scope.
        ValidationFailedError: ``confirm`` does not echo ``insight_id``.
        StorageError: The insight failed to delete.
    """
    insight = storage.load_insight(insight_id, scope_id)
    if insight is None:
        raise NotFoundError("insight", insight_id)
    require_delete_confirmation("insight", insight_id, confirm, alternative=_NO_INSIGHT_ARCHIVE)
    if not storage.delete_insight(insight_id, scope_id):
        raise StorageError(f"failed to delete insight '{insight_id}'")
    log.warning("eval.delete_insight insight=%s campaign=%s", insight_id, insight.source_campaign_id)
    return {"insight_id": insight_id, "source_campaign_id": insight.source_campaign_id}


@serialized_campaign_write
def _detach_run_from_all_campaigns(
    storage: CurationStore, run_id: str, scope_id: str, *, results_already_deleted: int
) -> list[str]:
    """Remove ``run_id`` from every campaign holding it; return those campaign ids.

    Called on the delete path so destroying a run cannot leave a campaign
    pointing at an id that no longer resolves — a dangling member reads as an
    ``unresolved_run_ids`` gap in every later bundle and would misreport a
    deliberate curation as a lookup failure.

    Only the run's own scope is scanned, and that is complete rather than narrow: a
    campaign holds only runs in its own scope (:mod:`threetears.evals.analysis.campaigns`
    refuses any other), so no campaign elsewhere can hold this one. Campaigns are few,
    so membership is scanned in Python rather than pushed into a store predicate.

    Args:
        storage: Eval storage backend.
        run_id: The run being removed from campaign membership.
        scope_id: The scope the run, and so every campaign holding it, lives in.
        results_already_deleted: How many of the run's results the caller has
            already destroyed. Required, not defaulted: this helper only ever runs
            mid-cascade, and a default would let a future caller emit a truthful-
            looking "0 result(s) already destroyed" while results were in fact gone.
            It is carried in for the failure message alone. This raises from the
            MIDDLE of an unrecoverable cascade, so the message has to say how much
            is already gone ("... is PARTIALLY DELETED: N result(s) already
            destroyed ..."); an operator who read only that a detach failed, and
            abandoned there, would be left holding a run that still reads intact
            through ``get_run`` while its results are already destroyed.

    Raises:
        StorageError: A campaign failed to persist. The message names what has
            already been destroyed and what has already been detached.
    """
    touched: list[str] = []
    for campaign in storage.list_campaigns(scope_id):
        if run_id not in campaign.run_ids:
            continue
        campaign.run_ids = [rid for rid in campaign.run_ids if rid != run_id]
        try:
            storage.save_campaign(campaign)
        except StorageError as e:
            # Log BEFORE raising: the cascade is already part-done and irreversible, and
            # the exception body is transient. Without this line the only record of what
            # was destroyed dies with the response — see delete_run's failure branches.
            log.error(
                "eval.delete_run PARTIAL run=%s results_deleted=%d campaigns_detached=%s failed_detach=%s",
                run_id,
                results_already_deleted,
                touched or "(none)",
                campaign.id,
            )
            raise StorageError(
                f"run '{run_id}' is PARTIALLY DELETED: {results_already_deleted} result(s) already "
                f"destroyed and {len(touched)} campaign(s) already detached, but detaching it from "
                f"campaign '{campaign.id}' failed. The run document still exists and still reads as "
                f"intact — retry the delete to finish it."
            ) from e
        touched.append(campaign.id)
    return touched


__all__ = [
    "CurationStore",
    "delete_analysis",
    "delete_insight",
    "delete_result",
    "delete_run",
    "load_run_as_listed",
    "require_delete_confirmation",
    "set_analysis_archived",
    "set_run_archived",
]
