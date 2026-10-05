"""Retiring and restoring a reporter case — the one curation write the reporter kind owns.

The rest of the curation family (archive, detach, delete over runs, results, analyses,
campaigns and insights) lives in :mod:`threetears.evals.run.curation`. This one lives beside the
reporter case bank instead, because the rule it enforces is the bank's: a restore is refused when
it would give a (campaign, memo) pair a second live case, and "live" is decided by
:func:`~threetears.evals.analysis.reporter_bank.decidable_reporter_case_bank`, the same derivation
the launch reads. The reporter kind is the analysis package's, so the write that has to agree
with its bank is too; in the run package it would be a run -> analysis import the dependency
matrix refuses.

It takes its own narrow store port, :class:`ReporterCaseStore`, cut to the three calls below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from threetears.evals.analysis.reporter_bank import case_pair, decidable_reporter_case_bank
from threetears.evals.analysis.reporter_kind import reporter_case_of
from threetears.evals.contracts.errors import NotFoundError, ValidationFailedError
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.storage import EvalStorage
    from threetears.evals.contracts.models import EvalTestCase

log = get_logger(__name__)


class ReporterCaseStore(Protocol):
    """The three storage calls a reporter case's retirement and restore make — and nothing else.

    Structural, so a host's own storage satisfies it by having the methods. Positional parameters
    are positional-only, which lets the port say ``scope_id`` — the engine's word for a partition
    it never interprets — while an implementation names the thing it partitions by.
    """

    def load_test_case(self, test_case_id: str, scope_id: str, /) -> EvalTestCase | None:
        """Load one test case within a scope, or ``None`` when it does not resolve there."""
        ...

    def save_test_case(self, test_case: EvalTestCase, /) -> None:
        """Write a test case; raises ``StorageError`` rather than returning a flag."""
        ...

    def query_test_cases(self, scope_id: str, /, *, template_id: str | None = None) -> list[EvalTestCase]:
        """Every test case in a scope, optionally narrowed to one template — a restore reads its template's bank."""
        ...


def set_reporter_case_archived(
    storage: ReporterCaseStore,
    test_case_id: str,
    scope_id: str,
    *,
    archived: bool,
    reason: str | None = None,
) -> EvalTestCase:
    """Retire (archive) a reporter case, or restore one — the answer to a case that can no longer measure anything.

    A retired case is not live: no launch runs it and no price counts it (the case bank's
    :meth:`~threetears.evals.analysis.reporter_bank.ReporterCaseBank.is_live`), while it stays
    readable, so every run already measured against it still resolves it and its calibration read
    says it was retired. Stored cases are immutable, so this is the one mutation a case takes —
    curation state beside its content, on the terms the run and analysis archive flags are. A case
    whose frozen bundle a schema change orphaned otherwise runs as an apparatus error on every cell
    of every launch, with no way out short of a re-freeze, which re-assembles today's evidence
    rather than the evidence its labels were written against.

    **Restoring is refused when it would make a second live case of the pair** — a case frozen for
    the same campaign and memo since this one was retired. Two live cases of one pair make every
    launch refuse; the state is prevented here rather than left for the launch to name. Restoring a
    case something has since superseded is allowed and changes nothing a launch reads, because
    supersession still holds. The restore reads the template's bank through
    :func:`~threetears.evals.analysis.reporter_bank.decidable_reporter_case_bank`, the derivation the
    launch reads, so the two cannot disagree about which case is live.

    Idempotent: the state and reason the case already carries write nothing. The reason is cleared
    on restore, since a reason outliving the retirement describes a state the case is no longer in.
    No optimistic-concurrency guard: a case is written once by its freeze and never again but here.

    Args:
        storage: Anything satisfying :class:`ReporterCaseStore`.
        test_case_id: The stored case.
        scope_id: The partition it lives in.
        archived: ``True`` retires it, ``False`` restores it.
        reason: Why, in the operator's words. Recorded when retiring; cleared when restoring.

    Returns:
        The case as persisted.

    Raises:
        NotFoundError: No such case in that scope.
        ValidationFailedError: It carries no reporter case — the reporter case bank is the one
            launch path that reads the flag, so retiring any other case would record a retirement
            no launch honours — or carries one this build cannot read, or restoring it would make a
            second live case of its pair (or its template holds a case this build cannot read, or it
            names no template, so that cannot be decided).
        StorageError: The updated case could not be persisted.
    """
    from threetears.evals.contracts.models import EvalTestCase

    current = storage.load_test_case(test_case_id, scope_id)
    if current is None:
        raise NotFoundError("test case", test_case_id)
    try:
        case = reporter_case_of(current)
    except ValueError as e:
        raise ValidationFailedError(
            f"test case {test_case_id!r} carries a reporter case this build cannot read, so whether it is live cannot "
            f"be decided here: {e} — read it with the build that wrote it"
        ) from e
    if case is None:
        raise ValidationFailedError(
            f"test case {test_case_id!r} carries no reporter case. Only a reporter case is retired this way: the "
            "reporter case bank is the one launch path that reads a case's retirement, so retiring any other case "
            "would record a retirement no launch honours."
        )
    target_reason = reason if archived else None
    if current.archived == archived and current.archived_reason == target_reason:
        return current
    if not archived:
        template_id = current.template_id
        if template_id is None:
            # Its rivals are the cases of its template, so a case naming none cannot be decided live or not.
            raise ValidationFailedError(
                f"test case {test_case_id!r} carries a reporter case but names no template, so whether restoring it "
                "makes a second live case of its pair cannot be decided; a freeze stores every reporter case under "
                "its template"
            )
        bank = decidable_reporter_case_bank(
            storage.query_test_cases(scope_id, template_id=template_id), template_id=template_id
        )
        if current.id not in bank.superseded_by:
            campaign_id, analysis_id = case_pair(case)
            if rivals := sorted(
                stored.id for stored, _ in bank.pair(campaign_id, analysis_id) if bank.is_live(stored.id)
            ):
                memo_text = f"analysis {analysis_id!r}" if analysis_id else "no recorded memo"
                raise ValidationFailedError(
                    f"restoring case {test_case_id!r} would make it a second live case of campaign {campaign_id!r} "
                    f"with {memo_text}, beside {', '.join(repr(case_id) for case_id in rivals)} — and a pair with two "
                    "live cases refuses every launch. Retire that case first, or leave this one retired."
                )

    data = current.to_dict()
    data["archived"] = archived
    data["archived_reason"] = target_reason
    storage.save_test_case(EvalTestCase.from_dict(data))
    log.info(
        "eval.set_reporter_case_archived case=%s template=%s scope=%s archived=%s reason=%s",
        test_case_id,
        current.template_id,
        scope_id,
        archived,
        target_reason or "(none given)",
    )
    # Re-read rather than returning the constructed document: the caller is told this is "the
    # case as persisted", and only storage can say what that is.
    persisted = storage.load_test_case(test_case_id, scope_id)
    if persisted is None:
        raise NotFoundError("test case", test_case_id)
    return persisted


if TYPE_CHECKING:

    def _eval_storage_satisfies_the_port(storage: EvalStorage) -> None:
        """Hold the engine's own store to this consumer's port, so a drifted signature fails typecheck."""
        store: ReporterCaseStore = storage
        del store


__all__ = ["ReporterCaseStore", "set_reporter_case_archived"]
