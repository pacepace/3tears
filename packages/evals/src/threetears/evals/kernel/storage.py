"""Storage layer for eval documents, over one document store.

Every eval document lives in a ``scope_id`` it names itself: definitions (``EvalTemplate`` /
``JudgeConfig`` / ``CatalogRubricDim``, campaigns, analyses, generation attempts, insights) and
runtime data (``EvalTestCase`` / ``EvalRun`` / ``EvalResult`` / ``EvalTrace`` / ``EvalCassette``)
alike. A write takes the scope from the document; every read and delete takes it as an argument.
The engine never stamps, defaults or interprets a scope — see
:mod:`threetears.evals.schema.store_port`.

A host may keep documents of its own in the same store. This module does not read or write them
and does not know their ``doc_type`` values: the host declares them when it builds
:class:`EvalStorage` (``host_doc_types``), and the operator wipe sweeps them beside the engine's own
(:data:`EVAL_DOC_TYPES`).

**This module names no backend.** Every read and write goes through the one
:class:`~threetears.evals.schema.store_port.DocumentStore` handed in at construction, so nothing here
knows which database, table or isolation mechanism sits behind it. Splitting a high-volume kind of
document into its own table, or partitioning by scope, is the adapter's routing by ``doc_type``
and ``scope_id`` — what the core keeps is the document model: which ``doc_type`` a model is, and
which scope a read is asking about.

Strict reads: every load path goes through the model's own ``from_dict(...)`` (:meth:`_hydrate`),
which validates exactly as construction does. A stored document with a field the model does not
declare, or without one it requires, is refused, and nothing is filtered on the way in. A core
document at an older core major is upgraded by that read before it is validated; one from a
newer major or from before the first release, or a regenerable document at another version, is
refused (:mod:`~threetears.evals.schema.versioning`). The one exception is a core document of this
build's major at a newer minor: it is read, the fields this build does not declare are kept on the
model, and every write here — each one a model's ``to_dict`` — puts them back, so a document a newer
build wrote survives this build's rewrite of it whole. Rows
arrive with the storage stamp already removed (the port's contract), so nothing but the model's
own fields reaches validation. A read can therefore raise, which is why the one read on the boot
path (:meth:`EvalStorage.query_non_terminal_eval_runs`) isolates a refused row rather than
failing the list, as does the operator's scan that names such rows (:meth:`EvalStorage.scan_eval_runs`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol, cast

from pydantic import ValidationError

from threetears.evals.kernel.campaign import (
    EvalAnalysis,
    EvalAnalysisAttempt,
    EvalCampaign,
    EvalGenerationSlot,
    EvalInsight,
    EvalSweep,
    generation_slot_id,
)
from threetears.evals.schema.base import EvalBaseModel, EvalDocumentModel
from threetears.evals.kernel.errors import ConflictError, EvalServiceError, StorageError
from threetears.evals.kernel.remedies import FIND_UNREADABLE_RUNS, REREAD_AND_RETRY, UNREADABLE_JOB_RECORD
from threetears.evals.kernel.judge_profiles import EvalJudgeProfile, JudgeProfileStore
from threetears.evals.schema.versioning import CORE_RESET_DOC_TYPES, DURABLE_DOC_TYPES
from threetears.evals.schema.models import (
    EVAL_DOC_TYPES,
    NON_TERMINAL_RUN_STATUSES,
    CalibrationRating,
    CassetteKey,
    CaseSet,
    CatalogRubricDim,
    EvalCassette,
    EvalResult,
    EvalRun,
    EvalCaseStratum,
    EvalRunStamp,
    EvalRunStatus,
    EvalTemplate,
    EvalTestCase,
    EvalTrace,
    JudgeConfig,
    JudgeEvidence,
    LabelKey,
    RubricDimTombstone,
    JudgeConfigTombstone,
    case_set_doc_id,
    eval_trace_doc_id,
    current_revision,
)
from threetears.evals.schema.out_of_run_spend import KnownOutOfRunPurpose, OutOfRunSpend, OutOfRunSpendStore
from threetears.evals.schema.store_port import DocumentStore, StoreClock, StoreConflict, refuse_on_event_loop
from threetears.evals.schema.logs import get_logger


#: The status a run written lost takes (:data:`~threetears.evals.kernel.job_leases.RUN_JOB`): what a run that stopped
#: without finishing has always read, its cause on ``cancellation_reason`` and ``lost``.
RUN_LOST_STATUS: Final = "cancelled"

log = get_logger(__name__)

#: The fields a batch read of judge evidence keeps of each trace (:meth:`EvalStorage.load_judge_evidence`): those that
#: locate and version an :class:`EvalTrace`, and the judge evidence with the declaration it is stored beside, so the
#: reduced trace still reads as one. Never the turns, the spans, the call ledger or the end state, which are most of
#: what a trace weighs.
_JUDGE_EVIDENCE_FIELDS: Final = (
    "id",
    "doc_type",
    "schema_version",
    "schema_minor",
    "scope_id",
    "result_id",
    "eval_run_id",
    "judge_evidence",
    "judged_artifact",
)


def _payload_exclusions(elide_payload: frozenset[str]) -> list[str]:
    """The store ``exclude`` paths for payload paths a run read leaves out, in a stable order."""
    return sorted(f"host_payload.{path}" for path in elide_payload)


class CaseRecordings(NamedTuple):
    """One case's recordings in one corpus, read whole before a replay cell starts: each row, by id, read or refused.

    A row that does not load is kept as why, not dropped: the cell raises it the moment its call asks for that
    recording, exactly as a read of that one row would, and a row the cell never asks for costs it nothing.
    """

    #: Each row that loads as this build's :class:`EvalCassette`, by its id.
    readable: dict[str, EvalCassette]
    #: Each row that does not, by its id: why.
    unreadable: dict[str, str]


def _refusal_reasons(refused: ValidationError) -> list[str]:
    """Each distinct thing a model refused in a row, as ``field: what it said``, sorted — never the value itself."""
    return sorted(
        {
            f"{'.'.join(str(part) for part in error['loc']) or '(the document)'}: {error['msg']}"
            for error in refused.errors(include_url=False)
        }
    ) or [str(refused)]


class RefusedRow(NamedTuple):
    """One stored row the current model refused to reconstruct: what the row itself says it is, and why.

    The id and status are read off the raw row, since the model never accepted it: either may be absent, or not
    text, on a row broken badly enough, and is then ``None`` rather than a guess.
    """

    row_id: str | None
    """The ``id`` the row states, or ``None`` when it states none as text."""

    status: str | None
    """The ``status`` the row was stored with, or ``None`` when it states none as text."""

    refused: ValidationError
    """What the model said of it."""

    lease_expires_at: str | None = None
    """The ``lease.expires_at`` the row states, as stored, or ``None`` when it states none as text: for a caller that
    must tell whether some process may still hold the row (a delete of a run this build cannot read)."""

    lease_stated: bool = False
    """Whether the row states a lease at all (a ``lease`` that is not null), whether or not its expiry reads: a row
    with a lease this build cannot read is not a row with none."""

    @property
    def reasons(self) -> list[str]:
        """Each distinct refusal, as ``field: what the model said``, sorted; the refused value is never repeated."""
        return _refusal_reasons(self.refused)


class UnreadableRecordError(EvalServiceError):
    """A stored record this build's model refuses, met by a read that cannot answer without it (status 500).

    Raised in place of the model's ``ValidationError`` wherever a surface reads a record it cannot leave out, so the
    caller is refused as by any engine refusal, naming the record and why, rather than failing with the model's own
    error. The stored row is what is wrong, not the call: retrying it answers the same. A run's refusal is
    :class:`UnreadableRunError`, whose remedy names the listing and the delete for runs.
    """

    def __init__(self, message: str, *, remedy: str, details: dict[str, Any] | None = None):
        """Build a 500.

        :param message: Which record, in which scope, and why.
        :ptype message: str
        :param remedy: What the caller does about it.
        :ptype remedy: str
        :param details: Structured facts about the record, for a host's code.
        :ptype details: dict[str, Any] | None
        """
        super().__init__(message=message, status_code=500, details=details or {}, remedy=remedy)

    @classmethod
    def of_job_record(cls, kind: str, record_id: str, scope_id: str, refused: ValidationError) -> UnreadableRecordError:
        """The refusal of a job record's read, where the record or a run it names and settles with it did not read.

        For the lease machinery's reads, which keep raising ``ValidationError`` because the boot reclaim and the job
        loops tell a refused row from a lost race by it: a surface that reads a job record through one of them
        translates it here.

        :param kind: The job kind's name (``sweep``, ``launch``, ...).
        :ptype kind: str
        :param record_id: The record asked for.
        :ptype record_id: str
        :param scope_id: The scope read.
        :ptype scope_id: str
        :param refused: What the model said.
        :ptype refused: ValidationError
        :return: The error, its remedy :data:`~threetears.evals.kernel.remedies.UNREADABLE_JOB_RECORD`.
        :rtype: UnreadableRecordError
        """
        return cls(
            f"the {kind} job record {record_id!r} in scope {scope_id!r}, or a run it names, does not validate under "
            f"this build's model, as read or as written lost — {', '.join(_refusal_reasons(refused))}",
            remedy=UNREADABLE_JOB_RECORD,
            details={"kind": kind, "record_id": record_id},
        )


class UnreadableRunError(UnreadableRecordError):
    """A read that cannot leave a run out met one this build's model refuses (status 500).

    A run's own read, or a listing whose answer would be wrong without every run it matched — a comparison, a
    campaign's members, a cost — refuses rather than answer over the rows it could read. Raised by
    :class:`EvalStorage`'s run reads in place of the model's ``ValidationError``, so every surface refuses it as an
    engine refusal rather than failing with the model's own error. Its remedy is the class's own
    (:data:`~threetears.evals.kernel.remedies.FIND_UNREADABLE_RUNS`): the stored rows are what is wrong, not the
    call, so the caller's next step is to see which and why.

    :ivar run_ids: The ids the refused rows state, as read off the rows; empty when none stated one.
    """

    def __init__(self, message: str, *, run_ids: Sequence[str]):
        """Build a 500 naming the rows.

        :param message: Which rows, in which scope, and why.
        :ptype message: str
        :param run_ids: The ids the refused rows state.
        :ptype run_ids: Sequence[str]
        """
        self.run_ids = list(run_ids)
        super().__init__(message, remedy=FIND_UNREADABLE_RUNS, details={"run_ids": self.run_ids})

    @classmethod
    def of_rows(cls, scope_id: str, rows: Sequence[RefusedRow]) -> UnreadableRunError:
        """The refusal of a read that met ``rows``, each named with its reasons.

        :param scope_id: The scope read.
        :ptype scope_id: str
        :param rows: The rows the model refused, at least one.
        :ptype rows: Sequence[RefusedRow]
        :return: The error.
        :rtype: UnreadableRunError
        """
        named = "; ".join(f"{row.row_id or '(no id on the row)'}: {', '.join(row.reasons)}" for row in rows)
        return cls(
            f"{len(rows)} stored eval run(s) in scope {scope_id!r} cannot be read under this build's model, so this "
            f"read cannot answer without them — {named}",
            run_ids=[row.row_id for row in rows if row.row_id is not None],
        )

    @classmethod
    def of_run(cls, scope_id: str, run_id: str, refused: ValidationError) -> UnreadableRunError:
        """The refusal of a read of one named run, through a read that raises the model's own error for it.

        For the lease machinery's reads (:meth:`EvalStorage.load_job_record`,
        :meth:`EvalStorage.load_eval_run_with_etag`), which keep raising ``ValidationError`` because the boot reclaim
        and the job loops tell a refused row from a lost race by it: a surface that reads a run through one of them
        translates it here, so its caller is refused as by every other run read.

        :param scope_id: The scope read.
        :ptype scope_id: str
        :param run_id: The run asked for.
        :ptype run_id: str
        :param refused: What the model said, of the stored run or of writing it lost.
        :ptype refused: ValidationError
        :return: The error.
        :rtype: UnreadableRunError
        """
        return cls(
            f"stored eval run {run_id!r} in scope {scope_id!r} does not validate under this build's model, as read "
            f"or as written lost — {', '.join(_refusal_reasons(refused))}",
            run_ids=[run_id],
        )

    @classmethod
    def of_settling(cls, scope_id: str, refused: ValidationError) -> UnreadableRunError:
        """The refusal of a read whose runs read, but one of which a model refused while it was written lost.

        :param scope_id: The scope read.
        :ptype scope_id: str
        :param refused: What the model said.
        :ptype refused: ValidationError
        :return: The error.
        :rtype: UnreadableRunError
        """
        return cls(
            f"a stored eval run in scope {scope_id!r} read, but writing it lost was refused by this build's model, "
            f"so it cannot be read as every process reads it — {', '.join(_refusal_reasons(refused))}",
            run_ids=[],
        )


class RunsRead(NamedTuple):
    """A run listing that leaves out what it cannot read, and says so (:meth:`EvalStorage.query_eval_runs_isolating`)."""

    runs: list[EvalRun]
    """The matched runs that read, each settled as every read settles it."""

    unreadable: list[RefusedRow]
    """The matched rows the model refused, each with why, in the store's order."""


class EvalRunScan(NamedTuple):
    """What a scan of every run in one scope found (:meth:`EvalStorage.scan_eval_runs`): how many read, which did not."""

    readable: int
    """Rows the current model reconstructed."""

    unreadable: list[RefusedRow]
    """Rows it refused, each with why, in the store's order."""


class NonTerminalRunScan(NamedTuple):
    """What one scope's scan for non-terminal runs found.

    Two facts, because a scan that could not read part of what it matched is
    not the same as one that found less. ``unreadable`` is a **count**, not a
    list of ids, and the split is deliberate: the ids are logged at ERROR as
    each row fails, where identity belongs, while the count is what a caller
    can put in an arithmetic that has to add up. A row that cannot be
    reconstructed cannot be trusted to report its own id either, so making the
    list the authority would tie the tally to the least reliable field on the
    least reliable row.
    """

    runs: list[EvalRun]
    """Runs the scan matched AND could reconstruct."""

    unreadable: int
    """Rows the scan matched but could not reconstruct under today's model.

    Never silently zero-by-omission: these rows stay inside the caller's
    ``scanned`` arithmetic, because a run nothing can read is precisely the one
    an operator most needs counted.
    """


def save_document(
    repo: DocumentStore, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False
) -> None:
    """Upsert ``document`` through ``repo``, raising on any failure.

    The one write path for eval documents over a :class:`~threetears.evals.schema.store_port.DocumentStore`,
    shared by :class:`EvalStorage` and by any host store that keeps its own documents in the same
    store, so a failed write means the same thing whichever of them made it.

    There is no success flag to ignore: a write either lands or raises. A lost
    ``if_match`` race is a :class:`ConflictError`, because its remedy is the
    caller's — re-read the winner's document and re-apply — and every other failure
    is a :class:`StorageError`. A write that reported failure as ``False`` let a
    caller that never looked at the value report a success that did not happen.

    :param repo: The store to write through.
    :ptype repo: DocumentStore
    :param document: The document, carrying its ``id`` and ``doc_type``.
    :ptype document: dict[str, Any]
    :param if_match: The etag the stored document must still have, for a conditional write.
    :ptype if_match: str | None
    :param create_only: Write only if nothing is stored under the document's scope and id.
    :ptype create_only: bool
    :raises ConflictError: ``if_match`` was given and the stored document has moved on, or ``create_only`` was
        set and a document is stored there.
    :raises StorageError: Any other backend failure.
    """
    try:
        if create_only:
            repo.upsert(document, if_match=if_match, create_only=True)
        else:
            # Only a create names the precondition, so a write that needs none asks of the store exactly
            # what it always did.
            repo.upsert(document, if_match=if_match)
    except StoreConflict as e:
        raise ConflictError(
            f"{document.get('doc_type')} '{document.get('id')}' changed since it was read", remedy=REREAD_AND_RETRY
        ) from e
    # prawduct:ok-broad-except — DB write boundary; re-raised as a typed eval error
    except Exception as e:
        log.error(
            "Failed to upsert eval document: id=%s doc_type=%s — %s",
            document.get("id"),
            document.get("doc_type"),
            e,
        )
        raise StorageError(f"failed to write {document.get('doc_type')} '{document.get('id')}': {e}") from e


# --- the run side's named ports ---------------------------------------------------------------------
#
# Each is one area of :class:`EvalStorage`, stated as a protocol so a consumer names the area it
# reads rather than the whole store: a function typed ``DefinitionStore`` cannot reach a run. The
# engine satisfies every one with :class:`EvalStorage` over the one :class:`DocumentStore` an app
# implements; the methods carry EvalStorage's own semantics, documented there. A consumer whose reads
# are a small set across areas keeps a protocol of its own beside it (the analysis lenses, the
# judge's inputs, curation), on the same terms.


class JobStore(Protocol):
    """The run document's read-modify-write: what the job manager persists a run's status through.

    The smallest run-side port, and a part of :class:`RunStore`. See
    :func:`threetears.evals.run.run_document.update_eval_run` for the retry it supports, which is
    stated over these two methods so a test double drives the same policy production does.

    Positional parameters here and on every port below are positional-only, so an implementation
    is free to name them after whatever it partitions by; the engine passes them positionally.
    """

    def load_eval_run_with_etag(self, run_id: str, scope_id: str, /) -> tuple[EvalRun | None, str | None]:
        """See :meth:`EvalStorage.load_eval_run_with_etag`."""
        ...

    def save_eval_run(self, run: EvalRun, /, *, if_match: str | None = None) -> None:
        """See :meth:`EvalStorage.save_eval_run`."""
        ...


class JobRecordStore(Protocol):
    """A job record's read-modify-write: what a job's own guarded write of its record goes through.

    Generic over the record's model, because every job record — a run, a sweep, an analysis attempt — carries the
    same lease fields (:mod:`~threetears.evals.schema.job_lease`) and is written by its job the same way
    (:func:`~threetears.evals.kernel.job_leases.write_held_record`).
    """

    def load_job_record[R: EvalDocumentModel](
        self, model: type[R], record_id: str, scope_id: str, /
    ) -> tuple[R | None, str | None]:
        """See :meth:`EvalStorage.load_job_record`."""
        ...

    def save_job_record(self, document: dict[str, Any], /, *, if_match: str | None, create_only: bool = False) -> None:
        """See :meth:`EvalStorage.save_job_record`."""
        ...


class LeaseClock(Protocol):
    """The time job leases are stamped and compared in — one store-wide clock, so it names no scope."""

    def store_time(self) -> datetime:
        """See :meth:`EvalStorage.store_time`."""
        ...


class LeaseStore(JobRecordStore, LeaseClock, Protocol):
    """A job record's read-modify-write, a cancel asked of it, and the store's time: what job leases are kept through.

    Every job record is renewed, cancelled and written lost the same way (:mod:`threetears.evals.kernel.job_leases`).
    A run's results are here too: a run written lost has its completeness reconstructed from them in the same write.
    """

    def merge_job_record_fields(self, record_id: str, scope_id: str, fields: dict[str, Any], /) -> bool:
        """See :meth:`EvalStorage.merge_job_record_fields`."""
        ...

    def query_eval_results_by_run(self, run_id: str, scope_id: str, /) -> list[EvalResult]:
        """See :meth:`EvalStorage.query_eval_results_by_run`."""
        ...


class RunStore(JobStore, Protocol):
    """The eval runs of a scope: their documents, listings, stamps, archive flag and boot scan."""

    def load_eval_run(self, run_id: str, scope_id: str, /) -> EvalRun | None:
        """See :meth:`EvalStorage.load_eval_run`."""
        ...

    def query_eval_runs(
        self,
        scope_id: str,
        status: EvalRunStatus | None = None,
        *,
        elide_payload: frozenset[str] = frozenset(),
    ) -> list[EvalRun]:
        """See :meth:`EvalStorage.query_eval_runs`."""
        ...

    def load_eval_runs(
        self,
        run_ids: Sequence[str],
        scope_id: str,
        /,
        *,
        elide_payload: frozenset[str] = frozenset(),
    ) -> list[EvalRun]:
        """See :meth:`EvalStorage.load_eval_runs`."""
        ...

    def load_eval_run_stamps(self, run_ids: Sequence[str], scope_id: str, /) -> list[EvalRunStamp]:
        """See :meth:`EvalStorage.load_eval_run_stamps`."""
        ...

    def set_eval_run_archived(self, run_id: str, scope_id: str, /, *, archived: bool) -> bool:
        """See :meth:`EvalStorage.set_eval_run_archived`."""
        ...

    def query_non_terminal_eval_runs(self, scope_id: str, /) -> NonTerminalRunScan:
        """See :meth:`EvalStorage.query_non_terminal_eval_runs`."""
        ...

    def delete_eval_run(self, run_id: str, scope_id: str, /) -> bool:
        """See :meth:`EvalStorage.delete_eval_run`."""
        ...


class ResultStore(Protocol):
    """The cells a run recorded: each :class:`EvalResult` and the :class:`EvalTrace` beside it."""

    def save_eval_result(self, result: EvalResult, trace: EvalTrace | None = None, /) -> None:
        """See :meth:`EvalStorage.save_eval_result`."""
        ...

    def load_eval_result(self, result_id: str, scope_id: str, /) -> EvalResult | None:
        """See :meth:`EvalStorage.load_eval_result`."""
        ...

    def load_eval_result_with_etag(self, result_id: str, scope_id: str, /) -> tuple[EvalResult | None, str | None]:
        """See :meth:`EvalStorage.load_eval_result_with_etag`."""
        ...

    def replace_eval_result(self, result: EvalResult, /, *, if_match: str | None) -> None:
        """See :meth:`EvalStorage.replace_eval_result`."""
        ...

    def load_eval_trace(self, result_id: str, scope_id: str, /) -> EvalTrace | None:
        """See :meth:`EvalStorage.load_eval_trace`."""
        ...

    def query_eval_results(
        self,
        scope_id: str,
        /,
        *,
        run_id: str | None = None,
        test_case_id: str | None = None,
        model: str | None = None,
    ) -> list[EvalResult]:
        """See :meth:`EvalStorage.query_eval_results`."""
        ...

    def query_eval_results_by_run(self, run_id: str, scope_id: str, /) -> list[EvalResult]:
        """See :meth:`EvalStorage.query_eval_results_by_run`."""
        ...

    def delete_eval_result(self, result_id: str, scope_id: str, /) -> bool:
        """See :meth:`EvalStorage.delete_eval_result`."""
        ...


class RunRecordStore(RunStore, ResultStore, Protocol):
    """A run together with its cells: for an operation that reads one and rewrites the other.

    Cancelling a run counts the cells it recorded before stamping it, and the boot-time sweep of
    abandoned runs cancels each one it finds, so both need the two areas at once.
    """


class DefinitionStore(Protocol):
    """What runs are launched FROM: templates, their test cases, rubric dimensions and judge configs."""

    def save_template(self, template: EvalTemplate, /) -> None:
        """See :meth:`EvalStorage.save_template`."""
        ...

    def load_template(self, template_id: str, scope_id: str, /) -> EvalTemplate | None:
        """See :meth:`EvalStorage.load_template`."""
        ...

    def load_template_by_name(self, name: str, scope_id: str, /) -> EvalTemplate | None:
        """See :meth:`EvalStorage.load_template_by_name`."""
        ...

    def query_templates(
        self,
        scope_id: str,
        /,
        *,
        archived: bool | None = None,
        required_tool: str | None = None,
        universal: bool | None = None,
    ) -> list[EvalTemplate]:
        """See :meth:`EvalStorage.query_templates`."""
        ...

    def delete_template(self, template_id: str, scope_id: str, /) -> bool:
        """See :meth:`EvalStorage.delete_template`."""
        ...

    def save_test_case(self, test_case: EvalTestCase, /, *, if_match: str | None = None) -> None:
        """See :meth:`EvalStorage.save_test_case`."""
        ...

    def load_test_case(self, test_case_id: str, scope_id: str, /) -> EvalTestCase | None:
        """See :meth:`EvalStorage.load_test_case`."""
        ...

    def load_test_case_with_etag(self, test_case_id: str, scope_id: str, /) -> tuple[EvalTestCase | None, str | None]:
        """See :meth:`EvalStorage.load_test_case_with_etag`."""
        ...

    def query_test_cases(self, scope_id: str, /, *, template_id: str | None = None) -> list[EvalTestCase]:
        """See :meth:`EvalStorage.query_test_cases`."""
        ...

    def load_test_cases_by_ids(self, test_case_ids: list[str], scope_id: str, /) -> list[EvalTestCase]:
        """See :meth:`EvalStorage.load_test_cases_by_ids`."""
        ...

    def load_case_strata(self, test_case_ids: Sequence[str], scope_id: str, /) -> list[EvalCaseStratum]:
        """See :meth:`EvalStorage.load_case_strata`."""
        ...

    def delete_test_case(self, test_case_id: str, scope_id: str, /) -> bool:
        """See :meth:`EvalStorage.delete_test_case`."""
        ...

    def save_rubric_dim(self, dim: CatalogRubricDim, /) -> None:
        """See :meth:`EvalStorage.save_rubric_dim`."""
        ...

    def load_rubric_dim(self, dim_id: str, scope_id: str, /) -> CatalogRubricDim | None:
        """See :meth:`EvalStorage.load_rubric_dim`."""
        ...

    def load_active_rubric_dim(self, key: str, scope_id: str, /) -> CatalogRubricDim | None:
        """See :meth:`EvalStorage.load_active_rubric_dim`."""
        ...

    def query_rubric_dims(
        self,
        scope_id: str,
        /,
        *,
        axis: str | None = None,
        universal: bool | None = None,
        archived: bool | None = None,
    ) -> list[CatalogRubricDim]:
        """See :meth:`EvalStorage.query_rubric_dims`."""
        ...

    def delete_rubric_dim(self, dim_id: str, scope_id: str, /) -> bool:
        """See :meth:`EvalStorage.delete_rubric_dim`."""
        ...

    def save_rubric_dim_tombstone(self, tombstone: RubricDimTombstone, /) -> None:
        """See :meth:`EvalStorage.save_rubric_dim_tombstone`."""
        ...

    def query_rubric_dim_tombstones(self, scope_id: str, /) -> list[RubricDimTombstone]:
        """See :meth:`EvalStorage.query_rubric_dim_tombstones`."""
        ...

    def save_judge_config(self, config: JudgeConfig, /) -> None:
        """See :meth:`EvalStorage.save_judge_config`."""
        ...

    def load_judge_config(self, config_id: str, scope_id: str, /) -> JudgeConfig | None:
        """See :meth:`EvalStorage.load_judge_config`."""
        ...

    def load_active_judge_config(self, rubric_dim_id: str, scope_id: str, /) -> JudgeConfig | None:
        """See :meth:`EvalStorage.load_active_judge_config`."""
        ...

    def query_judge_configs(
        self,
        scope_id: str,
        /,
        *,
        rubric_dim_id: str | None = None,
        archived: bool | None = None,
    ) -> list[JudgeConfig]:
        """See :meth:`EvalStorage.query_judge_configs`."""
        ...

    def delete_judge_config(self, config_id: str, scope_id: str, /) -> bool:
        """See :meth:`EvalStorage.delete_judge_config`."""
        ...

    def save_judge_config_tombstone(self, tombstone: JudgeConfigTombstone, /) -> None:
        """See :meth:`EvalStorage.save_judge_config_tombstone`."""
        ...

    def query_judge_config_tombstones(self, scope_id: str, /) -> list[JudgeConfigTombstone]:
        """See :meth:`EvalStorage.query_judge_config_tombstones`."""
        ...


class CaseSetStore(Protocol):
    """Named, versioned case sets, and the test cases they name — what minting and resolving a set reads."""

    def save_case_set(self, case_set: CaseSet, /) -> None:
        """See :meth:`EvalStorage.save_case_set`."""
        ...

    def load_case_set(self, name: str, version: int, scope_id: str, /) -> CaseSet | None:
        """See :meth:`EvalStorage.load_case_set`."""
        ...

    def query_case_sets(self, scope_id: str, /, *, name: str | None = None) -> list[CaseSet]:
        """See :meth:`EvalStorage.query_case_sets`."""
        ...

    def load_test_cases_by_ids(self, test_case_ids: list[str], scope_id: str, /) -> list[EvalTestCase]:
        """See :meth:`EvalStorage.load_test_cases_by_ids`."""
        ...

    def load_template(self, template_id: str, scope_id: str, /) -> EvalTemplate | None:
        """See :meth:`EvalStorage.load_template`; read for the name a set's template and a new version's share."""
        ...


class CassetteStore(Protocol):
    """The recordings a capture run made and a replay run is served."""

    def save_cassette(self, cassette: EvalCassette, /) -> None:
        """See :meth:`EvalStorage.save_cassette`."""
        ...

    def get_cassette(self, key: CassetteKey, scope_id: str, /) -> EvalCassette | None:
        """See :meth:`EvalStorage.get_cassette`."""
        ...

    def list_cassettes_for_template(
        self, template_id: str, scope_id: str, /, *, limit: int | None = None
    ) -> list[EvalCassette]:
        """See :meth:`EvalStorage.list_cassettes_for_template`."""
        ...

    def list_case_cassettes(
        self, *, corpus_id: str, template_id: str, test_case_id: str, scope_id: str
    ) -> list[EvalCassette]:
        """See :meth:`EvalStorage.list_case_cassettes`."""
        ...

    def read_case_cassettes(
        self, *, corpus_id: str, template_id: str, test_case_id: str, scope_id: str
    ) -> CaseRecordings:
        """See :meth:`EvalStorage.read_case_cassettes`."""
        ...

    def delete_case_cassettes(self, *, corpus_id: str, template_id: str, test_case_id: str, scope_id: str) -> int:
        """See :meth:`EvalStorage.delete_case_cassettes`."""
        ...


class EvalStorage:
    """Storage for eval documents over one document store.

    Load paths go through each model's own ``from_dict``, which validates exactly as construction
    does: a stored document with a field the model does not declare, without one it requires, or
    written under another ``schema_version`` is refused, never coerced or filtered — see the module
    docstring.
    """

    def __init__(self, store: DocumentStore, *, host_doc_types: Sequence[str] = ()):
        """Initialize storage over the store the host supplies.

        :param store: The one store every eval document is read from and written to.
        :ptype store: DocumentStore
        :param host_doc_types: The ``doc_type`` values of documents the host keeps in the same store.
            This class never reads or writes them; :meth:`nuke_all_eval_data` sweeps them, so a
            wipe leaves nothing of the host's behind. Empty means the host keeps none there,
            which is a statement about the host rather than a default standing in for one.
        :ptype host_doc_types: Sequence[str]
        :raises ValueError: A host type is also one of :data:`EVAL_DOC_TYPES` — the engine's own, which
            the host does not get to declare twice.
        """
        claimed = set(host_doc_types) & set(EVAL_DOC_TYPES)
        if claimed:
            raise ValueError(f"host_doc_types names the engine's own doc types: {sorted(claimed)}")
        self._document_store = store
        self._host_doc_types = tuple(host_doc_types)

    @property
    def _store(self) -> DocumentStore:
        """The store, for a call about to be made on it: refused on a thread running an event loop.

        Every read and write here reaches the store through this property, so no method can call it
        on the loop, whichever store the host handed in.

        :raises StoreCalledOnEventLoop: The calling thread is running an event loop.
        """
        refuse_on_event_loop()
        return self._document_store

    @staticmethod
    def _hydrate[M: EvalBaseModel](model_cls: type[M], data: dict[str, Any]) -> M:
        """Reconstruct one model from a stored document.

        The document arrives with the storage stamp already removed — that is the
        port's contract.
        """
        return model_cls.from_dict(data)

    def _hydrate_all[M: EvalBaseModel](self, model_cls: type[M], rows: list[dict[str, Any]]) -> list[M]:
        """Reconstruct a list of models from stored documents (see :meth:`_hydrate`), each job record settled."""
        return self._settled(model_cls, [self._hydrate(model_cls, row) for row in rows])

    def _settled[M: EvalBaseModel](self, model_cls: type[M], records: list[M]) -> list[M]:
        """``records`` as every process reads them: a job record whose lease lapsed before it ended, written lost.

        The one settling read every surface goes through — a run's get and its listings, a sweep's, an analysis
        attempt's, and any other job record a reader loads: a record of a leased kind
        (:func:`~threetears.evals.kernel.job_leases.leased_kind_of`) that has not ended, and whose lease has
        lapsed or which carries none, is settled exactly as ``job_poll`` settles it
        (:func:`~threetears.evals.kernel.job_leases.settle_job_record`), so a process that died holding a job
        does not leave it reading ``running`` on any surface, and no count of what is still running counts it.
        A record that has ended, or whose holder still renews it, is returned as read; only a lapsed one costs a
        read and a conditional write. The lease machinery's own reads (:meth:`load_job_record`, the reads that
        return an etag) and the boot reclaim's scan (:meth:`query_non_terminal_eval_runs`) are not settled here:
        they are what settling is made of, and the reclaim counts what it writes lost itself.

        **A record another writer keeps moving never breaks the read.** Each is settled on its own
        (:func:`~threetears.evals.kernel.job_leases.settle_tolerantly`): one whose lost write keeps losing its race is
        returned as stored and logged, and the rest of the list is still read; one this build cannot read or write
        lost raises its ``ValidationError``, as reading it does. Settling a run reads nothing past the run that can
        fail it: a result that cannot be read leaves the lost run's completeness unknown, named in its
        ``error_details``, rather than refusing the write.
        """
        from threetears.evals.kernel.job_leases import lease_lapsed, leased_kind_of, settle_tolerantly

        kind = leased_kind_of(model_cls) if issubclass(model_cls, EvalDocumentModel) else None
        if kind is None:
            return records
        open_records = [index for index, record in enumerate(records) if not kind.ended(record)]
        if not open_records:
            return records
        now = self.store_time()
        settled = list(records)
        for index in open_records:
            record: Any = records[index]
            if lease_lapsed(record, now):
                # Per record, as the boot reclaim settles its rows: one another writer keeps moving reads as stored.
                current = settle_tolerantly(self, kind, record.id, record.scope_id)
                if current is not None:
                    settled[index] = cast(M, current)
        return settled

    @staticmethod
    def _of_type[M: EvalBaseModel](model_cls: type[M], data: dict[str, Any] | None) -> dict[str, Any] | None:
        """``data`` when it is a document of ``model_cls``'s type, else ``None`` — the one id-read type check.

        A point read is by id alone (:meth:`~threetears.evals.schema.store_port.DocumentStore.get`),
        and ids of different types share a scope: a run's id, its results' ids and each result's
        ``<id>:trace`` all resolve there. A document of another type is not the document asked for, so
        it reads as none rather than reaching the wrong model's validator: every load by id goes through
        here, and every caller's absent branch (``NotFoundError`` at a read surface) is the answer. A
        document with no ``doc_type`` at all is left to the model's own validation.
        """
        if data is None:
            return None
        stored = data.get("doc_type")
        expected = model_cls.model_fields["doc_type"].default
        return None if stored is not None and stored != expected else data

    def _save(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Upsert ``document`` with :func:`save_document`, which raises on any failure."""
        save_document(self._store, document, if_match=if_match, create_only=create_only)

    # =========================================================================
    # Job records — the lease fields every job record carries (JobRecordStore, LeaseStore)
    # =========================================================================

    def load_job_record[R: EvalDocumentModel](
        self, model: type[R], record_id: str, scope_id: str, /
    ) -> tuple[R | None, str | None]:
        """Load one job record of ``model``'s type plus the token a conditional write of it must present.

        :param model: The record's model — :class:`EvalRun`, :class:`EvalSweep` or :class:`EvalAnalysisAttempt`.
        :ptype model: type[R]
        :param record_id: Its id.
        :ptype record_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :return: ``(record, etag)``, or ``(None, None)`` when no record of that type resolves there (:meth:`_of_type`).
        :rtype: tuple[R | None, str | None]
        """
        data, etag = self._store.get_with_etag(record_id, scope_id)
        data = self._of_type(model, data)
        if data is None:
            return None, None
        return self._hydrate(model, data), etag

    def save_job_record(self, document: dict[str, Any], /, *, if_match: str | None, create_only: bool = False) -> None:
        """Write a job record whole — as its model's ``to_dict`` writes it — conditionally when ``if_match`` is given.

        A document rather than a model, so one write serves every job record type; the record carries its own
        scope. The lease machinery writes only records it read whole (:meth:`load_job_record`), so a run here never
        lacks payload a listing left out. A job's first write is ``create_only``: it starts the job, and a record
        already stored under its id is a job already started, never one to write over.

        :raises ConflictError: ``if_match`` was given and the stored record has moved on, or ``create_only`` was given
            and a record is already stored under the id.
        :raises StorageError: Any other write failure.
        """
        self._save(document, if_match=if_match, create_only=create_only)

    def merge_job_record_fields(self, record_id: str, scope_id: str, fields: dict[str, Any], /) -> bool:
        """Set top-level fields on a job record in place — a cancel request — without reading or re-sending it.

        Unconditional, as :meth:`~threetears.evals.schema.store_port.DocumentStore.merge_fields` is, and it mints
        a new etag, so the holder's next conditional write re-reads and sees the request.

        :param record_id: The record's id.
        :ptype record_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :param fields: The fields and their new values.
        :ptype fields: dict[str, Any]
        :return: ``True`` when the record was written; ``False`` when there is no such document.
        :rtype: bool
        :raises StorageError: The write failed.
        """
        try:
            return self._store.merge_fields(record_id, scope_id, fields)
        # prawduct:ok-broad-except — DB write boundary; re-raised as a typed eval error
        except Exception as e:
            log.error("Failed to merge %s into job record %s (scope %s) — %s", sorted(fields), record_id, scope_id, e)
            raise StorageError(f"failed to write job record '{record_id}': {e}") from e

    def store_time(self) -> datetime:
        """The time leases are stamped and compared in: the store's own when it keeps a clock, this process's otherwise.

        A store offering :class:`~threetears.evals.schema.store_port.StoreClock` is asked, so every process sharing it
        reads one clock; any other store is read on this process's clock, in UTC.

        :return: An aware datetime, in UTC.
        :rtype: datetime
        :raises StorageError: The store's clock answered a naive datetime, which names no instant.
        """
        if not isinstance(self._store, StoreClock):
            return datetime.now(UTC)
        now = self._store.now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise StorageError(
                f"the store's clock answered {now.isoformat()}, a time with no zone, which names no instant"
            )
        return now.astimezone(UTC)

    def _load[M: EvalBaseModel](self, model_cls: type[M], doc_id: str, scope_id: str) -> M | None:
        """One document by id within a scope, hydrated and, a job record, settled (:meth:`_settled`), or ``None``
        when no document of its type resolves there."""
        data = self._of_type(model_cls, self._store.get(doc_id, scope_id))
        if data is None:
            return None
        (loaded,) = self._settled(model_cls, [self._hydrate(model_cls, data)])
        return loaded

    # =========================================================================
    # EvalTemplate
    # =========================================================================

    def save_template(self, template: EvalTemplate) -> None:
        """Persist a template in the scope it names."""
        self._save(template.to_dict())

    def load_template(self, template_id: str, scope_id: str) -> EvalTemplate | None:
        """Load a template by id within a scope."""
        return self._load(EvalTemplate, template_id, scope_id)

    def load_template_by_name(self, name: str, scope_id: str) -> EvalTemplate | None:
        """Load the template a ``name`` resolves to within a scope: its current revision.

        A name holds one record per revision (``EvalTemplate.revision``; a definition reset writes the next as a new
        record), so it resolves to its newest non-archived revision, and, when every revision is archived, to its
        newest one. An older revision stays readable by id (:meth:`load_template`). Newest is the highest
        ``revision``, then the latest ``created_at``, then the id, so two records an older writer left under one name
        resolve the same way on every read.
        """
        revisions = self._hydrate_all(EvalTemplate, self._store.by_doc_type("eval_template", scope_id, name=name))
        return current_revision(revisions)

    def query_templates(
        self,
        scope_id: str,
        *,
        archived: bool | None = None,
        required_tool: str | None = None,
        universal: bool | None = None,
    ) -> list[EvalTemplate]:
        """Query templates in a scope.

        ``required_tool`` filters Python-side (template counts are small).
        ``archived=None`` returns both archived and active. ``universal=None``
        returns both universal (boundary-battery) and subject-scoped templates.
        """
        field_eq: dict[str, Any] = {}
        if archived is not None:
            field_eq["archived"] = archived
        if universal is not None:
            field_eq["universal"] = universal

        templates = self._hydrate_all(EvalTemplate, self._store.by_doc_type("eval_template", scope_id, **field_eq))
        if required_tool:
            templates = [t for t in templates if required_tool in t.tools_required]
        return templates

    def delete_template(self, template_id: str, scope_id: str) -> bool:
        """Delete a template by id within a scope."""
        return self._store.delete(template_id, scope_id)

    # =========================================================================
    # JudgeConfig — versioned judge prompt + model + decoding params
    # =========================================================================

    def save_judge_config(self, config: JudgeConfig) -> None:
        """Persist a judge config in the scope it names."""
        self._save(config.to_dict())

    def load_judge_config(self, config_id: str, scope_id: str) -> JudgeConfig | None:
        """Load a judge config by id within a scope."""
        return self._load(JudgeConfig, config_id, scope_id)

    def load_active_judge_config(self, rubric_dim_id: str, scope_id: str) -> JudgeConfig | None:
        """Return the active judge config for a rubric dim in a scope, or ``None``.

        Active = the non-archived config with the latest ``created_at`` for this
        ``rubric_dim_id``. That's the versioning rule: authoring a new config supersedes the
        prior one for live runs while the old one stays queryable for historical-run
        interpretation. ``None`` means the judge service falls back to its built-in default
        prompt for the dim. ``created_at`` is an ISO-8601 string, ordered lexicographically.
        """
        items = self._store.by_doc_type(
            "judge_config",
            scope_id,
            rubric_dim_id=rubric_dim_id,
            archived=False,
            order_by="created_at",
            descending=True,
            limit=1,
        )
        if not items:
            return None
        return self._hydrate(JudgeConfig, items[0])

    def query_judge_configs(
        self,
        scope_id: str,
        *,
        rubric_dim_id: str | None = None,
        archived: bool | None = None,
    ) -> list[JudgeConfig]:
        """Query judge configs in a scope, newest first.

        ``rubric_dim_id`` and ``archived`` are optional filters; both ``None``
        returns every config in the scope (active and archived, all dims).
        """
        field_eq: dict[str, Any] = {}
        if rubric_dim_id is not None:
            field_eq["rubric_dim_id"] = rubric_dim_id
        if archived is not None:
            field_eq["archived"] = archived
        items = self._store.by_doc_type("judge_config", scope_id, order_by="created_at", descending=True, **field_eq)
        return self._hydrate_all(JudgeConfig, items)

    def delete_judge_config(self, config_id: str, scope_id: str) -> bool:
        """Delete a judge config by id within a scope."""
        return self._store.delete(config_id, scope_id)

    def save_judge_config_tombstone(self, tombstone: JudgeConfigTombstone) -> None:
        """Persist the record that a judge config slot was deleted, in the scope it names."""
        self._save(tombstone.to_dict())

    def query_judge_config_tombstones(self, scope_id: str) -> list[JudgeConfigTombstone]:
        """Every judge config tombstone in a scope, newest first — the slots a seed must not write back."""
        return self._hydrate_all(
            JudgeConfigTombstone,
            self._store.by_doc_type("judge_config_tombstone", scope_id, order_by="deleted_at", descending=True),
        )

    # =========================================================================
    # CatalogRubricDim — shared, versioned, reusable rubric dimensions
    # =========================================================================

    def save_rubric_dim(self, dim: CatalogRubricDim) -> None:
        """Persist a catalog rubric dim in the scope it names."""
        self._save(dim.to_dict())

    def load_rubric_dim(self, dim_id: str, scope_id: str) -> CatalogRubricDim | None:
        """Load a catalog rubric dim by id within a scope."""
        return self._load(CatalogRubricDim, dim_id, scope_id)

    def load_active_rubric_dim(self, key: str, scope_id: str) -> CatalogRubricDim | None:
        """Return the active catalog rubric dim for a key in a scope, or ``None``.

        Active = the non-archived record with the latest ``created_at`` for this
        ``key``. That's the versioning rule: re-authoring a dim writes a new record (new id,
        same key) that supersedes the prior one while the old one stays queryable for
        historical-run / calibration interpretation. ``None`` means no active dim for that key.
        """
        items = self._store.by_doc_type(
            "rubric_dim",
            scope_id,
            key=key,
            archived=False,
            order_by="created_at",
            descending=True,
            limit=1,
        )
        if not items:
            return None
        return self._hydrate(CatalogRubricDim, items[0])

    def query_rubric_dims(
        self,
        scope_id: str,
        *,
        axis: str | None = None,
        universal: bool | None = None,
        archived: bool | None = None,
    ) -> list[CatalogRubricDim]:
        """Query catalog rubric dims in a scope, newest first.

        ``axis`` / ``universal`` / ``archived`` filter in the store. All ``None`` returns every dim
        in the scope (active and archived, all axes).
        """
        field_eq: dict[str, Any] = {}
        if axis is not None:
            field_eq["axis"] = axis
        if universal is not None:
            field_eq["universal"] = universal
        if archived is not None:
            field_eq["archived"] = archived
        return self._hydrate_all(
            CatalogRubricDim,
            self._store.by_doc_type("rubric_dim", scope_id, order_by="created_at", descending=True, **field_eq),
        )

    def delete_rubric_dim(self, dim_id: str, scope_id: str) -> bool:
        """Delete a catalog rubric dim by id within a scope."""
        return self._store.delete(dim_id, scope_id)

    def save_rubric_dim_tombstone(self, tombstone: RubricDimTombstone) -> None:
        """Persist the record that a rubric dim key was deleted, in the scope it names."""
        self._save(tombstone.to_dict())

    def query_rubric_dim_tombstones(self, scope_id: str) -> list[RubricDimTombstone]:
        """Every rubric dim tombstone in a scope, newest first — the keys a seed must not write back."""
        return self._hydrate_all(
            RubricDimTombstone,
            self._store.by_doc_type("rubric_dim_tombstone", scope_id, order_by="deleted_at", descending=True),
        )

    # =========================================================================
    # CaseSet — named, versioned, append-only case lists
    # =========================================================================

    def save_case_set(self, case_set: CaseSet) -> None:
        """Store a new version of a case set, refusing one whose ``(scope, name, version)`` exists.

        Append-only: a stored version is never rewritten, so a name and version mean one list of cases for as
        long as the store holds it. Changing a set is minting ``version + 1``
        (:func:`~threetears.evals.run.case_sets.mint_case_set`). The check reads before it writes; the store
        port has no create-if-absent write, so two writers minting the same version at the same instant can
        both pass it, and the later write lands — the window a host serialises minting against if it allows
        concurrent writers to one name.

        :raises ConflictError: That version of that set is already stored.
        """
        if self._store.get(case_set.id, case_set.scope_id) is not None:
            raise ConflictError(
                f"case set {case_set.name!r} v{case_set.version} already exists in scope {case_set.scope_id!r}, and a "
                "case set is append-only",
                remedy="mint the next version with case_set_mint instead of rewriting this one",
            )
        self._save(case_set.to_dict())

    def load_case_set(self, name: str, version: int, scope_id: str) -> CaseSet | None:
        """One version of a case set by name, or ``None`` when that version is not stored in the scope."""
        return self._load(CaseSet, case_set_doc_id(name, version), scope_id)

    def query_case_sets(self, scope_id: str, *, name: str | None = None) -> list[CaseSet]:
        """Every case set in a scope (or every version of one name), newest version first within a name."""
        field_eq: dict[str, Any] = {} if name is None else {"name": name}
        sets = self._hydrate_all(CaseSet, self._store.by_doc_type("case_set", scope_id, **field_eq))
        return sorted(sets, key=lambda case_set: (case_set.name, -case_set.version))

    # =========================================================================
    # EvalSweep — a multi-arm launch's own record
    # =========================================================================

    def save_sweep(self, sweep: EvalSweep, *, if_match: str | None = None) -> None:
        """Persist a sweep's record in the scope it names; conditionally when ``if_match`` is given.

        :raises ConflictError: ``if_match`` was given and the stored record has moved on.
        :raises StorageError: Any other write failure.
        """
        self._save(sweep.to_dict(), if_match=if_match)

    def load_sweep(self, sweep_id: str, scope_id: str) -> EvalSweep | None:
        """Load a sweep's record by id within a scope."""
        return self._load(EvalSweep, sweep_id, scope_id)

    # =========================================================================
    # EvalCampaign — the analysis hub
    # =========================================================================

    def save_campaign(self, campaign: EvalCampaign) -> None:
        """Persist a campaign in the scope it names."""
        self._save(campaign.to_dict())

    def load_campaign(self, campaign_id: str, scope_id: str) -> EvalCampaign | None:
        """Load a campaign by id within a scope."""
        return self._load(EvalCampaign, campaign_id, scope_id)

    def list_campaigns(
        self,
        scope_id: str,
        *,
        subject_id: str | None = None,
        behavior: str | None = None,
        archived: bool | None = None,
    ) -> list[EvalCampaign]:
        """Query campaigns in a scope, newest first.

        ``subject_id`` / ``behavior`` / ``archived`` are optional
        equality filters; all ``None`` returns every campaign in the scope.
        """
        field_eq: dict[str, Any] = {}
        if subject_id is not None:
            field_eq["subject_id"] = subject_id
        if behavior is not None:
            field_eq["behavior"] = behavior
        if archived is not None:
            field_eq["archived"] = archived
        items = self._store.by_doc_type("eval_campaign", scope_id, order_by="created_at", descending=True, **field_eq)
        return self._hydrate_all(EvalCampaign, items)

    # =========================================================================
    # EvalAnalysis / EvalInsight
    # Generated, stored analysis of a campaign + the insights it mints.
    # =========================================================================

    def save_analysis(self, analysis: EvalAnalysis) -> None:
        """Persist an analysis in the scope it names."""
        self._save(analysis.to_dict())

    def load_analysis(self, analysis_id: str, scope_id: str) -> EvalAnalysis | None:
        """Load an analysis by id within a scope."""
        return self._load(EvalAnalysis, analysis_id, scope_id)

    def analysis_archived(self, analysis_id: str, scope_id: str) -> bool | None:
        """Whether one stored analysis is archived, read off the document without hydrating it.

        Retraction-by-archive needs this one flag, and it is asked on every bundle assembly and
        every insight listing. Hydrating the whole analysis to answer it would make one stored
        document that a newer validator rejects abort both reads — including the listing an
        operator uses to find the insight to delete. A document that does not resolve answers
        ``None``; one with no ``archived`` field was never archived.

        :param analysis_id: The analysis to ask about.
        :ptype analysis_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :return: ``True`` or ``False`` for a stored analysis, ``None`` when none resolves.
        :rtype: bool | None
        """
        data = self._of_type(EvalAnalysis, self._store.get(analysis_id, scope_id))
        if data is None:
            return None
        return data.get("archived") is True

    def list_analyses_by_campaign(self, campaign_id: str, scope_id: str) -> list[EvalAnalysis]:
        """Return every analysis attached to a campaign, newest first."""
        items = self._store.by_doc_type(
            "eval_analysis",
            scope_id,
            campaign_id=campaign_id,
            order_by="created_at",
            descending=True,
        )
        return self._hydrate_all(EvalAnalysis, items)

    def delete_analysis(self, analysis_id: str, scope_id: str) -> bool:
        """Delete an analysis by id within a scope."""
        return self._store.delete(analysis_id, scope_id)

    def save_analysis_attempt(self, attempt: EvalAnalysisAttempt, *, if_match: str | None = None) -> None:
        """Persist one generation attempt's record in the scope it names; conditionally when ``if_match`` is given.

        :raises ConflictError: ``if_match`` was given and the stored record has moved on.
        :raises StorageError: Any other write failure.
        """
        self._save(attempt.to_dict(), if_match=if_match)

    def load_generation_slot_with_etag(
        self, campaign_id: str, scope_id: str
    ) -> tuple[EvalGenerationSlot | None, str | None]:
        """Load a campaign's generation slot plus the token a takeover must present; ``(None, None)`` when unclaimed."""
        return self.load_job_record(EvalGenerationSlot, generation_slot_id(campaign_id), scope_id)

    def claim_generation_slot(self, slot: EvalGenerationSlot, *, if_match: str | None) -> None:
        """Claim a campaign's generation slot: created when ``if_match`` is None, taken over with it otherwise.

        Never unconditional: a slot that does not exist is created with a create-only write, so of two processes
        creating it at once exactly one lands.

        :raises ConflictError: Another start claimed the slot first; re-read it.
        :raises StorageError: Any other write failure.
        """
        if if_match is None:
            self._save(slot.to_dict(), create_only=True)
        else:
            self._save(slot.to_dict(), if_match=if_match)

    def list_analysis_attempts_by_campaign(self, campaign_id: str, scope_id: str) -> list[EvalAnalysisAttempt]:
        """Return every generation attempt recorded against a campaign, newest first."""
        items = self._store.by_doc_type(
            "eval_analysis_attempt",
            scope_id,
            campaign_id=campaign_id,
            order_by="created_at",
            descending=True,
        )
        return self._hydrate_all(EvalAnalysisAttempt, items)

    def save_insight(self, insight: EvalInsight) -> None:
        """Persist an insight in the scope it names."""
        self._save(insight.to_dict())

    def query_insights(
        self,
        scope_id: str,
        *,
        subject_id: str | None = None,
        source_campaign_id: str | None = None,
    ) -> list[EvalInsight]:
        """Query insights in a scope, newest first.

        ``subject_id`` / ``source_campaign_id`` are optional equality filters; both
        ``None`` returns every insight in the scope.

        **The insight's free-prose ``scope`` field is deliberately not a parameter here.** It
        matches by substring for the reason ``threetears.evals.analysis.service.list_insights``
        gives — it is free prose, so an exact match is not a filter an operator can type — and
        ``by_doc_type`` takes only ``field_eq``, which has no substring channel. Offering the name
        here with equality semantics is what made every hand-formed argument return an empty list.

        **This read is deliberately unlimited, and something depends on that.**
        ``by_doc_type`` takes ``limit``; this caller does not pass it, and
        ``threetears.evals.analysis.service.list_insights`` filters that prose by substring over
        what comes back. Paging here silently turns that into "matches within the first page"
        and reports a filtered miss as an absence — the exact defect the substring filter was
        added to end. Whoever bounds this read owes that filter a substring channel in the same
        change.

        :param scope_id: The scope whose insights to read.
        :ptype scope_id: str
        :param subject_id: Optional equality filter on the subject.
        :ptype subject_id: str | None
        :param source_campaign_id: Optional equality filter on the minting campaign.
        :ptype source_campaign_id: str | None
        :return: The matching insights, newest observation first — all of them.
        :rtype: list[EvalInsight]
        """
        field_eq: dict[str, Any] = {}
        if subject_id is not None:
            field_eq["subject_id"] = subject_id
        if source_campaign_id is not None:
            field_eq["source_campaign_id"] = source_campaign_id
        items = self._store.by_doc_type("eval_insight", scope_id, order_by="observed_at", descending=True, **field_eq)
        return self._hydrate_all(EvalInsight, items)

    def load_insight(self, insight_id: str, scope_id: str) -> EvalInsight | None:
        """Load an insight by id within a scope."""
        return self._load(EvalInsight, insight_id, scope_id)

    def delete_insight(self, insight_id: str, scope_id: str) -> bool:
        """Delete an insight by id within a scope."""
        return self._store.delete(insight_id, scope_id)

    # =========================================================================
    # CalibrationRating
    # =========================================================================

    def save_calibration_rating(self, rating: CalibrationRating) -> None:
        """Persist a rating in the scope it names, replacing that rater's earlier rating of the same thing.

        The replacement is the id's doing, not this method's: a rating's id is derived from its
        result, dimension, rater and kind of rater, so the upsert lands on the earlier rating's row — and a
        person and an agent of one name rating the same thing are two rows.
        """
        self._save(rating.to_dict())

    def query_calibration_ratings(
        self,
        scope_id: str,
        *,
        run_id: str | None = None,
        result_id: str | None = None,
        label_key: LabelKey | None = None,
    ) -> list[CalibrationRating]:
        """Ratings in a scope, oldest first, optionally narrowed by run, by result and by what was read.

        Unlimited, like every agreement input: a paged read would compute agreement over the first
        page and report it as the dimension's.

        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param run_id: Optional equality filter on the rated result's run.
        :ptype run_id: str | None
        :param result_id: Optional equality filter on the rated result.
        :ptype result_id: str | None
        :param label_key: Optional filter on the output and criterion the rating was given on (#628): every
            label of a byte-identical judged output on the same criterion, whichever result it was given
            on — the read a frozen judge case makes (build the key with
            :func:`~threetears.evals.schema.models.label_key_of`). A rating with no key never matches.
        :ptype label_key: LabelKey | None
        :return: The matching ratings, ordered by when they were rated.
        :rtype: list[CalibrationRating]
        """
        field_eq: dict[str, Any] = {}
        if run_id is not None:
            field_eq["run_id"] = run_id
        if result_id is not None:
            field_eq["result_id"] = result_id
        if label_key is not None:
            field_eq["output_fingerprint"] = label_key.output_fingerprint
            field_eq["criterion_fingerprint"] = label_key.criterion_fingerprint
        items = self._store.by_doc_type(
            "calibration_rating", scope_id, order_by="rated_at", descending=False, **field_eq
        )
        return self._hydrate_all(CalibrationRating, items)

    # =========================================================================
    # EvalJudgeProfile — a judge's measured reliability, for other campaigns to read (#628)
    # =========================================================================

    def save_judge_profile(self, profile: EvalJudgeProfile) -> None:
        """Persist a profile in the scope it names, replacing the stored profile of the same judge and criterion.

        The replacement is the id's doing: a profile's id derives from its judge and criterion, so the upsert lands
        on the earlier profile's row.
        """
        self._save(profile.to_dict())

    def load_judge_profile(self, profile_id: str, scope_id: str) -> EvalJudgeProfile | None:
        """Load a profile by id within a scope."""
        return self._load(EvalJudgeProfile, profile_id, scope_id)

    def query_judge_profiles(self, scope_id: str, *, rubric_dim: str | None = None) -> list[EvalJudgeProfile]:
        """Every profile in a scope, oldest recording first, optionally of one dim.

        Unlimited: the bundle looks a judge's profile up among them, and a paged read would miss one.

        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param rubric_dim: Optional equality filter on the criterion's dim.
        :ptype rubric_dim: str | None
        :return: The matching profiles, ordered by when they were recorded.
        :rtype: list[EvalJudgeProfile]
        """
        field_eq: dict[str, Any] = {}
        if rubric_dim is not None:
            field_eq["rubric_dim"] = rubric_dim
        items = self._store.by_doc_type(
            "eval_judge_profile", scope_id, order_by="recorded_at", descending=False, **field_eq
        )
        return self._hydrate_all(EvalJudgeProfile, items)

    # =========================================================================
    # OutOfRunSpend — the ledger of calls made outside any run
    # =========================================================================

    def save_out_of_run_spend(self, spend: OutOfRunSpend) -> None:
        """Persist one out-of-run call's ledger row in the scope it names. Rows are written once, never rewritten."""
        self._save(spend.to_dict())

    def query_out_of_run_spend(
        self,
        scope_id: str,
        *,
        purpose: KnownOutOfRunPurpose | None = None,
        launch_group_id: str | None = None,
        template_id: str | None = None,
    ) -> list[OutOfRunSpend]:
        """The out-of-run calls ledgered in a scope, oldest first, optionally narrowed.

        Unlimited: a paged read summed into a spend total would report the first page as the whole.

        :param scope_id: The scope to read.
        :ptype scope_id: str
        :param purpose: Only calls made for this purpose.
        :ptype purpose: KnownOutOfRunPurpose | None
        :param launch_group_id: Only the calls a launch's case generation made; its runs carry the same id.
        :ptype launch_group_id: str | None
        :param template_id: Only calls made for this template.
        :ptype template_id: str | None
        :return: The matching rows, ordered by when they were written.
        :rtype: list[OutOfRunSpend]
        """
        field_eq: dict[str, Any] = {}
        if purpose is not None:
            field_eq["purpose"] = purpose
        if launch_group_id is not None:
            field_eq["launch_group_id"] = launch_group_id
        if template_id is not None:
            field_eq["template_id"] = template_id
        items = self._store.by_doc_type(
            "eval_out_of_run_spend", scope_id, order_by="created_at", descending=False, **field_eq
        )
        return self._hydrate_all(OutOfRunSpend, items)

    # =========================================================================
    # EvalTestCase
    # =========================================================================

    def save_test_case(self, test_case: EvalTestCase, *, if_match: str | None = None) -> None:
        """Persist a test case; with ``if_match`` (from :meth:`load_test_case_with_etag`), only if nothing else wrote
        it since it was read, else :class:`ConflictError`."""
        self._save(test_case.to_dict(), if_match=if_match)

    def load_test_case(self, test_case_id: str, scope_id: str) -> EvalTestCase | None:
        """Load a test case by id + scope."""
        return self._load(EvalTestCase, test_case_id, scope_id)

    def load_test_case_with_etag(self, test_case_id: str, scope_id: str) -> tuple[EvalTestCase | None, str | None]:
        """Load a test case plus the token a conditional write of it must present (:meth:`save_test_case`)."""
        data, etag = self._store.get_with_etag(test_case_id, scope_id)
        data = self._of_type(EvalTestCase, data)
        if data is None:
            return None, None
        return self._hydrate(EvalTestCase, data), etag

    def query_test_cases(
        self,
        scope_id: str,
        *,
        template_id: str | None = None,
    ) -> list[EvalTestCase]:
        """Return test cases in a scope, optionally filtered by template."""
        field_eq: dict[str, Any] = {}
        if template_id:
            field_eq["template_id"] = template_id
        items = self._store.by_doc_type("eval_test_case", scope_id, **field_eq)
        return self._hydrate_all(EvalTestCase, items)

    def load_test_cases_by_ids(
        self,
        test_case_ids: list[str],
        scope_id: str,
    ) -> list[EvalTestCase]:
        """Load specific test cases by id list.

        Silently skips missing ids — a batch read of ids that may or may not be
        there, not an assertion that they are. One store call for the whole list;
        whether that costs one round trip is the adapter's business.
        """
        items = self._store.get_many("eval_test_case", test_case_ids, scope_id)
        return self._hydrate_all(EvalTestCase, items)

    def load_case_strata(self, test_case_ids: Sequence[str], scope_id: str, /) -> list[EvalCaseStratum]:
        """Load the id, stratum and given name of the named test cases, in one batch read.

        The store reduces each case to those three fields before it is shipped, so a reader of a
        campaign's strata pays for two short strings a case rather than the host's whole stimulus.
        Absent ids are skipped, as :meth:`load_test_cases_by_ids` skips them.

        :param test_case_ids: The cases to read. Empty asks the store nothing.
        :ptype test_case_ids: Sequence[str]
        :param scope_id: The scope they live in.
        :ptype scope_id: str
        :return: One entry per case that resolved.
        :rtype: list[EvalCaseStratum]
        """
        if not test_case_ids:
            return []
        items = self._store.get_many(
            "eval_test_case", list(test_case_ids), scope_id, keep=list(EvalCaseStratum.model_fields)
        )
        return self._hydrate_all(EvalCaseStratum, items)

    def delete_test_case(self, test_case_id: str, scope_id: str) -> bool:
        """Delete a test case by id + scope."""
        return self._store.delete(test_case_id, scope_id)

    # =========================================================================
    # EvalRun
    # =========================================================================

    def save_eval_run(self, run: EvalRun, *, if_match: str | None = None) -> None:
        """Persist an eval run.

        When ``if_match`` is provided the write uses optimistic concurrency:
        the upsert succeeds only if the document's ``_etag`` still matches, and a
        lost race raises :class:`ConflictError` (see :meth:`_save`).

        :raises ValueError: ``run`` came from a listing that left payload paths out
            (:attr:`EvalRun.elided_payload_paths`). Writing it back would delete them from the
            stored run for good; save a run read whole with :meth:`load_eval_run`.
        """
        if run.elided_payload_paths:
            raise ValueError(
                f"eval run {run.id} was read by a listing that left {sorted(run.elided_payload_paths)} out of "
                "its host_payload; saving it would delete them — load the run whole before writing it"
            )
        self._save(run.to_dict(), if_match=if_match)

    def load_eval_run(self, run_id: str, scope_id: str) -> EvalRun | None:
        """Load an eval run by id + scope; ``None`` when the id names no run there (:meth:`_of_type`).

        :raises UnreadableRunError: The stored run does not validate under this build's model, or writing it lost did
            not.
        """
        data = self._of_type(EvalRun, self._store.get(run_id, scope_id))
        if data is None:
            return None
        (run,) = self._hydrate_runs([data], frozenset(), scope_id)
        return run

    def load_eval_run_with_etag(
        self,
        run_id: str,
        scope_id: str,
    ) -> tuple[EvalRun | None, str | None]:
        """Load an eval run plus the token a conditional write must present.

        A separate store call from :meth:`load_eval_run` because the port strips
        the token off an ordinary read along with the rest of the storage stamp —
        the models reject it — so a caller that intends a read-modify-write has
        to say so.
        """
        data, etag = self._store.get_with_etag(run_id, scope_id)
        data = self._of_type(EvalRun, data)
        if data is None:
            return None, None
        return self._hydrate(EvalRun, data), etag

    def query_eval_runs(
        self,
        scope_id: str,
        status: EvalRunStatus | None = None,
        *,
        elide_payload: frozenset[str] = frozenset(),
    ) -> list[EvalRun]:
        """Return eval runs in a scope, optionally filtered by status.

        :param scope_id: The scope whose runs to read.
        :ptype scope_id: str
        :param status: Only runs with this status; ``None`` reads every run.
        :ptype status: EvalRunStatus | None
        :param elide_payload: Paths relative to ``host_payload`` the store leaves out of every
            returned run — the host's declared listing elisions. Each returned run records
            them (:attr:`EvalRun.elided_payload_paths`), so a reader that needs one refuses
            instead of mistaking its absence for "never recorded".
        :ptype elide_payload: frozenset[str]
        :return: The matching runs.
        :rtype: list[EvalRun]
        :raises UnreadableRunError: A matched row does not validate under this build's model, or writing a run lost
            did not: the listing refuses rather than answer without it, naming every such row it matched.
            :meth:`query_eval_runs_isolating` leaves them out and names them instead.
        """
        runs = self._hydrate_runs(self._run_rows(scope_id, status, elide_payload), elide_payload, scope_id)
        return [run for run in runs if not status or run.status == status]

    def query_eval_runs_isolating(
        self,
        scope_id: str,
        status: EvalRunStatus | None = None,
        *,
        elide_payload: frozenset[str] = frozenset(),
    ) -> RunsRead:
        """:meth:`query_eval_runs`, leaving out each matched row this build's model refuses and naming it instead.

        For an inventory, whose reader is better served by every run that reads and a count of the rest than by no
        answer: ``runs_list``, the listing an operator opens to find a run. Every reader whose answer would be wrong
        without a run it matched — a comparison, a campaign's members, a cost — reads through
        :meth:`query_eval_runs`, which refuses instead. A row is refused by the isolating read the boot reclaim and
        :meth:`scan_eval_runs` go through (:meth:`_hydrate_isolating`), so the rows left out are ones
        ``runs_unreadable`` names.

        :param scope_id: The scope whose runs to read.
        :ptype scope_id: str
        :param status: As :meth:`query_eval_runs` takes it. A refused row is named when its stored status is the one
            asked for, or for any status when none is.
        :ptype status: EvalRunStatus | None
        :param elide_payload: As :meth:`query_eval_runs` takes it.
        :ptype elide_payload: frozenset[str]
        :return: The runs that read, settled, and each matched row that did not.
        :rtype: RunsRead
        :raises UnreadableRunError: A run read, but writing it lost was refused by a model.
        """
        read, refused = self._hydrate_isolating(EvalRun, self._run_rows(scope_id, status, elide_payload))
        runs = self._settled_runs(read, elide_payload, scope_id)
        return RunsRead(
            runs=[run for run in runs if not status or run.status == status],
            unreadable=[row for row in refused if not status or row.status == status],
        )

    def _run_rows(self, scope_id: str, status: str | None, elide_payload: frozenset[str]) -> list[dict[str, Any]]:
        """The stored run rows a listing by ``status`` matches, before they are read (:meth:`query_eval_runs`)."""
        field_eq: dict[str, Any] = {}
        if elide_payload:
            # Only when there is something to drop: a store written to the port before ``exclude``
            # would read it as an equality predicate and return no runs at all.
            field_eq["exclude"] = _payload_exclusions(elide_payload)
        if not status:
            return self._store.by_doc_type("eval_run", scope_id, **field_eq)
        # A run is settled as it is read, so the status asked for is the settled one: a lapsed ``running`` run reads
        # ``cancelled`` and leaves a ``running`` listing, and a ``cancelled`` listing reads the lapsed ones too.
        asked = {status} | (NON_TERMINAL_RUN_STATUSES if status == RUN_LOST_STATUS else set())
        return [
            item
            for asked_for in sorted(asked)
            for item in self._store.by_doc_type("eval_run", scope_id, status=asked_for, **field_eq)
        ]

    def load_eval_runs(
        self,
        run_ids: Sequence[str],
        scope_id: str,
        /,
        *,
        elide_payload: frozenset[str] = frozenset(),
    ) -> list[EvalRun]:
        """Load the runs named by ``run_ids`` within one scope, in one batch read.

        For a reader that needs many named runs but not every run's whole payload — a campaign's
        members, or one run whose scalars are all a surface renders. Absent ids are skipped, as a
        miss is for :meth:`load_eval_run`; the order is the store's, not ``run_ids``'.

        :param run_ids: The runs to load. Empty asks the store nothing.
        :ptype run_ids: Sequence[str]
        :param scope_id: The scope they live in.
        :ptype scope_id: str
        :param elide_payload: Paths relative to ``host_payload`` the store leaves out of every returned
            run, with the meaning :meth:`query_eval_runs` gives them: each returned run records
            them, and a reader that needs one refuses.
        :ptype elide_payload: frozenset[str]
        :return: The runs that resolved.
        :rtype: list[EvalRun]
        :raises UnreadableRunError: A resolved row does not validate under this build's model, or writing a run lost
            did not.
        """
        exclude = _payload_exclusions(elide_payload) if elide_payload else ()
        items = self._store.get_many("eval_run", list(run_ids), scope_id, exclude=exclude)
        return self._hydrate_runs(items, elide_payload, scope_id)

    def load_eval_run_stamps(self, run_ids: Sequence[str], scope_id: str, /) -> list[EvalRunStamp]:
        """Load the id, archived flag and start time of the named runs, in one batch read.

        The store reduces each document to those three fields before it is shipped, so a reader
        of a many-run cohort's shape pays for three scalars a run rather than the run. Absent ids
        are skipped.

        :param run_ids: The runs to read. Empty asks the store nothing.
        :ptype run_ids: Sequence[str]
        :param scope_id: The scope they live in.
        :ptype scope_id: str
        :return: One stamp per run that resolved.
        :rtype: list[EvalRunStamp]
        """
        items = self._store.get_many("eval_run", list(run_ids), scope_id, keep=list(EvalRunStamp.model_fields))
        return self._hydrate_all(EvalRunStamp, items)

    def set_eval_run_archived(self, run_id: str, scope_id: str, /, *, archived: bool) -> bool:
        """Write a run's ``archived`` flag in place, without reading or re-sending the rest of it.

        The run document is mostly the host's frozen payload, and a whole-document rewrite to flip
        one flag reads all of it into this process and sends all of it back. This sets the one
        field; the new etag it mints refuses a conditional writer still holding the old one, which
        re-reads and keeps the flag. A writer that saves the run with no ``if_match`` is not
        refused, and can put the old flag back.

        :param run_id: The run to curate.
        :ptype run_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :param archived: The flag's new value.
        :ptype archived: bool
        :return: ``True`` when the run was written; ``False`` when there is no such run.
        :rtype: bool
        :raises StorageError: The write failed.
        """
        try:
            return self._store.merge_fields(run_id, scope_id, {"archived": archived})
        # prawduct:ok-broad-except — DB write boundary; re-raised as a typed eval error
        except Exception as e:
            log.error("Failed to set archived on eval run %s (scope %s) — %s", run_id, scope_id, e)
            raise StorageError(f"failed to write eval_run '{run_id}': {e}") from e

    def _hydrate_runs(self, items: list[dict[str, Any]], elide_payload: frozenset[str], scope_id: str) -> list[EvalRun]:
        """Hydrate and settle stored runs, marking each with the payload paths the read left out.

        Every run read that cannot leave a run out goes through here, so a row this build's model refuses is
        refused as an engine refusal naming every such row it met, never as the model's own error for the first.
        """
        read, refused = self._hydrate_isolating(EvalRun, items)
        if refused:
            raise UnreadableRunError.of_rows(scope_id, refused) from refused[0].refused
        return self._settled_runs(read, elide_payload, scope_id)

    def _settled_runs(self, runs: list[EvalRun], elide_payload: frozenset[str], scope_id: str) -> list[EvalRun]:
        """Runs that read, settled (:meth:`_settled`) and marked with the payload paths the read left out."""
        try:
            settled = self._settled(EvalRun, runs)
        except ValidationError as refused:
            raise UnreadableRunError.of_settling(scope_id, refused) from refused
        if elide_payload:
            for run in settled:
                run.note_elided_payload(elide_payload)
        return settled

    @staticmethod
    def _hydrate_isolating[M: EvalBaseModel](
        model_cls: type[M], rows: Iterable[dict[str, Any]]
    ) -> tuple[list[M], list[RefusedRow]]:
        """Reconstruct each row (see :meth:`_hydrate`), keeping a row the model refuses as a :class:`RefusedRow`.

        The one definition of a row this build cannot read, for the reads that must survive one: what the boot
        reclaim's scan counts (:meth:`query_non_terminal_eval_runs`) and what an operator's scan names
        (:meth:`scan_eval_runs`) are the same rows because both go through here. Only the model's refusal is
        kept per row; anything else raises for the list, since it is not a fact about that row.
        """
        read: list[M] = []
        refused: list[RefusedRow] = []
        for row in rows:
            try:
                read.append(EvalStorage._hydrate(model_cls, row))
            except ValidationError as why:
                row_id, status, lease = row.get("id"), row.get("status"), row.get("lease")
                expires_at = lease.get("expires_at") if isinstance(lease, dict) else None
                refused.append(
                    RefusedRow(
                        row_id=row_id if isinstance(row_id, str) else None,
                        status=status if isinstance(status, str) else None,
                        refused=why,
                        lease_expires_at=expires_at if isinstance(expires_at, str) else None,
                        lease_stated=lease is not None,
                    )
                )
        return read, refused

    def scan_eval_runs(self, scope_id: str, *, elide_payload: frozenset[str] = frozenset()) -> EvalRunScan:
        """Read every run in one scope, whatever its status, counting what reads and naming each row that does not.

        The operator's question the boot reclaim cannot answer: which stored runs has this build's model stopped
        reading, and why. A row is refused exactly as the reclaim's scan refuses one, by the same isolating read,
        so every row that scan counts unreadable (one stored ``pending`` or ``running``) is named here, beside the
        terminal ones it never reads. Nothing is settled or written: a run read here is counted, not returned.

        :param scope_id: The scope whose runs to read.
        :ptype scope_id: str
        :param elide_payload: Paths relative to ``host_payload`` the store leaves out of every row, with the meaning
            :meth:`query_eval_runs` gives them. The payload is opaque to the model, so leaving part of it out
            never changes whether a row reads; it only keeps a scan of a large scope from holding every run's
            heavy payload at once.
        :ptype elide_payload: frozenset[str]
        :return: How many rows read, and each refused row with why.
        :rtype: EvalRunScan
        :raises StorageError: The store failed to answer — never an empty scan.
        """
        store = self._store
        field_eq: dict[str, Any] = {"exclude": _payload_exclusions(elide_payload)} if elide_payload else {}
        try:
            items = store.by_doc_type("eval_run", scope_id, **field_eq)
        # prawduct:ok-broad-except — DB read boundary; re-raised as a typed eval error
        except Exception as e:
            raise StorageError(f"failed to read the eval runs of scope {scope_id!r}: {e}") from e
        read, refused = self._hydrate_isolating(EvalRun, items)
        return EvalRunScan(readable=len(read), unreadable=refused)

    def scan_eval_run(self, run_id: str, scope_id: str) -> EvalRunScan:
        """Read one run by id without letting the model's refusal raise: whether it reads, or why not, or that it is
        absent.

        :meth:`scan_eval_runs` for one id, through the same isolating read, so a run named here as refused is one
        ``runs_unreadable`` names. For a surface that must act on a run this build cannot read — its delete — where
        :meth:`load_eval_run` refuses. Nothing is settled or written.

        :param run_id: The run.
        :ptype run_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :return: ``readable`` 1 when it reads; one ``unreadable`` row, with why, when the model refuses it; neither when
            no run of that id is stored there.
        :rtype: EvalRunScan
        """
        data = self._of_type(EvalRun, self._store.get(run_id, scope_id))
        if data is None:
            return EvalRunScan(readable=0, unreadable=[])
        read, refused = self._hydrate_isolating(EvalRun, [data])
        return EvalRunScan(readable=len(read), unreadable=refused)

    def query_non_terminal_eval_runs(self, scope_id: str) -> NonTerminalRunScan:
        """Return every ``pending``/``running`` run in one scope.

        The process-restart reclaim's read. It is scoped like every other read: the reclaim
        is told by its host which scopes to sweep and asks each in turn, rather than relying
        on one backend's notion of "every scope this connection can see". It reads one status
        at a time, because ``by_doc_type`` takes equality predicates and has no disjunction.

        **Isolates a refused row — one of the few reads on this class that survive a document
        it cannot reconstruct, beside :meth:`scan_eval_runs`, which names what this one counts
        through the same isolating read (:meth:`_hydrate_isolating`).** The row is not read
        leniently: it is left out, reported, and stays as stored. (:func:`~threetears.evals.run.run_document.update_eval_run` survives a
        different thing: a read *fault*, where the document never arrived at all.)
        Every other query here hydrates through :meth:`_hydrate_all`, where one
        unreconstructable row raises for the whole list — which is correct for
        them: a listing that
        silently omits rows is a worse answer than no answer. This read is the
        exception because of who calls it. Its caller is the boot-path reclaim,
        and the process must come up far enough to serve the admin surface that
        repairs the corpus; a run left unreadable by a deliberate model change
        would otherwise stop the web process binding at all, which trades one
        stranded run for no way to reach any of them. The rows are not dropped
        silently: each is logged at ERROR with its id, and the count rides back
        on :class:`NonTerminalRunScan` so the sweep's report can disclose it, while
        :meth:`scan_eval_runs` names each one with why for an operator who asks.
        """
        runs: list[EvalRun] = []
        unreadable = 0
        for status in sorted(NON_TERMINAL_RUN_STATUSES):
            read, refused = self._hydrate_isolating(
                EvalRun, self._store.by_doc_type("eval_run", scope_id, status=status)
            )
            runs.extend(read)
            for row in refused:
                unreadable += 1
                log.error(
                    "Non-terminal eval run %s (scope=%s status=%s) cannot be read under the current "
                    "model and will NOT be reclaimed. It stays stamped %s until its document "
                    "is dropped or the model can read it again. runs_unreadable names it with why; run_delete removes it.",
                    row.row_id or "<no id on the row>",
                    scope_id,
                    status,
                    status,
                )
        return NonTerminalRunScan(runs=runs, unreadable=unreadable)

    def delete_eval_run(self, run_id: str, scope_id: str) -> bool:
        """Delete an eval run by id + scope."""
        return self._store.delete(run_id, scope_id)

    # =========================================================================
    # EvalResult
    # =========================================================================

    def save_eval_result(self, result: EvalResult, trace: EvalTrace | None = None) -> None:
        """Persist an eval result, and its trace payload as a sibling document.

        **Trace first, then the result, and ``has_trace`` records that the cell produced one.**
        Writing the result first would leave a window where a reader sees ``has_trace=True``
        and no document to fetch, so the payload is written first.

        **A failed trace write does not fail the result, and is not hidden either.**
        :func:`~threetears.evals.run.runner.execute_run` treats a raise here as a lost cell
        and counts it against the run's completeness, so failing the whole write
        because a debug payload did not persist would trade the measurement for the
        thing that exists to explain it. The result lands with ``has_trace=True`` all the
        same: the cell produced a trace, and a marker written ``False`` would read on every
        surface as a cell that produced none, with the failure in the log alone. A result
        whose marker says a trace was produced and no document backs it reads ``missing``
        (:func:`~threetears.evals.run.reads.get_result_trace`, ``result_get``) — the fault it is —
        and a later re-judge of it refuses, since the judge evidence it would resend was in
        the payload that did not land.

        :param result: The result to persist. Never mutated — the marker is applied to
            a copy, so a caller holding this object cannot be surprised by it.
        :ptype result: EvalResult
        :param trace: The sibling payload, or ``None`` for a cell that produced none.
            An entirely empty trace writes no document: there is nothing to
            fetch, and a row per empty payload is the row count this split
            exists to keep down.
        :ptype trace: EvalTrace | None
        :raises StorageError: The RESULT was not persisted. A trace-write failure is logged
            by :meth:`_save` and reads as ``missing`` through ``has_trace``, not raised.
        """
        produced = trace is not None and bool(
            trace.trace
            or trace.otel_trace
            or trace.judge_evidence is not None
            or trace.call_ledger is not None
            or trace.end_state is not None
        )
        if trace is not None and produced:
            try:
                self._save(trace.to_dict())
            except StorageError as unstored:
                # NOSILENT: the result below is written with has_trace=True and no document backs it, which reads
                # ``missing`` on every surface and is refused by a re-judge — the fault it is, never "no trace".
                log.error(
                    "eval_result %s (run %s, scope %s) is stored with has_trace=True and its trace was NOT: it reads "
                    "missing on every surface — %s",
                    result.id,
                    result.eval_run_id,
                    result.scope_id,
                    unstored,
                )
        self._save(result.model_copy(update={"has_trace": produced}).to_dict())

    def load_eval_result(self, result_id: str, scope_id: str) -> EvalResult | None:
        """Load an eval result by id + scope. Never carries the trace — see :meth:`load_eval_trace`.

        ``None`` when the id names no result there, a run's or a trace's id among them (:meth:`_of_type`).
        """
        return self._load(EvalResult, result_id, scope_id)

    def load_eval_result_with_etag(self, result_id: str, scope_id: str) -> tuple[EvalResult | None, str | None]:
        """Load an eval result plus the token a conditional rewrite of it must present.

        The result counterpart of :meth:`load_eval_run_with_etag`, for the one caller that
        rewrites a stored result after its cell has finished (a re-judge).
        """
        data, etag = self._store.get_with_etag(result_id, scope_id)
        data = self._of_type(EvalResult, data)
        if data is None:
            return None, None
        return self._hydrate(EvalResult, data), etag

    def replace_eval_result(self, result: EvalResult, *, if_match: str | None) -> None:
        """Rewrite a stored result in place, only if nothing else wrote it since it was read.

        Distinct from :meth:`save_eval_result` because that method derives ``has_trace``
        from the trace it is handed, and a rewrite hands none: it would mark every
        rewritten result as having no trace while its trace document still exists. Here
        the result's own ``has_trace`` is written as read.

        :param result: The result as it should now stand.
        :ptype result: EvalResult
        :param if_match: The token :meth:`load_eval_result_with_etag` returned with it. ``None`` writes
            unconditionally; every store hands a found document's token back
            (:meth:`~threetears.evals.schema.store_port.DocumentStore.get_with_etag`), so a
            rewrite of a result just read is always conditional.
        :ptype if_match: str | None
        :raises ConflictError: Something else wrote the result since it was read.
        :raises StorageError: The write failed for any other reason.
        """
        self._save(result.to_dict(), if_match=if_match)

    def load_eval_trace(self, result_id: str, scope_id: str) -> EvalTrace | None:
        """Load one result's trace payload; ``None`` when it has none.

        A point read on the result's own partition, so a detail view costs one extra
        lookup and every other read path costs nothing. Callers that only need to
        know whether detail EXISTS should read ``EvalResult.has_trace`` instead —
        that is what the marker is for.
        """
        return self._load(EvalTrace, eval_trace_doc_id(result_id), scope_id)

    def load_judge_evidence(self, result_ids: Sequence[str], scope_id: str, /) -> dict[str, JudgeEvidence]:
        """What each named result's judge was handed, read off their traces in one batch read.

        For a reader of many results' evidence at once (a campaign's trials): one store call however many results
        are named, where a :meth:`load_eval_trace` per result is one round trip each. The store reduces each trace to
        the fields that locate and version it and the judge evidence with the declaration stored beside it, so the
        turns and spans that are most of a trace's weight are never shipped. The reduced trace is read as an
        :class:`EvalTrace`, upgraded and validated as a whole one is, and never leaves this method: only its evidence
        is returned, so nothing downstream can take the reduced trace for the whole one.

        :param result_ids: The results whose evidence to read. Empty asks the store nothing; one named twice is read
            once.
        :ptype result_ids: Sequence[str]
        :param scope_id: The scope they live in.
        :ptype scope_id: str
        :return: Result id → the evidence its judge was handed, for each result whose trace resolves and carries
            evidence. A result with no stored trace, or whose trace holds no evidence (an unjudged kind), is absent.
        :rtype: dict[str, JudgeEvidence]
        :raises ValidationError: A stored trace this build cannot read, as :meth:`load_eval_trace` raises for one.
        """
        if not result_ids:
            return {}
        trace_ids = [eval_trace_doc_id(result_id) for result_id in dict.fromkeys(result_ids)]
        items = self._store.get_many("eval_trace", trace_ids, scope_id, keep=list(_JUDGE_EVIDENCE_FIELDS))
        return {
            trace.result_id: trace.judge_evidence
            for trace in self._hydrate_all(EvalTrace, items)
            if trace.judge_evidence is not None
        }

    def query_eval_results(
        self,
        scope_id: str,
        *,
        run_id: str | None = None,
        test_case_id: str | None = None,
        model: str | None = None,
    ) -> list[EvalResult]:
        """Return eval results with optional filters (unpaginated).

        There is no ``include_trace`` switch and no projection: the stored
        ``eval_result`` document does not contain the turn-by-turn record or the OTel
        spans at all, so every read is already narrow. The parameter this replaced
        defaulted to True and was passed ``False`` by all eighteen production call
        sites — it existed only to ask the database to haul less of a row than the
        model claimed the row held.
        """
        items = self._store.by_doc_type("eval_result", scope_id, **self._result_filters(run_id, test_case_id, model))
        return self._hydrate_all(EvalResult, items)

    def query_eval_results_by_run(self, run_id: str, scope_id: str) -> list[EvalResult]:
        """Return all results for a specific eval run."""
        return self.query_eval_results(scope_id, run_id=run_id)

    def query_eval_result_ids_by_run(self, run_id: str, scope_id: str) -> list[str]:
        """The ids of every result stored for one run, read off the rows without reconstructing them.

        For a delete of a run this build cannot read, whose results were written by the same model and may not read
        either: :meth:`delete_eval_result` works from the stored row, so finding them by the run id they carry is
        all the cascade needs, and a refused result never stops it.

        :param run_id: The run.
        :ptype run_id: str
        :param scope_id: The scope it lives in.
        :ptype scope_id: str
        :return: Each result row's id, in the store's order.
        :rtype: list[str]
        :raises StorageError: The store failed to answer, or a row matched carries no id as text.
        """
        store = self._store
        try:
            items = store.by_doc_type("eval_result", scope_id, eval_run_id=run_id)
        # prawduct:ok-broad-except — DB read boundary; re-raised as a typed eval error
        except Exception as e:
            raise StorageError(f"failed to read the results of eval run {run_id!r}: {e}") from e
        ids = [item.get("id") for item in items]
        if not all(isinstance(result_id, str) for result_id in ids):
            raise StorageError(f"a stored result of eval run {run_id!r} carries no id as text, so it cannot be deleted")
        return cast(list[str], ids)

    @staticmethod
    def _result_filters(
        run_id: str | None,
        test_case_id: str | None,
        model: str | None,
    ) -> dict[str, Any]:
        """Build the shared eval-result equality predicates (run/test-case/model)."""
        field_eq: dict[str, Any] = {}
        if run_id:
            field_eq["eval_run_id"] = run_id
        if test_case_id:
            field_eq["test_case_id"] = test_case_id
        if model:
            field_eq["model"] = model
        return field_eq

    def delete_eval_result(self, result_id: str, scope_id: str) -> bool:
        """Delete an eval result and its trace sibling, refusing to orphan the trace.

        The trace is deleted first, and the result's own ``has_trace`` decides what that
        delete's outcome means. ``save_eval_result`` stamps the marker when the cell produced
        a trace, so it is the stored record of whether a trace document should be there:

        - ``has_trace`` false: no trace was written, a trace delete that removes nothing is the
          expected case, and the result is deleted as before.
        - ``has_trace`` true and the trace delete removed nothing: the trace is read
          back. Still there means the delete did not take, and deleting the result
          would leave the largest row a result has unreachable for good (nothing finds
          a trace but its result's id), so the result is **kept** and this raises.
          Gone means the marker already disagreed with storage (the ``missing`` state
          :func:`~threetears.evals.run.reads.get_result_trace` logs: the trace write failed, or
          the trace was deleted without its result); there is nothing
          to orphan, so the result is deleted and the disagreement logged, rather than
          leaving a result no delete can ever remove.

        An id naming no result deletes nothing of another type and returns ``False``; a
        stray trace under it is still removed. A raising trace delete or read propagates
        before the result is touched.

        :return: ``True`` when the result was deleted; ``False`` when there was no result.
        :rtype: bool
        :raises StorageError: The result says it has a trace, and the trace survived its
            delete. The result was not deleted; a retry is safe.
        """
        stored = self._of_type(EvalResult, self._store.get(result_id, scope_id))
        has_trace = bool(stored.get("has_trace")) if stored is not None else False
        trace_id = eval_trace_doc_id(result_id)
        if not self._store.delete(trace_id, scope_id) and has_trace:
            if self._store.get(trace_id, scope_id) is not None:
                log.error(
                    "eval_result %s (scope %s) left intact: it is stamped has_trace=True and its "
                    "trace %s survived the delete, so deleting the result would orphan the trace",
                    result_id,
                    scope_id,
                    trace_id,
                )
                raise StorageError(
                    f"result '{result_id}' left intact: its trace '{trace_id}' could not be deleted, and "
                    f"deleting the result alone would leave the trace unreachable — retry the delete"
                )
            log.warning(
                "eval_result %s (scope %s) is stamped has_trace=True but has no trace document; "
                "deleting the result (nothing to orphan)",
                result_id,
                scope_id,
            )
        if stored is None:
            # The id names no result here; a document of another type under it is not ours to delete.
            return False
        return self._store.delete(result_id, scope_id)

    # =========================================================================
    # EvalCassette — one capture run's corpus of recorded answers
    # =========================================================================

    def save_cassette(self, cassette: EvalCassette) -> None:
        """Persist a cassette in the scope it names, replacing any recording of the same key.

        :raises StorageError: The write failed.
        """
        self._save(cassette.to_dict())

    def get_cassette(self, key: CassetteKey, scope_id: str) -> EvalCassette | None:
        """The recording of one key, or ``None`` when the corpus has none.

        A point read by the key's deterministic id. Strict like every read: a stored row that no
        longer loads raises rather than reaching a replay half-understood.

        :param key: The ask whose recording to read.
        :ptype key: CassetteKey
        :param scope_id: The scope the corpus lives in.
        :ptype scope_id: str
        :return: The recording, or ``None``.
        :rtype: EvalCassette | None
        :raises StorageError: The store failed to answer.
        :raises ValueError: The stored row does not load as this build's :class:`EvalCassette`.
        """
        try:
            data = self._store.get(key.doc_id, scope_id)
        # prawduct:ok-broad-except — DB read boundary; re-raised as a typed eval error
        except Exception as e:
            raise StorageError(f"failed to read eval_cassette '{key.doc_id}': {e}") from e
        data = self._of_type(EvalCassette, data)
        if data is None:
            return None
        return self._hydrate(EvalCassette, data)

    def list_cassettes_for_template(
        self,
        template_id: str,
        scope_id: str,
        *,
        limit: int | None = None,
    ) -> list[EvalCassette]:
        """Return every cassette captured for a template within a scope, newest first.

        The operator's inspection read — when a replay misses, list what the corpora hold for the
        template and bisect from there. ``limit`` is a safety cap (``None`` = unbounded; an operator
        surface should pass one). ``captured_at`` is an ISO-8601 string ordered lexicographically.
        """
        items = self._store.by_doc_type(
            "eval_cassette",
            scope_id,
            template_id=template_id,
            order_by="captured_at",
            descending=True,
            limit=limit,
        )
        return self._hydrate_all(EvalCassette, items)

    def list_case_cassettes(
        self, *, corpus_id: str, template_id: str, test_case_id: str, scope_id: str
    ) -> list[EvalCassette]:
        """Every recording one corpus holds for one case, in no particular order.

        Reads this case's rows only, so a stale row recorded for another case cannot break a read
        of this one.

        :param corpus_id: The capture run whose corpus to read.
        :ptype corpus_id: str
        :param template_id: The template it ran.
        :ptype template_id: str
        :param test_case_id: The case.
        :ptype test_case_id: str
        :param scope_id: The scope the corpus lives in.
        :ptype scope_id: str
        :return: The case's recordings.
        :rtype: list[EvalCassette]
        """
        items = self._store.by_doc_type(
            "eval_cassette",
            scope_id,
            corpus_id=corpus_id,
            template_id=template_id,
            test_case_id=test_case_id,
        )
        return self._hydrate_all(EvalCassette, items)

    def read_case_cassettes(
        self, *, corpus_id: str, template_id: str, test_case_id: str, scope_id: str
    ) -> CaseRecordings:
        """Every recording one corpus holds for one case, each row read on its own — what a replay cell is served from.

        One read of the case's rows, made before the cell starts so that nothing the cell serves touches the store.
        A row that does not load as this build's :class:`EvalCassette` is returned as why (:class:`CaseRecordings`),
        so it fails the call that asks for it and no other, as a point read of it would.

        :param corpus_id: The capture run whose corpus to read.
        :ptype corpus_id: str
        :param template_id: The template it ran.
        :ptype template_id: str
        :param test_case_id: The case.
        :ptype test_case_id: str
        :param scope_id: The scope the corpus lives in.
        :ptype scope_id: str
        :return: The case's rows, read and refused.
        :rtype: CaseRecordings
        :raises StorageError: The store failed to answer.
        """
        try:
            items = self._store.by_doc_type(
                "eval_cassette",
                scope_id,
                corpus_id=corpus_id,
                template_id=template_id,
                test_case_id=test_case_id,
            )
        # prawduct:ok-broad-except — DB read boundary; re-raised as a typed eval error
        except Exception as e:
            raise StorageError(
                f"failed to read the {test_case_id!r} recordings of cassette corpus {corpus_id!r}: {e}"
            ) from e
        readable: dict[str, EvalCassette] = {}
        unreadable: dict[str, str] = {}
        for item in items:
            try:
                readable[item["id"]] = self._hydrate(EvalCassette, item)
            except ValueError as refused:
                # NOSILENT: kept, not dropped: returned as this row's why, and the cell raises it as CassetteCorrupt
                # for the call that asks for this recording, as a point read of the row would.
                unreadable[item["id"]] = str(refused)
        return CaseRecordings(readable=readable, unreadable=unreadable)

    def delete_case_cassettes(self, *, corpus_id: str, template_id: str, test_case_id: str, scope_id: str) -> int:
        """Delete everything one corpus recorded for one case, so a capture of it starts clean.

        Deletes by id without loading a row, so a row this build can no longer read — written
        under another schema, say — is cleared like any other rather than blocking the clear.

        :param corpus_id: The capture run whose corpus to clear.
        :ptype corpus_id: str
        :param template_id: The template it ran.
        :ptype template_id: str
        :param test_case_id: The case.
        :ptype test_case_id: str
        :param scope_id: The scope the corpus lives in.
        :ptype scope_id: str
        :return: How many recordings were deleted.
        :rtype: int
        :raises StorageError: The store failed to list or delete them.
        """
        try:
            items = self._store.by_doc_type(
                "eval_cassette",
                scope_id,
                corpus_id=corpus_id,
                template_id=template_id,
                test_case_id=test_case_id,
            )
            return sum(1 for item in items if self._store.delete(item["id"], scope_id))
        # prawduct:ok-broad-except — DB boundary; re-raised as a typed eval error
        except Exception as e:
            raise StorageError(
                f"failed to clear the {test_case_id!r} recordings of cassette corpus {corpus_id!r}: {e}"
            ) from e

    # =========================================================================
    # Operator action — nuke all eval data
    # =========================================================================

    def nuke_all_eval_data(self, scopes: Iterable[str], *, include_durable: bool) -> dict[str, int]:
        """Delete every eval document in the scopes named; what people author too only when asked by name.

        Sweeps every type in :data:`EVAL_DOC_TYPES` plus the host's own documents named at construction, less the
        durable types (:data:`~threetears.evals.schema.versioning.DURABLE_DOC_TYPES`: templates, rubric dims, judge
        configs and their tombstones, case sets, test cases, calibration ratings) unless ``include_durable`` is
        true. ``include_durable`` has no default, so every caller decides, in its own source, whether the labels
        and definitions people wrote go too: they are what no engine can make again. With it true the two name
        **every** ``doc_type`` the store can hold, which is what makes this a wipe rather than a partial sweep. A
        core baseline is not this operation: it is :meth:`reset_core`, which leaves the host's documents alone too.

        The scopes are the host's to name: the engine has no way to enumerate them and does not
        assume the store can. A scope the store holds nothing in is swept to zero, which is
        indistinguishable from a wipe that found it already empty — so naming the wrong scope
        is answered with honest zeroes, not a refusal.

        Callers must gate this behind explicit operator confirmation — the
        action is unrecoverable.

        :param scopes: The scopes to wipe.
        :ptype scopes: Iterable[str]
        :param include_durable: Whether the durable types are deleted too. Required.
        :ptype include_durable: bool
        :return: ``{doc_type: count_deleted, ...}`` for each swept doc_type, summed over the scopes. A durable type
            kept is not a key; how many of each were kept is logged.
        :rtype: dict[str, int]
        """
        swept = [*EVAL_DOC_TYPES, *self._host_doc_types]
        if not include_durable:
            swept = [doc_type for doc_type in swept if doc_type not in DURABLE_DOC_TYPES]
        return self._sweep("nuke_all_eval_data", list(scopes), swept)

    def reset_core(self, scopes: Iterable[str]) -> dict[str, int]:
        """The core baseline operation: delete every core and regenerable document in the scopes named, nothing else.

        A core baseline reset (``schema/versioning.py``) refuses every core document below the new baseline, so
        they are deleted, with the regenerable documents derived from them; what people author (the durable types)
        is never touched, and neither is a host's own document or anything outside the named scopes (a routed
        store's placement record lives in its routing scope, never a document scope). It goes through the store
        port, so a host names no table, and the store is bootable as soon as it returns. On a
        :class:`~threetears.evals.storage.RoutedDocumentStore` that may not start (the durable types newly placed
        apart), call :meth:`~threetears.evals.storage.RoutedDocumentStore.reset_core` instead, with the engine
        stopped: it is this call over the router's stores, followed by the relocation that re-adopts them.

        Callers must gate this behind explicit operator confirmation — the action is unrecoverable for the core.

        :param scopes: The scopes to reset.
        :ptype scopes: Iterable[str]
        :return: ``{doc_type: count_deleted, ...}`` for each core and regenerable type, summed over the scopes. How many
            durable documents were kept is logged, per type.
        :rtype: dict[str, int]
        """
        swept = [doc_type for doc_type in EVAL_DOC_TYPES if doc_type in CORE_RESET_DOC_TYPES]
        return self._sweep("reset_core", list(scopes), swept)

    def _sweep(self, operation: str, scopes: list[str], doc_types: list[str]) -> dict[str, int]:
        """Delete every document of ``doc_types`` in ``scopes``, logging each drop and every durable type kept."""
        counts: dict[str, int] = dict.fromkeys(doc_types, 0)
        for scope_id in scopes:
            for doc_type in counts:
                deleted = self._delete_every(doc_type, scope_id)
                counts[doc_type] += deleted
                if deleted:
                    log.warning("%s: dropped %d %s docs from scope=%s", operation, deleted, doc_type, scope_id)
            for doc_type in sorted(DURABLE_DOC_TYPES - set(counts)):
                kept = sum(1 for _ in self._store.iter_by_doc_type(doc_type, scope_id))
                if kept:
                    log.info("%s: kept %d %s docs (durable) in scope=%s", operation, kept, doc_type, scope_id)
        return counts

    def drop_all_analyses(self, scopes: Iterable[str]) -> int:
        """Delete every stored analysis in the scopes named, leaving every other eval document in place.

        The operator step for a boundary that retires only the analysis shape: analyses
        regenerate from their campaigns, and nothing depends on one existing — an insight names
        its source analysis as provenance, a reporter case pins its recorded memo as text. It
        iterates ids and deletes raw, like :meth:`nuke_all_eval_data` and :meth:`reset_core`, because the documents it
        exists to remove are exactly the ones no current model can read.

        Callers must gate this behind explicit operator confirmation — the action is unrecoverable.

        :param scopes: The scopes whose analyses to drop.
        :ptype scopes: Iterable[str]
        :return: How many analyses were deleted.
        :rtype: int
        """
        deleted = 0
        for scope_id in scopes:
            dropped = self._delete_every("eval_analysis", scope_id)
            deleted += dropped
            if dropped:
                log.warning("drop_all_analyses: dropped %d eval_analysis docs from scope=%s", dropped, scope_id)
        return deleted

    def _delete_every(self, doc_type: str, scope_id: str) -> int:
        """Delete every document of one type in one scope, by id, without reading any of them."""
        return sum(
            1
            for doc_id in list(self._store.iter_by_doc_type(doc_type, scope_id))
            if self._store.delete(doc_id, scope_id)
        )


if TYPE_CHECKING:

    def _eval_storage_satisfies_every_port(storage: EvalStorage) -> None:
        """Hold :class:`EvalStorage` to each run-side port, so a drifted signature fails typecheck."""
        ports: tuple[
            JobStore,
            RunStore,
            ResultStore,
            RunRecordStore,
            DefinitionStore,
            CassetteStore,
            OutOfRunSpendStore,
            CaseSetStore,
            JudgeProfileStore,
            JobRecordStore,
            LeaseStore,
        ] = (
            storage,
            storage,
            storage,
            storage,
            storage,
            storage,
            storage,
            storage,
            storage,
            storage,
            storage,
        )
        del ports


__all__ = [
    "CaseSetStore",
    "CassetteStore",
    "DefinitionStore",
    "EvalRunScan",
    "EvalStorage",
    "JobRecordStore",
    "JobStore",
    "LeaseClock",
    "LeaseStore",
    "NonTerminalRunScan",
    "RefusedRow",
    "ResultStore",
    "RunsRead",
    "UnreadableRecordError",
    "UnreadableRunError",
    "RunRecordStore",
    "RunStore",
    "save_document",
]
