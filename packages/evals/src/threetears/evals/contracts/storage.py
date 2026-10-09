"""Storage layer for eval documents, over one document store.

Every eval document lives in a ``scope_id`` it names itself: definitions (``EvalTemplate`` /
``JudgeConfig`` / ``CatalogRubricDim``, campaigns, analyses, generation attempts, insights) and
runtime data (``EvalTestCase`` / ``EvalRun`` / ``EvalResult`` / ``EvalTrace`` / ``EvalCassette``)
alike. A write takes the scope from the document; every read and delete takes it as an argument.
The engine never stamps, defaults or interprets a scope — see
:mod:`threetears.evals.contracts.store_port`.

A host may keep documents of its own in the same store. This module does not read or write them
and does not know their ``doc_type`` values: the host declares them when it builds
:class:`EvalStorage` (``host_doc_types``), and the operator wipe sweeps them beside the engine's own
(:data:`EVAL_DOC_TYPES`).

**This module names no backend.** Every read and write goes through the one
:class:`~threetears.evals.contracts.store_port.DocumentStore` handed in at construction, so nothing here
knows which database, table or isolation mechanism sits behind it. Splitting a high-volume kind of
document into its own table, or partitioning by scope, is the adapter's routing by ``doc_type``
and ``scope_id`` — what the core keeps is the document model: which ``doc_type`` a model is, and
which scope a read is asking about.

Strict reads: every load path goes through the model's own ``from_dict(...)`` (:meth:`_hydrate`),
which validates exactly as construction does. A stored document with a field the model does not
declare, without one it requires, or written under another ``schema_version`` is refused — stored
eval documents are dropped across a schema change, never migrated or filtered on the way in. Rows
arrive with the storage stamp already removed (the port's contract), so nothing but the model's
own fields reaches validation. A read can therefore raise, which is why the one read on the boot
path (:meth:`EvalStorage.query_non_terminal_eval_runs`) isolates a refused row rather than
failing the list.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from pydantic import ValidationError

from threetears.evals.contracts.campaign import EvalAnalysis, EvalAnalysisAttempt, EvalCampaign, EvalInsight
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.errors import ConflictError, StorageError
from threetears.evals.contracts.models import (
    NON_TERMINAL_RUN_STATUSES,
    CalibrationRating,
    CassetteKey,
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
    RubricDimTombstone,
    eval_trace_doc_id,
)
from threetears.evals.contracts.out_of_run import OutOfRunPurpose, OutOfRunSpend, OutOfRunSpendStore
from threetears.evals.contracts.store_port import DocumentStore, StoreConflict
from threetears.observe import get_logger

log = get_logger(__name__)


def _payload_exclusions(elide_payload: frozenset[str]) -> list[str]:
    """The store ``exclude`` paths for payload paths a run read leaves out, in a stable order."""
    return sorted(f"host_payload.{path}" for path in elide_payload)


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


#: Every ``doc_type`` the engine writes — the set the operator wipe sweeps.
#:
#: Exactly the engine's own types. A host's documents in the same store are the host's to name, and
#: reach the wipe as ``EvalStorage(host_doc_types=...)`` rather than by an edit here. A type added
#: to the schema and not to this tuple leaves documents behind that an operator was told were gone,
#: which ``tests/test_storage_one_store.py`` pins against the model graph.
EVAL_DOC_TYPES = (
    "eval_template",
    "judge_config",
    "rubric_dim",
    "rubric_dim_tombstone",
    "eval_campaign",
    "eval_analysis",
    "eval_analysis_attempt",
    "eval_insight",
    "eval_run",
    "eval_result",
    "eval_trace",
    "eval_test_case",
    "eval_cassette",
    "calibration_rating",
    "eval_out_of_run_spend",
)


def save_document(repo: DocumentStore, document: dict[str, Any], *, if_match: str | None = None) -> None:
    """Upsert ``document`` through ``repo``, raising on any failure.

    The one write path for eval documents over a :class:`~threetears.evals.contracts.store_port.DocumentStore`,
    shared by :class:`EvalStorage` and by any host store that keeps its own documents in the same
    store, so a failed write means the same thing whichever of them made it.

    There is no success flag to ignore: a write either lands or raises. A lost
    ``if_match`` race is a :class:`ConflictError`, because its remedy is the
    caller's — re-read the winner's document and re-apply — and every other failure
    is a :class:`StorageError`. A write that reported failure as ``False`` let a
    caller that never looked at the value report a success that did not happen.

    Args:
        repo: The store to write through.
        document: The document, carrying its ``id`` and ``doc_type``.
        if_match: The etag the stored document must still have, for a conditional write.

    Raises:
        ConflictError: ``if_match`` was given and the stored document has moved on.
        StorageError: Any other backend failure.
    """
    try:
        repo.upsert(document, if_match=if_match)
    except StoreConflict as e:
        raise ConflictError(
            f"{document.get('doc_type')} '{document.get('id')}' changed since it was read — re-read it and re-apply"
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

    def save_test_case(self, test_case: EvalTestCase, /) -> None:
        """See :meth:`EvalStorage.save_test_case`."""
        ...

    def load_test_case(self, test_case_id: str, scope_id: str, /) -> EvalTestCase | None:
        """See :meth:`EvalStorage.load_test_case`."""
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

    def delete_case_cassettes(self, *, corpus_id: str, template_id: str, test_case_id: str, scope_id: str) -> int:
        """See :meth:`EvalStorage.delete_case_cassettes`."""
        ...


class EvalStorage:
    """Storage for v1-shape eval documents over one document store.

    Load paths go through each model's own ``from_dict``, which validates exactly as construction
    does: a stored document with a field the model does not declare, without one it requires, or
    written under another ``schema_version`` is refused, never coerced or filtered — see the module
    docstring.
    """

    def __init__(self, store: DocumentStore, *, host_doc_types: Sequence[str] = ()):
        """Initialize storage over the store the host supplies.

        Args:
            store: The one store every eval document is read from and written to.
            host_doc_types: The ``doc_type`` values of documents the host keeps in the same store.
                This class never reads or writes them; :meth:`nuke_all_eval_data` sweeps them, so a
                wipe leaves nothing of the host's behind. Empty means the host keeps none there,
                which is a statement about the host rather than a default standing in for one.

        Raises:
            ValueError: A host type is also one of :data:`EVAL_DOC_TYPES` — the engine's own, which
                the host does not get to declare twice.
        """
        claimed = set(host_doc_types) & set(EVAL_DOC_TYPES)
        if claimed:
            raise ValueError(f"host_doc_types names the engine's own doc types: {sorted(claimed)}")
        self._store = store
        self._host_doc_types = tuple(host_doc_types)

    @staticmethod
    def _hydrate[M: EvalBaseModel](model_cls: type[M], data: dict[str, Any]) -> M:
        """Reconstruct one model from a stored document.

        The document arrives with the storage stamp already removed — that is the
        port's contract.
        """
        return model_cls.from_dict(data)

    @classmethod
    def _hydrate_all[M: EvalBaseModel](cls, model_cls: type[M], rows: list[dict[str, Any]]) -> list[M]:
        """Reconstruct a list of models from stored documents (see :meth:`_hydrate`)."""
        return [cls._hydrate(model_cls, row) for row in rows]

    @staticmethod
    def _of_type[M: EvalBaseModel](model_cls: type[M], data: dict[str, Any] | None) -> dict[str, Any] | None:
        """``data`` when it is a document of ``model_cls``'s type, else ``None`` — the one id-read type check.

        A point read is by id alone (:meth:`~threetears.evals.contracts.store_port.DocumentStore.get`),
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

    def _save(self, document: dict[str, Any], *, if_match: str | None = None) -> None:
        """Upsert ``document`` with :func:`save_document`, which raises on any failure."""
        save_document(self._store, document, if_match=if_match)

    def _load[M: EvalBaseModel](self, model_cls: type[M], doc_id: str, scope_id: str) -> M | None:
        """One document by id within a scope, hydrated, or ``None`` when no document of its type resolves there."""
        data = self._of_type(model_cls, self._store.get(doc_id, scope_id))
        if data is None:
            return None
        return self._hydrate(model_cls, data)

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
        """Load a template by ``name`` within a scope.

        Names are unique within a scope by convention (upsert by id, so two templates with the
        same name and different ids can coexist; first match wins).
        """
        items = self._store.by_doc_type("eval_template", scope_id, name=name, limit=1)
        if not items:
            return None
        return self._hydrate(EvalTemplate, items[0])

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

        Args:
            analysis_id: The analysis to ask about.
            scope_id: The scope it lives in.

        Returns:
            ``True`` or ``False`` for a stored analysis, ``None`` when none resolves.
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

    def save_analysis_attempt(self, attempt: EvalAnalysisAttempt) -> None:
        """Persist one generation attempt's record in the scope it names."""
        self._save(attempt.to_dict())

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

        Args:
            scope_id: The scope whose insights to read.
            subject_id: Optional equality filter on the subject.
            source_campaign_id: Optional equality filter on the minting campaign.

        Returns:
            The matching insights, newest observation first — all of them.
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
        self, scope_id: str, *, run_id: str | None = None, result_id: str | None = None
    ) -> list[CalibrationRating]:
        """Ratings in a scope, oldest first, optionally narrowed by run and by result.

        Unlimited, like every agreement input: a paged read would compute agreement over the first
        page and report it as the dimension's.

        Args:
            scope_id: The scope to read.
            run_id: Optional equality filter on the rated result's run.
            result_id: Optional equality filter on the rated result.

        Returns:
            The matching ratings, ordered by when they were rated.
        """
        field_eq: dict[str, Any] = {}
        if run_id is not None:
            field_eq["run_id"] = run_id
        if result_id is not None:
            field_eq["result_id"] = result_id
        items = self._store.by_doc_type(
            "calibration_rating", scope_id, order_by="rated_at", descending=False, **field_eq
        )
        return self._hydrate_all(CalibrationRating, items)

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
        purpose: OutOfRunPurpose | None = None,
        launch_group_id: str | None = None,
        template_id: str | None = None,
    ) -> list[OutOfRunSpend]:
        """The out-of-run calls ledgered in a scope, oldest first, optionally narrowed.

        Unlimited: a paged read summed into a spend total would report the first page as the whole.

        Args:
            scope_id: The scope to read.
            purpose: Only calls made for this purpose.
            launch_group_id: Only the calls a launch's case generation made; its runs carry the same id.
            template_id: Only calls made for this template.

        Returns:
            The matching rows, ordered by when they were written.
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

    def save_test_case(self, test_case: EvalTestCase) -> None:
        """Persist a test case."""
        self._save(test_case.to_dict())

    def load_test_case(self, test_case_id: str, scope_id: str) -> EvalTestCase | None:
        """Load a test case by id + scope."""
        return self._load(EvalTestCase, test_case_id, scope_id)

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
        """Load the id and stratum of the named test cases, in one batch read.

        The store reduces each case to those two fields before it is shipped, so a reader of a
        campaign's strata pays for one short string a case rather than the host's whole stimulus.
        Absent ids are skipped, as :meth:`load_test_cases_by_ids` skips them.

        Args:
            test_case_ids: The cases to read. Empty asks the store nothing.
            scope_id: The scope they live in.

        Returns:
            One entry per case that resolved.
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

        Raises:
            ValueError: ``run`` came from a listing that left payload paths out
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
        """Load an eval run by id + scope; ``None`` when the id names no run there (:meth:`_of_type`)."""
        return self._load(EvalRun, run_id, scope_id)

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

        Args:
            scope_id: The scope whose runs to read.
            status: Only runs with this status; ``None`` reads every run.
            elide_payload: Paths relative to ``host_payload`` the store leaves out of every
                returned run — the host's declared listing elisions. Each returned run records
                them (:attr:`EvalRun.elided_payload_paths`), so a reader that needs one refuses
                instead of mistaking its absence for "never recorded".

        Returns:
            The matching runs.
        """
        field_eq: dict[str, Any] = {}
        if status:
            field_eq["status"] = status
        if elide_payload:
            # Only when there is something to drop: a store written to the port before ``exclude``
            # would read it as an equality predicate and return no runs at all.
            field_eq["exclude"] = _payload_exclusions(elide_payload)
        items = self._store.by_doc_type("eval_run", scope_id, **field_eq)
        return self._hydrate_runs(items, elide_payload)

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

        Args:
            run_ids: The runs to load. Empty asks the store nothing.
            scope_id: The scope they live in.
            elide_payload: Paths relative to ``host_payload`` the store leaves out of every returned
                run, with the meaning :meth:`query_eval_runs` gives them: each returned run records
                them, and a reader that needs one refuses.

        Returns:
            The runs that resolved.
        """
        exclude = _payload_exclusions(elide_payload) if elide_payload else ()
        items = self._store.get_many("eval_run", list(run_ids), scope_id, exclude=exclude)
        return self._hydrate_runs(items, elide_payload)

    def load_eval_run_stamps(self, run_ids: Sequence[str], scope_id: str, /) -> list[EvalRunStamp]:
        """Load the id, archived flag and start time of the named runs, in one batch read.

        The store reduces each document to those three fields before it is shipped, so a reader
        of a many-run cohort's shape pays for three scalars a run rather than the run. Absent ids
        are skipped.

        Args:
            run_ids: The runs to read. Empty asks the store nothing.
            scope_id: The scope they live in.

        Returns:
            One stamp per run that resolved.
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

        Args:
            run_id: The run to curate.
            scope_id: The scope it lives in.
            archived: The flag's new value.

        Returns:
            ``True`` when the run was written; ``False`` when there is no such run.

        Raises:
            StorageError: The write failed.
        """
        try:
            return self._store.merge_fields(run_id, scope_id, {"archived": archived})
        # prawduct:ok-broad-except — DB write boundary; re-raised as a typed eval error
        except Exception as e:
            log.error("Failed to set archived on eval run %s (scope %s) — %s", run_id, scope_id, e)
            raise StorageError(f"failed to write eval_run '{run_id}': {e}") from e

    def _hydrate_runs(self, items: list[dict[str, Any]], elide_payload: frozenset[str]) -> list[EvalRun]:
        """Hydrate stored runs, marking each with the payload paths the read left out."""
        runs = self._hydrate_all(EvalRun, items)
        if elide_payload:
            for run in runs:
                run.note_elided_payload(elide_payload)
        return runs

    def query_non_terminal_eval_runs(self, scope_id: str) -> NonTerminalRunScan:
        """Return every ``pending``/``running`` run in one scope.

        The process-restart reclaim's read. It is scoped like every other read: the reclaim
        is told by its host which scopes to sweep and asks each in turn, rather than relying
        on one backend's notion of "every scope this connection can see". It reads one status
        at a time, because ``by_doc_type`` takes equality predicates and has no disjunction.

        **Isolates a refused row — the only read on this class that survives a document
        it cannot reconstruct.** The row is not read leniently: it is left out, reported, and
        stays as stored. (:func:`~threetears.evals.run.run_document.update_eval_run` survives a
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
        on :class:`NonTerminalRunScan` so the sweep's report can disclose it.
        """
        runs: list[EvalRun] = []
        unreadable = 0
        for status in sorted(NON_TERMINAL_RUN_STATUSES):
            for item in self._store.by_doc_type("eval_run", scope_id, status=status):
                try:
                    runs.append(self._hydrate(EvalRun, item))
                except ValidationError:
                    unreadable += 1
                    log.error(
                        "Non-terminal eval run %s (scope=%s status=%s) cannot be read under the current "
                        "model and will NOT be reclaimed. It stays stamped %s until its document "
                        "is dropped or the model can read it again.",
                        item.get("id", "<no id on the row>"),
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

        **Trace first, then the result, and ``has_trace`` records which of those
        actually happened.** Writing the result first would leave a window where a
        reader sees ``has_trace=True`` and no document to fetch; deriving the flag
        from ``trace is not None`` would make the same claim from intent rather than
        outcome. Setting it from the write's own return value is what keeps the
        marker honest under a partial failure.

        **A failed trace write does not fail the result.**
        :func:`~threetears.evals.run.runner.execute_run` treats a raise here as a lost cell
        and counts it against the run's completeness, so failing the whole write
        because a debug payload did not persist would trade the measurement for the
        thing that exists to explain it. The result lands with ``has_trace=False``,
        which is true — and a later re-judge of it refuses, since the judge evidence it
        would resend was in the payload that did not land.

        Args:
            result: The result to persist. Never mutated — the marker is applied to
                a copy, so a caller holding this object cannot be surprised by it.
            trace: The sibling payload, or ``None`` for a cell that produced none.
                An entirely empty trace writes no document: there is nothing to
                fetch, and a row per empty payload is the row count this split
                exists to keep down.

        Raises:
            StorageError: The RESULT was not persisted. A trace-write failure is logged
                by :meth:`_save` and reported through ``has_trace``, not raised.
        """
        stored = False
        if trace is not None and (
            trace.trace
            or trace.otel_trace
            or trace.judge_evidence is not None
            or trace.call_ledger is not None
            or trace.end_state is not None
        ):
            try:
                self._save(trace.to_dict())
                stored = True
            except StorageError:
                stored = False
        self._save(result.model_copy(update={"has_trace": stored}).to_dict())

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

        Args:
            result: The result as it should now stand.
            if_match: The token :meth:`load_eval_result_with_etag` returned with it. ``None`` writes
                unconditionally; every store hands a found document's token back
                (:meth:`~threetears.evals.contracts.store_port.DocumentStore.get_with_etag`), so a
                rewrite of a result just read is always conditional.

        Raises:
            ConflictError: Something else wrote the result since it was read.
            StorageError: The write failed for any other reason.
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
        """Delete an eval result and its trace sibling.

        The trace is deleted first and its outcome deliberately ignored: it may not
        exist (a cell that produced none writes no document), and a trace surviving
        its result would be unreachable — nothing queries these by anything but a
        result id. Reporting only the result's delete keeps this method's contract
        the one every caller already branches on.
        """
        self._store.delete(eval_trace_doc_id(result_id), scope_id)
        return self._store.delete(result_id, scope_id)

    # =========================================================================
    # EvalCassette — one capture run's corpus of recorded answers
    # =========================================================================

    def save_cassette(self, cassette: EvalCassette) -> None:
        """Persist a cassette in the scope it names, replacing any recording of the same key.

        Raises:
            StorageError: The write failed.
        """
        self._save(cassette.to_dict())

    def get_cassette(self, key: CassetteKey, scope_id: str) -> EvalCassette | None:
        """The recording of one key, or ``None`` when the corpus has none.

        A point read by the key's deterministic id. Strict like every read: a stored row that no
        longer loads raises rather than reaching a replay half-understood.

        Args:
            key: The ask whose recording to read.
            scope_id: The scope the corpus lives in.

        Returns:
            The recording, or ``None``.

        Raises:
            StorageError: The store failed to answer.
            ValueError: The stored row does not load as this build's :class:`EvalCassette`.
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

        Args:
            corpus_id: The capture run whose corpus to read.
            template_id: The template it ran.
            test_case_id: The case.
            scope_id: The scope the corpus lives in.

        Returns:
            The case's recordings.
        """
        items = self._store.by_doc_type(
            "eval_cassette",
            scope_id,
            corpus_id=corpus_id,
            template_id=template_id,
            test_case_id=test_case_id,
        )
        return self._hydrate_all(EvalCassette, items)

    def delete_case_cassettes(self, *, corpus_id: str, template_id: str, test_case_id: str, scope_id: str) -> int:
        """Delete everything one corpus recorded for one case, so a capture of it starts clean.

        Deletes by id without loading a row, so a row this build can no longer read — written
        under another schema, say — is cleared like any other rather than blocking the clear.

        Args:
            corpus_id: The capture run whose corpus to clear.
            template_id: The template it ran.
            test_case_id: The case.
            scope_id: The scope the corpus lives in.

        Returns:
            How many recordings were deleted.

        Raises:
            StorageError: The store failed to list or delete them.
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

    def nuke_all_eval_data(self, scopes: Iterable[str]) -> dict[str, int]:
        """Delete every eval document in the scopes named.

        Sweeps every type in :data:`EVAL_DOC_TYPES` plus the host's own documents named at
        construction, so a wipe of a scope leaves nothing behind. Between them the two name
        **every** ``doc_type`` the store can hold, which is what makes this a wipe rather than a
        partial sweep.

        The scopes are the host's to name: the engine has no way to enumerate them and does not
        assume the store can. A scope the store holds nothing in is swept to zero, which is
        indistinguishable from a wipe that found it already empty — so naming the wrong scope
        is answered with honest zeroes, not a refusal.

        Callers must gate this behind explicit operator confirmation — the
        action is unrecoverable.

        Args:
            scopes: The scopes to wipe.

        Returns:
            ``{doc_type: count_deleted, ...}`` for each swept doc_type, summed over the scopes.
        """
        counts: dict[str, int] = dict.fromkeys((*EVAL_DOC_TYPES, *self._host_doc_types), 0)
        for scope_id in scopes:
            for doc_type in counts:
                deleted = self._delete_every(doc_type, scope_id)
                counts[doc_type] += deleted
                if deleted:
                    log.warning("nuke_all_eval_data: dropped %d %s docs from scope=%s", deleted, doc_type, scope_id)
        return counts

    def drop_all_analyses(self, scopes: Iterable[str]) -> int:
        """Delete every stored analysis in the scopes named, leaving every other eval document in place.

        The operator step for a boundary that retires only the analysis shape: analyses
        regenerate from their campaigns, and nothing depends on one existing — an insight names
        its source analysis as provenance, a reporter case pins its recorded memo as text. It
        iterates ids and deletes raw, like :meth:`nuke_all_eval_data`, because the documents it
        exists to remove are exactly the ones no current model can read.

        Callers must gate this behind explicit operator confirmation — the action is unrecoverable.

        Args:
            scopes: The scopes whose analyses to drop.

        Returns:
            How many analyses were deleted.
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
            JobStore, RunStore, ResultStore, RunRecordStore, DefinitionStore, CassetteStore, OutOfRunSpendStore
        ] = (
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
    "EVAL_DOC_TYPES",
    "CassetteStore",
    "DefinitionStore",
    "EvalStorage",
    "JobStore",
    "NonTerminalRunScan",
    "ResultStore",
    "RunRecordStore",
    "RunStore",
    "save_document",
]
