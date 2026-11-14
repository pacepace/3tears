"""One campaign's closure, exported as the three files a writer case is minted from: the mint's producer.

A host mints a held-out writer case from one of its own campaigns
(:func:`~threetears.evals.harness.writer_case_from_export`, ``python -m threetears.evals.harness mint``) out of three
files. :func:`campaign_export` assembles them, read-only, from the host's store:

- ``documents``: every stored document reachable from the campaign, verbatim as the store returns it (each keeps its
  ``id``, ``scope_id`` and ``doc_type``), each ``(doc_type, id)`` once, in ``(doc_type, id)`` order;
- ``profile``: the host's declarations (:class:`ProfileDeclaration`), with ``host_declarations_digest``
  (:func:`~threetears.evals.analysis.host_declarations_digest`, the digest every analysis bundle records) and
  ``built_by``, the ``module:callable`` the profile came from, as provenance: the mint checks the digest and never
  imports ``built_by``, since its operator names the factory;
- ``manifest``: the documents that reasoned on the campaign (so a case withholds the notes they lead to), the count of
  every document type the engine stores, what the closure could not resolve, and the package version that wrote it.

**The closure follows the stored links, never the scope.** It starts at the campaign document and follows each link
field the stored documents carry (a campaign's runs, a run's template and cases, a result's run, a report document's
campaign); a document of a sibling campaign in the same scope is reached only through a link the campaign's own
documents hold. Two rules are wider than a link: every insight on the campaign's SUBJECT is included, whichever
campaign minted it, because an analysis reads insights by subject; and the launches and judge profiles naming any of
the campaign's runs are found by a whole-type scan, since a store compares scalar fields only (and a judge config's
rubric dim by the dim's nested name, the same way). A run and a result record their judges in two forms under one field
name: a run the bare config id it committed to, a result the configured judge's identity, ``<config id>@<sha12>``, read
by the package's own reader (:func:`~threetears.evals.schema.judge_config_of`), or a built-in prompt's
``builtin:<sha12>``. A run's built-in dim has no key (``RunJudge.config_ids``); a ``builtin:`` value on a run names no
config and is listed with the built-in identities, never refused. A value in the other document type's form, or one the reader refuses, is refused here, naming
the document and the form its type records. The manifest states every rule (``closure_rules``), and the types no
link reaches.

**An export is a pure function of the stored closure.** Nothing in it records when it was made, so two exports of an
unchanged closure are equal, and its files (:meth:`CampaignExport.files`) are canonical JSON, the same bytes.

**Paged for an action.** A closure can run to many megabytes (each result's trace among them), so the
``campaign_export`` action answers a page at a time (``paged=True``): whole documents in the export's order, up to
:attr:`CampaignExportPage.byte_ceiling` bytes of them, each page carrying the profile and the manifest whole, and
naming the document the next page starts after. The pages' documents, joined, are the single call's. Paging bounds a
response for a client, not the read: every page call reads the whole closure, so a very large campaign is exported
once, whole. So the action holds a ceiling (``max_bytes``, its host's
:attr:`~threetears.evals.ops.OpsHost.campaign_export_max_bytes`, in characters of canonical JSON), counted as the
closure is gathered and refused the moment it passes. Under a ceiling no read by id or by field returns more than a
chunk of whole documents: a field read fetches ids first, every other field its type's model declares left out, then
the documents a chunk at a time, so those reads leave a refused call holding the ceiling and one chunk. The three
whole-type scans (rubric dims, launches, judge profiles) are outside that bound: each reads its small type whole in
one read, however many documents it holds, and only the documents a scan keeps are counted. The command line and the function, which
read the closure once in a process of their own, hold no ceiling and read each type in one read.

Pages are not a snapshot: a write between two calls can drop a document from the join or move the manifest. The
manifest's ``closure_keys_digest`` hashes the closure's ``(doc_type, id)`` keys, not its documents, so it detects a
document added, removed or replaced under another id between two calls, and a page whose digest differs from the first
page's was read after the closure moved. It does not detect a document edited in place under its own id: an equal
digest says the closure holds the same documents by key, not that none of them changed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis import host_declarations_digest
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.kernel import MeasureFamily, MetricDescriptor
from threetears.evals.kernel.build_stamp import package_version
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.storage import UnreadableRecordError
from threetears.evals.kernel.host import Bar, EvalHost, HostProfile, PassThreshold
from threetears.evals.kernel.remedies import FIND_CAMPAIGN
from threetears.evals.schema import DOC_TYPE_FACTS, is_builtin_judge_config_id, judge_config_of
from threetears.evals.schema.hashing import canonical_digest, canonical_json
from threetears.evals.kernel.analyst_gap import EvalAnalystGap
from threetears.evals.kernel.campaign import (
    EvalAnalysis,
    EvalAnalysisAttempt,
    EvalGenerationSlot,
    EvalInsight,
    EvalReportPlan,
    EvalSweep,
)
from threetears.evals.kernel.report_doc import EvalReportDoc, EvalReportSession
from threetears.evals.kernel.report_snapshot import EvalReportSnapshot
from threetears.evals.run.judge_second import EvalSecondJudgePass
from threetears.evals.run.judge_temperature import EvalJudgeTemperatureCheck
from threetears.evals.schema.models import (
    CalibrationRating,
    CaseSet,
    EvalCassette,
    EvalResult,
    EvalTrace,
    JudgeConfigTombstone,
)
from threetears.evals.schema.out_of_run_spend import OutOfRunSpend

#: Every model here is a value inside an export, never a stored document: immutable, refusing a key it does not
#: declare. Not ``EvalBaseModel``, whose subclasses declaring a ``doc_type`` are the stored documents, and a
#: ``doc_type`` here names one of the host's stored documents instead.
_VALUE = ConfigDict(frozen=True, extra="forbid")

#: The most bytes of documents (each as canonical JSON) one page of a paged export holds; a document larger than it is
#: a page alone. An action's result is read whole into a client, and a page at this ceiling stays near a megabyte.
_PAGE_BYTES: Final = 1_000_000

#: The most whole documents one store read by id or by field returns while a ceiling holds (:meth:`_Closure._fetch`):
#: a read's documents arrive together before any is counted, so what those reads hold is at most the ceiling and one
#: chunk. A whole-type scan (:meth:`_Closure.scan`) is not chunked. Sixteen keeps
#: a chunk of the heaviest stored documents (a trace of a long conversation, a report snapshot's frozen bundle, at a
#: few hundred kilobytes to a few megabytes each) to tens of megabytes at worst, at one store round trip per sixteen.
_CHUNK: Final = 16

#: Every field each type's model declares but its id, for the read of ids a ceilinged ``where`` makes before it reads
#: the documents in chunks. Only the types the closure reads by field: the others are read by id.
_BEYOND_ID: Final[Mapping[str, tuple[str, ...]]] = {
    str(model.model_fields["doc_type"].default): tuple(field for field in model.model_fields if field != "id")
    for model in (
        CalibrationRating,
        CaseSet,
        EvalAnalysis,
        EvalAnalysisAttempt,
        EvalAnalystGap,
        EvalCassette,
        EvalGenerationSlot,
        EvalInsight,
        EvalJudgeTemperatureCheck,
        EvalReportDoc,
        EvalReportPlan,
        EvalReportSession,
        EvalReportSnapshot,
        EvalResult,
        EvalSecondJudgePass,
        EvalSweep,
        EvalTrace,
        JudgeConfigTombstone,
        OutOfRunSpend,
    )
}

#: What a caller does about a closure past the action's ceiling: export it once, in a process of its own.
_PAST_CEILING: Final = (
    "export it in a process of its own, which reads the closure once and holds no ceiling: python -m "
    "threetears.evals.harness export --host module:callable --scope SCOPE --campaign ID --out DIR, or "
    "threetears.evals.ops.campaign_export(host, scope_id, campaign_id, built_by=...)"
)

#: What a caller does about a stored judge identity in a form its document type does not record: the record is wrong.
_UNREAD_JUDGE: Final = (
    "the stored record is what is wrong: correct its judge_config_ids to the form its document type records, "
    "then export again"
)

#: The form each document type records its judges in, under the one field name ``judge_config_ids``.
_RECORDED_JUDGE_FORM: Final[Mapping[str, str]] = {
    "eval_run": "an eval_run records the bare config id it committed to",
    "eval_result": "an eval_result records <config id>@<sha12> or builtin:<sha12>",
}

#: The most characters a ``reasoned_on`` name holds; a longer one is cut at a word boundary and ends in an ellipsis.
_NAME_CHARS: Final = 80

#: The engine's document types no link from a campaign reaches, each with why.
_NOT_REACHABLE: Final[Mapping[str, str]] = {
    "eval_action_job": "an action job names no campaign and no run, so no link reaches it",
    "rubric_dim_tombstone": (
        "a dim tombstone carries the deleted dim's key and id; a judge config (or its tombstone) names its dim by NAME,"
        " which neither is, so no link reaches it"
    ),
}

#: The rules the closure follows, in the order it follows them, as the manifest states them.
_CLOSURE_RULES: Final[tuple[str, ...]] = (
    "the campaign document, by id; a campaign of another id is reached by no link, whatever runs it shares",
    (
        "the runs its aspects name (aspects[].run_ids), by id; each run's template and test cases by id, its case set by"
        " the (name, version) the run pins, its cassettes by its corpus id, its sweep by its launch group id"
    ),
    (
        "each run's recorded judge configs and each result's recorded judge identities: a run's bare config id (the set"
        " it committed to at launch) and a result's <config id>@<sha12> (read by the package's judge_config_of) name its"
        " stored judge config (or the config's tombstone), builtin:<sha12> names the engine's own wording and no"
        " document (listed in builtin_judge_config_ids), and a value in the other type's form, or one the reader"
        " refuses, refuses the export; each config's rubric dim by the dim's NAME (rubric_dim.dim.name, every stored"
        " version), by a whole-type scan since the name is nested"
    ),
    (
        "each run's results and traces (the documents whose eval_run_id is the run's id), its calibration ratings,"
        " second judge passes, judge temperature checks and out-of-run spend (whose run_id is); the campaign's own"
        " out-of-run spend, sweeps, analysis attempts and generation slot (whose campaign_id is); every document is"
        " exported verbatim under its stored id, and how the export reached it binds no reader"
    ),
    (
        "the campaign's analyses, the eval_report_plan documents the retired planner left, report snapshots and report"
        " documents; each report document's sessions, each snapshot's and report document's analyst gaps"
    ),
    (
        "every insight on the campaign's SUBJECT in the scope, whichever campaign minted it (an analysis reads insights"
        " by subject); a sibling campaign's insight on another subject is excluded"
    ),
    (
        "launches and judge profiles naming any of the campaign's runs, and judge profiles of its judge configs, by a"
        " whole-type scan (list-valued links a store cannot filter on)"
    ),
    (
        "reasoned_on names a report document `report <id>: <campaign name>`, never by its headline: a stored headline"
        " carries the writer's figures as unrendered keys ({cases|words}), which no page prints; every name is at most"
        f" {_NAME_CHARS} characters, cut at a word boundary"
    ),
)


# --- the models ------------------------------------------------------------------------------------------------------


class ReasonedOn(BaseModel):
    """One stored document that reasoned on a campaign, as a host's mint manifest names it."""

    model_config = _VALUE

    doc_type: str = Field(min_length=1, description="Its document type: an analysis, a report, an insight.")
    id: str = Field(min_length=1, description="Its id in the host's store.")
    name: str | None = Field(
        default=None,
        min_length=1,
        description="What a reader knows it by: an insight's statement, a report's title, an analysis's date and model.",
    )


class SweepableDeclaration(BaseModel):
    """One sweepable a host declares, as its profile declaration records it: everything but its callables (its reader,
    and an open family's ``member_reader_name``)."""

    model_config = _VALUE

    name: str
    role: str
    indeterminate_when_blank: bool
    rig_dimension: bool
    open_family: str | None
    confounds: str | None
    reader_name: str | None = Field(
        default=None,
        description="What a reader calls the lever (Sweepable.reader_name), as the host declared it; None where it "
        "declared none (an apparatus or label input, an open family, which names its members by a callable). A record, "
        "never identity: it enters no host declarations digest.",
    )


class KindDeclaration(BaseModel):
    """One candidate kind's contract, as much of it as is a declaration: its name, its lever namespace, its seats."""

    model_config = _VALUE

    kind: str
    lever_prefix: str
    seats: list[str] | None = Field(description="The seats it declares, sorted; None for a kind that declares none.")


class ProfileDeclaration(BaseModel):
    """A host profile's declarations, serialized: everything of a :class:`~threetears.evals.kernel.host.HostProfile`
    that is not a callable.

    The mint checks ``host_declarations_digest``, which the live profile its operator names must reproduce (the freeze
    calls the profile's readers, which no file carries). ``built_by`` is provenance, never imported: a caller of the
    ``campaign_export`` action chooses it, so the mint runs only the factory its operator names. The rest is the
    host's record of what its campaign was measured under.
    """

    model_config = _VALUE

    host_id: str
    built_by: str | None = Field(
        description="Provenance: the module:callable the profile came from, as its exporter named it; None when it "
        "named none. The mint never imports it; its operator names the factory (--host)."
    )
    host_declarations_digest: str = Field(
        min_length=64,
        max_length=64,
        description="The digest of the declarations that partition observations, as every analysis bundle records it.",
    )
    sweepables: list[SweepableDeclaration]
    measures: list[MetricDescriptor]
    measure_families: list[MeasureFamily]
    bars: list[Bar]
    pass_thresholds: list[PassThreshold]
    world: list[str] | None = Field(description="The world's dimension names; None for a host that instantiates none.")
    kinds: list[KindDeclaration]
    release_label: str | None
    observed_model_levers: dict[str, str]
    analysis_writer_models: list[str]


class UnresolvedLink(BaseModel):
    """A document the campaign's documents name that the scope does not hold: by ``id``, or by the ``field`` named."""

    model_config = _VALUE

    doc_type: str
    id: str = Field(description="The value the link carries: the document's id, or the value of `field`.")
    field: str = Field(default="id", description="The field the link names the document by; `id` for an id.")


class CampaignExportManifest(BaseModel):
    """What one export holds and how it was assembled: the third file, which the mint reads ``reasoned_on`` from."""

    model_config = _VALUE

    campaign_id: str
    scope_id: str
    subject_id: str = Field(description="The campaign's subject, whose insights the closure includes.")
    reasoned_on: list[ReasonedOn] = Field(
        description="Every analysis, report document and insight in the closure, in the export's order."
    )
    doc_type_counts: dict[str, int] = Field(description="Every document type the engine stores; zero where none.")
    closure_documents: int = Field(description="How many documents the whole closure holds, on every page.")
    closure_keys_digest: str = Field(
        min_length=64,
        max_length=64,
        description="sha256 of the closure's sorted (doc_type, id) keys: pages of one export carry the same, and a "
        "page whose differs from the first page's was read after the closure moved.",
    )
    not_reachable: list[str] = Field(description="Types no link from a campaign reaches; closure_rules says why.")
    closure_rules: list[str] = Field(description="How the closure was followed, one rule a line.")
    unresolved: list[UnresolvedLink] = Field(description="Links the campaign's documents hold that the scope does not.")
    builtin_judge_config_ids: list[str] = Field(
        description="Recorded judge identities of the engine's built-in wording (builtin:<sha12>): judges with no "
        "document."
    )
    package_version: str | None = Field(description="The 3tears-evals version that exported it; None when unknown.")


class CampaignExportPage(BaseModel):
    """Which documents one page of a paged export holds, and the call that reads the next."""

    model_config = _VALUE

    byte_ceiling: int = Field(
        description="The most bytes of documents (each as canonical JSON) a page holds; a larger document is a page "
        "alone."
    )
    documents_before: int = Field(description="Documents of the export the pages before this one hold.")
    documents_after: int = Field(description="Documents of the export the pages after this one hold.")
    next_after_doc_type: str | None = Field(
        description="The type of the document the next page starts after, which the next call names; None on the last."
    )
    next_after_document_id: str | None = Field(
        description="The id of the document the next page starts after, which the next call names; None on the last."
    )


class CampaignExport(BaseModel):
    """One campaign's closure: its documents, the host's profile declaration and the manifest.

    ``page`` is set on one page of a paged export and ``None`` on an export holding every document; the pages'
    ``documents``, joined in order, are the whole export's, and each carries the same profile and manifest.
    """

    model_config = _VALUE

    documents: list[dict[str, Any]] = Field(
        description="Every document of the closure (of this page, when paged), verbatim as stored, in (doc_type, id) "
        "order."
    )
    profile: ProfileDeclaration
    manifest: CampaignExportManifest
    page: CampaignExportPage | None = Field(
        default=None,
        description="On one page of a paged export, which documents it holds and where the next starts; None on an "
        "export holding every document.",
    )

    def files(self) -> dict[str, bytes]:
        """The three files the mint reads, by name: each canonical JSON (sorted keys, no incidental whitespace).

        :return: ``documents.json``, ``profile.json`` and ``manifest.json``, each its bytes.
        :rtype: dict[str, bytes]
        :raises ValueError: This is one page of a paged export, which holds only part of the documents.
        """
        if self.page is not None:
            raise ValueError("one page of a paged export holds part of the documents: join the pages, or export whole")
        return {
            "documents.json": _file_bytes(self.documents),
            "profile.json": _file_bytes(self.profile.model_dump(mode="json")),
            "manifest.json": _file_bytes(self.manifest.model_dump(mode="json")),
        }


def _file_bytes(payload: Any) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


# --- the profile -----------------------------------------------------------------------------------------------------


def _declaration(profile: HostProfile, built_by: str | None) -> ProfileDeclaration:
    """``profile``'s declarations, its digest computed by the package's own function."""
    return ProfileDeclaration(
        host_id=profile.host_id,
        built_by=built_by,
        host_declarations_digest=host_declarations_digest(profile),
        sweepables=[
            SweepableDeclaration(
                name=declared.name,
                role=declared.role,
                indeterminate_when_blank=declared.indeterminate_when_blank,
                rig_dimension=declared.rig_dimension,
                open_family=declared.open_family,
                confounds=declared.confounds,
                reader_name=declared.reader_name,
            )
            for declared in profile.sweepables.declarations
        ],
        measures=[
            descriptor for name in profile.measures.names if (descriptor := profile.measures.get(name)) is not None
        ],
        measure_families=list(profile.measures.families),
        bars=list(profile.bars.bars),
        pass_thresholds=list(profile.bars.pass_thresholds),
        world=list(profile.world.names) if profile.world is not None else None,
        kinds=[
            KindDeclaration(
                kind=contract.kind,
                lever_prefix=contract.lever_prefix,
                seats=sorted(contract.seats) if contract.seats is not None else None,
            )
            for contract in profile.kinds
        ],
        release_label=profile.release_label,
        observed_model_levers=dict(profile.observed_model_levers),
        analysis_writer_models=list(profile.analysis_writer_models),
    )


def _checked_built_by(built_by: str | None) -> str | None:
    """``built_by`` as provenance names a factory, ``module:callable``, or a refusal: other text names no code."""
    if built_by is None:
        return None
    module_name, colon, attribute = built_by.partition(":")
    if not colon or not module_name.strip() or not attribute.strip():
        raise ValidationFailedError(
            f"built_by names a profile factory as module:callable, not {built_by!r}",
            remedy="name the zero-argument callable that returns the host's profile, or its host, as "
            "package.module:callable; or leave it out",
        )
    return built_by


# --- the closure -----------------------------------------------------------------------------------------------------


def _unique(ids: Iterable[str | None]) -> list[str]:
    """The ids given, blanks dropped, each once, in first-seen order."""
    return list(dict.fromkeys(doc_id for doc_id in ids if doc_id))


class _Closure:
    """The documents gathered so far, keyed ``(doc_type, id)``, and what the scope did not hold."""

    def __init__(self, host: EvalHost, scope_id: str, campaign_id: str, max_bytes: int | None) -> None:
        self._storage = host.storage
        self._scope = scope_id
        self._campaign_id = campaign_id
        self._max_bytes = max_bytes
        self.kept_bytes = 0
        self.documents: dict[tuple[str, str], dict[str, Any]] = {}
        self.unresolved: list[UnresolvedLink] = []
        self.builtin_judge_config_ids: list[str] = []

    def _keep(self, doc_type: str, documents: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        kept = []
        for document in documents:
            key = (doc_type, str(document["id"]))
            if key not in self.documents:
                self.documents[key] = dict(document)
                self._count(document)
            kept.append(self.documents[key])
        return kept

    def _count(self, document: Mapping[str, Any]) -> None:
        """Add a kept document's size to the running total, refusing the moment it passes the ceiling.

        Counted as each document is kept, and every read by id or by field under a ceiling returns at most
        :data:`_CHUNK` whole documents (:meth:`_fetch`), so those reads leave a refused export holding at most the
        ceiling and one chunk. A whole-type scan (:meth:`scan`) returns its type whole, past the chunk, and only what
        the caller keeps of it is counted here.
        """
        if self._max_bytes is None:
            return
        self.kept_bytes += len(canonical_json(document))
        if self.kept_bytes > self._max_bytes:
            forms = PageForms.off_page()
            raise ValidationFailedError(
                f"the closure of campaign {self._campaign_id!r} passed the export ceiling of "
                f"{forms.count(self._max_bytes).text} bytes, at {forms.count(self.kept_bytes).text} bytes after "
                f"{forms.count(len(self.documents)).text} documents: every page call of "
                "the action reads the whole closure, so a closure past the ceiling is not served here",
                {"max_bytes": self._max_bytes, "kept_bytes": self.kept_bytes, "kept_documents": len(self.documents)},
                remedy=_PAST_CEILING,
            )

    def _fetch(self, doc_type: str, ids: Sequence[str]) -> list[dict[str, Any]]:
        """The documents of ``doc_type`` with these ids, kept: those already held from memory, the rest read whole.

        Under a ceiling they are read :data:`_CHUNK` at a time, each chunk counted before the next is read, so no read
        returns more than a chunk; without one, in one read.
        """
        found = [self.documents[key] for doc_id in ids if (key := (doc_type, doc_id)) in self.documents]
        missing = [doc_id for doc_id in ids if (doc_type, doc_id) not in self.documents]
        step = _CHUNK if self._max_bytes is not None else max(len(missing), 1)
        for start in range(0, len(missing), step):
            chunk = missing[start : start + step]
            found.extend(self._keep(doc_type, self._storage.stored_documents(doc_type, self._scope, ids=chunk)))
        return found

    def by_ids(self, doc_type: str, ids: Sequence[str], *, record_missing: bool = True) -> list[dict[str, Any]]:
        """The documents of ``doc_type`` with these ids; with ``record_missing``, an id not held is unresolved."""
        wanted = _unique(ids)
        if not wanted:
            return []
        found = self._fetch(doc_type, wanted)
        if record_missing:
            held = {document["id"] for document in found}
            self.unresolved.extend(
                UnresolvedLink(doc_type=doc_type, id=doc_id) for doc_id in wanted if doc_id not in held
            )
        return found

    def where(self, doc_type: str, **field_eq: Any) -> list[dict[str, Any]]:
        """The documents of ``doc_type`` whose scalar fields equal ``field_eq``.

        Under a ceiling, their ids are read first, every other field the type's model declares left out by the store,
        and the documents then read whole a chunk at a time (:meth:`_fetch`): one run's results or traces never arrive
        in one read. Without one, they are read whole in one read.
        """
        if self._max_bytes is None:
            return self._keep(doc_type, self._storage.stored_documents(doc_type, self._scope, **field_eq))
        light = self._storage.stored_documents(doc_type, self._scope, exclude=_BEYOND_ID[doc_type], **field_eq)
        return self._fetch(doc_type, _unique(str(document["id"]) for document in light))

    def scan(self, doc_type: str) -> list[dict[str, Any]]:
        """Every document of ``doc_type`` in the scope, NOT kept: for a list-valued link the caller filters itself.

        One read of the whole type, under a ceiling too: it is not chunked, and nothing it returns is counted until
        the caller keeps it. Only small types are scanned (rubric dims, launches, judge profiles).
        """
        return self._storage.stored_documents(doc_type, self._scope)

    def keep(self, doc_type: str, documents: Iterable[Mapping[str, Any]]) -> None:
        """Keep documents a scan selected."""
        self._keep(doc_type, documents)

    def ids_of(self, doc_type: str) -> list[str]:
        """The ids held of ``doc_type``, in the order gathered."""
        return [doc_id for held_type, doc_id in self.documents if held_type == doc_type]


def _unread_judge(document: Mapping[str, Any], recorded: str, scope_id: str, why: str) -> UnreadableRecordError:
    """The refusal of a recorded judge value its document's type does not record, naming the record and its form."""
    return UnreadableRecordError(
        f"the {document['doc_type']} {document['id']!r} in scope {scope_id!r} records a judge identity the package "
        f"does not read: {why}; {_RECORDED_JUDGE_FORM[document['doc_type']]}",
        remedy=_UNREAD_JUDGE,
        details={"doc_type": document["doc_type"], "id": document["id"], "recorded": recorded},
    )


def _recorded_config(document: Mapping[str, Any], recorded: str, scope_id: str) -> str | None:
    """The stored config one recorded value names, read in the form its document's type records, or a refusal.

    A run records the config set it committed to at launch, so its value IS the config's id; a result records the
    prompt that asked it, so its value names the config before the reply format's fingerprint.
    """
    if document["doc_type"] == "eval_run":
        if not recorded:
            raise _unread_judge(document, recorded, scope_id, f"{recorded!r} names no config")
        try:
            judge_config_of(recorded)
        except ValueError:
            return recorded
        raise _unread_judge(
            document, recorded, scope_id, f"{recorded!r} is a result's judge identity, the prompt sent, not a config"
        )
    try:
        return judge_config_of(recorded)
    except ValueError as unread:
        raise _unread_judge(document, recorded, scope_id, str(unread)) from unread


def _gather(closure: _Closure, campaign: Mapping[str, Any], scope_id: str) -> None:
    """Follow every link from ``campaign`` into ``closure``."""
    # A campaign carries its runs under its aspects (``aspects[].run_ids``) and nowhere else.
    run_ids = _unique(rid for aspect in campaign.get("aspects") or [] for rid in aspect.get("run_ids") or [])
    runs = closure.by_ids("eval_run", run_ids)
    held_run_ids = [run["id"] for run in runs]

    results: list[dict[str, Any]] = []
    for run_id in held_run_ids:
        results.extend(closure.where("eval_result", eval_run_id=run_id))
        closure.where("eval_trace", eval_run_id=run_id)
        closure.where("calibration_rating", run_id=run_id)
        closure.where("eval_second_judge_pass", run_id=run_id)
        closure.where("eval_judge_temperature_check", run_id=run_id)
        closure.where("eval_out_of_run_spend", run_id=run_id)

    # Definitions the runs ran: templates, cases, case sets, judge configs and their rubric dims, cassettes.
    case_sets: list[dict[str, Any]] = []
    for run in runs:
        pinned = run.get("case_set")
        if pinned:
            case_sets.extend(closure.where("case_set", name=pinned["name"], version=pinned["version"]))
        if run.get("cassette_corpus_id"):
            closure.where("eval_cassette", corpus_id=run["cassette_corpus_id"])
        if run.get("launch_group_id"):
            closure.where("eval_sweep", launch_group_id=run["launch_group_id"])
    test_cases = closure.by_ids(
        "eval_test_case",
        _unique(
            [
                *(case_id for run in runs for case_id in run.get("test_case_ids") or []),
                *(result.get("test_case_id") for result in results),
                *(case_id for case_set in case_sets for case_id in case_set.get("test_case_ids") or []),
            ]
        ),
    )
    closure.by_ids(
        "eval_template",
        _unique(
            [
                *(run.get("template_id") for run in runs),
                *(case.get("template_id") for case in test_cases),
                *(case_set.get("template_id") for case_set in case_sets),
            ]
        ),
    )
    # A run's value and a result's are read each in its own type's form: None is a dim no judge asked.
    builtin: list[str] = []
    configured: list[str | None] = []
    for document in (*runs, *results):
        for recorded in (document.get("judge_config_ids") or {}).values():
            if recorded is None:
                continue
            if is_builtin_judge_config_id(recorded):
                builtin.append(recorded)
            else:
                configured.append(_recorded_config(document, recorded, scope_id))
    closure.builtin_judge_config_ids = sorted(set(builtin))
    judge_config_ids = _unique(configured)
    judge_configs = closure.by_ids("judge_config", judge_config_ids, record_missing=False)
    held_configs = {config["id"] for config in judge_configs}
    tombstoned_configs = [
        tombstone
        for config_id in judge_config_ids
        if config_id not in held_configs
        for tombstone in closure.where("judge_config_tombstone", deleted_config_id=config_id)
    ]
    resolved_configs = held_configs | {tombstone["deleted_config_id"] for tombstone in tombstoned_configs}
    closure.unresolved.extend(
        UnresolvedLink(doc_type="judge_config", id=config_id)
        for config_id in judge_config_ids
        if config_id not in resolved_configs
    )
    # A config names its dim by NAME; the stored dim carries that name nested (``dim.name``), which a store cannot
    # filter on, so every stored version of each named dim is selected here over the whole type.
    dim_names = _unique(document.get("rubric_dim_id") for document in (*judge_configs, *tombstoned_configs))
    if dim_names:
        wanted = set(dim_names)
        dims = [dim for dim in closure.scan("rubric_dim") if (dim.get("dim") or {}).get("name") in wanted]
        closure.keep("rubric_dim", dims)
        found = {dim["dim"]["name"] for dim in dims}
        closure.unresolved.extend(
            UnresolvedLink(doc_type="rubric_dim", id=name, field="dim.name") for name in dim_names if name not in found
        )

    # What reasoned on the campaign, and the records of doing so.
    campaign_id = str(campaign["id"])
    closure.where("eval_out_of_run_spend", campaign_id=campaign_id)
    closure.where("eval_sweep", campaign_id=campaign_id)
    closure.where("eval_analysis_attempt", campaign_id=campaign_id)
    closure.where("eval_generation_slot", campaign_id=campaign_id)
    analyses = closure.where("eval_analysis", campaign_id=campaign_id)
    closure.where("eval_report_plan", campaign_id=campaign_id)
    closure.by_ids("eval_report_plan", _unique(analysis.get("report_plan_id") for analysis in analyses))
    snapshots = closure.where("eval_report_snapshot", campaign_id=campaign_id)
    closure.where("eval_report_doc", campaign_id=campaign_id)
    closure.by_ids("eval_report_doc", _unique(analysis.get("report_doc_id") for analysis in analyses))
    for report_doc_id in closure.ids_of("eval_report_doc"):
        closure.where("eval_report_session", report_doc_id=report_doc_id)
        closure.where("eval_analyst_gap", report_doc_id=report_doc_id)
    for snapshot in snapshots:
        closure.where("eval_analyst_gap", snapshot_id=snapshot["id"])
    closure.where("eval_insight", subject_id=campaign["subject_id"])

    # List-valued links: a store compares scalars, so these are selected here over the whole type.
    run_set = set(held_run_ids)
    closure.keep(
        "eval_launch", (launch for launch in closure.scan("eval_launch") if run_set & set(launch.get("run_ids") or []))
    )
    closure.keep(
        "eval_judge_profile",
        (
            profile
            for profile in closure.scan("eval_judge_profile")
            if run_set & set(profile.get("run_ids") or []) or profile.get("judge_config_id") in held_configs
        ),
    )


def _short(text: str) -> str | None:
    """``text`` with its whitespace collapsed and cut to at most :data:`_NAME_CHARS` characters at a word boundary,
    trailing punctuation stripped before the ellipsis; ``None`` when nothing is left.

    A cut never ends mid-word: the mint withholds a note that holds the name, and a word's head matches nothing.
    """
    text = " ".join(text.split())
    if len(text) <= _NAME_CHARS:
        return text or None
    room = text[: _NAME_CHARS - 1]
    cut = room.rsplit(" ", 1)[0] if " " in room else room
    return cut.rstrip(" ,;:.-") + "…"


def _reasoned_on(documents: Iterable[Mapping[str, Any]], campaign_name: str) -> list[ReasonedOn]:
    """Name each analysis, report document and insight among ``documents``.

    A report document is named ``report <id>: <campaign name>``, never by its headline: the stored headline is the
    writer's prose with its figures as unrendered keys (``{cases|words}``), which no page prints and no note repeats.
    The id leads, so a cut never reaches it.
    """
    named: list[ReasonedOn] = []
    for document in documents:
        doc_type = document["doc_type"]
        if doc_type == "eval_analysis":
            generation = document.get("generation") or {}
            name = _short(f"analysis {document.get('created_at', '')} {generation.get('generator_model', '')}")
        elif doc_type == "eval_report_doc":
            name = _short(f"report {document['id']}: {campaign_name}")
        elif doc_type == "eval_insight":
            name = _short(str(document.get("statement") or ""))
        else:
            continue
        named.append(ReasonedOn(doc_type=doc_type, id=str(document["id"]), name=name))
    return named


def _page(export: CampaignExport, after_doc_type: str | None, after_document_id: str | None) -> CampaignExport:
    """One page of ``export``: whole documents after the cursor, up to :data:`_PAGE_BYTES` of them."""
    keys = [(str(document["doc_type"]), str(document["id"])) for document in export.documents]
    start = 0
    if after_doc_type is not None and after_document_id is not None:
        cursor = (after_doc_type, after_document_id)
        start = next((at for at, key in enumerate(keys) if key > cursor), len(keys))
    if start == 0 and sum(len(canonical_json(document)) for document in export.documents) <= _PAGE_BYTES:
        return export
    end, spent = start, 0
    while end < len(keys):
        size = len(canonical_json(export.documents[end]))
        if end > start and spent + size > _PAGE_BYTES:
            break
        spent += size
        end += 1
    last = keys[end - 1] if end < len(keys) and end > start else None
    page = CampaignExportPage(
        byte_ceiling=_PAGE_BYTES,
        documents_before=start,
        documents_after=len(keys) - end,
        next_after_doc_type=last[0] if last is not None else None,
        next_after_document_id=last[1] if last is not None else None,
    )
    return export.model_copy(update={"documents": export.documents[start:end], "page": page})


def _cursor(paged: bool, after_doc_type: str | None, after_document_id: str | None) -> None:
    """Refuse a cursor a paged export would misread: half of one, one on a whole export, a type the engine lacks."""
    if after_doc_type is None and after_document_id is None:
        return
    if not paged:
        raise ValidationFailedError(
            "a cursor names a page of a paged export, and this export is whole",
            remedy="export with paged=True to read it a page at a time, or name no cursor",
        )
    if after_doc_type is None or after_document_id is None:
        raise ValidationFailedError(
            "a page's cursor is a document: its type and its id together",
            remedy="name both, as the previous page's next_after_doc_type and next_after_document_id",
        )
    if after_doc_type not in DOC_TYPE_FACTS:
        raise ValidationFailedError(
            f"{after_doc_type!r} is no document type the engine stores, so it names no page's place",
            remedy="name the previous page's next_after_doc_type, as it gave it",
        )


def campaign_export(
    host: EvalHost,
    scope_id: str,
    campaign_id: str,
    *,
    built_by: str | None,
    paged: bool = False,
    after_doc_type: str | None = None,
    after_document_id: str | None = None,
    max_bytes: int | None = None,
) -> CampaignExport:
    """Assemble one campaign's closure, read-only: the documents, the profile declaration and the manifest a writer
    case is minted from.

    :param host: The host whose store holds the campaign, and whose profile its runs were measured under.
    :ptype host: EvalHost
    :param scope_id: The scope the campaign lives in.
    :ptype scope_id: str
    :param campaign_id: The campaign.
    :ptype campaign_id: str
    :param built_by: Provenance: the ``module:callable`` ``host``'s profile (or ``host``) comes from, recorded in the
        profile; ``None`` to name none. The mint never imports it: its operator names the factory (``--host``).
    :ptype built_by: str | None
    :param paged: Read one page of the export (:class:`CampaignExportPage`) rather than all of it.
    :ptype paged: bool
    :param after_doc_type: With ``paged``, the previous page's ``next_after_doc_type``; ``None`` for the first page.
    :ptype after_doc_type: str | None
    :param after_document_id: With ``paged``, the previous page's ``next_after_document_id``.
    :ptype after_document_id: str | None
    :param max_bytes: The most bytes of canonical JSON (ASCII, so its characters; a decoded document takes several
        times that on the heap) the kept documents may total, counted as they are gathered, a chunk at a time, and
        refused the moment the total passes; ``None``, the default, for no ceiling. The ``campaign_export``
        action passes its host's (:attr:`~threetears.evals.ops.OpsHost.campaign_export_max_bytes`), since it reads the
        whole closure on every page call; a process exporting once passes none.
    :ptype max_bytes: int | None
    :return: The export: every document in ``(doc_type, id)`` order (one page of them, when paged and more than one
        page's worth), the profile's declaration, the manifest.
    :rtype: CampaignExport
    :raises NotFoundError: No campaign of that id in the scope.
    :raises ValidationFailedError: ``built_by`` is not ``module:callable``, a cursor is given without ``paged``,
        half of one, or with a type the engine does not store, or the closure passes ``max_bytes``.
    :raises UnreadableRecordError: A run or a result records a judge identity the package's reader refuses.
    """
    built = _checked_built_by(built_by)
    _cursor(paged, after_doc_type, after_document_id)
    closure = _Closure(host, scope_id, campaign_id, max_bytes)
    campaigns = closure.by_ids("eval_campaign", [campaign_id], record_missing=False)
    if not campaigns:
        raise NotFoundError("campaign", campaign_id, remedy=FIND_CAMPAIGN)
    (campaign,) = campaigns
    _gather(closure, campaign, scope_id)
    documents = [closure.documents[key] for key in sorted(closure.documents)]
    counts = Counter(document["doc_type"] for document in documents)
    manifest = CampaignExportManifest(
        campaign_id=campaign_id,
        scope_id=scope_id,
        subject_id=str(campaign["subject_id"]),
        reasoned_on=_reasoned_on(documents, str(campaign.get("name") or "")),
        doc_type_counts={doc_type: counts.get(doc_type, 0) for doc_type in DOC_TYPE_FACTS},
        closure_documents=len(documents),
        # The counts alone miss a document replaced by another of its type between two page calls; the keys do not.
        closure_keys_digest=canonical_digest([[doc_type, doc_id] for doc_type, doc_id in sorted(closure.documents)]),
        not_reachable=sorted(_NOT_REACHABLE),
        closure_rules=[*_CLOSURE_RULES, *(f"{doc_type}: {why}" for doc_type, why in sorted(_NOT_REACHABLE.items()))],
        unresolved=closure.unresolved,
        builtin_judge_config_ids=closure.builtin_judge_config_ids,
        package_version=package_version(),
    )
    export = CampaignExport(documents=documents, profile=_declaration(host.profile, built), manifest=manifest)
    return _page(export, after_doc_type, after_document_id) if paged else export


__all__ = [
    "CampaignExport",
    "CampaignExportManifest",
    "CampaignExportPage",
    "KindDeclaration",
    "ProfileDeclaration",
    "ReasonedOn",
    "SweepableDeclaration",
    "UnresolvedLink",
    "campaign_export",
]
