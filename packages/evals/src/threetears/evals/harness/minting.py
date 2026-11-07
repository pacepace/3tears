"""Minting a writer case from a production export: a host whose store may only be read hands over the campaign's
documents, and gets back the portable case file.

A host mints a case from one of its own campaigns by exporting the campaign's documents as stored (its closure: the
campaign, its runs and their results, traces, templates and test cases, and the insights on its subject), naming the
profile its runs were measured under, and freezing them here. Nothing is written to the host's store: the documents
are loaded into a store in memory, and the campaign is frozen through a reading host over it
(:meth:`~threetears.evals.ops.host.OpsHost.reading`), as :func:`~threetears.evals.harness.writer_cases.writer_case_freeze`
freezes any campaign.

**What the door needs, by document type** (:data:`REQUIRED_DOC_TYPES`): the campaign named, every run it names, the
template every run ran, and the runs' results. Each is refused by name when the export lacks it. The traces are
needed when a result records one (a judge's evidence is read from them). The test cases (their strata) and the
insights on the campaign's subject are read when present. Any other document is loaded and never read.

**The same evidence mints the same case.** A snapshot's id is derived from its campaign and its report input, so
it does not move with the clock; the freeze's two readings of the clock (the bundle's assembly and the freeze) are
pinned to one instant, ``as_of``, so the bytes do not move either. Unnamed, the instant is the moment the campaign's
evidence was complete: the latest of the campaign's creation and each run's completion (its creation, for a run that
never completed). An exported insight observed at or after that instant is left out of the store, so no insight a
later analysis of this campaign minted reaches the case's bundle. Whether an insight kept is retracted is read from
the export as it stands: its minting analysis's archived state, as exported.

**The profile is code, and its declaration checks it.** The freeze calls the profile's live readers on every run: each
sweepable's ``read`` (:meth:`~threetears.evals.kernel.host.sweepables.SweepableRegistry.resolve_levers`, from the
bundle's design and configuration lenses), the host's ``variant_levers`` and each kind contract's ``levels``
(:meth:`~threetears.evals.kernel.host.HostProfile.engine_levels`, from its design and coverage lenses). They read a
run's payload in the host's own vocabulary, so no file can carry them. The door takes the live profile; the command
line builds it from its factory (:func:`host_profile_from`). The host's declaration of it (:func:`read_profile_file`)
names that factory and the digest of the declarations that partition its observations
(``host_declarations_digest``, the one every bundle records), and a profile whose digest differs is refused: the
campaign was measured under other sweepables or another world.

**Withholding is by note.** A case withholds notes of the analyst skill, by file name. The documents that reasoned on
the campaign (its analyses, reports and insights, as the host's manifest names them) lead to the notes that hold their
id or their name (an insight's statement, a report's title), read whitespace- and case-blind; a note naming the
campaign's own id is withheld too. The case records which document led to each note (:class:`CaseMint`), and the
documents no note holds, so nothing that reasoned on the campaign is dropped unsaid.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import Field, ValidationError

from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.bundle.schema import host_declarations_digest
from threetears.evals.harness.writer_cases import (
    CaseMint,
    ReasonedOn,
    WithheldNote,
    WriterCaseFile,
    writer_case_freeze,
)
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.host import EvalHost, HostProfile, default_cell_timeout
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.storage import InMemoryDocumentStore

#: The document types a mint refuses to go without, in the order it checks them.
REQUIRED_DOC_TYPES: Final = ("eval_campaign", "eval_run", "eval_template", "eval_result")

#: The remedy every refusal of an incomplete export names.
_EXPORT_REMEDY: Final = (
    "export the campaign's closure: the campaign, its runs, their results and traces, the templates and test cases "
    "they ran, and the insights on its subject"
)


#: Where a skill's notes sit, relative to its root, as :class:`~threetears.evals.analysis.analyst.skill.AnalystSkill`
#: lays them out.
NOTES_DIR: Final = "exemplars/"

#: The fewest characters an id or a name matches a note on, so a one-word title cannot withhold every note.
MIN_MATCH: Final = 12


class MintManifest(EvalBaseModel):
    """What a host says about the campaign it exports: the documents that reasoned on it.

    Read from ``{"campaign_id": ..., "reasoned_on": [{"doc_type": ..., "id": ..., "name": ...}]}``; any other key of
    the file (what the export could not reach, say) is the host's, and not read.
    """

    campaign_id: str = Field(min_length=1, description="The campaign exported; a mint of another is refused.")
    reasoned_on: list[ReasonedOn] = Field(
        default_factory=list, description="Every analysis, report and insight that reasoned on the campaign."
    )

    @classmethod
    def read(cls, path: str | Path) -> MintManifest:
        """Read a manifest from its JSON file.

        :param path: The file.
        :ptype path: str | Path
        :return: The manifest.
        :rtype: MintManifest
        :raises RefusedInput: The file cannot be read, or does not read as a manifest.
        """
        source = Path(path)
        remedy = (
            'write {"campaign_id": "<id>", "reasoned_on": [{"doc_type": "<type>", "id": "<id>", "name": "<name>"}]}'
        )
        try:
            content = json.loads(_text(source))
        except json.JSONDecodeError as unread:
            raise RefusedInput(f"{source} is not JSON: {unread}", remedy=remedy) from unread
        if not isinstance(content, dict):
            raise RefusedInput(f"{source} is not a JSON object", remedy=remedy)
        try:
            return cls.model_validate(
                {"campaign_id": content.get("campaign_id"), "reasoned_on": content.get("reasoned_on", [])}
            )
        except ValidationError as unread:
            first = unread.errors()[0]
            raise RefusedInput(
                f"{source} does not read as a mint manifest: {unread.error_count()} fault(s), the first "
                f"{first['msg']!r} at {'.'.join(str(at) for at in first['loc'])}",
                remedy=remedy,
            ) from unread


@dataclass(frozen=True)
class ProfileFile:
    """What a mint reads of a host's profile declaration: its factory and its declarations' digest.

    :ivar factory: ``module:callable`` building the live profile (the declaration's ``built_by``), or ``None`` when
        it names no callable.
    :ivar digest: The declaration's ``host_declarations_digest``, or ``None`` when it carries none.
    """

    factory: str | None
    digest: str | None


def host_profile_from(spec: str) -> HostProfile:
    """The profile a ``module:factory`` spec builds: the factory's profile, or the profile of the host it returns.

    :param spec: ``package.module:callable``, a zero-argument callable returning a
        :class:`~threetears.evals.kernel.host.HostProfile` or an :class:`~threetears.evals.kernel.host.EvalHost`
        (only its profile is read; nothing is read from or written to its storage).
    :ptype spec: str
    :return: The profile.
    :rtype: HostProfile
    :raises RefusedInput: The spec is not ``module:attribute``, the module does not import, it has no such callable,
        or what it returns is neither a profile nor a host.
    """
    module_name, colon, attribute = spec.partition(":")
    if not colon or not module_name or not attribute:
        raise RefusedInput(
            f"a profile factory is named module:factory, a module path and a callable in it, not {spec!r}",
            remedy="name it as package.module:build_profile",
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as missing:
        raise RefusedInput(
            f"{spec}: cannot import {module_name!r}: {missing}",
            remedy="run the mint in the host's own environment, where its module imports",
        ) from missing
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise RefusedInput(
            f"{spec}: module {module_name!r} has no callable {attribute!r}",
            remedy="name a zero-argument callable returning the host's profile or its host",
        )
    made = factory()
    if isinstance(made, EvalHost):
        return made.profile
    if isinstance(made, HostProfile):
        return made
    raise RefusedInput(
        f"{spec} returned a {type(made).__name__}, not a HostProfile or an EvalHost",
        remedy="return the profile the host's runs were measured under, or the host carrying it",
    )


def read_profile_file(path: str | Path) -> ProfileFile:
    """What a host's profile declaration names: ``built_by`` (``module:callable``) and ``host_declarations_digest``.

    Every other key (the declarations themselves: the sweepables, measures, bars, world, kinds) is the host's record,
    and not read: the digest stands for the declarations that partition observations.

    :param path: The file, a JSON object.
    :ptype path: str | Path
    :return: Its factory and its digest, each ``None`` when it names none.
    :rtype: ProfileFile
    :raises RefusedInput: The file cannot be read, is not a JSON object, or either key is there and not a string.
    """
    source = Path(path)
    remedy = 'write {"built_by": "module:callable", "host_declarations_digest": "<64 hex>", ...}'
    try:
        content = json.loads(_text(source))
    except json.JSONDecodeError as unread:
        raise RefusedInput(f"{source} is not JSON: {unread}", remedy=remedy) from unread
    if not isinstance(content, dict):
        raise RefusedInput(f"{source} is not a JSON object", remedy=remedy)
    built_by, digest = content.get("built_by"), content.get("host_declarations_digest")
    for key, value in (("built_by", built_by), ("host_declarations_digest", digest)):
        if value is not None and not isinstance(value, str):
            raise RefusedInput(f"{source}: {key} is a {type(value).__name__}, not a string", remedy=remedy)
    factory = built_by if isinstance(built_by, str) and ":" in built_by else None
    return ProfileFile(factory=factory, digest=digest if isinstance(digest, str) else None)


def _text(source: Path) -> str:
    """A file's text, or a refusal naming why it cannot be read."""
    try:
        return source.read_text(encoding="utf-8")
    except OSError as unread:
        raise RefusedInput(f"{source} cannot be read: {unread}", remedy="name a file this process can read") from unread


def _folded(text: str) -> str:
    return " ".join(text.split()).casefold()


def _withheld(
    skill: AnalystSkill, campaign_id: str, reasoned_on: Collection[ReasonedOn], named: Collection[str]
) -> tuple[list[str], CaseMint | None]:
    """The notes a case withholds, in the skill's order then as named, and the record of why."""
    notes = skill.analyst().exemplars
    if unknown := sorted(set(named) - set(notes)):
        raise RefusedInput(
            f"the skill holds no note {', '.join(unknown)}; its notes are {', '.join(notes)}",
            remedy="withhold notes of the skill the case's arms run, by file name",
        )
    texts = {note: _folded(skill.files[NOTES_DIR + note]) for note in notes}
    campaign = ReasonedOn(doc_type="eval_campaign", id=campaign_id)
    led: dict[str, list[ReasonedOn]] = {}
    unmatched: list[ReasonedOn] = []
    for document in (*reasoned_on, campaign):
        needles = [_folded(text) for text in (document.id, document.name) if text is not None]
        needles = [needle for needle in needles if len(needle) >= MIN_MATCH]
        holding = [note for note in notes if any(needle in texts[note] for needle in needles)]
        for note in holding:
            led.setdefault(note, []).append(document)
        if not holding and document is not campaign:
            unmatched.append(document)
    withhold = [*(note for note in notes if note in led), *(note for note in dict.fromkeys(named) if note not in led)]
    if not withhold and not reasoned_on:
        return [], None
    record = CaseMint(
        withheld=[WithheldNote(note=note, reasoned_on=led.get(note, [])) for note in withhold], unmatched=unmatched
    )
    return withhold, record


def read_documents_file(path: str | Path) -> list[dict[str, Any]]:
    """The documents an export file holds: one JSON list of documents, each as stored.

    :param path: The file.
    :ptype path: str | Path
    :return: The documents, in the file's order.
    :rtype: list[dict[str, Any]]
    :raises RefusedInput: The file cannot be read, or is not a JSON list of objects.
    """
    source = Path(path)
    try:
        content = json.loads(_text(source))
    except json.JSONDecodeError as unread:
        raise RefusedInput(f"{source} is not JSON: {unread}", remedy=_EXPORT_REMEDY + ", as one JSON list") from unread
    if not isinstance(content, list) or not all(isinstance(document, dict) for document in content):
        raise RefusedInput(f"{source} is not a JSON list of documents", remedy=_EXPORT_REMEDY + ", as one JSON list")
    return content


def _instant(value: str, what: str) -> str:
    """An ISO-8601 instant with its offset, written as the engine writes one (UTC, ``+00:00``)."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as unread:
        raise RefusedInput(
            f"{what} is not an ISO-8601 instant: {value!r}", remedy="name one as 2026-11-03T00:00:00+00:00"
        ) from unread
    if parsed.tzinfo is None:
        raise RefusedInput(f"{what} names no offset: {value!r}", remedy="name one as 2026-11-03T00:00:00+00:00")
    return parsed.astimezone(UTC).isoformat()


def _missing(doc_type: str, ids: Collection[str], campaign_id: str) -> RefusedInput:
    listed = ", ".join(sorted(ids))
    return RefusedInput(
        f"the export holds no {doc_type} {listed}, which campaign {campaign_id!r} needs",
        remedy=_EXPORT_REMEDY,
    )


def _store(documents: Iterable[Mapping[str, Any]]) -> tuple[InMemoryDocumentStore, list[dict[str, Any]]]:
    """A store in memory holding the documents, and the documents as loaded."""
    store = InMemoryDocumentStore()
    loaded: list[dict[str, Any]] = []
    for at, document in enumerate(documents):
        if not all(isinstance(document.get(key), str) for key in ("doc_type", "id", "scope_id")):
            raise RefusedInput(
                f"document {at} of the export carries no doc_type, id or scope_id",
                remedy="export each document verbatim as stored, with its doc_type, id and scope_id",
            )
        loaded.append(dict(document))
    for document in loaded:
        store.upsert(dict(document))
    return store, loaded


@dataclass(frozen=True)
class _Evidence:
    """An export loaded and checked: its store, the campaign's scope and the instant it is frozen at."""

    storage: EvalStorage
    scope_id: str
    instant: str


def _evidence(
    documents: Iterable[Mapping[str, Any]], campaign_id: str, scope_id: str | None, as_of: str | None
) -> _Evidence:
    """Load the export into a store in memory and check it holds what the freeze reads. Blocking."""
    store, loaded = _store(documents)
    present = {document["doc_type"] for document in loaded}
    for doc_type in REQUIRED_DOC_TYPES:
        if doc_type not in present:
            raise RefusedInput(f"the export holds no {doc_type} document", remedy=_EXPORT_REMEDY)
    scopes = sorted(
        {
            document["scope_id"]
            for document in loaded
            if (document["doc_type"], document["id"]) == ("eval_campaign", campaign_id)
        }
    )
    if not scopes:
        raise _missing("eval_campaign", [campaign_id], campaign_id)
    if scope_id is None:
        if len(scopes) > 1:
            raise RefusedInput(
                f"the export holds campaign {campaign_id!r} in {len(scopes)} scopes: {', '.join(scopes)}",
                remedy="name the scope to freeze it in",
            )
        (scope_id,) = scopes
    elif scope_id not in scopes:
        raise RefusedInput(
            f"the export holds campaign {campaign_id!r} in {', '.join(scopes)}, not in {scope_id!r}",
            remedy="name the scope the campaign document carries, or none",
        )
    storage = EvalStorage(store)
    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise _missing("eval_campaign", [campaign_id], campaign_id)
    runs = storage.load_eval_runs(campaign.run_ids, scope_id)
    if unresolved := set(campaign.run_ids) - {run.id for run in runs}:
        raise _missing("eval_run", unresolved, campaign_id)
    template_ids = {run.template_id for run in runs if run.template_id is not None}
    if untemplated := template_ids - {
        template.id for template in storage.load_templates(sorted(template_ids), scope_id)
    }:
        raise _missing("eval_template", untemplated, campaign_id)
    run_ids = set(campaign.run_ids)
    results = [
        document
        for document in loaded
        if document["doc_type"] == "eval_result" and document.get("eval_run_id") in run_ids
    ]
    if not results:
        raise RefusedInput(f"the export holds no eval_result of campaign {campaign_id!r}'s runs", remedy=_EXPORT_REMEDY)
    if "eval_trace" not in present and any(document.get("has_trace") is True for document in results):
        raise RefusedInput(
            f"campaign {campaign_id!r}'s results record traces and the export holds no eval_trace",
            remedy=_EXPORT_REMEDY,
        )
    instant = (
        _instant(as_of, "as_of")
        if as_of is not None
        else max(
            _instant(stamp, f"the {what} {doc_id!r}'s timestamp")
            for what, doc_id, stamp in (
                ("campaign", campaign.id, campaign.created_at),
                *(("run", run.id, run.completed_at or run.created_at) for run in runs),
            )
        )
    )
    for document in loaded:
        observed = document.get("observed_at")
        if (
            document["doc_type"] == "eval_insight"
            and isinstance(observed, str)
            and _instant(observed, "an insight's observed_at") >= instant
        ):
            store.delete(document["id"], document["scope_id"])
    return _Evidence(storage=storage, scope_id=scope_id, instant=instant)


async def writer_case_from_export(
    documents: Iterable[Mapping[str, Any]],
    profile: HostProfile,
    *,
    campaign_id: str,
    name: str,
    created_by: str,
    scope_id: str | None = None,
    group: str | None = None,
    stratum: str | None = None,
    reasoned_on: Collection[ReasonedOn] = (),
    withhold: Collection[str] = (),
    skill: AnalystSkill | None = None,
    declarations_digest: str | None = None,
    as_of: str | None = None,
) -> WriterCaseFile:
    """Freeze one campaign of a host's export as a portable writer case, writing nothing to the host's store.

    :param documents: The campaign's documents, each as stored with its ``doc_type``, ``id`` and ``scope_id``: at
        least the campaign, every run it names, the template every run ran and the runs' results
        (:data:`REQUIRED_DOC_TYPES`); the traces when a result records one; the test cases and the insights on its
        subject when there are any. Any other document is loaded and never read.
    :ptype documents: Iterable[Mapping[str, Any]]
    :param profile: The host's live profile, the one the campaign's runs were measured under: the freeze calls its
        readers.
    :ptype profile: HostProfile
    :param campaign_id: The campaign to freeze.
    :ptype campaign_id: str
    :param name: The case's name.
    :ptype name: str
    :param created_by: Who mints it, as the host names the caller; the snapshot records it.
    :ptype created_by: str
    :param scope_id: The scope the campaign lives in; ``None`` for the campaign document's own.
    :ptype scope_id: str | None
    :param group: The split group; ``None`` for the case's name.
    :ptype group: str | None
    :param stratum: The kind of case it is (the domain it comes from).
    :ptype stratum: str | None
    :param reasoned_on: The documents that reasoned on the campaign (its analyses, reports and insights): every note
        of ``skill`` holding one's id or name is withheld, and the case records which led to each note and which led
        to none.
    :ptype reasoned_on: Collection[ReasonedOn]
    :param withhold: Notes withheld by file name besides those ``reasoned_on`` leads to.
    :ptype withhold: Collection[str]
    :param skill: The skill whose notes are withheld, the one the case's arms run; ``None`` for the skill this build
        ships.
    :ptype skill: AnalystSkill | None
    :param declarations_digest: The host's declaration of the profile, by its ``host_declarations_digest``; ``None``
        to check nothing.
    :ptype declarations_digest: str | None
    :param as_of: The instant the snapshot is assembled and frozen at (ISO-8601, with its offset), and the insight
        ledger's cutoff; ``None`` for the moment the campaign's evidence was complete.
    :ptype as_of: str | None
    :return: The case, unwritten (:meth:`~threetears.evals.harness.WriterCaseFile.write` writes it).
    :rtype: WriterCaseFile
    :raises RefusedInput: The export lacks a document the freeze needs, the campaign is not in the scope named,
        ``as_of`` is not an instant, the profile's declarations are not the ones declared, or ``withhold`` names a
        note the skill does not hold.
    :raises ValidationFailedError: As :func:`~threetears.evals.ops.analyst.report_snapshot_freeze` refuses.
    """
    live = host_declarations_digest(profile)
    if declarations_digest is not None and declarations_digest != live:
        raise RefusedInput(
            f"host {profile.host_id!r}'s profile declares {live}, and the export's declaration is "
            f"{declarations_digest}: the campaign was measured under other sweepables or another world",
            remedy="mint with the profile of the build that measured the campaign, or export its declaration again",
        )
    notes, record = _withheld(skill or load_skill(), campaign_id, reasoned_on, withhold)
    evidence = await run_blocking(None, _evidence, documents, campaign_id, scope_id, as_of)
    host = OpsHost.reading(
        EvalHost(
            profile=profile,
            storage=evidence.storage,
            failure_describer=withhold_failure_detail,  # a freeze calls no provider
            trace_sink=None,
            blocking_executor=None,
            cell_timeout=default_cell_timeout,
        )
    )
    case = await writer_case_freeze(
        host,
        evidence.scope_id,
        campaign_id=campaign_id,
        name=name,
        created_by=created_by,
        group=group,
        stratum=stratum,
        withhold=notes,
    )
    pinned = case.snapshot.model_copy(update={"bundle_assembled_at": evidence.instant, "created_at": evidence.instant})
    return case.model_copy(update={"snapshot": pinned, "minted": record})


#: The door is on the harness's root; the file readers and the manifest are the command line's.
__all__ = [
    "MintManifest",
    "ProfileFile",
    "host_profile_from",
    "read_documents_file",
    "read_profile_file",
    "writer_case_from_export",
]
