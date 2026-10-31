"""The report writer's case bank: a host mints writer cases from its own campaigns, into a dev and a held-out set.

A writer case (:class:`~threetears.evals.ops.report_writer.writer_kind.WriterCase`) is never generated: each is one
report snapshot frozen into a test case of a ``report_writer`` template, with the exemplar notes its analyst is not
given, as a reporter case is one campaign's bundle frozen into a reporter template's. Cases come from hosts, each
from its own campaigns, so minting is the mechanism, and a package fixture is only the first case a bank holds.

- :func:`writer_case_mint` freezes one: the snapshot through
  :func:`~threetears.evals.ops.analyst.report_snapshot_freeze` (or one the host already froze), the case stored under
  the template, and the case put in its side's named :class:`~threetears.evals.schema.models.CaseSet`, as the next
  version of that set (:func:`~threetears.evals.run.case_sets.mint_case_set`).
- :func:`writer_case_bank` reads a template's stored writer cases once, for every decision the mint makes.
- :func:`writer_split` resolves both sets of a split, refusing a case listed on both sides.
- :func:`writer_case_archive` retires a case, or restores one: how a source whose evidence moved is minted again.

**One source is one case.** A case is keyed by what it froze (:attr:`WriterCase.source_key`): the campaign, or the
report input of a snapshot with no campaign behind it. A mint matching the key's live case (the same snapshot, the
same notes withheld) returns it, so minting again is idempotent and the case keeps its id and its set. A mint that
differs (the campaign's evidence moved since, or other notes withheld) is refused: a stored case is immutable, and a
run measured against it would otherwise be read against evidence it never had. Deduplicating rather than refusing
the exact repeat lets a host mint every campaign it holds on a schedule without first asking which it has minted.
To take the moved evidence instead, the host retires the source's case (:func:`writer_case_archive`) and mints again:
the new case joins the set's next version, which no longer lists the retired one, and every run already measured
against the retired case still resolves it.

**The split is by group, by digest, and recorded.** A case's group is the host's name for a family of related
campaigns (a campaign and its re-runs), or the case's own source key when the host names none, so related cases
never sit on both sides. A group's side is the one recorded on any case of it already stored, retired ones
included; a new group's side is read off the digest of its name against the held-out fraction, so the same group is
always placed on the same side, whichever host mints it and in whatever order. A case's side, its set and the rule
that placed it are recorded on the case (:class:`~threetears.evals.ops.report_writer.writer_kind.WriterCaseOrigin`),
and the sets list it; a case never moves between them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Final

from pydantic import Field, ValidationError, model_validator

from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.kernel.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.remedies import FIND_TEMPLATE, FIND_TEST_CASE
from threetears.evals.kernel.report_snapshot import FIND_SNAPSHOT, EvalReportSnapshot
from threetears.evals.ops.analyst import report_snapshot_freeze, snapshot_input
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.report_writer.writer_contract import (
    WRITER_KIND,
    refuse_undeliverable_writer_template,
    writer_scoring_rubric,
)
from threetears.evals.ops.report_writer.writer_kind import (
    WriterCase,
    WriterCaseOrigin,
    WriterCaseSource,
    WriterSide,
    writer_case_of,
    writer_case_payload,
)
from threetears.evals.ops.runs import LaunchArguments, TemplateLine
from threetears.evals.run.authoring import get_template
from threetears.evals.run.case_sets import mint_case_set, resolve_case_set
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.hashing import bytes_digest
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import CaseSet, CaseSetRef, EvalTemplate, EvalTestCase

log = get_logger(__name__)

#: The share of groups a split puts on the held-out side unless the host names another.
HELD_OUT_FRACTION: Final = 0.3

#: What a group's name is prefixed with before its digest is read, so a split's placement is its own and never
#: shares bits with another digest of the same text.
_SPLIT_DOMAIN = "writer-split:"

#: The hex digits of a group's digest read as its point in the unit interval.
_POINT_DIGITS = 8


class WriterSplit(EvalBaseModel):
    """A split's two named sets, and the share of groups placed on the held-out side.

    The names are the host's: two banks of one scope are two splits under other names.
    """

    dev: str = Field(default="writer-dev", min_length=1, description="The set the cases a change is tuned on join.")
    held_out: str = Field(
        default="writer-held-out", min_length=1, description="The set the cases a change is decided on join."
    )
    held_out_fraction: float = Field(
        default=HELD_OUT_FRACTION,
        gt=0,
        lt=1,
        description="The share of new groups placed on the held-out side. A group already placed keeps its side.",
    )

    @model_validator(mode="after")
    def _two_sets(self) -> WriterSplit:
        """Refuse one name for both sides, which would put every case in both."""
        if self.dev == self.held_out:
            raise ValueError(f"a split's two sets need two names, and both are {self.dev!r}")
        return self

    def set_of(self, side: WriterSide) -> str:
        """The set a side's cases join.

        :param side: The side.
        :ptype side: WriterSide
        :return: Its set's name.
        :rtype: str
        """
        return self.held_out if side == "held_out" else self.dev


class WriterCaseMint(EvalBaseModel):
    """What minting a writer case names: the writer template, exactly one source, the notes withheld and the split."""

    template_id: Annotated[str, LaunchArguments.model_fields["template_id"]]
    campaign_id: str | None = Field(
        default=None, min_length=1, description="A campaign whose evidence is frozen now, as campaigns_list names it."
    )
    analysis_id: str | None = Field(
        default=None,
        min_length=1,
        description="A stored analysis whose frozen bundle is frozen, as analyses_list names it.",
    )
    snapshot_id: str | None = Field(
        default=None,
        min_length=1,
        description="A snapshot already frozen in the scope, as report_snapshot_freeze named it.",
    )
    withhold: list[str] = Field(
        default_factory=list,
        description="The exemplar notes, by file name, the case's analyst is never given: the note reasoned on this "
        "campaign, for a blind run. Each must be a note of the skill this build ships; empty withholds none.",
    )
    group: str | None = Field(
        default=None,
        min_length=1,
        description="The split group: related campaigns (one and its re-runs) named alike sit on one side. Omitted, "
        "the case is its own group.",
    )
    split: WriterSplit = Field(default_factory=WriterSplit, description="The sets the case joins, by side.")

    @model_validator(mode="after")
    def _one_source(self) -> WriterCaseMint:
        """Refuse a mint naming no source, or more than one."""
        named = [name for name in ("campaign_id", "analysis_id", "snapshot_id") if getattr(self, name) is not None]
        if len(named) != 1:
            raise ValueError(
                "a writer case is minted from exactly one of campaign_id, analysis_id and snapshot_id; "
                + ("none was named" if not named else f"{' and '.join(named)} were named")
            )
        return self


class MintedWriterCase(EvalBaseModel):
    """What a mint stored, or found already stored: the receipt every surface renders."""

    test_case_id: str = Field(description="The stored case: the existing one when the mint matched it.")
    template_id: str = Field(description="The writer template it is stored under.")
    snapshot_id: str = Field(description="The snapshot it is written over.")
    source_key: str = Field(description="What makes two mints one case: campaign:<id>, or report-input:<digest>.")
    withhold: list[str] = Field(description="The notes its analyst is never given.")
    group: str = Field(description="Its split group.")
    side: WriterSide = Field(description="The side of the split it sits on.")
    case_set: CaseSetRef = Field(description="The version of its side's set that lists it.")
    reused: bool = Field(description="True when the same case was minted before, and this is that case.")


@dataclass(frozen=True)
class WriterCaseBank:
    """One writer template's stored cases, read once for every decision a mint makes.

    :ivar cases: Every readable writer case, ``(stored case, writer case)`` in storage order, retired ones included.
    :ivar unreadable: ``(case id, why)`` for every stored case carrying a writer case this build cannot read.
    """

    cases: tuple[tuple[EvalTestCase, WriterCase], ...]
    unreadable: tuple[tuple[str, str], ...]

    def live_of(self, source_key: str) -> list[tuple[EvalTestCase, WriterCase]]:
        """The live cases of one source: those no operator retired.

        :param source_key: The source (:attr:`WriterCase.source_key`).
        :ptype source_key: str
        :return: Its live cases, in storage order.
        :rtype: list[tuple[EvalTestCase, WriterCase]]
        """
        return [(stored, case) for stored, case in self.cases if case.source_key == source_key and not stored.archived]

    def side_of(self, group: str) -> WriterSide | None:
        """The side a group was placed on: the side recorded on any of its stored cases, retired ones included.

        :param group: The split group.
        :ptype group: str
        :return: Its side, or ``None`` when no stored case of it was placed.
        :rtype: WriterSide | None
        :raises ValidationFailedError: Its stored cases record both sides.
        """
        sides = {case.origin.side for _, case in self.cases if case.origin is not None and case.origin.group == group}
        if len(sides) > 1:
            raise ValidationFailedError(
                f"split group {group!r} has cases recorded on both sides, so it has no one side to place a case on",
                remedy="mint the case under another group",
            )
        return next(iter(sides), None)


def writer_case_bank(stored: Iterable[EvalTestCase]) -> WriterCaseBank:
    """Read a writer template's stored cases: the ones this build reads, and the ones it does not.

    :param stored: Every stored case of one template in one scope, in storage order. Cases carrying no writer case are
        skipped.
    :ptype stored: Iterable[EvalTestCase]
    :return: The bank.
    :rtype: WriterCaseBank
    """
    cases: list[tuple[EvalTestCase, WriterCase]] = []
    unreadable: list[tuple[str, str]] = []
    for test_case in stored:
        try:
            case = writer_case_of(test_case)
        except ValidationError as unread:
            unreadable.append((test_case.id, f"{unread.error_count()} fault(s): {unread}"))
            continue
        if case is not None:
            cases.append((test_case, case))
    return WriterCaseBank(cases=tuple(cases), unreadable=tuple(unreadable))


def writer_group_side(group: str, held_out_fraction: float) -> WriterSide:
    """The side a new group is placed on: its name's digest, read as a point in the unit interval, against the share.

    :param group: The split group.
    :ptype group: str
    :param held_out_fraction: The share of groups placed on the held-out side.
    :ptype held_out_fraction: float
    :return: ``held_out`` when the group's point falls below the share, else ``dev``.
    :rtype: WriterSide
    """
    point = int(bytes_digest(f"{_SPLIT_DOMAIN}{group}".encode())[:_POINT_DIGITS], 16) / 16**_POINT_DIGITS
    return "held_out" if point < held_out_fraction else "dev"


def _writer_template(host: OpsHost, template_id: str, scope_id: str) -> EvalTemplate:
    """The template, refused unless it is of the writer kind: the only kind whose cases a mint writes."""
    template = get_template(host.eval_host, template_id, scope_id)
    if template.candidate_kind != WRITER_KIND:
        raise ValidationFailedError(
            f"template {template_id!r} is of kind {template.candidate_kind!r}, not {WRITER_KIND!r}, so it runs no "
            "writer case",
            remedy=f"name a template of the {WRITER_KIND!r} kind ({FIND_TEMPLATE})",
        )
    return template


def _refuse_unknown_notes(withhold: list[str]) -> str:
    """Refuse a note the shipped skill lacks, before anything is frozen; the shipped skill's digest otherwise."""
    shipped = load_skill()
    try:
        shipped.analyst(withhold=withhold)
    except ValueError as unknown:
        raise ValidationFailedError(
            f"the case would withhold a note the skill this build ships lacks: {unknown}",
            remedy="withhold only notes the shipped skill holds, by file name, or none",
        ) from unknown
    return shipped.digest


async def _snapshot(host: OpsHost, mint: WriterCaseMint, scope_id: str, created_by: str) -> EvalReportSnapshot:
    """The snapshot the case is written over: frozen now from the campaign or the analysis, or the one named, each
    checked to read as every analyst tool reads it."""
    if mint.snapshot_id is None:
        frozen = await report_snapshot_freeze(
            host, scope_id, campaign_id=mint.campaign_id, analysis_id=mint.analysis_id, created_by=created_by
        )
        snapshot_id = frozen.snapshot_id
    else:
        snapshot_id = mint.snapshot_id
    await snapshot_input(host, scope_id, snapshot_id)
    eval_host = host.eval_host
    snapshot = await run_blocking(
        eval_host.blocking_executor, eval_host.storage.load_report_snapshot, snapshot_id, scope_id
    )
    if snapshot is None:
        raise NotFoundError("report snapshot", snapshot_id, remedy=FIND_SNAPSHOT)
    return snapshot


def _source(mint: WriterCaseMint) -> WriterCaseSource:
    """What the mint named."""
    if mint.campaign_id is not None:
        return "campaign"
    return "analysis" if mint.analysis_id is not None else "snapshot"


def _listed_by(storage_sets: list[CaseSet]) -> set[str]:
    """Every case any version of a set lists."""
    return {case_id for version in storage_sets for case_id in version.test_case_ids}


def _mint_blocking(
    host: OpsHost,
    mint: WriterCaseMint,
    template: EvalTemplate,
    snapshot: EvalReportSnapshot,
    scope_id: str,
    *,
    created_by: str,
    skill: str,
) -> MintedWriterCase:
    """Find or store the case, then list it in its side's set. Blocking."""
    storage = host.eval_host.storage
    bank = writer_case_bank(storage.query_test_cases(scope_id, template_id=template.id))
    if bank.unreadable:
        raise ValidationFailedError(
            f"template {template.id!r} holds writer case(s) this build cannot read: "
            + "; ".join(f"{case_id!r} ({why})" for case_id, why in bank.unreadable)
            + ". One may be this source's case, so whether this mint repeats one cannot be decided",
            remedy="read them with the build that wrote them",
        )
    unplaced = WriterCase.of(snapshot, withhold=mint.withhold)
    key = unplaced.source_key
    live = bank.live_of(key)
    if len(live) > 1:
        raise ValidationFailedError(
            f"source {key} has {len(live)} live writer cases ({', '.join(stored.id for stored, _ in live)}); a source "
            "is one case",
            remedy="retire all but one of them",
        )
    reused = bool(live)
    if live:
        stored, case = live[0]
        if case.snapshot.get("id") != snapshot.id or sorted(case.withhold) != sorted(mint.withhold):
            differs = (
                f"over snapshot {snapshot.id} where it froze {case.snapshot.get('id')} (the evidence moved since)"
                if case.snapshot.get("id") != snapshot.id
                else f"withholding {sorted(mint.withhold)} where it withholds {sorted(case.withhold)}"
            )
            raise ValidationFailedError(
                f"case {stored.id!r} already freezes {key}, and this mint differs: {differs}. A source is one case, "
                "and a stored case never changes, so a run measured against it is read against what it froze",
                remedy=f"launch case {stored.id!r} as it was frozen; to mint this source again, retire it first",
            )
        if case.origin is None:
            raise ValidationFailedError(
                f"case {stored.id!r} freezes {key} and records no split side: no mint wrote it",
                remedy="retire it, then mint the source",
            )
        origin = case.origin
    else:
        group = mint.group or key
        side = bank.side_of(group) or writer_group_side(group, mint.split.held_out_fraction)
        origin = WriterCaseOrigin(
            host_id=host.eval_host.profile.host_id,
            minted_by=created_by,
            source=_source(mint),
            campaign_id=mint.campaign_id,
            analysis_id=mint.analysis_id,
            group=group,
            side=side,
            case_set=mint.split.set_of(side),
            held_out_fraction=mint.split.held_out_fraction,
            skill=skill,
        )
        stored = EvalTestCase(
            template_id=template.id,
            scope_id=scope_id,
            host_payload=writer_case_payload(WriterCase.of(snapshot, withhold=mint.withhold, origin=origin)),
        )
    # The case lands in its own side's set, and never in a set of the other side of the split it is minted under.
    other = mint.split.set_of("dev" if origin.side == "held_out" else "held_out")
    if stored.id in _listed_by(storage.query_case_sets(scope_id, name=other)):
        raise ValidationFailedError(
            f"case {stored.id!r} sits on the {origin.side} side, in {origin.case_set!r}, and set {other!r} lists it "
            "too: a case decided on is never one a change was tuned on",
            remedy=f"mint a version of {other!r} without it (case_set_mint)",
        )
    if not reused:
        storage.save_test_case(stored)
        log.info(
            "eval.writer_case_mint template=%s case=%s source=%s group=%s side=%s",
            template.id,
            stored.id,
            key,
            origin.group,
            origin.side,
        )
    name = mint.split.set_of(origin.side)
    versions = storage.query_case_sets(scope_id, name=name)
    latest = versions[0] if versions else None
    if latest is not None and stored.id in latest.test_case_ids:
        listed = latest
    else:
        # A retired case leaves the set at its next version: no launch over the set runs it again.
        retired = {case.id for case, _ in bank.cases if case.archived}
        kept = [case_id for case_id in (latest.test_case_ids if latest is not None else ()) if case_id not in retired]
        listed = mint_case_set(
            storage, scope_id=scope_id, name=name, template_id=template.id, test_case_ids=[*kept, stored.id]
        )
    return MintedWriterCase(
        test_case_id=stored.id,
        template_id=template.id,
        snapshot_id=snapshot.id,
        source_key=key,
        withhold=list(mint.withhold),
        group=origin.group,
        side=origin.side,
        case_set=listed.ref,
        reused=reused,
    )


async def writer_case_mint(host: OpsHost, mint: WriterCaseMint, scope_id: str, *, created_by: str) -> MintedWriterCase:
    """Mint a writer case from a campaign, a stored analysis or a frozen snapshot, and list it in its side's set.

    In order, each refusal before anything after it: the template must be of the writer kind; each withheld note must
    be a note of the skill this build ships, checked before anything is frozen; the snapshot is frozen
    (:func:`~threetears.evals.ops.analyst.report_snapshot_freeze`) or read, and must read as every analyst tool
    reads it; the bank must read whole. Then the source's live case is returned when this mint repeats it, or the mint
    is refused when it differs; else the case is stored, placed on its group's side. Last, the case is listed in its
    side's set as the next version, unless the latest version lists it already, and refused when the other side's
    set lists it in any version.

    :param host: The host whose store holds the source and receives the case.
    :ptype host: OpsHost
    :param mint: What to mint.
    :ptype mint: WriterCaseMint
    :param scope_id: The scope the template, the source and the sets live in.
    :ptype scope_id: str
    :param created_by: Who mints it, as the host names the caller.
    :ptype created_by: str
    :return: The receipt: the case, its side and the set version listing it.
    :rtype: MintedWriterCase
    :raises NotFoundError: The template, campaign, analysis or snapshot is not in the scope.
    :raises ValidationFailedError: The template is not of the writer kind; a withheld note is not the shipped skill's;
        the snapshot does not read; a stored case of the template does not read; the source's live case differs from
        this mint, or it has several; the other side's set lists the case; or a set refused the version.
    :raises ConflictError: Another mint stored the set's next version first.
    """
    template = await run_blocking(host.eval_host.blocking_executor, _writer_template, host, mint.template_id, scope_id)
    skill = _refuse_unknown_notes(mint.withhold)
    snapshot = await _snapshot(host, mint, scope_id, created_by)
    return await run_blocking(
        host.eval_host.blocking_executor,
        lambda: _mint_blocking(host, mint, template, snapshot, scope_id, created_by=created_by, skill=skill),
    )


def _archive_blocking(
    host: OpsHost, test_case_id: str, scope_id: str, *, archived: bool, reason: str | None
) -> EvalTestCase:
    """Set a writer case's retirement, refusing what no bank honours or a restore that makes two live cases. Blocking."""
    storage = host.eval_host.storage
    current = storage.load_test_case(test_case_id, scope_id)
    if current is None:
        raise NotFoundError("test case", test_case_id, remedy=FIND_TEST_CASE)
    try:
        case = writer_case_of(current)
    except ValidationError as unread:
        raise ValidationFailedError(
            f"test case {test_case_id!r} carries a writer case this build cannot read ({unread.error_count()} "
            "fault(s)), so whether it is live cannot be decided here",
            remedy="read it with the build that wrote it",
        ) from unread
    if case is None or current.template_id is None:
        raise ValidationFailedError(
            f"test case {test_case_id!r} carries no writer case under a template, and only a writer case is retired "
            "here: a writer template's bank is what reads a case's retirement",
            remedy="name a writer case of a writer template",
        )
    target_reason = reason if archived else None
    if current.archived == archived and current.archived_reason == target_reason:
        return current
    if not archived:
        bank = writer_case_bank(storage.query_test_cases(scope_id, template_id=current.template_id))
        if rivals := [stored.id for stored, _ in bank.live_of(case.source_key) if stored.id != test_case_id]:
            raise ValidationFailedError(
                f"restoring case {test_case_id!r} would make a second live case of {case.source_key}, beside "
                f"{', '.join(repr(rival) for rival in rivals)}: a source is one case",
                remedy="retire that case first, or leave this one retired",
            )
    storage.save_test_case(
        EvalTestCase.from_dict({**current.to_dict(), "archived": archived, "archived_reason": target_reason})
    )
    log.info(
        "eval.writer_case_archive case=%s template=%s source=%s archived=%s reason=%s",
        test_case_id,
        current.template_id,
        case.source_key,
        archived,
        target_reason or "(none given)",
    )
    persisted = storage.load_test_case(test_case_id, scope_id)
    if persisted is None:
        raise NotFoundError("test case", test_case_id, remedy=FIND_TEST_CASE)
    return persisted


async def writer_case_archive(
    host: OpsHost, test_case_id: str, scope_id: str, *, archived: bool, reason: str | None = None
) -> EvalTestCase:
    """Retire a writer case, or restore one: a retired case is never launched again, and stays readable.

    Retiring is how a source whose evidence moved is minted again: :func:`writer_case_mint` refuses a mint differing
    from the source's live case, and a retired case is not live, so the next mint stores the new case and lists it in
    its side's next set version, without the retired one. Every run already measured against the retired case still
    resolves it. A restore is refused when the source has another live case, since a source is one case. Idempotent:
    the state and reason the case already carries write nothing. The reason is cleared on restore.

    :param host: The host whose store holds the case.
    :ptype host: OpsHost
    :param test_case_id: The case.
    :ptype test_case_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param archived: ``True`` retires it, ``False`` restores it.
    :ptype archived: bool
    :param reason: Why it is retired, in the operator's words; cleared on restore.
    :ptype reason: str | None
    :return: The case as persisted.
    :rtype: EvalTestCase
    :raises NotFoundError: No such case in the scope.
    :raises ValidationFailedError: It carries no writer case, or one this build cannot read, or names no template; or
        restoring it would make a second live case of its source.
    :raises StorageError: The write failed.
    """
    return await run_blocking(
        host.eval_host.blocking_executor,
        lambda: _archive_blocking(host, test_case_id, scope_id, archived=archived, reason=reason),
    )


@dataclass(frozen=True)
class WriterSplitSets:
    """Both sets of a split, each with its cases in order.

    :ivar dev: The dev set's version and its cases.
    :ivar held_out: The held-out set's version and its cases.
    """

    dev: tuple[CaseSet, list[EvalTestCase]]
    held_out: tuple[CaseSet, list[EvalTestCase]]


def writer_split(
    host: OpsHost,
    template: EvalTemplate,
    scope_id: str,
    *,
    split: WriterSplit,
    dev: int | None = None,
    held_out: int | None = None,
) -> WriterSplitSets:
    """Resolve both sets of a split, refusing a case on both sides. Blocking: it reads the store.

    :param host: The host whose store holds the sets.
    :ptype host: OpsHost
    :param template: The writer template the sets list cases of.
    :ptype template: EvalTemplate
    :param scope_id: The scope they live in.
    :ptype scope_id: str
    :param split: The split's names.
    :ptype split: WriterSplit
    :param dev: The dev set's version; ``None`` for its latest.
    :ptype dev: int | None
    :param held_out: The held-out set's version; ``None`` for its latest.
    :ptype held_out: int | None
    :return: Both sets with their cases.
    :rtype: WriterSplitSets
    :raises NotFoundError: A set, or the version named, is not stored.
    :raises ValidationFailedError: A set lists another template's cases or a case that no longer resolves, or a case
        is listed by both.
    """
    storage = host.eval_host.storage

    def resolved(name: str, version: int | None) -> tuple[CaseSet, list[EvalTestCase]]:
        if version is None:
            versions = storage.query_case_sets(scope_id, name=name)
            if not versions:
                raise NotFoundError("case set", name, remedy="mint a writer case into it (writer_case_mint)")
            version = versions[0].version
        return resolve_case_set(storage, CaseSetRef(name=name, version=version), template=template, scope_id=scope_id)

    dev_set, held_out_set = resolved(split.dev, dev), resolved(split.held_out, held_out)
    if both := sorted(set(dev_set[0].test_case_ids) & set(held_out_set[0].test_case_ids)):
        raise ValidationFailedError(
            f"case(s) {', '.join(both)} are listed by {dev_set[0].ref.label!r} and {held_out_set[0].ref.label!r}: a "
            "case decided on is never one a change was tuned on",
            remedy="mint a version of the set each does not belong to without it (case_set_mint)",
        )
    return WriterSplitSets(dev=dev_set, held_out=held_out_set)


class WriterTemplateCreate(EvalBaseModel):
    """What authoring a report writer template names: what it is called and what its campaigns set out to measure.

    Nothing else: its rubric is the scoring judge's, seeded from the shipped skill, and its cases are minted into it.
    """

    template_name: str = Field(min_length=1, description="The writer template's name, unique in the scope.")
    intent: str = Field(
        min_length=1, description="What its campaigns measure, in a sentence a reader of their reports is told."
    )


def writer_template_create(host: OpsHost, create: WriterTemplateCreate, scope_id: str) -> TemplateLine:
    """Write a template of the report writer kind, its scoring rubric seeded from the skill this build ships.

    The rubric is the editor rubric's items as the shipped ``editor.md`` states them
    (:func:`~threetears.evals.ops.report_writer.writer_contract.writer_scoring_rubric`), copied into the template now,
    so the scoring judge of every campaign launched on it asks what it asked when the template was written, whatever
    skill an arm runs or a later build ships. Refused on a host whose launch registry has no writer kind, where it
    could never run.

    :param host: The launching host, whose registry must launch the writer kind and whose store receives the template.
    :ptype host: OpsHost
    :param create: What the template is called and measures.
    :ptype create: WriterTemplateCreate
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The template as written.
    :rtype: TemplateLine
    :raises ValidationFailedError: The host launches no writer kind.
    :raises ConflictError: A template of that name is already in the scope.
    :raises StorageError: The template could not be written.
    """
    if WRITER_KIND not in host.launch.kinds:
        raise ValidationFailedError(
            f"host {host.eval_host.profile.host_id!r} launches no {WRITER_KIND!r} kind, so a writer template could "
            f"never run here: its LaunchHost registers no {WRITER_KIND!r} kind (launchable_report_writer_kind)",
            remedy="ask this host's operator to enable report writer campaigns, a setting of the host no action changes",
        )
    storage = host.eval_host.storage
    if storage.load_template_by_name(create.template_name, scope_id) is not None:
        raise ConflictError(
            f"a template named '{create.template_name}' already exists",
            remedy=f"name the writer template something else, or mint into the existing one ({FIND_TEMPLATE})",
        )
    template = EvalTemplate(
        scope_id=scope_id,
        name=create.template_name,
        intent=create.intent,
        candidate_kind=WRITER_KIND,
        rubric=writer_scoring_rubric(),
    )
    refuse_undeliverable_writer_template(template)
    storage.save_template(template)
    return TemplateLine(
        id=template.id,
        name=template.name,
        revision=template.revision,
        candidate_kind=template.candidate_kind,
        intent=template.intent,
        archived=template.archived,
    )


#: The writer's launch reads a bank (``writer_case_bank``); a host's mint, retirement and template creation (Discodon,
#: then metallm) are the callers of the rest, none in this repo yet, so each waits for one
#: (``tests/test_report_writer_exports.py``).
__all__ = ["writer_case_bank"]
