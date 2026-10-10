"""The campaign family: authoring a campaign, its declaration, its membership and its control.

A campaign is the analysis hub — a curated set of runs under one subject×behavior that a generated
analysis attaches to — so the operations on it are analysis's. Each is a function over a
:class:`CampaignStore` and typed parameters, the shape the curation family set
(:mod:`threetears.evals.run.curation`), so a client of the package can author and curate campaigns
without a host's service layer; a host's service keeps its own public names and delegates here, and
every surface reaches these, so they translate failures alike.

Launching a campaign's arms is not here: it composes the run package with this one, and neither
package may import the other, so the launch is the adapter's and calls these.

**Every writer of an existing campaign holds the campaign write lock**
(:func:`~threetears.evals.contracts.campaign_writes.serialized_campaign_write`). Campaign documents
carry no ETag, so each edit is a blind read-modify-write, and the run-delete cascade in the run
package detaches runs under the same lock.

**A campaign lives in one scope, and so do its runs.** ``scope_id`` is the engine's word for a
partition it never interprets; the host chooses what a scope is and passes it through. Every
operation here names the campaign's scope, and a run outside it is refused at attachment — at
creation and at :func:`add_runs_to_campaign` alike — because every read of a campaign's members is
a read within that one scope, and a member it cannot see is one no analysis of the campaign reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field

from threetears.evals.analysis.bundle import variant_key_of_run
from threetears.evals.contracts.authoring_fields import reject_unknown_authoring_fields
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.campaign_writes import serialized_campaign_write
from threetears.evals.contracts.host.profile import HostProfile
from threetears.evals.contracts.errors import NotFoundError, ValidationFailedError
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.storage import EvalStorage
    from threetears.evals.contracts.campaign import CampaignView, EvalCampaign
    from threetears.evals.contracts.declaration import CampaignDesign
    from threetears.evals.contracts.models import EvalResult, EvalRun, EvalRunStamp, EvalTemplate

log = get_logger(__name__)


class CampaignStore(Protocol):
    """Everything the campaign family reads and writes — and nothing else.

    Cut to the calls this module makes rather than to what a storage layer offers: campaigns
    themselves, the template a declaration's bars are checked against, and — for the view's
    window and a control's designation — a member run's stamp, the run, and its results.

    **Not composed from** :class:`~threetears.evals.analysis.bundle.CampaignReadStore`, though one
    member is redeclared from it. That port is the bundle's four reads and writes nothing;
    composing would hand a consumer of the bundle a contract naming campaign writes, and a
    consumer of this family the bundle's insight reads — the union-port problem the curation
    family's port declines for the same reason.

    Structural, so a host's own storage satisfies it by having the methods.
    :class:`~threetears.evals.contracts.storage.EvalStorage` does, with no inheritance and no registration.
    Positional parameters are positional-only, so an implementation's own parameter names never
    have to match the port's.
    """

    def load_campaign(self, campaign_id: str, scope_id: str, /) -> EvalCampaign | None:
        """Load one campaign within a scope, or ``None``."""
        ...

    def save_campaign(self, campaign: EvalCampaign, /) -> None:
        """Write a campaign; raises ``StorageError`` rather than returning a flag."""
        ...

    def list_campaigns(
        self,
        scope_id: str,
        /,
        *,
        subject_id: str | None = None,
        behavior: str | None = None,
        archived: bool | None = None,
    ) -> list[EvalCampaign]:
        """Campaigns in a scope, newest first; each keyword is an optional equality filter."""
        ...

    def load_template(self, template_id: str, scope_id: str, /) -> EvalTemplate | None:
        """Load one template within a scope, or ``None``."""
        ...

    def load_eval_run_stamps(self, run_ids: Sequence[str], scope_id: str, /) -> list[EvalRunStamp]:
        """The id, archived flag and start time of the named runs in one read; absent ids are skipped."""
        ...

    def load_eval_run(self, run_id: str, scope_id: str, /) -> EvalRun | None:
        """Load one run within a scope, or ``None``."""
        ...

    def query_eval_results_by_run(self, run_id: str, scope_id: str, /) -> list[EvalResult]:
        """Every result belonging to one run within a scope."""
        ...


#: Identity/lifecycle fields the family owns — never taken from caller input.
#:
#: ``created_by`` is here rather than in the authoring vocabulary because authorship a
#: caller can type is not authorship: the field spent its whole life advertised as an
#: optional key and written by nobody, on either surface. The surface knows who it is
#: talking to and the caller does not have to be trusted about it, so the surface says.
CAMPAIGN_SERVER_FIELDS: tuple[str, ...] = (
    "id",
    "doc_type",
    "schema_version",
    "created_at",
    "created_by",
    "scope_id",
)

#: What an operator may change on a campaign after it exists.
#:
#: Deliberately short. Membership (``run_ids``) and the control designation have their own
#: write paths carrying rules this one does not enforce, and letting a generic update reach
#: them would be a second door into the same state with none of the guards. ``created_by``
#: is absent because authorship is not editable — a later editor is not the author.
_CAMPAIGN_UPDATABLE_FIELDS: tuple[str, ...] = ("name", "description", "declared_design")


class OpenAxisFamily(EvalBaseModel):
    """An open family of axes: a container whose members are declarable, though not enumerable."""

    name: str = Field(description="The family's name. Not itself an axis — it identifies no knob")
    recognises_members: bool = Field(
        description=(
            "Whether this host can tell a member from a typo. False means no member is accepted as an axis at all, "
            "whatever it is called"
        )
    )
    why_open: str = Field(description="Why the members cannot be listed, in the host's words")


class DeclarableAxes(EvalBaseModel):
    """What a campaign may declare it sweeps on this host — the vocabulary the authoring gate reads.

    Served so an authoring form can offer the names rather than learn them from a refusal. It is
    read off the same registry the gate decides with, and ``remedy`` is the very sentence the gate's
    refusal ends with, so the offer and the refusal cannot disagree.
    """

    host_id: str = Field(description="The host whose registry this is")
    levers: list[str] = Field(
        description="Every fixed lever, in the host's reporting order — each one a declarable axis"
    )
    open_families: list[OpenAxisFamily] = Field(
        description="Families whose members are declarable by name — `<family member>` rather than the family itself"
    )
    remedy: str = Field(description="The vocabulary as one sentence — exactly what a refused declaration is told")


def create_campaign(
    storage: CampaignStore,
    definition: dict[str, Any],
    *,
    scope_id: str,
    created_by: str,
    profile: HostProfile,
    control_from_run_id: str | None = None,
) -> EvalCampaign:
    """Create and persist a campaign from an authoring definition.

    A name the model does not declare is refused; server-owned identity/lifecycle fields in ``definition`` are ignored — a
    fresh ``id`` and ``created_at`` are always assigned, and ``scope_id`` and ``created_by`` come
    from the surface rather than from the caller. ``name`` / ``subject_id`` /
    ``behavior`` are required (non-empty); the rest default per
    :class:`~threetears.evals.contracts.campaign.EvalCampaign`.

    **A campaign may be declared as it is created.** A ``declared_design`` in ``definition`` is
    validated by the campaign contract, stamped with its author, and gated as :func:`update_campaign`
    gates one. Its control is a variant key, which nobody types: ``control_from_run_id`` names one of the
    campaign's runs and the key is resolved from it exactly as :func:`set_campaign_control` resolves
    one. A design that types ``control`` itself is held to the key's shape by the contract.

    **The design is optional.** A campaign created without one is exploratory — someone learning to run
    evals, or testing an intuition with no settled question — and that is derived from
    ``declared_design is None`` wherever it is read, never stored as a second flag: its report and analysis
    bundle say so once, and its analysis reads a design inferred from the runs.

    Args:
        storage: The campaign store.
        definition: Campaign fields. Any ``run_ids`` it carries must live in ``scope_id``.
        scope_id: The scope the campaign lives in — the one its member runs live in.
        created_by: Who is creating it, as the calling surface knows them. Keyword-only and
            required, so a new surface cannot reach this function without deciding what it
            records — which is the whole defect: the field existed for the campaign's entire
            life and neither surface ever wrote it.
        profile: The host whose vocabulary this reads.
        control_from_run_id: One of ``definition``'s runs whose variant becomes the declared control, or
            None. Requires a ``declared_design``, and one that does not type ``control`` itself.

    Returns:
        The persisted :class:`~threetears.evals.contracts.campaign.EvalCampaign`.

    Raises:
        ValidationFailedError: ``definition`` fails campaign validation, names a run that does
            not exist in ``scope_id``, or declares a design this host cannot honour; or
            ``control_from_run_id`` is given with no design, beside a typed ``control``, or names a run
            the campaign does not hold.
        LeverCoordinateError: ``control_from_run_id``'s run has no results and recorded no lever map, and
            the host's variant map disagrees with its own registry.
        StorageError: The campaign failed to persist (a failed write must not
            read back as a created 201 for an id ``campaign_get`` then 404s).
    """
    from pydantic import ValidationError

    from threetears.evals.contracts.campaign import EvalCampaign

    # `created_by` is IGNORED here rather than refused, and that is deliberate: authorship is
    # recorded by the server, never asked of the caller. Refusing it was once proposed on the
    # strength of a surface's help text, "`created_by` is NOT accepted", which read in full means "your value will not be used" rather than
    # "your request will be rejected". Refusing would also single out one member of
    # `CAMPAIGN_SERVER_FIELDS` while `id` and `created_at` beside it stay ignored, so a
    # caller round-tripping a campaign it just read would be refused for one of the three
    # fields it echoed back — the case those fields are ignored FOR.
    reject_unknown_authoring_fields("create_campaign", definition, EvalCampaign, CAMPAIGN_SERVER_FIELDS)
    clean = {k: v for k, v in definition.items() if k not in CAMPAIGN_SERVER_FIELDS}
    clean["created_by"] = created_by
    clean["scope_id"] = scope_id
    try:
        campaign = EvalCampaign(**clean)
    except ValidationError as e:
        raise ValidationFailedError(f"invalid campaign definition: {e}") from e
    _refuse_runs_outside_scope(storage, campaign.run_ids, scope_id)
    if control_from_run_id is not None:
        if campaign.declared_design is None:
            raise ValidationFailedError(
                "control_from_run_id names the run whose variant is the control, and the control lives on the "
                "declaration — declare a design (`declared_design`) with it. A control is the reference point of a "
                "comparison, so there is nothing for it to reference yet"
            )
        if campaign.declared_design.control is not None:
            raise ValidationFailedError(
                "the design types a `control` and control_from_run_id names one too — give one: the run (control_from_run_id), "
                "whose variant key is resolved for you, or the key itself"
            )
        control = _resolve_control_variant(storage, campaign, control_from_run_id, profile=profile)
        campaign = campaign.model_copy(
            update={"declared_design": campaign.declared_design.model_copy(update={"control": control})}
        )
    if campaign.declared_design is not None:
        campaign = campaign.model_copy(
            update={"declared_design": _stamp_declaration(campaign.declared_design, created_by)}
        )

    # A gate, not a page. A declaration naming an axis this host cannot vary, or a bar
    # looser than the registered standard, is refused HERE — at authoring time, before any
    # run is launched and any money is spent. Checked at analysis time instead, the campaign
    # would run to completion and the defect would surface as a report nobody can act on.
    if campaign.declared_design is not None:
        _gate_declaration(storage, campaign, campaign.declared_design, profile=profile)

    storage.save_campaign(campaign)
    return campaign


def _refuse_runs_outside_scope(storage: CampaignStore, run_ids: Sequence[str], scope_id: str) -> None:
    """Refuse membership of any run that does not resolve in the campaign's scope.

    A run in another scope and a run that does not exist are the same thing from here — no read
    within this scope returns either — and both are refused for one reason: every read of a
    campaign's members is a read within its scope, so a member outside it is a run the campaign
    claims and no analysis of the campaign can see. Refused all-or-nothing, naming every id, so a
    partly-wrong request leaves the campaign as the caller last saw it.

    Args:
        storage: The campaign store.
        run_ids: The runs to be held.
        scope_id: The campaign's scope.

    Raises:
        ValidationFailedError: One or more of ``run_ids`` does not resolve in ``scope_id``.
    """
    if not run_ids:
        return
    resolved = {stamp.id for stamp in storage.load_eval_run_stamps(run_ids, scope_id)}
    if outside := sorted({run_id for run_id in run_ids if run_id not in resolved}):
        raise ValidationFailedError(
            f"run(s) {', '.join(repr(r) for r in outside)} do not exist in scope {scope_id!r} — a campaign holds "
            "only runs in its own scope, because every read of its members is a read within that scope"
        )


def _stamp_declaration(design: CampaignDesign, declared_by: str) -> CampaignDesign:
    """Stamp who stated a declaration and when, on every write of one.

    Authorship of the declaration is the surface's to record for the same reason the
    campaign's is: a caller-typed name is not a record of who did it. ``declared_at`` moves
    with it — an amended declaration is a new statement of intent, and dating it to when the
    campaign was first declared would put yesterday's date on today's claim. What must NOT
    move is the per-question history, which is why ``asked_at`` lives on the question and is
    preserved by :func:`~threetears.evals.contracts.declaration.reconcile_question_edits`.

    Args:
        design: The declaration being written.
        declared_by: Who is writing it, as the calling surface knows them.

    Returns:
        The declaration with its authorship stamped.
    """
    from threetears.evals.contracts.models import utc_now_iso

    return design.model_copy(update={"declared_by": declared_by, "declared_at": utc_now_iso()})


def _gate_declaration(
    storage: CampaignStore, campaign: EvalCampaign, design: CampaignDesign, *, profile: HostProfile
) -> None:
    """Refuse a declaration this host cannot honour — a gate, not a page.

    Runs on every authoring write, not only creation: a declaration amended onto an existing
    campaign is the same claim about what this host can vary, and gating only the first one
    would leave the second door unguarded. It runs ONLY on writes — a stored declaration is
    never re-gated on read, so a campaign authored before a rule was added to this gate still
    loads, lists and analyses; the rule binds the next write of its declaration.

    The campaign's template is read here because a bar may name one of its rubric dimensions or
    goal-state checks, and which of those a campaign's results will carry is a fact about the
    template the runs are scored against, not about the host. The template is read straight from
    storage rather than through the service's ``get_template``, whose own refusals are about
    launching runs from it and have nothing to say about what it declares.

    Args:
        storage: The campaign store.
        campaign: The campaign the declaration is written on. Its ``behavior`` is what a registered
            bar is keyed under; its ``template_id`` is read in the campaign's scope, and may be
            blank or dangle — ``EvalCampaign.template_id`` is a reference, not a foreign key.
            Either way no rubric dimension or goal-state check is known, and a bar naming one is
            refused saying so.
        design: The declaration to check.
        profile: The host whose vocabulary this reads.

    Raises:
        ValidationFailedError: An axis the host will not accept — neither a registered
            lever nor a recognised open-family member, pointing at :func:`declarable_axes` — a bar naming neither a described
            measure nor a rubric dimension or goal-state check of the campaign's template, a
            bar is looser than the registered incumbent, a bar contradicts the better-direction
            of what it names.
    """
    from threetears.evals.contracts.declaration import UndeclarableAxisError, refuse_an_undeclarable_design

    template = storage.load_template(campaign.template_id, campaign.scope_id) if campaign.template_id else None
    try:
        refuse_an_undeclarable_design(design, behavior=campaign.behavior, template=template, profile=profile)
    except UndeclarableAxisError as e:
        raise ValidationFailedError(
            f"invalid campaign design: {str(e).rstrip('.')}. `declarable_axes()` lists every axis this host accepts — its levers and "
            "the open families whose members it recognises — read from the registry this gate decides with"
        ) from e
    except ValueError as e:
        raise ValidationFailedError(f"invalid campaign design: {e}") from e


def declarable_axes(profile: HostProfile) -> DeclarableAxes:
    """The axes a campaign may declare on this host, read off the registry the gate decides with.

    Read from the same profile the gate
    (:func:`~threetears.evals.contracts.declaration.refuse_an_undeclarable_design`) is handed: an
    offer drawn from any other registry could promise an axis the gate then refuses.

    Args:
        profile: The host whose registry the campaign is declared against.

    Returns:
        The fixed levers, the open families whose members are declarable, and the remedy
        sentence a refused declaration carries.
    """
    registry = profile.sweepables
    return DeclarableAxes(
        host_id=profile.host_id,
        levers=list(registry.lever_names),
        open_families=[
            OpenAxisFamily(
                name=family.name,
                recognises_members=family.owns_member is not None,
                why_open=family.open_family,
            )
            # The filter drops nothing: `open_families` is the declarations whose `open_family` is set.
            for family in registry.open_families
            if family.open_family is not None
        ],
        remedy=registry.axis_remedy,
    )


@serialized_campaign_write
def update_campaign(
    storage: CampaignStore,
    campaign_id: str,
    scope_id: str,
    updates: dict[str, Any],
    *,
    updated_by: str,
    profile: HostProfile,
) -> EvalCampaign:
    """Amend a campaign's authored fields — the write half the declaration needs.

    A declaration is not a create-time-only fact: an operator learns what they were actually
    asking somewhere in the middle of a campaign, and without this there is no way to say so
    short of starting over. Without it, ``created_by`` would be carried faithfully by
    storage, rendered correctly by a UI, and written by nobody.

    Only ``_CAMPAIGN_UPDATABLE_FIELDS`` may be named, and an unrecognised key is refused
    rather than ignored — a silently dropped key reads as an update that worked. Questions
    inside a supplied ``declared_design`` are reconciled rather than replaced
    (:func:`~threetears.evals.contracts.declaration.reconcile_question_edits`), and the declaration gate
    runs here exactly as it does at creation.

    Args:
        storage: The campaign store.
        campaign_id: The campaign to amend.
        scope_id: The scope it lives in.
        updates: Partial field map; every key must be updatable.
        updated_by: Who is amending it, as the calling surface knows them. Recorded on the
            declaration when this update writes one; it never becomes ``created_by``, since
            a later editor is not the author.
        profile: The host whose vocabulary this reads.

    Returns:
        The persisted, updated :class:`~threetears.evals.contracts.campaign.EvalCampaign`.

    Raises:
        NotFoundError: No campaign with that id in the scope.
        ValidationFailedError: ``updates`` is empty, names a field that is not updatable,
            drops a stored question, fails campaign validation, or declares a design this
            host cannot honour.
        StorageError: The updated campaign failed to persist.
    """
    from pydantic import ValidationError

    from threetears.evals.contracts.campaign import EvalCampaign
    from threetears.evals.contracts.declaration import CampaignDesign, reconcile_question_edits

    if not updates:
        raise ValidationFailedError("update_campaign requires at least one field to change")
    if unknown := sorted(set(updates) - set(_CAMPAIGN_UPDATABLE_FIELDS)):
        raise ValidationFailedError(
            f"campaign fields not updatable here: {', '.join(unknown)}. "
            f"Updatable: {', '.join(_CAMPAIGN_UPDATABLE_FIELDS)}. "
            f"Membership and the control designation have their own write paths "
            f"(add_runs_to_campaign / remove_runs_from_campaign / set_campaign_control)"
        )

    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id)

    merged = campaign.model_dump(mode="json")
    merged.update({k: v for k, v in updates.items() if k != "declared_design"})

    if "declared_design" in updates:
        supplied = updates["declared_design"]
        if supplied is None:
            # Not offered as "clear it": a declaration is what an analysis of this campaign
            # is judged against, and un-declaring one retroactively turns every coverage
            # shortfall it named back into data nobody meant to collect.
            raise ValidationFailedError(
                "a declaration cannot be withdrawn — amend it instead, or retire the questions it asks"
            )
        if not isinstance(supplied, CampaignDesign | dict):
            # `**supplied` on a string, list or number raises TypeError, which the guard
            # below does not catch and `translate_service_errors` does not translate — so a
            # hand-authored declaration of the wrong shape reached the operator as a 500 on
            # PATCH and as an unhandled failure over MCP. `create_campaign` builds the same
            # field through `EvalCampaign(**clean)` and answers 400, and the two authoring
            # doors must not disagree about the same value.
            raise ValidationFailedError(f"invalid campaign design: expected an object, got {type(supplied).__name__}")
        try:
            incoming = supplied if isinstance(supplied, CampaignDesign) else CampaignDesign(**supplied)
        except ValidationError as e:
            raise ValidationFailedError(f"invalid campaign design: {e}") from e
        stored_questions = campaign.declared_design.questions if campaign.declared_design else []
        try:
            incoming = incoming.model_copy(
                update={"questions": reconcile_question_edits(stored_questions, incoming.questions)}
            )
        except ValueError as e:
            raise ValidationFailedError(f"invalid campaign design: {e}") from e
        _gate_declaration(storage, campaign, incoming, profile=profile)
        merged["declared_design"] = _stamp_declaration(incoming, updated_by).model_dump(mode="json")

    try:
        updated = EvalCampaign(**merged)
    except ValidationError as e:
        raise ValidationFailedError(f"invalid campaign update: {e}") from e

    storage.save_campaign(updated)
    log.info("eval.update_campaign campaign=%s fields=%s by=%s", campaign_id, ",".join(sorted(updates)), updated_by)
    return updated


def get_campaign(storage: CampaignStore, campaign_id: str, scope_id: str) -> EvalCampaign:
    """Load a campaign by id.

    Args:
        storage: The campaign store.
        campaign_id: The campaign to load.
        scope_id: The scope it lives in.

    Returns:
        The campaign.

    Raises:
        NotFoundError: No campaign with that id in the scope.
    """
    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id)
    return campaign


def get_campaign_view(storage: CampaignStore, campaign_id: str, scope_id: str) -> CampaignView:
    """Load a campaign and enrich it with its read-time-derived window.

    The window (a campaign's [start, end] time span) is DERIVED from the
    member runs' ``created_at`` rather than stored, so it never drifts as runs
    join or leave. Deriving it means reading the runs — only their id, archived
    flag and start time (``CampaignStore.load_eval_run_stamps``), in one batch,
    within the campaign's scope, which is where its members live.

    **A member run that does not resolve is REPORTED, not skipped.** All three
    fates are disclosed — resolved (feeds the window), archived, unresolvable —
    because the third one is not rare: bulk eval-schema wipes delete runs
    without going through the run-delete cascade, which is what detaches
    memberships, so a campaign outlives its runs. A campaign can claim a
    score of members and resolve none of them. Expect it to recur on
    every rebuild; the disclosure is the durable answer, not a cleanup pass.

    Every surface that reads one campaign reaches this, so they all answer
    with the same facts.

    Args:
        storage: The campaign store.
        campaign_id: Campaign to load.
        scope_id: The scope the campaign and its member runs live in.

    Returns:
        A :class:`~threetears.evals.contracts.campaign.CampaignView` — the campaign
        plus its derived window (or ``None``).

    Raises:
        NotFoundError: No campaign with that id in the scope.
    """
    from threetears.evals.contracts.campaign import CampaignView, derive_window

    campaign = get_campaign(storage, campaign_id, scope_id)
    archived_run_ids: list[str] = []
    unresolved_run_ids: list[str] = []
    created_ats: list[str] = []
    # One batch read of the three scalars the window needs, reduced in the store: a
    # member run is mostly the host's frozen payload, and a campaign has many of them.
    loaded = {stamp.id: stamp for stamp in storage.load_eval_run_stamps(campaign.run_ids, scope_id)}
    for run_id in campaign.run_ids:
        run = loaded.get(run_id)
        if run is None:
            # Recorded rather than dropped: membership is refused outside the scope, so a
            # member that does not resolve is a run destroyed outside the delete cascade —
            # a fact about the campaign the reader needs, and not visible from run_ids.
            unresolved_run_ids.append(run_id)
            continue
        # The window describes the COHORT — the runs a report over this
        # campaign would actually read — so an archived member must not
        # stretch it. Naming those members beside it keeps the narrower
        # window from reading as missing data.
        if run.archived:
            archived_run_ids.append(run_id)
        else:
            created_ats.append(run.created_at)
    window = derive_window(created_ats)
    return CampaignView(
        campaign=campaign,
        window=window,
        archived_run_ids=archived_run_ids,
        unresolved_run_ids=unresolved_run_ids,
    )


def list_campaigns(
    storage: CampaignStore,
    scope_id: str,
    *,
    subject_id: str | None = None,
    behavior: str | None = None,
    archived: bool | None = None,
) -> list[EvalCampaign]:
    """List campaigns in a scope, newest first.

    ``subject_id`` / ``behavior`` / ``archived`` are optional
    equality filters; all ``None`` returns every campaign.

    Args:
        storage: The campaign store.
        scope_id: The scope to list.
        subject_id: Only campaigns over this subject.
        behavior: Only campaigns measuring this behavior.
        archived: Only archived (``True``) or only active (``False``) campaigns.

    Returns:
        The matching campaigns, newest first.
    """
    return storage.list_campaigns(
        scope_id,
        subject_id=subject_id,
        behavior=behavior,
        archived=archived,
    )


@serialized_campaign_write
def add_runs_to_campaign(storage: CampaignStore, campaign_id: str, scope_id: str, run_ids: list[str]) -> EvalCampaign:
    """Attach runs to a campaign (de-duplicated, order-preserving) and persist.

    Loads, merges, and re-saves here — rather than delegating a buried save to
    storage — so a failed persist raises ``StorageError`` (the same write honesty
    as analysis generation) instead of returning the in-memory campaign as if it
    had been stored. Already-attached ids are ignored; a run may legitimately
    belong to more than one campaign, but not twice to the same one. A run that
    does not resolve in the campaign's scope is refused, as at creation.

    Args:
        storage: The campaign store.
        campaign_id: Campaign to attach to.
        scope_id: The scope the campaign and the runs live in.
        run_ids: Run ids to add; already-attached ids are ignored.

    Returns:
        The persisted, updated :class:`~threetears.evals.contracts.campaign.EvalCampaign`.

    Raises:
        NotFoundError: No campaign with that id in the scope.
        ValidationFailedError: A run does not exist in the scope; nothing was attached.
        StorageError: The updated campaign failed to persist.
    """
    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id)
    _refuse_runs_outside_scope(storage, [run_id for run_id in run_ids if run_id not in campaign.run_ids], scope_id)
    seen = set(campaign.run_ids)
    merged = list(campaign.run_ids)
    for run_id in run_ids:
        if run_id not in seen:
            merged.append(run_id)
            seen.add(run_id)
    campaign.run_ids = merged
    storage.save_campaign(campaign)
    return campaign


@serialized_campaign_write
def remove_runs_from_campaign(
    storage: CampaignStore, campaign_id: str, scope_id: str, run_ids: list[str]
) -> EvalCampaign:
    """Detach runs from a campaign — the inverse of :func:`add_runs_to_campaign`.

    Detaching removes campaign MEMBERSHIP only; the run document and its
    results are untouched and stay readable, and the run keeps whatever other
    campaigns it belongs to. This is how a campaign is curated back to the
    cells it was designed with when a run was attached by mistake.

    A requested id that is not attached is **refused**, naming the ids — a
    detach that reports success for an id it never held is the same
    confident-wrong answer as a filter that is accepted and silently ignored.
    The refusal is all-or-nothing, so a partly-wrong request leaves the
    campaign exactly as the caller last saw it.

    Args:
        storage: The campaign store.
        campaign_id: Campaign to detach from.
        scope_id: The scope it lives in.
        run_ids: Run ids to remove; every one must currently be attached.

    Returns:
        The persisted, updated :class:`~threetears.evals.contracts.campaign.EvalCampaign`.

    Raises:
        NotFoundError: No campaign with that id in the scope.
        ValidationFailedError: One or more ids are not attached to this campaign.
        StorageError: The updated campaign failed to persist.
    """
    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id)
    attached = set(campaign.run_ids)
    missing = [run_id for run_id in run_ids if run_id not in attached]
    if missing:
        raise ValidationFailedError(
            f"campaign '{campaign_id}' does not hold run(s) {', '.join(repr(m) for m in missing)} — nothing was detached"
        )
    removing = set(run_ids)
    # No control guard here, and its absence is the point. This path used to refuse when the
    # detached run was the campaign's designated control, because the designation was a RUN ID
    # and detaching it left a pointer at a non-member. The declaration names a VARIANT, so
    # detaching a run cannot invalidate it: where another member carries the same contestant
    # stack the design is untouched, and where none does, the bundle reports the control as
    # unobserved rather than the campaign silently reading as an undesignated bag of runs. That
    # is also the only place the question is answerable — resolving a variant means reading the
    # observations, which a detach has no reason to pay for.
    campaign.run_ids = [run_id for run_id in campaign.run_ids if run_id not in removing]
    storage.save_campaign(campaign)
    log.info("eval.remove_runs_from_campaign campaign=%s detached=%d", campaign_id, len(removing))
    return campaign


@serialized_campaign_write
def set_campaign_control(
    storage: CampaignStore,
    campaign_id: str,
    scope_id: str,
    run_id: str | None,
    *,
    set_by: str,
    profile: HostProfile,
) -> EvalCampaign:
    """Designate (or clear) the campaign's control, ADDRESSED from a member observation.

    The control is a variant key — the resolved contestant stack every other cell is read
    against — and a key is a sha256 nobody types. So the operator still points at something
    they can see, a member run, and this resolves it: the run's observations are keyed
    through the same predicate the analysis uses, and the key is written to
    ``declared_design.control``. That is the same shape a swept level is authored in
    (:meth:`~threetears.evals.contracts.declaration.SweptAxis._content_address_authored_levels`)
    — the surface holding the content addresses it; the surface holding a key sends the key,
    through :func:`update_campaign`.

    **A run with no observation yet can be designated**, which is what lets an operator pick
    the control while launching the arms rather than after the first result lands: the run
    records the lever map its launch resolved, and the key is read from that map exactly as
    the runner reads it to stamp every result the run will write.

    **A run carries one arm, so it names one variant.**

    **The campaign must already declare a design**, because that is where the control lives
    and a declaration requires at least one axis. A control is the reference point of a
    comparison, so a campaign with nothing declared to compare has nothing for one to
    reference.

    **The declaration is re-stamped**, because changing the control changes what the campaign
    says it set out to learn. Leaving ``declared_by`` / ``declared_at`` where the axes' author
    put them would render, directly above the control on the campaign page, the name and date
    of somebody who did not designate it.

    Args:
        storage: The campaign store.
        campaign_id: Campaign to designate on.
        scope_id: The scope the campaign and its runs live in.
        run_id: An attached run whose variant becomes the control, or None to clear.
        set_by: Who is designating, as the calling surface knows them. Keyword-only and
            required for the same reason :func:`create_campaign`'s ``created_by`` is: this
            writes the declaration, and a surface that reaches the write path without deciding
            what it records is how authorship went unwritten for a field's entire life.
        profile: The host whose vocabulary this reads.

    Returns:
        The persisted, updated :class:`~threetears.evals.contracts.campaign.EvalCampaign` — or the
        campaign untouched when CLEARING a control on one that declares no design, since
        there is nothing to clear and a refusal would answer the request with its opposite.

    Raises:
        NotFoundError: No campaign with that id in the scope.
        ValidationFailedError: A DESIGNATION was asked for and the campaign declares no
            design; or ``run_id`` is not attached or does not resolve in the scope.
        LeverCoordinateError: The run has no results and recorded no lever map, and the host's
            variant map disagrees with its own registry.
        StorageError: The updated campaign failed to persist.
    """
    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id)
    if campaign.declared_design is None:
        # Below the clear, deliberately: the narrowing argues that a control has nothing to
        # REFERENCE without a declared comparison, which is an argument about DESIGNATING.
        # Refusing a clear here would answer "remove the control" with "declare a design" —
        # the opposite of what was asked, for a call that is already a no-op.
        if run_id is None:
            return campaign
        raise ValidationFailedError(
            f"campaign '{campaign_id}' declares no design, and the control lives on the declaration — "
            f"declare one first (update_campaign with `declared_design`). A control is the "
            f"reference point of a comparison, so there is nothing for it to reference yet"
        )
    control = None if run_id is None else _resolve_control_variant(storage, campaign, run_id, profile=profile)

    # Logged because the design is what every later analysis is computed against, and a
    # designation is the one design fact an operator supplies by hand.
    previous = campaign.declared_design.control
    amended = _stamp_declaration(campaign.declared_design.model_copy(update={"control": control}), set_by)
    campaign = campaign.model_copy(update={"declared_design": amended})
    storage.save_campaign(campaign)
    log.info(
        "eval.set_campaign_control campaign=%s control=%s was=%s from_run=%s by=%s",
        campaign_id,
        control or "(cleared)",
        previous or "(none)",
        run_id or "(n/a)",
        set_by,
    )
    return campaign


def _resolve_control_variant(
    storage: CampaignStore, campaign: EvalCampaign, run_id: str, *, profile: HostProfile
) -> str:
    """The variant key a member run's observations carry, or a refusal naming what is missing.

    Args:
        storage: The campaign store.
        campaign: The campaign being designated on.
        run_id: The member run to address, read in the campaign's scope.
        profile: The host whose vocabulary this reads.

    Returns:
        The 64-hex variant key.

    Raises:
        ValidationFailedError: The run is not attached, or does not resolve in the campaign's
            scope.
        LeverCoordinateError: The run has no results and recorded no lever map (its host assembled
            it without the launch), and the host's variant map disagrees with its own registry.
    """
    from threetears.evals.contracts.identity import resolve_variant_identity

    if run_id not in campaign.run_ids:
        raise ValidationFailedError(
            f"run '{run_id}' is not attached to campaign '{campaign.id}' — attach it before designating its variant the control"
        )
    scope_id = campaign.scope_id
    run = storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise ValidationFailedError(
            f"run '{run_id}' does not exist in scope '{scope_id}' — nothing to resolve a variant from"
        )
    key = variant_key_of_run(storage.query_eval_results_by_run(run_id, scope_id))
    if key is None:
        # A run launched a moment ago has no observation yet, and a designation made AT launch
        # is the common case, not an edge: the operator picks the control while choosing the
        # arms. Every result such a run writes is stamped by digesting the lever map
        # `resolve_variant_identity` reads — the runner's own read — so the key its observations
        # WILL carry is known now, from the same read, not a second derivation.
        key = resolve_variant_identity(run=run, profile=profile).variant_key
    return key


if TYPE_CHECKING:

    def _eval_storage_satisfies_the_port(storage: EvalStorage) -> None:
        """Hold the engine's own store to this consumer's port, so a drifted signature fails typecheck."""
        store: CampaignStore = storage
        del store


__all__ = [
    "CAMPAIGN_SERVER_FIELDS",
    "CampaignStore",
    "DeclarableAxes",
    "OpenAxisFamily",
    "add_runs_to_campaign",
    "create_campaign",
    "declarable_axes",
    "get_campaign",
    "get_campaign_view",
    "list_campaigns",
    "remove_runs_from_campaign",
    "set_campaign_control",
    "update_campaign",
]
