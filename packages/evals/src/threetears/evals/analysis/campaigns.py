"""The campaign family: authoring a campaign, its aspects, its membership, its arms and its control.

A campaign is the analysis hub — one question asked of its arms and measured on one or more aspects, each a template
and the runs gathered on it — so the operations on it are analysis's. Each is a function over a
:class:`CampaignStore` and typed parameters, the shape the curation family set
(:mod:`threetears.evals.run.curation`), so a client of the package can author and curate campaigns
without a host's service layer; a host's service keeps its own public names and delegates here, and
every surface reaches these, so they translate failures alike.

Launching a campaign's arms is not here: it composes the run package with this one, and neither
package may import the other, so the launch is the adapter's and calls these.

**Every writer of an existing campaign holds the campaign write lock**
(:func:`~threetears.evals.kernel.campaign_writes.serialized_campaign_write`). Campaign documents
carry no ETag, so each edit is a blind read-modify-write, and the run-delete cascade in the run
package detaches runs under the same lock.

**A campaign lives in one scope, and so do its runs.** ``scope_id`` is the engine's word for a
partition it never interprets; the host chooses what a scope is and passes it through. Every
operation here names the campaign's scope, and a run outside it is refused at attachment — at
creation and at :func:`add_runs_to_campaign` alike — because every read of a campaign's members is
a read within that one scope, and a member it cannot see is one no analysis of the campaign reads.

**A run belongs to one aspect.** Adding a run names its aspect, or leaves it to the run's template: the one aspect
gathered on that template takes it, and a run no aspect or several aspects could take is refused, naming them.

**One freeze rule.** Once any aspect holds a run, what the campaign declared — its question, its aspects' designs and
margins, its floors, and an arm already named — is what those runs were gathered to answer, and is refused in any
form; a new campaign carries the amended declaration (``aspects_from`` adopts the aspects it keeps).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from pydantic import Field, ValidationError

from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.bundle.observations import variant_key_of_run
from threetears.evals.analysis.revisions import refuse_pooled_revisions
from threetears.evals.kernel.authoring_fields import edited_core_document, reject_unknown_authoring_fields
from threetears.evals.kernel.campaign import EvalCampaign, VariantIndexEntry
from threetears.evals.schema.campaign_upgrade import QUESTION_SEPARATOR, UPGRADED_CONTROL_LABEL
from threetears.evals.kernel.declaration import (
    Aspect,
    AspectDesign,
    ArmRole,
    CampaignArm,
    CampaignDesign,
    DecisionClause,
    DecisionRule,
    Floor,
    Margin,
    Question,
    aware_instant,
    resolve_control_arm,
)
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.campaign_writes import serialized_campaign_write
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.kernel.remedies import FIND_CAMPAIGN, FIND_RUN, FIND_TEMPLATE, SCHEMA_REMEDY

if TYPE_CHECKING:
    from threetears.evals.kernel.storage import EvalStorage
    from threetears.evals.kernel.campaign import CampaignView
    from threetears.evals.schema.models import EvalResult, EvalRun, EvalRunStamp, EvalTemplate
    from threetears.evals.schema.values import SweepableValue

log = get_logger(__name__)


class CampaignStore(Protocol):
    """Everything the campaign family reads and writes — and nothing else.

    Cut to the calls this module makes rather than to what a storage layer offers: campaigns
    themselves, the template a declaration's bars are checked against, and — for the view's
    window and a control's designation — a member run's stamp, the run, and its results.

    **Not composed from** :class:`~threetears.evals.analysis.bundle.assemble.CampaignReadStore`, though one
    member is redeclared from it. That port is the bundle's four reads and writes nothing;
    composing would hand a consumer of the bundle a contract naming campaign writes, and a
    consumer of this family the bundle's insight reads — the union-port problem the curation
    family's port declines for the same reason.

    Structural, so a host's own storage satisfies it by having the methods.
    :class:`~threetears.evals.kernel.storage.EvalStorage` does, with no inheritance and no registration.
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
#: caller can type is not authorship: the surface knows who it is talking to and the caller does not have to be
#: trusted about it, so the surface says. ``declared_at`` and ``declared_by`` are stamped on every write of the
#: declaration for the same reason; ``call`` is recorded by its own write. The derived ``run_ids`` and
#: ``exploratory`` are not here: a create naming either is refused, since runs belong to aspects and whether a
#: campaign is exploratory is said beside the definition, never inside it.
CAMPAIGN_SERVER_FIELDS: tuple[str, ...] = (
    "id",
    "doc_type",
    "schema_version",
    "schema_minor",
    "created_at",
    "created_by",
    "scope_id",
    "declared_at",
    "declared_by",
    "call",
)

#: What an operator may change on a campaign after it exists. Membership and the control have their own write paths
#: carrying rules this one does not enforce; ``created_by`` is absent because a later editor is not the author.
_CAMPAIGN_UPDATABLE_FIELDS: tuple[str, ...] = ("name", "description", "question", "aspects", "floors")

#: The part of an update that rewrites the declaration: frozen once any aspect holds a run.
_DECLARATION_FIELDS = frozenset({"question", "aspects", "floors"})


#: A margin, as an aspect declares one: points of a 0-100 pass rate, strictly between the ends.
_MARGIN_DESCRIPTION = (
    "How far, in points of a 0-100 pass rate, a candidate may fall below the control on this aspect before it is "
    "shown worse: more than 0 and less than 100."
)


class ArmDefinition(EvalBaseModel):
    """An arm as a caller names it: a label, a role, and the runs it ran, from which the server reads its variant."""

    label: str = Field(min_length=1, description="What a reader calls the arm; unique within the campaign.")
    role: ArmRole = Field(
        default="candidate",
        description="`control`: the arm every other is read against, one at most. `candidate`: an arm read against it.",
    )
    from_run_ids: list[str] = Field(
        min_length=1,
        description=(
            "Runs of this arm, each held by one of the campaign's aspects: one is enough, one per aspect it ran in "
            "says more. Its variant key and lever map are read from the first; every other must carry the same key."
        ),
    )


class AspectDefinition(EvalBaseModel):
    """An aspect as a caller declares one: its label, its template, its margin, its design and its runs."""

    label: str = Field(min_length=1, description="What a reader calls the aspect; unique within the campaign.")
    template_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The template the aspect's runs ran. Omitted, it is the one template every run of the aspect ran, and "
            "none when they ran several."
        ),
    )
    margin: float | None = Field(default=None, gt=0.0, lt=100.0, allow_inf_nan=False, description=_MARGIN_DESCRIPTION)
    design: AspectDesign | None = Field(
        default=None,
        description="What the aspect sets out to vary and hold itself to. Omitted, the aspect is exploratory.",
    )
    run_ids: list[str] = Field(default_factory=list, description="The aspect's runs, each in the campaign's scope.")


class FloorDefinition(EvalBaseModel):
    """A floor as a caller declares one: the aspect it is read on and its clause. When it was declared is the server's."""

    aspect: str = Field(min_length=1, description="The label of the aspect whose readings the floor is read on.")
    clause: DecisionClause = Field(
        description="`{kind: rate_drop, max_points, reading?}` or `{kind: cost_lower}`: the line a candidate is warned for crossing."
    )


class AspectSource(EvalBaseModel):
    """A campaign whose aspects a create adopts, and, for a campaign of one aspect, the label it is adopted under."""

    campaign_id: str = Field(min_length=1, description="The campaign of this scope whose aspects are adopted.")
    label: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The label its one aspect is adopted under, its floors with it, where its own would collide with another "
            "aspect's. Omitted, the aspect keeps its label."
        ),
    )


class CampaignAuthoring(EvalBaseModel):
    """What a create may say, as :func:`create_campaign` validates it: the campaign's names, its question, arms,
    aspects and floors, and the margins of the aspects it adopts."""

    name: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    subject_kind: str = ""
    behavior: str = Field(min_length=1)
    description: str = ""
    archived: bool = False
    question: str = ""
    arms: list[ArmDefinition] = Field(default_factory=list)
    aspects: list[AspectDefinition] = Field(default_factory=list)
    floors: list[FloorDefinition] = Field(default_factory=list)
    margins: dict[str, Margin] = Field(default_factory=dict)


class _AddressedArm(NamedTuple):
    """An arm as a run addresses it: the key, the map it was digested from, and the predicate."""

    key: str
    levers: dict[str, SweepableValue] | None
    identity_version: int


def create_campaign(
    storage: CampaignStore,
    definition: dict[str, Any],
    *,
    scope_id: str,
    created_by: str,
    profile: HostProfile,
    exploratory: bool = False,
    aspects_from: Sequence[str | Mapping[str, Any]] = (),
) -> EvalCampaign:
    """Create and persist a campaign: one question, its arms, its aspects and its floors.

    ``definition`` names ``name``, ``subject_id`` and ``behavior`` (required), and optionally ``subject_kind``,
    ``description``, ``question``, ``arms`` (:class:`ArmDefinition`), ``aspects`` (:class:`AspectDefinition`),
    ``floors`` (:class:`FloorDefinition`) and ``margins`` (an adopted aspect's label to its margin). A name it does
    not declare is refused; server-owned fields (:data:`CAMPAIGN_SERVER_FIELDS`) are ignored.

    **Each aspect is a template and its runs.** Its template is the one it names, else the one every run it holds
    ran, else none; its runs must live in ``scope_id`` and may not span two revisions of one template. A declared
    aspect (one with a design) declares its margin where the campaign asks a question, and is asked it now.

    **Adoption keeps what it adopts, and dates what it adds.** ``aspects_from`` names campaigns of this scope (each a
    campaign id, or :class:`AspectSource` to relabel a campaign's one aspect) whose aspects this one adopts whole —
    template, runs, design, margin, when its design was declared and when its question was asked (``asked_at``) —
    with their floors and arms. An adopted aspect's readings stay as confirmatory as they were: nothing added here
    re-dates it. What is added is dated now, by itself: a margin (``Aspect.margin_declared_at``), a floor
    (``Floor.declared_at``), a control or candidate arm not already the source's (``CampaignArm.named_at``), and a
    question (the campaign's ``declared_at``). With no question given, the sources' questions are joined, each once.
    Two adopted aspects under one label, or a run two sources hold, are refused naming them.

    **Arms are addressed from runs** (:class:`ArmDefinition`): the variant key and lever map are the server's, read
    from the arm's first run, and every other run named for it must carry the same key, or the create is refused,
    naming each run and its key. An arm an adopted aspect already names must be the same stack, or the create is
    refused naming both keys. The control comes first.

    **A campaign is declared or exploratory, and the caller says which.** It is declared when an aspect, its own or
    an adopted one, declares a design; ``exploratory=True`` says none does, and is refused beside one, as a campaign
    declaring nothing is refused without it. An exploratory campaign names no question, arm, floor or margin.

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param definition: The campaign, as above.
    :ptype definition: dict[str, Any]
    :param scope_id: The scope the campaign lives in — the one its member runs live in.
    :ptype scope_id: str
    :param created_by: Who is creating it, as the calling surface knows them; also who declared what it declares.
    :ptype created_by: str
    :param profile: The host whose vocabulary this reads.
    :ptype profile: HostProfile
    :param exploratory: True to create the campaign with no design, saying so.
    :ptype exploratory: bool
    :param aspects_from: The campaigns whose aspects this one adopts, in order: each an id, or an
        :class:`AspectSource` (or its fields) relabelling a campaign's one aspect.
    :ptype aspects_from: Sequence[str | Mapping[str, Any]]
    :return: The persisted :class:`~threetears.evals.kernel.campaign.EvalCampaign`.
    :rtype: EvalCampaign
    :raises NotFoundError: A source campaign, or a template an aspect names, is not in the scope.
    :raises ValidationFailedError: The definition fails validation; names no aspect and adopts none; holds a run
        outside the scope, in two aspects, or of two revisions of one template in one aspect; adopts two aspects under
        one label, or a run two sources hold; asks a question and declares an aspect with no margin; gives a margin
        for an aspect it did not adopt, or a floor on an aspect that declares no design; says neither declared nor
        exploratory, or both; names an arm from a run no aspect holds, from runs carrying two keys, or one an adopted
        aspect knows under another key; or declares a design this host cannot honour.
    :raises LeverCoordinateError: An arm's run recorded no lever map, and the host's variant map disagrees with its
        own registry.
    :raises StorageError: The campaign failed to persist.
    """
    reject_unknown_authoring_fields("create_campaign", definition, CampaignAuthoring, CAMPAIGN_SERVER_FIELDS)
    try:
        authored = CampaignAuthoring.model_validate(
            {key: value for key, value in definition.items() if key not in CAMPAIGN_SERVER_FIELDS}
        )
        named_sources = [
            AspectSource(campaign_id=source) if isinstance(source, str) else AspectSource.model_validate(source)
            for source in aspects_from
        ]
    except ValidationError as e:
        raise ValidationFailedError(f"invalid campaign definition: {e}", remedy=SCHEMA_REMEDY) from e
    now = utc_now_iso()
    sources = _sources(storage, named_sources, scope_id)
    question = authored.question or QUESTION_SEPARATOR.join(
        dict.fromkeys(source.question for source in sources if source.question)
    )
    adopted = _adopted_aspects(sources, authored.margins, at=now)
    own = [_declared_aspect(storage, scope_id, aspect, at=now, asks=bool(question)) for aspect in authored.aspects]
    if not adopted and not own:
        raise ValidationFailedError(
            "a campaign measures its question on at least one aspect, and this create names none and adopts none",
            remedy="name one in aspects ({label, template_id?, margin, design?, run_ids}) or adopt one with aspects_from",
        )
    _refuse_collisions(sources, own)
    aspects = [*adopted, *own]
    declared = any(aspect.design is not None for aspect in aspects)
    _refuse_an_unsaid_kind(declared=declared, exploratory=exploratory)
    if not declared and (authored.question or authored.arms or authored.floors or authored.margins):
        raise ValidationFailedError(
            "an exploratory campaign declares no question, arm, floor or margin: each is read against a declared "
            "design, and no aspect of this one declares a design",
            remedy="declare a design on an aspect, or leave question, arms, floors and margins out",
        )
    adopted_labels = {aspect.label for aspect in adopted}
    floors = _floors(
        sources,
        [
            Floor(aspect=floor.aspect, clause=floor.clause, declared_at=now if floor.aspect in adopted_labels else None)
            for floor in authored.floors
        ],
    )
    _refuse_floors_on_undeclared_aspects(floors, aspects)
    aspects = _merit_axes_follow_the_question(aspects, question, own={aspect.label for aspect in own})
    declared_at, declared_by = _when_declared(
        sources, added=bool(authored.question) or any(a.design is not None for a in own), now=now, by=created_by
    )
    arms = (
        _resolve_arms(storage, scope_id, aspects, authored.arms, sources, at=now, profile=profile) if declared else []
    )
    try:
        campaign = EvalCampaign(
            scope_id=scope_id,
            name=authored.name,
            subject_id=authored.subject_id,
            subject_kind=authored.subject_kind,
            behavior=authored.behavior,
            description=authored.description,
            archived=authored.archived,
            question=question,
            arms=arms,
            aspects=aspects,
            floors=floors,
            declared_at=declared_at if declared else None,
            declared_by=declared_by if declared else "",
            created_by=created_by,
        )
    except ValidationError as e:
        raise ValidationFailedError(f"invalid campaign definition: {e}", remedy=SCHEMA_REMEDY) from e

    # A gate, not a page: a declaration this host cannot honour is refused before any run is launched. An adopted
    # aspect's design was gated when its source declared it, so it is gated again only when this create adds a
    # floor on it, which is read against its template.
    floored = {floor.aspect for floor in authored.floors}
    own_labels = {aspect.label for aspect in own}
    for aspect in campaign.aspects:
        if aspect.design is not None and (aspect.label in own_labels or aspect.label in floored):
            _gate_aspect(storage, campaign, aspect, profile=profile)

    storage.save_campaign(campaign)
    log.info(
        "eval.create_campaign campaign=%s aspects=%s adopted_from=%s by=%s",
        campaign.id,
        ",".join(aspect.label for aspect in campaign.aspects),
        ",".join(source.id for source in sources) or "(none)",
        created_by,
    )
    return campaign


#: What a campaign is when it is not declared, said once for every refusal that asks the caller to choose.
_THE_TWO_KINDS = (
    "an aspect's design states what the campaign sets out to learn; exploratory: true says it has no settled "
    "question yet, and its report will say its readings confirm nothing"
)


def _refuse_an_unsaid_kind(*, declared: bool, exploratory: bool) -> None:
    """Refuse a write that does not say, or contradicts itself about, whether the campaign is declared.

    :param declared: Whether any aspect the write leaves declares a design.
    :ptype declared: bool
    :param exploratory: The caller's assertion that none does.
    :ptype exploratory: bool
    :raises ValidationFailedError: Neither was said, or both were.
    """
    if declared and exploratory:
        raise ValidationFailedError(
            "the campaign both declares a design and says it is exploratory — exploratory means no aspect declares one",
            remedy="send one: an aspect with a design, or exploratory: true with none",
        )
    if not declared and not exploratory:
        raise ValidationFailedError(
            "a campaign is either declared or exploratory, and this write says neither",
            remedy=f"declare a design on an aspect, or send exploratory: true — {_THE_TWO_KINDS}",
        )


def _merit_axes_follow_the_question(aspects: Sequence[Aspect], question: str, *, own: set[str]) -> list[Aspect]:
    """``aspects`` with each design's merit axes the question's: a list when there is one, None when there is not.

    A declared aspect's ``merit_axes`` say what the question asks about there, so when the campaign asks one, an
    aspect that names none is asked about every axis (``[]``, unscoped); and when it asks none, an aspect of the
    caller's naming merit axes asks about something nobody asked.

    :raises ValidationFailedError: The campaign asks no question and an aspect the caller declares names merit axes.
    """
    if not question:
        if stray := [a.label for a in aspects if a.label in own and a.design is not None and a.design.merit_axes]:
            raise ValidationFailedError(
                f"aspect(s) {', '.join(map(repr, stray))} name merit axes, which say what the campaign's question asks "
                "about there, and the campaign asks none",
                remedy="ask the question (question), or leave merit_axes out",
            )
        return list(aspects)
    return [
        aspect.model_copy(update={"design": aspect.design.model_copy(update={"merit_axes": []})})
        if aspect.design is not None and aspect.design.merit_axes is None
        else aspect
        for aspect in aspects
    ]


def _sources(storage: CampaignStore, named: Sequence[AspectSource], scope_id: str) -> list[EvalCampaign]:
    """The campaigns a create adopts aspects from, each once, in order, each relabelled as named.

    :raises ValidationFailedError: A campaign is named twice, or relabelled while holding several aspects.
    :raises NotFoundError: One is not in the scope.
    """
    ids = [source.campaign_id for source in named]
    if repeated := sorted({campaign_id for campaign_id in ids if ids.count(campaign_id) > 1}):
        raise ValidationFailedError(
            f"aspects_from names {', '.join(map(repr, repeated))} more than once", remedy="name each campaign once"
        )
    found: list[EvalCampaign] = []
    for source in named:
        campaign = get_campaign(storage, source.campaign_id, scope_id)
        if source.label is not None:
            if len(campaign.aspects) != 1:
                raise ValidationFailedError(
                    f"aspects_from relabels campaign {campaign.id!r}, which has {len(campaign.aspects)} aspects, so "
                    "which one the label names is not implied",
                    remedy="relabel only a campaign of one aspect, or adopt this one under its own labels",
                )
            (aspect,) = campaign.aspects
            campaign = campaign.model_copy(
                update={
                    "aspects": [aspect.model_copy(update={"label": source.label})],
                    "floors": [floor.model_copy(update={"aspect": source.label}) for floor in campaign.floors],
                }
            )
        found.append(campaign)
    return found


def _adopted_aspects(sources: Sequence[EvalCampaign], margins: Mapping[str, float], *, at: str) -> list[Aspect]:
    """Every source's aspects, whole, with a margin given at adoption set and dated ``at``.

    Rebuilt through validation, so a margin the caller gave is held to the bounds every stored margin is.

    :raises ValidationFailedError: A margin names no adopted aspect.
    """
    adopted = [aspect for source in sources for aspect in source.aspects]
    labels = {aspect.label for aspect in adopted}
    if stray := sorted(set(margins) - labels):
        raise ValidationFailedError(
            f"margins names {', '.join(map(repr, stray))}, which is no adopted aspect: a margin given beside aspects "
            "is for an aspect this create adopts",
            remedy=f"declare an own aspect's margin on it; adopted aspects: {', '.join(map(repr, sorted(labels))) or 'none'}",
        )
    return [
        Aspect.model_validate({**aspect.model_dump(), "margin": margins[aspect.label], "margin_declared_at": at})
        if aspect.label in margins
        else aspect
        for aspect in adopted
    ]


def _refuse_collisions(sources: Sequence[EvalCampaign], own: Sequence[Aspect]) -> None:
    """Refuse two aspects under one label, or a run two of them hold, naming where each came from.

    :raises ValidationFailedError: Naming the label or the run, and the campaigns (or this create) holding it.
    """
    placed = [(aspect, f"campaign {source.id!r}") for source in sources for aspect in source.aspects]
    placed.extend((aspect, "this create") for aspect in own)
    seen: dict[str, str] = {}
    holder: dict[str, str] = {}
    for aspect, where in placed:
        if aspect.label in seen:
            raise ValidationFailedError(
                f"two aspects would be labelled {aspect.label!r}: {seen[aspect.label]}'s and {where}'s",
                remedy=(
                    "adopt one under another label (aspects_from: [{campaign_id, label}]), or label your own aspect "
                    "otherwise"
                ),
            )
        seen[aspect.label] = where
        for run_id in aspect.run_ids:
            held_by = f"{where} (aspect {aspect.label!r})"
            if run_id in holder:
                raise ValidationFailedError(
                    f"run {run_id!r} is held by {holder[run_id]} and by {held_by}, and a run belongs to one aspect",
                    remedy="adopt only one of the aspects holding it, or remove the run from one of them first",
                )
            holder[run_id] = held_by


def _declared_aspect(
    storage: CampaignStore, scope_id: str, defined: AspectDefinition, *, at: str, asks: bool
) -> Aspect:
    """An aspect a caller declares, its runs held to the scope and to one revision, its template resolved.

    :raises ValidationFailedError: A run is outside the scope or of another revision of a template it holds, or the
        aspect declares a design and no margin while the campaign asks a question.
    :raises NotFoundError: The template it names is not in the scope.
    """
    if defined.design is not None and defined.margin is None and asks:
        raise ValidationFailedError(
            f"aspect {defined.label!r} declares a design and no margin, and the campaign asks a question, so nothing "
            "would say how far a candidate may fall below the control there before it is shown worse",
            remedy=f"declare margin on aspect {defined.label!r}: {_MARGIN_DESCRIPTION}",
        )
    _refuse_runs_outside_scope(storage, defined.run_ids, scope_id)
    _refuse_pooled_revisions(storage, defined.run_ids, scope_id, what=f"aspect {defined.label!r}")
    declares = defined.design is not None
    return Aspect(
        label=defined.label,
        template_id=_aspect_template(storage, scope_id, defined.template_id, defined.run_ids),
        margin=defined.margin,
        design=defined.design,
        run_ids=list(dict.fromkeys(defined.run_ids)),
        declared_at=at if declares else None,
        asked_at=at if declares else None,
    )


def _aspect_template(
    storage: CampaignStore, scope_id: str, template_id: str | None, run_ids: Sequence[str]
) -> str | None:
    """The template an aspect is over: the one it names, else the one every run it holds ran, else none.

    A bar or a guardrail margin on a rubric dimension or goal-state check is read against the aspect's template,
    so an aspect whose runs all ran one template is over that template whether or not its author typed the id.

    :raises NotFoundError: The aspect names a template its scope does not hold.
    """
    if template_id:
        if storage.load_template(template_id, scope_id) is None:
            raise NotFoundError("template", template_id, remedy=FIND_TEMPLATE)
        return template_id
    ran = _templates_run(storage, scope_id, run_ids)
    return ran[0] if len(ran) == 1 and ran[0] != _NO_TEMPLATE else None


#: How an ad-hoc run's template is named where templates are listed.
_NO_TEMPLATE = "(no template)"


def _templates_run(storage: CampaignStore, scope_id: str, run_ids: Sequence[str]) -> list[str]:
    """The templates these runs ran, sorted, an ad-hoc run's as ``(no template)``."""
    ran = {
        run.template_id or _NO_TEMPLATE
        for run_id in run_ids
        if (run := storage.load_eval_run(run_id, scope_id)) is not None
    }
    return sorted(ran)


def _floors(sources: Sequence[EvalCampaign], authored: Sequence[Floor]) -> list[Floor]:
    """The floors a create leaves: each source's, then the caller's, a caller's floor replacing a source's of its key."""
    kept: dict[str, Floor] = {floor.key: floor for source in sources for floor in source.floors}
    kept.update({floor.key: floor for floor in authored})
    return list(kept.values())


def _refuse_floors_on_undeclared_aspects(floors: Sequence[Floor], aspects: Sequence[Aspect]) -> None:
    """Refuse a floor on an aspect that declares no design: what it warns of is read against one.

    :raises ValidationFailedError: Naming each such floor.
    """
    exploratory = {aspect.label for aspect in aspects if aspect.design is None}
    if stray := [floor.key for floor in floors if floor.aspect in exploratory]:
        raise ValidationFailedError(
            f"floor(s) {', '.join(stray)} are on an aspect that declares no design, whose readings confirm nothing",
            remedy="declare a design on the aspect, or put the floor on a declared one",
        )


def _instant_or_none(stamp: str | None) -> datetime | None:
    """``stamp`` as an aware instant, or None when it is absent or cannot be ordered."""
    if stamp is None:
        return None
    try:
        return aware_instant(stamp, "a declaration")
    except ValueError:
        return None


def _when_declared(sources: Sequence[EvalCampaign], *, added: bool, now: str, by: str) -> tuple[str | None, str]:
    """When, and by whom, a created campaign's question, as it is worded, was declared in full.

    Now, by its creator, when the create words a question or declares an aspect of its own; else the latest of its
    sources' declarations, since the questions it joins were all asked only once the last was. This dates the words,
    never an adopted aspect: each keeps its own ``asked_at``.
    """
    if added or not sources:
        return now, by
    dated = [(instant, source) for source in sources if (instant := _instant_or_none(source.declared_at)) is not None]
    if not dated:
        return None, ""
    latest = max(dated, key=lambda pair: pair[0])[1]
    return latest.declared_at, latest.declared_by


def _resolve_arms(
    storage: CampaignStore,
    scope_id: str,
    aspects: Sequence[Aspect],
    defined: Sequence[ArmDefinition],
    sources: Sequence[EvalCampaign],
    *,
    at: str,
    profile: HostProfile,
) -> list[CampaignArm]:
    """The arms a create names, each addressed from its runs, held to the arms its adopted aspects already name.

    An arm named here that is a source's arm keeps the source's ``named_at``: naming it again changes nothing that
    was read against it. Every other arm named here is named ``at``.

    :raises ValidationFailedError: An arm's run is in no aspect, its runs carry two keys, or an adopted aspect names
        an arm of the same label or role under another key; or two adopted aspects disagree about one.
    """
    holder = {run_id: aspect.label for aspect in aspects for run_id in aspect.run_ids}
    named = [_arm_from_runs(storage, scope_id, arm, holder, at=at, profile=profile) for arm in defined]
    for source in sources:
        for other in source.arms:
            index, same = next(
                (
                    (index, arm)
                    for index, arm in enumerate(named)
                    if arm.label == other.label or arm.role == other.role == "control" or _one_arm(arm, other)
                ),
                (None, None),
            )
            if same is None or index is None:
                named.append(other)
            elif not _one_arm(same, other):
                raise ValidationFailedError(
                    f"arm {same.label!r} carries variant key {same.variant_key}, and campaign {source.id!r}, whose "
                    f"aspect(s) {', '.join(repr(aspect.label) for aspect in source.aspects)} this create adopts, "
                    f"names its arm {other.label!r} by {other.variant_key}: one arm is one contestant stack in every "
                    "aspect, so its runs there carry one key",
                    remedy=(
                        "address the arm from a run carrying the adopted arm's key, or adopt only aspects that "
                        "agree about it"
                    ),
                )
            elif same.named_at == at:
                named[index] = same.model_copy(update={"named_at": other.named_at})
    return sorted(named, key=lambda arm: arm.role != "control")


def _one_arm(arm: CampaignArm, other: CampaignArm) -> bool:
    """Whether two arms name one variant: one's key is among the keys the other resolves to under this build."""
    return bool(
        {arm.variant_key, *resolve_control_arm(arm).variant_keys}
        & {other.variant_key, *resolve_control_arm(other).variant_keys}
    )


def _arm_from_runs(
    storage: CampaignStore,
    scope_id: str,
    defined: ArmDefinition,
    holder: Mapping[str, str],
    *,
    at: str,
    profile: HostProfile,
) -> CampaignArm:
    """One arm addressed from its runs, named ``at``: the first run's key and pre-image, every other run's the same.

    :raises ValidationFailedError: A run is held by no aspect, or the runs carry more than one key, naming each.
    """
    if outside := [run_id for run_id in defined.from_run_ids if run_id not in holder]:
        raise ValidationFailedError(
            f"arm {defined.label!r} is addressed from run(s) {', '.join(map(repr, outside))}, which no aspect of the "
            "campaign holds",
            remedy="name runs one of the campaign's aspects holds, or add them to an aspect",
        )
    addressed = {
        run_id: _address_arm(storage, scope_id, run_id, profile=profile)
        for run_id in dict.fromkeys(defined.from_run_ids)
    }
    first = next(iter(addressed.values()))
    if len({arm.key for arm in addressed.values()}) > 1:
        carried = "; ".join(
            f"run {run_id!r} (aspect {holder[run_id]!r}) carries {arm.key}" for run_id, arm in addressed.items()
        )
        raise ValidationFailedError(
            f"arm {defined.label!r} is addressed from runs that carry different variant keys — {carried}. An arm is "
            "one contestant stack, so every run of it carries one key",
            remedy="address the arm from runs of one stack, or name the other stack as an arm of its own",
        )
    return CampaignArm(
        label=defined.label,
        role=defined.role,
        variant_key=first.key,
        levers=first.levers,
        identity_version=first.identity_version,
        named_at=at,
    )


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


def _refuse_pooled_revisions(storage: CampaignStore, run_ids: Sequence[str], scope_id: str, *, what: str) -> None:
    """Refuse an aspect that would hold runs of two revisions of one template (:mod:`.revisions`).

    :raises ValidationFailedError: Naming the template and its revisions; nothing was written.
    """
    ran = (run.template_id for run_id in run_ids if (run := storage.load_eval_run(run_id, scope_id)) is not None)
    refuse_pooled_revisions(ran, lambda template_id: storage.load_template(template_id, scope_id), what=what)


def _refuse_runs_outside_scope(storage: CampaignStore, run_ids: Sequence[str], scope_id: str) -> None:
    """Refuse membership of any run that does not resolve in the campaign's scope.

    A run in another scope and a run that does not exist are the same thing from here — no read
    within this scope returns either — and both are refused for one reason: every read of a
    campaign's members is a read within its scope, so a member outside it is a run the campaign
    claims and no analysis of the campaign can see. Refused all-or-nothing, naming every id, so a
    partly-wrong request leaves the campaign as the caller last saw it.

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param run_ids: The runs to be held.
    :ptype run_ids: Sequence[str]
    :param scope_id: The campaign's scope.
    :ptype scope_id: str
    :raises ValidationFailedError: One or more of ``run_ids`` does not resolve in ``scope_id``.
    """
    if not run_ids:
        return
    resolved = {stamp.id for stamp in storage.load_eval_run_stamps(run_ids, scope_id)}
    if outside := sorted({run_id for run_id in run_ids if run_id not in resolved}):
        raise ValidationFailedError(
            f"run(s) {', '.join(repr(r) for r in outside)} do not exist in scope {scope_id!r} — a campaign holds "
            "only runs in its own scope, because every read of its members is a read within that scope",
            remedy=f"name runs of this scope ({FIND_RUN})",
        )


def _gate_aspect(storage: CampaignStore, campaign: EvalCampaign, aspect: Aspect, *, profile: HostProfile) -> None:
    """Refuse an aspect's declaration this host cannot honour — a gate, not a page.

    Runs on every authoring write of a declaration, never on a read: a campaign authored before a rule was added to
    the gate still loads, lists and analyses. The aspect's template is read because a bar, a guardrail margin or a
    floor may name one of its rubric dimensions or goal-state checks. The aspect is gated with the campaign's
    question and the floors on it, as the one question its design is asked under.

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param campaign: The campaign the aspect belongs to: its behavior keys a registered bar, its question and
        floors are what the aspect is asked.
    :ptype campaign: EvalCampaign
    :param aspect: The declared aspect to check.
    :ptype aspect: Aspect
    :param profile: The host whose vocabulary this reads.
    :ptype profile: HostProfile
    :raises ValidationFailedError: An axis the host will not accept, a bar naming nothing the aspect's results carry,
        looser than the registered incumbent or against its measure's direction, a guardrail margin or a floor on
        nothing the aspect measures, or latency asked about without being under test.
    """
    from threetears.evals.kernel.declaration import NO_TEMPLATE_REMEDY, DesignRefused, refuse_an_undeclarable_design

    if aspect.design is None:
        return
    template = storage.load_template(aspect.template_id, campaign.scope_id) if aspect.template_id else None
    try:
        refuse_an_undeclarable_design(
            _as_gated(aspect.design, campaign, aspect.label),
            behavior=campaign.behavior,
            template=template,
            profile=profile,
        )
    except DesignRefused as e:
        ran = (
            _templates_run(storage, campaign.scope_id, aspect.run_ids)
            if template is None and NO_TEMPLATE_REMEDY in e.fault
            else []
        )
        ran_note = (
            f". Aspect {aspect.label!r}'s runs ran {len(ran)} templates ({', '.join(ran)}), so none of them is its "
            "template until one is named"
            if len(ran) > 1
            else ""
        )
        raise ValidationFailedError(
            f"invalid design on aspect {aspect.label!r}: {e.fault}{ran_note}", remedy=e.remedy
        ) from e


#: The words a gated aspect's question carries when the campaign asks none: the gate reads a question's axes and rule.
_UNWORDED_QUESTION = "(the campaign states no question)"


def _as_gated(design: AspectDesign, campaign: EvalCampaign, label: str) -> CampaignDesign:
    """One aspect's declaration in the shape the authoring gate reads: its design, asked the campaign's question.

    The question carries the aspect's merit axes and its floors' clauses as its rule, so the gate checks the floors'
    readings and the latency the question asks about exactly as it checks a design's own.
    """
    clauses = tuple(floor.clause for floor in campaign.floors if floor.aspect == label)
    axes = list(design.merit_axes or [])
    asks = bool(campaign.question or clauses or axes)
    questions = (
        [
            Question(
                text=campaign.question or _UNWORDED_QUESTION,
                decision=DecisionRule(clauses=clauses) if clauses else None,
                merit_axes=axes,
            )
        ]
        if asks
        else []
    )
    shared = {name: getattr(design, name) for name in AspectDesign.model_fields if name != "merit_axes"}
    return CampaignDesign(**shared, questions=questions)


def declarable_axes(profile: HostProfile) -> DeclarableAxes:
    """The axes a campaign may declare on this host, read off the registry the gate decides with.

    Read from the same profile the gate
    (:func:`~threetears.evals.kernel.declaration.refuse_an_undeclarable_design`) is handed: an
    offer drawn from any other registry could promise an axis the gate then refuses.

    :param profile: The host whose registry the campaign is declared against.
    :ptype profile: HostProfile
    :return: The fixed levers, the open families whose members are declarable, and the remedy
        sentence a refused declaration carries.
    :rtype: DeclarableAxes
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
    exploratory: bool = False,
    control_from_run_id: str | None = None,
    clear_control: bool = False,
) -> EvalCampaign:
    """Amend a campaign's name, description, declaration or control.

    Only ``name``, ``description``, ``question``, ``aspects`` (:class:`AspectDefinition`, replacing the aspects whole)
    and ``floors`` may be named, and an unrecognised key is refused rather than ignored. A write of the declaration —
    ``question``, ``aspects`` or ``floors`` — is stamped with ``updated_by`` and dated now, and every aspect it
    declares is dated now too; the declaration gate runs here exactly as it does at creation.

    **The declaration is frozen once any aspect holds a run.** It is what those runs were gathered to answer, so once
    the campaign holds a run, ``question``, ``aspects`` and ``floors`` are refused in any form, and so is re-pointing
    or clearing a control already named. ``name`` and ``description`` stay editable, and a control may be named once,
    from a member run, while none is (which is how a comparison names its control after launching its arms). The
    amended declaration is a new campaign (``campaign_create``, which adopts the aspects it keeps with
    ``aspects_from``).

    **Aspects an update declares hold no runs**: runs join an aspect through :func:`add_runs_to_campaign`, and an
    update reaches the aspects only while the campaign holds none. A declared aspect declares its margin.

    **Withdrawing the declaration is said, not implied.** Aspects none of which declares a design leave the campaign
    exploratory only beside ``exploratory=True``, which also clears its question, floors and arms; and
    ``exploratory=True`` is refused beside a design, and on a declared campaign this update leaves declared.

    **The control is named here too** (``control_from_run_id``, resolved as :func:`set_campaign_control` resolves it)
    and ``clear_control`` clears it. Either needs a declared campaign — as this update leaves it.

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param campaign_id: The campaign to amend.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param updates: Partial field map; every key must be updatable.
    :ptype updates: dict[str, Any]
    :param updated_by: Who is amending it, as the calling surface knows them. Recorded on the declaration when this
        update writes one; it never becomes ``created_by``, since a later editor is not the author.
    :ptype updated_by: str
    :param profile: The host whose vocabulary this reads.
    :ptype profile: HostProfile
    :param exploratory: True to say the campaign this update leaves declares no design.
    :ptype exploratory: bool
    :param control_from_run_id: One of the campaign's runs whose variant becomes the control arm, or None.
    :ptype control_from_run_id: str | None
    :param clear_control: True to clear the control arm.
    :ptype clear_control: bool
    :return: The persisted, updated :class:`~threetears.evals.kernel.campaign.EvalCampaign`.
    :rtype: EvalCampaign
    :raises NotFoundError: No campaign with that id in the scope, or an aspect names a template the scope lacks.
    :raises ConflictError: The campaign holds a run, and the update writes the declaration, clears its control, or
        names a control where one is already named (:func:`_refuse_frozen_declaration`).
    :raises ValidationFailedError: Nothing to change, a field that is not updatable, an aspect holding runs or a
        declared one with no margin, a floor on an undeclared aspect, a declaration this host cannot honour; a
        withdrawal without ``exploratory``, or ``exploratory`` beside a design; or a control named beside
        ``clear_control``, on an exploratory campaign, or from a run the campaign does not hold.
    :raises LeverCoordinateError: ``control_from_run_id``'s run recorded no lever map, and the host's variant
        map disagrees with its own registry.
    :raises StorageError: The updated campaign failed to persist.
    """
    names_control = control_from_run_id is not None or clear_control
    if not updates and not names_control:
        raise ValidationFailedError(
            "the campaign update names nothing to change",
            remedy=f"name at least one of {', '.join(_CAMPAIGN_UPDATABLE_FIELDS)}, or the control (control_from_run_id)",
        )
    if unknown := sorted(set(updates) - set(_CAMPAIGN_UPDATABLE_FIELDS)):
        raise ValidationFailedError(
            f"campaign fields not updatable here: {', '.join(unknown)}",
            remedy=(
                f"update only {', '.join(_CAMPAIGN_UPDATABLE_FIELDS)}, and name the control with control_from_run_id; "
                "runs join and leave a campaign through campaign_runs_add and campaign_runs_remove"
            ),
        )
    if control_from_run_id is not None and clear_control:
        raise ValidationFailedError(
            "the update both names a control run and clears the control",
            remedy="send one: control_from_run_id to name the control, or clear_control alone to clear it",
        )

    campaign = get_campaign(storage, campaign_id, scope_id)
    writes = _DECLARATION_FIELDS & set(updates)
    _refuse_frozen_declaration(
        campaign,
        writes_declaration=bool(writes),
        clears_control=clear_control,
        names_control=control_from_run_id is not None,
    )
    changes: dict[str, Any] = {key: updates[key] for key in ("name", "description") if key in updates}
    now = utc_now_iso()
    question = updates["question"] if "question" in updates else campaign.question
    if not isinstance(question, str):
        raise ValidationFailedError("invalid campaign update: question is a string", remedy=SCHEMA_REMEDY)
    try:
        aspects = (
            [
                _updated_aspect(storage, scope_id, AspectDefinition.model_validate(item), at=now, asks=bool(question))
                for item in updates["aspects"]
            ]
            if "aspects" in updates
            else list(campaign.aspects)
        )
        floors = (
            [
                Floor(aspect=floor.aspect, clause=floor.clause)
                for floor in (FloorDefinition.model_validate(item) for item in updates["floors"])
            ]
            if "floors" in updates
            else list(campaign.floors)
        )
    except (ValidationError, TypeError) as e:
        raise ValidationFailedError(f"invalid campaign update: {e}", remedy=SCHEMA_REMEDY) from e
    declared = any(aspect.design is not None for aspect in aspects)
    if exploratory and declared:
        raise ValidationFailedError(
            f"campaign '{campaign_id}' would declare a design after this update, so exploratory: true says something "
            "it would not be",
            remedy="withdraw the declaration with it (aspects that declare no design), or leave exploratory out",
        )
    arms = list(campaign.arms)
    if not declared and not campaign.exploratory and not exploratory:
        # A declaration is what an analysis of the campaign is judged against, and un-declaring one turns every
        # reading after it into one that confirms nothing, so the caller says so, as a create with no design does.
        raise ValidationFailedError(
            "this update leaves no aspect declaring a design, which withdraws the declaration and leaves the campaign "
            "exploratory — which an update says rather than implies",
            remedy="send exploratory: true with it to withdraw the declaration, or keep a design on an aspect",
        )
    if not declared:
        question, floors, arms = "", [], []
        if writes:
            writes = writes | {"question"}
    _refuse_floors_on_undeclared_aspects(floors, aspects)
    if writes:
        declared_here = {aspect.label for aspect in aspects} if "aspects" in updates else set()
        # A declaration written anew is asked anew: before any run, so nothing it re-dates was read yet.
        aspects = [
            aspect.model_copy(update={"asked_at": now, "margin_declared_at": None})
            if aspect.design is not None
            else aspect
            for aspect in _merit_axes_follow_the_question(aspects, question, own=declared_here)
        ]
        floors = [floor.model_copy(update={"declared_at": None}) for floor in floors]
        changes.update(
            question=question,
            aspects=aspects,
            floors=floors,
            arms=arms,
            declared_at=now if declared else None,
            declared_by=updated_by if declared else "",
        )
    if names_control:
        if not declared:
            if control_from_run_id is not None:
                raise ValidationFailedError(
                    f"control_from_run_id names the run whose variant is the control, and campaign '{campaign_id}' "
                    + (
                        "would declare nothing once this update withdraws its declaration"
                        if writes
                        else "declares nothing"
                    )
                    + ". A control is the reference point of a comparison, so there is nothing for it to reference",
                    remedy="declare a design on an aspect in the same update, or leave control_from_run_id out",
                )
        else:
            held = campaign.model_copy(update={"aspects": aspects})
            changes["arms"] = _with_control(storage, held, arms, control_from_run_id, profile=profile)

    try:
        updated = edited_core_document(EvalCampaign, campaign, changes)
    except ValidationError as e:
        raise ValidationFailedError(f"invalid campaign update: {e}", remedy=SCHEMA_REMEDY) from e
    if writes:
        for aspect in updated.aspects:
            _gate_aspect(storage, updated, aspect, profile=profile)

    storage.save_campaign(updated)
    log.info(
        "eval.update_campaign campaign=%s fields=%s control=%s by=%s",
        campaign_id,
        ",".join(sorted(updates)) or "(none)",
        "(cleared)" if clear_control else f"from run {control_from_run_id}" if control_from_run_id else "(unchanged)",
        updated_by,
    )
    return updated


def _updated_aspect(storage: CampaignStore, scope_id: str, defined: AspectDefinition, *, at: str, asks: bool) -> Aspect:
    """An aspect an update declares: no runs (they join through :func:`add_runs_to_campaign`), dated now.

    :raises ValidationFailedError: The aspect names runs, or declares a design and no margin.
    """
    if defined.run_ids:
        raise ValidationFailedError(
            f"aspect {defined.label!r} names runs, and runs join an aspect through campaign_runs_add, whose rules an "
            "update would skip",
            remedy="leave run_ids out of the update, then add the runs with campaign_runs_add naming the aspect",
        )
    return _declared_aspect(storage, scope_id, defined, at=at, asks=asks)


def get_campaign(storage: CampaignStore, campaign_id: str, scope_id: str) -> EvalCampaign:
    """Load a campaign by id.

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param campaign_id: The campaign to load.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The campaign.
    :rtype: EvalCampaign
    :raises NotFoundError: No campaign with that id in the scope.
    """
    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id, remedy=FIND_CAMPAIGN)
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

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param campaign_id: Campaign to load.
    :ptype campaign_id: str
    :param scope_id: The scope the campaign and its member runs live in.
    :ptype scope_id: str
    :return: A :class:`~threetears.evals.kernel.campaign.CampaignView` — the campaign
        plus its derived window (or ``None``).
    :rtype: CampaignView
    :raises NotFoundError: No campaign with that id in the scope.
    """
    from threetears.evals.kernel.campaign import CampaignView, derive_window

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

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param scope_id: The scope to list.
    :ptype scope_id: str
    :param subject_id: Only campaigns over this subject.
    :ptype subject_id: str | None
    :param behavior: Only campaigns measuring this behavior.
    :ptype behavior: str | None
    :param archived: Only archived (``True``) or only active (``False``) campaigns.
    :ptype archived: bool | None
    :return: The matching campaigns, newest first.
    :rtype: list[EvalCampaign]
    """
    return storage.list_campaigns(
        scope_id,
        subject_id=subject_id,
        behavior=behavior,
        archived=archived,
    )


@serialized_campaign_write
def add_runs_to_campaign(
    storage: CampaignStore, campaign_id: str, scope_id: str, run_ids: list[str], *, aspect: str | None = None
) -> EvalCampaign:
    """Attach runs to one aspect of a campaign each (de-duplicated, order-preserving) and persist.

    **Each run joins one aspect.** ``aspect`` names it for every run; None routes each run by its template: the one
    aspect of the campaign gathered on the template the run ran takes it, and a campaign of one aspect takes it. A
    run no aspect or several aspects could take is refused, naming the run, its template and the aspects; so is a run
    of another template joining an aspect that names one, routed or named.

    A run the campaign already holds in the aspect it would join is ignored; one it holds in another aspect is
    refused, since a run belongs to one aspect. A run that does not resolve in the campaign's scope is refused, as at
    creation, and so is an aspect that would hold runs of two revisions of one template. Every refusal is
    all-or-nothing. Loads, merges and re-saves here, so a failed persist raises ``StorageError``.

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param campaign_id: Campaign to attach to.
    :ptype campaign_id: str
    :param scope_id: The scope the campaign and the runs live in.
    :ptype scope_id: str
    :param run_ids: Run ids to add; already-attached ids are ignored.
    :ptype run_ids: list[str]
    :param aspect: The aspect every run joins, by label; None routes each by its template.
    :ptype aspect: str | None
    :return: The persisted, updated :class:`~threetears.evals.kernel.campaign.EvalCampaign`.
    :rtype: EvalCampaign
    :raises NotFoundError: No campaign with that id in the scope.
    :raises ValidationFailedError: The aspect is not the campaign's; a run is outside the scope, held by another
        aspect, or of a template no aspect or several aspects are gathered on; or an aspect would hold two revisions
        of one template. Nothing was attached.
    :raises StorageError: The updated campaign failed to persist.
    """
    campaign = get_campaign(storage, campaign_id, scope_id)
    target = _named_aspect(campaign, aspect)
    requested = list(dict.fromkeys(run_ids))
    fresh = [run_id for run_id in requested if campaign.aspect_of_run(run_id) is None]
    if target is not None and (
        elsewhere := [
            (run_id, held.label)
            for run_id in requested
            if (held := campaign.aspect_of_run(run_id)) is not None and held.label != target.label
        ]
    ):
        raise ValidationFailedError(
            "run(s) "
            + ", ".join(f"{run_id!r} (held by aspect {label!r})" for run_id, label in elsewhere)
            + f" cannot join aspect {target.label!r}: a run belongs to one aspect of a campaign — nothing was attached",
            remedy="remove the run from its aspect first (campaign_runs_remove), or leave it where it is",
        )
    _refuse_runs_outside_scope(storage, fresh, scope_id)
    routed = (
        {run_id: target.label for run_id in fresh}
        if target is not None
        else _route_by_template(storage, campaign, fresh)
    )
    aspects = [
        held.model_copy(
            update={"run_ids": [*held.run_ids, *(run_id for run_id in fresh if routed[run_id] == held.label)]}
        )
        for held in campaign.aspects
    ]
    for joined in aspects:
        if any(routed.get(run_id) == joined.label for run_id in fresh):
            _refuse_pooled_revisions(
                storage, joined.run_ids, scope_id, what=f"aspect {joined.label!r} of campaign {campaign.name!r}"
            )
    _refuse_another_template(storage, campaign, routed)
    campaign = campaign.model_copy(update={"aspects": aspects})
    storage.save_campaign(campaign)
    log.info("eval.add_runs_to_campaign campaign=%s attached=%d", campaign_id, len(fresh))
    return campaign


def _refuse_another_template(storage: CampaignStore, campaign: EvalCampaign, routed: Mapping[str, str]) -> None:
    """Refuse a run joining an aspect gathered on another template: the aspect's bars and floors read its own.

    An aspect that names no template takes a run of any; one that names a template takes only that template's runs,
    whether it was named or is the campaign's lone aspect.

    :raises ValidationFailedError: Naming each such run, the template it ran and the aspect's.
    """
    strays = []
    for run_id, label in routed.items():
        aspect = campaign.aspect(label)
        if aspect is None or not aspect.template_id:
            continue
        run = storage.load_eval_run(run_id, campaign.scope_id)
        ran = run.template_id if run is not None else None
        if ran != aspect.template_id:
            strays.append(
                f"run {run_id!r} ran {f'template {ran!r}' if ran else 'no template'}, and aspect {label!r} is gathered on {aspect.template_id!r}"
            )
    if strays:
        raise ValidationFailedError(
            f"{'; '.join(strays)} — an aspect reads its own template's runs, so nothing was attached",
            remedy="add the run to the aspect gathered on its template, or to a new campaign's aspect over it",
        )


def _named_aspect(campaign: EvalCampaign, label: str | None) -> Aspect | None:
    """The campaign's aspect labelled ``label``, or None when no label is given.

    :raises ValidationFailedError: The campaign has no aspect of that label.
    """
    if label is None:
        return None
    found = campaign.aspect(label)
    if found is None:
        raise ValidationFailedError(
            f"campaign '{campaign.id}' has no aspect {label!r}",
            remedy=f"name one of its aspects: {', '.join(repr(aspect.label) for aspect in campaign.aspects)}",
        )
    return found


def _route_by_template(storage: CampaignStore, campaign: EvalCampaign, run_ids: Sequence[str]) -> dict[str, str]:
    """Each run's aspect, by the template it ran: the one aspect gathered on it; a lone aspect takes every run.

    :raises ValidationFailedError: A run ran a template no aspect, or more than one aspect, is gathered on — naming
        each such run, its template and every aspect's.
    """
    if len(campaign.aspects) == 1:
        return {run_id: campaign.aspects[0].label for run_id in run_ids}
    routed: dict[str, str] = {}
    unroutable: list[str] = []
    for run_id in run_ids:
        run = storage.load_eval_run(run_id, campaign.scope_id)
        template_id = run.template_id if run is not None else None
        takers = [aspect.label for aspect in campaign.aspects if template_id and aspect.template_id == template_id]
        if len(takers) == 1:
            routed[run_id] = takers[0]
        else:
            ran = f"template {template_id!r}" if template_id else "no template"
            why = "no aspect is gathered on it" if not takers else f"aspects {', '.join(map(repr, takers))} all are"
            unroutable.append(f"run {run_id!r} ran {ran}, and {why}")
    if unroutable:
        gathered = ", ".join(
            f"{aspect.label!r} on {aspect.template_id or 'no template'}" for aspect in campaign.aspects
        )
        raise ValidationFailedError(
            f"which aspect of campaign '{campaign.id}' a run joins is not implied — {'; '.join(unroutable)}. Its "
            f"aspects: {gathered} — nothing was attached",
            remedy="name the aspect the runs join (aspect)",
        )
    return routed


@serialized_campaign_write
def remove_runs_from_campaign(
    storage: CampaignStore, campaign_id: str, scope_id: str, run_ids: list[str], *, aspect: str | None = None
) -> EvalCampaign:
    """Detach runs from a campaign — the inverse of :func:`add_runs_to_campaign`.

    Detaching removes MEMBERSHIP only; the run document and its results are untouched and stay readable, and the run
    keeps whatever other campaigns it belongs to. Each run leaves the aspect holding it (a run is held by one aspect
    at most, so no aspect need be named); ``aspect``, when named, must be the one that holds every run.

    A requested id that is not attached — or attached to another aspect than the one named — is **refused**, naming
    the ids, all-or-nothing, so a partly-wrong request leaves the campaign exactly as the caller last saw it.

    No control guard: an arm names a VARIANT, so detaching a run cannot invalidate it. Where another member carries
    the same stack the arm is untouched, and where none does the bundle reports the arm as unobserved.

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param campaign_id: Campaign to detach from.
    :ptype campaign_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param run_ids: Run ids to remove; every one must currently be attached.
    :ptype run_ids: list[str]
    :param aspect: The aspect holding them, by label, or None for wherever each is held.
    :ptype aspect: str | None
    :return: The persisted, updated :class:`~threetears.evals.kernel.campaign.EvalCampaign`.
    :rtype: EvalCampaign
    :raises NotFoundError: No campaign with that id in the scope.
    :raises ValidationFailedError: The aspect is not the campaign's, or an id is not attached (to it).
    :raises StorageError: The updated campaign failed to persist.
    """
    campaign = get_campaign(storage, campaign_id, scope_id)
    target = _named_aspect(campaign, aspect)
    held = set(target.run_ids) if target is not None else set(campaign.run_ids)
    if missing := [run_id for run_id in run_ids if run_id not in held]:
        where = (
            f"aspect {target.label!r} of campaign '{campaign_id}'"
            if target is not None
            else f"campaign '{campaign_id}'"
        )
        raise ValidationFailedError(
            f"{where} does not hold run(s) {', '.join(repr(m) for m in missing)} — nothing was detached",
            remedy=f"detach only runs it holds: {', '.join(sorted(held)) or 'it holds none'}",
        )
    removing = set(run_ids)
    campaign = campaign.model_copy(
        update={
            "aspects": [
                kept.model_copy(update={"run_ids": [run_id for run_id in kept.run_ids if run_id not in removing]})
                for kept in campaign.aspects
            ]
        }
    )
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
    """Name (or clear) the campaign's control arm, ADDRESSED from a member run.

    The control is a variant key — the resolved contestant stack every other arm is read against — and a key is a
    sha256 nobody types. So the operator points at a member run, and this resolves it: the key the run's
    observations carry, beside its pre-image (the run's lever map and the predicate that digested it), so a read
    under a later predicate re-keys the arm instead of losing it
    (:func:`~threetears.evals.kernel.declaration.resolve_control_arm`). A run with no observation yet can be
    addressed: the key is read from the lever map its launch recorded, as the runner reads it.

    The arm keeps the label a control already named carries, and is labelled ``control`` otherwise; it is the
    campaign's first arm. **The campaign must declare a design on some aspect**: a control is the reference point of
    a comparison, and an exploratory campaign names no arms. **Once, on a campaign with runs**: re-pointing or
    clearing a control already named once a run has joined is refused, as every change to a frozen declaration is.
    Naming the control does not re-date the question: when it was asked is not when its control was addressed.

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param campaign_id: Campaign to designate on.
    :ptype campaign_id: str
    :param scope_id: The scope the campaign and its runs live in.
    :ptype scope_id: str
    :param run_id: A member run whose variant becomes the control, or None to clear it.
    :ptype run_id: str | None
    :param set_by: Who is designating, as the calling surface knows them; logged beside the arm.
    :ptype set_by: str
    :param profile: The host whose vocabulary this reads.
    :ptype profile: HostProfile
    :return: The persisted campaign — or the campaign untouched when CLEARING a control on one that names none.
    :rtype: EvalCampaign
    :raises NotFoundError: No campaign with that id in the scope.
    :raises ConflictError: The campaign holds a run and already names a control (:func:`_refuse_frozen_declaration`).
    :raises ValidationFailedError: A designation on an exploratory campaign, or ``run_id`` is not a member.
    :raises LeverCoordinateError: The run recorded no lever map, and the host's variant map disagrees with its own
        registry.
    :raises StorageError: The updated campaign failed to persist.
    """
    campaign = get_campaign(storage, campaign_id, scope_id)
    if run_id is None and campaign.control is None:
        return campaign
    if campaign.exploratory:
        raise ValidationFailedError(
            f"campaign '{campaign_id}' declares no design on any aspect, and a control is the reference point of a "
            "comparison, so there is nothing for it to reference yet",
            remedy="declare a design on an aspect with campaign_update, which names the control in the same call "
            "(control_from_run_id)",
        )
    _refuse_frozen_declaration(
        campaign, writes_declaration=False, clears_control=run_id is None, names_control=run_id is not None
    )
    previous = campaign.control
    arms = _with_control(storage, campaign, list(campaign.arms), run_id, profile=profile)
    try:
        campaign = edited_core_document(EvalCampaign, campaign, {"arms": arms})
    except ValidationError as e:
        raise ValidationFailedError(f"invalid control: {e}", remedy=SCHEMA_REMEDY) from e
    storage.save_campaign(campaign)
    log.info(
        "eval.set_campaign_control campaign=%s control=%s was=%s from_run=%s by=%s",
        campaign_id,
        campaign.control.variant_key if campaign.control is not None else "(cleared)",
        previous.variant_key if previous is not None else "(none)",
        run_id or "(n/a)",
        set_by,
    )
    return campaign


def _with_control(
    storage: CampaignStore,
    campaign: EvalCampaign,
    arms: Sequence[CampaignArm],
    run_id: str | None,
    *,
    profile: HostProfile,
) -> list[CampaignArm]:
    """``arms`` with the control addressed from a member run first, named now, or with no control.

    When it is named is recorded on the arm (``named_at``): a control named after an aspect's first run was chosen
    once the evidence existed, so that aspect's comparisons read as post hoc
    (:func:`~threetears.evals.analysis.aspects.aspect_view` dates the aspect's question no earlier than it).

    :raises ValidationFailedError: The run is not a member of the campaign.
    """
    candidates = [arm for arm in arms if arm.role != "control"]
    if run_id is None:
        return candidates
    if campaign.aspect_of_run(run_id) is None:
        raise ValidationFailedError(
            f"run '{run_id}' is not attached to campaign '{campaign.id}'",
            remedy="add the run to an aspect of the campaign before addressing its variant as the control",
        )
    addressed = _address_arm(storage, campaign.scope_id, run_id, profile=profile)
    label = campaign.control.label if campaign.control is not None else UPGRADED_CONTROL_LABEL
    control = CampaignArm(
        label=label,
        role="control",
        variant_key=addressed.key,
        levers=addressed.levers,
        identity_version=addressed.identity_version,
        named_at=utc_now_iso(),
    )
    return [control, *candidates]


def _refuse_frozen_declaration(
    campaign: EvalCampaign, *, writes_declaration: bool, clears_control: bool, names_control: bool
) -> None:
    """Refuse a change to what a campaign holding a run declared: it is what those runs were gathered to answer.

    The one rule :func:`update_campaign` and :func:`set_campaign_control` hold a campaign to: with no run in any
    aspect, nothing is frozen. With one, the declaration (question, aspects, floors) is refused in any form, and so is
    a control re-pointed or cleared; a control named for the first time is allowed. ``name`` and ``description`` are
    never asked about here.

    :raises ConflictError: The campaign holds a run and the change does one of the refused things.
    """
    held = campaign.run_ids
    if not held:
        return
    if writes_declaration:
        changed = "its declaration (question, aspects or floors)"
    elif clears_control and campaign.control is not None:
        changed = "its control (clearing it)"
    elif names_control and campaign.control is not None:
        changed = "its control (re-pointing it; a control is addressed once)"
    else:
        return
    raise ConflictError(
        f"campaign '{campaign.id}' holds {len(held)} run(s), so what it declared is frozen: it is what those runs were "
        f"gathered to answer, and changing {changed} now would make a question asked after the results read as one "
        "asked before them",
        remedy=(
            "create a new campaign with the amended declaration (campaign_create, adopting the aspects it keeps with "
            "aspects_from) and launch into it; name and description stay editable here"
        ),
    )


def campaign_arms(storage: CampaignStore, campaign: EvalCampaign, *, profile: HostProfile) -> list[CampaignArm]:
    """The campaign's arms as a reader reads them: those a person named, then one per other variant its runs carry.

    A campaign stored before arms existed names its control alone, and a campaign named from a sweep names none, so
    every variant among an aspect's runs that no named arm resolves to is read as a candidate of its own, in aspect
    order then the order its runs first carry it. A derived candidate is labelled by the levers on which the
    campaign's variants differ (:func:`~threetears.evals.analysis.arms.arm_names`), its digest beside the label where
    two would read alike, and a ``(2)``, ``(3)`` … where a label is still taken. Derived on every read and never
    stored, so a person naming the arms later replaces them whole.

    **A derived label is not an identity.** It is read over the variants the runs carry now, so a variant joining
    the campaign can change the levers the variants differ on, and with them every derived label. Whatever names a
    derived arm durably names it by its ``variant_key``: a recorded call (``CampaignCall``) is matched to its arm by
    ``variant_key``, never by ``arm``, which is the label it was read under.

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param profile: The host whose variant map a run assembled without the launch is read through.
    :ptype profile: HostProfile
    :return: The named arms, the control first, then the derived candidates.
    :rtype: list[CampaignArm]
    :raises LeverCoordinateError: A member run recorded no lever map, and the host's variant map disagrees with its
        own registry.
    """
    named = list(campaign.arms)
    known = {key for arm in named for key in (arm.variant_key, *resolve_control_arm(arm).variant_keys)}
    derived: dict[str, _AddressedArm] = {}
    for aspect in campaign.aspects:
        for run_id in aspect.run_ids:
            if storage.load_eval_run(run_id, campaign.scope_id) is None:
                continue
            addressed = _address_arm(storage, campaign.scope_id, run_id, profile=profile)
            if addressed.key not in known and addressed.key not in derived:
                derived[addressed.key] = addressed
    if not derived:
        return named
    names = arm_names([_index_entry(arm) for arm in [*map(_addressed, named), *derived.values()]])
    taken = {arm.label for arm in named}
    arms = list(named)
    for key, addressed in derived.items():
        label, suffix = names[key], 2
        while label in taken:
            label, suffix = f"{names[key]} ({suffix})", suffix + 1
        taken.add(label)
        arms.append(
            CampaignArm(
                label=label,
                role="candidate",
                variant_key=key,
                levers=addressed.levers,
                identity_version=addressed.identity_version,
            )
        )
    return arms


def _addressed(arm: CampaignArm) -> _AddressedArm:
    """A named arm as its pre-image addresses it."""
    return _AddressedArm(arm.variant_key, arm.levers, arm.identity_version)


def _index_entry(arm: _AddressedArm) -> VariantIndexEntry:
    """The variant index entry an arm is named from: its levers where they digest to its key under this build."""
    if arm.levers is not None and compute_variant_key(arm.levers) == arm.key:
        return VariantIndexEntry(variant_key=arm.key, levers=dict(arm.levers))
    return VariantIndexEntry(
        variant_key=arm.key,
        levers={},
        levels_unavailable="its lever map was not recorded, or was digested by another identity predicate",
    )


def _address_arm(storage: CampaignStore, scope_id: str, run_id: str, *, profile: HostProfile) -> _AddressedArm:
    """The variant key a run's observations carry, with its pre-image, or a refusal naming what is missing.

    **The key and its pre-image are read together, as the run reads them.** The key is the one the
    run's observations were stamped with, under the predicate stamped beside it; the lever map is the
    one the run recorded at launch (or, for a run its host assembled without the launch, the one this
    build derives — kept only where it reproduces the stamped key). Recording the map is what lets a
    read under a later predicate re-key the arm
    (:func:`~threetears.evals.kernel.declaration.resolve_control_arm`); a key alone would match no
    run of its own arm the moment the predicate moved, and nothing could recover it.

    A map that does not reproduce the key under this build's predicate is not recorded: it is not the
    key's pre-image, and re-keying from it would move the arm onto a stack nobody chose. The arm
    is then the bare key, as a typed one is, and says so where it is read. A recorded map stamped under
    ANOTHER predicate is kept, because it is what that predicate digested — holding it to this build's
    arithmetic is the version coupling the map exists to remove.

    :param storage: The campaign store.
    :ptype storage: CampaignStore
    :param scope_id: The campaign's scope, which the run is read in.
    :ptype scope_id: str
    :param run_id: The run to address; the caller holds it to the campaign's membership.
    :ptype run_id: str
    :param profile: The host whose vocabulary this reads.
    :ptype profile: HostProfile
    :return: The 64-hex variant key, the lever map it was digested from (None where none reproduces it), and
        the identity version it was computed under.
    :rtype: _AddressedArm
    :raises ValidationFailedError: The run does not resolve in the campaign's scope.
    :raises LeverCoordinateError: The run recorded no lever map (its host assembled it without the launch),
        and the host's variant map disagrees with its own registry — as assembling a bundle over the
        run would refuse too.
    """
    from threetears.evals.kernel.identity import IDENTITY_VERSION, resolve_variant_identity

    run = storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise ValidationFailedError(
            f"run '{run_id}' does not exist in scope '{scope_id}' — nothing to resolve a variant from",
            remedy=f"name a run of this scope ({FIND_RUN})",
        )
    identity = resolve_variant_identity(run=run, profile=profile)
    results = storage.query_eval_results_by_run(run_id, scope_id)
    key = variant_key_of_run(results)
    if key is None:
        # A run launched a moment ago has no observation yet, and a designation made AT launch
        # is the common case, not an edge: the operator picks the control while choosing the
        # arms. Every result such a run writes is stamped by digesting the lever map
        # `resolve_variant_identity` reads — the runner's own read — so the key its observations
        # WILL carry is known now, from the same read, not a second derivation.
        return _AddressedArm(identity.variant_key, identity.levers, identity.identity_version)
    version = results[0].identity_version
    reproduces = identity.variant_key == key
    recorded_under_another = run.variant_levers is not None and version != IDENTITY_VERSION
    return _AddressedArm(key, identity.levers if reproduces or recorded_under_another else None, version)


if TYPE_CHECKING:

    def _eval_storage_satisfies_the_port(storage: EvalStorage) -> None:
        """Hold the engine's own store to this consumer's port, so a drifted signature fails typecheck."""
        store: CampaignStore = storage
        del store


__all__ = [
    "CAMPAIGN_SERVER_FIELDS",
    "ArmDefinition",
    "AspectDefinition",
    "AspectSource",
    "CampaignAuthoring",
    "FloorDefinition",
    "CampaignStore",
    "DeclarableAxes",
    "OpenAxisFamily",
    "add_runs_to_campaign",
    "campaign_arms",
    "create_campaign",
    "declarable_axes",
    "get_campaign",
    "get_campaign_view",
    "list_campaigns",
    "remove_runs_from_campaign",
    "set_campaign_control",
    "update_campaign",
]
