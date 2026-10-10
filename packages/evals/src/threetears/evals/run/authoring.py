"""Authoring the definitions a run is built from: templates, catalog rubric dims and judge configs.

Each operation is a function over :class:`~threetears.evals.kernel.storage.DefinitionStore` and typed
parameters, the shape the curation family set (:mod:`threetears.evals.run.curation`), so a client of
the package can author definitions without a host's service layer, and every surface that reaches
these translates failures the same way (NotFound → 404, Conflict → 409, Validation → 422).

**Every definition lives in one scope, named by the caller.** ``scope_id`` is a keyword argument of
every operation here and a server-owned field of every definition: a value in the authored fields
is ignored like ``id``, because which scope a definition lands in is the surface's to say, not the
document body's.

**The refusals here are the engine's: rules over engine models.** A rubric dim's name carries its
scoring context; a template's preconditions and goal-state checks are ``EvalTemplate`` fields any
world-bearing kind evaluates, so whether a host's world can supply what they name is asked against
the host's profile's registries. What only a host can answer arrives as a callable the caller
passes in:

* ``require_known_tools_allowed`` — whether each name in ``tools_allowed`` is a tool type the
  host's catalog has;
* ``refuse_undeclared_world_seed`` — whether a ``world_seed`` names only paths the host's world
  can seed, which is the host's walk over its own namespaces;
* ``refuse_undeliverable_template`` — whether the template declares apparatus its candidate kind
  can honour, which is the host's kind capability table.

Each hook raises :class:`~threetears.evals.kernel.errors.ValidationFailedError` to refuse and
returns ``None`` to admit. They are required rather than defaulted: a host with nothing to check
passes a function that checks nothing, which is a decision its code states.

The world reads go through the host's profile, handed in: a template function that reads both
the store and the world takes the :class:`~threetears.evals.kernel.host.eval_host.EvalHost`, and
the two gates that read the world alone take its profile.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from threetears.evals.kernel.authoring_fields import reject_unknown_authoring_fields
from threetears.evals.schema.goal_grammar import DSLError, extract_paths
from threetears.evals.kernel.dsl import (
    call_parameter_matches,
    undefined_call_references,
    undefined_fire_references,
    world_prose_matches,
)
from threetears.evals.kernel.errors import ConflictError, NotFoundError, StorageError, ValidationFailedError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.host.kinds import freeze
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.host.world import resolve_preconditions
from threetears.evals.schema.models import (
    CatalogRubricDim,
    EvalTemplate,
    JudgeConfig,
    RubricDimTombstone,
    JudgeConfigTombstone,
    utc_now_iso,
)
from threetears.evals.run.check_controls import refuse_non_discriminating_checks
from threetears.evals.run.curation import require_delete_confirmation
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.kernel.storage import DefinitionStore

log = get_logger(__name__)


#: Identity/lifecycle fields the template family owns — never taken from caller input.
#:
#: PUBLIC (no leading underscore) because the MCP handlers read these to build their
#: "what changed" summaries, and must read the same set the authoring path enforces rather than
#: a copy of it. They once carried copies, and the two had already diverged by
#: one field — correctly, and only because someone noticed. A private name reached
#: across a package boundary is the other way that goes wrong, so these are declared as
#: what they are: part of what a host's service tells its surfaces.
TEMPLATE_SERVER_FIELDS: tuple[str, ...] = (
    "id",
    "doc_type",
    "schema_version",
    "created_at",
    "updated_at",
    "scope_id",
)


def validated_kind_spec(template: EvalTemplate, *, profile: HostProfile) -> BaseModel | None:
    """The template's ``kind_spec`` as its kind's spec model validates it, refused by field.

    Asked where a template is authored and again where it launches: the spec model is the kind's
    code, so a spec valid when it was written can be refused after the model moves, and the launch
    is where that must be found rather than mid-run.

    Args:
        template: The template, whose ``candidate_kind`` names the kind.
        profile: The host, whose kind contracts declare each kind's spec model.

    Returns:
        The validated spec, or ``None`` for a kind with no spec model and a template stating nothing.

    Raises:
        ValidationFailedError: The kind's spec model refuses the spec — naming each field and the
            template — or the kind declares no spec and the template states one.
    """
    try:
        return profile.kind_contract(template.candidate_kind).validate_spec(template.kind_spec)
    except ValidationFailedError as refused:
        raise ValidationFailedError(f"template {template.name!r}: {refused.message}", refused.details) from refused


def create_template(
    host: EvalHost,
    definition: dict[str, Any],
    *,
    scope_id: str,
    require_known_tools_allowed: Callable[[Sequence[str] | None], None],
    refuse_undeclared_world_seed: Callable[[EvalTemplate], None],
    refuse_undeliverable_template: Callable[[EvalTemplate], None],
) -> EvalTemplate:
    """Create and persist a template from an authoring definition.

    A name the model does not declare is refused; server-owned identity/lifecycle fields in ``definition`` are ignored —
    a fresh ``id`` and ``created_at`` / ``updated_at`` are always assigned.
    Names are unique within a scope: a second create with an
    existing name raises :class:`ConflictError` rather than silently
    shadowing it (``load_template_by_name`` resolves first-match, so a
    duplicate would be unreachable by name).

    The checks run in a fixed order, so a template failing several meets the same refusal on
    every surface: the kind's spec model, the rubric's names, the host's tool catalog, the world its
    expressions name, whether each goal check is proven to discriminate, the host's seed walk, the
    host's kind capabilities, then the name's uniqueness. The template is stored with its
    ``kind_spec`` as the spec model resolved it — every field, defaults included.

    Args:
        host: The host: where templates are read and written, and the world the template's
            expressions are held to.
        definition: Template fields (``name`` + ``intent`` required; the
            rest optional per :class:`~threetears.evals.schema.models.EvalTemplate`).
        scope_id: The scope the template lives in.
        require_known_tools_allowed: The host's check that each name in ``tools_allowed`` is a
            tool type its catalog has; raises ``ValidationFailedError`` to refuse.
        refuse_undeclared_world_seed: The host's check that the template's ``world_seed``
            names only paths its world can seed; raises ``ValidationFailedError`` to refuse.
        refuse_undeliverable_template: The host's check that the template declares only
            apparatus its candidate kind can honour; raises ``ValidationFailedError`` to refuse.

    Returns:
        The persisted :class:`~threetears.evals.schema.models.EvalTemplate`.

    Raises:
        ValidationFailedError: ``definition`` fails template validation, its
            ``kind_spec`` is refused by the kind's spec model (naming the field),
            its rubric carries a non-namespaced dim name, its ``tools_allowed``
            names a tool type the catalog does not have, one of its
            expressions names world state this host cannot supply, a goal check
            carries no control or does not discriminate between its controls
            (:func:`~threetears.evals.run.check_controls.refuse_non_discriminating_checks`), its
            ``world_seed`` names a dimension no run of this host can seed, or
            it declares apparatus its candidate kind cannot honour.
        ConflictError: a template with the same name already exists.
    """
    reject_unknown_authoring_fields("create_template", definition, EvalTemplate, TEMPLATE_SERVER_FIELDS)
    clean = {k: v for k, v in definition.items() if k not in TEMPLATE_SERVER_FIELDS}
    try:
        template = EvalTemplate(**clean, scope_id=scope_id)
    except ValidationError as e:
        raise ValidationFailedError(f"invalid template definition: {e}") from e

    template = admit_template(
        template,
        profile=host.profile,
        require_known_tools_allowed=require_known_tools_allowed,
        refuse_undeclared_world_seed=refuse_undeclared_world_seed,
        refuse_undeliverable_template=refuse_undeliverable_template,
    )

    if host.storage.load_template_by_name(template.name, scope_id) is not None:
        raise ConflictError(f"a template named '{template.name}' already exists")

    host.storage.save_template(template)
    return template


def admit_template(
    template: EvalTemplate,
    *,
    profile: HostProfile,
    require_known_tools_allowed: Callable[[Sequence[str] | None], None],
    refuse_undeclared_world_seed: Callable[[EvalTemplate], None],
    refuse_undeliverable_template: Callable[[EvalTemplate], None],
) -> EvalTemplate:
    """Every refusal a template meets before it is first written, in :func:`create_template`'s order.

    **The one list of create-time gates.** :func:`create_template` and the definition seeder
    (:func:`~threetears.evals.run.definition_seed.seed_eval_definitions`) both admit a new template
    through here, so a gate added to authoring reaches a host's seed corpus by construction rather
    than by someone remembering to mirror it. What this does not ask is the store: the name's
    uniqueness is the caller's question, because the two callers answer it differently — authoring
    refuses a taken name, a seed reads it as an occupied slot.

    Args:
        template: The constructed template, in its scope.
        profile: The host whose kinds, world, tools and action schemas the template is held to.
        require_known_tools_allowed: The host's tool-catalog check, as for :func:`create_template`.
        refuse_undeclared_world_seed: The host's seed walk, as for :func:`create_template`.
        refuse_undeliverable_template: The host's kind-capability check, as for :func:`create_template`.

    Returns:
        The template with its ``kind_spec`` as the kind's spec model resolves it — every field,
        defaults included — which is the shape it is stored in.

    Raises:
        ValidationFailedError: Any refusal :func:`create_template` documents other than the name's.
    """
    template = template.model_copy(update={"kind_spec": freeze(validated_kind_spec(template, profile=profile))})
    require_known_tools_allowed(template.tools_allowed)
    refuse_unsupplied_world(template, profile=profile)
    refuse_non_discriminating_checks(template, profile=profile)
    refuse_undeclared_world_seed(template)
    refuse_undeliverable_template(template)
    return template


def refuse_stale_presumptions(template: EvalTemplate, *, profile: HostProfile) -> None:
    """Refuse a template presuming world state this host's registry no longer declares.

    **Called where a template is USED, not where one is listed**, and the distinction is the
    whole placement. Resolving a removed dimension to nothing would leave a template quietly
    presuming less than it says, which is the defect the world contract exists to catch — but
    a refusal reaching every enumeration is worse than the defect: the catalogue read is how
    an operator finds the offending template, one of its callers seeds definitions at web
    startup, and a listing that stops answering takes the recovery down with the problem. A
    list is a catalogue; a get and a launch are uses.

    Args:
        template: The template about to be used.
        profile: The host whose world the presumptions resolve against.

    Raises:
        ValidationFailedError: A precondition names a path no declared dimension covers.
            Translated here rather than left as the resolution's ``ValueError`` — an untranslated
            one reaches an operator as a 500 with no message, which is the opposite of failing
            loudly.
    """
    if not template.preconditions:
        return
    try:
        resolve_preconditions(template, profile.world)
    except ValueError as e:
        raise ValidationFailedError(str(e)) from e


def refuse_unsupplied_world(
    template: EvalTemplate, *, profile: HostProfile, authored: Container[str] | None = None
) -> None:
    """Refuse a template whose expressions name world state this host cannot supply.

    The authoring-time half of the world gate, and the only moment it can be asked at host
    level: a template is written with no subject in hand, so *does THIS subject reach it* is
    unanswerable here and is asked again at run assembly. What IS answerable is whether any
    run of this host could put a subject in the state the template describes.

    **Two questions, not one, because the two halves of a template ask different things.**

    * A **precondition** presumes the world was SET before the subject started, so a declared
      dimension is not enough — one no run can seed is real, disclosed, and still not
      something a scenario may presume it put in place. That is the founding incident: a probe
      written for a state nobody could instantiate, run against whatever the world happened to
      hold, scoring the subject on a situation it was never placed in.
    * A **goal check** reads whatever the world ended up holding, so reading back a dimension
      no run controls is ordinary and correct. Only the vocabulary is checked. Nothing checked
      it before: an unresolvable path evaluates to ``Missing``, comparisons against ``Missing``
      are ``False``, and the typo scores the subject down while reporting as a failed check —
      the same incident in the postcondition half, which is why one registry closes both.

    **A precondition, never an area** (R10). Each defect names the path that could not be
    supplied and, for a precondition, the prose it presumed — because the remedy differs per
    expression and "this template is unevaluable" would have an author abandon a probe another
    precondition could have carried. All defects are reported at once, so an author fixing one
    sees the rest.

    A host that instantiates no world at all refuses no world path here: ``inapplicable`` is not
    ``uncovered``, and a gate over an absent registry that refused every expression would make
    the contract's adoption cost its own refusal. A text comparison over a ``calls()`` parameter
    is the exception, refused unless the action's schema closes the value, because nothing but
    that schema can say a parameter is not text the model wrote.

    **What fires is asked too.** A goal check's ``fired("<dimension>")`` or ``fired_armed("<dimension>")`` must name a triggered dimension
    the host declares, and a seed scheduling ambient perturbation needs a world with a
    ``perturb_ambient`` handle — each is a template asking for a world event no run of this host could
    produce, and a check over one scores False on every trial.

    Args:
        template: The template being written.
        profile: The host whose world, tools and action schemas the expressions are held to.
        authored: The field names this write AUTHORS, or None to check every field. An update
            merges over the stored shape, so checking unconditionally would make a template
            whose world the registry has since moved uneditable in all its other fields —
            while still refusing to let a bad expression be WRITTEN, which is the point. The
            same authoring-scoped rule ``rubric`` and ``tools_allowed`` carry above.

    Raises:
        ValidationFailedError: An expression names state this host cannot supply, or a goal
            check the language cannot read. The latter is refused here for the reason
            :class:`~threetears.evals.schema.models.Precondition` refuses its own at parse: an
            expression no run can evaluate is an apparatus failure discovered after the
            spend, attributed to whatever the run was doing. Also a ``fired()`` name that is not a
            triggered dimension, or ambient perturbation on a world with no handle for it. Also a goal check that
            string-matches a field the host's world marks as model prose: code checks
            structure, never prose, so that text is a rubric dimension's to judge.
    """
    defects: list[str] = []
    prose_matches: list[str] = []
    undefined: list[str] = []
    unfireable: list[str] = []
    if (authored is None or "world_seed" in authored) and template.world_seed.ambient_perturbation_turns:
        world = profile.world
        if world is None or world.perturb_ambient is None:
            unfireable.append(
                "its world_seed schedules ambient perturbation before turn(s) "
                f"{template.world_seed.ambient_perturbation_turns!r}, and this host's world declares no "
                "perturb_ambient handle to move undeclared state with"
            )
    if authored is None or "preconditions" in authored:
        for precondition in template.preconditions:
            for path in precondition.presumed_paths:
                verdict = profile.presumable(path)
                if verdict.state == "uncovered":
                    defects.append(f"{verdict.reason} (presuming {precondition.presumes!r})")
    if authored is None or "goal_state_checks" in authored:
        for expression in template.goal_state_checks:
            try:
                paths = extract_paths(expression).world
            except DSLError as malformed:
                defects.append(f"goal check {expression!r} does not parse: {malformed}")
                continue
            defects.extend(
                verdict.reason
                for verdict in (profile.addressable(path) for path in paths)
                if verdict.state == "uncovered"
            )
            prose_matches.extend(
                f"goal check {expression!r} matches text over model prose ({match.source})"
                for match in world_prose_matches(expression, profile.world)
            )
            prose_matches.extend(
                f"goal check {expression!r} matches text over a call parameter ({match.source}: {reason})"
                for match, reason in call_parameter_matches(expression, profile.action_parameters)
            )
            undefined.extend(
                f"goal check {expression!r}: {reason}"
                for reason in undefined_call_references(expression, profile.tool_actions, profile.action_parameters)
            )
            unfireable.extend(
                f"goal check {expression!r}: {reason}"
                for reason in undefined_fire_references(expression, profile.world)
            )
    if unfireable:
        raise ValidationFailedError(
            f"template {template.name!r} names world events this host cannot produce: "
            + "; ".join(unfireable)
            + " — a check on a dimension that can never fire scores False on every trial, which reads as the"
            " candidate's failure; name a triggered dimension the host declares, or drop the schedule"
        )
    if undefined:
        raise ValidationFailedError(
            f"template {template.name!r} names calls this host does not define: "
            + "; ".join(undefined)
            + " — a check over a call that cannot happen scores False on every trial, which reads as the"
            " candidate's failure; correct the name"
        )
    if prose_matches:
        raise ValidationFailedError(
            f"template {template.name!r} checks what a model wrote with a string match: "
            + "; ".join(prose_matches)
            + " — code checks structure, never prose; judge the text with a rubric dimension instead,"
            " test membership in a structured list, or test whether a parameter is set and its length"
        )
    if defects:
        raise ValidationFailedError(
            f"template {template.name!r} names world state this host cannot supply: "
            + "; ".join(defects)
            # One tail for several causes, so it names the remedies rather than picking one:
            # `presumable` deliberately returns different reasons BECAUSE the remedies differ,
            # and stapling "register the dimension" onto all of them tells an author to
            # register something already registered — which is what a witnessed dimension is.
            + " — a path naming no dimension wants the dimension declared or the expression"
            " corrected; a dimension that cannot be presumed wants the scenario rewritten"
            " rather than seeded, or the expression moved to a goal check, which reads what"
            " the world ended up holding rather than claiming a run put it there"
        )


def get_template(host: EvalHost, template_id: str, scope_id: str) -> EvalTemplate:
    """Load a template by id within a scope.

    Args:
        host: The host: where the template is read, and the world its presumptions resolve in.
        template_id: The template.
        scope_id: The scope it lives in.

    Returns:
        The template.

    Raises:
        NotFoundError: No template with that id in the scope.
        ValidationFailedError: It presumes world state this host no longer declares. An
            update does NOT go through here, deliberately, so a template refused for that
            reason stays editable into a shape that resolves.
    """
    template = host.storage.load_template(template_id, scope_id)
    if template is None:
        raise NotFoundError("template", template_id)
    refuse_stale_presumptions(template, profile=host.profile)
    return template


def list_templates(
    storage: DefinitionStore,
    scope_id: str,
    *,
    archived: bool = False,
    required_tool: str | None = None,
    universal: bool | None = None,
) -> list[EvalTemplate]:
    """List templates in a scope.

    ``archived`` defaults to ``False`` (active templates only). ``universal=None`` returns both
    universal (boundary-battery) and subject-scoped templates; set ``True``/``False`` to filter.
    """
    return storage.query_templates(scope_id, archived=archived, required_tool=required_tool, universal=universal)


def update_template(
    host: EvalHost,
    template_id: str,
    scope_id: str,
    fields: dict[str, Any],
    *,
    require_known_tools_allowed: Callable[[Sequence[str] | None], None],
    refuse_undeclared_world_seed: Callable[[EvalTemplate], None],
    refuse_undeliverable_template: Callable[[EvalTemplate], None],
) -> EvalTemplate:
    """Apply a partial update to a template and persist it.

    Merges ``fields`` over the existing template, re-validates the merged
    shape, stamps ``updated_at``, and saves. Server-owned identity/lifecycle
    fields in ``fields`` are ignored.

    Args:
        host: The host: where templates are read and written, and the world an authored
            expression is held to.
        template_id: Template to update.
        scope_id: The scope it lives in.
        fields: Partial field map to overlay onto the existing template.
        require_known_tools_allowed: The host's tool-catalog check, as for
            :func:`create_template`; asked only when this update writes ``tools_allowed``.
        refuse_undeclared_world_seed: The host's seed walk, as for :func:`create_template`;
            asked only when this update writes ``world_seed``.
        refuse_undeliverable_template: The host's kind-capability check, as for
            :func:`create_template`; asked of the merged template unless the update only sets
            ``archived``.

    Returns:
        The persisted, updated :class:`~threetears.evals.schema.models.EvalTemplate`.

    Raises:
        NotFoundError: No template with that id.
        ValidationFailedError: ``fields`` is empty, the merged shape is
            invalid, an update writing ``kind_spec`` or ``candidate_kind`` leaves a spec the
            kind's spec model refuses, or an authored ``rubric`` / ``tools_allowed`` /
            ``preconditions`` / ``goal_state_checks`` / ``goal_check_controls`` /
            ``world_seed`` fails the authoring guard that field carries.
    """
    existing = host.storage.load_template(template_id, scope_id)
    if existing is None:
        raise NotFoundError("template", template_id)
    if not fields:
        raise ValidationFailedError("update_template requires at least one field to change")
    reject_unknown_authoring_fields("update_template", fields, EvalTemplate, TEMPLATE_SERVER_FIELDS)

    merged = existing.to_dict()
    for key, value in fields.items():
        if key in TEMPLATE_SERVER_FIELDS:
            continue
        merged[key] = value
    merged["updated_at"] = utc_now_iso()
    try:
        template = EvalTemplate(**merged)
    except ValidationError as e:
        raise ValidationFailedError(f"invalid template update: {e}") from e

    # Scoped to the writes that touch it, like the guards below: the spec itself, or the kind whose
    # model reads it. A stored template whose spec the model has since outgrown stays editable in
    # its other fields — and into a spec that validates, through this one — while its launch
    # refuses it.
    if {"kind_spec", "candidate_kind"} & fields.keys():
        template = template.model_copy(
            update={"kind_spec": freeze(validated_kind_spec(template, profile=host.profile))}
        )
    # Same authoring-scoped rule, and for the same reason: check the names this
    # update WRITES, so a stored template whose allow-list names a tool type
    # since removed from the catalog stays editable in its other fields.
    if "tools_allowed" in fields:
        require_known_tools_allowed(template.tools_allowed)
    # Same authoring-scoped rule again, one field pair over — checked against the merged
    # template so an update that authors a precondition is resolved as it will be stored,
    # but only for the halves this write actually authors.
    refuse_unsupplied_world(template, profile=host.profile, authored=fields.keys())
    # Scoped the same way, to the writes that touch a check's proof: the checks, their controls,
    # and the seed every control is laid over. A template written past authoring without controls stays editable
    # in its other fields; the first write that authors its checks has to prove them.
    refuse_non_discriminating_checks(template, profile=host.profile, authored=fields.keys())
    # And once more for the world seed, only when this update WRITES it: a stored template
    # whose seed the registry refuses — a shape some stored templates carry — stays editable
    # in its other fields and, through this field, into one that resolves.
    if "world_seed" in fields:
        refuse_undeclared_world_seed(template)
    # Unscoped, unlike its neighbours above, and the asymmetry is the point: those check the
    # names an update WRITES so a stored template stays editable around a name the catalog has
    # since dropped, whereas this one is checked against the MERGED template because the
    # contradiction it catches is between two fields rather than inside one. An update that
    # only sets `candidate_kind='classifier'` contradicts a rubric it does not touch, and an
    # update that only adds a rubric contradicts a kind it does not touch; scoping to the
    # written keys would miss whichever half arrived first. It self-selects on the kind, so
    # this is a no-op for a conversational template.
    #
    # **ARCHIVING is exempt, and without the exemption this guard is a trap.** A single-shot
    # template written past authoring can fail it, and then every update fails —
    # `{"archived": True}` included. There is no delete or archive action beside this path,
    # so a template that cannot be updated cannot be retired either, and the operator's only
    # remaining move is to leave a broken template in the catalogue forever. Retiring
    # something is the one edit that must never depend on it being well formed; the exemption
    # is narrow on purpose — an update whose ONLY field is the archive flag — so it cannot be
    # used to smuggle a contradiction past the guard alongside it.
    #
    # It is symmetric, and that is deliberate rather than overlooked: un-archiving a template
    # the guard would refuse puts it back in the catalogue still broken, and the launch path
    # guards independently (both single-shot launches call this), so what
    # comes back is a template that refuses to RUN rather than one that runs wrong. Refusing
    # the un-archive instead would only move the trap one step, to a template that can be
    # retired and never brought back.
    if set(fields) != {"archived"}:
        refuse_undeliverable_template(template)

    host.storage.save_template(template)
    return template


#: Identity/lifecycle fields the rubric-dim family owns — never taken from caller input.
RUBRIC_DIM_SERVER_FIELDS: tuple[str, ...] = (
    "id",
    "doc_type",
    "schema_version",
    "created_at",
    "updated_at",
    "scope_id",
)


def create_rubric_dim(storage: DefinitionStore, definition: dict[str, Any], *, scope_id: str) -> CatalogRubricDim:
    """Create and persist a catalog rubric dim from an authoring definition.

    A name the model does not declare is refused; server-owned identity/lifecycle fields in ``definition`` are ignored —
    a fresh ``id`` and ``created_at`` / ``updated_at`` are always assigned.
    ``key`` is unique among active records within a scope: a second
    create reusing an active key raises :class:`ConflictError` (re-authoring
    archives the old record first; ``load_active_rubric_dim`` resolves the
    latest non-archived record for a key).

    Args:
        storage: Where catalog rubric dims are read and written.
        definition: Catalog-dim fields (``key`` + ``dim`` required; the rest
            optional per :class:`~threetears.evals.schema.models.CatalogRubricDim`).
        scope_id: The scope the dim lives in.

    Returns:
        The persisted :class:`~threetears.evals.schema.models.CatalogRubricDim`.

    Raises:
        ValidationFailedError: ``definition`` fails catalog-dim validation.
        ConflictError: an active dim with the same ``key`` already exists.
    """
    reject_unknown_authoring_fields("create_rubric_dim", definition, CatalogRubricDim, RUBRIC_DIM_SERVER_FIELDS)
    clean = {k: v for k, v in definition.items() if k not in RUBRIC_DIM_SERVER_FIELDS}
    try:
        dim = CatalogRubricDim(**clean, scope_id=scope_id)
    except ValidationError as e:
        raise ValidationFailedError(f"invalid rubric dim definition: {e}") from e

    if storage.load_active_rubric_dim(dim.key, scope_id) is not None:
        raise ConflictError(f"an active rubric dim with key '{dim.key}' already exists")

    storage.save_rubric_dim(dim)
    return dim


def get_rubric_dim(storage: DefinitionStore, dim_id: str, scope_id: str) -> CatalogRubricDim:
    """Load a catalog rubric dim by id within a scope.

    Raises:
        NotFoundError: No rubric dim with that id in the scope.
    """
    dim = storage.load_rubric_dim(dim_id, scope_id)
    if dim is None:
        raise NotFoundError("rubric dim", dim_id)
    return dim


def list_rubric_dims(
    storage: DefinitionStore,
    scope_id: str,
    *,
    axis: str | None = None,
    universal: bool | None = None,
    archived: bool = False,
) -> list[CatalogRubricDim]:
    """List catalog rubric dims in a scope.

    ``archived`` defaults to ``False`` (active dims only). ``axis`` / ``universal`` are optional
    filters forwarded to storage.
    """
    return storage.query_rubric_dims(scope_id, axis=axis, universal=universal, archived=archived)


def update_rubric_dim(storage: DefinitionStore, dim_id: str, scope_id: str, fields: dict[str, Any]) -> CatalogRubricDim:
    """Apply a partial update to a catalog rubric dim and persist it.

    Merges ``fields`` over the existing dim, re-validates the merged shape,
    stamps ``updated_at``, and saves. Server-owned identity/lifecycle fields
    in ``fields`` are ignored.

    Args:
        storage: Where catalog rubric dims are read and written.
        dim_id: Catalog dim to update.
        scope_id: The scope it lives in.
        fields: Partial field map to overlay onto the existing dim.

    Returns:
        The persisted, updated :class:`~threetears.evals.schema.models.CatalogRubricDim`.

    Raises:
        NotFoundError: No rubric dim with that id.
        ValidationFailedError: ``fields`` is empty or the merged shape is invalid.
    """
    existing = storage.load_rubric_dim(dim_id, scope_id)
    if existing is None:
        raise NotFoundError("rubric dim", dim_id)
    if not fields:
        raise ValidationFailedError("update_rubric_dim requires at least one field to change")
    reject_unknown_authoring_fields("update_rubric_dim", fields, CatalogRubricDim, RUBRIC_DIM_SERVER_FIELDS)

    merged = existing.to_dict()
    for key, value in fields.items():
        if key in RUBRIC_DIM_SERVER_FIELDS:
            continue
        merged[key] = value
    merged["updated_at"] = utc_now_iso()
    try:
        dim = CatalogRubricDim(**merged)
    except ValidationError as e:
        raise ValidationFailedError(f"invalid rubric dim update: {e}") from e

    storage.save_rubric_dim(dim)
    return dim


def delete_rubric_dim(storage: DefinitionStore, dim_id: str, scope_id: str, *, confirm: str | None = None) -> None:
    """Delete a catalog rubric dim by id, after an id-echo confirmation, and retire its key from the seed.

    **Gated on an id echo, like every other destructive eval delete** — see
    :func:`threetears.evals.run.curation.require_delete_confirmation`. The gate is
    here so both surfaces (REST ``DELETE /rubric-dims/{dim_id}``
    and MCP ``rubric_dim_delete``) inherit one contract.

    **A delete sticks, seeded dim or not.** Seeding writes any corpus dim whose key no record in the
    scope carries (:func:`~threetears.evals.run.definition_seed.seed_eval_definitions`), so a delete
    alone would hand a seeded dim's key back to the next boot, which would write the dim again —
    while archiving it, the reversible choice, kept it retired. So the delete first writes a
    :class:`~threetears.evals.schema.models.RubricDimTombstone` for the dim's key, and the seeder
    never writes a tombstoned key back. Archive and delete now both retire a seeded dim; what differs
    is that archive keeps the record and can be undone, and delete destroys it. To have a deleted key
    back, author it again (``create_rubric_dim``), which the tombstone does not block.

    **Nothing else goes with it, and nothing is orphaned.** Storage removes
    exactly one dim document (:meth:`~threetears.evals.kernel.storage.EvalStorage.delete_rubric_dim`
    is a single ``delete`` by id), and no stored record points at a catalog
    dim's ``id``: :attr:`~threetears.evals.schema.models.JudgeConfig.rubric_dim_id`
    binds **by name**, ``EvalTemplate.rubric`` embeds name-keyed
    :class:`~threetears.evals.schema.models.RubricDim` value objects rather than
    references, and the catalog's own DECISION lock-in is that it does not
    rebind the judge's name-keying. The catalog is read as a *feed* (the rubric
    proposer's catalog feed, non-archived only), which copies definitions into a draft
    rather than pointing at them.

    What the delete destroys is therefore the record itself: the ``dim.description`` and
    ``dim.scoring_guide`` prose a judge is meant to score against. For a hand-authored dim nothing
    restores it. For a seeded dim the host's corpus still holds the seeded version, but the operator's
    edits to it are gone, and the seed will not write it back. The gate is unconditional because a dim
    nothing has drawn from yet is the freshly authored one least likely to have a second copy.

    Args:
        storage: Where catalog rubric dims are read, deleted and tombstoned.
        dim_id: Catalog rubric dim to destroy.
        scope_id: The scope it lives in.
        confirm: Must echo ``dim_id``; see
            :func:`threetears.evals.run.curation.require_delete_confirmation`.

    Raises:
        NotFoundError: No rubric dim with that id in the scope.
        ValidationFailedError: ``confirm`` does not echo ``dim_id``.
        StorageError: The tombstone or the delete failed to write. A failed tombstone leaves the dim
            in place, so no delete happens that a seed could undo.
    """
    dim = storage.load_rubric_dim(dim_id, scope_id)
    if dim is None:
        raise NotFoundError("rubric dim", dim_id)
    require_delete_confirmation(
        "rubric dim",
        dim_id,
        confirm,
        # No `cascade`: the delete takes no other definition, and naming a
        # phantom one would misstate the weight of the call.
        alternative=(
            'archive it instead (`rubric_dim_update` with `{"archived": true}`) to drop it from the '
            "proposer's catalog feed without destroying the definition and scoring guide it carries"
        ),
    )
    # The tombstone first: a delete whose tombstone failed would be one the next seed undoes.
    storage.save_rubric_dim_tombstone(RubricDimTombstone(scope_id=scope_id, key=dim.key, deleted_dim_id=dim_id))
    if not storage.delete_rubric_dim(dim_id, scope_id):
        raise StorageError(f"failed to delete rubric dim '{dim_id}'")
    log.warning("eval.delete_rubric_dim dim=%s key=%s", dim_id, dim.key)


#: Identity/lifecycle fields the judge-config family owns — never taken from caller input.
JUDGE_CONFIG_SERVER_FIELDS: tuple[str, ...] = (
    "id",
    "doc_type",
    "schema_version",
    "created_at",
    "scope_id",
)


def create_judge_config(storage: DefinitionStore, definition: dict[str, Any], *, scope_id: str) -> JudgeConfig:
    """Create and persist a judge config from an authoring definition.

    A name the model does not declare is refused; server-owned identity/lifecycle fields in ``definition`` are ignored —
    a fresh ``id`` and ``created_at`` are always assigned. Only one active
    (non-archived) config may exist per ``rubric_dim_id``: a create against
    a dim that already has one raises :class:`ConflictError`, forcing the
    operator through :func:`update_judge_config` (archive-and-recreate) to
    re-author rather than silently shadowing the live config
    (``load_active_judge_config`` resolves the latest non-archived record).

    Args:
        storage: Where judge configs are read and written.
        definition: Judge-config fields (``name`` + ``rubric_dim_id`` +
            ``prompt_template`` required; the rest optional per
            :class:`~threetears.evals.schema.models.JudgeConfig`).
        scope_id: The scope the config lives in.

    Returns:
        The persisted :class:`~threetears.evals.schema.models.JudgeConfig`.

    Raises:
        ValidationFailedError: ``definition`` fails judge-config validation.
        ConflictError: an active config already exists for that ``rubric_dim_id``.
    """
    reject_unknown_authoring_fields("create_judge_config", definition, JudgeConfig, JUDGE_CONFIG_SERVER_FIELDS)
    clean = {k: v for k, v in definition.items() if k not in JUDGE_CONFIG_SERVER_FIELDS}
    try:
        config = JudgeConfig(**clean, scope_id=scope_id)
    except ValidationError as e:
        raise ValidationFailedError(f"invalid judge config definition: {e}") from e

    if storage.load_active_judge_config(config.rubric_dim_id, scope_id) is not None:
        raise ConflictError(f"an active judge config already exists for rubric dim '{config.rubric_dim_id}'")

    storage.save_judge_config(config)
    return config


def get_judge_config(storage: DefinitionStore, config_id: str, scope_id: str) -> JudgeConfig:
    """Load a judge config by id within a scope.

    Raises:
        NotFoundError: No judge config with that id in the scope.
    """
    config = storage.load_judge_config(config_id, scope_id)
    if config is None:
        raise NotFoundError("judge config", config_id)
    return config


def list_judge_configs(
    storage: DefinitionStore,
    scope_id: str,
    *,
    rubric_dim_id: str | None = None,
    archived: bool = False,
) -> list[JudgeConfig]:
    """List judge configs in a scope.

    ``archived`` defaults to ``False`` (active configs only). ``rubric_dim_id``
    is an optional filter forwarded to storage.
    """
    return storage.query_judge_configs(
        scope_id,
        rubric_dim_id=rubric_dim_id,
        archived=archived,
    )


def update_judge_config(storage: DefinitionStore, config_id: str, scope_id: str, fields: dict[str, Any]) -> JudgeConfig:
    """Re-author a judge config via archive-and-recreate (immutable versioning).

    :class:`~threetears.evals.schema.models.JudgeConfig` has no ``updated_at`` and is
    immutable-versioned (``load_active_judge_config`` = latest non-archived
    record per ``rubric_dim_id``), so there is no in-place edit. An update
    therefore: (1) builds a NEW version — the existing fields overlaid with
    ``fields`` (server-owned identity/lifecycle fields ignored), with a fresh
    ``id`` and ``created_at`` and ``archived=False``, keeping the same
    ``rubric_dim_id`` — validates it and persists it, then (2) archives the
    existing record (sets ``archived=True`` and saves it, preserving it for
    historical-run interpretation), and returns the new version, which becomes
    the active config for that dim. New before old, so a failed write leaves
    the dim with its old config active rather than with none.

    Args:
        storage: Where judge configs are read and written.
        config_id: Judge config to re-author.
        scope_id: The scope it lives in; the new version lives there too.
        fields: Partial field map to overlay onto the existing config.

    Returns:
        The persisted new-version :class:`~threetears.evals.schema.models.JudgeConfig`.

    Raises:
        NotFoundError: No judge config with that id.
        ValidationFailedError: ``fields`` is empty or the merged shape is invalid.
        StorageError: A write failed; if it was the new version's, nothing changed.
    """
    existing = storage.load_judge_config(config_id, scope_id)
    if existing is None:
        raise NotFoundError("judge config", config_id)
    if not fields:
        raise ValidationFailedError("update_judge_config requires at least one field to change")
    # BEFORE the archive-and-recreate below, which is the whole reason this one matters:
    # a refusal reached after the archive would leave the dim's active config replaced by
    # a new id differing in nothing.
    reject_unknown_authoring_fields("update_judge_config", fields, JudgeConfig, JUDGE_CONFIG_SERVER_FIELDS)

    # Build the new version from the existing fields overlaid with the
    # caller's changes, then drop server-owned identity/lifecycle fields so
    # the new record gets a fresh id/created_at and stays non-archived.
    merged = existing.to_dict()
    for key, value in fields.items():
        if key in JUDGE_CONFIG_SERVER_FIELDS:
            continue
        merged[key] = value
    for key in JUDGE_CONFIG_SERVER_FIELDS:
        merged.pop(key, None)
    merged["rubric_dim_id"] = existing.rubric_dim_id
    merged["scope_id"] = scope_id
    merged["archived"] = False
    try:
        new_config = JudgeConfig(**merged)
    except ValidationError as e:
        raise ValidationFailedError(f"invalid judge config update: {e}") from e

    # Archive the old record only after the new one is stored, so neither a bad
    # update nor a failed write can leave the dim with no active config.
    storage.save_judge_config(new_config)
    existing.archived = True
    storage.save_judge_config(existing)
    return new_config


def delete_judge_config(storage: DefinitionStore, config_id: str, scope_id: str, *, confirm: str | None = None) -> None:
    """Delete a judge config by id, after an id-echo confirmation.

    **Gated on an id echo, like every other destructive eval delete** — see
    :func:`threetears.evals.run.curation.require_delete_confirmation`. The gate is
    here so both surfaces (REST ``DELETE /judge-configs/{config_id}``
    and MCP ``judge_config_delete``) inherit one contract.

    **Storage cascades to nothing, which is exactly the problem.** It removes
    one document; the pointers at it are left dangling, in two places that
    both read as attribution rather than as a foreign key:

    - :attr:`~threetears.evals.schema.models.EvalResult.judge_config_ids` — every result
      this config scored records its id, and that map is what the per-dim
      judge attribution renders from.
    - :attr:`~threetears.evals.schema.models.EvalRun.judge_config_ids` /
      ``judge_config_provenance`` — the set a run committed to at launch, and
      the set the launch's judge-config selection re-loads by id when an
      A/B control arm is re-run against a *superseded* version. That path
      accepts an archived config deliberately; a deleted one refuses.

    **A delete sticks, seeded config or not.** The seed writes any corpus config whose slot
    (``rubric_dim_id``, ``name``) no record in the scope carries, so the delete first writes a
    :class:`~threetears.evals.schema.models.JudgeConfigTombstone` for the slot, and the seeder never
    writes a tombstoned slot back — the mechanism :func:`delete_rubric_dim` uses. Authoring a config into
    the slot again is the way back; the tombstone does not block it.

    **Not refused outright when in use, and that is a cost judgement, not a
    preference.** A referent count is not computable here: ``judge_config_ids``
    lives on :class:`~threetears.evals.schema.models.EvalResult`, and counting would mean
    a scan of every stored result in the scope with Python-side filtering on a map that
    is not an indexed field — the largest collection in the store, walked on every delete. So the refusal names
    *what kind* of provenance goes rather than how much, and the gate stays
    unconditional the way the other delete gates' do: scaling it to hidden state would make
    the same call refuse or destroy depending on whether anyone had run an
    eval yet.

    Args:
        storage: Where judge configs are read and deleted.
        config_id: Judge config to destroy.
        scope_id: The scope it lives in.
        confirm: Must echo ``config_id``; see
            :func:`threetears.evals.run.curation.require_delete_confirmation`.

    Raises:
        NotFoundError: No judge config with that id in the scope.
        ValidationFailedError: ``confirm`` does not echo ``config_id``.
        StorageError: The tombstone or the delete failed to write. A failed tombstone leaves the config in
            place, so no delete happens that a seed could undo.
    """
    config = storage.load_judge_config(config_id, scope_id)
    if config is None:
        raise NotFoundError("judge config", config_id)
    require_delete_confirmation(
        "judge config",
        config_id,
        confirm,
        # A phrase, not a count — see the cost judgement above for why no number
        # is available to put here.
        cascade="and the judge attribution of every stored result and run that names it",
        # `JudgeConfig` has an `archived` field, so the default "archive it
        # instead" is nearly right — but there is no set-archived call for one:
        # `update_judge_config` is the archive path, and it archives this record
        # as a side effect of superseding it. Say that rather than pointing at a
        # verb the surface does not have.
        alternative=(
            "supersede it with `judge_config_update`, which archives this record and leaves it resolvable for the runs it scored"
        ),
    )
    # The tombstone first, as `delete_rubric_dim` writes its own: a delete whose tombstone failed would be one the
    # next seed undoes, writing the seeded config back into the slot the delete emptied.
    storage.save_judge_config_tombstone(
        JudgeConfigTombstone(
            scope_id=scope_id, rubric_dim_id=config.rubric_dim_id, name=config.name, deleted_config_id=config_id
        )
    )
    if not storage.delete_judge_config(config_id, scope_id):
        raise StorageError(f"failed to delete judge config '{config_id}'")
    log.warning("eval.delete_judge_config config=%s", config_id)


__all__ = [
    "JUDGE_CONFIG_SERVER_FIELDS",
    "RUBRIC_DIM_SERVER_FIELDS",
    "TEMPLATE_SERVER_FIELDS",
    "admit_template",
    "create_judge_config",
    "create_rubric_dim",
    "create_template",
    "delete_judge_config",
    "delete_rubric_dim",
    "get_judge_config",
    "get_rubric_dim",
    "get_template",
    "list_judge_configs",
    "list_rubric_dims",
    "list_templates",
    "refuse_stale_presumptions",
    "refuse_unsupplied_world",
    "update_judge_config",
    "update_rubric_dim",
    "update_template",
    "validated_kind_spec",
]
