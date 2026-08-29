"""the vocabulary of things an application can be authorized to do.

:attr:`~threetears.agent.acl.types.Role.permissions` is free text. a role
may name any resource type and any action, and both halves are compared
by string equality at evaluation time
(:mod:`threetears.agent.acl.evaluator` resolves
``role.actions_for(namespace.namespace_type)`` and then asks whether the
requested action is in the result). a resource type nothing serves, or an
action nothing checks, therefore does not fail — it evaluates to an empty
set and the role grants silence. that failure mode is invisible: the role
exists, it looks granted, and it does nothing.

a catalog is an application's statement of which pairs mean something, so
a role write path can refuse the rest and a role builder can offer only
what is real.

**the resource type is a namespace type, not a free label.** this is the
constraint everything else here follows from. evaluation is
namespace-centric end to end:
:func:`~threetears.agent.acl.authorize.authorize` resolves a canonical
namespace name to a row, :meth:`Role.actions_for`'s own parameter
documentation reads "namespace type to look up", and
:class:`~threetears.agent.acl.types.RoleAssignment` scopes to a namespace
id or to a namespace type within a customer. so a declared resource type
that is not a value ``namespaces.namespace_type`` admits has
nothing to bind to, and a role naming it can never be reached.

**the action is the whole canonical action string**, spelled exactly as a
caller passes it to :func:`~threetears.agent.acl.authorize.authorize` —
``memory.read``, ``conversation.write``, ``tool.call``. the platform's
established convention prefixes the action with its resource
(:mod:`threetears.agent.memory.authorize` declares ``memory.read`` and
evaluates it "against the ``memory`` bucket of the caller's roles"), so
the bucket key and the action prefix agree by convention rather than by
construction. this module does not enforce that agreement: it is a
convention with real exceptions already shipped
(:data:`~threetears.agent.acl.evaluator.READ_FILE_MATCHING_PREFIX` names
``read_file_matching:`` in the ``workspace`` bucket), and a catalog that
refused those would refuse the platform's own roles.

**why the declaring application is a field rather than a prefix on the
resource type.** an application-prefixed wire form —
``survey/report`` — would have to be the bucket key to have any effect,
and the bucket key is looked up by
``role.actions_for(namespace.namespace_type)``. that would require a
``survey/report``-type namespace row per application per resource, which
the platform's ``namespaces_namespace_type_ck`` CHECK does not admit and
which no product can widen from its own repo. so the application rides
the descriptor as data — it is who to attribute an entry to, who to ask
before retiring it, and what a role builder groups by — while the
resource type stays the bare namespace type the evaluator already looks
up. two applications claiming one resource type is consequently refused
rather than disambiguated: see :class:`PermissionCatalog`.

**the wildcard is skipped on purpose.**
:data:`~threetears.agent.acl.types.WILDCARD_RESOURCE_TYPE` names no
resource type, so there is no catalog entry it could be checked against,
and ``Reader``-shaped roles built out of it already ship.
:func:`validate_permissions` therefore skips that bucket by an explicit
branch rather than by failing to find an entry for ``"*"`` — the two are
indistinguishable in behaviour today and would stop being so the moment
somebody declared an entry named ``"*"``, which
:class:`ResourceTypeDescriptor` separately refuses.

**these are pydantic models rather than the frozen dataclasses in**
:mod:`threetears.agent.acl.types`. the types there are evaluator values,
built in-process and never parsed; a catalog is operator-authored data
that arrives from a YAML file or a JSONB column, and the invariants below
are only worth stating if they run on the parse rather than on the one
construction path that happens to be in Python. that is the same argument
:mod:`threetears.iam.connection_types` makes for the same shape.

nothing in this module participates in evaluation. it is a write-path and
authoring-path concern; a role that evaluated one way before a catalog
existed evaluates the same way after.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from threetears.agent.acl.types import WILDCARD_RESOURCE_TYPE

__all__ = [
    "ActionDescriptor",
    "CatalogViolation",
    "CatalogViolationKind",
    "PermissionCatalog",
    "ResourceTypeDescriptor",
    "UndeclaredPermission",
    "enforce_declared_permissions",
    "validate_permissions",
]


#: separator between a parameterized action's stem and its argument, as
#: :data:`~threetears.agent.acl.evaluator.READ_FILE_MATCHING_PREFIX`
#: already spells it (``read_file_matching:**/*.yaml``). a stem ends with
#: this character so a prefix match cannot run past the boundary and
#: admit an unrelated action that merely starts with the same letters.
ACTION_ARGUMENT_SEPARATOR = ":"


class CatalogViolationKind(StrEnum):
    """which half of a ``(resource_type, action)`` pair was undeclared.

    :cvar UNDECLARED_RESOURCE_TYPE: no catalog entry exists for the
        bucket key at all; reported once per bucket rather than once per
        action, because every action under it fails for the same reason
        and an operator fixing a mistyped key wants one message
    :cvar UNDECLARED_ACTION: an entry exists for the bucket key, but it
        does not declare this action
    """

    UNDECLARED_RESOURCE_TYPE = "undeclared_resource_type"
    UNDECLARED_ACTION = "undeclared_action"


class CatalogViolation(BaseModel):
    """one reason a permissions map failed catalog validation.

    returned rather than raised by :func:`validate_permissions` so a role
    builder can render every problem at once instead of one per round
    trip. :func:`enforce_declared_permissions` is the raising form for
    callers with nothing to render.

    :param kind: which half of the pair was undeclared
    :ptype kind: CatalogViolationKind
    :param resource_type: bucket key the role named
    :ptype resource_type: str
    :param action: action the role named, or ``None`` when the resource
        type itself was undeclared and no single action is at fault
    :ptype action: str | None
    :param message: operator-facing sentence naming what was undeclared
    :ptype message: str
    """

    model_config = ConfigDict(frozen=True)

    kind: CatalogViolationKind
    resource_type: str
    action: str | None
    message: str


class UndeclaredPermission(ValueError):
    """raised when a permissions map references pairs no catalog declares.

    carries every violation rather than only the first, so a caller that
    catches this can surface the same detail
    :func:`validate_permissions` returns.

    :param violations: every undeclared pair found, in the deterministic
        order :func:`validate_permissions` produces
    :ptype violations: tuple[CatalogViolation, ...]
    """

    def __init__(self, violations: tuple[CatalogViolation, ...]) -> None:
        """build with rendered detail and keep the structured form.

        :param violations: every undeclared pair found
        :ptype violations: tuple[CatalogViolation, ...]
        """
        self.violations = violations
        super().__init__("; ".join(violation.message for violation in violations))


class ActionDescriptor(BaseModel):
    """one action an application declares on one resource type.

    :attr:`name` is the whole canonical action string, spelled as a
    caller passes it to :func:`~threetears.agent.acl.authorize.authorize`
    — the evaluator's final test is ``ctx.action in effective_actions``,
    so anything short of the exact string is an action that never
    matches.

    :attr:`label` exists because a role builder renders these to a tenant
    admin who has never read this module. shipping the display string
    with the declaration is what stops each consuming surface inventing
    its own.

    :param name: canonical action string, or the stem ending in ``:``
        when :attr:`parameterized`
    :ptype name: str
    :param label: operator-facing name, e.g. ``"Open collector"``
    :ptype label: str
    :param description: longer operator-facing prose, optional
    :ptype description: str
    :param parameterized: whether :attr:`name` is a prefix stem that
        matches any argument after the separator, the shape
        :data:`~threetears.agent.acl.evaluator.READ_FILE_MATCHING_PREFIX`
        already ships
    :ptype parameterized: bool
    """

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    description: str = ""
    parameterized: bool = False

    @model_validator(mode="after")
    def _name_is_non_empty_and_unspaced(self) -> ActionDescriptor:
        """an action is compared by string equality against what a caller passes
        to ``authorize``; whitespace in it is a typo that could never match, and
        an empty name is an entry nothing can reference."""
        if not self.name.strip():
            raise ValueError("action declares an empty name")
        if any(character.isspace() for character in self.name):
            raise ValueError(f"action {self.name!r} contains whitespace")
        return self

    @model_validator(mode="after")
    def _label_is_present(self) -> ActionDescriptor:
        """shard 13's role builder renders this; a blank label is a blank row an
        admin has to guess at."""
        if not self.label.strip():
            raise ValueError(f"action {self.name!r} declares an empty label")
        return self

    @model_validator(mode="after")
    def _parameterization_matches_the_name(self) -> ActionDescriptor:
        """the two must agree in both directions.

        a stem without the separator prefix-matches anything starting with the
        same letters, so ``collector.open`` declared parameterized would also
        admit ``collector.opened``. a name carrying the separator but NOT
        declared parameterized is the shape written without saying so, and it
        then matches only the literal argument-free string — silently granting
        nothing, which is the failure this module exists to make impossible.
        """
        has_separator = ACTION_ARGUMENT_SEPARATOR in self.name
        if self.parameterized and not self.name.endswith(ACTION_ARGUMENT_SEPARATOR):
            raise ValueError(
                f"action {self.name!r} is parameterized but does not end with {ACTION_ARGUMENT_SEPARATOR!r}",
            )
        if not self.parameterized and has_separator:
            raise ValueError(
                f"action {self.name!r} carries {ACTION_ARGUMENT_SEPARATOR!r} but is not declared parameterized",
            )
        return self

    def matches(self, action: str) -> bool:
        """whether a role's action string is this declared action.

        exact equality for an ordinary action; prefix match against the
        stem for a parameterized one.

        :param action: action string as it appears in a role's
            permissions bucket
        :ptype action: str
        :return: whether this descriptor declares that action
        :rtype: bool
        """
        if self.parameterized:
            result = action.startswith(self.name)
        else:
            result = action == self.name
        return result


class ResourceTypeDescriptor(BaseModel):
    """one application's statement about one namespace type.

    :attr:`resource_type` is the bare ``namespaces.namespace_type``
    value a role uses as its permissions bucket key — the same string
    :meth:`~threetears.agent.acl.types.Role.actions_for` is called with.
    it is NOT prefixed by :attr:`application`; the module docstring gives
    the reason.

    **whether the namespace type exists is not checked here, and that is
    deliberate.** the authoritative closed set is the
    ``namespaces_namespace_type_ck`` CHECK on the platform's own
    ``namespaces`` table, which lives in the deploying hub rather than in
    this package. the nearest thing on this side,
    :data:`threetears.core.namespaces.PLURAL_PREFIX_BY_NAMESPACE_TYPE`, is
    a partial mirror already known to disagree with it in both directions
    (it carries ``hitl``, which no CHECK admits; it omits ``intention``
    and ``identity``, which ship as role buckets). validating against a
    list known to be wrong would refuse correct declarations and admit
    incorrect ones, so the check belongs where the CHECK is.

    :param application: which application declares this resource type;
        required, because an entry nobody owns cannot be attributed or
        retired. there is no application id-space on the platform today,
        so this is a name the declaring application chooses, in the same
        way a capability source chooses its ``name``
    :ptype application: str
    :param resource_type: the ``namespace_type`` value roles bucket on
    :ptype resource_type: str
    :param label: operator-facing name for the resource type
    :ptype label: str
    :param description: longer operator-facing prose, optional
    :ptype description: str
    :param actions: the closed set of actions declared on this resource
        type; an action absent from it cannot appear in a role
    :ptype actions: tuple[ActionDescriptor, ...]
    """

    model_config = ConfigDict(frozen=True)

    application: str
    resource_type: str
    label: str
    description: str = ""
    actions: tuple[ActionDescriptor, ...]

    @model_validator(mode="after")
    def _identity_fields_are_present(self) -> ResourceTypeDescriptor:
        """an entry with no application cannot be attributed or retired; one with
        no resource type is not addressable at all."""
        if not self.application.strip():
            raise ValueError(f"resource type {self.resource_type!r} declares no application")
        if not self.resource_type.strip():
            raise ValueError(f"application {self.application!r} declares a resource type with an empty name")
        if not self.label.strip():
            raise ValueError(f"resource type {self.resource_type!r} declares an empty label")
        return self

    @model_validator(mode="after")
    def _wildcard_is_not_declarable(self) -> ResourceTypeDescriptor:
        """``"*"`` is the evaluator's type-agnostic bucket, not a namespace type.

        an entry named ``"*"`` would make every wildcard role — the shipped
        ``Reader`` shape — validate against one application's action list, which
        is both wrong and unowned.
        """
        if self.resource_type == WILDCARD_RESOURCE_TYPE:
            raise ValueError(
                f"application {self.application!r} declares the wildcard "
                f"{WILDCARD_RESOURCE_TYPE!r} as a resource type; the wildcard names no "
                "namespace type and is never validated against a catalog",
            )
        return self

    @model_validator(mode="after")
    def _actions_are_present_and_unique(self) -> ResourceTypeDescriptor:
        """a resource type granting nothing is a row no role can reference, and two
        descriptors for one action means one label is silently dropped — which one
        depending on iteration order at whichever consumer reads it."""
        if not self.actions:
            raise ValueError(f"resource type {self.resource_type!r} declares no actions")
        names = [action.name for action in self.actions]
        if len(set(names)) != len(names):
            raise ValueError(f"resource type {self.resource_type!r} declares a duplicate action: {sorted(names)}")
        return self

    def declares(self, action: str) -> bool:
        """whether this resource type declares an action.

        :param action: action string as it appears in a role's
            permissions bucket
        :ptype action: str
        :return: whether any declared action matches
        :rtype: bool
        """
        return any(declared.matches(action) for declared in self.actions)


class PermissionCatalog(BaseModel):
    """every resource type declared across every application, indexed.

    **two applications may not claim one resource type.** the shard that
    commissioned this called that a namespacing problem; it is not one.
    ``role.actions_for(namespace.namespace_type)`` takes the namespace
    type and nothing else, so at evaluation time there is no application
    dimension available to tell two claimants apart — merging their
    action sets would let one application's role grant the other's
    action on the same namespace rows. the collision is refused at
    construction instead, loudly and by name, which is the only outcome
    that stays true of what the evaluator actually does.

    lookup stays the exact-match dict hit :meth:`Role.actions_for` is:
    the index is built once here rather than scanned per call, so
    validating a role costs one dict lookup per bucket.

    :param entries: declared resource types, in any order; an empty
        catalog is valid and declares nothing
    :ptype entries: tuple[ResourceTypeDescriptor, ...]
    """

    model_config = ConfigDict(frozen=True)

    entries: tuple[ResourceTypeDescriptor, ...] = ()

    _by_resource_type: Mapping[str, ResourceTypeDescriptor] = PrivateAttr(
        default_factory=lambda: MappingProxyType({}),
    )

    @model_validator(mode="after")
    def _index_entries_refusing_a_second_claim(self) -> PermissionCatalog:
        """build the resource-type index, refusing a type already claimed.

        :raises ValueError: if two entries declare the same resource type
        """
        index: dict[str, ResourceTypeDescriptor] = {}
        for entry in self.entries:
            claimed = index.get(entry.resource_type)
            if claimed is not None:
                raise ValueError(
                    f"application {entry.application!r} declares resource type "
                    f"{entry.resource_type!r}, already declared by {claimed.application!r}; "
                    "the evaluator looks a permissions bucket up by namespace type alone "
                    "and cannot tell two claimants apart",
                )
            index[entry.resource_type] = entry
        self._by_resource_type = MappingProxyType(index)
        return self

    def entry_for(self, resource_type: str) -> ResourceTypeDescriptor | None:
        """resolve a resource type to its declaring entry.

        :param resource_type: bucket key as a role spells it
        :ptype resource_type: str
        :return: declaring entry, or ``None`` when nothing declares it
        :rtype: ResourceTypeDescriptor | None
        """
        return self._by_resource_type.get(resource_type)


def validate_permissions(
    permissions: Mapping[str, Iterable[str]],
    catalog: PermissionCatalog,
) -> tuple[CatalogViolation, ...]:
    """report every pair in a permissions map that no catalog entry declares.

    accepts the shape a role write path holds (``dict[str, list[str]]``)
    and the shape a loaded :class:`~threetears.agent.acl.types.Role`
    holds (``Mapping[str, frozenset[str]]``) alike.

    the :data:`~threetears.agent.acl.types.WILDCARD_RESOURCE_TYPE` bucket
    is skipped by an explicit branch: it names no resource type, so there
    is no entry it could be checked against, and existing wildcard roles
    depend on it. an undeclared resource type is reported once rather
    than once per action under it.

    violations come back in a deterministic order — resource types
    sorted, then actions sorted within each — so the message an operator
    reads does not depend on set iteration order.

    :param permissions: ``{resource_type: [action, ...]}`` mapping to
        check
    :ptype permissions: Mapping[str, Iterable[str]]
    :param catalog: declared vocabulary to check against
    :ptype catalog: PermissionCatalog
    :return: every undeclared pair, empty when the map is fully declared
    :rtype: tuple[CatalogViolation, ...]
    """
    violations: list[CatalogViolation] = []
    for resource_type in sorted(permissions):
        if resource_type == WILDCARD_RESOURCE_TYPE:
            # the wildcard names no resource type; skipping it is the
            # decision, not a side effect of finding no entry for "*".
            continue
        entry = catalog.entry_for(resource_type)
        if entry is None:
            violations.append(
                CatalogViolation(
                    kind=CatalogViolationKind.UNDECLARED_RESOURCE_TYPE,
                    resource_type=resource_type,
                    action=None,
                    message=f"resource type {resource_type!r} is not declared by any application",
                ),
            )
            continue
        for action in sorted(permissions[resource_type]):
            if not entry.declares(action):
                violations.append(
                    CatalogViolation(
                        kind=CatalogViolationKind.UNDECLARED_ACTION,
                        resource_type=resource_type,
                        action=action,
                        message=(
                            f"action {action!r} is not declared on resource type "
                            f"{resource_type!r} by application {entry.application!r}"
                        ),
                    ),
                )
    return tuple(violations)


def enforce_declared_permissions(
    permissions: Mapping[str, Iterable[str]],
    catalog: PermissionCatalog,
) -> None:
    """raise unless every pair in a permissions map is declared.

    the raising form of :func:`validate_permissions`, for a role write
    path that refuses rather than renders.

    :param permissions: ``{resource_type: [action, ...]}`` mapping to
        check
    :ptype permissions: Mapping[str, Iterable[str]]
    :param catalog: declared vocabulary to check against
    :ptype catalog: PermissionCatalog
    :return: nothing; the refusal is the exception
    :rtype: None
    :raises UndeclaredPermission: when any pair is undeclared, carrying
        every violation rather than only the first
    """
    violations = validate_permissions(permissions, catalog)
    if violations:
        raise UndeclaredPermission(violations)
