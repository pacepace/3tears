"""seven-name predicate namespace, and which names bind at which stage.

Predicates in a dataset definition reference the resolution's fact source,
the bridge row's own attributes, the entity attribute table, per-unit join
aliases, declared measures, the materialized long row, and run parameters.
Seven names, and every reference in the audience-builder corpus lands in
one of them:

===================  =========================================
namespace            binds to
===================  =========================================
``source.*``         the resolution's fact source
``bridge.*``         the bridge row
``entity.*``         the entity attribute table
``rel.<alias>.*``    a declared relation, by its authored alias
``measure.<name>``   a declared measure, in ``having`` only
``resolved.*``       the materialized long row, at qualification and later
``param.*``          run parameters
===================  =========================================

``resolved.*`` is not a convenience. Qualification predicates run against
the long table, where the quality column is already an aggregate and
``record_year`` may be a ``MAX`` or a group key depending on the
resolution path. Binding them to ``source.*`` or ``bridge.*`` -- which
name pre-aggregate rows that no longer exist at that stage -- silently
mis-binds the working-age filter in every audience that uses one. So
``source`` and ``bridge`` are ABSENT from the qualification stage's
bindable set rather than merely discouraged there.

``source.*`` and ``bridge.*`` also close a real defect the corpus carries:
the prototype's aliases drifted (``empl.*`` in the Amazon audiences,
``facts.*`` in the UHG and universal ones, for the same slot), so
definitions were coupled to a template's internal alias choice. A
namespace decouples them.

Stages later than qualification -- composition, delivery, provenance --
land with the shards that introduce them (``dsm-task-01b`` and
``-01d``); they bind the qualification set.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator

__all__ = [
    "BindingStage",
    "Namespace",
    "Reference",
    "ReferenceLike",
    "bindable_namespaces",
    "reject_unbindable",
]


class Namespace(StrEnum):
    """seven names a predicate reference may be bound to.

    :cvar SOURCE: resolution's fact source
    :cvar BRIDGE: bridge row, carrying its own quality attributes
    :cvar ENTITY: entity attribute table
    :cvar REL: declared relation, addressed by its authored alias
    :cvar MEASURE: declared measure, bindable in ``having`` only
    :cvar RESOLVED: materialized long row, at qualification and later
    :cvar PARAM: run parameters
    """

    SOURCE = "source"
    BRIDGE = "bridge"
    ENTITY = "entity"
    REL = "rel"
    MEASURE = "measure"
    RESOLVED = "resolved"
    PARAM = "param"


class BindingStage(StrEnum):
    """build stages that bind a different set of namespaces.

    :cvar RESOLUTION: per-unit resolution, before any aggregate exists
    :cvar HAVING: the resolution's post-aggregate filter, over declared
        measures
    :cvar QUALIFICATION: stage reading the materialized long row
    """

    RESOLUTION = "resolution"
    HAVING = "having"
    QUALIFICATION = "qualification"


_RESOLUTION_NAMESPACES = frozenset(
    {Namespace.SOURCE, Namespace.BRIDGE, Namespace.ENTITY, Namespace.REL, Namespace.PARAM}
)

_QUALIFICATION_NAMESPACES = frozenset({Namespace.RESOLVED, Namespace.ENTITY, Namespace.REL, Namespace.PARAM})

_STAGE_NAMESPACES: dict[BindingStage, frozenset[Namespace]] = {
    BindingStage.RESOLUTION: _RESOLUTION_NAMESPACES,
    BindingStage.HAVING: _RESOLUTION_NAMESPACES | frozenset({Namespace.MEASURE}),
    BindingStage.QUALIFICATION: _QUALIFICATION_NAMESPACES,
}


def bindable_namespaces(stage: BindingStage) -> frozenset[Namespace]:
    """namespaces a predicate may reference at one stage.

    :param stage: build stage the predicate runs at
    :ptype stage: BindingStage
    :returns: bindable namespaces
    :rtype: frozenset[Namespace]
    """
    return _STAGE_NAMESPACES[stage]


def _parse_reference(text: str) -> tuple[Namespace, str | None, str]:
    """split an authored reference into namespace, optional alias, and name.

    :param text: authored reference, e.g. ``rel.old_knowwho.pid``
    :ptype text: str
    :returns: ``(namespace, alias, name)``; alias is set only for ``rel``
    :rtype: tuple[Namespace, str | None, str]
    :raises ValueError: text is not a legal reference in the seven-name
        namespace
    """
    segments = text.split(".")
    if len(segments) < 2:
        raise ValueError(
            f"{text!r} is not a namespaced reference: every predicate reference binds one of "
            f"{sorted(member.value for member in Namespace)}, so a bare column name is refused "
            "rather than resolved by accident of FROM order"
        )
    head = segments[0]
    if head not in {member.value for member in Namespace}:
        raise ValueError(
            f"{text!r} names an unknown namespace {head!r}; the seven names are "
            f"{sorted(member.value for member in Namespace)}"
        )
    namespace = Namespace(head)
    expected_segments = 3 if namespace is Namespace.REL else 2
    if len(segments) != expected_segments:
        shape = "rel.<alias>.<column>" if namespace is Namespace.REL else f"{head}.<name>"
        raise ValueError(f"{text!r} is malformed; {head!r} references are written {shape}")
    alias = segments[1] if namespace is Namespace.REL else None
    name = segments[-1]
    for segment in segments[1:]:
        if not segment.isidentifier():
            raise ValueError(f"{text!r} carries a non-identifier segment {segment!r}")
    return namespace, alias, name


class Reference(BaseModel):
    """one namespaced reference, carried verbatim and parsed on validation.

    The authored text is the single stored value so it re-emits
    byte-exactly; :attr:`namespace`, :attr:`alias`, and :attr:`name` are
    computed from it rather than stored beside it, because a stored copy
    is a second source of truth that drifts.

    :ivar ref: authored reference text, e.g. ``resolved.record_year``
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str

    @model_validator(mode="after")
    def _reference_is_well_formed(self) -> Self:
        """reject a reference that binds no namespace.

        :returns: validated reference
        :rtype: Reference
        :raises ValueError: text is not a legal namespaced reference
        """
        _parse_reference(self.ref)
        return self

    @property
    def namespace(self) -> Namespace:
        """namespace this reference binds.

        :returns: bound namespace
        :rtype: Namespace
        """
        namespace, _, _ = _parse_reference(self.ref)
        return namespace

    @property
    def alias(self) -> str | None:
        """relation alias, for ``rel.<alias>.*`` references only.

        :returns: authored alias, or None outside the ``rel`` namespace
        :rtype: str | None
        """
        _, alias, _ = _parse_reference(self.ref)
        return alias

    @property
    def name(self) -> str:
        """column, measure, or parameter name this reference selects.

        :returns: trailing name segment
        :rtype: str
        """
        _, _, name = _parse_reference(self.ref)
        return name

    def is_bindable_at(self, stage: BindingStage) -> bool:
        """whether this reference's namespace binds at one stage.

        :param stage: build stage the predicate runs at
        :ptype stage: BindingStage
        :returns: True when the namespace is bindable there
        :rtype: bool
        """
        return self.namespace in bindable_namespaces(stage)


def _coerce_reference(value: object) -> object:
    """map an authored reference string onto the model's field shape.

    :param value: raw authored reference
    :ptype value: object
    :returns: value in a shape :class:`Reference` can validate
    :rtype: object
    """
    return {"ref": value} if isinstance(value, str) else value


ReferenceLike: TypeAlias = Annotated[Reference, BeforeValidator(_coerce_reference)]
"""reference field that also accepts the bare authored string."""


def reject_unbindable(references: tuple[Reference, ...], stage: BindingStage, field_label: str) -> None:
    """raise when any reference binds a namespace the stage does not carry.

    The message names the offending reference, the stage, and the legal
    set, because the failure this guards against -- a qualification
    predicate bound to ``source.*`` -- otherwise produces a wrong audience
    with no error at all.

    :param references: every reference reachable from the predicate tree
    :ptype references: tuple[Reference, ...]
    :param stage: build stage the predicate runs at
    :ptype stage: BindingStage
    :param field_label: field being validated, named in the message
    :ptype field_label: str
    :returns: None
    :rtype: None
    :raises ValueError: a reference is not bindable at the stage
    """
    bindable = bindable_namespaces(stage)
    offending = next((ref for ref in references if ref.namespace not in bindable), None)
    if offending is not None:
        legal = ", ".join(sorted(member.value for member in bindable))
        raise ValueError(
            f"{field_label} references {offending.ref!r}: namespace "
            f"{offending.namespace.value!r} is not bindable at the {stage.value} stage "
            f"(bindable: {legal}). source.* and bridge.* name pre-aggregate rows that no "
            "longer exist once the long row is materialized, and resolved.* / measure.* name "
            "rows that do not exist yet during resolution"
        )
