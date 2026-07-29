"""the top-level definition, and the content hash that IS its version.

A definition is the durable artifact; materialized tables are disposable.
Every property of the hash below is therefore a correctness bar rather
than a nicety, because the hash is what mints a version and a version is
immutable.

**Stable under field reordering.** The canonical form is built from
declared field order and serialized with sorted keys, so re-ordering keys
in an authored file cannot mint a version. Instability here means a
cosmetic edit fires ``definition_drift`` over every historical run.

**Stable across serialise then deserialise.** A definition that hashes
differently after a store-and-load cycle mints a version by itself. The
hazard was real rather than theoretical: a literal operand stored its
scalar with no type tag, and JSON erases the difference between a text
``'5'`` and a decimal ``5``. Observed, before the fix::

    LiteralExpression(literal=Decimal('5'))  ->  {"literal": "5"}  ->  '5'

which is the reverse spelling of what the build plan recorded but the
same defect, and it cost more than round-trip equality: a text ``'1.0'``
compared against a numeric column -- which a real committed definition
does -- hashed IDENTICALLY to the numeric ``1.0``, so the edit between
them minted no version at all.
:class:`~threetears.datasources.definition.expression.LiteralType` closes
both halves.

**Excluded before serialization, never filtered afterwards.** Three
things are excluded, and each exclusion is declared on the model that
owns the field rather than applied by a pattern over a serialized blob. A
filter misses a nested occurrence, and the miss surfaces months later as
drift nobody can explain:

- ``UpstreamPin.run`` -- the resolved or pinned upstream RUN. Hashing it
  would mint a version on every re-point, fire drift on every consumer,
  and make every referenced draft permanently unreapable, since versions
  are immutable. Only the dataset name and the resolution POLICY are
  hashed.
- ``DeliverySpec.grants`` and ``DeliverySpec.layout``, and
  ``ArtifactSpec.layout`` -- policy and tuning rather than semantics
  (D1). Revoking a group's read access or changing a ``SORTKEY`` must not
  mint a version.
- ``DatasetDefinition.version`` -- the number the hash assigns. Hashing it
  would be circular.

**It MOVES on any semantic edit.** Without this the other properties are
satisfiable by a constant, which is why the projection is built by
walking every non-excluded field rather than by listing the fields worth
hashing.

Two things this model deliberately does NOT carry. Per D21 there is no
``visibility`` and no dataset grant: a dataset is a customer-owned
resource with an audience, and that audience is platform state validated
at promotion and at every grant, not authored content. Per D1 there is no
``retention_days`` either -- retention lives on the dataset record for
exactly the reason the hash excludes grants.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import ClassVar, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.datasources.definition.delivery import ArtifactSpec, DeliverySpec
from threetears.datasources.definition.exclusion import ExclusionSpec, expand_all_prior
from threetears.datasources.definition.expansion import Expansion
from threetears.datasources.definition.grain import GrainSpec
from threetears.datasources.definition.namespace import Namespace
from threetears.datasources.definition.parameters import ParameterSpecList, SentinelBinding
from threetears.datasources.definition.rollup import Rollup
from threetears.datasources.definition.setexpr import SetExpr
from threetears.datasources.definition.source import RawSelect
from threetears.datasources.definition.unit import (
    Qualification,
    Unit,
    validate_qualification_coverage,
    validate_unique_unit_names,
)

__all__ = ["DatasetDefinition", "canonical_content", "content_hash"]


def canonical_content(model: BaseModel) -> object:
    """the projection of a model the content hash is taken over.

    Walks declared fields, skipping every field a model declares in its
    own ``hash_excluded_fields``, and defers to a model's own
    ``content_payload`` where it has one -- which is how a literal entity
    set is hashed over its NORMALIZED ids, so cleaning whitespace out of
    an authored set produces the identical audience and mints no version.

    :param model: model to project
    :ptype model: pydantic.BaseModel
    :returns: plain-Python structure carrying only hashed content
    :rtype: object
    """
    return _canonical(model)


def _canonical(node: object) -> object:
    """recursively project one node into hashable plain Python.

    :param node: model, container, or scalar
    :ptype node: object
    :returns: plain-Python projection
    :rtype: object
    """
    projected: object
    if isinstance(node, BaseModel):
        projected = _canonical_model(node)
    elif isinstance(node, Enum):
        projected = node.value
    elif isinstance(node, Decimal):
        projected = f"decimal:{node!s}"
    elif isinstance(node, UUID):
        projected = str(node)
    elif isinstance(node, datetime | date):
        projected = node.isoformat()
    elif isinstance(node, dict):
        projected = {str(key): _canonical(value) for key, value in node.items()}
    elif isinstance(node, list | tuple):
        projected = [_canonical(item) for item in node]
    else:
        projected = node
    return projected


def _canonical_model(model: BaseModel) -> object:
    """project one model, honouring its own exclusions and payload override.

    :param model: model to project
    :ptype model: pydantic.BaseModel
    :returns: plain-Python projection
    :rtype: object
    """
    payload = getattr(model, "content_payload", None)
    if callable(payload):
        return _canonical(payload())
    excluded: frozenset[str] = getattr(type(model), "hash_excluded_fields", frozenset())
    return {name: _canonical(getattr(model, name)) for name in type(model).model_fields if name not in excluded}


def content_hash(model: BaseModel) -> str:
    """deterministic digest over a model's hashed content.

    :param model: model to hash
    :ptype model: pydantic.BaseModel
    :returns: hex-encoded SHA-256 digest
    :rtype: str
    """
    rendered = json.dumps(
        canonical_content(model), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


class DatasetDefinition(BaseModel):
    """a versioned, content-hashed dataset specification.

    :ivar name: definition name
    :ivar datasource: PRIMARY datasource, where unqualified names bind
    :ivar additional_datasources: further catalogs this definition names;
        a definition is NOT confined to one datasource (D20), and the
        binding constraint is per-STATEMENT execution rather than
        per-definition
    :ivar grain: entity column and its optional delivered alias
    :ivar parameters: declared run parameters
    :ivar sentinel_bindings: sentinel domains declared against columns of
        this definition's OWN long artifact, one binding per column
    :ivar units: declared units, at least one
    :ivar qualification: definition-level arms, scoped by ``applies_to``
    :ivar composition: set algebra over dataset terms; ``None`` is the
        default union of every unit
    :ivar rollups: named unit groups and the labels they emit
    :ivar expansions: edges walked out of the qualified set
    :ivar artifacts: which artifacts this definition emits, each with its
        own column set
    :ivar delivery: derived columns, warehouse grants, delivered layout
    :ivar version: assigned when the hash mints one; never hashed
    """

    model_config = ConfigDict(extra="forbid")

    hash_excluded_fields: ClassVar[frozenset[str]] = frozenset({"version"})

    name: str = Field(min_length=1)
    datasource: str = Field(min_length=1)
    additional_datasources: list[str] = Field(default_factory=list)
    grain: GrainSpec
    parameters: ParameterSpecList = Field(default_factory=list)
    sentinel_bindings: list[SentinelBinding] = Field(default_factory=list)
    units: list[Unit] = Field(min_length=1)
    qualification: list[Qualification] = Field(default_factory=list)
    composition: SetExpr | None = None
    rollups: list[Rollup] = Field(default_factory=list)
    expansions: list[Expansion] = Field(default_factory=list)
    artifacts: list[ArtifactSpec] = Field(min_length=1)
    delivery: DeliverySpec = Field(default_factory=DeliverySpec)
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _definition_is_coherent(self) -> Self:
        """check every definition-level invariant, then expand ``all_prior``.

        :returns: validated definition, with residual extents expanded
        :rtype: DatasetDefinition
        :raises ValueError: a datasource, unit, artifact, rollup,
            expansion, or raw-select signature is inconsistent
        """
        self._datasources_are_distinct()
        self._sentinel_bindings_name_this_definitions_own_columns()
        validate_unique_unit_names(self.units)
        validate_qualification_coverage([unit.name for unit in self.units], self.qualification)
        self._artifacts_are_distinct()
        self._members_name_declared_units()
        self._raw_selects_read_declared_parameters()
        return self._with_expanded_residuals()

    def _datasources_are_distinct(self) -> None:
        """reject a further datasource repeating the primary or itself.

        :returns: None
        :rtype: None
        :raises ValueError: a further datasource repeats
        """
        if self.datasource in self.additional_datasources:
            raise ValueError(
                f"{self.datasource!r} is the primary datasource and is listed again in "
                "additional_datasources; the primary is where unqualified names bind"
            )
        if len(self.additional_datasources) != len(set(self.additional_datasources)):
            raise ValueError("additional_datasources names a datasource twice")

    def _sentinel_bindings_name_this_definitions_own_columns(self) -> None:
        """reject a sentinel bound to a column this definition does not write.

        A binding declares a value the definition MANUFACTURES into its
        own long artifact, which is why ``resolved.*`` is the one
        namespace it takes. A warehouse column's sentinel is a fact about
        the warehouse and belongs to the governed schema layer, where one
        declaration serves every definition that reads the column; a
        parameter's belongs to that parameter. Restating either here would
        be a second source of truth that drifts.

        :returns: None
        :rtype: None
        :raises ValueError: two bindings claim one column, or a target
            binds a namespace other than ``resolved``
        """
        targets = [binding.target.ref for binding in self.sentinel_bindings]
        repeated = sorted({target for target in targets if targets.count(target) > 1})
        if repeated:
            raise ValueError(
                f"sentinel_bindings declares {repeated} twice; one column carries ONE declared "
                "domain, and a second declaration would make the effect depend on authored order"
            )
        for binding in self.sentinel_bindings:
            namespace = binding.target.namespace
            if namespace is Namespace.RESOLVED:
                continue
            hint = (
                "declare it on that parameter, as ParameterSpec.sentinels"
                if namespace is Namespace.PARAM
                else "a warehouse column's domain is a schema fact and belongs to the governed "
                "schema layer, where one declaration serves every definition that reads it"
            )
            raise ValueError(
                f"sentinel binding targets {binding.target.ref!r}, and a definition declares "
                f"sentinels for resolved.* only -- the columns it writes into its own long "
                f"artifact. {hint}"
            )

    def _artifacts_are_distinct(self) -> None:
        """reject a repeated artifact and more than one delivered one.

        :returns: None
        :rtype: None
        :raises ValueError: an artifact is declared twice, or two are
            marked delivered
        """
        kinds = [artifact.artifact for artifact in self.artifacts]
        duplicates = sorted({kind.value for kind in kinds if kinds.count(kind) > 1})
        if duplicates:
            raise ValueError(
                f"artifacts declares {duplicates} twice; each artifact carries ONE column set and a "
                "second declaration would make the emitted contract depend on authored order"
            )
        delivered = [artifact.artifact.value for artifact in self.artifacts if artifact.delivered]
        if len(delivered) > 1:
            raise ValueError(f"artifacts marks {delivered} as delivered; a definition delivers at most one")

    def _members_name_declared_units(self) -> None:
        """reject a rollup or expansion scoped to a unit nobody declares.

        A rollup member may itself name another rollup, which
        ``amazon_tech_audience`` does, so both vocabularies are admitted.

        :returns: None
        :rtype: None
        :raises ValueError: a rollup member or expansion scope names
            neither a declared unit nor a declared rollup
        """
        declared = {label for unit in self.units for label in unit.emitted_labels}
        known = declared | {rollup.name for rollup in self.rollups}
        for rollup in self.rollups:
            unknown = sorted(member for member in rollup.members if member not in known)
            if unknown:
                raise ValueError(
                    f"rollup {rollup.name!r} names {unknown}, which no unit and no rollup declares. "
                    "one committed audience emits a unit that appears in no rollup list and in no "
                    "qualification arm, and every one of its rows is silently dropped"
                )
        for expansion in self.expansions:
            unknown = sorted(unit for unit in expansion.applies_to if unit not in declared)
            if unknown:
                raise ValueError(f"expansion {expansion.name!r} is scoped to undeclared units {unknown}")

    def _raw_selects_read_declared_parameters(self) -> None:
        """reject a raw body whose signature names an undeclared parameter.

        Checked in the direction the prototype does not check: its
        renderer raises on a supplied-but-unused parameter and discovers a
        used-but-unsupplied one only when rendering fails.

        :returns: None
        :rtype: None
        :raises ParameterSignatureViolation: a signature names an
            undeclared parameter
        """
        declared = [spec.name for spec in self.parameters]
        for unit in self.units:
            for resolution in unit.resolutions:
                if isinstance(resolution.source, RawSelect):
                    resolution.source.verify_parameters_against(declared)

    def _with_expanded_residuals(self) -> Self:
        """expand every ``all_prior`` exclusion against the authored order.

        Done at parse rather than at plan time, because "prior" is
        otherwise ambiguous between authored order and a topological order
        free to reorder, and the two give different audiences with no
        error. Expanding here also puts the chosen edges in the content
        hash, where the corpus has only unsorted ``os.listdir()``.

        :returns: this definition, with residual extents expanded
        :rtype: DatasetDefinition
        """
        authored = [unit.name for unit in self.units]
        exclusions: dict[str, ExclusionSpec] = {
            unit.name: unit.exclude for unit in self.units if unit.exclude is not None
        }
        if not exclusions:
            return self
        expanded = expand_all_prior(authored, exclusions)
        for unit in self.units:
            if unit.name in expanded:
                unit.exclude = expanded[unit.name]
        return self

    @property
    def referenced_datasources(self) -> tuple[str, ...]:
        """every catalog this definition may name, primary first.

        :returns: the primary datasource then the further ones
        :rtype: tuple[str, ...]
        """
        return (self.datasource, *self.additional_datasources)

    @property
    def delivered_artifact(self) -> ArtifactSpec | None:
        """the artifact handed to the client, when one is marked.

        :returns: the delivered artifact, or None when none is marked
        :rtype: ArtifactSpec | None
        """
        return next((artifact for artifact in self.artifacts if artifact.delivered), None)

    @property
    def content_hash(self) -> str:
        """the version-minting digest over this definition's semantics.

        Derived, never authored: a stored copy is a second source of truth
        that drifts, and the one thing a version identifier must never do
        is disagree with the content it identifies.

        :returns: hex-encoded SHA-256 digest
        :rtype: str
        """
        return content_hash(self)
