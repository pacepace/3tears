"""every place a set of rows can come from, and how an upstream is pinned.

``SourceRef`` is four kinds, usable in three positions -- as a
resolution's source, as a
:class:`~threetears.datasources.definition.relation.RelationRef` target,
and inside a :class:`~threetears.datasources.definition.expression.Predicate`
subquery. This module is where the model's recursion closes, so it is
also where the forward references
:mod:`~threetears.datasources.definition.expression`,
:mod:`~threetears.datasources.definition.relation`,
:mod:`~threetears.datasources.definition.bridge`, and
:mod:`~threetears.datasources.definition.provenance` declare are rebuilt.

**``LiteralEntities`` stores what was authored and normalizes what is
emitted, and says which.** Five committed ``voterbase_id``s carry a
leading space and would silently match zero rows; two sets repeat ids.
Stripping silently is better than matching nothing, but a definition
whose ids were altered has to be able to report it -- so the authored
list is stored verbatim, :attr:`LiteralEntities.normalized_ids` is what
the compiler emits, and :attr:`LiteralEntities.normalization` says what
changed. The content hash is taken over the NORMALIZED set, so cleaning
whitespace produces the identical audience and therefore mints no
version.

**``RawSelect`` closes the three holes ``custom_audience_units/`` has.**
Its provenance spec is REQUIRED (D9a) -- an optional one reopens the
escape hatch the type exists to close. Its projection is declared and
verified against a pinned contract, because the readme promises four
columns where the template selects five and every Amazon custom unit is
broken on that today. And its parameters are a DECLARED SIGNATURE rather
than whatever the caller happens to pass: the prototype's renderer
accepts a supplied-but-unused parameter and discovers a used-but-unsupplied
one only when rendering fails, so ``{{ context.job_level }}`` leaves a
committed file unrenderable with no earlier warning.

**The definition names the upstream dataset; the RUN pins the upstream
run.** A pinned run id inside the definition would be part of the content
hash, so re-pointing at a fresh upstream would mint a version, fire drift
on every consumer and -- versions being immutable -- make every
referenced draft permanently unreapable. :class:`UpstreamPin` therefore
expresses a resolution POLICY, and :attr:`UpstreamPin.run` is excluded
from the hash. A draft reference is the one case that pins a specific
run, because forbidding drafts outright would make a shipped audience
unauthorable: one real pair is built the same day. What is forbidden is a
POLICY reference to a draft, which is
:func:`reject_policy_reference_to_draft`.

**An upstream reference is authorized, not merely named.** Naming a
dataset must never be sufficient to read it, or anyone who can author a
definition could grant themselves access. The check is against the
target's visibility and grants and is implemented Hub-side, because this
package cannot see either; cross-customer is permitted WHEN GRANTED, and
the rule is the grant rather than the customer boundary.

**A definition may span datasources** (D20), so every source carries an
optional ``datasource`` qualifier defaulting to the definition's primary.
Reachability is deliberately NOT validated here: the model has no
catalog, the constraint is per-STATEMENT execution, and an unqualified
name resolving in two catalogs is an error naming both rather than a
silent precedence rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import ClassVar, Self, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.datasources.definition.bridge import BridgeRef
from threetears.datasources.definition.expression import Expression, LiteralExpression, Predicate
from threetears.datasources.definition.namespace import Reference
from threetears.datasources.definition.provenance import ProvenanceSpec
from threetears.datasources.definition.relation import Projection, RelationRef, TypedDerivedTable

__all__ = [
    "ArtifactKind",
    "ArtifactProjection",
    "ArtifactRef",
    "ArtifactScope",
    "ArtifactStage",
    "DraftPolicyReference",
    "EntityIdNormalization",
    "FactSource",
    "LiteralEntities",
    "Membership",
    "ParameterSignatureViolation",
    "ProjectionContractViolation",
    "RawSelect",
    "SourceRef",
    "UpstreamPin",
    "UpstreamPolicy",
    "raw_selects_in",
    "reject_policy_reference_to_draft",
]


class ArtifactStage(StrEnum):
    """stage of a set of rows a reference or an exclusion names.

    Lives here rather than beside ``ExclusionSpec`` because it is an
    ARTIFACT property that the exclusion happens to name: the same two
    values scope an :class:`ArtifactRef`, and ``exclusion`` imports this
    module rather than the other way round.

    :cvar RESOLVED: materialized long rows, before qualification
    :cvar QUALIFIED: rows surviving the qualification stage
    """

    RESOLVED = "resolved"
    QUALIFIED = "qualified"


class ProjectionContractViolation(ValueError):
    """raised when a ``RawSelect`` projects something the contract does not.

    The prototype's equivalent is silence: a custom unit projecting four
    columns into a five-column table is spliced as a derived table and
    fails, or does not, depending on what the outer query happens to
    select.
    """


class ParameterSignatureViolation(ValueError):
    """raised when a ``RawSelect``'s signature disagrees with the definition.

    Checked in BOTH directions, because the prototype checks only one:
    ``check_params`` raises on a supplied-but-unused parameter and never
    on a used-but-unsupplied one, so a body reaching for a parameter
    nobody declares is a render-time failure with no earlier warning.
    """


class DraftPolicyReference(ValueError):
    """raised when a resolution POLICY resolves to a draft run.

    A draft may be referenced -- one shipped pair is built the same day --
    but only by pinning that specific run. Letting a policy resolve to one
    would make the draft immortal, since retention protects the resolvable
    target of every live definition.
    """


class ArtifactKind(StrEnum):
    """artifact of a dataset a reference may name.

    FIVE members, not four. ``relationship_union`` is a first-class
    artifact with 8 columns and is the ONLY place expansion members'
    per-person rationale exists -- its ``fact`` column carries the
    influencer who brought each member in. One committed audience reads it
    by name, and ``parity-task-01``'s five-artifact bar already implies
    it. It carries no quality measure at all: ``candidate_count`` is
    ``NULL::int`` and ``source_name`` is ``NULL``, which is the concrete
    case behind D16's explicit unmeasured band.

    :cvar LONG: materialized long rows, grain ``(unit, source record, entity)``
    :cvar QUALIFIED: rows surviving qualification and composition
    :cvar WIDE: the pivot, one column per unit
    :cvar PROVENANCE: per-person rationale for direct members
    :cvar RELATIONSHIP_UNION: per-person rationale for expansion members
    """

    LONG = "long"
    QUALIFIED = "qualified"
    WIDE = "wide"
    PROVENANCE = "provenance"
    RELATIONSHIP_UNION = "relationship_union"


class ArtifactScope(StrEnum):
    """what a reference is scoped to.

    :cvar THIS_DEFINITION: a unit at a named stage, or an artifact of this
        definition
    :cvar DATASET: another definition's output, resolved by policy
    :cvar EXTERNAL: a governed relation, schema-qualified per run
    """

    THIS_DEFINITION = "this_definition"
    DATASET = "dataset"
    EXTERNAL = "external"


class UpstreamPolicy(StrEnum):
    """how an upstream reference resolves to a run.

    :cvar NAMED_RELEASE: a release identified by label; reproducible
    :cvar LATEST_RELEASE: whatever release is current at run time; NOT
        reproducible even though each run is, which is what the
        ``upstream_drift`` imperative exists to disclose
    :cvar DRAFT_RUN: one specific draft run, pinned; no policy resolution
        and therefore no drift
    """

    NAMED_RELEASE = "named_release"
    LATEST_RELEASE = "latest_release"
    DRAFT_RUN = "draft_run"


class UpstreamPin(BaseModel):
    """how an upstream reference resolves, and what it resolved to.

    :attr:`run` is the one field here that is NOT part of the content
    hash. Hashing it would mint a version on every re-point, fire drift on
    every consumer, and make every referenced draft permanently unreapable
    because versions are immutable.

    :ivar policy: how this reference resolves
    :ivar release: release label, for :attr:`UpstreamPolicy.NAMED_RELEASE`
    :ivar run: the run pinned by a draft reference, or recorded by a
        policy resolution; never hashed
    """

    model_config = ConfigDict(extra="forbid")

    hash_excluded_fields: ClassVar[frozenset[str]] = frozenset({"run"})

    policy: UpstreamPolicy
    release: str | None = Field(default=None, min_length=1)
    run: UUID | None = None

    @model_validator(mode="after")
    def _policy_carries_its_own_declarations(self) -> Self:
        """require the fields one policy needs and refuse another's.

        :returns: validated pin
        :rtype: UpstreamPin
        :raises ValueError: a named release carries no label, a latest
            release carries one, or a draft reference carries no run
        """
        if self.policy is UpstreamPolicy.NAMED_RELEASE and self.release is None:
            raise ValueError("a named_release policy names the release it resolves to")
        if self.policy is not UpstreamPolicy.NAMED_RELEASE and self.release is not None:
            raise ValueError(f"{self.policy.value!r} resolves no named release; drop the release label")
        if self.policy is UpstreamPolicy.DRAFT_RUN and self.run is None:
            raise ValueError(
                "a draft reference pins that specific run: there is no policy that resolves to a "
                "draft, because a resolvable draft would be held past its TTL forever"
            )
        return self

    @property
    def is_policy_resolution(self) -> bool:
        """whether this reference resolves by policy rather than by pin.

        :returns: True for a release policy, False for a pinned draft
        :rtype: bool
        """
        return self.policy is not UpstreamPolicy.DRAFT_RUN


def reject_policy_reference_to_draft(pin: UpstreamPin, target_is_draft: bool, dataset: str) -> None:
    """reject a resolution policy pointed at a draft.

    The model cannot see run status, so the caller supplies it. Stated
    here rather than Hub-side so the rule travels with the type it
    constrains.

    :param pin: the authored upstream pin
    :ptype pin: UpstreamPin
    :param target_is_draft: whether the reference resolved to a draft run
    :ptype target_is_draft: bool
    :param dataset: upstream dataset name, named in the message
    :ptype dataset: str
    :returns: None
    :rtype: None
    :raises DraftPolicyReference: a policy reference resolved to a draft
    """
    if target_is_draft and pin.is_policy_resolution:
        raise DraftPolicyReference(
            f"{pin.policy.value!r} resolved dataset {dataset!r} to a draft run. a draft IS "
            "referenceable -- one shipped pair is built the same day -- but only by pinning that "
            "run with policy 'draft_run'. a policy reference would hold the draft past its TTL for "
            "as long as the referencing definition lives"
        )


class ArtifactProjection(BaseModel):
    """the column select, filter, grouping, and ``HAVING`` a reference carries.

    Distinct from
    :class:`~threetears.datasources.definition.relation.Projection`, which
    is ONE output column. A reference needs the whole shape because one
    real unit derives its entire allowlist from an upstream PROVENANCE
    table under a ``HAVING`` over a count.

    :ivar columns: output columns, in authored order
    :ivar distinct: whether the projection de-duplicates its own rows
    :ivar where: filter over the referenced artifact's own scope
    :ivar group_by: group key; an integer literal is a positional ordinal
        into :attr:`columns`, which is how the corpus authors it
    :ivar having: post-aggregate filter
    """

    model_config = ConfigDict(extra="forbid")

    columns: list[Projection] = Field(default_factory=list)
    distinct: bool = False
    where: Predicate | None = None
    group_by: list[Expression] = Field(default_factory=list)
    having: Predicate | None = None

    @model_validator(mode="after")
    def _projection_declares_something(self) -> Self:
        """reject an empty projection and an out-of-range group ordinal.

        :returns: validated projection
        :rtype: ArtifactProjection
        :raises ValueError: nothing is declared, or a group key is a
            literal that is not a positional ordinal in range
        """
        if not (self.columns or self.where or self.group_by or self.having or self.distinct):
            raise ValueError(
                "projection declares nothing; omit it to read the referenced artifact whole, or "
                "declare columns, a filter, a grouping, or a having"
            )
        for term in self.group_by:
            if not isinstance(term, LiteralExpression):
                continue
            ordinal = term.literal
            if not isinstance(ordinal, int) or isinstance(ordinal, bool):
                raise ValueError(
                    f"group_by literal {ordinal!r} is not a positional ordinal; a group key is a "
                    "reference, an arithmetic expression, or an integer position into columns"
                )
            if not 1 <= ordinal <= len(self.columns):
                raise ValueError(
                    f"group_by ordinal {ordinal} names no column; this projection projects {len(self.columns)} columns"
                )
        return self

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from this projection's own scope.

        :returns: references in authored order
        :rtype: tuple[Reference, ...]
        """
        collected: tuple[Reference, ...] = ()
        for column in self.columns:
            collected = collected + column.references
        for term in self.group_by:
            collected = collected + ((term,) if isinstance(term, Reference) else term.references)
        for predicate in (self.where, self.having):
            if predicate is not None:
                collected = collected + predicate.references
        return collected


class ArtifactRef(BaseModel):
    """reference to one set of rows, at one of three scopes.

    :ivar datasource: catalog this reference binds in; ``None`` is the
        definition's primary
    :ivar scope: what the reference is scoped to
    :ivar unit: unit of this definition, for ``this_definition``
    :ivar stage: which stage of that unit; required with :attr:`unit`,
        because the resolved and qualified sets of one unit are different
        sets and subtracting the wrong one changes the audience silently
    :ivar dataset: another definition, for ``dataset``
    :ivar run: how the upstream resolves, for ``dataset``
    :ivar artifact: which artifact is read
    :ivar table: governed relation, for ``external``
    :ivar schema_expr: per-run schema, e.g. ``param.release_schema``
    :ivar projection: column select, filter, grouping, and ``HAVING``
    """

    model_config = ConfigDict(extra="forbid")

    datasource: str | None = Field(default=None, min_length=1)
    scope: ArtifactScope
    unit: str | None = Field(default=None, min_length=1)
    stage: ArtifactStage | None = None
    dataset: str | None = Field(default=None, min_length=1)
    run: UpstreamPin | None = None
    artifact: ArtifactKind | None = None
    table: str | None = Field(default=None, min_length=1)
    schema_expr: Expression | None = None
    projection: ArtifactProjection | None = None

    @model_validator(mode="after")
    def _scope_carries_its_own_fields(self) -> Self:
        """require the fields one scope needs and refuse another's.

        :returns: validated reference
        :rtype: ArtifactRef
        :raises ValueError: a scope carries a field belonging to another,
            or omits one of its own
        """
        if self.scope is ArtifactScope.THIS_DEFINITION:
            self._this_definition_is_complete()
        elif self.scope is ArtifactScope.DATASET:
            self._dataset_is_complete()
        else:
            self._external_is_complete()
        return self

    def _this_definition_is_complete(self) -> None:
        """check a reference to this definition's own rows.

        :returns: None
        :rtype: None
        :raises ValueError: the reference names neither a unit nor an
            artifact, a unit carries no stage, or a foreign field is set
        """
        self._forbid("this_definition", ("dataset", "run", "table", "schema_expr"))
        if self.unit is None and self.artifact is None:
            raise ValueError("a this_definition reference names a unit at a stage, or an artifact of this definition")
        if self.unit is not None and self.stage is None:
            raise ValueError(
                f"unit {self.unit!r} carries no stage; the resolved and qualified sets of one unit "
                "are different sets, and reading the wrong one changes the audience silently"
            )
        if self.unit is None and self.stage is not None:
            raise ValueError(f"stage {self.stage.value!r} applies to a unit of this definition only")

    def _dataset_is_complete(self) -> None:
        """check a reference to another definition's output.

        :returns: None
        :rtype: None
        :raises ValueError: the dataset or artifact is absent, or a
            foreign field is set
        """
        self._forbid("dataset", ("unit", "stage", "table", "schema_expr"))
        if self.dataset is None:
            raise ValueError("a dataset reference names the upstream definition")
        if self.artifact is None:
            raise ValueError(
                f"dataset {self.dataset!r} carries no artifact; an upstream's long, qualified, wide, "
                "provenance, and relationship_union artifacts are five different sets, and one real "
                "unit derives its whole allowlist from the provenance one"
            )

    def _external_is_complete(self) -> None:
        """check a reference to a governed external relation.

        :returns: None
        :rtype: None
        :raises ValueError: the table is absent, or a foreign field is set
        """
        self._forbid("external", ("unit", "stage", "dataset", "run", "artifact"))
        if self.table is None:
            raise ValueError("an external reference names the governed relation it reads")

    def _forbid(self, scope_label: str, field_names: Sequence[str]) -> None:
        """reject fields belonging to another scope.

        :param scope_label: scope named in the message
        :ptype scope_label: str
        :param field_names: fields this scope does not carry
        :ptype field_names: collections.abc.Sequence[str]
        :returns: None
        :rtype: None
        :raises ValueError: one of the named fields is set
        """
        offending = [name for name in field_names if getattr(self, name) is not None]
        if offending:
            raise ValueError(f"scope {scope_label!r} carries no {offending}")

    @property
    def is_upstream(self) -> bool:
        """whether this reference reads another dataset.

        :returns: True for a ``dataset`` scope
        :rtype: bool
        """
        return self.scope is ArtifactScope.DATASET

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from this reference's own fields.

        :returns: references in schema-then-projection order
        :rtype: tuple[Reference, ...]
        """
        schema: tuple[Reference, ...] = ()
        if self.schema_expr is not None:
            schema = (self.schema_expr,) if isinstance(self.schema_expr, Reference) else self.schema_expr.references
        projection = () if self.projection is None else self.projection.references
        return schema + projection


class FactSource(BaseModel):
    """the fact table a resolution reads, and the schema it lives in per run.

    The corpus selects it with a two-valued ``facts_table`` key
    interpolated into a table name, and predicates address it through the
    ``source.*`` namespace rather than through whatever alias a template
    chose -- which is the defect the namespace exists to close, since the
    prototype's aliases drifted between ``empl.*`` and ``facts.*`` for one
    slot.

    :ivar relation: governed relation name
    :ivar datasource: catalog it binds in; ``None`` is the primary
    :ivar schema_expr: per-run schema, e.g. ``param.match_schema``
    """

    model_config = ConfigDict(extra="forbid")

    relation: str = Field(min_length=1)
    datasource: str | None = Field(default=None, min_length=1)
    schema_expr: Expression | None = None

    @model_validator(mode="after")
    def _relation_is_not_blank(self) -> Self:
        """reject a relation name that is only whitespace.

        :returns: validated source
        :rtype: FactSource
        :raises ValueError: relation carries no text
        """
        if not self.relation.strip():
            raise ValueError("fact source relation carries no text")
        return self

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from the per-run schema expression.

        :returns: references in appearance order
        :rtype: tuple[Reference, ...]
        """
        collected: tuple[Reference, ...] = ()
        if self.schema_expr is not None:
            collected = (self.schema_expr,) if isinstance(self.schema_expr, Reference) else self.schema_expr.references
        return collected


class EntityIdNormalization(BaseModel):
    """what normalizing an authored entity-id set changed.

    Derived from the authored list rather than stored beside it, so it can
    never disagree with the ids it describes and can never enter the
    content hash.

    :ivar trimmed: authored ids whose surrounding whitespace was removed,
        in authored form and authored order
    :ivar duplicates: normalized ids that appeared more than once
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trimmed: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()

    @property
    def changed_anything(self) -> bool:
        """whether normalization altered the authored set at all.

        :returns: True when anything was trimmed or de-duplicated
        :rtype: bool
        """
        return bool(self.trimmed or self.duplicates)


class LiteralEntities(BaseModel):
    """an enumerated set of entity ids, with no bridge and no quality measure.

    The five hand-match units and their thousand-plus ids, which a
    standalone script materializes out of band today -- a script that has
    never run as written, since it is defined with five parameters and
    called with four.

    The authored list is stored VERBATIM so the definition round-trips and
    :attr:`normalization` can report against it. What the compiler emits
    is :attr:`normalized_ids`, and the content hash is taken over that, so
    cleaning whitespace out of an authored set produces the identical
    audience and therefore mints no version.

    :ivar entity_ids: ids exactly as authored
    :ivar datasource: catalog this set binds in; ``None`` is the primary
    """

    model_config = ConfigDict(extra="forbid")

    entity_ids: list[str] = Field(min_length=1)
    datasource: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _every_id_survives_normalization(self) -> Self:
        """reject an id that normalizes to nothing.

        Trimming rescues a leading space. Nothing rescues an id that is
        only whitespace, and the prototype's failure mode for one is to
        match zero rows in silence.

        :returns: validated set
        :rtype: LiteralEntities
        :raises ValueError: an authored id is blank once trimmed
        """
        blank = [entity_id for entity_id in self.entity_ids if not entity_id.strip()]
        if blank:
            raise ValueError(
                f"{len(blank)} authored entity id(s) carry no text and normalization cannot rescue "
                "them, so the set matches nothing for those rows. remove them or write the id"
            )
        return self

    @property
    def normalized_ids(self) -> tuple[str, ...]:
        """ids as the compiler emits them: trimmed, de-duplicated, ordered.

        :returns: normalized ids in first-appearance order
        :rtype: tuple[str, ...]
        """
        seen: dict[str, None] = {}
        for entity_id in self.entity_ids:
            seen[entity_id.strip()] = None
        return tuple(seen)

    @property
    def normalization(self) -> EntityIdNormalization:
        """what normalization changed about the authored set.

        :returns: the trimmed ids and the repeated ones
        :rtype: EntityIdNormalization
        """
        trimmed = tuple(entity_id for entity_id in self.entity_ids if entity_id != entity_id.strip())
        stripped = [entity_id.strip() for entity_id in self.entity_ids]
        repeated: dict[str, None] = {}
        for index, entity_id in enumerate(stripped):
            if entity_id in stripped[:index]:
                repeated[entity_id] = None
        return EntityIdNormalization(trimmed=trimmed, duplicates=tuple(repeated))

    def content_payload(self) -> dict[str, object]:
        """projection of this set the content hash is taken over.

        The NORMALIZED ids, so a whitespace cleanup -- which produces the
        identical audience -- does not mint a version and does not fire
        ``definition_drift`` over every historical run.

        :returns: hashable projection
        :rtype: dict[str, object]
        """
        return {"entity_ids": list(self.normalized_ids), "datasource": self.datasource}


class RawSelect(BaseModel):
    """a source body held as a SQL string, with its holes closed.

    An escape hatch that ``parity-task-03`` scores as a ``raw:`` use, and
    a separate TYPE rather than a flag so the gate is structural.

    :ivar raw_sql: body text, re-emitted byte-exactly
    :ivar projection: columns this body projects, in projection order
    :ivar provenance: the rationale body this source emits; REQUIRED
    :ivar parameters: declared parameter signature, checked in both
        directions
    :ivar datasource: catalog it binds in; ``None`` is the primary
    """

    model_config = ConfigDict(extra="forbid")

    raw_sql: str
    projection: list[str] = Field(min_length=1)
    provenance: ProvenanceSpec
    parameters: list[str] = Field(default_factory=list)
    datasource: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _body_and_signature_are_coherent(self) -> Self:
        """reject a blank body, a repeated column, or a repeated parameter.

        :returns: validated source
        :rtype: RawSelect
        :raises ValueError: the body is blank, a projected column is blank
            or repeats, or a parameter name is not an identifier or
            repeats
        """
        if not self.raw_sql.strip():
            raise ValueError("raw_sql carries no text")
        if any(not column.strip() for column in self.projection):
            raise ValueError("projection carries a blank column name")
        if len(self.projection) != len(set(self.projection)):
            raise ValueError(f"projection names a column twice: {self.projection!r}")
        for name in self.parameters:
            if not name.isidentifier():
                raise ValueError(f"parameter {name!r} is not a parameter name")
        if len(self.parameters) != len(set(self.parameters)):
            raise ValueError(f"parameters names a parameter twice: {self.parameters!r}")
        return self

    @property
    def is_escape_hatch(self) -> bool:
        """whether this body escapes the typed model.

        :returns: True; a SQL string is always an escape hatch
        :rtype: bool
        """
        return True

    def verify_projection_against(self, contract: Sequence[str]) -> None:
        """reject a projection that disagrees with the pinned contract.

        :param contract: pinned contract columns, in contract order
        :ptype contract: collections.abc.Sequence[str]
        :returns: None
        :rtype: None
        :raises ProjectionContractViolation: a column is missing or extra
        """
        pinned = tuple(contract)
        missing = [name for name in pinned if name not in self.projection]
        extra = [name for name in self.projection if name not in pinned]
        if missing or extra:
            raise ProjectionContractViolation(
                f"raw select projects {list(self.projection)!r} against contract {list(pinned)!r}: "
                f"missing {missing}, extra {extra}. the readme promises four long columns where the "
                "template selects five, which is why every committed custom unit fails this"
            )

    def verify_parameters_against(self, declared: Sequence[str]) -> None:
        """reject a signature naming a parameter the definition does not declare.

        :param declared: parameter names the definition declares
        :ptype declared: collections.abc.Sequence[str]
        :returns: None
        :rtype: None
        :raises ParameterSignatureViolation: the signature names an
            undeclared parameter
        """
        undeclared = [name for name in self.parameters if name not in set(declared)]
        if undeclared:
            raise ParameterSignatureViolation(
                f"raw select reads undeclared parameters {undeclared}; the definition declares "
                f"{sorted(declared)}. the signature is checked in both directions here, because the "
                "prototype checks supplied-to-used and never used-to-supplied, so a body reaching "
                "for a parameter nobody declares fails only at render"
            )


SourceRef: TypeAlias = FactSource | RawSelect | LiteralEntities | ArtifactRef
"""every place a set of rows can come from.

Usable as a resolution's source, as a
:class:`~threetears.datasources.definition.relation.RelationRef` target,
and inside a
:class:`~threetears.datasources.definition.expression.Predicate`
subquery. The upstream kind design section 7 calls ``UpstreamSource`` is
:class:`ArtifactRef` at :attr:`ArtifactScope.DATASET` -- one type rather
than two, because an upstream artifact and this definition's own differ
only in scope and every consumer resolves them the same way.
"""


class Membership(BaseModel):
    """``IN`` / ``NOT IN`` over a value list or over a source.

    Both forms are real: the corpus tests an industry column against an
    inline list with an AUTHORED operator, and tests entities against
    another audience's output. Its ``NOT IN`` over a ``LEFT``-joined
    column is also the corpus's three-valued-logic inversion -- ``NULL NOT
    IN (...)`` is ``NULL``, which drops every unmatched source and
    reverses the authored intent -- so the negation is a declared field
    rather than an operator spelling.

    :ivar expression: what is tested
    :ivar negate: whether the test is ``NOT IN``
    :ivar values: inline value list; exactly one of this and :attr:`source`
    :ivar source: the set tested against
    """

    model_config = ConfigDict(extra="forbid")

    expression: Expression
    negate: bool = False
    values: list[Expression] | None = None
    source: SourceRef | None = None

    @model_validator(mode="after")
    def _names_exactly_one_set(self) -> Self:
        """require exactly one of a value list and a source.

        :returns: validated membership
        :rtype: Membership
        :raises ValueError: neither or both are named, or the value list
            is empty
        """
        named = [name for name in ("values", "source") if getattr(self, name) is not None]
        if len(named) != 1:
            raise ValueError(f"a membership tests against exactly one of values / source; got {named or 'none'}")
        if self.values is not None and not self.values:
            raise ValueError(
                "values is empty, which renders NOT IN ('') -- a filter rather than a no-op, "
                "dropping every NULL row by three-valued logic. omit the predicate instead"
            )
        return self

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from the tested expression and the set.

        :returns: references in expression-then-set order
        :rtype: tuple[Reference, ...]
        """
        collected: tuple[Reference, ...] = (
            (self.expression,) if isinstance(self.expression, Reference) else self.expression.references
        )
        for value in self.values or ():
            collected = collected + ((value,) if isinstance(value, Reference) else value.references)
        if isinstance(self.source, ArtifactRef | FactSource):
            collected = collected + self.source.references
        return collected


def raw_selects_in(sources: Sequence[SourceRef]) -> tuple[RawSelect, ...]:
    """the raw bodies among a sequence of sources.

    :param sources: authored sources
    :ptype sources: collections.abc.Sequence[SourceRef]
    :returns: raw bodies, in authored order
    :rtype: tuple[RawSelect, ...]
    """
    return tuple(source for source in sources if isinstance(source, RawSelect))


BridgeRef.model_rebuild()
Membership.model_rebuild()
Predicate.model_rebuild()
ProvenanceSpec.model_rebuild()
RelationRef.model_rebuild()
TypedDerivedTable.model_rebuild()
