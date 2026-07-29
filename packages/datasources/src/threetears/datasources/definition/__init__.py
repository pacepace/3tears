"""dataset-definition model: the durable artifact a dataset build is cut from.

A definition is a versioned, content-hashed specification. Materialized
tables are disposable; the definition is not. This package is PURE
pydantic and takes no dependency beyond what ``3tears-datasources``
already carries -- in particular no SQL parser. The compiler that parses
and emits lives Hub-side precisely so a parser dependency here does not
tax every consumer of this wheel.

``dsm-task-01a`` lands the spine:

- :mod:`~threetears.datasources.definition.namespace` -- the seven-name
  predicate namespace and which names bind at which stage
- :mod:`~threetears.datasources.definition.expression` --
  :class:`~threetears.datasources.definition.expression.Comparison`, the
  operator vocabulary, and the recursive
  :class:`~threetears.datasources.definition.expression.Predicate`
- :mod:`~threetears.datasources.definition.grain` --
  :class:`~threetears.datasources.definition.grain.GrainSpec`
- :mod:`~threetears.datasources.definition.parameters` --
  :class:`~threetears.datasources.definition.parameters.ParameterSpec`,
  its enumerations, derivations, cross-parameter constraints, sentinel
  domains, and value sweep
- :mod:`~threetears.datasources.definition.unit` --
  :class:`~threetears.datasources.definition.unit.Unit`,
  :class:`~threetears.datasources.definition.unit.Resolution`, and
  :class:`~threetears.datasources.definition.unit.Qualification`

``dsm-task-01b`` lands set membership and its algebra:

- :mod:`~threetears.datasources.definition.exclusion` --
  :class:`~threetears.datasources.definition.exclusion.ExclusionSpec`,
  its required stage / key / level triple, and ``all_prior`` expansion
- :mod:`~threetears.datasources.definition.rollup` --
  :class:`~threetears.datasources.definition.rollup.Rollup` and its
  label obligation
- :mod:`~threetears.datasources.definition.setexpr` --
  :class:`~threetears.datasources.definition.setexpr.SetExpr` over
  dataset terms, including ``ranked_precedence``

``dsm-task-01c`` lands relations, bridges, and measures:

- :mod:`~threetears.datasources.definition.relation` --
  :class:`~threetears.datasources.definition.relation.RelationRef` and
  the typed / raw derived-table split
- :mod:`~threetears.datasources.definition.bridge` --
  :class:`~threetears.datasources.definition.bridge.BridgeRef` and its
  plural
  :class:`~threetears.datasources.definition.bridge.QualityMeasure`
- :mod:`~threetears.datasources.definition.measure` --
  :class:`~threetears.datasources.definition.measure.Measure`, its
  grain, and ``having``

``dsm-task-01d`` closes the model:

- :mod:`~threetears.datasources.definition.source` -- the four source
  kinds, :class:`~threetears.datasources.definition.source.ArtifactRef`
  with its five artifact kinds, and
  :class:`~threetears.datasources.definition.source.UpstreamPin`, which
  expresses a resolution POLICY rather than a run id
- :mod:`~threetears.datasources.definition.provenance` --
  :class:`~threetears.datasources.definition.provenance.ProvenanceSpec`
  and the positional contract check
- :mod:`~threetears.datasources.definition.delivery` --
  :class:`~threetears.datasources.definition.delivery.ArtifactSpec`,
  :class:`~threetears.datasources.definition.delivery.PhysicalLayout` and
  its ``ENCODE`` strategy,
  :class:`~threetears.datasources.definition.delivery.DerivedColumn`, and
  :class:`~threetears.datasources.definition.delivery.GrantSpec`
- :mod:`~threetears.datasources.definition.expansion` --
  :class:`~threetears.datasources.definition.expansion.Expansion`
- :mod:`~threetears.datasources.definition.dataset` --
  :class:`~threetears.datasources.definition.dataset.DatasetDefinition`
  and the content hash

The model's recursion closes in
:mod:`~threetears.datasources.definition.source`: a predicate may test
membership against a source, and a source may carry a predicate. That
module rebuilds every forward reference the cycle needs, and importing
THIS package always imports it, which is why no consumer has to.
"""

from __future__ import annotations

from threetears.datasources.definition.bridge import (
    BridgeRef,
    BridgeRelation,
    ConflictingQualityMeasure,
    QualityDirection,
    QualityMeasure,
    ThresholdSemantics,
    union_quality_measures,
)
from threetears.datasources.definition.dataset import (
    DatasetDefinition,
    canonical_content,
    content_hash,
)
from threetears.datasources.definition.delivery import (
    ArtifactSpec,
    ColumnEncoding,
    DeliverySpec,
    DerivedColumn,
    DistStyle,
    GranteeKind,
    GrantSpec,
    MaterializationStrategy,
    PhysicalLayout,
    WarehousePrivilege,
)
from threetears.datasources.definition.exclusion import (
    ExclusionLevel,
    ExclusionSpec,
    UnexpandedExclusion,
    expand_all_prior,
    reject_unexpanded_exclusions,
)
from threetears.datasources.definition.expansion import Expansion
from threetears.datasources.definition.expression import (
    ArithmeticExpression,
    Comparison,
    ComparisonOperator,
    Expression,
    LiteralExpression,
    LiteralType,
    Predicate,
    ScalarValue,
    scan_references,
)
from threetears.datasources.definition.grain import GrainSpec
from threetears.datasources.definition.measure import (
    DuplicateMeasureName,
    FilterPosition,
    Measure,
    MeasureExpression,
    MeasureScope,
    UndeclaredMeasure,
    validate_having_measures,
    validate_unique_measure_names,
)
from threetears.datasources.definition.namespace import (
    BindingStage,
    Namespace,
    Reference,
    ReferenceLike,
    bindable_namespaces,
    reject_unbindable,
)
from threetears.datasources.definition.parameters import (
    ParameterConstraint,
    ParameterConstraintViolated,
    ParameterDerivation,
    ParameterSpec,
    ParameterSpecList,
    ParameterSweep,
    ParameterType,
    ParameterValueRejected,
    SentinelBinding,
    SentinelDomain,
    SentinelEffect,
    SentinelKind,
    SubstringDerivation,
    TemplateDerivation,
    validate_parameter_specs,
    validate_parameter_values,
)
from threetears.datasources.definition.provenance import (
    ProvenanceColumn,
    ProvenanceContractViolation,
    ProvenanceSpec,
)
from threetears.datasources.definition.relation import (
    DerivedTable,
    DuplicateRelationAlias,
    JoinKind,
    Projection,
    RawDerivedTable,
    RelationBody,
    RelationRef,
    TypedDerivedTable,
    UndeclaredRelationAlias,
    validate_relation_aliases,
)
from threetears.datasources.definition.rollup import LabelArm, Rollup, RollupEmit
from threetears.datasources.definition.setexpr import (
    COMPOSITION_FILTERED_ARTIFACTS,
    CategoryPosition,
    CompositionPlacement,
    IntersectColumn,
    ResolutionIntersect,
    ResolutionIntersectColumn,
    SetExpr,
    SetOperator,
    SetTerm,
    TermColumn,
)
from threetears.datasources.definition.source import (
    ArtifactKind,
    ArtifactProjection,
    ArtifactRef,
    ArtifactScope,
    ArtifactStage,
    DraftPolicyReference,
    EntityIdNormalization,
    FactSource,
    LiteralEntities,
    Membership,
    ParameterSignatureViolation,
    ProjectionContractViolation,
    RawSelect,
    SourceRef,
    UpstreamPin,
    UpstreamPolicy,
    raw_selects_in,
    reject_policy_reference_to_draft,
)
from threetears.datasources.definition.unit import (
    DuplicateUnitName,
    Qualification,
    Resolution,
    Unit,
    UnqualifiedUnits,
    units_without_qualification,
    validate_qualification_coverage,
    validate_unique_unit_names,
)

__all__ = [
    "COMPOSITION_FILTERED_ARTIFACTS",
    "ArithmeticExpression",
    "ArtifactKind",
    "ArtifactProjection",
    "ArtifactRef",
    "ArtifactScope",
    "ArtifactSpec",
    "ArtifactStage",
    "BindingStage",
    "BridgeRef",
    "BridgeRelation",
    "CategoryPosition",
    "ColumnEncoding",
    "Comparison",
    "ComparisonOperator",
    "CompositionPlacement",
    "ConflictingQualityMeasure",
    "DatasetDefinition",
    "DeliverySpec",
    "DerivedColumn",
    "DerivedTable",
    "DistStyle",
    "DraftPolicyReference",
    "DuplicateMeasureName",
    "DuplicateRelationAlias",
    "DuplicateUnitName",
    "EntityIdNormalization",
    "ExclusionLevel",
    "ExclusionSpec",
    "Expansion",
    "Expression",
    "FactSource",
    "FilterPosition",
    "GrainSpec",
    "GrantSpec",
    "GranteeKind",
    "IntersectColumn",
    "JoinKind",
    "LabelArm",
    "LiteralEntities",
    "LiteralExpression",
    "LiteralType",
    "MaterializationStrategy",
    "Measure",
    "MeasureExpression",
    "MeasureScope",
    "Membership",
    "Namespace",
    "ParameterConstraint",
    "ParameterConstraintViolated",
    "ParameterDerivation",
    "ParameterSignatureViolation",
    "ParameterSpec",
    "ParameterSpecList",
    "ParameterSweep",
    "ParameterType",
    "ParameterValueRejected",
    "PhysicalLayout",
    "Predicate",
    "Projection",
    "ProjectionContractViolation",
    "ProvenanceColumn",
    "ProvenanceContractViolation",
    "ProvenanceSpec",
    "Qualification",
    "QualityDirection",
    "QualityMeasure",
    "RawDerivedTable",
    "RawSelect",
    "Reference",
    "ReferenceLike",
    "RelationBody",
    "RelationRef",
    "Resolution",
    "ResolutionIntersect",
    "ResolutionIntersectColumn",
    "Rollup",
    "RollupEmit",
    "ScalarValue",
    "SentinelBinding",
    "SentinelDomain",
    "SentinelEffect",
    "SentinelKind",
    "SetExpr",
    "SetOperator",
    "SetTerm",
    "SourceRef",
    "SubstringDerivation",
    "TemplateDerivation",
    "TermColumn",
    "ThresholdSemantics",
    "TypedDerivedTable",
    "UndeclaredMeasure",
    "UndeclaredRelationAlias",
    "UnexpandedExclusion",
    "Unit",
    "UnqualifiedUnits",
    "UpstreamPin",
    "UpstreamPolicy",
    "WarehousePrivilege",
    "bindable_namespaces",
    "canonical_content",
    "content_hash",
    "expand_all_prior",
    "raw_selects_in",
    "reject_policy_reference_to_draft",
    "reject_unbindable",
    "reject_unexpanded_exclusions",
    "scan_references",
    "union_quality_measures",
    "units_without_qualification",
    "validate_having_measures",
    "validate_parameter_specs",
    "validate_parameter_values",
    "validate_qualification_coverage",
    "validate_relation_aliases",
    "validate_unique_measure_names",
    "validate_unique_unit_names",
]
