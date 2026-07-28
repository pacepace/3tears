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

Still to land, named here so an absent symbol reads as a scheduled seam
rather than an omission: the source union, ``ProvenanceSpec``,
``DeliverySpec``, ``ArtifactSpec``, ``ArtifactRef``,
``DatasetDefinition``, and the content hash (``dsm-task-01d``).
"""

from __future__ import annotations

from threetears.datasources.definition.bridge import (
    BridgeRef,
    ConflictingQualityMeasure,
    QualityDirection,
    QualityMeasure,
    ThresholdSemantics,
    union_quality_measures,
)
from threetears.datasources.definition.exclusion import (
    ArtifactHandle,
    ArtifactStage,
    ExclusionLevel,
    ExclusionSpec,
    UnexpandedExclusion,
    expand_all_prior,
    reject_unexpanded_exclusions,
)
from threetears.datasources.definition.expression import (
    ArithmeticExpression,
    Comparison,
    ComparisonOperator,
    Expression,
    LiteralExpression,
    Predicate,
    ScalarValue,
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
    "ArtifactHandle",
    "ArtifactStage",
    "BindingStage",
    "BridgeRef",
    "CategoryPosition",
    "CompositionPlacement",
    "ConflictingQualityMeasure",
    "DerivedTable",
    "DuplicateMeasureName",
    "DuplicateRelationAlias",
    "ExclusionLevel",
    "ExclusionSpec",
    "FilterPosition",
    "IntersectColumn",
    "JoinKind",
    "LabelArm",
    "Measure",
    "MeasureExpression",
    "MeasureScope",
    "Projection",
    "QualityDirection",
    "QualityMeasure",
    "RawDerivedTable",
    "RelationBody",
    "RelationRef",
    "ResolutionIntersect",
    "ResolutionIntersectColumn",
    "Rollup",
    "RollupEmit",
    "SetExpr",
    "SetOperator",
    "SetTerm",
    "TermColumn",
    "ThresholdSemantics",
    "TypedDerivedTable",
    "UndeclaredMeasure",
    "UndeclaredRelationAlias",
    "UnexpandedExclusion",
    "Comparison",
    "ComparisonOperator",
    "DuplicateUnitName",
    "Expression",
    "GrainSpec",
    "LiteralExpression",
    "Namespace",
    "ParameterConstraint",
    "ParameterConstraintViolated",
    "ParameterDerivation",
    "ParameterSpec",
    "ParameterSpecList",
    "ParameterSweep",
    "ParameterType",
    "ParameterValueRejected",
    "Predicate",
    "Qualification",
    "Reference",
    "ReferenceLike",
    "Resolution",
    "ScalarValue",
    "SentinelBinding",
    "SentinelDomain",
    "SentinelEffect",
    "SentinelKind",
    "SubstringDerivation",
    "TemplateDerivation",
    "Unit",
    "UnqualifiedUnits",
    "bindable_namespaces",
    "expand_all_prior",
    "reject_unbindable",
    "reject_unexpanded_exclusions",
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
