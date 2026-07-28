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

Still to land, named here so an absent symbol reads as a scheduled seam
rather than an omission: ``ExclusionSpec``, ``SetExpr``, and ``Rollup``
(``dsm-task-01b``); ``RelationRef``, ``BridgeRef``, and ``Measure``
(``dsm-task-01c``); the source union, ``ProvenanceSpec``,
``DeliverySpec``, ``ArtifactSpec``, ``DatasetDefinition``, and the
content hash (``dsm-task-01d``).
"""

from __future__ import annotations

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
    "ArithmeticExpression",
    "BindingStage",
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
    "reject_unbindable",
    "units_without_qualification",
    "validate_parameter_specs",
    "validate_parameter_values",
    "validate_qualification_coverage",
    "validate_unique_unit_names",
]
