"""run parameters: enumerations, derivations, cross-constraints, sentinels, sweeps.

This is a deliberately MINIMAL vocabulary, in the house style of
:class:`threetears.knowledge.merge.EntryEnforcement`. It is not a general
expression language for parameters; it carries exactly the shapes the
audience-builder corpus needs, and it grows only when a concrete trap
needs a new field. Four such traps, each verified against source:

- **an enumeration nobody enforces.** ``--vf_suffix`` accepts ``l2`` and
  ``ts``, validated only when the value is truthy, so ``None`` bypasses
  the check entirely.
- **a derivation documented wrong.** ``analytics_schema``'s argparse help
  describes ``{release_schema}_{vf_suffix}_analytics``; the code computes
  ``{match_schema}_analytics``, where ``match_schema`` itself derives.
  With no suffix the real value is ``{release_schema}_analytics``, which
  the help does not describe at all. The code is authoritative and the
  help is drift; declaring the derivation once removes the second place
  for a description to disagree.
- **a cross-parameter constraint living in a help string.**
  ``--tsmart_comm`` is documented "ONLY APPLICABLE WHEN USING
  TARGETSMART" and nothing checks it. Violating it joins the wrong
  commercial file and yields a SMALLER audience with no error. That is
  the worst failure mode in the whole model, because nothing looks wrong,
  so it is a first-class constraint here with a message naming both
  parameters -- never a free-text note.
- **sentinel domains that are implied rather than declared.**
  ``record_year = -1`` makes the working-age ceiling a no-op (at
  ``run_year = 2025`` the bound becomes ``age < 2096``);
  ``record_year = NULL::int`` makes the same comparison ``NULL`` and drops
  EVERY row of the unit. They are opposite failures, both silent, and a
  projection check that verifies columns catches neither. Both are
  declarable, and the declaration states the EFFECT, because "the model
  must be able to say so" is the requirement.

One further shape has no prototype equivalent at all.
``pull_householder_counts.sql`` is a hand-unrolled sweep -- ten
near-identical arms over thresholds 1 through 10 for one knob, stacked
into one result set with the swept value projected as a labelling column.
:class:`ParameterSweep` expresses it, rather than leaving ten copy-pasted
definitions as the only way to author it.

What this module does NOT do is RESOLVE a derivation. Computing
``{release_schema}_{vf_suffix}`` from run values is compiler work; this
package declares the derivation, checks it is well-formed, checks it
names declared parameters, and checks the graph is acyclic.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Self, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from threetears.datasources.definition.expression import ScalarValue
from threetears.datasources.definition.namespace import ReferenceLike

__all__ = [
    "ParameterConstraint",
    "ParameterConstraintViolated",
    "ParameterDerivation",
    "ParameterSpec",
    "ParameterSpecList",
    "ParameterSweep",
    "ParameterType",
    "ParameterValueRejected",
    "SentinelBinding",
    "SentinelDomain",
    "SentinelEffect",
    "SentinelKind",
    "SubstringDerivation",
    "TemplateDerivation",
    "validate_parameter_specs",
    "validate_parameter_values",
]

_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


class ParameterValueRejected(ValueError):
    """raised when a resolved parameter value fails its own declaration.

    Covers a missing required parameter and a value outside a declared
    enumeration. Cross-parameter failures raise the narrower
    :class:`ParameterConstraintViolated`.
    """


class ParameterConstraintViolated(ParameterValueRejected):
    """raised when a cross-parameter constraint is armed and unsatisfied.

    Message names BOTH parameters, because the prototype's version of this
    failure produces a smaller audience with no error and nothing in the
    output says which two knobs disagreed.
    """


class ParameterType(StrEnum):
    """declared type of a run parameter.

    :cvar STRING: text, including schema names
    :cvar INTEGER: whole number
    :cvar DECIMAL: exact decimal; never a binary float
    :cvar BOOLEAN: flag
    :cvar DATE: calendar date; the corpus authors ``YYYYmmdd`` and slices
        the run year out of it
    """

    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"


class SentinelKind(StrEnum):
    """how a sentinel is spelled in the data.

    :cvar VALUE: in-band constant, e.g. ``-1``
    :cvar NULL: SQL ``NULL``
    """

    VALUE = "value"
    NULL = "null"


class SentinelEffect(StrEnum):
    """what a sentinel does to a predicate that compares against it.

    The two corpus sentinels sit at opposite ends of this enum, which is
    the whole reason the effect is declared rather than inferred.

    :cvar WIDENS_PREDICATE: comparison stops constraining, e.g.
        ``record_year = -1`` turning ``age < (70 + run_year - record_year)``
        into ``age < 2096``
    :cvar DROPS_ROW: comparison evaluates ``NULL`` and the row vanishes,
        e.g. ``record_year = NULL::int``
    :cvar UNKNOWN: sentinel is declared but its effect is not yet
        characterised; states the gap rather than hiding it
    """

    WIDENS_PREDICATE = "widens_predicate"
    DROPS_ROW = "drops_row"
    UNKNOWN = "unknown"


class SentinelDomain(BaseModel):
    """one declared sentinel value and what it does.

    :ivar kind: whether the sentinel is an in-band value or ``NULL``
    :ivar value: in-band constant; required for :attr:`SentinelKind.VALUE`
        and forbidden for :attr:`SentinelKind.NULL`
    :ivar meaning: what the sentinel means to whoever wrote the data
    :ivar effect: what it does to a comparison against it
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SentinelKind
    value: ScalarValue = None
    meaning: str = Field(min_length=1)
    effect: SentinelEffect

    @model_validator(mode="after")
    def _value_matches_the_kind(self) -> Self:
        """require a value for an in-band sentinel and forbid one for ``NULL``.

        :returns: validated sentinel
        :rtype: SentinelDomain
        :raises ValueError: kind and value disagree
        """
        if self.kind is SentinelKind.VALUE and self.value is None:
            raise ValueError("a value sentinel declares the constant it stands for")
        if self.kind is SentinelKind.NULL and self.value is not None:
            raise ValueError(f"a null sentinel carries no value; got {self.value!r}")
        return self


class SentinelBinding(BaseModel):
    """sentinels declared against one namespaced column rather than a parameter.

    The corpus's flagship sentinel is ``resolved.record_year``, which is a
    COLUMN, not a parameter -- so declaring sentinels only on
    :class:`ParameterSpec` would leave the real instance unsayable. Where
    a binding attaches on the definition (beside the measure that
    projects the column, or on the definition itself) is settled by
    ``dsm-task-01c`` / ``-01d``; the declaration shape is settled here.

    :ivar target: namespaced column the sentinels apply to
    :ivar sentinels: declared sentinel domains, at least one
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: ReferenceLike
    sentinels: list[SentinelDomain] = Field(min_length=1)

    @model_validator(mode="after")
    def _sentinels_are_distinct(self) -> Self:
        """reject two sentinels declaring the same value.

        :returns: validated binding
        :rtype: SentinelBinding
        :raises ValueError: a sentinel is declared twice
        """
        keys = [(sentinel.kind, sentinel.value) for sentinel in self.sentinels]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.target.ref} declares the same sentinel twice")
        return self


class TemplateDerivation(BaseModel):
    """value computed by interpolating other parameters into a template.

    ``match_schema`` is ``{release_schema}_{vf_suffix}`` when a suffix is
    supplied and plain ``{release_schema}`` when it is not, so
    :attr:`fallback` is a real requirement rather than a convenience.

    :ivar template: interpolation template, e.g. ``{release_schema}_{vf_suffix}``
    :ivar fallback: template used when a placeholder resolves to nothing,
        or ``None`` when every input is required
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    template: str = Field(min_length=1)
    fallback: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _placeholders_are_parameter_names(self) -> Self:
        """require at least one placeholder, and every placeholder an identifier.

        A template with no placeholder is a constant, which belongs in
        :attr:`ParameterSpec.default` where it is visible as one.

        :returns: validated derivation
        :rtype: TemplateDerivation
        :raises ValueError: template carries no placeholder, or carries
            one that is not a parameter name
        """
        for text in (self.template, self.fallback):
            if text is None:
                continue
            names = _PLACEHOLDER.findall(text)
            if not names:
                raise ValueError(f"{text!r} carries no placeholder; declare a default instead")
            for name in names:
                if not name.isidentifier():
                    raise ValueError(f"{text!r} carries a non-identifier placeholder {name!r}")
        return self

    @property
    def inputs(self) -> tuple[str, ...]:
        """parameter names this derivation reads.

        :returns: names in first-appearance order across template then
            fallback, de-duplicated
        :rtype: tuple[str, ...]
        """
        ordered: dict[str, None] = {}
        for text in (self.template, self.fallback):
            for name in _PLACEHOLDER.findall(text or ""):
                ordered[name] = None
        return tuple(ordered)


class SubstringDerivation(BaseModel):
    """value computed by slicing another parameter's text.

    The run year is ``date[:4]`` in the prototype, and a malformed date
    therefore produces a nonsense age ceiling with no error. Declaring the
    slice puts the derivation somewhere a reader can see it.

    :ivar source: parameter this slice reads
    :ivar start: zero-based offset of the slice
    :ivar length: slice length in characters
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    start: int = Field(default=0, ge=0)
    length: int = Field(ge=1)

    @model_validator(mode="after")
    def _source_is_a_parameter_name(self) -> Self:
        """require the source to be spelled as a parameter name.

        :returns: validated derivation
        :rtype: SubstringDerivation
        :raises ValueError: source is not an identifier
        """
        if not self.source.isidentifier():
            raise ValueError(f"source {self.source!r} is not a parameter name")
        return self

    @property
    def inputs(self) -> tuple[str, ...]:
        """parameter names this derivation reads.

        :returns: single-element tuple naming the source
        :rtype: tuple[str, ...]
        """
        return (self.source,)


ParameterDerivation: TypeAlias = TemplateDerivation | SubstringDerivation
"""how one parameter's value is computed from others."""


class ParameterConstraint(BaseModel):
    """cross-parameter rule armed by the owning parameter's value.

    Declared on the parameter that ARMS it, so ``tsmart_comm`` carries
    ``when_value=True, requires_parameter="vf_suffix",
    requires_one_of=["ts"]``. Enforcement raises
    :class:`ParameterConstraintViolated` naming both parameters.

    :ivar when_value: value of the owning parameter that arms this rule
    :ivar requires_parameter: other parameter the rule constrains
    :ivar requires_one_of: values that other parameter may take, at least one
    :ivar message: authored explanation, surfaced in the rejection
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    when_value: ScalarValue
    requires_parameter: str
    requires_one_of: list[ScalarValue] = Field(min_length=1)
    message: str | None = None

    @model_validator(mode="after")
    def _requires_parameter_is_a_name(self) -> Self:
        """require the constrained parameter to be spelled as a name.

        :returns: validated constraint
        :rtype: ParameterConstraint
        :raises ValueError: requires_parameter is not an identifier
        """
        if not self.requires_parameter.isidentifier():
            raise ValueError(f"requires_parameter {self.requires_parameter!r} is not a name")
        return self


class ParameterSweep(BaseModel):
    """several values of one parameter, run in one build and stacked.

    ``pull_householder_counts.sql`` is this shape written out by hand: ten
    arms over thresholds 1 through 10, ``union all``-ed, each projecting
    ``N as candidate_count_cutoff``. The swept value has to reach the
    result set or the stacked rows are indistinguishable, so
    :attr:`emit_column` is required rather than optional.

    :ivar values: distinct values to sweep, in authored order
    :ivar emit_column: column the swept value is projected as
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    values: list[ScalarValue] = Field(min_length=1)
    emit_column: str

    @model_validator(mode="after")
    def _values_are_distinct_and_column_is_a_name(self) -> Self:
        """reject a repeated sweep value or a non-identifier emit column.

        :returns: validated sweep
        :rtype: ParameterSweep
        :raises ValueError: values repeat, or emit_column is not an identifier
        """
        if len(self.values) != len(set(self.values)):
            raise ValueError("sweep values repeat; each arm is one distinct value")
        if not self.emit_column.isidentifier():
            raise ValueError(f"emit_column {self.emit_column!r} is not an identifier")
        return self


class ParameterSpec(BaseModel):
    """one declared run parameter.

    :ivar name: parameter name; ``param.<name>`` binds it in predicates
    :ivar parameter_type: declared type
    :ivar description: authored explanation
    :ivar required: whether a run must supply a value
    :ivar default: value used when a run supplies none
    :ivar enum: values this parameter may take, when a closed set exists
    :ivar derivation: how the value is computed from other parameters
    :ivar constraints: cross-parameter rules this parameter arms
    :ivar sentinels: declared sentinel domains for this parameter's value
    :ivar sweep: several values run in one build, stacked into one result
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    parameter_type: ParameterType
    description: str | None = None
    required: bool = True
    default: ScalarValue = None
    enum: list[ScalarValue] | None = None
    derivation: ParameterDerivation | None = None
    constraints: list[ParameterConstraint] = Field(default_factory=list)
    sentinels: list[SentinelDomain] = Field(default_factory=list)
    sweep: ParameterSweep | None = None

    @model_validator(mode="after")
    def _declaration_is_coherent(self) -> Self:
        """reject a self-contradictory parameter declaration.

        :returns: validated spec
        :rtype: ParameterSpec
        :raises ValueError: name is not an identifier, enumeration is
            empty or repeats, default or a swept value falls outside the
            enumeration, a derived parameter is marked required or
            derives from itself, or a derived parameter is swept
        """
        if not self.name.isidentifier():
            raise ValueError(f"parameter name {self.name!r} is not an identifier")
        if self.enum is not None:
            if not self.enum:
                raise ValueError(f"{self.name}: enum is declared but empty")
            if len(self.enum) != len(set(self.enum)):
                raise ValueError(f"{self.name}: enum repeats a value")
            if self.default is not None and self.default not in self.enum:
                raise ValueError(f"{self.name}: default {self.default!r} is outside the declared enum")
        if self.derivation is not None:
            if self.required:
                raise ValueError(
                    f"{self.name}: a derived parameter is never required, because its value is "
                    "always computable; set required=False"
                )
            if self.name in self.derivation.inputs:
                raise ValueError(f"{self.name}: derivation reads the parameter it defines")
            if self.sweep is not None:
                raise ValueError(f"{self.name}: a derived parameter is not swept; sweep the parameter it derives from")
        if self.sweep is not None and self.enum is not None:
            outside = [value for value in self.sweep.values if value not in self.enum]
            if outside:
                raise ValueError(f"{self.name}: swept values {outside!r} fall outside the enum")
        keys = [(sentinel.kind, sentinel.value) for sentinel in self.sentinels]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.name}: declares the same sentinel twice")
        return self


def _derivation_order(specs: list[ParameterSpec]) -> None:
    """reject a cycle among derived parameters.

    A cycle makes the value unresolvable, and a resolver that discovers it
    at run time discovers it after the build has already started.

    :param specs: declared parameters
    :ptype specs: list[ParameterSpec]
    :returns: None
    :rtype: None
    :raises ValueError: derivations form a cycle
    """
    edges = {spec.name: list(spec.derivation.inputs) for spec in specs if spec.derivation is not None}
    pending = set(edges)
    while pending:
        ready = {name for name in pending if all(dep not in pending for dep in edges[name])}
        if not ready:
            raise ValueError(
                f"parameter derivations form a cycle among {sorted(pending)}; a derived value "
                "that reads itself can never resolve"
            )
        pending -= ready


def validate_parameter_specs(specs: list[ParameterSpec]) -> list[ParameterSpec]:
    """check definition-level invariants over a whole parameter list.

    :param specs: declared parameters, in authored order
    :ptype specs: list[ParameterSpec]
    :returns: the same list, unchanged
    :rtype: list[ParameterSpec]
    :raises ValueError: a name repeats, a derivation or constraint names
        an undeclared parameter, or derivations form a cycle
    """
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"parameter names repeat: {duplicates}")
    declared = set(names)
    for spec in specs:
        if spec.derivation is not None:
            missing = [name for name in spec.derivation.inputs if name not in declared]
            if missing:
                raise ValueError(f"{spec.name}: derivation reads undeclared parameters {missing}")
        for constraint in spec.constraints:
            if constraint.requires_parameter not in declared:
                raise ValueError(
                    f"{spec.name}: constraint names undeclared parameter {constraint.requires_parameter!r}"
                )
    _derivation_order(specs)
    return specs


ParameterSpecList: TypeAlias = Annotated[list[ParameterSpec], AfterValidator(validate_parameter_specs)]
"""parameter list carrying its definition-level invariants."""


def validate_parameter_values(specs: list[ParameterSpec], values: dict[str, ScalarValue]) -> None:
    """check resolved run values against their declarations.

    Runs three checks in one place, so a build cannot start on a
    parameter set that is internally inconsistent:

    1. every required, non-derived parameter carrying no default has a value
    2. every value falls inside its declared enumeration
    3. every armed cross-parameter constraint is satisfied

    :param specs: declared parameters
    :ptype specs: list[ParameterSpec]
    :param values: resolved parameter values for one run
    :ptype values: dict[str, ScalarValue]
    :returns: None
    :rtype: None
    :raises ParameterValueRejected: a required value is absent or a value
        falls outside its enumeration
    :raises ParameterConstraintViolated: an armed cross-parameter
        constraint is unsatisfied
    """
    by_name = {spec.name: spec for spec in specs}
    for spec in specs:
        supplied = values.get(spec.name, spec.default)
        if spec.required and spec.derivation is None and supplied is None:
            raise ParameterValueRejected(f"required parameter {spec.name!r} has no value")
        if spec.enum is not None and supplied is not None and supplied not in spec.enum:
            raise ParameterValueRejected(
                f"parameter {spec.name!r} = {supplied!r} falls outside its declared enum {list(spec.enum)!r}"
            )
    for spec in specs:
        armed_value = values.get(spec.name, spec.default)
        for constraint in spec.constraints:
            if armed_value != constraint.when_value:
                continue
            other = by_name.get(constraint.requires_parameter)
            other_value = values.get(constraint.requires_parameter, other.default if other else None)
            if other_value not in constraint.requires_one_of:
                note = f" ({constraint.message})" if constraint.message else ""
                raise ParameterConstraintViolated(
                    f"parameter {spec.name!r} = {armed_value!r} requires parameter "
                    f"{constraint.requires_parameter!r} to be one of "
                    f"{list(constraint.requires_one_of)!r}; got {other_value!r}{note}"
                )
