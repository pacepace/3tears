"""unit tests for :class:`ParameterSpec` and its enums, derivations, constraints, sentinels, and sweep.

three corpus semantics drive this module:

- ``--tsmart_comm`` is valid only with the TargetSmart voter file. that
  constraint lives in an argparse help string today and nothing checks
  it, so violating it produces a SMALLER audience with no error.
- ``record_year`` carries two opposite sentinels. ``-1`` makes the
  working-age ceiling a no-op (``age < 2096``); ``NULL::int`` drops every
  row of the unit. both are silent and both must be declarable.
- ``pull_householder_counts.sql`` is a hand-unrolled sweep, ten arms over
  thresholds 1 through 10 for one knob, projecting the swept value as a
  labelling column.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from threetears.datasources.definition import (
    ParameterConstraint,
    ParameterConstraintViolated,
    ParameterSpec,
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


def _release_schema() -> ParameterSpec:
    return ParameterSpec(name="release_schema", parameter_type=ParameterType.STRING)


def _vf_suffix() -> ParameterSpec:
    return ParameterSpec(
        name="vf_suffix",
        parameter_type=ParameterType.STRING,
        required=False,
        enum=["l2", "ts"],
    )


def _match_schema() -> ParameterSpec:
    return ParameterSpec(
        name="match_schema",
        parameter_type=ParameterType.STRING,
        required=False,
        derivation=TemplateDerivation(template="{release_schema}_{vf_suffix}", fallback="{release_schema}"),
    )


def _analytics_schema() -> ParameterSpec:
    return ParameterSpec(
        name="analytics_schema",
        parameter_type=ParameterType.STRING,
        required=False,
        derivation=TemplateDerivation(template="{match_schema}_analytics"),
    )


def _tsmart_comm() -> ParameterSpec:
    return ParameterSpec(
        name="tsmart_comm",
        parameter_type=ParameterType.BOOLEAN,
        required=False,
        default=False,
        constraints=[
            ParameterConstraint(
                when_value=True,
                requires_parameter="vf_suffix",
                requires_one_of=["ts"],
                message="the commercial file is only published for the TargetSmart voter file",
            )
        ],
    )


class TestParameterEnumeration:
    """``vf_suffix`` carries the corpus's only enumerated parameter."""

    def test_enum_is_declarable(self) -> None:
        assert _vf_suffix().enum == ["l2", "ts"]

    def test_empty_enum_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ParameterSpec(name="vf_suffix", parameter_type=ParameterType.STRING, enum=[])

    def test_duplicate_enum_members_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ParameterSpec(name="vf_suffix", parameter_type=ParameterType.STRING, enum=["ts", "ts"])

    def test_default_outside_the_enum_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ParameterSpec(
                name="vf_suffix",
                parameter_type=ParameterType.STRING,
                enum=["l2", "ts"],
                default="wolverine",
            )
        assert "wolverine" in str(excinfo.value)

    def test_value_outside_the_enum_is_rejected_at_binding(self) -> None:
        with pytest.raises(ParameterValueRejected) as excinfo:
            validate_parameter_values(
                [_release_schema(), _vf_suffix()], {"release_schema": "anhinga", "vf_suffix": "l3"}
            )
        assert "vf_suffix" in str(excinfo.value)


class TestParameterDerivation:
    """one schema computed from another, and a run year computed from a date."""

    def test_template_inputs_are_scanned_from_the_template(self) -> None:
        derivation = TemplateDerivation(template="{release_schema}_{vf_suffix}", fallback="{release_schema}")
        assert derivation.inputs == ("release_schema", "vf_suffix")

    def test_two_level_derivation(self) -> None:
        assert _analytics_schema().derivation is not None
        specs = [_release_schema(), _vf_suffix(), _match_schema(), _analytics_schema()]
        assert validate_parameter_specs(specs) == specs

    def test_substring_derivation_for_the_run_year(self) -> None:
        derivation = SubstringDerivation(source="date", start=0, length=4)
        assert derivation.inputs == ("date",)

    def test_template_with_no_placeholder_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TemplateDerivation(template="anhinga_analytics")

    def test_template_with_a_non_identifier_placeholder_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TemplateDerivation(template="{release schema}_x")

    def test_a_derived_parameter_may_not_be_required(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ParameterSpec(
                name="match_schema",
                parameter_type=ParameterType.STRING,
                required=True,
                derivation=TemplateDerivation(template="{release_schema}_{vf_suffix}"),
            )
        assert "match_schema" in str(excinfo.value)

    def test_self_derivation_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ParameterSpec(
                name="match_schema",
                parameter_type=ParameterType.STRING,
                required=False,
                derivation=TemplateDerivation(template="{match_schema}_x"),
            )
        assert "match_schema" in str(excinfo.value)

    def test_derivation_naming_an_undeclared_parameter_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            validate_parameter_specs([_match_schema()])
        assert "release_schema" in str(excinfo.value)

    def test_derivation_cycle_is_rejected(self) -> None:
        left = ParameterSpec(
            name="left",
            parameter_type=ParameterType.STRING,
            required=False,
            derivation=TemplateDerivation(template="{right}_a"),
        )
        right = ParameterSpec(
            name="right",
            parameter_type=ParameterType.STRING,
            required=False,
            derivation=TemplateDerivation(template="{left}_b"),
        )
        with pytest.raises(ValueError) as excinfo:
            validate_parameter_specs([left, right])
        assert "cycle" in str(excinfo.value).lower()


class TestCrossParameterConstraints:
    """the commercial-file flag is valid for exactly one voter-file vendor."""

    def test_constraint_is_expressible(self) -> None:
        constraint = _tsmart_comm().constraints[0]
        assert constraint.requires_parameter == "vf_suffix"
        assert constraint.requires_one_of == ["ts"]

    def test_violation_names_both_parameters(self) -> None:
        specs = [_release_schema(), _vf_suffix(), _tsmart_comm()]
        with pytest.raises(ParameterConstraintViolated) as excinfo:
            validate_parameter_values(specs, {"release_schema": "anhinga", "vf_suffix": "l2", "tsmart_comm": True})
        message = str(excinfo.value)
        assert "tsmart_comm" in message
        assert "vf_suffix" in message

    def test_violation_carries_the_authored_message(self) -> None:
        specs = [_release_schema(), _vf_suffix(), _tsmart_comm()]
        with pytest.raises(ParameterConstraintViolated) as excinfo:
            validate_parameter_values(specs, {"release_schema": "anhinga", "vf_suffix": "l2", "tsmart_comm": True})
        assert "TargetSmart" in str(excinfo.value)

    def test_satisfied_constraint_passes(self) -> None:
        specs = [_release_schema(), _vf_suffix(), _tsmart_comm()]
        validate_parameter_values(specs, {"release_schema": "anhinga", "vf_suffix": "ts", "tsmart_comm": True})

    def test_unarmed_constraint_passes(self) -> None:
        specs = [_release_schema(), _vf_suffix(), _tsmart_comm()]
        validate_parameter_values(specs, {"release_schema": "anhinga", "vf_suffix": "l2", "tsmart_comm": False})

    def test_constraint_naming_an_undeclared_parameter_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            validate_parameter_specs([_release_schema(), _tsmart_comm()])
        assert "vf_suffix" in str(excinfo.value)

    def test_constraint_requires_a_non_empty_alternative_set(self) -> None:
        with pytest.raises(ValidationError):
            ParameterConstraint(when_value=True, requires_parameter="vf_suffix", requires_one_of=[])

    def test_missing_required_parameter_is_rejected(self) -> None:
        with pytest.raises(ParameterValueRejected) as excinfo:
            validate_parameter_values([_release_schema()], {})
        assert "release_schema" in str(excinfo.value)


class TestSentinelDomains:
    """both record_year sentinels, and they are opposite failures."""

    def test_minus_one_sentinel_widens_the_predicate(self) -> None:
        sentinel = SentinelDomain(
            kind=SentinelKind.VALUE,
            value=-1,
            meaning="unknown record vintage",
            effect=SentinelEffect.WIDENS_PREDICATE,
        )
        assert sentinel.value == -1
        assert sentinel.effect is SentinelEffect.WIDENS_PREDICATE

    def test_null_sentinel_drops_the_row(self) -> None:
        sentinel = SentinelDomain(
            kind=SentinelKind.NULL,
            meaning="bridge-only resolution has no fact vintage",
            effect=SentinelEffect.DROPS_ROW,
        )
        assert sentinel.value is None
        assert sentinel.effect is SentinelEffect.DROPS_ROW

    def test_value_sentinel_requires_a_value(self) -> None:
        with pytest.raises(ValidationError):
            SentinelDomain(kind=SentinelKind.VALUE, meaning="unknown", effect=SentinelEffect.WIDENS_PREDICATE)

    def test_null_sentinel_may_not_carry_a_value(self) -> None:
        with pytest.raises(ValidationError):
            SentinelDomain(
                kind=SentinelKind.NULL,
                value=-1,
                meaning="unknown",
                effect=SentinelEffect.DROPS_ROW,
            )

    def test_a_parameter_declares_both_sentinels(self) -> None:
        spec = ParameterSpec(
            name="record_year",
            parameter_type=ParameterType.INTEGER,
            required=False,
            sentinels=[
                SentinelDomain(
                    kind=SentinelKind.VALUE,
                    value=-1,
                    meaning="unknown record vintage",
                    effect=SentinelEffect.WIDENS_PREDICATE,
                ),
                SentinelDomain(
                    kind=SentinelKind.NULL,
                    meaning="bridge-only resolution has no fact vintage",
                    effect=SentinelEffect.DROPS_ROW,
                ),
            ],
        )
        assert {sentinel.effect for sentinel in spec.sentinels} == {
            SentinelEffect.WIDENS_PREDICATE,
            SentinelEffect.DROPS_ROW,
        }

    def test_duplicate_sentinels_on_a_parameter_are_rejected(self) -> None:
        duplicate = SentinelDomain(kind=SentinelKind.NULL, meaning="unknown", effect=SentinelEffect.DROPS_ROW)
        with pytest.raises(ValidationError):
            ParameterSpec(
                name="record_year",
                parameter_type=ParameterType.INTEGER,
                required=False,
                sentinels=[duplicate, duplicate],
            )

    def test_sentinel_binding_targets_a_namespaced_column(self) -> None:
        binding = SentinelBinding.model_validate(
            {
                "target": "resolved.record_year",
                "sentinels": [
                    {
                        "kind": "value",
                        "value": -1,
                        "meaning": "unknown record vintage",
                        "effect": "widens_predicate",
                    },
                    {
                        "kind": "null",
                        "meaning": "bridge-only resolution has no fact vintage",
                        "effect": "drops_row",
                    },
                ],
            }
        )
        assert binding.target.ref == "resolved.record_year"
        assert len(binding.sentinels) == 2

    def test_sentinel_binding_requires_at_least_one_sentinel(self) -> None:
        with pytest.raises(ValidationError):
            SentinelBinding.model_validate({"target": "resolved.record_year", "sentinels": []})


class TestParameterSweep:
    """one definition, N parameter values, one stacked result."""

    def test_ten_arm_threshold_sweep(self) -> None:
        spec = ParameterSpec(
            name="vb_candidates",
            parameter_type=ParameterType.INTEGER,
            required=False,
            default=10,
            sweep=ParameterSweep(values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], emit_column="candidate_count_cutoff"),
        )
        assert spec.sweep is not None
        assert len(spec.sweep.values) == 10
        assert spec.sweep.emit_column == "candidate_count_cutoff"

    def test_sweep_values_must_be_unique(self) -> None:
        with pytest.raises(ValidationError):
            ParameterSweep(values=[1, 1], emit_column="cutoff")

    def test_sweep_values_must_satisfy_the_enum(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ParameterSpec(
                name="vf_suffix",
                parameter_type=ParameterType.STRING,
                required=False,
                enum=["l2", "ts"],
                sweep=ParameterSweep(values=["l2", "l3"], emit_column="suffix"),
            )
        assert "l3" in str(excinfo.value)

    def test_a_derived_parameter_may_not_be_swept(self) -> None:
        with pytest.raises(ValidationError):
            ParameterSpec(
                name="match_schema",
                parameter_type=ParameterType.STRING,
                required=False,
                derivation=TemplateDerivation(template="{release_schema}_{vf_suffix}"),
                sweep=ParameterSweep(values=["a", "b"], emit_column="schema_arm"),
            )

    def test_emit_column_must_be_an_identifier(self) -> None:
        with pytest.raises(ValidationError):
            ParameterSweep(values=[1, 2], emit_column="candidate count cutoff")


class TestParameterSpecList:
    """definition-level invariants over the whole parameter list."""

    def test_duplicate_parameter_names_are_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            validate_parameter_specs([_release_schema(), _release_schema()])
        assert "release_schema" in str(excinfo.value)

    def test_the_corpus_parameter_set_validates(self) -> None:
        run_date = ParameterSpec(name="date", parameter_type=ParameterType.DATE)
        run_year = ParameterSpec(
            name="run_year",
            parameter_type=ParameterType.INTEGER,
            required=False,
            derivation=SubstringDerivation(source="date", start=0, length=4),
        )
        specs = [
            _release_schema(),
            _vf_suffix(),
            _match_schema(),
            _analytics_schema(),
            _tsmart_comm(),
            run_date,
            run_year,
            ParameterSpec(name="target_schema", parameter_type=ParameterType.STRING),
        ]
        assert validate_parameter_specs(specs) == specs


class TestParameterSpecShape:
    """field-level posture."""

    def test_extra_fields_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ParameterSpec.model_validate({"name": "release_schema", "parameter_type": "string", "choices": ["a"]})

    def test_name_must_be_an_identifier(self) -> None:
        with pytest.raises(ValidationError):
            ParameterSpec(name="release schema", parameter_type=ParameterType.STRING)

    def test_decimal_default_is_not_a_float(self) -> None:
        spec = ParameterSpec(name="threshold", parameter_type=ParameterType.DECIMAL, default=Decimal("1.5"))
        assert not isinstance(spec.default, float)
