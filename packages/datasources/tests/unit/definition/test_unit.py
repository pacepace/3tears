"""unit tests for :class:`Unit`, :class:`Resolution`, and :class:`Qualification`.

three corpus semantics drive this module:

- a unit is a SET of resolutions. ``uhg_opinion_elites`` declares
  ``journalists_health_policy`` twice with different sources and the
  emitted SQL unions both under one label, while the prototype's metadata
  dict keeps only the last -- so its xtabs already under-report.
- a duplicate ``Unit.name`` is a parse error. one name means "the union
  of two" in SQL and "the second one" in every reporting path, and no
  migration should have to choose silently.
- ``source.*`` and ``bridge.*`` are structurally unbindable at
  qualification. they name pre-aggregate rows that no longer exist at
  that stage.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.definition import (
    DuplicateUnitName,
    Qualification,
    Resolution,
    Unit,
    UnqualifiedUnits,
    units_without_qualification,
    validate_qualification_coverage,
    validate_unique_unit_names,
)


class TestResolutionSet:
    """``Unit.resolutions`` is a list, and duplicates are not merged."""

    def test_duplicate_declarations_become_two_resolutions(self) -> None:
        unit = Unit.model_validate(
            {"name": "journalists_health_policy", "resolutions": [{"source": "a"}, {"source": "b"}]}
        )
        assert len(unit.resolutions) == 2

    def test_a_unit_needs_at_least_one_resolution(self) -> None:
        with pytest.raises(ValidationError):
            Unit.model_validate({"name": "x", "resolutions": []})

    def test_bridge_only_resolution_has_no_source(self) -> None:
        resolution = Resolution.model_validate(
            {"bridge": {"relation": "match_union", "alias": "mat"}},
        )
        assert resolution.source is None

    def test_resolution_with_no_predicate_is_legal(self) -> None:
        assert Resolution().predicate is None


class TestUnitNameAndEmits:
    """the logical name is free-form; the identifier is derived, never authored."""

    @pytest.mark.parametrize(
        "name",
        [
            "fec_contribution>2000",
            "fec_contributions_republican>10",
            "ai company tech employees",
            "tech decision makers at all companies",
            "existing knowwho fed_exec_agency",
            "café_goers",
        ],
    )
    def test_free_form_names_are_accepted(self, name: str) -> None:
        assert Unit(name=name, resolutions=[Resolution()]).name == name

    def test_empty_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Unit(name="", resolutions=[Resolution()])

    def test_there_is_no_physical_identifier_field(self) -> None:
        forbidden = {"column_identifier", "identifier", "physical_name", "column_name"}
        assert forbidden.isdisjoint(set(Unit.model_fields))

    def test_a_physical_identifier_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Unit.model_validate(
                {
                    "name": "fec_contribution>2000",
                    "resolutions": [{}],
                    "column_identifier": "fec_contribution_2000",
                }
            )

    def test_emits_defaults_to_the_name(self) -> None:
        unit = Unit(name="business_executives", resolutions=[Resolution()])
        assert unit.emits is None
        assert unit.emitted_labels == ("business_executives",)

    def test_emits_admits_two_labels(self) -> None:
        unit = Unit(
            name="knowwho_leg",
            emits=["federal_legislators", "state_legislators"],
            resolutions=[Resolution()],
        )
        assert unit.emitted_labels == ("federal_legislators", "state_legislators")

    def test_empty_emits_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Unit(name="knowwho_leg", emits=[], resolutions=[Resolution()])

    def test_duplicate_emits_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Unit(name="knowwho_leg", emits=["a", "a"], resolutions=[Resolution()])


class TestDuplicateUnitNames:
    """a duplicate name is a parse error, not a silent overwrite."""

    def test_unique_names_pass(self) -> None:
        units = [
            Unit(name="a", resolutions=[Resolution()]),
            Unit(name="b", resolutions=[Resolution()]),
        ]
        assert validate_unique_unit_names(units) == units

    def test_duplicate_name_raises(self) -> None:
        units = [
            Unit(name="ai company tech employees", resolutions=[Resolution()]),
            Unit(name="ai company tech employees", resolutions=[Resolution()]),
        ]
        with pytest.raises(DuplicateUnitName) as excinfo:
            validate_unique_unit_names(units)
        assert "ai company tech employees" in str(excinfo.value)

    def test_duplicate_across_emitted_labels_raises(self) -> None:
        units = [
            Unit(name="knowwho_leg", emits=["federal_legislators"], resolutions=[Resolution()]),
            Unit(name="federal_legislators", resolutions=[Resolution()]),
        ]
        with pytest.raises(DuplicateUnitName):
            validate_unique_unit_names(units)


class TestQualificationScoping:
    """authored per unit, declarable over a named unit set."""

    def test_applies_to_defaults_to_the_owning_unit(self) -> None:
        assert Qualification().applies_to is None

    def test_applies_to_names_a_unit_set(self) -> None:
        qualification = Qualification.model_validate(
            {
                "name": "candidate_count_tier_6",
                "applies_to": ["doj", "fda", "hhs", "sec"],
                "predicate": {"compare": {"left": "resolved.candidate_count", "op": "<=", "right": 6}},
            }
        )
        assert qualification.name == "candidate_count_tier_6"
        assert qualification.applies_to == ["doj", "fda", "hhs", "sec"]

    def test_empty_applies_to_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Qualification(applies_to=[])

    def test_duplicate_applies_to_entries_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Qualification(applies_to=["doj", "doj"])

    def test_qualification_carries_relations(self) -> None:
        """the unconditional entity join is a relation, not a predicate.

        ``voter_file_segmented`` is INNER-joined at qualification with no
        authored predicate mentioning it, so any entity absent from the
        voter file is dropped from the qualified set. That is a definition
        of the set rather than a filter over it, and a ``Qualification``
        carrying only a predicate cannot express it.
        """
        qualification = Qualification.model_validate(
            {
                "relations": [
                    {"relation": "voter_file_segmented", "alias": "vf", "join": "inner"},
                ],
            },
        )
        assert [r.relation for r in qualification.relations] == ["voter_file_segmented"]
        assert qualification.relations[0].alias == "vf"


class TestQualificationNamespaceGuard:
    """``source.*`` and ``bridge.*`` are refused at qualification."""

    def test_source_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Qualification.model_validate(
                {"predicate": {"compare": {"left": "source.job_title", "op": "=", "right": {"literal": "x"}}}}
            )
        message = str(excinfo.value)
        assert "source" in message
        assert "qualification" in message

    def test_bridge_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Qualification.model_validate(
                {"predicate": {"compare": {"left": "bridge.candidate_count", "op": "<=", "right": 10}}}
            )
        assert "bridge" in str(excinfo.value)

    def test_measure_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Qualification.model_validate(
                {"predicate": {"compare": {"left": "measure.sum_of_contributions", "op": ">", "right": 2000}}}
            )

    def test_source_inside_an_arithmetic_expression_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Qualification.model_validate(
                {
                    "predicate": {
                        "compare": {
                            "left": "entity.vb_voterbase_age",
                            "op": "<",
                            "right": {"arith": "70 + param.run_year - source.record_year"},
                        }
                    }
                }
            )

    def test_the_working_age_filter_parses(self) -> None:
        qualification = Qualification.model_validate(
            {
                "predicate": {
                    "all_of": [
                        {"compare": {"left": "entity.vb_voterbase_age", "op": ">", "right": 20}},
                        {
                            "compare": {
                                "left": "entity.vb_voterbase_age",
                                "op": "<",
                                "right": {"arith": "70 + param.run_year - resolved.record_year"},
                            }
                        },
                    ]
                }
            }
        )
        assert qualification.predicate is not None

    def test_resolved_and_rel_are_bindable(self) -> None:
        qualification = Qualification.model_validate(
            {
                "predicate": {
                    "all_of": [
                        {"compare": {"left": "resolved.unit", "op": "=", "right": {"literal": "doj"}}},
                        {"compare": {"left": "rel.sa.source_archetype", "op": "IS NOT NULL"}},
                    ]
                }
            }
        )
        assert qualification.predicate is not None


class TestResolutionNamespaceGuard:
    """``resolved.*`` and ``measure.*`` are refused in a resolution predicate."""

    def test_resolved_is_refused_in_a_resolution_predicate(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Resolution.model_validate(
                {"predicate": {"compare": {"left": "resolved.candidate_count", "op": "<=", "right": 10}}}
            )
        message = str(excinfo.value)
        assert "resolved" in message
        assert "resolution" in message

    def test_measure_is_refused_in_a_resolution_predicate(self) -> None:
        with pytest.raises(ValidationError):
            Resolution.model_validate(
                {"predicate": {"compare": {"left": "measure.contribution_sum", "op": ">", "right": 2000}}}
            )

    def test_measure_is_bindable_in_having(self) -> None:
        """``measure.<name>`` binds in having, and only for a declared one.

        The measure has to exist on the resolution: a having naming one
        nothing computes is an authoring error the warehouse would
        otherwise report as an unresolved identifier at build time, after
        the schema is already half built.
        """
        resolution = Resolution.model_validate(
            {
                "measures": [
                    {
                        "name": "contribution_sum",
                        "expression": "SUM(source.contribution::float * 1.0 / bridge.candidate_count::float)",
                        "grain": ["voterbase_id", "list_id"],
                        "scope": "resolution",
                    }
                ],
                "having": {"compare": {"left": "measure.contribution_sum", "op": ">", "right": 2000}},
            }
        )
        assert resolution.having is not None

    def test_having_naming_an_undeclared_measure_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Resolution.model_validate(
                {"having": {"compare": {"left": "measure.contribution_sum", "op": ">", "right": 2000}}}
            )
        assert "contribution_sum" in str(excinfo.value)

    def test_resolved_is_refused_in_having(self) -> None:
        with pytest.raises(ValidationError):
            Resolution.model_validate(
                {"having": {"compare": {"left": "resolved.candidate_count", "op": "<=", "right": 10}}}
            )

    def test_source_bridge_entity_rel_param_are_bindable_in_a_resolution(self) -> None:
        resolution = Resolution.model_validate(
            {
                "predicate": {
                    "all_of": [
                        {"compare": {"left": "source.job_title", "op": "ILIKE", "right": {"literal": "%producer%"}}},
                        {"compare": {"left": "bridge.source_name", "op": "=", "right": {"literal": "knowwho_exec"}}},
                        {"compare": {"left": "entity.vb_voterbase_age", "op": "<=", "right": 75}},
                        {"compare": {"left": "rel.edu.institution", "op": "IS NOT NULL"}},
                        {"compare": {"left": "param.release_schema", "op": "IS NOT NULL"}},
                    ]
                }
            }
        )
        assert resolution.predicate is not None


class TestQualificationCoverage:
    """a unit named by no qualification is silent data loss."""

    def test_a_unit_named_by_no_qualification_is_reported(self) -> None:
        qualifications = [Qualification(applies_to=["doj", "fda"])]
        assert units_without_qualification(["doj", "fda", "department_of_commerce"], qualifications) == (
            "department_of_commerce",
        )

    def test_coverage_validation_raises_naming_the_unit(self) -> None:
        qualifications = [Qualification(applies_to=["doj", "fda"])]
        with pytest.raises(UnqualifiedUnits) as excinfo:
            validate_qualification_coverage(["doj", "fda", "department_of_commerce"], qualifications)
        assert "department_of_commerce" in str(excinfo.value)

    def test_no_qualification_at_all_is_legal(self) -> None:
        validate_qualification_coverage(["doj", "fda"], [])

    def test_an_unscoped_qualification_covers_every_unit(self) -> None:
        validate_qualification_coverage(["doj", "fda"], [Qualification(applies_to=None)])


class TestModelShape:
    """field-level posture."""

    def test_unit_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Unit.model_validate({"name": "x", "resolutions": [{}], "rollup_unit": "tech media"})

    def test_resolution_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Resolution.model_validate({"facts_table": "employment"})

    def test_qualification_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Qualification.model_validate({"vb_candidates": 10})

    def test_unit_round_trips(self) -> None:
        """the corpus's duplicate-name unit round-trips through both seams.

        ``journalists_health_policy`` is declared twice with different
        sources and joins, and both bodies emit, so it is the shape that
        forces ``resolutions`` to be a list. Carrying the real
        ``BridgeRef`` and ``RelationRef`` here rather than a name keeps
        the round-trip honest about what a resolution actually holds.
        """
        bridge = {
            "relation": "match_union",
            "alias": "mat",
            "quality_measures": [
                {
                    "name": "candidate_count",
                    "column": "candidate_count",
                    "direction": "lower_is_better",
                    "threshold_semantics": "at_most",
                    "unmeasured_is_null": True,
                }
            ],
            "on": None,
        }
        payload = {
            "name": "journalists_health_policy",
            "emits": None,
            "resolutions": [
                {
                    "source": "employment_facts",
                    "bridge": bridge,
                    "relations": [
                        {
                            "relation": "education_ext",
                            "alias": "edu",
                            "join": "left",
                            "on": None,
                            "optional": True,
                            "when": None,
                        }
                    ],
                    "measures": [],
                    "predicate": None,
                    "having": None,
                },
                {
                    "source": None,
                    "bridge": bridge,
                    "relations": [],
                    "measures": [],
                    "predicate": None,
                    "having": None,
                },
            ],
            "qualify": None,
            "exclude": None,
        }
        assert Unit.model_validate(payload).model_dump() == payload
