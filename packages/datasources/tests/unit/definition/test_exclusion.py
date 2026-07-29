"""unit tests for :class:`ExclusionSpec` and the ``all_prior`` expansion.

Four corpus semantics drive this module:

- the residual triple is authored, never defaulted. ``stage``,
  ``key_columns``, and ``level`` each silently pick a different audience,
  and the prototype picked one of each by accident.
- ``all_prior`` expands to explicit edges at parse time. Left as a
  runtime notion, "prior" is ambiguous between authored order and a
  topological order free to reorder, and the two give different
  audiences with no error.
- D7b's direction is expressible: ``omnibus_other_sources_non_overlap``
  is authored FIRST, so it does not subtract
  ``omnibus_other_sources_overlap`` while ``overlap`` does subtract it.
- exclusion happens at TWO stages in the committed corpus -- the
  residual units anti-join the RESOLVED long table while
  ``exclude_existing`` anti-joins the QUALIFIED set -- which is why
  ``stage`` cannot carry a default.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.definition import (
    ArtifactKind,
    ArtifactRef,
    ArtifactScope,
    ArtifactStage,
    ExclusionLevel,
    ExclusionSpec,
    UnexpandedExclusion,
    expand_all_prior,
    reject_unexpanded_exclusions,
)

_RESIDUAL = {
    "all_prior": True,
    "stage": "resolved",
    "key_columns": ["voterbase_id"],
    "level": "pre_aggregate",
}


class TestSubtrahendReference:
    """what an exclusion subtracts, named by ``ArtifactRef``.

    ``dsm-task-01d`` replaced ``dsm-task-01b``'s narrow ``ArtifactHandle``
    with the one reference type, so a subtrahend, a membership target, and
    a relation body are the same kind of thing. The cases below are the
    three the corpus's exclusion rows exercise; the reference's own scope
    rules are in ``test_source.py``.
    """

    def test_a_unit_of_this_definition_carries_its_stage(self) -> None:
        handle = ArtifactRef.model_validate({"scope": "this_definition", "unit": "knowwho_all", "stage": "resolved"})
        assert handle.unit == "knowwho_all"
        assert handle.stage is ArtifactStage.RESOLVED

    def test_an_upstream_dataset_is_nameable(self) -> None:
        handle = ArtifactRef.model_validate(
            {"scope": "dataset", "dataset": "universal_2026_core", "artifact": "qualified"}
        )
        assert handle.dataset == "universal_2026_core"
        assert handle.artifact is ArtifactKind.QUALIFIED
        assert handle.unit is None

    def test_an_external_table_is_nameable(self) -> None:
        assert ArtifactRef.model_validate({"scope": "external", "table": "ehowells.amz_universe_2024"}).table

    def test_rejects_naming_nothing(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate({"scope": "this_definition"})

    def test_rejects_naming_two_scopes_at_once(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate({"scope": "this_definition", "unit": "a", "stage": "resolved", "dataset": "b"})

    def test_a_unit_handle_requires_a_stage(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ArtifactRef.model_validate({"scope": "this_definition", "unit": "a"})
        assert "stage" in str(excinfo.value)

    def test_a_dataset_handle_carries_no_stage(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate({"scope": "dataset", "dataset": "a", "artifact": "wide", "stage": "resolved"})

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactRef.model_validate({"scope": "this_definition", "unit": "a", "stage": "resolved", "kind": "unit"})


class TestTheResidualTripleHasNoDefaults:
    """D7a: ``stage``, ``key_columns``, and ``level`` are required, always."""

    @pytest.mark.parametrize("missing", ["stage", "key_columns", "level"])
    def test_omitting_any_one_fails_validation(self, missing: str) -> None:
        payload = dict(_RESIDUAL)
        payload.pop(missing)
        with pytest.raises(ValidationError):
            ExclusionSpec.model_validate(payload)

    @pytest.mark.parametrize("missing", ["stage", "key_columns", "level"])
    def test_the_message_names_the_consequence(self, missing: str) -> None:
        payload = dict(_RESIDUAL)
        payload.pop(missing)
        with pytest.raises(ValidationError) as excinfo:
            ExclusionSpec.model_validate(payload)
        message = str(excinfo.value)
        assert missing in message
        assert "audience" in message

    @pytest.mark.parametrize("field_name", ["stage", "key_columns", "level"])
    def test_the_field_carries_no_default(self, field_name: str) -> None:
        field = ExclusionSpec.model_fields[field_name]
        assert field.is_required()

    def test_all_prior_does_default_and_defaults_to_false(self) -> None:
        assert ExclusionSpec.model_fields["all_prior"].default is False


class TestExclusionSpecShape:
    """the four authored properties, and what each one changes."""

    def test_both_stages_are_expressible(self) -> None:
        resolved = ExclusionSpec.model_validate(_RESIDUAL)
        qualified = ExclusionSpec.model_validate({**_RESIDUAL, "stage": "qualified"})
        assert resolved.stage is ArtifactStage.RESOLVED
        assert qualified.stage is ArtifactStage.QUALIFIED

    def test_both_levels_are_expressible(self) -> None:
        pre = ExclusionSpec.model_validate(_RESIDUAL)
        post = ExclusionSpec.model_validate({**_RESIDUAL, "level": "post_aggregate"})
        assert pre.level is ExclusionLevel.PRE_AGGREGATE
        assert post.level is ExclusionLevel.POST_AGGREGATE

    def test_entity_and_record_keys_are_different_declarations(self) -> None:
        entity = ExclusionSpec.model_validate(_RESIDUAL)
        record = ExclusionSpec.model_validate({**_RESIDUAL, "key_columns": ["unit", "list_id", "voterbase_id"]})
        assert entity.key_columns != record.key_columns

    def test_two_specs_differing_only_in_stage_are_not_equal(self) -> None:
        resolved = ExclusionSpec.model_validate(_RESIDUAL)
        qualified = ExclusionSpec.model_validate({**_RESIDUAL, "stage": "qualified"})
        assert resolved != qualified
        assert resolved.model_dump() != qualified.model_dump()

    def test_rejects_an_empty_key(self) -> None:
        with pytest.raises(ValidationError):
            ExclusionSpec.model_validate({**_RESIDUAL, "key_columns": []})

    def test_rejects_a_repeated_key_column(self) -> None:
        with pytest.raises(ValidationError):
            ExclusionSpec.model_validate({**_RESIDUAL, "key_columns": ["voterbase_id", "voterbase_id"]})

    def test_rejects_a_blank_key_column(self) -> None:
        with pytest.raises(ValidationError):
            ExclusionSpec.model_validate({**_RESIDUAL, "key_columns": [" "]})

    def test_rejects_subtracting_nothing(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ExclusionSpec.model_validate(
                {"stage": "resolved", "key_columns": ["voterbase_id"], "level": "pre_aggregate"}
            )
        assert "subtracts nothing" in str(excinfo.value)

    def test_rejects_a_repeated_subtrahend(self) -> None:
        edge = {"scope": "this_definition", "unit": "a", "stage": "resolved"}
        with pytest.raises(ValidationError):
            ExclusionSpec.model_validate(
                {
                    "subtrahends": [edge, edge],
                    "stage": "resolved",
                    "key_columns": ["voterbase_id"],
                    "level": "pre_aggregate",
                }
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExclusionSpec.model_validate({**_RESIDUAL, "order": ["a", "b"]})

    def test_round_trips(self) -> None:
        payload = {
            "subtrahends": [{"scope": "this_definition", "unit": "a", "stage": "resolved"}],
            "all_prior": False,
            "key_columns": ["voterbase_id"],
            "level": "pre_aggregate",
            "stage": "resolved",
        }
        spec = ExclusionSpec.model_validate(payload)
        assert ExclusionSpec.model_validate(spec.model_dump(mode="json")) == spec
        assert spec.subtrahends[0].scope is ArtifactScope.THIS_DEFINITION


class TestUpstreamAndCohortSubtrahends:
    """the corpus subtracts upstream audiences and computed cohorts too."""

    def test_two_upstream_datasets_union_into_one_exclusion_set(self) -> None:
        spec = ExclusionSpec.model_validate(
            {
                "subtrahends": [
                    {"scope": "dataset", "dataset": "uhg_opinion_elites", "artifact": "qualified"},
                    {"scope": "dataset", "dataset": "uhg_policymakers", "artifact": "qualified"},
                ],
                "stage": "qualified",
                "key_columns": ["voterbase_id"],
                "level": "post_aggregate",
            }
        )
        assert [handle.dataset for handle in spec.subtrahends] == ["uhg_opinion_elites", "uhg_policymakers"]

    def test_a_post_composition_cohort_is_subtractable(self) -> None:
        spec = ExclusionSpec.model_validate(
            {
                "subtrahends": [{"scope": "external", "table": "eteitsworth.uhg_staff_20250618"}],
                "stage": "qualified",
                "key_columns": ["voterbase_id"],
                "level": "post_aggregate",
            }
        )
        assert spec.level is ExclusionLevel.POST_AGGREGATE


class TestAllPriorExpansion:
    """``all_prior`` becomes explicit edges, and the edges are inspectable."""

    def test_unexpanded_all_prior_is_visible_on_the_model(self) -> None:
        spec = ExclusionSpec.model_validate(_RESIDUAL)
        assert spec.all_prior is True
        assert spec.subtrahends == []
        assert spec.is_expanded is False

    def test_expansion_produces_one_edge_per_prior_unit(self) -> None:
        spec = ExclusionSpec.model_validate(_RESIDUAL).expanded_against(["a", "b", "c"])
        assert [handle.unit for handle in spec.subtrahends] == ["a", "b", "c"]
        assert spec.is_expanded is True

    def test_expanded_edges_carry_the_declared_stage(self) -> None:
        spec = ExclusionSpec.model_validate({**_RESIDUAL, "stage": "qualified"}).expanded_against(["a"])
        assert spec.subtrahends[0].stage is ArtifactStage.QUALIFIED

    def test_all_prior_survives_expansion_for_round_tripping(self) -> None:
        spec = ExclusionSpec.model_validate(_RESIDUAL).expanded_against(["a"])
        assert spec.all_prior is True

    def test_expansion_preserves_authored_subtrahends_first(self) -> None:
        spec = ExclusionSpec.model_validate(
            {**_RESIDUAL, "subtrahends": [{"scope": "dataset", "dataset": "amz_universe_2024", "artifact": "wide"}]}
        ).expanded_against(["a"])
        assert spec.subtrahends[0].dataset == "amz_universe_2024"
        assert spec.subtrahends[1].unit == "a"

    def test_expansion_does_not_duplicate_an_authored_edge(self) -> None:
        spec = ExclusionSpec.model_validate(
            {**_RESIDUAL, "subtrahends": [{"scope": "this_definition", "unit": "a", "stage": "resolved"}]}
        ).expanded_against(["a", "b"])
        assert [handle.unit for handle in spec.subtrahends] == ["a", "b"]

    def test_expansion_is_a_no_op_without_all_prior(self) -> None:
        spec = ExclusionSpec.model_validate(
            {
                **_RESIDUAL,
                "all_prior": False,
                "subtrahends": [{"scope": "this_definition", "unit": "a", "stage": "resolved"}],
            }
        )
        assert spec.expanded_against(["b", "c"]).subtrahends == spec.subtrahends

    def test_expansion_against_no_prior_units_leaves_it_unexpanded(self) -> None:
        spec = ExclusionSpec.model_validate(_RESIDUAL).expanded_against([])
        assert spec.subtrahends == []
        assert spec.is_expanded is False


class TestDefinitionLevelExpansion:
    """authored unit order is what "prior" means, and it is recorded."""

    def test_each_unit_subtracts_only_the_units_authored_before_it(self) -> None:
        expanded = expand_all_prior(
            ["first", "second", "third"],
            {"second": ExclusionSpec.model_validate(_RESIDUAL), "third": ExclusionSpec.model_validate(_RESIDUAL)},
        )
        assert [handle.unit for handle in expanded["second"].subtrahends] == ["first"]
        assert [handle.unit for handle in expanded["third"].subtrahends] == ["first", "second"]

    def test_d7b_direction_non_overlap_is_authored_first(self) -> None:
        units = [
            "omnibus_other_sources_non_overlap",
            "omnibus_other_sources_overlap",
            "executive_coworkers_of_linkedin_execs",
            "manager_coworkers_of_linkedin_execs",
        ]
        expanded = expand_all_prior(units, {name: ExclusionSpec.model_validate(_RESIDUAL) for name in units})
        non_overlap = [handle.unit for handle in expanded["omnibus_other_sources_non_overlap"].subtrahends]
        overlap = [handle.unit for handle in expanded["omnibus_other_sources_overlap"].subtrahends]
        assert non_overlap == []
        assert overlap == ["omnibus_other_sources_non_overlap"]

    def test_the_chosen_order_reaches_the_model_and_therefore_the_hash(self) -> None:
        units = ["non_overlap", "overlap"]
        specs = {name: ExclusionSpec.model_validate(_RESIDUAL) for name in units}
        forward = expand_all_prior(units, specs)
        reverse = expand_all_prior(list(reversed(units)), specs)
        assert forward["overlap"].model_dump() != reverse["overlap"].model_dump()

    def test_rejects_an_exclusion_owned_by_an_undeclared_unit(self) -> None:
        with pytest.raises(ValueError, match="undeclared"):
            expand_all_prior(["a"], {"b": ExclusionSpec.model_validate(_RESIDUAL)})

    def test_rejects_a_repeated_unit_name(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            expand_all_prior(["a", "a"], {"a": ExclusionSpec.model_validate(_RESIDUAL)})


class TestCompilerBoundaryGuard:
    """no unexpanded ``all_prior`` reaches the compiler."""

    def test_rejects_an_unexpanded_exclusion(self) -> None:
        with pytest.raises(UnexpandedExclusion) as excinfo:
            reject_unexpanded_exclusions({"omnibus": ExclusionSpec.model_validate(_RESIDUAL)})
        assert "omnibus" in str(excinfo.value)

    def test_accepts_an_expanded_exclusion(self) -> None:
        expanded = expand_all_prior(["a", "omnibus"], {"omnibus": ExclusionSpec.model_validate(_RESIDUAL)})
        reject_unexpanded_exclusions(expanded)

    def test_accepts_an_exclusion_that_never_used_all_prior(self) -> None:
        reject_unexpanded_exclusions(
            {
                "omnibus": ExclusionSpec.model_validate(
                    {
                        **_RESIDUAL,
                        "all_prior": False,
                        "subtrahends": [{"scope": "dataset", "dataset": "prior_audience", "artifact": "wide"}],
                    }
                )
            }
        )
