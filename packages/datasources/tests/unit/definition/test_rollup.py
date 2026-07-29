"""unit tests for :class:`Rollup` and its label obligation.

Three corpus semantics drive this module:

- ``members`` is ORDERED and first match wins. An unordered set silently
  changes which label a multi-member entity gets.
- the rollup emits a LABEL on the long and provenance projections, not
  only a wide flag. ``amazon_tech_audience`` stamps ``rollup_unit`` per
  long row at 19 INSERT sites, and the rollup is the grain two of the
  four client deliverables are sold at.
- rollups are not a partition and not exclusive. In
  ``uhg_policymakers`` the two rollups cover 22 of 23 units and 83
  entities are in both.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.definition.source import ArtifactRef
from threetears.datasources.definition.rollup import LabelArm, Rollup, RollupEmit

_FEDERAL_LEVEL = (
    "custom_policy_makers",
    "doj",
    "fda",
    "federal_legislators",
    "federal_legislators_staff",
    "hhs",
    "knowwho_fed_exec_agency_existing",
    "knowwho_fed_exec_agency_new",
    "knowwho_fed_exec_department_existing",
    "knowwho_fed_exec_department_new",
    "knowwho_fed_exec_office_existing",
    "knowwho_fed_exec_office_new",
    "sec",
    "treasury",
)


class TestRollupShape:
    """``name``, ``members``, ``otherwise``, ``over``, ``emit``."""

    def test_a_named_group_of_units_emitting_a_wide_flag(self) -> None:
        rollup = Rollup.model_validate(
            {"name": "federal_level", "members": list(_FEDERAL_LEVEL), "emit": ["wide_flag"]}
        )
        assert len(rollup.members) == 14
        assert rollup.emit == [RollupEmit.WIDE_FLAG]

    def test_members_is_an_ordered_list_not_a_set(self) -> None:
        assert Rollup.model_fields["members"].annotation == list[str]

    def test_member_order_is_preserved(self) -> None:
        rollup = Rollup.model_validate({"name": "r", "members": ["c", "a", "b"], "emit": ["wide_flag"]})
        assert rollup.members == ["c", "a", "b"]

    def test_a_single_member_rollup_is_legal(self) -> None:
        rollup = Rollup.model_validate(
            {
                "name": "tech decision makers at all companies",
                "members": ["tech decision makers at all companies"],
                "emit": ["long_label"],
            }
        )
        assert rollup.members == [rollup.name]

    def test_otherwise_defaults_to_none(self) -> None:
        rollup = Rollup.model_validate({"name": "r", "members": ["a"], "emit": ["wide_flag"]})
        assert rollup.otherwise is None

    def test_otherwise_carries_the_else_bucket(self) -> None:
        rollup = Rollup.model_validate(
            {
                "name": "01_academy_members",
                "members": ["academy_members"],
                "otherwise": "unmapped_core",
                "emit": ["long_label"],
            }
        )
        assert rollup.otherwise == "unmapped_core"

    def test_over_names_the_artifact_it_is_computed_against(self) -> None:
        rollup = Rollup.model_validate(
            {
                "name": "tagz_segment",
                "members": ["academy_members"],
                "over": {"scope": "dataset", "dataset": "universal_2026_core", "artifact": "provenance"},
                "emit": ["long_label"],
            }
        )
        assert isinstance(rollup.over, ArtifactRef)
        assert rollup.over.dataset == "universal_2026_core"

    def test_over_defaults_to_none(self) -> None:
        assert Rollup.model_validate({"name": "r", "members": ["a"], "emit": ["wide_flag"]}).over is None

    def test_rejects_an_empty_member_list(self) -> None:
        with pytest.raises(ValidationError):
            Rollup.model_validate({"name": "r", "members": [], "emit": ["wide_flag"]})

    def test_rejects_a_repeated_member(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Rollup.model_validate({"name": "r", "members": ["a", "a"], "emit": ["wide_flag"]})
        assert "twice" in str(excinfo.value)

    def test_rejects_a_blank_member(self) -> None:
        with pytest.raises(ValidationError):
            Rollup.model_validate({"name": "r", "members": [" "], "emit": ["wide_flag"]})

    def test_rejects_an_otherwise_equal_to_the_rollup_name(self) -> None:
        with pytest.raises(ValidationError):
            Rollup.model_validate({"name": "r", "members": ["a"], "otherwise": "r", "emit": ["wide_flag"]})

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Rollup.model_validate({"name": "r", "members": ["a"], "emit": ["wide_flag"], "column": "rollup_unit"})

    def test_round_trips(self) -> None:
        payload = {
            "name": "state_level",
            "members": ["governors", "state_cabinet"],
            "otherwise": None,
            "over": None,
            "emit": ["wide_flag", "long_label"],
        }
        assert Rollup.model_validate(payload).model_dump(mode="json") == payload


class TestEmitAdmitsAllThreeProjections:
    """the label obligation: not only a wide flag."""

    @pytest.mark.parametrize("kind", ["wide_flag", "long_label", "provenance_label"])
    def test_every_emit_kind_is_admitted(self, kind: str) -> None:
        rollup = Rollup.model_validate({"name": "r", "members": ["a"], "emit": [kind]})
        assert rollup.emit == [RollupEmit(kind)]

    def test_the_emit_vocabulary_is_exactly_three_kinds(self) -> None:
        assert {member.value for member in RollupEmit} == {"wide_flag", "long_label", "provenance_label"}

    def test_a_rollup_may_emit_all_three(self) -> None:
        rollup = Rollup.model_validate(
            {"name": "r", "members": ["a"], "emit": ["wide_flag", "long_label", "provenance_label"]}
        )
        assert rollup.emits_wide_flag is True
        assert rollup.emits_label is True

    def test_a_wide_flag_only_rollup_emits_no_label(self) -> None:
        assert Rollup.model_validate({"name": "r", "members": ["a"], "emit": ["wide_flag"]}).emits_label is False

    def test_a_long_label_rollup_emits_no_wide_flag(self) -> None:
        rollup = Rollup.model_validate({"name": "r", "members": ["a"], "emit": ["long_label"]})
        assert rollup.emits_wide_flag is False
        assert rollup.emits_label is True

    def test_rejects_an_empty_emit_list(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Rollup.model_validate({"name": "r", "members": ["a"], "emit": []})
        assert "emit" in str(excinfo.value)

    def test_rejects_a_repeated_emit_kind(self) -> None:
        with pytest.raises(ValidationError):
            Rollup.model_validate({"name": "r", "members": ["a"], "emit": ["wide_flag", "wide_flag"]})

    def test_rejects_an_unknown_emit_kind(self) -> None:
        with pytest.raises(ValidationError):
            Rollup.model_validate({"name": "r", "members": ["a"], "emit": ["delivered_label"]})


class TestRollupsOverlap:
    """membership is not exclusive and rollups are not a partition."""

    def test_two_rollups_may_share_a_member(self) -> None:
        federal = Rollup.model_validate({"name": "federal_level", "members": ["hhs"], "emit": ["wide_flag"]})
        state = Rollup.model_validate({"name": "state_level", "members": ["hhs"], "emit": ["wide_flag"]})
        assert federal.members == state.members

    def test_a_rollup_may_name_another_rollup_as_a_member(self) -> None:
        rollup = Rollup.model_validate(
            {"name": "government_level", "members": ["federal_level", "state_level"], "emit": ["wide_flag"]}
        )
        assert rollup.members == ["federal_level", "state_level"]


class TestLabelArm:
    """the ordered (predicate, label) shape both rollups and precedence use."""

    def test_an_arm_pairs_a_predicate_with_a_label(self) -> None:
        arm = LabelArm.model_validate(
            {"when": {"compare": {"left": "resolved.householders", "op": "=", "right": 1}}, "label": "householder"}
        )
        assert arm.label == "householder"
        assert arm.when.compare is not None

    def test_rejects_a_blank_label(self) -> None:
        with pytest.raises(ValidationError):
            LabelArm.model_validate({"when": {"compare": {"left": "resolved.a", "op": "=", "right": 1}}, "label": " "})

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            LabelArm.model_validate(
                {"when": {"compare": {"left": "resolved.a", "op": "=", "right": 1}}, "label": "x", "rank": 1}
            )
