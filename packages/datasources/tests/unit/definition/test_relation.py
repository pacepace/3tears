"""unit tests for :class:`RelationRef`, its two body types, and alias scoping.

four corpus semantics drive this module:

- the join type is AUTHORED. the prototype makes two joins ``INNER`` or
  ``LEFT`` depending on whether the unit happens to declare an unrelated
  ``industries`` block, and nothing in the YAML says so.
- ``when`` omits the join entirely and ``optional`` changes its type.
  they are different mechanisms and a definition that omits a join and
  one that outer-joins it produce different audiences.
- ``when`` is per relation, not per definition. ``tsmart_comm`` injects a
  join at four stages across seven sites, and at the pivot it applies to
  the expansion branch only.
- a derived-table body that is itself typed is a first-class field; one
  holding a SQL string is an escape hatch. the two are distinct TYPES so
  ``parity-task-03`` can score them structurally.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.datasources.definition.expression import LiteralExpression
from threetears.datasources.definition.relation import (
    DuplicateRelationAlias,
    JoinKind,
    Projection,
    RawDerivedTable,
    RelationRef,
    TypedDerivedTable,
    UndeclaredRelationAlias,
    validate_relation_aliases,
)


def _committee_body() -> dict[str, Any]:
    """authored form of ``amazon_audience_l2_2025/standard_audience_units.yaml:11-23``.

    a parenthesised ``SELECT`` over a nested ``LEFT JOIN`` to an inner
    aggregate, plus a nine-element ``IN`` list.

    :returns: authored mapping for a typed derived-table body
    :rtype: dict[str, typing.Any]
    """
    committees = (
        "WINRED",
        "TRUMP MAKE AMERICA GREAT AGAIN COMMITTEE",
        "TRUMP SAVE AMERICA JOINT FUNDRAISING COMMITTEE",
        "TEAM SCALISE",
        "CRUZ FOR PRESIDENT",
        "TRUMP NATIONAL COMMITTEE JFC, INC.",
        "SENATE CONSERVATIVES FUND",
        "BLACKBURN TENNESSEE VICTORY FUND",
        "NATIONAL RIFLE ASSOCIATION OF AMERICA POLITICAL VICTORY FUND",
    )
    return {
        "projections": [{"expression": "rel.cmte.cmte_id"}],
        "source": "xavin.fec_committees_ext",
        "source_alias": "cmte",
        "relations": [
            {
                "relation": {
                    "projections": [
                        {"expression": "rel.cand_src.cand_id"},
                        {"expression": {"arith": "max(rel.cand_src.republican)"}, "alias": "republican"},
                    ],
                    "source": "xavin.fec_candidates_ext",
                    "source_alias": "cand_src",
                    "group_by": [{"literal": 1}],
                },
                "alias": "cand",
                "join": "LEFT",
                "optional": True,
                "on": {"compare": {"left": "rel.cmte.cand_id", "op": "=", "right": "rel.cand.cand_id"}},
            }
        ],
        "where": {
            "any_of": [
                {"compare": {"left": "rel.cmte.cmte_pty_affiliation", "op": "=", "right": {"literal": "REP"}}},
                {"compare": {"left": "rel.cand.republican", "op": "=", "right": {"literal": 1}}},
                {
                    "any_of": [
                        {"compare": {"left": "rel.cmte.cmte_nm", "op": "=", "right": {"literal": name}}}
                        for name in committees
                    ]
                },
            ]
        },
    }


class TestRelationRefShape:
    """``rel.<alias>.*`` is a namespace key, so ``alias`` is required."""

    def test_alias_is_required(self) -> None:
        with pytest.raises(ValidationError):
            RelationRef.model_validate({"relation": "modeling_frame_comm", "join": "inner"})

    def test_alias_must_be_an_identifier(self) -> None:
        with pytest.raises(ValidationError):
            RelationRef.model_validate({"relation": "cat_union", "alias": "cat union", "join": "inner"})

    def test_join_is_required_and_never_inferred(self) -> None:
        with pytest.raises(ValidationError):
            RelationRef.model_validate({"relation": "linkedin.job_title_fct", "alias": "fct"})

    def test_no_field_derives_the_join_from_another_field_s_presence(self) -> None:
        forbidden = {"industries", "infer_join", "join_from", "join_type_source"}
        assert forbidden.isdisjoint(set(RelationRef.model_fields))

    @pytest.mark.parametrize(("authored", "expected"), [("INNER", JoinKind.INNER), ("LEFT", JoinKind.LEFT)])
    def test_the_authored_join_keyword_canonicalises(self, authored: str, expected: JoinKind) -> None:
        relation = RelationRef.model_validate({"relation": "cat_union", "alias": "cat", "join": authored})
        assert relation.join is expected

    def test_on_defaults_to_the_governed_join_path(self) -> None:
        relation = RelationRef.model_validate({"relation": "match_union", "alias": "mat", "join": "inner"})
        assert relation.on is None

    def test_on_carries_the_condition_distinct_from_the_body(self) -> None:
        relation = RelationRef.model_validate(
            {
                "relation": "xavin.fec_contributions_ext",
                "alias": "ext",
                "join": "inner",
                "on": {"compare": {"left": "rel.ext.list_id", "op": "=", "right": "source.list_id"}},
            }
        )
        assert relation.on is not None
        assert relation.relation == "xavin.fec_contributions_ext"

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            RelationRef.model_validate(
                {"relation": "cat_union", "alias": "cat", "join": "inner", "to_schema": "analytics"}
            )

    def test_a_cross_join_carries_no_condition(self) -> None:
        with pytest.raises(ValidationError):
            RelationRef.model_validate(
                {
                    "relation": "calendar",
                    "alias": "cal",
                    "join": "cross",
                    "on": {"compare": {"left": "rel.cal.day", "op": "=", "right": "source.day"}},
                }
            )


class TestWhenOmitsAndOptionalOuterJoins:
    """``when`` removes the join; ``optional`` changes its type."""

    def test_when_omits_the_join_entirely(self) -> None:
        relation = RelationRef.model_validate(
            {
                "relation": "modeling_frame_comm",
                "alias": "modeling",
                "join": "inner",
                "when": {"compare": {"left": "param.tsmart_comm", "op": "=", "right": True}},
            }
        )
        assert relation.is_conditional is True
        assert relation.optional is False
        assert relation.join is JoinKind.INNER

    def test_optional_outer_joins_the_relation(self) -> None:
        relation = RelationRef.model_validate(
            {"relation": "linkedin.job_title_fct", "alias": "fct", "join": "left", "optional": True}
        )
        assert relation.optional is True
        assert relation.is_conditional is False
        assert relation.join is JoinKind.LEFT

    def test_when_omits_and_optional_outer_are_distinguishable_on_the_validated_model(self) -> None:
        omitted = RelationRef.model_validate(
            {
                "relation": "modeling_frame_comm",
                "alias": "modeling",
                "join": "inner",
                "when": {"compare": {"left": "param.tsmart_comm", "op": "=", "right": True}},
            }
        )
        outer = RelationRef.model_validate(
            {"relation": "linkedin.job_title_fct", "alias": "fct", "join": "left", "optional": True}
        )
        assert (omitted.is_conditional, omitted.optional) != (outer.is_conditional, outer.optional)

    def test_when_may_gate_the_same_relation_at_one_stage_and_not_another(self) -> None:
        gate = {"compare": {"left": "param.tsmart_comm", "op": "=", "right": True}}
        expansion_branch = RelationRef.model_validate(
            {"relation": "modeling_frame_comm", "alias": "modeling", "join": "inner", "when": gate}
        )
        influencer_branch = RelationRef.model_validate(
            {"relation": "modeling_frame_comm", "alias": "modeling", "join": "inner"}
        )
        assert expansion_branch.is_conditional != influencer_branch.is_conditional

    def test_when_binds_run_parameters_only(self) -> None:
        with pytest.raises(ValidationError):
            RelationRef.model_validate(
                {
                    "relation": "modeling_frame_comm",
                    "alias": "modeling",
                    "join": "inner",
                    "when": {"compare": {"left": "source.record_year", "op": ">", "right": 2020}},
                }
            )

    def test_optional_over_an_inner_join_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            RelationRef.model_validate(
                {"relation": "linkedin.companies", "alias": "companies", "join": "inner", "optional": True}
            )

    def test_a_left_join_may_declare_optional_false(self) -> None:
        relation = RelationRef.model_validate(
            {"relation": "ehirschfeld.source_archetypes", "alias": "sa", "join": "left", "optional": False}
        )
        assert relation.join is JoinKind.LEFT
        assert relation.optional is False

    def test_optional_defaults_to_false(self) -> None:
        relation = RelationRef.model_validate({"relation": "cat_union", "alias": "cat", "join": "inner"})
        assert relation.optional is False


class TestDerivedTableBodies:
    """a typed body is a first-class field; a SQL string is an escape hatch."""

    def test_derived_table_typed_vs_string(self) -> None:
        typed = RelationRef.model_validate(
            {
                "relation": {
                    "projections": [{"expression": "rel.probe.pid"}],
                    "source": "xavin.knowwho_exec_ext",
                    "source_alias": "probe",
                    "distinct": True,
                },
                "alias": "old_knowwho",
                "join": "left",
                "optional": True,
            }
        )
        stringly = RelationRef.model_validate(
            {
                "relation": {"raw_sql": "SELECT DISTINCT pid FROM xavin.knowwho_exec_ext"},
                "alias": "old_knowwho",
                "join": "left",
                "optional": True,
            }
        )
        assert type(typed.relation) is TypedDerivedTable
        assert type(stringly.relation) is RawDerivedTable
        assert TypedDerivedTable is not RawDerivedTable
        assert not issubclass(TypedDerivedTable, RawDerivedTable)
        assert not issubclass(RawDerivedTable, TypedDerivedTable)

    def test_only_the_string_body_is_an_escape_hatch(self) -> None:
        typed = TypedDerivedTable.model_validate(
            {"projections": [{"expression": "rel.probe.pid"}], "source": "t", "source_alias": "probe"}
        )
        stringly = RawDerivedTable.model_validate({"raw_sql": "SELECT 1"})
        assert stringly.is_escape_hatch is True
        assert typed.is_escape_hatch is False

    def test_a_typed_body_carries_no_sql_string_field(self) -> None:
        assert "raw_sql" not in TypedDerivedTable.model_fields

    def test_a_string_body_carries_no_typed_field(self) -> None:
        assert {"projections", "relations", "where", "group_by", "having"}.isdisjoint(set(RawDerivedTable.model_fields))

    def test_a_string_body_refuses_blank_text(self) -> None:
        with pytest.raises(ValidationError):
            RawDerivedTable.model_validate({"raw_sql": "   "})

    def test_the_amazon_l2_committee_body_is_expressible_as_a_typed_body(self) -> None:
        relation = RelationRef.model_validate(
            {"relation": _committee_body(), "alias": "cmte", "join": "INNER", "on": None}
        )
        body = relation.relation
        assert isinstance(body, TypedDerivedTable)
        nested = body.relations[0]
        assert nested.join is JoinKind.LEFT
        assert isinstance(nested.relation, TypedDerivedTable)
        ordinal = nested.relation.group_by[0]
        assert isinstance(ordinal, LiteralExpression)
        assert ordinal.literal == 1
        assert body.where is not None
        assert len(body.where.any_of or []) == 3
        membership_arm = (body.where.any_of or [])[2]
        assert len(membership_arm.any_of or []) == 9

    def test_a_typed_body_carries_its_own_group_by_and_having(self) -> None:
        body = TypedDerivedTable.model_validate(
            {
                "projections": [{"expression": "rel.facts.company_id"}],
                "source": "zuri_l2_analytics.employment_facts",
                "source_alias": "facts",
                "group_by": [{"literal": 1}],
                "having": {"compare": {"left": {"arith": "COUNT(DISTINCT rel.facts.list_id)"}, "op": ">", "right": 5}},
            }
        )
        assert body.having is not None
        assert len(body.group_by) == 1

    def test_a_group_by_ordinal_must_index_a_projection(self) -> None:
        with pytest.raises(ValidationError):
            TypedDerivedTable.model_validate(
                {
                    "projections": [{"expression": "rel.facts.company_id"}],
                    "source": "t",
                    "source_alias": "facts",
                    "group_by": [{"literal": 2}],
                }
            )

    def test_a_group_by_string_literal_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            TypedDerivedTable.model_validate(
                {
                    "projections": [{"expression": "rel.facts.company_id"}],
                    "source": "t",
                    "source_alias": "facts",
                    "group_by": [{"literal": "company_id"}],
                }
            )

    def test_a_typed_body_needs_at_least_one_projection(self) -> None:
        with pytest.raises(ValidationError):
            TypedDerivedTable.model_validate({"projections": [], "source": "t", "source_alias": "x"})

    def test_a_projection_alias_must_be_an_identifier(self) -> None:
        with pytest.raises(ValidationError):
            Projection.model_validate({"expression": "rel.x.y", "alias": "not an identifier"})

    def test_a_typed_body_nesting_a_string_body_is_an_escape_hatch(self) -> None:
        body = TypedDerivedTable.model_validate(
            {
                "projections": [{"expression": "rel.inner.pid"}],
                "source": {"raw_sql": "SELECT pid FROM legacy.table"},
                "source_alias": "inner",
            }
        )
        assert isinstance(body.source, RawDerivedTable)
        assert body.is_escape_hatch is True

    def test_a_wholly_typed_body_is_not_an_escape_hatch(self) -> None:
        relation = RelationRef.model_validate({"relation": _committee_body(), "alias": "cmte", "join": "inner"})
        assert isinstance(relation.relation, TypedDerivedTable)
        assert relation.relation.is_escape_hatch is False


class TestRelationScoping:
    """aliases are unique, and ``on`` sees only aliases declared before it."""

    def test_a_duplicate_alias_is_refused(self) -> None:
        relations = [
            RelationRef.model_validate({"relation": "a", "alias": "ext", "join": "inner"}),
            RelationRef.model_validate({"relation": "b", "alias": "ext", "join": "inner"}),
        ]
        with pytest.raises(DuplicateRelationAlias):
            validate_relation_aliases(relations)

    def test_on_may_reference_an_earlier_alias(self) -> None:
        relations = [
            RelationRef.model_validate({"relation": "xavin.fec_contributions_ext", "alias": "ext", "join": "inner"}),
            RelationRef.model_validate(
                {
                    "relation": "xavin.fec_committees_ext",
                    "alias": "cmte",
                    "join": "inner",
                    "on": {"compare": {"left": "rel.cmte.cmte_id", "op": "=", "right": "rel.ext.cmte_id"}},
                }
            ),
        ]
        assert validate_relation_aliases(relations) == relations

    def test_on_may_reference_its_own_alias(self) -> None:
        relations = [
            RelationRef.model_validate(
                {
                    "relation": "cat_union",
                    "alias": "cat",
                    "join": "inner",
                    "on": {"compare": {"left": "rel.cat.list_id", "op": "=", "right": "bridge.list_id"}},
                }
            )
        ]
        assert validate_relation_aliases(relations) == relations

    def test_on_may_not_reference_a_later_alias(self) -> None:
        relations = [
            RelationRef.model_validate(
                {
                    "relation": "xavin.fec_committees_ext",
                    "alias": "cmte",
                    "join": "inner",
                    "on": {"compare": {"left": "rel.cmte.cmte_id", "op": "=", "right": "rel.ext.cmte_id"}},
                }
            ),
            RelationRef.model_validate({"relation": "xavin.fec_contributions_ext", "alias": "ext", "join": "inner"}),
        ]
        with pytest.raises(UndeclaredRelationAlias):
            validate_relation_aliases(relations)

    def test_references_are_the_enclosing_scope_only(self) -> None:
        relation = RelationRef.model_validate({"relation": _committee_body(), "alias": "cmte", "join": "inner"})
        assert relation.references == ()

    def test_references_carry_the_condition_and_the_gate(self) -> None:
        relation = RelationRef.model_validate(
            {
                "relation": "modeling_frame_comm",
                "alias": "modeling",
                "join": "inner",
                "on": {"compare": {"left": "rel.modeling.voterbase_id", "op": "=", "right": "bridge.voterbase_id"}},
                "when": {"compare": {"left": "param.tsmart_comm", "op": "=", "right": True}},
            }
        )
        assert [reference.ref for reference in relation.references] == [
            "rel.modeling.voterbase_id",
            "bridge.voterbase_id",
            "param.tsmart_comm",
        ]
