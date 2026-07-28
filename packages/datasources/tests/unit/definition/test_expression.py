"""unit tests for the expression / comparison / predicate algebra.

the operator vocabulary and the byte-exact literal round trip are the
load-bearing assertions here. production uses ``ILIKE`` 28 times in one
file and POSIX ``~`` / ``~*`` in three audiences, with doubled backslash
classes that have to survive authoring, serialization, and re-emission
without a single byte changing. the literals below are copied verbatim
from the corpus; they are deliberately NOT simplified.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import TypeAdapter, ValidationError

from threetears.datasources.definition import (
    ArithmeticExpression,
    Comparison,
    ComparisonOperator,
    Expression,
    LiteralExpression,
    Namespace,
    Predicate,
    Reference,
)

# ``uhg_opinion_elites/standard_audience_units.yaml:41`` -- a single POSIX
# ``~*`` literal mixing SINGLE (``(alphabet(\s|$))``) and DOUBLED
# (``((\\s(inc|corp))|$)``) backslash classes. the mixture is the point.
CORPUS_EMPLOYER_REGEX = (
    "(cyber(.|)security)|crowdstrike|fortinet|(palo alto networks)|zscaler|cloudflare|"
    "(cisco tech)|darktrace|mcafee|okta|rapid7|(trend micro)|bettercloud|"
    "(abnormal security)|sophos|(alphabet(\\s|$))|nvidia|(^apple(\\s(comput|inc)|$))|"
    "(gen digital)|(symantec corp)|nortonlifelock|(^(ibm|intel|oracle|meta)"
    "((\\\\s(inc|corp))|$))|(international business machine)|(^dell($|\\\\s(tech|compu)))"
)

# ``uhg_opinion_elites/standard_audience_units.yaml:40``
CORPUS_JOB_TITLE_REGEX = (
    "(cyber(.|)(security|crime))|security analyst|information security specialist|"
    "digital forensic|security systems administrator|security (engineer|architect)|cryptography"
)

# ``universal_2026_expansion/custom_audience_units/"
# "linkedin_entertainment_provider_companies.sql.jinja2:98`` -- a doubled
# single quote inside a SQL string literal.
CORPUS_ESCAPED_QUOTE_LITERAL = "knott''s-berry-farm"

_EXPRESSION_ADAPTER: TypeAdapter[Expression] = TypeAdapter(Expression)


class TestExpressionCoercion:
    """a bare string is ALWAYS a reference; a literal is written explicitly."""

    def test_bare_string_becomes_a_reference(self) -> None:
        parsed = _EXPRESSION_ADAPTER.validate_python("entity.vb_voterbase_age")
        assert isinstance(parsed, Reference)
        assert parsed.namespace is Namespace.ENTITY

    def test_bare_string_that_is_not_a_reference_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _EXPRESSION_ADAPTER.validate_python("x")

    def test_bare_int_becomes_a_literal(self) -> None:
        parsed = _EXPRESSION_ADAPTER.validate_python(20)
        assert isinstance(parsed, LiteralExpression)
        assert parsed.literal == 20

    def test_bare_bool_stays_a_bool(self) -> None:
        parsed = _EXPRESSION_ADAPTER.validate_python(True)
        assert isinstance(parsed, LiteralExpression)
        assert parsed.literal is True

    def test_explicit_string_literal(self) -> None:
        parsed = _EXPRESSION_ADAPTER.validate_python({"literal": "state_staffers"})
        assert isinstance(parsed, LiteralExpression)
        assert parsed.literal == "state_staffers"

    def test_decimal_literal_is_not_a_float(self) -> None:
        parsed = _EXPRESSION_ADAPTER.validate_python({"literal": Decimal("1.5")})
        assert isinstance(parsed, LiteralExpression)
        assert parsed.literal == Decimal("1.5")
        assert not isinstance(parsed.literal, float)

    def test_null_literal_is_written_explicitly(self) -> None:
        parsed = _EXPRESSION_ADAPTER.validate_python({"literal": None})
        assert isinstance(parsed, LiteralExpression)
        assert parsed.literal is None


class TestArithmeticExpression:
    """``arith`` carries the text verbatim and surfaces its references."""

    def test_working_age_ceiling(self) -> None:
        arith = ArithmeticExpression(arith="70 + param.run_year - resolved.record_year")
        assert {ref.ref for ref in arith.references} == {
            "param.run_year",
            "resolved.record_year",
        }

    def test_text_round_trips_byte_exactly(self) -> None:
        text = "SUM(contribution::float * 1.0/bridge.candidate_count::float)"
        assert ArithmeticExpression(arith=text).model_dump() == {"arith": text}

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            ArithmeticExpression(arith="   ")


class TestComparisonOperators:
    """the vocabulary the corpus actually uses."""

    @pytest.mark.parametrize(
        "literal",
        ["=", "!=", "<>", "<", "<=", ">", ">=", "LIKE", "NOT LIKE", "ILIKE", "NOT ILIKE"],
    )
    def test_scalar_and_pattern_operators_are_present(self, literal: str) -> None:
        assert ComparisonOperator(literal).value == literal

    @pytest.mark.parametrize("literal", ["~", "~*", "!~", "!~*"])
    def test_posix_regex_operators_are_present(self, literal: str) -> None:
        assert ComparisonOperator(literal).value == literal

    def test_ne_and_ne_ansi_are_distinct_members(self) -> None:
        assert ComparisonOperator("!=") is not ComparisonOperator("<>")

    def test_lowercase_word_operator_is_accepted_and_canonicalised(self) -> None:
        comparison = Comparison.model_validate(
            {"left": "source.job_title", "op": "ilike", "right": {"literal": "%producer%"}}
        )
        assert comparison.op is ComparisonOperator.ILIKE

    def test_unary_operators_take_no_right_operand(self) -> None:
        comparison = Comparison.model_validate({"left": "rel.wiki.sports", "op": "IS NULL"})
        assert comparison.right is None

    def test_unary_operator_with_a_right_operand_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Comparison.model_validate({"left": "rel.wiki.sports", "op": "IS NULL", "right": 1})

    def test_binary_operator_without_a_right_operand_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Comparison.model_validate({"left": "source.job_title", "op": "="})


class TestComparisonExpressionsOnBothSides:
    """the working-age filter requires an expression on the right."""

    def test_working_age_ceiling_parses(self) -> None:
        comparison = Comparison.model_validate(
            {
                "left": "entity.vb_voterbase_age",
                "op": "<",
                "right": {"arith": "70 + param.run_year - resolved.record_year"},
            }
        )
        assert isinstance(comparison.right, ArithmeticExpression)
        assert {ref.ref for ref in comparison.references} == {
            "entity.vb_voterbase_age",
            "param.run_year",
            "resolved.record_year",
        }


class TestRegexRoundtrip:
    """the corpus's doubled-escape literals survive every boundary."""

    def test_regex_roundtrip_employer_python(self) -> None:
        payload = {
            "left": "source.employer",
            "op": "~*",
            "right": {"literal": CORPUS_EMPLOYER_REGEX},
        }
        comparison = Comparison.model_validate(payload)
        assert comparison.model_dump() == {
            "left": {"ref": "source.employer"},
            "op": "~*",
            "right": {"literal": CORPUS_EMPLOYER_REGEX},
        }

    def test_regex_roundtrip_employer_json(self) -> None:
        comparison = Comparison.model_validate(
            {
                "left": "source.employer",
                "op": "~*",
                "right": {"literal": CORPUS_EMPLOYER_REGEX},
            }
        )
        rehydrated = Comparison.model_validate_json(comparison.model_dump_json())
        assert isinstance(rehydrated.right, LiteralExpression)
        assert rehydrated.right.literal == CORPUS_EMPLOYER_REGEX

    def test_regex_roundtrip_preserves_single_and_doubled_backslashes(self) -> None:
        comparison = Comparison.model_validate(
            {
                "left": "source.employer",
                "op": "~*",
                "right": {"literal": CORPUS_EMPLOYER_REGEX},
            }
        )
        assert isinstance(comparison.right, LiteralExpression)
        rendered = comparison.right.literal
        assert isinstance(rendered, str)
        assert "(alphabet(\\s|$))" in rendered
        assert "((\\\\s(inc|corp))|$)" in rendered
        assert "(^dell($|\\\\s(tech|compu)))" in rendered

    def test_regex_roundtrip_job_title(self) -> None:
        payload = {
            "left": "source.job_title",
            "op": "~*",
            "right": {"literal": CORPUS_JOB_TITLE_REGEX},
        }
        assert Comparison.model_validate(payload).model_dump()["right"] == {"literal": CORPUS_JOB_TITLE_REGEX}

    def test_regex_roundtrip_escaped_single_quote(self) -> None:
        payload = {
            "left": "source.organization_bucketed",
            "op": "=",
            "right": {"literal": CORPUS_ESCAPED_QUOTE_LITERAL},
        }
        comparison = Comparison.model_validate(payload)
        as_json = json.loads(comparison.model_dump_json())
        assert as_json["right"]["literal"] == CORPUS_ESCAPED_QUOTE_LITERAL

    def test_ilike_pattern_roundtrip(self) -> None:
        payload = {
            "left": "source.job_title",
            "op": "ILIKE",
            "right": {"literal": "% actor"},
        }
        assert Comparison.model_validate(payload).model_dump()["right"] == {"literal": "% actor"}


class TestPredicate:
    """recursive boolean structure; exactly one field set."""

    def test_all_of(self) -> None:
        predicate = Predicate.model_validate(
            {
                "all_of": [
                    {"compare": {"left": "entity.vb_voterbase_age", "op": ">", "right": 20}},
                    {"compare": {"left": "resolved.candidate_count", "op": "<=", "right": 10}},
                ]
            }
        )
        assert predicate.all_of is not None
        assert len(predicate.all_of) == 2

    def test_any_of_nested_in_negate(self) -> None:
        predicate = Predicate.model_validate(
            {
                "negate": {
                    "any_of": [
                        {"compare": {"left": "source.job_title", "op": "LIKE", "right": {"literal": "%partner%"}}},
                        {"compare": {"left": "source.job_title", "op": "LIKE", "right": {"literal": "%owner%"}}},
                    ]
                }
            }
        )
        assert predicate.negate is not None

    def test_exactly_one_field_must_be_set(self) -> None:
        with pytest.raises(ValidationError):
            Predicate.model_validate(
                {
                    "all_of": [{"compare": {"left": "source.a", "op": "=", "right": 1}}],
                    "negate": {"compare": {"left": "source.b", "op": "=", "right": 1}},
                }
            )

    def test_empty_predicate_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Predicate.model_validate({})

    def test_empty_all_of_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Predicate.model_validate({"all_of": []})

    def test_references_walks_the_whole_tree(self) -> None:
        predicate = Predicate.model_validate(
            {
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
        )
        assert {ref.ref for ref in predicate.references} == {
            "entity.vb_voterbase_age",
            "param.run_year",
            "resolved.record_year",
        }

    def test_authored_conjunct_order_is_preserved(self) -> None:
        predicate = Predicate.model_validate(
            {
                "all_of": [
                    {"compare": {"left": "rel.old_knowwho.pid", "op": "IS NULL"}},
                    {
                        "compare": {
                            "left": "source.orgname",
                            "op": "=",
                            "right": {"literal": "Antitrust Division"},
                        }
                    },
                ]
            }
        )
        assert predicate.all_of is not None
        assert [arm.compare.left.ref for arm in predicate.all_of if arm.compare is not None] == [
            "rel.old_knowwho.pid",
            "source.orgname",
        ]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Predicate.model_validate({"raw": "1=1"})
