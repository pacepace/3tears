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
    DerivedColumn,
    Expression,
    LiteralExpression,
    LiteralType,
    Measure,
    Namespace,
    Predicate,
    ProvenanceColumn,
    Reference,
    TermColumn,
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
            "right": {"literal": CORPUS_EMPLOYER_REGEX, "literal_type": LiteralType.TEXT},
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
        assert Comparison.model_validate(payload).model_dump()["right"] == {
            "literal": CORPUS_JOB_TITLE_REGEX,
            "literal_type": LiteralType.TEXT,
        }

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
        assert Comparison.model_validate(payload).model_dump()["right"] == {
            "literal": "% actor",
            "literal_type": LiteralType.TEXT,
        }


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
            Predicate.model_validate({"exists": "1=1"})

    def test_the_string_forms_land_and_are_scanned_for_references(self) -> None:
        # dsm-task-01d adds membership / concept / raw. a raw fragment is
        # scanned for namespaced references, so the stage guard sees them.
        raw = Predicate.model_validate({"raw": "resolved.record_year = 2024"})
        assert raw.is_escape_hatch is True
        assert [reference.ref for reference in raw.references] == ["resolved.record_year"]

    def test_a_concept_is_not_an_escape_hatch(self) -> None:
        assert Predicate.model_validate({"concept": "c_suite"}).is_escape_hatch is False

    def test_a_blank_string_form_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Predicate.model_validate({"raw": "   "})


#: the five real deliverables ``delivery.py``'s "do not tighten it" note
#: rests on, abbreviated only in ARITY. each is a distinct SHAPE, and a
#: closed union of aggregate / flag / ranked-category kinds expresses
#: none of the last three.
CORPUS_OPEN_EXPRESSIONS: tuple[tuple[str, str], ...] = (
    (
        "healthcare classifier",
        "MAX(CASE WHEN source.job_title LIKE '%nurse%' THEN 1 WHEN source.job_title LIKE '%md%' THEN 1 ELSE 0 END)",
    ),
    (
        "government_level",
        "CASE WHEN resolved.federal_level = 1 THEN 'federal' WHEN resolved.state_level = 1 THEN 'state' END",
    ),
    (
        "is_unique_match",
        "MAX(CASE WHEN resolved.quality_candidate_count = 1 THEN 1 ELSE 0 END)",
    ),
    (
        "units",
        "LISTAGG(DISTINCT resolved.unit, '|') WITHIN GROUP (ORDER BY resolved.unit)",
    ),
    (
        "opinion elite branch",
        "substring(resolved.units_temp, 1, len(resolved.units_temp) - 1)",
    ),
)


class TestArithmeticExpressionIsOpenByDesign:
    """F-05: the largest untyped surface, and the decision to leave it open.

    ``delivery.py`` records the reasoning and it holds: a closed union of
    kinds cannot express "concatenate two labels, trim the result, then
    classify on the exact value of that concatenation", which is one real
    shipped deliverable. This class pins the two properties that make the
    decision safe rather than lazy -- an open expression is NOT an escape
    hatch, and the honesty layer still sees every reference inside one --
    so tightening the field, or quietly reclassifying it as a hatch, fails
    here rather than moving a headline number with nothing re-authored.
    """

    @pytest.mark.parametrize(("deliverable", "text"), CORPUS_OPEN_EXPRESSIONS)
    def test_every_open_deliverable_is_carried_verbatim(self, deliverable: str, text: str) -> None:
        """the shipped shapes survive authoring byte for byte.

        :param deliverable: corpus deliverable the expression computes
        :ptype deliverable: str
        :param text: the aggregate as the corpus writes it
        :ptype text: str
        :returns: none
        :rtype: None
        """
        assert ArithmeticExpression(arith=text).arith == text, deliverable

    @pytest.mark.parametrize(("deliverable", "text"), CORPUS_OPEN_EXPRESSIONS)
    def test_no_open_deliverable_counts_as_an_escape_hatch(self, deliverable: str, text: str) -> None:
        """the audit is a TYPE check, and this type is not one of the three.

        ``parity-task-03``'s zero is over ``Predicate.raw``,
        ``RawSelect``, and ``RawDerivedTable``. Were an arithmetic
        expression to start reporting as a hatch, the headline would move
        from zero to five with nothing re-authored.

        :param deliverable: corpus deliverable the expression computes
        :ptype deliverable: str
        :param text: the aggregate as the corpus writes it
        :ptype text: str
        :returns: none
        :rtype: None
        """
        predicate = Predicate(
            compare=Comparison(
                left=ArithmeticExpression(arith=text),
                op=ComparisonOperator.EQUAL,
                right={"literal": 1},
            )
        )
        assert predicate.is_escape_hatch is False, deliverable

    def test_the_references_inside_an_open_expression_are_still_bound(self) -> None:
        """``scan_references`` over-detects rather than under-detects.

        The aggregate's own semantics are opaque -- that is F-04 -- but a
        reference inside one is surfaced, so the stage guard judges it and
        the compiler resolves and types it.

        :returns: none
        :rtype: None
        """
        classifier = ArithmeticExpression(arith=CORPUS_OPEN_EXPRESSIONS[0][1])
        listagg = ArithmeticExpression(arith=CORPUS_OPEN_EXPRESSIONS[3][1])
        assert {reference.ref for reference in classifier.references} == {"source.job_title"}
        assert {reference.ref for reference in listagg.references} == {"resolved.unit"}

    def test_a_reference_inside_a_nested_call_is_not_missed(self) -> None:
        """the substring case: the reference is a function ARGUMENT.

        :returns: none
        :rtype: None
        """
        substring = ArithmeticExpression(arith=CORPUS_OPEN_EXPRESSIONS[4][1])
        assert {reference.ref for reference in substring.references} == {"resolved.units_temp"}

    def test_an_open_expression_is_admitted_on_every_surface_that_needs_one(self) -> None:
        """four fields admit it, and each has a corpus deliverable behind it.

        Asserted through the surfaces rather than by reading annotations:
        a surface that stopped accepting the shape fails here, at the
        deliverable it drops.

        :returns: none
        :rtype: None
        """
        classifier = CORPUS_OPEN_EXPRESSIONS[0][1]
        listagg = CORPUS_OPEN_EXPRESSIONS[3][1]
        measure = Measure.model_validate(
            {
                "name": "has_relevant_linkedin_job_title",
                "expression": classifier,
                "grain": ["voterbase_id"],
                "scope": "resolution",
            }
        )
        derived = DerivedColumn.model_validate(
            {"name": "units", "sql_type": "character varying", "expression": {"arith": listagg}}
        )
        provenance = ProvenanceColumn.model_validate({"name": "units", "expression": {"arith": listagg}})
        term = TermColumn.model_validate({"name": "units", "value": {"arith": listagg}})
        assert measure.expression.arith == classifier
        assert isinstance(derived.expression, ArithmeticExpression)
        assert isinstance(provenance.expression, ArithmeticExpression)
        assert isinstance(term.value, ArithmeticExpression)
