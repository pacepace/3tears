"""A seeded value must conform to its dimension's schema — presence and type, never plausibility."""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.kernel.host import schema_violations
from threetears.evals.kernel.host.world_schema import UnsupportedSchemaError


_JOB = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "duration_ms": {"type": "integer", "minimum": 0},
        "render_state": {"enum": ["not_started", "preparing", "ready"]},
    },
    "required": ["title", "duration_ms"],
}
_JOBS = {"type": "array", "items": _JOB, "maxItems": 3}


def test_a_conforming_value_passes():
    assert schema_violations(_JOBS, [{"title": "Invoice 7", "duration_ms": 1, "render_state": "ready"}], at="q") == []


def test_an_implausible_value_of_the_right_type_passes():
    # A timestamp in 1970 or a zero-length job is the author's to answer for, not the check's.
    assert schema_violations(_JOB, {"title": "", "duration_ms": 0}, at="t") == []


@pytest.mark.parametrize(
    ("schema", "value", "says"),
    [
        (_JOBS, [{"title": "x"}], "q[0] is missing 'duration_ms'"),
        (_JOBS, [{"title": 3, "duration_ms": 1}], "q[0].title is int 3, not string"),
        (_JOBS, [{"title": "x", "duration_ms": True}], "q[0].duration_ms is bool True, not integer"),
        (_JOBS, [{"title": "x", "duration_ms": -1}], "below the minimum 0"),
        (_JOBS, [{"title": "x", "duration_ms": 1, "render_state": "done"}], "q[0].render_state is 'done', not one of"),
        (_JOBS, [{}] * 4, "more than 3"),
        (_JOBS, {"title": "x"}, "q is dict"),
        ({"type": "integer", "maximum": 9}, 10, "q is 10, above the maximum 9"),
        ({"type": "string", "minLength": 2}, "x", "q is shorter than 2 characters"),
        ({"type": "string", "maxLength": 2}, "xyz", "q is longer than 2 characters"),
        ({"type": "array", "items": {"type": "string"}, "minItems": 2}, ["x"], "q has 1 items, fewer than 2"),
        ({"const": "ready"}, "done", "q is 'done', not one of ['ready']"),
        (
            {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": {"type": "integer"}},
            {"a": "x", "b": "y"},
            "q.b is str 'y', not integer",
        ),
    ],
)
def test_each_violation_is_named_at_its_path(schema, value, says):
    found = schema_violations(schema, value, at="q")
    assert any(says in sentence for sentence in found), found


def test_a_closed_object_refuses_an_undeclared_key():
    closed = {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False}

    assert schema_violations(closed, {"a": "x", "b": 1}, at="o") == ["o carries 'b', which the schema does not declare"]


def test_a_keyword_no_reader_honours_is_refused_rather_than_passed():
    with pytest.raises(UnsupportedSchemaError, match="exclusiveMinimum"):
        schema_violations({"type": "integer", "exclusiveMinimum": 0}, 0, at="n")


class TestABooleanIsNotAnInteger:
    """``True == 1`` in Python; in JSON Schema, and in the registry's enum-versus-type check, it is not."""

    @pytest.mark.parametrize(
        ("schema", "value"),
        [
            ({"enum": [0, 1]}, True),
            ({"enum": [0, 1]}, False),
            ({"const": 1}, True),
            ({"enum": [True]}, 1),
            ({"const": [1, {"a": 0}]}, [True, {"a": 0}]),
            ({"const": {"a": 1}}, {"a": True}),
        ],
    )
    def test_a_boolean_never_matches_an_enumerated_number_or_the_reverse(
        self, schema: dict[str, Any], value: Any
    ) -> None:
        assert schema_violations(schema, value, at="b") != []

    @pytest.mark.parametrize(
        ("schema", "value"),
        [
            ({"enum": [0, 1]}, 1),
            ({"enum": [True, False]}, False),
            ({"const": 1}, 1.0),
            ({"const": [1, {"a": True}]}, [1, {"a": True}]),
        ],
    )
    def test_an_equal_value_of_the_same_json_type_still_matches(self, schema: dict[str, Any], value: Any) -> None:
        assert schema_violations(schema, value, at="b") == []


_IDLE = {"title": "an idle printer", "type": "object", "properties": {}, "additionalProperties": False}
_PRINTING = {
    "title": "a job printing",
    "type": "object",
    "properties": {"title": {"type": "string"}, "elapsed_s": {"type": "number"}},
    "required": ["title", "elapsed_s"],
}
_EMPTY_OR_COMPLETE = {"anyOf": [_IDLE, _PRINTING]}


class TestAValueThatTakesOneOfSeveralShapes:
    """``anyOf`` is how a schema says "empty, or complete" — the one statement no other honoured keyword makes.

    An idle printer's ``{}`` and a job printing with every field are both legal; a half-stated job is
    neither. ``required`` alone refuses the first and leaving it out admits the last, which is why this used
    to be a side table consulted inside the seed walk.
    """

    @pytest.mark.parametrize("value", [{}, {"title": "Invoice 7", "elapsed_s": 12}])
    def test_a_value_either_shape_accepts_passes(self, value: dict[str, Any]) -> None:
        assert schema_violations(_EMPTY_OR_COMPLETE, value, at="job") == []

    def test_a_value_no_shape_accepts_names_what_each_refused(self) -> None:
        found = schema_violations(_EMPTY_OR_COMPLETE, {"title": "Invoice 7"}, at="job")

        assert found == [
            (
                "job fits none of the 2 shapes its schema accepts — "
                "as an idle printer, job carries 'title', which the schema does not declare"
                " | as a job printing, job is missing 'elapsed_s'"
            )
        ]

    def test_an_untitled_branch_is_named_by_its_position(self) -> None:
        found = schema_violations({"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, "x", at="v")

        assert found == [
            (
                "v fits none of the 2 shapes its schema accepts — "
                "as shape 1 of 2, v is str 'x', not integer | as shape 2 of 2, v is str 'x', not boolean"
            )
        ]

    def test_a_shape_nested_inside_a_shape_is_checked(self) -> None:
        nested = {"type": "array", "items": {"anyOf": [{"const": 0}, {"anyOf": [{"type": "string"}]}]}}

        assert schema_violations(nested, [0, "x"], at="a") == []
        assert schema_violations(nested, [0, 1], at="a") != []

    def test_a_boolean_does_not_pass_as_an_integer_shape(self) -> None:
        assert schema_violations({"anyOf": [{"const": 1}, {"type": "string"}]}, True, at="b") != []
