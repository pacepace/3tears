"""behavioral tests for input coercion of loose LLM-supplied kwargs.

these tests pin the contract that ``normalize_kwargs`` and
``coerce_value`` accept the LLM mistakes the framework was built to
absorb: empty strings, JSON-encoded strings, and explicit ``None`` for
object / array fields. they also exercise the malformed-JSON branch
(``json.loads`` raises ``ValueError`` on garbage) so a future refactor
of the except clause cannot silently drop the swallow.

the file was empty before this commit, even though both
``test_coercion_coverage.py`` (a structural enforcement test) and
the ``TearsTool.run`` fallback (which swallows coercion exceptions
broadly) gave the appearance of coverage. behavioral coverage of
``coerce_value`` and ``normalize_kwargs`` themselves lives only here.
"""

from __future__ import annotations

import pytest

from threetears.agent.tools._coercion import coerce_value, normalize_kwargs


# ---------------------------------------------------------------------------
# coerce_value: array
# ---------------------------------------------------------------------------


def test_coerce_value_array_passthrough_when_already_list():
    """already-list values pass through untouched."""
    assert coerce_value([1, 2, 3], "array") == [1, 2, 3]


def test_coerce_value_array_empty_string_becomes_empty_list():
    """empty string for an array field is treated as 'give me []'."""
    assert coerce_value("", "array") == []


def test_coerce_value_array_json_encoded_string_is_decoded():
    """JSON-encoded array string is parsed back into a list."""
    assert coerce_value("[1, 2, 3]", "array") == [1, 2, 3]


def test_coerce_value_array_json_encoded_uuid_strings():
    """common LLM pattern: stringified list of UUIDs decodes correctly."""
    uuids = ["06a0ccd8-9199-75d6-8000-329082f7144d", "06a0ce89-6e83-7317-8000-3be8cf86198e"]
    assert coerce_value(str(uuids).replace("'", '"'), "array") == uuids


def test_coerce_value_array_malformed_json_passes_through():
    """malformed JSON raises ValueError inside json.loads; coerce_value
    must swallow it and return the original value so the subclass
    ``execute`` can decide how to handle it. without the except clause
    catching ValueError, every LLM mistake on an array field would
    crash the outer ``TearsTool.run`` coercion and the tool would
    receive the raw string anyway (via the broad fallback) -- this
    test pins the explicit, intentional swallow.
    """
    assert coerce_value("[not, valid, json", "array") == "[not, valid, json"
    assert coerce_value("garbage", "array") == "garbage"


def test_coerce_value_array_json_object_string_does_not_become_array():
    """a JSON object string supplied for an array field is NOT coerced.

    coercion only succeeds when the decoded type matches the declared
    type. a string that decodes to a dict, supplied for an array
    field, returns unchanged.
    """
    assert coerce_value('{"a": 1}', "array") == '{"a": 1}'


# ---------------------------------------------------------------------------
# coerce_value: object
# ---------------------------------------------------------------------------


def test_coerce_value_object_passthrough_when_already_dict():
    """already-dict values pass through untouched."""
    assert coerce_value({"a": 1}, "object") == {"a": 1}


def test_coerce_value_object_empty_string_becomes_empty_dict():
    """empty string for an object field is treated as 'give me {}'."""
    assert coerce_value("", "object") == {}


def test_coerce_value_object_json_encoded_string_is_decoded():
    """JSON-encoded dict string is parsed back into a dict."""
    assert coerce_value('{"a": 1, "b": "two"}', "object") == {"a": 1, "b": "two"}


def test_coerce_value_object_nested_json_encoded():
    """nested JSON (object containing array) decodes correctly."""
    payload = '{"name": "demo", "items": [1, 2, 3]}'
    assert coerce_value(payload, "object") == {"name": "demo", "items": [1, 2, 3]}


def test_coerce_value_object_malformed_json_passes_through():
    """malformed JSON for an object field must be swallowed and the
    original string returned unchanged. mirrors the array branch.
    """
    assert coerce_value("{this is not json", "object") == "{this is not json"


def test_coerce_value_object_json_array_string_does_not_become_object():
    """a JSON array string supplied for an object field is NOT coerced."""
    assert coerce_value("[1, 2, 3]", "object") == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# coerce_value: other types pass through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("declared_type", ["string", "integer", "number", "boolean", None])
def test_coerce_value_non_complex_types_pass_through(declared_type):
    """only object and array engage coercion; everything else is untouched."""
    assert coerce_value("[1,2,3]", declared_type) == "[1,2,3]"
    assert coerce_value("", declared_type) == ""
    assert coerce_value(42, declared_type) == 42


def test_coerce_value_preserves_falsy_scalars_for_array():
    """literal 0, False, empty tuple must not be coerced to empty container.

    only the empty STRING triggers the empty-container shortcut. other
    falsy values must pass through so type-strict subclasses still see
    them and can reject them.
    """
    assert coerce_value(0, "array") == 0
    assert coerce_value(False, "array") is False
    assert coerce_value((), "array") == ()


# ---------------------------------------------------------------------------
# normalize_kwargs: schema-driven coercion
# ---------------------------------------------------------------------------


def _schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties}


def test_normalize_kwargs_coerces_only_declared_array_and_object_fields():
    """kwargs not in the schema and non-complex declared types pass through."""
    schema = _schema({
        "name": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
    })
    raw = {
        "name": "hello",
        "tags": '["a", "b"]',
        "metadata": '{"k": "v"}',
        "unknown_field": "[also json]",  # not in schema -> passes through
    }
    result = normalize_kwargs(raw, schema)
    assert result == {
        "name": "hello",
        "tags": ["a", "b"],
        "metadata": {"k": "v"},
        "unknown_field": "[also json]",
    }


def test_normalize_kwargs_preserves_explicit_none():
    """explicit None survives so required-field checks in subclasses fire."""
    schema = _schema({
        "tags": {"type": "array"},
        "metadata": {"type": "object"},
    })
    result = normalize_kwargs({"tags": None, "metadata": None}, schema)
    assert result == {"tags": None, "metadata": None}


def test_normalize_kwargs_empty_schema_returns_kwargs_unchanged():
    """schema with no properties block is a no-op."""
    raw = {"any": "thing"}
    assert normalize_kwargs(raw, {}) == raw
    assert normalize_kwargs(raw, {"type": "object"}) == raw


def test_normalize_kwargs_malformed_json_does_not_crash():
    """if ``coerce_value`` ever started raising on malformed JSON, the
    outer ``TearsTool.run`` would catch it and silently fall back to
    the raw kwargs -- every tool would then see the malformed string
    and surface a misleading downstream error. this test pins
    ``normalize_kwargs`` as never-raising on garbage input.
    """
    schema = _schema({
        "datasource_ids": {"type": "array", "items": {"type": "string"}},
        "config": {"type": "object"},
    })
    # this call MUST NOT raise. malformed values pass through unchanged.
    result = normalize_kwargs(
        {"datasource_ids": "[broken json", "config": "{also broken"},
        schema,
    )
    assert result["datasource_ids"] == "[broken json"
    assert result["config"] == "{also broken"
