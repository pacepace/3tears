"""Contract for the one shared LLM-JSON-object extractor, :func:`threetears.evals.contracts.provider.extract_json`.

Every caller in the repo, eval and host alike, parses with this one function, so
its strategies and its raising contract are pinned here rather than at any caller:
fenced and bare objects parse, leading and trailing prose is tolerated by the
``raw_decode`` strategy, a top-level non-object is refused, and failure raises
``ValueError`` rather than returning ``None``.
"""

from __future__ import annotations

import pytest

from threetears.evals.contracts.provider import extract_json


class TestDirectParse:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_leading_and_trailing_whitespace(self):
        assert extract_json('\n  {"a": 1}\t\n') == {"a": 1}

    def test_nested_object(self):
        assert extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}


class TestFenceStripping:
    def test_markdown_fence_with_lang(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_code_fence_no_lang(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_fence_with_prose_around_it(self):
        text = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
        assert extract_json(text) == {"a": 1}


class TestRawDecodeStrategy:
    """The strategy the eval-side parser contributed: first complete object wins.

    ``json.JSONDecoder.raw_decode`` stops at the end of the first complete value,
    so trailing text that itself contains braces cannot defeat the parse. The
    outermost-brace slice the other parser used would splice the two together and
    fail — which is why that slice is subsumed rather than kept as a third tier.
    """

    def test_prose_on_both_sides(self):
        assert extract_json('here is the result: {"a": 1} (trailing commentary)') == {"a": 1}

    def test_object_followed_by_a_second_object(self):
        assert extract_json('{"a": 1} and then {"b": 2}') == {"a": 1}

    def test_object_followed_by_prose_containing_a_brace(self):
        assert extract_json('{"a": 1}\n\nNote: use {} for an empty dict.') == {"a": 1}

    def test_trailing_unbalanced_brace(self):
        assert extract_json('{"a": 1}}') == {"a": 1}

    def test_object_nested_in_a_top_level_array(self):
        # The document is an array, so the direct parse yields a non-dict and falls
        # through; ``raw_decode`` starts at the first brace and finds the object.
        # Both pre-reconciliation parsers behaved this way.
        assert extract_json('[{"a": 1}]') == {"a": 1}


class TestRejections:
    def test_no_object_raises(self):
        with pytest.raises(ValueError, match="No valid JSON object"):
            extract_json("plain text, no JSON")

    def test_top_level_array_raises(self):
        # Top-level arrays are not dicts — callers require a JSON object.
        with pytest.raises(ValueError, match="No valid JSON object"):
            extract_json("[1, 2, 3]")

    def test_top_level_scalar_raises(self):
        with pytest.raises(ValueError, match="No valid JSON object"):
            extract_json("42")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="No valid JSON object"):
            extract_json("")

    def test_message_reports_the_original_length_not_the_stripped_one(self):
        # The analysis generator propagates this message verbatim, so the number has
        # to describe what the model actually returned.
        with pytest.raises(ValueError, match=r"\(20 chars\)"):
            extract_json("   not json at all  ")

    def test_message_names_no_component(self):
        # It used to say "judge response" and was raised from six non-judge callers,
        # which is how a prod analysis failure came back naming a component that had
        # not run. The noun has to fit every caller, so it names none.
        with pytest.raises(ValueError) as excinfo:
            extract_json("nope")
        assert "judge" not in str(excinfo.value).lower()
