"""tests for :mod:`threetears.models.tool_name_validation`."""

from __future__ import annotations

import pytest

from threetears.models.tool_name_validation import (
    ToolNameValidationError,
    filter_invalid_tool_calls,
    is_valid_tool_name,
    validate_tool_name,
)


class TestIsValidToolName:
    """tests for the canonical tool-name regex."""

    @pytest.mark.parametrize(
        "name",
        [
            "threetears.calculator",
            "3tears.schema.dictionary_ingest",
            "threetears.workspace.fs_read",
            "threetears.web_search",
        ],
    )
    def test_dotted_canonical_accepted(self, name: str) -> None:
        """canonical dotted-form tool names pass the regex."""
        assert is_valid_tool_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "threetears_calculator",
            "3tears_schema_dictionary_ingest",
            "threetears_workspace_fs_read",
        ],
    )
    def test_underscored_wire_accepted(self, name: str) -> None:
        """wire-form (underscored) tool names pass the regex."""
        assert is_valid_tool_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "datasource-central_reporting_schema",
            "datasource_central-reporting_schema",
            "legacy-tool-name",
        ],
    )
    def test_hyphenated_accepted(self, name: str) -> None:
        """hyphenated names (some legacy tool ids) pass the regex."""
        assert is_valid_tool_name(name) is True

    def test_single_char_accepted(self) -> None:
        """a single-character name passes (length lower bound = 1)."""
        assert is_valid_tool_name("a") is True

    def test_max_length_accepted(self) -> None:
        """a 64-character name passes; length upper bound is inclusive."""
        assert is_valid_tool_name("a" * 64) is True

    def test_rejects_observed_xml_attribute_leak(self) -> None:
        """exact pattern observed in prod incident 2026-05-19.

        a production conversation
        ``019e3e26-9870-7a03-8f04-8cc6a4f5f418`` recorded a tool call
        whose ``name`` field was set to
        ``memory_recall" name="memory_recall`` -- the result of a
        misbehaving model emitting an XML-attribute fragment inline
        with the tool-call dispatch. The junk name carries an
        embedded escaped quote and inner whitespace, both of which
        the canonical regex rejects. This test pins that contract so
        the failure mode cannot regress silently.
        """
        bad = 'memory_recall" name="memory_recall'
        assert is_valid_tool_name(bad) is False

    @pytest.mark.parametrize(
        "name",
        [
            "<tool>",
            "<memory_recall",
            "memory_recall>",
        ],
    )
    def test_rejects_leading_or_trailing_angle_bracket(self, name: str) -> None:
        """leading or trailing ``<`` / ``>`` (XML leakage) is rejected."""
        assert is_valid_tool_name(name) is False

    def test_rejects_embedded_newline(self) -> None:
        """embedded newlines are rejected (multi-line names are always junk)."""
        assert is_valid_tool_name("memory_recall\nmemory_recall") is False

    def test_rejects_empty_string(self) -> None:
        """the empty string is rejected (length lower bound = 1)."""
        assert is_valid_tool_name("") is False

    def test_rejects_length_over_64(self) -> None:
        """names longer than 64 characters are rejected."""
        assert is_valid_tool_name("a" * 65) is False

    @pytest.mark.parametrize(
        "name",
        [
            "tool with spaces",
            "tool\twith\ttabs",
            "tool!bang",
            "tool$dollar",
            'tool"quote',
            "tool'apost",
            "tool/slash",
            "tool\\backslash",
        ],
    )
    def test_rejects_punctuation_and_whitespace(self, name: str) -> None:
        """names with whitespace or non-allowed punctuation are rejected."""
        assert is_valid_tool_name(name) is False

    def test_rejects_non_string_input(self) -> None:
        """non-string inputs (None, int, list) return ``False``, not raise.

        is_valid_tool_name is invoked on whatever a provider put in
        the ``name`` field of a tool call; a misbehaving provider may
        ship a non-string value. The function MUST not raise on that
        input -- the caller would have to wrap every invocation in
        try/except if it did. Returning ``False`` matches the
        ``invalid name`` semantics.
        """
        assert is_valid_tool_name(None) is False  # type: ignore[arg-type]
        assert is_valid_tool_name(42) is False  # type: ignore[arg-type]
        assert is_valid_tool_name(["tool"]) is False  # type: ignore[arg-type]


class TestValidateToolName:
    """tests for the raising form of validation."""

    def test_valid_name_returns_none(self) -> None:
        """a valid name returns ``None`` without raising."""
        assert validate_tool_name("threetears.calculator") is None

    def test_invalid_name_raises_tool_name_validation_error(self) -> None:
        """an invalid name raises :class:`ToolNameValidationError`."""
        with pytest.raises(ToolNameValidationError):
            validate_tool_name('memory_recall" name="memory_recall')

    def test_error_is_value_error_subclass(self) -> None:
        """callers catching ``ValueError`` also catch the validation error."""
        with pytest.raises(ValueError):
            validate_tool_name("")

    def test_error_carries_bad_name(self) -> None:
        """the raised error exposes the rejected name as ``bad_name``."""
        try:
            validate_tool_name('memory_recall" name="memory_recall')
        except ToolNameValidationError as exc:
            assert exc.bad_name == 'memory_recall" name="memory_recall'
        else:
            pytest.fail("expected ToolNameValidationError")


class TestFilterInvalidToolCalls:
    """tests for the streaming-recovery filter."""

    def test_empty_list_returns_empty_tuple(self) -> None:
        """an empty input yields ``([], [])``."""
        kept, rejected = filter_invalid_tool_calls([])
        assert kept == []
        assert rejected == []

    def test_all_valid_passes_through(self) -> None:
        """if every name is valid, ``kept`` matches the input verbatim."""
        calls = [
            {"name": "threetears.calculator", "args": "{}", "id": "1"},
            {"name": "threetears_web_search", "args": "{}", "id": "2"},
        ]
        kept, rejected = filter_invalid_tool_calls(calls)
        assert kept == calls
        assert rejected == []

    def test_all_junk_names_moved_to_rejected(self) -> None:
        """non-empty malformed names all collect into ``rejected``."""
        calls = [
            {"name": 'memory_recall" name="memory_recall', "args": "{}", "id": "1"},
            {"name": "<tool>", "args": "{}", "id": "2"},
            {"name": "tool with spaces", "args": "{}", "id": "3"},
        ]
        kept, rejected = filter_invalid_tool_calls(calls)
        assert kept == []
        assert rejected == calls

    def test_mixed_splits_correctly(self) -> None:
        """valid / invalid entries split into ``kept`` / ``rejected``."""
        good = {"name": "threetears.calculator", "args": "{partial", "id": "1"}
        bad = {
            "name": 'memory_recall" name="memory_recall',
            "args": "{}",
            "id": "2",
        }
        kept, rejected = filter_invalid_tool_calls([good, bad])
        assert kept == [good]
        assert rejected == [bad]

    def test_missing_name_field_is_kept_not_junk(self) -> None:
        """a call dict without a ``name`` field is a streaming continuation.

        Only the first delta of a tool call carries the name; the rest
        accumulate by index in ``tool_call_chunks`` and merge into a
        valid tool call. A nameless entry is therefore NOT junk -- it
        must not be logged or dropped. (Before this fix it was rejected,
        producing a WARNING per streamed chunk: prod conv 019ecdfd,
        2026-06-16.)
        """
        call = {"args": "{}", "id": "1", "error": "JSONDecodeError"}
        kept, rejected = filter_invalid_tool_calls([call])
        assert kept == [call]
        assert rejected == []

    def test_none_name_is_kept_as_streaming_continuation(self) -> None:
        """``name=None`` is the normal continuation-fragment signature, not junk."""
        call = {"name": None, "args": "{}", "id": "1"}
        kept, rejected = filter_invalid_tool_calls([call])
        assert kept == [call]
        assert rejected == []

    def test_empty_string_name_is_kept_not_junk(self) -> None:
        """an empty-string name is a degenerate continuation, kept not rejected."""
        call = {"name": "", "args": "{}", "id": "1"}
        kept, rejected = filter_invalid_tool_calls([call])
        assert kept == [call]
        assert rejected == []

    def test_whitespace_only_name_is_rejected(self) -> None:
        """a whitespace-only name is a concrete name that can't dispatch -> junk.

        Distinct from the empty-string/None continuation case: spaces are a
        non-empty string that fails the canonical regex, so it is real junk,
        not a streaming fragment.
        """
        call = {"name": "   ", "args": "{}", "id": "1"}
        kept, rejected = filter_invalid_tool_calls([call])
        assert kept == []
        assert rejected == [call]

    def test_non_string_name_is_rejected(self) -> None:
        """a non-string, non-None name (e.g. an int) is still rejected.

        Only None / absent / empty (a *missing* name) counts as a streaming
        continuation. A non-string value is a malformed concrete claim, not
        a continuation, so the streaming-continuation fix must NOT widen to
        keep it.
        """
        call = {"name": 5, "args": "{}", "id": "1"}
        kept, rejected = filter_invalid_tool_calls([call])  # type: ignore[list-item]
        assert kept == []
        assert rejected == [call]

    def test_non_dict_entry_is_rejected(self) -> None:
        """a non-dict entry has no name to dispatch and is not a fragment."""
        call = "not-a-dict"
        kept, rejected = filter_invalid_tool_calls([call])  # type: ignore[list-item]
        assert kept == []
        assert rejected == [call]
