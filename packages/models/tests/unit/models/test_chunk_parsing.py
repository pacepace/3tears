"""tests for :func:`threetears.models.chunk_parsing.parse_chunk`."""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk

from threetears.models.chunk_parsing import ChunkParsed, parse_chunk


class TestParseChunkStringContent:
    """tests for the string-``content`` shape (OpenAI / OpenRouter
    non-reasoning models)."""

    def test_non_empty_string(self) -> None:
        """a non-empty string ``content`` becomes ``text``; ``reasoning`` empty."""
        chunk = AIMessageChunk(content="hello world")
        parsed = parse_chunk(chunk)
        assert parsed.text == "hello world"
        assert parsed.reasoning == ""

    def test_empty_string(self) -> None:
        """an empty string ``content`` produces empty ``text`` and ``reasoning``."""
        chunk = AIMessageChunk(content="")
        parsed = parse_chunk(chunk)
        assert parsed.text == ""
        assert parsed.reasoning == ""

    def test_returns_named_tuple(self) -> None:
        """the return is a ``ChunkParsed`` NamedTuple so call sites use
        named accessors instead of positional unpacking."""
        chunk = AIMessageChunk(content="x")
        parsed = parse_chunk(chunk)
        assert isinstance(parsed, ChunkParsed)
        assert parsed.text == "x"


class TestParseChunkListContent:
    """tests for the list-of-blocks ``content`` shape (Anthropic-direct v1)."""

    def test_single_text_block(self) -> None:
        """a single ``type=="text"`` block contributes its ``text`` field."""
        chunk = AIMessageChunk(
            content=[{"type": "text", "text": "anthropic text"}],
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == "anthropic text"
        assert parsed.reasoning == ""

    def test_single_thinking_block(self) -> None:
        """a single ``type=="thinking"`` block contributes its ``thinking`` field."""
        chunk = AIMessageChunk(
            content=[{"type": "thinking", "thinking": "let me consider"}],
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == ""
        assert parsed.reasoning == "let me consider"

    def test_mixed_text_and_thinking_blocks(self) -> None:
        """text and thinking blocks contribute to their respective fields."""
        chunk = AIMessageChunk(
            content=[
                {"type": "thinking", "thinking": "I should answer "},
                {"type": "text", "text": "Hello, "},
                {"type": "thinking", "thinking": "politely."},
                {"type": "text", "text": "human."},
            ],
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == "Hello, human."
        assert parsed.reasoning == "I should answer politely."

    def test_multiple_text_blocks_concatenate(self) -> None:
        """multiple ``text`` blocks concatenate in encounter order."""
        chunk = AIMessageChunk(
            content=[
                {"type": "text", "text": "one "},
                {"type": "text", "text": "two "},
                {"type": "text", "text": "three"},
            ],
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == "one two three"
        assert parsed.reasoning == ""

    def test_unknown_block_type_silently_skipped(self) -> None:
        """unknown block types are ignored without raising.

        consistent with production behavior in the consumer's personality
        node: a future provider that ships a ``type=="tool_use"`` or
        ``type=="image"`` block must not crash the chunk parser.
        Visibility for unknown types belongs at the call site, not in
        this hot-path helper.
        """
        chunk = AIMessageChunk(
            content=[
                {"type": "text", "text": "visible"},
                {"type": "future_block", "data": "ignored"},
                {"type": "thinking", "thinking": "hidden"},
            ],
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == "visible"
        assert parsed.reasoning == "hidden"

    def test_non_dict_block_skipped(self) -> None:
        """list entries that are not dicts are skipped (defensive).

        LangChain's ``AIMessageChunk`` permits ``str`` entries inside
        a list-form ``content`` (it accepts ``Union[str, dict]``); a
        future provider may emit raw-string entries alongside block
        dicts. The parser must skip those rather than treat the
        string as a block.
        """
        chunk = AIMessageChunk(
            content=[
                "raw string entry",
                {"type": "text", "text": "kept"},
            ],
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == "kept"
        assert parsed.reasoning == ""

    def test_empty_list_content(self) -> None:
        """empty list content yields empty fields without raising."""
        chunk = AIMessageChunk(content=[])
        parsed = parse_chunk(chunk)
        assert parsed.text == ""
        assert parsed.reasoning == ""


class TestParseChunkReasoningContent:
    """tests for the ``additional_kwargs["reasoning_content"]`` shape
    (OpenRouter / OpenAI reasoning models)."""

    def test_reasoning_content_only(self) -> None:
        """``additional_kwargs["reasoning_content"]`` populates ``reasoning``
        when ``content`` is empty (deepseek-r1, o1-style reasoning chunks)."""
        chunk = AIMessageChunk(
            content="",
            additional_kwargs={"reasoning_content": "thinking step one"},
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == ""
        assert parsed.reasoning == "thinking step one"

    def test_reasoning_content_with_string_text(self) -> None:
        """a chunk can carry both visible text and reasoning content
        simultaneously (provider reasoning-trailer attached to a
        final-answer chunk); both fields are extracted."""
        chunk = AIMessageChunk(
            content="answer is 42",
            additional_kwargs={"reasoning_content": "calculated via..."},
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == "answer is 42"
        assert parsed.reasoning == "calculated via..."

    def test_reasoning_content_with_list_content(self) -> None:
        """``additional_kwargs["reasoning_content"]`` adds to ``reasoning``
        on top of any ``thinking`` blocks in list-form content."""
        chunk = AIMessageChunk(
            content=[
                {"type": "thinking", "thinking": "block reasoning. "},
                {"type": "text", "text": "visible"},
            ],
            additional_kwargs={"reasoning_content": "kwarg reasoning."},
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == "visible"
        assert parsed.reasoning == "block reasoning. kwarg reasoning."

    def test_non_string_reasoning_content_ignored(self) -> None:
        """a non-string ``reasoning_content`` value is silently ignored.

        defensive: a misbehaving upstream that sets the key to ``None``
        or a structured value must not crash the chunk parser. The
        upstream contract is "string or absent"; anything else is
        treated as absent.
        """
        chunk = AIMessageChunk(
            content="x",
            additional_kwargs={"reasoning_content": None},
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == "x"
        assert parsed.reasoning == ""

    def test_missing_additional_kwargs_safe(self) -> None:
        """absence of ``additional_kwargs`` is handled defensively.

        LangChain's ``AIMessageChunk`` always populates the field, but
        the helper is duck-typed and may be called on mock chunks in
        tests; ``getattr(..., None) or {}`` keeps that ergonomic.
        """
        chunk = AIMessageChunk(content="just text")
        # AIMessageChunk has additional_kwargs by default, so clear it
        # to simulate the missing-attribute path.
        chunk.additional_kwargs.clear()
        parsed = parse_chunk(chunk)
        assert parsed.text == "just text"
        assert parsed.reasoning == ""


class TestParseChunkEmptyEdges:
    """edge cases for empty / missing fields."""

    def test_all_empty(self) -> None:
        """a chunk with no content of any kind yields empty fields."""
        chunk = AIMessageChunk(content="")
        parsed = parse_chunk(chunk)
        assert parsed.text == ""
        assert parsed.reasoning == ""

    def test_text_block_with_missing_text_field(self) -> None:
        """a ``type=="text"`` block without a ``text`` field contributes
        nothing -- ``block.get("text", "")`` default is the empty
        string, so the field stays empty without raising."""
        chunk = AIMessageChunk(
            content=[{"type": "text"}],
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == ""
        assert parsed.reasoning == ""

    def test_thinking_block_with_missing_thinking_field(self) -> None:
        """a ``type=="thinking"`` block without a ``thinking`` field
        contributes nothing without raising."""
        chunk = AIMessageChunk(
            content=[{"type": "thinking"}],
        )
        parsed = parse_chunk(chunk)
        assert parsed.text == ""
        assert parsed.reasoning == ""
