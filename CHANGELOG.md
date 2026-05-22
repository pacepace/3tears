# Changelog

All notable changes to the 3tears platform packages are recorded here.
This project follows semantic versioning across all 16 workspace
packages (bumped in lock-step).

## v0.9.0 -- 2026-05-20

### Added

- `threetears.models.chunk_merging.merge_chunks` -- canonical merge of
  streamed `AIMessageChunk` lists into a single `AIMessage`. Wraps
  LangChain's `AIMessageChunk.__add__` for the merge, finalizes to a
  concrete `AIMessage`, and preserves `invalid_tool_calls` for
  downstream recovery. Replaces inline duplicates across consumers
  (metallm personality node, 14-eng-ai-bot router,
  14-eng-ai-bot-agents tool loop).
- `threetears.models.chunk_parsing.parse_chunk` -- canonical extractor
  of `(text, reasoning)` per streamed chunk. Covers all three
  observed shapes (OpenAI / OpenRouter string content, Anthropic-direct
  list-of-blocks, OpenRouter / OpenAI reasoning models'
  `additional_kwargs["reasoning_content"]`) and mixed cases. Pure,
  no-I/O hot-path helper.
- `threetears.models.tool_name_validation` -- canonical tool-name
  validator (`is_valid_tool_name`, `validate_tool_name`,
  `filter_invalid_tool_calls`, `ToolNameValidationError`). Pins the
  3tears tool-name regex (`^[a-zA-Z0-9_.-]{1,64}$`) covering every
  observed provider validator plus the dotted canonical form.

### Fixed

- Closes the metallm 2026-05-19 prod incident (conv
  `019e3e26-9870-7a03-8f04-8cc6a4f5f418`) where a misbehaving
  model response surfaced a tool-call name with an embedded
  XML-attribute fragment (`memory_recall" name="memory_recall`).
  The junk name reached metallm's dispatch layer through the
  chat-model wrapper unfiltered and was persisted as an
  unrecoverable invocation. The OpenRouter and Anthropic provider
  wrappers now call `filter_invalid_tool_calls` on every streamed
  chunk and every `_agenerate` result, dropping junk entries with
  one `WARNING` log per drop (name truncated to 80 chars). This
  blocks `function.name` junk from reaching downstream dispatch in
  any 3tears consumer.

### Notes

- v0.9.0 is a minor bump because it establishes new wrapper-layer
  contracts that downstream consumers can rely on: clean tool
  names guaranteed at the chat-model boundary, plus the canonical
  chunk-parsing / chunk-merging utilities. Bugfix patch would have
  been wrong given the new public API surface.
- All 16 workspace packages bumped to 0.9.0 in lock-step.
- No backwards-incompatible changes. Existing consumers that
  inline their own chunk parsing / merging continue to work; the
  new utilities are opt-in.
