# Changelog

All notable changes to the 3tears platform packages are recorded here.
This project follows semantic versioning across all 17 workspace
packages (bumped in lock-step).

## v0.9.1 -- 2026-05-23

### Changed

- **`3tears-datasources` — pluggable secret resolution (Path A).**
  Datasource credentials are no longer named by an env var
  (`password_env` / `credentials_json_env`). They now carry a
  `scheme://locator` *reference* in `password_ref` /
  `credentials_json_ref`, resolved at driver-creation time (Hub-side,
  scoped to one datasource) by a pluggable backend in the new
  `threetears.datasources.secrets` module. The secret value never
  lives in agent.yaml, never lands plaintext in the Hub DB, and never
  sits in a long-lived process variable — it is only ever held inside
  a `SecretStr` and unwrapped at the last moment when handed to the
  backend lib. Shipped backends:
    - `env://NAME` — read process env var `NAME` (the devx backend;
      devx mounts the agent project `.env` into the Hub container so
      every datasource credential resolves on a fresh stack with no
      per-secret hand-listing).
    - `k8s://rel/path` — read a projected-Secret file under
      `AIBOTS_DATASOURCE_SECRETS_DIR` (default `/var/run/secrets/aibots`);
      the prod shape (k8s `Secret` as a volume).
  `vault://`, `aws-secretsmanager://` and `gcp-sm://` are registered
  but raise a clear "not implemented" error so the scheme surface is
  stable for config authors today. Config validators call
  `validate_ref` at load time (shape/scheme check, no env/fs touch);
  resolution stays a use-time concern. This is a hard rename with no
  backwards-compatibility shim.
- **`3tears-datasources` realigned to the monorepo lockstep version.**
  The package had been on an independent `0.1.x` line; it now versions
  with every other workspace package (`0.9.1`). Its README "Versioning
  policy" and CHANGELOG were rewritten accordingly.

### Notes

- Patch bump: the only behavioural change is internal to
  `3tears-datasources` (the credential-reference rename + resolver).
  No other package's public API changed.
- All 17 workspace packages bumped to 0.9.1 in lock-step (the
  `3tears-datasources` package joined the lockstep this release).
- The platform Docker image stamp tracks this tag (`v0.9.1`); the
  devx compose now injects the whole agent `.env` into the Hub
  container generically, retiring the per-secret passthrough.

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
