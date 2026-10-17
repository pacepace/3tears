# S1: what a `3tears-models` chat model exposes of the evals completion contract

Investigation for `eval-providers-design-2026-10-13.md`. Nothing in any checkout changed and no test suite was run. All empirical results come
from building real factory models with dummy keys and replacing every `httpx.AsyncClient` transport with an
`httpx.MockTransport` that returns canned provider replies. No provider was called. The probe scripts were throwaway and are not kept.

**Which tree was read.** `packages/models` at `origin/develop` `ff00526c5` (models 0.65.0). The eval-adopt `.23`
integration branch at `741c00467` carries the same `packages/models` byte for byte apart from the pyproject
version, which reads 0.66.0, so this is the code a `3tears-evals[models]` adapter would ship against. The probes
ran in that branch's workspace venv. An older develop checkout (models 0.32.1, langchain-core 1.4.9,
langchain-openrouter 0.2.6) has no `declare_served_provider` or `served_provider`; do not design against it.

Path shorthands:
- `M/` = `packages/models/src/threetears/models/`
- `E/` = `packages/evals/src/threetears/evals/` (at `741c00467`)
- `SP/` = the workspace venv's `lib/python3.14/site-packages/`

"claude_cli" means the anthropic provider given an OAuth `sk-ant-oat…` token. `create_anthropic_chat` routes that
token to `create_subscription_chat` (`M/providers/anthropic.py:271`). `create_chat_model` has no `claude_cli`
provider name of its own.

---

## The table

Rows 1-7 are the ten `CompletionResult` members. Rows 8-12 are the call-side inputs and behaviours.
**E** = exposed directly, **D** = derivable by the adapter, **H** = hidden.

| # | Item | anthropic (`ChatAnthropic`) | openai (`ChatOpenAI`) | openrouter (`ChatOpenRouter`, streamed) | claude_cli (`ClaudeCodeChatModel`) |
|---|---|---|---|---|---|
| 1 | `content` | **E** `msg.text`. `msg.content` is a block **list** whenever thinking or tool_use blocks are present | **E** `msg.content` (str) / `msg.text`. A refusal arrives as `additional_kwargs["refusal"]` with `""` content | **E** `msg.text`. Reasoning text sits apart, in `additional_kwargs["reasoning_content"]` | **E** str. Under a schema it is `json.dumps(structured_output)`, re-serialised, with the model's prose dropped |
| 2a | `input_tokens` | **E** `usage_metadata.input_tokens` (cache read + write included). An unreported count reads as `0`, never `None` | **E** `usage_metadata.input_tokens`; `usage_metadata` is `None` when the reply has no usage | **E** from the usage-only stream chunk; `None` when there is none | **E** `usage_metadata.input_tokens` (uncached + cache read + cache write); `None` without CLI usage |
| 2b | `output_tokens` | **E** `usage_metadata.output_tokens` (thinking included) | **E** (reasoning included) | **E** | **E** |
| 3 | `reasoning_tokens` | **D** `response_metadata["usage"]["output_tokens_details"]["thinking_tokens"]`. LangChain does not map it into `output_token_details` | **E** `usage_metadata["output_token_details"]["reasoning"]` | **E** `usage_metadata["output_token_details"]["reasoning"]` | **H** models' `_usage_metadata` drops it. The raw `response_metadata["usage"]` *may* carry it (unverified) |
| 4a | `cost_usd` | **D** tokens x `price_lookup` rate. Nothing in models puts a cost on the message | **D** same as anthropic | **E** `response_metadata["cost"]`, OpenRouter's own figure | **E** `response_metadata["total_cost_usd"]`, the CLI's figure. Notional on a subscription |
| 4b | `price_source` | **D** adapter names it (e.g. catalogue + `matched_id`) | **D** same as anthropic | **D** adapter names it `openrouter:reported` | **D** adapter names it (CLI-reported, notional) |
| 5a | `model` | **E** request id (`inner.model`, or the adapter's own `model_id`) | **E** `inner.model_name` | **E** `inner.model_name` | **E** `inner.model` |
| 5b | `served_model` | **E** `response_metadata["model"]`, the raw response field | **E** `response_metadata["model_name"]`, the response `model` (`None` when null) | **E\*** `response_metadata["model_name"]`. **\*If the chunk's `model` is `""` this falls back to the requested id.** The serving upstream is `response_metadata["provider"]` | **H** `AssistantMessage.model` and `ResultMessage.model_usage` are dropped by models. `fallback_model` can swap the model silently |
| 6 | `stop_reason` | **D** map `response_metadata["stop_reason"]` | **D** map `response_metadata["finish_reason"]` | **D** map `response_metadata["finish_reason"]` (+ `native_finish_reason`) | **H** models synthesises `"stop"` / `"tool_calls"`. The CLI's real `ResultMessage.stop_reason` is dropped, so **a truncation reads as finished** |
| 7 | `temperature` (as sent) | **D** what the adapter bound. `None` is omitted from the payload. With thinking on, the API refuses a temperature and LangChain does not drop it | **D** read `inner.temperature` **after** construction: gpt-5 (non-chat) silently pops it at build (0.2 became `None`). A per-call bind is **not** filtered | **D** what the adapter bound (sent verbatim). Whether the upstream honoured it is unknowable unless `require_parameters` is on | **D** always `None`. The CLI takes no temperature; models drops it at build and warns when it is bound per call |
| 8 | `response_format` | `structured_output_kwargs("anthropic", schema)` gives `output_config.format` (transform_schema). Raw JSON text is in `msg.text`. **No JSON mode exists**; `{"type":"object"}` transforms to an empty-object schema | gives `extra_body.response_format` (plain `.create` path, raw text). `json_object` also goes via `extra_body` | gives top-level `response_format` + `provider.require_parameters=True`. An unsupporting upstream is refused (404 `NotFoundResponseError`), not answered in prose | `output_config` becomes CLI `--json-schema`. The CLI retries the model itself, up to 5 attempts / 6 turns |
| 9 | reasoning effort | `effort` takes only `max/xhigh/high/medium/low` (`minimal` and `none` refused at build). Budget: `thinking={"type":"enabled","budget_tokens":N}`. Per call: `bind(thinking=…)` and `bind(output_config={"effort":…})` work; `bind(effort=…)` raises `TypeError` | `reasoning_effort` (str) per instance or per call (`bind(reasoning_effort=…)` confirmed). No token budget | `reasoning={"effort": w}` takes all 7 engine words. **`reasoning.max_tokens` and `reasoning.enabled` are silently stripped by the SDK** (sent as `reasoning: {}`) | `thinking` / `effort` per instance or per call. The defaults are pinned to thinking off and effort `high` |
| 10 | failures | raw `anthropic.*Error` (`status_code`, `body`); `AnthropicContextOverflowError` (LangChain subclass); `CircuitOpenError` | raw `openai.*Error`; `OpenAIContextOverflowError`; a 200 carrying an in-body error raises a bare `ValueError` | `openrouter.errors.*` (`status_code`, `body`); a mid-stream error is a bare `ValueError`; `ModelCallTimeout`; a malformed chunk is a pydantic `ValidationError` | `ModelProviderError` / `ModelRateLimitError` (`status`, `reason`, `resets`); `ClaudeCliPool*`; SDK errors |
| 11 | lifecycle | no close; the httpx client is `lru_cache`-shared process-wide, so **must not** be closed | same as anthropic | the SDK client and its httpx clients are owned **per instance** (GC finaliser). Closable via `await inner.client.__aexit__(None,None,None)` | process-wide CLI subprocess pool; the host calls `close_claude_cli_pool()` at shutdown. Nothing per model |
| 12 | hidden retries | SDK `max_retries=2` on 408/409/429/5xx/529, timeouts and connection errors (529,529,200 = 3 hits, invisible). Off with `max_retries=0` | SDK `max_retries=2`; **retries 429 `insufficient_quota` too** (3 hits). Off with `max_retries=0` | SDK backoff on 5XX, connection and timeout errors, **bounded by time (300 s), not count**; 429 is not retried. **Not off with `max_retries=0`**: that falls back to the SDK's one-hour default (12 hits on a 502); off with `model_kwargs={"retries": None}` | CLI-internal API retries (`CLAUDE_CODE_MAX_RETRIES`, default 10), plus schema re-attempts. **Models gives no way to turn either off** |

Fully exposed, of the ten members (`content`, `input_tokens`, `output_tokens`, `cost_usd`, `price_source`,
`model`, `served_model`, `reasoning_tokens`, `temperature`, `stop_reason`):

- **anthropic: 5** (content, input, output, model, served_model). Derivable: cost, price_source, reasoning, temperature, stop_reason. Hidden: none.
- **openai: 6** (+ reasoning_tokens). Derivable: cost, price_source, temperature, stop_reason. Hidden: none.
- **openrouter: 7** (+ cost_usd). Derivable: price_source, temperature, stop_reason. Hidden: none.
- **claude_cli: 5** (content, input, output, model, cost). Derivable: price_source, temperature. **Hidden: served_model, reasoning_tokens, stop_reason.**

`price_source` and `temperature` are never exposed by LangChain on any provider. Both are facts the adapter
holds itself: which price it applied, and what its request carried.

---

## Detail, per item

### 1. `content`

- **anthropic.** `ChatAnthropic._format_output` (`SP/langchain_anthropic/chat_models.py:1732`) returns
  `content=<str>` only for a single plain text block. With a thinking block, a tool_use block or citations,
  `content` is the block list. The probe (thinking + text) got `type(content)=list`; `msg.text` gave only the
  text block's text (`SP/langchain_core/messages/base.py:263`). **The adapter must read `.text`, never
  `str(content)`.**
- **openai.** `content` is a str. Reasoning summaries do not appear on the chat-completions path. A refusal
  (structured mode) is `additional_kwargs["refusal"]` beside empty content, and `finish_reason` can still be
  `stop`.
- **openrouter.** `content` is a str. Reasoning text is `additional_kwargs["reasoning_content"]` (and
  `reasoning_details`), which `.text` excludes. Confirmed in the probe.
- **claude_cli.** `_aquery` / `_astream` (`M/providers/claude_cli.py:842-1050`) join `TextBlock` text. Under a
  schema the content is `json.dumps(answer)` (`:898`), the CLI's `structured_output` re-serialised, with the
  model's preamble prose dropped. That is valid JSON text, but not the bytes the model wrote.

### 2. `input_tokens` / `output_tokens`

- **anthropic.** `_create_usage_metadata` (`SP/langchain_anthropic/chat_models.py:2353`) sets `input_tokens` =
  `input_tokens + cache_read + cache_creation` and `output_tokens` from the API (thinking included). It defaults
  every absent count to `0`, so unreported and zero cannot be told apart. Anthropic always returns usage, so in
  practice this is safe. Probe: API 11 in + 3 cache read became `input_tokens=14`.
- **openai.** `usage_metadata` is `None` when the reply has no `usage` (probe confirmed), which is the right
  "unknown".
- **openrouter.** Models forces the streamed path. `NameTranslatingChatMixin._agenerate` uses
  `agenerate_from_stream` whenever `call_deadline_s()` is set (`M/providers/_name_translation_mixin.py:325-337`),
  and OpenRouter's is 120 s (`M/providers/openrouter.py:315`). So usage comes from the usage-only chunk
  (`SP/langchain_openrouter/chat_models.py:710-723`, `stream_options.include_usage` because the factory leaves
  `stream_usage=True`). If that chunk is absent, `usage_metadata` is `None` (probe confirmed).
- **claude_cli.** `_usage_metadata` (`M/providers/claude_cli.py:1462`) builds input from uncached + cache read +
  cache write, and output. It returns `None` when the CLI reported none. On a structured call that took several
  turns, `ResultMessage.usage` is the cumulative figure.
- **Do not use models' own usage helpers for the ledger.** `extract_usage` (`M/tracking.py:427`) and
  `UsageAccumulator` (`M/usage.py:78`) coerce missing counts to `0` and **estimate** tokens from text at about
  4 characters a token. That is the opposite of the engine's rule that missing is not zero.

### 3. `reasoning_tokens`

- **anthropic.** SDK 0.116 `Usage.output_tokens_details.thinking_tokens` exists. LangChain does not copy it into
  `usage_metadata.output_token_details`, but `llm_output` (the whole response dump) is merged into
  `response_metadata` for a single-generation result (`SP/langchain_core/language_models/chat_models.py:2199`).
  So it is at `response_metadata["usage"]["output_tokens_details"]["thinking_tokens"]` (probe:
  `{'thinking_tokens': 5}`). When the API omits it the value is `None`, which reads correctly as "no split".
  This holds on the **non-streamed** path only. If an enclosing `astream_events` tap makes LangChain stream
  (see item 11), `response_metadata["usage"]` is not the response dump.
- **openai / openrouter.** `output_token_details.reasoning` comes from
  `completion_tokens_details.reasoning_tokens` (`SP/langchain_openrouter/chat_models.py:1566`). It is `None`
  when unreported.
- **claude_cli.** Hidden. `_usage_metadata` builds no `output_token_details`. The raw CLI usage dict rides on
  `response_metadata["usage"]` (`M/providers/claude_cli.py:1458`). Whether the bundled CLI forwards
  `output_tokens_details` there could not be checked without running the CLI.

### 4. `cost_usd` and `price_source`

- **Models computes cost in exactly one place, and never returns it.** `UsageTrackingCallback.on_llm_end`
  (`M/tracking.py:1104`) multiplies **capability** per-token rates (`ModelCapabilities.cost_per_input_token` /
  `cost_per_output_token`, wired by the factory at `M/factory.py:161`) by the extracted tokens. The result goes
  to OTel, Prometheus and the attached sinks. Nothing reaches the `AIMessage`. The built-in capability
  registrations carry no costs (`M/providers/*.py`), so it is `None` unless a host loaded rates through
  `registry_loader`. `price_lookup` is never called by the factory or the tracker.
- **`price_lookup`** (`M/price_lookup.py`) works like this:
  - It is async and network-only: `lookup_price(client, *, provider, api_name)` (`:613`). The caller owns an
    httpx-like client.
  - The OpenRouter catch-all (`:279`) GETs the **whole** `https://openrouter.ai/api/v1/models` catalogue on
    **every** lookup (`:271`). There is no cache at any level.
  - Offline, the transport error propagates (`raise_for_status` or a connection error). A parse miss returns an
    all-`None` `PriceLookup`.
  - Identifier: the `(provider, api_name)` pair, matched conservatively (`match_price`, `:163`): exact id, then
    `provider/api_name`, then a last-segment or normalised match confined to the provider's namespace.
  - Probe (offline `match_price`):
    - `anthropic` + `claude-sonnet-4-6` matched `anthropic/claude-sonnet-4.6`.
    - `openai` + `gpt-5-mini` matched.
    - `openai` + `gpt-5-mini-2025-08-07` (a **served** id) did **not** match.
    - `openrouter` + `~anthropic/claude-sonnet-latest` matched exactly.
    - `openrouter` + a served id `anthropic/claude-4.6-sonnet-20260201` did **not** match.
  - So price by the **requested** `model`, which is also what the engine's `model` is for (spend attribution).
  - Returns per-1M `Decimal` **input and output only** (`_result`, `:206`). OpenRouter's `input_cache_read` /
    `input_cache_write` rates are ignored, so pricing a cached Anthropic call from `input_tokens` overstates it.
    There is no request fee. Reasoning tokens are billed as output on both providers and are already inside
    `output_tokens`.
  - `PriceLookup.source` is `"openrouter"` / `"voyage"` and `matched_id` names the catalogue row: material for
    `price_source`.
- **openrouter** reports its own figure: `response_metadata["cost"]` (and `cost_details`)
  (`SP/langchain_openrouter/chat_models.py:714-717`; probe: `0.00123`). The SDK's `ChatUsage` declares `cost`,
  `cost_details` and `is_byok`. LangChain copies `cost` and `cost_details` but **drops `is_byok`**. Under BYOK,
  `cost` is OpenRouter's fee and the upstream charge is in `cost_details.upstream_inference_cost`, so the
  adapter should read `cost_details` too.
- **claude_cli** puts `total_cost_usd` from `ResultMessage` into `response_metadata`
  (`M/providers/claude_cli.py:1447`). On a subscription this is the CLI's API-equivalent estimate, not a charge.
  `price_source` must say so, or a subscription run reads as having spent dollars.

### 5. `model` and `served_model`

- `model`: every provider holds the requested id on the instance (`ChatAnthropic.model`,
  `ChatOpenAI.model_name`, `ChatOpenRouter.model_name`, `ClaudeCodeChatModel.model`). The adapter also has the
  `model_id` it passed to `create_chat_model`.
- **anthropic.** `response_metadata["model"]` is the raw response field. LangChain copies it to `model_name`
  only when present (`SP/langchain_anthropic/chat_models.py:1765`). Probe: requested `claude-sonnet-4-6`,
  response `claude-sonnet-4-6-20260201`, both visible.
- **openai.** `response_metadata["model_name"] = response_dict.get("model", self.model_name)`
  (`SP/langchain_openai/chat_models/base.py:1862`). The SDK dump always carries the key, so a null response
  model gives `None` (probe confirmed), not the request.
- **openrouter.** `generation_info["model_name"] = chunk_dict.get("model") or model_name`
  (`SP/langchain_openrouter/chat_models.py:90`). **An empty `model` falls back to the requested id.** The probe
  sent `model: ""` and got back `~anthropic/claude-sonnet-latest`, the alias, which is exactly the
  served-equals-requested defect the engine's docstring warns about. A missing `model` key cannot get this far:
  the SDK's `ChatStreamChunk.model` is required and raises a pydantic `ValidationError` (probe). The serving
  upstream (`"provider": "Anthropic"`) reaches `response_metadata["provider"]` only because models monkeypatches
  the SDK's response models (`declare_served_provider`, `M/providers/openrouter.py:122`, run at class build
  `:173`); `M/tracking.py:401` `served_provider()` reads it.
- **claude_cli.** Hidden. `AssistantMessage.model`, `ResultMessage.model_usage` (keyed by model) and
  `stop_reason` exist in claude-agent-sdk 0.2.163, but `_generation_info` (`M/providers/claude_cli.py:1431`)
  copies none of them. The only route is the private, unsynchronised `self._last_result` (`:891`, `:1021`),
  which concurrent calls on one instance overwrite. `fallback_model` is a forwarded kwarg (`:217`), so the CLI
  can answer from another model with nothing on the message saying so.

### 6. `stop_reason`

The engine's vocabulary is `end_turn | max_tokens | content_filter | error` (`E/schema/completion.py:48`). The
three non-`end_turn` members are the cut-short set (`E/kernel/provider.py:190-210`). `error`'s wording is "the
provider returned no choices at all".

| provider value | engine | note |
|---|---|---|
| anthropic `end_turn`, `stop_sequence` | `end_turn` | |
| anthropic `max_tokens` | `max_tokens` | |
| anthropic `refusal` | `content_filter` | probe: `stop_reason="refusal"` surfaced |
| anthropic `tool_use` | `end_turn` | the adapter binds no tools; the content would be empty |
| anthropic `pause_turn` | **none fits** | a server-tool pause, so cut short. `error`'s wording ("no choices") would mislead. Gap |
| openai / openrouter `stop` | `end_turn` | |
| `length` | `max_tokens` | |
| `content_filter` | `content_filter` | |
| `tool_calls` / `function_call` | `end_turn` | |
| openrouter `error` (a real finish reason there) | `error` | the wording mismatch again |
| finish reason absent or `None` (stream ended without a finish chunk) | **none fits** | must be treated as cut short. Gap |
| openai `refusal` non-null with `stop` | `content_filter` | adapter-detected |
| claude_cli | **hidden** | models emits only `stop` / `tool_calls` (`M/providers/claude_cli.py:1453`). A real `max_tokens` or `refusal` reads `stop`, so the engine reads finished |

- The engine has no member for "cut short, cause unknown" (pause, absent, an unknown future value) or for
  refusal as distinct from a filter.
- The engine's `error` ("no choices at all") is **unreachable through LangChain** for openai and anthropic. An
  empty or null `choices` raises inside LangChain (`KeyError` / `TypeError`,
  `SP/langchain_openai/chat_models/base.py:1820-1840`), so the adapter sees an exception, not a result.
- An unknown value must not default to `end_turn`.

### 7. `temperature`

- **anthropic.** A `None` temperature is not sent (probe: payload had no `temperature`). A bound one is sent
  verbatim (probe: `bind(temperature=0.0, thinking=…)` sent both). The API refuses any temperature but 1
  together with extended thinking, and LangChain does not drop it. The adapter must omit it when it enables
  thinking, and report `None`.
- **openai.** `validate_temperature` (`SP/langchain_openai/chat_models/base.py:1155`) **silently pops** a
  non-1 temperature at construction for `gpt-5*` (non-chat, unless effort is `none`), and sets `temperature=1`
  for `o1*` when none is given. Probe: `temperature=0.2` on `gpt-5-mini` left `inner.temperature=None` and the
  request carried none. So "as sent" is `inner.temperature` read after construction. A per-call
  `bind(temperature=…)` is **not** filtered (probe: `0.5` sent), so a per-call temperature on gpt-5 is a 400.
  **Build temperature into the instance, then read it back.**
- **openrouter.** Sent verbatim when set (`_default_params`, `SP/langchain_openrouter/chat_models.py:786`). By
  default OpenRouter drops a parameter the routed upstream ignores. Under `require_parameters` (structured
  calls) an upstream without temperature support is excluded from routing. Either way, "as sent" is what the
  adapter bound.
- **claude_cli.** Never sent. `create_subscription_chat` keeps only `_FORWARDED_KWARGS`
  (`M/providers/claude_cli.py:1514`), and a per-call temperature is named in the "were not sent" warning
  (`:633`). The bundled CLI takes no temperature option. Report `None`.

### 8. `response_format`

The engine sends `{"type":"json_schema","json_schema":{"name","strict":True,"schema"}}`, already projected onto
the portable strict subset (`E/kernel/authored.py:329`; the judge's at `E/run/judge.py:461`). `json_object`
(`E/schema/completion.py:27`) is sent by no engine call. The `SimulatorLLM` docstring (`:371`) still says the
variation generator sends it; that line is stale against the `JSON_OBJECT_RESPONSE_FORMAT` comment.

- **Use `structured_output_kwargs(provider, schema, name=, strict=)` + `model.bind(**kwargs)`**
  (`M/providers/structured_output.py:144`), **not** `with_structured_output`. The bind route keeps a plain
  `AIMessage`, so the raw JSON text is `msg.text` and every metadata path above still holds.
  `with_structured_output` returns the parsed dict. `include_raw=True` returns `{raw, parsed, parsing_error}`,
  and on a dict schema it requires a top-level `title` (probe: `ValueError: … must have a top-level 'title'`).
  The factory's `_CallbacksKeptBinding` keeps tracker and breaker through `bind`
  (`M/factory.py:52`; probe: `type(bound)` stays `_CallbacksKeptBinding`).
- **anthropic.** `output_config={"format":{"type":"json_schema","schema": transform_schema(schema)}}`
  (`M/providers/anthropic.py:77-170`). `transform_schema` moves non-portable keywords into `description`, which
  the engine has already stripped. A bad schema raises `StructuredOutputSchemaError` locally.
  - **No JSON-object mode exists.** `transform_schema({"type":"object"})` gives
    `{"type":"object","properties":{},"additionalProperties":false}` (probe), which forces `{}`. The adapter
    must not translate `json_object` that way: send no directive and validate on return.
  - **Bind trap (probe):** `bind(output_config={format}).bind(output_config={"effort":…})` sent only
    `{"effort": "low"}`. The second bind **replaced** the first and the schema was lost silently. Format and
    effort must go in **one** `output_config` (probe: one merged bind sent both).
- **openai.** `extra_body={"response_format": {...}}` (`M/providers/openai.py:104`) stays on the plain
  `.create()` path and returns raw text (probe: sent intact). `json_object` should also go via `extra_body`:
  a top-level `response_format` kwarg diverts ChatOpenAI to `.parse()`
  (`SP/langchain_openai/chat_models/base.py:1639`).
- **openrouter.** Top-level `response_format` + `provider={"require_parameters": True}`
  (`M/providers/openrouter.py:109-119`), merged into the model's own routing (`_with_model_routing`, `:182`).
  Probe: sent intact through the SDK's typed `ChatFormatJSONSchemaConfig`. When the served model cannot honour
  the directive, `require_parameters` makes OpenRouter refuse the request. OpenRouter documents that refusal as
  HTTP 404 "No endpoints found that support the requested parameters", which the SDK raises as
  `NotFoundResponseError` (the probe confirms the SDK mapping; the live wording is from OpenRouter's docs, not
  observed here). Without `require_parameters` (for instance a raw `bind(response_format={"type":
  "json_object"})`, which probe-sent fine) the parameter is silently dropped and the model answers in prose.
- **claude_cli.** The `output_config` is translated to the Agent SDK's `output_format`, which becomes CLI
  `--json-schema` (`M/providers/claude_cli.py:289`, `:631`). Any other `output_config` shape raises
  `ValueError`. The CLI does not constrain decoding. It validates the model's `StructuredOutput` tool call and
  re-asks, up to `_STRUCTURED_OUTPUT_ATTEMPTS=5` attempts in `_STRUCTURED_OUTPUT_TURNS=6` turns (`:240-255`,
  `:660`). Each re-ask is a billed generation that is invisible except as `response_metadata["num_turns"]`.
  The JSON schema is part of the CLI pool's launch key (`M/claude_cli_pool.py` module docstring), so each
  distinct judge criteria set launches its own CLI (about 2.4 s).

### 9. Reasoning effort

The engine has seven words, `max, xhigh, high, medium, low, minimal, none` (`E/schema/models.py:2795`). It asks
either by effort level or by a token budget, never both (`ClientRequestSettings`, `:2805`). The judge uses a
budget: `reasoning_max_tokens=8192` plus `strict_output` (`E/run/judge.py:87-131`). The simulator uses effort
`minimal` (`E/run/simulator.py:141`). `capabilities.py` has **no** reasoning, effort or temperature metadata.
Everything is a provider kwarg passed through `create_chat_model(**provider_kwargs)`, or a per-call `bind`.

- **anthropic** (`SP/langchain_anthropic/chat_models.py:1002-1060`, merge at `:1317-1330`):
  - `effort: Literal["max","xhigh","high","medium","low"]`. **`minimal` and `none` raise `ValidationError` at
    build** (probe), so the simulator's `minimal` cannot be sent.
  - A budget is `thinking={"type":"enabled","budget_tokens":N}`. Anthropic requires N ≥ 1024 and N below
    `max_tokens`.
  - `none` means: send no `thinking`, or `{"type":"disabled"}`.
  - Per call: `bind(thinking=…)` works (probe). `bind(output_config={"effort":…})` works, **merged with any
    format into one dict**. `bind(effort=…)` raises `TypeError: AsyncMessages.create() got an unexpected
    keyword argument 'effort'` (probe).
- **openai.** `reasoning_effort: str | None` (the API takes `none`/`minimal`/`low`/`medium`/`high`/`xhigh`,
  model-dependent; `max` is not an OpenAI word). Per-instance or `bind(reasoning_effort=…)` (probe: sent).
  **There is no token budget**, so `reasoning_max_tokens` cannot be expressed.
- **openrouter.** `ChatOpenRouter.reasoning: dict` is passed to the SDK's typed `ChatRequestReasoning`, whose
  only fields in openrouter 0.10.8 are `effort` (Literal of exactly the engine's 7 words) and `summary`. **The
  SDK strips every other key without error.** Probe: `bind(reasoning={"max_tokens": 8192})` and
  `bind(reasoning={"enabled": False})` both went out as `"reasoning": {}`; only `{"effort": "minimal"}`
  survived. `extra_body` is not accepted (`TypeError`), and the SDK has no raw-body escape. So **the judge's
  `JUDGE_REQUEST_SETTINGS` budget cannot be sent through models' OpenRouter model**, and neither can a
  classifier's `{enabled: false}`. Effort words pass per call.
- **claude_cli.** `thinking` / `effort` (Messages-API shapes) per instance (`_FORWARDED_KWARGS`) or per call
  (bind kwargs reach `_build_options(**overrides)`, `M/providers/claude_cli.py:618-628`). Unset, they are pinned
  to `{"type":"disabled"}` / `"high"` (`:274-278`), and the effort is also written to
  `CLAUDE_CODE_EFFORT_LEVEL`.

### 10. Failures

Neither LangChain nor models wraps a provider exception, apart from two LangChain context-overflow subclasses
and models' own CLI types. The circuit breaker adds one type. Probe chains:

- **anthropic.**
  - `RateLimitError` (429), `BadRequestError` (400), `AuthenticationError` (401), `OverloadedError` (529).
  - A 402 is a **bare `APIStatusError`**; there is no billing subclass.
  - `AnthropicContextOverflowError(BadRequestError)` when the message says "prompt is too long"
    (`SP/langchain_anthropic/chat_models.py:860-875`).
  - `APIConnectionError` / `APITimeoutError` for transport.
  - All carry `status_code` and `body` (a dict). **`str(exc)` is `"Error code: N - {body repr}"`**: payload
    included, so it must be withheld.
- **openai.**
  - `RateLimitError` / `BadRequestError` / … (`status_code`, `body`, `code`). `str(exc)` embeds the body.
  - 429 with `code="insufficient_quota"` is an **account** refusal that looks like a rate limit.
  - `OpenAIContextOverflowError`.
  - A 200 whose body carries `error` raises a bare **`ValueError(<the error dict>)`**: payload in the text, and
    no status.
- **openrouter.**
  - `openrouter.errors.*ResponseError` (`PaymentRequiredResponseError` 402, `UnauthorizedResponseError` 401,
    `ForbiddenResponseError` 403, `TooManyRequestsResponseError` 429, `BadRequestResponseError` 400,
    `NotFoundResponseError` 404, `ProviderOverloadedResponseError`, …).
  - All are `OpenRouterError` with `status_code`, `body` (str), `headers`, `raw_response`. Typed errors'
    `str()` is the message only (probe: `'Insufficient credits'`). **`OpenRouterDefaultError`'s `str()`
    appends `Body: <full body>`** (`SP/openrouter/errors/openrouterdefaulterror.py`).
  - A mid-stream error is a bare **`ValueError("OpenRouter API returned an error during streaming: <msg>
    (code: N)")`** (`SP/langchain_openrouter/chat_models.py:704-709`); the status exists only inside the text.
  - `ModelCallTimeout` (`M/errors.py:22`) when no chunk arrives within 120 s
    (`M/providers/_name_translation_mixin.py:237-245`).
  - A response that fails the SDK's schema is a **raw `pydantic_core.ValidationError`** (probe), which
    `is_provider_error` classifies as **not** a provider failure.
- **claude_cli.**
  - `ModelProviderError` / `ModelRateLimitError` (`M/errors.py:30`, `:75`), with `detail`, `provider`,
    `reason` (the CLI's subtype or code, e.g. `authentication_failed`, `billing_error`, `rate_limit`,
    `error_max_structured_output_retries`), `status`, `resets`, `rejected_output`, `rejection`. Raised by
    `_cli_failure` (`M/providers/claude_cli.py:1311`).
  - `ClaudeCliPoolExhausted` / `ClaudeCliSessionError` are caught by the model and fall back to a private CLI
    (`:764-810`).
  - `ValueError` for an unsupported `output_config`.
  - claude-agent-sdk transport errors.
- **circuit breaker** (`M/circuit_breaker.py`):
  - The factory's default breaker is **process-wide, keyed by (provider, api_key)**, not by model or role
    (`M/factory.py:176`).
  - **Every** `on_llm_error` counts as a failure, caller 400s included. Five consecutive failures open it for
    30 s (`:110`, `:318-348`).
  - While open, `on_chat_model_start` raises `CircuitOpenError` (`raise_error=True`, `:396`) **before any
    request**. Probe: 0 transport hits. In half-open, a second concurrent call gets `CircuitOpenError(0.0)`
    (`:272-276`).
  - So the breaker never retries. It turns failures, possibly another role's or another consumer's on the same
    key, into later unattempted, unbilled `CircuitOpenError`s.
- **models' helpers do not cover the port.**
  - `friendly_api_error` (`M/errors.py:222`) recognises only **anthropic's** `APIStatusError` (`:251`).
    OpenAI and OpenRouter status errors fall to "Something unexpected went wrong (…)". It also assigns no
    account flag.
  - `is_provider_error` (`:287`) is a bool.
  - No helper extracts the status or the provider message across SDKs.

Mapping onto `ProviderFailure` (`E/schema/completion.py:416`):

- **`description`**: provider name + status + the provider's own message.
  - anthropic / openai: `body["error"]["message"]`; models' `_extract_provider_body_message` (`M/errors.py:164`)
    is private but shows the shape.
  - openrouter: `exc.message` for typed errors.
  - claude_cli: `exc.reason` / `exc.status` / `exc.detail`.
- **`payload_withheld=True`** for anthropic and openai `APIStatusError`, `OpenRouterDefaultError`, and the bare
  in-body `ValueError`s, whose `str()` or chain carries the body. It is `False` for `ModelCallTimeout`,
  `CircuitOpenError` and `ModelProviderError`, whose text is the provider's notice and nothing else.
- **`account_refused=True`** for:
  - 401 / 403 `AuthenticationError` / `PermissionDeniedError`.
  - Anthropic 402, or a 400 whose message is the credit-balance notice.
  - OpenAI 429 `insufficient_quota`.
  - OpenRouter 401 / 402. **Not** 403, which OpenRouter also uses for moderation.
  - CLI `authentication_failed` / `billing_error`, and `ModelRateLimitError` when it carries a subscription
    reset time.
  - `CircuitOpenError` with `credential_scoped=True` is a revoked or failing key on this credential. But it is
    the breaker's inference, and models' default scopes every breaker by credential, so the flag is set
    whenever the default breaker is in use.

### 11. Lifecycle

- `BaseChatModel` has no `aclose` and no async context manager (none in `SP/langchain_core/language_models/`
  except the stream object).
- **anthropic / openai.** The async SDK client is a `cached_property` per instance wrapping an
  **`lru_cache`d, process-shared** httpx client: `_get_default_async_httpx_client`
  (`SP/langchain_anthropic/_client_utils.py:65`, `SP/langchain_openai/chat_models/_client_utils.py:489-515`),
  keyed by base URL and timeout. Closing it would break every other instance in the process. The adapter's
  `aclose` must be a no-op. Nothing leaks per instance.
- **openrouter.** `ChatOpenRouter.validate_environment` builds a fresh `openrouter.OpenRouter` per instance
  (`SP/langchain_openrouter/chat_models.py:409-490`), which creates its own sync and async httpx clients
  (models passes `app_title=None`, `app_url=None`, so the SDK owns them). They are closed by a
  `weakref.finalize` at GC (`SP/openrouter/sdk.py:229`) or by the public context-manager protocol:
  `await chat.bound.client.__aexit__(None, None, None)` (`:291-296`). The adapter owns one SDK client per
  `create_chat_model` and should close it in `aclose`. If a host passes `default_headers`/`app_title`,
  LangChain supplies the httpx clients itself, the SDK will not close them, and only GC frees them.
- **claude_cli.** CLI subprocesses live in a process-wide pool (`claude_cli_pool()`,
  `M/claude_cli_pool.py:1869`), stopped by the host's `close_claude_cli_pool()` (`:1891`) or an `atexit` kill.
  The adapter must not close the pool per unit of work.
- **Ambient-context hazard (all providers).** The mixin merges the caller's context config
  (`merge_configs(ensure_config(None), config)`, `M/providers/_name_translation_mixin.py:433-441`). If the
  engine runs inside a host's `astream_events` tap, a streaming handler from that context makes anthropic and
  openai take the **streamed** path (`_should_stream`, `SP/langchain_core/language_models/chat_models.py:549`).
  That changes which metadata paths exist (item 3) and emits the judge's tokens into the host's UI stream.
  Building with `disable_streaming=True` (a `BaseChatModel` field, passed as a provider kwarg) stops it for
  anthropic and openai. OpenRouter streams regardless and is unaffected.
- A process-global LangChain LLM cache (`set_llm_cache`) would answer from cache with the original call's
  metadata and no spend. The models are built with `cache=None`, so this happens only if a host sets one.

### 12. Retries

Every retry below is invisible on the returned `AIMessage`. The probe got the same metadata keys after
529,529,200 as after a clean 200.

- **anthropic.** The SDK retries 408/409/429/≥500 (529 included) and connection and timeout errors, honouring
  `retry-after` (`SP/anthropic/_base_client.py:842-876`). The factory default is `max_retries=2`
  (`M/providers/anthropic.py:237`). Probe: 529,529,200 gave 3 transport hits and a result; 429 gave 3 hits,
  then raised. Note: a non-streamed call whose reply takes longer than `timeout=120` s raises
  `APITimeoutError`, **which is retried**. The provider may still finish and bill the first attempt.
  **Off:** `create_chat_model(..., max_retries=0)` (probe: 1 hit).
- **openai.** The same SDK pattern (`max_retries=2`, `M/providers/openai.py:154`). Probe: 500,500,200 gave 3
  hits; 429 `insufficient_quota` gave **3 hits**, i.e. it retries an account refusal. **Off:**
  `max_retries=0`.
- **openrouter.** `max_retries>0` installs a backoff `RetryConfig` (`SP/langchain_openrouter/chat_models.py:457-466`):
  initial 500 ms, ×1.5, `max_elapsed_time = max_retries × 150 000 ms`. The SDK retries **only `5XX`** plus
  connection and timeout errors (`SP/openrouter/chat.py:597-638`, `SP/openrouter/utils/retries.py:175-260`).
  The bound is **elapsed time (300 s at the default 2), not a count**. Probe: 502,502,200 gave 3 hits in
  2.0 s; 402, 429, 400 and 404 gave 1 hit each. **`max_retries=0` does not turn it off.** LangChain then
  passes no `retry_config`, `retries` stays UNSET, and the SDK's chat call reads UNSET as its own default:
  `BackoffStrategy(500, 60000, 1.5, 3600000)` on 5XX, a one-hour budget (`SP/openrouter/chat.py:596-606`). A
  later probe on the adapter branch sent a mocked 502 twelve times. **Off:** `model_kwargs={"retries": None}`,
  which the SDK reads as no retry (issue #776). This corrects the first pass of this investigation.
- **claude_cli.** The bundled CLI (2.1.286) retries its own API calls. `CLAUDE_CODE_MAX_RETRIES` defaults to
  10 (300 in one mode), clamped at 15 (read from the minified bundle: `gMo=10`, `hMo=300`, `OIe=15`).
  Models never sets it, `_FORWARDED_KWARGS` drops `max_retries`, and `env` is not forwarded either. Separately,
  the structured-output re-asks (item 8) are retries of a different kind. **Neither can be turned off through
  models.** The only partial signal is `response_metadata["num_turns"]`.
- **LangChain itself** adds no retry. `BaseChatModel` has none, and models applies no `with_retry` or
  `rate_limiter`.
- **The circuit breaker** never retries (item 10).
- **On the engine side.** `RequestCeiling` (`E/schema/completion.py:40`) and `price_ceiling` (`:330`) are
  written to *include* "SDK retries, their back-off sleeps" in one request's worst case. If the adapter keeps
  any SDK retries, those two must be sized for them:
  - anthropic / openai: (1 + max_retries) × (timeout + back-off).
  - openrouter: up to 300 s of back-off on top of a 120 s-per-chunk deadline.

  The design's "no hidden retries" rule is met only by `max_retries=0` on anthropic and openai, plus
  `model_kwargs={"retries": None}` on openrouter.

---

## What the adapter must do itself (no models change needed)

1. **Build per role.** Call `create_chat_model(model_id, api_key=…, provider=…, max_retries=0,
   disable_streaming=True, max_tokens=<role cap>, temperature=<t>, <reasoning kwargs>, breaker=<own>)`.
   - Anthropic's default `max_tokens` is the model profile's 64 000 (probe). OpenAI and OpenRouter default to
     none.
   - For the breaker, pass one of its own, e.g. `CircuitBreaker(provider, failure_threshold=sys.maxsize)`,
     or a breaker per role, so the evals run neither trips nor is tripped by the process-wide (provider, key)
     breaker.
   - Unwrap `chat.bound` (the public `RunnableBinding.bound`) to read instance fields.
2. **Send messages** as `[SystemMessage(system), HumanMessage(user)]` with `ainvoke`. Build everything
   per call (concurrency-safe: no tools are bound, so the mixin's reverse map is untouched).
3. **content** = `msg.text`, never `msg.content`.
4. **Tokens.** Read `usage_metadata` (`None` means unknown). Do **not** use `extract_usage` or
   `UsageAccumulator`, which coerce missing counts to 0 and estimate.
5. **reasoning_tokens.**
   - anthropic: `response_metadata["usage"]["output_tokens_details"]["thinking_tokens"]`.
   - openai and openrouter: `usage_metadata["output_token_details"]["reasoning"]`.
   - `None` when absent.
6. **cost_usd / price_source.**
   - OpenRouter: `response_metadata["cost"]` (BYOK: also `cost_details.upstream_inference_cost`), named
     `"openrouter:reported"`.
   - claude_cli: `total_cost_usd`, named as the CLI's notional figure.
   - anthropic and openai: `input_tokens × in_rate + output_tokens × out_rate` from `lookup_price(client,
     provider=<models provider>, api_name=<requested model_id>)`, with
     `price_source=f"{lookup.source}:{lookup.matched_id}"`.
   - **Cache the catalogue** (one fetch per run; `price_lookup` caches nothing).
   - Own the httpx client.
   - Treat a transport error or an all-`None` lookup as `cost_usd=None` / `price_source=None`, never `0`.
   - Note that cache-read discounts are not in `PriceLookup`.
7. **served_model.**
   - anthropic: `response_metadata["model"]`.
   - openai: `response_metadata["model_name"]`.
   - openrouter: `response_metadata["model_name"]`, but report `None` when it equals the requested id **and**
     the requested id is an alias (`~…` / `-latest`). The value is then LangChain's request fallback, not
     evidence. `response_metadata["provider"]` is the serving upstream, worth keeping on the side.
8. **stop_reason.** Map per the table in item 6. An unknown or absent value maps to a cut-short member, never
   `end_turn`. An OpenAI non-null `refusal` maps to `content_filter`.
9. **temperature** = what the request carried:
   - anthropic: what it bound, or `None` when thinking is on (and it must then not send one).
   - openai: `chat.bound.temperature` after construction (LangChain pops it for gpt-5); never pass temperature
     per call.
   - openrouter: what it bound.
   - claude_cli: `None`.
10. **response_format.**
    - `json_schema`: `chat.bind(**structured_output_kwargs(provider, rf["json_schema"]["schema"],
      name=rf["json_schema"]["name"], strict=rf["json_schema"].get("strict", True)))`.
    - anthropic: **merge** any effort into that same `output_config` (one bind; a second bind replaces it).
    - `json_object`: openai `bind(extra_body={"response_format": {"type": "json_object"}})`; openrouter
      `bind(response_format={"type": "json_object"})`; anthropic and claude_cli send nothing (no JSON mode;
      `{"type":"object"}` would force `{}`).
    - `ClientRequestSettings.strict_output` maps to OpenRouter's `require_parameters`, which the structured
      builder already sets. For an unstructured call bind `provider={"require_parameters": True}` yourself.
11. **Reasoning.** Map the 7 words.
    - anthropic: `effort` for `max…low`; `minimal` as `low` or refused (a policy choice, record it); `none` as
      no thinking / `{"type":"disabled"}`. `reasoning_max_tokens` as `thinking={"type":"enabled",
      "budget_tokens":N}`.
    - openai: `reasoning_effort` (the word; `max` maps to `xhigh` or is refused). `reasoning_max_tokens` is not
      expressible.
    - openrouter: `reasoning={"effort": w}`. `reasoning_max_tokens` is not expressible (see the next list).
    - claude_cli: `thinking` / `effort`.
    - Every "cannot send" must surface, never silently degrade, or the recorded `ClientRequestSettings`
      misstates the request.
12. **ProviderFailureDescriber.** Write it over the SDK types directly (item 10's mapping). Use `exc.body` /
    `exc.message` and `exc.status_code`, never `str(exc)`. Make it total over `BaseException`
    (`pydantic.ValidationError`, bare `ValueError`, `TimeoutError`, `CircuitOpenError`). models'
    `friendly_api_error` is not a substitute.
13. **aclose.** OpenRouter: `await chat.bound.client.__aexit__(None, None, None)`. Everything else is a
    no-op; never close the shared anthropic and openai httpx clients or the CLI pool.
14. **Retries.** `max_retries=0` on anthropic and openai; on openrouter also `model_kwargs={"retries": None}`,
    since `max_retries=0` alone falls back to the SDK's one-hour default. Otherwise size `RequestCeiling` and
    `price_ceiling` for them.

## What it cannot do without models changes

Each entry names the smallest change.

1. **claude_cli `stop_reason` (hidden; a truncation reads as finished).** Copy `ResultMessage.stop_reason` and
   `terminal_reason` into `_generation_info` and derive `finish_reason` from them, instead of synthesising
   `stop` (`M/providers/claude_cli.py:1431-1458`).
2. **claude_cli `served_model` (hidden).** In the same function, copy the last `AssistantMessage.model` (or the
   `ResultMessage.model_usage` keys) as `generation_info["model_name"]`. That also reveals a `fallback_model`
   switch.
3. **claude_cli `reasoning_tokens` (hidden).** In `_usage_metadata` (`:1462`), map the CLI usage's
   `output_tokens_details.thinking_tokens` (if the CLI forwards it) into `output_token_details.reasoning`, and
   say `None` when it does not.
4. **OpenRouter `reasoning.max_tokens` / `reasoning.enabled` (silently stripped).** This blocks the judge's
   `JUDGE_REQUEST_SETTINGS` budget on the OpenRouter route. Extend `declare_served_provider`'s pattern
   (`M/providers/openrouter.py:122`): declare `max_tokens`, `enabled` and `exclude` on the SDK's
   `ChatRequestReasoning`, so the typed request keeps them. (Or bump to an `openrouter` SDK release that
   declares them.)
5. **claude_cli hidden retries.** Forward a `max_retries` (and `max_tokens`) kwarg onto the CLI's env
   (`CLAUDE_CODE_MAX_RETRIES`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS`) in `_build_options` (`:618-650`). Today neither
   the retry count nor the output cap can be set, so `price_ceiling` for a CLI model can only assume
   `capabilities.max_output_tokens` and an 11-attempt worst case. The structured-output re-asks are a separate
   design choice; exposing `num_turns` (already on `response_metadata`) is the visible half.
6. **Optional (convenience; the adapter can work around each):**
   - a "no breaker" factory option (`breaker=False`);
   - `PriceLookup` carrying cache-read and cache-write rates (`M/price_lookup.py:206`) plus a catalogue cache;
   - a cross-SDK `describe_failure(exc) -> (provider, status, provider_message, account_refused,
     payload_withheld)` beside `friendly_api_error`.

   None of these blocks the adapter. An `as_completion_client()` in models is not needed: everything else is
   reachable through public LangChain surfaces.

## Versions inspected

Taken from the eval-adopt `.23` branch's workspace venv, whose lock is `origin/develop`'s:

| package | version |
|---|---|
| 3tears-models | 0.66.0 (`origin/develop` `ff00526c5` plus the version bump) |
| 3tears-evals | 0.66.0 (eval-adopt `.23` branch at `741c00467`) |
| langchain-core | 1.6.3 |
| langchain-anthropic | 1.4.8 |
| langchain-openai | 1.3.5 |
| langchain-openrouter | 0.2.9 |
| openrouter (SDK) | 0.10.8 |
| anthropic | 0.116.0 |
| openai | 2.45.0 |
| langchain-claude-code | 0.1.0 |
| claude-agent-sdk | 0.2.163 (bundled Claude Code CLI 2.1.286) |
| httpx | 0.28.1 |
| Python | 3.14.6 |

**Not verified empirically** (no network and no CLI run):
- the live OpenRouter 404 wording;
- whether the CLI's usage dict carries thinking tokens;
- CLI retry behaviour beyond the constants read from its bundle.
