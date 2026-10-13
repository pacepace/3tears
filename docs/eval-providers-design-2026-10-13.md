# Providers for 3tears-evals: `3tears-evals[models]`, recorded completions, live examples

Status: design, 2026-10-13. Executes S1 of `eval-readiness-report-2026-10-13.md`, and settles the
examples question that came out of the complexity pass. Ships on the eval-adopt line as `.24`.

## BLUF

The engine stays provider-neutral and pydantic-only. `3tears-evals[models]` adds ONE adapter that turns any
chat model built by `3tears-models` into the evals completion client, priced through the family's price-source
registry and with failures described through models' error taxonomy. No provider SDK code and no price table
enter evals. A recording client in `threetears.evals.testing` records a live run once and replays it, so the
examples run live by default and exactly offline with no stand-ins, and the adapter is tested per provider
without network. The dimensions LangChain hides are pinned by a test, not left to drift.

## Why this shape

- The family already has a provider layer. `3tears-models` ships factories for Anthropic, OpenAI, OpenRouter,
  the Claude CLI (subscription token), Voyage and Whisper, with capability metadata, usage tracking, circuit
  breakers, and `price_lookup`: a registry of price sources whose catch-all is OpenRouter's public models
  endpoint, which prices Anthropic and OpenAI models too and needs no key. Models bakes no prices; a host
  registers its own source ahead of the catch-all. Writing three SDK adapters and a price table in evals would
  duplicate all of it and then drift from it.
- Evals' port is the cleaner contract: structural protocols, injected by the host, one call. Making models
  its implementation converges the family on port-plus-injection without moving any code today. The port's
  module stays free of evals-specific types so it can be lifted into a contracts leaf later (the
  `media-contracts` pattern) without change.
- The one real consumer keeps its own client. Discodon's OpenRouter client with its catalogue pricing already
  implements the port and is unaffected.

## What ships

### `threetears.evals.providers` (new public root, extra `models`)

```
3tears-evals[models]  ->  3tears-models>=0.66.0,<0.67.0   (the lockstep bound; the bounds test enforces it)
```

- `models_client(model_id, *, api_key, provider=None, temperature=None, reasoning_effort=None, **provider_kwargs)
  -> CompletionClient`. Builds the chat model through `create_chat_model` and wraps it. One instance per
  (model, temperature, effort), which is how the engine's role factory asks for clients.
- `models_clients(*, api_keys: Mapping[str, str] | None = None, defaults: Mapping[CompletionRole, str])
  -> CompletionClients`. The host-profile factory: role, model and temperature to a client. Provider
  resolution is models' capabilities registry (an id the registry knows resolves on its own; `provider/…` is
  accepted as an explicit override; an id the registry cannot place is refused by name). Keys come from the
  mapping or the providers' own environment variables, and a missing key is refused by name at first use,
  never at import.
- `ModelsFailureDescriber -> ProviderFailureDescriber`. Maps models' `errors.py` taxonomy (and the raw SDK
  errors it wraps, with HTTP status where carried) onto `ProviderFailure` kinds. Nothing is retried under the
  engine: the adapter builds the chat model with retries off and no circuit breaker, and says so in its
  docstring; a host that wants a breaker composes it outside the engine's cell.
- The engine never imports `providers`; `providers` imports only public roots of evals and of models. The
  package matrix test already holds the first; a test holds the second.

### The adapter's `generate`

1. Messages: `system` as the system message, `user` as one human message.
2. `response_format`: `{"type": "json_schema", "json_schema": {"schema": …}}` goes through models'
   `structured_output` helper for the provider, and the adapter returns the RAW JSON TEXT (the engine parses
   and checks it; a parsed object is not what the port promises). `{"type": "json_object"}` is a prompt-side
   ask on every provider models supports; nothing is sent. A served model that cannot honour a schema is a
   refusal by name, not a silent plain-text reply.
3. Reasoning effort: the engine's seven words map per provider (budget tokens on Anthropic-style models,
   effort words on OpenAI-style, pass-through on OpenRouter) using models' capability metadata; a model that
   takes neither refuses an effort other than `none` by name.
4. The ten members, from the `AIMessage`'s `usage_metadata` and `response_metadata`:
   `content`, `input_tokens`, `output_tokens`, `reasoning_tokens`, `model` (as asked), `served_model` (as the
   provider named it; on OpenRouter also the serving provider, in the form the engine's floating-alias
   disclosure reads), `stop_reason` mapped into the engine's vocabulary, `temperature` as sent,
   `cost_usd` and `price_source` from `price_lookup` (per-million rates for provider + api model name; the
   source and its fetch time are the `price_source`; unmatched or offline is `None`, unpriced, never 0).
   What LangChain does not expose for a provider is `None`, and the dimensions test (below) pins which.
5. `aclose` closes whatever HTTP client the chat model owns, if LangChain exposes it; otherwise it is a no-op
   that the docstring admits.

### `threetears.evals.testing.RecordedCompletions`

A client that wraps any `CompletionGenerator` and records or replays it.

- `RecordedCompletions.record(inner, path)`: calls through, appends each result to a JSONL file keyed by
  sha256 of (system, user, response_format, model, temperature) plus a per-key call index, storing all ten
  members. `RecordedCompletions.replay(path)`: serves by key and index; a miss refuses by name with the
  re-record command. Prices are replayed as recorded, so no network.
- Public in the `testing` root, tier host-contract: recorded completions in CI is a pattern hosts use.
- Also the adapter's test fixture: one recording per provider under `tests/fixtures/providers/`, made once
  with real keys in the paid lane, committed.

### Examples, live by default

- Every model-calling example builds its client with `models_client(MODEL, api_key=…)` when the key is set,
  and `RecordedCompletions.replay(examples/recordings/<example>.jsonl)` when it is not. The code has no other
  branch: no stand-ins, no "built to differ", no `_live.py`. The printed line says which it did.
- `llm_analysis.py` keeps its demonstration of the citation refusal with a canned bad first draft loaded
  from a file, since a live first draft may not refuse.
- `rung_zero.py`, `cassettes.py` and `reports.py` are unchanged: they call no model.
- Recording is a documented command in `examples/README.md` (`RECORD=1 python llm_judge.py` re-records that
  example), run once with keys before a tag; the example tests replay.
- Outside a checkout the examples need `pip install "3tears-evals[models]"` plus the provider's SDK extra
  from models; the README says which.

### Tests

- `tests/test_models_adapter.py`: the conformance kit (`check_completion_conformance`) over the adapter per
  provider, on the recordings. Structured output returns raw JSON text. Effort mapping per provider. Failure
  description for each kind. No retry configured.
- `tests/test_models_adapter_dimensions.py`: per provider, a pinned table of which of the ten members the
  adapter can populate from LangChain and which are `None`. A LangChain or models upgrade that starts hiding
  or exposing a member moves the pin, and the commit says so. This is the "what LangChain hides" report as a
  contract.
- The paid lane (`test_live_example.py` pattern): each provider live, when its key is set.
- Doc and example tests replay the recordings; a stale recording (a prompt changed under it) fails by name.

### Docs, policy, tiers

- `docs/adopting-a-host.md`: a "Bringing a provider" section: the port, the shipped adapter, the recording
  client, and when to keep your own client (a catalogue, a gateway, a provider models lacks).
- `README.md` and `docs/tutorial.md`: install line gains `[models]`; the tutorial's judge step uses the
  adapter. `examples/README.md` rewritten for live-by-default.
- `MIGRATIONS.toml` unreleased notes: the new root and extra; `_live.py` gone from the examples.
- Tiers: `models_client`, `models_clients`, `ModelsFailureDescriber`, `RecordedCompletions` are
  host-contract. `price_source` text format is closure.
- `schema/completion.py` gets a module docstring sentence: it is the family's completion contract and must
  stay free of evals-specific types.

## What LangChain hides

Probed against models 0.66.0 (langchain-core 1.6.3, langchain-anthropic 1.4.8, langchain-openai 1.3.5,
langchain-openrouter 0.2.9, anthropic 0.116, openai 2.45, openrouter SDK 0.10.8) with mocked transports and
no provider call. Of the ten members, exposed directly: anthropic 5, openai 6, openrouter 7, claude_cli 5.
Everything else is derivable by the adapter from `usage_metadata` and `response_metadata`, except three
members on the Claude CLI route, which models' wrapper drops.

| | anthropic | openai | openrouter | claude_cli |
|---|---|---|---|---|
| `reasoning_tokens` | derivable (`response_metadata.usage.output_tokens_details.thinking_tokens`) | exposed | exposed | **hidden** |
| `cost_usd` | derivable (tokens × `price_lookup`) | derivable | exposed (OpenRouter's own figure) | exposed (notional on a subscription) |
| `served_model` | exposed | exposed | exposed, but falls back to the requested id when the chunk's model is empty | **hidden** (and a `fallback_model` switch is invisible) |
| `stop_reason` | derivable (map `stop_reason`) | derivable (map `finish_reason`; a `refusal` is `content_filter`) | derivable (`finish_reason` + `native_finish_reason`) | **hidden**: models synthesises `stop`, so a truncation reads as finished |
| `temperature` | what the adapter bound; must not be sent with thinking on | read back after build: LangChain pops it for gpt-5 | what the adapter bound | always `None` |
| `price_source` | adapter's own fact on every provider | | | |

Call-side facts the adapter must handle rather than inherit:

- **JSON mode.** Anthropic has none; `{"type": "json_object"}` sends nothing there and goes through
  `extra_body` on OpenAI and `response_format` on OpenRouter. A schema goes through models'
  `structured_output_kwargs` per provider and the raw JSON text comes back in `msg.text`. OpenRouter refuses
  an upstream that cannot honour a schema (404) rather than answering in prose, which is the behaviour we want.
- **Reasoning.** Anthropic's `effort` takes five of the engine's seven words (`minimal` and `none` are
  refused at build) and a budget goes as `thinking={"budget_tokens": N}`; OpenAI takes the effort word and no
  budget; OpenRouter takes all seven words but **the SDK silently strips `reasoning.max_tokens`**, so the
  judge's token budget cannot be sent on that route today. Every "cannot send" surfaces as a refusal by
  name, never a silent downgrade, or the recorded request settings would lie.
- **Retries.** The Anthropic and OpenAI SDKs retry twice on 408/409/429/5xx and timeouts (OpenAI retries
  `insufficient_quota` too); OpenRouter retries on a 300-second budget. All three are turned off with
  `max_retries=0`. The Claude CLI retries internally up to ten times and re-asks for a schema up to five
  times, and models gives no way to turn either off.
- **Breaker.** `create_chat_model` wires the process-wide (provider, key) circuit breaker; the adapter passes
  its own breaker with an unreachable threshold so an eval run neither trips nor is tripped by it.
- **Lifecycle.** Anthropic and OpenAI share a process-wide cached httpx client that must not be closed;
  OpenRouter owns one per instance and is closable; the CLI pool is closed by the host at shutdown.
- **Tokens.** Read `usage_metadata` directly; models' `extract_usage` and `UsageAccumulator` coerce missing
  counts to 0 and estimate, which the engine must never see.
- **Failures.** Describe over the SDK exception types directly, using `status_code` and `body`, total over
  `BaseException` (pydantic `ValidationError` for a malformed chunk, bare `ValueError` for an in-body error,
  `TimeoutError`, `CircuitOpenError`).

**Decision for `.24`:** the adapter supports `anthropic`, `openai` and `openrouter`. The Claude CLI route is
refused by name until models carries its real stop reason and served model, because a judge that cannot see a
truncation is a correctness hazard, not a missing nicety. The dimensions test pins the table above.

## Models-side changes

None block the adapter. Five would improve it and go to develop in their own PR (`packages/models` is not
on the eval-adopt line), after which a later tag enables the CLI route and the OpenRouter budget:

1. Claude CLI: carry `ResultMessage.stop_reason` / `terminal_reason` into `generation_info` and derive
   `finish_reason` from them instead of synthesising `stop` (`providers/claude_cli.py:~1431-1458`).
2. Claude CLI: copy the last `AssistantMessage.model` into `generation_info["model_name"]`, which also
   reveals a `fallback_model` switch.
3. Claude CLI: map thinking tokens into `output_token_details.reasoning` in `_usage_metadata` (`:~1462`),
   `None` when the CLI does not forward them.
4. OpenRouter: declare `max_tokens`, `enabled` and `exclude` on the SDK's `ChatRequestReasoning` the way
   `declare_served_provider` patches the response type (`providers/openrouter.py:122`), so the judge's budget
   survives the typed request.
5. Claude CLI: forward `max_retries` and an output cap onto the CLI's environment in `_build_options`
   (`:~618-650`), so the hidden retries and the price ceiling become settable.

Optional conveniences, not scheduled: `breaker=False` on the factory; cache-read and cache-write rates on
`PriceLookup` plus a catalogue cache; a cross-SDK `describe_failure` beside `friendly_api_error`.

## Sequencing and build plan

1. `.23` lands and is tagged (in flight).
2. `.24` on `evals/adopt-24`, three builders from briefs, then one review pass:
   - A: `providers` root, adapter, describer, factory, dimensions test, conformance test over recordings.
   - B: `RecordedCompletions`, its tests, the paid-lane fixture recorder.
   - C (after A and B merge): examples rewrite, recordings command, docs, tiers, MIGRATIONS, CHANGELOG.
3. The recording run, with the owner's keys: the per-provider adapter fixtures and the example recordings.
4. Full gates on the integration branch, PR into `feature/eval-adopt`, the owner's go, tag `.24`, Discodon
   told (no host change expected; its client is its own).

## Backlog

- Raw SDK adapters for a consumer who refuses LangChain.
- Lifting the completion port into a contracts leaf; `create_chat_model(...).as_completion_client()`.
- A `--live` flag on the example tests so the paid lane covers them per provider.
