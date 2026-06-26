# Feature request: provider-agnostic structured output on `NatsChatModel`

| | |
|---|---|
| **Type** | Feature / requirement (not a defect) |
| **Filed** | 2026-06-26 |
| **Filed by** | DIPP team (`14-eng-ai-dipp`), via the onboarding-agent work |
| **Requesting consumer** | DIPP runtime extractor-authorship epics (EXT-3X9M / EXT-7Q2K) |
| **Priority** | Medium — an *enabler/optimization*, not a blocker (DIPP can ship on the tool-calling path today; this upgrades it) |
| **DIPP tracking item** | `LLM-7K3Q` (in `14-eng-ai-dipp/.prawduct/backlog.md`) |
| **Background research** | `14-eng-ai-dipp/.prawduct/artifacts/research-extractor-authorship-2026.md` |

---

## 1. Summary (the ask)

Add a **provider-agnostic structured-output capability** to the gateway chat path so a caller can
say, at a high level, *"return an object matching this schema"* — and the gateway transparently
**uses the best mechanism the currently-routed model supports**, falling back and self-repairing
for models that don't support the strong path. Concretely: implement LangChain's
`with_structured_output(schema)` on `NatsChatModel` backed by a **capability-negotiated ladder**:

```
native json_schema/strict   →   forced tool-call (= schema)   →   json_object + schema-in-prompt
   (per-model upgrade)              (portable default)               (last resort)
                         ↓  every rung terminates in  ↓
        pydantic-validate  →  bounded reask on failure  →  caller's gate is final authority
```

The caller passes a Pydantic model (or JSON Schema); the mechanism is invisible to them; a validated
object comes back. This is the standard shape used by LangChain (`ProviderStrategy`/`ToolStrategy`)
and Instructor (`Mode.TOOLS → JSON` + reask) — we are adopting a proven pattern, not inventing one.

## 2. Motivation

DIPP's onboarding agent increasingly needs the LLM to **emit a typed, declarative spec** (a transform
proposal; later, an acquisition recipe) that downstream code validates and executes. That requires
reliably getting a **schema-conformant object** back from whatever model we're routed to.

Operating constraints that shape this request:
- **OpenRouter is the primary provider**, and provider/model choice is **deliberately agile** (expected
  to change often). The abstraction must not bind to any one vendor's structured-output API.
- **DeepSeek-V4 (`deepseek-v4-pro` / `deepseek-v4-flash`) is a likely primary model.** (Anthropic will
  *mostly not* be used.)
- Callers **always** validate the result against a Pydantic model and have a deterministic gate
  downstream, so this feature's job is **reliability + portability**, **not** correctness guarantees.

Today every consumer re-implements the same `bind_tools` + read-`tool_calls` + Pydantic-parse +
hand-rolled degrade dance (see DIPP's `src/dipp/agents/mapping_suggest.py`). That belongs in the
gateway, once.

## 3. Current state (what exists today)

- `NatsChatModel` (`aibots_agents.runtime.gateway_model`) is a LangChain `BaseChatModel` proxy that
  routes generation over NATS to the aibots Hub, which holds provider credentials and brokers to the
  upstream provider (OpenRouter et al.).
- It implements **`bind_tools`** and emits parsed **`tool_calls`** on the returned `AIMessage` — so
  the *tool-calling* structured path already works.
- It does **not** implement `with_structured_output` for the native (`json_schema`) or `json_object`
  mechanisms, and the **Hub does not pass `response_format` through** to the upstream provider.
- **Important existing behavior (must be preserved/handled):** per DIPP's notes, *"the gateway NEVER
  raises on model failure — `NatsChatModel._agenerate` catches and returns no `tool_calls`"*
  (`14-eng-ai-dipp/src/dipp/agents/mapping_suggest.py:15`). And *"`NatsChatModel` only emits a
  `parameters` block to the LLM when an `args_schema` is set"*
  (`.../tools/inspect_source.py:71`). The structured-output layer must treat **"failure == empty /
  no `tool_calls`"** as a first-class outcome (trigger reask or documented degrade), never assume an
  exception will surface.

## 4. The requirement

### R1 — High-level interface
Callers get `model.with_structured_output(MyPydanticModel)` (LangChain-standard) returning a runnable
whose output is a validated instance of `MyPydanticModel`. The caller specifies **what** structure they
want (the schema); they never specify **how** (the mechanism). A `method=` override and an
`include_raw=True` escape hatch should match LangChain's signature for drop-in compatibility.

### R2 — Capability-negotiated mechanism ladder (chosen per current model)
1. **Native `json_schema` / strict** — emit OpenRouter
   `response_format: {type:"json_schema", json_schema:{…, strict:true}}` **with**
   `provider:{require_parameters:true}` — **only when the routed model advertises support** (see R3).
   This is a per-model *upgrade*, not the baseline.
2. **Forced tool call (portable default)** — define one tool whose `parameters` is the schema and
   force it via `tool_choice`. This is `NatsChatModel`'s **existing** `bind_tools` path; it is the
   **most broadly supported** mechanism across OpenRouter's fleet (DeepSeek / Llama / Qwen / Claude…).
3. **`json_object` + schema-in-prompt (last resort)** — for models with neither `structured_outputs`
   nor `tools`. Inline the schema into the prompt (DeepSeek's JSON mode additionally **requires the
   word "json" in the prompt** and benefits from an example).

### R3 — Capability detection (no hand-maintained table)
Determine the rung from **OpenRouter's own model metadata**: `GET /api/v1/models` →
`supported_parameters` contains `structured_outputs` and/or `tools` (and you can server-side filter
`?supported_parameters=structured_outputs`). The Hub should surface this so `NatsChatModel` can pick
the rung at bind time (and/or the Hub picks it). Do **not** silently send `json_schema` to a
non-supporting model — without `require_parameters:true` OpenRouter **silently ignores** it and
returns unconstrained output (the dangerous case for a validation pipeline).

### R4 — Validation + bounded repair on EVERY rung
After every mechanism, **Pydantic-validate**; on `ValidationError`/`JSONDecodeError`, run a **bounded
reask** that feeds the validation error back into the next call (Instructor's pattern), capped at a
small N. This is **always on** — even native `json_schema` guarantees *shape, not values*, OpenRouter
does **not** re-validate against the schema, and `json_object` guarantees only valid JSON syntax. The
caller's deterministic gate remains the final authority regardless of mechanism.

### R5 — Hub plumbing
The aibots Hub must **pass `response_format` and `provider.require_parameters` through** to OpenRouter
(rung 1), and continue to pass `tools`/`tool_choice` (rung 2). Confirm whether the Hub currently
*strips* `response_format` or merely *doesn't forward* it — that changes the effort.

## 5. Verified technical facts (mid-2026, checked against primary sources)

These were verified this session; they justify the design above.

- **Tool-calling is the portable lowest common denominator.** OpenRouter standardizes `tools`/
  `tool_choice`/`tool_calls` across providers, and the set of tool-calling models is a **superset** of
  the native-`structured_outputs` set. OpenRouter's own framing: *"Structured Outputs is a subset of
  Tool Calling."*
- **DeepSeek-V4 has NO native `json_schema` on the final message.** Its `response_format` accepts only
  `text` / `json_object`. Strict JSON-Schema exists **only** as a *beta* mode constraining **tool-call
  arguments** (reduced keyword subset: no `minLength`/`maxLength`, no `minItems`/`maxItems`). LangChain's
  `ChatDeepSeek` defaults to `method="function_calling"` for exactly this reason. → For our primary
  stack, **tool-calling is the only reliable schema-shaped path.**
- **OpenRouter `json_schema` errors or silent-ignores on non-supporting models** unless
  `provider.require_parameters:true` (then it excludes/falls-back instead).
- **DeepSeek `json_object` gotchas:** requires the word "json" in the prompt; can **return empty
  content "occasionally"** (officially acknowledged); truncates if `max_tokens` too low. Reinforces
  R4 (validate + reask + treat empty as failure).
- **LangChain base `with_structured_output(method="function_calling")` is implemented in terms of
  `bind_tools`** — so on a model that already implements `bind_tools` + parsed `tool_calls` (which
  `NatsChatModel` does), the tool-calling path is reachable **almost for free**. Native `json_schema`
  and `json_object` require **overriding `with_structured_output`** on the model (what every provider
  integration does).
- **Don't bolt on Instructor** alongside LangChain (two competing structured-output stacks). It's
  MIT — **borrow** its reask wording + Tenacity retry shape as reference, don't adopt the client.
- **Anthropic** structured outputs are GA (`output_config.format` json_schema + strict tool use,
  constrained decoding) — supported, but **low priority** here since Anthropic is mostly not used.

## 6. Code surface (spans three sibling repos — please route accordingly)

> ⚠️ **Ownership note for triage:** `NatsChatModel` is **not** in 3tears core. This was filed into
> `3tears/incoming-bugs/` per the requester's instruction, but the implementation surface is primarily
> the **aibots** repos. Please re-route if 3tears isn't the owning team.

| Repo | Module / path | Role in this work |
|---|---|---|
| `14-eng-ai-bot-agents` | `aibots_agents.runtime.gateway_model.NatsChatModel` | **Primary.** Override `with_structured_output`; implement the rung ladder + reask; reuse existing `bind_tools`. |
| `14-eng-ai-bot` | the aibots **Hub** gateway | Pass `response_format` + `provider.require_parameters` through to OpenRouter; surface per-model `supported_parameters`. |
| `3tears` | `threetears.langgraph.caching.register_capability_provider` (`packages/langgraph/src/threetears/langgraph/caching.py`) | Existing capability-provider registration seam that already knows about `NatsChatModel` as a proxy — candidate hook for capability/cache integration. |

Reference consumer (the pattern to generalize): `14-eng-ai-dipp/src/dipp/agents/mapping_suggest.py`
(`bind_tools` → read `tool_calls` → Pydantic parse → degrade on no-tool_calls), and
`.../live_runner.py:816` (how `NatsChatModel` is constructed).

## 7. Acceptance criteria

1. `NatsChatModel(...).with_structured_output(SomePydanticModel)` returns a runnable that yields a
   validated `SomePydanticModel` instance against **DeepSeek-V4 via OpenRouter** (i.e. the
   tool-calling rung works end-to-end with no native `json_schema`).
2. Against a model that advertises `structured_outputs` (e.g. an OpenAI/Gemini/Fireworks model on
   OpenRouter), the **native `json_schema` rung** is used and sends `require_parameters:true`.
3. Against a tool-less, SO-less model, the **`json_object` rung** is used with the schema inlined.
4. A model that returns schema-violating output triggers **bounded reask** and ultimately either
   validates or returns a typed failure — **never** a silently-unvalidated object.
5. The gateway's **non-raising failure mode** (empty / no `tool_calls`) is handled as a retryable/
   degradable outcome, not an unhandled empty result.
6. Rung selection is driven by **model capability metadata**, not a hard-coded per-model table.
7. Existing `bind_tools` / agent-loop behavior is unchanged for callers that don't opt in.

## 8. Non-goals / out of scope

- Not a correctness oracle — value-level validation stays with the **caller's** deterministic gate.
- Not adopting Instructor as a dependency (reference only).
- Not logit-level constrained decoding (Outlines/XGrammar/etc.) — irrelevant to a hosted-gateway path.
- Not building DIPP's transform/recipe schemas — those live in DIPP; this only needs to *carry* them.

## 9. Open questions for the implementing team

1. Does the Hub currently **strip** `response_format`, or just not forward it? (Sets the effort.)
2. Should rung selection live in **`NatsChatModel`** (client-side, reading Hub-surfaced capability) or
   in the **Hub** (server-side, returning the validated object)? Recommendation: negotiation in
   `NatsChatModel`; passthrough + capability surfacing in the Hub.
3. Where should the **reask loop** live — in `NatsChatModel.with_structured_output`, or as a thin
   LangGraph wrapper? (Reusing the `3tears.langgraph` caching/provider seam is an option.)
4. Default **reask cap** and backoff (Tenacity-style)?

## 10. References (verified primary sources)

- OpenRouter structured outputs — https://openrouter.ai/docs/guides/features/structured-outputs
- OpenRouter provider routing (`require_parameters`, `allow_fallbacks`) — https://openrouter.ai/docs/guides/routing/provider-selection
- OpenRouter models / `supported_parameters` — https://openrouter.ai/docs/guides/overview/models
- OpenRouter tool calling — https://openrouter.ai/docs/guides/features/tool-calling
- DeepSeek JSON mode — https://api-docs.deepseek.com/guides/json_mode
- DeepSeek function calling — https://api-docs.deepseek.com/guides/function_calling
- DeepSeek-V4 release / model IDs — https://api-docs.deepseek.com/news/news260424 ; https://api-docs.deepseek.com/api/create-chat-completion
- LangChain `with_structured_output` / strategies — https://docs.langchain.com/oss/python/langchain/structured-output ; https://docs.langchain.com/oss/python/langchain/models
- LangChain `ChatDeepSeek` (default `function_calling`) — https://reference.langchain.com/python/langchain-deepseek/chat_models/ChatDeepSeek/with_structured_output
- Instructor modes + reask (MIT) — https://python.useinstructor.com/concepts/patching/ ; https://python.useinstructor.com/concepts/reask_validation/ ; https://python.useinstructor.com/integrations/openrouter/
- Anthropic structured outputs (GA) — https://platform.claude.com/docs/en/build-with-claude/structured-outputs
