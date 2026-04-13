# Discodon Memory Migration to 3tears Memory v2

How discodon's memory systems map onto 3tears memory v2, what moves, what stays, and where the seams are. This covers the entire lifecycle: collection, storage, retrieval, injection, administration, distillation, logging, and MCP review.

---

## Current discodon memory architecture

Discodon manages memory entirely in-process and in Cosmos DB. There is no vector store, no PostgreSQL, no embedding-based search. The system has two tiers of its own design:

| Discodon concept | Storage | Lifespan | Search method |
|---|---|---|---|
| **WorkingNote** | In-memory list, persisted to Cosmos entity doc | Capacity-bounded FIFO (default 50) | Keyword overlap scoring |
| **NarrativeWisdom** | In-memory list, persisted to Cosmos entity doc | Permanent (relevance-decayed) | Keyword overlap scoring |
| **Goal** | In-memory list, persisted to Cosmos entity doc | Until explicitly updated/removed | Direct enumeration |
| **TodoItem** | In-memory list, persisted to Cosmos entity doc | Until completed | Direct enumeration |
| **TurnRecord** | In-memory list (last 5), Cosmos persona_turns | 5-turn window + 7-day TTL in Cosmos | Not searched by content |
| **Identity (backstory, traits, style, directives)** | Cosmos entity doc | Permanent, admin-mutable | Not searched |
| **BigFiveTraits** | Cosmos entity doc | Permanent, admin-mutable | Not searched |

Key architectural facts:
- **No embeddings.** All retrieval is keyword-overlap scoring against current-turn perceptions.
- **No deduplication by similarity.** Note dedup uses normalized word overlap heuristics, not cosine distance.
- **Entity association is per-note.** `WorkingNote.about` (display name) and `about_source_id` (platform-qualified like `discord:123456`) tag person-specific notes. Wisdom has no entity tagging.
- **Distillation is notes → wisdom only.** No tiers beyond that. No cross-entity synthesis.
- **The LLM manages its own memory.** The reflect tool lets the entity create/update/delete notes and wisdom via tool calls during turns. 3tears extraction is a passive pipeline; discodon's is agent-driven.

---

## What moves to 3tears

### 1. WorkingNote → 3tears observation-tier memories

**Current:** `WorkingNote` has `content`, `created_at`, `created_turn`, `about` (display name), `about_source_id` (platform-qualified ID). Capacity-bounded at 50, FIFO eviction. Scored by keyword overlap with perceptions for prompt injection.

**v2 mapping:**

| WorkingNote field | v2 memory field | Notes |
|---|---|---|
| `content` | `content` | Direct |
| `created_at` | `date_created` | Direct |
| `created_turn` | `source_context.turn_number` | Turn number moves to provenance metadata |
| `about` | *(resolved via identity module)* | Display name → canonical UUID via `threetears.identity` |
| `about_source_id` | *(identity module platform mapping)* | `discord:123456` → `(discord, 123456) → canonical_id` via identity |
| *(implied from content)* | `about_ids` | Canonical UUID(s) from identity resolution |
| *(implied from content)* | `mentioned_entity_ids` | Secondary entities mentioned in the note |
| *(none)* | `importance` | Assigned at extraction (1-10) |
| *(none)* | `tier` | `observation` |
| *(none)* | `type_memory` | `preference`, `fact`, `action`, `belief` — determined by extraction |
| *(none)* | `source_type` | `discord_message`, `reflection`, etc. |
| *(none)* | `source_id` | Message ID or turn ID |

**What changes:**
- Notes stop being capacity-bounded FIFO. 3tears observations are permanent until consumed by distillation. The "50 notes max, oldest dropped" model is replaced by distillation's CONSUME operation.
- Entity association switches from display-name-based (`about: "Alice"`) to canonical-UUID-based (`about_ids: [uuid]`). The identity module handles the mapping. This is a fundamental shift — discodon currently relies on the LLM writing display names into the `about` field; v2 requires the extraction pipeline to resolve names to canonical IDs.
- Keyword-overlap scoring is replaced by multi-signal scoring (semantic + keyword + recency + importance + entity boost). This requires embeddings — the single biggest infrastructure addition.
- Per-note relevance scoring moves from the prompt-injection layer (section_formatters.py) to the retrieval layer.
- Note creation shifts from LLM tool calls (reflect__note) to the extraction pipeline. The LLM no longer explicitly decides what to remember — the extraction pipeline analyzes the conversation and extracts memories automatically.

**Migration complexity: HIGH.** This touches collection, storage, retrieval, and prompt injection. It's not a rename — it's a different philosophy of memory management.

### 2. NarrativeWisdom → 3tears pattern-tier memories

**Current:** `NarrativeWisdom` has `insight` (text), `learned_turn`, `relevance` (0-1). Created by distillation from notes or directly by the LLM via `reflect__new_wisdom`. Permanent. Scored by keyword overlap for prompt inclusion. Explicitly forbidden from being person-specific (distillation prompt rule 2).

**v2 mapping:**

| NarrativeWisdom field | v2 memory field | Notes |
|---|---|---|
| `insight` | `content` | Direct |
| `learned_turn` | `source_context.turn_number` | Provenance metadata |
| `relevance` | `importance` (partially) | Relevance 0-1 maps loosely to importance 1-10, but they're different concepts. Relevance is a decay/scoring signal; importance is an extraction-time assessment. |
| *(none)* | `tier` | `pattern` |
| *(none)* | `derived_from_ids` | Links to the observations this was distilled from |
| *(none)* | `consumed_by_id` | *(only on observations consumed into this pattern)* |
| *(none)* | `about_ids` | **Change:** v2 patterns CAN be entity-specific. Discodon wisdom is always general. |
| *(none)* | `confidence`, `epistemic_type` | Foundation fields — not used in discodon today |

**What changes:**
- Wisdom gains provenance. `derived_from_ids` traces back to the observations that were consumed. Discodon's distillation CONSUME operations don't preserve this link — the note just disappears.
- v2 patterns can be entity-specific (e.g., "Bob is a thrash metal fan"). Discodon's distillation prompt explicitly forbids person-specific wisdom. v2's richer type system means person-specific patterns are valid at the pattern tier — they're just `about_ids=[Bob], tier=pattern, type=preference`.
- The relevance field disappears. v2 uses importance (extraction-time) + recency (access-time decay) + semantic similarity. There's no persistent "how relevant is this wisdom" field that decays over time.
- Wisdom consolidation (dedup/merge) continues to exist in v2 but as part of the distillation engine rather than a separate post-processing step.

**Migration complexity: MEDIUM.** The core concept maps directly. The differences are in metadata, provenance, and the relaxation of the "no person-specific wisdom" constraint.

### 3. Distillation engine → 3tears within-entity distillation

**Current system in discodon:**

- **Trigger conditions:** 5 checks: global enabled, API key available, minimum notes count (per-entity or global), turns since last distillation, time since last distillation.
- **Process:** Snapshot notes + wisdom → LLM call with structured JSON response → apply operations (ADD/UPDATE wisdom, CONSUME/REFINE/KEEP notes).
- **Post-distillation:** Wisdom consolidation pass (dedup/merge overlapping wisdom entries).
- **Scheduling:** Background asyncio task spawned by `DistillationManager` after each turn if conditions met. Also force-able via MCP `entity distill`.
- **Prompts:** Customizable via ARIA template system. Default prompt is DJ-domain-specific (references "listeners", "tracks", "sessions").

**v2 mapping:**

| Discodon distillation | v2 distillation | Notes |
|---|---|---|
| Turn/time interval triggers | Importance-sum threshold + session boundary triggers | Different trigger model — importance-based rather than interval-based |
| CONSUME/REFINE/KEEP operations | Same names, but non-destructive CONSUME (soft-delete) | Discodon hard-deletes consumed notes. v2 preserves them. |
| Wisdom consolidation (dedup pass) | Part of within-entity distillation | Same concept, integrated |
| Entity-name-scoped prompts | `DistillationStrategy` protocol | App provides prompts via protocol implementation |
| No cross-entity synthesis | Cross-entity distillation (daily background) | **New.** Synthesizes `about=null` principles from patterns across entities. |
| No theory of mind | ToM inference in distillation prompts | **New.** Distillation asks "why?" not just "what?" |
| No belief handling | Belief-type patterns with confidence | **New.** Subjective interpretations become first-class memories. |

**What changes:**
- **Non-destructive CONSUME** is the big one. Discodon's distillation drops notes after consuming them. v2 soft-deletes and links via `consumed_by_id` / `derived_from_ids`. This means storage grows rather than cycling, but provides audit trails, deletion tracing, and the ability to recover from bad distillation.
- **Trigger model shifts from intervals to importance-sum.** Discodon triggers distillation after N turns or T seconds. v2 triggers when accumulated importance on new observations exceeds a threshold. Active entities with high-importance observations distill sooner; quiet entities with low-importance observations distill later. Discodon's per-entity overrides (`distillation_turns_interval`, `distillation_time_interval_s`, `distillation_min_notes`) would be replaced by per-owner threshold configuration.
- **Cross-entity distillation** is entirely new. A DJ entity that's observed hundreds of listeners' preferences can now synthesize `about=null` principles like "Metallica fans tend to enjoy Pantera." This runs on a background schedule, not triggered by individual turns.
- **Theory of mind** is new. Distillation prompts guide the LLM to infer motivations and classify results as `type=belief` with appropriate confidence levels, rather than just summarizing behavioral patterns.

**Migration complexity: HIGH.** The distillation engine is rewritten, not wrapped. Trigger logic, operation semantics (destructive → non-destructive), output format, and prompt structure all change. The app provides a `DistillationStrategy` implementation rather than a template string.

### 4. Entity identity → 3tears-identity module

**Current:** Discodon has no formal identity system. Entities are identified by Cosmos document ID (`id_entity`), and people the entity knows about are tracked by display name (`WorkingNote.about`) with an optional platform-qualified string (`about_source_id` like `discord:123456`). The MCP `entity people` action derives a "known people" list by grouping notes by `about` field.

**v2 mapping:**

| Discodon identity concept | v2 identity concept | Notes |
|---|---|---|
| `entity.id_entity` (UUID) | `owner_id` (canonical UUID) | The entity persona becomes a memory owner |
| `WorkingNote.about` (display name) | Identity module: display name → canonical UUID | Name resolution moves to the identity module |
| `WorkingNote.about_source_id` (`discord:123456`) | Identity module: `(platform, platform_id) → canonical_id` | Platform mapping is the identity module's core job |
| *(none — notes grouped by display name)* | Identity module: canonical entity with platform mappings | Formal entity records with merge/split |
| *(none)* | Cross-platform identity resolution | Same person on Discord and Mastodon gets one canonical ID |

**What changes:**
- **Person identification becomes a first-class system.** Instead of the LLM writing "Alice" into a note's `about` field and hoping names are consistent, the identity module maintains a canonical registry with platform mappings.
- **The `about_source_id` pattern gets formalized.** `discord:123456` becomes a platform mapping `(discord, 123456) → canonical_uuid`. This is what `about_source_id` was always trying to be — v2 makes it a real system.
- **Entity merge becomes possible.** If "Alice" and "Alice_gaming" turn out to be the same person, the identity module merges their canonical IDs and calls `merge_entity_ids()` on the memory module. Discodon has no equivalent — notes about "Alice" and "Alice_gaming" would stay separate forever.
- **The `entity people` MCP action becomes a query on the identity module** rather than a note-grouping heuristic.

**Migration complexity: MEDIUM.** The identity module is new infrastructure, but discodon already has the raw data (`about_source_id`). The migration is: for each unique `about_source_id`, create a canonical identity record. For notes without `about_source_id`, attempt name-based resolution.

### 5. Memory retrieval → 3tears entity-aware retrieval

**Current:** Discodon retrieval is entirely keyword-based:
- `score_wisdom()` in `section_formatters.py`: keyword overlap between wisdom text and current-turn perception keywords, weighted by the wisdom's relevance field.
- `score_note()`: meaningful-word overlap between note content and perception keywords, normalized by perception keyword count.
- Two-tier selection for notes: relevant notes above threshold, then recent backfill to minimum count.
- No embeddings. No semantic search. No entity boosting. No importance scoring.

**v2 mapping:**

| Discodon retrieval | v2 retrieval | Notes |
|---|---|---|
| Keyword overlap scoring | Keyword match (PostgreSQL FTS) — one of four signals | Keyword matching survives but becomes 10% of the score |
| *(none)* | Semantic similarity (pgvector) | **New.** Requires embedding infrastructure. |
| *(none)* | Recency decay on `date_last_accessed` | **New.** Replaces static relevance field. |
| *(none)* | Importance scoring (from extraction) | **New.** |
| *(none)* | Entity boosting (about_ids / mentioned_entity_ids match) | **New.** Memories about present entities score higher. |
| Per-section scoring (wisdom vs notes) | Unified multi-signal + budget allocation | Entity-specific 70% / general 30% budget split |
| Two-tier note selection (relevant + recent backfill) | MMR reranking within budget partitions | MMR prevents redundancy |
| `format_wisdom()` + `format_notes()` injection | Entity-grouped retrieval results | App still owns prompt formatting, but input is grouped by entity |

**What changes:**
- **Embeddings are required.** This is the single biggest infrastructure change. Discodon has zero embedding infrastructure today. v2 requires an embedding provider (VoyageAI, OpenAI, etc.) and vector storage (pgvector in PostgreSQL).
- **The database changes.** Discodon uses Cosmos DB for everything. v2 uses PostgreSQL with pgvector. This isn't just a library swap — it's a different database engine with different operational characteristics.
- **Scoring is unified.** Instead of separate scoring functions for wisdom and notes, v2 has one multi-signal scorer that works across all tiers. The app no longer manually formats and scores each memory type separately.
- **Entity-grouped output.** Instead of a flat scored list, v2 returns memories grouped by entity (`About Bob: ... About Alice: ... General: ...`). The app still formats this into prompts, but the structure is richer.

**Migration complexity: VERY HIGH.** Requires PostgreSQL + pgvector infrastructure, an embedding provider, and a complete rewrite of the retrieval path. This is the hardest single migration item.

---

## What stays in discodon

### 1. Goals

Goals are a cognitive-architecture concept, not a memory concept. 3tears doesn't model goals — it models what the agent remembers about the world. An agent's goal "engage listeners and help them discover music" is a directive, not a memory.

Goals stay in discodon's PersonaComponent. If goal-related observations arise ("I successfully introduced Bob and Alice over Metallica"), those become 3tears `action`-type memories. The goal itself remains in the app's cognitive architecture.

**Interaction with 3tears:** The reflect__goal_update tool stays in discodon. Goal updates might generate observations (via extraction) if they represent significant decisions.

### 2. TodoItems

Same reasoning as goals. Todos are forward-looking intentions in the agent's cognitive architecture. 3tears v2 has `commitment`-type memories as a [foundation] type, but the lifecycle engine doesn't ship in v2.

Todos stay in discodon. If/when 3tears ships commitment lifecycle tracking, there's a natural migration path: `TodoItem` → `commitment`-type memory with `active → fulfilled / broken / expired` lifecycle.

### 3. BigFiveTraits, backstory, style, directives, example_statements

These are identity/persona configuration, not memory. They define who the entity *is*, not what it *remembers*. They stay in discodon.

**Interaction with 3tears-identity:** The identity module stores entity metadata (display names, types, app-specific attributes). Some persona metadata might live in the identity module's entity record, but the OCEAN model, backstory, and behavioral directives are too domain-specific.

### 4. TurnRecord and turn history

Turn records are operational telemetry, not long-term memory. The 5-turn sliding window serves as immediate conversation context. Turn logs in Cosmos (7-day TTL) are observability data.

Turn records stay in discodon's logging system. If memories need to be extracted from turn content, that happens through the 3tears extraction pipeline, not by storing turn records in 3tears.

### 5. Attention budget

The attention budget (`AttentionBudget` with total/spent/remaining) is a per-turn cognitive resource constraint. It's part of the agent's decision-making architecture, not memory.

Stays in discodon.

### 6. Relevance classifier

The classifier that determines whether to process a turn (relevance level, complexity scoring) is an app-level behavioral decision. 3tears extraction has its own gate system (heuristic gates + worthiness check), but the classifier's role in deciding whether the entity should even *engage* is a cognitive-architecture concern.

The classifier stays. The extraction gate config may reference classifier outputs (e.g., "don't extract from low-relevance turns").

### 7. Role system (orchestrator/participant/observer)

Role-based behavior is cognitive architecture. An observer's relaxed extraction gates are configured in 3tears (per-owner gate config), but the role assignment and behavioral rules stay in discodon.

---

## The seams: where discodon and 3tears interact

### Seam 1: Memory collection — who decides what to remember?

**Current discodon model:** The LLM decides. The reflect tool gives the entity explicit control: `reflect__note` to record observations, `reflect__new_wisdom` for insights. The entity is an active memory manager.

**3tears v2 model:** The extraction pipeline decides. It analyzes conversations, identifies memories, classifies them, assigns importance, resolves entities, and stores them. The LLM is consulted (worthiness gate, candidate extraction, resolution), but the *pipeline* drives the process. The `add_memory` tool exists for explicit storage, but the primary path is passive extraction.

**The tension:** Discodon's entities have a strong "I choose what to remember" model. 3tears v2's entities are more "the system remembers for me." These are philosophically different.

**Resolution options:**

1. **Pure 3tears extraction.** Remove `reflect__note` and `reflect__new_wisdom`. Let the extraction pipeline handle everything. Discodon entities lose explicit memory control.
   - *Pro:* Simpler. The entity focuses on acting; the system handles remembering.
   - *Con:* Loses the agent-driven memory model that MemGPT/Letta research validated. The entity can't decide "this is important, remember it" or "this observation is stale, forget it."

2. **Hybrid: extraction + explicit tools.** Keep `reflect__note` as a wrapper around `add_memory`. The extraction pipeline runs on conversations; the entity can also explicitly store memories via tool calls.
   - *Pro:* Best of both. Passive extraction catches what the entity misses; explicit storage handles what the entity considers important.
   - *Con:* Potential duplication. The extraction pipeline extracts "Alice likes jazz" and the entity also stores "Alice likes jazz" via reflect__note. 3tears dedup (0.90 similarity) should catch this, but it's an extra processing step.

3. **Extraction-primary, reflect-as-override.** The extraction pipeline is the default. The reflect tool becomes a way to add memories the pipeline missed, correct memories the pipeline got wrong, or explicitly bump importance on things the entity considers critical.
   - *Pro:* Clean separation. Pipeline for breadth, reflect for precision.
   - *Con:* Requires rethinking the reflect tool's semantics. It's no longer "my working memory" — it's "corrections and emphasis on the system's memory."

**Recommendation: Option 3.** The extraction pipeline handles the bulk. The reflect tool becomes a memory-correction/emphasis interface. `reflect__note` maps to `add_memory` with explicit importance. `reflect__new_wisdom` is removed — pattern synthesis is the distillation engine's job, not the entity's.

### Seam 2: Memory injection — how do memories reach the LLM prompt?

**Current discodon model:** `section_formatters.py` formats wisdom, notes, goals, todos, and turn history into prompt sections. Each section has its own scoring and formatting logic. The `format_wisdom()` and `format_notes()` functions score and select what to include.

**3tears v2 model:** The retriever returns entity-grouped, scored, typed results. The app formats these into prompts.

**What changes:**
- `format_wisdom()` and `format_notes()` merge into a single retrieval call. The app receives a `RetrievalResult` with entity groups and formats them.
- Scoring moves from the formatters to the retriever. `score_wisdom()` and `score_note()` are replaced by v2's multi-signal scoring.
- The two-tier note selection (relevant + recent backfill) is replaced by budget allocation (entity-specific 70% / general 30%).
- Topic hints for hidden notes (`extract_note_topics()`) are no longer needed — the retriever handles what to surface.

**The app still owns:**
- How to render the retrieval results into prompt text
- Whether to show entity grouping or flatten
- Token budgeting across prompt sections
- Goals, todos, and turn history formatting (these aren't memories in v2)

### Seam 3: The reflect tool — what stays, what changes?

| Reflect action | Migration | Notes |
|---|---|---|
| `reflect__note` | → `add_memory` (3tears tool) | Content becomes an observation-tier memory. `about` resolved via identity module. |
| `reflect__note` with `update_index` | → 3tears UPDATE resolution | Instead of replacing a note by index, the extraction pipeline's UPDATE creates a new memory and stamps `valid_until` on the old one. The entity can't address memories by index. |
| `reflect__note_delete` | → app-level soft delete or importance downgrade | 3tears doesn't expose "delete memory by index." The entity could mark a memory as superseded or low-importance. |
| `reflect__new_wisdom` | **Remove.** | Pattern synthesis is the distillation engine's job. The entity shouldn't manually create wisdom — that bypasses the evidence chain. |
| `reflect__goal_update` | **Stays in discodon.** | Goals aren't 3tears memories. |
| `reflect__new_todo` | **Stays in discodon.** | Todos aren't 3tears memories (yet). |
| `reflect__complete_todo` | **Stays in discodon.** | Same. |
| `reflect__pass` | **Stays in discodon.** | Cognitive architecture, not memory. |
| `reflect__tool_change` | **Stays in discodon.** | Tool management, not memory. |

**Key design decision:** The entity loses index-based memory addressing. v2 memories are addressed by ID (UUID), not by position in a list. This means the entity can't say "update note 3" — it would need to say "update the memory about Alice's jazz preference." This is a significant UX change for the LLM.

### Seam 4: The MCP entity tool — what changes?

| MCP action | Migration | Notes |
|---|---|---|
| `entity history` | **Stays.** Turn history is operational telemetry. | |
| `entity distill` | → 3tears API for forced distillation | Trigger API changes; non-destructive semantics. |
| `entity people` | → 3tears identity module query | No longer derived from note grouping. Query canonical entities with platform mappings. |
| `entity person_notes` | → 3tears retrieval with `relevant_entities=[person_canonical_id]` | Entity-filtered retrieval replaces note-field filtering. |
| `entity social_wisdom` | → 3tears retrieval with `type_filter=[belief]` or tier/type queries | Social wisdom filtering moves to type-aware retrieval. |
| `entity export_bundle` | **Partially stays.** Export includes both 3tears memories and discodon-specific state (goals, todos, traits). | Two export sources: 3tears for memories, discodon for cognitive state. |
| `entity import_bundle` | Same split. | |
| `entity update` (memory fields) | → 3tears API for memory CRUD | Direct memory field updates move to 3tears. Goals/todos/traits stay in discodon. |

### Seam 5: Observability — where does logging go?

**Current:** Discodon has a sophisticated OpenTelemetry-compatible logging system. `TurnLog` documents in Cosmos capture span trees, structured inputs (every prompt section with token counts), quality scores, and turn summaries.

**3tears:** `threetears.observe` provides structured logging and OpenTelemetry tracing with a zero-cost passthrough design. Libraries instrument; the host configures exporters.

**What changes:**
- Memory extraction, retrieval, and distillation operations get 3tears-observe instrumentation (traces, spans). These are new signals that don't exist today.
- Turn-level logging (TurnLog, span trees) stays in discodon. 3tears doesn't model turns.
- The prompt section capture (`CapturedSection`) stays in discodon but now includes 3tears retrieval results as a section source.
- Token accounting for memory operations (extraction: ~2,100 tokens, distillation: ~2,000 tokens) becomes visible through 3tears-observe. Today discodon tracks distillation cost in `DistillationUsage` — that moves to 3tears instrumentation.

**Integration pattern:** Discodon's turn-level trace creates a parent span. 3tears memory operations create child spans under it. The combined trace shows the full picture: turn → perception → retrieval → decision → action → extraction → distillation.

---

## Cutover plan

Hard cutover — no data migration, no dual-write. Entities start with empty 3tears memory and rebuild through normal operation. R&D phase, little data to lose.

### Infrastructure prerequisites
1. **Stand up PostgreSQL + pgvector.** New dependency.
2. **Deploy 3tears-identity.** Empty — entities and people are discovered at runtime as the extraction pipeline encounters them.
3. **Configure `EmbeddingProvider`.** ~20 lines in a config module.
4. **Configure `social` memory profile preset** as baseline (DJ pattern with entity observation, action memory, cross-entity patterns).

### Discodon code changes

**Remove:**
- `WorkingNote`, `NarrativeWisdom` from persona.py
- `score_wisdom()`, `score_note()`, `format_wisdom()`, `format_notes()`, `extract_note_topics()`, `extract_perception_keywords()` from section_formatters.py
- `distillation.py` (distill_notes, consolidate_wisdom, all models)
- `distillation_manager.py` (DistillationManager)
- `reflect__note`, `reflect__new_wisdom`, `reflect__note_delete` from reflect_tool.py
- `notes`, `wisdom`, distillation config fields from PersonaComponent
- `_note_should_replace()` dedup logic from behaviors.py
- Note/wisdom fields from Cosmos entity document schema
- `extract_social_wisdom()` from cognitive_style.py

**Add:**
- `EmbeddingProvider` implementation (config module)
- `DistillationStrategy` implementation (derived from current `DEFAULT_DISTILLATION_PROMPT`, adapted for v2 output format)
- 3tears `extract()` call after each turn in the turn pipeline (replaces reflect-based collection)
- 3tears `retrieve()` call in prompt construction (replaces keyword scoring)
- New prompt formatter for entity-grouped retrieval results
- Identity resolution in the perception pipeline: Discord user IDs → canonical UUIDs via `threetears.identity`
- Per-entity-role gate config (observers need relaxed response-length gates)

**Modify:**
- reflect_tool.py: keep `reflect__goal_update`, `reflect__new_todo`, `reflect__complete_todo`, `reflect__pass`, `reflect__tool_change`
- MCP entity tool: `people` → identity module query, `person_notes` → `retrieve(relevant_entities=[id])`, `social_wisdom` → `retrieve(type_filter=[belief])`, `distill` → 3tears forced distillation API
- Entity export/import bundles: memories come from 3tears, cognitive state from Cosmos
- Turn pipeline in behaviors.py: wire `extract()` where note processing used to happen
- Prompt construction: single `retrieve()` replaces per-section scoring

### Day-one experience

Entities start blank. First few sessions will feel like a new entity — no memory of prior listeners, no accumulated wisdom. Within a few active sessions + one distillation cycle, the entity has a working observation corpus. Within a week of normal operation, patterns emerge. Cross-entity principles take longer (daily background cycle, needs 3+ entities with overlapping patterns).

---

## Eval system intersections

The eval system has deep memory entanglement. Five areas need rework.

### 1. Context injection in the eval runner

**Current:** `runner.py` injects `context_notes` (as `ContextNote` objects with `about`, `about_source_id`, `age_label`) and `context_wisdom` (as plain strings) into eval turn prompts via `_format_context_notes()` and `_format_context_wisdom()`. The prompt section ordering mirrors production: wisdom → previous context → notes → perceptions.

**What changes:** The eval runner needs to inject 3tears retrieval results instead of hand-formatted notes/wisdom sections. Two options:

1. **Seed 3tears directly.** Before running the eval, call `add_memory()` to populate the entity's 3tears store with the test case's context memories. Then the eval turn uses the real `retrieve()` path. This tests the full pipeline — retrieval scoring, entity grouping, budget allocation — but makes eval results dependent on retrieval tuning.

2. **Synthetic retrieval results.** Build a `RetrievalResult` directly from the test case context and pass it to the prompt formatter. This isolates what's being tested (the entity's response to memories) from retrieval quality. Faster, deterministic, no database needed.

**Recommendation: Option 2 for unit-style evals, option 1 for integration evals.** The social cognition templates are testing the entity's *use* of memory, not retrieval quality. Synthetic injection keeps that focused. A separate retrieval-quality eval suite tests whether the right memories surface.

### 2. Accumulated state extraction

**Current:** `_extract_accumulated_state()` scans actions for `reflect__note` and `reflect__new_wisdom` tool calls, extracts content from parameters, and carries them forward into subsequent turns as `ContextNote` objects and wisdom strings.

**What changes:** With the reflect tool losing note/wisdom actions:

- If **extraction-primary** (recommended): the eval runner can't observe what the extraction pipeline would store, because extraction is a background process analyzing conversation content — not a tool call the entity makes. The runner would need to either:
  - Call `extract()` on the turn's conversation content and use the result as accumulated state
  - Skip accumulated state entirely and pre-seed all context per-turn in the test case

- If **reflect-as-correction survives** as `add_memory`: the runner scans for `add_memory` tool calls instead of `reflect__note`. The `ContextNote` model changes — `about`/`about_source_id` become `about_ids` (canonical UUIDs), `age_label` might become recency metadata.

**Recommendation:** For multi-turn evals, pre-seed each turn's context explicitly. Relying on extraction mid-eval adds LLM cost and non-determinism. The test case author controls exactly what memories are available at each turn.

### 3. ContextNote model

**Current:** `ContextNote` in `eval/models.py` has `content`, `age_label`, `about` (display name), `about_source_id` (platform-qualified ID).

**What changes:** The model needs to align with v2 memory fields:

| Current ContextNote field | v2 equivalent | Notes |
|---|---|---|
| `content` | `content` | Direct |
| `age_label` | Derivable from `date_created` + retrieval recency | Or keep as eval-specific synthetic metadata |
| `about` | `about_ids` (canonical UUIDs) + identity module display names | The eval can use synthetic UUIDs |
| `about_source_id` | Identity module platform mapping | Not needed if using canonical UUIDs |
| *(none)* | `type_memory` | preference, fact, action, belief |
| *(none)* | `importance` | 1-10 |
| *(none)* | `tier` | observation, pattern |
| *(none)* | `epistemic_type` | observed, stated, inferred, etc. |

The richer model lets eval test cases exercise v2 features: "Does the entity treat a low-confidence belief differently from a high-confidence fact?" "Does the entity reference supporting evidence for a decision?"

### 4. Social cognition templates

The five social cognition templates (`social_cognition.py`) are the most memory-dependent eval scenarios:

| Template | Memory dependency | What changes |
|---|---|---|
| `social_awareness` (Level 1) | No pre-seeded memory | **No change.** Tests baseline awareness without prior knowledge. |
| `preference_learning` (Level 2) | Pre-seeded person-tagged notes | Context notes become v2 observations with `about_ids`, `type=preference`. The `person_model_consultation` criterion still measures whether the entity uses them. |
| `surprise_and_update` (Level 2) | Pre-seeded prediction/assumption note | The assumption becomes a `type=belief, confidence=0.6` observation. The `surprise_recognition` criterion now also tests whether the entity notices the `belief` epistemic type and treats it as revisable. |
| `personality_differentiation` | Optional person-tagged notes | Same as preference_learning. |
| `longitudinal_learning` | Multi-turn accumulation | **Most affected.** Currently relies on `_extract_accumulated_state()` scanning reflect__note calls. Needs rework per section 2 above. |
| `group_dynamics` | Multi-entity shared + private memories | Naturally maps to v2's entity-grouped retrieval. Pre-seed observations with different `about_ids` values. The eval can now test whether entity-grouped output helps the entity reason about relationships. |

**New eval opportunities with v2:**
- **Distillation quality:** Pre-seed observations, run distillation, check pattern quality. Not possible today because discodon's distillation is hard to isolate.
- **Temporal validity:** Pre-seed a superseded fact (`valid_until` set) and a current fact. Does the entity use the current one?
- **Decision evidence:** Pre-seed a decision with `supporting_evidence_ids`. Does the entity reference the evidence when the decision is challenged?
- **Cross-entity patterns:** Pre-seed an `about=null` principle derived from multiple entities. Does the entity apply it to a new entity?

### 5. Quality scoring dimensions

**Current production quality dimensions** (in `quality/models.py`):

| Dimension | Memory relevance | What changes |
|---|---|---|
| `response_present` | None | No change |
| `tool_selection_accuracy` | None | No change |
| `action_efficiency` | None | No change |
| `attention_efficiency` | None | No change |
| `cost_per_action` | Indirect (extraction adds cost) | Extraction/retrieval token costs are now visible |
| `error_recovery` | None | No change |
| `persona_adherence` (LLM) | Indirect (wisdom shapes persona) | v2 patterns and principles contribute to persona consistency |
| `response_relevance` (LLM) | Indirect (memory shapes relevance) | v2 retrieval quality directly affects this score |
| `medium_appropriateness` (LLM) | None | No change |

**Missing dimensions that v2 enables:**
- `memory_consultation` — did the entity reference retrieved memories? (Generalizes `person_model_consultation` from social cognition evals to all turns.)
- `memory_accuracy` — did the entity correctly use the memories it consulted, or hallucinate/misapply them?
- `belief_handling` — when the entity encountered a `type=belief` memory, did it treat it with appropriate uncertainty?

These are optional — the current dimensions still work. But v2's richer memory model makes memory-specific quality scoring feasible for production turns, not just eval scenarios.

### 6. Structured inputs capture

**Current:** `CapturedSection` in `logging/models.py` records each prompt section with name, position, content, and token count. Sections include "Your Wisdom" and "Your Working Notes".

**What changes:** The section names change. Instead of separate wisdom and notes sections, there's a single memory section populated from 3tears retrieval results. The `CapturedSection` model itself doesn't change — it's already self-describing. But downstream analysis that looks for specific section names ("Your Wisdom", "Your Working Notes") needs to look for the new section name(s).

The `structured_inputs.items` field (optional structured items within a section) becomes more useful — each retrieved memory can be an item with its type, tier, importance, entity association, etc. This makes it possible to analyze which memories were surfaced, at what importance level, and how they correlate with quality scores — without parsing free text.

---

## Risk analysis

### High risk: Collection philosophy change
The shift from "entity manages its own memory" to "extraction pipeline manages memory" changes the agent's relationship with its memories. The entity might produce worse memories if it can't explicitly curate them, or better memories because the pipeline catches things the entity would miss. Worth monitoring extraction quality closely in the first week.

### Medium risk: Retrieval quality
Multi-signal scoring with embeddings is more powerful but less predictable than keyword overlap. The `social` preset should be a reasonable starting point, but expect a tuning period.

### Medium risk: Corpus growth
Without the 50-note FIFO cap, observation count grows until distillation consumes them. Importance-based scoring should handle this, but if thresholds aren't well-calibrated, retrieval could get noisy.

### Low risk: Cold start
Entities lose all memory. For R&D this is fine. In production this would be unacceptable — but we're not there yet.

---

## Open questions

1. **Should reflect__note survive as an `add_memory` wrapper?** Or remove explicit memory creation entirely and let the extraction pipeline handle everything?

2. **How does the attention budget interact with extraction?** Is extraction a background process, or does it consume the entity's cognitive resources?

3. **Cross-entity distillation scope:** Should it run across all entities globally, or scoped per-universe (discodon's multi-tenant isolation)?

4. **MCP tool location:** Should memory-related MCP actions move to a 3tears MCP server, or stay in discodon as wrappers?
