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
- **No deduplication by similarity.** Note dedup uses normalized word overlap heuristics, not cosine distance. Person-aware: notes about different people never replace each other even with high word overlap (20% threshold for same-person, 50% for general notes).
- **Entity association is per-note, and more sophisticated than it appears.** `WorkingNote.about` (display name) and `about_source_id` (platform-qualified like `discord:123456`) tag person-specific notes. The system auto-resolves identity each turn: `behaviors.py` builds `_author_source_ids` from perception metadata, mapping lowercased display names (and Mastodon @handles) to platform IDs. When the LLM tags a note `about="Alice"`, the system stamps `about_source_id="discord:123456789"`. Retrieval in `section_formatters.py:compute_social_context()` matches on source_id first, falls back to display name, and presents notes under the person's **current** display name. This handles name changes transparently. Wisdom has no entity tagging.
- **Distillation is notes → wisdom only.** No tiers beyond that. No cross-entity synthesis. Post-distillation consolidation pass dedup/merges overlapping wisdom entries when `len(wisdom) >= 10`.
- **The LLM manages its own memory.** The reflect tool lets the entity create/update/delete notes and wisdom via tool calls during turns. 3tears extraction is a passive pipeline; discodon's is agent-driven.
- **Cost tracking is comprehensive.** Per-call token/cost extraction from OpenRouter responses (`LLMUsage`), per-turn aggregation in `TurnRecord` (with per-round debug data including cache tokens), session-level rollup in `SessionTokenUsage` (per-model and per-tool breakdowns), and a three-tier budget hierarchy (entity → universe → system) with four states (healthy/cautious/tight/exhausted) mapping to enforcement tiers (none/warn/throttle/hardstop). Budget periods have category allocations (execution 60%, monitoring 15%, optimization 20%, reserve 5%). Cost events persist to Cosmos DB. Distillation and classifier costs tracked separately in `DistillationUsage` and `ClassifierUsage`.

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
| `about` | `about_names` (Phase 1) → `about_ids` (Phase 2) | Phase 1: display name stored directly on memory row. Phase 2: identity module resolves to canonical UUID. |
| `about_source_id` | `about_platform_ids` (Phase 1) → identity module (Phase 2) | Phase 1: `discord:123456` stored directly. Phase 2: identity module maps to canonical ID. |
| *(implied from content)* | `about_ids` | NULL in Phase 1. Populated by identity module in Phase 2. |
| *(implied from content)* | `mentioned_entity_ids` / `mentioned_names` / `mentioned_platform_ids` | Secondary entities mentioned in the note. Same phasing as about_ids. |
| *(none)* | `importance` | Assigned at extraction (1-10) |
| *(none)* | `tier` | `observation` |
| *(none)* | `type_memory` | `preference`, `fact`, `action`, `belief` — determined by extraction |
| *(none)* | `source_type` | `discord_message`, `reflection`, etc. |
| *(none)* | `source_id` | Message ID or turn ID |

**What changes:**
- Notes stop being capacity-bounded FIFO. 3tears observations are permanent until consumed by distillation. The "50 notes max, oldest dropped" model is replaced by distillation's CONSUME operation.
- Entity association switches from per-note fields (`about`/`about_source_id`) to memory-row arrays (`about_names`/`about_platform_ids`). In Phase 1 this is a location change, not a philosophy change — the LLM still writes display names, the system still auto-resolves to platform IDs from perception metadata, and retrieval still matches on platform IDs. The `_author_source_ids` dict becomes the `about_context` parameter to `extract()`. In Phase 2, the identity module resolves platform IDs to canonical UUIDs in `about_ids`.
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

### 4. Entity identity → 3tears entity references (Phase 1) → 3tears-identity module (Phase 2)

**Current:** Discodon has an effective but informal identity system — more sophisticated than "just display names." The LLM writes display names into `WorkingNote.about`. The system auto-resolves them: `behaviors.py` builds `_author_source_ids` from perception metadata each turn (lowercased display names and Mastodon @handles → platform-qualified IDs). When the LLM creates a note with `about="Alice"`, the system stamps `about_source_id="discord:123456789"`. Retrieval in `compute_social_context()` matches on source_id first (stable across name changes), falls back to display-name matching, and presents notes under the person's **current** display name. The MCP `entity people` action groups notes by `about` field. `entity person_notes` accepts either display name or source_id.

**What's NOT handled today:** Cross-platform identity (same person on Discord and Mastodon = two separate note sets), entity merge/split, and any identity resolution beyond the current turn's perception authors (if a person isn't in the current perceptions, their name can't be auto-resolved to a source_id).

**Phased v2 mapping:**

| Discodon identity concept | Phase 1 (no identity module) | Phase 2 (identity module) |
|---|---|---|
| `entity.id_entity` (UUID) | `owner_id` (same UUID) | `owner_id` (same UUID) |
| `WorkingNote.about` (display name) | `about_names` on memory row | Identity module: display name → canonical UUID |
| `WorkingNote.about_source_id` (`discord:123456`) | `about_platform_ids` on memory row | Identity module: `(platform, platform_id) → canonical_id` |
| `_author_source_ids` dict (name → platform ID) | `about_context` dict passed to extraction/add_memory | Identity module resolves + `about_context` |
| *(none — notes grouped by display name)* | Notes grouped by `about_platform_ids` | Identity module: canonical entity with platform mappings |
| *(none)* | *(none — still per-platform)* | Cross-platform identity resolution, merge/split |

**Phase 1 (migration target):**
- `about` → `about_names` (text array on memory row). The LLM still writes human-readable names.
- `about_source_id` → `about_platform_ids` (text array on memory row). The system still auto-resolves from perception metadata.
- The auto-resolution pattern (`_author_source_ids` dict) maps directly to 3tears extraction's `about_context` parameter. Discodon builds the dict from perception metadata each turn, same as today, and passes it to `extract()` and `add_memory()`.
- Retrieval boosts by `about_platform_ids` match — same as the current `compute_social_context()` source_id matching, but in the retriever instead of section_formatters.
- `entity people` MCP action queries 3tears memories grouped by `about_platform_ids` instead of note `about` field. Functionally identical.
- **No identity module required.** No new infrastructure beyond the 3tears memory schema changes.

**Phase 2 (post-migration enhancement):**
- The identity module maps `about_platform_ids` → canonical UUIDs and populates `about_ids`.
- Cross-platform identity becomes possible: `discord:123456` and `mastodon:@alice@social.example` merge into one canonical ID.
- `entity people` becomes a rich query on canonical entities with platform mappings, types, and metadata.
- A backfill job resolves existing memories.

**Migration complexity: LOW for Phase 1** (the patterns already exist in discodon — it's a field rename and location change). **MEDIUM for Phase 2** (new identity module, backfill, cross-platform resolution). Phase 1 is the migration target. Phase 2 is a post-migration enhancement.

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

**Prompt compositor pattern:** `section_formatters.py` doesn't go away — it loses two sections (wisdom, notes) and gains one (3tears retrieval). The integration seam:

```python
# In discodon's turn prompt builder (replacing format_wisdom + format_notes):
memory_result = await retriever.retrieve(
    pool=pool,
    owner_id=entity_id,
    query=build_memory_query(perceptions),
    relevant_entities=extract_author_platform_ids(perceptions),  # Phase 1: platform IDs
    mode="conversational",
    ledger=memory_ledger,
    customer_id=None,
)
memory_section = memory_result.context  # Pre-formatted by 3tears

# Existing sections remain:
sections = [
    format_traits(persona),
    format_goals(persona),
    format_todos(persona),
    format_directives(persona),
    memory_section,  # Replaces format_wisdom() + format_notes()
]
turn_prompt = "\n\n".join(s for s in sections if s)
```

**Attention budget interaction:** Discodon's attention budget allocates a token budget per prompt section. The retriever's `context_budget` (max items) and the retriever's formatted output must fit within the attention budget's memory allocation. The retriever accepts `max_items` but not a token limit directly — the app truncates the `context` string if it exceeds the section's token allocation. In practice, 15 memories at ~30 tokens each = ~450 tokens, well within typical section budgets.

### Seam 3: The reflect tool — what stays, what changes?

| Reflect action | Migration | Notes |
|---|---|---|
| `reflect__note` | → **Keep as thin wrapper** over 3tears `add_memory` | Same tool name preserves eval compatibility (see eval section). Internally calls `add_memory` with `about` resolved to `about_platform_ids` via `about_context`. The v2 `add_memory` tool accepts `about` (display name), `type_memory`, and `importance` — enrichments over v0.5's content-only interface. |
| `reflect__note` with `update_index` | → 3tears UPDATE resolution | Instead of replacing a note by index, the extraction pipeline's UPDATE creates a new memory and stamps `valid_until` on the old one. The entity can't address memories by index. The wrapper translates `update_index` to a semantic lookup ("find the memory most similar to this note"). |
| `reflect__note_delete` | → app-level soft delete or importance downgrade | 3tears doesn't expose "delete memory by index." The entity could mark a memory as superseded or low-importance. |
| `reflect__new_wisdom` | **Remove.** | Pattern synthesis is the distillation engine's job. The entity shouldn't manually create wisdom — that bypasses the evidence chain. |
| `reflect__goal_update` | **Stays in discodon.** | Goals aren't 3tears memories. |
| `reflect__new_todo` | **Stays in discodon.** | Todos aren't 3tears memories (yet). |
| `reflect__complete_todo` | **Stays in discodon.** | Same. |
| `reflect__pass` | **Stays in discodon.** | Cognitive architecture, not memory. |
| `reflect__tool_change` | **Stays in discodon.** | Tool management, not memory. |

**Key design decision:** The entity loses index-based memory addressing. v2 memories are addressed by ID (UUID), not by position in a list. This means the entity can't say "update note 3" — it would need to say "update the memory about Alice's jazz preference." This is a significant UX change for the LLM.

**Why keep `reflect__note` as a wrapper:** The eval system's `_extract_accumulated_state()` scans for `target_tool == "reflect"` and `action == "note"`. Keeping the tool name unchanged means the eval runner, social cognition templates, test cases, and accumulated-state extraction all work without modification. The wrapper is trivial (~15 lines) and can be retired after the eval system is updated to scan for `add_memory` instead.

### Seam 4: The MCP entity tool — what changes?

| MCP action | Migration | Notes |
|---|---|---|
| `entity history` | **Stays.** Turn history is operational telemetry. | |
| `entity distill` | → 3tears API for forced distillation | Trigger API changes; non-destructive semantics. |
| `entity people` | → 3tears memory query grouped by `about_platform_ids` (Phase 1) / identity module query (Phase 2) | Phase 1: same concept as today's note-grouping, but queries `about_platform_ids` on memory rows. Phase 2: query canonical entities with platform mappings. |
| `entity person_notes` | → 3tears retrieval with `relevant_entities=[platform_id]` (Phase 1) / `[canonical_id]` (Phase 2) | Entity-filtered retrieval replaces note-field filtering. |
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

### Seam 6: Cost tracking and budget enforcement

**Current discodon model:** Comprehensive three-tier budget system.

| Layer | How it works |
|---|---|
| Per-call | `extract_usage()` pulls tokens/cost from OpenRouter responses into `LLMUsage` |
| Per-turn | `TurnRecord` accumulates across multi-round tool calls; `UsageTracker` rolls into `SessionTokenUsage` with per-model and per-tool breakdowns |
| Budget enforcement | `BudgetEnforcer` checks hierarchy (entity → universe → system) before each turn. Four states (healthy/cautious/tight/exhausted) map to four tiers (none/warn/throttle/hardstop) |
| Persistence | `CostEvent` and `BudgetPeriod` documents in Cosmos DB with cross-process cache invalidation via ZMQ |
| Admin | MCP budget tool (8 actions: check, hierarchy, allocate, history, projection, summary, reset), REST API, category-based allocation |

Cost sources tracked today: `entity_turn`, `classifier`, `distillation`, `optimization`, `monitoring`.

**What 3tears adds:** New LLM and embedding calls that discodon's budget system needs to track:

| 3tears operation | Token cost | When it fires |
|---|---|---|
| Extraction: worthiness gate | ~350 tokens | Most turns (after heuristic gates pass) |
| Extraction: candidate extraction | ~1,000 tokens | When worthiness gate passes |
| Extraction: resolution | ~750 tokens | When similar memories exist |
| Distillation: within-entity | ~2,000 tokens | Per importance-sum threshold or session boundary |
| Distillation: cross-entity | ~12,000 tokens | Daily background cycle |
| Distillation: consolidation | ~1,500 tokens | When pattern count exceeds threshold post-distillation |
| Embedding | ~50 tokens per call | Every extraction (per candidate) + every retrieval query |

**3tears reports costs; discodon enforces budgets.** 3tears provides a `CostCallback` protocol — a function that fires after each LLM or embedding call with a `CostReport` (operation name, input/output/embed tokens, model, estimated cost). Discodon wires this to `BudgetEnforcer.record_cost()`.

**Integration:**

```python
# In discodon's 3tears initialization:
async def _memory_cost_callback(report: CostReport) -> None:
    budget_enforcer.record_cost(
        entity_id=current_entity_id,
        universe_id=current_universe_id,
        cost_usd=report.estimated_cost_usd or 0.0,
        input_tokens=report.input_tokens,
        output_tokens=report.output_tokens,
        model=report.model,
        turn_number=current_turn_number,
        source=f"memory_{report.operation}",   # e.g., "memory_extraction_candidates"
        category="execution",
    )

extractor = MemoryExtractor(pool, embedding, chat_model, config, cost_callback=_memory_cost_callback)
retriever = MemoryRetriever(pool, embedding, config, cost_callback=_memory_cost_callback)
```

**New `CostEventSource` values:** Add `memory_extraction`, `memory_distillation`, `memory_embedding` to the enum. These appear in budget history, MCP budget tool queries, and cost projections.

**Known bugs to fix during integration:**
- ARIA optimizer uses `source="aria_optimization"` as a plain string instead of `CostEventSource` enum — costs silently drop out of the budget hierarchy.
- Classifier and distillation costs bump `process_cost_usd` but are NOT recorded to `BudgetService` — they're invisible to budget enforcement. Fix: wire classifier and distillation through `record_cost()` like turn costs.
- `CostTrackingCallback.summary()` is dead code (never called). Remove it.

**Budget category for memory operations:** Memory extraction and distillation are `execution` category (same as entity turns). Embedding calls are negligible cost ($0.02/M tokens) but should still be tracked for completeness.

**Budget enforcement timing:** Extraction runs after the turn completes (fire-and-forget). Its cost is recorded post-turn and affects the *next* turn's budget check. Retrieval runs before the decision and its embedding cost is negligible. This means memory costs don't block the current turn — they count against the next one. This is acceptable because extraction costs are modest (~2,100 tokens max) relative to turn costs (~5,000-15,000 tokens).

---

## Cutover plan

Hard cutover — no data migration, no dual-write. Entities start with empty 3tears memory and rebuild through normal operation. R&D phase, little data to lose.

### What 3tears is

3tears is a **Python library suite**, not a service. You `pip install 3tears-agent-memory` into discodon. No separate container, no sidecar, no API gateway. The library runs in-process inside discodon's existing container.

But it brings infrastructure dependencies that discodon doesn't have today.

### What discodon has now

| Component | Technology | Where |
|---|---|---|
| Database | Azure Cosmos DB (NoSQL API) | Azure |
| IPC | ZeroMQ (REQ/REP, PUB/SUB) | In-container |
| LLM | OpenRouter (via LangChain) | API |
| Deployment | Fly.io (single Docker container) | Cloud |
| Embeddings | **None** | — |
| PostgreSQL | **None** | — |
| NATS | **None** | — |

### What 3tears agent-memory requires

| Dependency | What it does | Required? |
|---|---|---|
| **PostgreSQL + pgvector** | Memory storage, vector similarity search, full-text search. L3 (source of truth) in the three-tier cache. | Yes — core storage engine |
| **NATS server** | L2 distributed cache (cross-pod KV store). Also used by 3tears-registry for tool discovery. | Yes for L2 cache, but **can skip in single-pod deployments** — L1 (SQLite) + L3 (PostgreSQL) still work. L2 matters when multiple pods need cache consistency. |
| **Embedding API** | App implements `EmbeddingProvider` protocol. Any provider works (VoyageAI, OpenAI, local model). ~20 lines of code. | Yes — semantic search requires embeddings |
| **LLM for extraction/distillation** | App implements `ChatModelFactory` protocol. Discodon already has OpenRouter — wrap it. | Yes — but discodon already has this |
| **asyncpg connection pool** | Passed to 3tears at initialization. Pool lifecycle owned by the app. | Yes — new dependency, standard Python lib |

### Infrastructure plan

#### 1. PostgreSQL + pgvector

**Option A: Managed service (recommended for Fly.io).** Fly Postgres or Supabase. Both support pgvector. Fly Postgres runs in the same region as your app VM — low latency. Supabase is managed with backups/monitoring out of the box.

**Option B: Sidecar container.** Run PostgreSQL in a second Fly machine or docker-compose service. You manage backups and upgrades.

**Option C: Azure Database for PostgreSQL.** Keeps everything in Azure alongside Cosmos DB. Higher latency from Fly.io unless you also move the app to Azure.

Whichever option: enable the `vector` extension (`CREATE EXTENSION vector`), create the 3tears schema, and provide a connection string to discodon at startup.

**Sizing:** Memory corpus will be small (R&D). A minimal Postgres instance (1 vCPU, 256MB) is fine to start. pgvector indexes don't matter until you have >10K memories per owner.

#### 2. NATS (optional for now)

Discodon runs as a single container on Fly.io. No multi-pod scaling. L2 cache provides cross-pod consistency — which you don't need with one pod.

**Skip NATS initially.** Run with L1 (SQLite, in-process) + L3 (PostgreSQL). If you later scale to multiple pods, add NATS. The 3tears cache layer degrades gracefully — L2 miss just falls through to L3.

This means the `3tears-registry` package (which requires NATS for tool discovery) isn't usable yet either, but discodon has its own tool routing via ZeroMQ.

#### 3. Embedding provider: VoyageAI voyage-4-lite

Implement `EmbeddingProvider` — a protocol with two methods:

```python
class EmbeddingProvider(Protocol):
    async def embed_text(self, text: str) -> tuple[list[float] | None, int]: ...

    @property
    def dimensions(self) -> int: ...
```

**Model choice: `voyage-4-lite`** at 1024 dimensions (default).

The Voyage 4 family (Jan 2026) shares an embedding space across all models — embeddings from different models are interchangeable. This means you can upgrade the document embedding model later without re-embedding existing memories.

| Model | Price/1M tokens | Quality (RTEB NDCG@10) | Notes |
|---|---|---|---|
| voyage-4-large | $0.12 | Best (MoE architecture) | 14% better than OpenAI v3 Large |
| voyage-4 | $0.06 | Strong | Good balance |
| **voyage-4-lite** | **$0.02** | Good | **Our choice** |
| voyage-4-nano | Free (open weight) | Baseline | Self-hosted, Apache 2.0 |

**Why voyage-4-lite:**

- **R&D corpus is tiny.** Hundreds of memories, not millions. Quality differences between lite and large are negligible at this scale — they matter when retrieval must distinguish among thousands of similar memories.
- **200M free tokens per account.** A memory is ~50 tokens. That's ~4 million memories embedded before we pay anything. We won't hit the paid tier for a long time.
- **Upgrade path requires zero re-embedding.** When quality matters, switch new document embeddings to voyage-4 or voyage-4-large. Queries still use voyage-4-lite. Old lite-embedded documents remain searchable — shared embedding space. This is called asymmetric retrieval.
- **voyage-4-nano (self-hosted) isn't worth it.** It runs on CPU — 100-500ms per embedding vs ~20ms for the API. Extraction isn't blocking the response, so latency is tolerable, but it competes for CPU on Fly.io's shared-cpu-2x with the LLM client and ZMQ mesh.

**Configuration:**

```
model:       voyage-4-lite
dimensions:  1024 (default)
input_type:  "document" at extraction time, "query" at retrieval time
output_type: float (default — storage isn't a concern at our scale)
```

Always pass `input_type`. It tells the model to optimize the embedding for its role — queries get a "find relevant documents" bias, documents get a "be findable" bias. Omitting it works but leaves retrieval quality on the table.

**Secret:** Add `DD_VOYAGEAI_API_KEY` to discodon's Cosmos-backed secret store (same pattern as `DD_OPENROUTER_API_KEY`).

The implementation is a thin `httpx` wrapper around the Voyage API (~20 lines).

#### 4. ChatModelFactory for extraction/distillation

3tears needs an LLM for the extraction pipeline (worthiness gate, candidate extraction, resolution) and distillation. Discodon already has `OpenRouterClient` with LangChain integration. Wrap it:

```python
class ChatModelFactory(Protocol):
    def create_chat_model(self, purpose: str) -> BaseChatModel: ...
```

The `purpose` parameter lets you route different 3tears operations to different models: `"extraction"` → cheap/fast (Haiku), `"distillation"` → better (Sonnet). Discodon already supports per-entity model overrides — same pattern.

#### 5. asyncpg pool

Add `asyncpg` to discodon's dependencies. Create a connection pool at startup (in `main.py` after Cosmos secrets load), pass it to 3tears initialization. Pool lifecycle follows discodon's process supervisor — create on start, close on drain.

```python
pool = await asyncpg.create_pool(dsn=config.threetears.postgres_dsn)
```

Add `DD_THREETEARS_POSTGRES_DSN` to the config system (new config section, 4 lines of Pydantic).

#### 6. 3tears initialization in discodon

Wire it together at entity coordinator startup:

```python
from threetears.agent.memory import MemoryExtractor, MemoryRetriever, MemoryConfig

# Pool created earlier in startup
memory_config = MemoryConfig(preset="social", ...)
extractor = MemoryExtractor(pool, embedding_provider, chat_model_factory, memory_config)
retriever = MemoryRetriever(pool, embedding_provider, memory_config)
```

These are passed to `PersonaComponent` (or a new memory adapter) and called during the turn pipeline.

#### 7. Schema initialization

3tears has a `MigrationRunner` that creates and migrates tables. Run it once at startup (idempotent):

```python
from threetears.core.data.migrations import MigrationRunner
runner = MigrationRunner(pool)
await runner.run()
```

Tables are created in an agent-namespaced schema (`agent_{agent_id.hex}`). pgvector indexes are created automatically.

### Updated Dockerfile changes

```dockerfile
# Add to existing Dockerfile
RUN pip install 3tears-agent-memory asyncpg
# No new services — everything runs in-process
# 3tears-identity is NOT required for Phase 1 (platform ID-based entity references)
```

No new containers. No new processes. The Docker image gets slightly larger (asyncpg + pgvector Python bindings). The only new external dependency is the PostgreSQL server.

### Updated fly.toml changes

If using Fly Postgres:

```toml
# Existing fly.toml — add the Postgres connection string as a secret
# fly secrets set DD_THREETEARS_POSTGRES_DSN="postgres://user:pass@db.internal:5432/discodon"
```

If using an external embedding API:

```toml
# fly secrets set DD_EMBEDDING_API_KEY="voy-..."
```

### Operational summary

| Before | After |
|---|---|
| 1 service: Cosmos DB | 2 services: Cosmos DB + PostgreSQL |
| 0 embedding calls | Embedding calls on every extraction + every retrieval query |
| Distillation: in-process LLM call | Distillation: in-process LLM call (same, but non-destructive) |
| Memory in Cosmos entity doc | Memory in PostgreSQL, cognitive state in Cosmos entity doc |
| No vector search | pgvector semantic search |
| ~$0/month infra beyond Cosmos | +$0-7/month Fly Postgres + $0/month embeddings (200M free tokens covers ~4M memories) |

### Discodon code changes

**Remove:**
- `WorkingNote`, `NarrativeWisdom` from persona.py
- `score_wisdom()`, `score_note()`, `format_wisdom()`, `format_notes()`, `extract_note_topics()`, `extract_perception_keywords()`, `compute_social_context()`, `format_social_context()` from section_formatters.py
- `distillation.py` (distill_notes, consolidate_wisdom, all models)
- `distillation_manager.py` (DistillationManager)
- `reflect__new_wisdom` from reflect_tool.py
- `notes`, `wisdom`, distillation config fields from PersonaComponent
- `_note_should_replace()` dedup logic from behaviors.py
- Note/wisdom fields from Cosmos entity document schema
- `extract_social_wisdom()` from cognitive_style.py

**Add:**
- `EmbeddingProvider` implementation (config module, ~20 lines wrapping VoyageAI)
- `DistillationStrategy` implementation (derived from current `DEFAULT_DISTILLATION_PROMPT`, adapted for v2 output format)
- 3tears `extract()` call after each turn in the turn pipeline (replaces reflect-based collection)
- 3tears `retrieve()` call in prompt construction (replaces keyword scoring)
- Memory section in turn prompt (replaces `format_wisdom()` + `format_notes()` — uses `retrieval_result.context`)
- `about_context` dict construction from perception metadata (same pattern as `_author_source_ids`, passed to `extract()` and `add_memory()`)
- Per-entity-role gate config (observers need relaxed response-length gates)
- `CostCallback` wiring: memory cost reports → `BudgetEnforcer.record_cost()` with new source values
- New `CostEventSource` enum values: `memory_extraction`, `memory_distillation`, `memory_embedding`

**Modify:**
- reflect_tool.py: `reflect__note` becomes thin wrapper over 3tears `add_memory` (preserves eval compatibility). `reflect__note_delete` becomes importance downgrade or app-level soft delete. Keep `reflect__goal_update`, `reflect__new_todo`, `reflect__complete_todo`, `reflect__pass`, `reflect__tool_change`.
- MCP entity tool: `people` → 3tears memory query grouped by `about_platform_ids` (Phase 1), `person_notes` → `retrieve(relevant_entities=[platform_id])`, `social_wisdom` → `retrieve(type_filter=[belief])`, `distill` → 3tears forced distillation API
- Entity export/import bundles: memories come from 3tears, cognitive state from Cosmos
- Turn pipeline in behaviors.py: wire `extract()` where note processing used to happen, pass `about_context` built from perception metadata
- Prompt construction: single `retrieve()` replaces per-section scoring
- Budget enforcement: fix existing bugs (classifier/distillation costs not recorded to BudgetService, ARIA source string mismatch), add memory cost sources

### Day-one experience

Entities start blank. First few sessions will feel like a new entity — no memory of prior listeners, no accumulated wisdom. Within a few active sessions + one distillation cycle, the entity has a working observation corpus. Within a week of normal operation, patterns emerge. Cross-entity principles take longer (daily background cycle, needs 3+ entities with overlapping patterns).

---

## Eval system intersections

The eval system has deep memory entanglement. Six areas need rework. The key mitigation: **keeping `reflect__note` as a wrapper** means most eval code works unchanged during migration. The wrapper emits the same tool name and action, so `_extract_accumulated_state()` continues to work, social cognition templates continue to work, and accumulated state flows across turns as before. The changes below describe what's needed for full v2 eval support after the migration stabilizes.

### 1. Context injection in the eval runner

**Current:** `runner.py` injects `context_notes` (as `ContextNote` objects with `about`, `about_source_id`, `age_label`) and `context_wisdom` (as plain strings) into eval turn prompts via `_format_context_notes()` and `_format_context_wisdom()`. The prompt section ordering mirrors production: wisdom → previous context → notes → perceptions.

**What changes:** The eval runner needs to inject 3tears retrieval results instead of hand-formatted notes/wisdom sections. Two options:

1. **Seed 3tears directly.** Before running the eval, call `add_memory()` to populate the entity's 3tears store with the test case's context memories. Then the eval turn uses the real `retrieve()` path. This tests the full pipeline — retrieval scoring, entity grouping, budget allocation — but makes eval results dependent on retrieval tuning.

2. **Synthetic retrieval results.** Build a `RetrievalResult` directly from the test case context and pass it to the prompt formatter. This isolates what's being tested (the entity's response to memories) from retrieval quality. Faster, deterministic, no database needed.

**Recommendation: Option 2 for unit-style evals, option 1 for integration evals.** The social cognition templates are testing the entity's *use* of memory, not retrieval quality. Synthetic injection keeps that focused. A separate retrieval-quality eval suite tests whether the right memories surface.

### 2. Accumulated state extraction

**Current:** `_extract_accumulated_state()` scans actions for `reflect__note` and `reflect__new_wisdom` tool calls, extracts content from parameters, and carries them forward into subsequent turns as `ContextNote` objects and wisdom strings.

**What changes during migration: nothing.** Because `reflect__note` survives as a wrapper, `_extract_accumulated_state()` still finds `target_tool == "reflect"` and `action == "note"` in the action list. Accumulated state flows across turns exactly as before. `reflect__new_wisdom` is removed, so wisdom accumulation stops — but this is correct, as pattern synthesis moves to the distillation engine.

**What changes for full v2 eval support (post-migration):** Once the `reflect__note` wrapper is retired in favor of native `add_memory`:

- The runner scans for `add_memory` tool calls instead of `reflect__note`. The `ContextNote` model gains `about_names`/`about_platform_ids` alongside (not replacing) `about`/`about_source_id`.
- For multi-turn evals testing the extraction pipeline itself, the runner can call `extract()` on the turn's conversation content and use the result as accumulated state. This is an integration eval, not a unit eval.
- For unit-style social cognition evals, continue pre-seeding each turn's context explicitly. Relying on extraction mid-eval adds LLM cost and non-determinism.

### 3. ContextNote model

**Current:** `ContextNote` in `eval/models.py` has `content`, `age_label`, `about` (display name), `about_source_id` (platform-qualified ID).

**What changes during migration: nothing.** `ContextNote` keeps `about` and `about_source_id` — these map directly to `about_names` and `about_platform_ids` in v2 Phase 1. The fields are semantically identical.

**What changes for full v2 eval support (Phase 3):** The model gains optional v2 fields:

| Current ContextNote field | v2 addition | Notes |
|---|---|---|
| `content` | — | Unchanged |
| `age_label` | — | Keep as eval-specific synthetic metadata |
| `about` | — | Unchanged. Maps to `about_names`. |
| `about_source_id` | — | Unchanged. Maps to `about_platform_ids`. |
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
The shift from "entity manages its own memory" to "extraction pipeline manages memory" changes the agent's relationship with its memories. The entity might produce worse memories if it can't explicitly curate them, or better memories because the pipeline catches things the entity would miss. Mitigated by keeping `reflect__note` as a correction/emphasis tool. Worth monitoring extraction quality closely in the first week.

### Medium risk: Retrieval quality
Multi-signal scoring with embeddings is more powerful but less predictable than keyword overlap. The `social` preset should be a reasonable starting point, but expect a tuning period. The current `compute_social_context()` is well-tuned for discodon — regression testing against known-good social context output would catch quality drops.

### Medium risk: Corpus growth
Without the 50-note FIFO cap, observation count grows until distillation consumes them. Importance-based scoring should handle this, but if thresholds aren't well-calibrated, retrieval could get noisy. The consolidation sub-phase (dedup/merge patterns at threshold 10) prevents pattern-tier bloat.

### Medium risk: Cost visibility gap during migration
During migration, memory operations add LLM and embedding costs that weren't there before (~2,100 tokens per extraction firing, ~50 tokens per retrieval query). If the `CostCallback` isn't wired correctly, these costs become invisible to the budget system. Combined with the existing bugs (classifier/distillation costs not recorded to BudgetService), the budget hierarchy could show significantly less spending than reality. Fix the existing bugs before adding new cost sources.

### Low risk: Cold start
Entities lose all memory. For R&D this is fine. In production this would be unacceptable — but we're not there yet.

### Low risk: Identity regression
Phase 1 identity (platform IDs on memory rows) is functionally equivalent to the current `about_source_id` system. The same auto-resolution pattern (`_author_source_ids` → `about_context`) applies. Cross-platform identity (the gap the identity module fills) doesn't exist today, so the migration doesn't regress it.

---

## Resolved questions

1. **Should reflect__note survive as an `add_memory` wrapper?** **Yes.** Keep `reflect__note` as a thin wrapper (~15 lines) over 3tears `add_memory`. This preserves eval system compatibility (`_extract_accumulated_state()` scans for `reflect__note`), avoids reworking social cognition templates, and gives the entity an explicit memory-correction tool (extraction-primary, reflect-as-override). The wrapper can be retired once the eval system is updated to scan for `add_memory`. See Seam 3.

2. **How does the attention budget interact with extraction?** Extraction is a background process (fire-and-forget after the turn). It does NOT consume the entity's attention budget. Its cost is recorded post-turn and counts against the *next* turn's budget check via `CostCallback` → `BudgetEnforcer.record_cost()`. Retrieval runs pre-decision; its embedding cost is negligible (~50 tokens) and doesn't warrant attention budget tracking. The attention budget's memory allocation governs how many retrieved memories fit in the prompt (via `max_items` on the retriever). See Seam 2 and Seam 6.

3. **Cross-entity distillation scope:** **Per-universe** (discodon's multi-tenant isolation boundary). A universe is the natural scope — entities within a universe share a domain context (e.g., all DJ entities in the same music universe). Cross-universe distillation would produce irrelevant principles.

4. **MCP tool location:** **Stay in discodon as wrappers.** Memory-related MCP actions (`people`, `person_notes`, `social_wisdom`, `distill`) stay in discodon's MCP entity tool but call 3tears APIs internally. The eval MCP tool (51 actions) stays entirely in discodon. Rationale: MCP tools need discodon-specific context (entity IDs, universe IDs, persona state) that a standalone 3tears MCP server wouldn't have. The wrapper pattern is low-overhead and avoids a new service.

## Open questions

1. **`update_index` translation:** The `reflect__note` wrapper needs to translate index-based addressing to semantic lookup. What similarity threshold makes this reliable? Should the wrapper expose index-based addressing at all, or force the LLM to describe what it wants to update?

2. **Distillation prompt migration:** Discodon's `DEFAULT_DISTILLATION_PROMPT` is DJ-domain-specific. The `DistillationStrategy` implementation needs to generalize or remain DJ-specific. Should it be entity-role-aware (different prompts for DJ vs. facilitator)?

3. **Two-database operational story:** Cosmos DB (entities, turns, prompts, budgets, evals) + PostgreSQL (memories, embeddings). What's the backup/restore procedure? Is this permanent architecture or transitional toward all-PostgreSQL?

---

## Implementation order

Work is organized in phases. Items within a phase are independent and parallelizable.

### Phase 1: 3tears schema and extraction (no identity module)

| Item | Work | Where | Blocked by |
|------|------|-------|-----------|
| 1a | Add `about_names`, `about_platform_ids`, `mentioned_names`, `mentioned_platform_ids` columns to memories schema | 3tears agent-memory | Nothing |
| 1b | Update extraction prompt to identify subjects; accept `about_context` dict for name → platform ID resolution | 3tears agent-memory | Nothing |
| 1c | Add `CostCallback` protocol and wire through MemoryExtractor, MemoryRetriever, distillation | 3tears agent-memory | Nothing |
| 1d | Enrich `add_memory` tool: accept `about` (display name), `type_memory`, `importance` | 3tears agent-memory | 1a |

### Phase 2: Discodon integration

| Item | Work | Where | Blocked by |
|------|------|-------|-----------|
| 2a | Implement `EmbeddingProvider` (VoyageAI wrapper), `ChatModelFactory` wrapper, asyncpg pool setup | discodon | Phase 1 |
| 2b | Build `reflect__note` thin wrapper over `add_memory` | discodon | 1d |
| 2c | Wire `extract()` into turn pipeline (post-turn, fire-and-forget) with `about_context` from perception metadata | discodon | 1b, 2a |
| 2d | Replace `format_wisdom()` + `format_notes()` with `retrieve()` call in prompt construction | discodon | 1a, 2a |
| 2e | Wire `CostCallback` → `BudgetEnforcer.record_cost()` with new `CostEventSource` values | discodon | 1c, 2a |
| 2f | Fix existing budget bugs (classifier/distillation costs not recorded, ARIA source string) | discodon | Nothing |
| 2g | Update MCP entity tool: `people`, `person_notes`, `social_wisdom`, `distill` as 3tears wrappers | discodon | 2c, 2d |
| 2h | Remove `WorkingNote`, `NarrativeWisdom`, `distillation.py`, `distillation_manager.py`, old section formatters | discodon | 2b, 2c, 2d |
| 2i | Implement `DistillationStrategy` (adapted from `DEFAULT_DISTILLATION_PROMPT`) | discodon | Phase 1 |

### Phase 3: Post-migration enhancements (not blocking migration)

| Item | Work | Where | Blocked by |
|------|------|-------|-----------|
| 3a | Build `3tears-identity` package (canonical UUIDs, platform mappings, merge/split) | 3tears | Nothing |
| 3b | Backfill `about_ids` from `about_platform_ids` via identity module | 3tears + discodon | 3a |
| 3c | Switch retrieval to boost by `about_ids` instead of `about_platform_ids` | 3tears | 3b |
| 3d | Update eval system: `_extract_accumulated_state()` scans for `add_memory`, retire `reflect__note` wrapper | discodon | Phase 2 stable |
| 3e | Add v2 eval capabilities: distillation quality, temporal validity, decision evidence, cross-entity patterns | discodon | Phase 2 stable |
| 3f | Enrich `ContextNote` model with `type_memory`, `importance`, `tier`, `epistemic_type` | discodon | Phase 2 stable |
| 3g | Cross-entity distillation (daily background, per-universe scope) | 3tears + discodon | 2i |

Phase 1 is pure 3tears work — no discodon changes. Phase 2 is the actual migration. Phase 3 items are enhancements that build on a working migration.
