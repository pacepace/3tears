# Memory v2: Vision and Requirements

## Purpose

This document describes the next evolution of the 3tears agent-memory system. The changes extend the memory model, retrieval, and distillation to support multi-perspective, multi-entity, multi-platform applications — while keeping 3tears domain-agnostic so that the same framework serves DJ bots, game masters, book clubs, journaling apps, customer support, and other use cases.

The guiding principle: if a capability is needed across different app types, it belongs in 3tears. If it's specific to a single domain, it belongs in the consuming app.

---

## Current state

The agent-memory package provides:

- **Memory extraction** from conversations via a gated, multi-stage LLM pipeline (heuristic gates, worthiness gate, candidate extraction, embedding, similarity-based dedup, LLM resolution with ADD/UPDATE/DELETE/NOOP)
- **Hybrid retrieval** combining semantic search (pgvector), keyword search (PostgreSQL FTS), and recency decay, with MMR reranking for diversity
- **Three-tier caching** (L1 SQLite / L2 NATS KV / L3 PostgreSQL) inherited from threetears.core
- **Memory ledger** tracking which items have been surfaced in a conversation to prevent redundancy
- **Five memory types**: preference, fact, decision, topical_context, relational_context

Memories are scoped by `user_id` (who owns the memory), with `conversation_id` and `message_id_source` tracking provenance. There is no concept of who or what a memory is *about*, no memory tiers beyond the flat type classification, and no automatic distillation or pattern synthesis.

---

## What changes

### A. Multi-perspective memory model

**Problem:** Memories are scoped only by `user_id`. There is no way to express "DJ's memory about Bob's taste" versus "DJ's memory about Abigail's taste" versus "DJ's general wisdom about music." All memories for a given owner live in one undifferentiated pool.

**Solution:** Add entity association to every memory.

- `owner_id` — who holds this memory (a persona, a bot, a user). Replaces `user_id`.
- `about_id` — the canonical identifier of who or what this memory concerns. Nullable — null means the memory is general knowledge, not about any specific entity.
- `about_name` — human-readable label for the entity (for prompt rendering). Updated when the entity's display name changes.
- `about_type` — category hint: person, project, topic, location, goal, etc. Consuming apps define the vocabulary. Used for retrieval grouping and filtering.

The `about` fields are not limited to people. A memory can be about a person (Bob), a project (the novel we're brainstorming), a goal (career transition), a game-world location (the north bridge), or a topic (reinforcement learning). 3tears stores and indexes these fields without interpreting their semantics.

**Why generalized entities, not just people:** Every app type needs memories about non-person things. A DJ has memories about venues. A DM has memories about locations and items. A brainstorming group has memories about the project they're building. A journaling app has memories about the user's goals. Limiting `about` to people would push every app to reinvent topic-scoped memory.

**Why `about` is singular:** A memory like "the dynamic between Alice and Bob" is stored as about=Alice with Bob referenced in the content (discoverable via semantic search). Multi-valued `about` adds schema complexity without clear benefit for v1. If a consuming app needs cross-entity relationship memories, it can create them about each entity separately, or rely on general (about=null) memories.

**Why `about=null` is critical:** General knowledge — wisdom not tied to any specific entity — is what transfers across contexts. A DM's principle "design encounters with initial barriers" applies regardless of which players are present. A DJ's insight "ambient keeps energy steady during chill vibes" applies regardless of who's in the channel. General memories are always eligible for retrieval, never filtered out by entity-matching. They represent the owner's accumulated domain expertise.

### B. Entity resolution

**Problem:** The same person appears as different identifiers on different platforms. "discord:123456789" and "mastodon:@alice@social.example" are the same person, but the memory system has no way to know that.

**Solution:** A lightweight identity resolution registry within 3tears.

- Maps multiple platform-qualified source IDs to a single canonical `about_id`
- When storing a memory, the consuming app can pass a raw source ID; 3tears resolves it to canonical form
- When retrieving, resolution ensures that memories about "discord:123" and "mastodon:@alice" both surface when either identifier is queried
- The consuming app registers identity links (e.g., "these three platform IDs are the same person"); 3tears handles the fan-out

**Scope:** Identity resolution is specialized for person-type entities. For non-person entities (projects, topics, locations), the consuming app assigns canonical IDs directly. 3tears doesn't need to resolve "the novel project" across platforms.

**What stays in consuming apps:** The decision of *when* to link identities. A DJ bot might auto-link when a user says "I'm @alice on Mastodon." A customer support system might link via CRM lookup. 3tears provides the registry; apps provide the data.

### C. Flexible provenance

**Problem:** Provenance is modeled as `conversation_id` + `message_id_source`, assuming a chat-app context. Memories can come from many sources: Discord messages, Mastodon posts, RSS feeds, tool results, reflection turns, orchestrator instructions.

**Solution:** Replace the fixed provenance fields with a generic model:

- `source_type` — app-defined string identifying the source category (e.g., "discord_message", "mastodon_post", "rss_item", "tool_result", "reflection", "orchestrator")
- `source_id` — opaque string referencing the specific source item (a message ID, a feed URL, an internal reference)
- `source_context` — optional structured metadata (channel name, guild, tool name, feed title — whatever the app considers useful for audit)

**Design principle:** Provenance is audit/debug metadata, not a retrieval signal. None of the scenarios we analyzed required "show me where I learned this" as part of normal operation. The extraction and retrieval pipelines pass provenance through without interpreting it. It's stored, queryable for debugging, but not factored into relevance scoring.

### D. Memory tiers and distillation

**Problem:** Memories are flat — every memory is a 1-2 sentence fact at the same level of abstraction. Over weeks of interaction, a DJ accumulates hundreds of individual observations ("Bob liked this song", "Bob liked that song") with no mechanism to synthesize them into patterns ("Bob is a thrash metal fan") or principles ("Metallica fans tend to also enjoy Pantera"). Without synthesis, memory becomes noise.

**Solution:** Introduce memory tiers and a pluggable distillation engine.

#### Tiers

Every memory belongs to a tier representing its level of abstraction:

- **Observation** — a specific fact from a specific moment. "Bob requested Aphex Twin SAW II and loved it." Raw material for pattern recognition.
- **Pattern** — a recurring theme synthesized from multiple observations. "Bob is a deep-cuts ambient electronic fan." More durable and compact than individual observations.
- **Principle** — a generalized insight that transcends specific entities or situations. "Listeners who prefer deep cuts respond poorly to greatest-hits playlists." Transfers across contexts.

#### Within-entity distillation

Periodically, the distillation engine reviews an owner's memories about a specific entity:

1. Identify clusters of observations about the same entity that share thematic similarity
2. Present them to the LLM with a distillation prompt (pluggable by the consuming app)
3. The LLM produces operations:
   - **CONSUME** — observation fully captured in a new or existing pattern; delete the observation
   - **REFINE** — extract the generalizable part into a pattern; rewrite the observation to keep only the operational/specific remainder
   - **KEEP** — observation is too recent or too unique to distill; leave it
4. Patterns may further distill into principles via the same process

#### Cross-entity distillation

When content themes recur across 3+ distinct `about_id` values, the engine triggers cross-entity pattern extraction:

1. Notice that multiple entities share a theme (e.g., several listeners have Metallica-related memories)
2. Run a topic scan retrieving all memories matching that theme, grouped by entity
3. Present to the LLM with a cross-entity distillation prompt
4. Produce `about=null` general patterns or principles ("Metallica fans tend to also enjoy Pantera but are less interested in Queensryche")

Cross-entity distillation is what builds domain expertise — the insights that apply regardless of which specific entities are involved.

#### Type-aware distillation

Not all memory types distill the same way:

- **Observations** are the primary input to distillation. They are consumed as patterns form.
- **Decisions** persist and update-in-place. "The protagonist is a retired detective" is a decision, not an observation — it shouldn't be consumed or generalized. It stays until explicitly superseded by a new decision.
- **Patterns and principles** are the output of distillation. They may be refined or merged as more evidence accumulates.

#### Action memories

The extraction pipeline must capture the owner's own significant actions, not just facts learned from others. "I introduced Bob and Abigail over shared Metallica fandom" and "I recommended Snarky Puppy to Charlie" are durable facts that prevent the owner from repeating itself.

Action memories participate in distillation normally. Individual actions ("I played Metallica for Bob", "I played Pantera for Bob") distill into relationship patterns ("I frequently play thrash metal for Bob, he responds well"). Over time, this builds a picture of relationship state — the DJ doesn't just avoid repeating an introduction, it understands that Bob and Abigail are Metallica buddies and builds on that.

Without action memory, the owner has memories about what it *knows* but not what it *did*. It re-introduces people, re-recommends songs, re-asks questions — because it has no record of having done so before.

#### Pluggable prompts and strategies

3tears provides the distillation engine (scheduling, memory scanning, threshold logic, tier management). Consuming apps provide:

- Distillation prompt templates (how should observations be synthesized in this domain?)
- Tier names and semantics (if the defaults don't fit)
- Trigger thresholds (how many observations before distillation fires? how much time between runs?)

This separation keeps the engine reusable while letting each app tune the cognitive style of distillation.

### E. Entity-aware retrieval

**Problem:** The retriever accepts a single query string and returns scored memories filtered by owner. It has no concept of "who is present" or "what are we working on" as retrieval signals, and its MMR-based output format doesn't support pattern discovery across entities.

**Solution:** Extend the retriever with richer query context and two retrieval modes.

#### Retrieval context

The retriever accepts a structured query context:

- `owner_id` — whose memories to search (required, as today)
- `query` — text for semantic and keyword search (as today)
- `relevant_entities` — list of entity IDs to boost. Memories where `about_id` matches any relevant entity are boosted regardless of semantic distance to the query. This is the primary mechanism for "who is present" and "what are we working on"
- `tier_weights` — optional per-tier scoring weights (e.g., boost principles during reflection, boost observations during active interaction)

#### Two retrieval modes

**Conversational retrieval** — used during active interactions (stimulus turns, live conversation). Optimized for token efficiency and diversity.

- Entity-boosted scoring: memories about relevant entities are lifted
- MMR reranking: diverse results, avoids redundancy
- Tight budget: returns a manageable number of results for prompt inclusion
- Entity-grouped output: results organized by `about_id`, so the LLM can see "what I know about each participant" as a coherent picture rather than an interleaved flat list

**Topic scan** — used during reflection, pattern discovery, and cross-entity analysis. Optimized for completeness.

- Content-first: no entity boosting, pure semantic/keyword matching
- No MMR: returns all memories above threshold, not a diversity-filtered subset
- Entity-grouped output: essential here, since the purpose is identifying which entities share a theme
- Larger budget: comprehensive rather than token-efficient

Both modes share the same underlying search infrastructure (embeddings, FTS, recency scoring). The difference is in post-processing: whether MMR runs, whether entity boosting applies, and how results are formatted.

**Why entity-grouped output matters:** Flat lists of memories hide cross-entity patterns. When the retriever returns memories about Bob and Abigail interleaved by score, the LLM may not notice they share Metallica fandom. When results are grouped by entity, the connection is visually obvious:

> About Bob: loves Metallica, seen them live many times
> About Abigail: loves Metallica, deep fan

This grouping aids the LLM's reasoning without requiring the memory system to do inference.

#### Extraction gate configuration

The extraction pipeline's heuristic gates (message length, response length, turn count, rate limiting) must be configurable per owner or per use case. A quiet facilitator persona that listens but rarely speaks will never pass the "assistant response length >= 100 chars" gate under the current defaults. The gates themselves are sound; they just need per-owner configuration rather than global defaults.

---

## Scenarios

These scenarios illustrate how the components work together across different app types. They are not exhaustive but cover the key interaction patterns.

### DJ: participant joins with known history

A DJ persona is in a channel with humans Albert and Betty, who have been requesting music. Charlie joins. Charlie's discord ID resolves via entity resolution to a canonical person ID. The retriever runs with `relevant_entities=[Albert, Betty, Charlie]`. Memories about Charlie from previous sessions — different channels, weeks ago — surface via entity boosting. The DJ greets Charlie with awareness of their taste and queues appropriate music.

When Charlie leaves and returns days later, the entity boost works the same way. If the DJ played music for Charlie last time, the action memory "I played jazz fusion for Charlie and they loved it" also surfaces, preventing the DJ from re-making the same introduction but allowing it to build on the relationship.

### DJ: connecting listeners with shared taste

Bob says he loves Metallica. Weeks later, Abigail says something similar in a separate session. During Abigail's session, semantic search surfaces Bob's Metallica memory (strong embedding match) even though Bob isn't present. The DJ notes the overlap. Later, when both are present, entity-grouped retrieval shows Metallica memories under both Bob and Abigail. The DJ connects them.

The next time both are present, the DJ's action memory "I introduced Bob and Abigail over shared Metallica fandom" surfaces alongside their taste memories. The DJ doesn't repeat the introduction — it builds on the established connection. Over time, distillation promotes these individual observations into a relationship pattern.

### DJ: cross-entity taste patterns

During a reflection turn, the DJ retrieves listener preferences using topic scan mode. Across many listeners, a theme emerges: several enjoy Metallica and Pantera, but listeners who liked Metallica were unimpressed by Queensryche. Cross-entity distillation synthesizes this into an `about=null` principle: "Metallica fans tend to also enjoy Pantera but are less interested in Queensryche." This principle surfaces in future interactions with any Metallica fan, including new listeners the DJ hasn't met before.

### DJ: multi-fact extraction and inference

Bob says "I love Metallica, I've seen them live eight times but they haven't been to Melbourne in 10 years." The extraction pipeline produces multiple memories from this single message: Bob's Metallica fandom (preference), Bob's likely location near Melbourne (fact), and his lament about touring (fact). These are separate memories, all `about=Bob`, each with its own embedding. Later, when the DJ learns about a Metallica tribute band playing in Melbourne (from another user, an RSS feed, or a web search), semantic search on the event text matches both Bob's Metallica preference and Melbourne location. Entity-grouped output helps the DJ's LLM connect the dots across memories.

### DM: campaign lifecycle

A DM runs a campaign with four player personas across many sessions. Early sessions produce observations: "P2 charged into combat without scouting", "P2 rushed the bandit camp solo." Distillation consumes these into a pattern: "P2 consistently rushes into combat without tactical preparation." Further distillation produces a principle: "Design encounters with initial barriers to prevent rushing."

At campaign's end, the DM's memory is mostly patterns and principles. Hundreds of session-specific observations have been consumed. When the DM starts a new campaign with four *different* personas, person-specific memories from campaign 1 don't surface (different `about_id` values, not present as relevant entities). But general principles do — the DM is a better DM because of what it learned. If a player from campaign 1 joins campaign 2, their specific patterns surface immediately via entity boosting.

Campaign-specific facts ("P3's character is named Elara") naturally age out: they stay as observations (not generalizable), recency decay lowers their score, and they eventually fall below retrieval thresholds. No explicit "campaign scope" tagging is needed.

### Book brainstorming: collaborative artifacts

Three personas brainstorm a novel. They produce decisions: the protagonist is a retired detective, the setting is 1940s Chicago, the tone is noir with magical realism. Each persona stores these as memories with `about_id` pointing to the novel project entity, typed as decisions.

When one persona later sits down to write chapter 1, the retriever is called with `relevant_entities=[novel-project, protagonist-character, antagonist-character]`. All decisions about these entities surface via entity boosting, regardless of semantic distance from "chapter 1 opening." The theme decision, the character backstory, the setting details — all surface because they're about relevant entities, not because they embed close to the query text.

Decisions persist and update-in-place (not consumed by distillation). If the group later changes the protagonist's name, the UPDATE resolution mechanism supersedes the old decision. General observations about the creative process ("brainstorming sessions are more productive when we start with character before plot") may distill into principles.

Each persona stores its own version of the decisions. Slight perspective differences are expected and acceptable. If a consuming app needs a canonical shared record, it designates a "scribe" persona or reconciles across owners at the application layer. 3tears does not implement multi-writer shared state.

### Quiet facilitator: proactive memory surfacing

A facilitator persona observes a channel where humans discuss daily topics. It rarely speaks unless directly addressed or a highly relevant memory is available. Karen says to Brooks: "I don't remember when you were in Spain." The retriever runs with `relevant_entities=[Karen, Brooks]` and query text about Spain. A memory from weeks ago — `about=Brooks: "Brooks traveled to Spain in March 2024"` — surfaces with high relevance (entity boost plus strong semantic match). The facilitator sees a high-confidence match and offers help.

This works because: (1) the facilitator's extraction pipeline captures facts from conversations it observes, even when it doesn't respond — extraction gates are configured for this persona to not require assistant responses; (2) entity-boosted retrieval surfaces the Spain memory because Brooks is present; (3) the consuming app's behavioral logic interprets the high retrieval score as grounds for proactive engagement.

### Book club: evolving member understanding

A moderator persona runs a weekly book club with humans and personas. After several weeks, per-member observations distill into patterns: "Alice consistently engages most with literary technique regardless of genre", "Bob dominates discussions about politically charged books." These patterns inform the moderator's facilitation — asking Alice about symbolism, managing Bob's airtime during political books.

General principles also emerge: "Asking each person for a key takeaway before open debate keeps things structured" and "For politically charged books, establish speaking order." These are `about=null` insights that apply regardless of which book is being discussed or which members attend.

### Single-user journaling: goal tracking

A journaling chatbot tracks a human's goals over months. Memories are stored with `about_id` pointing to goal entities: career-transition, fitness, relationship-with-sister. When the user asks "how am I doing?", the retriever is called with `relevant_entities=[career-transition, fitness, relationship-with-sister]`. All memories about each goal surface, grouped by entity. Distillation has synthesized months of daily entries into patterns: "Client makes progress on fitness when accountability partner is involved" and "Career transition stalled after networking event anxiety." Cross-entity distillation might notice: "When the client makes progress on fitness, career motivation also increases."

---

## Boundaries: what lives where

### 3tears owns

- **Memory storage and retrieval** — three-tier caching, embedding, hybrid scoring, entity-boosted and topic-scan retrieval modes, entity-grouped output
- **Memory model** — owner, about (generalized entities), provenance, tier, type
- **Entity resolution** — registry mapping platform IDs to canonical IDs for person-type entities
- **Extraction pipeline** — gated, multi-stage extraction of both learned facts and action memories, with configurable gates
- **Distillation engine** — scheduling, memory scanning, tier management, within-entity and cross-entity pattern synthesis. Pluggable prompts and strategies.
- **Memory ledger** — within-conversation tracking of surfaced items to prevent redundancy

### Consuming apps own

- **Entity registry** — what entities exist and what they mean (personas, characters, projects, goals, locations). 3tears stores `about_id`/`about_name`/`about_type`; the app decides what entities to create and how to model them.
- **Relevance determination** — which entities are relevant for a given interaction ("who is present", "what project are we working on"). The app passes relevant entity IDs to the retriever; 3tears boosts them.
- **Identity linking** — deciding *when* to link platform identities. The app registers links with 3tears's resolution registry; 3tears resolves them on storage and retrieval.
- **Behavioral rules** — "speak only when directly addressed or high-confidence memory available", "don't repeat introductions", etc. 3tears provides scored retrieval; the app interprets scores and decides actions.
- **Prompt formatting** — how retrieved memories are rendered into LLM prompts. 3tears provides entity-grouped structured output; the app formats it into its prompt template.
- **Distillation prompts** — domain-specific guidance for how observations should be synthesized. The app provides prompt templates; 3tears runs the engine.
- **Multi-owner reconciliation** — if an app needs a canonical shared artifact from multi-persona collaboration, it reconciles across owners at the application layer. 3tears stores per-owner memories.
- **Cognitive/behavioral architecture** — persona traits, goals, scheduling, turn management, reflection triggers. These are consuming app concerns that *use* the memory system but don't live in it.

### Why these boundaries

The boundary test is reuse: if every app rebuilding the same capability would produce essentially the same code, it belongs in 3tears. If apps would produce meaningfully different implementations reflecting their domain, it belongs in the app.

Retrieval scoring, distillation scheduling, tier management, and entity resolution are the same everywhere — only the configuration differs. These belong in 3tears.

Entity semantics (what is a "project"?), behavioral rules (when should I speak?), and prompt formatting (how does my persona present memories?) are fundamentally domain-specific. These belong in consuming apps.

---

## Dependency order

```
A (memory model) ──> B (entity resolution) ──> D (distillation)
                └──> C (provenance)        ──> E (retrieval)
```

**A is prerequisite to everything.** The owner/about/type schema is the foundation. B and C are independent of each other but both depend on A. D and E are independent of each other but each depends on the work before it.

Work within each item can be incremental. For example, A can ship with just the schema changes before B adds resolution on top, and E can ship conversational retrieval before adding topic scan mode.
