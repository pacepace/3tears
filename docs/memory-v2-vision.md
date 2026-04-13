# Memory v2: Vision and Requirements

## Purpose

This document describes the next evolution of the 3tears agent-memory system — extending the memory model, retrieval, and distillation to support multi-perspective, multi-entity, multi-platform applications while keeping 3tears domain-agnostic. The same framework should serve DJ bots, game masters, customer support agents, personal assistants, and whatever else people build.

**Guiding principle:** If a capability is needed across different app types, it belongs in 3tears. If it's specific to a single domain, it belongs in the consuming app.

**Foundation principle:** The schema defines the ceiling of what the processing pipelines can eventually do. Several fields described here enable capabilities that won't ship in v2. We add them now because retrofitting columns into a populated database is painful, and adding nullable columns to the initial schema is cheap. Fields that support future capabilities are marked *[foundation]*.

---

## Research foundations

This design draws on the growing body of work on agent memory. We won't repeat the full literature here — just the findings that directly shaped our architecture.

**Generative Agents** (Park et al., 2023) showed that reflection — synthesizing observations into higher-level insights — was the single most impactful component in their agent architecture, outranking retrieval quality and planning in ablation studies. Their three-factor retrieval scoring (recency + importance + relevance) has become the de facto standard.

**MemGPT / Letta** (Packer et al., 2023) demonstrated that agents controlling their own memory outperform passive systems. Their "sleep-time compute" — asynchronous refinement during idle periods — directly informs our distillation scheduling.

**Zep / Graphiti** (2024-2025) proved that when facts change over time, systems that invalidate-but-preserve old facts dramatically outperform systems that overwrite. This is why our UPDATE operation creates a new memory rather than clobbering the old one.

**ExpeL** (Zhao et al., 2023) and **Reflexion** (Shinn et al., 2023) both showed that accumulated lessons from experience improve future performance — and that quality of insights matters more than quantity. A few good principles beat a hundred shallow observations.

**CoALA** (Sumers et al., 2024) surveyed 100+ agent architectures and found the best systems distinguish episodic memory (what happened) from semantic memory (what is generally true), with different strategies for each.

**LoCoMo**, **StreamBench**, **LONGMEM-Bench** (2024) established that all memory systems degrade beyond ~20 interactions; temporal reasoning is the hardest task; and hierarchical memory with consolidation outperforms flat stores by 15-25% at long horizons.

**Memory drift research** (2024-2025) documented ~5-10% error at key decision points per LLM processing step. Errors compound through multi-stage pipelines. This is why we don't hard-delete consumed observations.

**Project Sid** (Altera, 2024) scaled to 1,000+ agents and found that pure narrative memory becomes noise at scale — agents needed structured relationship state alongside free-text memory.

**LARP** (Shao et al., 2023) found that action memory (remembering what the agent *did*) matters more for persona consistency than detailed identity descriptions. An agent that forgets its own actions will re-introduce people, re-recommend songs, and re-ask questions.

---

## Current state

The agent-memory package provides:

- **Memory extraction** via a gated, multi-stage LLM pipeline (heuristic gates, worthiness gate, candidate extraction, embedding, dedup, LLM resolution with ADD/UPDATE/DELETE/NOOP)
- **Hybrid retrieval** combining semantic search (pgvector), keyword search (PostgreSQL FTS), and recency decay, with MMR reranking
- **Three-tier caching** (L1 SQLite / L2 NATS KV / L3 PostgreSQL) inherited from threetears.core
- **Memory ledger** tracking surfaced items to prevent redundancy
- **Five memory types**: preference, fact, decision, topical_context, relational_context

Memories are scoped by `user_id` with `conversation_id` and `message_id_source` as provenance. There's no concept of who or what a memory is *about*, no epistemic metadata, no abstraction tiers, and no automatic synthesis.

---

## What changes

### A. Multi-perspective memory model

**Problem:** All memories for an owner live in one undifferentiated pool. There's no way to express "DJ's memory about Bob's taste" versus "DJ's general wisdom about music."

**Solution:** Add entity association to every memory.

- `owner_id` — who holds this memory (a persona, a bot, a user). Replaces `user_id`.
- `about_id` — the primary entity this memory concerns. Nullable — null means general knowledge.
- `about_type` — category hint: person, project, topic, location, goal, etc. Apps define the vocabulary.
- `mentioned_entity_ids` — *[foundation]* other entity IDs referenced in this memory beyond the primary `about_id`.

Entities aren't limited to people. A DJ remembers venues. A DM remembers locations. A support agent remembers products. 3tears stores and indexes these fields without interpreting their semantics.

**Why `about` is singular with `mentioned_entity_ids`:** Embedding models are unreliable for proper nouns (ACM Web Conference 2024). Storing "Bob and Alice share Metallica fandom" as `about=Bob` and hoping semantic search finds "Alice" in the content is a gamble we'd rather not take. `mentioned_entity_ids=[Alice]` makes it explicit. Retrieval boosts memories where *any* relevant entity appears in either field.

We considered multi-valued `about` (co-equal primary entities) and rejected it — it complicates entity-grouped output (which group does the memory appear in?) without clear benefit over the primary/mentioned split.

**Why `about=null` matters:** General knowledge transfers across contexts. A DM's principle "design encounters with initial barriers" applies regardless of which players are present. A support agent's insight "customers who mention 'manager' need immediate de-escalation" works for any customer. These memories are always retrieval-eligible, never filtered out by entity-matching. They are the owner's accumulated expertise.

**`about_name` resolution:** Display names are not stored on memory rows — that would mean updating 500 rows when someone changes their handle. Instead, the consuming app provides a name lookup (a callback or post-processing on results). 3tears returns opaque UUIDs; the app adds human-readable labels.

### B. Epistemic metadata

**Problem:** "Bob lives in Melbourne" (inferred from one offhand comment) and "Bob likes thrash metal" (stated directly, confirmed repeatedly) are treated identically. They shouldn't be.

The Generative Agents showed importance scoring was critical for separating signal from noise. CoALA found the best systems use different strategies for different epistemic categories. LLM theory-of-mind research (PNAS 2024) shows ~25% error on false-belief tasks — the system needs to track that uncertainty rather than pretending everything is equally certain.

**Solution:** Add epistemic metadata to every memory.

- `importance` — integer 1-10, assigned at extraction. "Bob said hello" (2) vs "Bob announced he's moving to London" (8). Used for retrieval scoring and as a distillation trigger.

- `confidence` — *[foundation]* float 0.0-1.0. Directly observed facts get high confidence; inferences get medium; hypotheses about motivations get low. Retrieval can weight by confidence; prompt rendering can flag low-confidence memories.

- `epistemic_type` — *[foundation]* how the agent came to believe this: `observed`, `stated`, `inferred`, `hypothesized`, `decided`, `reported`. This determines revision behavior: decisions persist until superseded; inferences are open to revision; hypotheses actively invite contradiction.

**In practice:**

- **Chatbot:** "User prefers PostgreSQL over MongoDB" — `stated`, confidence 0.95, importance 7.
- **DJ:** "Bob probably lives near Melbourne" — `inferred`, confidence 0.5, importance 5.
- **DM:** "P2 is probably impatient rather than strategically aggressive" — `hypothesized`, confidence 0.4, importance 7.
- **Support:** "Customer may be evaluating competitors" — `hypothesized`, confidence 0.3, importance 8. (Low confidence, but if it's true, it matters a lot.)

### C. Memory type taxonomy

**Problem:** The v1 types were designed for a flat, single-entity model. `topical_context` and `relational_context` are now redundant with entity scoping. Meanwhile, there's no distinction between facts learned and actions taken, no representation of uncertain beliefs, and no forward-looking commitments.

**Solution:** Six types covering the distinct epistemic categories an agent needs.

- **`preference`** — Likes, dislikes, tastes, working style. *"Bob prefers deep cuts over greatest hits."*
- **`fact`** — Biographical, situational, or world-state facts. *"The campaign is in Act 2." "Customer has been on Enterprise since 2024."*
- **`decision`** — Declared choices, established by authority. Never consumed by distillation — stays until explicitly superseded. *"The protagonist is a retired detective." "The refund exception was approved by the team lead."*
- **`action`** — Something the owner did. Structurally distinct so retrieval can answer "what have I done with this entity?" and prevent repetition. (LARP research found action memory more important for consistency than identity descriptions.) *"I introduced Bob and Alice over shared Metallica fandom." "I escalated this ticket to Tier 2."*
- **`belief`** — Theory of mind, motivational inference, hypotheses. Lower default confidence than facts; actively revised when contradicted. *"Brian positions others as shields — likely strategic self-preservation." "This customer is probably evaluating competitors."*
- **`commitment`** — *[foundation]* Forward-looking promises or intentions. Lifecycle: active → fulfilled / broken / expired. *"I promised to find that Aphex Twin deep cut for Charlie." "I told the customer I'd follow up by Friday."*

**Dropped:** `topical_context` (→ `fact` + entity scoping), `relational_context` (→ `fact` or `decision` + entity scoping).

### D. Entity identity is external to 3tears

**Problem:** The same person appears as "discord:123456789" and "mastodon:@alice@social.example." The same project is "the novel" and "Chicago Noir."

**Why this isn't 3tears's problem:** Identity resolution fails the boundary test. A DJ bot auto-linking handles, a support system integrating Salesforce, an enterprise app federating SSO — these are fundamentally different implementations. Building a simplified version inside a memory system means maintaining an identity service that's worse than purpose-built alternatives.

**3tears's contract:** `about_id` and `mentioned_entity_ids` are opaque canonical UUIDs. 3tears never resolves them. The consuming app maps platform identifiers to canonical IDs *before* calling `extract()` or `retrieve()`.

**Entity merge utility:** When an external identity system merges two entities, the app calls `merge_entity_ids(old_id, new_id)` — a bulk update, not an identity service. It doesn't know *why* the IDs merged, just that they did.

**Platform/channel awareness:** "Remember what I said on Slack?" is a `source_context` query, not an identity question. Memories carry platform metadata in provenance fields: `source_type='slack_message', source_context={"channel": "#general"}`.

**`about_context` implications:** The extraction pipeline's `about_context` must include not just "who's present" but a broader entity vocabulary — known entities the conversation might reference. If Bob mentions Charlie who isn't in the channel, the app needs Charlie's canonical ID available for extraction to associate the memory correctly.

Requirements for an external identity system are in the appendix.

### E. Flexible provenance

**Problem:** Provenance assumes a chat context (`conversation_id` + `message_id_source`). Memories can come from Discord, Mastodon, RSS, tool results, support tickets, reflection turns, or orchestrator instructions.

**Solution:** Replace fixed provenance with a generic model:

- `source_type` — app-defined category (e.g., "discord_message", "support_ticket", "reflection")
- `source_id` — opaque reference to the specific source item
- `source_context` — optional structured metadata (channel, ticket number, feed title)

Provenance is mostly audit/debug metadata. The one exception: `epistemic_type=reported` memories (third-party information) may get lower confidence based on source type. A DJ learning about Bob's taste from Bob directly is more reliable than hearing about it from Charlie.

### F. Memory tiers and distillation

**Problem:** Every memory is a 1-2 sentence fact at the same abstraction level. Over weeks, a DJ accumulates hundreds of "Bob liked this song" observations with no mechanism to synthesize "Bob is a thrash metal fan" or "Metallica fans tend to enjoy Pantera." Without synthesis, memory becomes noise. LONGMEM-Bench (2024) confirmed: hierarchical memory outperforms flat stores by 15-25% at long horizons.

**Solution:** Tiers and a pluggable distillation engine.

#### Tiers

- **Observation** — a specific fact from a specific moment. *"Bob requested Aphex Twin SAW II and loved it." "Customer called about billing discrepancy, ref #4821."*
- **Pattern** — a recurring theme synthesized from multiple observations. *"Bob is a deep-cuts ambient electronic fan." "This customer escalates when response time exceeds 24 hours."*
- **Principle** — a generalized insight that transcends specific entities. *"Listeners who prefer deep cuts respond poorly to greatest-hits playlists." "Customers who mention competitors by name are 3x more likely to churn."*

This mirrors the Generative Agents' observation → reflection hierarchy, but with two improvements: explicit CONSUME operations that keep the corpus manageable, and cross-entity distillation that single-agent architectures can't do.

#### Distillation triggers

Distillation isn't continuous — it fires at specific moments.

**Within-entity distillation** triggers when:
1. Accumulated importance on an entity's observations since last distillation exceeds a threshold (default: 50). Busy entities distill often; quiet ones rarely.
2. A session boundary occurs with at least N new observations (default: 5).
3. The app explicitly requests it (e.g., during a reflection turn).

**Cross-entity distillation** runs on a background schedule (default: daily), matching Letta's sleep-time compute model:
1. Retrieve patterns produced since the last run.
2. Cluster them by embedding similarity.
3. For clusters spanning 3+ distinct entities, synthesize `about=null` principles.

Both triggers are configurable per-owner or per-preset.

#### Non-destructive distillation operations

The LLM produces one operation per observation under review:

- **CONSUME** — observation fully captured in a pattern. The observation is **soft-deleted** and linked to the pattern via `consumed_by_id`. The pattern records the observation in `derived_from_ids`. The observation is excluded from retrieval but preserved for audit, re-processing, and deletion tracing.
- **REFINE** — generalizable part becomes a pattern; observation is rewritten to keep only the specific remainder. Both are preserved.
- **KEEP** — too recent, too unique, or too important to distill. Left alone.

Patterns may further distill into principles via the same process.

**Why non-destructive:** Memory drift research documents ~5-10% error per LLM processing step. By the time an observation is consumed, it's been through 4+ LLM calls. Hard-deleting the source evidence means you can never audit whether the pattern accurately captured what was consumed, you can't reverse bad distillation, and you can't trace which individual's data contributed to which patterns (hello, GDPR). The storage cost of keeping soft-deleted observations is negligible.

#### Type-aware distillation

- **Observations** (preference, fact, action, belief) are the primary distillation input.
- **Decisions** are skipped entirely — they persist until explicitly superseded.
- **Commitments** *[foundation]* have their own lifecycle, managed separately.
- **Beliefs** distill into belief-type patterns with theory-of-mind language, not behavioral summaries. The prompts handle this.

#### Theory of mind in distillation

"Brian frequently suggests risky actions" is shallow. "Brian consistently positions other players in high-risk roles while keeping himself safe — likely strategic self-preservation" is useful, because it's a predictive model, not a behavioral log.

GPT-4-class models achieve ~75% on standard false-belief tasks (PNAS 2024) and adult-level performance on higher-order recursive reasoning. Good enough to be useful; unreliable enough that results must be tracked as beliefs, not facts.

Default distillation prompts guide the LLM to:

1. **Infer motivations:** "Why might this person behave this way?"
2. **Classify epistemic status:** Theory-of-mind patterns are automatically `type_memory=belief` with confidence reflecting evidence strength.
3. **Make predictions testable:** "If this theory is correct, what would we expect next?" — giving the system a basis for future revision.

The extraction pipeline must also capture beliefs and inferences, not just facts. When a DM reflects "I think Brian is using P1 as a shield," that's a valuable memory even though it's subjective. The worthiness gate must not reject it.

Theories revise naturally through distillation. When Brian sacrifices himself after a history of self-preservation, REFINE updates the theory and `derived_from_ids` links back to both the original evidence and the contradiction.

#### Pluggable prompts and strategies

3tears provides the engine (scheduling, scanning, triggers, tier management). Apps provide:

- Distillation prompt templates
- Tier names and semantics (if defaults don't fit)
- Trigger threshold overrides

```python
class DistillationStrategy(Protocol):
    def within_entity_prompt(self, owner_context: str, entity_context: str) -> str: ...
    def cross_entity_prompt(self, owner_context: str, theme_context: str) -> str: ...
    def importance_threshold(self) -> int: ...          # default: 50
    def min_observations(self) -> int: ...              # default: 5
    def background_schedule(self) -> timedelta | None: ...  # default: 24h
    def tier_semantics(self) -> dict[str, str]: ...     # default: observation/pattern/principle
```

A default strategy ships with 3tears. Apps that don't customize get sensible defaults.

### G. Relationship modeling

**Problem:** Relationships — a DJ connecting listeners, a DM tracking player dynamics, a support agent noting account referrals — are inherently about the *connection between* entities, not about either one. Project Sid (2024) found this at scale: agents needed explicit relationship state alongside narrative memory.

**Solution for v2:** Relationship awareness through the memory model. `about_id` + `mentioned_entity_ids` ensures relationship memories surface for all involved entities. A DJ stores "I introduced Bob and Alice over Metallica" as `about_id=Bob, mentioned_entity_ids=[Alice], type=action`. When either is present, it surfaces. When both are, it gets double entity boost.

Distillation synthesizes relationship patterns from recurring entity-pair observations. "Bob and Alice have a strong shared connection around Metallica that I've actively cultivated."

**What this doesn't handle (future):** Typed relationship edges (friend/rival/collaborator), trust scores, relationship strength. These would live in a dedicated `relationships` table populated by distillation when it detects recurring entity-pair patterns. The v2 schema supports this — `mentioned_entity_ids` already captures the pairs.

### H. Temporal validity

**Problem:** When Bob moves from Melbourne to Sydney, UPDATE overwrites the old memory. Now "where did Bob used to live?" is unanswerable. LoCoMo (2024) found temporal ordering to be the hardest task category across all systems. StreamBench (2024) showed most vector systems fail when facts contradict over time.

**Solution:**

- `valid_until` — *[foundation]* timestamptz, nullable. Null means "currently true." When a fact is superseded, the old memory gets `valid_until` set; the new fact is a new memory. Retrieval defaults to `valid_until IS NULL` but can include historical facts.

The resolution pipeline's UPDATE is reinterpreted: instead of overwriting content, it (1) stamps `valid_until` on the old memory, (2) creates a new memory. The LLM still produces UPDATE with memory_id and new content; the engine handles the bookkeeping transparently.

This isn't a full bi-temporal model (Zep uses four timestamps), but it covers the common case without the operational weight.

### I. Entity-aware retrieval

**Problem:** The retriever takes a query string, returns scored memories filtered by owner. No concept of "who is present," no importance scoring, no output structure for cross-entity reasoning.

#### Multi-signal scoring

Four signals, extending the Generative Agents' validated three-factor model:

1. **Semantic similarity** — cosine distance between query and memory embeddings.
2. **Keyword match** — PostgreSQL FTS. Catches exact matches embeddings miss, especially proper nouns.
3. **Recency** — exponential decay on `date_last_accessed`, not just `date_created`. Frequently-retrieved memories stay fresh; unused ones fade. *(Change from v1.)*
4. **Importance** — the memory's importance score, normalized to [0, 1].

Default weights: `semantic=0.40, keyword=0.10, recency=0.20, importance=0.15, entity_boost=0.15`. Configurable per-preset.

#### Entity boosting

Memories where `about_id` or `mentioned_entity_ids` matches a relevant entity get an additive boost — independent of semantic similarity. A memory about Bob's Melbourne location surfaces when Bob is present, even if the query is about music.

#### Budget allocation

To prevent general principles from crowding out entity-specific context (or vice versa):

- **Entity-specific budget** (default 70%) — memories matching a relevant entity.
- **General budget** (default 30%) — `about_id IS NULL` memories (principles, domain wisdom).

Within each partition: multi-signal scoring + MMR selection. Ratios are configurable.

#### Retrieval context

- `owner_id` — whose memories (required)
- `query` — text for semantic/keyword search
- `relevant_entities` — entity IDs to boost
- `tier_weights` — optional per-tier scoring adjustments
- `type_filter` — *[foundation]* include/exclude memory types
- `include_historical` — *[foundation]* include superseded memories (default: false)

#### Two modes

**Conversational** — active interactions. Entity boosting, budget allocation, MMR, tight budget, entity-grouped output.

**Topic scan** — reflection and cross-entity analysis. No entity boosting, no MMR, all memories above threshold, entity-grouped, large budget.

Both share the same search infrastructure; the difference is post-processing.

**Why entity-grouped output:** Flat lists hide patterns. When memories about Bob and Alice are interleaved by score, the LLM might not notice they both love Metallica. Grouped output makes the connection obvious without requiring the memory system to do inference.

#### Extraction gate configuration

Per-owner gate overrides are essential. A quiet facilitator that rarely speaks will never pass the default "response length >= 100 chars" gate. A support bot processing tickets has no concept of "turn count." The gates are sound; they just need per-owner configuration.

---

## Scenarios

These illustrate how the components work together. Not exhaustive — just enough to exercise each part of the architecture.

### Chatbot: evolving user understanding

A personal assistant serves one human over months. Early observations: "User prefers PostgreSQL — cites JSONB, mature tooling, pgvector" (preference, stated, importance 7), "Senior backend engineer at Acme Corp" (fact, stated, importance 6).

Over weeks, observations distill into patterns: "User is building MetaLLM, a self-hosted LLM orchestrator with Python/FastAPI, LangGraph, pgvector." The pattern is more compact than the dozen individual observations about architecture decisions.

The assistant stores actions too: "I suggested switching from OpenAI embeddings to VoyageAI and the user agreed" (action, importance 6). This prevents re-suggesting the same migration. Over time, actions distill into a relationship pattern: "User trusts my infrastructure recommendations but prefers to make their own API design choices."

### DJ: participant joins with known history

Charlie joins a channel. The consuming app resolves Charlie's discord ID to a canonical UUID and passes it to the retriever. Memories from previous sessions — different channels, weeks ago — surface via entity boosting. Charlie's memories include: "Charlie loves jazz fusion, especially Snarky Puppy" and "I recommended Weather Report to Charlie and they loved it." The DJ builds on the relationship rather than starting from scratch.

### DJ: connecting listeners

Bob mentions loving Metallica. Weeks later, Alice says the same in a different session. When both are later present, entity-grouped retrieval shows:

> **About Bob:** loves Metallica, seen them live eight times (importance 8)
> **About Alice:** deep Metallica fan, prefers deep cuts (importance 7)

The connection is obvious in grouped output. The DJ introduces them and stores: `about_id=Bob, mentioned_entity_ids=[Alice], type=action, importance 7`. Next time either is present, the introduction memory surfaces. The DJ doesn't repeat it — it builds on the connection.

### DJ: cross-entity principles

Background distillation notices patterns spanning 4+ listeners around Metallica, Pantera, and Megadeth. It synthesizes an `about=null` principle: "Metallica fans tend to enjoy other thrash metal but are less interested in progressive metal." This surfaces for any future Metallica fan, including people the DJ has never met.

### DM: theory of mind across a campaign

A DM runs a D&D campaign with humans Brian and Cass and AI personas P1 and P3. Early observations:

- "Brian told P1 to charge the dragon while Brian stayed behind cover" — about Brian, mentioned [P1], importance 6
- "Brian suggested P1 scout the cave alone" — about Brian, mentioned [P1], importance 5
- "Brian volunteered to guard the exit while others entered the dungeon" — about Brian, importance 5

Distillation synthesizes a belief: "Brian consistently positions others — especially P1 — in high-risk roles while keeping himself safe. Likely strategic self-preservation." Confidence 0.5, importance 8. `derived_from_ids` links to all three source observations (soft-deleted but preserved).

Then Brian sacrifices his character to save Cass from a collapsing tunnel. Importance 9. On the next distillation trigger, REFINE updates the theory: "Brian's earlier self-preserving behavior may have been strategic resource management rather than cowardice — he proved willing to sacrifice when a valued ally was in genuine danger." The refined pattern links back to both the original evidence and the contradiction.

The DM uses this to design moral dilemmas for Brian rather than routine combat.

### DM: campaign lifecycle

By the end of Campaign 1, hundreds of observations have been consumed into patterns and principles. Campaign 2 starts with different players. Player-specific memories from Campaign 1 don't surface (different entities, not in `relevant_entities`). But general principles do:

- "Design encounters with initial barriers requiring planning."
- "Players engage more when NPCs have personal stakes."
- "Mixing puzzle and combat in the same encounter beats sequential encounters."

The DM is better at Campaign 2 because of what it learned. If a Campaign 1 player joins Campaign 2, their behavioral patterns surface immediately.

### Customer support: escalation patterns

A support agent handles hundreds of customers. Observations accumulate:

- "Customer 4821 asked about data export formats twice in one week" — fact, importance 5
- "Customer 4821 mentioned 'evaluating options' re: contract terms" — fact, importance 7
- "Customer 4821's tone shifted from friendly to terse after the billing discrepancy" — belief (inferred), confidence 0.6, importance 6

Distillation synthesizes: "Customer 4821 is likely evaluating competitors — churn risk." Belief-type pattern, confidence 0.4, importance 9.

Actions and commitments: "I escalated 4821's billing discrepancy to Tier 2" (action), "I told 4821 I'd follow up by Friday" (commitment, importance 7). When 4821 calls again, entity-boosted retrieval surfaces the escalation history, churn assessment, and outstanding commitment — preventing the classic support failure of "can you explain the situation again?"

Cross-entity principles emerge: "Customers who ask about export formats and contract terms in the same week are 3x more likely to churn." These surface for any customer showing similar patterns.

### Customer support: multi-entity relationship

Customer 4821 mentions customer 5102 referred them: `about_id=4821, mentioned_entity_ids=[5102], type=fact`. When either contacts support, the referral surfaces. If 5102 has a bad experience, the agent knows 4821's loyalty may be affected.

### Quiet facilitator: proactive surfacing

A facilitator observes a channel but rarely speaks. Karen says to Brooks: "I don't remember when you were in Spain." The retriever runs with `relevant_entities=[Karen, Brooks]` and query text about Spain. A weeks-old memory surfaces: "Brooks traveled to Spain in March 2024" — strong entity boost plus semantic match, importance 6. The facilitator offers: "Brooks mentioned being in Spain last March."

This works because extraction gates are configured with the response-length gate disabled (the facilitator doesn't speak to generate responses), and the app's behavioral logic treats high-scoring retrieval as grounds for proactive engagement.

---

## Agent autonomy modes

The same schema supports everything from passive assistants to fully autonomous agents. The differences are in configuration, not architecture.

### Passive assistants

Single-user chatbots, copilots, journaling apps. One owner, one entity. Mostly preferences and facts, little theory of mind, simple retrieval. Distillation focuses on preference synthesis and goal tracking. What matters: accurate capture, temporal validity (preferences change), avoiding repetition.

### Facilitators and observers

Channel observers, meeting note-takers. One owner, many entities, low initiative. Extraction gates relaxed (no response-length requirement). Action memory prevents repeating interventions. What matters: extraction from observed conversations without participation, high-precision entity attribution, proactive surfacing at the right moment.

### Social agents

DJs, book club moderators, community facilitators. Many entities, high initiative. Full theory of mind, heavy use of `mentioned_entity_ids`, cross-entity distillation, both retrieval modes. What matters: theory of mind, action memory (don't repeat introductions), cross-entity patterns, temporal validity.

### Autonomous creators

DM agents, author agents, planners. Everything social agents need, plus: persistent decisions (world-building choices must survive distillation), strategic beliefs, commitment tracking (open plot threads, promises to players), action memory for creative choices. What matters: decision persistence, strategic memory, commitment tracking, knowledge transfer across campaigns/projects.

### Customer support agents

One owner, very many entities. High-volume extraction at moderate importance thresholds. Actions and commitments are critical ("what did I promise this customer?"). Cross-entity distillation produces escalation and churn patterns. Temporal validity matters — customer status changes constantly. What matters: commitment tracking, action memory, cross-entity principles, temporal validity.

---

## Boundaries: what lives where

### 3tears owns

- **Memory storage and retrieval** — three-tier caching, multi-signal scoring, entity-boosted and topic-scan modes, entity-grouped output, budget allocation
- **Memory model** — owner, about (with mentioned_entity_ids), provenance, tier, type, epistemic metadata, temporal validity
- **Entity merge utility** — bulk ID update when external identity systems merge entities
- **Extraction pipeline** — gated, multi-stage, with entity identification, action capture, importance/confidence assignment, belief acceptance, per-owner gate config
- **Distillation engine** — triggers, within-entity and cross-entity synthesis, non-destructive CONSUME, theory-of-mind defaults, pluggable prompts
- **Memory ledger** — within-conversation tracking to prevent redundancy

### Consuming apps own

- **Entity identity and directory** — canonical ID management, cross-platform linking, display names. 3tears stores opaque UUIDs; the app owns everything else. (See appendix.)
- **Relevance determination** — which entities matter for a given interaction
- **Entity name resolution** — human-readable labels for `about_id` UUIDs in retrieval results
- **Behavioral rules** — when to speak, when to stay quiet, how to interpret retrieval scores
- **Prompt formatting** — how to render entity-grouped results into LLM prompts
- **Distillation prompts** — domain-specific synthesis guidance
- **Multi-owner reconciliation** — canonical shared artifacts from multi-persona collaboration
- **Cognitive architecture** — persona traits, goals, scheduling, turn management
- **Privacy and access control** — who reads whose memories, sensitivity classification, deletion policy

### Why these boundaries

The test is reuse. If every app would produce essentially the same code, it belongs in 3tears. If implementations would differ meaningfully, it belongs in the app.

Retrieval scoring, distillation scheduling, tier management, and non-destructive CONSUME are the same everywhere. Entity identity, behavioral rules, privacy policy, and prompt formatting are fundamentally domain-specific.

**Boundaries are not walls.** Distillation prompts are "the app's concern," but that doesn't mean 3tears ships an empty text field. See the next section.

---

## Privacy and deletion

### What 3tears provides

**Deletion by entity.** `delete_memories_about(entity_id)` finds all memories where `about_id` or `mentioned_entity_ids` matches and soft-deletes them.

**Provenance tracing through distillation.** Non-destructive CONSUME + `derived_from_ids` makes it possible to trace which entity's observations contributed to a given pattern. When deleting entity X, the system identifies patterns derived from X's observations. The app decides: delete the pattern, re-run distillation without X, or leave it.

**Cross-entity principle contamination.** The hardest case. "Metallica fans enjoy Pantera" may derive from multiple entities. Deleting one shouldn't necessarily invalidate the principle. 3tears tracks `contributing_entity_ids` on patterns and principles — the app uses this to decide whether a principle survives a deletion.

### What consuming apps own

- **When to delete** — GDPR, opt-out, account closure, retention policies
- **Sensitivity classification** — "Daughter named Lily, kindergarten fall 2026" and "prefers PostgreSQL" have very different implications
- **Access control** — can persona A read persona B's memories about entity X? 3tears scopes retrieval by `owner_id`; the app defines cross-owner policy
- **Retention schedules** — how long to keep soft-deleted memories, when to hard-delete

---

## Cost model

### Extraction (per turn)

Up to 3 LLM calls: worthiness gate (~350 tokens), candidate extraction (~1,000 tokens), resolution (~750 tokens). Total when extraction fires: ~2,100 tokens. Most turns are rejected by heuristic gates (zero cost) or worthiness (~350 tokens).

### Within-entity distillation (per entity per run)

~2,000 tokens (prompt + 5-15 observations + existing patterns). Fires roughly every 5-15 interactions per entity with default triggers.

### Cross-entity distillation (per daily cycle)

Depends on corpus. For a modest corpus (50 entities, 200 patterns): 2-5 clusters detected, ~12,000 tokens total.

### Retrieval

No LLM cost. Database queries only. p95 latency under 100ms at typical corpus sizes (<100K memories per owner).

### Scaling

Per-owner corpora are unlikely to hit pgvector limits (even power users generate ~10K memories over years). The cross-entity topic scan is the performance concern: it queries across all entities without MMR. For high-volume owners (support agents with thousands of customers), bound topic scans by recency or tier.

---

## Helping apps succeed

Flexibility without guidance produces shallow memory systems. Most apps will ship with defaults, never revisit their prompts, and wonder why their personas feel flat. Theory of mind won't emerge from a prompt that says "summarize recurring patterns."

3tears takes responsibility for the quality of the whole system, not just the correctness of the architecture.

### Opinionated defaults

The default prompts are 3tears's best answer for the general case, not neutral starting points:

- **Entity awareness** — extraction identifies the primary entity and mentioned entities for each memory
- **Action capture** — extraction notes the owner's own actions, not just facts learned from others
- **Importance scoring** — 1-10 per memory, matching the Generative Agents' validated approach
- **Epistemic classification** — type and confidence on each memory
- **Motivational inference** — distillation asks "why?" not just "what?"
- **Belief acceptance** — worthiness gates don't reject subjective interpretations
- **Tier-appropriate synthesis** — distillation produces genuine patterns, not reworded observations

An app running entirely on defaults should still get entity-aware extraction, action memory, importance scoring, and basic theory of mind. Defaults are the floor, not the ceiling.

### Memory profile presets

Tested configurations bundling prompts, gates, retrieval defaults, and distillation thresholds:

- **assistant** — single-user chatbot. Preferences, facts, decisions. Goal tracking. No cross-entity distillation.
- **social** — multi-persona interaction (DJ pattern). Social observation, action memory, relationship dynamics, theory of mind, cross-entity patterns. Relaxed gates for observers.
- **collaborative** — group work on shared artifacts. Decisions and commitments. Distillation preserves decisions.
- **game** — game master and players. World state, player behavior, strategic inference, cross-player patterns, theory of mind. Heavy action memory for the DM.
- **support** — customer-facing, many entities. Actions and commitments critical. Cross-entity distillation for escalation/churn patterns. Temporal validity important.

Pick a preset, customize from there. Outgrow it, export and modify directly.

### Memory health diagnostics

A linter for memory quality. Runs on-demand, examines an owner's corpus:

- **Entity coverage** — % of memories with non-null `about_id`
- **Mentioned entity usage** — are multi-entity memories populating `mentioned_entity_ids`?
- **Tier distribution** — observations vs patterns vs principles (500 observations and 0 patterns = distillation isn't working)
- **Importance distribution** — well-spread or all clustered at 5?
- **Epistemic diversity** — any beliefs/hypotheses, or all facts?
- **Action memory presence** — if zero, the owner will repeat itself
- **Distillation depth** — are patterns genuinely synthetic or just reworded observations?
- **Theory of mind indicators** — motivational language (intent, strategy, goal) vs behavioral language (frequently, tends to)
- **Temporal health** — conflicting facts without `valid_until` resolution
- **Retrieval simulation** — dry run: what would the retriever return for a sample query?

Output is structured JSON and human-readable.

### Prompt review tool

Static analysis of configured prompts against a checklist: action capture, entity identification, importance scoring, epistemic classification, belief acceptance, motivational inference, decision preservation, observer/facilitator support. Fast, free, runs in CI.

Optionally: run sample scenarios through the pipeline and evaluate output. Costs tokens but catches what static analysis misses.

### Memory configuration skill

3tears ships a Claude Code skill (`/3tears:configure-memory`) that reads an app's docs, selects and adapts presets, generates tailored configuration, and walks through test scenarios. Specification in `docs/skills/configure-memory-skill.md`.

---

## Known limitations and future directions

### v2 limitations

**No knowledge graph.** `about_id` + `mentioned_entity_ids` enables 1-hop entity queries, not graph traversal. "Which listeners have friends who like jazz?" requires app-level joins. `mentioned_entity_ids` is the foundation for a future graph layer.

**No formal relationship model.** No typed edges, trust scores, or strength metrics. The Beta Reputation System and FIRE model from multi-agent research are natural future additions, populated by distillation detecting recurring entity-pair patterns.

**No full bi-temporal model.** `valid_until` handles "what's true now" vs "what was true before" but not Zep's four-timestamp model. Sufficient for most cases.

**Commitment lifecycle is [foundation] only.** The type exists; the fulfillment/expiration engine doesn't. Apps implement tracking at the application layer for now.

**Theory of mind accuracy.** ~25% error rate (PNAS 2024). Beliefs + confidence tracking mitigates but doesn't solve. Treat belief-type patterns as hypotheses, not facts.

**Memory drift.** Each LLM call can introduce subtle errors; extraction + distillation means 4+ calls. Non-destructive CONSUME preserves evidence for audit but doesn't prevent drift.

### Future directions

- **Relationship table** populated by distillation: typed edges, trust scores, relationship strength
- **Graph traversal** leveraging `mentioned_entity_ids` and the relationship table
- **Commitment lifecycle engine**: proactive surfacing, fulfillment tracking
- **Full bi-temporal model** if apps demonstrate need
- **Sensitivity classification** at extraction: LLM-assigned scores for tiered retention
- **Embedding migration tooling**: re-embedding across all tiers including distillation-produced content

---

## Migration guide

v2 is a breaking change. No shims, no fallback. This covers every touching point.

### Database schema

**memories table:**

| v1 column | v2 column | Action |
|---|---|---|
| `user_id` | `owner_id` | Rename |
| `conversation_id` | *(removed)* | Drop (→ `source_type`/`source_id`/`source_context`) |
| `message_id_source` | *(removed)* | Drop (same) |
| *(new)* | `about_id` | Nullable UUID. Existing → `NULL`. |
| *(new)* | `about_type` | Nullable text. |
| *(new)* | `mentioned_entity_ids` | Nullable UUID[]. |
| *(new)* | `source_type` | Nullable text. |
| *(new)* | `source_id` | Nullable text. |
| *(new)* | `source_context` | Nullable JSONB. |
| *(new)* | `tier` | NOT NULL text, default `'observation'`. |
| *(new)* | `importance` | Nullable smallint. Existing → `NULL` (scored as 5). |
| *(new)* | `confidence` | Nullable float. *[foundation]* |
| *(new)* | `epistemic_type` | Nullable text. *[foundation]* |
| *(new)* | `derived_from_ids` | Nullable UUID[]. |
| *(new)* | `consumed_by_id` | Nullable UUID. |
| *(new)* | `date_last_accessed` | Nullable timestamptz. |
| *(new)* | `valid_until` | Nullable timestamptz. *[foundation]* |
| `type_memory` | `type_memory` | `topical_context` → `fact`, `relational_context` → `fact`. New: `action`, `belief`, `commitment`. |

Backfill provenance from old columns if audit matters. Rename `user_id` → `owner_id` on media_content, media, memory_chunks tables.

**Indexes:** `about_id`, `about_type`, `tier`, `importance`, `(owner_id, about_id)`, GIN on `mentioned_entity_ids`.

### MemoryEntity

| v1 | v2 |
|---|---|
| `entity.user_id` | `entity.owner_id` |
| `entity.conversation_id` | *(removed)* |
| `entity.message_id_source` | *(removed)* |
| *(new)* | `entity.about_id`, `.about_type`, `.mentioned_entity_ids`, `.source_type`, `.source_id`, `.source_context`, `.tier`, `.importance`, `.confidence`, `.epistemic_type`, `.derived_from_ids`, `.consumed_by_id`, `.date_last_accessed`, `.valid_until` |

### MemoryExtractor

```python
# v1
extract(pool, user_id, conversation_id, message_id_source,
        user_message, assistant_response, turn_count)

# v2
extract(pool, owner_id, user_message, assistant_response,
        source_type, source_id,
        turn_count=..., source_context=...,
        about_context=..., gate_config=...)
```

`about_context` provides canonical entity IDs and metadata — the app resolves platform identifiers via its own identity service before passing them. When extraction identifies a name not in `about_context`, it emits the raw name for the app to resolve.

Extraction output format:

```json
{
  "type": "preference|fact|decision|action|belief|commitment",
  "content": "1-2 dense sentences",
  "tier": "observation|decision",
  "about_name": "primary entity name or null (app resolves to canonical ID)",
  "mentioned_names": ["other", "entity", "names"],
  "importance": 7,
  "confidence": 0.85,
  "epistemic_type": "observed|stated|inferred|hypothesized|decided|reported"
}
```

### MemoryRetriever

```python
# v1
retrieve(pool, user_id, user_text, ledger=None)

# v2
retrieve(pool, owner_id, query, relevant_entities=None,
         tier_weights=None, mode="conversational", ledger=None)
```

`relevant_entities` matches against both `about_id` and `mentioned_entity_ids`. Results include epistemic metadata and entity grouping.

### MemoryConfig additions

| Field | Purpose |
|---|---|
| `importance_weight` | Scoring weight for importance signal |
| `entity_boost_weight` | Scoring weight for entity match |
| `entity_budget_ratio` | Fraction of budget for entity-specific memories (default: 0.7) |
| `general_budget_ratio` | Fraction for general memories (default: 0.3) |
| `tier_weights` | Per-tier scoring weights |
| `topic_scan_budget` | Max results for topic scan |
| `topic_scan_threshold` | Minimum score for topic scan |
| `distillation_importance_threshold` | Within-entity trigger (default: 50) |
| `distillation_min_observations` | Session-boundary trigger (default: 5) |
| `distillation_background_interval` | Cross-entity schedule (default: 24h) |

### MemoriesCollection

`find_by_user()` → `find_by_owner()`. New methods: `find_by_owner_and_entity()` (matches `about_id` or `mentioned_entity_ids`), `find_by_owner_and_tier()`, `find_by_owner_and_type()`, `find_consumed_by()`, `find_derived_from()`.

### Protocols

**ChatModelFactory:** New `"distillation"` purpose.
**DistillationStrategy:** New protocol. Default ships with 3tears.
**EmbeddingProvider:** Unchanged.

### Tools

`user_id` → `owner_id`. `MemorySearchInput` adds `about_id`, `about_type`, `tier`, `type_memory`, `min_importance` filters.

### Migration checklist

1. **Database:** Schema migration. All existing memories → `tier='observation'`, `about_id=NULL`, `importance=NULL`. Migrate type values.
2. **Entity access:** `user_id` → `owner_id` everywhere.
3. **Extraction calls:** New signature. Minimal: rename, move IDs to source fields, pass `about_context=None`.
4. **Retrieval calls:** New signature. Minimal: rename params. Add `relevant_entities` and `tier_weights` to opt in.
5. **Custom prompts:** Update for new output format with importance, confidence, epistemic_type, mentioned entities.
6. **ChatModelFactory:** Handle `"distillation"` purpose.
7. **Result consumers:** Handle new fields, entity-grouped output, epistemic annotations.
8. **Collection queries:** `find_by_user()` → `find_by_owner()`.

Steps 1-3 are required. Steps 4-8 are incremental — new params default to v1-like behavior. [foundation] fields are nullable and unused until ready.

---

## Dependency order

```
A (memory model + schema)
├──> B (provenance)
├──> C (epistemic metadata + type taxonomy)
├──> D (retrieval: multi-signal scoring, entity boosting, budget allocation)
├──> E (within-entity distillation: triggers, non-destructive CONSUME, ToM prompts)
├──> F (cross-entity distillation: clustering, principle synthesis)
└──> G (temporal validity: supersession semantics)
```

**A is prerequisite to everything.** B and C are independent of each other. D depends on A+C (importance is a scoring signal). E depends on A+C (importance triggers distillation). F depends on E (cross-entity operates on patterns from within-entity). G is independent of everything except A.

Entity identity is external and develops independently. The only integration point is `merge_entity_ids()` in A.

Work within each item is incremental. D can ship with basic entity boosting before budget allocation. E can ship with session-boundary triggers before importance-sum triggers. [foundation] fields are in the schema from A; their processing logic ships with the item that uses them.

---

## Appendix: Requirements for an identity management system

Entity identity is outside 3tears's scope, but apps operating across platforms need an identity service. This describes what it must provide to integrate well with 3tears.

### Why identity is separate

Identity resolution fails the boundary test. A DJ bot auto-linking handles, a support system with Salesforce, an enterprise app with SSO — fundamentally different implementations. The `source_type`/`source_context` fields handle platform awareness. "Remember what I said on Slack?" is a provenance query, not identity.

### What the identity system must provide

**Canonical ID management.** Every entity gets a stable UUID — this is what 3tears stores in `about_id` and `mentioned_entity_ids`.

**Platform-to-canonical mapping.** `(platform, platform_id) → canonical_id`. Queryable in both directions.

**Type-agnostic resolution.** Projects, topics, and game-world locations can have aliases across contexts, not just people.

**Entity metadata.** Display names, types, app-specific attributes. 3tears needs display names for prompt rendering; the app provides them from the identity system.

**Merge operation.** When two canonical IDs turn out to be the same entity:
1. Choose a surviving ID.
2. Update internal mappings.
3. Call 3tears's `merge_entity_ids(old_id, new_id)`.
4. Notify other systems.

Merge early and aggressively. Entity resolution errors compound — a missed merge forks an entity's memory into two incomplete records that semantic search alone may never reunite (KARMA, 2025).

**Split operation (optional).** If a merge was incorrect, the identity system assigns a new canonical ID and selectively updates 3tears memories. Rare but important for high-volume apps.

### Integration pattern

**At extraction:** App resolves platform IDs to canonical IDs, passes them in `about_context` (including a broader vocabulary of entities that might be referenced, not just who's present). Unrecognized names are emitted raw for the app to resolve.

**At retrieval:** App passes canonical IDs as `relevant_entities`. 3tears returns results with raw UUIDs. App decorates with display names.

**On merge:** App calls `merge_entity_ids(old_id, new_id)`. 3tears does a batch update. Done.

### What identity does NOT need to know

Memory content, tiers, types, epistemic metadata, retrieval scoring, distillation state. The integration surface is canonical UUIDs and one batch update. The identity system can be a lookup table or a full CRM — 3tears doesn't care.
