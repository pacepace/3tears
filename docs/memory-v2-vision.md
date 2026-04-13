# Memory v2: Vision and Requirements

## Purpose

This document describes the next evolution of the 3tears agent-memory system. The changes extend the memory model, retrieval, and distillation to support multi-perspective, multi-entity, multi-platform applications — while keeping 3tears domain-agnostic so that the same framework serves DJ bots, game masters, customer support, personal assistants, and other use cases.

The guiding principle: if a capability is needed across different app types, it belongs in 3tears. If it's specific to a single domain, it belongs in the consuming app.

The foundation principle: **the schema defines the ceiling of what the processing pipelines can eventually do.** Several schema fields described here enable capabilities that won't be fully built in v2. We add them now because retrofitting columns into a populated database and migrating consuming apps is far more expensive than adding nullable columns to the initial schema. Fields that support future capabilities are marked with *[foundation]* — they're in the schema from day one but their processing pipelines ship incrementally.

---

## Research foundations

The design in this document is informed by the academic and industry literature on agent memory systems. Key influences:

- **Generative Agents** (Park et al., 2023): Demonstrated that a memory architecture with observation streams and a reflection mechanism produces significantly more believable agent behavior than flat memory. In ablation studies, reflection was the single most impactful component — removing it caused the largest degradation in human-rated believability. Their three-factor retrieval scoring (recency + importance + relevance) has become the de facto standard.

- **MemGPT / Letta** (Packer et al., 2023): Showed that agents with active control over their own memory operations outperform passive memory systems. Their tiered architecture (core memory, recall memory, archival memory) validates the principle that different memory tiers serve different purposes with different access patterns. Their "sleep-time compute" concept — asynchronous memory refinement during idle periods — directly informs our distillation scheduling.

- **Zep / Graphiti** (2024-2025): Demonstrated that temporal knowledge graphs with bi-temporal validity tracking significantly outperform flat vector retrieval for tasks requiring mutable state tracking. Their key finding: when facts change over time, systems that invalidate-but-preserve old facts dramatically outperform systems that overwrite.

- **ExpeL** (Zhao et al., 2023) and **Reflexion** (Shinn et al., 2023): Showed that agents that extract structured lessons from experience and store them for future retrieval significantly outperform agents without accumulated insights. Quality of insights matters more than quantity.

- **CoALA** (Sumers et al., 2024): Survey of 100+ agent architectures finding that the most capable systems distinguish between episodic memory (what happened) and semantic memory (what is generally true), with different write and retrieval strategies for each.

- **LoCoMo** (Maharana et al., 2024), **StreamBench** (2024), **LONGMEM-Bench** (2024): Benchmarks showing that all existing memory systems degrade significantly beyond 20 interactions; temporal reasoning is the hardest task category; and hierarchical memory with consolidation outperforms flat stores by 15-25% at long horizons.

- **Memory drift research** (2024-2025): Documented that each LLM processing step introduces approximately 5-10% error at key decision points (entity identification, temporal disambiguation, contradiction detection). Error compounds through multi-stage pipelines. This directly informs our non-destructive distillation design.

- **Project Sid** (Altera, 2024): Scaled the Generative Agents concept to 1,000+ agents and found that at scale, pure narrative memory becomes too noisy — agents needed explicit structured relationship state alongside free-text memory.

- **LARP** (Shao et al., 2023): Found that agents remembering their own past actions were more consistent than agents with detailed persona descriptions but no action memory. Action memory is more important for persona consistency than identity descriptions.

These references are cited throughout the document where they inform specific design decisions.

---

## Current state

The agent-memory package provides:

- **Memory extraction** from conversations via a gated, multi-stage LLM pipeline (heuristic gates, worthiness gate, candidate extraction, embedding, similarity-based dedup, LLM resolution with ADD/UPDATE/DELETE/NOOP)
- **Hybrid retrieval** combining semantic search (pgvector), keyword search (PostgreSQL FTS), and recency decay, with MMR reranking for diversity
- **Three-tier caching** (L1 SQLite / L2 NATS KV / L3 PostgreSQL) inherited from threetears.core
- **Memory ledger** tracking which items have been surfaced in a conversation to prevent redundancy
- **Five memory types**: preference, fact, decision, topical_context, relational_context

Memories are scoped by `user_id` (who owns the memory), with `conversation_id` and `message_id_source` tracking provenance. There is no concept of who or what a memory is *about*, no epistemic metadata distinguishing facts from hypotheses, no memory tiers beyond the flat type classification, and no automatic distillation or pattern synthesis.

---

## What changes

### A. Multi-perspective memory model

**Problem:** Memories are scoped only by `user_id`. There is no way to express "DJ's memory about Bob's taste" versus "DJ's memory about Abigail's taste" versus "DJ's general wisdom about music." All memories for a given owner live in one undifferentiated pool.

**Solution:** Add entity association to every memory.

- `owner_id` — who holds this memory (a persona, a bot, a user). Replaces `user_id`.
- `about_id` — the canonical identifier of the primary entity this memory concerns. Nullable — null means the memory is general knowledge, not about any specific entity.
- `about_type` — category hint: person, project, topic, location, goal, etc. Consuming apps define the vocabulary. Used for retrieval grouping and filtering.
- `mentioned_entity_ids` — *[foundation]* array of canonical entity IDs referenced in the memory beyond the primary `about_id`. When a memory involves multiple entities, the primary goes in `about_id` and the rest go here.

The `about` fields are not limited to people. A memory can be about a person (Bob), a project (the novel we're brainstorming), a goal (career transition), a game-world location (the north bridge), or a topic (reinforcement learning). 3tears stores and indexes these fields without interpreting their semantics.

**Why generalized entities, not just people:** Every app type needs memories about non-person things. A DJ has memories about venues. A DM has memories about locations and items. A customer support agent has memories about products and open tickets. Limiting `about` to people would push every app to reinvent topic-scoped memory.

**Why `about` is singular with `mentioned_entity_ids`:** Research on embedding quality for proper nouns shows that entity names don't embed reliably (ACM Web Conference 2024 found cosine similarity can be "rendered meaningless" for proper nouns depending on regularization). A memory like "Bob and Alice share deep Metallica fandom" stored as `about=Bob` would be invisible to entity-boosted retrieval when Alice joins. Relying on semantic search to find "Alice" in the content text is unreliable.

`mentioned_entity_ids` solves this cheaply. The memory gets `about_id=Bob, mentioned_entity_ids=[Alice]`. Retrieval boosts memories where *any* relevant entity appears in either field. The primary `about_id` determines entity-grouped output organization; `mentioned_entity_ids` ensures multi-entity memories surface for all involved entities.

Multi-valued `about` (a set of co-equal primary entities) was considered and rejected: it complicates entity-grouped output (which group does the memory appear in?), adds schema complexity to every query, and the primary/mentioned split handles the retrieval need without ambiguity.

**Why `about=null` is critical:** General knowledge — wisdom not tied to any specific entity — is what transfers across contexts. A DM's principle "design encounters with initial barriers" applies regardless of which players are present. A customer support agent's principle "when the customer mentions 'manager,' de-escalation urgency doubles" applies regardless of which customer is present. General memories are always eligible for retrieval, never filtered out by entity-matching. They represent the owner's accumulated domain expertise.

**`about_name` resolution:** Entity display names are resolved at read time from the entity resolution registry, not stored on every memory row. Storing per-row would create an update cascade problem (an entity referenced by 500 memories changes display name → 500 row updates). The retrieval layer looks up `about_id → display_name` once per retrieval call. This is a change from the original v1 design draft that stored `about_name` per row.

### B. Epistemic metadata

**Problem:** All memories are treated as equally confident facts. But agents form memories with very different epistemic status: directly observed events, inferred facts, subjective beliefs about others' motivations, and speculative hypotheses. Treating "Bob lives in Melbourne" (inferred from one offhand comment) the same as "Bob likes thrash metal" (stated directly, confirmed repeatedly) leads to brittle behavior.

Research validates this concern. The Generative Agents architecture assigns a 1-10 importance score at creation time; in ablation studies, importance scoring was critical for filtering mundane observations from significant ones (Park et al., 2023). The CoALA survey (Sumers et al., 2024) found that the most capable agent systems distinguish episodic from semantic memory with different retrieval strategies for each. LLM theory-of-mind research (PNAS 2024) shows ~25% error rate on false-belief tasks, meaning agent inferences about others' mental states will often be wrong — the system must track this uncertainty.

**Solution:** Add epistemic metadata to every memory.

- `importance` — integer 1-10, assigned at extraction time. Measures how significant this memory is for future interactions. A DJ hearing "Bob said hello" (importance: 2) versus "Bob announced he's moving to London" (importance: 8). Used as a retrieval scoring signal and as a distillation trigger (see section F).

- `confidence` — *[foundation]* float 0.0-1.0. How certain the agent is that this memory is accurate. Directly observed facts get high confidence. Inferences from context get medium. Hypotheses about motivations get low. The extraction prompt assigns this. Retrieval can use it to weight results; prompt rendering can annotate low-confidence memories so the LLM knows to treat them tentatively.

- `epistemic_type` — *[foundation]* how the agent came to believe this: `observed` (directly witnessed), `stated` (told by the subject), `inferred` (derived from evidence), `hypothesized` (speculative theory about motivations or beliefs), `decided` (declared true by authority), `reported` (learned from a third party). This classification determines revision behavior: a `decided` memory persists until explicitly superseded; an `inferred` memory is open to revision by new evidence; a `hypothesized` memory actively invites contradiction.

**Examples across app types:**

- **Chatbot:** "User prefers PostgreSQL over MongoDB" — `stated`, confidence 0.95, importance 7. "User seems frustrated with the deployment process" — `inferred`, confidence 0.6, importance 4.
- **DJ:** "Bob loves Metallica, seen them live eight times" — `stated`, confidence 0.95, importance 8. "Bob probably lives near Melbourne" — `inferred`, confidence 0.5, importance 5.
- **DM:** "P2 charged into combat without scouting" — `observed`, confidence 1.0, importance 6. "P2 is probably impatient rather than strategically aggressive — they don't scout even when they have time" — `hypothesized`, confidence 0.4, importance 7.
- **Customer support:** "Customer's subscription expires next month" — `stated`, confidence 0.95, importance 7. "Customer may be evaluating competitors based on their questions about export formats" — `hypothesized`, confidence 0.3, importance 8.

### C. Memory type taxonomy

**Problem:** The v1 types (preference, fact, decision, topical_context, relational_context) were designed for a flat, single-entity model. With tiers, entity scoping, and epistemic metadata, some types are redundant and important categories are missing. `topical_context` is subsumed by entity scoping (`about_type=topic` + `fact`). `relational_context` is subsumed by `about_type=system` + `decision`. Meanwhile, there's no structural distinction between facts the agent learned and actions the agent took, no way to mark a memory as an uncertain belief versus an established fact, and no representation of forward-looking commitments.

**Solution:** Revise the type taxonomy to cover the distinct epistemic categories an agent needs.

- **`preference`** — Likes, dislikes, tastes, working style. Retained from v1. *"Bob prefers deep cuts over greatest hits." "This customer always wants email follow-ups, never phone calls."*

- **`fact`** — Biographical, situational, or world-state facts. Retained from v1, narrowed. *"Bob lives near Melbourne." "The campaign is in Act 2." "Customer has been on the Enterprise plan since 2024."*

- **`decision`** — A declared choice, established by authority rather than inferred. Retained from v1, with explicit "persists through distillation" semantics. Decisions are never consumed by distillation — they stay until explicitly superseded. *"The protagonist is a retired detective." "We're using structured speaking order for this book club." "The refund policy exception was approved by the team lead."*

- **`action`** — Something the memory owner did. New. Structurally distinct so the retrieval layer can answer "what have I done with this entity?" and prevent the agent from repeating itself. Research (LARP, Shao et al., 2023) found action memory more important for persona consistency than detailed persona descriptions. *"I introduced Bob and Alice over shared Metallica fandom." "I recommended Snarky Puppy to Charlie." "I escalated this ticket to Tier 2 support." "I set a trap at the north bridge for the players."*

- **`belief`** — Theory of mind, motivational inference, hypotheses about others' mental states or future behavior. New. Distinguished from `fact` by lower default confidence and different revision behavior — beliefs are actively updated when contradicting evidence appears. *"Brian positions others as shields — likely strategic self-preservation." "This customer is probably evaluating competitors." "Alice seems to value literary craft over plot."*

- **`commitment`** — *[foundation]* Forward-looking promise, plan, or intention. New. Has a lifecycle: active → fulfilled / broken / expired. Commitments should surface proactively when the relevant context arises. *"I promised to find that Aphex Twin deep cut for Charlie." "I told the customer I'd follow up by Friday." "I plan to connect Bob and Alice next time they're both online." "I need to resolve the plot thread about the missing artifact."*

**Dropped:** `topical_context` (subsumed by entity scoping), `relational_context` (subsumed by entity scoping + decision type).

**Migration:** Existing `topical_context` memories map to `fact`. Existing `relational_context` memories map to `fact` or `decision` depending on content — the migration can default to `fact` with a backfill pass.

### D. Entity resolution

**Problem:** The same entity appears under different identifiers on different platforms or in different contexts. "discord:123456789" and "mastodon:@alice@social.example" are the same person. "The novel project" and "Chicago Noir" are the same project.

**Solution:** A lightweight identity resolution registry within 3tears.

- Maps multiple platform-qualified source IDs to a single canonical entity ID
- When storing a memory, the consuming app can pass a raw source ID; 3tears resolves it to canonical form
- When retrieving, resolution ensures that memories about "discord:123" and "mastodon:@alice" both surface when either identifier is queried
- The consuming app registers identity links (e.g., "these identifiers refer to the same entity"); 3tears handles the fan-out

**Type-agnostic resolution:** The registry is not limited to person-type entities. Any entity type can have aliases. A project known as "the novel" in casual conversation and "Chicago Noir" in the writing tool needs the same resolution as a person known by different handles. The registry stores `(source_qualifier, source_id) → canonical_id` mappings without interpreting what kind of entity it is.

**Retroactive linking:** When identities are linked *after* memories have been stored, existing memories with the old identifier must be updated. 3tears provides a `link_identities(canonical_id, source_ids)` operation that: (1) registers the mapping, (2) updates `about_id` and entries in `mentioned_entity_ids` on existing memories that reference any of the source IDs. This is a batch update bounded by the number of affected memories. Consuming apps should link identities early to minimize the batch size.

**What stays in consuming apps:** The decision of *when* to link identities. A DJ bot might auto-link when a user says "I'm @alice on Mastodon." A customer support system might link via CRM lookup. 3tears provides the registry; apps provide the data.

### E. Flexible provenance

**Problem:** Provenance is modeled as `conversation_id` + `message_id_source`, assuming a chat-app context. Memories can come from many sources: Discord messages, Mastodon posts, RSS feeds, tool results, support tickets, reflection turns, orchestrator instructions.

**Solution:** Replace the fixed provenance fields with a generic model:

- `source_type` — app-defined string identifying the source category (e.g., "discord_message", "support_ticket", "rss_item", "tool_result", "reflection", "orchestrator")
- `source_id` — opaque string referencing the specific source item
- `source_context` — optional structured metadata (channel name, ticket number, feed title — whatever the app considers useful for audit)

**Design principle:** Provenance is audit/debug metadata with one exception: `epistemic_type=reported` memories (learned from third parties rather than direct observation) may receive lower confidence during extraction based on source type. A DJ learning about Bob's taste directly from Bob is more reliable than hearing about it from Charlie. The extraction pipeline can factor `source_type` into the `confidence` assignment. Beyond this, provenance is stored and queryable but not factored into relevance scoring.

### F. Memory tiers and distillation

**Problem:** Memories are flat — every memory is a 1-2 sentence fact at the same level of abstraction. Over weeks of interaction, a DJ accumulates hundreds of individual observations ("Bob liked this song", "Bob liked that song") with no mechanism to synthesize them into patterns ("Bob is a thrash metal fan") or principles ("Metallica fans tend to also enjoy Pantera"). Without synthesis, memory becomes noise. Research confirms this: LONGMEM-Bench (2024) showed hierarchical memory with consolidation outperforms flat stores by 15-25% at long horizons.

**Solution:** Introduce memory tiers and a pluggable distillation engine.

#### Tiers

Every memory belongs to a tier representing its level of abstraction:

- **Observation** — a specific fact from a specific moment. "Bob requested Aphex Twin SAW II and loved it." "Customer called about billing discrepancy, reference #4821." Raw material for pattern recognition.
- **Pattern** — a recurring theme synthesized from multiple observations. "Bob is a deep-cuts ambient electronic fan." "This customer escalates to manager requests when response time exceeds 24 hours." More durable and compact than individual observations.
- **Principle** — a generalized insight that transcends specific entities or situations. "Listeners who prefer deep cuts respond poorly to greatest-hits playlists." "Customers who mention competitors by name are 3x more likely to churn within 30 days." Transfers across contexts.

This hierarchy mirrors the Generative Agents' observation → reflection structure (Park et al., 2023), but with two key improvements: explicit tier management with CONSUME operations that keep the corpus manageable, and cross-entity distillation that the single-agent Generative Agents architecture cannot perform.

#### Distillation triggers

Distillation is not continuous. It fires at specific moments, informed by the Generative Agents' validated importance-sum trigger and Letta's sleep-time compute model.

**Within-entity distillation** triggers when:
1. An entity's accumulated importance score (sum of `importance` values on observations since last distillation) exceeds a configurable threshold (default: 50). This naturally fires more often for entities generating significant interactions and less often for entities with only mundane observations.
2. A session boundary occurs (end of conversation, end of interaction period) and the entity has at least N new observations since last distillation (default: 5).
3. The consuming app explicitly requests it (e.g., during a scheduled reflection turn).

**Cross-entity distillation** triggers on a background schedule (default: daily or on explicit app request), matching Letta's sleep-time compute model. The detection algorithm:
1. Retrieve all patterns produced since the last cross-entity run.
2. Cluster them by embedding similarity.
3. For clusters spanning 3+ distinct `about_id` values, run the cross-entity distillation prompt to produce `about=null` principles.

Both triggers are configurable per-owner or per-preset. A DJ in a busy channel might trigger within-entity distillation after every session; a personal assistant might trigger it weekly.

#### Non-destructive distillation operations

The LLM produces operations for each observation under review:

- **CONSUME** — observation is fully captured in a new or existing pattern. The observation is **soft-deleted** (`is_deleted=true`) and linked to the pattern via `consumed_by_id`. The consuming pattern records the observation in its `derived_from_ids`. The observation is excluded from future retrieval but remains in the database for audit, re-processing, and deletion tracing.
- **REFINE** — extract the generalizable part into a pattern; rewrite the observation to keep only the operational/specific remainder. Both the pattern and the rewritten observation are preserved.
- **KEEP** — observation is too recent, too unique, or too important to distill. Leave it unchanged.

Patterns may further distill into principles via the same process.

**Why non-destructive CONSUME:** Research on memory drift (2024-2025) documents that each LLM processing step introduces ~5-10% error at key decision points. The extraction pipeline already runs 3 LLM calls; distillation adds another. If CONSUME hard-deletes the source observations, the evidence for the pattern is permanently destroyed after 4+ error-prone processing steps. With soft-delete and provenance links:

- Distillation quality is auditable ("which observations produced this pattern? were they accurately captured?")
- Incorrect patterns can be reversed by restoring consumed observations and re-running distillation
- GDPR/privacy deletion can trace which individual's data contributed to which patterns
- The memory health diagnostic can compare patterns against their source observations to detect shallow distillation

The storage cost is negligible — the rows already exist; we just keep them marked as consumed instead of hard-deleting.

#### Type-aware distillation

Not all memory types distill the same way:

- **Observations** (type=preference, fact, action, belief) are the primary input to distillation. They are consumed as patterns form.
- **Decisions** persist and update-in-place. "The protagonist is a retired detective" is a decision — it shouldn't be consumed or generalized. It stays until explicitly superseded. The distillation engine skips `type_memory=decision` memories entirely.
- **Commitments** *[foundation]* are not distilled. They have their own lifecycle (active → fulfilled/broken/expired) managed separately.
- **Beliefs** distill differently from facts. A cluster of behavioral observations about an entity distills into a belief-type pattern with theory-of-mind language, not a fact-type pattern with behavioral summary. The distillation prompt handles this distinction.

#### Theory of mind in distillation

Distillation that only summarizes behavior ("Brian frequently suggests risky actions") is shallow. Useful distillation asks *why*: "Brian consistently encourages persona players to take frontline risks while positioning himself safely — likely strategic self-preservation." The second version is far more useful because it gives the owner a predictive model, not just a behavioral log.

Research on LLM theory of mind (PNAS 2024) shows GPT-4-class models achieve ~75% accuracy on standard false-belief tasks and adult-level performance on higher-order recursive reasoning (up to 6th order). This is good enough to be useful but unreliable enough that the results must be tracked as hypotheses, not facts.

Default distillation prompts guide the LLM to:

1. **Infer motivations:** "Why might this person behave this way? What does the pattern suggest about their goals, values, or strategy?"
2. **Classify epistemic status:** Theory-of-mind patterns are automatically assigned `type_memory=belief` and `confidence` reflecting the evidence strength.
3. **Make predictions testable:** "If this theory is correct, what behavior would we expect to see next?" This gives the system a basis for future revision.

Theory of mind also requires that the extraction pipeline captures beliefs and inferences, not just facts. When a DM reflects "I think Brian is using P1 as a shield," that belief has different epistemic status than "Brian told P1 to charge the dragon" — but both are valuable memories. The extraction worthiness gate must not reject subjective interpretations about others' motivations; these are often the most durable and useful memories an owner forms.

Theories revise naturally through distillation. When new evidence contradicts an existing belief-type pattern (Brian sacrifices himself to protect P1 after a history of risk-avoidance), distillation updates the theory: "Brian's earlier behavior may have been strategic optimization rather than disregard — he proved willing to sacrifice when it mattered." The REFINE operation handles this without special machinery, and the `derived_from_ids` on the refined pattern links back to both the original evidence and the contradicting evidence.

#### Pluggable prompts and strategies

3tears provides the distillation engine (scheduling, memory scanning, trigger logic, tier management). Consuming apps provide:

- Distillation prompt templates (how should observations be synthesized in this domain?)
- Tier names and semantics (if the defaults don't fit)
- Trigger threshold overrides (importance-sum threshold, minimum observation count, background schedule)

This separation keeps the engine reusable while letting each app tune the cognitive style of distillation.

**DistillationStrategy protocol:**

```python
class DistillationStrategy(Protocol):
    def within_entity_prompt(self, owner_context: str, entity_context: str) -> str: ...
    def cross_entity_prompt(self, owner_context: str, theme_context: str) -> str: ...
    def importance_threshold(self) -> int: ...          # default: 50
    def min_observations(self) -> int: ...              # default: 5
    def background_schedule(self) -> timedelta | None: ...  # default: 24h
    def tier_semantics(self) -> dict[str, str]: ...     # default: observation/pattern/principle
```

A default strategy ships with 3tears. Apps that don't provide a custom strategy get default distillation behavior.

### G. Relationship modeling

**Problem:** Relationships are a core concern across app types — a DJ connecting listeners, a DM tracking inter-player dynamics, a support agent noting account relationships — but they don't fit cleanly into single-entity-scoped memory. Research at scale (Project Sid, 2024) found that agents needed explicit structured relationship state alongside narrative memory.

**Solution:** In v2, relationship awareness is achieved through the memory model rather than a separate data structure. The `about_id` + `mentioned_entity_ids` combination enables relationship memories to surface for all involved entities. Formal relationship modeling (dedicated relationship table with typed edges, trust scores, and strength metrics) is a future capability that the v2 schema supports but does not implement.

**How relationship memories work in v2:**

A DJ stores "Bob and Alice share deep Metallica fandom — I introduced them" as:
- `about_id=Bob`, `mentioned_entity_ids=[Alice]`, `type_memory=action`

When either Bob or Alice is in `relevant_entities`, this memory surfaces via entity boosting on either field. When both are present, it surfaces with even stronger entity boost (matching on both `about_id` and `mentioned_entity_ids`).

Distillation can synthesize relationship-oriented patterns from observations involving the same entity pair. Individual observations ("I played Metallica for Bob and Alice", "Bob and Alice compared favorite Metallica albums", "Alice told Bob about a Metallica tribute band") distill into a pattern: "Bob and Alice have a strong shared connection around Metallica that I've actively cultivated." The pattern gets `about_id=Bob, mentioned_entity_ids=[Alice]`.

**Retrieval support:** The retrieval layer matches `relevant_entities` against both `about_id` and `mentioned_entity_ids`. A memory scores as an entity match if any relevant entity appears in either field. This is implemented as a GIN index on `mentioned_entity_ids` and a simple OR in the entity-boost scoring.

**What this doesn't handle (future direction):** Typed relationship edges (friend/rival/collaborator), explicit trust scores, relationship strength metrics, and efficient "show me all of Bob's relationships" queries. These would be served by a dedicated `relationships` table populated by distillation when it detects recurring entity-pair patterns. The v2 schema supports this evolution because `mentioned_entity_ids` already captures the entity pairs, and distillation can be extended to produce relationship records.

### H. Temporal validity

**Problem:** When facts change (Bob moves from Melbourne to Sydney, a customer changes plans, a character dies in the campaign), the current system overwrites the old memory with UPDATE. This destroys the history — "where did Bob used to live?" and "when did Bob move?" become unanswerable. Research confirms this matters: LoCoMo (2024) found temporal ordering to be the hardest task category across all evaluated memory systems. StreamBench (2024) showed most vector-retrieval systems fail when receiving contradictory information over time.

**Solution:** Add temporal validity tracking.

- `valid_until` — *[foundation]* timestamptz, nullable. Null means "currently believed true." When a fact is superseded, the old memory gets `valid_until` set to the time of supersession. The new fact is stored as a new memory. Both coexist in the database; retrieval defaults to filtering for `valid_until IS NULL` (current facts) but can optionally include historical facts.

This preserves history without the operational weight of a full bi-temporal model (Zep's four-timestamp approach). A DM can query "what was true about P2's character at the start of Act 2?" A customer support agent can see "this customer was on the Starter plan before upgrading to Enterprise." The DJ knows Bob used to be in Melbourne and recently moved to Sydney.

**How supersession works:** The resolution pipeline's UPDATE action is reinterpreted. Instead of overwriting the existing memory's content, UPDATE now: (1) sets `valid_until` on the existing memory, (2) creates a new memory with the updated content and `valid_until=NULL`. The old memory remains queryable for historical context. This is transparent to the resolution prompt — the LLM still produces UPDATE with memory_id and new content; the engine handles the temporal bookkeeping.

### I. Entity-aware retrieval

**Problem:** The retriever accepts a single query string and returns scored memories filtered by owner. It has no concept of "who is present" or "what are we working on" as retrieval signals, it lacks importance-based scoring, and its output format doesn't support pattern discovery across entities.

**Solution:** Extend the retriever with richer query context, multi-signal scoring, and two retrieval modes.

#### Multi-signal scoring

The retriever scores memories using four signals, extending the Generative Agents' validated three-factor model (Park et al., 2023):

1. **Semantic similarity** — cosine distance between query embedding and memory embedding. The primary relevance signal.
2. **Keyword match** — PostgreSQL FTS score. Catches exact matches that embeddings may miss, especially for proper nouns and domain terminology.
3. **Recency** — exponential decay based on `date_last_accessed` (not just `date_created`). Memories that keep being retrieved stay fresh; unused memories naturally decay. This matches the Generative Agents' access-based recency, which produces a natural spaced-repetition effect. *[Change from v1: v1 used `date_created` only.]*
4. **Importance** — the memory's `importance` score, normalized to [0, 1]. High-importance memories surface even when semantic and recency scores are marginal.

Default signal weights: `semantic=0.40, keyword=0.10, recency=0.20, importance=0.15, entity_boost=0.15`. These are configurable per-preset and per-app.

#### Entity boosting

Memories where `about_id` or any entry in `mentioned_entity_ids` matches a relevant entity receive an additive entity-boost score. This is the primary mechanism for "who is present" and "what are we working on."

Entity boosting operates independently of semantic similarity. A memory about Bob's Melbourne location surfaces when Bob is present, even if the current query is about music — because the memory is *about a relevant entity*, not because it's semantically similar to the query.

#### Budget allocation

To prevent general principles from crowding out entity-specific memories (or vice versa), the retrieval budget is partitioned:

- **Entity-specific budget** (default: 70% of `context_budget`) — memories where `about_id` or `mentioned_entity_ids` matches a relevant entity.
- **General budget** (default: 30% of `context_budget`) — memories where `about_id IS NULL` (general principles and domain wisdom).

Within each partition, memories are ranked by the multi-signal score and selected via MMR. The partition percentages are configurable.

#### Retrieval context

The retriever accepts a structured query context:

- `owner_id` — whose memories to search (required, as today)
- `query` — text for semantic and keyword search (as today)
- `relevant_entities` — list of entity IDs to boost
- `tier_weights` — optional per-tier scoring weights (e.g., boost principles during reflection, boost observations during active interaction)
- `type_filter` — *[foundation]* optional list of memory types to include/exclude (e.g., retrieve only actions and commitments)
- `include_historical` — *[foundation]* whether to include memories with non-null `valid_until` (default: false)

#### Two retrieval modes

**Conversational retrieval** — used during active interactions. Optimized for token efficiency and diversity.

- Entity-boosted scoring with budget allocation
- MMR reranking for diversity within each budget partition
- Tight budget: returns a manageable number of results for prompt inclusion
- Entity-grouped output: results organized by `about_id`, so the LLM sees "what I know about each participant" as a coherent picture

**Topic scan** — used during reflection, distillation, and cross-entity analysis. Optimized for completeness.

- Content-first: no entity boosting, pure semantic/keyword/importance matching
- No MMR: returns all memories above threshold
- Entity-grouped output: essential for identifying which entities share a theme
- Larger budget: comprehensive rather than token-efficient

Both modes share the same underlying search infrastructure. The difference is in post-processing: whether entity boosting and MMR run, and how results are formatted.

**Why entity-grouped output matters:** Flat lists of memories hide cross-entity patterns. When the retriever returns memories about Bob and Alice interleaved by score, the LLM may not notice they share Metallica fandom. When results are grouped by entity, the connection is visually obvious. This grouping aids the LLM's reasoning without requiring the memory system to do inference.

#### Extraction gate configuration

The extraction pipeline's heuristic gates (message length, response length, turn count, rate limiting) must be configurable per owner or per use case. A quiet facilitator persona that listens but rarely speaks will never pass the "assistant response length >= 100 chars" gate under the current defaults. A customer support bot processing ticket updates has no concept of "turn count." The gates themselves are sound; they need per-owner configuration rather than global defaults.

---

## Scenarios

These scenarios illustrate how the components work together across different app types. They are not exhaustive but cover key interaction patterns that exercise different parts of the architecture.

### Chatbot: evolving user understanding

A personal assistant chatbot serves one human over months. Early interactions produce observations: "User prefers PostgreSQL over MongoDB — cites JSONB, mature tooling, pgvector" (preference, stated, importance 7), "User is a senior backend engineer at Acme Corp" (fact, stated, importance 6), "User has a daughter named Lily starting kindergarten fall 2026" (fact, stated, importance 5).

Over weeks, observations about the user's work patterns distill into patterns: "User is building a self-hosted LLM orchestrator (MetaLLM) with Python/FastAPI, LangGraph, pgvector, multi-provider support." The pattern is more durable and compact than the dozen individual observations about code changes, architecture decisions, and technology choices.

The assistant stores its own actions too: "I suggested switching from OpenAI embeddings to VoyageAI and the user agreed" (action, importance 6). This prevents the assistant from re-suggesting the same migration. Over time, the action memories distill into a relationship pattern: "I've helped the user through several major architectural decisions; they trust my recommendations on infrastructure but prefer to make their own API design choices."

When the user starts a new project months later, entity-specific patterns about their preferences and working style surface immediately. General principles about software architecture that emerged from their earlier work also surface via the general budget. The assistant is a better collaborator because of accumulated experience.

### DJ: participant joins with known history

A DJ persona is in a channel with humans Albert and Betty. Charlie joins. Charlie's discord ID resolves via entity resolution to a canonical person ID. The retriever runs with `relevant_entities=[Albert, Betty, Charlie]`. Memories about Charlie from previous sessions — different channels, weeks ago — surface via entity boosting. The DJ greets Charlie with awareness of their taste.

Charlie's memories include: "Charlie loves jazz fusion, especially Snarky Puppy" (preference, stated, confidence 0.95, importance 7) and "I recommended Weather Report to Charlie and they loved it" (action, observed, importance 6). The DJ builds on the established relationship rather than starting from scratch.

### DJ: connecting listeners with shared taste

Bob says he loves Metallica. Weeks later, Alice says something similar in a separate session. During Alice's session, semantic search surfaces Bob's Metallica memory even though Bob isn't present. The DJ notes the overlap. Later, when both are present, entity-grouped retrieval shows:

> **About Bob:** loves Metallica, seen them live eight times (preference, importance 8)
> **About Alice:** deep Metallica fan, prefers deep cuts over radio hits (preference, importance 7)

The connection is visually obvious in grouped output. The DJ introduces them.

The DJ stores: "I introduced Bob and Alice over shared Metallica fandom" — `about_id=Bob, mentioned_entity_ids=[Alice], type=action, importance 7`. Next time both are present, this memory surfaces for *either* Bob or Alice via entity matching on both fields. The DJ doesn't repeat the introduction — it builds on the established connection.

### DJ: cross-entity principles

During a background distillation run, the engine notices patterns spanning 4+ listeners: "Listener A loves Metallica and Pantera", "Listener B loves Metallica and Megadeth", "Listener C liked Metallica but was unimpressed by Queensryche." Cross-entity distillation produces an `about=null` principle: "Metallica fans tend to enjoy other thrash metal (Pantera, Megadeth) but are less interested in progressive metal (Queensryche)." This principle surfaces in future interactions with any Metallica fan, including new listeners the DJ has never met.

### DM: theory of mind across a campaign

A DM agent runs a D&D campaign with four players: two humans (Brian and Cass) and two AI personas (P1 and P3). Early sessions produce observations:

- "Brian told P1 to charge the dragon while Brian stayed behind cover" — about Brian, mentioned [P1], observed, importance 6
- "Brian suggested P1 scout the cave alone" — about Brian, mentioned [P1], observed, importance 5
- "Brian volunteered to guard the exit while others entered the dungeon" — about Brian, observed, importance 5
- "Cass asked whether the village children were safe before agreeing to leave" — about Cass, observed, importance 6

Within-entity distillation synthesizes the Brian observations into a belief-type pattern: "Brian consistently positions other players — especially P1 — in high-risk roles while keeping himself safe. Likely strategic self-preservation, using expendable allies as a buffer." Confidence 0.5 (hypothesis), importance 8. The `derived_from_ids` links back to all three source observations (which are soft-deleted but preserved).

Later, Brian sacrifices his character to save Cass from a collapsing tunnel. The extraction pipeline captures this: "Brian sacrificed his character to save Cass in the collapsing tunnel" — about Brian, mentioned [Cass], observed, confidence 1.0, importance 9. On the next distillation trigger, this contradicts the existing theory. REFINE updates: "Brian's earlier self-preserving behavior may have been strategic resource management rather than cowardice — he proved willing to sacrifice when a valued ally was in genuine danger. Brian prioritizes meaningful stakes over routine risk." The refined pattern's `derived_from_ids` includes both the original evidence and the contradicting observation.

The DM uses this updated theory to design future encounters that create meaningful moral dilemmas for Brian rather than routine combat.

For Cass, distillation produces: "Cass consistently prioritizes civilian safety and emotional stakes over tactical advantage. Appeals to protecting innocents are the strongest motivator for Cass." The DM designs hooks accordingly.

Cross-entity distillation across all four players might produce a principle: "Players engage more deeply with encounters that have moral stakes than with encounters that are purely tactical."

### DM: campaign lifecycle and knowledge transfer

Hundreds of session-specific observations from Campaign 1 are gradually consumed into patterns and principles. When the DM starts Campaign 2 with entirely different players, the Campaign 1 player-specific memories don't surface (different `about_id` values, not in `relevant_entities`). But general principles do:

- "Design encounters with initial barriers that require planning before combat."
- "Players engage more when NPCs have personal stakes in the outcome."
- "Mixing puzzle and combat elements in the same encounter sustains engagement better than sequential encounters."

The DM is better at running Campaign 2 because of what it learned in Campaign 1. If a player from Campaign 1 joins Campaign 2, their specific behavioral patterns surface immediately via entity boosting.

Campaign-specific world state ("P3's character is named Elara") stays as a decision-type memory. Decisions don't decay through recency, but they also don't surface without entity boosting — and with no Campaign 1 entities in the relevant set, they effectively become dormant without needing explicit "campaign scope" tagging.

### Customer support: escalation patterns

A support agent persona handles tickets across hundreds of customers. Individual observations accumulate:

- "Customer 4821 asked about data export formats twice in one week" — fact, importance 5
- "Customer 4821 mentioned they were 'evaluating options' when asking about contract terms" — fact, importance 7
- "Customer 4821's tone shifted from friendly to terse after the billing discrepancy" — belief (inferred), confidence 0.6, importance 6

Distillation synthesizes: "Customer 4821 is likely evaluating competitors — repeated export questions, contract inquiries, and tone shift suggest churn risk." This is a belief-type pattern, confidence 0.4, importance 9. The low confidence reflects that it's a hypothesis; the high importance reflects the business impact if true.

The agent also stores actions: "I escalated customer 4821's billing discrepancy to Tier 2" (action, importance 6) and commitments: "I told customer 4821 I'd follow up on the billing resolution by Friday" (commitment, importance 7). When the customer contacts support again, entity-boosted retrieval surfaces the escalation history, the churn-risk assessment, and the outstanding commitment — preventing the common support failure of asking the customer to re-explain their situation.

Cross-entity distillation produces general principles: "Customers who ask about data export formats and contract terms in the same week are 3x more likely to churn within 30 days." "When a customer mentions a competitor by name, immediately offer a proactive account review." These principles surface for any customer showing similar patterns.

### Customer support: multi-entity relationship

Customer 4821 mentions that customer 5102 referred them. The extraction captures: "Customer 5102 referred customer 4821" — `about_id=4821, mentioned_entity_ids=[5102], type=fact, importance 6`. When either customer contacts support, the referral relationship surfaces. If customer 5102 later has a bad experience, the agent is aware that 4821's loyalty may also be affected.

### Quiet facilitator: proactive memory surfacing

A facilitator persona observes a channel where humans discuss daily topics. It rarely speaks unless directly addressed or a highly relevant memory is available. Karen says to Brooks: "I don't remember when you were in Spain."

The retriever runs with `relevant_entities=[Karen, Brooks]` and query text about Spain. A memory from weeks ago — about Brooks: "Brooks traveled to Spain in March 2024" — surfaces with high relevance (entity boost plus strong semantic match, importance 6). The facilitator sees a high-confidence match and offers: "Brooks mentioned being in Spain last March."

This works because: (1) the facilitator's extraction pipeline captures facts from conversations it observes, even when it doesn't respond — extraction gates are configured for this persona with the response-length gate disabled; (2) entity-boosted retrieval surfaces the Spain memory because Brooks is present; (3) the consuming app's behavioral logic interprets the combined retrieval score and importance as grounds for proactive engagement.

---

## Agent autonomy modes

The same memory architecture supports a spectrum from passive assistants to fully autonomous agents. The differences are not in the schema — they're in which capabilities are activated and how aggressively. This section describes how different agent types use the memory system, informing the preset configurations.

### Passive assistants

Single-user chatbots, copilots, journaling apps. One owner, one primary entity (the human).

**Memory profile:**
- Mostly observations and derived facts about the user. Some decisions. Few actions (the assistant responds to requests more than it initiates).
- Little theory of mind — the human is the primary entity, and inferring their motivations is less useful than recording their stated preferences.
- No relationship modeling (one relationship: agent ↔ human).
- Simple retrieval: semantic + recency + importance, low entity complexity. Entity boosting is unnecessary when there's only one entity.
- Distillation focused on preference synthesis and goal tracking. Within-entity only; cross-entity distillation is irrelevant.

**What matters most:** Accurate preference capture, temporal validity (preferences change), and avoiding repetition (remembering what was already discussed or recommended).

### Facilitators and observers

Channel observers, meeting note-takers, quiet helpers. One owner, many entities, low initiative.

**Memory profile:**
- Observations about multiple entities, captured passively from observed conversations.
- Basic relationship awareness — who knows whom, shared interests.
- Action memory for the facilitator's own interventions (to avoid repeating them).
- Extraction gates relaxed: no response-length requirement (the facilitator may not speak), no turn-count requirement (it observes continuously).
- Proactive retrieval: high-confidence entity-boosted matches may trigger engagement.
- Limited distillation: within-entity patterns form naturally, but the facilitator's corpus is smaller than active participants'.

**What matters most:** Extraction from observed conversations without participation. High precision on entity identification (the facilitator must correctly attribute "who said what"). Proactive surfacing of high-relevance memories at appropriate moments.

### Social agents

DJ bots, book club moderators, community facilitators. One or more owners, many entities, high initiative, active relationship building.

**Memory profile:**
- All observation types including actions, beliefs, and commitments.
- Active theory of mind — modeling each participant's preferences, motivations, and relationships.
- `mentioned_entity_ids` heavily used for multi-entity relationship memories.
- Cross-entity pattern detection (shared tastes, group dynamics).
- Distillation running at session boundaries and on background schedule.
- Both retrieval modes active: conversational during interaction, topic scan during reflection.

**What matters most:** Theory of mind, action memory (don't repeat introductions, build on previous interactions), cross-entity distillation (discover connections between participants), and temporal validity (tastes evolve).

### Autonomous creators

DM agents, author agents, planners, project managers. One owner, many entities, high initiative, maintains a world model, makes creative or strategic decisions.

**Memory profile:**
- Everything the social agent needs, plus:
- Decision memory with strong persistence (world-building and project decisions must never be forgotten or consumed).
- Strategic belief patterns: "encounter design principles," "this player type responds well to X."
- Commitment tracking: plot threads, promises to players, unresolved questions.
- Action memory for the DM's own choices: "I placed the artifact in the northern ruins," "I introduced the BBEG via dream sequence."
- Cross-entity distillation producing tactical principles.

**What matters most:** Decision persistence, theory of mind about players/participants, strategic memory (what approaches work), commitment tracking (open plot threads, promises), and campaign/project lifecycle (knowledge that transfers to the next campaign/project).

### Customer support agents

One owner, very many entities (hundreds to thousands of customers), medium initiative, relationship memory with business context.

**Memory profile:**
- High volume of per-entity observations, most at low-to-medium importance.
- Actions and commitments are critical: "what did I promise this customer?" and "what actions have I already taken?"
- Theory of mind focused on customer satisfaction and churn risk rather than personality modeling.
- Cross-entity distillation is high-value: patterns across customers (escalation triggers, churn indicators) directly improve service quality.
- Temporal validity matters — customer plans, contact preferences, and account status change frequently.

**What matters most:** Commitment tracking (outstanding promises), action memory (don't ask the customer to repeat themselves), cross-entity principles (escalation and churn patterns), and temporal validity (customer status changes).

---

## Boundaries: what lives where

### 3tears owns

- **Memory storage and retrieval** — three-tier caching, embedding, multi-signal hybrid scoring (semantic + keyword + recency + importance), entity-boosted and topic-scan retrieval modes, entity-grouped output with budget allocation
- **Memory model** — owner, about (generalized entities with mentioned_entity_ids), provenance, tier, type, epistemic metadata (importance, confidence, epistemic_type), temporal validity
- **Entity resolution** — type-agnostic registry mapping platform IDs to canonical IDs, with retroactive linking
- **Extraction pipeline** — gated, multi-stage extraction with entity identification, action capture, importance and confidence assignment, belief acceptance, configurable per-owner gates
- **Distillation engine** — importance-sum and session-boundary triggers, within-entity and cross-entity pattern synthesis, non-destructive CONSUME with provenance, theory-of-mind-oriented defaults, pluggable prompts and strategies
- **Memory ledger** — within-conversation tracking of surfaced items to prevent redundancy

### Consuming apps own

- **Entity registry** — what entities exist and what they mean. 3tears stores `about_id`/`about_type`; the app decides what entities to create and how to model them.
- **Relevance determination** — which entities are relevant for a given interaction. The app passes relevant entity IDs to the retriever; 3tears boosts them.
- **Identity linking** — deciding *when* to link platform identities. The app registers links with the resolution registry; 3tears resolves and retroactively updates.
- **Behavioral rules** — "speak only when directly addressed or high-confidence memory available", etc. 3tears provides scored retrieval with importance; the app interprets scores and decides actions.
- **Prompt formatting** — how retrieved memories are rendered into LLM prompts. 3tears provides entity-grouped structured output with epistemic metadata; the app formats it.
- **Distillation prompts** — domain-specific guidance for how observations should be synthesized. The app provides prompt templates; 3tears runs the engine.
- **Multi-owner reconciliation** — if an app needs a canonical shared artifact from multi-persona collaboration, it reconciles across owners at the application layer.
- **Cognitive/behavioral architecture** — persona traits, goals, scheduling, turn management, reflection triggers.
- **Privacy and access control** — who can read whose memories, sensitivity classification of entities, GDPR/deletion policy implementation. See the privacy section below.

### Why these boundaries

The boundary test is reuse: if every app rebuilding the same capability would produce essentially the same code, it belongs in 3tears. If apps would produce meaningfully different implementations reflecting their domain, it belongs in the app.

Retrieval scoring, distillation scheduling, tier management, entity resolution, and non-destructive CONSUME with provenance are the same everywhere — only the configuration differs. These belong in 3tears.

Entity semantics (what is a "project"?), behavioral rules (when should I speak?), privacy policy (what data can we retain?), and prompt formatting (how does my persona present memories?) are fundamentally domain-specific. These belong in consuming apps.

**Boundaries are not walls.** The fact that distillation prompts are "the app's concern" doesn't mean 3tears shrugs and ships an empty text field. See the next section.

---

## Privacy and deletion

Agent memory raises privacy concerns that the architecture must support, even though policy decisions belong to consuming apps.

### What 3tears provides

**Deletion by entity.** `delete_memories_about(entity_id)` finds all memories where `about_id = entity_id` OR `entity_id IN mentioned_entity_ids` and soft-deletes them. This handles the common "remove everything about this person" request.

**Provenance tracing through distillation.** Because CONSUME is non-destructive and `derived_from_ids` links patterns to their source observations, it's possible to trace which entity's observations contributed to a given pattern or principle. When deleting entity X's data, the system can identify patterns that were derived partly from X's observations. The consuming app decides whether to delete the entire pattern, re-run distillation without X's observations, or leave the pattern (which no longer attributes anything to X specifically).

**Cross-entity principle contamination.** This is the hardest case. An `about=null` principle like "Metallica fans tend to enjoy Pantera" may have been derived from observations about multiple entities. Deleting one entity's data shouldn't necessarily invalidate the principle. 3tears provides `contributing_entity_ids` on pattern and principle memories — a metadata field populated during distillation that records which `about_id` values contributed to the synthesis. Consuming apps use this to decide: if the deleted entity was one of many contributors, the principle likely stands; if they were the primary or sole contributor, the principle should be reviewed.

### What consuming apps own

- **When to delete.** GDPR requests, user opt-out, account closure, data retention policies. These are app-domain decisions.
- **Sensitivity classification.** "Daughter named Lily, starting kindergarten fall 2026" and "prefers PostgreSQL" have very different privacy implications. If the app needs sensitivity-aware handling, it classifies entities or memory types at the application layer.
- **Access control between owners.** In multi-owner systems, can persona A read persona B's memories about entity X? This is an app-level policy. 3tears's retrieval is scoped by `owner_id` — it never returns memories belonging to a different owner unless the app explicitly queries across owners.
- **Retention policies.** How long to keep soft-deleted memories, when to hard-delete, how long to retain consumed observations. 3tears provides the soft-delete and provenance infrastructure; apps define the schedule.

---

## Cost model

Memory operations consume LLM tokens and database resources. Understanding the cost profile is essential for production deployment.

### Extraction cost per turn

Each extraction runs up to 3 LLM calls:
1. **Worthiness gate** — ~300 input tokens (prompt + message preview), ~50 output tokens. Skipped if heuristic gates reject the turn.
2. **Candidate extraction** — ~800 input tokens (prompt + full message + response), ~200 output tokens. Produces 0-3 candidate memories.
3. **Resolution** — ~600 input tokens (prompt + candidates + similar existing memories), ~150 output tokens. Runs only if candidates were produced.

Typical cost per turn: 1,700 input + 400 output tokens when extraction fires. Most turns are rejected by heuristic gates (zero cost) or the worthiness gate (~350 tokens).

### Distillation cost per entity per run

Within-entity distillation reviews a cluster of observations:
1. **Cluster retrieval** — database query, no LLM cost.
2. **Distillation prompt** — ~1,500 input tokens (prompt + 5-15 observations + existing patterns), ~500 output tokens.
3. **Re-embedding** — embedding API call for each new or modified memory.

Typical cost per entity: 2,000 tokens per distillation run. With default triggers (importance-sum threshold 50, ~10 observations), this fires roughly every 5-15 interactions per entity.

### Cross-entity distillation cost

Depends on corpus size. The detection step (clustering recent patterns) is embedding-only. The synthesis step runs one LLM call per detected cluster: ~2,000 input tokens (prompt + clustered patterns across entities), ~300 output tokens. With daily background runs and a modest corpus (50 entities, 200 patterns), expect 2-5 clusters per run: ~12,000 tokens per daily cycle.

### Retrieval cost

No LLM cost. Database queries (pgvector similarity + FTS) plus optional re-ranking. At typical corpus sizes (<100K memories per owner), p95 retrieval latency is under 100ms.

### Scaling considerations

Per-owner memory systems are unlikely to hit pgvector limits. Even a power user generates ~10K memories over years. The cross-entity topic scan is the most likely performance concern: it queries across all entities without entity boosting or MMR filtering. For owners with very large corpora (customer support agents with thousands of customers), topic scan queries should be bounded by recency or tier (scan only recent patterns, not all observations).

---

## Helping apps succeed

The architecture is pluggable: apps provide prompts, configure gates, choose retrieval modes. This flexibility is necessary — a DJ bot and a customer support system need different memory behavior. But flexibility without guidance produces shallow memory systems. Most apps will ship with defaults, never revisit their prompts, and wonder why their personas feel flat. Theory of mind won't emerge from a distillation prompt that says "summarize recurring patterns." Action memories won't appear if extraction only looks for facts about users.

3tears must take responsibility for the quality of the whole system, not just the correctness of the architecture. This means opinionated defaults, diagnostic tools, and presets that encode hard-won knowledge about what makes memory work.

### Opinionated defaults

The default extraction and distillation prompts are not neutral starting points — they are 3tears's best answer for the general case. They encode:

- **Entity awareness.** Default extraction prompts guide the LLM to identify the primary entity and any mentioned entities for each memory. Without this, `about_id` stays null and entity-boosted retrieval finds nothing.
- **Action capture.** Default extraction prompts instruct the LLM to note the owner's own significant actions (introductions, recommendations, escalations, commitments), not just facts learned from others. Research (LARP, Shao et al., 2023) validates this as critical for persona consistency.
- **Importance scoring.** Default extraction prompts assign a 1-10 importance score to each candidate memory, matching the Generative Agents' validated approach (Park et al., 2023). Without this, retrieval can't distinguish significant memories from noise, and distillation has no principled trigger.
- **Epistemic classification.** Default extraction prompts classify each memory's epistemic type (observed, stated, inferred, hypothesized) and assign confidence. Without this, the system treats established facts and uncertain hypotheses identically.
- **Motivational inference.** Default distillation prompts guide the LLM to ask "why might this person behave this way?" when synthesizing behavioral patterns — producing theory of mind, not just behavioral logs.
- **Belief acceptance.** Default extraction worthiness gates do not reject subjective interpretations about others' motivations. "I think Brian is using me as a shield" is a valuable memory even though it's not an objective fact.
- **Tier-appropriate output.** Default distillation prompts produce genuinely synthetic patterns and principles, not concatenations of observations with slightly different wording.

An app that uses every default and never customizes a prompt should still get entity-aware extraction, action memory, importance scoring, and basic theory of mind. The defaults are the floor, not the ceiling.

### Memory profile presets

Rather than requiring every app to write prompts from scratch, 3tears ships preset profiles tuned for common scenarios. Each preset bundles extraction prompts, distillation prompts, gate configuration, retrieval defaults, and distillation trigger thresholds into a tested configuration.

- **assistant** — single-user chatbot or copilot. One owner, one human. Extraction tuned for preferences, facts, and decisions. Distillation tuned for goal tracking and user modeling. No cross-entity distillation needed. Importance threshold relaxed (the single entity generates all observations). Good default for apps getting started.
- **social** — multi-persona social interaction (the DJ pattern). Multiple owners, many entities. Extraction tuned for social observation, action memory, and relationship dynamics. Distillation tuned for theory of mind and cross-entity patterns. Extraction gates relaxed for observer/facilitator personas. `mentioned_entity_ids` heavily populated.
- **collaborative** — group work on shared artifacts (brainstorming, planning). Multiple owners, project/topic entities. Extraction tuned for decisions and commitments. Distillation preserves decisions rather than consuming them. Commitment lifecycle tracking emphasized.
- **game** — game master and players. Extraction tuned for world state, player behavior patterns, and strategic inference. Distillation tuned for cross-player patterns and theory of mind. Action memory emphasized for the DM. Belief-type patterns encouraged for player modeling.
- **support** — customer-facing agent handling many entities. High-volume observation extraction with moderate importance thresholds. Action and commitment tracking critical. Cross-entity distillation emphasized for discovering escalation and churn patterns. Temporal validity important (customer status changes frequently).

Apps pick a preset and customize from there. The preset handles the 80% case; the app overrides what's domain-specific. An app that outgrows its preset can export the preset's configuration and modify it directly.

### Memory health diagnostics

A diagnostic tool that analyzes an owner's actual memory corpus and reports on quality. This runs on-demand or periodically, not on every extraction. It examines:

- **Entity coverage.** What percentage of memories have non-null `about_id`? If most memories are unassociated, entity-boosted retrieval is ineffective. The diagnostic flags this and suggests checking `about_context` in extraction calls.
- **Mentioned entity usage.** What percentage of multi-entity memories populate `mentioned_entity_ids`? If most only set `about_id`, relationship memories are invisible to secondary entities.
- **Tier distribution.** What's the ratio of observations to patterns to principles? A healthy corpus has distillation producing patterns over time. If there are 500 observations and 0 patterns, distillation isn't running or its prompts aren't producing synthesis.
- **Importance distribution.** Are importance scores well-distributed, or are most memories clustered at 5? Flat distributions suggest the extraction prompt isn't differentiating significance.
- **Epistemic diversity.** Are beliefs and hypotheses present alongside facts? If all memories are `epistemic_type=stated` or `observed`, the system isn't capturing inferences or theory of mind.
- **Action memory presence.** Are any memories recording the owner's own actions? If not, the owner will repeat itself.
- **Distillation depth.** Are patterns genuinely synthetic, or just slightly reworded observations? The diagnostic samples patterns, retrieves their `derived_from_ids`, and compares content, flagging patterns that don't add abstraction.
- **Theory of mind indicators.** Do belief-type patterns include motivational language (intent, belief, strategy, motivation, goal) or only behavioral language (frequently, usually, tends to)?
- **Temporal health.** Are there memories that should have `valid_until` set but don't? (Multiple conflicting facts about the same entity and topic without temporal resolution.)
- **Retrieval simulation.** Given a sample query and set of relevant entities, what would the retriever actually return? Does entity boosting surface the right memories? Does the budget allocation balance entity-specific and general memories appropriately?

The diagnostic output is structured (JSON) for programmatic consumption and human-readable for developer review. It should feel like a linter for memory quality.

### Prompt review tool

A lightweight analysis tool that examines an app's configured prompts (extraction, worthiness, distillation) against a checklist of known requirements:

- Does the extraction prompt mention capturing the owner's own actions?
- Does the extraction prompt guide entity identification (primary and mentioned entities)?
- Does the extraction prompt assign importance scores?
- Does the extraction prompt classify epistemic type and confidence?
- Does the worthiness gate accept beliefs and inferences, not just objective facts?
- Does the distillation prompt encourage motivational inference ("why"), not just behavioral summary ("what")?
- Does the distillation prompt distinguish between observations that should be consumed and decisions that should persist?
- Does the extraction prompt handle the case where the owner didn't respond (facilitator/observer personas)?

This is static analysis — fast, free, and can run in CI. For deeper analysis, the tool can optionally run sample scenarios through the extraction and distillation pipelines and evaluate the output against expected results.

### Memory configuration skill for Claude Code

3tears ships an interactive Claude Code skill (`/3tears:configure-memory`) that reads a consuming app's documentation and codebase, selects and adapts presets, generates tailored extraction/distillation/retrieval configuration, and walks through app-specific test scenarios. The skill combines the presets, diagnostics, and prompt review tool with app-specific context.

The skill specification is maintained as a separate document (see `docs/skills/configure-memory-skill.md`). It covers: what the skill knows (data model, presets, checklist, scenario patterns, anti-patterns), what it does (understand the app, select preset, generate prompts, walk through scenarios, output configuration), and how it stays current with the evolving memory system.

---

## Known limitations and future directions

### Limitations in v2

**No knowledge graph layer.** Research converges on hybrid architectures (Zep/Graphiti, Cognee, Mem0g) where a knowledge graph enables multi-hop reasoning that vector search alone cannot. 3tears v2 uses `about_id` + `mentioned_entity_ids` for entity association, which enables 1-hop entity queries but not graph traversal. A query like "which listeners have friends who like jazz?" requires application-level joins. The `mentioned_entity_ids` field is the foundation for a future graph layer — the entity pairs are captured, but traversal logic is not implemented.

**No formal relationship model.** Relationships are represented as memories that mention multiple entities, but there's no typed relationship schema (friend/rival/collaborator), no trust scores, and no relationship strength metrics. The Beta Reputation System (Bayesian trust updating from interaction outcomes) and FIRE model (multi-dimensional trust combining direct experience, role-based trust, and witness information) from multi-agent systems research are natural future additions. The v2 distillation engine can be extended to produce relationship records when it detects recurring entity-pair patterns.

**No full bi-temporal model.** `valid_until` provides basic temporal supersession ("what's true now" vs "what was true before") but not the four-timestamp model (event time, ingestion time, validity start, validity end) that Zep/Graphiti implements. For most use cases, the simpler model suffices. Apps requiring historical temporal queries ("what did we believe about Bob on March 15th?") would need the full model.

**Commitment lifecycle is [foundation] only.** The `commitment` memory type is in the schema but the fulfillment/expiration lifecycle (tracking whether a promise was kept) is not implemented in the processing pipeline. Apps that need commitment tracking implement it at the application layer for now.

**Theory of mind accuracy.** LLM-based motivational inference is wrong ~25% of the time (PNAS 2024). The belief type with confidence tracking mitigates this but doesn't solve it. Consuming apps should treat belief-type patterns as working hypotheses, not established facts. The diagnostic tool's theory-of-mind indicators check for the presence of motivational inference but cannot verify its accuracy.

**Memory drift through LLM processing.** Each LLM call in the extraction and distillation pipeline can introduce subtle errors. With 3-4 LLM calls per extraction and 1-2 per distillation, errors compound. Non-destructive CONSUME preserves the evidence chain for audit but doesn't prevent drift. The memory health diagnostic's distillation depth analysis helps detect it.

### Future directions

- **Relationship table** populated by distillation: typed edges, trust scores, relationship strength, populated automatically when distillation detects recurring entity-pair patterns in the observation corpus.
- **Graph traversal** for multi-hop reasoning: leverage `mentioned_entity_ids` and the relationship table to answer queries spanning entity connections.
- **Commitment lifecycle engine**: track active commitments, surface them proactively, mark them fulfilled/broken/expired based on observed outcomes.
- **Full bi-temporal model** if consuming apps demonstrate need for historical temporal queries.
- **Sensitivity classification** at extraction time: LLM-assigned sensitivity scores on memories containing personal information (health, family, financial), enabling apps to implement tiered retention policies.
- **Embedding model migration support**: re-embedding tooling that handles all tiers, including distillation-produced content that may reference consumed observations.

---

## Migration guide for existing consuming apps

Memory v2 is a breaking change. There are no backward-compatibility shims or fallback modes. Consuming apps must migrate their database schema, API call sites, and configuration. This section covers every touching point.

### Database schema

**memories table:**

| v1 column | v2 column | Action |
|---|---|---|
| `user_id` | `owner_id` | Rename. Existing values carry over. |
| `conversation_id` | *(removed)* | Drop. Provenance is now in `source_type`/`source_id`/`source_context`. |
| `message_id_source` | *(removed)* | Drop. Same as above. |
| *(new)* | `about_id` | Add, nullable UUID. Existing memories get `NULL`. |
| *(new)* | `about_type` | Add, nullable text. |
| *(new)* | `mentioned_entity_ids` | Add, nullable UUID[]. Default `NULL`. |
| *(new)* | `source_type` | Add, nullable text. |
| *(new)* | `source_id` | Add, nullable text. |
| *(new)* | `source_context` | Add, nullable JSONB. |
| *(new)* | `tier` | Add, NOT NULL text, default `'observation'`. All existing memories become observations. |
| *(new)* | `importance` | Add, nullable smallint. Existing memories get `NULL` (treated as importance 5 in scoring). |
| *(new)* | `confidence` | Add, nullable float. *[foundation]* |
| *(new)* | `epistemic_type` | Add, nullable text. *[foundation]* |
| *(new)* | `derived_from_ids` | Add, nullable UUID[]. |
| *(new)* | `consumed_by_id` | Add, nullable UUID. |
| *(new)* | `date_last_accessed` | Add, nullable timestamptz. |
| *(new)* | `valid_until` | Add, nullable timestamptz. *[foundation]* |
| `type_memory` | `type_memory` | Values updated: `topical_context` → `fact`, `relational_context` → `fact`. New values: `action`, `belief`, `commitment`. |

Apps that stored `conversation_id` and `message_id_source` for audit purposes should migrate those values into `source_type`/`source_id`/`source_context` before dropping the old columns. For example, a chat app might set `source_type='conversation_message'`, `source_id=message_id_source`, `source_context='{"conversation_id": "..."}'`.

**media_content, media, memory_chunks tables:** Rename `user_id` to `owner_id`. No other structural changes.

**conversation_memory_refs table:** Unchanged.

**New table: entity_identities.** Stores the identity resolution registry (source-qualified IDs mapped to canonical entity IDs). Apps that don't use identity resolution can ignore this table, but it must exist.

**Indexes:** New indexes on `about_id`, `about_type`, `tier`, `importance`, `(owner_id, about_id)`, and a GIN index on `mentioned_entity_ids` for entity-scoped queries. The existing `user_id` index becomes the `owner_id` index.

### MemoryEntity

All field access on the entity proxy changes:

| v1 property | v2 property |
|---|---|
| `entity.user_id` | `entity.owner_id` |
| `entity.conversation_id` | *(removed — use source fields)* |
| `entity.message_id_source` | *(removed — use source fields)* |
| *(new)* | `entity.about_id` |
| *(new)* | `entity.about_type` |
| *(new)* | `entity.mentioned_entity_ids` |
| *(new)* | `entity.source_type` |
| *(new)* | `entity.source_id` |
| *(new)* | `entity.source_context` |
| *(new)* | `entity.tier` |
| *(new)* | `entity.importance` |
| *(new)* | `entity.confidence` |
| *(new)* | `entity.epistemic_type` |
| *(new)* | `entity.derived_from_ids` |
| *(new)* | `entity.consumed_by_id` |
| *(new)* | `entity.date_last_accessed` |
| *(new)* | `entity.valid_until` |

### MemoryExtractor

The `extract()` method signature changes:

**v1:**
```python
extract(pool, user_id, conversation_id, message_id_source,
        user_message, assistant_response, turn_count)
```

**v2:**
```python
extract(pool, owner_id, user_message, assistant_response,
        source_type, source_id,
        turn_count=..., source_context=...,
        about_context=..., gate_config=...)
```

Key changes at call sites:
- `user_id` → `owner_id`
- `conversation_id` and `message_id_source` replaced by `source_type` and `source_id`
- New `source_context` for optional structured metadata
- New `about_context` providing information about entities present in the interaction AND a broader entity vocabulary (known entities the conversation might reference), so extraction can populate `about_id`, `about_type`, and `mentioned_entity_ids`. When extraction identifies an entity not in the `about_context`, it sets `about_name` in the output and the consuming app can resolve it to a canonical ID or create a new entity.
- New `gate_config` for per-owner extraction gate overrides (or `None` to use defaults)
- `turn_count` becomes optional/keyword-only since not all source types have a concept of turns

The extraction prompt's expected JSON output format:

**v1:** `{"type": "...", "content": "..."}`

**v2:**
```json
{
  "type": "preference|fact|decision|action|belief|commitment",
  "content": "1-2 dense sentences",
  "tier": "observation|decision",
  "about_name": "primary entity name or null",
  "mentioned_names": ["other", "entity", "names"],
  "importance": 7,
  "confidence": 0.85,
  "epistemic_type": "observed|stated|inferred|hypothesized|decided|reported"
}
```

Apps using default prompts get this automatically. Apps with custom prompts must update the output format.

### MemoryRetriever

The `retrieve()` and `retrieve_with_candidates()` signatures change:

**v1:**
```python
retrieve(pool, user_id, user_text, ledger=None)
```

**v2:**
```python
retrieve(pool, owner_id, query, relevant_entities=None,
         tier_weights=None, mode="conversational", ledger=None)
```

Key changes at call sites:
- `user_id` → `owner_id`
- `user_text` → `query`
- New `relevant_entities`: list of entity IDs to boost (matched against both `about_id` and `mentioned_entity_ids`)
- New `tier_weights`: per-tier scoring adjustments
- New `mode`: `"conversational"` (default, with entity boosting, budget allocation, and MMR) or `"topic_scan"` (complete, entity-grouped, no MMR)

**Return type changes:**

`RetrievalResult` adds entity grouping. Each memory item includes `about_id`, `about_type`, `mentioned_entity_ids`, `tier`, `importance`, `confidence`, and `epistemic_type`. A new `entity_groups` field provides results pre-grouped by `about_id`.

The `context` string (formatted text for prompt inclusion) presents memories grouped by entity with epistemic annotations (e.g., beliefs are marked as hypotheses, low-confidence memories are annotated).

### MemoryConfig

New configuration fields:

| Field | Purpose |
|---|---|
| `importance_weight` | Scoring weight for the importance signal |
| `entity_boost_weight` | Scoring weight for entity-match signal |
| `entity_budget_ratio` | Fraction of context_budget for entity-specific memories (default: 0.7) |
| `general_budget_ratio` | Fraction of context_budget for general memories (default: 0.3) |
| `tier_weights` | Default per-tier scoring weights |
| `topic_scan_budget` | Max results for topic scan mode |
| `topic_scan_threshold` | Minimum score for topic scan mode |
| `distillation_importance_threshold` | Importance-sum trigger for within-entity distillation (default: 50) |
| `distillation_min_observations` | Minimum observations for session-boundary trigger (default: 5) |
| `distillation_background_interval` | Interval for cross-entity distillation (default: 24h, None to disable) |

Existing fields are unchanged. Apps can adopt new fields incrementally.

### MemoriesCollection

`find_by_user()` becomes `find_by_owner()`. New query methods:

- `find_by_owner_and_entity(owner_id, about_id, ...)` — memories where `about_id` matches or `mentioned_entity_ids` contains the entity
- `find_by_owner_and_tier(owner_id, tier, ...)`
- `find_by_owner_and_type(owner_id, type_memory, ...)`
- `find_consumed_by(pattern_id)` — observations consumed into a specific pattern
- `find_derived_from(memory_id)` — patterns derived from a specific observation

### Protocols

**ChatModelFactory:** New purpose value `"distillation"` for creating models used by the distillation engine.

**EmbeddingProvider:** Unchanged.

**New protocol — DistillationStrategy:** Apps that want custom distillation behavior implement this protocol. A default strategy ships with 3tears.

### Tools

`load_memory_search_tool()` and `load_recall_memory_tool()` signature changes:

- `user_id` → `owner_id`
- `MemorySearchInput` adds optional `about_id`, `about_type`, `tier`, `type_memory`, and `min_importance` filter fields
- Tool descriptions update to reflect entity, tier, and epistemic concepts

### Migration checklist

1. **Database:** Run the schema migration (rename columns, add columns, add table, add indexes). Backfill provenance fields from old columns if needed. All existing memories get `tier='observation'`, `about_id=NULL`, `importance=NULL`. Migrate `topical_context` → `fact` and `relational_context` → `fact`.

2. **Entity access:** Find-and-replace `user_id` → `owner_id` on all MemoryEntity property access, MemoriesCollection queries, and tool factory calls.

3. **Extraction calls:** Update all `extract()` call sites with the new signature. At minimum: rename `user_id`, move conversation/message IDs to source fields, pass `about_context=None`.

4. **Retrieval calls:** Update all `retrieve()` call sites. At minimum: rename parameters. Pass `relevant_entities` and `tier_weights` to opt into new retrieval behavior.

5. **Custom prompts:** If using `ExtractionPrompts` with custom text, update prompt templates to include new placeholders and expect the expanded JSON output format with importance, confidence, epistemic_type, and mentioned entities.

6. **ChatModelFactory:** Add handling for the `"distillation"` purpose.

7. **Result consumers:** Update code that reads `RetrievalResult` to handle new fields. Update prompt formatting to use entity-grouped output with epistemic annotations.

8. **Collection queries:** Replace `find_by_user()` calls with `find_by_owner()`.

Steps 1-3 are required for the app to function. Steps 4-8 can be done incrementally — the new parameters have defaults that preserve v1-like behavior. [foundation] fields (confidence, epistemic_type, valid_until, commitment type) are nullable and unused by default; apps adopt them when ready.

---

## Dependency order

```
A (memory model + schema)
├──> B (entity resolution)
├──> C (provenance)
├──> D (epistemic metadata + type taxonomy)
├──> E (retrieval: multi-signal scoring, entity boosting, budget allocation)
├──> F (within-entity distillation: triggers, non-destructive CONSUME, ToM prompts)
├──> G (cross-entity distillation: clustering, principle synthesis)
└──> H (temporal validity: supersession semantics)
```

**A is prerequisite to everything.** The owner/about/mentioned/tier/importance schema is the foundation. B, C, and D are independent of each other and can ship in any order. E depends on A and D (importance is a scoring signal). F depends on A and D (importance triggers distillation, epistemic_type affects distillation behavior). G depends on F (cross-entity distillation operates on patterns produced by within-entity distillation). H is independent of everything except A.

Work within each item can be incremental. For example, E can ship with basic entity boosting before adding budget allocation. F can ship with session-boundary triggers before adding importance-sum triggers. The [foundation] fields are in the schema from A but their processing logic ships with the item that uses them.
