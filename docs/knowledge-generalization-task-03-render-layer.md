# knowledge-generalization-task-03: Split the render layer — product vocabulary, two audiences

**Status:** READY. Independent of the registry contract that blocks task-01/02 (this shard touches
rendering, not the serving SQL), so it can start now. It does not depend on the renames — it can be
written against today's field names and re-pointed when they land, or land after them.
**Scope:** `3tears-agent-knowledge` (`middleware.py`, `integration.py`) + a new render seam in
`3tears-core` or `3tears-agent-knowledge`. Consumers: `14-eng-ai-bot` (supplies SQL vocabulary; gains
a runbook surface).
**Origin:** `knowledge-generalization-design.md` D6 / D7.

---

## Objective

Two changes to how governed knowledge becomes output:

1. **Vocabulary moves to the product.** 3tears owns *which sections in what order*; the consuming
   product supplies *the words*. Today the words are hardcoded SQL.
2. **Rendering splits by audience.** The same resolved knowledge produces a **runbook** (human) or an
   **injected governed block** (AI). Today only the AI renderer exists, welded to the AI pipeline.

## Part 1 — Vocabulary is the product's

Everything 3tears currently hardcodes about SQL in the render path:

| location | today | why it is vocabulary |
|---|---|---|
| `middleware.py:117` `_GLOSSARY_INVARIANT_HEADER` | *"…before writing SQL."* | the action verb is the product's |
| `middleware.py:123` `_GLOSSARY_SITUATIONAL_HEADER` | *"…before writing SQL."* | " |
| `middleware.py:136` `_SITUATIONAL_HEADER` | *"…touches the tables or columns in your query"* | " |
| `middleware.py:152` `_BLOCK_PREAMBLE` | *"Apply every rule that touches your query before you run it"* | " |
| `middleware.py:989` `_render_concept` | `"Bound table: …"` | the binding's *label* is the product's |

**Structure that stays in 3tears** (do not let the vocabulary pack absorb it): glossary before
procedures; invariant before situational; invariants in stable scope order and situational in the
ranked order the trim produced (`_render_block:848` is explicit that the renderer must **not**
re-sort the situational lists — preserve that); shadow-disclosure and ambiguity lines; the
per-item render isolation in `_render_governed_items:782`.

**Shape:** a vocabulary pack the product supplies (via `configurable`, alongside the integration).
The SQL pack reproduces today's strings **byte-for-byte** — that is the regression test.

The `target_kind` question the design doc leaves Open resolves here: the renderer is the first thing
that genuinely needs to know the kind, because it picks the pack. A turn is scoped to one target
today (`knowledge_capability_source_id` on `configurable`, per task-01), so the pack is arguably a per-turn input, not a
per-row one — but that filter is **optional** (`None` returns every visible entry), so a turn *can*
span targets and therefore kinds. Decide here: either scope a turn to one kind, or carry `target_kind`
on the row and render per-kind sections. **Do not push it back to core.**

## Part 2 — Two audiences

Today `_build_governed_injection` (`middleware.py:371`) does the whole pipeline in one function:

```
retrieve → split invariant/situational → embed query → fetch vectors → rank → trim to 2000 tokens → render as imperative prompt
```

Every step after `retrieve` is **AI-specific**. A human reading a runbook wants none of it:

| | AI — injected block | human — runbook |
|---|---|---|
| budget | trim to `_DEFAULT_KNOWLEDGE_RETRIEVAL_TOKEN_BUDGET` (`:109`, 2000) | none |
| ranking | cosine similarity vs the turn query | none — stable order |
| invariant/situational split | drives the trim exemption | a section label at most |
| provenance | discarded (ledgers → `metadata`) | **the point** — show the layered view |
| voice | imperative, binding (*"you MUST"*) | descriptive |

**The human-facing data shape already exists and was built for this.** `EntryLayered` (`merge.py:211`)
/ `ConceptLayered` (`concept_merge.py:193`) retain the winner *plus* the shadowed ancestors,
nearest-first, explicitly so *"a UI / review surface can show 'this user entry shadows the platform
entry'"*. `merge_entry_views` returns `(effective, layered)` on every call and the AI path throws the
layered half away. The runbook renderer consumes it.

**Work:** extract retrieval+merge from rendering so both audiences share one resolve; keep the AI
pipeline byte-identical; add a runbook renderer over the layered views. The hub exposes it (it
already has the HTTP surface in `hub/knowledge/playbook_routes.py`).

## Design constraints

- **Do not change what the AI sees.** The injected block must be byte-identical before and after,
  with the SQL vocabulary pack installed. Snapshot-test it first, then refactor.
- **`GovernedKnowledgeRenderError` (`middleware.py:90`) survives untouched.** An invariant that
  cannot render fails the turn closed. That is D11 and it is correct — do not soften it into a
  warning while moving code around.
- **The runbook renderer is descriptive, never imperative.** Same data, different voice: the AI block
  is binding instruction, the runbook is documentation. Do not reuse `_BLOCK_PREAMBLE`'s wording.
- **The soft-fail contract is per-tier and stays that way** (`_render_governed_items:782`): a
  *situational* item that fails to render is skipped and logged; an *invariant* raises. Both
  audiences honour it.

## Acceptance

- No string in `3tears-agent-knowledge` contains "SQL", "query", "table", or "column" as *vocabulary*
  (structure and field names excepted).
- With the SQL vocabulary pack installed, the injected block is byte-identical to today's — proven by
  a snapshot test written **before** the refactor.
- A runbook renders from one leaf entry and shows the layered provenance the AI path discards.
- Retrieval + merge run once and feed both renderers; no second round-trip for the human path.
- An invariant render fault still raises `GovernedKnowledgeRenderError` on the AI path.
- `./scripts/check-all.sh` green.
