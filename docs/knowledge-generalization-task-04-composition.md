# knowledge-generalization-task-04: Composition — entries that reference other entries

**Status:** DEFERRED — designed, **not scheduled**. No consumer needs it yet. It is captured now
because it is the devops system's likely next need, and because task-01's anchor decisions had to be
made knowing it was coming. They do not foreclose it (see "Why the anchor still works").
**Scope:** `3tears-core` (`threetears.knowledge`) + `3tears-agent-knowledge` (retrieval, budget,
render). Cross-repo when it lands.
**Origin:** `knowledge-generalization-design.md` D8–D13.

> **Do not start this shard to "get ahead".** Design rule 3: no speculative generality. It lands when
> a real devops task needs one governed unit to reference another — not before.

---

## Objective

Let a knowledge entry reference **other entries, in order**, so "how to accomplish this task by
combining these things" is governed the same way "how to do this one thing" already is.

The distinction that motivates it (user, 2026-07-15): *"a chain can produce a runbook, so can a
single kubectl knowledge that doesn't chain."* A **runbook is an output** (D7). Composition is about
what the knowledge layer can *store*, not what it can print.

## The shape

**One entity, nullable `steps`.** An entry with no steps is a **leaf**; an entry with steps is a
**composite**. Both are entries. Not a second entity.

**Steps reference lineage, not text** (D9). A step points at an entry's **chain root**; the merge
resolves it to *the caller's effective winner*. A customer who shadows step B with B' gets B'
composed into every composite referencing B — no composite edit, that customer only. **This is the
entire reason composition belongs in this system rather than beside it.** Inline-text steps would
make a composite nothing but an entry with a numbered body, and buy nothing.

**Why one entity and not two** — shadow resolution works *across* the shapes. Platform authors
`deploy = [A, B, C]`; a customer shadows it with a one-line leaf (*"just run `make deploy`"*), or the
reverse. Nearest-scope-wins, whole-unit replace: coherent, desirable, free from the existing rule.
Two entities forbid that override for no benefit.

The objection that `steps` "is not inert like `enforcement`, so it cannot ride as an optional column"
does not survive: the budget **already tiers on a column** — `always_inject` exempts an entry from the
trim entirely (`_split_invariant_entries`, `middleware.py:493`). "Has steps → all-or-nothing" is a
third tier in an existing tiering, not a new axis.

## What is genuinely new

**1. All-or-nothing budget units (D10).** `_rank_and_trim_shared` (`middleware.py:582`) is greedy
per-item: sort the pool, accumulate rendered cost, `break` at the budget. **Trimming step 3 of a
5-step composite hands the agent a confidently wrong procedure.** A composite is included whole or
excluded whole. The greedy loop cannot express that today — this is the real new machinery, and it is
the bulk of this shard.

**2. Cycles on a second axis.** An entry can reference an entry that references back.
`assert_no_origin_cycle` (`merge.py:229`) + `MAX_SHADOW_CHAIN_DEPTH` (`merge.py:77`) is exactly the
right pattern — visited set, bounded depth, rejected at **write** time — and it needs a sibling for
the steps axis. The two axes are **independent**: A may shadow B while B steps through A'.

**3. Cross-target visibility (D11).** A composite anchors to one target, but its step refs may point
at entries on others. **Visible iff the caller can see every target it touches.** Never render a
partial composite — a composite with a hole is wrong instructions, which is worse than none.

## Fail closed — this is a requirement, not a caveat

> "just like an agent without access to knowledge on a datasource it has fails closed. This is
> critical failure and it needs to be or it will be doing something wrong, wrong data, wrong
> kubectl." — user, 2026-07-15

A broken step reference (deleted target entry, invisible target) **fails the turn closed**, the same
way an unrenderable invariant raises `GovernedKnowledgeRenderError` (`middleware.py:90`) today. This
converts a referential-integrity problem into an availability problem **on purpose**: an agent missing
governing knowledge does not do less, it confidently does the wrong thing to a real cluster.

Therefore **referential integrity on `steps` is not optional** — FK or write-time check. A dangling
ref is not a rendering nuisance; it is an outage, by design.

## Knowledge describes; it never executes (D12)

The line: knowledge tells the agent what to do → the agent decides → the tool executes → enforcement
gates the execution. Composites sit **entirely on the left**.

The moment a step carries `on_failure` / `retry` / `timeout`, this is a workflow engine inside a
knowledge base — owning a DSL, a scheduler and error semantics forever — and an executable step
bypasses both the agent's judgment *and* the enforcement gate that stands between a governed
constraint and a live `kubectl delete`. If deterministic devops execution is wanted, that is a
separate product that **consumes** this knowledge.

**This is the constraint most likely to erode under delivery pressure.** Guard it in review.

## Naming: "chain" is reserved (D13)

`chain` already means a **shadow lineage** here (`chains.py:69` `ResolvedChain`, `chains.py:140`
`resolve_shadow_chains`, `merge.py:77` `MAX_SHADOW_CHAIN_DEPTH`, *"the nearest-scope member of a
chain"*) — the **semantic inverse** of composition:

| | existing `chain` | composition |
|---|---|---|
| members | N units, **one wins** | N units, **all run** |
| logic | disjunctive — pick the nearest | conjunctive — do them in order |
| axis | vertical, across scopes | horizontal, across steps |

Plus LangChain (an LCEL composition) is imported throughout `middleware.py`. Three meanings, one word.

**Use `steps` / leaf / composite. "Chain" must not appear in the composition concept** — otherwise
someone writes "the chain's step resolves through its chain", in a file called `chains.py`, forever.

## Why the anchor still works (task-01 does not foreclose this)

`capability_source_id` stays **single and required** for every entry, composite included — it is the RBAC carrier
(D2), and a composite has an owner like anything else. Cross-target composition is handled by the
step-ref visibility rule above, not by relaxing the anchor. The two decisions compose; task-01 needs
no change when this lands.

## Acceptance (when it lands)

- `steps` is nullable on the entry; a leaf is byte-identical to today's entry in every path.
- A step resolves through the merge to the caller's effective winner — proven by a test where a
  customer shadow of a referenced entry changes the composite's rendering **for that customer only**.
- A leaf shadows a composite and vice versa; nearest scope wins, whole-unit replace.
- A composite is included whole or excluded whole by the budget; never partially rendered.
- A cycle on the steps axis is rejected at write time; the shadow axis is unaffected.
- A broken or invisible step ref fails closed.
- No step field expresses execution semantics (`on_failure` / `retry` / `timeout`).
- `./scripts/check-all.sh` green.
