# search-task-02 — Aggregate and Select (Phase 3 item 9)

**Status:** rulings recorded, not started. **Blocks:** Gate B.
**Blocked by:** nothing for §5 steps 1-2; steps 3-4 want samsung (C5/C6), which
is being pulled forward rather than waited on.

Read `search-spec.md` §3.4 and §3.6 for *what these modules are*, and D1/D2/D3
for the rulings they inherit. This document is *what to build*, and the rulings
taken before building it.

Recorded ahead of the build for the reason the §3.5 precedent exists: a ruling
that lives only in the session that took it is a ruling the next session
re-litigates. Each is vetoable; a veto lands here and in
`search-requirements.md` §13.

---

## 1. The one-paragraph version

Aggregate turns many `CandidateSet`s into one `Corpus` — owning the dedup key,
the merge rule, fan-out accounting and spend rollup. Select turns a corpus plus
criteria into an ordered, filtered subset — owning local criteria application
and the cull, and exposing a ranker slot it never fills. The hard part is
neither of those sentences: it is that D1 forbids a single comparable score, so
every merge and every cull has to stay honest about judgments that do not share
a scale.

## 2. What already works, and must not be re-done

Verified against the shipped code, 2026-08-12.

**The producer seam's vocabulary shipped in Phase 1.** `Provenance` already
carries `producer`, and both well-known values already exist:

```python
PRODUCER_API_PROVIDER: Final[str] = "api-provider"
PRODUCER_MODEL_MEDIATED: Final[str] = "model-mediated"
producer: str = PRODUCER_API_PROVIDER
```

D3's mandatory clause — *"provenance carries a `producer` distinction from day
one"* — is discharged. Do not design a producer vocabulary; it is there, it is
open, and its docstring already states that a model-mediated candidate can never
impersonate an API provider. What Phase 3 owes is the *ingestion path*, which is
a much smaller thing than the phrase "producer seam" suggests.

**The score model is complete and is the constraint, not a gap.** `ScoreEntry`
carries `name` / `value` / `scale` / `source` / `comparable`, with
`SCALE_UNIT_INTERVAL`, `SCALE_UNBOUNDED` and `SCALE_RANK` named, and
`ScoreEntry.provider_native()` forcing `comparable=False` by construction. Do
not add a score type. `SCALE_RANK` in particular already exists, which matters
for R4 below.

**`CandidateSet` is Call's return and explicitly not the corpus.** Its docstring
says so, citing D2. Zero candidates is a success (SR-J2); `dispositions`,
`spend` and `notices` are already on it. The corpus is a new type, not a
widening of this one.

**`CriterionDisposition` already answers per criterion** with pushdown / local /
unsatisfied / ignored-unknown, plus a `detail` string. Select's local
application records into this vocabulary rather than inventing a second.

## 3. Rulings taken before the build

### R1 — The corpus is a set of *groups*, not a bag of merged candidates

This is the ruling most worth reading, because the obvious implementation is
wrong in a way that passes tests.

`Candidate.provenance` is **singular**. So a dedup that collapses the same URL
returned by SearXNG and by Tavily into one `Candidate` has exactly one slot for
an origin, and must therefore discard one — destroying the per-result grounding
SR-A3 exists to keep, and taking one provider's score entries with it.

So the corpus entry holds the contributing candidates whole:

```python
class CorpusEntry(ContractModel):
    identity: str
    contributions: tuple[Candidate, ...]   # one per producing call, each intact
```

Merge becomes a **view over** the contributions, never a mutation of them. Every
provenance survives, every score entry keeps its own `source`, and the rank a
candidate held within its own `CandidateSet` is still recoverable — which is
what R4 needs. D2 sanctions exactly this: *"two types, two dedup/merge
stories"*. The corpus is Aggregate's own type and is under no obligation to be a
bag of `Candidate`.

**The alternative that was not taken**, so it stops being re-proposed: adding
`additional_provenance: tuple[Provenance, ...]` to `Candidate`. It makes the
per-candidate shape lie about itself — one first-class provenance and a
second-class pile beside it — and every consumer reading `candidate.provenance`
would silently see whichever provider happened to win the merge.

### R2 — Dedup keys on `identity`, and identity is not normalised here

`Candidate.identity` is documented as *"by convention the canonical URL;
providers without URLs use their native id."* Aggregate keys on it verbatim.

It is tempting to normalise (strip `utm_*`, unify scheme/host case, drop
trailing slashes) and it is refused for v1: URL normalisation is a policy with
real false-merge risk — `?page=2` is not tracking chaff, and two hosts serving
different content at case-differing paths exist. If a consumer wants it, it
belongs where the identity is *minted* (the adapter, which knows the provider's
conventions) or in a caller-supplied key function, not baked into the corpus.
Recorded as an accepted limitation rather than a gap: a corpus that under-merges
is honest and costs a duplicate; one that over-merges silently destroys a
distinct result.

### R3 — Merged scores are never combined into a value

D1's rule applied to the merge path. When an identity has several contributions,
the corpus exposes their score entries as a **union of distinct entries**, each
keeping its own `source` and `comparable=False`. No averaging, no max, no sum.

The evidence is in the requirements: SearXNG's weight is unbounded above (two
agreeing engines score 4.0, three score 9.0) while Tavily's relevance is
`[0,1]`. Any arithmetic across those is a number with no meaning that reads as
a ranking.

### R4 — RRF is offered, never the default, and its output is a derived entry

§3.4 says Aggregate MAY implement reciprocal-rank fusion and MUST NOT require
it. Both halves are kept, and the reason RRF is the fusion worth offering is
that it consumes **rank position, not score value** — so it is the one fusion
that is correct-by-construction under D1 rather than papering over
incomparability.

When a caller asks for it, it emits one derived `ScoreEntry`:

- `scale=SCALE_RANK` (already in the vocabulary),
- `source=` the stage name, never a provider instance,
- `comparable=True` — permitted here and *only* here, because `ScoreEntry`'s own
  docstring sanctions it for *"a pipeline stage that normalised across
  providers"*, which is precisely what fusion did.

Default is **no fusion**: the corpus preserves per-call provider order and says
it is unranked.

### R5 — Select's ranker slot ships empty, and "no ranker" is a visible state

§4.14 rules rerank out of search entirely (MMR lives in `agent-memory`, rerank
metadata in `3tears-models`, a cross-encoder arrives as a models provider), and
`search-requirements.md` §13 already records Select drifting once into owning
"filtering, reranking, scoring, cull" and being corrected back. Building a
ranker re-commits a drift that was caught and written down.

Specifically **not** a pass-through default implementation. SR-L2/P8 require
unranked output be *marked* unranked; something occupying the slot while
returning input order is a ranking implementation that lies about being one, and
the mark either disappears or becomes false. No ranker → an explicit unranked
notice → provider order preserved and labelled as provider order.

There is a second reason the empty slot is load-bearing rather than tidy:
**success check 5 is "a Pi deployment installs it without torch"**, and a
bundled ranker is exactly how torch arrives.

### R6 — The cull reads criteria, never a score threshold

D1 is explicit: *"Select's cull MUST NOT read `score > 0` as 'relevant' — a
`priority: low` engine scores everything 0."* So the cull applies criteria and
an explicit `max_results`, and never a threshold on a provider-native score. A
caller that wants a threshold names the score and supplies the bound, and gets
told (via a disposition) that it was applied locally.

### R7 — One failing call never poisons the corpus

SR-H3's fan-out rule, and the same shape `extract.py` already ruled for
per-candidate outcomes. A `CandidateSet` from a call that failed contributes its
`notices` and its `spend` to the corpus and no candidates; it does not raise
through Aggregate. Spend rolls up whether or not candidates arrived — D4's
"budget follows the bill" does not care that the bill bought nothing.

### R8 — Corpus-level dispositions report the weakest honest answer

Each contributing `CandidateSet` answers per criterion for itself, and two
providers can answer differently for the same criterion — one pushed
`time_range` down, another could not. The corpus keeps the per-call dispositions
and, at corpus level, reports the **weakest** answer across contributors
(`unsatisfied` beats `local` beats `pushdown`), with `detail` naming which
contributors diverged.

Reporting the strongest would read as a filtered corpus that is not filtered,
which is the exact defect P8 exists to prevent.

## 4. What is missing (the build)

- `contracts/corpus.py` — `CorpusEntry` and `Corpus` per R1, with the
  disposition rollup of R8.
- `aggregate.py` — accumulation, dedup on R2's key, the R3 merge view, R7's
  failure accounting, spend rollup, and the optional R4 fusion.
- `select.py` — local criteria application into `CriterionDisposition`, R6's
  cull, R5's ranker slot and unranked marking.

## 5. Sequencing

1. `contracts/corpus.py` — the types. Nothing else can reference them first.
2. `aggregate.py` — API-provider candidates only. **No samsung dependency.**
3. `select.py` — criteria, cull, empty ranker slot. **No samsung dependency.**
4. Producer seam driven by **samsung C5** as a real second producer.
5. Carrier dispatch driven by **samsung C6** (image, deep) — §3.5's Phase 3 half.

Steps 1-3 start now. Steps 4-5 are why samsung is being pulled forward: a seam
with one implementer and a dispatch with one carrier are the shape this repo has
shipped inert three times in a fortnight (the context-save node's default set,
`chunker.py`'s bare-name strategy, and item 5's untested tool/transport seam).
D3 deferred the seam to "when samsung pulls" on the same reasoning that
correctly blocks item 8 on discodon's storage port — but discodon's port does
not exist yet and samsung is available, so the deferral inherited a constraint
that does not apply to it.

samsung is also the consumer that has **already recorded package rejections**
for `3tears-core` and `3tears-models` on `MemoryMax` grounds. Success checks 5
and 9 (no torch; a synchronous one-shot `asyncio.run()`) are its constraints,
and the requirements doc's warning is that *"a capability that does not state
its deployment constraints gets refused the same way, on the same evidence,
after it is built."*

## 6. Tests the build owes

- The same URL from two providers merges to **one entry with two
  contributions**, both provenances intact, both providers' score entries
  present and each still `comparable=False` (R1, R3).
- A corpus built from a SearXNG set scoring `9.0` and a Tavily set scoring `0.8`
  exposes **no combined value anywhere** (R3) — the regression that matters,
  since averaging them produces a plausible number.
- A `priority: low` engine scoring every result `0.0` survives the cull (R6).
  This is D1's named failure and it must have a test with that shape.
- No fusion requested → output carries the unranked notice; fusion requested →
  exactly one derived `SCALE_RANK` entry sourced to the stage, `comparable=True`
  (R4).
- A corpus with no ranker reports unranked; there is **no code path** in which
  the slot is filled by a default (R5).
- One call raising mid-fan-out → its spend and notices land, its siblings'
  candidates all arrive (R7).
- Two contributors disagreeing on one criterion → corpus disposition is the
  weaker, and `detail` names the divergence (R8).
- Import cost: `test_import_cost.py` already exists — the new modules must not
  pull anything new into the default install (success check 5).

## 7. Explicitly out of scope

- **Any ranking implementation.** R5. It is `agent-memory`'s and
  `3tears-models`' by §4.14.
- **URL normalisation.** R2 — belongs at identity minting or in a caller-supplied
  key.
- **`replay.py`** — Phase 3 item 8, still blocked on discodon's carved storage
  port, and deliberately so.
- **Cross-run corpus persistence.** D14's stance and SR-M2/SR-O3 govern; a
  corpus lives for the caller that built it.
