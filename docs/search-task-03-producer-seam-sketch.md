# search-task-03 — the producer seam (D3), sketch for review

**Status:** SKETCH — for contract review, **not approved to build**.
**Blocks:** nothing. **Blocked by:** nothing to review; a *build* is blocked on
a consumer that can drive it (see §7).

D3 rules model-mediated search *out* of Adapter and Call and *in* at Aggregate,
as a candidate producer. `Provenance.producer` and both its constants shipped in
Phase 1, so the vocabulary exists. This document is the ingestion path, and it is
deliberately a sketch: reviewing a contract costs nothing, and catching a
boundary mismatch on paper is the whole point of doing it before the build.

Elicited against samsung's real code (`curation/src/curation/discovery/`), whose
session reviewed the questions and corrected two of the answers. Its findings are
recorded at `search-task-02-aggregate-and-select.md` §3b; this document turns
them into a shape.

---

## 1. What the seam is, and what it is not

**Is:** a way for a caller to hand Aggregate candidates that no adapter fetched,
so they dedup and merge alongside fetched ones without either class pretending to
be the other.

**Is not:** a plugin system, a registry, or a protocol this package calls. There
is no `CandidateProducer` protocol in this sketch, and that is deliberate —
Aggregate never *invokes* a producer. The caller runs its own producer, on its
own schedule, priced on its own ledger, and passes the result in. A protocol
would imply an inversion of control that D3 does not ask for and that samsung —
a synchronous one-shot `asyncio.run()` on a Pi (check 9) — should not have to
satisfy.

Today's `aggregate(extra_candidates=...)` already accepts produced candidates.
Everything below exists because that parameter cannot carry three facts a real
producer has: what it spent, what went wrong, and the guarantee that it is not
an adapter in disguise.

## 2. The shape

Two new contract types. Both are payload; neither carries a port.

```python
class SpendReference(ContractModel):
    """A pointer to spend priced on someone else's ledger (D3)."""

    #: which ledger owns the row -- a namespaced name, e.g.
    #: "curation.spend_records". Not a URL and not a table name.
    ledger: str
    #: the row's own identifier in that ledger, opaque to this package.
    record_id: str
    #: what kind of spend it was, for display only. Never priced, never
    #: summed, never compared.
    category: str | None = None


class ProducedSet(ContractModel):
    """What one producer contributed to a corpus (D3)."""

    #: the produced candidates. Each carries its producer class on its own
    #: provenance; this type does not restate it.
    candidates: tuple[Candidate, ...] = ()
    #: pointers to what producing them cost, on the producer's ledger.
    spend_references: tuple[SpendReference, ...] = ()
    #: degradations the producer reported (a truncated run, a refused
    #: intent), in the same open-string form CandidateSet uses.
    notices: tuple[str, ...] = ()
```

`aggregate()` gains `produced: Iterable[ProducedSet] = ()` and loses nothing;
`extra_candidates` collapses into it.

`Corpus` gains one field:

```python
    #: pointers to spend priced elsewhere (D3). Never summed into
    #: :attr:`spend`, which is search spend and stays search spend.
    producer_spend: tuple[SpendReference, ...] = ()
```

## 3. Rulings this sketch proposes

### S1 — The producer class lives on provenance, and nowhere else

`ProducedSet` does **not** carry a `producer` field. `Provenance.producer` already
does, per candidate, and two representations of one fact drift — the argument R1
makes about provenance, applied to the label rather than the record.

`aggregate()` **validates** rather than trusts: every candidate in a `ProducedSet`
must carry a `producer` other than `PRODUCER_API_PROVIDER`. An adapter-made
candidate arriving through this door is the impersonation D3 exists to prevent,
and a caller that has one should pass a `CandidateSet`.

### S2 — Producer spend is referenced, never re-priced

D3: *"the producer seam records a reference to that spend and MUST NOT re-price it
into search spend (no double counting)."* `SpendReference` makes that structural
rather than a matter of discipline — the corpus holds a pointer and has no field
an amount could go in.

samsung's session settled the shape and the reason. Its `EngineSpend` is
**pre-persistence**: no id, no timestamp, two instances indistinguishable, so a
reference to one refers to nothing durable. `SpendRecord` is the priced ledger
row and has an id. And its `SpendCategory` grows, with attribution rules not
derivable from the shape — `CONVERSATION_TOKENS` is deliberately *not* attributed
to the run a conversation seeds, so a naive roll-up double-counts intent-forming
spend into a run's actuals. Anything richer than a pointer re-implements those
rules wrongly.

`category` is display-only and typed `str | None` for that reason. If it ever
tempts someone to sum, it should be deleted.

### S3 — A produced candidate may carry zero locators

`Candidate.locators` is typed `tuple[Locator, ...]`, which permits an empty tuple,
and documented *"at least the canonical locator"*, which does not. The type is
right and the docstring is wrong for this case.

samsung's `ProposedWork` is `title` / `rationale` / `artist`. Nothing in its phase
1 produces a URL — locators arrive when phase 2 resolves images. So a produced
candidate is locator-less by nature, not by omission, and the docstring is
amended to say so rather than leaving the case technically-passing and
contract-illegal.

**Consequence worth stating:** `select.py`'s domain filters read locator hosts and
return `None` (unjudgeable) when there are none, so a locator-less candidate is
*dropped* by a domains criterion and counted in the "lacked the data to be judged"
notice. That is the honest outcome — a candidate with no URL cannot be shown to be
on a permitted domain — but a consumer mixing produced and fetched candidates
under a domain filter should expect it.

### S4 — Identity is the producer's own key, adopted rather than re-minted

A produced candidate's `identity` is whatever the producer's own dedup key
derives. For samsung that is `work_dedup_key(title=, artist=)`. This package does
not mint a second one.

Two identities for one thing are free to disagree, and the disagreement is
invisible. This is the argument `WorkList.searches_used` already makes on the
other side of the boundary about not keeping a second tally beside the priced one.

**Stated coupling, not a discovered one:** that key is a *derived* value under an
opinionated normalisation contract — accent folding, iterative stripping of
trailing dates, catalogue clauses and alternate titles, and a deliberate refusal
to collapse `Untitled (…)` because the residue identifies no work. A change there
moves our identities. Recorded here so it is a dependency rather than a surprise.

### S5 — A producer-mandatory facet is a producer obligation, not a contract field

samsung's `rationale` is required on `ProposedWork`, required on its stored row,
and rendered on a live review card. A producer omitting it hands a curator a bare
title to judge — the failure both its docstrings were written against.

But it is one producer's need, and a first-class `Candidate` field for it would be
the wrong trade. So the two halves are separated, which the first draft conflated:

- **Ignorable by consumers** (SR-C2): it rides `facets`, and a consumer that does
  not recognise it ignores it rather than failing.
- **Mandatory for the producer**: the *producer's own* contract requires it, and
  that repo asserts it at its boundary and fails loudly rather than rendering a
  bare title. This package does not enforce it — it is not this package's
  invariant to hold.

The key is namespaced: **`curation:match_rationale`**, following
`Criterion.namespaced`'s `<namespace>:<name>` house style. The namespace names the
**producing component**, not the consuming product: `curation` reads an intent and
proposes works and would produce the same sentence if the pictures ended up on a
projector; `samsung` names the display plane the facet never travels to. And
`match_rationale` rather than bare `rationale` because that repo has a second,
different sentence — `selection_rationale`, *why this instance was selected* — at
a later hop, and a bare key would eventually render the wrong one.

### S6 — A produced candidate holds no rank, and therefore no fused score

Already how `aggregate()` behaves: RRF sums over ranks read from `CandidateSet`s,
a produced candidate appears in none, and `_fused_score(None)` emits nothing
rather than a fabricated zero. Absent and last-place are different claims.

This ruling promotes that from an implementation consequence to a decision,
because the two repos reached **opposite rules from the same observation** and the
collision surfaces exactly here:

- *This package:* a rank is only **recoverable** inside the call that produced it,
  so capture it before grouping destroys it.
- *samsung's `browse.py`:* a rank is only **meaningful** inside the call that
  produced it, so do not export it — measured, not assumed: the Art Institute
  returns one Ellsworth Kelly at 13,535 against siblings scoring 6 to 8, so a
  caller handed that order gets a ranking that looks real and is not.

Resolved in samsung's favour: **`ProducedSet` carries no position field.** If a
producer's ordering is meaningful it can say so as a named `ScoreEntry`, where
`scale` and `source` make the claim inspectable and `comparable=False` keeps it
from being fused with anything. A bare position would be the thing that repo
refuses to hand its own callers.

### S7 — A producer that answers no criteria weakens the corpus disposition, and Select may win it back

R8 rolls dispositions up to the weakest honest answer. A `ProducedSet` answers no
criteria at all, so any criterion the corpus reports becomes at best `unsatisfied`
once produced candidates are in it — there are now candidates nobody filtered.

That reads alarming and is correct, and it composes cleanly with Select: for
criteria Select *can* apply locally, it applies them to produced and fetched
candidates alike and upgrades the disposition back to `local`. So the honest
sequence is *weakened by the producer, recovered by the cull, and left
`unsatisfied` only where nothing could satisfy it* — which is what P8 asks for.

**Open for review:** whether a `ProducedSet` should be able to *state*
dispositions (a producer that did honour a carrier constraint could say so).
Recommendation: **not in v1.** No consumer needs it, and adding it later is
additive. Mentioned because "the producer cannot answer" is currently a structural
fact, and one reviewer may prefer it be a choice.

## 4. What changes in existing contracts

| Change | Kind | Notes |
|---|---|---|
| `SpendReference`, `ProducedSet` | new types | additive |
| `Corpus.producer_spend` | new field, defaulted | additive |
| `aggregate(produced=...)` | new parameter, defaulted | `extra_candidates` collapses into it |
| `Candidate.locators` docstring | wording only | S3 — no type change |

All additive within a family minor (D13). No migration, no wire break for a
same-version reader.

## 5. What this sketch does not decide

- **Carrier dispatch** (§3.5's Phase 3 half, samsung C6 — image, deep). Separate
  task; the producer seam does not depend on it.
- **Whether `page_finder` should label its invented URLs `model-mediated`.**
  It computes the fact already (`url_was_a_search_result=False`) and would be an
  in-repo second producer, but that is a `scrape` change with its own callers and
  belongs in its own task.
- **Any producer implementation.** This package ships none, ever — the same
  stance the ranker slot takes for the same reason.

## 6. Tests a build would owe

- A `ProducedSet` whose candidate claims `PRODUCER_API_PROVIDER` is refused (S1).
- Producer spend appears in `Corpus.producer_spend` and **nowhere** in
  `Corpus.spend` — the double-counting regression D3 names (S2).
- A locator-less produced candidate survives aggregation, and is dropped *with a
  notice* by a domains criterion rather than silently (S3).
- A produced candidate and a fetched candidate sharing an identity merge to one
  entry with two contributions, and the entry's provenances show both producer
  classes (S1, R1).
- Fusion over a corpus containing produced candidates emits no score for them and
  is unchanged for the rest (S6).
- A corpus containing a `ProducedSet` reports `unsatisfied` for a criterion no
  contributor answered, and Select upgrades it to `local` for one it can apply
  (S7).

## 7. Why this is not approved to build

The build wants a consumer that can drive it, and as of 2026-08-12 there is not
one:

- **samsung** — its active plan is an eleven-chunk curation surface with no search
  producer in it, and `3tears-core` is deliberately absent from that repo (its
  durable tier matches `DurableStore` structurally, with no framework import).
  Available to *review* this, not to consume it.
- **discodon** — `requires-python = ">=3.12"` and declares no `3tears` packages;
  3tears needs 3.14, so it cannot install today. Its piece is replay (item 8)
  regardless.
- **metallm** — checkout last advanced 2026-06-06; its family lag must close
  first, and its Phase 5 work consumes Call and Extract rather than Aggregate.

Building now would produce a seam whose only caller is its own test — the shape
this repo shipped inert three times in a fortnight. The elicitation was still
worth doing early: two of samsung's four answers (S3's zero locators, S2's
opaque ledger reference) contradict what this document would otherwise have
specified, and both would have been found after the build rather than before it.

**One constraint to hold in the meantime:** the seam must stay importable without
pulling `3tears-core`, or samsung's dependency argument bites again and the answer
will be the one it was in July. True today —
`threetears.search.{aggregate,select}` load only `threetears.media` and
`threetears.search` — but that is a fact someone checked, not a test that holds
it. `test_import_cost.py` probes `threetears.search.contracts` alone; extending
it to the stage modules is owed and should land before the seam does.
