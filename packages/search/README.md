# 3tears-search

Provider-agnostic web and media search for the 3tears family.

The authority for everything in this package is
[`docs/search-spec.md`](../../docs/search-spec.md) (decisions D1–D28), with
requirement IDs (`SR-*`, `G*`, `P*`) defined in
[`docs/search-requirements.md`](../../docs/search-requirements.md).

## Layout

```
threetears/search/
  contracts/        # the leaf within the leaf — types, protocols, errors, keys
```

`contracts/` is the lingua franca every layer and every consumer speaks:

- `SearchRequest` and the open criteria vocabulary (typed constructors for
  well-known criteria, namespaced keys for everything else), with per-criterion
  dispositions (`pushdown | local | unsatisfied | ignored-unknown`).
- `Candidate` — the carrier-neutral result core: identity, locators,
  provenance, named provenanced scores (never a single `score` field, D1),
  fidelity available/achieved, an optional content slot, and additive facets
  keyed by the `media-contracts` vocabulary.
- `Spend` — every resource a call consumed: money (Decimal), wall-clock,
  call count, weighted provider units, bytes.
- The typed error taxonomy (SR-J1), every error carrying `Spend` (SR-E3).
  Zero results is a success value, not an error (SR-J2).
- `SearchTransport` — the injected transport seam (SR-N1, P9). A thin
  host-side adapter over `threetears.core.http_client.TracedHttpClient`
  satisfies it structurally; this package never imports core.
- `SEARCH_RESULTS_METADATA_KEY` and the versioned metadata projection (D13,
  D22).
- Canonical serialization of request/parameter types — one canonical form
  consumed by both the D26 replay key and eval run identity (SR-F1).

## Import-cleanliness

Importing `threetears.search.contracts` pulls nothing beyond stdlib, pydantic,
and `3tears-media-contracts`. Nothing in this package imports
`threetears.core`, `threetears.agent.*`, langchain, or NATS. Nothing reads
environment variables — the host passes base URLs, secret references, and
transport (SR-K1).
