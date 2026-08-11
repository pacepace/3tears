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
  adapters/
    searxng.py      # one provider's API, over the injected transport
  call.py           # a query → one candidate set, bounded and negotiated
  bind.py           # prose for a model + the metadata projection
  standalone.py     # bare-httpx transport   [standalone] — the sanctioned path (D19)
  testing/          # the shared provider-conformance suite + declared doubles
```

Layer names (Adapter, Call, Bind, …) are module vocabulary and never type
names, so a later re-cut of the layers stays cheap.

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
- `ProviderCapabilities` — what a provider can express, declared and
  queryable so a consumer branches before sending rather than after failing
  (SR-B4), following the `3tears-models` capability-metadata pattern.
- `SearchProvider` — the provider seam Call depends on and the conformance
  suite parametrizes over.
- Canonical serialization of request/parameter types — one canonical form
  consumed by both the D26 replay key and eval run identity (SR-F1).

## Using it

```python
import asyncio

from threetears.search.adapters.searxng import SearxngAdapter
from threetears.search.bind import bind_search
from threetears.search.contracts import Criterion, SearchRequest
from threetears.search.standalone import StandaloneTransport   # or your own


async def main() -> None:
    adapter = SearxngAdapter(
        base_url="https://searx.internal.example",   # deployment config, never env
        transport=StandaloneTransport(allow_private_addresses=True),
        provider_instance="searxng-main",
    )
    rendered = await bind_search(
        SearchRequest(query="capybara habitat range", criteria=(Criterion.max_results(5),)),
        provider=adapter,
    )
    print(rendered.content)                                  # prose for a model
    print(rendered.metadata["search_results"]["candidates"])  # structure for a program


asyncio.run(main())
```

`bind_search` never raises: a typed failure arrives as a failed
`RenderedSearch` carrying its spend under the same metadata key (D10). Callers
that want the exception go through `threetears.search.call.search` instead.

Budgets and pacing pass through the same entry point: hand `bind_search` (or
`search`) a `budget=` implementing `BudgetPort`, a `limiter=` such as
`threetears.search.limiter.InProcessRateLimiter` — construct **one per
process** and share it, or pacing paces nothing — and the `egress=` name your
transport actually exits by (D8, D20). A budget refusal or pacing denial
renders as a failed result like any other typed failure; omitting the ports
means no budget is consulted and no pacing applies.

Hosts that already have `threetears.core` should inject a thin adapter over
`TracedHttpClient` rather than take the `[standalone]` extra — it brings
timeouts, retry, circuit-breaking and spans for free.

## Provider conformance

`threetears.search.testing` ships the suite every adapter passes — contract
shape, spend on failure, error taxonomy, disposition honesty,
zero-results-is-success (SR-O5). It imports no test framework, so a consumer
can run it against its own wiring:

```python
from threetears.search.testing import ProviderConformanceCase, ProviderConformanceSuite


class TestMyProviderConformance(ProviderConformanceSuite):
    case = ProviderConformanceCase(...)
```

## Not here yet

`aggregate.py`, `extract.py`, `select.py`, `limiter.py` and `replay.py` are
later phases of `docs/search-spec.md` §7. Budget-port consultation and pacing
are marked seams inside `call.py`: the port types are Phase 1 PR 2, and a
placeholder protocol would only be a second vocabulary to migrate off.

## Import-cleanliness

Importing `threetears.search.contracts` pulls nothing beyond stdlib, pydantic,
and `3tears-media-contracts`. Nothing in this package imports
`threetears.core`, `threetears.agent.*`, langchain, or NATS. Nothing reads
environment variables — the host passes base URLs, secret references, and
transport (SR-K1).

`standalone.py` is the only module that imports `httpx`, and nothing in the
package imports `standalone` at module level: the extra stays opt-in, and a
host that injects its own transport never installs it. Both facts are pinned
by `tests/test_package_boundaries.py`, and the module's path is the D19
widening of the no-bespoke-client norm in
`tests/enforcement/test_no_bespoke_reuse.py` — a sanctioned transport, with no
exemption filed.
