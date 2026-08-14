# search-task-01 — conditional revalidation on the fetch path

**Ruling:** SR-M4 / D30, 2026-08-12. **Status:** steps 1-4 BUILT 2026-08-14;
step 5 (scrape) remains a separate decision with a separate owner.

*What the build changed against this document.* One correction, in §3.1's
"additive with a default, so the existing scrape-side implementer stays
conformant". A default keeps an implementer conformant only if the caller does
not pass the argument: an implementer written before the parameter exists has no
such parameter, so passing it is a ``TypeError``, not a graceful ignore. The
published ``ScriptedHeavyFetcher`` proved it by failing. Extract therefore passes
``headers`` **only when there is something to send**, which is the opt-in path
and nothing else -- so an implementer grows the parameter when somebody actually
asks it to revalidate, not when this lands.
**Blocks:** nothing. **Blocked by:** nothing — Phase 2's remaining item (the
context-save node) and Phase 3 are independent of this.

Read `search-requirements.md` SR-M4 for *why*. This document is *what to build*.

---

## 1. The one-paragraph version

The consumer holds the bytes (D7) and owns retention (D12). It therefore holds —
or could hold — the `ETag` / `Last-Modified` those bytes arrived with. Nothing in
the stack lets it spend them, so every re-read of an unchanged page pays full
freight: a fetch, and on the scrape path a render *and* an LLM extraction. This
task threads a caller-supplied validator down to the transport and threads a
*not modified* outcome back up.

It is not caching. Nothing is stored anywhere in the capability, not even for the
duration of a call. D14 is untouched, for the same reason the robots memo left it
untouched, only more so.

## 2. What already works, and must not be re-done

**The transport seam is complete.** Verified 2026-08-12 against the shipped code:

- `FetchTransport.fetch` already accepts `headers: Mapping[str, str] | None`.
  A caller-supplied `If-None-Match` needs no protocol change to reach the wire.
- `TransportResponse` already carries `status_code` and `headers` (lower-cased
  keys by contract). A `304`, and the `ETag` on any `200`, are already readable.
- A `304` already survives `StandaloneTransport` end to end. Two rules that
  could plausibly have broken it both decline to fire, each for a reason written
  for something else:
  - `_gate_content_type` returns early unless `200 <= status < 300`, so a `304`
    declaring no content type is not refused as an unknown carrier;
  - the redirect branch is `300 <= status < 400 and bool(location)`, and a `304`
    carries no `Location`, so it is not mistaken for a redirect.

Do not "add header support to the transport." It is there. Start above it.

## 3. What is missing

In rough order of contract weight.

### 3.1 `HeavyFetcher.fetch_rendered` has no `headers` parameter

```python
async def fetch_rendered(
    self, url: str, *, max_bytes: int, timeout_seconds: float | None = None
) -> TransportResponse: ...
```

The browser path cannot revalidate even in principle. This is the one genuine
protocol change, and it is the one that matters most: the heavy path is exactly
where a `304` saves the most (a render avoided, not just a GET).

Additive with a default (`headers: Mapping[str, str] | None = None`), so the
existing scrape-side implementer stays conformant until it chooses to honour it.
Note in the docstring that an implementer which *ignores* `headers` MUST NOT
report `304` — silently returning stale bytes under a validator the caller
supplied is worse than not supporting the feature.

### 3.2 `extraction_status` has no `unchanged`

Today: `none` / `pending` / `complete` / `failed` / `refused`
(`media-contracts`, added by #315). A `304` is none of these — it is a success
that produced no content *and did not need to*.

Add `EXTRACTION_STATUS_UNCHANGED` alongside the others, in the same module, and
extend the vocabulary test that pins them. This is a `media-contracts` change,
so it lands first and the family bound moves with it.

**Watch the readers.** `web_fetch` currently treats "content is None or status
is not `complete`" as failure:

```python
if fetched.content is None or status != EXTRACTION_STATUS_COMPLETE:
```

`unchanged` satisfies both halves of that condition and would be reported as a
failed fetch. As of 2026-08-12 that is the **only** such reader outside
`media-contracts` itself and `extract.py`'s own writer (`extract.py:235`, which
sets the facet) — so this is one line to change, not a sweep. Re-grep
`EXTRACTION_STATUS_COMPLETE` before writing code in case a second has appeared.

Note the status rides `Candidate.facets` under `EXTRACTION_STATUS_FACET` rather
than a first-class field; that is the existing arrangement and this task does not
disturb it. The validators in 3.3 are a different question and get a different
answer, for the reason given there.

### 3.3 Nowhere to keep the validators

`ContentSlot` carries `text` / `origin` / `mime_type` / `size_bytes`. `Candidate`
carries `facets` (open, `JsonValue`).

Validators belong as **first-class fields, not facets**, by the argument
`published_at` already makes in its own docstring — *"a first-class field rather
than a facet: a publication date is not carrier-specific"*. Neither is an ETag.
Facets are the carrier-specific escape hatch (SR-C2/C3); putting a
carrier-neutral HTTP validator there would be the wrong precedent for the next
person reading it.

Proposed, on `ContentSlot`:

```python
etag: str | None = None
last_modified: str | None = None   # the header's own string, not a datetime
```

`last_modified` stays a **string**, deliberately, unlike `published_at`. It is an
opaque token to be echoed back verbatim in `If-Modified-Since`; parsing it to a
datetime and re-rendering it risks changing the bytes and failing the match, and
buys nothing — nobody compares it, they only return it.

Also extend `ContentSlot.origin`, currently
`Literal["provider-response", "later-fetch"]`, with a value meaning *the caller
already had this and upstream confirmed it* — `"revalidated"`. That keeps SR-A2's
existing "where did this content come from" question answerable, which is what
`origin` is for.

### 3.4 `extract()` accepts no validators and cannot report *not modified*

The shape that composes with what already exists: SR-A2 already makes `extract()`
a **no-op when the candidate carries content**. So the natural form is an
extension of that rule, not a new one.

- The caller passes the candidate **with** its stored `ContentSlot` — content,
  `etag`, `last_modified` — plus an explicit opt-in (see below).
- `extract()` sends the conditional request.
- `304` → return the candidate **untouched**, `extraction_status` = `unchanged`,
  `origin` = `revalidated`. The caller's copy is confirmed good.
- `200` → replace content as today, storing the new validators off the response.
- No validator supplied, or opt-in absent → byte-for-byte today's behaviour.

**The opt-in is load-bearing and must not be inferred.** SR-A2's existing no-op
means "candidate has content → don't fetch." Conditional revalidation means
"candidate has content → fetch anyway, conditionally." Those are opposite
behaviours triggered by the same state, so a caller must say which it wants. A
`revalidate: bool = False` parameter, defaulting to today's meaning. Inferring it
from the presence of an `etag` would silently turn every content-carrying
candidate into a network call — including Tavily's, whose content arrived with
the search response and has nothing to revalidate against.

### 3.5 `WebFetchTool` input is URL-only

The tool border needs to accept the validators and report the outcome. Structure
already rides `metadata` under `SEARCH_RESULTS_METADATA_KEY` (D22), so the
*result* side is a matter of the status value arriving; the *input* side needs
schema fields. Both additive, per D13's additive-within-a-minor rule.

## 4. Sequencing

1. `media-contracts`: `EXTRACTION_STATUS_UNCHANGED` + vocabulary test. Nothing
   else can reference it until this lands.
2. `search` contracts: `ContentSlot` validator fields and the `origin` value;
   `HeavyFetcher.fetch_rendered` gains `headers`.
3. `search`: `extract()` opt-in, conditional request, `304` handling. Update
   every `EXTRACTION_STATUS_COMPLETE` reader found in 3.2.
4. `agent-tools`: `WebFetchTool` input schema.
5. `scrape`: store validators per target and pass them on re-fetch. **This step
   is where the payoff actually lands** — steps 1-4 build a capability nobody
   is yet using. No target stores a validator today; that is a schema change and
   a migration, and it is the scrape pipeline's call whether to take it.

Steps 1-4 are additive and independently releasable. Step 5 is a separate
decision with a separate owner.

## 5. Tests the build owes

- A `304` over a real socket (`LocalHttpServer`, the helper #321 published)
  through `StandaloneTransport` → `extract()` → `unchanged`, with the caller's
  content returned untouched and unmodified.
- A `200` after a `304` — validator changed, content replaced, new validators
  stored.
- **No validator supplied → the request carries no conditional headers.** The
  regression that matters: an unconditional fetch that silently became
  conditional would return `304` to a caller with no copy, i.e. nothing.
- Opt-in absent + content present → no network call at all (SR-A2 unchanged).
- A `HeavyFetcher` that ignores `headers` never reports `unchanged` (3.1).
- Every `EXTRACTION_STATUS_COMPLETE` reader, driven with `unchanged`, does not
  report failure.
- A malformed or absent `ETag` on a `200` leaves the slot's validator `None`
  rather than storing `""` — an empty validator echoed back matches nothing and
  would make every subsequent request unconditional-but-slower.

## 6. Explicitly out of scope

- **Any stored cache.** D14 stands. If a future change wants one, it goes
  through SR-M2/SR-O3 and it is a `BaseCollection` or it is a violation.
- **The search path.** Provider queries are not conditionalised — see SR-M4.
- **`_verify_candidate_page` in `scrape`.** It runs once per candidate at
  discovery time and holds no copy; a `304` would leave it nothing to inspect.
  Its docstring already records this so the next reader does not re-derive it.
