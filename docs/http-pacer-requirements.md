# In-process request pacer and 429 wait for `threetears.core.http_client`: request (not yet scoped)

Status: queued, not started. Recorded 2026-11-26 from Hearsay's request (owner-approved 2026-11-25). Lower priority than
the model key pool (`docs/model-key-pool-requirements.md`). The claims about existing 3tears code below are Hearsay's,
not yet checked against the code.

## Why

Hearsay's seed fetcher reads two rate-limited APIs through `threetears.core.http_client.TracedHttpClient`:
- **Discogs:** 25 requests/min unauthenticated, 60 with a token.
- **MusicBrainz:** 1 request/s.

The client already keeps response bodies exactly as returned and retries 5xx. Hearsay says 3tears has no pacer that
works inside one process:
- `TokenBucket` needs NATS KV.
- The scrape `ApiDriver` rewrites JSON into HTML.

So Hearsay carries its own small `Pacer` and 429 wait in `src/hearsay/seed/source.py`. The owner's rule is 3tears
first, so Hearsay switches to the 3tears one when it exists.

## Requested

1. **A rate per host, or per client.** N requests per period, enforced in-process, safe across concurrent async callers,
   with no external store.
2. **429 handling.**
   - Wait out `Retry-After`, in either form: seconds or an HTTP date.
   - When the header is absent, back off instead.
   - Give up after a configurable number of waits, raising a typed error that says how many waits were made. Never a
     silent None.
3. **Fits the existing client.** It composes with `TracedHttpClient`'s current retry and backoff for 5xx and connection
   errors, and leaves every 4xx except 429 to the caller.
4. **Tests with no network.** Pacing under concurrency, `Retry-After` in both forms, giving up after N waits.

## Before building

- Check Hearsay's claims about `TokenBucket`, `ApiDriver` and `TracedHttpClient`'s retry path against the code.
- Decide whether the pacer is per host, per client instance or both, and how it relates to the model key pool's
  rate-limit handling.

## When it ships

Reply to the Hearsay session with the version and the API.
