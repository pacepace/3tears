# scrape-task-04: Multi-document (listing → N notices) driver + sidecar browser-download mode

**Status:** DESIGN, not yet approved to build — presenting for review before any code, per this
project's own "Sequence strictly" precedent (scrape-task-01 through 03 each got explicit
go-ahead before implementation started).
**Driver:** faidh's WARN Act push (2026-07-15) found two states -- West Virginia and Hawaii --
that publish one PDF per individual notice with no consolidated current-period listing, a
materially different shape from every driver this package has today.

**Revision note:** the first version of this doc scoped West Virginia OUT as blocked by a real
Cloudflare managed challenge. Direct user pushback ("isn't the determination for how to
display/download a setting in chrome?") was correct and changed the design -- verified live
against the real sidecar container (see "Verified, not assumed" below) that Chrome's
PDF-viewer-vs-download behavior is exactly a browser preference
(`plugins.always_open_pdf_externally`) plus a CDP command (`Browser.setDownloadBehavior`), not
an unsolvable bot-challenge problem. A real, passing browser session downloads the genuine file;
no active challenge-solving is involved. West Virginia is back in scope.

---

## The requirement

Two distinct problems, both real:

1. **Multi-hop fetch:** a listing page (or API) enumerates N documents; each document is one
   complete notice (not a row in a table). No existing driver turns "N separate document fetches"
   into the one `RenderedPage` the eval loop expects.
2. **Browser-backed document fetch:** West Virginia's individual PDF downloads sit behind a real
   Cloudflare managed challenge that blocks any plain HTTP client (verified: `cf-mitigated:
   challenge`, a "Just a moment..." JS-solve page) -- only a genuine browser session gets past it,
   and even then Chrome's own built-in PDF viewer intercepts the navigation before any bytes are
   accessible, unless download behavior is explicitly forced.

## Verified, not assumed — real research this session, including live sidecar testing

| State | Listing | Individual document fetch |
|---|---|---|
| **Hawaii** | Plain `httpx` GET on `labor.hawaii.gov/wdc/2026-warn-notices/` -- 200, real content, 19 real PDF links directly in the HTML. **No browser needed anywhere in the chain.** | Plain `httpx` GET -- 200, real PDF bytes (619KB verified). **No browser needed.** |
| **West Virginia** | `workforcewv.org/wp-json/wp/v2/media?search=WARN` -- a **WordPress REST API endpoint, NOT behind Cloudflare** (verified: 200, clean JSON, `title`/`source_url`/`date` per item). The browsable listing page itself (`/job-seeker/layoffs-downsizing/warn-listing/`) IS challenged, but this API makes that page unnecessary entirely. | `source_url` (`/wp-content/uploads/...`) IS behind the real Cloudflare challenge for a plain HTTP client. **Solved and live-verified**: a real nodriver browser session, launched with the Chrome preference `plugins.always_open_pdf_externally: true` set in its profile before start, plus `Browser.setDownloadBehavior(behavior="allow", download_path=...)` before navigating, passes the Cloudflare challenge (genuine browser JS execution -- no active solving) and Chrome performs a REAL file download instead of opening its internal viewer. Verified end-to-end: downloaded a real 155KB PDF from West Virginia's actual protected URL, parsed it through the unmodified `parse_document`, got a real, complete WARN notice letter (Conduent Commercial Solutions, real dates, real employee/location detail). |

**Both states are in scope.** Two failed intermediate attempts, recorded because they informed
the final design: `Page.setDownloadBehavior` alone (deprecated, tab-scoped) did nothing; a fresh
browser launched with `--disable-extensions` did nothing (Chrome's PDF viewer is a built-in
component, not a regular extension `--disable-extensions` touches) -- only the actual Preferences
file (`plugins.always_open_pdf_externally`) written before Chrome starts worked.

---

## Design

### Two-axis composition, not one monolithic driver

`MultiDocumentDriver` handles "listing → N document URLs → combine," with two independently
pluggable strategies, matching how heterogeneous these two real targets already are:

**Discovery strategy — how to find the N document URLs:**
- **HTML link mode** (Hawaii): fetch the listing as HTML, `BeautifulSoup.select(link_selector)`,
  resolve `href`s to absolute URLs.
- **JSON API mode** (West Virginia): fetch the listing as JSON, reuse `ApiDriver`'s own
  `_resolve_path(data, results_path)` helper (already handles dotted-path and empty-string-root
  cases) to get the records list, extract each record's `fragment_field`-named key as the document
  URL (`source_url` for West Virginia's WordPress API) -- deliberately reuses `ApiDriver`'s
  existing helper rather than writing a second JSON-path walker.

**Fetch strategy — how to retrieve one document's bytes, injected per instance:**
- **Plain fetch** (Hawaii): the existing, unmodified `DocumentDriver`.
- **Browser-download fetch** (West Virginia): a new `NodriverDownloadDriver` (see below), for
  documents behind a real bot challenge.

```python
class MultiDocumentDriver(ScrapeDriver):
    def __init__(
        self, *, document_driver: ScrapeDriver, client: httpx.AsyncClient | None = None,
        max_documents: int = 20,
    ) -> None: ...

    async def render(
        self, url: str, *, timeout: float = 30.0, wait_for: str | None = None,
        capture_network: bool = False, nav_steps: list[NavStep] | None = None,
        results_path: str | None = None, fragment_field: str | None = None,
        link_selector: str | None = None,
    ) -> RenderedPage:
        """
        Discovery: link_selector set -> HTML mode. results_path/fragment_field set (results_path
        may be "" for a root-array response, matching ApiDriver's own convention) -> JSON mode.
        Exactly one of the two must be configured -- MultiDocumentDriverError otherwise.
        """
```

### Flow (both modes converge after URL discovery)

1. Fetch the listing (`httpx` GET, plain -- neither Hawaii's HTML page nor West Virginia's
   `wp-json` endpoint needs a browser for the LISTING step, only West Virginia's individual
   document fetch does).
2. Resolve the first `max_documents` document URLs (assumes newest-first order, the same
   assumption `warn_act_or`'s own design already live-verified and relies on).
3. For each, `await self._document_driver.render(doc_url)` -- the injected fetch strategy. One
   document's failure is logged and skipped, never fatal to the whole call (mirrors
   `_regenerate_row_recipe`'s own per-candidate resilience).
4. Extract each successful `RenderedPage.html`'s `<body>` inner content, wrap in a delimiting
   `<div class="notice">...</div>`, concatenate into one combined page.
5. Return one `RenderedPage`. From here, **completely unmodified** existing machinery:
   `extraction_strategy_type: regex`, `multi_row: true` (Pennsylvania/Michigan's own precedent) --
   one regex pattern learned once against the combined page, cached as a normal `ScrapeRecipe`,
   reused every later poll with zero further LLM calls.

### New sidecar capability: browser-forced download

**Sidecar startup (`main.py`'s `_lifespan`):** pin `user_data_dir` to a fixed path (today it's
auto-generated/random) and write `{"plugins": {"always_open_pdf_externally": true}}` into
`<user_data_dir>/Default/Preferences` before `uc.start()` -- verified this only affects direct
navigation to a PDF response; normal HTML page rendering is unaffected (nothing about a page that
merely *links* to a PDF triggers this pref).

**New endpoint** `POST /v1/download` (a distinct contract from `/v1/render` -- the response shape
is fundamentally different, raw file bytes vs. HTML, not another optional field bolted onto the
existing one):
1. `Target.createBrowserContext()` -- an isolated context per request, so
   `Browser.setDownloadBehavior`'s `browser_context_id` scoping means concurrent `/v1/download`
   calls never race each other's download directories (verified this parameter exists and is
   plumbed through nodriver's own CDP wrapper; live concurrency test is an implementation-time
   task, not just a design-time read of the function signature).
2. Fresh temp download directory per request (`tempfile.mkdtemp()`), passed to
   `Browser.setDownloadBehavior(behavior="allow", download_path=..., browser_context_id=...)`.
3. Open a tab within that context, navigate to the target URL.
4. Poll the download directory (bounded timeout, matching `_render`'s own settle-wait shape) for a
   completed file (not a `.crdownload` in-progress marker).
5. Read the file, base64-encode it into the response body, clean up the temp directory and
   browser context.
6. Same fail-open/error-code discipline as `/v1/render` (`DownloadResponse{status, content_type,
   filename, content_base64}` on success; the existing `{"error": {"code", "message"}}` shape on
   failure -- new codes: `download_timeout`, `download_failed`).

**New driver (3tears-scrape side):** `NodriverDownloadDriver(ScrapeDriver)` -- calls the sidecar's
`/v1/download`, base64-decodes the body, and runs it through **the same parse-and-convert logic
`DocumentDriver` already has** (refactor: extract `DocumentDriver`'s "bytes + filename → parsed →
synthetic HTML" body into a shared module-level helper both drivers call, so this isn't a second
copy of that logic). `driver.name == "nodriver_download"`.

---

## Files to create/modify

- `packages/scrape/sidecar/main.py` -- pinned `user_data_dir` + Preferences file write in
  `_lifespan`; new `POST /v1/download` endpoint; new `DownloadRequest`/`DownloadResponse` models;
  new error codes.
- `packages/scrape/sidecar/tests/test_render_contract.py` -- new tests for the download endpoint
  (mocked CDP calls, matching this file's own existing convention).
- `packages/scrape/src/threetears/scrape/drivers/nodriver_download.py` -- `NodriverDownloadDriver`.
- `packages/scrape/src/threetears/scrape/drivers/document.py` -- extract the shared
  bytes-to-synthetic-HTML helper for reuse (behavior-preserving refactor, existing tests must
  still pass unchanged).
- `packages/scrape/src/threetears/scrape/drivers/multi_document.py` -- `MultiDocumentDriver`,
  `MultiDocumentDriverError`.
- `packages/scrape/src/threetears/scrape/driver.py` -- add `link_selector: str | None = None` to
  `ScrapeDriver.render()`'s abstract signature; every existing driver accepts and ignores it,
  matching `results_path`/`fragment_field`'s own precedent.
- `packages/scrape/src/threetears/scrape/collections.py` -- `ScrapeTarget.link_selector` property.
- New unit tests for each new driver; sidecar tests for the download endpoint.
- faidh's `src/faidh/tools/scrape_tool.py` -- `_build_drivers()` gains `multi_document` and
  `nodriver_download` entries if `ScrapeTool`'s ad-hoc MCP surface should expose them too (confirm
  scope during implementation; the WARN Act targets themselves drive through
  `poll_scrape_targets`, not `ScrapeTool`).
- faidh's `warn_act_targets.yaml` -- `warn_act_hi` (HTML/link_selector mode, plain `DocumentDriver`
  fetch) and `warn_act_wv` (JSON/results_path mode, `NodriverDownloadDriver` fetch).

## Anti-patterns

- **DO NOT build active Cloudflare-challenge-solving.** Not needed and not what this is --
  a real, passing browser session gets through Cloudflare's managed challenge on its own; the new
  capability is "get real bytes out of a passing session," not "defeat the challenge."
- **DO NOT let `Browser.setDownloadBehavior` be global/unscoped.** Isolated browser context per
  `/v1/download` request, or concurrent requests race each other's download directories.
- **DO NOT duplicate `DocumentDriver`'s parse-and-convert logic in the new download driver.**
  Extract and share it.
- **DO NOT let one bad document fetch fail `MultiDocumentDriver`'s whole render.** Skip and log.
- **DO NOT add a `max_documents` YAML field speculatively.** Fixed driver default until a real
  target needs otherwise.
- **DO NOT write a second JSON-path walker for the API discovery mode.** Reuse `ApiDriver`'s own
  `_resolve_path`.

## Acceptance Criteria

- [x] `MultiDocumentDriver` in HTML mode, live-verified against Hawaii end-to-end (real fetch,
      real per-document extraction -- **not** regex as originally planned, see Outcome below --
      real structural validation, real recipe persistence).
- [x] `MultiDocumentDriver` in JSON mode + `NodriverDownloadDriver`, live-verified against West
      Virginia end-to-end, including a real download through the real Cloudflare challenge.
- [x] `/v1/download` concurrency-safe -- a live test with ≥2 concurrent requests for different
      URLs, confirming no cross-request file/directory collisions.
- [x] A single document's fetch failure doesn't fail the whole `MultiDocumentDriver.render()` --
      unit-tested.
- [x] Every existing driver's `render()` accepts and ignores `link_selector` -- confirmed via the
      shared contract test or an equivalent per-driver check.
- [x] `DocumentDriver`'s existing test suite passes unchanged after the shared-helper refactor
      (behavior-preserving, not a rewrite).
- [x] mypy/ruff clean on both the package and the sidecar, full test suites green, zero new
      pre-existing-baseline drift.

## Outcome (scrape-task-05/06, 2026-07-15/16 -- both states shipped, design revised twice more)

This design's own regex assumption didn't survive contact with real data, twice:

1. **regex → per_document (scrape-task-05).** West Virginia's real letters are independently
   worded per employer, sharing no boilerplate a page-wide cached pattern could ever generalize
   across (live-verified: a regex candidate matched 1 of 10 real documents). `eval_loop.py` grew a
   third `StrategyType`, `"per_document"` -- no cached recipe, a fresh LLM extraction call per
   document every poll (`extraction.extract_fields_directly`, later
   `extract_fields_directly_chunked` once a single call-for-every-field also proved less reliable
   than several smaller calls). Both states use it.
2. **OCR'd text → vision (scrape-task-06).** Hawaii's real PDFs are ~90% scanned images; OCR
   (Tesseract, PSM 4) recovers most fields but a full-set live comparison against ALL of West
   Virginia's documents found OCR'd text measurably less reliable overall (2/10 complete records
   vs. a vision-capable model reading the page images directly, 10/10, same documents). Routing is
   now by document shape, not by state: `RenderedPage.was_ocr` (set by `DocumentDriver`/
   `NodriverDownloadDriver`) flows through `MultiDocumentDriver`'s own `data-was-ocr` attribute to
   `eval_loop._run_per_document_extraction`, which picks `extract_fields_from_images`
   (`anthropic/claude-sonnet-5` via OpenRouter -- no new API key needed) for a scanned document and
   `extract_fields_directly_chunked` for a born-digital one. Hawaii's own `affected_count` field
   was dropped from its schema after vision confirmed (not just OCR failing to find) the real
   source redacts or omits it in three distinct, live-verified ways -- New Jersey's own
   missing-field precedent, not a gap this rearchitecture claims to close. See backlog `SCR-8M3H`
   for the real remaining gap (table-structure recognition from scanned images) and its trigger
   condition.

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears-scrape
uv run pytest packages/scrape/tests/test_driver_multi_document.py packages/scrape/tests/test_driver_nodriver_download.py packages/scrape/tests/test_driver_document.py -q
uv run pytest packages/scrape --ignore=packages/scrape/sidecar -q -m "not live"
cd packages/scrape/sidecar && uv run pytest -q
uv run ruff check packages/scrape/ && uv run mypy
docker buildx bake nodriver-sidecar   # confirm the sidecar image still builds after the changes
```
