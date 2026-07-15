# scrape-task-02: Page-finding research agent

**Status:** APPROVED TO START (scrape-task-01 shipped and Critic-reviewed, `feature/scrape` commit
`7a52b50`). **Origin:** `docs/scrape-lift-design.md` D5. This doc is the "full design deferred until
after task 01 ships" the parent doc promised.
**Scope:** one new module in `packages/scrape/` — a bounded-turn agent that takes a plain-language
query ("Ohio WARN Act notices"), searches and fetches candidate pages, self-verifies the winner has
real extractable structure, and returns a `ScrapeTarget`-shaped result. Independently callable —
never a forced prerequisite to extraction (Design Rule 4).

---

## Objective

Give any 3tears-scrape caller a way to go from "find me the right page for X" to a verified
`(url, driver_backend guess, wait_for guess)` without a human first tracking down the URL by hand —
the gap that made faidh's `warn_act_targets.yaml` require a human to search-and-verify every one of
its ~24 rows before it could be added.

---

## What already exists (verified — reuse, don't rebuild)

Researched directly against this repo before writing this doc (`packages/agent/tools`,
`packages/scrape`, faidh's `query_agent`) rather than assumed:

| Need | Existing piece | Verdict |
|---|---|---|
| Bounded N-round tool-calling loop | `threetears.agent.tools.executor.ToolExecutor` (`packages/agent/tools/.../executor.py`) — plain `async def invoke_with_tools(chat_model, messages, service_tools) -> ToolExecutionResult`, loops `chat_model.ainvoke` + tool dispatch until a tool-call-free response or `max_rounds` exhausted | **Reuse as-is.** Currently has zero real callers anywhere in either repo (only its own unit test) — this is its first production use. |
| WebSearch tool | `threetears.agent.tools.builtin.web_search.WebSearchTool` + `create_web_search_tool(config, description) -> StructuredTool` (SearXNG-backed, needs `config["base_url"]`) | **Reuse as-is.** faidh's `query_agent.nodes.web_search_node` already uses `WebSearchTool` directly (not as a bound tool) — same instance shape, different calling convention. |
| WebFetch tool (self-verification) | `threetears.agent.tools.builtin.web_fetch.WebFetchTool` + `create_web_fetch_tool(config, description) -> StructuredTool` (httpx + trafilatura, redirect/meta-refresh handling, 429/403 single retry) | **Reuse as-is for the agent's own in-loop fetches** (this will be its first real consumer anywhere). **NOT reused for the deterministic structural-verification step** — see below, trafilatura's readable-text extraction strips the `<table>`/link markup that step needs to inspect. |
| Tool → LangChain adapter | `threetears.agent.tools.langchain_adapter.to_langchain_tool(tool, description=None, args_schema=None) -> StructuredTool` | **Reuse as-is.** `create_web_search_tool`/`create_web_fetch_tool` already call it internally. |
| Coercing free text into a structured result | `threetears.scrape.llm_retry.bounded_retry_structured_call` (single structured-output call, bounded retries, degrades to `None` rather than raising) | **Reuse as-is** for the "turn the agent's final free-text answer into a `_CandidatePage` object" step — same infra `extraction.py`/`eval_loop.py`/`enrichment.py` already use. |
| Target output shape | `threetears.scrape.collections.ScrapeTarget` (`url`, `driver_backend`, `wait_for`, ...) | The shape this feature *produces a guess toward* — this task does not persist a `ScrapeTarget` itself, it hands back plain data a caller turns into one (see Anti-patterns). |

**Important contract detail, verified by reading `executor.py` directly (not assumed from its
docstring):** `ToolExecutor.invoke_with_tools` calls `chat_model.ainvoke(messages)` — it does
**not** call `.bind_tools()` itself. The caller must pass a `chat_model` already bound:
`create_chat_model(...).bind_tools([search_lc_tool, fetch_lc_tool])`, then pass the *same* two
`StructuredTool` instances as `service_tools`.

No LangGraph self-loop precedent exists anywhere in either repo (`grep` for
`max_turns`/`turn_count`/a conditional edge back to the same node returns nothing) — building this
as a plain async function calling `ToolExecutor` is both the simplest option and the only one with
existing precedent to follow. Skip LangGraph entirely for this task.

---

## Design

### Module: `packages/scrape/src/threetears/scrape/page_finder.py`

```python
@dataclass
class PageFinderResult:
    url: str
    driver_backend: str          # best guess: "nodriver" | "document" | "api" (never camoufox/network_capture — see below)
    wait_for: str | None
    verified: bool               # True iff _verify_candidate_page found real structure
    verification_note: str       # what verification found, or why it failed
    reasoning: str                # the agent's own final free-text summary, kept for operator review
    turns_used: int
    search_queries_tried: list[str]


async def find_target_page(
    query: str,
    *,
    api_key: str,
    searxng_url: str,
    model_id: str = DEFAULT_PAGE_FINDER_MODEL_ID,
    max_turns: int = 6,
) -> PageFinderResult:
    ...
```

**Flow** (mirrors `eval_loop.py`'s own propose → structural-check → judge shape, one stage removed):

1. **Search-and-fetch loop.** Build `web_search_lc = create_web_search_tool({"base_url": searxng_url}, ...)`, `web_fetch_lc = create_web_fetch_tool({}, ...)`. `chat_model = create_chat_model(model_id, api_key=api_key, purpose=LlmPurpose.TOOL_SELECTION).bind_tools([web_search_lc, web_fetch_lc])`. Seed `messages` with a system/human prompt: find the real page for `{query}`; use `web_search` for candidates, `web_fetch` to inspect one *before* concluding — never conclude from a search snippet alone. Run `ToolExecutor(max_rounds=max_turns).invoke_with_tools(chat_model, messages, [web_search_lc, web_fetch_lc])`.
2. **Structured coercion.** The loop's `ToolExecutionResult.output` is free text (a URL plus reasoning, not guaranteed parseable). Feed it through `bounded_retry_structured_call(prompt=..., response_model=_CandidatePage, model_id=..., api_key=..., purpose=LlmPurpose.EXTRACTION, ...)` where `_CandidatePage` is a small local Pydantic model (`url: str`, `driver_backend_guess: str | None`, `wait_for_guess: str | None`, `summary: str`). `is_acceptable=lambda c: bool(c.url and c.url.startswith(("http://", "https://")))`.
3. **Deterministic structural verification** (`_verify_candidate_page`, new, no LLM call): a direct `httpx.get(candidate.url, follow_redirects=True)` (no sidecar, no browser — this task stays independently callable without a running nodriver container, Design Rule 4). Inspect the raw response with `BeautifulSoup` (already a package dependency) for, in order: (a) a `<table>` with ≥2 `<tr>` rows → `driver_backend="nodriver"`, matches most onboarded WARN targets today; (b) an `<a href>` ending in `.pdf`/`.doc`/`.docx`/`.xlsx`/`.csv` → `driver_backend="document"`; (c) a JSON `Content-Type` whose body is a list or has an obvious list-valued key → `driver_backend="api"`. None found → `verified=False`, `driver_backend` falls back to the agent's own `driver_backend_guess` (or `"nodriver"` if that's also empty), `verification_note` explains what was checked and found nothing. This is a real, literal reading of D5's own wording ("check for real structure -- a table, a document link, *something*"), not an approximation of it.
4. **Never verified to `camoufox`/`network_capture`.** Both require knowledge this stateless check can't produce (`camoufox`: does the page need JS rendering nodriver's plainer fetch can't get? `network_capture`: does the real data arrive via an authenticated in-session XHR?). If structural verification fails and the agent's own guess doesn't name one of the three checkable backends, default to `"nodriver"` and leave `verified=False` — an operator or the eval loop's own empty-extraction signal surfaces the mismatch downstream, the same way any manually-onboarded target's driver-backend mistake would.
5. Return `PageFinderResult`. Turn exhaustion without a resolvable candidate still returns a result (`verified=False`, best-guess fields, `reasoning` explaining the loop ran out of turns) — "surface for review, never silently drop," the same discipline `ScrapeExtraction.validation_status` already establishes for the extraction path (see the package README's §4).

### Why a raw `httpx.get`, not `WebFetchTool`, for verification

`WebFetchTool` exists precisely to give the *agent* readable content to reason about mid-loop — that's the right tool for step 1. Verification in step 3 needs the opposite: raw structural markup (`<table>`, `<a href>`), which `trafilatura.extract()` is designed to strip away in favor of prose. Reusing `WebFetchTool` there would silently defeat the verification it's supposed to perform. A second, purpose-built fetch is correct here, not redundant — same reasoning `ApiDriver`/`DocumentDriver` already apply (each does its own stateless direct fetch rather than routing through a shared "generic fetch" abstraction).

### `DEFAULT_PAGE_FINDER_MODEL_ID`

New module-level constant, following `extraction.py`'s own `DEFAULT_EXTRACTION_MODEL_ID` precedent (currently `"deepseek/deepseek-chat-v3-0324"`) — reuse the same value unless a tool-calling-specific reason to diverge turns up during implementation (note it in the module docstring if so, don't silently pick a different model).

---

## Files to create

- `packages/scrape/src/threetears/scrape/page_finder.py` — `PageFinderResult`, `_CandidatePage`, `find_target_page()`, `_verify_candidate_page()`.
- `packages/scrape/tests/test_page_finder.py` — unit tests, all three stages mocked/faked independently (ToolExecutor's loop, the structured-coercion call, and the verification HTTP fetch), plus one test exercising the full `find_target_page()` composition with all three faked.

## Files to modify

- `packages/scrape/pyproject.toml` — add `3tears-agent-tools` if not already a dependency (it already is, per task-01's own pyproject — confirm, don't assume, at implementation time) and confirm `httpx`/`beautifulsoup4` (already present) cover verification's needs.
- `packages/scrape/README.md` — new bullet under "What you get" once shipped.

---

## Anti-patterns

- **DO NOT persist a `ScrapeTarget` from inside `find_target_page()`.** It returns plain data; a
  caller decides whether/how to turn that into a real target row (mirrors `ScrapeTool`'s own
  constructor-injection philosophy — no hidden I/O a caller didn't ask for).
- **DO NOT force page-finder → extraction into one pipeline.** `find_target_page()` takes a query,
  returns a result — nothing wires it into `run_eval_loop`/`run_eval_loop_multi_row` automatically.
  A caller with a known URL never has to touch this module at all (Design Rule 4, restated in the
  parent design's D5 verbatim).
- **DO NOT reuse `WebFetchTool`'s trafilatura-extracted content for structural verification.** It
  strips exactly the markup verification needs (see "Why a raw httpx.get" above).
- **DO NOT guess `camoufox` or `network_capture` as a driver-backend output.** Neither is
  verifiable by a stateless check; guessing either would be worse than defaulting to `"nodriver"`
  and flagging `verified=False`.
- **DO NOT build this as a LangGraph node/graph.** No precedent for a bounded self-loop exists in
  either repo; `ToolExecutor` already covers exactly this shape as a plain async function.
- **DO NOT let a turn-exhausted or failed-verification run silently return nothing.** Always return
  a `PageFinderResult` with `verified=False` and an honest `reasoning`/`verification_note` — never
  raise, never return `None` (matches `bounded_retry_structured_call`'s own "degrade, don't raise"
  contract, which this module builds directly on top of).

---

## Acceptance Criteria

- [ ] `find_target_page(query, ...)` runs a real bounded WebSearch/WebFetch loop (verified live, at
      least one real query against a real SearXNG instance) and returns a `PageFinderResult`.
      **NOT DONE — flagged, not silently skipped:** this build environment has no local SearXNG
      instance and no live OpenRouter API key, so the search-loop half of the design could not be
      exercised live. The composition logic is covered by unit tests with the loop faked (see
      below); the *loop itself* (`ToolExecutor`) has its own passing test suite
      (`packages/agent/tools/tests/test_executor.py`), and `WebSearchTool`/`WebFetchTool` are each
      independently tested in their own package. What's unverified specifically is the real,
      end-to-end wiring of all three together against live services. Needs a live pass with a real
      SearXNG URL + API key before this checkbox can be honestly marked done.
- [x] Turn cap is real and enforced — a query engineered to never converge exhausts `max_turns` and
      still returns a `PageFinderResult` (not an exception, not a hang). Unit-tested
      (`test_turn_exhaustion_with_no_usable_output_returns_honest_result`).
- [x] `_verify_candidate_page` correctly identifies at least: a real HTML table page, a page whose
      only content is a PDF/DOCX link, and a page with neither. **Live-verified**, not just
      mocked: a real fetch of Maryland's and New York's actual WARN Act pages (both correctly
      identified as `nodriver`+verified) and `example.com` (correctly identified as
      unverified/no structure). This live pass caught and fixed a real ordering bug — a page can
      carry both a real notices table AND an unrelated incidental PDF link elsewhere (Maryland's
      real page does); the original document-link check ran before the table check and
      misclassified it. Fixed (table now checked first) and regression-tested
      (`test_table_wins_over_an_incidental_document_link_on_the_same_page`).
- [x] Calling `find_target_page()` never requires a running nodriver sidecar — confirmed via grep,
      no import or call path in this module reaches `NodriverSidecarDriver`/anything
      sidecar-dependent.
- [x] Unit tests cover: loop convergence on a valid candidate, turn exhaustion, structural
      verification hitting each of its three positive branches plus the none-found fallback, and the
      full composition with all three stages faked. 17 tests, `packages/scrape/tests/test_page_finder.py`.
- [x] `packages/scrape/README.md` documents the new capability under "What you get."
- [x] mypy/ruff clean (142 pre-existing, unrelated errors unchanged; zero new), full package test
      suite green (276/276, including the 17 new).

**Critic-caught and fixed before ship (chunk-mode review):** `_extract_search_queries` originally
filtered `tool_calls_made` on the bare string `"web_search"`, but `WebSearchTool.mcp_name()` (the
name `ToolExecutor` actually records) is `"threetears.web_search"` -- the filter never matched, so
`PageFinderResult.search_queries_tried` was always `[]` in production despite both unit tests
passing (they'd fabricated `tool_calls_made` entries using the wrong bare name, a tautological
mock the Critic specifically flagged). Fixed: `_extract_search_queries` now takes the search tool's
real bound `.name` as a parameter rather than hardcoding a guess; both tests updated to the real
name, plus a new regression test (`test_bare_web_search_name_does_not_match_the_real_bound_name`)
proving the bare string is deliberately rejected. Live-confirmed:
`create_web_search_tool(...).name == "threetears.web_search"` matches exactly what the fix now
filters on. Critic also flagged (not blocking) that this dotted name should be live-confirmed
against a real OpenAI-compatible tool-calling API before ship, since some providers restrict tool
names to `[a-zA-Z0-9_-]` -- rolled into the same live-verification gap noted above.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears-scrape
uv run pytest packages/scrape/tests/test_page_finder.py -q
uv run pytest packages/scrape/tests/ -q -m "not live"
uv run ruff check packages/scrape/ && uv run mypy   # workspace mypy, per task-01's own files/mypy_path convention
```
