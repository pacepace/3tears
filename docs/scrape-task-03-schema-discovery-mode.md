# scrape-task-03: Schema-discovery mode

**Status:** APPROVED TO START (scrape-task-01 and scrape-task-02 shipped and Critic-reviewed, commits
`7a52b50`/`d5b2e3b` on `feature/scrape`). **Origin:** `docs/scrape-lift-design.md` D6. This doc is
the "full design deferred until after task 01 ships" the parent doc promised.
**Scope:** "capture every variable on this page" -- given only HTML (no caller-supplied
`field_schema`), propose field names/types/selectors, validate them against the real page the same
way a schema-driven candidate is validated today, and return what's genuinely there for a human or
caller to review. Independently callable -- never a forced prerequisite, never auto-creates a
`ScrapeTarget`/`ScrapeRecipe`.

---

## A deliberate placement deviation from D6's literal wording (stated up front, not buried)

D6 says: *"The existing propose→validate→judge extraction mechanism gains a second mode... using
the same structured-LLM-call infrastructure (`bounded_retry_structured_call`) the schema-consuming
candidate generators already use, just inverted."* Read literally, "the existing mechanism" could
mean `eval_loop.py`'s `run_eval_loop`/`run_eval_loop_multi_row` -- the orchestration functions that
actually run propose→validate→judge→persist end to end.

Having read both `extraction.py` and `eval_loop.py` in full before designing this (not assumed):
**`run_eval_loop`/`run_eval_loop_multi_row` are recipe-lifecycle functions** -- they require a
`target_id`, a `recipe_collection`, an `extraction_collection`, and persist a `ScrapeRecipe` +
`ScrapeExtraction` row on every call. Schema discovery is a **pre-onboarding** operation: a caller
runs it *before* a `ScrapeTarget` exists, to decide what `field_schema` to configure in the first
place. Threading a `schema: FieldSchema | None = None` branch into the recipe-persistence functions
would mean bolting a "skip persistence, there's no target yet" escape hatch onto functions whose
entire job is persistence -- real complexity for a fit that doesn't match the operation's own
lifecycle stage.

**Decision: implement discovery as new sibling functions in `extraction.py`** (`discover_candidates`
/ `discover_row_candidates`, mirroring `generate_candidates`/`generate_row_candidates`'s existing
split), reusing `validate_candidate`/`validate_row_candidate` **completely unchanged** and
`bounded_retry_structured_call` exactly as every other candidate generator does. This is the "same
infrastructure, just inverted" D6 asks for, applied at the layer where "propose" and "validate"
actually live -- `eval_loop.py` is untouched by this task, matching Design Rule 4 (independently
callable, never a forced pipeline) more precisely than bolting a mode flag onto a persistence
function would. Flagging this as a real design decision, not silently deviating from the parent doc.

---

## Design

### Two new functions in `extraction.py` (mirroring the existing single/row split)

```python
class DiscoveredField:
    name: str
    python_type: type          # decoded from a constrained "str"|"int"|"float"|"bool" LLM response
    selector: str
    sample_value: Any          # the real validated value from THIS page, not an LLM guess

@dataclass
class DiscoverySchemaResult:
    validated: bool                       # True iff at least one field structurally validated
    fields: list[DiscoveredField]          # only fields that validated -- never a hallucinated field
    field_schema: FieldSchema              # {field.name: field.python_type for field in fields} -- ready to hand
                                            # straight to a ScrapeTarget.field_schema / run_eval_loop call
    strategy: dict[str, Any]               # the winning selector strategy -- same shape run_eval_loop's
                                            # own ScrapeRecipe.extraction_strategy already uses, so a caller
                                            # who likes the result can seed a recipe with zero re-derivation
    sample_records: list[dict[str, Any]]   # 1 record (discover_candidates) or up to N (discover_row_candidates)


async def discover_candidates(
    html: str, *, n: int = 3, model_id: str = DEFAULT_EXTRACTION_MODEL_ID, api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS, backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> DiscoverySchemaResult: ...

async def discover_row_candidates(
    html: str, *, n: int = 3, model_id: str = DEFAULT_EXTRACTION_MODEL_ID, api_key: str,
    attempts: int = _EXTRACTION_ATTEMPTS, backoff_seconds: float = _EXTRACTION_BACKOFF_SECONDS,
) -> DiscoverySchemaResult: ...
```

**Flow** (`discover_candidates`, single-record; `discover_row_candidates` is the direct multi-row
counterpart, same shape as `generate_candidates`/`generate_row_candidates`'s own split):

1. **Discovery prompt.** Same `strip_boilerplate` + `MAX_HTML_CHARS_IN_PROMPT` truncation
   `_build_candidate_prompt` already uses -- but instead of "here are the fields, propose selectors
   for them," the prompt is "identify every distinct, genuinely useful field of structured data on
   this page (e.g. a table's columns) and propose a name, an inferred type
   (`str`/`int`/`float`/`bool` only), and a CSS selector for each." Forced response shape:
   `_DiscoveredCandidate(fields: list[_DiscoveredFieldProposal])`, `_DiscoveredFieldProposal(name,
   type_name: Literal["str","int","float","bool"], selector, sample_value_hint: str)`. `n` candidate
   proposals requested per call, same as `generate_candidates`' own `n` parameter.
2. **Derive schema + strategy per candidate, validate unchanged.** For each of the `n` proposals:
   `schema = {f.name: _DISCOVERY_TYPE_NAMES[f.type_name] for f in proposal.fields}`,
   `strategy = {f.name: f.selector for f in proposal.fields}`, then
   `validate_candidate(html, strategy, schema)` -- **the exact same function every schema-driven
   candidate already goes through**, zero modification.
3. **Selection: most fields validated wins, no judge.** Unlike the schema-known path, there is no
   external ground truth to judge semantic correctness against -- the LLM invented the fields, so
   "is this really the county field" isn't a question a judge call can meaningfully answer. Picking
   the candidate whose validation kept the most fields (a real, objective, comparable signal, the
   same kind of tiebreak `_regenerate_row_recipe`'s own `needs_review` fallback already uses via
   `max(survivors, key=lambda pair: len(pair[1].records))`) is honest and simple. If every candidate
   validates zero fields, `DiscoverySchemaResult(validated=False, fields=[], ...)` -- an honest empty,
   never a crash, matching every other degrade path in this module.
4. **No persistence.** Nothing is written to `ScrapeRecipeCollection`/`ScrapeExtractionCollection`.
   A caller who wants to act on the result constructs a `ScrapeTarget` (or seeds a first
   `ScrapeRecipe`) themselves, using `DiscoverySchemaResult.field_schema`/`.strategy` directly --
   both are already shaped exactly as `ScrapeTarget.field_schema` / `ScrapeRecipe.extraction_strategy`
   expect, so no caller-side re-derivation is needed.

### Local type-name mapping, not a cross-module import

`collections.py` already has an identical `_FIELD_SCHEMA_TYPE_NAMES: dict[str, type] = {"str": str,
"int": int, "float": float, "bool": bool}` for `decode_field_schema`/`encode_field_schema`.
`extraction.py` has zero dependency on `collections.py` today (verified: its only intra-package
import is `.llm_retry`) -- adding one for a 4-entry dict would be a real, avoidable new coupling.
`extraction.py` gets its own `_DISCOVERY_TYPE_NAMES`, the same 4 entries, kept independently (a
comment cross-references `collections.py`'s copy so a future type addition doesn't drift silently
without at least a pointer to update the other one).

### Explicitly out of scope for this task (disclosed, not silently dropped)

- **Regex/text-block discovery.** CSS-selector discovery only. A text-block page has no schema-free
  anchor a discovery prompt could reliably invent a *pattern* against (the regex candidate
  generators already need a known field list to build named groups around) -- meaningfully harder,
  and CSS/table pages are the dominant shape (23 of faidh's 24 onboarded WARN states use `"css"`).
  Flagged as real future work, not pretended-complete.
- **Auto-detecting single-record vs. multi-row page shape.** `discover_candidates` vs.
  `discover_row_candidates` stays a caller choice, exactly like `ScrapeTarget.multi_row` already is
  today (a per-target config value, never auto-detected by the eval loop). Detecting page shape is a
  different, unscoped inference problem.
- **Wiring discovery into `ScrapeTool`'s MCP surface.** `ScrapeTool` (task-01's own scope) stays
  schema-*required* for this task; exposing discovery as a callable MCP tool is a follow-up, not
  bundled here (D5/D6 were scoped as two SEPARATE capabilities, and task-02 didn't touch `ScrapeTool`
  either).

---

## Files to modify

- `packages/scrape/src/threetears/scrape/extraction.py` -- `DiscoveredField`, `DiscoverySchemaResult`,
  `_DiscoveredFieldProposal`, `_DiscoveredCandidate` (pydantic), `_DISCOVERY_TYPE_NAMES`,
  `_build_discovery_prompt`/`_build_row_discovery_prompt`, `discover_candidates`,
  `discover_row_candidates`. Added to `__all__`.
- `packages/scrape/tests/test_extraction.py` -- new test classes, same file (mirrors how
  `generate_candidates`/`generate_row_candidates` tests already live alongside `validate_candidate`'s
  own tests in this one file).
- `packages/scrape/README.md` -- new bullet under "What you get."

## Anti-patterns

- **DO NOT thread this into `eval_loop.py`'s recipe-persistence functions.** See the placement
  deviation section above -- this is a considered decision, not an oversight.
- **DO NOT invent a judge step for discovery.** There's no ground truth to judge against; "most
  fields validated" is the correct, honest selection signal (see Design §3).
- **DO NOT auto-persist a `ScrapeTarget`/`ScrapeRecipe` from a discovery result.** Plain data out,
  same as `find_target_page` (task-02) -- a caller decides what to do with it.
- **DO NOT silently fabricate a field that didn't structurally validate.** `DiscoverySchemaResult`
  only ever contains fields `validate_candidate`/`validate_row_candidate` actually confirmed match
  real page content -- an LLM-proposed field that didn't validate is dropped, not included with a
  caveat (matches every other candidate's own "structural validity is non-negotiable" discipline).
- **DO NOT build regex-shaped discovery as part of this task.** Explicitly descoped above.

## Acceptance Criteria

- [x] `discover_candidates(html, ...)` and `discover_row_candidates(html, ...)` both run a real
      discovery prompt and return a `DiscoverySchemaResult` whose `field_schema`/`strategy` are
      directly usable as `ScrapeTarget.field_schema`/`ScrapeRecipe.extraction_strategy` with zero
      caller-side transformation. **Live-verified**, not just mocked.
- [x] Both reuse `validate_candidate`/`validate_row_candidate` completely unchanged in their own
      logic -- confirmed via the diff neither function's *body* gained new branches (both did gain
      one call-site substitution, `soup.select_one`/`soup.select` -> `_select_one_safe`/
      `_select_safe`, a real bug fix -- see below -- not new validation behavior).
- [x] A discovered field that fails structural validation never appears in the result. Unit-tested
      for both the single-record and row paths.
- [x] Zero candidates validating any fields returns an honest `validated=False` empty result, never
      raises.
- [x] **Live-verified against a real, already-onboarded WARN Act page** (Maryland,
      `warn_act_md` in `warn_act_targets.yaml`) using a real OpenRouter key available in this build
      environment (`faidh/.env`'s `FAIDH_OPENROUTER_API_KEY`) -- this closes what task-02 had to
      leave as an honest gap; a real key was available for this task. One live run of
      `discover_row_candidates` against the real page returned 8 fully-validated fields
      (`notice_date`, `naics_code`, `company_name`, `location`, `local_area`, `total_employees`,
      `effective_date`, `type`) across all 80 real records on the page -- genuine overlap with
      `warn_act_targets.yaml`'s human-configured schema (`employer`≈`company_name`,
      `notice_date`, `effective_date`, `affected_count`≈`total_employees`, `county`≈`local_area`),
      discovered independently with no knowledge of that config.
      **Also observed, honestly reported, not hidden:** this specific model/provider combination
      (`deepseek/deepseek-chat-v3-0324` via OpenRouter) has an already-documented ~50%
      single-call structured-output failure rate (see `DEFAULT_EXTRACTION_MODEL_ID`'s own comment,
      predates this task) -- some discovery runs against the same real page returned zero fields
      (a proposal with an elaborate `row_selector` but an empty `fields` list). This is the same
      known, already-mitigated-everywhere-else-in-this-module characteristic (retries + multiple
      candidates per call), not a new reliability gap this task introduces; `n`/`attempts` are
      caller-tunable the same way every other candidate generator's are.
- [x] **Live testing found and fixed a real, pre-existing crash bug, not new to this task.** An
      LLM-proposed CSS selector is not guaranteed to be valid CSS -- a real proposal used the
      jQuery-ism `td:contains('*RESCINDED*')`/`:eq()`-style syntax, which raises
      `soupsieve.util.SelectorSyntaxError`, UNCAUGHT, crashing the entire call. This was already
      true of `validate_candidate`/`validate_row_candidate` (both call `soup.select_one`/
      `soup.select` directly, no exception handling) -- every existing `eval_loop.py`
      candidate-generation path was equally exposed to a malformed LLM-proposed selector crashing
      the whole eval loop, in production, since Chunk 2. Fixed at the source (`_select_one_safe`/
      `_select_safe` wrappers, used by `validate_candidate`, `validate_row_candidate`, and this
      task's own `_fields_matching_any_row`), not worked around only in new code -- per "no
      pre-existing exception."
- [x] `packages/scrape/README.md` documents the new capability.
- [x] mypy/ruff clean, full package test suite green (290/290 -- new tests cover
      `discover_candidates`/`discover_row_candidates` plus regression tests for the
      `SelectorSyntaxError` fix on the pre-existing `validate_candidate`/`validate_row_candidate`),
      zero new pre-existing-baseline drift (142 unchanged).

**Second Critic-caught and fixed before ship (chunk-mode review):** `discover_row_candidates`'
selection logic accepted a candidate based on `_fields_matching_any_row`'s pre-filter alone,
without confirming the survivors ever co-occurred in one real row. Two fields whose non-empty rows
are disjoint (field A only ever present in row 1, field B only ever present in row 2) each
independently survive the pre-filter, but `validate_row_candidate`'s all-or-nothing-per-record
check then correctly returns zero records for the combined schema -- the original code still
accepted this as `validated=True` with an empty `sample_records`, exactly the "reports success but
extracts nothing" failure the README/this doc explicitly invite a caller to trust (`field_schema`
handed straight to a `ScrapeTarget`). Fixed: a candidate is only accepted if
`validate_row_candidate`'s own `.records` is non-empty -- a real, coherent record must actually
exist, not just individually-plausible fields. Regression test:
`test_disjoint_fields_that_never_co_occur_in_one_row_are_rejected`.

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears-scrape
uv run pytest packages/scrape/tests/test_extraction.py -q
uv run pytest packages/scrape --ignore=packages/scrape/sidecar -q -m "not live"
uv run ruff check packages/scrape/ && uv run mypy
```
