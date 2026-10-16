# Adopting the engine: the host, the scope and the kind

**For** the developer wiring 3tears-evals into an app for real: your own storage, your own launch path, runs
you can compare over weeks. **Answers:** what you write (a host, a store, a kind, a launcher) and what the engine
does with each. If you only want to grade a function, and keep its runs in a file, an `Eval` (as on
[the ladder](ladder.md), and [kept and read back](serving.md#keep-your-runs-and-read-them-back)) builds all of this
for you. Read
[Concepts](concepts.md) first: this guide uses its terms (host, kind, lever, apparatus, scope, cell)
without stopping to define them.

## The shape of it, in plain words

You write four things, and the engine does the rest:

1. **A host** — one value that holds your vocabulary (what you can change, what you can measure), your
   store, and a few services. Every engine function takes it; nothing is global.
2. **A store** — one small key-value port (`DocumentStore`), proven correct by a conformance kit you run
   in your own tests.
3. **A kind** — the adapter that turns a case into a call to your production code: `prepare` builds a
   candidate, `invoke` runs it.
4. **A launcher** per kind — the code that, when someone launches a run, freezes the cases and hands the
   engine the wiring.

Everything past those — the trial loop, the stored documents, judging, pricing, the analysis — is the
engine's. The sections after "Launching" cover what you need only if your subject has a stateful world,
converses, does background work, or calls tools you want to record and replay.

Quick's pieces build some of these as ordinary contract objects you mix with your own: `@measure(...)`
(`threetears.evals.quick`) declares a `MetricDescriptor` on the function that computes it, so you register its
`.descriptor` and grade with the function; `as_host([evals], store=...)` (`threetears.evals.quick`) is the host an
`Eval` runs on, its kind over a plain function; and a `World`'s (`threetears.evals.World`) `registry` and
`bindings(state)` are a simple world's declaration and its handles. The toy host's kind computes its measures
this way.

## Read the reference hosts, in this order

Two example hosts live in this repository (not in the wheel), written as reference code that imports
nothing but the public roots and itself:

1. **[`tests/fixtures/courierhost/`](../tests/fixtures/courierhost/__init__.py)** — the least a product
   writes when it builds its own host, in one module (a courier route planner). Its docstring lists what a
   product supplies, in the order it is written. Start here.
2. **[`tests/fixtures/toyhost/README.md`](../tests/fixtures/toyhost/README.md)** — a host that exercises
   every shape of the host contract (invoice field extraction), with a map of which file holds which step.
   Read it once you have the courier host's outline and want to see a particular piece in full. It exercises the
   engine past the declared surface, so a few of its imports are `provisional` (a variant's identity, a lease's
   default); the courier host imports declared names only.

Both run on `InMemoryDocumentStore` (`threetears.evals.storage`), the engine's in-memory reference
`DocumentStore`: scoped, with conditional writes, and the shape to compare your own adapter against. Its
durable sibling, `SqliteDocumentStore`, keeps every document in one SQLite file through the standard library;
a host with no database of its own can use it as its store.

Import only from a public root (`threetears.evals.PUBLIC_ROOTS` lists them, including
`threetears.evals.kernel.host`), and only the names
its `__all__` declares; never from a module below one. Every engine type a public signature hands you — a protocol you implement, a value you
receive, an exception you catch, a literal you annotate with — is exported from one of those roots.
The stored documents and the ports you implement (`DocumentStore`, `CompletionClient`, `TraceSink`) are in
`threetears.evals.schema`; the behaviour over them (identity, scoring, `EvalStorage`, the candidate kind) is in
`threetears.evals.kernel`, with the host contract under `threetears.evals.kernel.host`.

One root fails to import without an extra, and `threetears.evals.ROOT_EXTRAS` says which and what it needs: a
`RootExtra` with the extra's name and every module the root imports past the base install (`transports.fastmcp`
needs `fastmcp`). A public-API gate reads it to skip a root
whose modules are absent, and to name the missing one, without reading the manifest or keeping a list of its own:

```python
import importlib.util

from threetears.evals import PUBLIC_ROOTS, ROOT_EXTRAS


def missing(root: str) -> list[str]:
    needs = ROOT_EXTRAS.get(root)
    return [module for module in needs.modules if importlib.util.find_spec(module) is None] if needs else []


importable = [root for root in PUBLIC_ROOTS if not missing(root)]
```

## The host

**Your app is one value, the host.** Build an `EvalHost` (in `threetears.evals.kernel.host`) and
pass it to every entrypoint. It holds:

- your `HostProfile` — the levers you sweep, the measures you record, your bars and your world. Every measure
  you declare (`MetricDescriptor`) carries a `reader_name`, the few words a table header, chart title or axis
  label prints in place of its key ("Output speed", not `candidate_output_tokens_per_s`). `MeasureRegistry`
  refuses a measure without one, and one whose `reader_name` matches another measure's (yours or the engine's,
  ignoring case), so the mistake surfaces at startup, not on a chart. The key stays what the store files the
  number under and what a reader cites. Your bars (`BarRegistry`, each a `Bar`, with any per-behavior
  `PassThreshold`) can be built in code or kept as data: `load_bar_registry(mapping)` reads
  `{"bars": [...], "pass_thresholds": [...]}`, each entry its type's fields. It refuses a field it does not
  know and a value of the wrong type rather than ignoring or coercing them, because a misspelt `rationale`
  dropped silently would register a standard other than the one you wrote. A bar may stand on a judged dimension
  rather than a measure you declare: set `judged=True` (`"judged": true` in a declaration) and name the rubric
  dimension (`<context>.<dim>`) or a reserved axis. Each campaign of the behavior resolves it against its template
  and is refused when the template does not declare it, exactly as a campaign's own bar on a judged dimension is;

- the `EvalStorage` you read and write through;
- a factory for the completion clients the engine's own judge, simulator and analysis roles call. Every
  completion a client returns is read by `CompletionResult`'s attribute names, and the usage ledger reads the
  token and cost attributes with a default, so a renamed attribute would degrade every usage row to
  "unreported" in silence. Check your completion type in your own suite with
  `threetears.evals.testing.check_completion_conformance(an_instance)`, which fails naming the attribute.
  `served_model` must be the model the provider's RESPONSE named, never the request: it is how a candidate
  launched on a floating alias records which model answered it. Read it off the provider's response (the
  `model` its reply body names), and report `None` when the response names none rather than copying the
  request in, which would make two different models compare equal. A `None` is never refused: the judge's
  scores and the usage rows keep the id requested, and the run's summary (`run_get`), `run_summary`, the
  analysis bundle and the campaign report each say once that which model answered is unconfirmed, counting
  the results and scores it covers ([Reading reports](reading-reports.md#which-model-answered)). Your
  candidate kind's own usage rows (`CandidateTelemetry.usage`, below) carry it the same way: fold each
  completion with `RoleUsageLedger.add_llm_result`, which reads it. `temperature` must be what the request
  carried: the judge asks for 0 unless a config says otherwise, and a client whose model refuses a temperature
  sends none and reports `None`, which the score records as `model_default`. A judge completion with no
  `temperature` attribute at all is your code's defect, not a judging outcome: the engine raises
  `CompletionProtocolDefect` (a `TypeError`, in `threetears.evals.schema`) and the run fails with that message,
  as it does for a kind's `CandidateKindDefect`, rather than finishing with an error on every dimension. The
  error carries the calls already billed and their cost (`calls`, `cost_usd`), and the cell it surfaced in is
  recorded excluded (`apparatus_failed`) with what it was billed, so the run's totals and its cost cap count it, as
  a cell failed by a `CandidateKindDefect` is. A judged witnessed cell is not recorded, but its billed judge calls
  count against the run's judge ceiling. A re-judge, a judge repeat, a second judge and a witnessed cell's judging raise it the same way;
- your `failure_describer` — how a raised provider call reads. It is the only thing that can say a call
  was refused for the calling account, which stops a run rather than excluding its cells;
  `withhold_failure_detail` is the honest one for an app with no error types of its own;
- your tracing, executor and cell-timeout choices. None of these last three has a default: each is named
  where the host is built;
- your template checks, `template_checks=TemplateChecks(...)` (from `threetears.evals.kernel.host`): the questions
  about a template only you can answer, and the kinds you launch. `require_known_tools_allowed` asks whether each tool
  it allows is in your catalog, `refuse_undeclared_world_seed` whether its world seed names only paths your world can
  seed, and `refuse_undeliverable_template` whether its kind can honour what it declares; each raises
  `ValidationFailedError` to refuse and returns `None` to admit. `kinds` (required) is your launch registry, by
  kind name, stated here and nowhere else: your `LaunchHost` reads it off its `EvalHost`'s checks and takes no `kinds`
  of its own, so no template is admitted against a kind your launch does not carry, or the reverse. A template stating a field its kind's entry never acts on
  (`LaunchableKind.ignored_template_fields`) is refused with the launch dispatch's own message, so your
  `refuse_undeliverable_template` need not check those fields itself; it keeps only the refusals the package cannot
  make for you. `refuse_inadmissible_template_case` (required) is the same question about one of a template's own cases
  (a `SeedTemplateCase`, `host_payload` included): the boot seed, a definition reset and `case_rename` each ask it
  before they write, so a case you refuse is written by none of them and each reports your reason; the engine has
  already refused a case with no name. `create_template`, `update_template`, the boot seed, a definition reset and the `definitions_reset`
  action read the checks off the host they are handed, so every path admits a template under the same checks. A host
  with no template checks (`None`, the default) is refused by each of them, since a template it admitted would go
  unchecked; a check with nothing to refuse is a function that returns `None`. To test one of your checks alone,
  call `admit_template(template, profile=, checks=, active_judge_config=)` (from `threetears.evals.run`), the one list
  of create-time gates `create_template` and the seed both admit through, with your `TemplateChecks`: it takes no
  store, and returns the admitted template or raises the refusal.

Nothing is ambient: there is no installed or default host, so two hosts in one process never meet.

An app that starts runs builds a `LaunchHost` (in `threetears.evals.run`) around its `EvalHost`: the
launch settings (its registry of launchable kinds is the `kinds` of the `EvalHost`'s template checks, above), a job timeout, `world_placements` (required exactly
when the profile declares a world: what each assembled run did with every dimension) and a `launch_pricer`
(without one, an arm under an enforced cap is refused unless the launch names its own cap; see
[Cost and budgets](cost-and-budgets.md#every-arm-is-priced-before-any-launcher-runs)); it builds its job manager over
the host's own storage. Hand its `eval_host` to the analysis side.

Each kind in the registry (`LaunchableKind`) carries `resolve_subject`: whether the host holds the subject a launch
of that kind names — `None`, or an `UnknownSubject` saying how a caller finds one it does. It is the kind's, not the
host's, because one subject id means a live entity to one kind, a fixed prompt key to another and a free label to a
third; `launchable_judge_kind` brings its own, which admits any non-blank label. It is asked at admission, once the
template's kind is known, so `launch_estimate` and a launch refuse an unknown subject alike, before anything is priced.

### Recording completions for CI

**Record your real client once, and replay it in your suite.** `RecordedCompletions` (in
`threetears.evals.testing`) wraps your completion client: `record` sends every call through and appends what it
returned to a JSONL file, and `replay` serves those completions back in order with no network, so a suite that
judges, simulates or analyses runs offline on what a real model said rather than on a stand-in's text. Build the
recorder once per suite, a session fixture or a module-level object, and hand that one object to every unit of work:
`replace=True` empties the file each time a recorder is built with it, so one built per test would keep only the
last test's calls.

```python
from threetears.evals.run import JUDGE_REQUEST_SETTINGS
from threetears.evals.testing import RecordedCompletions

RECORDING = Path("tests/recordings/judge.jsonl")


@pytest.fixture(scope="session")
def judge_client():
    # The settings my_judge_client() is built with: for the judge, the value its builder applies.
    if os.environ.get("RECORD"):
        client = RecordedCompletions.record(
            my_judge_client(), RECORDING, settings=JUDGE_REQUEST_SETTINGS, replace=True
        )
    else:
        client = RecordedCompletions.replay(
            RECORDING, settings=JUDGE_REQUEST_SETTINGS, rerecord="RECORD=1 pytest tests/test_judge.py"
        )
    return client
```

**Pass the settings your client is built with.** `record` and `replay` both require `settings=`: the
`ClientRequestSettings` your client builder applies to that client (for the judge, `threetears.evals.run`'s
`JUDGE_REQUEST_SETTINGS`), or `None` for a client built with none. They never reach `generate`, so the recorder
cannot read them off the client, and they change the answer: a judge asked to reason hard thinks before it
scores and one asked for nothing may not. Every line carries them, and a recording made under other settings is refused,
not served: `RecordingRefused` names each setting that differs, recorded value against asked
(`recorded with reasoning_max_tokens=8192, reasoning_effort=None, asked with reasoning_max_tokens=None,
reasoning_effort='high'`), so moving your judge's reasoning effort or output cap tells you to re-record rather than
replaying answers the new configuration never asked for. Each line also keeps what the call's request was actually
SENT with, when your completion reports it (`sent_settings`, below), and a replayed completion reports the same, so
a replay records what the recorded call was sent with, never the settings it was asked under. A recording made by a
build before `.25` (recording format 1, `RecordedCompletions.FORMAT_VERSION` is 2) kept none, and is refused by name:
re-record it.

A call is matched on its prompt (`system`, `user`, `response_format`), the model and temperature its completion
reported, the settings the client was built with, and how many times that prompt was asked before, so `k` repeats
replay as the `k` answers they were. A prompt the recording does not hold, or one asked once more than it was, is
refused (`RecordingRefused`) with the prompt's first 80 characters and the re-record command, never sent live; it is
an `ApparatusError`, so a candidate that meets one has its cell excluded rather than scored. To name a prompt that
changed under its recording before a replay starts, hand
`RecordedCompletions.check_covers(RECORDING, prompts, settings=...)` the prompts your code sends. A file holds one
model's calls at one temperature, under any number of settings (one model at two efforts), and each client replays
its own with `replay(path, settings=...)`; record a second model into a file of its own, since recording it onto
the first is refused.

Recording onto a file that exists appends to it, and only `replace=True` starts afresh. A replayed price
(`cost_usd`, `price_source`) is what the call cost when it was recorded, not a quote at today's rates. Every
attribute your completion type declares beyond the ten and `sent_settings` is kept under the replayed completion's
`extra`, and one JSON cannot hold is refused when it is recorded.

The package's own examples run this way: each calls a cheap model through `Model` (DeepSeek's flash model runs,
Gemini's flash-lite judges, both through OpenRouter), records the run when `RECORD=1` is set with `OPENROUTER_API_KEY`,
and otherwise replays the recording kept beside it, so it runs offline on what a real model said rather than on a
stand-in's text, and its test checks the recording still covers its prompts ([the examples' README](../examples/README.md#recording)).

## Bringing a provider

**The engine calls models through one port, and names no provider.** A client is anything with
`async generate(*, system, user, response_format=None)` returning a `CompletionResult`, plus `aclose`
(`CompletionClient`, in `threetears.evals.schema`); the failure half is your `failure_describer`. You can write
both yourself, or use the client the package ships (`threetears.evals.providers`, on the base install), which speaks
OpenRouter's, Anthropic's and any OpenAI-compatible chat API over `httpx`, with no SDK and no other 3tears package:

```python
from threetears.evals.providers import Clients, describe_failure

async def build_host() -> EvalHost:
    clients = Clients(default="deepseek/deepseek-v4-flash", judge="google/gemini-3.5-flash-lite")
    unpriced = await clients.load_prices()  # once, at boot: {model: why} for any model left unpriced
    return EvalHost(..., clients=clients, failure_describer=describe_failure)
```

`Clients` builds one `Model` per role, model and temperature, all sharing one `Catalogue` (OpenRouter's public model
list, fetched once: the rates and the parameters each model takes) and what any of them learned from a refusal, and
`await clients.aclose()` closes them at shutdown (and the `http=` client handed to `Clients`, when it was handed with
`own_http=True`). A role is a model id or a `Model` whose settings that role's models
are built with: an id with a `/` goes to OpenRouter on `OPENROUTER_API_KEY`; a bare id needs
`Model(id, provider="anthropic")` (`ANTHROPIC_API_KEY`) or `provider="openai"` (`OPENAI_API_KEY`), never guessed;
`base_url=` takes any OpenAI-compatible endpoint. A missing key is refused when the model is built, naming the
variable. **The judge is built from `JUDGE_REQUEST_SETTINGS`** when given as an id, the request settings a judged run
records it asked for (`max_tokens=10240`, a reasoning budget of 8192, `strict_output=True`): sent as a budget wherever
the provider takes one (Anthropic's `thinking.budget_tokens`, OpenRouter's `reasoning.max_tokens`), so the budget binds
and not only the cap, and as the effort word its share of the cap reads as (`high`) where only words are taken
(OpenAI's chat completions, a model that refused a budget), which the completion's `sent_settings` names
(`unsent_reasoning_max_tokens=8192`); a budget not below `max_tokens` is refused naming both, never clamped;
the simulated user likewise from `SIMULATOR_REQUEST_SETTINGS` (`low`). Nothing is retried: the only resends answer a
provider's refusal of a setting, below. `describe_failure` reads the client's one error type, `ProviderError` (the
provider, the status and its own message, never the body, and `account` for a refused key or spent credit, which
stops a run), and an `httpx` transport error by the host it was sending to.

**A model that lacks a setting still answers, sent the closest one it takes.** The choosing, the refusal reading and
the resending are `threetears.evals.negotiation`'s, the same a client of your own uses (below), and each completion's
`sent_settings` says what its request carried. What is known of a model before any request: a Claude 5.5 takes no
temperature, and on OpenRouter the catalogue says which parameters no endpoint takes and the model's output cap;
unknown is sent as asked. A refusal of a setting the request carried is answered by sending it once more with the
closest setting the model then takes, learned for every model sharing the catalogue and logged once at WARNING; on
OpenRouter a strict call no endpoint takes under `require_parameters` (a 404 naming nothing) is resent without its
temperature, and only then without enforcement, for that call alone. The refused request generated nothing and is not
billed, so the price ceiling is one request's, but each refused send can take up to a request timeout
(`MAX_RESENDS`, 4).

**Pricing.** An OpenRouter call is priced by OpenRouter's own figure (`usage.cost`); under bring-your-own-key that
figure is OpenRouter's fee alone, so the cost is it plus `cost_details.upstream_inference_cost`, and a BYOK reply
without the upstream figure is unpriced. A reply that carries no `is_byok` is unpriced too (`cost_usd` and
`price_source` are `None`, `pricing_failure` says why), since its `cost` may be the fee alone. A direct Anthropic or OpenAI call is priced by the catalogue row
for the matching `vendor/model`, from the same reading of the row as the price ceiling: at the tier its prompt fell
in, its reasoning tokens at the reasoning rate where the row lists one apart from the completion rate, its cache reads
and writes at their rates, plus the request fee. Where a price applies that the reply's counts cannot price (reasoning
tokens it does not split out) the call is unpriced, never priced lower; its `price_source` ends `;cache-approximate`
when a cache share was priced at a rate the row does not list for it. The `price_source` words are `PriceSource`'s (host contract, from
`threetears.evals.providers`, each member the string itself): `OPENROUTER_REPORTED`, `OPENROUTER_REPORTED_UPSTREAM`
(the BYOK sum, `openrouter:reported+upstream`) and `OPENROUTER_CATALOGUE`, the prefix of
`openrouter-catalogue:<id>`. Compare a stored `price_source` with them rather than spelling the words. Unpriced is `None` with `pricing_failure` saying why, never `0`. A
model's `price_ceiling` answers once its catalogue is loaded. The engine awaits the model's `refresh_prices()`
before it prices a call, which loads it; `await clients.load_prices()` at boot reads it early and says which models
it cannot price. A long-lived process is not priced from its first
fetch forever: the catalogue refreshes on the first live call after `refresh_after` (`Catalogue(refresh_after=...)`,
six hours by default). A refresh that fails keeps the last rates rather than failing the call, logs it, and marks the
completions they price: `Completion.pricing_stale_since` is when those rates expired (`None` while fresh, and on an
OpenRouter call, which its own figure prices), and `Catalogue.stale_since` and `refresh_failure` say the same.
Several loads at once share one fetch, and one request makes at most one: a model's call takes the read its
`refresh_prices()` made for it in the same task as its own, so a read that failed (a first read included) is tried
again by the next request, never by the call it was made for. `refresh_prices()` returns what its read found, a
`PriceRefresh` (`refreshed`, `fetched_at`, `failure`), which the model keeps as `last_price_refresh` and
`clients.refresh_prices()` answers once for every role: a `failure` with no `fetched_at` is a first read that failed,
so no model can be priced, and one with a `fetched_at` is a refresh that failed, the rows read then kept. A host
writes its own pricing-outage line from it.

**The price ceiling is a true upper bound, or `None`.** `price_ceiling` bounds the prompt at one token per UTF-8 byte
(after NFKC normalisation, where that is longer) plus a framing allowance, and prices it with every price the row
lists: a prompt token at the dearest of the plain and cache rates, an output token (reasoning included) at the dearest
of the completion, reasoning and output-image or -audio rates, the per-request fee, and a long-context tier wherever
the prompt bound reaches it. A price it cannot bound (one it does not know, a web search on a model that searches
unasked) answers `None`, logged once with the reason. So do rows older than `Catalogue.ceiling_stale_after`
(`refresh_after` unless given), though a completion is still priced from them. Rows do not reach that age by sitting
idle: before it prices a call, in a run or out of one, the engine awaits the client's `refresh_prices()` if it has
one, which on `Model` refreshes rows past `refresh_after`. Only a refresh that failed leaves them old. Under an
enforced cap, `None` refuses an out-of-run call; a call in a run is made either way, and the cap is read between
cells ([Cost and budgets](cost-and-budgets.md#caps)). A client of your own whose rates go stale offers the same
`async refresh_prices()`. Optionally, to ride out a short outage without refusing calls, refresh more often than the
bound (`Catalogue(refresh_after=timedelta(hours=1), ceiling_stale_after=timedelta(hours=6))`), or await
`catalogue.load()` on a timer of your own; neither is needed for an idle host.

**The HTTP client is the model's, or yours.** A `Model` opens its own client on its first live call, set up by
`transport=` (any `httpx` transport: HTTP/2, a proxy, a test's mock), `timeout=` and `headers=`, and `aclose()` closes
it, and with it, as `httpx` does, the transport it was opened on: a transport shared between models is closed by the
first one closed. Or hand it a client with `http=`, used as you built it (beside it the other three are refused), and
say who closes it: `own_http=True` makes it the model's, so `aclose()` closes it; the default leaves it yours. A model
built from another (`derive()`, and so a `Clients` role given as a `Model`) sends through that one's `http` client
without ever closing it, or opens its own on that one's `transport`. So a host building a model per unit of work needs
no wrapper to release its client:

```python
model = Model(model_id, transport=httpx.AsyncHTTPTransport(http2=True), timeout=120.0, headers=attribution)
...
await model.aclose()  # closes the client it opened, and its transport
```

A catalogue you hand an `http` client fetches with that client's timeout; `CATALOGUE_FETCH_TIMEOUT_S` (30 seconds) is
the one it uses on its own. `Model(catalogue_fetch_timeout=)` sets that per model: a model reading a catalogue another
task is already fetching waits at most that long, then gives up and runs unpriced while the fetch goes on for the task
that started it, so a host needs no timeout of its own around the read.

**An app already on 3tears-models, LangChain or a provider SDK keeps its own client** and wraps it in the port: a
class whose `generate` calls the chat model it already builds and returns an object with the ten `CompletionResult`
members (and `sent_settings`, when it can say what the request carried), whose `aclose` releases what it owns, and a
`failure_describer` over that SDK's errors. The package ships no adapter to any of them, so evals depends on no other
3tears package; `check_completion_conformance` on your client's completions, and `check_refusal_negotiation` on the
client, in your suite, hold the wrapper to what the engine reads.

**A client builder of your own sends every reasoning field.** The judge asks by a token budget
(`JUDGE_REQUEST_SETTINGS.reasoning_max_tokens`, with `reasoning_effort=None`), so its reasoning is bounded where a
model takes a budget; `SIMULATOR_REQUEST_SETTINGS` and `analysis_gen_request_settings_for`'s value ask by
`reasoning_effort` and hold `reasoning_max_tokens=None`. A builder that reads only one of the two fields sends no
reasoning parameter to the roles that ask by the other, so they reason at the provider's default while the run
records what they asked for. Build the request from every field: `wire_settings_for(settings, temperature)`
(`threetears.evals.negotiation`) turns the engine's value into what `negotiate` sends, so nothing is hand-copied and
nothing is dropped.

**A client of your own maps its provider's finish reason onto five words** (`StopReason`): `end_turn` for a
completion that finished, `max_tokens` for one stopped at the output cap (OpenAI's `length`), `content_filter` for a
refusal or a filtered reply, `truncated` for output that stopped early for a reason the provider did not name or
named in words your mapping does not know (no finish reason at all, a value newer than your mapping, a pause such as
Anthropic's `pause_turn`), and `error` only when the provider reported an error in place of a completion
(OpenRouter's `error` finish reason) or returned no choices. Map anything you do not recognise to `truncated`, never to
`end_turn`, which would read a cut-short reply as finished, and never to `error`, which tells the operator the call
failed when it produced output. Every word but `end_turn` is cut short: the judge and the analysis generator never
parse or repair such a reply, and their message says why for each; for `truncated` the provider's raw value is what
to look at, so log it.

**Every enum a strict reply format sends is a string.** A provider's strict mode can accept an integer enum without
enforcing it (a Gemini model through OpenRouter left every ordinal score out), so the engine's own directives
enumerate strings only: the judge asks for an ordinal score as `"1"` to `"5"` and reads it back to its integer, a
bare integer still read. `strict_response_format` (`threetears.evals.kernel`), which your own calls can project a
reply type through, refuses a schema whose `enum` or `const` holds anything but strings with `UnportableSchema`,
naming where it sits, rather than rewrite a type your parser expects: declare such values as strings (a `Literal` of
strings, a `str` enum) and convert them on read.

### Reporting what each request was sent with

**A completion may say what its request actually carried: `sent_settings`.** The run records what the engine ASKED
each role for (`EvalRun.judge_asked_settings`, `simulator_asked_settings`); what a request was SENT with is a
different fact, and only your client knows it: it may send a model the closest setting it accepts in place of the one
asked for, drop one the model refuses, or resend a strict call without enforcement. Give your completion type a
`sent_settings` attribute holding a `ClientRequestSettings` built from what your client builder actually put on that
request, after every per-call change, and the engine records it per call: on each judge score
(`RubricScore.judge_sent_settings`), on each usage row (`RoleUsage.sent_settings`, every distinct value the row's
calls were sent with, the simulator's included) and on each out-of-run spend row (`OutOfRunSpend.sent_settings`).

- **One value per completion**, the request that produced it. A completion your client resent is reported as the
  resend went out.
- **What the request carried, never what was asked.** Echoing the settings the client was built with records a
  setting nobody checked as sent, and makes a judge sent one thing compare equal to a judge sent another. Report an
  effort word as `reasoning_effort`, a token budget as `reasoning_max_tokens`, and neither when no reasoning parameter
  went out.
- **`strict_output` is the enforcement actually sent**: `True` when the request enforced every parameter it carried
  (a strict `response_format` a direct provider refuses rather than ignores, a router's `require_parameters`),
  `False` for a request sent, or resent, without it.
- **`None`, or no attribute at all, is "my client does not say".** Nothing fails: the engine records the call as
  unverified and says so, and never refuses a call, a score or a run for it. It is not a `CompletionResult` property,
  so a completion without it still passes `check_completion_conformance`; a value of any other type is refused
  (`CompletionProtocolDefect`, from `sent_settings_of` in `threetears.evals.schema`, the engine's one reader of it).
  A client built with no output cap cannot report one, since `max_tokens` is required, and reports `None`.

**Never fail a call for a setting, and drop enforcement last.** A client of your own keeps the rule the shipped
client keeps: a model that lacks a setting still answers. When a provider refuses a request for a setting it
carried, send it once more with the closest setting the model takes, or without it, and report what the resend
carried. On a router that enforces parameters (OpenRouter's `require_parameters`, whose "no endpoint" answer names
no parameter), drop the refused settings first, the temperature before any other, keeping enforcement; only when
the request still finds no endpoint, send it without enforcement, and report `strict_output=False`. The engine's
judge, simulated user and analysis generator all ask for `strict_output=True`, so a strict call your router cannot
place is answered unenforced rather than failed: the call is recorded as sent unenforced, disclosed, and never
pooled with an enforced one.

Once your completion reports it, hold your suite to it:
`check_completion_conformance(completion, require_sent_settings=True)` fails the day the member is dropped, renamed or
left `None`, rather than every later run quietly recording unverified calls. **Ship it in the same deploy that adopts
`.25`:** a run whose calls report what they were sent with and one whose calls do not are not the same record, and
reporting from the first `.25` deploy keeps your campaigns from splitting at a later one.

### A client of your own that never fails a request

**No model may fail a request because it lacks a setting**: a judge model with no temperature, an effort word it
does not have, a router that finds no endpoint for a strict call under `require_parameters`. The shipped client
never does, and neither should your own client, so the logic is one public root, on the base install
(`threetears.evals.negotiation`, no SDK and no HTTP client): `negotiate` sends the closest settings the model is
known to take (`closest_settings`), hands any failure to your refusal reader, and on a refusal of a setting the
request carried learns it and sends again, in a fixed order (a refused temperature first, keeping enforcement, and
only then enforcement), at most once per setting and `MAX_RESENDS` (4) times per call; anything else is raised as
your transport raised it. Each refused send costs nothing but can cost up to your request timeout, so a wall-clock
ceiling you hold around a call allows `MAX_RESENDS` × that timeout past the served request's. It returns what the
served request carried, which is what your completion reports. You write the request and the parsing; you never
write the logic. A minimal OpenRouter client over raw HTTP (OpenRouter's request shape is the `router` family;
`read_router_refusal` reads its 404 and the 400s it relays from an upstream):

```python
from dataclasses import dataclass, replace

from threetears.evals.negotiation import (
    LearnedSettings,
    WireSettings,
    facts_from_router_catalogue,
    negotiate,
    read_router_refusal,
)
from threetears.evals.schema import ClientRequestSettings, StopReason


class RouterRefused(Exception):
    """The router's answer to a request it did not serve: what a refusal reader reads."""

    def __init__(self, status: int, body: object) -> None:
        super().__init__(f"OpenRouter answered {status}")
        self.status, self.body = status, body


@dataclass(frozen=True)
class RouterCompletion:
    content: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    model: str
    served_model: str | None
    temperature: float | None
    stop_reason: StopReason
    sent_settings: ClientRequestSettings | None


STOPS: dict[str, StopReason] = {
    "stop": "end_turn", "length": "max_tokens", "content_filter": "content_filter", "error": "error",
}


class RouterClient:
    def __init__(
        self, model: str, *, asked: WireSettings, post, learned: LearnedSettings, supported_parameters=None
    ) -> None:
        # post(body) -> (status, json): your HTTP call to /chat/completions. learned: one per process or factory.
        # asked: wire_settings_for(JUDGE_REQUEST_SETTINGS, temperature), never hand-copied constants.
        # supported_parameters: the model's row of OpenRouter's catalogue, read once at boot, or None.
        self.model, self.asked, self.post, self.learned = model, asked, post, learned
        self.declared = facts_from_router_catalogue(supported_parameters)  # only what no endpoint takes

    async def generate(self, *, system: str, user: str, response_format: dict | None = None) -> RouterCompletion:
        async def send(wire: WireSettings, facts) -> dict:
            body: dict = {"model": self.model, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user}]}
            if wire.max_tokens is not None:
                body["max_tokens"] = wire.max_tokens
            if wire.temperature is not None:
                body["temperature"] = wire.temperature
            if wire.reasoning_max_tokens is not None:  # a budget, as OpenRouter takes one: never beside a word
                body["reasoning"] = {"max_tokens": wire.reasoning_max_tokens}
            elif wire.reasoning_effort is not None:
                body["reasoning"] = {"effort": wire.reasoning_effort}
            if response_format is not None:
                body["response_format"] = response_format
                body["provider"] = {"require_parameters": wire.strict_output}
            status, answer = await self.post(body)
            if status != 200:
                raise RouterRefused(status, answer)
            return answer

        done = await negotiate(
            replace(self.asked, strict_output=response_format is not None),  # a schema asks for enforcement
            provider_family="router", provider="openrouter", model=self.model, send=send,
            read_refusal=lambda exc: read_router_refusal(exc.status, exc.body) if isinstance(exc, RouterRefused) else None,
            learned=self.learned, declared=self.declared,
        )
        choices, usage = done.reply.get("choices") or [], done.reply.get("usage", {})
        if choices:
            choice = choices[0]
            content, stop = choice["message"].get("content") or "", STOPS.get(choice.get("finish_reason"), "truncated")
        else:  # a 200 with no choices: the router reported a failure in place of a completion
            content, stop = "", "error"
        return RouterCompletion(
            content=content, input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"), reasoning_tokens=None, cost_usd=None,
            price_source=None, model=self.model, served_model=done.reply.get("model"),
            temperature=done.temperature, stop_reason=stop,
            sent_settings=done.sent_settings,  # what the SERVED request carried, never what was asked
        )

    async def aclose(self) -> None:
        pass
```

**Tell it what you already know: `declared=`.** `negotiate` takes the facts you hold about the model before any
refusal (`SettingFacts`), and leaves out a setting they say the model does not take. For OpenRouter,
`facts_from_router_catalogue(supported_parameters)` reads the model's row of OpenRouter's models catalogue into them,
as the client above does. Only the negative facts count: a parameter the row does not list is not sent (no
temperature, no effort word, or no strict schema, the request then going unenforced), and a listed one is sent as
asked, because `supported_parameters` says some endpoint takes it, not that one endpoint takes it beside the
request's others; so it never rules a temperature out of a 404, and the temperature is still the first thing dropped.
Read the catalogue once, at boot, and keep the facts: facts that moved mid-run because something else fetched the
catalogue would send two calls of one run differently for no reason the run records. Build `asked` from the engine's
own value (`wire_settings_for(JUDGE_REQUEST_SETTINGS, temperature)`, or the simulator's or the analysis generator's),
never from constants copied by hand.

**What a refusal teaches lasts for the process; enforcement is decided per call.** `LearnedSettings` holds what a
served request proved the model refuses, keyed by your provider name and the model, and every client sharing it
sends without that setting from then on: keep one per process, or per client factory as the shipped client does,
never one per call, or every call pays the refusal again. Only what was proved is learned: a temperature dropped on
the way to a request that was served only once enforcement went too taught nothing, and is sent again. Enforcement
itself is never learned. Every call that asks for it starts enforced and drops it for that call alone, because a
router's "no endpoint" can be an outage, and one outage must not leave a process unenforced for life; each such call
records `strict_output=False` and is disclosed.

Price it and read its served provider your own way; the negotiation is the part not to copy. For another wire,
`read_anthropic_refusal` and `read_openai_refusal` read Anthropic's and OpenAI's error bodies the same way (an SDK
keeps both on its exception, `exc.status_code` and `exc.body`), and `anthropic_thinking` builds Anthropic's
`thinking` for the settings sent. **Hold your suite to it** with `check_refusal_negotiation`
(`threetears.evals.testing`): it scripts routers that refuse the temperature, then enforcement, then an effort word
and an output cap, and fails naming each mistake unless your client resends in that order, reports what was served,
sends its next call at once without what a served request proved, and asks for enforcement again on every call. It
refuses an effort word, so build the client asking for one (the simulator's settings do; the judge's ask by a budget,
which goes as `reasoning.max_tokens`):

```python
from threetears.evals.negotiation import wire_settings_for
from threetears.evals.run import SIMULATOR_REQUEST_SETTINGS
from threetears.evals.testing import check_refusal_negotiation

await check_refusal_negotiation(
    lambda transport: RouterClient(
        "openai/gpt-6-luna",
        asked=wire_settings_for(SIMULATOR_REQUEST_SETTINGS, 0.0),  # an effort word, and a temperature
        post=transport,
        learned=LearnedSettings(),
    )
)
```

## Tenancy: the scope

**Tenancy is one opaque `scope_id`.** Every stored document carries a non-empty `scope_id`, and the
engine never interprets it, defaults it or branches on it: it is your tenant, project or environment,
whatever you partition by. (The one default is the quick path's: an `Eval` runs under a scope named after
itself unless `run(scope=...)` names one.) All of it goes through one `DocumentStore` you implement, keyed by
`(scope_id, doc_type, id)`. Every port call names its scope (an upsert takes it from the document it
writes), and there is no scope-free read: a caller that
needs several scopes is told which by you and asks each. A campaign and the runs it compares live in one
scope. Your adapter strips whatever it injects (an etag, a timestamp) before handing a document back,
because every stored model reads strictly.

## The store, and the conformance kit that proves it

**Prove your store with the conformance kit.** `threetears.evals.testing.STORE_CONFORMANCE_CASES`
states every rule of the port — scoping, the strip on read, `exclude`/`keep` projection, `by_doc_type`'s
equality and membership (`field_in`) predicates, ordering, etag conflicts and the re-read that recovers one, create-only writes, merge and delete — as a case you
hand a fresh, empty store. Every case is mandatory: in particular a store must implement both write
preconditions — `if_match` (the stored document still carries this etag), because several writers share a
run document as it finishes, and `create_only` (nothing is stored under this scope and id, of any type),
because two processes racing to start one campaign's analysis generation both create its one generation
slot and exactly one may land. A store that forwards writes to a backend passes both through. A store
may also offer `StoreClock` (`now()`, the store's own time), which job leases are then stamped in; offer it
only when the time is the store's, not the calling process's. `by_doc_type(..., field_in={"eval_run_id": [...]})`
is how the engine reads a campaign's results across its runs in one call: push it into your backend (a SQL store
writes `= ANY(%s)` per field, on the stored JSON type as its equality predicates are, and asks nothing for an empty
list), and call `field_in_values` (in `threetears.evals.schema`) first, which refuses what the port refuses and
drops repeated values. Parametrise your test runner over it:

```python
@pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_my_store_conforms(case: StoreConformanceCase, tmp_path: Path) -> None:
    case.run(MyDocumentStore(tmp_path / "evals.sqlite"))
```

**The store is called off the event loop.** The port is synchronous and the engine is async: every store call
the engine makes runs on the host's `blocking_executor`, never on the loop, and `EvalStorage` refuses a call made
on a thread running an event loop with `StoreCalledOnEventLoop` (in `threetears.evals.schema`). Your own async
code follows the same rule. A synchronous function that takes the storage or the host — most operations in
`threetears.evals.ops` (`runs_list`, `report_read`, `scope_history`), and the reads in `threetears.evals.run` and
`threetears.evals.analysis` — blocks: from async code, offload it with `await run_blocking(host.blocking_executor,
fn, ...)`, as the action catalogue does for every synchronous operation. An async operation (`run_launch`,
`job_poll`) offloads its own.

**If your database driver is async**, write your adapter against it as an `AsyncDocumentStore` (in
`threetears.evals.schema`: the same port, every method awaited and `iter_by_doc_type` an async iterator;
`AsyncStoreClock` for its clock) and hand the
engine `bridge_async_store(store, loop)` (in `threetears.evals.storage`), built with the loop the store's
connections belong to — in an async app, the running loop at startup. Each call the engine makes runs as a task on
that loop, in the calling context (a context variable your row-level security reads arrives set), while the
executor thread waits. Keep `blocking_executor` a thread pool. Prove the adapter through the bridge, with its loop
on a thread of its own, since a conformance case calls the store synchronously:

```python
@pytest.mark.parametrize("case", STORE_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_my_async_store_conforms(case: StoreConformanceCase) -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        store = asyncio.run_coroutine_threadsafe(open_my_async_store(), loop).result()
        case.run(bridge_async_store(store, loop))
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()
```

The engine reads and writes through `EvalStorage`, built over that one store. Its consumers name the
area they touch rather than the whole of it — `RunStore`, `ResultStore`, `RunRecordStore`,
`DefinitionStore`, `CassetteStore` and `JobStore` (in `threetears.evals.kernel`) — so a function
typed `DefinitionStore` cannot reach a run, and a test of one hands it only that area.

**Watch the read tier's size.** The lenses that read a whole scope — `pivot` and `export_results` (and the
`scope_pivot` / `scope_export` operations over them) — project and aggregate every result in Python on each
call, which suits a scope of dozens-to-hundreds of cells. Each call logs one line on the
`threetears.evals.analysis.reads` logger, at INFO:

```
eval.read_tier lens=pivot scope_id=acme runs_in=12 results_in=480 records_out=2880 cells_out=24 elapsed_ms=41.7
```

`runs_in` and `results_in` are what the call read, `records_out` the score records it projected, `cells_out`
the pivot's cells (`refused` when the pivot was refused; absent on an export) and `elapsed_ms` its wall time.
Once `records_out` passes `READ_TIER_ROW_BUDGET` (50,000, in `threetears.evals.analysis`) the line is logged
at WARNING and ends `over READ_TIER_ROW_BUDGET=50000`, so you can alert on the level alone. Crossing it means
the scope has outgrown the in-Python projection, the signal to push it down into your store; it fires
before a page is slow, and you can alert on `elapsed_ms` instead.

## Stored data: what is kept

**Every stored type has facts.** `DOC_TYPE_FACTS` (in `threetears.evals.schema`) is the one list of the
types the engine stores, and says three things about each:

- **origin**: `input` (what runs read as given, whoever wrote it: templates, judge configs, rubric dims, case
  sets, test cases, campaigns), `record` (what happened: runs, results, traces, calibration ratings, the
  out-of-run spend ledger, sweeps, analysis attempts, cassettes, generation slots, launches, second judge passes,
  judge temperature comparisons, host action jobs) or `derived` (computed from other stored
  documents: analyses, insights, judge profiles).
- **lifecycle**: `kept` (the evidence core: read from the core baseline on, and deleted by a core baseline reset),
  `durable` (what people author: templates, rubric dims, judge configs and their tombstones, case sets, test cases
  and calibration ratings; never wiped by a core baseline, read forever from durable v15, changed only additively
  or through an upgrader) or `regenerable` (refused after a format change, and made again: an analysis, insight or
  judge profile is regenerated, a cassette re-captured).
- **suppresses**: for a tombstone, the type it suppresses (`rubric_dim_tombstone` holds a `rubric_dim`'s
  slot, `judge_config_tombstone` a `judge_config`'s).

`EVAL_DOC_TYPES` (what the operator wipe sweeps), `CORE_DOC_TYPES` (the evidence core) and `DURABLE_DOC_TYPES`
(what people author) are derived from it.

**The evidence core is kept until its baseline moves.** Every release reads it from the core baseline (v15) on.
The core is what a run produced or binds: runs, results, traces, the out-of-run spend ledger, the campaigns people
declared and gathered runs into, and the records of how each sweep ended, what each analysis generation spent, and
which generation holds a campaign's slot (a later build that could not read a slot would block the campaign's
generations, or start a second beside a live one), and the records of every other job: a launch (the runs it
started), a second judge's pass (its report), a judge temperature comparison (every answer, stored nowhere else) and
a host action's job (its result). `CORE_DOC_TYPES` lists them. The core is baseline-resettable: a release may move
`CORE_BASELINE_VERSION` instead of shipping an upgrader, and every core document below it is then refused by name
and deleted with `reset_core` (below).

**What people author is durable, and no core baseline ever touches it.** Templates, rubric dims, judge configs
(with their tombstones), case sets, test cases and calibration ratings (`DURABLE_DOC_TYPES`) have a version space
of their own, forked from core v15 at `.26` (`DURABLE_SCHEMA_VERSION`, `DURABLE_SCHEMA_MINOR`). Its baseline,
`DURABLE_BASELINE_VERSION`, is 15 forever, and a durable shape changes only additively (a minor) or with an
upgrader (`DURABLE_UPGRADERS`); a durable major never resets. No durable document needs a core one to be read: a
rating names the run and result it was given on as provenance, and carries what was rated
(`CalibrationRating.rated`), so it still calibrates a judge after the result is gone. Budget storage at about one
judged transcript per rating. **Back the durable store up like source code**: it is what people wrote, and nothing
can make it again.

**The first public release is the baseline.** It writes core v15.1 (major `CORE_BASELINE_VERSION`, equal to
`CORE_SCHEMA_VERSION`, and minor `CORE_SCHEMA_MINOR`, 1, which added refused calls to the call ledger and to
control end states, a judged dimension's judge declaration and a judge reply's cut-short record, and reads every
v15.0 document), durable v15.1 (`DURABLE_SCHEMA_VERSION` and
`DURABLE_SCHEMA_MINOR`: a rating carries what was rated), regenerable v10 (`REGENERABLE_SCHEMA_VERSION`) and
cassette v1 (`CASSETTE_SCHEMA_VERSION`), and registers no upgrade steps (the core ships none, ever, and
`DURABLE_UPGRADERS` is empty): there is nothing older for it to read. A durable document `.25` wrote (as core v15.0) reads as durable
v15.0 with no step.

**A core version is a major and a minor, and your store needs no migration for either.** Every core document
carries `schema_version`, the major, and `schema_minor`, the minor within it. Every change to a core shape is one
or the other, an added optional field included:

- **A minor is additive only**: a new optional field with a default, a new optional sub-object, a new value of an
  open vocabulary, a new doc type. Its ledger entry says, in words, that an older build that ignores the change
  reads the document correctly and its writes leave it true (`MINOR_ADDITIVITY_SENTENCE`); when that is false, the
  change is a major. **A minor rolls with no stop, in either direction.** A build reads any newer minor of its own
  major: every field it does not declare, at every depth, is kept on the model, out of every attribute, dump and
  digest, and written back where it was the next time the engine writes that document, stamped with the newer
  minor. So while a fleet is half upgraded, an older process reads what a newer one wrote as if the new fields
  were not there, and keeps them on every part of the document it carries over (a part it replaces with one it
  built afresh, such as a renewed lease, takes them with it); two builds can print different numbers from one
  store for that window. A bundle does not name the build that assembled it: it carries its own shape
  (`schema_version`), the models each role ran (`model_versions`) and the cell model (`cell_model_version`), so
  record the package version beside what you store (`threetears.evals.__version__`) to tell the two builds apart.
  **The tolerance follows the stored bytes, which are trusted:** a host that writes core documents straight into the
  store, with a `schema_minor` above the build's, gets it too, so write the minor your build writes. **The roll has
  a floor, `.25` (`v0.67.0-evaladopt.25`):** it wrote the v15 baseline, and every build before it reads and writes
  core v14 or older, so it refuses every core document `.25` or a later build writes. Roll back no further than `.25`.
- **A major is everything else**, and is a baseline reset: it moves `CORE_BASELINE_VERSION` with
  `CORE_SCHEMA_VERSION`, ships no upgrade step, and refuses every core document below it by name, and its release
  notes say to delete them before deploying it (`EvalStorage.reset_core`). A newer major is refused too, so every
  process sharing the store moves to it together. The durable space is the one with upgrade steps
  (`DURABLE_UPGRADERS`): its baseline never moves.

So one scope can hold documents at several core versions. Neither a minor nor a step changes a field the store
filters, sorts or projects on (`CORE_ADDRESSING_FIELDS`, which the minor is not one of), so an index your adapter
keeps on those fields stays valid. Two things an older build cannot do for a new doc type a minor adds: its
operator wipe does not sweep it, and its deletes (`run_delete` and the cascades it runs) cannot reach it, so a
document of the new type can be left orphaned for the upgraded build. Wipe and delete from the newest build
while a fleet is mixed.

**Some vocabularies are open.** A usage role, a simulator purpose, a ceiling's or a role model's origin, a
reasoning effort, an out-of-run purpose and a stored refusal kind are strings, with the values this build knows
registered beside each (`USAGE_ROLES`, `COST_CAP_ORIGINS`, …). A newer minor can add a value; this build reads it
from that minor's document and names it wherever it reports one (a role's spend on a line of its own, an origin
by name), and refuses it anywhere else. Only what is stored is open: a parameter you pass in (an action's filter,
a writer's argument) takes a value this build knows (`KnownUsageRole`, …) and refuses any other. Every other
vocabulary on a core document (a run's status, a cell's
termination, a scale, a provenance, …) is closed: a new value is a major. `schema/versioning.py` lists each.

**Everything else is regenerated.** Analyses, insights and judge profiles are built over the core. When one of
their shapes changes, `REGENERABLE_SCHEMA_VERSION` is bumped and a document written under another version is
refused: make it again over the kept runs (generate the analysis, profile the judge). After a bump,
`analyses_undescribable` (operation and action) lists the scope's analyses holding an arm this build can no longer
describe. A cassette is not touched by that bump: its format has its own `CASSETTE_SCHEMA_VERSION`, bumped only
when the cassette itself changes, because re-capturing a corpus calls your third-party tools again. A cassette
written under another cassette version is refused, by name, and you re-capture it (a new capture run, so a new
corpus); a replay that asks for a refused recording stops its cell with a `CassetteCorrupt` naming both versions.

**Each version space moves on its own.** Core documents (v15.1), durable documents (v15.1), regenerable documents
(v11), cassettes (`CASSETTE_SCHEMA_VERSION`, 1), the bundle shape (the bundle's `schema_version`, 55), the cell model
(`CELL_MODEL_VERSION`, 13) and the identity keys (`IDENTITY_VERSION`, 24) carry separate versions, each at its first
public release's value. The report has none: it is a page rendered from the bundle, which is the versioned machine
form.
**A judge's identity moves with its wording, and no version number moves with it.** Each result records what asked
each dim (`EvalResult.judge_config_ids`): `builtin:<sha12>`, a fingerprint of the engine's own judge prompt, or
`<config id>@<sha12>`, your `JudgeConfig` and a fingerprint of the reply-format suffix the engine appends to its
prompt. A release that rewords either changes the fingerprint, so every result judged before it is another judge
for agreement, judge profiles, drift and evidence tiers, and a `calibrated` tier reads `undetermined` until you
calibrate the new judge. Nothing stored is refused or rewritten. Such a release's notes (`MIGRATIONS.toml`, printed
by `scripts/release_notes.py`) carry `JUDGE_WORDING_CHANGE_SENTENCE` (in `threetears.evals.schema`) and name
the wording that moved, so read them before you upgrade a scope whose readings rest on calibration. A run's own `judge_config_ids` (the
config set it launched with, part of its context key) holds bare config ids and no built-in dim, so a wording change
does not split runs' context: only the results' recorded judges tell the two judges apart.
An upgrade never recomputes an identity key, so keys from different predicates never silently pool. A kept
key is stored beside its pre-image and the identity version that computed it, so a read under a later
`IDENTITY_VERSION` needs no upgrader: a run keeps `variant_levers` beside the keys its results carry, and a
campaign's declared control keeps `control_levers` and `control_identity_version` beside `control`, recorded when
the control is addressed from a run (`control_from_run_id`, `set_campaign_control`) and never taken from a caller.
Under another version the control matches its stored key, which the cells stamped under its own version keep, and
the key recomputed from that map, which the arm's new runs carry, so it never goes stale while its old cells are in
the campaign. Its arm is one set of cells: the new runs' cells where there are any, with the old cells named beside
it as the control under the earlier version (`RealizedDesign.earlier_control`), disclosed, in no contrast, and no contestant on the frontier; the
control's numbers are the new cells only. The recomputation follows a change to how a map is digested only: a lever,
kind or overlay field your host renames, or a lever added to every map, gives new runs a different map, and the
control then matches only the cells from before it. A control key typed by hand
records no map; when no cell carries it under another version it reads as `stale`
(`RealizedDesign.control_excluded`), and the remedy is to address it again from a run.

**Reads stay strict everywhere the contract cannot vouch for the unknown.** Every stored model refuses an
unknown field and a missing required one, at any depth, with one exception: a core document of this build's major
at a newer minor, read as above. A core document written by a newer major is refused, never guessed at, so when
two builds share one store across a major, upgrade the older one. Only the stored read is tolerant: a document a
caller hands an authoring surface is validated strictly, and `schema_minor` is a server field it ignores. A core
document written before `.25` (`v0.67.0-evaladopt.25`, core below v15) is refused
too: its read fails validation on `schema_version`, with a message that names the baseline it is below. So is a
core document below a later core baseline.

**On a core baseline, call `reset_core`; do not truncate tables by name.** It is the one operation your deploy
runbook runs, with the engine stopped, and the store boots the moment it returns:

- **On a `RoutedDocumentStore`**: `RoutedDocumentStore.reset_core(stores, route, scope_ids=..., routing_scope=...,
  host_doc_types=...)`. It deletes every core and regenerable document (`CORE_RESET_DOC_TYPES`) of the named scopes
  from every store, wherever it sits, and never a durable document, one of your own types or the router's placement
  record. Then it relocates: a durable type your route now keeps in a store of its own is moved there once, and
  every store's placement record is written, so the router starts even if a hand-run TRUNCATE once wiped the
  records (which is what makes a runbook that names tables fail to boot with `RouteChanged`). It returns a
  `CoreReset` naming what it deleted and moved.
- **On any other store**: `EvalStorage.reset_core(scopes)` deletes the same types through the store port.

Never truncate the tables holding durable documents. A store written before `.25` (core and durable below v15) is
never read; reset it from `.25` or later, and reset the durable tables too only for that pre-release store
(`nuke_all_eval_data(scopes, include_durable=True)`), since no release reads what it holds.

**`nuke_all_eval_data` requires `include_durable=`.** The full operator wipe takes a keyword with no default, so
your code states whether the labels and definitions people wrote go too. With `include_durable=False` it is every
engine type but the durable ones, and your own types; `reset_core` is narrower (core and regenerable only) and is
the baseline operation.
A refused run stays as stored, and the boot reclaim (`sweep_abandoned_runs`) counts it `unreadable` rather than
settling it. `runs_unreadable` (operation and action) names each stored run of the scope the current model refuses,
of any status, with the fields it refused and why. The ones stored pending or running are marked stranded: they are
the reclaim's `unreadable` count for that scope, and nothing settles them. `run_delete` removes any of them from its
stored row, with the results and campaign memberships a readable run's delete removes. A stranded one is refused
while its stored lease has not lapsed, and when its lease cannot be read either: then whether a process still holds it
cannot be told, so deploy the build that wrote it, which can settle or delete it. Or deploy a build that reads them. Every other read of such a run refuses with `UnreadableRunError` (in
`threetears.evals.kernel`, status 500, remedy naming `runs_unreadable`), except the `runs_list` inventory, which lists
the runs that read and counts the rest; map it in your own routes like any other engine refusal.

### Partition by facts: one store per kind of document

You can keep each kind of document in a store suited to it, such as the evidence in your database and
everything the engine can remake in a cheaper store. `RoutedDocumentStore` (in `threetears.evals.storage`) is
one `DocumentStore` over several. A `DocTypeRoute` keys on one fact, `origin` or `lifecycle`, and names a store
for every value of it, plus one store for the types the registry does not hold (your own documents):

```python
from threetears.evals.kernel import EvalStorage
from threetears.evals.storage import DocTypeRoute, RoutedDocumentStore, SqliteDocumentStore

stores = {
    "evidence": MyPostgresStore(dsn),
    "authored": MyPostgresStore(dsn, table="eval_authored"),  # what people author: back it up like source code
    "scratch": SqliteDocumentStore("regenerable.sqlite"),
}
route = DocTypeRoute(
    by="lifecycle", to={"kept": "evidence", "durable": "authored", "regenerable": "scratch"}, unregistered="evidence"
)

ROUTING = "evals-routing.production"  # yours to name; never a scope you keep documents in

# Once, with the engine stopped: move what the stores already hold to where the route puts it.
RoutedDocumentStore.relocate(
    stores, route, scope_ids=["acme", "globex"], routing_scope=ROUTING, host_doc_types=["host_note"]
)

storage = EvalStorage(RoutedDocumentStore(stores, route, routing_scope=ROUTING), host_doc_types=["host_note"])
```

A route keys on facts, never on type names, so a type a later release adds is placed by what it is. Every
mistake a route can hold is refused when the router is built: a fact value left unmapped, a store that is
not among `stores`, one store passed under two names, a tombstone kept apart from the type it suppresses.

**Typed calls go to one store; id-only calls ask every store.** A write, `get_many`, `by_doc_type` and
`iter_by_doc_type` go to the store the type is kept in, and a conditional write's `if_match` or `create_only` reaches that store
unchanged. `get`, `get_with_etag`, `merge_fields` and `delete` carry only an id, so they ask every store. A store
that does not hold the id answers "not here", as the port says. One id lives in one store: a read that finds an
id in two stores, or in a store its type is not kept in, raises `MisplacedDocument` rather than picking one. A
`create_only` write asks every other store for the id first and is refused with `StoreConflict` if one holds
it, as a single store would refuse it. Only the owning store's create is atomic, so a write racing that check
can leave the id in two stores, and the next id-only read reports it as `MisplacedDocument`.

**Changing the route is an operation.** Each store records the placements it was written under, in the routing
scope you name (`routing_scope`, on the router and on `relocate` alike, with no default). Never keep documents in
it, and your adapter must be able to hold it. If you run several partitions in one database, such as
environments or tenants behind row-level security over tables whose key is global, run one router per partition
and give each its own routing scope: one shared scope would hold every partition's record under one key, which
only the first partition could write. A router built with another routing scope than `relocate` was given finds
no record and refuses to start.
The router refuses to start (`RouteChanged`) when a type would be read from a store it was not written to, when
two stores are passed under each other's names, and when no store has a record yet. `RoutedDocumentStore.relocate`
moves every document of the scopes and host types you name to its new store, then writes the records. Run it
once when you adopt the router and again whenever you change the route, with the engine stopped. If it is
interrupted, run it again. A release that adds a type needs no relocation unless your route places the new type
somewhere other than your `unregistered` store. A release that adds a fact value needs your route to map it: a
lifecycle route must map `durable` (from `.26`), or it is refused when built. Giving `durable` a store other than
the one `kept` types were written to moves the eight durable types, so the first start raises `RouteChanged` for
them until you relocate (or run `RoutedDocumentStore.reset_core`, which relocates).

**Ask whether the stores hold anything before you relocate.** A host's start that adopts tables, rather than
relocating on every start, asks first: `RoutedDocumentStore.holds_documents(stores, scope_ids, host_doc_types=...)`
answers whether any store holds any document of any registered type, or of your own types, in the scopes you
name, and stops at the first one it finds. It reaches what `relocate` reaches and nothing more: the scopes and
host types you name. It does not count the router's placement record, which lives in the routing scope (never a
document scope) under a type no registry holds, so stores holding only a record answer `False`.

```python
if RoutedDocumentStore.holds_documents(stores, ["acme", "globex"], host_doc_types=["host_note"]):
    ...  # documents are there: relocate them, with the engine stopped, before the router starts
```

`RouteChanged` and `MisplacedDocument` carry `fault` and `remedy` apart (their `str()` reads both). A host that
offers its own relocation tool states its own remedy beside the router's `fault` instead of rewording the message.

**A store's clock passes through when every store has one.** When each of your stores offers `StoreClock`
(its own `now()`), so does the router, answering from the store your runs are kept in, so every process stamps
job leases on one clock. If any store has none, neither has the router, and each process reads its own clock.

Prove the router as you prove any store: the conformance kit runs over a `RoutedDocumentStore` of your stores.


**A record your host writes itself is held to what the engine writes.** A run or result the engine did not
launch (a witnessed session, a run your host assembles) is validated as strictly as one it did:

- a run that names a `judge_model` names the `judge_temperature` its unconfigured dims were requested at, and a
  run that records `goal_check_proofs` records the `goal_check_proof_rules` they were derived under;
- every `RubricScore` states its `axis` (`capability` or `boundary`). A score your own reviewers gave, not a judge
  call, leaves `judge_temperature` and `served_model` as `None` and records no judge identity for its dim: it reads
  as an unknown judge, never as the same as a recorded one;
- a judge identity in `EvalResult.judge_config_ids` is `builtin:<sha12>` or `<config id>@<sha12>`, as the judge
  service writes it; a bare config id is refused.

## Stability tiers

**Every public name has a tier, and the tier is what a release promises about it.** From 0.70.0, the first tagged
release, on:

- **`host-contract`** is what a host writes against: the host and its profile, the registries, the kind seam,
  launching, the stored documents a kind builds, the store port with its conformance kits and the engine's own
  stores, the refusals, the version constants, the `Known*` vocabularies and their value tuples (`USAGE_ROLES`, …),
  the quick path and what it returns, and every name this guide, the [ladder](ladder.md), the README or an
  example imports. Such a name moves only with
  a `MIGRATIONS.toml` record and a deprecation cycle: the release that moves it exports the new name beside the old
  one, which keeps working for that minor release and is named in its notes, and the next minor release removes the
  old name, recorded as any removal is.
- **`stable`** is the rest of the declared surface: what a topic guide, the agent and MCP guides or the courier host
  imports, the quick path's whole root, and what the one production host binds beyond the contract. Such a name may
  move in any release, recorded in `MIGRATIONS.toml` and printed in that release's notes, never shimmed.
- **`provisional`** is every other name a root exports, most of them because a declared signature reaches them (a
  field's type, a value a function returns, an exception it documents), so you can type your code without reaching
  below a root. Such a name is promised nothing: it may change or move in any release, and the
  [reference](reference.md#public-api) does not list it.

The first two tiers are the declared surface, about 520 names, mapped by rung in [the surface](surface.md).
`packages/evals/api/surface.json` records each name's tier beside its kind (and, for a declared name, its
signature), and `HOST_CONTRACT` and `STABLE` in `packages/evals/scripts/public_surface.py` declare the first two
tiers, with the reason for each group of names. The [reference](reference.md#public-api) lists the declared names
and marks each host-contract one *host contract*. Nothing a host imports is provisional: each adopting host's import
list sits in `packages/evals/tests/fixtures/host_imports/`, and a test fails while a name on one is provisional. If
your host binds to a provisional name, send your list to be added: promoting a name costs nothing, and demoting one
is a break.

**The installed build ships its tiers**, so your suite can gate its own imports against the version it runs
(`surface_tier` raises `KeyError` for a name no root exports, so a reach below a root fails too):

```python
from threetears.evals import surface_tier

def test_we_bind_only_declared_names() -> None:
    ours = ["threetears.evals.kernel.host.EvalHost", "threetears.evals.run.start_run"]  # every name we import
    assert [name for name in ours if surface_tier(name) == "provisional"] == []
```

## Upgrading: names that move

**Every name that leaves a public root is written down, and only a host-contract name is kept for compatibility.**
When a release moves a public name to another root, renames it, or removes it, `packages/evals/MIGRATIONS.toml`
records it under that release: the old name, its new home or `removed`, and why. A `stable` or `provisional` name
has no shim at the old name; a `host-contract` one keeps working for a minor release first (above). The record starts
at 0.70.0, the package's first release, which moved nothing. Pre-release tags are not releases and are not in it.
The file is
in the repository, not the installed package: read it at the tag you are moving to, on GitHub
(`https://github.com/pacepace/3tears/blob/<tag>/packages/evals/MIGRATIONS.toml`) or in a checkout. In a checkout,
the release notes are printed from it, so they hold the full old-to-new map:

```bash
uv run python packages/evals/scripts/release_notes.py v0.70.0
```

Each line reads `old → new`, grouped under the reason for the move. Changes that are not a public name, such as
a renamed field, a moved submodule or a new report version, are listed after the map. To port, find each old name
you import in that list and import it from its new root.

**A name has one home.** It is exported from the root that holds its defining module. A second root exports it
only when that root needs it, and `packages/evals/api/reexports.toml` says why. For example, `ops` re-exports the
analysis results the action catalogue names, because `actions` may reach `analysis` only through `ops`. Import a
name from its home, which is the root the reference lists it under first.

**How the package holds this (for contributors).** `packages/evals/api/surface.json` is a checked-in snapshot of
every public root's `__all__`, one entry per `root.name` with its kind, signature and tier. The same walk that renders
the [reference](reference.md) writes it. `tests/test_api_surface.py` compares the installed package with the
snapshot:

- **A name added, or a kind, signature or tier changed.** The test fails until you regenerate the snapshot, so the
  change shows up as a diff in review. A signature change needs no record in `MIGRATIONS.toml`. When it breaks
  callers, add it to the newest section's `notes`. A tier changes only by editing `HOST_CONTRACT`, and every name
  the README, this guide, the ladder or an example imports must be on it (`tests/test_api_tiers.py`); no name this
  guide's prose teaches may be provisional (`tests/test_guide_names_are_declared.py`).
- **A name moved, renamed or removed.** The test fails until `MIGRATIONS.toml`'s first section (the release being
  prepared) records it. The generator refuses to write a snapshot that drops an unrecorded name. A name the last
  final release did not export never reached a porter, so it needs only a regeneration, and a record of it is
  refused.
- **A section that names the tag before it (`since`).** The test reads both surfaces from git and holds the
  section to exactly the names that left a root between that tag and this version's. Before the version is
  tagged, it compares against the working tree. `since` is always a final release tag (`v<x>.<y>.<z>`). A tag
  from before the package existed (`v0.69.0`, for 0.70.0) has an empty surface. A checkout without the tags fails
  this check rather than passing it.
- **A name exported from two roots.** The test fails unless it is one object and `api/reexports.toml` declares
  it. Two different objects under one name always fail: rename one.

```bash
uv run python packages/evals/scripts/generate_api_snapshot.py          # regenerate (refused while a removal is unrecorded)
uv run python packages/evals/scripts/generate_api_snapshot.py --check  # report every finding, write nothing
```

**When a version is tagged**, rename the `unreleased` section of `MIGRATIONS.toml` to the version, without the
leading `v`. Then paste the output of `release_notes.py v<version>` into the release description. The next release
starts a new `unreleased` section at the top, with `since` set to the tag just cut. A pre-release tag (any suffix
after `v<x>.<y>.<z>`) gets no section, and the record never follows one.

## The kind: what you are evaluating

**A kind is what you are evaluating.** You implement the candidate-kind seam: `prepare` builds a
candidate for one cell from the run's subject, its variant configuration and the seeded world, and
`invoke` runs it on a test case and returns a `CandidateOutput`.

What a kind adds to the engine's own fields is two Pydantic models you name once, on the profile's
`kinds`, in a `KindContract`:

- Its **overlays** are the knobs a launch may turn: validated by field before any run exists, frozen onto
  the run, and read back as levers whose levels enter the variant key.
- Its **spec** is what a template of that kind declares (`kind_spec`), such as a label set or a table
  setup: refused by field at authoring, then validated again and frozen onto each run.

The contract also takes `seats`, the parts of the rig the kind's runs fill (`judge`, `simulator`, or an
apparatus dimension; the default, `None`, holds the kind to every dimension), and `prefix`, the namespace of
its levers (`<prefix>.<field>`; the kind's name when unset).

An overlay field takes markers beside its type, in `Annotated[...]`: `Ordinal()` ranks a `Literal` or `Enum`
in declaration order, `Interval(unit=...)` names a number's unit, `ActsOn(measure)` names the measure the knob
is supposed to move, and `ResolvesInto(lever)` names the host lever the knob is written into. A map field's
entries are distinct knobs, so `ActsOn` is refused there; `MemberActsOn({"max_search_calls": "search_calls"})`
names the measure per entry instead, and every entry it does not list reads `unchecked`.

`ResolvesInto` is for a knob whose effect your host also records, resolved, as a lever of its own: a
`reasoning_effort` overlay and an `llm_parameters` lever hashing the model parameters it resolved into, say.
Without it every sweep of the knob reports two moved levers, each confounded by the other. With
`Annotated[Literal["low", "high"], ResolvesInto("llm_parameters")]`, the surface folds into the knob wherever the
compared runs show every level of the knob carrying one level of the surface: it gets no coverage row or confound
of its own, and the arm is named by the knob (the surface stays in the key, listed in the variant index's
`folded`). Where runs at one knob level carried different surfaces, something else wrote into it, and it stays a
lever and a confound. A run whose surface reader returned `None` folds nothing. On a map field the marker rides on the map's own
lever, and the surface is compared with the swept entries removed.

Only arms can test a fold: every run of one arm resolves the same surface, so only a knob level held by two or
more arms (differing in, say, the model) can show the surface moving on its own. A plain two-arm `low` / `high`
sweep still folds, but every comparison that does carries an `unverified_fold:llm_parameters` confound, which the
analysis generator and the code-only report both state.

A residual reader checks that sweep. `ResolvesInto("llm_parameters", read_residual=reader)` takes a
`reader(run, results, removed)` that returns the surface with the knob's contribution taken out (`removed` is the
knob's lever name), as JSON-safe content, or `None` where the run cannot say: the model parameters without their
`reasoning_effort` entry, say. Where every knob level was run by one arm, the engine compares the residuals across
the runs. Equal, the fold is checked and carries no mark; unequal, something else wrote into the surface and it
stays a lever and a confound; any `None` leaves it `unverified_fold`. A host's own fixed lever takes the same reader
as `Sweepable(resolves_into=..., read_residual=...)`. With no reader, nothing changes.

The lever named must be a fixed `lever` your host declares, not an open family, and not one another knob already
names (`RegistrationError`); a field marked twice is refused (`KindContractError`). The marker moves no variant key.
A host's own fixed lever declares the same thing as `Sweepable(resolves_into=...)`. One surface takes one knob: if
several knobs hash into one surface, give each its own resolved lever.

You register neither anywhere else: the profile adds the contract's levers to its `sweepables`, and the
engine resolves every run's level of them — with its `candidate_model` and its `candidate_kind` — into the
variant key. Your profile's `host_sweepables` (the shared core extended with your own levers) and its
`variant_levers` reader cover only the levers you declare beyond those; a host with none wires no reader.

**How many cells your kind runs at once is yours to declare.** A run not measuring latency executes its cells
side by side, and one kind instance may then be driving several of them: whether that is safe — the clients it
shares across cells, a pool, a world it does not bind per cell — only your kind's code knows. So the kind
declares it, as `max_concurrent_cells` beside `judged_artifact`: `1` is serial, a larger int is that many, and
`None` is no limit of its own. **A kind that declares nothing runs one cell at a time.** Its launcher hands the
same value to the launch (`KindWiring(max_concurrent_cells=kind.max_concurrent_cells)`), which runs the run at
the narrowest of that, the host's `LaunchSettings.max_concurrent_cells` (a ceiling, 4 by default) and the matrix.
The runner holds every cell's kind to it, refusing a kind that declares less than the run executes at before
anything is spent.

**Your kind's paid calls are counted by the run's cost cap when its cell lands, and never refused.** The cell's
sink keeps the cell's spend by source, and the cap is read between cells, against the cells already recorded: a
cell running when the cap is reached finishes whole, and a run may land past it by up to the width's worth of cells
([Cost and budgets: caps](cost-and-budgets.md#caps)). Report each paid call's cost as it lands: wrap the client in
`CappedCompletion(client, sink=sink, source="candidate")`, which does it for every call, or take a handle,
`held = sink.reserve("candidate", ceiling)`, then `held.settle(cost)` when it returns and `held.release()` if it
raises. Spend you learn of any other way, report as it lands (`sink.add_spend(source, cost)`); it adds, and never
replaces what another source reported. Return the cell's spend on its output; that is what the cap counts.
`sink.reserve` never refuses, `sink.cost_cap_reached()` is always `False`, and `CostCapReached` is no longer raised
by the engine: they stay so a host that imports them keeps importing, and a kind has nothing to catch.

A run is one arm with one `candidate_model`; a launch naming several models starts one run per model, and
runs of different kinds are always different variants. A reader of yours that answers for a run its lever
does not apply to (a run of another kind) returns `SweepableValue.not_this_kind(kind)` — the level a kind
contract's own levers sit at there — rather than a value of its own displayed alike: that level is what a
report recognises, by its content hash, as a lever the arm did not run, and never prints.

## Launching

**A kind's launcher resolves only what is the kind's.** `start_run` refuses a launch argument the
kind declares it cannot honour before the launcher runs, then hands the launcher a `LaunchRequest`.
The launcher captures the subject the request names (admission has already asked the kind's `resolve_subject`
whether the host holds it, so a launcher never meets one it does not), freezes the cases, builds its judge
(`build_judge_service`) if it has one, and returns `launch_run(host, request, KindWiring(...))`. The
engine stamps everything the request and the template already say — scope, model, repeats, cassette
mode, overlays, spec, world seed, tool bound, ceilings — and refuses a wiring that contradicts them.

Before any launcher runs, the engine asks each kind what each arm will run (`LaunchableKind.plan_arm`) and
prices it. `plan_arm` is therefore where a kind makes its request-level refusals (no model and no default —
`require_candidate_model` — a judge or simulator it needs), and the launch tail then holds each launcher to
its plan. The full rule is in [Cost and budgets: every arm is priced before any launcher
runs](cost-and-budgets.md#every-arm-is-priced-before-any-launcher-runs).

**Price a floor under your subject kind's judge and simulator calls.** A judged subject kind — a persona, a
classifier — registers `price_subject_call_floor` (from `threetears.evals.run`) as its `price_largest_call`:

```python
LaunchableKind(
    launch=your_persona_launcher,
    resolve_subject=your_persona_subject,
    plan_arm=plan_persona_arm,  # plan_judge(eval_host, request.template, pin, request.judge_config_ids, ...)
    price_largest_call=lambda request: price_subject_call_floor(
        eval_host, request, judged_artifact=JudgedArtifact.TRANSCRIPT  # or DOCUMENT: what your judge reads
    ),
)
```

It prices the smallest judge call every judged case makes, and the smallest utterance your simulated user is
bound to make, with the transcript and evidence left out, on the clients your cells call through. The estimate
and the launch then refuse an arm whose cap cannot afford even that, before anything is spent. That catches a
judge or simulator client whose output cap and retries alone put one call above the cap. Pass the `eval_host`
your launcher builds the judge on and the `judged_artifact` it hands `build_judge_service`. The figure is a
floor, not the largest call (`CallCeiling.lower_bound`): a cap above it is admitted, and a call its transcript
makes larger is still held to the cap as the run reserves it. A kind with no judge and no simulated user has
nothing to price here, and returns `None`.

**A transport launches in two halves.** `run_launch` admits the launch in its call (`admit_launch`: every refusal
above, every arm planned and priced, the runs' room reserved) and runs the rest as one job
(`launch_job_work`): each arm's launcher — where your launcher builds its rig and generates cases — then the runs,
waited for to their end. So your launcher runs inside a job and may call a model; `plan_arm` and your pricer run
in the call and must not. A script awaits both halves in a row through `start_run`.

### Your own actions are jobs too

A host adds its own actions to the catalogue (`eval_catalogue(my_actions)`). One whose work calls a model — a
rubric proposer, a drafter of a classifier's cases — is a `spend` action and keeps the engine's rule: **its call
makes no completion request.** It prices and refuses in the call, then starts the work as a job with
`start_action_job`, which writes an `EvalActionJob` record carrying the job's lease and, once the work returns,
its typed result as JSON — what `job_poll` returns. **Name its wall-clock ceiling** (`budget_s`): no action job
has a default, so a forgotten one is a `TypeError` where you start it rather than an hour nobody chose. Declare
it `long_running=True` with result `JobsStarted`. It
needs no `render`: long work reads back as the jobs it started, with `render_jobs_started` (exported from
`threetears.evals.actions`), which is the default, and which an action returning anything else may not use.

An action whose work is **one paid model call** takes `start_paid_call` (in `threetears.evals.ops`), which keeps
the whole rule for it: it builds the role's client, admits the call against your out-of-run budget in the action's
call (refusing, with nothing started or spent, a call the cap cannot pay for), starts `make` as the job, and
releases the client on every path that does not reach `make`. Work of any other shape builds the same steps on
`start_action_job`:

```python
async def _propose(host: OpsHost, caller: Caller, params: ProposeParams) -> JobsStarted:
    budget = my_budget(host, caller.scope_id)
    call = plan_draft(axis=params.axis, subject_feed=..., catalog_feed=...)   # the shipped prompt

    async def draft(client: BoundCompletionClient, admitted: AdmittedCall) -> RubricProposal:
        async with client:                                    # the job's work owns the client from here
            proposed = await propose_draft(client, admitted, budget=budget, axis=params.axis, subject_id=...)
        return proposed.proposal

    return await start_paid_call(host, caller.scope_id, action="rubric_propose", role="proposer", model=None,
                                 purpose="proposer", budget=budget, call=call, make=draft, label=params.topic,
                                 budget_s=600.0)                  # required: how long a draft may take

async def _propose_happy_path(stage: ConformanceStage) -> Mapping[str, Any]:
    return {"topic": "tone"}                             # what reaches the work, with anything it must arrange

Action(name="rubric_propose", permission="spend", result=JobsStarted, long_running=True,
       handler=_propose, happy_path=_propose_happy_path, ...)
```

Every `spend` action declares a `happy_path`, and one without is refused where it is declared. **Prove it with the
jobs conformance kit**: build your `OpsHost` over a fresh store with your client factory wrapped by the kit's
guard, name a sample launch your host runs end to end and judges through that factory, and parametrise over
the cases. Every action's call must make no completion request; every spend action's happy path must return
`JobsStarted`, and its job, polled to its end, must make at least one — a happy path that never reaches the work
is refused as vacuous. Route every client your actions use through the factory you hand the guard; a client
built outside it is not watched.

```python
from threetears.evals.testing import CompletionGuard, JobsConformanceHost, jobs_conformance_cases

def build(guard: CompletionGuard) -> JobsConformanceHost:
    host = my_ops_host(store=InMemoryDocumentStore(), clients=guard.wrap(my_clients))
    return JobsConformanceHost(host=host, caller=Caller(scope_id="dev", identity="kit"),
                               sample=ConformanceSample(template_id="t", subject_id="s", models=("m",),
                                                        second_judge_model="judge/other"))

@pytest.mark.parametrize("case", jobs_conformance_cases(eval_catalogue(my_actions)), ids=lambda case: case.name)
async def test_no_call_waits_on_a_model(case: JobsConformanceCase) -> None:
    await case.check(build)
```

### Deriving your budgets from the engine's ceilings

The engine holds some of its own work to inner ceilings, and every outer budget you set around that work (an action
job's `budget_s`, the output cap of the client you build for the simulator or the proposer, a retry budget around a
scheduling call, a deadline over a generation or a cell's judge phase) must sit at or above the ceiling inside it, or
you cancel work the engine would still finish inside its own bound, after it was billed. `Ceilings` (host contract,
in `threetears.evals.run`) carries them as one value: `Ceilings()` holds `scheduler_call_attempts`,
`simulator_max_tokens`, `simulator_answer_budget_tokens` and `proposer_max_tokens`, and the two functions
`generation_ceiling_s` and `judge_phase_ceiling_s`, which turn your own ceiling for one request into the ceiling of a
generation or a judge phase. Derive each outer budget from a field, never from a number you copied: a release that
raises an inner ceiling then raises your budget with it, so the budget cannot drift below it. The package exports
no other name for them: the six separate constants and functions `.32` kept beside `Ceilings` were removed in `.33`.

### Hearing about a regression when a run completes

`scope_history` flags a regression only when someone reads it. To hear about one as soon as a run
completes, build a `RegressionWatch` (from `threetears.evals.ops`) over your host, a sink of your own and
the measures to track, and hand it to the `LaunchHost` as `on_run_end`:

```python
watch = RegressionWatch(host=eval_host, sink=page_the_owner, measures=("composite", "cost_usd"),
                        min_absolute_change=0.2)
launch_host = LaunchHost(eval_host=eval_host, ..., on_run_end=watch)
```

When a run's `completed` status is stored, the watch reads each measure's history with the same test and
thresholds `scope_history` uses. For each of the run's contestants whose step into this run reads
`regressed`, it awaits your sink with one `RegressionAlert`. The alert names the contestant, the measure
and the two runs of the step, and carries the history's flag whole: the verdict, its test and its
thresholds. It also states whether code or a judge graded the measure, and the judges' evidence tier.

A judged measure fires only when every judge behind the step's scores is `calibrated` (see
[Judges and calibration](judges-and-calibration.md)). A run that ended any other way than `completed` is
not a point on the history, so it is never checked. The engine ships no delivery: with no watch, nothing
is checked. A sink that raises is logged, the next alert is still delivered, and the run's status never
changes.

### Generating cases at launch

A launch with `n_variations` > 0 asks for that many new cases from the template's variation axes,
generated once for every arm:

- Call `generate_variations` inside `request.launch_group.resolve_once(...)`, passing
  `blocking_executor=host.blocking_executor` (its case reads and writes run there; its model calls stay
  on the loop), and hand its counts on as `KindWiring(variation_counts=...)`.
- An `llm` axis is written by the model the launch names as `variation_model` — required then, refused
  when nothing would call it. The launch's call builds the writer from the host's client factory in the
  `variation` role (never the simulator's), plans and admits its calls there, and hands them to the launcher
  as `request.generation`. The launcher builds no writer of its own.
- The run records the model the client resolved to on `variation_counts.variation_model`; it enters no
  identity, because the cases it wrote are already hashed through `test_case_ids`.
- The launcher hands `generate_variations` the request's `generation=request.generation`, which makes exactly
  the calls the launch admitted. A rubric proposal is planned (`plan_draft`), admitted in your call and made in
  your job (`propose_draft(client, admitted, ...)`) the same way.

Generation runs before any run exists, so its calls are outside every run's cost cap and metered-call
ceiling. How they are priced and ledgered instead is in [Cost and budgets: spend outside any
run](cost-and-budgets.md#spend-outside-any-run).

### Setting the rig at launch

**`apparatus_settings`** sets host-declared apparatus values — who sits in an adjudicator's seat, say —
so one template can be run at two of them and compared. A kind lists the ones its launcher reads, each
with the value its rig takes when a launch sets none
(`LaunchableKind.apparatus_settings={"adjudicator_seat": "model:default"}`, each a non-engine `apparatus`
declaration of the host). The dispatch resolves a launch's settings against those defaults, so the
launcher reads every one off `request.apparatus_settings` with no default of its own, and the run records
the rig as set up (`EvalRun.apparatus_settings`), a component of its measurement context — a launch naming
a default and one leaving it out are one condition.

## Declaring what a campaign decides

A campaign that compares arms to make a decision declares the question it asks and the rule that answers it, before
its first run. The rule is typed clauses, never words: the engine settles a `DecisionRule`, and reads no rule out of
a question's text.

```python
from threetears.evals.kernel import CostLower, DecisionRule, Question, RateDrop

asked = Question(
    text="Does v2 hold every pass rate within 2 points of v1, at a lower cost per reply?",
    decision=DecisionRule(clauses=(RateDrop(max_points=2), CostLower())),
)
```

It goes on the declared design's `questions`. The page quotes the question whole and opens its Decision section with
the rule in a sentence it writes, the arms by name: `Switch to v2 only if no pass rate drops by more than 2 points
and v2 costs less per reply. Otherwise stay on v1.`

- **`RateDrop(max_points=2)`** reads every pass/fail reading the arms share, the overall pass rate included.
  `RateDrop(max_points=2, reading="tone")` reads one: a goal-state check of the template by its verbatim text, a
  pass/fail judged dimension by its name, or a check measure by its key: a true/false measure, its passes counted
  at its better end, or a scorer returning a bool (`no tone rate drops by more than 2 points`). A reading the campaign does not measure is refused when the design is declared, listing the ones it does.
- **`CostLower()`** reads each arm's cost per reply.
- **The logic is fixed**: switch only if every clause settles switch, keep the moment one settles keep, otherwise
  unsettled. Keep settles on what the run observed; switch needs every arm at the case floor and an interval
  ([Reading reports](reading-reports.md)).
- **A rule written as prose is refused** when the design is declared: a question whose text says `Decision rule:`
  and declares no `decision` names `Question.decision` in the refusal. A question stored that way before rules were
  typed still loads; its page says the rule went unread, and its verdict is exploratory.
- **The rule is part of what was asked**: changing it mints a new question that supersedes the old, as changing the
  text does.

## The engine's prompts: shipped defaults, optional overrides

The engine steers three LLM calls of its own with a prompt: the analysis generator (`ANALYSIS_GEN_PROMPT`), the
capability proposer (`PROPOSER_PROMPT`) and the boundary proposer (`BOUNDARY_GEN_PROMPT`). **A new host does
nothing here.** Each call runs the default the package ships (`EVAL_ANALYSIS_GEN_DEFAULT` in `analysis`,
`EVAL_PROPOSER_TEMPLATE_DEFAULT` and `EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT` in `gen`) unless you hand it a prompt
of your own: an `AnalysisGeneration` with no `resolve_prompt`, and `plan_draft(axis=..., subject_feed=...,
catalog_feed=...)` with no `system_prompt` (`default_system_prompt(axis)` is the text it sends).

**An override is yours to keep.** If an operator tunes a prompt without a deploy, keep the copy in your own
registry and hand it in: `AnalysisGeneration(resolve_prompt=..., prompt_id=...)` (the id is recorded on every
analysis as its provenance), and `plan_draft(..., system_prompt=..., reset=...)`. `SEED_PROMPTS` (from
`threetears.evals.kernel`) lists the three, one `SeedPrompt` each: its id (`prompt_type`), its registry kind
(`template` or `text`) and `seed_ref`, the `(module, constant)` its default is imported from. Seed an empty slot
from each and never overwrite a stored one. The engine's ids start with `eval_`, so your own prompts can use any
other name.

**A proposer override states the draft shape it was written for, and the engine checks it.** Each shipped
proposer default ends with its output-contract line, `eval-output-contract: rubric_proposal <fingerprint>`
(`gen.RUBRIC_PROPOSAL_CONTRACT.line`): the digest of the draft's schema as the call sends it, its field
documentation aside, and of a revision for the rules the schema cannot state. `plan_draft` reads an override
before anything is priced and raises `PromptContractError` (a `ConflictError`, status 409) when it carries
another fingerprint (`outdated`) or none (`undeclared`). The error names the prompt and the remedy: reset the slot
to the shipped default, or rewrite the edit's output instructions for the current shape and make its line read the
current one. The registry is yours, so name your reset as `reset=` (an action or a command your operator can
run), which an override requires. An edited copy that carries the current line passes: the line is the editor's
statement that the edit was written for this shape. The analysis generator's prompt states no shape (its schema is
sent and its grammar appended by code), so it binds no contract.

**The drift check is opt-in.** A stored copy seeded from an older default keeps being sent after an upgrade
changes the default. To see that before you pay for it, record `seed_digest(default)` when you seed a slot and read
each slot with `prompt_drift` on start-up:

```python
drift = prompt_drift(entry, shipped=default, stored=stored_copy, seeded_digest=recorded_digest,
                     reset=f"call prompt_reset with prompt_type={entry.prompt_type!r}",  # your own reset
                     customized=operator_owns_it)
if drift.stale or drift.refused:
    log.warning("%s: %s", drift.prompt_type, drift.remedy)
```

A stored copy reads `current` when it is the shipped default, `customized` when it changed since you seeded it or
you say it is deliberately yours (`customized=True`), and `stale` otherwise. `refused` is true when `plan_draft`
would refuse the copy (`contract` is `outdated` or `undeclared`). Pass a template's stored copy as
`SeedTemplate.from_dict(payload)` and a text prompt's as its string. A host that keeps no copies has nothing to
check.

### Launching the analysis reporter

The analysis reporter (`REPORTER_KIND`) evaluates the analysis writer itself. Its cases are frozen campaign
bundles (see [Serving](serving.md#reporter-runs-evaluating-the-analysis-writer-itself)). A
candidate is a preset of the `ANALYSIS_GEN_PROMPT` prompt on a generator model, or `AS_RECORDED_MODEL`,
which replays the memo each campaign got.

The kind is declared by the engine, so your host supplies only two things:

- **Your prompt registry**, as a `ReporterPrompts`: `resolve(preset)` returns that preset's text, or the
  system default for `None`. It returns `None` for a preset you do not hold, never the default in its
  place. A registry that fell back would bill the incumbent prompt under the candidate's name.
- **Your model resolution**, as a `ReporterModelResolver`: `(role, named)` returns the provider model,
  or the role's default when `named` is `None`. The roles are `generator` and `judge`. Raise
  `ValidationFailedError` for a name you do not know.

Declare `REPORTER_KIND_CONTRACT` (from `threetears.evals.analysis`) on your profile's `kinds`. Its one
overlay is `prompt_preset`, and it seats the judge and the spend ceiling. If the profile refuses it for
leaving one of your own apparatus dimensions unseated, build it with
`reporter_kind_contract(host_seats=[...])`. On a reporter template, your `refuse_undeliverable_template`
hook calls `refuse_undeliverable_reporter_template`.

Then register the kind's launch entry, a `LaunchableKind` (from `threetears.evals.run`) whose arguments come from
`threetears.evals.ops` (`REPORTER_UNHONOURED_LAUNCH_ARGUMENTS`, `plan_reporter_arm`, `price_largest_reporter_call`):

```python
LaunchableKind(
    launch=your_reporter_launcher,
    resolve_subject=your_reporter_subject,  # admits the one subject id your reporter launches label
    unhonoured_launch_arguments=REPORTER_UNHONOURED_LAUNCH_ARGUMENTS,
    plan_arm=lambda request: plan_reporter_arm(eval_host, request, prompts=load_prompts(), resolve_model=resolve),
    price_largest_call=lambda request: price_largest_reporter_call(
        eval_host, request, prompts=load_prompts(), resolve_model=resolve
    ),
)
```

`price_largest_reporter_call` prices each case's first generator request before anything is spent, built as
the cell builds it, on the client your factory builds in the `analysis` role for the arm's model. The
estimate and the launch then refuse an arm whose largest generator call is reserved above its cap. Without
it, that call is refused only when the run reaches it, after the cases before it were paid for. Pass the
`eval_host` your launcher builds the generator client on, so the price is that client's. The memo's judge
call stays unpriced, since the memo it judges does not exist yet.

`plan_reporter_arm` and `admit_reporter_launch` make every refusal before the arm is priced, in this order:

1. a template asking for apparatus a memo writer cannot honour, or with no rubric;
2. a preset you do not hold, or no default;
3. a model named by alias (a name you resolve to another model), since the run records the name and an
   alias's target moves;
4. a template with no live case, or a campaign-and-memo pair holding two live cases. When the launch
   names a case set, its cases run instead, and each must carry a reporter case;
5. the replay over a case with no recorded memo.

Your launcher calls `admit_reporter_launch` again, off the event loop. It builds `ReporterKind` (prompt id
`ANALYSIS_GEN_PROMPT`) or `AsRecordedReporterKind` from what that returns, and wires its client, subject,
cell ceiling (`reporter_cell_timeout_s`) and judge as for any kind.

## A world, through the cell's session

Skip this section if your subject is stateless (a classifier, an extractor). A world is for a subject that
acts on something — a ticket queue it can close tickets in, a game table — whose state you want seeded
before each cell and checked after it.

Before building a host for one, try the quick path: `Eval(cases, World(State, tools=[...])(agent), [goal checks])`
declares a small world, seeds each case's starting state (its `world` key) through this same session, hands the candidate
tools that act on it and grades the end state with the engine's goal-state checks (`examples/4-booking/agent.py`).
What follows is the host-side contract that path is built on.

**A world, through the cell's session.** A host whose subject lives in a stateful world declares it
on the profile (`WorldRegistry`), and each cell's `prepare` is handed a `WorldSession` over it as
`world` (`None` on a host with no world).

- **Seeding.** The kind seeds through it — the seed walk, the dimensions' own `seed` handles, then each
  attached carrier's `settle` handle before the first turn.
- **Turns.** It announces each turn (`at_turn`, which applies any ambient perturbation the template's seed
  scheduled), and fires or observes its triggered dimensions (`fire`, `observe`; triggered dimensions are
  [provisional](#provisional-mechanisms)).
- **Reading back.** Once `invoke` returns, the runner reads every attached dimension back through its
  `read` handle and stores it as the cell's end state; what fired is stored on the result as
  `world_events`.
- **Per-cell worlds.** The profile's registry is the declaration and its handles are the world
  conformance proves; if your world is real per-cell state, declare
  `WorldRegistry(..., binds_per_cell=True)` and have each cell's `prepare` call
  `world.bind(<that cell's handle table>)` before `seed` — every call the session makes then lands in
  that cell's world, and a session that seeds unbound is refused.
- **Grading.** A goal check reads the end state as `state.<dimension>`, the calls the kind recorded on its
  `CallLedger`, and what fired as `fired("<dimension>")`; grade them with `grade_goal_checks`.
- **Witnessed cells.** The runner's session is `commissioned`; a host grading a cell it witnessed through
  a session of its own constructs `WorldSession(registry, provenance="witnessed")`, so `fired_armed(...)`
  is not established there — no seed armed the session — and `record_witnessed_cell` refuses any world
  event claiming `armed=True` or `caused_by="rig"`.

## What the judge reads, and conversations

**Every judged dimension your templates carry declares its judge.** `RubricDim.judge` is `"builtin"` (the engine's
own prompt) or `"configured"` (the scope's active `JudgeConfig` for that dimension), in your seed corpus and in
anything you author. `create_template`, `seed_eval_definitions` and an `update_template` that writes `rubric`
refuse a dimension that declares neither, naming every one at once; a `"configured"` one needs its config active
(the seed counts a config the same corpus seeds), and a `"builtin"` one must have none. Every launch checks the same
before any spend, so a template stored before the declaration existed is refused until `update_template` declares
its dimensions, and a configured dimension whose config was archived since is refused rather than scored by the
engine's prompt. A quick-path `Judge` declares its own dimensions, so an `Eval` that grades with one states
nothing.

**Your boot seed corpus carries authored judge cases too.** Beside `templates/`, `rubric_dims/` and `judge_configs/`,
`load_seed_corpus` reads `judge_cases/*.json`, each a `SeedJudgeCase` (host contract, from `threetears.evals.run`;
in memory, `SeedCorpus(judge_cases=...)`): `judge_case_author`'s arguments with its two templates named by `name`
(`judge_template`, `criterion_template`, since a corpus cannot know a seeded template's id), the `criterion`, the
`output`, its `case_material`, the `expected` verdict, your `note` and the case's `author`. `seed_eval_definitions`
builds each through the door `judge_case_author` uses (`author_judge_case`: the criterion read from the template
that words it, the verdict held to its scale) before anything is written, refusing the whole seed by name for any
case authoring would refuse, then freezes it into its judge template after the templates are written, as the same
`AuthoredJudgeCase` under the same content-derived id. A case already stored is reported `already present` under
`judge_case`. A launch naming no case set runs every live case of the judge template, so a host boots with its hard
negatives in place and no authoring tool of its own over `judge_case_author`. **Since `.30` the seed fills a case only
into a judge template holding no cases yet** (one it creates in the same pass, or an empty one): a new case, or one
whose words or verdict changed, for a template that already holds cases would change what its runs were measured
against under the same id. It is no longer written beside the old ones. It is withheld and reported under
`needs_reset` on every boot, with any case set listing it, until you reset the template (below).

**A template's own cases seed with it.** A template whose stimuli you author as cases rather than variation axes (a
scenario to play out, a set of inputs to classify) carries them in the corpus too: `load_seed_corpus` reads
`template_cases/*.json`, each a `SeedTemplateCase` (host contract, from `threetears.evals.run`; in memory,
`SeedCorpus(template_cases=...)`) naming its `template` and carrying the case's `variation_params`, its `host_payload`
(your stimulus, stored as written and never read by the engine) and its `stratum`. Its name is its file's name without
`.json` (or its own `name`), unique within its template, and is stored as the case's `name`, which the report calls it
by. The template must be one the corpus carries, and not a judge template, whose cases are its judge cases. The seed
writes the cases into the template's current revision after the templates and reports them under `template_case`. As
for a judge case, it writes them only into a revision holding no cases yet, so a case you add, or one whose content
changed, for a template already holding cases is withheld under `needs_reset` until you reset the template (below).
A case whose content the revision already holds is `already present`, so cases your own seed wrote before `.32` with
the same content are kept as they are.
A template whose every revision is archived is one an operator retired, and is absent to the seed as an archived
rubric dim or judge config is: no case is written into any of its revisions, on any boot, and each case is reported
under `SeedOutcome.archived` (`template_case`, keyed `<template>/<name>`), neither created, present nor failed.
A stored case whose `name` is empty and whose content matches a corpus case (its `variation_params` and `host_payload`,
as its id is derived) is given the corpus case's name on a boot, and reported under `SeedOutcome.named`. It is
fill-only: a name a stored case already holds is never overwritten, and only `name` is written, so the case keeps its
id, `content_hash` and `seed_digest` and its history.

**A host that validates its cases moves onto `template_cases` with its check.** A classifier bank whose every case
carries a label and an input moves its cases into `template_cases/*.json` (`host_payload` is `{"label": ..., "text":
...}`) and its validation into the one check the seed, a reset and `case_rename` all ask:

```python
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.host import TemplateCaseFields, TemplateChecks

LABELS = {"refund", "complaint", "praise"}

def refuse_bad_bank_case(case: TemplateCaseFields) -> None:
    payload = case.host_payload
    if payload.get("label") not in LABELS or not isinstance(payload.get("text"), str):
        raise ValidationFailedError(
            f"case {case.name!r} needs a `label` of {sorted(LABELS)} and a string `text`",
            remedy="correct the case file",
        )

checks = TemplateChecks(..., refuse_inadmissible_template_case=refuse_bad_bank_case)
```

A corpus holding a case it refuses seeds nothing and names the case with your message.

**A stored case is renamed through the package, never by writing its document.** The seed and a reset fill a name
only into a case they write, so `case_rename(host, scope_id, case_id, name)` (from `threetears.evals.ops`, over your
`EvalHost`; the `cases_rename` action takes `test_case_id` and `case_name`) is how an operator's rename lands. It
writes the case's `name` and nothing else, through the store's own write path for a case, so the case keeps its id,
its `content_hash`, every result measured against it and its seed record (`seed_name`, `seed_digest`: a renamed seeded
case still reads stale or customized by its content alone), and the next report calls it by the new name. It answers
with a `CaseRenamed` (the case's id, its old and its new name). It refuses, as the action does, a name the report
would not print (`humanize("case", name)` empty: spaces, a bare number, a generated id, over 40 characters), a name
another case of the same template or of a case set listing it already carries (`ConflictError`), and an unknown case
(`NotFoundError`). A reset writes the corpus's names into the template's next revision, so a rename the corpus should
keep belongs in the corpus too.

**A case set a launch targets seeds from the same corpus.** A template carries no list of its cases, and an authored
case's stored id is derived from its judge template's generated id, so a set such as `hard_negatives v1` is neither
implied by a template nor writable as ids: `load_seed_corpus` reads `case_sets/*.json`, each a `SeedCaseSet` (host contract, from `threetears.evals.run`; in
memory, `SeedCorpus(case_sets=...)`) with the set's `name`, its judge `template` by name and its `judge_cases` by
their corpus names, in order. A judge case's corpus name is its file's name without `.json` (or its own `name`, which
an in-memory case must set to be listed). A set naming a case twice, a case the corpus does not carry or a case of
another judge template refuses the whole corpus when it is built. Once the cases are frozen, each set whose name the
scope holds no version of is minted over their stored ids through `mint_case_set`, the path `case_set_mint` takes,
and reported under `case_set` as `<name> v1`; a scope holding any version keeps it (the next version is
`case_set_mint`'s), so a host's own minting script over `case_set_mint` can be dropped.

**A corpus edit reaches a seeded scope only through a reset.** The seed fills empty slots and never updates, so a
template whose cases grew in git stays as first seeded. Every definition the seed writes records the digest of what it
wrote, and a release step reads drift and resets what is behind, deliberately (never at boot, so a store an operator
tuned is never rewritten by a deploy):

```python
from threetears.evals.run import definition_drift, reset_definitions

corpus = boot_seed_corpus(scope_id)  # the SeedCorpus your boot seed loads for the scope
drift = definition_drift(host, corpus, seeded_digests=PRE_30_DIGESTS)  # writes nothing
print(drift.render())  # "drift: 1 stale, 0 customized, 1 unknown, 0 absent, 40 current", then each line not current
if stale := drift.keys("stale"):
    reset = reset_definitions(host, corpus, stale)  # admits each template under the host's template_checks
    print(reset.render())
```

A definition reads `current` (the corpus's), `stale` (what was seeded, and the corpus has moved; also a template
whose boot reported `needs_reset`), `customized` (changed since: edited, archived or deleted), `unknown` (no record:
a row seeded before `.30`) or `absent` (the next boot seeds it).

A reset never edits in place. A template's new content is its next **revision**: a new record under the same name
(`EvalTemplate.revision`), the old one archived and readable by id, so a run, its results and its cases keep the
content they ran against, and a launch, the seed and `load_template_by_name` resolve the name to the newest revision,
which `templates_list` shows beside each template (`revision`). A judge template's new revision holds the corpus's
whole battery, and the corpus's case sets over it are minted again; a template the corpus gives cases gets every one
of them in its new revision. The reset refuses `customized` and `unknown`, and a judge template (or a template the
corpus gives cases) holding live cases the corpus does not carry, unless `overwrite_customized=True`; it writes
nothing when it refuses and names every refusal. Each row of the result (`DefinitionReset.reset`) names what it wrote
and the stored record it replaced (`replaced_id`), and reads `discarded: true` only where that stored copy read
`customized` or `unknown` and was written over under the flag. A campaign holding runs of two revisions of one
template is refused (at creation, adding a run, and its bundle), since their results are two datasets; two templates
that only share a name at one revision number are not revisions of each other and are not refused.

Rows seeded before the seed kept a record read `unknown`: judge cases and templates seeded before `.30`, and template
cases your own seed wrote before `.32`, unless the corpus still has their content (`current`), and any new one.
Compute `corpus_digests(host, corpus)` over the corpus as you seeded it (from your history) and pass it as
`seeded_digests`: a row the map names then reads `stale`, or `customized` when gone from the store, and a case it
does not name reads `stale`. `definition_drift_actions(boot_seed_corpus)` (from `threetears.evals.actions`) adds
`definitions_drift` and `definitions_reset` to your catalogue; the reset reads the template checks of the host each
call is made in, so the factory needs only your corpus.

**A stored verdict is graded again through one door: `regrade`** (`threetears.evals.run`, and the `result_regrade`
action). `await regrade(host, result_id, scope_id)` re-grades the result's check dims (its goal checks) from what its
cell stored (the trace's call ledger and end state, and the result's world events) through the function the cell was
graded with, and names its judged dims whose judging failed; it returns every verdict that moved, old beside new,
and each check outcome it left as stored, with why. It writes nothing until `apply=True`, which stores the moved
check verdicts (refused if the result changed since it was read) and asks the failed judged dims again through the
re-judge, which spends. `checks=False` or `judges=False` leaves either half out. The action never asks a judge:
applying over it while a judged dim would be asked is refused, so a host re-asks those from its own code.

**What the judge reads, the kind renders.** A judged kind returns `JudgeEvidence` with every
non-empty output: the subject as the judge should see it, the case material, and the artifact (for a
conversation, the transcript as your kind writes it). The engine places those strings and reads none
of them, so hidden information and per-player visibility are your kind's rules. The evidence is
stored on the cell's `EvalTrace`, and a re-judge sends exactly what the first judge read.

**Conversations.** A conversing kind's template carries a `ConversationSpec`: its simulated actors, who
speaks next, and the turn limit. A document or classifier template carries none. The kind runs it with
`drive_conversation(driver, candidate_turn, post_user_turn, llm=..., sink=sink)`, handing over its cell's
sink: every simulator call goes through it, and its cost lands on the cell's `simulator` total. The loop asks no
ceiling: a conversation runs to its turn limit or its actors' departure, and the run's cost cap is read between
cells, so a conversation the run has outspent still finishes. What bounds one is its turn budget and the cell
timeout. Fold the driver's calls with `fold_usage`: the stored
`simulator` rows are one per actor and purpose (`RoleUsage.actor_id`, `RoleUsage.purpose`).

**A round can be the world's.** `ConversationSpec.world_rounds` names rounds whose stimulus is a triggered
world dimension firing rather than an actor's line. Before that answer, the loop fires the dimension through
the cell's world session (pass `world=`), and the case's seed must arm it. Your `candidate_turn` receives one
turn spoken by `WORLD_SPEAKER` naming what fired, and nothing is posted. A template whose world supplies every
round declares no actors at all, and its conversation makes no simulator call.

## Background work, payloads and spend

**Background work** a candidate hands off and gets back turns later is recorded as `async_deliveries`,
one `AsyncDelivery` each: who asked, when it was acknowledged and delivered, on what model, whether a
harness supplied it, and what it spent — its tokens, model calls, `cost_usd` with its own
`price_source`, and any paid non-LLM calls (`external_spend`) — their calls, the provider's own units, and
`money` where the provider itself billed a figure. The engine folds that spend into the result's
`inner_agent` and `external` usage, for work still in flight when the cell ended as well as work that
delivered; a substituted entry reports no spend. Both shapes and both roles are
[provisional](#provisional-mechanisms).

**Payloads.** Anything else your kind wants kept with a result goes in `kind_payload`, which the engine
stores and never reads; register a measure for whatever should be compared.

**Spend.** Your kind reports its own calls' spend only as usage rows (`CandidateTelemetry.usage`). How
the engine turns those rows into a result's `cost_usd`, and what happens when a call cannot be priced, is
in [Cost and budgets: how a result's cost is counted](cost-and-budgets.md#how-a-results-cost-is-counted).

## Cassettes: recording and replaying tools

A [cassette](concepts.md#cassette) lets two arms face exactly the same tool answers. An `Eval` does the wiring
below for you (`World(tools=[...], recording=path)`); this section is for a kind of
your own.

**Cassettes record and replay through seams the kind supplies.** Launch a run with
`cassette_mode="capture"` to run its tools live and record what they answered into that run's own
corpus, or `"replay"` with `cassette_corpus_id` naming a capture run to be served that corpus instead.
The engine builds the run's lane and hands each cell's `prepare` a `cassettes` handle (`None` with
cassettes off) already bound to the corpus, template and case, so one kind instance serves every
cell.

- **Wiring.** Your kind calls `cassettes.wire(seams)` once with its candidate's `CassetteSeams`: an
  `ActionSeam` for the synchronous tools to wrap, and a `DeliverySeam` for each asynchronous tool, keyed
  by that tool's name.
- **Tool shape.** A wrapped tool is a `ToolLike`: a `name`, `can_dispatch(action)` and an async
  `act(action, parameters)`; a tool that is a plain function is adapted to that shape.
- **Blocking tools.** A candidate whose turn loop calls its tools as plain blocking calls declares a
  `SyncActionSeam` instead (`arm_sync_tools`), its tools are `SyncToolLike`
  (`act_sync(action, parameters)`), and it calls the wrapped tools' `act_sync`. Both seams record and replay through one
  implementation, so a corpus captured on either replays on either.
- **Background work.** A delivery seam reports each piece of work where it starts (`recorder.started(request)`)
  and settles it where it ends; under replay it takes `replay.next(request)` instead, reporting
  `substituted=True`.
- **Matching.** Every recording is keyed by what was asked and which time it was asked, so a repeated dice roll
  replays both rolls in order; an ask the capture never made, or made fewer times, stops the cell as the rig's
  failure rather than running live.
- **No silent live runs.** A kind that does not wire the handle it was given, or a candidate with no
  seams, is refused rather than run live under a replay.

## Rig failures: a broken rig costs one cell, never the run

**A broken rig costs one cell, never the run.** A replay miss, a corrupt recording or any other
fault of the measuring rig rather than of the world under test is an `ApparatusError`; your kind's
tool boundary re-raises it instead of turning it into an ordinary tool failure. Out of `prepare` or
`invoke`, the engine records that cell excluded (termination `apparatus_failed`) with whatever spend
the kind had reported through its sink, and goes on. A run whose every cell the rig excluded measured
nothing, and ends `failed`.

A cell cut off by its deadline or by its run's cancel is recorded the same way, from its sink; when the
cut lands while the cell is being judged, the evidence and the scores that came back are kept and the
unfinished dims are the cell's judge errors, which a re-judge can repair. A re-judge re-asks only dims
that errored: a judge's "can't tell" is an answer. Every dim the judge phase asked records the prompt that asked
it in `EvalResult.judge_config_ids`, an unfinished one included: a config's id with the reply format's
fingerprint (`<config id>@<sha12>`; `judge_config_of` reads the id back), or the built-in prompt's `builtin:`
identity. A re-judge refuses a dim whose recorded identity is not the prompt it would send today.

## Provisional mechanisms

Each mechanism below exists because the first host to adopt the engine needed it. Each was generalised out of
that host's vocabulary, and none has yet been used by a second host. Use whichever fits your subject. Your
adoption, as the second host, decides whether each one stays an engine concept. A mechanism no second host uses
may be removed in a later release, and that release's `MIGRATIONS.toml` notes say so (the core ones then move
with a core major and an upgrader). None is deprecated, and nothing here changes what is stored. Two of them hang
off [`host-contract`](#stability-tiers) names: `KnownUsageRole` carries `inner_agent` and `external`, and
`WorldDimension` reaches `Triggered` through its `when`. Removing either is a signature change to a contract name,
so it goes through the deprecation cycle, never straight out in one release's notes.

- **`AsyncDelivery` and `AsyncExternalSpend`** (on `EvalResult.async_deliveries`, a core document). They cover
  background work a candidate hands off and gets back turns later, with its own model spend and the paid
  non-LLM calls it made. Skip them and the engine does not see work that finishes after the turn that asked
  for it: its spend, its timing and whether a harness substituted its payload are yours to keep, in
  `kind_payload`, which the engine never reads ([Background work](#background-work-payloads-and-spend)).
- **The `inner_agent` and `external` usage roles** (`KnownUsageRole`). They put a background task's own model
  calls and its paid third-party calls on spend lines of their own. Skip them and that spend is not in the
  role breakdown or the blended cost.
- **`MeteredCallLedger`** (`run.metering`). It bounds the third-party calls a run's tools may make, counted at
  your action dispatcher (`max_metered_calls`), for a provider quota the cost cap cannot see (search credits,
  a billed API). The ceiling is read between cells against the calls of the cells already recorded: once it is
  reached no new cell starts and the run ends `budget_stopped`; it never refuses a call inside a cell. A host that
  declared no metered tools is the exception: a metered call there is refused at dispatch. Skip it and a run's
  third-party quota is bounded only by your tools' own configuration.
- **Triggered world dimensions** (`Triggered`, `ConversationSpec.world_rounds`). State that arrives on a
  condition (a turn, an event, a person) rather than at the first turn: a case's seed arms it, and a world
  round or your kind fires it. Skip them and your world is set once, before the first turn, and no goal check
  can ask what `fired(...)` ([A world](#a-world-through-the-cells-session)).
- **`SeedSection`'s zones** (`primacy`, `reference`, `recency`, in `kernel`). The sectioned prompt template the
  two proposer prompts are declared in, ordered by zone, each section with its own condition, formatter and
  cache class. A host with no registry for sectioned templates never meets them: `plan_draft` sends the shipped
  default when handed no prompt ([The engine's prompts](#the-engines-prompts-shipped-defaults-optional-overrides)).
