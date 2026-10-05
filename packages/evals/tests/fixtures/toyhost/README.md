# The toy host

A complete host of the eval engine, written as reference code: **invoice field extraction**. No
conversation, no simulated user; scalar levers rather than prose blobs; an
observational corpus beside a commissioned run. It imports the engine's public roots and itself,
and nothing else.

## What a host writes, and where it is here

| Step | File | What it holds |
|---|---|---|
| Measures, bars, style | `profile.py` | The five measures, two bars in opposite directions, a style, a caveat kind the engine does not own, and the apparatus the host declares inapplicable — assembled into the `HostProfile` |
| Levers and apparatus | `sweepables.py` | The declared inputs and the two pinned roles, extending the engine's `SHARED_CORE` |
| How one run resolves its levers | `variant.py` | The host's `VariantLeverReader`; the engine resolves the model, the kind and the kind contract's levers itself |
| The world | `world.py` | Seven world dimensions, their handles, the mutable world they move, and switchable faults |
| The candidate kind | `kind.py` | The `CandidateKind` — an invoice extractor and the scripted client that drives it |
| The product's own call, and proof the eval shares it | `product.py`, `fidelity.py` | The one constructor of an extraction request, which production and the kind both call, and the `FidelityContract` registering it; `tests/test_fidelity_adoption.py` is the source canary over it |
| What a launch may turn | `contract.py` | The extractor's `KindContract`: its overlay model and its template spec model |
| The host value | `host.py` | The one `EvalHost` handed to every entrypoint |
| Launching | `launch.py` | The `LaunchHost` and the extractor's launcher, for `start_run` |
| Tracing | `tracing.py` | The host's own `TraceSink`, implemented without OpenTelemetry |
| A model judge | `judge.py` | A rubric and a scripted `CompletionClient` the engine's `JudgeService` scores through |
| Data | `corpus.py`, `run.py`, `campaign.py` | A hand-built observational corpus, a run path through the runner, and a declared campaign over each |

## Adopting the engine: `start_run`

A product launches runs through **`start_run`** (`threetears.evals.run`), on a `LaunchHost` that
composes its `EvalHost` with a launcher per candidate kind (`launch.py`). `start_run` loads the
template, refuses what no kind can run, validates the launch's overlays against the kind's
contract, admits the runs against the host's settings, and starts each arm as a job; the kind's
launcher builds only what that kind has (its subject, its extractor, its cases) and hands the shared
tail, `launch_run`, a typed `KindWiring`.

**`execute_run` is the low-level path**: one assembled run's matrix, nothing else. It sets no run
status and stamps nothing a launch stamps. `run.py` drives it directly because what it pins is the
runner's own output, cell by cell.

## What each element exercises

- **Scalar levers** (`chunk_tokens`, `retriever_top_k`) — numeric axes with real spacing.
- **A kind contract** (`contract.py`): the overlays are an ordinal, an interval with a unit, a text
  joined by content and an open family (`field_aliases`), each read as a lever `extractor.<field>`
  with no per-knob code; the spec is which invoice fields a template grades, validated at authoring
  and at launch, frozen onto each run and honoured by the extractor. The contract is named once, on
  the profile's `kinds`: the profile registers its levers and the engine resolves their levels.
- **An open family on the registry** (`retrieval_overrides`) and the resolved surface it is merged
  into (`resolved_retrieval_config`), so one overlaid knob moves two levers. Opt-in:
  `toyhost_profile(tunable_retrieval=True)`.
- **A designed `unknown`** (`ocr_engine_version`) — a value the host cannot always record.
- **A non-model grader** (`grader_version`) nominated into the engine's own `judge` role, and **a
  role the engine never declared** (`adjudicator`): who graded the work is a seat every product has.
- **A model-graded variant** (`judge.py`): the same extractor and invoices under a rubric, scored
  by the engine's `JudgeService` through a scripted judge that grades what it was shown. The
  profile still declares the judge axes inapplicable host-wide, which is not true of this variant.
- **Measures in four value shapes** — a bounded ratio, two unbounded quantities in opposite
  better-directions, and an unbounded count — across **all four merit axes**.
- **Two bars in opposite directions**, so the ratchet's lower-is-better branch runs.
- **One world dimension in every registrable quadrant**, including `witnessed`.
- **A caveat kind the engine does not own** (`adjudication_scope`).
- **A style unlike the default on every axis.**
- **Switchable faults** (`ToyWorldFaults`), so every `failed` verdict of the world conformance kit
  has a fixture, and **a poorer variant of the same world** (`optional_capabilities=False`).

It does not declare a `no_own_coordinate` waiver or a measure containment: its shape does not raise
those questions. Read the contract, not this fixture, for the full vocabulary.

## Two corpora, and why both

The **hand-built** corpus (`corpus.py`) forces designed states a run cannot: an apparatus dimension
one side never recorded, a sweep with a rival explanation moving underneath it, a pair whose grader
version moved. Those are statements about what the engine ACCEPTS.

The **run path** (`run.py`) is a statement about what the engine PRODUCES: a template naming
`candidate_kind="toy-extractor"`, two extractor models over three invoices at `k=2`, driven through
the engine's runner — twelve results the runner wrote, which the analysis then bundles. Its own
measure (`field_accuracy`) reaches each result through `CandidateOutput.host_measures`, and the
fields an extraction missed through `kind_payload`, which the engine stores verbatim and never reads.

The two campaigns' observational shapes are deliberate opposites: the corpus campaign declares an
uncontrolled stimulus and a witnessed apparatus, the run-path campaign a controlled stimulus and a
commissioned one.

## The store

`host.py` builds its `EvalStorage` over `InMemoryDocumentStore` (`threetears.evals.storage`),
the engine's in-memory reference `DocumentStore`: scoped, with conditional writes, and nothing
persisted past the process. A product passes its own `DocumentStore` there, and proves it with the
store conformance kit (`threetears.evals.testing`).

## The smallest host

`../courierhost/` is the least a product writes: one module, its six steps in order, and one drive.
Start there for the shape; come here for each element in use.
