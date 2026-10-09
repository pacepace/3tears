# Prior art

Read this when you want to know which published practice a part of the engine follows, where it departs from the
field and why, or how the industry's words map onto the engine's. Sources were checked against their own text, and
verdicts against the package code, in 2026-10. **Adopted** means taken as recommended, **adapted** taken with a stated
change, **not adopted** declined on purpose, and **not built** accepted but absent ([open problems](open-problems.md)).

## Methodology and statistics

| Source | What it recommends | Here | Why |
|---|---|---|---|
| [Miller, "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640) (2024) | Report the standard error. Cluster it when questions come in related groups, where it reached about 3× the naive value on one benchmark. Resample each question K times, and compare models on paired question-level differences. Size an eval by power analysis: about 1,000 questions to detect a 3-point gap. | Adapted | Repeats are averaged within a case, and arms are tested paired on shared frozen cases (Welch's test when they share fewer than two). A reading's standard error is Miller's cluster-robust form over cases, read on `n_cases − 1` degrees of freedom, and a rate's Wilson interval uses the effective sample size the clustering leaves. There is no power pre-flight. |
| [Bowyer, Aitchison & Ivanova](https://arxiv.org/abs/2503.01747) (ICML 2025); [Brown, Cai & DasGupta](https://doi.org/10.1214/ss/1009213286) (2001) | Central-limit intervals undercover below a few hundred datapoints; use exact or Bayesian methods, and Wilson or Jeffreys intervals for proportions. | Adapted | Every 0/1 measure takes a Wilson interval, so 3 of 3 keeps a width. Means use t, clipped to the measure's scale. In the run-history read, a zero-variance gap counts as significant only from six pairs, the first n at which an exact sign-flip test can reach α=0.05; a campaign contrast with zero spread reads `untested` at any n. |
| [Holm](https://www.jstor.org/stable/4615733) (1979); [Benjamini & Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) (1995) | Control error across a family of tests: the family-wise rate (Holm) or the false discovery rate (BH). | Adapted | Holm runs within each declared question's family, or campaign-wide when the campaign declares no question, and a verdict reads the adjusted p. There is no FDR across campaigns and no sequential testing. The run-history lens tests each adjacent pair of runs uncorrected. |
| [Lakens, equivalence testing (TOST)](https://doi.org/10.1177/1948550617697177) (2017) | Claim "no difference" only when the interval sits inside a declared equivalence margin. | Not built | Campaign comparisons say `not_separated`, which claims no null. A host's materiality threshold marks a delta as too small to act on, but no equivalence test runs. The history lens still labels a sub-threshold move `flat` ([open problem](open-problems.md#a-sub-threshold-change-in-history-reads-flat)). |
| [Gelman & Stern](https://doi.org/10.1198/000313006X152649) (2006) | Test a difference directly. "A moved and B did not" is not evidence that A and B differ. | Adopted | Each contrast is tested against the control itself, never inferred from two separate verdicts. |
| [τ-bench, Yao et al.](https://arxiv.org/abs/2406.12045) (2024); [Chen et al.](https://arxiv.org/abs/2107.03374) (2021) | pass^k is the chance that all k trials succeed, estimated without bias as E[C(c,k)/C(n,k)] from n ≥ k trials. pass@k is "at least one of k". | Not built (estimator) | The engine's pass^k is the all-pass indicator per case, with the scored depth reported beside it. It is unbiased only at uniform depth, flatters mixed depth, and cannot use extra trials ([open problem](open-problems.md#passk-is-the-all-pass-indicator-stored-under-the-opposite-name)). |
| [Inspect](https://inspect.aisi.org.uk/reference/inspect_ai.scorer.html) (UK AISI) | Epoch reducers, including `pass_at` and `pass_k`; `stderr(cluster=<metadata key>)`; bootstrap stderr; LLM roles named and resolved late. | Adapted | Inspect serves as the vocabulary reference. The judge and simulator models are resolved at launch and recorded in the [measurement context](concepts.md#measurement-context). Stderr is clustered by case on every reading; bootstrap stderr is not built. |
| [Kohavi, Henne & Sommerfield](https://ai.stanford.edu/~ronnyk/2007GuideControlledExperiments.pdf) (2007); [NIST/SEMATECH e-Handbook](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm) | Factors and levels; control against treatments; one overall evaluation criterion; blocking on nuisance factors; A/A tests to measure the system's own noise. | Adapted | A [lever](concepts.md#lever) is a factor, and the [rig](concepts.md#apparatus-rig) is the held-constant nuisance factors. Arms launched together share drift, which is blocking. The composite stays secondary to per-dimension results. A/A runs are not a built practice. |
| [Kapoor et al., "AI Agents That Matter"](https://arxiv.org/abs/2407.01502) (2024) | Optimise accuracy and cost jointly, and hold out data against overfitting. | Adapted | The frontier ranks on what shipping costs, and the cost of measuring is kept apart ([cost and budgets](cost-and-budgets.md)). Dominance is decided on point estimates. There is no held-out split. |
| [Center for Open Science, preregistration](https://www.cos.io/initiatives/prereg) | State the plan before the study; keep confirmatory work apart from exploratory work. | Adapted | A [campaign](concepts.md#campaign) declares its design, and the bundle compares it with the design actually realised. Findings carry no confirmatory or exploratory tag. |
| [Messing, "Hidden Measurement Error in LLM Pipelines"](https://arxiv.org/abs/2604.11581) (2026); [Kotawala, "Resolution Diagnostics for Paired LLM Evaluation"](https://arxiv.org/abs/2605.30315) (2026) | Naive standard errors omit judge and prompt variance. Invert the paired test to find the smallest detectable effect; many published gaps do not resolve. | Not built | One judge per dimension; its noise sets the evidence tier, not the interval. No minimum-detectable-effect pre-flight ([open problem](open-problems.md#no-power-pre-flight)). |

## Judging and graders

| Source | What it recommends | Here | Why |
|---|---|---|---|
| [Anthropic, "Demystifying evals for AI agents"](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) (2026-01) | Use code graders where possible. Grade the outcome, not the path. Give each dimension its own judge, with an "Unknown" way out. Separate capability suites from regression suites and watch for saturation. Start from 20–50 real failures. Test where a behaviour should and should not occur. Give each task a reference solution. | Adapted | [Goal-state checks](concepts.md#goal-state-check) are code, and each dimension is one judge call. "Can't tell" excludes the trial and is counted. Reference solutions become goal-check controls: a do-nothing end state and an authored one must get opposite verdicts, or the check is refused. Suite purpose tags and saturation tracking are not built. |
| [Zheng et al., MT-Bench](https://arxiv.org/abs/2306.05685) (2023) | Judges show position, verbosity and self-enhancement bias. Strong judges agree with humans over 80% of the time, as often as humans agree with each other. | Adapted | Pointwise scoring removes position bias. Verbosity bias has no control. Self-enhancement is handled in the next row. |
| [Panickssery, Bowman & Feng](https://arxiv.org/abs/2404.13076) (2024) | LLM evaluators recognise and favour their own generations. | Adopted (disclosure) | When the default judge is one of the candidate models, a configured alternate judge substitutes. Without one, every surface that lists judges says so. |
| [Husain, "Using LLM-as-a-Judge for Evaluation"](https://hamel.dev/blog/posts/llm-judge/) (2024) | Use binary pass/fail with a written critique. Validate the judge on a train/dev/test split of expert labels, and report true-positive and true-negative rates. | Adapted | `pass_fail` ships beside the 1–5 scale and every dimension states its scale; prefer it for new criteria. Agreement with people is Cohen's κ, not TPR/TNR, and labels are not split. |
| [Husain & Shankar, evals FAQ](https://hamel.dev/blog/posts/evals-faq/) | Do error analysis first: open coding, then axial coding into a failure taxonomy, then about 100 traces to saturation. Generic metrics create false confidence. | Not built | No error-analysis workflow ships. The rubric proposer drafts criteria, and reviewing them against observed failures is left to the author. |
| [Yan, "Evaluating the Effectiveness of LLM-Evaluators"](https://eugeneyan.com/writing/llm-evaluators/) (2024) | Use binary outputs where possible and pairwise comparison for subjective qualities. Measure agreement with κ, τ or ρ. | Adapted | κ is adopted. Pairwise judging is declined for now ([open problem](open-problems.md#pairwise-judging-declined-for-now)). |
| [Shankar et al., "Who Validates the Validators?"](https://arxiv.org/abs/2404.12272) (2024) | Criteria drift: grading outputs changes the criteria you grade them by. | Adapted | Judge configs are versioned and pinned to each run, and a changed prompt counts as a different judge. Re-scoring stored evidence measures self-agreement. A drift check on every config change is not built ([open problem](open-problems.md#no-check-for-judge-drift-across-configurations)). |
| [Dev et al., Judge Reliability Harness](https://arxiv.org/abs/2603.05399) (2026) | No judge is uniformly reliable across benchmarks. | Adopted | Reliability is measured per dimension, scale, served model and judge config, and never pooled. The claim, sometimes attributed to this paper, that ordinal scoring is more fragile than binary is not in its abstract. |
| [Norman et al., "Reliability without Validity"](https://arxiv.org/abs/2606.19544) (2026); [Cohen, weighted κ](https://doi.org/10.1037/h0026256) (1968) | Exact-match agreement overstates reliability, so use a chance-corrected statistic. High test–retest reliability can coexist with severe bias. | Adopted | Both judge measurements use κ, quadratic-weighted on 1–5. The self-agreement tier is named as precision, never accuracy ([evidence tiers](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers)). |
| [Angelopoulos et al., prediction-powered inference](https://arxiv.org/abs/2301.09633) (2023) | Combine a small human-labelled set with many model predictions to get valid intervals. | Not built | Human labels only set the `calibrated` tier ([open problem](open-problems.md#human-labels-and-judge-scores-are-not-combined)). |
| [Ding, AdaRubric](https://arxiv.org/abs/2603.21362) (2026, single-author preprint) | Task-adapted rubrics track humans better than a static rubric (reported r≈0.79 against ≈0.46). | Adapted | Judged dimensions belong to a [template](concepts.md#template), not to a global fixed set. |
| [Seshadri et al., "Lost in Simulation"](https://arxiv.org/abs/2601.17087) (2026) | Swapping the simulated user's model moves agent success by up to 9 points, and simulators are miscalibrated against humans. | Adopted | The [simulator](concepts.md#simulator)'s model and settings are rig, so a run under a different simulator never pools with one under the first. |
| [Zhang et al., Who&When](https://arxiv.org/abs/2505.00212) (ICML 2025) | The best automated failure attribution names the responsible agent 53.5% of the time and the decisive step 14.2%. | Adopted | No judge is asked to assign blame among the participants of a transcript. |
| [Langfuse score configs](https://langfuse.com/docs/evaluation/scores/data-model) | Register each score's name, type and range once. | Adopted, extended | Every measure declares unit, direction and family in a `MetricDescriptor`. |

## Agent environments and worlds

The [world model](world-model.md#prior-art) gives the design lineage. This table records the verdicts.

| Source | What it recommends | Here | Why |
|---|---|---|---|
| [Gymnasium `Env`](https://gymnasium.farama.org/api/env/); [OpenEnv core API](https://huggingface.co/docs/openenv/main/en/reference/core.md) | Type the action and observation spaces. `reset` takes a seed and an untyped options dict (Gymnasium) or untyped keyword arguments (OpenEnv). | Not adopted | Neither declares what a harness may set before the agent starts. The world registry declares exactly that. |
| [Model Context Protocol](https://modelcontextprotocol.io/specification) | A server declares tools (`inputSchema`, `outputSchema`) and resources for a client that does not know it in advance. | Adapted (precedent) | It is the precedent for "the app declares, a generic engine consumes". It lacks "can a harness set this?" and "does the subject perceive this?", which the world contract adds. |
| [METR Task Standard](https://github.com/METR/task-standard) | Define a task as a class with imperative `install`, `start` and `score`. METR has since moved to Inspect. | Not adopted | Imperative setup cannot be introspected. |
| [τ-bench](https://arxiv.org/abs/2406.12045); [AppWorld](https://arxiv.org/abs/2407.18901) | Score the final environment state against the goal; AppWorld's state-based tests also catch collateral damage. | Adopted | Goal checks read the end state, the calls made and the world events that fired. |
| [Zhu et al., Agentic Benchmark Checklist](https://arxiv.org/abs/2507.02825) (2025) | Flaws in task setup and reward design misestimate performance by up to 100% in relative terms. On τ-bench's airline split, an agent that does nothing scores 38%. | Adopted as justification | Goal-check controls refuse a check that cannot tell doing nothing from doing the task. Preconditions exclude a trial whose world did not start where the case says. |
| [Meta ARE / Gaia2](https://arxiv.org/abs/2509.17158) (2025) | Seed initial state, and let events arrive independently of the agent. | Adapted | Triggered world dimensions fire during a cell, and goal checks read them with `fired()`. |
| [Java Technology Compatibility Kit](https://jcp.org/en/resources/guide-tck) | The specification owner ships tests that every implementation runs. | Adopted | The conformance kits a host runs against its own code: the [store kit](adopting-a-host.md#the-store-and-the-conformance-kit-that-proves-it) and the world kit, `check_world_conformance` ([world model](world-model.md#the-rules-the-contract-enforces)). |
| [PDDL](https://planning.wiki/ref/pddl) | The domain declares predicates, problems use them, and the planner's language never changes. | Adopted | The goal language has a closed grammar over an open host vocabulary. |
| [OpenTelemetry `gen_ai.evaluation.result`](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md) | Emit eval scores as events parented to the evaluated span. | Not adopted | The convention is still at Development status. The engine only reads spans and latency through a host's `TraceSink`. |

## Reporting and communicating uncertainty

| Source | What it recommends | Here | Why |
|---|---|---|---|
| [Mitchell et al., Model Cards](https://arxiv.org/abs/1810.03993) (2019) | Use fixed sections that end in caveats, and report results disaggregated by subgroup. | Adapted | A caveat attaches to the finding it qualifies, not to a closing section. [Strata](reading-reports.md#results-by-kind-of-case-strata) give the disaggregation. |
| [HELM, Liang et al.](https://arxiv.org/abs/2211.09110) (2022) | Never report a single number: measure many metrics, expose trade-offs, and name what is not covered. | Adopted | Reports give a frontier and bars, never an unqualified winner, and disclosures name what was not measured. |
| [van der Bles et al.](https://doi.org/10.1098/rsos.181870) (2019) | Keep direct uncertainty (the range) apart from indirect uncertainty (the quality of the evidence). | Adopted | Intervals and [evidence tiers](concepts.md#evidence-tier) render as separate axes. |
| [Padilla et al.](https://doi.org/10.3389/fpsyg.2020.579267) (2021) | Show verbal confidence alongside quantified uncertainty, never instead of it. | Adapted | Tier words sit beside n and intervals. Confidence is a closed tier, so no field can carry a probability a model made up; nothing calibrates one. |
| [Frans et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10623599/) (2023) | Quantile dot plots were read more accurately than error bars, with no clear effect on decision quality. | Adapted | Distribution charts draw individual values under the interval. Quantile dot plots are not built. |
| [NeurIPS paper checklist](https://neurips.cc/Conferences/2022/PaperInformation/PaperChecklist), statistical significance item | Put error bars on main claims, say what variability they capture, and draw no impossible bounds. | Adopted | Each interval carries a `variability` phrase, and intervals are clipped to the measure's scale. |
| [GRADE](https://book.gradepro.org/guideline/overview-of-the-grade-approach) | Rate certainty high, moderate, low or very low, downgrading for bias, imprecision and inconsistency. | Adapted | Evidence tiers are GRADE-like, but they grade the judge (`calibrated`, `separation`, `incidental`, `undetermined`), and code assigns them. |
| [Vega-Lite](https://vega.github.io/vega-lite/) | A declarative grammar in which a chart is data. | Adopted | [Chart blocks carry an intent](reading-reports.md#chart-blocks-carry-an-intent-not-a-librarys-spec), and the Vega-Lite adapter compiles it. A spec is data, so it can be validated rather than executed. Rejected: hand-written SVG (about 200 lines per chart shape) and plotnine (static raster that cannot read the palette's tokens). |

## Where this engine departs from common practice

Each departure states the principle behind it. The full set is in [principles](principles.md).

- **Comparability is decided by a key, not by memory.** Variant and context keys are content digests, and a report
  names which component differs, where common tools compare "experiments on the same dataset" by convention.
- **Declared and witnessed evidence never pool.** Experimental and observational data answer different questions;
  observational data cannot license a causal claim.
- **The judge is part of the instrument.** Its config is versioned and pinned to each run, so a judge that changes
  is a different instrument, never a silent edit under results already scored.
- **Shipping cost and measuring cost are separate.** The frontier ranks on production cost; judge, simulator and
  analysis spend are the bill for measuring.
- **A rig failure excludes, and a candidate failure fails.** Common tooling rarely separates the two causes. Here a
  failed precondition or a broken rig costs one cell and is counted, never charged to the candidate.
- **The denominator is part of the number.** A run that measured fewer trials than promised says so on every
  surface.
- **Evidence tiers instead of "calibrate first".** The common advice is to validate a judge against human labels
  before trusting it. The engine never gates on calibration. It labels every judged reading with the tier its
  evidence reached, computed by code, so judged scores stay usable while their standing stays visible.
- **Code renders every number in a written analysis, and an eval grades its prose.** The surveyed tools show dashboards,
  not memos. A deterministic gate over free prose never converged, so the writer's prose is measured like any other
  candidate ([reading reports](reading-reports.md#having-a-model-write-the-analysis-over-frozen-evidence)).
- **"Best" is never unqualified.** Reports give a quality–cost–latency frontier plus bars (satisficing thresholds),
  not one headline score.

## Vocabulary map

| Industry term (where from) | This engine | Note |
|---|---|---|
| System under test (ISTQB); agent (Anthropic) | [subject](concepts.md#subject); candidate | The candidate is the subject built for one cell. |
| SUT adapter; solver (Inspect); provider (promptfoo) | [kind](concepts.md#kind-candidate-kind) | |
| Task, test case (Anthropic); sample (Inspect) | [case](concepts.md#case-test-case) | Avoid "task": Inspect uses it for the whole eval, and Braintrust uses it for the system under test. |
| Scenario (HELM); task definition | [template](concepts.md#template) | |
| Case dimension (Husain & Shankar) | [variation axis](concepts.md#variation-axis) | |
| Suite, dataset | [battery](concepts.md#battery); a template's frozen cases | |
| Trial (Anthropic); epoch (Inspect); repetition (LangSmith) | [result (observation)](concepts.md#result-observation); a cell while running | |
| Trials per case; epochs; replicates (DOE) | [k](concepts.md#k-repeats) (`k_runs`) | `k_runs` counts trials, not runs. |
| Eval run (Inspect); experiment (Braintrust, LangSmith); job (Harbor) | [run](concepts.md#run) | In DOE and W&B a "run" is one trial. |
| Factor, with levels (Kohavi; NIST) | [lever](concepts.md#lever) | |
| Nuisance or held-constant factor; evaluation harness | [apparatus, rig](concepts.md#apparatus-rig) | |
| Variant, treatment (Kohavi); arm (clinical trials) | [variant](concepts.md#variant-and-variant-key), [arm](concepts.md#arm) | |
| Experiment, study; preregistered design | [campaign](concepts.md#campaign) | Its control is a variant, contemporaneous, never a temporal "baseline". |
| Grader, scorer, evaluator | [scorer](concepts.md#scorer); [goal-state check](concepts.md#goal-state-check); [judge](concepts.md#judge) | |
| Rubric criterion | [judged dimension](concepts.md#judged-dimension-rubric-dimension) | |
| `NOANSWER` (Inspect); "Unknown" (Anthropic) | "can't tell" (`cannot_tell`) | It excludes the trial on that dimension, and it is counted. |
| Certainty of evidence (GRADE) | [evidence tier](concepts.md#evidence-tier) | It grades the judge. |
| pass^k (τ-bench; Inspect `pass_k`) | the stored field **`pass_at_k`** | The field holds **pass^k** (all of k), the opposite of the field's pass@k name (at least one of k, Inspect `pass_at`). Anyone joining exports to Inspect or τ-bench numbers must read it as pass^k. |
