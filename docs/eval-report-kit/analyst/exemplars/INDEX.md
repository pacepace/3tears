# Exemplars

Each note shows how one report was reached: what the analyst noticed, what it led with, what it left out, and the trap it avoided. Read them for the thinking.

**Don't copy an exemplar's structure.** Your eval differs from all of these in ways that matter. Its claims, its blocks and their order come from your evidence, not from the note nearest in design. Two evals with the same design can need opposite stories. An eval that matches no note is normal; reason it through with the same tools. Each note ends with what didn't transfer, and that section matters as much as the rest.

| Note | The eval | The story | The trap |
|---|---|---|---|
| `kairo-round2.md` | A production persona model against two candidates on one judged aspect, with a margin set late and an uncalibrated judge | Decision: keep the control, because it is cheaper and nothing beats it | A candidate that is clearly worse reads as a pass when the margin verdict leads instead of the size |
| `r1-model-selection.md` | Four models across five tasks with gates, one of them a test category that occurs inside every task | Decision: one model everywhere, because it alone passes every gate | Routing each task to its cheapest passing model, when a cross-cutting gate blocks every cheap model from all traffic |
| `r2-config-sweep.md` | A full factorial sweep of 18 configurations against a pass-rate gate and a latency gate | Decision: the one configuration inside both gates, with its thin margin as the main risk | "The only one in the box" read as a clean win, when it clears the gate by a fraction of its interval and was picked as the best of many |
| `r3-classifier.md` | Nine model and prompt configurations of a two-label classifier over a target matrix of cells | Decision: the cheapest configuration that holds every cell | An overall average that hides the cell ruling out the cheap model, and a prompt effect that reverses for one model |
| `r4-tradeoff.md` | Production against three challengers on personality, with standing guardrails and no declared exchange rate | Tradeoff: priced, with a lean on a modeled hybrid | Pricing an option that breaks a guardrail as the middle path, and deciding a trade the campaign gave no rule for |

Add a note whenever a report teaches something new, especially for an eval that surprised everyone.
