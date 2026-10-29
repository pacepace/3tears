# Reasoning note: turn router classifier (synthetic reference r3)

**The eval.** A classifier that labels each user turn by complexity (simple, medium, complex) and domain (science, conversational, news). Three models by three prompts (zero-shot, few-shot, rubric with reasoning): nine configurations. 48,210 human-labelled turns in a 3 × 3 target matrix. A turn is correct only if both labels match. Gates: balanced accuracy (cells weighted equally) at least 90%, every cell at least 80%, p95 at most 250 ms. Volume is 30 million turns a month.

**What I noticed first.**
- Nano-XS with few-shot cost about a twenty-first of anything that passes in this run, with 90.1% balanced and 92.0% traffic-weighted accuracy. It fails one cell: complex news at 76.7%.
- That ratio is this run's provider routing's, not the models' own. The snapshot names no provider for any configuration, so the routing is unknown, and every cost comparison on the page says so.
- Complex news is the weakest cell for every configuration.
- The best prompt depends on the model. The rubric prompt helps Atlas-M and Brisk-S, and costs Nano-XS six points against few-shot.

**The trap.** The overall number. Simple conversational turns are almost a third of the corpus, so a traffic-weighted average hides the cell that rules Nano-XS out. Lead with overall accuracy and the lowest-cost configuration looks like the answer. The campaign's gates already guard against this (balanced accuracy and a cell floor), but the analyst has to lead with the cell, not the average. A second trap: describing the prompt as a main effect ("few-shot helps", "the rubric helps"). Averaged over models, that would be true and wrong for Nano-XS.

**Story.** Decision. "Use Brisk-S with the few-shot prompt. Of the configurations that stay above 80% in every cell, it cost the least in this run." The cost claim says the routing behind that order is unknown. Labels are human-adjudicated, so there is no judge caveat. Confidence is high.

**Lead evidence.** The model × prompt lattice, with cost, latency and gate status per tile. It shows the pick, the interaction, and why the more accurate options are out (latency), in one grid. Then the target-matrix heatmaps for the pick, the lower-cost alternative and the most accurate, so the reader sees the cell that decides.

**What I looked for and found.**
- What kind of errors the pick makes. Almost every complexity error is off by one step (medium against complex). Domain errors are rare. The confusion matrix earns a support claim because it says what to fix.
- Whether the labels themselves are solid. Annotators agreed less on complexity (κ 0.78) than on domain (κ 0.91). The pick's main error type is exactly where humans disagree, so part of it is label ambiguity. The effect is similar for every configuration, so the ranking holds. Absolute cell accuracy may be understated, which goes in Method.

**What I left out, and why.**
- Macro F1. A third summary of the same errors, which adds nothing to the call. Full table only.
- The traffic-weighted view, except as an explore control. The campaign declared balanced accuracy, and showing both up top would invite choosing the weighting after seeing the results.

**What I'd watch.** The nearest line isn't the weakest cell. The pick's balanced accuracy is 90.7% against a 90% floor, 0.7 points. Complex news clears its floor by about 2.9. I would name both in Watch, the overall first. The reference page names only the cell.

**Next steps, each with what it settles.**
- Add complex-news examples to the few-shot set and re-run: settles whether the hardest cell can lift, and whether Nano-XS few-shot, which needs about 3.3 points there, becomes the pick at a fraction of this run's cost.
- Re-adjudicate a sample of medium/complex disagreements: settles how much of the main error type is label noise.

**What doesn't transfer.** The cell floor and balanced weighting were declared here. Without them, the honest report would have to choose a weighting and say why. The interaction was large enough to change Nano-XS's best prompt. Elsewhere, check before averaging.
