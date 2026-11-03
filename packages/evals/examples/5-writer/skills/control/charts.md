# The block catalog

Each claim carries one block. Choose the one that answers the claim's question most directly. Then test it: if the values flipped, would the claim have to change? If not, the block isn't evidence for the claim. `block_catalog` returns the current kinds with their query fields. `block_render` lets you try one before committing to it.

**A table and prose are always available.** When no chart kind fits the data, a table states the figures and a sentence states the finding. That is never a failure, and it is often the clearest choice for few numbers.

| Kind | The question it answers | Needs from the data | Misleads when |
|---|---|---|---|
| `forest` | How much did each option change against the control, and does the interval clear zero or the margin? | A change and interval per option, one unit, one level; the margin if one was declared | Confirmatory readings and corrected contrasts share one chart; the margin band is read as the verdict and the size is left out |
| `bars` | How big is each option on one magnitude (cost, error rate, tasks passed)? | One value per row in one unit; a declared budget for a reference line | The axis doesn't start at zero; a rate near its ceiling is drawn as a pass rate instead of a failure rate from zero |
| `diverging_bars` | Where, across strata or levers, is a difference positive or negative, and how large? | A signed difference per row; intervals if any row is read as separated | Uncorrected stratum deltas are read as confirmed; rows not separated from zero are drawn as strongly as those that are |
| `dot_interval` | Where does each option's rate sit against a floor, with its uncertainty? | A rate and interval per row; the floor's criterion | The gap is a few points and bars from zero would hide it (use this, not bars), or it shows no interval and the reader assumes certainty |
| `lines` | How does a measure move along an ordered axis: turn depth, version, date, input length? | Ordered categories; values per series; bands on at most two series | The x axis has no order; too many series each carry a band |
| `scatter_frontier` | Which options trade one measure against another, and which are on the frontier? | Two measures per option; a floor or price line where declared | The reader reads slope as value across unlike scales; never swap it for a dual-axis chart |
| `gate_scatter` | Which options land inside two gates at once, and how far inside? | Two gated measures per option, both thresholds | An option just inside a corner is read as safely in; its intervals belong in the tooltip or Watch |
| `range_bars` | How do latency distributions compare against a budget? | p50 and p95 (or similar) per option | A mean alone is shown, or the tail was shortened by declining work |
| `gate_matrix` | Which option passes which criteria in which task, and what binds? | Per task and option: status and the binding reason; cost | A cross-cutting aspect is read per task; the reader assumes routing to the cheapest cell is allowed |
| `lattice` | How do two crossed factors combine? Is there an interaction? | A full grid of factor levels with one metric, cost and status per tile | The grid is incomplete; a main-effect summary would read better and the tiles add noise |
| `heatmap` | Which cells of a target matrix are weak? | A value and n per cell; a cell floor | Cells have very different n and shading treats them alike; one overall number would hide the weak cell (this is the remedy) |
| `diff_heatmap` | Where do two configurations differ, cell by cell? | Two configurations over the same cells | The difference is within noise everywhere but saturates anyway; check the neutral band |
| `confusion` | What kind of errors does a classifier make? | Row-normalized counts per label axis | The matrix is too large for a phone; labels are ambiguous to humans too (state agreement in Method) |
| `examples` | What do failures actually look like? | Trials with stratum, arm, judge reason and excerpt | Cards are chosen to dramatize; pick them from where the loss concentrates and say how |
| `table` | Exact figures a reader will look up or compare across many columns | Any rows | It replaces a chart whose shape is the finding |
| prose | One fact that needs no picture | A figure key | It carries several numbers a table would hold better |

## Sources the analyses feed

Some blocks draw what an analysis computed, not a stored figure. `block_catalog` lists each source's kinds and parameters; these are the ones to reach for:

| Source | Draws | Use it for |
|---|---|---|
| `compare` | Two named arms' difference on one measure, overall and per slice, with its interval (paired where a contrast pairs them, else unpaired) | A claim that one option beats another when there is no control, or the control isn't one of them |
| `settle` | `settle_cases` rows: distance to the line, cases needed one- and two-sided, how often today's size reads the wrong way | Watch and next steps: what a re-run must look like to decide a thin clearance |
| `levers` | What one setting buys between arms differing in it alone, and where that effect reverses | Factorial sweeps: the lever and its interaction, instead of reading a lattice by eye |
| `bar_reads` | Each arm's reading against a declared bar, with its interval and the bar's line | A bar the question treats as the condition |
| `prices` | The price frontier's steps, price per point with its interval | The trade between cost and quality |

A `lattice` over configs also draws from the arms' own settings when the snapshot holds no configs rows but two or more settings cross fully.

An interval marked **computed** was computed here from a rate's count (binomial), not stated by the pipeline. On a repeated design it is read on the distinct cases, at the widest spread the repeats allow, so it errs wide. Say "computed" when you cite one.

## Mechanics that keep a chart honest

- **One unit and one interval system per chart, with the level printed.** A 95% confirmatory reading and a family-corrected contrast belong in separate charts. Mixed together, the reader compares intervals that mean different things.
- **Log axes when a measure spans more than about ten times.** The caption says so. On a linear axis, cheap options pile up at zero and the comparison disappears.
- **A reference line sits only at a declared threshold.** A floor, budget or margin the campaign declared, read where the chart reads. A line at a convenient value looks like a rule nobody set.
- **Small multiples only where the kind draws panels.** One panel per arm, aspect or task, sharing axes so the eye compares like with like.
- **One identity per arm for the whole page.** The same color and tag everywhere, with the control marked. Color is never the only cue, and status always carries an icon and a word.
- **Descriptive versus inferential.** Breakdowns computed after the fact are uncorrected. Bold only the rows separated at the stated level, and call them leads.
- **Every chart's figures are in the appendix.** Hover adds detail but is never the only way to read a value.

## Fitting the block to the claim

- **A claim about a gap draws the gap.** "Costs more than", "beats", "falls behind": draw the difference between the arms with its interval (`compare`, a `forest` over `contrasts`, `diverging_bars`), paired where a contrast pairs them, not each arm's level. A percent or a multiple beside the gap comes from the same basis as the gap: a paired gap's share of the control, never one read off unpaired means while the block shows the paired gap. Where no key states it on that basis, state the gap and its interval, and log a `capability_gap`.
- **Effects with intervals are drawn with them.** When the figures a claim compares carry intervals (a lever's effect, a contrast), choose a block that draws them. A lattice or bars of point estimates hides which effects are separated.
- **The claim's figure is in its block.** The figure a claim turns on appears in its block's values table. If it doesn't, change the block or the claim.
- **A caption describes only what its block draws.** Bold rows, intervals, a band, an overall row: name one only if `block_render` shows it there. A caption promising a row or a mark the block lacks reads as a missing figure.
- **Every arm a claim compares gets its figure.** A claim about several arms states the figure for each one ("53% for one, 48% for the other"), not for one and lets the reader assume the rest.

## Choosing between candidates

When two kinds both fit, choose the one whose most visible feature is your claim. A forest makes the interval against zero the most visible thing. Bars make magnitude most visible. If your claim is "clearly worse", use the forest. If it is "costs a quarter more", use the bars. Two charts side by side are right when a call has two reasons. Then each is its own claim.

When the kind you want doesn't exist, use a table and call `capability_gap`. Never bend a chart to carry data it wasn't built for.
