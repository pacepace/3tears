# Blind handoff test, 2026-10-25

A fresh agent got only this bundle (with the Kairo reference, UPSTREAM.md and CONVERGENCE.md removed, and Kairo-specific examples scrubbed) plus the raw Kairo round 2 writer-input package. It had to produce the memo from scratch.

## Result

It reached the same decision as `references/r5-kairo-round2`: shape D, decide mode, "Keep September", and the same flag: four late criteria plus the uncalibrated judge. (At the time the spec capped confidence at low for these; it now states a conditional confidence instead: "high, if the caveats below aren't material", with the late margin checked immaterial by code, GENERATOR.md §2 item 5.) It found the off-lane weakness on its own and passed `check.py` at 1200 and 390 px in both themes. It reported 17 gaps.

## Gaps and what changed

| Gap it reported | Resolution |
|---|---|
| No renderer; the figure-key grammar existed only as one example | GENERATOR.md §6.2 now defines the grammar and how to hand-build pages until CONVERGENCE P3–P4 |
| A green "Within, worse" pill on a regression | §3: status color follows the direction of the change, not the margin |
| 1017's floor speaks for a different measure than its reading | §3 rule (gate on readings, show the floor separately) · UPSTREAM U12 |
| The contrast family says 16 comparisons; the input carries 8 | UPSTREAM U3 |
| No definition of "pass" | §3 rule (say so plainly) · UPSTREAM U13 |
| No reader names for strata and measures | UPSTREAM U10 |
| Cost measure n differs from trials; no volume | UPSTREAM U2 and U7 |
| No reference for shape D decide mode | `references/r5-kairo-round2` (held out of the test) |
| Flip requirement doesn't fit a dominated decision | §2 item 10 and §8 item 5: say it can't flip, then show where each reading flips |
| Plan schema couldn't express the control's color | `color_slot` minimum is −1 |
| `score_type: mixed` missing from §1 | Added |
| Example selection alphabetical, so off-lane never appeared; escaped excerpts | Adapter orders strata by loss and unescapes excerpts |
| Short runs not mapped to arms | Adapter maps them; UPSTREAM U14 |
| Contradicted prior insight had no rule | §2 rule (Watch slot) · UPSTREAM U15 |
| Interval level only in prose | Adapter adds `contrast_families[].level`; UPSTREAM U2 |
| Phone-width label truncation and crowded ticks; `check.py` let touching labels pass | Kit sizes label columns to the labels, caps tick counts to what fits, and offers short labels; `check.py` now counts touching labels as overlaps |
| `.flag` and example cards were page-local; adapter docstring overstated the synthetic references | Moved into `kit.css`; docs corrected |
