# The two skills rung 5 compares

- [`control/`](control/) is a frozen copy of the analyst skill this package shipped on 2026-11-03, digest
  `4e14e276b568647e68450b3b025204213d2bddcc3a2ff6b838550224149788b8`. It is frozen because every writer recording is
  keyed by the skill's digest: the shipped skill changes often, and a recording made over it would go stale with each
  edit, costing a re-record each time. The rung measures the copy that was recorded, not the skill shipped today.
- [`lead-first.md`](lead-first.md) is the candidate's `SKILL.md`: the control's, with two sentences changed so the
  first claim of every report is its lead, the one finding the headline rests on, before any context, method or
  caveat. Every other file is the control's (`control.with_file("SKILL.md", ...)`).

This README sits outside `control/` because the loader reads every Markdown file under a skill's directory as part
of the skill: a line added there would change the copy's digest.
