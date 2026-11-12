# The analyst skill: notes for whoever edits it

This file is for the skill's maintainers. The loader (`analysis/analyst/skill.py`) reads it as part of the skill, so it counts in the skill's digest, but it never reaches an analyst's or an editor's system text.

- **What each role reads.** The analyst reads `SKILL.md`, then `principles.md`, `stories.md`, `charts.md`, `statistics.md` and `writing.md`, then `exemplars/INDEX.md` and every note it lists, except those a run withholds. A blind run withholds the note reasoned on its own campaign, or, for a variant, on the campaign it was built from; the note's index row goes with it. The report editor reads `editor.md` alone.
- **Adding a note.** Add one whenever a report teaches something new, especially for an eval that surprised everyone. Give it a row in `exemplars/INDEX.md`: the eval, the story, the trap, and its Structure. The loader refuses a note with no row and a row with no note.
- **The Structure column** records facts of the input the note's campaign had, such as an arm being the control (`control`), a margin declared (`margin`), a criterion applying within each task or cell (`slice_criteria`), a task inside every other (`cross_cutting_task`), or arms laid on a grid of settings (`factors`). It documents the notes for their readers. No code reads it, and no note is chosen by it.
- **Only Markdown is read.** Any other file here, such as a `.DS_Store`, is ignored.
