# Owner Rulings

<!-- The owner's own words, each time they decided something about this product, newest first.
     Each entry says where the decision landed; the governing text lives there. A ruling that
     reverses an earlier one is a new entry naming the one it replaces. Plain prose; nothing
     parses this file. -->

**2026-11-22.** "yes push, but we cannot merge PR's." Sessions push work branches and
never merge a PR; the maintainer merges. Landed in `project-preferences.md` § Workflow (PR merge).

**2026-11-22.** "For test execution, everything that can run on testament should run on testament."
Landed in `project-preferences.md` § Workflow (Inner-loop verification, Delegate verification).

**2026-11-22.** "For delegation, yes light testing for delegates, and if several delegates are
running a dedicated integrator subagent should be used to maintain and merge into an integration
branch. Main model can use judgment for when that becomes more efficient than doing integration
itself." Landed in `project-preferences.md` § Workflow (Delegation, Delegate verification).

**2026-11-22.** Write the seven missing strategy-class artifacts properly (not stubs), with
subagents "when efficient and when sequencing makes sense", then push.

**2026-11-22.** On the prawduct health check's repair plan: "yes approved on all." That covered
narrowing `.gitignore` so prawduct's governance state is tracked, scaffolding the core state
(`init-product`), moving the governance anchor into `AGENTS.md`, adding the unloaded-plugin
tripwire hook, filling `project-state.yaml` and `project-preferences.md` from the repo, declaring
testament as the test-evidence command, and configuring the review-notes transport. Landed in
`.gitignore`, `.prawduct/`, `AGENTS.md`, `CLAUDE.md` and `.claude/settings.json`.

**2026-11-22.** On the build plans whose branches were gone: "I believe all shipped, and it's
harmless to be wrong." Every live plan was archived as completed under
`.prawduct/artifacts/archive/`, with the release that carried it where a tag holds it.

**2026-11-22.** On recording the API versioning decision: "Yes to the API versioning policy."
Landed in `project-state.yaml` `design_decisions.api_versioning_approach`: the lockstep family
version, minor-line bounds, and API additions in a minor bump, as `docs/releasing.md` and the two
enforcement tests already hold. No deprecation rule beyond the alpha notice was given; that is
recorded as such, not invented.

**2026-11-22.** "Please also make sure change logs are the modern fragment style." The change
record is `.prawduct/changes/`, one fragment per scope; there is no `.prawduct/change-log.md`.
