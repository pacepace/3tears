# Owner Rulings

<!-- The owner's own words, each time they decided something about this product, newest first.
     Each entry says where the decision landed; the governing text lives there. A ruling that
     reverses an earlier one is a new entry naming the one it replaces. Plain prose; nothing
     parses this file. -->

**2026-11-26.** On the seven security-model rules with no enforcement gate: "yes to all", as proposed. Rules
1, 2, 5 and 7 (session token carries identity only; an admin grants only what they hold at every scope; tool-pod
identity from a verified token; secrets by reference) are ratified Critic-only. Rules 4 and 6 are ratified
in-transition: erasure never deletes audit rows (tracked in #1313), and the SSRF check on every request (tracked in
#1314, retitled to the gap the code names: browser-rendered fetches check only their first URL). Rule 3 (an unowned
namespace matches no caller) was proposed in-transition on the premise that the evaluator had no gate; it has one
(`test_a_null_owner_matches_nothing` in `packages/agent/acl/tests/unit/test_evaluation.py`, with
`tests/enforcement/test_namespace_owner_is_always_carried.py` over construction sites), so it was recorded
steady-state and no tracking issue was filed. Landed in `security-model.md` § Direction.

**2026-11-26.** On the governance advisories report: "Yes to all recommendations." That covered:
- Ratify the strategy artifacts' test-backed candidate norms in bulk, once duplicates are removed. Security-model
  rules with no gate are not ratified; each goes to the owner for its own ruling.
- Ratify the test-fakes rule, with the why "a fake that names no protocol can drift from production and pass
  against an API production lacks".
- Keep the force-push why as ratified on 2026-11-26.
- From `CLAUDE.md` § Conventions, ratify only "uv only, no poetry"; the rest stays descriptive, held by config.
- Do not ratify "consumers pin the whole family to one exact version" here: it binds downstream repos.
- Record and ratify two earlier rulings that lived only in a session's notes (below, at their own dates).
- Migrate the markdown backlog to GitHub Issues in `pacepace/3tears` with the approved batch (MOD-7R2K, MOD-4QW9),
  the cutover landing on #1305's branch.
Landed in the `## Direction` sections of the strategy artifacts, the `project-preferences.md` Enforcement table,
`.prawduct/artifacts/migration-scrub-decisions.md` and `project-state.yaml` `backlog_service_repo`.

**2026-10-28** (recorded 2026-11-26). Pricing is the host's: the engine reports tokens and models, the host injects
prices, and 3tears ships only a swappable dev default. The owner rejected a rate table inside the engine as too
monolithic: "for all we know the consumer has a special deal with OpenAI". Landed in `architecture.md` § Direction.

**2026-10-19** (recorded 2026-11-26). The maintainer sets versions at release; a work PR never pre-picks one. Given
after v0.71.0 shipped without the evals stack that had pre-claimed 0.71.0, forcing a re-version of three stacked PRs.
Landed in `operational-spec.md` § Direction.

**2026-11-22.** On the untracked notes in the `evals/main` worktree's `.prawduct/artifacts/`: "yes to
your recommendations, and move those discodon-*.md files to the discodon repo's docs for its own
agent to sort out", then "yes commit" for the four dated analyses. The design notes and analyses
are committed on `evals/main`; the second opinions and upgrade notes are deleted; the two discodon
notes went to Discodon, whose session deleted them and filed pacepace/discodon#3593. Raw eval data
stays local under `.prawduct/artifacts/blind-read-*/` (`.gitignore`).

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

**2026-11-24.** On the seven unintegrated evals delegate branches (all pushed; worktrees removed,
branches kept):
- Re-root (`evals/bp-x0`, `bp-x1`, `bp-x2`, #1199): "Still wanted, later." Keep the branches as the
  starting point: `bp-x1`'s rewrite script as is, `bp-x0`'s name map regenerated at the cut-over
  base (its 541-name map predates the removal of the unreached exports), `bp-x2` redone.
- S5a-3 (`evals/bp-s5a3`): "Drop S5a-3." Structure work is paused and main has moved past it. If
  S5b returns, it is rebuilt from current main; the pushed branch is reference only.
- B1 (`evals/bp-b1`): "Park it as source." Not merged. When a report needs the people-corrected
  difference, a fresh branch from evals/main takes only the estimator, without the 20/80 floors
  (the 2026-11-16 ruling).
- R1 (`evals/bp-r1`) and M0 (`evals/bp-m0`) were not asked; they follow from earlier rulings and
  are recorded in the evals program plan (`program-big-picture.md`, 2026-11-24).

**2026-11-24.** The extract fix's Critic verify pass may run with `--force`: the worktree had no
session marker, and the suite had run on that exact tree.

**2026-11-24.** On Discodon moving its 3tears pins to tag `discodon-search-extract.1` (1830f1c06: evals/main
cf1fb6585 plus the search extract fix, #1306) now, before a release carries #1306, the owner said: "Yes agreed". That
moves search, interfaces, observe and models from PyPI 0.77.0 to that one git commit and replaces `media-contracts`
with `interfaces`. In the same answer the owner agreed to push the 2026-11-24 rulings onto #1305's branch.
