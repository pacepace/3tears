# Adoption Docs -- Maintenance Prompt

This is the saved prompt that generates and maintains `docs/adoption/`. Run
it whenever a package is added, removed, or renamed, whenever a package's
purpose or design changes meaningfully, or periodically to catch drift.

Paste everything below the line into an AI assistant with read access to this
repository.

---

You are updating `docs/adoption/` in the 3tears monorepo. This directory is
the AI-adoption reference: a set of documents that let another engineer or AI
system decide, quickly and correctly, whether and how to adopt each 3tears
package. Follow this procedure exactly. Do not skip the discovery step and do
not hand-write the package list from memory -- it drifts.

## 1. Discover the current package set

Read `[tool.uv.workspace]` in the root `pyproject.toml`. As of this writing
it is:

```toml
[tool.uv.workspace]
members = ["packages/*", "packages/agent/*"]
exclude = ["packages/agent"]
```

Treat that block as the source of truth, not the reproduction above -- it may
have changed. Expand the globs against the actual filesystem (every directory
matching a member glob that contains its own `pyproject.toml`) to get the
authoritative package list. This is a two-glob-plus-exclude pattern: replicate
it exactly, don't approximate with a single `packages/**` walk, or you'll
either miss the `agent/*` family or wrongly include `packages/agent` itself.

For each discovered package, note:
- its directory path (e.g. `packages/core`, `packages/agent/tools`)
- its PyPI name and one-line `description` from `pyproject.toml`
- its import root (the `threetears.*` path)

## 2. Diff against existing docs

Every package gets one doc: `docs/adoption/<slug>.md`, where `<slug>` is the
package's path relative to `packages/`, with `/` replaced by `-`
(`packages/core` -> `core.md`, `packages/agent/tools` -> `agent-tools.md`).

- **Package exists, no doc** -- new package. Go to step 3.
- **Doc exists, package directory gone** -- package was removed. Delete the
  doc and remove its entry from `README.md`'s module index. Do not leave a
  stale doc "just in case."
- **Both exist** -- check whether the doc is still accurate (step 4).
- **Package renamed** (import root or PyPI name changed, directory
  effectively the same package) -- rename the doc file, update its content,
  update every cross-reference to it in other docs.

## 3. Write or update each module doc

Gather source material, in this priority order:
1. The package's own `README.md` -- especially any section with a header
   like "Model," "Design," "Why," or the prose paragraph right after the
   title. That is almost always where the real rationale lives, not the
   install/API-reference sections.
2. The package's `pyproject.toml` `description`.
3. Root `docs/*.md` files that discuss the package's design (e.g.
   `docs/integration-guide.md`, `docs/channels-cross-pod-design.md`,
   `docs/separate-concerns-decisions.md`, `docs/partition-column-pattern.md`
   -- check what exists; more accumulate over time).
4. Recent git history / commit messages for the package, if the README is
   thin and a design decision is only recorded in a commit or PR.

**Do not invent philosophy.** If a package has no rationale prose anywhere
and is genuinely just an API-reference README, write "Design rationale is not
separately documented; the primitives below are the contract" rather than
manufacturing a justification. A false "why" is worse than an honest gap --
flag it back to the human instead of papering over it.

Use this exact template for every module doc:

```markdown
# 3tears-<pypi-suffix>

`threetears.<import.path>` -- <one-line description>

## Problem

<2-4 sentences. What breaks, or what gets reimplemented badly, without this
package. Concrete, not abstract -- name the failure mode.>

## What it does

<3-6 bullets. The core primitives/capabilities, not a full API reference.>

## Design philosophy

<1-3 short paragraphs or a bulleted list of the stated design principles,
drawn from real source material. Cite trade-offs explicitly if the package
made one (e.g. "deliberately ships no X" or "chose Y over Z because...").>

## When to adopt

<1-3 sentences or bullets. Who needs this, and what it requires (e.g.
pgvector, a NATS deployment, another 3tears package).>

## Composes with

<Bulleted links to sibling docs/adoption/*.md files: what this depends on,
and (if notable) what depends on this.>

**This section is the single most error-prone part of the whole doc set --
verify it against the dependency graph, never against narrative
plausibility.** An adversarial review of the first generation of these docs
found real dependency omissions in 18 of 23 flawed docs, plus several
composed relationships that sounded architecturally sensible but had no
code behind them. For every package, before writing this section:
1. Read its `pyproject.toml` `dependencies` array. Every `3tears-*` entry
   there is a real composition -- include it, or state explicitly why you're
   omitting it (e.g. it's foundational and already implied).
2. Grep its `src/` for `from threetears.` / `import threetears.` to catch
   dependencies used but not obvious from the package name.
3. Do not claim a composition in the reverse direction (`B` composes with
   `A`) unless `A`'s own `pyproject.toml` lists it, or you've confirmed a
   real import. "These two packages seem related" is not evidence.
4. If two packages are commonly used together but have no code-level
   relationship (e.g. `mcp` and `registry`), say so explicitly rather than
   implying they're pre-wired -- a downstream AI will otherwise assume
   integration code that doesn't exist.

## Install

\`\`\`bash
pip install 3tears-<pypi-suffix>
\`\`\`
```

Target length: 60-120 lines per doc. This is a decision aid, not a manual --
if you're explaining a full API, you've gone too deep; link to the package's
own README instead.

## 4. Update the platform overview

Whenever the package set changes, update `docs/adoption/README.md`:
- the module index tables (add/remove/rename rows, keep them grouped by
  family)
- the family table, if a new family emerges (unlikely, but don't force a new
  package into a bad-fit family just to avoid adding one)
- "How to adopt" entry points, if a new package changes the minimal-install
  story for a common use case

Do not touch the "What 3tears is," "Platform-wide design principles," or
"Core mental model" sections unless the underlying platform philosophy has
actually changed -- these are stable and should not churn on every package
addition.

## 5. Verify before finishing

- Every link in `README.md`'s module index resolves to a real file.
- Every "Composes with" link in every module doc resolves to a real file,
  and every claimed composition is backed per the checklist in step 3
  (pyproject dependency or a real import -- not narrative plausibility).
- Every package discovered in step 1 has exactly one doc, and every doc in
  `docs/adoption/` (other than `README.md` and `PROMPT.md`) corresponds to a
  real package.
- No doc invents a design rationale that isn't traceable to source material.
- **A package's own README can be stale.** If a doc names a specific class,
  method, or hook (e.g. a middleware hook name, a builder function), verify
  it still exists in current `src/` -- don't just echo the package README.
  A prior generation of these docs cited a deleted API and a wrong hook name
  this way; both were traceable to stale prose in the package's own README
  rather than the actual source.
- Style check (section below) passes on anything you wrote or touched.

## 6. Adversarial review (required, not optional)

Do not self-certify. The agent that wrote a doc believes its own claims;
that's exactly why the first generation of these docs had 53 factual issues
across 23 of 26 files despite passing its own author's read-through. Dispatch
two independent reviewers with **no memory of writing the docs** -- fresh
agents/sessions, not a second pass by the same context:

1. **Fact-checker.** For every doc touched in this run (new, updated, or
   renamed -- not the whole set if only a few changed), verify against
   ground truth: the `pip install` name and import path against
   `pyproject.toml`; every named class/method/hook against current `src/`;
   every "Composes with" entry against the dependency checklist in step 3.
   Report file:line for the claim and file:line for the ground truth on
   every mismatch.
2. **Cold-read adoption simulator.** Given only the touched docs (plus
   `README.md` for context), attempt to actually adopt the package(s) for a
   realistic scenario. Flag every point of confusion, undefined term,
   contradiction with another doc, or "Composes with" chain that dead-ends
   somewhere it shouldn't.

If this is a full regeneration (not an incremental update), run both
reviewers across the entire doc set, not a sample -- an incremental "spot
check a few files" pass is how systemic issues (e.g. the same category of
omission recurring in 18 of 23 docs) go undetected.

Fix everything both reviewers report before considering the run done. Do
not weaken a doc's claims to make a finding go away without checking which
side -- the doc or the review -- was actually right.

## Style

Match <https://pace.org/ead/whitepaper>: clear, concise, declarative. No
hedging ("might," "could potentially," "in some cases" -- say the thing or
say it's uncertain and why). No marketing language ("seamless," "powerful,"
"robust," "empowers"). Short sentences carry the weight; use them for
emphasis. Use tables and bullets to keep dense information scannable rather
than burying it in paragraphs. Em-dash only as `" -- "` (space-dash-dash-space,
ASCII), used sparingly. No emoji.

## Output

Write files under `docs/adoption/`. Do not create a PR unless asked. Report
back a short summary: packages added, removed, renamed, or updated; any doc
where you had to write "rationale not documented" so a human can go fill
the gap; and the adversarial review tally (issues found, issues fixed) so
the human can see the review actually ran rather than being skipped.
