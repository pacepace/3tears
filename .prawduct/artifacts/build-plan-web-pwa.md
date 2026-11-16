---
artifact: build-plan
version: 1
scope: web-pwa
branch: feat/web-pwa
partition: "serial -- one builder; the workspace, the package and the release wiring land together"
last_validated: 2026-11-16
---

# Build plan: web/ workspace and @3tears/pwa

## Goals

**Response taken:** proceed, citing it. The owner's agreement of 2026-11-09 (the session that
discussed the station PWA) and the rulings relayed on 2026-11-16 say what this is:

- 3tears gets a JavaScript workspace `web/` (not "frontend"), npm scope `@3tears/`; one job per
  package, named `@3tears/web-<name>` (owner, 2026-11-16); `@3tears/web-pwa` first. Agreed 2026-11-09
  as private on GitHub Packages; on 2026-11-16 the registry became one setting in release.yml,
  npmjs.com by default (3tears is public, MIT), the owner's choice pending.
- Versioned in lockstep with the Python family (one version, one tag, one release run), targeting
  0.95.0; the bump tooling and release workflow move and publish it; an enforcement test holds it.
- General for any 3tears app, validated against metallm (React, Vite, Workbox, push), scriob (React,
  hand-written manifest and worker, online only) and the station (plain modules, no build,
  per-station identities): framework-free core, React and Vite as optional entries, configuration
  over app details, each feature opt-in, an example app its tests exercise.
- Extracted from working code: metallm's worker order, update prompt (and its force-reload escape
  hatch), push subscribe and notification taps; scriob's pass-through worker and version check; the
  station's Media Session and wake lock; discodon's iOS install findings.
- Service-worker policy: cache the app shell only; never audio, API or sign-in; an update prompt,
  never a silent replace.
- Out of scope here: switching metallm and scriob (other agents), `3tears-webpush` (next).

**Decisions made mid-build:**
- [DECISION: `shell-worker.js` is a classic script configured by its URL and a JSON file | the
  station and scriob have no worker build; a module worker cannot be imported by a classic one]
- [DECISION: the shell is network-first in `shell-worker.js` | file names there are not hashed, so
  cache-first would serve a page and scripts from two versions; Workbox apps keep cache-first on
  hashed precache entries]
- [DECISION: `appShell` takes the app's Workbox modules as an argument | no Workbox dependency in
  the package, and the app's tests that replace Workbox with recorders keep working]
- [DECISION: `npm-lockstep-files` in `[tool.threetears-release]` | one bump moves Python and npm
  versions together; JSON rewrite keeps npm's formatting byte for byte]
- [DECISION: the web tests run in their own workflow (`web.yml`) | they need Node and Chromium, not
  the Python shards]
- [DECISION: npm-publish is gated by NPM_PUBLISH, its registry is NPM_REGISTRY (npmjs.com by
  default, trusted publishing with provenance) | the registry is the owner's call; GitHub Packages
  would refuse `@3tears/*` from `pacepace/3tears` (the GitHub account `3tears` is someone else's)]

Critic mode: final
