---
artifact: build-plan
version: 1
scope: pwa
branch: feat/pwa
partition: "serial -- one builder; one new package plus its wiring, then its first consumer (ranch.station) in that repo"
last_validated: 2026-11-16
---

# Build plan: 3tears-pwa, installable web apps for the family

## Goals

**Response taken:** proceed, citing it. The owner's brief (2026-11-16) says what this is for: a
reusable family package that makes web apps installable and well-behaved installed, then the
station player as its first consumer. Success: the package (identity, manifest, icons, head tags,
safe-area CSS, small JS helpers), tests, docs and a changelog entry; the station installs per
station; PRs only, no release.

**Product:** any family web app installs as its own home-screen or desktop app, with a stable
identity per app (several per host), the icon set each platform reads, and the head tags iOS reads
instead of the manifest.

**Architecture that must hold:**
- Framework-free: bytes and strings out, the app routes them. No web framework, no family
  dependency. Pillow is the floor; resvg-py rides an `svg` extra.
- No service worker (family norm, discodon's PWA requirements section 2): installing needs none on
  current platforms. Evidence in the package README.
- Current platforms only (Safari 26+, current Chrome, Edge, Firefox).

**Constraints:** lockstep version 0.95.0 with intra-family bounds (none needed: no family
dependency); no agent attribution; namespace dir `threetears/` carries no `__init__.py`.

**Prior art reused (discodon `.prawduct/artifacts/standalone-pwa-requirements.md`, PRs #2136,
#2138, issues #2151, #2191; metallm e5354668 and 84660763):** id from a stable slug, never a
display name; iOS name and icon from head tags, written per request; manifest and icons served
without credentials; sign-in returns to the identity URL; `calc`/`max` with `env(safe-area-inset-*)`
on every edge region including fixed ones; `100dvh`; no service worker; Media Session with
pause as the listener's own.

**Decisions made mid-build:**
- [DECISION: browser tests for `pwa.js` run in headless Chromium in `packages/pwa`, so CI installs
  Chromium on the rest shard too and `test_browser_marker` allows `packages/pwa` | the JS helpers
  are the package's own code and need a browser; the marker rule said to move the install with
  any browser test leaving `packages/evals`]
- [DECISION: Chromium installability (`Page.getInstallabilityErrors`) is checked in the
  consumer's e2e tool (ranch.station `tools/e2e_pwa/pwa_check.py`), not in this repo's suite | it
  needs full Chromium with a persistent profile; CI and the pool install the headless shell only,
  which reports no installability errors at all]
- [DECISION: the public surface is what the first consumer calls; `IconFile`, `IconLink`,
  `StatusBarStyle`, `Display`, `VIEWPORT`, `render_svg` stay importable from their modules
  but are not exported, and the browser files are reached through `export_static` | the
  public-API-has-a-caller gate]
- [DECISION: monochrome icons are opt-in | a full-bleed square source makes a solid silhouette]

## Chunks

1. Package: identity, icons (PNG and SVG), manifest and cache rules, head tags, browser files,
   CLI; unit and browser tests; README with the service-worker evidence; adoption page, README
   and changelog rows; workspace, typecheck and CI wiring; dependency floor; public-API ledger.

Critic mode: final
