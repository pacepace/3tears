# 3tears-pwa

`threetears.pwa` -- what a web app needs to install as a home-screen or desktop
app and to behave well installed: a stable per-app identity, the manifest, the
icon set, the head tags, safe-area CSS and small browser helpers.

## Problem

Every web app in the family that people keep open -- a radio player, a listen
page, a chat client -- gets asked to be "an app on my phone", and each one
rediscovers the same traps. **iOS reads the page, not the manifest**: the
home-screen name comes from `apple-mobile-web-app-title` and the icon from
`apple-touch-icon`, so an app whose manifest is perfect still installs as the
page title with a screenshot for an icon. **The manifest `id` is an identity,
compared and never re-read**: derive it from a display name or a URL and the
first rename orphans every install, and two apps on one host that share an id
install as one. **Edge-to-edge needs every edge padded**: `viewport-fit=cover`
draws under the notch and home indicator, and a fixed bar that ignores the
safe-area insets is clipped there. **Icons have per-platform rules**: iOS turns
transparency black, maskable icons are cropped to a circle of 80 % of their
width. And the folklore says a service worker is required, which it has not been
since Chrome 112.

## What it does

- **Identity** (`AppIdentity`, `install_id`) -- a stable `id` from a slug,
  `start_url` checked inside `scope`, names and colours validated. Several apps
  per host, each its own install.
- **Icons** (`render_icons`; `threetears.pwa.svg.render_svg_icons` with the
  `svg` extra) -- one square PNG or SVG becomes the 192/512, maskable,
  apple-touch, favicon and optional monochrome files, deterministically, with
  content-hash versions for immutable caching.
- **Manifest and caching** (`build_manifest`, `manifest_body`,
  `MANIFEST_MEDIA_TYPE`, `icon_cache_control`, `not_modified`).
- **Head** (`head_tags`, `identity_head_tags`, `insert_head_tags`)
  -- the manifest link, `theme-color` per colour scheme, and the iOS tags.
- **Browser files** (`pwa.js`, `safe-area.css`, `export_static`) -- display
  mode, Media Session for the lock screen with no seeking on a live stream,
  Screen Wake Lock; safe-area custom properties and utilities.

## Design philosophy

**Framework-free.** It produces bytes and strings; the app routes them. No web
framework is imported, so any host can adopt it.

**No service worker.** None is needed to install on current platforms, and a
fetch handler in front of a live page can bypass a sign-in gate and freeze
runtime configuration. An app that wants push or offline adds its own. The
evidence is in the package README, "Service worker".

**Current platforms only.** Safari 26 and later, current Chrome, Edge and
Firefox; no fallbacks for older ones.

## When to adopt

Adopt it for any family web app meant to be installed, and especially for a host
that serves several apps (one per station or persona) from one page. Do **not**
adopt it for offline-first apps that need a precache and an update flow: this
package deliberately has neither.

## Composes with

Nothing in the family is required. [`iam`](iam.md) is the usual sign-in; send
its post-sign-in redirect back to the app's `start_url` so the installed app
keeps its identity.

## Install

```bash
pip install 3tears-pwa
pip install '3tears-pwa[svg]'   # SVG icon sources
```
