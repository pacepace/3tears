# @3tears/web-pwa

`@3tears/web-pwa` (JavaScript, `web/packages/pwa`) -- installable web apps: install metadata, a
shell-only service worker with an update prompt, Web Push, notification routing, Media Session,
wake lock, display mode and safe areas.

## Problem

Every family web app people keep open gets asked to be "an app on my phone", and each one
rediscovered the same traps: iOS reads the home-screen name and icon from the page, not the
manifest; a manifest `id` that changes orphans every install; a service worker that caches the
wrong thing serves a stale app, a stale config or someone else's sign-in redirect; a deploy that
reloads a page throws away what the person was doing; a push the worker does not show gets push
revoked. metallm, scriob and the station each solved part of this on their own.

## What it does

Plain ES modules, each feature opt-in, configured by the app: `appManifest`, `installId` and
`headTags`; `shell-worker.js` for apps with no build step and `appShell` for Workbox apps;
`registerWorker`, `watchVersion`, `updatePrompt` and `forceReload` for updates; the page and worker
halves of Web Push and notification taps; `MediaSessionControl`; `ScreenWakeLock`; display mode;
`safe-area.css`. React hooks (`/react`) and a Vite helper (`/vite`) are separate entries. The package
README has the full table and how metallm, scriob and the station use it.

## Design philosophy

**Cache the shell, never the rest.** Audio, API calls and sign-in are never cached; the shell is
network-first where file names are not hashed, so a page and its scripts never come from two
versions. **The person decides when to update.** **Extracted, not designed ahead:** every module
comes from code already working in one of the apps.

## When to adopt

Any family web app meant to be installed, or that wants a lock-screen player, push or a safe update
prompt. Not for offline-first apps: there is no offline data layer here.

## Composes with

The Python side: [`iam`](iam.md) for sign-in (send its redirect back to the app's `start_url`). The
server half of push is planned as `3tears-webpush`.

## Install

```bash
npm install @3tears/web-pwa
```
