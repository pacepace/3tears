# @3tears/web-pwa

Installable web apps for the 3tears family: what an app needs to install as a home-screen or
desktop app, stay current without losing what the person is doing, and behave well installed.

- **General, not shaped around one app.** Every feature is its own module and opt-in. An app
  imports only what it uses and configures it (identity, icons, theme, cache rules, push routing),
  and draws its own UI (the update banner, the push settings).
- **Framework-free core.** Plain ES modules with no dependencies, usable with a bundler or with
  no build step at all (served as files beside the app's own scripts). React hooks and a Vite
  helper are separate, optional entry points.
- **Current platforms only:** Safari 26+ (iOS, iPadOS, macOS), current Chrome, Edge, Firefox.
- **Versioned with 3tears.** Same version as the Python packages, released by the same tag.

Extracted from what already works in the family's apps: metallm's service worker, update prompt,
Web Push and notification routing; scriob's online-only worker and version check; the station
player's Media Session, wake lock and safe areas; discodon's iOS install findings.

## Install

```bash
npm install @3tears/web-pwa
```

Published to npm with the 3tears release (the registry is release.yml's `NPM_REGISTRY`, npmjs.com
by default). An app with no build step copies `node_modules/@3tears/web-pwa/src/` into its static
files (its image build does it) and imports `./pwa/index.js`.

## Features and where they live

| Feature | Entry | What the app configures |
|---|---|---|
| Install: manifest, head tags | `appManifest`, `installId`, `headTags`, `VIEWPORT` (`@3tears/web-pwa`) | identity (`id`, `start_url`, `scope`), names, theme, icons |
| Service worker, no build step | `shell-worker.js` | its shell list, never-list, version, update mode (a small script) |
| Service worker, Workbox (Vite) | `appShell` (`@3tears/web-pwa/workbox`), `vitePwa` (`@3tears/web-pwa/vite`) | precache, never-shell paths, network-first files |
| Update prompt | `registerWorker`, `watchVersion`, `updatePrompt`, `forceReload`; `useUpdatePrompt` (`/react`) | the banner it draws; how often to check |
| Web Push, page side | `pushSupported`, `subscribePush`, `unsubscribePush`, `keyBytes`, `workerRegistration` | its key endpoint and device API |
| Web Push, worker side | `showPushes`, `readPushData` (`@3tears/web-pwa/worker`) | how a payload becomes a notification |
| Notification taps | `openOnNotificationClick` (`/worker`); `useServiceWorkerMessage` (`/react`) | where a tap goes, what it tells an open window |
| Lock screen, media keys | `MediaSessionControl` | now playing, artwork, play/pause/stop handlers, live or not |
| Screen kept on | `ScreenWakeLock` | when it is wanted |
| Installed or in a tab | `displayMode`, `isStandalone`, `watchDisplayMode`, `markDisplayMode` | nothing |
| Safe areas | `safe-area.css` | its gutters |

## How the family's apps use it

Guidance for each app's integration (the station's is done; metallm's and scriob's are next).

| App | Imports | Configures |
|---|---|---|
| **metallm** (React, Vite, vite-plugin-pwa, Workbox, push) | `vitePwa` in `vite.config.ts`; in `src/sw.ts`: `appShell` with its own Workbox modules, `listenForSkipWaiting`, `showPushes`, `openOnNotificationClick`; in the page: `useUpdatePrompt(() => updateServiceWorker(true))` with `checkForUpdates` from `useRegisterSW`'s `onRegisteredSW`, `subscribePush`/`unsubscribePush` around its `/api/v1/push` calls, `isStandalone` | manifest (as today, no `id`), `runtimeFiles: ["config.js"]`, `neverShell: [...NEVER_SHELL, /^\/s3\//]`, `networkFirst: [{ path: "/config.js", cacheName: "metallm-runtime-config" }]`, its payload-to-notification and tap-to-conversation functions, its banner markup |
| **scriob** (React, Vite, hand-written manifest and worker, online only) | `shell-worker.js` copied to `public/` and registered with no config and `?update=immediate` (caches nothing, as its `sw.js` does); `watchVersion("/version.json", buildId, ..., { field: "buildId", win: window })` for its banner (at once, on focus and on return to view, as its own check does); `appManifest` in its manifest test | `public/manifest.webmanifest` as today; its banner |
| **station** (plain modules, no build, one app per station) | `src/` served as `/player/pwa/`; `registerWorker` with `shell-worker.js?config=/player/shell.js`; `MediaSessionControl`, `ScreenWakeLock`, `markDisplayMode`, `safe-area.css`; the server writes the manifest and head per station following `appManifest`/`headTags` | `/player/shell.js` (its files, never `/api/`, `/auth/`, audio), `Service-Worker-Allowed: /player/`, per-station `id` `/stations/<id>`, its notice for an update |

## Install conventions

- **`id` is the app's identity**, compared and never re-read: changing it orphans every install.
  For several apps on one host build it with `installId(prefix, slug)` from a stable slug, never
  a display name.
- **iOS reads the page, not the manifest**, for the home-screen name (`apple-mobile-web-app-title`)
  and icon (`apple-touch-icon`, 180 px, opaque: transparency turns black). `headTags` writes both.
  A host serving several apps from one page writes them per request.
- **Serve the manifest and icons without sign-in.** Browsers fetch them without cookies.
- **`scope` covers every page of a flow**, sign-in included: on iOS a navigation out of scope opens
  a browser sheet over the app. Send sign-in back to the URL that carries the app's identity.
- **Icons:** a 192 and a 512 px PNG (`appManifest` checks), a maskable 512 with the mark inside the
  centre circle of 80 % of its width, the 180 px opaque apple-touch icon, a favicon.

## Service worker policy

Agreed for the family (2026-11-09):

- **Cache the app shell only.** Never audio, API calls or sign-in.
- **Show an update prompt; never replace a running page behind the person's back.** A new worker
  waits until the person agrees (`SKIP_WAITING` from the page), then the page reloads into it.
- An app that needs no cache (online only) still gets the update protocol, and caches nothing.

### `shell-worker.js` (no build step)

A complete classic service worker, served as is from the package's files, configured by a small
script the app serves (its server or build writes it):

```js
import { registerWorker } from "./pwa/index.js";

const worker = await registerWorker("/app/pwa/shell-worker.js?config=/app/shell.js", {
  scope: "/app/",
  onUpdate: () => showBanner(),          // the app draws it
});
reloadButton.onclick = () => worker.applyUpdate();
```

```js
// /app/shell.js
self.SHELL_CONFIG = { version: "1.4.2", cache: "app", shell: ["/app/", "/app/app.js"], never: ["^/app/api/"], update: "prompt" };
```

- **The config is part of the worker.** It is loaded with `importScripts`, and browsers compare a
  worker's imported scripts as well as its own bytes when they look for an update, so a release
  that changes it (a new `version`, new files) is a new worker, which waits for the person; and a
  worker the browser stopped and started again has its config at once, even offline.
- Only the `shell` files are touched; anything else goes to the network untouched, and `never`
  is checked first.
- The shell is **network-first**: online, every load is the server's current file (a page and its
  scripts never come from two versions); the cache answers only when the network does not. Only a
  plain 200 is stored, never a redirect to sign-in. Shell paths match exactly: a deep link of a
  single-page app has no offline copy unless it is listed.
- Served from a path below the scope it serves (`/app/pwa/` for `/app/`), the worker needs the
  response header `Service-Worker-Allowed: /app/`.
- No config: caches nothing, passes everything through, deletes nothing. `?update=immediate`
  activates a new version at once, for an app with nothing to lose on a reload.

### Workbox (`@3tears/web-pwa/workbox`)

For a bundled app that precaches hashed build files (vite-plugin-pwa `injectManifest`), `appShell`
runs the policy in the generated worker's order: claim, precache, clean old precaches, the shell
for navigations except `neverShell` paths (those go to the network, nothing stored), and
network-first runtime files. metallm's worker also routed navigations network-first after the
shell route; that route only ever saw the never-shell navigations and stored them, so `appShell`
leaves it out: switching metallm drops the `metallm-navigation` cache on purpose. The app
passes its own Workbox modules in, so this package does not depend on Workbox and the app's test
doubles keep working.

## Updates without a worker cache

`watchVersion(url, current, onUpdate, { field, win })` looks at a small JSON file at once, every
five minutes and (with `win`) on focus and on return to view, with `no-store` and a `ts` query, and
calls `onUpdate` once when it differs. `forceReload()` is the escape hatch when a worker never
takes over: it unregisters every worker, deletes every cache and reloads.

## Web Push

Page: `subscribePush(registration, () => fetchKey())` asks permission first and fetches the key only
once granted, then reuses or creates the subscription; the app sends `subscription.toJSON()` to
its server. Worker: `showPushes(self, toNotification)` shows every push (a browser revokes push from
a worker that shows nothing); `openOnNotificationClick(self, route)` brings an open window forward
and posts it a message, else opens the app at a URL. On iPhone and iPad push works only in an
installed app. The server half (VAPID keys, subscriptions, sending) is the app's; a shared
Python package is planned as `3tears-webpush`.

## Media Session, wake lock, display mode, safe areas

- `new MediaSessionControl(navigator, { onPlay, onPause, onStop, live: true })`; `setNowPlaying({
  title, artist, album, artwork })`, `setPlaying(bool)`, `clear()`. Live streams get no seek or
  skip buttons. Handlers that throw or reject are logged, never unhandled.
- `new ScreenWakeLock(navigator, document)`; `set(true)`, and `sync()` on `visibilitychange`.
- `markDisplayMode(document, window)` keeps `<html data-display-mode>` current.
- `safe-area.css`: `--safe-top/right/bottom/left` and `.safe-area-pad`, `.safe-area-top`,
  `.safe-area-bottom`, `.safe-area-fill`, with `VIEWPORT` (`viewport-fit=cover`). Pad every edge
  region, fixed ones included; `100dvh`, not `100vh`.

## Tests

`npm test` (Node, every module with stand-ins) and `npm run test:browser` from `web/`: the
example app (`web/examples/basic`, served by `web/examples/server.js`) in headless Chromium with
the real shell worker: what it caches and never caches, offline, an update that waits for the
person, Media Session and display mode.
