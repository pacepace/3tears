# 3tears-pwa

`threetears.pwa`: what a web app needs to install as a home-screen or desktop app, and to behave
well once installed. Framework-free: it builds bytes and strings, and the app serves them with
whatever web framework it has.

Current platforms only: Safari on iOS, iPadOS and macOS 26 and later, current Chrome, Edge and
Firefox. No legacy fallbacks.

```bash
pip install 3tears-pwa          # PNG icon source
pip install '3tears-pwa[svg]'   # SVG icon source (resvg, no system libraries)
```

## What it does

| Piece | API | Why it exists |
|---|---|---|
| Identity | `AppIdentity`, `install_id(prefix, slug)` | A stable manifest `id` per app, so several apps on one host (one per station, one per persona) each install as their own app, and renaming one never orphans its installs |
| Icons | `render_icons(png, background=...)`, `threetears.pwa.svg.render_svg_icons(svg, ...)` | One square source becomes 192/512 PNGs, maskable 192/512 (the mark inside the safe circle on an opaque background), the 180 px opaque `apple-touch-icon`, a 32 px favicon, optionally a monochrome silhouette and the SVG itself. Deterministic bytes, content-hash versions |
| Manifest | `build_manifest(identity, icons, icon_base_url=...)`, `manifest_body`, `MANIFEST_MEDIA_TYPE` | The manifest with `id`, `start_url`, `scope`, names, `display: standalone`, colours, categories, icons |
| Caching | `MANIFEST_CACHE_CONTROL`, `icon_cache_control(version, requested)`, `not_modified(if_none_match, etag)` | The manifest revalidates every time; an icon under `?v=<hash>` is immutable for a year, without it a day |
| Head | `head_tags(...)`, `identity_head_tags(identity, ...)`, `insert_head_tags(html, tags)`, `threetears.pwa.head.VIEWPORT` | The manifest link, `theme-color` per colour scheme, and the iOS tags iOS reads *instead of* the manifest |
| Browser | `pwa.js`, `safe-area.css` (`export_static(dir)`, `python -m threetears.pwa export-static <dir>`) | Display-mode detection, Media Session (lock screen, Control Center), Screen Wake Lock; safe-area insets for an edge-to-edge page |

## Serving an app

```python
from threetears.pwa import (
    AppIdentity, MANIFEST_CACHE_CONTROL, MANIFEST_MEDIA_TYPE, build_manifest,
    icon_cache_control, identity_head_tags, insert_head_tags, install_id, manifest_body,
    not_modified, render_icons,
)

identity = AppIdentity(
    id=install_id("/stations", "kairo"),        # never changes
    name="Kairo", start_url="/player/?station=kairo", scope="/",
    theme_color="#07060c", background_color="#07060c", categories=("music",),
)
icons = render_icons(open("mark.png", "rb").read(), background="#07060c")   # once, at start
base = "/api/kairo/app"
manifest = manifest_body(build_manifest(identity, icons, icon_base_url=base))
head = identity_head_tags(identity, manifest_url=f"{base}/manifest.webmanifest",
                          icon_set=icons, icon_base_url=base)
page = insert_head_tags(index_html, head)       # at <!-- pwa-head -->, else before </head>

# GET {base}/manifest.webmanifest -> manifest, MANIFEST_MEDIA_TYPE, Cache-Control MANIFEST_CACHE_CONTROL
# GET {base}/<icon name>?v=...    -> icons[name].body, .media_type, ETag icons[name].etag,
#                                    Cache-Control icon_cache_control(icons[name].version, request v),
#                                    304 when not_modified(If-None-Match, etag)
```

**Serve the manifest and the icons without sign-in.** Browsers fetch both without cookies: a
`<link rel="manifest">` request carries credentials only with `crossorigin="use-credentials"`, and
the installer fetches icons on its own. They hold nothing private.

**Write the head per request when a host has several apps.** iOS takes the home-screen name from
`apple-mobile-web-app-title` (else `<title>`) and the icon from `apple-touch-icon`, never from the
manifest, so a page shared by several apps must carry the right app's tags.

**Choose the scope to cover every page of a flow.** On iOS, a navigation out of scope opens in a
browser sheet over the app; a sign-in whose callback is out of scope finishes in that sheet. The
site root is usually right when sign-in lives under `/auth/`.

**Send sign-in back to the URL that carries the identity** (`start_url`, with its query): an app
that lands on a bare page after signing in has lost which app it is.

## Safe areas

With `threetears.pwa.head.VIEWPORT` (`viewport-fit=cover`) and the status bar style `black-translucent` (the default of
`head_tags`), an installed page draws under the status bar, the notch or Dynamic Island and the home
indicator. `safe-area.css` defines `--safe-top`, `--safe-right`, `--safe-bottom` and `--safe-left`
(the `env(safe-area-inset-*)` values) and utility classes. Pad every region at an edge, including
each `position: fixed` element, which ignores the body's padding:

```css
body   { padding: max(16px, var(--safe-top)) max(16px, var(--safe-right))
                  max(16px, var(--safe-bottom)) max(16px, var(--safe-left));
         min-height: 100dvh; }                       /* not 100vh: it runs under iOS's toolbar */
.dock  { position: fixed; bottom: 0; left: 0; right: 0;
         padding-bottom: max(16px, var(--safe-bottom)); }
```

## pwa.js

```js
import { displayMode, markDisplayMode, MediaSessionControl, ScreenWakeLock } from "./pwa/pwa.js";

markDisplayMode(document, window);   // <html data-display-mode="standalone"> once installed
const media = new MediaSessionControl(navigator, { onPlay, onPause, onStop });  // live: no seek/skip
media.setNowPlaying({ title, artist, album, artwork: [{ src, sizes: "512x512", type: "image/png" }] });
media.setPlaying(true);
const wake = new ScreenWakeLock(navigator, document);
await wake.set(true);
document.addEventListener("visibilitychange", () => wake.sync());
```

Every helper takes its globals as arguments, does nothing where the browser lacks the API, and is
tested in headless Chromium (`tests/unit/test_pwa_browser.py`).

## Service worker

**None is needed to install, and this package ships none.** Evidence, 2026-11-16:

- **Chrome and Edge.** Chrome dropped the service-worker (fetch handler) requirement from its
  install criteria in Chrome 108 on Android and 112 on desktop
  ([Revisiting Chrome's installability criteria](https://developer.chrome.com/blog/update-install-criteria)).
  The current criteria ([web.dev](https://web.dev/articles/install-criteria)) are HTTPS, a manifest
  with a name, 192 and 512 px icons, a `start_url` and a standalone-like display, and some user
  engagement; no service worker. Checked directly in Chromium 148: a page with this package's head
  tags and manifest and no service worker reports no installability errors
  (`Page.getInstallabilityErrors` over the DevTools protocol, full Chromium, persistent profile),
  and its app id is the manifest `id`; a manifest with `display: browser` and no icons reports
  `manifest-display-not-supported` and `manifest-missing-suitable-icon` on the same check.
- **Safari.** Add to Home Screen (iOS, iPadOS) and Add to Dock (macOS) work for any page; since iOS
  26 every site added to the Home Screen opens as a web app by default, manifest or not. No
  service worker is involved.
- **Firefox.** Firefox on Android adds any site to the home screen; no worker is asked for.
  (Desktop Firefox's own install support was not checked here.)

The family norm stands: live pages stay live. A service worker with a fetch handler sits in front
of every navigation, so a cached shell can bypass a sign-in gate and freeze runtime configuration
(a kill switch read from the page goes stale silently), and offline caching has nothing to offer a
live stream. Push needs a worker; an app that wants push or offline adds its own and owns its update
story (cache headers, a forced-reload path, periodic `registration.update()`).

## Verifying an install

- **Chromium:** load the page in full Chromium (not the headless shell, which reports nothing) with
  a persistent profile (an incognito context reports `in-incognito`), then
  `Page.getInstallabilityErrors` (empty) and `Page.getAppId` (the manifest `id`).
- **WebKit:** WebKit has no installability API; check the head (`apple-mobile-web-app-title`,
  `apple-touch-icon`, `theme-color`) and the manifest link in a WebKit page, and the layout at
  phone size with safe-area insets. Add to Home Screen itself is checked on a device.
