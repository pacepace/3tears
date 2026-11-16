// @3tears/pwa shell worker: a complete service worker for an app with no build step. A classic
// script (no imports of its own), registered as is, configured by a script the app serves:
//
//   registerWorker("/app/pwa/shell-worker.js?config=/app/shell.js", { scope: "/app/" })
//
//   // /app/shell.js, written by the app's server or build:
//   self.SHELL_CONFIG = {
//     version: "1.4.2",                 // the build; names the shell cache
//     cache: "app",                     // cache name prefix (default "shell")
//     shell: ["/app/", "/app/app.js"],  // the app's own files, kept for when the network is gone
//     never: ["^/app/api/"],            // regexes of paths never touched (checked first)
//     update: "prompt",                 // "prompt" (default) or "immediate"
//   };
//
// The config is loaded with importScripts, so it is part of the worker: browsers compare a
// worker's imported scripts as well as its own bytes when they look for an update, and a release
// that changes the config (its version, its files) is a new worker, though this file never
// changes. It is also there, synchronously, every time the browser starts the worker again.
//
// With no config the worker caches nothing: every request goes to the network as if there were
// no worker (an app that wants only to be installable, or only the update protocol).
//
// Policy (3tears, agreed 2026-11-09):
// - Only the shell list is ever cached. Audio, API calls and sign-in are never cached: anything
//   not on the list passes through untouched, and the "never" list is checked before it.
// - The shell is network-first: online, every load is the server's current file, so the page and
//   its scripts never come from two versions; the cache answers only when the network does not.
//   A response that is not a plain 200 (a redirect to sign-in, an error) is never stored.
// - A new version waits ("prompt") until the page posts {type: "SKIP_WAITING"} (the person agreed:
//   @3tears/pwa's registerWorker + applyUpdate). "immediate" activates at once, for an app with
//   nothing to lose.
"use strict";

const SKIP_WAITING = "SKIP_WAITING";
const params = new URL(self.location.href).searchParams;
const configUrl = params.get("config");
if (configUrl) importScripts(configUrl);
const raw = (configUrl && self.SHELL_CONFIG) || {};
const config = {
  version: String(raw.version || ""),
  cache: String(raw.cache || "shell"),
  shell: Array.isArray(raw.shell) ? raw.shell.map((p) => new URL(p, self.location.origin).pathname) : [],
  never: Array.isArray(raw.never) ? raw.never.map((s) => new RegExp(s)) : [],
  update: raw.update === "immediate" || params.get("update") === "immediate" ? "immediate" : "prompt",
};
const CACHE = `${config.cache}-${config.version}`;

function isNever(path) {
  return config.never.some((re) => re.test(path));
}

/** A response worth keeping: a plain 200 from this origin, not reached through a redirect. */
function storable(response) {
  return response && response.ok && response.status === 200 && !response.redirected && response.type === "basic";
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      if (config.shell.length) {
        const cache = await caches.open(CACHE);
        await Promise.all(
          config.shell.map(async (path) => {
            try {
              const response = await fetch(path, { cache: "no-store", credentials: "same-origin" });
              if (storable(response)) await cache.put(path, response);
            } catch {
              // not reachable now: the shell fills in as the app is used
            }
          }),
        );
      }
      if (config.update === "immediate") await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      if (configUrl) {
        // this build's shell replaces the last one's; nothing else is touched
        for (const name of await caches.keys()) {
          if (name.startsWith(`${config.cache}-`) && name !== CACHE) await caches.delete(name);
        }
      }
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("message", (event) => {
  if (event.data && event.data.type === SKIP_WAITING) self.skipWaiting();
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  const path = url.pathname;
  if (isNever(path) || !config.shell.includes(path)) return; // the network, untouched
  event.respondWith(
    (async () => {
      try {
        const response = await fetch(request);
        if (storable(response)) {
          const copy = response.clone();
          event.waitUntil(caches.open(CACHE).then((cache) => cache.put(path, copy)));
        }
        return response;
      } catch (err) {
        const cached = await caches.match(path, { cacheName: CACHE });
        if (cached) return cached;
        throw err;
      }
    })(),
  );
});

// The worker's config, for tests and for a page that wants to show it.
self.threeTearsShell = { config };
