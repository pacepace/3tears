// @3tears/pwa shell worker: a complete service worker for an app with no build step. A classic
// script (no imports), registered as is, configured by its own URL:
//
//   navigator.serviceWorker.register("/app/pwa/shell-worker.js?config=/app/shell.json", { scope: "/app/" })
//
// With no config it caches nothing: every request goes to the network as if there were no worker
// (an app that wants only to be installable, or only the update protocol). With a config
// (JSON, fetched fresh at install):
//
//   { "version": "1.4.2",                 the build; a new one makes a new worker and cache
//     "cache": "player",                  cache name prefix (default "shell")
//     "shell": ["/app/", "/app/app.js"],  the app's own files, kept for when the network is gone
//     "never": ["^/api/", "^/auth/"],     regexes of paths never touched (checked first)
//     "update": "prompt" }                "prompt" (default) or "immediate"
//
// Policy (3tears, agreed 2026-11-09):
// - Only the shell list is ever cached. Audio, API calls and sign-in are never cached: anything
//   not on the list passes through untouched, and the "never" list is checked before it.
// - The shell is network-first: online, every load is the server's current file, so the page and
//   its scripts never come from two versions; the cache answers only when the network does not.
//   A response that is not a plain 200 (a redirect to sign-in, an error) is never stored.
// - A new version waits ("prompt") until the page posts {type: "SKIP_WAITING"} (the person agreed:
//   @3tears/pwa's registerWorker + applyUpdate), so a running page is never replaced behind the
//   person's back. "immediate" activates at once, for an app with nothing to lose on a reload.
"use strict";

const SKIP_WAITING = "SKIP_WAITING";
const params = new URL(self.location.href).searchParams;
const configUrl = params.get("config");
let config = { version: "", cache: "shell", shell: [], never: [], update: params.get("update") || "prompt" };
let ready = null;
// The config is kept beside the shell, so a worker the browser stopped and started again (it does,
// between events) knows its shell before the next fetch arrives.
const CONFIG_CACHE = "3tears-shell-config";
const CONFIG_KEY = new URL(self.location.href).href;

function cacheName() {
  return `${config.cache}-${config.version}`;
}

function shellPath(url) {
  return url.origin === self.location.origin ? url.pathname : null;
}

function isNever(path) {
  return config.never.some((re) => re.test(path));
}

function isShell(path) {
  return config.shell.includes(path);
}

/** A response worth keeping: a plain 200 from this origin, not reached through a redirect. */
function storable(response) {
  return response && response.ok && response.status === 200 && !response.redirected && response.type === "basic";
}

function applyConfig(raw) {
  config = {
    version: String(raw.version || ""),
    cache: String(raw.cache || "shell"),
    shell: Array.isArray(raw.shell) ? raw.shell.map((p) => new URL(p, self.location.origin).pathname) : [],
    never: Array.isArray(raw.never) ? raw.never.map((s) => new RegExp(s)) : [],
    update: raw.update === "immediate" ? "immediate" : params.get("update") === "immediate" ? "immediate" : "prompt",
  };
}

async function loadConfig() {
  if (!configUrl) return;
  const response = await fetch(new URL(configUrl, self.location.href), { cache: "no-store" });
  if (!response.ok) throw new Error(`shell worker config ${configUrl}: HTTP ${response.status}`);
  const raw = await response.json();
  applyConfig(raw);
  const store = await caches.open(CONFIG_CACHE);
  await store.put(CONFIG_KEY, new Response(JSON.stringify(raw), { headers: { "Content-Type": "application/json" } }));
}

async function restoreConfig() {
  if (!configUrl) return;
  const kept = await (await caches.open(CONFIG_CACHE)).match(CONFIG_KEY);
  if (kept) applyConfig(await kept.json());
}

// A worker started again after the browser stopped it: its config, from where install kept it.
const restored = restoreConfig().catch(() => {});

self.addEventListener("install", (event) => {
  ready = loadConfig();
  event.waitUntil(
    (async () => {
      await ready;
      if (config.shell.length) {
        const cache = await caches.open(cacheName());
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
      await (ready || restored).catch(() => {});
      const keep = cacheName();
      for (const name of await caches.keys()) {
        if (name.startsWith(`${config.cache}-`) && name !== keep) await caches.delete(name);
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
  const path = shellPath(new URL(request.url));
  if (path === null || isNever(path) || !isShell(path)) return; // the network, untouched
  event.respondWith(
    (async () => {
      try {
        const response = await fetch(request);
        if (storable(response)) {
          const copy = response.clone();
          event.waitUntil(caches.open(cacheName()).then((cache) => cache.put(path, copy)));
        }
        return response;
      } catch (err) {
        const cached = await caches.match(path, { cacheName: cacheName() });
        if (cached) return cached;
        throw err;
      }
    })(),
  );
});

// The worker's state, for tests and for a page that wants to show it.
self.threeTearsShell = {
  get config() {
    return config;
  },
};
