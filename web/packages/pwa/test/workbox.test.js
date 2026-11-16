import test from "node:test";
import assert from "node:assert/strict";
import { NEVER_SHELL, appShell } from "../src/workbox.js";

function recorder() {
  const calls = [];
  const routes = [];
  const wb = {
    clientsClaim: () => calls.push("clientsClaim"),
    precacheAndRoute: (m) => calls.push(`precacheAndRoute:${JSON.stringify(m)}`),
    cleanupOutdatedCaches: () => calls.push("cleanupOutdatedCaches"),
    createHandlerBoundToURL: (url) => ({ boundTo: url }),
    NavigationRoute: class { constructor(handler, options) { this.handler = handler; this.options = options; } },
    registerRoute: (match, handler, method) => { calls.push("registerRoute"); routes.push({ match, handler, method }); },
    NetworkFirst: class { constructor(options) { this.options = options; } },
    ExpirationPlugin: class { constructor(options) { this.options = options; } },
  };
  return { wb, calls, routes };
}

test("the generated worker's order: claim, precache, clean, shell navigation, network-first routes", () => {
  const { wb, calls, routes } = recorder();
  appShell(wb, { precache: [{ url: "index.html", revision: "a" }], neverShell: [...NEVER_SHELL, /^\/s3\//],
    navigationCache: "app-navigation", networkFirst: [{ path: "/config.js", cacheName: "app-runtime-config" }] });
  assert.deepEqual(calls, ["clientsClaim", 'precacheAndRoute:[{"url":"index.html","revision":"a"}]', "cleanupOutdatedCaches",
    "registerRoute", "registerRoute", "registerRoute"]);
  assert.deepEqual(routes[0].match.handler, { boundTo: "index.html" });
  assert.deepEqual(routes[0].match.options.denylist.map(String), ["/^\\/api/", "/^\\/ws/", "/^\\/healthz/", "/^\\/s3\\//"]);
  assert.equal(routes[1].method, "GET");
  assert.equal(routes[1].match({ request: { mode: "navigate" } }), true);
  assert.equal(routes[1].match({ request: { mode: "cors" } }), false);
  assert.deepEqual(routes[1].handler.options.cacheName, "app-navigation");
  assert.equal(routes[1].handler.options.networkTimeoutSeconds, 3);
  assert.deepEqual(routes[1].handler.options.plugins[0].options, { maxEntries: 1, maxAgeSeconds: 86400 });
  assert.equal(routes[2].match({ url: new URL("https://x/config.js") }), true);
  assert.equal(routes[2].match({ url: new URL("https://x/config.json") }), false);
  assert.equal(routes[2].handler.options.cacheName, "app-runtime-config");
});

test("defaults: index.html, the standard never-shell paths, no extra routes", () => {
  const { wb, routes } = recorder();
  appShell(wb, { precache: [], navigationCache: "nav" });
  assert.equal(routes.length, 2);
  assert.deepEqual(routes[0].match.options.denylist, [...NEVER_SHELL]);
});
