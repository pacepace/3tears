// The example app in headless Chromium, with the real shell worker: what it caches and never
// caches, working offline, and an update that waits for the person's click.
//
// A machine without Chromium skips with the launch error; where THREETEARS_REQUIRE_CHROMIUM=1 (CI,
// the testament pool) a launch that fails fails instead, so a missing browser never reads as a pass.
import test, { after, before } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { startExample } from "../server.js";

/** @type {Awaited<ReturnType<typeof startExample>>} */
let app;
/** @type {import("playwright").Browser | null} */
let browser = null;
let skip = "";

before(async () => {
  app = await startExample();
  try {
    browser = await chromium.launch({ args: ["--disable-dev-shm-usage"] });
  } catch (e) {
    if (process.env.THREETEARS_REQUIRE_CHROMIUM === "1") throw e;
    skip = `headless Chromium cannot launch: ${String(e).split("\n")[0]}`;
  }
});

after(async () => {
  await browser?.close();
  await app?.close();
});

/** A fresh profile on the example, its worker installed and in control. */
async function openApp() {
  const context = await /** @type {import("playwright").Browser} */ (browser).newContext();
  const page = await context.newPage();
  await page.goto(`${app.url}/app/`);
  await page.evaluate(() => window.exampleWorker.then(() => true));
  await page.evaluate(() => navigator.serviceWorker.ready);
  if (!(await page.evaluate(() => Boolean(navigator.serviceWorker.controller)))) {
    await page.waitForFunction(() => Boolean(navigator.serviceWorker.controller));
  }
  return { context, page };
}

/** The paths in the example's shell cache. */
function cachedPaths(page) {
  return page.evaluate(async () => {
    const names = await caches.keys();
    const paths = [];
    for (const name of names.filter((n) => n.startsWith("example-"))) {
      for (const req of await (await caches.open(name)).keys()) paths.push(new URL(req.url).pathname);
    }
    return { names, paths: paths.sort() };
  });
}

test("the shell is cached, the API and sign-in never are", async (t) => {
  if (skip) return t.skip(skip);
  app.setVersion("1");
  const { context, page } = await openApp();
  const { names, paths } = await cachedPaths(page);
  assert.deepEqual(names.filter((n) => n.startsWith("example-")), ["example-1"]);
  for (const p of ["/app/", "/app/app.js", "/app/pwa/media-session.js", "/app/icon-512.png"]) assert.ok(paths.includes(p), p);
  // an API call, a redirect to sign-in and a file reached through one: passed through, never stored
  await page.evaluate(async () => {
    await fetch("/app/api/now");
    await fetch("/app/private.js");
  });
  const after = await cachedPaths(page);
  assert.ok(!after.paths.some((p) => p.startsWith("/app/api/") || p === "/app/private.js" || p === "/app/login"), after.paths.join(" "));
  await context.close();
});

test("online, every load is the server's current file; offline, the shell answers", async (t) => {
  if (skip) return t.skip(skip);
  app.setVersion("1");
  const { context, page } = await openApp();
  const before = app.hits.get("/app/app.js") ?? 0;
  await page.reload();
  await page.waitForFunction(() => document.getElementById("version").textContent === "version 1");
  assert.ok((app.hits.get("/app/app.js") ?? 0) > before, "app.js came from the network while online");
  await context.setOffline(true);
  await page.reload();
  await page.waitForFunction(() => document.getElementById("version").textContent === "version 1");
  assert.equal(await page.evaluate(() => fetch("/app/api/now").then(() => "answered", () => "failed")), "failed");
  await context.close();
});

test("a new version waits for the person, then the page reloads into it", async (t) => {
  if (skip) return t.skip(skip);
  app.setVersion("1");
  const { context, page } = await openApp();
  app.setVersion("2"); // a deploy
  await page.evaluate(() => window.exampleWorker.then((w) => w.registration.update()));
  await page.waitForFunction(() => !document.getElementById("update").hidden);
  // still the old version: nothing replaced the running page
  assert.equal(await page.textContent("#version"), "version 1");
  const reloaded = page.waitForEvent("load");
  await page.click("#reload");
  await reloaded;
  await page.waitForFunction(() => document.getElementById("version").textContent === "version 2");
  const { names } = await cachedPaths(page);
  assert.deepEqual(names.filter((n) => n.startsWith("example-")), ["example-2"]);
  await context.close();
});

test("the lock screen gets the app's now playing, and the page its display mode", async (t) => {
  if (skip) return t.skip(skip);
  const { context, page } = await openApp();
  const state = await page.evaluate(() => ({
    title: navigator.mediaSession.metadata?.title,
    art: navigator.mediaSession.metadata?.artwork[0]?.src,
    mode: document.documentElement.dataset.displayMode,
  }));
  assert.deepEqual(state, { title: "Example", art: `${app.url}/app/icon-512.png`, mode: "browser" });
  await page.click("#play");
  assert.equal(await page.evaluate(() => navigator.mediaSession.playbackState), "playing");
  await context.close();
});

test("with no config the worker caches nothing and passes everything through", async (t) => {
  if (skip) return t.skip(skip);
  const context = await /** @type {import("playwright").Browser} */ (browser).newContext();
  const page = await context.newPage();
  await page.goto(`${app.url}/app/api/blank`);
  const result = await page.evaluate(async () => {
    const reg = await navigator.serviceWorker.register("/app/pwa/shell-worker.js?update=immediate", { scope: "/app/api/" });
    await navigator.serviceWorker.ready;
    await new Promise((r) => (reg.active ? r() : reg.installing?.addEventListener("statechange", r)));
    await fetch("/app/api/now");
    return caches.keys();
  });
  assert.deepEqual(result.filter((n) => n !== "3tears-shell-config"), []);
  await context.close();
});
