// The shell worker never answers worker or worklet loads, even listed in its shell: the example's
// worker lab (basic/lab/) in headless Chromium, with every worker file in the shell list.
import test from "node:test";
import assert from "node:assert/strict";
import { LAB_WORKERS } from "../server.js";
import { useChromium } from "./chromium.js";

const env = useChromium();

const WORKS = { module: "abc", classic: "d", worklet: 0.5 };

test("worker and worklet loads go to the network, never to the shell worker, though listed in the shell", async (t) => {
  if (env.skip) return t.skip(env.skip);
  const context = await env.browser.newContext();
  /** @type {Map<string, boolean>} path to whether its last response came from the service worker */
  const seen = new Map();
  context.on("response", (response) => {
    const url = new URL(response.url());
    if (url.pathname.startsWith("/app/lab/")) seen.set(url.pathname, response.fromServiceWorker());
  });
  const page = await context.newPage();
  await page.goto(`${env.app.url}/app/lab/`);
  await page.evaluate(() => window.lab.ready.then(() => true));
  await page.reload(); // now under the worker's control
  await page.waitForFunction(() => Boolean(navigator.serviceWorker.controller));

  // the worker took the listed files into its cache, except the one named like a worker
  const cached = await page.evaluate(async () => {
    const out = [];
    for (const name of (await caches.keys()).filter((n) => n.startsWith("lab-"))) {
      for (const req of await (await caches.open(name)).keys()) out.push(new URL(req.url).pathname);
    }
    return out;
  });
  for (const p of LAB_WORKERS.filter((p) => !p.endsWith("decoder-worker.js"))) assert.ok(cached.includes(p), `${p} cached`);
  assert.ok(!cached.includes("/app/lab/decoder-worker.js"), "the worker-named entry was dropped");

  // online: every worker and the worklet start, their files from the network
  const before = Object.fromEntries(LAB_WORKERS.map((p) => [p, env.app.hits.get(p) ?? 0]));
  seen.clear();
  assert.deepEqual(await page.evaluate(() => window.lab.run()), WORKS);
  for (const p of LAB_WORKERS.filter((p) => !p.endsWith("decoder-worker.js"))) {
    assert.ok((env.app.hits.get(p) ?? 0) > before[p], `${p} came from the server`);
    // Chromium reports the workers' loads (not the worklet's): none answered by the service worker
    if (seen.has(p)) assert.equal(seen.get(p), false, `${p} was answered by the service worker`);
  }
  assert.ok(seen.has("/app/lab/engine.js") && seen.has("/app/lab/classic-dep.js"), [...seen.keys()].join(" "));

  // offline: the page comes from the shell, and no worker load is answered from the cache
  await context.setOffline(true);
  await page.reload();
  await page.waitForFunction(() => document.getElementById("state").textContent === "ready");
  assert.equal(seen.get("/app/lab/page.js"), true, "page.js answered by the service worker offline");
  const hits = Object.fromEntries(LAB_WORKERS.map((p) => [p, env.app.hits.get(p) ?? 0]));
  const offline = await page.evaluate(() => window.lab.run());
  assert.notEqual(offline.module, WORKS.module, `module worker started offline: ${offline.module}`);
  assert.notEqual(offline.classic, WORKS.classic, `classic worker started offline: ${offline.classic}`);
  // Chromium's offline emulation does not reach worklet fetches, so the worklet still starts: from
  // the server, both its files, not from the shell cache
  assert.equal(offline.worklet, WORKS.worklet);
  for (const p of ["/app/lab/tone.js", "/app/lab/tone-dep.js"]) assert.ok((env.app.hits.get(p) ?? 0) > hits[p], `${p} from the server`);
  await context.close();
});
