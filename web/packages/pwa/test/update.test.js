import test from "node:test";
import assert from "node:assert/strict";
import { SKIP_WAITING, SW_RELOAD_TIMEOUT_MS, checkForUpdates, forceReload, registerWorker, updatePrompt, watchVersion } from "../src/update.js";

function timers() {
  const t = { fns: new Map(), next: 1,
    setInterval(fn) { const id = t.next++; t.fns.set(id, fn); return id; },
    clearInterval(id) { t.fns.delete(id); },
    tick() { for (const fn of [...t.fns.values()]) fn(); } };
  return t;
}

class Target {
  constructor() { this.l = {}; }
  addEventListener(t, f) { (this.l[t] ??= []).push(f); }
  removeEventListener(t, f) { this.l[t] = (this.l[t] ?? []).filter((x) => x !== f); }
  emit(t, e = {}) { for (const f of this.l[t] ?? []) f(e); }
}

function fakeBrowser({ controlled }) {
  const reg = new Target();
  reg.installing = null; reg.waiting = null; reg.updates = 0;
  reg.update = async () => { reg.updates += 1; };
  const container = new Target();
  container.controller = controlled ? {} : null;
  container.register = async (url, opts) => { container.registered = { url, opts }; return reg; };
  const t = timers();
  const win = { navigator: { serviceWorker: container }, location: { reloads: 0, reload() { this.reloads += 1; } }, ...t };
  return { reg, container, win, t };
}

function newWorker(reg) {
  const w = new Target();
  w.state = "installing"; w.posted = [];
  w.postMessage = (m) => w.posted.push(m);
  reg.installing = w;
  reg.emit("updatefound");
  return w;
}

test("a first install is not an update", async () => {
  const { reg, win } = fakeBrowser({ controlled: false });
  let updates = 0;
  await registerWorker("/sw.js", { win, onUpdate: () => updates++ });
  const w = newWorker(reg);
  w.state = "installed"; reg.waiting = w; reg.installing = null;
  w.emit("statechange");
  assert.equal(updates, 0);
});

test("a later version waits; applyUpdate activates it and reloads once it controls", async () => {
  const { reg, container, win } = fakeBrowser({ controlled: true });
  let updates = 0;
  const w0 = await registerWorker("/app/sw.js?x=1", { win, scope: "/app/", type: "classic", onUpdate: () => updates++ });
  assert.deepEqual(container.registered, { url: "/app/sw.js?x=1", opts: { scope: "/app/", type: "classic" } });
  assert.equal(w0.applyUpdate(), false); // nothing waiting
  const w = newWorker(reg);
  w.state = "installed"; reg.waiting = w; reg.installing = null;
  w.emit("statechange");
  w.emit("statechange");
  assert.equal(updates, 1);
  container.emit("controllerchange"); // not asked for: no reload
  assert.equal(win.location.reloads, 0);
  assert.equal(w0.applyUpdate(), true);
  assert.deepEqual(w.posted, [{ type: SKIP_WAITING }]);
  container.emit("controllerchange");
  assert.equal(win.location.reloads, 1);
  w0.stop();
});

test("a version already waiting at registration is reported", async () => {
  const { reg, win } = fakeBrowser({ controlled: true });
  reg.waiting = { postMessage() {} };
  let updates = 0;
  await registerWorker("/sw.js", { win, onUpdate: () => updates++ });
  assert.equal(updates, 1);
});

test("updates are looked for every interval until stopped, or not at all", async () => {
  const { reg, win, t } = fakeBrowser({ controlled: true });
  const w = await registerWorker("/sw.js", { win });
  t.tick(); t.tick();
  assert.equal(reg.updates, 2);
  w.stop(); t.tick();
  assert.equal(reg.updates, 2);
  const b = fakeBrowser({ controlled: true });
  await registerWorker("/sw.js", { win: b.win, checkEveryMs: null });
  assert.equal(b.t.fns.size, 0);
  const own = timers();
  const stop = checkForUpdates({ update: async () => { own.n = (own.n ?? 0) + 1; } }, 1000, own);
  own.tick(); stop(); own.tick();
  assert.equal(own.n, 1);
});

test("without service workers, registration is a no-op", async () => {
  const w = await registerWorker("/sw.js", { win: { navigator: {} } });
  assert.equal(w.registration, null);
  assert.equal(w.applyUpdate(), false);
});

test("watchVersion reports a new version once, and survives a failed fetch", async () => {
  const t = timers();
  let served = { version: "1" };
  let failing = false;
  const fetch = async (url, opts) => {
    assert.equal(opts.cache, "no-store");
    assert.match(url, /^\/version\.json\?ts=\d+$/);
    if (failing) throw new Error("offline");
    return { ok: true, json: async () => served };
  };
  const seen = [];
  const w = watchVersion("/version.json", "1", (v) => seen.push(v), { fetch, timers: t });
  await new Promise((r) => setImmediate(r)); // the first look, at once
  assert.equal(await w.check(), "1");
  failing = true;
  assert.equal(await w.check(), null);
  failing = false;
  served = { version: "2" };
  assert.equal(await w.check(), "2");
  assert.equal(await w.check(), null);
  assert.deepEqual(seen, ["2"]);
  const build = [];
  const byField = watchVersion("/version.json", "b1", (v) => build.push(v), { fetch: async () => ({ ok: true, json: async () => ({ buildId: "b2" }) }), field: "buildId", timers: timers() });
  await byField.check();
  assert.deepEqual(build, ["b2"]);
  w.stop(); byField.stop();
});

test("watchVersion looks at once, and again on focus or when the page comes back into view", async () => {
  class Win extends Target { constructor() { super(); this.document = new Target(); this.document.visibilityState = "visible"; } }
  const win = new Win();
  let looks = 0;
  const fetch = async () => { looks += 1; return { ok: true, json: async () => ({ version: "1" }) }; };
  const w = watchVersion("/v.json?x=1", "1", () => {}, { fetch, timers: timers(), win });
  await new Promise((r) => setImmediate(r));
  assert.equal(looks, 1);
  win.emit("focus");
  win.document.visibilityState = "hidden";
  win.document.emit("visibilitychange");
  win.document.visibilityState = "visible";
  win.document.emit("visibilitychange");
  await new Promise((r) => setImmediate(r));
  assert.equal(looks, 3);
  w.stop();
  win.emit("focus");
  assert.equal(looks, 3);
});

test("the prompt never reloads by itself; it offers the escape hatch when stalled", () => {
  const pending = [];
  const states = [];
  let applied = 0;
  const p = updatePrompt(() => applied++, (s) => states.push(s), { setTimeout: (fn, ms) => pending.push([fn, ms]) });
  assert.equal(p.state, "idle");
  p.reload(); p.reload();
  assert.equal(applied, 1);
  assert.equal(pending[0][1], SW_RELOAD_TIMEOUT_MS);
  pending[0][0]();
  assert.deepEqual(states, ["reloading", "stalled"]);
  const cleared = [];
  const q = updatePrompt(() => {}, () => {}, { setTimeout: () => 7, clearTimeout: (id) => cleared.push(id) });
  q.reload();
  q.cancel();
  assert.deepEqual(cleared, [7]);
});

test("forceReload unregisters every worker, deletes every cache and reloads, whatever fails", async () => {
  const log = [];
  const win = {
    navigator: { serviceWorker: { getRegistrations: async () => [{ unregister: async () => log.push("unregister") }] } },
    caches: { keys: async () => ["a", "b"], delete: async (n) => log.push(`delete ${n}`) },
    location: { reload: (force) => log.push(`reload ${force}`) },
  };
  await forceReload(win);
  assert.deepEqual(log, ["unregister", "delete a", "delete b", "reload true"]);
  const broken = { navigator: { serviceWorker: { getRegistrations: async () => { throw new Error("x"); } } },
    caches: { keys: async () => { throw new Error("y"); } }, location: { reload: () => log.push("reloaded anyway") } };
  await forceReload(broken);
  assert.equal(log.at(-1), "reloaded anyway");
});
