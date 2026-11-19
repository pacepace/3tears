import test from "node:test";
import assert from "node:assert/strict";
import { ScreenWakeLock } from "../src/wake-lock.js";

function lockApi(log) {
  return { wakeLock: { request: async (type) => { log.push(`request ${type}`);
    return { released: false, async release() { this.released = true; log.push("release"); } }; } } };
}

test("held while wanted and visible, released when hidden, taken again on return", async () => {
  const log = [];
  const doc = { visibilityState: "visible" };
  const lock = new ScreenWakeLock(lockApi(log), doc);
  assert.equal(await lock.set(true), true);
  assert.equal(await lock.sync(), true);
  doc.visibilityState = "hidden";
  assert.equal(await lock.sync(), false);
  doc.visibilityState = "visible";
  assert.equal(await lock.sync(), true);
  assert.equal(await lock.set(false), false);
  assert.deepEqual(log, ["request screen", "release", "request screen", "release"]);
});

test("a refused or missing API, or one that throws at once, leaves nothing pending", async () => {
  const doc = { visibilityState: "visible" };
  assert.equal(await new ScreenWakeLock({ wakeLock: { request: async () => { throw new Error("no"); } } }, doc).set(true), false);
  assert.equal(await new ScreenWakeLock({}, doc).set(true), false);
  let n = 0;
  const sync = new ScreenWakeLock({ wakeLock: { request: () => { n += 1; throw new Error("at once"); } } }, doc);
  assert.equal(await sync.set(true), false);
  assert.equal(sync.pending, null);
  assert.equal(await sync.sync(), false);
  assert.equal(n, 2); // tried again, not stuck on a settled promise
});

test("given up while the request ran: released at once", async () => {
  const log = [];
  const doc = { visibilityState: "visible" };
  const lock = new ScreenWakeLock(lockApi(log), doc);
  const p = lock.set(true);
  lock.wanted = false;
  assert.equal(await p, false);
  assert.deepEqual(log, ["request screen", "release"]);
});
