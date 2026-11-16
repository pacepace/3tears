import test from "node:test";
import assert from "node:assert/strict";
import { keyBytes, pushSupported, subscribePush, unsubscribePush, workerRegistration } from "../src/push.js";

const KEY = "BAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0-P0A";

test("keyBytes decodes base64url without padding", () => {
  const bytes = keyBytes(KEY);
  assert.equal(bytes.length, 65);
  assert.equal(bytes[0], 4);
  assert.equal(bytes[62], 62);
  assert.equal(bytes[63], 63);
});

test("pushSupported needs a worker container, PushManager and Notification", async () => {
  assert.equal(pushSupported({ navigator: { serviceWorker: {} }, PushManager: 1, Notification: 1 }), true);
  assert.equal(pushSupported({ navigator: {}, PushManager: 1, Notification: 1 }), false);
  assert.equal(pushSupported({ navigator: { serviceWorker: {} }, Notification: 1 }), false);
  assert.equal(await workerRegistration({ navigator: {} }), null);
  const reg = {};
  assert.equal(await workerRegistration({ navigator: { serviceWorker: { getRegistration: async () => reg } }, PushManager: 1, Notification: 1 }), reg);
});

function registration(existing) {
  const calls = [];
  return { calls, pushManager: {
    getSubscription: async () => existing,
    subscribe: async (opts) => { calls.push(opts); return { endpoint: "new", unsubscribe: async () => true }; },
  } };
}

test("subscribing asks first, fetches the key only once granted, reuses a subscription", async () => {
  let keyFetched = 0;
  const key = async () => { keyFetched += 1; return KEY; };
  assert.equal(await subscribePush(registration(null), key, { Notification: { requestPermission: async () => "denied" } }), "denied");
  assert.equal(keyFetched, 0);
  const fresh = registration(null);
  const sub = await subscribePush(fresh, key, { Notification: { requestPermission: async () => "granted" } });
  assert.equal(sub.endpoint, "new");
  assert.equal(fresh.calls[0].userVisibleOnly, true);
  assert.equal(fresh.calls[0].applicationServerKey.length, 65);
  const existing = { endpoint: "old" };
  assert.equal(await subscribePush(registration(existing), key, { Notification: { requestPermission: async () => "granted" } }), existing);
});

test("unsubscribing forgets the subscription, if any", async () => {
  let gone = false;
  assert.equal(await unsubscribePush(registration({ unsubscribe: async () => { gone = true; } })), true);
  assert.equal(gone, true);
  assert.equal(await unsubscribePush(registration(null)), false);
});
