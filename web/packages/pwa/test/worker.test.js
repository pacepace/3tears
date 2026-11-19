import test from "node:test";
import assert from "node:assert/strict";
import { SKIP_WAITING, listenForSkipWaiting, openOnNotificationClick, readPushData, showPushes } from "../src/worker.js";

function fakeSelf({ windows = [] } = {}) {
  const self = { handlers: {}, shown: [], opened: [], skipped: 0,
    location: { origin: "https://app.example" },
    addEventListener(t, f) { this.handlers[t] = f; },
    skipWaiting: async () => { self.skipped += 1; },
    registration: { showNotification: async (title, options) => self.shown.push({ title, options }) },
    clients: { matchAll: async () => windows, openWindow: async (url) => self.opened.push(url) } };
  return self;
}

async function dispatch(self, type, event) {
  let work = Promise.resolve();
  self.handlers[type]({ ...event, waitUntil: (p) => (work = p) });
  await work;
}

test("only the page's SKIP_WAITING activates a waiting worker", () => {
  const self = fakeSelf();
  listenForSkipWaiting(self);
  self.handlers.message({ data: { type: "other" } });
  self.handlers.message({ data: null });
  assert.equal(self.skipped, 0);
  self.handlers.message({ data: { type: SKIP_WAITING } });
  assert.equal(self.skipped, 1);
});

test("push data is JSON, else text, else null", () => {
  assert.deepEqual(readPushData({ data: { json: () => ({ a: 1 }), text: () => "x" } }), { a: 1 });
  assert.equal(readPushData({ data: { json: () => { throw new SyntaxError(); }, text: () => "hi" } }), "hi");
  assert.equal(readPushData({ data: null }), null);
});

test("every push is shown, as the app turns it into a notification", async () => {
  const self = fakeSelf();
  showPushes(self, (data) => ({ title: typeof data === "string" ? "App" : data.title, options: { body: String(data.body ?? data) } }));
  await dispatch(self, "push", { data: { json: () => ({ title: "Hi", body: "there" }) } });
  await dispatch(self, "push", { data: { json: () => { throw new SyntaxError(); }, text: () => "plain" } });
  assert.deepEqual(self.shown, [{ title: "Hi", options: { body: "there" } }, { title: "App", options: { body: "plain" } }]);
});

test("a tap brings an open window forward with the message, else opens the app at the url", async () => {
  const posted = [];
  const focused = [];
  const open = { url: "https://app.example/x", focus: async () => focused.push(1), postMessage: (m) => posted.push(m) };
  const other = { url: "https://elsewhere.example/", focus: async () => focused.push(2), postMessage: () => {} };
  const route = (data) => ({ url: `/?c=${data?.id ?? ""}`, message: data?.id ? { type: "open", id: data.id } : undefined });
  const withWindow = fakeSelf({ windows: [other, open] });
  openOnNotificationClick(withWindow, route);
  let closed = 0;
  await dispatch(withWindow, "notificationclick", { notification: { data: { id: "7" }, close: () => closed++ } });
  assert.deepEqual([closed, focused, posted], [1, [1], [{ type: "open", id: "7" }]]);
  await dispatch(withWindow, "notificationclick", { notification: { data: null, close() {} } });
  assert.equal(posted.length, 1); // no message: focused only
  const none = fakeSelf({ windows: [other] });
  openOnNotificationClick(none, route);
  await dispatch(none, "notificationclick", { notification: { data: { id: "7" }, close() {} } });
  assert.deepEqual(none.opened, ["https://app.example/?c=7"]);
});
