// shell-worker.js in a stand-in service worker global: which requests it answers, and that worker
// and worklet loads never are.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const SOURCE = await readFile(new URL("../src/shell-worker.js", import.meta.url), "utf8");
const ORIGIN = "https://app.example";

/**
 * Runs the worker with a config and stand-ins for what it uses.
 * @param {object | null} shellConfig
 * @param {Record<string, string>} [clientTypes] client id to its type, as clients.get reports it
 */
function startWorker(shellConfig, clientTypes = {}) {
  const listeners = {};
  const warnings = [];
  const lookups = [];
  const self = {
    location: { href: `${ORIGIN}/app/pwa/shell-worker.js${shellConfig ? "?config=/app/shell.js" : ""}`, origin: ORIGIN },
    addEventListener: (type, fn) => {
      listeners[type] = fn;
    },
    clients: {
      get: async (id) => {
        lookups.push(id);
        if (clientTypes[id] === "fails") throw new Error("lookup failed");
        return clientTypes[id] ? { id, type: clientTypes[id] } : undefined;
      },
      claim: async () => {},
    },
    skipWaiting: async () => {},
  };
  const context = vm.createContext({
    self,
    URL,
    Set,
    Promise,
    console: { warn: (...args) => warnings.push(args.join(" ")) },
    importScripts: () => {
      self.SHELL_CONFIG = shellConfig;
    },
    caches: { open: async () => ({ put: async () => {} }), keys: async () => [], match: async () => undefined },
    fetch: async () => ({ ok: true, status: 200 }),
  });
  vm.runInContext(SOURCE, context);
  /** Dispatches a fetch event; true when the worker answered it. */
  const request = (path, { destination = "", clientId = "", resultingClientId = "", method = "GET" } = {}) => {
    let answered = false;
    listeners.fetch({
      request: { url: `${ORIGIN}${path}`, method, destination },
      clientId,
      resultingClientId,
      respondWith: () => {
        answered = true;
      },
      waitUntil: () => {},
    });
    return answered;
  };
  return { request, warnings, lookups, state: self.threeTearsShell };
}

const SHELL = { version: "1", shell: ["/app/", "/app/app.js", "/app/engine.js", "/app/engine-dep.js", "/app/tone.js"] };

test("a page's loads of shell files are answered; everything else passes through", () => {
  const w = startWorker(SHELL);
  assert.equal(w.request("/app/", { destination: "document", resultingClientId: "page" }), true);
  assert.equal(w.request("/app/app.js", { destination: "script", clientId: "page" }), true);
  assert.equal(w.request("/app/api/now", { clientId: "page" }), false);
  assert.equal(w.request("/app/app.js", { method: "POST", clientId: "page" }), false);
});

test("worker and worklet scripts are never answered, even when listed", () => {
  const w = startWorker(SHELL);
  for (const destination of ["worker", "sharedworker", "serviceworker", "audioworklet", "paintworklet"]) {
    assert.equal(w.request("/app/engine.js", { destination, clientId: "page" }), false, destination);
  }
  assert.equal(w.request("/app/tone.js", { destination: "audioworklet", clientId: "page" }), false);
  // told once per listed path, naming it
  assert.equal(w.warnings.length, 2, w.warnings.join("\n"));
  assert.match(w.warnings[0], /\/app\/engine\.js is in the shell list/);
  assert.deepEqual([...w.state.workerLoads], ["/app/engine.js", "/app/tone.js"]);
});

test("every request from a worker passes through: its imports, dynamic imports and fetches", () => {
  const w = startWorker(SHELL);
  // the worker's own script names the client it creates
  assert.equal(w.request("/app/engine.js", { destination: "worker", clientId: "page", resultingClientId: "w1" }), false);
  // importScripts and import() from it are destination "script"
  assert.equal(w.request("/app/engine-dep.js", { destination: "script", clientId: "w1" }), false);
  assert.equal(w.request("/app/app.js", { destination: "script", clientId: "w1" }), false);
  assert.equal(w.request("/app/", { clientId: "w1" }), false);
  // the page that started it is still answered
  assert.equal(w.request("/app/app.js", { destination: "script", clientId: "page" }), true);
});

test("a worker client first seen after the service worker restarted is looked up, then passed through", async () => {
  const w = startWorker(SHELL, { w2: "worker", s1: "sharedworker", page: "window" });
  w.request("/app/data.json", { clientId: "w2" }); // not in the shell: passes through, and teaches
  w.request("/app/data.json", { clientId: "s1" });
  w.request("/app/data.json", { clientId: "page" });
  await new Promise((r) => setImmediate(r));
  assert.equal(w.request("/app/engine-dep.js", { destination: "script", clientId: "w2" }), false);
  assert.equal(w.request("/app/engine-dep.js", { destination: "script", clientId: "s1" }), false);
  assert.equal(w.request("/app/engine-dep.js", { destination: "script", clientId: "page" }), true);
  // each client is looked up once
  w.request("/app/data.json", { clientId: "page" });
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(w.lookups, ["w2", "s1", "page"]);
});

test("a lookup that fails or finds no client is asked again; it leaves a warning", async () => {
  const types = { w3: "fails", w4: undefined };
  const w = startWorker(SHELL, types);
  w.request("/app/data.json", { clientId: "w3" });
  w.request("/app/data.json", { clientId: "w4" });
  await new Promise((r) => setImmediate(r));
  assert.match(w.warnings.join("\n"), /could not look up client w3/);
  types.w3 = "worker";
  types.w4 = "worker";
  w.request("/app/data.json", { clientId: "w3" });
  w.request("/app/data.json", { clientId: "w4" });
  await new Promise((r) => setImmediate(r));
  assert.equal(w.request("/app/engine-dep.js", { destination: "script", clientId: "w3" }), false);
  assert.equal(w.request("/app/engine-dep.js", { destination: "script", clientId: "w4" }), false);
});

test("a worker forgotten when the client memory fills is learned again, never stuck as looked up", async () => {
  const types = { w5: "worker" };
  for (let i = 0; i < 600; i++) types[`page${i}`] = "window";
  const w = startWorker(SHELL, types);
  w.request("/app/engine.js", { destination: "worker", clientId: "page0", resultingClientId: "w5" });
  for (let i = 0; i < 600; i++) w.request("/app/data.json", { clientId: `page${i}` });
  await new Promise((r) => setImmediate(r));
  w.request("/app/data.json", { clientId: "w5" }); // forgotten: looked up again
  await new Promise((r) => setImmediate(r));
  assert.equal(w.request("/app/engine-dep.js", { destination: "script", clientId: "w5" }), false);
});

test("shell entries named like worker or worklet scripts are dropped, with a warning naming them", () => {
  const w = startWorker({
    version: "1",
    shell: ["/app/", "/app/moq-worker.js", "/app/decoder-worker.mjs", "/app/audio-worklet.js", "/app/pwa/shell-worker.js",
      "/app/decoderWorker.js", "/app/worker.js", "/app/workers.html", "/app/networker.js", "/app/worker-docs/app.js"],
  });
  assert.deepEqual([...w.state.config.shell], ["/app/", "/app/workers.html", "/app/networker.js", "/app/worker-docs/app.js"]);
  assert.deepEqual([...w.state.rejected], ["/app/moq-worker.js", "/app/decoder-worker.mjs", "/app/audio-worklet.js",
    "/app/pwa/shell-worker.js", "/app/decoderWorker.js", "/app/worker.js"]);
  assert.equal(w.warnings.length, 1);
  assert.match(w.warnings[0], /dropped \/app\/moq-worker\.js, .* from the shell list/);
  assert.equal(w.request("/app/moq-worker.js", { destination: "script", clientId: "page" }), false);
});

test("with no config nothing is answered and nothing is warned", () => {
  const w = startWorker(null);
  assert.equal(w.request("/app/", { destination: "document" }), false);
  assert.equal(w.request("/app/x.js", { destination: "worker", resultingClientId: "w" }), false);
  assert.deepEqual(w.warnings, []);
});
