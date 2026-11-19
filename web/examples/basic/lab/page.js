// A page with a module worker, a classic worker and an AudioWorklet, under the shell worker. The
// tests list every one of their files in the shell (/app/shell-lab.js) and check that the worker
// never answers their loads: online they start from the network, offline they do not start at all.
const LIMIT_MS = 5000;

/** Resolves with what fn resolves, or the error's message, or "timeout". */
function settle(fn) {
  return Promise.race([
    fn().catch((e) => `error: ${e && e.message ? e.message : String(e)}`),
    new Promise((resolve) => setTimeout(() => resolve("timeout"), LIMIT_MS)),
  ]);
}

function fromWorker(worker) {
  return new Promise((resolve, reject) => {
    worker.onmessage = (e) => resolve(e.data);
    worker.onerror = (e) => {
      e.preventDefault();
      reject(new Error(e.message || "worker failed to load"));
    };
  });
}

async function run() {
  return {
    module: await settle(() => fromWorker(new Worker("/app/lab/engine.js", { type: "module" }))),
    classic: await settle(() => fromWorker(new Worker("/app/lab/classic.js"))),
    worklet: await settle(async () => {
      const ctx = new OfflineAudioContext(1, 128, 48000);
      await ctx.audioWorklet.addModule("/app/lab/tone.js");
      new AudioWorkletNode(ctx, "tone").connect(ctx.destination);
      return (await ctx.startRendering()).getChannelData(0)[64];
    }),
  };
}

const ready = navigator.serviceWorker
  .register("/app/pwa/shell-worker.js?config=/app/shell-lab.js&update=immediate", { scope: "/app/" })
  .then(() => navigator.serviceWorker.ready);
window.lab = { ready, run };
document.getElementById("state").textContent = "ready";
