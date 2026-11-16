import test from "node:test";
import assert from "node:assert/strict";
import { vitePwa } from "../src/vite.js";

const manifest = { name: "App", start_url: "/", scope: "/", display: "standalone",
  icons: [{ src: "a.png", sizes: "192x192", type: "image/png" }, { src: "b.png", sizes: "512x512", type: "image/png" }] };

test("the policy: prompt, the app's own worker, runtime files out of the precache", () => {
  assert.deepEqual(vitePwa({ manifest, runtimeFiles: ["config.js"] }), {
    registerType: "prompt", injectRegister: "auto", manifest, strategies: "injectManifest", srcDir: "src", filename: "sw.ts",
    injectManifest: { globIgnores: ["**/config.js"] },
  });
  assert.equal(vitePwa({ manifest, registerType: "autoUpdate", devOptions: { enabled: true } }).devOptions.enabled, true);
  assert.throws(() => vitePwa({ manifest: { ...manifest, start_url: "https://x/" } }), /start_url/);
});
