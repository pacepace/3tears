// The example's manifest passes @3tears/web-pwa's install checks, and the shell it caches is its own.
import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { appManifest } from "@3tears/web-pwa";
import { SHELL } from "./server.js";

test("the example manifest is installable as written", async () => {
  const manifest = JSON.parse(await readFile(new URL("./basic/manifest.webmanifest", import.meta.url), "utf8"));
  assert.deepEqual(appManifest(manifest), manifest);
});

test("the shell lists only the app's own files, never its API or sign-in", () => {
  for (const path of SHELL) {
    assert.ok(path.startsWith("/app/"), path);
    assert.ok(!path.startsWith("/app/api/") && !path.startsWith("/app/login"), path);
  }
});
