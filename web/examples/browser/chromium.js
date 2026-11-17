// Headless Chromium for the browser tests, and the example app it opens.
//
// A machine without Chromium skips with the launch error; where THREETEARS_REQUIRE_CHROMIUM=1 (CI,
// the testament pool) a launch that fails fails instead, so a missing browser never reads as a pass.
import { after, before } from "node:test";
import { chromium } from "playwright";
import { startExample } from "../server.js";

/**
 * Starts the example and Chromium before the file's tests and stops both after. Read the returned
 * object's fields inside the tests: they are set once `before` has run.
 * @returns {{ app: Awaited<ReturnType<typeof startExample>>, browser: import("playwright").Browser, skip: string }}
 */
export function useChromium() {
  const env = /** @type {any} */ ({ app: null, browser: null, skip: "" });
  before(async () => {
    env.app = await startExample();
    try {
      env.browser = await chromium.launch({ args: ["--disable-dev-shm-usage"] });
    } catch (e) {
      if (process.env.THREETEARS_REQUIRE_CHROMIUM === "1") throw e;
      env.skip = `headless Chromium cannot launch: ${String(e).split("\n")[0]}`;
    }
  });
  after(async () => {
    await env.browser?.close();
    await env.app?.close();
  });
  return env;
}
