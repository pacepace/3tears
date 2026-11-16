/**
 * Options for vite-plugin-pwa (`VitePWA(vitePwa({...}))`) carrying the 3tears policy, for a Vite
 * app. Optional: an app without Vite never imports this entry, and it imports nothing itself.
 *
 * - `registerType: "prompt"`: a new version waits for the person (`useUpdatePrompt`), never an
 *   automatic reload of a page in use.
 * - `strategies: "injectManifest"`: the app owns its worker (`src/sw.ts`), which uses
 *   `@3tears/pwa/workbox`'s `appShell` and `@3tears/pwa/worker`'s push and tap helpers.
 * - `manifest`: checked by `appManifest`.
 * - `runtimeFiles`: files the server rewrites at run time (a runtime config), kept out of the
 *   precache so a stale copy is never served cache-first.
 */
import { appManifest } from "./manifest.js";

/**
 * @param {{ manifest: import("./manifest.js").Manifest, srcDir?: string, filename?: string,
 *   runtimeFiles?: string[], registerType?: "prompt" | "autoUpdate", [key: string]: unknown }} options
 * @returns {Record<string, unknown>}
 */
export function vitePwa({ manifest, srcDir = "src", filename = "sw.ts", runtimeFiles = [], registerType = "prompt", ...rest }) {
  return {
    registerType,
    injectRegister: "auto",
    manifest: appManifest(manifest),
    strategies: "injectManifest",
    srcDir,
    filename,
    injectManifest: { globIgnores: runtimeFiles.map((f) => `**/${f}`) },
    ...rest,
  };
}
