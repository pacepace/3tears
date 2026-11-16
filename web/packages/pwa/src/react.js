/**
 * Thin React hooks over the framework-free core, for React apps. Each is optional: import only
 * `@3tears/pwa/react`'s hooks an app uses. React is a peer dependency of this entry only.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { forceReload, updatePrompt } from "./update.js";

/**
 * The update prompt's state, for a banner the app draws: `reload()` activates the update (calls
 * `apply`, e.g. Workbox's `() => updateServiceWorker(true)` or `registerWorker`'s `applyUpdate`);
 * `stalled` turns true when the page is still here `SW_RELOAD_TIMEOUT_MS` later, when the banner
 * should offer `forceReload()`. Never reloads on its own.
 * @param {() => unknown} apply
 * @param {{ timeoutMs?: number }} [options]
 * @returns {{ reload: () => void, reloading: boolean, stalled: boolean, forceReload: () => Promise<void> }}
 */
export function useUpdatePrompt(apply, options = {}) {
  const [state, setState] = useState(/** @type {"idle" | "reloading" | "stalled"} */ ("idle"));
  const applyRef = useRef(apply);
  applyRef.current = apply;
  // one prompt for the component's life, so its state never resets under a reload in progress
  const promptRef = useRef(/** @type {ReturnType<typeof updatePrompt> | null} */ (null));
  if (promptRef.current === null) {
    promptRef.current = updatePrompt(() => applyRef.current(), setState, { timeoutMs: options.timeoutMs });
  }
  useEffect(() => () => promptRef.current?.cancel(), []);
  const reload = useCallback(() => promptRef.current?.reload(), []);
  const force = useCallback(() => forceReload(), []);
  return { reload, reloading: state !== "idle", stalled: state === "stalled", forceReload: force };
}

/**
 * Calls `onMessage(data)` for each message the service worker posts this page (a tapped
 * notification asking it to open something), while `enabled`.
 * @param {(data: unknown) => void} onMessage
 * @param {boolean} [enabled]
 */
export function useServiceWorkerMessage(onMessage, enabled = true) {
  useEffect(() => {
    if (!enabled) return undefined;
    const worker = typeof navigator !== "undefined" ? navigator.serviceWorker : undefined;
    if (!worker) return undefined;
    const listener = (/** @type {MessageEvent} */ event) => onMessage(event.data);
    worker.addEventListener("message", listener);
    return () => worker.removeEventListener("message", listener);
  }, [onMessage, enabled]);
}
