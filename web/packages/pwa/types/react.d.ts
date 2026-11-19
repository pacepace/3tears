/**
 * The update prompt's state, for a banner the app draws: `reload()` activates the update (calls
 * `apply`, e.g. Workbox's `() => updateServiceWorker(true)` or `registerWorker`'s `applyUpdate`);
 * `stalled` turns true when the page is still here `SW_RELOAD_TIMEOUT_MS` later, when the banner
 * should offer `forceReload()`. Never reloads on its own.
 * @param {() => unknown} apply
 * @param {{ timeoutMs?: number }} [options]
 * @returns {{ reload: () => void, reloading: boolean, stalled: boolean, forceReload: () => Promise<void> }}
 */
export function useUpdatePrompt(apply: () => unknown, options?: {
    timeoutMs?: number;
}): {
    reload: () => void;
    reloading: boolean;
    stalled: boolean;
    forceReload: () => Promise<void>;
};
/**
 * Calls `onMessage(data)` for each message the service worker posts this page (a tapped
 * notification asking it to open something), while `enabled`.
 * @param {(data: unknown) => void} onMessage
 * @param {boolean} [enabled]
 */
export function useServiceWorkerMessage(onMessage: (data: unknown) => void, enabled?: boolean): void;
