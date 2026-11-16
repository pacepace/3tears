/**
 * Asks the registration's worker script for a new version every `everyMs`. Returns a stop function.
 * @param {ServiceWorkerRegistration} registration
 * @param {number} [everyMs]
 * @param {{ setInterval: typeof setInterval, clearInterval: typeof clearInterval }} [timers]
 * @returns {() => void}
 */
export function checkForUpdates(registration: ServiceWorkerRegistration, everyMs?: number, timers?: {
    setInterval: typeof setInterval;
    clearInterval: typeof clearInterval;
}): () => void;
/**
 * Registers a service worker and reports a new version waiting.
 *
 * The first install is not an update (nothing to replace). A later one waits until the person
 * agrees: `applyUpdate()` posts SKIP_WAITING to it and reloads the page once it controls it.
 * @param {string} url the worker script
 * @param {{ scope?: string, type?: WorkerType, onUpdate?: () => void, checkEveryMs?: number | null,
 *   win?: Window & typeof globalThis }} [options]
 * @returns {Promise<{ registration: ServiceWorkerRegistration | null, applyUpdate: () => boolean, stop: () => void }>}
 */
export function registerWorker(url: string, options?: {
    scope?: string;
    type?: WorkerType;
    onUpdate?: () => void;
    checkEveryMs?: number | null;
    win?: Window & typeof globalThis;
}): Promise<{
    registration: ServiceWorkerRegistration | null;
    applyUpdate: () => boolean;
    stop: () => void;
}>;
/**
 * Polls a version file (`{"version": "..."}` or any JSON whose `field` names the build) and calls
 * `onUpdate(newVersion)` once when it differs from `current`. Fetched with `no-store`. Returns a
 * stop function.
 * @param {string} url
 * @param {string} current
 * @param {(version: string) => void} onUpdate
 * @param {{ everyMs?: number, field?: string, fetch?: typeof fetch,
 *   timers?: { setInterval: typeof setInterval, clearInterval: typeof clearInterval } }} [options]
 * @returns {{ check: () => Promise<string | null>, stop: () => void }}
 */
export function watchVersion(url: string, current: string, onUpdate: (version: string) => void, options?: {
    everyMs?: number;
    field?: string;
    fetch?: typeof fetch;
    timers?: {
        setInterval: typeof setInterval;
        clearInterval: typeof clearInterval;
    };
}): {
    check: () => Promise<string | null>;
    stop: () => void;
};
/**
 * The escape hatch: unregister every service worker, delete every cache, reload past the HTTP
 * cache. Each step is best effort; the reload always happens.
 * @param {Window & typeof globalThis} [win]
 * @returns {Promise<void>}
 */
export function forceReload(win?: Window & typeof globalThis): Promise<void>;
/**
 * The update prompt's state, for an app drawing its own: idle, then `reloading` after the
 * person's click, then `stalled` if the page is still here `SW_RELOAD_TIMEOUT_MS` later (offer
 * `forceReload`). The React hook in `@3tears/pwa/react` wraps it.
 * @param {() => unknown} apply what activates the update (`applyUpdate`, or Workbox's `updateServiceWorker(true)`)
 * @param {(state: "idle" | "reloading" | "stalled") => void} onState
 * @param {{ timeoutMs?: number, setTimeout?: typeof setTimeout }} [options]
 * @returns {{ reload: () => void, readonly state: "idle" | "reloading" | "stalled" }}
 */
export function updatePrompt(apply: () => unknown, onState: (state: "idle" | "reloading" | "stalled") => void, options?: {
    timeoutMs?: number;
    setTimeout?: typeof setTimeout;
}): {
    reload: () => void;
    readonly state: "idle" | "reloading" | "stalled";
};
/**
 * Updates, the page's half: a new version is shown as a prompt the app draws, never a reload
 * behind the person's back (a deploy mid-task must not throw away their work or interrupt what
 * plays).
 *
 * Two ways to learn of a new version, for two kinds of app:
 * - `registerWorker`: an app whose service worker caches its shell. A new worker installs and
 *   waits; `onUpdate` fires; `applyUpdate()` (the person's click) activates it and reloads once
 *   it controls the page.
 * - `watchVersion`: an app whose worker caches nothing (or that has none): poll a small version
 *   file and call `onUpdate` when it changes; the app's reload button reloads.
 * `forceReload` is the escape hatch when a worker never takes over (seen in installed apps).
 */
/** How often to look for a new version. */
export const UPDATE_CHECK_INTERVAL_MS: number;
/** How long `applyUpdate` waits for the new worker before the app should offer `forceReload`. */
export const SW_RELOAD_TIMEOUT_MS: 5000;
/** The message a page posts a waiting worker to activate it (Workbox's convention). */
export const SKIP_WAITING: "SKIP_WAITING";
