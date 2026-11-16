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
 * Asks the browser to look for a new version of the registration's worker (`registration.update()`)
 * on an interval, when the page comes back into view, and whenever the app calls `check()`.
 *
 * An open page never navigates, and a browser looks for a new worker only on a navigation (and
 * then about daily), so a page left open for hours, or an installed app brought back from the
 * background (where a phone stops the page's timers), would not learn of a deploy. Call `check()`
 * when the app knows one is likely: its server says it restarted, or runs another version. Nothing
 * here reloads: a new worker installs and waits, and `registerWorker`'s `onUpdate` fires.
 * `check()` and returning to view ask at most once per `minGapMs` (several reasons at once are one
 * look); the timer always asks.
 * @param {{ update: () => Promise<unknown> }} registration
 * @param {{ everyMs?: number | null, onVisible?: boolean, minGapMs?: number,
 *   win?: { document?: { visibilityState?: string, addEventListener: (type: string, listener: () => void) => void,
 *     removeEventListener: (type: string, listener: () => void) => void } },
 *   timers?: { setInterval: typeof setInterval, clearInterval: typeof clearInterval },
 *   now?: () => number }} [options] `everyMs` null: no interval; `onVisible` false: not on return
 * @returns {{ check: () => boolean, stop: () => void, readonly checks: number }} `check()` says
 *   whether it asked (false within `minGapMs` of the last ask, or after `stop()`)
 */
export function watchForUpdates(registration: {
    update: () => Promise<unknown>;
}, options?: {
    everyMs?: number | null;
    onVisible?: boolean;
    minGapMs?: number;
    win?: {
        document?: {
            visibilityState?: string;
            addEventListener: (type: string, listener: () => void) => void;
            removeEventListener: (type: string, listener: () => void) => void;
        };
    };
    timers?: {
        setInterval: typeof setInterval;
        clearInterval: typeof clearInterval;
    };
    now?: () => number;
}): {
    check: () => boolean;
    stop: () => void;
    readonly checks: number;
};
/**
 * Registers a service worker and reports a new version waiting.
 *
 * The first install is not an update (nothing to replace). A later one waits until the person
 * agrees: `applyUpdate()` posts SKIP_WAITING to it and reloads the page once it controls it.
 * The browser is asked to look for one (`watchForUpdates`) every `checkEveryMs` (null: never on a
 * timer), when the page comes back into view (unless `checkOnVisible` is false), and on
 * `check()`, which the app calls when it knows a deploy is likely.
 * @param {string} url the worker script
 * @param {{ scope?: string, type?: WorkerType, onUpdate?: () => void, checkEveryMs?: number | null,
 *   checkOnVisible?: boolean, win?: Window & typeof globalThis }} [options]
 * @returns {Promise<{ registration: ServiceWorkerRegistration | null, applyUpdate: () => boolean,
 *   check: () => boolean, stop: () => void }>}
 */
export function registerWorker(url: string, options?: {
    scope?: string;
    type?: WorkerType;
    onUpdate?: () => void;
    checkEveryMs?: number | null;
    checkOnVisible?: boolean;
    win?: Window & typeof globalThis;
}): Promise<{
    registration: ServiceWorkerRegistration | null;
    applyUpdate: () => boolean;
    check: () => boolean;
    stop: () => void;
}>;
/**
 * Watches a version file (`{"version": "..."}` or any JSON whose `field` names the build) and calls
 * `onUpdate(newVersion)` once when it differs from `current`. Looks at once, every `everyMs`, and,
 * given `win`, whenever the page comes back into view or gets focus (when a person returns to an
 * installed app). Fetched with `no-store` and a `ts` query, so no cache in between answers.
 * @param {string} url
 * @param {string} current
 * @param {(version: string) => void} onUpdate
 * @param {{ everyMs?: number, field?: string, fetch?: typeof fetch, win?: Window,
 *   timers?: { setInterval: typeof setInterval, clearInterval: typeof clearInterval } }} [options]
 * @returns {{ check: () => Promise<string | null>, stop: () => void }}
 */
export function watchVersion(url: string, current: string, onUpdate: (version: string) => void, options?: {
    everyMs?: number;
    field?: string;
    fetch?: typeof fetch;
    win?: Window;
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
 * `forceReload`). The React hook in `@3tears/web-pwa/react` wraps it.
 * @param {() => unknown} apply what activates the update (`applyUpdate`, or Workbox's `updateServiceWorker(true)`)
 * @param {(state: "idle" | "reloading" | "stalled") => void} onState
 * @param {{ timeoutMs?: number, setTimeout?: typeof setTimeout, clearTimeout?: typeof clearTimeout }} [options]
 * @returns {{ reload: () => void, cancel: () => void, readonly state: "idle" | "reloading" | "stalled" }}
 */
export function updatePrompt(apply: () => unknown, onState: (state: "idle" | "reloading" | "stalled") => void, options?: {
    timeoutMs?: number;
    setTimeout?: typeof setTimeout;
    clearTimeout?: typeof clearTimeout;
}): {
    reload: () => void;
    cancel: () => void;
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
 * `watchForUpdates` is how `registerWorker` asks: on a timer, on return to view, and on demand.
 * `forceReload` is the escape hatch when a worker never takes over (seen in installed apps).
 */
/** How often to look for a new version. */
export const UPDATE_CHECK_INTERVAL_MS: number;
/** `watchForUpdates` asks the browser at most this often, however many reasons arrive at once. */
export const UPDATE_CHECK_MIN_GAP_MS: 2000;
/** How long `applyUpdate` waits for the new worker before the app should offer `forceReload`. */
export const SW_RELOAD_TIMEOUT_MS: 5000;
/** The message a page posts a waiting worker to activate it (Workbox's convention). */
export const SKIP_WAITING: "SKIP_WAITING";
