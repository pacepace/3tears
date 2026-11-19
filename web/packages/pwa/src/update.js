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
export const UPDATE_CHECK_INTERVAL_MS = 5 * 60 * 1000;
/** `watchForUpdates` asks the browser at most this often, however many reasons arrive at once. */
export const UPDATE_CHECK_MIN_GAP_MS = 2_000;
/** How long `applyUpdate` waits for the new worker before the app should offer `forceReload`. */
export const SW_RELOAD_TIMEOUT_MS = 5_000;
/** The message a page posts a waiting worker to activate it (Workbox's convention). */
export const SKIP_WAITING = "SKIP_WAITING";

/**
 * Asks the registration's worker script for a new version every `everyMs`. Returns a stop function.
 * @param {ServiceWorkerRegistration} registration
 * @param {number} [everyMs]
 * @param {{ setInterval: typeof setInterval, clearInterval: typeof clearInterval }} [timers]
 * @returns {() => void}
 */
export function checkForUpdates(registration, everyMs = UPDATE_CHECK_INTERVAL_MS, timers = globalThis) {
  const id = timers.setInterval(() => {
    registration.update().catch(() => {});
  }, everyMs);
  return () => timers.clearInterval(id);
}

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
export function watchForUpdates(registration, options = {}) {
  const timers = options.timers ?? globalThis;
  const now = options.now ?? (() => Date.now());
  const gap = options.minGapMs ?? UPDATE_CHECK_MIN_GAP_MS;
  const doc = options.onVisible === false ? null : (options.win ?? globalThis).document;
  let last = -Infinity;
  let checks = 0;
  let stopped = false;
  const ask = () => {
    last = now();
    checks++;
    // a failed look (offline, the server away) is no news: the next reason looks again
    registration.update().catch(() => {});
  };
  const check = () => {
    if (stopped || now() - last < gap) return false;
    ask();
    return true;
  };
  const everyMs = options.everyMs === undefined ? UPDATE_CHECK_INTERVAL_MS : options.everyMs;
  // the timer always asks: it is the floor, minutes apart; the gap only merges bursts of reasons
  const id = everyMs === null ? null : timers.setInterval(ask, everyMs);
  const onVisible = () => {
    if (doc?.visibilityState !== "hidden") check();
  };
  doc?.addEventListener("visibilitychange", onVisible);
  return {
    check,
    stop() {
      stopped = true;
      if (id !== null) timers.clearInterval(id);
      doc?.removeEventListener("visibilitychange", onVisible);
    },
    get checks() {
      return checks;
    },
  };
}

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
export async function registerWorker(url, options = {}) {
  const win = options.win ?? /** @type {Window & typeof globalThis} */ (globalThis);
  const container = win.navigator?.serviceWorker;
  if (!container) return { registration: null, applyUpdate: () => false, check: () => false, stop: () => {} };
  const registration = await container.register(url, {
    ...(options.scope ? { scope: options.scope } : {}),
    ...(options.type ? { type: options.type } : {}),
  });
  let notified = false;
  const report = () => {
    if (registration.waiting && container.controller && !notified) {
      notified = true;
      options.onUpdate?.();
    }
  };
  const watch = (/** @type {ServiceWorker | null} */ worker) => {
    worker?.addEventListener("statechange", () => {
      if (worker.state === "installed") report();
    });
  };
  registration.addEventListener("updatefound", () => watch(registration.installing));
  watch(registration.installing);
  report();
  const checks = watchForUpdates(registration, {
    everyMs: options.checkEveryMs === null ? null : (options.checkEveryMs ?? UPDATE_CHECK_INTERVAL_MS),
    onVisible: options.checkOnVisible !== false,
    win,
    timers: win,
  });
  let reloading = false;
  const onControllerChange = () => {
    if (reloading) win.location.reload();
  };
  container.addEventListener("controllerchange", onControllerChange);
  return {
    registration,
    check: checks.check,
    applyUpdate() {
      const waiting = registration.waiting;
      if (!waiting) return false;
      reloading = true;
      waiting.postMessage({ type: SKIP_WAITING });
      return true;
    },
    stop() {
      checks.stop();
      container.removeEventListener("controllerchange", onControllerChange);
    },
  };
}

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
export function watchVersion(url, current, onUpdate, options = {}) {
  const get = options.fetch ?? globalThis.fetch.bind(globalThis);
  const timers = options.timers ?? globalThis;
  const field = options.field ?? "version";
  let done = false;
  const check = async () => {
    if (done) return null;
    try {
      const busted = `${url}${url.includes("?") ? "&" : "?"}ts=${Date.now()}`;
      const r = await get(busted, { cache: "no-store" });
      if (!r.ok) return null;
      const seen = String((await r.json())?.[field] ?? "");
      // two looks in flight at once (focus while a timer's look runs) report it once
      if (seen && seen !== current && !done) {
        done = true;
        onUpdate(seen);
      }
      return seen || null;
    } catch {
      return null;
    }
  };
  const id = timers.setInterval(() => void check(), options.everyMs ?? UPDATE_CHECK_INTERVAL_MS);
  const win = options.win;
  const onVisible = () => {
    if (win?.document?.visibilityState !== "hidden") void check();
  };
  win?.addEventListener("focus", onVisible);
  win?.document?.addEventListener("visibilitychange", onVisible);
  void check();
  return {
    check,
    stop() {
      done = true;
      timers.clearInterval(id);
      win?.removeEventListener("focus", onVisible);
      win?.document?.removeEventListener("visibilitychange", onVisible);
    },
  };
}

/**
 * The escape hatch: unregister every service worker, delete every cache, reload past the HTTP
 * cache. Each step is best effort; the reload always happens.
 * @param {Window & typeof globalThis} [win]
 * @returns {Promise<void>}
 */
export async function forceReload(win = /** @type {Window & typeof globalThis} */ (globalThis)) {
  try {
    if (win.navigator.serviceWorker) {
      const regs = await win.navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map((reg) => reg.unregister()));
    }
  } catch {
    // the reload below is the escape hatch
  }
  try {
    if ("caches" in win) {
      const names = await win.caches.keys();
      await Promise.all(names.map((n) => win.caches.delete(n)));
    }
  } catch {
    // the reload below is the escape hatch
  }
  // Some browsers still honour the old boolean (bypass the HTTP cache); the rest ignore it.
  /** @type {{ reload: (force?: boolean) => void }} */ (/** @type {unknown} */ (win.location)).reload(true);
}

/**
 * The update prompt's state, for an app drawing its own: idle, then `reloading` after the
 * person's click, then `stalled` if the page is still here `SW_RELOAD_TIMEOUT_MS` later (offer
 * `forceReload`). The React hook in `@3tears/web-pwa/react` wraps it.
 * @param {() => unknown} apply what activates the update (`applyUpdate`, or Workbox's `updateServiceWorker(true)`)
 * @param {(state: "idle" | "reloading" | "stalled") => void} onState
 * @param {{ timeoutMs?: number, setTimeout?: typeof setTimeout, clearTimeout?: typeof clearTimeout }} [options]
 * @returns {{ reload: () => void, cancel: () => void, readonly state: "idle" | "reloading" | "stalled" }}
 */
export function updatePrompt(apply, onState, options = {}) {
  /** @type {"idle" | "reloading" | "stalled"} */
  let state = "idle";
  const later = options.setTimeout ?? globalThis.setTimeout;
  const clear = options.clearTimeout ?? globalThis.clearTimeout;
  /** @type {ReturnType<typeof setTimeout> | null} */
  let timer = null;
  const set = (/** @type {"idle" | "reloading" | "stalled"} */ s) => {
    state = s;
    onState(s);
  };
  return {
    reload() {
      if (state !== "idle") return;
      set("reloading");
      void apply();
      timer = later(() => {
        timer = null;
        if (state === "reloading") set("stalled");
      }, options.timeoutMs ?? SW_RELOAD_TIMEOUT_MS);
    },
    cancel() {
      if (timer !== null) clear(timer);
      timer = null;
    },
    get state() {
      return state;
    },
  };
}
