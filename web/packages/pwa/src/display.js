/**
 * How the page is shown: in a browser tab, or as an installed app.
 *
 * Every function takes the window it reads, so it runs under test with a stand-in, and answers
 * "browser" where the window has no `matchMedia`.
 */

/** The display modes a page can be shown in, most app-like first. */
export const DISPLAY_MODES = Object.freeze(["fullscreen", "standalone", "minimal-ui", "window-controls-overlay", "browser"]);

/**
 * How the page is shown now: "fullscreen", "standalone" (installed, its own window), "minimal-ui",
 * "window-controls-overlay", or "browser" (a tab), from the `display-mode` media query.
 * @param {Window} [win]
 * @returns {string}
 */
export function displayMode(win = /** @type {Window} */ (/** @type {unknown} */ (globalThis))) {
  if (typeof win?.matchMedia !== "function") return "browser";
  for (const mode of DISPLAY_MODES) {
    if (mode !== "browser" && win.matchMedia(`(display-mode: ${mode})`).matches) return mode;
  }
  return "browser";
}

/**
 * Whether the page runs as an installed app in its own standalone window: the
 * `display-mode: standalone` query, or iOS's `navigator.standalone` (a page opened from the Home
 * Screen). Installed windows treat `window.open` and new tabs differently from a browser tab.
 * @param {Window} [win]
 * @returns {boolean}
 */
export function isStandalone(win = /** @type {Window} */ (/** @type {unknown} */ (globalThis))) {
  const nav = /** @type {Navigator & { standalone?: boolean }} */ (win?.navigator);
  return win?.matchMedia?.("(display-mode: standalone)").matches === true || nav?.standalone === true;
}

/**
 * Calls `onChange(mode)` whenever the display mode changes (a desktop app popped out to a tab, a
 * page entering fullscreen). Returns a function that stops watching.
 * @param {Window} win
 * @param {(mode: string) => void} onChange
 * @returns {() => void}
 */
export function watchDisplayMode(win, onChange) {
  if (typeof win?.matchMedia !== "function") return () => {};
  let last = displayMode(win);
  const check = () => {
    const now = displayMode(win);
    if (now !== last) {
      last = now;
      onChange(now);
    }
  };
  const lists = DISPLAY_MODES.filter((m) => m !== "browser").map((m) => win.matchMedia(`(display-mode: ${m})`));
  for (const l of lists) l.addEventListener?.("change", check);
  return () => {
    for (const l of lists) l.removeEventListener?.("change", check);
  };
}

/**
 * Keeps `<html data-display-mode="...">` set to the display mode, for CSS that differs once
 * installed. Returns a function that stops it.
 * @param {Document} [doc]
 * @param {Window} [win]
 * @returns {() => void}
 */
export function markDisplayMode(doc = globalThis.document, win = /** @type {Window} */ (/** @type {unknown} */ (globalThis))) {
  const root = doc.documentElement;
  root.dataset.displayMode = displayMode(win);
  return watchDisplayMode(win, (mode) => {
    root.dataset.displayMode = mode;
  });
}
