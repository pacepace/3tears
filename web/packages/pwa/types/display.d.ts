/**
 * How the page is shown now: "fullscreen", "standalone" (installed, its own window), "minimal-ui",
 * "window-controls-overlay", or "browser" (a tab), from the `display-mode` media query.
 * @param {Window} [win]
 * @returns {string}
 */
export function displayMode(win?: Window): string;
/**
 * Whether the page runs as an installed app in its own standalone window: the
 * `display-mode: standalone` query, or iOS's `navigator.standalone` (a page opened from the Home
 * Screen). Installed windows treat `window.open` and new tabs differently from a browser tab.
 * @param {Window} [win]
 * @returns {boolean}
 */
export function isStandalone(win?: Window): boolean;
/**
 * Calls `onChange(mode)` whenever the display mode changes (a desktop app popped out to a tab, a
 * page entering fullscreen). Returns a function that stops watching.
 * @param {Window} win
 * @param {(mode: string) => void} onChange
 * @returns {() => void}
 */
export function watchDisplayMode(win: Window, onChange: (mode: string) => void): () => void;
/**
 * Keeps `<html data-display-mode="...">` set to the display mode, for CSS that differs once
 * installed. Returns a function that stops it.
 * @param {Document} [doc]
 * @param {Window} [win]
 * @returns {() => void}
 */
export function markDisplayMode(doc?: Document, win?: Window): () => void;
/**
 * How the page is shown: in a browser tab, or as an installed app.
 *
 * Every function takes the window it reads, so it runs under test with a stand-in, and answers
 * "browser" where the window has no `matchMedia`.
 */
/** The display modes a page can be shown in, most app-like first. */
export const DISPLAY_MODES: readonly string[];
