// 3tears-pwa browser helpers (ES module, no dependencies). Served by the app beside its own
// scripts: `python -m threetears.pwa export-static <dir>` copies this file there.
//
// Every helper takes the globals it uses (window, navigator, document) as arguments, so it runs
// under test with fakes, and does nothing where the browser lacks the API.

/** The display modes a page can be shown in, most app-like first. */
export const DISPLAY_MODES = Object.freeze(["fullscreen", "standalone", "minimal-ui", "window-controls-overlay", "browser"]);

/**
 * How the page is shown now: "standalone" (installed, its own window), "fullscreen", "minimal-ui",
 * "window-controls-overlay", or "browser" (a tab). From the `display-mode` media query, which
 * Safari, Chrome, Edge and Firefox all answer.
 * @param {Window} win
 * @returns {string}
 */
export function displayMode(win = globalThis) {
  if (typeof win?.matchMedia !== "function") return "browser";
  for (const mode of DISPLAY_MODES) {
    if (mode !== "browser" && win.matchMedia(`(display-mode: ${mode})`).matches) return mode;
  }
  return "browser";
}

/** Whether the page runs as an installed app (any display mode but "browser"). */
export function isInstalled(win = globalThis) {
  return displayMode(win) !== "browser";
}

/**
 * Calls `onChange(mode)` whenever the display mode changes (a desktop app popped out to a tab, a
 * page entering fullscreen). Returns a function that stops watching.
 * @param {Window} win
 * @param {(mode: string) => void} onChange
 */
export function watchDisplayMode(win, onChange) {
  if (typeof win?.matchMedia !== "function") return () => {};
  let last = displayMode(win);
  const check = () => {
    const now = displayMode(win);
    if (now !== last) { last = now; onChange(now); }
  };
  const lists = DISPLAY_MODES.filter((m) => m !== "browser").map((m) => win.matchMedia(`(display-mode: ${m})`));
  for (const l of lists) l.addEventListener?.("change", check);
  return () => { for (const l of lists) l.removeEventListener?.("change", check); };
}

/**
 * Keeps `<html data-display-mode="...">` set to the display mode, for CSS that differs installed.
 * Returns a function that stops it.
 * @param {Document} doc
 * @param {Window} win
 */
export function markDisplayMode(doc = globalThis.document, win = globalThis) {
  const root = doc.documentElement;
  root.dataset.displayMode = displayMode(win);
  return watchDisplayMode(win, (mode) => { root.dataset.displayMode = mode; });
}

/** Media Session actions that seek or skip: meaningless on a live stream. */
const SEEK_ACTIONS = ["seekbackward", "seekforward", "seekto", "previoustrack", "nexttrack"];

/**
 * The lock screen, Control Center and media keys: what plays (title, artist, album, artwork) and
 * play/pause/stop. Handlers are installed once; metadata is replaced as the track changes.
 *
 * For a live stream (`live: true`, the default) the seek and skip actions are cleared, so the
 * platform shows no scrubber or skip buttons, and no position is reported.
 *
 * Artwork: give absolute or same-origin URLs the platform can fetch by itself (iOS fetches them
 * outside the page), several sizes where there are any (`{ src, sizes: "512x512", type }`).
 */
export class MediaSessionControl {
  /**
   * @param {Navigator} nav
   * @param {{ onPlay?: () => any, onPause?: () => any, onStop?: () => any, live?: boolean,
   *           MediaMetadata?: typeof MediaMetadata }} [options]
   */
  constructor(nav = globalThis.navigator, options = {}) {
    this.session = nav?.mediaSession ?? null;
    this.Metadata = options.MediaMetadata ?? globalThis.MediaMetadata;
    this.key = null;
    if (!this.session) return;
    this.handle("play", options.onPlay);
    this.handle("pause", options.onPause);
    this.handle("stop", options.onStop);
    if (options.live ?? true) for (const action of SEEK_ACTIONS) this.handle(action, null);
  }

  get supported() { return this.session !== null; }

  /** Installs (or, with null, clears) one action's handler; false where the platform lacks it. */
  handle(action, fn) {
    if (!this.session) return false;
    try {
      // a handler may throw or return a promise that rejects (audio.play() refused): both are logged
      const run = fn ? () => { Promise.resolve().then(fn).catch((e) => console.warn(`media session ${action}:`, e)); } : null;
      this.session.setActionHandler(action, run);
      return true;
    } catch {
      return false; // an action this platform does not know
    }
  }

  /**
   * What plays now. Unchanged metadata is not set again (setting it makes some platforms refetch
   * the artwork).
   * @param {{ title: string, artist?: string, album?: string,
   *           artwork?: Array<{ src: string, sizes?: string, type?: string }> }} item
   */
  setNowPlaying({ title, artist = "", album = "", artwork = [] }) {
    if (!this.session || typeof this.Metadata !== "function") return;
    const key = JSON.stringify([title, artist, album, artwork]);
    if (key === this.key) return;
    this.session.metadata = new this.Metadata({ title, artist, album, artwork });
    this.key = key; // only once set: metadata the platform refused is tried again next time
  }

  /** Tells the platform whether it plays (the lock screen's play/pause button follows it). */
  setPlaying(playing) {
    if (this.session) this.session.playbackState = playing ? "playing" : "paused";
  }

  /** Nothing plays any more: the lock screen forgets it. */
  clear() {
    if (!this.session) return;
    this.key = null;
    this.session.metadata = null;
    this.session.playbackState = "none";
  }
}

/**
 * A Screen Wake Lock kept while wanted and the page is visible. The browser releases it when the
 * page is hidden; `sync()` (call it on visibilitychange) takes it again. Where the API is missing
 * or refuses (no secure context, a permissions policy, low battery) the screen follows the
 * device's own timeout.
 */
export class ScreenWakeLock {
  constructor(nav = globalThis.navigator, doc = globalThis.document) {
    this.nav = nav;
    this.doc = doc;
    this.wanted = false;
    this.sentinel = null;
    this.pending = null;
  }

  get supported() { return typeof this.nav?.wakeLock?.request === "function"; }

  /** Wants the screen kept on, or not. Resolves whether it is held now. */
  set(wanted) {
    this.wanted = wanted;
    return this.sync();
  }

  /** Takes or releases the lock to match what is wanted and whether the page is visible. */
  async sync() {
    if (!this.supported) return false;
    const want = this.wanted && this.doc.visibilityState === "visible";
    if (!want) {
      const s = this.sentinel;
      this.sentinel = null;
      if (s && !s.released) await s.release().catch(() => {});
      return false;
    }
    if (this.sentinel && !this.sentinel.released) return true;
    if (this.pending) return this.pending;
    this.pending = (async () => {
      await null; // run the body after this.pending is assigned, so its finally clears the right one
      try {
        const s = await this.nav.wakeLock.request("screen");
        // Given up while the request ran: let it go at once.
        if (!this.wanted || this.doc.visibilityState !== "visible") { await s.release().catch(() => {}); return false; }
        this.sentinel = s;
        return true;
      } catch {
        return false;
      } finally {
        this.pending = null;
      }
    })();
    return this.pending;
  }
}
