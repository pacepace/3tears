/**
 * The lock screen, Control Center, headsets and media keys: what plays and the play/pause/stop
 * buttons, through the Media Session API.
 */

/** Media Session actions that seek or skip: meaningless on a live stream. */
const SEEK_ACTIONS = ["seekbackward", "seekforward", "seekto", "previoustrack", "nexttrack"];

/**
 * What plays (title, artist, album, artwork) and what play/pause/stop do. Handlers are installed
 * once; metadata is replaced as the item changes.
 *
 * For a live stream (`live: true`, the default) the seek and skip actions are cleared, so the
 * platform shows no scrubber or skip buttons, and no position is reported.
 *
 * Artwork: give absolute or same-origin URLs the platform can fetch by itself (iOS fetches them
 * outside the page), several sizes where there are any (`{ src, sizes: "512x512", type }`).
 */
export class MediaSessionControl {
  /**
   * @param {Navigator} [nav]
   * @param {{ onPlay?: () => unknown, onPause?: () => unknown, onStop?: () => unknown, live?: boolean,
   *           MediaMetadata?: typeof MediaMetadata }} [options]
   */
  constructor(nav = globalThis.navigator, options = {}) {
    /** @type {MediaSession | null} */
    this.session = nav?.mediaSession ?? null;
    this.Metadata = options.MediaMetadata ?? globalThis.MediaMetadata;
    /** @type {string | null} */
    this.key = null;
    if (!this.session) return;
    this.handle("play", options.onPlay);
    this.handle("pause", options.onPause);
    this.handle("stop", options.onStop);
    if (options.live ?? true) for (const action of SEEK_ACTIONS) this.handle(action, null);
  }

  /** Whether the platform has a Media Session at all. */
  get supported() {
    return this.session !== null;
  }

  /**
   * Installs (or, with null, clears) one action's handler; false where the platform lacks it. A
   * handler that throws, or returns a promise that rejects (`audio.play()` refused), is logged.
   * @param {string} action
   * @param {(() => unknown) | null | undefined} fn
   * @returns {boolean}
   */
  handle(action, fn) {
    if (!this.session) return false;
    const run = fn
      ? () => {
          Promise.resolve()
            .then(fn)
            .catch((e) => console.warn(`media session ${action}:`, e));
        }
      : null;
    try {
      this.session.setActionHandler(/** @type {MediaSessionAction} */ (action), run);
      return true;
    } catch {
      return false; // an action this platform does not know
    }
  }

  /**
   * What plays now. Unchanged metadata is not set again (setting it makes some platforms refetch
   * the artwork); metadata the platform refused is tried again next time.
   * @param {{ title: string, artist?: string, album?: string,
   *           artwork?: Array<{ src: string, sizes?: string, type?: string }> }} item
   */
  setNowPlaying({ title, artist = "", album = "", artwork = [] }) {
    if (!this.session || typeof this.Metadata !== "function") return;
    const key = JSON.stringify([title, artist, album, artwork]);
    if (key === this.key) return;
    this.session.metadata = new this.Metadata({ title, artist, album, artwork });
    this.key = key;
  }

  /**
   * Tells the platform whether it plays (the lock screen's play/pause button follows it).
   * @param {boolean} playing
   */
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
