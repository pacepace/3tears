/**
 * A Screen Wake Lock kept while wanted and the page is visible. The browser releases it when the
 * page is hidden; `sync()` (call it on visibilitychange) takes it again. Where the API is missing
 * or refuses (no secure context, a permissions policy, low battery) the screen follows the
 * device's own timeout.
 */
export class ScreenWakeLock {
  /**
   * @param {Navigator} [nav]
   * @param {Document} [doc]
   */
  constructor(nav = globalThis.navigator, doc = globalThis.document) {
    this.nav = nav;
    this.doc = doc;
    this.wanted = false;
    /** @type {WakeLockSentinel | null} */
    this.sentinel = null;
    /** @type {Promise<boolean> | null} */
    this.pending = null;
  }

  /** Whether the browser has the API. */
  get supported() {
    return typeof this.nav?.wakeLock?.request === "function";
  }

  /**
   * Wants the screen kept on, or not. Resolves whether it is held now.
   * @param {boolean} wanted
   * @returns {Promise<boolean>}
   */
  set(wanted) {
    this.wanted = wanted;
    return this.sync();
  }

  /**
   * Takes or releases the lock to match what is wanted and whether the page is visible.
   * @returns {Promise<boolean>}
   */
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
      await null; // the body runs after this.pending is assigned, so its finally clears the right one
      try {
        const s = await this.nav.wakeLock.request("screen");
        // Given up while the request ran: let it go at once.
        if (!this.wanted || this.doc.visibilityState !== "visible") {
          await s.release().catch(() => {});
          return false;
        }
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
