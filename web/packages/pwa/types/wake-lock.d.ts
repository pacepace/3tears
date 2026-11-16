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
    constructor(nav?: Navigator, doc?: Document);
    nav: Navigator;
    doc: Document;
    wanted: boolean;
    /** @type {WakeLockSentinel | null} */
    sentinel: WakeLockSentinel | null;
    /** @type {Promise<boolean> | null} */
    pending: Promise<boolean> | null;
    /** Whether the browser has the API. */
    get supported(): boolean;
    /**
     * Wants the screen kept on, or not. Resolves whether it is held now.
     * @param {boolean} wanted
     * @returns {Promise<boolean>}
     */
    set(wanted: boolean): Promise<boolean>;
    /**
     * Takes or releases the lock to match what is wanted and whether the page is visible.
     * @returns {Promise<boolean>}
     */
    sync(): Promise<boolean>;
}
