/**
 * Web Push, the page's half: whether this browser can take it, subscribing with the server's
 * VAPID public key, and unsubscribing. The server's half (keys, storing subscriptions, sending)
 * and the app's own API calls stay with the app; see the worker half in `@3tears/web-pwa/worker`.
 *
 * On iPhone and iPad push works only in an installed (Home Screen) app.
 */
/**
 * Whether this browser can take push at all.
 * @param {Window} [win]
 * @returns {boolean}
 */
export function pushSupported(win?: Window): boolean;
/**
 * This page's service worker registration, or null when none runs (a dev server, or one not yet
 * installed).
 * @param {Window} [win]
 * @returns {Promise<ServiceWorkerRegistration | null>}
 */
export function workerRegistration(win?: Window): Promise<ServiceWorkerRegistration | null>;
/**
 * A VAPID public key (base64url) as the bytes `pushManager.subscribe` takes.
 * @param {string} base64url
 * @returns {Uint8Array<ArrayBuffer>}
 */
export function keyBytes(base64url: string): Uint8Array<ArrayBuffer>;
/**
 * Turns push on here: asks permission first, then (only when granted) gets the server's key and
 * subscribes, reusing a subscription this browser already holds.
 * @param {ServiceWorkerRegistration} registration
 * @param {() => Promise<string>} publicKey the server's VAPID public key, base64url, fetched only once permission is granted
 * @param {{ Notification?: { requestPermission(): Promise<NotificationPermission> } }} [env]
 * @returns {Promise<PushSubscription | "denied">}
 */
export function subscribePush(registration: ServiceWorkerRegistration, publicKey: () => Promise<string>, env?: {
    Notification?: {
        requestPermission(): Promise<NotificationPermission>;
    };
}): Promise<PushSubscription | "denied">;
/**
 * Turns push off here: the browser forgets its subscription. Resolves whether there was one.
 * @param {ServiceWorkerRegistration} registration
 * @returns {Promise<boolean>}
 */
export function unsubscribePush(registration: ServiceWorkerRegistration): Promise<boolean>;
