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
export function pushSupported(win = /** @type {Window} */ (/** @type {unknown} */ (globalThis))) {
  return typeof win !== "undefined" && !!win.navigator && "serviceWorker" in win.navigator && "PushManager" in win && "Notification" in win;
}

/**
 * This page's service worker registration, or null when none runs (a dev server, or one not yet
 * installed).
 * @param {Window} [win]
 * @returns {Promise<ServiceWorkerRegistration | null>}
 */
export async function workerRegistration(win = /** @type {Window} */ (/** @type {unknown} */ (globalThis))) {
  if (!pushSupported(win)) return null;
  return (await win.navigator.serviceWorker.getRegistration()) ?? null;
}

/**
 * A VAPID public key (base64url) as the bytes `pushManager.subscribe` takes.
 * @param {string} base64url
 * @returns {Uint8Array<ArrayBuffer>}
 */
export function keyBytes(base64url) {
  const padded = (base64url + "=".repeat((4 - (base64url.length % 4)) % 4)).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(padded);
  const bytes = new Uint8Array(new ArrayBuffer(raw.length));
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

/**
 * Turns push on here: asks permission first, then (only when granted) gets the server's key and
 * subscribes, reusing a subscription this browser already holds.
 * @param {ServiceWorkerRegistration} registration
 * @param {() => Promise<string>} publicKey the server's VAPID public key, base64url, fetched only once permission is granted
 * @param {{ Notification?: { requestPermission(): Promise<NotificationPermission> } }} [env]
 * @returns {Promise<PushSubscription | "denied">}
 */
export async function subscribePush(registration, publicKey, env = {}) {
  const N = env.Notification ?? globalThis.Notification;
  const permission = await N.requestPermission();
  if (permission !== "granted") return "denied";
  const key = await publicKey();
  return (
    (await registration.pushManager.getSubscription()) ??
    (await registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: keyBytes(key) }))
  );
}

/**
 * Turns push off here: the browser forgets its subscription. Resolves whether there was one.
 * @param {ServiceWorkerRegistration} registration
 * @returns {Promise<boolean>}
 */
export async function unsubscribePush(registration) {
  const subscription = await registration.pushManager.getSubscription();
  if (!subscription) return false;
  await subscription.unsubscribe();
  return true;
}
