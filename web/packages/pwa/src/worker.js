/**
 * The service worker's half, for a worker built as a module (a bundler's, or `type: "module"`):
 * the update protocol, showing pushes and routing a tap on a notification. No DOM, no Workbox.
 * Every function takes the worker's global scope (`self`) so it runs under test with a stand-in.
 *
 * A worker with no build step uses `shell-worker.js` instead, which does the update protocol
 * and the shell cache on its own.
 */

/** The message a page posts a waiting worker to activate it (Workbox's convention). */
export const SKIP_WAITING = "SKIP_WAITING";

/**
 * Activates on the page's SKIP_WAITING (the person agreed to the update) and never on its own,
 * so a deploy never swaps the code under a running page.
 * @param {ServiceWorkerGlobalScope} self
 */
export function listenForSkipWaiting(self) {
  self.addEventListener("message", (event) => {
    if (event.data && event.data.type === SKIP_WAITING) void self.skipWaiting();
  });
}

/**
 * A push's data: its JSON, else its text, else null.
 * @param {PushEvent} event
 * @returns {unknown}
 */
export function readPushData(event) {
  try {
    return event.data?.json() ?? null;
  } catch {
    return event.data?.text() ?? null;
  }
}

/**
 * Shows every push as a notification. A browser revokes push from a worker that receives one and
 * shows nothing, so even a push the app cannot read is shown (`toNotification` decides how).
 * @param {ServiceWorkerGlobalScope} self
 * @param {(data: unknown) => { title: string, options?: NotificationOptions }} toNotification
 */
export function showPushes(self, toNotification) {
  self.addEventListener("push", (event) => {
    const { title, options } = toNotification(readPushData(event));
    event.waitUntil(self.registration.showNotification(title, options));
  });
}

/**
 * Where a tapped notification goes. An open window of the app is brought forward and, when the
 * tap names a message, sent it (the page opens the thing in place); with none open, the app
 * starts at `url`.
 * @param {ServiceWorkerGlobalScope} self
 * @param {(data: unknown) => { url: string, message?: unknown }} route from the notification's data
 */
export function openOnNotificationClick(self, route) {
  self.addEventListener("notificationclick", (event) => {
    event.notification.close();
    const { url, message } = route(event.notification.data);
    event.waitUntil(
      (async () => {
        const windows = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
        const open = windows.find((w) => new URL(w.url).origin === self.location.origin);
        if (open) {
          await open.focus();
          if (message !== undefined && message !== null) open.postMessage(message);
          return;
        }
        await self.clients.openWindow(new URL(url, self.location.origin).href);
      })(),
    );
  });
}
