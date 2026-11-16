/**
 * @typedef {{
 *   clientsClaim: () => void,
 *   precacheAndRoute: (entries: unknown) => void,
 *   cleanupOutdatedCaches: () => void,
 *   createHandlerBoundToURL: (url: string) => unknown,
 *   NavigationRoute: new (handler: any, options: { denylist: RegExp[] }) => unknown,
 *   registerRoute: (match: any, handler?: any, method?: string) => unknown,
 *   NetworkFirst: new (options: { cacheName: string, networkTimeoutSeconds: number, plugins: unknown[] }) => unknown,
 *   ExpirationPlugin: new (options: { maxEntries: number, maxAgeSeconds: number }) => unknown,
 * }} WorkboxModules
 */
/**
 * @param {WorkboxModules} wb
 * @param {{ precache: unknown, shell?: string, neverShell?: readonly RegExp[], navigationCache: string,
 *   networkFirst?: Array<{ path: string, cacheName: string }>, timeoutSeconds?: number }} options
 */
export function appShell(wb: WorkboxModules, options: {
    precache: unknown;
    shell?: string;
    neverShell?: readonly RegExp[];
    navigationCache: string;
    networkFirst?: Array<{
        path: string;
        cacheName: string;
    }>;
    timeoutSeconds?: number;
}): void;
/**
 * The app-shell policy for a worker built with Workbox (vite-plugin-pwa's `injectManifest`), in
 * the order a generated worker runs it:
 *
 * 1. claim open pages once activated (so the page that asked for the update reloads into it);
 * 2. precache the build's files (hashed names, so cache-first is safe) and delete precaches of
 *    older builds;
 * 3. navigations get the precached `index.html`, except paths that are never the app's shell
 *    (`neverShell`: the API, sockets, health, sign-in and media);
 * 4. navigations and each `networkFirst` path go to the network first, the cache only when the
 *    network does not answer within `timeoutSeconds`, one entry for a day.
 *
 * Audio, the API and sign-in are never cached: only the precache list and the named paths are.
 * The Workbox modules are passed in, so the app's own Workbox (and its test doubles) are the ones
 * used; this package does not depend on Workbox.
 */
/** Paths that are never the app's shell. Add the app's own (media, uploads) to these. */
export const NEVER_SHELL: readonly RegExp[];
export type WorkboxModules = {
    clientsClaim: () => void;
    precacheAndRoute: (entries: unknown) => void;
    cleanupOutdatedCaches: () => void;
    createHandlerBoundToURL: (url: string) => unknown;
    NavigationRoute: new (handler: any, options: {
        denylist: RegExp[];
    }) => unknown;
    registerRoute: (match: any, handler?: any, method?: string) => unknown;
    NetworkFirst: new (options: {
        cacheName: string;
        networkTimeoutSeconds: number;
        plugins: unknown[];
    }) => unknown;
    ExpirationPlugin: new (options: {
        maxEntries: number;
        maxAgeSeconds: number;
    }) => unknown;
};
