// The package @3tears/web-pwa: installable web apps for the 3tears family. Every feature is its own
// module and opt-in; this entry re-exports the page-side ones (no framework, no worker code). The
// README covers each entry point: "/worker", "/workbox", "/react", "/vite", "shell-worker.js" and
// "safe-area.css".
export { DISPLAY_MODES, displayMode, isStandalone, markDisplayMode, watchDisplayMode } from "./display.js";
export { MANIFEST_MEDIA_TYPE, VIEWPORT, appManifest, headTags, installId } from "./manifest.js";
export { MediaSessionControl } from "./media-session.js";
export { keyBytes, pushSupported, subscribePush, unsubscribePush, workerRegistration } from "./push.js";
export {
  SKIP_WAITING,
  SW_RELOAD_TIMEOUT_MS,
  UPDATE_CHECK_INTERVAL_MS,
  checkForUpdates,
  forceReload,
  registerWorker,
  updatePrompt,
  watchVersion,
} from "./update.js";
export { ScreenWakeLock } from "./wake-lock.js";
