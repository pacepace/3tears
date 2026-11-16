// A minimal app built on @3tears/web-pwa with no build step: plain modules served beside the app,
// the shell worker caching only the app's own files, an update prompt it draws itself, the
// display mode on <html>, and the lock screen's now-playing through Media Session.
import { MediaSessionControl, markDisplayMode, registerWorker } from "./pwa/index.js";

export const VERSION = "1";
document.getElementById("version").textContent = `version ${VERSION}`;

markDisplayMode(document, window);
document.getElementById("mode").textContent = document.documentElement.dataset.displayMode;

let playing = false;
const media = new MediaSessionControl(navigator, {
  onPlay: () => setPlaying(true),
  onPause: () => setPlaying(false),
  onStop: () => setPlaying(false),
});
function setPlaying(on) {
  playing = on;
  media.setPlaying(on);
  document.getElementById("play").textContent = on ? "Pause" : "Play";
}
media.setNowPlaying({ title: "Example", album: "Example", artwork: [{ src: "/app/icon-512.png", sizes: "512x512", type: "image/png" }] });
document.getElementById("play").addEventListener("click", () => setPlaying(!playing));

export const worker = registerWorker("/app/pwa/shell-worker.js?config=/app/shell.js", {
  scope: "/app/",
  checkEveryMs: null,
  onUpdate: () => {
    document.getElementById("update").hidden = false;
  },
}).then((w) => {
  document.getElementById("reload").addEventListener("click", () => w.applyUpdate());
  return w;
});
window.exampleWorker = worker; // read by the tests
