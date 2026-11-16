import test from "node:test";
import assert from "node:assert/strict";
import { displayMode, isStandalone, markDisplayMode, watchDisplayMode } from "../src/display.js";

function fakeWindow(mode, standalone) {
  const lists = [];
  const win = {
    navigator: standalone === undefined ? {} : { standalone },
    matchMedia(q) {
      const l = { q, fns: [], get matches() { return q === `(display-mode: ${win.mode})`; },
        addEventListener(_, f) { this.fns.push(f); }, removeEventListener(_, f) { this.fns = this.fns.filter((x) => x !== f); } };
      lists.push(l);
      return l;
    },
    mode,
    fire() { for (const l of lists) for (const f of l.fns) f(); },
  };
  return win;
}

test("a tab is browser mode; an installed window is standalone", () => {
  assert.equal(displayMode(fakeWindow("browser")), "browser");
  assert.equal(displayMode(fakeWindow("standalone")), "standalone");
  assert.equal(displayMode(fakeWindow("fullscreen")), "fullscreen");
  assert.equal(displayMode({}), "browser");
});

test("isStandalone reads the media query or iOS's navigator.standalone", () => {
  assert.equal(isStandalone(fakeWindow("standalone")), true);
  assert.equal(isStandalone(fakeWindow("browser", true)), true);
  assert.equal(isStandalone(fakeWindow("browser", false)), false);
  assert.equal(isStandalone(fakeWindow("fullscreen")), false);
  assert.equal(isStandalone({}), false);
});

test("watching reports changes only, and stops", () => {
  const win = fakeWindow("browser");
  const seen = [];
  const stop = watchDisplayMode(win, (m) => seen.push(m));
  win.fire();
  win.mode = "standalone";
  win.fire();
  stop();
  win.mode = "fullscreen";
  win.fire();
  assert.deepEqual(seen, ["standalone"]);
});

test("markDisplayMode keeps <html data-display-mode> current", () => {
  const win = fakeWindow("browser");
  const doc = { documentElement: { dataset: {} } };
  const stop = markDisplayMode(doc, win);
  assert.equal(doc.documentElement.dataset.displayMode, "browser");
  win.mode = "standalone";
  win.fire();
  assert.equal(doc.documentElement.dataset.displayMode, "standalone");
  stop();
});
