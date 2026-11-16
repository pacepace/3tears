import test from "node:test";
import assert from "node:assert/strict";
import { MediaSessionControl } from "../src/media-session.js";

class Meta { constructor(o) { Object.assign(this, o); } }

function session() {
  const s = { handlers: {}, sets: 0, metadata: null, playbackState: "none",
    setActionHandler(a, f) { if (a === "seekto") throw new TypeError("unknown"); this.handlers[a] = f; } };
  return s;
}

test("actions run the app's handlers; a live stream clears seek and skip", async () => {
  const s = session();
  const calls = [];
  new MediaSessionControl({ mediaSession: s }, { onPlay: () => calls.push("play"), onPause: () => calls.push("pause"),
    onStop: () => calls.push("stop"), MediaMetadata: Meta });
  s.handlers.play(); s.handlers.pause(); s.handlers.stop();
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(calls, ["play", "pause", "stop"]);
  for (const a of ["seekbackward", "seekforward", "previoustrack", "nexttrack"]) assert.equal(s.handlers[a], null);
  assert.ok(!("seekto" in s.handlers));
});

test("a handler that throws or rejects is contained", async () => {
  const s = session();
  const unhandled = [];
  const on = (e) => unhandled.push(e);
  process.on("unhandledRejection", on);
  new MediaSessionControl({ mediaSession: s }, { onPlay: () => Promise.reject(new Error("NotAllowedError")),
    onPause: () => { throw new Error("boom"); }, MediaMetadata: Meta });
  const warn = console.warn;
  const warned = [];
  console.warn = (...a) => warned.push(a[0]);
  s.handlers.play(); s.handlers.pause();
  await new Promise((r) => setTimeout(r, 10));
  console.warn = warn;
  process.off("unhandledRejection", on);
  assert.deepEqual(unhandled, []);
  assert.deepEqual(warned.sort(), ["media session pause:", "media session play:"]);
});

test("not on a seekable item: live: false leaves seek actions alone", () => {
  const s = session();
  new MediaSessionControl({ mediaSession: s }, { live: false, MediaMetadata: Meta });
  assert.ok(!("nexttrack" in s.handlers));
});

test("metadata is set once per item, refused metadata is tried again, state and clear", () => {
  let sets = 0;
  let fail = true;
  class Picky { constructor(o) { if (fail) throw new TypeError("bad artwork"); Object.assign(this, o); } }
  const s = { setActionHandler() {}, playbackState: "none", set metadata(v) { sets += 1; this._m = v; }, get metadata() { return this._m; } };
  const ms = new MediaSessionControl({ mediaSession: s }, { MediaMetadata: Picky });
  assert.throws(() => ms.setNowPlaying({ title: "A" }));
  fail = false;
  ms.setNowPlaying({ title: "A", artist: "B" });
  ms.setNowPlaying({ title: "A", artist: "B" });
  ms.setNowPlaying({ title: "C" });
  assert.equal(sets, 2);
  assert.equal(s.metadata.title, "C");
  ms.setPlaying(true);
  assert.equal(s.playbackState, "playing");
  ms.clear();
  assert.equal(s.metadata, null);
  assert.equal(s.playbackState, "none");
});

test("without a Media Session everything is a no-op", () => {
  const ms = new MediaSessionControl({});
  assert.equal(ms.supported, false);
  ms.setNowPlaying({ title: "A" }); ms.setPlaying(true); ms.clear();
  assert.equal(ms.handle("play", () => {}), false);
});
