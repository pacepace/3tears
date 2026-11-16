import test from "node:test";
import assert from "node:assert/strict";
import { MANIFEST_MEDIA_TYPE, VIEWPORT, appManifest, headTags, installId } from "../src/manifest.js";

const base = {
  name: "Kairo", start_url: "/player/?station=kairo", scope: "/", id: "/stations/kairo", display: "standalone",
  theme_color: "#07060c", background_color: "#07060c",
  icons: [{ src: "/i/192.png", sizes: "192x192", type: "image/png" }, { src: "/i/512.png", sizes: "512x512", type: "image/png" }],
};

test("installId joins a prefix and a stable slug", () => {
  assert.equal(installId("/stations", "kairo"), "/stations/kairo");
  assert.equal(installId("/stations/", "dj-2"), "/stations/dj-2");
  for (const bad of ["", "Kairo", "kai ro", "-k", "k-", "a/b", "x".repeat(64)]) assert.throws(() => installId("/s", bad), /slug/);
  for (const bad of ["s", "//evil/x", "https://evil/", "/a/../b"]) assert.throws(() => installId(bad, "k"), /path-absolute/);
});

test("appManifest returns the manifest as given, keys in order", () => {
  const out = appManifest(base);
  assert.deepEqual(out, base);
  assert.notEqual(out, base);
  assert.deepEqual(Object.keys(out), Object.keys(base));
  // metallm's: no id, icons listed by size, maskable reusing the 512
  const metallm = { name: "MetaLLM", short_name: "MetaLLM", theme_color: "#2563eb", background_color: "#111111",
    display: "standalone", scope: "/", start_url: "/", icons: base.icons };
  assert.deepEqual(appManifest(metallm), metallm);
});

test("appManifest refuses what would not install or would leave its scope", () => {
  assert.throws(() => appManifest({ ...base, name: " " }), /name/);
  assert.throws(() => appManifest({ ...base, start_url: "https://elsewhere/" }), /start_url/);
  assert.throws(() => appManifest({ ...base, scope: "/player/", start_url: "/auth/login" }), /outside scope/);
  assert.throws(() => appManifest({ ...base, scope: "/player/", start_url: "/player/../admin/" }), /path-absolute/);
  assert.throws(() => appManifest({ ...base, scope: "/player" }), /end in/);
  assert.throws(() => appManifest({ ...base, theme_color: "black" }), /hex/);
  assert.throws(() => appManifest({ ...base, icons: [base.icons[0]] }), /512x512/);
  assert.deepEqual(appManifest({ ...base, icons: [{ src: "/i.svg", sizes: "any", type: "image/svg+xml" }] }).icons[0].sizes, "any");
});

test("headTags writes what iOS reads instead of the manifest, escaped", () => {
  const tags = headTags({ manifestUrl: "/m?a=1&b=2", title: "<K>", themeColor: "#fff", darkThemeColor: "#000",
    appleTouchIcon: "/a.png", icons: [{ href: "/f.png", type: "image/png", sizes: "32x32" }], statusBarStyle: "black-translucent" });
  const lines = tags.split("\n");
  assert.ok(lines.includes('<link rel="manifest" href="/m?a=1&amp;b=2">'));
  assert.ok(lines.includes('<meta name="theme-color" content="#fff" media="(prefers-color-scheme: light)">'));
  assert.ok(lines.includes('<meta name="theme-color" content="#000" media="(prefers-color-scheme: dark)">'));
  assert.ok(lines.includes('<meta name="apple-mobile-web-app-title" content="&lt;K&gt;">'));
  assert.ok(lines.includes('<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">'));
  assert.ok(lines.includes('<link rel="apple-touch-icon" sizes="180x180" href="/a.png">'));
  assert.ok(lines.includes('<link rel="icon" type="image/png" sizes="32x32" href="/f.png">'));
  assert.equal(headTags({ manifestUrl: "/m", title: "T", themeColor: "#000" }).match(/theme-color/g).length, 1);
  assert.equal(MANIFEST_MEDIA_TYPE, "application/manifest+json");
  assert.match(VIEWPORT, /viewport-fit=cover/);
});
