// The launch state of an installed iPhone app (WebKit bug 301994): laid out short of the home
// indicator's strip, with a bottom inset. The example's backdrop (safe-area.css's
// .safe-area-backdrop) and root background (--safe-background) must paint that strip, and the
// backdrop must add no scrolling. Headless Chromium stands in: a viewport short by the inset, the
// insets overridden, and a screenshot beyond the viewport down to the screen's edge.
import test from "node:test";
import assert from "node:assert/strict";
import { useChromium } from "./chromium.js";

const env = useChromium();

const IPHONE = { width: 393, height: 852 }; // iPhone 15/16, CSS px
const INSETS = { top: 59, bottom: 34, left: 0, right: 0 };
const SENTINEL = [255, 0, 255]; // a root background nobody would choose, set by the test
const BACKDROP_BOTTOM = [0x14, 0x16, 0x1a]; // app.css: the backdrop's gradient ends here

/** Opens the example short of its bottom strip, as an installed iPhone app launches. */
async function launchShort() {
  const context = await env.browser.newContext({
    viewport: { width: IPHONE.width, height: IPHONE.height - INSETS.bottom },
    deviceScaleFactor: 1,
    serviceWorkers: "block",
  });
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Emulation.setSafeAreaInsetsOverride", { insets: INSETS });
  await page.goto(`${env.app.url}/app/`);
  await page.waitForFunction(() => document.getElementById("version").textContent !== "");
  return { context, page, cdp };
}

/**
 * The colours of the screen's bottom strip, below the short layout: a screenshot beyond the
 * viewport, decoded by the browser itself, sampled across the strip.
 * @returns {Promise<number[][]>}
 */
async function strip(page, cdp) {
  const shot = await cdp.send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
    clip: { x: 0, y: 0, width: IPHONE.width, height: IPHONE.height, scale: 1 },
  });
  return page.evaluate(
    async ({ data, top, height, width }) => {
      const image = await createImageBitmap(await (await fetch(`data:image/png;base64,${data}`)).blob());
      const canvas = new OffscreenCanvas(image.width, image.height);
      const g = canvas.getContext("2d");
      g.drawImage(image, 0, 0);
      const out = [];
      for (let y = top + 1; y < top + height - 1; y += 4) {
        for (let x = 8; x < width - 8; x += 48) out.push([...g.getImageData(x, y, 1, 1).data.slice(0, 3)]);
      }
      return out;
    },
    { data: shot.data, top: IPHONE.height - INSETS.bottom, height: INSETS.bottom, width: IPHONE.width },
  );
}

const near = (a, b) => a.every((v, i) => Math.abs(v - b[i]) <= 6);

test("at launch, short of the bottom strip: the backdrop covers the strip and adds no scrolling", async (t) => {
  if (env.skip) return t.skip(env.skip);
  const { context, page, cdp } = await launchShort();
  assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--safe-bottom").trim() !== ""), true);
  // the test's own root colour, so the backdrop is told apart from the root background
  await page.addStyleTag({ content: `:root { --safe-background: rgb(${SENTINEL.join(",")}); }` });
  const pixels = await strip(page, cdp);
  assert.ok(pixels.length > 0);
  for (const px of pixels) assert.ok(near(px, BACKDROP_BOTTOM), `strip pixel ${px} is not the backdrop's`);

  const scroll = await page.evaluate(() => {
    const backdrop = /** @type {HTMLElement} */ (document.querySelector(".safe-area-backdrop"));
    const withIt = document.documentElement.scrollHeight;
    const box = backdrop.getBoundingClientRect();
    backdrop.style.display = "none";
    const without = document.documentElement.scrollHeight;
    backdrop.style.display = "";
    return { withIt, without, inner: innerHeight, bottom: box.bottom };
  });
  assert.equal(scroll.withIt, scroll.without, "the backdrop adds scroll height");
  assert.ok(scroll.bottom >= scroll.inner + INSETS.bottom, `the backdrop ends at ${scroll.bottom}, inside ${scroll.inner + INSETS.bottom}`);
  await context.close();
});

test("a backdrop of the large viewport's height alone leaves the strip to the root background, which covers it", async (t) => {
  if (env.skip) return t.skip(env.skip);
  const { context, page, cdp } = await launchShort();
  await page.addStyleTag({ content: `:root { --safe-background: rgb(${SENTINEL.join(",")}); } .safe-area-backdrop { height: 100lvh; }` });
  // without the inset the backdrop stops at the short layout: the strip is the root background,
  // never the browser's default white or black
  for (const px of await strip(page, cdp)) assert.ok(near(px, SENTINEL), `strip pixel ${px} is not the root background`);
  await context.close();
});

test("with no --safe-background, a body background still paints the strip", async (t) => {
  if (env.skip) return t.skip(env.skip);
  const { context, page, cdp } = await launchShort();
  await page.addStyleTag({
    content: `:root { --safe-background: initial; } body { background: rgb(${SENTINEL.join(",")}); } .safe-area-backdrop { display: none; }`,
  });
  for (const px of await strip(page, cdp)) assert.ok(near(px, SENTINEL), `strip pixel ${px} is not the body's background`);
  await context.close();
});
