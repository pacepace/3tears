/**
 * The web app manifest and the head tags every platform reads, as plain data: usable in a
 * browser, in Node (a build script, a Vite config, a server rendering pages) and in tests.
 *
 * What the platforms read:
 * - Chrome, Edge and Firefox read the manifest: `name`, `start_url`, `display`, and a 192 and a
 *   512 px icon make it installable. `id` is the app's identity: compared, never re-read, so it
 *   must never change once people have installed the app.
 * - iOS and iPadOS take the home-screen name from `apple-mobile-web-app-title` (else `<title>`)
 *   and the icon from `<link rel="apple-touch-icon">` (180 px, opaque), never from the manifest.
 *   A host serving several apps from one page writes those tags per request.
 */

const HEX = /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/;
const SLUG = /^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$/;

/**
 * A path-absolute URL on the app's own origin, with no dot segments.
 * @param {string} value
 * @param {string} what
 */
function requirePath(value, what) {
  const ok =
    typeof value === "string" &&
    value.startsWith("/") &&
    !value.startsWith("//") &&
    !value.includes("\\") &&
    !/^[a-z][a-z0-9+.-]*:/i.test(value) &&
    !value.split(/[?#]/)[0].split("/").some((s) => s === "." || s === "..");
  if (!ok) throw new TypeError(`${what} must be a path-absolute URL on the app's own origin: ${JSON.stringify(value)}`);
}

/**
 * A stable install id for one of several apps on a host: `<prefix>/<slug>`. The slug is chosen
 * once (a station id, a persona's slug), never a display name.
 * @param {string} prefix path-absolute, e.g. "/stations"
 * @param {string} slug lowercase letters, digits, "-" or "_"
 * @returns {string}
 */
export function installId(prefix, slug) {
  if (!SLUG.test(slug)) throw new TypeError(`an install slug is lowercase letters, digits, '-' or '_' (max 63): ${JSON.stringify(slug)}`);
  requirePath(prefix, "prefix");
  return `${prefix.replace(/\/+$/, "")}/${slug}`;
}

/**
 * @typedef {{ src: string, sizes: string, type?: string, purpose?: string }} ManifestIcon
 * @typedef {{ name: string, short_name?: string, description?: string, id?: string,
 *   start_url: string, scope?: string, display?: string, orientation?: string,
 *   theme_color?: string, background_color?: string, icons: ManifestIcon[],
 *   categories?: string[], lang?: string, [key: string]: unknown }} Manifest
 */

/**
 * The manifest, checked: the fields install needs are there, `id`, `start_url` and `scope` are
 * on the app's own origin, `start_url` is inside `scope`, colours are hex, and there is a 192 and
 * a 512 px icon. Returned as given (a shallow copy, keys in their order), so a static manifest
 * file, a Vite config and a server all write the same thing.
 * @param {Manifest} manifest
 * @returns {Manifest}
 */
export function appManifest(manifest) {
  if (!manifest?.name?.trim()) throw new TypeError("a manifest needs a name");
  requirePath(manifest.start_url, "start_url");
  if (manifest.id !== undefined) requirePath(manifest.id, "id");
  if (manifest.scope !== undefined) {
    requirePath(manifest.scope, "scope");
    if (!manifest.scope.endsWith("/")) throw new TypeError(`scope must end in '/': ${manifest.scope}`);
    if (!manifest.start_url.split(/[?#]/)[0].startsWith(manifest.scope)) {
      throw new TypeError(`start_url ${manifest.start_url} is outside scope ${manifest.scope}`);
    }
  }
  for (const key of ["theme_color", "background_color"]) {
    if (manifest[key] !== undefined && !HEX.test(String(manifest[key]))) throw new TypeError(`${key} must be a hex colour`);
  }
  const sizes = new Set((manifest.icons ?? []).flatMap((i) => String(i.sizes).split(/\s+/)));
  for (const need of ["192x192", "512x512"]) {
    if (!sizes.has(need) && !sizes.has("any")) throw new TypeError(`install needs a ${need} icon`);
  }
  return { ...manifest };
}

/**
 * The manifest's media type, to serve it with.
 */
export const MANIFEST_MEDIA_TYPE = "application/manifest+json";

/** The viewport an edge-to-edge installed page wants. */
export const VIEWPORT = "width=device-width, initial-scale=1, viewport-fit=cover";

/** @param {string} value */
function attr(value) {
  return String(value).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * The head tags for an installable page, one per line, every value escaped.
 *
 * `statusBarStyle` "black-translucent" draws the page under the iOS status bar in standalone
 * mode; pad the edges with the safe-area insets (safe-area.css) when using it.
 * @param {{ manifestUrl: string, title: string, themeColor: string, darkThemeColor?: string,
 *   appleTouchIcon?: string, icons?: Array<{ href: string, type: string, sizes: string }>,
 *   statusBarStyle?: "default" | "black" | "black-translucent" }} options
 * @returns {string}
 */
export function headTags({ manifestUrl, title, themeColor, darkThemeColor, appleTouchIcon, icons = [], statusBarStyle = "default" }) {
  const lines = [`<link rel="manifest" href="${attr(manifestUrl)}">`];
  if (!darkThemeColor || darkThemeColor === themeColor) {
    lines.push(`<meta name="theme-color" content="${attr(themeColor)}">`);
  } else {
    lines.push(`<meta name="theme-color" content="${attr(themeColor)}" media="(prefers-color-scheme: light)">`);
    lines.push(`<meta name="theme-color" content="${attr(darkThemeColor)}" media="(prefers-color-scheme: dark)">`);
  }
  lines.push('<meta name="mobile-web-app-capable" content="yes">');
  lines.push('<meta name="apple-mobile-web-app-capable" content="yes">');
  lines.push(`<meta name="apple-mobile-web-app-title" content="${attr(title)}">`);
  lines.push(`<meta name="apple-mobile-web-app-status-bar-style" content="${attr(statusBarStyle)}">`);
  if (appleTouchIcon) lines.push(`<link rel="apple-touch-icon" sizes="180x180" href="${attr(appleTouchIcon)}">`);
  for (const i of icons) lines.push(`<link rel="icon" type="${attr(i.type)}" sizes="${attr(i.sizes)}" href="${attr(i.href)}">`);
  return lines.join("\n");
}
