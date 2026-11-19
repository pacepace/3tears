/**
 * A stable install id for one of several apps on a host: `<prefix>/<slug>`. The slug is chosen
 * once (a station id, a persona's slug), never a display name.
 * @param {string} prefix path-absolute, e.g. "/stations"
 * @param {string} slug lowercase letters, digits, "-" or "_"
 * @returns {string}
 */
export function installId(prefix: string, slug: string): string;
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
export function appManifest(manifest: Manifest): Manifest;
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
export function headTags({ manifestUrl, title, themeColor, darkThemeColor, appleTouchIcon, icons, statusBarStyle }: {
    manifestUrl: string;
    title: string;
    themeColor: string;
    darkThemeColor?: string;
    appleTouchIcon?: string;
    icons?: Array<{
        href: string;
        type: string;
        sizes: string;
    }>;
    statusBarStyle?: "default" | "black" | "black-translucent";
}): string;
/**
 * The manifest's media type, to serve it with.
 */
export const MANIFEST_MEDIA_TYPE: "application/manifest+json";
/** The viewport an edge-to-edge installed page wants. */
export const VIEWPORT: "width=device-width, initial-scale=1, viewport-fit=cover";
export type ManifestIcon = {
    src: string;
    sizes: string;
    type?: string;
    purpose?: string;
};
export type Manifest = {
    name: string;
    short_name?: string;
    description?: string;
    id?: string;
    start_url: string;
    scope?: string;
    display?: string;
    orientation?: string;
    theme_color?: string;
    background_color?: string;
    icons: ManifestIcon[];
    categories?: string[];
    lang?: string;
    [key: string]: unknown;
};
