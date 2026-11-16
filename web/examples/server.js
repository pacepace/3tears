// Serves the example app (basic/) under /app/, with @3tears/web-pwa's files under /app/pwa/ as an app
// with no build step would serve them, plus what a real app has around its shell: an API that
// must never be cached, a sign-in redirect that must never be stored, and a version that a
// "deploy" (setVersion) changes.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const APP = join(here, "basic");
const PWA = join(here, "..", "packages", "pwa", "src");
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".webmanifest": "application/manifest+json",
  ".png": "image/png",
};

/** The files of the app's shell: what the shell worker keeps. */
export const SHELL = ["/app/", "/app/app.js", "/app/app.css", "/app/pwa/index.js", "/app/pwa/display.js",
  "/app/pwa/manifest.js", "/app/pwa/media-session.js", "/app/pwa/push.js", "/app/pwa/update.js",
  "/app/pwa/wake-lock.js", "/app/pwa/safe-area.css", "/app/icon-192.png", "/app/icon-512.png"];

/**
 * @param {{ port?: number }} [options]
 * @returns {Promise<{ url: string, setVersion: (v: string) => void, hits: Map<string, number>, close: () => Promise<void> }>}
 */
export async function startExample(options = {}) {
  let version = "1";
  const hits = new Map();
  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://x");
    const path = url.pathname;
    hits.set(path, (hits.get(path) ?? 0) + 1);
    const send = (status, body, type, headers = {}) => {
      res.writeHead(status, { "Content-Type": type, "Cache-Control": "no-cache", ...headers });
      res.end(body);
    };
    if (path === "/app/shell.js") {
      const config = { version, cache: "example", shell: SHELL, never: ["^/app/api/", "^/app/login"] };
      return send(200, `self.SHELL_CONFIG = ${JSON.stringify(config)};\n`, TYPES[".js"]);
    }
    if (path === "/app/version.json") return send(200, JSON.stringify({ version }), TYPES[".json"]);
    if (path.startsWith("/app/api/")) return send(200, JSON.stringify({ at: Date.now(), version }), TYPES[".json"]);
    if (path === "/app/login") return send(302, "", "text/plain", { Location: "/app/" });
    if (path === "/app/private.js") return send(302, "", "text/plain", { Location: "/app/login" });
    let file;
    if (path.startsWith("/app/pwa/")) file = join(PWA, normalize(path.slice("/app/pwa/".length)));
    else if (path === "/app/" || path === "/app/index.html") file = join(APP, "index.html");
    else if (path.startsWith("/app/")) file = join(APP, normalize(path.slice("/app/".length)));
    if (!file || !(file.startsWith(APP) || file.startsWith(PWA))) return send(404, "not found", "text/plain");
    try {
      let body = await readFile(file);
      if (path === "/app/app.js") body = Buffer.from(body.toString().replace('VERSION = "1"', `VERSION = ${JSON.stringify(version)}`));
      // the worker lives with the package's files but serves the whole app: allow its scope
      const extra = path.endsWith("shell-worker.js") ? { "Service-Worker-Allowed": "/app/" } : {};
      return send(200, body, TYPES[extname(file)] ?? "application/octet-stream", extra);
    } catch {
      return send(404, "not found", "text/plain");
    }
  });
  await new Promise((resolve) => server.listen(options.port ?? 0, "127.0.0.1", resolve));
  const { port } = /** @type {import("node:net").AddressInfo} */ (server.address());
  return {
    url: `http://127.0.0.1:${port}`,
    setVersion: (v) => {
      version = v;
    },
    hits,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const app = await startExample({ port: Number(process.env.PORT ?? 4180) });
  console.log(`example app at ${app.url}/app/`);
}
