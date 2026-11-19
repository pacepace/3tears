/**
 * @param {{ manifest: import("./manifest.js").Manifest, srcDir?: string, filename?: string,
 *   runtimeFiles?: string[], registerType?: "prompt" | "autoUpdate", [key: string]: unknown }} options
 * @returns {Record<string, unknown>}
 */
export function vitePwa({ manifest, srcDir, filename, runtimeFiles, registerType, ...rest }: {
    manifest: import("./manifest.js").Manifest;
    srcDir?: string;
    filename?: string;
    runtimeFiles?: string[];
    registerType?: "prompt" | "autoUpdate";
    [key: string]: unknown;
}): Record<string, unknown>;
