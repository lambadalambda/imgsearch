#!/usr/bin/env node
// A structural delivery gate, not a replacement for the browser smoke test.
import { realpathSync } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

export const atelierDist = fileURLToPath(new URL("../internal/webui/atelier/dist/", import.meta.url));

export async function validateAtelierBuild(dist = atelierDist) {
  try {
    const root = resolve(dist);
    const index = await readFile(join(root, "index.html"), "utf8");
    // These are the stable parts of our Vite output, absent from the developer
    // placeholder and uncompiled frontend/index.html.
    if (!/<div\b[^>]*\bid=["']app["']/.test(index) ||
        !/<script\b[^>]*\btype=["']module["'][^>]*\bsrc=["']\/assets\/[^"']+\.js["']/.test(index) ||
        !/<link\b[^>]*\bhref=["']\/assets\/[^"']+\.css["']/.test(index)) {
      throw new Error("index.html must contain the app mount and emitted JS/CSS assets (not a placeholder or source index)");
    }
    for (const [, url] of index.matchAll(/(?:src|href)=["'](\/assets\/[^"']+)["']/g)) {
      const asset = resolve(root, `.${url}`);
      if (!asset.startsWith(root + sep)) throw new Error(`invalid asset path: ${url}`);
      const info = await stat(asset);
      if (!info.isFile() || info.size === 0) throw new Error(`missing or empty asset: ${url}`);
    }
  } catch (error) {
    throw new Error(`Atelier build is incomplete: ${error.message}. Run mise run build:frontend before building Go.`, { cause: error });
  }
}

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    await validateAtelierBuild(process.argv[2]);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
