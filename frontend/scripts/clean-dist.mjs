#!/usr/bin/env node
/**
 * Pre-build cleanup for the Atelier dist directory.
 *
 * Vite's `emptyOutDir: true` would wipe `.gitkeep`, which we keep so the Go
 * //go:embed directive in internal/webui/atelier/embed.go always has at least
 * one file at compile time. Instead we manually delete only known build
 * artefacts (index.html and assets/) so stale chunks don't accumulate, while
 * preserving any tracked sentinel files.
 */
import { rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.resolve(here, "../../internal/webui/atelier/dist");

const targets = [path.join(distDir, "index.html"), path.join(distDir, "assets")];

for (const target of targets) {
  await rm(target, { recursive: true, force: true });
}
