#!/usr/bin/env node
/**
 * Browser smoke test for the Atelier SPA.
 *
 * Boots an in-process HTTP server that serves the embedded Atelier build
 * (and a tiny stub API) so the test does not require the real backend or a
 * populated database. We exercise just the core flows the MVP must support:
 *
 *   1. Library mode renders a masonry of pins with titles and tag chips.
 *   2. Typing a query into the search bar puts the SPA into search mode and
 *      shows match badges on result pins.
 *   3. Clicking "Similar" on a pin switches to similar mode keyed on its id.
 *   4. The lightbox opens when a pin is clicked and closes via Escape.
 *
 * If the embedded Atelier build is missing (placeholder shell), the test
 * skips with exit code 0 so a fresh checkout that has not yet run
 * `npm run build` does not fail CI.
 */

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, dirname, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..");
const distRoot = join(repoRoot, "internal", "webui", "atelier", "dist");

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

if (!(await exists(join(distRoot, "index.html")))) {
  console.log("atelier dist missing; skipping atelier smoke test (run `npm run build` in frontend/)");
  process.exit(0);
}

const contentTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".js", "application/javascript; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".webp", "image/webp"],
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
]);

function jsonResponse(res, status, payload) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(payload));
}

const sampleImages = Array.from({ length: 24 }, (_, i) => ({
  image_id: 1000 + i,
  original_name: `sample-${i}.jpg`,
  storage_path: `images/sample-${i}`,
  mime_type: "image/jpeg",
  width: 800 + (i % 4) * 80,
  height: 600 + (i % 5) * 60,
  index_state: "done",
  description:
    i % 2 === 0
      ? "A close-up of a tabby cat looking at the camera."
      : "Studio portrait against a warm beige backdrop.",
  tags: ["portrait", "cat", "indoor", "warm-tone", "test"],
}));

const sampleSearchResults = sampleImages.slice(0, 12).map((image, i) => ({
  image_id: image.image_id,
  media_type: "image",
  preview_path: image.storage_path,
  storage_path: image.storage_path,
  mime_type: image.mime_type,
  width: image.width,
  height: image.height,
  distance: 0.1 + i * 0.04,
  original_name: image.original_name,
  description: image.description,
  tags: image.tags,
}));

async function serveDist(res, pathname) {
  let rel = pathname === "/" ? "index.html" : pathname.replace(/^\//, "");
  const filePath = join(distRoot, rel);
  if (filePath !== distRoot && !filePath.startsWith(distRoot + sep)) {
    res.writeHead(404);
    res.end("not found");
    return;
  }
  try {
    const body = await readFile(filePath);
    res.writeHead(200, { "Content-Type": contentTypes.get(extname(filePath)) || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end("not found");
  }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", "http://127.0.0.1");
  try {
    if (url.pathname === "/api/stats") {
      jsonResponse(res, 200, {
        images_total: 24,
        standalone_images_total: 24,
        videos_total: 7,
      });
      return;
    }
    if (url.pathname === "/api/search/tag-cloud") {
      jsonResponse(res, 200, {
        tags: [
          { tag: "portrait", count: 24 },
          { tag: "cat", count: 22 },
          { tag: "warm-tone", count: 18 },
        ],
      });
      return;
    }
    if (url.pathname === "/api/images") {
      const limit = Number(url.searchParams.get("limit") || 24);
      jsonResponse(res, 200, { images: sampleImages.slice(0, limit), total: sampleImages.length });
      return;
    }
    if (url.pathname === "/api/videos") {
      jsonResponse(res, 200, { videos: [], total: 0 });
      return;
    }
    if (url.pathname === "/api/search/text") {
      jsonResponse(res, 200, {
        results: sampleSearchResults,
        total: sampleSearchResults.length,
        debug: { duration_ms: 12 },
      });
      return;
    }
    if (url.pathname === "/api/search/similar") {
      const seed = Number(url.searchParams.get("image_id") || 0);
      jsonResponse(res, 200, {
        results: sampleSearchResults.slice(0, 6).map((r, idx) => ({
          ...r,
          is_anchor: idx === 0,
          image_id: idx === 0 ? seed : r.image_id,
        })),
        total: 6,
      });
      return;
    }
    if (url.pathname.startsWith("/media/")) {
      // Return a tiny 1x1 transparent PNG for any media request.
      const png = Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
        "base64",
      );
      res.writeHead(200, { "Content-Type": "image/png" });
      res.end(png);
      return;
    }
    await serveDist(res, url.pathname);
  } catch (err) {
    res.writeHead(500, { "Content-Type": "text/plain" });
    res.end(String(err && err.stack ? err.stack : err));
  }
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const address = server.address();
const baseURL = `http://127.0.0.1:${address.port}`;

let browser;
try {
  browser = await chromium.launch();
  const page = await browser.newPage();
  page.on("pageerror", (err) => {
    throw new Error(`atelier pageerror: ${err.message}`);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      throw new Error(`atelier console error: ${msg.text()}`);
    }
  });

  await page.goto(`${baseURL}/`, { waitUntil: "networkidle" });

  // 1. Shell renders.
  await page.locator(".pin").first().waitFor({ state: "visible", timeout: 10000 });
  const pinCount = await page.locator(".pin").count();
  if (pinCount < 3) {
    throw new Error(`expected at least 3 library pins, got ${pinCount}`);
  }
  const headline = (await page.locator("h1").first().textContent() || "").trim();
  if (headline !== "Library") {
    throw new Error(`expected library headline, got ${JSON.stringify(headline)}`);
  }

  // 2. Search flow.
  await page.locator("#atelier-search").fill("warm portrait");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /\?q=/.test(window.location.search), {}, { timeout: 5000 });
  await page.locator(".pin .pin-match").first().waitFor({ state: "visible", timeout: 5000 });
  const searchHeadline = (await page.locator("h1").first().textContent() || "").trim();
  if (searchHeadline !== "warm portrait") {
    throw new Error(`expected search headline to mirror query, got ${JSON.stringify(searchHeadline)}`);
  }

  // 3. Similar flow via the corner action.
  const firstPin = page.locator(".pin").first();
  await firstPin.hover();
  await firstPin.locator("button", { hasText: "Similar" }).click();
  await page.waitForFunction(() => /\?similar=/.test(window.location.search), {}, { timeout: 5000 });
  await page.locator(".pin.is-anchor").waitFor({ state: "visible", timeout: 5000 });
  const similarHeadline = (await page.locator("h1").first().textContent() || "").trim();
  if (similarHeadline !== "Similar in your library") {
    throw new Error(`expected similar headline, got ${JSON.stringify(similarHeadline)}`);
  }

  // 4. Lightbox open + Escape close.
  await page.locator(".pin .pin-media").first().click();
  await page.locator(".lightbox").waitFor({ state: "visible", timeout: 5000 });
  await page.keyboard.press("Escape");
  await page.locator(".lightbox").waitFor({ state: "hidden", timeout: 5000 });

  // 5. Rail "Library" returns to library mode and clears query.
  await page.locator('a[aria-label="imgsearch home"]').click();
  await page.waitForFunction(() => window.location.search === "", {}, { timeout: 5000 });
  await page.locator(".pin").first().waitFor({ state: "visible", timeout: 5000 });
  const libraryHeadline = (await page.locator("h1").first().textContent() || "").trim();
  if (libraryHeadline !== "Library") {
    throw new Error(`expected return to library mode, got ${JSON.stringify(libraryHeadline)}`);
  }

  console.log("atelier smoke checks passed");
} finally {
  if (browser) {
    await browser.close();
  }
  server.close();
}
