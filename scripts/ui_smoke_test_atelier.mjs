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
 *   3. Clicking a tag chip moves to ?tag=foo and runs a tag-restricted
 *      search (no similarity badges).
 *   4. Clicking "Similar" on a pin switches to similar mode keyed on its id.
 *   5. The lightbox opens when a pin is clicked and closes via Escape.
 *   6. The pin overflow menu exposes Flag NSFW / Re-annotate / Delete and
 *      Delete optimistically removes the pin from the masonry.
 *   7. "Load more" pulls the next offset and appends pins.
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

const sampleImages = Array.from({ length: 96 }, (_, i) => ({
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

// Synthetic videos that the Feed flow can use as seeds. We expose them via
// /api/videos for the Rail launcher and as search/similar-videos results for
// the per-pin Feed flow.
const sampleVideos = Array.from({ length: 6 }, (_, i) => ({
  image_id: 5000 + i,
  video_id: 200 + i,
  original_name: `clip-${i}.mp4`,
  storage_path: `videos/clip-${i}`,
  preview_path: `images/clip-${i}-frame`,
  mime_type: "video/mp4",
  width: 720,
  height: 1280,
  duration_ms: 8500,
  description: `Short cinematic clip ${i} of cats playing in warm light.`,
  tags: ["video", "cat", "warm-tone"],
}));

function searchResults(seedOffset = 0) {
  const imageHits = sampleImages.slice(0, 10).map((image, i) => ({
    image_id: image.image_id,
    media_type: "image",
    preview_path: image.storage_path,
    storage_path: image.storage_path,
    mime_type: image.mime_type,
    width: image.width,
    height: image.height,
    distance: 0.1 + (i + seedOffset) * 0.04,
    original_name: image.original_name,
    description: image.description,
    tags: image.tags,
  }));
  const videoHits = sampleVideos.slice(0, 2).map((video, i) => ({
    image_id: video.image_id,
    video_id: video.video_id,
    media_type: "video",
    preview_path: video.preview_path,
    storage_path: video.storage_path,
    mime_type: video.mime_type,
    width: video.width,
    height: video.height,
    duration_ms: video.duration_ms,
    distance: 0.05 + (i + seedOffset) * 0.04,
    original_name: video.original_name,
    description: video.description,
    tags: video.tags,
  }));
  return [...videoHits, ...imageHits];
}

function similarVideoResults(excludeIds) {
  const exclude = new Set(excludeIds);
  return sampleVideos
    .filter((v) => !exclude.has(v.video_id))
    .map((video, i) => ({
      image_id: video.image_id,
      video_id: video.video_id,
      media_type: "video",
      preview_path: video.preview_path,
      storage_path: video.storage_path,
      mime_type: video.mime_type,
      width: video.width,
      height: video.height,
      duration_ms: video.duration_ms,
      distance: 0.1 + i * 0.04,
      original_name: video.original_name,
      description: video.description,
      tags: video.tags,
    }));
}

async function serveDist(res, pathname) {
  const rel = pathname === "/" ? "index.html" : pathname.replace(/^\//, "");
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

let nsfwToggleCount = 0;
let reannotateCount = 0;
let deleteCount = 0;
const tagSearchRequests = [];
const imagesRequests = [];
const videosRequests = [];
const uploadRequests = [];
const similarVideoRequests = [];
const requestOrder = [];

const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", "http://127.0.0.1");
  try {
    if (url.pathname === "/api/stats") {
      jsonResponse(res, 200, {
        images_total: sampleImages.length,
        standalone_images_total: sampleImages.length,
        videos_total: 7,
      });
      return;
    }
    if (url.pathname === "/api/search/tag-cloud") {
      requestOrder.push("tag-cloud");
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
      requestOrder.push("images");
      const limit = Number(url.searchParams.get("limit") || 24);
      const offset = Number(url.searchParams.get("offset") || 0);
      imagesRequests.push({ limit, offset });
      jsonResponse(res, 200, {
        images: sampleImages.slice(offset, offset + limit),
        total: sampleImages.length,
      });
      return;
    }
    if (url.pathname === "/api/videos") {
      const limit = Number(url.searchParams.get("limit") || 24);
      const offset = Number(url.searchParams.get("offset") || 0);
      videosRequests.push({ limit, offset });
      jsonResponse(res, 200, {
        videos: sampleVideos.slice(offset, offset + limit),
        total: sampleVideos.length,
      });
      return;
    }
    if (url.pathname === "/api/search/text") {
      jsonResponse(res, 200, {
        results: searchResults(),
        total: 12,
        debug: { duration_ms: 12 },
      });
      return;
    }
    if (url.pathname === "/api/search/similar-videos") {
      const videoId = Number(url.searchParams.get("video_id") || 0);
      const seenIds = (url.searchParams.get("seen") || "")
        .split(",")
        .map(Number)
        .filter((n) => !Number.isNaN(n) && n > 0);
      const preferTags = (url.searchParams.get("prefer_tags") || "")
        .split(",")
        .filter(Boolean);
      const avoidTags = (url.searchParams.get("avoid_tags") || "")
        .split(",")
        .filter(Boolean);
      const limit = Number(url.searchParams.get("limit") || 12);
      similarVideoRequests.push({ videoId, seenIds, preferTags, avoidTags, limit });
      const candidates = similarVideoResults([videoId, ...seenIds]).slice(0, limit);
      jsonResponse(res, 200, {
        results: candidates,
        total: candidates.length,
      });
      return;
    }
    if (url.pathname === "/api/search/similar") {
      const seed = Number(url.searchParams.get("image_id") || 0);
      jsonResponse(res, 200, {
        results: searchResults().slice(0, 6).map((r, idx) => ({
          ...r,
          is_anchor: idx === 0,
          image_id: idx === 0 ? seed : r.image_id,
        })),
        total: 6,
      });
      return;
    }
    if (url.pathname === "/api/search/tags") {
      const tags = url.searchParams.getAll("tag");
      const mode = url.searchParams.get("tag_mode") || "any";
      tagSearchRequests.push({ tags, mode });
      jsonResponse(res, 200, {
        results: searchResults().map((r) => ({
          ...r,
          search_source: "tag",
          distance: 0,
          tags: Array.from(new Set([...(r.tags || []), ...tags])),
        })),
        total: 12,
      });
      return;
    }
    if (/^\/api\/(images|videos)\/\d+\/toggle-nsfw$/.test(url.pathname) && req.method === "POST") {
      nsfwToggleCount += 1;
      jsonResponse(res, 200, { ok: true });
      return;
    }
    if (/^\/api\/(images|videos)\/\d+\/reannotate$/.test(url.pathname) && req.method === "POST") {
      reannotateCount += 1;
      jsonResponse(res, 200, { ok: true });
      return;
    }
    if (/^\/api\/(images|videos)\/\d+$/.test(url.pathname) && req.method === "DELETE") {
      deleteCount += 1;
      res.writeHead(204);
      res.end();
      return;
    }
    if (url.pathname === "/api/upload" && req.method === "POST") {
      // We don't need to fully parse multipart; counting the per-part
      // `name="file"; filename="..."` headers is enough for the smoke test
      // to know what filenames the client sent.
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      const body = Buffer.concat(chunks).toString("binary");
      const filenames = Array.from(
        body.matchAll(/name="file";\s*filename="([^"]*)"/g),
        (m) => m[1] || "unknown",
      );
      uploadRequests.push({ filenames });
      const uploads = filenames.map((name, i) => {
        // Mark the second file as duplicate so the row-state mapping is
        // exercised; the rest are reported as "created".
        const duplicate = i === 1;
        return {
          filename: name,
          media_type: "image",
          image_id: 9000 + i,
          sha256: `deadbeef${i.toString(16).padStart(2, "0")}`,
          duplicate,
        };
      });
      const duplicates = uploads.filter((u) => u.duplicate).length;
      jsonResponse(res, uploads.length === 1 ? 201 : 207, {
        uploads,
        created: uploads.length - duplicates,
        duplicates,
        failed: 0,
      });
      return;
    }
    if (url.pathname.startsWith("/media/")) {
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

  // 1. Library shell renders.
  await page.locator("[data-pin]").first().waitFor({ state: "visible", timeout: 10000 });
  const pinCount = await page.locator("[data-pin]").count();
  if (pinCount < 12) {
    throw new Error(`expected initial 48-page library to render at least 12 pins, got ${pinCount}`);
  }
  const headline = (await page.locator("h1").first().textContent() || "").trim();
  if (headline !== "Library") {
    throw new Error(`expected library headline, got ${JSON.stringify(headline)}`);
  }
  const firstImagesRequest = requestOrder.indexOf("images");
  const firstTagCloudRequest = requestOrder.indexOf("tag-cloud");
  if (firstImagesRequest < 0) {
    throw new Error(`expected initial /api/images request, got order ${JSON.stringify(requestOrder)}`);
  }
  if (firstTagCloudRequest >= 0 && firstTagCloudRequest < firstImagesRequest) {
    throw new Error(
      `expected first /api/images request before tag-cloud bootstrap, got order ${JSON.stringify(requestOrder)}`,
    );
  }

  // 1b. Rail Feed launcher — starts from a random video even though the
  //     library grid itself is image-only.
  const baselineRailSimilarVideos = similarVideoRequests.length;
  await page.locator('button[aria-label^="Feed"]').click();
  await page.locator("[data-feed-overlay]").waitFor({ state: "visible", timeout: 5000 });
  await page.waitForFunction(
    () => {
      const overlay = document.querySelector("[data-feed-overlay]");
      const size = Number(overlay?.getAttribute("data-feed-queue-size") || 0);
      return size > 1;
    },
    {},
    { timeout: 5000 },
  );
  if (videosRequests.length === 0) {
    throw new Error("expected Rail Feed click to request /api/videos");
  }
  const railFeedRequest = similarVideoRequests[baselineRailSimilarVideos];
  if (!railFeedRequest || railFeedRequest.videoId < 200 || railFeedRequest.videoId > 205) {
    throw new Error(
      `expected Rail Feed to seed one sample video, got ${JSON.stringify(railFeedRequest)}`,
    );
  }
  await page.keyboard.press("Escape");
  await page.locator("[data-feed-overlay]").waitFor({ state: "hidden", timeout: 5000 });

  // 2. Search flow.
  await page.locator("#atelier-search").fill("warm portrait");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /\?q=/.test(window.location.search), {}, { timeout: 5000 });
  await page.locator("[data-pin-match]").first().waitFor({ state: "visible", timeout: 5000 });
  const searchHeadline = (await page.locator("h1").first().textContent() || "").trim();
  if (searchHeadline !== "warm portrait") {
    throw new Error(`expected search headline to mirror query, got ${JSON.stringify(searchHeadline)}`);
  }

  const visualTopRowMatches = await page.locator("[data-pin]").evaluateAll((pins) =>
    pins
      .map((pin) => {
        const rect = pin.getBoundingClientRect();
        const match = pin.querySelector("[data-pin-match]")?.textContent?.trim() || "";
        return { left: rect.left, top: rect.top, match };
      })
      .filter((pin) => pin.match)
      .sort((a, b) => a.top - b.top || a.left - b.left)
      .slice(0, 4)
      .map((pin) => pin.match),
  );
  const expectedTopRowMatches = ["95%", "91%", "90%", "86%"];
  if (JSON.stringify(visualTopRowMatches) !== JSON.stringify(expectedTopRowMatches)) {
    throw new Error(
      `expected visual top row to preserve search rank ${JSON.stringify(expectedTopRowMatches)}, got ${JSON.stringify(visualTopRowMatches)}`,
    );
  }

  // 2b. Feed flow — search results include video pins; clicking the Feed
  //     corner action on one opens the fullscreen overlay, kicks a
  //     /api/search/similar-videos fetch with the seed video_id, and
  //     responds to keyboard navigation + Escape close.
  const feedTrigger = page.locator('[data-pin-action="feed"]').first();
  await feedTrigger.waitFor({ state: "visible", timeout: 5000 });
  const baselineSimilarVideos = similarVideoRequests.length;
  await feedTrigger.click();
  await page.locator("[data-feed-overlay]").waitFor({ state: "visible", timeout: 5000 });

  await page.waitForFunction(
    (baseline) => {
      const overlay = document.querySelector("[data-feed-overlay]");
      const size = Number(overlay?.getAttribute("data-feed-queue-size") || 0);
      return size > 1; // seed + at least one similar candidate
    },
    baselineSimilarVideos,
    { timeout: 5000 },
  );

  if (similarVideoRequests.length <= baselineSimilarVideos) {
    throw new Error(
      `expected at least one /api/search/similar-videos request, got ${similarVideoRequests.length}`,
    );
  }
  const firstFeedRequest = similarVideoRequests[baselineSimilarVideos];
  if (!firstFeedRequest.videoId || firstFeedRequest.videoId < 200) {
    throw new Error(
      `expected similar-videos request to carry the seed video_id (>=200), got ${JSON.stringify(firstFeedRequest)}`,
    );
  }
  if (!firstFeedRequest.seenIds.includes(firstFeedRequest.videoId)) {
    throw new Error(
      `expected seen list to include the seed videoId, got ${JSON.stringify(firstFeedRequest)}`,
    );
  }

  // ArrowDown advances. Read the data-feed-current-index attr before/after.
  const overlayHandle = await page.locator("[data-feed-overlay]").elementHandle();
  if (!overlayHandle) throw new Error("feed overlay handle missing");
  const beforeIdx = Number(await overlayHandle.getAttribute("data-feed-current-index"));
  await page.keyboard.press("ArrowDown");
  await page.waitForFunction(
    (before) => {
      const overlay = document.querySelector("[data-feed-overlay]");
      const idx = Number(overlay?.getAttribute("data-feed-current-index") || 0);
      return idx > before;
    },
    beforeIdx,
    { timeout: 5000 },
  );

  await page.keyboard.press("Escape");
  await page.locator("[data-feed-overlay]").waitFor({ state: "hidden", timeout: 5000 });

  // 3. Tag chip flow — clicks the "Tag · portrait" quick-row chip and
  //    expects to land on ?tag=portrait with no similarity badges (tag
  //    search results have search_source="tag" / distance=0).
  await page.getByRole("button", { name: "Tag · portrait" }).click();
  await page.waitForFunction(
    () => /\?tag=portrait/.test(window.location.search),
    {},
    { timeout: 5000 },
  );
  await page.waitForFunction(
    () => document.querySelectorAll("[data-pin-match]").length === 0,
    {},
    { timeout: 5000 },
  );
  const tagHeadline = (await page.locator("h1").first().textContent() || "").trim();
  if (tagHeadline !== "portrait") {
    throw new Error(`expected tag headline, got ${JSON.stringify(tagHeadline)}`);
  }
  if (tagSearchRequests.length === 0 || tagSearchRequests[0].tags[0] !== "portrait") {
    throw new Error(`expected /api/search/tags request for "portrait", got ${JSON.stringify(tagSearchRequests)}`);
  }

  // Reset to library before similar/lightbox/menu checks.
  await page.locator('a[aria-label="imgsearch home"]').click();
  await page.waitForFunction(() => window.location.search === "", {}, { timeout: 5000 });
  await page.locator("[data-pin]").first().waitFor({ state: "visible", timeout: 5000 });

  // 4. Similar flow via the corner action.
  const firstPin = page.locator("[data-pin]").first();
  await firstPin.hover();
  await firstPin.locator('[data-pin-action="similar"]').click();
  await page.waitForFunction(() => /\?similar=/.test(window.location.search), {}, { timeout: 5000 });
  await page.locator("[data-pin-anchor]").waitFor({ state: "visible", timeout: 5000 });
  const similarHeadline = (await page.locator("h1").first().textContent() || "").trim();
  if (similarHeadline !== "Similar in your library") {
    throw new Error(`expected similar headline, got ${JSON.stringify(similarHeadline)}`);
  }

  // Back to library for lightbox / menu / load-more.
  await page.locator('a[aria-label="imgsearch home"]').click();
  await page.waitForFunction(() => window.location.search === "", {}, { timeout: 5000 });
  await page.locator("[data-pin]").first().waitFor({ state: "visible", timeout: 5000 });

  // 5. Lightbox open + Escape close.
  await page.locator('[data-pin-media]').first().click();
  await page.locator("[data-lightbox]").waitFor({ state: "visible", timeout: 5000 });
  await page.keyboard.press("Escape");
  await page.locator("[data-lightbox]").waitFor({ state: "hidden", timeout: 5000 });

  // 6. Pin overflow menu — Re-annotate hits the API; Delete drops the pin.
  const targetPin = page.locator("[data-pin]").first();
  const beforeFirst = await targetPin.evaluate((el) => el.getAttribute("data-pin-anchor"));
  if (beforeFirst === "true") {
    throw new Error("smoke test assumed first library pin is not an anchor");
  }
  const initialCount = await page.locator("[data-pin]").count();

  await targetPin.hover();
  await targetPin.locator('[data-pin-action="more"]').click();
  await targetPin.locator('[data-pin-menu="reannotate"]').click();
  // Wait for the in-flight POST to settle so menu items become enabled
  // again before the next interaction.
  const reannotateDeadline = Date.now() + 5000;
  while (reannotateCount !== 1 && Date.now() < reannotateDeadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  if (reannotateCount !== 1) {
    throw new Error(`expected one re-annotate POST, got ${reannotateCount}`);
  }
  await page.waitForFunction(
    () =>
      Array.from(document.querySelectorAll("[data-pin-menu]")).every(
        (el) => !el.disabled,
      ),
    {},
    { timeout: 5000 },
  );

  // Confirm dialog must auto-accept for the delete branch.
  page.once("dialog", (dialog) => dialog.accept());
  const targetPin2 = page.locator("[data-pin]").first();
  await targetPin2.hover();
  await targetPin2.locator('[data-pin-action="more"]').click();
  await targetPin2.locator('[data-pin-menu="delete"]').click();
  await page.waitForFunction(
    (initial) => document.querySelectorAll("[data-pin]").length === initial - 1,
    initialCount,
    { timeout: 5000 },
  );
  if (deleteCount !== 1) {
    throw new Error(`expected one DELETE, got ${deleteCount}`);
  }

  // 7. Load more — ensure clicking it grows the masonry.
  const beforeLoadMore = await page.locator("[data-pin]").count();
  await page.locator("[data-load-more]").click();
  await page.waitForFunction(
    (before) => document.querySelectorAll("[data-pin]").length > before,
    beforeLoadMore,
    { timeout: 5000 },
  );
  if (imagesRequests.length < 2) {
    throw new Error(`expected at least two /api/images requests after load-more, got ${imagesRequests.length}`);
  }
  const lastRequest = imagesRequests[imagesRequests.length - 1];
  if (lastRequest.offset === 0) {
    throw new Error(`expected load-more to request a non-zero offset, got ${JSON.stringify(lastRequest)}`);
  }

  // 7b. Upload flow — open modal via header trigger, attach two files, submit,
  //     verify per-file row states (created + duplicate), summary, and that
  //     the library re-fetches via the dataEpoch bump.
  const imagesRequestsBeforeUpload = imagesRequests.length;
  await page.locator('[data-upload-trigger]').first().click();
  await page.locator("[data-upload-modal]").waitFor({ state: "visible", timeout: 5000 });

  await page.locator("[data-upload-input]").setInputFiles([
    {
      name: "alpha.png",
      mimeType: "image/png",
      buffer: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
        "base64",
      ),
    },
    {
      name: "beta.png",
      mimeType: "image/png",
      buffer: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
        "base64",
      ),
    },
  ]);
  await page.waitForFunction(
    () => document.querySelectorAll("[data-upload-row]").length === 2,
    {},
    { timeout: 5000 },
  );

  await page.locator("[data-upload-submit]").click();
  await page.waitForFunction(
    () =>
      Array.from(document.querySelectorAll("[data-upload-row]")).every((row) => {
        const s = row.getAttribute("data-upload-state");
        return s === "created" || s === "duplicate" || s === "failed";
      }),
    {},
    { timeout: 5000 },
  );

  if (uploadRequests.length !== 1) {
    throw new Error(`expected one /api/upload request, got ${uploadRequests.length}`);
  }
  const sentNames = uploadRequests[0].filenames;
  if (sentNames.length !== 2 || sentNames[0] !== "alpha.png" || sentNames[1] !== "beta.png") {
    throw new Error(`unexpected upload filenames: ${JSON.stringify(sentNames)}`);
  }

  const states = await page
    .locator("[data-upload-row]")
    .evaluateAll((rows) => rows.map((r) => r.getAttribute("data-upload-state")));
  if (states[0] !== "created" || states[1] !== "duplicate") {
    throw new Error(`expected [created, duplicate] row states, got ${JSON.stringify(states)}`);
  }

  await page.locator("[data-upload-summary]").waitFor({ state: "visible", timeout: 2000 });

  // dataEpoch bump should have triggered a fresh /api/images call.
  if (imagesRequests.length <= imagesRequestsBeforeUpload) {
    throw new Error(
      `expected library to re-fetch after upload, /api/images count stayed at ${imagesRequests.length}`,
    );
  }

  await page.locator("[data-upload-close]").click();
  await page.locator("[data-upload-modal]").waitFor({ state: "hidden", timeout: 5000 });

  // 8. Rail "Library" returns to library mode and clears query.
  await page.locator('a[aria-label="imgsearch home"]').click();
  await page.waitForFunction(() => window.location.search === "", {}, { timeout: 5000 });
  await page.locator("[data-pin]").first().waitFor({ state: "visible", timeout: 5000 });
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
