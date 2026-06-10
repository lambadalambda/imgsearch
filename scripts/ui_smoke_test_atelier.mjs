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
 *   4. Clicking "Similar" on a pin switches to similar mode keyed on its id
 *      and scrolls back to the first result.
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

const PAGE_SIZE = 48;

const sampleImages = Array.from({ length: 144 }, (_, i) => ({
  image_id: 1000 + i,
  original_name: `sample-${i}.jpg`,
  storage_path: `images/sample-${i}`,
  mime_type: "image/jpeg",
  width: 800 + (i % 4) * 80,
  height: 600 + (i % 5) * 60,
  // The first pin carries a deliberately long title that wraps to multiple
  // lines in the lightbox's narrow description column, so the desktop test
  // can prove the close button never overlaps the wrapped title.
  title:
    i === 0
      ? "Image card title that intentionally wraps across two lines in the lightbox side panel"
      : `Image card title ${i}`,
  summary: `Image overview summary ${i} for the card preview.`,
  index_state: "done",
  description: `Image overview summary ${i} for the card preview.`,
  full_description:
    i % 2 === 0
      ? `Full generated annotation ${i}: a close-up of a tabby cat looking at the camera with detailed visible context.`
      : `Full generated annotation ${i}: a studio portrait against a warm beige backdrop with detailed visible context.`,
  tags: ["portrait", "cat", "indoor", "warm-tone", "test"],
}));

// Synthetic videos that the Feed flow can use as seeds. We expose them via
// /api/videos for the Rail launcher and as search/similar-videos results for
// the per-pin Feed flow. One record intentionally carries a WebM MIME type;
// the test stubs canPlayType() to return a mobile-style false negative for
// WebM, but a Feed seed only needs a video_id and should not disappear.
const sampleVideos = Array.from({ length: 6 }, (_, i) => ({
  image_id: 5000 + i,
  video_id: 200 + i,
  original_name: i === 2 ? `clip-${i}.webm` : `clip-${i}.mp4`,
  storage_path: `videos/clip-${i}`,
  preview_path: `images/clip-${i}-frame`,
  mime_type: i === 2 ? "video/webm" : "video/mp4",
  width: 720,
  height: 1280,
  duration_ms: 8500,
  title: `Video card title ${i}`,
  summary: `Video overview summary ${i} for the card preview.`,
  description: `Video overview summary ${i} for the card preview.`,
  full_description: `Full generated annotation video ${i}: a short cinematic clip of cats playing in warm light with detailed visible context.`,
  tags: i === 0 ? ["seed-only"] : i === 1 || i === 4 ? ["cat", "warm-tone", "rerank-target"] : ["video", `clip-${i}`],
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
    title: image.title,
    summary: image.summary,
    description: image.description,
    full_description: image.full_description,
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
    title: video.title,
    summary: video.summary,
    description: video.description,
    full_description: video.full_description,
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
      distance: video.video_id === 204 ? 0.13 : 0.1 + i * 0.04,
      original_name: video.original_name,
      title: video.title,
      summary: video.summary,
      description: video.description,
      full_description: video.full_description,
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
let statsServed = 0;
let reannotateCount = 0;
let deleteCount = 0;
const tagSearchRequests = [];
const imagesRequests = [];
const videosRequests = [];
const uploadRequests = [];
const similarVideoRequests = [];
const requestOrder = [];
let similarVideoFailuresRemaining = 0;
let expectedFetchFailureConsoleMessages = 0;

const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", "http://127.0.0.1");
  try {
    if (url.pathname === "/api/stats") {
      statsServed += 1;
      jsonResponse(res, 200, {
        images_total: sampleImages.length + 18,
        standalone_images_total: sampleImages.length,
        video_frame_images_total: 18,
        videos_total: 7,
        queue: {
          total: 162,
          tracked: 147,
          missing: 15,
          annotations_missing: 9,
          runnable: 14,
          pending: 21,
          leased: 3,
          done: 72,
          failed: 2,
          oldest_runnable_age_seconds: 3600,
        },
        image_annotation_expected: sampleImages.length,
        image_annotation_missing: sampleImages.length - 100,
        video_annotation_expected: 7,
        video_annotation_missing: 1,
        video_transcription_expected: 7,
        video_transcription_missing: 2,
        job_kinds: {
          embed_image: {
            tracked: 147,
            runnable: 14,
            // Changes on every fetch so the smoke test can prove the stats
            // pane refreshes itself while open (meta/issues/081).
            pending: 21 + statsServed,
            leased: 3,
            done: 72,
            failed: 2,
            oldest_runnable_age_seconds: 3600,
          },
          annotate_image: {
            tracked: 100,
            runnable: 8,
            pending: 12,
            leased: 1,
            done: 85,
            failed: 2,
            oldest_runnable_age_seconds: 1200,
          },
          annotate_video: {
            tracked: 6,
            runnable: 2,
            pending: 2,
            leased: 0,
            done: 4,
            failed: 0,
            oldest_runnable_age_seconds: 600,
          },
          transcribe_video: {
            tracked: 5,
            runnable: 1,
            pending: 2,
            leased: 0,
            done: 3,
            failed: 1,
            oldest_runnable_age_seconds: 800,
          },
        },
        recent_failures: [
          {
            job_id: 9991,
            kind: "embed_image",
            media_type: "image",
            image_id: 1234,
            original_name: "broken-decode.jpg",
            attempts: 3,
            last_error: "decode error: invalid JPEG",
            updated_at: "2026-06-05T15:23:00Z",
          },
          {
            job_id: 9992,
            kind: "transcribe_video",
            media_type: "video",
            video_id: 42,
            original_name: "silent-track.mp4",
            attempts: 3,
            last_error: "no audio track found",
            updated_at: "2026-06-05T15:10:00Z",
          },
        ],
      });
      return;
    }
    if (url.pathname === "/api/search/tag-cloud") {
      requestOrder.push("tag-cloud");
      jsonResponse(res, 200, {
        // Enough tags to fill the quick-row past its container width so the
        // overflow affordance (meta/issues/082) and the mobile single-line
        // layout (meta/issues/077) are exercised.
        tags: [
          { tag: "portrait", count: 24 },
          { tag: "cat", count: 22 },
          { tag: "warm-tone", count: 18 },
          { tag: "indoor", count: 16 },
          { tag: "studio-light", count: 14 },
          { tag: "close-up", count: 12 },
          { tag: "illustration", count: 11 },
          { tag: "landscape", count: 9 },
        ],
      });
      return;
    }
    if (url.pathname === "/api/images") {
      requestOrder.push("images");
      const limit = Number(url.searchParams.get("limit") || 24);
      const offset = Number(url.searchParams.get("offset") || 0);
      const order = url.searchParams.get("order") || "";
      const seed = url.searchParams.get("seed") || "";
      imagesRequests.push({ limit, offset, order, seed });
      jsonResponse(res, 200, {
        images: sampleImages.slice(offset, offset + limit),
        total: sampleImages.length,
      });
      return;
    }
    if (url.pathname === "/api/videos") {
      const limit = Number(url.searchParams.get("limit") || 24);
      const offset = Number(url.searchParams.get("offset") || 0);
      const order = url.searchParams.get("order") || "";
      const seed = url.searchParams.get("seed") || "";
      videosRequests.push({ limit, offset, order, seed });
      jsonResponse(res, 200, {
        videos: sampleVideos.slice(offset, offset + limit),
        total: sampleVideos.length,
      });
      return;
    }
    if (url.pathname === "/api/search/text") {
      // A query with no matches, for the indexing-progress empty state
      // (meta/issues/080). The stub stats report incomplete embedding.
      if (url.searchParams.get("q") === "tofu") {
        jsonResponse(res, 200, { results: [], total: 0, debug: { duration_ms: 4 } });
        return;
      }
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
      const positiveImageIds = (url.searchParams.get("positive_image_ids") || "")
        .split(",")
        .map(Number)
        .filter((n) => !Number.isNaN(n) && n > 0);
      const softNegativeImageIds = (url.searchParams.get("soft_negative_image_ids") || "")
        .split(",")
        .map(Number)
        .filter((n) => !Number.isNaN(n) && n > 0);
      const limit = Number(url.searchParams.get("limit") || 12);
      similarVideoRequests.push({
        videoId,
        seenIds,
        preferTags,
        avoidTags,
        positiveImageIds,
        softNegativeImageIds,
        limit,
      });
      if (similarVideoFailuresRemaining > 0) {
        similarVideoFailuresRemaining -= 1;
        jsonResponse(res, 503, { error: "temporary similar-video outage" });
        return;
      }
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
      const offset = Number(url.searchParams.get("offset") || 0);
      const limit = Number(url.searchParams.get("limit") || 24);
      tagSearchRequests.push({ tags, mode, offset, limit });
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
  await page.addInitScript(() => {
    const originalCanPlayType = HTMLMediaElement.prototype.canPlayType;
    HTMLMediaElement.prototype.canPlayType = function canPlayType(type) {
      if (String(type).toLowerCase().startsWith("video/webm")) return "";
      return originalCanPlayType.call(this, type);
    };
  });
  page.on("pageerror", (err) => {
    throw new Error(`atelier pageerror: ${err.message}`);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      if (
        expectedFetchFailureConsoleMessages > 0 &&
        msg.text().includes("Failed to load resource") &&
        msg.text().includes("status of 503")
      ) {
        expectedFetchFailureConsoleMessages -= 1;
        return;
      }
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
  const firstPinText = (await page.locator("[data-pin]").first().textContent() || "").trim();
  if (!firstPinText.includes("card title")) {
    throw new Error(`expected explicit annotation title on first pin, got ${JSON.stringify(firstPinText)}`);
  }
  const firstPinSummary = (await page.locator("[data-pin-summary]").first().textContent() || "").trim();
  if (!firstPinSummary.includes("overview summary")) {
    throw new Error(`expected annotation summary on a card, got ${JSON.stringify(firstPinSummary)}`);
  }
  const headline = (await page.locator("h1").first().textContent() || "").trim();
  if (headline !== "Library") {
    throw new Error(`expected library headline, got ${JSON.stringify(headline)}`);
  }

  // 1a*. The quick-row signals horizontal overflow and supports wheel
  //      scrolling, instead of clipping chips with a hidden scrollbar
  //      (meta/issues/082).
  const quickRow = page.locator('nav[aria-label="Quick collections"]');
  await page.waitForFunction(
    () => document.querySelectorAll('nav[aria-label="Quick collections"] button').length >= 9,
    {},
    { timeout: 5000 },
  );
  const quickOverflows = await quickRow.evaluate((el) => el.scrollWidth > el.clientWidth);
  if (!quickOverflows) {
    throw new Error("expected the stub's nine quick-row chips to overflow the container");
  }
  await page.locator("[data-quick-overflow]").waitFor({ state: "visible", timeout: 5000 });
  await quickRow.hover();
  await page.mouse.wheel(0, 240);
  await page.waitForFunction(
    () => (document.querySelector('nav[aria-label="Quick collections"]')?.scrollLeft ?? 0) > 0,
    {},
    { timeout: 5000 },
  );

  // 1a. Results meta is a live region so assistive tech hears search/filter
  //     updates (meta/issues/088).
  const resultsMetaRole = await page.locator("[data-results-meta]").getAttribute("role");
  if (resultsMetaRole !== "status") {
    throw new Error(`expected results meta to be a role=status live region, got ${JSON.stringify(resultsMetaRole)}`);
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
  const initialImagesRequest = imagesRequests[0];
  if (initialImagesRequest.order !== "random" || !initialImagesRequest.seed) {
    throw new Error(
      `expected initial library request to use seeded random order, got ${JSON.stringify(initialImagesRequest)}`,
    );
  }
  const mediaSelect = page.locator("[data-library-media]");
  const initialMedia = await mediaSelect.inputValue();
  if (initialMedia !== "all") {
    throw new Error(`expected library media filter to default to all, got ${JSON.stringify(initialMedia)}`);
  }
  const initialVideosRequest = videosRequests[0];
  if (!initialVideosRequest || initialVideosRequest.order !== "random" || !initialVideosRequest.seed) {
    throw new Error(
      `expected initial library request to include seeded random videos, got ${JSON.stringify(initialVideosRequest)}`,
    );
  }

  // Library view no longer renders the stats pane inline — it now lives on
  // its own page reachable from the Rail.
  if ((await page.locator("[data-stats-pane]").count()) > 0) {
    throw new Error("expected library view not to render the stats pane inline");
  }

  // Stats page renders all sections in the new dedicated view.
  await page.locator('button[aria-label="Statistics"]').click();
  await page.waitForURL(/view=stats/, { timeout: 5000 });
  const statsPane = page.locator("[data-stats-pane]");
  await statsPane.waitFor({ state: "visible", timeout: 5000 });
  // Breadcrumb must name the stats view, not fall through to "Similar"
  // (meta/issues/075).
  const statsCrumb = (await page.locator("header p").first().textContent() || "").trim();
  if (!statsCrumb.includes("Statistics") || statsCrumb.includes("Similar")) {
    throw new Error(`expected stats breadcrumb to name Statistics, got ${JSON.stringify(statsCrumb)}`);
  }
  const statsText = (await statsPane.textContent()) || "";
  for (const expected of [
    "Statistics",
    "Media ingested",
    "144",
    "standalone images",
    "7",
    "videos",
    "18",
    "video frames",
    "Image embedding",
    "72 / 162 processed",
    "Image annotation",
    "85 / 144 processed",
    "Video annotation",
    "4 / 7 processed",
    "Video transcription",
    "3 / 7 processed",
    "Recent failures",
    "broken-decode.jpg",
    "silent-track.mp4",
    "Most frequent tags",
    "portrait",
    "cat",
  ]) {
    if (!statsText.includes(expected)) {
      throw new Error(`expected stats page to include ${JSON.stringify(expected)}, got ${JSON.stringify(statsText)}`);
    }
  }
  // The pane refreshes itself while open: the stub bumps the embed queue's
  // pending count on every /api/stats fetch, so the rendered number must
  // change without a reload (meta/issues/081).
  const queuedBefore = await statsPane.evaluate((el) => {
    const match = (el.textContent || "").match(/(\d+) queued/);
    return match ? Number(match[1]) : -1;
  });
  if (queuedBefore < 0) {
    throw new Error("expected an embed queued count on the stats pane");
  }
  await page.waitForFunction(
    (before) => {
      const pane = document.querySelector("[data-stats-pane]");
      const match = (pane?.textContent || "").match(/(\d+) queued/);
      return match ? Number(match[1]) > before : false;
    },
    queuedBefore,
    { timeout: 10000 },
  );

  // Returning to library clears the stats pane and the view= URL param.
  await page.locator('button[aria-label="Library"]').click();
  await page.waitForFunction(
    () => !window.location.search.includes("view=stats"),
    {},
    { timeout: 5000 },
  );
  if ((await page.locator("[data-stats-pane]").count()) > 0) {
    throw new Error("expected library view not to render the stats pane after returning from stats");
  }
  // Polling must stop once the stats view is left (meta/issues/081).
  const statsServedAfterLeave = statsServed;
  await new Promise((resolve) => setTimeout(resolve, 3500));
  if (statsServed > statsServedAfterLeave) {
    throw new Error(
      `expected stats polling to stop after leaving the view, served ${statsServed - statsServedAfterLeave} more`,
    );
  }

  const imageOnlyStart = imagesRequests.length;
  await mediaSelect.selectOption("images");
  const imageOnlyDeadline = Date.now() + 5000;
  while (imagesRequests.length <= imageOnlyStart && Date.now() < imageOnlyDeadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  await page.waitForFunction(
    () => {
      const pins = Array.from(document.querySelectorAll("[data-pin]"));
      return pins.length > 0 && pins.every((pin) => pin.getAttribute("data-pin-media-type") === "image");
    },
    {},
    { timeout: 5000 },
  );
  const imageOnlyTypes = await page
    .locator("[data-pin]")
    .evaluateAll((pins) => pins.map((pin) => pin.getAttribute("data-pin-media-type")));
  if (imageOnlyTypes.length === 0 || imageOnlyTypes.some((type) => type !== "image")) {
    throw new Error(`expected image-only library pins, got ${JSON.stringify(imageOnlyTypes)}`);
  }

  const videoOnlyStart = videosRequests.length;
  await mediaSelect.selectOption("videos");
  const videoOnlyDeadline = Date.now() + 5000;
  while (videosRequests.length <= videoOnlyStart && Date.now() < videoOnlyDeadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  await page.waitForFunction(
    () => {
      const pins = Array.from(document.querySelectorAll("[data-pin]"));
      return pins.length > 0 && pins.every((pin) => pin.getAttribute("data-pin-media-type") === "video");
    },
    {},
    { timeout: 5000 },
  );
  const videoOnlyTypes = await page
    .locator("[data-pin]")
    .evaluateAll((pins) => pins.map((pin) => pin.getAttribute("data-pin-media-type")));
  if (videoOnlyTypes.length === 0 || videoOnlyTypes.some((type) => type !== "video")) {
    throw new Error(`expected video-only library pins, got ${JSON.stringify(videoOnlyTypes)}`);
  }
  const videoFeedActions = await page.locator('[data-pin-media-type="video"] [data-pin-action="feed"]').count();
  const videoPlayFallbacks = await page.locator('[data-pin-media-type="video"] [aria-label="Play video"]').count();
  if (videoFeedActions !== videoOnlyTypes.length || videoPlayFallbacks !== 0) {
    throw new Error(
      `expected every video pin to expose Feed, got feed=${videoFeedActions} play=${videoPlayFallbacks} videos=${videoOnlyTypes.length}`,
    );
  }

  const allMediaImageStart = imagesRequests.length;
  const allMediaVideoStart = videosRequests.length;
  await mediaSelect.selectOption("all");
  const allMediaDeadline = Date.now() + 5000;
  while (
    (imagesRequests.length <= allMediaImageStart || videosRequests.length <= allMediaVideoStart) &&
    Date.now() < allMediaDeadline
  ) {
    await new Promise((r) => setTimeout(r, 50));
  }
  await page.waitForFunction(
    () => {
      const types = new Set(
        Array.from(document.querySelectorAll("[data-pin]")).map((pin) => pin.getAttribute("data-pin-media-type")),
      );
      return types.has("image") && types.has("video");
    },
    {},
    { timeout: 5000 },
  );
  const allMediaTypes = await page
    .locator("[data-pin]")
    .evaluateAll((pins) => Array.from(new Set(pins.map((pin) => pin.getAttribute("data-pin-media-type")))));
  if (!allMediaTypes.includes("image") || !allMediaTypes.includes("video")) {
    throw new Error(`expected mixed image/video library pins, got ${JSON.stringify(allMediaTypes)}`);
  }

  const sortSelect = page.locator("[data-library-sort]");
  const initialSort = await sortSelect.inputValue();
  if (initialSort !== "random") {
    throw new Error(`expected library sort to default to random, got ${JSON.stringify(initialSort)}`);
  }
  const requestsBeforeNewestSort = imagesRequests.length;
  await sortSelect.selectOption("newest");
  const newestSortDeadline = Date.now() + 5000;
  while (imagesRequests.length <= requestsBeforeNewestSort && Date.now() < newestSortDeadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  const newestSortRequest = imagesRequests[imagesRequests.length - 1];
  if (newestSortRequest.offset !== 0 || newestSortRequest.order !== "newest" || newestSortRequest.seed) {
    throw new Error(`expected recently-added sort to request newest first page, got ${JSON.stringify(newestSortRequest)}`);
  }
  const requestsBeforeRandomSort = imagesRequests.length;
  await sortSelect.selectOption("random");
  const randomSortDeadline = Date.now() + 5000;
  while (imagesRequests.length <= requestsBeforeRandomSort && Date.now() < randomSortDeadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  const randomSortRequest = imagesRequests[imagesRequests.length - 1];
  if (randomSortRequest.offset !== 0 || randomSortRequest.order !== "random" || !randomSortRequest.seed) {
    throw new Error(`expected random sort to request seeded random first page, got ${JSON.stringify(randomSortRequest)}`);
  }

  // 1b. Rail Feed launcher — starts from a random video.
  const baselineRailSimilarVideos = similarVideoRequests.length;
  const baselineRailVideos = videosRequests.length;
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
  if (videosRequests.length <= baselineRailVideos) {
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

  // 1c. A transient similar-video fetch error must not present as true
  //     exhaustion. It should keep the Feed session open and expose a retry
  //     path that can fetch the first successful batch.
  similarVideoFailuresRemaining = 1;
  expectedFetchFailureConsoleMessages = 1;
  const baselineErroredFeedRequests = similarVideoRequests.length;
  await page.locator('button[aria-label^="Feed"]').click();
  await page.locator("[data-feed-overlay]").waitFor({ state: "visible", timeout: 5000 });
  await page.locator("[data-feed-error]").waitFor({ state: "visible", timeout: 5000 });
  const erroredFeedExhausted = await page.locator("[data-feed-overlay]").getAttribute("data-feed-exhausted");
  if (erroredFeedExhausted === "true") {
    throw new Error("expected transient Feed fetch error not to mark the feed exhausted");
  }
  if ((await page.locator("[data-feed-end]").count()) > 0) {
    throw new Error("expected transient Feed fetch error not to render the end-of-feed state");
  }
  if (similarVideoRequests.length !== baselineErroredFeedRequests + 1) {
    throw new Error(
      `expected one failing similar-videos request, got ${JSON.stringify(similarVideoRequests.slice(baselineErroredFeedRequests))}`,
    );
  }
  await page.locator("[data-feed-error-retry]").click();
  await page.waitForFunction(
    () => {
      const overlay = document.querySelector("[data-feed-overlay]");
      const size = Number(overlay?.getAttribute("data-feed-queue-size") || 0);
      return size > 1;
    },
    {},
    { timeout: 5000 },
  );
  if (similarVideoRequests.length < baselineErroredFeedRequests + 2) {
    throw new Error(
      `expected retry to issue another similar-videos request, got ${JSON.stringify(similarVideoRequests.slice(baselineErroredFeedRequests))}`,
    );
  }
  await page.keyboard.press("Escape");
  await page.locator("[data-feed-overlay]").waitFor({ state: "hidden", timeout: 5000 });

  // 2. Search flow.
  await page.locator("#atelier-search").fill("warm portrait");
  const searchInputType = await page.locator("#atelier-search").getAttribute("type");
  if (searchInputType !== "text") {
    throw new Error(`expected Atelier search input to avoid the native browser clear button, got type=${JSON.stringify(searchInputType)}`);
  }
  const searchInputRole = await page.locator("#atelier-search").getAttribute("role");
  if (searchInputRole !== "searchbox") {
    throw new Error(`expected Atelier search input to preserve searchbox semantics, got role=${JSON.stringify(searchInputRole)}`);
  }
  const searchInputLabel = await page.locator("#atelier-search").getAttribute("aria-label");
  if (searchInputLabel !== "Search library") {
    throw new Error(`expected Atelier search input to have an accessible label, got aria-label=${JSON.stringify(searchInputLabel)}`);
  }
  const clearButtonCount = await page.locator('button[aria-label="Clear search"]').count();
  if (clearButtonCount !== 1) {
    throw new Error(`expected one custom clear search button, got ${clearButtonCount}`);
  }
  const valueAfterComposingEscape = await page.locator("#atelier-search").evaluate((input) => {
    const event = new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true });
    Object.defineProperty(event, "isComposing", { value: true });
    input.dispatchEvent(event);
    return input.value;
  });
  if (valueAfterComposingEscape !== "warm portrait") {
    throw new Error(`expected composing Escape to preserve search input, got ${JSON.stringify(valueAfterComposingEscape)}`);
  }
  await page.keyboard.press("Escape");
  await page.waitForFunction(
    () => window.location.search === "" && document.querySelector("#atelier-search")?.value === "",
    {},
    { timeout: 5000 },
  );
  const clearButtonCountAfterEscape = await page.locator('button[aria-label="Clear search"]').count();
  if (clearButtonCountAfterEscape !== 0) {
    throw new Error(`expected Escape to hide the clear search button, got ${clearButtonCountAfterEscape}`);
  }
  await page.locator("#atelier-search").fill("warm portrait");
  await page.locator('button[aria-label="Clear search"]').click();
  await page.waitForFunction(() => document.querySelector("#atelier-search")?.value === "", {}, { timeout: 5000 });
  const clearButtonCountAfterClick = await page.locator('button[aria-label="Clear search"]').count();
  if (clearButtonCountAfterClick !== 0) {
    throw new Error(`expected clicking clear to hide the clear search button, got ${clearButtonCountAfterClick}`);
  }
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

  await page.locator("#atelier-search").focus();
  await page.keyboard.press("Escape");
  await page.waitForFunction(
    () => window.location.search === "" && document.querySelector("#atelier-search")?.value === "",
    {},
    { timeout: 5000 },
  );
  const clearedSearchHeadline = (await page.locator("h1").first().textContent() || "").trim();
  if (clearedSearchHeadline !== "Library") {
    throw new Error(`expected Escape to return to library mode, got ${JSON.stringify(clearedSearchHeadline)}`);
  }
  await page.locator("#atelier-search").fill("warm portrait");
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => /\?q=/.test(window.location.search), {}, { timeout: 5000 });
  await page.locator("[data-pin-match]").first().waitFor({ state: "visible", timeout: 5000 });

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
  if (firstFeedRequest.positiveImageIds.length !== 0 || firstFeedRequest.softNegativeImageIds.length !== 0) {
    throw new Error(
      `expected initial Feed request to have no vector feedback ids, got ${JSON.stringify(firstFeedRequest)}`,
    );
  }

  // Transport controls use directional glyphs and a stateful play/pause icon
  // (meta/issues/076).
  const prevIconName = await page.locator("[data-feed-prev] svg").getAttribute("data-icon");
  const nextIconName = await page.locator("[data-feed-next] svg").getAttribute("data-icon");
  if (prevIconName !== "chevron-up" || nextIconName !== "chevron-down") {
    throw new Error(`expected directional feed nav icons, got prev=${prevIconName} next=${nextIconName}`);
  }
  const pausedIconName = await page.locator("[data-feed-playpause] svg").getAttribute("data-icon");
  if (pausedIconName !== "play") {
    throw new Error(`expected play glyph while feed video is not playing, got ${pausedIconName}`);
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
      return idx === before + 1;
    },
    beforeIdx,
    { timeout: 5000 },
  );
  const firstCandidateSrc = await page.locator('video[data-feed-current="true"]').evaluate((video) => {
    video.pause();
    return video.getAttribute("src") || "";
  });
  if (!firstCandidateSrc.includes("/media/videos/clip-1")) {
    throw new Error(`expected first Feed advance to land on clip-1, got ${JSON.stringify(firstCandidateSrc)}`);
  }

  // Synthetic playback events classify the current item as positive and should
  // affect only a later lookahead request, not the current/next queue entries.
  await page.evaluate(() => {
    const video = document.querySelector('video[data-feed-current="true"]');
    if (!video) throw new Error("current Feed video missing");
    Object.defineProperty(video, "duration", { value: 1, configurable: true });
    video.currentTime = 0.9;
    video.dispatchEvent(new Event("play"));
  });
  // The play/pause glyph reflects playback state (meta/issues/076).
  await page.waitForFunction(
    () => document.querySelector("[data-feed-playpause] svg")?.getAttribute("data-icon") === "pause",
    {},
    { timeout: 5000 },
  );
  await page.keyboard.press("ArrowDown");
  await page.waitForFunction(
    () => Number(document.querySelector("[data-feed-overlay]")?.getAttribute("data-feed-current-index") || 0) === 2,
    {},
    { timeout: 5000 },
  );
  const rerankedBufferedSrc = await page.locator('video[data-feed-current="true"]').getAttribute("src");
  if (!rerankedBufferedSrc?.includes("/media/videos/clip-4")) {
    throw new Error(`expected buffered future queue to promote clip-4 after feedback, got ${JSON.stringify(rerankedBufferedSrc)}`);
  }
  const vectorFeedbackDeadline = Date.now() + 5000;
  while (similarVideoRequests.length <= baselineSimilarVideos + 1 && Date.now() < vectorFeedbackDeadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  const vectorFeedbackRequest = similarVideoRequests.slice(baselineSimilarVideos + 1).find((request) => request.positiveImageIds.length > 0);
  if (!vectorFeedbackRequest) {
    throw new Error(
      `expected a later Feed request to include positive_image_ids, got ${JSON.stringify(similarVideoRequests.slice(baselineSimilarVideos))}`,
    );
  }
  const expectedPositiveImageId = sampleVideos.find((video) => video.video_id !== firstFeedRequest.videoId)?.image_id;
  if (
    !expectedPositiveImageId ||
    vectorFeedbackRequest.positiveImageIds.length !== 1 ||
    !vectorFeedbackRequest.positiveImageIds.includes(expectedPositiveImageId) ||
    vectorFeedbackRequest.softNegativeImageIds.includes(expectedPositiveImageId)
  ) {
    throw new Error(
      `expected positive vector feedback for image ${expectedPositiveImageId} without cross-list overlap, got ${JSON.stringify(vectorFeedbackRequest)}`,
    );
  }
  await page.keyboard.press("Escape");
  await page.locator("[data-feed-overlay]").waitFor({ state: "hidden", timeout: 5000 });

  // 3. Tag chip flow — clicks the "Tag · portrait" quick-row chip and
  //    expects to land on ?tag=portrait with no similarity badges (tag
  //    search results have search_source="tag" / distance=0).
  const searchBeforeTag = await page.evaluate(() => window.location.search);
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

  // 3a. Arrow keys and on-screen chevrons flip through the surrounding
  //     results without closing the lightbox (meta/issues/078). Tag results
  //     are deterministic and settled here, so navigate by index.
  await page.waitForFunction(
    () => document.querySelectorAll("[data-pin]").length >= 12,
    {},
    { timeout: 5000 },
  );
  await page.locator('[data-pin][data-pin-key="image:1000"] [data-pin-media]').click();
  await page.locator("[data-lightbox]").waitFor({ state: "visible", timeout: 5000 });
  const lightboxStartIndex = Number(
    await page.locator("[data-lightbox]").getAttribute("data-lightbox-index"),
  );
  await page.keyboard.press("ArrowRight");
  await page.waitForFunction(
    (start) =>
      Number(document.querySelector("[data-lightbox]")?.getAttribute("data-lightbox-index")) ===
      start + 1,
    lightboxStartIndex,
    { timeout: 5000 },
  );
  await page.locator("[data-lightbox-prev]").click();
  await page.waitForFunction(
    (start) =>
      Number(document.querySelector("[data-lightbox]")?.getAttribute("data-lightbox-index")) ===
      start,
    lightboxStartIndex,
    { timeout: 5000 },
  );
  if (lightboxStartIndex === 0) {
    if (!(await page.locator("[data-lightbox-prev]").isDisabled())) {
      throw new Error("expected prev control to be disabled on the first result");
    }
  }
  if (await page.locator("[data-lightbox-next]").isDisabled()) {
    throw new Error("expected next control to be enabled mid-list");
  }
  await page.keyboard.press("Escape");
  await page.locator("[data-lightbox]").waitFor({ state: "hidden", timeout: 5000 });

  // 3b. In-app navigation creates history entries: Back returns to the view
  //     before the tag click instead of leaving the site, Forward restores
  //     the tag view (meta/issues/074).
  await page.goBack();
  await page.waitForFunction(
    (prev) => window.location.search === prev && window.location.port !== "",
    searchBeforeTag,
    { timeout: 5000 },
  );
  await page.goForward();
  await page.waitForFunction(
    () => /\?tag=portrait/.test(window.location.search),
    {},
    { timeout: 5000 },
  );
  const tagHeadlineAfterForward = (await page.locator("h1").first().textContent() || "").trim();
  if (tagHeadlineAfterForward !== "portrait") {
    throw new Error(`expected tag headline after history forward, got ${JSON.stringify(tagHeadlineAfterForward)}`);
  }

  // Reset to library before similar/lightbox/menu checks.
  await page.locator('a[aria-label="imgsearch home"]').click();
  await page.waitForFunction(() => window.location.search === "", {}, { timeout: 5000 });
  await page.locator("[data-pin]").first().waitFor({ state: "visible", timeout: 5000 });

  // 4. Similar flow via the corner action. Trigger from a lower card to prove
  //    the new result set scrolls back to its first image.
  const similarSeedPin = page.locator("[data-pin]").nth(20);
  await similarSeedPin.scrollIntoViewIfNeeded();
  const scrollBeforeSimilar = await page.evaluate(() => window.scrollY);
  if (scrollBeforeSimilar < 100) {
    throw new Error(`expected test page to be scrolled before Similar, got ${scrollBeforeSimilar}`);
  }
  await similarSeedPin.hover();
  await similarSeedPin.locator('[data-pin-action="similar"]').click();
  await page.waitForFunction(() => /\?similar=/.test(window.location.search), {}, { timeout: 5000 });
  await page.locator("[data-pin-anchor]").waitFor({ state: "visible", timeout: 5000 });
  await page.waitForFunction(
    () => {
      const firstPin = document.querySelector("[data-results-grid] [data-pin]");
      if (!firstPin) return false;
      const top = firstPin.getBoundingClientRect().top;
      return top >= 0 && top < Math.min(260, window.innerHeight * 0.4);
    },
    {},
    { timeout: 5000 },
  );
  const similarFirstIsAnchor = await page.locator("[data-pin]").first().getAttribute("data-pin-anchor");
  if (similarFirstIsAnchor !== "true") {
    throw new Error("expected similar search to keep the search image as the first anchored pin");
  }
  const anchorLabel = (await page.locator("[data-pin-anchor-label]").first().textContent() || "").trim();
  if (anchorLabel !== "Search image") {
    throw new Error(`expected clear similar-search anchor label, got ${JSON.stringify(anchorLabel)}`);
  }
  const similarHeadline = (await page.locator("h1").first().textContent() || "").trim();
  if (similarHeadline !== "Similar in your library") {
    throw new Error(`expected similar headline, got ${JSON.stringify(similarHeadline)}`);
  }

  // Back to library for lightbox / menu / load-more.
  await page.locator('a[aria-label="imgsearch home"]').click();
  await page.waitForFunction(() => window.location.search === "", {}, { timeout: 5000 });
  await page.locator("[data-pin]").first().waitFor({ state: "visible", timeout: 5000 });

  // 5. Lightbox exposes full descriptions and clickable tags.
  await page.locator('[data-pin-media]').first().click();
  await page.locator("[data-lightbox]").waitFor({ state: "visible", timeout: 5000 });
  const lightboxDescription = (await page.locator("[data-lightbox-description]").textContent() || "").trim();
  if (!lightboxDescription.includes("Full generated annotation")) {
    throw new Error(`expected full generated annotation in lightbox, got ${JSON.stringify(lightboxDescription)}`);
  }
  const lightboxTag = (await page.locator("[data-lightbox-tag]").first().textContent() || "").trim();
  if (!lightboxTag) {
    throw new Error("expected at least one clickable lightbox tag");
  }

  await page.locator("[data-lightbox-tag]").first().click();
  await page.waitForFunction(
    (tag) => new URLSearchParams(window.location.search).getAll("tag").includes(tag),
    lightboxTag,
    { timeout: 5000 },
  );
  await page.locator("[data-lightbox]").waitFor({ state: "hidden", timeout: 5000 });
  const lightboxTagHeadline = (await page.locator("h1").first().textContent() || "").trim();
  if (lightboxTagHeadline !== lightboxTag) {
    throw new Error(`expected lightbox tag headline ${JSON.stringify(lightboxTag)}, got ${JSON.stringify(lightboxTagHeadline)}`);
  }

  // Back to library and keep the Escape-close path covered. Along the way,
  // dialogs must take focus, trap Tab, and restore focus to their opener on
  // close (meta/issues/086).
  await page.locator('a[aria-label="imgsearch home"]').click();
  await page.waitForFunction(() => window.location.search === "", {}, { timeout: 5000 });
  // Wait for the full library page so the opener pin is not replaced by an
  // in-flight refetch while the dialog is open (focus must return to it).
  await page.waitForFunction(
    () => document.querySelectorAll("[data-pin]").length >= 40,
    {},
    { timeout: 5000 },
  );
  await page.locator('[data-pin][data-pin-key="image:1000"] [data-pin-media]').click();
  await page.locator("[data-lightbox]").waitFor({ state: "visible", timeout: 5000 });
  const focusInsideOnOpen = await page.evaluate(
    () => Boolean(document.querySelector("[data-lightbox]")?.contains(document.activeElement)),
  );
  if (!focusInsideOnOpen) {
    throw new Error("expected focus to move into the lightbox dialog on open");
  }
  for (let i = 0; i < 10; i += 1) {
    await page.keyboard.press("Tab");
    const stillInside = await page.evaluate(
      () => Boolean(document.querySelector("[data-lightbox]")?.contains(document.activeElement)),
    );
    if (!stillInside) {
      throw new Error(`expected Tab to stay trapped inside the lightbox (escaped on press ${i + 1})`);
    }
  }
  await page.keyboard.press("Escape");
  await page.locator("[data-lightbox]").waitFor({ state: "hidden", timeout: 5000 });
  const focusRestored = await page.evaluate(() => {
    const active = document.activeElement;
    return Boolean(active && active.closest("[data-pin]"));
  });
  if (!focusRestored) {
    throw new Error("expected focus to return to the opening pin after the lightbox closes");
  }

  // 5b. Mobile lightbox layout — on a phone-sized viewport the description
  //     must stay inside the modal card and must not render on top of the
  //     media. The image must also be visually contained within the dark
  //     media cell, not overflowing into the description area.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('a[aria-label="imgsearch home"]').click();
  await page.waitForFunction(() => window.location.search === "", {}, { timeout: 5000 });
  // 5b*. Mobile chrome stays compact: the quick-row keeps to a single
  //      scrollable line and the first pin starts within the upper 60% of
  //      the viewport (meta/issues/077).
  const mobileQuickRowHeight = await page
    .locator('nav[aria-label="Quick collections"]')
    .evaluate((el) => el.getBoundingClientRect().height);
  if (mobileQuickRowHeight > 60) {
    throw new Error(`expected a single-line mobile quick-row, got height ${mobileQuickRowHeight}px`);
  }
  // Measure the chrome via the static results-section offset; pin positions
  // churn while the masonry re-measures and would make this flaky.
  const resultsTop = await page.evaluate(() => {
    const el = document.querySelector("[data-results]");
    return el ? el.getBoundingClientRect().top + window.scrollY : Infinity;
  });
  if (resultsTop > 844 * 0.6) {
    throw new Error(`expected results to start within the upper 60% of a phone viewport, got top ${resultsTop}px`);
  }

  // image:1000 carries the deliberately long title, which the layout and
  // title-clamp checks below depend on.
  await page.locator('[data-pin][data-pin-key="image:1000"]').waitFor({ state: "attached", timeout: 5000 });
  await page.locator('[data-pin][data-pin-key="image:1000"] [data-pin-media]').click();
  await page.locator("[data-lightbox]").waitFor({ state: "visible", timeout: 5000 });
  const lightboxMetrics = await page.evaluate(() => {
    const lightbox = document.querySelector("[data-lightbox]");
    const card = lightbox?.querySelector(":scope > div");
    const description = document.querySelector("[data-lightbox-description]");
    const media = lightbox?.querySelector("img, video");
    if (!lightbox || !card || !description || !media) {
      throw new Error("lightbox structure missing required nodes");
    }
    const cardRect = card.getBoundingClientRect();
    const descRect = description.getBoundingClientRect();
    const mediaRect = media.getBoundingClientRect();
    return {
      card: { top: cardRect.top, bottom: cardRect.bottom, left: cardRect.left, right: cardRect.right },
      desc: { top: descRect.top, bottom: descRect.bottom, left: descRect.left, right: descRect.right },
      media: { top: mediaRect.top, bottom: mediaRect.bottom, left: mediaRect.left, right: mediaRect.right },
    };
  });
  if (lightboxMetrics.desc.top < lightboxMetrics.card.top - 1) {
    throw new Error(
      `mobile lightbox: description extends above modal card (desc top ${lightboxMetrics.desc.top}, card top ${lightboxMetrics.card.top})`,
    );
  }
  if (lightboxMetrics.desc.bottom > lightboxMetrics.card.bottom + 1) {
    throw new Error(
      `mobile lightbox: description extends below modal card (desc bottom ${lightboxMetrics.desc.bottom}, card bottom ${lightboxMetrics.card.bottom})`,
    );
  }
  if (lightboxMetrics.desc.left < lightboxMetrics.card.left - 1 || lightboxMetrics.desc.right > lightboxMetrics.card.right + 1) {
    throw new Error(
      `mobile lightbox: description extends outside modal card horizontally (desc ${JSON.stringify(lightboxMetrics.desc)}, card ${JSON.stringify(lightboxMetrics.card)})`,
    );
  }
  if (lightboxMetrics.media.bottom > lightboxMetrics.desc.top + 1) {
    throw new Error(
      `mobile lightbox: media overlaps description (media bottom ${lightboxMetrics.media.bottom}, desc top ${lightboxMetrics.desc.top})`,
    );
  }
  // 5b'. Sentence-length derived titles must not balloon into a huge
  //      multi-line headline on phones (meta/issues/091).
  const titleMetrics = await page.locator("[data-lightbox] h2").evaluate((el) => {
    const styles = getComputedStyle(el);
    return { height: el.clientHeight, lineHeight: parseFloat(styles.lineHeight) };
  });
  if (titleMetrics.height > titleMetrics.lineHeight * 3.5) {
    throw new Error(
      `mobile lightbox: title taller than 3 lines (${JSON.stringify(titleMetrics)})`,
    );
  }
  await page.keyboard.press("Escape");
  await page.locator("[data-lightbox]").waitFor({ state: "hidden", timeout: 5000 });
  // Reset the viewport for the remaining desktop checks.
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.locator("[data-pin]").first().waitFor({ state: "visible", timeout: 5000 });

  // 5c. Desktop lightbox close button must not overlap the title even when
  //     the title wraps to multiple lines in the narrow description column.
  await page.locator('[data-pin-media]').first().click();
  await page.locator("[data-lightbox]").waitFor({ state: "visible", timeout: 5000 });
  const desktopOverlap = await page.evaluate(() => {
    const lightbox = document.querySelector("[data-lightbox]");
    const card = lightbox?.querySelector(":scope > div");
    const closeBtn = lightbox?.querySelector('button[aria-label="Close"]');
    const heading = card?.querySelector("h2");
    if (!lightbox || !card || !closeBtn || !heading) {
      throw new Error("desktop lightbox: missing required nodes");
    }
    const closeRect = closeBtn.getBoundingClientRect();
    // Use Range.getClientRects() so we measure the actual rendered text glyphs
    // across every wrapped line, not the heading's full padded box.
    const range = document.createRange();
    range.selectNodeContents(heading);
    const lineRects = Array.from(range.getClientRects());
    if (lineRects.length === 0) {
      throw new Error("desktop lightbox: heading produced no client rects");
    }
    const textLeft = Math.min(...lineRects.map((r) => r.left));
    const textRight = Math.max(...lineRects.map((r) => r.right));
    const textTop = Math.min(...lineRects.map((r) => r.top));
    const textBottom = Math.max(...lineRects.map((r) => r.bottom));
    // The close button is positioned in the modal's top-right corner and the
    // title text must never run underneath it.
    const horizontallyOverlapping = closeRect.left < textRight && closeRect.right > textLeft;
    const verticallyOverlapping = closeRect.top < textBottom && closeRect.bottom > textTop;
    return {
      close: { left: closeRect.left, right: closeRect.right, top: closeRect.top, bottom: closeRect.bottom },
      text: { left: textLeft, right: textRight, top: textTop, bottom: textBottom },
      horizontallyOverlapping,
      verticallyOverlapping,
      lineCount: lineRects.length,
    };
  });
  if (desktopOverlap.horizontallyOverlapping && desktopOverlap.verticallyOverlapping) {
    throw new Error(
      `desktop lightbox: close button overlaps wrapped title (close ${JSON.stringify(desktopOverlap.close)}, text ${JSON.stringify(desktopOverlap.text)}, lines ${desktopOverlap.lineCount})`,
    );
  }
  if (desktopOverlap.overlapsHorizontally) {
    throw new Error(
      `desktop lightbox: close button overlaps wrapped title (close ${JSON.stringify(desktopOverlap.close)}, heading ${JSON.stringify(desktopOverlap.heading)})`,
    );
  }
  await page.keyboard.press("Escape");
  await page.locator("[data-lightbox]").waitFor({ state: "hidden", timeout: 5000 });

  // 6. Pin overflow menu — Re-annotate hits the API; Delete drops the pin.
  const targetPin = page.locator("[data-pin]").first();
  const beforeFirst = await targetPin.evaluate((el) => el.getAttribute("data-pin-anchor"));
  if (beforeFirst === "true") {
    throw new Error("smoke test assumed first library pin is not an anchor");
  }
  const initialCount = await page.locator("[data-pin]").count();

  // 6a. Clicking outside an open overflow menu closes it (meta/issues/083).
  await targetPin.hover();
  await targetPin.locator('[data-pin-action="more"]').click();
  await targetPin.locator('[data-pin-menu="reannotate"]').waitFor({ state: "visible", timeout: 5000 });
  await page.locator("h1").first().click({ position: { x: 4, y: 4 } });
  await page.waitForFunction(
    () => !document.querySelector("[data-pin] details[open]"),
    {},
    { timeout: 5000 },
  );

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

  // Delete asks via the in-app confirmation dialog, not window.confirm
  // (meta/issues/090). Cancel leaves the pin untouched...
  const targetPin2 = page.locator("[data-pin]").first();
  await targetPin2.hover();
  await targetPin2.locator('[data-pin-action="more"]').click();
  await targetPin2.locator('[data-pin-menu="delete"]').click();
  await page.locator("[data-confirm-dialog]").waitFor({ state: "visible", timeout: 5000 });
  await page.locator("[data-confirm-cancel]").click();
  await page.locator("[data-confirm-dialog]").waitFor({ state: "hidden", timeout: 5000 });
  if ((await page.locator("[data-pin]").count()) !== initialCount) {
    throw new Error("expected cancelling the delete dialog to keep the pin");
  }
  if (deleteCount !== 0) {
    throw new Error(`expected no DELETE after cancel, got ${deleteCount}`);
  }
  // ...and confirming removes it.
  await targetPin2.hover();
  await targetPin2.locator('[data-pin-action="more"]').click();
  await targetPin2.locator('[data-pin-menu="delete"]').click();
  await page.locator("[data-confirm-dialog]").waitFor({ state: "visible", timeout: 5000 });
  await page.locator("[data-confirm-accept]").click();
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
  const libraryRequestBeforeLoadMore = imagesRequests[imagesRequests.length - 1];
  const videoRequestBeforeLoadMore = videosRequests[videosRequests.length - 1];
  await page.locator("[data-load-more]").click();
  await page.waitForFunction(
    (before) => document.querySelectorAll("[data-pin]").length > before,
    beforeLoadMore,
    { timeout: 5000 },
  );
  if (imagesRequests.length < 2 || videosRequests.length < 2) {
    throw new Error(`expected image and video requests after load-more, got images=${imagesRequests.length} videos=${videosRequests.length}`);
  }
  const lastRequest = imagesRequests[imagesRequests.length - 1];
  const lastVideoRequest = videosRequests[videosRequests.length - 1];
  if (lastRequest.offset !== 0 || lastVideoRequest.offset !== 0) {
    throw new Error(`expected mixed-media load-more to fetch from offset 0 for merging, got images=${JSON.stringify(lastRequest)} videos=${JSON.stringify(lastVideoRequest)}`);
  }
  if (lastRequest.limit <= libraryRequestBeforeLoadMore.limit || lastVideoRequest.limit <= videoRequestBeforeLoadMore.limit) {
    throw new Error(`expected mixed-media load-more to grow request limits, got images=${JSON.stringify(lastRequest)} videos=${JSON.stringify(lastVideoRequest)}`);
  }
  if (lastRequest.order !== "random" || lastRequest.seed !== libraryRequestBeforeLoadMore.seed || lastVideoRequest.order !== "random" || lastVideoRequest.seed !== videoRequestBeforeLoadMore.seed) {
    throw new Error(
      `expected load-more to keep seeded random order images=${JSON.stringify(libraryRequestBeforeLoadMore)} videos=${JSON.stringify(videoRequestBeforeLoadMore)}, got images=${JSON.stringify(lastRequest)} videos=${JSON.stringify(lastVideoRequest)}`,
    );
  }

  // 7a. Mode change after Load More must replace, not append. Regression
  //     coverage for meta/issues/056: clicking a tag chip after the user
  //     has already paged the library must reset the offset to 0 and
  //     replace the rendered pins instead of appending onto a stale
  //     library page.
  const pinsAfterLoadMore = await page.locator("[data-pin]").count();
  if (pinsAfterLoadMore <= PAGE_SIZE) {
    throw new Error(`expected Load More to grow past PAGE_SIZE, got ${pinsAfterLoadMore}`);
  }
  const tagSearchesBefore = tagSearchRequests.length;
  await page.getByRole("button", { name: "Tag · portrait" }).click();
  await page.waitForFunction(
    () => /\?tag=portrait/.test(window.location.search),
    {},
    { timeout: 5000 },
  );
  if (tagSearchRequests.length <= tagSearchesBefore) {
    throw new Error(
      `expected /api/search/tags after mode switch, got ${tagSearchRequests.length} (was ${tagSearchesBefore})`,
    );
  }
  const tagAfterLoadMore = tagSearchRequests[tagSearchRequests.length - 1];
  if (tagAfterLoadMore.offset !== 0) {
    throw new Error(
      `expected tag search after Load More to reset offset to 0, got ${JSON.stringify(tagAfterLoadMore)}`,
    );
  }
  // Wait for the tag response to actually replace the rendered pins, not
  // just be issued. If the bug regresses, the count stays at the
  // post-Load-More number and this wait times out.
  await page.waitForFunction(
    (before) => document.querySelectorAll("[data-pin]").length < before,
    pinsAfterLoadMore,
    { timeout: 5000 },
  );
  const pinsAfterModeSwitch = await page.locator("[data-pin]").count();
  if (pinsAfterModeSwitch >= pinsAfterLoadMore) {
    throw new Error(
      `expected mode switch to replace pins, got ${pinsAfterModeSwitch} (>= ${pinsAfterLoadMore} from Load More)`,
    );
  }
  const tagHeadlineAfterSwitch = (await page.locator("h1").first().textContent() || "").trim();
  if (tagHeadlineAfterSwitch !== "portrait") {
    throw new Error(`expected tag headline after mode switch, got ${JSON.stringify(tagHeadlineAfterSwitch)}`);
  }
  // Return to library so the rest of the smoke flow (Upload, etc.) starts
  // from a clean offset.
  await page.locator('a[aria-label="imgsearch home"]').click();
  await page.waitForFunction(() => window.location.search === "", {}, { timeout: 5000 });
  await page.waitForFunction(
    (pageSize) => {
      const pins = document.querySelectorAll("[data-pin]").length;
      return pins > 0 && pins <= pageSize;
    },
    PAGE_SIZE,
    { timeout: 5000 },
  );
  const pinsAfterReturn = await page.locator("[data-pin]").count();
  if (pinsAfterReturn > PAGE_SIZE) {
    throw new Error(`expected Library return to reset to a single page, got ${pinsAfterReturn} pins`);
  }

  // 7b. Upload flow — open modal via header trigger, attach two files, submit,
  //     verify per-file row states (created + duplicate), summary, and that
  //     the library re-fetches via the dataEpoch bump.
  const imagesRequestsBeforeUpload = imagesRequests.length;
  const libraryRequestBeforeUpload = imagesRequests[imagesRequests.length - 1];
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
  const uploadRefreshRequest = imagesRequests[imagesRequests.length - 1];
  if (uploadRefreshRequest.offset !== 0 || uploadRefreshRequest.order !== "random" || !uploadRefreshRequest.seed) {
    throw new Error(
      `expected upload refresh after ${JSON.stringify(libraryRequestBeforeUpload)} to replace with seeded random first page, got ${JSON.stringify(uploadRefreshRequest)}`,
    );
  }
  await page.waitForFunction(
    () => document.querySelectorAll("[data-pin]").length <= 48,
    {},
    { timeout: 5000 },
  );
  const postUploadPinCount = await page.locator("[data-pin]").count();
  if (postUploadPinCount > 48) {
    throw new Error(`expected upload refresh to replace pins, got ${postUploadPinCount} rendered pins`);
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

  // 8a. An empty search against an incompletely-embedded library explains
  //     that indexing is still running instead of a bare "No matches"
  //     (meta/issues/080). The stub stats report 72/162 embedded.
  await page.locator("#atelier-search").fill("tofu");
  await page.keyboard.press("Enter");
  await page.waitForFunction(
    () => /tofu/.test(window.location.search),
    {},
    { timeout: 5000 },
  );
  await page.locator("[data-results-empty]").waitFor({ state: "visible", timeout: 5000 });
  const emptyStateText = await page.locator("[data-results-empty]").textContent();
  if (!/still indexing/i.test(emptyStateText || "") || !/162/.test(emptyStateText || "")) {
    throw new Error(
      `expected empty search on a half-indexed library to mention indexing progress, got ${JSON.stringify(emptyStateText)}`,
    );
  }
  await page.locator('a[aria-label="imgsearch home"]').click();
  await page.waitForFunction(() => window.location.search === "", {}, { timeout: 5000 });

  // 9. View preferences persist across reloads via localStorage
  //    (meta/issues/085).
  await page.locator("[data-library-sort]").selectOption("newest");
  await page.locator("[data-library-media]").selectOption("videos");
  await page.locator("[data-nsfw-toggle]").check();
  await page.reload({ waitUntil: "networkidle" });
  const persistedSort = await page.locator("[data-library-sort]").inputValue();
  const persistedMedia = await page.locator("[data-library-media]").inputValue();
  const persistedNSFW = await page.locator("[data-nsfw-toggle]").isChecked();
  if (persistedSort !== "newest" || persistedMedia !== "videos" || !persistedNSFW) {
    throw new Error(
      `expected view preferences to survive a reload, got sort=${persistedSort} media=${persistedMedia} nsfw=${persistedNSFW}`,
    );
  }

  console.log("atelier smoke checks passed");
} finally {
  if (browser) {
    await browser.close();
  }
  server.close();
}
