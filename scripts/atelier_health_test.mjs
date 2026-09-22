import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";

const errorCopy = "The library couldn’t complete this request. Try again in a moment.";
const connectionCopy = "Couldn’t connect to the library. Check your connection, then retry.";
const record = (id) => ({ media_type: "image", image_id: id, original_name: `${id}.jpg`, storage_path: "images/sample-0", thumbnail_path: "images/thumbnails/1000.webp", mime_type: "image/jpeg", width: 800, height: 600, title: `Recovery ${id}`, tags: ["cat"], annotation_state: "done", distance: 0.2 });
const job = (overrides = {}) => ({ tracked: 5, waiting: 1, processing: 1, pending: 1, leased: 1, runnable: 1, done: 2, failed: 1, oldest_runnable_age_seconds: 5, ...overrides });
const snapshot = (state) => ({
  images_total: 15, standalone_images_total: 10, video_frame_images_total: 5, videos_total: 2,
  image_annotation_expected: 10, video_annotation_expected: 2, video_transcription_expected: 2,
  capabilities: { worker: state === "unknown" ? "unknown" : "local", annotations: state, transcription: state },
  video_transcription_empty: 1,
  job_kinds: { embed_image: job(), annotate_image: job(), annotate_video: job(), transcribe_video: job({ done: 1, failed: 1 }) },
  recent_failures: [{ job_id: 4, kind: "transcribe_video", media_type: "video", video_id: 5, original_name: "speech.mp4", attempts: 3, last_error: "raw decoder detail", updated_at: "2026-09-22" }],
});

async function pasteImage(page) {
  await page.locator('input[type="file"][accept="image/*"]').setInputFiles({ name: "health-query.png", mimeType: "image/png", buffer: Buffer.from([1, 2, 3, 4]) });
}

export async function healthChecks(browser, baseURL) {
  for (const width of [1280, 390]) {
    for (const mode of ["search", "tag", "byimage", "similar", "library"]) {
      const page = await browser.newPage({ viewport: { width, height: 900 } });
      page.setDefaultTimeout(5000);
      await page.addInitScript(() => {
        localStorage.setItem("imgsearch.includeNSFW", "true");
        localStorage.setItem("imgsearch.libraryMedia", "images");
        window.__imgsearchAnnotationPollMs = 600000;
      });
      let fail = "network";
      let paginating = false;
      const requests = [];
      const path = { search: "/api/search/text", tag: "/api/search/tags", byimage: "/api/search/by-image", similar: "/api/search/similar", library: "/api/images" }[mode];
      await page.route((url) => url.pathname === path, async (route) => {
        const request = route.request();
        requests.push({ url: request.url(), body: request.postData()?.replace(/----WebKitFormBoundary[^\r\n]+/g, "BOUNDARY") });
        if (fail === "network") return route.abort("failed");
        if (fail === "http") return route.fulfill({ status: 503, json: { error: "Failed to fetch" } });
        const offset = Number(new URL(request.url()).searchParams.get("offset") || 0);
        const records = paginating && offset === 0 ? Array.from({ length: 48 }, (_, i) => record(8000 + i)) : [record(8048)];
        return route.fulfill({ json: { [mode === "library" ? "images" : "results"]: records, total: paginating ? 49 : 1 } });
      });
      try {
        const query = { search: "?q=cat%20%26%20dog", tag: "?tag=cat&tag=portrait&tag_mode=all", byimage: "", similar: "?similar=1000", library: "" }[mode];
        await page.goto(baseURL + "/" + query);
        if (mode === "byimage") await pasteImage(page);
        const retry = page.getByRole("button", { name: "Retry", exact: true });
        await retry.waitFor();
        assert.equal(await page.getByText(connectionCopy, { exact: true }).count(), 1);
        assert.equal(await page.getByText(/Failed to fetch/).count(), 0);
        assert.equal(await page.locator("[data-results-empty]").count(), 0, "errors are not empty successes");
        const original = requests.at(-1);
        const params = new URL(original.url).searchParams;
        assert.equal(params.get("include_nsfw"), "1");
        if (mode === "search") assert.equal(params.get("q"), "cat & dog");
        if (mode === "tag") {
          assert.deepEqual(params.getAll("tag"), ["cat", "portrait"]);
          assert.equal(params.get("tag_mode"), "all");
        }
        if (mode === "byimage") assert.match(original.body, /health-query.png/);
        if (mode === "similar") assert.equal(params.get("image_id"), "1000");
        fail = "http";
        await retry.click();
        await page.getByText(errorCopy, { exact: true }).waitFor();
        assert.deepEqual(requests.at(-1), original, `${mode} preserves payload/filter on HTTP retry`);
        fail = "";
        paginating = ["library", "search", "tag"].includes(mode);
        await retry.click();
        await page.locator("[data-pin]").first().waitFor();
        assert.deepEqual(requests.at(-1), original, `${mode} preserves payload/filter on recovery`);
        assert.equal(await retry.count(), 0);
        assert.equal(await page.locator("[data-request-error]").count(), 0);
        if (paginating) {
          fail = "http";
          await page.getByRole("button", { name: "Load more" }).click();
          await retry.waitFor();
          assert.equal(await page.locator("[data-pin]").count(), 48, "append failure retains current scope");
          const append = requests.at(-1);
          assert.equal(new URL(append.url).searchParams.get("offset"), "48");
          fail = "";
          await retry.click();
          await page.locator('[data-pin-key="image:8048"]').waitFor();
          assert.deepEqual(requests.at(-1), append, "append retry preserves offset");
          assert.equal(await page.locator("[data-pin]").count(), 49);
        }
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      } finally { await page.close(); }
    }

    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(5000);
    let state = "disabled";
    let statsFail = false;
    let recoveryFail = true;
    let recoveryURL = "";
    await page.route("**/api/stats", (route) => statsFail ? route.abort() : route.fulfill({ json: snapshot(state) }));
    await page.route("**/api/jobs/retry-failed*", (route) => {
      recoveryURL = route.request().url();
      return route.fulfill(recoveryFail ? { status: 503, json: { error: "Failed to fetch" } } : { json: { retried: 1, enqueued_missing: 0 } });
    });
    try {
      for (state of ["disabled", "unconfigured", "unknown", "configured"]) {
        await page.goto(`${baseURL}/?view=stats`);
        const transcription = page.locator('[data-health-kind="transcribe_video"]');
        await transcription.waitFor();
        const label = { disabled: "Disabled", unconfigured: "Not configured", unknown: "Worker configuration unknown", configured: "Configured" }[state];
        await transcription.getByText(label, { exact: true }).waitFor();
        const retry = transcription.getByRole("button", { name: "Retry failed transcription" });
        assert.equal(await retry.count(), ["unknown", "configured"].includes(state) ? 1 : 0);
        assert.match(await page.locator('[data-health-kind="embed_image"]').innerText(), /2 \/ 15 Complete/);
        assert.match(await page.locator('[data-health-kind="annotate_image"]').innerText(), /2 \/ 10 Complete/);
        assert.match(await transcription.innerText(), /completed without recognized speech/);
        assert.doesNotMatch(await transcription.innerText(), /No audio/);
        assert.doesNotMatch(await transcription.innerText(), /ready|leased/);
      }
      const retry = page.getByRole("button", { name: "Retry failed transcription" });
      await retry.click();
      await page.getByText("Couldn’t retry failed transcription. Try again; no missing jobs will be created.", { exact: true }).waitFor();
      recoveryFail = false;
      await retry.click();
      await page.getByText("1 failed transcription job returned to Waiting. No missing jobs were created.", { exact: true }).waitFor();
      assert.equal(new URL(recoveryURL).searchParams.get("kind"), "transcribe_video");
      statsFail = true;
      await page.getByRole("button", { name: "Refresh health" }).click();
      await page.getByText("Couldn’t refresh library health. Showing the last snapshot.", { exact: true }).waitFor();
      statsFail = false;
      await page.getByRole("button", { name: "Retry health refresh" }).click();
      await page.getByText("Couldn’t refresh library health. Showing the last snapshot.", { exact: true }).waitFor({ state: "hidden" });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, "health fits mobile");
      await mkdir("/tmp/ux08", { recursive: true });
      await page.screenshot({ path: `/tmp/ux08/health-${width}.png`, fullPage: true });
    } finally { await page.close(); }
  }
  console.log("UX08 health and request recovery checks passed (desktop/mobile)");
}
