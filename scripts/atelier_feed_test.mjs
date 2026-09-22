import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";

/** Keep synthetic media tests from accidentally testing a PNG decode error. */
export async function holdFeedMedia(page) {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const pending = [];
  const handler = (route) => {
    const response = gate.then(() => route.fulfill({ contentType: "video/mp4", body: "fixture unavailable" }));
    pending.push(response);
    return response;
  };
  await page.route("**/media/videos/**", handler);
  return async () => {
    release();
    await Promise.all(pending);
    await page.unroute("**/media/videos/**", handler);
  };
}

/** UX10: metadata events are controlled fixtures, not a playback diagnosis. */
export async function feedChecks(browser, baseURL, videos) {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  page.setDefaultTimeout(5000);
  page.on("pageerror", (error) => { throw error; });
  const out = process.env.UX_SCREENSHOTS || "/tmp/atelier-ux06-ux10";
  await mkdir(out, { recursive: true });
  let batch = "success";
  let launcher = "success";
  const title = "Cats playing in the warm afternoon light beside the kitchen window";
  // Hold the media request open to make metadata loading deterministic.
  const releaseMedia = await holdFeedMedia(page);
  const poster = await readFile(new URL("../fixtures/images/cat_1.jpg", import.meta.url));
  await page.route("**/media/images/**", (route) => route.fulfill({ contentType: "image/jpeg", body: poster }));
  await page.route("**/api/videos?*", (route) => route.fulfill(launcher === "failed"
    ? { status: 503, json: { error: "Fixture launcher failure" } }
    : { json: { videos: launcher === "empty" ? [] : [{ ...videos[0], title }], total: launcher === "empty" ? 0 : 1 } }));
  await page.route("**/api/search/similar-videos?*", (route) => route.fulfill(batch === "failed"
    ? { status: 503, json: { error: "Fixture batch failure" } }
    : { json: { results: batch === "empty" ? [] : videos.slice(1).map((v) => ({ ...v, media_type: "video", distance: 0.1 })), total: batch === "empty" ? 0 : 5 } }));
  const overlay = page.locator("[data-feed-overlay]");
  const clickLauncher = async () => {
    if (!(await page.locator('[data-rail-item="feed"]').isVisible())) {
      await page.getByRole("button", { name: "More", exact: true }).click();
    }
    await page.locator('[data-rail-item="feed"]').click();
  };
  const launch = async () => {
    await clickLauncher();
    await overlay.waitFor();
    await page.locator("[data-feed-next]:not([disabled]), [data-feed-error], [data-feed-end]").filter({ visible: true }).first().waitFor();
  };
  try {
    await page.goto(`${baseURL}/?view=settings`, { waitUntil: "networkidle" });
    await launch();
    await page.getByText("Loading video…", { exact: true }).waitFor();
    assert(await page.locator("[data-feed-playpause]").isDisabled(), "no active zero-duration play transport");
    assert.equal(await page.locator("[data-feed-time]").count(), 0);
    assert.equal(await page.locator("[data-feed-scrubber]").count(), 0);
    assert.equal(await page.locator("[data-feed-save]").innerText(), "Save");
    assert.equal(await page.locator("[data-feed-save]").getAttribute("aria-label"), "Save to collection");
    const heading = overlay.locator("h2");
    assert.equal(await heading.innerText(), title);
    assert(await heading.evaluate((el) => el.scrollHeight <= el.clientHeight && el.scrollWidth <= el.clientWidth), "mobile title is not truncated");
    await page.screenshot({ path: `${out}/feed-loading-mobile.png` });
    // Keyboard stays in the modal; Save and Close are reachable while loading.
    for (const selector of ["[data-feed-save]", "[data-feed-close]"]) {
      let reached = false;
      for (let i = 0; i < 12; i++) {
        await page.keyboard.press("Tab");
        if (await page.locator(selector).evaluate((el) => el === document.activeElement)) { reached = true; break; }
      }
      assert(reached, `${selector} is keyboard reachable`);
    }
    // Metadata success activates time + seeking even before a timeupdate.
    await page.locator('video[data-feed-current="true"]').evaluate((video) => {
      Object.defineProperty(video, "duration", { value: 10, configurable: true });
      video.dispatchEvent(new Event("loadedmetadata"));
    });
    await page.locator("[data-feed-scrubber]").waitFor();
    assert.equal(await page.locator("[data-feed-playpause]").isDisabled(), false);
    assert.match(await page.locator("[data-feed-time]").innerText(), /0:00 \/ 0:10/);
    await page.screenshot({ path: `${out}/feed-ready-mobile.png` });
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.screenshot({ path: `${out}/feed-ready-desktop.png` });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator('video[data-feed-current="true"]').dispatchEvent("error");
    await page.getByText("Cannot play this video.", { exact: true }).waitFor();
    assert(await page.locator("[data-feed-playpause]").isDisabled());
    assert.equal(await page.locator("[data-feed-time]").count(), 0);
    await page.screenshot({ path: `${out}/feed-cannot-play-mobile.png` });
    await page.locator("[data-feed-next]").click();
    await page.getByText("Loading video…", { exact: true }).waitFor();
    await page.locator("[data-feed-prev]").click();
    await page.getByText("Cannot play this video.", { exact: true }).waitFor();
    await page.locator("[data-feed-close]").click();
    assert.equal(new URL(page.url()).searchParams.get("view"), "settings");

    for (const state of ["failed", "empty"]) {
      batch = state;
      await launch();
      const terminal = page.locator(state === "failed" ? "[data-feed-error]" : "[data-feed-end]");
      await terminal.waitFor();
      assert.equal(await terminal.getByRole("button", { name: "Close feed", exact: true }).count(), 1, "terminal action names actual underlying context");
      assert.equal(await page.locator(state === "failed" ? "[data-feed-end]" : "[data-feed-error]").count(), 0);
      let terminalCloseReachable = false;
      for (let i = 0; i < 10; i++) {
        await page.keyboard.press("Tab");
        if (await terminal.getByRole("button", { name: "Close feed", exact: true }).evaluate((el) => el === document.activeElement)) { terminalCloseReachable = true; break; }
      }
      assert(terminalCloseReachable, "terminal Close feed must be keyboard reachable, not trapped behind the overlay");
      await page.screenshot({ path: `${out}/feed-${state}-mobile.png` });
      if (state === "failed") {
        batch = "success";
        await terminal.getByRole("button", { name: "Retry", exact: true }).click();
        await terminal.waitFor({ state: "hidden" });
        await page.locator("[data-feed-close]").click();
        batch = "failed";
        await launch();
      }
      await terminal.getByRole("button", { name: "Close feed", exact: true }).click();
      await overlay.waitFor({ state: "hidden" });
      assert.equal(new URL(page.url()).searchParams.get("view"), "settings");
      assert(await page.getByRole("button", { name: "More", exact: true }).evaluate((el) => el === document.activeElement), "close restores launch focus");
    }
    // The same close action also preserves Library and search contexts.
    for (const context of ["/", "/?q=cat"]) {
      await page.goto(`${baseURL}${context}`, { waitUntil: "networkidle" });
      const underlyingURL = page.url();
      batch = "empty";
      await launch();
      await page.locator("[data-feed-end]").getByRole("button", { name: "Close feed", exact: true }).click();
      await overlay.waitFor({ state: "hidden" });
      assert.equal(page.url(), underlyingURL);
    }
    await page.goto(`${baseURL}/?view=settings`, { waitUntil: "networkidle" });
    for (const state of ["empty", "failed"]) {
      launcher = state;
      await clickLauncher();
      const notice = page.locator("[data-feed-launch-notice]");
      await notice.waitFor();
      assert.match(await notice.innerText(), state === "empty" ? /No videos available/ : /Could not open feed/);
      assert.equal(await overlay.count(), 0);
      assert.equal(new URL(page.url()).searchParams.get("view"), "settings");
      await page.screenshot({ path: `${out}/feed-launch-${state}-mobile.png` });
      await notice.getByRole("button", { name: "Dismiss" }).click();
    }
    // Optional decoder sanity check: required UX regressions above use synthetic
    // events because some automation environments never initialize valid media.
    if (process.env.UX_REAL_MEDIA === "1") {
      await releaseMedia();
      const clip = await readFile(new URL("../fixtures/videos/Tis better to remain silent and be thought a fool [EdDQOVpIpQA].mp4", import.meta.url));
      await page.route("**/media/videos/**", (route) => route.fulfill({ contentType: "video/mp4", body: clip }));
      launcher = "success";
      batch = "success";
      await launch();
      await page.locator("[data-feed-scrubber]").waitFor();
      await page.waitForFunction(() => {
        const video = document.querySelector('video[data-feed-current="true"]');
        return video && video.currentTime > 0 && !video.paused;
      });
      await page.screenshot({ path: `${out}/feed-local-video-mobile.png` });
      await page.setViewportSize({ width: 1280, height: 900 });
      await page.screenshot({ path: `${out}/feed-local-video-desktop.png` });
      await page.locator("[data-feed-close]").click();
      console.log("UX10 optional local MP4 playback check passed (not production/mobile verification)");
    }
  } finally {
    await releaseMedia();
    await page.close();
  }
  console.log("UX10 Feed checks passed");
}
