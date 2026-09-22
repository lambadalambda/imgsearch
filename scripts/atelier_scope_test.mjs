import assert from "node:assert/strict";

const paths = [
  "/api/images",
  "/api/search/text",
  "/api/search/tags",
  "/api/search/similar",
  "/api/search/by-image",
];
const record = (id, unsafe = false) => ({
  image_id: id,
  media_type: "image",
  original_name: `${id}.jpg`,
  storage_path: "images/sample-0",
  preview_path: "images/sample-0",
  thumbnail_path: "images/thumbnails/1000.webp",
  mime_type: "image/jpeg",
  width: 800,
  height: 600,
  title: unsafe ? "Unsafe old scope" : "Safe new scope",
  tags: unsafe ? ["nsfw"] : ["portrait"],
  annotation_state: "done",
  distance: 0.2,
});

async function waitFor(promise) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("scope regression request timed out")),
          5000,
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

// Isolated routes exercise only replacement requests, not annotation live updates.
export async function replacementScopeChecks(browser, baseURL) {
  for (const mode of ["library", "search", "tag", "similar", "byimage"]) {
    const page = await browser.newPage({
      viewport: { width: 1280, height: 900 },
    });
    page.setDefaultTimeout(5000);
    await page.addInitScript(() => {
      localStorage.setItem("imgsearch.includeNSFW", "true");
      localStorage.setItem("imgsearch.libraryMedia", "images");
      // Late responses must be rejected even when transport ignores cancellation.
      const fetch = window.fetch;
      window.fetch = (input, init) =>
        fetch(input, { ...init, signal: undefined });
      window.__imgsearchAnnotationPollMs = 600000;
    });
    const held = new Set();
    let nextHold;
    let paginating = false;
    function holdNext(fail = false) {
      const hold = {
        fail,
        started: Promise.withResolvers(),
        release: Promise.withResolvers(),
        completed: Promise.withResolvers(),
      };
      nextHold = hold;
      held.add(hold);
      return hold;
    }
    await page.route(
      (url) => paths.includes(url.pathname),
      async (route) => {
        const url = new URL(route.request().url());
        const unsafe = url.searchParams.get("include_nsfw") === "1";
        const offset = Number(url.searchParams.get("offset") || 0);
        const records = paginating
          ? offset === 0
            ? Array.from({ length: 48 }, (_, i) => record(9100 + i))
            : [record(9148)]
          : [record(unsafe ? 9001 : 9002, unsafe)];
        const response = {
          [url.pathname === "/api/images" ? "images" : "results"]: records,
          total: paginating ? 49 : records.length,
        };
        const hold = nextHold;
        nextHold = undefined;
        if (hold) {
          hold.started.resolve();
          await hold.release.promise;
        }
        await route.fulfill(
          hold?.fail
            ? { status: 503, json: { error: "Scope replacement failed" } }
            : { json: response },
        );
        hold?.completed.resolve();
      },
    );
    const toggle = page.locator("[data-nsfw-toggle]");
    const unsafe = page.locator('[data-pin-key="image:9001"]');
    const safe = page.locator('[data-pin-key="image:9002"]');
    try {
      const query = {
        library: "",
        search: "?q=old-query",
        tag: "?tag=portrait",
        similar: "?similar=1000",
        byimage: "",
      }[mode];
      await page.goto(`${baseURL}/${query}`, { waitUntil: "networkidle" });
      if (mode === "byimage") {
        await page
          .locator('input[type="file"][accept="image/*"]')
          .setInputFiles({
            name: "query.png",
            mimeType: "image/png",
            buffer: Buffer.from([1, 2, 3]),
          });
      }
      await unsafe.waitFor();
      // Hold a failed safe replacement while the old unsafe lightbox is open.
      await unsafe.locator("[data-pin-media]").click();
      await page.locator("[data-lightbox]").waitFor();
      const failed = holdNext(true);
      // Model a filter change from outside the modal (e.g. navigation/history).
      await toggle.evaluate((el) => el.click());
      await waitFor(failed.started.promise);
      assert.match(
        await page.locator("[data-filter-summary]").innerText(),
        /NSFW hidden/,
      );
      assert.equal(
        await page.locator("[data-pin]").count(),
        0,
        `${mode}: old pins removed while safe replacement waits`,
      );
      assert.equal(
        await page.locator("[data-lightbox]").count(),
        0,
        `${mode}: stale lightbox closes immediately`,
      );
      failed.release.resolve();
      await waitFor(failed.completed.promise);
      await page
        .locator("[data-results-meta]")
        .filter({ hasText: "Scope replacement failed" })
        .waitFor();
      assert.equal(
        await page.locator("[data-pin]").count(),
        0,
        `${mode}: failure never restores old pins`,
      );

      // Two replacements in flight: late unsafe success must not overwrite either
      // a safe success or a safe error, even with AbortSignal deliberately ignored.
      for (const failSafe of [false, true]) {
        const late = holdNext();
        await toggle.check();
        await waitFor(late.started.promise);
        assert.equal(await page.locator("[data-pin]").count(), 0);
        const replacement = holdNext(failSafe);
        await toggle.uncheck();
        await waitFor(replacement.started.promise);
        replacement.release.resolve();
        await waitFor(replacement.completed.promise);
        if (failSafe)
          await page
            .locator("[data-results-meta]")
            .filter({ hasText: "Scope replacement failed" })
            .waitFor();
        else await safe.waitFor();
        late.release.resolve();
        await waitFor(late.completed.promise);
        await page.waitForTimeout(50);
        assert.equal(
          await unsafe.count(),
          0,
          `${mode}: ignored-abort late response rejected`,
        );
        assert.equal(
          await page.locator("[data-pin]").count(),
          failSafe ? 0 : 1,
        );
        assert.equal(await page.locator("[data-lightbox]").count(), 0);
      }

      // Changing query with NSFW unchanged must also invalidate cards, not just
      // filter toggles. A different search response is held until assertions finish.
      await toggle.check();
      await unsafe.waitFor();
      const changedQuery = holdNext();
      if (mode === "library")
        await page.locator("[data-library-sort]").selectOption("newest");
      else if (mode === "tag")
        await page
          .getByRole("navigation", { name: "Browse tags" })
          .getByRole("button", { name: "Browse tag cat", exact: true })
          .click();
      else if (mode === "similar")
        await unsafe.locator('[data-pin-action="similar"]').click();
      else if (mode === "byimage")
        await page
          .locator('input[type="file"][accept="image/*"]')
          .setInputFiles({
            name: "other.png",
            mimeType: "image/png",
            buffer: Buffer.from([4, 5, 6]),
          });
      else {
        await page.locator("#atelier-search").fill("new-query");
        await page.locator("#atelier-search").press("Enter");
      }
      await waitFor(changedQuery.started.promise);
      assert.equal(
        await page.locator("[data-pin]").count(),
        0,
        `${mode}: query/order replacement invalidates old results`,
      );
      changedQuery.release.resolve();
      await waitFor(changedQuery.completed.promise);
      await unsafe.waitFor();
      const count = await page.locator("[data-results-meta]").innerText();
      if (["search", "similar", "byimage"].includes(mode)) {
        assert.match(count, /Top results returned \(not exhaustive\)/);
        assert.doesNotMatch(count, /results? in this scope/);
      } else assert.match(count, /1 result in this scope/);

      if (mode === "library") {
        paginating = true;
        await toggle.uncheck();
        await page.locator('[data-pin-key="image:9100"]').waitFor();
        for (const fail of [true, false]) {
          const append = holdNext(fail);
          await page.locator("[data-load-more]").click();
          await waitFor(append.started.promise);
          assert.equal(
            await page.locator("[data-pin]").count(),
            48,
            "append retains loaded cards",
          );
          await page
            .locator('[data-pin-key="image:9100"] [data-pin-media]')
            .click();
          await page.locator("[data-lightbox]").waitFor();
          append.release.resolve();
          await waitFor(append.completed.promise);
          if (fail)
            await page
              .locator("[data-results-meta]")
              .filter({ hasText: "Scope replacement failed" })
              .waitFor();
          else await page.locator('[data-pin-key="image:9148"]').waitFor();
          assert.equal(
            await page.locator("[data-pin]").count(),
            fail ? 48 : 49,
          );
          assert.equal(
            await page.locator("[data-lightbox]").isVisible(),
            true,
            "append keeps current lightbox",
          );
          await page.keyboard.press("Escape");
        }
      }
      console.log(
        `UX03/04 ${mode} replacement/late-response scope checks passed`,
      );
    } finally {
      for (const hold of held) hold.release.resolve();
      await page.unrouteAll({ behavior: "ignoreErrors" });
      await page.close();
    }
  }
}
