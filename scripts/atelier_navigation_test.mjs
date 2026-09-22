import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";

// Local fixture server only; deliberately isolated from the main smoke's preferences.
export async function navigationChecks(browser, baseURL) {
  const page = await browser.newPage({
    viewport: { width: 1280, height: 900 },
  });
  page.setDefaultTimeout(5000);
  page.on("pageerror", (err) => {
    throw err;
  });
  const library = () => page.locator('[data-rail-item="library"]').click();
  const openTags = async () => {
    const toggle = page.getByRole("button", {
      name: "Browse tags",
      exact: true,
    });
    if (
      (await toggle.isVisible()) &&
      (await toggle.getAttribute("aria-expanded")) === "false"
    )
      await toggle.click();
  };
  try {
    await mkdir("/tmp/ux03-04", { recursive: true });
    await page.goto(baseURL, { waitUntil: "networkidle" });
    await page.locator("[data-pin]").first().waitFor();
    assert.match(
      await page.locator("[data-library-totals]").innerText(),
      /Library totals.*images.*videos/,
    );
    assert.match(
      await page.locator("[data-results-meta]").innerText(),
      /Visible results: 48.*150 results/,
    );
    for (const name of [
      "library",
      "search",
      "collections",
      "settings",
      "stats",
      "duplicates",
      "feed",
    ]) {
      const button = page.locator(`[data-rail-item="${name}"]`);
      assert.notEqual(
        (await button.innerText()).trim(),
        "",
        `desktop ${name} has persistent text`,
      );
      const rect = await button.boundingBox();
      assert(rect.width >= 44 && rect.height >= 44, `desktop ${name} target`);
    }
    assert.notEqual(
      await page.locator('[data-icon="collections"]').innerHTML(),
      await page.locator('[data-icon="library"]').innerHTML(),
    );
    assert.equal(await page.getByText("Tags (soon)").count(), 0);
    await page.locator("[data-library-media]").selectOption("videos");
    await page.locator("#atelier-search").fill("a quiet forest at dawn");
    await page.locator("#atelier-search").press("Enter");
    await page.locator("[data-pin-match]").first().waitFor();
    assert.equal(
      await page.locator("h1").innerText(),
      "a quiet forest at dawn",
    );
    assert.match(
      await page.locator("[data-filter-summary]").innerText(),
      /All media · Relevance/,
    );
    assert.equal(
      await page.locator("[data-library-media]").count(),
      0,
      "unsupported type filtering is not offered in search",
    );
    assert.match(
      await page.locator("[data-filter-options]").innerText(),
      /Type and sort are available in Library only/,
    );
    await page
      .getByText("Match scores are relative, not probabilities.", {
        exact: true,
      })
      .waitFor();
    const badge = page.locator("[data-pin-match]").first();
    assert.match(await badge.innerText(), /^Match \d+\/100$/);
    assert.match(
      await badge.getAttribute("title"),
      /Relative.*not.*probability/,
    );
    await page.screenshot({ path: "/tmp/ux03-04/search-desktop.png" });
    const tags = page.getByRole("navigation", { name: "Browse tags" });
    assert.match(
      await page.locator("[data-browse-tags]").innerText(),
      /Replaces current query/,
    );
    assert.equal(
      await tags.locator("xpath=ancestor::form").count(),
      0,
      "tag shortcuts are separate from search",
    );
    await tags
      .getByRole("button", { name: "Browse tag portrait", exact: true })
      .click();
    await page.waitForURL(/tag=portrait/);
    assert.equal(await page.locator("#atelier-search").inputValue(), "");
    assert.match(
      await page.locator("[data-search-scope]").innerText(),
      /Browse tag/,
    );
    await page
      .locator("[data-pin-match]")
      .first()
      .waitFor({ state: "detached" });
    await page.goBack();
    await page.locator("[data-pin-match]").first().waitFor();
    assert.equal(
      await page.locator("#atelier-search").inputValue(),
      "a quiet forest at dawn",
    );

    // Real keyboard activation, not setInputFiles on a hidden input alone.
    const picker = page.getByRole("button", {
      name: "Search by image",
      exact: true,
    });
    for (const [key, width] of [
      ["Enter", 1280],
      ["Space", 390],
    ]) {
      await page.setViewportSize({ width, height: 844 });
      await picker.focus();
      const chooser = page.waitForEvent("filechooser");
      await page.keyboard.press(key);
      await (
        await chooser
      ).setFiles({
        name: "query.png",
        mimeType: "image/png",
        buffer: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aI9sAAAAASUVORK5CYII=",
          "base64",
        ),
      });
      await page.waitForURL(/view=byimage/);
      await page.locator("[data-query-image]").waitFor();
      await page.locator("[data-query-image-clear]").click();
    }
    // The visible picker supplements, rather than replaces, paste/drop.
    for (const eventType of ["paste", "drop"]) {
      await page.evaluate((eventType) => {
        const transfer = new DataTransfer();
        transfer.items.add(
          new File([new Uint8Array([1, 2, 3])], "shortcut.png", {
            type: "image/png",
          }),
        );
        const event =
          eventType === "paste"
            ? new ClipboardEvent("paste", {
                clipboardData: transfer,
                bubbles: true,
                cancelable: true,
              })
            : new DragEvent("drop", {
                dataTransfer: transfer,
                bubbles: true,
                cancelable: true,
              });
        document
          .querySelector(
            eventType === "paste" ? "#atelier-search" : "[data-search-form]",
          )
          .dispatchEvent(event);
      }, eventType);
      await page.waitForURL(/view=byimage/);
      await page.locator("[data-query-image]").waitFor();
      await page.locator("[data-query-image-clear]").click();
    }
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.locator("[data-library-media]").selectOption("all");
    await page.screenshot({ path: "/tmp/ux03-04/library-desktop.png" });

    await page.setViewportSize({ width: 390, height: 844 });
    await library();
    await page.locator("[data-pin]").first().waitFor();
    await page.evaluate(() => window.scrollTo(0, 0));
    for (const label of ["Library", "Search", "Collections", "More"]) {
      const button = page.getByRole("button", { name: label, exact: true });
      const rect = await button.boundingBox();
      assert(rect.width >= 44 && rect.height >= 44, `mobile ${label} target`);
      assert(
        rect.y >= 0 && rect.y + rect.height <= 844,
        `${label} fits viewport`,
      );
    }
    assert.equal(await page.locator("[data-library-media]").isVisible(), false);
    assert.equal(await page.locator("[data-library-sort]").isVisible(), false);
    assert.equal(await page.locator("[data-nsfw-toggle]").isVisible(), false);
    assert.equal(await page.locator("[data-filter-summary]").isVisible(), true);
    const firstMedia = await page
      .locator("[data-pin-media]")
      .first()
      .boundingBox();
    assert(
      firstMedia.y <= 844 / 3,
      `first media in top third, got ${firstMedia.y}`,
    );
    console.log(
      `UX03 mobile first media y=${firstMedia.y.toFixed(1)} (target ≤281.3)`,
    );
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
    await page.screenshot({ path: "/tmp/ux03-04/library-mobile.png" });
    await page.getByRole("button", { name: "Filters", exact: true }).click();
    await page.locator("[data-library-media]").selectOption("images");
    await page.locator("[data-library-sort]").selectOption("newest");
    await page.locator("[data-nsfw-toggle]").check();
    await page.getByRole("button", { name: "Filters", exact: true }).click();
    assert.match(
      await page.locator("[data-filter-summary]").innerText(),
      /Images only · Recently added · NSFW shown/,
    );
    await openTags();
    await tags
      .getByRole("button", { name: "Browse tag portrait", exact: true })
      .click();
    await page.waitForURL(/tag=portrait/);
    assert.equal(
      await tags.isVisible(),
      false,
      "mobile tag shortcut closes after navigation",
    );

    const disclosure = page.getByRole("button", {
      name: "Browse tags",
      exact: true,
    });
    assert.equal(
      await disclosure.evaluate((el) => el === document.activeElement),
      true,
      "tag navigation returns focus to visible disclosure",
    );
    await openTags();
    await tags
      .getByRole("button", { name: "Browse library", exact: true })
      .focus();
    await page.keyboard.press("Enter");
    assert.equal(await tags.isVisible(), false);
    assert.equal(
      await disclosure.evaluate((el) => el === document.activeElement),
      true,
      "Browse library returns focus to visible disclosure",
    );

    const more = page.getByRole("button", { name: "More", exact: true });
    await more.focus();
    await page.keyboard.press("Enter");
    assert.equal(await more.getAttribute("aria-expanded"), "true");
    await page.screenshot({ path: "/tmp/ux03-04/more-mobile.png" });
    await page.keyboard.press("Escape");
    assert.equal(await more.getAttribute("aria-expanded"), "false");
    assert.equal(
      await more.evaluate((el) => el === document.activeElement),
      true,
    );
    for (const [id, title] of [
      ["settings", "Settings"],
      ["stats", "Statistics"],
      ["duplicates", "Duplicates"],
    ]) {
      await more.click();
      const target = page.locator(`[data-rail-item="${id}"]`);
      const rect = await target.boundingBox();
      assert(rect.width >= 44 && rect.height >= 44);
      await target.click();
      await page
        .getByRole("heading", { name: title, exact: true })
        .first()
        .waitFor();
      assert.equal(await page.locator("[data-search-form]").count(), 0);
      assert.equal(await page.locator("[data-browse-tags]").count(), 0);
      assert.equal(await page.locator("[data-library-totals]").count(), 0);
      assert.equal(await more.getAttribute("aria-expanded"), "false");
    }
    await page.screenshot({ path: "/tmp/ux03-04/utility-mobile.png" });
    await page.locator('[data-rail-item="search"]').click();
    assert.equal(
      await page
        .locator("#atelier-search")
        .evaluate((el) => el === document.activeElement),
      true,
      "Search navigates out of utility then focuses entry",
    );
    await page.locator('[data-rail-item="collections"]').click();
    assert.equal(await page.locator("[data-search-form]").count(), 0);
    await more.click();
    await page.locator('[data-rail-item="feed"]').click();
    await page.locator("[data-feed-overlay]").waitFor();
    await page.keyboard.press("Escape");
    await page.locator("[data-feed-overlay]").waitFor({ state: "detached" });
    await page.locator("header [data-upload-trigger]").click();
    await page.getByRole("dialog").waitFor();
    await page.keyboard.press("Escape");
    for (const width of [320, 640, 768, 1280]) {
      await page.setViewportSize({ width, height: 844 });
      await library();
      assert.equal(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        true,
        `no overflow at ${width}`,
      );
    }
    console.log(
      "UX03/04 navigation, picker keyboard, scope, mobile layout checks passed",
    );
  } finally {
    await page.close();
  }
}
