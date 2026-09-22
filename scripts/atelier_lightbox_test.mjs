import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";

/** UX06: isolated fixture routes; no mutations to the shared smoke server. */
export async function lightboxChecks(browser, baseURL, images) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.setDefaultTimeout(5000);
  page.on("pageerror", (error) => { throw error; });
  const out = process.env.UX_SCREENSHOTS || "/tmp/atelier-ux06-ux10";
  await mkdir(out, { recursive: true });
  const image = await readFile(new URL("../fixtures/images/cat_1.jpg", import.meta.url));
  await page.route("**/media/images/**", (route) => route.fulfill({ contentType: "image/jpeg", body: image }));
  const filename = `${"a9f0".repeat(20)}.jpg`;
  await page.route("**/api/search/text?*", async (route) => {
    const response = await route.fetch();
    const payload = await response.json();
    payload.results = payload.results.map((pin) => pin.image_id === 1000
      ? { ...pin, original_name: filename, full_description: `${pin.full_description}\n`.repeat(12) }
      : pin);
    await route.fulfill({ json: payload });
  });
  let releaseSave;
  const saveGate = new Promise((resolve) => { releaseSave = resolve; });
  let failSave = true;
  await page.route("**/api/images/1000", async (route) => {
    if (route.request().method() !== "PATCH") return route.fallback();
    const patch = route.request().postDataJSON();
    if (failSave) await saveGate;
    await route.fulfill(failSave
      ? { status: 503, json: { error: "Fixture save failed" } }
      : { json: { ...images[0], ...patch } });
  });
  try {
    await page.goto(`${baseURL}/?q=cat`, { waitUntil: "networkidle" });
    const opener = page.locator('[data-pin-key="image:1000"] [data-pin-media]');
    await opener.click();
    const box = page.locator("[data-lightbox]");
    await box.waitFor();
    assert.equal(await box.locator('[data-lightbox-action="delete"]').isVisible(), false, "destructive actions start hidden in More");
    assert.match(await box.innerText(), /Match \d+\/100/);
    assert.doesNotMatch(await box.innerText(), /\d+% match/);
    const details = box.locator("details").filter({ hasText: "File details" });
    assert.equal(await details.getAttribute("open"), null);
    assert.equal(await box.getByText(filename, { exact: true }).isVisible(), false);
    await details.locator("summary").focus();
    await page.keyboard.press("Enter");
    assert(await box.getByText(filename, { exact: true }).isVisible());
    await details.locator("summary").click();
    const more = box.getByRole("button", { name: "More", exact: true });
    await more.focus();
    await page.keyboard.press("Enter");
    for (const action of ["delete", "reannotate", "nsfw"]) assert(await box.locator(`[data-lightbox-action="${action}"]`).isVisible());
    assert(await box.getByRole("link", { name: "Open original" }).isVisible());
    await page.keyboard.press("Escape");
    assert(await more.evaluate((el) => el === document.activeElement));
    assert(await box.isVisible(), "Escape dismisses More before lightbox");
    await box.locator("[data-lightbox-content]").evaluate((el) => { el.scrollTop = 0; });
    await page.screenshot({ path: `${out}/lightbox-desktop.png` });

    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 720 });
      for (const name of ["Save to collection", "Similar", "More", "Edit"]) {
        const action = box.getByRole("button", { name, exact: true });
        const rect = await action.boundingBox();
        assert(rect && rect.x >= 0 && rect.x + rect.width <= width && rect.y + rect.height <= 720, `${name} reachable at ${width}px`);
      }
      await box.locator("[data-lightbox-content]").evaluate((el) => { el.scrollTop = el.scrollHeight; });
      await box.getByRole("button", { name: "Edit", exact: true }).click();
      const input = box.locator("[data-lightbox-title-input]");
      assert(await input.evaluate((el) => el === document.activeElement), "edit focuses title");
      assert.equal(await box.getByRole("button", { name: "Save to collection", exact: true }).count(), 0);
      assert.equal(await box.getByRole("button", { name: "Similar", exact: true }).count(), 0);
      assert.equal(await box.getByRole("button", { name: "More", exact: true }).count(), 0);
      await input.fill("My edited title");
      const save = box.getByRole("button", { name: "Save changes", exact: true });
      const rect = await save.boundingBox();
      assert(rect && rect.y + rect.height <= 720, "Save changes remains in mobile sheet");
      if (width === 390) {
        await save.click();
        const saving = box.getByRole("button", { name: "Saving…", exact: true });
        await saving.waitFor();
        assert(await saving.isDisabled(), "pending save cannot be submitted twice");
        releaseSave();
        await box.getByRole("alert").waitFor();
        assert.equal(await input.inputValue(), "My edited title", "failed save keeps draft");
        await page.screenshot({ path: `${out}/lightbox-edit-failed-mobile.png` });
        failSave = false;
        await save.click();
        await save.waitFor({ state: "hidden" });
        assert.equal(await box.locator("h2").innerText(), "My edited title");
        assert(await box.getByRole("button", { name: "Edit", exact: true }).evaluate((el) => el === document.activeElement));
        await box.getByRole("button", { name: "Edit", exact: true }).click();
      }
      await page.screenshot({ path: `${out}/lightbox-edit-${width}.png` });
      let reachedSave = false;
      for (let i = 0; i < 20; i++) {
        await page.keyboard.press("Tab");
        if (await save.evaluate((el) => el === document.activeElement)) { reachedSave = true; break; }
      }
      assert(reachedSave, "Save changes remains keyboard reachable after the tag editor");
      await page.keyboard.press("Tab");
      assert(await box.getByRole("button", { name: "Cancel", exact: true }).evaluate((el) => el === document.activeElement));
      await page.keyboard.press("Escape");
      assert(await box.isVisible(), "Escape cancels editing, not viewing");
      await page.screenshot({ path: `${out}/lightbox-view-${width}.png` });
    }
    await box.getByRole("button", { name: "Edit", exact: true }).click();
    await page.setViewportSize({ width: 390, height: 430 });
    const shortSave = await box.getByRole("button", { name: "Save changes", exact: true }).boundingBox();
    assert(shortSave && shortSave.y >= 0 && shortSave.y + shortSave.height <= 430, "reduced mobile viewport keeps save reachable");
    await page.screenshot({ path: `${out}/lightbox-edit-short-viewport.png` });
    await box.getByRole("button", { name: "Cancel", exact: true }).click();
    await page.setViewportSize({ width: 320, height: 720 });
    await box.getByRole("button", { name: "More", exact: true }).click();
    await page.screenshot({ path: `${out}/lightbox-more-mobile.png` });
    await page.keyboard.press("Escape");
    await page.keyboard.press("ArrowRight");
    assert.notEqual(await box.getAttribute("data-lightbox-key"), "image:1000", "previous/next preserved");
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("Escape");
    assert(await opener.evaluate((el) => el === document.activeElement), "close restores opener focus");
  } finally {
    releaseSave();
    await page.close();
  }
  console.log("UX06 Lightbox checks passed");
}
