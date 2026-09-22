import assert from "node:assert/strict";

// Isolated settings API fixtures keep UX05 checks independent of the main smoke flow.
export async function settingsDraftChecks(browser, baseURL, initialSettings) {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const navigate = async id => {
      const target = page.locator(`[data-rail-item="${id}"]`);
      if (!(await target.isVisible())) await page.getByRole("button", { name: "More", exact: true }).click();
      await target.click();
    };
    let baseline = structuredClone(initialSettings);
    let failSave = false;
    let puts = 0;
    let releaseSave;
    let saveGate;
    await page.addInitScript(() => {
      window.settingsStorageWrites = [];
      const setItem = Storage.prototype.setItem;
      Storage.prototype.setItem = function (key, value) {
        window.settingsStorageWrites.push([key, value]);
        return setItem.call(this, key, value);
      };
    });
    await page.route("**/api/settings", async route => {
      if (route.request().method() === "PUT") {
        const body = route.request().postDataJSON();
        puts += 1;
        if (saveGate) await saveGate;
        if (failSave) return route.fulfill({ status: 503, json: { error: "Settings service unavailable" } });
        baseline = {
          ...baseline, version: baseline.version + 1,
          annotation: {
            ...body.annotation,
            openai: { ...body.annotation.openai, api_key: undefined, api_key_set: body.clear_api_key ? false : Boolean(body.annotation.openai.api_key) || baseline.annotation.openai.api_key_set },
          },
        };
      }
      await route.fulfill({ json: baseline });
    });
    await page.goto(`${baseURL}/?view=settings`, { waitUntil: "networkidle" });
    const variant = page.locator("[data-settings-native-variant]");
    const videoMode = page.locator("#settings-video-mode");
    const frameCount = page.locator("#settings-video-frame-count");
    await frameCount.waitFor({ timeout: 5000 });
    assert.equal(await frameCount.inputValue(), "5");
    assert.equal(await videoMode.isDisabled(), true, "native retains the frame-description path");
    assert.match(await page.locator("#settings-video-help").innerText(), /timestamp/i);
    assert.match(await page.locator("#settings-frame-count-help").innerText(), /new uploads/i);
    for (const invalid of ["0", "33", "1.5", ""]) {
      await frameCount.fill(invalid);
      assert.equal(await frameCount.evaluate(el => el.checkValidity()), false);
      await page.getByRole("button", { name: "Save changes", exact: true }).click();
      assert.equal(puts, 0, "invalid frame count must not submit settings");
    }
    for (const boundary of ["1", "32"]) {
      await frameCount.fill(boundary);
      assert.equal(await frameCount.evaluate(el => el.checkValidity()), true);
    }
    await frameCount.fill("5");
    const save = page.getByRole("button", { name: "Save changes", exact: true });
    const discard = page.getByRole("button", { name: "Discard changes", exact: true });
    const warningArmed = () => page.evaluate(() => {
      const event = new Event("beforeunload", { cancelable: true });
      window.dispatchEvent(event);
      return event.defaultPrevented;
    });
    assert.equal(await warningArmed(), false, "clean baseline has no reload warning");
    await variant.selectOption("26b");
    for (const destination of ["library", "collections"]) {
      await navigate(destination);
      await page.locator("[data-settings-pane]").waitFor({ state: "detached" });
      assert.equal(await warningArmed(), true, "reload warning survives unmount");
      await page.goBack();
      await variant.waitFor();
      assert.equal(await variant.inputValue(), "26b", `${destination}/back retains draft`);
      assert.match(await page.locator("[data-settings-dirty]").innerText(), /Unsaved changes/);
    }
    await navigate("feed");
    await page.locator("[data-feed-overlay]").waitFor();
    await page.keyboard.press("Escape");
    await variant.waitFor();
    assert.equal(await variant.inputValue(), "26b", "Feed round-trip retains draft");
    await discard.click();
    assert.equal(await variant.inputValue(), "e4b");
    assert.equal(await warningArmed(), false, "discard removes reload warning");

    await page.locator('[data-settings-backend="openai"]').check();
    await page.locator("[data-settings-base-url]").fill("https://example.test/v1");
    await page.locator("[data-settings-api-key]").fill("ux05-secret-not-for-storage");
    await page.locator("[data-settings-model]").fill("draft-model");
    await page.locator("[data-settings-timeout]").fill("240");
    await page.locator("[data-settings-concurrency]").fill("4");
    await videoMode.selectOption("direct_frames");
    await frameCount.fill("10");
    await page.locator("#settings-output-image").fill("2048");
    await page.locator("#settings-video-title").scrollIntoViewIfNeeded();
    await page.screenshot({ path: `/tmp/imgsearch-video-settings-${width}.png` });
    const panels = page.locator("form article");
    if (width > 1000) {
      const heights = await panels.evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height));
      assert(heights[0] < heights[1] - 80, "backend panel does not stretch to remote form height");
    }
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "no horizontal overflow");
    const bar = await page.locator("[data-settings-actions]").boundingBox();
    assert(bar && bar.width <= width && bar.x >= 0, "save/discard bar fits viewport");
    const storage = await page.evaluate(() => JSON.stringify([localStorage, sessionStorage, window.settingsStorageWrites]));
    assert(!storage.includes("ux05-secret-not-for-storage"), "API key never written to browser storage");
    await navigate("library");
    const dismissed = page.waitForEvent("dialog");
    await page.evaluate(() => { setTimeout(() => location.reload(), 0); });
    const dialog = await dismissed;
    assert.equal(dialog.type(), "beforeunload", "real reload warns outside Settings");
    await dialog.dismiss();
    await navigate("settings");
    assert.equal(await page.locator("[data-settings-api-key]").inputValue(), "ux05-secret-not-for-storage");
    for (const [selector, value] of [
      ["[data-settings-base-url]", "https://example.test/v1"],
      ["[data-settings-model]", "draft-model"],
      ["[data-settings-timeout]", "240"],
      ["[data-settings-concurrency]", "4"],
      ["#settings-output-image", "2048"],
      ["#settings-video-mode", "direct_frames"],
      ["#settings-video-frame-count", "10"],
    ]) assert.equal(await page.locator(selector).inputValue(), value, `${selector} survives navigation`);

    // Failure and success both complete in the store after the submitting pane unmounts.
    failSave = true;
    saveGate = new Promise(resolve => { releaseSave = resolve; });
    await save.click();
    await navigate("collections");
    await navigate("settings");
    assert(await page.locator("[data-settings-save]").isDisabled(), "remount keeps in-flight save locked");
    releaseSave();
    await page.locator("[data-settings-save-error]").waitFor();
    assert.equal(await page.locator("[data-settings-api-key]").inputValue(), "ux05-secret-not-for-storage");
    assert.equal(await warningArmed(), true, "failed save keeps dirty guard");
    failSave = false;
    saveGate = new Promise(resolve => { releaseSave = resolve; });
    await save.click();
    await navigate("library");
    releaseSave();
    await navigate("settings");
    await page.locator("[data-settings-save-notice]").waitFor();
    assert.match(await page.locator("[data-settings-save-notice]").innerText(), /Saved/);
    assert.equal(await page.locator("[data-settings-api-key]").inputValue(), "");
    assert.equal(await warningArmed(), false, "successful save clears dirty guard");
    assert.equal(baseline.annotation.video_mode, "direct_frames");
    assert.equal(baseline.annotation.video_frame_count, 10);
    await page.locator('[data-settings-backend="native"]').check();
    assert.equal(await videoMode.isDisabled(), true);
    saveGate = null;
    await save.click();
    await page.locator("[data-settings-save-notice]").waitFor();
    await page.reload({ waitUntil: "networkidle" });
    assert.equal(await videoMode.isDisabled(), true);
    assert.equal(await videoMode.inputValue(), "direct_frames", "native save retains remote preference");
    await page.locator('[data-settings-backend="openai"]').check();
    await save.click();
    await page.locator("[data-settings-save-notice]").waitFor();
    assert.equal(await videoMode.inputValue(), "direct_frames", "native switch retains remote preference");
    assert.match(await page.locator("[data-settings-active]").innerText(), /native · e4b/, "worker remains distinct from saved remote config");
    await page.locator("[data-settings-model]").fill("discard-me");
    await discard.click();
    assert.equal(await page.locator("[data-settings-model]").inputValue(), "draft-model", "discard uses saved baseline");
    await page.reload({ waitUntil: "networkidle" });
    assert.equal(await page.locator("[data-settings-api-key]").inputValue(), "");
    await page.locator("[data-settings-clear-key]").check();
    await navigate("collections");
    await navigate("settings");
    assert(await page.locator("[data-settings-clear-key]").isChecked(), "key-clearing intent survives navigation");
    await discard.click();
    assert.equal(await page.locator("[data-settings-clear-key]").isChecked(), false);
    await page.locator("[data-settings-api-key]").fill("another-memory-only-secret");
    page.once("dialog", async reloadDialog => {
      assert.equal(reloadDialog.type(), "beforeunload");
      await reloadDialog.accept();
    });
    await page.reload({ waitUntil: "networkidle" });
    assert.equal(await page.locator("[data-settings-api-key]").inputValue(), "", "accepted reload drops unsaved key");
    assert.equal(await warningArmed(), false, "reloaded baseline is clean");
    await page.close();
  }
  console.log("UX05 desktop/mobile draft retention, save lifecycle, reload warning and credential checks passed");
}
