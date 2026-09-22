import { describe, expect, it, vi } from "vitest";
import { get } from "svelte/store";
import { createSettingsDraft, isSettingsDirty, settingsRequest } from "./settingsDraft";
import type { SettingsResponse } from "./types";

const baseline = (): SettingsResponse => ({
  version: 1, native_variant_locked: false, annotations_disabled: false,
  annotation: {
    backend: "native", native_variant: "e4b",
    openai: { base_url: "http://localhost/v1", model: "vision", api_key_set: true, timeout_seconds: 120, concurrency: 2 },
    output_tokens: { image: 1024, image_retry: 512, video_frame: 320, video_frame_retry: 256, video: 1200, video_retry: 640 },
  },
});
const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};

async function setup() {
  const store = createSettingsDraft();
  await store.load(async () => baseline());
  return store;
}

describe("settings draft", () => {
  it("tracks every input including empty numeric values, key replacement and clearing", async () => {
    const store = await setup();
    expect(isSettingsDirty(get(store))).toBe(false);
    for (const patch of [
      { backend: "openai" as const }, { nativeVariant: "26b" as const },
      { baseUrl: "https://remote/v1" }, { model: "other" }, { apiKey: "secret" },
      { timeoutSeconds: undefined }, { concurrency: 3 },
      { outputTokens: { ...get(store).form.outputTokens, image: undefined } },
    ]) {
      store.update(s => ({ ...s, form: { ...s.form, ...patch } }));
      expect(isSettingsDirty(get(store))).toBe(true);
      store.discard();
      expect(isSettingsDirty(get(store))).toBe(false);
    }
    store.update(s => ({ ...s, clearApiKey: true }));
    expect(isSettingsDirty(get(store))).toBe(true);
    expect(settingsRequest(get(store)).clear_api_key).toBe(true);
  });

  it("retains drafts across subscribers and loads; discard uses latest loaded baseline", async () => {
    const store = await setup();
    const unsubscribe = store.subscribe(() => {});
    store.update(s => ({ ...s, form: { ...s.form, nativeVariant: "26b", apiKey: "secret" } }));
    unsubscribe();
    const current = baseline();
    current.annotation.openai.model = "changed elsewhere";
    await store.load(async () => current);
    expect(get(store).form.nativeVariant).toBe("26b");
    expect(settingsRequest(get(store)).annotation.openai.api_key).toBe("secret");
    store.discard();
    expect(get(store).form.model).toBe("changed elsewhere");
    expect(get(store).form.apiKey).toBe("");
    expect(isSettingsDirty(get(store))).toBe(false);
    expect(settingsRequest(get(store)).annotation.openai.api_key).toBeUndefined();
  });

  it("finishes one in-flight save across unmount/remount and ignores stale loads", async () => {
    const store = await setup();
    const lateLoad = deferred<SettingsResponse>();
    const loading = store.load(() => lateLoad.promise);
    store.update(s => ({ ...s, form: { ...s.form, nativeVariant: "26b", apiKey: "secret" } }));
    const pending = deferred<SettingsResponse>();
    const put = vi.fn(() => pending.promise);
    const saving = store.save(put);
    await store.save(put);
    await store.load(async () => baseline());
    expect(put).toHaveBeenCalledTimes(1);
    expect(get(store).saving).toBe(true);
    expect(get(store).form.apiKey).toBe("secret");
    const saved = baseline();
    saved.version = 2;
    saved.annotation.native_variant = "26b";
    pending.resolve(saved);
    await saving;
    lateLoad.resolve(baseline());
    await loading;
    expect(get(store).loaded?.version).toBe(2);
    expect(get(store).form.apiKey).toBe("");
    expect(get(store).saving).toBe(false);
    expect(get(store).saveNotice).toMatch(/^Saved/);
    expect(isSettingsDirty(get(store))).toBe(false);
  });

  it("keeps the complete draft after failed save or refresh", async () => {
    const store = await setup();
    store.update(s => ({ ...s, clearApiKey: true, form: { ...s.form, apiKey: "secret" } }));
    await store.save(async () => { throw new Error("offline"); });
    await store.load(async () => { throw new Error("refresh offline"); });
    expect(get(store).saveError).toBe("offline");
    expect(get(store).loadError).toBe("refresh offline");
    expect(get(store).form.apiKey).toBe("secret");
    expect(get(store).clearApiKey).toBe(true);
    expect(isSettingsDirty(get(store))).toBe(true);
    store.discard();
    expect(get(store).saveError).toBe("");
    expect(isSettingsDirty(get(store))).toBe(false);
  });
});
