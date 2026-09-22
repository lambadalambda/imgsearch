import { get, writable } from "svelte/store";
import type { AnnotationBackend, AnnotationOutputTokens, VideoAnnotationMode, NativeVariant, SettingsResponse, SettingsUpdateRequest } from "./types";

export interface SettingsForm {
  videoMode: VideoAnnotationMode;
  videoFrameCount: number | undefined;
  backend: AnnotationBackend;
  nativeVariant: NativeVariant;
  baseUrl: string;
  apiKey: string;
  model: string;
  timeoutSeconds: number | undefined;
  concurrency: number | undefined;
  outputTokens: { [K in keyof AnnotationOutputTokens]: number | undefined };
}

interface SettingsDraft {
  loaded: SettingsResponse | null;
  form: SettingsForm;
  clearApiKey: boolean;
  saving: boolean;
  loadError: string;
  saveError: string;
  saveNotice: string;
}

export function formFromResponse(response: SettingsResponse): SettingsForm {
  const a = response.annotation;
  return {
    backend: a.backend, nativeVariant: a.native_variant,
    videoMode: a.video_mode ?? "frame_descriptions", videoFrameCount: a.video_frame_count ?? 5,
    baseUrl: a.openai.base_url, apiKey: "", model: a.openai.model,
    timeoutSeconds: a.openai.timeout_seconds, concurrency: a.openai.concurrency,
    outputTokens: { ...a.output_tokens },
  };
}

export function settingsRequest({ form, clearApiKey }: Pick<SettingsDraft, "form" | "clearApiKey">): SettingsUpdateRequest {
  return {
    clear_api_key: clearApiKey || undefined,
    annotation: {
      backend: form.backend, native_variant: form.nativeVariant,
      video_mode: form.videoMode, video_frame_count: form.videoFrameCount,
      output_tokens: { ...form.outputTokens },
      openai: {
        base_url: form.baseUrl, api_key: form.apiKey || undefined, model: form.model,
        timeout_seconds: Number(form.timeoutSeconds) || 0, concurrency: Number(form.concurrency) || 0,
      },
    },
  };
}

export function isSettingsDirty(state: SettingsDraft): boolean {
  if (!state.loaded) return false;
  const base = formFromResponse(state.loaded);
  return state.clearApiKey || Object.keys(base).some(key => {
    if (key === "outputTokens") {
      return (Object.keys(base.outputTokens) as (keyof AnnotationOutputTokens)[])
        .some(token => state.form.outputTokens[token] !== base.outputTokens[token]);
    }
    return state.form[key as keyof SettingsForm] !== base[key as keyof SettingsForm];
  });
}

const message = (error: unknown) => error instanceof Error ? error.message : String(error);

/** Memory only: survives pane unmounts, never persists credentials in browser storage.
 * Requests belong to this store, not the pane that happened to start them.
 */
export function createSettingsDraft() {
  const store = writable<SettingsDraft>({
    loaded: null,
    form: {
      backend: "native", nativeVariant: "e4b", baseUrl: "", apiKey: "", model: "",
      timeoutSeconds: 120, concurrency: 2, videoMode: "frame_descriptions", videoFrameCount: 5,
      outputTokens: { image: 1024, image_retry: 512, video_frame: 320, video_frame_retry: 256, video: 1200, video_retry: 640 },
    },
    clearApiKey: false, saving: false, loadError: "", saveError: "", saveNotice: "",
  });
  let loadGeneration = 0;
  const reset = (state: SettingsDraft, loaded: SettingsResponse): SettingsDraft => ({
    ...state, loaded, form: formFromResponse(loaded), clearApiKey: false, saveError: "", saveNotice: "",
  });
  return {
    subscribe: store.subscribe,
    set: store.set,
    update: store.update,
    async load(fetch: () => Promise<SettingsResponse>) {
      if (get(store).saving) return;
      const generation = ++loadGeneration;
      store.update(s => ({ ...s, loadError: "" }));
      try {
        const response = await fetch();
        if (generation !== loadGeneration) return;
        store.update(s => isSettingsDirty(s)
          ? { ...s, loaded: response }
          : { ...reset(s, response), saveNotice: s.saveNotice });
      } catch (error) {
        if (generation === loadGeneration) store.update(s => ({ ...s, loadError: message(error) }));
      }
    },
    discard() {
      store.update(s => s.loaded && !s.saving ? reset(s, s.loaded) : s);
    },
    async save(put: (request: SettingsUpdateRequest) => Promise<SettingsResponse>) {
      const state = get(store);
      if (state.saving || !isSettingsDirty(state)) return;
      ++loadGeneration; // An older GET must not overwrite this save's response.
      const request = settingsRequest(state);
      store.update(s => ({ ...s, saving: true, saveError: "", saveNotice: "" }));
      try {
        const response = await put(request);
        store.update(s => ({
          ...reset(s, response), loadError: "",
          saveNotice: "Saved. Changes apply to subsequent annotation requests after the worker refreshes its settings.",
        }));
      } catch (error) {
        store.update(s => ({ ...s, saveError: message(error) }));
      } finally {
        store.update(s => ({ ...s, saving: false }));
      }
    },
  };
}

export const settingsDraft = createSettingsDraft();

// Owned by the page-lifetime draft, deliberately not removed on pane unmount.
if (typeof window !== "undefined") {
  const warn = (event: BeforeUnloadEvent) => {
    event.preventDefault();
    event.returnValue = "";
  };
  let armed = false;
  settingsDraft.subscribe(state => {
    const dirty = isSettingsDirty(state);
    if (dirty === armed) return;
    armed = dirty;
    if (armed) window.addEventListener("beforeunload", warn);
    else window.removeEventListener("beforeunload", warn);
  });
}
