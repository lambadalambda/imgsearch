<script lang="ts">
  import { onMount } from "svelte";
  import ConfirmDialog from "./ConfirmDialog.svelte";
  import {
    getSettings,
    listAnnotationModels,
    reannotateAll,
    testAnnotationConnection,
    updateSettings,
  } from "../lib/api";
  import { formatCount } from "../lib/utils";
  import type {
    AnnotationBackend,
    AnnotationSettingsInput,
    NativeVariant,
    SettingsResponse,
    SettingsUpdateRequest,
  } from "../lib/types";

  interface FormState {
    backend: AnnotationBackend;
    nativeVariant: NativeVariant;
    baseUrl: string;
    apiKey: string;
    model: string;
    timeoutSeconds: number;
    concurrency: number;
  }

  let loaded = $state<SettingsResponse | null>(null);
  let loadError = $state("");
  let form = $state<FormState>({
    backend: "native",
    nativeVariant: "e4b",
    baseUrl: "",
    apiKey: "",
    model: "",
    timeoutSeconds: 120,
    concurrency: 2,
  });
  let clearApiKey = $state(false);
  let saving = $state(false);
  let saveError = $state("");
  let saveNotice = $state("");
  let testing = $state(false);
  let testResult = $state<{ ok: boolean; error?: string } | null>(null);
  let fetchingModels = $state(false);
  let modelsError = $state("");
  let models = $state<string[]>([]);
  let confirmReannotate = $state(false);
  let reannotating = $state(false);
  let reannotateNotice = $state("");
  let reannotateError = $state("");

  function formFromResponse(response: SettingsResponse): FormState {
    const a = response.annotation;
    return {
      backend: a.backend,
      nativeVariant: a.native_variant,
      baseUrl: a.openai.base_url,
      apiKey: "",
      model: a.openai.model,
      timeoutSeconds: a.openai.timeout_seconds,
      concurrency: a.openai.concurrency,
    };
  }

  function toInput(state: FormState): AnnotationSettingsInput {
    return {
      backend: state.backend,
      native_variant: state.nativeVariant,
      openai: {
        base_url: state.baseUrl,
        api_key: state.apiKey || undefined,
        model: state.model,
        timeout_seconds: Number(state.timeoutSeconds) || 0,
        concurrency: Number(state.concurrency) || 0,
      },
    };
  }

  function toRequest(): SettingsUpdateRequest {
    return { annotation: toInput(form), clear_api_key: clearApiKey || undefined };
  }

  async function load() {
    loadError = "";
    try {
      const response = await getSettings();
      loaded = response;
      form = formFromResponse(response);
      clearApiKey = false;
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    }
  }

  onMount(() => {
    void load();
  });

  /** Any edit invalidates feedback from the previous probe or save. */
  function onEdit() {
    testResult = null;
    saveNotice = "";
    saveError = "";
  }

  const dirty = $derived.by(() => {
    if (!loaded) return false;
    const base = formFromResponse(loaded);
    return (
      clearApiKey ||
      form.apiKey !== "" ||
      form.backend !== base.backend ||
      form.nativeVariant !== base.nativeVariant ||
      form.baseUrl !== base.baseUrl ||
      form.model !== base.model ||
      Number(form.timeoutSeconds) !== base.timeoutSeconds ||
      Number(form.concurrency) !== base.concurrency
    );
  });

  const remoteIncomplete = $derived(form.backend === "openai" && (!form.baseUrl.trim() || !form.model.trim()));

  async function save() {
    if (saving || !dirty || remoteIncomplete) return;
    saving = true;
    saveError = "";
    saveNotice = "";
    try {
      const response = await updateSettings(toRequest());
      loaded = response;
      form = formFromResponse(response);
      clearApiKey = false;
      testResult = null;
      saveNotice = "Saved. The worker picks this up before its next annotation job.";
    } catch (err) {
      saveError = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }

  async function runTest() {
    if (testing) return;
    testing = true;
    testResult = null;
    try {
      testResult = await testAnnotationConnection(toRequest());
    } finally {
      testing = false;
    }
  }

  async function fetchModels() {
    if (fetchingModels) return;
    fetchingModels = true;
    modelsError = "";
    try {
      const result = await listAnnotationModels(toRequest());
      models = result.models;
      if (result.error) {
        modelsError = result.error;
      } else if (models.length === 0) {
        modelsError = "The server returned no models.";
      } else if (!form.model) {
        form.model = models[0];
      }
    } finally {
      fetchingModels = false;
    }
  }

  async function runReannotateAll() {
    confirmReannotate = false;
    if (reannotating) return;
    reannotating = true;
    reannotateError = "";
    reannotateNotice = "";
    try {
      const result = await reannotateAll("all");
      const parts = [
        `${formatCount(result.queued_images)} images`,
        `${formatCount(result.queued_videos)} videos`,
      ];
      reannotateNotice = `Queued ${parts.join(" and ")} for re-annotation.`;
      if (result.skipped_leased > 0) {
        reannotateNotice += ` ${formatCount(result.skipped_leased)} in-progress jobs will finish on the previous backend first.`;
      }
    } catch (err) {
      reannotateError = err instanceof Error ? err.message : String(err);
    } finally {
      reannotating = false;
    }
  }

  const activeLabel = $derived.by(() => {
    if (!loaded) return "";
    if (loaded.annotations_disabled) return "annotations are disabled by the -enable-annotations=false flag; settings are saved but not applied";
    if (loaded.active_error) return `unavailable: ${loaded.active_error}`;
    const active = loaded.active;
    if (!active) return "unknown";
    const where = active.backend === "openai" ? `remote · ${active.model}${active.detail ? ` @ ${active.detail}` : ""}` : `native · ${active.model}`;
    const source = active.source === "worker" ? "in use by the worker" : "selected in settings (worker runs separately)";
    return `${where} — ${source}`;
  });

  const inputClass =
    "w-full min-h-[44px] sm:min-h-0 rounded-[10px] border border-line-2 bg-surface px-3 py-2 text-[13.5px] text-ink placeholder:text-muted-2 focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/25 disabled:opacity-60";
  const labelClass = "block text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2 mb-1.5";
  const primaryBtn =
    "px-[14px] py-[9px] min-h-[44px] sm:min-h-0 bg-accent text-[#fffdf8] border border-accent rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-accent-strong disabled:opacity-50 disabled:cursor-not-allowed";
  const secondaryBtn =
    "px-[14px] py-[9px] min-h-[44px] sm:min-h-0 bg-surface text-ink-2 border border-line-2 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2 disabled:opacity-50 disabled:cursor-not-allowed";
  const dangerBtn =
    "px-[14px] py-[9px] min-h-[44px] sm:min-h-0 bg-surface text-bad border border-bad/60 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-[#f7dddd] disabled:opacity-50 disabled:cursor-not-allowed";
</script>

<section data-settings-pane aria-labelledby="settings-pane-title" class="mx-5 sm:mx-9 mt-4 mb-10 flex flex-col gap-4">
  <header class="px-1 sm:px-0 pt-2">
    <p class="m-0 text-[11px] font-semibold uppercase tracking-[0.08em] text-accent-strong">Settings</p>
    <h2 id="settings-pane-title" class="m-0 mt-1 font-display text-[22px] leading-tight text-ink">Descriptions and analysis</h2>
    <p class="m-0 mt-1 text-[12.5px] text-muted-2">
      Choose where titles, descriptions, and tags are generated. Changes apply between annotation jobs, no restart needed.
    </p>
  </header>

  {#if loadError}
    <p class="m-0 text-[13px] text-bad" role="alert">Couldn't load settings: {loadError}</p>
  {:else if !loaded}
    <p class="m-0 text-[13px] text-muted-2">Loading…</p>
  {:else}
    <p data-settings-active class="m-0 px-1 sm:px-0 text-[12.5px] {loaded.active_error ? 'text-bad' : 'text-muted'}">
      <span class="font-semibold text-ink-2">Active backend:</span> {activeLabel}
    </p>

    <form
      class="grid grid-cols-1 xl:grid-cols-2 gap-4"
      oninput={onEdit}
      onchange={onEdit}
      onsubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5 flex flex-col gap-4">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Annotation backend</p>
        <fieldset class="m-0 p-0 border-0 flex flex-col gap-2">
          <legend class="sr-only">Backend</legend>
          <label class="flex items-start gap-3 min-h-[44px] cursor-pointer">
            <input type="radio" name="backend" value="native" bind:group={form.backend} data-settings-backend="native" class="mt-1 accent-accent" />
            <span>
              <span class="block text-[14px] font-medium text-ink">Native (in-process)</span>
              <span class="block text-[12.5px] text-muted-2">Runs the bundled Gemma model locally. Uses GPU or CPU memory on this machine.</span>
            </span>
          </label>
          <label class="flex items-start gap-3 min-h-[44px] cursor-pointer">
            <input type="radio" name="backend" value="openai" bind:group={form.backend} data-settings-backend="openai" class="mt-1 accent-accent" />
            <span>
              <span class="block text-[14px] font-medium text-ink">Remote server (OpenAI-compatible)</span>
              <span class="block text-[12.5px] text-muted-2">llama-server, Ollama, LM Studio, vLLM, OpenAI, OpenRouter, and similar.</span>
            </span>
          </label>
        </fieldset>

        {#if form.backend === "native"}
          <div>
            <label class={labelClass} for="settings-native-variant">Model variant</label>
            <select
              id="settings-native-variant"
              data-settings-native-variant
              class={inputClass}
              bind:value={form.nativeVariant}
              disabled={loaded.native_variant_locked}
            >
              <option value="e4b">e4b — default, lower memory</option>
              <option value="26b">26b — richer output, high-memory systems</option>
            </select>
            {#if loaded.native_variant_locked}
              <p class="m-0 mt-1.5 text-[12.5px] text-muted-2">Custom model paths were pinned by flags, so the variant cannot be changed here.</p>
            {:else}
              <p class="m-0 mt-1.5 text-[12.5px] text-muted-2">Switching downloads the model on first use and reloads it without a restart.</p>
            {/if}
          </div>
        {/if}
      </article>

      {#if form.backend === "openai"}
        <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5 flex flex-col gap-4">
          <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Remote server</p>
          <div>
            <label class={labelClass} for="settings-base-url">Base URL</label>
            <input
              id="settings-base-url"
              data-settings-base-url
              class={inputClass}
              type="url"
              placeholder="http://127.0.0.1:11434/v1"
              bind:value={form.baseUrl}
              autocomplete="off"
              spellcheck="false"
            />
          </div>
          <div>
            <label class={labelClass} for="settings-api-key">API key</label>
            <input
              id="settings-api-key"
              data-settings-api-key
              class={inputClass}
              type="password"
              placeholder={loaded.annotation.openai.api_key_set && !clearApiKey ? "Key is set — leave blank to keep it" : "Optional for local servers"}
              bind:value={form.apiKey}
              autocomplete="off"
            />
            {#if loaded.annotation.openai.api_key_set}
              <label class="mt-1.5 flex items-center gap-2 min-h-[44px] sm:min-h-0 text-[12.5px] text-muted-2 cursor-pointer">
                <input type="checkbox" bind:checked={clearApiKey} data-settings-clear-key class="accent-accent" />
                Clear the stored key
              </label>
            {/if}
          </div>
          <div>
            <label class={labelClass} for="settings-model">Model</label>
            <div class="flex gap-2">
              <input
                id="settings-model"
                data-settings-model
                class={inputClass}
                type="text"
                list="settings-model-options"
                placeholder="e.g. llava, qwen2.5-vl, gpt-4o-mini"
                bind:value={form.model}
                autocomplete="off"
                spellcheck="false"
              />
              <datalist id="settings-model-options">
                {#each models as id (id)}
                  <option value={id}></option>
                {/each}
              </datalist>
              <button
                type="button"
                class="{secondaryBtn} whitespace-nowrap"
                data-settings-fetch-models
                disabled={fetchingModels || !form.baseUrl.trim()}
                aria-busy={fetchingModels ? "true" : undefined}
                onclick={() => void fetchModels()}
              >
                {fetchingModels ? "Fetching…" : "Fetch models"}
              </button>
            </div>
            {#if modelsError}
              <p class="m-0 mt-1.5 text-[12.5px] text-bad" data-settings-models-error>{modelsError}</p>
            {:else if models.length > 0}
              <p class="m-0 mt-1.5 text-[12.5px] text-muted-2" data-settings-models-count>{models.length} chat-capable models available; clear the field to see all of them, or start typing to narrow the list.</p>
            {/if}
          </div>
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class={labelClass} for="settings-timeout">Timeout (seconds)</label>
              <input id="settings-timeout" data-settings-timeout class={inputClass} type="number" min="1" max="3600" bind:value={form.timeoutSeconds} />
            </div>
            <div>
              <label class={labelClass} for="settings-concurrency">Parallel requests</label>
              <input id="settings-concurrency" data-settings-concurrency class={inputClass} type="number" min="1" max="64" bind:value={form.concurrency} />
            </div>
          </div>
          <div class="flex flex-wrap items-center gap-2">
            <button
              type="button"
              class={secondaryBtn}
              data-settings-test
              disabled={testing || remoteIncomplete}
              aria-busy={testing ? "true" : undefined}
              onclick={() => void runTest()}
            >
              {testing ? "Testing…" : "Test connection"}
            </button>
            {#if testResult}
              <span data-settings-test-result class="text-[12.5px] {testResult.ok ? 'text-good' : 'text-bad'}" role="status">
                {testResult.ok ? "Connected and the model is available." : `Failed: ${testResult.error ?? "unknown error"}`}
              </span>
            {/if}
          </div>
        </article>
      {/if}

      <div class="xl:col-span-2 flex flex-wrap items-center gap-3 px-1 sm:px-0">
        <button type="submit" class={primaryBtn} data-settings-save disabled={saving || !dirty || remoteIncomplete} aria-busy={saving ? "true" : undefined}>
          {saving ? "Saving…" : "Save settings"}
        </button>
        {#if dirty && !saving}
          <span class="text-[12.5px] text-muted-2" data-settings-dirty>
            {remoteIncomplete ? "Base URL and model are required before saving" : "Unsaved changes"}
          </span>
        {/if}
        {#if saveError}
          <span class="text-[12.5px] text-bad" role="alert" data-settings-save-error>{saveError}</span>
        {:else if saveNotice}
          <span class="text-[12.5px] text-good" role="status" data-settings-save-notice>{saveNotice}</span>
        {/if}
      </div>
    </form>

    <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5 flex flex-col gap-3">
      <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Re-annotate the library</p>
      <p class="m-0 text-[13px] text-muted">
        Existing descriptions are kept when you switch backends. Queue every image and video for a fresh annotation with the current backend; cards keep their text until it is replaced. On paid remote APIs this is billed per item.
      </p>
      <div class="flex flex-wrap items-center gap-3">
        <button type="button" class={dangerBtn} data-settings-reannotate-all disabled={reannotating} aria-busy={reannotating ? "true" : undefined} onclick={() => (confirmReannotate = true)}>
          {reannotating ? "Queueing…" : "Re-annotate all"}
        </button>
        {#if reannotateError}
          <span class="text-[12.5px] text-bad" role="alert">{reannotateError}</span>
        {:else if reannotateNotice}
          <span class="text-[12.5px] text-good" role="status" data-settings-reannotate-notice>{reannotateNotice}</span>
        {/if}
      </div>
    </article>
  {/if}
</section>

{#if confirmReannotate}
  <ConfirmDialog
    title="Re-annotate every image and video?"
    detail="All annotation jobs are queued again with the current backend. Existing text stays until each item is replaced."
    confirmLabel="Re-annotate all"
    note="Queued jobs run in the background; you can keep browsing."
    onconfirm={() => void runReannotateAll()}
    oncancel={() => (confirmReannotate = false)}
  />
{/if}
