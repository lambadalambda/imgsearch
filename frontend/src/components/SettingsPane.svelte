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
  import { isSettingsDirty, settingsDraft, settingsRequest } from "../lib/settingsDraft";

  const outputTokenGroups = [
    { label: "Images", first: "image", retry: "image_retry" },
    { label: "Video frames", first: "video_frame", retry: "video_frame_retry" },
    { label: "Video summaries", first: "video", retry: "video_retry" },
  ] as const;

  const loaded = $derived($settingsDraft.loaded);
  const saving = $derived($settingsDraft.saving);
  const loadError = $derived($settingsDraft.loadError);
  const saveError = $derived($settingsDraft.saveError);
  const saveNotice = $derived($settingsDraft.saveNotice);
  let testing = $state(false);
  let testResult = $state<{ ok: boolean; error?: string } | null>(null);
  let fetchingModels = $state(false);
  let modelsError = $state("");
  let models = $state<string[]>([]);
  let confirmReannotate = $state(false);
  let reannotating = $state(false);
  let reannotateNotice = $state("");
  let reannotateError = $state("");

  let mounted = false;
  let editVersion = 0;
  onMount(() => {
    mounted = true;
    void settingsDraft.load(getSettings);
    return () => { mounted = false; };
  });

  function toRequest() {
    return settingsRequest($settingsDraft);
  }

  /** Any edit invalidates feedback from the previous probe or save. */
  function onEdit(event?: Event) {
    // Checked/group bindings settle on change, after input. Notifying the store
    // during input would re-render the checkbox with its previous value.
    if (event?.type === "input" && event.target instanceof HTMLInputElement &&
      (event.target.type === "checkbox" || event.target.type === "radio")) return;
    editVersion += 1;
    testResult = null;
    $settingsDraft.saveNotice = "";
    $settingsDraft.saveError = "";
  }

  function discard() {
    settingsDraft.discard();
    onEdit();
    models = [];
    modelsError = "";
  }

  const dirty = $derived(isSettingsDirty($settingsDraft));
  const remoteIncomplete = $derived($settingsDraft.form.backend === "openai" && (!$settingsDraft.form.baseUrl.trim() || !$settingsDraft.form.model.trim()));

  async function save() {
    if (remoteIncomplete) return;
    testResult = null;
    await settingsDraft.save(updateSettings);
  }

  async function runTest() {
    if (testing || saving) return;
    testing = true;
    testResult = null;
    try {
      testResult = await testAnnotationConnection(toRequest());
    } finally {
      testing = false;
    }
  }

  async function fetchModels() {
    if (fetchingModels || saving) return;
    const version = editVersion;
    fetchingModels = true;
    modelsError = "";
    try {
      const result = await listAnnotationModels(toRequest());
      if (!mounted || saving || version !== editVersion) return;
      models = result.models;
      if (result.error) {
        modelsError = result.error;
      } else if (models.length === 0) {
        modelsError = "The server returned no models.";
      } else if (!$settingsDraft.form.model) {
        $settingsDraft.form.model = models[0];
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
  {/if}
  {#if !loaded}
    <p class="m-0 text-[13px] text-muted-2">{loadError ? "Settings unavailable." : "Loading…"}</p>
  {:else}
    <p data-settings-active class="m-0 px-1 sm:px-0 text-[12.5px] {loaded.active_error ? 'text-bad' : 'text-muted'}">
      <span class="font-semibold text-ink-2">Active worker configuration:</span> {activeLabel}
    </p>

    <form
      class="flex flex-col gap-4"
      oninput={onEdit}
      onchange={onEdit}
      onsubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      <div data-settings-actions class="sticky top-2 z-10 flex flex-wrap items-center gap-3 rounded-card border border-line bg-surface shadow-card p-3" aria-live="polite">
        <span class="text-[12.5px] font-semibold text-ink-2" data-settings-dirty={dirty ? "" : undefined}>
          {dirty ? "Unsaved changes" : "Saved"}
        </span>
        <button type="submit" class={primaryBtn} data-settings-save disabled={saving || !dirty || remoteIncomplete} aria-busy={saving ? "true" : undefined}>
          {saving ? "Saving…" : "Save changes"}
        </button>
        <button type="button" class={secondaryBtn} data-settings-discard disabled={saving || !dirty} onclick={discard}>Discard changes</button>
        {#if dirty}
          <span class="text-[12.5px] text-muted-2">Draft kept while browsing this tab. Not applied to the worker.</span>
        {/if}
        {#if remoteIncomplete}
          <span class="text-[12.5px] text-muted-2">Base URL and model are required before saving.</span>
        {/if}
        {#if saveError}
          <span class="text-[12.5px] text-bad" role="alert" data-settings-save-error>{saveError}</span>
        {:else if saveNotice}
          <span class="text-[12.5px] text-good" role="status" data-settings-save-notice>{saveNotice}</span>
        {/if}
      </div>
      <fieldset disabled={saving} class="m-0 min-w-0 p-0 border-0 grid grid-cols-1 xl:grid-cols-2 items-start gap-4">
        <legend class="sr-only">Annotation settings</legend>
      <article class="rounded-card border border-line bg-surface shadow-card p-4 sm:p-5 flex flex-col gap-4">
        <p class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Annotation backend</p>
        <fieldset class="m-0 p-0 border-0 flex flex-col gap-2">
          <legend class="sr-only">Backend</legend>
          <label class="flex items-start gap-3 min-h-[44px] cursor-pointer">
            <input type="radio" name="backend" value="native" bind:group={$settingsDraft.form.backend} data-settings-backend="native" class="mt-1 accent-accent" />
            <span>
              <span class="block text-[14px] font-medium text-ink">Native (in-process)</span>
              <span class="block text-[12.5px] text-muted-2">Runs the bundled Gemma model locally. Uses GPU or CPU memory on this machine.</span>
            </span>
          </label>
          <label class="flex items-start gap-3 min-h-[44px] cursor-pointer">
            <input type="radio" name="backend" value="openai" bind:group={$settingsDraft.form.backend} data-settings-backend="openai" class="mt-1 accent-accent" />
            <span>
              <span class="block text-[14px] font-medium text-ink">Remote server (OpenAI-compatible)</span>
              <span class="block text-[12.5px] text-muted-2">llama-server, Ollama, LM Studio, vLLM, OpenAI, OpenRouter, and similar.</span>
            </span>
          </label>
        </fieldset>

        {#if $settingsDraft.form.backend === "native"}
          <div>
            <label class={labelClass} for="settings-native-variant">Model variant</label>
            <select
              id="settings-native-variant"
              data-settings-native-variant
              class={inputClass}
              bind:value={$settingsDraft.form.nativeVariant}
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

      {#if $settingsDraft.form.backend === "openai"}
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
              bind:value={$settingsDraft.form.baseUrl}
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
              placeholder={loaded.annotation.openai.api_key_set && !$settingsDraft.clearApiKey ? "Key is set — leave blank to keep it" : "Optional for local servers"}
              bind:value={$settingsDraft.form.apiKey}
              autocomplete="off"
            />
            {#if loaded.annotation.openai.api_key_set}
              <label class="mt-1.5 flex items-center gap-2 min-h-[44px] sm:min-h-0 text-[12.5px] text-muted-2 cursor-pointer">
                <input type="checkbox" bind:checked={$settingsDraft.clearApiKey} data-settings-clear-key class="accent-accent" />
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
                bind:value={$settingsDraft.form.model}
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
                disabled={fetchingModels || !$settingsDraft.form.baseUrl.trim()}
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
              <input id="settings-timeout" data-settings-timeout class={inputClass} type="number" min="1" max="3600" bind:value={$settingsDraft.form.timeoutSeconds} />
            </div>
            <div>
              <label class={labelClass} for="settings-concurrency">Parallel annotations</label>
              <input id="settings-concurrency" data-settings-concurrency class={inputClass} type="number" min="1" max="64" aria-describedby="settings-concurrency-help" bind:value={$settingsDraft.form.concurrency} />
              <p id="settings-concurrency-help" class="m-0 mt-1.5 text-[12.5px] text-muted-2">Remote annotation jobs at once: 1 is serial, 2 allows two jobs to overlap. Saved changes apply without a restart after active requests finish. Native inference stays serial.</p>
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

      <section aria-labelledby="settings-video-title" class="xl:col-span-2 rounded-card border border-line bg-surface shadow-card p-4 sm:p-5 flex flex-col gap-4">
        <h3 id="settings-video-title" class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Video summaries and sampling</h3>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label class={labelClass} for="settings-video-mode">External API video input</label>
            <select id="settings-video-mode" class={inputClass} bind:value={$settingsDraft.form.videoMode} disabled={$settingsDraft.form.backend !== "openai"} aria-describedby="settings-video-help">
              <option value="frame_descriptions">Frame descriptions (current approach)</option>
              <option value="direct_frames">All sampled images in one request</option>
            </select>
            <p id="settings-video-help" class="m-0 mt-1.5 text-[12.5px] text-muted-2">Direct mode sends every stored sampled image in order with its timestamp, skipping individual frame descriptions. Your external model/server must support multiple images; this uses more visual context and may cost more per request. Native annotation always uses frame descriptions.</p>
          </div>
          <div>
            <label class={labelClass} for="settings-video-frame-count">Frames per new video</label>
            <input id="settings-video-frame-count" class={inputClass} type="number" min="1" max="32" step="1" required bind:value={$settingsDraft.form.videoFrameCount} aria-describedby="settings-frame-count-help" />
            <p id="settings-frame-count-help" class="m-0 mt-1.5 text-[12.5px] text-muted-2">1–32 evenly spaced samples; default 5. Applies to new uploads only, for both backends. Existing videos keep their stored samples, even when re-annotated. More frames increase embedding work and annotation cost. Frame embeddings remain enabled in either mode.</p>
          </div>
        </div>
        <p class="m-0 text-[12.5px] text-muted-2">Saving does not reprocess existing videos. The video input mode applies to subsequent annotation jobs after the worker refreshes settings; frame count applies to new uploads immediately. Direct mode uses the Video summaries token limits below, not the per-frame limits. Test connection checks connectivity, not multi-image support.</p>
      </section>

      <section aria-labelledby="settings-output-title" aria-describedby="settings-output-help" class="xl:col-span-2 rounded-card border border-line bg-surface shadow-card p-4 sm:p-5 flex flex-col gap-4">
        <h3 id="settings-output-title" class="m-0 text-[12px] font-semibold uppercase tracking-[0.07em] text-muted-2">Output token limits</h3>
        <p id="settings-output-help" class="m-0 text-[12.5px] text-muted-2">
          Generation/output token limits for both native and remote annotation, not context-window sizes or visual/image tokens.
          Higher budgets may increase runtime and do not guarantee successful generation. Requests cannot exceed the provider/model context window, which also includes input tokens.
          Saved limits apply to subsequent requests after the worker refreshes its settings, without changing in-flight requests or retrying old failed jobs.
          Each limit must be a whole number from 64 to 32768 tokens.
        </p>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {#each outputTokenGroups as group (group.first)}
            <fieldset class="m-0 p-0 border-0 min-w-0">
              <legend class="mb-3 text-[14px] font-medium text-ink">{group.label}</legend>
              <div class="flex flex-col gap-3">
                {#each [{ key: group.first, label: "First attempt" }, { key: group.retry, label: "Compact retry" }] as field (field.key)}
                  <div>
                    <label class={labelClass} for={`settings-output-${field.key}`}>{field.label}</label>
                    <input
                      id={`settings-output-${field.key}`}
                      class={inputClass}
                      type="number"
                      min="64"
                      max="32768"
                      step="1"
                      required
                      aria-describedby="settings-output-help"
                      bind:value={$settingsDraft.form.outputTokens[field.key]}
                    />
                  </div>
                {/each}
              </div>
            </fieldset>
          {/each}
        </div>
      </section>

      </fieldset>
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
