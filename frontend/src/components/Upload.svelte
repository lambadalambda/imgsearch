<script lang="ts">
  import {
    bumpDataEpoch,
    closeUpload,
    getMode,
    setLibrary,
    uploadOpen,
  } from "../lib/stores";
  import {
    ApiError,
    UPLOAD_ACCEPT,
    UPLOAD_MAX_BYTES,
    UPLOAD_MAX_FILES,
    uploadFiles,
  } from "../lib/api";
  import type { UploadEntry } from "../lib/types";
  import Icon from "./Icon.svelte";

  type RowState = "pending" | "uploading" | "created" | "duplicate" | "failed";

  interface Row {
    file: File;
    state: RowState;
    message?: string;
    entry?: UploadEntry;
  }

  let rows = $state<Row[]>([]);
  let uploading = $state(false);
  let dragOver = $state(false);
  let error = $state<string | null>(null);
  let summary = $state<{ created: number; duplicates: number; failed: number } | null>(
    null,
  );
  let inputEl: HTMLInputElement | undefined = $state();

  function reset(): void {
    rows = [];
    error = null;
    summary = null;
    dragOver = false;
  }

  function close(): void {
    if (uploading) return;
    closeUpload();
    // Defer reset so a brief reopen feels stable.
    rows = [];
    summary = null;
    error = null;
  }

  $effect(() => {
    if ($uploadOpen) {
      document.body.classList.add("modal-open");
    } else {
      document.body.classList.remove("modal-open");
    }
    return () => document.body.classList.remove("modal-open");
  });

  function onKey(event: KeyboardEvent): void {
    if (event.key === "Escape" && $uploadOpen) close();
  }

  function addFiles(files: FileList | File[]): void {
    const incoming = Array.from(files);
    if (incoming.length === 0) return;
    error = null;
    summary = null;
    // Drop already-uploading rows from the merge (shouldn't happen because the
    // dropzone is disabled mid-flight, but be defensive).
    const kept = rows.filter((r) => r.state !== "uploading");
    let merged: Row[] = [
      ...kept,
      ...incoming.map<Row>((file) => ({ file, state: "pending" })),
    ];
    if (merged.length > UPLOAD_MAX_FILES) {
      error = `Up to ${UPLOAD_MAX_FILES} files per upload — kept the first ${UPLOAD_MAX_FILES}.`;
      merged = merged.slice(0, UPLOAD_MAX_FILES);
    }
    rows = merged;
  }

  function removeRow(index: number): void {
    if (uploading) return;
    rows = rows.filter((_, i) => i !== index);
  }

  function pickClick(): void {
    inputEl?.click();
  }

  function onPick(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    if (input.files) addFiles(input.files);
    input.value = "";
  }

  function onDragEnter(event: DragEvent): void {
    if (uploading) return;
    if (event.dataTransfer?.types?.includes("Files")) {
      event.preventDefault();
      dragOver = true;
    }
  }

  function onDragOver(event: DragEvent): void {
    if (uploading) return;
    if (event.dataTransfer?.types?.includes("Files")) {
      event.preventDefault();
      dragOver = true;
    }
  }

  function onDragLeave(): void {
    dragOver = false;
  }

  function onDrop(event: DragEvent): void {
    if (uploading) return;
    event.preventDefault();
    dragOver = false;
    if (event.dataTransfer?.files) addFiles(event.dataTransfer.files);
  }

  function bytesLabel(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  async function submit(): Promise<void> {
    const pending = rows.filter((r) => r.state === "pending");
    if (pending.length === 0 || uploading) return;
    const totalBytes = pending.reduce((sum, r) => sum + r.file.size, 0);
    if (totalBytes > UPLOAD_MAX_BYTES) {
      error = `Total selection ${(totalBytes / (1024 * 1024)).toFixed(
        1,
      )} MiB exceeds the 64 MiB request limit. Try fewer files at once.`;
      return;
    }
    uploading = true;
    error = null;
    summary = null;
    rows = rows.map((r) =>
      r.state === "pending" ? { ...r, state: "uploading" as const, message: undefined } : r,
    );
    try {
      const response = await uploadFiles(pending.map((r) => r.file));
      // Backend returns one entry per uploaded file in submission order. Build a
      // FIFO map keyed on filename so duplicate filenames in the batch still
      // line up correctly.
      const queues = new Map<string, UploadEntry[]>();
      for (const entry of response.uploads) {
        const list = queues.get(entry.filename) ?? [];
        list.push(entry);
        queues.set(entry.filename, list);
      }
      rows = rows.map((row) => {
        if (row.state !== "uploading") return row;
        const list = queues.get(row.file.name);
        const entry = list?.shift();
        if (!entry) {
          return { ...row, state: "failed", message: "no result returned" };
        }
        if (entry.error) {
          return { ...row, state: "failed", message: entry.error, entry };
        }
        if (entry.duplicate) {
          return { ...row, state: "duplicate", entry };
        }
        return { ...row, state: "created", entry };
      });
      summary = {
        created: response.created,
        duplicates: response.duplicates,
        failed: response.failed,
      };
      if (response.created > 0) {
        if (getMode().mode !== "library") setLibrary();
        bumpDataEpoch();
      }
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "upload failed";
      error = message;
      rows = rows.map((row) =>
        row.state === "uploading"
          ? { ...row, state: "failed", message }
          : row,
      );
    } finally {
      uploading = false;
    }
  }

  let pendingCount = $derived(rows.filter((r) => r.state === "pending").length);
</script>

<svelte:window onkeydown={onKey} />

{#if $uploadOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    data-upload-modal
    role="dialog"
    aria-modal="true"
    aria-labelledby="upload-title"
    tabindex="-1"
    class="fixed inset-0 z-[1300] grid place-items-center bg-black/[0.65] p-4 sm:p-8 backdrop-blur-md"
    onclick={(event) => {
      if (event.target === event.currentTarget) close();
    }}
  >
    <div
      class="bg-surface rounded-[20px] shadow-[0_30px_80px_rgba(0,0,0,0.45)] w-[min(640px,calc(100vw-32px))] max-h-[calc(100vh-32px)] flex flex-col overflow-hidden"
    >
      <div class="flex items-start justify-between gap-4 px-6 pt-6 pb-3">
        <div>
          <p class="m-0 text-[12px] text-muted uppercase tracking-[0.08em]">Import</p>
          <h2
            id="upload-title"
            class="font-display m-0 mt-0.5 text-[24px] font-semibold text-ink leading-tight"
          >
            Add media
          </h2>
        </div>
        <button
          type="button"
          data-upload-close
          aria-label="Close"
          disabled={uploading}
          onclick={close}
          class="grid place-items-center w-9 h-9 bg-surface-2 text-ink-2 border-0 rounded-full cursor-pointer transition-colors duration-150 ease-soft hover:bg-bg-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Icon name="close" class="w-4 h-4" />
        </button>
      </div>

      <div class="flex-1 overflow-auto px-6 pb-3 flex flex-col gap-4">
        <button
          type="button"
          data-upload-dropzone
          disabled={uploading}
          onclick={pickClick}
          ondragenter={onDragEnter}
          ondragover={onDragOver}
          ondragleave={onDragLeave}
          ondrop={onDrop}
          class="block w-full rounded-[14px] border-2 border-dashed px-5 py-8 text-left transition-colors duration-150 ease-soft cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed {dragOver
            ? 'border-accent bg-accent-soft'
            : 'border-line-2 bg-bg-2 hover:bg-surface-2'}"
        >
          <p class="m-0 text-ink font-medium">Drop files here or click to choose</p>
          <p class="mt-1 text-[13px] text-muted">
            JPEG, PNG, WEBP, AVIF, MP4, MOV, WEBM, MKV — up to {UPLOAD_MAX_FILES} files / 64 MiB per batch.
          </p>
        </button>
        <input
          bind:this={inputEl}
          data-upload-input
          type="file"
          multiple
          accept={UPLOAD_ACCEPT}
          class="hidden"
          onchange={onPick}
        />

        {#if error}
          <p
            data-upload-error
            class="m-0 px-3 py-2 rounded-[10px] bg-accent-soft text-accent-strong text-[13px]"
          >
            {error}
          </p>
        {/if}

        {#if rows.length > 0}
          <ul class="m-0 p-0 list-none flex flex-col gap-1.5">
            {#each rows as row, idx (`${row.file.name}-${idx}`)}
              <li
                data-upload-row
                data-upload-state={row.state}
                class="flex items-center gap-3 px-3 py-2 rounded-[10px] bg-bg-2 border border-line text-[13px]"
              >
                <span
                  aria-hidden="true"
                  class="grid place-items-center w-7 h-7 rounded-full {row.state === 'created'
                    ? 'tone-moss'
                    : row.state === 'duplicate'
                      ? 'tone-gold'
                      : row.state === 'failed'
                        ? 'bg-accent-soft text-accent-strong'
                        : 'bg-surface-2 text-muted'}"
                >
                  {#if row.state === "created" || row.state === "duplicate"}
                    <Icon name="check" class="w-4 h-4" />
                  {:else if row.state === "failed"}
                    <Icon name="close" class="w-4 h-4" />
                  {:else if row.state === "uploading"}
                    <span class="w-2 h-2 rounded-full bg-current animate-pulse"></span>
                  {:else}
                    <span class="w-1.5 h-1.5 rounded-full bg-current"></span>
                  {/if}
                </span>
                <div class="flex-1 min-w-0">
                  <p class="m-0 truncate text-ink-2 font-medium">{row.file.name}</p>
                  <p class="m-0 text-[12px] text-muted-2 truncate">
                    {bytesLabel(row.file.size)}{#if row.state === "duplicate"} · already in library{:else if row.state === "created"} · queued for indexing{:else if row.state === "failed" && row.message} · {row.message}{:else if row.state === "uploading"} · uploading…{/if}
                  </p>
                </div>
                {#if row.state === "pending"}
                  <button
                    type="button"
                    aria-label="Remove {row.file.name}"
                    onclick={() => removeRow(idx)}
                    class="grid place-items-center w-7 h-7 rounded-full bg-surface text-muted-2 border-0 cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2 hover:text-ink"
                  >
                    <Icon name="close" class="w-3.5 h-3.5" />
                  </button>
                {/if}
              </li>
            {/each}
          </ul>
        {/if}

        {#if summary}
          <p data-upload-summary class="m-0 text-[13px] text-muted-2">
            {summary.created} queued · {summary.duplicates} already existed · {summary.failed} failed
          </p>
        {/if}
      </div>

      <div
        class="flex items-center justify-end gap-2 px-6 py-4 border-t border-line bg-bg-2"
      >
        <button
          type="button"
          onclick={close}
          disabled={uploading}
          class="px-[14px] py-[9px] bg-surface text-ink-2 border border-line-2 rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-surface-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {summary ? "Done" : "Cancel"}
        </button>
        <button
          type="button"
          data-upload-submit
          onclick={submit}
          disabled={uploading || pendingCount === 0}
          class="px-[14px] py-[9px] bg-ink text-[#fffdf8] border border-ink rounded-full text-[13.5px] font-medium leading-none cursor-pointer transition-colors duration-150 ease-soft hover:bg-[#2d2924] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {uploading
            ? "Uploading…"
            : pendingCount > 0
              ? `Upload ${pendingCount}`
              : "Upload"}
        </button>
      </div>
    </div>
  </div>
{/if}
